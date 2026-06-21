# SPDX-License-Identifier: Apache-2.0
"""Disturbance HARNESS v2 — a deterministic, gentle, per-env shove of the TARGET object fired at a RANDOM
time during the grasp APPROACH, with a perception SENSE-DELAY, so the god-mode solver responds in one of two
collaborative ways depending on WHEN it learns about the move:

  * informed BEFORE the gripper has closed  ->  it ABORTS the current grasp, rises a LITTLE, and SMOOTHLY
    PIVOTS / "chases" the cube to its NEW pose, then closes there.  (chase-a-moving-object data)
  * informed only AFTER the close           ->  too late: the grasp already closed on nothing -> it FAILS,
    rises a LITTLE, re-locates, and re-grasps the new pose.        (failure -> recover -> replan data)

ONE mechanism, BOTH data modes. This is the analogue, for the *grasp*, of the 50/50 distractor pattern: a
per-env PROBABILITY decides whether a trial is disturbed, exactly like ``has_dist = rng.rand(N) < 0.5``.

WHY A HARNESS (not an LLM subagent):
  The disturbance is a privileged, deterministic SIM INJECTION (a small horizontal velocity impulse on the
  target's free-joint at a scheduled control step). The "sensing" (a per-env delay then a privileged read of
  the cube's new pose) and the RECOVERY/CHASE are god-mode CONTROL inside the task (extra batched motion
  phases), not a planning subagent. Keeping it a harness makes the data perfectly reproducible from
  ``(seed, probability)``.

THE INJECTION MECHANISM (chosen + verified):
  A free rigid body (``gs.morphs.Box``) carries a 6-DOF FREE joint whose local dofs are
  ``[0,1,2] = translation (x,y,z)`` and ``[3,4,5] = rotation``. We add a small HORIZONTAL velocity impulse on
  dofs ``[0,1]`` (the in-plane x,y) of the disturbed envs ONLY, via the BATCHED
  ``entity.set_dofs_velocity(v[K,6], envs_idx=disturbed)``. The firm Newton solver then CARRIES that impulse
  for a few steps and friction brings the cube to rest a few cm away -- a *physical* shove, not a teleport
  jump. The magnitude is tuned so the cube reliably shifts ~3-5 cm (enough that the gripper, already committed
  to the OLD pose, misses) but never launches it off the table. Direction is random in the table plane.

  We use a velocity impulse (not ``set_pos``) on purpose:
    * it is PHYSICAL -- the cube accelerates, slides under friction, and decelerates, exactly like a real
      nudge; the contact/penetration gate stays valid (no instantaneous overlap from a teleport);
    * it composes with whatever the cube is already doing (e.g. mid-settle), so it never fights the solver's
      state the way a hard ``set_pos`` would.

SCHEDULING v2 (RANDOM approach-timing + sense-delay):
  The shove lands at a per-env RANDOM control step inside the APPROACH window (the ``pre``/``at`` segments,
  BEFORE the close), so the moment of the move varies trial-to-trial. The task drives the harness one control
  step at a time through :meth:`tick`; the harness fires the (latched) impulse the FIRST time it is ticked
  inside the approach window past that env's random fire-step, then counts a per-env ``sense_delay`` of control
  steps (simulating perception latency) before reporting the env as *informed*. Whether the solver is informed
  BEFORE vs AFTER the gripper closes is what selects CHASE vs RETRY -- and because the fire-step is random and
  the delay is fixed-ish, both branches occur across a batch.

SINGLE-CONTINUOUS-PASS re-architecture (v3, the NO-HOLD fix):
  The old staged seg1/seg2/seg3 path made the NORMAL envs FREEZE LIFTED IN THE AIR while the chase/recover envs
  did their longer pivot (a per-env barrier the owner forbids: every trial must be fully INDEPENDENT, never
  waiting for another env). v3 pre-plans EACH env's FULL trajectory up front and runs them in ONE continuous
  pass, exactly like the clean path -- a clean env terminates early (natural termination -> shorter demo), a
  chased/recovered env's trajectory is simply LONGER, but NO env ever holds waiting.

  To pre-plan the chase/recover re-grasp BEFORE the run, we PREDICT the shoved resting pose at build time. The
  shove is a god-mode impulse WE control (known velocity ``vel_xy`` + the per-env object friction), so the cube
  slides a known distance and stops: ``rest = fire_pose + unit(v) * |v|^2 / (2*mu*g)`` (Coulomb friction,
  ``mu`` = the object's friction, ``g`` = 9.81). Calibrated (scripts/temp/calib_shove_predict.py, GPU1, cube):
  at ``mu = 1.0`` the predicted rest XY matches the REAL settled XY to ~1.1 cm mean / 1.7 cm max -- well within
  the open-claw span, so the re-grasp (planned at the predicted pose) still cages the real cube. The shove still
  fires via the during_step hook (a real physics slide); we only PREDICT where it lands so the whole per-env
  path is known up front. :meth:`predict_shoved_xy` returns the per-env predicted rest XY.

  ``informed_before_close`` (chase vs after-close) is fully determined at sample time -- it depends only on the
  fire-fraction band + the sense-delay, NOT on any sim read -- so v3 resolves it at build time from the plan and
  pre-plans the matching trajectory SHAPE per env (chase: re-aim before the close; after-close: close-on-nothing
  then recover). :meth:`will_be_informed_before_close` computes it from the (known) approach length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Free-joint dof layout for a Genesis free rigid body: [x,y,z, rx,ry,rz]. The in-plane translation dofs.
_XY_DOFS = (0, 1)


@dataclass
class DisturbanceSpec:
    """Per-build disturbance plan: which envs are disturbed, the random in-plane shove for each, the random
    fire-step within the approach window, and the per-env sense-delay. Built once (deterministic from the stage
    rng); the task ticks it every control step of the approach phase.

    Fields
    ------
    prob : float
        Per-env probability a trial is disturbed (mirrors the 50/50 distractor ``has_dist`` pattern).
        Default 0.34. ``DISTURB=0`` forces 0; ``DISTURB=p`` overrides.
    target : entity
        The target object (the cube) whose batched free-joint velocity we impulse.
    speed_range : (lo, hi)
        Magnitude of the horizontal velocity impulse (m/s) sampled per disturbed env. Tuned so the cube slides
        ~3-5 cm on the firm high-friction table and stops -- a GENTLE shove, never a launch.
    sense_delay_range : (lo, hi)
        Per-env perception latency in CONTROL STEPS between the shove landing and the solver being *informed*
        of the new pose (~0.15-0.3 s at the 100 Hz control rate -> ~15-30 steps).
    fire_frac_range : (lo, hi)
        Where in the APPROACH window (fraction 0..1 of the approach control steps) the shove fires per env.
        Drawn wide so some envs are informed before the close (CHASE) and some after (RETRY).
    disturbed : (N,) bool
        Which envs are disturbed this build (``rng.rand(N) < prob``).
    vel_xy : (N, 2)
        The per-env (vx, vy) impulse (zero on undisturbed envs).
    sense_delay : (N,) int
        Per-env sense delay in control steps.
    fire_frac : (N,) float
        Per-env fire fraction within the approach window.
    """

    prob: float
    target: object
    n_envs: int
    speed_range: tuple = (0.70, 0.95)
    sense_delay_range: tuple = (15, 30)      # control steps (~0.15-0.30 s @ 100 Hz)
    fire_frac_range: tuple = (0.25, 0.85)    # fraction of the approach window
    disturbed: np.ndarray = field(default=None)
    vel_xy: np.ndarray = field(default=None)
    sense_delay: np.ndarray = field(default=None)
    fire_frac: np.ndarray = field(default=None)
    # --- live tick state (set by reset_window/tick), not part of the deterministic plan ---
    _approach_T: int = field(default=0, repr=False)        # length (control steps) of the current approach phase
    _fire_step: np.ndarray = field(default=None, repr=False)   # per-env absolute step in the approach to fire at
    _fired: np.ndarray = field(default=None, repr=False)       # per-env: has the shove fired yet
    _fired_step: np.ndarray = field(default=None, repr=False)  # per-env: the step it fired (-1 = not yet)

    @classmethod
    def sample(cls, target, n_envs, rng, prob=0.34, speed_range=(0.70, 0.95),
               sense_delay_range=(15, 30), early_frac_band=(0.20, 0.65), late_frac_band=(0.93, 1.0),
               late_prob=0.5):
        """Draw the per-env disturbance plan with the stage rng (deterministic from seed). ``rng`` is the SAME
        ``RandomState`` the task uses, so the disturbance choice is reproducible alongside the DR.

        The shove still fires at a RANDOM time during the approach, but to reliably generate BOTH data modes
        across a batch each disturbed env is assigned (with prob ``late_prob``) to an EARLY band (fire mid-
        approach -> the solver, after the sense-delay, learns the new pose BEFORE the close -> CHASE) or a LATE
        band (fire just before the close -> the sense-delay pushes "informed" PAST the close -> FAIL + RETRY).
        The exact fire-fraction within the chosen band is still random, so the timing varies trial-to-trial.
        The approach window is long (~3 s) relative to the perception latency (~0.2 s), so without this split a
        uniform fire-time would almost always be sensed before the close; the band split makes the after-close
        case occur at a controlled rate (``late_prob``)."""
        prob = float(prob)
        disturbed = rng.rand(n_envs) < prob
        ang = rng.rand(n_envs) * 2.0 * np.pi                       # random in-plane direction
        spd = speed_range[0] + (speed_range[1] - speed_range[0]) * rng.rand(n_envs)
        vel_xy = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1).astype(np.float32)
        vel_xy[~disturbed] = 0.0
        sd = (sense_delay_range[0]
              + (sense_delay_range[1] - sense_delay_range[0]) * rng.rand(n_envs)).round().astype(np.int32)
        is_late = rng.rand(n_envs) < float(late_prob)              # late band -> after-close (retry); else early
        u = rng.rand(n_envs)
        lo = np.where(is_late, late_frac_band[0], early_frac_band[0])
        hi = np.where(is_late, late_frac_band[1], early_frac_band[1])
        ff = (lo + (hi - lo) * u).astype(np.float32)
        return cls(prob=prob, target=target, n_envs=int(n_envs), speed_range=tuple(speed_range),
                   sense_delay_range=tuple(sense_delay_range),
                   fire_frac_range=(float(min(early_frac_band[0], late_frac_band[0])),
                                    float(max(early_frac_band[1], late_frac_band[1]))),
                   disturbed=disturbed, vel_xy=vel_xy, sense_delay=sd, fire_frac=ff)

    @property
    def any(self) -> bool:
        return bool(self.disturbed.any())

    # ------------------------------------------------------------------ #
    # Live tick API (used by the task's approach phase)
    # ------------------------------------------------------------------ #
    def reset_window(self, approach_T: int):
        """Begin a fresh APPROACH window of ``approach_T`` control steps. Resolves each disturbed env's random
        fire-step from its ``fire_frac`` and clears the per-env fired/latch state. Called once at the start of
        the approach phase (Phase A1)."""
        self._approach_T = int(approach_T)
        self._fire_step = np.clip((self.fire_frac * max(approach_T - 1, 1)).round().astype(np.int32),
                                  0, max(approach_T - 1, 0))
        self._fired = np.zeros(self.n_envs, bool)
        self._fired_step = np.full(self.n_envs, -1, np.int32)

    def tick(self, t: int):
        """Advance the harness one control step of the approach window. Fires the GENTLE horizontal velocity
        impulse, BATCHED, for any disturbed env whose random fire-step has arrived and that hasn't fired yet.
        Latched per-env so each env shoves exactly once. Returns the list of env indices shoved THIS step."""
        if self._fire_step is None:
            return []
        due = self.disturbed & (~self._fired) & (t >= self._fire_step)
        if not due.any():
            return []
        idx = np.where(due)[0]
        v = self.vel_xy[idx]                                       # (K,2) the per-due-env (vx,vy)
        self.target.set_dofs_velocity(v.astype(np.float32), dofs_idx_local=list(_XY_DOFS),
                                      envs_idx=idx.tolist())
        self._fired[idx] = True
        self._fired_step[idx] = t
        return idx.tolist()

    def informed_by(self, t: int) -> np.ndarray:
        """Per-env bool: has the solver been *informed* of the shove by control step ``t`` of the approach
        window? An env is informed once its shove has fired AND ``sense_delay`` control steps have elapsed since
        (perception latency). Undisturbed envs are never informed (False)."""
        if self._fired_step is None:
            return np.zeros(self.n_envs, bool)
        return self._fired & (self._fired_step >= 0) & (t >= self._fired_step + self.sense_delay)

    def fired_any(self) -> np.ndarray:
        """Per-env bool: has this env's shove fired at all (regardless of sense delay)."""
        return self._fired.copy() if self._fired is not None else np.zeros(self.n_envs, bool)

    # ------------------------------------------------------------------ #
    # SINGLE-CONTINUOUS-PASS (v3) build-time helpers
    # ------------------------------------------------------------------ #
    def predict_shoved_xy(self, fire_xy: np.ndarray, mu, g: float = 9.81) -> np.ndarray:
        """PREDICT each disturbed env's shoved RESTING xy at BUILD TIME (Coulomb friction slide):

            rest = fire_xy + unit(vel_xy) * |vel_xy|^2 / (2 * mu * g)

        ``fire_xy`` :(N,2) the object's xy at the moment the shove fires (the settled grasp-centre xy here, since
        the shove fires early in the approach while the object is still at its settled pose). ``mu`` is the
        object's friction (scalar or (N,)); ``g`` = 9.81. Returns (N,2): the predicted rest xy (== ``fire_xy``
        for undisturbed envs). This is what lets the chase/recover re-grasp be PRE-PLANNED for the single
        continuous pass (no sim read). Calibrated to ~1.1 cm mean error on the cube (see module docstring)."""
        fire_xy = np.asarray(fire_xy, float).copy()
        mu = np.broadcast_to(np.asarray(mu, float), (self.n_envs,))
        spd = np.hypot(self.vel_xy[:, 0], self.vel_xy[:, 1])              # |v| per env (0 on undisturbed)
        dist = np.where(spd > 1e-6, spd ** 2 / (2.0 * np.maximum(mu, 1e-6) * g), 0.0)
        ux = np.where(spd > 1e-6, self.vel_xy[:, 0] / np.maximum(spd, 1e-6), 0.0)
        uy = np.where(spd > 1e-6, self.vel_xy[:, 1] / np.maximum(spd, 1e-6), 0.0)
        out = fire_xy.copy()
        out[self.disturbed, 0] = fire_xy[self.disturbed, 0] + (ux * dist)[self.disturbed]
        out[self.disturbed, 1] = fire_xy[self.disturbed, 1] + (uy * dist)[self.disturbed]
        return out

    def will_be_informed_before_close(self, approach_T: int) -> np.ndarray:
        """BUILD-TIME resolution of CHASE vs AFTER-CLOSE per env, with NO sim read. An env is informed before the
        close iff its shove fires AND ``sense_delay`` control steps elapse, all WITHIN the approach window of
        ``approach_T`` control steps (the fire-step comes from ``fire_frac``; the close happens at the end of the
        approach). This is the same condition the live ``informed_by(approach_T-1)`` checked, but computed up
        front so the single-pass planner can choose each disturbed env's trajectory SHAPE (chase re-aim before
        the close, vs close-on-nothing then recover) BEFORE the run. Returns a per-env bool (False if undisturbed)."""
        fire_step = np.clip((self.fire_frac * max(approach_T - 1, 1)).round().astype(np.int32),
                            0, max(approach_T - 1, 0))
        informed_step = fire_step + self.sense_delay
        return self.disturbed & (informed_step <= (approach_T - 1))

    def arm_single_pass(self, fire_step_abs: np.ndarray):
        """Arm the harness for the SINGLE continuous pass: set each disturbed env's ABSOLUTE fire-step (its step
        within the ONE densified run, located inside ITS OWN first approach) and clear the latch. The task then
        ticks every control step; :meth:`tick` fires the impulse at each env's ``fire_step_abs``. Unlike
        ``reset_window`` (which resolved a fire-step from a fraction of a SHARED approach length), the absolute
        step is per-env because each env's pre-planned trajectory has its own length/approach."""
        self._fire_step = np.asarray(fire_step_abs, np.int32).copy()
        self._fired = np.zeros(self.n_envs, bool)
        self._fired_step = np.full(self.n_envs, -1, np.int32)
