# SPDX-License-Identifier: Apache-2.0
"""Disturbance HARNESS — a per-env shove of the TARGET object at a RANDOM time during the grasp APPROACH, with a
perception SENSE-DELAY, so the god-mode solver reacts in one of two collaborative ways depending on WHEN it
learns about the move. ONE mechanism, BOTH failure-recovery data modes; a per-env PROBABILITY decides whether a
trial is disturbed, exactly like the 50/50 distractor pattern (``has_dist = rng.rand(N) < 0.5``).

  * informed BEFORE the close command  ->  CHASE: don't close yet; rise a little to recover the view, read the
    cube's NEW ground-truth pose, and grasp it there. (chase-a-moving-object data)
  * informed only AFTER the close       ->  the grasp ATTEMPT already fired. Two outcomes:
        - the cube was still caught + lifted despite the shove  ->  it SUCCEEDED: keep going, place it.
        - the grasp missed (closed on nothing)                  ->  reopen, rise to recover the view, read the
          ground-truth pose, and re-grasp.   (failure -> recover -> replan data)

THE INJECTION (privileged, deterministic, reproducible from ``(seed, probability)``):
  The target is a free rigid body (``gs.morphs.Box``) with a 6-DOF FREE joint (local dofs ``[0,1,2]=xyz,
  [3,4,5]=rot``). We add a HORIZONTAL velocity impulse on dofs ``[0,1]`` of the disturbed envs only, BATCHED via
  ``entity.set_dofs_velocity``. The firm solver CARRIES the impulse (a physical slide, not a teleport, so the
  penetration gate stays valid; it composes with the solver state where a ``set_pos`` would fight it). The
  disturbance is UNCONSTRAINED -- the slide may relocate, ROTATE, or TUMBLE the cube onto another face; the
  god-mode recovery handles ANY new 6-DOF pose, so the shove is never softened to keep the cube "nice".

THE RECOVERY IS GOD-MODE + READ-BASED (it does NOT predict the shoved pose):
  When a re-grasp is needed (a chase, or an after-close miss), the task READS the cube's ACTUAL full 6-DOF
  ground-truth pose (position + orientation) and grasps it there. ``recovery_grasp_quat`` builds the grasp from
  the cube's actual world quaternion -- the parallel jaws close across whichever face-pair is graspable from
  above, at the true geometric centre -- so the re-grasp HITS the cube however it relocated / rotated / landed
  (always on a face: gravity forbids an edge rest). NO prediction, NO assumption the cube stayed flat.

HOLD-FREE TWO-PHASE EXECUTION (the read must not freeze the arm; no env ever idles -- owner HARD rule):
  PHASE 1 (one batch, all envs): undisturbed envs run their full pick->place->home; a CHASE approaches then
    aborts before the close and rises (gripper OPEN); an AFTER-CLOSE runs the full grasp ATTEMPT (close + lift).
    The shove fires (real physics) via the per-step :meth:`tick` hook.
  READ (god-mode, between batches): check each after-close grasp outcome (still-grasped vs missed), let any
    missed/relocated cube settle on a face (a few UNRECORDED sim steps -- they add no frame to the demo), and
    read the ground-truth pose. Instantaneous; the arm does not wait.
  PHASE 2 (one batch): place the still-grasped cube as-is; re-grasp the missed/chase cube at its read pose ->
    place -> home; undisturbed envs hold at home (a trimmed tail). The writer assembles each env's demo from ITS
    own phase frames and collapses the inter-phase static pad, so every saved demo is one continuous hold-free
    trajectory (the only static moment is the brief grasp-close settle).

This module owns the HARNESS (the shove sampling + firing) and the DISTURBED-TRAJECTORY ASSEMBLY (phase-1/2
builders + the god-mode full-6-DOF recovery grasp). It imports the grasp PLANNERS from ``skills/grasp.py`` (it
does not duplicate them); the thin task (``tasks/pickplace.py``) builds a ``DisturbExec`` context and calls in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# the grasp ORIENTATION / WRIST-MARGIN planning the disturbed re-grasp re-plans at the (shoved / actual) pose.
# Modularity 3/3: the disturbance trajectory assembly lives HERE and imports from skills/grasp.py (it does NOT
# duplicate the planners). _R_from_wxyz maps a cube quat -> R so we can read the cube's ACTUAL in-plane yaw.
from skills import grasp as _grasp
from skills.grasp import _R_from_wxyz as _R_from_wxyz


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
    force_after_close: np.ndarray = field(default=None)   # per-env: EXPLICITLY an after-close (recover) env
    # --- live fire state (set by arm()/tick()), not part of the deterministic plan ---
    _fire_step: np.ndarray = field(default=None, repr=False)   # per-env absolute step in the run to fire at
    _fired: np.ndarray = field(default=None, repr=False)       # per-env: has the shove fired yet
    _fired_step: np.ndarray = field(default=None, repr=False)  # per-env: the step it fired (-1 = not yet)

    @classmethod
    def sample(cls, target, n_envs, rng, prob=0.34, speed_range=(0.70, 0.95),
               sense_delay_range=(15, 30), early_frac_band=(0.20, 0.65), late_frac_band=(0.30, 0.65),
               late_prob=0.5):
        """Draw the per-env disturbance plan with the stage rng (deterministic from seed). ``rng`` is the SAME
        ``RandomState`` the task uses, so the disturbance choice is reproducible alongside the DR.

        The disturbance is DELIBERATELY UNCONSTRAINED: a per-env in-plane velocity impulse SLIDES the cube, and
        slide/contact may ROTATE or even TUMBLE it onto an edge/another face -- its new pose+orientation can be
        COMPLETELY different from the original. The god-mode RECOVERY handles ANY new 6-DOF pose (it reads the
        ground truth + re-grasps there), so we do NOT soften the shove to keep the cube flat.

        Each disturbed env is assigned (prob ``late_prob``) to AFTER-CLOSE (recover) or CHASE. The CHASE vs
        AFTER-CLOSE distinction is made EXPLICITLY (``force_after_close``), not derived from the sense-delay
        timing -- decoupling it lets the after-close shove fire mid-approach (the cube relocates) while still
        doing the close-on-nothing -> recover. The fire-fraction within the chosen band stays random."""
        prob = float(prob)
        import os as _os
        lo_e = _os.environ.get("SHOVE_SPEED_LO"); hi_e = _os.environ.get("SHOVE_SPEED_HI")
        if lo_e is not None or hi_e is not None:
            speed_range = (float(lo_e) if lo_e is not None else speed_range[0],
                           float(hi_e) if hi_e is not None else speed_range[1])
        disturbed = rng.rand(n_envs) < prob
        ang = rng.rand(n_envs) * 2.0 * np.pi                       # random in-plane direction
        spd = speed_range[0] + (speed_range[1] - speed_range[0]) * rng.rand(n_envs)
        vel_xy = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1).astype(np.float32)
        vel_xy[~disturbed] = 0.0
        sd = (sense_delay_range[0]
              + (sense_delay_range[1] - sense_delay_range[0]) * rng.rand(n_envs)).round().astype(np.int32)
        is_late = (rng.rand(n_envs) < float(late_prob)) & disturbed   # EXPLICITLY after-close (recover) vs chase
        u = rng.rand(n_envs)
        lo = np.where(is_late, late_frac_band[0], early_frac_band[0])
        hi = np.where(is_late, late_frac_band[1], early_frac_band[1])
        ff = (lo + (hi - lo) * u).astype(np.float32)
        return cls(prob=prob, target=target, n_envs=int(n_envs), speed_range=tuple(speed_range),
                   sense_delay_range=tuple(sense_delay_range),
                   fire_frac_range=(float(min(early_frac_band[0], late_frac_band[0])),
                                    float(max(early_frac_band[1], late_frac_band[1]))),
                   disturbed=disturbed, vel_xy=vel_xy, sense_delay=sd, fire_frac=ff,
                   force_after_close=is_late)

    @property
    def any(self) -> bool:
        return bool(self.disturbed.any())

    # ------------------------------------------------------------------ #
    # Live fire API (driven one control step at a time by the task's phase-1 run)
    # ------------------------------------------------------------------ #
    def arm(self, fire_step_abs: np.ndarray):
        """Arm the harness: set each disturbed env's ABSOLUTE fire-step (its step within the phase-1 densified run,
        located inside ITS OWN approach) and clear the per-env latch. The task then drives :meth:`tick` every
        control step; the impulse fires at each env's ``fire_step_abs``. (The fire-step is per-env because each
        env's pre-planned trajectory has its own approach length.)"""
        self._fire_step = np.asarray(fire_step_abs, np.int32).copy()
        self._fired = np.zeros(self.n_envs, bool)
        self._fired_step = np.full(self.n_envs, -1, np.int32)

    def tick(self, t: int):
        """Advance one control step. Fires the horizontal velocity impulse, BATCHED, for any disturbed env whose
        fire-step has arrived and that hasn't fired yet (latched so each env shoves exactly once). The firm solver
        then carries the impulse -- the object slides + may rotate/tumble. Returns the env indices shoved THIS step."""
        if self._fire_step is None:
            return []
        due = self.disturbed & (~self._fired) & (t >= self._fire_step)
        if not due.any():
            return []
        idx = np.where(due)[0]
        self.target.set_dofs_velocity(self.vel_xy[idx].astype(np.float32),
                                      dofs_idx_local=list(_XY_DOFS), envs_idx=idx.tolist())
        self._fired[idx] = True
        self._fired_step[idx] = t
        return idx.tolist()

    def fired_any(self) -> np.ndarray:
        """Per-env bool: has this env's shove fired at all."""
        return self._fired.copy() if self._fired is not None else np.zeros(self.n_envs, bool)


# ================================================================================================ #
# GOD-MODE FULL-6-DOF RECOVERY GRASP  — re-grasp the object at its ACTUAL settled pose+ORIENTATION
# ================================================================================================ #
# The disturbance is UNCONSTRAINED: the shove may SLIDE, ROTATE, or TUMBLE the object onto an edge / a different
# face -- its new pose+orientation can be COMPLETELY different. The recovery is god-mode, so it does NOT need the
# object to stay flat: it READS the object's full 6-DOF ground-truth pose and grasps it THERE. For a box that
# means closing the parallel jaws across whichever opposite FACE-PAIR is currently graspable from above (the pair
# whose world axis is most HORIZONTAL -> perpendicular to a top-down/tilted approach), centred on the object's
# true geometric centre (read god-mode). This works whether the cube is flat, rotated in plane, or tumbled on an
# edge: the jaws always cage the body across a horizontal face-pair at the true centre.


def settled_yaw_from_quat(quat_wxyz) -> np.ndarray:
    """In-plane (about world +Z) yaw of each object from its world quat (N,4 wxyz) -- a DIAGNOSTIC convenience
    (the recovery itself uses the full 6-DOF ``recovery_grasp_quat`` below, not just yaw). yaw = atan2 of the
    object's local +X projected to the table plane. Returns (N,) radians."""
    q = np.asarray(quat_wxyz, float)
    out = np.zeros(q.shape[0])
    for i in range(q.shape[0]):
        rx = _R_from_wxyz(q[i]) @ np.array([1.0, 0.0, 0.0])
        out[i] = float(np.arctan2(rx[1], rx[0]))
    return out


def recovery_grasp_quat(obj_quat_wxyz, reach_dir_xy, tilt_deg, htR_R, laxis_local):
    """GOD-MODE full-6-DOF grasp orientation for ANY object at its ACTUAL settled world quaternion
    ``obj_quat_wxyz`` -- works for the cube, the elongated banana/pen, and the round apple/tennis, at ANY
    orientation (flat / rotated / tumbled / a different face up). This is exactly ``grasp.grasp_quat_at``'s
    orientation logic, but the object's reference axis is rotated by the object's FULL read quaternion (not just
    an in-plane yaw), so it stays correct when the shove rotated the object out of the table plane.

    Build a 'mostly top-down' base approach tilted ``tilt_deg`` toward the reach direction (wrist comfort, same as
    the first grasp). ``laxis_local`` is the object's local grasp reference axis (``spec.long_axis_local()``: a
    face normal for the cube, the long axis for an elongated object, ``None`` for a round object). Express it in
    WORLD via the read quat and align the gripper's closing axis PERPENDICULAR to it (so the jaws close across the
    short axis / the opposite face-pair). For a ROUND object (``None``) the roll is free -> snap it to the branch
    closest to the HOME wrist orientation ``htR_R`` (so the carry->home wrist never flips), exactly as the first
    grasp does. Returns the grasp quat (ee_link wxyz). Reuses skills/grasp.py primitives only."""
    base_q = _grasp.tilted_base_quat(np.asarray(reach_dir_xy, float), float(tilt_deg))
    ref_axis_world = None
    if laxis_local is not None:
        ref_axis_world = _R_from_wxyz(np.asarray(obj_quat_wxyz, float)) @ np.asarray(laxis_local, float)
    gq = _grasp.orientation_aware_grasp_quat(ref_axis_world, base_q, reference_R=htR_R)
    if laxis_local is None:                                      # round object: snap the free roll to the home branch
        gq = _grasp.transport_quats(gq, reference_quat=_grasp._wxyz_from_R(htR_R))[0]
    return gq


# ================================================================================================ #
# DISTURBED-TRAJECTORY ASSEMBLY  (modularity 3/3 — moved out of tasks/pickplace.py's collect())
# ================================================================================================ #
# These were CLOSURES over collect()'s locals; they are now PURE functions over an explicit ``DisturbExec``
# context (the de-closure, mirroring how grasp.py took a GraspContext). The context bundles the immutable
# per-collect knobs + the grasp PLANNERS (imported from skills/grasp.py: ``grasp_quat_at`` / ``select_grasp_
# tilt_at`` / ``select_place_tilt_at`` / ``cquat``) the re-grasp re-plans with, plus the task's own waypoint
# builders (pick / pick_nodwell / place_tail) passed in as callables. The CLEAN (DISTURB=0) path stays in the
# task and is byte-identical -- only the DISTURBED branch calls these.


@dataclass
class DisturbExec:
    """Immutable per-collect bundle the disturbed-trajectory assembly needs (the de-closured former environment
    of the in-line collect() functions). The grasp PLANNERS come from skills/grasp.py via ``gctx`` + the
    ``solve`` callable; this just carries the heights/grips/geometry + the task's own waypoint builders.

    Fields
    ------
    gctx : grasp.GraspContext       the grasp/place planning context (skills/grasp.py)
    solve : callable                IK both arms, pick active per env -> active_arm_joints[N,6]
    N : int                         number of envs
    gc : (N,3)                      the settled grasp centre (object body centre + grasp_dz)
    grasp_dz : float                grasp-height nudge along +Z from the body centre
    APP, LIFT : float               pre-grasp standoff / post-grasp lift heights
    RETRY_RISE : float              the small recover/chase rise (clear the cube, stay dexterous)
    OPEN, CLOSE : float             the gripper open / firm-close commands
    pick_wps : callable(i)          home->pre->at->at->close->lift (double-at; used [:3] for the chase approach)
    pick_wps_nodwell : callable(i)  home->pre->at->close->lift (single-at; the undisturbed pick / the attempt)
    place_tail : callable(i, liftp, gqi)  lift->carry->lower->release->retract->go_home
    """
    gctx: object
    solve: object
    N: int
    gc: np.ndarray
    grasp_dz: float
    APP: float
    LIFT: float
    RETRY_RISE: float
    OPEN: float
    CLOSE: float
    pick_wps: object
    pick_wps_nodwell: object
    place_tail: object


def select_recovery_grasp(dx: DisturbExec, env_i, gci, cube_quat_wxyz, apex_z):
    """GOD-MODE wrist-margin grasp orientation+tilt for the RECOVERY at the cube's ACTUAL FULL 6-DOF pose. Walks
    the same prefer-top-down relax ladder as the first grasp (smallest tilt keeping |j4| off its limit AND the
    elbow bent through pre-grasp + the rise apex + lift), but builds the grasp quat from the cube's FULL world
    quaternion via ``recovery_grasp_quat`` (handles a tumbled / edge-resting / re-faced cube), NOT the yaw-only
    ``grasp_quat_at``. The probe reuses the same batched ``solve`` + thresholds as ``grasp.select_grasp_tilt_at``
    so the SCORING is identical; only the orientation source differs. Returns ``(tilt_deg, grasp_quat)``."""
    gctx, solve, N = dx.gctx, dx.solve, dx.N
    s = "left" if gctx.side_is_left[env_i] else "right"
    reach_dir = np.asarray([gci[0], gci[1]]) - gctx.base[s]
    WL, WM, EM, APP, LIFT = gctx.WRIST_LIMIT, gctx.WRIST_MARGIN, gctx.ELBOW_MIN, gctx.APP, gctx.LIFT
    # Track the MOST wrist-comfortable (smallest |j4|) candidate seen, so that when NO tilt fully clears the
    # threshold (the hard tumbled/edge-resting cube poses) we fall back to the least-saturated orientation --
    # NOT the most top-down one. The old fallback returned ``best_q`` = the FIRST (most top-down) quat together
    # with ``tilt_steps[-1]``, an inconsistent (tilt, quat) pair that re-grasped at the WORST wrist orientation
    # -> the j4=1.57 over-stretch on 4-7/16 recovered envs. Mirrors the clean path's "most relaxed when none clears".
    best_q, best_t, best_j4 = None, float(gctx.tilt_steps[-1]), np.inf
    for t in gctx.tilt_steps:
        gq = recovery_grasp_quat(cube_quat_wxyz, reach_dir, float(t), gctx.htR[s], gctx.laxis).astype(np.float64)
        tz = _R_from_wxyz(gq) @ np.array([0, 0, 1.0])
        gcN = np.tile(np.asarray(gci, float), (N, 1)); gqN = np.tile(gq, (N, 1))
        pre = gcN - APP * tz
        lift = gcN.copy(); lift[:, 2] += LIFT
        qpre = solve(pre, gqN); solve(gcN.copy(), gqN); qlift = solve(lift, gqN)   # warm pre->grasp->lift chain
        j4 = max(abs(float(qpre[env_i, 3])), abs(float(qlift[env_i, 3])))
        j3 = min(float(qpre[env_i, 2]), float(qlift[env_i, 2]))
        if apex_z is not None:                                   # the rise-apex reorient (high-EE) frame too
            apexN = gcN.copy(); apexN[:, 2] = float(apex_z)
            mid = 0.5 * (apexN + pre)
            for wp_ in (apexN, mid, pre):
                qd = solve(wp_, gqN)
                j4 = max(j4, abs(float(qd[env_i, 3]))); j3 = min(j3, float(qd[env_i, 2]))
        if j4 < best_j4:                                         # remember the least-saturated tilt+quat (consistent pair)
            best_j4, best_q, best_t = j4, gq, float(t)
        if (j4 <= (WL - WM)) and (j3 >= EM):
            return float(t), gq
    return best_t, best_q                                        # none cleared -> the SMALLEST-|j4| (most relaxed) tilt


def regrasp_wps(dx: DisturbExec, start_p, start_q, start_grip, gci, grasp_quat, rise_z):
    """Build a SMOOTH, singularity-robust re-grasp at grasp-centre ``gci`` with orientation ``grasp_quat`` (the
    god-mode full-6-DOF recovery quat), FROM the arm's current pose ``start_p``/``start_q``/``start_grip``. Used
    for BOTH the chase and the after-close recovery. Returns ``(waypoints, grasp_quat)``.

    NO-HOLD design: the RISE segment translates UP+OVER to an apex over the grasp xy WHILE slerping the wrist
    start_q->grasp_quat (reorient) and ramping the gripper start_grip->OPEN (reopen) -- one continuously-MOVING
    densified segment, so the reopen+reorient ride motion instead of freezing at a static apex. The descend->pre
    ->at is pure translation; the close is the only dwell (the brief grasp settle). The apex aims at the grasp xy
    (not the start xy) so the start->apex segment always MOVES. ``rise_z`` caps the apex height (clear the cube
    without climbing into the wrist-saturation band)."""
    gqi = np.asarray(grasp_quat, float)
    tz = _R_from_wxyz(gqi) @ np.array([0, 0, 1.0])
    apex = np.asarray(gci, float).copy()
    apex[2] = min(float(rise_z), float(gci[2]) + dx.RETRY_RISE)
    return [
        ("start", np.asarray(start_p, float),  start_q, start_grip),
        ("rise",  apex,                        gqi,     dx.OPEN),   # transit up+over: reorient + reopen folded in
        ("pre",   gci - dx.APP * tz,           gqi,     dx.OPEN),   # pure-translation descend to pre-grasp
        ("at",    gci,                         gqi,     dx.OPEN),
        ("close", gci,                         gqi,     dx.CLOSE),  # the ONLY dwell: the brief grasp-CLOSE settle
        ("lift",  gci + [0, 0, dx.LIFT],       gqi,     dx.CLOSE),
    ], gqi


def resolve_branches(dspec: DisturbanceSpec, ex, pick_wps, N):
    """Split the disturbed envs into CHASE (informed before the close command) vs AFTER-CLOSE (informed only
    after). The split is made EXPLICIT at sample time (``force_after_close``) rather than derived from the
    sense-delay timing, so each mode is generated at a controlled rate. ``approach_T`` (the densified length of
    the open approach) is returned for diagnostics. Returns ``(chased, afterclose, approach_T)``."""
    _, _, _, _lbl, T_App = ex.plan([pick_wps(i)[:4] for i in range(N)])   # home->pre->at->at (the open approach)
    afterclose = dspec.disturbed & np.asarray(dspec.force_after_close, bool)
    chased = dspec.disturbed & ~afterclose
    return chased, afterclose, int(T_App)


def build_phase1(dx: DisturbExec, dspec: DisturbanceSpec, chased, afterclose, gqA):
    """PHASE 1 of the disturbed run (one batch, all envs). The shove fires (real physics) during this batch at a
    random time in the approach; the solver is "informed" after a delay. PHASE 1 does exactly what the solver
    would do KNOWING ONLY the timing of the disturbance vs the close command (owner's simple framing):

    EVERY env's phase 1 is a PICK to LIFT -- uniform length -- so no env idles waiting (the inter-phase pad is
    just the brief grasp settle, NOT a long lifted freeze; that is the ROOT no-hold fix). Phase 2 does ALL the
    placing/recovery. Per env, phase 1:

      * undisturbed env : home->pre->at->close->lift   (the normal SINGLE-at pick).
      * AFTER-CLOSE     : home->pre->at->close->lift   (a real grasp ATTEMPT at the old pose; phase 2 checks the
                          outcome god-mode -- still grasped -> place; missed -> reopen/rise/read/re-grasp).
      * CHASE (informed BEFORE the close) : home->pre->at(old) [DON'T close] -> rise (gripper OPEN) to recover the
                          view. Phase 2 reads the cube's current pose + grasps there.

    Returns the per-env phase-1 streams."""
    N = dx.N
    full_wps = []
    for i in range(N):
        if dspec.disturbed[i] and not afterclose[i]:
            # CHASE: informed before the close -> DON'T close. Approach to at(old), then rise (gripper OPEN) to the
            # comfortable lift height (LIFT) to recover the view -- not LIFT+RETRY_RISE, whose high top-down apex
            # SATURATES the wrist. Phase 2 reads the cube's current pose + grasps there (its own rise re-aims).
            approach = dx.pick_wps(i)[:3]                       # home->pre->at(old), single, all OPEN (no settle)
            ap = approach[-1][1]                                # the at(old) pose, tool frame
            apex = ap.copy(); apex[2] = ap[2] + dx.LIFT
            full_wps.append(approach + [("rise", apex, approach[-1][2], dx.OPEN)])
        else:
            # undisturbed OR after-close: the SINGLE-at pick to lift (a real grasp / a grasp ATTEMPT).
            full_wps.append(dx.pick_wps_nodwell(i))             # home->pre->at->close->lift
    return full_wps


def grasp_succeeded(cube_pos, ee_pos, cube_rest_z, *, lift_min=0.03, near_ee=0.10):
    """Per-env god-mode check: did the (after-close) grasp ATTEMPT actually CATCH the cube despite the
    disturbance? True iff the cube is LIFTED clear of its rest (``cube_z - rest_z > lift_min``) AND still HELD
    near the end-effector (``|cube - ee| < near_ee``). A disturbed-but-still-grasped cube reads exactly this (the
    +8cm 'anomaly' was a SUCCESS -- the close caught it and the lift carried it up). Returns (N,) bool."""
    cube_pos = np.asarray(cube_pos, float); ee_pos = np.asarray(ee_pos, float)
    lifted = (cube_pos[:, 2] - np.asarray(cube_rest_z, float)) > lift_min
    held = np.linalg.norm(cube_pos - ee_pos, axis=1) < near_ee
    return lifted & held


def build_phase2(dx: DisturbExec, chased, afterclose, held, recoverable, lift_start_p, lift_start_q,
                 actual_xy, actual_z, actual_quat, gqA, place_tilt, home_tool, home_tquat):
    """PHASE 2 of the disturbed run (one batch). Per-env, following the owner's simple branching on what phase 1
    saw + a god-mode check of the phase-1 grasp outcome (``held``):

      * undisturbed / AFTER-CLOSE STILL GRASPED (``held``) : the arm is lifted HOLDING the cube (the after-close
                          grasp ATTEMPT caught it despite the shove -- the '+8cm' is a SUCCESS) -> just place->home
                          from the lift (NO drop, NO re-grasp).
      * AFTER-CLOSE + MISSED : closed on nothing -> reopen + rise + READ the cube's ground-truth pose + re-grasp ->
                          place -> home.
      * CHASE : phase 1 aborted before the close (arm risen, OPEN) -> READ the cube's ground-truth pose + grasp ->
                          place -> home.
      * UNRECOVERABLE (``not recoverable[i]``) : the shove knocked the object OFF the table / out of reach -> don't
                          chase a phantom target (it would NaN) -> open the gripper + go home (the place gate marks
                          it failed -- a legit hard-edge outcome).

    Every re-grasp (the after-close MISS + every chase) is built from the cube's ACTUAL FULL 6-DOF pose: position
    (``actual_xy``/``actual_z``) + the full world quaternion (``actual_quat``) via ``select_recovery_grasp`` /
    ``recovery_grasp_quat`` -- the jaws close across whichever face-pair is graspable from above, at the true
    centre, so it HITS the cube however it relocated/rotated/landed (on a face). ``gqA``/``place_tilt`` are
    MUTATED for the re-grasping envs. Returns ``(phase2_wps, recovery_attempts, diag)``."""
    N, gctx, solve = dx.N, dx.gctx, dx.solve
    actual_quat = np.asarray(actual_quat, float)
    phase2 = []
    recovery_attempts = np.zeros(N, np.int32)
    diag = []
    for i in range(N):
        regrasp = bool(chased[i] or (afterclose[i] and not held[i]))   # needs a read-based re-grasp
        if regrasp and not recoverable[i]:
            # the shove knocked the object OFF the table / out of reach -> UNRECOVERABLE. Don't chase a phantom
            # target (it would NaN): just open the gripper and go home. The place gate marks the demo failed.
            sg = dx.CLOSE if afterclose[i] else dx.OPEN
            phase2.append([("start", lift_start_p[i], lift_start_q[i], sg),
                           ("go_home", home_tool[i], home_tquat[i], dx.OPEN)])
            diag.append((i, "unrecoverable", (float(actual_xy[i, 0]), float(actual_xy[i, 1])), 0.0, (0.0, 0.0)))
            continue
        if not regrasp:
            # undisturbed OR after-close-still-grasped: the arm is lifted holding the cube -> place from the lift.
            liftp = (dx.gc[i] + [0, 0, dx.LIFT]).astype(float)    # the phase-1 lift pose (tool frame), gripper CLOSED
            phase2.append(dx.place_tail(i, liftp, gqA[i]))
            if afterclose[i]:
                diag.append((i, "grasped-thru", (float(actual_xy[i, 0]), float(actual_xy[i, 1])), 0.0, (0.0, place_tilt[i])))
            continue
        # MISS (after-close) or CHASE: re-grasp at the cube's ACTUAL full 6-DOF ground-truth pose.
        recovery_attempts[i] = 1
        ax, ay, az = float(actual_xy[i, 0]), float(actual_xy[i, 1]), float(actual_z[i])
        gci = np.array([ax, ay, az + dx.grasp_dz])
        rise_z = max(float(lift_start_p[i][2]), gci[2]) + dx.RETRY_RISE
        apex_z = min(rise_z, float(gci[2]) + dx.RETRY_RISE)
        rt, gq = select_recovery_grasp(dx, i, gci, actual_quat[i], apex_z)   # full-6-DOF grasp at the read pose
        # start_grip: an after-close MISS arrives with the gripper CLOSED-on-nothing (the rise reopens it WHILE
        # moving); a CHASE arrives already OPEN (it never closed). Either way the rise->descend->close is a clean
        # fresh grasp at the read pose.
        sg = dx.CLOSE if afterclose[i] else dx.OPEN
        wps, gqi = regrasp_wps(dx, lift_start_p[i], lift_start_q[i], sg, gci, gq, rise_z)
        gqA[i] = gqi
        place_tilt[i] = _grasp.select_place_tilt_at(gctx, solve, i, gqi)
        phase2.append(wps + dx.place_tail(i, wps[-1][1], gqi))
        diag.append((i, "chase" if chased[i] else "recover", (ax, ay), float(rt), (rt, place_tilt[i])))
    return phase2, recovery_attempts, diag


def fire_steps_from_plan(dspec: DisturbanceSpec, ex, full_wps, N):
    """Per-env ABSOLUTE fire-step within EACH env's OWN first open-approach window of the densified ``full_wps``
    plan (the last pre/at step BEFORE the first close). Firing at ``fire_frac`` of that window lands the shove
    early enough that a chase env can re-aim and an after-close env's cube has slid away by its (old-pose) close.
    Returns ``(fire_step_abs:(N,), T_full)``."""
    _, _, _, lblFull, T_full = ex.plan(full_wps)
    fire_step_abs = np.zeros(N, np.int32)
    for i in range(N):
        if not dspec.disturbed[i]:
            continue
        pre_close = [t for t in range(T_full) if lblFull[t][i] in ("pre", "at")]
        close_ts = [t for t in range(T_full) if lblFull[t][i] == "close"]
        if close_ts:
            pre_close = [t for t in pre_close if t < close_ts[0]]
        if pre_close:
            j = int(round(float(dspec.fire_frac[i]) * (len(pre_close) - 1)))
            fire_step_abs[i] = pre_close[max(0, min(j, len(pre_close) - 1))]
    return fire_step_abs, int(T_full)
