# SPDX-License-Identifier: Apache-2.0
"""Disturbance HARNESS — a deterministic, gentle, per-env shove of the TARGET object that makes a
grasp FAIL, so the god-mode solver must DETECT the failure and RECOVER (rise, reopen, re-locate,
re-grasp). This is the analogue, for the *grasp*, of the 50/50 distractor pattern: a per-env
PROBABILITY decides whether a trial is disturbed, exactly like ``has_dist = rng.rand(N) < 0.5``.

WHY A HARNESS (not an LLM subagent):
  The disturbance is a privileged, deterministic SIM INJECTION (a small horizontal velocity impulse on
  the target's free-joint at a scheduled control step). The RECOVERY is a god-mode CONTROL inside the
  task (extra batched motion phases), not a planning subagent. Keeping it a harness makes the
  failure-and-recovery data perfectly reproducible from (seed, probability).

THE INJECTION MECHANISM (chosen + verified):
  A free rigid body (``gs.morphs.Box``) carries a 6-DOF FREE joint whose local dofs are
  ``[0,1,2] = translation (x,y,z)`` and ``[3,4,5] = rotation``. We add a small HORIZONTAL velocity
  impulse on dofs ``[0,1]`` (the in-plane x,y) of the disturbed envs ONLY, via the BATCHED
  ``entity.set_dofs_velocity(v[K,6], envs_idx=disturbed)``. The firm Newton solver then CARRIES that
  impulse for a few steps and friction brings the cube to rest a few cm away — a *physical* shove, not
  a teleport jump. The magnitude is tuned so the cube reliably shifts ~2-5 cm (enough that the gripper,
  already committed to the OLD pose, closes on nothing / bumps the cube) but never launches it off the
  table. Direction is random in the table plane.

  We use a velocity impulse (not ``set_pos``) on purpose:
    * it is PHYSICAL — the cube accelerates, slides under friction, and decelerates, exactly like a real
      nudge; the contact/penetration gate stays valid (no instantaneous overlap from a teleport);
    * it composes with whatever the cube is already doing (e.g. mid-settle), so it never fights the
      solver's state the way a hard ``set_pos`` would.

SCHEDULING (the trigger phase):
  The shove must land DURING the grasp approach/close, AFTER the arm has committed to the (now stale)
  pre-grasp pose, so the planned grasp genuinely misses. The task tells the harness, per the recorded
  cadence, when the active arm is in the ``pre``/``at`` approach window and fires the impulse ONCE there
  for the disturbed envs. After firing it is latched (``_fired``) so it never re-shoves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Free-joint dof layout for a Genesis free rigid body: [x,y,z, rx,ry,rz]. The in-plane translation dofs.
_XY_DOFS = (0, 1)


@dataclass
class DisturbanceSpec:
    """Per-build disturbance plan: which envs are disturbed, the random in-plane shove for each, and the
    one control-step window in which to inject it. Built once (deterministic from the stage rng), then the
    task calls :meth:`maybe_inject` every control step; it fires the batched impulse exactly once.

    Fields
    ------
    prob : float
        Per-env probability a trial is disturbed (mirrors the 50/50 distractor ``has_dist`` pattern).
        Default 0.34 (~1/3 of trials carry a disturbed grasp + recovery). ``DISTURB=0`` forces 0.
    target : entity
        The target object (the cube) whose batched free-joint velocity we impulse.
    speed_range : (lo, hi)
        Magnitude of the horizontal velocity impulse (m/s) sampled per disturbed env. Tuned so the cube
        slides ~2-5 cm on the firm high-friction table and stops — a GENTLE shove, never a launch.
    disturbed : (N,) bool
        Which envs are disturbed this build (``rng.rand(N) < prob``).
    vel_xy : (N, 2)
        The per-env (vx, vy) impulse (zero on undisturbed envs).
    """

    prob: float
    target: object
    n_envs: int
    speed_range: tuple = (0.70, 0.95)
    disturbed: np.ndarray = field(default=None)
    vel_xy: np.ndarray = field(default=None)
    _fired: bool = field(default=False, repr=False)

    @classmethod
    def sample(cls, target, n_envs, rng, prob=0.34, speed_range=(0.70, 0.95)):
        """Draw the per-env disturbance plan with the stage rng (deterministic from seed). ``rng`` is the
        SAME ``RandomState`` the task uses, so the disturbance choice is reproducible alongside the DR."""
        prob = float(prob)
        disturbed = rng.rand(n_envs) < prob
        ang = rng.rand(n_envs) * 2.0 * np.pi                       # random in-plane direction
        spd = speed_range[0] + (speed_range[1] - speed_range[0]) * rng.rand(n_envs)
        vel_xy = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1).astype(np.float32)
        vel_xy[~disturbed] = 0.0
        return cls(prob=prob, target=target, n_envs=int(n_envs), speed_range=tuple(speed_range),
                   disturbed=disturbed, vel_xy=vel_xy)

    @property
    def any(self) -> bool:
        return bool(self.disturbed.any())

    def inject(self):
        """Apply the GENTLE horizontal velocity impulse to the disturbed envs' target, BATCHED, once.
        Sets the cube's free-joint in-plane (x,y) velocity dofs for the disturbed-env rows only; undisturbed
        envs are untouched. Latches ``_fired`` so a repeated call is a no-op (the shove happens exactly once
        in the scheduled approach window). Returns the list of disturbed env indices actually shoved."""
        if self._fired or not self.any:
            self._fired = True
            return []
        env_idx = np.where(self.disturbed)[0]
        v = self.vel_xy[env_idx]                                   # (K,2) the per-disturbed-env (vx,vy)
        # set_dofs_velocity wants the velocity over the SELECTED dofs for the SELECTED envs.
        self.target.set_dofs_velocity(v.astype(np.float32), dofs_idx_local=list(_XY_DOFS),
                                      envs_idx=env_idx.tolist())
        self._fired = True
        return env_idx.tolist()
