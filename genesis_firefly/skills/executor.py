# SPDX-License-Identifier: Apache-2.0
"""Smooth batched dual-arm trajectory executor — the ONE motion path (RoboLab-faithful).

A task supplies, per env, a list of SPARSE end-effector waypoints ``(label, pos, quat_wxyz, grip)``
for that env's ACTIVE arm. This executor:

  1. ``skills.trajectory.densify`` each env's waypoints into a constant-Cartesian-speed, SLERP'd,
     smoothstep-eased EE stream with gripper open/close ramped over its own dwell (the gentle
     RoboLab motion — no pose-to-pose lurching, no gripper snap);
  2. pads every env's stream to a common length T (an env that finishes early HOLDS its last pose);
  3. runs the batch: each control step it IK-solves the active arm for every env (Genesis seeds the
     solve from the current joint config -> C1-continuous), commands the dual arm (the UNUSED arm
     holds ``home_full``), steps physics, and calls back ``on_step`` at the task's record cadence;
  4. holds the final command for ``settle_steps`` so the release settles.

This replaces hand-rolled joint-space linear blends with fixed per-segment step counts (which made
long moves fast/abrupt). SIM-COUPLED only through ``scene.step`` + the robot command helpers; the
geometry (``densify``) is sim-agnostic. Reused unchanged by every task.
"""
from __future__ import annotations

import numpy as np

from skills.trajectory import densify


class BatchExecutor:
    """Densify + batch-execute per-env EE waypoints with the smooth RoboLab motion profile."""

    def __init__(self, scene, robot, side_is_left, *, lin_speed=0.13, ang_speed=1.2, dt=0.01,
                 dwell_steps=80, min_move_steps=12, rec_every=10, ik_every=2):
        self.scene, self.robot = scene, robot
        self.left = np.asarray(side_is_left, bool)
        self.li = np.where(self.left)[0]
        self.ri = np.where(~self.left)[0]
        self.N = int(self.left.shape[0])
        self.rec_every = rec_every
        self.max_dq = np.zeros(self.N)                        # set by run(): worst per-step active-arm joint jump
        # IK is the per-step cost driver; the densified EE stream moves in tiny steps, so we re-solve
        # the arm only every ``ik_every`` control steps (zero-order-hold the joint target between) -- the
        # PD low-passes the staircase. Grip + physics still update EVERY step. ~ik_every x fewer IK calls.
        self.ik_every = max(1, int(ik_every))
        self._dk = dict(lin_speed=lin_speed, ang_speed=ang_speed, dt=dt,
                        dwell_steps=dwell_steps, min_move_steps=min_move_steps)

    def plan(self, waypoints):
        """Densify each env's waypoint list, pad (hold-last) to a common T.
        Returns pos[T,N,3], quat[T,N,4], grip[T,N], labels[T][N], T."""
        dense = [densify(w, **self._dk) for w in waypoints]
        T = max(len(d) for d in dense)
        pos = np.zeros((T, self.N, 3), np.float64)
        quat = np.zeros((T, self.N, 4), np.float64)
        grip = np.zeros((T, self.N), np.float64)
        labels = [["" for _ in range(self.N)] for _ in range(T)]
        for e, d in enumerate(dense):
            ld = len(d)
            for t in range(T):
                p, q, g, l = d[t if t < ld else ld - 1]      # hold last pose after this env finishes
                pos[t, e], quat[t, e], grip[t, e], labels[t][e] = p, q, g, l
        return pos, quat, grip, labels, T

    def run(self, waypoints, solve, home_full, *, on_step=None, settle_steps=40):
        """Execute the smooth batch.

        ``solve(pos[N,3], quat[N,4]) -> active_arm_joints[N,6]`` IK-solves BOTH arms and selects the
        active one per env. ``home_full`` is the [N, n_dofs] command the UNUSED arm holds. ``on_step
        (t, full_cmd[N,n_dofs], labels_t[N])`` is called every ``rec_every`` steps (and once at the
        end) for the task to record state + render. Returns T (number of control steps executed)."""
        from robots.firefly_dual import GR100_MIMIC

        pos, quat, grip, labels, T = self.plan(waypoints)
        ent, ad = self.robot.entity, self.robot.arm
        gdrv, gmim = self.robot.grip_driven, self.robot.grip_mimic
        li, ri = self.li, self.ri

        def apply(aq, g):                                     # build dual-arm cmd + step (grip every step)
            full = home_full.copy()
            for k, d in enumerate(ad["left"]):
                full[li, d] = aq[li, k]
            for k, d in enumerate(ad["right"]):
                full[ri, d] = aq[ri, k]
            g = np.asarray(g, np.float32)
            full[li, gdrv["left"]] = g[li]; full[li, gmim["left"]] = GR100_MIMIC * g[li]
            full[ri, gdrv["right"]] = g[ri]; full[ri, gmim["right"]] = GR100_MIMIC * g[ri]
            ent.control_dofs_position(full); self.scene.step()
            return full

        aq = None
        prev_aq = None
        self.max_dq = np.zeros(self.N)                        # diagnostic: worst per-step active-arm joint jump
        full = home_full
        for t in range(T):
            if aq is None or t % self.ik_every == 0:          # re-solve arm IK every ik_every steps (ZOH between)
                aq = solve(pos[t], quat[t])
                if prev_aq is not None:                       # a branch flip = a >1 rad single-step spike
                    self.max_dq = np.maximum(self.max_dq, np.abs(aq - prev_aq).max(axis=1))
                prev_aq = aq
            full = apply(aq, grip[t])
            if on_step is not None and (t % self.rec_every == 0):
                on_step(t, full, labels[t])
        for _ in range(settle_steps):                         # let the release settle
            ent.control_dofs_position(full); self.scene.step()
        if on_step is not None:
            on_step(T, full, labels[-1])
        return T
