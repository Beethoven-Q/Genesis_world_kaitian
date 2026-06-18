# SPDX-License-Identifier: Apache-2.0
"""Firefly Y6 dual-arm + GR100 gripper — Genesis loader (Line B reproduction of RoboLab's firefly_dual.py).

Genesis counterpart of `robolab/robots/firefly_dual.py`. Loads the SAME self-contained dual URDF, then maps
joint NAMES -> Genesis dof indices (mandatory: Genesis INTERLEAVES the dual-arm dofs, e.g. left_joint_1=dof0,
right_joint_1=dof1, ... so the layout is NOT contiguous), exposes the 14-D policy layout, the MIT-controller
PD gains, the GR100 open/close/mimic constants (gripper coupled at the ACTION layer, not URDF mimic), and a
thin command helper. Pure Genesis API — no Isaac. Constants are reproduced from the RoboLab build spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import genesis as gs

# --- asset (self-contained: mesh paths rewritten relative to the URDF, no RoboLab dependency) ---
ASSET = Path(__file__).resolve().parents[1] / "assets/robots/firefly_y6_gr100"
DUAL_URDF = str(ASSET / "dual_firefly_y6_gr100.urdf")

# --- joint names (verified loaded; match RoboLab exactly) ---
LEFT_ARM_JOINTS = [f"left_joint_{i}" for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"right_joint_{i}" for i in range(1, 7)]
LEFT_GRIPPER_DRIVEN_JOINT = "left_gripper_left_joint_1"
LEFT_GRIPPER_MIMIC_JOINT = "left_gripper_right_joint_1"
RIGHT_GRIPPER_DRIVEN_JOINT = "right_gripper_left_joint_1"
RIGHT_GRIPPER_MIMIC_JOINT = "right_gripper_right_joint_1"
LEFT_EE_LINK = "left_ee_link"
RIGHT_EE_LINK = "right_ee_link"
LEFT_LINK6 = "left_link_6"
RIGHT_LINK6 = "right_link_6"

# 14-D policy/state layout (6 arm + 1 driven gripper) x 2 arms — the mimic joints are NOT in the state.
STATE_JOINTS_14 = (LEFT_ARM_JOINTS + [LEFT_GRIPPER_DRIVEN_JOINT]
                   + RIGHT_ARM_JOINTS + [RIGHT_GRIPPER_DRIVEN_JOINT])

# --- gripper constants (MEASURED, from RoboLab) ---
GR100_OPEN = 0.0
GR100_CLOSE = 0.9          # firm pinch; q~0.45 -> 2cm gap, q~0.59 -> touch, q=0.9 -> no-penetration close
GR100_MIMIC = -1.0         # right_claw = -1 * driven (URDF axis flipped to match MuJoCo); coupled in action

# --- home pose (per arm) + base height ---
FIREFLY_HOME = [0.0, -0.75, 2.3, 0.9, 0.0, 0.0]
BASE_Z = 0.25              # arm-table height; dual base offsets (LEFT y=+0.224, RIGHT y=-0.224) are in the URDF

# --- PD gains from the MuJoCo MIT controller (per dof; firm gripper = do NOT lower) ---
ARM_KP = [200, 200, 200, 75, 15, 15]          # J1-3 proximal, J4-6 wrist
ARM_KV = [12.5, 12.5, 12.5, 6.0, 0.31, 0.31]
ARM_EFFORT = [28, 28, 28, 10, 10, 10]
GRIP_KP, GRIP_KV, GRIP_EFFORT = 200.0, 8.0, 10.0


class FireflyDual:
    """Loads the dual-arm robot into a Genesis scene and exposes name->dof maps + a command helper."""

    def __init__(self, scene: "gs.Scene", pos=(0.0, 0.0, BASE_Z)):
        self.entity = scene.add_entity(gs.morphs.URDF(
            file=DUAL_URDF, fixed=True, merge_fixed_links=False, pos=pos))
        self._built = False

    def _didx(self, jname: str) -> int:
        return int(self.entity.get_joint(jname).dofs_idx_local[0])

    def finalize(self):
        """Call AFTER scene.build(): resolve dof indices + push PD gains/effort limits."""
        self.arm = {"left": [self._didx(n) for n in LEFT_ARM_JOINTS],
                    "right": [self._didx(n) for n in RIGHT_ARM_JOINTS]}
        self.grip_driven = {"left": self._didx(LEFT_GRIPPER_DRIVEN_JOINT),
                            "right": self._didx(RIGHT_GRIPPER_DRIVEN_JOINT)}
        self.grip_mimic = {"left": self._didx(LEFT_GRIPPER_MIMIC_JOINT),
                          "right": self._didx(RIGHT_GRIPPER_MIMIC_JOINT)}
        self.state14_idx = [self._didx(n) for n in STATE_JOINTS_14]
        self.ee = {"left": LEFT_EE_LINK, "right": RIGHT_EE_LINK}
        # gains, laid out per the resolved dof order
        n = self.entity.n_dofs
        kp = np.zeros(n, np.float32); kv = np.zeros(n, np.float32); ef = np.zeros(n, np.float32)
        for side in ("left", "right"):
            for k, d in enumerate(self.arm[side]):
                kp[d], kv[d], ef[d] = ARM_KP[k], ARM_KV[k], ARM_EFFORT[k]
            for d in (self.grip_driven[side], self.grip_mimic[side]):
                kp[d], kv[d], ef[d] = GRIP_KP, GRIP_KV, GRIP_EFFORT
        self.entity.set_dofs_kp(kp); self.entity.set_dofs_kv(kv)
        self.entity.set_dofs_force_range(-ef, ef)
        self._built = True
        return self

    def home_qpos(self) -> np.ndarray:
        """Full dof vector at the home pose (arms = FIREFLY_HOME, grippers = open)."""
        q = np.zeros(self.entity.n_dofs, np.float32)
        for side in ("left", "right"):
            for d, v in zip(self.arm[side], FIREFLY_HOME):
                q[d] = v
            q[self.grip_driven[side]] = GR100_OPEN
            q[self.grip_mimic[side]] = GR100_MIMIC * GR100_OPEN
        return q

    def command(self, side: str, arm_q, grip: float):
        """PD-target the 6 arm joints + the coupled gripper (driven=grip, mimic=GR100_MIMIC*grip)."""
        self.entity.control_dofs_position(np.asarray(arm_q, np.float32), self.arm[side])
        self.entity.control_dofs_position(
            np.asarray([grip, GR100_MIMIC * grip], np.float32),
            [self.grip_driven[side], self.grip_mimic[side]])

    def ee_pose(self, side: str):
        """world (pos[3], quat_wxyz[4]) of the ee_link, as numpy (Genesis returns CUDA tensors)."""
        lk = self.entity.get_link(self.ee[side])
        return lk.get_pos().cpu().numpy(), lk.get_quat().cpu().numpy()
