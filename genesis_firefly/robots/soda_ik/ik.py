# SPDX-License-Identifier: Apache-2.0
"""In-process IK bridge to soda-bimanual's analytic solver (clean, isolated).

Wraps soda_os' ``Kinematics`` (analytic IK + pinocchio FK) — one solver per arm — and handles the
world<->base translation so callers work purely in WORLD coordinates. No server / subprocess / ZMQ /
second venv (v1's ~330-line glue is gone): the Isaac env owns state + actuation, this owns geometry.

Targets are the gripper TCP (*_ee_link, use_tcp_frame=True), the same frame the sim measures.
Quaternions are wxyz on both sides. The gripper close/open semantics live in the robot config's
actuator (firefly_dual.py), NOT here — this layer is geometry only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from . import config as C

# Measured constant frame offset between soda's TCP frame (what solve_ik/forward use) and Isaac's
# *_ee_link body frame (what the sim reports via body_quat_w). Origins coincide (round-trip pos 0.00mm);
# the axes differ by EXACTLY +90 deg about local Z, identical for both arms (scripts/ik_verify). Baking
# it here makes this bridge speak Isaac's ee_link frame natively, so a commanded EE pose and the measured
# EE pose are in the SAME frame everywhere downstream:
#   R_eelink = R_tcp @ Rz(+90deg)   (forward)      R_tcp = R_eelink @ Rz(-90deg)   (inverse)
# NOTE: was Rz(-90) until the gripper mount was flipped Rz(-90)->Rz(+90) in build_firefly_dual_urdf.py
# (to match MuJoCo's gripper orientation). That 180deg flip rotated ee_link 180deg about Z, so this
# offset flipped sign with it. soda's TCP frame (link_6 + 0.187 Z) is unchanged.
_RZ_P90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(+90 deg)


def _quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _ensure_soda_on_path() -> None:
    for p in (C.SODA_BIMANUAL_ROOT, f"{C.SODA_BIMANUAL_ROOT}/utils"):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class IKSolution:
    success: bool
    joints: np.ndarray | None      # (6,) arm joint solution
    err_pos_m: float
    err_ori_rad: float


class FireflyDualIK:
    """Dual-arm IK + FK in WORLD coordinates. ``base_z`` = arm-table height (keep == the scene)."""

    def __init__(self, base_z: float = 0.25):
        _ensure_soda_on_path()
        from soda_os.core.kinematics import Kinematics
        self.base_z = float(base_z)
        self._kin = {side: Kinematics(urdf_path=C.FIREFLY_IK_URDF, use_tcp_frame=True,
                                      tcp_offset=C.TCP_OFFSET)
                     for side in ("left", "right")}

    def _base(self, side: str) -> np.ndarray:
        return np.array([0.0, C.BASE_Y[side], self.base_z])

    def solve(self, side: str, pos_world, quat_wxyz=None, q_init=None) -> IKSolution:
        """Solve arm joints so the ee_link reaches ``pos_world`` (+ optional orientation, wxyz, ee_link
        frame -- the same frame Isaac reports). The TCP frame offset is handled internally."""
        pos_base = np.asarray(pos_world, float) - self._base(side)
        ori = None
        if quat_wxyz is not None:
            # incoming target is the ee_link frame; convert to soda's TCP frame: R_tcp = R_eelink @ Rz(-90)
            ori = _R_to_quat_wxyz(_quat_wxyz_to_R(quat_wxyz) @ _RZ_P90.T)
        q0 = np.zeros(C.ARM_DOF) if q_init is None else np.asarray(q_init, float)
        r = self._kin[side].solve_ik(pos_base, target_ori=ori, q_init=q0, method="auto")
        return IKSolution(bool(r.success), None if r.joints is None else np.asarray(r.joints, float),
                          float(r.error_pos or 0.0), float(r.error_ori or 0.0))

    def fk(self, side: str, q6) -> tuple[np.ndarray, np.ndarray]:
        """World ee_link pose (pos, 3x3 R) for arm joints ``q6`` -- the frame Isaac reports.
        Origin == TCP point; orientation = R_tcp @ Rz(+90) (measured offset; see _RZ_P90)."""
        T = np.asarray(self._kin[side].forward(np.asarray(q6, float)))
        return T[:3, 3] + self._base(side), T[:3, :3] @ _RZ_P90

    def home(self) -> np.ndarray:
        return np.array(C.ARM_HOME)
