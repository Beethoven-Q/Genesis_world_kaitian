# SPDX-License-Identifier: Apache-2.0
"""Genesis-native IK adapter — the IK backend for the Genesis line.

First-principles choice (verified): target the SIM's OWN ee_link with Genesis's built-in solver, which is
sub-millimetre exact (0.1 mm / 0.01 deg over the workspace) because it uses the dual URDF's exact kinematics.
RoboLab reused SODA's analytic IK, but SODA solves on a SEPARATE single-arm IK-chain URDF (gr100.urdf) whose
wrist/gripper geometry does NOT rigidly map to the dual sim URDF's ee_link (~1 cm residual in Genesis) — so
forcing it here would bake in grasp error. Genesis-native IK is simpler AND exact. (SODA's analytic IK still
runs on the REAL robot via soda-bimanual; the policy is sensor-only, so the collector's IK choice is internal.)

Exposes the SAME ``solve(ee_pos, ee_quat, q_init) -> IKSolution`` interface the reusable skills expect, so
``plan_pick_place`` / ``reachable_grasp_quat`` work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IKSolution:
    success: bool
    joints: np.ndarray | None   # (6,) arm joint solution, in the arm's dof order
    err_pos_m: float
    err_ori_rad: float


class GenesisArmIK:
    """Per-arm IK callable backed by Genesis ``robot.inverse_kinematics`` on the arm's ee_link.

    ``robot`` is a built+finalized ``FireflyDual``. Solving leaves the entity at the solution config; the
    caller (collector) then commands the chosen waypoint via PD, so that's harmless. q_init warm-starts."""

    def __init__(self, robot, side: str, pos_tol=5e-4, rot_tol=5e-3, max_iters=30):
        self.robot = robot
        self.side = side
        self.link = robot.entity.get_link(robot.ee[side])
        self.dofs = robot.arm[side]
        self.pos_tol, self.rot_tol, self.max_iters = pos_tol, rot_tol, max_iters

    def solve(self, ee_pos, ee_quat=None, q_init=None) -> IKSolution:
        """Arm joints so ee_link reaches ``ee_pos`` (+ optional ee_quat, wxyz). q_init is the arm-dof seed."""
        init = None
        if q_init is not None:
            init = self.robot.entity.get_dofs_position().clone() if hasattr(
                self.robot.entity.get_dofs_position(), "clone") else None
            # write the seed into a full-dof init vector
            full = self.robot.entity.get_dofs_position()
            full = full.cpu().numpy() if hasattr(full, "cpu") else np.asarray(full, float)
            for d, v in zip(self.dofs, np.asarray(q_init, float)):
                full[d] = v
            init = full
        kw = dict(link=self.link, pos=np.asarray(ee_pos, np.float32), dofs_idx_local=self.dofs,
                  max_solver_iters=self.max_iters, pos_tol=self.pos_tol, rot_tol=self.rot_tol,
                  return_error=True)
        if ee_quat is not None:
            kw["quat"] = np.asarray(ee_quat, np.float32)
        if init is not None:
            kw["init_qpos"] = np.asarray(init, np.float32)
        qsol, err = self.robot.entity.inverse_kinematics(**kw)
        qsol = qsol.cpu().numpy() if hasattr(qsol, "cpu") else np.asarray(qsol, float)
        err = err.cpu().numpy() if hasattr(err, "cpu") else np.asarray(err, float)
        joints = np.array([float(qsol[d]) for d in self.dofs], float)
        epos = float(np.linalg.norm(np.asarray(err).ravel()[:3]))
        success = epos < (self.pos_tol * 4)   # convergence guard
        return IKSolution(success, joints if success else None, epos, 0.0)
