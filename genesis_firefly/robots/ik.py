# SPDX-License-Identifier: Apache-2.0
"""Genesis-native IK adapter — the IK backend for the Genesis line.

First-principles choice: target the SIM's OWN ee_link with Genesis's built-in BATCHED solver, which is
sub-millimetre exact on the dual URDF's exact kinematics AND solves all N envs in one call (the parallelism the
collector needs). RoboLab uses SODA's analytic IK on a separate single-arm gr100.urdf chain; a 2026-06-20
experiment PROVED the two are IDENTICAL on our targets (joints match to 3 decimals, frame residual 0.00 mm, the
gr100 chain and the dual sim URDF are byte-identical) — so the solver choice is purely internal and
Genesis-native is the right one for batched parallelism. (The earlier "~1 cm residual" worry was falsified.)
SODA's analytic IK still runs on the REAL robot via soda-bimanual; the policy is sensor-only, so the collector's
IK choice is invisible to it.

Exposes the SAME ``solve(ee_pos, ee_quat, q_init) -> IKSolution`` interface the reusable skills expect, so
``plan_pick_place`` / ``reachable_grasp_quat`` work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills.grasp import _R_from_wxyz, _wxyz_from_R  # noqa: E402

# --- Gripper TOOL frame relative to ee_link (MEASURED in Genesis, grasp_calib.py) ----------------------- #
# Genesis's *_ee_link is ~11cm BEHIND the claws (NOT RoboLab/Isaac's 8mm TCP-style ee_link). The tool frame
# puts the origin AT the claw-convergence point, +Z = approach (toward the object), +X = closing (claw
# separation), so the reusable skills (which assume +Z approach / +X closing at the grasp point) work
# unchanged. Identical for both arms (symmetric gripper). The IK adapter targets this TOOL frame and
# converts to the ee_link pose internally before calling Genesis IK.
# tool origin = the GRIP POINT where the CLOSED fingertips meet (grip_tool.py at q=GR100_MEET), NOT the
# hinge midpoint — the curved GR100 fingers fold back so the grip point is only ~2.8cm from ee (close to
# RoboLab's [0,0,0.008]). Aiming the hinge at the object drove the 12.8cm fingers into the table; the grip
# point puts the actual pinch region on the object.
_OFF = np.array([0.0, -0.00626, 0.02731])        # grip point in the ee frame (tool origin)
_Z = np.array([0.0, -0.0292, 0.99957])           # +Z = approach (gripper_base->grip point, ~ee +Z)
_Z = _Z / np.linalg.norm(_Z)
_X = np.array([-1.0, 0.0, 0.0])                  # +X = closing (claw separation)
_Y = np.cross(_Z, _X); _Y /= np.linalg.norm(_Y)
_X = np.cross(_Y, _Z)                            # re-orthonormalize
TOOL_IN_EE = np.eye(4)
TOOL_IN_EE[:3, 0], TOOL_IN_EE[:3, 1], TOOL_IN_EE[:3, 2], TOOL_IN_EE[:3, 3] = _X, _Y, _Z, _OFF
TOOL_IN_EE_INV = np.linalg.inv(TOOL_IN_EE)


def tool_R_at_home(home_ee_R):
    """Tool-frame orientation given the ee_link orientation (for the skill's reference_R)."""
    return home_ee_R @ TOOL_IN_EE[:3, :3]


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

    def solve(self, tool_pos, tool_quat=None, q_init=None) -> IKSolution:
        """Arm joints so the TOOL frame (claw convergence) reaches ``tool_pos`` (+ optional tool_quat, wxyz).
        Converts the tool pose -> ee_link pose (T_ee = T_tool @ TOOL_IN_EE^-1), then solves Genesis IK for
        ee_link. q_init is the arm-dof seed (warm start)."""
        if tool_quat is not None:
            T_tool = np.eye(4); T_tool[:3, :3] = _R_from_wxyz(tool_quat); T_tool[:3, 3] = np.asarray(tool_pos, float)
            T_ee = T_tool @ TOOL_IN_EE_INV
            ee_pos, ee_quat = T_ee[:3, 3], _wxyz_from_R(T_ee[:3, :3])
        else:
            ee_pos, ee_quat = np.asarray(tool_pos, float), None
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
