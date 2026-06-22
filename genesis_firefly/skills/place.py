# SPDX-License-Identifier: Apache-2.0
"""skills/place.py — the per-env PLACE action (de-closured from tasks/pickplace.py::place_tail).

SIM-AGNOSTIC: returns a list of labelled ``(label, pos, quat_wxyz, grip)`` end-effector waypoints in the
ee_link frame the FireflyDualIK bridge / Isaac both speak (the same contract as skills/grasp.py). A thin
executor drives them through the IK bridge.

The place action is APPENDED to the same continuous per-env waypoint stream as the grasp action, so each
env runs pick->place->home as ONE smooth trajectory and TERMINATES at home -- no staged barrier, no mid-air
wait for other envs. This is the reusable place SKILL: a future pick-place-style task imports + composes it
(with skills/grasp.py::grasp_action_wps) instead of reproducing the carry/lower/release/retract/home shape.

The CARRY orientation ``carry_quat`` is computed by the CALLER (e.g. via skills.grasp.cquat, the wrist-margin-
aware carry tilt) and passed in, so this module stays a pure geometric waypoint builder with NO dependency on
the grasp-planning GraspContext -- exactly the de-closure pattern the grasp-planning functions use.
"""
import numpy as np


def place_action_wps(i, lift_pose, carry_quat, bowl_xyz, papp, open_g, close_g, home_tool_i, home_tquat_i):
    """The per-env PLACE action: from the LIFTED grasp pose ``lift_pose`` (held closed) re-yaw to the carry
    orientation ``carry_quat``, carry over the bowl, lower in, release, retract, GO HOME. Returns labelled
    (ee_link) waypoints ``(label, pos, quat_wxyz, grip)``.

    De-closured from the in-line ``place_tail`` closure: the caller supplies the lift pose, the (already
    wrist-margin-relaxed) carry quat, the over-bowl release point ``bowl_xyz``, the carry-hover / retreat
    height ``papp`` above the bowl, the gripper open/close scalars, and the env's home tool pose -- so a
    FUTURE task imports + composes this without reproducing the carry/release/home shape.

    ``i`` is accepted for caller-side bookkeeping / symmetry with ``grasp_action_wps`` (the action itself reads
    only the explicit args)."""
    lift_pose = np.asarray(lift_pose, float)
    bowl_xyz = np.asarray(bowl_xyz, float)
    return [
        ("lift",    lift_pose + [0, 0, 0.02],   carry_quat,   close_g),   # small settle + re-yaw to the carry
        ("carry",   bowl_xyz + [0, 0, papp],    carry_quat,   close_g),
        ("lower",   bowl_xyz,                    carry_quat,   close_g),
        ("rel",     bowl_xyz,                    carry_quat,   open_g),    # pose held -> gripper release ramp
        ("ret",     bowl_xyz + [0, 0, papp],    carry_quat,   open_g),
        ("go_home", home_tool_i,                home_tquat_i, open_g),     # smooth densified RETURN HOME (recorded)
    ]
