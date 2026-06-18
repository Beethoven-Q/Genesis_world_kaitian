# SPDX-License-Identifier: Apache-2.0
"""Reusable PICK-AND-PLACE skill: from an object's geometry + an IK solver, produce a fully reachable
pick+place waypoint plan, and score the outcome by PHYSICS (not by command success).

Object-agnostic glue between the geometry/grasp primitives (``robolab.skills.grasp``) and a thin sim
executor (the collector). It owns NOTHING sim-specific -- numbers in, a plan / a verdict out -- so the
SAME code serves apple/banana/pen/cube and any future object/task.

ORIENTATION POLICY (the reusable part worth getting right): **prefer top-down, but not strictly.** A
top-down grasp keeps the gripper vertical (orientation-aware: closing axis perpendicular to the object's
reference axis) and a top-down place lowers straight into the bowl / lifts straight out -- the cleanest,
collision-safe motion. But the arm's top-down reach is finite, so we RELAX to the SMALLEST tilt that makes
IK solvable (v1's ``grasp_min_tilt_deg`` idea, generalised). This is exposed as two composable skills --
``reachable_grasp_quat`` and ``reachable_place_quat`` -- which pen-uncap and other tasks reuse directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grasp import (
    grasp_waypoints, place_waypoints, orientation_aware_grasp_quat, GR100_GRASP_OFFSET_EE,
    tilted_base_quat, transport_quats, _R_from_wxyz)

# Prefer pure top-down (tilt 0); relax to the smallest of these tilts that makes IK solvable.
_TILT_STEPS_DEG = (0.0, 8.0, 16.0, 24.0, 32.0)


def _reachable(ik_solve, candidates, target_pos, q_init, grasp_offset_ee):
    """First orientation in ``candidates`` whose ee pose (so the claw convergence lands at ``target_pos``)
    is IK-solvable, or None. ``ik_solve(ee_pos, ee_quat, q_init) -> sol`` with ``.success`` / ``.joints``."""
    for q in candidates:
        ee = np.asarray(target_pos, float) - _R_from_wxyz(q) @ np.asarray(grasp_offset_ee, float)
        sol = ik_solve(ee, q, q_init)
        if sol is not None and getattr(sol, "joints", None) is not None and sol.success:
            return q
    return None


def reachable_grasp_quat(ik_solve, grasp_center, ref_axis_world, q_init, *, reference_R=None,
                         arm_base_xy=(0.0, 0.0), grasp_offset_ee=GR100_GRASP_OFFSET_EE,
                         tilt_steps_deg=_TILT_STEPS_DEG):
    """PREFER TOP-DOWN grasp; relax to the smallest tilt (toward the reach direction) that makes the
    orientation-aware grasp IK-solvable at ``grasp_center``. Returns ``(grasp_quat, tilt_deg_used)``. The
    closing axis stays perpendicular to ``ref_axis_world`` at every tilt, so an elongated object is still
    gripped across its short side and a cube on an opposite face pair."""
    reach = (np.asarray(grasp_center, float)[:2] - np.asarray(arm_base_xy, float))
    fallback = None
    for tilt in tilt_steps_deg:
        gq = orientation_aware_grasp_quat(ref_axis_world, tilted_base_quat(reach, tilt), reference_R=reference_R)
        if _reachable(ik_solve, [gq], grasp_center, q_init, grasp_offset_ee) is not None:
            return gq, float(tilt)
        fallback = fallback if fallback is not None else gq
    return fallback, float(tilt_steps_deg[-1])


def reachable_place_quat(ik_solve, over_bowl, grasp_quat, q_init, *, arm_base_xy=(0.0, 0.0),
                         grasp_offset_ee=GR100_GRASP_OFFSET_EE, tilt_steps_deg=_TILT_STEPS_DEG):
    """PREFER TOP-DOWN transport at the bowl (wrist-yaw branches around the grasp, least rotation first);
    relax to the smallest tilt (toward the bowl) only if no top-down branch is IK-solvable at ``over_bowl``.
    Returns ``(place_quat, tilt_deg_used)``. A held object is orientation-free, so any reachable carry
    works -- we keep the tilt minimal so the gripper still lowers ~straight into the bowl (no deep dip)."""
    reach = (np.asarray(over_bowl, float)[:2] - np.asarray(arm_base_xy, float))
    for tilt in tilt_steps_deg:
        base = grasp_quat if tilt == 0 else tilted_base_quat(reach, tilt)
        cand = _reachable(ik_solve, transport_quats(base, reference_quat=grasp_quat),
                          over_bowl, q_init, grasp_offset_ee)
        if cand is not None:
            return cand, float(tilt)
    return np.asarray(grasp_quat, float), float(tilt_steps_deg[-1])


@dataclass
class PickPlacePlan:
    grasp_quat: np.ndarray   # chosen grasp orientation (ee_link wxyz)
    place_quat: np.ndarray   # chosen carry/place orientation
    grasp_tilt_deg: float    # tilt from top-down the grasp needed (0 = pure top-down)
    place_tilt_deg: float    # tilt from top-down the place needed
    pick: list               # labelled grasp waypoints (pre_grasp/at_object/close/lift)
    place: list              # labelled place waypoints ([reorient]/carry/lower/release/retreat)


def plan_pick_place(grasp_center, bowl_pos, ref_axis_world, ik_solve, q_init, *, open_g, close_g,
                    arm_base_xy=(0.0, 0.0), reference_R=None, grasp_offset_ee=GR100_GRASP_OFFSET_EE,
                    grasp_dz=0.0, approach_h=0.12, lift_h=0.18,
                    place_approach_h=0.08, release_dz=0.05, release_g=None,
                    tilt_steps_deg=_TILT_STEPS_DEG) -> PickPlacePlan:
    """ONE call: object geometry -> a fully reachable pick+place plan.

    Chooses the GRASP orientation (prefer top-down, relax minimal tilt) and the CARRY/PLACE orientation
    (prefer top-down transport, relax minimal tilt), then builds the waypoints. ``ref_axis_world`` is the
    object's reference axis (long axis for elongated / a face normal for a cube / None for round).
    ``ik_solve(ee_pos, ee_quat, q_init) -> sol`` decouples this from any IK backend. If the carry
    orientation ends up different from the grasp orientation, the wrist is REORIENTED IN PLACE at the lift
    before translating (doing both at once let the IK -- seeded from the lift config -- stall short)."""
    grasp_center = np.asarray(grasp_center, float)
    bowl_pos = np.asarray(bowl_pos, float)
    gq, gtilt = reachable_grasp_quat(ik_solve, grasp_center, ref_axis_world, q_init,
                                     reference_R=reference_R, arm_base_xy=arm_base_xy,
                                     grasp_offset_ee=grasp_offset_ee, tilt_steps_deg=tilt_steps_deg)
    over_bowl = bowl_pos + np.array([0.0, 0.0, place_approach_h])
    cq, ptilt = reachable_place_quat(ik_solve, over_bowl, gq, q_init, arm_base_xy=arm_base_xy,
                                     grasp_offset_ee=grasp_offset_ee, tilt_steps_deg=tilt_steps_deg)
    pick = grasp_waypoints(grasp_center, open_g, close_g, quat=gq, grasp_offset_ee=grasp_offset_ee,
                           grasp_dz=grasp_dz, approach_h=approach_h, lift_h=lift_h)
    place = place_waypoints(bowl_pos, open_g, close_g, quat=cq, grasp_offset_ee=grasp_offset_ee,
                            approach_h=place_approach_h, release_dz=release_dz, release_g=release_g)
    if not np.allclose(cq, gq):
        place = [("reorient", pick[-1][1], cq, close_g)] + place
    return PickPlacePlan(gq, cq, gtilt, ptilt, pick, place)


def score_pick_place(obj_start_pos, obj_lifted_pos, obj_final_pos, bowl_pos, *, obj_ee_dist_m=None,
                     lift_thresh_cm=3.0, xy_thresh_cm=8.0, z_margin_m=0.02, z_settle_max_m=0.045,
                     min_ee_dist_m=0.08) -> dict:
    """PHYSICAL success (god-mode ground truth), NOT command success:
      * grasped = the object was lifted clear of the table during the PICK (``obj_lifted_pos`` Z >=
        start Z + lift_thresh). MUST be measured right after the lift, BEFORE the place lowers it again.
      * placed  = grasped AND the object came to REST IN the bowl: final XY within the bowl mouth, and
        final Z in a band around the bowl (NOT below = on the floor, and NOT well above = still HELD /
        lodged on a finger hovering over the bowl). If ``obj_ee_dist_m`` is given, the object must also
        be clear of the ee (>= ``min_ee_dist_m``) -- catches a thin object stuck to the gripper that the
        z-band alone might miss. (This closes the false-positive where a pen lodged on a claw retreats
        OVER the bowl and scores as placed.)
    Returns the booleans + raw metrics (cm) for logging / DR analysis."""
    s = np.asarray(obj_start_pos, float); l = np.asarray(obj_lifted_pos, float)
    f = np.asarray(obj_final_pos, float); b = np.asarray(bowl_pos, float)
    lift_cm = (l[2] - s[2]) * 100.0
    xy_to_bowl = float(np.linalg.norm(f[:2] - b[:2])) * 100.0
    grasped = lift_cm > lift_thresh_cm
    in_bowl = (b[2] - z_margin_m) < f[2] < (b[2] + z_settle_max_m)   # settled IN the bowl, not hovering
    released = (obj_ee_dist_m is None) or (obj_ee_dist_m >= min_ee_dist_m)
    placed = bool(grasped and xy_to_bowl < xy_thresh_cm and in_bowl and released)
    return dict(grasped=bool(grasped), placed=placed,
                lift_cm=round(lift_cm, 1), xy_to_bowl=round(xy_to_bowl, 1),
                final_dz_cm=round((f[2] - b[2]) * 100, 1))
