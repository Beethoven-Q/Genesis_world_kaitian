# SPDX-License-Identifier: Apache-2.0
"""Grasp + place skills: plan EE-space waypoints for a top-down (or tilted) pick and a place.

SIM-AGNOSTIC: these return lists of ``(ee_pos_world, ee_quat_wxyz, gripper_scalar)`` waypoints in the
ee_link frame the FireflyDualIK bridge / Isaac both speak. A thin executor drives them through the IK
bridge. Reachability across the whole object-table workspace (incl. table-level top-down) was verified
by the pure-IK probe (scripts/grasp/temp/probe_reach.py): 100% reachable, ~0.001mm IK error.

The gripper scalar is the driven-claw target (e.g. GR100_OPEN / GR100_CLOSE); the executor mirrors it to
the mimic claw. Yaw rotates the jaws about the vertical approach axis to align with an object's minor
axis (for elongated objects); a cube is yaw-agnostic.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _R

# GR100 grasp-frame offset: the vector from ee_link to where the (axis-corrected) claws CONVERGE at the
# pinch angle, in the ee_link frame. MEASURED via FK at q=0.45 (scripts/grasp/temp/measure_grasp_frame.py):
# laterally CENTRED (x=y~0) -- the claws converge on the ee_link centreline -- and +0.8cm along the tool
# axis (+Z). Getting the LATERAL component right matters: a spurious 1cm Y here shifts the convergence off
# a small cube and the grasp misses at DR poses. Applied in the EE frame: ee_target = grasp_point - R @ offset.
GR100_GRASP_OFFSET_EE = np.array([0.0, 0.0, 0.008])

# The gripper CLOSING axis in the ee_link frame: the unit direction along which the two claws
# separate/converge (so an elongated object should lie PERPENDICULAR to it). MEASURED from the live claw
# body poses at the open angle (scripts/grasp/temp/verify_grasp_geometry.py): the claws separate along
# ee_link +X. (ee +Z is the approach/tool axis; +X is the open/close axis; +Y is the jaw "width".)
GR100_CLOSING_AXIS_EE = np.array([1.0, 0.0, 0.0])


def _R_from_wxyz(q):
    q = np.asarray(q, float)
    return _R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _wxyz_from_R(R: np.ndarray) -> np.ndarray:
    q = _R.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def world_long_axis(local_axis, root_quat_wxyz) -> np.ndarray:
    """Express an object's local long/reference axis (e.g. ObjectSpec.long_axis_local()) in the WORLD
    frame given the object's current world quaternion. Returns a unit vector (or None if local_axis is
    None -- a round object with no preferred grasp axis)."""
    if local_axis is None:
        return None
    v = _R_from_wxyz(root_quat_wxyz) @ np.asarray(local_axis, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def world_grasp_center(root_pos, root_quat_wxyz, local_center) -> np.ndarray:
    """True grasp centre = the object's GEOMETRY bbox centre in world (root_pos + R(root_quat)@local_center).
    Use this, NOT root_pos: some YCB rigid bodies (banana) pivot off their visual/collision centre."""
    return np.asarray(root_pos, float) + _R_from_wxyz(root_quat_wxyz) @ np.asarray(local_center, float)


_TRANSPORT_DELTAS = (0.0, np.pi, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4)


def transport_quats(base_quat, reference_quat=None, *, deltas=_TRANSPORT_DELTAS):
    """Candidate CARRY/PLACE orientations: ``base_quat`` rotated about its OWN approach axis (ee +Z) by
    each delta -- i.e. same approach tilt, different wrist yaw -- ordered by LEAST rotation from
    ``reference_quat`` (default = base_quat). Generalises v1's ``_top_down_transport_Rs`` to any base tilt:
    pass a top-down base for pure top-down transport, or a slightly-tilted base when top-down can't reach.
    The collector tries these until one is IK-solvable, so a HELD object is carried to a reachable wrist
    orientation without changing the (preferred) approach direction."""
    base_quat = np.asarray(base_quat, float)
    Rb = _R_from_wxyz(base_quat)
    a = Rb @ np.array([0.0, 0.0, 1.0]); a /= np.linalg.norm(a)   # approach axis (ee +Z) in world
    Rref = _R_from_wxyz(np.asarray(reference_quat, float)) if reference_quat is not None else Rb
    quats = [_wxyz_from_R(_R.from_rotvec(a * d).as_matrix() @ Rb) for d in deltas]
    quats.sort(key=lambda q: -float(np.trace(Rref.T @ _R_from_wxyz(q))))   # least rotation first
    return quats


def top_down_transport_quats(grasp_quat, **_kw):
    """Back-compat alias: top-down transport = transport_quats around the (top-down) grasp orientation."""
    return transport_quats(grasp_quat)


def tilted_base_quat(reach_dir_xy, tilt_deg: float) -> np.ndarray:
    """A 'mostly top-down' base orientation: the approach axis (ee +Z) tilted ``tilt_deg`` away from
    straight-down (-Z) toward the horizontal ``reach_dir_xy`` (the direction the arm reaches OUT). tilt=0
    is pure top-down. The reachable_* skills relax top-down to the MINIMAL tilt IK needs by stepping this
    up. ee X/Y are arbitrary perpendiculars (the orientation skill / transport re-yaw them)."""
    r = np.asarray(reach_dir_xy, float)[:2]; n = np.linalg.norm(r)
    r = r / n if n > 1e-9 else np.array([1.0, 0.0])
    th = np.radians(float(tilt_deg))
    a = np.sin(th) * np.array([r[0], r[1], 0.0]) + np.cos(th) * np.array([0.0, 0.0, -1.0])
    a /= np.linalg.norm(a)
    x = np.cross(np.array([0.0, 0.0, 1.0]), a)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    y = np.cross(a, x); y /= np.linalg.norm(y)
    return _wxyz_from_R(np.column_stack([x, y, a]))


def orientation_aware_grasp_quat(ref_axis_world, base_quat, *, closing_axis_ee=GR100_CLOSING_AXIS_EE,
                                 reference_R=None) -> np.ndarray:
    """Reusable orientation-aware grasp skill. Given an object's world reference axis ``ref_axis_world``
    (long axis for an elongated object; a face normal for a cube; None for round) and a reachable
    approach orientation ``base_quat`` (ee_link wxyz; its +Z is the approach/tool axis), return the
    grasp quat obtained by rotating ``base_quat`` about its OWN approach axis so the gripper's closing
    axis ends up PERPENDICULAR to ``ref_axis_world`` (projected onto the plane perpendicular to the
    approach axis). The gripper therefore closes ACROSS the object's long axis (elongated) or onto an
    opposite FACE pair (cube), never along it / across the diagonal.

    Parallel-jaw symmetry gives two branches (psi and psi+180 deg, same physical grip); we pick the one
    requiring the smaller wrist rotation from ``reference_R`` (the arm's current ee R) if given, else the
    one closest to ``base_quat`` -- so transport stays smooth. ``ref_axis_world=None`` returns base_quat
    unchanged (round object: any yaw grips equally)."""
    base_quat = np.asarray(base_quat, float)
    if ref_axis_world is None:
        return base_quat
    Rb = _R_from_wxyz(base_quat)
    a = Rb @ np.array([0.0, 0.0, 1.0])                  # approach/tool axis in world (ee +Z)
    a /= np.linalg.norm(a)
    ref = np.asarray(ref_axis_world, float)
    ref_p = ref - (ref @ a) * a                         # project ref onto plane perpendicular to approach
    if np.linalg.norm(ref_p) < 1e-3:                    # ref ~parallel to approach -> no in-plane constraint
        return base_quat
    ref_p /= np.linalg.norm(ref_p)
    desired = np.cross(a, ref_p)                        # in-plane, perpendicular to ref -> target closing dir
    desired /= np.linalg.norm(desired)
    c0 = Rb @ np.asarray(closing_axis_ee, float)        # current closing axis in world
    c0_p = c0 - (c0 @ a) * a
    if np.linalg.norm(c0_p) < 1e-6:
        return base_quat
    c0_p /= np.linalg.norm(c0_p)
    Rref = reference_R if reference_R is not None else Rb
    best, best_score = base_quat, -np.inf
    for sgn in (+1.0, -1.0):
        tgt = sgn * desired
        psi = np.arctan2(float(np.cross(c0_p, tgt) @ a), float(c0_p @ tgt))   # signed angle about a
        Rnew = _R.from_rotvec(a * psi).as_matrix() @ Rb
        score = float(np.trace(Rref.T @ Rnew))         # higher = smaller relative wrist rotation
        if score > best_score:
            best, best_score = _wxyz_from_R(Rnew), score
    return best


def top_down_quat(yaw_deg: float = 0.0) -> np.ndarray:
    """ee_link orientation with the tool/approach axis (+Z) pointing straight DOWN, jaws yawed by
    ``yaw_deg`` about the vertical. Returns wxyz."""
    Rf = _R.from_euler("z", yaw_deg, degrees=True) * _R.from_euler("x", 180, degrees=True)
    q = Rf.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def grasp_waypoints(obj_pos, open_g: float, close_g: float, *, quat=None, grasp_offset_ee=GR100_GRASP_OFFSET_EE,
                    grasp_yaw: float = 0.0, approach_h: float = 0.12, lift_h: float = 0.18, grasp_dz: float = 0.0):
    """Pick. Returns labelled (ee_link) waypoints:
    pre-grasp(open, above) -> at-object(open) -> close -> lift(closed, above).
    ``quat`` = approach orientation (ee_link wxyz); default top-down. For this arm at the MuJoCo object
    height (~0.47) the reachable grasp is TILTED (use the home EE orientation), not top-down.
    ``grasp_offset_ee`` (EE-frame) places the claw convergence point at the grasp point: the returned
    positions are ee_link targets = grasp_point - R(quat) @ offset. ``grasp_dz`` nudges the grasp height."""
    q = top_down_quat(grasp_yaw) if quat is None else np.asarray(quat, float)
    R = _R_from_wxyz(q)
    tool_z = R @ np.array([0.0, 0.0, 1.0])   # the gripper's approach axis in world (ee_link +Z, toward object)
    grasp_pt = np.asarray(obj_pos, float) + np.array([0.0, 0.0, grasp_dz])
    at = grasp_pt - R @ np.asarray(grasp_offset_ee, float)   # ee_link target so claws converge on grasp_pt
    # pre-grasp backs off ALONG the (tilted) approach axis, so the descent comes straight in along the
    # gripper's pointing direction -- not sideways in world-z, which would sweep the claws into the object.
    above = at - approach_h * tool_z
    lift = at + np.array([0.0, 0.0, lift_h])  # lift straight up to clear the table
    return [("pre_grasp", above, q, open_g),
            ("at_object", at, q, open_g),
            ("close", at, q, close_g),
            ("lift", lift, q, close_g)]


def place_waypoints(target_pos, open_g: float, close_g: float, *, quat=None, grasp_offset_ee=GR100_GRASP_OFFSET_EE,
                    place_yaw: float = 0.0, approach_h: float = 0.16, release_dz: float = 0.06,
                    release_g: float | None = None):
    """Carry-to-target(closed, above) -> lower(closed) -> release(partial-open) -> retreat(full-open, above).
    ``quat`` = carry orientation (ee_link wxyz); default top-down. ee_link targets are offset by the same
    grasp-frame transform so the HELD object (at the claw convergence point) lands over the target.

    ``release_g`` = the gripper command at the in-bowl RELEASE: a PARTIAL open (between close_g and open_g)
    that frees the object WITHOUT fully splaying the claws into the (small) bowl -- the claws then open
    FULLY only on the retreat, once they have lifted clear of the rim. Default = open_g (full open, the
    old behaviour). A partial release_g keeps the big GR100 from flinging the light bowl on deposit."""
    q = top_down_quat(place_yaw) if quat is None else np.asarray(quat, float)
    rg = open_g if release_g is None else release_g
    R = _R_from_wxyz(q)
    off = R @ np.asarray(grasp_offset_ee, float)
    tgt = np.asarray(target_pos, float)
    over = tgt + np.array([0.0, 0.0, approach_h]) - off
    drop = tgt + np.array([0.0, 0.0, release_dz]) - off
    return [("carry", over, q, close_g),
            ("lower", drop, q, close_g),
            ("release", drop, q, rg),          # partial open: free the object, don't splay into the bowl
            ("retreat", over, q, open_g)]      # full open only after lifting clear of the rim
