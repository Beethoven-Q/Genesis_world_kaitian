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

from dataclasses import dataclass

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


# ============================================================================================ #
# HIGHER-LEVEL GRASP-ORIENTATION / WRIST-MARGIN PLANNING  (moved here from tasks/pickplace.py)
# ============================================================================================ #
# These BUILD ON the low-level primitives above (tilted_base_quat, orientation_aware_grasp_quat,
# transport_quats, world_long_axis, _R_from_wxyz, _wxyz_from_R) to plan the PER-ENV grasp + carry
# ORIENTATION and the RoboLab-faithful WRIST-MARGIN relax-tilt selection. They were closures over a
# collect()'s locals; they are now PURE functions taking an immutable ``GraspContext`` (the per-collect
# bundle built once) plus the working ``solve``/``gqA`` passed explicitly. Behaviour is byte-identical to
# the in-line collector (the cube 8/8 · T=977 · RNG-order regression proves it); only the structure moved.
# The DISTURBANCE trajectory ASSEMBLY (chase/recover waypoint construction) stays in the task and merely
# CALLS ``grasp_quat_at`` / ``select_grasp_tilt_at`` / ``select_place_tilt_at`` from here.


@dataclass(frozen=True)
class GraspContext:
    """The immutable per-collect bundle the grasp-planning functions need (built ONCE in collect()). Holds the
    per-env DR arrays, the arm/object geometry, the lift/approach heights, and the wrist-margin thresholds -- the
    former closure environment, frozen so a moved function reads exactly what the in-line closure did. The two
    values that VARY during a run (the active-arm ``solve`` callable and the working grasp quats ``gqA``) are NOT
    stored here; they are passed explicitly to each function so the de-closure is faithful (gqA is mutated by the
    disturbance path AFTER select_place_tilt runs)."""
    N: int
    side_is_left: np.ndarray
    base: dict          # {"left": (x,y), "right": (x,y)} -- the arm-base xy the reach direction is measured from
    htR: dict           # {"left": R, "right": R} -- the home tool R (the wrist-roll reference for round objects)
    laxis: object       # spec.long_axis_local() (a unit vec, or None for a round object)
    spec: object        # the ObjectSpec (only .is_cube is read -- the symmetry fold)
    yaw: np.ndarray     # per-env in-plane object yaw
    gc: np.ndarray      # (N,3) the grasp centre (object body centre + grasp_dz)
    bxyz: np.ndarray    # (N,3) the over-bowl release point
    bowx: np.ndarray    # per-env bowl x
    bowy: np.ndarray    # per-env bowl y
    APP: float          # pre-grasp standoff along the approach axis
    LIFT: float         # post-grasp lift height
    PAPP: float         # carry-hover / retreat height above the bowl
    tilt_steps: tuple   # _GRASP_TILT_STEPS_DEG (the relax ladder, capped per object)
    WRIST_LIMIT: float  # joint_4 hard limit (URDF)
    WRIST_MARGIN: float  # keep |j4| <= WRIST_LIMIT - WRIST_MARGIN
    ELBOW_MIN: float    # keep j3 (elbow) >= this


_NO_REF = object()   # sentinel: "derive the ref axis from ctx.yaw[i]" (None is a VALID ref axis = round object)


def grasp_quat_at(ctx: GraspContext, i, cx, cy, tilt_deg=0.0, *, ref_axis_world=_NO_REF):
    """The orientation-aware grasp quat for env i with the cube at (cx,cy) -- reuses the LOCKED grasp
    builders (yaw-folded for the cube, tilt base toward the cube). Used both for the first grasp and to
    RE-PLAN the recovery grasp at the cube's NEW (shoved) location.

    ``tilt_deg`` (default 0 = pure top-down) tilts the approach axis AWAY from straight-down TOWARD the
    reach direction (cube - arm base), EXACTLY as RoboLab's ``reachable_grasp_quat`` relax-tilt does. A
    small forward tilt keeps the WRIST off its limit + the ELBOW bent through the lift in this LOW (table-
    at-base-level) workspace where a pure top-down lift is near-singular. The per-env tilt is chosen by
    ``select_grasp_tilt`` below (prefer top-down; relax to the smallest tilt that stays wrist-comfortable).

    ``ref_axis_world`` (default sentinel _NO_REF): the OBJECT's grasp reference axis ALREADY expressed in the
    WORLD frame, used INSTEAD of the one derived from ``ctx.yaw[i]``. The grasp-retry's phase-2 re-grasp passes
    the axis built from the object's RE-READ quaternion here, so a noisy graze that ROTATED the object is
    RESPECTED (the gripper yaw tracks the re-read orientation, not the stale stored yaw). Pass ``None`` for a
    round object (no preferred axis -> free roll, snapped to home below). The DEFAULT sentinel keeps the
    clean/first-attempt path byte-identical (it derives the axis from the stored yaw exactly as before)."""
    s = "left" if ctx.side_is_left[i] else "right"
    base_q = tilted_base_quat(np.array([cx, cy]) - ctx.base[s], float(tilt_deg))
    if ref_axis_world is not _NO_REF:
        # RE-READ ORIENTATION (grasp-retry phase 2): the caller supplies the world reference axis straight from
        # the object's live quaternion, so the grasp closes ACROSS the object's ACTUAL (possibly rotated) axis.
        gq = orientation_aware_grasp_quat(ref_axis_world, base_q, reference_R=ctx.htR[s])
        laxis = ref_axis_world
    else:
        # Fold the in-plane yaw into the object's SYMMETRY wedge before building the reference axis. The CUBE is
        # 4-fold symmetric (a face repeats every 90deg) -> fold to [-45,45]. An ELONGATED object (banana/pen) is
        # only 2-fold symmetric about its LONG axis (the line repeats every 180deg) -> fold to [-90,90]; folding
        # it into the cube's 45deg wedge computed the grasp for the WRONG axis and the claws MISSED the banana at
        # |yaw|>45 (verified: collector banana failures were exactly the large-|yaw| envs). A ROUND object has no
        # axis (laxis is None) so the fold is irrelevant. fold = pi/2 (cube) or pi (elongated).
        fold = (np.pi / 2) if ctx.spec.is_cube else np.pi
        yr = ((ctx.yaw[i] + fold / 2) % fold) - fold / 2
        qzr = np.array([np.cos(yr / 2), 0.0, 0.0, np.sin(yr / 2)])
        laxis = ctx.laxis
        gq = orientation_aware_grasp_quat(world_long_axis(laxis, qzr), base_q, reference_R=ctx.htR[s])
    if laxis is None:
        # ROUND object (no preferred grasp axis): the wrist ROLL is FREE, so orientation_aware_grasp_quat
        # returned base_q WITHOUT aligning the roll. Left free, the roll varies with the reach direction and
        # can land ~pi from the HOME wrist roll -> the carry inherits it and the go_home then FLIPS joint_6
        # (a ~3rad snap on the empty return, verified on 3/12 apple demos). Snap the round grasp roll to the
        # branch CLOSEST to the home wrist orientation (same branch-pick transport_quats uses), so the whole
        # pick->carry->home chain stays near the home roll and joint_6 never flips. (No-op for cube/elongated,
        # whose roll is already determined by their ref axis -> their motion is unchanged.)
        gq = transport_quats(gq, reference_quat=_wxyz_from_R(ctx.htR[s]))[0]
    return gq


def cquat(ctx: GraspContext, i, gqi, tilt_deg=8.0):
    """Carry/place orientation for env i: a base tilted ``tilt_deg`` toward the BOWL reach direction,
    re-yawed (transport_quats) to the branch closest to the grasp orientation ``gqi`` (smooth wrist).
    ``tilt_deg`` defaults to 8 (the locked gentle tilt); ``select_place_tilt`` relaxes it per-env when a
    higher carry-over-bowl pose would otherwise SATURATE the wrist (RoboLab reachable_place_quat policy,
    extended with the wrist margin -- the carry over the bowl is the OTHER frame that over-stretches)."""
    s = "left" if ctx.side_is_left[i] else "right"
    return transport_quats(tilted_base_quat(np.array([ctx.bowx[i], ctx.bowy[i]]) - ctx.base[s], float(tilt_deg)),
                           reference_quat=gqi)[0]


def _posture_at_pick(ctx: GraspContext, solve, tilt_deg):
    """Batched: for the given per-env grasp tilt, return (grasp_quat, worst |j4| wrist, worst-low j3 elbow)
    over the PICK's binding frames -- the PRE-GRASP (a reached-out approach) and the LIFT -- on the
    EXECUTION-FAITHFUL warm-start chain. The executor reaches the lift by tracking CONTINUOUSLY from the
    grasp/close config UP the +Z column (warm-started single-sample IK), so the lift lands on the branch
    C1-continuous from the grasp -- NOT the globally-best branch a cold solve from home would pick. We
    replicate that (pre-grasp warm from home as the run starts, then grasp, then lift warm from grasp), so
    the predicted j4/j3 match the demo (a cold solve under-predicts the over-stretch + stops relaxing early).
    A forward tilt BOTH unsaturates the wrist AND bends the elbow, so one relax ladder satisfies both."""
    N, gc, APP, LIFT = ctx.N, ctx.gc, ctx.APP, ctx.LIFT
    gq = np.stack([grasp_quat_at(ctx, i, gc[i, 0], gc[i, 1], tilt_deg[i]) for i in range(N)]).astype(np.float64)
    tz = np.stack([_R_from_wxyz(q) @ np.array([0, 0, 1.0]) for q in gq])   # per-env approach axis (ee +Z)
    pre_tool = gc - APP * tz                           # the PRE-GRASP waypoint (back off along approach axis)
    lift_tool = gc.copy(); lift_tool[:, 2] += LIFT     # the lift waypoint the collector actually commands
    qpre = solve(pre_tool, gq)                         # PRE-GRASP solve (warm from home, as the run starts)
    solve(gc.copy(), gq)                               # GRASP solve -> warms the IK at the grasp config
    qlift = solve(lift_tool, gq)                       # LIFT solve, continuous from the grasp branch
    j4 = np.maximum(np.abs(qpre[:, 3]), np.abs(qlift[:, 3]))   # worst wrist over the pick
    j3 = np.minimum(qpre[:, 2], qlift[:, 2])                   # worst-low elbow over the pick
    return gq, j4, j3


def select_grasp_tilt(ctx: GraspContext, solve):
    """Per-env grasp tilt: the SMALLEST of ``_GRASP_TILT_STEPS_DEG`` whose pick keeps the wrist OFF its limit
    (|j4| <= WRIST_LIMIT-WRIST_MARGIN) AND the elbow BENT (j3 >= ELBOW_MIN) through the pre-grasp + lift.
    Returns (tilt_deg:(N,), grasp_quat:(N,4)). Starts every env at top-down (0) and only relaxes the envs
    that still over-stretch -- so a comfortable env keeps the cleanest pure-vertical grasp, and only the
    over-stretched envs tilt forward (exactly RoboLab's minimal-tilt policy)."""
    N = ctx.N
    WRIST_LIMIT, WRIST_MARGIN, ELBOW_MIN, LIFT = ctx.WRIST_LIMIT, ctx.WRIST_MARGIN, ctx.ELBOW_MIN, ctx.LIFT
    tilt = np.zeros(N)
    gq, j4, j3 = _posture_at_pick(ctx, solve, tilt)
    need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)   # wrist saturating OR elbow too straight
    for t in ctx.tilt_steps[1:]:
        if not need.any():
            break
        tilt[need] = t
        gq_t, j4_t, j3_t = _posture_at_pick(ctx, solve, tilt)
        gq[need] = gq_t[need]                           # adopt the relaxed-tilt quat for the still-needy envs
        j4[need] = j4_t[need]; j3[need] = j3_t[need]
        need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)   # re-test; the rest are already OK
    print(f"[COLLECT] grasp tilt (RoboLab relax, wrist-margin {WRIST_MARGIN:.2f}, elbow>={ELBOW_MIN:.2f}, "
          f"lift={LIFT}): per-env deg={np.round(tilt, 0).astype(int).tolist()} ; "
          f"predicted pick wrist|j4|max={float(j4.max()):.3f} elbow|j3|min={float(j3.min()):.3f} "
          f"(still-over-stretched={int(need.sum())})", flush=True)
    return tilt, gq.astype(np.float64)


def _carry_posture(ctx: GraspContext, solve, gqA, tilt):
    """Batched carry-over-bowl wrist + elbow for a per-env carry tilt, on the execution-faithful warm chain
    (warm at the in-bowl config, then the over-bowl hover -- the executor reaches the hover via the carry)."""
    N, bxyz, PAPP = ctx.N, ctx.bxyz, ctx.PAPP
    over_bowl = bxyz.copy(); over_bowl[:, 2] += PAPP        # the carry hover the collector commands
    cq = np.stack([cquat(ctx, i, gqA[i], tilt[i]) for i in range(N)]).astype(np.float64)
    solve(bxyz.copy(), cq)                                  # warm the IK at the lowered-in-bowl config first
    q = solve(over_bowl, cq)
    return cq, np.abs(q[:, 3]), q[:, 2]                     # carry quat, |wrist j4|, elbow j3


def select_place_tilt(ctx: GraspContext, solve, gqA):
    N = ctx.N
    WRIST_LIMIT, WRIST_MARGIN, ELBOW_MIN = ctx.WRIST_LIMIT, ctx.WRIST_MARGIN, ctx.ELBOW_MIN
    tilt = np.full(N, 8.0)                                  # the locked gentle carry tilt (top-down-ish)
    cq, j4, j3 = _carry_posture(ctx, solve, gqA, tilt)
    need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)
    for t in (16.0, 24.0, 32.0, 40.0):                     # relax toward the bowl (prefer the small tilt)
        if not need.any():
            break
        tilt[need] = t
        cq_t, j4_t, j3_t = _carry_posture(ctx, solve, gqA, tilt)
        cq[need] = cq_t[need]; j4[need] = j4_t[need]; j3[need] = j3_t[need]
        need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)
    print(f"[COLLECT] place tilt (RoboLab relax, wrist-margin {WRIST_MARGIN:.2f}, elbow>={ELBOW_MIN:.2f}): "
          f"per-env deg={np.round(tilt, 0).astype(int).tolist()} ; "
          f"predicted carry wrist|j4|max={float(j4.max()):.3f} elbow|j3|min={float(j3.min()):.3f} "
          f"(still-over-stretched={int(need.sum())})", flush=True)
    return tilt


def _posture_at_pick_env(ctx: GraspContext, solve, env_i, gci, tilt_deg, apex_z=None, *, ref_axis_world=_NO_REF):
    """Single-env (gq, |j4|, j3) over the PICK binding frames at grasp centre ``gci`` and the given tilt -- the
    env_i row of ``_posture_at_pick``, so the SCORING is byte-identical. ``apex_z`` (the recovery rise apex
    height): the recovery/chase re-grasp does rise->REORIENT-at-apex->descend->close->lift, so the high apex
    REORIENT (top-down at high EEz) is ALSO a binding frame that can saturate the wrist (the same high-EE
    saturation the lift fix addressed). When given, the apex pose at ``gqi`` is folded into the worst-case so
    the relax ladder picks a tilt comfortable AT THE APEX too -- otherwise the probe (pre+lift only) approves
    a tilt that the recovery's apex reorient still saturates (the 2026-06-20 recovered-demo j4=1.57 bug).
    ``ref_axis_world`` (grasp-retry phase 2) re-plans the grasp orientation from the object's RE-READ axis so
    the tilt is scored for the SAME wrist posture the recovery actually commands (re-read orientation respected)."""
    N, APP, LIFT = ctx.N, ctx.APP, ctx.LIFT
    gq = grasp_quat_at(ctx, env_i, gci[0], gci[1], float(tilt_deg), ref_axis_world=ref_axis_world).astype(np.float64)
    tz = _R_from_wxyz(gq) @ np.array([0, 0, 1.0])
    gcN = np.tile(np.asarray(gci, float), (N, 1))          # broadcast to the batched solve (we read row env_i)
    gqN = np.tile(gq, (N, 1))
    pre = gcN.copy(); pre -= APP * tz
    lift = gcN.copy(); lift[:, 2] += LIFT
    qpre = solve(pre, gqN); solve(gcN.copy(), gqN); qlift = solve(lift, gqN)   # warm pre->grasp->lift chain
    j4 = max(abs(float(qpre[env_i, 3])), abs(float(qlift[env_i, 3])))
    j3 = min(float(qpre[env_i, 2]), float(qlift[env_i, 2]))
    if apex_z is not None:                                  # the recovery rise+reorient apex (high EE) frame
        # the recovery does reorient@apex -> descend(pre) -> at(gci). The wrist trajectory along the steep
        # descent is NON-monotonic and can PEAK between the (checked) apex/pre endpoints, so we also probe a
        # couple of DESCENT samples (apex, 2/3-down, pre) at gq -- the worst of these binds the relax ladder.
        apexN = gcN.copy(); apexN[:, 2] = float(apex_z)
        mid = 0.5 * (apexN + pre)                           # apex->pre descent midpoint (the real recovery path)
        for wp_ in (apexN, mid, pre):                       # warm chain apex -> mid -> pre (the real descent)
            qd = solve(wp_, gqN)
            j4 = max(j4, abs(float(qd[env_i, 3])))
            j3 = min(j3, float(qd[env_i, 2]))
    return gq, j4, j3


def select_grasp_tilt_at(ctx: GraspContext, solve, env_i, gci, apex_z=None, *, ref_axis_world=_NO_REF):
    """Smallest of _GRASP_TILT_STEPS_DEG keeping |j4|<=limit-margin AND j3>=ELBOW_MIN through the pick at the
    SHOVED grasp centre ``gci`` -- the per-env relax ladder of ``select_grasp_tilt`` (identical thresholds).
    ``apex_z`` folds the recovery/chase rise-apex reorient into the worst-case (see _posture_at_pick_env).
    ``ref_axis_world`` re-plans the grasp from the object's RE-READ axis so the tilt matches the recovery pose."""
    WRIST_LIMIT, WRIST_MARGIN, ELBOW_MIN = ctx.WRIST_LIMIT, ctx.WRIST_MARGIN, ctx.ELBOW_MIN
    for t in ctx.tilt_steps:
        _, j4, j3 = _posture_at_pick_env(ctx, solve, env_i, gci, t, apex_z=apex_z, ref_axis_world=ref_axis_world)
        if (j4 <= (WRIST_LIMIT - WRIST_MARGIN)) and (j3 >= ELBOW_MIN):
            return float(t)
    return float(ctx.tilt_steps[-1])                        # none fully clears -> the largest (most-relaxed) tilt


def select_place_tilt_at(ctx: GraspContext, solve, env_i, gqi, lift_pose=None):
    """Smallest carry tilt keeping the carry-over-bowl wrist/elbow comfortable at the re-grasp orientation
    ``gqi`` (the carry quat re-yaws to gqi's branch) -- the per-env ladder of ``select_place_tilt``.

    ``lift_pose`` (the recovery lift the carry STARTS from): the wrist RE-YAWS from the grasp branch ``gqi`` to
    the carry branch ``cqi`` as the arm SWINGS lift->over-bowl, and that swing is NON-monotonic -- it can PEAK
    through the wrist limit BETWEEN the (checked) lift/over-bowl endpoints (the recovered-demo carry j4=1.57 spike
    mid-swing). When given, the lift pose at ``cqi`` AND the lift->over-bowl midpoint are folded into the worst-
    case so the relax ladder picks a carry tilt comfortable ACROSS the whole swing, not just at its ends."""
    N, bxyz, PAPP = ctx.N, ctx.bxyz, ctx.PAPP
    WRIST_LIMIT, WRIST_MARGIN, ELBOW_MIN = ctx.WRIST_LIMIT, ctx.WRIST_MARGIN, ctx.ELBOW_MIN
    over_bowl = bxyz.copy(); over_bowl[:, 2] += PAPP
    for t in (8.0, 16.0, 24.0, 32.0, 40.0):
        cqi = cquat(ctx, env_i, gqi, float(t))
        cqN = np.tile(cqi, (N, 1)).astype(np.float64)
        solve(bxyz.copy(), cqN); q = solve(over_bowl, cqN)
        j4 = abs(float(q[env_i, 3])); j3 = float(q[env_i, 2])
        if lift_pose is not None:
            # warm chain lift -> mid -> over-bowl at the carry quat (the real recovery carry swing). The wrist
            # re-yaw from the grasp branch peaks mid-swing, so probe the lift + the lift->over-bowl midpoint too.
            lp = np.tile(np.asarray(lift_pose, float), (N, 1))
            mid = 0.5 * (lp + over_bowl)
            for wp_ in (lp, mid, over_bowl):
                qd = solve(wp_, cqN)
                j4 = max(j4, abs(float(qd[env_i, 3])))
                j3 = min(j3, float(qd[env_i, 2]))
        if (j4 <= (WRIST_LIMIT - WRIST_MARGIN)) and (j3 >= ELBOW_MIN):
            return float(t)
    return 40.0
