#!/usr/bin/env python3
"""Cube->bowl PICK-PLACE task collector — a THIN task on top of the reusable ManipulationStage.

The whole robot + cameras + photoreal rendering + per-env immersive HDRI background DR lives in
`scenes/manipulation_stage.py` and is shared by every task. THIS file is only the TASK:
  - objects      : a coloured cube + a convex-decomposition bowl (RoboLab-faithful collision -> a cube on the
                   rim rolls in/out, never tunnels; cube spawned CLEAR of the bowl).
  - physics DR   : which ARM (L/R) · object-table height +/-5cm · bowl xy · cube xy + full in-plane YAW ·
                   cube mass · friction (per env, independent).
  - skill        : orientation-aware grasp + gentle top-down place (reused RoboLab skills + SODA IK).
  - output       : the §4 HDF5 (14-D actions/states/ee_pose) + the 3 policy-cam videos per demo, plus a
                   sqrt(N) third-person tile + 10 four-view tiles.

All N trials run in ONE fully-parallel build (stage.build()), each in its own random real room.

  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/tasks/pickplace.py <N> [seed]
  DATA_DIR (/data3/genesis_fulldr), OUT_DIR (output/temp/fulldr_collect), SPP (32).
"""
import math
import os
import sys
import time
import numpy as np
import genesis as gs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                      # genesis_firefly/
from world.manipulation_stage import ManipulationStage, np_   # the reusable setup  # noqa: E402
from world.firefly_scene import BOWL_HALF_H  # noqa: E402
from robots.firefly_dual import GR100_OPEN, GR100_CLOSE, GR100_MIMIC, GR100_MEET  # noqa: E402
from robots.ik import TOOL_IN_EE_INV, tool_R_at_home  # noqa: E402
from skills.grasp import (world_long_axis, orientation_aware_grasp_quat, tilted_base_quat,
                          transport_quats, _R_from_wxyz, _wxyz_from_R)  # noqa: E402
from skills.executor import BatchExecutor  # the ONE smooth motion path (densify + batch IK)  # noqa: E402
from skills.penetration import PenetrationTracker, ABNORMAL_THRESH_M  # the #1 collision gate  # noqa: E402
from skills.disturbance import DisturbanceSpec  # gentle target-shove harness -> failure-recovery data  # noqa: E402
from registry.object_spec import REGISTRY  # noqa: E402
from world.firefly_scene import OBJECTS, _rho_for  # asset root + spec->density helper  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import cv2  # noqa: E402
import h5py  # noqa: E402

REC_EVERY = 10                                                 # record state + render every K sim steps
BOWL_OBJ = os.path.join(os.path.dirname(_HERE), "assets/objects/ycb/bowl_clean.obj")  # Nyx-safe bowl visual

# ---- CONFIGURABLE TARGET object (the thing that gets picked & placed) -----------------------------------------
# The SAME task picks ANY registry object as the grasp target. ``TARGET`` (env var) selects it; default ``cube``
# so ``pickplace.py N seed`` reproduces the cube collection EXACTLY (the regression gate). Switching the target
# is a small per-object ADAPTATION, not a fork: the orientation-aware grasp (ref-axis: long-axis for elongated,
# a face-pair for the cube, None for round), the full DR, the 50/50 distractors, the disturbance harness, the
# penetration gate and go-home are ALL reused unchanged. The only target-specific pieces are (1) which spec we
# spawn, (2) its ref-axis (already encoded by the spec's elongated/is_cube flags), (3) the spawn-clear floor
# (derived from the target's footprint so a banana/book never spawns half-in the bowl), and (4) the distractor
# pool, which EXCLUDES the target's type so the target is never ambiguous among lookalikes.
TARGET = os.environ.get("TARGET", "cube")


# ---- DR-STRATEGIST sweep hooks (docs/agents.md DR strategist + dr/sweep.py) -----------------------------------
# The DR strategist EXPLORES wider/narrower ranges WITHOUT editing this file: it sets a few env-var multipliers
# that scale the per-env range HALF-WIDTHS in sample_phys_dr. Defaults are EXACTLY 1.0, so an unset environment
# reproduces the v2 collection byte-for-byte (the verify gate proves this). The multipliers scale ONLY the
# half-width of each band about its FIXED centre — they never move a centre, never touch the anti-coupling sgn/
# arm-side logic, and never relax the cube<->bowl clearance floor (so a wider pose can never spawn cube-in-bowl).
#   DR_POSE_SCALE  -> bowl xy + cube xy half-widths (pose breadth; the main axis to push to MAX extent)
#   DR_MASS_SCALE  -> cube mass-shift half-width
#   DR_FRIC_SCALE  -> robot link friction-ratio half-width
# This is the MINIMAL hook the sweep tool needs to test a candidate range to the edge of usability before the
# main agent commits an edit to the (still hand-written) ranges above. A future fully-declarative dr/ config
# would replace these in-line scales; for the MVP this keeps the locked collector untouched and reproducible.
def _dr_scale(name):
    try:
        return float(os.environ.get(name, "1.0"))
    except (TypeError, ValueError):
        return 1.0


def sample_phys_dr(N, rng, lay, spec):
    """Per-env physics DR (no visuals — the stage owns the environment/background DR)."""
    ho = lay.object_table_height
    ps = _dr_scale("DR_POSE_SCALE")                            # pose half-width multiplier (1.0 = v2 default)
    side_is_left = rng.rand(N) < 0.5
    sgn = np.where(side_is_left, 1.0, -1.0)
    tabZ = ho + (rng.rand(N) - 0.5) * 0.10                     # object-table height +/-5cm
    bowx = 0.40 + (rng.rand(N) - 0.5) * 0.10 * ps
    bowy = sgn * (0.05 + (rng.rand(N) - 0.5) * 0.07 * ps)
    cubx = 0.40 + (rng.rand(N) - 0.5) * 0.12 * ps
    cuby = sgn * (0.185 + (rng.rand(N) - 0.5) * 0.10 * ps)
    # keep the TARGET CLEAR of the bowl so the open gripper doesn't bump the bowl on the grasp descent. The
    # MAJORITY (~80%) get a generous clearance; a MINORITY (~20%) are allowed close (hard edge cases -> useful
    # recovery data, per the DR "accept hard edge cases" rule). The hard floor forbids target-in-bowl.
    # HARD floor: the cube uses the LOCKED 0.125 (cube circumradius ~3.5cm + bowl radius ~7.5cm ~= 11cm, so
    # 12.5cm centre-to-centre keeps the cube body fully OUTSIDE the bowl wall). A BIGGER target (banana/book,
    # footprint radius up to ~11cm) needs MORE clearance or its body would still overlap the bowl wall, so for a
    # non-cube target the floor SCALES with the target footprint: footprint_radius + bowl_radius(0.075) + 1.5cm
    # margin. (For the cube this formula gives ~0.1254 ~= the locked 0.125, so cube stays byte-for-byte on the
    # literal.) The previous loop could EXHAUST its tries on an unlucky/infeasible draw and SILENTLY ship a still-
    # overlapping target -> the firm solver ejected it off the (ground-plane-less) table -> a 6m garbage waypoint
    # -> the 10x trajectory blowup (2026-06-19, docs/roadmap.md). FIX: after rejection sampling, CLAMP any still-
    # bad env's target radially outward from the bowl to exactly the hard floor (a guaranteed-clear, on-table,
    # correct-side pose) so the target can NEVER spawn intersecting the bowl, regardless of luck/feasibility.
    CLR_HARD = 0.125 if spec.is_cube else float(_footprint_radius(spec) + 0.075 + 0.015)
    clr = np.where(rng.rand(N) < 0.8, max(0.17, CLR_HARD + 0.045), CLR_HARD)
    for _ in range(60):
        bad = np.hypot(cubx - bowx, cuby - bowy) < clr
        if not bad.any():
            break
        nb = int(bad.sum())
        cubx[bad] = 0.40 + (rng.rand(nb) - 0.5) * 0.12 * ps
        cuby[bad] = sgn[bad] * (0.185 + (rng.rand(nb) - 0.5) * 0.10 * ps)
    # GUARANTEED fallback: push any env STILL inside the hard floor radially out to exactly CLR_HARD. Direction
    # = bowl->cube (away from the bowl); if the cube sits exactly on the bowl centre, push along the arm side
    # (+sgn y) so it stays on the object table and on the correct half. This makes a cube-in-bowl spawn -- and
    # thus the off-table ejection -- impossible by construction.
    bad = np.hypot(cubx - bowx, cuby - bowy) < CLR_HARD
    if bad.any():
        dx, dy = cubx[bad] - bowx[bad], cuby[bad] - bowy[bad]
        dn = np.hypot(dx, dy)
        ux = np.where(dn > 1e-6, dx / np.maximum(dn, 1e-6), 0.0)
        uy = np.where(dn > 1e-6, dy / np.maximum(dn, 1e-6), sgn[bad])   # degenerate: push along the arm side
        cubx[bad] = bowx[bad] + ux * CLR_HARD
        cuby[bad] = bowy[bad] + uy * CLR_HARD
    yaw = (rng.rand(N) - 0.5) * np.radians(180)
    ms = _dr_scale("DR_MASS_SCALE")                            # cube mass-shift half-width multiplier
    mass_shift = ((rng.rand(N, 1) - 0.5) * 0.04 * ms).astype(np.float32)
    return dict(side_is_left=side_is_left, sgn=sgn, tabZ=tabZ, bowx=bowx, bowy=bowy,
                cubx=cubx, cuby=cuby, yaw=yaw, mass_shift=mass_shift)


# ============================================================================ #
# DISTRACTOR / CLUTTER OBJECTS  (REQUIRED DR feature, docs/domain_randomization.md scope B)
# ============================================================================ #
# Pool of REALISTIC irrelevant objects. They render from their own assets/spec colours (apple red/green, banana
# yellow, pen, tennis ball yellow-green, book dark-red) and are NEVER the grasp target. The TARGET's own type is
# EXCLUDED from the pool (per-build, in choose_distractor_types) so the target is never ambiguous among
# lookalikes -- e.g. when the target is the banana, the clutter is drawn from {pen, apple, tennis_ball, book, cube}.
DISTRACTOR_UNIVERSE = ["pen", "banana", "apple", "tennis_ball", "book", "cube"]
# back-compat: the cube-target collection's pool (the cube is the target, so it is not a distractor) -- this is
# exactly DISTRACTOR_UNIVERSE minus "cube", which is what choose_distractor_types(target="cube") produces, so the
# cube regression draws the SAME clutter types as before.
DISTRACTOR_POOL = ["pen", "banana", "apple", "tennis_ball", "book"]

# Geometry of the active arm's swept XY corridor (the planner is pure waypoint-IK with NO obstacle avoidance, so
# collision-freeness is achieved by PLACEMENT: distractors are rejection-sampled OUT of everything the arm
# sweeps). All radii are half-clearances in metres about a point/segment; tuned on the verify gate so the max
# distractor displacement stays well under 2 cm.
_HOME_EE_XY = np.array([0.267, 0.224])     # active-arm EE at home (measured); y is mirrored by sgn
# Corridor half-clearances (metres) to the distractor CENTRE about each swept feature; the sampler ADDS each
# object's footprint radius so the whole BODY clears. They also cover the GRIPPER's physical extent (fingers +
# wrist reach ~5-8cm beyond the EE/object centreline), so a long banana's far tip is never clipped. Tuned so
# the arm never swipes a distractor (verified by the sim displacement gate: max distractor XY move < 2cm) while
# leaving enough open table to place 2-3 objects out of the corridor.
CLR_CUBE = 0.15     # around the cube: top-down grasp descent + open-gripper finger span + wrist width + margin
CLR_BOWL = 0.15     # around the bowl: the release/lower + the gripper hovering above it
CLR_CARRY = 0.15    # half-width of the cube->bowl transport tube (held object + gripper + wrist sweep)
CLR_HOME = 0.14     # half-width of the bowl->home diagonal return tube (a long banana's far end can clip it)
CLR_ARMHOME = 0.12  # keepout disk around EACH arm home (0.267, +/-0.224): the active arm transits its home and
#                     the idle arm hovers at the other; a long distractor parked just past a home gets nudged.
CLR_DIST = 0.03     # SURFACE gap between two distractors' footprint disks (added to BOTH footprint radii ->
#                     centre-to-centre >= r_i + r_j + gap, so two bodies never overlap/stack at spawn)
TABLE_MARGIN = 0.06  # inset from every table edge so a distractor never spawns half-off / on the rim
SEAM_KEEPOUT_X = 0.30   # forbid the near-seam strip x < this (the arm links + base sweep it on every move)


# the two LONG/large objects (banana ~18cm, book ~18cm): with the arm corridor removed, the open table area is
# one ~0.3x0.45m region. TWO of these need ~0.18m spacing and don't reliably both fit there, so a build carries
# AT MOST ONE large object (the rest small: apple/pen/tennis ball). This keeps every trial physically placeable
# with real spacing (no stacking -> the firm solver never NaNs on a spawn interpenetration).
DISTRACTOR_LARGE = {"banana", "book"}


def choose_distractor_types(rng, target="cube", pool=None):
    """PER-BUILD: how many distractors (K in {2,3}) and which TYPES (entities are created before scene.build,
    so the count + identities are fixed for the whole build). Drawn WITHOUT replacement so the K distractors
    are visually distinct lookalikes (the policy must disambiguate the TARGET from a varied clutter set). At most
    ONE large/long object (banana/book) per build so every set fits on the table out-of-corridor with real
    spacing (the open area can't hold two 18cm objects clear of the arm path).

    The TARGET's own type is EXCLUDED from the pool so the target is never ambiguous (e.g. a banana target draws
    clutter from {pen, apple, tennis_ball, book, cube}). For ``target="cube"`` the pool is exactly the legacy
    ``DISTRACTOR_POOL`` (universe minus cube, same items/order), so the cube collection draws IDENTICAL clutter."""
    if pool is None:
        pool = [n for n in DISTRACTOR_UNIVERSE if n != target]
    k = int(rng.choice([2, 3]))
    for _ in range(40):
        names = list(rng.choice(pool, size=k, replace=False))
        if sum(n in DISTRACTOR_LARGE for n in names) <= 1:
            return names
    return names


def _seg_clearance(px, py, ax, ay, bx, by):
    """Per-env XY distance from points (px,py) to the segment a->b (a,b are per-env arrays). Vectorised."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby + 1e-12
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy)


def _footprint_radius(spec):
    """Half the XY diagonal of an object's AABB = the radius of the disk that contains the object at ANY yaw.
    Insetting the table bounds + every clearance by this guarantees the WHOLE body (not just its centre) stays
    on the table and out of the corridor, regardless of the random in-plane spin. Used for edge + corridor
    clearance (safety-critical, and there is room there)."""
    e = spec.scaled_extents()
    return 0.5 * float(np.hypot(e[0], e[1]))


def _space_radius(spec):
    """A TIGHTER radius for inter-distractor SPACING only = half the object's longer horizontal extent (its
    enclosing-square half-side), not the diagonal. Two flat-resting bodies whose enclosing squares are
    separated by CLR_DIST never deeply interpenetrate (the firm solver only NaNs on a DEEP spawn overlap),
    and this makes packing 3 long objects (banana/book) onto the table FEASIBLE where the circumscribed radius
    would not. Edge/corridor still use the conservative circumscribed radius."""
    e = spec.scaled_extents()
    return 0.5 * float(max(e[0], e[1]))


def _corridor_clear(cx, cy, rfoot, sgn, cubx, cuby, bowx, bowy, homx, homy):
    """Bool mask: is footprint (cx,cy,rfoot) clear of the ARM corridor? The corridor = the near-seam strip + the
    cube + the bowl + the cube->bowl carry tube + the bowl->ACTIVE-home return tube + a keepout around BOTH arm
    homes (the active arm starts/ends there; the inactive arm holds there all episode, and either elbow can swing
    over the near-home region). Every threshold ADDS rfoot so the whole BODY (not just the centre) clears."""
    ok = (cx - rfoot >= SEAM_KEEPOUT_X)
    ok &= np.hypot(cx - cubx, cy - cuby) >= CLR_CUBE + rfoot
    ok &= np.hypot(cx - bowx, cy - bowy) >= CLR_BOWL + rfoot
    ok &= _seg_clearance(cx, cy, cubx, cuby, bowx, bowy) >= CLR_CARRY + rfoot
    ok &= _seg_clearance(cx, cy, bowx, bowy, homx, homy) >= CLR_HOME + rfoot
    # the ACTIVE arm transits its own home (homx,homy) on the way out/back -> a small keepout disk there. (The
    # idle arm hovers at the OTHER home but at z>=0.43, well above the table, so it can't touch a resting object.)
    ok &= np.hypot(cx - homx, cy - homy) >= CLR_ARMHOME + rfoot
    return ok


def _anchor_grid(lay, rmax):
    """A fine grid of candidate anchor CENTRES over the object table (~3.5cm pitch), inset by rmax so any object
    centred on a cell stays fully on the table at any yaw. The placement greedily picks K mutually-spaced,
    corridor-clear cells from this grid (one per distractor) -> non-overlap is GUARANTEED by construction."""
    x0 = lay.seam_x + TABLE_MARGIN + rmax
    x1 = lay.seam_x + lay.object_table_depth - TABLE_MARGIN - rmax
    y1 = lay.common_width / 2 - TABLE_MARGIN - rmax
    xs = np.arange(x0, x1 + 1e-6, 0.035)
    ys = np.arange(-y1, y1 + 1e-6, 0.035)
    if xs.size == 0:
        xs = np.array([(x0 + x1) / 2])
    if ys.size == 0:
        ys = np.array([0.0])
    return np.array([(x, y) for x in xs for y in ys])


def sample_distractor_poses(N, specs, dr, rng, lay):
    """PER-ENV: choose a clear XY + yaw for each distractor. GUARANTEES: (a) the whole BODY stays on the table
    and OUT of the arm corridor (cube + bowl + cube->bowl carry tube + bowl->home return tube + near-seam strip)
    at ANY yaw (circumscribed-radius clearance), and (b) no two distractors ever deeply interpenetrate at spawn
    (enclosing-square spacing). Method = GREEDY GRID ASSIGNMENT: tile the table with a fine anchor grid, keep
    the corridor-clear cells (favouring the opposite-y side from the active arm for variety), then greedily pick
    K cells that are pairwise spaced >= rs_i+rs_j+CLR_DIST. Because each distractor gets its OWN well-separated
    grid cell, nothing stacks (the firm solver NaNs on a deep spawn overlap) and nothing sits in the arm's path
    (so it is never swiped). A tiny in-cell jitter (< half the leftover slack) keeps the variety without breaking
    the spacing. Returns xy:(K,N,2), yaw:(K,N)."""
    sgn = dr["sgn"]
    cubx, cuby, bowx, bowy = dr["cubx"], dr["cuby"], dr["bowx"], dr["bowy"]
    homx = np.full(N, _HOME_EE_XY[0]); homy = sgn * _HOME_EE_XY[1]
    K = len(specs)
    rfoot = np.array([_footprint_radius(s) for s in specs])         # circumscribed: edge + corridor (safe@any yaw)
    rspace = np.array([_space_radius(s) for s in specs])            # enclosing-square: inter-object spacing
    rmax = float(rfoot.max())
    grid = _anchor_grid(lay, rmax)
    M = grid.shape[0]
    out_xy = np.zeros((K, N, 2), np.float64)
    out_yaw = ((rng.rand(K, N) - 0.5) * np.radians(180))
    # largest distractor first -> the hardest-to-place objects claim space before the small ones
    order = list(np.argsort(-rspace))
    for e in range(N):
        # corridor-clear cells for this env (use rmax so the test is valid for every distractor's footprint)
        cc = _corridor_clear(grid[:, 0], grid[:, 1], rmax, np.full(M, sgn[e]),
                             np.full(M, cubx[e]), np.full(M, cuby[e]), np.full(M, bowx[e]),
                             np.full(M, bowy[e]), np.full(M, homx[e]), np.full(M, homy[e]))
        cand = grid[cc]
        # favour the opposite-y side (more open) then random, so clutter spreads & varies build-to-build
        if cand.shape[0]:
            opp = (np.sign(cand[:, 1]) != np.sign(sgn[e]))
            cand = cand[np.lexsort((rng.rand(cand.shape[0]), ~opp))]
        chosen_x = []; chosen_y = []; chosen_rs = []
        for oi, k in enumerate(order):
            rs = float(rspace[k]); pick = None
            for ci in range(cand.shape[0]):                         # first corridor-clear cell spaced from chosen
                cx, cy = cand[ci]
                if all(np.hypot(cx - px, cy - py) >= CLR_DIST + rs + prs
                       for px, py, prs in zip(chosen_x, chosen_y, chosen_rs)):
                    pick = (cx, cy); break
            if pick is None:                                        # no corridor-clear cell fits (very rare) ->
                # park on the FAR-x edge (far from the seam/arm -> corridor-clear by construction), offset in Y
                # by placement index so two un-placeable objects never coincide. X stays at the far edge.
                xb = lay.seam_x + lay.object_table_depth - TABLE_MARGIN - rfoot[k]
                yb = lay.common_width / 2 - TABLE_MARGIN - rfoot[k]
                step = 2.0 * (float(rspace.max()) + CLR_DIST)
                yy = float(np.clip(yb - oi * step, -yb, yb))
                pick = (xb, -sgn[e] * yy)
            out_xy[k, e] = pick
            chosen_x.append(pick[0]); chosen_y.append(pick[1]); chosen_rs.append(rs)
    return out_xy, out_yaw


def _spawn_distractor_entity(scene, spec, pos_xy, z):
    """Spawn ONE distractor as a real collidable rigid body, but with a NYX-SAFE visual. cuboid/sphere are
    procedural (Nyx-safe already); USD-sourced objects (apple/banana/pen) render via their extracted clean
    .obj mesh + a realistic flat colour (Nyx SEGFAULTS on the textured USD material binding, the same reason
    the bowl uses bowl_clean.obj). Collision fidelity (convex decomposition) is identical to the grasp path."""
    x, y = pos_xy
    # distractors only need to REST (not be grasped), so use firm friction so they sit put under a graze.
    mat = gs.materials.Rigid(rho=_rho_for(spec), friction=max(1.1, float(spec.friction[0])))
    if spec.source == "cuboid":
        return scene.add_entity(gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(x, y, z)),
                                material=mat, surface=gs.surfaces.Plastic(color=spec.color, roughness=0.6))
    if spec.source == "sphere":
        return scene.add_entity(gs.morphs.Sphere(radius=float(spec.scaled_extents()[0] / 2), pos=(x, y, z)),
                                material=mat, surface=gs.surfaces.Rough(color=spec.color))
    # USD object -> Nyx-safe clean mesh + flat realistic colour. Collision = a SINGLE convex hull (no
    # decomposition): a distractor is never grasped, so it doesn't need a faithful concave collider, and a hull
    # gives a FLATTER, stable resting base -> a curved banana doesn't slowly roll/creep off a rounded decomposed
    # facet (that intrinsic creep, not the arm, was reading as a 2-3cm "knock" in the displacement metric).
    col = spec.dist_color or spec.color
    return scene.add_entity(
        gs.morphs.Mesh(file=str(OBJECTS / spec.mesh_subpath), pos=(x, y, z), scale=spec.scale, convexify=True),
        material=mat, surface=gs.surfaces.Plastic(color=col, roughness=0.5))


def spawn_target(scene, spec, color, pos_xy=(0.40, 0.18), z=0.30):
    """Spawn the GRASP TARGET as a real collidable rigid body with a FAITHFUL grasp collider + a Nyx-safe visual.
    The target is the one object that gets GRASPED, so (unlike a distractor) its USD-mesh collider is a convex
    DECOMPOSITION (coacd, the same recipe as the bowl), not a single hull, so the gripper closes on the real
    elongated/flat shape (a banana's curve, a pen's thin body, a book's flat slab) rather than a fat envelope.
    Procedural cube/sphere keep their exact box/sphere collider (already faithful). The visual is the procedural
    box/sphere for those sources, or the Nyx-safe clean .obj for USD sources (the textured USD segfaults Nyx, the
    same reason the bowl + distractors render from clean meshes). ``color`` comes from the stage's distinct-from-
    table picker so the target stays visible against the randomized table.
    Returns the entity (its per-env pose/yaw/mass DR is applied by the caller after build, exactly as the cube)."""
    x, y = pos_xy
    fr = float(os.environ.get("TGT_FRIC", spec.friction[0]))   # higher friction -> a shallower grip still holds
    mat = gs.materials.Rigid(rho=_rho_for(spec), friction=fr)
    if spec.source == "cuboid":
        # EXACT cube path (byte-for-byte for the regression): procedural box, plastic PBR, the spec's grasp friction.
        return scene.add_entity(gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(x, y, z)),
                                material=gs.materials.Rigid(rho=600.0, friction=1.0),
                                surface=gs.surfaces.Plastic(color=color, roughness=0.35))
    if spec.source == "sphere":
        return scene.add_entity(gs.morphs.Sphere(radius=float(spec.scaled_extents()[0] / 2), pos=(x, y, z)),
                                material=mat, surface=gs.surfaces.Rough(color=color))
    # USD object -> grasp collider on the clean mesh + the distinct-from-table colour.
    # COLLIDER CHOICE (spec.grasp_single_hull):
    #  * decomposition (default): coacd splits the mesh into solid hulls -> faithful concave shape (needed for a
    #    hollow/handled object). But for a ROUNDED CONVEX body (banana) the decomposition's internal hull seams +
    #    the firm high-kp pinch drive the claw DEEP into a seam -> >9mm penetration AND huge constraint forces
    #    that NaN the Newton solver (verified: dz<=-0.005 banana -> 'Invalid constraint forces' crash).
    #  * single convex hull (grasp_single_hull=True): a SMOOTH convex envelope of the body. A banana is already
    #    near-convex, so the hull is a faithful smooth banana the claws pinch CLEANLY -- stable contact, shallow
    #    penetration, no NaN. This is the SAME stable single-hull collider the distractors use, promoted to the
    #    grasp target for convex rounded objects.
    single = os.environ.get("TGT_SINGLE_HULL")
    single = (single == "1") if single is not None else bool(getattr(spec, "grasp_single_hull", False))
    if single:
        return scene.add_entity(
            gs.morphs.Mesh(file=str(OBJECTS / spec.mesh_subpath), pos=(x, y, z), scale=spec.scale, convexify=True),
            material=mat, surface=gs.surfaces.Plastic(color=color, roughness=0.5))
    decomp = float(os.environ.get("TGT_DECOMP", getattr(spec, "grasp_decompose_err", 0.04)))
    return scene.add_entity(
        gs.morphs.Mesh(file=str(OBJECTS / spec.mesh_subpath), pos=(x, y, z), scale=spec.scale,
                       convexify=True, decompose_object_error_threshold=decomp, decimate=False),
        material=mat, surface=gs.surfaces.Plastic(color=color, roughness=0.5))


def spawn_distractors(stage, dr, rng, target="cube"):
    """Create the per-build distractor ENTITIES (called before stage.build()) and return (entities, names,
    xy, yaw). TYPES are per-build (chosen here with the stage rng, EXCLUDING the target type); POSES are per-env
    (placed after build via the returned batched arrays). Each entity is a real collidable rigid body from the
    REGISTRY (realistic colour/size/mass/friction) and drops+settles with the target/bowl during the settle."""
    names = choose_distractor_types(rng, target=target)
    specs = [REGISTRY[nm] for nm in names]
    lay = stage.lay
    ents = []
    for spec in specs:
        # parked off to the far corner at build; the real per-env pose is set after build() (set_pos overrides).
        e = _spawn_distractor_entity(stage.scene, spec,
                                     pos_xy=(lay.seam_x + lay.object_table_depth - 0.05, 0.0),
                                     z=spec.rest_root_z(lay.object_table_height))
        ents.append(e)
    xy, yaw = sample_distractor_poses(stage.n_envs, specs, dr, rng, lay)
    return ents, names, xy, yaw


def _label(img, text):
    im = img.copy()
    cv2.rectangle(im, (0, 0), (len(text) * 7 + 6, 16), (0, 0, 0), -1)
    cv2.putText(im, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def tile(frames, H, W):
    N = len(frames); g = int(math.ceil(math.sqrt(N)))
    t = np.zeros((g * H, g * W, 3), np.uint8)
    for i in range(N):
        r, c = divmod(i, g); t[r * H:(r + 1) * H, c * W:(c + 1) * W] = frames[i]
    return t


def collect(N, seed, data_dir, out_dir, target=None):
    t0 = time.time()
    target = TARGET if target is None else target
    if target not in REGISTRY:
        raise SystemExit(f"unknown TARGET '{target}' (choose from {sorted(REGISTRY)})")
    stage = ManipulationStage(N, seed=seed)                   # the reusable robot+cameras+rendering+env-DR setup
    # ``spec`` is the CONFIGURABLE grasp target (default cube; TARGET env var selects any registry object). The
    # variable below stays named ``cube`` so the ~600 lines of locked grasp/score/disturbance logic that reference
    # it are unchanged -- it is the TARGET entity, which the orientation-aware grasp handles via spec.ref-axis.
    lay, spec, rng = stage.lay, REGISTRY[target], stage.rng
    print(f"[COLLECT] TARGET = {target!r} ({spec.language_name}); source={spec.source} "
          f"extents={np.round(spec.scaled_extents(), 3).tolist()} "
          f"ref_axis={'long' if spec.elongated else ('face' if spec.is_cube else 'none(round)')}", flush=True)
    dr = sample_phys_dr(N, rng, lay, spec)

    # --- task objects: the coloured TARGET + a convex-decomposition bowl, distinct from the table colour ---
    ch, cube_col = stage.distinct_object_color()
    _, bowl_col = stage.distinct_object_color(ch)
    cube = spawn_target(stage.scene, spec, cube_col, pos_xy=(0.40, 0.18), z=0.30)   # the TARGET entity
    bowl = stage.scene.add_entity(gs.morphs.Mesh(file=BOWL_OBJ, convexify=True,
                                  decompose_object_error_threshold=0.04, decimate=False),
                                  material=gs.materials.Rigid(rho=400.0, friction=1.0),
                                  surface=gs.surfaces.Smooth(color=bowl_col))

    # --- distractor / clutter objects (REQUIRED DR): 2-3 random irrelevant objects on the OBJECT table, in OPEN
    # areas, rejection-sampled OUT of the active arm's swept corridor (grasp + cube->bowl carry + bowl->home).
    # ENTITIES are created here (per-build types); their per-env POSES are applied after build() below. ---
    dist_ents, dist_names, dist_xy, dist_yaw = spawn_distractors(stage, dr, rng, target=target)
    dist_specs = [REGISTRY[nm] for nm in dist_names]
    print(f"[COLLECT] distractors (per-build, K={len(dist_names)}): {dist_names}", flush=True)

    stage.build()
    robot, cams = stage.robot, stage.cams
    H, W = stage.H, stage.W
    ee = {"l": robot.entity.get_link(robot.ee["left"]), "r": robot.entity.get_link(robot.ee["right"])}
    gdrv, gmim, armdof, state14 = robot.grip_driven, robot.grip_mimic, robot.arm, robot.state14_idx
    print(f"[COLLECT] built N={N} (one parallel build, {N} environments) in {time.time()-t0:.1f}s", flush=True)

    # ---- apply DR + settle ----
    ho = lay.object_table_height
    tabZ, bowx, bowy, cubx, cuby = dr["tabZ"], dr["bowx"], dr["bowy"], dr["cubx"], dr["cuby"]
    side_is_left, yaw = dr["side_is_left"], dr["yaw"]
    qz = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1).astype(np.float32)
    stage.otable.set_pos(np.stack([np.full(N, lay.seam_x + lay.object_table_depth / 2), np.zeros(N), tabZ - ho / 2], 1).astype(np.float32))
    stage.set_otable_top_z(tabZ)                                # keep the textured table-top flush with the height DR
    bowl.set_pos(np.stack([bowx, bowy, tabZ + BOWL_HALF_H + 0.003], 1).astype(np.float32))
    cube_top = tabZ + spec.scaled_extents()[2] / 2 + 0.002
    cube.set_pos(np.stack([cubx, cuby, cube_top + 0.01], 1).astype(np.float32)); cube.set_quat(qz)
    # 50/50 per-env (owner directive, standing for ALL tasks): ~half the trials have NO distractors (SIMPLE
    # data -> the robot learns the core task fast), ~half are cluttered. Absence = park the (fixed-count)
    # distractor entities far BELOW the scene (out of every camera + out of collision) for those envs; they
    # fall away harmlessly and never touch the workspace.
    has_dist = rng.rand(N) < 0.5
    # distractor per-env poses: drop each just above the table at its rejection-sampled XY with a random yaw
    # (in-plane spin). They settle with the cube/bowl during settle_home() below.
    for k, e in enumerate(dist_ents):
        dsp = dist_specs[k]
        drz = np.array([dsp.rest_root_z(z) for z in tabZ], np.float32) + 0.01   # per-env table-height-aware drop
        drz = np.where(has_dist, drz, -2.0).astype(np.float32)                  # hide (park below) where no clutter
        e.set_pos(np.stack([dist_xy[k, :, 0], dist_xy[k, :, 1], drz], 1).astype(np.float32))
        dy = dist_yaw[k]
        e.set_quat(np.stack([np.cos(dy / 2), 0 * dy, 0 * dy, np.sin(dy / 2)], 1).astype(np.float32))
    try:
        cube.set_mass_shift(dr["mass_shift"])
        # robot link friction-ratio band 1.0 +/- 0.3 (DR_FRIC_SCALE scales the half-width; clamp >=0 so a wide
        # sweep can't request negative friction). Default scale 1.0 -> the v2 0.7..1.3 band, byte-for-byte.
        fs = _dr_scale("DR_FRIC_SCALE")
        fric = (1.0 + (rng.rand(N, robot.entity.n_links) - 0.5) * 0.6 * fs).clip(0.0)
        robot.entity.set_friction_ratio(fric.astype(np.float32))
    except Exception as e:
        print(f"[COLLECT] mass/fric DR skipped: {e}", flush=True)
    # Lower each distractor's CoM below its geometric centre so it SELF-RIGHTS and rests stably instead of slowly
    # rolling (a curved banana on a convex base is otherwise metastable and creeps a few cm over the episode --
    # which is intrinsic, NOT an arm contact, but it inflates the displacement metric). A real banana's mass is
    # in its body, so a lowered CoM is also the more faithful model.
    for k, e in enumerate(dist_ents):
        try:
            dz = -0.4 * dist_specs[k].scaled_extents()[2]          # CoM 40% of half-height below centre
            e.set_COM_shift(np.tile([0.0, 0.0, dz], (N, 1)).astype(np.float32))
        except Exception:
            pass
    # settle longer with distractors so every object (esp. a curved banana) reaches true rest BEFORE the run ->
    # the run-time displacement then reflects ARM contact only.
    home16 = stage.settle_home(140 if dist_ents else 70)
    for e in dist_ents:                                            # start the episode at REST (objects-at-rest is
        try:                                                       # the correct initial condition)
            e.zero_all_dofs_velocity()
        except Exception:
            pass
    try:
        cube.zero_all_dofs_velocity()    # the TARGET too: a curved/round body can be metastable at settle end and
        #                                  CREEP during the approach so the grasp (planned from the settled pose)
        #                                  closes where it WAS, not where it IS -> an empty close. Starting it at
        #                                  true rest removes that creep. (The cube is already at rest, so this is
        #                                  a no-op for the regression.)
    except Exception:
        pass
    root0 = np_(cube.get_pos())
    dist_xy0 = np.stack([np_(e.get_pos())[:, :2] for e in dist_ents], 0) if dist_ents else np.zeros((0, N, 2))

    # ---- DEGENERATE-SETTLE GUARD (root-cause fix for the 2026-06-19 trajectory blowup) ----
    # If the cube spawned overlapping the bowl wall (the cube-vs-bowl rejection loop can EXHAUST its tries and
    # ship a still-overlapping pose), the firm solver EJECTS it on settle step 0; with NO ground plane in the
    # immersive scene it then FREE-FALLS off the table to z=-6m. That garbage settled pos feeds the grasp
    # waypoints (gc = cube pos), so the home->pre / lift->carry segments span metres -> densify emitted ~5000
    # steps -> the BatchExecutor padded ALL envs to ~10000 (a 10x, 41-min build + a jerky demo). We detect such
    # an env here (cube far off its intended on-table spawn, non-finite, or below the table) and (1) CLAMP its
    # cube pos back to the intended spawn XY at table height so its OWN trajectory is sane (smooth, well-
    # conditioned IK), and (2) FLAG it ``degenerate`` so the demo is REJECTED (success=False, never shipped):
    # a demo grasping at a phantom cube location is not valid training data regardless of where it lands.
    spawn_xy = np.stack([cubx, cuby], 1)                          # the cube's INTENDED on-table spawn XY
    off_xy = np.linalg.norm(root0[:, :2] - spawn_xy, axis=1)      # how far the settled cube drifted in XY
    cube_floor = tabZ - 0.05                                      # a settled cube can't be below the table top
    degenerate = (~np.isfinite(root0).all(axis=1)) | (off_xy > 0.05) | (root0[:, 2] < cube_floor) \
        | (root0[:, 2] > tabZ + 0.30)
    if degenerate.any():
        bad = np.where(degenerate)[0]
        print(f"[COLLECT] DEGENERATE settle in {len(bad)}/{N} env(s) {list(bad)} "
              f"(cube ejected off table -> demo REJECTED, waypoints clamped): "
              f"offXY={np.round(off_xy[bad], 3).tolist()} z={np.round(root0[bad, 2], 3).tolist()}", flush=True)
        # clamp the bad envs' cube pos to a SANE on-table pose so densify/IK stay well-behaved for those envs
        root0[bad, 0] = cubx[bad]
        root0[bad, 1] = cuby[bad]
        root0[bad, 2] = tabZ[bad] + spec.scaled_extents()[2] / 2 + 0.002

    # ---- per-env grasp + GENTLE top-down place plan (RoboLab skills + SODA IK) ----
    _, heqL = robot.ee_pose("left"); _, heqR = robot.ee_pose("right")
    htR = {"left": tool_R_at_home(_R_from_wxyz(np_(heqL)[0])), "right": tool_R_at_home(_R_from_wxyz(np_(heqR)[0]))}
    base = {"left": np.array([0.0, 0.224]), "right": np.array([0.0, -0.224])}
    laxis = spec.long_axis_local()

    # GRASP-CENTRE offset (object-specific): for a CURVED object (banana) the AABB centre sits in the hollow of
    # the curve, so the grasp point is shifted along the SHORT axis onto the body by spec.grasp_center_offset_local
    # (a LOCAL-frame vector). We rotate it by the object's live world yaw and add it to the root -> the world grasp
    # centre. For every other object the offset is (0,0,0), so the grasp centre IS the root (the cube path is
    # unchanged). ``grasp_center_world(root_pos, root_quat)`` maps a LIVE object pose to its grasp centre; it is
    # re-evaluated whenever the object moves (the disturbance chase/retry re-reads the shoved pose).
    _GOFF = np.asarray(spec.grasp_center_offset_local, float)
    _HAS_GOFF = bool(np.linalg.norm(_GOFF) > 1e-9)
    # grasp height nudge along world +Z from the body centre. spec.grasp_dz is the locked per-object value;
    # GRASP_DZ env overrides it for quick tuning (a rounded object like the banana grips more securely a few mm
    # BELOW its equator, where the closing claws cradle UNDER the widest cross-section instead of skidding off
    # the top). Default = the spec value, so unset reproduces the spec.
    grasp_dz = float(os.environ.get("GRASP_DZ", spec.grasp_dz))

    def grasp_center_world(root_pos, root_quat):
        """World grasp centre = root + R(root_quat) @ grasp_center_offset_local (per-env, vectorised over N)."""
        rp = np.asarray(root_pos, float)
        if not _HAS_GOFF:
            return rp.copy()
        out = rp.copy()
        rq = np.asarray(root_quat, float)
        for i in range(rp.shape[0]):
            out[i] = rp[i] + _R_from_wxyz(rq[i]) @ _GOFF
        return out

    # the GRASP centre for the settled object (root0 stays the raw root for the degenerate guard + scoring + rest_z)
    groot0 = grasp_center_world(root0, np_(cube.get_quat()))

    # NOTE: the orientation-aware GRASP quat is built by ``grasp_quat_at(i, cx, cy)`` below (used both for the
    # first grasp AND the recovery re-grasp at the cube's relocated pose); ``cquat`` builds the carry/place quat.
    def cquat(i, gqi):
        s = "left" if side_is_left[i] else "right"
        return transport_quats(tilted_base_quat(np.array([bowx[i], bowy[i]]) - base[s], 8.0), reference_quat=gqi)[0]

    def ik(link, tool_pos, tool_quat):
        Rt = np.stack([_R_from_wxyz(q) for q in tool_quat])
        ee_pos = tool_pos + np.einsum("nij,j->ni", Rt, TOOL_IN_EE_INV[:3, 3])
        ee_R = np.einsum("nij,jk->nik", Rt, TOOL_IN_EE_INV[:3, :3])
        ee_quat = np.stack([_wxyz_from_R(R) for R in ee_R])
        idx = armdof["left"] if link is ee["l"] else armdof["right"]
        # Pin the solve to the WARM-STARTED branch. Genesis defaults max_samples=50: when the warm start
        # (current qpos) fails to converge in the iter budget -- which happens during a big wrist/elbow
        # reorientation -- it RANDOM-restarts up to 50x over the full joint range and returns whatever branch
        # converges first (an elbow/wrist-FLIPPED solution). The executor then PD-drives across the branch jump
        # in one step -> the visible wrist SNAP. max_samples=1 never leaves the warm start; a smaller
        # max_step_size + a little DLS damping keep each solve C0-continuous and singularity-robust. (Verified
        # against genesis .../rigid/abd/inverse_kinematics.py: the resample block is gated on i_sample<max_samples-1.)
        q = np_(robot.entity.inverse_kinematics(
                link=link, pos=ee_pos.astype(np.float32), quat=ee_quat.astype(np.float32), dofs_idx_local=idx,
                max_solver_iters=30, max_samples=1, max_step_size=0.2, damping=0.05, return_error=False))
        return q[:, idx]

    gc = groot0.copy(); gc[:, 2] += grasp_dz   # grasp the BODY centre (offset onto the curve for a banana)
    drop_z = tabZ + 2 * BOWL_HALF_H + spec.scaled_extents()[2] / 2 + 0.012   # release ABOVE the rim (gentle)
    bxyz = np.stack([bowx, bowy, drop_z], 1).astype(np.float64)
    APP, LIFT, PAPP = 0.12, 0.18, 0.08   # match RoboLab/plan_pick_place defaults (lower lift = gentler, safe)
    # CLOSE = the driven-claw firm-pinch TARGET. GR100_CLOSE (0.9) is a hard squeeze that the OBJECT stops; for a
    # boxy object the flat face stops the claws early (~2.7mm skin). A ROUNDED body (banana) presents a narrow
    # contact, so the high-kp PD over-drives the claw deep into it before the constraint balances (>7mm gate).
    # spec.grasp_close (default GR100_CLOSE) lets a rounded/soft object use a GENTLER target so the pinch holds
    # by friction without over-penetrating; CLOSE_G env overrides for tuning. Cube keeps the locked 0.9.
    OPEN = GR100_OPEN
    CLOSE = float(os.environ.get("CLOSE_G", getattr(spec, "grasp_close", GR100_CLOSE)))

    # home tool pose per env, so the FIRST move (home->pre) is densified+smooth too (not a PD snap).
    hposL, hquatL = robot.ee_pose("left"); hposR, hquatR = robot.ee_pose("right")
    M_te, t_te = TOOL_IN_EE_INV[:3, :3], TOOL_IN_EE_INV[:3, 3]

    def ee_to_tool(p, q):                                       # inverse of ik()'s tool->ee map
        Rt = _R_from_wxyz(q) @ M_te.T
        return p - Rt @ t_te, _wxyz_from_R(Rt)

    def solve(p, q):                                           # IK both arms, pick the active one per env
        return np.where(side_is_left[:, None], ik(ee["l"], p, q), ik(ee["r"], p, q))

    # per-env active-arm HOME tool pose (so the first move home->pre is densified+smooth, not a PD snap).
    home_tool = np.zeros((N, 3)); home_tquat = np.zeros((N, 4))
    for i in range(N):
        hp, hq = ee_to_tool(hposL[i], hquatL[i]) if side_is_left[i] else ee_to_tool(hposR[i], hquatR[i])
        home_tool[i], home_tquat[i] = hp, hq

    def cur_tool_pose():
        """The active arm's CURRENT ee_link world pose, converted to the TOOL frame (the frame the
        waypoints/IK speak). Used to SEED each staged phase from where the previous phase left the arm, so
        every phase boundary is a smooth densified segment (no PD snap between phases)."""
        ep = np.where(side_is_left[:, None], np_(ee["l"].get_pos()), np_(ee["r"].get_pos()))
        eq = np.where(side_is_left[:, None], np_(ee["l"].get_quat()), np_(ee["r"].get_quat()))
        tp = np.zeros((N, 3)); tq = np.zeros((N, 4))
        for i in range(N):
            tp[i], tq[i] = ee_to_tool(ep[i], eq[i])
        return tp, tq

    def grasp_quat_at(i, cx, cy):
        """The orientation-aware grasp quat for env i with the cube at (cx,cy) -- reuses the LOCKED grasp
        builders (yaw-folded for the cube, tilt base toward the cube). Used both for the first grasp and to
        RE-PLAN the recovery grasp at the cube's NEW (shoved) location."""
        s = "left" if side_is_left[i] else "right"
        yr = ((yaw[i] + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        qzr = np.array([np.cos(yr / 2), 0.0, 0.0, np.sin(yr / 2)])
        return orientation_aware_grasp_quat(world_long_axis(laxis, qzr),
                                            tilted_base_quat(np.array([cx, cy]) - base[s], 0.0), reference_R=htR[s])

    # NOTE: the per-env SPARSE pick waypoints (pre->at->at->close->lift) are built inline below by the staged
    # phases A1 (approach, open) + A2 (close/lift or chase) and by ``approach_close_lift_wps`` (chase + retry),
    # all from the SAME locked grasp builders -- one motion path, no extra engine.

    # ---- execute the smooth batch + RECORD (states/actions + render 4 Nyx cams every REC_EVERY) ----
    acts, jpos, jvel, eepos, eequat, cubez = [], [], [], [], [], []
    cam_steps = {nm: [] for nm in cams}
    t0 = time.time()

    def record_state(full_cmd):
        acts.append(full_cmd[:, state14].copy())
        qp = np_(robot.entity.get_dofs_position()); qv = np_(robot.entity.get_dofs_velocity())
        jpos.append(qp[:, state14].copy()); jvel.append(qv[:, state14].copy())
        ep = np.where(side_is_left[:, None], np_(ee["l"].get_pos()), np_(ee["r"].get_pos()))
        eq = np.where(side_is_left[:, None], np_(ee["l"].get_quat()), np_(ee["r"].get_quat()))
        eepos.append(ep.copy()); eequat.append(eq.copy())

    dist_trace = []                                                # per-recorded-frame distractor XY (debug only)

    # ---- PENETRATION GATE (owner #1): track the WORST-ever solid-solid interpenetration across the whole
    # trajectory (the firm grasp is usually the peak), read straight from the solver's contact buffer. Any
    # demo whose worst penetration exceeds ABNORMAL_THRESH_M is REJECTED (not a clean success). ----
    # IGNORE the two opposing claws of the SAME gripper touching: when the gripper CLOSES ON NOTHING (a
    # deliberate empty close -- exactly the failed-grasp signal the disturbance harness wants), the L+R claws
    # of that gripper meet and the firm solver reports their mutual overlap (~1 cm). That is geometrically
    # EXPECTED designed contact (the gripper closes until the jaws nearly touch), NOT an abnormal penetration
    # defect -- the same category as the documented ~3 mm finger-into-cube contact skin the gate already
    # tolerates. We allow-list ONLY same-gripper L-claw<->R-claw geom pairs (computed by link name, robust to
    # geom-index changes); ALL other self-contact (arm-into-arm, claw-into-other-gripper) still counts.
    def _empty_close_ignore_pairs(entity):
        claw = {l.name: [g.idx for g in l.geoms] for l in entity.links
                if "gripper" in l.name and "link_1" in l.name}
        pairs = []
        for side in ("left", "right"):
            lg = claw.get(f"{side}_gripper_left_link_1", [])
            rg = claw.get(f"{side}_gripper_right_link_1", [])
            for a in lg:
                for b in rg:
                    pairs.append((a, b))
        return pairs
    pen_tracker = PenetrationTracker(stage.scene, n_envs=N,
                                     ignore_pairs=_empty_close_ignore_pairs(robot.entity))

    # FAST = a DEBUG-ONLY knob (grasp/DR tuning): skip the Nyx path-traced render in the run loop (the slow part,
    # ~6x the sim cost) and substitute tiny placeholder frames so the video/tile writers still run. The HDF5
    # state/actions + the grasp/place/penetration METRICS are unaffected (they read sim state, not pixels), so it
    # gives the COLLECT verdict line quickly while iterating object grasps. NEVER use it for a real collection
    # (the policy-cam videos would be blank). Unset (default) = full photoreal render.
    FAST = bool(os.environ.get("FAST"))
    _blank = {nm: np.zeros((H, W, 3), np.uint8) for nm in cams} if FAST else None

    def on_step(t, full, labels_t):
        record_state(full)
        pen_tracker.update()                                       # fold this step's contact buffer into the max
        views = _blank if FAST else stage.render()
        for nm in cams:
            cam_steps[nm].append(views[nm])
        cubez.append(np_(cube.get_pos())[:, 2].copy())
        if os.environ.get("DIST_DEBUG") and dist_ents:
            dist_trace.append(np.stack([np_(e.get_pos())[:, :2] for e in dist_ents], 0).copy())

    # ============================================================================================ #
    # DISTURBANCE HARNESS v2 — FAILURE-RECOVERY DATA (opt-in augmentation, per-env, NO barrier)
    # ============================================================================================ #
    # IDEA (owner spec, roadmap A): with a per-env PROBABILITY, a GENTLE random in-plane shove is injected on the
    # TARGET at a RANDOM time during the grasp APPROACH; the solver is "informed" only after a perception
    # SENSE-DELAY. Two collaborative branches follow:
    #   * informed BEFORE the close -> ABORT, rise a LITTLE, SMOOTHLY PIVOT / "chase" the new pose, close there.
    #   * informed only AFTER the close -> the grasp closed on nothing -> FAIL -> rise a LITTLE -> re-locate +
    #     re-grasp the new pose (failure -> recover -> replan data).
    # ONE mechanism, BOTH modes, on the SAME smooth path (BatchExecutor + densify). The recovery/chase RISE is
    # SMALL (RETRY_RISE) so the arm stays dexterous (no near-singularity straighten). The execution is PER-ENV
    # with NO cross-env barrier: a held env never idles in the air through another env's recovery (owner HARD
    # rule -- see the seg1/seg2/seg3 design below).
    # ``DISTURB`` DEFAULTS TO 0 -> the CLEAN single-trajectory pick-place (the foundation, the bulk fine-tune
    # data, and what the grasp-solving runs use). Set DISTURB>0 (e.g. 0.34) to OPT IN to failure-recovery data.
    DIST_PROB = float(os.environ.get("DISTURB", "0"))
    dspec = DisturbanceSpec.sample(cube, N, rng, prob=DIST_PROB)
    # the running per-env cube REST z (table top + half cube) -> "did the cube rise" test after each pick.
    cube_rest_z = root0[:, 2].copy()
    GRASP_RISE_M = 0.03                                            # cube must rise > 3cm to count as grasped
    EMPTY_CLOSE = GR100_MEET - 0.04                               # driven gripper near the empty-close stop
    FINGER_REACH = 0.12                                           # cube within this of the EE => something held
    # SMOOTH LOW retry/chase lift: rise only ~10 cm -- just enough to clear the cube + free the view to re-approach
    # the new pose -- NOT to a high "safe height" that straightens the arm toward singularity (the v1 ~jerk). The
    # densify cap + the warm-started single-sample IK keep every segment C1-continuous at this low height.
    RETRY_RISE = 0.10

    # gentle everywhere: lin 0.13 m/s, ang 0.9 rad/s (slows the wrist reorientation, the flip-prone part);
    # ik_every=1 so there's no zero-order-hold staircase (cheap now that IK does a single warm-started solve).
    ex = BatchExecutor(stage.scene, robot, side_is_left, rec_every=REC_EVERY, ang_speed=0.9, ik_every=1)
    # The executor RESETS its per-step jerk diagnostic (``ex.max_dq``) at the start of every ``run()`` (one
    # phase). To verify smoothness across ALL phases -- incl. the CHASE pivot and the RETRY re-grasp, not just
    # the final place -- accumulate the GLOBAL worst per-step active-arm joint jump here over every phase.
    global_max_dq = np.zeros(N)

    def run_phase(wps, settle_steps=0, during_step=None, tag=""):
        """Run ONE staged phase through the LOCKED executor, accumulating recording into the shared lists.
        Each phase seeds from the current arm pose (its first waypoint), so phase boundaries stay smooth."""
        nonlocal global_max_dq
        _pen_before = pen_tracker.depth_mm().max() if os.environ.get("PEN_TRACE") else 0.0
        T_ = ex.run(wps, solve, home16, on_step=on_step, settle_steps=settle_steps, during_step=during_step)
        global_max_dq = np.maximum(global_max_dq, ex.max_dq)       # fold this phase's worst jump into the global
        if os.environ.get("PEN_TRACE"):
            _pen_after = pen_tracker.depth_mm().max()
            print(f"[PEN_TRACE] phase {tag}: worst-ever pen {_pen_before:.1f} -> {_pen_after:.1f}mm "
                  f"(+{_pen_after-_pen_before:.1f} this phase)", flush=True)
        if os.environ.get("DQ_DEBUG"):
            we = int(np.argmax(ex.max_dq))
            lbl = ex.max_dq_label[we] if hasattr(ex, "max_dq_label") else "?"
            stp = int(ex.max_dq_step[we]) if hasattr(ex, "max_dq_step") else -1
            print(f"[DQ_DEBUG] phase {tag}: max|dq|={float(ex.max_dq.max()):.3f} rad @ env{we} "
                  f"step{stp} seg[{lbl}]", flush=True)
        return T_

    def detect_grasp_failed():
        """God-mode failure detection (privileged sim signals, owner's examples). A grasp FAILED if either:
        (1) the object did NOT rise (max object z so far this phase - rest z < GRASP_RISE_M), OR
        (2) the gripper closed on NOTHING: the driven claw sits at/near the empty-close stop AND the object is
            far from the EE (so nothing is between the fingers). Returns a per-env bool.

        OBJECT-AWARE (generalisation): the distance test uses the GRASP CENTRE (the offset-corrected point the
        claws actually converge on), NOT the raw root -- for a curved banana the root sits ~3cm off the grasped
        body point, which made the root-to-ee distance read 'far' and falsely flag a held banana as an empty
        close. For the cube grasp_center == root, so this is identical to the locked cube behaviour. The empty-
        close ANGLE test is gated to thin objects only by the distance check (a thin object legitimately closes
        the claws near GR100_MEET; the rise + grasp-centre-distance still catch a true empty close)."""
        cz = np_(cube.get_pos())
        gcen = grasp_center_world(cz, np_(cube.get_quat()))      # the point the claws hold (offset-corrected)
        rise = cz[:, 2] - cube_rest_z                             # how high the object currently is vs its rest
        gdpos = np_(robot.entity.get_dofs_position())
        gw = np.where(side_is_left, gdpos[:, gdrv["left"]], gdpos[:, gdrv["right"]])   # driven claw angle
        eep_now = np.where(side_is_left[:, None], np_(ee["l"].get_pos()), np_(ee["r"].get_pos()))
        obj_to_ee = np.linalg.norm(gcen - eep_now, axis=1)       # grasp-centre (claw point) to ee_link
        empty_close = (gw > EMPTY_CLOSE) & (obj_to_ee > FINGER_REACH)
        did_not_rise = rise < GRASP_RISE_M
        if os.environ.get("DETECT_DEBUG"):
            print(f"[DETECT_DEBUG] rise(cm)={np.round(rise*100,1).tolist()} gw={np.round(gw,3).tolist()} "
                  f"obj_to_ee(cm)={np.round(obj_to_ee*100,1).tolist()} "
                  f"empty={empty_close.tolist()} no_rise={did_not_rise.tolist()}", flush=True)
        return did_not_rise | empty_close

    def approach_close_lift_wps(start_p, start_q, rise_p, cx, cy, env_i):
        """Build a SMOOTH, singularity-robust re-grasp at the cube's CURRENT pose (cx,cy) FROM the arm's current
        pose -- the ONE builder for BOTH the CHASE pivot and the RETRY re-grasp. ``env_i`` selects the per-env
        arm side / yaw for the grasp quat.

        Smoothness fix (the v1 ~0.20 rad jerk): the re-grasp must change BOTH the wrist orientation (to the new
        orientation-aware grasp quat ``gqi``, re-tilted toward the moved cube) AND the position. Doing both in ONE
        rise->pre segment let the warm-started IK cross a branch near the top of the rise (a near-singularity
        spike). We DECOMPOSE it: (1) rise straight up keeping the CURRENT orientation (pure translation), (2)
        REORIENT to ``gqi`` at the apex as a pure rotation (densify dwells the position + SLERPs -- the arm is
        most dexterous here, elbow bent, well clear of the table), (3) descend to ``pre`` keeping ``gqi`` (pure
        translation), then the locked at->at->close->lift. Each segment is now either pure-translation or
        pure-rotation, so the IK never has to flip a branch -> every step stays C1-continuous (<~0.10 rad)."""
        gci = np.array([cx, cy, cube_rest_z[env_i] + grasp_dz])
        gqi = grasp_quat_at(env_i, cx, cy)
        tzi = _R_from_wxyz(gqi) @ np.array([0, 0, 1.0])
        apex = rise_p.copy()                                       # the rise apex (above the new cube)
        return [
            ("start",   start_p,             start_q, OPEN),
            ("rise",    rise_p,              start_q, OPEN),        # (1) pure-translation rise (keep orientation)
            ("reorient", apex,               gqi,     OPEN),        # (2) pure-rotation reorient at the apex
            ("pre",     gci - APP * tzi,     gqi,     OPEN),        # (3) pure-translation descend to pre-grasp
            ("at",      gci,                 gqi,     OPEN),
            ("at",      gci,                 gqi,     OPEN),
            ("close",   gci,                 gqi,     CLOSE),
            ("lift",    gci + [0, 0, LIFT],  gqi,     CLOSE),
        ], gqi

    Tlist = []
    recovery_attempts = np.zeros(N, np.int32)
    disturb_phase = np.array(["none"] * N, dtype=object)          # "before_close" | "after_close" | "none"
    chased = np.zeros(N, bool)                                    # informed-before-close -> pivoted/chased
    gqA = np.stack([grasp_quat_at(i, groot0[i, 0], groot0[i, 1]) for i in range(N)]).astype(np.float64)

    def grasp_xy_now():
        """The offset-corrected GRASP-centre xy from the object's LIVE (possibly shoved) pose -- used by the
        disturbance chase/retry so the re-grasp aims at the banana's BODY, not its AABB centre in the hollow."""
        return grasp_center_world(np_(cube.get_pos()), np_(cube.get_quat()))[:, :2]

    def pick_wps(i):
        """home -> pre -> at -> at -> close -> lift for env i: the orientation-aware top-down approach + a FIRM
        GR100 close (the object's own collision stops the claws; the high-kp PD holds the force) + a gentle lift."""
        gci = gc[i]
        tzi = _R_from_wxyz(gqA[i]) @ np.array([0, 0, 1.0])
        return [
            ("start", home_tool[i],       home_tquat[i], OPEN),
            ("pre",   gci - APP * tzi,    gqA[i], OPEN),
            ("at",    gci,                gqA[i], OPEN),
            ("at",    gci,                gqA[i], OPEN),
            ("close", gci,                gqA[i], CLOSE),
            ("lift",  gci + [0, 0, LIFT], gqA[i], CLOSE),
        ]

    def place_tail(i, liftp, gqi):
        """From the LIFTED grasp pose ``liftp`` (held at ``gqi``): re-yaw to the carry orientation, carry over the
        bowl, lower in, release, retract, GO HOME. This is APPENDED to the same continuous per-env waypoint stream
        (the pick, or a recovery re-grasp) so each env runs pick->place->home as ONE smooth trajectory and
        TERMINATES at home -- no staged barrier, no mid-air wait for other envs."""
        cqi = cquat(i, gqi)
        return [
            ("lift",    liftp + [0, 0, 0.02],   cqi, CLOSE),       # small settle + re-yaw to the carry orientation
            ("carry",   bxyz[i] + [0, 0, PAPP], cqi, CLOSE),
            ("lower",   bxyz[i],                cqi, CLOSE),
            ("rel",     bxyz[i],                cqi, OPEN),         # pose held -> gripper release ramp
            ("ret",     bxyz[i] + [0, 0, PAPP], cqi, OPEN),
            ("go_home", home_tool[i],           home_tquat[i], OPEN),  # smooth densified RETURN HOME (recorded)
        ]

    if not (dspec.any and DIST_PROB > 0):
        # ===================== CLEAN PATH: ONE continuous per-env trajectory (NO barrier) =====================
        # home -> pre -> at -> close -> lift -> carry -> lower -> release -> HOME, run as a SINGLE smooth batch.
        # Every env runs its OWN trajectory to completion; NO env ever holds a lifted object waiting for a slower
        # env (the idle-in-the-air bug came entirely from the old staged per-phase barriers). Faster envs reach
        # home first and the recorder TRIMS each env's idle-home tail -> VARIABLE-LENGTH demos (natural). ----
        wps = [pick_wps(i) + place_tail(i, gc[i] + [0, 0, LIFT], gqA[i]) for i in range(N)]
        Tlist.append(run_phase(wps, settle_steps=40, tag="pick-place-home"))
    else:
        # ============= DISTURBANCE PATH: failure-recovery data, still per-env with NO lifted barrier =============
        # seg1 APPROACH (fire the gentle shove) -> chase-vs-after-close; seg2 CLOSE+LIFT (chase envs pivot to the
        # moved cube); detect the closed-on-nothing failures; seg3 = each env's OWN continuous remainder -- a HELD
        # env runs place->home, a FAILED env runs rise->reopen->relocate->re-grasp->lift->place->home, ALL in one
        # batch. The held env NEVER waits in the air through the recovery (the old B-retry loop's ~5s idle is
        # gone): it places + goes home + terminates on its own timeline; the recorder trims its idle-home tail.
        # One recovery attempt ("rise a bit, try the new pose"). ``DISTURB=0`` (default) skips all of this. ----
        wpsA1 = [pick_wps(i)[:4] for i in range(N)]               # approach only (home -> pre -> at -> at)
        _, _, _, _, T_A1 = ex.plan(wpsA1)
        dspec.reset_window(T_A1)
        Tlist.append(run_phase(wpsA1, settle_steps=0, during_step=lambda t, lab: dspec.tick(t), tag="approach"))
        informed_before_close = dspec.informed_by(T_A1 - 1) & dspec.disturbed
        chased = informed_before_close.copy()
        disturb_phase[dspec.disturbed & informed_before_close] = "before_close"
        disturb_phase[dspec.disturbed & ~informed_before_close] = "after_close"
        fired = dspec.fired_any()
        print(f"[COLLECT] disturbance: shoved {int(fired.sum())} env(s) {np.where(fired)[0].tolist()} "
              f"(prob={DIST_PROB}, approach_T={T_A1}); before_close(chase)={int(informed_before_close.sum())} "
              f"after_close(retry)={int((dspec.disturbed & ~informed_before_close).sum())}", flush=True)

        # seg2: CLOSE + LIFT (chase envs pivot to the cube's NEW sensed pose, then close+lift there)
        cur_p, cur_q = cur_tool_pose()
        cube_now = np_(cube.get_pos()); gxy_now = grasp_xy_now()
        wpsA2 = []
        for i in range(N):
            if chased[i]:
                sp = cur_p[i].copy(); sp[2] = max(cur_p[i, 2], cube_now[i, 2] + grasp_dz) + RETRY_RISE
                wp, gqi = approach_close_lift_wps(cur_p[i], cur_q[i], sp, gxy_now[i, 0], gxy_now[i, 1], i)
                wpsA2.append(wp); gqA[i] = gqi
            else:
                wpsA2.append([
                    ("start", cur_p[i],                cur_q[i], OPEN),
                    ("close", cur_p[i],                cur_q[i], CLOSE),
                    ("lift",  cur_p[i] + [0, 0, LIFT], cur_q[i], CLOSE),
                ])
        Tlist.append(run_phase(wpsA2, settle_steps=20, tag="close/chase"))

        failed = detect_grasp_failed()
        chased &= ~failed                                        # a chase that still missed isn't a clean chase
        print(f"[COLLECT] detect after pick: failed={int(failed.sum())}/{N} "
              f"(disturbed {int((failed & dspec.disturbed).sum())}/{int(dspec.disturbed.sum())})", flush=True)

        # seg3: per-env remainder -- a FAILED env recovers (rise->reopen->relocate->re-grasp->lift) then places;
        # a HELD env places straight away. ALL in one batch, so a held env runs its place->home concurrently with
        # the recovery (never idling in the air) and TERMINATES at its own home; the trim drops the held tail.
        cur_p, cur_q = cur_tool_pose(); gxy_f = grasp_xy_now()
        wpsC = []
        for i in range(N):
            if failed[i]:
                recovery_attempts[i] = 1
                sp = cur_p[i].copy(); sp[2] = max(cur_p[i, 2], np_(cube.get_pos())[i, 2] + grasp_dz) + RETRY_RISE
                wp, gqi = approach_close_lift_wps(cur_p[i], cur_q[i], sp, gxy_f[i, 0], gxy_f[i, 1], i)
                gqA[i] = gqi
                wpsC.append(wp + place_tail(i, wp[-1][1], gqi))   # wp[-1] = the re-grasp's ("lift", pos, ...)
            else:
                wpsC.append([("start", cur_p[i], cur_q[i], CLOSE)] + place_tail(i, cur_p[i], gqA[i]))
        Tlist.append(run_phase(wpsC, settle_steps=40, tag="place/recover"))

    T = sum(Tlist)
    lift_pos_z = np.max(np.stack(cubez, 1), axis=1)            # per-env max cube height reached during the run
    wall = time.time() - t0
    li, ri = np.where(side_is_left)[0], np.where(~side_is_left)[0]
    # quantitative jerk gate: worst single-step active-arm joint jump across ALL phases (pick + CHASE + RETRY +
    # place). A branch flip / near-singularity = a >1 rad spike on a wrist joint; with the smooth low retry-lift
    # + the max_samples=1 warm-started IK this stays small (target <~0.12 rad, no spike).
    print(f"[COLLECT] executed T={T} ({len(li)} left / {len(ri)} right arm)  render+sim {wall:.1f}s  "
          f"max per-step |dq|={float(global_max_dq.max()):.3f} rad (all phases)", flush=True)

    # ---- score + realistic penetration check (per-env bowl centre) ----
    objf = np_(cube.get_pos())
    eep = np.where(side_is_left[:, None], np_(ee["l"].get_pos()), np_(ee["r"].get_pos()))
    lift_cm = (lift_pos_z - root0[:, 2]) * 100
    ch2 = spec.scaled_extents()[2] / 2
    rxy = np.hypot(objf[:, 0] - bowx, objf[:, 1] - bowy)
    rim_z = tabZ + 2 * BOWL_HALF_H
    # placed-XY tolerance from the SPEC: a cube's tight 6cm stays exactly 6cm (default place_xy_tol_cm=8 -> the
    # cube path historically used a tighter 0.06; keep 0.06 for the cube, the spec tol for bigger objects whose
    # bbox centre can rest a few cm off the bowl centre while the body still lies IN the bowl). Cap at the bowl
    # mouth radius so it never accepts a target resting OUTSIDE the bowl.
    place_r = 0.06 if spec.is_cube else min(0.085, max(0.06, spec.place_xy_tol_cm / 100.0))
    # height band: a compact object's bbox centre rests near the bowl floor (< rim); an ELONGATED/flat target
    # draped across the ~7.5cm bowl mouth rests with its bbox centre HIGHER (part of the body bridges the rim),
    # so allow the centre up to ~one body-half above the rim for non-cube targets (still rejects a target perched
    # well ABOVE the bowl or stuck on a finger -- the ee-distance test below catches the held case).
    top_margin = 0.01 if spec.is_cube else float(ch2 + 0.02)
    placed = (lift_cm > 3) & (rxy < place_r) & (objf[:, 2] - ch2 > tabZ - 0.005) & \
             (objf[:, 2] - ch2 < rim_z + top_margin) & (np.linalg.norm(objf - eep, axis=1) > 0.08)
    bottom = objf[:, 2] - ch2
    wall_pen = (((rxy > 0.065) & (rxy < 0.11) & (bottom > tabZ + 0.012) & (bottom < rim_z)) | (bottom < tabZ - 0.015))
    print(f"[COLLECT] {int((lift_cm>3).sum())}/{N} grasped, {int(placed.sum())}/{N} placed, "
          f"through-wall={int(wall_pen.sum())}/{N}  render+sim {wall:.1f}s", flush=True)

    # ---- AUTHORITATIVE penetration gate (owner #1): worst-ever solid-solid interpenetration per env, read
    # straight from the solver. A demo with abnormal interpenetration is REJECTED -- it does NOT count as a
    # clean success (so success_only export drops it), regardless of whether the cube landed in the bowl. The
    # through-wall metric above stays (a useful task-specific check) but THIS detector is the gate. ----
    pen_mm = pen_tracker.depth_mm()                                # (N,) worst-ever penetration depth in mm
    penetrating = pen_tracker.abnormal(ABNORMAL_THRESH_M)          # (N,) bool: exceeds the abnormal threshold
    placed = placed & ~penetrating & ~degenerate                  # penetrating OR degenerate-settle => NOT clean
    worst_e = int(np.argmax(pen_mm))
    print(f"[COLLECT] penetration: max={float(pen_mm.max()):.1f}mm "
          f"(thresh={ABNORMAL_THRESH_M*1000:.0f}mm), abnormal={int(penetrating.sum())}/{N}"
          + (f"  worst env{worst_e}: {pen_tracker.worst_names()[worst_e]}" if pen_mm.max() > 0 else ""),
          flush=True)

    # ---- DISTURBANCE / RECOVERY summary (BOTH training signals). ``disturbed``: the cube was shoved during the
    # grasp. ``disturb_phase``: was the solver informed before/after the close. ``disturb_outcome`` per env:
    #   "chased"    -> informed BEFORE close: ABORTED the close, pivoted to the new pose, ended PLACED
    #                  (moving-object CHASE data). ``recovery_attempts``==0.
    #   "recovered" -> informed AFTER close and ended PLACED: either the close-on-nothing FAILED and a smooth
    #                  low retry re-grasped the new pose (``recovery_attempts``>0, the failure->recover data),
    #                  OR the late shove was benign and the grasp rode through it (``recovery_attempts``==0).
    #   "failed"    -> disturbed but did NOT end cleanly placed (a legit hard edge the place/pen gate rejects).
    #   "none"      -> not disturbed.
    # ``recovery_attempts`` separates the genuine retry (>0) from the benign-survival (0) inside "recovered".
    # ``recovered`` (legacy bool attr) = disturbed AND ended placed (chased OR recovered), kept for back-compat. ----
    disturbed = dspec.disturbed.copy()
    disturb_outcome = np.array(["none"] * N, dtype=object)
    for i in np.where(disturbed)[0]:
        if placed[i] and chased[i]:
            disturb_outcome[i] = "chased"
        elif placed[i]:
            disturb_outcome[i] = "recovered"
        else:
            disturb_outcome[i] = "failed"
    recovered = disturbed & placed                                # disturbed AND ended in a clean place
    n_dist = int(disturbed.sum())
    n_chased = int((disturb_outcome == "chased").sum())
    n_recov = int((disturb_outcome == "recovered").sum())
    n_failed = int((disturb_outcome == "failed").sum())
    print(f"[COLLECT] disturbance: {n_dist} disturbed ({n_chased} chased, {n_recov} recovered, {n_failed} failed)"
          f"  recovery_attempts(disturbed)={recovery_attempts[disturbed].tolist() if n_dist else []}", flush=True)

    # ---- distractor collision-free metric: each distractor's XY displacement from its SETTLED pose to its
    # FINAL pose. If the arm avoided them (the corridor placement worked), this is ~0; a big value means the
    # arm swiped one (TIGHTEN the corridor clearances and re-run). ----
    if dist_ents:
        dist_xy_f = np.stack([np_(e.get_pos())[:, :2] for e in dist_ents], 0)   # (K,N,2)
        dist_disp = np.hypot(dist_xy_f[..., 0] - dist_xy0[..., 0], dist_xy_f[..., 1] - dist_xy0[..., 1])  # (K,N)
        disp_cm = dist_disp * 100
        n_pairs = disp_cm.size
        n_clean = int((disp_cm < 2.0).sum())
        print(f"[COLLECT] distractors: max XY disp={float(disp_cm.max()):.2f}cm  mean={float(disp_cm.mean()):.2f}cm  "
              f"clean(<2cm)={n_clean}/{n_pairs} ({100*n_clean/n_pairs:.0f}%)  knocked(>=2cm): "
              f"{[(dist_names[k], int((disp_cm[k]>=2.0).sum())) for k in range(len(dist_names))]}", flush=True)
        if os.environ.get("DIST_DEBUG"):
            tr = np.stack(dist_trace, 0) if dist_trace else None    # (F,K,N,2)
            for k in range(len(dist_names)):
                for e in range(N):
                    if disp_cm[k, e] >= 2.0:
                        msg = (f"[DIST_DEBUG] env{e} {dist_names[k]}: settled={np.round(dist_xy0[k,e],3)} "
                               f"final={np.round(dist_xy_f[k,e],3)} disp={disp_cm[k,e]:.1f}cm | "
                               f"arm={'L' if side_is_left[e] else 'R'} cube={np.round([cubx[e],cuby[e]],3)} "
                               f"bowl={np.round([bowx[e],bowy[e]],3)}")
                        if tr is not None:                          # cumulative move from frame0, sampled across run
                            d = np.hypot(tr[:, k, e, 0] - tr[0, k, e, 0], tr[:, k, e, 1] - tr[0, k, e, 1]) * 100
                            F = len(d); idx = [0, F // 4, F // 2, 3 * F // 4, F - 1]
                            msg += "  trace@[0,25,50,75,100]%=" + str([round(float(d[i]), 1) for i in idx])
                        print(msg, flush=True)

    # ---- write the fine-tune data (HDF5 §4 schema + 3 policy-cam videos) + tiles ----
    os.makedirs(data_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    acts = np.stack(acts, 1); jpos = np.stack(jpos, 1); jvel = np.stack(jvel, 1)   # (N,Tr,14)
    eepos = np.stack(eepos, 1); eequat = np.stack(eequat, 1)
    Tr = acts.shape[1]
    # ---- PER-ENV NATURAL TERMINATION (owner HARD rule): each trial ends when ITS arm reaches home. The executor
    # pads every env to the global max T (hold-last-pose) so the slowest env (longest reach / a disturbance
    # recovery) can finish; a faster env then sits STATIC at home for the tail. That idle-home tail is NOT
    # recorded -- trim each env to its own last MOTION frame (+ a small settle margin). Demos come out
    # VARIABLE-LENGTH and that is natural + accepted (LeRobot supports it). Detection is purely on the recorded
    # joint stream (active + idle arm + gripper): the home hold is exactly static (PD jitter < 1e-4 rad/frame),
    # real motion is >> 1.5e-3, so the two separate cleanly. NB this only ever trims a STATIC tail -- the
    # single continuous trajectory has no mid-trajectory hold to trim. ----
    _frame_motion = np.abs(np.diff(jpos, axis=1)).max(axis=2)     # (N, Tr-1) per-frame max joint delta
    end_idx = np.full(N, Tr - 1, int)
    for e in range(N):
        _mv = np.where(_frame_motion[e] > 1.5e-3)[0]             # frames with real joint motion arriving into them
        if len(_mv):
            end_idx[e] = min(Tr - 1, int(_mv[-1]) + 1 + 2)       # +1 the arrival frame, +2 a small settle margin
    demo_len = (end_idx + 1).astype(int)                         # per-env recorded length (variable)
    print(f"[COLLECT] per-env demo length (natural termination): min={int(demo_len.min())} "
          f"max={int(demo_len.max())} mean={demo_len.mean():.0f} of Tr={Tr} recframes", flush=True)
    vdir = os.path.join(data_dir, "videos")
    for nm in ("cam_side", "cam_lw", "cam_rw"):
        os.makedirs(os.path.join(vdir, nm), exist_ok=True)
    with h5py.File(os.path.join(data_dir, "demos.hdf5"), "w") as f:
        for e in range(N):
            Te = int(demo_len[e])                              # this env's natural length (trimmed at home)
            d = f.create_group(f"data/demo_{e}")
            d.create_dataset("actions", data=acts[e, :Te].astype(np.float32))
            sg = d.create_group("states/articulation/robot")
            sg.create_dataset("joint_position", data=jpos[e, :Te].astype(np.float32))
            sg.create_dataset("joint_velocity", data=jvel[e, :Te].astype(np.float32))
            pg = d.create_group("ee_pose")
            pg.create_dataset("position", data=eepos[e, :Te].astype(np.float32))
            pg.create_dataset("orientation", data=eequat[e, :Te].astype(np.float32))
            d.attrs["num_samples"] = Te; d.attrs["success"] = bool(placed[e]); d.attrs["seed"] = int(seed)
            d.attrs["arm"] = "left" if side_is_left[e] else "right"
            # PENETRATION GATE attrs (owner #1): worst-ever solid-solid interpenetration (mm) + the abnormal
            # flag. ``penetrating`` True means the demo is REJECTED -- ``success`` is already forced False
            # above, so the success_only LeRobot export drops it; this poisoned data is never shipped.
            d.attrs["max_penetration_mm"] = float(pen_mm[e])
            d.attrs["penetrating"] = bool(penetrating[e])
            # DEGENERATE-SETTLE flag: the cube was ejected off the table at spawn (overlapping bowl) -> the
            # grasp targets a phantom location -> demo REJECTED (success already forced False above).
            d.attrs["degenerate"] = bool(degenerate[e])
            d.attrs["hdr"] = os.path.basename(stage.hdrs[e])
            d.attrs["has_distractors"] = bool(has_dist[e])     # 50/50 per-env: was this a cluttered trial?
            d.attrs["distractors"] = ",".join(dist_names) if has_dist[e] else ""
            # DISTURBANCE attrs (this feature, v2): ``disturbed`` = the cube was gently shoved at a RANDOM time
            # during the grasp approach (per-env probability, like the 50/50 distractors). ``disturb_phase`` =
            # whether the solver was informed "before_close" (-> chase) or "after_close" (-> fail+retry).
            # ``disturb_outcome`` in {"chased","recovered","failed","none"} (see summary above): "chased" is the
            # moving-object-chase signal, "recovered" is the failure->recover signal. ``recovery_attempts`` =
            # how many extra re-grasps the god-mode solver needed (0 for a clean chase). ``recovered`` (legacy)
            # = disturbed AND ended cleanly placed (chased OR recovered).
            d.attrs["disturbed"] = bool(disturbed[e])
            d.attrs["disturb_phase"] = str(disturb_phase[e])
            d.attrs["disturb_outcome"] = str(disturb_outcome[e])
            d.attrs["recovered"] = bool(recovered[e])
            d.attrs["recovery_attempts"] = int(recovery_attempts[e])
            # PER-DEMO DR PLAN (the ``DRPlan`` of docs/domain_randomization.md, made traceable): the exact
            # per-env physics-DR VALUES this demo sampled, plus the sweep multipliers in force. This is what
            # makes the DR strategist's failure diagnosis DEFENSIBLE -- it can correlate an outcome (success/
            # penetrating/degenerate) with WHERE in the DR space the env landed (cube/bowl pose, table height,
            # mass, yaw, reach), and measure the achieved DIVERSITY. Prefixed ``dr_`` so it never collides with
            # the outcome attrs above. Values are arm-frame: cube/bowl y are signed by the active arm side.
            d.attrs["dr_cubx"] = float(cubx[e]); d.attrs["dr_cuby"] = float(cuby[e])
            d.attrs["dr_bowx"] = float(bowx[e]); d.attrs["dr_bowy"] = float(bowy[e])
            d.attrs["dr_tabZ"] = float(tabZ[e]); d.attrs["dr_yaw"] = float(yaw[e])
            d.attrs["dr_mass_shift"] = float(dr["mass_shift"][e, 0])
            d.attrs["dr_clr"] = float(np.hypot(cubx[e] - bowx[e], cuby[e] - bowy[e]))  # cube<->bowl centre dist
            d.attrs["dr_reach"] = float(np.hypot(cubx[e], cuby[e]))                    # cube dist from arm base
            d.attrs["dr_pose_scale"] = _dr_scale("DR_POSE_SCALE")
            d.attrs["dr_mass_scale"] = _dr_scale("DR_MASS_SCALE")
            d.attrs["dr_fric_scale"] = _dr_scale("DR_FRIC_SCALE")
            for nm in ("cam_side", "cam_lw", "cam_rw"):        # the sensor-only policy stream (trimmed to Te)
                vid = np.stack([cam_steps[nm][t][e] for t in range(Te)])
                iio.imwrite(os.path.join(vdir, nm, f"demo_{e}.mp4"), vid, fps=12, codec="libx264")
    third_tiles = [tile([cam_steps["third"][t][e] for e in range(N)], H, W) for t in range(Tr)]
    iio.imwrite(os.path.join(out_dir, f"fulldr_third_{N}.mp4"), np.stack(third_tiles), fps=12, codec="libx264")
    chosen = sorted(rng.choice(N, size=min(10, N), replace=False).tolist())
    lab = {"third": "third", "cam_side": "side", "cam_lw": "wrist-L", "cam_rw": "wrist-R"}
    for e in chosen:
        frames = []
        for t in range(Tr):
            q = {nm: _label(cam_steps[nm][t][e], lab[nm]) for nm in lab}
            frames.append(np.vstack([np.hstack([q["third"], q["cam_side"]]),
                                     np.hstack([q["cam_lw"], q["cam_rw"]])]))
        iio.imwrite(os.path.join(out_dir, f"fourview_demo_{e}.mp4"), np.stack(frames), fps=12, codec="libx264")

    # ---- distractor PREVIEW (third-person + four-view) at the settled first frame, for review. Pick the env
    # whose distractors are most spread on the table (largest pairwise XY spread) so the clutter is clearly
    # visible alongside the cube+bowl. Stacks [third | side] over [wrist-L | wrist-R]. ----
    if dist_ents:
        d0 = dist_xy0                                            # (K,N,2) settled distractor XY
        spread = d0[:, :, 0].std(0) + d0[:, :, 1].std(0) if d0.shape[0] > 1 else np.zeros(N)
        pe = int(np.argmax(spread))                             # the most-spread (clearest clutter) env
        q0 = {nm: _label(cam_steps[nm][0][pe], lab[nm]) for nm in lab}
        preview = np.vstack([np.hstack([q0["third"], q0["cam_side"]]),
                             np.hstack([q0["cam_lw"], q0["cam_rw"]])])
        iio.imwrite(os.path.join(out_dir, "distractors_preview.png"), preview)
        iio.imwrite(os.path.join(out_dir, "distractors_third.png"), cam_steps["third"][0][pe])
        print(f"[COLLECT] distractor preview env={pe} ({dist_names}) -> {out_dir}/distractors_preview.png", flush=True)

    print(f"[COLLECT] wrote {N} demos ({int(placed.sum())} placed) -> {data_dir}/demos.hdf5 + videos ; "
          f"tiles -> {out_dir} (4view demos {chosen})")
    print("COLLECT_DONE")
    return dict(placed=int(placed.sum()), grasped=int((lift_cm > 3).sum()), through_wall=int(wall_pen.sum()),
                max_penetration_mm=float(pen_mm.max()), abnormal_penetration=int(penetrating.sum()))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    # The CONFIGURABLE target: default "cube" so ``pickplace.py N seed`` reproduces the cube collection EXACTLY
    # (paths + DR + clutter byte-for-byte). For a non-cube target the default DATA/OUT dirs are suffixed by the
    # target so different objects never overwrite each other; DATA_DIR/OUT_DIR env vars still override.
    suffix = "" if TARGET == "cube" else f"_{TARGET}"
    DATA = os.environ.get("DATA_DIR", f"/data3/genesis_fulldr{suffix}")
    OUT = os.environ.get("OUT_DIR", f"genesis_firefly/output/temp/fulldr_collect{suffix}")
    gs.init(backend=gs.gpu)
    collect(N, SEED, DATA, OUT, target=TARGET)
