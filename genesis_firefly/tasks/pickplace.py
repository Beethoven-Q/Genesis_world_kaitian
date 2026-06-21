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
from robots.firefly_dual import GR100_OPEN, GR100_CLOSE, GR100_MIMIC  # noqa: E402
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


def target_color(spec, free_color, rng):
    """The GRASP TARGET's render colour (DR-strategist-owned per-object COLOUR POLICY; see the colour-policy table
    in .claude/workbooks/dr_workbook.md). Three classes:
      * NATIVE TEXTURE (apple): the spec declares ``native_texture`` -> the object renders its OWN UV-mapped skin
        in spawn_target, so the flat colour here is unused (NO colour DR). We return a FIXED palette base (no
        jitter) purely as the NATIVE_TEX=0 fallback colour -- the texture is the real colour.
      * REALISTIC PALETTE with colour-DR (banana/pen): pick ONE palette entry + a SMALL per-channel jitter
        (+/-0.04) for natural variation (banana yellow/green not pink; pen black/blue/red).
      * FIXED colour, no DR (tennis_ball): a single-entry palette stays its one true regulation colour (the
        +/-0.04 jitter on one entry is negligible -> effectively fixed).
      * FREE random (cube): no palette -> the task's FREE distinct-from-table colour (cube byte-for-byte).
    This is the FIX for the bug where the cube's free random colour was applied to EVERY target (the PINK banana)."""
    pal = getattr(spec, "target_palette", None)
    if not pal:
        return free_color                                       # cube: keep the free random distinct colour
    if getattr(spec, "native_texture", None):                   # native-texture object: fixed fallback, NO DR
        return tuple(np.asarray(pal[0], float).tolist())        # (texture wins in spawn_target; colour unused)
    base = np.asarray(pal[int(rng.randint(len(pal)))], float)
    jit = (rng.rand(3) - 0.5) * 0.08                            # +/-0.04 per channel -> subtle natural variation
    return tuple(np.clip(base + jit, 0.0, 1.0).tolist())


def _target_usd_surface(spec, color):
    """The GRASP-TARGET visual surface for a USD-sourced object. NATIVE TEXTURE (DR-strategist colour policy):
    if the spec declares a ``native_texture`` (a UV-mapped diffuse image, e.g. the apple's apple_02.png) AND the
    object's clean .obj actually carries UVs, render the object's OWN photoreal skin via gs.textures.ImageTexture
    -- the SAME idiom the table tops use (verified in Nyx, no segfault). This makes the apple a realistic textured
    apple instead of a flat pink blob. Otherwise (banana/pen clean.obj have NO UVs; the cube/sphere don't reach
    here) fall back to the realistic flat ``color`` from the per-object palette. NATIVE_TEX=0 forces the flat
    fallback (ablation)."""
    use_tex = bool(getattr(spec, "native_texture", None)) and os.environ.get("NATIVE_TEX", "1") != "0"
    if use_tex:
        tex_path = OBJECTS / spec.native_texture
        return gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(tex_path)), roughness=0.5)
    return gs.surfaces.Plastic(color=color, roughness=0.5)


def spawn_target(scene, spec, color, pos_xy=(0.40, 0.18), z=0.30):
    """Spawn the GRASP TARGET as a real collidable rigid body with a FAITHFUL grasp collider + a Nyx-safe visual.
    The target is the one object that gets GRASPED, so (unlike a distractor) its USD-mesh collider is a convex
    DECOMPOSITION (coacd, the same recipe as the bowl), not a single hull, so the gripper closes on the real
    elongated/flat shape (a banana's curve, a pen's thin body, a book's flat slab) rather than a fat envelope.
    Procedural cube/sphere keep their exact box/sphere collider (already faithful). The visual is the procedural
    box/sphere for those sources, or the Nyx-safe clean .obj for USD sources (the textured USD segfaults Nyx, the
    same reason the bowl + distractors render from clean meshes). VISUAL COLOUR: a USD target with a declared
    NATIVE TEXTURE (the apple) renders its OWN UV-mapped skin (apple_02.png) via _target_usd_surface; otherwise
    ``color`` (the per-object realistic palette) is used so the target stays visible against the randomized table.
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
            material=mat, surface=_target_usd_surface(spec, color))
    decomp = float(os.environ.get("TGT_DECOMP", getattr(spec, "grasp_decompose_err", 0.04)))
    return scene.add_entity(
        gs.morphs.Mesh(file=str(OBJECTS / spec.mesh_subpath), pos=(x, y, z), scale=spec.scale,
                       convexify=True, decompose_object_error_threshold=decomp, decimate=False),
        material=mat, surface=_target_usd_surface(spec, color))


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
    # PER-OBJECT contact noslip: the flat-faced cube holds with a leaky cone (and a tight one over-penetrates it),
    # so it uses 0; a round/curved/thin target needs the tight cone (5) to not eject/slip. Derived from the spec.
    _noslip = int(getattr(REGISTRY[target], "grasp_noslip", 5))
    stage = ManipulationStage(N, seed=seed, noslip=_noslip)   # the reusable robot+cameras+rendering+env-DR setup
    # ``spec`` is the CONFIGURABLE grasp target (default cube; TARGET env var selects any registry object). The
    # variable below stays named ``cube`` so the ~600 lines of locked grasp/score/disturbance logic that reference
    # it are unchanged -- it is the TARGET entity, which the orientation-aware grasp handles via spec.ref-axis.
    lay, spec, rng = stage.lay, REGISTRY[target], stage.rng
    print(f"[COLLECT] TARGET = {target!r} ({spec.language_name}); source={spec.source} "
          f"extents={np.round(spec.scaled_extents(), 3).tolist()} "
          f"ref_axis={'long' if spec.elongated else ('face' if spec.is_cube else 'none(round)')}", flush=True)
    dr = sample_phys_dr(N, rng, lay, spec)

    # --- task objects: the coloured TARGET + a convex-decomposition bowl, distinct from the table colour ---
    # COLOUR (owner: a banana rendered PINK -- WRONG): the GRASP TARGET's colour comes from its REALISTIC
    # per-object palette (spec.target_palette) -- a banana is yellow (mostly) or green (unripe), an apple red or
    # green, a pen a normal marker colour, a tennis ball yellow-green. This FIXES the bug where the cube's FREE
    # random distinct-from-table colour (``cube_col``) was applied to EVERY target. The CUBE has no palette
    # (target_palette=None) so it keeps its free random colour -> the cube collection stays byte-for-byte. The
    # bowl always uses a free distinct colour (it is a container, not a realistically-coloured object).
    ch, cube_col = stage.distinct_object_color()                 # the cube's free random colour (+ a hue to avoid)
    tgt_col = target_color(spec, cube_col, stage.rng)            # realistic palette colour (cube -> cube_col)
    _, bowl_col = stage.distinct_object_color(ch)
    cube = spawn_target(stage.scene, spec, tgt_col, pos_xy=(0.40, 0.18), z=0.30)   # the TARGET entity
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
    armdof, state14 = robot.arm, robot.state14_idx
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
    def cquat(i, gqi, tilt_deg=8.0):
        """Carry/place orientation for env i: a base tilted ``tilt_deg`` toward the BOWL reach direction,
        re-yawed (transport_quats) to the branch closest to the grasp orientation ``gqi`` (smooth wrist).
        ``tilt_deg`` defaults to 8 (the locked gentle tilt); ``select_place_tilt`` relaxes it per-env when a
        higher carry-over-bowl pose would otherwise SATURATE the wrist (RoboLab reachable_place_quat policy,
        extended with the wrist margin -- the carry over the bowl is the OTHER frame that over-stretches)."""
        s = "left" if side_is_left[i] else "right"
        return transport_quats(tilted_base_quat(np.array([bowx[i], bowy[i]]) - base[s], float(tilt_deg)),
                               reference_quat=gqi)[0]

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
    # APP = pre-grasp standoff; PAPP = place approach height (both RoboLab/plan_pick_place defaults).
    # LIFT: lowered from the plan_pick_place 0.18 default to 0.10 -- the NATURAL-POSTURE fix. A pure +Z lift
    # raises the EE straight up; in this LOW workspace (table at the arm-base level) a big 0.18 lift climbs to
    # EEz~0.50-0.58 where the wrist SATURATES at its limit + the elbow collapses (the over-stretch). The
    # MEASURED wrist-vs-lift curve (the SODA/Genesis-identical probe + the diagnosis) is decisive: +0.18 -> j4
    # 1.57 SATURATED, +0.10 -> 1.43, +0.08 -> 1.35. 0.10 still clears the cube + table for a clean carry (the
    # cube only needs to rise >3cm to count as grasped) and matches RoboLab's gentle milestone lift (its
    # pick_place_cube.py uses 0.07). Combined with the wrist-margin grasp TILT below, the wrist stays well off
    # its limit and the elbow stays bent through the WHOLE pick + lift + carry. Override with LIFT_H to tune.
    APP = float(os.environ.get("APP_H", "0.12"))
    LIFT = float(os.environ.get("LIFT_H", "0.10"))
    # PAPP = the carry-hover / retreat height ABOVE the bowl. Lowered from 0.08 to 0.06: the carry hover sits on
    # the same low table, so a tall hover climbs the EE into the wrist-saturating region (the carry was the
    # residual over-stretch). 0.06 still clears the bowl rim for a gentle lower-in/retreat. Combined with the
    # wrist-margin carry TILT (select_place_tilt) the wrist stays off its limit through the whole place.
    PAPP = float(os.environ.get("PLACE_APP_H", "0.06"))
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

    def grasp_quat_at(i, cx, cy, tilt_deg=0.0):
        """The orientation-aware grasp quat for env i with the cube at (cx,cy) -- reuses the LOCKED grasp
        builders (yaw-folded for the cube, tilt base toward the cube). Used both for the first grasp and to
        RE-PLAN the recovery grasp at the cube's NEW (shoved) location.

        ``tilt_deg`` (default 0 = pure top-down) tilts the approach axis AWAY from straight-down TOWARD the
        reach direction (cube - arm base), EXACTLY as RoboLab's ``reachable_grasp_quat`` relax-tilt does. A
        small forward tilt keeps the WRIST off its limit + the ELBOW bent through the lift in this LOW (table-
        at-base-level) workspace where a pure top-down lift is near-singular. The per-env tilt is chosen by
        ``select_grasp_tilt`` below (prefer top-down; relax to the smallest tilt that stays wrist-comfortable)."""
        s = "left" if side_is_left[i] else "right"
        # Fold the in-plane yaw into the object's SYMMETRY wedge before building the reference axis. The CUBE is
        # 4-fold symmetric (a face repeats every 90deg) -> fold to [-45,45]. An ELONGATED object (banana/pen) is
        # only 2-fold symmetric about its LONG axis (the line repeats every 180deg) -> fold to [-90,90]; folding
        # it into the cube's 45deg wedge computed the grasp for the WRONG axis and the claws MISSED the banana at
        # |yaw|>45 (verified: collector banana failures were exactly the large-|yaw| envs). A ROUND object has no
        # axis (laxis is None) so the fold is irrelevant. fold = pi/2 (cube) or pi (elongated).
        fold = (np.pi / 2) if spec.is_cube else np.pi
        yr = ((yaw[i] + fold / 2) % fold) - fold / 2
        qzr = np.array([np.cos(yr / 2), 0.0, 0.0, np.sin(yr / 2)])
        base_q = tilted_base_quat(np.array([cx, cy]) - base[s], float(tilt_deg))
        gq = orientation_aware_grasp_quat(world_long_axis(laxis, qzr), base_q, reference_R=htR[s])
        if laxis is None:
            # ROUND object (no preferred grasp axis): the wrist ROLL is FREE, so orientation_aware_grasp_quat
            # returned base_q WITHOUT aligning the roll. Left free, the roll varies with the reach direction and
            # can land ~pi from the HOME wrist roll -> the carry inherits it and the go_home then FLIPS joint_6
            # (a ~3rad snap on the empty return, verified on 3/12 apple demos). Snap the round grasp roll to the
            # branch CLOSEST to the home wrist orientation (same branch-pick transport_quats uses), so the whole
            # pick->carry->home chain stays near the home roll and joint_6 never flips. (No-op for cube/elongated,
            # whose roll is already determined by their ref axis -> their motion is unchanged.)
            gq = transport_quats(gq, reference_quat=_wxyz_from_R(htR[s]))[0]
        return gq

    # ---- WRIST-MARGIN-AWARE grasp tilt selection (the RoboLab-faithful natural-posture fix) -------------------
    # WHY: RoboLab's pick-place is NATURAL because ``plan_pick_place`` PREFERS top-down but RELAXES to the
    # smallest tilt (toward the reach direction) that the arm can reach -- ``reachable_grasp_quat`` +
    # ``_TILT_STEPS_DEG=(0,8,16,24,32)`` (robolab/skills/pick_place.py). The Genesis collector had this relax
    # idea STUBBED OUT: it hard-coded ``tilt=0`` (pure top-down). With the object table at the ARM-BASE level
    # (z=0.25=base z, RoboLab-faithful), a pure top-down GRASP is comfortable (EEz~0.29) but the straight-up
    # +LIFT (EEz~0.47) is NEAR-SINGULAR: wrist1 SATURATES at +1.57 and the elbow COLLAPSES (~0.4) -- the arm
    # over-stretches to a near-vertical extension. (Measured: scripts/grasp/temp/tilt_lift_margin_probe.py.)
    # FIX (RoboLab-faithful, generalises to apple/banana/pen/tennis_ball): restore the prefer-top-down relax
    # loop, but make the reachability test the one RoboLab's omitted -- a WRIST MARGIN through the LIFT, not
    # just IK-solvability at the grasp centre. We pick, per env, the SMALLEST tilt whose grasp orientation keeps
    # wrist1 OFF its limit (|j4| <= WRIST_LIMIT - WRIST_MARGIN) at the lifted pose (the binding frame; the grasp
    # is already comfortable). This uses the SAME batched Genesis IK as the run (no IK-solver change, no
    # per-env loop in the hot path), so the batched parallel structure + the proven solver stay intact.
    # Per-object CAP on the relax ladder (MAX_TILT env / spec.max_grasp_tilt_deg). Default 40 = the full ladder,
    # used by EVERY object now -- the round-object ejection that once forced a 0-cap is fixed at the source
    # (noslip_iterations in firm_rigid_options), so apple/tennis relax-tilt exactly like the cube. Kept as a knob
    # for ablation; restricts the ladder only, never the SCORING.
    _MAX_TILT = float(os.environ.get("MAX_TILT", getattr(spec, "max_grasp_tilt_deg", 40.0)))
    _GRASP_TILT_STEPS_DEG = tuple(t for t in (0.0, 8.0, 16.0, 24.0, 32.0, 40.0) if t <= _MAX_TILT + 1e-6)
    WRIST_LIMIT = 1.57                                      # joint_4 (wrist1) hard limit (URDF)
    WRIST_MARGIN = float(os.environ.get("WRIST_MARGIN", "0.17"))   # keep |j4| <= 1.40 -> >=0.17 off the limit
    ELBOW_MIN = float(os.environ.get("ELBOW_MIN", "1.05"))   # keep j3(elbow) >= this (bent; home 2.30, limit [0,3.14])

    def _posture_at_pick(tilt_deg):
        """Batched: for the given per-env grasp tilt, return (grasp_quat, worst |j4| wrist, worst-low j3 elbow)
        over the PICK's binding frames -- the PRE-GRASP (a reached-out approach) and the LIFT -- on the
        EXECUTION-FAITHFUL warm-start chain. The executor reaches the lift by tracking CONTINUOUSLY from the
        grasp/close config UP the +Z column (warm-started single-sample IK), so the lift lands on the branch
        C1-continuous from the grasp -- NOT the globally-best branch a cold solve from home would pick. We
        replicate that (pre-grasp warm from home as the run starts, then grasp, then lift warm from grasp), so
        the predicted j4/j3 match the demo (a cold solve under-predicts the over-stretch + stops relaxing early).
        A forward tilt BOTH unsaturates the wrist AND bends the elbow, so one relax ladder satisfies both."""
        gq = np.stack([grasp_quat_at(i, gc[i, 0], gc[i, 1], tilt_deg[i]) for i in range(N)]).astype(np.float64)
        tz = np.stack([_R_from_wxyz(q) @ np.array([0, 0, 1.0]) for q in gq])   # per-env approach axis (ee +Z)
        pre_tool = gc - APP * tz                           # the PRE-GRASP waypoint (back off along approach axis)
        lift_tool = gc.copy(); lift_tool[:, 2] += LIFT     # the lift waypoint the collector actually commands
        qpre = solve(pre_tool, gq)                         # PRE-GRASP solve (warm from home, as the run starts)
        solve(gc.copy(), gq)                               # GRASP solve -> warms the IK at the grasp config
        qlift = solve(lift_tool, gq)                       # LIFT solve, continuous from the grasp branch
        j4 = np.maximum(np.abs(qpre[:, 3]), np.abs(qlift[:, 3]))   # worst wrist over the pick
        j3 = np.minimum(qpre[:, 2], qlift[:, 2])                   # worst-low elbow over the pick
        return gq, j4, j3

    def select_grasp_tilt():
        """Per-env grasp tilt: the SMALLEST of ``_GRASP_TILT_STEPS_DEG`` whose pick keeps the wrist OFF its limit
        (|j4| <= WRIST_LIMIT-WRIST_MARGIN) AND the elbow BENT (j3 >= ELBOW_MIN) through the pre-grasp + lift.
        Returns (tilt_deg:(N,), grasp_quat:(N,4)). Starts every env at top-down (0) and only relaxes the envs
        that still over-stretch -- so a comfortable env keeps the cleanest pure-vertical grasp, and only the
        over-stretched envs tilt forward (exactly RoboLab's minimal-tilt policy)."""
        tilt = np.zeros(N)
        gq, j4, j3 = _posture_at_pick(tilt)
        need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)   # wrist saturating OR elbow too straight
        for t in _GRASP_TILT_STEPS_DEG[1:]:
            if not need.any():
                break
            tilt[need] = t
            gq_t, j4_t, j3_t = _posture_at_pick(tilt)
            gq[need] = gq_t[need]                           # adopt the relaxed-tilt quat for the still-needy envs
            j4[need] = j4_t[need]; j3[need] = j3_t[need]
            need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)   # re-test; the rest are already OK
        print(f"[COLLECT] grasp tilt (RoboLab relax, wrist-margin {WRIST_MARGIN:.2f}, elbow>={ELBOW_MIN:.2f}, "
              f"lift={LIFT}): per-env deg={np.round(tilt, 0).astype(int).tolist()} ; "
              f"predicted pick wrist|j4|max={float(j4.max()):.3f} elbow|j3|min={float(j3.min()):.3f} "
              f"(still-over-stretched={int(need.sum())})", flush=True)
        return tilt, gq.astype(np.float64)

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

    # FAST = a DEBUG-ONLY knob (grasp/posture/penetration iteration): skip the Nyx path-traced render in the run
    # loop (the slow part, ~6x the sim cost) AND skip ALL video/tile/montage/preview WRITING (see the write
    # section below). It writes ONLY demos.hdf5 (the numeric data the posture/penetration checks read) + the
    # [COLLECT] prints -- NO .mp4/.png at all (the old FAST wrote blank placeholder frames as "thin black stripe"
    # junk videos; that is gone). The HDF5 state/actions + the grasp/place/penetration METRICS are unaffected
    # (they read sim state, not pixels). NEVER use it for a real collection (there are no policy-cam videos).
    # Unset (default) = full photoreal render + all videos/tiles written.
    FAST = bool(os.environ.get("FAST"))

    def on_step(t, full, labels_t):
        record_state(full)
        pen_tracker.update()                                       # fold this step's contact buffer into the max
        if not FAST:                                               # FAST: skip the render entirely (no pixels kept)
            views = stage.render()
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
    # the running per-env cube REST z (table top + half cube) -> the predicted-grasp z for the re-grasp.
    cube_rest_z = root0[:, 2].copy()
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

    def approach_close_lift_wps(start_p, start_q, rise_p, cx, cy, env_i, regrasp_tilt=None,
                                start_grip=OPEN):
        """Build a SMOOTH, singularity-robust re-grasp at the cube's CURRENT pose (cx,cy) FROM the arm's current
        pose -- the ONE builder for BOTH the CHASE pivot and the RETRY re-grasp. ``env_i`` selects the per-env
        arm side / yaw for the grasp quat.

        ``regrasp_tilt`` (TASK 0 fix): the grasp tilt to use AT the shoved pose. When the cube is shoved to a NEW
        pose, the ORIGINALLY-planned ``grasp_tilt[env_i]`` (chosen for the OLD pose) can SATURATE the wrist at the
        new reach (verified: DISTURB=0.5 -> 2/16 recovery demos hit j4=1.57). So the caller RE-RUNS the
        wrist-margin relax-ladder (``select_grasp_tilt_at``) at the shoved (cx,cy) and passes the result here;
        ``None`` falls back to the original tilt (used by no current call site, kept for safety).

        ``start_grip`` (the NO-HOLD fix): the gripper value the arm arrives at the start pose holding. For the
        AFTER-CLOSE recovery the arm arrives with the gripper CLOSED-on-nothing (start_grip=CLOSE); for the CHASE
        the gripper is already OPEN through the aborted approach (start_grip=OPEN). The RISE segment commands the
        gripper OPEN, so densify RAMPS it CLOSE->OPEN *WHILE THE ARM IS RISING* -- the reopen overlaps the rise
        MOTION instead of being a static hold-pose-while-gripper-opens dwell at the apex (the 2.2 s freeze the
        owner rejects). The arm is never static while the hand reopens.

        NO-HOLD re-architecture (the ZERO-mid-trajectory-hold fix): the OLD builder REORIENTED LOW as a SEPARATE
        pure-rotation segment (``reorient`` waypoint at the start pose), THEN rose, THEN descended. With the
        gripper reopening across the lift->start boundary that produced a static-arm REOPEN dwell at the apex, AND
        a second static dwell on the (often near-no-op) low reorient -- a ~22-frame freeze. The fix OVERLAPS the
        reorient AND the reopen WITH the RISE: the rise segment translates UP, SLERPs the wrist to the re-selected
        comfortable grasp quat ``gqi``, and ramps the gripper OPEN, ALL in one continuously-moving densified
        segment (no ``lin<1e-4 & ang<1e-3`` dwell -> no static run). The descend then completes at ``gqi`` (pure
        translation). The arm keeps MOVING from the moment it leaves the start pose until the brief grasp-CLOSE
        settle at the re-grasp -- the only remaining dwell, identical to the clean grasp.

        TASK-0 wrist comfort (the recovered-demo j4=1.57 saturation): the rise apex is a HIGH EE pose, and a
        recovery at a HIGH table already starts the close near EEz~0.44 -- a +10cm rise at a steep (old) top-down
        orientation SATURATES the wrist. We still CAP the apex EE height (rise only enough to clear the cube for a
        clean re-descend, not into the saturation band) AND reach the apex ALREADY at the relaxed tilt ``gqi`` (the
        rise SLERPs to it), so the high-EE rise + descend happen off the wrist limit. The branch-flip the old split
        guarded against is avoided by ``select_grasp_tilt_at(apex_z=...)`` having vetted the wrist along the apex->
        descend column at ``gqi``, and by the warm-started single-sample IK tracking the rise continuously (the
        reorient is a moderate <=~24deg tilt change, well inside one branch -- verified by the all-phase dq gate)."""
        gci = np.array([cx, cy, cube_rest_z[env_i] + grasp_dz])
        tlt = float(grasp_tilt[env_i] if regrasp_tilt is None else regrasp_tilt)
        gqi = grasp_quat_at(env_i, cx, cy, tlt)                 # re-selected wrist-margin tilt at the shoved pose
        tzi = _R_from_wxyz(gqi) @ np.array([0, 0, 1.0])
        # CAP the apex: rise only enough above the cube to clear it for the re-descend (RETRY_RISE), but never so
        # high that the EE climbs into the wrist-saturation band. The apex tool-z is capped at the cube grasp z +
        # a modest clearance so EEz stays comfortable even at a high table (the over-stretch ceiling fix, applied
        # to the recovery rise). The rise reaches the apex already at the relaxed tilt ``gqi``.
        #
        # NO-HOLD apex PLACEMENT (the critical geometric fix): the apex sits over the PREDICTED grasp XY (not the
        # OLD/start XY) at a height that clears the cube (capped to cube_grasp_z + RETRY_RISE). This makes the
        # start->apex segment a REAL translation FOR BOTH branches -- and that is what lets the reopen+reorient
        # ride MOTION instead of freezing:
        #   * CHASE       (start LOW at the old pose, grip already OPEN): apex is up AND over to the predicted XY
        #                 -> a real diagonal transit.
        #   * AFTER-CLOSE (start already LIFTED at the old pose, grip CLOSED): the apex z == the empty-lift z, but
        #                 the apex is OVER the predicted XY, so the segment is a real HORIZONTAL transit (the arm
        #                 slides across to above the new pose). If the apex coincided with the start (old XY, same
        #                 z), the segment would be a zero-length DWELL and the reopen/reorient would freeze there
        #                 -- exactly the 22-frame apex hold this fix removes. Aiming the apex at the predicted XY
        #                 guarantees motion to carry the gripper reopen + the wrist reorient.
        apex = gci.copy()                                            # over the PREDICTED grasp XY ...
        apex[2] = min(float(rise_p[2]), float(gci[2]) + RETRY_RISE)  # ... at a cube-clearing, non-saturating height
        return [
            ("start",    start_p,            start_q, start_grip),
            # (1) TRANSIT to the apex OVER the predicted pose: translate up/over, SLERP the wrist start_q->gqi
            # (reorient folded in), and ramp the gripper start_grip->OPEN -- ALL in this one moving segment. The
            # apex aims at the predicted XY so the segment always MOVES (never a dwell) -> no static reopen/reorient
            # hold; the arm keeps moving while the hand reopens and the wrist reorients.
            ("rise",     apex,               gqi,     OPEN),
            ("pre",      gci - APP * tzi,    gqi,     OPEN),        # (2) pure-translation descend to pre-grasp
            ("at",       gci,                gqi,     OPEN),
            ("close",    gci,                gqi,     CLOSE),       # the ONLY dwell: the brief grasp-CLOSE settle
            ("lift",     gci + [0, 0, LIFT], gqi,     CLOSE),
        ], gqi

    Tlist = []
    recovery_attempts = np.zeros(N, np.int32)
    disturb_phase = np.array(["none"] * N, dtype=object)          # "before_close" | "after_close" | "none"
    chased = np.zeros(N, bool)                                    # informed-before-close -> pivoted/chased
    # Per-env grasp orientation: prefer top-down, relax to the smallest forward tilt that keeps the WRIST off
    # its limit through the LIFT (the RoboLab-faithful natural-posture fix; ``select_grasp_tilt`` above). The
    # chosen per-env tilt is kept so the disturbance chase/retry re-grasps reuse it (stay natural too).
    grasp_tilt, gqA = select_grasp_tilt()
    # rebuild gqA at the settled grasp CENTRE (groot0) with the chosen tilt -- select_grasp_tilt built it at gc
    # (== groot0 + grasp_dz in xy, identical xy), so this just re-keys to groot0's xy for the cube (no-op for
    # the cube where grasp_dz has no xy component; faithful for an offset banana whose gc xy differs from root).
    gqA = np.stack([grasp_quat_at(i, groot0[i, 0], groot0[i, 1], grasp_tilt[i]) for i in range(N)]).astype(np.float64)

    # ---- WRIST-MARGIN-AWARE carry/place tilt (the SAME natural-posture fix, for the OTHER binding frame) ----
    # The carry HOVERS over the bowl at ``bxyz + PAPP`` -- a HIGH EE pose (the bowl sits on the same low table
    # so the hover climbs to EEz~0.44-0.51) where a near-top-down carry ALSO saturates the wrist + collapses
    # the elbow (the lift-vs-carry region was the residual over-stretch after the grasp tilt fixed the pick).
    # We relax the CARRY tilt toward the bowl reach by the same prefer-minimal ladder until wrist1 is off its
    # limit at the over-bowl hover -- exactly RoboLab's ``reachable_place_quat`` (prefer top-down transport,
    # relax minimal tilt), extended with the explicit wrist margin. Per-env scalar, evaluated once on the
    # batched IK (no solver change). Smoothness is preserved: transport_quats still picks the carry branch
    # CLOSEST to the grasp orientation, so the lift->carry re-yaw stays a small rotation.
    def _carry_posture(tilt):
        """Batched carry-over-bowl wrist + elbow for a per-env carry tilt, on the execution-faithful warm chain
        (warm at the in-bowl config, then the over-bowl hover -- the executor reaches the hover via the carry)."""
        over_bowl = bxyz.copy(); over_bowl[:, 2] += PAPP        # the carry hover the collector commands
        cq = np.stack([cquat(i, gqA[i], tilt[i]) for i in range(N)]).astype(np.float64)
        solve(bxyz.copy(), cq)                                  # warm the IK at the lowered-in-bowl config first
        q = solve(over_bowl, cq)
        return cq, np.abs(q[:, 3]), q[:, 2]                     # carry quat, |wrist j4|, elbow j3

    def select_place_tilt():
        tilt = np.full(N, 8.0)                                  # the locked gentle carry tilt (top-down-ish)
        cq, j4, j3 = _carry_posture(tilt)
        need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)
        for t in (16.0, 24.0, 32.0, 40.0):                     # relax toward the bowl (prefer the small tilt)
            if not need.any():
                break
            tilt[need] = t
            cq_t, j4_t, j3_t = _carry_posture(tilt)
            cq[need] = cq_t[need]; j4[need] = j4_t[need]; j3[need] = j3_t[need]
            need = (j4 > (WRIST_LIMIT - WRIST_MARGIN)) | (j3 < ELBOW_MIN)
        print(f"[COLLECT] place tilt (RoboLab relax, wrist-margin {WRIST_MARGIN:.2f}, elbow>={ELBOW_MIN:.2f}): "
              f"per-env deg={np.round(tilt, 0).astype(int).tolist()} ; "
              f"predicted carry wrist|j4|max={float(j4.max()):.3f} elbow|j3|min={float(j3.min()):.3f} "
              f"(still-over-stretched={int(need.sum())})", flush=True)
        return tilt
    place_tilt = select_place_tilt()

    # ---- TASK 0: RE-SELECT the grasp/place tilt AT the SHOVED cube pose for the disturbance chase/recovery -----
    # The chase (seg2) + recovery (seg3) re-grasp the cube at its NEW (shoved) pose. Reusing the ORIGINAL
    # ``grasp_tilt``/``place_tilt`` (chosen for the OLD pose + OLD reach) can SATURATE the wrist there (verified:
    # DISTURB=0.5 N=16 -> 2/16 recovery demos j4=1.57). FIX: re-run the EXACT SAME wrist-margin relax-ladder
    # (same WRIST_LIMIT/WRIST_MARGIN/ELBOW_MIN, same warm-start pre+lift / carry probe -- the SCORING is
    # unchanged) at the shoved grasp centre / re-grasp orientation, per env. These are single-env wrappers around
    # the same batched ``solve()`` probe used by select_grasp_tilt/_carry_posture (no IK-solver change, the
    # relax policy is identical), so the re-grasp stays as natural as the first grasp.
    def _posture_at_pick_env(env_i, gci, tilt_deg, apex_z=None):
        """Single-env (gq, |j4|, j3) over the PICK binding frames at grasp centre ``gci`` and the given tilt -- the
        env_i row of ``_posture_at_pick``, so the SCORING is byte-identical. ``apex_z`` (the recovery rise apex
        height): the recovery/chase re-grasp does rise->REORIENT-at-apex->descend->close->lift, so the high apex
        REORIENT (top-down at high EEz) is ALSO a binding frame that can saturate the wrist (the same high-EE
        saturation the lift fix addressed). When given, the apex pose at ``gqi`` is folded into the worst-case so
        the relax ladder picks a tilt comfortable AT THE APEX too -- otherwise the probe (pre+lift only) approves
        a tilt that the recovery's apex reorient still saturates (the 2026-06-20 recovered-demo j4=1.57 bug)."""
        gq = grasp_quat_at(env_i, gci[0], gci[1], float(tilt_deg)).astype(np.float64)
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

    def select_grasp_tilt_at(env_i, gci, apex_z=None):
        """Smallest of _GRASP_TILT_STEPS_DEG keeping |j4|<=limit-margin AND j3>=ELBOW_MIN through the pick at the
        SHOVED grasp centre ``gci`` -- the per-env relax ladder of ``select_grasp_tilt`` (identical thresholds).
        ``apex_z`` folds the recovery/chase rise-apex reorient into the worst-case (see _posture_at_pick_env)."""
        for t in _GRASP_TILT_STEPS_DEG:
            _, j4, j3 = _posture_at_pick_env(env_i, gci, t, apex_z=apex_z)
            if (j4 <= (WRIST_LIMIT - WRIST_MARGIN)) and (j3 >= ELBOW_MIN):
                return float(t)
        return float(_GRASP_TILT_STEPS_DEG[-1])                 # none fully clears -> the largest (most-relaxed) tilt

    def select_place_tilt_at(env_i, gqi):
        """Smallest carry tilt keeping the carry-over-bowl wrist/elbow comfortable at the re-grasp orientation
        ``gqi`` (the carry quat re-yaws to gqi's branch) -- the per-env ladder of ``select_place_tilt``."""
        over_bowl = bxyz.copy(); over_bowl[:, 2] += PAPP
        for t in (8.0, 16.0, 24.0, 32.0, 40.0):
            cqi = cquat(env_i, gqi, float(t))
            cqN = np.tile(cqi, (N, 1)).astype(np.float64)
            solve(bxyz.copy(), cqN); q = solve(over_bowl, cqN)
            if (abs(float(q[env_i, 3])) <= (WRIST_LIMIT - WRIST_MARGIN)) and (float(q[env_i, 2]) >= ELBOW_MIN):
                return float(t)
        return 40.0

    def pick_wps(i):
        """home -> pre -> at -> at -> close -> lift for env i: the orientation-aware top-down approach + a FIRM
        GR100 close (the object's own collision stops the claws; the high-kp PD holds the force) + a gentle lift.
        This is OBJECT-AGNOSTIC: the cube, the round apple/tennis, and the elongated banana/pen all use this same
        path. (Round objects USED to need a special deep-seat + reorient approach to avoid ejecting under the firm
        pinch, but that was a symptom of the leaky friction cone -- fixed at the source by noslip_iterations in
        firm_rigid_options; round objects now grasp top-down+tilt exactly like the cube, no special machinery.)"""
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

    def pick_wps_nodwell(i):
        """The pick with a SINGLE pre-close ``at`` (home->pre->at->close->lift), used by the DISTURBANCE path's
        UNDISTURBED envs. The clean ``pick_wps`` has a DOUBLE ``at,at`` -- a pointless 8-frame static HOLD where
        the arm idles at the grasp pose with the gripper still OPEN before the close, on TOP of the actual close
        dwell (so the close region reads ~14-16 static frames -> over the no-hold gate). The chase path already
        uses a single ``at``; this gives the disturbance run's undisturbed envs the SAME single-at close dwell
        (~8 frames, just the grasp-CLOSE settle) so EVERY env in a disturbed batch holds <= the gate. The CLEAN
        DISTURB=0 path keeps ``pick_wps`` (double-at) byte-for-byte -- it never reaches this branch."""
        gci = gc[i]
        tzi = _R_from_wxyz(gqA[i]) @ np.array([0, 0, 1.0])
        return [
            ("start", home_tool[i],       home_tquat[i], OPEN),
            ("pre",   gci - APP * tzi,    gqA[i], OPEN),
            ("at",    gci,                gqA[i], OPEN),
            ("close", gci,                gqA[i], CLOSE),
            ("lift",  gci + [0, 0, LIFT], gqA[i], CLOSE),
        ]

    def place_tail(i, liftp, gqi):
        """From the LIFTED grasp pose ``liftp`` (held at ``gqi``): re-yaw to the carry orientation, carry over the
        bowl, lower in, release, retract, GO HOME. This is APPENDED to the same continuous per-env waypoint stream
        (the pick, or a recovery re-grasp) so each env runs pick->place->home as ONE smooth trajectory and
        TERMINATES at home -- no staged barrier, no mid-air wait for other envs."""
        cqi = cquat(i, gqi, place_tilt[i])                        # wrist-margin-aware carry tilt (natural posture)
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
        # ===== DISTURBANCE PATH v3: a SINGLE continuous per-env trajectory, ZERO mid-trajectory holds =====
        # The old staged seg1/seg2/seg3 path FROZE the normal envs LIFTED IN THE AIR while the chase/recover envs
        # pivoted (a per-env barrier the owner forbids). v3 pre-plans EACH env's FULL trajectory up front and runs
        # it in ONE continuous pass, EXACTLY like the clean path -- a clean env terminates early (shorter demo), a
        # chased/recovered env's trajectory is simply LONGER, but NO env ever holds waiting for another.
        #
        # The chase/recover re-grasp must aim at the cube's SHOVED pose, which physically happens DURING the run.
        # Because the shove is a god-mode impulse WE control (known velocity + the per-env object friction), we
        # PREDICT the shoved resting pose at BUILD TIME: rest = fire_xy + unit(v) * |v|^2 / (2*mu*g) (Coulomb
        # slide; mu = the object's friction). Calibrated to ~1.1cm mean / 1.7cm max error on the cube
        # (scripts/temp/calib_shove_predict.py) -- well within the open-claw span, so the re-grasp (planned at the
        # predicted pose) still cages the real cube. The shove STILL fires via the during_step hook (a real
        # physics slide); we only PREDICT where it lands so the whole per-env path is known up front (required for
        # the single continuous run). chase-vs-after-close is ALSO resolved at build time (it depends only on the
        # fire-fraction band + the sense-delay, no sim read), so the matching trajectory SHAPE is pre-planned.
        # ``DISTURB=0`` (default) never reaches here.

        # (1) approach length (control steps) -> resolve chase-vs-after-close at build time. The approach is the
        # SAME home->pre->at->at the clean pick uses; its densified length sets the window the close lands at.
        wpsApp = [pick_wps(i)[:4] for i in range(N)]              # home -> pre -> at -> at (open, the approach)
        _, _, _, lblApp, T_App = ex.plan(wpsApp)
        chased = dspec.will_be_informed_before_close(T_App)      # informed BEFORE the close -> chase (else recover)
        afterclose = dspec.disturbed & ~chased                   # informed only after the close -> close-on-nothing
        disturb_phase[chased] = "before_close"
        disturb_phase[afterclose] = "after_close"
        print(f"[COLLECT] disturbance v3 (single-pass): disturbed {int(dspec.disturbed.sum())} env(s) "
              f"{np.where(dspec.disturbed)[0].tolist()} (prob={DIST_PROB}, approach_T={T_App}); "
              f"before_close(chase)={int(chased.sum())} after_close(recover)={int(afterclose.sum())}", flush=True)

        # (2) PREDICT each disturbed env's shoved grasp-centre xy (build-time physics; the owner's Coulomb-slide
        # formula d = |v|^2 / (2*mu*g)). ``mu`` is the EFFECTIVE slide friction of the object on the table -- NOT
        # the nominal spec friction (1.0) but the value CALIBRATED against the real shove on the firm table
        # (scripts/temp/calib_shove_predict.py): at mu=0.85 the predicted rest xy matches the REAL settled xy to
        # ~1.2cm mean / 1.6cm max (mu=1.0 slightly UNDER-predicts the slide -> the re-grasp lands a touch short).
        # SHOVE_MU overrides for tuning; defaults to the calibrated 0.85. The shove fires early in the approach
        # while the object is still settled, so the fire xy is the settled grasp-centre xy (groot0). Undisturbed
        # envs keep their settled xy (no shove).
        mu_obj = float(os.environ.get("SHOVE_MU", 0.85))
        shoved_xy = dspec.predict_shoved_xy(groot0[:, :2], mu_obj)   # (N,2) predicted rest xy of the grasp centre

        # (3) build EACH env's FULL trajectory up front (one continuous waypoint stream that DENSIFIES to its own
        # length -> per-env natural termination, no padding hold mid-trajectory):
        #   * undisturbed  : home->pre->at->close->lift->place->home  (single-at pick_wps_nodwell + place_tail)
        #   * chased       : home->pre->at(old) -> re-aim(rise[reorient+reopen folded in])->pre->at(PREDICTED)->
        #                    close->lift->place->home  (the close happens ONLY at the predicted pose)
        #   * after-close  : home->pre->at->close(empty,old)->lift(empty) -> recover(rise[reorient+REOPEN folded in
        #                    over the predicted pose]->pre->at->close->lift) ->place->home  (the old-pose close
        #                    grabs nothing -> recover re-grasps; the gripper REOPENS *during* the rise, not frozen)
        # The chase/recover RE-GRASP reuses the approach_close_lift_wps builder, whose RISE segment translates to
        # the apex OVER the predicted pose WHILE slerping the wrist to gqi (reorient) and ramping the gripper open
        # (reopen) -- one continuously-MOVING segment, so there is NO static reopen/reorient hold (the no-hold fix);
        # the descend->at is pure translation. The grasp/carry tilt is RE-SELECTED at the predicted shoved pose
        # (the same wrist-margin relax ladders), and the warm single-sample IK tracks the rise without a branch flip.
        recovery_attempts[afterclose] = 1                        # the after-close env does one recover re-grasp

        full_wps = []
        for i in range(N):
            if not dspec.disturbed[i]:
                # undisturbed env: the normal pick+place, but with the SINGLE-at pick (pick_wps_nodwell) so its
                # close dwell stays at the gate (~8 frames) -- the clean DISTURB=0 path (separate branch above)
                # still uses the double-at pick_wps byte-for-byte.
                full_wps.append(pick_wps_nodwell(i) + place_tail(i, gc[i] + [0, 0, LIFT], gqA[i]))
                continue
            # end of the FIRST approach = the "at" OLD pose (a known waypoint: gc[i] held at gqA[i]); chain the
            # re-grasp off it WITHOUT a sim read.
            old_p = gc[i].copy()
            old_q = gqA[i].copy()
            px, py = float(shoved_xy[i, 0]), float(shoved_xy[i, 1])
            gci_pred = np.array([px, py, cube_rest_z[i] + grasp_dz])
            # rise apex above the OLD pose, capped to the predicted-grasp z + RETRY_RISE (matches the builder cap)
            sp = old_p.copy(); sp[2] = max(old_p[2], cube_rest_z[i] + grasp_dz) + RETRY_RISE
            apex_z = min(float(sp[2]), float(gci_pred[2]) + RETRY_RISE)
            rt = select_grasp_tilt_at(i, gci_pred, apex_z=apex_z)    # natural posture at the predicted shoved pose
            if chased[i]:
                # CHASE: the first approach reaches the old pose but NEVER closes (gripper stays OPEN through it);
                # immediately re-aim to the predicted pose and close THERE. The first approach uses a SINGLE "at"
                # (NOT the clean pick's double "at,at" pre-close settle dwell) -- the chase never closes at the old
                # pose, so a settle dwell there would be a pointless mid-trajectory HOLD (the arm idling low with
                # the gripper open before it rises to chase). Flowing home->pre->at(old) straight into the re-aim
                # (rise[reorient folded in]->pre(pred)->at->close->lift) keeps the whole chase continuously moving:
                # the builder's RISE segment carries the reorient (start_q->gqi SLERP) WHILE translating up, so
                # there is NO static reorient/rise dwell -- the arm flows at(old) straight into the rising re-aim.
                regrasp, gqi = approach_close_lift_wps(old_p, old_q, sp, px, py, i, regrasp_tilt=rt,
                                                       start_grip=OPEN)
                approach = pick_wps(i)[:3]                        # home->pre->at(old), single, all OPEN (no settle)
                wp = approach + regrasp[1:]                       # regrasp[0]=="start"==at(old); drop the dup
            else:
                # AFTER-CLOSE: the first approach CLOSES on the old (now-empty) pose -> grabs nothing -> recover.
                # The empty pick uses a SINGLE "at" (not the clean pick's "at,at" pre-close settle dwell): the
                # empty close on nothing needs no extra pre-close settle, so a second at would just be a pointless
                # static HOLD at the old pose. Then the recover re-grasp at the predicted pose, then place.
                empty_pick = pick_wps_nodwell(i)                  # home->pre->at->close(empty)->lift(empty)
                lift_old_p = empty_pick[-1][1]                    # the empty lift pose (old gc + LIFT)
                # recover transits from the empty lift over to the predicted pose, reopening + reorienting EN ROUTE.
                # start_grip=CLOSE: the arm arrives at the empty lift with the gripper CLOSED-on-nothing. The
                # builder's RISE segment ramps it CLOSE->OPEN WHILE rising (the reopen overlaps the rise MOTION),
                # so there is NO static reopen-while-frozen dwell at the apex (the 2.2 s freeze the owner rejects).
                regrasp, gqi = approach_close_lift_wps(lift_old_p, gqA[i], sp, px, py, i, regrasp_tilt=rt,
                                                       start_grip=CLOSE)
                # regrasp[0]=="start"==the empty-lift pose (same pos/quat/grip=CLOSE as empty_pick[-1]); DROP it so
                # the empty lift flows STRAIGHT into the rise (no duplicate-waypoint dwell). The rise then ramps the
                # gripper open while moving up -- the reopen rides the rise, never a static frozen-apex dwell.
                wp = empty_pick + regrasp[1:]
            gqA[i] = gqi
            place_tilt[i] = select_place_tilt_at(i, gqi)          # carry tilt at the new (predicted) grasp quat
            full_wps.append(wp + place_tail(i, wp[-1][1], gqi))   # wp[-1] = the re-grasp's final ("lift", pos, ..)
            if os.environ.get("TILT_DEBUG"):
                tag = "chase" if chased[i] else "recover"
                print(f"[TILT_DEBUG] {tag} env{i}: orig grasp_tilt={grasp_tilt[i]:.0f} -> reselect={rt:.0f} "
                      f"place_tilt={place_tilt[i]:.0f} (pred shoved xy={np.round([px,py],3).tolist()})", flush=True)

        # (4) per-env ABSOLUTE fire-step within EACH env's OWN first approach (located by the densified label
        # stream: the last step whose label is still in the open approach -- pre/at -- BEFORE the first close).
        # Firing at ``fire_frac`` of the way into that pre-close window lands the shove early enough that (a) a
        # chase env can re-aim and (b) an after-close env's cube has slid away by its (old-pose) close. This
        # replaces the staged reset_window(SHARED approach_T) with a per-env step from the SINGLE plan.
        _, _, _, lblFull, T_full = ex.plan(full_wps)
        fire_step_abs = np.zeros(N, np.int32)
        for i in range(N):
            if not dspec.disturbed[i]:
                continue
            pre_close = [t for t in range(T_full)
                         if lblFull[t][i] in ("pre", "at") ]       # this env's first open-approach window
            # restrict to BEFORE the first 'close' label (the empty close for after-close; the only close otherwise
            # appears after the re-aim, which is also fine to fire before)
            close_ts = [t for t in range(T_full) if lblFull[t][i] == "close"]
            if close_ts:
                pre_close = [t for t in pre_close if t < close_ts[0]]
            if pre_close:
                j = int(round(float(dspec.fire_frac[i]) * (len(pre_close) - 1)))
                fire_step_abs[i] = pre_close[max(0, min(j, len(pre_close) - 1))]
        dspec.arm_single_pass(fire_step_abs)

        # (5) run the WHOLE thing as ONE continuous batch -- the shove fires via the during_step hook at each
        # env's fire-step (a real physics slide). NO mid-pass barrier; every env runs to its own home + terminates.
        Tlist.append(run_phase(full_wps, settle_steps=40,
                               during_step=lambda t, lab: dspec.tick(t), tag="pick-place-home(disturbed)"))
        fired = dspec.fired_any()
        print(f"[COLLECT] disturbance: shoved {int(fired.sum())} env(s) {np.where(fired)[0].tolist()} "
              f"(fire steps={fire_step_abs[dspec.disturbed].tolist()})", flush=True)
        if os.environ.get("DISTURB_DIAG"):
            cf = np_(cube.get_pos())
            for i in np.where(dspec.disturbed)[0]:
                # predicted shoved xy vs the cube's FINAL xy (the re-grasp aimed at the prediction; the cube
                # should now be in/near the bowl if placed, or near the predicted pose if the grasp missed).
                pe = np.hypot(cf[i, 0] - shoved_xy[i, 0], cf[i, 1] - shoved_xy[i, 1]) * 100
                print(f"[DISTURB_DIAG] env{i} {'chase' if chased[i] else 'recover'}: "
                      f"settled_xy={np.round(groot0[i,:2],3).tolist()} pred_shoved={np.round(shoved_xy[i],3).tolist()} "
                      f"final_cube_xy={np.round(cf[i,:2],3).tolist()} |final-pred|={pe:.1f}cm "
                      f"bowl_xy={np.round([bowx[i],bowy[i]],3).tolist()}", flush=True)

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
    if not FAST:                                                  # FAST writes ONLY demos.hdf5 (no videos)
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
            if not FAST:                                        # the sensor-only policy stream (trimmed to Te)
                for nm in ("cam_side", "cam_lw", "cam_rw"):
                    vid = np.stack([cam_steps[nm][t][e] for t in range(Te)])
                    iio.imwrite(os.path.join(vdir, nm, f"demo_{e}.mp4"), vid, fps=12, codec="libx264")
    # FAST: skip ALL video/tile/montage/preview writing entirely (the per-cam demo mp4s above, the third-person
    # tile, the fourview clips, the distractor preview PNGs) -- FAST writes ONLY demos.hdf5 + the [COLLECT] prints
    # (no pixels were rendered, so there is nothing to write; this removes the old blank-frame "black stripe" junk).
    chosen = []
    if not FAST:
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
    # visible alongside the cube+bowl. Stacks [third | side] over [wrist-L | wrist-R]. (FAST: skipped -- no pixels.)
    if dist_ents and not FAST:
        d0 = dist_xy0                                            # (K,N,2) settled distractor XY
        spread = d0[:, :, 0].std(0) + d0[:, :, 1].std(0) if d0.shape[0] > 1 else np.zeros(N)
        pe = int(np.argmax(spread))                             # the most-spread (clearest clutter) env
        q0 = {nm: _label(cam_steps[nm][0][pe], lab[nm]) for nm in lab}
        preview = np.vstack([np.hstack([q0["third"], q0["cam_side"]]),
                             np.hstack([q0["cam_lw"], q0["cam_rw"]])])
        iio.imwrite(os.path.join(out_dir, "distractors_preview.png"), preview)
        iio.imwrite(os.path.join(out_dir, "distractors_third.png"), cam_steps["third"][0][pe])
        print(f"[COLLECT] distractor preview env={pe} ({dist_names}) -> {out_dir}/distractors_preview.png", flush=True)

    print(f"[COLLECT] wrote {N} demos ({int(placed.sum())} placed) -> {data_dir}/demos.hdf5"
          + ("  [FAST: hdf5 only, no videos/tiles/preview]" if FAST else f" + videos ; tiles -> {out_dir} "
             f"(4view demos {chosen})"))
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
