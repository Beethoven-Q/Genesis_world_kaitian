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
from registry.object_spec import REGISTRY  # noqa: E402
from world.firefly_scene import OBJECTS, _rho_for  # asset root + spec->density helper  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import cv2  # noqa: E402
import h5py  # noqa: E402

REC_EVERY = 10                                                 # record state + render every K sim steps
BOWL_OBJ = os.path.join(os.path.dirname(_HERE), "assets/objects/ycb/bowl_clean.obj")  # Nyx-safe bowl visual


def sample_phys_dr(N, rng, lay, spec):
    """Per-env physics DR (no visuals — the stage owns the environment/background DR)."""
    ho = lay.object_table_height
    side_is_left = rng.rand(N) < 0.5
    sgn = np.where(side_is_left, 1.0, -1.0)
    tabZ = ho + (rng.rand(N) - 0.5) * 0.10                     # object-table height +/-5cm
    bowx = 0.40 + (rng.rand(N) - 0.5) * 0.10
    bowy = sgn * (0.05 + (rng.rand(N) - 0.5) * 0.07)
    cubx = 0.40 + (rng.rand(N) - 0.5) * 0.12
    cuby = sgn * (0.185 + (rng.rand(N) - 0.5) * 0.10)
    # keep the cube CLEAR of the bowl so the open gripper doesn't bump the bowl on the grasp descent. The
    # MAJORITY (~80%) get a generous clearance; a MINORITY (~20%) are allowed close (hard edge cases -> useful
    # recovery data, per the DR "accept hard edge cases" rule). The 0.125 floor still forbids cube-in-bowl.
    clr = np.where(rng.rand(N) < 0.8, 0.17, 0.125)
    for _ in range(60):
        bad = np.hypot(cubx - bowx, cuby - bowy) < clr
        if not bad.any():
            break
        nb = int(bad.sum())
        cubx[bad] = 0.40 + (rng.rand(nb) - 0.5) * 0.12
        cuby[bad] = sgn[bad] * (0.185 + (rng.rand(nb) - 0.5) * 0.10)
    yaw = (rng.rand(N) - 0.5) * np.radians(180)
    mass_shift = ((rng.rand(N, 1) - 0.5) * 0.04).astype(np.float32)
    return dict(side_is_left=side_is_left, sgn=sgn, tabZ=tabZ, bowx=bowx, bowy=bowy,
                cubx=cubx, cuby=cuby, yaw=yaw, mass_shift=mass_shift)


# ============================================================================ #
# DISTRACTOR / CLUTTER OBJECTS  (REQUIRED DR feature, docs/domain_randomization.md scope B)
# ============================================================================ #
# Pool of REALISTIC irrelevant objects for the cube->bowl task. They render from their own assets/spec colours
# (apple red/green, banana yellow, pen, tennis ball yellow-green, book dark-red) and are NEVER the grasp target.
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


def choose_distractor_types(rng, pool=DISTRACTOR_POOL):
    """PER-BUILD: how many distractors (K in {2,3}) and which TYPES (entities are created before scene.build,
    so the count + identities are fixed for the whole build). Drawn WITHOUT replacement so the K distractors
    are visually distinct lookalikes (the policy must disambiguate the cube from a varied clutter set). At most
    ONE large/long object (banana/book) per build so every set fits on the table out-of-corridor with real
    spacing (the open area can't hold two 18cm objects clear of the arm path)."""
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


def spawn_distractors(stage, dr, rng):
    """Create the per-build distractor ENTITIES (called before stage.build()) and return (entities, names,
    xy, yaw). TYPES are per-build (chosen here with the stage rng); POSES are per-env (placed after build via
    the returned batched arrays). Each entity is a real collidable rigid body from the REGISTRY (realistic
    colour/size/mass/friction) and drops+settles with the cube/bowl during the existing settle."""
    names = choose_distractor_types(rng)
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


def collect(N, seed, data_dir, out_dir):
    t0 = time.time()
    stage = ManipulationStage(N, seed=seed)                   # the reusable robot+cameras+rendering+env-DR setup
    lay, spec, rng = stage.lay, REGISTRY["cube"], stage.rng
    dr = sample_phys_dr(N, rng, lay, spec)

    # --- task objects: a coloured cube + a convex-decomposition bowl, distinct from the table colour ---
    ch, cube_col = stage.distinct_object_color()
    _, bowl_col = stage.distinct_object_color(ch)
    cube = stage.scene.add_entity(gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(0.40, 0.18, 0.30)),
                                  material=gs.materials.Rigid(rho=600.0, friction=1.0),
                                  surface=gs.surfaces.Plastic(color=cube_col, roughness=0.35))
    bowl = stage.scene.add_entity(gs.morphs.Mesh(file=BOWL_OBJ, convexify=True,
                                  decompose_object_error_threshold=0.04, decimate=False),
                                  material=gs.materials.Rigid(rho=400.0, friction=1.0),
                                  surface=gs.surfaces.Smooth(color=bowl_col))

    # --- distractor / clutter objects (REQUIRED DR): 2-3 random irrelevant objects on the OBJECT table, in OPEN
    # areas, rejection-sampled OUT of the active arm's swept corridor (grasp + cube->bowl carry + bowl->home).
    # ENTITIES are created here (per-build types); their per-env POSES are applied after build() below. ---
    dist_ents, dist_names, dist_xy, dist_yaw = spawn_distractors(stage, dr, rng)
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
        robot.entity.set_friction_ratio((0.7 + 0.6 * rng.rand(N, robot.entity.n_links)).astype(np.float32))
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
    root0 = np_(cube.get_pos())
    dist_xy0 = np.stack([np_(e.get_pos())[:, :2] for e in dist_ents], 0) if dist_ents else np.zeros((0, N, 2))

    # ---- per-env grasp + GENTLE top-down place plan (RoboLab skills + SODA IK) ----
    _, heqL = robot.ee_pose("left"); _, heqR = robot.ee_pose("right")
    htR = {"left": tool_R_at_home(_R_from_wxyz(np_(heqL)[0])), "right": tool_R_at_home(_R_from_wxyz(np_(heqR)[0]))}
    base = {"left": np.array([0.0, 0.224]), "right": np.array([0.0, -0.224])}
    laxis = spec.long_axis_local()

    def gquat(i):
        s = "left" if side_is_left[i] else "right"
        yr = ((yaw[i] + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        qzr = np.array([np.cos(yr / 2), 0.0, 0.0, np.sin(yr / 2)])
        return orientation_aware_grasp_quat(world_long_axis(laxis, qzr),
                                            tilted_base_quat(np.array([cubx[i], cuby[i]]) - base[s], 0.0), reference_R=htR[s])

    def cquat(i, gqi):
        s = "left" if side_is_left[i] else "right"
        return transport_quats(tilted_base_quat(np.array([bowx[i], bowy[i]]) - base[s], 8.0), reference_quat=gqi)[0]

    gq = np.stack([gquat(i) for i in range(N)]).astype(np.float64)
    cq = np.stack([cquat(i, gq[i]) for i in range(N)]).astype(np.float64)

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

    gc = root0.copy(); gc[:, 2] += spec.grasp_dz
    tz = np.stack([_R_from_wxyz(q) @ [0, 0, 1.0] for q in gq])
    drop_z = tabZ + 2 * BOWL_HALF_H + spec.scaled_extents()[2] / 2 + 0.012   # release ABOVE the rim (gentle)
    bxyz = np.stack([bowx, bowy, drop_z], 1).astype(np.float64)
    APP, LIFT, PAPP = 0.12, 0.18, 0.08   # match RoboLab/plan_pick_place defaults (lower lift = gentler, safe)
    OPEN, CLOSE = GR100_OPEN, GR100_CLOSE

    # home tool pose per env, so the FIRST move (home->pre) is densified+smooth too (not a PD snap).
    hposL, hquatL = robot.ee_pose("left"); hposR, hquatR = robot.ee_pose("right")
    M_te, t_te = TOOL_IN_EE_INV[:3, :3], TOOL_IN_EE_INV[:3, 3]

    def ee_to_tool(p, q):                                       # inverse of ik()'s tool->ee map
        Rt = _R_from_wxyz(q) @ M_te.T
        return p - Rt @ t_te, _wxyz_from_R(Rt)

    # per-env SPARSE EE waypoints for the ACTIVE arm; BatchExecutor densifies each into the gentle
    # constant-Cartesian-speed motion (a pose held across two waypoints -> a smooth gripper dwell ramp).
    wps = []
    for i in range(N):
        hp, hq = ee_to_tool(hposL[i], hquatL[i]) if side_is_left[i] else ee_to_tool(hposR[i], hquatR[i])
        wps.append([
            ("home",  hp,                    hq,    OPEN),        # everything gentle (constant slow speed)
            ("pre",   gc[i] - APP * tz[i],   gq[i], OPEN),
            ("at",    gc[i],                 gq[i], OPEN),
            ("at",    gc[i],                 gq[i], OPEN),
            ("close", gc[i],                 gq[i], CLOSE),     # pose held -> gripper close ramp + grasp settle
            ("lift",  gc[i] + [0, 0, LIFT],  cq[i], CLOSE),
            ("carry", bxyz[i] + [0, 0, PAPP], cq[i], CLOSE),
            ("lower", bxyz[i],               cq[i], CLOSE),
            ("rel",   bxyz[i],               cq[i], OPEN),      # pose held -> gripper release ramp
            ("ret",   bxyz[i] + [0, 0, PAPP], cq[i], OPEN),
            ("go_home", hp,                  hq,    OPEN),      # smooth, densified RETURN HOME (recorded in
        ])                                                     # video + data -> the policy learns to go home

    def solve(p, q):                                           # IK both arms, pick the active one per env
        return np.where(side_is_left[:, None], ik(ee["l"], p, q), ik(ee["r"], p, q))

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
    pen_tracker = PenetrationTracker(stage.scene, n_envs=N)

    def on_step(t, full, labels_t):
        record_state(full)
        pen_tracker.update()                                       # fold this step's contact buffer into the max
        views = stage.render()
        for nm in cams:
            cam_steps[nm].append(views[nm])
        cubez.append(np_(cube.get_pos())[:, 2].copy())
        if os.environ.get("DIST_DEBUG") and dist_ents:
            dist_trace.append(np.stack([np_(e.get_pos())[:, :2] for e in dist_ents], 0).copy())

    # gentle everywhere: lin 0.13 m/s, ang 0.9 rad/s (slows the wrist reorientation, the flip-prone part);
    # ik_every=1 so there's no zero-order-hold staircase (cheap now that IK does a single warm-started solve).
    ex = BatchExecutor(stage.scene, robot, side_is_left, rec_every=REC_EVERY, ang_speed=0.9, ik_every=1)
    T = ex.run(wps, solve, home16, on_step=on_step, settle_steps=40)
    lift_pos_z = np.max(np.stack(cubez, 1), axis=1)            # per-env max cube height reached during the run
    wall = time.time() - t0
    li, ri = np.where(side_is_left)[0], np.where(~side_is_left)[0]
    # quantitative jerk gate: worst single-step active-arm joint jump. A branch flip = a >1 rad spike on a
    # wrist joint; after the max_samples=1 fix this should be small (<~0.15 rad).
    print(f"[COLLECT] executed T={T} ({len(li)} left / {len(ri)} right arm)  render+sim {wall:.1f}s  "
          f"max per-step |dq|={float(ex.max_dq.max()):.3f} rad", flush=True)

    # ---- score + realistic penetration check (per-env bowl centre) ----
    objf = np_(cube.get_pos())
    eep = np.where(side_is_left[:, None], np_(ee["l"].get_pos()), np_(ee["r"].get_pos()))
    lift_cm = (lift_pos_z - root0[:, 2]) * 100
    ch2 = spec.scaled_extents()[2] / 2
    rxy = np.hypot(objf[:, 0] - bowx, objf[:, 1] - bowy)
    rim_z = tabZ + 2 * BOWL_HALF_H
    placed = (lift_cm > 3) & (rxy < 0.06) & (objf[:, 2] - ch2 > tabZ - 0.005) & \
             (objf[:, 2] - ch2 < rim_z + 0.01) & (np.linalg.norm(objf - eep, axis=1) > 0.08)
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
    placed = placed & ~penetrating                                # an abnormally penetrating demo is NOT clean
    worst_e = int(np.argmax(pen_mm))
    print(f"[COLLECT] penetration: max={float(pen_mm.max()):.1f}mm "
          f"(thresh={ABNORMAL_THRESH_M*1000:.0f}mm), abnormal={int(penetrating.sum())}/{N}"
          + (f"  worst env{worst_e}: {pen_tracker.worst_names()[worst_e]}" if pen_mm.max() > 0 else ""),
          flush=True)

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
    vdir = os.path.join(data_dir, "videos")
    for nm in ("cam_side", "cam_lw", "cam_rw"):
        os.makedirs(os.path.join(vdir, nm), exist_ok=True)
    with h5py.File(os.path.join(data_dir, "demos.hdf5"), "w") as f:
        for e in range(N):
            d = f.create_group(f"data/demo_{e}")
            d.create_dataset("actions", data=acts[e].astype(np.float32))
            sg = d.create_group("states/articulation/robot")
            sg.create_dataset("joint_position", data=jpos[e].astype(np.float32))
            sg.create_dataset("joint_velocity", data=jvel[e].astype(np.float32))
            pg = d.create_group("ee_pose")
            pg.create_dataset("position", data=eepos[e].astype(np.float32))
            pg.create_dataset("orientation", data=eequat[e].astype(np.float32))
            d.attrs["num_samples"] = Tr; d.attrs["success"] = bool(placed[e]); d.attrs["seed"] = int(seed)
            d.attrs["arm"] = "left" if side_is_left[e] else "right"
            # PENETRATION GATE attrs (owner #1): worst-ever solid-solid interpenetration (mm) + the abnormal
            # flag. ``penetrating`` True means the demo is REJECTED -- ``success`` is already forced False
            # above, so the success_only LeRobot export drops it; this poisoned data is never shipped.
            d.attrs["max_penetration_mm"] = float(pen_mm[e])
            d.attrs["penetrating"] = bool(penetrating[e])
            d.attrs["hdr"] = os.path.basename(stage.hdrs[e])
            d.attrs["has_distractors"] = bool(has_dist[e])     # 50/50 per-env: was this a cluttered trial?
            d.attrs["distractors"] = ",".join(dist_names) if has_dist[e] else ""
            for nm in ("cam_side", "cam_lw", "cam_rw"):        # the sensor-only policy stream
                vid = np.stack([cam_steps[nm][t][e] for t in range(Tr)])
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
    DATA = os.environ.get("DATA_DIR", "/data3/genesis_fulldr")
    OUT = os.environ.get("OUT_DIR", "genesis_firefly/output/temp/fulldr_collect")
    gs.init(backend=gs.gpu)
    collect(N, SEED, DATA, OUT)
