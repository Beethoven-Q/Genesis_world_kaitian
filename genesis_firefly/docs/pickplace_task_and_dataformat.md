# Pick-place task collector + data format (and how to add a new task)

This is the manual for the cube→bowl pick-place **task** in the Genesis "Line B" firefly stack, the
**HDF5 + video data format** it writes, and **how to add a new task** on top of the same reusable world.

The whole robot + cameras + photoreal rendering + per-env immersive HDRI background DR live in the
**reusable `ManipulationStage`** (`scenes/manipulation_stage.py`). The task file is *thin*: it only adds the
objects, samples physics DR, plans the skill, scores, and writes data.

| | |
|---|---|
| **Task collector** | `genesis_firefly/collectors/pickplace_collector.py` |
| **Reusable world** | `genesis_firefly/scenes/manipulation_stage.py` (`ManipulationStage`) |
| **Entry script** | `genesis_firefly/scripts/collect_pickplace.py` |
| **Data root (default)** | `/data3/genesis_fulldr` (`DATA_DIR` env) |
| **Tile/video root (default)** | `genesis_firefly/output/temp/fulldr_collect` (`OUT_DIR` env) |
| **Latest result** | N=100, seed 7: **100/100 grasped, 97/100 placed, 3/100 through-wall**, one ~5-min parallel build |

> All N trials run in **ONE fully-parallel build** (`stage.build()`), each demo in its own batched env with
> its own random real room. No subprocess batches.

---

## 1. The task: god-mode scripted cube → bowl

The task is a **fully scripted (god-mode) cube-into-bowl pick-place** — no policy, no teleop, no imitation
seed. It uses the reused RoboLab skills (`skills/grasp.py`) for the orientation-aware grasp + the gentle
top-down place, and **Genesis-native IK** (`robots/ik.py`, the SODA-IK-equivalent tool-frame adapter) to turn
EE-space waypoints into arm joints. The motion is a 9-waypoint plan, joint-space-densified with an ease curve.

### 1a. The objects

Both objects are added to `stage.scene` by the task **before** `stage.build()`:

```python
cube = stage.scene.add_entity(
    gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(0.40, 0.18, 0.30)),
    material=gs.materials.Rigid(rho=600.0, friction=1.0),
    surface=gs.surfaces.Plastic(color=cube_col, roughness=0.35))
bowl = stage.scene.add_entity(
    gs.morphs.Mesh(file=BOWL_OBJ, convexify=True,
                   decompose_object_error_threshold=0.04, decimate=False),
    material=gs.materials.Rigid(rho=400.0, friction=1.0),
    surface=gs.surfaces.Smooth(color=bowl_col))
```

- **Cube** — a 5cm procedural box (`REGISTRY["cube"].scaled_extents()` = `(0.05, 0.05, 0.05)`), plastic PBR,
  `is_cube=True` → orientation-aware **face-pair** grasp.
- **Bowl** — `assets/objects/ycb/bowl_clean.obj` (`BOWL_OBJ`), the **Nyx-safe** visual mesh extracted from
  the YCB bowl USD (the textured `bowl.usd` segfaults Nyx — "MaterialBindingAPI not applied" — so the visual
  geometry is baked to a clean OBJ). Glossy ceramic `gs.surfaces.Smooth`.

**The bowl collision model is the single most important realism choice.** It is loaded as a **convex
DECOMPOSITION** collider (`convexify=True, decompose_object_error_threshold=0.04, decimate=False` — coacd),
exactly RoboLab's PhysX `convexDecomposition` recipe. The thin concave shell is split into a set of **solid
convex hulls** tiling the wall+floor; convex-vs-convex contact is robust **everywhere, including the thin
~2–3mm rim**, so a cube that lands on the rim **rolls in or out and can never tunnel through the wall**.

> **Do not** use the nonconvex watertight-SDF (`convexify=False`) bowl. It gives a *degenerate ~0 contact*
> where a box **corner** straddles the rim (`narrowphase.py:300-309` emits only a synthetic `1e-4 m`), so a
> corner on the rim tunnels. The rim-drop test (tilted cube on the rim, offset 4/6/7cm): **SDF → stuck in the
> wall; coacd → rolled in, never in-wall.** coacd's cosmetic ~8mm floor-raise is accepted (RoboLab accepts it).

**The cube is spawned CLEAR of the bowl** — a cube can never start embedded in a bowl in reality. The DR
rejection-samples the cube xy until `hypot(cube − bowl) ≥ 0.125` (see §2). The cube spawns ~1cm above the
table and settles during `settle_home`.

### 1b. The skill: orientation-aware grasp + gentle top-down place

The plan is 9 EE-space waypoints (`WP`), each `(label, tool_pos, tool_quat_wxyz, gripper_scalar)`:

```python
WP = [("pre",   gc - APP*tz,        gq, GR100_OPEN),
      ("at",    gc,                 gq, GR100_OPEN),
      ("close", gc,                 gq, GR100_CLOSE),
      ("lift",  gc + [0,0,LIFT],    cq, GR100_CLOSE),
      ("carry", bxyz + [0,0,PAPP],  cq, GR100_CLOSE),
      ("lower", bxyz,               cq, GR100_CLOSE),
      ("hold",  bxyz,               cq, GR100_CLOSE),   # HOLD stationary so velocity -> 0 before release
      ("rel",   bxyz,               cq, GR100_OPEN),    # gentle release ABOVE the rim
      ("ret",   bxyz + [0,0,PAPP],  cq, GR100_OPEN)]
SEG = [40, 50, 60, 95, 95, 50, 40, 55, 40]   # densification steps per segment
APP, LIFT, PAPP = 0.12, 0.20, 0.09
```

- **Grasp orientation** `gq` (per env, `gquat(i)`): the cube's world reference axis is a face normal
  (`spec.long_axis_local()` = local +X for a cube), folded into the `[-45°, +45°)` band so the jaws align to
  the **nearest face pair**. Built with `orientation_aware_grasp_quat(world_long_axis(...), tilted_base_quat(reach_dir, 0.0), reference_R=htR[s])`
  — the gripper closes **across** a face pair, never across the diagonal.
- **Carry/place orientation** `cq` (per env, `cquat(i, gq)`): `transport_quats(tilted_base_quat(reach_to_bowl, 8.0), reference_quat=gq)[0]`
  — the **least-wrist-rotation** carry orientation (8° tilt toward the bowl) so the wrist re-yaws smoothly
  during the lift, not in place at the top.
- **Gentle top-down place — the second realism rule.** The cube is **released ABOVE the rim and free-drops**
  into the solid bowl; it is *never* lowered while gripped to a target inside the wall. A position-controlled
  arm (`kp=200`) would otherwise **drive the held cube through the wall** no matter how solid the collider is.
  The release height is:
  ```python
  drop_z = tabZ + 2*BOWL_HALF_H + spec.scaled_extents()[2]/2 + 0.012   # above the rim
  ```
  Both `"lower"` and `"rel"`/`"hold"` are at `drop_z` (the gripper stays above the rim). `BOWL_HALF_H = 0.02748`.
- **The `"hold"` waypoint** keeps the cube stationary over the bowl so its lateral carry velocity → 0 **before**
  the jaws open — otherwise a residual carry velocity slips a cube corner through a hull seam (this fixed the
  last ~1/100 real penetration).

The grasp point is `gc = root0 + [0,0,grasp_dz]` from the settled cube root. IK is batched per arm
(`robot.entity.inverse_kinematics(..., dofs_idx_local=idx, max_solver_iters=24)`), converting the tool pose →
ee_link pose with `TOOL_IN_EE_INV` (the measured ~11cm ee-link-behind-claws offset). Waypoints are
joint-interpolated with a smoothstep ease `ease(u)=3u²−2u³`; the gripper scalar is interpolated and mirrored
to the mimic claw (`GR100_MIMIC * g`).

---

## 2. Full domain randomization

DR has two halves: **physics** (per-env, owned by the **task** in `sample_phys_dr`) and **visual** (owned by
the **stage**, baked per build-batch / per env). All N envs run in one build.

### 2a. Physics DR — `sample_phys_dr(N, rng, lay, spec)` (per env, independent)

| Axis | How | Range |
|---|---|---|
| **which arm** | `side_is_left = rng.rand(N) < 0.5`; `sgn=+1` left / `−1` right | 50/50 L/R |
| **object-table height** | `tabZ = ho + (rng−0.5)*0.10` (`ho` = `lay.object_table_height` = 0.25; **arm table fixed**) | ±5cm |
| **bowl xy** | `bowx = 0.40 + (rng−0.5)*0.10`; `bowy = sgn*(0.05 + (rng−0.5)*0.07)` (bowl on the active arm's side) | x ±5cm, y ±3.5cm |
| **cube xy** | `cubx = 0.40 + (rng−0.5)*0.12`; `cuby = sgn*(0.185 + (rng−0.5)*0.10)` | x ±6cm, y ±5cm |
| **cube spawn-clear** | reject-sample (≤40 tries) until `hypot(cube − bowl) ≥ 0.125` | never embedded in the bowl |
| **cube yaw** | `yaw = (rng−0.5)*radians(180)` — **full in-plane yaw** about world +Z | ±90° |
| **cube mass** | `mass_shift = (rng−0.5)*0.04` (kg), via `cube.set_mass_shift(...)` | ±20g around ~75g |
| **friction** | `robot.entity.set_friction_ratio(0.7 + 0.6*rng)` (per link) | ratio 0.7–1.3 |

(mass/friction DR is wrapped in a `try/except` — it prints `mass/fric DR skipped` and continues if the backend
build lacks the per-env mass/friction support.)

### 2b. Visual DR — owned by `ManipulationStage` (per env via the Nyx stage)

The stage does the **environment** visual DR so it never needs re-tuning per task:

- **Per-env immersive HDRI room** — one random 1K HDRI per env (`stage.hdrs[e]`). Each of the N batched envs
  renders in its **own random real room** via Nyx per-env env maps (`update_scene → set_env_map`). **No ground
  plane** — the HDRI *is* the immersive floor + walls + image-based light. The pool is downsampled to **1K**
  (`/data3/hdr1k`, built lazily by `hdr_pool()`) because **Nyx segfaults past ~50–60 2K env maps**; 1K (¼ the
  memory) lets all 100 per-env maps fit in one build (100@1K builds in ~22s).
- **Distinct cube / bowl / table colours** — `stage._pick_table()` randomizes the two table colours
  (grey-or-HSV); `stage.distinct_object_color(*avoid_h)` returns a saturated colour **guaranteed ≠ the table**
  (and ≠ any avoided hue). The task picks `cube_col` then `bowl_col` (passing the cube hue to avoid):
  ```python
  ch, cube_col = stage.distinct_object_color()
  _, bowl_col  = stage.distinct_object_color(ch)
  ```
- **Lighting** — one soft neutral directional key (`LIGHTS`) plus the HDRI image-based light; the arm uses a
  **matte** entity-surface override (`gs.surfaces.Default(metallic=0.0, roughness=0.7)`) so the silver SOMA
  livery reads neutral in any warm room (no env-mirror).

Cameras/rendering: 4 Nyx path-traced sensors at `RES=(320,180)`, `SPP` (default 32, `SPP` env):
`third` (low third-person), `cam_side` (world-fixed D435i), `cam_lw` / `cam_rw` (egocentric D405 wrist cams
attached to `left_link_6` / `right_link_6`). `stage.render()` reads them via the **sensor API** (`cam._stale=True;
cam.read().rgb`) so the wrist cams re-attach to link_6 each frame (true egocentric).

---

## 3. Success + penetration scoring (per-env, after the trial)

After the trajectory + a 40-step settle, the task reads the final cube pose, the active-arm ee pose, and the
lift-peak pose, then computes (per env):

```python
objf    = cube.get_pos()                       # final cube pos
lift_cm = (lift_pos[:,2] - root0[:,2]) * 100   # rise captured at the LIFT waypoint peak
rxy     = hypot(objf[:,0]-bowx, objf[:,1]-bowy)   # radial dist from the PER-ENV bowl centre
rim_z   = tabZ + 2*BOWL_HALF_H
ch2     = spec.scaled_extents()[2] / 2         # cube half-height

placed = (lift_cm > 3)                              # lifted clear of the table
       & (rxy < 0.06)                               # settled within the bowl footprint
       & (objf[:,2]-ch2 > tabZ - 0.005)             # not below the table
       & (objf[:,2]-ch2 < rim_z + 0.01)             # not perched above the rim
       & (norm(objf - eep) > 0.08)                  # released (clear of the gripper)
```

**A demo counts as `placed` (= `success`) iff: lifted clear + settled inside the bowl footprint + at a
plausible height (above the table, not above the rim) + released from the gripper.** "grasped" alone is
`lift_cm > 3`.

**Penetration metric (per-env bowl-centre wall annulus).** The old fell-through / far-outside check was
*blind* to wall penetration (reported 0 while cubes tunnelled), so a dedicated **wall-annulus** metric is used,
measured from the **per-env** bowl centre:

```python
bottom   = objf[:,2] - ch2                          # cube's lowest point
wall_pen = ((rxy > 0.065) & (rxy < 0.11) &          # in the wall annulus (not cavity, not table)
            (bottom > tabZ + 0.012) & (bottom < rim_z))   # held UP in the wall, not resting on the table
         | (bottom < tabZ - 0.015)                  # OR sunk below the table
```

`wall_pen` flags a cube **held up inside the wall annulus** (a real through-wall) or **sunk below the table**.
A cube that rolled out cleanly onto the table is *not* a penetration. Always confirm visually with the **tight
bowl close-up** in the four-view tiles — the wide third-person view hides wall penetration.

Console line per run:
```
[COLLECT] 100/100 grasped, 97/100 placed, through-wall=3/100  render+sim <wall>s
```

---

## 4. The HDF5 schema (exact) + the videos & tiles

`<DATA_DIR>/demos.hdf5` — one group per demo, `T` = recorded frames (state + render captured every
`REC_EVERY = 10` sim steps, plus one final frame). Arrays are `float32`.

```
/data/demo_<i>/
  actions                                  (T, 14)   absolute 14-D joint targets (the PD command @ state14_idx)
  states/articulation/robot/
    joint_position                         (T, 14)   measured joint positions (state14_idx)
    joint_velocity                         (T, 14)   measured joint velocities (state14_idx)
  ee_pose/
    position                               (T, 3)    active-arm ee_link world position
    orientation                            (T, 4)    active-arm ee_link world quaternion, WXYZ
  attrs:
    num_samples = T          (int)
    success     = bool(placed[i])
    seed        = int(seed)
    arm         = "left" | "right"         (which arm did this demo)
    hdr         = basename(stage.hdrs[i])  (the HDRI room for this env)
```

The **14-D layout** is `STATE_JOINTS_14 = [L_j1..L_j6, L_grip_driven, R_j1..R_j6, R_grip_driven]`
(6 arm + 1 driven gripper, ×2 arms; the mimic gripper joints are **not** in the state). `actions` are the
absolute PD position targets at those 14 dof indices (`full_cmd[:, state14]`), not deltas. Quaternions are
**wxyz** (Genesis convention). `ee_pose` follows the **active arm** per demo (`np.where(side_is_left, ee_l, ee_r)`).

### Per-demo policy videos (the sensor-only stream)

```
<DATA_DIR>/videos/cam_side/demo_<i>.mp4     # world-fixed D435i side cam
<DATA_DIR>/videos/cam_lw/demo_<i>.mp4       # left  D405 wrist cam (egocentric, on left_link_6)
<DATA_DIR>/videos/cam_rw/demo_<i>.mp4       # right D405 wrist cam (egocentric, on right_link_6)
```

`fps=12`, `codec=libx264` (H.264, VSCode-friendly). Only the **3 policy cameras** are written per demo —
they are the policy's observation stream. (`third` is the rig camera, used only for the tiles below.)

### Visualization tiles (under `<OUT_DIR>`)

```
<OUT_DIR>/fulldr_third_<N>.mp4              # one ceil(sqrt(N)) x ceil(sqrt(N)) grid of EVERY demo's THIRD-person view
<OUT_DIR>/fourview_demo_<i>.mp4             # 2x2 (third | side / wrist-L | wrist-R) tile, for 10 random demos
```

The third tile is `ceil(sqrt(N))` per side (10×10 for N=100). The four-view tiles are 10 random demos
(`rng.choice(N, 10)`), each frame stacked `[[third, side], [wrist-L, wrist-R]]` with text labels. These are
the human-facing QA artifacts (the tile is where you eyeball DR coverage + grasp/place quality across all demos).

---

## 5. How to run

```bash
# Direct (the collector is runnable as a script):
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/collectors/pickplace_collector.py <N> [seed]

# Or via the entry script (identical; thin wrapper):
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/data3/genesis_fulldr \
  ./.venv/bin/python genesis_firefly/scripts/collect_pickplace.py <N> [seed]

# Example: 100 demos, seed 7
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/collectors/pickplace_collector.py 100 7
```

Args: `N` (number of demos / parallel envs, default 100), `seed` (default 7). Env vars:

| Env var | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `/data3/genesis_fulldr` | the fine-tune dataset (HDF5 + `videos/`); large, keep on `/data3` |
| `OUT_DIR` | `genesis_firefly/output/temp/fulldr_collect` | the visualization tiles |
| `SPP` | `32` | Nyx samples/pixel (denoised; 32 is clean) |
| `CUDA_VISIBLE_DEVICES` | — | **pin to one GPU**; Genesis batches the N envs internally |

> **One GPU per process.** Genesis vectorizes the N demos into one batched build — do **not** launch parallel
> processes on a single GPU (multiple Isaac/Genesis instances on one GPU silently break timed grasps).

**Latest result (N=100, seed 7):** **100/100 grasped, 97/100 placed, 3/100 bowl-penetration**, one ~5-min
parallel build (build + render + sim). The 3 non-placed are cubes that rolled cleanly out of the bowl onto the
table (realistic), not tunnelling. Earlier runs of the same collector logged 98/100 placed at ~139s on a warm
HDR cache.

---

## 6. How to add a new task

The whole point of the split is that a new task is a **copy of this one file** — the stage, robot, cameras,
rendering, HDRI/colour DR, the 14-D HDF5 schema, the video/tile machinery, and the IK adapter are all reused
**unchanged**. To add a task:

1. **Copy** `collectors/pickplace_collector.py` → `collectors/<yourtask>_collector.py`.
2. **Swap the objects.** Replace the cube + bowl `stage.scene.add_entity(...)` calls with your task's entities
   (add them to `stage.scene` **before** `stage.build()`). Reuse `stage.distinct_object_color()` to get
   colours guaranteed distinct from the (randomized) table. For a registered graspable, pull geometry from
   `_core_vendored/object_spec.py REGISTRY[...]` (or add an `ObjectSpec` entry — no per-object code needed).
   For a concave container, follow `build_bowl`'s **convex-decomposition** recipe (`convexify=True,
   decompose_object_error_threshold=0.04`) and a Nyx-safe visual mesh.
3. **Swap the plan.** Edit `sample_phys_dr` for your object's pose/yaw/mass DR, and rewrite the `WP`/`SEG`
   waypoint list for your manipulation. Reuse `skills/grasp.py` (`orientation_aware_grasp_quat`,
   `tilted_base_quat`, `transport_quats`, `world_long_axis`) and the `ik(...)` helper for EE→joint. For any
   container task, keep the **release-above + free-drop** rule (never drive a held object to a target inside a
   wall) and a **hold** waypoint to zero velocity before release.
4. **Swap the scorer.** Rewrite the `placed` predicate and the penetration metric for your geometry. Keep the
   per-env, measured-pose pattern (read final pose vs the per-env target; measure penetration from the per-env
   container centre, never a global assumption). MEASURE penetration and render a tight close-up before
   claiming "no penetration".
5. **Reuse everything else unchanged** — `ManipulationStage` (do not edit it; it owns the world), the §4 HDF5
   write, the `videos/cam_{side,lw,rw}` policy stream, the `sqrt(N)` third tile + 10 four-view tiles, and the
   `DATA_DIR`/`OUT_DIR`/`SPP` env-var run interface.

### Reusable API quick reference

```python
# scenes/manipulation_stage.py
stage = ManipulationStage(n_envs, seed=0, res=(320,180), spp=32)
stage.scene                 # the gs.Scene -> add your objects to it BEFORE build
stage.distinct_object_color(*avoid_h) -> (hue, rgb)   # colour != table (and != avoided hues)
stage.build()                                          # build(n_envs, env_spacing=(0,0)) + robot.finalize()
stage.settle_home(steps=70) -> home_cmd (N, n_dofs)
stage.render() -> {"third"|"cam_side"|"cam_lw"|"cam_rw": (N,H,W,3) uint8}
stage.robot / stage.cams / stage.lay / stage.rng / stage.hdrs / stage.otable / stage.H / stage.W
np_(x)                       # CUDA tensor -> numpy at the sim<->skills boundary (quat = wxyz)

# robots/firefly_dual.py  (FireflyDual, accessed via stage.robot)
robot.arm["left"|"right"]            # 6 arm dof indices (Genesis INTERLEAVES dual-arm dofs)
robot.grip_driven / robot.grip_mimic # gripper dof indices per side
robot.state14_idx                    # the 14 dof indices for actions/states (STATE_JOINTS_14)
robot.ee["left"|"right"]             # ee_link names; robot.entity.get_link(...)
robot.ee_pose(side) -> (pos[3], quat_wxyz[4])
GR100_OPEN=0.0  GR100_CLOSE=0.9  GR100_MIMIC=-1.0    # gripper scalars (mimic = -1*driven)

# robots/ik.py
TOOL_IN_EE_INV, tool_R_at_home(home_ee_R)            # tool(claw)-frame <-> ee_link; ~11cm offset
# (GenesisArmIK.solve(tool_pos, tool_quat, q_init) -> IKSolution is the per-arm OO wrapper)

# skills/grasp.py  (sim-agnostic; all wxyz)
world_long_axis(local_axis, root_quat_wxyz)
orientation_aware_grasp_quat(ref_axis_world, base_quat, *, reference_R=None)
tilted_base_quat(reach_dir_xy, tilt_deg)             # tilt_deg=0 -> pure top-down
transport_quats(base_quat, reference_quat=None)      # least-wrist-rotation carry orientations
grasp_waypoints(...) / place_waypoints(...)          # canned pick/place waypoint lists (if you don't hand-roll WP)

# scenes/firefly_scene.py
TableLayout()  BOWL_HALF_H=0.02748  firm_rigid_options()  build_bowl(scene, xy, top_z, surface=...)

# _core_vendored/object_spec.py
REGISTRY["cube"|"apple"|"banana"|"pen"|"tennis_ball"]   # ObjectSpec: extents, mass, long_axis, grasp_dz, ...
```

### Gotchas to carry into a new task

- **Genesis returns CUDA tensors** from `get_pos/get_quat/render`; wrap them in `np_(...)` at the sim↔skills
  boundary. **Quaternions are wxyz.**
- **Dual-arm dofs are INTERLEAVED** (left_joint_1=dof0, right_joint_1=dof1, …) — always go through
  `robot.arm[side]` / `robot.state14_idx`, never assume contiguous indices.
- **Concave containers need convex decomposition**, not SDF, or a corner on the rim tunnels.
- **Never drive a gripped object to a target inside a wall** (`kp=200` out-forces any collider) — release
  above + free-drop, with a **hold** waypoint to zero velocity first.
- **Nyx env-map memory** caps at ~50–60 2K maps — keep the per-env HDRIs at 1K (`hdr_pool()` already does this).
- **The textured YCB bowl USD segfaults Nyx** — load a clean visual mesh (`bowl_clean.obj`), not `bowl.usd`,
  for any task that renders a USD-textured container.
- **One GPU per process** — let Genesis batch the envs; do not parallelize across processes on one GPU.
- **Capture the lift pose at the lift PEAK** (last lift step), not the first, or `grasped`/`lift_cm` falsely
  read ~0.
