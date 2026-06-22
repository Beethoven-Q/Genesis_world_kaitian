# ManipulationStage — the reusable manipulation + rendering setup

`world/manipulation_stage.py` is **THE** reusable "world" every Genesis Line-B manipulation task
runs in. You build it **once** and never re-tune it. A task adds only its **objects + skill +
scoring** on top. The robot, the cameras, the photoreal rendering, the immersive HDRI background
domain-randomization, and the parallel-collecting harness are all **fully encapsulated** here — a
task never touches any of it.

- **Source:** `/home/kaitianchao/Projects/Genesis_world_kaitian/genesis_firefly/world/manipulation_stage.py`
- **Reference task using it:** `/home/kaitianchao/Projects/Genesis_world_kaitian/genesis_firefly/tasks/pickplace.py`
- **Self-check:** `CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/world/manipulation_stage.py`

---

## 1. Why it exists

Before this class, every collector re-wired the robot, the cameras, the Nyx renderer, the livery
override, the HDRI background DR, and the parallel build from scratch — and each one drifted (a
collector once shipped without the side-camera rig; another left the wrist cams at the third-person
Nyx default pose). The hard-won fixes (matte-livery override, convex-decomposed collision, the
per-env 1K-HDRI immersive background, the sensor-API render that re-attaches wrist cams) all had to
be reproduced correctly *per task*, which is exactly how regressions creep in.

`ManipulationStage` **isolates everything that should never change between tasks** into one place:

- the **robot** (dual Firefly Y6 + GR100, baked SOMA livery, matte override, MIT gains,
  convex-decomposed collision),
- the **tables**,
- the **3 policy cameras + the visible side-camera body/stick**,
- **Nyx photoreal rendering**,
- the **per-env immersive HDRI background DR**, and
- the **one fully-parallel build** of N environments.

The owner's directive: a setup he **never re-tunes**. Tasks only add objects + skills. If you find
yourself editing rendering, livery, cameras, or the background harness inside a collector, that code
belongs in the stage instead.

---

## 2. What it provides

### Robot — dual Firefly Y6 + GR100
Constructed as `FireflyDual(self.scene, pos=(0,0,arm_table_height), surface=gs.surfaces.Default(**ARM_SURF))`
(`robots/firefly_dual.py`). It bundles:
- **Baked SOMA livery** — `FireflyDual` defaults to `dual_firefly_y6_gr100_livery.urdf`, whose
  `<visual>` meshes are GLBs with per-link PBR baked in (silver metallic links, grey link_4, dark
  wrist/base, orange-carbon + silver-SOMA panels on link_2/3). Nyx reads a URDF's embedded mesh
  materials, so this is how the true livery shows.
- **Matte entity-surface override** — `ARM_SURF = dict(metallic=0.0, roughness=0.7)`, passed as
  `gs.surfaces.Default(**ARM_SURF)`. Nyx honors the **entity-level** `surface` as a matOverride for a
  URDF, and because `color=None` here it sets only metallic/roughness (per-mesh texture **colours are
  preserved**). This kills the warm-HDRI mirror so the silver livery reads neutral in any room.
- **Convex-DECOMPOSED collision** (`decompose_robot_error_threshold=0.05`) so each finger/link
  collider hugs its visual mesh — no finger/link interpenetration under a firm grip.
- **MIT PD gains** (`ARM_KP=[200,200,200,75,15,15]`, `ARM_KV=[12.5,12.5,12.5,6,0.31,0.31]`),
  `default_armature=0.01`, and `gravity_compensation=1.0` — all reproduced exactly from RoboLab; do
  not retune.

### Tables — two static collidable boxes
Built directly from `TableLayout` (`world/firefly_scene.py`):
- `self.atable` — the **arm table** (under the robot), depth `0.34`, colour `0.7 * table_color`.
- `self.otable` — the **object table** (in front), depth `0.70`, the place surface; this is the entity
  a task moves to do table-height DR.

Both are `gs.morphs.Box(... fixed=True, collision=True)` with `gs.surfaces.Plastic`. Seam at `x=0.13`,
common width `0.90`, default top `z=0.25`.

### Cameras — 3 policy cameras + the visible side rig
All four are **Nyx sensors** (`NyxCameraOptions`), all at `res=RES=(640,360)` (16:9; short side 360 ≥ pi0.5's
~224 input, so no upscaling):

| key        | type                          | placement                                                            |
|------------|-------------------------------|----------------------------------------------------------------------|
| `third`    | low third-person (witness)    | `pos=(1.15,-0.95,0.62) lookat=(0.30,0,0.34) fov=48` — room shows behind the arm |
| `cam_side` | world-fixed side **D435i**    | `pos=SIDE[0] lookat=(0.40,0,0.28) fov=SIDE_VFOV` (43.2°)              |
| `cam_lw`   | left egocentric wrist **D405**| attached to `left_link_6`, `offset_T=_T(*LEFT_WRIST) fov=WRIST_VFOV` (57.95°) |
| `cam_rw`   | right egocentric wrist **D405**| attached to `right_link_6`, `offset_T=_T(*RIGHT_WRIST) fov=WRIST_VFOV` |

The **3 policy cameras** are `cam_side`, `cam_lw`, `cam_rw` (the LeRobot streams). `third` is a
witness/tiling view, not a policy input. The wrist cams use `entity_idx`/`link_idx_local`/`offset_T`
so Nyx re-attaches them to `link_6` every render → truly **egocentric** (fingers-at-bottom, looking
down). The visible **side-camera body + support stick** are added by `add_side_camera_rig(scene)`
(real `camera_link.STL` D435i body at the calibrated pose + an 8 mm cylinder stick from the floor).
Calibrated poses/FOV live in `world/firefly_cameras.py` and are baked, not tuned.

### Rendering — Nyx photoreal
Path-traced, denoised PBR via `gs_nyx_plugin`. `SPP=32` samples/pixel (clean after denoise; override
via the `SPP` env var). One soft neutral directional key light (`LIGHTS`) for contact shadows; the
HDRI does the bulk of the lighting. Rendering is driven through the **sensor API** (`cam.read().rgb`),
which is what makes the wrist cams egocentric — see `render()`.

### Immersive per-env HDRI background DR
**No ground plane.** Each of the N parallel envs renders its **own random real room**: the HDRI *is*
the floor + walls + image-based light, so the arm genuinely sits in a real lounge / office / bathroom,
and the two tables are the only local surfaces. The stage picks one random HDRI per env and hands the
N-tuple of `EnvironmentMapAsset`s to every camera (`env_maps=...`); Nyx's per-env
`update_scene`/`set_env_map` renders each batched env with its own HDRI in **one build**. The stage
also draws a per-build **table texture** (`self.table_texture`) + table colours (`self.table_color`)
and, with `full_dr=True`, the per-build scope-A/C scene DR (table-size grow, side-cam pose, key light).

**HDRI resolution = 2K when `n_envs ≤ MAX_2K_ENVS` (45), else the 1K pool.** Build-batches keep E ≤ 45
so each build renders at full **2K**; only a single >45-env build falls back to the 1K pool (Nyx
segfaults past ~50–60 2K env maps — see "Why a 1K HDRI pool" below).

### One fully-parallel build of N envs
`stage.build()` calls `self.scene.build(n_envs=self.n_envs, env_spacing=(0.0, 0.0))` — a single
batched scene. `env_spacing=(0,0)` is mandatory so each env renders its own immersive room with no
cross-env bleed. All N demos run, render, and score in parallel.

### Why a 1K HDRI pool (only for a single large >45-env build)
Nyx **segfaults** past ~50–60 **2K** env maps (50×2K builds, 80×2K core-dumps). The large *scene*
itself is fine; only the env-map memory crashes. For a single build with `n_envs > MAX_2K_ENVS` (45),
`hdr_pool()` downsamples the 2K library to **1K** (¼ the memory) so all per-env env maps fit in one
build, cached under `/data3/hdr1k`. Corrupt 2K `.hdr` files are skipped (loading one would reintroduce a
2K map and crash). **The normal path is build-batches with E ≤ 45 → original 2K** (`valid_2k_pool()`).

---

## 3. Public API

### Module-level functions

```python
hdr_pool(n_target=140) -> list[str]
```
Returns a sorted list of 1K `.hdr` paths (downsampled from
`/home/kaitianchao/Projects/RoboLab_firefly/assets/backgrounds/{indoors,outdoors}/*.hdr`, valid-only),
cached under `/data3/hdr1k`, built lazily on first call.

```python
np_(x) -> np.ndarray
```
`.cpu().numpy()` if `x` is a torch CUDA tensor, else `np.asarray(x)`. Use this at **every** sim↔numpy
boundary — Genesis `get_pos/get_quat/render` return CUDA tensors.

### `ManipulationStage`

```python
ManipulationStage(n_envs, seed=0, res=RES, spp=SPP, noslip=0, full_dr=True)
```
`RES=(640,360)`, `SPP` from env var (default 32). `noslip` = per-object `noslip_iterations` (for a
round/thin grasp target; the task passes `spec.grasp_noslip`). `full_dr` (also gated by the `FULL_DR`
env var) enables the per-build scope-A/C scene DR (table-size grow, side-cam pose, key light). On
construction it draws the per-env HDRIs + the per-build table texture / colours / scene DR, creates
`self.scene` (no plane), adds the robot + both tables + the side rig + the 4 Nyx cameras. **The scene
is NOT built yet** — the task adds objects first, then calls `build()`.

**Attributes (set in `__init__`):**

| attr               | meaning                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `.n_envs`          | number of parallel environments                                         |
| `.res`, `.W`, `.H` | render resolution `(640,360)` and its width/height                      |
| `.rng`             | `np.random.RandomState(seed)` — the task's shared RNG                   |
| `.scene`           | the `gs.Scene` (task adds objects to **this**)                          |
| `.robot`           | the `FireflyDual` instance                                              |
| `.lay`             | the `TableLayout` (heights, depths, `seam_x`)                           |
| `.otable`          | object-table entity (move it for table-height DR)                       |
| `.atable`          | arm-table entity                                                        |
| `.cams`            | dict `{"third","cam_side","cam_lw","cam_rw"}` of Nyx sensors            |
| `.hdrs`            | list of N HDRI paths (one per env) — record `basename` per demo         |
| `.table_color`     | the chosen table RGB (objects must be distinct from it)                 |
| `.table_texture`   | the per-build table-texture albedo path (both tables share it)          |
| `.full_dr`         | whether the per-build scope-A/C scene DR is active                      |

**Methods:**

```python
.build() -> self
```
Builds the batched scene (`n_envs`, `env_spacing=(0,0)`) and `robot.finalize()` (resolves dof indices,
pushes PD gains). **Call AFTER the task has added its objects to `self.scene`.**

```python
.settle_home(steps=70) -> np.ndarray   # home command, shape (N, n_dofs)
```
Holds both arms at the home pose for `steps` so the scene settles. The returned `home` command (a
`(N, n_dofs)` float32 array) is the base you overwrite per-arm-dof during execution.

```python
.render() -> dict[str, np.ndarray]   # {name: (N, H, W, 3) uint8}
```
Renders all 4 batched Nyx cameras via the sensor API. Internally sets `cam._stale = True` and calls
`cam.read().rgb` per camera — this re-attaches the wrist cams to `link_6` each frame (true egocentric)
and applies the per-env env maps (each env its own room).

```python
.distinct_object_color(*avoid_h) -> (hue, rgb)
```
A saturated RGB for a task object that is **never** ≈ the table colour (and avoids any given hues).
Returns `(hue_float, rgb_tuple)`. Call once per object, passing prior hues to keep objects distinct
(e.g. `ch, cube_col = stage.distinct_object_color(); _, bowl_col = stage.distinct_object_color(ch)`).

### What the TASK owns (NOT the stage)
- **The objects** (added to `stage.scene` before `build()`).
- **Per-env physics DR** — object poses, yaw, mass (`set_mass_shift`), friction
  (`set_friction_ratio`), object-table height (move `stage.otable`).
- **The skill** (grasp/place plan + IK) and **scoring/output** (HDF5 + videos).

---

## 4. Copy-paste template: a NEW task on the stage

The pattern is always: **construct stage → add objects to `stage.scene` → `stage.build()` →
`stage.settle_home()` → loop (`control_dofs_position` + `stage.scene.step()` + `stage.render()`) →
score + write.**

```python
#!/usr/bin/env python3
"""A NEW task — a thin layer on the reusable ManipulationStage."""
import os, sys, numpy as np, genesis as gs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # genesis_firefly/
from world.manipulation_stage import ManipulationStage, np_      # the reusable setup


def collect(N, seed, data_dir, out_dir):
    # 1) CONSTRUCT the stage (robot + cameras + Nyx + immersive HDRI DR + parallel build harness)
    stage = ManipulationStage(N, seed=seed)
    lay, rng = stage.lay, stage.rng

    # 2) ADD YOUR OBJECTS to stage.scene (colours distinct from the table)
    _, obj_col = stage.distinct_object_color()
    obj = stage.scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.40, 0.18, 0.30)),
        material=gs.materials.Rigid(rho=600.0, friction=1.0),
        surface=gs.surfaces.Plastic(color=obj_col, roughness=0.35))

    # 3) BUILD the single fully-parallel scene (after objects are added)
    stage.build()
    robot, cams = stage.robot, stage.cams

    # 3b) per-env PHYSICS DR (poses/yaw/mass/friction/table-height) — the TASK's job
    obj.set_pos(np.stack([... , ... , ...], 1).astype(np.float32))   # (N,3)
    # obj.set_mass_shift(...);  robot.entity.set_friction_ratio(...); stage.otable.set_pos(...)

    # 4) SETTLE at home -> the (N, n_dofs) base command you overwrite per arm-dof
    home = stage.settle_home(70)

    # 5) PLAN your skill (IK -> joint waypoints -> densified joint trajectory `traj`, gripper `grip`) ...

    # 6) EXECUTE + RECORD: control_dofs_position -> scene.step() -> render() every K steps
    cam_steps = {nm: [] for nm in cams}
    for t in range(T):
        full = home.copy()
        for k, d in enumerate(robot.arm["left"]):  full[:, d] = traj[t][:, k]   # (per-arm in real tasks)
        full[:, robot.grip_driven["left"]] = grip[t]
        full[:, robot.grip_mimic["left"]]  = -1.0 * grip[t]   # GR100_MIMIC coupling
        robot.entity.control_dofs_position(full)
        stage.scene.step()
        if t % 10 == 0:
            views = stage.render()                 # {name: (N,H,W,3)}
            for nm in cams:
                cam_steps[nm].append(views[nm])

    # 7) SCORE (per-env, use np_() at the sim boundary) + WRITE HDF5 / videos / tiles
    final = np_(obj.get_pos())                     # (N,3); record stage.hdrs[e] per demo
    # ... write demos.hdf5 + per-demo policy mp4s (cam_side/cam_lw/cam_rw) ...


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    gs.init(backend=gs.gpu)                         # NOTE: gs.gpu, not gs.cuda
    collect(N, SEED, os.environ.get("DATA_DIR", "/data3/<task>"),
                     os.environ.get("OUT_DIR", "genesis_firefly/output/temp/<task>"))
```

**See the real, working instance:** `tasks/pickplace.py` — the THIN composer that spawns a target
(`world/object_factory.py`) + a convex-decomposition bowl, applies the full DR (`dr/apply.py`), plans an
orientation-aware grasp + gentle release-above-rim place via the reusable skills
(`skills/grasp.py::grasp_action_wps` + `skills/place.py::place_action_wps`, optionally
`skills/grasp_retry.py`), scores with `skills/score.py` + the `skills/penetration.py` gate, and writes
the §4 HDF5 + 3 policy-cam videos + tiles. Its `collect()` is the canonical example of every step above.
(Real, modern templates also use `world/object_factory.py` rather than hand-rolling `add_entity` — the
cube-`add_entity` shown above is the minimal illustrative form.)

### Notes / gotchas for task authors
- `gs.init(backend=gs.gpu)` — **not** `gs.cuda` (it logs "backend gs.cuda" at runtime, but the call is
  `gs.gpu`). Init **once** before constructing the stage.
- Genesis interleaves the dual-arm dofs (`left_joint_1=dof0`, `right_joint_1=dof1`, …) — the layout is
  **not** contiguous. Always index via `robot.arm[side]`, `robot.grip_driven/grip_mimic[side]`,
  `robot.state14_idx`; never hardcode dof indices.
- The gripper is coupled at the **action layer**, not URDF mimic: command
  `driven = g`, `mimic = GR100_MIMIC * g` (`GR100_MIMIC = -1.0`). The 14-D policy state excludes the
  mimic joints.
- quaternions are **wxyz**. `render`/`get_pos`/`get_quat` return CUDA tensors → wrap with `np_()`.
- For container tasks, **release above the rim and free-drop** — never drive a held object to a target
  inside a wall (a position-controlled arm out-forces any collider). This is task logic, not stage
  logic.
- Don't add a ground `Plane` — it would occlude the immersive HDRI floor. The stage deliberately omits
  it.
- Add objects **before** `build()`. After `build()` the scene topology is fixed; you can only set
  poses / DR.

---

## 5. Self-check

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/world/manipulation_stage.py
```

Builds a 2-env stage (seed 1), settles 50 steps, renders, and writes a third-person frame to
`/tmp/stage_selfcheck.png`, then prints e.g.:

```
STAGE_OK third (2, 180, 320, 3) rooms: ['<room_a>.hdr', '<room_b>.hdr']
```

This confirms: the parallel build works, the matte-livery arm + tables + side-rig render, the wrist
cams attach, and the two envs each get a different random room. If it prints `STAGE_OK` with two
distinct room basenames and the PNG shows the arm sitting in a photoreal room, the harness is healthy.

---

## Summary

The harness — **immersive HDRI background, Nyx photoreal rendering, parallel N-env collecting,
livery, cameras, side rig, collision fidelity** — is **fully encapsulated** in `ManipulationStage`.
Tasks never touch it. To build a new task: construct the stage, add your objects + skill + scoring,
and call `build / settle_home / step / render`. Nothing in §2 should ever need re-tuning per task.
