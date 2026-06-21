# Genesis firefly — dual-arm manipulation + photoreal data collection (Line B)

Reproduction of the RoboLab_firefly (IsaacLab/PhysX) dual **Firefly Y6 + GR100** pick-place stack in the
**Genesis** simulator, with **RTX-grade Nyx** rendering, **full domain randomization**, and **fully-parallel**
god-mode data collection for VLA (pi0.5) fine-tuning.

The design rule: a **reusable robot-manipulation + rendering SETUP** that you never re-tune, with **TASKS**
layered on top. You build a new task by swapping objects + a skill + a scorer — never the robot, cameras,
rendering, backgrounds, collision, or the parallel harness.

```
ManipulationStage  (scenes/manipulation_stage.py)   ← the SETUP (never re-tune)
  ├─ robot      : dual Firefly Y6 + GR100, baked SOMA livery, matte override, convex-decomposed collision
  ├─ tables     : 2 static collidable tables
  ├─ cameras    : 2 egocentric wrist D405 + 1 side D435i (policy stream) + visible side body/stick
  ├─ rendering  : Nyx path tracer (PBR), sensor read() egocentric wrist cams
  └─ env DR     : per-env IMMERSIVE HDRI background (each parallel env = its own real room), 1K pool
TASK  (collectors/pickplace_collector.py)            ← thin: objects + physics DR + skill + score + write
```

## Quick start
```bash
# 20 fully-parallel cube→bowl demos, full DR, photoreal, one build
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/tasks/pickplace.py 20 7
#   -> /data3/genesis_fulldr/demos.hdf5  + videos/cam_{side,lw,rw}/demo_*.mp4   (the fine-tune dataset)
#   -> output/temp/fulldr_collect/fulldr_third_20.mp4 (√N tile) + fourview_demo_*.mp4 (four-view 2×2 tiles)
# knobs: DATA_DIR, OUT_DIR, SPP (default 32). Entry wrapper: genesis_firefly/runner/collect.py

# render-check the setup:
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/world/manipulation_stage.py
```

## Latest result
100 trials in **one parallel build**, each in its own random immersive room (1K HDRI):
**100/100 grasped · 97/100 placed · 3/100 bowl-penetration**. Collision is in good mode (convex-decomposed
arm/gripper/camera colliders, self-collision on, firm Newton solver — see the collider image below).

![colliders](img_colliders_third.png)

## Docs
| doc | what it covers |
|-----|----------------|
| [project_overview.md](project_overview.md) | **The big blueprint** — mission, the end-to-end sim-to-real pipeline, SETUP-vs-TASK, the harnesses, the DR subagent + near-future agents (object-refiner), the virtual-EE skill, build-batch parallelism, data contract, repo layout, and how to add a task. **Read this first.** |
| [domain_randomization.md](domain_randomization.md) | **The complete full-DR spec** for EVERY task — scopes A (scene), B (object/task), C (visual); the per-build vs per-env renderer constraint; pose anti-coupling rules. The authoritative DR reference. |
| [manipulation_stage.md](manipulation_stage.md) | **The reusable setup** — what it provides, the full API, and a copy-paste template for a new task. |
| [rendering_and_livery.md](rendering_and_livery.md) | Nyx photoreal rendering, immersive per-env HDRI DR, the matte override, the livery bakers, and every Nyx gotcha. |
| [robot_collision_cameras.md](robot_collision_cameras.md) | The dual arm, GR100 gripper, gains/IK, the "good mode" collision model, and the 3 cameras + side rig. |
| [pickplace_task_and_dataformat.md](pickplace_task_and_dataformat.md) | The cube→bowl task, full DR, success/penetration scoring, the §4 HDF5 schema, and how to add a new task. |
| [lessons_genesis_nyx.md](lessons_genesis_nyx.md) | Symptom→cause→fix logbook of every hard-won Genesis/Nyx finding (read before debugging). |
| [genesis_reproduction_plan.md](genesis_reproduction_plan.md) | The original two-line (RoboLab + Genesis) plan. |

## Repo layout (`genesis_firefly/`)
```
world/manipulation_stage.py    the reusable SETUP (robot+cameras+rendering+immersive HDRI DR+parallel)
world/firefly_scene.py         TableLayout, firm_rigid_options (firm Newton collision solver), object factory
world/firefly_cameras.py       3 policy cameras + the visible side-camera body/stick rig
robots/firefly_dual.py         dual-arm loader: livery URDF, 14-D layout, MIT gains, decomposed collision
robots/ik.py                   Genesis-native IK adapter (robots/soda_ik/ = the analytic SODA IK, kept)
skills/                        grasp · pick_place · trajectory · executor (the smooth motion path), pure numpy
registry/object_spec.py        the object library (ObjectSpec/REGISTRY) + constants
tasks/pickplace.py             the cube→bowl TASK (thin, on top of ManipulationStage)
runner/collect.py              entry point: one parallel build of N demos
dataio/lerobot_exporter.py     HDF5 → LeRobot v2 export for pi0.5
scripts/bake_firefly_livery.py, bake_soma_panels.py   the livery bakers (run once -> the livery URDF + GLBs)
assets/                        URDF + meshes + the baked livery GLBs + bowl_clean.obj + object library
docs/                          you are here  (start with project_overview.md)
```

## Hard constraints (carried from RoboLab)
god-mode scripted collectors ↔ sensor-only policy (3 RGB + 14-D proprio + language); collision realism is
first-class; one sim process per GPU; the §4 HDF5 schema is a fixed contract (exporter reused unchanged).
