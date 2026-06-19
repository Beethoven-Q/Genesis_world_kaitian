# Full-DR parallel pick-place collector

Collects N cube→bowl demos **in parallel** (one batched Genesis env per demo) under **full domain
randomization**, and writes the sensor-only fine-tune dataset + visualization tiles in one pass.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/data3/genesis_fulldr \
  ./.venv/bin/python genesis_firefly/scripts/collect_pickplace.py <N> [seed]
# e.g. 100 demos:  ... collect_pickplace.py 100 7
```

- `DATA_DIR` (default `/data3/genesis_fulldr`) — the fine-tune dataset (large; keep on /data3).
- `OUT_DIR` (default `genesis_firefly/output/temp/fulldr_collect`) — the visualization tiles.
- One GPU per process (Genesis batches the N envs internally — do **not** launch parallel processes on one GPU).

## What "full DR" randomizes (per env, independently)

| | |
|---|---|
| **physics** | which **arm** (L/R) · cube **xy** + **yaw** · bowl **xy** (spawn-separated from the cube) · object-**table height** ±5cm (arm table fixed) · cube **mass** · cube/arm **friction** |
| **visual** | immersive room/outdoor **background** · pure **cube colour** · pure **bowl colour** (both forced ≠ table) · table **colour-or-texture** · **lighting** colour-temperature + brightness |

Realism guarantees: the bowl is a **convex-decomposition** collider (a cube landing on the rim rolls in/out,
never through), the cube is **spawned clear of the bowl**, and it is placed by a **gentle top-down release**
above the rim (never driven into the wall). See `[[project_genesis_firefly_line]]` memory for the recipe.

## Outputs

```
<DATA_DIR>/
  demos.hdf5                       # /data/demo_<i>/  (the §4 LeRobot schema)
    actions                        (T,14) float32   absolute 14-D joint targets
    states/articulation/robot/joint_position (T,14), joint_velocity (T,14)
    ee_pose/position (T,3), orientation (T,4 wxyz)
    attrs: num_samples, success, seed, arm
  videos/cam_side/demo_<i>.mp4     # the 3 POLICY cameras (sensor stream), per demo
  videos/cam_lw/demo_<i>.mp4       #   cam_lw / cam_rw = D405 wrist cams (attached to link_6)
  videos/cam_rw/demo_<i>.mp4       #   cam_side       = D435i side cam
<OUT_DIR>/
  fulldr_third_<N>.mp4             # sqrt(N) tile of every demo's THIRD-PERSON view (10×10 for N=100)
  fourview_demo_<i>.mp4            # 2×2 (third + 3 policy) tiles for 10 random demos
```

The 14-D layout is `STATE_JOINTS_14 = [L_j1..L_j6, L_grip, R_j1..R_j6, R_grip]`. The HDF5 + `videos/` feed the
vendored `_core_vendored/export/lerobot_exporter.py` to produce a LeRobot v3 dataset for pi0.5 fine-tuning.

## Architecture (reusable, agent-native)

`genesis_firefly/collectors/pickplace_collector.py` is the single reusable module:
- `sample_dr(N, rng, lay, spec)` → all per-env DR params (physics + visual).
- `build_scene(N, lay)` → scene + robot + entities + the 4 BatchRenderer cameras.
- `make_compositor(dr, ids)` / `render_all(...)` → per-camera neutral-render × per-env albedo + immersive bg.
- `collect(N, seed, data_dir, out_dir)` → the whole pipeline (DR → grasp/place → record → score → write).

It reuses the sim-agnostic core: `scenes/firefly_scene.py` (tables, `build_bowl`), `scenes/firefly_cameras.py`
(calibrated cam poses/FOVs), `robots/firefly_dual.py`, `robots/ik.py`, `skills/grasp.py`. A new task = a new
collector that swaps the object + place logic and reuses the DR/render/record machinery unchanged.
