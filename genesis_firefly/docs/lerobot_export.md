# Genesis -> LeRobot v2 export (for pi0.5 fine-tuning)

Convert a Genesis full-DR pick-place dataset (merged `demos.hdf5` + per-camera MP4s)
into a **LeRobot v2.1** dataset that OpenPI / pi0.5 training consumes directly. This is
the Genesis "Line B" analogue of the proven RoboLab/OpenPI converter
(`openpi_hex/examples/hexarm/convert_robolab_demos_to_lerobot.py`) and produces the
identical layout, so the downstream norm-stats + fine-tune commands are the same apart
from the `--repo-id`.

| | |
|---|---|
| **Converter** | `genesis_firefly/dataio/convert_genesis_to_lerobot.py` |
| **Interpreter** | `/home/kaitianchao/Projects/openpi_hex/.venv/bin/python` (has `lerobot==0.1.0`, `CODEBASE_VERSION=v2.1`) — the Genesis repo venv does **not** have `lerobot` |
| **Source** | `/data3/genesis_fulldr/cube_fulldr_v2/` (`demos.hdf5` + `videos/cam_{side,lw,rw}/demo_<i>.mp4`) |
| **Output** | `/data3/genesis_fulldr/lerobot/genesis_cube_fulldr_v2/` (the `HF_LEROBOT_HOME/<repo-id>` convention) |
| **In-workspace preview** | `genesis_firefly/output/genesis_cube_fulldr_v2` symlink -> the /data3 output |

## Why a separate converter (not the vendored `dataio/lerobot_exporter.py`)

The vendored `lerobot_exporter.py` emits **LeRobot v3.0** and is hardcoded to RoboLab's
HDF5 group names + `__camera`-suffixed video filenames. pi0.5 / OpenPI training wants
**v2.1**, and the Genesis source has a different layout (single merged HDF5, separate
per-camera MP4s, `states/articulation/robot/joint_position`). So we use the v2.1
`lerobot` package path (same as the proven RoboLab pipeline) with a Genesis-specific
reader.

## Camera + feature mapping

```text
states/articulation/robot/joint_position  -> observation.state            (14-D float32)
actions                                    -> action                        (14-D float32)
videos/cam_side/demo_<i>.mp4               -> observation.images.cam_high         (3, 368, 640)
videos/cam_lw/demo_<i>.mp4                 -> observation.images.cam_left_wrist   (3, 368, 640)
videos/cam_rw/demo_<i>.mp4                 -> observation.images.cam_right_wrist  (3, 368, 640)
"put the cube in the bowl"                 -> task / prompt
```

14-D layout: `[L_j1..L_j6, L_grip, R_j1..R_j6, R_grip]`. fps = 12 (probed from the video).
Image height/width are probed from the source video (640x368 for `cube_fulldr_v2`).

## success_only gate

A Genesis demo is included only if **ALL** hold (from the HDF5 demo attrs):

```text
success == True   AND   penetrating == False   AND   degenerate == False
```

For `cube_fulldr_v2` this is **200/200** demos (the dataset is already 100% clean).

## Run the export

```bash
cd /home/kaitianchao/Projects/Genesis_world_kaitian

HF_LEROBOT_HOME=/data3/genesis_fulldr/lerobot \
/home/kaitianchao/Projects/openpi_hex/.venv/bin/python \
  genesis_firefly/dataio/convert_genesis_to_lerobot.py \
  --source /data3/genesis_fulldr/cube_fulldr_v2 \
  --repo-id genesis_cube_fulldr_v2 \
  --task "put the cube in the bowl"
```

Optional flags: `--no-success-only` (keep every demo), `--fps N` (override the probed
12), `--mode image` (store raw frames instead of re-encoded video), `--robot-type STR`.

Output dataset (verified: `total_episodes=200`, `total_frames=21460`, `fps=12`, 575 MB):

```bash
/data3/genesis_fulldr/lerobot/genesis_cube_fulldr_v2/
  meta/{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl}
  data/chunk-000/episode_000000.parquet ... episode_000199.parquet
  videos/chunk-000/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_*.mp4
```

> **Video codec note:** `lerobot==0.1.0` encodes the episode MP4s as **AV1**
> (`info.json` -> `video.codec: "av1"`), not avc1. torchvision/pyav decode AV1 fine
> (the load-check below reads all 3 image streams), and this is exactly what the proven
> RoboLab pipeline produces with the same package — it is a property of the lerobot
> package, not a defect.

Then add the preview symlink (large data stays on /data3):

```bash
ln -sfn /data3/genesis_fulldr/lerobot/genesis_cube_fulldr_v2 \
  genesis_firefly/output/genesis_cube_fulldr_v2
```

## Verify it loaded (sanity check)

```bash
HF_LEROBOT_HOME=/data3/genesis_fulldr/lerobot \
/home/kaitianchao/Projects/openpi_hex/.venv/bin/python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("genesis_cube_fulldr_v2")
print("episodes", ds.num_episodes, "frames", ds.num_frames, "fps", ds.fps)
s = ds[0]
for k in ["observation.state","action",
          "observation.images.cam_high",
          "observation.images.cam_left_wrist",
          "observation.images.cam_right_wrist"]:
    print(k, tuple(s[k].shape), s[k].dtype)
print("task:", s["task"])
PY
```

## pi0.5 fine-tune (adapted from the RoboLab native pipeline)

These mirror `RoboLab/docs/hex_pi05_native_pipeline.md`. They require an OpenPI training
config whose `repo_id` points at `genesis_cube_fulldr_v2` (clone `pi05_hexarm_bowl_lora`
to e.g. `pi05_genesis_cube_lora` in `openpi_hex/src/openpi/training/config.py`, set its
LeRobot `repo_id="genesis_cube_fulldr_v2"`, keep the 14-D `HexInputs`/`HexOutputs`
transforms and the 3 cam names). The state/action contract is identical to the RoboLab
HexArm dataset, so no transform changes are needed.

**1. Compute norm stats** (run once after conversion):

```bash
cd /home/kaitianchao/Projects/openpi_hex

OPENPI_DATA_HOME=/data3/kaitian/pi05finetune/cache/openpi \
HF_HOME=/data3/kaitian/pi05finetune/cache/huggingface \
HF_DATASETS_CACHE=/data3/kaitian/pi05finetune/cache/huggingface/datasets \
HF_LEROBOT_HOME=/data3/genesis_fulldr/lerobot \
UV_CACHE_DIR=/data3/kaitian/pi05finetune/cache/uv \
JAX_COMPILATION_CACHE_DIR=/data3/kaitian/pi05finetune/cache/jax \
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_genesis_cube_lora
```

**2. Fine-tune pi0.5:**

```bash
cd /home/kaitianchao/Projects/openpi_hex

OPENPI_DATA_HOME=/data3/kaitian/pi05finetune/cache/openpi \
HF_HOME=/data3/kaitian/pi05finetune/cache/huggingface \
HF_DATASETS_CACHE=/data3/kaitian/pi05finetune/cache/huggingface/datasets \
HF_LEROBOT_HOME=/data3/genesis_fulldr/lerobot \
UV_CACHE_DIR=/data3/kaitian/pi05finetune/cache/uv \
JAX_COMPILATION_CACHE_DIR=/data3/kaitian/pi05finetune/cache/jax \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
./.venv/bin/python scripts/train.py \
  pi05_genesis_cube_lora \
  --exp-name genesis_cube_fulldr_v2_lora
```

Checkpoints land under
`/data3/kaitian/pi05finetune/checkpoints/pi05_genesis_cube_lora/genesis_cube_fulldr_v2_lora/`.
Do not commit checkpoints or the LeRobot dataset (large; live on /data3).
