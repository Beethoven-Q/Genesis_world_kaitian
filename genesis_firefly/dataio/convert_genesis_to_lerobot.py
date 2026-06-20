"""Convert a Genesis full-DR pick-place dataset to LeRobot v2.1 format for pi0.5.

This is the Genesis "Line B" analogue of the proven RoboLab/OpenPI converter
``openpi_hex/examples/hexarm/convert_robolab_demos_to_lerobot.py``. It produces
exactly the same LeRobot v2.1 layout (the format pi0.5 / OpenPI training and
``compute_norm_stats`` consume), so the downstream fine-tune commands are
identical apart from the ``--repo-id``.

It MUST be run with an interpreter that has the ``lerobot`` package installed.
The Genesis repo venv does NOT have it, so use the OpenPI venv (which does):

    cd /home/kaitianchao/Projects/Genesis_world_kaitian

    HF_LEROBOT_HOME=/data3/genesis_fulldr/lerobot \\
    /home/kaitianchao/Projects/openpi_hex/.venv/bin/python \\
      genesis_firefly/dataio/convert_genesis_to_lerobot.py \\
      --source /data3/genesis_fulldr/cube_fulldr_v2 \\
      --repo-id genesis_cube_fulldr_v2 \\
      --task "put the cube in the bowl"

Genesis source layout (one merged HDF5 + per-camera MP4s, NOT embedded image
arrays like RoboLab):

    <source>/demos.hdf5
        /data/demo_<i>
            actions                                  (T, 14)  float32
            states/articulation/robot/joint_position (T, 14)  float32
            states/articulation/robot/joint_velocity (T, 14)  float32
            ee_pose/{position,orientation}
            attrs: success, penetrating, degenerate, arm, seed, build, ...
    <source>/videos/cam_side/demo_<i>.mp4   (T, H, W, 3) H.264
    <source>/videos/cam_lw/demo_<i>.mp4
    <source>/videos/cam_rw/demo_<i>.mp4

14-D layout: [L_j1..L_j6, L_grip, R_j1..R_j6, R_grip].

Mapping to LeRobot / OpenPI standard names:

    joint_position  -> observation.state           (14-D)
    actions         -> action                       (14-D)
    cam_side        -> observation.images.cam_high
    cam_lw          -> observation.images.cam_left_wrist
    cam_rw          -> observation.images.cam_right_wrist
    <task string>   -> task / prompt

success_only gate (a Genesis demo ships only if ALL hold):
    success == True  AND  penetrating == False  AND  degenerate == False
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import shutil
from typing import Literal

# Keep the large generated LeRobot dataset off the workspace by default. The
# caller can override HF_LEROBOT_HOME before launching.
os.environ.setdefault("HF_LEROBOT_HOME", "/data3/genesis_fulldr/lerobot")

import cv2
import h5py
import numpy as np
import torch
import tqdm
import tyro

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


MOTORS = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
]

# LeRobot/OpenPI camera name -> Genesis per-camera video subdir.
CAMERA_MAP = {
    "cam_high": "cam_side",
    "cam_left_wrist": "cam_lw",
    "cam_right_wrist": "cam_rw",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


def _is_clean(attrs) -> bool:
    """success_only gate: success AND not penetrating AND not degenerate."""
    success = bool(attrs.get("success", False))
    penetrating = bool(attrs.get("penetrating", False))
    degenerate = bool(attrs.get("degenerate", False))
    return success and not penetrating and not degenerate


def _read_video_frames(path: Path, expected: int | None = None) -> np.ndarray:
    """Decode an MP4 to an (T, H, W, 3) uint8 RGB array (cv2 returns BGR)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[:, :, ::-1])  # BGR -> RGB
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    arr = np.ascontiguousarray(np.stack(frames))
    if expected is not None and arr.shape[0] != expected:
        raise ValueError(
            f"{path} decoded {arr.shape[0]} frames, expected {expected}"
        )
    return arr


def _probe_image_shape(source: Path) -> tuple[int, int]:
    """Probe (height, width) from the first cam_side video."""
    sample = next((source / "videos" / "cam_side").glob("*.mp4"))
    cap = cv2.VideoCapture(str(sample))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return h, w


def _probe_fps(source: Path) -> int:
    sample = next((source / "videos" / "cam_side").glob("*.mp4"))
    cap = cv2.VideoCapture(str(sample))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(fps)) if fps and fps > 0 else 12


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    img_hw: tuple[int, int],
    mode: Literal["video", "image"],
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    h, w = img_hw
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    for camera in CAMERA_MAP:
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }

    output_path = Path(HF_LEROBOT_HOME) / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _select_demos(source: Path, success_only: bool) -> list[str]:
    """Return sorted demo names passing the success_only gate."""
    with h5py.File(source / "demos.hdf5", "r") as f:
        data = f["data"]
        names = [k for k in data.keys() if k.startswith("demo_")]
        names.sort(key=lambda x: int(x.split("_")[1]))
        if success_only:
            names = [n for n in names if _is_clean(data[n].attrs)]
    return names


def populate_dataset(
    dataset: LeRobotDataset,
    source: Path,
    demo_names: list[str],
    task: str,
) -> LeRobotDataset:
    with h5py.File(source / "demos.hdf5", "r") as f:
        data = f["data"]
        for name in tqdm.tqdm(demo_names, desc="Converting episodes"):
            d = data[name]
            state = torch.from_numpy(
                np.asarray(d["states"]["articulation"]["robot"]["joint_position"]).astype(np.float32)
            )
            action = torch.from_numpy(np.asarray(d["actions"]).astype(np.float32))
            num_frames = state.shape[0]
            if action.shape[0] != num_frames:
                raise ValueError(
                    f"{name} state/action length mismatch: {num_frames} vs {action.shape[0]}"
                )

            images = {}
            for camera, subdir in CAMERA_MAP.items():
                vid = source / "videos" / subdir / f"{name}.mp4"
                images[camera] = _read_video_frames(vid, expected=num_frames)

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task,
                }
                for camera, frames in images.items():
                    frame[f"observation.images.{camera}"] = frames[i]
                dataset.add_frame(frame)
            dataset.save_episode()
    return dataset


def convert_genesis_demos(
    source: Path = Path("/data3/genesis_fulldr/cube_fulldr_v2"),
    repo_id: str = "genesis_cube_fulldr_v2",
    *,
    task: str = "put the cube in the bowl",
    success_only: bool = True,
    fps: int | None = None,
    robot_type: str = "hex_archer_l6y_gp100_dual",
    mode: Literal["video", "image"] = "video",
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DatasetConfig(),
) -> None:
    source = Path(source)
    if not (source / "demos.hdf5").exists():
        raise FileNotFoundError(f"No demos.hdf5 under {source}")

    demo_names = _select_demos(source, success_only=success_only)
    if not demo_names:
        raise ValueError(f"No demos passed the success_only={success_only} gate in {source}")

    img_hw = _probe_image_shape(source)
    if fps is None:
        fps = _probe_fps(source)

    print(f"HF_LEROBOT_HOME={HF_LEROBOT_HOME}")
    print(f"source={source}")
    print(f"repo_id={repo_id}")
    print(f"episodes={len(demo_names)} success_only={success_only} fps={fps} img_hw={img_hw}")
    print(f"task={task!r}")

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        img_hw=img_hw,
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(dataset, source, demo_names, task)
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()

    print(f"DONE: {Path(HF_LEROBOT_HOME) / repo_id}")


if __name__ == "__main__":
    tyro.cli(convert_genesis_demos)
