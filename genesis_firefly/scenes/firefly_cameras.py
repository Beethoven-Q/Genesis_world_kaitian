# SPDX-License-Identifier: Apache-2.0
"""The 3 policy cameras + the visible side-camera rig — Genesis reproduction of RoboLab's firefly_cameras.py.

EXACT calibrated reproduction:
  - 2 wrist RealSense D405 (vfov 57.95deg), attached to each *_link_6 at the calibrated optical pose.
  - 1 world-fixed side D435i (vfov 43.2deg) at the calibrated pose from side.json.
  - The side camera's visible BODY (collidable, fixed) on a SOLID support STICK (r=8mm), like the real rig.
Conventions: Genesis cameras are OpenGL (-Z fwd, +Y up) — the SAME convention RoboLab baked, so the
calibrated rot_opengl quats (wxyz) transfer directly. Render keys: cam_lw, cam_rw, cam_side (LeRobot).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import genesis as gs

# Intel RealSense D435i body mesh (the real device), used for the visible side-camera body.
CAMERA_BODY_MESH = str(Path(__file__).resolve().parents[1]
                       / "assets/robots/firefly_y6_gr100/source/firefly_y6/camera_link.STL")

# --- intrinsics -> Genesis vertical FOV (deg). focal 24, apertures from the calibrated D405/D435i. ---
_FOCAL = 24.0
WRIST_VFOV = float(np.degrees(2 * np.arctan(26.5769 / 2 / _FOCAL)))   # 57.95 deg (D405)
SIDE_VFOV = float(np.degrees(2 * np.arctan(19.004544 / 2 / _FOCAL)))  # 43.2 deg (D435i)
RES = (640, 360)   # 16:9 policy framing (matches RoboLab)

# --- calibrated poses (pos, rot_opengl wxyz). wrist = in link_6 frame; side = world. ---
LEFT_WRIST = (np.array([-0.0810919, 0.0098250, 0.0452576]),
              np.array([0.1236164, 0.6853585, -0.7037882, -0.1403031]))
RIGHT_WRIST = (np.array([-0.0775236, 0.0049461, 0.0478084]),
               np.array([0.1369354, 0.7101610, -0.6780283, -0.1311402]))
SIDE = (np.array([0.0450464, 0.0325720, 0.8155021]),
        np.array([0.6637329, 0.1897607, -0.1917309, -0.6976309]))
# visible side rig — EXACT RoboLab values (firefly_cameras.py SIDE_CAM_BODY/SIDE_CAM_STICK): the real D435i
# body MESH sits at the calibrated device-centre pose (optical + 0.0325m along the baseline) with the mesh
# orientation RoboLab calibrated; the STICK rises from the floor to that xy to hold it up.
SIDE_BODY_POS = np.array([0.0435222, 0.0001094, 0.8151689])
SIDE_BODY_QUAT = np.array([0.0659103, -0.3983885, -0.6888077, -0.6020684])   # mesh rot (wxyz), calibrated
SIDE_STICK_R, SIDE_STICK_H = 0.008, 0.8151689           # ground -> camera (r=8mm)


def _R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _T(pos, quat):
    T = np.eye(4); T[:3, :3] = _R(quat); T[:3, 3] = pos
    return T


def add_policy_cameras(scene):
    """Add the 3 policy cameras. Returns a dict {cam_lw, cam_rw, cam_side} of Genesis cameras.
    Wrist cams are placed initially; call ``attach_wrist_cams(robot, cams)`` AFTER build to bolt them to
    link_6, then ``update_wrist_cams(cams)`` each step."""
    cams = {
        "cam_lw": scene.add_camera(res=RES, pos=(0.3, 0.3, 0.6), lookat=(0.3, 0.2, 0.3), fov=WRIST_VFOV, GUI=False),
        "cam_rw": scene.add_camera(res=RES, pos=(0.3, -0.3, 0.6), lookat=(0.3, -0.2, 0.3), fov=WRIST_VFOV, GUI=False),
        "cam_side": scene.add_camera(res=RES, pos=tuple(SIDE[0]), lookat=(0.35, 0.0, 0.3), fov=SIDE_VFOV, GUI=False),
    }
    return cams


def attach_wrist_cams(robot, cams):
    """Bolt wrist cams to each link_6 at the calibrated optical pose (call after scene.build())."""
    cams["cam_lw"].attach(robot.entity.get_link("left_link_6"), _T(*LEFT_WRIST))
    cams["cam_rw"].attach(robot.entity.get_link("right_link_6"), _T(*RIGHT_WRIST))
    cams["cam_side"].set_pose(transform=_T(*SIDE))   # world-fixed; set once


def update_wrist_cams(cams):
    """Follow link_6 — call every sim step before rendering the wrist views."""
    cams["cam_lw"].move_to_attach()
    cams["cam_rw"].move_to_attach()


def add_side_camera_rig(scene, body_surface=None, stick_surface=None):
    """The visible, collidable, world-fixed side-camera BODY (real D435i mesh) + support STICK — faithful to
    RoboLab's SIDE_CAM_BODY / SIDE_CAM_STICK. The mesh entity honours its own surface in Nyx (per-vgeom), so
    the body renders dark like a real RealSense; the stick is a thin dark pole from the floor to the camera."""
    body = scene.add_entity(
        gs.morphs.Mesh(file=CAMERA_BODY_MESH, pos=tuple(SIDE_BODY_POS), quat=tuple(SIDE_BODY_QUAT),
                       fixed=True, collision=True, convexify=True),
        surface=body_surface or gs.surfaces.Plastic(color=(0.13, 0.13, 0.14), roughness=0.5))
    stick = scene.add_entity(
        gs.morphs.Cylinder(radius=SIDE_STICK_R, height=SIDE_STICK_H,
                           pos=(float(SIDE_BODY_POS[0]), float(SIDE_BODY_POS[1]), SIDE_STICK_H / 2),
                           fixed=True, collision=True),
        surface=stick_surface or gs.surfaces.Plastic(color=(0.25, 0.25, 0.28), roughness=0.6))
    return body, stick
