# SPDX-License-Identifier: Apache-2.0
"""The firefly pick-place WORLD in Genesis — tables, the object factory, the bowl, robot+livery+cameras+rig.
Genesis reproduction of RoboLab's firefly_scene.py + firefly_pickplace_scene.py + firefly_play_scene.py.

Layout (reproduced exactly): arm-table 0.34x0.90 (top z=0.25, under the robot), object-table 0.70x0.90
(top z=0.25, in front), flush seam at x=0.13. Objects rest on the object table; the bowl is the place target.
Contact: firm-grip RigidOptions (Newton solver, many iters, low timeconst, noslip) — the Genesis penetration
fix (RoboLab's PhysX 64/4 + rest_offset equivalent). build_object handles usd / cuboid / sphere specs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import genesis as gs

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_core_vendored"))
from object_spec import REGISTRY  # noqa: E402

from robots.firefly_dual import FireflyDual  # noqa: E402
from robots.livery import apply_livery  # noqa: E402
from scenes.firefly_cameras import (add_policy_cameras, attach_wrist_cams, add_side_camera_rig)  # noqa: E402

OBJECTS = Path(__file__).resolve().parents[1] / "assets/objects"
BOWL_USD = OBJECTS / "ycb/bowl.usd"
BOWL_HALF_H = 0.02748


@dataclass
class TableLayout:
    arm_table_height: float = 0.25
    object_table_height: float = 0.25
    common_width: float = 0.90
    arm_table_depth: float = 0.34
    object_table_depth: float = 0.70
    seam_x: float = 0.13


def firm_rigid_options(dt=0.01):
    """The Genesis penetration fix for firm grasps: Newton solver + many iters + low constraint timeconst
    + noslip (reproduces RoboLab's PhysX 64/4 + rest_offset firm-contact recipe)."""
    return gs.options.RigidOptions(
        dt=dt, constraint_solver=gs.constraint_solver.Newton, iterations=120,
        constraint_timeconst=0.005, enable_self_collision=True, enable_collision=True)


def add_tables(scene, layout: TableLayout):
    ha, ho = layout.arm_table_height, layout.object_table_height
    arm = scene.add_entity(gs.morphs.Box(
        size=(layout.arm_table_depth, layout.common_width, ha),
        pos=(layout.seam_x - layout.arm_table_depth / 2.0, 0.0, ha / 2.0), fixed=True, collision=True),
        surface=gs.surfaces.Plastic(color=(0.30, 0.30, 0.33), roughness=0.7))
    obj = scene.add_entity(gs.morphs.Box(
        size=(layout.object_table_depth, layout.common_width, ho),
        pos=(layout.seam_x + layout.object_table_depth / 2.0, 0.0, ho / 2.0), fixed=True, collision=True),
        surface=gs.surfaces.Plastic(color=(0.30, 0.30, 0.33), roughness=0.7))
    return arm, obj


def _rho_for(spec, mass=None):
    """Density so the object's mass matches the spec (Genesis Rigid uses density, not mass)."""
    m = spec.mass if mass is None else mass
    e = spec.scaled_extents()
    if spec.source == "sphere":
        vol = 4.0 / 3.0 * np.pi * (e[0] / 2) ** 3
    elif spec.source == "cuboid":
        vol = e[0] * e[1] * e[2]
    else:                       # usd mesh: approx by the AABB filled ~45%
        vol = e[0] * e[1] * e[2] * 0.45
    return float(m / max(vol, 1e-6))


def build_object(scene, spec, pos_xy, table_top_z, mass=None):
    """Spawn one graspable object from its ObjectSpec (source = usd | cuboid | sphere), resting on the table."""
    x, y = pos_xy
    z = spec.rest_root_z(table_top_z)
    mat = gs.materials.Rigid(rho=_rho_for(spec, mass), friction=float(spec.friction[0]))
    if spec.source == "cuboid":
        morph = gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(x, y, z))
        surf = gs.surfaces.Plastic(color=spec.color, roughness=0.6)
    elif spec.source == "sphere":
        morph = gs.morphs.Sphere(radius=float(spec.scaled_extents()[0] / 2), pos=(x, y, z))
        surf = gs.surfaces.Rough(color=spec.color)
    else:
        morph = gs.morphs.USD(file=str(OBJECTS / spec.usd_subpath), pos=(x, y, z),
                              scale=spec.scale, convexify=True)
        surf = None
    return scene.add_entity(morph, material=mat, surface=surf)


def build_bowl(scene, bowl_xy, table_top_z, scale=1.0):
    z = table_top_z + BOWL_HALF_H * scale + 0.003
    return scene.add_entity(
        gs.morphs.USD(file=str(BOWL_USD), pos=(bowl_xy[0], bowl_xy[1], z), scale=scale, convexify=False),
        material=gs.materials.Rigid(rho=400.0, friction=1.0))


class PickPlaceWorld:
    """Assembles the full pick-place world; build() then finalize()."""

    def __init__(self, object_name: str, object_xy=(0.38, 0.17), bowl_xy=(0.40, 0.02),
                 layout: TableLayout | None = None, mass=None, with_cameras=True):
        self.spec = REGISTRY[object_name]
        self.layout = layout or TableLayout()
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01),
                              rigid_options=firm_rigid_options(), show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = FireflyDual(self.scene, pos=(0, 0, self.layout.arm_table_height))
        apply_livery(self.robot)
        add_tables(self.scene, self.layout)
        self.obj = build_object(self.scene, self.spec, object_xy, self.layout.object_table_height, mass)
        self.bowl = build_bowl(self.scene, bowl_xy, self.layout.object_table_height)
        self.bowl_xy = bowl_xy
        self.bowl_rest_z = self.layout.object_table_height + BOWL_HALF_H + 0.003
        self.cams = add_policy_cameras(self.scene) if with_cameras else None
        if with_cameras:
            add_side_camera_rig(self.scene)
        self.dbg = self.scene.add_camera(res=(640, 360), pos=(1.3, -0.9, 0.95),
                                         lookat=(0.25, 0.0, 0.35), fov=42, GUI=False)

    def build(self):
        self.scene.build()
        self.robot.finalize()
        if self.cams is not None:
            attach_wrist_cams(self.robot, self.cams)
        return self
