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


def firm_rigid_options(dt=0.01, noslip=0):
    """The Genesis penetration fix for firm grasps: Newton solver + many iters + low constraint timeconst
    + noslip (reproduces RoboLab's PhysX 64/4 + rest_offset firm-contact recipe).

    noslip_iterations (2026-06-20 ROUND-OBJECT FIX): the docstring CLAIMED "+ noslip" but the param was never
    actually set, so it defaulted to 0 -- the friction cone stayed soft/leaky. A flat-faced cube is caged by two
    coplanar contact patches, but a ROUND/curved/THIN body reduces to a single tangent contact per finger, and a
    leaky cone lets the firm GR100 pinch (GRIP_KP=200) squirt it out (verified: apple/tennis EJECT 10-40cm; the
    thin pen slips, 0/12 grasp). This is the PhysX-vs-Genesis divergence: PhysX's velocity-iteration friction
    solve (solver_velocity_iteration_count=4 in RoboLab) tightens the cone; the Genesis analogue is
    noslip_iterations. Turning it on makes apple/tennis/banana/pen grasp at the SAME simple top-down(+tilt) path
    as the cube -- NO round-grasp special machinery (deep seat / tilt cap / approach reorient).

    PER-OBJECT (the caller passes ``noslip``): a tighter cone RAISES the firm-pinch contact force, which deepens
    penetration -- the flat-faced CUBE needs NO noslip (it holds with a leaky cone) and a tight cone pushes its
    pinch OVER the 7mm abnormal gate on the odd env (verified: cube env1 2.5mm@noslip0 -> 7.7mm@noslip2, a
    REGRESSION). So the CUBE uses noslip=0 (its locked 2.5mm) and the round/curved/thin objects use noslip=5
    (which holds them while their own contact stays ~6mm). The collector derives the value from the target spec
    (cube -> 0, else -> 5); NOSLIP_ITERS env overrides for ablation."""
    import os as _os
    env = _os.environ.get("NOSLIP_ITERS")
    if env is not None:
        noslip = int(env)
    return gs.options.RigidOptions(
        dt=dt, constraint_solver=gs.constraint_solver.Newton, iterations=120,
        constraint_timeconst=0.005, enable_self_collision=True, enable_collision=True,
        noslip_iterations=noslip,                # tighten the friction cone -> a round body doesn't squirt out
        integrator=gs.integrator.implicitfast)   # exact MuJoCo/Isaac implicit PD (the approximate default
    #                                              under-damps J5/J6 -> wrist jitter). Faithful to RoboLab.


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


# ``build_object`` now lives in the ONE shared object builder ``world/object_factory.py`` (consolidated so there
# is a SINGLE spec->sim-entity path for every task — collision + visual + native texture). Re-exported here for
# back-compat: any historical ``from world.firefly_scene import build_object`` keeps working unchanged.
def build_object(scene, spec, pos_xy, table_top_z, mass=None):
    """Spawn one graspable object from its ObjectSpec (source = usd | cuboid | sphere), resting on the table.

    Thin shim -> ``world.object_factory.build_object`` (the consolidated single object-building path). Imported
    lazily to avoid a circular import (object_factory imports OBJECTS/_rho_for from this module)."""
    from world.object_factory import build_object as _build_object
    return _build_object(scene, spec, pos_xy, table_top_z, mass=mass)


def build_bowl(scene, bowl_xy, table_top_z, scale=1.0, surface=None):
    z = table_top_z + BOWL_HALF_H * scale + 0.003
    kw = {"surface": surface} if surface is not None else {}
    # REALISTIC SOLID collision model = convex DECOMPOSITION (coacd) -- the SAME approach RoboLab/PhysX uses
    # (baked convexDecomposition). The thin concave bowl shell is split into a set of SOLID convex hulls
    # tiling the wall+floor; convex-vs-convex contact is robust EVERYWHERE, including the thin rim, so a cube
    # that lands on the rim rolls off/in and can NEVER pass through the wall. (The single nonconvex-SDF
    # envelope gives a degenerate ~0 contact where a corner straddles the 2-3mm rim -> tunnelling.) The cavity
    # stays open (cube settles inside); coacd raises the rest height a few mm, which RoboLab also accepts.
    return scene.add_entity(
        gs.morphs.USD(file=str(BOWL_USD), pos=(bowl_xy[0], bowl_xy[1], z), scale=scale,
                      convexify=True, decompose_object_error_threshold=0.04, decimate=False),
        material=gs.materials.Rigid(rho=400.0, friction=1.0), **kw)
