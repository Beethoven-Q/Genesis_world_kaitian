# SPDX-License-Identifier: Apache-2.0
"""Graspable-object REGISTRY: one declarative spec per object + a single ``build_object`` factory.

Design goals (owner's brief: clean, reusable, agent-native -- NOT v1's "pile everything together"):
  * ONE place describes every graspable object (asset, mass, measured geometry, grasp hints).
  * ``build_object(spec, ...)`` is the ONLY way the scene spawns one -- realistic collision, settled
    on the table, no per-object code in the scene/collector.
  * Geometry (AABB extents + bbox centre + long axis) is MEASURED from the asset USD
    (scripts/grasp/temp/inspect_object_usd.py) and recorded here as constants, so the grasp skill can
    read an object's long axis / true grasp centre WITHOUT a live USD query. The verify script
    re-measures and flags drift.

COLLISION MODEL (owner cares deeply -- "make everything as realistic as possible"):
  * Every asset USD already bakes a **convexDecomposition** collider (verified) -- a CONCAVE-faithful
    approximation, so the curved banana is a banana (not a fat convex blob) and the bowl is hollow.
    This build has no ``MeshCollisionPropertiesCfg``, so the approximation MUST stay baked (it is).
  * Graspable objects get the SAME firm-contact recipe as the proven cube: 64/4 solver iters (match
    the articulation so the closing claw's contact converges -- no finger-in-object penetration) +
    contact_offset 0.008 / rest_offset 0.0 + friction 1.0/0.9.

The grasp point is the GEOMETRY bbox centre (root_pos + R(root_quat)@local_center), NOT root_pos:
some YCB rigid bodies (banana) pivot off their visual centre. ``local_center``/``extents`` below are
in the rigid-body local frame at scale=1; ``build_object`` applies ``scale``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from robolab.constants import OBJECT_DIR


@dataclass(frozen=True)
class ObjectSpec:
    """Everything needed to spawn + grasp one object. Geometry constants are MEASURED (see module doc)."""
    name: str                       # registry key / prim name
    language_name: str              # natural-language token for the policy instruction
    source: str                     # "usd" | "cuboid"
    mass: float                     # kg (DR multiplies this)
    extents: tuple                  # local AABB size (ex,ey,ez) at scale=1, MEASURED
    local_center: tuple = (0.0, 0.0, 0.0)   # local AABB centre (grasp aims here), MEASURED
    usd_subpath: str | None = None  # under OBJECT_DIR (for source="usd")
    scale: float = 1.0
    elongated: bool = False         # True -> close ACROSS the long axis (banana/pen)
    is_cube: bool = False           # True -> orientation-aware face-pair grasp (close on opposite faces)
    # TRUE principal (long) axis in the rigid-body local frame, from a PCA of the mesh points
    # (scripts/grasp/temp/pca_axis.py). NOT the AABB-argmax basis vector: the banana is authored ~20deg
    # tilted in its local frame, so AABB-Y is 19.7deg off the real tip-to-tip chord -- PCA recovers it.
    local_long_axis: tuple | None = None
    friction: tuple = (1.0, 0.9)    # (static, dynamic)
    color: tuple = (0.85, 0.15, 0.15)       # visual (cuboid only)
    grasp_dz: float = 0.0           # nudge the grasp point along world +Z from the geometry centre
    # height above the bowl centre at which the held object is opened/released. The default 0.05 is a clean
    # top-down drop for compact objects (apple/cube/banana settle where they land). A THIN elongated pen
    # dropped from 5cm bounces/rolls to the bowl rim (or perches across the mouth) -> lower & gentler so it
    # settles in the well. Per-object handling hint (like grasp_dz), so the shared place executor stays generic.
    release_dz: float = 0.05
    # collision contact band (speculative-contact distance) + resting standoff. A THIN object under the
    # firm 10 N.m grip is the hard case: the claw is DRIVEN toward full-close (it wants to close PAST the
    # thin object), and with rest_offset=0 it drives THROUGH the contact and INTERPENETRATES the object,
    # which then lodges on the finger (never releases) -- exactly the pen bug. A small ``rest_offset`` is
    # a hard contact standoff: the claw is stopped ~rest_offset OUTSIDE the object surface (can't be driven
    # in), so there is never finger<->object penetration and the object drops free on release. Thicker
    # round/box objects don't over-drive, so they stay firm at the default 0.
    contact_offset: float = 0.008
    rest_offset: float = 0.0
    # DR sampling defaults (per-object reachable x range; y is mirrored per arm by the eval)
    x_range: tuple = (0.34, 0.42)
    # physical-success XY tolerance (cm): how far off the bowl centre the SETTLED object may rest and still
    # count as placed. A long banana/pen settles farther off-centre than a compact apple, so it's per-object
    # -- kept in the spec (single source of per-object truth) rather than hardcoded in the eval.
    place_xy_tol_cm: float = 8.0

    # --- derived geometry helpers (no USD I/O) ---
    def scaled_extents(self) -> np.ndarray:
        return np.asarray(self.extents, float) * self.scale

    def scaled_center(self) -> np.ndarray:
        return np.asarray(self.local_center, float) * self.scale

    def long_axis_local(self) -> np.ndarray | None:
        """Unit reference axis in the rigid-body local frame, or None for round objects (no preferred
        axis). Cube -> a FACE normal (local +X); elongated -> the PCA principal axis. The grasp skill
        closes PERPENDICULAR to this (across the long axis / onto an opposite face pair)."""
        if self.is_cube:
            return np.array([1.0, 0.0, 0.0])
        if not self.elongated or self.local_long_axis is None:
            return None
        v = np.asarray(self.local_long_axis, float)
        return v / max(1e-9, np.linalg.norm(v))

    def rest_root_z(self, table_top_z: float, gap: float = 0.006) -> float:
        """Root Z so the object's lowest geometry point sits ``gap`` above the table top (then settles).
        lowest_local_z = center_z - extent_z/2 ; root_z = table_top - lowest_local_z + gap."""
        c = self.scaled_center(); e = self.scaled_extents()
        lowest_local_z = c[2] - e[2] / 2.0
        return float(table_top_z - lowest_local_z + gap)


# ============================================================================ #
# THE REGISTRY  (geometry MEASURED by scripts/grasp/temp/inspect_object_usd.py, 2026-06-17)
# ============================================================================ #
REGISTRY: dict[str, ObjectSpec] = {
    # apple_02: clean ~7cm round apple, centred, 34-part convexDecomp. (apple_01 is authored at ~100x
    # scale + off-centre -> messy; apple_02 is the clean choice.) Round -> no preferred grasp axis.
    # local_center=0 -> grasp at the rigid-body ROOT (= the physical/collider centre that physics rests
    # on the table). apple_02's VISUAL bbox centre is ~1.5cm below the collider centre (stray low verts);
    # the root is the robust grasp point for a round body (verified: rests cleanly, no visual sink).
    "apple": ObjectSpec(
        name="apple", language_name="apple", source="usd", usd_subpath="objaverse/apple_02.usd",
        mass=0.050, extents=(0.0702, 0.0754, 0.0733), local_center=(0.0, 0.0, 0.0),
        elongated=False, x_range=(0.34, 0.44), place_xy_tol_cm=7.0),
    # banana: curved -> convexDecomp essential; PCA long axis is ~20deg off local Y (authored tilted).
    "banana": ObjectSpec(
        name="banana", language_name="banana", source="usd", usd_subpath="ycb/banana.usd",
        mass=0.080, extents=(0.1089, 0.1784, 0.0367), local_center=(0.0, 0.0, 0.0),
        elongated=True, local_long_axis=(0.3372, 0.9414, 0.0), grasp_dz=0.0, x_range=(0.34, 0.44),
        place_xy_tol_cm=9.0),
    # pen (dry-erase marker): very elongated, PCA long axis ~= local Y (1.7deg off), ~2cm thick. The thin
    # body is the ONE case the firm 10 N.m close over-drives: with rest_offset=0 the claw closes PAST the
    # pen surface (measured grasp q=0.61, well past the q~0.45 "claws-at-surface" point) and interpenetrates
    # -> the pen lodges on a finger and never releases. rest_offset=0.004 is a hard contact standoff that
    # stops the claw ON the surface: grip stays firm (still lifts 13cm) and the pen drops free on release
    # (diag_pen_grip.py sweep: q=0 STUCK; >=0.003 FELL cleanly; 0.010 starts loosening). Pen-only -- thicker
    # round/box objects never over-drive, so they keep the default 0 (no table-rest gap).
    "pen": ObjectSpec(
        name="pen", language_name="pen", source="usd", usd_subpath="ycb/dry_erase_marker.usd",
        mass=0.020, extents=(0.0210, 0.1208, 0.0189), local_center=(0.0, 0.0, 0.0),
        elongated=True, local_long_axis=(-0.0303, 0.9995, 0.0), grasp_dz=0.0, rest_offset=0.004,
        release_dz=0.03, x_range=(0.34, 0.44), place_xy_tol_cm=9.0),
    # cube: programmatic 5cm box (clean collider, known size/mass). Orientation-aware: grasp two OPPOSITE
    # faces (close along a face normal), never across the diagonal -> is_cube path in the grasp skill.
    "cube": ObjectSpec(
        name="cube", language_name="cube", source="cuboid", mass=0.040,
        extents=(0.05, 0.05, 0.05), local_center=(0.0, 0.0, 0.0), is_cube=True,
        color=(0.85, 0.15, 0.15), x_range=(0.34, 0.44)),
}


def _firm_contact_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    # 64/4 matches the articulation + the proven cube, so the closing-claw contact CONVERGES (PhysX caps
    # depenetration by the LOWER of the two bodies' position iters) -> firm grip, no finger penetration.
    return sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=64, solver_velocity_iteration_count=4,
        max_depenetration_velocity=1.0)


def build_object(spec: ObjectSpec, pos_xy: tuple, table_top_z: float) -> RigidObjectCfg:
    """Spawn ``spec`` resting on the object table at ``pos_xy`` with a realistic (convexDecomp/box)
    collider + firm-contact physics. Returns a RigidObjectCfg(prim_path="{ENV_REGEX_NS}/object")."""
    x, y = pos_xy
    z = spec.rest_root_z(table_top_z)
    common = dict(
        rigid_props=_firm_contact_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=spec.contact_offset, rest_offset=spec.rest_offset),
    )
    if spec.source == "cuboid":
        spawn = sim_utils.CuboidCfg(
            size=tuple(spec.extents),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=spec.friction[0], dynamic_friction=spec.friction[1]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color, roughness=0.6),
            **common)
    else:
        spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(OBJECT_DIR, spec.usd_subpath),
            scale=(spec.scale,) * 3,
            # convexDecomposition collider is BAKED in the asset (verified) -- do NOT override.
            **common)
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/object", spawn=spawn,
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z)))
