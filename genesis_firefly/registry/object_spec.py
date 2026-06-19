# SPDX-License-Identifier: Apache-2.0
"""Graspable-object REGISTRY (DATA ONLY — pure, sim-agnostic). The Genesis spawn factory lives in the scene
module (build_object), so this file has NO engine imports and is shared verbatim by both lines.

One declarative ``ObjectSpec`` per object: asset, mass, MEASURED geometry (AABB extents + bbox centre + PCA
long axis), and per-object grasp/handling hints (friction, contact/rest_offset, grasp_dz, release_dz,
place_xy_tol_cm, x_range). Adding an object = adding a spec entry; no per-object code anywhere.

Sources: "usd" (asset under assets/objects/<usd_subpath>), "cuboid" (procedural box, size=extents),
"sphere" (procedural ball, radius=extents[0]/2). Geometry constants MEASURED in RoboLab (inspect_object_usd.py
/ pca_axis.py), reused unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    """Everything needed to spawn + grasp one object. Geometry constants are MEASURED."""
    name: str
    language_name: str
    source: str                     # "usd" | "cuboid" | "sphere"
    mass: float                     # kg (DR multiplies this)
    extents: tuple                  # local AABB size (ex,ey,ez) at scale=1, MEASURED
    local_center: tuple = (0.0, 0.0, 0.0)
    usd_subpath: str | None = None  # under assets/objects/ (for source="usd")
    mesh_subpath: str | None = None  # Nyx-SAFE clean .obj (under assets/objects/) extracted from the USD; used
    #                                  for rendering distractors (Nyx SEGFAULTS on the textured USD material bind)
    dist_color: tuple | None = None  # realistic flat colour for the clean-mesh distractor render (USD texture is
    #                                  lost when extracting the OBJ); None -> fall back to `color`
    scale: float = 1.0
    elongated: bool = False         # True -> close ACROSS the long axis (banana/pen)
    is_cube: bool = False           # True -> orientation-aware face-pair grasp
    local_long_axis: tuple | None = None   # PCA principal axis in the rigid-body local frame
    friction: tuple = (1.0, 0.9)
    color: tuple = (0.85, 0.15, 0.15)       # visual (procedural cuboid/sphere)
    grasp_dz: float = 0.0           # nudge the grasp point along world +Z from the geometry centre
    release_dz: float = 0.05        # height above the bowl centre to open/release (thin pen -> lower)
    contact_offset: float = 0.008   # speculative-contact band
    rest_offset: float = 0.0        # hard contact standoff (thin pen -> 0.004 to stop claw ON the surface)
    x_range: tuple = (0.34, 0.42)   # DR reachable x range (y mirrored per arm)
    place_xy_tol_cm: float = 8.0    # physical-success XY tolerance from the bowl centre

    # --- derived geometry helpers (no I/O) ---
    def scaled_extents(self) -> np.ndarray:
        return np.asarray(self.extents, float) * self.scale

    def scaled_center(self) -> np.ndarray:
        return np.asarray(self.local_center, float) * self.scale

    def long_axis_local(self) -> np.ndarray | None:
        """Unit reference axis in the local frame, or None for round objects. Cube -> a FACE normal
        (local +X); elongated -> the PCA principal axis. The grasp closes PERPENDICULAR to this."""
        if self.is_cube:
            return np.array([1.0, 0.0, 0.0])
        if not self.elongated or self.local_long_axis is None:
            return None
        v = np.asarray(self.local_long_axis, float)
        return v / max(1e-9, np.linalg.norm(v))

    def rest_root_z(self, table_top_z: float, gap: float = 0.006) -> float:
        """Root Z so the object's lowest geometry point sits ``gap`` above the table top (then settles)."""
        c = self.scaled_center(); e = self.scaled_extents()
        lowest_local_z = c[2] - e[2] / 2.0
        return float(table_top_z - lowest_local_z + gap)


# ============================================================================ #
# THE REGISTRY  (geometry MEASURED in RoboLab, 2026-06-17; tennis_ball added 2026-06-18 for the Genesis gate;
#                book added + Nyx-safe clean .obj meshes (apple/banana/pen) added 2026-06-19 for distractors)
# ============================================================================ #
REGISTRY: dict[str, ObjectSpec] = {
    "apple": ObjectSpec(
        name="apple", language_name="apple", source="usd", usd_subpath="objaverse/apple_02.usd",
        mesh_subpath="objaverse/apple_clean.obj", dist_color=(0.80, 0.12, 0.10),
        mass=0.050, extents=(0.0702, 0.0754, 0.0733), local_center=(0.0, 0.0, 0.0),
        elongated=False, x_range=(0.34, 0.44), place_xy_tol_cm=7.0),
    "banana": ObjectSpec(
        name="banana", language_name="banana", source="usd", usd_subpath="ycb/banana.usd",
        mesh_subpath="ycb/banana_clean.obj", dist_color=(0.92, 0.80, 0.15),
        mass=0.080, extents=(0.1089, 0.1784, 0.0367), local_center=(0.0, 0.0, 0.0),
        elongated=True, local_long_axis=(0.3372, 0.9414, 0.0), grasp_dz=0.0, x_range=(0.34, 0.44),
        place_xy_tol_cm=9.0),
    # marker pen (dry-erase marker): very elongated, ~2cm thick. rest_offset=0.004 stops the firm claw ON the
    # surface (else it over-drives PAST the thin body and the pen lodges on a finger). release_dz lower so it
    # settles in the bowl instead of rolling off the rim.
    "pen": ObjectSpec(
        name="pen", language_name="pen", source="usd", usd_subpath="ycb/dry_erase_marker.usd",
        mesh_subpath="ycb/dry_erase_marker_clean.obj", dist_color=(0.10, 0.10, 0.12),
        mass=0.020, extents=(0.0210, 0.1208, 0.0189), local_center=(0.0, 0.0, 0.0),
        elongated=True, local_long_axis=(-0.0303, 0.9995, 0.0), grasp_dz=0.0, rest_offset=0.004,
        release_dz=0.03, x_range=(0.34, 0.44), place_xy_tol_cm=9.0),
    "cube": ObjectSpec(
        name="cube", language_name="cube", source="cuboid", mass=0.040,
        extents=(0.05, 0.05, 0.05), local_center=(0.0, 0.0, 0.0), is_cube=True,
        color=(0.85, 0.15, 0.15), x_range=(0.34, 0.44)),
    # tennis ball: regulation ~6.7cm diameter, ~57g, round (no preferred axis) -> procedural sphere; yellow-
    # green felt. Round handling like the apple (grasp at root, place tol 7cm). NEW for the Genesis 5-object gate.
    "tennis_ball": ObjectSpec(
        name="tennis_ball", language_name="tennis ball", source="sphere", mass=0.057,
        extents=(0.067, 0.067, 0.067), local_center=(0.0, 0.0, 0.0), elongated=False,
        color=(0.85, 0.95, 0.20), friction=(1.1, 1.0), x_range=(0.34, 0.44), place_xy_tol_cm=7.0),
    # book: a flat hardcover (~18x13x3 cm, ~0.30 kg). Procedural box with a realistic dark-red cover colour;
    # high friction so it rests flat and is hard to nudge. Used as a clutter/distractor (not a grasp target
    # in pick-place), so no elongated/cube grasp hints are needed. NEW 2026-06-19 for the distractor pool.
    "book": ObjectSpec(
        name="book", language_name="book", source="cuboid", mass=0.300,
        extents=(0.18, 0.13, 0.03), local_center=(0.0, 0.0, 0.0), elongated=False,
        color=(0.45, 0.10, 0.12), friction=(1.2, 1.0), x_range=(0.34, 0.44), place_xy_tol_cm=8.0),
}
