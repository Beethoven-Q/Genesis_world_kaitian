#!/usr/bin/env python3
"""Bake SOLID, slightly-inflated convex collision meshes for the GR100 fingers (L1/L2), so the two fingers
OVERLAP by a few mm when their gripping pads meet — generating a firm contact that stops them at the touch
point instead of sliding past (Genesis has no MuJoCo-style contact margin; contacts fire only on overlap).
This is the Genesis equivalent of RoboLab's contact_offset band. Realistic: the finger IS a solid plate;
the inflation (~2.5mm) just compensates for the zero-margin contact so the SOLID parts can't interpenetrate
at ANY grip force (owner's principle). Writes L1_coll.STL / L2_coll.STL next to the source meshes.

  ./.venv/bin/python genesis_firefly/robots/bake_finger_colliders.py
"""
from pathlib import Path
import numpy as np
import trimesh

MESHES = Path(__file__).resolve().parents[1] / "assets/robots/firefly_y6_gr100/source/xpkg_urdf_gr100/meshes"
INFLATE = 0.0025   # 2.5mm outward -> two fingers overlap ~5mm at the touch point


def bake(name):
    m = trimesh.load(MESHES / f"{name}.STL", process=False)
    hull = m.convex_hull                       # SOLID convex (no thin-concave GJK slip)
    v = hull.vertices.copy()
    c = v.mean(axis=0)
    d = v - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    v = v + INFLATE * d / np.clip(n, 1e-6, None)   # inflate outward from centroid by INFLATE
    out = trimesh.Trimesh(vertices=v, faces=hull.faces, process=True)
    p = MESHES / f"{name}_coll.STL"
    out.export(p)
    print(f"[finger-coll] {name}: hull {len(hull.vertices)}v inflated {INFLATE*1000:.1f}mm -> {p.name} "
          f"(extents {np.round(out.extents,3)})")


if __name__ == "__main__":
    for nm in ("L1", "L2"):
        bake(nm)
    print("FINGER_COLL_DONE")
