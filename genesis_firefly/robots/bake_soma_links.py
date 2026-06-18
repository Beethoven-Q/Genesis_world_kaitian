#!/usr/bin/env python3
"""Bake the SOMA panel TEXTURE onto link_2 (upper arm) + link_3 (forearm), reproducing RoboLab's
recolor exactly (recolor_firefly_usd.py::texture_long_link), but as portable textured GLBs that Genesis
loads as the link visuals:
  - +/-Y side faces  -> SILVER + white SOMA logo   (panel_silver.png; UV u=length z, v=width x)
  - +/-X, +/-Z faces -> ORANGE + carbon strip      (panel_carbon.png; UV u=length z, v=width y)
Face split by the dominant normal axis; planar projection from the FULL-mesh bounds (consistent).
Other links get SOLID PBR surfaces at runtime (robots/livery.py).

  ./.venv/bin/python genesis_firefly/robots/bake_soma_links.py    # writes textures/link_{2,3}_soma.glb
"""
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

ASSET = Path(__file__).resolve().parents[1] / "assets/robots/firefly_y6_gr100"
MESHES = ASSET / "source/xpkg_urdf_firefly_y6/meshes"
TEX = ASSET / "textures"
CARBON = Image.open(TEX / "panel_carbon.png").convert("RGB")   # orange + carbon  (x/z faces)
SILVER = Image.open(TEX / "panel_silver.png").convert("RGB")   # silver + SOMA    (y faces)


def bake(link: str):
    m = trimesh.load(MESHES / f"{link}.STL", process=False)
    n = m.face_normals
    p = m.vertices
    zmin, zr = p[:, 2].min(), max(np.ptp(p[:, 2]), 1e-6)
    xmin, xr = p[:, 0].min(), max(np.ptp(p[:, 0]), 1e-6)
    ymin, yr = p[:, 1].min(), max(np.ptp(p[:, 1]), 1e-6)
    y_dom = (np.abs(n[:, 1]) >= np.abs(n[:, 0])) & (np.abs(n[:, 1]) >= np.abs(n[:, 2]))   # y-facing
    y_faces = np.where(y_dom)[0]
    o_faces = np.where(~y_dom)[0]
    scene = trimesh.Scene()
    for faces, img, axis_v, nm in ((y_faces, SILVER, 0, "soma_y"), (o_faces, CARBON, 1, "carbon_xz")):
        if len(faces) == 0:
            continue
        sub = m.submesh([faces], append=True)
        sp = sub.vertices
        u = (sp[:, 2] - zmin) / zr
        v = (sp[:, axis_v] - (xmin if axis_v == 0 else ymin)) / (xr if axis_v == 0 else yr)
        sub.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([u, v]), image=img)
        scene.add_geometry(sub, geom_name=f"{link}_{nm}")
    out = TEX / f"{link}_soma.glb"
    scene.export(out)
    print(f"[bake] {link}: y-faces {len(y_faces)} (SOMA) + others {len(o_faces)} (carbon) -> {out.name} "
          f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    for lk in ("link_2", "link_3"):
        bake(lk)
    print("BAKE_DONE")
