#!/usr/bin/env python3
"""Bake link_2/link_3 panels as ONE GLB with ONE material (Nyx renders only ONE material per GLB AND only the
FIRST <visual> per link in a URDF subscene — so multi-material GLBs and multi-visual links both fail). The
single material uses a SIDE-BY-SIDE texture atlas so each face group samples its own region:

    atlas = hstack( panel_carbon (left) , solid Y-face colour (right) )
    - +/-X,+/-Z faces -> the LEFT (carbon) region, with the ORIGINAL v-mapping (v = width y) -> carbon is
      UNCHANGED (it's uniform along u, so scaling u into the left fraction doesn't alter the look).
    - +/-Y "Y faces"  -> a fixed point in the RIGHT (solid) strip -> a clean SOLID colour, no carbon bleed,
      no UV-flip fragility (the strip is uniform).

Restores link_2/3 to a single textures/link_X_soma.glb <visual>. SOMA can later be stamped into the strip.

  ./.venv/bin/python genesis_firefly/scripts/bake_soma_panels.py
"""
import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

ASSET = "/home/kaitianchao/Projects/Genesis_world_kaitian/genesis_firefly/assets/robots/firefly_y6_gr100"
MESH_DIR = f"{ASSET}/source/xpkg_urdf_firefly_y6/meshes"
TEX_DIR = f"{ASSET}/textures"
URDF = f"{ASSET}/dual_firefly_y6_gr100.urdf"
CARBON = np.asarray(Image.open(f"{TEX_DIR}/panel_carbon.png").convert("RGB").resize((2048, 512)))

YFACE_RGB = (185, 186, 188)     # <-- the +/-Y "Y face" colour. Neutral SILVER to match the rest of the arm.
_SIL_W = 512                    # width of the silver strip appended to the right of the carbon panel
_CARB_FRAC = 2048 / (2048 + _SIL_W)
_Y_U = _CARB_FRAC + 0.5 * (_SIL_W / (2048 + _SIL_W))     # u that lands in the middle of the silver strip


def bake(link):
    m = trimesh.load(f"{MESH_DIR}/{link}.STL", force="mesh")
    pts = m.vertices
    zmin, zr = pts[:, 2].min(), max(np.ptp(pts[:, 2]), 1e-6)
    ymin, yr = pts[:, 1].min(), max(np.ptp(pts[:, 1]), 1e-6)
    ay = np.abs(m.face_normals)
    is_y = (ay[:, 1] >= ay[:, 0]) & (ay[:, 1] >= ay[:, 2])           # +/-Y = the "Y faces" (narrow sides)

    atlas = np.hstack([CARBON, np.full((512, _SIL_W, 3), YFACE_RGB, np.uint8)])   # carbon | silver
    atlas_img = Image.fromarray(atlas)

    xz = np.where(~is_y)[0]
    subx = m.submesh([xz], append=True, repair=False)
    sx = subx.vertices
    uvx = np.column_stack([(sx[:, 2] - zmin) / zr * _CARB_FRAC, (sx[:, 1] - ymin) / yr])   # left carbon region

    yf = np.where(is_y)[0]
    suby = m.submesh([yf], append=True, repair=False)
    uvy = np.full((len(suby.vertices), 2), [_Y_U, 0.5], np.float32)                         # silver strip

    V = np.vstack([subx.vertices, suby.vertices])
    F = np.vstack([subx.faces, suby.faces + len(subx.vertices)])
    UV = np.vstack([uvx, uvy]).astype(np.float32)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mesh.visual = TextureVisuals(uv=UV, material=PBRMaterial(name=f"{link}_atlas", baseColorTexture=atlas_img),
                                 image=atlas_img)
    out = f"{TEX_DIR}/{link}_soma.glb"
    mesh.export(out)
    print(f"  {link}: carbon {len(xz)} + yface {len(yf)} faces (Y={YFACE_RGB}) -> {out}")


def normalize_urdf():
    """Ensure each link_2/3 has exactly ONE <visual> = textures/link_X_soma.glb (undo any carbon/yface split)."""
    tree = ET.parse(URDF)
    root = tree.getroot()
    changed = 0
    for link in root.findall("link"):
        name = link.get("name")
        stem = "link_2" if "link_2" in name else ("link_3" if "link_3" in name else None)
        if stem is None:
            continue
        panels = [v for v in link.findall("visual")
                  if (v.find("geometry/mesh") is not None
                      and any(t in v.find("geometry/mesh").get("filename")
                              for t in ("_soma.glb", "_carbon.glb", "_yface.glb")))]
        if not panels:
            continue
        origin = panels[0].find("origin")
        for v in panels:
            link.remove(v)
        nv = ET.SubElement(link, "visual")
        if origin is not None:
            nv.append(ET.fromstring(ET.tostring(origin)))
        g = ET.SubElement(nv, "geometry")
        ET.SubElement(g, "mesh", filename=f"textures/{stem}_soma.glb")
        changed += 1
    tree.write(URDF)
    print(f"[URDF] normalized {changed} link(s) to a single _soma.glb visual")


if __name__ == "__main__":
    for link in ("link_2", "link_3"):
        bake(link)
    normalize_urdf()
    print("SOMA_PANELS_DONE")
