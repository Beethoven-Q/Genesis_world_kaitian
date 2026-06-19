#!/usr/bin/env python3
"""Bake the REAL Firefly Y6 SOMA livery into the dual-arm meshes (Genesis/Nyx port of RoboLab's
scripts/robot_setup/recolor_firefly_usd.py).

WHY a baker (not a runtime recolor): Nyx loads a URDF entity as a *SubScene* and reads each mesh file's
EMBEDDED PBR material directly (per-vgeom `surface` overrides are ignored for URDFs). So to get RoboLab's
per-link OmniPBR look we bake the same materials INTO the meshes as glTF PBR (metallic-roughness) and emit a
livery URDF whose <visual> meshes point at the baked GLBs (collision meshes stay the original STL).

Per-link materials reproduce RoboLab's mat_for() EXACTLY:
  SILVER (0.55,0.56,0.59) m0.85 r0.40  -> link_1/5, yokes, gripper fingers, link_2/3 STL body
  GREY   (0.42,0.43,0.45) m0.70 r0.45  -> link_4
  DARK   (0.09,0.09,0.10) m0.50 r0.50  -> link_6 (wrist motor) + every *base_link
  CAMG   (0.66,0.69,0.76) m0.20 r0.50  -> camera bodies / d405
link_2/link_3 already ship as link_{2,3}_soma.glb (orange-carbon + silver-SOMA panels) -> left untouched.

  ./.venv/bin/python genesis_firefly/scripts/bake_firefly_livery.py
Emits: assets/robots/firefly_y6_gr100/livery/*.glb  +  dual_firefly_y6_gr100_livery.urdf
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

ASSET = "/home/kaitianchao/Projects/Genesis_world_kaitian/genesis_firefly/assets/robots/firefly_y6_gr100"
URDF_IN = f"{ASSET}/dual_firefly_y6_gr100.urdf"
URDF_OUT = f"{ASSET}/dual_firefly_y6_gr100_livery.urdf"
LIVERY_DIR = f"{ASSET}/livery"

SILVER = ((0.70, 0.71, 0.74), 0.55, 0.42)       # brighter brushed silver (yokes/links) — reads silver
GREY = ((0.50, 0.51, 0.53), 0.55, 0.45)
DARK = ((0.09, 0.09, 0.10), 0.50, 0.50)
CAMG = ((0.74, 0.73, 0.71), 0.20, 0.50)     # camera bodies: warm-neutral silver (was lavender 0.66,0.69,0.76)
BLACK = ((0.025, 0.025, 0.027), 0.45, 0.45)     # gripper ROOT (gr100 base) — near-black like the real GR100
GREEN = ((0.06, 0.42, 0.17), 0.20, 0.45)        # gripper FINGERS — dark green with a slight metallic sheen


def mat_for(link_name):
    if "camera_body" in link_name:
        return CAMG, "camg"
    if "gripper" in link_name:                  # the GR100 hand
        if "base" in link_name:
            return BLACK, "black"               # gripper root = black
        return GREEN, "green"                   # gripper fingers (L1/L2) = dark green metallic
    if "link_6" in link_name or "base_link" in link_name:
        return DARK, "dark"
    if "link_4" in link_name:
        return BLACK, "black"          # the cap/collar at the top of the wrist (owner: green arrow) -> black
    return SILVER, "silver"


def _solid_texture(col):
    """An 8x8 solid-colour sRGB image of `col` (0-1 rgb). Nyx's URDF loader renders baseColorTEXTURE but
    IGNORES baseColorFactor, so every link colour must be carried as an image."""
    rgb = (np.clip(np.array(col), 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(np.tile(rgb, (8, 8, 1)))


def bake(stl_rel, mat, matname):
    """Load the STL, bake the colour as a solid baseColorTEXTURE (so Nyx renders it), export GLB."""
    (col, met, rough) = mat
    src = os.path.join(ASSET, stl_rel)
    m = trimesh.load(src, force="mesh")
    stem = os.path.splitext(os.path.basename(stl_rel))[0]
    out_rel = f"livery/{stem}__{matname}.glb"
    out_abs = os.path.join(ASSET, out_rel)
    img = _solid_texture(col)
    pbr = PBRMaterial(name=f"{stem}_{matname}", baseColorTexture=img,
                      metallicFactor=float(met), roughnessFactor=float(rough))
    # uv=0 everywhere -> samples the single solid colour; the image is what Nyx actually honours.
    m.visual = TextureVisuals(uv=np.zeros((len(m.vertices), 2), np.float32), material=pbr, image=img)
    m.export(out_abs)
    return out_rel


def main():
    os.makedirs(LIVERY_DIR, exist_ok=True)
    tree = ET.parse(URDF_IN)
    root = tree.getroot()
    baked = {}
    n = 0
    for link in root.findall("link"):
        lname = link.get("name")
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            fn = mesh.get("filename")
            if fn.lower().endswith((".glb", ".gltf")):
                continue                                   # already textured (link_2/3 SOMA panels) -> keep
            mat, matname = mat_for(lname)
            key = (fn, matname)
            if key not in baked:
                baked[key] = bake(fn, mat, matname)
                print(f"  baked {os.path.basename(fn):24s} -> {matname:6s} -> {baked[key]}")
            mesh.set("filename", baked[key])
            n += 1
    tree.write(URDF_OUT)
    # ET drops the XML declaration; not required by loaders, but keep it tidy
    txt = open(URDF_OUT).read()
    if not txt.startswith("<?xml"):
        open(URDF_OUT, "w").write('<?xml version="1.0"?>\n' + txt)
    print(f"[BAKE] recoloured {n} visuals across {len(baked)} unique meshes")
    print(f"[BAKE] livery URDF -> {URDF_OUT}")


if __name__ == "__main__":
    main()
