# SPDX-License-Identifier: Apache-2.0
"""SOMA livery for the firefly arm in Genesis — reproduces RoboLab's recolor scheme exactly.

Solid PBR per link (recolor_firefly_usd.py::mat_for): metallic SILVER structure, GREY link_4, near-black
DARK link_6 + base, light camera bodies. The two long links (link_2 upper arm, link_3 forearm) are NOT
recolored here — they carry the baked SOMA panel TEXTURE via their GLB visuals (robots/bake_soma_links.py:
orange+carbon on x/z faces, silver+SOMA-logo on y faces). Call apply_livery(robot) BEFORE scene.build().
"""
from __future__ import annotations

import genesis as gs

# colors (r,g,b), metallic, roughness — from recolor_firefly_usd.py
SILVER = ((0.55, 0.56, 0.59), 0.85, 0.40)
GREY = ((0.42, 0.43, 0.45), 0.70, 0.45)
DARK = ((0.09, 0.09, 0.10), 0.50, 0.50)
CAMG = ((0.66, 0.69, 0.76), 0.20, 0.50)
TEXTURED = ("link_2", "link_3")   # GLB SOMA-panel texture — leave untouched


def _surf(spec):
    color, metallic, rough = spec
    return gs.surfaces.Plastic(color=color, metallic=metallic, roughness=rough)


def mat_for(name: str):
    if "camera" in name:
        return _surf(CAMG)
    if "link_6" in name or "base_link" in name or "dual_base" in name:
        return _surf(DARK)
    if "link_4" in name:
        return _surf(GREY)
    return _surf(SILVER)


def apply_livery(robot) -> int:
    """Set solid SOMA surfaces on every link except the GLB-textured long links. Returns #links recolored."""
    n = 0
    for link in robot.entity.links:
        if any(t in link.name for t in TEXTURED):
            continue
        link.surface = mat_for(link.name)
        n += 1
    return n
