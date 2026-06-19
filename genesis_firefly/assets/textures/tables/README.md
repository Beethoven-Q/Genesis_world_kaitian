# Table-texture DR pack

Albedo (diffuse) maps for the **per-build table-texture domain randomization** (DR spec scope A —
see `../../../docs/domain_randomization.md`). The stage's `table_texture_pool()`
(`world/manipulation_stage.py`) globs this directory, so adding/removing a PNG changes the DR pool
with **zero code change**. Each map is a 1024×1024 sRGB PNG and is applied to **both** table tops
this build (the arm table + the object table share one texture — they are flush at the seam, so one
continuous surface reads realistic). Rebuild with:

```bash
./.venv/bin/python genesis_firefly/scripts/build_table_textures.py
```

## Files (12: 4 wood · 3 steel/metal · 5 tablecloth, incl. 3 bright)

| file | category | source | notes |
|---|---|---|---|
| `wood_particleboard.png` | wood | **real** (RoboLab `Wood_Pressed_A1`) | OSB / particle-board, warm tan |
| `wood_oak_planks.png` | wood | procedural | horizontal oak planks + grain + seams |
| `wood_walnut_planks.png` | wood | procedural | dark walnut planks |
| `wood_birch_planks.png` | wood | procedural | pale birch planks |
| `steel_brushed_grey.png` | steel | **real** (RoboLab `T_Metal_Rough_A`) | brushed grey steel |
| `metal_painted_white.png` | metal | **real** (RoboLab `Metal_Painted_White_Rough_A`) | white painted metal |
| `steel_brushed_dark.png` | steel | procedural | dark brushed-steel variant |
| `cloth_gingham_red.png` | tablecloth | procedural | **bright** red-white gingham check |
| `cloth_gingham_blue.png` | tablecloth | procedural | **bright** blue-white gingham check |
| `cloth_floral_red_white.png` | tablecloth | procedural | **bright** red-white country floral |
| `cloth_checker_red.png` | tablecloth | procedural | bold red-white diner checker |
| `cloth_polka_teal.png` | tablecloth | procedural | teal polka-dot cloth |

The **filename prefix** (`wood` / `steel` / `metal` / `cloth`) selects the physical texel period in
`world/manipulation_stage.py::_TEX_PERIOD` (so a small arm table and a big object table read at the
same real-world scale — gingham squares ~2–3 cm, a wood plank ~1 board across). Name new files with a
matching prefix to inherit a sensible tiling; anything else falls back to a 0.45 m default period.

> Why a separate textured **Plane** on top (not a textured Box): a `gs.morphs.Box` has no UV
> coordinates, so Nyx cannot map a texture onto it (it warns `Texture given but asset missing uv
> info` and smears a degenerate fallback). A `gs.morphs.Plane` carries UVs and is UV-handled by the
> Nyx exporter, so the texture maps cleanly. See `../../../docs/rendering_and_livery.md` §8.
