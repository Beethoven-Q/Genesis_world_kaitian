#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the TABLE-TEXTURE domain-randomization pack -> genesis_firefly/assets/textures/tables/.

The DR spec (docs/domain_randomization.md, scope A) wants per-build table textures: wood / steel /
tablecloth albedo maps, >=10, INCLUDING several BRIGHT tablecloths (e.g. red-white floral). NOT pure
colour, NOT too fancy. This script assembles that pack:

  - REAL albedo maps copied + downsampled (<=1024^2) from the RoboLab asset library where good ones exist
    (a wood particle-board, a brushed steel, a white painted metal).
  - PROCEDURAL maps generated with numpy/PIL for the categories the library lacks (extra wood-grain planks,
    a brushed-steel variant, and the tablecloths: red/blue gingham, red-white floral, checkered, polka).

Every output is a 1024x1024 sRGB PNG, seamless-ish, matte-friendly. Re-runnable + deterministic (fixed
seeds). The stage's ``table_texture_pool()`` just globs this directory, so adding/removing a file changes
the DR pool with no code change.

Run:
  ./.venv/bin/python genesis_firefly/scripts/build_table_textures.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "assets", "textures", "tables"))
os.makedirs(OUT, exist_ok=True)
SZ = 1024

# Real albedo maps in the RoboLab library that read as a believable TABLE surface.
REAL = {
    # dest_name : (source_path, category)
    "wood_particleboard.png": (
        "/home/kaitianchao/Projects/RoboLab_firefly/assets/objects/vomp/large_storage_rack/"
        "textures/Wood_Pressed_A/T_Wood_Pressed_A1_Albedo.png", "wood"),
    "steel_brushed_grey.png": (
        "/home/kaitianchao/Projects/RoboLab_firefly/assets/objects/vomp/case_d04/"
        "textures/T_Metal_Rough_A_Albedo.png", "steel"),
    "metal_painted_white.png": (
        "/home/kaitianchao/Projects/RoboLab_firefly/assets/objects/vomp/large_storage_rack/"
        "textures/Metal_Painted_White_Rough_A/T_Metal_Painted_White_Rough_A_Albedo.png", "steel"),
}


def _save(arr, name):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(OUT, name))
    print(f"  wrote {name}  {arr.shape[1]}x{arr.shape[0]}")


def copy_real():
    for name, (src, _cat) in REAL.items():
        if not os.path.exists(src):
            print(f"  SKIP (missing real source) {name}: {src}")
            continue
        im = Image.open(src).convert("RGB").resize((SZ, SZ), Image.LANCZOS)
        im.save(os.path.join(OUT, name))
        print(f"  wrote {name}  (real, downsampled {SZ}x{SZ})")


# ---- procedural wood ---------------------------------------------------------------------------------------
def wood_plank(name, base, dark, n_planks=6, seed=0):
    """Horizontal wood planks with along-grain streaks + plank seams. base/dark = light/dark RGB of the wood."""
    rng = np.random.RandomState(seed)
    base = np.array(base, np.float32); dark = np.array(dark, np.float32)
    y = np.linspace(0, 1, SZ)[:, None]
    x = np.linspace(0, 1, SZ)[None, :]
    # long grain: sum of stretched sinusoids along x (the plank length) modulated per row
    grain = np.zeros((SZ, SZ), np.float32)
    for f, a in [(18, 0.5), (37, 0.3), (71, 0.2), (140, 0.12)]:
        ph = rng.rand(SZ, 1) * 2 * np.pi
        grain += a * np.sin(2 * np.pi * f * x + ph + 3.0 * np.sin(2 * np.pi * 2 * y))
    grain += 0.15 * rng.randn(SZ, SZ)
    grain = (grain - grain.min()) / (np.ptp(grain) + 1e-6)
    t = grain[..., None]
    img = dark * t + base * (1 - t)
    # plank seams (dark horizontal lines) + slight per-plank tone shift
    pe = (y * n_planks).astype(int).repeat(SZ, axis=1)
    for p in range(n_planks):
        img[pe == p] *= (0.9 + 0.2 * rng.rand())
    seam = (np.abs(((y * n_planks) % 1.0) - 0.0) < 0.012) | (np.abs(((y * n_planks) % 1.0) - 1.0) < 0.012)
    img[np.broadcast_to(seam, (SZ, SZ))] = dark * 0.6
    _save(img, name)


# ---- procedural brushed steel ------------------------------------------------------------------------------
def brushed_steel(name, tone=180, seed=0):
    rng = np.random.RandomState(seed)
    # horizontal brushing: 1D noise smeared along x
    line = rng.randn(SZ) * 10
    img = np.tile(line, (SZ, 1))
    img = np.array(Image.fromarray((img - img.min()).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)),
                   np.float32)
    img += 6 * rng.randn(SZ, SZ)
    img = tone + (img - img.mean()) * 0.8
    g = np.linspace(-12, 12, SZ)[None, :].repeat(SZ, 0)   # gentle cross sheen
    out = np.clip(img + g, 0, 255)
    _save(np.dstack([out, out, out * 1.01]), name)


# ---- procedural tablecloths --------------------------------------------------------------------------------
def gingham(name, color, seed=0, n=14):
    """Classic gingham check: white base, colour bands; overlaps go darker (woven look)."""
    rng = np.random.RandomState(seed)
    col = np.array(color, np.float32)
    white = np.array([248, 246, 240], np.float32)
    cell = SZ // n
    cx = (np.arange(SZ) // cell) % 2
    bx = cx[None, :].repeat(SZ, 0)
    by = cx[:, None].repeat(SZ, 1)
    img = np.empty((SZ, SZ, 3), np.float32)
    img[:] = white
    half = 0.5 * (col + white)
    img[(bx == 1) & (by == 0)] = half
    img[(bx == 0) & (by == 1)] = half
    img[(bx == 1) & (by == 1)] = col
    img += 4 * rng.randn(SZ, SZ, 1)        # fabric grain
    _save(img, name)


def floral(name, bg, seed=0, n_flowers=420):
    """Bright red-white country floral on a light cloth: scattered 5-petal blooms + leaves."""
    rng = np.random.RandomState(seed)
    im = Image.new("RGB", (SZ, SZ), tuple(int(c) for c in bg))
    d = ImageDraw.Draw(im)
    reds = [(196, 30, 40), (215, 55, 50), (170, 25, 45)]
    greens = [(60, 120, 55), (80, 140, 60)]
    for _ in range(n_flowers):
        cx, cy = rng.randint(0, SZ), rng.randint(0, SZ)
        r = rng.randint(10, 26)
        col = reds[rng.randint(len(reds))]
        # 5 petals around the centre
        for k in range(5):
            a = 2 * np.pi * k / 5 + rng.rand()
            px, py = cx + r * np.cos(a), cy + r * np.sin(a)
            d.ellipse([px - r * 0.6, py - r * 0.6, px + r * 0.6, py + r * 0.6], fill=col)
        d.ellipse([cx - r * 0.4, cy - r * 0.4, cx + r * 0.4, cy + r * 0.4],
                  fill=(245, 210, 90))   # yellow centre
        if rng.rand() < 0.5:             # a leaf
            g = greens[rng.randint(len(greens))]
            lx, ly = cx + rng.randint(-r, r), cy + r
            d.ellipse([lx - 6, ly - 3, lx + 6, ly + 3], fill=g)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.array(im, np.float32) + 3 * rng.randn(SZ, SZ, 1)
    _save(arr, name)


def checkered(name, color, seed=0, n=8):
    """Bold two-tone diner checker (bright)."""
    col = np.array(color, np.float32); white = np.array([245, 243, 236], np.float32)
    cell = SZ // n
    cx = (np.arange(SZ) // cell) % 2
    chk = (cx[None, :] ^ cx[:, None]).astype(bool)
    img = np.where(chk[..., None], col, white).astype(np.float32)
    rng = np.random.RandomState(seed)
    img += 3 * rng.randn(SZ, SZ, 1)
    _save(img, name)


def polka(name, bg, dot, seed=0, n=12):
    """Polka-dot tablecloth (bright, cheerful)."""
    im = Image.new("RGB", (SZ, SZ), tuple(int(c) for c in bg))
    d = ImageDraw.Draw(im)
    cell = SZ / n
    for i in range(n + 1):
        for j in range(n + 1):
            off = (cell / 2) if (j % 2) else 0
            cx, cy = i * cell + off, j * cell
            r = cell * 0.22
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(c) for c in dot))
    rng = np.random.RandomState(seed)
    arr = np.array(im, np.float32) + 3 * rng.randn(SZ, SZ, 1)
    _save(arr, name)


def main():
    print(f"Building table-texture pack -> {OUT}")
    print("[real]")
    copy_real()
    print("[wood]")
    wood_plank("wood_oak_planks.png", base=(178, 134, 86), dark=(120, 82, 47), seed=1)
    wood_plank("wood_walnut_planks.png", base=(120, 82, 54), dark=(70, 44, 28), seed=2)
    wood_plank("wood_birch_planks.png", base=(210, 184, 140), dark=(165, 135, 95), seed=3)
    print("[steel]")
    brushed_steel("steel_brushed_dark.png", tone=120, seed=4)
    print("[tablecloth]")
    gingham("cloth_gingham_red.png", color=(200, 40, 45), seed=5)
    gingham("cloth_gingham_blue.png", color=(45, 80, 170), seed=6)
    floral("cloth_floral_red_white.png", bg=(243, 240, 232), seed=7)
    checkered("cloth_checker_red.png", color=(198, 36, 42), seed=8)
    polka("cloth_polka_teal.png", bg=(238, 235, 228), dot=(20, 140, 140), seed=9)
    n = len([f for f in os.listdir(OUT) if f.lower().endswith((".png", ".jpg"))])
    print(f"DONE: {n} textures in {OUT}")


if __name__ == "__main__":
    main()
