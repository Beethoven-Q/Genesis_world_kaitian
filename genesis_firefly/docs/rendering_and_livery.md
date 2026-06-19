# Photoreal rendering (Nyx) + arm livery — setup & hard-won gotchas

This is the manual for the **photoreal render path** of the Genesis "Line B" firefly stack:
the Nyx path-traced renderer, the per-env immersive HDRI domain-randomization, the matte
arm override, and the **livery-baking** pipeline that gives the Firefly Y6 + GR100 dual arm
its real SOMA paint job. It is written to be reproducible: every API call, constant, file
path, and failure mode below is quoted from the actual code.

> **Why Nyx, not Madrona / LuisaRender.** Genesis's batch renderer (Madrona) is a flat unshaded
> rasterizer — composited albedo reads as "moving colour patches" with no specular, shadows, or
> IBL, and it *cannot* do per-env materials. LuisaRender's `ext/LuisaRender` submodule ships
> empty/uncompiled. The photoreal path that actually works is **Nyx** (`gs-nyx-plugin`, a
> prebuilt Apache-2.0 wheel — no CUDA-toolkit compile). Proven on Driver 580 / RTX A6000.

Authoritative code for everything here:
- `genesis_firefly/scenes/manipulation_stage.py` — the reusable render+DR stage (Nyx setup, HDRI DR, matte override).
- `genesis_firefly/scenes/firefly_cameras.py` — the 3 policy cameras, wrist offsets, side-rig.
- `genesis_firefly/robots/firefly_dual.py` — loads the **livery** URDF and applies the matte surface.
- `genesis_firefly/scripts/bake_firefly_livery.py` — bakes per-link colour into GLBs + a livery URDF.
- `genesis_firefly/scripts/bake_soma_panels.py` — bakes the two-colour link_2/link_3 carbon panels.

---

## 1. Nyx setup

### Install / imports
```bash
pip install gs-nyx-plugin     # prebuilt wheel; no CUDA-toolkit compile needed
```
```python
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
from gs_nyx import nyx_py_sdk as nps
```

### Cameras are **sensors**, not `scene.add_camera`
A Nyx camera is added with `scene.add_sensor(NyxCameraOptions(...))`. From
`manipulation_stage.py`:

```python
nc = dict(lights=LIGHTS, env_maps=env_maps, spp=spp, denoise=True)
self.cams = {
    "third":   self.scene.add_sensor(NyxCameraOptions(res=res, pos=(1.15, -0.95, 0.62),
                   lookat=(0.30, 0.0, 0.34), fov=48, **nc)),
    "cam_side": self.scene.add_sensor(NyxCameraOptions(res=res, pos=tuple(SIDE[0]),
                   lookat=(0.40, 0.0, 0.28), fov=SIDE_VFOV, **nc)),
    "cam_lw":  self.scene.add_sensor(NyxCameraOptions(res=res, fov=WRIST_VFOV, entity_idx=eidx,
                   link_idx_local=self._link_local("left_link_6"),  offset_T=_T(*LEFT_WRIST),  **nc)),
    "cam_rw":  self.scene.add_sensor(NyxCameraOptions(res=res, fov=WRIST_VFOV, entity_idx=eidx,
                   link_idx_local=self._link_local("right_link_6"), offset_T=_T(*RIGHT_WRIST), **nc)),
}
```

`NyxCameraOptions` keys used here:
- `res=(W,H)`, `pos`, `lookat`, `fov` (vertical FOV in degrees) — world-fixed cameras.
- `lights` — a list of light dicts (see below); **Nyx still needs at least a soft key light or contact shadows vanish.**
- `env_maps` — a **tuple** of `nps.EnvironmentMapAsset` (the per-env HDRIs; section 2).
- `spp=32` — samples per pixel. With `denoise=True`, **32 is clean** (`SPP` env var, default `"32"`). The memory notes also ran 192; 32 denoised is the production setting in the stage.
- `denoise=True` — path-tracer denoiser on.
- **Wrist cams** are attached by passing `entity_idx`, `link_idx_local`, and `offset_T` (a 4×4 pose in the link frame) instead of `pos`/`lookat`. This is what makes them follow `link_6`.

The light config used (`manipulation_stage.LIGHTS`):
```python
LIGHTS = [{"dir": (-0.4, 0.3, -0.85), "color": (1, 1, 1),
           "intensity": 1.2, "directional": True, "castshadow": True}]
```
A soft neutral key so contact shadows show; the HDRI does the bulk of the lighting.

### Resolving `link_idx_local`
The wrist cams need the **link-local** index, not the global link index. The stage resolves it
defensively (the attribute name has varied across Genesis versions):
```python
def _link_local(self, name):
    lk = self.robot.entity.get_link(name)
    for a in ("idx_local", "_idx_local"):
        if hasattr(lk, a):
            return int(getattr(lk, a))
    return int(lk.idx - self.robot.entity.link_start)
```

### Render via the **sensor API** — this is what makes wrist cams egocentric
```python
def render(self):
    out = {}
    for nm, cam in self.cams.items():
        cam._stale = True
        out[nm] = np_(cam.read().rgb).astype(np.uint8)   # (N, H, W, 3) uint8
    return out
```
`cam.read()` routes through `BatchRendererCameraSensor` / `_NyxSensor._render_current_state`,
which calls `move_to_attach()` on **all attached cams every frame**. That re-attachment is the
whole point: the wrist cam ends up **truly egocentric** — fingers at the bottom of frame,
looking down (`pol_cam_lw`).

> **Gotcha (the original wrist-cam bug).** The low-level `renderer.render()` does **not**
> auto-attach. Using it leaves wrist cams at the `NyxCameraOptions` default pose
> `(3.5, 0, 1.5)` → a far third-person view. **Always render through `cam.read()`**, set
> `cam._stale = True` first to force a fresh render.

`cam.read().rgb` returns a CUDA tensor → convert at the boundary. The stage's helper:
```python
def np_(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
```

### Batching
Nyx renders batched multi-env. With `scene.build(n_envs=N)`, one `cam.read().rgb` returns shape
`(N, H, W, 3)` — **all envs path-traced in one render call.** Set `env_spacing=(0.0, 0.0)` at
build (see `build()`), so envs don't bleed across each other and each renders its own room.

### Camera FOVs / poses (from `firefly_cameras.py`)
- Wrist D405: `WRIST_VFOV = 57.95°`; side D435i: `SIDE_VFOV = 43.2°` (derived from calibrated apertures / focal 24).
- `LEFT_WRIST`, `RIGHT_WRIST` = `(pos, rot_opengl wxyz)` in the `link_6` frame; `_T(pos, quat)` builds the 4×4 offset.
- `SIDE` = world pose of the side cam. Genesis cameras are **OpenGL convention** (−Z forward, +Y up) — the same convention RoboLab baked, so the calibrated quats transfer directly.

---

## 2. The immersive HDRI environment DR (per-env env maps)

The single biggest realism lever, and the mechanism for full-parallel background DR.

### Build a per-env env-map tuple
Each `EnvironmentMapAsset` is a thin struct: `.texture` is a **string path** to an `.hdr`
(not a `gs.textures.ImageTexture`), `.layout` is the projection, `.multiplier` scales brightness.
From `manipulation_stage.__init__`:
```python
pool = hdr_pool()
self.hdrs = [pool[i] for i in self.rng.choice(len(pool), self.n_envs,
                                              replace=len(pool) < self.n_envs)]
emaps = []
for hp in self.hdrs:
    e = nps.EnvironmentMapAsset()
    e.texture = hp                          # STRING path to a .hdr
    e.layout = nps.EEnvMapLayout.LongLat    # equirectangular
    e.multiplier = 1.0
    emaps.append(e)
env_maps = tuple(emaps)                      # len == n_envs
```
Pass `env_maps=env_maps` to **every** camera. Nyx's `update_scene(env_index)` calls
`set_env_map(env_index)` per env, so each of the N batched envs renders with **its own HDRI**
in a single build (verified 3 envs → 3 different rooms). One build of `n_envs=N` therefore
gives N fully-parallel demos, each in a random real room.

### DROP the ground plane
There is **no ground `Plane`** in the stage. The HDRI *is* the immersive floor + walls + light;
the two tables are the only local surfaces. From `build()`:
```python
self.scene.build(n_envs=self.n_envs, env_spacing=(0.0, 0.0))
```
Swapping the HDRI per env swaps the whole environment (lounge / bathroom / cinema / office —
all verified renders). Keep the third-person cam low (`pos≈(1.15,-0.95,0.62)`,
`lookat≈(0.30,0,0.34)`) so the room shows *behind* the arm.

### Source HDRIs
The pool reuses RoboLab's HDRI library:
```python
_BG_DIR = "/home/kaitianchao/Projects/RoboLab_firefly/assets/backgrounds"
_HDRS_2K = sorted(glob.glob(f"{_BG_DIR}/indoors/*.hdr") + glob.glob(f"{_BG_DIR}/outdoors/*.hdr"))
```

---

## 3. The env-map **MEMORY limit** (the segfault) and the 1K fix

> **Hard gotcha.** Nyx **SEGFAULTS** past ~50–60 **2K** env maps (50×2K builds; 80×2K
> core-dumps). The *scene* is fine — a 100-env scene builds in ~18s with one env map; only the
> env-map **memory** crashes.

**Fix: downsample to 1K (¼ the memory).** All 100 per-env env maps then fit in one build
(verified 100@1K builds in ~22s). The stage maintains a lazily-built, cached 1K pool under
`/data3/hdr1k`:

```python
HDR_1K = "/data3/hdr1k"

def hdr_pool(n_target=140):
    import cv2
    os.makedirs(HDR_1K, exist_ok=True)
    if len(glob.glob(f"{HDR_1K}/*.hdr")) < 100:
        for p in _HDRS_2K:
            if len(glob.glob(f"{HDR_1K}/*.hdr")) >= n_target:
                break
            o = os.path.join(HDR_1K, os.path.basename(p))
            if os.path.exists(o):
                continue
            im = cv2.imread(p, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if im is not None:                      # VALID-ONLY: skip corrupt 2K .hdr
                cv2.imwrite(o, cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA))
    return sorted(glob.glob(f"{HDR_1K}/*.hdr"))
```
Key points:
- Read with `cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR` (preserves the float radiance), resize to **1024×512** with `INTER_AREA`.
- **Valid-only:** `if im is not None` — a corrupt 2K `.hdr` that slips through would reintroduce a 2K map and crash Nyx.
- Cached on `/data3/hdr1k`; built once on first call. Don't point Nyx at the raw 2K library.

---

## 4. The matte entity-surface override (why silver reads neutral)

Nyx renders a URDF's baked image textures with its **default reflective material**, so a
light-neutral surface mirrors the (often warm) HDRI and reads tan/yellow. The fix is an
**entity-level surface override** that forces the whole arm matte:

```python
# manipulation_stage.py
ARM_SURF = dict(metallic=0.0, roughness=0.7)
...
self.robot = FireflyDual(self.scene, pos=(0, 0, ha),
                         surface=gs.surfaces.Default(**ARM_SURF))
```

Why this works and is needed:
- The Nyx URDF exporter **does** honour the entity-level `surface` as a material override
  (`surface_for_material` → `_build_material_override`), and it sets albedo **only when
  `surface.color is not None`**. With `color=None` (the default), the **per-mesh baked texture
  colours are preserved** while `metallic`/`roughness` are forced for the whole arm.
- `metallic=0.0, roughness=0.7` kills the env-map mirror → silver reads **neutral** (R−B ≈ +4
  even in a warm room) instead of tan.
- This is also *why the livery colours must be baked as image textures* (section 5/6): Nyx
  ignores `baseColorFactor`/`metallicFactor`/`roughnessFactor` on a URDF mesh, and the
  metallic/roughness it *does* use comes from this entity override, not from the GLB.

> Verify colour with **pixel sampling** (a colour extractor on the rendered frame), not by eye.

---

## 5. The livery-baking pipeline

Nyx loads a URDF entity as a **SubScene** and reads each mesh file's **embedded** glTF PBR
material — per-vgeom `surface` overrides are **ignored for URDFs**. So bare STL links render
white. The fix is to **bake** the per-link paint into GLBs and emit a *livery URDF* whose
`<visual>` meshes point at those GLBs (collision stays the original STL).

`FireflyDual` loads the livery URDF by default and falls back to the plain URDF if it hasn't been baked:
```python
# robots/firefly_dual.py
_PLAIN_URDF  = str(ASSET / "dual_firefly_y6_gr100.urdf")
_LIVERY_URDF = str(ASSET / "dual_firefly_y6_gr100_livery.urdf")
DUAL_URDF = _LIVERY_URDF if Path(_LIVERY_URDF).exists() else _PLAIN_URDF
```

### 5a. `bake_firefly_livery.py` — per-link solid colours

Run:
```bash
./.venv/bin/python genesis_firefly/scripts/bake_firefly_livery.py
```
Emits `assets/robots/firefly_y6_gr100/livery/*.glb` plus
`dual_firefly_y6_gr100_livery.urdf`.

**Per-link colour palette** (RGB 0–1, metallicFactor, roughnessFactor):
```python
SILVER = ((0.70, 0.71, 0.74), 0.55, 0.42)   # brushed silver — most links
GREY   = ((0.50, 0.51, 0.53), 0.55, 0.45)
DARK   = ((0.09, 0.09, 0.10), 0.50, 0.50)
CAMG   = ((0.74, 0.73, 0.71), 0.20, 0.50)   # camera bodies (warm-neutral silver)
BLACK  = ((0.025, 0.025, 0.027), 0.45, 0.45) # gripper ROOT / link_4 cap
GREEN  = ((0.06, 0.42, 0.17), 0.20, 0.45)    # gripper FINGERS — dark green
```

**Link → material mapping** (`mat_for`), exactly as coded:
```python
def mat_for(link_name):
    if "camera_body" in link_name:               return CAMG,  "camg"
    if "gripper" in link_name:                    # the GR100 hand
        if "base" in link_name: return BLACK, "black"   # gripper ROOT = black
        return GREEN, "green"                            # gripper FINGERS (L1/L2) = dark green
    if "link_6" in link_name or "base_link" in link_name: return DARK,  "dark"
    if "link_4" in link_name:                     return BLACK, "black"  # wrist cap -> black
    return SILVER, "silver"                        # default: silver
```
So: silver on most links, dark wrist motor (`link_6`) + every `*base_link`, **black gripper
root**, **dark-green fingers**, **black `link_4` cap**, neutral-silver camera bodies.

**Each colour is baked as a SOLID-COLOUR IMAGE texture** (because Nyx ignores `baseColorFactor`):
```python
def _solid_texture(col):
    rgb = (np.clip(np.array(col), 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(np.tile(rgb, (8, 8, 1)))   # 8x8 solid sRGB image

def bake(stl_rel, mat, matname):
    (col, met, rough) = mat
    m = trimesh.load(os.path.join(ASSET, stl_rel), force="mesh")
    img = _solid_texture(col)
    pbr = PBRMaterial(name=f"{stem}_{matname}", baseColorTexture=img,
                      metallicFactor=float(met), roughnessFactor=float(rough))
    # uv = 0 everywhere -> every vertex samples the single solid colour
    m.visual = TextureVisuals(uv=np.zeros((len(m.vertices), 2), np.float32),
                              material=pbr, image=img)
    m.export(out_abs)
```
The main loop walks the input URDF, **skips meshes that are already `.glb`/`.gltf`** (so the
hand-baked SOMA panels on link_2/3 are left untouched), bakes the rest, rewrites the
`<visual>` `filename` to the GLB, and writes the livery URDF.

### 5b. `bake_soma_panels.py` — the two-colour link_2 / link_3 panels

link_2 and link_3 need **two** colours on one link: a carbon panel on the broad faces and a
silver strip on the narrow "Y faces". Because Nyx renders only ONE material per GLB and only
the FIRST `<visual>` per link (section 6), this is done with **one GLB, one material, one
visual, and a side-by-side texture ATLAS**.

Run (do this **before** `bake_firefly_livery.py`, since that script keeps the resulting
`_soma.glb` visuals):
```bash
./.venv/bin/python genesis_firefly/scripts/bake_soma_panels.py
```
Emits `assets/robots/firefly_y6_gr100/textures/link_2_soma.glb` and `link_3_soma.glb`, and
normalizes the URDF so each link has exactly one `_soma.glb` visual.

**The atlas** — carbon panel on the **left**, a solid silver strip on the **right**:
```python
CARBON   = np.asarray(Image.open(f"{TEX_DIR}/panel_carbon.png").convert("RGB").resize((2048, 512)))
YFACE_RGB = (185, 186, 188)     # the +/-Y "Y face" colour: neutral silver
_SIL_W   = 512                  # width of the silver strip appended to the right
_CARB_FRAC = 2048 / (2048 + _SIL_W)
_Y_U     = _CARB_FRAC + 0.5 * (_SIL_W / (2048 + _SIL_W))   # u landing mid-strip

atlas = np.hstack([CARBON, np.full((512, _SIL_W, 3), YFACE_RGB, np.uint8)])  # carbon | silver
```

**Face classification + UV mapping** (the load-bearing part):
```python
ay = np.abs(m.face_normals)
is_y = (ay[:, 1] >= ay[:, 0]) & (ay[:, 1] >= ay[:, 2])   # +/-Y = narrow "Y faces"

# broad +/-X,+/-Z faces -> LEFT carbon region, ORIGINAL v = width-y mapping
xz  = np.where(~is_y)[0]
subx = m.submesh([xz], append=True, repair=False)
uvx = np.column_stack([(sx[:, 2] - zmin) / zr * _CARB_FRAC,   # u scaled into the carbon fraction
                       (sx[:, 1] - ymin) / yr])

# narrow +/-Y faces -> a FIXED point in the RIGHT silver strip
yf   = np.where(is_y)[0]
suby = m.submesh([yf], append=True, repair=False)
uvy  = np.full((len(suby.vertices), 2), [_Y_U, 0.5], np.float32)
```
The two submeshes are then merged into **one** `Trimesh` with a single
`TextureVisuals(uv=..., material=PBRMaterial(baseColorTexture=atlas))` and exported as one GLB.

Why this exact layout:
- Carbon is uniform along `u`, so scaling `u` into the left fraction leaves the carbon look
  **unchanged**.
- The silver strip is uniform, so the Y faces sample a fixed point — **no UV-flip fragility,
  no carbon bleed, opaque.**
- Owner-confirmed mapping: carbon on ±X,±Z (the broad faces) is correct; the "Y face" is the
  narrow ±Y side. The Y colour is the `YFACE_RGB` knob.

`normalize_urdf()` rewrites link_2/3 to exactly one `<visual>` pointing at `textures/{link}_soma.glb`,
undoing any earlier carbon/yface split (so re-runs are idempotent).

---

## 6. The CRITICAL Nyx gotchas (and *why*)

A consolidated list — each cost real iterations.

1. **Nyx renders only ONE material per GLB in a URDF SubScene.** A multi-material GLB bleeds
   one material onto everything. → Each link's mesh must be one material; two-colour links
   (link_2/3) use a single material + a texture **atlas** (section 5b).

2. **Nyx renders only the FIRST `<visual>` per link.** A second `<visual>` is dropped → holes /
   a "transparent"-looking link. → You cannot give a link two colours with two visuals; use one
   visual + atlas.

3. **Nyx IGNORES `baseColorFactor` / `metallicFactor` / `roughnessFactor` on URDF meshes.**
   (Verified: a vivid-green `baseColorFactor` produced 0 green pixels; matte-vs-glossy factors
   rendered identical.) → **Colours must be image textures** (`_solid_texture`, the carbon
   atlas), and **metallic/roughness come from the entity matte override**
   (`gs.surfaces.Default(metallic=0, roughness=0.7)`), *not* from the GLB factors.

4. **Per-vgeom `surface` overrides are ignored for URDFs** — only one entity-level
   `entity.surface` is honoured (`should_export_at_geom_level` returns False for `gs.morphs.URDF`).
   → This is the whole reason for baking (section 5) and for the single matte override
   (section 4). *Exception:* a standalone `Mesh`/primitive entity (e.g. the side-camera body in
   `add_side_camera_rig`) **does** honour its own `add_entity(surface=...)` in Nyx, because
   it's per-vgeom not a URDF SubScene.

5. **The textured YCB `bowl.usd` CRASHES Nyx** (segfault — "MaterialBindingAPI not applied").
   → Extract the bowl visual mesh from a Genesis-loaded entity
   (`e.vgeoms[0].vmesh` → trimesh → `bowl_clean.obj`) and load it as a clean OBJ:
   `gs.morphs.Mesh(file=bowl_clean.obj, convexify=True, decompose_object_error_threshold=0.04)`.
   (arm / cube / box / sphere render fine; only the textured-USD bowl crashed.)

6. **The env-map memory segfault** (section 3): past ~50–60 **2K** env maps Nyx core-dumps.
   → Downsample to a **1K** valid-only pool under `/data3/hdr1k`.

7. **A VSTACK atlas FAILED** for link_2/3 — the UV v-flip made the Y faces sample the carbon
   black stripe. → Use the **HSTACK** (side-by-side) atlas instead. (The per-submesh and
   two-visual approaches also failed, per gotchas 1 and 2.)

8. **Render through `cam.read()`, not `renderer.render()`** (section 1): only the sensor path
   re-attaches the wrist cams each frame, giving true egocentric views. The low-level call
   leaves them at the options default `pos=(3.5,0,1.5)`.

9. **A `gs.morphs.Box` has NO UVs → Nyx can't map an image texture onto it.** Giving a textured
   Box a `diffuse_texture` logs `Texture given but asset missing uv info (or failed to load)` and
   renders a **garbled/smeared** fallback (the texture streaks down the side faces). → For a textured
   flat surface use a **`gs.morphs.Plane`** (it carries UVs and the exporter UV-handles it, below),
   or bake explicit UVs into a `gs.morphs.Mesh`. This is how the **table-texture DR** works: the
   collidable table stays a Box (physics), a thin **visual-only Plane** on top carries the texture
   (`world/manipulation_stage.py::_top_plane`).

### Nyx `diffuse_texture` mechanism (image textures on standalone primitives)

A standalone primitive/`Mesh` entity (NOT a URDF — see gotcha 4) honours its own
`add_entity(surface=…)` in Nyx. To put an image albedo on it:

```python
surf = gs.surfaces.Plastic(
    diffuse_texture=gs.textures.ImageTexture(image_path="/abs/path/to/albedo.png"),
    roughness=0.65)
```

The Nyx exporter (`gs_nyx_plugin/nyx_scene_exporter.py`) reads `surface.diffuse_texture`: if it's an
`ImageTexture` with an `image_path`, it sets that path as the material's **AlbedoTexture**
(`_apply_texture` → `mat.albedoTexture = texture.image_path`). A `ColorTexture` / `surface.color`
instead sets a flat `albedoColor`.

**Plane UV scaling (the tiling math).** `create_plane` builds mesh UVs spanning `0..plane_size/tile_size`,
then the exporter additionally applies `mat.uvScale = plane_size` **for a Plane with a diffuse texture**.
Net repeats over an edge of length `L` are therefore `L² / tile_size`. To get `L/period` repeats
(i.e. one texture every `period` metres) set **`tile_size = L · period`**, and use the **same** value
for both U and V so texels stay **square** (`world/manipulation_stage.py::_top_plane` does exactly this,
with `period` per texture category in `_TEX_PERIOD`).

---

## 7. How to recolour (the knobs)

Everything is a small, named constant — change the value, re-bake, the livery URDF + GLBs are
regenerated.

| Want to change | Edit | Then run |
| --- | --- | --- |
| Any per-link arm colour (silver/grey/dark/black/green/camera) | `SILVER`/`GREY`/`DARK`/`CAMG`/`BLACK`/`GREEN` in `bake_firefly_livery.py` | `bake_firefly_livery.py` |
| Which link gets which colour | the `mat_for(link_name)` rules in `bake_firefly_livery.py` | `bake_firefly_livery.py` |
| Per-link metallic / roughness *factors* (baked into GLB) | the 2nd/3rd tuple entries of each colour constant | `bake_firefly_livery.py` |
| The carbon panel image (link_2/3 broad faces) | replace `assets/.../textures/panel_carbon.png` | `bake_soma_panels.py` |
| The link_2/3 "Y face" strip colour | `YFACE_RGB` in `bake_soma_panels.py` | `bake_soma_panels.py` |
| Whole-arm sheen (matte ↔ glossy) at render time — no re-bake | `ARM_SURF = dict(metallic=, roughness=)` in `manipulation_stage.py` | nothing |
| Table colours / DR | `_pick_table` / `distinct_object_color` in `manipulation_stage.py` | nothing |
| Table **textures** / DR | the pack in `assets/textures/tables/` (globbed by `table_texture_pool`); tiling in `_TEX_PERIOD` | `scripts/build_table_textures.py` to (re)build the pack |
| Background rooms / brightness | the HDRI pool (`hdr_pool`, `/data3/hdr1k`) and `e.multiplier` | nothing |
| Render quality | `SPP` env var (default `32`, denoised) | nothing |

Re-bake order matters: run **`bake_soma_panels.py` first**, then **`bake_firefly_livery.py`**
(the livery baker keeps the `_soma.glb` visuals and bakes everything else). After any re-bake,
`FireflyDual` automatically picks up `dual_firefly_y6_gr100_livery.urdf` (it loads the livery
URDF when present, plain otherwise). **Verify with pixel sampling, not the naked eye.**

---

## Quick self-check

`manipulation_stage.py` renders a third-person frame to `/tmp`:
```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/scenes/manipulation_stage.py
# -> /tmp/stage_selfcheck.png ; prints STAGE_OK + the per-env room names
```

### What WORKS
- Per-env photoreal Nyx rendering, batched: `cam.read().rgb` → `(N,H,W,3)`, one call.
- True egocentric wrist cams (via the sensor `read()` re-attach + `entity_idx/link_idx_local/offset_T`).
- Per-env immersive HDRI rooms (no ground plane), 100 envs @ 1K in one build.
- Per-build table **textures** (wood / steel / tablecloth albedo maps) on visual-only Plane tops over the collidable table Boxes (`_top_plane`); flush under the per-env object-table height DR.
- Full SOMA livery: silver links, dark wrist/base, black gripper root, dark-green fingers, black link_4 cap, carbon + silver-strip link_2/3 panels.
- Neutral silver in any room via the matte entity override.

### What does NOT work
- Madrona / LuisaRender for photoreal (use Nyx).
- `baseColorFactor`/`metallicFactor`/`roughnessFactor` on URDF meshes (ignored — bake textures + use the matte override).
- An image texture on a `gs.morphs.Box` (no UVs — garbled/smeared; use a `Plane` or a UV'd `Mesh`).
- Multi-material GLBs or multi-`<visual>` links for a two-colour link (use a one-material atlas).
- A VSTACK atlas (UV v-flip — use HSTACK).
- The textured YCB `bowl.usd` (segfault — use a clean extracted OBJ).
- More than ~50–60 **2K** env maps (segfault — downsample to 1K).
- `renderer.render()` for wrist cams (leaves them third-person — use `cam.read()`).
