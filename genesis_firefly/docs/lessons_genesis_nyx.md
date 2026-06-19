# Genesis + Nyx — lessons learned (what works, what doesn't)

This is the debugging-lessons logbook for the **Genesis "Line B"** firefly dual-arm
manipulation + photoreal-rendering reproduction (the Genesis sibling of RoboLab's
`simulation_task_debugging_lessons.md`). Every entry below is a **hard-won finding**
written as **SYMPTOM → ROOT CAUSE → FIX**, with the exact file/API it lives in. Do
not re-derive these; do not regress them.

- Engine: **Genesis 1.1.2 + torch 2.7.0+cu128**, editable install (`pip install -e .`),
  branch `genesis_firefly`. Photoreal renderer: **Nyx** (`pip install gs-nyx-plugin`,
  prebuilt wheel — NOT LuisaRender, whose `ext/LuisaRender` submodule is empty).
- Our code lives entirely under `genesis_firefly/`. Key files referenced here:
  - `scenes/manipulation_stage.py` — the reusable robot + cameras + Nyx + env-DR world.
  - `scenes/firefly_scene.py` — `firm_rigid_options()`, tables, `build_bowl()`.
  - `scenes/firefly_cameras.py` — calibrated cam poses, side-camera rig.
  - `robots/firefly_dual.py` — `FireflyDual` loader, gains, dof maps.
  - `collectors/pickplace_collector.py` — the single-build parallel collector.
  - `scripts/bake_firefly_livery.py`, `scripts/bake_soma_panels.py` — the GLB bakers.

> **Standing rule (owner #1):** collision is first-class — what can't happen in reality
> must not happen in sim. **MEASURE** penetration and render a **CLOSE-UP** on the
> container before claiming "no penetration." And **verify colours by PIXEL SAMPLING,
> not by eyeballing.**

---

## 1. Nyx material / URDF gotchas (cost the most iterations)

### 1.1 Nyx renders only ONE material per GLB in a URDF subscene
- **SYMPTOM:** a multi-material GLB (e.g. `link_2` with carbon panels + silver SOMA
  decals as separate materials) renders with ONE material bled onto the entire link.
- **ROOT CAUSE:** Nyx loads a URDF entity as a *SubScene* and collapses each mesh to a
  single material; multi-material glTF is not honoured.
- **FIX:** bake exactly **one GLB, one material, one `<visual>`** per link. For two
  colours on one link, use a **texture ATLAS** sampled by UV (see §1.4), not multiple
  materials. Implemented in `scripts/bake_soma_panels.py`.

### 1.2 Nyx renders only the FIRST `<visual>` per link in a URDF
- **SYMPTOM:** a link with a second `<visual>` (e.g. a separate carbon mesh + a Y-face
  mesh) renders with holes / a "transparent" / missing region — the 2nd visual is dropped.
- **ROOT CAUSE:** the Nyx URDF SubScene loader keeps only the first `<visual>` per link.
- **FIX:** collapse to a single `<visual>`. `bake_soma_panels.normalize_urdf()` rewrites
  each `link_2/3` to exactly one `<visual>` → `textures/link_X_soma.glb`.

### 1.3 Nyx IGNORES baseColorFactor / metallicFactor / roughnessFactor for URDF meshes
- **SYMPTOM:** a vivid-green `baseColorFactor` on a URDF mesh → **0 green pixels**; a
  matte vs glossy `roughnessFactor` → identical render. Per-vgeom `RigidVisGeom.surface`
  overrides are ALSO ignored for URDFs (`should_export_at_geom_level` returns `False` for
  `gs.morphs.URDF`). So bare-STL links render **white**.
- **ROOT CAUSE:** Nyx's URDF SubScene loader only reads a mesh's glTF
  **`baseColorTEXTURE` (image)** — it does not read the scalar PBR *factors*, and it does
  not read per-vgeom surface overrides for URDF entities.
- **FIX (two parts):**
  1. **Bake every link colour as a tiny SOLID-COLOUR IMAGE texture** (uv=0 everywhere),
     never as a factor — `bake_firefly_livery._solid_texture()` (an 8×8 solid sRGB image),
     wired in via `TextureVisuals(uv=zeros, material=PBRMaterial(baseColorTexture=img), image=img)`.
  2. Emit a **livery URDF** whose `<visual>` meshes point at the baked GLBs; collision
     meshes stay the original STL. `bake_firefly_livery.py` →
     `assets/robots/firefly_y6_gr100/livery/*.glb` + `dual_firefly_y6_gr100_livery.urdf`.
     `FireflyDual` loads the livery URDF by default (falls back to the plain URDF).

### 1.4 Side-by-side (hstack) atlas WORKS; vstack atlas FAILS
- **SYMPTOM:** a **vstack** carbon/Y-face atlas rendered the narrow ±Y "Y faces" sampling
  the *carbon black stripe* (wrong colour); per-submesh GLBs and two-`<visual>` approaches
  also failed (they trip §1.1 and §1.2).
- **ROOT CAUSE:** the UV **v-flip** on the Y faces lands them on the carbon region of a
  vertically-stacked atlas.
- **FIX:** an **hstack side-by-side** atlas (`scripts/bake_soma_panels.py`):
  `atlas = hstack(panel_carbon[2048×512] | solid Y-face strip[512×512])`.
  - Broad ±X,±Z faces (`~is_y`) → the LEFT carbon region with the original `v = (y−ymin)/yr`
    mapping (`u` scaled into the carbon fraction; carbon is uniform along `u` so the look is
    unchanged).
  - Narrow ±Y "Y faces" (`is_y`, detected by face-normal `ay[:,1] >= ay[:,0] & ay[:,2]`)
    → a single fixed point `(_Y_U, 0.5)` in the RIGHT silver strip — uniform, so no UV-flip
    fragility, no carbon bleed, opaque. Y colour knob = `YFACE_RGB`.

### 1.5 The matte entity-surface override IS honoured for URDFs (and neutralises the warm-room metal tint)
- **SYMPTOM:** even with correct baked colours, light-neutral surfaces (silver links)
  **mirror the warm HDRI** and read tan/yellow; dark colours (black/green/orange) hid it.
- **ROOT CAUSE:** Nyx renders the baked textures with its **default reflective** material,
  so a near-neutral albedo mirrors the (often warm) environment map.
- **FIX:** the Nyx exporter DOES honour an **entity-level `surface`** as a material override
  for a URDF (`surface_for_material` → `_build_material_override`), and it sets albedo ONLY
  when `surface.color is None`. So pass a colourless matte default:
  ```python
  FireflyDual(scene, surface=gs.surfaces.Default(metallic=0.0, roughness=0.7))
  ```
  (`ARM_SURF = dict(metallic=0.0, roughness=0.7)` in `manipulation_stage.py`). This forces
  the whole arm MATTE — kills the env mirror so silver reads neutral (R−B ≈ +4 even in a
  warm room) — while the per-mesh baked **texture colours are preserved** (because
  `surface.color is None`). **Verify with pixel sampling.**

### 1.6 Standalone Mesh / Primitive entities DO honour `add_entity(surface=...)` in Nyx
- **SYMPTOM:** the side-camera body/stick rendered white when relying on a URDF-style
  override assumption.
- **ROOT CAUSE:** the per-vgeom surface limitation is **URDF-only**; standalone
  `gs.morphs.Mesh` / primitives are exported at geom level.
- **FIX:** `scenes/firefly_cameras.add_side_camera_rig()` passes
  `gs.surfaces.Plastic(color=...)` directly to each `add_entity` — the D435i body renders
  dark, the stick renders a thin dark pole. The collector's stage **must call this**
  (it was once missing).

---

## 2. Nyx environment maps / backgrounds

### 2.1 Per-env env maps WORK → per-env immersive backgrounds in ONE build
- **SYMPTOM/QUESTION:** can N parallel envs each have their own room in a single build?
- **FINDING:** YES. Nyx's `update_scene(env_index)` calls `set_env_map(env_index)` per env,
  so passing `env_maps=tuple(N maps)` to a camera renders each of the N batched envs with
  **its own HDRI** in one build (verified: 3 envs → 3 different rooms).
- **FIX/RECIPE** (`manipulation_stage.py.__init__`):
  ```python
  from gs_nyx import nyx_py_sdk as nps
  e = nps.EnvironmentMapAsset(); e.texture = hdr_path_string   # NOTE: a STRING path, not a gs texture
  e.layout = nps.EEnvMapLayout.LongLat; e.multiplier = 1.0
  env_maps = tuple(emaps)   # one per env
  scene.add_sensor(NyxCameraOptions(..., env_maps=env_maps, lights=LIGHTS, spp=32, denoise=True))
  ```
  `env.texture` is a **string path**, NOT a `gs.textures.ImageTexture`.

### 2.2 DROP the ground Plane so the HDRI is the immersive floor + walls
- **SYMPTOM:** a flat ground plane occludes the room; the arm looks like it floats on a
  grey disc.
- **FIX:** add **no `gs.morphs.Plane()`** in the stage; the only local surfaces are the two
  tables. `scene.build(n_envs=N, env_spacing=(0.0, 0.0))` — spacing 0 stops cross-env
  bleed. Use a **low** third-person cam (`pos=(1.15,-0.95,0.62), lookat=(0.30,0,0.34)`) so
  the room shows behind the arm. Swapping the HDRI per trial swaps the whole environment.

### 2.3 Env-map MEMORY limit — Nyx SEGFAULTS past ~50–60 2K maps
- **SYMPTOM:** a build with 50×2K HDRIs builds; **80×2K core-dumps**. The 100-env *scene*
  itself is fine (~18s with 1 env map) — only the env-map memory crashes.
- **ROOT CAUSE:** Nyx holds all env maps resident; 2K radiance maps blow the budget at ~50–60.
- **FIX:** **downsample HDRIs to 1K** (¼ the memory) → all 100 per-env maps fit in ONE
  build (verified 100@1K builds in ~22s). `manipulation_stage.hdr_pool()` / `_to_1k`-style
  caching: `cv2.imread(p, IMREAD_ANYDEPTH|IMREAD_COLOR)` → `cv2.resize(1024,512, INTER_AREA)`
  → cached under `/data3/hdr1k`. **Skip corrupt 2K `.hdr`** (a single failed downsample
  silently reintroduces a 2K map and crashes the build). Source HDRIs are reused read-only
  from RoboLab: `RoboLab_firefly/assets/backgrounds/{indoors,outdoors}/*.hdr`.

---

## 3. The bowl (concave container) — segfault + penetration

### 3.1 Nyx SEGFAULTS on the YCB `bowl.usd`
- **SYMPTOM:** Nyx segfaults loading the textured YCB bowl USD ("MaterialBindingAPI not
  applied"). The arm/cube/box/sphere render fine — only the textured-USD bowl crashed.
- **ROOT CAUSE:** Nyx chokes on the USD's material binding.
- **FIX:** extract a clean **visual OBJ** from a Genesis-loaded entity
  (`e.vgeoms[0].vmesh` → trimesh → `assets/objects/ycb/bowl_clean.obj`) and load it as a
  plain mesh:
  ```python
  gs.morphs.Mesh(file=bowl_clean.obj, convexify=True,
                 decompose_object_error_threshold=0.04, decimate=False)
  ```
  Used in `pickplace_collector.py` (`BOWL_OBJ`). The USD path (`firefly_scene.build_bowl`)
  is still the collision reference but is **not** fed to Nyx directly. Test: `scripts/temp/nyx_test.py`.

### 3.2 Realistic bowl collision = convex DECOMPOSITION (not SDF, not scaling)
- **SYMPTOM:** a cube dropped onto the thin bowl rim **tunnels through the wall**; the
  wide third-person view hid it.
- **ROOT CAUSE:** the nonconvex watertight **SDF** envelope (`convexify=False`) gives a
  **degenerate ~0 contact** where a box CORNER straddles the 2–3mm rim — `narrowphase.py:300-309`
  emits only a synthetic 1e-4m contact, so the corner passes through. (The face-on wall-slam
  test passed only because a face has many corners; the rim-corner case is the real failure.)
- **FIX:** reproduce RoboLab's PhysX recipe = **convex DECOMPOSITION** collider:
  `gs.morphs.USD/Mesh(convexify=True, decompose_object_error_threshold=0.04, decimate=False)`.
  Solid convex hulls tile the wall; convex-vs-convex contact is robust at the thin rim, so a
  cube on the rim **rolls in/out and never passes through**.
  - **`coacd` raises the rest floor ~8mm** (cosmetic; accepted, as RoboLab does).
  - **Scaling the bowl is FORBIDDEN** (changes the asset / breaks parity).
  - Rim-drop test (tilted cube, offsets 4/6/7cm): **SDF → stuck in wall; coacd → rolled in,
    never in wall.**

### 3.3 Place by RELEASE-ABOVE + free-drop, never drive a held object into the wall
- **SYMPTOM:** even with a perfect collider, the placed cube ends up **inside the bowl wall**.
- **ROOT CAUSE:** a position-controlled arm (kp=200) commanded to a target *below* the rim
  **drives the gripped cube THROUGH the wall** — the actuator out-forces the contact. No solid
  bowl can stop an actuator commanding a through-the-wall pose.
- **FIX (three, all "what reality does"):**
  1. **Release ABOVE the rim and free-drop.** `drop_z = tabZ + 2*BOWL_HALF_H + cube_h/2 + 0.012`;
     `"lower"`, `"hold"`, `"rel"` waypoints all sit at `drop_z`, gripper stays above the rim
     (`pickplace_collector.WP`).
  2. **Release GENTLY** — HOLD the cube stationary over the bowl so velocity → 0 before
     opening (the `"hold"` waypoint). Lateral carry velocity otherwise slips a corner through
     a hull seam (this fixed the last ~1/100).
  3. **Spawn the cube CLEAR of the bowl** — rejection-sample `hypot(cube−bowl) ≥ 0.125`
     (`sample_phys_dr` loops up to 40 times). A cube can never start embedded in reality.
  - Keep `substeps=4` (substeps=8 destabilised and launched a cube). Result at N=100:
    **100/100 grasped, ~98/100 placed, 0 through-wall, 0 spawn-embedded** (the rest rolled
    out cleanly = realistic).

### 3.4 The penetration DETECTOR must use a per-env wall metric + a close-up tile
- **SYMPTOM:** the old "fell-through / far-outside" check reported **0/0 while cubes
  tunnelled** — it was blind to wall penetration.
- **FIX:** a **per-env** wall metric from the **per-env** bowl centre: flag a cube held UP in
  the wall annulus (not on the table, not in the cavity). In `pickplace_collector`:
  ```python
  rxy = hypot(objf-bowx, objf-bowy); bottom = objf_z - cube_h/2; rim_z = tabZ + 2*BOWL_HALF_H
  wall_pen = ((rxy>0.065)&(rxy<0.11)&(bottom>tabZ+0.012)&(bottom<rim_z)) | (bottom < tabZ-0.015)
  ```
  AND render a tight **bowl close-up tile** — the wide third-person view hides penetration.

---

## 4. Grasp physics (pure rigid friction grip — do NOT regress)

All in `robots/firefly_dual.py` + `scenes/firefly_scene.firm_rigid_options()`.

| # | SYMPTOM | ROOT CAUSE | FIX |
|---|---------|-----------|-----|
| 1 | Firm grip sinks a finger **32mm** into the cube | `substeps=1` resolves contact too softly; `constraint_timeconst` is floored at `2·solver_dt`, so substeps is the ONLY stiffening knob | `SimOptions(substeps=4)` → ~0mm. `>4` gives diminishing returns; `8` destabilises. |
| 2 | Loose / asymmetric grip → cube slips | a hard clamp at `GR100_MEET=0.58` capped the grip | **No `clamp_gripper` during a grasp** — let the jaws PD-close to `GR100_CLOSE=0.9`; the OBJECT stops them (cube at q≈0.41). Clamp only for an *empty* close (anti-scissor). |
| 3 | PD trajectory sags ~2–3cm short → jaws close OFF-CENTRE | the arm can't hold its own weight | `gs.materials.Rigid(gravity_compensation=1.0)` on the robot (computed-torque). The grasped object is a separate entity (no comp) → still falls → must be physically held. |
| 4 | Visual finger pokes through the object | default `decompose_robot_error_threshold=inf` → ONE convex hull fills the curved finger's concavity | `URDF(convexify=True, decompose_robot_error_threshold=0.05)` — coacd decompose so the collider HUGS the visual mesh. Collision = real L1/L2 mesh (no inflated-hull hack). |
| 5 | Fingertips catch the cube at ONE point above its COM → it **pendulums up to 60°** | Genesis **HARDCODES contact `margin=0`** (no PhysX `contact_offset`; verified `box_contact.py:220`, `narrowphase.py:775`) | Bake a **~3mm SHAPE-PRESERVING skin** on each finger collider (offset verts along normals — NOT a convex-hull inflate, which flings the cube) and point the URDF finger COLLISION at the padded mesh (visual stays the real finger). **3mm is the sweet spot; 5–7mm flings the cube.** This is exactly PhysX `rest_offset`, so it MATCHES RoboLab. Friction was a red herring (already 1.0/0.9 both sides; Genesis MAX-combines surface friction). |
| 6 | `grasped`/`lift_cm` falsely read ~0 even when the cube rode up | sampled `obj_at_lift` at the FIRST lift step | capture at the **lift PEAK** (last lift step) — `pickplace_collector` records `lift_pos` while `labs[t]=="lift"`. |

### 4.1 Wrist jitter (J5/J6 rolling) — NOT the gains
- **SYMPTOM:** the wrist (J5/J6) oscillates / rings during motion.
- **ROOT CAUSE:** two **sim-faithfulness gaps**, not the PID:
  - (a) Genesis defaults `default_armature=0.1`; RoboLab/MuJoCo uses **0.01** → 10× more
    reflected wrist inertia → lower damping ratio → ringing.
  - (b) Genesis defaults `integrator=approximate_implicitfast`, which under-damps the
    implicit PD.
- **FIX:** reproduce RoboLab's PID **exactly** — `ARM_KP=[200,200,200,75,15,15]`,
  `ARM_KV=[12.5,12.5,12.5,6,0.31,0.31]` — and close the sim gap with
  `URDF(default_armature=0.01)` + `RigidOptions(integrator=gs.integrator.implicitfast)`.
  Result: wrist tracks to ~0.2° with zero oscillation at the stock `kv=0.31`.
  **Owner rule: reproduce the RoboLab arm INCLUDING its PID; fix divergence via
  armature + integrator, never by retuning kp/kv.**

### 4.2 Motion smoothness — roll the wrist DURING the lift
- **SYMPTOM:** an air-pause + stop-go at the lift.
- **ROOT CAUSE:** reorienting the wrist IN PLACE at the lift.
- **FIX:** retarget the LIFT waypoint to the carry orientation so the wrist rolls during the
  (well-conditioned) vertical raise. (More grip force *hurt* stability: effort 10→20 made
  the swing worse. The stability metric is GPU-noisy — run ≥3 trials to compare.)

---

## 5. Genesis sim-to-sim gotchas (Genesis vs Isaac)

- **Backend is `gs.init(backend=gs.gpu)`** — NOT `gs.cuda` (it *logs* "backend gs.cuda" at
  runtime, which is confusing but correct).
- **`get_pos` / `get_quat` / `render` / `read()` return torch CUDA tensors** → convert at
  the sim↔skills boundary: `x.cpu().numpy()`. The vendored numpy skills + HDF5 collector all
  need this. Use the `np_()` helper (`manipulation_stage.np_`, `firefly_dual` inline).
- **Quaternions are `wxyz`** (not xyzw).
- **Cameras are FOV-only** — principal point is forced to the image centre; convention is
  **OpenGL (−Z forward, +Y up)**, the SAME convention RoboLab baked, so calibrated
  `rot_opengl` wxyz quats transfer directly (`firefly_cameras.py`). Wrist VFOV 57.95° (D405),
  side VFOV 43.2° (D435i).
- **Dual-arm dofs are INTERLEAVED**, not contiguous (e.g. `left_joint_1=dof0`,
  `right_joint_1=dof1`, …). Always resolve by joint NAME →
  `entity.get_joint(name).dofs_idx_local[0]` (`FireflyDual.finalize()`), never assume a
  contiguous slice.
- **Per-env physics DR via batched setters:** `set_pos((N,3))`, `set_quat((N,4))`,
  `set_mass_shift((N,n_links))`, `set_friction_ratio((N,n_links))`; per-env `set_mass`
  needs `RigidOptions(batch_links_info=True)`. Batched `inverse_kinematics(pos=(N,3),
  quat=(N,4), dofs_idx_local=idx)` solves all N targets in one call; then
  `control_dofs_position((N,ndof))` + `scene.step()`.
- **Density, not mass:** `gs.materials.Rigid` takes `rho` — convert from target mass via the
  object volume (`firefly_scene._rho_for`).
- **The "Date/random-unavailable" note is RoboLab-only** — it does NOT apply to Genesis.
  Genesis collectors use ordinary `numpy.random.RandomState(seed)`.

---

## 6. ONE scene per process (Vulkan single-shot) → parallelism is per-env DR, not subprocesses

- **SYMPTOM:** building a **second** `gs.Scene` in the same process **segfaults**.
- **ROOT CAUSE:** the Vulkan/Nyx backend is single-shot per process — one build only.
- **FIX / consequence:** the parallelism is **per-env domain randomization inside ONE build**
  (`scene.build(n_envs=N)`), **NOT** a fleet of subprocess batches.
  - `scene.build(n_envs=N)` vectorizes physics AND Nyx rendering: one `cam.read().rgb`
    returns `(N,H,W,3)` — all envs photoreal in one render call.
  - The whole 100-trial collector is now a **single fully-parallel build**
    (`collectors/pickplace_collector.py` + `scenes/manipulation_stage.py`): one scene,
    N envs, no ground plane, per-env 1K HDRI rooms, matte-livery arm, PBR cube/bowl/table,
    4 Nyx cams (third + side + 2 egocentric wrists), per-env physics DR. The 17 one-off temp
    scripts were archived to `scripts/temp/archive/`.
  - Result N=100, seed 7: **100/100 grasped, ~98/100 placed, 0/100 through-wall, ~139s.**
  - Visual DR that bakes at build (per-env material colour, HDRI) varies across builds; each
    build renders all its envs in parallel, so speed is a non-issue.

### 6.1 The wrist cam is egocentric ONLY via the SENSOR `read()` path
- **SYMPTOM:** the wrist cameras render a far third-person view from `pos=(3.5,0,1.5)`.
- **ROOT CAUSE:** the low-level `renderer.render()` does NOT auto-attach the wrist cams, so
  they sit at the `NyxCameraOptions` default pose.
- **FIX:** render through the **sensor API**:
  ```python
  for cam in cams.values():
      cam._stale = True
      img = cam.read().rgb        # (N,H,W,3)
  ```
  This routes through `_NyxSensor._render_current_state`, which calls `move_to_attach()` on
  ALL attached cams each frame → the wrist cams are truly egocentric (fingers-at-bottom,
  looking down). Defined with `entity_idx=eidx, link_idx_local=<link_6 local idx>,
  offset_T=_T(*LEFT_WRIST)` (`manipulation_stage.render()`).

---

## 7. Verification discipline (owner-enforced)

- **Verify colours by PIXEL SAMPLING, not eyeballing** — use a colour extractor on the
  rendered frame (e.g. R−B on the silver links to confirm the matte override killed the warm
  mirror). The owner has caught over-eager "it looks fine" claims.
- **Before claiming "no penetration": MEASURE** (rest-z + wall-slam + through-wall/escaped
  counts via the per-env wall metric) AND render a **CLOSE-UP** on the container. The wide
  third-person view hides wall penetration.
- For grasp stability, the metric is **GPU-noisy** — run ≥3 trials before concluding.

---

## 8. Things that wasted time — don't repeat

- **Photoreal ≠ Madrona.** The Madrona `BatchRenderer` is a flat unshaded rasterizer; with a
  neutral×albedo composite it produces cartoonish "moving colour patches" — no specular,
  shadows, or IBL — and **cannot do per-env materials**. Genesis's photoreal path is **Nyx**.
  (Madrona is still useful as a fast batched preview, but it is not the deliverable renderer.)
- **LuisaRender is a dead end here** — its `ext/LuisaRender` submodule ships empty/uncompiled.
  Use the prebuilt `gs-nyx-plugin` wheel (no CUDA-toolkit compile; Driver 580 / RTX A6000 OK).
- **Don't fight Nyx's URDF material limits with multi-material GLBs or extra `<visual>`s** —
  both silently fail (§1.1, §1.2). Go straight to one-GLB/one-material/one-visual + a texture
  atlas + the entity matte override.
- **Don't set link colours via PBR factors** — Nyx ignores them for URDFs (§1.3). Bake solid
  IMAGE textures.
- **Don't chase grasp slip with friction or grip force** — friction was already correct, and
  more force *worsened* swing. The fix was the **3mm contact-margin skin** (§4, row 5).
- **Don't blame the PID for wrist jitter** — it's armature + integrator (§4.1).
- **Don't use a watertight SDF collider for the bowl, and don't scale it** — use convex
  decomposition (§3.2).
- **Don't feed the textured YCB `bowl.usd` to Nyx** — extract a clean OBJ first (§3.1).
- **Don't trust the old fell-through detector** — it was blind to wall penetration; use the
  per-env annulus metric + close-up tile (§3.4).
- **Don't try to scale via subprocess batches** — a second scene segfaults; scale via
  `n_envs` in one build (§6).
- **Don't pile up >50 2K env maps** — segfault; downsample to 1K (§2.3), and skip corrupt
  `.hdr` files so a 2K one doesn't sneak back in.
- **Don't render wrist cams with `renderer.render()`** — they won't attach; use `cam.read()`
  via the sensor API (§6.1).
- **Don't forget the side-camera body + stick** — it was once missing from the stage build;
  `add_side_camera_rig()` is mandatory (§1.6).
