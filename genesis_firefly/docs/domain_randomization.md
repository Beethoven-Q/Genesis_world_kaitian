# Full Domain Randomization (DR) — the spec for EVERY task

This is the authoritative, complete DR specification. It applies to **every** manipulation task we collect.
Scopes **A (manipulation-stage)** and **C (visual background)** are **automatic for all tasks** (the harness
applies them with zero task involvement). Scope **B (object/task)** is per-object/per-task. The owner's rule:
**randomize as much as is realistic** — variety for sim-to-real robustness, but never unphysical
(no blue watermelon, no apple the size of a watermelon).

> Goal: god-mode scripted demos under full DR → a rich, realistic data distribution → train/finetune VLAs for
> sim-to-real. Failures from hard edge cases are *valuable* (recovery data), not bugs to avoid.

---

## ⭐ THE RECOGNIZABILITY RULE (the #1 realism principle — load-bearing, owner 2026-06-20)

Randomize an object's appearance (size, colour, texture, shape) **ONLY within the envelope where a human still
INSTANTLY RECOGNIZES it as that specific object.** The moment a randomization makes the object unrecognizable —
or collapses it into a **trivial geometric primitive** (a "yellow sphere", a "pink brick", a "red blob") — it is
**WRONG and forbidden.** An unrecognizable object is *worse* than no randomization: it poisons the data and the
language grounding (the policy is told "pick the tennis ball / the book" but is shown a plain yellow ball / a pink
brick). We must NOT let our objects degenerate into simple, trivial geometry shapes.

- **Objects keep their IDENTITY.** A tennis ball must look like a tennis ball (felt + the curved seam, regulation
  yellow-green); a book like a book (a readable cover, pages); an apple like an apple (red/green skin texture +
  stem); a banana like a banana (curved, yellow/green). **Never let an object degenerate into a bare primitive.**
- **Identity-from-TEXTURE objects MUST use their NATIVE TEXTURE, not a flat colour.** Apple, tennis ball, book —
  their identity *is* their surface. A flat colour destroys it (2026-06-20 screenshot: tennis → flat yellow ball,
  book → flat pink brick, apple → flat pink blob = all WRONG). If a native texture isn't available yet, **SOURCE
  or MAKE one** (a real asset / UV-mapped texture) — do NOT settle for a flat-colour primitive.
- **The randomization ENVELOPE is per-object + identity-preserving:** apple a bit bigger/smaller, usually red
  (native) / sometimes green — never blue / flat-pink / watermelon-sized. Banana yellow (mostly) / green (unripe).
  Tennis ball regulation yellow-green felt + seam, size ≈ regulation. Book real covers + sizes, always readable.
- **GENERIC objects (a toy cube / block) are the ONLY exception** — they have no identity beyond "a coloured
  block", so free-colour DR keeps them recognizable. This exception is NARROW; it NEVER extends to an
  identity-bearing object (you cannot "fix-colour a sphere and call it a tennis ball").
- **Applies to BOTH the grasp TARGET and the DISTRACTORS** — a distractor "book" that's a pink brick is just as
  wrong as a target one. Every object on the table must be recognizable.
- **The DR-strategist OWNS + ENFORCES this** (see its contract). A flat-colour primitive of an identity-bearing
  object is a **BUG to fix** (source the texture/asset), not an acceptable "FIXED-colour" state. When in doubt,
  NARROW the envelope rather than risk an unrecognizable object.
- **UNIVERSAL + PROACTIVE — not case-by-case.** EVERY object gets a recognizable texture up front (apple, banana,
  tennis ball, book, pen, and every future object). The owner must NOT have to flag objects one by one: the
  moment a new object enters the registry, the DR-strategist gives it a real native texture BEFORE collection.
  Only a generic toy block (the cube) is exempt.

---

## The hard renderer constraint (why "per-build" vs "per-env")

Verified in the Nyx SDK: **COLOR, TEXTURE, and geometry SIZE bake at build time** and are shared by all envs
in one build; the per-env render loop only updates **transforms** and the **env-map** (`set_env_map(env_index)`).
A second `gs.Scene` in one process segfaults (Vulkan single-shot). Therefore:

- **per-env** (free, varies every trial inside one parallel build): pose, orientation, mass, friction,
  damping/stiffness, container fill, object-table **height**, HDRI **background**, **light** color/intensity.
- **per-build** (constant within a build; varied across **build-batches**): table **texture**, object **color**,
  object **size**, object **type**, object-table **size**, side-camera **height/pitch**.

**Collection model = BUILD-BATCHES** (owner's decision): run **B** parallel subprocess builds × **E** envs.
Each build draws one {table-texture, object-colors, object-sizes, object-type, table-size, side-cam-pose} set;
within each build all E envs draw the full per-env space. Total = **B × E** demos, ≈B distinct "looks", native
photoreal quality, and SIZE/object-type variety (only achievable this way). HDRI: original **2K** when
`E ≤ 45` (build-batches); 1K pool only for a single large build.

---

## Scope A — Manipulation-stage / scene  (AUTOMATIC for every task)

| field | what | range | variability |
|---|---|---|---|
| **table texture** ✅ IMPLEMENTED | wooden / steel / tablecloth albedo maps — NOT pure colour, NOT too fancy; **≥10** incl. several **bright tablecloths** (e.g. red-white floral) | choice over the texture pack | per-build |
| **object-table relative height** ✅ IMPLEMENTED | the relative height between the **object table** and the arm table — lower/raise the **object table only** | **uniform ±5 cm** | per-env |
| **object-table size** ✅ IMPLEMENTED (Phase 2) | current size is the **minimum**; grow width (both y edges) + length. **Length extends ONLY away from the arm table** (the seam end pinned at `seam_x`); the textured top **Plane rescales** to stay flush | width +0..`OTABLE_GROW_W` (def 0.18m), length-away +0..`OTABLE_GROW_L` (def 0.22m) | per-build |
| **side-camera height + pitch** ✅ IMPLEMENTED (Phase 2) | **side cam only** (never the wrist cams). Current height = minimum; raise up to **+5 cm**; when raised, **pitch down** (lookat dropped by the same Δz) to keep the workspace framed. The visible D435i rig + stick rise with it | height +0..`SIDECAM_DZ_MAX` (5cm), pitch = re-frame | per-build |
| **table friction** ✅ IMPLEMENTED (Phase 2) | metal ↔ wood ↔ wool/fabric, **independent of texture** — a friction-ratio band on the collidable table Boxes, never coupled to the texture choice | uniform ratio band ~[0.55, 1.45] | per-env |

> **Table-texture — how it's implemented** (`world/manipulation_stage.py`): a 12-map pack lives in
> `assets/textures/tables/` (4 wood · 3 steel/metal · 5 tablecloth incl. red/blue gingham + red-white
> floral; built by `scripts/build_table_textures.py`, listed in that dir's `README.md`).
> `table_texture_pool()` globs it; `ManipulationStage.__init__` picks **one** map per build with the
> stage RNG → `self.table_texture`, applied to **BOTH** tables (they're flush at the seam → one
> continuous surface). The collidable table **Boxes** keep all physics/friction; a thin **visual-only,
> no-collision `gs.morphs.Plane`** sits on each top and carries the texture as a
> `gs.surfaces.Plastic(diffuse_texture=gs.textures.ImageTexture(image_path=…))` (a Box has no UVs so Nyx
> can't texture it — see `rendering_and_livery.md` §8). The object-table top Plane is **batched**
> (`batch_fixed_verts=True`) and the task calls `stage.set_otable_top_z(tabZ)` right after the per-env
> height DR so the texture stays flush on the randomized table. **Friction is untouched and stays
> per-env, fully decoupled from texture** (this code never sets friction).

## Scope B — Object / Task  (per-object, per-task; REALISTIC only)

| field | what | rule | variability |
|---|---|---|---|
| **color** ✅ IMPLEMENTED | object colour, **realistic** | banana = yellow (mostly) / green (unripe); apple = red / green; bowl/plate = china-white / plastic colours; cube = free; **never a blue watermelon** | per-build |
| **size** ✅ IMPLEMENTED (Phase 2) | object scale, **realistic** | usually **±`spec.size_band_frac`** (def ±10%, the thin pen ±6% to stay off its penetration floor); spawned at `spec.scale*(1±u)` so `scaled_extents()` feeds the grasp planner unchanged — no watermelon-sized apple | per-build |
| **mass** ✅ IMPLEMENTED | object mass | reasonable per-object range (±0.02 kg about `spec.mass`) | per-env |
| **friction** ✅ IMPLEMENTED (Phase 2) | object friction (the TARGET) | small realistic ratio band about `spec.friction` (`spec.friction_band`, def ±0.20), **distinct** from the robot-link friction knob | per-env |
| **damping / stiffness** | for **future articulated** objects: cabinet-door hinge, stapler joint, suitcase joint, drawer-track damping/friction | per-joint reasonable range | per-env |
| **container fill** | for pouring: water **volume** inside a cup → affects **weight** | volume range | per-env |
| **object type** | object **variants**: different staplers / cabinets / cups / bowls | choice over variants | per-build |
| **pose** (position + orientation) | randomize **as widely as task + arm + inter-object constraints allow** | see Pose rules ↓ | per-env |

### Arm assignment (per task structure — REQUIRED)
- **Single-arm tasks** (e.g. cube→bowl pick-place): split trials **half LEFT arm / half RIGHT arm** (balanced
  ~50/50, per-env). Place the object in the CHOSEN arm's reachable workspace (arm-relative — this is also what
  breaks the "always left-of-bowl for the left arm" coupling in the pose rules below). Both arms must be
  exercised equally so the policy learns to use either.
- **Dual-arm tasks** (e.g. pen-uncap, mug-hang, pouring two-handed): use **both arms together** (bimanual
  coordination) — no left/right split; the task assigns roles (e.g. one arm holds, one acts).

### Pose rules (the rich part of scope B)
- **Task-permitted**: never init in a state that trivially violates the task (e.g. cube already inside the bowl).
- **Arm-permitted**: some poses are hard for the arm (e.g. a mug pose hard to grasp/hang). Start with a **wide**
  range (at least inside the arm's reachable workspace), then **explore** for sensitive areas/poses.
- **Break obvious couplings**: do NOT always put the cube to the left of the bowl for the left arm — also try
  front / back / **right** (shift the bowl left so the left arm still reaches around). Do NOT always place all
  objects at the arm's **left-front** — also try right-front / front / cross-angle.
- **Inter-object constraints** (multi-object): respect spacing (e.g. don't spawn the mug too close to the
  hanger tree — the upper branches block grasping). Randomize the **relative pose between objects** AND the
  **relative pose to the arm** together, with min-clearance rejection.
- **Edge cases are welcome**: because we collect in parallel, we can be **open to hard edge cases** (they
  produce hard/failure data); explore ranges in advance only enough to keep success usable.

### Distractor / clutter objects (REQUIRED for every task) — ✅ IMPLEMENTED (cube→bowl; 2026-06-19)
A real table is never empty except for the target. So **every task spawns 2–3 RANDOM irrelevant objects** on the
table in OPEN areas, with **realistic physics + collision** (they rest, can be bumped, are solid):
- **Pool:** drawn from the object library — e.g. for cube→bowl, distractors ∈ {pen, banana, apple, tennis ball,
  book, …}. Count 2–3 per trial, identities + poses randomized (per-build for which set, per-env for poses).
- **Placement:** in open table area, respecting min-clearance from the target, the container, each other, and
  the arm's grasp corridor (rejection-sampled), so they clutter the scene without trivially blocking the task.
- **Collision-free planning (REQUIRED):** the motion planner must produce trajectories that do **not** collide
  with the distractors (avoid them on approach/transport). This bakes **native obstacle-avoidance** into the
  collected data → the trained policy learns it for free.
- **Why:** the policy must (a) pick the **correct** object among lookalikes (visual disambiguation) and (b) not
  swipe the others (collision avoidance). Clean single-object scenes teach neither.
- **Applies to ALL tasks** (not just pick-place), scaled to the task's geometry.

> **How it's implemented** (`tasks/pickplace.py`: `DISTRACTOR_POOL`, `choose_distractor_types`,
> `sample_distractor_poses`, `spawn_distractors`, `_spawn_distractor_entity`).
> - **Pool:** `{pen, banana, apple, tennis_ball, book}` from the REGISTRY (`registry/object_spec.py`). Each
>   renders realistically (apple red, banana yellow, pen dark, tennis ball yellow-green, book dark-red).
> - **Per-build = which TYPES** (entities are created before `scene.build()`): `choose_distractor_types` draws
>   K∈{2,3} distinct types with `stage.rng` (WITHOUT replacement → varied lookalikes); **at most ONE large/long
>   object (banana/book)** per build so all fit on the table out of the arm path.
> - **Per-env = POSES** (batched): `sample_distractor_poses` chooses each object's XY + in-plane yaw per env.
> - **Corridor-clearance rule (the key constraint):** the planner is pure waypoint-IK (no obstacle avoidance),
>   so collision-freeness is achieved by **PLACEMENT** — each distractor's whole footprint (circumscribed
>   radius, valid at any yaw) is kept clear of the **cube**, the **bowl**, the **cube→bowl carry tube**, the
>   **bowl→home return tube**, the **near-seam strip**, and the **active arm's home**, plus pairwise spacing so
>   nothing stacks. Placement = a fine anchor grid → greedily pick K mutually-spaced, corridor-clear cells
>   (favouring the opposite-y side from the active arm). On-table + corridor-clear + non-overlap verified
>   over 1500 seeds (0 corridor violations).
> - **Physics:** real collidable rigid bodies (USD objects render via Nyx-safe extracted `*_clean.obj` meshes —
>   the textured USDs segfault Nyx, same as the bowl), firm friction, settled with the cube/bowl; their CoM is
>   shifted **down** so a curved banana rests stably instead of slowly rolling.
> - **Not-knocked verification (the gate):** the task measures each distractor's XY displacement from its
>   settled pose to its final pose. **Verified across seeds 6/17/23/42/103 (8 envs each): max distractor
>   displacement ≤ 1.2 cm, 100% under 2 cm, none knocked**, while parity stayed 8/8 grasped+placed, 0
>   through-wall. Set `DIST_DEBUG=1` to print per-distractor displacement + a per-phase trace.

## Scope C — Visual background  (AUTOMATIC for every task)

| field | what | range | variability |
|---|---|---|---|
| **immersive background** ✅ IMPLEMENTED | the trial happens in a real scene: office / theatre / bedroom / outside / … (HDRI is the floor+walls+light; no ground plane) | choice over the HDRI pool | per-env |
| **light** ✅ IMPLEMENTED (Phase 2) | directional key-light colour: orange / white / yellow / light-blue / sunlight; + **intensity** | colour choice + intensity band ~[0.7, 1.7] | **per-build** (see limit ↓) |

> **Light DR — the honest per-env limit (verified in the Nyx SDK).** The Nyx renderer's per-env render loop can
> only switch the **env-map** (`renderer.set_env_map(env_index)`); the **directional light bakes at build**
> (`scene_asset.set_light(...)`, collected from the sensors once) and is **NOT** per-env settable — there is no
> `set_light(env_index)`. So the directional-light DR is **per-build** (one coloured key light per build, varied
> across build-batches — still adds variety). The **HDRI already supplies per-env image-based lighting** (each env
> renders in its own room with that room's light), so per-env illumination variety is preserved; only the
> directional *key* light is per-build. (`world/manipulation_stage.py`: drawn in `__init__` via
> `dr.sampler.sample_build_scene_dr`, baked into the `lights=` passed to all 4 Nyx cameras.)

---

## How it's applied (the DR harness)
- `dr/scopes.py` holds `SCENE_DR` (A) + `VISUAL_DR` (C) — the harness applies them to **every** task. As of Phase 2
  **every** scope-A/C field is `implemented` (table texture/height/**size**/**friction**, **side-cam pose**, HDRI,
  **light**); the only honest limit is the per-env→per-build light (above).
- `dr/object_dr.py` holds scope-B, keyed by object name, with realistic per-object palettes/ranges + the
  **size band** (`size_band_frac`) and **object-friction band** (`friction_band`) read off the `ObjectSpec`.
- `dr/sampler.py` splits a task's DR into a **BuildDR** (per-build draws: colours, **object size**, + the realised
  build-baked **table-grow / side-cam / light**) + a batched **EnvDR** (per-env arrays incl. **table friction** +
  **object friction**). `sample_build_scene_dr(rng)` draws the build-baked scope-A/C scene fields the stage needs
  at construction time (geometry/visual bake at build).
- `dr/apply.py` `apply_build_dr` spawns the target at the **size-DR'd** spec (`scaled_target_spec`) before build;
  `apply_env_dr` runs the batched per-env setters after build (object-table height on the **grown** Box, **table
  friction**, bowl/target pose+yaw, mass, **object friction**, robot-link friction). The table-grow / side-cam /
  light are realised inside `ManipulationStage` at build (it owns its build-time geometry/visual setup).
- `dr/plan.py` writes the exact sampled values per demo into the HDF5 attrs — every demo is fully traceable. New
  Phase-2 keys: `dr_obj_scale`, `dr_table_fric`, `dr_obj_fric`, `dr_otable_w/l` (+ `dr_otable_grow_w/l`),
  `dr_sidecam_dz`/`dr_sidecam_z`, `dr_light_name`/`dr_light_r,g,b`/`dr_light_intensity`. The **DR subagent**
  (`.claude/agents/dr-strategist.md`) learns from it (see [agents.md](agents.md)).
- **Toggles** (regression/ablation): `FULL_DR=0` (no build-baked scene DR), `DR_SIZE_SCALE`, `DR_OTABLE_SCALE`,
  `DR_SIDECAM_SCALE`, `DR_TABLE_FRIC_SCALE`, `DR_OBJ_FRIC_SCALE` (each a half-width multiplier; 0 = field off).

A task NEVER writes DR logic — it names which scope-B fields apply (its `TaskSpec`); scopes A and C are free.
