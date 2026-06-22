# Pick-place task collector + data format (and how to add a new task)

This is the manual for the cube→bowl pick-place **task** in the Genesis "Line B" firefly stack, the
**HDF5 + video data format** it writes, and **how to add a new task** on top of the same reusable world.

The whole robot + cameras + photoreal rendering + per-env immersive HDRI background DR live in the
**reusable `ManipulationStage`** (`world/manipulation_stage.py`). The task file is a *thin composer*: it spawns
the objects (via `world/object_factory.py`), applies the full DR (`dr/`), plans with the reusable skills,
scores, and writes data.

| | |
|---|---|
| **Task (thin composer)** | `genesis_firefly/tasks/pickplace.py` (978 lines) |
| **Reusable world** | `genesis_firefly/world/manipulation_stage.py` (`ManipulationStage`) |
| **Entry / one build** | `genesis_firefly/runner/collect.py` (thin wrapper) |
| **Orchestrator (B builds → /data3)** | `genesis_firefly/runner/orchestrate.py` |
| **Data root (default)** | `/data3/genesis_fulldr` (`DATA_DIR` env) |
| **Tile/video root (default)** | `genesis_firefly/output/temp/fulldr_collect` (`OUT_DIR` env) |
| **Reference dataset** | `cube_fulldr_v3`: 200 demos, 197 placed, 0 abnormal penetration → LeRobot 197 ep |

> All N trials run in **ONE fully-parallel build** (`stage.build()`), each demo in its own batched env with
> its own random real room. No subprocess batches.

---

## 1. The task: god-mode scripted cube → bowl

The task is a **fully scripted (god-mode) cube-into-bowl pick-place** — no policy, no teleop, no imitation
seed. It uses the reused RoboLab skills (`skills/grasp.py`) for the orientation-aware grasp + the gentle
top-down place, and **Genesis-native IK** (`robots/ik.py`, the SODA-IK-equivalent tool-frame adapter) to turn
EE-space waypoints into arm joints. The motion is a 9-waypoint plan, joint-space-densified with an ease curve.

### 1a. The objects

Both objects are added to `stage.scene` by the task **before** `stage.build()`:

```python
cube = stage.scene.add_entity(
    gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(0.40, 0.18, 0.30)),
    material=gs.materials.Rigid(rho=600.0, friction=1.0),
    surface=gs.surfaces.Plastic(color=cube_col, roughness=0.35))
bowl = stage.scene.add_entity(
    gs.morphs.Mesh(file=BOWL_OBJ, convexify=True,
                   decompose_object_error_threshold=0.04, decimate=False),
    material=gs.materials.Rigid(rho=400.0, friction=1.0),
    surface=gs.surfaces.Smooth(color=bowl_col))
```

- **Cube** — a 5cm procedural box (`REGISTRY["cube"].scaled_extents()` = `(0.05, 0.05, 0.05)`), plastic PBR,
  `is_cube=True` → orientation-aware **face-pair** grasp.
- **Bowl** — `assets/objects/ycb/bowl_clean.obj` (`BOWL_OBJ`), the **Nyx-safe** visual mesh extracted from
  the YCB bowl USD (the textured `bowl.usd` segfaults Nyx — "MaterialBindingAPI not applied" — so the visual
  geometry is baked to a clean OBJ). Glossy ceramic `gs.surfaces.Smooth`.

> The task file lives at **`tasks/pickplace.py`** and the reusable stage at **`world/manipulation_stage.py`**
> (the `collectors/`·`scenes/` paths in older revisions of this doc were renamed in the P0 reorg).

### 1a-bis. CONFIGURABLE TARGET object (any registry object, default `cube`)

The **same task** picks-and-places **any registry object** as the grasp target. The **`TARGET` env var** selects
it; the default is `cube`, so `pickplace.py N seed` reproduces the cube collection **exactly** (the regression
gate: `TARGET=cube pickplace.py 8 7` → 8/8 grasp+place, 0 penetration, max|dq|=0.068, byte-for-byte). Switching
the target is a **small per-object adaptation, not a fork** — the orientation-aware grasp, the full DR, the 50/50
distractors, the penetration gate, and go-home are **all reused unchanged**.

```bash
TARGET=banana CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/tasks/pickplace.py 20 7
# default DATA/OUT dirs are suffixed by the target (…/genesis_fulldr_banana) so objects never overwrite.
```

**What is target-specific (everything else is shared):**

| Piece | Mechanism |
|---|---|
| **which object** | `spec = REGISTRY[TARGET]` (default `cube`); `spawn_target(scene, spec, color, …)` builds it. |
| **collider** | procedural box/sphere keep their exact collider; a **USD-mesh** target gets a **faithful grasp collider** on its Nyx-safe `*_clean.obj` — convex **decomposition** by default, or a **single convex hull** (`spec.grasp_single_hull`) for a near-convex rounded body so the firm pinch can't drive into a decomposition seam and NaN the solver. |
| **ref-axis** | already encoded by the spec: **long axis** (elongated) · **face pair** (cube) · **None** (round) — fed to `orientation_aware_grasp_quat`. |
| **grasp point** | `grasp_center = root + R(quat)·spec.grasp_center_offset_local` (shift onto the **body** of a CURVED object — a banana's AABB centre sits in the hollow of the curve ~3cm off the fruit) `+ [0,0,spec.grasp_dz]`. |
| **grasp tuning** | per-object `grasp_dz`, `grasp_close` (gentler pinch for a rounded/soft body), `grasp_single_hull`, `grasp_decompose_err`, `grasp_center_offset_local`, `friction`, `place_xy_tol_cm` — all `ObjectSpec` fields. |
| **spawn-clear floor** | the cube uses the locked 0.125; a bigger target scales it: `footprint_radius + bowl_radius + 1.5cm` (so a banana/book never spawns half-in the bowl). |
| **distractor pool** | **excludes the target's own type** so the target is never ambiguous among lookalikes (cube target → the legacy `{pen,banana,apple,tennis_ball,book}`; a banana target → `{pen,apple,tennis_ball,book,cube}`). |
| **scoring** | `place_xy_tol` = 6cm (cube) or `spec.place_xy_tol_cm` (bigger objects whose bbox centre rests a few cm off the bowl centre); a non-cube height band allows an elongated body draped across the rim. |

**Debug knobs (env, off by default — for grasp/posture/penetration tuning only):** `FAST=1` skips the Nyx render
in the run loop (~6× faster) AND skips ALL video/tile/montage/preview writing — it writes **ONLY `demos.hdf5`**
(the numeric data the posture/penetration checks read) + the `[COLLECT]` prints, **no `.mp4`/`.png` at all** (the
old FAST wrote blank "black-stripe" placeholder videos; that is gone — never use FAST for a real collection, it
produces no policy videos). `GRASP_DZ`/`CLOSE_G`/`TGT_FRIC`/`TGT_SINGLE_HULL`/`TGT_DECOMP` override the per-object
grasp params, `PEN_TRACE=1` prints the worst-ever penetration per phase.

### 1a-ter. Per-object status (SOLVED on the natural-motion foundation, 2026-06-20)

The orientation-aware grasp + the configurable-target machinery work for **every** object, and the "needs a dex
hand" conclusion of the earliest attempt was a **broken-foundation symptom**. Three real root-cause fixes — NO
new grasp machinery — solved the round/thin/elongated objects (full detail in `roadmap.md` 2026-06-20):
1. **Round-object ejection = a leaky friction cone, not geometry.** `firm_rigid_options()` never set
   `noslip_iterations` (defaulted 0); a curved/thin body is a single tangent contact per finger so the leaky cone
   squirted it out. FIX: set it **PER-OBJECT** (`spec.grasp_noslip`: cube=0, round/curved/thin=5).
2. **Elongated large-yaw misses = wrong symmetry fold.** `grasp_quat_at` folded every yaw into the cube's 4-fold
   wedge; an elongated object is 2-fold. FIX: π/2 for the cube, π for elongated → banana 6/12 → 12/12.
3. **Round go-home wrist-roll snap = yaw-degenerate roll.** FIX: snap the round grasp roll to the home-nearest
   branch → max|dq| 3.3 → 0.02.

| Object | Geometry | Status (clean path, real renders) |
|---|---|---|
| **cube** | 5cm flat box | ✅ **8/8** · 2.5mm (0 abnormal) · posture 1.40/1.14 |
| **apple** | ~7cm sphere (native texture) | ✅ **12/12** · 4.4mm (0) · 1.43/1.07 |
| **tennis_ball** | 6.7cm felt sphere, 57g | ✅ **12/12** · 6.4mm (0) · 1.43/0.98 |
| **banana** | curved, ~3.8cm girth | ✅ **12/12** · 6.7mm (0) · 1.41/0.85 |
| **pen** | thin 2.1×1.9×12cm, 20g | ✅ **12/12** · 6.9mm (0) — the framework's HARDEST penetration corner (rides the 7mm gate; see residuals) |
| **book** | flat slab 18×13×3cm | ❌ **geometric** — both flat dims exceed the ~6cm jaw and the only graspable (3cm) dim is vertical when flat; needs a side-approach / edge-pinch skill (future) |

> **Penetration columns:** `placed/N · max_pen(abnormal) · posture (j4max/j3min)`. All max|dq| ≈ 0.02 (smooth).
> The **pen** is logged as the hardest penetration corner — it passes the 7mm gate with the least margin; the
> grasp-retry's big-clean-miss mode is what keeps its noised attempts off the gate (see `grasp_retry.md`).
> The **book** is the one geometrically-unsolved object for a parallel jaw (deferred to a side-approach skill or
> the Line C dexterous hand).

**The bowl collision model is the single most important realism choice.** It is loaded as a **convex
DECOMPOSITION** collider (`convexify=True, decompose_object_error_threshold=0.04, decimate=False` — coacd),
exactly RoboLab's PhysX `convexDecomposition` recipe. The thin concave shell is split into a set of **solid
convex hulls** tiling the wall+floor; convex-vs-convex contact is robust **everywhere, including the thin
~2–3mm rim**, so a cube that lands on the rim **rolls in or out and can never tunnel through the wall**.

> **Do not** use the nonconvex watertight-SDF (`convexify=False`) bowl. It gives a *degenerate ~0 contact*
> where a box **corner** straddles the rim (`narrowphase.py:300-309` emits only a synthetic `1e-4 m`), so a
> corner on the rim tunnels. The rim-drop test (tilted cube on the rim, offset 4/6/7cm): **SDF → stuck in the
> wall; coacd → rolled in, never in-wall.** coacd's cosmetic ~8mm floor-raise is accepted (RoboLab accepts it).

**The cube is spawned CLEAR of the bowl** — a cube can never start embedded in a bowl in reality. The DR
rejection-samples the cube xy until `hypot(cube − bowl) ≥ 0.125` (see §2). The cube spawns ~1cm above the
table and settles during `settle_home`. **GUARANTEED (2026-06-19 fix):** the rejection loop used to be able to
*exhaust* its tries on an unlucky/infeasible draw and silently ship a still-overlapping cube — the firm solver
then **ejected** it off the (ground-plane-less) table to z≈−6 m, whose garbage settled pose blew the grasp
trajectory up to 10× (see roadmap). The loop now **clamps** any still-overlapping cube radially outward to
exactly the 0.125 floor, so a cube can NEVER spawn intersecting the bowl by construction.

### 1b. The skill: orientation-aware grasp + gentle top-down place (the two extracted ACTION skills)

The motion is composed from two **reusable, de-closured ACTION skills** (a future grasp-based task imports +
composes them; both return labelled EE-space waypoints `(label, tool_pos, tool_quat_wxyz, gripper_scalar)`):

- **Grasp action** `skills/grasp.py::grasp_action_wps(...)` → `home → pre → at → at → close → lift` (the
  orientation-aware top-down approach + a FIRM GR100 close — the object's own collision stops the claws, the
  high-kp PD holds the force — + a gentle lift).
- **Place action** `skills/place.py::place_action_wps(...)` → `lift(settle + re-yaw) → carry → lower → rel → ret
  → go_home` (re-yaw to the carry orientation, carry over the bowl, lower, release ABOVE the rim, retract, and a
  smooth densified RETURN HOME — recorded). The carry quat is computed by the caller (via `grasp.cquat`) and
  passed in, so the place module has NO GraspContext dependency.

The two actions are APPENDED into ONE continuous per-env waypoint stream → each env runs pick→place→home as a
single smooth trajectory and TERMINATES at home (no staged barrier, no mid-air wait). The opt-in
`skills/grasp_retry.py` re-uses the SAME two builders for the miss→retry path (see `grasp_retry.md`).

- **Grasp orientation** `gq` (per env, `grasp.grasp_quat_at(gctx, i, …)`): the object's world reference axis is a
  face normal (cube) / long axis (elongated) / None (round), folded into the symmetry wedge (**π/2 for the cube,
  π for an elongated object** — the fold-bug fix). Built with `orientation_aware_grasp_quat(...)` so the jaws
  close **across** the short axis, never across the diagonal, with the relax-tilt (`select_grasp_tilt`).
- **Carry/place orientation** `cq` (per env, `grasp.cquat(gctx, i, gq)`): the **least-wrist-rotation** carry
  orientation (small tilt toward the bowl + the place relax-tilt) so the wrist re-yaws smoothly during the lift,
  not in place at the top. A round grasp's free roll is snapped to the home-nearest branch (the go-home snap fix).
- **Gentle top-down place — the second realism rule.** The target is **released ABOVE the rim and free-drops**
  into the solid bowl; it is *never* lowered while gripped to a target inside the wall. A position-controlled
  arm (`kp=200`) would otherwise **drive the held target through the wall** no matter how solid the collider is.
  The release height is `drop_z = tabZ + 2*BOWL_HALF_H + spec.scaled_extents()[2]/2 + 0.012` (above the rim;
  `BOWL_HALF_H = 0.02748`). The `lift` settle waypoint keeps the target stationary over the bowl so its lateral
  carry velocity → 0 **before** the jaws open — otherwise a residual carry velocity slips a corner through a hull
  seam (this fixed the last ~1/100 real penetration).

The grasp point is `gc = root0 + R(quat)·spec.grasp_center_offset_local + [0,0,grasp_dz]` from the settled target
root (the offset shifts the grasp onto the BODY of a curved object). IK is batched per arm
(`robot.entity.inverse_kinematics(..., dofs_idx_local=idx, max_solver_iters=24)`), converting the tool pose →
ee_link pose with `TOOL_IN_EE_INV` (the measured ~11cm ee-link-behind-claws offset). Waypoints are
joint-interpolated with a smoothstep ease `ease(u)=3u²−2u³`; the gripper scalar is interpolated and mirrored
to the mimic claw (`GR100_MIMIC * g`).

- **Densify step-count CAP (`max_move_steps=600`, 2026-06-19).** Because `gc = root0` (the *settled* cube pose),
  a cube that settled wrong (e.g. ejected off the table) makes the `home→pre` / `lift→carry` segments span
  metres, and `densify`'s per-segment count `n = max(lin/speed, ang/ang_speed)/dt` is then huge (~5000 steps for
  6.5 m). Since `BatchExecutor` pads EVERY env to the common max length T, ONE such env 10×'d the whole build.
  `skills/trajectory.py:densify` now HARD-caps `n` at `max_move_steps` (default 600 ≈ 6 s, well above the ~540 a
  real 0.7 m workspace move needs), threaded through `BatchExecutor`. Normal moves are unchanged; a degenerate
  waypoint simply can't dominate T. (Layer 1 of a 3-layer defense — see roadmap 2026-06-19.)
- **Degenerate-settle guard + REJECT (2026-06-19).** Right after `root0 = cube.get_pos()`, the task flags any env
  whose settled cube is non-finite, drifted >5 cm in XY from its intended spawn, or off the table in z as
  `degenerate`: it **clamps** that env's `root0` back to a sane on-table pose (so its own motion + IK stay
  well-conditioned) AND forces `success=False` (a demo grasping at a phantom location never ships; HDF5 attr
  `degenerate`). This is Layer 2; Layer 3 is the spawn-clear clamp in §1a / §2a that prevents the ejection at the
  source. The clean fix is the spawn clamp; the cap + guard are belt-and-braces for any future degenerate source.

---

## 2. Full domain randomization

> **The complete, authoritative DR spec is [domain_randomization.md](domain_randomization.md), and it is now
> FULLY APPLIED via the reusable `dr/` package** (`scopes.py` scope-A + scope-C · `object_dr.py` scope-B ·
> `sampler.py` split → per-build `BuildDR` + batched per-env `EnvDR` · `apply.py` · `plan.py` trace). This section
> documents the per-env physics axes the cube→bowl task exercises; the once-missing fields (object-table size,
> side-cam pose, table friction, object size, object friction, light) are all live through `dr/`. The one honest
> limit: **light is PER-BUILD** (Nyx can't set the directional light per-env; the HDRI supplies per-env
> image-based lighting). All N envs run in one build.

### 2a. Physics DR — the per-env, independent axes (sampled by `dr/sampler.py::sample_env_phys`)

| Axis | How | Range |
|---|---|---|
| **which arm** | `side_is_left = rng.rand(N) < 0.5`; `sgn=+1` left / `−1` right | 50/50 L/R |
| **object-table height** | `tabZ = ho + (rng−0.5)*0.10` (`ho` = `lay.object_table_height` = 0.25; **arm table fixed**) | ±5cm |
| **bowl xy** | `bowx = 0.40 + (rng−0.5)*0.10`; `bowy = sgn*(0.05 + (rng−0.5)*0.07)` (bowl on the active arm's side) | x ±5cm, y ±3.5cm |
| **cube xy** | `cubx = 0.40 + (rng−0.5)*0.12`; `cuby = sgn*(0.185 + (rng−0.5)*0.10)` | x ±6cm, y ±5cm |
| **cube spawn-clear** | reject-sample (≤60 tries) until `hypot(cube − bowl) ≥ clr` (clr=0.17 ~80%, 0.125 ~20%), THEN **clamp** any still-overlapping cube radially out to the 0.125 floor | never embedded in the bowl — GUARANTEED (the clamp can't fail; 2026-06-19) |
| **cube yaw** | `yaw = (rng−0.5)*radians(180)` — **full in-plane yaw** about world +Z | ±90° |
| **cube mass** | `mass_shift = (rng−0.5)*0.04` (kg), via `cube.set_mass_shift(...)` | ±20g around ~75g |
| **friction** | `robot.entity.set_friction_ratio(0.7 + 0.6*rng)` (per link) | ratio 0.7–1.3 |

(mass/friction DR is wrapped in a `try/except` — it prints `mass/fric DR skipped` and continues if the backend
build lacks the per-env mass/friction support.)

### 2b. Visual + scene DR — scope A + scope C (AUTOMATIC, applied via `dr/` + `ManipulationStage`)

The stage + the `dr/` package do the **scene + environment** DR so it never needs re-tuning per task:

- **Per-env immersive HDRI room** — one random HDRI per env (`stage.hdrs[e]`). Each of the N batched envs
  renders in its **own random real room** via Nyx per-env env maps (`update_scene → set_env_map`). **No ground
  plane** — the HDRI *is* the immersive floor + walls + image-based light. **Resolution = 2K when `n_envs ≤ 45`
  (the build-batch path), else the 1K pool** (`/data3/hdr1k`, built lazily by `hdr_pool()`) because **Nyx
  segfaults past ~50–60 2K env maps**.
- **Per-build table texture + colours** — `stage.table_texture` (one albedo from the ≥10-map pack, both tables
  share it) + `stage._pick_table()` table colours; `stage.distinct_object_color(*avoid_h)` returns a saturated
  object colour **guaranteed ≠ the table** (and ≠ any avoided hue). The target colour follows its per-object
  policy (native texture / realistic palette / fixed / free cube) via `world/object_factory.py::target_color`.
- **Per-build scene DR (scope A/C, `full_dr=True`)** — object-table size grow, side-cam height/pitch, and the
  **per-build directional key light** colour/intensity (the honest per-env→per-build limit) are drawn in
  `dr/sampler.py` and realised inside the stage at build (geometry/visual bake at build).
- **Per-env scene DR** — object-table height (±5cm) + **table friction** (a friction band decoupled from the
  texture) applied by `dr/apply.py::apply_env_dr`.
- **Lighting** — the per-build directional key light + the HDRI image-based light; the arm uses a **matte**
  entity-surface override (`gs.surfaces.Default(metallic=0.0, roughness=0.7)`) so the silver SOMA livery reads
  neutral in any warm room (no env-mirror).

Cameras/rendering: 4 Nyx path-traced sensors at `RES=(640,360)`, `SPP` (default 32, `SPP` env):
`third` (low third-person witness), `cam_side` (world-fixed D435i), `cam_lw` / `cam_rw` (egocentric D405 wrist
cams attached to `left_link_6` / `right_link_6`). `stage.render()` reads them via the **sensor API**
(`cam._stale=True; cam.read().rgb`) so the wrist cams re-attach to link_6 each frame (true egocentric).

### 2c. Distractor / clutter objects (scope-B REQUIRED) — 2–3 irrelevant objects per trial

Every trial spawns **2–3 random irrelevant objects** on the OBJECT table in open areas (so the policy learns to
pick the **correct** object among lookalikes AND to **not swipe** the others → native collision-avoidance in the
data). The corridor-aware **PLACEMENT** (which types + which cells) is the reusable `skills/distractors.py` skill
(the task passes its keep-out corridors); the **SPAWN** (sim entities) is `world/object_factory.py::spawn_distractors`;
the task (`tasks/pickplace.py`) wires the two with its `DISTRACTOR_POOL`:

- **Pool** (`DISTRACTOR_POOL`): `{pen, banana, apple, tennis_ball, book}` from the REGISTRY — each renders with a
  realistic colour/texture/size/mass/friction. USD-sourced objects (apple/banana/pen) render via their **Nyx-safe
  extracted `*_clean.obj`** mesh (the textured USDs segfault Nyx, same as the bowl).
- **Per-build = TYPES** (`choose_distractor_types`, drawn with `stage.rng` BEFORE `build()`): K∈{2,3} distinct
  types, **at most one large/long object** (banana/book) so all fit on the table out of the arm path.
- **Per-env = POSES** (`sample_distractor_poses`, batched): each object's XY + in-plane yaw per env.
- **Corridor-clearance rule** (the key constraint — the executor does NO obstacle avoidance, so this is how the
  trajectory stays collision-free): every distractor's whole footprint (circumscribed radius → valid at any yaw)
  is kept clear of the **cube**, the **bowl**, the **cube→bowl carry tube**, the **bowl→home return tube**, the
  **near-seam strip**, and the **active-arm home**, plus pairwise spacing (no stacking). Placement = a fine
  anchor grid → greedily pick K mutually-spaced, corridor-clear cells (biased to the opposite-y open area).
- **Physics:** real collidable rigid bodies (firm friction; single-convex-hull collider since they're never
  grasped) settled with the cube/bowl; their **CoM is shifted down** so a curved banana rests stably (doesn't
  slowly roll).
- **Not-knocked gate:** the task measures each distractor's settled→final XY displacement and prints it; verified
  **≤ 1.2 cm, 100 % under 2 cm** across seeds while parity stays 8/8. `DIST_DEBUG=1` adds a per-distractor +
  per-phase trace. See `domain_randomization.md` (distractor subsection, ✅ IMPLEMENTED).

---

## 3. Success + penetration scoring (per-env, after the trial)

After the trajectory + a 40-step settle, the task reads the final cube pose, the active-arm ee pose, and the
lift-peak pose, then computes (per env):

```python
objf    = cube.get_pos()                       # final cube pos
lift_cm = (lift_pos[:,2] - root0[:,2]) * 100   # rise captured at the LIFT waypoint peak
rxy     = hypot(objf[:,0]-bowx, objf[:,1]-bowy)   # radial dist from the PER-ENV bowl centre
rim_z   = tabZ + 2*BOWL_HALF_H
ch2     = spec.scaled_extents()[2] / 2         # cube half-height

placed = (lift_cm > 3)                              # lifted clear of the table
       & (rxy < 0.06)                               # settled within the bowl footprint
       & (objf[:,2]-ch2 > tabZ - 0.005)             # not below the table
       & (objf[:,2]-ch2 < rim_z + 0.01)             # not perched above the rim
       & (norm(objf - eep) > 0.08)                  # released (clear of the gripper)
```

**A demo counts as `placed` (= `success`) iff: lifted clear + settled inside the bowl footprint + at a
plausible height (above the table, not above the rim) + released from the gripper.** "grasped" alone is
`lift_cm > 3`.

**Penetration metric (per-env bowl-centre wall annulus).** The old fell-through / far-outside check was
*blind* to wall penetration (reported 0 while cubes tunnelled), so a dedicated **wall-annulus** metric is used,
measured from the **per-env** bowl centre:

```python
bottom   = objf[:,2] - ch2                          # cube's lowest point
wall_pen = ((rxy > 0.065) & (rxy < 0.11) &          # in the wall annulus (not cavity, not table)
            (bottom > tabZ + 0.012) & (bottom < rim_z))   # held UP in the wall, not resting on the table
         | (bottom < tabZ - 0.015)                  # OR sunk below the table
```

`wall_pen` flags a cube **held up inside the wall annulus** (a real through-wall) or **sunk below the table**.
A cube that rolled out cleanly onto the table is *not* a penetration. Always confirm visually with the **tight
bowl close-up** in the four-view tiles — the wide third-person view hides wall penetration.

Console line per run (e.g. the cube reference, N=20):
```
[COLLECT] 20/20 grasped, 20/20 placed, through-wall=0/20  render+sim <wall>s
[COLLECT] penetration: max=2.5mm, abnormal=0/20
```

---

## 4. The HDF5 schema (exact) + the videos & tiles

`<DATA_DIR>/demos.hdf5` — one group per demo, `T` = recorded frames (state + render captured every
`REC_EVERY = 10` sim steps, plus one final frame). Arrays are `float32`.

```
/data/demo_<i>/
  actions                                  (T, 14)   absolute 14-D joint targets (the PD command @ state14_idx)
  states/articulation/robot/
    joint_position                         (T, 14)   measured joint positions (state14_idx)
    joint_velocity                         (T, 14)   measured joint velocities (state14_idx)
  ee_pose/
    position                               (T, 3)    active-arm ee_link world position
    orientation                            (T, 4)    active-arm ee_link world quaternion, WXYZ
  attrs:
    num_samples = T          (int)
    success     = bool(placed[i])
    seed        = int(seed)
    arm         = "left" | "right"         (which arm did this demo)
    hdr         = basename(stage.hdrs[i])  (the HDRI room for this env)
    max_penetration_mm = float             (the worst-ever solid-solid overlap; the #1 collision gate)
    penetrating        = bool              (max_penetration_mm > ABNORMAL_THRESH_M=7mm → dropped from success)
    degenerate         = bool              (settled target was non-finite / off-table → success forced False)
    dr_*               = the per-demo DR trace (every sampled DR value, written by dr/plan.py — fully traceable)
```

The **14-D layout** is `STATE_JOINTS_14 = [L_j1..L_j6, L_grip_driven, R_j1..R_j6, R_grip_driven]`
(6 arm + 1 driven gripper, ×2 arms; the mimic gripper joints are **not** in the state). `actions` are the
absolute PD position targets at those 14 dof indices (`full_cmd[:, state14]`), not deltas. Quaternions are
**wxyz** (Genesis convention). `ee_pose` follows the **active arm** per demo (`np.where(side_is_left, ee_l, ee_r)`).

### Per-demo policy videos (the sensor-only stream)

```
<DATA_DIR>/videos/cam_side/demo_<i>.mp4     # world-fixed D435i side cam
<DATA_DIR>/videos/cam_lw/demo_<i>.mp4       # left  D405 wrist cam (egocentric, on left_link_6)
<DATA_DIR>/videos/cam_rw/demo_<i>.mp4       # right D405 wrist cam (egocentric, on right_link_6)
```

`fps=12`, `codec=libx264` (H.264, VSCode-friendly). Only the **3 policy cameras** are written per demo —
they are the policy's observation stream. (`third` is the rig camera, used only for the tiles below.)

### Visualization tiles (under `<OUT_DIR>`)

```
<OUT_DIR>/fulldr_third_<N>.mp4              # one ceil(sqrt(N)) x ceil(sqrt(N)) grid of EVERY demo's THIRD-person view
<OUT_DIR>/fourview_demo_<i>.mp4             # 2x2 (third | side / wrist-L | wrist-R) tile, for 10 random demos
```

The third tile is `ceil(sqrt(N))` per side (10×10 for N=100). The four-view tiles are 10 random demos
(`rng.choice(N, 10)`), each frame stacked `[[third, side], [wrist-L, wrist-R]]` with text labels. These are
the human-facing QA artifacts (the tile is where you eyeball DR coverage + grasp/place quality across all demos).

---

## 5. How to run

```bash
# One build (the task is runnable as a script):
CUDA_VISIBLE_DEVICES=0 [TARGET=cube] ./.venv/bin/python genesis_firefly/tasks/pickplace.py <N> [seed]

# Or via the thin entry wrapper (identical):
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/runner/collect.py <N> [seed]

# Scale to a full-DR dataset on /data3: B subprocess builds × E envs (one sim process per GPU), merged
GPUS=0,1 ./.venv/bin/python genesis_firefly/runner/orchestrate.py <B> <E> [seed0] [dataset]

# Examples
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/tasks/pickplace.py 20 7         # one 20-env build
CUDA_VISIBLE_DEVICES=0 FAST=1 NOISE_RETRY=0.45 ./.venv/bin/python genesis_firefly/tasks/pickplace.py 16 7
GPUS=0,1 ./.venv/bin/python genesis_firefly/runner/orchestrate.py 10 20 7 cube_fulldr_v3 # 200 demos, 10 looks
```

Args: `N` (number of demos / parallel envs), `seed`. Env vars:

| Env var | Default | Meaning |
|---|---|---|
| `TARGET` | `cube` | which registry object to pick-place (any `ObjectSpec`); DATA/OUT default dirs are suffixed by it |
| `NOISE_RETRY` | off | opt-in object-agnostic miss→retry fraction (see `grasp_retry.md`); off = byte-identical clean path |
| `DATA_DIR` | `/data3/genesis_fulldr[_<target>]` | the fine-tune dataset (HDF5 + `videos/`); large, keep on `/data3` |
| `OUT_DIR` | `output/temp/fulldr_collect` | the visualization tiles |
| `FAST` | off | metrics only — writes ONLY `demos.hdf5` + the `[COLLECT]` prints, **no `.mp4`/`.png`** (never for a real dataset) |
| `SPP` | `32` | Nyx samples/pixel (denoised; 32 is clean) |
| `FULL_DR` | `1` | the per-build scope-A/C scene DR (table-size grow, side-cam, light) |
| `CUDA_VISIBLE_DEVICES` | — | **pin to one GPU**; Genesis batches the N envs internally |

> **One GPU per process.** Genesis vectorizes the N demos into one batched build — do **not** launch parallel
> processes on a single GPU (multiple Isaac/Genesis instances on one GPU silently break timed grasps). To scale,
> use `runner/orchestrate.py` (one subprocess build per GPU).

**Reference dataset (`cube_fulldr_v3`, 10×20):** **200 demos, 197 placed, 0 abnormal penetration**, ~50/50 L/R,
natural variable-length motion → LeRobot `genesis_cube_fulldr_v3` (197 episodes, pi0.5-ready). The 3 non-placed
are cubes that rolled cleanly out of the bowl onto the table (realistic), not tunnelling.

---

## 6. How to add a new task

A new task is a **THIN composer on top of the same stage + skills + DR** — the robot, cameras, rendering, the
full-DR harness, the 14-D HDF5 schema, the video/tile machinery, and the IK adapter are all reused **unchanged**.
To add a task (mirror `tasks/pickplace.py`):

1. **Spawn the objects** via `world/object_factory.py` — `spawn_target(scene, spec, color, …)` for the grasp
   target (faithful convex-decomposition / single-hull collider + native texture) and `spawn_distractors(...)` for
   the clutter, added to `stage.scene` **before** `stage.build()`. Add an `ObjectSpec` to the registry first
   (the object-refiner agent can produce one). For a concave container, follow the bowl's **convex-decomposition**
   recipe (`convexify=True, decompose_object_error_threshold=0.04`) + a Nyx-safe visual mesh.
2. **Apply the DR** — name which scope-B fields apply (scopes A + C are free); `dr/apply.py::apply_build_dr` before
   build + `apply_env_dr` after. The DR-strategist owns the ranges + the recognizability/colour policy.
3. **Plan with the skills** — compose `skills/grasp.py::grasp_action_wps` + `skills/place.py::place_action_wps`
   (optionally `skills/grasp_retry.py`, later a `virtual_ee`), built from `grasp.grasp_quat_at`/`cquat` + the
   relax-tilt selection; run through `skills/executor.py::BatchExecutor`. For any container task, keep the
   **release-above + free-drop** rule (never drive a held object to a target inside a wall) + a settle waypoint to
   zero velocity before release.
4. **Score** with `skills/score.py::score_placement` (spec-aware) + the `skills/penetration.py` GATE. Keep the
   per-env, measured-pose pattern (read final pose vs the per-env target; measure penetration from the solver
   buffer). MEASURE penetration and render a tight close-up before claiming "no penetration".
5. **Reuse everything else unchanged** — `ManipulationStage` (do not edit it; it owns the world), the §4 HDF5
   write + the `dr_*` trace, the `videos/cam_{side,lw,rw}` policy stream, the `sqrt(N)` third tile + four-view
   tiles, and the `TARGET`/`DATA_DIR`/`OUT_DIR`/`SPP`/`NOISE_RETRY` run interface. To scale, use
   `runner/orchestrate.py`.

### Reusable API quick reference

```python
# world/manipulation_stage.py
stage = ManipulationStage(n_envs, seed=0, res=(640,360), spp=32, noslip=0, full_dr=True)
stage.scene                 # the gs.Scene -> add your objects to it BEFORE build
stage.distinct_object_color(*avoid_h) -> (hue, rgb)   # colour != table (and != avoided hues)
stage.build()                                          # build(n_envs, env_spacing=(0,0)) + robot.finalize()
stage.settle_home(steps=70) -> home_cmd (N, n_dofs)
stage.render() -> {"third"|"cam_side"|"cam_lw"|"cam_rw": (N,H,W,3) uint8}
stage.set_otable_top_z(top_z) / stage.set_table_friction(ratio)   # used by dr/apply.py::apply_env_dr
stage.robot / stage.cams / stage.lay / stage.rng / stage.hdrs / stage.otable / stage.table_texture / stage.H / stage.W
np_(x)                       # CUDA tensor -> numpy at the sim<->skills boundary (quat = wxyz)

# world/object_factory.py
spawn_target(scene, spec, color, ...)      # the grasp target: faithful collider + native texture
spawn_distractors(scene, ...)              # the clutter (the task passes its placement samplers / keep-outs)
build_object(...) / target_color(spec, rng)

# robots/firefly_dual.py  (FireflyDual, accessed via stage.robot)
robot.arm["left"|"right"]            # 6 arm dof indices (Genesis INTERLEAVES dual-arm dofs)
robot.grip_driven / robot.grip_mimic # gripper dof indices per side
robot.state14_idx                    # the 14 dof indices for actions/states (STATE_JOINTS_14)
robot.ee["left"|"right"]             # ee_link names; robot.entity.get_link(...)
robot.ee_pose(side) -> (pos[3], quat_wxyz[4])
GR100_OPEN=0.0  GR100_CLOSE=0.9  GR100_MIMIC=-1.0    # gripper scalars (mimic = -1*driven)

# robots/ik.py
TOOL_IN_EE_INV, tool_R_at_home(home_ee_R)            # tool(claw)-frame <-> ee_link; ~11cm offset

# skills/  (sim-agnostic; all wxyz)
grasp.world_long_axis / grasp.orientation_aware_grasp_quat / grasp.tilted_base_quat / grasp.transport_quats
grasp.GraspContext / grasp.grasp_quat_at / grasp.cquat / grasp.select_grasp_tilt / grasp.select_place_tilt
grasp.grasp_action_wps(ctx, i, gc, gq, open_g, close_g, app, lift, home_tool, home_tquat)   # the GRASP action
place.place_action_wps(i, lift_pose, carry_quat, bowl_xyz, papp, open_g, close_g, home_tool, home_tquat)  # PLACE
grasp_retry.run_grasp_retry(...)     # opt-in miss->retry (default off)
score.score_placement(...) -> (placed, metrics)  ;  score.through_wall(...)
penetration.PenetrationTracker / penetration.max_penetration / penetration.ABNORMAL_THRESH_M (=0.007)
executor.BatchExecutor(...).run(...)

# world/firefly_scene.py
TableLayout()  BOWL_HALF_H=0.02748  firm_rigid_options(noslip_iterations=...)  build_bowl(scene, xy, top_z, surface=...)

# registry/object_spec.py
REGISTRY["cube"|"apple"|"banana"|"pen"|"tennis_ball"|"book"]   # ObjectSpec: extents, mass, long_axis, grasp_dz,
                                                              # grasp_noslip, target_palette, native_texture, keypoints, ...
```

### Gotchas to carry into a new task

- **Genesis returns CUDA tensors** from `get_pos/get_quat/render`; wrap them in `np_(...)` at the sim↔skills
  boundary. **Quaternions are wxyz.**
- **Dual-arm dofs are INTERLEAVED** (left_joint_1=dof0, right_joint_1=dof1, …) — always go through
  `robot.arm[side]` / `robot.state14_idx`, never assume contiguous indices.
- **Concave containers need convex decomposition**, not SDF, or a corner on the rim tunnels.
- **Never drive a gripped object to a target inside a wall** (`kp=200` out-forces any collider) — release
  above + free-drop, with a **hold** waypoint to zero velocity first.
- **Nyx env-map memory** caps at ~50–60 2K maps — keep the per-env HDRIs at 1K (`hdr_pool()` already does this).
- **The textured YCB bowl USD segfaults Nyx** — load a clean visual mesh (`bowl_clean.obj`), not `bowl.usd`,
  for any task that renders a USD-textured container.
- **One GPU per process** — let Genesis batch the envs; do not parallelize across processes on one GPU.
- **Capture the lift pose at the lift PEAK** (last lift step), not the first, or `grasped`/`lift_cm` falsely
  read ~0.
