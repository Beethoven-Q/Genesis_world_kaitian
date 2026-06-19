# Genesis firefly — ROADMAP & progress log

A living log of the phased plan + every important problem, change, and **why**. Keep it updated; commit when a
phase lands. Requirements live in [project_overview.md](project_overview.md) (blueprint) +
[domain_randomization.md](domain_randomization.md) (DR spec) + [agents.md](agents.md) (agent contracts).

> **HARD RULE for every agent:** maintain this file. When you fix a real problem, change a contract, or make a
> non-obvious decision, append a dated entry (problem → change → why → result). This is how we trace back.

## Phases (the framework build-out)
- **P1 — Smooth motion + cameras + backgrounds** *(DONE, verified)*: smooth/gentle motion path, wrist-cam
  gripper fix, 2K immersive backgrounds.
- **P0 — Repo reorg** *(next)*: `world/ · dr/ · skills/ · registry/ · tasks/ · io/ · runner/` + `.claude/
  agents/ + .claude/workbooks/`; dissolve `_core_vendored`, delete dead code, fix imports. Gate: pickplace parity.
- **P2 — Full-DR harness**: table **TEXTURE** pack (≥10 incl. florals), object color/size/type, **distractor
  objects** (2–3 random clutter, real physics, collision-free planning), all scopes A/B/C; per-build vs per-env;
  per-demo `DRPlan` → HDF5.
- **P3 — Build-batch orchestration**: B subprocess builds × E envs, half-left/half-right, merge shards; **/data3
  storage + output symlink**. → the **200-trial full-DR collection**.
- **P4 — Agency**: DR subagent + workbook; object-refiner + disturbance-agent stubs; the virtual-EE skill.
- **P5 — More objects + Line C**: solve apple / tennis-ball / banana / marker-pen pick-place; then **Line C
  dexterous-hand integration** (candidate for a dedicated worktree-isolated subagent).

## Emerging reusable skills (build + name as we go)
- `BatchExecutor` (`skills/executor.py`) — the ONE smooth motion path (densify + batch IK, gentle everywhere).
- `plan_pick_place` / `reachable_grasp_quat` / `score_pick_place` — pick-place + physical scoring.
- `virtual_ee` *(planned)* — treat an object feature (mug ring center+normal, screw tip, peg) as a virtual EE →
  threading / screwing / pegging.
- (more: grasp primitives, penetration metric, collision-free planning around distractors.)

## Progress log
- **2026-06-17/18 — B0–B8 reproduction.** Dual Firefly Y6 + GR100 imported to Genesis (interleaved dofs, MIT
  gains, ARMATURE=0.01, implicitfast integrator → wrist jitter gone). Good-mode collision (convex-DECOMPOSE
  robot @0.05 + bowl @0.04, firm Newton, self-collision). Pure-friction grasp (substeps=4, 3mm finger contact
  skin, gravity_compensation). Pick-place SOLVED: 0.00mm penetration; release-ABOVE-rim free-drop (never drive a
  held object to a target inside a wall). See [[lessons_genesis_nyx]] / memory for the full recipe.
- **2026-06-18/19 — Nyx photoreal + parallel collection.** Nyx path-traced PBR (not Madrona). SOMA livery baked
  as solid-color image textures + matte entity override (Nyx ignores PBR factors for URDFs; renders 1 material/
  GLB + first `<visual>` only → hstack atlas for link_2/3). Per-env immersive HDRI backgrounds (no ground plane).
  Fully-parallel single build (n_envs=N). 100-env: 100/100 grasp, 97/100 place.
- **2026-06-19 — Agent-native reframe + blueprint docs.** Owner: the deliverable is a fully autonomous agentic
  framework (solve tasks + scale god-mode sim data → sim-to-real VLA), not any single task. Wrote
  `project_overview.md` (blueprint), `domain_randomization.md` (complete DR spec), `agents.md` (agent contracts).
- **2026-06-19 — P1 fixes.** (a) **Transparent gripper in wrist cams** = Nyx near-plane clip (cams ~5cm from
  fingers, default near=0.1) → `near=0.01` on the wrist cams. (b) **2K backgrounds restored** for build-batches
  (1K was only a workaround for 100 maps/build); `valid_2k_pool()` drops the 1 malformed HDRI (95 source vs 94
  validated) that segfaulted Nyx. (c) **Smooth motion** = new `skills/executor.py::BatchExecutor` (densify EE
  waypoints → constant-speed + SLERP + smoothstep + gripper dwell; batch-IK; gentle everywhere, no fast
  free-space).
- **2026-06-19 — IK BRANCH-FLIP fix (the jerk).** Problem: arm snapped at pre-grasp/lift. Diagnosed (workflow:
  4 readers + 3 adversarial verifiers, source-verified against the Genesis IK kernel) NOT as full extension
  (reach is only 45–57% of max) but as an **IK branch flip**: `entity.inverse_kinematics` defaults
  `max_samples=50` → on a non-converged warm start it RANDOM-restarts up to 50× over the full joint range and
  returns a flipped branch → the executor PD-drives across the jump in one step. Change: in the batched `ik()`,
  `max_samples=1` (pin to the warm start) + `max_step_size=0.2` + `damping=0.05` + `max_solver_iters=30`. Why:
  removes the random resample → C0-continuous solve. Rejected the heavier "route through plan_pick_place +
  GenesisArmIK" path (GenesisArmIK is single-env → throws on batched N; frame double-offset). Gate added: per-step
  max active-arm `|dq|` (a flip = >1 rad spike). Result (N=20, validated-2K): **20/20 grasp+place, 0 through-wall,
  max|dq|=0.034 rad** — jerk gone. Also: `LIFT 0.20→0.18`, cube↔bowl clearance ≥0.17 for ~80% (≥0.125 floor for
  the hard ~20%).
- **2026-06-19 — go-home recording.** Append a smooth return-to-home waypoint at the end of every demo so the
  return is densified (gentle) and RECORDED (video + data) → the policy learns to return home when finished.
- **2026-06-19 — TABLE-TEXTURE DR (scope-A item, was flagged-missing).** *Problem:* the DR spec wanted per-build
  table textures (wood / steel / tablecloth albedo maps, ≥10 incl. bright florals), but the stage only had a
  per-build table **colour** — flat, no material variety. *Change:* (a) a 12-map pack in
  `assets/textures/tables/` (4 wood · 3 steel/metal · 5 tablecloth incl. red/blue gingham + red-white floral +
  checker + polka), built by `scripts/build_table_textures.py` (3 real RoboLab albedos downsampled to 1024², the
  rest procedural numpy/PIL); (b) `manipulation_stage.py`: `table_texture_pool()` globs the pack,
  `__init__` picks one per build with the stage RNG → `self.table_texture`, applied to **both** tables (flush at
  the seam → one continuous surface); (c) the collidable table **Boxes** keep physics, a thin **visual-only,
  no-collision `gs.morphs.Plane`** on each top carries the texture as
  `Plastic(diffuse_texture=ImageTexture(image_path=…))`. *Why a Plane, not a textured Box:* a `gs.morphs.Box`
  has **no UVs** → Nyx warns `Texture given but asset missing uv info` and renders a **garbled smear**; a Plane
  carries UVs and the Nyx exporter UV-handles it (verified by a side-by-side probe render). The object-table top
  Plane is batched (`batch_fixed_verts=True`) and the task calls `stage.set_otable_top_z(tabZ)` after the per-env
  height DR so the texture stays **flush** on the randomized table. Box edges tinted to the texture's mean colour
  so the table edge matches the top. **Friction untouched + stays per-env, decoupled from texture.** *Result:*
  texture VISIBLY + cleanly renders on the table top in third / side / **both wrist** cams (preview
  `output/temp/table_texture_preview.png` = 4 distinct textures; gingham four-view confirms wrist cams). Parity:
  `pickplace.py 8 13` → **8/8 grasped, 8/8 placed, 0 through-wall, max|dq|=0.027 rad** — collection unbroken. See
  `rendering_and_livery.md` §8 (Box-UV gotcha + the Plane `uvScale` tiling math) +
  `domain_randomization.md` (scope-A row now ✅).
