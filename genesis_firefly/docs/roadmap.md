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
- `penetration` (`skills/penetration.py`) — faithful batched solid-solid interpenetration monitor read straight
  from the solver contact buffer (`max_penetration` / `abnormal_penetration` / `PenetrationTracker`); the owner
  #1 collision gate that rejects any demo with abnormal penetration. *(built — see §3g + progress log)*
- (more: grasp primitives, collision-free planning around distractors.)

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
- **2026-06-19 — DISTRACTOR / CLUTTER OBJECTS (scope-B REQUIRED feature).** *Problem:* the DR spec requires every
  task to spawn 2–3 random irrelevant objects on the table (so the policy learns visual disambiguation + native
  collision-avoidance), but pickplace had only the cube+bowl — a clean single-object scene that teaches neither.
  *Change* (`tasks/pickplace.py`): `DISTRACTOR_POOL = {pen, banana, apple, tennis_ball, book}`;
  `choose_distractor_types` (per-build: K∈{2,3} distinct types, **≤1 large/long object** so they fit);
  `sample_distractor_poses` (per-env XY+yaw via a fine **anchor-grid + greedy mutually-spaced** assignment that
  keeps each object's whole footprint clear of the cube / bowl / cube→bowl carry tube / bowl→home return tube /
  near-seam strip / active-arm home, biased to the opposite-y open area); `spawn_distractors` +
  `_spawn_distractor_entity` (real collidable rigid bodies). Added `book` to the REGISTRY + extracted **Nyx-safe
  `*_clean.obj`** visual meshes for apple/banana/pen (the textured USDs **segfault Nyx**, same root cause as the
  bowl — `scripts/temp/extract_clean_objs.py`). *Why placement, not a planner change:* the executor is pure
  waypoint-IK with NO obstacle avoidance, so collision-freeness is achieved by **placing distractors out of the
  swept corridor** (rejection/greedy-grid) — exactly what the spec's rejection-sampling prescribes; this bakes
  obstacle-avoidance into the data for free without touching the (locked) motion path. *Two real bugs found +
  fixed:* (1) a long banana centred near the far edge **tipped off** → inset every bound by the object's
  circumscribed footprint radius (valid at any yaw); (2) a curved banana **slowly rolled ~3 cm** over the episode
  (intrinsic metastability, NOT an arm knock — proven by a monotonic per-phase displacement trace with the arm on
  the opposite side) → **lowered the distractor CoM** so it self-rights + rests stably (a single convex-hull
  collider + longer settle also help). *Result:* over seeds **6 / 17 / 23 / 42 / 103** (8 envs each, incl. K=3
  with banana): **8/8 grasped, 8/8 placed, 0 through-wall**, and the collision-free gate **max distractor XY
  displacement ≤ 1.2 cm, 100 % under 2 cm, none knocked** (`pickplace.py 8 17` → 8/8/8/0, max disp 0.18 cm).
  Offline placement verified over **1500 seeds**: 0 corridor violations, all on-table, no stacking. Preview:
  `output/temp/distractors_preview.png` (four-view, K=3 banana+apple+pen) + `distractors_third.png`. See
  `domain_randomization.md` (distractor subsection now ✅).
- **2026-06-19 — PENETRATION DETECTOR / GATE (owner #1 enforcement).** *Problem:* collision correctness was
  asserted but not **measured + enforced every demo**; abnormal interpenetration produces unphysical, harmful
  training data, so we need a faithful detector that flags & **rejects** any penetrating demo (never ship it).
  *Change:* new reusable `skills/penetration.py` — `max_penetration(scene, redetect=False)` returns per-env
  (N,) MAX solid-solid overlap depth (m) + the worst geom pair, `abnormal_penetration(scene, thresh_m)` flags
  envs over threshold, and `PenetrationTracker` folds the per-step buffer into a per-env **worst-ever** across
  the trajectory. *The faithful solver API:* it reads the solver's persistent contact buffer directly —
  `scene.rigid_solver.collider._collider_state.contact_data.{penetration, geom_a, geom_b}` + `n_contacts`,
  via `qd_to_torch(..., transpose=True, copy=False)`. `penetration` is metres, **positive = overlap** (sign
  confirmed at `collider/box_contact.py:91` + `constraint/solver.py:698`). *Why NOT `get_contacts`:*
  `collider.get_contacts` runs a torch `gather` over `contact_sort_idx` whose dtype is not int64 in this build →
  `RuntimeError: gather(): Expected dtype int64 for index` whenever pruning/sort is live (reproduced); the raw
  buffer read is the documented bypass and what the project memory calls "read penetration straight from the
  solver." *Redetect timing:* `redetect=True` runs `collider.clear()+detection()` to read the TRUE un-resolved
  overlap (a probe), `redetect=False` reads the post-`scene.step()` buffer (the `on_step` running max). *Hollow
  stays hollow:* the buffer only holds overlapping pairs, so a finger in the convex-decomposed bowl cavity emits
  no deep contact — only solid-solid counts. *Threshold:* `ABNORMAL_THRESH_M = 7 mm`, chosen empirically — the
  healthy-run per-env max sits at the firm-grasp **contact skin** (measured **1.2–2.5 mm** over 8 envs), and
  7 mm is ~2.8× above that, far below any tunnelling (32 mm at substeps=1). *Integration* (`tasks/pickplace.py`):
  a `PenetrationTracker` updates each `on_step`; after the run `placed = placed & ~penetrating` (a penetrating
  demo is **not** a clean success → dropped by `success_only` export); HDF5 attrs `max_penetration_mm` (float)
  + `penetrating` (bool) per demo; prints `[COLLECT] penetration: max=X.Xmm, abnormal=k/N`. The existing
  through-wall metric stays (task-specific) but this detector is the **authoritative** gate. *Two-sided gate
  (verified):* (1) healthy `pickplace.py 8 17` → **8/8 grasped, 8/8 placed, penetration max=2.5mm, abnormal=0/8**
  (no false positive; worst pair = the real `box↔gripper` grasp contact); (2) deliberate-overlap probe
  `scripts/temp/pen_probe.py` (two solid boxes spawned overlapping 2 cm) → detector reports **20.00 mm**, all
  envs flagged abnormal → **PASS** (it CATCHES). See `robot_collision_cameras.md` §3g.
- **2026-06-19 — BUILD-BATCH ORCHESTRATOR (`runner/orchestrate.py`).** *Problem:* `collect.py` runs only ONE
  build (one "look"); scaling to a large fully-DR dataset needs **B parallel subprocess builds × E envs** with B
  distinct looks (table-texture/colors/sizes/type/distractor-set bake per build; a 2nd `gs.Scene` in one process
  segfaults). *Change:* new `runner/orchestrate.py B E [seed0] [dataset]` that **reuses the collector verbatim**.
  (1) **GPU pool** — detect the pool from `nvidia-smi --query-gpu=index` (or a `GPUS=0,1` override) and run a
  worker pool of size = n_gpus; each worker pulls the next build off a queue and runs it as a fresh `collect.py`
  subprocess pinned with `CUDA_VISIBLE_DEVICES=<gpu>` + `DATA_DIR`/`OUT_DIR` shard dirs (**one sim process per
  GPU**), relaying the key `[COLLECT]` lines tagged `[b g]`. (2) **Shard → merge → symlink** — each build writes
  `/data3/genesis_fulldr/<dataset>/_shards/build_<b>/` (its own `demos.hdf5`+videos); after all builds, every
  shard's `/data/demo_<i>` group is **h5py-group-copied** (ALL datasets + ALL attrs preserved) into the merged
  `<dataset>/demos.hdf5` under a **globally renumbered** `demo_<g>` (g=0..B*E−1) with a `build` attr added; videos
  are copied+renumbered to `videos/cam_*/demo_<g>.mp4`. Shards are **kept** (failed merge recoverable). An
  `output/<dataset>` **symlink** → the /data3 dataset gives in-workspace preview. **/data3 writability is gated:
  FAIL LOUDLY (exit 3) — never silently write into the repo.** (3) **Summary** → `<dataset>/summary.json`
  (printed): requested vs merged demos, builds ok/failed, clean successes (placed AND not penetrating), grasp
  rate, total abnormal-penetration count, per-cam video counts, per-build looks. *Verified* (small correctness
  test, NOT a large run): `orchestrate.py 2 2 900 orch_test` on **2× A6000** → both builds ran **in parallel on
  GPU 0 & 1**, **2 distinct looks** (build0 distractors `banana,tennis_ball` / build1 `tennis_ball,banana,apple`,
  different HDRIs/arms), merged **demo_0..demo_3** each with intact `actions`/`states`/`ee_pose` + all attrs
  (success/arm/seed/hdr/has_distractors/distractors/max_penetration_mm/penetrating) + `build`, **4 matching
  videos per cam**, `output/orch_test` symlink resolving to /data3, `summary.json` written (4 demos, 2 clean
  successes, grasp 0.75, 0 abnormal). *Recommended 200-demo run on 2 GPUs:* **B=10 × E=20** (10 looks, ~16 min)
  or **B=20 × E=10** (20 looks, longer) — prefer more builds (more looks) while E amortizes the build/settle
  overhead; keep E≤45 for 2K HDRIs. See `runner/README.md` + `project_overview.md` §8.
- **2026-06-19 — DEGENERATE-TRAJECTORY BLOWUP (cube ejected off the table → a 10× build).** *Symptom:* in the
  200-trial run, builds with seeds **1001** and **1005** each had ONE env whose densified trajectory blew up to
  **T=10,769 steps** (10× the normal ~1075) with **max per-step |dq| = 0.936 rad** (normal ~0.03). Because the
  `BatchExecutor` pads ALL envs to the max length T, that 10×'d every env in those builds → a **~41-min build**
  (vs ~5 min) and at least one jerky demo. *Root cause* (diagnosed plan-only via `scripts/temp/diag_degen*.py`,
  which run the REAL `collect()` but monkeypatch `BatchExecutor.run`/`settle_home` to capture waypoints +
  per-step cube pose WITHOUT executing): at seed 1001 **env 7** spawned its cube only **7.0 cm** (centre-to-
  centre) from the bowl — INSIDE the bowl wall (cube circumradius 3.5 cm + bowl radius ~7.5 cm ≈ 11 cm needed).
  `sample_phys_dr`'s cube↔bowl rejection loop **exhausted its 60 tries and SILENTLY shipped the still-overlapping
  pose** (no fallback; ~6 % of `clr=0.17` bowls are even geometrically infeasible). The firm solver then **ejected
  the overlapping cube on settle step 0** (|v| 1.8 m/s, climbing) and — because the immersive scene has **NO ground
  plane** — it **free-fell off the table to z=−6 m** (|v|=11 m/s by step 139). That garbage settled pose feeds the
  grasp waypoints (`gc = cube.get_pos()`), so the `home→pre` segment spanned **6.5 m** → `densify` emitted
  `n = 6.5/0.13/0.01 ≈ 5021` steps → the whole build padded to ~10,700. (NOT the home-pose-displacement
  hypothesis — the home tool pose was sane; it was the CUBE waypoint.) *Fix — defense in depth (3 layers):*
  **(1) cap `densify`** (`skills/trajectory.py`): new `max_move_steps=600` kwarg HARD-caps any single segment's
  step count (6 s at dt=0.01, well above the ~540 a real 0.7 m move needs), threaded through `BatchExecutor`
  (`skills/executor.py`); a degenerate waypoint can no longer dominate T regardless of distance, normal moves
  unchanged. **(2) waypoint sanity guard + REJECT** (`tasks/pickplace.py`): after settle, flag any env whose
  settled cube is non-finite, drifted >5 cm in XY from its spawn, or off the table in z as `degenerate` → CLAMP
  its cube pose back on-table (sane motion) AND force `success=False` (a phantom-target demo never ships; HDF5
  attr `degenerate`). **(3) root-cause spawn fix** (`tasks/pickplace.py` `sample_phys_dr`): after the rejection
  loop, push any env still inside the 0.125 m hard floor RADIALLY OUT from the bowl to exactly that floor — a
  cube can no longer spawn intersecting the bowl by construction, so the ejection can't happen. *Verified:*
  `pickplace.py 20 1001` → **T=1051** (was 10,716), wall **232 s** (was ~2400), **|dq|=0.033 rad**, **20/20
  grasped, 20/20 placed**, abnormal-penetration **0/20**; regression `20 1005` → T=1058, 267 s, |dq|=0.037,
  20/20, 0/20; `20 7` → T=1041, 228 s, |dq|=0.033, 20/20, 0/20. The blowup is gone and smooth motion + parity
  are preserved. See `pickplace_task_and_dataformat.md` (densify cap + degenerate guard).
- **2026-06-19 — LEROBOT v2 EXPORT for pi0.5 (`dataio/convert_genesis_to_lerobot.py`).** *Goal:* turn the
  full-DR `cube_fulldr_v2` collection into a fine-tunable dataset. *Decision:* the vendored
  `dataio/lerobot_exporter.py` (RoboLab) emits **v3.0** and parses RoboLab's HDF5/`__camera`-suffixed video
  names — **wrong format and wrong layout** for the Genesis source + for pi0.5 (which consumes **v2.1**). So I
  wrote a **Genesis-specific converter** modeled on the proven RoboLab/OpenPI path
  (`openpi_hex/examples/hexarm/convert_robolab_demos_to_lerobot.py`): it reads state/action from the merged
  `demos.hdf5` `/data/demo_<i>` (`states/articulation/robot/joint_position` → `observation.state`, `actions` →
  `action`, both 14-D) and decodes the per-camera MP4s
  (`cam_side`→`observation.images.cam_high`, `cam_lw`→`cam_left_wrist`, `cam_rw`→`cam_right_wrist`), task
  `"put the cube in the bowl"`, fps=12 (probed), img 640×368 (probed — the source is 368-tall, not 360).
  *Interpreter:* the repo `.venv` has **no `lerobot`**; the proven path is the OpenPI venv
  (`/home/kaitianchao/Projects/openpi_hex/.venv`, `lerobot==0.1.0`, `CODEBASE_VERSION=v2.1`) — used that.
  *success_only gate:* ship a demo only if `success AND not penetrating AND not degenerate` → **200/200** (the
  v2 collection is already 100 % clean). *Output:* `/data3/genesis_fulldr/lerobot/genesis_cube_fulldr_v2/`
  (575 MB, off-repo) + `output/genesis_cube_fulldr_v2` preview symlink. *Verified:* `LeRobotDataset(...)` loads
  it — `total_episodes=200`, `total_frames=21460`, `fps=12`, `tasks={0:'put the cube in the bowl'}`; `ds[0]` and
  a mid-dataset sample (ep 99) both yield `observation.state (14,) f32`, `action (14,) f32`, three images
  `(3,368,640) f32`. info.json `features` lists exactly the 3 cams + named 14-D state/action. *Note:* lerobot
  0.1.0 encodes the videos as **AV1** (`video.codec: "av1"`); torchvision/pyav decode them fine (the loader
  decoded all 3 streams in verification) — same as the RoboLab pipeline produces. *Next:* clone
  `pi05_hexarm_bowl_lora` → `pi05_genesis_cube_lora` (set LeRobot `repo_id="genesis_cube_fulldr_v2"`; the 14-D
  HexArm transforms + cam names are unchanged) → `compute_norm_stats` → fine-tune pi0.5. Full how-to:
  `lerobot_export.md`.
- **2026-06-19 — Disturbance → failure-recovery (a HARNESS, not an LLM subagent).** Per-trial probability (~0.34,
  `DISTURB=0` to disable) gently shoves the target cube in-plane DURING the grasp (`skills/disturbance.py`,
  batched `set_dofs_velocity` on the cube free-joint x/y, calibrated to ~3-5cm miss, latched once) so the grasp
  misses; the god-mode solver DETECTS it (cube rose <3cm OR empty-close + cube far) and RECOVERS by staged
  replanning (rise→reopen→re-locate→re-grasp, ≤2 attempts, batched; clean envs hold) before place. The demo
  records failed-grasp+recovery+success = the training signal. Verified: DISTURB=0 reproduces parity (16/16,
  pen 0); DISTURB=0.34 detection EXACT (0 false-flags on 21 clean grasps), every detected failure recovered,
  ~75-86% disturbed envs recover+place, pen 0, motion smooth (max|dq|~0.10). Added a keyword-only `during_step`
  hook to BatchExecutor (additive). HDF5 attrs `disturbed/recovered/recovery_attempts`. Note: the staged refactor
  raised the base no-disturbance max|dq| 0.03→0.098 (still smooth) — a minor phase-boundary-continuity polish for
  later. Full spec: `disturbance_recovery.md`.
- **2026-06-19 — DR-STRATEGIST agent layer (MVP) + sweep probe.** *Goal* (owner): make full DR more automatic —
  a specialized subagent + harness that pushes DR ranges to the MAX representative extent, models cross-axis
  couplings, and builds experience over runs. *Built* (P4 first slice): **(1)** `.claude/agents/dr-strategist.md`
  — a real Claude Code subagent implementing the `agents.md` contract: selects scope-B fields, recommends
  realistic MAX-extent ranges respecting couplings (clearance↔pose, reach↔arm-side, mass/friction↔grasp) + the
  anti-coupling pose rules, and after a run diagnoses which DR fields drove failures from the per-demo HDF5 attrs
  and APPENDS to its workbook. Tools: Read/Grep/Glob + Edit the workbook ONLY + Bash to run the small sweep
  probe. Boundaries: it ADVISES (proposes range edits the main agent applies), NEVER auto-mutates global ranges,
  NEVER edits robot/task/stage code, NEVER runs a full collection. **(2)** `.claude/workbooks/dr_workbook.md` —
  seeded from the cube task + v2 (200/200 clean): a current-best-ranges table (per axis: range + confidence),
  an append-only dated experience log (incl. the clearance-floor root-cause carried from this roadmap), and a
  known-hard-corners list (tight clearance, far-reach, degenerate-spawn). **(3)** `genesis_firefly/dr/` package
  (`__init__.py` + `sweep.py`) — the agent-native explore→measure probe: runs the EXISTING collector (reused
  verbatim, fresh subprocess like `orchestrate.py` — Nyx is single-shot) on a SMALL batch (E≤24, default 12)
  for a candidate config and reports success rate + an achieved-DIVERSITY measure (std + bbox extent of the
  cube/bowl pose, tableZ, yaw, reach) + WHERE failures cluster (per-axis z-score). *The minimal range hook:*
  `sample_phys_dr` now reads `DR_POSE_SCALE`/`DR_MASS_SCALE`/`DR_FRIC_SCALE` (default **1.0**) that scale only
  the per-env range HALF-WIDTHS about their fixed centres — never moving a centre, never relaxing the cube↔bowl
  0.125 m floor, never touching the anti-coupling sgn/arm logic. Also added the per-demo **DR plan** to the HDF5
  (`dr_cubx/cuby/bowx/bowy/tabZ/yaw/mass_shift/clr/reach` + `dr_*_scale`) so the strategist's failure diagnosis
  is defensible (the `DRPlan→attrs["dr"]` of `domain_randomization.md`, made real). *Verified:* **(a)** DEFAULT
  reproducibility — `DR_*_SCALE=1 ... pickplace.py 8 7` → **8/8 grasped, 8/8 placed, through-wall 0/8,
  penetration max 2.9 mm abnormal 0/8, max|dq|=0.098 rad** (the hooks didn't change default behaviour). **(b)**
  the sweep tool runs end-to-end and prints a real success-rate + diversity + clustering report. **(c)** the
  explore→measure→record loop demonstrated on ONE widened axis (`DR_POSE_SCALE=1.3`, E=12, seed 7) vs default —
  finding + delta recorded as a dated workbook entry. *Deferred (future, not MVP):* the fully-declarative `dr/`
  config (scopes.py/object_dr.py/sampler.py/apply.py/plan.py) replacing the hand-written `sample_phys_dr` is a
  large refactor; the MVP advises edits to the existing ranges instead. See `agents.md` (DR strategist now
  [BUILT — MVP]) + the workbook.
- **2026-06-19 — OBJECT-REFINER agent + harness (MVP).** *Goal* (owner): the harness + agent that turns a RAW
  object into a SIM-READY asset with a CORRECT collision model (the #1 priority), reusing our verifiers,
  grounded in the ASSET2SIM paper. *Built:* **(1)** `genesis_firefly/registry/refine.py` — the reusable refine
  HARNESS: **AUGMENT** (`analyze_mesh`: trimesh is_watertight / voxel-fill solid volume / area / extents / CoM /
  PCA long-axis / hollowness / suggested-scale — pure, no engine) -> **INFER** (`infer_physics`: mass =
  solid_volume x a per-category density from `CATEGORY_PRIORS` (ceramic/glass/wood/plastic/metal/rubber/fruit) +
  friction + restitution) -> **COLLISION** (`extract_visual_obj`: the GOOD collider = convex DECOMPOSITION via
  coacd `threshold=0.04` — the SAME bowl recipe so hollow stays hollow, NEVER a single filling hull — + a
  Nyx-safe clean OBJ, since Nyx segfaults on the textured USD) -> **VERIFY** (`verify_in_sim`: reuses
  `skills/penetration.py` for abnormal ~= 0 at rest + the paper's **initialization-stability test** = settle
  with ZERO actions ~1.2 s then measure root drift `D_pos<=3mm`/`D_ori<=0.02rad`/no-explosion + a thin-cylinder
  **hollow probe** threaded through a declared feature). Returns a structured `RefineReport`. The
  engine-touching stages run in FRESH subprocesses (one `gs.Scene`/process — a 2nd build segfaults). **(2)**
  `.claude/agents/object-refiner.md` — the Claude Code subagent (use CLAUDE) implementing the contract: SOURCE
  (Objaverse/YCB rigid; PartNet-Mobility articulated — note WHERE) -> run the harness -> VERIFY -> label
  KEYPOINTS -> emit the `ObjectSpec`; edits assets/registry for its object only, REUSES the penetration gate,
  never touches the locked stage/robot/IK/DR/collectors. **(3)** `ObjectSpec.keypoints` field added (LOCAL-frame
  `name -> {center, normal, [radius]}` — the virtual-EE frames). **(4)** the workbook
  `.claude/workbooks/object_refiner_workbook.md` (category densities + a per-object log). *Demonstrated
  END-TO-END on the YCB mug* (`registry/demo_mug.py`; a hollow handle ring + a hollow cup mouth):
  augment extents (0.117,0.093,0.081) m, solid 237 cm^3, hollowness 0.55 -> infer ceramic 2400 kg/m^3 -> mass
  0.569 kg -> collision **34 convex hulls** (ring + cavity stay OPEN) + `ycb/mug_clean.obj` -> **VERIFY all
  PASS**: penetration abnormal **0/1** (max 0.04 mm), init-stability **D_pos 0.03 mm / D_ori 0.0012 rad** no
  explosion, hollow-probe object<->branch overlap **0.00 mm** (a 4 mm branch through the ~10.5 mm handle hole) ->
  **SIM-READY: YES**. The `mug` `ObjectSpec` + `handle_ring`/`cup_opening` keypoints were added to the registry;
  a verification render (mug resting stable, cup mouth open, a brown branch threaded through the OPEN handle
  ring) saved to `output/temp/mug_refine_verify.png`. *Geometry-probe lesson:* the FIRST handle-hole guess had
  the X sign wrong (a vert-count shell mis-attributed the cup wall as the handle, probe overlap 6.2 mm); fixing
  it via an X-Z silhouette + an **encircled-hole search** (require >=7/8 angular sectors occupied — a
  corner-of-mesh max-clearance point is a false positive) put the keypoint on the TRUE open hole -> 0.00 mm. Mass
  on the heavy side (the voxel-fill counts the whole wall+interior volume; a real EMPTY mug ~0.35 kg) — noted in
  the workbook. *Deferred (future):* the articulated PartNet-Mobility joint-oscillation gate + a Nyx multi-view
  augment render (the MVP did the rigid-hollow case). See `object_refiner.md` + `agents.md`
  (object-refiner now [BUILT — MVP]).

## Planned refinements + forward plan (2026-06-20, owner directives)
**A. Disturbance v2 — smoother + collaborative timing (failure-recover AND moving-object "chase" data).**
- *Smoothness:* on a failed grasp the arm must rise only a SMALL amount to clear the view + retry — NOT lift high
  to near-singularity (the current ~jerk). Detect the miss right at the gripper-close-on-nothing. All gentle.
- *Collaborative timing:* fire the gentle shove at a RANDOM time during the APPROACH, then inform the task solver
  after a reasonable sense-DELAY. If the solver is informed BEFORE the gripper closes → it GIVES UP the current
  grasp, rises a little, and SMOOTHLY PIVOTS to the cube's new pose (gripper gently "chases" the moving object) →
  *chase-a-moving-object* data. If informed AFTER close → too late, grasp fails → rise-a-bit + retry new pose →
  *failure→recover→replan* data. One mechanism, both data modes.
- *Future:* a rotating Lazy-Susan with the target on it → the solver reads the real-time pose + chases it →
  moving-object grasp data.
**B. Object-refiner v2 — multi-view collider check + verified keypoint/part labeling (virtual, physics-less).**
- *Multi-perspective COLLIDER render* (the final collision check): after collision is correct, render the
  COLLIDER (vis_mode=collision) from several views for the AGENT to inspect + confirm (hollow stays hollow, solid
  never penetrates) — like the arm/camera collider render. SAVE these images WITH the asset so the owner can check.
- *Keypoint/part labeling:* the refiner DETECTS + labels meaningful keypoints/parts from geometry (mug handle-ring
  center+normal, cup-top opening; towel 4 corners; etc.) so the task solver never has to hunt for them (the
  RoboLab time-sink — once mistook the cup opening for the ring). Bake each as a VIRTUAL annotation: a massless,
  COLLISION-LESS, INVISIBLE labeled frame/point (a virtual link for URDFs) the task solver reads from sim info.
  MUST NOT perturb the object's physics/topology. For PartNet-Mobility: give raw link/joint indices (link1/2/3)
  correct SEMANTIC names — VERIFIED by geometry/render, NEVER hallucinated (a wrong name misleads the solver).
**C. Forward plan (after A+B land clean):** use the agentic system to SOLVE other-object pick-place (apple/
banana/pen/tennis-ball; round-object grasp = the caging challenge) → after owner exam, collect their full-DR data
→ then the **virtual-EE skill + mug-hang**. **Dexterous hand:** a NEW git BRANCH (it substitutes the gripper),
clean + safe; goal = pick-place with dex-hand+arm, then throw-and-catch a tennis ball in a parabola. Stay
agent-native; subagents for context; rigorous, no hallucination.
- **2026-06-20 — Refinements A+B DONE & verified.** (A) Disturbance v2 (`a398a43`): random approach-timing shove
  + sense-delay → before-close **smooth CHASE** to the moved cube (moving-object data) vs after-close **gentle
  RETRY** (recovery data); jerk fixed (RETRY_RISE 0.20→0.10m + rise→reorient→descend decomposition → retry
  max|dq| 0.198→0.068, global all-phase ≤0.098). Empty-close claw-claw contact allow-listed in the penetration
  gate (same-gripper pair only; all else still counts). HDF5 attrs disturbed/disturb_phase/disturb_outcome/
  recovery_attempts. (B) Object-refiner v2 (`6f2e59b`): multi-view COLLIDER renders saved with each asset (read
  the hulls directly; mug ring open + mouth open + body solid); keypoints/parts as VERIFIED virtual massless/
  collision-less/invisible labels (mug ring+opening, proven 0 physics perturbation); PartNet semantic naming
  verified live (bottle link_0→cap, link_1→bottle_body vs semantics.txt) + a synthetic-stapler unit test.
- **NEXT (forward plan §C):** generalize the pick-place task to a CONFIGURABLE TARGET object → solve banana/pen
  (elongated, the orientation-aware grasp handles them) + ATTEMPT apple/tennis-ball (round = the sphere-ejection
  caging challenge); then full-DR collect (owner exam); then virtual-EE + mug-hang; then the dex-hand BRANCH.
- **2026-06-20 — Configurable-target pick-place DONE; only the cube grasps cleanly with the firm GR100 jaw.**
  *Change:* the **same `tasks/pickplace.py`** now picks ANY registry object via the **`TARGET` env var** (default
  `cube`). `spawn_target` builds the target with a faithful grasp collider (decomposition, or a single convex hull
  for a near-convex body); the distractor pool **excludes the target type**; new `ObjectSpec` grasp fields
  (`grasp_center_offset_local`, `grasp_close`, `grasp_single_hull`, `grasp_decompose_err`); the spawn-clear floor
  scales with the target footprint; scoring uses `place_xy_tol_cm`. Env debug knobs `FAST`/`GRASP_DZ`/`CLOSE_G`/
  `TGT_FRIC`/`TGT_SINGLE_HULL`/`DETECT_DEBUG`/`PEN_TRACE`. The grasp-failure detector uses the **grasp centre**
  (not the raw root) so an offset-grasped banana isn't mis-flagged. Target velocity zeroed at settle end (anti-creep).
  *Regression (HARD gate):* `TARGET=cube pickplace.py 8 7` → **8/8 grasp+place, 0 pen, max|dq|=0.068** — byte-for-byte
  the pre-change cube (`spawn_target`'s cuboid branch is the exact old Box; clearance/rng draws unchanged).
  *Per-object (real DISTURB=0 runs, E≤20):* **cube 8/8 ✅**. **banana ⚠️ partial** (E12: 7/12 grasp, **6/12 placed**,
  2 over-pen): the body-centre offset + single-hull land the claws on the fruit, but the 2-finger pinch on the
  **curved 3.8cm girth misses ~50%** of first attempts → retries → the retry re-grasp drives **>7mm** into the
  rounded body (first grasp alone is a clean ~6.4mm). **pen ❌ 0/20**: the descending open claws **sweep the light
  thin pen ~12cm aside** → empty close; a deeper grasp NaNs the solver near the table. **book ❌ 0/20 (geometric)**:
  both flat dims (18,13cm) exceed the ~6cm jaw and the 3cm thickness is vertical when flat. **apple ❌ 0/20** /
  **tennis_ball ❌ 0/20**: round → the firm pinch **ejects** the sphere (tennis ball launched **km** away — the
  textbook sphere-squirt). *Root cause:* the firm GR100 parallel-jaw pinch is tuned for the cube's flat,
  jaw-matched faces; rounded/thin/oversized bodies skid, eject, or over-penetrate. More grasp_dz/friction tuning
  made it WORSE. *Recommendation:* the rounded objects (apple/tennis/banana) need an **under-actuated/caging or
  multi-finger hand** → deferred to the **dexterous-hand branch**; the thin pen + the rounded banana would also be
  helped by a **compliant / contact-stopping close** (close-to-first-contact-then-hold instead of PD-ramping to a
  fixed firm q=0.9) — a gripper-control change, out of scope for the per-object task tuning. Demos:
  `output/temp/pickplace_cube_demo.mp4`, `output/temp/pickplace_banana_demo.mp4`.

- **2026-06-20 — NO-WAIT ROOT RE-ARCHITECTURE + clean reset (the foundation fix).** *Problem:* the owner kept
  seeing the arm **idle in the air after the grasp** and trials waiting on each other. *Root cause:* the staged
  `run_phase` A1/A2/**B**/C structure (added with disturbance-v2) is a per-phase BARRIER — in the B-retry loop
  every successful env HELD its lifted cube through the slowest env's retries (≈ the ~5 s mid-air idle), and each
  phase padded all envs to its own max. A grasp-solving subagent, not knowing the *hold* was the bug, then built
  a pile of machinery to fight the symptom (cradle-depth, `carry_keep_grasp_quat`, slow-close dwell). *Change:*
  (1) **reverted** all that uncommitted churn + swept the debug junk (clean reset; the documented rules kept);
  (2) **re-architected** `tasks/pickplace.py`: the clean default (`DISTURB=0`) is now ONE continuous per-env
  trajectory `home→pre→at→close→lift→carry→lower→release→home` in a single `run_phase` — **no barrier**; the
  disturbance path is a per-env seg1/seg2/seg3 where a held env runs `place→home` concurrently with another env's
  recovery (no lifted-hold); (3) **per-env natural termination** — the writer trims each env's idle-home tail →
  **variable-length demos** (owner: "different lengths are natural"); (4) `DISTURB` now defaults to **0**
  (disturbance is opt-in augmentation). *Result (real runs):* clean cube **8/8 grasp+place, 0 pen, max|dq|≈0.03**,
  demo lengths **104–110** (variable, no mid-air hold); disturbance `DISTURB=0.5` **8/8 placed, 0 pen**, demo
  lengths **140–214** — the held envs terminate ~74 recframes before the recovering ones, *each at its own home*
  = cross-env independence proven in the data. Docs: [disturbance_recovery.md](disturbance_recovery.md) revised.
  *Next (owner steer):* re-solve **apple / tennis / banana / pen** on this clean foundation with a SIMPLE firm
  top-down grasp — the prior "needs a dex hand" conclusion was likely the broken-foundation symptom; the owner is
  confident these are easy when the foundation is correct (RoboLab found apple the easiest). Re-evaluate from
  scratch before adding any grasp machinery.
