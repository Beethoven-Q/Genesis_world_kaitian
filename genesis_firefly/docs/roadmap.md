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
