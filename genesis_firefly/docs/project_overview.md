# Project overview — the agent-native sim-to-real data framework (the big blueprint)

> **Read this first.** This is the single source of truth for *what we are building and why*. Module docs
> ([manipulation_stage](manipulation_stage.md), [domain_randomization](domain_randomization.md),
> [agents](agents.md), [rendering_and_livery](rendering_and_livery.md),
> [robot_collision_cameras](robot_collision_cameras.md), [pickplace_task_and_dataformat](pickplace_task_and_dataformat.md),
> [lessons_genesis_nyx](lessons_genesis_nyx.md)) go deeper on each piece. Nothing here is dropped for brevity.

---

## 0. Mission (the one thing that matters)
Solving any single task is **not** the goal. The goal — and our most valuable contribution — is a **fully
autonomous, agent-native framework** that:
1. **solves** manipulation tasks in simulation with *god-mode* scripted policies that exploit privileged
   ground-truth sim state + analytic IK,
2. **scales** that into large, clean, fully-domain-randomized, photoreal datasets, fully in parallel,
3. feeds those datasets into **VLA fine-tuning (pi0.5)** for **sim-to-real** transfer,
4. and gets **better over time** — every harness and subagent accumulates experience in a *workbook* and trends
   toward mastery and full autonomy.

Future tasks reuse the infrastructure with near-zero plumbing. A new task is objects + a skill + a scorer; the
robot, cameras, rendering, backgrounds, collision, IK, DR, and the parallel collection harness are all reused.
This is the same bet as fully-sim systems like **MolmoBot**: a robust sim script + enough domain randomization
→ large clean policy datasets *before* attempting sim-to-real. The cube→bowl pick-place is only the reference
instantiation.

This Genesis line (Line B) reproduces our RoboLab/Isaac dual **Firefly Y6 + GR100** stack in the **Genesis**
simulator with **RTX-grade Nyx** photoreal rendering, then generalizes it into this framework.

---

## 0b. Working conventions (every agent, every task — non-negotiable)
- **COLLISION CORRECTNESS IS FIRST-CLASS (priority #1).** A correct collision model + ZERO abnormal
  interpenetration is the foundation everything else stands on — abnormal collision/penetration produces weird
  behavior and **harmful, unphysical data** that poisons training. Rules: (1) what cannot happen in reality must
  not happen in sim — two solids never interpenetrate under any grip force; (2) a **faithful, accurate
  penetration detector** monitors + ENFORCES this every demo (flag/reject any demo with abnormal penetration —
  never ship it); (3) **good collision = hollow stays hollow**: hollow features (mug-handle ring, cup/tip
  opening, bottle mouth) must have NO collision filling them (convex-**DECOMPOSITION**, not a single hull) — only
  then can a ring thread onto a branch or a cap seat in an opening; (4) when ANYTHING is weird/buggy, **suspect
  the collision model first** and check it. (5) Every NEW object gets a verified good collision model BEFORE use
  (solid never penetrates, hollow stays hollow) — see the object-refiner agent ([agents.md](agents.md)) +
  [robot_collision_cameras.md](robot_collision_cameras.md) + [lessons_genesis_nyx.md](lessons_genesis_nyx.md).
  (Owner directive, re-emphasized 2026-06-19.)
- **GRASPS ARE ALWAYS FIRM / FORCEFUL — never a loose close (HARD RULE, owner 2026-06-20).** The gripper closes
  WITH FULL FORCE; the object's COLLISION naturally stops the claws at the object's width; the firm PD MAINTAINS
  force against the object → a firm, strong grasp (this is how the cube grasp works). NEVER "detect contact and
  stop/relax the force" — that gives a loose grasp. When a round/thin object EJECTS, the bug is **grasp DEPTH /
  caging**, NOT the force: the default offset put the object at the curved GR100 claws' TIPS (uncaged → the firm
  close squirts it out); the fix is a DEEPER per-object grasp so the object sits in the claws' CRADLE (~Z_ee
  −2 cm) where the firm close CAGES it (verified: apple held + lifted, penetration ~2 mm). Fix the depth/cage so
  the FIRM grasp holds — do not go loose.
- **EACH TRIAL IS FULLY INDEPENDENT — NO CROSS-ENV WAITING (HARD RULE, owner 2026-06-20).** An env must NEVER
  idle/hold waiting for other envs to finish (not in sim, not in data). Each env runs its OWN continuous
  trajectory pick→place→home (or pick→[recovery]→place→home) to completion. Do NOT use a STAGED global barrier
  (e.g. "all envs finish the pick phase, then all place") — that makes early finishers hold the lifted object in
  the air = forbidden idle. Build each env's FULL trajectory and run it in ONE pass (tail-padded only); for the
  disturbance case, the chase-vs-fail outcome is fixed at build time (the fire-band), so each env's full path is
  known up front — no runtime barrier needed. This is NOT a record/trim issue; the WAIT LOGIC itself must not exist.
- **Maintain `docs/roadmap.md`** — a living problem→change→why→result log. Append a dated entry whenever you fix
  a real problem, change a contract, or make a non-obvious decision. This is how we trace back what we did.
- **2×2 four-view preview tile** for every collection (see §7). **Storage:** smoke → `output/`; full collections
  → `/data3/<dataset>/` + an `output/` symlink (see §7).
- **Distractor objects every task** — spawn 2–3 random irrelevant objects with real physics; plan collision-free
  around them (see [domain_randomization.md](domain_randomization.md)).
- **Full DR every trial** (scopes A/B/C, half-left/half-right arm) — never ship a "DR" run that forgot a scope
  (e.g. table texture). Keep everything organized, clean, elegant, agent-native.
- **Every task ends with a smooth GO-HOME stage** — after the task succeeds, the arm(s) return to home gently
  (densified, enough waypoints), RECORDED in both the data and the video. Applies to **all** tasks, single-arm
  AND dual-arm, so the policy learns to return home when finished. (Owner directive, 2026-06-19.)
- **Use subagents to keep the main context lean.** Delegate well-scoped mechanical/build tasks (repo reorg,
  feature implementation, broad investigations) to focused subagents — each with a precise recipe, a **hard
  verification gate** (compile + a real parity/render run, paste real output), and a **docs-update requirement**.
  The main agent reviews the diff + the gate result, then commits/pushes. This conserves context and keeps every
  change documented. (Owner directive, 2026-06-19.)

## 1. The end-to-end sim-to-real pipeline (the loop the framework automates)
Adapted from the proven RoboLab native pi0.5 pipeline (`RoboLab/docs/hex_pi05_native_pipeline.md`) — the big
idea is shared; only the simulator (Genesis) and renderer (Nyx) differ:

```
(1) god-mode scripted solver  ──>  (2) FULL DR + photoreal + parallel collection
        uses sim ground truth                 every trial randomized (scopes A/B/C)
        + analytic/native IK                          │
                                                       v
(6) deploy + evaluate  <──  (5) fine-tune pi0.5  <──  (4) convert to LeRobot v2  <──  (3) filter by PHYSICAL
     native client, sensor-only      (LoRA / full)        (success_only=True)            success (not command
     observation contract                                                                success)
        │                                                                                       
        └────────────────────────────  (7) iterate: fix at the real failure layer, widen DR, scale ──────────┘
```

**The separation principle (hard invariant).** The collector is *privileged* — it may use ground-truth poses,
analytic IK, god-mode resets. The learned policy receives ONLY the **observation contract**: 3 RGB cameras +
14-D robot state + a language instruction. This is what makes the data sim-to-real-able and the policy
deployable on the real robot (where SODA's analytic IK runs natively, but the policy stays sensor-only).

**Physical success filtering.** Success is judged by *physical outcome* (object lifted ≥ threshold, came to rest
in the bowl, clear of the gripper), NEVER by "the IK/motion command succeeded." A command can succeed while the
object fails the task. Failed trials are still valuable (debugging, and — via the disturbance agent — explicit
failure-recovery data), but the first fine-tune dataset is `success_only`.

**The task-solving loop** (what the main agent masters, per task):
1. build the sim scene with real assets; 2. write a privileged scripted solver using all useful sim state;
3. add strict physical success detection; 4. deterministic smoke tests with videos; 5. add DR only after
deterministic success; 6. small randomized batches, inspect failures; 7. fix at the *real* failure layer
(geometry / IK / gripper semantics / contact / transport / scoring); 8. scale collection in parts; 9. convert
to LeRobot; 10. fine-tune + evaluate; loop. The framework turns each of these steps into a reusable harness or
subagent so the agent gets fluent and eventually autonomous.

---

## 2. Architecture: SETUP vs TASK
```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ SETUP — the manipulation stage (you NEVER re-tune this)                                │
│   robot (dual Firefly Y6 + GR100, baked SOMA livery, convex-DECOMPOSED "good-mode"     │
│          collision)  ·  2 collidable tables  ·  3 photoreal cameras (2 wrist D405 +     │
│          1 side D435i) + visible side rig  ·  Nyx path-traced PBR  ·  per-env immersive │
│          HDRI background + light  ·  dexterity-aware IK  ·  smooth motion executor  ·    │
│          fully-parallel build                                                          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ FULL-DR HARNESS (+ DR subagent + workbook) — AUTOMATIC for every task; see              │
│   domain_randomization.md.  scope A (scene) + scope C (visual) apply for free; scope B   │
│   (object/task) per task.                                                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ SKILLS (reusable, pure-numpy)  grasp · pick_place · trajectory + smooth executor ·       │
│   virtual_ee (re-frame a feature as the EE → threading/screwing/pegging) · penetration   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ TASK (THIN, ~70 lines)  objects + a skill + a scorer + which scope-B fields apply.       │
│   Never touches robot / cameras / rendering / background / collision / IK / DR engine.   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The harnesses the main agent calls
The main agent never re-derives infrastructure; it *calls* harnesses:

- **ManipulationStage harness** (`world/manipulation_stage.py`): one object owns the robot (livery + good-mode
  collision), the 2 tables, the side-camera rig, the 3 policy cameras + 1 third-person camera, the Nyx photoreal
  renderer, per-env immersive HDRI background + light, and the parallel build. Methods: `.build()`,
  `.settle_home()`, `.render()`. The task just adds its objects and reads cameras. **"Set up the scene/robot/
  cameras/rendering/IK correctly" is a solved, reused call — not per-task work.**
- **Full-DR harness** (`dr/`, + the DR subagent): encodes the *complete* DR spec once and applies it. See §4.
- **Dexterity-aware IK** (`robots/ik.py` + `skills/pick_place.py`): Genesis-native sub-mm IK targeting the tool
  frame, wrapped by `reachable_grasp_quat`/`reachable_place_quat` (prefer top-down, relax to the smallest tilt
  that keeps the arm IK-solvable **and dexterous** — away from full extension/singularity). This is what keeps
  motion accurate and jerk-free near the workspace boundary.
- **Smooth motion executor** (`skills/executor.py`): the ONE motion path — densify sparse EE waypoints into
  constant-Cartesian-speed, SLERP'd, smoothstep-eased motion with gripper dwell ramps; batch-IK per step; the
  unused arm holds home. Gentle and slow *everywhere* (approach, grasp, transport, place) — the RoboLab motion
  we always reproduce. No fast free-space rush.

---

## 4. The Full-DR harness + the DR subagent (the agency core)
**The complete DR spec lives in [domain_randomization.md](domain_randomization.md)** (scope A scene, scope B
object/task, scope C visual — every field, range, and per-build-vs-per-env variability, plus the pose
anti-coupling rules). Summary of the design:
- Scopes **A (manipulation-stage)** and **C (visual background)** are **shared across all tasks** and applied
  automatically. Scope **B (object/task)** is per-object/per-task.
- The hard renderer constraint (Nyx bakes color/texture/size at build) → **build-batches**: B parallel
  subprocess builds (each one {table-texture, object-colors, object-sizes, object-type, table-size, side-cam
  pose}) × E envs (each env: pose, mass, friction, damping/stiffness, fill, table-height, HDRI, light). Every
  trial is fully randomized while staying parallel.

**The DR subagent (`dr-strategist`)** is the professional that the main agent delegates randomization to:
- it **reads the full DR requirement set**, **selects the fields relevant to the current task** (scopes A+C
  always; scope B by the object/task), and recommends *professional, realistic* ranges (banana yellow/green, not
  blue; apple ±10%, not watermelon-sized; pose wide but arm/task/inter-object permitted);
- it owns a **workbook** (`.claude/workbooks/dr_workbook.md`): per-(task, object) current-best ranges +
  confidence, an append-only dated experience log, and a "known hard corners (accept, label, keep)" list. After
  each run it diagnoses which DR fields drove failures (the per-demo `DRPlan` in the HDF5 makes this defensible)
  and **appends what it learned** — so it gets more fluent and professional over time;
- it proposes range edits (human/main-agent applied); it never runs sims or silently mutates global registries.

See [agents.md](agents.md) for the exact agent contracts.

---

## 5. Near-future harnesses / subagents (designed now, built as we go)
The framework is **extensible** — new harnesses/agents slot in. Planned next:

- **Object-refiner agent** — makes an object *physically and visually real*: refine its URDF (geometry,
  materials), **label trackable keypoints** (e.g. the mug handle ring's **center + normal**, a cap's axis, a
  drawer handle) into `ObjectSpec.keypoints` for skills like the virtual-EE, and run a **collision audit**:
  hollow parts stay hollow (convex-**DECOMPOSITION**, not a single filling hull), solid parts never interpenetrate
  under firm grip. Output: a realistic, keypoint-annotated, collision-correct object the registry can use.
- **Disturbance agent** — injects **failure cases during the main agent's execution** so the data contains
  **failure-and-recovery** modes (which makes the trained policy robust): nudge the object as the arm reaches to
  grasp (did the grasp succeed? regrasp if missed), knock the object over, shift the target mid-transport. The
  main agent must *detect* the failure from privileged state and *recover*. A `DisturbanceSpec` is injected at a
  scheduled step in the executor.
- **More to come** — e.g. a task-authoring agent that masters the §1 loop end-to-end, a camera/viewpoint agent,
  a scene-composition agent. Each new agent follows the same pattern: a narrow contract + a workbook that
  accumulates experience → trends to autonomy.

---

## 6. Reusable skills + the virtual-EE abstraction
Skills are pure-numpy and object-agnostic, promoted from RoboLab: `grasp` (orientation-aware grasp quats,
transport quats, tilted-base), `pick_place` (`plan_pick_place`, `reachable_grasp/place_quat`, `score_pick_place`),
`trajectory`+`executor` (the smooth motion), `penetration` (wall-penetration metric).

The **main agent develops its own reusable skills**. The flagship abstraction is the **virtual end-effector**
(`virtual_ee.py`): declare that some **feature** of an object is the frame the planner controls — e.g. the
**mug-handle ring's center + normal**. Then "thread the ring onto a hanger branch" reuses `plan_pick_place` + the
smooth executor with the ring as the controlled EE. The *same* skill generalizes to **screwing in screws,
pegging-in-hole, key-in-lock** — any "align a feature frame to a target frame and insert." Keypoints come from
`ObjectSpec.keypoints` (populated by the object-refiner agent). Skills, once proven, are extracted and reused
across tasks.

---

## 7. Data contract (what every demo writes)
- **§4 HDF5 schema** (a fixed contract; the exporter is reused unchanged): per-demo `actions` [T,14], `states/
  articulation/robot/{joint_position,joint_velocity}` [T,14], `ee_pose/{position,orientation}`, and attrs
  (`num_samples`, `success`, `seed`, `arm`, `hdr`, …). The **14-D layout** is `[L_j1..L_j6, L_grip, R_j1..R_j6,
  R_grip]`.
- **3 RGB streams** (the sensor-only policy input): side D435i + left/right wrist D405. LeRobot/OpenPI names:
  `observation.images.cam_high ← side`, `cam_left_wrist ← left wrist`, `cam_right_wrist ← right wrist`.
- **Per-demo `DRPlan`** (frozen JSON of every sampled DR value) in the demo attrs → fully traceable for
  sim-to-real *and* the substrate the DR subagent learns from.
- **Physical success** filtering (§1). **LeRobot v2** export (`io/lerobot_exporter.py`, `success_only=True`) →
  pi0.5 fine-tune.
- **2×2 four-view preview tile (REQUIRED for every collection):** every run writes a **2×2 tile video** —
  third-person · side(D435i) · left-wrist(D405) · right-wrist(D405) — for a sampled set of demos, plus a √N
  third-person grid, so a human can eyeball motion/grasp/placement quality at a glance. This is the standard QA
  artifact; never ship a collection without it.
- **Storage convention:** **smoke / small tests** → `output/` (under `output/temp/<name>/` when throwaway).
  **Large-scale full collections** → `/data3/<dataset>/` (the §4 HDF5 + per-cam videos), and create a **symlink
  in `output/`** pointing at the `/data3` dataset so it previews in-workspace. Large generated data never lives
  in the repo workspace.
- **Commit vs external:** commit code, asset definitions, collectors, transforms, configs, docs. Do **NOT**
  commit `output/` runs, HDF5, MP4, LeRobot datasets, checkpoints, or caches. Large generated data lives off the
  workspace (e.g. `/data3/...`); videos are H.264/yuv420p for VS Code preview, training uses the arrays.

---

## 8. Parallelism: build-batches
`runner/collect.py` runs one build (one BuildDR, E envs). `runner/orchestrate.py` spawns B fresh subprocess
builds (a 2nd `gs.Scene` in one process segfaults — Vulkan single-shot), each its own BuildDR, and merges the
HDF5 shards → B×E demos with ≈B distinct "looks" at native photoreal quality + size/type variety. Per build,
HDRIs are original **2K** when `E ≤ 45` (the 1K downscale was only for a single 100-env build). The B×E split is
a per-dataset flag (e.g. 30×10, 50×20). One sim process per GPU.

**Build-batch orchestration (`runner/orchestrate.py`, see [runner/README.md](../runner/README.md)).**
`orchestrate.py B E [seed0] [dataset_name]` runs **B builds × E envs = B×E demos**.
- **GPU pool (one sim process per GPU).** Detects the pool from `nvidia-smi --query-gpu=index` (or a `GPUS=0,1`
  env override) and runs a worker pool of size = n_gpus. Each worker pulls the next build off a queue, runs it
  as a **fresh `collect.py` subprocess** with `CUDA_VISIBLE_DEVICES=<gpu>` pinned (the subprocess is mandatory —
  a 2nd `gs.Scene` in one process segfaults), waits, then pulls the next. Each build's key `[COLLECT]` lines are
  relayed live, tagged `[b<b> g<gpu>]`. Different seed per build ⇒ a different per-build look (texture / colors /
  sizes / distractor set) — that IS the cross-build DR variety.
- **Shard → merge → symlink.** Each build writes a shard at `/data3/genesis_fulldr/<dataset>/_shards/build_<b>/`
  (its own `demos.hdf5` + `videos/`). After all builds finish, the shards are **merged** into
  `/data3/genesis_fulldr/<dataset>/demos.hdf5`: each shard's `/data/demo_<i>` group is copied (h5py group copy →
  ALL datasets + ALL attrs preserved) to a **globally renumbered** `/data/demo_<g>` (g = 0..B*E−1) with a
  `build` attr added; the per-cam videos are copied+renumbered to `videos/cam_*/demo_<g>.mp4`. Shards are
  **kept** (a failed merge is recoverable). An `output/<dataset>` **symlink** → the `/data3` dataset gives
  in-workspace preview. If `/data3` isn't writable the orchestrator **fails loudly** (never writes into the
  repo).
- **Summary.** Writes + prints `<dataset>/summary.json`: requested vs merged demos, builds ok/failed, clean
  successes (placed AND not penetrating), grasp rate, total abnormal-penetration count (the collision gate),
  per-cam video counts, and the per-build "looks" (each build's distractor set + counts + seed + GPU + wall).
- **Sizing a 200-demo run.** Total = B×E; wall ≈ `ceil(B / n_gpus) × per-build-time` (a 16-env 640×360 build
  ≈ 190 s). On 2 GPUs, **B=10 × E=20** (10 looks, ~16 min) or **B=20 × E=10** (20 looks, longer). Prefer more
  builds (more looks) while E stays large enough to amortize the ~30–40 s build/settle overhead.

---

## 9. Repo layout
```
world/        manipulation_stage.py · layout.py · object_factory.py · cameras.py · backgrounds.py   (the SETUP)
robots/       firefly_dual.py · ik.py · bake_*.py                                                    (the robot)
dr/           spec.py · scopes.py · object_dr.py · sampler.py · apply.py · plan.py                   (DR harness)
skills/       grasp.py · pick_place.py · trajectory.py · executor.py · virtual_ee.py · penetration.py
registry/     object_spec.py(+keypoints) · tasks.py(TaskSpec)
tasks/        _base.py · pickplace.py (reference) · (future: mug_hang.py, pour.py)                   (THIN tasks)
dataio/                hdf5_writer.py · lerobot_exporter.py
runner/       collect.py (one build) · orchestrate.py (B subprocess builds → merge shards)
assets/       robots/ · objects/ · textures/ (≥10 table textures) · backgrounds (HDRI pool)
docs/         project_overview.md (this) · domain_randomization.md · agents.md · manipulation_stage.md · …
.claude/agents/      dr-strategist.md · object-refiner.md · disturbance-designer.md
.claude/workbooks/   dr_workbook.md
```

## 10. Adding a new task (the whole contract)
1. Add/confirm the object(s) in `registry/object_spec.py` (extents, mass, grasp dz, keypoints, …) — the
   object-refiner agent can produce these.
2. Add a `TaskSpec` in `registry/tasks.py`: objects, skill, scorer, which scope-B fields apply.
3. Write `tasks/<task>.py` (~70 lines): sample DR (DR subagent) → build the stage → spawn objects → apply env DR
   → plan with a skill (often `plan_pick_place`, possibly via a `VirtualEE`) → `BatchExecutor.run` → score →
   write HDF5. Follow the §1 loop: deterministic first, then DR, then scale, then fine-tune/eval.
4. `runner/collect.py` to validate one build, then `runner/orchestrate.py B E` to scale.

Scopes A (scene) + C (visual) + photoreal + parallel + collision + dexterous IK + smooth motion come for free.

## 11. Status / roadmap
- **Done:** ManipulationStage (livery robot + good-mode collision + side rig), 3 Nyx cameras, per-env immersive
  HDRI background, fully-parallel build, thin pick-place (100/100 grasp · 97/100 place), 2K HDRI restore + wrist
  near-clip fix, smooth `BatchExecutor`, this blueprint + the full DR spec.
- **In progress:** dexterity-aware fix for jerk near full extension (route through `plan_pick_place` + dexterity
  guard + workspace-aware poses); the `dr/` full-DR harness + per-demo `DRPlan`; `runner/orchestrate.py`
  build-batches; the DR subagent + workbook; the object-refiner + disturbance agents; the virtual-EE skill +
  `ObjectSpec.keypoints`; a 2nd task (mug-hang / pour) to prove generalization with no harness edits; LeRobot
  export + a first pi0.5 fine-tune of Genesis data.
```
