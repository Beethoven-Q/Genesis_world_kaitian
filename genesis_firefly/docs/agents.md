# Agents — the agency layer (contracts, workbooks, evolution)

The framework is **agent-native**: the main agent *calls harnesses* and *delegates specialties to subagents*.
Each subagent has a narrow contract and a **workbook** — an append-only experience store — so it accumulates
knowledge per task/object and trends from "useful" → "professional" → "master" → autonomous. This doc is the
contract reference; the big picture is in [project_overview.md](project_overview.md), the full DR spec in
[domain_randomization.md](domain_randomization.md).

Legend: **[BUILT]** exists, **[PLANNED]** designed here, built as we go.

---

## The main (task-solving) agent
Masters the end-to-end loop (see project_overview §1): build scene → privileged scripted solver → strict
physical success → deterministic smoke test → add DR → small randomized batches → fix at the *real* failure
layer → scale → LeRobot → fine-tune/eval → iterate. It does **not** re-derive infrastructure: it calls the
ManipulationStage harness (scene/robot/cameras/rendering/IK), delegates randomization to the **DR strategist**,
plans with reusable **skills**, executes with the **smooth executor**, scores by physics, and writes the data
contract. It develops new reusable skills (e.g. the virtual-EE) and promotes them to `skills/`.

---

## DR strategist  `.claude/agents/dr-strategist.md`  [BUILT — MVP]
The professional that applies **full domain randomization** so the main agent never re-specifies it.
**MVP built 2026-06-19:** the agent definition, the workbook (seeded from the cube task + v2), and the
`dr/sweep.py` probe all exist and the explore→measure→record loop has been demonstrated on a widened pose axis
(see `roadmap.md`). The fully-declarative `dr/` config (scopes.py/object_dr.py/sampler.py/apply.py/plan.py) that
would replace the hand-written ranges is **future work** — the MVP advises edits to the existing
`sample_phys_dr` + stage instead.

- **Role.** Given a task + its objects, (a) read the complete DR requirement set, (b) **select the relevant
  fields** — scopes **A (scene)** and **C (visual)** always apply (shared across all tasks); scope **B
  (object/task)** by the specific objects — and (c) recommend *realistic, MAX-extent* ranges (banana
  yellow/green not blue; apple ±10% not watermelon-sized; pose wide but task/arm/inter-object permitted, with
  the anti-coupling rules from domain_randomization.md), modelling cross-axis couplings (clearance↔pose,
  reach↔arm-side, mass/friction↔grasp).
- **Inputs.** The task + its `ObjectSpec`s; **[MVP]** the hand-written ranges in
  `tasks/pickplace.py::sample_phys_dr` + `world/manipulation_stage.py` (NOT yet a declarative `dr/scopes.py` /
  `dr/object_dr.py`); its **workbook**; and (after a run) the per-demo **DR plan + outcomes** from the HDF5.
  **[BUILT]** the per-demo DR plan is now real — `data/demo_<i>.attrs` carries `dr_cubx/cuby/bowx/bowy/tabZ/
  yaw/mass_shift/clr/reach` + `dr_pose_scale/mass_scale/fric_scale` alongside the outcome attrs (`success/
  penetrating/degenerate/has_distractors/arm/seed`).
- **Outputs.** (1) the selected scope-B fields + recommended ranges for this task; (2) after a run, a diagnosis
  of *which DR fields drove failures* (the per-demo DR plan makes this defensible) and an **append to the
  workbook**; (3) proposed range edits (human/main-agent applied).
- **Tools / boundaries.** Read / Grep / Glob + Edit **the workbook only** + **Bash to run the small
  `dr/sweep.py` probe only**. It NEVER runs a full collection, NEVER auto-mutates the global ranges, NEVER edits
  task/robot/stage code. It advises; the main agent applies.
- **The sweep/probe tool** `dr/sweep.py`  **[BUILT — MVP]**: the agent-native way to TEST a candidate DR
  setting to the edge of usability WITHOUT a full collection. It runs the existing collector (reused verbatim)
  on a SMALL batch (E≤24, default 12) for a candidate config and reports **success rate + an achieved-diversity
  measure (spread/bbox coverage of the cube/bowl poses, table height, yaw, reach) + where failures cluster**
  (per-axis z-score of the failing envs). The candidate range is expressed as env-var **half-width multipliers**
  the collector reads — `DR_POSE_SCALE`, `DR_MASS_SCALE`, `DR_FRIC_SCALE` (all default **1.0** → the v2
  collection byte-for-byte; the verify gate proves the defaults didn't move). The minimal hook that lets the
  strategist test WIDER ranges without editing the locked collector.
- **Why a subagent.** Randomization is a deep, accumulating specialty (per object, per task). Isolating it keeps
  the main agent thin and lets the workbook compound into expertise.

### The DR workbook  `.claude/workbooks/dr_workbook.md`  [BUILT — MVP]
Per `(task, object)` section, three parts (seeded for cube→bowl from the v2 200/200 run):
1. **Current best ranges** — a table of each DR field's range + a confidence level (widens as evidence grows).
2. **Experience log** — append-only, dated entries: "run X: pose y-range 0.10 → failures at far edge; bowl too
   close to cube caused gripper-bowl bump; narrowed."
3. **Known hard corners (accept, label, keep)** — edge poses/values that are legitimately hard; parallel
   collection *accepts* them as failure-recovery data rather than excluding them.

This is the mechanism by which the DR agent "gets more professional and fluent over time."

---

## Object-refiner  `.claude/agents/object-refiner.md`  [BUILT — MVP]
Makes an object *physically real and collision-correct* (the owner's #1 priority) before it enters the registry.
**MVP built 2026-06-19:** the agent definition, the reusable refine HARNESS (`registry/refine.py`), the mug demo
driver (`registry/demo_mug.py`), and the workbook all exist, and the augment->infer->collision->verify pipeline
was demonstrated END-TO-END on the YCB mug (a hollow handle ring + a hollow cup mouth) -> **SIM-READY: YES**
(penetration abnormal 0, init-stability D_pos 0.03 mm / D_ori 0.0012 rad, hollow-probe overlap 0.00 mm), with
the verification render + the `ObjectSpec` + the `handle_ring`/`cup_opening` keypoints added to the registry.
Full spec: [object_refiner.md](object_refiner.md). The articulated (PartNet-Mobility joint-oscillation) path
and a Nyx multi-view augment render are designed but **deferred to future** (the MVP did the rigid-hollow case).

- **Role.** (a) **SOURCE** the object (Objaverse/YCB rigid; PartNet-Mobility articulated — note WHERE); (b) run
  the refine harness — **AUGMENT** (trimesh mesh analysis: watertight/volume/area/extents/CoM/scale/long-axis/
  hollowness) -> **INFER** physics (mass = solid_volume x category density, friction, restitution) -> **COLLISION**
  (a GOOD collider = convex-**DECOMPOSITION** so hollow stays hollow, never a single filling hull, + a Nyx-safe
  clean OBJ) -> **VERIFY** sim-readiness; (c) **label trackable keypoints** into `ObjectSpec.keypoints` (the mug
  handle-ring center+normal, a cup opening, a cap axis, a peg tip — the virtual-EE frames); (d) emit the
  `ObjectSpec`.
- **The verify gate (REUSES our tools).** (i) `skills/penetration.py` -> abnormal ~= 0 at rest; (ii) the paper's
  **initialization-stability test** = settle with ZERO actions ~1 s then measure root drift (`D_pos <= 3 mm`,
  `D_ori <= 0.02 rad`, no explosion); (iii) for a hollow object, a thin probe threaded through the feature must
  read ~0 overlap (a branch can thread the ring). Real run + render or it didn't happen.
- **Inputs.** A raw object asset (URDF/USD/mesh) + the task's needs. **Outputs.** A realistic,
  keypoint-annotated, **sim-ready-VERIFIED** object + its `ObjectSpec` + a verification render.
- **Boundaries.** Edits assets/registry + a `demo_<obj>.py` driver for ITS object only; REUSES
  `skills/penetration.py` (never reimplements it); never touches the locked stage/robot/IK/gripper/DR engine/
  executor/collectors. Verifies with real runs; never ships a faked/unverified asset.
- **Workbook** `.claude/workbooks/object_refiner_workbook.md` [BUILT — MVP]: category density priors + a
  per-object log (the mug entry, the geometry-probe recipe that found the handle hole, hard corners).

## More to come  [PLANNED]
The framework is extensible: a task-authoring agent (masters the §1 loop), a viewpoint/camera agent, a
scene-composition agent, etc. Each follows the same pattern — a narrow contract + a workbook → toward autonomy.

---

## How agents evolve toward autonomy
Every subagent owns a workbook. Each run produces traceable evidence (the per-demo `DRPlan`, physical outcomes).
The agent diagnoses, appends, and refines. Over many tasks the workbooks become deep
playbooks; the agents need less and less human steering. The endpoint is a framework where the main agent picks
a task and the subagents set up, randomize, refine objects, collect at scale, and feed
fine-tuning — autonomously. That autonomous, self-improving, data-scaling framework — not any single solved
task — is the deliverable.
