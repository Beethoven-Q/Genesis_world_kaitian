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

## DR strategist  `.claude/agents/dr-strategist.md`  [PLANNED]
The professional that applies **full domain randomization** so the main agent never re-specifies it.

- **Role.** Given a task + its objects, (a) read the complete DR requirement set, (b) **select the relevant
  fields** — scopes **A (scene)** and **C (visual)** always apply (shared across all tasks); scope **B
  (object/task)** by the specific objects — and (c) recommend *realistic, professional* ranges (banana
  yellow/green not blue; apple ±10% not watermelon-sized; pose wide but task/arm/inter-object permitted, with
  the anti-coupling rules from domain_randomization.md).
- **Inputs.** The `TaskSpec` + `ObjectSpec`s; the DR registries (`dr/scopes.py`, `dr/object_dr.py`); its
  **workbook**; and (after a run) the per-demo `DRPlan` + physical outcomes from the HDF5.
- **Outputs.** (1) the selected scope-B fields + recommended ranges for this task; (2) after a run, a diagnosis
  of *which DR fields drove failures* (the per-demo `DRPlan` makes this defensible) and an **append to the
  workbook**; (3) proposed range edits (human/main-agent applied).
- **Tools / boundaries.** Read / Grep / Glob + Edit **the workbook only**. It NEVER runs sims, NEVER auto-mutates
  the global DR registries, NEVER edits task/robot code. It advises; the main agent applies.
- **Why a subagent.** Randomization is a deep, accumulating specialty (per object, per task). Isolating it keeps
  the main agent thin and lets the workbook compound into expertise.

### The DR workbook  `.claude/workbooks/dr_workbook.md`  [PLANNED]
Per `(task, object)` section, three parts:
1. **Current best ranges** — a table of each DR field's range + a confidence level (widens as evidence grows).
2. **Experience log** — append-only, dated entries: "run X: pose y-range 0.10 → failures at far edge; bowl too
   close to cube caused gripper-bowl bump; narrowed."
3. **Known hard corners (accept, label, keep)** — edge poses/values that are legitimately hard; parallel
   collection *accepts* them as failure-recovery data rather than excluding them.

This is the mechanism by which the DR agent "gets more professional and fluent over time."

---

## Object-refiner  `.claude/agents/object-refiner.md`  [PLANNED]
Makes an object *physically and visually real* before it enters the registry.

- **Role.** (a) Refine the object **URDF** (geometry, inertials, materials) for realism; (b) **label trackable
  keypoints** into `ObjectSpec.keypoints` — e.g. the **mug handle ring's center + normal**, a cap's screw axis,
  a drawer handle, a peg tip — the frames skills like the virtual-EE control; (c) run a **collision audit**:
  hollow parts stay **hollow** (convex-**DECOMPOSITION**, never a single filling hull) and solid parts **never
  interpenetrate** under firm grip.
- **Inputs.** A raw object asset (URDF/USD/mesh) + the task's needs. **Outputs.** A realistic,
  keypoint-annotated, collision-correct object + its `ObjectSpec`.
- **Boundaries.** Edits assets/registry for its object only; does not touch the stage/robot/DR engine.

## Disturbance-designer  `.claude/agents/disturbance-designer.md`  [PLANNED]
Generates **failure-and-recovery** data so the trained policy is robust.

- **Role.** Inject controlled disturbances during the main agent's execution: nudge the object as the arm
  reaches to grasp (→ did the grasp succeed? regrasp if missed), knock the object over, shift the target
  mid-transport. The main agent must *detect* the failure from privileged state and *recover* — and the demo
  records the recovery.
- **Mechanism.** A `DisturbanceSpec` (what / when / how strong) injected at a scheduled step inside
  `skills/executor.py::BatchExecutor.run`. Tunable so most trials are clean and a minority are perturbed.
- **Why.** Clean-only data yields brittle policies; failure-recovery modes teach the policy to handle the
  imperfect real world.

## More to come  [PLANNED]
The framework is extensible: a task-authoring agent (masters the §1 loop), a viewpoint/camera agent, a
scene-composition agent, etc. Each follows the same pattern — a narrow contract + a workbook → toward autonomy.

---

## How agents evolve toward autonomy
Every subagent owns a workbook. Each run produces traceable evidence (the per-demo `DRPlan`, physical outcomes,
disturbance logs). The agent diagnoses, appends, and refines. Over many tasks the workbooks become deep
playbooks; the agents need less and less human steering. The endpoint is a framework where the main agent picks
a task and the subagents set up, randomize, refine objects, inject disturbances, collect at scale, and feed
fine-tuning — autonomously. That autonomous, self-improving, data-scaling framework — not any single solved
task — is the deliverable.
