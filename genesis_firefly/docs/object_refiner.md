# Object-refiner — raw object -> SIM-READY asset with a CORRECT collision model

> The refiner is the harness + agent that makes a NEW object **physically real and collision-correct BEFORE it
> enters the registry** — the owner's **#1 priority** (a SOLID never penetrates; a HOLLOW feature STAYS HOLLOW).
> It is grounded in the owner's ASSET2SIM paper, mapped onto our existing tools: the paper's **penetration
> score** == our `skills/penetration.py`, the paper's **initialization-stability test** == our zero-action
> settle-and-measure, and the paper's pipeline (augment -> infer physics -> good collision -> verify) == this
> harness. Authority for the bigger picture: [project_overview.md](project_overview.md) §0b,
> [lessons_genesis_nyx.md](lessons_genesis_nyx.md) §3 (the bowl collision recipe), [agents.md](agents.md).

## What it is
- **Harness** — `registry/refine.py`: a clean, reusable module that, given a mesh/USD path, runs
  **AUGMENT -> INFER -> COLLISION -> VERIFY** and returns a structured `RefineReport` (+ emits an `ObjectSpec`).
- **Agent** — `.claude/agents/object-refiner.md`: a Claude Code subagent (use **Claude**, not another VLM) that
  SOURCES the object, runs the harness, VERIFIES sim-readiness, labels trackable KEYPOINTS, and emits the spec.
- **Demo driver** — `registry/demo_mug.py`: the end-to-end demonstration on the YCB mug (a hollow handle ring +
  a hollow cup mouth) that produces the verification render. Copy it to refine a new object.

## The pipeline (augment -> infer -> collision -> verify)

### (a) AUGMENT — `refine.analyze_mesh(path, target_size_m=None, target_axis=None) -> MeshStats`
Pure trimesh geometry analysis (NO engine): `is_watertight`, **solid volume** (voxel-fill + flood-fill, robust
to OPEN meshes — `trimesh.volume` is invalid on a mug), surface area, AABB extents + bbox lo/hi, center-of-mass
(solid CoM if watertight else centroid), convex-hull + bbox volumes, **hollowness** (`1 - solid/hull`), the PCA
**long axis** (local), and a **suggested scale** to hit a target real size on a chosen axis. This is the paper's
"augment" stage (mesh analysis; an optional Nyx multi-view render can be added for the record).

### (b) INFER physics — `refine.infer_physics(mesh, category, ...) -> PhysicsInfer`
Pick the closest **category** (ceramic / glass / wood / plastic / metal / rubber / fruit — see
`refine.CATEGORY_PRIORS`) -> a realistic **density** -> **mass = solid_volume x density** (at `scale`), plus a
grippable **friction** + **restitution**. Any field is overridable (the agent reasons per object like Claude
would). Note: the voxel-fill solid volume counts the whole wall+interior, so a thick hollow object lands on the
heavy side of a real EMPTY object (acceptable; override `density` to trim).

### (c) COLLISION — `refine.extract_visual_obj(src, out_obj, scale, decompose_error_threshold) -> CollisionModel`
The GOOD collider is **convex DECOMPOSITION** (coacd, `decompose_object_error_threshold=0.04`) — the SAME recipe
`world/firefly_scene.build_bowl` uses. coacd splits the shell into many SOLID convex hulls that tile the wall,
leaving the **hollow features genuinely OPEN** between them (a single convex hull would FILL the ring + cavity —
a branch could never thread it; see lessons §3.2). It simultaneously extracts a **Nyx-safe clean OBJ** from the
Genesis-loaded geometry (Nyx segfaults on the textured USD material bind; lessons §3.1). The `CollisionModel`
records the method, the coacd threshold, the **hull count** (plural = hollow preserved), and the OBJ path.

### (d) VERIFY — `refine.verify_in_sim(visual_obj, physics, hollow_probe=None, ...) -> dict`
Loads the refined collider into a MINIMAL stage (a fixed collidable table + the object, the **same firm Newton
solver** the real stage uses — `firm_rigid_options()`, `substeps=4`) and runs the sim-readiness gates:

1. **PENETRATION** (the #1 gate — REUSES `skills/penetration.py`): `abnormal_penetration(scene, redetect=False)`
   on the just-stepped contact buffer at rest -> the abnormal count **must be 0** (threshold 7 mm, the same
   `ABNORMAL_THRESH_M` the collector uses).
2. **STABILITY** (the paper's initialization-stability test): settle with **ZERO robot actions** (~1.2 s,
   `settle_steps`), then over a `window_steps` window measure the root-pose drift **D_pos** (max XYZ move from
   the settled pose) + **D_ori** (max quaternion angle). **PASS** if `D_pos <= POS_TOL_M (3 mm)`,
   `D_ori <= ORI_TOL_RAD (0.02 rad)`, and the object did not explode (`> EXPLODE_M = 20 cm` = launched). The
   paper's strict bound is 1 mm / 0.01 rad; we loosen slightly to the firm-solver reality (coacd raises the rest
   floor a few mm; the object micro-settles on its hulls) — still tight enough to catch drift/roll/explosion.
3. **HOLLOW PROOF** (hollow objects, optional `hollow_probe`): drop a thin probe CYLINDER (a stand-in branch)
   THROUGH the declared feature (center + axis in the LOCAL frame), step, then **redetect** the TRUE
   object<->probe overlap. ~0 mm proves the feature is genuinely OPEN — a branch could thread the ring (a peg a
   hole). MEASURE the geometry first (an X-Z silhouette + an encircled-hole search) so the probe goes through
   the TRUE hole, not the handle bar.

The harness orchestrator `refine(...)` runs AUGMENT + INFER in-process (trimesh only) and dispatches the
engine-touching COLLISION + VERIFY stages to **fresh subprocesses** (one `gs.Scene` per process — a 2nd build
segfaults, exactly like `runner/orchestrate.py`). It returns a fully-populated `RefineReport`; `sim_ready` is
the AND of penetration PASS + stability PASS (+ hollow PASS if a probe was given).

## The `RefineReport`
`MeshStats` (augment) + `PhysicsInfer` (infer) + `CollisionModel` (collision) + a list of `Verdict`s (verify) +
`keypoints` + `sim_ready`. `report.pretty()` prints the human report; `report.to_dict()` is the JSON record.

## Keypoints (the trackable feature frames)
The refiner labels feature frames into `ObjectSpec.keypoints` (added 2026-06-19) — LOCAL-frame
`name -> {center, normal, [radius]}`. These are the frames a skill like the **virtual-EE** controls (thread a
ring onto a branch, seat a cap, peg a hole). The mug's `handle_ring` (center + normal = the branch axis +
radius) is the future mug-hang controlled frame; `cup_opening` is the future "drop into / pour" frame.

## How to refine a NEW object (the recipe)
1. SOURCE it: Objaverse/YCB (rigid) or PartNet-Mobility (articulated). Copy the source into
   `genesis_firefly/assets/objects/<set>/`. Note WHERE it came from.
2. Copy `registry/demo_mug.py` to `registry/demo_<obj>.py`; set the source path, the category, and the measured
   keypoints + hollow probe (run a small trimesh probe first — see `demo_mug.py`'s header for the X-Z silhouette
   + encircled-hole recipe).
3. Run: `CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/registry/demo_<obj>.py`. It prints the
   RefineReport + the ready-to-paste ObjectSpec line, writes `output/temp/<obj>_refine_verify.png` + a JSON.
4. Confirm `SIM-READY: YES`, paste the ObjectSpec line into `registry/object_spec.py::REGISTRY`, and append a
   per-object entry to `.claude/workbooks/object_refiner_workbook.md` + a `docs/roadmap.md` line.

## Run the mug demo (the demonstration)
```
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/registry/demo_mug.py
```
Prereqs: the repo `./.venv` (trimesh, coacd, genesis 1.1.2, Nyx) from the repo root; a GPU. The YCB
`mug.usd` is already under `genesis_firefly/assets/objects/ycb/`. Output: the RefineReport, the ObjectSpec line,
`genesis_firefly/output/temp/mug_refine_verify.png` (the mug resting stable, the cup mouth open, a branch
threaded through the open handle ring), and `mug_refine_report.json`.

## Sim-readiness GATES (what "done" means) — never ship a faked/unverified asset
`SIM-READY: YES` requires a REAL run reporting: `penetration_at_rest` PASS (abnormal 0) AND `init_stability`
PASS (drift within tolerance, no explosion) AND (hollow objects) `hollow_feature_open` PASS (probe overlap ~0).
If a piece can't work (no usable source, the feature fills, it won't settle), STOP and report — do not fabricate.

## Reuse + boundaries
- REUSES `skills/penetration.py` as the collision verifier (never reimplemented), `world/firefly_scene.py`'s
  `firm_rigid_options()` + the bowl convex-decomposition recipe, and the clean-OBJ extraction pattern.
- The agent edits ONLY assets + the registry + a demo driver for ITS object; it NEVER touches the locked
  stage / robot / IK / gripper / DR engine / executor / penetration detector / collectors.
