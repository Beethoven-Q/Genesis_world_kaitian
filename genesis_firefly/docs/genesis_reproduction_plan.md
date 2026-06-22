# Two-Line Plan: RoboLab(IsaacLab) rigid scale-up  +  Genesis full reproduction → deformables/fluids

> On approval, this plan is saved as a durable markdown at **`/home/kaitianchao/Projects/genesis_reproduction_plan.md`**
> (parallel to RoboLab), and copied into the new Genesis project's `docs/`. Project-folder name
> `genesis_firefly` below is a placeholder — rename freely (it must NOT be called "RoboLab_*", since
> "RoboLab" = the IsaacLab-based line).

---

## Context (why this, what outcome)

We are NOT abandoning what works. We run **two parallel lines**, sharing one sim-neutral core:

- **Line A — RoboLab (IsaacSim/IsaacLab):** keep the *verified* end-to-end pipeline productive and scale up
  more **rigid-object** tasks/data for VLA fine-tuning. Penetration/collision stay a hazard here — we just
  stay disciplined (the proven CCD + rest_offset + 64/4 recipe). Pipeline of record:
  `/home/kaitianchao/Projects/RoboLab/docs/hex_pi05_native_pipeline.md`.
- **Line B — Genesis:** reproduce the firefly dual-arm setup **100%** (arms, gripper open/close, IK,
  cameras, texture, scenes, skills, collectors, exporter), **verify completeness with pick-and-place**
  (same numbers as Line A), then exploit Genesis's deformables/fluids for **cloth folding** and **water
  pouring** — the tasks IsaacSim/PhysX is bad at.

Engine findings driving this (full eval saved in memory `project_simulator_pivot_eval.md`): Genesis is the
only engine covering cloth+paper+water in one pipeline and has the best *measured* penetration fix (IPC,
0 mm vs PhysX 4.8–8.7 mm); its cost is a from-scratch rebuild of the Isaac-coupled half. Newton (Isaac Lab
3.0 backend) can't pour water and its deformables aren't first-class in Isaac Lab yet — so Genesis is the
destination for the deformable/fluid future; Newton stays a possible future hedge, not in this plan.

The agentic big picture is unchanged in BOTH lines: **god-mode scripted task design → solve → scale data →
train/finetune VLAs (pi0.5)**, with a **sensor-only policy** (3 RGB + 14-D proprio + language).

**Shared-core decision (locked):** the import-clean sim-neutral modules become a sibling package
**`agent_sim_core`** (`/home/kaitianchao/Projects/agent_sim_core/`), pip-installed by both lines.
Sequencing: **COPY the clean modules into Genesis now (zero disruption to RoboLab), PROMOTE to
`agent_sim_core` only AFTER Genesis passes its pick-place gate on that copied code** — passing the *same*
skill/IK/exporter code through *both* simulators' gates is the proof the extraction is bug-free, before we
flip RoboLab over. Both lines import it as separate processes → no runtime interference; RoboLab can collect
while Genesis is built.

---

## 0. Governance — constraints BOTH lines MUST honor
(from `docs/firefly_migration/charter.md`, `status.md`, `task_design_recipe.md`, and
`/home/kaitianchao/Projects/RoboLab/docs/simulation_task_debugging_lessons.md`)

1. **God-mode collector ↔ sensor-only policy (absolute).** Collectors use privileged sim truth + scripted
   IK; the learned policy sees ONLY 3 RGB cams + 14-D proprio + language. **No teleop in the data pipeline**
   (teleop is verification-only).
2. **Collision realism is first-class.** Gripper↔object and object↔container penetration must NEVER happen.
   Confirm with rendering AND quantitative measurement (penetration depth, lift, settle). In Genesis this is
   the headline upgrade (IPC) — but **validate on our real pen/thin-cap geometry, don't assume**.
3. **Orientation policy = prefer top-down, relax minimal tilt** (never a forced "home" carry). Reuse
   `skills/pick_place.py` (`reachable_grasp_quat`/`reachable_place_quat`, tilt steps 0/8/16/24/32°).
4. **Modular / reusable / agent-native.** One object registry, one grasp skill, one executor, one scorer;
   no per-object branching in scene/executor; no ugly patches. New task = new scene + new collector built
   SEPARATELY (don't mutate shared skills/IK/gripper to fix a single task).
5. **Document every feature.** Every new script/scene/skill gets a user-facing doc under `docs/` (run
   command, prereqs, controls). Keep a `docs/` charter + status + task-recipe in the Genesis project too.
6. **Repo organization (hard).** Temp scripts under `scripts/**/temp/`; temp results under
   `output/temp/<name>/`; deliverables under `output/`; never dump renders/data in source dirs; videos
   H.264 (yuv420p, VSCode-friendly).
7. **Process hygiene.** ONE sim process per GPU (parallel sims on one GPU silently break timed grasps);
   long runs in background + wait on a completion signal (don't poll-spam); scripts exit cleanly.
8. **Success = physics ground truth**, not command success (`score_pick_place`: lifted clear + settled in
   container + released/clear-of-gripper). Keep the `obj_ee` released-guard + in-bowl z-band.
9. **Data format is a fixed contract** (§4) — a Genesis collector writes the SAME HDF5 so the LeRobot
   exporter is reused unchanged.
10. **Lessons not to repeat** (from `simulation_task_debugging_lessons.md`): debug layer-by-layer (asset →
    IK → contact → camera → policy); never move a table/object mid-episode via physics; weld/settle before
    reading state; verify camera views match the real robot before trusting data; one Isaac/sim process per
    GPU.

---

## LINE A — RoboLab (IsaacLab) rigid-object scale-up  [keep productive]

Goal: keep shipping verified rigid-object VLA data on the engine that already works end-to-end. No engine
change. Disciplined collision (existing CCD + rest_offset + 64/4 recipe).

A1. **Expand the object registry** (`robolab/scenes/object_registry.py`): add more rigid YCB/Objaverse
   objects (from the ~323-asset library + `object_catalog.json`), each as one `ObjectSpec` with MEASURED
   geometry (use `scripts/grasp/temp/inspect_object_usd.py` + `pca_axis.py`) and per-object handling hints
   (friction, contact/rest_offset, grasp_dz, release_dz, place_xy_tol_cm, x_range). No new code per object.
A2. **DR-verify each new object** with `scripts/grasp/eval_pickplace_multi.py --object X --dr --mass-dr
   --table-h` (±5 cm) → target ≥ the current bar (deterministic 16/16; full-DR ~95%+; thin/rolling objects
   lower is acceptable, ~70%+). Deliver video (successes/failures reels) + npz, per the existing harness.
A3. **Finish the in-flight rigid tasks** already scaffolded: pen-uncap (task #61 — collector to a clean
   rate), mug-hang (#37/#40 high-branch ring-servo). These reuse the same skills.
A4. **Scale fine-tune datasets** (god-mode scripted) → LeRobot v3 via `core/export/lerobot_exporter.py` →
   pi0.5 fine-tune → eval/deploy, exactly per `hex_pi05_native_pipeline.md`. Keep raw HDF5.
A5. **Commit cadence** on `firefly_y6_gr100` branch; document each new task under `docs/firefly_migration/`.

(Line A needs no new architecture; it continues the proven loop. It also serves as the *reference* the
Genesis line must match.)

---

## LINE B — Genesis full reproduction → verify pick-place → deformables/fluids

New project: **`/home/kaitianchao/Projects/genesis_firefly/`** (placeholder name). Internal package
`genesis_firefly/` mirroring RoboLab_firefly's layout: `robots/ scenes/ skills/(thin) collectors/ core/`.
Genesis docs cheat-sheet in §5; data contract in §4; transfer/rebuild map in §6.

### B0 — Scaffold + shared-core copy
- **Repo structure = FORK-AND-BRANCH (owner's decision).** Genesis is alpha (breaking changes every 1–3 wk),
  so we FORK the engine for a pinned, patchable snapshot: GitHub fork `Beethoven-Q/Genesis_world_kaitian`
  (of `Genesis-Embodied-AI/genesis-world`), local working repo = the fork on branch **`genesis_firefly`**,
  installed EDITABLE (`pip install -e .`) so our engine patches take effect. Two guardrails to keep it clean:
  (1) ALL our application code lives under ONE new top-level dir **`genesis_firefly/`** (`scenes/ collectors/
  skills/ assets/ docs/`) — we never edit upstream engine files, so `git fetch upstream && merge` into our
  branch stays conflict-free and our work is trivially extractable later; (2) the sim-neutral
  **`agent_sim_core` stays its OWN separate repo** (pip-installed by BOTH lines), NOT buried in the engine
  fork (else the RoboLab line can't import it). Remotes: `origin`=our fork, `upstream`=Genesis-Embodied-AI.
  (Trade-off accepted: our `git log/status` intertwines with the 300 MB engine tree — noisier, but worth the
  alpha-engine control.)
- Create the project dir + a Python venv with `pip install genesis-world` (pin a version — see Risk R1),
  `h5py pyarrow imageio numpy scipy pinocchio`. `gs.init(backend=gs.gpu)` (Genesis backend is `gs.gpu`, not
  `gs.cuda`). Optionally `pip install gs-nyx-plugin` for photoreal (needs CUDA 12.9+/driver 575+).
- **Copy (don't yet package) the import-clean core** into `genesis_firefly/_core_vendored/` (becomes
  `agent_sim_core` after the gate): `robolab/skills/{grasp,pick_place,trajectory}.py`,
  `robolab/integrations/soda_bimanual/{ik.py,config.py}`, the ObjectSpec data half of
  `object_registry.py`, `core/export/lerobot_exporter.py`, and a sim-neutral subset of `constants.py`
  (paths only; drop `DEVICE/VISUALIZE/RECORD_*` Isaac flags). Keep SODA on `sys.path`
  (`SODA_BIMANUAL_ROOT=/home/kaitianchao/Projects/soda-bimanual`).
- **Treat the copied core as a STARTING POINT, not gospel (owner directive — stay skeptical).** Sim-to-sim
  differs: be ready to REFINE each module for Genesis where conventions diverge (quaternion wxyz handling,
  axis/joint-order, the action-layer gripper coupling, any scipy/Isaac-frame assumptions baked into the
  skills). Aggressively PRUNE anything RoboLab/Isaac-only — some past effort was Isaac-specific and is NOT
  needed: e.g. `streaming_hdf5_handler.py` imports Isaac `EpisodeData` → replace with a tiny direct h5py
  writer to the §4 schema; drop unused helpers/flags. Keep the eventual `agent_sim_core` MINIMAL, elegant,
  agent-native — only what BOTH lines truly need; optimize hard. The B10 promotion ships the pruned,
  twice-proven core, never a verbatim copy.
- **Copy the source-of-truth assets** (sim-agnostic): the composed dual URDF
  `assets/robots/firefly_y6_gr100/dual_firefly_y6_gr100.urdf` + `source/` STL meshes + `calibration/dual/*.json`
  + the object library `assets/objects/` (+ `object_catalog.json`). Reuse
  `scripts/robot_setup/build_firefly_dual_urdf.py` if the URDF must be recomposed.
- Set up `docs/` (charter copy, status, task-recipe) and the repo-org dirs (`scripts/**/temp/`, `output/temp/`).

### B1 — Robot asset in Genesis  [rebuild: import; reuse: URDF+meshes]
- Import via `gs.morphs.URDF(file=dual_firefly_y6_gr100.urdf, fixed=True, merge_fixed_links=False,
  links_to_keep=[left_ee_link,right_ee_link,left_link_6,right_link_6,camera mounts])`; `scene.add_entity`.
  Keep fixed-frame links (ee, link_6, camera mounts) for IK + camera attach.
- Map joint names → Genesis dof indices: `[robot.get_joint(n).dof_idx_local for n in LEFT_ARM_JOINTS+...]`.
  Preserve the **14-D layout** `STATE_JOINTS_14 = [L_j1..L_j6, L_grip_driven, R_j1..R_j6, R_grip_driven]`.
- Base offsets preserved by the URDF (dual base, LEFT y=+0.224, RIGHT y=−0.224); base z = arm-table 0.25.
- **Validation:** load + render front/top views; confirm link poses, joint count, no self-explosion. Apply
  the self-collision filter equivalent for the one spurious pair (link_5 ↔ gripper_base_link).

### B2 — IK bridge (reuse SODA, re-measure one constant)  [reuse]
- Reuse `soda_bimanual/ik.py` (`FireflyDualIK`, pure NumPy/Pinocchio, zero engine imports) UNCHANGED.
- The ONLY engine-coupled constant: `_RZ_P90` (Rz(+90°) between SODA TCP frame and the `ee_link` body frame
  Isaac reports). **Re-measure it against Genesis's `ee_link` FK** (1–2 h calibration script): command a
  known joint config, read Genesis `ee_link` pose, solve for the offset; verify origin coincidence ≈0 mm.
- TCP offset 0.187 m (+Z of gripper_base) and `use_tcp_frame=True` stay. Confirm Genesis dof ordering equals
  SODA's joint vector (build a joint-name→index remap if Genesis reorders).
- **Validation (quantitative, per owner's preference):** sweep N target EE poses → IK → `control_dofs_position`
  → read back `ee_link` pose → report position/orientation error (target ≈ sub-mm / <1°). Port
  `scripts/ik_verify/verify_dual_ik.py` logic.

### B3 — Gripper open/close + firm contact  [reuse constants; choose Genesis coupler]
- Constants reused: `GR100_OPEN=0.0`, `GR100_CLOSE=0.9`, `GR100_MIMIC=-1.0` (right_claw = −1·left_claw,
  coupled at the ACTION layer, not physics mimic — same as Line A). Driven joint
  `left_gripper_left_joint_1`, mimic `left_gripper_right_joint_1` (+ right).
- Drive gains from the MIT controller: grippers kp=200, kd=8.0, effort=10 N·m (the FIRM pinch — do NOT
  lower force). Arm gains: proximal J1–3 kp=200/kd=12.5/eff=28; wrist J4–6 kp=[75,15,15]/kd=[6,0.31,0.31]/
  eff=10. Set via `robot.set_dofs_kp/kv` + `set_dofs_force_range`.
- **Contact strategy (penetration fix), corrected per source:** for RIGID-RIGID grasps (pen/cap/cube), the
  lever is the rigid solver, NOT IPC: `gs.options.RigidOptions(constraint_solver=Newton, iterations=100+,
  constraint_timeconst≈0.005, noslip_iterations=5, enable_self_collision=True)`. Only escalate to
  intersection-free `IPCCouplerOptions(enable_rigid_rigid_contact=True, contact_d_hat=0.001)` if the thin cap
  still tunnels (slower). (IPC's primary value is rigid↔cloth/FEM — that pays off in B11.) Re-derive the
  closure calibration (Line A: q≈0.45→2 cm gap, q≈0.59→touch, q=0.9→firm no-penetration).
- **Validation:** port `diag_pen_grip.py` — grasp the pen, open mid-air, confirm FELL not STUCK; measure
  penetration depth on the thin cap (target ~0 mm). This is the *whole reason* for the pivot — prove it here
  on our geometry, and pick the cheapest solver setting that achieves it.

### B4 — Cameras (3) + intrinsics + convention  [reuse calib numbers; rebuild API]
- Reuse calibrated values from `calibration/dual/{left_hand,right_hand,side}.json` (D405 wrist ×2, D435i
  side). **Genesis intrinsics = vertical FOV only** (no fx/fy/cx/cy): compute `fov = 2·atan(0.5·H / f_pixel)`
  from each camera's calibrated focal at its render height. **Caveat:** Genesis forces the principal point to
  the image centre, so a calibrated off-centre cx/cy can't be matched exactly — quantify the pixel offset; it
  was negligible for the side cam (cy≈center) and should be small for the wrist cams (acceptable for a
  sensor-only RGB policy; note it in the camera doc).
- `cam=scene.add_camera(model="pinhole", res=, pos=, lookat=, up=, fov=)`. **Wrist cams attach to link_6**:
  `cam.attach(robot.get_link("left_link_6"), offset_T)` + `cam.move_to_attach()` every step (offset_T = the
  calibrated optical pose in link frame, pos≈(−0.081,0.0098,0.0453) L). Side cam world-fixed via `set_pose`
  at pos≈(0.045,0.0326,0.8155).
- **Coordinate convention:** Genesis cameras use **OpenGL (-Z fwd, +Y up)** — the SAME convention we already
  baked from the OpenCV calib for Isaac. So reuse `sync_cameras_from_calib.py`'s existing OpenCV→OpenGL
  conversion to build `offset_T`/poses; do NOT re-derive a new mapping. Render keys MUST be `cam_lw`,
  `cam_rw`, `cam_side` (§4).
- **Validation:** render all 3 views (Rasterizer for speed); overlay vs the real-robot reference frames;
  confirm the "views match the real robot" charter check BEFORE collecting data. Render one RayTracer
  (photoreal) frame to gauge the sim-to-real RGB look.

### B5 — Texture / livery  [rebuild — prior #1 time-sink]
- Reproduce the real Firefly Y6 scheme: metallic SILVER arm links, near-black wrist motor (link_6), dark
  base; metallic≈0.85, roughness≈0.42. Apply per-link via Genesis surface/material API (PBR). Add the
  visible camera bodies (D405 wrist mesh `source/firefly_y6/d405.stl`; side D435i body + support stick
  cylinder r=8 mm, h=0.815).
- Renderer for the sensor RGB: start on the **Rasterizer** (fast, deterministic) to stand the pipeline up;
  for photoreal sim-to-real fidelity use **LuisaRender** (in-repo) or the **Nyx** plugin (`gs-nyx-plugin`,
  Apache-2.0, PBR + Gaussian splats + multi-env) — pick per the throughput/fidelity trade (Risk R2). Bulk
  collection can run the Rasterizer/BatchRenderer; mint a photoreal subset for the sim-to-real check.
  **Validation:** side-by-side render vs the Isaac livery; confirm the sensor RGB looks plausible for the
  policy.

### B6 — Scene (two tables + objects + contact)  [rebuild scene cfg; reuse layout numbers + object specs]
- Two-table layout (reuse `TableLayout`: arm-table 0.34×0.90, object-table 0.70×0.90, both height 0.25,
  seam 0.13). Static box colliders with a small standoff so the gripper stops ON the table.
- Spawn objects from the registry via a NEW Genesis `build_object` factory (the ObjectSpec DATA is reused;
  the factory is rebuilt): `gs.morphs.Mesh/URDF` from the object USD/OBJ, set mass/friction, settle on table.
  Bowl = reuse the procedural geometry (BOWL_HALF_H=0.0375). Pen-uncap two-part pen later (parity, optional).
- Light/dome equivalent (intensity ~2500). Object DR (pose/yaw/mass/table-h) mirrors Line A's eval.

### B7 — Skills + planner  [reuse verbatim]
- `skills/grasp.py` (`orientation_aware_grasp_quat`, `world_long_axis`, `world_grasp_center`,
  `tilted_base_quat`, `transport_quats`), `skills/pick_place.py` (`plan_pick_place`, reachable grasp/place,
  `score_pick_place`), `skills/trajectory.py` (`densify`) — all pure NumPy/scipy, used UNCHANGED. They
  consume the IK callable (B2) and object geometry (B6); they own nothing sim-specific.

### B8 — Collector + recording + exporter  [rebuild collector loop; reuse exporter + schema]
- New Genesis collector (mirrors `eval_pickplace_multi.py`): per trial — settle object, read god-mode state,
  `plan_pick_place(...)` → `densify` → per-step `ik.solve` → `robot.control_dofs_position` + gripper command;
  step the scene; capture 3 cameras every Nth step; read `obj.get_pos/get_quat` for scoring.
- **Write the EXACT HDF5 schema** (§4) so `lerobot_exporter.py` is reused unchanged: `/data/demo_N/actions`
  (T,14), `states/articulation/robot/joint_position` (T,14) [+velocity], optional `ee_pose/{position,
  orientation}`; attrs `num_samples,success,seed`. Camera keys `cam_side/cam_lw/cam_rw`.
- **Validation:** collect 5 trials → run the SAME exporter → load the LeRobot v3 dataset in the HF viewer →
  confirm columns/videos/state-action dims match Line A's datasets.

### B9 — VERIFICATION GATE = pick-place parity  [the "100% reproduced" milestone]
- Run the Genesis pick-place eval on apple/banana/pen/cube: deterministic (fixed pose, both arms) AND full
  DR (pose + yaw 0–360° + mass ±30% + table-h ±5 cm). **Pass = match Line A's numbers** (deterministic
  16/16; full-DR apple/banana ~10/10, cube ~9/10, pen ~8/10) with **zero penetration** (released-guard +
  measured contact). Deliver success/failure video reels + npz + a parity table vs Line A.
- This gate proves arms+IK+gripper+cameras+texture+scenes+skills+collector+exporter all reproduce.

### B10 — Promote shared core → `agent_sim_core`  [after B9 passes]
- Extract the now-twice-proven import-clean modules into `/home/kaitianchao/Projects/agent_sim_core/`
  (`pip install -e`): `skills/`, `soda_ik/` (the bridge), `object_spec/` (ObjectSpec data + registry data),
  `export/lerobot_exporter.py`, neutral `constants/`. Add a tiny smoke-test (`pytest`) covering grasp-quat
  math, IK round-trip, exporter round-trip.
- Flip BOTH lines to `import agent_sim_core ...` (Genesis first, then RoboLab — RoboLab keeps its own
  sim-coupled `build_object`/scenes/hdf5-handler). Pin the version; freeze during active collection; re-run
  both lines' pick-place smoke tests on any core change.

### B11 — Deformables/fluids payoff  [the reason for Genesis]
- **Water pouring (highest payoff, lowest sim-risk):** new scene with two cups + `gs.materials.SPH.Liquid`
  (mu/gamma tunables); scripted bimanual lift-tilt-pour; god-mode success = poured-fraction (SoftGym-style:
  particles-in-target / total). Watch the open SPH-scatter bug (Risk R3) — validate a clean pour first.
- **Cloth folding:** new scene with `gs.morphs.PBD.Cloth` (or FEM) + IPC coupler for stable grasp-hold;
  scripted dual-arm grasp-corner → fold; success = fold-overlap / flatness metric. Expect R&D tuning
  (gripper-hold contact instability has a track record) — keep the grasp on IPC.
- **Paper folding (optional next):** `gs.materials.MPM.ElastoPlastic` thin shell for a crease that stays.
- Each: reuse the agentic loop (god-mode scripted collector → SAME HDF5 schema → `agent_sim_core` exporter →
  LeRobot v3 → pi0.5 fine-tune). New scene + new collector + new success scorer, built SEPARATELY (rule 4).
  Document each under the Genesis project's `docs/`.

---

## 4. Data contract (fixed — Genesis collector writes this; exporter reused unchanged)

**HDF5 written per episode** (read by `core/export/lerobot_exporter.py`):
```
/data/demo_N/
  actions                                  (T, 14) float32     # absolute joint targets, 14-D
  states/articulation/robot/joint_position (T, 14) float32
  states/articulation/robot/joint_velocity (T, 14) float32     # optional but recommended
  ee_pose/position                         (T, 3)  float32     # optional
  ee_pose/orientation                      (T, 4)  float32 wxyz # optional
  attrs: num_samples=T, success=bool, seed=int(optional)
```
14-D order = `STATE_JOINTS_14` = [L_j1..L_j6, L_grip_driven, R_j1..R_j6, R_grip_driven].

**LeRobot v3 output** (produced by the exporter): `meta/info.json` (robot_type, fps int, features,
totals); parquet columns `episode_index, frame_index, index, task_index, timestamp, next.done, action(list),
observation.state(list)[, observation.velocity, observation.ee_position, observation.ee_orientation]`;
camera video keys `observation.images.cam_side / cam_lw / cam_rw` → `videos/<key>/chunk-000/file-000.mp4`
(avc1/yuv420p). Camera RGB capture keys in the collector MUST be `cam_side, cam_lw, cam_rw`.

---

## 5. Genesis API cheat-sheet — SOURCE-VERIFIED against the cloned repo (v1.1.2)
Repo cloned read-only to `/home/kaitianchao/Projects/genesis-world` (v1.1.2); narrative docs
https://genesis-world.readthedocs.io/. Every signature below is confirmed against source (file:line) +
working examples — NOT guessed.
- **Init/scene/headless/parallel:** `gs.init(backend=gs.gpu)` (**NOT** `gs.cuda`); `scene=gs.Scene(
  sim_options=gs.options.SimOptions(dt=0.01), rigid_options=gs.options.RigidOptions(...),
  coupler_options=..., renderer=..., show_viewer=False)`; `scene.build(n_envs=N, env_spacing=(1,1))`;
  `scene.step()`. (scene.py:94/852)
- **Robot import:** `e=scene.add_entity(gs.morphs.URDF(file=..., fixed=True, merge_fixed_links=False,
  links_to_keep=[ee/link_6/cam mounts], default_armature=0.1, requires_jac_and_IK=True))`. quat=**wxyz**,
  euler=degrees. (morphs.py:989)
- **DOF indexing:** `j=e.get_joint(name); j.dofs_idx_local`; `lk=e.get_link(name)`. Build the 14-D index
  vector from joint names. (rigid_entity.py:1422/1456)
- **Control from external IK:** `e.control_dofs_position(position, dofs_idx_local)` (PD);
  `e.set_dofs_position(qpos, dofs_idx_local, zero_velocity=True)` (god-mode reset);
  `e.get_dofs_position/velocity(dofs_idx_local)`; gains `e.set_dofs_kp(kp_arr, idx)` /
  `set_dofs_kv(kv_arr, idx)`; limits `e.set_dofs_force_range(lo_arr, hi_arr, idx)`; firm gripper close
  via `e.control_dofs_force(force, idx)` if PD isn't enough. Genesis has its OWN `e.inverse_kinematics(...)`
  — we BYPASS it and feed SODA joint targets. (rigid_entity.py:3813/1790/3602/3674/2734)
- **EE pose read (for IK frame calib):** `lk.get_pos(relative=True)` (3,), `lk.get_quat(relative=True)`
  (4, **wxyz**). (rigid_link.py:254/269)
- **Cameras:** `cam=scene.add_camera(model="pinhole", res=(w,h), pos=, lookat=, up=, fov=, GUI=False)`.
  **Intrinsics = FOV only** (no fx/fy/cx/cy; principal point forced to image CENTRE): `f=0.5*H/tan(fov/2)`
  → set `fov = 2*atan(0.5*H / f_pixel)`. `cam.set_pose(transform=4x4 | pos/lookat/up)`. **Convention =
  OpenGL (-Z fwd, +Y up)** → convert our OpenCV-optical calib by negating the Y,Z columns. Wrist-cam follow:
  `cam.attach(link, offset_T_4x4)` + `cam.move_to_attach()` EACH step. `rgb,depth,seg,normal =
  cam.render(rgb=True, depth=True)` → rgb torch (H,W,3) float32 [0,1]. (scene.py:687, camera.py:198/379/581/941)
- **Renderers (all open):** in-repo = Rasterizer (default, fast), **RayTracer = LuisaRender (photoreal path
  tracer)**, BatchRenderer (Madrona, CUDA, batched multi-env). PLUS **Nyx** — a SEPARATE **Apache-2.0** pip
  plugin (`pip install gs-nyx-plugin`, repo `Genesis-Embodied-AI/genesis-nyx`) that plugs in as a camera
  sensor: GPU path tracer + PBR + **3D Gaussian splats** + **multi-env rendering** (needs CUDA 12.9+/driver
  575+; early-stage, no tagged releases yet). Plan: Rasterizer/BatchRenderer for BULK data collection;
  Nyx **or** LuisaRender for a photoreal subset + sim-to-real RGB validation. (renderers.py)
- **Livery/materials (per link):** after add_entity, set `e.get_link(name).surface = gs.surfaces.Metal(
  color=, ...)` / `gs.surfaces.Rough/Smooth/Plastic/BSDF(color=, roughness=, metallic=)`; predefined
  metals (Iron/Aluminium/Gold) for the brushed-metal arm. (surfaces.py)
- **Objects + god-mode state:** `scene.add_entity(gs.morphs.Mesh(file=.obj/.stl/.glb, scale, pos, euler,
  convexify=True) | gs.morphs.USD(file=.usd), material=gs.materials.Rigid(rho=, friction=, needs_coup=True))`;
  read `e.get_pos()/get_quat()(wxyz)/get_vel()/get_ang()`, multi-link `get_links_pos/quat`. **.usd imports
  directly via `gs.morphs.USD`** → the 323-object library likely loads with little re-authoring (verify).
  (morphs.py:686, rigid_entity.py:1491)
- **Contact / penetration control — RIGID-RIGID grasp (our pen/cap):** tune the rigid solver, NOT IPC first:
  `gs.options.RigidOptions(constraint_solver=gs.constraint_solver.Newton, iterations=50→100+,
  constraint_timeconst=0.01→0.005, noslip_iterations=5, enable_self_collision=True, max_collision_pairs=,
  contact_pruning_tolerance=)`. Escalate to intersection-free IPC ONLY if the thin cap still penetrates:
  `coupler_options=gs.options.IPCCouplerOptions(enable_rigid_rigid_contact=True, contact_d_hat=0.001,
  contact_resistance=1e7)` (slower). (solvers.py:396/187)
- **Deformables/fluids (B11) — ALL verified present in v1.1.2 with working examples:** cloth
  `gs.materials.PBD.Cloth(stretch_compliance, bending_compliance)` (fast, Legacy coupler) OR
  `gs.materials.FEM.Cloth(E, nu, thickness, bending_stiffness)` (intersection-free, needs
  `IPCCouplerOptions`); water `gs.materials.SPH.Liquid(rho, stiffness, mu, gamma)` / `PBD.Liquid` /
  `MPM.Liquid`; paper `gs.materials.MPM.ElastoPlastic(E, nu, von_mises_yield_stress)`. Rigid↔deformable
  TWO-WAY coupling is automatic via `gs.materials.Rigid(needs_coup=True, coup_links=(claw links),
  coup_type="two_way_soft_constraint")`. Examples to copy: `examples/coupling/cloth_on_rigid.py`,
  `examples/IPC_Solver/ipc_robot_cloth_teleop.py`, `examples/pbd_liquid.py`, `examples/coupling/sph_rigid.py`.

---

## 6. Transfer vs rebuild map (the firefly stack in Genesis)
**Reuse verbatim (sim-agnostic, → `agent_sim_core` after B9):** SODA IK bridge (`ik.py`,`config.py`) [re-measure `_RZ_P90` only]; `skills/{grasp,pick_place,trajectory}.py`; ObjectSpec data + registry data; `lerobot_exporter.py`; the URDF + STL meshes + calibration JSONs; the object library; the HDF5 *schema*.
**Rebuild in Genesis (Isaac-coupled):** USD/robot import (use Genesis URDF import instead); `firefly_dual.py` actuator cfg → Genesis `set_dofs_kp/kv/force_range`; `firefly_cameras.py` → Genesis camera + OpenCV→Genesis convention; all scene cfgs (`firefly_*_scene.py`) → Genesis scene/entity API; `object_registry.build_object()` factory; the collector loop (`eval_pickplace_multi.py`); the HDF5 *writer* (the handler imports Isaac `EpisodeData` — write h5py directly to the §4 schema); texture/livery; self-collision filtering.
**Unaffected (downstream of data):** sensor-only pi0.5 policy + real-robot ZMQ deploy (engine-independent).

---

## 7. Risk register
- **R1 — Genesis API churn / hype.** 1–3 wk release cadence with breaking changes; Dec-2024 speed claims
  were debunked ~100×. → **Pin a version**, don't track HEAD; discount marketing; benchmark our own scenes.
- **R2 — Photoreal rendering throughput / maturity (NOT a licensing problem).** Photoreal is OPEN —
  LuisaRender (in-repo) and Nyx (`gs-nyx-plugin`, Apache-2.0). Real risks: path-tracing is slow for
  data-scale (mitigate: Rasterizer/BatchRenderer for bulk, photoreal for a subset + sim-to-real check); Nyx
  is early-stage (no releases, CUDA 12.9+/driver 575+). FOV-only intrinsics can't match an off-centre
  principal point (quantify the pixel offset; expected negligible). → validate the sensor RGB early (B5).
- **R3 — Fluids not turnkey.** Open SPH-scatter bug hits pouring; cloth grasp-hold has instability history.
  → de-risk water/cloth as scoped pilots (B11) before committing a dataset; keep grasp contact on IPC.
- **R4 — Joint-order / drive-gain desync.** Genesis may reorder dofs vs SODA's 14-D vector; gains may not
  materialize. → explicit joint-name→index remap + an IK round-trip test (B2) before any collection.
- **R5 — Camera convention.** OpenCV↔Genesis frame mismatch silently corrupts views. → render-vs-real check
  (B4) is a hard gate before data.
- **R6 — Shared-core extraction bugs.** → the copy-now/promote-after-gate sequencing + a pytest smoke-test
  (B10) makes both gates the proof.

---

## 8. Verification strategy (confirm via rendering + quantitative — owner's standard)
Per-phase gates: B1 render+joint check · B2 IK error <sub-mm/<1° · B3 mid-air-release + ~0 mm penetration ·
B4 views-match-real · B6 objects settle no-penetration · B8 5-trial → exporter → HF-viewer round-trip ·
**B9 pick-place parity with Line A (the headline gate)** · B11 poured-fraction / fold-overlap metrics.
Every gate produces a render (or video reel) AND a number, logged under `output/temp/<name>/`.

---

## 9. Execution step 0 (first actions on approval)
1. Save this plan to `/home/kaitianchao/Projects/genesis_reproduction_plan.md` (durable, parallel to RoboLab).
2. **Fork** `Genesis-Embodied-AI/genesis-world` → `Beethoven-Q/Genesis_world_kaitian` (gh, authed); set the
   local clone's `origin`=fork, `upstream`=original; branch **`genesis_firefly`**; create the `genesis_firefly/`
   top-dir for our code; venv + `pip install -e .` (editable engine) + deps; copy charter/status/task-recipe +
   this plan into `genesis_firefly/docs/`.
3. Begin B0 (copy core + assets into `genesis_firefly/`), then B1.
Line A continues independently (A1–A5) whenever a GPU is free — one sim process per GPU.

---

## LINE C (BONUS, added 2026-06-18) — AERO dexterous-hand branch
Owner add-on. AFTER the gripper Genesis line is done (safety: keep as a SEPARATE branch, don't perturb the
working gripper pipeline). Goal: make the SAME agentic design->solve->collect pipeline compatible with a
DEXTEROUS HAND so the agent can solve tasks with fingers, not just a parallel gripper.
- Hand: **AERO open hand** (https://tetheria.github.io/aero-hand-open/) — fetch its URDF/MJCF + meshes.
- Architecture (agent-native, reuse): swap ONLY the end-effector. Reuse the firefly arm, Genesis-native arm
  IK (ee_link target), cameras, livery, two-table scene, collectors, HDF5 schema + LeRobot exporter, DR.
  New: a hand model + a hand-control abstraction (joint targets / grasp synergies / a small set of named
  pregrasps), and hand-aware grasp planning in the skills (replace the 1-DOF gripper close with a finger
  closure policy). Keep the `EndEffector` interface generic so gripper and hand are interchangeable.
- MILESTONE = pick-and-place WITH THE HAND, identical deliverables (full-DR 2x2 4-cam videos + training
  data, 5 objects, 10 trials = 5 left + 5 right).
- STRETCH SUPER-GOAL: two-arm **throw-and-catch a tennis ball** back and forth (design->solve->collect a
  dataset). Needs dynamic release/catch timing + a ballistic/predictive catch controller — a great showcase
  of the agentic pipeline on a dynamic bimanual task.
- Plan it as its own branch off the Genesis line once Lines B (gripper) is verified.

**Line C refinement (2026-06-18):** AUTO-TRIGGER — once gripper pick-place is verified with good success
(all deliverables met), proceed to the AERO hands WITHOUT waiting for owner verification. Substitute BOTH
grippers with a PAIR of AERO hands on the dual arm. Deliverables: (1) pick-and-place with hands (same as
gripper), (2) two-arm THROW-AND-CATCH a tennis ball in a PARABOLIC air trajectory (only possible with hands).
Behind a generic `EndEffector` interface (gripper | hand interchangeable). Document every step for traceability.
