# Line C — Integrating the TetherIA AERO Open dexterous hand (plan)

Research + planning deliverable (2026-06-20). **Not yet started** — awaiting owner green-light. The AERO repo is
cloned to `/home/kaitianchao/Projects/aero-hand-open` (HEAD `b27463c`, with the `sim_rl/simulation` =
`aero-open-sim` submodule initialised). Sources: the project page (tetheria.github.io/aero-hand-open), the docs
(docs.tetheria.ai), and the repo (`hardware/`, `ros2/...description`, `sdk/`, `sim_rl/`).

> **Bottom line:** a usable, high-quality sim model exists TODAY — but it is a **MuJoCo MJCF**, not a
> Genesis-ready asset, and its tendon actuation is something Genesis-rigid cannot ingest directly. The first real
> cost is a **modest asset *conversion* (a few days), not a from-scratch build**: the URDF, meshes, joint tree,
> primitive colliders, and a **closed-form tendon-coupling map** all ship and are directly reusable.

This extends the existing harness (never forks), per the owner HARD rule + [project_overview.md](project_overview.md) §0b.

---

## 1. The hand (spec)
- **16 joints, 7 actuated** (7 motors; 9 passive/underactuated). 5 fingers: index/middle/ring/pinky (3 joints
  each: mcp_flex/pip/dip) + thumb (4: cmc_abd/cmc_flex/mcp/ip). Confirmed in the URDF + SDK constants.
- **Tendon-driven** (cable+spring+pulley), 100% backdrivable, low gearing. The 7 channels: `thumb_cmc_abd`
  (a DIRECT revolute joint, not a tendon), `thumb_cmc_flex`, `thumb_tendon`, `{index,middle,ring,pinky}_tendon`.
- **Closed-form coupling SHIPS** (`sdk/.../joints_to_actuations.py` + `actuations_to_joints.py`) — the single
  most valuable artifact. Per finger `tendon = (12.4912·mcp + 7.3211·pip + 9.0·dip)/9.0`; the INVERSE (what we
  drive) assumes equal angles mcp=pip=dip per finger (one tendon curls the three together). Matches the MJCF
  `<equality>` constraints (`right_hand.xml:743-749`).
- **Sensors:** tendon force, motor current/temp/speed, actuator position (7 of them). **No fingertip tactile, no
  per-joint encoders.** Control modes: position / torque / tendon-force.
- **Physical:** 198×95×53.5 mm, <400 g; fingertip ~10–12 N; full open-close ~1.2 Hz. Demonstrated **catching a
  flying tennis ball** (the owner's stretch goal) + lifting an 18 kg jug. 6 V DC ≤8 A, USB via ESP32-S3.
- **Wrist mount:** a flat surface with four M3 inserts; adapters ship for Piper/Unitree/tripod — **none for
  Firefly Y6** (we model the transform in sim; no physical adapter needed). Hole pattern in the open CAD.

## 2. Sim-asset assessment
Two assets ship; both useful:
- **URDF** `ros2/src/aero_hand_open_description/urdf/aero_hand_open_right.urdf` (+ `_left`) + 25 STL meshes; clean
  kinematic tree, per-link inertials + joint limits; **no `mimic`** (coupling lives in the SDK/MJCF, not URDF);
  **collision = full visual STL** (fails our collision-#1 rule as-is → needs decomposition or the MJCF boxes).
- **MJCF** `sim_rl/simulation/mujoco/right_hand.xml` (+ `scene_right.xml`) — the BETTER source: tendon-accurate
  (`<tendon>` + pulleys + springs), 7 position actuators, tendon/abduction sensors, the `<equality>` coupling,
  and **hand-tuned PRIMITIVE collision geometry** (palm = 11 boxes, each phalanx a box, friction-tuned fingertip
  boxes) + a self-collision `<contact><exclude>` list (`:118-181`). This primitive collider set is exactly the
  faithful-but-cheap model our penetration gate wants, and the exclude list → our `ignore_pairs`.

**Genesis-readiness:** NOT drop-in. Genesis-rigid does not ingest MuJoCo spatial tendons/pulleys/springs. So we
do NOT port the tendon mechanism — instead **drive the 16 URDF revolute joints directly** (`control_dofs_position`)
and reproduce the underactuation in the **action layer** via the SDK's closed-form 7→16 map (the same pattern the
GR100 already uses: driven+mimic coupled in `firefly_dual.py::command()`, not in the URDF). Reuse the MJCF
primitive boxes as the Genesis collider. → a **conversion**, not a build.

## 3. Integration design (cite OUR files)
- **Mount:** the GR100 attaches at `*_link_6 → *_gripper_base_link`; `*_ee_link` is +0.187 m beyond. The AERO
  hand **replaces the GR100 subtree** — fixed-joint `aero_base_link` to `link_6` (both arms), transform mirroring
  the GR100 +Z-along-wrist convention. The arm `ik.py` is **UNCHANGED** (it targets `ee_link`); we re-define a new
  TCP rigidly fixed to the palm grasp point (the MJCF `grasp_site` at `pos="0.11 0 0.03"` is the ready reference)
  and re-measure `TOOL_IN_EE` once, exactly as the GR100 tool frame was measured.
- **Finger control:** a 7-vector `hand_cmd` (actuator space) → 16 joint targets via the SDK map → `control_dofs_
  position` on the hand dofs. PD gains transcribed from the MJCF. The BatchExecutor's per-env gripper scalar
  (`executor.py` `apply()`, currently 1 driven + 1 mimic) generalises to the 7-vector; the arm path + densify are
  unchanged — the "open/close scalar" becomes an open-pose↔grasp-pose 7-vector interpolation, ramped identically.
  **No change to the no-wait / per-env-termination contract.**
- **Harness extension (never a fork):** new `robots/aero_hand.py` (mount, dof map, the 7↔16 coupling lifted from
  the SDK [Apache-2.0], gains, open/grasp poses, `hand_command`/`clamp_fingers`) + `robots/firefly_dual_aero.py`
  (or a `hand=` arg) swapping the GR100 subtree; `assets/robots/firefly_y6_aero/` (merged URDF + meshes) +
  `robots/bake_aero_colliders.py` (mirror `bake_finger_colliders.py`; emit the MJCF primitive boxes / coacd
  hulls); `skills/hand_grasp.py` (`open_pose`/`power_grasp_pose`/`precision_pinch_pose`/`synthesize_grasp` — reuse
  `orientation_aware_grasp_quat` for the wrist, pick power vs precision + curl depth from `ObjectSpec` extents);
  `skills/penetration.py` UNCHANGED (robot-agnostic — pass the AERO `ignore_pairs`); `dr/` extended with hand-DR;
  a thin `tasks/aero_pickplace.py` (~70 lines) reusing ManipulationStage + full-DR + distractors + go-home + the
  penetration gate + per-env termination.

## 4. Phased plan
- **Phase 0 — asset sim-ready on the Firefly flange.** Build `assets/robots/firefly_y6_aero/` (graft AERO at
  `link_6`, copy meshes, set+calibrate the mount transform); `bake_aero_colliders.py` (primitive boxes +
  `AERO_IGNORE_PAIRS`); `robots/aero_hand.py`. **Gate (first GPU run):** arm-IK parity (ik.py untouched) + clean
  collider render (object-refiner-v2 multi-view) + 0 abnormal penetration at rest/full-close + a finger
  open→close→open cycle.
- **Phase 1 — control + grasp primitive.** `hand_command` via the coupling; `open/power/precision` poses; verify
  a firm power-close cages a cube at 0 abnormal penetration (the firm-grasp HARD rule). **Gate:** firm grasp holds.
- **Phase 2 — first dexterous pick-place** of the existing objects, reusing the no-wait single-trajectory +
  per-env termination + full-DR + collision-#1 harness. **Target the round/thin objects** (a multi-finger cage
  should handle the tennis ball etc.). **Gate:** ≥ the GR100 placed-rate, 0 abnormal pen, smooth max|dq|.
- **Phase 3 — harder:** in-hand reorient (port the MJCF Playground cube-rotation task, sim-to-sim) → then the
  **throw-and-catch tennis-ball parabola** (the headline owner goal; STRETCH — ballistic release + catch under
  Genesis-rigid contact is unproven; stage last).

## 5. Risks + the single biggest open question
- **Tendon-fidelity loss (biggest):** replacing spatial tendons with the rigid 7→16 map loses compliance/
  backdrivability/spring dynamics — fine for position-controlled pick-place, questionable for in-hand + throw-
  catch. Mitigate: tune finger PD/damping from the MJCF; accept the rigid approximation for Phases 0–2.
- Mount transform (one calibration); collider transcription (multi-view render gate); no sim tactile (low impact,
  policies are sensor-light); throw-catch dynamics (stretch); **one-hand-one-gripper vs two-hand** (decide before
  Phase 2 — affects the URDF graft + the executor's per-arm command shape).
- **THE open question:** *is the rigid 7→16 coupling faithful enough for physically-valid data, or do we need a
  Genesis tendon/spring emulation (soft equality + per-joint springs)?* This decides whether Phase 0 is days
  (direct joint drive) or a couple of weeks (tendon emulation).

## 6. Branch + agent strategy
Line C on a **new branch off `main`** (e.g. `genesis_aero_handC`) or a worktree-isolated subagent. Phase 0/1 =
a worktree subagent with a hard gate (compile + arm-IK parity + collider render + finger cycle). Reuse the
**object-refiner** agent for AERO collider verification + the **DR-strategist** for hand-DR. Every new file ships
a `docs/` how-to per the document-features rule.

**Key paths.** Clone: `/home/kaitianchao/Projects/aero-hand-open` (sim: `sim_rl/simulation/mujoco/right_hand.xml`;
URDF: `ros2/src/aero_hand_open_description/urdf/aero_hand_open_right.urdf`; coupling: `sdk/src/aero_open_sdk/
joints_to_actuations.py`). Ours: `robots/firefly_dual.py`, `robots/ik.py`, `skills/{executor,grasp,penetration}.py`,
`tasks/pickplace.py`, `assets/robots/firefly_y6_gr100/dual_firefly_y6_gr100.urdf`.
