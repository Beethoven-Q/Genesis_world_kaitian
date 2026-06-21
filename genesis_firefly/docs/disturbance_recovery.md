# Disturbance v2 — collaborative-timing CHASE + smooth failure-RECOVERY (god-mode control)

An **opt-in augmentation** for the Genesis firefly pick→bowl task that generates **two** failure-mode training
signals from **one** mechanism: a gentle, privileged shove of the TARGET at a **random time during the grasp
approach**, sensed by the god-mode solver only after a perception **sense-delay**. Depending on WHEN the solver
learns the target moved, it responds in one of two collaborative ways:

- **informed BEFORE the gripper closes → CHASE.** The solver ABORTS the current grasp, rises a *little*,
  reorients, and SMOOTHLY re-targets ("chases") the target to its NEW pose, then closes + lifts there. This is
  **chase-a-moving-object** data.
- **informed only AFTER the close → FAIL + RECOVER.** Too late — the grasp already closed on nothing → it FAILS →
  rises a *little* → re-locates + re-grasps the new pose. This is **failure → recover → replan** data.

> **This is NOT an LLM subagent.** It is a deterministic injection **HARNESS** (`skills/disturbance.py`) plus a
> god-mode **CONTROL** in the task (`tasks/pickplace.py`) — the same first-principles pattern as the penetration
> gate and the 50/50 distractors. Both data modes are perfectly reproducible from `(seed, probability)`.

> **`DISTURB` DEFAULTS TO 0 (off).** The default pick-place run is the CLEAN single continuous trajectory (the
> foundation + the bulk fine-tune data + what the grasp-solving runs use). This harness is an explicit
> augmentation you opt into with `DISTURB>0`. **Every trial is fully independent — there is NO cross-env
> barrier:** a held env NEVER idles in the air while another env recovers (owner HARD rule). The disturbance is
> a per-env EXTENSION of that env's own trajectory; other envs finish and terminate at their own home (demos are
> variable-length — see §2).

> **v3 (2026-06-20) — SINGLE-CONTINUOUS-PASS re-architecture (the NO-HOLD fix).** The disturbance path used to
> run STAGED seg1(approach)/seg2(close+lift)/seg3(place+recover). In seg2 the NORMAL envs finished close+lift
> early and **FROZE LIFTED IN THE AIR (~4–5 s)** while the chase envs did their longer pivot — a per-env barrier
> the owner forbids. v3 pre-plans **each env's FULL trajectory up front** and runs them in **ONE continuous
> pass**, exactly like the clean path: a clean env terminates early (shorter demo), a chased/recovered env's
> trajectory is simply LONGER, but **no env ever freezes waiting**. The mid-air hold is gone (verified — §6: the
> worst mid-trajectory static-arm run is ~5–6 recorded frames, the natural pre-grasp/close settle, vs the old
> 40–50+ frame frozen-lift). To pre-plan the chase/recover re-grasp BEFORE the run, the shoved resting pose is
> **PREDICTED at build time** (we own the impulse): `rest = fire_xy + unit(v)·|v|²/(2·µ·g)` — see §1.

Files: `skills/disturbance.py` (the harness: random-timing shove + sense-delay + **the build-time shoved-pose
prediction + chase/after-close resolution**), `tasks/pickplace.py` (the **single-pass per-env trajectory
assembly**: clean / chase / after-close-recover, all pre-planned + run in ONE `run_phase` + natural
termination), `skills/executor.py` (the additive `during_step` hook + an optional `DQ_DEBUG` jerk-localiser —
the only motion-path touch).

---

## 1. The disturbance harness (`skills/disturbance.py`)

### Per-trial probability — like the 50/50 distractors
A per-env draw decides whether a trial is disturbed, exactly like the distractor `has_dist = rng.rand(N) < 0.5`:

```python
disturbed = rng.rand(N) < prob        # default prob = 0.34;  env var DISTURB overrides (DISTURB=0 -> off)
```

`DisturbanceSpec.sample(cube, N, rng, prob=...)` draws — from the SAME stage `rng`, so it is reproducible
alongside the DR — the disturbed mask, a per-env in-plane velocity impulse `(vx, vy)`, a per-env **sense-delay**
(control steps), and a per-env **fire-fraction** within the approach window.

### The injection mechanism (chosen + verified, unchanged from v1)
The cube is a free rigid body (`gs.morphs.Box`) with a 6-DOF FREE joint (`[0,1,2]=xyz, [3,4,5]=rot`). We add a
small **horizontal velocity impulse** on dofs `[0,1]` of the disturbed envs ONLY, BATCHED:

```python
target.set_dofs_velocity(vel_xy[due], dofs_idx_local=[0, 1], envs_idx=where(due))
```

The firm Newton solver carries the impulse and friction brings the cube to rest a few cm away — a **physical
shove, not a teleport** (so the penetration gate stays valid; a velocity impulse composes with the solver state
where a `set_pos` would fight it). Magnitude `speed_range = (0.70, 0.95) m/s` → a reliable **~3–5 cm** lateral
shift (`z`-change ≈ 0; nothing is launched). Calibration table: see §6.

### Random approach-timing + sense-delay (the v2 timing model)
The shove fires at a **random control step inside the APPROACH window** (the `pre`/`at` segments, before the
close), not at a fixed point. The task drives the harness one control step at a time:

```python
dspec.reset_window(T_A1)              # begin the approach window of T_A1 control steps; resolve each env's fire-step
for t in approach:  dspec.tick(t)     # fires the latched impulse at each disturbed env's random fire-step
informed = dspec.informed_by(t)       # True once shove fired AND sense_delay steps elapsed (perception latency)
```

- **`sense_delay`** ∈ `(15, 30)` control steps ≈ **0.15–0.30 s** at the 100 Hz control rate — simulating
  perception latency. The solver only *learns* the new pose `sense_delay` steps after the shove.
- **fire-fraction bands.** The approach is long (~3 s) relative to the sense-delay (~0.2 s), so a uniform fire
  time would almost always be sensed *before* the close (everything would chase). To reliably generate BOTH
  modes, each disturbed env is assigned (prob `late_prob=0.5`) to an **early band** `(0.20, 0.65)` → sensed
  before close → CHASE, or a **late band** `(0.93, 1.0)` → the sense-delay pushes "informed" *past* the close →
  FAIL + RETRY. The exact fire-fraction within the chosen band is still random (timing varies trial-to-trial).
  `late_prob` is the single knob for the chase-vs-retry mix.

The decision is read once at the end of the approach: `informed_before_close = informed_by(T_A1 - 1)`. That
boolean selects CHASE vs RETRY per env.

### Build-time shoved-pose PREDICTION (v3, what makes the single continuous pass possible)
For the single-pass re-architecture (§2) the chase/recover re-grasp must be PRE-PLANNED — but the shove physically
lands DURING the run. Because the shove is a god-mode impulse WE control (known `vel_xy` + the per-env object
friction), we PREDICT where the object comes to rest, at BUILD TIME, with a Coulomb-friction slide:

```python
rest_xy = fire_xy + unit(vel_xy) * |vel_xy|**2 / (2 * mu * g)     # DisturbanceSpec.predict_shoved_xy(...)
```

- `fire_xy` = the object's settled grasp-centre xy (the shove fires early in the approach, before the arm gets
  near, so the object is still at its settled pose when it fires).
- `mu` = the **effective slide friction**, **CALIBRATED** (not the nominal spec friction): `scripts/temp/
  calib_shove_predict.py` fires the real shove on the firm table and fits `mu`. At **`mu = 0.85`** the predicted
  rest xy matches the REAL settled xy to **~1.2 cm mean / 1.6 cm max** — well within the open-claw span, so the
  re-grasp planned at the predicted pose still cages the real cube. (`mu = 1.0`, the nominal cube friction,
  slightly UNDER-predicts the slide → the re-grasp lands a touch short.) Override with `SHOVE_MU`.
- The shove **still fires** via the `during_step` hook at the per-env fire-step (a REAL physics slide); we only
  PREDICT where it lands so the whole per-env path is known up front.

`chase`-vs-`after-close` is ALSO resolved at build time (`DisturbanceSpec.will_be_informed_before_close
(approach_T)`) — it depends only on the fire-fraction band + the sense-delay, no sim read — so the matching
trajectory SHAPE is pre-planned per env.

---

## 2. Execution model — per-env, NO cross-env barrier (`tasks/pickplace.py`)

Everything runs on the **one smooth motion path** (BatchExecutor + densify) — chase and recovery add no new
motion engine. The load-bearing rule (owner): **each trial is fully independent; no env ever idles/holds waiting
for another env.** Two cases:

### Clean case — `DISTURB=0` (the default, the foundation)
One **single continuous per-env trajectory**, run in ONE `run_phase`:

```
home → pre → at → close → lift → carry → lower → release → GO HOME
```

There are **no staged phases**, so there is structurally no point at which a fast env holds a lifted object
waiting for a slow env (the old idle-in-the-air bug). Each env's stream densifies to its own length; faster envs
reach home first and the recorder **trims each env's idle-home tail** → **variable-length demos** (per-env
natural termination — see below). This is the path the grasp-solving runs and the bulk fine-tune collections use.

### Disturbed case — `DISTURB>0` (opt-in): a SINGLE continuous per-env trajectory, ZERO mid-trajectory holds
**No staging, no barrier, no sim read mid-run.** Each env's FULL trajectory is pre-planned at build time (using
the shoved-pose PREDICTION, §1) and the whole batch runs in ONE `run_phase` — exactly like the clean path. Three
trajectory SHAPES, chosen per env at build time:

- **undisturbed** — `home → pre → at → close → lift → carry → lower → release → home` (byte-identical to the
  clean `pick_wps + place_tail`).
- **chased** (informed before the close) — `home → pre → at(old) → [re-aim: reorient → rise → pre → at] →
  close → lift → place → home`. The first approach reaches the OLD pose but **never closes there** (the gripper
  stays OPEN); the arm immediately re-aims to the PREDICTED shoved pose and closes THERE. The first approach uses
  a **single** `at` (not the clean pick's `at,at` pre-close settle) — the chase never closes at the old pose, so
  a settle dwell there would be a pointless mid-trajectory HOLD. A near-no-op re-aim `reorient` (when the
  re-selected tilt ≈ the original) is **skipped** so the chase flows `at(old) → rise` continuously.
- **after-close** (informed only after the close) — `home → pre → at → close(EMPTY, old) → lift(empty) →
  [recover: rise → reorient → pre → at → close → lift] → place → home`. The first approach genuinely **closes on
  nothing** at the old (now-vacated) pose — a real empty close in the sim (the failed-grasp signal we want) —
  then the recover re-grasp picks up the cube at the PREDICTED pose. `recovery_attempts = 1`.

The chase/recover re-grasp reuses the LOCKED `approach_close_lift_wps` builder (`reorient → rise → pre → at →
close → lift`, each segment pure-translation or pure-rotation so the warm IK never flips a branch — §3), with the
grasp/carry tilt **re-selected at the predicted shoved pose** (the same wrist-margin relax ladders → natural
posture). `RETRY_RISE = 0.10 m` keeps the re-grasp in the dexterous workspace.

Because every env runs its own pre-planned stream to completion in the SAME batch, a clean/chase env reaches home
and **terminates early** (shorter demo) while a recovering env runs longer — **no env ever freezes lifted waiting
for another** (the old seg2 mid-air hold is structurally impossible now: there is no point where a finished env
holds while another runs). **Both modes ship hold-free** — the after-close mode was NOT deferred.

### Per-env natural termination (variable-length demos)
The executor pads every env to the global max T (hold-last-pose) so the slowest env can finish; a faster env
then sits STATIC at home for the tail. That idle-home tail is **not recorded** — the writer trims each env to its
own last MOTION frame (+ a tiny settle margin), detected on the recorded joint stream (the home hold is exactly
static, PD jitter < 1e-4 rad/frame; real motion is >> 1.5e-3). **Demos come out variable-length and that is
natural + accepted** (LeRobot supports it). Because **both** paths (clean AND disturbed) are now one continuous
per-env trajectory, this only ever trims a true static HOME tail — there is no mid-trajectory hold to trim (the
NO-HOLD checker confirms the worst mid-trajectory static-arm run is ~5–6 frames, the pre-grasp/close settle).

---

## 3. Smooth low retry/chase lift — how the near-singularity jerk was eliminated

**The v1 problem (owner watched the demo):** on a failed grasp the arm rose to a high "safe height"
(`SAFE_RISE = 0.20 m`) → the arm nearly straightened → near a singularity → a visible **jerk** on the retry.

**Two fixes, both verified by the per-step jerk gate (`max |dq|`):**

1. **Small lift (`RETRY_RISE = 0.10 m`).** Rise only ~10 cm — just enough to clear the cube + free the view —
   keeping the elbow bent and the arm in its dexterous workspace, never straightening toward singularity.
   Measured: the retry EE apex tops out at ee_z ≈ 0.63 m (vs the v1 ~0.72 m straight-arm height).

2. **Decompose the re-grasp into pure-translation / pure-rotation segments** (`approach_close_lift_wps`). The
   re-grasp must change BOTH the wrist orientation (the orientation-aware grasp quat re-tilts toward the moved
   cube) AND the position. Doing both in one `rise → pre` segment let the warm-started IK cross a branch near
   the top of the rise — a single-step **0.198 rad** spike (localised with `DQ_DEBUG` to phase B, seg `→pre`).
   The fix builds **rise (keep orientation) → reorient at the apex (pure rotation, position dwelt + SLERP'd) →
   descend to pre (keep new orientation) → at → close → lift**. Each segment is now pure-translation or
   pure-rotation, so the IK never has to flip a branch.

   **Result: the phase-B retry max |dq| dropped 0.198 → 0.068 rad**, and the global all-phase max |dq| is
   **≤ 0.10 rad** (smooth everywhere, incl. the chase + the retry). The chase (Phase A2) was already smooth
   (~0.05 rad) because the arm had not yet closed.

> `DQ_DEBUG=1` makes the executor report, per phase, the worst per-step `|dq|`, the step it occurred at, and the
> `from→to` waypoint-label segment — the tool used to localise and confirm the fix. It changes no motion.

A `pure-dwell` close (hold the arm's current pose, ramp only the gripper) is used for the non-chase close, so
the jaws never re-press into the cube while closing — this is also what keeps the **parity** penetration at 0.

---

## 4. HDF5 attrs (per demo)

In addition to the existing attrs (`success`, `max_penetration_mm`, `penetrating`, `degenerate`,
`has_distractors`, `dr_*`, …):

| attr | type | meaning |
|---|---|---|
| `disturbed` | bool | the cube was gently shoved at a random time during this trial's grasp approach |
| `disturb_phase` | str | `"before_close"` (→ chase) · `"after_close"` (→ fail+retry) · `"none"` (undisturbed) |
| `disturb_outcome` | str | `"chased"` · `"recovered"` · `"failed"` · `"none"` (see below) |
| `recovery_attempts` | int | extra re-grasps the solver needed (0 for a clean chase or a benign late shove) |
| `recovered` | bool | *(legacy, kept)* `disturbed AND ended cleanly placed` (chased OR recovered) |

`disturb_outcome` semantics:
- **`chased`** — informed before close: aborted the close, pivoted/chased to the new pose, ended PLACED
  (moving-object data). `recovery_attempts == 0`.
- **`recovered`** — informed after close and ended PLACED: the (pre-planned) empty close at the old pose grabbed
  nothing → a smooth low recover re-grasp picked the cube up at the PREDICTED pose (`recovery_attempts == 1`, the
  failure→recover data).
- **`failed`** — disturbed but did NOT end cleanly placed — a legit hard edge case the place/penetration gate
  correctly REJECTS (`success=False`), labelled rather than hidden.
- **`none`** — not disturbed.

The **penetration gate** (`penetrating`) and **degenerate-settle gate** (`degenerate`) still apply unchanged;
`success = placed & ~penetrating & ~degenerate`. **Empty-close allow-list:** when the gripper closes on nothing
(the failed-grasp signal we WANT), the two opposing claws of the *same* gripper meet and the firm solver reports
their mutual overlap (~1 cm). That is geometrically EXPECTED designed contact (the same category as the
documented ~3 mm finger-into-cube contact skin), **not** an abnormal penetration defect, so the task adds ONLY
the same-gripper L-claw↔R-claw geom pairs to the penetration tracker's `ignore_pairs` (computed by link name).
ALL other self-contact (arm-into-arm, claw-into-other-gripper) still counts.

Each run prints (the per-env demo-length line appears on EVERY run, disturbed or not):

```
[COLLECT] disturbance v3 (single-pass): disturbed d env(s) [...] (prob=p, approach_T=..); before_close(chase)=c after_close(recover)=r
[COLLECT] disturbance: shoved K env(s) [...] (fire steps=[...])
[COLLECT] disturbance: d disturbed (c chased, r recovered, f failed)  recovery_attempts(disturbed)=[...]
[COLLECT] per-env demo length (natural termination): min=.. max=.. mean=.. of Tr=.. recframes
```
(`DISTURB_DIAG=1` adds a per-disturbed-env line comparing the PREDICTED shoved xy to the cube's final xy.)

---

## 5. Running it

```bash
# CLEAN single-trajectory pick-place (the DEFAULT -- DISTURB unset == 0)
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/path OUT_DIR=genesis_firefly/output/temp/run \
  ./.venv/bin/python genesis_firefly/tasks/pickplace.py 8 7

# OPT IN to failure-recovery augmentation (per-env, no barrier)
CUDA_VISIBLE_DEVICES=0 DISTURB=0.5 ... ./.venv/bin/python genesis_firefly/tasks/pickplace.py 16 7

# localise a jerk (per-phase worst |dq| + the segment it occurred on)
CUDA_VISIBLE_DEVICES=0 DISTURB=0.5 DQ_DEBUG=1 ...
```

Env vars: `DISTURB` (per-env disturbance probability; **default 0 = off**, set e.g. 0.5 to opt in), `SHOVE_MU`
(shoved-pose prediction friction; default 0.85 calibrated), `DISTURB_DIAG` (per-env predicted-vs-actual shoved
pose), `FAST` (hdf5-only, no videos — for fast iteration), `DQ_DEBUG` (per-phase jerk localiser), `DATA_DIR`,
`OUT_DIR`, `DIST_DEBUG` (distractor displacement trace). Prereqs: the venv (`./.venv`), one GPU
(`CUDA_VISIBLE_DEVICES=k`, **one sim process per GPU**), the Genesis firefly assets. Run from the repo root.

---

## 6. Verification (GPU 0, real runs)

### Shove-magnitude calibration (firm high-friction object table)
| impulse `|v|` (m/s) | cube XY shift |
|---|---|
| 0.4–0.6 | ~1.1–1.6 cm (too small to break the grasp) |
| **0.70** | **~3.3 cm** |
| **0.80** | **~4.0 cm** |
| **0.95** | **~5.5 cm** |
| 1.2 | ~9.8 cm (too large) |

`z`-change ≈ 0 throughout. We use `(0.70, 0.95)` → a reliable ~3–5 cm shift.

### Shoved-pose PREDICTION calibration (`scripts/temp/calib_shove_predict.py`, GPU 1, cube)
Predicted rest xy (`d = |v|²/(2µg)`) vs the REAL settled xy, per `mu`:

| `mu` | predicted slide (mean) | **prediction error** (mean / max) |
|---|---|---|
| 0.70 | 4.7 cm | 1.7 / 2.6 cm |
| 0.80 | 4.1 cm | 1.3 / 1.9 cm |
| **0.85** | **3.9 cm** | **1.2 / 1.6 cm** ← default |
| 1.00 | 3.3 cm | 1.1 / 1.7 cm (under-predicts the slide) |

Actual slide mean 3.5 cm. We use **`mu = 0.85`** (`SHOVE_MU` override) — the re-grasp planned at the predicted
pose lands within ~1.2 cm of the real cube, inside the open-claw span.

### (a) Clean default (`DISTURB=0`, E=8, seed 7, cube) — the single-trajectory foundation (UNCHANGED)
```
[COLLECT] 8/8 grasped, 8/8 placed, through-wall=0/8
[COLLECT] penetration: max=2.7mm (thresh=7mm), abnormal=0/8
[COLLECT] disturbance: 0 disturbed (0 chased, 0 recovered, 0 failed)
[COLLECT] per-env demo length (natural termination): min=91 max=99 mean=96 of Tr=99 recframes
```
8/8 grasp+place, **penetration 0**, **max |dq| = 0.020 rad** (very smooth), **VARIABLE-LENGTH demos (91–99)** —
each env runs one continuous `pick→place→home` and terminates at its own home, no mid-air hold, no barrier. The
trajectory is **byte-identical** to before the v3 disturbance re-architecture (only the disturbance branch + the
FAST write-skip changed; the clean branch is untouched).

### (b) Disturbance opt-in (`DISTURB=0.5`, E=16, seed 7) — both modes appear, single pass, NO HOLD
```
[COLLECT] disturbance v3 (single-pass): disturbed 7 env(s) [...] (prob=0.5, approach_T=304); before_close(chase)=5 after_close(recover)=2
[COLLECT] disturbance: shoved 7 env(s) [...] (fire steps=[200, 266, 316, 299, 343, 266, 287])
[COLLECT] executed T=1584 (6 left / 10 right arm)  max per-step |dq|=0.020 rad (all phases)
[COLLECT] 15/16 grasped, 15/16 placed, through-wall=0/16
[COLLECT] penetration: max=3.1mm (thresh=7mm), abnormal=0/16
[COLLECT] disturbance: 7 disturbed (4 chased, 2 recovered, 1 failed)  recovery_attempts(disturbed)=[0, 1, 0, 0, 0, 1, 0]
[COLLECT] per-env demo length (natural termination): min=95 max=160 mean=113 of Tr=160 recframes
```

**Reading it.**
- **Both modes occur, both hold-free:** 5 before-close CHASE + 2 after-close RECOVER (the after-close mode was
  NOT deferred). 7 disturbed → **4 chased + 2 recovered = 6 placed**, 1 failed (a hard edge-of-workspace shove,
  correctly labelled `failed` and rejected). All 16 → 15/16 placed.
- **NO-HOLD gate (the load-bearing one) — `scripts/temp/check_no_hold.py`.** Per demo, the longest run of
  consecutive recorded frames with the active-arm joints static (max |Δj| < 1e-3) BEFORE home arrival, excluding
  the home tail + the gripper-ramp dwells. **Worst mid-trajectory static-arm run across all 16 demos = 6 frames**
  (a chase's pre-grasp/close settle), per-demo 0–6. A frozen LIFTED arm (the old seg2 barrier) would be 40–50+
  frames — **ELIMINATED.** The disturbed (chase/recover) demos show the SAME tiny static profile as the clean
  demos → no extra hold was introduced.
- **Per-env termination:** demo lengths **95 → 160** (clean/chase envs terminate ~95–120, recovering envs run to
  ~160), each at its OWN home — no env idles waiting.
- **Posture (natural):** worst wrist |j4| = 1.445 (≤ ~1.45), worst-low elbow j3 = 0.961 (≥ ~1.0) — both from
  CLEAN envs; the disturbed re-grasps are well inside (j4 ≤ 1.37, j3 ≥ 1.02).
- **Motion SMOOTH:** all-phase max |dq| **0.020 rad** (no near-singularity spike), incl. the chase + the recovery
  (the rise→reorient→descend decomposition, §3). **Penetration 0** abnormal (empty-close finger↔finger contact
  correctly allow-listed).

> Re-verify: `CUDA_VISIBLE_DEVICES=1 DISTURB=0.5 FAST=1 ./.venv/bin/python genesis_firefly/tasks/pickplace.py 16 7`
> (drop `FAST=1` for the photoreal videos), then
> `./.venv/bin/python genesis_firefly/scripts/temp/check_no_hold.py <DATA_DIR>/demos.hdf5`.

---

## 7. Future cases

The same harness generalises by choosing a different **target / trigger-phase / impulse**:

- **Future: rotating Lazy-Susan moving-target grasp (owner roadmap A).** Put the target on a rotating turntable;
  the solver reads the real-time pose every control step and CHASES it continuously (not a single re-plan) —
  a true moving-object grasp. The v2 sense-delay + smooth re-target builder are the seed for this: replace the
  one-shot `informed_by` decision with a per-step pose read feeding a rolling re-target through the same
  pure-translation/pure-rotation, branch-stable IK path so the gripper tracks a continuously moving object.
- **Cup bumped over** — angular impulse on dofs `[3,4,5]` tips it; detect the tip (orientation away from
  upright) → re-right / re-grasp.
- **Target shifted mid-transport** — fire during `carry` so the held object is knocked loose; detect the drop
  (object far below the EE) → re-pick from the floor.
- **Heavier/lighter shove distribution** — widen `speed_range` to mix near-misses with full misses.

Each is the same pattern: a privileged, gentle, reproducible sim injection + god-mode detection from privileged
state + a batched chase/recovery built from the locked skill/waypoint builders.
