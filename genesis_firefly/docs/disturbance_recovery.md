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

Files: `skills/disturbance.py` (the harness: random-timing shove + sense-delay), `tasks/pickplace.py` (per-env
execution: approach → close/chase → place-or-recover + detection + natural termination), `skills/executor.py`
(the additive `during_step` hook + an optional `DQ_DEBUG` jerk-localiser — the only motion-path touch).

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

### Disturbed case — `DISTURB>0` (opt-in), still per-env with NO lifted barrier
Three lightweight segments — only because the chase/recovery needs the post-shove pose READ from the sim (the
god-mode privilege). It is **not** a global barrier:

- **seg1 APPROACH** (`home → pre → at → at`, OPEN). The shove fires at each disturbed env's random step inside
  this window; the harness counts the sense-delay from there. The gripper doesn't close yet, so an env informed
  here can ABORT and chase.
- **seg2 CLOSE + LIFT** (chase envs pivot to the cube's NEW sensed pose). **CHASE envs** get the smooth
  singularity-robust re-grasp (§3): rise → reorient → pre → at → close → lift. **Other envs** do a pure-dwell
  close + lift at the current pose; an after-close env's cube has slid away → closes on nothing → fails.
- **Detection (god-mode):** `grasp_failed = did_not_rise OR empty_close`, read from `cube.get_pos()` + the
  driven gripper joint + the EE link (`did_not_rise = rise < 3 cm`; `empty_close = claws met the empty stop AND
  the grasp-centre is far from the EE`).
- **seg3 PLACE-or-RECOVER (one batch, the barrier-killer).** Each env runs its OWN continuous remainder:
  - a **HELD** env runs `place → home` straight away;
  - a **FAILED** env runs `rise(small) → reopen → relocate → re-grasp → lift → place → home` — **one** recovery
    attempt ("rise a bit, try the new pose").

  Both run in the SAME batch, so a held env executes its place→home **concurrently** with the recovery and
  **terminates at its own home** — it never idles in the air through the recovery (this is what replaced the old
  Phase-B retry loop, where every successful env HELD its lifted cube through ≤2 retry iterations of the slowest
  env ≈ the ~5 s mid-air idle the owner flagged). `RETRY_RISE = 0.10 m` keeps the re-grasp in the dexterous
  workspace (no near-singularity straighten — §3). The carry/place orientation follows the achieved grasp quat.

### Per-env natural termination (variable-length demos)
The executor pads every env to the global max T (hold-last-pose) so the slowest env can finish; a faster env
then sits STATIC at home for the tail. That idle-home tail is **not recorded** — the writer trims each env to its
own last MOTION frame (+ a tiny settle margin), detected on the recorded joint stream (the home hold is exactly
static, PD jitter < 1e-4 rad/frame; real motion is >> 1.5e-3). **Demos come out variable-length and that is
natural + accepted** (LeRobot supports it). Because the clean path is one continuous trajectory, this only ever
trims a true static tail — there is no mid-trajectory hold to trim.

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
- **`recovered`** — informed after close and ended PLACED: either the empty close FAILED and a smooth low retry
  re-grasped the new pose (`recovery_attempts > 0`, the failure→recover data), OR the late shove was benign and
  the grasp rode through it (`recovery_attempts == 0`). The attempt count distinguishes the two.
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
[COLLECT] disturbance: shoved K env(s) [...] (prob=p, approach_T=..); before_close(chase)=c after_close(retry)=r
[COLLECT] detect after pick: failed=f/N (disturbed ../..)
[COLLECT] disturbance: d disturbed (c chased, r recovered, f failed)  recovery_attempts(disturbed)=[...]
[COLLECT] per-env demo length (natural termination): min=.. max=.. mean=.. of Tr=.. recframes
```

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

Env vars: `DISTURB` (per-env disturbance probability; **default 0 = off**, set e.g. 0.5 to opt in), `DQ_DEBUG`
(per-phase jerk localiser), `DATA_DIR`, `OUT_DIR`, `DIST_DEBUG` (distractor displacement trace). Prereqs: the
venv (`./.venv`), one GPU (`CUDA_VISIBLE_DEVICES=k`, one sim process per GPU), the Genesis firefly assets. Run
from the repo root.

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

### (a) Clean default (`DISTURB=0`, E=8, seed 7, cube) — the single-trajectory foundation
```
[COLLECT] 8/8 grasped, 8/8 placed, through-wall=0/8
[COLLECT] penetration: max=2.7mm (thresh=7mm), abnormal=0/8
[COLLECT] disturbance: 0 disturbed (0 chased, 0 recovered, 0 failed)
[COLLECT] per-env demo length (natural termination): min=104 max=110 mean=107 of Tr=110 recframes
```
8/8 grasp+place, **penetration 0**, **max |dq| ≈ 0.03 rad** (very smooth), **VARIABLE-LENGTH demos (104–110)** —
each env runs one continuous `pick→place→home` and terminates at its own home, no mid-air hold, no barrier.

### (b) Disturbance opt-in (`DISTURB=0.5`, E=8, seed 7) — both modes appear, NO cross-env wait
```
[COLLECT] disturbance: shoved 5 env(s) [0, 1, 2, 4, 5] (prob=0.5, approach_T=292); before_close(chase)=1 after_close(retry)=4
[COLLECT] detect after pick: failed=3/8 (disturbed 3/5)
[COLLECT] executed T=2091 (3 left / 5 right arm)  max per-step |dq|=0.076 rad (all phases)
[COLLECT] 8/8 grasped, 8/8 placed, through-wall=0/8
[COLLECT] penetration: max=3.0mm (thresh=7mm), abnormal=0/8
[COLLECT] disturbance: 5 disturbed (1 chased, 4 recovered, 0 failed)  recovery_attempts(disturbed)=[1, 0, 1, 1, 0]
[COLLECT] per-env demo length (natural termination): min=140 max=214 mean=171 of Tr=214 recframes
```

**Reading it.**
- **Both modes occur:** 1 before-close CHASE + 4 after-close; detection flagged exactly the 3 genuine
  closed-on-nothing misses; **all recovered → 8/8 placed** in one attempt (`recovery_attempts=[1,0,1,1,0]`).
- **NO cross-env wait — the headline.** Demo lengths span **140 → 214 recframes**: the clean/held envs terminate
  at ~140 while the recovering envs run to 214, **each at its OWN home**. A held env is ~74 recframes (~7 s)
  shorter — it does NOT idle in the air through another env's recovery (the old Phase-B barrier is gone). This
  is the owner's per-env-independence rule, proven in the data.
- **Motion SMOOTH:** all-phase max |dq| **0.076 rad < 0.12** (no near-singularity spike), incl. the chase + the
  recovery (the rise→reorient→descend decomposition, §3). **Penetration 0** abnormal (empty-close finger↔finger
  contact correctly allow-listed).

> Re-verify on the per-env code: `CUDA_VISIBLE_DEVICES=0 DISTURB=0.5 FAST=1 ./.venv/bin/python
> genesis_firefly/tasks/pickplace.py 8 7` (drop `FAST=1` for the photoreal videos).

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
