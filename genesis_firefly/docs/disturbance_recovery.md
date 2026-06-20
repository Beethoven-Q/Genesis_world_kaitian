# Disturbance v2 — collaborative-timing CHASE + smooth failure-RECOVERY (god-mode control)

A capability for the Genesis firefly cube→bowl task that generates **two** failure-mode training signals from
**one** mechanism: a gentle, privileged shove of the TARGET cube at a **random time during the grasp approach**,
sensed by the god-mode solver only after a perception **sense-delay**. Depending on WHEN the solver learns the
cube moved, it responds in one of two collaborative ways:

- **informed BEFORE the gripper closes → CHASE.** The solver ABORTS the current grasp, rises a *little*,
  reorients, and SMOOTHLY re-targets ("chases") the cube to its NEW pose, then closes + lifts there. This is
  **chase-a-moving-object** data.
- **informed only AFTER the close → FAIL + RETRY.** Too late — the grasp already closed on nothing → it FAILS →
  rises a *little* → re-locates + re-grasps the new pose. This is **failure → recover → replan** data.

> **This is NOT an LLM subagent.** It is a deterministic injection **HARNESS** (`skills/disturbance.py`) plus a
> god-mode **CONTROL** staged into the task (`tasks/pickplace.py`) — the same first-principles pattern as the
> penetration gate and the 50/50 distractors. Both data modes are perfectly reproducible from `(seed,
> probability)`.

Files: `skills/disturbance.py` (the harness: random-timing shove + sense-delay), `tasks/pickplace.py` (staged
execution: approach → chase/close → retry → place + detection), `skills/executor.py` (the additive `during_step`
hook + an optional `DQ_DEBUG` jerk-localiser — the only motion-path touch).

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

## 2. Staged execution + god-mode CHASE / RETRY (`tasks/pickplace.py`)

The run is restructured into PHASES on the **one locked motion path** (BatchExecutor + densify). Chase and
recovery are just ADDITIONAL batched phases — **no new motion engine**. All phases share the SAME recording
callback (`on_step`), so the HDF5 demo is the full disturbed + chase/recover + success trajectory. Each phase
seeds from the arm's CURRENT pose, so phase boundaries stay smooth.

### Phase A1 — APPROACH (home → pre → at → at), gripper OPEN
The shove fires at each disturbed env's random step inside this window; the harness counts the sense-delay from
there. The gripper does **not** close yet, so an env informed during A1 can ABORT and chase instead of closing.

### Phase A2 — CLOSE+LIFT (non-chase) / CHASE (chase envs)
- **CHASE envs** (informed before close) do NOT close. They get the smooth singularity-robust re-grasp at the
  cube's NEW sensed pose (see §3): rise → reorient → pre → at → close → lift. The gripper gently *chases* the
  moved cube.
- **All other envs** (clean, or disturbed-but-informed-only-after-close) do a **pure-dwell close + lift** at
  the arm's current pose (hold pose, ramp the gripper). An after-close env's cube has already slid away → this
  closes on nothing → it fails (caught by detection below).

### Detection (god-mode privileged signals)
After Phase A, per env, a grasp FAILED if either:

```python
did_not_rise       = (cube_z_now - cube_rest_z) < 3 cm                    # the cube never came up
empty_close        = (driven_gripper_angle > GR100_MEET - 0.04) AND       # claws met the empty-close stop
                     (cube-to-EE distance > 12 cm)                        # nothing is between the fingers
grasp_failed       = did_not_rise OR empty_close
```

read straight from `cube.get_pos()` + the driven gripper joint + the EE link.

### Phase B — RETRY recovery (batched, ≤ 2 attempts) for the after-close failures
Failed envs: rise a **SMALL** amount (`RETRY_RISE = 0.10 m`, *not* a high safe height), REOPEN, RE-LOCATE the
cube, and re-grasp the new pose via the same smooth re-grasp builder (§3). Successful/chased envs HOLD their
grasp so they ride along untouched. Re-detect after each attempt; loop up to 2. The recovery is fully BATCHED.

### Phase C — place (carry → lower → rel → ret → go_home) for ALL envs
The locked place sequence from the current pose. The carry/place orientation reference follows whichever grasp
actually succeeded (`gqA[i]` is updated by the chase/retry to the achieved grasp quat).

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

Each run prints:

```
[COLLECT] disturbance: shoved K env(s) [...] (prob=p, |v|~0.70-0.95 m/s, approach_T=.., sense_delay~15-30 steps)
[COLLECT] disturbance branch: before_close(chase)=c [...]  after_close(retry)=r [...]
[COLLECT] detect after PhaseA: failed=f/N (of disturbed ../..; chased-still-failed 0/..; false-flags on clean 0/..)
[COLLECT] recovery attempt 1: still-failed=0/N
[COLLECT] disturbance: d disturbed (c chased, r recovered, f failed)  recovery_attempts(disturbed)=[...]
```

---

## 5. Running it

```bash
# WITH disturbance (default prob 0.34)
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/path OUT_DIR=genesis_firefly/output/temp/disturb_run \
  ./.venv/bin/python genesis_firefly/tasks/pickplace.py 16 7

# NO disturbance (parity with the prior locked task)
CUDA_VISIBLE_DEVICES=0 DISTURB=0 ... ./.venv/bin/python genesis_firefly/tasks/pickplace.py 8 7

# tune the chase/retry mix probability
CUDA_VISIBLE_DEVICES=0 DISTURB=0.5 ...

# localise a jerk (per-phase worst |dq| + the segment it occurred on)
CUDA_VISIBLE_DEVICES=0 DISTURB=0.5 DQ_DEBUG=1 ...
```

Env vars: `DISTURB` (per-env disturbance probability; 0 = off, default 0.34), `DQ_DEBUG` (per-phase jerk
localiser), `DATA_DIR`, `OUT_DIR`, `DIST_DEBUG` (distractor displacement trace). Prereqs: the venv (`./.venv`),
GPU 0 (`CUDA_VISIBLE_DEVICES=0`), the Genesis firefly assets. Run from the repo root.

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

### (a) No-disturbance parity (`DISTURB=0`, E=8, seed 7) — proves the staged refactor didn't regress
```
[COLLECT] executed T=1121 (3 left / 5 right arm)  render+sim 112.8s  max per-step |dq|=0.084 rad (all phases)
[COLLECT] 8/8 grasped, 8/8 placed, through-wall=0/8
[COLLECT] penetration: max=2.9mm (thresh=7mm), abnormal=0/8
[COLLECT] disturbance: 0 disturbed (0 chased, 0 recovered, 0 failed)
```
8/8 grasp+place, **penetration 0**, **max |dq| 0.084 rad** (< 0.12), T normal → the locked task is reproduced.

### (b) With disturbance (`DISTURB=0.5`, E=16) — both data modes appear, motion stays smooth
**Seed 7:**
```
[COLLECT] disturbance: shoved 7 env(s) [0, 2, 5, 7, 12, 14, 15] (prob=0.5, |v|~0.70-0.95 m/s, approach_T=316, sense_delay~15-30 steps)
[COLLECT] disturbance branch: before_close(chase)=5 [0, 5, 7, 12, 15]  after_close(retry)=2 [2, 14]
[COLLECT] detect after PhaseA: failed=2/16 (of disturbed 2/7; chased-still-failed 0/5; false-flags on clean 0/9)
[COLLECT] recovery attempt 1: still-failed=0/16
[COLLECT] executed T=2239 (6 left / 10 right arm)  render+sim 400.5s  max per-step |dq|=0.096 rad (all phases)
[COLLECT] 16/16 grasped, 16/16 placed, through-wall=0/16
[COLLECT] penetration: max=2.9mm (thresh=7mm), abnormal=0/16
[COLLECT] disturbance: 7 disturbed (5 chased, 2 recovered, 0 failed)  recovery_attempts(disturbed)=[0, 1, 0, 0, 0, 1, 0]
```
**Seed 23:**
```
[COLLECT] disturbance: shoved 10 env(s) [0, 2, 3, 4, 6, 7, 8, 13, 14, 15] (prob=0.5, ..., approach_T=309, sense_delay~15-30 steps)
[COLLECT] disturbance branch: before_close(chase)=5 [0, 2, 3, 8, 14]  after_close(retry)=5 [4, 6, 7, 13, 15]
[COLLECT] detect after PhaseA: failed=1/16 (of disturbed 1/10; chased-still-failed 0/5; false-flags on clean 0/6)
[COLLECT] recovery attempt 1: still-failed=0/16
[COLLECT] executed T=2086 (9 left / 7 right arm)  render+sim 411.9s  max per-step |dq|=0.098 rad (all phases)
[COLLECT] 16/16 grasped, 14/16 placed, through-wall=0/16
[COLLECT] penetration: max=2.6mm (thresh=7mm), abnormal=0/16
[COLLECT] disturbance: 10 disturbed (5 chased, 3 recovered, 2 failed)  recovery_attempts(disturbed)=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

**Reading it.**
- **Both modes occur, every run.** Seed 7: **5 CHASE + 2 RETRY**; seed 23: **5 CHASE + 5 after-close** (3
  recovered + 2 failed). Some disturbed envs hit BEFORE-close → smooth CHASE → success; some hit AFTER-close →
  fail → smooth retry → success — exactly as specified.
- **Detection is exact:** 0 false-flags on clean grasps (0/9 + 0/6), 0 chased-still-failed (0/5 + 0/5); every
  *detected* failure recovered in 1 attempt.
- **Motion is SMOOTH everywhere:** global all-phase max |dq| **0.096 / 0.098 rad < 0.12** (no near-singularity
  spike), incl. the chase and the retry. The smooth-retry fix is confirmed by `DQ_DEBUG`: the phase-B retry
  worst jump fell **0.198 → 0.068 rad** after the rise→reorient→descend decomposition.
- **Penetration 0** abnormal in every run (the empty-close finger↔finger contact is correctly allow-listed).
- The 2 **`failed`** envs in seed 23 are marginal late-shove grasps the place/penetration gate correctly
  rejects (`success=False`) — labelled hard-edge data, not hidden.

### Showcase clips
- `output/temp/disturb_chase_demo.mp4` — a **before-close pivot/CHASE** (seed 7 env 0: shove sensed before
  close → abort + small rise + reorient + chase the new pose → place).
- `output/temp/disturb_retry_demo.mp4` — an **after-close FAIL → smooth RETRY** (seed 7 env 2: shove sensed
  after close → empty close fails → smooth low retry-lift + reorient + re-grasp the new pose → place).

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
