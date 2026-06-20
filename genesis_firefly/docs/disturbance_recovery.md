# Disturbance → failure-recovery (the harness + god-mode recovery control)

A capability for the Genesis firefly cube→bowl task that generates **failure-and-recovery** demonstrations:
in some trials the TARGET cube is gently shoved DURING the grasp so the grasp FAILS, the god-mode solver
DETECTS the failure from privileged sim signals, and RECOVERS by replanning (rise, reopen, re-locate the cube,
re-grasp) before continuing to place + go-home. The recorded demo therefore contains **failed-grasp +
recovery + success** — that is the training signal that teaches the policy to detect a disturbed/inaccurate
grasp and replan.

> **This is NOT an LLM subagent.** It is a deterministic injection **HARNESS** (`skills/disturbance.py`) plus a
> god-mode recovery **CONTROL** staged into the task (`tasks/pickplace.py`) — the same first-principles pattern
> as the penetration gate and the 50/50 distractors. The failure-and-recovery data is perfectly reproducible
> from `(seed, probability)`.

Files: `skills/disturbance.py` (the harness), `tasks/pickplace.py` (staged execution + detection + recovery),
`skills/executor.py` (a small additive `during_step` hook — the only motion-path change).

---

## 1. The disturbance harness (`skills/disturbance.py`)

### Per-trial probability — like the 50/50 distractors
A per-env draw decides whether a trial is disturbed, exactly like the distractor `has_dist = rng.rand(N) < 0.5`
pattern:

```python
disturbed = rng.rand(N) < prob        # default prob = 0.34  (~1/3 of trials carry a disturbed grasp)
```

`DisturbanceSpec.sample(cube, N, rng, prob=0.34)` draws — from the SAME stage `rng` so it is reproducible
alongside the DR — the disturbed-env mask plus a per-env in-plane velocity impulse `(vx, vy)` (random direction,
magnitude from `speed_range`). The env var `DISTURB` overrides `prob` (`DISTURB=0` forces it off → the parity
behaviour).

### The injection mechanism (chosen + verified)
A free rigid body (`gs.morphs.Box`, our cube) carries a **6-DOF FREE joint** whose local dofs are
`[0,1,2] = translation (x,y,z)` and `[3,4,5] = rotation` (verified live). We add a small **horizontal velocity
impulse** on dofs `[0,1]` (the in-plane x,y) of the disturbed envs ONLY, BATCHED:

```python
target.set_dofs_velocity(vel_xy[disturbed], dofs_idx_local=[0, 1], envs_idx=where(disturbed))
```

The firm Newton solver then CARRIES that impulse for a few steps and friction brings the cube to rest a few cm
away — a **physical shove, not a teleport jump**. We use a velocity impulse rather than a `set_pos` nudge on
purpose:

- it is PHYSICAL — the cube accelerates, slides under friction, decelerates exactly like a real nudge, so the
  contact/penetration gate stays valid (a teleport would create an instantaneous overlap);
- it composes with whatever the cube is already doing, so it never fights the solver's state.

### Magnitude — gentle, reliably 2–5 cm, never launched
Calibrated on the REAL firm high-friction object table (`scripts/temp/calib_disturbance.py`):

| impulse `|v|` (m/s) | resulting cube XY shift |
|---|---|
| 0.4–0.6 | ~1.1–1.6 cm (too small to break the grasp) |
| **0.70** | **~3.3 cm** |
| **0.80** | **~4.0 cm** |
| **0.95** | **~5.5 cm** |
| 1.2 | ~9.8 cm (too large) |

`z`-change is ~0 throughout (the cube stays on the table; nothing is launched). We therefore use
`speed_range = (0.70, 0.95)` → a reliable **~3–5 cm** lateral shift, enough that the gripper — already committed
to the OLD pre-grasp pose — closes on nothing / bumps the cube, but never a launch.

### Scheduling — DURING the grasp approach/close
The shove must land AFTER the arm has committed to the (now stale) approach pose so the planned grasp genuinely
misses. The task fires the impulse via a `during_step(t, labels_t)` hook (added to `BatchExecutor.run`) the
first time the active arm enters the `at`/`close` window. `inject()` is **latched** (`_fired`) so it shoves
exactly once.

---

## 2. Staged execution + detection + recovery (`tasks/pickplace.py`)

The run is restructured into PHASES on the **one locked motion path** (BatchExecutor + densify). Recovery is
just ADDITIONAL batched phases — **no new motion engine**. All phases share the SAME recording callback
(`on_step`), so the HDF5 demo is the full failed-grasp + recovery + success trajectory. Each phase seeds from
the arm's CURRENT pose (its first waypoint = `cur_tool_pose()`), so phase boundaries stay smooth.

### Phase A — pick (home → pre → at → at → close → lift)
The existing pick waypoints. The disturbance fires during the `at`/`close` window for the disturbed envs, so
the cube slides out from under the committed grasp.

### Detection (god-mode privileged signals)
After Phase A, per env:

```python
grasp_failed = (cube_did_NOT_rise)  OR  (gripper_closed_on_nothing)
  cube_did_NOT_rise      = (cube_z_now - cube_rest_z) < 3 cm
  gripper_closed_on_nothing = (driven_gripper_angle near the empty-close stop) AND (cube far from the EE)
```

read straight from `cube.get_pos()` + the driven gripper joint position (`get_dofs_position()`) + the EE link
position. The empty-close test uses `GR100_MEET - 0.04` (the driven claw only reaches the mechanical empty-close
stop when nothing is between the fingers) AND a cube-to-EE distance > 12 cm (so nothing is held).

### Phase B — recovery (batched, ≤ 2 attempts)
For the FAILED envs: **rise** straight up to a safe height (clear of the table), **REOPEN** the gripper,
**RE-LOCATE** the cube from the sim (`cube.get_pos()`), re-plan the grasp at the NEW pose (reusing the LOCKED
`grasp_quat_at` + `pick_waypoints` builders), and execute pre → at → close → lift again. SUCCESSFUL envs HOLD
their grasp in place (a dwell at the current pose with CLOSE grip) so they ride along untouched. After each
attempt, re-detect; loop up to 2 attempts. An env still failed after the budget keeps `success=False`.

The recovery is fully BATCHED: every env gets a waypoint list each attempt (failed → re-grasp, successful →
hold), padded to a common T by the executor.

### Phase C — place (carry → lower → rel → ret → go_home)
The existing place sequence, for ALL envs, from the current pose (a small lift + re-yaw to the carry
orientation first).

---

## 3. HDF5 attrs (per demo)

In addition to the existing attrs (`success`, `max_penetration_mm`, `penetrating`, `degenerate`,
`has_distractors`, …):

| attr | type | meaning |
|---|---|---|
| `disturbed` | bool | the cube was gently shoved during this trial's grasp |
| `recovered` | bool | `disturbed` AND ended cleanly PLACED (demo holds failed-grasp + recovery + success) |
| `recovery_attempts` | int | how many extra re-grasps the god-mode solver needed (0 if not disturbed/never failed) |

The **penetration gate** (`penetrating`) and **degenerate-settle gate** (`degenerate`) still apply unchanged —
a disturbed demo that ends penetrating is still rejected. `success` is `placed & ~penetrating & ~degenerate`.

Each run prints:

```
[COLLECT] disturbance: k/N disturbed, r/k recovered (placed); recovery_attempts(disturbed)=[...]
```

plus per-phase audit lines (`[COLLECT] disturbance: shoved …`, `[COLLECT] detect after PhaseA: failed=…`,
`[COLLECT] recovery attempt N: still-failed=…`).

---

## 4. Running it

```bash
# WITH disturbance (default prob 0.34)
CUDA_VISIBLE_DEVICES=1 DATA_DIR=/path OUT_DIR=genesis_firefly/output/temp/disturb_run \
  ./.venv/bin/python genesis_firefly/tasks/pickplace.py 16 7

# NO disturbance (parity with the prior locked task)
CUDA_VISIBLE_DEVICES=1 DISTURB=0 ... ./.venv/bin/python genesis_firefly/tasks/pickplace.py 16 7

# tune the probability
CUDA_VISIBLE_DEVICES=1 DISTURB=0.5 ...
```

Controls / env vars: `DISTURB` (per-env disturbance probability; 0 = off, default 0.34), `DATA_DIR`, `OUT_DIR`,
`DIST_DEBUG` (distractor displacement trace). Prereqs: the venv (`./.venv`), a GPU (pin with
`CUDA_VISIBLE_DEVICES`), the Genesis firefly assets. Run from the repo root.

---

## 5. Verification (E = 16, seed 7)

### (a) No-disturbance parity (`DISTURB=0`) — proves the staged refactor didn't regress
```
[COLLECT] executed T=1164 (6 left / 10 right arm)  render+sim 139.2s  max per-step |dq|=0.098 rad
[COLLECT] 16/16 grasped, 16/16 placed, through-wall=0/16
[COLLECT] penetration: max=2.5mm (thresh=7mm), abnormal=0/16
[COLLECT] disturbance: 0/16 disturbed, 0/1 recovered (placed); recovery_attempts(disturbed)=[]
```
16/16 grasp+place, 0 penetration, max|dq| 0.098 rad (< 0.15), T normal → the locked task is reproduced.

### (b) With disturbance (`DISTURB=0.34`)
**Seed 7** (same scene as the parity run):
```
[COLLECT] disturbance: shoved 4 env(s) [0, 2, 7, 12] (prob=0.34, |v|~0.70-0.95 m/s)
[COLLECT] detect after PhaseA: failed=2/16 (of disturbed 2/4; false-flags on clean 0/12)
[COLLECT] recovery attempt 1: still-failed=0/16
[COLLECT] executed T=1798 (6 left / 10 right arm)  render+sim 214.7s  max per-step |dq|=0.100 rad
[COLLECT] 16/16 grasped, 15/16 placed, through-wall=0/16
[COLLECT] penetration: max=4.6mm (thresh=7mm), abnormal=0/16
[COLLECT] disturbance: 4/16 disturbed, 3/4 recovered (placed); recovery_attempts(disturbed)=[0, 1, 0, 1]
```
**Seed 23** (different disturbed set):
```
[COLLECT] disturbance: shoved 7 env(s) [0, 2, 6, 7, 8, 13, 15] (prob=0.34, |v|~0.70-0.95 m/s)
[COLLECT] detect after PhaseA: failed=3/16 (of disturbed 3/7; false-flags on clean 0/9)
[COLLECT] recovery attempt 1: still-failed=0/16
[COLLECT] executed T=1782 (9 left / 7 right arm)  render+sim 226.0s  max per-step |dq|=0.099 rad
[COLLECT] 16/16 grasped, 15/16 placed, through-wall=0/16
[COLLECT] penetration: max=2.7mm (thresh=7mm), abnormal=0/16
[COLLECT] disturbance: 7/16 disturbed, 6/7 recovered (placed); recovery_attempts(disturbed)=[1, 1, 0, 0, 1, 0, 0]
```

**Reading it.** Detection is exact: **0 false-flags on clean grasps** across both seeds (0/12 + 0/9 = 0/21),
and **every detected failed grasp recovered** in 1 attempt (still-failed 0/16 after recovery). ~40–45 % of
shoves produce a FULL miss the gate flags (`recovery_attempts=1`); the rest are caught-but-marginal
(`recovery_attempts=0` — the cube rose > 3 cm so it is correctly NOT a "did-not-rise / empty-close" failure by
the owner's detection definition). **Most disturbed envs recover and place** (seed 7: 3/4 = 75 %; seed 23:
6/7 = 86 %). The single non-recovered disturbed env per run is a *marginal* grasp that rose but slipped at the
place — a legitimate hard edge case the place + penetration gate correctly rejects (`success=False`), exactly
the kind of failure-recovery data the owner wants labelled, not hidden. Motion stays smooth THROUGHOUT incl.
the recovery (max|dq| 0.100 / 0.099 rad < 0.15), penetration 0 abnormal.

A four-view of a disturbed+recovered demo (seed 7, env 12: disturbed → detected failure → rise+reopen+
re-locate+re-grasp → place) is saved to `genesis_firefly/output/temp/disturbance_recovery_demo.mp4`.

---

## 6. Future cases

The same harness generalises by choosing a different **target / trigger-phase / impulse**:

- **Cup bumped over** — target a cup/container, fire a larger impulse (or an angular impulse on dofs `[3,4,5]`)
  so it tips; the solver detects the tip (orientation away from upright) and re-rights / re-grasps.
- **Target shifted mid-transport** — fire during the `carry` phase so the held object is knocked loose; the
  solver detects the drop (object far below the EE) and re-picks from the floor.
- **Heavier/lighter shove distribution** — widen `speed_range` to mix near-misses (1–2 cm, the grasp still
  catches an edge) with full misses, so the policy sees a spectrum of grasp errors.

Each is the same pattern: a privileged, gentle, reproducible sim injection + god-mode detection from privileged
state + a batched recovery built from the locked skill/waypoint builders.
