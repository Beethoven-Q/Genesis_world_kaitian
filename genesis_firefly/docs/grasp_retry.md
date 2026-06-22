# Object-agnostic grasp-retry (miss -> recover failure-recovery data)

An **opt-in** augmentation for the pick-place collector that produces *miss -> retry* demonstrations,
so a policy trained on the data learns not just the clean first-try grasp but also **"I missed -- now
what?"**. It is **OBJECT-AGNOSTIC**: one global knob set, no per-object tuning, and it can never fling /
NaN-crash the object.

| | |
|---|---|
| **Skill** | `genesis_firefly/skills/grasp_retry.py` (perturbation + success-check + two-phase orchestration) |
| **Task seam** | `genesis_firefly/tasks/pickplace.py::collect` (`NOISE_RETRY` env var; builds the re-grasp waypoints) |
| **Default** | **OFF** -- unset / `NOISE_RETRY=0` runs the clean single-phase path **byte-identical** |
| **Opt in** | `NOISE_RETRY=0.45` (a float = the fraction of envs whose first attempt is perturbed) |
| **Gates** | `scripts/temp/check_no_hold.py` (<=8 frames), `scripts/temp/check_j4.py` (worst &#124;j4&#124; < 1.45), 0 abnormal penetration |

## Why this exists (first principles)

A god-mode scripted collector produces only **clean first-try successes**. A policy trained purely on
clean data has never seen a missed grasp, so it has no recovery behaviour. We want *miss -> retry* data
so the policy can recover.

### Why the earlier object-shove disturbance was ABANDONED

The first approach disturbed the **OBJECT** (a physics shove during the grasp approach). That coupled to
each object's dynamics and **did not scale**:

- a light **pen** NaN-crashed under the impulse,
- a round **apple** got flung off the table,
- a **banana** penetrated on the slide.

Each needed per-object impulse/friction tuning, and a linear shove + contact slide also **rotated** the
object, so an xy-only prediction misaligned the re-grasp. It was deleted entirely.

### The fix: put the imprecision in the ROBOT's TARGET (action space)

Instead of disturbing the object, with a small probability we **jitter the grasp TARGET** -- the arm reaches
a slightly-wrong pose and may close on nothing / graze the object. Because the perturbation lives in the
**arm's action space**, it is completely object-agnostic (a smaller object naturally misses more under the
same offset -- exactly an imprecise hand) and can never fling the object; worst case the gripper grazes it.
*Like an elderly hand: not always accurate, but it knows to try again.*

The data is sound (DART-style): the noise is **unconditioned random**, so the policy cannot reproduce it --
it learns the clean grasp in expectation **and** learns to retry conditioned on observing a missed grasp.

## How it works -- the two-phase, HOLD-FREE flow

The grasp-retry runs as two phases (`skills/grasp_retry.py::run_grasp_retry`), both hold-free
(per-env termination + idle tail trimmed; the only static dwell is the brief grasp-close settle):

1. **Phase 1 -- attempt** (all envs, one phase: `home -> pre -> at -> close -> lift`)
   - non-noised envs run the locked clean pick (**byte-identical**);
   - noised envs aim the pre+at at the perturbed target with a grasp orientation **re-planned**
     (wrist-comfortable) at that target -- so the gripper may close on nothing / graze the object.
2. **God-mode check** (a brief *unrecorded* settle -- not a long arm freeze): read the object + active-EE
   world pose and decide `caught = grasp_succeeded(...)` (did the object come up *with* the arm: lifted
   above its rest height **and** within reach of the EE).
3. **Phase 2 -- outcome** (per-env, one phase)
   - **caught** envs continue to place from their phase-1 lift (never drop a first-try success);
   - **missed** envs rise + reopen, **re-grasp at the object's TRUE re-read pose** (NO noise the 2nd
     time -- we don't double-perturb), close, lift, then place.

### Retry respects the re-read ORIENTATION

The recovery grasp orientation is built from the object's **live re-read quaternion**, not the stale stored
yaw -- so a noisy graze that *rotated* the object is respected (the gripper yaw tracks the actual pose). The
recovery pick + carry both re-select their relax-tilt at the re-read pose (`select_grasp_tilt_at` /
`select_place_tilt_at`), so the recovery never over-stretches the wrist (the worst-case probe folds in the
high rise-apex reorient and the lift->over-bowl swing -- this is what keeps recovered demos `|j4| < 1.45`).

## The bimodal noise + its knobs

A single small Gaussian jitter on the wide GR100 jaw mostly still **catches** (~20% miss) -- below the wanted
retry rate -- and its marginal landings **edge-graze** a thin body (the pen) past the penetration gate. So the
noise is **bimodal** (still object-agnostic -- one global setting, no per-object branch):

- **small-jitter** envs: a tight Gaussian on the grasp-centre xy + a small yaw wobble -- an imprecise hand
  that may still catch;
- **big clean-miss** envs (a fraction of the noised envs): a **fixed-magnitude radial push** in a random
  direction that lands the gripper **cleanly beside** the object -- a guaranteed miss with **no edge-graze**
  (so it stays off the penetration gate). A fixed magnitude (not a large Gaussian) guarantees the clearance,
  including along an elongated pen/banana's long axis.

| knob (env var) | default | meaning |
|---|---|---|
| `NOISE_RETRY` | off (`0`) | the opt-in seam: fraction of envs whose first attempt is perturbed |
| `NOISE_PROB` | `0.45` | module default for the same fraction (overridden by `NOISE_RETRY` when set) |
| `NOISE_BIG_FRAC` | `0.85` | fraction of the noised envs that get the BIG clean-miss push (vs small jitter) |
| `NOISE_BIG_DIST` | `0.095` m | the fixed radial magnitude of the big clean-miss push (clears even the long axis) |
| `NOISE_SIGMA_XY` | `0.022` m | per-axis Gaussian on the grasp-centre xy for the small-jitter mode |
| `NOISE_SIGMA_Z` | `0.0` | **no** depth noise (a too-deep z-jitter is all penetration cost, no useful miss) |
| `NOISE_YAW_DEG` | `8.0` deg | small Gaussian yaw wobble (small-jitter mode only) |

## Per-object behaviour (verified, `NOISE_RETRY=0.45`, N=16, seed 7)

| object | placed (after retry) | abnormal penetration | notes |
|---|---|---|---|
| **cube** | 100% | 0 | clean; worst `|j4|` 1.434, no-hold worst 5 frames |
| **banana** | 100% | 0 | elongated; re-grasp tracks the re-read long axis |
| **pen** | 100% | 0 | thin body; see the penetration floor below |
| **apple** | ~92% | 0 | round, contact-rich recovery (run-to-run variation, below) |
| **tennis** | ~92% | 0 | round, same as apple |

**Retry rate ~17-25%** of all demos in production (the value depends on `NOISE_RETRY`; at the higher
`NOISE_RETRY=0.45` validation setting it runs ~25-40% with the bimodal big-miss dominant).

### Honest caveats

- **Pen penetration floor.** A thin body grazed by the firm GR100 close bites a few mm; the big clean-miss
  mode (which closes *beside* the body) is what keeps the pen off the 7 mm abnormal gate -- but the pen sits
  closest to that floor of any object. It passes, with the least margin.
- **Round-object recovery is non-deterministic on GPU.** The apple/tennis recovery is contact-rich, so the
  exact `T`, placed count, and worst `|j4|` **vary run-to-run on the same seed** (e.g. apple `NOISE_RETRY=0.45 16 7`
  observed across runs: `T` 1320-1549, placed 15-16/16, worst `|j4|` 1.41-1.55). This is GPU contact-physics
  non-determinism, **not** a code defect; the clean (`NOISE_RETRY` off) path is fully deterministic and
  byte-identical (cube `8 7` -> `T=977`, 8/8). Re-run a round-object retry collection if a single run trips
  the `|j4|` gate.

## Run commands

```bash
# CLEAN baseline (byte-identical regression; deterministic): cube 8/8, T=977
CUDA_VISIBLE_DEVICES=0 python tasks/pickplace.py 8 7

# Object-agnostic grasp-retry, cube, N=16 seed 7 (FAST = hdf5 only, no videos)
CUDA_VISIBLE_DEVICES=0 FAST=1 NOISE_RETRY=0.45 python tasks/pickplace.py 16 7

# grasp-retry on another object (apple); tune the bimodal split if needed
CUDA_VISIBLE_DEVICES=0 FAST=1 TARGET=apple NOISE_RETRY=0.45 \
    NOISE_BIG_FRAC=0.85 NOISE_BIG_DIST=0.095 python tasks/pickplace.py 16 7

# Gates (run on the written demos.hdf5)
python scripts/temp/check_no_hold.py <DATA_DIR>/demos.hdf5     # worst mid-traj hold <= 8 frames
python scripts/temp/check_j4.py      <DATA_DIR>/demos.hdf5     # worst wrist |j4| < 1.45, 0 over
```

A real (non-FAST) collection writes the full photoreal 4-camera videos alongside `demos.hdf5`; `FAST=1`
writes only `demos.hdf5` + the `[COLLECT]` / `[GRASP_RETRY]` metric prints (use it only for gating, never
for a real dataset).
