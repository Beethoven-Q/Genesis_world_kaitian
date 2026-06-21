# Disturbance — collaborative CHASE + god-mode, read-based failure RECOVERY

An **opt-in augmentation** (`DISTURB>0`, default 0 = off) for the Genesis firefly pick→bowl task that generates
**failure-recovery** training data: a privileged, per-env **shove** of the TARGET object at a **random time during
the grasp approach**, sensed by the god-mode solver only after a perception **sense-delay**. Depending on WHEN the
solver learns the object moved, it reacts on a simple, robust logic:

- **informed BEFORE the close command → CHASE.** Don't close yet. Rise a little to recover the view, **read the
  object's new ground-truth pose (position + orientation)** and grasp it there.
- **informed only AFTER the close command → the grasp ATTEMPT already fired.** Check the outcome god-mode:
  - the object was **still caught + lifted** despite the shove → it SUCCEEDED: **keep going, place it** (no drop,
    no re-grasp). *(disturbed-but-grasped data)*
  - the grasp **missed** (closed on nothing) → reopen, rise to recover the view, **read the ground-truth pose** and
    re-grasp. *(failure → recover data)*

> **This is NOT an LLM subagent.** It is a deterministic injection **HARNESS** (`skills/disturbance.py`) plus
> god-mode **CONTROL** in the task (`tasks/pickplace.py`) — the same first-principles pattern as the penetration
> gate and the 50/50 distractors. Both data modes are reproducible from `(seed, probability)`.

> **The disturbance is UNCONSTRAINED, and the recovery is GENERAL.** The shove may slide, rotate, or tumble the
> object onto another face; its new 6-DOF pose can be completely different. The recovery is god-mode — it READS the
> object's full ground-truth pose and grasps it there — so it never needs the object to stay "nice", and it works
> for **every** target (cube, elongated banana/pen, round apple/tennis), not just the cube.

Files: `skills/disturbance.py` (the harness: shove sampling + firing; the disturbed-trajectory assembly: phase-1/2
builders + the **god-mode full-6-DOF recovery grasp**), `tasks/pickplace.py` (the thin task: builds a `DisturbExec`
context and calls in; owns the clean `DISTURB=0` path), `skills/executor.py` (the additive `during_step` hook — the
only motion-path touch).

---

## 1. The harness (`skills/disturbance.py`)

**Per-trial probability — like the 50/50 distractors.** A per-env draw decides whether a trial is disturbed:
`disturbed = rng.rand(N) < prob` (default `prob = 0.34`; env var `DISTURB` overrides; `DISTURB=0` = off).
`DisturbanceSpec.sample(target, N, rng, prob=...)` draws — from the SAME stage `rng`, so it is reproducible
alongside the DR — the disturbed mask, a per-env in-plane velocity impulse `(vx, vy)`, a per-env sense-delay, a
per-env fire-fraction within the approach, and the **explicit chase-vs-after-close split** (`force_after_close`).

**The injection (physical, not a teleport).** The target is a free rigid body (`gs.morphs.Box`) with a 6-DOF FREE
joint (local dofs `[0,1,2]=xyz, [3,4,5]=rot`). The shove is a HORIZONTAL velocity impulse on dofs `[0,1]` of the
disturbed envs only, BATCHED via `entity.set_dofs_velocity(...)`. The firm solver CARRIES the impulse — the object
accelerates, slides under friction, and may rotate/tumble — so the penetration gate stays valid (no instantaneous
overlap), and it composes with the solver state where a `set_pos` would fight it. The task drives the harness one
control step at a time (`spec.arm(fire_steps)` then `spec.tick(t)` every step); each disturbed env fires once, at a
random fire-step inside its own approach window.

**Chase vs after-close** is split explicitly at sample time (`force_after_close`, at rate `late_prob`) rather than
derived from the sense-delay timing, so each mode is generated at a controlled rate.

---

## 2. The god-mode, read-based recovery grasp (works for ALL objects)

When a re-grasp is needed (a chase, or an after-close miss), the task READS the object's **actual full 6-DOF
ground-truth pose** (`cube.get_pos()` + `cube.get_quat()`) and grasps it there — **no prediction**.

`recovery_grasp_quat(obj_quat, reach_dir, tilt, htR, laxis_local)` builds the grasp orientation from the object's
actual world quaternion using the **same** orientation logic as the first grasp (`grasp.grasp_quat_at`), but with
the reference axis rotated by the FULL read quat (not just an in-plane yaw):

- the object's local grasp axis `laxis_local = spec.long_axis_local()` — a **face normal** for the cube, the **long
  axis** for an elongated banana/pen, **`None`** for a round apple/tennis — is expressed in WORLD via the read quat;
- the parallel jaws are aligned to close **perpendicular** to it (across the short axis / opposite face-pair);
- for a **round** object (`None`) the roll is free → snapped to the branch closest to the HOME wrist (so the
  carry→home wrist never flips), exactly as the first grasp does.

The grasp **position** is the object's true geometric centre (read god-mode). The wrist-margin relax-tilt ladder
(`select_recovery_grasp`) is re-run at the actual pose (same thresholds + warm-start probe as the first grasp), so
the recovery posture is as natural as the original. Result: the re-grasp HITS the object however the shove relocated
/ rotated / re-faced it, for every target.

---

## 3. Execution model — per-env, HOLD-FREE, no cross-env barrier (`tasks/pickplace.py`)

Everything runs on the **one smooth motion path** (BatchExecutor + densify). The load-bearing rule (owner): each
trial is fully independent; **no env ever idles/holds** (especially not lifted). `DISTURB=0` is the **clean** single
continuous `home→pre→at→close→lift→carry→lower→release→home` (the foundation + bulk data + grasp-solving runs),
unchanged and byte-identical.

`DISTURB>0` runs **two batches** so the god-mode read between them never freezes the arm in the recorded demo:

- **Phase 1 (all envs).** undisturbed → full pick→place→home; a **chase** approaches then aborts before the close
  and rises (gripper OPEN); an **after-close** runs the full grasp ATTEMPT (close + lift). The shove fires (real
  physics) via the per-step `tick` hook.
- **Read (god-mode, between batches).** Check each after-close outcome (still-grasped vs missed). Step the sim a few
  **unrecorded** control steps (arm held at its phase-1-end pose) so any missed/relocated object settles **on a
  face** (a cube cannot rest on an edge — gravity forbids it), then read the ground-truth pose. Instantaneous; no
  recorded frame, so no hold in the demo.
- **Phase 2 (all envs).** place the still-grasped object as-is; re-grasp the missed/chase object at its read pose →
  place → home; undisturbed envs hold at home (a trimmed tail).

**Per-env, hold-free assembly.** The shared recording covers both batches for all envs. Each env's demo is assembled
from ITS own phase frames (`env_phases[e]`: undisturbed = phase 0; disturbed = phases 0+1), and any long static-arm
run where the gripper is **also** static (an inter-phase lifted pad, a home idle, a closed-on-nothing hold) is
collapsed to a short settle; gripper-ramp dwells (the grasp close/open/release) are kept. So every saved demo is one
continuous **hold-free** trajectory (the only static moment is the brief grasp-close settle ≤ the gate). Demos come
out **variable-length** (natural; LeRobot supports it).

---

## 4. HDF5 attrs (per demo)

In addition to the existing attrs (`success`, `max_penetration_mm`, `penetrating`, `degenerate`, `has_distractors`,
`dr_*`, …):

| attr | type | meaning |
|---|---|---|
| `disturbed` | bool | the object was shoved at a random time during this trial's grasp approach |
| `disturb_phase` | str | `"before_close"` (→ chase) · `"after_close"` · `"none"` |
| `disturb_outcome` | str | `"chased"` · `"grasped_thru"` · `"recovered"` · `"failed"` · `"none"` (below) |
| `recovery_attempts` | int | re-grasps the solver needed (1 for a chase or an after-close miss; 0 for a grasped-through success) |
| `recovered` | bool | *(legacy)* `disturbed AND ended cleanly placed` |

`disturb_outcome` semantics:
- **`chased`** — informed before the close: didn't close, read the new pose + grasped it, PLACED.
- **`grasped_thru`** — informed after the close but the attempt still caught the object despite the shove → kept
  going + PLACED (`recovery_attempts == 0`).
- **`recovered`** — informed after the close, the grasp missed → reopen/rise/read/re-grasp, PLACED
  (`recovery_attempts == 1`, the failure→recover data).
- **`failed`** — disturbed but did NOT end cleanly placed — a legit hard edge the place/penetration gate rejects.
- **`none`** — not disturbed.

The **penetration gate** and **degenerate-settle gate** still apply (`success = placed & ~penetrating &
~degenerate`). **Empty-close allow-list:** when the gripper closes on nothing, the two opposing claws of the *same*
gripper meet and the firm solver reports their mutual overlap (~1 cm) — geometrically EXPECTED designed contact, so
the task adds ONLY the same-gripper L-claw↔R-claw geom pairs to the penetration tracker's `ignore_pairs`.

---

## 5. Running it

```bash
# CLEAN single-trajectory pick-place (the DEFAULT -- DISTURB unset == 0), any TARGET
CUDA_VISIBLE_DEVICES=1 DATA_DIR=/path OUT_DIR=genesis_firefly/output/temp/run \
  ./.venv/bin/python genesis_firefly/tasks/pickplace.py 8 7

# OPT IN to failure-recovery augmentation (works for any target object)
CUDA_VISIBLE_DEVICES=1 DISTURB=0.5 TARGET=banana ... ./.venv/bin/python genesis_firefly/tasks/pickplace.py 16 7

# NO-HOLD gate + per-env outcome diagnostics
DISTURB_DIAG=1 ...    # per-disturbed-env: ground-truth pose read + per-env outcome
./.venv/bin/python genesis_firefly/scripts/temp/gate_no_hold.py <DATA_DIR>/demos.hdf5
```

Env vars: `DISTURB` (per-env disturbance probability; **default 0 = off**), `SHOVE_SPEED_LO`/`SHOVE_SPEED_HI`
(impulse magnitude m/s, default ~0.70–0.95), `SHOVE_SETTLE` (unrecorded settle steps before the god-mode read,
default 100), `DISTURB_DIAG`, `FAST` (hdf5-only, no videos), `TARGET`, `DATA_DIR`, `OUT_DIR`. One GPU per sim
process. Run from the repo root.

---

## 6. Verification (GPU 1, real runs)

Dual gate — `DISTURB=0.5 N=16`, seeds 7 and 23 (cube):

```
seed 7 : 16/16 placed (5 chased, 2 recovered, 0 failed)   penetration 0 abnormal   NO-HOLD worst = 8 frames  PASS
seed 23: 16/16 placed (3 chased, 3 grasped-thru, 4 recovered, 0 failed)   0 abnormal   NO-HOLD worst = 8  PASS
```

- **Recovery works (both modes), ≥80% placed, recovered > 0.** The `grasped_thru` count is the disturbed-but-still-
  grasped success path (kept, not dropped). The read-based re-grasp HITS the relocated/rotated/tumbled object.
- **NO-HOLD ≤ 8 frames** (the grasp-close settle, the only allowed static moment) — no lifted freeze; the god-mode
  settle/read is unrecorded.
- **Clean path byte-identical** (`DISTURB=0`, E=8, seed 7, cube → 8/8 placed, `T=977`, 0 abnormal; trajectory
  lengths identical to HEAD, numeric deltas at sim-nondeterminism scale only).
- **General over objects:** `DISTURB=0.5` runs on banana / pen / apple / tennis recover via the same god-mode
  full-6-DOF grasp (object-aware reference axis: long-axis for elongated, face for cube, roll-snapped for round).

> Re-verify: `CUDA_VISIBLE_DEVICES=1 DISTURB=0.5 FAST=1 [TARGET=<obj>] ./.venv/bin/python
> genesis_firefly/tasks/pickplace.py 16 7`, then `./.venv/bin/python genesis_firefly/scripts/temp/gate_no_hold.py
> <DATA_DIR>/demos.hdf5`.

---

## 7. Future cases

The same harness generalises by choosing a different trigger-phase / impulse: a turntable moving-target grasp
(per-step pose read → continuous chase), a cup tipped by an angular impulse (detect the tip → re-right), an object
knocked loose mid-transport (fire during `carry` → re-pick). Each is the same pattern: a privileged, reproducible
sim injection + god-mode detection from privileged state + a god-mode read-based re-grasp built from the locked
skill/waypoint builders.
