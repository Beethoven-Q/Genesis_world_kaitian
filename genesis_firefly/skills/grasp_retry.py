# SPDX-License-Identifier: Apache-2.0
"""
skills/grasp_retry.py — OBJECT-AGNOSTIC imprecise-grasp-with-retry (failure-recovery data).

WHY (first principles)
----------------------
A god-mode scripted collector produces only clean first-try successes, so a policy trained on
it never learns "I missed — now what?". We want miss -> retry data so the policy can recover.

The earlier approach disturbed the OBJECT (a physics shove). That couples to each object's
dynamics (a light pen NaN-crashes, a round apple gets flung, a banana penetrates) — it needs
per-object tuning and does not scale. THIS module puts the imprecision in the ROBOT's TARGET
instead: with a small probability we Gaussian-jitter the grasp target (position only + a small
yaw); the arm reaches a slightly-wrong pose and may close on nothing / graze the object. Because
the perturbation lives in the arm's action space it is COMPLETELY OBJECT-AGNOSTIC (one global
knob, no per-object tuning) and can never "fling" the object — worst case the gripper grazes it.
Like an elderly hand: not always accurate, but it knows to try again.

THE BEHAVIOUR (encoded by this skill; orchestration wired by the task — see CONTRACT below)
  1. attempt the grasp aiming at  true_target (+ noise, for the chosen envs)
  2. god-mode check: did the object come up WITH the arm? (``grasp_succeeded``)
  3. if caught  -> continue to place (do NOT drop a first-try success)
  4. if missed  -> rise a little, reopen, RE-grasp at the TRUE re-read pose (NO noise the 2nd
                   time — we don't double-perturb), close, lift, then place.  up to ``max_retries``.

Data semantics (sound, DART-style): the noise only GENERATES misses; it is unconditioned random
so the policy cannot reproduce it — it learns the clean grasp in expectation AND learns to retry
conditioned on observing a missed grasp. The recovery is HOLD-FREE (per-env termination + idle
tail trimmed); the only static dwell is the brief grasp-close settle.

CONTRACT (how the task composes this — kept reusable for ANY grasp-based task, not just pick-place):
  noise = sample_target_noise(N, rng)                      # per-env perturbation (most envs = 0)
  # phase 1: attempt aiming at (grasp_centre + noise.dxy/dz) with quat re-yawed by noise.dyaw
  # phase 2: caught = grasp_succeeded(obj_pos_now, ee_pos_now, rest_z)
  #          caught  envs  -> place from the lift
  #          missed  envs  -> rise+reopen+re-grasp at the TRUE re-read pose -> place
The task supplies its own grasp/place waypoint builders + IK; this module owns only the
OBJECT-AGNOSTIC perturbation + success-check (+ the retry waypoint shape, added when wired).
"""
import numpy as np

# Default perturbation level — ONE global setting, not per-object. Position-only + small yaw.
# Tuned (GPU-validated downstream) so a noisy attempt MISSES often enough to yield recovery data
# but the gripper never slams the object/table. xy dominates (a lateral miss); z/yaw are small.
#
# BIMODAL noise (the miss-rate fix): a SINGLE small Gaussian jitter on a WIDE GR100 jaw mostly still CATCHES
# (~20% miss) -- below the wanted retry-rate -- AND its marginal landings EDGE-GRAZE the object (the penetration
# the gate flags). So the noise is BIMODAL, object-agnostic (one global setting, no per-object branch):
#   * SMALL-jitter envs (the majority): a tight Gaussian (sigma_xy) -- an imprecise hand that may still catch.
#   * BIG-MISS envs (a fraction NOISE_BIG_FRAC of the noised envs): a FIXED-MAGNITUDE radial push (NOISE_BIG_DIST,
#     a random direction) that lands the gripper CLEANLY BESIDE the object -- guaranteed clear of the body, so it
#     closes on AIR (a real miss -> retry) WITHOUT any edge-graze (the big offset AVOIDS the marginal penetration
#     the small jitter risks). A fixed magnitude (not a large Gaussian) GUARANTEES the clearance: a large Gaussian
#     would still occasionally land near the rim and graze.
# The two modes together push ~25-40% of ALL demos into a retry while the clean-miss mode keeps penetration off.
NOISE_PROB = 0.45          # fraction of envs whose FIRST attempt is perturbed (the rest stay byte-identical clean)
NOISE_SIGMA_XY = 0.022     # m, per-axis Gaussian on the grasp-centre xy for the SMALL-jitter mode (imprecise hand;
#                            a wide GR100 jaw tolerates ~2-3cm, so this mode catches ~80% -- the bimodal big-miss
#                            mode below supplies the rest of the wanted retry-rate without grazing).
NOISE_BIG_FRAC = 0.85      # fraction of the NOISED envs that get the BIG clean-miss offset instead of small jitter.
#                            With p=0.45 this is the dominant mode -> retry-rate ~= 0.45*(0.85*1.0 + 0.15*~0.2)
#                            ~= 0.40 of ALL demos -- top of the 25-40% target, object-agnostic. WHY so high: the
#                            SMALL-jitter mode EDGE-GRAZES a THIN body (the pen) and the firm close bites the graze
#                            past the gate (verified: small-jitter is the phase-1 pen-penetration source). The big
#                            clean-miss closes BESIDE the body -> NO graze -> no penetration AND a guaranteed miss.
#                            So a high big-frac BOTH lifts the retry-rate AND keeps the noised attempts off the gate;
#                            a small residual small-jitter (~15%) keeps some realistic 'imprecise-but-caught' data.
NOISE_BIG_DIST = 0.095     # m, the FIXED radial magnitude of the big clean-miss push (a random in-plane direction).
#                            ~9.5cm clears the largest grasp body BESIDE it so the gripper closes on AIR -- a clean
#                            miss, no edge-graze -- INCLUDING an ELONGATED pen/banana along its LONG axis (the pen is
#                            12cm long => ±6cm; a 7.5cm push along it still overlapped and CAUGHT, dropping the pen's
#                            retry-rate; 9.5cm clears even the long axis). Object-agnostic (one global magnitude; a
#                            smaller/rounder object clears with even more margin).
NOISE_SIGMA_Z = 0.0        # NO depth noise. A too-deep z-jitter drives the firm pinch into a THIN object (the
#                            pen) past the 7mm penetration gate WITHOUT adding any useful miss -- the miss that
#                            yields recovery data is LATERAL (xy). So depth-noise is all collision cost, no benefit.
NOISE_YAW_DEG = 8.0        # deg, small Gaussian yaw wobble (SMALL-jitter mode only); orientation otherwise INTACT


class TargetNoise:
    """Per-env grasp-target perturbation (the result of :func:`sample_target_noise`)."""
    __slots__ = ("mask", "dxy", "dz", "dyaw", "big")

    def __init__(self, mask, dxy, dz, dyaw, big=None):
        self.mask = mask        # (N,) bool: which envs' FIRST attempt is perturbed
        self.dxy = dxy          # (N,2) m: xy offset added to the grasp centre
        self.dz = dz            # (N,)  m: z offset added to the grasp centre
        self.dyaw = dyaw        # (N,)  rad: yaw offset folded into the grasp quat
        self.big = big if big is not None else np.zeros_like(mask)   # (N,) bool: the BIG clean-miss envs

    @property
    def any(self):
        return bool(self.mask.any())


def sample_target_noise(n_envs, rng, *, p=NOISE_PROB, sigma_xy=NOISE_SIGMA_XY,
                        sigma_z=NOISE_SIGMA_Z, yaw_deg=NOISE_YAW_DEG,
                        big_frac=NOISE_BIG_FRAC, big_dist=NOISE_BIG_DIST):
    """Sample the per-env BIMODAL grasp-target perturbation for the FIRST attempt. OBJECT-AGNOSTIC: the
    same global ``p``/``sigma``/``big_*`` for every object (a smaller object naturally misses more under the
    same offset — exactly an imprecise hand). Non-perturbed envs get an exact-zero offset so their clean
    first-try grasp is byte-for-byte unchanged. Of the perturbed envs, a fraction ``big_frac`` get the BIG
    fixed-magnitude radial CLEAN-MISS push (random direction, ``big_dist`` metres — closes BESIDE the object,
    a real miss with NO edge-graze); the rest get the small Gaussian jitter (may still catch). Returns a
    :class:`TargetNoise` (``.big`` flags the clean-miss envs). Env-overridable via NOISE_BIG_FRAC/NOISE_BIG_DIST."""
    import os
    big_frac = float(os.environ.get("NOISE_BIG_FRAC", big_frac))
    big_dist = float(os.environ.get("NOISE_BIG_DIST", big_dist))
    mask = rng.rand(n_envs) < float(p)
    big = mask & (rng.rand(n_envs) < big_frac)                 # the clean-miss subset of the noised envs
    # SMALL-jitter mode: tight Gaussian on every noised env (the big-miss envs OVERWRITE their dxy below).
    dxy = (rng.randn(n_envs, 2) * float(sigma_xy)).astype(np.float32)
    dz = (rng.randn(n_envs) * float(sigma_z)).astype(np.float32)
    dyaw = (np.radians(rng.randn(n_envs) * float(yaw_deg))).astype(np.float32)
    # BIG clean-miss mode: a FIXED-magnitude push in a random in-plane direction -> lands cleanly beside the body.
    if big.any():
        th = rng.rand(n_envs) * (2.0 * np.pi)
        big_off = (np.stack([np.cos(th), np.sin(th)], axis=1) * float(big_dist)).astype(np.float32)
        dxy[big] = big_off[big]
        dyaw[big] = 0.0                                        # a clean miss BESIDE the object -> no yaw wobble
    dxy[~mask] = 0.0
    dz[~mask] = 0.0
    dyaw[~mask] = 0.0
    return TargetNoise(mask, dxy, dz, dyaw, big=big)


def grasp_succeeded(obj_pos, ee_pos, rest_z, *, lift_min=0.03, near_ee=0.10):
    """OBJECT-AGNOSTIC god-mode grasp-success check: did the object come up WITH the arm?
    True iff the object lifted >= ``lift_min`` above its rest height AND is within ``near_ee`` of
    the active EE (it rose IN the gripper). A miss (closed on nothing / grazed) leaves the object
    on the table -> not lifted and/or far from the EE -> False. Works for any object/shape.

    obj_pos : (N,3) world xyz of the object (god-mode read)
    ee_pos  : (N,3) world xyz of the active end-effector
    rest_z  : (N,) or scalar — the object's resting height (table_top + object half-height)
    """
    obj_pos = np.asarray(obj_pos, float)
    ee_pos = np.asarray(ee_pos, float)
    lifted = obj_pos[:, 2] > (np.asarray(rest_z, float) + float(lift_min))
    with_ee = np.linalg.norm(obj_pos - ee_pos, axis=1) < float(near_ee)
    return lifted & with_ee


# --------------------------------------------------------------------------------------------------- #
# small wxyz/rotation helper (kept local so this skill has NO dependency on the task or skills.grasp).
# --------------------------------------------------------------------------------------------------- #
def yaw_wxyz(q, dyaw):
    """Rotate the quaternion ``q`` (wxyz) about the WORLD +Z axis by ``dyaw`` radians (the small yaw
    wobble of the noised first attempt). Pre-multiply by the world-Z rotation so the grasp's approach
    direction is preserved and only the jaw yaw shifts -- OBJECT-AGNOSTIC (no per-object orientation)."""
    from scipy.spatial.transform import Rotation as _Rot
    q = np.asarray(q, float)
    rz = _Rot.from_rotvec([0.0, 0.0, float(dyaw)])
    r = rz * _Rot.from_quat([q[1], q[2], q[3], q[0]])      # world-Z * current -> re-yawed grasp
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


# =================================================================================================== #
# THE TWO-PHASE, HOLD-FREE ORCHESTRATION  (the reusable skill the task calls; the task stays thin)
# =================================================================================================== #
def run_grasp_retry(*, N, noise, gc, gqA, LIFT, APP, OPEN, CLOSE,
                    pick_wps, noised_pick_wps, retry_pick_wps, place_tail, run_phase,
                    read_obj_pos, read_ee_pos, brief_settle, rest_z):
    """OBJECT-AGNOSTIC grasp-retry orchestration. Runs the TWO-PHASE, HOLD-FREE attempt->check->retry
    described in the module CONTRACT and returns ``env_phases`` (the per-env phase list the task's
    recorder uses to build each variable-length, hold-free demo). The task supplies its OWN grasp/place
    waypoint BUILDERS + IK (which it already owns); this skill owns ONLY the object-agnostic perturbation
    VALUES (sampled into ``noise``), the success-check, the phase SEQUENCING, and the god-mode check.

    PHASE 1 (uniform, all envs) -- the grasp ATTEMPT (home->pre->at->close->lift):
        * NON-noised envs run the task's clean phase-0 attempt ``pick_wps(i)`` (the locked clean pick).
        * NOISED envs run ``noised_pick_wps(i, dxy, dz, dyaw)``: the task aims the pre+at at the perturbed
          target ``gc[i]+[dxy,dz]`` with a grasp orientation RE-PLANNED (wrist-comfortable) at that target
          + the noise yaw wobble -- a slightly-wrong target, so the gripper may close on nothing / graze it.
      Run as ONE phase via ``run_phase`` (-> phase index 0).
    GOD-MODE CHECK (a brief UNRECORDED settle only -- NO long arm freeze): read object + active-EE world
      pos and ``caught = grasp_succeeded(...)``.
    PHASE 2 (per-env, variable length) -- run as ONE phase (-> phase index 1):
        * CAUGHT envs continue to place from their PHASE-1 lift: ``place_tail(i, lift_pose, gqA[i])``
          (never drop a first-try success; the lift pose is the clean ``gc[i] + [0,0,LIFT]``).
        * MISSED envs re-grasp at the object's ACTUAL re-read pose via ``retry_pick_wps(i, obj_pos_now)``
          (the task RE-PLANS the grasp AT the re-read pose -- NO noise the 2nd time -- with its relax-tilt
          so there is NO over-stretch), then places from that recovery lift.
      Both branches APPEND to the same per-env stream, so each env runs as ONE continuous trajectory and
      TERMINATES at home -- never idling waiting for a slower env.

    Returns (``env_phases`` = [[0, 1] for every env], ``caught``). The only static-arm dwell anywhere is
    the grasp-close settle (gripper ramp); the recorder's idle-home-tail trim removes the rest.

    Readers (god-mode, privileged): ``read_obj_pos()->(N,3)`` ``read_ee_pos()->(N,3)``
    ``brief_settle(steps)`` steps physics holding the last command WITHOUT recording (so the check is a
    settle, NOT a hold in the data). ``rest_z`` (N,) object resting height."""
    lift_pose = gc + np.array([0.0, 0.0, float(LIFT)])         # the clean phase-1 lift pose (== pick_wps lift)

    # ---- PHASE 1: the grasp ATTEMPT (one phase, all envs). Noised envs aim slightly wrong. -------------
    def attempt_wps(i):
        if not bool(noise.mask[i]):
            return pick_wps(i)                                 # byte-identical clean attempt
        return noised_pick_wps(i, noise.dxy[i], noise.dz[i], noise.dyaw[i])   # task re-plans the noised grasp

    run_phase([attempt_wps(i) for i in range(N)], settle_steps=0, tag="retry-attempt")

    # ---- GOD-MODE CHECK: a brief UNRECORDED settle, then read object + active EE (NO long arm freeze) --
    brief_settle(20)
    obj_now = read_obj_pos()                                   # the object's ACTUAL pose for the re-grasp
    caught = grasp_succeeded(obj_now, read_ee_pos(), rest_z)
    retried = ~caught                                          # every missed env runs a phase-2 RETRY
    print(f"[GRASP_RETRY] noised={int(noise.mask.sum())}/{N} (big-miss={int(noise.big.sum())}, "
          f"small-jitter={int((noise.mask & ~noise.big).sum())})  missed(of noised)="
          f"{int((noise.mask & ~caught).sum())}  caught={int(caught.sum())}/{N}  "
          f"RETRIED={int(retried.sum())}/{N} ({100.0*retried.mean():.0f}%)", flush=True)

    # ---- PHASE 2: per-env (one phase). Caught -> place; missed -> re-grasp at the TRUE re-read pose. ----
    def outcome_wps(i):
        if bool(caught[i]):                                  # FIRST-TRY SUCCESS: continue to place, never drop it
            return place_tail(i, lift_pose[i], gqA[i])
        return retry_pick_wps(i, obj_now)                    # MISSED: rise+reopen+re-grasp@re-read pose + place

    run_phase([outcome_wps(i) for i in range(N)], settle_steps=40, tag="retry-outcome")
    return [[0, 1] for _ in range(N)], caught
