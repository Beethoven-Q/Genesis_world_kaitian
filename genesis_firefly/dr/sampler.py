# SPDX-License-Identifier: Apache-2.0
"""DR SAMPLER — split a task's DR into a per-build draw (``BuildDR``) + a batched per-env draw (``EnvDR``).

This is the engine half of the harness (docs/domain_randomization.md "How it's applied"). Given the stage RNG,
the env count and a ``TaskSpec``, it draws every IMPLEMENTED scope-A/B/C value once, returning:

  * ``BuildDR`` — the per-build draws (constant within one parallel build): the grasp-target render colour + the
                  bowl colour. (The per-build table TEXTURE + per-env HDRI are drawn inside the stage in Phase 1
                  — it owns its visual setup — so they are RECORDED onto BuildDR/EnvDR from the stage, not redrawn
                  here; that keeps the RNG order byte-identical and avoids a second draw.)
  * ``EnvDR``  — the ``(N,)``-batched per-env arrays: arm side, object-table height, bowl/target XY, target yaw,
                  target mass-shift, and (post-build) the distractor on/off mask + the robot-link friction ratio.

THE PARITY LEVER (Phase 1): this module reproduces the EXACT RNG call ORDER of the pre-refactor collector so
every sampled value is byte-identical. The order, all on ``stage.rng`` after stage construction, is:

    sample_env_phys_dr():    side_is_left, tabZ, bowx, bowy, cubx, cuby, [clearance-reject loop], yaw, mass_shift
    sample_build_colours():  target free colour (distinct_object_color), target_color(...), bowl colour
    --- stage.build() ---
    sample_post_build():     has_distractors, robot-link friction ratio

DO NOT reorder these calls or insert/remove an RNG draw — that shifts every downstream value and breaks parity.
The distractor TYPE/POSE draws happen between the colour draws and build() (they are task-layout-specific and
stay in the task for Phase 1; Phase 3 extracts them) — the sampler is called around them in the same order.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import world.object_factory as obj_factory
from dr.object_dr import object_dr


# ---- scope-A object-table GROW + scope-A side-cam + scope-C light bands (per-build) -----------------------------
# Object-table SIZE grow (scope A): the current size is the MINIMUM; grow the width (both y edges) up to +dW and
# the length AWAY FROM THE ARM (the seam end stays pinned) up to +dL. Realistic table-size variety; the texture
# rescales to stay flush. (env overrides let the DR strategist widen/narrow without editing code.)
OTABLE_GROW_W = float(os.environ.get("OTABLE_GROW_W", "0.18"))     # max width grow (m), split over both y edges
OTABLE_GROW_L = float(os.environ.get("OTABLE_GROW_L", "0.22"))     # max length grow (m), AWAY from the arm only
# Side-cam (scope A): raise the SIDE cam up to +SIDECAM_DZ and pitch DOWN to keep the workspace framed (the stage
# re-aims the lookat). Side cam ONLY -- the wrist cams are never touched.
SIDECAM_DZ_MAX = float(os.environ.get("SIDECAM_DZ_MAX", "0.05"))   # max side-cam height raise (m)
# Light (scope C, PER-BUILD -- Nyx can't switch the directional light per-env, only the env-map; see scopes.py):
# a coloured directional KEY light. Realistic illuminant tints (kept mild so objects stay recognizable) + an
# intensity band. The HDRI still supplies per-env image-based lighting on top.
LIGHT_COLORS = {
    "white":     (1.00, 1.00, 1.00),
    "orange":    (1.00, 0.82, 0.60),   # warm tungsten
    "yellow":    (1.00, 0.95, 0.70),   # warm white
    "light_blue": (0.78, 0.88, 1.00),  # cool / overcast
    "sun":       (1.00, 0.97, 0.88),   # daylight sun
}
LIGHT_INT_LO = float(os.environ.get("LIGHT_INT_LO", "0.7"))
LIGHT_INT_HI = float(os.environ.get("LIGHT_INT_HI", "1.7"))


def _dr_scale(name):
    """A per-env range HALF-WIDTH multiplier read from the environment (the DR-strategist sweep hook). Defaults
    to EXACTLY 1.0 so an unset environment reproduces the v2 collection byte-for-byte. Scales only the band
    half-width about its FIXED centre — never moves a centre, never touches the anti-coupling sgn / arm-side
    logic, never relaxes the target<->bowl clearance floor (see docs + the sweep tool)."""
    try:
        return float(os.environ.get(name, "1.0"))
    except (TypeError, ValueError):
        return 1.0


@dataclass
class TaskSpec:
    """What a task tells the DR harness about ITS objects + layout. Scopes A (scene) and C (visual) are free and
    automatic — the task names only its scope-B object(s) + the per-task knobs the pose draw needs.

    target          : the grasp-target object name (registry key).
    target_spec     : the target's ObjectSpec (carries the colour policy + footprint for the clearance floor).
    object_table_height : the nominal object-table top z (the height-DR centre).
    """
    target: str
    target_spec: object
    object_table_height: float


@dataclass
class BuildDR:
    """Per-build draws (constant within one parallel build): the object render colours + the per-build scope-A/C
    fields that BAKE at build (geometry SIZE, table SIZE, side-cam pose, the coloured light). The colours + table
    texture are RECORDED off the stage (it owns its visual setup); the rest are drawn here and PASSED to the stage
    (table-size/side-cam/light) or used at spawn (object size)."""
    target_color: tuple            # the grasp-target render colour (palette / free / native-fallback)
    bowl_color: tuple              # the bowl's free distinct colour
    target_free_color: tuple       # the FREE distinct-from-table colour drawn first (cube uses it directly)
    table_texture: object = None   # the per-build table-texture path realised by the stage (record-only)
    # --- Phase-2 per-build fields (scope B object SIZE; scope A table-size + side-cam; scope C light) ---
    obj_scale: float = 1.0         # the target's SIZE multiplier (scope B): spawned at spec.scale * obj_scale
    otable_grow_w: float = 0.0     # object-table WIDTH grow (m), split over both y edges (scope A)
    otable_grow_l: float = 0.0     # object-table LENGTH grow (m), AWAY from the arm only; seam end pinned (scope A)
    sidecam_dz: float = 0.0        # side-cam height raise (m); the stage pitches DOWN to re-frame (scope A)
    light_name: str = "white"      # the chosen illuminant tint name (scope C)
    light_color: tuple = (1.0, 1.0, 1.0)   # the directional key-light RGB (scope C)
    light_intensity: float = 1.2   # the directional key-light intensity (scope C)


@dataclass
class EnvDR:
    """The ``(N,)``-batched per-env arrays. Field names match the pre-refactor ``dr`` dict so the task + plan
    read them unchanged. ``has_dist`` + ``fric`` are filled by ``sample_post_build`` (they are drawn AFTER
    ``stage.build()`` to preserve the RNG order)."""
    side_is_left: np.ndarray
    sgn: np.ndarray
    tabZ: np.ndarray
    bowx: np.ndarray
    bowy: np.ndarray
    cubx: np.ndarray
    cuby: np.ndarray
    yaw: np.ndarray
    mass_shift: np.ndarray
    # --- Phase-2 per-env physics bands (scope A table friction; scope B object friction) ---
    table_fric: np.ndarray = None   # (N,) per-env table friction-ratio about the Box spawn friction (scope A)
    obj_fric: np.ndarray = None     # (N,) per-env TARGET friction-ratio about spec.friction (scope B, distinct knob)
    has_dist: np.ndarray = None
    fric: np.ndarray = None
    hdrs: list = field(default_factory=list)   # the realised per-env HDRI paths (recorded off the stage)

    def as_dict(self):
        """The legacy ``dr`` dict the task's samplers/skills index by key (byte-for-byte the same keys)."""
        return dict(side_is_left=self.side_is_left, sgn=self.sgn, tabZ=self.tabZ, bowx=self.bowx,
                    bowy=self.bowy, cubx=self.cubx, cuby=self.cuby, yaw=self.yaw, mass_shift=self.mass_shift)


def _footprint_radius(spec):
    """Half the XY diagonal of an object's AABB = the radius of the disk that contains the object at ANY yaw
    (used by the target<->bowl clearance FLOOR so a big target never spawns intersecting the bowl wall)."""
    e = spec.scaled_extents()
    return 0.5 * float(np.hypot(e[0], e[1]))


def sample_env_phys(N, rng, task_spec) -> EnvDR:
    """PER-ENV physics DR (no visuals — the stage owns the environment/background DR). Byte-for-byte the
    pre-refactor ``sample_phys_dr``: the SAME ``rng`` calls in the SAME order (the parity lever). Returns an
    ``EnvDR`` with the pre-build per-env arrays filled (has_dist/fric come from ``sample_post_build``)."""
    spec = task_spec.target_spec
    ho = task_spec.object_table_height
    ps = _dr_scale("DR_POSE_SCALE")                            # pose half-width multiplier (1.0 = v2 default)
    side_is_left = rng.rand(N) < 0.5
    sgn = np.where(side_is_left, 1.0, -1.0)
    tabZ = ho + (rng.rand(N) - 0.5) * 0.10                     # object-table height +/-5cm
    bowx = 0.40 + (rng.rand(N) - 0.5) * 0.10 * ps
    bowy = sgn * (0.05 + (rng.rand(N) - 0.5) * 0.07 * ps)
    cubx = 0.40 + (rng.rand(N) - 0.5) * 0.12 * ps
    cuby = sgn * (0.185 + (rng.rand(N) - 0.5) * 0.10 * ps)
    # keep the TARGET CLEAR of the bowl so the open gripper doesn't bump the bowl on the grasp descent (see
    # tasks/pickplace.py for the full rationale + the degenerate-spawn root-cause). The hard floor forbids
    # target-in-bowl: 0.125 for the cube (locked literal -> byte-for-byte) else footprint+bowl+margin.
    CLR_HARD = 0.125 if spec.is_cube else float(_footprint_radius(spec) + 0.075 + 0.015)
    clr = np.where(rng.rand(N) < 0.8, max(0.17, CLR_HARD + 0.045), CLR_HARD)
    for _ in range(60):
        bad = np.hypot(cubx - bowx, cuby - bowy) < clr
        if not bad.any():
            break
        nb = int(bad.sum())
        cubx[bad] = 0.40 + (rng.rand(nb) - 0.5) * 0.12 * ps
        cuby[bad] = sgn[bad] * (0.185 + (rng.rand(nb) - 0.5) * 0.10 * ps)
    # GUARANTEED fallback: push any env STILL inside the hard floor radially out to exactly CLR_HARD.
    bad = np.hypot(cubx - bowx, cuby - bowy) < CLR_HARD
    if bad.any():
        dx, dy = cubx[bad] - bowx[bad], cuby[bad] - bowy[bad]
        dn = np.hypot(dx, dy)
        ux = np.where(dn > 1e-6, dx / np.maximum(dn, 1e-6), 0.0)
        uy = np.where(dn > 1e-6, dy / np.maximum(dn, 1e-6), sgn[bad])   # degenerate: push along the arm side
        cubx[bad] = bowx[bad] + ux * CLR_HARD
        cuby[bad] = bowy[bad] + uy * CLR_HARD
    yaw = (rng.rand(N) - 0.5) * np.radians(180)
    ms = _dr_scale("DR_MASS_SCALE")                            # cube mass-shift half-width multiplier
    mass_shift = ((rng.rand(N, 1) - 0.5) * 0.04 * ms).astype(np.float32)
    # --- Phase-2 per-env FRICTION bands (drawn LAST so the pose/mass draws above keep their positions) ---
    # (A) TABLE friction (scope A): a per-env ratio about the table-Box spawn friction, metal<->wood<->wool, kept
    # DECOUPLED from the texture (the stage's texture choice never reads this). band [0.55,1.45] ~= metal..wool.
    # (B) OBJECT friction (scope B): a per-env ratio on the TARGET about spec.friction (a small surface-finish
    # spread), DISTINCT from the robot-link friction knob. Both are scaled by DR_FRIC_SCALE (the sweep hook) and
    # clamped >=0. (Setting DR_TABLE_FRIC_SCALE / DR_OBJ_FRIC_SCALE to 0 toggles each field off for regression.)
    fs = _dr_scale("DR_FRIC_SCALE")
    tfs = _dr_scale("DR_TABLE_FRIC_SCALE") * fs
    ofs = _dr_scale("DR_OBJ_FRIC_SCALE") * fs
    table_fric = (1.0 + (rng.rand(N) - 0.5) * 0.90 * tfs).clip(0.0).astype(np.float32)   # ~[0.55,1.45]
    ofb = float(getattr(object_dr(task_spec.target), "friction_band", 0.20))
    obj_fric = (1.0 + (rng.rand(N) - 0.5) * 2.0 * ofb * ofs).clip(0.0).astype(np.float32)
    return EnvDR(side_is_left=side_is_left, sgn=sgn, tabZ=tabZ, bowx=bowx, bowy=bowy,
                 cubx=cubx, cuby=cuby, yaw=yaw, mass_shift=mass_shift,
                 table_fric=table_fric, obj_fric=obj_fric)


def sample_build_colours(stage, task_spec) -> BuildDR:
    """PER-BUILD object render colours, drawn on ``stage.rng`` in the EXACT pre-refactor order: (1) the target's
    FREE distinct-from-table colour, (2) the target's realistic-palette colour (cube -> the free colour), (3) the
    bowl's free distinct colour (avoiding the target's hue). Also records the stage's realised per-build table
    texture (for the trace). Returns a ``BuildDR``."""
    spec = task_spec.target_spec
    ch, free_color = stage.distinct_object_color()                 # the target's free random colour (+ hue to avoid)
    tgt_col = obj_factory.target_color(spec, free_color, stage.rng)  # realistic palette colour (cube -> free)
    _, bowl_col = stage.distinct_object_color(ch)
    # object SIZE (scope B): the target spawns at spec.scale * obj_scale, obj_scale in [1-frac, 1+frac] (realistic
    # +/-frac; grasp planning reads scaled_extents() so it adapts). Drawn HERE (after the colours, on stage.rng so
    # it is part of the per-build draw sequence). DR_SIZE_SCALE=0 -> obj_scale=1 (toggle off for regression).
    sbf = float(getattr(object_dr(task_spec.target), "size_band_frac", 0.10)) * _dr_scale("DR_SIZE_SCALE")
    obj_scale = float(1.0 + (stage.rng.rand() - 0.5) * 2.0 * sbf)
    # The BUILD-BAKED scope-A/C scene fields (object-table GROW, side-cam pose, the coloured LIGHT) are drawn
    # INSIDE the stage __init__ (it owns its build-time geometry/visual setup, exactly like the table TEXTURE +
    # per-env HDRI) -- RECORDED here off the stage so BuildDR is the single source for the trace.
    return BuildDR(target_color=tgt_col, bowl_color=bowl_col, target_free_color=free_color,
                   table_texture=getattr(stage, "table_texture", None), obj_scale=obj_scale,
                   otable_grow_w=getattr(stage, "otable_grow_w", 0.0),
                   otable_grow_l=getattr(stage, "otable_grow_l", 0.0),
                   sidecam_dz=getattr(stage, "sidecam_dz", 0.0),
                   light_name=getattr(stage, "light_name", "white"),
                   light_color=tuple(getattr(stage, "light_color", (1.0, 1.0, 1.0))),
                   light_intensity=float(getattr(stage, "light_intensity", 1.2)))


def sample_build_scene_dr(rng):
    """Draw the BUILD-BAKED scope-A/C scene fields the STAGE needs at construction time (geometry/visual bake at
    build, so they must be known before the entities are created): the object-table GROW (width + length-away),
    the side-cam height raise, and the per-build coloured directional LIGHT. Called by ``ManipulationStage`` on
    its own ``self.rng`` (the harness owns the catalogue + ranges; the stage owns WHEN). Returns a dict of the
    realised values -> set onto the stage; ``sample_build_colours`` later records them onto ``BuildDR`` for the
    trace. Each field has an env-var range/scale hook (DR_OTABLE_SCALE / DR_SIDECAM_SCALE; 0 toggles it off)."""
    gw = float(rng.rand() * OTABLE_GROW_W * _dr_scale("DR_OTABLE_SCALE"))   # width grow (both y edges)
    gl = float(rng.rand() * OTABLE_GROW_L * _dr_scale("DR_OTABLE_SCALE"))   # length grow AWAY from the arm only
    sdz = float(rng.rand() * SIDECAM_DZ_MAX * _dr_scale("DR_SIDECAM_SCALE"))  # side-cam height raise (re-framed)
    names = sorted(LIGHT_COLORS)
    lname = names[int(rng.randint(len(names)))]
    lint = float(LIGHT_INT_LO + rng.rand() * (LIGHT_INT_HI - LIGHT_INT_LO))
    return dict(otable_grow_w=gw, otable_grow_l=gl, sidecam_dz=sdz,
                light_name=lname, light_color=LIGHT_COLORS[lname], light_intensity=lint)


def sample_post_build(N, stage, robot) -> dict:
    """POST-BUILD per-env draws, in the EXACT pre-refactor order: (1) the 50/50 distractor on/off mask, then
    (2) the robot-link friction ratio band (1.0 +/- 0.3, DR_FRIC_SCALE-scaled, clamped >=0). Drawn here (not in
    sample_env_phys) because they happen AFTER ``stage.build()`` in the collector — preserving that boundary is
    part of the RNG-order parity. Returns ``{has_dist, fric}``."""
    rng = stage.rng
    has_dist = rng.rand(N) < 0.5
    fs = _dr_scale("DR_FRIC_SCALE")
    fric = (1.0 + (rng.rand(N, robot.entity.n_links) - 0.5) * 0.6 * fs).clip(0.0).astype(np.float32)
    return dict(has_dist=has_dist, fric=fric)
