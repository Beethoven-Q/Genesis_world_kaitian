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
    """Per-build draws (constant within one parallel build). Phase 1: the object render colours + the realised
    per-build scope-A/C visual choices RECORDED off the stage (table texture path; HDRI is per-env)."""
    target_color: tuple            # the grasp-target render colour (palette / free / native-fallback)
    bowl_color: tuple              # the bowl's free distinct colour
    target_free_color: tuple       # the FREE distinct-from-table colour drawn first (cube uses it directly)
    table_texture: object = None   # the per-build table-texture path realised by the stage (record-only)


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
    return EnvDR(side_is_left=side_is_left, sgn=sgn, tabZ=tabZ, bowx=bowx, bowy=bowy,
                 cubx=cubx, cuby=cuby, yaw=yaw, mass_shift=mass_shift)


def sample_build_colours(stage, task_spec) -> BuildDR:
    """PER-BUILD object render colours, drawn on ``stage.rng`` in the EXACT pre-refactor order: (1) the target's
    FREE distinct-from-table colour, (2) the target's realistic-palette colour (cube -> the free colour), (3) the
    bowl's free distinct colour (avoiding the target's hue). Also records the stage's realised per-build table
    texture (for the trace). Returns a ``BuildDR``."""
    spec = task_spec.target_spec
    ch, free_color = stage.distinct_object_color()                 # the target's free random colour (+ hue to avoid)
    tgt_col = obj_factory.target_color(spec, free_color, stage.rng)  # realistic palette colour (cube -> free)
    _, bowl_col = stage.distinct_object_color(ch)
    return BuildDR(target_color=tgt_col, bowl_color=bowl_col, target_free_color=free_color,
                   table_texture=getattr(stage, "table_texture", None))


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
