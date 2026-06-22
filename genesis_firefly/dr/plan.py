# SPDX-License-Identifier: Apache-2.0
"""DR PLAN — assemble the per-demo DR attrs written into each HDF5 demo (the traceable ``DRPlan`` of
docs/domain_randomization.md). Every demo records the EXACT DR values it sampled so the DR strategist can
correlate an outcome (success / penetrating / degenerate) with WHERE in the DR space the env landed and measure
the achieved diversity.

This is the single place that knows the attr key names + how to derive the convenience columns (cube<->bowl
clearance, reach). The task calls ``demo_dr_attrs(env_dr, build_dr, e)`` per env and merges the returned dict
into the demo's ``.attrs`` (alongside the outcome attrs the task owns: success / penetrating / degenerate / …).

PARITY (Phase 1): the keys + values are byte-for-byte what ``tasks/pickplace.py`` wrote inline — the ``dr_*``
physics-plan attrs plus the scope-A/B/C trace attrs (``arm``, ``hdr``, ``has_distractors``, ``distractors``).
"""
from __future__ import annotations

import os

import numpy as np


def _dr_scale(name):
    """The in-force per-env range half-width multiplier (DR-strategist sweep hook); 1.0 = the v2 default."""
    try:
        return float(os.environ.get(name, "1.0"))
    except (TypeError, ValueError):
        return 1.0


def demo_dr_attrs(env_dr, e, *, dist_names=None) -> dict:
    """The DR attrs for demo ``e``: the per-env physics-DR plan (``dr_*``) + the scope-A/B/C trace attrs.

    Values are arm-frame (bowl/target y are signed by the active arm side). ``dist_names`` is the per-build
    distractor type list (the task owns the on/off mask -> the distractor string is only written when present).
    Returns a flat ``{attr_name: value}`` dict the task merges into the demo group's ``.attrs``."""
    cubx, cuby = float(env_dr.cubx[e]), float(env_dr.cuby[e])
    bowx, bowy = float(env_dr.bowx[e]), float(env_dr.bowy[e])
    has_dist = bool(env_dr.has_dist[e]) if env_dr.has_dist is not None else False
    attrs = {
        # --- scope-A/B/C TRACE (the realised per-env choices) ---
        "arm": "left" if bool(env_dr.side_is_left[e]) else "right",
        "hdr": os.path.basename(env_dr.hdrs[e]) if env_dr.hdrs else "",
        "has_distractors": has_dist,
        "distractors": ",".join(dist_names) if (has_dist and dist_names) else "",
        # --- the per-env physics-DR PLAN (prefixed dr_ so it never collides with the outcome attrs) ---
        "dr_cubx": cubx, "dr_cuby": cuby,
        "dr_bowx": bowx, "dr_bowy": bowy,
        "dr_tabZ": float(env_dr.tabZ[e]), "dr_yaw": float(env_dr.yaw[e]),
        "dr_mass_shift": float(env_dr.mass_shift[e, 0]),
        "dr_clr": float(np.hypot(cubx - bowx, cuby - bowy)),     # cube<->bowl centre distance
        "dr_reach": float(np.hypot(cubx, cuby)),                 # cube distance from the arm base
        "dr_pose_scale": _dr_scale("DR_POSE_SCALE"),
        "dr_mass_scale": _dr_scale("DR_MASS_SCALE"),
        "dr_fric_scale": _dr_scale("DR_FRIC_SCALE"),
    }
    return attrs
