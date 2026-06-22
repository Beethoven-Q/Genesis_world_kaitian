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


def demo_dr_attrs(env_dr, e, *, dist_names=None, build_dr=None, stage=None) -> dict:
    """The DR attrs for demo ``e``: the per-env physics-DR plan (``dr_*``) + the scope-A/B/C trace attrs.

    Values are arm-frame (bowl/target y are signed by the active arm side). ``dist_names`` is the per-build
    distractor type list (the task owns the on/off mask -> the distractor string is only written when present).
    ``build_dr`` (the per-build draws) + ``stage`` (the realised build-baked values) add the Phase-2 PER-BUILD
    trace attrs (object size, object-table grow, side-cam raise, light) -- per-build, so constant across the
    build's demos. Returns a flat ``{attr_name: value}`` dict the task merges into the demo group's ``.attrs``."""
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
    # --- Phase-2 PER-ENV physics fields (scope A table friction; scope B object friction) ---
    if getattr(env_dr, "table_fric", None) is not None:
        attrs["dr_table_fric"] = float(env_dr.table_fric[e])
    if getattr(env_dr, "obj_fric", None) is not None:
        attrs["dr_obj_fric"] = float(env_dr.obj_fric[e])
    # --- Phase-2 PER-BUILD fields (object SIZE; object-table GROW; side-cam raise; coloured LIGHT) ---
    if build_dr is not None:
        attrs["dr_obj_scale"] = float(getattr(build_dr, "obj_scale", 1.0))
        attrs["dr_otable_w"] = float(getattr(stage, "otable_width", 0.0)) if stage is not None \
            else float(getattr(build_dr, "otable_grow_w", 0.0))
        attrs["dr_otable_l"] = float(getattr(stage, "otable_depth", 0.0)) if stage is not None \
            else float(getattr(build_dr, "otable_grow_l", 0.0))
        attrs["dr_otable_grow_w"] = float(getattr(build_dr, "otable_grow_w", 0.0))
        attrs["dr_otable_grow_l"] = float(getattr(build_dr, "otable_grow_l", 0.0))
        attrs["dr_sidecam_dz"] = float(getattr(build_dr, "sidecam_dz", 0.0))
        # the realised SIDE-cam world z (calibrated base + the raise) for an absolute trace
        attrs["dr_sidecam_z"] = float(getattr(stage, "sidecam_dz", 0.0)) + 0.8155021 \
            if stage is not None else float(getattr(build_dr, "sidecam_dz", 0.0))
        attrs["dr_light_name"] = str(getattr(build_dr, "light_name", "white"))
        lc = getattr(build_dr, "light_color", (1.0, 1.0, 1.0))
        attrs["dr_light_r"], attrs["dr_light_g"], attrs["dr_light_b"] = float(lc[0]), float(lc[1]), float(lc[2])
        attrs["dr_light_intensity"] = float(getattr(build_dr, "light_intensity", 1.2))
    return attrs
