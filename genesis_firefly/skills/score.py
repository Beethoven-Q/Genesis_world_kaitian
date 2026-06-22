# SPDX-License-Identifier: Apache-2.0
"""Reusable PICK-PLACE PLACEMENT scorer — PHYSICAL success (god-mode ground truth), spec-aware.

WHY A SEPARATE SCORE SKILL: ``skills/pick_place.py::score_pick_place`` is the GENERIC, object-agnostic
pick-place verdict (start/lifted/final positions -> grasped/placed booleans). This module is the SPEC-AWARE
collector scorer the cube->bowl task evolved: it derives the placed-XY tolerance and the height band FROM THE
OBJECT SPEC (a tight 6 cm for a cube; a looser, body-size-scaled band for an elongated/flat object draped across
the bowl mouth), and adds the task's ``through_wall`` geometric check (a target lodged in the bowl WALL rather
than resting in the cavity). It is the placement scorer ANY pick-place-into-a-container task reuses; the
penetration GATE stays ``skills/penetration.py`` (this scorer never owns the collision verdict).

PARITY: the arithmetic is extracted verbatim from tasks/pickplace.py::collect (the placed / wall_pen block), so
the cube 8/8 and the apple/pen runs score IDENTICALLY. The penetration / degenerate rejection is applied by the
caller AFTER this scorer (``placed &= ~penetrating & ~degenerate``) so this module stays a pure placement verdict.
"""
from __future__ import annotations

import numpy as np


def score_placement(obj_final_pos, ee_pos, bowx, bowy, tabZ, rim_z, lift_cm, spec,
                    *, lift_thresh_cm=3.0, floor_margin_m=0.005, ee_clear_m=0.08):
    """Per-env PHYSICAL placement verdict for a pick-place-into-bowl task (vectorised over N).

      placed = grasped (``lift_cm`` > ``lift_thresh_cm``) AND the target came to REST IN the bowl: its bbox
               BOTTOM sits above the table (not on the floor) and below the rim + a spec-aware top margin (not
               perched well above / hovering), its XY is within a spec-aware mouth radius, AND it is CLEAR of the
               end-effector (>= ``ee_clear_m``) -- catches a thin object stuck to the gripper that retreats over
               the bowl and would otherwise read as placed.

    The placed-XY tolerance + height band come from the SPEC: a cube uses a tight 6 cm radius + ~1 cm top
    margin; a non-cube uses min(8.5cm, max(6cm, spec.place_xy_tol_cm)) + a top margin of ~one body-half (an
    elongated/flat body draped across the ~7.5 cm bowl mouth rests with its bbox centre higher than a compact one).

    Inputs are per-env arrays (or broadcastable scalars): ``obj_final_pos`` (N,3), ``ee_pos`` (N,3), ``bowx/bowy``
    (N,), ``tabZ`` (N,), ``rim_z`` (N,), ``lift_cm`` (N,). Returns (placed:(N,) bool, metrics:dict).
    """
    objf = np.asarray(obj_final_pos, float)
    eep = np.asarray(ee_pos, float)
    ch2 = spec.scaled_extents()[2] / 2
    rxy = np.hypot(objf[:, 0] - bowx, objf[:, 1] - bowy)
    # placed-XY tolerance from the SPEC: a cube's tight 6cm stays exactly 6cm; bigger objects whose bbox centre
    # can rest a few cm off the bowl centre while the body still lies IN the bowl use the spec tol, capped at the
    # bowl mouth radius so it never accepts a target resting OUTSIDE the bowl.
    place_r = 0.06 if spec.is_cube else min(0.085, max(0.06, spec.place_xy_tol_cm / 100.0))
    # height band: a compact object's bbox centre rests near the bowl floor (< rim); an ELONGATED/flat target
    # draped across the bowl mouth rests with its bbox centre HIGHER (part of the body bridges the rim), so allow
    # the centre up to ~one body-half above the rim for non-cube targets (still rejects a target perched well
    # ABOVE the bowl or stuck on a finger -- the ee-distance test below catches the held case).
    top_margin = 0.01 if spec.is_cube else float(ch2 + 0.02)
    placed = (lift_cm > lift_thresh_cm) & (rxy < place_r) & (objf[:, 2] - ch2 > tabZ - floor_margin_m) & \
             (objf[:, 2] - ch2 < rim_z + top_margin) & (np.linalg.norm(objf - eep, axis=1) > ee_clear_m)
    return placed, dict(rxy=rxy, place_r=place_r, top_margin=top_margin, ch2=ch2)


def through_wall(obj_final_pos, bowx, bowy, tabZ, rim_z, spec):
    """Task-specific GEOMETRIC through-wall metric (a useful diagnostic, NOT the authoritative gate -- that is
    skills/penetration.py). True where the target's bbox bottom sits inside the annulus of the bowl WALL
    (rxy in [6.5, 11] cm, bottom above the table+1.2cm but below the rim) OR clearly below the table top
    (tunnelled through the floor). Vectorised; ``obj_final_pos`` (N,3), bowx/bowy/tabZ/rim_z (N,)."""
    objf = np.asarray(obj_final_pos, float)
    rxy = np.hypot(objf[:, 0] - bowx, objf[:, 1] - bowy)
    bottom = objf[:, 2] - spec.scaled_extents()[2] / 2
    return (((rxy > 0.065) & (rxy < 0.11) & (bottom > tabZ + 0.012) & (bottom < rim_z)) | (bottom < tabZ - 0.015))
