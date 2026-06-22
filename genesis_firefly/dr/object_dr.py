# SPDX-License-Identifier: Apache-2.0
"""SCOPE B — OBJECT / TASK domain randomization, keyed by object name (per-object, REALISTIC only).

Scope B is the per-object half of the DR catalogue (docs/domain_randomization.md scope B). Unlike scopes A
(scene) and C (visual) — which are AUTOMATIC for every task — scope B is named per object/task: the task lists
which objects it spawns and which scope-B fields apply, and this module supplies the per-object ranges/policies.

The authoritative per-object data already lives in ``registry/object_spec.py`` (one declarative ``ObjectSpec``
per object: mass, friction, the REALISTIC ``target_palette`` / ``native_texture`` colour policy, the elongated/
is_cube/round grasp hints that drive pose). This module is the DR-facing VIEW of that data — it does NOT
duplicate the numbers, it reads them off the spec and pairs them with the (Phase-1) sample/apply ranges.

  * COLOUR   — per-object palette / native-texture / free-random policy (``world.object_factory.target_color``,
               which reads ``spec.target_palette`` + ``spec.native_texture``). The RECOGNIZABILITY RULE
               (docs/domain_randomization.md) is enforced there: identity-bearing objects keep a native texture
               / realistic palette; only the generic cube gets a free random colour.
  * MASS     — a per-object additive mass-shift band about ``spec.mass`` (Phase-1: the cube's +/- 0.02 kg).
  * FRICTION — Phase 1 randomizes the ROBOT-LINK friction ratio (a scene-level grasp-realism knob), NOT a
               per-object friction; the object's own ``spec.friction`` is used as-is. A per-object friction band
               is a documented Phase-2 stub.
  * POSE     — per-env XY + in-plane yaw, sampled within the task-permitted / arm-permitted envelope. The pose
               RANGES are task-layout-specific (they depend on the arm's reachable workspace + inter-object
               clearance), so the literal pose numbers stay in the task; this module records that pose IS a
               scope-B field for every object.
  * SIZE     — per-object realistic scale band (usually +/-10%). Phase-2 STUB (not sampled/applied yet).

Phase 1 is a PARITY refactor: the colour/mass ranges below are exactly what ``tasks/pickplace.py`` +
``world/object_factory.py`` already use. The ``ObjectDR`` view just makes them addressable per object name.
"""
from __future__ import annotations

from dataclasses import dataclass

from registry.object_spec import REGISTRY


# Phase-1 mass-shift HALF-WIDTH (kg) — the additive band about spec.mass. The cube collection uses +/- 0.02 kg
# (sampled as ``(rand-0.5) * 0.04``); this is the per-object default. A future per-object override would key off
# the object name here (a heavier book tolerates a wider absolute band, a light pen a narrower one).
MASS_SHIFT_HALFWIDTH_KG = 0.02


@dataclass(frozen=True)
class ObjectDR:
    """The scope-B DR VIEW for ONE object (read off its ObjectSpec). Carries the colour policy class, the
    mass-shift band, and flags for which scope-B fields are randomized for this object. Pure data — the actual
    sampling/application lives in ``dr/sampler.py`` / ``dr/apply.py`` / ``world.object_factory``."""
    name: str
    colour_policy: str            # "native_texture" | "palette" | "fixed" | "free" (see classify_colour)
    mass_shift_halfwidth_kg: float
    randomizes_pose: bool = True  # every object's pose (XY + yaw) is per-env DR within the task envelope
    randomizes_mass: bool = True
    # --- Phase-2 STUBS (documented; NOT sampled/applied yet) ---
    size_band_frac: float = 0.10        # STUB: realistic +/-10% scale band (per-build); not applied in Phase 1
    randomizes_size: bool = False       # STUB: flip + wire into sampler/apply for Phase 2
    friction_band: float = 0.0          # STUB: per-object friction band; Phase 1 uses spec.friction as-is


def classify_colour(spec) -> str:
    """The object's COLOUR-DR policy class (mirrors world.object_factory.target_color — the single source of the
    actual render colour). NATIVE-TEXTURE -> the object's own UV skin (no colour DR); PALETTE -> pick + small
    jitter; FIXED -> a one-entry palette (effectively no DR); FREE -> the cube's free distinct-from-table colour."""
    pal = getattr(spec, "target_palette", None)
    if not pal:
        return "free"                                    # cube: free random distinct-from-table colour
    if getattr(spec, "native_texture", None):
        return "native_texture"                          # native UV skin wins; flat colour is only a fallback
    return "fixed" if len(pal) == 1 else "palette"       # one entry -> effectively fixed; else pick+jitter


def object_dr(name: str) -> ObjectDR:
    """The scope-B DR view for an object by registry name (the task names which objects apply)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown object {name!r} (choose from {sorted(REGISTRY)})")
    spec = REGISTRY[name]
    return ObjectDR(name=name, colour_policy=classify_colour(spec),
                    mass_shift_halfwidth_kg=MASS_SHIFT_HALFWIDTH_KG)
