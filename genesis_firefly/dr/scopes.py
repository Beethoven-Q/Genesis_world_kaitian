# SPDX-License-Identifier: Apache-2.0
"""SCOPE DEFINITIONS — the TASK-AGNOSTIC DR field tables that apply to EVERY task.

This is the declarative half of the DR harness (docs/domain_randomization.md "How it's applied"). Two scopes
are AUTOMATIC for every task and live here:

  * ``SCENE_DR``  (scope A — manipulation-stage / scene): table texture, object-table height, table friction, …
  * ``VISUAL_DR`` (scope C — visual background): immersive HDRI background + light.

Scope B (object / task) is PER-OBJECT and lives in ``dr/object_dr.py`` (keyed by object name); a task names
which scope-B fields apply via its ``TaskSpec``. Scopes A and C come free — a task NEVER writes their logic.

Each field is a ``DRField`` describing WHAT it is, WHERE it varies (per-build vs per-env), and its RANGE. The
``implemented`` flag separates the 8 fields CURRENTLY sampled+applied (Phase 1 parity) from the documented
Phase-2 STUBS (size, table-size, side-cam, table-friction, light) that are NOT yet sampled or applied. Adding a
Phase-2 field = flip ``implemented`` and wire it into ``dr/sampler.py`` + ``dr/apply.py`` (Phase 2 work).

NOTE ON OWNERSHIP (Phase 1): the scope-A texture / scope-C HDRI draws currently happen inside
``world.manipulation_stage.ManipulationStage.__init__`` (it is the reusable world harness and owns its own
visual setup; the table texture + per-env HDRI bake at build time). These tables DOCUMENT those fields so the
``dr/`` package is the single source of truth for the field catalogue; ``dr/plan.py`` reads the realised values
back off the stage for the per-demo trace. Phase 2/3 may migrate the *sampling* of these into ``dr/sampler.py``.
"""
from __future__ import annotations

from dataclasses import dataclass


PER_BUILD = "per_build"   # constant within one parallel build; varies across build-batches (renderer bakes it)
PER_ENV = "per_env"       # free; varies every trial inside one parallel build (transforms / env-map only)


@dataclass(frozen=True)
class DRField:
    """One randomizable field: its name, the variability scope, a human range note, and whether Phase 1 wires it.

    ``rng`` is a human-readable description of the range (the LITERAL numeric ranges live where the field is
    sampled — Phase 1: the stage for A/C-texture/HDRI, ``dr/sampler.py`` for the per-env phys draws). ``applies_to``
    documents the variability (PER_BUILD / PER_ENV). ``implemented=False`` marks a documented Phase-2 STUB that is
    NOT sampled or applied yet (do not change behaviour on it this phase)."""
    name: str
    applies_to: str
    rng: str
    implemented: bool = True
    note: str = ""


# ============================================================================ #
# SCOPE A — manipulation-stage / scene  (AUTOMATIC for every task)
# ============================================================================ #
SCENE_DR: dict[str, DRField] = {
    # --- IMPLEMENTED (Phase 1) ---
    "table_texture": DRField(
        "table_texture", PER_BUILD,
        "choice over the >=10-map albedo pack (wood/steel/tablecloth) in assets/textures/tables/",
        implemented=True,
        note="sampled+applied in ManipulationStage.__init__ (Nyx bakes albedo at build); both tables share it"),
    "object_table_height": DRField(
        "object_table_height", PER_ENV, "object-table top z = nominal +/- 5 cm (uniform)",
        implemented=True, note="EnvDR.tabZ; applied by apply_env_dr via otable.set_pos + stage.set_otable_top_z"),
    # --- PHASE-2 STUBS (documented; NOT sampled or applied yet) ---
    "object_table_size": DRField(
        "object_table_size", PER_BUILD,
        "width +0..dW; length extends ONLY away from the arm table (seam end fixed); texture rescales",
        implemented=False, note="STUB (Phase 2): current size is the minimum"),
    "side_camera_pose": DRField(
        "side_camera_pose", PER_BUILD, "side cam height +0..5cm; pitch re-frames to keep the workspace framed",
        implemented=False, note="STUB (Phase 2): side cam only, never the wrist cams"),
    "table_friction": DRField(
        "table_friction", PER_ENV, "small uniform band, metal<->wood<->fabric, independent of the texture",
        implemented=False, note="STUB (Phase 2): friction stays decoupled from texture"),
}


# ============================================================================ #
# SCOPE C — visual background  (AUTOMATIC for every task)
# ============================================================================ #
VISUAL_DR: dict[str, DRField] = {
    # --- IMPLEMENTED (Phase 1) ---
    "hdri_background": DRField(
        "hdri_background", PER_ENV,
        "choice over the HDRI pool (2K when n_envs<=45, else the 1K pool); the HDRI is floor+walls+light",
        implemented=True,
        note="sampled+applied in ManipulationStage.__init__ (per-env env_maps); realised name read for the trace"),
    # --- PHASE-2 STUBS (documented; NOT sampled or applied yet) ---
    "light": DRField(
        "light", PER_ENV, "colour (orange/white/yellow/light-blue/sunlight) + reasonable brightness/intensity band",
        implemented=False, note="STUB (Phase 2): a fixed neutral key light is used now (LIGHTS in the stage)"),
}


def implemented_fields(scope: dict[str, DRField]) -> list[str]:
    """The field names in a scope that Phase 1 actually samples+applies (the rest are documented Phase-2 stubs)."""
    return [n for n, f in scope.items() if f.implemented]
