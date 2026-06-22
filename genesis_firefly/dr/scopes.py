# SPDX-License-Identifier: Apache-2.0
"""SCOPE DEFINITIONS — the TASK-AGNOSTIC DR field tables that apply to EVERY task.

This is the declarative half of the DR harness (docs/domain_randomization.md "How it's applied"). Two scopes
are AUTOMATIC for every task and live here:

  * ``SCENE_DR``  (scope A — manipulation-stage / scene): table texture, object-table height, table friction, …
  * ``VISUAL_DR`` (scope C — visual background): immersive HDRI background + light.

Scope B (object / task) is PER-OBJECT and lives in ``dr/object_dr.py`` (keyed by object name); a task names
which scope-B fields apply via its ``TaskSpec``. Scopes A and C come free — a task NEVER writes their logic.

Each field is a ``DRField`` describing WHAT it is, WHERE it varies (per-build vs per-env), and its RANGE. The
``implemented`` flag marks the fields the harness actually samples+applies. As of Phase 2 ALL of scope A / C are
implemented: the table-size grow, the side-camera pose, the table friction, and the per-build light DR all flow
through ``dr/sampler.py`` -> ``dr/apply.py`` (table-size + side-cam + light realised inside the stage at build
because COLOR/SIZE/geometry bake at build; the stage records the realised values for the trace). The ONE honest
limit (verified in the Nyx SDK, see ``light`` below): the per-env render loop can only switch the ENV-MAP
(``set_env_map(env_index)``), NOT the directional light — so the LIGHT DR is PER-BUILD (not per-env). The HDRI
already supplies per-env image-based lighting, so per-env visual variety is preserved.

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
    "object_table_size": DRField(
        "object_table_size", PER_BUILD,
        "width +0..OTABLE_GROW_W (def 0.18m); length extends ONLY away from the arm table (seam end fixed) "
        "+0..OTABLE_GROW_L (def 0.22m); the textured top Plane rescales to stay flush",
        implemented=True,
        note="BuildDR.otable_grow_w/l drawn in sampler; the stage rebuilds the otable Box+top at that size "
             "(geometry bakes at build); seam end pinned, growth away from the arm; realised size read for trace"),
    "side_camera_pose": DRField(
        "side_camera_pose", PER_BUILD, "side cam height +0..5cm; pitch DOWN to re-frame so the workspace stays framed",
        implemented=True,
        note="BuildDR.sidecam_dz drawn in sampler; the stage raises the cam_side sensor + the visible D435i rig "
             "and re-aims the lookat lower (pitch-down); side cam ONLY (the wrist cams are never touched)"),
    "table_friction": DRField(
        "table_friction", PER_ENV, "uniform band metal(~0.5)<->wood(~0.9)<->wool/fabric(~1.3), DECOUPLED from texture",
        implemented=True,
        note="EnvDR.table_fric drawn in sampler; apply_env_dr sets it on the collidable table Boxes per-env "
             "(set_friction_ratio about the Box's spawn friction); the texture/colour is never coupled to it"),
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
    "light": DRField(
        "light", PER_BUILD,
        "directional key-light colour (orange/white/yellow/light-blue/sunlight) + intensity band (~0.7..1.7)",
        implemented=True,
        note="HONEST LIMIT (verified in the Nyx SDK): the per-env render loop can only switch the ENV-MAP "
             "(renderer.set_env_map(env_index)) -- the directional LIGHT bakes at build (scene_asset.set_light) "
             "and is NOT per-env settable. So light DR is PER-BUILD (still adds across-build variety); the HDRI "
             "already supplies per-env image-based lighting. BuildDR.light_* drawn in sampler; the stage bakes the "
             "coloured key light at build and records the colour/intensity for the trace"),
}


def implemented_fields(scope: dict[str, DRField]) -> list[str]:
    """The field names in a scope that Phase 1 actually samples+applies (the rest are documented Phase-2 stubs)."""
    return [n for n, f in scope.items() if f.implemented]
