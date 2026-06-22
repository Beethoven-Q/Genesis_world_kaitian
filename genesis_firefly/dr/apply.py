# SPDX-License-Identifier: Apache-2.0
"""DR APPLICATION — inject the sampled DR into the sim. Two halves (docs/domain_randomization.md):

  * ``apply_build_dr``  : BEFORE ``stage.build()`` — spawn the grasp target + the bowl with their per-build
                          render COLOURS (Nyx bakes colour at build, so colour must be set at spawn time). The
                          per-build table TEXTURE is applied by the stage itself (it owns its visual setup).
  * ``apply_env_dr``    : AFTER ``stage.build()`` — the BATCHED per-env setters: object-table HEIGHT (+ re-glue
                          the textured top), bowl POSE, target POSE + YAW, target MASS-shift, robot-link FRICTION
                          ratio, and the per-env HDRI env-map (applied by the stage's per-env env_maps at build;
                          recorded here for the trace). All setters are batched over N — no per-env Python loop.

PARITY (Phase 1): the spawn morphs/materials + the batched setters are byte-for-byte what ``tasks/pickplace.py``
did inline. ``apply_build_dr`` reuses ``world.object_factory.spawn_target`` (the one shared object builder) and
the SAME bowl morph the task used; ``apply_env_dr`` issues the SAME ``set_pos`` / ``set_quat`` /
``set_mass_shift`` / ``set_friction_ratio`` calls in the SAME order.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import genesis as gs

import world.object_factory as obj_factory
from world.firefly_scene import BOWL_HALF_H

_BOWL_OBJ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets/objects/ycb/bowl_clean.obj")   # Nyx-safe bowl visual (textured USD segfaults Nyx)


def scaled_target_spec(task_spec, build_dr):
    """The target ObjectSpec with the per-build SIZE DR applied (scope B): a NEW frozen spec whose ``scale`` is
    ``spec.scale * build_dr.obj_scale`` so EVERY downstream consumer adapts automatically -- spawn morph geometry,
    density (``_rho_for`` reads ``scaled_extents()``), and the grasp planner (it reads ``scaled_extents()`` /
    ``rest_root_z`` / ``long_axis_local`` off the spec). No planner change is needed; this is the whole size-DR
    application. ``obj_scale==1.0`` returns the original spec object (the regression baseline, no realloc)."""
    obj_scale = float(getattr(build_dr, "obj_scale", 1.0) or 1.0)
    spec = task_spec.target_spec
    if abs(obj_scale - 1.0) < 1e-9:
        return spec
    return dataclasses.replace(spec, scale=float(spec.scale) * obj_scale)


def apply_build_dr(stage, task_spec, build_dr, pos_xy=(0.40, 0.18), z=0.30):
    """BEFORE build: spawn the grasp TARGET (with its per-build render colour + SIZE + faithful grasp collider, via
    the shared object factory) and the BOWL (with its per-build distinct colour). Returns ``(target_entity,
    bowl_entity)``. Their per-env pose/yaw/mass/friction DR is applied by ``apply_env_dr`` after build.

    SIZE DR (scope B): the target is spawned from ``scaled_target_spec`` (spec.scale * build_dr.obj_scale) so its
    geometry + collider + mass are all the scaled body; the grasp planner reads the SAME scaled spec (the task
    must use the spec this returns). For parity the task should re-read the scaled spec via ``scaled_target_spec``."""
    spec = scaled_target_spec(task_spec, build_dr)
    target = obj_factory.spawn_target(stage.scene, spec, build_dr.target_color, pos_xy=pos_xy, z=z)
    bowl = stage.scene.add_entity(
        gs.morphs.Mesh(file=_BOWL_OBJ, convexify=True, decompose_object_error_threshold=0.04, decimate=False),
        material=gs.materials.Rigid(rho=400.0, friction=1.0),
        surface=gs.surfaces.Smooth(color=build_dr.bowl_color))
    return target, bowl


def apply_env_dr(stage, task_spec, target, bowl, env_dr, robot, spec=None):
    """AFTER build: apply every IMPLEMENTED per-env scope-A/B value as a BATCHED setter (no per-env loop).

      A (scene)  : object-table HEIGHT -> move the (size-DR'd) otable Box + re-glue its textured top to tabZ;
                   TABLE FRICTION -> per-env friction ratio on the table Boxes (decoupled from texture).
      B (object) : bowl POSE; target POSE + in-plane YAW; target MASS-shift; target OBJECT FRICTION (a per-env
                   ratio on the target, distinct from the robot-link knob); (robot-link FRICTION ratio — a
                   scene-level grasp-realism knob applied to the whole URDF).
      C (visual) : the per-env HDRI is applied by the stage's per-env env_maps at build; recorded for the trace.

    ``spec`` = the SIZE-DR'd target spec (from ``scaled_target_spec``); defaults to the unscaled task spec for
    back-compat. Records the realised per-env HDRI paths onto ``env_dr.hdrs`` so ``dr/plan.py`` can write the
    trace. Returns nothing (mutates the sim + env_dr.hdrs)."""
    spec = task_spec.target_spec if spec is None else spec
    lay = stage.lay
    N = stage.n_envs
    ho = lay.object_table_height
    tabZ, bowx, bowy = env_dr.tabZ, env_dr.bowx, env_dr.bowy
    cubx, cuby, yaw = env_dr.cubx, env_dr.cuby, env_dr.yaw

    # --- scope A: object-table HEIGHT (move the GROWN-size Box, then re-glue the textured top so it stays flush) ---
    # Use the stage's grown otable centre x (size DR pins the seam edge, grows away from the arm) so the Box +
    # textured top track the size-DR'd table at the per-env height.
    otable_cx = getattr(stage, "otable_cx", lay.seam_x + lay.object_table_depth / 2)
    stage.otable.set_pos(np.stack([np.full(N, otable_cx), np.zeros(N), tabZ - ho / 2], 1).astype(np.float32))
    stage.set_otable_top_z(tabZ)

    # --- scope A: TABLE FRICTION (per-env, decoupled from the texture) ---
    if getattr(env_dr, "table_fric", None) is not None:
        stage.set_table_friction(env_dr.table_fric)

    # --- scope B: bowl + target POSE + target YAW ---
    bowl.set_pos(np.stack([bowx, bowy, tabZ + BOWL_HALF_H + 0.003], 1).astype(np.float32))
    cube_top = tabZ + spec.scaled_extents()[2] / 2 + 0.002
    target.set_pos(np.stack([cubx, cuby, cube_top + 0.01], 1).astype(np.float32))
    qz = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1).astype(np.float32)
    target.set_quat(qz)

    # --- scope B: target MASS-shift + target OBJECT FRICTION + robot-link FRICTION ratio (the grasp-realism knob) ---
    try:
        target.set_mass_shift(env_dr.mass_shift)
        if getattr(env_dr, "obj_fric", None) is not None:          # per-object friction band (distinct knob)
            nl = target.n_links
            target.set_friction_ratio(np.tile(env_dr.obj_fric[:, None], (1, nl)).astype(np.float32))
        robot.entity.set_friction_ratio(env_dr.fric)
    except Exception as e:                                          # keep the collector's non-fatal behaviour
        print(f"[DR] mass/obj-fric/link-fric DR skipped: {e}", flush=True)

    # --- scope C: record the realised per-env HDRI paths (sampled+applied by the stage) for the trace ---
    env_dr.hdrs = list(getattr(stage, "hdrs", []))
