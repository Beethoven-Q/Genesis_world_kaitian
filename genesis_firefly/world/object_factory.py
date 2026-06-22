# SPDX-License-Identifier: Apache-2.0
"""The ONE shared place that builds any ``ObjectSpec`` into a Genesis sim entity — collision + visual + native
texture — for EVERY task (agent-native modularity).

A task should NEVER hand-roll object spawning. Whatever the object (procedural cube/sphere, a USD-sourced
apple/banana/pen, a textured book/tennis ball), THIS module is the single path that maps a declarative
``ObjectSpec`` -> a real collidable rigid body with the correct collider (faithful convex decomposition or a
smooth single hull for the GRASP target, a stable single hull for distractors) and the correct Nyx-safe visual
(procedural PBR for cube/sphere, a clean .obj + flat palette colour, or the object's OWN UV-mapped native
texture via ``gs.textures.ImageTexture``). Extracting it out of ``tasks/pickplace.py`` makes objects standalone
and reusable: a new task imports ``spawn_target`` / ``spawn_distractors`` and gets the SAME verified collision +
texture behaviour for free, with no copy-paste fork.

What lives here (moved verbatim from pickplace.py, behaviour byte-identical):
  - ``target_color``            : the grasp-target render colour (per-object palette / native-texture / free).
  - ``_target_usd_surface``     : the USD grasp-target surface (native ImageTexture or flat palette colour).
  - ``spawn_target``            : the grasp-target entity (cuboid / sphere / usd-mesh + faithful grasp collider).
  - ``_spawn_distractor_entity``: one distractor as a collidable rigid body with a Nyx-safe visual.
  - ``spawn_distractors``       : the per-build distractor entities (types + batched per-env poses).
  - ``build_object``            : the generic "spawn one graspable object from its spec" path (consolidated from
                                  the orphaned ``firefly_scene.build_object`` so there is ONE object-building path).

The distractor TYPE/POSE sampling helpers (``choose_distractor_types`` / ``sample_distractor_poses`` and the
corridor-clearance geometry) stay in the task — those are task-layout-specific (they depend on the arm's swept
corridor), not object-construction. ``spawn_distractors`` receives them via the ``stage``/``dr`` it is passed.
"""
from __future__ import annotations

import os

import numpy as np
import genesis as gs

from registry.object_spec import REGISTRY
from world.firefly_scene import OBJECTS, _rho_for


def build_object(scene, spec, pos_xy, table_top_z, mass=None):
    """Spawn one graspable object from its ObjectSpec (source = usd | cuboid | sphere), resting on the table."""
    x, y = pos_xy
    z = spec.rest_root_z(table_top_z)
    mat = gs.materials.Rigid(rho=_rho_for(spec, mass), friction=float(spec.friction[0]))
    if spec.source == "cuboid":
        morph = gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(x, y, z))
        surf = gs.surfaces.Plastic(color=spec.color, roughness=0.6)
    elif spec.source == "sphere":
        morph = gs.morphs.Sphere(radius=float(spec.scaled_extents()[0] / 2), pos=(x, y, z))
        surf = gs.surfaces.Rough(color=spec.color)
    else:
        morph = gs.morphs.USD(file=str(OBJECTS / spec.usd_subpath), pos=(x, y, z),
                              scale=spec.scale, convexify=True)
        surf = None
    return scene.add_entity(morph, material=mat, surface=surf)


def _spawn_distractor_entity(scene, spec, pos_xy, z):
    """Spawn ONE distractor as a real collidable rigid body, but with a NYX-SAFE visual. cuboid/sphere are
    procedural (Nyx-safe already); USD-sourced objects (apple/banana/pen/book/tennis) render via their extracted
    clean .obj mesh (Nyx SEGFAULTS on the textured USD material binding, the same reason the bowl uses
    bowl_clean.obj). NATIVE TEXTURE (DR-strategist RECOGNIZABILITY RULE): a distractor with a declared
    ``native_texture`` renders its OWN UV-mapped photoreal skin via ``_target_usd_surface`` -- the SAME Nyx-safe
    ImageTexture idiom the GRASP TARGET uses -- so a distractor apple/banana/pen/book/tennis reads as a REAL
    object, never a flat pink blob / colour stick. Distractors with NO native_texture (none currently) fall back
    to the realistic flat ``dist_color``. Collision fidelity (single convex hull) is unchanged from before."""
    x, y = pos_xy
    # distractors only need to REST (not be grasped), so use firm friction so they sit put under a graze.
    mat = gs.materials.Rigid(rho=_rho_for(spec), friction=max(1.1, float(spec.friction[0])))
    if spec.source == "cuboid":
        return scene.add_entity(gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(x, y, z)),
                                material=mat, surface=gs.surfaces.Plastic(color=spec.color, roughness=0.6))
    if spec.source == "sphere":
        return scene.add_entity(gs.morphs.Sphere(radius=float(spec.scaled_extents()[0] / 2), pos=(x, y, z)),
                                material=mat, surface=gs.surfaces.Rough(color=spec.color))
    # USD object -> Nyx-safe clean mesh + native texture (or flat realistic fallback). Collision = a SINGLE
    # convex hull (no decomposition): a distractor is never grasped, so it doesn't need a faithful concave
    # collider, and a hull gives a FLATTER, stable resting base -> a curved banana doesn't slowly roll/creep off
    # a rounded decomposed facet (that intrinsic creep, not the arm, was reading as a 2-3cm "knock" in the
    # displacement metric). The VISUAL now reuses _target_usd_surface (native ImageTexture, Nyx-safe) so the
    # distractor shows its OWN photoreal skin instead of a flat colour (Fix 2). dist_color is the flat fallback.
    col = spec.dist_color or spec.color
    return scene.add_entity(
        gs.morphs.Mesh(file=str(OBJECTS / spec.mesh_subpath), pos=(x, y, z), scale=spec.scale, convexify=True),
        material=mat, surface=_target_usd_surface(spec, col))


def target_color(spec, free_color, rng):
    """The GRASP TARGET's render colour (DR-strategist-owned per-object COLOUR POLICY; see the colour-policy table
    in .claude/workbooks/dr_workbook.md). Three classes:
      * NATIVE TEXTURE (apple): the spec declares ``native_texture`` -> the object renders its OWN UV-mapped skin
        in spawn_target, so the flat colour here is unused (NO colour DR). We return a FIXED palette base (no
        jitter) purely as the NATIVE_TEX=0 fallback colour -- the texture is the real colour.
      * REALISTIC PALETTE with colour-DR (banana/pen): pick ONE palette entry + a SMALL per-channel jitter
        (+/-0.04) for natural variation (banana yellow/green not pink; pen black/blue/red).
      * FIXED colour, no DR (tennis_ball): a single-entry palette stays its one true regulation colour (the
        +/-0.04 jitter on one entry is negligible -> effectively fixed).
      * FREE random (cube): no palette -> the task's FREE distinct-from-table colour (cube byte-for-byte).
    This is the FIX for the bug where the cube's free random colour was applied to EVERY target (the PINK banana)."""
    pal = getattr(spec, "target_palette", None)
    if not pal:
        return free_color                                       # cube: keep the free random distinct colour
    if getattr(spec, "native_texture", None):                   # native-texture object: fixed fallback, NO DR
        return tuple(np.asarray(pal[0], float).tolist())        # (texture wins in spawn_target; colour unused)
    base = np.asarray(pal[int(rng.randint(len(pal)))], float)
    jit = (rng.rand(3) - 0.5) * 0.08                            # +/-0.04 per channel -> subtle natural variation
    return tuple(np.clip(base + jit, 0.0, 1.0).tolist())


def _target_usd_surface(spec, color):
    """The GRASP-TARGET visual surface for a USD-sourced object. NATIVE TEXTURE (DR-strategist colour policy):
    if the spec declares a ``native_texture`` (a UV-mapped diffuse image, e.g. the apple's apple_02.png) AND the
    object's clean .obj actually carries UVs, render the object's OWN photoreal skin via gs.textures.ImageTexture
    -- the SAME idiom the table tops use (verified in Nyx, no segfault). This makes the apple a realistic textured
    apple instead of a flat pink blob. Otherwise (banana/pen clean.obj have NO UVs; the cube/sphere don't reach
    here) fall back to the realistic flat ``color`` from the per-object palette. NATIVE_TEX=0 forces the flat
    fallback (ablation)."""
    use_tex = bool(getattr(spec, "native_texture", None)) and os.environ.get("NATIVE_TEX", "1") != "0"
    if use_tex:
        tex_path = OBJECTS / spec.native_texture
        return gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(tex_path)), roughness=0.5)
    return gs.surfaces.Plastic(color=color, roughness=0.5)


def spawn_target(scene, spec, color, pos_xy=(0.40, 0.18), z=0.30):
    """Spawn the GRASP TARGET as a real collidable rigid body with a FAITHFUL grasp collider + a Nyx-safe visual.
    The target is the one object that gets GRASPED, so (unlike a distractor) its USD-mesh collider is a convex
    DECOMPOSITION (coacd, the same recipe as the bowl), not a single hull, so the gripper closes on the real
    elongated/flat shape (a banana's curve, a pen's thin body, a book's flat slab) rather than a fat envelope.
    Procedural cube/sphere keep their exact box/sphere collider (already faithful). The visual is the procedural
    box/sphere for those sources, or the Nyx-safe clean .obj for USD sources (the textured USD segfaults Nyx, the
    same reason the bowl + distractors render from clean meshes). VISUAL COLOUR: a USD target with a declared
    NATIVE TEXTURE (the apple) renders its OWN UV-mapped skin (apple_02.png) via _target_usd_surface; otherwise
    ``color`` (the per-object realistic palette) is used so the target stays visible against the randomized table.
    Returns the entity (its per-env pose/yaw/mass DR is applied by the caller after build, exactly as the cube)."""
    x, y = pos_xy
    fr = float(os.environ.get("TGT_FRIC", spec.friction[0]))   # higher friction -> a shallower grip still holds
    mat = gs.materials.Rigid(rho=_rho_for(spec), friction=fr)
    if spec.source == "cuboid":
        # EXACT cube path (byte-for-byte for the regression): procedural box, plastic PBR, the spec's grasp friction.
        return scene.add_entity(gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(x, y, z)),
                                material=gs.materials.Rigid(rho=600.0, friction=1.0),
                                surface=gs.surfaces.Plastic(color=color, roughness=0.35))
    if spec.source == "sphere":
        return scene.add_entity(gs.morphs.Sphere(radius=float(spec.scaled_extents()[0] / 2), pos=(x, y, z)),
                                material=mat, surface=gs.surfaces.Rough(color=color))
    # USD object -> grasp collider on the clean mesh + the distinct-from-table colour.
    # COLLIDER CHOICE (spec.grasp_single_hull):
    #  * decomposition (default): coacd splits the mesh into solid hulls -> faithful concave shape (needed for a
    #    hollow/handled object). But for a ROUNDED CONVEX body (banana) the decomposition's internal hull seams +
    #    the firm high-kp pinch drive the claw DEEP into a seam -> >9mm penetration AND huge constraint forces
    #    that NaN the Newton solver (verified: dz<=-0.005 banana -> 'Invalid constraint forces' crash).
    #  * single convex hull (grasp_single_hull=True): a SMOOTH convex envelope of the body. A banana is already
    #    near-convex, so the hull is a faithful smooth banana the claws pinch CLEANLY -- stable contact, shallow
    #    penetration, no NaN. This is the SAME stable single-hull collider the distractors use, promoted to the
    #    grasp target for convex rounded objects.
    single = os.environ.get("TGT_SINGLE_HULL")
    single = (single == "1") if single is not None else bool(getattr(spec, "grasp_single_hull", False))
    if single:
        return scene.add_entity(
            gs.morphs.Mesh(file=str(OBJECTS / spec.mesh_subpath), pos=(x, y, z), scale=spec.scale, convexify=True),
            material=mat, surface=_target_usd_surface(spec, color))
    decomp = float(os.environ.get("TGT_DECOMP", getattr(spec, "grasp_decompose_err", 0.04)))
    return scene.add_entity(
        gs.morphs.Mesh(file=str(OBJECTS / spec.mesh_subpath), pos=(x, y, z), scale=spec.scale,
                       convexify=True, decompose_object_error_threshold=decomp, decimate=False),
        material=mat, surface=_target_usd_surface(spec, color))


def spawn_distractors(stage, dr, rng, choose_types, sample_poses, target="cube"):
    """Create the per-build distractor ENTITIES (called before stage.build()) and return (entities, names,
    xy, yaw). TYPES are per-build (chosen here with the stage rng, EXCLUDING the target type); POSES are per-env
    (placed after build via the returned batched arrays). Each entity is a real collidable rigid body from the
    REGISTRY (realistic colour/size/mass/friction) and drops+settles with the target/bowl during the settle.

    ``choose_types(rng, target=target)`` and ``sample_poses(n_envs, specs, dr, rng, lay)`` are the TASK's
    layout-specific samplers (the distractor count/identity policy + the corridor-aware pose placement) passed in
    by the caller, so this object-construction module stays free of task-corridor geometry."""
    names = choose_types(rng, target=target)
    specs = [REGISTRY[nm] for nm in names]
    lay = stage.lay
    ents = []
    for spec in specs:
        # parked off to the far corner at build; the real per-env pose is set after build() (set_pos overrides).
        e = _spawn_distractor_entity(stage.scene, spec,
                                     pos_xy=(lay.seam_x + lay.object_table_depth - 0.05, 0.0),
                                     z=spec.rest_root_z(lay.object_table_height))
        ents.append(e)
    xy, yaw = sample_poses(stage.n_envs, specs, dr, rng, lay)
    return ents, names, xy, yaw
