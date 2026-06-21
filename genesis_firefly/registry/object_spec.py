# SPDX-License-Identifier: Apache-2.0
"""Graspable-object REGISTRY (DATA ONLY — pure, sim-agnostic). The Genesis spawn factory lives in the scene
module (build_object), so this file has NO engine imports and is shared verbatim by both lines.

One declarative ``ObjectSpec`` per object: asset, mass, MEASURED geometry (AABB extents + bbox centre + PCA
long axis), and per-object grasp/handling hints (friction, contact/rest_offset, grasp_dz, release_dz,
place_xy_tol_cm, x_range). Adding an object = adding a spec entry; no per-object code anywhere.

Sources: "usd" (asset under assets/objects/<usd_subpath>), "cuboid" (procedural box, size=extents),
"sphere" (procedural ball, radius=extents[0]/2). Geometry constants MEASURED in RoboLab (inspect_object_usd.py
/ pca_axis.py), reused unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    """Everything needed to spawn + grasp one object. Geometry constants are MEASURED."""
    name: str
    language_name: str
    source: str                     # "usd" | "cuboid" | "sphere"
    mass: float                     # kg (DR multiplies this)
    extents: tuple                  # local AABB size (ex,ey,ez) at scale=1, MEASURED
    local_center: tuple = (0.0, 0.0, 0.0)
    usd_subpath: str | None = None  # under assets/objects/ (for source="usd")
    mesh_subpath: str | None = None  # Nyx-SAFE clean .obj (under assets/objects/) extracted from the USD; used
    #                                  for rendering distractors (Nyx SEGFAULTS on the textured USD material bind)
    dist_color: tuple | None = None  # realistic flat colour for the clean-mesh distractor render (USD texture is
    #                                  lost when extracting the OBJ); None -> fall back to `color`
    native_texture: str | None = None  # UV-mapped DIFFUSE-TEXTURE image (under assets/objects/) for the GRASP-
    #                                  TARGET render -- the object's OWN photoreal skin, used INSTEAD of a flat
    #                                  target_palette colour. Set ONLY when the clean .obj carries usable UVs AND
    #                                  the texture renders in Nyx (apple_clean.obj has full UVs -> apple_02.png; a
    #                                  YCB clean.obj with NO `vt` -- banana/pen -- can't map a texture, so it keeps
    #                                  the realistic palette). When set, the target renders with this image
    #                                  (gs.textures.ImageTexture, the same idiom the table tops use) and gets NO
    #                                  per-frame colour DR (the texture IS the colour). DR-STRATEGIST-OWNED policy
    #                                  (.claude/agents/dr-strategist.md + .claude/workbooks/dr_workbook.md): the
    #                                  per-object NATIVE-TEXTURE / REALISTIC-PALETTE / FIXED-colour classification
    #                                  is the DR strategist's responsibility (see the colour-policy table there).
    scale: float = 1.0
    elongated: bool = False         # True -> close ACROSS the long axis (banana/pen)
    is_cube: bool = False           # True -> orientation-aware face-pair grasp
    local_long_axis: tuple | None = None   # PCA principal axis in the rigid-body local frame
    friction: tuple = (1.0, 0.9)
    color: tuple = (0.85, 0.15, 0.15)       # visual (procedural cuboid/sphere)
    target_palette: tuple | None = None     # REALISTIC per-object colour palette for the GRASP TARGET render.
    #                                     Each entry is an (r,g,b) base hue the target picks from (small jitter
    #                                     added by the task); a one-colour palette (e.g. tennis ball yellow-green)
    #                                     gets near-zero randomization. ``None`` (the cube) -> the task's FREE
    #                                     distinct-from-table random colour (keeps the cube collection byte-for-
    #                                     byte). This FIXES the bug where the cube's free random colour was applied
    #                                     to EVERY target (a banana rendered PINK/blue); a banana now renders
    #                                     yellow (mostly) or green (unripe), an apple red or green, a pen a normal
    #                                     pen colour. The palette lives on the spec so the colour is per-object.
    grasp_dz: float = 0.0           # nudge the grasp point along world +Z from the geometry centre
    grasp_noslip: int = 5           # contact friction-cone tightening iters (firm_rigid_options noslip) the firm
    #                                 grasp of THIS object needs. A ROUND/curved/THIN body reduces to a single
    #                                 tangent contact per finger and a leaky cone ejects/slips it, so it needs the
    #                                 tight cone (5). The flat-faced CUBE holds WITHOUT it AND a tight cone deepens
    #                                 its pinch over the abnormal-penetration gate, so the cube uses 0 (its locked
    #                                 ~2.5mm). Per-object so each object's firm contact is correct (collision #1).
    max_grasp_tilt_deg: float = 40.0    # CAP on the prefer-top-down relax ladder (select_grasp_tilt). A forward
    #                                     tilt eases the wrist at the lift, but it also makes the claws approach a
    #                                     ROUND body OFF-AXIS and shove it out (verified: apple/tennis EJECT 40cm
    #                                     at tilt>=8, hold top-down). So a ROUND object caps this at 0 (stay pure
    #                                     top-down; the deep equator seat keeps the wrist OK without a tilt). The
    #                                     cube/elongated keep the full 40deg ladder. Per-object grasp tuning -- the
    #                                     select_grasp_tilt SCORING is unchanged; this only restricts its ladder.
    grasp_close: float = 0.9            # driven-claw firm-pinch TARGET (GR100_CLOSE). A ROUNDED/soft body uses a
    #                                     GENTLER target (e.g. ~0.7) so the high-kp PD doesn't over-drive the claw
    #                                     into it (the >7mm penetration gate). Keep >= ~0.6 so the empty-close
    #                                     detector (claw near GR100_MEET=0.58 on a miss) stays valid. 0.9 = cube.
    grasp_single_hull: bool = False     # GRASP collider = a SINGLE convex hull (smooth envelope) instead of a
    #                                     coacd decomposition. For a ROUNDED CONVEX body (banana) the smooth hull
    #                                     pinches cleanly (stable, shallow penetration); a decomposition's internal
    #                                     seams + the firm pinch over-penetrate -> NaN. Use ONLY for near-convex
    #                                     bodies (a hollow/handled object MUST stay decomposed to keep its cavity).
    grasp_decompose_err: float = 0.04   # coacd convex-decomposition error threshold for the GRASP collider (USD
    #                                     mesh). Lower = more/tighter hulls hugging a rounded body -> the firm
    #                                     pinch claw can't sink as deep (less grasp penetration). 0.04 = bowl recipe.
    grasp_center_offset_local: tuple = (0.0, 0.0, 0.0)  # LOCAL-frame XY(Z) shift of the grasp point off the AABB
    #                                 centre onto the actual BODY -- needed for a CURVED object (a banana's AABB
    #                                 centre sits in the hollow of the curve, ~3cm off the fruit); the task
    #                                 rotates this by the object's world yaw and adds it to the grasp centre.
    #                                 (0,0,0) = grasp at the AABB centre (cube/pen/round -- their body IS centred).
    release_dz: float = 0.05        # height above the bowl centre to open/release (thin pen -> lower)
    contact_offset: float = 0.008   # speculative-contact band
    rest_offset: float = 0.0        # hard contact standoff (thin pen -> 0.004 to stop claw ON the surface)
    x_range: tuple = (0.34, 0.42)   # DR reachable x range (y mirrored per arm)
    place_xy_tol_cm: float = 8.0    # physical-success XY tolerance from the bowl centre

    # Trackable FEATURE frames in the rigid-body LOCAL frame, populated by the object-refiner agent
    # (registry/refine.py). Each entry: name -> {"center": (x,y,z), "normal": (nx,ny,nz), ["radius": r]} — the
    # frames a skill like the virtual-EE controls (e.g. a mug handle-ring center+normal -> thread onto a branch;
    # a cup opening; a cap axis; a peg tip). Empty for objects without a labelled feature.
    keypoints: dict = field(default_factory=dict)

    # --- derived geometry helpers (no I/O) ---
    def scaled_extents(self) -> np.ndarray:
        return np.asarray(self.extents, float) * self.scale

    def scaled_center(self) -> np.ndarray:
        return np.asarray(self.local_center, float) * self.scale

    def long_axis_local(self) -> np.ndarray | None:
        """Unit reference axis in the local frame, or None for round objects. Cube -> a FACE normal
        (local +X); elongated -> the PCA principal axis. The grasp closes PERPENDICULAR to this."""
        if self.is_cube:
            return np.array([1.0, 0.0, 0.0])
        if not self.elongated or self.local_long_axis is None:
            return None
        v = np.asarray(self.local_long_axis, float)
        return v / max(1e-9, np.linalg.norm(v))

    def rest_root_z(self, table_top_z: float, gap: float = 0.006) -> float:
        """Root Z so the object's lowest geometry point sits ``gap`` above the table top (then settles)."""
        c = self.scaled_center(); e = self.scaled_extents()
        lowest_local_z = c[2] - e[2] / 2.0
        return float(table_top_z - lowest_local_z + gap)


# ============================================================================ #
# THE REGISTRY  (geometry MEASURED in RoboLab, 2026-06-17; tennis_ball added 2026-06-18 for the Genesis gate;
#                book added + Nyx-safe clean .obj meshes (apple/banana/pen) added 2026-06-19 for distractors)
# ============================================================================ #
REGISTRY: dict[str, ObjectSpec] = {
    # apple (round, ~7.3cm). Grasps at its CENTRE with the cube's top-down(+relax-tilt) path -- no special
    # depth/tilt needed once the friction cone is tight (noslip_iterations in firm_rigid_options; before that a
    # firm pinch ejected the curved body, the 2026-06-20 round-object bug).
    # COLOUR POLICY = NATIVE TEXTURE (DR-strategist-owned, see dr_workbook colour-policy table): the apple has its
    # OWN photoreal skin -- apple_clean.obj is fully UV-mapped (898 vt, all faces) to objaverse/textures/apple_02.png
    # (a 1024x1024 real apple texture, natural red mottling + stem). The collector renders it with that texture
    # (gs.textures.ImageTexture, the table-top idiom -- verified in Nyx, no segfault) instead of a flat colour, so
    # the apple reads as a REAL apple, not a flat pink blob. NO per-frame colour DR (the texture IS the colour).
    # target_palette is kept as a FALLBACK only (used if native_texture is unset/forced off via NATIVE_TEX=0).
    # grasp_dz=-0.006 seats the body a touch deeper in the curved GR100 claws (more stable cage; still 0 abnormal).
    "apple": ObjectSpec(
        name="apple", language_name="apple", source="usd", usd_subpath="objaverse/apple_02.usd",
        mesh_subpath="objaverse/apple_clean.obj", dist_color=(0.80, 0.12, 0.10),
        native_texture="objaverse/textures/apple_02.png", grasp_dz=-0.006,
        mass=0.050, extents=(0.0702, 0.0754, 0.0733), local_center=(0.0, 0.0, 0.0),
        elongated=False, x_range=(0.34, 0.44), place_xy_tol_cm=7.0,
        target_palette=((0.62, 0.06, 0.05), (0.74, 0.10, 0.07), (0.40, 0.58, 0.14))),  # FALLBACK: deep red x2 / green
    # banana: CURVED. The AABB centre sits in the HOLLOW of the curve (~3cm off the fruit), so a grasp at the
    # AABB centre closes on AIR. grasp_center_offset_local shifts the grasp point along the SHORT (closing) axis
    # onto the banana body (short-proj +0.030 m = the body's centre at the long-axis midpoint, MEASURED from the
    # clean mesh: at |long|<2cm the body spans short-proj 0.011..0.052). The gripper then closes across the
    # banana's real ~3.7cm thickness. grasp_dz lifted a touch so the claws bracket the body, not skim the table.
    # NATIVE TEXTURE (DR-strategist, RECOGNIZABILITY RULE): banana_tex.obj is the SAME geometry as banana_clean.obj
    # (byte-identical V/F, identical volume + convex hull) but UV-mapped (10710 vt, all faces) -> renders the REAL
    # BOP YCB-V banana scan (ycb/textures/obj_000010.png: yellow body, green stem, brown speckle/tips) via
    # gs.textures.ImageTexture in _target_usd_surface, so the banana reads as a REAL banana, not a flat yellow stick.
    # Swapping mesh_subpath does NOT disturb the verified banana grasp (same single-hull collider geometry).
    # target_palette kept as the NATIVE_TEX=0 fallback; native-texture -> NO per-frame colour DR.
    "banana": ObjectSpec(
        name="banana", language_name="banana", source="usd", usd_subpath="ycb/banana.usd",
        mesh_subpath="ycb/banana_tex.obj", dist_color=(0.92, 0.80, 0.15),
        native_texture="ycb/textures/obj_000010.png",
        mass=0.080, extents=(0.1089, 0.1784, 0.0367), local_center=(0.0, 0.0, 0.0),
        elongated=True, local_long_axis=(0.3372, 0.9414, 0.0), grasp_dz=-0.006,
        grasp_center_offset_local=(-0.0282, 0.0101, 0.0), grasp_single_hull=True, x_range=(0.34, 0.44),
        place_xy_tol_cm=9.0,
        target_palette=((0.92, 0.80, 0.15), (0.95, 0.85, 0.20), (0.62, 0.70, 0.18))),  # yellow x2 / green (unripe)
    # marker pen (dry-erase marker): very elongated, ~2cm thick. rest_offset=0.004 stops the firm claw ON the
    # surface (else it over-drives PAST the thin body and the pen lodges on a finger). release_dz lower so it
    # settles in the bowl instead of rolling off the rim.
    # marker pen (dry-erase marker): very elongated, ~2cm thick. grasp_single_hull=True (a SMOOTH convex
    # envelope of the thin body) so the firm claw pinches a FLAT face instead of sinking into a decomposition
    # seam -> shallower, more uniform contact (verified: the decomp collider over-penetrated 2/12 envs at >7.5mm;
    # the hull drops it to 1/12 and 12/12 physical placed). The thin pen still needs the tight friction cone
    # (noslip) to hold at all -- WITHOUT it the firm pinch slips and 0/12 grasp. release_dz lower so it settles in
    # the bowl instead of rolling off the rim.
    # DEEPER-GRASP TUNE (2026-06-20, owner #6): the firm pad-near-pad clamp on the THIN ~2cm pen body over-bit
    # it (verified E=24 stock seed7: 4/24 abnormal @ up to 8.1mm). A SMALL deeper seat (grasp_dz -0.004, cradle a
    # touch lower in the curved claws) COMBINED with a GENTLER firm-close TARGET (grasp_close 0.9 -> 0.80) seats
    # the pen more stably AND stops the high-kp PD from driving the pad through the thin body: E=24 seed7 ->
    # 24/24 placed, 0/24 abnormal, max 6.7mm (under the 7mm gate). The deeper seat ALONE (at the full 0.9 close)
    # made penetration WORSE (drives further in); the gentler close is what lets the slight deeper seat cage
    # without over-penetrating. grasp_close stays >= GR100_MEET(0.58) so the empty-close miss detector is valid.
    # grasp_close=0.78 is the sweet spot (E=24 seed7: 0/24 abnormal, max 6.5mm; 24/24 grasped); LOWER (<=0.76)
    # is non-monotonically WORSE (the gentle target lets the body shift into a deeper bite). The pen is the
    # framework's hardest penetration case (a thin body the firm pad-near-pad clamp wants to over-bite), so it
    # rides near the 7mm gate -- at E=12 the odd far-reach env can still nick ~7.3mm (1/12). Lower-risk than a
    # deeper seat at the full close, which over-bit it (4/24 abnormal).
    # NATIVE TEXTURE (DR-strategist, RECOGNIZABILITY RULE): dry_erase_marker_tex.obj is geometry-equal to the
    # _clean.obj (identical volume/extents/convex hull; 14043 vt, all faces UV-mapped) -> renders the REAL BOP
    # YCB-V large-marker scan (ycb/textures/obj_000018.png: white EXPO barrel + printed label band + black chisel
    # cap/tip) via gs.textures.ImageTexture, so the pen reads as a REAL EXPO marker, not a flat colour stick. The
    # geometry match preserves the verified (penetration-critical) pen grasp tuning. target_palette = the
    # NATIVE_TEX=0 fallback; native-texture -> NO per-frame colour DR.
    "pen": ObjectSpec(
        name="pen", language_name="pen", source="usd", usd_subpath="ycb/dry_erase_marker.usd",
        mesh_subpath="ycb/dry_erase_marker_tex.obj", dist_color=(0.10, 0.10, 0.12),
        native_texture="ycb/textures/obj_000018.png",
        mass=0.020, extents=(0.0210, 0.1208, 0.0189), local_center=(0.0, 0.0, 0.0),
        elongated=True, local_long_axis=(-0.0303, 0.9995, 0.0), grasp_dz=-0.004, grasp_single_hull=True,
        grasp_close=0.78, rest_offset=0.004, release_dz=0.03, x_range=(0.34, 0.44), place_xy_tol_cm=9.0,
        target_palette=((0.10, 0.10, 0.12), (0.12, 0.20, 0.55), (0.55, 0.12, 0.14))),  # black / blue / red marker
    "cube": ObjectSpec(
        name="cube", language_name="cube", source="cuboid", mass=0.040,
        extents=(0.05, 0.05, 0.05), local_center=(0.0, 0.0, 0.0), is_cube=True,
        color=(0.85, 0.15, 0.15), grasp_noslip=0, x_range=(0.34, 0.44)),   # flat faces hold w/o noslip; tight cone over-penetrates
    # tennis ball: regulation ~6.7cm diameter, ~57g, round (no preferred axis) -> procedural sphere; yellow-
    # green felt. Round handling like the apple (grasp at root, place tol 7cm). NEW for the Genesis 5-object gate.
    # tennis ball (round, ~6.7cm, pure sphere collider). Grasps at its CENTRE with the cube's top-down path once
    # the friction cone is tight (noslip_iterations) -- same round-object fix as the apple.
    # NATIVE TEXTURE (DR-strategist, RECOGNIZABILITY RULE): a procedural sphere can ONLY be a flat-colour ball
    # ("yellow sphere" = forbidden). Switched source="sphere" -> "usd" with a UV-mapped sphere mesh
    # (generated/tennis_ball_tex.obj: R=0.0335m, extents EXACTLY (0.067,0.067,0.067), 4753 vt, all faces) so it
    # renders the generated regulation optic-yellow-green FELT + the classic curved white SEAM texture
    # (generated/textures/tennis_ball.png) via gs.textures.ImageTexture -> reads as a REAL tennis ball. COLLIDER:
    # grasp_single_hull=True -> the convexified UV sphere = a clean convex SPHERE (faithful round collider, the
    # same round-object grasp recipe as before: no preferred axis, tight friction cone noslip=5). dist_color added
    # for the Nyx-safe distractor render (yellow-green felt). target_palette = NATIVE_TEX=0 fallback.
    "tennis_ball": ObjectSpec(
        name="tennis_ball", language_name="tennis ball", source="usd", mass=0.057,
        mesh_subpath="generated/tennis_ball_tex.obj", dist_color=(0.82, 0.92, 0.22),
        native_texture="generated/textures/tennis_ball.png", grasp_single_hull=True,
        extents=(0.067, 0.067, 0.067), local_center=(0.0, 0.0, 0.0), elongated=False,
        color=(0.85, 0.95, 0.20), friction=(1.1, 1.0), x_range=(0.34, 0.44), place_xy_tol_cm=7.0,
        target_palette=((0.82, 0.92, 0.22),)),                  # one colour: regulation yellow-green felt (no DR)
    # book: a flat hardcover (~18x13x3 cm, ~0.30 kg). Procedural box with a realistic dark-red cover colour;
    # high friction so it rests flat and is hard to nudge. Used as a clutter/distractor (not a grasp target
    # in pick-place), so no elongated/cube grasp hints are needed. NEW 2026-06-19 for the distractor pool.
    # NATIVE TEXTURE (DR-strategist, RECOGNIZABILITY RULE): a procedural Box can ONLY be a flat brick (a Box has no
    # UVs -> Nyx can't texture it; "pink brick" = forbidden). Switched source="cuboid" -> "usd" with a UV-mapped
    # box mesh (generated/book_tex.obj: extents EXACTLY (0.18,0.13,0.03), 24 vt per-face, all faces) so it renders
    # the generated hardcover atlas (generated/textures/book.png: teal cloth cover + gold-framed title plate +
    # darker spine + cream page-edges with striations on the open edges) via gs.textures.ImageTexture -> reads as a
    # REAL book, not a flat slab. COLLIDER: grasp_single_hull=True -> the convexified box = the same solid slab
    # (faithful box collider). Book is a distractor-only object (never a grasp target) so there's no grasp tuning
    # to preserve. dist_color added for the Nyx-safe distractor render; target_palette = NATIVE_TEX=0 fallback.
    "book": ObjectSpec(
        name="book", language_name="book", source="usd", mass=0.300,
        mesh_subpath="generated/book_tex.obj", dist_color=(0.10, 0.42, 0.45),
        native_texture="generated/textures/book.png", grasp_single_hull=True,
        extents=(0.18, 0.13, 0.03), local_center=(0.0, 0.0, 0.0), elongated=False,
        color=(0.45, 0.10, 0.12), friction=(1.2, 1.0), x_range=(0.34, 0.44), place_xy_tol_cm=8.0,
        target_palette=((0.10, 0.42, 0.45),)),                  # teal hardcover (NATIVE_TEX=0 fallback)
    # mug: a HOLLOW ceramic mug (YCB), REFINED + sim-ready-VERIFIED by the object-refiner harness
    # (registry/refine.py + demo_mug.py, 2026-06-19). The handle ring + cup mouth STAY HOLLOW via convex
    # DECOMPOSITION (34 hulls) — a branch threads the ring with 0.00mm overlap (verified). Mass from
    # solid_volume(237cm^3) x ceramic density(2400) ~= 0.57kg (a thick ceramic mug; on the heavy side because
    # the voxel-fill counts the whole wall+interior volume). Nyx renders the clean mug_clean.obj (USD segfaults
    # Nyx). long_axis is the PCA axis (handle-out X tilted by the cup body) — kept for the record; for a future
    # mug-hang the CONTROLLED frame is the handle_ring keypoint, not this grasp axis. Geometry MEASURED.
    "mug": ObjectSpec(
        name="mug", language_name="mug", source="usd", usd_subpath="ycb/mug.usd",
        mesh_subpath="ycb/mug_clean.obj", dist_color=(0.85, 0.85, 0.88),
        mass=0.569, extents=(0.1170, 0.0931, 0.0814),
        local_center=(0.0, 0.0, 0.0), friction=(1.0, 0.9),
        elongated=True, local_long_axis=(-0.4030, 0.0219, -0.9149),
        color=(0.90, 0.90, 0.93), grasp_dz=0.0, x_range=(0.34, 0.44), place_xy_tol_cm=8.0,
        # keypoints DETECTED by refine.detect_keypoints + sim-VERIFIED (object-refiner v2, 2026-06-20): the
        # handle_ring via an encircled-through-hole search (in-sim hollow probe 0.00mm overlap = open ring) and
        # the cup_opening via a topmost-rim + cavity-depth test (76mm cavity below the rim = a real mouth). Both
        # agree with the earlier hand-measured values to <1.5mm. The task reads the LIVE world frame via
        # refine.world_keypoint(local_kp, mug.get_pos(), mug.get_quat()) — a massless, physics-inert annotation.
        keypoints={
            "handle_ring": dict(center=(0.0409, -0.0016, -0.0001), normal=(0.0, 1.0, 0.0), radius=0.0091),
            "cup_opening": dict(center=(-0.0111, 0.0006, 0.0394), normal=(0.0, 0.0, 1.0), radius=0.0437),
        }),
}
