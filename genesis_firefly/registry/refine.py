# SPDX-License-Identifier: Apache-2.0
"""OBJECT-REFINER harness — turn a RAW object (mesh / USD) into a SIM-READY asset with a CORRECT collision
model, then VERIFY sim-readiness with our existing penetration gate + an initialization-stability test.

WHY THIS EXISTS (owner directive, priority #1; docs/project_overview.md §0b + docs/object_refiner.md):
  A correct collision model + zero abnormal interpenetration is the foundation everything stands on. Every NEW
  object must get a verified good collision model BEFORE it is used: a SOLID never penetrates, and a HOLLOW
  feature (a mug-handle ring, a cup mouth, a bottle neck) STAYS HOLLOW so a ring can thread onto a branch and a
  cap can seat in an opening. This module is the reusable harness that produces + verifies that asset.

This is the code mapping of the owner's ASSET2SIM paper onto what we already built:
  - AUGMENT   = trimesh mesh analysis (is_watertight / volume / area / bbox extents / center-of-mass /
                suggested scale to a target real size). The paper's "augment" stage (multi-view + mesh analysis).
  - INFER     = realistic physics: mass = solid_volume x a per-category density, plus friction / restitution
                (sensible category defaults; the agent may reason per category). The paper's "infer physics".
  - COLLISION = a GOOD collider via convex-DECOMPOSITION (coacd) so hollow features stay hollow (NEVER a single
                filling hull), plus a Nyx-safe clean visual OBJ (Nyx segfaults on the textured USD material
                bind -> extract the geometry). The paper's "set a good collision model".
  - VERIFY    = sim-readiness gates, reusing OUR tools:
                  (i)  PENETRATION  -- skills.penetration (the #1 collision gate): abnormal ~= 0 at rest.
                  (ii) STABILITY    -- the paper's initialization-stability test: load + settle with ZERO robot
                       actions for ~1 s, then measure root-pose drift D_pos / D_ori over a test window; PASS if
                       D_pos <= POS_TOL_M, D_ori <= ORI_TOL_RAD, and the object did not explode.
                  (iii)HOLLOW PROOF  -- pass a thin probe cylinder THROUGH a declared hollow feature; if it does
                       NOT report deep penetration, the feature is genuinely open (a branch could thread it).

The harness is PURE-ish: ``analyze_mesh`` + ``infer_physics`` have no engine dependency (trimesh + numpy only);
the collision extraction + the verify stages drive Genesis. Everything returns a structured ``RefineReport`` and
can emit/update an ``ObjectSpec`` line for ``registry/object_spec.py``.

Run the demo (refines + verifies the YCB mug, writes a verification render):
  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/registry/refine.py
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import trimesh

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                                   # genesis_firefly/
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

# ============================================================================ #
# CATEGORY PHYSICS PRIORS (the "infer physics" knowledge — what Claude reasons about per category).
#   density kg/m^3  -- of the SOLID material the object is made of (the inferred mass = solid_volume * density).
#   friction        -- (object_mu, finger_mu-ish) MAX-combined by Genesis; sensible grippable defaults.
#   restitution     -- bounciness 0..1 (Genesis rigid is largely inelastic; kept for the record + future use).
# These are deliberately coarse, realistic priors. A refiner agent picks the closest category (or overrides a
# field) the same way it would reason about a new object. Ceramic mug, wooden block, rubber ball, etc.
# ============================================================================ #
CATEGORY_PRIORS: dict[str, dict] = {
    "ceramic":   dict(density=2400.0, friction=(1.0, 0.9), restitution=0.05),   # mug / plate / bowl
    "glass":     dict(density=2500.0, friction=(0.9, 0.9), restitution=0.05),
    "wood":      dict(density=700.0,  friction=(1.1, 0.9), restitution=0.10),   # book / block
    "plastic":   dict(density=950.0,  friction=(0.9, 0.9), restitution=0.15),   # marker / cup
    "metal":     dict(density=7800.0, friction=(0.8, 0.9), restitution=0.10),
    "rubber":    dict(density=1100.0, friction=(1.2, 1.0), restitution=0.45),   # tennis-ball-ish felt/rubber
    "fruit":     dict(density=950.0,  friction=(1.0, 0.9), restitution=0.10),   # apple/banana (water-ish)
    "food":      dict(density=950.0,  friction=(1.0, 0.9), restitution=0.10),
    "generic":   dict(density=900.0,  friction=(1.0, 0.9), restitution=0.10),
}

# ---- sim-readiness gate tolerances (the paper's init-stability test, our values) ---------------------------
# The paper uses D_pos <= 1 mm / D_ori <= 0.01 rad for a perfectly-settled object. We loosen slightly to the
# firm-solver reality (coacd raises the rest floor a few mm; the object micro-settles on its convex hulls) so a
# genuinely-stable asset passes without being flagged for normal settling. Still tight enough to catch drift /
# rolling / explosion. Override per call if a task needs the paper-strict bound.
POS_TOL_M: float = 0.003        # max root XY/Z drift over the test window (3 mm)
ORI_TOL_RAD: float = 0.02       # max root orientation change over the test window (~1.15 deg)
EXPLODE_M: float = 0.20         # any root move > 20 cm in the window == exploded / launched (hard fail)


# ============================================================================ #
# REPORT
# ============================================================================ #
@dataclass
class MeshStats:
    """The AUGMENT result — pure trimesh geometry analysis (no engine)."""
    n_vertices: int
    n_faces: int
    is_watertight: bool
    extents_m: tuple                 # AABB size (ex, ey, ez) at the analyzed scale
    bbox_lo: tuple
    bbox_hi: tuple
    center_of_mass: tuple            # solid CoM (watertight) or centroid (open mesh)
    surface_area_m2: float
    solid_volume_m3: float           # voxel-filled solid volume (robust to open meshes); the mass basis
    hull_volume_m3: float            # convex-hull volume (reference; >= solid for a hollow object)
    bbox_volume_m3: float
    hollowness: float                # 1 - solid/hull : how "hollow" the object reads (0 solid .. ->1 thin shell)
    long_axis_local: tuple           # PCA principal axis (unit) in the local frame
    suggested_scale: Optional[float] = None   # scale to hit a target real size, if a target was given


@dataclass
class PhysicsInfer:
    """The INFER result — realistic physics from the category prior + the measured volume."""
    category: str
    density_kg_m3: float
    mass_kg: float                   # = solid_volume_m3 * density
    friction: tuple
    restitution: float


@dataclass
class CollisionModel:
    """The COLLISION result — the chosen good collider + the Nyx-safe visual."""
    method: str                      # "convex_decomposition" (coacd) | "convex_hull" | "primitive"
    decompose_error_threshold: float
    n_convex_hulls: int              # how many solid hulls coacd produced (hollow features stay open between them)
    visual_obj: Optional[str]        # path to the extracted Nyx-safe clean OBJ (relative to assets/objects)
    nyx_safe: bool                   # True once a clean OBJ visual exists (textured-USD segfault avoided)


@dataclass
class Verdict:
    """One sim-readiness gate's PASS/FAIL + its measured value."""
    name: str
    passed: bool
    detail: str


@dataclass
class RefineReport:
    """The structured output of refine() — mesh stats + chosen physics + collision method + the verdicts."""
    name: str
    source_path: str
    source_kind: str                 # "objaverse" | "ycb" | "partnet_mobility" | "procedural" | ...
    mesh: MeshStats
    physics: PhysicsInfer
    collision: CollisionModel
    verdicts: list = field(default_factory=list)   # list[Verdict]
    keypoints: dict = field(default_factory=dict)  # name -> {center, normal, (radius)} in the LOCAL frame
    sim_ready: bool = False                        # AND of the hard gates (penetration + stability [+ hollow])
    # --- v2 additions (object-refiner v2, 2026-06-20) ---------------------------------------------------------
    collider_views: dict = field(default_factory=dict)   # view_name -> abspath of the saved COLLIDER render PNG
    #   the final visual collision check: the convex-decomposition hulls rendered from canonical viewpoints so the
    #   agent (and owner) confirms hollow-stays-hollow + solid-never-fills. Saved NEXT TO the asset.
    detected_keypoints: dict = field(default_factory=dict)  # name -> {center, normal, [radius], method, verified}
    #   keypoints DETECTED from geometry by detect_keypoints() (vs the hand-supplied `keypoints`); `verified`/
    #   `method` record HOW each was confirmed so an unverified label is never silently baked.
    semantic_links: dict = field(default_factory=dict)   # raw link/joint name -> {semantic, role, evidence,
    #   confidence} for an articulated URDF — the PartNet semantic naming (NEVER renames the URDF; an annotation).

    def add(self, name, passed, detail):
        self.verdicts.append(Verdict(name, bool(passed), str(detail)))
        return passed

    def verdict(self, name):
        for v in self.verdicts:
            if v.name == name:
                return v
        return None

    def to_dict(self):
        d = asdict(self)
        return d

    def pretty(self) -> str:
        m, p, c = self.mesh, self.physics, self.collision
        L = []
        L.append("=" * 86)
        L.append(f"REFINE REPORT  '{self.name}'   source={self.source_kind}: {self.source_path}")
        L.append("=" * 86)
        L.append("AUGMENT (trimesh mesh analysis):")
        L.append(f"   verts={m.n_vertices}  faces={m.n_faces}  watertight={m.is_watertight}")
        L.append(f"   extents(m)= {tuple(round(x,4) for x in m.extents_m)}   "
                 f"CoM= {tuple(round(x,4) for x in m.center_of_mass)}")
        L.append(f"   surface_area= {m.surface_area_m2:.5f} m^2   solid_volume= {m.solid_volume_m3*1e6:.1f} cm^3  "
                 f"(hull {m.hull_volume_m3*1e6:.1f} cm^3, bbox {m.bbox_volume_m3*1e6:.1f} cm^3)")
        L.append(f"   hollowness(1-solid/hull)= {m.hollowness:.2f}   "
                 f"long_axis= {tuple(round(x,3) for x in m.long_axis_local)}"
                 + (f"   suggested_scale= {m.suggested_scale:.3f}" if m.suggested_scale else ""))
        L.append("-" * 86)
        L.append("INFER (physics from category prior x measured volume):")
        L.append(f"   category= {p.category}   density= {p.density_kg_m3:.0f} kg/m^3   "
                 f"mass= {p.mass_kg*1000:.1f} g   friction= {p.friction}   restitution= {p.restitution}")
        L.append("-" * 86)
        L.append("COLLISION (good collider = convex DECOMPOSITION; hollow stays hollow):")
        L.append(f"   method= {c.method}   coacd_error_threshold= {c.decompose_error_threshold}   "
                 f"n_convex_hulls= {c.n_convex_hulls}")
        L.append(f"   visual_obj= {c.visual_obj}   nyx_safe= {c.nyx_safe}")
        if self.keypoints:
            L.append("-" * 86)
            L.append("KEYPOINTS (trackable feature frames, LOCAL):")
            for kn, kv in self.keypoints.items():
                cen = tuple(round(x, 4) for x in kv["center"])
                nor = tuple(round(x, 3) for x in kv["normal"])
                extra = f"  radius={kv['radius']:.4f}" if "radius" in kv else ""
                L.append(f"   {kn:<14} center={cen}  normal={nor}{extra}")
        if self.detected_keypoints:
            L.append("-" * 86)
            L.append("DETECTED KEYPOINTS (from geometry; verified flag = never bake an unverified label):")
            for kn, kv in self.detected_keypoints.items():
                cen = tuple(round(x, 4) for x in kv["center"])
                nor = tuple(round(x, 3) for x in kv["normal"])
                extra = f"  radius={kv['radius']:.4f}" if "radius" in kv else ""
                tag = "VERIFIED" if kv.get("verified") else "UNVERIFIED"
                L.append(f"   [{tag}] {kn:<14} center={cen}  normal={nor}{extra}  via={kv.get('method','?')}")
        if self.semantic_links:
            L.append("-" * 86)
            L.append("SEMANTIC LINK NAMES (PartNet raw index -> semantic; annotation, URDF NOT renamed):")
            for raw, sv in self.semantic_links.items():
                tag = "VERIFIED" if sv.get("verified") else "uncertain"
                L.append(f"   [{tag}] {raw:<14} -> {sv.get('semantic'):<14} role={sv.get('role')}  "
                         f"conf={sv.get('confidence', 0.0):.2f}")
                L.append(f"                   evidence: {sv.get('evidence', '')}")
        if self.collider_views:
            L.append("-" * 86)
            L.append("COLLIDER MULTI-VIEW RENDERS (the FINAL visual collision check — open hulls = hollow kept):")
            for vn, vp in self.collider_views.items():
                L.append(f"   {vn:<10} {vp}")
        L.append("-" * 86)
        L.append("VERIFY (sim-readiness gates — reusing skills/penetration.py + the init-stability test):")
        for v in self.verdicts:
            L.append(f"   [{'PASS' if v.passed else 'FAIL'}] {v.name:<22} {v.detail}")
        L.append("-" * 86)
        L.append(f"SIM-READY: {'YES' if self.sim_ready else 'NO'}")
        L.append("=" * 86)
        return "\n".join(L)


# ============================================================================ #
# (a) AUGMENT — trimesh mesh analysis (NO engine; pure geometry)
# ============================================================================ #
def _load_trimesh(path: str) -> trimesh.Trimesh:
    """Load a mesh/scene path into a single Trimesh (concatenate a Scene). USD is NOT loaded here (trimesh has
    no USD reader) — for USD, extract the geometry via Genesis first (see extract_visual_obj) and pass the OBJ."""
    m = trimesh.load(path, process=False, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def _solid_volume(tm: trimesh.Trimesh, pitch: float = 0.003) -> float:
    """A robust SOLID volume estimate that works for OPEN meshes (a mug is not watertight): voxelize at `pitch`
    then flood-FILL the interior. (trimesh.volume is only valid for watertight meshes and would be garbage on a
    mug.) Falls back to the watertight volume if available, else the convex-hull volume."""
    try:
        vox = tm.voxelized(pitch=pitch)
        filled = vox.fill()
        v = float(filled.volume)
        if v > 0:
            return v
    except Exception:
        pass
    if tm.is_watertight and tm.volume > 0:
        return float(tm.volume)
    return float(tm.convex_hull.volume)


def _pca_long_axis(tm: trimesh.Trimesh) -> np.ndarray:
    """The PCA principal (longest-variance) axis of the vertices, unit, in the local frame."""
    V = np.asarray(tm.vertices, float)
    Vc = V - V.mean(0)
    cov = Vc.T @ Vc
    w, vecs = np.linalg.eigh(cov)
    a = vecs[:, int(np.argmax(w))]
    return a / max(1e-9, np.linalg.norm(a))


def analyze_mesh(path: str, target_size_m: Optional[float] = None,
                 target_axis: Optional[int] = None) -> MeshStats:
    """AUGMENT: full trimesh geometry analysis of a mesh path. Returns a ``MeshStats``.

    target_size_m / target_axis : if given, suggest a uniform scale so the chosen AABB axis (default the LARGEST
        extent) hits ``target_size_m`` real metres — the paper's "suggested scale to a target real size".
    """
    tm = _load_trimesh(path)
    bb = tm.bounds
    ext = (bb[1] - bb[0]).astype(float)
    com = (np.asarray(tm.center_mass, float) if tm.is_watertight else np.asarray(tm.centroid, float))
    solid_v = _solid_volume(tm)
    hull_v = float(tm.convex_hull.volume)
    bbox_v = float(np.prod(ext))
    hollowness = float(np.clip(1.0 - solid_v / max(hull_v, 1e-12), 0.0, 1.0))

    suggested = None
    if target_size_m is not None:
        ax = int(np.argmax(ext)) if target_axis is None else int(target_axis)
        cur = float(ext[ax])
        if cur > 0:
            suggested = float(target_size_m / cur)

    return MeshStats(
        n_vertices=int(len(tm.vertices)), n_faces=int(len(tm.faces)),
        is_watertight=bool(tm.is_watertight),
        extents_m=tuple(round(float(x), 6) for x in ext),
        bbox_lo=tuple(round(float(x), 6) for x in bb[0]),
        bbox_hi=tuple(round(float(x), 6) for x in bb[1]),
        center_of_mass=tuple(round(float(x), 6) for x in com),
        surface_area_m2=float(tm.area), solid_volume_m3=solid_v, hull_volume_m3=hull_v,
        bbox_volume_m3=bbox_v, hollowness=hollowness,
        long_axis_local=tuple(round(float(x), 6) for x in _pca_long_axis(tm)),
        suggested_scale=suggested,
    )


# ============================================================================ #
# (b) INFER — realistic physics from the category prior x the measured volume
# ============================================================================ #
def infer_physics(mesh: MeshStats, category: str = "generic", *, density: Optional[float] = None,
                  friction: Optional[tuple] = None, restitution: Optional[float] = None,
                  scale: float = 1.0) -> PhysicsInfer:
    """INFER: pick a per-category density (overridable) -> realistic mass = solid_volume * density (at `scale`),
    plus grippable friction + restitution. The agent chooses the closest category (ceramic/wood/plastic/...)."""
    prior = CATEGORY_PRIORS.get(category, CATEGORY_PRIORS["generic"])
    rho = float(density if density is not None else prior["density"])
    fr = tuple(friction if friction is not None else prior["friction"])
    rest = float(restitution if restitution is not None else prior["restitution"])
    vol = mesh.solid_volume_m3 * (scale ** 3)
    mass = float(rho * vol)
    return PhysicsInfer(category=category, density_kg_m3=rho, mass_kg=mass, friction=fr, restitution=rest)


# ============================================================================ #
# (c) COLLISION — a GOOD collider (convex DECOMPOSITION) + a Nyx-safe visual OBJ
# ============================================================================ #
def extract_visual_obj(src_usd_or_mesh: str, out_obj_abspath: str, scale: float = 1.0,
                       decompose_error_threshold: float = 0.04) -> CollisionModel:
    """COLLISION: load the source in Genesis with the GOOD collider (convex DECOMPOSITION via coacd, the SAME
    recipe the bowl uses) and extract a clean Nyx-safe visual OBJ from the loaded geometry. Returns a
    ``CollisionModel`` recording the method, the coacd error threshold, the hull count, and the OBJ path.

    WHY decomposition (owner #1 / docs/lessons_genesis_nyx.md §3.2): a single convex hull would FILL a mug's
    handle ring + cup cavity (a solid blob) — a branch could never thread it. coacd splits the shell into many
    SOLID convex hulls that tile the wall, leaving the ring + cavity genuinely OPEN. A clean OBJ is also the
    Nyx-safe visual (Nyx segfaults on the textured USD material bind — §3.1).

    This MUST run in a fresh process (one gs.Scene per process; a 2nd build segfaults) — refine() spawns it.
    """
    import genesis as gs
    os.makedirs(os.path.dirname(out_obj_abspath), exist_ok=True)
    # one scene, one entity, the convex-DECOMPOSITION collider (coacd) — identical recipe to build_bowl().
    sc = gs.Scene(show_viewer=False)
    if src_usd_or_mesh.lower().endswith(".usd") or src_usd_or_mesh.lower().endswith(".usda"):
        morph = gs.morphs.USD(file=src_usd_or_mesh, scale=scale, convexify=True,
                              decompose_object_error_threshold=decompose_error_threshold, decimate=False)
    else:
        morph = gs.morphs.Mesh(file=src_usd_or_mesh, scale=scale, convexify=True,
                               decompose_object_error_threshold=decompose_error_threshold, decimate=False)
    e = sc.add_entity(morph)
    sc.build(n_envs=0)
    # the visual trimesh (same path bowl_clean.obj / apple_clean.obj used) -> a clean Nyx-safe OBJ.
    vm = e.vgeoms[0].vmesh
    tm = vm.trimesh if hasattr(vm, "trimesh") else vm
    tm.export(out_obj_abspath)
    # count the convex collision hulls the decomposition produced (hollow features sit in the gaps between them)
    n_hulls = 0
    try:
        for g in e.geoms:
            n_hulls += 1
    except Exception:
        n_hulls = -1
    rel = os.path.relpath(out_obj_abspath, os.path.join(_PKG, "assets", "objects"))
    return CollisionModel(method="convex_decomposition", decompose_error_threshold=decompose_error_threshold,
                          n_convex_hulls=int(n_hulls), visual_obj=rel, nyx_safe=True)


# ============================================================================ #
# (d) VERIFY — the sim-readiness gates (reuse skills/penetration.py + the init-stability test)
# ============================================================================ #
def verify_in_sim(visual_obj_abspath: str, physics: PhysicsInfer, *, scale: float = 1.0,
                  table_z: float = 0.25, settle_steps: int = 120, window_steps: int = 60,
                  hollow_probe: Optional[dict] = None, friction: float = 1.0,
                  decompose_error_threshold: float = 0.04) -> dict:
    """VERIFY: load the refined collider into a MINIMAL stage (a fixed table + the object, the SAME firm Newton
    solver the real stage uses) and run the sim-readiness gates. MUST run in a fresh process (refine() spawns).

    Gates:
      (i)  PENETRATION : skills.penetration.abnormal_penetration at rest -> abnormal count must be 0.
      (ii) STABILITY   : settle with ZERO actions for `settle_steps`, then over a `window_steps` window measure
                         the root-pose drift D_pos (max XYZ move) + D_ori (max angle change). PASS if
                         D_pos <= POS_TOL_M, D_ori <= ORI_TOL_RAD, and no explosion (> EXPLODE_M).
      (iii)HOLLOW      : (optional) drop a thin probe CYLINDER through a declared hollow feature (center+axis in
                         the object LOCAL frame) and confirm it does NOT deeply penetrate the object — proving a
                         branch could thread the ring / a peg the hole.

    Returns a dict of raw measurements; refine() folds these into Verdicts. Keeps engine imports local.
    """
    import genesis as gs
    from world.firefly_scene import firm_rigid_options
    from skills.penetration import max_penetration, abnormal_penetration, ABNORMAL_THRESH_M

    sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
                  rigid_options=firm_rigid_options(), show_viewer=False)
    # a single fixed collidable table (no robot needed for the init-stability test — zero actions).
    table = sc.add_entity(gs.morphs.Box(size=(0.7, 0.9, table_z), pos=(0.4, 0.0, table_z / 2.0),
                                        fixed=True, collision=True))
    # the refined object with the GOOD collider, resting just above the table top.
    obj = sc.add_entity(
        gs.morphs.Mesh(file=visual_obj_abspath, scale=scale, convexify=True,
                       decompose_object_error_threshold=decompose_error_threshold, decimate=False,
                       pos=(0.4, 0.0, table_z + 0.06)),
        material=gs.materials.Rigid(rho=max(physics.density_kg_m3, 50.0), friction=float(friction)))
    probe = None
    if hollow_probe is not None:
        # a thin solid cylinder placed to pass THROUGH the declared hollow feature (along its axis). If the
        # feature is genuinely open, the probe ends up inside the ring/cavity with NO deep solid-solid overlap.
        pr = float(hollow_probe.get("radius", 0.004))
        ln = float(hollow_probe.get("length", 0.20))
        probe = sc.add_entity(
            gs.morphs.Cylinder(radius=pr, height=ln, fixed=True),    # fixed: a stand-in for a static branch
            material=gs.materials.Rigid(rho=400.0, friction=0.5))
    sc.build(n_envs=0)

    def root_pose():
        p = obj.get_pos(); q = obj.get_quat()
        p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
        q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
        return p.reshape(-1)[:3].astype(float), q.reshape(-1)[:4].astype(float)

    def quat_angle(q0, q1):
        d = abs(float(np.dot(q0 / max(np.linalg.norm(q0), 1e-9), q1 / max(np.linalg.norm(q1), 1e-9))))
        return float(2.0 * math.acos(min(1.0, d)))

    # --- settle with ZERO actions (nothing is actuated; the object free-falls the 6cm + settles) ---
    for _ in range(settle_steps):
        sc.step()
    p0, q0 = root_pose()
    if not np.all(np.isfinite(p0)):
        return dict(exploded=True, d_pos_m=float("inf"), d_ori_rad=float("inf"),
                    rest_abnormal=1, rest_max_pen_mm=float("inf"), settled_pos=None)

    # --- the test window: track the worst drift from the settled pose ---
    d_pos = 0.0
    d_ori = 0.0
    exploded = False
    for _ in range(window_steps):
        sc.step()
        p, q = root_pose()
        if not np.all(np.isfinite(p)):
            exploded = True
            break
        d_pos = max(d_pos, float(np.linalg.norm(p - p0)))
        d_ori = max(d_ori, quat_angle(q0, q))
        if d_pos > EXPLODE_M:
            exploded = True
            break

    # --- penetration at rest (the #1 gate): abnormal solid-solid overlap on the just-stepped buffer ---
    flagged, depth = abnormal_penetration(sc, redetect=False)
    res = max_penetration(sc, redetect=False)
    rest_abnormal = int(np.count_nonzero(flagged))
    rest_max_pen_mm = float(res["depth_mm"].max())
    worst = res["worst_pair_names"][int(np.argmax(res["depth_m"]))] if rest_max_pen_mm > 0 else None

    out = dict(exploded=bool(exploded), d_pos_m=float(d_pos), d_ori_rad=float(d_ori),
               rest_abnormal=rest_abnormal, rest_max_pen_mm=rest_max_pen_mm, worst_pair=worst,
               settled_pos=tuple(float(x) for x in p0))

    # --- hollow proof: settle the object, place the probe through the feature, redetect TRUE overlap ---
    if probe is not None and not exploded:
        # the feature center+axis are in the object LOCAL frame; the object settled near identity orientation
        # (it free-fell straight down), so local ~= world apart from the root translation. Place the probe so its
        # axis passes through the (translated) feature center, oriented along the feature normal.
        c = np.asarray(hollow_probe["center"], float) * scale
        axis = np.asarray(hollow_probe["normal"], float)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        world_c = p0 + c                              # object barely rotated -> add the root translation
        # orient the cylinder's +Z to `axis`
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, axis); s = np.linalg.norm(v); cth = float(np.dot(z, axis))
        if s < 1e-9:
            quat = np.array([1.0, 0.0, 0.0, 0.0]) if cth > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        else:
            ang = math.acos(max(-1.0, min(1.0, cth))); ax = v / s
            quat = np.array([math.cos(ang / 2)] + list(math.sin(ang / 2) * ax))
        probe.set_pos(np.asarray(world_c, np.float32))
        probe.set_quat(np.asarray(quat, np.float32))
        for _ in range(3):
            sc.step()
        # the geom pair (object, probe): redetect the TRUE un-resolved overlap between them.
        pres = max_penetration(sc, redetect=True)
        # isolate the object<->probe contact depth (filter to pairs that include a probe geom)
        probe_ent = probe.idx
        obj_ent = obj.idx
        solver = sc.rigid_solver
        # recompute per-pair max specifically for object<->probe using the same buffer the detector read
        pair_depth_mm = _object_probe_overlap_mm(sc, obj_ent, probe_ent)
        out["hollow_probe_pen_mm"] = float(pair_depth_mm)
        out["hollow_probe_overall_mm"] = float(pres["depth_mm"].max())
    return out


def _object_probe_overlap_mm(scene, obj_ent, probe_ent) -> float:
    """The max solid-solid penetration (mm) specifically between the OBJECT entity and the PROBE entity, read
    from the just-(re)detected contact buffer. ~0 means the probe sits in open space (the feature is hollow)."""
    from genesis.utils.misc import qd_to_torch
    solver = scene.rigid_solver
    collider = solver.collider
    cs = collider._collider_state
    pen = qd_to_torch(cs.contact_data.penetration, transpose=True, copy=False).detach().cpu().numpy()
    ga = qd_to_torch(cs.contact_data.geom_a, transpose=True, copy=False).detach().cpu().numpy()
    gb = qd_to_torch(cs.contact_data.geom_b, transpose=True, copy=False).detach().cpu().numpy()
    nc = qd_to_torch(cs.n_contacts, copy=False).detach().cpu().numpy()
    if pen.ndim == 1:
        pen, ga, gb = pen[None], ga[None], gb[None]
    nc = np.atleast_1d(nc).astype(int)
    geoms = solver.geoms
    n_geoms = len(geoms)
    ent_of = np.full(n_geoms, -1, int)
    for g in geoms:
        if 0 <= g.idx < n_geoms:
            ent_of[g.idx] = g.link.entity.idx
    best = 0.0
    k = int(nc[0])
    for i in range(k):
        a, b = int(ga[0, i]), int(gb[0, i])
        if a < 0 or b < 0:
            continue
        ea, eb = ent_of[a], ent_of[b]
        if {ea, eb} == {obj_ent, probe_ent} and float(pen[0, i]) > best:
            best = float(pen[0, i])
    return best * 1000.0


# ============================================================================ #
# (e) DETECT KEYPOINTS — find meaningful feature frames from GEOMETRY (pure trimesh; no engine).
#   Each keypoint is a LOCAL-frame {center, normal, [radius]} the task solver reads as a VIRTUAL annotation
#   (see make_virtual_keypoint_links / world_keypoint). The detector is general (works on any mesh) and
#   NEVER guesses: a feature is only emitted if its geometric test passes (an encircled hole for a ring; an
#   up-facing rim disc for an opening). The companion verify_keypoints() re-measures each in sim before bake.
# ============================================================================ #
def _encircled_hole(pts_2d: np.ndarray, n_sectors: int = 8, occ_frac: float = 0.875):
    """Given 2-D points (a feature projected onto a plane), find the largest EMPTY disc whose boundary is
    occupied in >= occ_frac of `n_sectors` angular sectors (so it is a genuine ENCIRCLED hole, not a notch at
    the mesh edge). Returns (center_2d, radius) or (None, None). This is the encircled-hole search the mug
    handle needed (a corner-of-mesh max-clearance point is a FALSE positive; require near-full encirclement)."""
    if len(pts_2d) < 12:
        return None, None
    lo, hi = pts_2d.min(0), pts_2d.max(0)
    # candidate centers on a coarse grid inside the bbox; score = distance to the nearest point (clearance)
    gx = np.linspace(lo[0], hi[0], 21)
    gy = np.linspace(lo[1], hi[1], 21)
    best_c, best_r = None, -1.0
    for cx in gx:
        for cy in gy:
            c = np.array([cx, cy])
            d = np.linalg.norm(pts_2d - c, axis=1)
            r = float(d.min())                       # clearance radius to the nearest boundary point
            if r <= best_r:
                continue
            # encirclement test: of the points within a thin annulus [r, 2.2r], how many sectors are occupied?
            near = pts_2d[(d >= r * 0.8) & (d <= r * 2.4)]
            if len(near) < n_sectors:
                continue
            ang = np.arctan2(near[:, 1] - cy, near[:, 0] - cx)
            sec = np.unique(((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors)
            if len(sec) >= occ_frac * n_sectors:
                best_c, best_r = c, r
    if best_c is None:
        return None, None
    return best_c, best_r


def _through_clearance(tm: trimesh.Trimesh, center3: np.ndarray, normal3: np.ndarray, r: float):
    """Probe whether the hole at `center3` (normal `normal3`, radius `r`) is a genuine TUNNEL through the mesh.
    Casts a ray from the hole center along ±normal and returns
        (min_clear_span_m, fully_exits_both_sides: bool)
    or None if either side is blocked within ~1.2 r. ``fully_exits_both_sides`` is True only when the ray hits
    NOTHING on BOTH sides — the unambiguous signature of a handle RING (a stick threads clean through). A cup
    CAVITY (open one end, a bottom on the other) fails this: the ray hits the bottom, so it is finite-clear,
    not a full exit. The span is the smaller of the two clear distances (the threadable depth)."""
    n = np.asarray(normal3, float); n = n / max(np.linalg.norm(n), 1e-9)
    try:
        inter = tm.ray
        origins = np.array([center3, center3])
        dirs = np.array([n, -n])
        locs, ray_idx, _ = inter.intersects_location(origins, dirs, multiple_hits=True)
    except Exception:
        return None
    need = 1.2 * r
    spans = []
    no_hit = [False, False]
    for side in (0, 1):
        hit_d = locs[ray_idx == side]
        if len(hit_d) == 0:
            spans.append(0.5)   # ray exits the mesh entirely on this side -> fully open
            no_hit[side] = True
            continue
        sd = (hit_d - center3) @ (n * (1 if side == 0 else -1))
        sd = sd[sd > 1e-6]
        if len(sd) == 0:
            spans.append(0.5); no_hit[side] = True
        else:
            spans.append(float(sd.min()))
    if min(spans) < need:
        return None
    return float(min(spans)), bool(no_hit[0] and no_hit[1])


def detect_keypoints(mesh_path: str, scale: float = 1.0, *, want_ring: bool = True,
                     want_opening: bool = True) -> dict:
    """DETECT meaningful keypoints from a mesh's GEOMETRY (no engine, no hard-coded constants). Returns a dict
    name -> {center, normal, [radius], method} in the LOCAL frame (so world_kp = object_pose ∘ local_kp).

    Two general detectors (each only emits if its geometric test PASSES — never a guess):
      * handle_ring : the largest ENCIRCLED hole found by sweeping thin slabs along each principal axis and
                      projecting the slab's vertices onto the perpendicular plane. The slab axis that yields the
                      best encircled hole IS the ring normal (the axis a branch threads through). A mug handle's
                      C-loop reads as a fully-encircled hole; a solid object yields NONE.
      * cup_opening : the topmost (max-`up`) roughly-circular RIM. We take the highest 8% of vertices along the
                      principal "up" axis (the one whose +extent is most open / least walled), fit their centroid
                      + a radius, and require they form a ring (low radial spread) — a cup mouth / bottle neck.

    The detector is deliberately conservative: a feature absent from the geometry is simply not returned (the
    caller then reports it as not-found rather than baking a hallucinated frame)."""
    tm = _load_trimesh(mesh_path)
    if scale != 1.0:
        tm = tm.copy(); tm.apply_scale(scale)
    V = np.asarray(tm.vertices, float)
    cen0 = V.mean(0)
    ext = V.max(0) - V.min(0)
    out: dict = {}

    # ---- handle_ring: a fully-ENCIRCLED TUNNEL — a ray along its normal EXITS the mesh on BOTH sides (a stick
    #      threads clean through). This is the unambiguous handle-ring signature. A cup CAVITY fails it (the cup
    #      bottom blocks one ray side -> a finite clear, NOT a full exit). We sweep thin slabs along each axis,
    #      find the fully-encircled (8/8 sectors) empty disc in each, and keep only those that FULLY EXIT both
    #      ways. The handle ring = the SMALLEST such tunnel (the thin loop), clustered across adjacent slabs for
    #      robustness. A solid object (or a one-sided cavity) yields NONE.
    if want_ring:
        tunnels = []   # (radius, center3, normal3, clear)
        for ax in range(3):
            other = [i for i in range(3) if i != ax]
            a = V[:, ax]
            lo, hi = a.min(), a.max()
            for frac in np.linspace(0.10, 0.90, 24):
                cpos = lo + frac * (hi - lo)
                half = 0.045 * (hi - lo) + 1e-4
                sel = np.abs(a - cpos) <= half
                if sel.sum() < 12:
                    continue
                c2, r = _encircled_hole(V[sel][:, other], n_sectors=8, occ_frac=1.0)   # FULL encirclement
                if c2 is None or r is None:
                    continue
                if r < 0.04 * max(ext[other]) or r > 0.5 * min(ext[other]):
                    continue
                center3 = np.zeros(3); center3[ax] = cpos
                center3[other[0]] = c2[0]; center3[other[1]] = c2[1]
                normal3 = np.zeros(3); normal3[ax] = 1.0
                res = _through_clearance(tm, center3, normal3, r)
                if res is None:
                    continue
                clear, full_exit = res
                if not full_exit:                       # ONLY a genuine both-sides-open tunnel (the ring)
                    continue
                tunnels.append((float(r), center3, normal3, float(clear)))
        if tunnels:
            # cluster tunnels by (normal axis, hole center): a real handle ring appears in SEVERAL adjacent
            # slabs at a consistent center; pick the cluster with the most members, then its median geometry.
            tunnels.sort(key=lambda t: t[0])             # smallest radius first (the thin loop)
            # group by normal axis + rounded center (2 mm bins)
            groups: dict = {}
            for r, c3, n3, clear in tunnels:
                key = (int(np.argmax(np.abs(n3))), tuple(np.round(c3 / 0.004).astype(int)))
                groups.setdefault(key, []).append((r, c3, n3, clear))
            # the handle ring's cluster: the most-populated (most slabs agree); tie-break to the smallest radius
            best_key = max(groups, key=lambda k: (len(groups[k]), -np.median([g[0] for g in groups[k]])))
            grp = groups[best_key]
            rs = np.array([g[0] for g in grp]); cs = np.array([g[1] for g in grp])
            n3 = grp[0][2]
            r = float(np.median(rs)); c3 = cs.mean(0); clear = float(np.median([g[3] for g in grp]))
            out["handle_ring"] = dict(center=tuple(round(float(x), 5) for x in c3),
                                      normal=tuple(round(abs(float(x)), 4) for x in n3),
                                      radius=round(float(r), 5), method="encircled_tunnel_both_sides_open",
                                      clear_span_m=round(float(clear), 5), n_slabs=len(grp))

    # ---- cup_opening: the most-open principal axis end; the topmost rim's centroid + radius ----
    if want_opening:
        # the "up" axis: the one whose POSITIVE end is most open. Heuristic: pick the axis with the largest extent
        # whose top slab vertices form a ring (a mouth). Try every axis+sign, keep the best ring-like rim.
        best_open = None  # (ringness, center3, normal3, radius)
        for ax in range(3):
            for sgn in (+1.0, -1.0):
                a = V[:, ax] * sgn
                thr = np.percentile(a, 92.0)
                sel = a >= thr
                if sel.sum() < 16:
                    continue
                rim = V[sel]
                c = rim.mean(0)
                other = [i for i in range(3) if i != ax]
                rad = np.linalg.norm(rim[:, other] - c[other], axis=1)
                rmean = float(rad.mean())
                if rmean < 0.15 * max(ext[other]):
                    continue
                ringness = float(rad.std() / max(rmean, 1e-9))   # low = a clean ring (uniform radius)
                if best_open is None or ringness < best_open[0]:
                    n3 = np.zeros(3); n3[ax] = sgn
                    best_open = (ringness, c.copy(), n3, rmean)
        if best_open is not None and best_open[0] < 0.45:        # require a reasonably circular rim
            _, c, n3, rmean = best_open
            # VERIFY it is a genuine CAVITY opening (measurable): from just inside the rim a ray along -normal
            # must travel a real depth before hitting the cup bottom (a cavity), and from just above the rim a
            # ray along +normal must immediately EXIT (an open top). cup_depth >= 0.3*radius -> a real mouth.
            cup_depth = _cavity_depth(tm, np.asarray(c, float), np.asarray(n3, float), rmean)
            verified = cup_depth is not None and cup_depth >= 0.3 * rmean
            out["cup_opening"] = dict(center=tuple(round(float(x), 5) for x in c),
                                      normal=tuple(round(float(x), 4) for x in n3),
                                      radius=round(float(rmean), 5), method="topmost_rim_centroid",
                                      verified=bool(verified),
                                      verify_detail=(f"cavity_depth={cup_depth:.4f}m (>= 0.3r={0.3*rmean:.4f}) "
                                                     f"-> a genuine open mouth" if cup_depth is not None
                                                     else "no cavity below the rim -> NOT a mouth"),
                                      cavity_depth_m=(round(float(cup_depth), 5) if cup_depth is not None else None))
    return out


def _cavity_depth(tm: trimesh.Trimesh, center3: np.ndarray, normal3: np.ndarray, r: float):
    """Depth of the cavity below a candidate opening (rim `center3`, outward `normal3`, radius `r`). Casts a ray
    from a point just BELOW the rim center (0.2r inside, off-axis to avoid grazing) along -normal: the first hit
    is the cavity floor. Returns that depth (m), or None if there is no floor (a hole, not a cup) or the ray hits
    immediately (a filled/solid top). Used to VERIFY a cup_opening is a real mouth, not a flat face."""
    n = np.asarray(normal3, float); n = n / max(np.linalg.norm(n), 1e-9)
    # a few off-axis origins just inside the rim so we sample the cavity, not the exact (possibly grazing) axis
    u = np.cross(n, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
    depths = []
    for frac in (0.0, 0.3, 0.5):
        origin = center3 - n * (0.02 * r) + u * (frac * r)     # just below the rim, off-axis
        try:
            locs, ridx, _ = tm.ray.intersects_location(origin[None], (-n)[None], multiple_hits=True)
        except Exception:
            return None
        if len(locs) == 0:
            continue
        sd = (locs - origin) @ (-n)
        sd = sd[sd > 1e-4]
        if len(sd):
            depths.append(float(sd.min()))
    if not depths:
        return None
    return float(np.median(depths))


# ============================================================================ #
# (f) VIRTUAL ANNOTATIONS — bake a keypoint as a MASSLESS, COLLISION-LESS, INVISIBLE labeled frame that does
#   NOT change the object's physics/topology/dynamics. Two mechanisms, by source kind:
#     * MESH object (mug, no URDF)  -> keep ObjectSpec.keypoints (LOCAL frame); world_keypoint() returns the
#       live world pose from sim privilege info (object_pose ∘ local_keypoint). NOTHING is added to the sim.
#     * URDF object                  -> append a virtual CHILD LINK: zero mass, NO <collision>, NO <visual>, a
#       FIXED joint at the keypoint pose+axis. Genesis loads it as a frame the solver tracks for free; with no
#       inertia / no geom it cannot perturb dynamics, topology, or contacts. make_virtual_keypoint_links()
#       returns the URDF <link>+<joint> XML the agent appends (the URDF is otherwise untouched).
# ============================================================================ #
def _axis_to_rpy(normal) -> tuple:
    """Roll-pitch-yaw (XYZ fixed) that rotates a link's local +Z onto `normal` — so a virtual link's frame Z is
    the keypoint axis (the branch/thread axis). Returns (r, p, y) radians for a URDF <origin rpy=...>."""
    n = np.asarray(normal, float); n = n / max(np.linalg.norm(n), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n); s = float(np.linalg.norm(v)); c = float(np.dot(z, n))
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    # XYZ-fixed (URDF) rpy from R
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2]); yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1]); yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def make_virtual_keypoint_links(keypoints: dict, parent_link: str, scale: float = 1.0) -> str:
    """Return URDF XML (a fixed <joint> + a massless geom-less <link> per keypoint) the agent APPENDS inside the
    object's <robot>...</robot> — turning each keypoint into a virtual frame the solver tracks WITHOUT changing
    physics. Each link has: zero mass (so no inertia is added), NO <collision> (so no contacts), NO <visual> (so
    invisible). The fixed joint places the frame at the keypoint center with +Z along the keypoint normal.

    NOTE: the link carries a tiny <inertial> with mass=0 and a tiny (1e-9) diagonal inertia — a URDF link must
    have an inertial block to be well-formed, and mass=0 + epsilon-inertia contributes nothing to the dynamics
    (Genesis treats a zero-mass fixed-jointed child as kinematic-only)."""
    xml = []
    for kn, kv in keypoints.items():
        c = np.asarray(kv["center"], float) * scale
        r, p, y = _axis_to_rpy(kv["normal"])
        xml.append(
            f'  <!-- VIRTUAL keypoint frame "{kn}": massless, NO collision/visual; does not perturb physics. -->\n'
            f'  <link name="kp_{kn}">\n'
            f'    <inertial><origin xyz="0 0 0"/><mass value="0.0"/>'
            f'<inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/></inertial>\n'
            f'  </link>\n'
            f'  <joint name="kp_{kn}_fixed" type="fixed">\n'
            f'    <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}"/>\n'
            f'    <parent link="{parent_link}"/>\n'
            f'    <child link="kp_{kn}"/>\n'
            f'  </joint>\n')
    return "".join(xml)


def world_keypoint(local_kp: dict, obj_pos, obj_quat) -> dict:
    """The LIVE world keypoint pose from the object's sim privilege pose (the helper a task calls each step for a
    MESH object). world_center = obj_pos + R(obj_quat) @ local_center; world_normal = R(obj_quat) @ local_normal.
    `obj_pos`/`obj_quat` are the object's root pose (quat wxyz, as Genesis returns). Pure numpy."""
    p = np.asarray(obj_pos, float).reshape(-1)[:3]
    q = np.asarray(obj_quat, float).reshape(-1)[:4]
    q = q / max(np.linalg.norm(q), 1e-9)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
    wc = p + R @ np.asarray(local_kp["center"], float)
    wn = R @ np.asarray(local_kp["normal"], float)
    wn = wn / max(np.linalg.norm(wn), 1e-9)
    out = dict(center=tuple(float(v) for v in wc), normal=tuple(float(v) for v in wn))
    if "radius" in local_kp:
        out["radius"] = float(local_kp["radius"])
    return out


# ============================================================================ #
# (g) SEMANTIC LINK NAMING — give a PartNet-Mobility URDF's raw link/joint INDICES (link_0/link_1/...) CORRECT
#   semantic names from GEOMETRY + KINEMATICS, as an ANNOTATION (the URDF is NEVER renamed — that would break
#   mesh/joint references; the task solver reads the map). VERIFIED by measurement, NEVER hallucinated.
# ============================================================================ #
# Per-category role -> semantic name lexicon. The category comes from the PartNet meta.json `model_cat`. The
# ROLE is derived from geometry+kinematics (movable-small-on-top vs large-fixed-base); the NAME maps the role to
# a word the task solver understands. Conservative: an unknown category falls back to generic role words.
_SEMANTIC_LEXICON = {
    "bottle":  {"movable_top": "cap",      "static_base": "bottle_body"},
    "mug":     {"movable_top": "lid",      "static_base": "mug_body"},
    "kettle":  {"movable_top": "lid",      "static_base": "kettle_body"},
    "stapler": {"movable_top": "top_arm",  "static_base": "base"},
    "laptop":  {"movable_top": "screen",   "static_base": "base"},
    "box":     {"movable_top": "lid",      "static_base": "box_body"},
    "cabinet": {"movable_top": "door",     "static_base": "cabinet_body"},
    "door":    {"movable_top": "door",     "static_base": "frame"},
    "_generic": {"movable_top": "movable_part", "static_base": "base"},
}


def _geom_origin_xyz(vc) -> np.ndarray:
    """The <origin xyz=...> of a <visual>/<collision> element (zero if absent)."""
    o = vc.find("origin")
    if o is not None and o.get("xyz"):
        return np.array([float(x) for x in o.get("xyz").split()], float)
    return np.zeros(3)


def _link_meshes(urdf_path: str, scale: float = 1.0) -> dict:
    """Parse a URDF -> {link_name: combined Trimesh of all its <visual>/<collision> geometry}, in the link frame.
    Handles <mesh> (the PartNet case) AND primitive <box>/<cylinder>/<sphere> (so a synthetic unit-test URDF or a
    primitive-based object also works). Used by semantic_label_links to MEASURE each link (size / volume / pos)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    base_dir = os.path.dirname(os.path.abspath(urdf_path))
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        parts = []
        for tag in ("visual", "collision"):
            for vc in link.findall(tag):
                geo = vc.find("geometry")
                if geo is None:
                    continue
                off = _geom_origin_xyz(vc)
                tm = None
                mesh = geo.find("mesh")
                box = geo.find("box")
                cyl = geo.find("cylinder")
                sph = geo.find("sphere")
                try:
                    if mesh is not None:
                        fn = mesh.get("filename")
                        msc = mesh.get("scale")
                        sc = float(msc.split()[0]) if msc else 1.0
                        mp = fn if os.path.isabs(fn) else os.path.join(base_dir, fn)
                        if os.path.exists(mp):
                            tm = _load_trimesh(mp); tm = tm.copy(); tm.apply_scale(sc)
                    elif box is not None:
                        sz = [float(x) for x in box.get("size").split()]
                        tm = trimesh.creation.box(extents=sz)
                    elif cyl is not None:
                        tm = trimesh.creation.cylinder(radius=float(cyl.get("radius")),
                                                       height=float(cyl.get("length")))
                    elif sph is not None:
                        tm = trimesh.creation.icosphere(radius=float(sph.get("radius")))
                except Exception:
                    tm = None
                if tm is None:
                    continue
                tm = tm.copy()
                tm.apply_translation(off)
                if scale != 1.0:
                    tm.apply_scale(scale)
                parts.append(tm)
            if parts:                          # one tag (visual OR collision) is enough for measurement
                break
        if parts:
            out[name] = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    return out


def semantic_label_links(urdf_path: str, category: str = None, renders: dict = None,
                         truth_path: str = None, scale: float = 1.0) -> dict:
    """Assign CORRECT semantic names to an articulated URDF's raw-index links, from GEOMETRY + JOINT KINEMATICS.
    Returns {raw_link_name: {semantic, role, evidence, confidence, verified}} — an ANNOTATION map (the URDF is
    NEVER renamed). `renders` (view_name->path, optional) is recorded as inspectable evidence; the decision is
    driven by MEASURABLE geometry+kinematics so it is reproducible and not a guess.

    Method (per moving DOF):
      1. Parse links + joints (xml). Classify each REAL link (skip helper/massless links) by:
         - VOLUME (convex-hull): the big link is the BODY/BASE; the small one is the movable part.
         - JOINT to its parent: a fixed joint -> static base; a continuous/revolute/prismatic joint -> movable.
         - POSITION along the joint axis: the movable part sits toward one end (top) of the base.
      2. role = "static_base" for the largest fixed-rooted link; "movable_top" for a small link on a moving joint.
      3. name = _SEMANTIC_LEXICON[category][role] (category from meta.json `model_cat`; generic fallback).
      4. VERIFY: if `truth_path` (a PartNet semantics.txt: `link_X <jointtype> <name>`) is given, compare our
         role-derived name family to the ground-truth name and set `verified` accordingly (NEVER assert verified
         without a check). Confidence reflects how decisively geometry+kinematics separated the links.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    cat = (category or "_generic").lower()
    lex = _SEMANTIC_LEXICON.get(cat, _SEMANTIC_LEXICON["_generic"])

    # joints: child_link -> (type, parent_link, axis)
    jinfo = {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        parent = j.find("parent").get("link")
        ax_el = j.find("axis")
        axis = tuple(float(x) for x in ax_el.get("xyz").split()) if ax_el is not None else (0, 0, 0)
        jinfo[child] = dict(type=j.get("type"), parent=parent, axis=axis)

    meshes = _link_meshes(urdf_path, scale=scale)
    # only links that HAVE geometry are real parts (helper/base massless links are skipped for naming)
    parts = {}
    for ln, tm in meshes.items():
        V = np.asarray(tm.vertices, float)
        parts[ln] = dict(vol=float(tm.convex_hull.volume), centroid=V.mean(0),
                         ext=(V.max(0) - V.min(0)), n=len(V))
    if not parts:
        return {}

    # the BASE = largest-volume link; MOVABLE = the rest (each on a non-fixed effective joint up the chain)
    big = max(parts, key=lambda k: parts[k]["vol"])
    big_vol = parts[big]["vol"]

    def effective_joint_type(link):
        """Walk parent joints up the chain; the first non-fixed joint is this link's effective DOF (PartNet
        often inserts a fixed/helper link between the body and a continuous lid joint)."""
        seen = set()
        cur = link
        while cur in jinfo and cur not in seen:
            seen.add(cur)
            jt = jinfo[cur]["type"]
            if jt != "fixed":
                return jt, jinfo[cur]["axis"]
            cur = jinfo[cur]["parent"]
        return "fixed", (0, 0, 0)

    out = {}
    truth = _parse_partnet_truth(truth_path) if truth_path else {}
    for ln, pinfo in parts.items():
        jt, axis = effective_joint_type(ln)
        is_base = (ln == big) and (jt == "fixed")
        # position of this link's centroid along the dominant joint axis, vs the base centroid (top vs bottom)
        role = "static_base" if is_base else "movable_top"
        semantic = lex[role]
        vol_ratio = pinfo["vol"] / max(big_vol, 1e-12)
        # confidence: a decisive size split + a clear joint signal -> high
        if role == "static_base":
            conf = 0.85 + 0.10 * float(vol_ratio >= 0.99)
        else:
            conf = float(np.clip(0.55 + 0.4 * (1.0 - vol_ratio) + (0.0 if jt == "fixed" else 0.05), 0.0, 0.98))
        ev = (f"vol={pinfo['vol']:.4f} (ratio {vol_ratio:.3f} of largest), eff_joint={jt}, axis={axis}, "
              f"centroid={tuple(round(float(x),3) for x in pinfo['centroid'])}, "
              f"ext={tuple(round(float(x),3) for x in pinfo['ext'])}"
              + (f", renders={list(renders)}" if renders else ""))
        verified = None
        if truth:
            gt = truth.get(ln, {})
            gt_name = gt.get("name", "")
            verified = _name_family_match(semantic, gt_name, gt.get("joint", ""))
            ev += f"  || GROUND-TRUTH({truth_path.split('/')[-1] if truth_path else ''}): " \
                  f"name='{gt_name}' joint='{gt.get('joint','')}' -> match={verified}"
        out[ln] = dict(semantic=semantic, role=role, evidence=ev, confidence=round(float(conf), 2),
                       verified=bool(verified) if verified is not None else None,
                       raw_joint_type=jt)
    return out


def _parse_partnet_truth(path: str) -> dict:
    """Parse a PartNet `semantics.txt` (`link_X <jointtype> <semantic_name>` per line) -> {link: {joint, name}}.
    Used ONLY to VERIFY our derived names against ground truth — never to drive the derivation."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            toks = line.split()
            if len(toks) >= 3:
                out[toks[0]] = dict(joint=toks[1], name=" ".join(toks[2:]))
            elif len(toks) == 2:
                out[toks[0]] = dict(joint="", name=toks[1])
    return out


def _name_family_match(derived: str, truth_name: str, truth_joint: str) -> bool:
    """Is our derived semantic name in the SAME family as the ground-truth name? (e.g. derived 'cap' vs truth
    'rotation_lid' both = a lid/cap; derived 'bottle_body' vs truth 'bottle_body' = body.) Family-based so a
    reasonable synonym counts, but a WRONG role (body labelled as lid) does not."""
    d = derived.lower(); t = truth_name.lower(); tj = (truth_joint or "").lower()
    LID = ("cap", "lid", "top", "door", "screen", "rotation_lid", "rotation")
    BODY = ("body", "base", "bottle_body", "frame", "static")
    d_lid = any(k in d for k in LID); d_body = any(k in d for k in BODY)
    t_lid = any(k in t for k in LID) or tj in ("slider", "slider+", "hinge")
    t_body = any(k in t for k in BODY) or tj == "free"
    if d_lid and t_lid:
        return True
    if d_body and t_body:
        return True
    return False


# ============================================================================ #
# (h) COLLIDER MULTI-VIEW RENDER — render the convex-decomposition HULLS from several canonical viewpoints so
#   the agent (and owner) can INSPECT them as the FINAL collision check (hollow stays hollow + solid never
#   fills). Nyx renders only VISUAL geometry (it exports entity.vgeoms, ignoring vis_mode='collision') — so we
#   BAKE the collision hulls (entity.geoms[*].get_trimesh() at their world poses, each hull tinted a distinct
#   colour so the decomposition + the gaps are visible) into a clean OBJ + matching per-hull material, then
#   render THAT as a visual mesh from 6 views. Runs in a FRESH subprocess (one gs.Scene per process).
# ============================================================================ #
# the 6 canonical viewpoints, as (name, unit direction the camera looks FROM, toward the object center).
COLLIDER_VIEWS = {
    "front":       (1.0, 0.0, 0.0),     # looking down -X (from +X)
    "back":        (-1.0, 0.0, 0.0),
    "left":        (0.0, 1.0, 0.0),     # along the ring normal for the mug -> the hole reads as an open disc
    "right":       (0.0, -1.0, 0.0),
    "top":         (0.0, 0.0, 1.0),     # the cup mouth reads as an open ring from the top
    "perspective": (0.85, -0.85, 0.7),  # a 3/4 view
}


def render_collider_views(visual_obj_abspath: str, out_dir_abspath: str, *, scale: float = 1.0,
                          decompose_error_threshold: float = 0.04, n_views: int = 6,
                          res=(720, 560), spp: int = 32, gpu: str = "0") -> dict:
    """Render the object's COLLIDER (the convex-decomposition hulls) from `n_views` canonical viewpoints and SAVE
    each PNG into `out_dir_abspath` (next to the asset). Returns {view_name: abspath}. Dispatches a FRESH
    subprocess (one gs.Scene per process). The agent/owner opens these as the FINAL collision check."""
    import json
    import subprocess
    os.makedirs(out_dir_abspath, exist_ok=True)
    pybin = os.path.join(os.path.dirname(_PKG), ".venv", "bin", "python")
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    payload = json.dumps(dict(obj=visual_obj_abspath, out_dir=out_dir_abspath, scale=scale,
                              thr=decompose_error_threshold, n_views=int(n_views),
                              res=list(res), spp=int(spp)))
    proc = subprocess.run([pybin, os.path.abspath(__file__), "_stage_collider_views", payload],
                          cwd=os.path.dirname(_PKG), env=env, capture_output=True, text=True)
    line = _grab_result(proc, "COLLIDER_VIEWS_RESULT ")
    if line is None:
        raise RuntimeError(f"collider-views stage failed:\n{proc.stdout[-2500:]}\n{proc.stderr[-2000:]}")
    return json.loads(line)


def _stage_collider_views(payload):
    """(subprocess) Load the object with the convex-DECOMPOSITION collider, BAKE the hulls into a per-hull
    tinted visual scene, and render it from the canonical viewpoints via Nyx. Saves one PNG per view."""
    import json
    import math as _m
    import glob
    import genesis as gs
    import imageio.v3 as iio
    d = json.loads(payload)
    gs.init(backend=gs.gpu)
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx import nyx_py_sdk as nps

    # 1) load the object once to READ its collision hulls (trimesh + world pose) — a throwaway scene.
    sc0 = gs.Scene(show_viewer=False)
    e0 = sc0.add_entity(gs.morphs.Mesh(file=d["obj"], scale=d["scale"], convexify=True,
                                       decompose_object_error_threshold=d["thr"], decimate=False,
                                       fixed=True, pos=(0, 0, 0)))
    sc0.build(n_envs=0)
    hulls = []
    for g in e0.geoms:
        tm = g.get_trimesh()
        p = g.get_pos(); q = g.get_quat()
        p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
        q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
        hulls.append((np.asarray(tm.vertices, float).copy(), np.asarray(tm.faces, int).copy(),
                      p.reshape(-1)[:3].astype(float), q.reshape(-1)[:4].astype(float)))
    # object center + radius (for camera framing), from all hull verts in world frame
    allv = []
    for verts, faces, p, q in hulls:
        allv.append(_apply_pose(verts, p, q))
    allv = np.concatenate(allv, 0)
    center = (allv.min(0) + allv.max(0)) / 2.0
    radius = float(np.linalg.norm(allv.max(0) - allv.min(0)) / 2.0)

    # 2) bake EACH hull into its own OBJ (in world frame), tinted a distinct colour so the decomposition shows.
    #    These are SCRATCH (only needed to feed Nyx the per-hull visual) -> written under a temp dir and removed
    #    after the renders, so only the deliverable PNGs remain next to the asset.
    import tempfile
    bake_dir = tempfile.mkdtemp(prefix="collider_hulls_")
    hull_objs = []
    for i, (verts, faces, p, q) in enumerate(hulls):
        wv = _apply_pose(verts, p, q)
        tm = trimesh.Trimesh(vertices=wv, faces=faces, process=False)
        hp = os.path.join(bake_dir, f"hull_{i:03d}.obj")
        tm.export(hp)
        hull_objs.append(hp)

    # 3) a fresh render scene: NO physics needed; add each hull as a fixed visual mesh with a distinct colour.
    palette = _hull_palette(len(hull_objs))
    sc = gs.Scene(show_viewer=False)
    for i, hp in enumerate(hull_objs):
        sc.add_entity(gs.morphs.Mesh(file=hp, fixed=True, collision=False, pos=(0, 0, 0)),
                      surface=gs.surfaces.Plastic(color=palette[i], roughness=0.55))
    # a neutral env map + key light so the hulls read clearly (a dark studio look so colour gaps pop)
    bg = "/home/kaitianchao/Projects/RoboLab_firefly/assets/backgrounds"
    hdrs = sorted(glob.glob(f"{bg}/indoors/*.hdr"))
    emaps = []
    if hdrs:
        em = nps.EnvironmentMapAsset(); em.texture = hdrs[0]; em.layout = nps.EEnvMapLayout.LongLat
        em.multiplier = 0.6; emaps = [em]
    lights = [{"dir": (-0.4, 0.3, -0.85), "color": (1, 1, 1), "intensity": 1.6,
               "directional": True, "castshadow": True}]
    # one camera per view; build once, read each after re-aiming
    views = list(COLLIDER_VIEWS.items())[:d["n_views"]]
    cams = {}
    dist = max(radius * 3.2, 0.18)
    for name, dir3 in views:
        dv = np.asarray(dir3, float); dv = dv / max(np.linalg.norm(dv), 1e-9)
        pos = center + dv * dist
        cams[name] = sc.add_sensor(NyxCameraOptions(res=tuple(d["res"]), pos=tuple(pos.tolist()),
                                   lookat=tuple(center.tolist()), fov=38, lights=lights,
                                   env_maps=tuple(emaps), spp=d["spp"], denoise=True))
    sc.build(n_envs=0)
    out = {}
    for name, cam in cams.items():
        cam._stale = True
        img = cam.read().rgb
        img = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
        op = os.path.join(d["out_dir"], f"{name}.png")
        iio.imwrite(op, img.astype(np.uint8))
        out[name] = op
    import shutil
    shutil.rmtree(bake_dir, ignore_errors=True)        # drop the scratch hull OBJs; keep only the PNGs
    print("COLLIDER_VIEWS_RESULT " + json.dumps(out), flush=True)


def _apply_pose(verts: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Transform local verts by (pos, quat wxyz) -> world."""
    w, x, y, z = q / max(np.linalg.norm(q), 1e-9)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
    return verts @ R.T + p


def _hull_palette(n: int) -> list:
    """n visually distinct, well-separated RGB colours (HSV golden-angle hop) so adjacent hulls differ and the
    decomposition + the OPEN gaps (hollow features) are easy to see in the render."""
    import colorsys
    cols = []
    h = 0.13
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.62, 0.92)
        cols.append((float(r), float(g), float(b)))
        h += 0.61803398875
    return cols


# ============================================================================ #
# (i) KEYPOINT-OVERLAY RENDER — draw the labelled keypoint frames (a coloured marker disc + a normal-axis arrow)
#   ON the settled object so the agent/owner can EYEBALL that the ring center+normal land on the real ring and
#   the cup opening on the real mouth. The frames are virtual (drawn only for this verification render).
# ============================================================================ #
def render_keypoint_overlay(visual_obj_abspath: str, keypoints: dict, out_png_abspath: str, *,
                            scale: float = 1.0, density: float = 1000.0,
                            decompose_error_threshold: float = 0.04, gpu: str = "0") -> str:
    """Render the object with each keypoint drawn as a coloured ring-disc (at its center) + an arrow along its
    normal. Saves a PNG; returns its path. Fresh subprocess (one gs.Scene per process)."""
    import json
    import subprocess
    os.makedirs(os.path.dirname(out_png_abspath), exist_ok=True)
    pybin = os.path.join(os.path.dirname(_PKG), ".venv", "bin", "python")
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    payload = json.dumps(dict(obj=visual_obj_abspath, keypoints=keypoints, out=out_png_abspath,
                              scale=scale, density=density, thr=decompose_error_threshold))
    proc = subprocess.run([pybin, os.path.abspath(__file__), "_stage_kp_overlay", payload],
                          cwd=os.path.dirname(_PKG), env=env, capture_output=True, text=True)
    line = _grab_result(proc, "KP_OVERLAY_RESULT ")
    if line is None:
        raise RuntimeError(f"keypoint-overlay stage failed:\n{proc.stdout[-2500:]}\n{proc.stderr[-2000:]}")
    return json.loads(line)


def _stage_kp_overlay(payload):
    """(subprocess) Settle the object, then draw each keypoint as a thin RING (a torus-ish disc of small spheres)
    at its world center + a coloured arrow (cylinder) along its world normal, and render one labelled frame."""
    import json
    import math as _m
    import glob
    import genesis as gs
    import imageio.v3 as iio
    from world.firefly_scene import firm_rigid_options
    d = json.loads(payload)
    gs.init(backend=gs.gpu)
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx import nyx_py_sdk as nps

    table_z = 0.25
    sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
                  rigid_options=firm_rigid_options(), show_viewer=False)
    sc.add_entity(gs.morphs.Box(size=(0.7, 0.9, table_z), pos=(0.4, 0.0, table_z / 2.0),
                                fixed=True, collision=True),
                  surface=gs.surfaces.Plastic(color=(0.30, 0.30, 0.33), roughness=0.7))
    # display the object at a FIXED known pose (handle +X local toward the camera) so the labelled frames are
    # unambiguous in the render. (Settle-stability is verified separately by the init-stability gate; this render
    # is purely to EYEBALL that the detected keypoints land on the real features.)
    disp_quat = d.get("disp_quat", [1.0, 0.0, 0.0, 0.0])
    obj = sc.add_entity(
        gs.morphs.Mesh(file=d["obj"], scale=d["scale"], convexify=True,
                       decompose_object_error_threshold=d["thr"], decimate=False, fixed=True,
                       pos=(0.40, 0.0, table_z + 0.05), quat=tuple(disp_quat)),
        material=gs.materials.Rigid(rho=d["density"], friction=1.0),
        surface=gs.surfaces.Plastic(color=(0.88, 0.88, 0.91), roughness=0.45))

    # one distinct colour per keypoint; build marker entities (a small sphere at center + an arrow cylinder)
    kp_colours = [(0.95, 0.20, 0.20), (0.20, 0.45, 0.95), (0.20, 0.85, 0.30), (0.95, 0.75, 0.15)]
    markers = []   # (kp, center_sphere, arrow_cyl, ring_spheres)
    kps = list(d["keypoints"].items())
    for i, (kn, kv) in enumerate(kps):
        col = kp_colours[i % len(kp_colours)]
        r = float(kv.get("radius", 0.012)) * d["scale"]
        center_sphere = sc.add_entity(gs.morphs.Sphere(radius=max(0.004, 0.18 * r), fixed=True, collision=False),
                                      surface=gs.surfaces.Plastic(color=col, roughness=0.4))
        arrow = sc.add_entity(gs.morphs.Cylinder(radius=0.0025, height=max(0.05, 3.0 * r), fixed=True,
                                                 collision=False),
                              surface=gs.surfaces.Plastic(color=col, roughness=0.4))
        # the ring: a circle of tiny spheres at radius r in the plane perpendicular to the normal
        ring = [sc.add_entity(gs.morphs.Sphere(radius=0.0022, fixed=True, collision=False),
                              surface=gs.surfaces.Plastic(color=col, roughness=0.4)) for _ in range(16)]
        markers.append((kv, center_sphere, arrow, ring, r))

    bg = "/home/kaitianchao/Projects/RoboLab_firefly/assets/backgrounds"
    hdrs = sorted(glob.glob(f"{bg}/indoors/*.hdr"))
    emaps = []
    if hdrs:
        em = nps.EnvironmentMapAsset(); em.texture = hdrs[0]; em.layout = nps.EEnvMapLayout.LongLat
        em.multiplier = 1.0; emaps = [em]
    lights = [{"dir": (-0.4, 0.3, -0.85), "color": (1, 1, 1), "intensity": 1.4,
               "directional": True, "castshadow": True}]
    # a tight 3/4 view from +X / -Y / above so BOTH the +X handle ring and the top cup mouth are in frame.
    cam = sc.add_sensor(NyxCameraOptions(res=(960, 740), pos=(0.66, -0.20, 0.40),
                        lookat=(0.40, 0.0, 0.285), fov=34, lights=lights,
                        env_maps=tuple(emaps), spp=40, denoise=True))
    sc.build(n_envs=0)

    for _ in range(2):                                # the object is fixed at its display pose (no settling)
        sc.step()
    mp = obj.get_pos(); mp = (mp.cpu().numpy() if hasattr(mp, "cpu") else np.asarray(mp)).reshape(-1)[:3]
    mq = obj.get_quat(); mq = (mq.cpu().numpy() if hasattr(mq, "cpu") else np.asarray(mq)).reshape(-1)[:4]

    for kv, csph, arrow, ring, r in markers:
        wk = world_keypoint(kv, mp, mq)             # live world center + normal from the settled object pose
        c = np.asarray(wk["center"], np.float32); n = np.asarray(wk["normal"], float)
        n = n / max(np.linalg.norm(n), 1e-9)
        csph.set_pos(c)
        # arrow along the normal (its +Z -> n), shifted so it starts at the center and points out
        quat = _quat_z_to(n)
        arrow.set_pos((c + n.astype(np.float32) * (max(0.05, 3.0 * r) * 0.5)).astype(np.float32))
        arrow.set_quat(np.asarray(quat, np.float32))
        # the ring of spheres in the plane perpendicular to n
        u = np.cross(n, [0, 0, 1.0]); u = u / max(np.linalg.norm(u), 1e-9) if np.linalg.norm(u) > 1e-6 \
            else np.array([1.0, 0, 0])
        v = np.cross(n, u)
        for k, s in enumerate(ring):
            a = 2 * _m.pi * k / len(ring)
            p = c + (np.cos(a) * u + np.sin(a) * v) * r
            s.set_pos(p.astype(np.float32))
    for _ in range(2):
        sc.step()
    cam._stale = True
    img = cam.read().rgb
    img = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    os.makedirs(os.path.dirname(d["out"]), exist_ok=True)
    iio.imwrite(d["out"], img.astype(np.uint8))
    print("KP_OVERLAY_RESULT " + json.dumps(d["out"]), flush=True)


def _quat_z_to(n):
    """Quaternion (wxyz) rotating +Z onto unit vector n."""
    import math as _m
    z = np.array([0.0, 0.0, 1.0]); n = np.asarray(n, float)
    v = np.cross(z, n); s = float(np.linalg.norm(v)); c = float(np.dot(z, n))
    if s < 1e-9:
        return [1.0, 0.0, 0.0, 0.0] if c > 0 else [0.0, 1.0, 0.0, 0.0]
    ang = _m.acos(max(-1.0, min(1.0, c))); ax = v / s
    return [_m.cos(ang / 2)] + list(_m.sin(ang / 2) * ax)


# ============================================================================ #
# refine() — the orchestrator: augment -> infer -> collision -> verify -> RefineReport
# (the collision-extract + the verify stages each run in a FRESH subprocess: one gs.Scene per process)
# ============================================================================ #
def refine(name: str, source_path: str, *, source_kind: str = "mesh", category: str = "generic",
           target_size_m: Optional[float] = None, target_axis: Optional[int] = None,
           scale: float = 1.0, out_obj_relpath: Optional[str] = None,
           decompose_error_threshold: float = 0.04, keypoints: Optional[dict] = None,
           hollow_probe: Optional[dict] = None, gpu: str = "0",
           density: Optional[float] = None, friction: Optional[tuple] = None,
           restitution: Optional[float] = None, run_sim: bool = True,
           detect_kps: bool = True, render_collider: bool = True, n_collider_views: int = 6) -> RefineReport:
    """Refine a raw object into a sim-ready asset + a verified RefineReport.

    The AUGMENT (analyze_mesh) + INFER (infer_physics) stages run in-process (trimesh only). The COLLISION
    extract + the VERIFY stages each touch Genesis, so — because a 2nd gs.Scene in one process segfaults — they
    are dispatched to FRESH subprocesses via __main__ stages (see `_stage_*`). For a USD source we first need a
    trimesh to analyze; we extract the clean OBJ (collision stage) FIRST, then analyze that OBJ.

    Returns a fully-populated RefineReport (sim_ready = penetration PASS AND stability PASS [AND hollow PASS if a
    probe was given]).
    """
    assets = os.path.join(_PKG, "assets", "objects")
    is_usd = source_path.lower().endswith((".usd", ".usda"))
    out_obj_rel = out_obj_relpath or f"{source_kind}/{name}_clean.obj"
    out_obj_abs = os.path.join(assets, out_obj_rel)

    # --- COLLISION (subprocess): extract the Nyx-safe clean OBJ with the convex-DECOMPOSITION collider ---
    collision = _run_stage_extract(source_path, out_obj_abs, scale, decompose_error_threshold, gpu)

    # --- AUGMENT: analyze the extracted clean OBJ (USD has no trimesh reader; the OBJ is the analyzable geometry)
    analyze_src = out_obj_abs if is_usd else source_path
    mesh = analyze_mesh(analyze_src, target_size_m=target_size_m, target_axis=target_axis)

    # --- INFER physics from the category prior x the measured solid volume ---
    physics = infer_physics(mesh, category=category, density=density, friction=friction,
                            restitution=restitution, scale=scale)

    report = RefineReport(name=name, source_path=source_path, source_kind=source_kind,
                          mesh=mesh, physics=physics, collision=collision,
                          keypoints=dict(keypoints or {}))

    # --- DETECT keypoints from the clean OBJ geometry (general; never guesses; pure trimesh, in-process) ---
    if detect_kps:
        try:
            report.detected_keypoints = detect_keypoints(out_obj_abs, scale=scale)
        except Exception as ex:                      # detection must never crash the refine; just note it
            report.detected_keypoints = {"_error": str(ex)}
        # auto-derive the hollow probe from the DETECTED handle_ring if the caller didn't pass one, so the same
        # in-sim hollow gate that proves "the ring is open" ALSO verifies the detected keypoint is correct.
        hr = report.detected_keypoints.get("handle_ring") if isinstance(report.detected_keypoints, dict) else None
        if hollow_probe is None and hr is not None:
            pr = min(0.004, 0.45 * float(hr.get("radius", 0.008)))      # a branch thinner than the hole
            hollow_probe = dict(center=hr["center"], normal=hr["normal"], radius=pr, length=0.20,
                                _verifies="handle_ring")

    # --- MULTI-VIEW COLLIDER RENDER (the final visual collision check; subprocess) -> saved NEXT TO the asset ---
    if render_collider:
        views_dir = os.path.join(os.path.dirname(out_obj_abs), f"{name}_collider_views")
        try:
            report.collider_views = render_collider_views(
                out_obj_abs, views_dir, scale=scale, decompose_error_threshold=decompose_error_threshold,
                n_views=n_collider_views, gpu=gpu)
        except Exception as ex:
            report.add("collider_render", False, f"collider multi-view render FAILED: {ex}")

    if not run_sim:
        return report

    # --- VERIFY (subprocess): penetration + stability [+ hollow probe] ---
    sim = _run_stage_verify(out_obj_abs, physics, scale, hollow_probe,
                            float(physics.friction[0]), decompose_error_threshold, gpu)

    # fold the raw measurements into hard-gate Verdicts
    pen_ok = (sim["rest_abnormal"] == 0)
    report.add("penetration_at_rest", pen_ok,
               f"abnormal={sim['rest_abnormal']} (thresh {1000*0.007:.0f}mm); max_pen={sim['rest_max_pen_mm']:.2f}mm"
               + (f" worst={sim.get('worst_pair')}" if sim.get('worst_pair') else ""))
    stable = (not sim["exploded"]) and sim["d_pos_m"] <= POS_TOL_M and sim["d_ori_rad"] <= ORI_TOL_RAD
    report.add("init_stability", stable,
               f"D_pos={sim['d_pos_m']*1000:.2f}mm (<= {POS_TOL_M*1000:.0f}mm)  "
               f"D_ori={sim['d_ori_rad']:.4f}rad (<= {ORI_TOL_RAD:.2f})  exploded={sim['exploded']}")
    hollow_ok = True
    if hollow_probe is not None and "hollow_probe_pen_mm" in sim:
        hp = sim["hollow_probe_pen_mm"]
        hollow_ok = hp <= 0.007 * 1000  # the same ABNORMAL_THRESH_M (7mm): a clean thread reads ~0
        report.add("hollow_feature_open", hollow_ok,
                   f"probe<->object overlap={hp:.2f}mm (<= 7mm) -> a branch CAN thread the ring")
        # MARK the detected keypoint VERIFIED: the same in-sim probe that proved the ring is open is the
        # measurement that confirms the DETECTED handle_ring center+normal is correct (a wrong center would
        # collide -> the gate would FAIL). Only set verified=True on a PASS; never bake an unverified label.
        which = hollow_probe.get("_verifies") if isinstance(hollow_probe, dict) else None
        if which and isinstance(report.detected_keypoints, dict) and which in report.detected_keypoints:
            report.detected_keypoints[which]["verified"] = bool(hollow_ok)
            report.detected_keypoints[which]["verify_detail"] = \
                f"in-sim hollow probe through detected center: overlap={hp:.2f}mm (PASS<=7mm)"

    report.sim_ready = bool(pen_ok and stable and hollow_ok)
    return report


# ---- subprocess stage dispatch (one gs.Scene per process) --------------------------------------------------
def _run_stage_extract(source_path, out_obj_abs, scale, thr, gpu) -> CollisionModel:
    import json
    import subprocess
    pybin = os.path.join(os.path.dirname(_PKG), ".venv", "bin", "python")
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    payload = json.dumps(dict(source=source_path, out=out_obj_abs, scale=scale, thr=thr))
    proc = subprocess.run([pybin, os.path.abspath(__file__), "_stage_extract", payload],
                          cwd=os.path.dirname(_PKG), env=env, capture_output=True, text=True)
    line = _grab_result(proc, "EXTRACT_RESULT ")
    if line is None:
        raise RuntimeError(f"extract stage failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    d = json.loads(line)
    return CollisionModel(**d)


def _run_stage_verify(out_obj_abs, physics, scale, hollow_probe, friction, thr, gpu) -> dict:
    import json
    import subprocess
    pybin = os.path.join(os.path.dirname(_PKG), ".venv", "bin", "python")
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    payload = json.dumps(dict(obj=out_obj_abs, density=physics.density_kg_m3, scale=scale,
                              hollow_probe=hollow_probe, friction=friction, thr=thr))
    proc = subprocess.run([pybin, os.path.abspath(__file__), "_stage_verify", payload],
                          cwd=os.path.dirname(_PKG), env=env, capture_output=True, text=True)
    line = _grab_result(proc, "VERIFY_RESULT ")
    if line is None:
        raise RuntimeError(f"verify stage failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return json.loads(line)


def _grab_result(proc, tag):
    for line in (proc.stdout or "").splitlines():
        if line.startswith(tag):
            return line[len(tag):]
    return None


def _main_stage_extract(payload):
    import json
    import genesis as gs
    d = json.loads(payload)
    gs.init(backend=gs.gpu)
    cm = extract_visual_obj(d["source"], d["out"], scale=d["scale"],
                            decompose_error_threshold=d["thr"])
    print("EXTRACT_RESULT " + json.dumps(asdict(cm)), flush=True)


def _main_stage_verify(payload):
    import json
    import genesis as gs
    d = json.loads(payload)
    gs.init(backend=gs.gpu)
    phys = PhysicsInfer(category="generic", density_kg_m3=d["density"], mass_kg=0.0,
                        friction=(d["friction"], 0.9), restitution=0.1)
    out = verify_in_sim(d["obj"], phys, scale=d["scale"], hollow_probe=d["hollow_probe"],
                        friction=d["friction"], decompose_error_threshold=d["thr"])
    print("VERIFY_RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    # subprocess stage entrypoints (refine() dispatches these so each gs.Scene lives in its own process)
    if len(sys.argv) >= 3 and sys.argv[1] == "_stage_extract":
        _main_stage_extract(sys.argv[2]); sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "_stage_verify":
        _main_stage_verify(sys.argv[2]); sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "_stage_collider_views":
        _stage_collider_views(sys.argv[2]); sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "_stage_kp_overlay":
        _stage_kp_overlay(sys.argv[2]); sys.exit(0)
    # default: run the MUG demo end-to-end (the deliverable). See demo_mug.py for the verification render.
    from registry.demo_mug import main as demo_main
    demo_main()
