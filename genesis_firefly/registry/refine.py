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
# refine() — the orchestrator: augment -> infer -> collision -> verify -> RefineReport
# (the collision-extract + the verify stages each run in a FRESH subprocess: one gs.Scene per process)
# ============================================================================ #
def refine(name: str, source_path: str, *, source_kind: str = "mesh", category: str = "generic",
           target_size_m: Optional[float] = None, target_axis: Optional[int] = None,
           scale: float = 1.0, out_obj_relpath: Optional[str] = None,
           decompose_error_threshold: float = 0.04, keypoints: Optional[dict] = None,
           hollow_probe: Optional[dict] = None, gpu: str = "0",
           density: Optional[float] = None, friction: Optional[tuple] = None,
           restitution: Optional[float] = None, run_sim: bool = True) -> RefineReport:
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
    # default: run the MUG demo end-to-end (the deliverable). See demo_mug.py for the verification render.
    from registry.demo_mug import main as demo_main
    demo_main()
