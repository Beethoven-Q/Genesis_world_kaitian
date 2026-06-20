# SPDX-License-Identifier: Apache-2.0
"""DEMO: refine the YCB mug into a sim-ready asset end-to-end, VERIFY it (penetration + stability + hollow),
emit the ObjectSpec line + the handle-ring keypoint, and save a verification render.

The mug is the ideal demonstrator: a HOLLOW handle ring (tests "hollow stays hollow" + the ring center+normal
keypoint for the future virtual-EE / mug-hang) and a hollow cup opening. We refine the RoboLab YCB mug.usd
(copied into genesis_firefly/assets/objects/ycb/mug.usd).

Geometry was MEASURED first (registry/refine.py analyze_mesh + an X-Z silhouette + an encircled-hole search):
  extents ~ (0.117, 0.093, 0.081) m  (X = handle-out axis, Z = cup vertical axis)
  handle (the C-loop) protrudes along +X, spans the cup height (Z), opening toward the cup body.
  handle HOLE center ~ (+0.0395, 0, 0) local, radius ~10.5 mm (fully encircled, 8/8 sectors); a branch
  threads it along +/-Y (the ring NORMAL). The hole radius bounds the branch thickness it can hang on.

Run:  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/registry/demo_mug.py
Writes the verification render to genesis_firefly/output/temp/mug_refine_verify.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from registry.refine import refine  # noqa: E402

MUG_USD = os.path.join(_PKG, "assets", "objects", "ycb", "mug.usd")
RENDER_OUT = os.path.join(_PKG, "output", "temp", "mug_refine_verify.png")

# --- the handle-ring keypoint (MEASURED; the future virtual-EE / mug-hang frame) ----------------------------
# center: the empty hole between the cup wall and the handle's outer arc, in the LOCAL frame.
# normal: +Y -- the axis a horizontal branch passes through to hang the mug (perpendicular to handle-out X and
#         cup-up Z). radius: the clear hole radius a branch up to this thickness can thread.
RING_CENTER = (0.0395, 0.0, 0.0)
RING_NORMAL = (0.0, 1.0, 0.0)
RING_RADIUS = 0.0105
CUP_OPENING = dict(center=(-0.012, 0.0, 0.040), normal=(0.0, 0.0, 1.0), radius=0.045)
KEYPOINTS = {
    "handle_ring": dict(center=RING_CENTER, normal=RING_NORMAL, radius=RING_RADIUS),
    # the cup opening (mouth) center+normal -- the future "drop a cube in the mug" / pour frame.
    "cup_opening": CUP_OPENING,
}
# the hollow probe: a thin branch-like cylinder threaded through the handle ring along its normal. radius 4mm
# leaves ~6mm clearance in the ~10.5mm hole, so a clean thread reads ~0 overlap (an unambiguous hollow proof).
HOLLOW_PROBE = dict(center=RING_CENTER, normal=RING_NORMAL, radius=0.004, length=0.20)


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # v2: detect_kps=True -> the refiner DETECTS the keypoints from geometry (no hand probe needed); the auto
    # hollow probe is derived from the detected handle_ring, so the same in-sim gate VERIFIES the detection.
    # render_collider=True -> the multi-view COLLIDER render (the final collision check) saved next to the asset.
    report = refine(
        name="mug", source_path=MUG_USD, source_kind="ycb", category="ceramic",
        out_obj_relpath="ycb/mug_clean.obj",
        decompose_error_threshold=0.04,
        keypoints=KEYPOINTS, hollow_probe=None,           # None -> auto-probe the DETECTED ring (verifies it)
        detect_kps=True, render_collider=True, n_collider_views=6,
        gpu=gpu,
    )
    print(report.pretty(), flush=True)

    # --- the DETECTED keypoints (geometry; verified in sim) vs the hand-MEASURED ones (cross-check) ---
    print("\n# --- DETECTED vs HAND-MEASURED keypoints (cross-check; detected are the ones to bake) ---")
    for kn in ("handle_ring", "cup_opening"):
        det = report.detected_keypoints.get(kn) if isinstance(report.detected_keypoints, dict) else None
        man = KEYPOINTS.get(kn)
        if det:
            v = "VERIFIED" if det.get("verified") else "unverified"
            print(f"   {kn:<12} detected[{v}] center={tuple(round(x,4) for x in det['center'])} "
                  f"normal={tuple(round(x,3) for x in det['normal'])} r={det.get('radius'):.4f}")
        print(f"   {kn:<12} hand      center={man['center']} normal={man['normal']} r={man['radius']}")

    # --- the VIRTUAL-ANNOTATION mechanism for the mug (a MESH object, no URDF) -------------------------------
    # The mug has no URDF, so its keypoints live in ObjectSpec.keypoints (LOCAL frame); the task gets the LIVE
    # world keypoint via refine.world_keypoint(local_kp, obj_pos, obj_quat) — object_pose ∘ local_keypoint. This
    # adds NOTHING to the sim (no mass, no collision, no link), so it CANNOT perturb the object's physics. (For a
    # URDF object the refiner instead appends a massless/collision-less/invisible virtual child LINK via
    # refine.make_virtual_keypoint_links + links_to_keep — proven physics-inert in scripts/temp.)
    print("\n# --- virtual-annotation: mug uses the ObjectSpec.keypoints local-frame mechanism (no sim object) ---")
    print("#     task reads the live world frame via refine.world_keypoint(local_kp, mug.get_pos(), mug.get_quat())")

    # --- emit the ObjectSpec line for registry/object_spec.py (the agent pastes this into the REGISTRY) ---
    # use the DETECTED + sim-VERIFIED keypoints (falls back to the hand-measured if detection is missing).
    m, p = report.mesh, report.physics
    la = m.long_axis_local
    dk = report.detected_keypoints if isinstance(report.detected_keypoints, dict) else {}
    hr = dk.get("handle_ring") or dict(center=RING_CENTER, normal=RING_NORMAL, radius=RING_RADIUS)
    co = dk.get("cup_opening") or CUP_OPENING
    hr_c = tuple(round(float(x), 4) for x in hr["center"]); hr_n = tuple(round(float(x), 3) for x in hr["normal"])
    co_c = tuple(round(float(x), 4) for x in co["center"]); co_n = tuple(round(float(x), 3) for x in co["normal"])
    print("\n# --- ObjectSpec to add to registry/object_spec.py REGISTRY (DETECTED+VERIFIED keypoints) ---")
    print(f'''    "mug": ObjectSpec(
        name="mug", language_name="mug", source="usd", usd_subpath="ycb/mug.usd",
        mesh_subpath="ycb/mug_clean.obj", dist_color=(0.85, 0.85, 0.88),
        mass={p.mass_kg:.3f}, extents=({m.extents_m[0]:.4f}, {m.extents_m[1]:.4f}, {m.extents_m[2]:.4f}),
        local_center=(0.0, 0.0, 0.0), friction=({p.friction[0]}, {p.friction[1]}),
        elongated=True, local_long_axis=({la[0]:.4f}, {la[1]:.4f}, {la[2]:.4f}),
        color=(0.90, 0.90, 0.93), grasp_dz=0.0, x_range=(0.34, 0.44), place_xy_tol_cm=8.0,
        keypoints={{
            "handle_ring": dict(center={hr_c}, normal={hr_n}, radius={round(float(hr["radius"]),4)}),
            "cup_opening": dict(center={co_c}, normal={co_n}, radius={round(float(co["radius"]),4)}),
        }}),''')

    # --- write a machine-readable report next to the render for the record ---
    os.makedirs(os.path.dirname(RENDER_OUT), exist_ok=True)
    with open(os.path.join(os.path.dirname(RENDER_OUT), "mug_refine_report.json"), "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # --- the verification RENDER: a tight close-up of the mug + a branch threaded through the handle ring,
    #     proving the ring is OPEN (hollow stays hollow). Rendered in a fresh subprocess (one gs.Scene/proc). ---
    _render_verify(report)
    print(f"\nVERIFICATION RENDER -> {RENDER_OUT}", flush=True)

    # --- the KEYPOINT-OVERLAY render: the labelled frames (ring + cup mouth, center+normal) drawn ON the mug ---
    from registry.refine import render_keypoint_overlay
    kp_for_overlay = {k: v for k, v in (report.detected_keypoints or {}).items() if k in ("handle_ring", "cup_opening")}
    kp_out = os.path.join(_PKG, "output", "temp", "mug_keypoints_verify.png")
    try:
        render_keypoint_overlay(os.path.join(_PKG, "assets", "objects", "ycb", "mug_clean.obj"),
                                kp_for_overlay, kp_out, density=report.physics.density_kg_m3, gpu=gpu)
        print(f"KEYPOINT-OVERLAY RENDER -> {kp_out}", flush=True)
    except Exception as ex:
        print(f"(keypoint overlay skipped: {ex})")

    print("DEMO_DONE sim_ready=" + str(report.sim_ready), flush=True)
    return report


def _render_verify(report):
    """Spawn a fresh subprocess to render the close-up (the main process already used a gs.Scene in the
    subprocess stages, but this driver process itself has not -- still, keep it isolated for cleanliness)."""
    import subprocess
    pybin = os.path.join(os.path.dirname(_PKG), ".venv", "bin", "python")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # thread the branch through the DETECTED handle-ring center (falls back to the hand-measured one)
    det = report.detected_keypoints.get("handle_ring") if isinstance(report.detected_keypoints, dict) else None
    rc = tuple(det["center"]) if det else RING_CENTER
    rn = tuple(det["normal"]) if det else RING_NORMAL
    payload = json.dumps(dict(
        obj=os.path.join(_PKG, "assets", "objects", "ycb", "mug_clean.obj"),
        out=RENDER_OUT, ring_center=rc, ring_normal=rn,
        density=report.physics.density_kg_m3))
    proc = subprocess.run([pybin, os.path.abspath(__file__), "_render", payload],
                          cwd=os.path.dirname(_PKG), env=env, capture_output=True, text=True)
    if not os.path.exists(RENDER_OUT):
        print("RENDER STAGE OUTPUT:\n", proc.stdout[-3000:], proc.stderr[-2000:])


def _stage_render(payload):
    """Render a tight close-up: the mug (clean OBJ visual + convex-decomposition collider) on a table with a
    thin BRANCH cylinder threaded through the handle ring. If the ring is genuinely hollow, the branch passes
    cleanly through the opening (visible gap), proving a mug-hang is geometrically possible."""
    import math
    import genesis as gs
    import imageio.v3 as iio
    from world.firefly_scene import firm_rigid_options

    d = json.loads(payload)
    gs.init(backend=gs.gpu)

    # --- Nyx photoreal close-up sensor (reuse the stage's Nyx wiring pattern) ---
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx import nyx_py_sdk as nps
    bg = "/home/kaitianchao/Projects/RoboLab_firefly/assets/backgrounds"
    import glob
    hdrs = sorted(glob.glob(f"{bg}/indoors/*.hdr"))
    emaps = []
    if hdrs:
        e = nps.EnvironmentMapAsset(); e.texture = hdrs[0]; e.layout = nps.EEnvMapLayout.LongLat
        e.multiplier = 1.0; emaps = [e]
    lights = [{"dir": (-0.4, 0.3, -0.85), "color": (1, 1, 1), "intensity": 1.4,
               "directional": True, "castshadow": True}]

    table_z = 0.25
    sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
                  rigid_options=firm_rigid_options(), show_viewer=False)
    sc.add_entity(gs.morphs.Box(size=(0.7, 0.9, table_z), pos=(0.4, 0.0, table_z / 2.0),
                                fixed=True, collision=True),
                  surface=gs.surfaces.Plastic(color=(0.30, 0.30, 0.33), roughness=0.7))
    mug = sc.add_entity(
        gs.morphs.Mesh(file=d["obj"], convexify=True, decompose_object_error_threshold=0.04,
                       decimate=False, pos=(0.40, 0.0, table_z + 0.06)),
        material=gs.materials.Rigid(rho=d["density"], friction=1.0),
        surface=gs.surfaces.Plastic(color=(0.90, 0.90, 0.93), roughness=0.45))
    # the BRANCH threaded through the handle ring (along the ring normal +Y), brown
    axis = np.asarray(d["ring_normal"], float); axis /= max(np.linalg.norm(axis), 1e-9)
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(z, axis); s = np.linalg.norm(v); cth = float(np.dot(z, axis))
    if s < 1e-9:
        quat = [1.0, 0.0, 0.0, 0.0] if cth > 0 else [0.0, 1.0, 0.0, 0.0]
    else:
        ang = math.acos(max(-1.0, min(1.0, cth))); ax = v / s
        quat = [math.cos(ang / 2)] + list(math.sin(ang / 2) * ax)
    branch = sc.add_entity(
        gs.morphs.Cylinder(radius=0.004, height=0.22, fixed=True),
        material=gs.materials.Rigid(rho=400.0, friction=0.5),
        surface=gs.surfaces.Plastic(color=(0.45, 0.30, 0.16), roughness=0.8))

    cam = sc.add_sensor(NyxCameraOptions(res=(900, 700), pos=(0.20, -0.34, 0.40),
                        lookat=(0.36, 0.0, 0.31), fov=42, lights=lights,
                        env_maps=tuple(emaps), spp=48, denoise=True))
    sc.build(n_envs=0)

    # settle the mug, then thread the branch through the (now-settled) ring center
    for _ in range(120):
        sc.step()
    mp = mug.get_pos(); mp = mp.cpu().numpy() if hasattr(mp, "cpu") else np.asarray(mp)
    rc = np.asarray(d["ring_center"], float)
    branch.set_pos(np.asarray(mp.reshape(-1)[:3] + rc, np.float32))
    branch.set_quat(np.asarray(quat, np.float32))
    for _ in range(3):
        sc.step()

    cam._stale = True
    img = cam.read().rgb
    img = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    os.makedirs(os.path.dirname(d["out"]), exist_ok=True)
    iio.imwrite(d["out"], img.astype(np.uint8))
    print("RENDER_OK " + d["out"], flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_render":
        _stage_render(sys.argv[2]); sys.exit(0)
    main()
