#!/usr/bin/env python3
"""PHOTOREAL full-DR pick-place collector (Genesis Line B) — RTX-grade Nyx, ONE fully-parallel build.

god-mode scripted cube->bowl demos under FULL domain randomization, all N trials in ONE batched build,
written as the sensor-only fine-tune dataset (LeRobot/§4 HDF5) + a third-person tile + four-view tiles.

FULL DR:
  physics (per env, independent): which ARM (L/R) · object-table height +/-5cm · bowl xy · cube xy + full
    in-plane YAW · cube mass · friction.
  visual  : a DIFFERENT random HDRI environment PER ENV (Nyx per-env env maps -> each trial is in a real
    room: office/bedroom/lounge/bathroom/... , IMMERSIVE — the room is the backdrop AND the light source;
    NO ground plane) · cube/bowl/table colours randomized and distinct (never == the table) per run.

Rendering is REAL (Nyx path tracer + PBR + HDRI IBL). The arm uses the baked silver/orange-carbon livery +
a matte entity-surface override so the metal reads neutral in any room. Collision is RoboLab-faithful
(convex-decomposition bowl -> a cube on the rim rolls in/out, never tunnels; cube spawned clear; gentle
top-down release).

  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/collectors/pickplace_collector.py <N> [seed]
  Knobs: SPP (default 32). DATA_DIR (/data3/genesis_fulldr), OUT_DIR (output/temp/fulldr_collect).
"""
import math
import os
import sys
import time
import glob
import numpy as np
import genesis as gs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                      # genesis_firefly/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "_core_vendored"))
from scenes.firefly_scene import TableLayout, BOWL_HALF_H, firm_rigid_options  # noqa: E402
from scenes.firefly_cameras import (LEFT_WRIST, RIGHT_WRIST, SIDE, WRIST_VFOV, SIDE_VFOV, _T,
                                    add_side_camera_rig)  # noqa: E402
from robots.firefly_dual import (FireflyDual, GR100_OPEN, GR100_CLOSE, GR100_MIMIC)  # noqa: E402
from robots.ik import TOOL_IN_EE_INV, tool_R_at_home  # noqa: E402
from skills.grasp import (world_long_axis, orientation_aware_grasp_quat, tilted_base_quat,
                          transport_quats, _R_from_wxyz, _wxyz_from_R)  # noqa: E402
from object_spec import REGISTRY  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import cv2  # noqa: E402
import h5py  # noqa: E402

RES = (320, 180)                                               # 16:9, all 4 policy/third cams share it
W, H = RES
REC_EVERY = 10                                                 # record state + render every K sim steps
SPP = int(os.environ.get("SPP", "32"))                         # Nyx samples/pixel (denoised; 32 is clean)
BOWL_OBJ = os.path.join(os.path.dirname(_HERE), "assets/objects/ycb/bowl_clean.obj")  # Nyx-safe bowl visual
BG_DIR = "/home/kaitianchao/Projects/RoboLab_firefly/assets/backgrounds"
_HDRS_2K = sorted(glob.glob(f"{BG_DIR}/indoors/*.hdr") + glob.glob(f"{BG_DIR}/outdoors/*.hdr"))
HDR_1K = "/data3/hdr1k"   # Nyx SEGFAULTS past ~50-60 2K env maps; 1K (1/4 memory) lets all 100 fit in one build


def _ensure_1k_pool(n_target=140):
    """Build/return a pool of 1K HDRIs (downsampled from the 2K library). ONLY valid (cv2-readable) ones —
    some 2K .hdr are corrupt/unsupported and must be skipped, else they'd reintroduce 2K maps and crash Nyx."""
    import cv2 as _cv2
    os.makedirs(HDR_1K, exist_ok=True)
    if len(glob.glob(f"{HDR_1K}/*.hdr")) < 100:
        for p in _HDRS_2K:
            if len(glob.glob(f"{HDR_1K}/*.hdr")) >= n_target:
                break
            o = os.path.join(HDR_1K, os.path.basename(p))
            if os.path.exists(o):
                continue
            im = _cv2.imread(p, _cv2.IMREAD_ANYDEPTH | _cv2.IMREAD_COLOR)
            if im is None:
                continue
            _cv2.imwrite(o, _cv2.resize(im, (1024, 512), interpolation=_cv2.INTER_AREA))
    return sorted(glob.glob(f"{HDR_1K}/*.hdr"))


_HDRS = _ensure_1k_pool()   # the per-env background pool (1K, valid-only)
# matte arm override: Nyx applies this entity surface to the whole URDF (color=None -> per-mesh texture
# colours kept), killing the warm-env mirror so the silver livery reads neutral in any room.
ARM_SURF = dict(metallic=0.0, roughness=0.7)
# a soft neutral key so the arm/objects read + contact shadows show; the HDRI does the bulk of the lighting.
LIGHTS = [{"dir": (-0.4, 0.3, -0.85), "color": (1, 1, 1), "intensity": 1.2, "directional": True, "castshadow": True}]


def np_(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


# ================================================================== 1. DOMAIN RANDOMIZATION ================
def pick_colors(rng):
    """One distinct cube/bowl/table colour set for the run (shared across envs — Nyx materials bake at build).
    Cube & bowl are saturated and NEVER ~= the table colour or each other."""
    def hsv_rgb(h, s, v):
        rgb = cv2.cvtColor(np.array([[[h, s * 255, v * 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        return tuple((rgb.astype(np.float32) / 255.0).tolist())

    def huedist(a, b):
        d = abs(a - b); return min(d, 179 - d)

    if rng.rand() < 0.4:
        g = 0.45 + 0.35 * rng.rand(); table = (g, g, g); tab_h, tab_s, tab_v = 0.0, 0.0, g
    else:
        tab_h = rng.rand() * 179; tab_s = 0.30 + 0.35 * rng.rand(); tab_v = 0.55 + 0.30 * rng.rand()
        table = hsv_rgb(tab_h, tab_s, tab_v)
    tab_grey = tab_s < 0.15

    def distinct(*avoid_h):
        for _ in range(80):
            h = rng.rand() * 179; s = 0.60 + 0.35 * rng.rand(); v = 0.75 + 0.22 * rng.rand()
            bad = (abs(v - tab_v) < 0.22) if tab_grey else (huedist(h, tab_h) < 30 and abs(v - tab_v) < 0.22)
            if any(huedist(h, ah) < 25 for ah in avoid_h):
                bad = True
            if not bad:
                return h, hsv_rgb(h, s, v)
        return h, hsv_rgb(h, s, v)

    ch, cube = distinct()
    _, bowl = distinct(ch)
    atable = tuple(0.7 * c for c in table)
    return dict(cube=cube, bowl=bowl, table=table, atable=atable)


def sample_dr(N, rng, lay, spec):
    """Per-env physics DR + per-env HDRI choice (the background of each trial)."""
    ho = lay.object_table_height
    side_is_left = rng.rand(N) < 0.5
    sgn = np.where(side_is_left, 1.0, -1.0)
    tabZ = ho + (rng.rand(N) - 0.5) * 0.10
    bowx = 0.40 + (rng.rand(N) - 0.5) * 0.10
    bowy = sgn * (0.05 + (rng.rand(N) - 0.5) * 0.07)
    cubx = 0.40 + (rng.rand(N) - 0.5) * 0.12
    cuby = sgn * (0.185 + (rng.rand(N) - 0.5) * 0.10)
    for _ in range(40):                                       # a cube can never start inside a bowl in reality
        bad = np.hypot(cubx - bowx, cuby - bowy) < 0.125
        if not bad.any():
            break
        nb = int(bad.sum())
        cubx[bad] = 0.40 + (rng.rand(nb) - 0.5) * 0.12
        cuby[bad] = sgn[bad] * (0.185 + (rng.rand(nb) - 0.5) * 0.10)
    yaw = (rng.rand(N) - 0.5) * np.radians(180)
    mass_shift = ((rng.rand(N, 1) - 0.5) * 0.04).astype(np.float32)
    hdr_idx = rng.choice(len(_HDRS), N, replace=len(_HDRS) < N)   # a different room per env
    return dict(side_is_left=side_is_left, sgn=sgn, tabZ=tabZ, bowx=bowx, bowy=bowy, cubx=cubx, cuby=cuby,
                yaw=yaw, mass_shift=mass_shift, hdrs=[_HDRS[i] for i in hdr_idx])


# ================================================================== 2. ONE FULLY-PARALLEL PHOTOREAL SCENE ==
def build_scene(N, lay, colors, hdr_paths):
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx import nyx_py_sdk as nps
    cap = int(os.environ.get("MAX_ENVMAPS", "0")) or len(hdr_paths)   # cap distinct env maps (memory/limit)
    emaps = []                                                # one (1K) HDRI per env -> per-env backgrounds
    for hp in hdr_paths[:cap]:                                # hp is already a 1K path from the pool
        e = nps.EnvironmentMapAsset(); e.texture = hp; e.layout = nps.EEnvMapLayout.LongLat
        e.multiplier = 1.0; emaps.append(e)
    env_maps = tuple(emaps)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4), rigid_options=firm_rigid_options(),
                     show_viewer=False)
    ha, ho = lay.arm_table_height, lay.object_table_height
    # NO ground plane -> the HDRI room IS the immersive floor+walls backdrop; the two tables are the surfaces.
    robot = FireflyDual(scene, pos=(0, 0, ha), surface=gs.surfaces.Default(**ARM_SURF))   # matte livery
    scene.add_entity(gs.morphs.Box(size=(lay.arm_table_depth, lay.common_width, ha),
                     pos=(lay.seam_x - lay.arm_table_depth / 2, 0, ha / 2), fixed=True, collision=True),
                     surface=gs.surfaces.Plastic(color=colors["atable"], roughness=0.5))
    otab = scene.add_entity(gs.morphs.Box(size=(lay.object_table_depth, lay.common_width, ho),
                            pos=(lay.seam_x + lay.object_table_depth / 2, 0, ho / 2), fixed=True, collision=True),
                            surface=gs.surfaces.Plastic(color=colors["table"], roughness=0.55))
    spec = REGISTRY["cube"]
    cube = scene.add_entity(gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(0.40, 0.18, 0.30)),
                            material=gs.materials.Rigid(rho=600.0, friction=1.0),
                            surface=gs.surfaces.Plastic(color=colors["cube"], roughness=0.35))
    bowl = scene.add_entity(gs.morphs.Mesh(file=BOWL_OBJ, convexify=True,
                            decompose_object_error_threshold=0.04, decimate=False),
                            material=gs.materials.Rigid(rho=400.0, friction=1.0),
                            surface=gs.surfaces.Smooth(color=colors["bowl"]))
    add_side_camera_rig(scene)                                # visible D435i body + support stick

    def link_local(name):
        lk = robot.entity.get_link(name)
        for a in ("idx_local", "_idx_local"):
            if hasattr(lk, a):
                return int(getattr(lk, a))
        return int(lk.idx - robot.entity.link_start)

    eidx = robot.entity.idx
    nc = dict(lights=LIGHTS, env_maps=env_maps, spp=SPP, denoise=True)
    cams = {
        "third": scene.add_sensor(NyxCameraOptions(res=RES, pos=(1.15, -0.95, 0.62), lookat=(0.30, 0.0, 0.34),
                 fov=48, **nc)),                              # low cam -> the room shows behind the arm
        "cam_side": scene.add_sensor(NyxCameraOptions(res=RES, pos=tuple(SIDE[0]), lookat=(0.40, 0.0, 0.28),
                    fov=SIDE_VFOV, **nc)),
        "cam_lw": scene.add_sensor(NyxCameraOptions(res=RES, fov=WRIST_VFOV, entity_idx=eidx,
                  link_idx_local=link_local("left_link_6"), offset_T=_T(*LEFT_WRIST), **nc)),
        "cam_rw": scene.add_sensor(NyxCameraOptions(res=RES, fov=WRIST_VFOV, entity_idx=eidx,
                  link_idx_local=link_local("right_link_6"), offset_T=_T(*RIGHT_WRIST), **nc)),
    }
    scene.build(n_envs=N, env_spacing=(0.0, 0.0))             # spacing 0 -> each env renders its own room
    robot.finalize()
    return scene, robot, dict(otab=otab, cube=cube, bowl=bowl, spec=spec), cams


def render_all(cams):
    """Render the 4 batched Nyx cameras via the SENSOR API (cam.read()) -> {name: (N,H,W,3) uint8}. read()
    re-attaches the wrist cams to link_6 each frame (true egocentric); env_maps make each env its own room."""
    out = {}
    for nm, cam in cams.items():
        cam._stale = True
        out[nm] = np_(cam.read().rgb).astype(np.uint8)
    return out


def _label(img, text):
    im = img.copy()
    cv2.rectangle(im, (0, 0), (len(text) * 7 + 6, 16), (0, 0, 0), -1)
    cv2.putText(im, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def tile(frames):
    N = len(frames); g = int(math.ceil(math.sqrt(N)))
    t = np.zeros((g * H, g * W, 3), np.uint8)
    for i in range(N):
        r, c = divmod(i, g); t[r * H:(r + 1) * H, c * W:(c + 1) * W] = frames[i]
    return t


# ================================================================== 3. COLLECT (single build, all N) ======
def collect(N, seed, data_dir, out_dir):
    rng = np.random.RandomState(seed)
    lay = TableLayout()
    spec = REGISTRY["cube"]
    colors = pick_colors(rng)
    dr = sample_dr(N, rng, lay, spec)
    t0 = time.time()
    scene, robot, ents, cams = build_scene(N, lay, colors, dr["hdrs"])
    cube, bowl, otab = ents["cube"], ents["bowl"], ents["otab"]
    ee = {"l": robot.entity.get_link(robot.ee["left"]), "r": robot.entity.get_link(robot.ee["right"])}
    gdrv, gmim, armdof = robot.grip_driven, robot.grip_mimic, robot.arm
    state14 = robot.state14_idx
    print(f"[COLLECT] built N={N} (one parallel build, {N} environments) in {time.time()-t0:.1f}s", flush=True)

    # ---- apply DR + settle ----
    ho = lay.object_table_height
    tabZ, bowx, bowy, cubx, cuby = dr["tabZ"], dr["bowx"], dr["bowy"], dr["cubx"], dr["cuby"]
    side_is_left, yaw = dr["side_is_left"], dr["yaw"]
    qz = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1).astype(np.float32)
    otab.set_pos(np.stack([np.full(N, lay.seam_x + lay.object_table_depth / 2), np.zeros(N), tabZ - ho / 2], 1).astype(np.float32))
    bowl.set_pos(np.stack([bowx, bowy, tabZ + BOWL_HALF_H + 0.003], 1).astype(np.float32))
    cube_top = tabZ + spec.scaled_extents()[2] / 2 + 0.002
    cube.set_pos(np.stack([cubx, cuby, cube_top + 0.01], 1).astype(np.float32)); cube.set_quat(qz)
    try:
        cube.set_mass_shift(dr["mass_shift"])
        robot.entity.set_friction_ratio((0.7 + 0.6 * rng.rand(N, robot.entity.n_links)).astype(np.float32))
    except Exception as e:
        print(f"[COLLECT] mass/fric DR skipped: {e}", flush=True)
    home16 = np.tile(robot.home_qpos(), (N, 1)).astype(np.float32)
    for _ in range(70):
        robot.entity.control_dofs_position(home16); scene.step()
    root0 = np_(cube.get_pos())

    # ---- per-env grasp + GENTLE top-down place plan (RoboLab skills) ----
    _, heqL = robot.ee_pose("left"); _, heqR = robot.ee_pose("right")
    htR = {"left": tool_R_at_home(_R_from_wxyz(np_(heqL)[0])), "right": tool_R_at_home(_R_from_wxyz(np_(heqR)[0]))}
    base = {"left": np.array([0.0, 0.224]), "right": np.array([0.0, -0.224])}
    laxis = spec.long_axis_local()

    def gquat(i):
        s = "left" if side_is_left[i] else "right"
        yr = ((yaw[i] + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        qzr = np.array([np.cos(yr / 2), 0.0, 0.0, np.sin(yr / 2)])
        return orientation_aware_grasp_quat(world_long_axis(laxis, qzr),
                                            tilted_base_quat(np.array([cubx[i], cuby[i]]) - base[s], 0.0), reference_R=htR[s])

    def cquat(i, gqi):
        s = "left" if side_is_left[i] else "right"
        return transport_quats(tilted_base_quat(np.array([bowx[i], bowy[i]]) - base[s], 8.0), reference_quat=gqi)[0]

    gq = np.stack([gquat(i) for i in range(N)]).astype(np.float64)
    cq = np.stack([cquat(i, gq[i]) for i in range(N)]).astype(np.float64)

    def ik(link, tool_pos, tool_quat):
        Rt = np.stack([_R_from_wxyz(q) for q in tool_quat])
        ee_pos = tool_pos + np.einsum("nij,j->ni", Rt, TOOL_IN_EE_INV[:3, 3])
        ee_R = np.einsum("nij,jk->nik", Rt, TOOL_IN_EE_INV[:3, :3])
        ee_quat = np.stack([_wxyz_from_R(R) for R in ee_R])
        idx = armdof["left"] if link is ee["l"] else armdof["right"]
        q = np_(robot.entity.inverse_kinematics(link=link, pos=ee_pos.astype(np.float32),
                quat=ee_quat.astype(np.float32), dofs_idx_local=idx, max_solver_iters=24, return_error=False))
        return q[:, idx]

    gc = root0.copy(); gc[:, 2] += spec.grasp_dz
    tz = np.stack([_R_from_wxyz(q) @ [0, 0, 1.0] for q in gq])
    drop_z = tabZ + 2 * BOWL_HALF_H + spec.scaled_extents()[2] / 2 + 0.012   # release ABOVE the rim (gentle)
    bxyz = np.stack([bowx, bowy, drop_z], 1).astype(np.float64)
    APP, LIFT, PAPP = 0.12, 0.20, 0.09
    WP = [("pre", gc - APP * tz, gq, GR100_OPEN), ("at", gc, gq, GR100_OPEN), ("close", gc, gq, GR100_CLOSE),
          ("lift", gc + [0, 0, LIFT], cq, GR100_CLOSE), ("carry", bxyz + [0, 0, PAPP], cq, GR100_CLOSE),
          ("lower", bxyz, cq, GR100_CLOSE), ("hold", bxyz, cq, GR100_CLOSE), ("rel", bxyz, cq, GR100_OPEN),
          ("ret", bxyz + [0, 0, PAPP], cq, GR100_OPEN)]
    SEG = [40, 50, 60, 95, 95, 50, 40, 55, 40]
    wpj = [np.where(side_is_left[:, None], ik(ee["l"], p.astype(np.float64), q), ik(ee["r"], p.astype(np.float64), q))
           for (_, p, q, _) in WP]

    def ease(u):
        return 3 * u * u - 2 * u * u * u

    traj, grip, labs = [], [], []
    for i in range(len(WP) - 1):
        for s in range(SEG[i]):
            u = ease((s + 1) / SEG[i])
            traj.append((wpj[i] * (1 - u) + wpj[i + 1] * u).astype(np.float32))
            grip.append(float(WP[i][3] + (WP[i + 1][3] - WP[i][3]) * u)); labs.append(WP[i + 1][0])
    T = len(traj)
    li, ri = np.where(side_is_left)[0], np.where(~side_is_left)[0]
    print(f"[COLLECT] planned T={T} ({len(li)} left / {len(ri)} right arm)", flush=True)

    # ---- execute + RECORD (states/actions + render 4 Nyx cams every REC_EVERY) ----
    acts, jpos, jvel, eepos, eequat = [], [], [], [], []
    cam_steps = {nm: [] for nm in cams}
    lift_pos = None
    t0 = time.time()

    def record_state(full_cmd):
        acts.append(full_cmd[:, state14].copy())
        qp = np_(robot.entity.get_dofs_position()); qv = np_(robot.entity.get_dofs_velocity())
        jpos.append(qp[:, state14].copy()); jvel.append(qv[:, state14].copy())
        ep = np.where(side_is_left[:, None], np_(ee["l"].get_pos()), np_(ee["r"].get_pos()))
        eq = np.where(side_is_left[:, None], np_(ee["l"].get_quat()), np_(ee["r"].get_quat()))
        eepos.append(ep.copy()); eequat.append(eq.copy())

    for t in range(T):
        aq = traj[t]; full = home16.copy()
        for k, d in enumerate(armdof["left"]):
            full[li, d] = aq[li, k]
        for k, d in enumerate(armdof["right"]):
            full[ri, d] = aq[ri, k]
        g = grip[t]
        full[li, gdrv["left"]] = g; full[li, gmim["left"]] = GR100_MIMIC * g
        full[ri, gdrv["right"]] = g; full[ri, gmim["right"]] = GR100_MIMIC * g
        robot.entity.control_dofs_position(full); scene.step()
        if labs[t] == "lift":
            lift_pos = np_(cube.get_pos()).copy()
        if t % REC_EVERY == 0:
            record_state(full)
            views = render_all(cams)
            for nm in cams:
                cam_steps[nm].append(views[nm])
    for _ in range(40):
        robot.entity.control_dofs_position(full); scene.step()
    record_state(full)
    views = render_all(cams)
    for nm in cams:
        cam_steps[nm].append(views[nm])
    if lift_pos is None:
        lift_pos = np_(cube.get_pos())
    wall = time.time() - t0

    # ---- score + realistic penetration check (per-env bowl centre) ----
    objf = np_(cube.get_pos())
    eep = np.where(side_is_left[:, None], np_(ee["l"].get_pos()), np_(ee["r"].get_pos()))
    lift_cm = (lift_pos[:, 2] - root0[:, 2]) * 100
    ch2 = spec.scaled_extents()[2] / 2
    rxy = np.hypot(objf[:, 0] - bowx, objf[:, 1] - bowy)
    rim_z = tabZ + 2 * BOWL_HALF_H
    placed = (lift_cm > 3) & (rxy < 0.06) & (objf[:, 2] - ch2 > tabZ - 0.005) & \
             (objf[:, 2] - ch2 < rim_z + 0.01) & (np.linalg.norm(objf - eep, axis=1) > 0.08)
    bottom = objf[:, 2] - ch2
    wall_pen = (((rxy > 0.065) & (rxy < 0.11) & (bottom > tabZ + 0.012) & (bottom < rim_z)) | (bottom < tabZ - 0.015))
    print(f"[COLLECT] {int((lift_cm>3).sum())}/{N} grasped, {int(placed.sum())}/{N} placed, "
          f"through-wall={int(wall_pen.sum())}/{N}  render+sim {wall:.1f}s", flush=True)

    # ---- write the fine-tune data (HDF5 §4 schema + 3 policy-cam videos) + tiles ----
    os.makedirs(data_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    acts = np.stack(acts, 1); jpos = np.stack(jpos, 1); jvel = np.stack(jvel, 1)   # (N,Tr,14)
    eepos = np.stack(eepos, 1); eequat = np.stack(eequat, 1)
    Tr = acts.shape[1]
    vdir = os.path.join(data_dir, "videos")
    for nm in ("cam_side", "cam_lw", "cam_rw"):
        os.makedirs(os.path.join(vdir, nm), exist_ok=True)
    with h5py.File(os.path.join(data_dir, "demos.hdf5"), "w") as f:
        for e in range(N):
            d = f.create_group(f"data/demo_{e}")
            d.create_dataset("actions", data=acts[e].astype(np.float32))
            sg = d.create_group("states/articulation/robot")
            sg.create_dataset("joint_position", data=jpos[e].astype(np.float32))
            sg.create_dataset("joint_velocity", data=jvel[e].astype(np.float32))
            pg = d.create_group("ee_pose")
            pg.create_dataset("position", data=eepos[e].astype(np.float32))
            pg.create_dataset("orientation", data=eequat[e].astype(np.float32))
            d.attrs["num_samples"] = Tr; d.attrs["success"] = bool(placed[e]); d.attrs["seed"] = int(seed)
            d.attrs["arm"] = "left" if side_is_left[e] else "right"
            d.attrs["hdr"] = os.path.basename(dr["hdrs"][e])
            for nm in ("cam_side", "cam_lw", "cam_rw"):        # the sensor-only policy stream
                vid = np.stack([cam_steps[nm][t][e] for t in range(Tr)])
                iio.imwrite(os.path.join(vdir, nm, f"demo_{e}.mp4"), vid, fps=12, codec="libx264")
    # sqrt(N) third-person tile video
    third_tiles = [tile([cam_steps["third"][t][e] for e in range(N)]) for t in range(Tr)]
    iio.imwrite(os.path.join(out_dir, f"fulldr_third_{N}.mp4"), np.stack(third_tiles), fps=12, codec="libx264")
    # 10 random demos -> labelled 2x2 (third + side + 2 wrist) tiles
    chosen = sorted(rng.choice(N, size=min(10, N), replace=False).tolist())
    lab = {"third": "third", "cam_side": "side", "cam_lw": "wrist-L", "cam_rw": "wrist-R"}
    for e in chosen:
        frames = []
        for t in range(Tr):
            q = {nm: _label(cam_steps[nm][t][e], lab[nm]) for nm in lab}
            frames.append(np.vstack([np.hstack([q["third"], q["cam_side"]]),
                                     np.hstack([q["cam_lw"], q["cam_rw"]])]))
        iio.imwrite(os.path.join(out_dir, f"fourview_demo_{e}.mp4"), np.stack(frames), fps=12, codec="libx264")
    print(f"[COLLECT] wrote {N} demos ({int(placed.sum())} placed) -> {data_dir}/demos.hdf5 + videos ; "
          f"tiles -> {out_dir} (4view demos {chosen})")
    print("COLLECT_DONE")
    return dict(placed=int(placed.sum()), grasped=int((lift_cm > 3).sum()), through_wall=int(wall_pen.sum()))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    DATA = os.environ.get("DATA_DIR", "/data3/genesis_fulldr")
    OUT = os.environ.get("OUT_DIR", "genesis_firefly/output/temp/fulldr_collect")
    gs.init(backend=gs.gpu)
    collect(N, SEED, DATA, OUT)
