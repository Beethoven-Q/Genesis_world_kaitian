#!/usr/bin/env python3
"""Cube->bowl PICK-PLACE task collector — a THIN task on top of the reusable ManipulationStage.

The whole robot + cameras + photoreal rendering + per-env immersive HDRI background DR lives in
`scenes/manipulation_stage.py` and is shared by every task. THIS file is only the TASK:
  - objects      : a coloured cube + a convex-decomposition bowl (RoboLab-faithful collision -> a cube on the
                   rim rolls in/out, never tunnels; cube spawned CLEAR of the bowl).
  - physics DR   : which ARM (L/R) · object-table height +/-5cm · bowl xy · cube xy + full in-plane YAW ·
                   cube mass · friction (per env, independent).
  - skill        : orientation-aware grasp + gentle top-down place (reused RoboLab skills + SODA IK).
  - output       : the §4 HDF5 (14-D actions/states/ee_pose) + the 3 policy-cam videos per demo, plus a
                   sqrt(N) third-person tile + 10 four-view tiles.

All N trials run in ONE fully-parallel build (stage.build()), each in its own random real room.

  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/collectors/pickplace_collector.py <N> [seed]
  DATA_DIR (/data3/genesis_fulldr), OUT_DIR (output/temp/fulldr_collect), SPP (32).
"""
import math
import os
import sys
import time
import numpy as np
import genesis as gs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                      # genesis_firefly/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "_core_vendored"))
from scenes.manipulation_stage import ManipulationStage, np_   # the reusable setup  # noqa: E402
from scenes.firefly_scene import BOWL_HALF_H  # noqa: E402
from robots.firefly_dual import GR100_OPEN, GR100_CLOSE, GR100_MIMIC  # noqa: E402
from robots.ik import TOOL_IN_EE_INV, tool_R_at_home  # noqa: E402
from skills.grasp import (world_long_axis, orientation_aware_grasp_quat, tilted_base_quat,
                          transport_quats, _R_from_wxyz, _wxyz_from_R)  # noqa: E402
from object_spec import REGISTRY  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import cv2  # noqa: E402
import h5py  # noqa: E402

REC_EVERY = 10                                                 # record state + render every K sim steps
BOWL_OBJ = os.path.join(os.path.dirname(_HERE), "assets/objects/ycb/bowl_clean.obj")  # Nyx-safe bowl visual


def sample_phys_dr(N, rng, lay, spec):
    """Per-env physics DR (no visuals — the stage owns the environment/background DR)."""
    ho = lay.object_table_height
    side_is_left = rng.rand(N) < 0.5
    sgn = np.where(side_is_left, 1.0, -1.0)
    tabZ = ho + (rng.rand(N) - 0.5) * 0.10                     # object-table height +/-5cm
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
    return dict(side_is_left=side_is_left, sgn=sgn, tabZ=tabZ, bowx=bowx, bowy=bowy,
                cubx=cubx, cuby=cuby, yaw=yaw, mass_shift=mass_shift)


def _label(img, text):
    im = img.copy()
    cv2.rectangle(im, (0, 0), (len(text) * 7 + 6, 16), (0, 0, 0), -1)
    cv2.putText(im, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def tile(frames, H, W):
    N = len(frames); g = int(math.ceil(math.sqrt(N)))
    t = np.zeros((g * H, g * W, 3), np.uint8)
    for i in range(N):
        r, c = divmod(i, g); t[r * H:(r + 1) * H, c * W:(c + 1) * W] = frames[i]
    return t


def collect(N, seed, data_dir, out_dir):
    t0 = time.time()
    stage = ManipulationStage(N, seed=seed)                   # the reusable robot+cameras+rendering+env-DR setup
    lay, spec, rng = stage.lay, REGISTRY["cube"], stage.rng
    dr = sample_phys_dr(N, rng, lay, spec)

    # --- task objects: a coloured cube + a convex-decomposition bowl, distinct from the table colour ---
    ch, cube_col = stage.distinct_object_color()
    _, bowl_col = stage.distinct_object_color(ch)
    cube = stage.scene.add_entity(gs.morphs.Box(size=tuple(spec.scaled_extents()), pos=(0.40, 0.18, 0.30)),
                                  material=gs.materials.Rigid(rho=600.0, friction=1.0),
                                  surface=gs.surfaces.Plastic(color=cube_col, roughness=0.35))
    bowl = stage.scene.add_entity(gs.morphs.Mesh(file=BOWL_OBJ, convexify=True,
                                  decompose_object_error_threshold=0.04, decimate=False),
                                  material=gs.materials.Rigid(rho=400.0, friction=1.0),
                                  surface=gs.surfaces.Smooth(color=bowl_col))
    stage.build()
    robot, cams = stage.robot, stage.cams
    H, W = stage.H, stage.W
    ee = {"l": robot.entity.get_link(robot.ee["left"]), "r": robot.entity.get_link(robot.ee["right"])}
    gdrv, gmim, armdof, state14 = robot.grip_driven, robot.grip_mimic, robot.arm, robot.state14_idx
    print(f"[COLLECT] built N={N} (one parallel build, {N} environments) in {time.time()-t0:.1f}s", flush=True)

    # ---- apply DR + settle ----
    ho = lay.object_table_height
    tabZ, bowx, bowy, cubx, cuby = dr["tabZ"], dr["bowx"], dr["bowy"], dr["cubx"], dr["cuby"]
    side_is_left, yaw = dr["side_is_left"], dr["yaw"]
    qz = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1).astype(np.float32)
    stage.otable.set_pos(np.stack([np.full(N, lay.seam_x + lay.object_table_depth / 2), np.zeros(N), tabZ - ho / 2], 1).astype(np.float32))
    bowl.set_pos(np.stack([bowx, bowy, tabZ + BOWL_HALF_H + 0.003], 1).astype(np.float32))
    cube_top = tabZ + spec.scaled_extents()[2] / 2 + 0.002
    cube.set_pos(np.stack([cubx, cuby, cube_top + 0.01], 1).astype(np.float32)); cube.set_quat(qz)
    try:
        cube.set_mass_shift(dr["mass_shift"])
        robot.entity.set_friction_ratio((0.7 + 0.6 * rng.rand(N, robot.entity.n_links)).astype(np.float32))
    except Exception as e:
        print(f"[COLLECT] mass/fric DR skipped: {e}", flush=True)
    home16 = stage.settle_home(70)
    root0 = np_(cube.get_pos())

    # ---- per-env grasp + GENTLE top-down place plan (RoboLab skills + SODA IK) ----
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
        robot.entity.control_dofs_position(full); stage.scene.step()
        if labs[t] == "lift":
            lift_pos = np_(cube.get_pos()).copy()
        if t % REC_EVERY == 0:
            record_state(full)
            views = stage.render()
            for nm in cams:
                cam_steps[nm].append(views[nm])
    for _ in range(40):
        robot.entity.control_dofs_position(full); stage.scene.step()
    record_state(full)
    views = stage.render()
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
            d.attrs["hdr"] = os.path.basename(stage.hdrs[e])
            for nm in ("cam_side", "cam_lw", "cam_rw"):        # the sensor-only policy stream
                vid = np.stack([cam_steps[nm][t][e] for t in range(Tr)])
                iio.imwrite(os.path.join(vdir, nm, f"demo_{e}.mp4"), vid, fps=12, codec="libx264")
    third_tiles = [tile([cam_steps["third"][t][e] for e in range(N)], H, W) for t in range(Tr)]
    iio.imwrite(os.path.join(out_dir, f"fulldr_third_{N}.mp4"), np.stack(third_tiles), fps=12, codec="libx264")
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
