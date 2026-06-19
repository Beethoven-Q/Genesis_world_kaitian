# SPDX-License-Identifier: Apache-2.0
"""ManipulationStage — the reusable robot-manipulation + photoreal-rendering SETUP, isolated from any task.

This is the "world" every manipulation task runs in. Build it once; tasks add their own objects + skills.
It owns everything that should NEVER need re-tuning per task:

  - ROBOT      : the dual Firefly Y6 + GR100 arm (baked SOMA livery, matte override, MIT gains, and the
                 convex-DECOMPOSED collision so fingers/links hug their visual meshes — no penetration).
  - TABLES     : the two static collidable tables (arm table + object table) from TableLayout.
  - CAMERAS    : the 3 policy cameras (2 egocentric wrist D405 + 1 world-fixed side D435i) at calibrated
                 poses/FOV, plus the visible side-camera body + support stick. All 4 are Nyx sensors.
  - RENDERING  : Nyx path-traced PBR. Per-env HDRI environment-map DR -> each of the N parallel envs renders
                 in its OWN random real room (immersive: no ground plane, the HDRI is the floor+walls+light).
  - ENV DR     : a 1K HDRI pool (Nyx segfaults past ~50-60 2K env maps; 1K fits 100), randomized table colours.

A TASK does only:  stage = ManipulationStage(n_envs); <add objects to stage.scene>; stage.build();
                   stage.settle_home(); <plan/execute>; stage.render(); <score + write>.

Per-env physics DR (object poses, mass, friction, table height) and the OBJECTS belong to the task, not here.

Quick self-check (renders a third-person frame to /tmp):
  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python genesis_firefly/scenes/manipulation_stage.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # genesis_firefly/
from world.firefly_scene import TableLayout, firm_rigid_options  # noqa: E402
from world.firefly_cameras import (LEFT_WRIST, RIGHT_WRIST, SIDE, WRIST_VFOV, SIDE_VFOV, _T,
                                    add_side_camera_rig)  # noqa: E402
from robots.firefly_dual import FireflyDual  # noqa: E402

# ---- rendering defaults -----------------------------------------------------------------------------------
RES = (320, 180)                                              # 16:9; all 4 cameras share it
SPP = int(os.environ.get("SPP", "32"))                       # Nyx samples/pixel (denoised; 32 is clean)
# matte entity-surface override: Nyx applies this to the whole URDF (color=None -> per-mesh texture colours
# kept), killing the warm-env mirror so the silver livery reads NEUTRAL in any room.
ARM_SURF = dict(metallic=0.0, roughness=0.7)
# a soft neutral key so the arm/objects read + contact shadows show; the HDRI does the bulk of the lighting.
LIGHTS = [{"dir": (-0.4, 0.3, -0.85), "color": (1, 1, 1), "intensity": 1.2, "directional": True, "castshadow": True}]

# ---- per-env HDRI background pool (the environment domain randomization) -----------------------------------
_BG_DIR = "/home/kaitianchao/Projects/RoboLab_firefly/assets/backgrounds"
_HDRS_2K = sorted(glob.glob(f"{_BG_DIR}/indoors/*.hdr") + glob.glob(f"{_BG_DIR}/outdoors/*.hdr"))
HDR_1K = "/data3/hdr1k"
MAX_2K_ENVS = 45   # Nyx fits ~50-60 2K env maps before segfault; <= this -> original 2K, else -> 1K pool


def hdr_pool(n_target=140):
    """A pool of 1K HDRIs (downsampled from the 2K library, valid-only). Nyx SEGFAULTS past ~50-60 2K env
    maps; 1K (1/4 memory) lets all 100 per-env env maps fit in one build. Corrupt 2K .hdr are skipped (else
    they'd reintroduce a 2K map and crash). Cached under /data3/hdr1k; built lazily on first call."""
    import cv2
    os.makedirs(HDR_1K, exist_ok=True)
    if len(glob.glob(f"{HDR_1K}/*.hdr")) < 100:
        for p in _HDRS_2K:
            if len(glob.glob(f"{HDR_1K}/*.hdr")) >= n_target:
                break
            o = os.path.join(HDR_1K, os.path.basename(p))
            if os.path.exists(o):
                continue
            im = cv2.imread(p, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if im is not None:
                cv2.imwrite(o, cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA))
    return sorted(glob.glob(f"{HDR_1K}/*.hdr"))


def valid_2k_pool():
    """The ORIGINAL 2K HDRIs, filtered to those that VALIDATE — a malformed .hdr that cv2 can't read (and that
    SEGFAULTS Nyx when loaded as an env map) has no 1K cache entry, so we exclude it. Keeps full 2K resolution
    for build-batches (few env maps) while dropping the bad file(s). Builds the validation cache if missing."""
    ok = {os.path.basename(p) for p in glob.glob(f"{HDR_1K}/*.hdr")}
    if not ok:                                                 # cache not built yet -> build it (validates)
        hdr_pool()
        ok = {os.path.basename(p) for p in glob.glob(f"{HDR_1K}/*.hdr")}
    valid = [p for p in _HDRS_2K if os.path.basename(p) in ok]
    return valid or _HDRS_2K


# ---- table-colour DR (the environment visual DR the stage owns) -------------------------------------------
def _hsv_rgb(rng, h, s, v):
    import cv2
    rgb = cv2.cvtColor(np.array([[[h, s * 255, v * 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple((rgb.astype(np.float32) / 255.0).tolist())


def _huedist(a, b):
    d = abs(a - b)
    return min(d, 179 - d)


def np_(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


class ManipulationStage:
    """The reusable manipulation world. Construct, let the task add objects to ``self.scene``, then build."""

    def __init__(self, n_envs, seed=0, res=RES, spp=SPP):
        self.n_envs = int(n_envs)
        self.res = res
        self.W, self.H = res
        self.rng = np.random.RandomState(seed)
        self.lay = TableLayout()
        self._built = False

        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
        from gs_nyx import nyx_py_sdk as nps

        # per-env background: a different HDRI per env (immersive room + image-based light). Restore the
        # original 2K HDRIs when the build has few envs (build-batches); fall back to the 1K pool only for a
        # single large build (Nyx SEGFAULTS past ~50-60 2K env maps; 1K fits 100).
        pool = valid_2k_pool() if (self.n_envs <= MAX_2K_ENVS and not os.environ.get("HDR1K")) else hdr_pool()
        self.hdrs = [pool[i] for i in self.rng.choice(len(pool), self.n_envs, replace=len(pool) < self.n_envs)]
        emaps = []
        for hp in self.hdrs:
            e = nps.EnvironmentMapAsset(); e.texture = hp; e.layout = nps.EEnvMapLayout.LongLat
            e.multiplier = 1.0; emaps.append(e)
        env_maps = tuple(emaps)

        # table colours (environment visual DR); the task picks object colours distinct from self.table_color
        self.table_color, self._tab_hsv = self._pick_table()
        self.atable_color = tuple(0.7 * c for c in self.table_color)

        # --- scene: NO ground plane -> the HDRI room is the immersive floor+walls; tables are the surfaces ---
        ha, ho = self.lay.arm_table_height, self.lay.object_table_height
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
                              rigid_options=firm_rigid_options(), show_viewer=False)
        self.robot = FireflyDual(self.scene, pos=(0, 0, ha), surface=gs.surfaces.Default(**ARM_SURF))
        self.atable = self.scene.add_entity(
            gs.morphs.Box(size=(self.lay.arm_table_depth, self.lay.common_width, ha),
                          pos=(self.lay.seam_x - self.lay.arm_table_depth / 2, 0, ha / 2), fixed=True, collision=True),
            surface=gs.surfaces.Plastic(color=self.atable_color, roughness=0.5))
        self.otable = self.scene.add_entity(
            gs.morphs.Box(size=(self.lay.object_table_depth, self.lay.common_width, ho),
                          pos=(self.lay.seam_x + self.lay.object_table_depth / 2, 0, ho / 2), fixed=True, collision=True),
            surface=gs.surfaces.Plastic(color=self.table_color, roughness=0.55))
        add_side_camera_rig(self.scene)                      # visible D435i body + support stick

        eidx = self.robot.entity.idx
        nc = dict(lights=LIGHTS, env_maps=env_maps, spp=spp, denoise=True)
        self.cams = {
            "third": self.scene.add_sensor(NyxCameraOptions(res=res, pos=(1.15, -0.95, 0.62),
                     lookat=(0.30, 0.0, 0.34), fov=48, **nc)),       # low cam -> the room shows behind the arm
            "cam_side": self.scene.add_sensor(NyxCameraOptions(res=res, pos=tuple(SIDE[0]), lookat=(0.40, 0.0, 0.28),
                        fov=SIDE_VFOV, **nc)),
            # wrist cams sit ~5cm from the gripper -> the default 0.1m near plane CLIPS the near finger
            # geometry (looked transparent). near=0.01 so the close fingers render solid. (side/third keep 0.1)
            "cam_lw": self.scene.add_sensor(NyxCameraOptions(res=res, fov=WRIST_VFOV, near=0.01, entity_idx=eidx,
                      link_idx_local=self._link_local("left_link_6"), offset_T=_T(*LEFT_WRIST), **nc)),
            "cam_rw": self.scene.add_sensor(NyxCameraOptions(res=res, fov=WRIST_VFOV, near=0.01, entity_idx=eidx,
                      link_idx_local=self._link_local("right_link_6"), offset_T=_T(*RIGHT_WRIST), **nc)),
        }

    # -- helpers ------------------------------------------------------------------------------------------
    def _link_local(self, name):
        lk = self.robot.entity.get_link(name)
        for a in ("idx_local", "_idx_local"):
            if hasattr(lk, a):
                return int(getattr(lk, a))
        return int(lk.idx - self.robot.entity.link_start)

    def _pick_table(self):
        rng = self.rng
        if rng.rand() < 0.4:
            g = 0.45 + 0.35 * rng.rand()
            return (g, g, g), (0.0, 0.0, g)
        h = rng.rand() * 179; s = 0.30 + 0.35 * rng.rand(); v = 0.55 + 0.30 * rng.rand()
        return _hsv_rgb(rng, h, s, v), (h, s, v)

    def distinct_object_color(self, *avoid_h):
        """A saturated colour for a task object that is NEVER ~= the table colour (or the given hues)."""
        rng = self.rng
        tab_h, tab_s, tab_v = self._tab_hsv
        tab_grey = tab_s < 0.15
        for _ in range(80):
            h = rng.rand() * 179; s = 0.60 + 0.35 * rng.rand(); v = 0.75 + 0.22 * rng.rand()
            bad = (abs(v - tab_v) < 0.22) if tab_grey else (_huedist(h, tab_h) < 30 and abs(v - tab_v) < 0.22)
            if any(_huedist(h, ah) < 25 for ah in avoid_h):
                bad = True
            if not bad:
                return h, _hsv_rgb(rng, h, s, v)
        return h, _hsv_rgb(rng, h, s, v)

    # -- lifecycle ----------------------------------------------------------------------------------------
    def build(self):
        """Build the batched scene (call AFTER the task has added its objects to ``self.scene``)."""
        self.scene.build(n_envs=self.n_envs, env_spacing=(0.0, 0.0))   # spacing 0 -> each env renders its room
        self.robot.finalize()
        self._built = True
        return self

    def settle_home(self, steps=70):
        """Hold the arms at the home pose for `steps` so the scene settles. Returns the home command (N,n_dofs)."""
        home = np.tile(self.robot.home_qpos(), (self.n_envs, 1)).astype(np.float32)
        for _ in range(steps):
            self.robot.entity.control_dofs_position(home); self.scene.step()
        return home

    def render(self):
        """Render the 4 batched Nyx cameras via the SENSOR API -> {name: (N,H,W,3) uint8}. read() re-attaches
        the wrist cams to link_6 each frame (true egocentric); per-env env maps make each env its own room."""
        out = {}
        for nm, cam in self.cams.items():
            cam._stale = True
            out[nm] = np_(cam.read().rgb).astype(np.uint8)
        return out


if __name__ == "__main__":
    import imageio.v3 as iio
    gs.init(backend=gs.gpu)
    st = ManipulationStage(n_envs=2, seed=1)
    st.build(); st.settle_home(50)
    v = st.render()
    iio.imwrite("/tmp/stage_selfcheck.png", v["third"][0])
    print("STAGE_OK third", v["third"].shape, "rooms:", [os.path.basename(h) for h in st.hdrs])
