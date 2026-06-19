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
RES = (640, 360)   # 16:9; all 4 cameras share it. Short side 360 >= pi0.5's ~224 input (no upscaling) for
#                    crisp wrist-cam grasp detail (owner decision 2026-06-19; was 320x180 for early-bringup speed).
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


# ---- per-build table-TEXTURE DR (wood / steel / tablecloth albedo maps) ------------------------------------
# DR spec (docs/domain_randomization.md, scope A): table texture is PER-BUILD (Nyx bakes albedo at build), a
# choice over a >=10 texture pack incl. several bright tablecloths. The pack lives next to the stage and is
# globbed (so adding/removing a PNG changes the pool with zero code change). Build it with
# `scripts/build_table_textures.py`. The texture is the table's VISUAL only — friction stays per-env + decoupled
# from texture (the task sets table/finger friction independently; this code never touches friction).
_TABLE_TEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "assets", "textures", "tables")
# Per-category physical texel period (metres per ONE full texture repeat) so a small arm table and a big object
# table read at the SAME real-world scale (gingham squares ~2-3cm, wood plank ~1 board across). Net repeats over
# a slab edge of length L = L / period (see rendering_and_livery.md for the Plane uvScale math).
_TEX_PERIOD = {"cloth": 0.34, "wood": 0.55, "steel": 0.60, "metal": 0.60}


def table_texture_pool():
    """The sorted list of table-texture albedo PNGs (the per-build DR pool). Globbed from
    ``assets/textures/tables`` — built by ``scripts/build_table_textures.py`` (>=10 incl. bright tablecloths)."""
    return sorted(glob.glob(f"{_TABLE_TEX_DIR}/*.png") + glob.glob(f"{_TABLE_TEX_DIR}/*.jpg"))


def _tex_period(path):
    """Metres-per-repeat for a texture, keyed by the category prefix in its filename (cloth/wood/steel/metal)."""
    name = os.path.basename(path).lower()
    for key, period in _TEX_PERIOD.items():
        if name.startswith(key):
            return period
    return 0.45                                                  # sensible default for an unlabelled texture


def _texture_mean_rgb(path):
    """The mean RGB (0..1) of a texture image — used to tint the table-Box EDGES so they match the textured
    top instead of clashing. Cheap (downsamples to 64px); failures fall back to neutral grey."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB").resize((64, 64))
        return tuple((np.asarray(im, np.float32).reshape(-1, 3).mean(0) / 255.0).tolist())
    except Exception:
        return (0.5, 0.5, 0.5)


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

        # PER-BUILD table TEXTURE: one albedo map (wood / steel / tablecloth) baked onto the table TOPS this
        # build. BOTH tables share the SAME texture (they are flush at the seam -> one continuous surface reads
        # realistic). The collidable Boxes keep physics; a thin visual-only textured Plane on top maps the
        # texture (a Box has no UVs -> Nyx can't map a texture onto it; a Plane does, see render notes). Falls
        # back to plain colour if the pack is missing.
        pool = table_texture_pool()
        self.table_texture = pool[int(self.rng.randint(len(pool)))] if pool else None
        # side colour for the collidable Box: the texture's MEAN tone (so the table EDGE matches the textured
        # top instead of a clashing bright colour); plain colour-DR is the fallback when no texture is loaded.
        side = _texture_mean_rgb(self.table_texture) if self.table_texture else self.table_color
        self.aside_color = tuple(0.85 * c for c in side)         # arm table edge slightly darker

        # --- scene: NO ground plane -> the HDRI room is the immersive floor+walls; tables are the surfaces ---
        ha, ho = self.lay.arm_table_height, self.lay.object_table_height
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
                              rigid_options=firm_rigid_options(), show_viewer=False)
        self.robot = FireflyDual(self.scene, pos=(0, 0, ha), surface=gs.surfaces.Default(**ARM_SURF))
        # collidable bodies (mean-tone edges; the textured top Plane covers what the cameras mostly see)
        self.atable = self.scene.add_entity(
            gs.morphs.Box(size=(self.lay.arm_table_depth, self.lay.common_width, ha),
                          pos=(self.lay.seam_x - self.lay.arm_table_depth / 2, 0, ha / 2), fixed=True, collision=True),
            surface=gs.surfaces.Plastic(color=self.aside_color, roughness=0.6))
        self.otable = self.scene.add_entity(
            gs.morphs.Box(size=(self.lay.object_table_depth, self.lay.common_width, ho),
                          pos=(self.lay.seam_x + self.lay.object_table_depth / 2, 0, ho / 2), fixed=True, collision=True),
            surface=gs.surfaces.Plastic(color=side, roughness=0.6))
        # textured tops (visual only, no collision). atable is static; otable is moved per-env by the task's
        # height DR -> its top Plane must track it (batch_fixed_verts=True) and the task calls set_otable_top_z.
        self.atable_top, self.otable_top = self._add_table_tops(ha, ho)
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

    def _table_top_surface(self):
        """The surface for a textured table top: the per-build albedo map as a diffuse_texture (matte), or a
        plain coloured fallback when the texture pack is missing. Each call makes its OWN texture instance so
        the two tables don't share a mutable surface object."""
        if not self.table_texture:
            return gs.surfaces.Plastic(color=self.table_color, roughness=0.6)
        return gs.surfaces.Plastic(diffuse_texture=gs.textures.ImageTexture(image_path=self.table_texture),
                                   roughness=0.65)

    def _top_plane(self, depth, width, cx, top_z, batch_fixed):
        """A thin VISUAL-ONLY (no collision) textured Plane covering one table top. `tile_size` is set from the
        texture's physical period so a small arm table and a big object table read at the same real-world scale;
        an isotropic tile (`p,p`) keeps texels square. Sits 0.2mm above the collidable Box top."""
        period = _tex_period(self.table_texture) if self.table_texture else 0.45
        ts = depth * period                                      # net repeats over `depth` = depth/period
        return self.scene.add_entity(
            gs.morphs.Plane(pos=(cx, 0.0, top_z + 2e-4), normal=(0, 0, 1), plane_size=(depth, width),
                            tile_size=(ts, ts), visualization=True, collision=False, fixed=True,
                            batch_fixed_verts=batch_fixed),
            surface=self._table_top_surface())

    def _add_table_tops(self, ha, ho):
        """Textured top Planes for both tables. The arm table is static; the object table is moved per-env by
        the task's height DR, so its top Plane is batched (set_otable_top_z re-glues it after each set_pos)."""
        atop = self._top_plane(self.lay.arm_table_depth, self.lay.common_width,
                               self.lay.seam_x - self.lay.arm_table_depth / 2, ha, batch_fixed=False)
        otop = self._top_plane(self.lay.object_table_depth, self.lay.common_width,
                               self.lay.seam_x + self.lay.object_table_depth / 2, ho, batch_fixed=True)
        return atop, otop

    def set_otable_top_z(self, top_z):
        """Glue the object-table TOP plane to the per-env object-table height. Call right after the task moves
        the object-table Box (``otable.set_pos(... z=tabZ-ho/2)``); `top_z` is the per-env table-TOP height
        (``tabZ``), shape (N,). Keeps the texture flush on the randomized table (no float/sink)."""
        top_z = np_(top_z).reshape(-1).astype(np.float32)
        cx = self.lay.seam_x + self.lay.object_table_depth / 2
        pos = np.stack([np.full_like(top_z, cx), np.zeros_like(top_z), top_z + 2e-4], 1).astype(np.float32)
        self.otable_top.set_pos(pos)

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
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    st = ManipulationStage(n_envs=2, seed=seed)
    st.build(); st.settle_home(50)
    v = st.render()
    iio.imwrite("/tmp/stage_selfcheck.png", v["third"][0])
    print("STAGE_OK third", v["third"].shape,
          "table_texture:", os.path.basename(st.table_texture) if st.table_texture else None,
          "rooms:", [os.path.basename(h) for h in st.hdrs])
