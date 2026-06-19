# SPDX-License-Identifier: Apache-2.0
"""Firefly Y6 dual-arm + GR100 gripper — Genesis loader (Line B reproduction of RoboLab's firefly_dual.py).

Genesis counterpart of `robolab/robots/firefly_dual.py`. Loads the SAME self-contained dual URDF, then maps
joint NAMES -> Genesis dof indices (mandatory: Genesis INTERLEAVES the dual-arm dofs, e.g. left_joint_1=dof0,
right_joint_1=dof1, ... so the layout is NOT contiguous), exposes the 14-D policy layout, the MIT-controller
PD gains, the GR100 open/close/mimic constants (gripper coupled at the ACTION layer, not URDF mimic), and a
thin command helper. Pure Genesis API — no Isaac. Constants are reproduced from the RoboLab build spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import genesis as gs

# --- asset (self-contained: mesh paths rewritten relative to the URDF, no RoboLab dependency) ---
# Default to the LIVERY urdf: its <visual> meshes are GLBs with the real Firefly Y6 SOMA materials baked in
# (silver metallic links, grey link_4, dark wrist/base, orange-carbon + silver-SOMA panels on link_2/3) so
# Nyx (which reads a URDF's embedded mesh materials) renders the true livery. Built by
# scripts/bake_firefly_livery.py from RoboLab's recolor mapping. Falls back to the plain URDF if not baked.
ASSET = Path(__file__).resolve().parents[1] / "assets/robots/firefly_y6_gr100"
_PLAIN_URDF = str(ASSET / "dual_firefly_y6_gr100.urdf")
_LIVERY_URDF = str(ASSET / "dual_firefly_y6_gr100_livery.urdf")
DUAL_URDF = _LIVERY_URDF if Path(_LIVERY_URDF).exists() else _PLAIN_URDF

# --- joint names (verified loaded; match RoboLab exactly) ---
LEFT_ARM_JOINTS = [f"left_joint_{i}" for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"right_joint_{i}" for i in range(1, 7)]
LEFT_GRIPPER_DRIVEN_JOINT = "left_gripper_left_joint_1"
LEFT_GRIPPER_MIMIC_JOINT = "left_gripper_right_joint_1"
RIGHT_GRIPPER_DRIVEN_JOINT = "right_gripper_left_joint_1"
RIGHT_GRIPPER_MIMIC_JOINT = "right_gripper_right_joint_1"
LEFT_EE_LINK = "left_ee_link"
RIGHT_EE_LINK = "right_ee_link"
LEFT_LINK6 = "left_link_6"
RIGHT_LINK6 = "right_link_6"

# 14-D policy/state layout (6 arm + 1 driven gripper) x 2 arms — the mimic joints are NOT in the state.
STATE_JOINTS_14 = (LEFT_ARM_JOINTS + [LEFT_GRIPPER_DRIVEN_JOINT]
                   + RIGHT_ARM_JOINTS + [RIGHT_GRIPPER_DRIVEN_JOINT])

# --- gripper constants (MEASURED, from RoboLab) ---
GR100_OPEN = 0.0
GR100_CLOSE = 0.9          # firm pinch COMMAND (PD presses hard toward this)
GR100_MIMIC = -1.0         # right_claw = -1 * driven (URDF axis flipped to match MuJoCo); coupled in action
GR100_MEET = 0.58          # the fingers' pads just MEET here (measured: 0.1mm gap at q=0.59). Hard mechanical
#                            close stop: the real GR100 cannot close past pad-contact, and Genesis's soft
#                            joint-limit/self-contact won't hold the thin scissoring fingers under the firm
#                            close -> we clamp the gripper dofs to +/-GR100_MEET each step so the fingers can
#                            NEVER over-rotate/cross at ANY grip force (object contact stops them earlier when
#                            grasping). The firm pinch force still comes from the PD pressing toward GR100_CLOSE.

# --- home pose (per arm) + base height ---
FIREFLY_HOME = [0.0, -0.75, 2.3, 0.9, 0.0, 0.0]
BASE_Z = 0.25              # arm-table height; dual base offsets (LEFT y=+0.224, RIGHT y=-0.224) are in the URDF

# --- PD gains from the MuJoCo MIT controller, IDENTICAL to RoboLab_firefly (ImplicitActuatorCfg). Do NOT
# retune: the earlier wrist jitter was NOT the gains — it was (a) Genesis's default armature 0.1 (vs RoboLab's
# 0.01) over-inertiaing the wrist, and (b) the approximate integrator under-damping the implicit PD. Both are
# fixed faithfully (ARMATURE=0.01 below + integrator=implicitfast in the scene), so kp/kv stay at RoboLab's. ---
ARM_KP = [200, 200, 200, 75, 15, 15]          # J1-3 proximal, J4-6 wrist
ARM_KV = [12.5, 12.5, 12.5, 6.0, 0.31, 0.31]
ARM_EFFORT = [28, 28, 28, 10, 10, 10]
ARMATURE = 0.01                                # reflected motor inertia on every joint (RoboLab uses 0.01)
GRIP_KP, GRIP_KV, GRIP_EFFORT = 200.0, 8.0, 10.0   # firm squeeze (verified: raising effort HURT stability)


class FireflyDual:
    """Loads the dual-arm robot into a Genesis scene and exposes name->dof maps + a command helper."""

    def __init__(self, scene: "gs.Scene", pos=(0.0, 0.0, BASE_Z), surface=None, vis_mode=None, **morph_kwargs):
        # COLLISION FIDELITY (root fix for finger<->object penetration): convexify=True alone gives each link a
        # SINGLE convex hull, and Genesis defaults `decompose_robot_error_threshold=inf` -> the curved GR100
        # finger's hull FILLS its concavity, so the (real) visual finger can poke through an object the (fatter,
        # wrong-shape) hull never contacts. Instead we convex-DECOMPOSE every robot mesh (coacd) to a FINITE
        # error so each collider faithfully HUGS its visual mesh -> the solid finger contacts the solid object
        # exactly where it looks like it does, and the stiff Newton solver then refuses to interpenetrate at any
        # grip force (owner's principle). Collision = the real L1/L2 finger mesh (no inflated-hull hack).
        kw = dict(file=DUAL_URDF, fixed=True, merge_fixed_links=False, pos=pos,
                  convexify=True, decompose_robot_error_threshold=0.05,
                  default_armature=ARMATURE)   # match RoboLab's 0.01 (Genesis default 0.1 over-damps -> jitter)
        kw.update(morph_kwargs)
        # gravity_compensation=1.0: the controller cancels the arm's OWN weight (computed-torque control, like
        # a real well-tuned arm), so the PD-tracked trajectory REACHES its commanded pose instead of sagging
        # ~2-3cm short -> the gripper closes CENTRED on the object. The grasped object is a separate entity
        # (no compensation) so it still falls under gravity and must be physically held by the grip.
        add_kw = dict(morph=gs.morphs.URDF(**kw),
                      material=gs.materials.Rigid(gravity_compensation=1.0))
        if surface is not None:
            add_kw["surface"] = surface
        if vis_mode is not None:                                  # vis_mode="collision" -> render the colliders
            add_kw["vis_mode"] = vis_mode
        self.entity = scene.add_entity(**add_kw)
        self._built = False

    def _didx(self, jname: str) -> int:
        return int(self.entity.get_joint(jname).dofs_idx_local[0])

    def finalize(self):
        """Call AFTER scene.build(): resolve dof indices + push PD gains/effort limits."""
        self.arm = {"left": [self._didx(n) for n in LEFT_ARM_JOINTS],
                    "right": [self._didx(n) for n in RIGHT_ARM_JOINTS]}
        self.grip_driven = {"left": self._didx(LEFT_GRIPPER_DRIVEN_JOINT),
                            "right": self._didx(RIGHT_GRIPPER_DRIVEN_JOINT)}
        self.grip_mimic = {"left": self._didx(LEFT_GRIPPER_MIMIC_JOINT),
                          "right": self._didx(RIGHT_GRIPPER_MIMIC_JOINT)}
        self.state14_idx = [self._didx(n) for n in STATE_JOINTS_14]
        self.ee = {"left": LEFT_EE_LINK, "right": RIGHT_EE_LINK}
        # gains, laid out per the resolved dof order
        n = self.entity.n_dofs
        kp = np.zeros(n, np.float32); kv = np.zeros(n, np.float32); ef = np.zeros(n, np.float32)
        for side in ("left", "right"):
            for k, d in enumerate(self.arm[side]):
                kp[d], kv[d], ef[d] = ARM_KP[k], ARM_KV[k], ARM_EFFORT[k]
            for d in (self.grip_driven[side], self.grip_mimic[side]):
                kp[d], kv[d], ef[d] = GRIP_KP, GRIP_KV, GRIP_EFFORT
        self.entity.set_dofs_kp(kp); self.entity.set_dofs_kv(kv)
        self.entity.set_dofs_force_range(-ef, ef)
        self._built = True
        return self

    def home_qpos(self) -> np.ndarray:
        """Full dof vector at the home pose (arms = FIREFLY_HOME, grippers = open)."""
        q = np.zeros(self.entity.n_dofs, np.float32)
        for side in ("left", "right"):
            for d, v in zip(self.arm[side], FIREFLY_HOME):
                q[d] = v
            q[self.grip_driven[side]] = GR100_OPEN
            q[self.grip_mimic[side]] = GR100_MIMIC * GR100_OPEN
        return q

    def command(self, side: str, arm_q, grip: float):
        """PD-target the 6 arm joints + the coupled gripper (driven=grip, mimic=GR100_MIMIC*grip)."""
        self.entity.control_dofs_position(np.asarray(arm_q, np.float32), self.arm[side])
        self.entity.control_dofs_position(
            np.asarray([grip, GR100_MIMIC * grip], np.float32),
            [self.grip_driven[side], self.grip_mimic[side]])

    def ee_pose(self, side: str):
        """world (pos[3], quat_wxyz[4]) of the ee_link, as numpy (Genesis returns CUDA tensors)."""
        lk = self.entity.get_link(self.ee[side])
        return lk.get_pos().cpu().numpy(), lk.get_quat().cpu().numpy()

    def clamp_gripper(self):
        """Hard mechanical close stop — call AFTER every scene.step(). Genesis's soft joint-limit and the
        thin scissoring fingers' self-contact won't hold under the firm PD close, so we clamp the gripper
        dofs to +/-GR100_MEET so the fingers can NEVER over-close/cross at any grip force. Object contact
        stops them EARLIER when grasping (so this only engages on an empty close)."""
        q = self.entity.get_dofs_position()
        q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
        for side in ("left", "right"):
            d, m = self.grip_driven[side], self.grip_mimic[side]
            if q[d] > GR100_MEET:
                self.entity.set_dofs_position(np.array([GR100_MEET], np.float32), [d], zero_velocity=True)
            if q[m] < -GR100_MEET:
                self.entity.set_dofs_position(np.array([-GR100_MEET], np.float32), [m], zero_velocity=True)
