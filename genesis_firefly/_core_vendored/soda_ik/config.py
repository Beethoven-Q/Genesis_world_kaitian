# SPDX-License-Identifier: Apache-2.0
"""Config for the soda-bimanual IK bridge (isolated; self-contained in genesis_firefly, no robolab dep)."""
from pathlib import Path

# soda-bimanual checkout (the low-level engine — referenced, NEVER modified).
SODA_BIMANUAL_ROOT = "/home/kaitianchao/Projects/soda-bimanual"
# 6-DOF arm + GR100-mass-lumped URDF (the IK chain). NOT the dual y6_gr100.urdf (movable gripper / sim model).
# Self-contained copy staged in this repo's assets (path resolved relative to this file, no robolab.constants).
_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robots" / "firefly_y6_gr100"
FIREFLY_IK_URDF = str(_ASSETS / "source" / "firefly_y6" / "gr100.urdf")
TCP_OFFSET = 0.187                         # *_ee_link = 0.187 m along +Z of gripper_base (build_spec.md)
ARM_DOF = 6
# Arm base world positions (identity rotation). Z is the arm-table height (TableLayout), so IK never
# desyncs from the rendered robot. World<->base is a pure translation.
BASE_Y = {"left": 0.224, "right": -0.224}
ARM_HOME = [0.0, -0.75, 2.3, 0.9, 0.0, 0.0]
