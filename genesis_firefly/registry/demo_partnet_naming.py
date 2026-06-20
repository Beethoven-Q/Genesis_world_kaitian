# SPDX-License-Identifier: Apache-2.0
"""DEMO: PartNet-Mobility SEMANTIC LINK NAMING — give a raw-index articulated URDF (link_0/link_1, joint_0/...)
CORRECT semantic names from GEOMETRY + JOINT KINEMATICS, VERIFIED against PartNet's ground-truth semantics.txt
(never hallucinated). Plus a tiny SYNTHETIC stapler URDF unit-test proving the function generalizes.

LIVE object: PartNet-Mobility 3763 (model_cat "Bottle") — copied into
  genesis_firefly/assets/objects/partnet_mobility/3763_bottle/ (mobility.urdf + semantics.txt + meshes).
  Ground truth (semantics.txt): link_0 = rotation_lid (a slider+ lid), link_1 = bottle_body (free base).
  Our function must independently derive link_0->cap/lid (small, movable, on top) and link_1->bottle_body
  (largest, fixed base) from geometry+kinematics, then we CHECK that against the ground truth.

Run:  CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python genesis_firefly/registry/demo_partnet_naming.py
Writes a multi-view inspection render of the bottle to genesis_firefly/output/temp/bottle_partnet_views.png
(evidence the agent/owner can open) + prints the verified semantic map.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from registry.refine import semantic_label_links, render_collider_views  # noqa: E402

BOTTLE_DIR = os.path.join(_PKG, "assets", "objects", "partnet_mobility", "3763_bottle")
BOTTLE_URDF = os.path.join(BOTTLE_DIR, "mobility.urdf")
BOTTLE_TRUTH = os.path.join(BOTTLE_DIR, "semantics.txt")
BOTTLE_META = os.path.join(BOTTLE_DIR, "meta.json")
VIEWS_OUT = os.path.join(BOTTLE_DIR, "bottle_views")


def _category_from_meta(meta_path):
    try:
        with open(meta_path) as f:
            return json.load(f).get("model_cat", "_generic").lower()
    except Exception:
        return "_generic"


# ---- a tiny SYNTHETIC stapler URDF (known answer) for a self-contained UNIT TEST of the function -------------
# base (large box, fixed) + top_arm (smaller box, on a revolute hinge above the base). The function must name
# link_0->base (largest, fixed) and link_1->top_arm (smaller, revolute/movable) for category "stapler".
SYNTH_STAPLER_URDF = """<?xml version="1.0"?>
<robot name="synthetic_stapler">
  <link name="link_0">
    <visual><geometry><box size="0.12 0.04 0.025"/></geometry></visual>
    <collision><geometry><box size="0.12 0.04 0.025"/></geometry></collision>
    <inertial><mass value="0.3"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial>
  </link>
  <link name="link_1">
    <visual><origin xyz="0 0 0.03"/><geometry><box size="0.11 0.035 0.015"/></geometry></visual>
    <collision><origin xyz="0 0 0.03"/><geometry><box size="0.11 0.035 0.015"/></geometry></collision>
    <inertial><origin xyz="0 0 0.03"/><mass value="0.08"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>
  </link>
  <joint name="joint_0" type="revolute">
    <origin xyz="-0.055 0 0.015"/><axis xyz="0 1 0"/>
    <parent link="link_0"/><child link="link_1"/>
    <limit lower="0" upper="1.2" effort="5" velocity="5"/>
  </joint>
</robot>
"""
SYNTH_TRUTH = "link_0 fixed base\nlink_1 hinge top_arm\n"


def unit_test_synthetic_stapler():
    """Self-contained unit test: write a known stapler URDF + its truth, run the function, assert VERIFIED."""
    import tempfile
    d = tempfile.mkdtemp(prefix="synth_stapler_")
    up = os.path.join(d, "stapler.urdf")
    tp = os.path.join(d, "semantics.txt")
    with open(up, "w") as f:
        f.write(SYNTH_STAPLER_URDF)
    with open(tp, "w") as f:
        f.write(SYNTH_TRUTH)
    res = semantic_label_links(up, category="stapler", truth_path=tp)
    print("\n=== UNIT TEST: synthetic stapler (known answer base/top_arm) ===")
    for ln, sv in res.items():
        print(f"   {ln} -> {sv['semantic']:<10} role={sv['role']:<12} verified={sv['verified']} "
              f"conf={sv['confidence']}")
    ok = (res.get("link_0", {}).get("semantic") == "base"
          and res.get("link_1", {}).get("semantic") == "top_arm"
          and res["link_0"]["verified"] and res["link_1"]["verified"])
    print(f"   UNIT_TEST {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    cat = _category_from_meta(BOTTLE_META)
    print(f"PartNet category (meta.json model_cat): {cat}")

    # 1) render the bottle for INSPECTION evidence (the collider multi-view; the agent/owner can open these).
    renders = {}
    try:
        # the URDF's first link mesh is the body; render the whole assembly's collider from canonical views.
        # (render_collider_views takes a mesh; for an articulated URDF we render the body+neck mesh as evidence.)
        body_mesh = os.path.join(BOTTLE_DIR, "textured_objs", "original-3.obj")
        renders = render_collider_views(body_mesh, VIEWS_OUT, scale=1.0, n_views=6,
                                        gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
        print("inspection renders:", json.dumps(renders, indent=2))
    except Exception as ex:
        print(f"(render evidence skipped: {ex})")

    # 2) the semantic naming, VERIFIED against ground truth.
    res = semantic_label_links(BOTTLE_URDF, category=cat, truth_path=BOTTLE_TRUTH, renders=renders)
    print("\n=== LIVE PartNet 3763 bottle: derived semantic names (VERIFIED vs semantics.txt) ===")
    all_ok = True
    for ln, sv in res.items():
        tag = "VERIFIED" if sv["verified"] else ("UNCERTAIN" if sv["verified"] is None else "MISMATCH")
        all_ok = all_ok and bool(sv["verified"])
        print(f"   [{tag}] {ln} -> {sv['semantic']:<14} role={sv['role']:<12} conf={sv['confidence']}")
        print(f"            {sv['evidence']}")

    # 3) the synthetic unit test (generalization).
    unit_ok = unit_test_synthetic_stapler()

    print(f"\nLIVE bottle verified: {all_ok}   |   synthetic unit-test: {'PASS' if unit_ok else 'FAIL'}")
    print("DEMO_NAMING_DONE bottle_verified=" + str(all_ok) + " unit_test=" + str(unit_ok))
    return res


if __name__ == "__main__":
    main()
