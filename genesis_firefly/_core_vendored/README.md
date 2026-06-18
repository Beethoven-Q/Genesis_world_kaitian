# _core_vendored — sim-agnostic core (copied from RoboLab_firefly, refined for Genesis)

COPIED then REFINED/PRUNED for Genesis (owner directive: stay skeptical, keep only what's needed).
Becomes the separate `agent_sim_core` package after the pick-place gate (B9).

KEPT (import-clean, reused verbatim):
- skills/{grasp,pick_place,trajectory}.py  — pure NumPy/scipy geometry (pick_place import made relative).
- object_spec.py                            — ObjectSpec DATA + REGISTRY (5 objects incl tennis_ball). No
                                              engine import; the Genesis build_object factory lives in the scene.
- export/lerobot_exporter.py                — HDF5 -> LeRobot v3 (engine-agnostic).
- constants.py                              — paths only (Isaac flags to be dropped as used).

PRUNED (not needed in Genesis):
- soda_ik/  (SODA analytic IK bridge) — REMOVED. Genesis's built-in IK on the dual URDF's own ee_link is
  sub-mm exact (0.1mm/0.01deg, verified); SODA solves a separate gr100.urdf IK chain that does NOT rigidly
  map to the dual ee_link (~1cm residual), so it would bake in grasp error. The Genesis IK adapter lives at
  `genesis_firefly/robots/ik.py`. (SODA's analytic IK still runs the REAL robot via soda-bimanual.)
- object_registry.py (Isaac build_object factory) — REMOVED; split to object_spec.py (data) + a Genesis factory.
