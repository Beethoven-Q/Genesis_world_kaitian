# _core_vendored — sim-agnostic core copied from RoboLab_firefly (plan B0)

COPIED, NOT yet promoted. Treat as a STARTING POINT (owner directive): refine for Genesis where sim-to-sim
conventions differ (quaternion wxyz, joint order, gripper action-coupling), and PRUNE anything Isaac-only.
- skills/{grasp,pick_place,trajectory}.py  — pure NumPy/scipy (import-clean).
- soda_ik/{ik,config}.py                    — SODA IK bridge (NumPy/Pinocchio; needs soda-bimanual on PYTHONPATH).
- object_registry.py                        — KEEP the ObjectSpec data; build_object() imports isaaclab.sim → SPLIT/remove.
- export/lerobot_exporter.py                — HDF5→LeRobot v3 (engine-agnostic).
- constants.py                              — KEEP paths; drop DEVICE/VISUALIZE/RECORD_* Isaac flags.
After Genesis passes the pick-place gate (B9), this becomes the separate `agent_sim_core` package (minimal/pruned).
