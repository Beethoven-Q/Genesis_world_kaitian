# SPDX-License-Identifier: Apache-2.0
"""Smooth Cartesian trajectory execution for the dual-arm skills.

``skills/grasp.py`` emits SPARSE labelled waypoints (pre-grasp, at-object, close, lift, ...).
Driving the arm by jumping the joint-position target straight to each waypoint and HOLDING it
makes the arm lurch from pose to pose with a settling pause at every waypoint -- jerky, unlike
the smooth RoboLab-v1 motion. This module DENSIFIES a waypoint list into a continuous EE-space
stream sampled at the control rate:

  * straight-line Cartesian segments at ~constant linear speed (step count from distance/speed),
  * SLERP'd orientation,
  * an ease-in/ease-out (smoothstep) profile so every segment starts and ends at zero velocity,
  * the gripper open/close as a RAMP over its own dwell (hold pose, ramp grip) -- no snap.

It is pure geometry: ``densify`` returns ``(pos, quat_wxyz, grip, label)`` samples that a thin
executor feeds one-per-sim-step through the IK bridge (seeding each solve from the previous
solution for C0/C1 continuity). SIM-AGNOSTIC -- no Isaac/MuJoCo imports.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _R, Slerp


def ease(u):
    """Smoothstep: maps [0,1]->[0,1] with zero slope at both ends (smooth accel/decel)."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _slerp(q0, q1, us):
    """SLERP between wxyz quats ``q0``->``q1`` at parameters ``us``; returns (len(us), 4) wxyz."""
    key = _R.from_quat(np.array([[q0[1], q0[2], q0[3], q0[0]],
                                 [q1[1], q1[2], q1[3], q1[0]]]))
    out = Slerp([0.0, 1.0], key)(np.clip(us, 0.0, 1.0)).as_quat()  # xyzw
    return np.column_stack([out[:, 3], out[:, 0], out[:, 1], out[:, 2]])


def _quat_angle(q0, q1):
    """Geodesic angle (rad) between two wxyz quaternions."""
    return 2.0 * float(np.arccos(min(1.0, abs(float(np.dot(q0, q1))))))


def densify(waypoints, *, lin_speed=0.15, ang_speed=1.5, dt=0.01,
            min_move_steps=12, dwell_steps=80):
    """Densify sparse ``(label, pos, quat_wxyz, grip)`` waypoints into per-control-step samples.

    Returns a list of ``(pos[3], quat_wxyz[4], grip, label)``. Each consecutive pair is connected by
    a straight-line Cartesian segment whose step count is ``max(lin/lin_speed, ang/ang_speed)/dt``
    (so the EE moves at ~constant speed), eased for a smooth start/stop. A waypoint that only changes
    the gripper (pose unchanged) becomes a DWELL: hold the pose and ramp the gripper over
    ``dwell_steps`` so the jaws close/open smoothly instead of snapping.
    """
    wps = [(l, np.asarray(p, float), np.asarray(q, float), float(g)) for (l, p, q, g) in waypoints]
    out: list = []
    for (l0, p0, q0, g0), (l1, p1, q1, g1) in zip(wps[:-1], wps[1:]):
        lin = float(np.linalg.norm(p1 - p0))
        ang = _quat_angle(q0, q1)
        if lin < 1e-4 and ang < 1e-3:                       # pure gripper move -> dwell ramp
            us = np.linspace(0.0, 1.0, dwell_steps + 1)[1:]
            for u in us:
                out.append((p1.copy(), q1.copy(), g0 + (g1 - g0) * float(u), l1))
            continue
        n = max(min_move_steps, int(np.ceil(max(lin / lin_speed, ang / ang_speed) / dt)))
        e = ease(np.linspace(0.0, 1.0, n + 1)[1:])
        pos = p0[None] + (p1 - p0)[None] * e[:, None]
        quat = _slerp(q0, q1, e)
        grip = g0 + (g1 - g0) * e
        for i in range(n):
            out.append((pos[i], quat[i], float(grip[i]), l1))
    return out
