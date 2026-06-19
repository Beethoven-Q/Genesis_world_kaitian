# SPDX-License-Identifier: Apache-2.0
"""Faithful, batched penetration monitor — the owner's #1 enforcement gate.

WHY THIS EXISTS (owner directive, priority #1; see docs/project_overview.md §0b):
  Abnormal solid-solid interpenetration produces *unphysical*, harmful training data that poisons the
  VLA policy. What cannot happen in reality must not happen in sim. Every demo is screened by this
  monitor; a demo with abnormal interpenetration is REJECTED, never shipped.

WHAT IT READS (the faithful solver API — NOT ``entity.get_contacts``):
  Genesis's rigid solver maintains a persistent per-env contact buffer in
  ``scene.rigid_solver.collider._collider_state.contact_data`` with these fields (physical layout,
  shape ``(n_contacts_max, n_envs)`` -> we transpose to ``(n_envs, n_contacts_max)``):
    - ``penetration``  : per-contact overlap depth in METRES, *positive = the two geoms interpenetrate*
                         (sign confirmed: ``box_contact``/``narrowphase`` set ``penetration = radius - dist``
                          when overlapping, and the constraint solver consumes ``-penetration`` as the
                          position error -> MuJoCo convention, positive == deeper overlap).
    - ``geom_a`` / ``geom_b`` : the two global geom indices of the contacting pair.
  plus ``collider._collider_state.n_contacts`` (shape ``(n_envs,)``) = #live contacts per env. Entries
  beyond ``n_contacts[e]`` are stale padding and are ignored.

  We read these buffers DIRECTLY via ``qd_to_torch`` (the Quadrants->torch zero-copy view). We deliberately
  do NOT call ``collider.get_contacts(...)`` nor ``entity.get_contacts(...)``:
    1. ``collider.get_contacts`` runs a torch ``gather`` over ``contact_sort_idx`` whose dtype is not
       int64 in this build, so it raises ``RuntimeError: gather(): Expected dtype int64 for index`` the
       moment contact pruning/spatial-sort is active (verified live). The raw buffer read has no such bug.
    2. ``entity.get_contacts`` (which wraps the same gather) is entity-scoped and pads with a
       ``valid_mask``; it is the "buggy get_contacts" the project memory warns about. Reading the solver
       buffer straight is the documented bypass.

  Geom -> human-readable identity for the worst offending pair comes from ``solver.geoms[g]``:
  ``g.link.name``, ``g.link.entity.idx``, ``g.link.is_fixed``.

DETECTION TIMING (snapshot vs redetect):
  The contact buffer is refreshed at the START of every ``scene.step()`` (collider detection runs before
  constraint resolution), so right AFTER a ``scene.step()`` the buffer holds the penetration the solver
  saw entering that step. Two modes:
    * ``redetect=False`` (default, "snapshot"): read the buffer as-is. Use this inside an executor
      ``on_step`` callback (called right after ``scene.step()``) to track the running-max penetration
      across the whole trajectory -- this catches the worst moment (e.g. the firm grasp), not just the
      final resting pose.
    * ``redetect=True``: run ``collider.clear()`` + ``collider.detection()`` first, so the buffer reflects
      the CURRENT geometry without any constraint resolution. Use this in a standalone probe where you
      placed geometry but have not stepped (a fresh build, or after ``set_pos``). This is what reads the
      TRUE un-resolved overlap (a deliberate 2 cm overlap reads 20.0 mm, not the post-step residual).

WHAT COUNTS AS PENETRATION (and what does NOT):
  - Only SOLID-SOLID overlap counts. The contact buffer only ever holds geom pairs the narrowphase found
    overlapping, so resting contact sits at ~0 depth and a finger sitting inside a bowl CAVITY produces no
    deep contact at all (the bowl is convex-DECOMPOSED -> the cavity is genuinely empty; hollow stays
    hollow). Nothing special is needed to "skip the cavity": there is simply no solid there to overlap.
  - The GR100 grasp has an expected ~3 mm finger contact-skin / rest_offset (the firm-solver residual the
    collision recipe accepts -- substeps=4 measured ~1.5 mm finger-into-cube). That is NORMAL contact, not
    abnormal interpenetration, and sits far under ``ABNORMAL_THRESH_M``.

THRESHOLD (``ABNORMAL_THRESH_M``):
  Chosen EMPIRICALLY. On a healthy cube->bowl run the per-env max solid-solid penetration sits at/under the
  ~3 mm contact skin. The abnormal threshold is set safely ABOVE that normal max at **7 mm** -- comfortably
  above the firm-grasp residual (~1.5-3 mm) yet far below any real tunnelling (a finger-through-cube was
  32 mm at substeps=1; a wall tunnel is >>1 cm). See docs/robot_collision_cameras.md "Penetration detector".

The monitor is sim-coupled ONLY through reading the scene's solver state; it never mutates the sim
(redetect's clear+detection rebuilds the same buffer the next step would, and is followed by no integrate).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from genesis.utils.misc import qd_to_torch

# ----------------------------------------------------------------------------------------------------
# Empirically chosen abnormal-penetration threshold (metres). See module docstring + the docs section.
#   normal cube->bowl run: per-env max solid-solid penetration <= ~3 mm (the firm-grasp contact skin).
#   7 mm sits safely above that and well below any real tunnelling (>=1 cm). Override per-call if needed.
ABNORMAL_THRESH_M: float = 0.007


def _read_contact_buffer(scene, redetect: bool):
    """Read the solver's persistent per-env contact buffer DIRECTLY (the faithful API).

    Returns (pen, ga, gb, nc) as numpy arrays:
      pen : (N, Cmax) float32  per-contact penetration depth in metres (positive == overlap)
      ga  : (N, Cmax) int      geom_a global index per contact
      gb  : (N, Cmax) int      geom_b global index per contact
      nc  : (N,)      int      number of LIVE contacts per env (entries >= nc[e] are stale padding)
    N == max(scene.n_envs, 1).
    """
    solver = scene.rigid_solver
    collider = solver.collider
    if redetect:
        # Rebuild the contact buffer from the CURRENT geometry, with NO constraint resolution / integration,
        # so we read the true un-resolved overlap. clear() zeroes n_contacts; detection() re-populates.
        collider.clear()
        collider.detection()

    cs = collider._collider_state
    # contact_data buffers are stored (n_contacts_max, B); transpose -> (B, n_contacts_max). copy=False = view.
    pen = qd_to_torch(cs.contact_data.penetration, transpose=True, copy=False)
    ga = qd_to_torch(cs.contact_data.geom_a, transpose=True, copy=False)
    gb = qd_to_torch(cs.contact_data.geom_b, transpose=True, copy=False)
    nc = qd_to_torch(cs.n_contacts, copy=False)

    pen = pen.detach().cpu().numpy()
    ga = ga.detach().cpu().numpy()
    gb = gb.detach().cpu().numpy()
    nc = nc.detach().cpu().numpy().astype(np.int64)

    # n_envs == 0 (un-parallelized) -> the solver still stores a batch dim of 1. Normalize to 2-D (N, Cmax).
    if pen.ndim == 1:
        pen = pen[None, :]
        ga = ga[None, :]
        gb = gb[None, :]
    if np.ndim(nc) == 0:
        nc = np.asarray([int(nc)], dtype=np.int64)
    return pen, ga, gb, nc


def _normalize_ignore_pairs(ignore_pairs: Iterable | None) -> set:
    """Normalize an ignore list of (geom_i, geom_j) global-index pairs into an order-independent set."""
    out = set()
    if not ignore_pairs:
        return out
    for p in ignore_pairs:
        i, j = int(p[0]), int(p[1])
        out.add((min(i, j), max(i, j)))
    return out


def max_penetration(scene, robot=None, ignore_pairs=None, redetect: bool = False,
                    self_collision: bool = True) -> dict:
    """Per-env MAX solid-solid interpenetration depth (metres), read straight from the solver.

    Reads the solver's persistent contact buffer (``geom_a/geom_b/penetration`` + ``n_contacts``) directly
    -- the faithful "read penetration straight from the solver, bypass the buggy get_contacts" path. The
    contact buffer only ever holds geom pairs the narrowphase found OVERLAPPING, so resting contact is ~0
    and a finger inside a (convex-decomposed) bowl cavity produces no deep contact (hollow stays hollow).

    Parameters
    ----------
    scene : gs.Scene
        The built scene. Must be parallel-built (or single-env); read after ``scene.step()`` (snapshot) or
        with ``redetect=True`` (fresh geometry, no step yet).
    robot : optional
        Unused for the core read (kept for API symmetry / future per-robot filtering). The detector is
        object-agnostic: ANY solid-solid overlap is flagged, robot-vs-object or object-vs-object alike.
    ignore_pairs : iterable of (geom_i, geom_j), optional
        Global geom-index pairs to EXCLUDE (e.g. a known-acceptable mating contact). Order-independent.
    redetect : bool, default False
        False -> read the existing buffer (use right after ``scene.step()`` / inside ``on_step``).
        True  -> ``collider.clear()`` + ``collider.detection()`` first (use for a freshly placed geometry
                 that has not been stepped; reads the TRUE un-resolved overlap).
    self_collision : bool, default True
        If False, drop contacts where both geoms belong to the same entity (robot self-contact). Default
        keeps them: a robot link tunnelling into another robot link is also abnormal.

    Returns
    -------
    dict with:
      'depth_m'       : (N,) float32  per-env max solid-solid penetration depth in metres (0 if none).
      'depth_mm'      : (N,) float32  same, in millimetres (convenience).
      'worst_pair'    : list[N] of (geom_a, geom_b) int tuples for the worst contact per env, or None.
      'worst_pair_names' : list[N] of "linkA<->linkB" strings (or None) for the worst contact per env.
    """
    solver = scene.rigid_solver
    N = max(int(scene.n_envs), 1)
    pen, ga, gb, nc = _read_contact_buffer(scene, redetect)
    ignore = _normalize_ignore_pairs(ignore_pairs)

    # Per-geom entity index, for the optional self-collision drop + readable names.
    geoms = solver.geoms
    n_geoms = len(geoms)
    ent_of_geom = np.full(n_geoms, -1, dtype=np.int64)
    for g in geoms:
        if 0 <= g.idx < n_geoms:
            ent_of_geom[g.idx] = g.link.entity.idx

    depth_m = np.zeros(N, np.float32)
    worst_pair = [None] * N
    worst_pair_names = [None] * N

    for e in range(N):
        k = int(nc[e]) if e < len(nc) else 0
        if k <= 0:
            continue
        best_d = 0.0
        best_ij = None
        for i in range(k):
            a = int(ga[e, i])
            b = int(gb[e, i])
            if a < 0 or b < 0:           # stale / unfilled slot
                continue
            d = float(pen[e, i])
            if d <= 0.0:                 # not actually overlapping (separated contact in the buffer)
                continue
            key = (min(a, b), max(a, b))
            if key in ignore:
                continue
            if not self_collision and 0 <= a < n_geoms and 0 <= b < n_geoms \
                    and ent_of_geom[a] == ent_of_geom[b] and ent_of_geom[a] >= 0:
                continue
            if d > best_d:
                best_d = d
                best_ij = (a, b)
        depth_m[e] = best_d
        if best_ij is not None:
            worst_pair[e] = best_ij
            na = geoms[best_ij[0]].link.name if 0 <= best_ij[0] < n_geoms else f"geom{best_ij[0]}"
            nb = geoms[best_ij[1]].link.name if 0 <= best_ij[1] < n_geoms else f"geom{best_ij[1]}"
            worst_pair_names[e] = f"{na}<->{nb}"

    return dict(
        depth_m=depth_m,
        depth_mm=(depth_m * 1000.0).astype(np.float32),
        worst_pair=worst_pair,
        worst_pair_names=worst_pair_names,
    )


def abnormal_penetration(scene, thresh_m: float = ABNORMAL_THRESH_M, *, robot=None, ignore_pairs=None,
                         redetect: bool = False, self_collision: bool = True):
    """Flag envs whose MAX solid-solid penetration exceeds ``thresh_m`` (the rejection gate).

    Returns (per_env_bool (N,), per_env_depth_m (N,)). An env flagged True has abnormal interpenetration
    and its demo MUST be rejected (not counted as a clean success, dropped from ``success_only`` export).

    ``thresh_m`` defaults to ``ABNORMAL_THRESH_M`` (7 mm) -- safely above the ~3 mm firm-grasp contact skin
    and well below any real tunnelling. See the module docstring + docs for the rationale.
    """
    res = max_penetration(scene, robot=robot, ignore_pairs=ignore_pairs, redetect=redetect,
                          self_collision=self_collision)
    depth = res["depth_m"]
    flagged = depth > float(thresh_m)
    return flagged, depth


class PenetrationTracker:
    """Running per-env MAX penetration across a trajectory, for an executor ``on_step`` callback.

    The contact buffer is per-step; a demo's verdict should be the WORST penetration seen at ANY moment
    (the firm grasp is usually the peak), not just the final resting pose. Wire it as::

        tracker = PenetrationTracker(scene, n_envs=N, ignore_pairs=...)
        def on_step(t, full_cmd, labels): ...; tracker.update()   # called right after each scene.step()
        ...run executor...
        depth_m   = tracker.depth_m()                              # (N,) worst-ever per env
        flagged   = tracker.abnormal(thresh_m)                     # (N,) bool

    ``update()`` reads the buffer left by the just-completed ``scene.step()`` (``redetect=False``), so it
    is cheap (no extra detection) and reflects that step's penetration.
    """

    def __init__(self, scene, n_envs: int | None = None, *, robot=None, ignore_pairs=None,
                 self_collision: bool = True):
        self.scene = scene
        self.robot = robot
        self.ignore_pairs = ignore_pairs
        self.self_collision = self_collision
        self.N = int(n_envs) if n_envs is not None else max(int(scene.n_envs), 1)
        self._max = np.zeros(self.N, np.float32)
        self._worst_names = [None] * self.N

    def update(self):
        """Read the current (post-step) contact buffer and fold its max into the running per-env max."""
        res = max_penetration(self.scene, robot=self.robot, ignore_pairs=self.ignore_pairs,
                              redetect=False, self_collision=self.self_collision)
        d = res["depth_m"]
        upd = d > self._max
        self._max = np.maximum(self._max, d)
        for e in np.where(upd)[0]:
            self._worst_names[e] = res["worst_pair_names"][e]
        return res

    def depth_m(self) -> np.ndarray:
        """(N,) worst-ever per-env penetration depth in metres."""
        return self._max.copy()

    def depth_mm(self) -> np.ndarray:
        """(N,) worst-ever per-env penetration depth in millimetres."""
        return (self._max * 1000.0).astype(np.float32)

    def worst_names(self):
        """list[N] of the "linkA<->linkB" string for each env's worst-ever contact (or None)."""
        return list(self._worst_names)

    def abnormal(self, thresh_m: float = ABNORMAL_THRESH_M) -> np.ndarray:
        """(N,) bool: envs whose worst-ever penetration exceeds ``thresh_m`` (reject these demos)."""
        return self._max > float(thresh_m)
