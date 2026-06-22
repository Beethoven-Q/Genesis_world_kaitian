# SPDX-License-Identifier: Apache-2.0
"""Reusable CLUTTER / DISTRACTOR placement skill — choose how many irrelevant objects to spawn, which TYPES,
and a clear on-table XY + yaw for each, GUARANTEED out of a task's swept keep-out corridor and never stacked.

WHY THIS IS A SKILL (not task-local): every manipulation task that wants realistic clutter needs the SAME two
things — (1) pick a varied, physically-placeable set of distractor types, and (2) drop them on the table in OPEN
areas that the arm never sweeps. The only task-specific input is the GEOMETRY of the swept corridor (where the
arm + the carried object travel). So this skill takes that corridor as DATA — a list of keep-out DISKS and
keep-out SEGMENT-TUBES (plus a near-seam strip) — and returns clear placements. A pick-place task passes its
cube/bowl/carry/home corridor; a future pour/stack/hand-over task passes its own. No corridor geometry is baked
in here.

WHAT IT GUARANTEES (the same correctness the in-task placement had):
  * the WHOLE body (not just the centre) stays ON the table and OUT of every keep-out feature at ANY yaw
    (circumscribed-radius clearance), so a long banana's far tip is never clipped;
  * no two distractors ever deeply interpenetrate at spawn (enclosing-square spacing) — the firm solver only
    NaNs on a DEEP spawn overlap, and greedy grid assignment gives each object its own well-separated cell;
  * at most ONE large/long object per build (the open area can't hold two ~18 cm objects clear of the corridor).

METHOD = greedy grid assignment: tile the table with a fine anchor grid (inset so any object stays on the table
at any yaw), keep the corridor-clear cells (favouring the open side), then greedily pick K cells pairwise spaced
by each pair's enclosing squares + a surface gap. A tiny in-cell jitter keeps variety without breaking spacing.

PARITY: extracted verbatim from tasks/pickplace.py (the cube collection's clutter draws the IDENTICAL types +
poses, byte-for-byte, because the RNG draw ORDER is preserved). The task now passes its corridor + table bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- DEFAULT clearance constants (metres) -- a task may override per call, but these match the verified
# pick-place corridor tuning (max distractor XY displacement stays well under 2 cm). All are half-clearances
# about a point/segment; the sampler ADDS each object's footprint radius so the whole BODY clears. ------------
CLR_DIST = 0.03      # SURFACE gap between two distractors' footprint disks (added to BOTH footprint radii ->
#                      centre-to-centre >= r_i + r_j + gap, so two bodies never overlap/stack at spawn)
TABLE_MARGIN = 0.06  # inset from every table edge so a distractor never spawns half-off / on the rim
GRID_PITCH = 0.035   # anchor-grid spacing (~3.5 cm): fine enough to find clear cells, coarse enough to be cheap


@dataclass
class KeepoutCorridor:
    """The TASK's swept keep-out geometry, as DATA (per-env arrays). The placement rejects any footprint that
    intrudes on ANY feature here. Reusable for ANY task: pass the disks/segments the arm + carried object sweep.

      disks    : list of (cx, cy, clr) -- a keep-out DISK; ``cx/cy`` are per-env (N,) arrays, ``clr`` a scalar
                 half-clearance to the distractor CENTRE (the object's footprint radius is ADDED on top).
      segments : list of (ax, ay, bx, by, clr) -- a keep-out TUBE about the segment a->b (per-env (N,) arrays),
                 ``clr`` the half-width (footprint radius added on top). Use for carry / return tubes.
      seam_x   : scalar -- forbid the near-seam strip x < seam_x (the arm links/base sweep it). 0 disables.
    """
    disks: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    seam_x: float = 0.0


def _seg_clearance(px, py, ax, ay, bx, by):
    """Per-env XY distance from points (px,py) to the segment a->b (a,b are per-env arrays). Vectorised."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby + 1e-12
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy)


def footprint_radius(spec):
    """Half the XY diagonal of an object's AABB = the radius of the disk that contains the object at ANY yaw.
    Insetting the table bounds + every clearance by this guarantees the WHOLE body (not just its centre) stays
    on the table and out of the corridor, regardless of the random in-plane spin. Used for edge + corridor
    clearance (safety-critical, and there is room there)."""
    e = spec.scaled_extents()
    return 0.5 * float(np.hypot(e[0], e[1]))


def space_radius(spec):
    """A TIGHTER radius for inter-distractor SPACING only = half the object's longer horizontal extent (its
    enclosing-square half-side), not the diagonal. Two flat-resting bodies whose enclosing squares are
    separated by CLR_DIST never deeply interpenetrate (the firm solver only NaNs on a DEEP spawn overlap),
    and this makes packing 3 long objects (banana/book) onto the table FEASIBLE where the circumscribed radius
    would not. Edge/corridor still use the conservative circumscribed radius."""
    e = spec.scaled_extents()
    return 0.5 * float(max(e[0], e[1]))


def choose_distractor_types(rng, universe, target, n_large_max=1, large=frozenset(),
                            counts=(2, 3), pool=None):
    """PER-BUILD: how many distractors (K drawn from ``counts``) and which TYPES (entities are created before
    scene.build, so the count + identities are fixed for the whole build). Drawn WITHOUT replacement so the K
    distractors are visually distinct lookalikes (the policy must disambiguate the TARGET from a varied clutter
    set). At most ``n_large_max`` large/long objects (those in ``large``) per build so every set fits on the
    table out-of-corridor with real spacing.

    The TARGET's own type is EXCLUDED from the pool so the target is never ambiguous (e.g. a banana target draws
    clutter from the universe minus banana). ``pool`` overrides the derived (universe minus target) pool."""
    if pool is None:
        pool = [n for n in universe if n != target]
    k = int(rng.choice(list(counts)))
    names = None
    for _ in range(40):
        names = list(rng.choice(pool, size=k, replace=False))
        if sum(n in large for n in names) <= n_large_max:
            return names
    return names


def _anchor_grid(table_bounds, rmax, pitch=GRID_PITCH):
    """A fine grid of candidate anchor CENTRES over the table (``table_bounds`` = (x0, x1, y0, y1) BEFORE the
    rmax inset), inset by rmax so any object centred on a cell stays fully on the table at any yaw. The
    placement greedily picks K mutually-spaced, corridor-clear cells from this grid (one per distractor) ->
    non-overlap is GUARANTEED by construction."""
    bx0, bx1, by0, by1 = table_bounds
    x0 = bx0 + rmax
    x1 = bx1 - rmax
    y0 = by0 + rmax
    y1 = by1 - rmax
    xs = np.arange(x0, x1 + 1e-6, pitch)
    ys = np.arange(y0, y1 + 1e-6, pitch)
    if xs.size == 0:
        xs = np.array([(x0 + x1) / 2])
    if ys.size == 0:
        ys = np.array([(y0 + y1) / 2])
    return np.array([(x, y) for x in xs for y in ys])


def _corridor_clear(cx, cy, rfoot, corridor: KeepoutCorridor, e):
    """Bool mask over grid cells: is footprint (cx,cy,rfoot) clear of EVERY keep-out feature for env ``e``?
    Every threshold ADDS rfoot so the whole BODY (not just the centre) clears. Generic over the corridor DATA:
    the near-seam strip + each keep-out disk + each keep-out segment-tube the task supplied."""
    M = cx.shape[0]
    ok = (cx - rfoot >= corridor.seam_x)
    for (dx, dy, clr) in corridor.disks:
        ok &= np.hypot(cx - dx[e], cy - dy[e]) >= clr + rfoot
    for (ax, ay, bx, by, clr) in corridor.segments:
        ok &= _seg_clearance(cx, cy, np.full(M, ax[e]), np.full(M, ay[e]),
                             np.full(M, bx[e]), np.full(M, by[e])) >= clr + rfoot
    return ok


def sample_distractor_poses(N, specs, corridor: KeepoutCorridor, table_bounds, rng, sgn,
                            *, clr_dist=CLR_DIST, pitch=GRID_PITCH):
    """PER-ENV: choose a clear XY + yaw for each distractor. GUARANTEES: (a) the whole BODY stays on the table
    and OUT of the task's keep-out corridor at ANY yaw (circumscribed-radius clearance), and (b) no two
    distractors ever deeply interpenetrate at spawn (enclosing-square spacing). Method = GREEDY GRID ASSIGNMENT:
    tile the table with a fine anchor grid, keep the corridor-clear cells (favouring the opposite-y side from
    the active arm for variety), then greedily pick K cells that are pairwise spaced >= rs_i+rs_j+clr_dist.
    Because each distractor gets its OWN well-separated grid cell, nothing stacks (the firm solver NaNs on a deep
    spawn overlap) and nothing sits in the arm's path (so it is never swiped). A tiny in-cell jitter (< half the
    leftover slack) keeps the variety without breaking the spacing.

    ``table_bounds`` = (x0, x1, y0, y1) the table-usable bounds (already insetting the edge margin). ``sgn`` =
    per-env active-arm side sign (used only to FAVOUR the opposite-y side for spread; never for clearance).
    Returns xy:(K,N,2), yaw:(K,N)."""
    K = len(specs)
    rfoot = np.array([footprint_radius(s) for s in specs])          # circumscribed: edge + corridor (safe@any yaw)
    rspace = np.array([space_radius(s) for s in specs])             # enclosing-square: inter-object spacing
    rmax = float(rfoot.max())
    grid = _anchor_grid(table_bounds, rmax, pitch=pitch)
    M = grid.shape[0]
    out_xy = np.zeros((K, N, 2), np.float64)
    out_yaw = ((rng.rand(K, N) - 0.5) * np.radians(180))
    # far-x edge fallback bounds (corridor-clear by construction: far from the seam/arm)
    bx0, bx1, by0, by1 = table_bounds
    # largest distractor first -> the hardest-to-place objects claim space before the small ones
    order = list(np.argsort(-rspace))
    for e in range(N):
        # corridor-clear cells for this env (use rmax so the test is valid for every distractor's footprint)
        cc = _corridor_clear(grid[:, 0], grid[:, 1], rmax, corridor, e)
        cand = grid[cc]
        # favour the opposite-y side (more open) then random, so clutter spreads & varies build-to-build
        if cand.shape[0]:
            opp = (np.sign(cand[:, 1]) != np.sign(sgn[e]))
            cand = cand[np.lexsort((rng.rand(cand.shape[0]), ~opp))]
        chosen_x = []; chosen_y = []; chosen_rs = []
        for oi, k in enumerate(order):
            rs = float(rspace[k]); pick = None
            for ci in range(cand.shape[0]):                         # first corridor-clear cell spaced from chosen
                cx, cy = cand[ci]
                if all(np.hypot(cx - px, cy - py) >= clr_dist + rs + prs
                       for px, py, prs in zip(chosen_x, chosen_y, chosen_rs)):
                    pick = (cx, cy); break
            if pick is None:                                        # no corridor-clear cell fits (very rare) ->
                # park on the FAR-x edge (far from the seam/arm -> corridor-clear by construction), offset in Y
                # by placement index so two un-placeable objects never coincide. X stays at the far edge.
                xb = bx1 - rfoot[k]
                yb = by1 - rfoot[k]
                step = 2.0 * (float(rspace.max()) + clr_dist)
                yy = float(np.clip(yb - oi * step, -yb, yb))
                pick = (xb, -sgn[e] * yy)
            out_xy[k, e] = pick
            chosen_x.append(pick[0]); chosen_y.append(pick[1]); chosen_rs.append(rs)
    return out_xy, out_yaw
