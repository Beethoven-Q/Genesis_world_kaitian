#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DR SWEEP / PROBE — the DR strategist's explore->measure tool (docs/agents.md "DR strategist").

The agent-native way to TEST a candidate DR setting to the EDGE OF USABILITY without paying for a full
collection. It runs the EXISTING collector (tasks/pickplace.py::collect, reused verbatim) on a SMALL batch
(default E=12) for a given DR config -- expressed as range MULTIPLIERS that sample_phys_dr reads from the
environment (DR_POSE_SCALE / DR_MASS_SCALE / DR_FRIC_SCALE, all default 1.0 == the v2 collection) -- then
reads the per-demo HDF5 attrs back and reports:

  * SUCCESS RATE          -- clean placed / N  (clean = success AND not penetrating AND not degenerate)
  * a DIVERSITY measure   -- the achieved spread + bounding-box COVERAGE of the cube/bowl poses, table height,
                             yaw and reach (how representative the batch actually is, not just what we asked for)
  * WHERE FAILURES CLUSTER -- for each DR axis, the failing envs' mean/extent vs the whole batch, so the
                             strategist can see e.g. "all 3 failures were far-reach + tight-clearance".

It NEVER runs a full collection (E is capped) and NEVER edits ranges -- it MEASURES a candidate so the
strategist can recommend a range edit (which the main agent applies) and append a workbook entry.

WHY A SUBPROCESS: `gs.init` / the Nyx Vulkan context is process-global and single-shot (a 2nd gs.Scene
segfaults), exactly as runner/orchestrate.py builds each run in its own subprocess. So sweep launches the
collector as a fresh `python tasks/pickplace.py E seed` subprocess with the multipliers + DATA_DIR/OUT_DIR
exported, then reads its demos.hdf5. This also guarantees the probe and a real collection take the IDENTICAL
code path.

CLI:
  ./.venv/bin/python genesis_firefly/dr/sweep.py [--envs E] [--seed S] [--gpu G] \
        [--pose-scale P] [--mass-scale M] [--fric-scale F] [--out DIR] [--keep] [--json]

  --envs E       batch size (default 12; HARD-capped at 24 -- this is a PROBE, not a collection)
  --seed S       collector seed (default 7, matching the verify gate)
  --gpu G        CUDA_VISIBLE_DEVICES (default 0)
  --pose-scale P pose half-width multiplier  -> DR_POSE_SCALE (default 1.0 == v2)
  --mass-scale M cube mass-shift multiplier  -> DR_MASS_SCALE
  --fric-scale F link friction multiplier    -> DR_FRIC_SCALE
  --out DIR      where the probe writes (default genesis_firefly/output/temp/dr_sweep/p<P>_m<M>_f<F>_s<S>)
  --keep         keep the probe's HDF5/videos (default: kept; pass nothing to keep -- they are small)
  --json         also print a machine-readable JSON block (for the strategist to parse)
  --report-only  SKIP the collector and re-build the report from an already-collected demos.hdf5 under --out
                 (or any finished run's <dataset>/demos.hdf5). Lets the strategist re-diagnose without re-running.

Example -- test a 30%-wider pose range vs the default, both E=12, seed 7:
  ./.venv/bin/python genesis_firefly/dr/sweep.py --pose-scale 1.0 --envs 12 --seed 7   # baseline
  ./.venv/bin/python genesis_firefly/dr/sweep.py --pose-scale 1.3 --envs 12 --seed 7   # widened
The success-rate delta + whether the widened batch stayed clean (0 penetrating/degenerate) + where the
failures cluster is the explore->measure->record loop; the strategist writes the finding to its workbook.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                                   # genesis_firefly/
_REPO = os.path.dirname(_PKG)                                   # repo root
COLLECT_PY = os.path.join(_PKG, "tasks", "pickplace.py")
PYBIN = os.path.join(_REPO, ".venv", "bin", "python")
MAX_PROBE_ENVS = 24                                            # HARD cap: a probe must stay small (not a collection)

# the per-demo DR-plan attrs the collector writes (tasks/pickplace.py HDF5 block). The diversity + clustering
# report is built from these; if an older HDF5 lacks them the report degrades gracefully to the outcome attrs.
_DR_AXES = ["dr_cubx", "dr_cuby", "dr_bowx", "dr_bowy", "dr_tabZ", "dr_yaw", "dr_clr", "dr_reach"]


def run_collector(envs, seed, gpu, scales, out_dir):
    """Launch the collector as a fresh subprocess with the DR multipliers + I/O dirs exported, pinned to one
    GPU. Returns the path to its demos.hdf5 (raises on a non-zero exit). Mirrors how runner/orchestrate.py runs
    a build -- the probe takes the IDENTICAL code path as a real collection, just on a small E."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["DATA_DIR"] = out_dir
    env["OUT_DIR"] = out_dir
    env["DR_POSE_SCALE"] = f"{scales['pose']:.4f}"
    env["DR_MASS_SCALE"] = f"{scales['mass']:.4f}"
    env["DR_FRIC_SCALE"] = f"{scales['fric']:.4f}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "sweep_collect.log")
    print(f"[SWEEP] running collector: E={envs} seed={seed} gpu={gpu} "
          f"pose={scales['pose']} mass={scales['mass']} fric={scales['fric']}", flush=True)
    print(f"[SWEEP] log -> {log_path}", flush=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run([PYBIN, COLLECT_PY, str(envs), str(seed)],
                              cwd=_REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    # echo the key COLLECT lines so the run is auditable inline
    with open(log_path) as lf:
        for line in lf:
            if line.startswith("[COLLECT] ") and any(
                    k in line for k in ("built N=", "grasped,", "penetration:", "DEGENERATE", "executed T=",
                                        "distractors (per-build")):
                print("   " + line.rstrip(), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"collector exited {proc.returncode}; see {log_path}")
    h5 = os.path.join(out_dir, "demos.hdf5")
    if not os.path.exists(h5):
        raise RuntimeError(f"collector wrote no demos.hdf5 (see {log_path})")
    return h5


def read_demos(h5_path):
    """Read every demo's outcome + DR-plan attrs into a list of dicts."""
    rows = []
    with h5py.File(h5_path, "r") as f:
        for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
            a = dict(f["data"][k].attrs)
            rows.append({kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in a.items()})
    return rows


def _clean(r):
    """A demo is a CLEAN success iff placed AND not penetrating AND not degenerate (the export gate)."""
    return bool(r.get("success", False)) and not bool(r.get("penetrating", False)) \
        and not bool(r.get("degenerate", False))


def diversity(rows):
    """Achieved-diversity measure per DR axis: the realised spread (std) + the bounding-box EXTENT (max-min) of
    the values the batch actually sampled. This says how REPRESENTATIVE the batch is (a wide range only helps if
    the draws actually cover it), independent of success. Returns {axis: {std, extent, lo, hi}}."""
    out = {}
    for ax in _DR_AXES:
        vals = np.array([r[ax] for r in rows if ax in r], float)
        if vals.size == 0:
            continue
        out[ax] = dict(std=float(vals.std()), extent=float(np.ptp(vals)),
                       lo=float(vals.min()), hi=float(vals.max()))
    return out


def failure_clusters(rows):
    """Where do failures cluster? For each DR axis, compare the FAILING envs' mean to the whole batch's mean,
    normalised by the batch std (a z-like score): a large |z| means failures are concentrated at one end of
    that axis (e.g. far reach, tight clearance). Returns {axis: {fail_mean, all_mean, z}} sorted by |z|."""
    fails = [r for r in rows if not _clean(r)]
    if not fails:
        return {}, fails
    clusters = {}
    for ax in _DR_AXES:
        allv = np.array([r[ax] for r in rows if ax in r], float)
        failv = np.array([r[ax] for r in fails if ax in r], float)
        if allv.size == 0 or failv.size == 0:
            continue
        sd = allv.std() or 1e-9
        clusters[ax] = dict(fail_mean=float(failv.mean()), all_mean=float(allv.mean()),
                            z=float((failv.mean() - allv.mean()) / sd))
    clusters = dict(sorted(clusters.items(), key=lambda kv: -abs(kv[1]["z"])))
    return clusters, fails


def report(rows, scales, envs, seed):
    """Build the human report + a machine summary dict."""
    n = len(rows)
    clean = sum(_clean(r) for r in rows)
    placed = sum(bool(r.get("success", False)) for r in rows)
    grasped_proxy = placed  # the collector's clean `success` already implies grasped+placed; raw grasp not in attrs
    pen = sum(bool(r.get("penetrating", False)) for r in rows)
    degen = sum(bool(r.get("degenerate", False)) for r in rows)
    arms = {"left": sum(r.get("arm") == "left" for r in rows),
            "right": sum(r.get("arm") == "right" for r in rows)}
    maxpen = max((float(r.get("max_penetration_mm", 0.0)) for r in rows), default=0.0)
    div = diversity(rows)
    clusters, fails = failure_clusters(rows)

    lines = []
    lines.append("=" * 78)
    lines.append(f"DR SWEEP REPORT  E={envs} seed={seed}  "
                 f"pose={scales['pose']} mass={scales['mass']} fric={scales['fric']}")
    lines.append("=" * 78)
    lines.append(f"SUCCESS : clean={clean}/{n} ({100*clean/n:.0f}%)  placed={placed}/{n}  "
                 f"arms L/R={arms['left']}/{arms['right']}")
    lines.append(f"CLEAN-GATE: penetrating={pen}/{n}  degenerate={degen}/{n}  max_penetration={maxpen:.1f}mm  "
                 f"-> stayed-clean={'YES' if (pen == 0 and degen == 0) else 'NO'}")
    lines.append("-" * 78)
    lines.append("DIVERSITY (achieved spread / bbox extent of the batch's draws):")
    pretty = {"dr_cubx": "cube x", "dr_cuby": "cube y", "dr_bowx": "bowl x", "dr_bowy": "bowl y",
              "dr_tabZ": "tableZ", "dr_yaw": "cube yaw", "dr_clr": "cube-bowl clr", "dr_reach": "reach"}
    for ax in _DR_AXES:
        if ax in div:
            d = div[ax]
            lines.append(f"   {pretty.get(ax, ax):<14} std={d['std']:.3f}  extent={d['extent']:.3f}  "
                         f"range=[{d['lo']:.3f}, {d['hi']:.3f}]")
    lines.append("-" * 78)
    if clusters:
        lines.append(f"FAILURE CLUSTERING ({len(fails)} non-clean env(s) -- |z| = how far the failing envs' mean "
                     f"sits from the batch mean, in batch-std units):")
        for ax, c in list(clusters.items())[:5]:
            arrow = "high end" if c["z"] > 0 else "low end"
            lines.append(f"   {pretty.get(ax, ax):<14} fail_mean={c['fail_mean']:.3f}  all_mean={c['all_mean']:.3f}"
                         f"  z={c['z']:+.2f}  ({arrow})")
        fa = [(i, r) for i, r in enumerate(rows) if not _clean(r)]
        lines.append("   failing envs: " + ", ".join(
            f"#{i}[{r.get('arm','?')[0].upper()} clr={r.get('dr_clr',float('nan')):.2f} "
            f"reach={r.get('dr_reach',float('nan')):.2f} "
            f"{'PEN' if r.get('penetrating') else ''}{'DEGEN' if r.get('degenerate') else ''}{'MISS' if not r.get('success') else ''}]"
            for i, r in fa))
    else:
        lines.append("FAILURE CLUSTERING: none -- every env was a clean success.")
    lines.append("=" * 78)

    summary = dict(envs=envs, seed=seed, scales=scales, n=n, clean=clean, clean_rate=clean / n,
                   placed=placed, penetrating=pen, degenerate=degen, max_penetration_mm=maxpen,
                   stayed_clean=(pen == 0 and degen == 0), arms=arms, diversity=div, clusters=clusters)
    return "\n".join(lines), summary


def main():
    ap = argparse.ArgumentParser(description="DR sweep/probe -- test a candidate DR setting on a small batch.")
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--pose-scale", type=float, default=1.0)
    ap.add_argument("--mass-scale", type=float, default=1.0)
    ap.add_argument("--fric-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true", help="(default) keep the probe outputs; they are small")
    ap.add_argument("--json", action="store_true", help="also print a machine-readable JSON summary")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the collector; re-report from an existing demos.hdf5 under --out")
    args = ap.parse_args()

    envs = min(args.envs, MAX_PROBE_ENVS)
    if envs != args.envs and not args.report_only:
        print(f"[SWEEP] E capped {args.envs} -> {envs} (a probe must stay small)", flush=True)
    scales = dict(pose=args.pose_scale, mass=args.mass_scale, fric=args.fric_scale)
    out_dir = args.out or os.path.join(
        _PKG, "output", "temp", "dr_sweep",
        f"p{scales['pose']}_m{scales['mass']}_f{scales['fric']}_s{args.seed}")

    if args.report_only:
        h5 = out_dir if out_dir.endswith(".hdf5") else os.path.join(out_dir, "demos.hdf5")
        if not os.path.exists(h5):
            sys.exit(f"[SWEEP] --report-only: no demos.hdf5 at {h5}")
        print(f"[SWEEP] report-only from {h5}", flush=True)
        rows = read_demos(h5)
        envs = len(rows)
    else:
        h5 = run_collector(envs, args.seed, args.gpu, scales, out_dir)
        rows = read_demos(h5)
    text, summary = report(rows, scales, envs, args.seed)
    print(text, flush=True)
    if args.json:
        print("SWEEP_JSON " + json.dumps(summary), flush=True)
    print(f"[SWEEP] probe outputs -> {out_dir}", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
