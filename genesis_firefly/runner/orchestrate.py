#!/usr/bin/env python3
"""BUILD-BATCH ORCHESTRATOR — scale the pick-place collector to B parallel subprocess builds x E envs.

WHY SUBPROCESSES (mandatory): a 2nd `gs.Scene` in one process SEGFAULTS (Nyx/Vulkan is single-shot). So each
BUILD is a FRESH subprocess of the collector (`runner/collect.py`), which the orchestrator reuses verbatim --
it never duplicates collection logic. Each build draws its OWN per-build "look" from its seed (table texture,
object colors, object sizes, object type, table size, side-cam pose, distractor SET) -- so B distinct looks =
the cross-build DR variety. Within a build, the E envs vary the per-env space (pose/physics/HDRI/light + the
50/50 distractors + ~50/50 L/R arm). See docs/domain_randomization.md (per-build vs per-env split) + §8.

GPU POOL (owner rule: ONE sim process per GPU). We detect the available GPUs (nvidia-smi, or the `GPUS=0,1`
env override) and run a worker pool of size = n_gpus. Each worker pulls the next build off a queue, runs it
with `CUDA_VISIBLE_DEVICES=<gpu>` pinned, waits, then pulls the next. At most one build per GPU concurrently.

SHARD -> MERGE -> SYMLINK (storage convention, project_overview.md §0b/§7):
  - the final dataset lives at  /data3/genesis_fulldr/<dataset>/  (HDF5 + per-cam videos).
  - each build writes a SHARD at /data3/genesis_fulldr/<dataset>/_shards/build_<b>/ (its own demos.hdf5+videos).
  - after all builds finish, MERGE the B shards into <dataset>/demos.hdf5 with GLOBALLY renumbered demo keys
    demo_0 .. demo_{B*E-1}, preserving every dataset + every attr (h5py group copy) and adding a `build` attr.
    The per-cam videos are copied+renumbered to <dataset>/videos/cam_*/demo_<g>.mp4 to match.
  - an output/<dataset> SYMLINK -> the /data3 dataset dir, so the owner can preview in-workspace.
  - shards are KEPT (not deleted) so a failed merge is recoverable; the dataset is the MERGED file.
  - if /data3 is not writable -> FAIL LOUDLY (never silently write into the repo).

A `summary.json` (totals: B*E demos, clean successes, grasp rate, abnormal-penetration count, per-build looks)
is written to <dataset>/ and printed.

  CLI:  ./.venv/bin/python genesis_firefly/runner/orchestrate.py B E [seed0] [dataset_name]
        B  = number of builds (seeds seed0, seed0+1, ...)        E = envs per build (demos per build)
        seed0 (default 0)        dataset_name (default fulldr_orch)
  ENV:  GPUS=0,1   override the GPU pool (default = all GPUs nvidia-smi reports)
        DATA_ROOT  (default /data3/genesis_fulldr)   SPP   (forwarded to the collector)
"""
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))                 # repo root (.../Genesis_world_kaitian)
_PKG = os.path.dirname(_HERE)                                   # genesis_firefly/
COLLECT_PY = os.path.join(_HERE, "collect.py")
DATA_ROOT = os.environ.get("DATA_ROOT", "/data3/genesis_fulldr")

# COLLECT lines we relay live (per build), tagged with the build id. These are the key signals the owner reads.
_RELAY_RE = re.compile(r"\[COLLECT\] (distractors \(per-build|built N=|executed T=|\d+/\d+ grasped|penetration:|"
                       r"distractors: max)")


def detect_gpus():
    """The GPU pool. `GPUS=0,1` env override wins; else parse nvidia-smi index list; else fall back to [0]."""
    env = os.environ.get("GPUS")
    if env:
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        gpus = [ln.strip() for ln in out.splitlines() if ln.strip() != ""]
        return gpus or ["0"]
    except Exception as e:
        print(f"[ORCH] nvidia-smi unavailable ({e}); defaulting to GPU 0", flush=True)
        return ["0"]


def run_build(b, seed, gpu, dataset_dir, E, spp):
    """Run ONE build as a fresh collector subprocess pinned to `gpu`, writing its shard. Streams key COLLECT
    lines (tagged [b<b> g<gpu>]) and parses the per-build look (distractors) + the result counts from stdout.
    Returns a dict: ok, seed, gpu, build, returncode, wall, shard_dir, distractors, grasped/placed/abnormal."""
    shard = os.path.join(dataset_dir, "_shards", f"build_{b}")
    tiles = os.path.join(shard, "tiles")
    os.makedirs(shard, exist_ok=True)
    os.makedirs(tiles, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)                      # owner rule: one sim process per GPU
    env["DATA_DIR"] = shard                                     # collector writes demos.hdf5 + videos/ here
    env["OUT_DIR"] = tiles                                      # collector writes preview tiles here
    if spp:
        env["SPP"] = str(spp)
    tag = f"[b{b} g{gpu}]"
    print(f"{tag} START seed={seed} E={E} -> {shard}", flush=True)
    t0 = time.time()
    info = dict(ok=False, build=b, seed=seed, gpu=str(gpu), shard_dir=shard,
                distractors=None, grasped=None, placed=None, through_wall=None, abnormal=None)
    lines = []
    proc = subprocess.Popen([sys.executable, COLLECT_PY, str(E), str(seed)],
                            cwd=_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if _RELAY_RE.search(line):                             # relay only the key COLLECT signals, tagged
            print(f"{tag} {line}", flush=True)
        m = re.search(r"\[COLLECT\] distractors \(per-build, K=\d+\): \[(.*)\]", line)
        if m:
            # the collector prints numpy str reprs (e.g.  np.str_('banana'), 'apple', "pen" ) -> pull the
            # quoted name out of each comma-separated token so summary.json carries clean strings.
            info["distractors"] = [re.sub(r"^np\.str_\(|\)$", "", t.strip()).strip("'\"")
                                   for t in m.group(1).split(",") if t.strip()]
        m = re.search(r"\[COLLECT\] (\d+)/(\d+) grasped, (\d+)/\d+ placed, through-wall=(\d+)", line)
        if m:
            info["grasped"], info["placed"], info["through_wall"] = int(m.group(1)), int(m.group(3)), int(m.group(4))
        m = re.search(r"\[COLLECT\] penetration: .*abnormal=(\d+)/\d+", line)
        if m:
            info["abnormal"] = int(m.group(1))
    rc = proc.wait()
    info["returncode"] = rc
    info["wall"] = time.time() - t0
    hdf5 = os.path.join(shard, "demos.hdf5")
    info["ok"] = (rc == 0) and os.path.exists(hdf5) and "COLLECT_DONE" in "\n".join(lines[-15:])
    if not info["ok"]:
        print(f"{tag} FAILED rc={rc} hdf5_exists={os.path.exists(hdf5)} (last lines below)", flush=True)
        for ln in lines[-25:]:
            print(f"{tag}   | {ln}", flush=True)
    else:
        print(f"{tag} DONE rc=0 in {info['wall']:.1f}s  "
              f"grasped={info['grasped']} placed={info['placed']} abnormal={info['abnormal']} "
              f"distractors={info['distractors']}", flush=True)
    return info


def worker(gpu, jobq, results, lock):
    """A GPU worker: pull the next build off the queue and run it on this GPU until the queue is empty."""
    while True:
        try:
            b, seed = jobq.get_nowait()
        except queue.Empty:
            return
        try:
            res = run_build(b, seed, gpu, results["_dataset_dir"], results["_E"], results["_spp"])
        except Exception as e:                                 # a worker crash must not lose the whole pool
            res = dict(ok=False, build=b, seed=seed, gpu=str(gpu), returncode=-1, error=repr(e),
                       wall=0.0, distractors=None, grasped=None, placed=None, abnormal=None,
                       shard_dir=os.path.join(results["_dataset_dir"], "_shards", f"build_{b}"))
            print(f"[b{b} g{gpu}] WORKER EXCEPTION: {e!r}", flush=True)
        with lock:
            results[b] = res
        jobq.task_done()


# ---------------------------------------------------------------------------- #
# MERGE the B shards -> the single dataset (globally renumbered demos + matching videos)
# ---------------------------------------------------------------------------- #
def merge_shards(dataset_dir, build_results, B, E):
    """Copy every shard's /data/demo_<i> group into <dataset>/demos.hdf5 under a GLOBALLY renumbered key
    /data/demo_<g> (g = 0 .. sum(E_b)-1), preserving ALL datasets (actions/states/ee_pose) AND ALL attrs
    (h5py group copy carries datasets + attrs), adding a `build` attr = b. The per-cam videos are copied +
    renumbered to <dataset>/videos/cam_*/demo_<g>.mp4 to match. Returns (n_merged, per_demo_meta, cam_counts).

    h5py.File.copy of a Group copies the whole subtree including attrs, so success/seed/arm/hdr/has_distractors/
    distractors/max_penetration_mm/penetrating ride along untouched -- we only ADD `build`. Shards are kept.
    """
    import h5py
    merged_path = os.path.join(dataset_dir, "demos.hdf5")
    vdir = os.path.join(dataset_dir, "videos")
    cams = ("cam_side", "cam_lw", "cam_rw")
    for nm in cams:
        os.makedirs(os.path.join(vdir, nm), exist_ok=True)
    per_demo = []
    cam_counts = {nm: 0 for nm in cams}
    g = 0
    if os.path.exists(merged_path):                            # a re-run -> start the merged file fresh
        os.remove(merged_path)
    with h5py.File(merged_path, "w") as fout:
        fout.create_group("data")
        for b in range(B):
            res = build_results.get(b)
            if not res or not res.get("ok"):
                print(f"[MERGE] build {b} not ok -> SKIPPED in merge", flush=True)
                continue
            shard = res["shard_dir"]
            shard_h5 = os.path.join(shard, "demos.hdf5")
            with h5py.File(shard_h5, "r") as fin:
                # source demo order: numeric (demo_0, demo_1, ...) NOT lexical (demo_0, demo_1, demo_10, ...)
                local_keys = sorted(fin["data"].keys(), key=lambda k: int(k.split("_")[1]))
                for lk in local_keys:
                    li = int(lk.split("_")[1])
                    dst = f"demo_{g}"
                    fin.copy(f"data/{lk}", fout["data"], name=dst)   # whole subtree: datasets + attrs
                    d = fout[f"data/{dst}"]
                    d.attrs["build"] = b                              # provenance: which build produced it
                    a = dict(d.attrs)
                    per_demo.append(dict(
                        g=g, build=b, src=lk, seed=int(a.get("seed", res["seed"])),
                        arm=a.get("arm", "?"), success=bool(a.get("success", False)),
                        has_distractors=bool(a.get("has_distractors", False)),
                        distractors=a.get("distractors", ""),
                        max_penetration_mm=float(a.get("max_penetration_mm", -1.0)),
                        penetrating=bool(a.get("penetrating", False))))
                    # copy + renumber the per-cam videos for this demo
                    for nm in cams:
                        src_mp4 = os.path.join(shard, "videos", nm, f"demo_{li}.mp4")
                        if os.path.exists(src_mp4):
                            shutil.copy2(src_mp4, os.path.join(vdir, nm, f"demo_{g}.mp4"))
                            cam_counts[nm] += 1
                    g += 1
    print(f"[MERGE] merged {g} demos -> {merged_path}; videos per cam: {cam_counts}", flush=True)
    return g, per_demo, cam_counts


def make_symlink(dataset_dir, dataset_name):
    """Create an output/<dataset_name> symlink -> the /data3 dataset dir, for in-workspace preview."""
    out_dir = os.path.join(_PKG, "output")
    os.makedirs(out_dir, exist_ok=True)
    link = os.path.join(out_dir, dataset_name)
    if os.path.islink(link) or os.path.exists(link):
        if os.path.islink(link):
            os.unlink(link)
        else:
            raise RuntimeError(f"{link} exists and is NOT a symlink -- refusing to clobber a real path")
    os.symlink(dataset_dir, link)
    resolved = os.path.realpath(link)
    print(f"[ORCH] symlink {link} -> {resolved}", flush=True)
    return link, resolved


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    B = int(sys.argv[1])
    E = int(sys.argv[2])
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dataset_name = sys.argv[4] if len(sys.argv) > 4 else "fulldr_orch"
    spp = os.environ.get("SPP")

    gpus = detect_gpus()
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)

    # ---- /data3 writability gate: FAIL LOUDLY rather than silently writing into the repo ----
    try:
        os.makedirs(os.path.join(dataset_dir, "_shards"), exist_ok=True)
        probe = os.path.join(dataset_dir, ".orch_write_probe")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except Exception as e:
        print(f"[ORCH] FATAL: dataset root {dataset_dir} is NOT writable ({e}). "
              f"Refusing to silently write into the repo. Fix /data3 (or DATA_ROOT) and re-run.", flush=True)
        sys.exit(3)

    print(f"[ORCH] B={B} builds x E={E} envs = {B*E} demos | seeds {seed0}..{seed0+B-1} | "
          f"GPUs={gpus} (<=1 build/GPU) | dataset={dataset_dir} | collector={COLLECT_PY}", flush=True)

    # ---- the build queue + the GPU worker pool (size = n_gpus) ----
    jobq = queue.Queue()
    for b in range(B):
        jobq.put((b, seed0 + b))
    results = {"_dataset_dir": dataset_dir, "_E": E, "_spp": spp}
    lock = threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(gpu, jobq, results, lock), daemon=True)
               for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pool_wall = time.time() - t0
    build_results = {b: results[b] for b in range(B) if b in results}

    n_ok = sum(1 for b in build_results.values() if b.get("ok"))
    n_fail = B - n_ok
    print(f"[ORCH] all builds finished in {pool_wall:.1f}s: {n_ok} ok / {n_fail} failed", flush=True)
    if n_fail:
        for b in range(B):
            r = build_results.get(b)
            if not r or not r.get("ok"):
                print(f"[ORCH]   build {b} (seed {seed0+b}) FAILED rc={r.get('returncode') if r else 'n/a'}", flush=True)

    # ---- MERGE the shards (only ok builds contribute; shards kept either way) ----
    n_merged, per_demo, cam_counts = merge_shards(dataset_dir, build_results, B, E)

    # ---- symlink output/<dataset> -> /data3 dataset ----
    link, resolved = make_symlink(dataset_dir, dataset_name)

    # ---- SUMMARY (totals + per-build looks) -> summary.json + printed ----
    clean_success = sum(1 for d in per_demo if d["success"] and not d["penetrating"])
    total_abnormal = sum(1 for d in per_demo if d["penetrating"])
    grasped = sum(r.get("grasped") or 0 for r in build_results.values() if r.get("ok"))
    builds_meta = []
    for b in range(B):
        r = build_results.get(b, {})
        builds_meta.append(dict(
            build=b, seed=seed0 + b, gpu=r.get("gpu"), ok=bool(r.get("ok")),
            wall_s=round(r.get("wall", 0.0), 1), returncode=r.get("returncode"),
            distractors=r.get("distractors"), grasped=r.get("grasped"),
            placed=r.get("placed"), abnormal=r.get("abnormal")))
    summary = dict(
        dataset=dataset_name, dataset_dir=dataset_dir, output_symlink=link,
        merged_hdf5=os.path.join(dataset_dir, "demos.hdf5"),
        B=B, E=E, requested_demos=B * E, merged_demos=n_merged,
        builds_ok=n_ok, builds_failed=n_fail,
        clean_successes=clean_success, grasped=grasped,
        grasp_rate=round(grasped / n_merged, 4) if n_merged else 0.0,
        clean_success_rate=round(clean_success / n_merged, 4) if n_merged else 0.0,
        total_abnormal_penetration=total_abnormal,
        videos_per_cam=cam_counts, gpus=gpus, pool_wall_s=round(pool_wall, 1),
        builds=builds_meta)
    with open(os.path.join(dataset_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[ORCH] ===== SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ORCH] summary.json -> {os.path.join(dataset_dir, 'summary.json')}", flush=True)

    # ---- exit code reflects whether every build succeeded (the parent can gate the real run on this) ----
    if n_fail:
        print(f"[ORCH] WARNING: {n_fail}/{B} builds FAILED -- the dataset has only {n_merged} demos.", flush=True)
        sys.exit(1)
    print(f"[ORCH] OK: {n_merged} demos merged into {dataset_dir}.", flush=True)


if __name__ == "__main__":
    main()
