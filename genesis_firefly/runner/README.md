# runner/ — collection entry points

Two entry points scale the cube→bowl (and every future) collector. The collector itself
(`tasks/pickplace.py`) is **reused verbatim** by both — neither duplicates collection logic.

| script | what it runs |
|--------|--------------|
| `collect.py` | **ONE** build: one `BuildDR` (one "look") × `N` parallel envs → `demos.hdf5` + videos in `DATA_DIR`. |
| `orchestrate.py` | **B** fresh-subprocess builds × `E` envs → `B×E` demos with ≈B distinct looks, merged into one dataset. |

---

## `orchestrate.py` — the build-batch orchestrator

```bash
./.venv/bin/python genesis_firefly/runner/orchestrate.py B E [seed0] [dataset_name]
#   B            = number of builds (each a fresh collector subprocess; seeds seed0, seed0+1, …)
#   E            = envs (demos) per build              → B×E demos total
#   seed0        = first build seed (default 0)
#   dataset_name = dataset folder name  (default fulldr_orch)
# env: GPUS=0,1   override the GPU pool (default = all GPUs nvidia-smi reports)
#      DATA_ROOT  default /data3/genesis_fulldr        SPP   forwarded to the collector
```

Example (the small correctness test): `orchestrate.py 2 2 900 orch_test` → 2 builds × 2 envs = 4 demos.

### Why subprocesses (mandatory)
A **2nd `gs.Scene` in one process SEGFAULTS** — Nyx/Vulkan is single-shot. So each build is a **fresh
subprocess** of `collect.py`. This is also what gives the cross-build DR variety: COLOR / TEXTURE / SIZE / object
TYPE bake at build time (the hard renderer constraint, see `docs/domain_randomization.md`), so each build's
seed draws ONE "look" (table texture, object colors/sizes/type, table size, side-cam pose, distractor SET), and
B builds = B distinct looks. Within a build, the E envs vary the per-env space (pose / mass / friction / HDRI /
light + the 50/50 distractors + ~50/50 L/R arm).

### The GPU pool (one sim process per GPU — owner rule)
The orchestrator detects the GPU pool (`nvidia-smi --query-gpu=index`, or the `GPUS=0,1` env override) and runs
a **worker pool of size = n_gpus**. Each worker pulls the next build off a queue, runs it with
`CUDA_VISIBLE_DEVICES=<gpu>` pinned, waits for it, then pulls the next. **At most one build per GPU
concurrently.** Each build's key `[COLLECT]` lines (grasped / placed / penetration / distractors) are relayed
live, tagged `[b<b> g<gpu>]`.

### Shard → merge → symlink (the storage flow)
```
/data3/genesis_fulldr/<dataset>/
  ├─ _shards/build_<b>/            ← each build writes its OWN demos.hdf5 + videos/ + tiles/ here (KEPT)
  │    ├─ demos.hdf5               (DATA_DIR for that subprocess)
  │    └─ videos/cam_{side,lw,rw}/demo_<i>.mp4
  ├─ demos.hdf5                    ← MERGED: every shard's /data/demo_<i> copied to /data/demo_<g>,
  │                                  g = 0..B*E-1, ALL datasets + ALL attrs preserved (h5py group copy)
  │                                  + a `build` attr added per demo
  ├─ videos/cam_{side,lw,rw}/demo_<g>.mp4   ← copied + renumbered to match the merged keys
  └─ summary.json                 ← totals + per-build looks
output/<dataset>  ──symlink──►  /data3/genesis_fulldr/<dataset>      (in-workspace preview)
```

- **Merge** uses `h5py.File.copy` on each `/data/demo_<i>` group → the whole subtree (actions, states,
  ee_pose) **and all attrs** (success / seed / arm / hdr / has_distractors / distractors /
  max_penetration_mm / penetrating) ride along untouched; only a `build` attr is added. Source demos are
  iterated in **numeric** order (demo_0, demo_1, … not lexical). Videos are copied+renumbered to match `g`.
- **Shards are KEPT** (never deleted) so a failed merge is recoverable. The dataset *is* the merged file.
- **`/data3` writability is gated**: if the dataset root isn't writable the orchestrator **FAILS LOUDLY**
  (exit 3) rather than silently writing into the repo.
- If a build subprocess fails, it is **skipped** in the merge, the failure is reported, and the orchestrator
  exits non-zero (1) so the merged demo count < B×E is visible.

### Summary
`summary.json` (printed too) carries: requested vs merged demos, builds ok/failed, **clean successes**
(`success==True` ⇒ placed AND not penetrating), grasp rate, **total abnormal-penetration count** (the collision
gate), per-cam video counts, and the **per-build looks** (each build's distractor set + grasp/place/abnormal
counts + seed + GPU + wall time).

### Recommended B×E for a 200-demo run
Total demos = B×E; wall time ≈ `ceil(B / n_gpus) × per-build-time`. A 16-env 640×360 build ≈ 190 s. With
**2 GPUs**, e.g. **B=10 × E=20** = 200 demos in ~5 waves ≈ 16 min and gives 10 distinct looks; **B=20 × E=10**
= 200 demos with 20 looks in ~10 waves (more look-variety, longer). Prefer **more builds** (more looks) up to the
point where E is still big enough to amortize the ~30–40 s per-build build/settle overhead — E≈10–20 is the
sweet spot. Keep `E ≤ 45` so HDRIs stay 2K (see `docs/domain_randomization.md`).
