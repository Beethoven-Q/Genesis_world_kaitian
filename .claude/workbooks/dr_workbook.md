# DR workbook — the DR strategist's append-only experience store

This is the mechanism by which the **dr-strategist** subagent (`.claude/agents/dr-strategist.md`) gets more
professional over time. One section per `(task, object)`. Each section has three parts:
1. **Current best ranges** — a table of each DR axis: the range in force + a **confidence** (widens as evidence
   grows). This is what the strategist recommends to the main agent before a collection.
2. **Experience log** — append-only, dated entries (run → observation → change → why → result).
3. **Known hard corners (accept / label / keep)** — edge poses/values that are legitimately hard; parallel
   collection ACCEPTS them as failure-recovery data rather than excluding them.

> **HARD rule:** the strategist ONLY appends here; it never edits the ranges in code. It proposes range edits;
> the main agent applies them to `tasks/pickplace.py::sample_phys_dr` / `world/manipulation_stage.py`.
> Always write concrete numbers (probe clean-rate, achieved diversity extent, failing envs' `dr_clr`/`dr_reach`)
> so the evidence is traceable.

Confidence legend: **LOW** (proposed/untested) · **MED** (one clean run or one probe) · **HIGH** (multiple
clean runs / probed both directions) · **MASTER** (probed to the failure edge; the corner is mapped).

---

## Task: cube→bowl pick-place  ·  objects: coloured cube + convex-decomposition bowl  ·  distractors: {pen, banana, apple, tennis_ball, book}

### Current best ranges (the v2 baseline = `DR_POSE_SCALE/MASS/FRIC = 1.0`)
Per-env physics DR lives in `tasks/pickplace.py::sample_phys_dr`; per-build visual DR in
`world/manipulation_stage.py`. The half-width is what `DR_*_SCALE` multiplies (centre fixed).

| axis | scope | range (v2, scale=1.0) | variability | confidence | notes / coupling |
|---|---|---|---|---|---|
| arm side (L/R) | B | per-env ~50/50 | per-env | HIGH | breaks the "always one-side-of-bowl" coupling; keep balanced |
| object-table height | A | ho ±5 cm | per-env | HIGH | orthogonal to pose; composes with every z |
| bowl x | B | 0.40 ±0.05 m | per-env | HIGH | centre 0.40; widen via `DR_POSE_SCALE` |
| bowl y | B | sgn·(0.05 ±0.035 m) | per-env | HIGH | arm-signed; keep sgn (anti-coupling) |
| cube x | B | 0.40 ±0.06 m (v2); **±0.078 m probed clean @scale 1.3** | per-env | HIGH | widen via `DR_POSE_SCALE` → raises `dr_reach`; 1.3 stayed clean (2026-06-19) |
| cube y | B | sgn·(0.185 ±0.05 m) (v2); **±0.065 m probed clean @scale 1.3** | per-env | HIGH | arm-signed; the main MAX-extent axis to push; probe 1.5–1.6 next for the reach edge |
| cube yaw | B | ±90° (full in-plane) | per-env | HIGH | already MAX (orientation-aware grasp folds it) |
| cube↔bowl clearance | B | ≥0.17 for ~80%, ≥0.125 hard floor | per-env | MASTER | **floor is load-bearing — never relax** (see hard corners) |
| cube mass-shift | B | ±0.02 kg | per-env | MED | widen via `DR_MASS_SCALE`; high end stresses grasp |
| link friction ratio | B | 1.0 ±0.3 | per-env | MED | widen via `DR_FRIC_SCALE`; low end → slip |
| distractors | B | K∈{2,3}, 50/50 present | per-build/env | HIGH | corridor-placed, never swiped (≤1.2 cm disp) |
| disturbance | harness | prob ~0.34 in-plane shove | per-env | HIGH | failure-recovery signal; `DISTURB=0` off |
| table texture | A | ≥10-map pack (wood/steel/cloth) | per-build | HIGH | stage-owned; always on |
| table/object colour | A/C | realistic, distinct-from-table | per-build | HIGH | stage-owned; cube colour free |
| HDRI background + light | C | 2K pool (E≤45) + per-env light | per-env | HIGH | stage-owned; immersive room per env |

**Selected scope-B fields for this task:** cube pose (xy+yaw), bowl pose (xy), cube mass, cube/link friction,
distractors. (Scopes A + C are automatic.)

### Per-object COLOUR / TEXTURE policy (DR-strategist-OWNED; see the contract "Per-object colour/texture policy")
The GRASP-TARGET's appearance is a per-object DR decision the strategist owns. Each object is one of three
classes: **NATIVE TEXTURE** (the object's own UV-mapped skin — preferred when usable; no colour DR), **REALISTIC
PALETTE** (a `target_palette` of realistic hues + ±0.04 jitter per demo, when no usable texture but a real colour
range exists), or **FIXED** (a single-entry palette for a regulation/canonical colour). The main agent reflects
the decision in `registry/object_spec.py` (`native_texture` / `target_palette` / `color`, commented as
DR-strategist-owned). A NEW object gets classified here BEFORE it is collected.

| object | class | render source | colour DR | evidence / rationale |
|---|---|---|---|---|
| **apple** | NATIVE TEXTURE | `objaverse/textures/apple_02.png` via `native_texture` | **none** (texture IS the colour) | `apple_clean.obj` fully UV-mapped (898 `vt`, all 1558 faces ref UVs); 1024² real apple texture (natural red mottling + stem); **renders in Nyx, no segfault** (probed 2026-06-20). Replaced the flat pink palette that overrode the real skin. |
| **banana** | REALISTIC PALETTE | clean-mesh + `target_palette` | yellow×2 / green (unripe), ±0.04 | `banana_clean.obj` has **0 `vt`** (no UVs) → a texture can't map onto the Nyx-safe mesh; YCB USD segfaults Nyx. Palette yellow looked GOOD to the owner. NEVER pink/blue. |
| **pen** (dry-erase marker) | REALISTIC PALETTE | clean-mesh + `target_palette` | black / blue / red, ±0.04 | `dry_erase_marker_clean.obj` has **0 `vt`** (no UVs). Normal marker colours only. |
| **tennis_ball** | FIXED | procedural sphere + 1-entry palette | **none** (regulation) | regulation yellow-green felt `(0.82,0.92,0.22)`; single-entry palette → effectively fixed. Special-coloured object → no DR. |
| **cube** | FREE random | procedural box + free distinct-from-table colour | full free hue | a generic shape with NO real-world colour → the only free-random target; keeps the cube collection byte-for-byte. |

**Rule of thumb for a new object:** prefer NATIVE TEXTURE if the clean .obj carries UVs (`grep -c '^vt '` > 0) AND
the texture renders in Nyx (probe it — the USD often segfaults, the clean mesh may lack UVs); else a REALISTIC
PALETTE (realistic hues only — no blue watermelon, no oversized); FIXED for a regulation colour; FREE only for a
generic colourless shape. Record the UV count + the Nyx render check as evidence.

### Experience log
- **2026-06-19 — v2 baseline (B=10 × E=20 = 200).** `/data3/genesis_fulldr/cube_fulldr_v2`:
  **200/200 clean** (grasp 1.0, place 1.0), **0 abnormal penetration**, arms L/R = 103/97. The full-DR ranges
  above are proven CLEAN at scale. → all axes at confidence ≥ MED, pose/yaw/clearance HIGH.
- **2026-06-19 — clearance-floor root-cause (carried from `docs/roadmap.md`).** At `clr=0.17`, ~6% of
  cube-in-bowl spawns are geometrically INFEASIBLE (cube circumradius ~3.5 cm + bowl radius ~7.5 cm ≈ 11 cm
  needed); the rejection loop could EXHAUST its 60 tries and SILENTLY ship an overlapping cube → the firm
  solver ejected it off the (ground-plane-less) table → a 6 m garbage waypoint → a 10× trajectory blowup.
  **Fix (in code, not a range):** after rejection, radially CLAMP any still-bad env's cube to exactly the
  0.125 m hard floor → cube-in-bowl is impossible by construction. **Lesson for the strategist:** the
  clearance floor is load-bearing; widening pose is fine BUT the floor must stay — `dr_clr` will ride 0.125
  more often as pose widens, and that is acceptable hard-edge data.
- **2026-06-19 — DR-strategist MVP layer built (this loop).** Added the `DR_POSE_SCALE/MASS/FRIC` sweep hooks
  to `sample_phys_dr` (default 1.0 → v2 reproducible), the per-demo DR-plan HDF5 attrs (`dr_*`), and the
  `dr/sweep.py` probe. Verified default reproducibility (the `1.0` run still gets 8/8 placed, 0 penetration,
  max|dq|=0.098).
- **2026-06-19 — explore→measure→record: widened the POSE axis (`DR_POSE_SCALE` 1.0 → 1.3).** Probed both with
  `dr/sweep.py --envs 12 --seed 7`:
  - **baseline pose=1.0** → clean **11/12 (92%)**, stayed-clean **YES** (penetrating 0, degenerate 0,
    max_pen 5.1 mm). The 1 non-clean env (#8) was a grasp **MISS** (not a penetration/degenerate), and the
    clustering put it at the **high `dr_cuby` / high `dr_clr`** end (z=+1.28 / +1.31) — a far-side, comfortably-
    clear grasp that just missed, i.e. NOT a range-safety failure. Achieved diversity: cube-x extent 0.118,
    cube-y 0.466, cube↔bowl clr 0.076 (floor 0.126).
  - **widened pose=1.3** → clean **12/12 (100%)**, stayed-clean **YES** (penetrating 0, degenerate 0,
    max_pen 2.6 mm). Diversity GREW with the range: cube-x extent **0.118 → 0.158**, cube-y **0.466 → 0.495**,
    cube↔bowl clr **0.076 → 0.096** (floor touched exactly at **0.125** — the clamp held, no cube-in-bowl).
  - **Finding / delta:** at +30% pose breadth the batch **stayed clean (0 pen / 0 degen)** and became **more
    representative** (wider achieved coverage on every pose axis), with **no penetration regression**. The
    clearance floor held (envs rode 0.125 more, as expected from the coupling). At E=12 the +1 clean delta
    (92%→100%) is within noise (the baseline miss was a non-penetrating grasp miss), so the **safety** evidence
    is strong but the **success-rate** improvement is not yet significant.
  - **Recommendation (for the main agent to apply):** pose breadth has clear headroom — adopt **`DR_POSE_SCALE`
    ≈ 1.3** for cube→bowl (cube-y half-width 0.10 → 0.13 m, cube-x 0.12 → 0.156 m, bowl xy similarly), keeping
    the 0.125 m floor. Then re-probe **1.5–1.6** to find where `dr_reach` finally degrades the grasp (the
    expected MAX-extent limiter) before a full re-collection. Confidence on pose range: **MED → HIGH** for 1.3
    (probed clean both directions). NOT yet applied to `sample_phys_dr` — proposed only.

- **2026-06-20 — DR-strategist takes OWNERSHIP of the per-object colour/texture policy + 3 object improvements.**
  The colour-policy table above is now the canonical per-object appearance decision (native-texture / realistic-
  palette / fixed / free), owned here. Concrete changes this run (main-agent applied to `object_spec.py`, verified
  by REAL renders, DISTURB=0, seed 7):
  - **apple → NATIVE TEXTURE.** The flat pink `target_palette` overrode the apple's real skin. `apple_clean.obj`
    is fully UV-mapped (898 `vt`) to `apple_02.png`; rendering it via `gs.textures.ImageTexture` (the table-top
    idiom) gives a realistic textured apple (verified in Nyx, no segfault). `target_palette` kept as the
    `NATIVE_TEX=0` fallback. Result E=12: **12/12 placed, 0 abnormal pen, max 4.4mm** (texture render proof saved).
  - **banana / pen → kept REALISTIC PALETTE** — their clean .obj meshes have **0 `vt`** (no UVs), so a texture
    can't map onto the Nyx-safe mesh; the YCB USD segfaults Nyx. (Owner already liked the banana yellow.)
  - **tennis_ball → kept FIXED** regulation yellow-green (single-entry palette, no DR).
  - **deeper-grasp tune (collision-#1 stays 0 abnormal):** apple `grasp_dz 0→-0.006` (cradles lower in the curved
    GR100 claws; peak pen DROPPED 5.9→4.4mm). pen `grasp_dz 0→-0.004` + `grasp_close 0.9→0.78` (the thin pen's
    firm pad-near-pad clamp over-bit it: E=24 stock **4/24 abnormal @8.1mm → 0/24 @6.5mm**; E=12 real 12/12,
    0 abnormal, max 6.9mm — the pen rides near the 7mm gate, the hardest penetration case). tennis/banana deeper
    seats REGRESSED (tennis place 11/12, banana 2/12 over-pen) → kept at their stock seat (already 12/12 @ ~6.5mm).
  - **regression:** CUBE unchanged (no palette/texture/grasp_dz change) — E=8 real **8/8 placed, 0 pen (2.9mm),
    posture natural** (|j4|≤1.40, elbow≥1.14). Confidence on the colour-policy classification: **HIGH** (each
    class proven by a real render).

### Known hard corners (accept / label / keep)
- **the PEN (thin ~2cm dry-erase marker) is the framework's hardest PENETRATION case.** The firm GR100 pinch
  clamps pad-near-pad (the dofs clamp at `GR100_MEET=0.58`) on the thin body, so the high-kp PD wants to over-
  bite it. `grasp_close=0.78` + `grasp_dz=-0.004` is the sweet spot (E=24 0/24 abnormal, max 6.5mm), but a
  far-reach env can still nick ~6.9–7.3mm at E=12 — it RIDES the 7mm gate. `grasp_close` is **non-monotonic**
  (≤0.76 is WORSE — the gentle target lets the body shift into a deeper bite); a deeper seat at the FULL close
  over-bit it (4/24 abnormal). **Accept** the occasional gate-edge env as the pen's hard corner; do NOT chase it
  with a deeper seat or a much gentler close. This is a GRASP-depth limit, not a DR-range limit.
- **cube↔bowl tight clearance (`dr_clr` ≈ 0.125 m floor).** The hardest ~20% spawns sit near the floor; as
  pose widens, MORE envs ride it. The open gripper nearly grazes the bowl on the grasp descent. **Keep** —
  this is exactly the close-quarters recovery data the policy needs. Never relax the floor.
- **far-reach edges (`dr_reach` high).** Cube xy at the far-x / far-y corners of the arm's reachable
  workspace is the usual MAX-extent limiter: past it the grasp orientation degrades. **Probe before
  committing** a pose widening; accept marginal-reach envs as hard data, but stop the range where success
  stops being usable.
- **degenerate spawn (handled in code, label-only).** A cube ejected off the table at spawn is flagged
  `degenerate` and REJECTED (never shipped); the strategist should see `degenerate=0` on a healthy run and
  treat any non-zero count as a spawn-feasibility regression, not a range to chase.
