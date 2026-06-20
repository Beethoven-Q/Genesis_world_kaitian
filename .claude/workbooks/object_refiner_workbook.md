# Object-refiner workbook — accumulated refinement experience

Append-only experience store for the **object-refiner** agent (`.claude/agents/object-refiner.md`). Per object /
category: the density + collision + keypoint choices that produced a sim-ready asset, the geometry-probe recipes
that found a feature, and any hard corner. Over many objects this becomes a deep playbook so refinement trends
toward autonomy. Always write concrete numbers (the measured volume, the chosen density, the verdict values) so
the evidence is traceable — never a vague "looked fine".

---

## Category density priors (the INFER knowledge; registry/refine.py CATEGORY_PRIORS)
| category | density kg/m^3 | friction | restitution | typical objects |
|----------|----------------|----------|-------------|-----------------|
| ceramic  | 2400 | (1.0, 0.9) | 0.05 | mug, plate, bowl |
| glass    | 2500 | (0.9, 0.9) | 0.05 | jar, tumbler |
| wood     | 700  | (1.1, 0.9) | 0.10 | book, block |
| plastic  | 950  | (0.9, 0.9) | 0.15 | marker, cup |
| metal    | 7800 | (0.8, 0.9) | 0.10 | can, tool |
| rubber   | 1100 | (1.2, 1.0) | 0.45 | ball, eraser |
| fruit    | 950  | (1.0, 0.9) | 0.10 | apple, banana |

Note the mass basis is the **voxel-filled SOLID volume** (robust to open meshes), so for a thick-walled hollow
object the inferred mass counts the whole wall+interior volume of the material — it lands on the HEAVY side of a
real empty hollow object (the mug came out 0.57 kg vs a real ~0.35 kg empty ceramic mug). Acceptable for the MVP
(within plausible range, and a heavier mug is harder to nudge = stable). To trim, override `density` lower or
subtract the cavity volume — a future refinement.

---

## Per-object log

### mug (YCB `ycb/mug.usd`)  — REFINED + sim-ready 2026-06-19
- **Source.** RoboLab YCB `mug.usd` (copied into `genesis_firefly/assets/objects/ycb/mug.usd`). Rigid, hollow.
- **Augment.** verts 16763, NOT watertight (open cup mouth — expected). extents (0.117, 0.093, 0.081) m. solid
  volume 237 cm^3 (voxel-fill; trimesh.volume is n/a on the open mesh — voxel-fill is the right tool). hull 524,
  bbox 886 cm^3. hollowness 0.55. PCA long axis (-0.40, 0.02, -0.91).
- **Infer.** category ceramic -> density 2400 -> mass 0.569 kg. friction (1.0, 0.9). restitution 0.05.
- **Collision.** convex DECOMPOSITION (coacd, threshold 0.04) -> **34 convex hulls**. The handle ring + cup
  cavity stay OPEN in the gaps between hulls (a single hull would FILL them -> a branch could never thread).
  Clean Nyx-safe visual `ycb/mug_clean.obj` extracted (Nyx segfaults on the textured USD).
- **Geometry probe (how the handle hole was found).** First guesses had the X SIGN WRONG (a vert-count shell
  test mis-attributed the cup wall as the handle). The fix: rasterize an **X-Z silhouette** (view along Y) ->
  the handle clearly protrudes on **+X**, with a visible empty hole. Then an **encircled-hole search** (max
  Y-line clearance whose neighbourhood occupies >=7 of 8 angular sectors) found the TRUE hole: center
  (0.0395, 0, 0) local, radius ~10.5 mm, 8/8 sectors. Lesson: a max-clearance point at the mesh CORNER is a
  false positive (only 2 sectors) — always require encirclement.
- **Keypoints.** `handle_ring` center (0.0395, 0, 0), normal (0,1,0) (= the branch axis to hang on), radius
  0.0105. `cup_opening` center (-0.012, 0, 0.040), normal (0,0,1), radius 0.045.
- **VERDICTS (real run).** penetration_at_rest PASS (abnormal 0/1, max 0.04 mm). init_stability PASS (D_pos
  0.03 mm, D_ori 0.0012 rad, no explosion). hollow_feature_open PASS (probe<->object overlap **0.00 mm** with a
  4 mm probe in the 10.5 mm hole -> a branch threads it cleanly). SIM-READY: YES.
- **Render.** `genesis_firefly/output/temp/mug_refine_verify.png` — the mug resting stable, the cup mouth open,
  and a brown branch threaded through the open handle ring (the hollow proof, visual + numeric).
- **Hard corner.** The probe overlap was a borderline 6.2 mm on the FIRST (wrong) keypoint; the correct hole
  center dropped it to 0.00 mm. Getting the keypoint on the TRUE open hole (not the handle bar) is what makes
  the hollow proof unambiguous — measure before you place the keypoint.
