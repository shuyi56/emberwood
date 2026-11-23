# Tension solver: the lapel and collar derived from the bare bodice's field

John, 2026-11-22: "Create a compression solver that reads our bodice without any lapels or collars then using the
tension as a guideline develop the ideal shape of lapels and collars to reduce tension."

This folder is self-contained so the solver consolidation thread can move it as one piece (to the shared solvers
folder). Nothing in `blender/collar/` imports it; it only reads `bodice_cut`'s CUT.blend and the builds of
`foldfront.py` and `notchcollar.py`, and writes foldfront options. `gauge.py` from `blender/zipboot/` is imported for
the Gaussian curvature map only (`derive.field_png`).

| File | Needs | What it does |
|---|---|---|
| `tension.py` | numpy, scipy | The solver. ARAP flattening (LSCM start), principal stretches and directions, the `Front` (one bare front, its best flat cut) and `Front.evaluate` (open the front on a roll line, turn the lapel onto it, score the strain). |
| `read_bodice.py` | Blender (bpy) | CUT.blend to an .npz: points, faces, front layers, measures, cloth thickness. |
| `derive.py` | numpy, scipy, matplotlib | The field (field.png, field.json), the roll-line search, the outline search, and the choice within the design's limits (derived.json with foldfront options). |
| `foldcollar_gauge.py` | numpy | The folded collar's crease (turn and sharpness on the roll line), body off the stand, and the share of its fall sunk in the coat's cloth. |
| `dump_build.py` | bpy | A built .blend (after foldfront and notchcollar) to an .npz for the contour sheet. |
| `contour_sheet.py` | numpy, matplotlib | Before against after: level, side and coronal sections. |
| `clay.py` | bpy | Cheap draft renders, clay or colour-coded by design part (fronts, each lapel, collar, buttons), on fixed ortho cameras with an off-centre `at`. |
| `montage.py` | Pillow | Before/after sheets of clay.py renders. |
| `landmarks.py` | bpy, tools/refmeasure | Projects a build's coat landmarks (roll tops, break, lapel points, collar points, shoulders, waist) into the front render and writes its refmeasure sheet. |
| `join.py` | numpy, scipy | The join between each lapel and the collar on a build: gap, step, turn across the gorge (plane-fit normals), seam kink, load on each side along the seam (evenness, peak, ratio), collar tip fold. |
| `collar_gauge.py` | numpy, scipy | The collar's ARAP flat cut and its stretches (join.py reads it). |
| `gauge_fold.py` | blender/zipboot/gauge.py | The sharpest dihedral of a patch. |
| `sweep.py` | foldfront, notchcollar | Builds one lapel and a grid of collars on it, each gauged; the coupled sweep's builder. |
| `rollcrease.py` | numpy | The angle between the collar's fall and its roll turn over the fronts: the crease a roll-top change leaves when the collar is not re-solved. |
| `polycheck.py` | numpy, scipy, contourpy, PIL | The polygon check against the riding-habit photo: finds the corners of the lapel and the collar front from the photo's camera by itself (depth buffer for the collar), fails on a wrong side count (lapel 4, collar front 4), then lists every edge off in direction (12 deg), share of the outline (a third) or bow, and every corner off in angle (15 deg) or rounding radius. `--photo` remeasures the photo's curvature. The contour sheet draws it. On the old triangle lapel (b/HF1) it fails: lapel 3 sides of 4. |
| `lapel_inverse.py` | numpy | The inverse of the lapel's bend: the cut corners that land on the photo's corners, from the solved lapel's cut-to-view map. |
| `flatview.py` | matplotlib | The flat cut beside the outline from the photo's camera, each side marked as a cut edge or an edge made by the fold or the view. |
| `feasible.py` | matplotlib | Stage 1 and 2 of the coupled solve: the limits (the approved build's own numbers), the feasible map, the join objective. |

## Run

```
python3 blender/tension/read_bodice.py CUT.blend BARE.npz
python3 blender/tension/derive.py BARE.npz OUT            # about 20 min; --quick about 4 min; --reuse redraws
python3 blender/collar/foldfront.py CUT.blend LAPEL.blend '<foldfront_options from OUT/derived.json>'
python3 blender/collar/notchcollar.py LAPEL.blend BUILD.blend '{}'
python3 blender/tension/dump_build.py BUILD.blend BUILD.npz
python3 blender/tension/contour_sheet.py BEFORE.npz BUILD.npz SHEET.png
```

`derive.py` on the cut of 2026-11-22 chooses a straight roll line (`belly` 0.001), top at 2.1, break and swell as
approved, point 0.30, outline width 0.35 (the area held). Built and measured, the field's small moves on the top,
point and width did nothing for the strain and narrowed the lapel (refmeasure: reach -5%), so the build this thread
settled on keeps everything approved but the line, with the outline widened just enough to keep the reach (collar
unchanged):

```
python3 blender/collar/foldfront.py CUT.blend LAPEL.blend '{"recipe":"derived_lapel"}'
```

The coupled lapel-collar solve (phase 2): build a sweep, map it, pick the join.

```
python3 blender/tension/sweep.py CUT.blend SWEEP w42_p24 '{"recipe":"joined_lapel"}' '{"notch_angle":[30,40,50],"notch_at":[0.35,0.5,0.65],"point_len":[0.55,0.7,0.85]}'
python3 -I blender/tension/feasible.py SWEEP OUT [--field preL.jsonl]     # OUT/feasible.png, OUT/sweep.json, best feasible joins
```

The chosen pair of 2026-11-23: `{"recipe":"eased_top_lapel"}` (phase 3: `joined_lapel` with the roll line's top eased,
`belly_top` -0.14) with notchcollar `{"notch_angle":30,"rise_gain":1.6,"end_bow":-0.2}` (phase 4: the collar's stand re-solved with
the roll line; phase 5: the collar end's shape, `end_bow`; see the retrospective).

Run Blender-module scripts with `python3 -I` when the working directory holds a file named like a standard module.

## Method, and where each piece comes from

1. **The best flat cut of the bare front.** Each front (closing edge to x = 3.5, waist to neck) is cut with its two
   darts slit open where `redingote/bodice.py` places them, and flattened by the as-rigid-as-possible
   parameterisation (Liu, Zhang, Xu, Gotsman & Gortler 2008, "A local/global approach to mesh parameterization":
   local step the SVD rotation of each triangle's Jacobian, global step the cotangent Laplacian, factored once),
   started from a least-squares conformal map (Levy et al. 2002). This is the energy-model flattening used for
   garment patterns (Wang, Smith & Yuen 2002, "Surface flattening based on energy model", CAD 34; their acceptable
   error is about 1% in length and area).
2. **The field.** Per triangle, the singular values of the flat-to-3D map: s1 > 1 is cloth pulled longer than it
   was cut (tension), s2 < 1 is more cloth than room (compression, what a tailor shrinks or a tape takes up), with
   their directions. By tension field theory a wrinkle or a turn runs along the tension direction where the other
   stress is compressive (Wong & Pellegrino 2006, "Wrinkled membranes" part II: "the wrinkles are assumed to be
   aligned with the major principal stress directions").
3. **The lapel turned.** For a roll line drawn exactly as foldfront draws it (top N on the neck at `neck_x`, break
   B on the front's edge at `foot`, swing `belly`), the front is re-flattened with the line as a free edge, the
   lapel outline is cut past the line on that pattern, mirrored over it, and laid on the front one cloth's thickness
   out. Its principal stretches are the strain the turned cloth must take. The lapel's area is held at the approved
   design's, so a candidate cannot win by being smaller.
4. **The limits** are the design's and the tailor's, not the field's (see `derive.py`): the roll-line top within one
   stand height of the side neck point, the point within the traced reference's band, the swell at most the
   approved one. Every choice is then built with foldfront and notchcollar and measured; the build decides.
