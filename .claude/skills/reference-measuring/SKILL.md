---
name: reference-measuring
description: Measure ratios and lines on a reference photo and on a render of our model with tools/refmeasure, then list the gaps. Use whenever a build follows a photo or reference image (lapels, collars, hair, boots, faces).
---

# Reference measuring

The habit, in order. Commands are `python3 tools/refmeasure/refmeasure.py <cmd>`; details and sources in
`tools/refmeasure/README.md`.

1. **Sheet.** `new IMAGE --view "<camera>" --projection photo`. Write the view honestly; compare checks it.
2. **Landmarks.** `grid IMAGE x0 y0 x1 y1` for a pixel-labelled zoom of each region, read coordinates off
   it, then `set SHEET name x y ...`. Use names from `names`. Same physical point in both images.
3. **Unit and plumb.** `unit SHEET a b --name head --axis y` (prefer a vertical span), `plumb SHEET name`.
4. **What matters.** `seg` for each span the build depends on (lapel width, shoulder width, neck to break,
   break to waist), `ratio` for each comparison John asked about (lapel to shoulder, lapel to neck).
5. **Read it.** `report SHEET`, `overlay SHEET`, and look at the overlay: every point on the right spot,
   the grid and plumb line where you expect. Write the key ratios into the thread before building.
6. **Model.** Render from the matching camera (long lens or ortho, same height and turn). Write the
   model sheet with `blender_landmarks.write_sheet(..., template=REF_SHEET)` or by `new` + `template` + `set`.
7. **Gaps.** `compare REF MODEL --png gaps.png`. Report every flagged row and attach `gaps.png`. Fix
   the cause, re-measure, and keep both sheets with the build.

Quick question about one pair: `probe SHEET a b c d` prints both spans in units and their ratio.
