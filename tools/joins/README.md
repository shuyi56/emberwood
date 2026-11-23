# Join diagnostics

How smoothly two surfaces run into each other where they are joined (neck into head, shin into instep, ear into
head, hair roots into the scalp, a boot's sole into its upper, a lapel's roll), measured the way class-A surfacing
measures a patch boundary: **G0** (do they meet), **G1** (do their tangents agree), **G2** (do their curvatures
agree). The numbers and the limits are in `blender/shared/continuity.py`, which is numpy only and runs both inside
Blender and outside it. The foot study's top-line smoothness check uses the same function
(`continuity.polyline_curvature`, `continuity.curvature_steps`).

| File | What it does |
| --- | --- |
| `blender/shared/continuity.py` | the measurement: plane-mesh sections, one-sided fits, G0/G1/G2 per station, principal/mean/Gaussian curvature, normal deviation, limits and their sources |
| `blender/shared/join_sheet.py` | the contour sheet (PNG) of a join or a profile line |
| `tools/joins/joins.py` | the CLI: `join`, `profile`, `preset` |
| `tools/joins/dump.py` | built scene (`.blend`, or the live build) to an `.npz` of meshes |
| `tools/joins/presets.json` | the joins the project checks, by mesh name |

## Quick use

```
# 1. a built scene: views.py --save writes the .blend (no render), dump.py turns it into meshes
python3 blender/base/views.py --figure barefoot --mannequin --head --save /tmp/foot.blend
python3 tools/joins/dump.py /tmp/foot.blend /tmp/foot.npz "Skin body" "Head shape"

# 2. the checks
python3 tools/joins/joins.py preset neck foot-top --npz /tmp/foot.npz --out /tmp/joins
python3 tools/joins/joins.py join --npz /tmp/foot.npz --a "Head shape" --b "Skin body" --near 0,0,53.2 --out /tmp/j
python3 tools/joins/joins.py profile --npz /tmp/foot.npz --a "Skin body" --plane x=-2.8 --polar -2,1,-5,115 \
    --clip "z<6,y<1" --skip-ends-cm 2.5 --joins-at kmax:y:-2.6:-0.6 --out /tmp/j
```

Each join gives `join-NAME.json` and `join-NAME.png`. Exit code 1 when a join fails (`--report-only` to always
exit 0).

As a pre-render check in the queue (report-only, on the build, before any render):

```
python3 /mnt/project-files/render-queue/rq.py add --thread <id> --branch <branch> --sha <sha> --outputs views \
    --view "blender/base/views.py --figure barefoot --mannequin --joins neck,foot-top --joins-only" \
    --quality check --note "join continuity"
```

`--joins NAMES` (or `all`, or `RQ_JOINS=NAMES` for every job) runs the presets whose meshes are in the scene and
adds a `joins` block to `geometry.json`. `--joins-only` stops after the checks, so the job costs one build and no
render. With `RQ_PRECHECK='{"joins": true}'` a failing join stops the job before it renders (`precheck.json` says
which).

## Kinds of join

* **intersect** (two meshes, one passes into the other): the join is the line where A crosses B (the zero line of
  A's signed distance to B). Only the visible part of each side is measured, so the result is the crease you see.
  G0 is closed by construction there and is reported, not graded.
* **boundary** (two meshes, A's open edge meets B's): A's section runs to its end, B's starts at its own; G0 is the
  gap between them.
* **seam** (one mesh): the join is a line you name on it (`--seam-plane ox,oy,oz,nx,ny,nz` within `--clip`), for
  a piece that was united or blended along a known line.
* **profile** (one line, no marked join): the line cut by a plane, scanned along its length for tangent and
  curvature steps; `--joins-at` names where a join lies on it (cm from the low end, or `kmax:AXIS:LO:HI`: the
  sample of greatest |k| with that coordinate in range), and the report says whether that join falls on a step.

## What is measured

At each of 24 stations along the join the surface is cut by the plane square to it. The section is fitted on each
side of the join on its own, a quadratic over the window (default 6 edge lengths, at least 0.25 u), and the two
fits are compared at the join:

* **G0**: the gap between the two sides, mm.
* **G1**: the angle between their tangents, degrees.
* **G2**: their curvatures kA, kB (1/cm) and the curvature deviation |kA - kB| / (|kA| + |kB|) (Alias's
  (R1 - R2)/(R1 + R2)), the denominator floored at two 1-head-height circles so two near-flat sides do not read as
  a 100% step.
* **noise**: each fit's 1-sigma error from its residuals; a step under 3 sigma is the mesh's facets and is not
  graded.
* **control**: the same numbers 1.5 windows to each side of the join. The cause line compares the two: a step that
  is more than twice the control's is the join's own (**modelling**: the two sides are shaped apart); one as large
  away from the join is the surface's (**tuning**: smoothing or resolution), or the **solver**'s when the piece was
  solved (`--solved`). A G0 gap is always modelling (the parts are not welded where they meet).
* **surface**: per-vertex principal curvatures k1, k2, mean H and Gaussian K (a quadric over the 2-ring), the
  medians in a band on each side (1/cm, 1/cm^2).

## The limits and where they come from

| | pass | fail | source |
| --- | --- | --- | --- |
| G0 | 0.07 mm (glossy 0.01) | 0.17 mm | Autodesk Alias, *Continuity 2: Construction Tolerances*: production surfaces 0.01 to 0.001 mm. For a rendered mesh, 0.1 px and 0.25 px at the queue's closest camera (the face crop: 10 u over 384 px, 0.68 mm/px at 26.19 mm/u) |
| G1 | 0.57 deg (glossy 0.1) | 4 deg | Alias: "a tangent break of less than 0.1 degrees will normally not be visible on most products" (glossy). Matte: a normal step dn changes Lambert shade by dI/I = tan(theta) dn, about dn under the queue's 45 deg key; a luminance edge is seen at about 1% Weber contrast (Blackwell 1946), so 0.01 rad = 0.57 deg. Fail at 4 deg, the lowest orientation difference observers judge even between separate patches (Norman, Todd, Norman, Clayton & McBride 2006, Vision Research 46:1057, "between 4 and 10 deg") |
| G2 | 0.10 | 0.23 | Phillips, Todd, Koenderink & Kappers 1997 (JEP:HPP 23:1481): curvature differences are discriminated at "on the order of 10-20%"; Norman et al. 2006 Table 2: 23% the best ordinal curvedness threshold. A curvature step kinks the shade's gradient, which the eye sharpens into a Mach band, so the low end is the pass |

The model unit is 26.19 mm (the figure's 63 u to a 165 cm woman, the scale the foot study uses); `mm_per_u` changes
it. The G2 numbers are from discrimination between patches, not from a step at an edge, which is likely seen
sooner: treat the G2 limits as a first calibration and tighten them with the logged rejections.

## The contour sheet

* Top row: sections at three stations (the worst among them), side A blue, side B orange, the join a black dot,
  with the curvature comb (teeth along the normal, length = curvature). A G2 join's comb outline runs through the
  dot with no step; a G1 join's comb jumps; a crease is a corner in the section itself.
* Middle row: G1 and G2 along the join with the pass (green) and fail (red) lines, the control in grey, and kA, kB.
* Bottom row, from one orthographic camera looking at the worst station: the normal-deviation heat map (the angle
  between each vertex normal and the smoothed normal field, grey 0 to red 4 deg; the join drawn by grade),
  zebra stripes (reflection lines: cut by a G0 break, kinked by G1, sharply bent by G2) and a clay matcap.

For a profile: the line with its comb and the steps marked, k(s), G1 and G2 along s, dk/ds, and the three views.

## Adding a join

Build the figure, list its meshes with `dump.py`, and add an entry to `presets.json` with the mesh names. Only add a
preset once its names are checked on a build; a preset whose meshes are not in a scene is skipped by the queue.
