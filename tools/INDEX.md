# Solvers and gauges: the index

**Start every project and every thread here.** Read the decision guide for your kind of work, take the solvers and
gauges it names, and read their entries before you build anything. A gauge you have not run cannot catch the problem
it was written for, and every one of these was written after a problem cost rounds.

- **Solvers** find a shape: a fold, a drape, a strand's fall, a heel's height, a lacing's proportions.
  Docs: [`solvers/README.md`](solvers/README.md).
- **Gauges** measure a shape and say pass, fail or where it hurts: contours, strain, curvature, gaps, colour.
  Docs: [`gauges/README.md`](gauges/README.md).

Each entry gives the tool's purpose, inputs and outputs, how to run it, what trouble looks like, when to use it, and
the research it is grounded in.

## Layout

```
tools/
  INDEX.md                  this file
  solvers/
    folds/                  authored folds baked into the local fabric solver (folds.py)
    footwear/heel/          the heel's height and shape by gradient descent, and its stylized body
    hair/                   a ponytail's locks as hanging elastic rods (tail_lock_rods.py)
    tests/                  the solvers' regression tests (numpy and scipy, no Blender)
  gauges/
    contours/               contour sheets: sections, profiles, strain and bend maps after a solve
    fit/                    ease from the measured body, and the gap between worn layers
    footwear/               differential-geometric leather gauges; the heel's support and tension gauge
    tailoring/              the lapel and collar base check
    color/                  the colour-map checks (garments and faces)
    hair/                   the pre-render hair surface check
```

The old paths still work. Each moved file left a shim at its old path that runs the real file under the old module
name, so `import folds` with `blender/shared` on `sys.path`, and `python3 blender/zipboot/gauge.py ...`, behave exactly
as before (the shim is the Python documentation's "importing a source file directly" recipe:
`importlib.util.spec_from_file_location` plus `module_from_spec`, with the module put in `sys.modules` under the old
name before it runs). Edit the real file under `tools/`, never the shim. New code should use the new paths.

`folds.py` was moved byte for byte: its fold cache key hashes its own source (`folds.cache_key`), so any edit to it,
even a path line, would throw away every cached fold John approved. The key and a test bake were checked equal before
and after the move.

## Decision guide

Find your kind of project, then take everything in its row in order. "Every project" applies on top of all of them.

### Every project

1. **Reference measured first.** Any photo or reference image: `refmeasure` (PR #89, `tools/refmeasure/`) before any
   cut or model, and again on the render from the same camera, listing every gap.
2. **Head measured first** for any body, head, hair or clothing work: the head sets the ratios (project rule). The
   anatomy sources are in `blender/base/female.py`; for faces use the face checks below.
3. **Contours after every solve, before every render**: [`gauges/contours`](gauges/README.md#contours). A solve
   without its contour sheet is not ready to render.
4. **Colour map before any textured render**: [`gauges/color/colorcheck.py`](gauges/README.md#colour-map-check).
5. **Before the queue**: `tools/render-review/review.py check` and `smoke.py` (exit 77 means it reached a render,
   0 means it rendered nothing). After the render: `rq_compare.py` for before and after on a fixed camera (PR #53/#83),
   and log every rejection with `rq_reject.py`.
6. **Stopping rule**: track the gauge numbers each round; two small gains, a worse round, a returning flaw or five
   rounds means stop and report.

### By kind of project

| Project | Solvers | Gauges | Then read |
| --- | --- | --- | --- |
| **Soft garment** (cloak, cape, petticoat, skirt, sheet) | local fabric solver `blender/shared/fabric.py` (game twin `src/world/fabric-core.js`, PR #83); `folds` for any authored fold | `fit/ease.py` before drafting; `contours`; `fit/layer_gaps.py` per pose; flow check `rq_flow.py` (PR #66/#85); `colorcheck` | `docs/clothing-solvers/solver-issues.md` |
| **Structured garment for a final render** (chemise, bodice) | Blender cloth, as `blender/chemise/solver.py` and `blender/petticoat/cloth.py` drive it; cloth hand set on the swatch bench (`blender/chemise/swatch.py`, PR #91) | `contours`; `blender/base/stress.py` and `spots.py` (PR #91); `layer_gaps`; draft topology `drafttopo.py` (PR #87) before sewing | `solver-issues.md` S3 (armpit creases) |
| **Tailoring** (lapels, collars, rolls, pressed hems) | `folds` (pressed and rolled creases to a measured angle, with the fold cache) | `tailoring/collar_basecheck.py` first (the base before any fold is tuned); `contours/collar_contours.py`; join continuity G0/G1/G2 (PR #86) | `docs/clothing-solvers/lapel-and-collar-retrospective.md` |
| **Footwear** (boots, shoes, heels) | `blender/boot/last.py` (the last and the cut pieces); `solvers/footwear/heel` (heel height and shape); lace tension solver (PR #76) | `footwear/leather_gauge.py` before and after every shape, weight or pose change; `footwear/heel/study.py` and `sheet.py`; `contours/boot_*`; `colorcheck boot`; `drafttopo` (PR #87) | `zip-ankle-boot-retrospective.md`, `boots-and-hose-retrospective.md` |
| **Laces and pleats** | lace study on the game solver (PR #79, `tools/lace-study/`); lace tension solver (PR #76) | the study's own contour flags; `colorcheck` (lace against eyelets) | `docs/clothing-solvers/lace-and-pleating.md` (PR #79) |
| **Hair** | hair foundations (PR #88, `tools/hair/`): `scalp_field.py`, `groom.py` (rod model), `cards.py` for game cards; `solvers/hair/tail_lock_rods.py` for a tail's locks (3D rod statics) | the tracer `tools/hair/creator_truth.py` (the creator's outline as ground truth, every gap marked); `hair/hair_check.py`; hair stress and bun hold tests (`rq_hair.py`, `rq_bun.py`, PR #85); `hair_bench.py` (PR #47); `colorcheck` cloth against hair | `/mnt/project-files/hair-rebuild/ranger-foundations.md` |
| **Face, head, skin** | (none: faces come only from the Face Creator) | `face_lines.py` (PR #75); `facecheck.py` (PR #26); face topology `rq_topo.py` (PR #69); `check-face.py` (PR #31); `color/face_colorcheck.py`; `skin-noise-check.py` (PR #48); neck-into-head joins (PR #86) | `tools/face-creator/README.md` |
| **Anatomy and body proportions** | (built in `blender/base/female.py` from the measured head) | `refmeasure` against references; join continuity at the neck, shoulders and instep (PR #86); `contours` on the bare figure (`blender/base/contours.py`, PR #91) | `/mnt/project-files/anatomy/` |
| **Accessories** (chains, pendants) | the chain's Verlet drop in `blender/shared/accessories.py` | `contours`; `layer_gaps` (the chain over the clothes) | the module's docstring |
| **Game runtime cloth and capes** | `src/world/fabric-core.js`, `fabric-stiff.js` (PR #83) | fabric bench `tools/fabric-bench/bench.mjs` (bending length against target); cape checks `tools/cape-lab/check.mjs` (loop drift, clearance, jitter) | `tools/fabric-bench/README.md` (PR #83) |

## Catalogue

Status: **main** is on main in this folder; **main (stays)** is on main outside this folder, for the reason given;
**PR #N** is on an open draft PR and moves here when it lands (the PR's own path is given).

| Tool | Kind | Domain | Runs on | Status and path |
| --- | --- | --- | --- | --- |
| Authored folds | solver | cloth, tailoring | numpy | main: `solvers/folds/folds.py` |
| Heel optimizer | solver | footwear | numpy | main: `solvers/footwear/heel/optimize.py` |
| Stylized heel | solver | footwear | numpy | main: `solvers/footwear/heel/stylize.py` |
| Tail lock rods | solver | hair | numpy, scipy | PR #97: `solvers/hair/tail_lock_rods.py` |
| Solver regression tests | test | cloth | numpy, scipy | main: `solvers/tests/` |
| Local fabric solver (Python) | solver | cloth | numpy (scipy for MeshBody) | main (stays): `blender/shared/fabric.py`; PR #85 edits it, moves after it lands |
| Local fabric solver (game) | solver | cloth, hair | node, browser | PR #83: `src/world/fabric-core.js`, `fabric-stiff.js` |
| Blender cloth drivers | solver | cloth | Blender | main (stays, per garment): `blender/chemise/solver.py`, `blender/petticoat/cloth.py`; PR #91 `blender/skirt/settle.py` |
| Boot last and pieces | builder | footwear | Blender | main (stays, the boot is built from it): `blender/boot/last.py` |
| Chain drop | solver | accessories | Blender | main (stays, inside the module): `blender/shared/accessories.py` |
| Lace tension solver | solver | footwear, laces | numpy | PR #76: `blender/shared/lace_tension.py` |
| Lace and pleating study | solver | laces | node | PR #79: `tools/lace-study/` |
| Scalp field, groom, cards | solver | hair | numpy | PR #88: `tools/hair/` |
| Bangs flow (B/D blend over the ponytail's side flow) | solver, gauge | hair | numpy, scipy | this PR: `tools/hair/bangs_flow.py`, `bangs_draw.py`, `bangs_render.py`; `bangs_join.py` (continuous G0/G1 join to the side hair, and its one-sided-fit gauge with an in-family control) |
| Contour sheet (folds) | gauge | cloth | numpy (PIL for labels) | main: `gauges/contours/contours.py` |
| Collar fold contour sheet | gauge | tailoring | numpy, matplotlib | main: `gauges/contours/collar_contours.py` |
| Boot contour dump and plot | gauge | footwear | Blender, then matplotlib | main: `gauges/contours/boot_dump.py`, `boot_plot.py` |
| Figure contours | gauge | any garment | Blender | PR #91: `blender/base/contours.py` |
| Leather gauges | gauge | footwear | Blender (numpy half without) | main: `gauges/footwear/leather_gauge.py` |
| Heel support and tension | gauge | footwear | numpy | main: `gauges/footwear/heel/study.py`, `sheet.py` |
| Ease | gauge | fit | Blender | main: `gauges/fit/ease.py` |
| Layer gaps | gauge | fit | Blender | main: `gauges/fit/layer_gaps.py` |
| Lapel and collar base check | gauge | tailoring | numpy, scipy, shapely | main: `gauges/tailoring/collar_basecheck.py` |
| Colour-map check | gauge | colour | Blender's Python modules (reads the ramps) | main: `gauges/color/colorcheck.py` |
| Face colour-map check | gauge | colour, face | numpy, PIL | main: `gauges/color/face_colorcheck.py` |
| Hair surface check | gauge | hair | Blender | main: `gauges/hair/hair_check.py`; PR #27 has a newer copy in `tools/render-review/hair_check.py` |
| Reference measuring (refmeasure) | gauge | measuring | numpy, PIL, browser page | PR #89: `tools/refmeasure/` |
| Join continuity (G0/G1/G2) | gauge | anatomy, any join | numpy | PR #86: `blender/shared/continuity.py`, `tools/joins/` |
| Draft topology | gauge | patterns | numpy, PIL | PR #87: `blender/shared/drafttopo.py`, `tools/render-queue/rq_drafttopo.py` |
| Flow check (strain field) | gauge | cloth | Blender | PR #66 and #85: `tools/render-queue/rq_flow.py` |
| Stress and bend on the flat | gauge | cloth | Blender | PR #91: `blender/base/stress.py` |
| Show-through spots | gauge | layering | Blender | PR #91: `blender/base/spots.py` |
| Skirt checks | gauge | cloth | Blender | PR #91: `blender/skirt/check.py` |
| Swatch benches | gauge | cloth | Blender | PR #91: `blender/chemise/swatch.py`, `blender/redingote/bench.py`; main `blender/petticoat/swatch.py` |
| Fabric bench | gauge | cloth (game) | node | PR #83: `tools/fabric-bench/` |
| Cape checks | gauge | cloth (game) | node | PR #83: `tools/cape-lab/check.mjs` |
| Creator hair tracer | gauge | hair | numpy | PR #88: `tools/hair/creator_truth.py` |
| Hair stress and bun hold | gauge | hair | Blender, node | PR #85: `tools/render-queue/rq_hair.py`, `rq_bun.py` |
| Hair bench | gauge | hair | Blender | PR #47: `blender/shared/hair_bench.py` |
| Face lines | gauge | face | numpy, PIL | PR #75: `tools/face-check/face_lines.py` |
| Face check without Blender | gauge | face | numpy, PIL | PR #26: `blender/facecheck/facecheck.py` |
| Face topology | gauge | face | Blender | PR #69: `tools/render-queue/rq_topo.py` |
| Creator export check | gauge | face | numpy | PR #31: `tools/face-creator/check-face.py` |
| Skin noise check | gauge | skin | Blender, node | PR #48: `tools/face-creator/skin-noise-check.py` |
| Render request review, smoke test | gate | render pipeline | Python | main (stays, run by the reviewer): `tools/render-review/review.py`, `smoke.py` |
| Before and after compare | gauge | render pipeline | PIL | PR #53 and #83: `tools/render-queue/rq_compare.py` |
| Post-render flags | gauge | render pipeline | PIL, numpy | PR #57: `tools/render-queue/rq_flags.py` |
| Rejection log | log | render pipeline | Python | main (stays, the queue's copy runs from `/mnt/project-files/render-queue/`): `tools/render-queue/rq_reject.py` |

The render queue and the reviewer stay in `tools/render-queue/` and `tools/render-review/`: they are the pipeline
that runs the gauges, the queue's live copy runs from `/mnt/project-files/render-queue/`, and most open PRs edit them.

## Adding a solver or a gauge

- Put it in `tools/solvers/<domain>/` or `tools/gauges/<domain>/`, and add its entry to the README there and a row
  here (catalogue, and the decision guide if it changes what a kind of project should run).
- Write its docstring in the same shape as the entries: purpose, inputs and outputs, how to run, what trouble looks
  like, when to use it, and the sources its numbers come from (EXPERIMENTAL where none was found).
- Numpy-only where it can be, so it runs outside Blender and in a test.
- When a PR listed above lands, move its tool here with `git mv` in its own commit (so `git log --follow` keeps its
  history), leave a shim at the old path if anything imports it, and update its row.
