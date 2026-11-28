# Gauges

A gauge measures a shape and says where it is wrong, in numbers, before a render spends minutes showing it. Read
[`../INDEX.md`](../INDEX.md) first for which ones your project needs. Most are report-only: they never move a garment
or stop a render, they flag the likely rejections early.

Each entry: **what it is for**, **in and out**, **run**, **trouble looks like**, **use it when**, **grounded in**.

---

## Contours

The project rule: after any solver run and before any render, inspect the contours yourself (section cuts at several
heights, side and front profiles, and a bend or strain map on the flat pattern), and include the sheet when you show
John the result. A solve without its contour sheet is not ready to render.

### Contour sheet for folded sheets: `contours/contours.py`

**What it is for.** The inspection of a sheet solved with `folds.py`: is each fold the shape it was pressed to, and
where is the cloth strained or bent.

**In and out.** A `Folds` object after `bake` (and optionally a target). Out: `path.png` and `path.json`: sections
across each fold at five stations along its line (solved cloth dark over the target grey, in the crease's own frame),
sections at five heights, side and front profiles, the angle along each crease hinge by hinge against its target, and
on the flat pattern the strain (each triangle's worst edge against its cut length, blue none to red 3 % or more) and
the bend (grey flat to red 30 degrees or more, the creases' own edges in black).

**Run.** `Folds.bake` writes one every time it converges; by hand, `F.sheet(path)` or
`contours.contour_sheet(F, path)`. Numpy only (labels need PIL, which Blender's Python lacks: its sheets are
unlabelled, panels in the order above).

**Trouble looks like.** A fold section that is not one smooth hairpin; a crease angle that wanders along the line;
red strain away from the creases; a bend line where no fold was asked for; a perfectly round section where the real
thing is not.

**Use it when.** After every `bake`, every time. Compare the json between rounds for the stopping rule.

**Grounded in.** The strain is each edge's length against its length as cut (the first fundamental form compared, as
in the leather gauge); the bend is the dihedral angle at each edge.

### Collar fold contour sheet: `contours/collar_contours.py`

The bodice front's fold under the lapels: level sections up the V, sections square across each roll line (it should be
one smooth hairpin, the lapel lying parallel to the front), side sections, the front outline of the cut edges, and
each hinge's bend on the flat pattern.
```
COLLAR_FOLD_OUT=fold.pkl <the collar build> ; python3 tools/gauges/contours/collar_contours.py fold.pkl sheet.png
```
Use it on every lapel or collar round, with the base check below.

### Boot contours: `contours/boot_dump.py`, `contours/boot_plot.py`

The tall boot's bent pieces after the solver: sections at eight heights seen from above (leg grey, last orange dashes,
panels blue, tongue green), front and side profiles, and a bend map on each flat pattern.
```
CONTOUR_OUT=/tmp/boot.npz python3 tools/gauges/contours/boot_dump.py      (in Blender's Python: builds the boot)
python3 tools/gauges/contours/boot_plot.py /tmp/boot.npz sheet.png
```
**Trouble looks like.** Jagged sections; a crumple on the tongue's bend map (both caught on the laced boots, see
`docs/clothing-solvers/boots-and-hose-retrospective.md`).

### On open PRs

- `blender/base/contours.py` (PR #91): any built figure's level sections and upright profiles, every garment's
  outline over the body's. `python3 blender/base/contours.py FIGURE.blend --out SHEET.png --objects ... --z ...`
- `blender/base/stress.py` (PR #91): a garment's thread stretch and fold sharpness drawn on its flat cloth (white as
  cut, red stretched, blue crushed; dark lines are sharp creases). Linen that folds like paper shows here.
- `tools/render-queue/rq_flow.py` (PR #66, #85): the flow check, the strain and stress field across each garment
  after its build (stretch and squash percentiles, shear on the grain, Green strain, bending energy), with a heatmap.
  Jagged peaks mark dart tips, unequal paired seams, grain errors and missing ease.

---

## Fit

### Ease: `fit/ease.py`

**What it is for.** Setting each layer's ease from the measured body instead of by feel.

**In and out.** The outfit's spec (`blender/shared/riding_habit.json`: every layer under to over, its fabric and cut)
and a built scene with a pose function. It measures girths (a tape round a band of skin: the convex hull of the
band's points) at rest and in the pose strip's extremes, and derives, region by region,
`ease over beneath = standing ease + movement ease + pi * thickness * turn`, and the ease over the body. Out: the
spec's `derived` block, and `radial_u` / `radial_w` (the ease as a stand-off, ease / 2 pi, in units and in w).

**Run.**
```python
import ease
spec = ease.load()
d = ease.derive(spec, apply=pose_fn)    # in a built scene; pose_fn(name) poses the rig
ease.save(spec, d)
```

**Trouble looks like.** A region whose movement ease is large and whose cloth has no stretch (it will pull in that
pose); a layer whose ease over the body is under the thicknesses beneath it.

**Use it when.** Before drafting any layer of an outfit, and again when the body or the stack changes.

**Grounded in.** Tailoring's ease as standing ease plus movement ease; the turn of cloth (the inner face of a thick
cloth is shorter than its outer face by about pi times the thickness); girths read with a tape's convex hull. Every
landmark comes from the base figure (`female.py`), so another body gives other numbers with nothing changed.

### Layer gaps: `fit/layer_gaps.py`

**What it is for.** How far each garment stands off the one under it in every pose, and where a lower layer comes
through (a negative gap). Report-only.

**In and out.** The spec (via `ease.load()`), the poses, and a pose function, in a built scene. From each sampled body
point a ray goes out along the skin's normal through every layer: `gap(L) = inner(L) - outer(lower layer)`. Out:
`DIR/layer_gaps.json` per pose, pair and region (count, smallest gap, 5th percentile in mm, share negative, share under
the spec's ease, worst point), and `show_through`, worst first.

**Run.**
```python
import layer_gaps, ease
rep = layer_gaps.check(ease.load(), poses=('rest', 'arms-up', 'step'), apply=pose_fn, out=DIR)
```
The queue runs it after the build on requests with `--poses` (`tools/render-queue/rq_views.py`).

**Trouble looks like.** Any `show_through` entry; a 5th percentile under the spec's ease in the arms-up or step pose.

**Use it when.** Every layered outfit, every pose strip, before the first render. PR #91's `blender/base/spots.py`
then renders only a crop of each show-through spot instead of a full turn.

---

## Footwear

### Leather gauges: `footwear/leather_gauge.py`

**What it is for.** What a sheet of leather is asked to do, measured on the boot's mesh, so a distortion is traced to
its cause instead of smoothed where it shows. CLAUDE.md: measure before and after every change to a boot's shape,
weights or pose.

**In and out.** A boot figure (`--figure zipboot | heelboot | boot`, `--side`). Out: `DIR/gauge.json` and
`DIR/gauge-<name>.png` (each gauge as a colour map, three views). Six gauges:

| Gauge | Measures | Trouble | Run it when |
| --- | --- | --- | --- |
| `gauss` (angle defect) | how far a piece is from a sheet that lies flat | large total curvature on a piece meant to be one bent sheet | a new piece is cut |
| `strain` (principal stretches) | pull and squeeze, cut to sewn or rest to posed | under 0.9 or over 1.15 | weights, bend bands or the pose change |
| `fold` (dihedral) | creases | a line over 25 deg that is not a seam or an edge | a line shows and you need to know if it is geometry |
| `mean` (cotangent Laplacian) | bumps and dents | isolated spots on a smooth area | after smoothing, or a lumpy toe or heel |
| `symmetry` by station | the draft's evenness side to side | the ridge or middle off the axis | after drafting a toe or vamp |
| `pose_asymmetry` (mirror partner, Kabsch) | whether the two sides bend alike | mean over 0.02 units | a lopsided fit, or any rig change |

**Run.** `python3 tools/gauges/footwear/leather_gauge.py --out DIR [--figure zipboot] [--side R] [--size PX]` (or the old
`python3 blender/zipboot/gauge.py ...`); `--side L` for the left boot (default R). The numpy half runs without
Blender and can go in a test. Compare the boot at rest with the boot posed before blaming the shape: a line that is
only there posed is the rig.

**Misses.** Fit against a neighbour (a sole ledge, a loose heel) and the shape against the photo: use sections of
leather against leg and upper against sole, and refmeasure.

**Grounded in.** Differential geometry of developable surfaces: a flat sheet keeps K = 0 however it bends (Gauss's
theorema egregium), so the integral of |K| is the turning it must be stretched to take; the principal stretches are
the square roots of the eigenvalues of the pulled-back metric; mean curvature from the cotangent Laplacian; the
Kabsch fit for rigid motion. Numbers and the zip boot's history: `docs/clothing-solvers/zip-ankle-boot-retrospective.md`.

### Heel support and tension: `footwear/heel/study.py`, `footwear/heel/sheet.py`

**What it is for.** How a foot is carried on a heel: load shares, peak pressure, the tip, the shank's sag and safety,
the push the upper holds. `study.py` is both the heel's derivation from the foot and the gauge; `sheet.py` draws it.

**In and out.** The feet a build saved (`HEEL_DUMP=PATH`), optionally the built parts. Out: `OUT.png` and `OUT.json`
(sections, plantar pressure with the tread and centres of pressure, load shares against Cavanagh 1987, a sweep over
heel height, the waist's moment and sag), every number marked SOURCED or EXPERIMENTAL.
```
python3 tools/gauges/footwear/heel/sheet.py DUMP.npz OUT.png [--parts PARTS.npz] [--h CM] [--arch 1] [--cup 1]
```

**Trouble looks like.** Midfoot, forefoot or toe shares off Cavanagh's (the gauge is calibrated on the heel's share
and peak only, so the rest is the test); a tip angle or shank safety outside the sweep's safe band.

**Grounded in.** Golding, "The Manufacture of Boots and Shoes" (c. 1902): the seat point 1/6 of the foot's length,
the joint 5/7, the seat 3/4 of the tread at the joint; One & Awl 2024; Dooley 1912; Chen, Li and Wang 2020 (PeerJ
8:e10239: postural control worsens from 3.9 cm); Cavanagh, Rodgers and Iiboshi 1987 (heel 60 %, midfoot 8 %, forefoot
28 % standing barefoot); Broch et al. 2004; Wibowo et al. 2017; Raychoudhury et al. 2014 (the metatarsal line at 73
degrees); ISO 18896 / EN 12959 (shank stiffness); Perry and Burnfield 2010.

### On open PRs

- `blender/shared/drafttopo.py` (PR #87): topology of a draft before anything is sewn: Euler characteristic and genus
  of each piece and of the assembled garment, seams, openings, singular points, flattening limits, anchors and holds,
  buttons. A solve cannot fix a draft that is wrong here (Edelsbrunner and Harer 2010; Botsch et al. 2010).

---

## Tailoring

### Lapel and collar base check: `tailoring/collar_basecheck.py`

**What it is for.** The surface and the joins a lapel and collar are built on, measured before any fold is tuned on
top. Each check prints its number, its limit, where the limit comes from, and PASS or FAIL; exit 1 if any fails.

**In and out.** `FOLD.pkl` and `COLLAR.pkl` (the build's `COLLAR_FOLD_OUT` and `COLLAR_OUT`) and the build's log.
`--json OUT` for the numbers.
```
python3 tools/gauges/tailoring/collar_basecheck.py FOLD.pkl COLLAR.pkl BUILD.log [--json OUT]
```

| Check | Limit | From |
| --- | --- | --- |
| roll bend (normal curvature along the roll) | abs(kn) <= 0.6 per unit (R >= 4.2 cm) | the inner layer of the turned lapel squeezed by t / (2R), held to 10 % |
| roll sideways (geodesic curvature) | abs(kg) <= 0.6 per unit | a straight roll on the pattern is a geodesic on the garment |
| hollow under each lapel | 0.1 units (2.6 mm) | hair canvas cannot follow a hollow (UKY CT-LMH.176, ASG chest shield) |
| gorge held | 60 % | the lapel's edge sewn to the collar |
| gorge length | 5 % | seams pair edge for edge |

**Use it when.** Before tuning any lapel or collar fold, and after any change to the front's surface.

### On open PRs

- `blender/shared/continuity.py` and `tools/joins/` (PR #86): join continuity measured as class-A surfacing measures
  a patch boundary: G0 gap (mm), G1 tangent angle (degrees), G2 curvature deviation |kA - kB| / (|kA| + |kB|)
  (Autodesk Alias, "Continuity 2: Construction Tolerances"), each with its noise and a control station away from the
  join, so a step is told apart as the join's (modelling), the surface's (tuning) or the solver's. Presets for the
  neck into the head, the foot's top line, a lapel's roll. Also the queue's optional pre-render check
  (`rq_views.py --joins`).

---

## Colour

### Colour-map check: `color/colorcheck.py`

**What it is for.** Before a textured render: reads the colours a build will paint (from the materials' own ramps, not
typed hexes) and flags touching pairs that do not separate, or a part that reads black against what it lies on.

**Run.** `python3 tools/gauges/color/colorcheck.py boot [--out DIR]` prints the pairs table and writes
`colour-map.png` (every material's five ramp tones, each pair's gaps). Needs the build's Python modules (bpy).

**Limits (CIE Lab, D65).**
- touching parts: lightness gap 12 or more, or colour distance dE 25 or more;
- under 25 of lightness gap, the hue must differ by 20 degrees (CLAUDE.md: brown beside brown vanishes; separate by
  hue, not value alone);
- grayscale (relative luminance) gap 6 of 100 or more;
- an accent (eyelet rings) stands out from the leather by dE 8 or more but no more than the lace plus 2;
- a cord over a surface is lighter than that surface's shadow tone by 10 L and not darker than its mid tone by more
  than 8 L (a dark cord in a dark opening reads as a black hole).

**Trouble looks like.** `BAD` lines and a `WATCH` verdict. Fix flagged pairs on swatches first; no textured render
with an unresolved flag unless John accepts it (project rule).

**Grounded in.** CIE L*a*b* (D65) and the CIE76 colour difference; relative luminance for the grayscale view.

### Face colour-map check: `color/face_colorcheck.py`

The same limits on a dumped Face Creator face (`tools/face-creator/dump-face.cjs`): skin, eye shadow, liner, brows,
lips, eye white, iris and the hair's base tone.
```
python3 tools/gauges/color/face_colorcheck.py DUMP [--out swatches.png] [--hair 7e4c34]
```
Run it before any textured face render, and with `--hair` whenever cloth framing the hair changes colour.

---

## Hair

### Join cover: `hair/join_cover.py`

**What it is for.** The interface between two sets of hair combed opposite ways (a fringe and the hair swept back from
it). Both sets leave the line between them, so cover is thinnest there, and hair that lies under the scalp cap is not
seen at all. A ridge or dark band across the head at a join is usually one of these, not the shading.

**In and out.** A head and a Face Creator strand asset (`bangs_creator_asset.py`). Rows across the join in three slabs,
each with `cover` (layers of hair over the scalp), `outer` (the surface's height, mm) and `buried` (the share of hair
under the cap), then the worst cover and the buried share within `--band-mm` of the join.
```
python3 tools/gauges/hair/join_cover.py HEAD.json.gz ASSET.json [--cap-mm 0.4] [--band-mm 12]
```

**Trouble.** Cover under 1 over more than a few mm; buried over 0.1; a step in `outer` over 2 mm between rows. On the
Warden's swept ponytail: cover 0.30 and buried 0.36 before the join work, 1.10 and 0.001 after.

**Misses.** What the join looks like lit: a covered join can still read as a line if the two directions catch the
light differently. Check the render from above as well.

**Grounded in.** Counting: cover is root density x distance from a line the hair leaves x a hair's width, so it must
fall to nothing at an unblended join. The thresholds are DRAFT.

### Face edge: `hair/face_edge.py`

**What it is for.** Hair that seems to cut the face or change its shape. An under layer of the strand asset (the
coverage layer, the mid layer) is drawn "in" from its strand; beside the face "in" is onto the cheek, so the layer
lies past the top hair as a dark line down each side and the face reads narrower.

**In and out.** Two close-ups of the face from the same camera, the creator's own hair and the groom
(`creator_shots.cjs PAGE PREFIX STYLE FACE creator|strands face=0/4/22/0.42`). The groom's face width against the
creator's, each edge's offset, and the tone of the hair right at the skin against the hair farther out.
```
python3 tools/gauges/hair/face_edge.py CREATOR-face.png GROOM-face.png
```

**Trouble.** Face width more than 1 mm under the creator's; the edge band's ratio well under the creator's. On the
Warden's wavy hair: -2.9 mm before, -0.6 mm after the hair beside the face lost its coverage and mid layers.

**Misses.** Which layer it is: render the layers in false colour for that (swap the strand shader's output per
layer). The skin test is by colour and assumes the creator's lighting. The mm scale is rough.

### Flow smoothness: `hair/flow_smooth.py`

**What it is for.** A hair flow that turns in corners. A flow built from separate eases (back behind the cheek, then
forward to the chest, then onto the body) can join them sharply and read as a zigzag; rank, turning-back and flux
gauges do not see it (John 2026-11-28 on the photo style's front fall).

**In and out.** The field's lines before any wave or curl is laid on them, and for each the node where it leaves the
head. Out: the bend radius read over 10 mm (minimum, 1st percentile, median), the share of nodes under the bound, the
worst lines with their heights, and the count of lines whose course round the body reverses more than once.
```
from flow_smooth import gauge;  gauge(lines, release, 'free')   # or 'set' for a fringe dried on a brush
```

**Trouble.** `ok` False. Free hair: over 1% of nodes under 45 mm (a hair's bending length: Goldstein, Warren, Ball
2012, "about 5 cm"; 45.4 mm from Robbins' modulus in `tail_lock_rods.py`), any node under 25 mm, or a line with more
than one reversal. Set hair: over 1% under 17.5 mm (a 35 mm round brush, ghd). The photo style's field v4: minimum
7 mm, 3.4% under the bound; after fairing: minimum 59 mm, none under.

**Misses.** Bends held by a contact (over an ear, at a tie) are real and will fail the free bound: pass only the free
part. It reads single lines, not whether neighbours stay parallel (that is the rank gauge).

**Grounded in.** The creator's own outline as ground truth (the project's hair rule). Thresholds DRAFT.

### Hair surface check: `hair/hair_check.py`

**What it is for.** Catching creases, seams and ledges in the fused hair, and clothes coming through it, before a
render is requested. It ray-casts; it never renders. About two minutes, no GPU.

**In and out.** A built `.blend` of any figure on the base rig (`warden.py --build-only --out DIR`). For each of the
eight facings an orthographic grid of rays reads the hair's smooth-shaded normal. Out: `hair-check.png` (problems red
and yellow) and exit 1 on failure.
```
python3 tools/gauges/hair/hair_check.py BLEND [--out DIR] [--views 0-7] [--cols 0,2,4,6] [--hair Hair]
```

**Trouble looks like.** `crease` (a fold inward sharper than a fillet of `CREASE_R`: a seam that renders as a dark
line), `ridge` (outward sharper than `RIDGE_R`: a piece's corner), `ledge` (hair stepping over hair by more than
`LEDGE_D` with the nearer side facing sideways: a lock lying on the fall instead of growing out of it), any blob over
`MIN_BLOB` pixels; clothes poking through in the walk frames.

**Use it when.** Any change to hair or to clothes next to it. PR #27 has a newer copy (`tools/render-review/hair_check.py`,
which also reads painted strand textures and runs inside `smoke.py`); take that one when it lands.

### On open PRs

- `tools/hair/creator_truth.py` (PR #88), the tracer: the Face Creator's own hair drawn from the groom's cameras, its
  outline in red and flow lines in blue over the groom, magenta where the creator has hair and the groom does not,
  cyan the reverse; `truth.json` gives overlap, missed and added area in cm2, and the outline's mean and 95th
  percentile distance in mm. John's hair rule: comb and cut to this outline, never by eye.
- `tools/render-queue/rq_hair.py`, `rq_bun.py` (PR #85): the hair stress test through walk, run, turn and twist, and
  the bun hold test (sag, drop, swivel, deformation, core and head piercing).
- `blender/shared/hair_bench.py` (PR #47): Blender's render of a hair asset beside the creator's, on the creator's
  cameras and lights, with a difference map.

---

## Faces, heads, measuring

### Face flow: `face/face_flow.py`, `face/render_clay.cjs`

**What it is for.** The brow-to-mouth lines a sculptor reads, traced on a reference photo and on a Face Creator head
from the same front orthographic camera, in one unit (the pupils' distance), with every gap listed: the brow-tip line
(the lit front plane's edge from the brow's underside down the bridge's side to the tip-defining point), the outer line
(the lid crease, the nasofacial groove, the wing's crease), the philtrum's columns and the upper lip's line; and the
landmarks between them (inner and outer eye corners, wings, tip-defining points, bow peaks, mouth corners).

**In and out.** `ref <photo> <refmeasure sheet>`: the photo's yellow strokes traced automatically (John marks his
references in yellow) and levelled on the pupils. `model <dump.json>`: a head dumped by `tools/face-check/dump-heads.cjs`
(PR #75). `compare <ref.flow.json> <model.flow.json>...`: `flow-gaps.md`, `flow-gaps.json` and `flow-sheet.png` (the
photo with both sets of lines, and each head's clay front with ours in cyan and the photo's in yellow).
`render_clay.cjs <creator.html> <out> [faces]` renders the cheap clay and raking-key matcap views (front, 3/4, side,
back, and close-ups of the brow-to-mouth region from the front and 3/4) with the hair hidden.

**Run.**
```
python3 tools/gauges/face/face_flow.py ref ref.png ref.measure.json --out flow
node tools/face-check/dump-heads.cjs face-creator.html dump Ranger,Rogue      (PR #75)
python3 tools/gauges/face/face_flow.py model dump/Ranger.json --out flow
python3 tools/gauges/face/face_flow.py compare flow/ref.flow.json flow/Ranger.flow.json --out flow
node tools/gauges/face/render_clay.cjs face-creator.html clay Ranger
```

**Trouble looks like.** The brow-tip line's waist far narrower than the tip-defining points (a fin of a bridge); the
outer line close in at mid-bridge (the side walls do not run out into the cheek); a large brow-to-nose join gap (the
brow's line and the nose's do not meet); many polygon sides at 1 mm or a sharp turn on a line (a step, not a flow);
the mouth's corners off the medial limbus, the wings off the inner corners.

**Use it when.** Any change between the brow and the mouth, before and after, with `face_lines.py` for the profile
angles a front view cannot see.

### Plane breaks: `face/plane_breaks.py`

**What it is for.** It reads the chisel from the tone, the way a draughtsman does, on a reference photo and on Face
Creator heads. Across the shadow trace's line at eight heights it measures where the front of the nose turns into its
side, how wide that turn is (sharp, firm or soft), how far the side turns, and how wide the climb from the shade back
onto the cheek is. Sources: Prokopenko and Hampton on edges (Proko), Horn on image intensities.

**In and out.** `ref <photo> <refmeasure sheet>`, `model <dump.json>` and `compare <ref.breaks.json> <model.breaks.json>...`.
It writes `plane-breaks.md`. Per height: `x_break` (pupil distances), `width`, `rise` and `radius` (mm, pupil distance
63 mm), `core` (the side's tone over the front's) and `turn` (degrees). On a head it adds `turn_geo` and `radius_geo`
from the true normals, a check on the tone reading.

**Run.**
```
python3 tools/gauges/face/plane_breaks.py ref ref.png ref.measure.json --out pb
python3 tools/gauges/face/plane_breaks.py model dump/Rogue.json --out pb
python3 tools/gauges/face/plane_breaks.py compare pb/ref.breaks.json pb/Rogue.breaks.json --out pb
```

**Limits.** The photo's light has more fill than the heads' key, so the photo's `turn` reads low; compare positions and
widths, which do not depend on it. Above the inner corners the brows' hair and the socket's tilt make `turn` unreliable.

### Shadow trace: `face/shadow_trace.py`

**What it is for.** It finds the shade's darkest line from under the brow's head, past the inner corner of the eye,
down the side of the nose and round its wing. It traces this on a reference photo and on Face Creator heads lit the
same way, from the same front camera, in the same unit as `face_flow.py`. It shows whether our shade runs where the
photo's does, as one smooth wave.

How each side is made:

- **The photo:** John's yellow marks are painted out (biharmonic inpainting), the image is levelled on the pupils,
  and its luminance is taken.
- **The heads:** a z-buffer of the head and eyes, with normals from it, lit by a soft frontal key from a little above
  (Lambert's law with a wrap).
- **On both:** the same opening is masked round each eye. The line is the darkest smooth path on each side: dynamic
  programming over rows 0.01 apart, with a penalty for bending.

**In and out.** `ref <photo> <refmeasure sheet>`, `model <dump.json>` and `compare <ref.shadow.json> <model.shadow.json>...`.
They write `shadow-gaps.md` and `shadow-sheet.png`. The sheet shows the photo's line in red, ours in blue, and the photo's
laid over ours in yellow. The measures are the line's x out from the midline at four heights:

- the brow's head (y -0.25);
- the inner corner (0);
- the bridge's side (0.25);
- round the wing (0.5).

From those come the wave (inset and outset against its narrowest point), its curvature's sign changes and its
sharpest turn.

**Run.**
```
python3 tools/gauges/face/shadow_trace.py ref ref.png ref.measure.json --out sh
python3 tools/gauges/face/shadow_trace.py model dump/Ranger.json --out sh
python3 tools/gauges/face/shadow_trace.py compare sh/ref.shadow.json sh/Ranger.shadow.json --out sh
```

**Trouble looks like.**

- The line out at 0.3-0.4 at the brow's head: the shade sits in the socket under the brow, not beside the bridge.
- The line out past 0.2 at the inner corner: the side of the bridge is lit down to the eyes.
- A sharp turn per 0.05 well above the photo's 27 degrees: a kink where one shade hands over to another.

**Use it when.** Any change to the nose's side, the socket's inner corner or the brow's head. Use it together with
`face_flow.py`.

**Grounded in.** Park 2011 (doi 10.3342/ceo.2011.4.2.55); Brito, Avashia and Rohrich 2020 (PMC7159929); Cangello and
Hilinski nasal analysis; plasticsurgerykey, Nasal analysis and anatomy; Bridgman, Constructive Anatomy; Proko, Planes
and forms of the head; Anatomy for Sculptors, Forming the head. The 35 degree edge of the lit plane is this gauge's own
choice (EXPERIMENTAL), between face_lines' 30 (tip-defining points) and 45 (dorsal lines). The numbers and the first
use: `docs/faces/face-flow.md`.

### On open PRs

All on open PRs; they land here under `gauges/face/` and `gauges/measuring/`.

- `tools/refmeasure/` (PR #89): comparative measurement of a reference image and of a render against it, the
  atelier method made repeatable: one unit, plumb and level lines, size, position and angle as ratios, the whole before
  the parts, one viewing position (`compare` warns when cameras differ). CLI plus a click-and-drag page. The project's
  standing rule: measure every reference with it before cutting, and the model after building.
- `tools/face-check/face_lines.py` (PR #75): Farkas' landmarks found on the head mesh, the standard soft-tissue
  profile and nasal analysis lines drawn on clay front, profile and base views, each measure against a published
  range.
- `blender/facecheck/facecheck.py` (PR #26): a figure's face built from its own Python without Blender, drawn and
  measured (lids against the eyeball, iris showing, paint on the lips, symmetry), in seconds.
- `tools/render-queue/rq_topo.py` (PR #69): the head mesh's topology (poles, edge loops round each feature, contour
  roughness) in zones measured from the head (H, IPD, the canon's eye width).
- `tools/face-creator/check-face.py` (PR #31): how closely Blender's lookup of a creator export lands on the
  creator's points.
- `tools/face-creator/skin-noise-check.py` (PR #48): the creator's GLSL noise against Blender's own (all under 1e-4).

---

## Game cloth (open PR #83)

- `tools/fabric-bench/bench.mjs`: fixed sheets on fixed bodies (hang, drape, fold, boot shaft, tasset), timed and
  measured the same way every run; `--cantilever` checks each material's bending length against its target.
- `tools/cape-lab/check.mjs`: the cape's loop drift, swing, settle time, frame-rate dependence, clearance, cloth in
  the body and jitter, without a browser.

---

## Render pipeline (stays in `tools/render-review/` and `tools/render-queue/`)

- `review.py check`: the reviewer's gate before the queue (shape, commit, merge, view flags, pitfalls, smoke, cache).
- `smoke.py`: runs a render script headless up to its first render. Exit 77 reached a render (good), 0 finished
  without rendering (good only if that was the intent, such as a .glb export), anything else is the script's failure.
- `rq_compare.py` (PR #53, #83): before and after on a fixed camera from cached results; refuses pairs whose cameras
  differ.
- `rq_flags.py` (PR #57): post-render artifact flags on every finished job.
- `rq_reject.py`: log every rejection before fixing it (project rule).

### Part cleft and bang seam: `hair/part_profile.py`, `hair/crown_step.py`

**Measures.** `part_profile.py`: the 95th percentile of the hair's height over the skull (mm) in 4 mm bands of distance
from the centre part, on top of the head and behind it. A cleft shows as a low first band. `crown_step.py`: where a
bang hair and a hair of the falls lie within 4 mm of each other on top of the head, the difference in their heights
over the skull (median, 10th and 90th percentile, mm). A seam shows as a step.

**Run.** `python3 tools/gauges/hair/part_profile.py HEAD.json.gz CACHE_DIR` and
`python3 tools/gauges/hair/crown_step.py HEAD.json.gz CACHE_DIR` (the photo field, `photo_waves.field2`, 16,000 lines).

**Read.** 2026-11-29, wavy photo hair: the cleft was 8 mm at the part against 25 mm 24 mm away (17 mm deep, two humps
in the front and back outlines); with the roots stood up at the part (`photo_waves._stand_at_part`) 17.7 against 24 mm.
The bang seam: 0.0 mm median once the bang's root lift used the falls' arc and envelope. Look at the crown as well
(`photo_render.py --crown`): the numbers do not show a ragged edge.
