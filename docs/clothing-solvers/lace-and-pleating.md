# Lace and pleating study

Code: `tools/lace-study/` (solver, lace, strips, study, contour sheets, viewer), `blender/lace-study/views.py`
(clay and colour-code views of the solved strips). Runs, sweep and contour sheets are in
`/mnt/project-files/lace-study/` (`runs/`, `contours/`, `cache/`).

    node tools/lace-study/study.mjs cases            # the eight cases, cached by their settings
    node tools/lace-study/study.mjs sweep            # one setting moved at a time
    python3 tools/lace-study/contours.py RUNS/*.json --out DIR
    node tools/lace-study/calibrate.mjs --check      # each lace's bending length on the cantilever bench
    node tools/lace-study/viewer/build.mjs           # the interactive bench, one self-contained page

## The lace

Fine cotton or silk lace of the 1870s to 1890s, the kind engageantes, jabots and collar frills were made of.
Five parts, each with its own weight and stiffness share (normalised so the strip's mean is the material's):
net ground (0.7 weight, 0.6 stiffness), motif (1.35, 1.5), cordonnet and scallop (1.8, 3.0), stitched heading
(1.4, 2.0), and the pressed folds, marked on top for the colour code. The edge is a true scallop: each column
ends at the scallop's depth there, so the outline is the mesh's, not a texture.

| lace     | weight   | thickness | bending length | source |
|----------|----------|-----------|----------------|--------|
| cotton   | 50 g/m²  | 0.3 mm    | 1.0 cm         | weight: Fabric Atlas 40–90, Tissura 30–180, TextilePulse 45–65 for Chantilly (weak source); thickness: Fabric Atlas 0.2–0.5 mm |
| silk     | 40 g/m²  | 0.25 mm   | 0.9 cm         | as above |
| starched | 58 g/m²  | 0.3 mm    | 2.5 cm         | the cotton, starched for fluting |

No measured bending length for lace was found. The Peirce flexometer's range for fabrics is 1.6–6.35 cm (Sanad and
Cassidy 2016) and the project's silk is 1.2 cm on the bench, so 1.0 cm for an open net is an estimate, and the
sweep below shows what it changes. The bend is the solver's implicit one (`fabric-stiff.js`, B = W·g·c³), checked
on the ASTM D1388 cantilever (Peirce 1930: c = overhang / 2 at 41.5°): cotton 0.96 / 0.91 / 0.88 cm at mesh
spacings 0.2 / 0.3 / 0.5 cm, starched 2.39 / 2.38 / 2.37. Silk holds at 0.2–0.3 cm (0.77, 0.81) and falls apart at
0.5 cm (0.43): keep silk at 0.3 cm or finer.

## The pleats and their ratios

| pleat | cut : finished | source |
|-------|----------------|--------|
| knife | 3 : 1 nominal, 2.45–2.86 here | Wikipedia "Pleat"; Minerva Patterns ("twice the pleat depth" behind each fold); Sewing With Numbers (B = 3A) |
| box   | 3 : 1 when the underlays meet (5 : 1 stacked), 2.89 here | same |
| accordion | 1 / cos(turn / 2), 2.5 here | Wikipedia "Pleat" |
| flute (goffered) | (arc / 2) / sin(arc / 2): 1.57 for half circles, 1.66 here | Saffron Walden Museum (starched, fluted on hot rollers; first US patent 1861); Every Woman's Encyclopaedia (goffering tongs on starched lace); The Quintessential Clothes Pen (oven at 275 °F, 1:1 vinegar and water) |
| gather | period 1.25–1.75, sheers to 2.5, "2× too much for a period look" | Rosbrugh, historicalsewing.com; modern 1.5–3, pleated ruffles 3 (The Shapes of Fabric) |

No source gives a flute pitch; the 0.35 cm radius is a choice. The knife pleat comes in under 3 because the solver
needs a full column at each fold's turn (see below), and its layers lean to take up that gap.

## How a strip is made

Cut flat at the ratio, laid in the pleat's cross-section, the heading row pinned along the mount (a band, a cuff
round the forearm, a collar standing round the neck, a jabot down the centre front), then hung. A pressed pleat
keeps its folded shape as the bend's rest curvature (`press` × the folded curvature: 1 a hard press, 0 only
stitched). A gather is never pressed. 180 frames at the lace's own damping, then settled at 0.3 damping until the
fastest point is under 2e-4 cm a step for 20 frames (or 600 frames). Every solve gets a contour sheet.

## Results (res 0.3 cm, cotton unless named)

| case | max / p99 stretch | folds held, heading / scallop | flare | verdict |
|------|-------------------|-------------------------------|-------|---------|
| cuff knife | 2.1 / 0.95 % | 165° / 145° | 1.00 | sound |
| cuff box | 2.0 / 0.98 % | 159° / 123° | 1.00 | sound |
| cuff accordion (480 Hz) | 4.9 / 1.4 % | 154° / 122° | 0.99 | holds its zigzag, but buzzes at about 5e-3 cm a step and never settles |
| cuff flute (starched) | 22 / 2.3 % | rolled | 1.09 | sound; the 22% is one corner at the strip's free end |
| cuff gather (2 cm stitch) | 6.4 / 2.2 % | soft | 1.06 | sound; stretch spots in the tight folds |
| jabot knife | 8.9 / 2.3 % | 157° / 120° | spreads 4× | droops into falling folds; the kink check misreads a frill hanging across its section plane |
| jabot gather | 8.5 / 3.2 % | soft | spreads 5× | as the jabot knife |
| skirt knife (wool, press 1, 0.75 cm) | 1.06 / 0.58 % | 176° / 176° | 1.00 | sound |
| collar flute (standing, starched) | 56 / 22 % | — | — | fails: a 4 cm standing frill never settles, at 240 or 480 Hz, with 1 or 3 held rows |

## The pleated skirt

A knee-length skirt in worsted wool (280 g/m², 0.5 mm, bending length 2.5 cm: worsted skirting runs 250–300 g/m²;
the bending length sits between the project's silk at 1.2 cm and its coat wool at 4 cm, an estimate), knife-pleated
onto a hip yoke 96 cm round, 50 cm from the hip line to the hem, pleats 3 cm wide and 3 cm deep (`skirt-knife`).
It hangs over the hips (a cylinder just inside the pleats' inner layer for 15 cm, tapering 14% to the thighs).

- Hard-pressed (press 1) it settles clean: 1.06% worst stretch, 0.58% p99, folds 176° at the yoke and at the hem,
  no flare (a pleated skirt on a yoke hangs straight). Mesh 0.75 cm, 21,775 points.
- At press 0.7 the inner layers sag onto the hips and the contact tears them (26% worst). Pleated wool skirts are
  heat-set permanently, so the hard press is the right model, and the contact is the thing to fix before softer pleats.
- The mesh must be at most a third of the pleat: at 1 cm it settles at 0.8%, at 1.25 cm it comes apart (the fold's
  bridge column is too large a share of each layer). A narrower bridge than a column fails here as on the lace (37%).
- The bridge leaves a gap between the layers as wide as a column, so the pleats lean like tiles and the ratio comes
  out 2.56 (at 0.75 cm) instead of 3.
- Speed. The study case takes about 260 ms a frame (3 cm pleats, 1 cm mesh, 4 bend passes, 240 steps a second); the
  bend's passes are four-fifths of that. The bench's "Pleated skirt" scene defaults to 4.5 cm pleats on a 1.5 cm
  mesh round and 2 cm down, 2 passes, 120 steps: 4,004 points, about 19 ms a frame, settling at 1.1% worst stretch
  with the folds at 172°. Going coarser down than 2 cm fails (17% at 2.5 cm with a 1 cm mesh round), and so does
  dropping to 2 constraint walks (9%).

## What matters (the sweep, cuff knife and cuff gather as the bases)

- **Press** is the knife pleat's main control. 0: the folds open (145° / 84°) and the strip tears itself (41%
  stretch) because nothing holds the authored folds. 0.25: 7%. 0.5: sound. 0.75–1: the folds stay sharp to the
  scallop (175° / 175° at 1). Use 0.5–0.75 for a pressed lace frill, 1 for a hard-pressed one.
- **Bending length** sets how far the folds open toward the scallop and how far a gather flares. Knife: 0.5–1.5 cm
  sound (folds at the scallop 158° to 140°), 2.5 cm tears (18%): a stiff lace cannot be pressed this sharply.
  Gather: 0.5 cm gives 13 small waves and 1.0 flare, 2.5 cm gives 4 waves and 1.43 flare, at 16% stretch.
- **Gathering stitch** (the fold size) matters more than the ratio. At 0.8 cm a 2× gather packs a fold every 4 mm,
  finer than the mesh can bend, and it stretches 11% and does not settle. 1.6–2.4 cm is sound (5–6%).
- **Gather ratio**: 1.25–2.0 sound (3.6–6.4%), 2.5 at 9%, 3.0 tears at 23%. Period gathers (1.25–1.75) are the
  easy range; the solver agrees with Rosbrugh that 2× is the limit.
- **Pitch** (knife, pleat depth with it): 1–3 cm all sound; wider pleats run the ratio up toward 3 (2.45 to 2.86).
- **Gravity** at ½, 1 and 2 g changes little in a pressed cuff (p99 stretch 0.66 to 1.41%): the press holds the shape.

## What is safe to hand-tune (cuff knife)

| setting | sound range | outside it |
|---------|-------------|------------|
| threads held in the bend solve (`stretchK`) | 30–600; 300 the default | 0: 17% stretch, never settles |
| bend solve passes | 4 | 1: 6.5%, slow (447 frames); 2: 4.4% |
| constraint walks a step | 2–8 | 1: 8% |
| steps a second | 240–480 (480 halves the stretch: p99 0.36%) | 120: p99 1.9%; the accordion needs 480 |
| mesh spacing | 0.25–0.5 cm | the shape holds; the knife ratio drops to 2.38 at 0.5 |
| bridge at a fold | one full column | 1 mm: 14%; 2 mm: 4–6% |

## Solver lessons

- Hinges cannot reach lace's bending length on a fine mesh (they cap out near 0.6 cm at 240 Hz on a 3 mm mesh) and
  their stiffness changes with the mesh. The implicit bend does not.
- The implicit bend stretches a hanging strip 13% on its own; holding the threads inside the bend solve
  (`stretchK` 300, 4 passes) brings that to 3%.
- A fold over 150° (90° for the accordion) at one column breaks the bend's rotations: split it over two columns
  with a bridge column between. A bridge thinner than a column is unstable.
- Lay a knife pleat's layers at a lean found by bisection, so a repeat ends where it began; detrending the drift
  instead distorts the pleat 18%.
- Map a round mount by arc length at each radius, so threads keep their cut lengths; otherwise the cuff starts 11–17%
  stretched.
- A pleat repeat should start mid-face so no split fold wraps round the repeat's end.

## Known limits

- The knife ratio is about 2.7, not 3, at a 3 mm mesh (the bridge).
- Gathers carry 2–3% p99 stretch in their tightest folds, against under 1% for pressed pleats.
- The standing collar (a ruff) fails; period ruffs stood on a wire supportasse, and a held frame is the next thing to try.
- The drape-on-a-ball scene in the viewer jitters at the contact (about 16% at the worst point).
- The collar roll in the viewer reaches 70–117° of 160°.
- Lace's bending length is an estimate (no measurement found).
