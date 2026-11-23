# The lapel folded from the front's own cloth: the recipe

John, 2026-11-21: "This is good, summarize entire process as a recipe and rebuild the classical lapel as a test."

The order is the tailor's and it is not negotiable: **cut, fit, fold, relax, button last**. The lapel is the front
panel's own cloth. There is no separate lapel piece and nothing is welded.

## Commands

```
# 1. cut and fit (101 s): the bodice whole to the neck, two lapped fronts, cloth 4.4 mm thick, no buttons
python3 blender/base/views.py --figure bodice_cut --mannequin --turn --angles front --size 300 --samples 8 \
    --save CUT.blend --out DIR

# 2. fold, relax, button (40 s)
python3 blender/collar/foldfront.py CUT.blend OUT.blend '{"recipe": "traced_lapel"}'      # wide lapel, break below the bust
python3 blender/collar/foldfront.py CUT.blend OUT.blend '{"recipe": "classical_lapel"}'   # riding habit: high break, small lapel
```

Any option after `recipe` overrides it, for example `'{"recipe": "traced_lapel", "foot_over_bust": 0.5}'`.

## The steps, and why each is there

1. **Ease the chest** (`ease_chest`, 150 rounds). The flat chest canvas meets the round of the shoulder in a ridge; a
   lapel laid over it shows the ridge. Smooth it first.
2. **Mark the roll line** from the neck edge beside the neck (`neck_x` 2.2) to the front's edge at the break
   (`foot_over_bust`). The line swings out slightly from the straight chord (`belly` 0.25): a true curved cut, made
   in 16 segments, so the mesh has an edge exactly on the line.
3. **Cut the lapel to a traced outline** (`outline`). The outline is measured along and out from the roll line, as
   shares of its length: `[[0.04, 0], [0.27, 0.40], [1.0, 0]]` is "start just under the neck, point 27% of the way
   down and 40% of the fold's length out, straight back to the break". The cloth past the line is rebuilt on a fine
   mesh (0.07) from that outline. Trace the outline from a reference; do not eyeball it.
4. **Close the seam** (`close_slits`). Rebuilding the lapel leaves small holes along the roll line wherever a point
   of the front's edge sat a hair off the line. They render as dark ticks. Fill every hole of 8 sides or fewer.
5. **Fold, the under front first.** Each point turns over its nearest point on the curve: a half-turn roll of radius
   `roll_radius` (0.24, 1.3 cm high), then a wide arc back down so the edge lies one thickness off the front.
   The under lapel lies on its own front; the upper lapel lies on **its own front only** (`cross_ease`). Laid on
   whatever a ray met, it climbed over the other lapel at the crossing and tore (edges at +200%).
6. **Press** (`press`). `{"left": 0.85}` irons her left lapel: the roll is lowered and the cloth past it laid flat.
7. **Soften the cusp** (`cusp` [0.55, 0.7]). Toward the foot the roll keeps a floor on its radius instead of
   shrinking to a hard crease, the tail is laid on a smooth cover, and no point may stand further off the line than
   its own distance along the cloth. The upper lapel's tail gets a fuller roll (`tail_out` 1.3) so it stands out
   over the lapel that passes beneath it.
8. **Slide the under lapel behind the upper front gradually** (`cross_run` 0.4), not in one step at the edge.
9. **Relax as cloth** (`relax`, 1500 rounds). Every edge of the turned cloth is drawn back toward its length in the
   flat cut. Stretch is resisted fully, compression at 0.7. The roll line is held, the cloth may not sink into what
   it lies on, and a light hold keeps the placed shape. This halves the strain.
   **The cusp is solved first, always** (`cusp_first`, 400 rounds, only the last 30% of the fold let go), then the
   full run. Read its line before anything else: it should bring the worst edge at the cusp under about 25%. If it
   does not, the cusp is wrongly placed and the full run will not rescue it. It does not change where the full run
   ends up; it is the early check.
10. **Pin the buttons and measure** (`button_pin`). During relaxation the front round the foot is let go and the
    cloth under each button is held; the pull on the held cloth is the load on that button. The fitted front is put
    back afterwards (`keep_front`).
11. **Clean the roll line** (`clean_foot`, `foot_reach` 1.7): remove zero-area faces along the lower line and at
    the foot. With thick cloth they stand out as spikes and notches.
12. **Button last** (`button_under_foot` 0.33). The top button's top edge sits 9 mm under the foot of the fold.
    A sweep of ten heights gave the pull on it: 9.2 where the folds meet, 0.41 beside the fold, 0.08 here. Lower
    gains little and leaves the front edge unheld.

## What to read in the output

Every build prints, and the last two are saved beside the blend (`OUT.blend.strain.npz`, `OUT.blend.buttons.json`):

- strain per front: mean, 95th percentile, worst, and at the cusp;
- per button: fold left at its height (must be 0), turned cloth under its footprint (must be 0), strain round it,
  gap between the fronts (should equal the cloth's thickness);
- pull on the button pins;
- with `"truth": "truth.npz"`, the distance from a saved shape (`"save_truth": path` saves one).

Good numbers for the wide lapel: mean strain 3.9% and 5.1%, 95% of edges under 12% and 15%, pull on the top button
under 0.05, no fold left at any button.

## Rules learned the hard way

- Measure before judging by eye. The button fighting the fold, 14 slits and 88 dead faces were all invisible in the
  renders until the numbers pointed at them.
- Never blend flat and folded positions to soften a fold. Their halfway point is the fold line itself, and the cloth
  is crushed onto it.
- A button set where the cloth is still turning carries the fold. Set it below the foot.
- Three layers of thick cloth meet where the lapels cross. Decide which lies on what; do not let a ray decide.
- Fixes that free the fitted front must put it back: a tenth of a millimetre of movement there shows as a crinkle.

## Measuring a reference photo (do this before cutting)

John: "get better at ground truth measuring ratios from photos like lapel to shoulder or lapel to neck."
An outline guessed by eye came out half the size of the habit's. Measure instead:

1. Crop the photo round the lapel, brighten it until the dark cloth shows its edges, enlarge it 2 to 4 times and
   draw a labelled pixel grid over it. Read points off the grid, in the photo's own pixels.
2. Use the side of the garment that faces the camera. The far side of a turned figure is foreshortened.
3. Read these points: the break B (bottom of the V), the fold's top T (where the roll line meets the neck or the
   collar), the lapel's point P, the shoulder point S (top of the sleeve seam), the waist W, the top button.
4. Take two yardsticks from the same photo: the half shoulder (B to S, across) for widths, and break to waist for
   heights. Never mix a photo's pixels with the model's units; only ratios carry over.
5. Turn the points into ratios, then into the recipe's numbers:

| Ratio | Read as | Sets |
|---|---|---|
| `top_x` | T across from the middle / half shoulder | `neck_x` = ratio x `x_sh` |
| `roll_rise` | T above B / neck pit to waist | `foot` = T's height - ratio x (neck pit to waist) |
| `top_edge` | T to P / T to B | the outline's width, as a share of the fold |
| `point_t` | how far down the fold P sits | where the outline's corner goes |
| `reach` | P across from B / half shoulder | check only: it should come out right |
| button step | spacing / neck pit to waist | `button_step` |

Every build prints the same ratios for the model (`foldfront: front N ratios: ...`). Compare, adjust, rebuild.

## Test: the classical lapel

`classical_lapel` is the same recipe cut to the habit's measured ratios:

| Ratio | Habit photo | Model |
|---|---|---|
| Fold's top, out from the middle / half shoulder | 0.47 | 0.42 |
| Fold's rise / neck pit to waist | 0.21 | 0.21 |
| Fold's length / neck pit to waist | 0.43 | 0.39 |
| Lapel's top edge / fold's length | 0.48 | 0.48 |
| Lapel point's reach / half shoulder | 0.62 | 0.63 |

Both lapels are pressed alike. 12 buttons run from under the break to the hem, strain is mean 6.6% on both fronts,
and no fold is left at any button. Two things do not carry over: the habit's break sits at the neck pit because its
lapel joins a separate standing collar (the model's is 0.12 lower, on a collarless neckline), and the collar itself
is not part of this recipe.

### The top of the fold

John: "bend the top more, too much strain on the left." By the neck the roll line climbs the slope of the shoulder,
where the cloth runs back as fast as it runs up, so the line is far longer there than it looks from the front. Two
settings in `classical_lapel` deal with it:

- `metric3d`: cut lengths along the fold are taken in the cloth's own measure (the roll line's real length), not its
  front view. Held to the short front-view measure, the top of the lapel could never lie easy against its own line.
- `top_out` [0.5, 0.8]: the roll keeps a full radius over the top half of the fold. It used to shrink to under half
  by the neck, and pressing shrank it again.

Worst stretched edge at the top of her right lapel: +67% before, about +31% after. `traced_lapel` does not use either
setting, so the approved wide lapel is unchanged.

### The point above the top button

John: "let stress transfer down, let the point above the top button flap." `flap` [0.7, 1.4] in `classical_lapel`:
over the last 30% of the fold the lapel is not pressed, its roll is left full (1.4 of the roll radius), the
relaxation does not hold it to where it was put, and the front under it keeps what it took up while the buttons hold
it. The knot at the break opens into a free point. Strain at the cusp falls from 9.4% to 3.8% mean on her left front
and from 6.0% to 4.6% on her right; the pull on her right front's top button rises from 0.021 to 0.027, which is the
load arriving there.

### Rounded points and curved edges

John: "the collar on the reference was far smoother the tips and has gentle curves all around." On the habit photo the
lapel's point is a rounded corner about 4 px in radius on a fold 145 px long, and its long edge swells slightly. A
three-point outline gives a spike and a ruled line. `outline_round` [0.05, 0.025] in `classical_lapel`:

- the corner is cut back 5% of the fold's length along each edge and turned as a curve;
- the long edge swells out by 2.5% of the fold's length at its middle;
- the outline is widened from 0.48 to 0.50 so the reach stays at 0.60 to 0.61 of the half shoulder (0.62 on the photo);
- the edge's points are set evenly along the curve and the rows of the mesh stop short of it. Row ends beside a
  curved edge leave thin triangles, and thick cloth turns each into a spike standing off the front.

Strain did not rise: mean 6.8% and 7.1%, worst edge +48% and +69% (it was +97% with the sharp point). Still rough,
seen from the side: the top corner of each lapel crumples where it meets the neck edge.

### The tight fold, kept

John: "use the tighter fold of the first draft with the new geometry." `classical_lapel` no longer sets `top_out` or
`flap`: the roll narrows toward the neck and the lapel is pressed flat to the break, as in the first draft, on the
rounded outline. `metric3d` stays, because it is what lets the top lie easy. This is also the lowest strain of any
classical build: mean 5.2% and 6.1%, 95% of edges under 15% and 17%, worst stretched edge +39% and +64%, pull on the
top button 0.034 and 0.018. Both options still work when passed by hand; the two sections above describe them.

### Tuned by measurement: button height, button spacing, lapel width

John: "tune more params like button height and lapel width." Each was swept on the tight-fold build and scored by
strain, button load and the habit photo's ratios.

| Setting | Swept | Strain | Photo says | Chosen |
|---|---|---|---|---|
| `button_under_foot` | 0.15 to 0.8 | lapel unchanged; pull on the top button 0.073, 0.052, 0.034, 0.022, 0.011, 0.005 (her right front) | top button 28 px under the break = 0.083 of neck pit to waist = 0.69 | 0.69: pull 0.007 on both fronts |
| outline width | 0.46 to 0.60 | flat, mean 5.2 to 5.5% / 5.5 to 6.4% | `top_edge` 0.48, `reach` 0.62 | 0.48: reach 0.62 and 0.63 |
| `button_step` | | | spacing 40.5 px = 0.12 of neck pit to waist | 0.67 (3.5 cm): 9 buttons from 48.08 to the hem |

The button height and the photo agree: lower carries less load, and the habit's top button sits exactly where the
load has dropped to almost nothing. The lapel's width does not change its strain over this range, so the photo decides
it. Spacing is taken against the height yardstick, as the method says; against the half shoulder it would come out
2.8 cm, because the model's torso is longer for her shoulders than the habit's.

The button pins in the relaxation now use the same `button_step` as the buttons.


## The notch collar on the wide lapel

John: "go back to the other lapel and add collar so it's 30 degrees notch with new lapels." The wide traced lapel
gets the rounded outline (`notch_lapel`), and a separate collar is sewn to its top edge with a 30° notch.

```
python3 blender/collar/foldfront.py CUT.blend LAPEL.blend '{"recipe":"notch_lapel"}'
python3 blender/collar/notchcollar.py LAPEL.blend OUT.blend '{}'
```

foldfront writes `LAPEL.blend.lapels.npz` (each lapel's points, its cut edge and its crease), which notchcollar reads.

The collar is two pieces of the bodice's wool, at the bodice's thickness:

1. **The stand.** It rises from the neckline round the back (2.5 cm at the back) and dies away over about 6 cm toward
   each lapel's crease. Its top is the roll line.
2. **The fall.** It is turned down over the stand and lies on whatever is under it. To lay it, the cloth is seen from a
   point F inside the chest. Each line of constant azimuth seen from F (a meridian) is walked down from the roll, and
   the chart is (azimuth, length down the cloth). The outline is drawn on that chart and meshed there. Its parts are
   the roll, each gorge, each collar end, and the outer edge (4.2 cm deep at the back, about 7 cm at the points).
3. **The notch.** G sits halfway along the lapel's top edge. The collar's end G to C is turned off the lapel's edge
   (G to the lapel's point P) by 30° in the cloth's own plane at G. C is 0.9 of GP from G.

Three things made the fronts clean. Each one fixed a flaw that showed in the renders:

- **The seam is measured along itself.** The gorge runs nearly along a meridian, on the lapel's steep edge. Looked up
  through the walked meridians, its points jumped up and down the chart: a lapel point and a point behind it share a
  direction from F. The cloth then folded on itself and solidify made spikes, and one side came out 3 cm deeper than
  the other. Now the gorge's place on the chart is its own length: each step along it is split into what goes across
  the meridians and what goes down them.
- **The collar's end is laid straight as cut.** The boundary points from G to where the point starts to round are put
  on the straight line G to C (pushed off what lies under them). The notch then opens on the cloth exactly as cut:
  31.4° and 30.6°. Turning it on the chart instead came out at 55° to 80°, because the chart is not true to the cloth
  at G.
- **The fall hangs as a membrane.** Over the lapels' rolls and the stand's dying end, the ground seen from F is steep
  and stepped. A nearest-point push sends neighbouring points different ways and crumples the cloth. So, after laying,
  the fall is smoothed hard (250 rounds) and only ever pushed outward along the line from F, which bridges the steps.
  The smoothing is eased in from the back (0.6 of full strength there) to the fronts, so it leaves no crease where it
  starts. Points near the seam are not pushed: they hang from it. The seam's own points slide along the lapel's edge,
  kept in order and evenly spaced, because two points on one spot make a spike.

`notch_lapel` also takes `metric3d`, as the classical lapel does. Her left lapel's strain went from 5.6% to 4.7%
mean, and on the visible cloth from 5.2% to 4.3%. The worst edges in the strain print (−84%) are on the part of the
under lapel hidden behind the upper front, which is pressed flat there on purpose.

What to read in notchcollar's output: the notch "as cut" and "as laid" should both be near 30, the fall's depth at the
two points should match, and the fall should be at least a few hundredths off the bodice.

### The notch angle, swept

John: "try angles for the collar, find one that works well that doesn't crease or blow up." The collar was built on
the same lapel at 20, 30, 40, 45, 50, 60, 75 and 90 degrees and rendered front and three-quarter.

| Angle | As laid (her right / left) | Collar point depth | Sharp folds | Looks |
|---|---|---|---|---|
| 20 | 22.1 / 21.0 | 7.4 / 8.0 cm | 32 | clean, but the notch is a slit that barely reads |
| 30 | 31.4 / 30.6 | 7.1 / 7.6 cm | 34 | clean |
| 40 | 41.0 / 40.4 | 6.9 / 7.3 cm | 34 | clean; the notch reads clearly from the front and the side |
| 45 | 45.5 / 45.4 | 6.8 / 7.2 cm | 34 | clean; the collar's end starts to look square |
| 50 | 50.3 / 50.3 | 6.8 / 7.1 cm | 34 | a faint crease across the collar's end on her left |
| 60 | 59.1 / 60.2 | 2.0 / 6.9 cm | 44 | blows up: on her right the outline folds and the point is 2 cm deep |
| 75, 90 | 73 to 87 | under 2 cm | 23 to 26 | blows up: the ends turn into boxy flaps |

The default is now 40. From 60 up, the collar's end turns so far back that it runs into the outer edge on the chart,
and the outline folds.

### A shorter collar

John: "fix the cut, it's creasing, the collar is too long", then "the ring of the collar, reduce that length."
Two ways to shorten it were tried:

- **Drawing the ring in round the neck** (`hug`, off by default). The collar's neck edge is set by the bodice's
  neckline, which already sits close to the neck: the most it gained was 1 cm (37.7 to 36.5 cm). It pinched the stand
  at the front of the neck and doubled the sharp folds (34 to 78).
- **A lower, shallower band** (kept). The stand is 1.6 cm at the back (was 2.5) and the fall 2.8 cm (was 4.2). The
  collar's points are 0.7 of the lapel's (3.4 cm, was 0.9 and 4.4 cm), so the fall is 6.1 and 6.5 cm deep at the
  points (was 6.9 and 7.3). The notch lays at 41.5 and 40.7 degrees. There are 34 sharp folds, the same as before.

### Lower and drawn in, together

John: "do a combined run: lower diameter and lower collar line." The low band above, with the ring drawn in to 0.25
off the neck (`hug`, now on by default). Two changes made the drawn-in ring lie clean:

- It is drawn in gently, over twice the stand's own run (`hug_run`), and the neck edge is smoothed harder.
- The ground-only skirt under the stand bridges from the drawn-in neck edge back out to the bodice's neckline. Before
  this, the fall sank into the gap between the two and creased at the sides of the neck.

The roll round the neck is 36.5 cm (was 38.3) and sits 1.1 cm off the neck on average (was 1.5). There are 41 sharp
folds, against 34 without the draw-in and 75 before the two changes. None of them shows in the renders.

### The lower collar, smoothed

John: "smooth the creases from the lower collar only." The drawn-in ring is off again (`hug` unset). Up close, the
lower collar had three creases, each with its own cause:

| Crease | Cause | Fix |
|---|---|---|
| a step where the collar's top meets the seam to the lapel, and a nick in the fall's top edge there | the roll edge was fixed over the stand, and the seam at the lapel's edge; the two did not meet | the roll edge is evened out along itself, fully at its ends (`roll_free` 0.45) and lightly round the back (`roll_even` 0.15) |
| a crease across the fall at each side of the neck | the bodice's own neckline edge, seen edge-on from F, propped the fall up from the side where no ray finds it (the fall was 0.13 off it, inside its 0.21 clearance) | a last polish (`polish` 40 rounds) evens the fall and keeps it off everything by its nearest point too, so it rounds over the edge |
| the same crease, sharpened where the stand dies away | the stand fell off over 5.8 cm | `stand_run` 3.2: it dies away over 8.4 cm |

Sharp folds (neighbouring faces more than 60 degrees apart) went from 34 to 0.

### Pressed down

John: "press down the collar." The fall already lay close: 0.39 cm over what it lies on, on average. Two ways to press
it were tried:

- **Drawing the fall in onto what lies under it** (`press`, off by default). Drawn in point by point, it copied every
  step under it, with 714 sharp folds. Drawn in by an evened-out share of the gap and kept clear of the roll, it gained
  only 0.15 cm and still creased at the sides of the neck (71 folds).
- **A lower stand** (kept). The stand is 1.0 cm at the back (was 1.6), and the fall reaches 0.6 past it, so the fall is
  2.6 cm deep at the back (was 2.8) and the points are unchanged (6.1 and 6.5 cm). The collar turns over lower and
  lies flatter round the neck, with 0 sharp folds.

### The back of the collar against the neck

John circled the collar's back seen past the neck from the front: it floated, with a dark gap between it and the neck
(1.4 cm on average, 2.2 cm at most). The stand now leans in more (`lean` 0.35, was 0.12) and sits closer to the neck
(`neck_clear` 0.05, was 0.10). The ring is drawn in again (`hug` 0.15), now all the way to near the fronts (`hug_run`
1.0). With the roll edge evened out and the last polish, this no longer creases: there are 0 sharp folds. The roll is
35.5 cm round the neck (was 37.9) and 0.8 cm off it on average, which includes the stand's own thickness.

### The neckline redrafted to the collar

John: "you have to redraft with this new neckline so the collar fits better." The collar used to be drawn in to the
neck on its own, over a bodice neckline that stood off the neck. Now the bodice is cut to the collar's neck edge, and
the collar is sewn straight onto it (`hug` off).

`bodice_cut.neck_fit`, in the cut step, before anything is folded, works like this:

- Round the sides and back, each point near the neckline moves level toward the neck until it lies `BODICE_NECK_HUG`
  (0.05) off it.
- The move is full at the neck edge and fades out `BODICE_NECK_DROP` (2.0) below it. It is evened out over the mesh,
  so the shoulders take it up as one swell.
- It starts only past 0.9 rad round from the front and is full past 1.4 (`BODICE_NECK_FROM`). Starting at 0.6, it
  reached the lapels' top corners and roughened the collar's right end.
- The lapels come out of the fold exactly as before: the same strain to the tenth of a percent.

The neckline round the back is 36.3 cm (was 37.7). The collar's roll is 36.0 cm and 0.95 cm off the neck on average,
which includes the stand's own thickness, with 0 sharp folds and the notch at 41.5 and 40.7 degrees.

```
python3 blender/base/views.py --figure bodice_cut --mannequin --turn --angles front --size 300 --samples 8 --save CUT.blend --out DIR
python3 blender/collar/foldfront.py CUT.blend LAPEL.blend '{"recipe":"notch_lapel"}'
python3 blender/collar/notchcollar.py LAPEL.blend OUT.blend '{}'
```

### The stand drafted from the neckline

John circled the collar's ends beside the neck, then wrote: "think about how the shape changes as a function of the
neckline." A notch collar's roll line is the lapel's roll line carried on round the neck. Before, the stand died away
over a fixed length (8.4 cm) toward each lapel's roll top N. There the collar's roll met the lapel's crease at 36
degrees, and the collar's end folded at that corner.

Now the stand's rise is drafted from the neckline itself (`from_neck`, on):

- At each roll top N, the stand rises off the neckline at the slope that keeps the roll straight on from the lapel's
  crease. The slope is found from the neckline's direction at N, the stand's lean there, and the crease's direction.
- It reaches its full height at the side neck point (90 degrees round from the front), where the neckline turns from
  the front to the back. The rise is a smooth exponential with that slope at N that levels out by then.
- So the shape follows the neckline. A different neckline (lower, wider, or a different roll top) gives a different
  rise and length with no new numbers. Here the rise runs over 8.9 and 8.3 cm.

As laid, the roll now turns 13 and 12 degrees off the lapels' creases (was about 36), with 0 sharp folds and the
notch at 41.5 and 40.7 degrees.

## How tailors fold a collar, and the soft roll

What the tailoring sources say about folding a collar without creasing it:

- The roll line is pressed into a soft roll, never creased flat. The stand turns over into the fall on a curve, and the top collar is cut larger than the under collar by the turn of the cloth (about 3 mm, up to 6 to 10 mm in thick wool) so it can roll over without pulling.
- The under collar is cut on the bias, stretched along its outer edge and shrunk along the roll line. The fall's outer edge has to match the length of the line it lies on: too short and the collar rides up and creases at the roll, too long and it ripples.
- The fall is a little deeper than the stand, so it covers the neckline seam. A straighter neck edge gives more stand.

Sources: Fashion Incubator (the three collars), University of Kentucky extension CT-LMH.185, A Tailor Made It (top collar manipulation), Minerva (collar and lapel construction), FreeSewing (Simon collar roll), NMSU extension C-223.

What the build does with it: `roll_r` (default 0.08, about 2 mm) turns the stand into the fall on a flat half round, as one cloth with one winding, instead of two layers meeting in a hard fold. The half round fades out as the square of the stand's height toward the lapels, so it doesn't leave a lump where the stand dies away. `roll_r 0` gives the old hard fold. The fall's length was already matched to where it lies (sections above), and the stand is already lower than the fall. The front views lose the double edge and the gap at the sides of the neck; sharp folds stay at 0. A small curl remains where the stand ends next to each lapel, the same as with the hard fold.

## The collar's ends closed

The two lapels cross, so the collar's two ends lie on different ground: the over lapel's roll top sits 0.2 (5 mm) further forward than the under lapel's. The stand's rise is the same on both sides, so that offset is not a cut difference. What showed was that the roll ended in an open mouth beside each lapel, because the fall's corner sits about 0.22 to 0.24 (6 mm) off the neckline's end, on top of the lapel, and the mouth looked worse on the over side.

- `end_taper` (default 0.25): where the stand is lower than this, the turn is drawn in onto the fall's own roll point, so the roll tapers to a closed tip at the fall's corner on the gorge. The cloth's thickness thins with it (the `thick` vertex group on the Solidify), because a turn tighter than the cloth is thick cuts through itself.
- `end_even` (default 0.6): the fall's inner points within this distance of each roll top are relaxed toward their neighbours, edges left alone, which evens the small fold the over side kept.

Both ends are now closed and smaller. A small pinch is still visible at each tip in the close front view, a little larger on the over side.

## The asymmetry: two causes in the cut and the lapels, not in the collar

John (2026-11-22): "the new collar is worse, figure out the root cause of the asymmetry and get the fix". The closed ends (`end_taper`, `end_even`) treated a symptom, so both are off by default now. The two sides differed because the two lapels they sit on differed.

1. **The cut raised the whole right front.** The bodice draft lifts the right front a cloth's thickness off the shell so that it lies over the left front. It did so everywhere on that side, out to the shoulder and up to the neck, so the right neck edge, the right lapel's roll top and the right collar end stood 5 mm further forward than the left's (roll tops at y -2.81 against -2.61). Fix: `bodice_cut.over_only_on_the_lap` (`BODICE_OVER_RUN`, default 0.6 w) keeps the lift where the fronts cross, and the front comes down onto the body past the underlap's edge. The two roll tops now sit at -2.608 and -2.606.
2. **Only the left lapel was pressed.** The traced lapel presses `{'left': 0.85}`. The right lapel's roll stood 1.6 cm high against 0.5 cm, all the way up to the neck, so the collar's two ends lay on different ground. Fix: the `notch_lapel` recipe presses both (`{'left': 0.85, 'right': 0.85}`); the right roll is now 0.7 cm.

Measured, the collar mirrored left to right: mean gap 0.058 before, 0.004 after (1.5 mm to 0.1 mm; 1 unit is 2.62 cm); the front-most point of each side -3.99 and -3.34 before, -3.35 and -3.34 after; the fall depth at the two points 6.1 and 6.5 cm before, 6.5 and 6.5 after; the lapels' mirror gap in the top 4 cm of the fold 0.34 before, 0.01 to 0.02 after. Sharp folds stay at 0.

The lapels still differ lower down, where the right one lies over the left at the crossing, as they should.

## Finishing the rolled collar (John, 2026-11-22: "read how to finish the rolled collar now")

What the sources say about finishing it:

- **Ends and corners.** Clip at the end of the roll line and miter the corners at the points; cut surplus cloth from the corners so they lie flat; trim seam allowances (3/8 in) so nothing bulks up at the end. The upper collar's seam turns under so it lies 1/16 to 1/8 in over the under collar (the build already cuts the fall larger by the turn of the cloth).
- **Pressing.** Steam-shape the under collar along the roll line over the small end of a ham, let it dry, and keep the fall outside the circle. Press the gorge, but never press the break itself or it sets as a hard fold. The roll can only be moved a little: collar tension and the chest canvas set it.
- **Roll-line tape.** Preshrunk twill tape 1/4 to 3/8 in wide, cut 1/2 to 3/4 in shorter than the roll line (shorter for heavy cloth), so the roll line is drawn in and the collar stands against the neck. The build does this with the redrafted neckline and the stand drawn from it.

Sources: UKY CT-LMH.176 (collar and lapel construction), Every Woman's Encyclopaedia, Practical Lessons in Tailoring (the sixth lesson on cutting out and making a coat), Permanent Style (how to correct a flattened lapel), Seamwork (expert coat tailoring techniques), Gentleman's Gazette (jacket ironing).

Applied: the closed ends (`end_taper` 0.25, `end_even` 0.6) are back on by default, now that the two sides sit on the same ground. They are the clipped and mitered end: the turn tapers onto the fall's corner, the cloth thins with it, and the corner is evened, so no open tube shows. Measured, sharp folds stay 0, and the notch pinch beside each lapel is smaller and the same on both sides. The break is left alone, as the sources say.

## The collar settled onto the folded lapels

John (2026-11-22): "the left collar needs more adjustment to its position to compensate for the folding of the lapel, it is too far up; add a smoothing round". Measured, both ends hovered: the fall is laid on a ground with its crevices closed and hung as a membrane between its edges, so over the folded lapels it stood 0.9 mm above the cloth on average and up to 5.9 mm at the points and along the end edge. It showed on the left of the picture because the key light throws the shadow under that side.

- `settle` (default 0.9, `settle_rounds` 4): each point over the bodice is drawn toward its own nearest point of it by an evened share of how far it hovers past the clearance. The move itself is evened over the mesh (`settle_even` 40), and each end is laid straight again from the notch to its lowered point.
- `final_smooth` (30): a last smoothing round over the fall, kept off everything by the nearest point.
- `edge_fair` (12), `edge_in` (20): the free edge faired along itself and the cloth just inside brought along, because where the collar's point crosses the lapel's edge the ground steps down a cloth's thickness.

After: the fall hovers 0.2 mm on average, the end edges lie on the cloth (gap 0.207 against a clearance of 0.207 over most of their length), the notch is 40.9 and 41.0 degrees as laid, sharp folds 0, and the two sides still mirror. The bottom edge at each point is slightly uneven in the close textured view where it crosses the lapel's edge.

## The roll line straightened by the tension solver (2026-11-22)

John: "Create a compression solver that reads our bodice without any lapels or collars then using the tension as a
guideline develop the ideal shape of lapels and collars to reduce tension." `blender/tension/` reads the bare cut's
field and scores every roll line by the strain the turned lapel must take. The current line's swing (`belly` 0.25)
bows into the ring of tension and gathering round the bust; a straight line stays out of it. Built:

```
python3 blender/collar/foldfront.py CUT.blend LAPEL.blend '{"recipe":"derived_lapel"}'
```

Strain mean 4.3/4.7% to 3.1/3.6%, 95% of edges 14.3/15.0% to 9.7/10.9%, worst visible edge +39% to +27%, the pull on
the button pins about halved, the notch laid 40.4 on both sides. The outline is 0.42 wide (was 0.40) so the lapel's
reach stays 0.64 of the half shoulder. The roll top, break, point and collar are unchanged: the field either did not
care or its preferences did nothing built. See `docs/clothing-solvers/lapel-tension-solver.md`.

## The lapel and collar solved together (2026-11-23)

John: "solve for smooth joins between lapel and collar that evenly distributes load". `blender/tension/feasible.py`
swept 57 lapel and collar builds against the approved build's limits (26 feasible), then searched the join. The
chosen pair: `{"recipe": "joined_lapel"}` (the derived lapel with its point raised from 0.27 to 0.24 of the roll line)
and notchcollar `{"notch_angle": 30}`. The collar meets the lapel at 39 degrees across the gorge (90th percentile)
against 50 to 54; lapel strain stays at the derived 9.5 / 10.8% (95th pct). A 0.09 point fillet scored a little
better and tore the right lapel's edge; the stand's settings do not reach the gorge. Numbers and rounds:
docs/clothing-solvers/lapel-tension-solver.md, phase 2.

## The roll line's top eased (2026-11-23)

John: "Fix the roll line". `{"recipe": "eased_top_lapel"}` is `joined_lapel` with `belly_top` -0.14: the line stays
straight over the bust and swings 3.7 mm toward the V near the neck, so it runs into the collar's roll line. With
the notch-30 collar the gorge carries its load more evenly than the approved build (cv 0.42 against 0.51) and the
lapel's edge strain drops to 7.8 / 8.9% (95th pct). Swinging it out instead (+0.12) evens the load but puts the strain
back to 14.9%. Rounds and numbers: docs/clothing-solvers/lapel-tension-solver.md, phase 3.

## The roll line and the collar solved together (2026-11-23)

John: "Can't solve independently the roll and collar the roll change add distortions to the collar you can see".
The eased roll top turns the lapel crease's top tangent, and notchcollar's stand, which rises off each roll top at the
crease's slope, rose at 0.71 instead of 1.13: the fall creased where it meets the roll turn at the side front of the
neck. Worn together: `{"recipe": "eased_top_lapel"}` with notchcollar `{"notch_angle": 30, "rise_gain": 1.6}`, which
restores the rise (1.14) and takes the crease out; every lapel number is phase 3's. Rounds: docs/clothing-solvers/
lapel-tension-solver.md, phase 4.

## The collar's shape (2026-11-23)

John: "How do you solve for shape not just angle of the collar". Three shape options in notchcollar (`stand_shoulder`,
`fall_side`, `end_bow`), started from Mueller & Sohn's stand taper and searched over three rounds. The taper creased
the roll on this low stand, so the stand stays full; a slight hollow in the collar end evens the gorge load (cv 0.43 to
0.39). Worn: `{"recipe": "eased_top_lapel"}` with notchcollar `{"notch_angle": 30, "rise_gain": 1.6, "end_bow": -0.2}`.
Rounds: docs/clothing-solvers/lapel-tension-solver.md, phase 5.

## The high (classical) lapel with the notch collar (2026-11-23)

John: "Run the solver now with the harder problem of a high lapel", "The classic one", then "Collar roll runs into the
lapel". On `classical_lapel` the collar needs the roll line to meet the neck 2.4 out (3.0 knots the collar ends), phase
1's straight roll line, and the new `join_h` / `join_run` options, which start the collar's turn as wide and round as
the lapel's roll at each roll top. Best so far: `{"recipe":"classical_lapel","neck_x":2.4,"foot":48.25,"belly":0.001}`
with notchcollar `{"notch_angle":20,"notch_at":0.6,"point_len":0.5,"roll_r":0.16,"join_h":0.08,"join_run":2}`. Not
approved: a small curl shows where the two rolls meet. Rounds: docs/clothing-solvers/lapel-tension-solver.md, phase 6.
