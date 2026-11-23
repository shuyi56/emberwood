# Lapel and notch collar: retrospective and lessons learned

Thread "Fit the notch collar", PR #59 (2026-11-22). The redingote bodice with a wide rounded lapel and a notch collar,
built in Blender in the tailor's order: cut, fit, fold, relax, button last. The step by step record is
`blender/collar/LAPEL_RECIPE.md`; this file keeps what to carry forward. Causes are marked **tuning** (a number set
wrong), **modelling** (the shape was drafted wrong) or **process** (how the work was checked or ordered).

Units: 1 model unit is 2.62 cm. The cloth is 0.167 thick (4.4 mm).

## What was built

- `blender/bodice_cut/bodice_cut.py`: the cut. Two whole fronts, the right lapped over the left. It now also
  redrafts the neckline to the collar (`neck_fit`) and raises the right front only where it lies over the left
  (`over_only_on_the_lap`).
- `blender/collar/foldfront.py`, recipe `notch_lapel`: the traced wide lapel with a rounded point, both lapels pressed
  alike, relaxed as cloth, buttoned last.
- `blender/collar/notchcollar.py`: the collar. A stand drafted from the neckline, a fall laid over the stand and the
  lapels on a chart round the neck, a 40 degree notch, a soft rolled turn at the roll line, closed and mitered ends,
  and a last settle onto the folded lapels.

Build chain and times: cut about 2 min, lapels about 40 s, collar about 45 s. A textured render at 32 to 48 samples
takes 4 to 6 minutes a view; clay renders at 8 to 16 samples take seconds and catch almost everything.

## What failed, and why

| Problem | Numbers | Cause | Fix |
|---|---|---|---|
| The two collar ends looked different | mirrored left to right the collar differed 1.5 mm on average; front-most points at y -3.99 and -3.34 | **modelling**, in two places upstream of the collar (below) | fixed at the source: 0.1 mm after |
| The cut raised the whole right front | a cloth's thickness (0.167) up to the neck and out to the side; roll tops at y -2.81 against -2.61 | **modelling**: the lift for the overlap was applied to the whole front, not to the overlap | lift kept between the lap lines, fading out over 0.6 w past the underlap |
| Only the left lapel was pressed | right roll 1.6 cm high against 0.5 cm; lapels 0.34 apart when mirrored | **tuning** inherited from the traced lapel recipe (`press={'left': 0.85}`) | both pressed 0.85: right roll 0.7 cm, lapels 0.01 to 0.02 apart at the top |
| The collar hovered over the lapels | 0.9 mm on average, 5.9 mm at the points and end edges | **modelling**: the fall lies on a ground with its crevices closed and hangs as a membrane between fixed edges | `settle`: drawn onto the cloth by an evened move, 0.2 mm after |
| Stand and fall met in a hard fold | two edges and a gap at the sides of the neck | **modelling**: two pieces, no turn of cloth | one cloth over a flat half round of 0.08 (2 mm) |
| The roll ended in an open mouth | the fall's corner sits 6 mm off the neckline's end | **modelling**: the roll simply stopped where the stand died away | the turn tapers onto the fall's corner and the cloth thins with it |
| The collar ring was too long and creased | 34 sharp folds | **modelling**: band too high and too wide for the neck | lower, shallower band; stand 1.0 cm; folds 0 |
| The back of the collar floated | visible past the neck in the front view | **modelling**: the bodice neckline was not drafted to the collar | `neck_fit` in the cut; stand drawn from the neckline (`from_neck`) |
| Pressing the fall point by point | 714 folds; evened version 71 | **modelling**: a point by point press copies every step under it | not used; a lower stand and `settle` do the job |
| Notch laid at 18 and 39 degrees for 30 asked | diverged when corrected on the chart | **modelling**: the end edge was left to the chart | the end edge is fixed straight from notch to point in 3D |

## The asymmetry, in detail

This cost the most time, so it is worth spelling out.

- The first read was that the collar needed a per-side adjustment of about 5 mm. That was wrong. The collar's stand
  profile was already identical on both sides to three decimals.
- Two rounds were then spent on the collar's ends (ending the stand earlier, closing the ends). Each made the ends
  smaller but left them different, and John called the result worse.
- The cause was found only by measuring each stage against its own mirror image: the cut, then the lapels, then the
  collar. The cut was off by a full cloth's thickness on the right front, and the lapels by 6 mm because of the
  one-sided press. The collar was faithfully following two different lapels.

Rule: **when a downstream piece is asymmetric, mirror-test each upstream stage before touching it.** Three short
scripts did this (`asymtrace`, `sidegap`, `edgegap` style checks): mirror the points in x, take the nearest-point
distance, and report by height band.

## Which checks caught problems, and which missed them

- **Caught:** the sharp-fold count (adjacent faces with a normal dot under 0.5) for creases; the mirror gap for
  asymmetry; the hover measure (distance to the cloth beneath, past the clearance) for floating; clay renders in
  front, both three-quarters, side and back.
- **Missed:** the sharp-fold count read 0 while the collar still looked wrong, three times: the hard fold at the roll,
  the open ends, and the hover. It only counts creases inside the fall's own mesh.
- **Missed:** symmetric defects. The hover was the same on both sides and showed on one only because of the key
  light. Mirror tests say nothing about it.
- **Missed:** clay renders hid two dark specks on the lapel edges that the textured render showed. Render textured
  before calling a round done.
- **Misleading:** close-up cameras that centre on x = 0. Several rounds of zooms looked at empty cloth. Use a camera
  with an `at` offset (`clay2.py`), or render wide at high resolution and crop.

## What tailoring sources say, and what was used

- The roll line is pressed into a soft roll and never creased flat; the top collar is cut larger by the turn of the
  cloth. Used: `roll_r`.
- The roll-line tape is cut 1/2 to 3/4 in shorter than the roll line so the collar is drawn in. Used: the neckline
  redraft and the stand drawn from it.
- Clip at the end of the roll line and miter the corners flat. Used: `end_taper`, `end_even`.
- Press the gorge, never the break. The break is left alone.
- The fall is slightly deeper than the stand, and the fall's outer edge must match the line it lies on.

Sources are listed in `LAPEL_RECIPE.md`.

## Still open

1. **The pinch at each collar tip.** Where the stand runs out beside the lapel, a small pinch is visible in the close
   front view, the same on both sides. The roll meets the lapel's own fold there as two separate surfaces. The fix
   is to join the collar's end to the lapel's fold as one surface along the gorge, not more smoothing.
2. **Two dark specks on the lapel edges in the V**, one on each side, seen only in the textured render. They look
   like stray points on the lapel's edge from `foldfront.py`. Not traced.
3. **The bottom edge at each collar point is slightly uneven** after `settle`, where the point crosses the lapel's
   edge and the ground steps down a cloth's thickness. `edge_fair` and `edge_in` reduce it.
4. The redingote modules the cut depends on (`blender/redingote/bodice.py`, `redingote_bodice/`) are not in this
   repository, so `over_only_on_the_lap` patches `bodice.surface_lift` from `bodice_cut.py`. The proper home for
   that fix is `surface_lift` itself.

## What to do differently

- Mirror-test the cut and the lapels as soon as they are built, before any collar work.
- Give a recipe's inherited one-sided settings (press, tail_out) a second look whenever a new piece is added on top.
- Fix at the stage that causes the problem. Symptom fixes on the collar were reverted once and re-enabled only after
  the ground under it was right.
- Stop after two rounds that change little and report, with the measurement that did not move.
- Measure before rendering: mirror gap, hover, sharp folds and the notch angle take seconds; a textured view takes
  minutes.
- Keep close-up render tooling that can aim off the centre line from the start.
