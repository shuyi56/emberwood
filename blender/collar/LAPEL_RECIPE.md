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

