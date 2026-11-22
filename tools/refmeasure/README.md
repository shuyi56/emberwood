# refmeasure: comparative measurement of reference photos and renders

One shared tool for measuring ratios and lines on a reference image (a photo of a garment, a face, a
boot) and on a render of our model, and listing the gaps between them. It runs headless from Python,
and `measure.html` is a click-and-drag page for placing points by hand. Both read and write the same
sheet format, so a sheet started in one can be finished in the other.

| File | What it is |
| --- | --- |
| `refmeasure.py` | The CLI and library: sheets, unit, plumb, segments, ratios, report, probe, overlay, grid, compare. |
| `measure.html` | Open in a browser. Load an image (and a sheet), click landmarks, read units, probe, save the sheet. |
| `blender_landmarks.py` | In Blender: project 3D points through the render camera and write the model's sheet. |
| `test_refmeasure.py` | Headless self-check (`python3 tools/refmeasure/test_refmeasure.py`). Needs Pillow only. |

## The method, and where it comes from

This is the comparative measurement that ateliers teach, made repeatable:

| Rule the tool follows | Source |
| --- | --- |
| Pick **one unit** and express every size as a multiple of it; every measurement is taken against that unit, never chained off the last one, because "small errors grow into larger ones". | [Wikipedia: Comparative measurement](https://en.wikipedia.org/wiki/Comparative_measurement) (shoulder length as the common unit); [Proko: Measuring techniques](https://www.proko.com/62) (counting heads) |
| Compare three things: **size, position and angle**. Ratios are scale-free (A:B = A′:B′), so the photo and the render need not be the same size. | [Vitruvian Studio, Comparative measurement part 1](https://vitruvianstudio.com/blog/drawing-tutorials-comparative-measurement-part-1/) (after Keller) |
| Prefer **plumb (vertical) and level (horizontal) offsets** to angles: "Horizontal and vertical measurements are more accurate than angles." Hence the plumb line, the level line through every landmark and the alignment list. | [Proko: Measuring techniques](https://www.proko.com/62) |
| Relate the **whole first**, so parts don't drift ("tunnel vision"). The report prints the whole span in units before the parts. | [Proko: Measuring techniques](https://www.proko.com/62) |
| Measure from **one viewing position**: "any variation in position will result in changes in scale and perspective", and foreshortening changes every relationship. `compare` warns when the cameras differ. | [Williamstown Art Conservator: two methods of seeing](https://artconservator.williamstownart.org/objective-life-drawing-two-methods-of-seeing); [Proko](https://www.proko.com/62) |
| Keep the **scale consistent**: each extra scaling step adds error. Every number comes from pixel coordinates in one pass. | [Sight-Size (Rousar): Train your comparative eye](https://www.sightsize.com/train-your-comparative-eye/) |

Sight-size (drawing at exactly the subject's size, side by side) is the other atelier method. The
`compare --png` sheet is its equivalent here: both images are scaled to the same unit and hung from the
same plumb line, so you can look back and forth between them.

## Workflow

Put the reference in a folder with the thread's other files. The sheet is saved next to the image as
`<image>.measure.json`.

```bash
R="python3 tools/refmeasure/refmeasure.py"
$R new habit.jpg --view "front 3/4 left" --projection photo   # start the sheet
$R grid habit.jpg 700 300 1220 650                            # pixel-labelled zoom: read coordinates off it
$R set habit.measure.json shoulder_l 773 408 shoulder_r 1159 397 neck_front 868 452 waist_l 800 836 waist_r 1053 830
$R unit habit.measure.json neck_front waist_l --name torso --axis y   # the unit
$R plumb habit.measure.json neck_front                         # plumb line and origin for positions
$R seg habit.measure.json shoulder_w shoulder_l shoulder_r --axis x
$R seg habit.measure.json waist_w waist_l waist_r --axis x
$R ratio habit.measure.json waist_to_shoulder waist_w shoulder_w
$R report habit.measure.json                                   # sizes, ratios, positions, alignments
$R overlay habit.measure.json                                  # habit.measure.png: unit grid, plumb, levels
$R probe habit.measure.json neck_front collar_point_r shoulder_l shoulder_r   # quick ratio of two spans
```

Then the model, from the same view:

```bash
# Blender (exact points): blender_landmarks.write_sheet(render_png, {name: world_co}, view=..., template=ref_sheet)
# or by hand: render, then
$R new render.png --view "front 3/4 left" --projection persp
$R template habit.measure.json render.measure.json    # same unit, plumb, segments and ratios
$R set render.measure.json ...                         # same landmark names
$R compare habit.measure.json render.measure.json --png gaps.png
```

`compare` lists every size, ratio, angle and position with the reference value, the model value and the
gap, largest first, and flags anything past `--tol` (default 5 %, positions 0.05 unit, angles 3°).
`--gaps` prints only the flagged rows, `--json` saves them. `--png` draws both overlays side by side at
one unit size, plumb lines level. If the render faces the other way, `flip` mirrors it first.

`measure.html` does the same by hand: click to place the landmark named in the box (it then moves to the
next suggested name), drag to adjust, wheel to zoom, Alt-drag to pan, Shift-click two or four points to
probe. "Save sheet" downloads the JSON; put it next to the image and the CLI reads it.

## Choosing landmarks and the unit

* Use the names from `refmeasure.py names` so sheets from different threads line up. `_l` / `_r` are
  image left and right as you look at the picture.
* Pick a unit that both images show clearly and that turning the figure changes least. A **vertical**
  span (head top to chin, neck front to waist) survives a 3/4 view; a width (shoulder to shoulder) is
  foreshortened by the turn, so use it as the unit only on a straight front view.
* A landmark must be the same physical point in both images. "Shoulder" on a puffed sleeve top and on a
  sleeveless bodice are different points, and every ratio built on them will be off.
* Place points on a zoomed grid (`grid`, or zoom in `measure.html`), not on the full picture. One pixel
  on a full 2560 px photo is a few millimetres of garment.
* Museum and catalogue photos are taken with a long lens, close to orthographic. Render the model with a
  long lens or ortho camera from the same height when comparing widths.
