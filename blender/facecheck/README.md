# Face checks without rendering

`facecheck.py` checks a figure's face straight from its Python, with no Blender and no render queue. It runs the
figure's own face code: the head's sections, `face_relief`, `mouth_relief`, `jaw_relief`, the lids, `eye_relief`,
and the paintings for the iris, sclera, eyeshadow, lashes, brows and lips. `bpy` and `bmesh` are replaced by
stubs. From that it draws the face in six views and measures what a face check looks for. A figure built on
`blender/base/female.py` gets the base face with its own colours, exactly as Blender builds it.

It needs only Python 3, numpy and Pillow. With 4 cores it takes about 17 seconds for six views.

**Every face check goes here.** Ask the render queue for a face only when the change is outside what this draws.
That covers hair, cloth, Cycles lighting, subsurface scattering and the sprite palette.

## Command line

```sh
python blender/facecheck/facecheck.py ranger                          # six views + measurements
python blender/facecheck/facecheck.py ranger --before origin/main     # side by side with main, and what changed
python blender/facecheck/facecheck.py rogue --views eyes,lips --size 640 --quality fine
python blender/facecheck/facecheck.py path/to/figure.py --no-images   # measurements only
```

It writes `sheet.png` (one row per run, with a label on each view), one PNG per view, and `report.json`. It also
prints the measurements. With `--before`, the folder also holds a list of every measurement that changed. Every
row is drawn and measured the same way, so a before costs nothing extra. (This is unlike the render queue, which
must never be asked for a before.)

Views: `front`, `three` (3/4), `profile`, `eyes`, `lips`, `ear` (a close-up of her lit right ear), `ear_back` (from
behind, to see how far the ears stand off the skull), and `game`, which is the 3/4 view from the game's high
camera, reduced to sprite scale and blown back up.

A figure that imports `ear_grids` from `blender/shared/ear.py` gets its ears drawn too, set on its own head and
coloured as its `skin_tones` colours them.

## From Python

```python
import sys; sys.path.insert(0, 'blender/facecheck')
import facecheck as fc

fig = fc.load('ranger')                 # a figure name, or the path to a figure's .py
face = fc.build(fig)                    # head, eyes and painting (numpy arrays, model units)
report = fc.measure(face)               # dict, see below
fc.draw(face, 'eyes', size=640).save('eyes.png')
print(fc.diff(report_before, report))   # the measurements that changed
```

## Measurements

| Key | What it is | Good |
| --- | --- | --- |
| `eye.poke_through_px` | White of the eye showing outside the lids' opening, counted in 0.01-unit pixels | near 0 at normal quality |
| `eye.gap_px` | Pixels well inside the opening where skin covers the ball | near 0 |
| `eye.open_w`, `open_h` | The opening's width, and its height from the upper lid's peak to the lower lid's dip (units) | |
| `eye.iris_showing` | Share of the iris disc that shows between the lids | |
| `eye.iris_top_hidden` | How far the upper lid covers the top of the iris (units) | |
| `eye.sprite_px` | The opening's size at game scale, with the head 17 px tall | |
| `features.*_z`, `*_fwd` | The height (in head heights) and forward reach (units) of the nose tip, chin, mouth line, eye line and brow | |
| `widths.*` | The face's half-width at the cheekbones, mouth and jaw (units) | |
| `lips.paint_on_sculpt` | Share of the painted lip that lies on the sculpted lips | high |
| `lips.sculpt_painted` | Share of the sculpted lips that the paint covers | |
| `brows.clearance` | Distance from the brow's lower edge to the lid crease at the eye's centre (units) | |
| `symmetry.max` | Largest left/right difference in the face's relief (units) | 0 |

Compare measurements only at the same `--quality`. The eye counts depend on the mesh's density.

## How it matches Blender

The head is sampled the way `sculpted_head` builds it, only on a denser grid that is finer through the face. The
lids are pushed in with `eye_relief`, and the eyeballs are shaped as in `build_eye`. The paintings come from the
figure's own functions and are laid on the face by their own patch extents (`SH_*`, `BR_*`, `LIP_*`). Three
numbers that live inside functions are read out of their source: the eyes' position in `build`/`build_head`,
and the iris radius and lift in `build_eye`. If one of them can't be found, the report says so under `notes`.

The skin's colour repeats the head's part of `skin_tones` (flush, cool shade, gloss). The drawing is lit simply
and has no shadows. Use it to judge shape, placement and paint, and use the render queue for the final look.
