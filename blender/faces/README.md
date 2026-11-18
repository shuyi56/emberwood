# Faces from the face creator

Each file here is one face exported from the face creator (`/mnt/project-files/face-creator/face-creator.html`)
by `tools/face-creator/export-face.cjs`: every setting, the colours the creator derives from them, and the
creator's own head surface. `blender/base/creator.py` puts one on any figure built on the base figure, so the
Blender head is the creator's head on the Blender body.

| file | the creator's head preset |
| --- | --- |
| `ranger.json.gz` | Ranger |
| `rogue.json.gz` | Rogue |
| `warden.json.gz` | Warden |
| `knight.json.gz` | Knight (Sex 1: the male head) |

## Making one

A preset:

    node tools/face-creator/export-face.cjs /mnt/project-files/face-creator/face-creator.html blender/faces/rogue.json.gz --face Rogue

Your own face: press "Copy settings (JSON)" in the creator, save what it copies to a file, then

    node tools/face-creator/export-face.cjs /mnt/project-files/face-creator/face-creator.html blender/faces/mine.json.gz --settings mine.json

(`--face` and `--settings` together apply the preset first, then your settings over it.) Check it with

    python3 tools/face-creator/check-face.py blender/faces/mine.json.gz

which reports how closely the Blender side reproduces the creator's head (it should be under 0.001 head heights
over the face).

## Using one

On any figure, without touching its script:

    python3 blender/base/views.py --figure warden --face blender/faces/mine.json.gz --head --out DIR

or in a figure's script, after its own face settings and before it builds:

    import creator
    creator.apply(os.path.join(HERE, '..', 'faces', 'mine.json.gz'))

## What carries over

- **The head's shape, all of it**: every head-shape slider, Sex, the jaw-to-chin taper, the cheekbones, eye
  spacing, the sculpted lips and nostrils, the throat. Blender reads the creator's surface instead of rebuilding it
  from the sliders, so nothing has to be ported when the creator's sculpt changes; export the face again.
- **Eyes**: the lids' construction (open, tilt, arch, corner angles) and the iris colour ramp.
- **Skin**: skin tone (the whole ramp), flush colour and amount, freckles, blotchiness, blush and its tone, the
  cool shade under the jaw.
- **Make-up**: eyeshadow colour and strength, how far it runs in, liner and wing, lip colour (the lipstick mixed
  in) and gloss.
- **Brows**: colour, arch, thickness and height (lower, flatter and heavier on a male face).
- **Hair colour**: the hair material takes the creator's hair tone. The figure keeps its own hairstyle.

Not carried over yet (`creator.UNUSED` lists them; `apply` prints them): contour and highlight, the smudge and
lower-lid shadow, the lash settings, brow length and density, the iris's fine detail (size, fibres, crypts), the
skin's oil, texture and subsurface amount, the hairstyle itself, and facial hair.
