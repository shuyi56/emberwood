# Face Creator tools

The Face Creator itself (`main.js` and `face-creator.html`) lives in the project's shared folder
(`face-creator/`), not in this repository. These tools work on it.

## The layered skin (`skin-layers.js`)

The creator draws Blender's layered skin (`blender/base/skin_layers.py`, PRs #41 and #42): a red and brown flesh under
the skin, veins where the skin is thin, the creator's own painted epidermis laid over both at `EPIDERMIS_OPACITY`, sun
spots, moles and blemishes, a slow red and yellow drift in the tone, patchy roughness, a thin oil film, a soft down,
and the warm bounce light from below. The Skin tab's "Layers" group has the controls (Layered skin, Veins, Sun spots,
Moles); Layered skin 0 is the creator's skin as it was.

How it is the same skin as Blender's:

- **The numbers.** `skin-params.mjs` writes every constant of `skin_layers.py`, the head's place on the figure
  (`female.py` `HEAD_H`, `HEAD_Z`) and the look's key and bounce lights (`views.py`) into `main.js` between its
  `SKIN_LAYERS` markers. Run it after any change there; `--check` fails if `main.js` is out of date.
- **The patterns.** Blender's Noise (Perlin fBm) and Voronoi (F1, distance to edge) are rebuilt in GLSL, hash for hash,
  and read at the same rest position (model units, chin at `HEAD_Z`). So a vein, a mole or a sun spot sits where
  Blender puts it. `skin-noise-check.py main.js` checks them against Blender's own nodes (all within 1e-4; at writing
  within 2e-5).
- **Where they show.** `skin_tones`' thin-skin and freckle zones on the head (the same gaussians), with `paint_face`'s
  hold of the face's thin skin to 0.3.

The patterns are baked once per head shape into three maps over the head's wrap-around coordinates, so the face costs
three texture reads; the paint sliders only change uniforms.

What the creator cannot do, and what it does instead:

- **Subsurface scattering**: normalised wrapped diffuse light per channel, red reaching furthest in Blender's radius
  ratio (1 : 0.36 : 0.18, `SSS_SCALE` over a 0.8-unit curvature), held back where pigment and pits sit as Blender holds
  them out of the scatter. It softens the terminator and reddens it, but it does not blur fine detail and does not
  deepen the colour as Cycles' random walk does (on the bench Blender stays about 4/255 darker in red and 9/255 in green
  and blue).
- **The key light's shadow** is kept off the diffuse where the skin turns away from the light. The shadow map shadows a
  convex cheek from itself at the terminator, and the wrapped light reaching past it was cut into a hard edge across the
  face. The specular still takes the shadow whole, and a cast shadow (the nose's) falls on skin that faces the light, so
  it stays.
- **The bounce light** is added to the skin alone, so the hair's look and the lights a hair asset exports do not change.
- **Pores and freckles** stay the creator's own (its pore relief, its ranked freckle patches that the Freckles slider
  drives). The creator takes from Blender only the roughness in the pits and the hold out of the scatter.
- **Ears** keep the creator's own ear shading.
- **The game export** (`tools/creator-head/export-head.js`) exports the plain material and painted maps, as before.
  The layers are shader work and do not export.

To lay the port onto the current `main.js` (for example after another change to the creator):

    python3 tools/face-creator/apply-skin-layers.py main.js tools/face-creator/skin-layers.js
    node tools/face-creator/skin-params.mjs main.js

## The skin bench

The bench renders a creator head in Blender with `skin_layers.py`, on the creator's cameras and light, beside the
creator's own render:

    node tools/face-creator/dump-head.cjs face-creator.html DUMP Ranger
    python3 blender/base/skin_creator_bench.py DUMP --out OUT [--views three,cheek,nose,temple] [--denoise]

The dump carries the head, its painted face, mask, eyes and ears, the views' cameras, the lights, and the creator's
ambient light measured on a white and a black sphere. `PLAIN=1` (or `PLAIN=white`) on the dump with `--plain`
(`--white`) on the bench checks the light alone: a plain grey head matches within about 2/255.

## Hair

`export-hair.cjs` exports a style for Blender (see its header). The layered skin does not touch the hair: the
Ranger's export is byte-identical with and without it.
