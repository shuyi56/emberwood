# The Face Creator

`face-creator.html` is the live Face Creator as it stands in the project files (`face-creator/face-creator.html`),
snapshotted on 2026-11-21 with all of the recent face work in it. Open it in a browser; it loads three.js 0.160.0 from
jsdelivr. `main.js` is the same module body without the page's first part (imports, eye constants, sclera), the file
the `apply-*.py` scripts edit and the body viewer is rebuilt from.

What the snapshot holds, oldest first:

| Change | Where it came from | Re-apply |
| --- | --- | --- |
| Eye sockets carved into the head, a round eyeball set in each | eye sockets thread, PR #71 (`orbit.js`, `blender/shared/orbit.py`) | in the snapshot |
| Layered lips: smooth pink under layer, rougher top coat with lip lines | lips thread (`blender/base/female.py`, `skin_layers.py`) | `apply-lip-layers.py` |
| Nose rebuilt (hybrid head grid of 93k vertices, 300 columns round the head, nose u1, new lights) | nose thread | in the snapshot |
| Creamy eye whites | lips thread | `apply-sclera-milk.py` |
| Eye-white vessels rooted at the corners, five trunks fanned per corner | lips thread | `apply-sclera-veins.py` |
| Pores in the skin's layers: 64/cm2 at 0.16 mm2 (Flament 2015), per-colour diffuse normals, occlusion not paint | pores thread | `apply-pores.py` (on `face-creator.html` or `main.js`) |

The grid code indexes the head through `NS` (columns) and `NR` (rings) everywhere, so the nose's change of column
count carries through the sockets, the mouth block and the lips without edits. The pores bake on the head's UVs and
the eye whites live on the eyeballs' own sphere, so neither depends on the grid.

One fix was made when combining: the pores' colour-coded check read `bumpMap` in the skin shader without
`#ifdef USE_BUMPMAP`. three.js declares that sampler only when the material has a bump map
(`bumpmap_pars_fragment`), and the skin's bump texture is attached a few frames after the first compile, so the first
compile failed ("'bumpMap' : undeclared identifier"). The read is now guarded; the look is unchanged.
