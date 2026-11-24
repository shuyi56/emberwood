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
| Face flow: the bridge's side walls run out into the cheeks and the inner corners (slope 2.0 to 1.2-1.28, crest wider at the radix, the upper wall turning back at 8 mm, not 5.5), and the bridge sinks into the glabella above the radix; tip, wings and nostrils unchanged | face flow thread (`docs/faces/face-flow.md`) | `apply-face-flow.py` |

The grid code indexes the head through `NS` (columns) and `NR` (rings) everywhere, so the nose's change of column
count carries through the sockets, the mouth block and the lips without edits. The pores bake on the head's UVs and
the eye whites live on the eyeballs' own sphere, so neither depends on the grid.

One fix was made when combining: the pores' colour-coded check read `bumpMap` in the skin shader without
`#ifdef USE_BUMPMAP`. three.js declares that sampler only when the material has a bump map
(`bumpmap_pars_fragment`), and the skin's bump texture is attached a few frames after the first compile, so the first
compile failed ("'bumpMap' : undeclared identifier"). The read is now guarded; the look is unchanged.

Two more fixes after combining (`apply-shade-fix.py`, `apply-eye-makeup.py`, each documented in its docstring):

- **The upper lip's dark trapezoid.** The baked cavity shade measured index neighbours 2 and 7 grid steps away, with
  weights tuned on the old 600-column grid; on the nose's 300-column hybrid grid and the mouth's loops those steps
  are different distances, so the upper lip and the sides of the nose went dark. It is now measured in world space
  (area-weighted discs of 0.07 and 0.25 units, the old calibration). The key light's shadow also had a 0.55 mm
  penumbra; it is now PCF radius 4 (1.5 mm), from the Blender key's size (24 across at about 64).
- **The Ranger's eye make-up.** A mulberry-brown shadow (#6e4652) opposite her green iris and beside her copper hair,
  and a tapered plum-brown liner (#3a2228, 0.8 to 2.4 mm at liner 1) with a short wing, as for a round, open eye.
- **The inner corner of the eye** (`apply-inner-corner.py`, mirrored in `blender/shared/orbit.py`). From 3/4 the near
  eye showed dark red flesh between the ball and the lids at the inner corner: the lid was held to the face's plane
  there, so over the inner 4.5 mm its margin stood up to 10 mm off the ball. Toward the inner corner the lid now lies
  on the ball down to the canthus (lids apposed to the globe; the canthus separated from it only by the small
  lacrimal lake and the caruncle).
- **The skin below the inner corner** (`apply-inner-corner-blend.py`, run after `apply-inner-corner.py`; mirrored in
  `blender/shared/orbit.py`). The inner-corner fix put the ball's depth back into the lid depth, which the corner's
  narrow 5e hand-over band gave back to the face at once, so below the canthus the lid stepped down beside the nose.
  The extra depth is now taken at the margin and added to the surface with a fade over 8e, so the skin slopes into the
  canthus with the lid.
