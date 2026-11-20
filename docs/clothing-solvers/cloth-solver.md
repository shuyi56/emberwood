# Cloth solver

## Holds

- Anything held at all is held whole. Partial pins exploded: cloth half held at the armhole was carried through the
  gambeson's edge and points were flung up to about 3390 units.
- Hold the shirt sleeve whole inside the gambeson sleeve. Do not use the gambeson sleeves as colliders.
- After settling, bring flung points back to their neighbours: any edge longer than 2.5 times its pre-settle length
  is repaired by placing the farther-moved end relative to its unflung neighbours.

## Diagnosing a blow-up

Run with `SOLVER_TRACE=1`. It logs the first frame with a step over 0.5 and the starting positions of the worst
points. The cause so far has been a hold or collider setup, not the stiffness numbers.

## Settings

- Find settings on `blender/chemise/swatch.py` (seconds per trial), not on a whole garment.
- Per-garment overrides go in `ratios['hand']` (see `SHIRT` in `blender/doublet/doublet.py`).

## Layering order

1. Use the lower layer's kept shape (`CHEMISE_CACHE`, `blender/chemise/kept/`).
2. Build only the new layer, no solver, and look at it.
3. Run `CHEMISE_SETTLE=1` on both layers only if needed.
