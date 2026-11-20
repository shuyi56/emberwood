# Solver issues: known limits, their symptoms, and the fixes

Kept by the solver-owner thread ("Own the cloth solver for tuners"). When a flaw looks like the solver, check here
first, then message that thread with the numbers and what you tried. Each entry says whether the cause is a setting,
a missing feature, a bug or geometry.

Solvers covered: the local fabric solver (`blender/shared/fabric.py`), its authored folds (`blender/shared/folds.py`),
and Blender's cloth (`blender/chemise/solver.py`).

## G1. A rolled fold on a curved roll line will not open or lift (geometry, not the solver)

- Symptom: a lapel's hinges close 8 to 16 degrees past target, or the lapel will not stand off the front, and more
  hinge passes (4 to 16) or softer bending (1e-6 to 1e-8) change nothing (stand-off 0.125, 0.125, 0.130, 0.131).
- Cause: the roll line arcs over the chest (3.22 along the cloth, 3.14 chord). Opening the flap needs its outer edge
  to grow about 8 % for 14 degrees, and threads at stretch 1e-8 cannot grow.
- Fix: change the draft (the roll line's curve, or more cloth at the lapel's outer edge), not the solver.

## S1. A floor or body under a folded flap tears the fold's turn (missing feature, fixed)

- Symptom: strain jumps inside the roll of a lapel or collar once a floor holds the flap off the front: 5/95 % of
  -6/15 with the worst edges 300 to 500 % on the lapel.
- Cause: `Folds.bake(body=...)` pushed every free point out of the body, including the turn's own rows, which rise
  from the roll line through the floor's height. The floor pushed the turn open while the hinges closed it.
- Fix: `bake(body=..., body_band=False)` leaves each crease's turn (`Folds.band()`: one row before the line to one
  row past `angle * radius`) out of the body; the hinges hold it. On the lapel: -1.3/3.7 %. On the test sheet
  (`blender/shared/test_folds_band.py`, fold 160 degrees at r 0.16 over a floor 0.28 up): 5/95 % -31.6/108 and worst
  520 % with the band in the body, -15.5/41.8 and worst 242 % without it. Default stays `True` so existing bakes do
  not change; set it to `False` whenever a body or floor lies under or over a rolled fold.

## S2. The fold angle can be on target while the flap lands off its line (missing measure, fixed)

- Symptom: `measure()` reports the fold near target, but the flap sits about 0.09 past its line.
- Cause: with the cloth beside the fold freed the turn leans and spends about 0.20 of cloth instead of its arc (0.307
  for 176 degrees at r 0.10). The rest goes into the flap. The angle across the fold does not see this.
- Fix: `Folds.land()` gives, per crease, where the flap past the band lands against the fold as drafted, in the
  crease's frame as it lies: `along` (+ the flap reaches past its draft: the turn took less cloth than its arc),
  `off` (+ stands off), `worst` and `rms`. `bake()` now prints it on every crease's report line and stores
  `land_along`, `land_off` and `land_worst` in the report. Treat `|along|` over one row as a failed fold even when
  the angle passes.

## S3. Cloth tears in creases of the body, the armpit above all (modelling first; solver mitigation added)

- Modelling cause found on the redingote bodice: the armhole was cut by `max(loop, f_seam)`, and the plumb seam
  surface ran through the crease under the arm, which left a toothed edge and a tongue pointing forward. Fixed by
  cutting by the seam loop alone (`/mnt/project-files/petticoat/bodice-armhole-on-the-loop.patch`). Check the armhole
  cut first when an armpit tears: a cut surface that passes through the crease makes the tear before any solver runs.

- Symptom: ragged edges, slits or knots at the bottom of the armhole and behind the armpit, on several garments
  (chemise knot at the armhole, holes between the vest's armhole and the plates, the bodice's slit at the armpit,
  the undersleeve torn between arm and back). The worst strain of the piece sits in the crease.
- Cause, solver side (it makes a crease worse wherever cloth is pushed off the body there): `MeshBody` pushed each point along the normal of its single nearest surface point. In a
  crease (the torso's side against the arm's inner face, and the rest-pose arm sinks 0.2 to 0.25 into the side
  there) the nearest point flips between the two walls from one cloth point to the next, so neighbours are pushed
  in opposite directions across the crease. The same flip hits any placement that projects onto "the nearest
  skin" (it is how the undersleeve tore between arm and back in `sewn.py`), and Blender's collision on a collider
  that intersects itself.
- Mitigation in the local solver: `MeshBody(..., planes=8)` keeps each point off both walls of a crease at once (the
  nearest sample's plane and the nearest sample facing more than 60 degrees away within `band`), so the push in a
  crease is out along its bisector. Test (`blender/shared/test_crease_body.py`: a torso and an arm with the arm sunk
  0.2, a strip cut 6 % long falling into the crease): worst body push jump between neighbours 5.6 to 2.2 per unit of
  spacing; worst strain in the crease 35 % with `planes=1`, 23 % with `planes=8` (away from the crease 15 to 17 %,
  the strip's own sag). Default stays 1 so existing pieces do not change.
- Still open on the body (anatomy): the rest-pose arm sinks 0.2 to 0.25 into the torso's side. Also, for the garment
  and anatomy threads: give the cloth a body with the armpit bridged, one
  smooth surface from the chest's and back's walls over the pit to the arm (as the gown's fuse fillet does), and
  the arm clear of the side instead of sunk into it. For Blender's cloth, collide with that bridged proxy, never with
  a mesh that intersects itself. Lay the seam and the first rows of the sleeve by one smooth curve from the
  stitching to the arm, never by projecting each point to its nearest skin.

## S4. `pin()` after `mesh()` turned a Gauss-Seidel sheet to NaN (bug, fixed)

- Symptom: every point NaN on the first step, or `cKDTree: 'x' must be finite` from the body.
- Cause: `ties()` drops ties between two held points, but only the points held when the ties are made. Pinned
  afterwards, a tie with both ends held divides 0 by 0 in the Gauss-Seidel walk.
- Fix: the walk skips ties with no free end. Pinning before `mesh()` is still the cleaner order.

## Settings worth knowing

- `Folds.bake(passes=)`: hinge walks per step. More passes do not move a fold that geometry blocks (G1).
- `Folds.bake(correct=True)`: at rest but off target, the hinges' rests are scaled by the shortfall, up to 178
  degrees. A fold that closes past target on a curved front is this correction meeting G1, not a hinge fault.
