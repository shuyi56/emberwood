# The tension solver for the lapel and collar: retrospective

Thread "Build compression solver for lapel shape" (2026-11-22). John: "Create a compression solver that reads our
bodice without any lapels or collars then using the tension as a guideline develop the ideal shape of lapels and
collars to reduce tension." Code in `blender/tension/` (see its README; kept in one folder so the solver
consolidation thread can move it whole). Causes are marked **tuning**, **modelling** or **process**, as in the lapel
and collar retrospective.

Units: 1 model unit is 2.62 cm. Strain numbers are foldfront's own (each edge as folded against its cut length)
unless marked as the solver's.

## What was built

- A solver that takes the bare cut (`bodice_cut`'s CUT.blend: two fronts to the neck, no lapel, no collar), cuts
  each front as a tailor would (its two darts slit open), finds its best flat pattern (ARAP, Liu et al. 2008, from an
  LSCM start) and reads the field: where the cloth is pulled longer than cut and where there is more cloth than room,
  with the directions.
- An evaluator that opens a front on a candidate roll line (drawn exactly as foldfront draws it), re-flattens it,
  turns the lapel over the line onto the front one cloth's thickness out, and gives the strain the turned cloth must
  take, with the lapel's area held at the approved design's.
- Searches over the roll line (top on the neck, break, swing off the chord: 144 designs per front) and the outline
  (point position and long-edge swell: 35 per front), then real builds of the promising ones through foldfront and
  notchcollar, which decide.
- A collar gauge (the best flat cut of the built fall and what it still has to stretch) and sweeps of the collar's
  stand, fall, lean, stand run and rise (15 builds).

## What the field says about the bare front

- Taken flat, each bare front needs little: 95% of the cloth within 1.7% (her right) and 1.9% (her left) of its
  cut, symmetric. The fronts agree with each other to 0.2%.
- Tension sits over the bust above the first dart's point; compression rings it, strongest at the front edge at
  bust level (z 44 to 46.5, 3 to 4% gathered) and at the side. The upper chest where the lapel lies is under 1%.
- Along the current roll line the bare front is neither pulled nor gathered (0.1%, a fraction of a millimetre).
  So the roll-line tape (1/2 to 3/4 in shorter, UKY CT-LMH.176) is not there to take up excess the shape leaves; it
  draws the roll in. The excess is at the break, which is where opening the front gives it somewhere to go.

## What the field changed, and what it left

| Choice | Field (solver's turned-lapel energy) | Built | Kept |
|---|---|---|---|
| Roll line swing (`belly`) | 0.25 (current) 0.0169; straight 0.0119; at every top and break the straight line wins, a hollow one (-0.1) gathers the lapel (5th pct stretch 0.80) | strain mean 4.3/4.7% to 3.1/3.6%, 95% 14.3/15.0% to 9.4/10.7% | **straight** |
| Roll line top (`neck_x`) | falls steadily to 1.5 (0.0062 against 0.0119 at 2.2) | 1.9: mean 3.1/3.6% (no gain), notch laid 37.5 for 40; 1.6: 2.9/3.5%, notch 43 | 2.2 (the tailoring position: one stand height past the side neck point) |
| Break (`foot`) | flat between 1.35 and 2.0 below the bust line | not built | -1.35 (approved) |
| Point down the line (`point_t`) | lower is better (0.45: 0.0048) | 0.35: mean 2.8/3.4%, but reach 0.56 of the half shoulder (photo 0.62), point_t 0.38: a different lapel | traced 0.27 |
| Long edge swell | 0.05 slightly better than 0.025 | cusp mean 3.0 to 2.7%; **a torn patch of inverted faces at the lapel's edge by the point** (rejection logged) | 0.025 (approved) |
| Outline width | (area held) | the straight line moved the point 3.5 mm in: reach 0.61 against 0.64 (refmeasure: -5.1%) | widened 0.40 to 0.42: reach 0.64 |
| Collar stand, fall, lean, stand run, rise | the built fall is already near flat-cuttable: 95% within 2.1 to 2.3% in every one of 15 builds; its strain sits at the sides of the neck where the fall turns over the shoulder | no measurable change | unchanged |

The derived lapel, built:

| | Current (notch_lapel) | Derived |
|---|---|---|
| foldfront options | `{"recipe":"notch_lapel"}` | `{"recipe":"notch_lapel","belly":0.001,"outline":[[0.04,0.0],[0.27,0.42],[1.0,0.0]]}` |
| strain mean, her right / left | 4.3% / 4.7% | 3.1% / 3.6% |
| 95% of edges under | 14.3% / 15.0% | 9.7% / 10.9% |
| worst, her right (visible) | -27.6% / +39.0% | -18.9% / +26.8% |
| at the cusp, quick solve 95% | 9.8% / 10.9% | 4.0% / 4.3% |
| top of the fold, mean | 3.0% / 2.9% | 2.8% / 2.8% |
| pull on the button pins | 0.031 / 0.009 | 0.016 / 0.013 |
| notch as laid (asked 40) | 40.9 / 41.0 | 40.4 / 40.4 |
| lapel mirror gap, top 4 cm | 0.011 | 0.013 |
| collar flat-cut 95% stretch | 2.1% | 2.2% |
| refmeasure (front ortho, shoulder unit) | | every ratio within 0.5% of current; the lapel points 2.6 mm lower |

Her left lapel's worst edge (-84 to -93%) is the under lapel hidden behind the upper front, pressed flat there on
purpose, in both builds.

## What failed, and why

| Problem | Numbers | Cause | Fix |
|---|---|---|---|
| The first flattening had triangles stretched 35 times | 345 bad triangles, all at the armhole's turn | **modelling**: the front taken out to the armhole, where the shell folds back on itself in the front view, and a poor start (the paint unwrap) | the front bounded at x 3.5, and an LSCM start: 0 bad triangles, 0 flipped |
| The field was ruled by the bust | 20% and more over the bust with no darts | **modelling**: the bodice has two front darts; a front flattened whole has to stretch over all of the bust | the darts slit where `redingote/bodice.py` places them |
| The turned lapel came out 6 times stretched | 95th pct 9.1 | **modelling**: the roll line's points on the re-flattened pattern jittered by a triangle | the line faired (a fifth-degree fit) before the lapel is cut along it |
| The search favoured the smallest lapel | best area 5.9 against 12.3 | **modelling**: strain per area grows with the lapel, so the field alone says "no lapel" | the area held at the design's |
| The field's best roll-line top did nothing built | 3.1% at both 2.2 and 1.9 | **modelling**: the evaluator does not see the collar, which has to come further down the neck to meet a lower top, and the turned cloth it loses off the neck edge at 2.2 (1.6%) is where the collar lies anyway | kept the tailoring position; noted in derive.py's limits |
| A fuller long edge tore the lapel's edge | inverted faces by the point (her right) | **modelling** in foldfront's outline mesher: a swell of 0.05 puts the edge points where the rows meet them badly | swell kept at 0.025; the mesher not changed here |
| Collar parameters moved nothing | flat-cut 95% stretch 2.05 to 2.27% over 15 builds | not a fault: the fall is laid to the cloth under it, so it is near developable whatever its depth; `stand_run` is ignored while `from_neck` is on | collar unchanged |

## Which checks caught problems, and which missed them

- **Caught:** the bad-triangle count of the flattening (s1 > 1.6 or s2 < 0.6) found the armhole fold and the bad
  start at once. The real build caught that the field's lower roll top gains nothing. refmeasure caught the 3.5 mm
  loss of reach that no strain number shows. The clay close-up off the centre line (`clay.py`'s `at`) found the
  torn edge.
- **Missed:** foldfront's strain print did not flag the torn edge: the patch is a few faces, inside the 95th
  percentile. A count of inverted faces on the turned cloth would catch it.
- **Misleading:** a developability gauge on the built lapel (its own best flat cut) read 1.7 to 2.2% for every
  build: foldfront's relaxation makes the turned cloth developable, just not to its own cut. The strain against the
  cut is the number that matters, and that is foldfront's.

## What to do differently

- Build every solver choice before believing it. Three of the field's five preferences did nothing or did harm
  built, because the evaluator leaves out the collar and the mesher.
- Hold the design (area, reach) fixed in any search, or the field just shrinks the piece.
- Measure the derived build with refmeasure before calling it done: a strain win that moves the point is a design
  change.
- Next: the evaluator should include the collar's ground (where the fall lies past the neck edge), and foldfront's
  outline mesher should be checked with an inverted-face count.
