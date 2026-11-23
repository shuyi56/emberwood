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

# Phase 2: the coupled lapel-collar solve (2026-11-23)

John: "Solve for the geometry with a couple solver first find a feasible space for tension then solve for smooth joins
between lapel and collar that evenly distributes load sweep angle shape size". Stage 1 maps where tension stays inside
limits across lapel and collar together; stage 2 picks, inside that region, the join that turns the collar onto the
lapel most smoothly and spreads the load most evenly along the gorge. Code: `blender/tension/join.py` (the join gauge)
and `blender/tension/feasible.py` (limits, map, objective).

## The join gauge

Per side, on the gorge seam (the collar's boundary that lies on the lapel's top edge): the gap to the lapel's edge,
the turn between the collar's and the lapel's surfaces across the seam (from a plane fitted to each cloth within
0.3 units, 7.9 mm), the seam line's kink, the strain on each side in a 0.4 band (the lapel's against its cut, the
collar's against its own ARAP flat cut), cut into eight stretches along the seam: the coefficient of variation of each
side's stretches (evenness, 0 is even) and the lapel's worst stretch over its mean (peak), and the collar tip's fold.

## Stage 1: the feasible region

No published strain limit for a fulled wool was found (Kawabata's KES-F gives tensile, shear and bending measures, not
tailoring thresholds), so every limit is the approved build's own number: lapel edge strain 95th percentile under 15.0%
on both fronts, the worst visible edge under +39%, the field's turned-lapel energy under 0.0169, the collar's flat cut
under 2.5% at the 95th percentile with no neighbouring faces over 60 degrees apart, the notch laid within 2.5 degrees
of the angle asked, the seam closed (under 8 mm, a cloth and a half), and the collar tip's fold under 15.6 degrees (the
approved builds' 13.6 to 14.1, plus 10%).

Sweep: 7 lapels (width 0.38 / 0.42 / 0.46 × point 0.24 / 0.27 / 0.30 of the roll line), a full 27-build collar grid on
the derived lapel (notch angle 30 / 40 / 50 × notch position 0.35 / 0.5 / 0.65 × collar point 0.55 / 0.7 / 0.85) and a
5-point star on each other lapel: 57 builds, **26 feasible**. Every lapel passes the tension limits (95th percentile
10.3 to 11.4%); what fails is the notch not laying at the angle asked (most failures, 2.6 to 8.0 degrees off, worst at
notch position 0.65 and point 0.55), one build with sharp folds (w0.38 p0.30, notch 50) and the tip fold at
w0.46 p0.24 notch 40 at 0.35. Map: `/mnt/project-files/lapel-tension/phase2-feasible-map.png`.

## Stage 2: the join, three rounds

Objective: evenness / 0.52 + turn (90th pct) / 53.3 + tip fold / 13.6, each over the approved build's value, so the
approved build scores 3.0. The lapel-to-collar load ratio is reported, not scored: lowering it by loading the collar
would add tension, not spread it.

| Round | Varied | Best | Gain | Verdict |
|---|---|---|---|---|
| 0, stage 1 | lapel width × point, notch angle × position × collar point | w0.42 p0.24, notch 30 at 0.5, point 0.7: 3.00 | derived 3.27 to 3.00 | notch 30 more even than 40 on all 6 star lapels (cv 0.51 to 0.63 against 0.59 to 0.72) |
| 1 | the stand at the gorge: stand_run 2.6 to 4.0, roll_r 0.05 to 0.14, end_taper 0.15 to 0.4 (27 builds) | all identical | none | the fall mesh came out byte-identical: the stand does not touch the gorge (rejection 118) |
| 2 | the gorge from the lapel side: where the top edge leaves the roll line (0.02 / 0.04 / 0.08), the point fillet (0.05 / 0.09), the collar tip rounding (0.15 / 0.30) | gorge from 0.04, fillet 0.09, notch 40: 2.90 | small (3%) | the 0.09 fillet tore the right lapel's edge at every mesh size tried (rejection 119); tip rounding 0.30 creased the collar (15 to 28 sharp folds) or stretched it to 11%; gorge from 0.08 pinched the tip (22 to 25 degrees) |

Two rounds of small or no gain: stopped, per the iteration rules. The chosen build is round 0's best that is clean in
close-up: `RECIPES['joined_lapel']` (the derived lapel with its point at 0.24) with notchcollar's notch_angle 30. Width
0.46 at the same point and notch tied it (2.997 against 2.999, inside the sweep's spread); 0.42 is kept because it
holds the reach refmeasure checked in phase 1, and a wider lapel carries more edge strain (11.2% against 10.8%).

## Before and after (her right / her left)

| Measure | Approved | Derived (phase 1) | Joined (phase 2) |
|---|---|---|---|
| lapel edge strain, mean | 4.3 / 4.7% | 3.1 / 3.6% | 3.1 / 3.6% |
| lapel edge strain, 95th pct | 14.3 / 15.0% | 9.7 / 10.9% | 9.5 / 10.8% |
| worst visible edge (her right) | +39% | +26.8% | +26.0% |
| turn across the gorge, 90th pct | 52.9 / 53.8° | 50.8 / 50.3° | **38.6 / 39.0°** |
| turn across the gorge, mean | 33.2 / 32.9° | 32.5 / 32.2° | 31.5 / 31.4° |
| seam kink, 90th pct | 39.0 / 38.3° | 41.0 / 40.9° | 37.4 / 37.1° |
| load evenness along the gorge (cv, lower is even) | 0.51 / 0.54 | 0.65 / 0.68 | 0.62 / 0.63 |
| lapel's worst stretch over its mean | 1.76 / 1.83 | 2.14 / 2.23 | 2.45 / 2.63 |
| lapel / collar load at the gorge | 2.85 / 0.52% | 2.83 / 0.49% | 2.63 / 0.58% |
| collar tip fold | 13.6 / 13.6° | 13.9 / 14.1° | 11.3 / 14.5° |
| collar flat cut, 95th pct | 2.12% | 2.16% | 2.16% |
| notch as laid | 40.9 / 41.0° | 40.4 / 40.4° | 30.2 / 30.1° |
| objective | 3.00 | 3.27 | 3.00 |
| refmeasure, lapel point reach / drop (shoulder unit) | | 0.550 / 0.442 | 0.561 / 0.418 (both sides equal to 0.001) |

What it bought: the collar meets the lapel at 39 degrees instead of 50 to 54 across most of the gorge, with the lapel
strain kept at the phase 1 level. What it did not: the load along the gorge is still less even than the approved
build's, because straightening the roll line in phase 1 put more of the lapel's strain at the roll end of the gorge
(the peak 1.8 to 2.4 times the mean). Nothing on the collar side moves that; the next lever is the roll line's top
itself, which ties back into phase 1's field. Her left lapel's hidden cusp (the last fifth of the under lapel, behind
the upper front) reads +74% worst against +60% approved and +64% derived.

## What failed in phase 2, and why

| Problem | Numbers | Cause | Fix |
|---|---|---|---|
| Every build failed "lapel torn" | dihedral 170 to 180° on all 57, the approved build too | **gauge**: the lapel's points are triangulated on their chart, and the slivers this makes read as folds | the gauge reported, not applied; the worst-edge limit is what caught phase 1's swell tear |
| The turn across the seam jumped 10 to 60° point to point | turn max 61 to 83°, the same on every build | **gauge**: face normals of the same chart triangulation | a plane fitted to each cloth within 0.3 units; the profile then holds at 0.12, 0.2 and 0.3 |
| Evenness mostly measured the lapel/collar ratio | pooled cv 1.0 on every build | **gauge**: one cv over both sides, one carrying 5 times the other's strain | cv per side, averaged |
| The stand's settings did nothing | 27 identical fall meshes | **design**: the stand hangs from the fall's roll line; the gorge is the fall's edge | rejection 118 |
| The best-scoring lapel tore | inverted faces on her right lapel's edge below the point | **modelling** in foldfront's outline mesher at a 0.09 fillet, at lapel_mesh 0.065, 0.07 and 0.075 alike | rejection 119; fillet kept at 0.05 |

## Which checks caught problems in phase 2

- **Caught:** rendering the approved build through every gauge exposed the torn-lapel and turn gauges as noise (the
  approved build failed them). The md5 of the fall mesh showed round 1 changed nothing. The clay neck close-ups
  (both 3/4 sides) and a 1.6-unit close-up found the fillet tear that no number flagged.
- **Missed:** none of the join numbers sees a torn edge: the tear is off the gorge.
- **Next:** an inverted-face count on the turned lapel, the gap noted in phase 1, would have caught rejection 119 in
  the sweep instead of at verification.

# Phase 3: the roll line's top (2026-11-23)

John: "Fix the roll line". Phase 2 left the gorge load less even than the approved build's (cv 0.62 against 0.51), with
the lapel's worst stretch of the gorge at its roll end, 2.4 times the mean. Straightening the roll line in phase 1 took
away the approved line's swing near the neck along with the swing over the bust.

Source: the historical-tailoring masterclass on Devere's 1866 manual (masterclasses.historical-tailoring.com, November
2019) calls the alignment of the roll lines at the neck "the most crucial point of the entire process, you want to make
sure that the roll line of the forepart and the collar align exactly", and has the collar's roll line curved near the
notch, "at about the half way point, the curve of the roll line straightens out". A roll line straight from the break to
the neck point meets the collar's curving roll line at an angle, and the lapel takes that angle as strain at the gorge's
roll end. Mechanics: the turned lapel is the cloth mirrored over the line, so a line bowed toward the V near the neck
shortens the mirrored distance there.

Change: foldfront's `belly_top`, a second swing near the neck on the otherwise straight line, `belly_top * 6.75 t (1-t)^2`
(peak at a third of the way down; positive swings out from the V like `belly`). Every build: `joined_lapel` with the
notch-30 collar, the same join gauge and objective as phase 2. **This change is experimental**: no tailoring text gives
the swing's size; it is the solver's.

| Round | belly_top | Gorge cv | Lapel peak / mean | Lapel 95% | Tip fold | Objective | Verdict |
|---|---|---|---|---|---|---|---|
| (joined) | 0 | 0.63 | 2.45 / 2.63 | 10.8% | 14.5° | 3.00 | |
| 1 | +0.06 | 0.61 | 2.63 / 2.59 | 12.8% | 17.7° | 3.21 | infeasible (tip) |
| 1 | +0.12 | 0.51 | 2.10 / 2.14 | 14.9% | 13.3° | 2.69 | gives back phase 1's strain (rejection 120) |
| 1 | -0.06 | 0.51 | 1.95 / 2.15 | 8.9% | 20.7° | 3.26 | infeasible (tip); collar retuned in round 2: notch at 0.65 cleared the tip but laid 6.5° off |
| 2 | -0.03 | 0.55 | 2.13 / 2.28 | 9.9% | 12.7° | 2.72 | |
| 2 | -0.10 | 0.47 | 1.70 / 1.83 | 8.4% | 15.2° | 2.79 | |
| 3 | -0.08 | 0.46 | 1.76 / 1.87 | 8.5% | 14.5° | 2.71 | |
| 3 | **-0.14** | **0.42** | **1.60 / 1.70** | 8.9% | 13.3° | **2.57** | chosen |
| 4 | -0.18 | 0.38 | 1.34 / 1.39 | 9.8% | 15.5° | 2.65 | worse: stopped |
| 4 | -0.22 | 0.37 | 1.29 / 1.34 | 10.9% | 22.0° | 3.10 | infeasible (tip) |

The gorge evens out steadily the further the top swings in; past -0.14 the lapel strain climbs back (8.9 to 10.9%) and the
collar tip pinches. Round 4 scored worse, so the rounds stopped there, per the iteration rules. A longer collar point
(0.85) tried at -0.08, -0.10 and -0.14 was slightly worse each time (2.65 to 2.82).

## Before and after (her right / her left)

| Measure | Approved | Joined (phase 2) | Roll top eased (phase 3) |
|---|---|---|---|
| lapel edge strain, mean | 4.3 / 4.7% | 3.1 / 3.6% | 3.2 / 3.7% |
| lapel edge strain, 95th pct | 14.3 / 15.0% | 9.5 / 10.8% | **7.8 / 8.9%** |
| worst visible edge (her right) | +39% | +26.0% | +26.6% |
| gorge load evenness (cv) | 0.51 / 0.54 | 0.62 / 0.63 | **0.43 / 0.42** |
| lapel side of the gorge: cv | 0.41 / 0.45 | 0.65 / 0.72 | 0.27 / 0.31 |
| lapel's worst stretch over its mean | 1.76 / 1.83 | 2.45 / 2.63 | 1.60 / 1.70 |
| lapel / collar load at the gorge | 2.85 / 0.52% | 2.63 / 0.58% | 3.90 / 0.56% |
| turn across the gorge, 90th pct | 52.9 / 53.8° | 38.6 / 39.0° | 40.1 / 41.5° |
| seam kink, 90th pct | 39.0 / 38.3° | 37.4 / 37.1° | 37.7 / 37.6° |
| collar tip fold | 13.6 / 13.6° | 11.3 / 14.5° | 13.3 / 12.4° |
| collar flat cut, 95th pct | 2.12% | 2.16% | 2.13% |
| notch as laid (30 asked; approved 40) | 40.9 / 41.0° | 30.2 / 30.1° | 29.8 / 30.8° |
| refmeasure lapel point reach / drop | | 0.561 / 0.418 | 0.545 / 0.413 (sides equal to 0.001) |

The gorge now carries its load more evenly than the approved build's, and the lapel's edge strain is the lowest of any
build. The lapel's strain along the gorge band is higher on average (3.9% against 2.6%) because it is no longer piled at
the roll end: spread along the seam instead of peaking there. The lapel point sits 3% closer to the centre (0.545 of
the half shoulder against 0.561), back to phase 1's 0.550. Her left lapel's hidden cusp (the under lapel's last fifth,
behind the upper front) reads +90% worst against +74% joined and +60% approved: it is pressed flat under the top front
and does not show in any view.

Checks: the recipe `eased_top_lapel` reproduces the swept lapel byte for byte; the contour sheet against the joined build
moves only the lapel's top edge, by about 0.1 unit (2.6 mm), the same on both sides; clay close-ups of both roll tops and both
3/4 neck views show no new marks (the grey along the collar's end rim is in the approved build too).

# Phase 4: the roll line and the collar solved together (2026-11-23)

John, on phase 3: "Can't solve independently the roll and collar the roll change add distortions to the collar you can
see" (rejection 121). The eased roll top was solved with the collar's settings held, and the collar creased: a short
dark crease on the fall where it meets the roll turn, at the side front of the neck (clearest on her right, about
(-2.86, -1.65, 51.75)). None of the phase 2 gauges saw it: the gorge numbers, the fall's dihedrals, its fairness (a
quadric fit within 0.3, under 0.25 mm everywhere in every build) and the stand's clearance were all unchanged.

## The coupling

notchcollar drafts the stand from the neckline so that it rises off each lapel's roll top at the slope that carries the
lapel's crease straight on round the neck. Easing the roll top toward the V turns the crease's top tangent, and the
stand's rise there flattened from 1.13 to 0.71 (printed by notchcollar now). With the stand lower at the side front,
the fall turns over a lower, tighter roll and creases where it meets the turn. The roll line and the collar's stand are
one line, so they have to be solved as one.

## The gauge

The crease is the angle between the fall and the roll turn across the edges they share, over the fronts (|x| > 2.0):
approved 36.5 (90th percentile), joined 36.1, eased with the collar held 41.1. It is added to the objective as a fourth
term over the approved value (so the approved build scores 4.0).

## Rounds (roll top -0.14 unless given; notch 30)

| Round | Varied | Fall/turn crease, fronts p90 | Seen in close-up | Verdict |
|---|---|---|---|---|
| 1 | roll top 0 / -0.08 / -0.14 × roll radius 0.08 / 0.12 / 0.16 × stand run 3.2 / 4.2 | 41.1 to 28.5 with the roll radius at -0.14 | | stand run changes nothing (the rise from the roll top sets the stand while `from_neck` is on) |
| 2 | roll top -0.10 / -0.14 / -0.18 × roll radius 0.16 / 0.20 / 0.24 | down to 23.4 at 0.24 | the crease still shows at 0.16 and 0.24 | a larger roll lowers the number without removing the cause; it also rounds the roll at the back |
| 3 | the stand's rise off the roll top: `rise_gain` 1.0 / 1.3 / 1.6 (new notchcollar option) | 36.3 at 1.6 with the approved roll radius 0.08 (joined 36.1) | gone on both sides | **chosen**: the rise slope back to 1.14 (joined 1.13) |

Round 3 is the fix of the cause, so the rounds stop there. The joint solution: `{"recipe": "eased_top_lapel"}` with
notchcollar `{"notch_angle": 30, "rise_gain": 1.6}`, everything else in the collar as approved. **Experimental:** the
1.6 is the solver's number, set to restore the straight top's rise; no tailoring text gives it.

## Before and after (her right / her left)

| Measure | Approved | Joined (2) | Eased, collar held (3, rejected) | Solved together (4) |
|---|---|---|---|---|
| lapel edge strain, 95th pct | 14.3 / 15.0% | 9.5 / 10.8% | 7.8 / 8.9% | 7.8 / 8.9% |
| gorge load evenness (cv) | 0.51 / 0.54 | 0.62 / 0.63 | 0.43 / 0.42 | 0.43 / 0.43 |
| lapel's worst gorge stretch over its mean | 1.76 / 1.83 | 2.45 / 2.63 | 1.60 / 1.70 | 1.60 / 1.70 |
| turn across the gorge, 90th pct | 52.9 / 53.8° | 38.6 / 39.0° | 40.1 / 41.5° | 39.7 / 41.2° |
| stand's rise off the roll top | | 1.13 | 0.71 | 1.14 |
| fall/turn crease, fronts p90 | 36.5° | 36.1° | 41.1° | 36.3° |
| collar tip fold | 13.6 / 13.6° | 11.3 / 14.5° | 13.3 / 12.4° | 13.3 / 12.4° |
| collar flat cut, 95th pct | 2.12% | 2.16% | 2.13% | 2.13% |
| notch as laid (30 asked) | 40.9 / 41.0° (40 asked) | 30.2 / 30.1° | 29.8 / 30.8° | 29.8 / 30.8° |
| refmeasure lapel point reach / drop, collar point reach | | 0.561 / 0.418, 0.526 | 0.545 / 0.413, 0.512 | 0.545 / 0.413, 0.512 (sides equal) |

## What the phase taught

- **Missed:** every geometric gauge on the collar. The crease is a change in where the fall meets the turn, a few faces
  along the roll; the fall/turn angle found it only once the render pointed at the spot. Phase 2's rejection 118
  ("the stand does not touch the gorge") was right about the gorge and wrong to imply the stand settings were idle:
  they shape the roll, which no phase 2 gauge measured.
- **Caught:** John's eye in the 3/4 neck views; then a ray cast from the close-up camera put the crease on the fall at
  the roll, and the stand's rise printed per build gave the cause.
- **Do differently:** solve any change at the lapel's roll top with the collar's stand free, and print the stand's rise
  for every build.
