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

# Phase 5: the collar's shape, not only its angle (2026-11-23)

John: "How do you solve for shape not just angle of the collar", then "Begin with some research to get a good shape
as a starting point then iterate".

## Research for the starting shape

| Source | What it gives | Used as |
|---|---|---|
| Mueller & Sohn, women's lapel collar | stand 2-2.5 cm at the front and shoulder, 2.5-3 cm at the back ("approx. 0.5 cm more than at the shoulder"); the outer edge set by a 0.5-1 cm overlap past the shoulder | `stand_shoulder` 0.82 (2.5/3.0 ratio) as the start |
| Devere, *Handbook of Practical Cutting* (1866) | stand 1¼" (3.2 cm), fall 3¾" (9.5 cm) | the fall depth range checked by `fall_side` |
| Minerva, collar drafting notes | a neck edge that curves like the neckline lies flatter | kept notchcollar's neckline-drafted stand |
| Fabrics-store, notched collar draft | the collar end is drafted straight | limit on `end_bow`: small bows only |

Three new notchcollar shape options, all default to the phase 4 collar:

- `stand_shoulder`: the stand at the side neck as a share of the back stand, eased in by azimuth.
- `fall_side`: the fall's outer edge deepened (or shallowed) at the sides by `fall_side * sin(pi u)^2`.
- `end_bow`: the straight end G to C bowed by this share of its length (negative: hollowed in toward the lapel).

## Rounds (objective cv/0.52 + turn/53.3 + tip/13.6 + crease/36.5; approved 4.0, phase 4 3.56)

| Round | Build | Objective | Notes |
|---|---|---|---|
| 1 | Mueller start, `stand_shoulder` 0.82 | 3.74 | fall/turn crease 36.3 -> 42.7°: worse (rejection 122) |
| 1 | `stand_shoulder` 0.7 / 0.95 | 3.94 / 3.62 | the crease grows steadily with the taper |
| 1 | + `fall_side` ±0.15, `rise` 1.0 / 1.6, `end_bow` ±0.1 | 3.71-3.80 | only `end_bow` -0.1 helped (cv 0.42 -> 0.41); +0.1 failed notch and sharp fold |
| 2 | `end_bow` -0.1 / -0.2, full stand | 3.53 / **3.50** | cv 0.43 -> 0.39, crease unchanged; `fall_side` and `rise` on top add nothing |
| 3 | `end_bow` -0.3 / -0.4 | 3.49 / 3.45 | -0.3 fails the notch (3.9°) with a sharp fold; -0.4 passes but the end cups in neck34 (rejection 123) |

Round 1 was worse and rounds 2 and 3 each gained under 2%, so the search stops. The chosen shape: notchcollar
`{"notch_angle": 30, "rise_gain": 1.6, "end_bow": -0.2}` on `eased_top_lapel`. **Experimental:** the -0.2 is the
solver's number; the research only says the end is drafted straight, and -0.2 is the largest bow that keeps a full
margin to the failing -0.3.

## Before and after (her right / her left)

| Measure | Phase 4 | Shape (end bow -0.2) |
|---|---|---|
| gorge load evenness (cv) | 0.43 / 0.43 | 0.40 / 0.39 |
| turn across the gorge, 90th pct | 39.7 / 41.2° | 39.8 / 41.3° |
| fall/turn crease, fronts p90 | 36.3° | 36.3° |
| collar tip fold | 13.3 / 12.4° | 13.3 / 12.4° |
| collar flat cut, 95th pct | 2.13% | 2.14% |
| notch as laid (30 asked) | 29.8 / 30.8° | 30.0 / 30.9° |
| refmeasure lapel point reach / drop, collar point reach | 0.545 / 0.413, 0.512 | 0.545 / 0.413, 0.513 / 0.512 |

Contour sections at z 44.3 to 50.9 and x ±1.4 / ±2.6 lie on phase 4's: the change stays at the collar end.

## What the phase taught

- **Missed:** the tailoring ratio. Mueller's stand taper assumes a 2.5-3 cm stand; on John's pressed-down 1.0 cm
  back stand, lowering the side further sharpens the roll turn. A published ratio needs checking against the stand it
  was written for.
- **Caught:** the fall/turn crease gauge from phase 4 flagged the taper at once; the close-ups caught the cupped end
  at -0.4 that the gauges passed.
- **Do differently:** a gauge pass next to a failing neighbour is no margin; pick inside the passing region.

# Phase 6: the high (classical) lapel (2026-11-23)

John: "Run the solver now with the harder problem of a high lapel", "The classic one": `classical_lapel`, cut to the
1890s riding habit's ratios (break 0.12 under the neck pit, roll line meeting the neck 3.0 out, small lapel), with the
phase 5 collar.

## Rounds

| Round | What | Best | Result |
|---|---|---|---|
| 0 | `classical_lapel` + phase 5 collar | objective 17.1 | the roll line runs nearly flat into the neck; the stand turns 94° off it and both collar ends knot (tip fold 175°) (rejection 124) |
| 1 | feasible map: `neck_x` 2.2 / 2.4 × break 48.25 / 49.0 × notch 15-30, notch_at 0.4 / 0.6, point 0.5 / 0.7 (64 builds) | 0 feasible | at 2.2-2.4 the stand follows the roll line (2-13°); notch_at 0.6 beats 0.4 everywhere (tip 20-30° against 27-40°); the classical lapel's own edge strain sits at its limit (14.4-16.4%, 95th pct) |
| 2 | phase 1's straight roll line (`belly` 0.001) on the classical lapel | lap95 10.9%, worst +22% | edge strain down a third; tip fold 18.0° (limit 15.6) |
| 3 | collar tip: point 0.35, tip_round 0.3, end_bow -0.2, rise_gain 1.3, notch 15, roll_r 0.16 | objective 4.38 (roll_r 0.16) | the tip fold reads 18.0° in every build: the lapel's tight roll at the top sets it; roll_r 0.16 lowers the fall/turn crease 60.7 -> 43.9° |
| 4 | the lapel's roll kept full at the top (`top_out`) | worse | lapel worst edge +57 to +137%, tip fold 152-169° (rejection 125) |

Stopped after round 4 got worse. The best build, `{"recipe":"classical_lapel","neck_x":2.4,"foot":48.25,"belly":0.001}`
with notchcollar `{"notch_angle":20,"notch_at":0.6,"point_len":0.5,"roll_r":0.16}`, fails the tip limit (18.0°) and
shows two defects in the close-ups: each collar front end stands up as a rolled knob at the side neck, and a crease
runs across the fall from the roll (rejection 126). Notch as laid 22.3 / 20.0° (20 asked); gorge 3.6 cm; fall 3.9 cm
at the points.

## What the phase taught

- **Caught:** the stand's turn off the crease (printed per build) found the cause of the knot at once; the
  insensitivity of the tip fold to every collar setting located its cause in the lapel.
- **Carried over:** phase 1's straight roll line works on any lapel: it cut the classical lapel's strain by a third.
- **Missed:** notchcollar assumes the gorge runs well below the roll top. On a high lapel the whole lapel is roll, the
  collar end sits on the turn, and the stand-from-the-neckline construction stands the end up at the side neck. The
  habit itself solves this with a separate standing collar. Fixing it needs a change of construction, not settings.

## Phase 6b: the collar's roll run into the lapel's (2026-11-23)

John chose "Collar roll runs into the lapel". Sections through the neck's axis round the roll top showed why the high
lapel creased: the collar's stand died to nothing at each roll top and its turn closed onto the fall (`end_taper`),
while the lapel's own roll there is an open loop about 0.25 wide. The collar stood off it with a gap, and its fall
rode over the lapel's roll ridge at the side neck.

New notchcollar options, off by default (the low-lapel build comes out byte for byte the same):

- `join_h`: the stand starts this high at each roll top instead of at nothing, and the turn is left open there
  (`end_taper` 0), so the collar's turn meets the lapel's roll as one roll at the gorge.
- `join_run`: over this length of roll line from each roll top the turn is a true half round, as the lapel's is.

| Round | What | Result |
|---|---|---|
| 1 | `join_h` 0.25, and 0.15 with run 1.5 | fall/turn crease 43.9 -> 24.8°, objective 4.38 -> 3.93-3.99; the gap at the roll top closes |
| 2 | `join_h` 0.08-0.2, run 1-6, notch 15 | lower is better: `join_h` 0.08, run 2, objective 3.89; the run moves the crease 26.0 -> 24.6° |
| 3 | lapel `roll_radius` 0.30 / 0.36, break 49.0 | tip fold 21-33°: worse (rejection 127); stopped |

Chosen: `{"recipe":"classical_lapel","neck_x":2.4,"foot":48.25,"belly":0.001}` with notchcollar
`{"notch_angle":20,"notch_at":0.6,"point_len":0.5,"roll_r":0.16,"join_h":0.08,"join_run":2}`. **Experimental:** every
number here is the solver's; the habit photo gives only the break and the lapel's ratios.

| Measure (her right / her left) | Low lapel (phase 5) | High, no join | High, roll joined |
|---|---|---|---|
| lapel edge strain, 95th pct | 7.8 / 8.9% | 8.7 / 10.9% | 8.7 / 10.9% |
| gorge load evenness (cv) | 0.40 / 0.39 | 0.59 | 0.59 / 0.62 |
| turn across the gorge, 90th pct | 39.8 / 41.3° | 38.7° | 38.7 / 37.8° |
| fall/turn crease, fronts p90 | 36.3° | 43.9° | 24.9° |
| collar tip fold (limit 15.6) | 13.3 / 12.4° | 18.0° | 17.2 / 18.0° |
| notch as laid | 30.0 / 30.9° (30 asked) | 22.3 / 20.0° (20 asked) | 22.4 / 20.2° (20 asked) |
| refmeasure lapel point reach / drop, collar point reach | 0.545 / 0.413, 0.513 | | 0.449 / 0.214, 0.449 (sides within 0.003) |

Still wrong in the close-ups (rejection 128): a small open curl where the collar's turn meets the lapel's roll, and
the collar's top standing round at the side neck. The tip fold is over its limit. The collar's turn is built as a
vertical half round off the stand; the lapel's roll at its top runs at the crease's angle, so the two loops meet
twisted. Turning the collar's turn cross-section at the roll top into the lapel's is the next change.

- **Caught:** sections through the neck's axis at fixed azimuths showed the gap and the ridge the renders only hinted at.
- **Missed:** the tip fold gauge sits at the top of the gorge seam, which on a high lapel is the roll junction: it
  reads the junction's twist, so collar point settings could never move it.

### The lapel freed as well

John: "Change the lapels as well if it's creating too much tension". The lapel's top (within 2.6 cm of the roll top)
carries about twice the strain of the rest (95th pct 14% against 7%), but it is under the 15% limit everywhere. Nine
lapel variants under the joined collar (top swing `belly_top` -0.14 / +0.12, top edge start 0.06, width 0.40, point
0.12, `neck_x` 2.2, press 0.5 / 0.3, press 0.5 with `roll_radius` 0.3) left the top strain at 12.9-16.2% and raised the
collar's junction fold to 17.2-38.8° (rejection 129). Two rounds without gain: stopped. The lapel is not what twists
the junction; the collar's turn is.

# Phase 7: the high lapel fitted to the reference photo (2026-11-23)

John: "Back to reference you need to use the shape of the collar as reference", "look at that join and angle", "Trace
the polygon lines of the lapel". The 1890s riding habit, her left side (the side facing the camera), traced on a
gamma-lifted 7x close-up with a 10 px grid. Unit: the lapel's lower edge span, break L0 to lapel point P, across.

| Point | Photo (px) | Normalized (across, up) |
|---|---|---|
| L0 visible break | (590, 302) | (0, 0) |
| P lapel point | (693, 292) | (1.00, 0.10) |
| G notch | (675, 263) | (0.83, 0.38) |
| C collar point | (705, 272) | (1.12, 0.29) |
| N roll top | (667, 250) | (0.75, 0.50) |

What the trace says about the join. The lapel's lower edge is almost level (5.5°). The gorge is short (0.15) and steep,
so the notch sits just below the roll top. In the fold's own measure, the notch and the lapel point both lie on one line
square off the roll line's top, the notch 0.31 of the way out (`notch_at` 0.3). The collar's end is as long as the
lapel's end (0.30 against 0.33, `point_len` 0.9). It runs out and slightly down (17° below level), so the collar's point
hangs past the lapel's point (0.12 further out). The notch opens 41° in the photo.

`blender/tension/fitref.py` measures a build the same way (az 30 ortho, her left; G and C as notchcollar cut them, now
printed in collar.log) and sums the squared gaps of the five points.

| Build | Polygon error | Lower edge | Roll rise | Notch | Collar end / direction | Gauges |
|---|---|---|---|---|---|---|
| Photo | 0 | 5.5° | 0.50 | 41° | 0.30 / -17° | |
| Phase 6b high lapel (notch 20 at 0.6, point 0.5) | 0.20 | 17.8° | 0.73 | no collar point past the notch | 0.00 | tip 18.0°, crease 24.9° |
| Collar fitted on that lapel (notch 40 at 0.3, point 0.9) | 0.12-0.13 | 13.4° | 0.64 | 42° | 0.30 / -22° | |
| **Fit A**: lapel break 49.0, neck_x 2.4, + fitted collar | **0.087** | 9.9° | 0.58 | 46° | 0.23 / -27° | tip 35°, notch laid 33 / 24 (40 asked), crease 33.5° |
| Fit B: wider lapel (0.55), neck_x 2.7, break 48.6 | 0.066 | 10.0° | 0.61 | 41° | | tip 94°, 3 sharp folds: crumpled |
| neck_x 2.7, break 49.3 | 0.091 | 5.0° | 0.52 | 48° | | tip 113°, 14 sharp folds |

The shape can be matched: Fit A follows the photo's polygon to within 0.09. The cloth cannot follow it yet. The closer
the roll line comes to the photo's (flatter, meeting the neck further out), the harder the collar's ends twist where
its turn meets the lapel's roll (tip fold 35° at Fit A, 94-113° beyond it). This is the same junction as rejection 128:
the collar's turn is built as a vertical half round, while the lapel's roll at its top runs at the crease's angle.
**Experimental:** Fit A is not approved (rejection 130).

## Phase 8: a four-sided lapel, and a polygon check that counts sides

John, 2026-11-23: "Your lapel is a triangle there's has 4 sides", "Properly get the number of sides right", "Read up on
trapezoidal lapels", "Get collar lapel right as well it's a combination of a rectangle and a triangle", "Add a proper
polygon check with side count".

**What went wrong in phase 7.** The fit compared five chosen points. The model's lapel had three sides (the roll line,
the end edge, the lower edge), because foldfront's outline ran straight from the roll top to the lapel's point. The
photo's lapel has four: the roll line, a level top edge along the neckline, the end edge, the lower edge. A triangle
scored 0.087 against it because nothing counted sides.

**Sources.** Gentleman's Gazette (the gorge is "the meeting point of collar and lapel"); the fabrics-store notched
collar draft (lapel about 3 in wide); The Delineator, Aug 1890, riding basque No. 3409 ("fashioned to roll in small
lapels, which meet a rolling collar in notches").

**foldfront `outline_inner`.** A list [[t, d], ..., [0, 0]]: the lapel's top edge from its corner back to the roll top.
It lies at t < 0 (past the roll top, along the neckline), which the width-over-fold outline could not reach. The cut
gets a lower boundary there. Default lapels are unchanged: the classical lapel built byte for byte the same.
notchcollar reads it: the gorge walks the top edge to the corner and then the end edge, and the notch defaults to the
corner.

**`polycheck.py`.** The check finds the corners on its own.
- Lapel: the corners of the cut outline.
- Collar: the corners of the collar patch visible in a depth buffer from the photo's camera.
- Corner detection: Douglas-Peucker, then turning of at least 25 degrees, then rounded runs merged.
- Fail rule: the side count must match the photo (lapel 4, collar front 4). After that it lists every edge off in
  direction, length share or bow, and every corner off in angle or rounding radius.
- On the triangle lapel of phase 7 (b/HF1) it fails: lapel 3 sides of 4.

**Pentagon or curve?** Measured on the photo, the collar front has four corners:

| Feature | Measurement |
|---|---|
| Corner T (top) | radius 0.04 lapel spans |
| Corner K (shoulder) | radius 0.12 |
| Front edge | nearly straight (bows 6%) |
| Outer edge T to K | one S-curve (+4% / -7%), with no corner along it |

The model's fifth side was a 27° bend in the collar's bottom edge, where its point hung below the lapel's top.
**Cut against view (`flatview.py`).**
- The collar's flat cut is a curved band.
- From the side, two of its front corners (the notch and the rounded point) are cut corners.
- T and K come from the bend: T is where the outline stops being the cut edge and becomes the roll's silhouette (the
  front edge up the neck is 72% fold), and K is where the outer edge turns away round the neck.

**The inverse (`lapel_inverse.py`).** Steps:
1. Fit the map from the flat cut (t, d) to the view on the solved lapel. It is close to affine, residual 0.04.
2. Put the photo's corners through its inverse.
3. Rebuild. The polygon error went from 0.07 to 0.001.

The angles came out as:

| Edge | Photo | Model |
|---|---|---|
| Roll | 53° | 51° |
| Top | 7° | 9° |
| End | 132° | 128° |
| Lower | 6° | 5° |

The size is set by where the lapel point lies between the front edge and the body's outline: 0.49, against 0.46 to 0.53
on the photo. The first wide fit (12.3 cm out) matched the shape at 0.67 and was too wide.

Lapel TZ41: `{"recipe":"classical_lapel","belly":0.001,"foot":49.0,"neck_x":0.4,"outline":[[-1.0,1.0],[-0.8,2.0],[1.0,0.0]],"outline_inner":[[-1.0,1.0],[0.0,0.0]]}`.

**The collar is not fitted yet** (experimental, not approved). On TZ41 the collar front reads 3 sides of 4. Its fall
covers the whole top edge from the roll top, so the patch's corner sits at the roll top instead of the notch. The tip
fold at the gorge's top is 149-178°, a hairpin. Changing stand, fall_over, point_len and notch_angle (four builds) did
not change the side count. Open question to John: is the collar's front edge against the scarf the collar's own rolled
edge, or the scarf lying over the collar?

**The corners rounded** (John, 2026-11-24: "Round the inside edge of the trapezoid so it distributes load on two
sides", then circling the lapel's point and the collar's shoulder: "Here as well").

New foldfront option `outline_fillet` {"N": r, "G": r, "P": r}:
- Each named corner becomes a circular arc tangent to both of its edges, with r in fold lengths.
- N is where the top edge leaves the roll line. Rounding it starts the top edge 0.124 down the roll line, on a curve.

TZ48 uses N 0.3, G 0.08, P 0.09 on the TZ41 lapel, with the same collar (notch 40, point 0.9):

| Gauge | Sharp (TZ41) | Rounded (TZ48) |
|---|---|---|
| Collar's tip fold, her left | 149° | 22° |
| Collar's tip fold, her right | 14° | 13° |

The hairpin at the gorge's top is gone. The collar lies on the lapel's top edge without the crimp, and the fall rises to
the back. The polycheck now counts 7 sides on the collar front: the patch stands up from the notch, narrower than the
photo's, and does not reach out to K. The default lapel is unchanged (classical_lapel byte for byte).

Two more changes went in:
- The top edge is faired along itself after the solve (`top_fair`), which took out a 1.3 mm average ripple.
- The cut is snapped along the roll line carried on past its top.

**Still open.** A row of spikes stays along the lapel's top edge near the roll top. The strain map shows edges there
stretched up to 3.4 times, at t -0.4 to -0.8. They were already there before the rounding.

## Phase 9: the collar as a folded rectangle

John, 2026-11-24: "Begin with a rectangle and deform", "Rounded corners on the rectangle as well", "draw only the collar
and get that correct", "Research how this type of collar is done with the sharp triangular bend".

**What the collar-only render showed.** Both earlier collars (notchcollar's drafted fall, and its `rect` band) are a
rolled tube round the neck with ends added on. Neither has a crease, so neither can make the photo's triangular peak.

**Sources.**
- A one-piece collar is drafted as a rectangle 2.5 to 4 in wide and the neckline long, its neck edge straight or
  slightly convex (dresspatternmaking.com, basic collar).
- The roll line lies about 1 in up from the neck edge (thecreativecurator.com/collars).
- The back collar is 2 3/4 in (blog.fabrics-store.com/?p=13206).
- The crease is set by pad stitching, a bias undercollar and a drawn-up tape (minervapatterns.com).

No source explains the triangle. My reading of the geometry: the roll line runs parallel to the neck edge at the back
and must come down to the neck edge's end at the front, so it crosses the band's end at an angle. A rectangle folded on
a crossing line shows its turned part as a triangle.

**`blender/collar/foldcollar.py`.** Steps:
1. The band is meshed flat (rounded rectangle, the roll line an edge of the mesh).
2. The stand is laid up the neck from the neckline.
3. The fall is mirrored across the roll line on the flat (every length kept) and opened off the stand.
4. The fall is pushed off the coat along the band's own normal by an evened amount.
5. The cloth is relaxed with bending.

Two failures on the way:
- A relaxation with no bending crumpled the ends.
- A push along the coat's nearest normals crumpled them again over the lapels' layers.

`foldcollar_blend.py` puts the result in the scene.

**The fit (`blender/tension/foldcollar_fit.py`).** A coordinate search on the outline gap to the photo's collar-front
polygon, from the photo's camera with the scarf's cover:

| Collar | Outline gap (lapel spans) |
|---|---|
| Drafted collar | 0.078 |
| `rect` band | 0.072 |
| Folded rectangle, first try (width 2.2, stand 0.9, fold 45°, open 25°) | 0.112 |
| **Folded rectangle, fitted** | **0.038** |

The fitted collar: width 1.9 (5.0 cm), stand 0.7 (1.8 cm), fold angle 21°, open 5°, corner 0.2, and the seam started 16
steps (0.3 units) further front than the lapel's corner. It reads 4 sides of 4. Its ends rest on the lapels.

**Experimental, not approved.** Open problems:
- The laid cloth is stretched 34% at the 95th percentile, where the roll line turns from parallel to diagonal. A real
  collar takes that up with a curved neck edge.
- `open` ended at its lower limit (5°) and the fold angle near its limit (20°).

## Phase 10: the collar's cut shaped to the neck (2026-11-24)

John: "Adjust the shape more for the neck deformation".

**Where the 34% came from** (measured by region, `strain` per edge against its flat length):

| Cause | Stretch it left (95th percentile) | Fix | After |
|---|---|---|---|
| The roll line turned from parallel to diagonal in a short run, so the mirrored fall folded onto itself there (a wedge of twice the fold angle, edges squeezed up to 97%) | 38% in the fall beside the turn | `fold_k` 3: the same end angle, the stand reached gradually | 15% |
| The push off the coat rose steeply over the lapels | 50 to 60% at the ends | `push_even` 600, `push_grow` 20 | 25% |
| The cloth's thickness at the crease taken in one edge | 70% mean on the crease's edges | `lift_run` 0.3 | 10% |
| A straight band on a neck that flares to the shoulders | 13 to 17% everywhere, the stand included | `shape_cut` (below) | 5.6% |

**`shape_cut` (experimental).** The flat pattern is solved from the lay: the flat points move until every edge on the
flat is as long as it lies on the neck (sparse least squares from the rectangle, `foldcollar.shape_cut`). The cut that
comes out is the pattern books' curved collar: the neck edge bows 4.4 cm over its 37 cm, high at the centre back and
dropping to the ends (source: dresspatternmaking.com, the straighter the neck edge the higher the collar stands, the
more curved the flatter it lies). Stretch after: mean 1.7%, 95th percentile 5.6%. No triangle of the mesh flips.

**Rejected on the way:** a harder relaxation (displacement smoothing with resets, 600 to 1500 rounds, the stand freed)
brought the stretch to 13 to 17% but crumpled the ends and rippled the stand's top. The lay is right; the rectangle
was the wrong cut for it.

**Refit** (`foldcollar_fit.py`, now with `fold_k` and `--fixed`/`--start`): width 2.05, corner 0.5, the rest as phase
9. Outline gap 0.036 (phase 9: 0.038), 4 sides of 4.

**Open:**
- The K corner's radius gauge on the pixel outline is unreliable (it read 0.006 to 0.10 for the same corner as
  `corner` moved 0.2 to 0.8); the corner is set by eye at 0.5 and not yet measured against the photo's 0.115.
- `open` is still at its lower limit of 5 degrees.
- The folded collar is not yet in the main build, and `join.py` has not been run on it.

## Phase 11: the collar was sunk in the coat, and too thin (2026-11-24)

John: "Why is the close now imperceptible nearly perfect with the bodice", "Adjust that triangle part to the neck bone
it's on", "Why does every adjustment we make reduce creasing and makes the collar thinner".

**The finding.** The close fit was a fault. The coat's cloth stands 0.167 (4.4 mm) outside its mesh; the collar's fall
was held 0.14 off that mesh, so 84% of the fall over the coat lay inside the coat's cloth. And `foldcollar_blend` gave
the collar a thickness of 0.06 (1.6 mm). My first answer to John (the shared seam, the solved cut) was wrong about the
cause; so was my claim that the roll line's crease had been softened (the new gauge reads its sharpness 0.73 fitted,
0.96 shaped).

**Why each round went that way.** Stretch, the join's kink and turn, and a one-camera outline gap all reward a smoother,
flatter, closer collar. Nothing measured body, crease or sinking. `blender/tension/foldcollar_gauge.py` now does:

| Collar | Crease turn, back | Crease sharpness | Fall off the stand (cloth thicknesses) | Sunk in the coat | Cloth drawn |
|---|---|---|---|---|---|
| First folded rectangle | 152 | 0.84 | 2.6 | 62% | 0.36 |
| Fitted (phase 9) | 170 | 0.73 | 1.3 | 91% | 0.36 |
| Shaped (phase 10) | 157 | 0.96 | 1.7 | 84% | 0.36 |
| **Lifted, full cloth (this phase)** | 144 | 0.93 | 2.6 | 20% | 1.0 |

**Changes (experimental).**
- `foldcollar_blend.py`: the collar's Solidify takes the coat's thickness, centred.
- Build options `lift` 0.19 (a thickness between stand and fall), `clear` 0.27 (1.5 thicknesses and a hair off the
  coat's mesh), `push_rounds` 3. More rounds overshoot (2.2 thicknesses) and stretch the rectangle 60 to 90%.
- `bone` (0 to 1): each end raised or lowered so its lower edge lies on the collarbone's line (the gown's
  `collarbones`: z 50.2 rising 0.2 from |x| 0.75 to 4.7). The edge fell away outward at slope -0.42 (23 degrees); with
  `bone` 1 it is -0.10 (6 degrees). The skin of this body has no collarbone ridge to measure against.

**Worse, and stopped on it (stop rule):** the outline gap to the photo went from 0.036 to 0.055 and the outline reads 5
sides. The lifted, thicker end is rounded where the photo's is a flat triangle, and the end's edge is ragged beside the
seam in the collar-alone view. The remaining 20% sunk is in the first 5 cm of each end.

**Rejected, same day: dragging the laid end onto the lapel (`gorge`, removed).** John: "How should the join with lapel be
done to flatten this". Measured first: the collar does not lie on the lapel (1% of its fall is over the turned lapel);
its end edge runs beside the lapel's top edge for about 2 cm, 0.28 to 0.31 units (8 mm) off it, then on past the lapel's
corner. With the cloth lifted, `join.py` finds no seam at all. I moved the end-edge points onto the lapel's top edge and
faded the shift into the cloth over 1.5 units. Result: the inner end was pulled down into a pleat (tip fold 90 degrees,
seam 0.4 to 0.5 cm, gap 5.2 mm, outline gap 0.056). Worse, so stopped. The seam has to be right in the build: the band's
end laid along the lapel's top edge from the start (the seam taken from that edge, not from the neckline 0.3 off it),
not a laid collar dragged there afterwards. Also fixed: the collarbone shift is taken up over 0.6 past the roll line
(`bone_run`), which removes the teeth it tore along the neck edge.

**Rejected, same day: the seam moved in the build (not committed).** Two more tries at the join, both worse than the
lifted collar (outline gap 0.055, no seam found by `join.py`):

| Try | Outline gap | Join |
|---|---|---|
| Seam started at each lapel's roll top (`seam_at` roll_top) | 0.081 | no seam found |
| The same, the seam snapped onto the lapel's top edge up to its corner (`seam_on_lapel`) | 0.068, 5 sides | seam 1.0 cm, gap 5 mm, kink 23 to 27 degrees, tip fold 51 and 119 degrees |

What the measuring did settle: the neckline starts at the lapel's roll top and runs parallel to the lapel's top edge
(direction cosine 0.99) for that edge's whole 2.45 units, 0.3 to 0.4 off it. So collar and lapel are two cloths lying
side by side along 6 cm, and any collar hung from the neckline folds its fall down over that gap. Stopped after two
worse rounds (stop rule). The open choice is which cloth moves: the collar's seam (tried three ways, each worse) or the
lapel's top edge (not tried).

## Phase 12: the lapel's top edge recut onto the neckline (2026-11-24)

John chose "Recut the lapel's top edge" (decision card) after three tries at moving the collar failed.

**Measured first.** The 0.3 to 0.4 between the lapel's top edge and the neckline is two things: 0.2 to 0.35 off the
coat's surface (the turned lapel lies a cloth above the front, which is right), and an in-plane drop below the neckline
that grows from 0 at the roll top to 0.38 at the corner G. Only the second is a fault of the cut.

**The recut (experimental).** The corner G moved up the fold, t -1.0 to -1.3, and the point P with it (-0.8 to -1.1) so
the end edge keeps its direction:
`{"recipe":"classical_lapel","belly":0.001,"foot":49.0,"neck_x":0.4,"outline":[[-1.3,1.0],[-1.1,2.0],[1.0,0.0]],"outline_inner":[[-1.3,1.0],[0.0,0.0]],"outline_fillet":{"N":0.3,"G":0.08,"P":0.09}}`

| | TZ48 (before) | G only to -1.3 | **G and P (TZ61)** |
|---|---|---|---|
| Top edge below the neckline, mean (her left / right) | 0.16 / 0.12 | 0.02 / -0.02 | **0.04 / -0.01** |
| At the corner | 0.38 / 0.36 | 0.23 / 0.18 | 0.21 / 0.18 (the corner's fillet) |
| polycheck, lapel | 4 of 4, no mismatch | end edge 21 degrees off, corners 24 off | **4 of 4, no mismatch** (top edge 13 degrees, photo 7) |

**The collar on it** (`foldcollar_build.sh`, with `gorge_flush` 1.6 and `gorge_level` 0.12: within 1.6 of the lapel's top
edge the collar is let down from its clearance toward the lapel's own level):

| | Lifted collar on TZ48 | On the recut lapel | **With `gorge_flush`** |
|---|---|---|---|
| Outline gap to the photo | 0.055, 5 sides | 0.028, 4 sides | **0.027, 4 sides** |
| Seam found by `join.py` | none | 1.2 / 0.4 cm | **1.7 / 1.5 cm** |
| Step at the seam | - | 5.8 / 7.5 mm | **2.9 / 4.8 mm** (cloth 4.4) |
| Turn across the seam, 90th pct | - | 20 / 18 degrees | 25 / 25 |
| Tip fold | - | 9 / 8 degrees | 11 / 9 |

**Rejected:** a last pass lifting collar cloth left under the coat's surface (`floor`): it lifted the collar off the seam
again (no seam found) and the outline gap went to 0.040.

**Open, and now the thing in the way:** the lapel's top edge is ragged (the stretched edges at t -0.4 to -0.8 found in
phase 8), and one of its spikes comes through the collar's face as a fin in the 3/4 close-ups. It is there with and
without `gorge_flush`. The join cannot read clean until that edge does.

**The ragged top edge, found and fixed (`top_ground`, foldfront, on for `outline_inner` lapels).** Open since phase 8.
Traced by dumping the ground each lapel point is laid on: above the coat's neckline the ray that reads the ground
misses the front and hits the neck or the shoulder 0.7 behind it, and it does so row by row (one row finds the front,
the next falls past its edge). 184 and 219 points per side had fallen. The cloth took the steps as stretched edges.
Now, above the roll top, no point's ground may lie further back than its neighbours' by more than 0.06 an edge, so the
ground is carried on from where the front still is.

| Lapel (recut) | Edges stretched over 2x (her right / left) | Worst edge | 95th percentile |
|---|---|---|---|
| Before | 35 / 24 | 3.7x / 3.8x | 17% / 21% |
| **`top_ground`** | **0 / 0** | 1.55x / 1.87x | 10% / 14% |

On it: polycheck passes (lapel 4 of 4, no mismatch), the collar's outline gap is 0.027 with 4 sides. Join: seam 1.5 /
0.8 cm, step 3.6 / 6.9 mm, tip fold 7 / 8 degrees. In the close-ups the spike through the collar is gone but for a small
bump, and the top edge is smoother, not yet clean: a few teeth remain near the roll top, and her left side's step at the
seam is still above a cloth's thickness.

**The top edge's own points (foldfront, `outline_round` with `outline_inner`).** The teeth that were left were not the
solve: dumping the flat cut showed no point on the top edge at all. The `outline_round` branch (which `classical_lapel`
uses) ran its rows from the roll line carried on and gave the top edge no points, so the edge was what the wedge filter
left of the rows, a staircase 0.04 either side of the line, turning 80 to 110 degrees at every step. The branch now sets
points evenly along the top edge and starts the rows 0.6 of a mesh step inside it, as it does for the outline. In the
close-ups the top edge is a clean line on both sides.

**The join's asymmetry (John: "left side is almost perfect while right sticks out"), measured, not solved.** Picture
right is her left. `join.py`: her right seam 1.7 cm, step 4.1 mm; her left 0.6 cm, 6.5 mm. Her left front lies over her
right, and the collar's left end sits 0.17 (4.5 mm) further forward than its right and reaches 0.13 over its lapel's
edge against 0.07. The collar is laid as a mirror image on fronts that are not one. Rejected: bringing each end both
ways to its clearance (`settle`, removed): `join.py` lost the seam on both sides and the spread of heights did not
fall, because the height read off the nearest mesh point is not reliable beside a cut edge. John's direction: the next
step is to design the bodice with the lapel, so the front carries the lapel's load and makes the flap; the asymmetry
belongs to that.

## Phase 13: the bodice cut with the fold in mind (both neck corners level)

John, 2026-11-24: "Join did not account for asymmetry left side is almost perfect while right sticks out next step is
co design the bodice so it can take load from the lapel".

**Measured first** (`~/work/FC/asym.py`, mirror test of each piece against the other side):

| Piece | Her right forward of her left |
|---|---|
| Roll line, whole length | 4.5 mm (one cloth, 4.4 mm) |
| Lapel top edge | 1.3 mm on average, 3.7 mm by the roll top |
| Coat past 2.5 units from the middle | 0.0 mm |

The fronts cross only below z 49.82, where the two folds meet. Above it they stand apart in the V, yet the cut kept the
right front a cloth forward all the way to the neck. One collar, laid as a mirror image, cannot sit level on two neck
corners 4.5 mm apart. The load the lapel puts on the front is not the cause: foldfront's log gives a pull on the
button pins of 0.001 and the front moving 0.02 to 0.05 mm.

**Change** (source: the fold's own numbers in foldfront's log; experimental): `BODICE_OVER_TOP=49.82,50.36` in
`bodice_cut.py` fades the right front's lift out between the folds' crossing and the roll lines' top.
`blender/bodice_cut/over_top.py CUT.blend OUT.blend 49.82,50.36` applies the same change to a saved cut (351 points,
4.4 mm at most).

| Gauge | Before, her right / her left | After |
|---|---|---|
| Roll top, depth | −3.42 / −3.26 | −3.26 / −3.26 |
| Lapel top edge, right forward of left | 1.3 mm | 0.2 mm |
| Join: seam length | 1.71 / 0.62 cm | 1.45 / 1.04 cm |
| Join: step | 4.1 / 6.5 mm | 5.7 / 5.7 mm |
| Join: turn | 29.6 / 26.4° | 27.8 / 28.0° |

Both ends now make the same join. The right's lower step before came from its lapel standing a cloth forward, so the
level cut brings it to the left's value; the step itself is the next thing to take out, on both sides at once.

**Tried on the collar and rejected:**

- `flush_rounds` 3 and 6 (the let-down repeated): the two ends kept their 2 mm difference and the turn rose from 30°
  to 39°. Removed. The difference was in the fronts, not in the collar.
- `gorge_level` 0.0 on the level fronts: step 4.6 / 4.3 mm, but a thin spike appears on the collar's face beside
  each lapel in the clay close-ups.
- `gorge_level` 0.0 with `gorge_flush` 2.2: step 3.2 / 2.5 mm, seams 1.76 / 1.78 cm, but a folded wedge and a tab at
  the shoulder on the collar's face. Kept at 0.12 and 1.6.

**Not reproducible:** the shared `cut.blend` holds the bodice 0.6 units further off the chest in the V than a fresh
build from `origin/claude/pr-cleanup/redingote-riding-habit` (front depth −3.73 against −3.13, same skin). On the
fresh build the same change gives seams 1.36 / 1.37 cm and steps 3.3 / 3.1 mm. The study above uses the shared cut
with `over_top.py`, so it compares with every earlier phase.

**Load on the fronts after the recut** (foldfront's log, the front round the foot let go and the buttons pinned):

| | Her right front | Her left front |
|---|---|---|
| Pull on the button pins | 0.002 (was 0.001) | 0.001 |
| Strain passed into the front, mean / worst | 0.20% / 9.1% (worst was 25.6%) | 0.11% / 8.5% |
| Lapel strain, mean / 95% | 3.4% / 10.1% (was 3.9% / 11.7%) | 4.4% / 14.6% |
| Lapel strain at the foot of the fold, mean | 7.1% | 18.4% |

The fronts carry almost nothing: the lapel is short and lies on the front. The one-sided number left is the under
(left) lapel's foot, where 106 of its points are pressed behind the upper front.

**Tried and rejected: room for the under lapel's foot.** The right front raised a further 1 and then 2 cloths over the
triangle between the two roll lines below their crossing (an `UNDER` argument of `over_top.py`, removed). One cloth:
left lapel 95% strain 14.6% to 11.9%, foot mean 18.4% to 18.1%. Two cloths: back to 14.6% and 18.4%, and the right
lapel worse (95% 10.1% to 12.2%). The crush is set by foldfront's press rule for the under lapel, not by the room the
bodice leaves. Stopped after two rounds.

## Phase 14: the join's flaws (John, 2026-11-24: "Fix the imperfections of the join then render in color")

Found on tight clay close-ups of both ends, each traced by hiding one object at a time and by sections through the
join (`~/work/FC/section.py`: vertical cuts at fixed x through the coat, the lapel, its top edge and the collar).

| Flaw | Cause | Fix | Result |
|---|---|---|---|
| A small fin on each collar end's face, \|x\| 3.3, z 51.5 | The coat's neck edge stood 1 mm through the collar. The push off the coat measures from the collar's side, against the nearest coat point's normal; beside the coat's neck edge that normal tilts, and a ridge between two collar points is never seen | `no_poke` 0.02: each point of the coat's outer face measured against the collar, and the collar lifted where the coat reaches past its middle; spread over the cloth, held still within 0.4 to 1.0 of the lapels | 13 points through the face to 0; gone in the close-ups; join gauges unchanged |
| A dark slit where each collar end meets the neck | The roll line comes down to the neck edge, so the stand runs out in a wedge thinner than the cloth; drawn a cloth thick, the wedge and the fall crossed | `end_trim` 0.19: the stand's faces under the roll line where it is narrower than 0.19 are cut away | Mostly gone in context; join gauges unchanged |
| A dark wedge across each collar end at the lapel's top edge | The collar's end is laid on the coat, but the turned lapel stands two cloths off it: the collar lies 6 mm behind the lapel's face and runs 3 to 9 mm on under its top edge | Not fixed | See below |

**Tried and removed (rejection #138):**

- `flush_rounds`, `lift_min` 0.17 (the folded layers kept a cloth apart at the crease): scallops along the roll line
  and a larger slit. `end_close` (the stand's wedge laid up under the fall): tore the end into spikes. `end_trim` by
  counting stand points per face: a saw edge along the roll line (the centroid rule replaced it).
- `gorge_match` in two forms (the collar's end raised to the lapel's face at the top edge; then the hidden strip
  taken in to the edge first): both tore the collar's end, a cliff where the lapel's edge ends and teeth by the neck.
  Stopped per the stop rule.

**What the third flaw needs:** the collar's end and the lapel at the gorge have to be built as one surface, the
collar's end edge laid along the lapel's top edge at the lapel's own face, rather than a collar laid on the coat and
moved afterwards. Any after-the-fact move of part of the end tears it.

## Phase 15: back to the low notch lapel (John, 2026-11-24: "Explore the lower lapel and what do correct collar needs to be with the recent knowledge acquired"; "I mean go back to the previous lapel shape")

**The fronts.** `eased_top_lapel` folded on the shared cut: its roll tops already match side to side (they sit at
|x| 2.18, past the lap), but the right lapel stood 1.8 mm forward of the left in its middle, as the right front was a
cloth forward above the folds' crossing (z 45.53). `over_top.py CUT 45.53,46.5`: top third 0.34 to 0.03 mm, middle 1.83
to 0.23 mm; the foot third keeps the lap's 5 mm (the fronts do cross there). Lapel strain 95%: 7.6 / 8.9%.

**The drafted notch collar** (`notchcollar` `{"notch_angle":30,"rise_gain":1.6,"end_bow":-0.2}`, the phase 5 pair) on
the level fronts: gorge seam 7.8 cm a side, step 4.0 mm, turn across the gorge 41°, a 55° kink, tip fold 12°, the same
both sides. It already carries the coat's cloth thickness (Solidify 0.1669) and sits 0.23 off the coat (13% within the
coat's cloth, against 84% for the first folded collar).

**The folded collar does not carry over (rejection #139).** Carried on down each gorge to the notch at the lapel's face
(`gorge_seam`, removed), the band crumples at the roll top: the seam turns a corner there from the neckline into the
gorge, and the fall lies on the inside of that corner and laps over itself (relaxed strain 73%, tip fold 140°). Round
the back it laid cleanly.

**What the low lapel's collar needs**, from phases 9 to 14:

1. The drafted notch collar's shape: it is cut for the gorge and the notch, which the folded band cannot turn.
2. The fronts level above the crossing (done here).
3. The coat's cloth thickness and a clearance measured from the coat's side, so nothing shows through (phase 11 and
   `no_poke`): it has the first; the second is to be checked on it.
4. Its gorge edge on the lapel's top edge at the lapel's own face, laid as one surface with it: the 55° kink and the
   4 mm step are where it is not.

## Phase 16: the drafted notch collar laid flush on the low lapel's gorge (John, 2026-11-24: "Try it")

**Where it stood.** A new gauge on the solidified meshes (`flush`: rays straight at the lapel's face, 1.6 mm inside
the lapel's top edge and 1.6 mm past it) showed the drafted collar lying *on* the lapel, not sewn to it: along the
gorge its face stood 7 to 9 mm proud of the lapel's, and in the first centimetre at the roll top the collar did not
reach the edge at all (the coat showed through the corner). The cause is in the draft: each gorge point sat the cloth's
thickness plus the gap over the lapel's top edge, along the one bodice normal at the notch, and the collar's own
thickness then went outward on top of that. The lapel's face is its mesh: its Solidify goes in behind it.

**The change (EXPERIMENTAL, `notchcollar` `"gorge_flush": 0.6`, off by default).** The gorge sewn as a tailor sews it,
edge to edge with the seam pressed open, so the collar's face runs on from the lapel's: each gorge point goes onto the
lapel's top edge, a cloth's thickness behind along the lapel's own normal there (`flush_lift`, so the collar's face is at
the lapel's), the move is carried into the cloth as a smooth field (held at the gorge, nothing past 0.6, evened between),
and within `flush_plane` 0.35 of the seam the cloth is laid into the lapel's plane so it leaves the seam at the lapel's
slope.

| her left (her right the same within 0.2) | drafted collar | gorge flush |
|---|---|---|
| collar face against the lapel's, middle of the gorge | 7 to 9 mm proud | within 0.3 mm |
| first cm at the roll top | collar short of the edge | covered, 0 to 3 mm |
| last half cm at the notch | 8 mm proud | 2.7 mm proud |
| tip fold | 12° | 20° |
| creases over 40° in the collar | 0 | 3 (53° at most, near the roll top) |
| coat showing through the collar (colour-coded render) | none | none |

`join.py`'s step, gap and turn no longer read this join: they measure the collar's mesh against the lapel's mid-surface
and fit planes across both sides of a seam that is now edge to edge (its turn p90 reads 63° on a seam the sections show
running flat out of the lapel's plane). The flush gauge and the sections replace them for this join.

**Seen in clay:** the ledge along each gorge is gone; the gorge reads as a seam line between two faces at one depth. A
shallow dent is left in the fall about 1 cm from each roll top, where the three creases are.

**Tried and removed (rejections #140 to #142):** carrying each point by its nearest gorge point's move (tore a knot at
the roll top, tip fold 91°); evening the cloth past the flat band (no change to the dent); carrying the move onto the
stand as well (no change, and a dark tick at the roll top). Stopped per the stop rule after two rounds without gain.

**What is left:** the dent by the roll top sits where the gorge turns into the roll line, the same corner that crumpled
the folded collar in phase 15. It wants the fall's chart laid to the flush gorge from the start, not the fall moved
after it is laid.

### The collar's front as a rod and a panel (John, 2026-11-24, from a made-up notch collar on a stand: "they do it in parts, the rod which transfers the loads and the trapezoid that moves load sideways")

The reading: the roll line is a rod carrying the pull along it up the neck; below it the collar's end is a flat
trapezoid (rod, gorge, notch's end, outer edge) that takes the load across to the gorge seam. Tried as a pass on the
flush build (rejections #143 to #145, removed):

- Each meridian of the chart laid straight from its top edge to its bottom edge: seen from F the gorge, the notch and
  the collar's point all lie *behind* each roll top in azimuth, and the gorge bows on the chart, so a meridian crosses
  the outline twice. 1784 to 2692 folds.
- The same, stopped short of the notch: the dent lies outside the ruled band, unchanged.
- The front laid as a membrane held on all four edges: the three creases by the roll top went (3 to 0), the gorge
  stayed flush, but the panel sank onto the stand, which then showed through as a hard ridge along the roll with a
  groove beside the gorge. Worse in clay. Five rounds: stopped.

**What it shows:** the rod and the panel cannot be got by moving one laid sheet. The rod needs its own round turn with
the cloth's thickness, the panel its own flat lay in the lapel's plane, and the two sewn along the roll: two pieces in
the draft, as the made-up collar is.

### Reading: how a tailored notch collar is made (John, 2026-11-24: "Read more on notch collars")

Sources: Strickland, *A Tailoring Manual* (Cornell, the Collar chapter, figs. 69 and 70); Univ. of Kentucky Extension
CT-LMH.171 (terms) and CT-LMH.176 (custom tailoring); *Every Woman's Encyclopaedia*, Practical Lessons in Tailoring 8
and 9 (the bridle).

1. **Two collars, two gorge seams.** The under collar (bias, a seam at the centre back, canvas pad stitched to it) is
   sewn to the *coat's* neck from collar end to collar end. The upper collar (length grain) is sewn to the *facing*.
   Both seams are steamed wide open and laid one on the other. "If collar is correctly stitched, the notch will form a
   sharp V, and the stitching seams appear to be continuous from lapel through collar" (Manual). So the gorge is a
   butt seam in one surface: the flush gorge of phase 16 is right, the collar laid on the lapel was wrong.
2. **The stand is a separate zone, and it ends on the lapel's break line.** The stand is the part from the fold to the
   neck edge. It is marked on the figure from the centre back "toward front where fold touches front break line"
   (Manual, fig. 69e): the collar's fold runs into the lapel's fold as one line, and the stand ends at that point.
3. **Stand and fall are shaped differently** (the rod and the panel). The stand is pad stitched in arcs parallel to
   the fold, close rows (1/8 in. apart), "shaping collar back to an inside curve to fit back neck". The fall is pad
   stitched loosely (1/2 in.) with the canvas eased over it, most ease at the fold "because of the sharp turn". Then
   the whole is steamed over a roll bent to the neck's size.
4. **The fold lines are drawn in.** The lapel's roll line carries a bridle, a straight tape 1/2 to 3/4 in. shorter
   than the line (Kentucky), which the crease is eased onto: it stops the bias fold stretching. The collar's roll line
   gets a drawing-in stitch. The roll line is the one edge held to length: the rod.
5. **Ease at the join.** Round the neck the coat is eased onto the collar; for 1/2 in. either side of where the stand
   meets the break line the collar is eased onto the coat instead; the collar end is matched "tightly into the lapel
   notch" and tacked 1/8 in. behind it.
6. **Turn of cloth.** The upper collar is cut about 1/4 in. larger than the under collar and eased onto it along the
   outer edge and the ends; its edge sits 1/16 to 1/8 in. past the under collar's.

**For the build:** (a) keep the flush gorge; (b) draw the collar's roll line as the continuation of the lapel's
crease, held to its length, with the stand ending where the two meet; (c) build stand and fall as two zones on that
line, the stand curved in to the neck, the fall a flat panel from the roll line to the outer edge and the gorge; (d)
put the join's ease in the last 1.3 cm each side of the roll top, where the dent is now.

### The gorge drafted flush from the start (John, 2026-11-24: "Try it"; EXPERIMENTAL, `notchcollar` `"gorge_draft": 1, "draft_fade": 0.6`)

Measured first: the collar's roll line already runs on from the lapel's crease (7 and 8° off it) and the stand already
ends at the roll top, so points (b) and the stand's end of the reading were in the draft. What was not: the seam. With
`gorge_draft` the gorge is drafted on the lapel's top edge a cloth behind the lapel's face, so the fall is laid, hung
and settled to that seam instead of being moved onto it afterwards.

| her left (her right the same) | drafted collar | flush by after-pass (`gorge_flush`) | flush in the draft (`gorge_draft`) |
|---|---|---|---|
| collar face against the lapel's, 1.3 to 4 cm along the gorge | 7 to 9 mm proud | within 0.3 mm | within 0.3 mm |
| the fall beside the gorge (clay) | ledge on the lapel | a dent 1 cm from the roll top | flat, no dent |
| creases over 40° | 0 | 3 (53°) | 9 (66°), all at the roll top |
| the roll top (clay) | clean | clean | torn into small teeth |

Five rounds (rejections #147, #148): the lapel's own normal is not defined at the roll top, where the cloth turns over
its fold (the seam zigzagged, 176°); with the top's normals taken from further along, a collar a cloth behind the
lapel's face at the roll top lies inside the lapel's fold and is pushed out (149°); faded up to the lapel's surface
over the first 0.6 it still tears (66°). Stopped at five.

**What it shows.** The fall as a flat panel comes from drafting the seam, not from flattening afterwards: the panel is
right in this build. What is wrong is the one point where our seam differs from the tailor's: ours *starts* at the
roll top; a tailor's gorge seam crosses the roll line and runs on under the fold to the neck, so collar and lapel
turn over the fold together as one cloth. That needs the lapel's top edge carried on past the crease in foldfront
(`lapels.npz` gives the edge from the crease out only) and the collar's outline cut to it.
