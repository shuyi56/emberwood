# Zip ankle boot: retrospective and lessons learned

Thread "Build zip-up ankle boots", PR #90, branch `claude/project-thread-ssgfnv`, 2026-11-21 to 2026-11-22. John asked
to save the lessons and merge. Causes are marked **tuning** (a number set wrong), **modelling** (the shape or the
method was wrong), **rig** (the pose or the weights) or **solver**.

## What was built

- `blender/zipboot/`: a sock-fit ankle boot with an inside zip on the heeled boot's foot, last, sole and stacked
  waisted heel (3 cm). One upper piece from the zip round the toe to the back seam, an inner quarter, a zip guard.
- A square toe box drafted as a lastmaker carves one (`last.py`, `toe_box`): superellipse sections over short upright
  sides, the height swept down to a low front wall, the front folded onto a raked cut plane with a soft clamp and a
  bow toward the corners. The sole follows it with an angle-weighted blur (`sole_toe`).
- The zip as real parts (`zipper.py`): a #5 coil, tapes, bottom stop, two top stops, slider with its bridge, a long
  tapered pull, nickel finish.
- The finish: dark chocolate suede (`closeup='suede'`), stitching tone on tone in a finer row.
- Gauges (`gauge.py`): six measures of the leather as a surface, run on any boot figure (`--figure`, `--side`).

## What failed, and why

| Problem John saw | Numbers | Cause | Fix |
| --- | --- | --- | --- |
| Toe round, then pinched, then bumpy | eleven toe drafts (v1 to v11) | modelling: the toe was a smoothed copy of the toes, not a carved box | sections as superellipses (power 2 to 2.2), a cut plane at the front, separate toe allowance (`toe_spare` 0.08) |
| Toes showed through the leather | red hose through the toe in the model view | tuning: rounder sections took room the toes needed | `toe_spare`, fewer ease passes |
| Sole rounder than the toe | sole corners stood out | modelling: one blur all round the sole | `sole_toe` 0.7 round the toe, `sole_blur` 4.0 elsewhere |
| Diagonal crease across the toe box | fold over 25 deg in the posed view only | rig: the hose's weights bent the leather where the toes bend, mid-box, under a 14 deg toe pitch | `bend_at_ball`: the toe bone carries the box whole; the band at the ball widens with height (`BEND_FAN` 0.8) |
| Shaft lopsided from the front | the two sides moved 0.17 units apart on average, 0.57 at the 99th (1.5 cm) | rig: the stance's 7 deg turn-out was made at the ankle, twisting the shaft on the foot | `female.TURNOUT_AT_HIP`: 0.005 and 0.024 (0.6 mm) after |
| Shaft too round, too wide | 28 cm round on a 22.5 cm leg; depth to width 1.4 against the leg's 1.2 | modelling: the foot's 8 mm clearance carried up the leg, the instep's room up the shin | clearance by height (`shaft_gap` 0.013 H from 0.16 H over the ankle), instep room ends at the ankle: 24 cm, 1.18 |
| No stretch in the fit | | modelling: the shaft drawn at its worn size | cut at 0.90 of worn girth (`SHAFT_CUT`), the topline 23.4 cm as cut |
| Ledge at the sole round the heel | up to 9 mm | modelling: the foot's heel is a ball, so the leather turned in under it; the sole was cut to the widest point | `seat_wall`: the upper's lowest 0.6 units stood out to the sole's rim, one push per direction, capped at 0.45 |
| A tighter shaft left the heel a bulb | | modelling: a change in one region left its neighbour loose | `heel_gap` 0.024 H behind the instep |
| The zip read as a stub | a 1 cm pull | tuning: sizes guessed | #5 coil (boot zips are sold in it), top stops, a pull a fifth of the zip's length |

### What went wrong in the process

- The diagonal crease was blamed on sewing, the blend length, the section fold and smoothing before a `--rest` render
  showed it was posed only. **Render at rest first.**
- `bend_at_ball` first diffused the non-bone group `skive` and left points unweighted; the sole had no groups and
  sank. **Write only bone groups, and give every part of a skinned object weights.**
- A taut re-lasting in posed space printed the foot through and was removed. A containment rule for the toe height
  blew the toe up and was removed.
- The pose asymmetry gauge was swamped by the figure's stance until the foot's rigid motion was taken out with a
  Kabsch fit. **Remove rigid motion before comparing sides.**

## The gauges, and when to use each

| Gauge | Measures | Trouble | Run it when |
| --- | --- | --- | --- |
| Gaussian curvature (angle defect) | how far a piece is from a sheet that lies flat | large total curvature on a piece meant to be one bent sheet | a new piece is cut |
| Strain (principal stretches of each triangle's metric) | pull and squeeze from rest to posed | under 0.9 or over 1.15 | weights, bend bands or the pose change |
| Fold angle (dihedral) | creases | a line over 25 deg that is not a seam or an edge | a line shows and you need to know if it is geometry |
| Mean curvature (cotangent Laplacian) | bumps and dents | isolated spots on a smooth area | after smoothing, or a lumpy toe or heel |
| Rest symmetry by station | the draft's evenness side to side | the ridge or middle off the axis | after drafting a toe or vamp |
| Pose asymmetry (mirror partner, Kabsch) | whether the two sides bend alike | mean over 0.02 units | a lopsided fit, or any rig change |

Caught: the turn-out twist (pose asymmetry), the toe's mid-box bend (fold and strain), the squeeze across the vamp
(strain). Missed: the sole ledge and the loose heel (they are fit against a neighbour, not a surface measure; the
section measure in `meas.py`-style checks, leather against leg and upper against sole, caught them), and the toe's
shape against the photo (no gauge; only John's eye).

## Applied to the laced leather boots

| Boot | Measure | Before | After |
| --- | --- | --- | --- |
| Tall laced, left (20 deg turn-out at the ankle) | shaft sides apart, mean / 99th | 1.29 / 2.17 units | 0.005 / 0.017 |
| Tall laced, left | posed stretch, outer shaft | 0.74 to 1.41 | 0.88 to 1.13 |
| Tall laced, left | vamp squeezed over 5% | 24% of its area | 0% |
| Tall laced, left | tongue stretch | 0.78 to 1.60 | 0.98 to 1.21 |
| Tall laced, right (no turn-out) | shaft sides apart | 0.008 | unchanged: nothing to fix |
| Heeled laced (7 deg turn-out, 14 deg toe pitch) | shaft sides apart, mean / 99th | 0.43 / 0.73 | 0.001 / 0.007 |
| Heeled laced | posed stretch, shaft | 0.90 to 1.13 | 0.97 to 1.03 |
| Heeled laced | toe cap squeezed over 5% | 26% | 17% |
| Heeled laced | outer shaft creased points at rest | 136 | 187 (the seat wall's turn at the sole: watch it) |

The hip turn-out is on for both (`boot.TURNOUT_AT_HIP`); the heeled boot also takes `bend_at_ball` and `seat_wall`
(`heelboot.BEND_AT_BALL`, `heelboot.SEAT_WALL`). The laced boots keep their own ease: they are laced, not sock-fit,
so the clearance-by-height and the stretch cut were not applied.

## Do differently

1. Gauge before and after every shape, weight or pose change, and read the numbers before the render.
2. Look for the cause in the rig before the shape: the twist and the crease were both the pose.
3. Fit by region, then check the region next door.
4. Take sizes from a source or a measurement (the zip gauge, the stretch), and mark the rest EXPERIMENTAL.

## Tools

- `python3 blender/zipboot/gauge.py --figure <zipboot|heelboot|boot> [--side L] --out DIR`
- `python3 blender/zipboot/closeup.py --figure <zipboot|heelboot> --out DIR --views outer,front,inner,back --looks clay,leather [--focus toe|heel|zip] [--rest] [--set=module.NAME=value]`
- `python3 blender/base/views.py --figure zipboot --mannequin --turn [--clay] --out DIR`
