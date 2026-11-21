# Boots and hose: retrospective and lessons learned

Thread "Hose and boots", PR #68 (squash-merged 2026-11-20, commit 91223c8). John accepted the boots and hose by merging.
Causes below are marked **tuning** (a number set wrong), **modelling** (the shape or proportion was drafted wrong) or
**solver** (the cloth solver cannot do it).

## What was built

- One last for the whole upper; the shaft cut from it as two flat panels and a tongue, each with a flat pattern from a
  least-stretch unwrap (flat lengths 96-105% of the last's), sewn by the local solver. The sewn shell is cached by shape
  and settings, so a material or texture run does not solve again (a first solve costs about 2.5 minutes on top of a
  156-277 s build; a cached run skips it).
- Lacing: eyelets punched slightly off their places, cords laid on the leather, thin eyelet rings.
- Tools: a contour sheet (`tools/contours/`: sections at eight heights, front and side profiles, a bend map on the flat
  patterns) and a colour-map check (`tools/render-review/colorcheck.py`).

## What failed, and why

| Problem | Numbers | Cause | Fix |
|---|---|---|---|
| Round collar ring and a top standing off the leg | top stood about 1 cm off the leg, 4 mm after | modelling (ease carried to the top; no fit to the taut leg above the calf) | ease let down to the top, `top_fit` blends to the leg's section |
| Tongue bend crumpled | five rounds: 28 degrees reached 30/29 but crumpled; 20 degrees overturned to 70; current-length rest gave 25/42 | modelling: the tongue is curved across the shin, so a fold along a line buckles, and the measured angle includes that curvature | stopped under the stopping rule; later the tongue top sat level with the panels, so no bend |
| Tongue slipped out from under the lacing | 47% of the boot's width, top 1.6 cm above the panels | modelling: sized to cover the gap, not to the boot's front | 0.34 of the shin's width per side under each panel: 71-73% of the width, top level with the panels |
| Laces floated | cords up to 7 cord radii (about 4 mm) over the tongue; worst run 11.5 radii, p90 3.2 | modelling: the lift was a fixed offset over a surface that steps down into the opening | each point laid on the face found straight under it; worst 6.2, p90 1.4 |
| Dark lace blackened the tongue | lace lightness 13 over a tongue shadow tone of 4 | tuning (colour), found late | colour check added; see below |
| Wrong lacing gap by eye | 0.9 -> 1.6 -> 2.5 cm, each a guess | modelling: no model of what the gap should be | six-gap sweep with the lace flow check: 2.0 cm |
| An empty top eyelet | rows of 19 and 18 | modelling: rows not paired | paired from the bottom |

### The lacing gap

The sweep (lace flow check on the boot at 0.9, 1.2, 1.5, 2.0, 2.3 and 2.8 cm): tension (cord length over the surface
path it crosses) tightest/median/slackest was 1.06/1.13/1.22 at 0.9 cm, 1.01/1.05/1.10 at 2.0 cm and 0.95/1.01/1.05 at
2.3 cm. Below 2.0 cm the cords are slack and run straight across the shin; from 2.3 cm the tightest drop under 0.97
(pulling) and some leave the surface by 37-39 radii. The lace tension solver in PR #76 independently puts the best gap
at about 1.6 cm, with 1.1-2.3 cm within 3%. This is a proxy, not a force simulation: the cloth solver has no leg
pushing on the leather, so the gap cannot come out of a solve. **Solver limit**, handled with a check.

### Colour

The check compares every touching pair: lightness gap, colour distance, hue gap, grayscale gap, plus a cord against
the shadow of what it lies on, and an accent rule (a ring may not out-contrast the lace). It caught a linen lace with the
brass eyelets' hue (hue gap 6 degrees), and the leather against the red hose (lightness gap 9, grayscale gap 3), which
was settled by moving the hose to a brick red. It could not make dark lace on dark leather pass (it flags two counts),
which John accepted. A bright ring took the eye from the lace; the rule came after the render, not before.

## Which checks caught problems, and which missed

- **Caught:** contour sheets (jagged sections; the bend map showed the tongue's crumple); ray measurements from each cord
  down to the surface (floating, in numbers); the colour check (hue clash, value gap); the lace flow check (tension).
- **Missed:** eyelet pairing (an empty eyelet was seen in a render); lace diameter against real laces (2.0 mm across
  against standard 3 mm round laces; flagged afterwards, labelled experimental); the white spots on the bend map
  (values above the 98th-percentile colour scale; no flipped triangles; left alone by John's call).
- Process misses: a smoke test that exited 0 (no render) was taken as a pass, and a full-body request rendered nothing
  because `--turn` was missing; the same views were requested four times in three hours and the queue flagged it;
  in-flight smoke tests had to be killed when a later change made them obsolete.

## Do differently

1. Take the number from research or a measurement before drafting a change, and name the source in the thread.
   Proportions are relative to the garment (the tongue to the boot's front), not to the gap they cover.
2. Choose the lacing gap, and any ratio John judges by eye, with a sweep and a check, not by steps of "a bit more".
3. Colour at the draft stage: a flat distinct colour per part, the check run on swatches (`--try`) before any file
   changes, the colour map shown with the draft.
4. Contour sheet after every solve and before every render; stop at the stopping rule (two small gains, a worse round,
   the same flaw returning, or five rounds) and give John the diagnosis, as done for the tongue bend.
5. Before queuing a render: the exit code matches the intent (77 means it reached a render; 0 means it did not), one
   view per real change, and the cheapest crop that shows the fault.
6. Count paired parts (eyelets, crossings) as a check.
