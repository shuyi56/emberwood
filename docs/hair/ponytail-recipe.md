# A ponytail from a photo: the recipe

John, 2026-11-28: "save the pipeline". This is the general pipeline the Warden's swept ponytail was built with
(2026-11-25 to 11-28), written so the next ponytail, or any style that is gathered to a tie and then falls, can be
built from it. Every tool named here is on main under `tools/`. The round-by-round log of the first build, with every
number and rejection, is in the project files: `hair-rebuild/ponytail-flow/README.md` (the field) and
`hair-rebuild/ponytail-flow/bangs/README.md` (bangs, creator asset, tail locks, shading).

The order matters and does not change: **measure, field, gauge, look, then hair**. Nothing is groomed until John has
seen the flow field.

**Run it from a style file.** Every step below runs from one parameter file through one runner:
`python3 tools/hair/pipeline/run.py tools/hair/pipeline/styles/warden-ponytail.yaml OUT`. A new ponytail is a copy
of that file with the new photo's ratios and the style's own numbers; the commands below are what the runner runs.
See `tools/hair/pipeline/README.md`.

## 0. Inputs

- The head: the Face Creator's export, `HEAD.json.gz`, from
  `node tools/render-queue/rq_hair_export.cjs face-creator.html DIR <style>` (for the Warden,
  `hair-rebuild/warden-face/warden-head.json.gz`). Faces and the scalp come only from the creator.
- The reference photo, measured with refmeasure (PR #89, `tools/refmeasure/`). Unit: head top to chin (menton).
  Plumb line through the ear canal. Write these ratios into the sheet, all in heads:

  | ratio | Warden value | what it sets |
  | --- | --- | --- |
  | tie above chin | 0.545 | the tie's height |
  | tie behind ear canal | 0.510 | the tie's place front to back |
  | bundle width at the tie | 0.252 | the tie's aperture (check it against packing, step 2) |
  | arc top above / behind the tie | 0.146 / 0.105 | the launch angle |
  | tail length (vertical) | 1.71 | the hair's length |
  | tail behind the tie, widest width | 0.20 / 0.63 | the envelope (R*) |
  | fall angle off plumb | 1.1 deg | the free fall |
  | wave length / crest-to-trough swing | 0.55 / 0.097 | the S wave |
  | fringe edge above chin | 0.575 | the bangs' length |
  | hair over the crown | 0.03 | correct the photo's unit (it includes hair) |

  **Check the face landmarks on gridded crops before trusting the sheet.** The first sheet put every face landmark
  60 to 100 px too far forward, which invented a 0.11 head "tie too low" gap and moved the temple lock 0.1 head.
  The ear's top is usually under the hair: record where the hair's edge crosses the ear and, separately, an estimate
  of the ear's own top.

## 1. The flow field (no hair): `tools/hair/ponytail_flow.py`

```
python3 tools/hair/ponytail_flow.py HEAD.json.gz OUT/field --photo reference.measure.json   # --no-cache rebuilds all
```

The model is John's: s runs from root to outlet, u across the flow, F(s, u) strands per unit width,
dF/ds + d(F m)/du = r, with F >= F_min, |m| <= m_max, |dm/ds| <= K, r >= 0, flux conserved, one sink per basin,
monotonic progress, no rank swaps. How each stage meets it:

1. **Boundary and initial conditions from the ratios** (the `PH` dict and the constants under it). Every constant is
   flagged SOURCED, MEASURED, CALIBRATED or DRAFT. To make a new ponytail, change `PH` and the flagged constants,
   not the code.
2. **Tie aperture against packing.** Narrowest tie = sqrt(N x pi d^2 / 4 / (pi phi)) x 2. Packing at a tied clamp
   is 0.95 [GWB12] (John's 0.6 to 0.7 was a guess at it). Hair diameter 80 um (`scalp_field.SHAFT_UM`). The
   photo's 46 to 55 mm tie is a loose tie at packing 0.2; a tight band packs to about 0.5 (`tie_funnel.py`).
3. **Funnel over the scalp.** Every root runs to the tie along the great arc from its direction to the tie's, seen
   from the skull's centre and laid on the skin. Arcs to one point never cross and never turn back, so rank and
   monotonic progress hold by construction (the gauges still re-measure them). Funnel length must be at least
   (W - w) / (2 tan theta_max).
4. **Gravity on the sides and nape.** The great arc is a taut hair. A hair with slack sags across its line where the
   skin is steep, over the ear's top, and comes up into the tie from below: `SAG_G` x gravity across the line, faded
   out near the tie (`SAG_FADE`). Re-solve only the lines that need it (`RESOLVE`), the rest come from the cache.
5. **The ear is a surface, not a wall.** The ear lines run to lanes over the ear's upper part at an offset
   (`EAR_COVER`, `EAR_OVER_MM`, `EAR_LANES_MM`), then the shortest way to the tie. Place them at the creator's ear,
   not the photo's: if the two ears differ in height, the route is kept and the height is the model's.
6. **Lift is not a free profile.** It is the stack's thickness: flux through the cut x one hair's section / packing.
   Packing rises from loose (0.165) to the tie's over the last `PHI_NEAR_MM` of scalp.
7. **Handover at the tie.** A quintic from the scalp line into the bundle's axis, matched in position, direction and
   curvature (G2) at both ends. A line's place in the bundle keeps its bearing (order) and its depth in the stack
   (inside to outside), so the density is continuous across the tie (gauge: sector density 1.000).
8. **Free fall.** The bundle's middle line is a heavy elastica launched at the measured angle (use its first
   integral, not a two-point solve, which converges on a climbing branch), fitted to both the arc's top and where the
   tail hangs. Its envelope is the ponytail shape equation [GWB12]
   l^3 R'''' - (L - s) R'' + R' - Pi(R) = 0, Pi = Pi0 (1 - R/R*), Pi0 0.85, bending length l about 5 cm.
   The S wave is laid on the middle line and moves every strand alike (tilting the section with the wave folds the
   strands inside each bend); it grows in over a quarter head below the tie, since [GWB12] finds the curl "reduced in
   the vicinity of the clamp".
9. **Bangs as their own basin:** `tools/hair/bangs_flow.py` (step 4 below), read off this field's cached side flow.
10. **Loose locks** are thin extra basins (1 to 5% of the flux), peeled off where the flow already fans (temple, in
    front of the ear), tangent matched at the peel, tapering, never crossing the main flow. When John draws a line
    for a lock, store it (`ponytail_john_lines.json` holds it in the photo's pixels and in the model) and lay the
    lock along it, gathering into a ribbon (`LOCK_GATHER`). Cut the lock at the ear's front edge so the sideburn
    hairline stays covered.

Outputs in `OUT/field`: `flow-field-views.png` (side at the photo's camera, front, back, 3/4; basins, tie and locks in
flat design colours), `flow-F-map.png` (F on the scalp unrolled round the tie, and the cut-by-cut balance sheet),
`flow-over-photo.png` (the field's outline on the photo with every gap), `hairline-gauge.png`, `gauges.json`.

Cache: `OUT/../cache` keeps the stages (scalp, blue lines, free fall, bangs) keyed on the head and the constants, and
content-addressed memos of the hairline gauge and the drawings. A repeat run takes about 15 s; change one basin and
only that basin is recomputed.

## 2. Gauges on the field, and the bar to pass

| gauge | bound | Warden (field6) |
| --- | --- | --- |
| flux per cut against the roots beyond | conserved (within 2%) | within 1.8% on 6 cuts |
| strands through the tie / total | one sink per basin | 90.9% (the rest: bangs, lock) |
| F at the tie | <= 0.95 packing | 0.21 |
| funnel length | >= (W - w) / (2 tan theta_max) | 78 mm against 34 |
| rank swaps between neighbours | 0 | 2 of 6,713 (inside the tail's locks) |
| lines turning back | 0 | 0 |
| hairline (`hairline_gap.py`, its coverage definition) | 0 bare cells, >= 4 per mm | 0 bare, thinnest 9.5 per mm |
| slack (arc / way along the skin) | >= 1.005 (DRAFT) | median 3.3% |
| handover | 0 mm, 0 deg, density continuous | 0, 0, 1.000 |
| loose fraction | 1 to 5% | 0.76% (only the temple lock is loose) |
| curl, max dm/ds | radius >= 8 mm (DRAFT, from the ear's rim) | fails at the fold into the tie |

Known: the curl bound fails where the bundle folds over the tie. A bound taken from the ear's rim is the wrong bound
for a fold at an elastic; a fold-at-the-tie bound is still to be found. Report it, do not tune the field to hide it.

Then **show John the field** (views, F map, balance sheet, overlay with gaps, gauges) and stop. Real hair is built only
after he approves it.

## 3. The field as fibres (a look before any groom): `tools/hair/ponytail_render.py`

```
python3 tools/hair/ponytail_render.py HEAD.json.gz OUT/render --per-root 1 --size 750
```

Every root's line drawn as fibres with `render.py`'s renderer, clay and shaded. No groom, no solve. This is where the
field's faults show as hair: a see-through fringe, a bare stripe on the part, bare skin at the ear, a tail that is one
smooth rope with a pointed tip. Fix them in the field (step 1), not in the render.

## 4. Bangs that join the sides: `tools/hair/bangs_flow.py`, `bangs_join.py`

```
python3 tools/hair/bangs_flow.py HEAD.json.gz OUT/bangs/curtain --field-cache OUT/cache --photo photo.measure.json --preset curtain
python3 tools/hair/bangs_render.py HEAD.json.gz OUT/bangs/curtain OUT/bangs/render --field-cache OUT/cache
```

A bang hair's direction is t = normalize(w(u) B + (1 - w(u)) D): B the side flow toward the tie read off the field's
own lines, D gravity on the forehead turned toward the cheekbone, w = smoothstep(w0, w1, |u|) by the distance from
the part. At w = 1 a bang takes the side flow's direction, so the bangs join the sides with the same tangent by
construction. Presets (`curtain`, `curtain-swept`, `straight`, `side-swept`) are parameter sets over the same two
fields. Gauges: fan opens away from the part, F >= F_min over the body, no stagnation, bounded curl, tips taper,
hairline clean, 0 rank swaps.

## 5. The hair: a Face Creator strand asset, `tools/hair/bangs_creator_asset.py`

```
python3 tools/hair/bangs_creator_asset.py HEAD.json.gz OUT/bangs/curtain OUT/asset.json --field-cache OUT/cache \
    --volume --funnel --mid-layer --trim --tail-locks --join-blend --join-zigzag --noise 1.0
python3 tools/face-creator/apply-hair-fidelity.py IN.html OUT.html ponytail=OUT/asset.json
```

Each flag is one stage, in the order a stylist would do it. Leave a flag off to see what it adds.

- `--volume` (`hair_volume.py`): section the hair as a stylist parts it (top, sides, nape) and lay the top clear of
  the sides it is combed over. Keep the lift flat right behind the fringe (`FRONT_FLAT`): every climb there is seen
  end-on from the front as a tick, and a row of ticks reads as a picket crest.
- `--funnel` (`tie_funnel.py`): the last stretch of every hair rebuilt as a Hermite curve to its own crossing point
  in the tie's plane, so the sides converge on the tie as a cone instead of ending in a cylinder stub. The band's
  radius comes from the hairs it holds at packing 0.5; the bundle flares back past the band.
- `--mid-layer` (`hair_layers.py`): coverage, mid, top. 40% of fibres gathered into clumps under the loose top. Build
  hair from the inside out: an opaque base layer, then breakup, then flyaways (Unreal Hair Card Generator layers;
  80.lv game-hair articles).
- `--trim` (`hair_trim.py`): the fringe and the loose locks cut to a line, point cut.
- `--join-blend`, `--join-zigzag`: where the fringe meets the hair swept back from it, both sets leave the line, so
  cover is thinnest there. Thicken the swept-back hair at the join and interleave the two root sets with a zigzag
  parting, both ways. Gauge it with `tools/gauges/hair/join_cover.py` (cover over 1 layer, buried under 0.1). Keep the
  scalp cap under the hair's roots (`CAP_LIFT_MM` 0.4).
- `--tail-locks`: the tail in locks that part as they fall (step 6).
- `--noise` (`noise_round.py`): lock noise shared within a lock, then smaller strand noise, then a few flyaways; all
  square to the hair, growing from the root. Keep it long and quiet for straight hair (short, strong noise read as
  frizz: rejection logged).

## 6. How the tail's locks diverge toward the ends

A tail drawn as offsets from one centre line is one rope with a texture. Real hair below a tie is a bundle of locks
that each hang and curl on their own, and nothing below the tie holds them together, so they part as they fall.

1. **Assign hairs to locks** where the tail is formed (`TAIL_STATION_MM` 150 mm along it): uneven locks, the nearest
   of random centres in the section (`TAIL_LOCKS`; 33 is right for the Warden, 64 read stringy with 14 rank swaps).
2. **Solve each lock in 3D** as a Kirchhoff rod clamped just past the tie and free at the tip:
   `tools/solvers/hair/tail_lock_rods.py`. Bending, natural curl (helix radius 12 mm, pitch 100 mm, off the photo,
   DRAFT) growing in from zero over 80 mm below the tie, gravity, contact with the other locks (they keep a lock's
   thickness apart) and the head. Under the band each lock is pressed to 0.45 of its radius and swells back over 45 mm.
   Settled means residual under 0.5 mm and joint imbalance under 0.5 rad.
3. **Join the field line to the rod** over 45 mm with a quintic ease (G2 across the join). A hair's place along its
   rod is its own length past the tie, not the centre line's station.
4. **Open each lock as it falls.** A lock is a clump, not a rod: its hairs part into 3 sub-locks whose middles
   spread 1.4x farther from the lock's line at the tip, and each hair spreads 0.7x farther from its sub-lock's middle.
   Both grow as ds/dt = k(t) s, k rising smoothly from 0 at the tie, starting 12% down the lock, so every hair keeps
   its order across the lock (no crossings). The half angle of the opening at the tip is 2.2 deg against a 7 deg
   limit (a diffuser stays attached under about 7 deg; used only to name the rate).
5. **Ride the rod as a ribbon** (thickness 0.5 of the width) that rolls once per 300 mm, drawing in toward a point at
   the tip; one tone per lock, lit round its own lock.
6. **Fair the lines.** `kink_gauge` (turn per node) and `fair` (only nodes turning over 8 deg move): no node over
   20 deg on the head, the join or the tail.

Gauges for the tail: outer-lock rank swaps at 25 / 50 / 75% of its length (0 is the target), locks nearer than half
their thickness, tail width at 0 / 33 / 66 / 100% against the photo (about 3 tie widths two thirds down).

## 7. Shading (creator page): `tools/face-creator/hair-fidelity.js`, `tools/hair/hair_light.py`

Light from the flow and the density, not by eye: forward-scattered light under other hair (dual scattering, Zinke et
al. 2008: d_f 0.7), fibre diffuse (Kajiya and Kay), an opaque coverage ribbon under every strand. A black patch in the
hair is usually the scalp cap showing where roots thin out: paint the cap magenta to prove it before touching light.
Glow terms exist in the shader and are off (John chose no glow).

## Lessons from the first build

- Measure the photo twice. A wrong landmark cost a round and invented a gap. Gridded crops catch it.
- The great arc is the taut hair. Gravity across the line is what puts side hair over the ear.
- Re-solve only the basins John points at; the cache keeps the approved ones exactly as they were.
- Fix faults in the field, then re-render. Every render fault of round 4 (bare part, bare ear, rope tail) came from
  the field.
- A tail is locks, and locks open. A rope with a texture never reads as hair.
- Darkness in hair is usually the cap or the scene, not the shader: debug with flat colours per layer first.

## Still open on the Warden

- Curl over the ear-rim bound at the fold into the tie (needs its own bound).
- From the corrected photo sheet: the arc over the tie is half the photo's height (launch at its 84 deg limit), the
  tail a fifth too narrow (R* calibrated to the old sheet), the side lines 12 to 29 deg flatter than the photo's
  strands behind the ear (the blue lines have no sag).
- Not sourced: a root tension ceiling, a slack number, a lift height, a loose lock's thickness. Clinical sources
  (StatPearls traction alopecia; Samrao 2011, fringe sign) are qualitative: no tenting, no pain.

## Sources

- [GWB12] Goldstein, Warren, Ball 2012, "Shape of a ponytail and the statistical physics of hair fiber bundles",
  PRL 108, 078101: the envelope equation, Pi0 0.85, bending length about 5 cm, outer fibres launched at 17 deg,
  packing 0.95 at the clamp, curl reduced near the clamp.
- Bertails et al. 2006, Super-Helices; Bergou et al. 2008, Discrete Elastic Rods: the lock rods.
- Robbins, Chemical and Physical Behavior of Human Hair, ch. 9: E 3.9 GPa, density 1.3 g/cm3.
- Zinke, Yuksel, Weber, Keyser 2008 (dual scattering); Marschner et al. 2003; Kajiya and Kay 1989: shading.
- Curtain bang sources (curlcentric, bangstyle, behindthechair, mindbodygreen) and Farkas 1994: in `bangs_flow.py`.
- StatPearls NBK470434 (traction alopecia); Samrao et al. 2011 (fringe sign): no numbers, used as qualitative limits.
