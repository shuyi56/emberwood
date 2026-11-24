# Face flow, brow to mouth: research, measurements and the first rebuild

John, 2026-11-24: "work on updating our faces to get total facial flow correct, read up on how it's done by real
sculptors, from eyebrow down to the mouth." His reference is a front close-up of a model's "baby face", with his
lines drawn over it in yellow: the brow running into the side of the nose, the lid crease running in past the inner
corner and down the nose's side round the wing, the philtrum's columns, and the upper lip's line.

## 1. How sculptors and surgeons build this region

| Rule | Source | Used for |
| --- | --- | --- |
| Block the face in order: break the front (tilt the forehead back, carve the sockets), shape the mid-face (a nose wedge, the cheekbones), add the mouth and chin; then refine the eyes, "nose wings, philtrum, lips, and mouth corners", the nasolabial fold; then "blend planar breaks into curves". | Anatomy for Sculptors, Forming the head: simple to complex | The order of the fixes: the bridge's planes before any detail. |
| The eye sockets are "angled up from the outside to the keystone"; their lines carried up through the brow give the forehead's planes; then the nasal aperture, the cartilage, the nose's planes, the dental barrel and the chin. | Proko, Planes and forms of the head | The brow and the bridge as one wedge between the sockets. |
| On the lid "three planes wedge together: nose bridge to eye; brow to cheekbone"; the face plane is broken "from the outer corner of the cheek bone to the center of the upper lip". | Bridgman, Constructive Anatomy (The head: planes) | The outer line (lid crease to the nose's side). |
| The brow-tip (dorsal aesthetic) line runs from the medial brow down the side of the bridge to the tip-defining points: "two gently diverging curves", an "hourglass", "slightly wider at the radix, narrow at the middle third and then widen at the tip", "subtle, unbroken, and fluid". | Park 2011, Fundamental principles in aesthetic rhinoplasty (doi 10.3342/ceo.2011.4.2.55); Cangello and Hilinski nasal analysis; plasticsurgerykey, Nasal analysis and anatomy | Line 1 of the gauge and its waist. |
| The dorsal lines' width should match the tip-defining points' (5-6 mm apart); the bony base is 70-80 % of the alar base. | Brito, Avashia, Rohrich 2020, the 10-7-5 nasal analysis (PMC7159929); plasticsurgerykey (80 %) | The bridge's crest and walls. |
| The alar base equals the intercanthal width (one eye's width); the wings sit inside the plumb lines from the inner corners. | plasticsurgerykey (fig. 17.9); Hilinski; Cangello; Park (1-2 mm inside) | Checked, unchanged. |
| The radix lies between the lash line and the lid crease; its depth, nasion to inner corner, about 15 mm. Nasofrontal angle 115-125 degrees (women), vertex within 1-2 mm of the lid crease. | plasticsurgerykey; Cangello; Hilinski | The side walls' slope: 15 mm over a half intercanthal width of about 16 mm, a mean slope near 1. |
| Nasolabial angle 100-105 degrees (women); columellar show 2-4 mm; Goode 0.55-0.60. | Hilinski, Cangello, Park | Measured by `face_lines.py`, unchanged here. |
| The philtrum: the midline groove bordered by the pillars, whose feet are the peaks of the Cupid's bow; the bow curves "medially and superiorly from the commissures". | NIH Elements of Morphology, lips | Line 3. |
| The vertical thirds; the face five eyes wide; the eye's width equals the intercanthal width (the canons hold for under a third of women: Al-Sebaei 2015). | Al-Sebaei 2015 (Head & Face Medicine 11:4); Proko | Read, not used as a target: John's reference is his target. |

Two sources could not be read (Thieme's brow-tip paper and two PMC papers sit behind bot walls); the rules above
come from the pages that could.

## 2. The reference, measured

Measured with `refmeasure` (PR #89) on the photo cropped from John's screenshot (`/mnt/project-files/face-flow/ref.png`,
sheet `ref.measure.json`), unit the pupils' distance (369.8 px). The photo has a 5.6 degree roll (levelled on the
pupils) and a slight turn to the image's left (its right eye measures 0.53 unit wide, its left 0.43), so both sides are
measured and averaged. The yellow strokes are traced from the pixels, not by hand.

| Measure | Reference |
| --- | --- |
| intercanthal width en-en | 0.620 |
| eye width ex-en (mean) | 0.477; eye / intercanthal 0.77 (wide-set, as a baby face is) |
| alar width al-al | 0.547; al / en-en 0.88 |
| mouth width ch-ch | 0.774; 0.97 of the medial limbi's distance (corners under the inner edge of the iris); 1.42 of the alar width |
| bow peaks apart | 0.187 |
| tip-defining points apart | 0.124 |
| subnasale to stomion / stomion to menton | 0.390 / 0.612 |
| brow-tip line half-width: pupils' line, 1/4, 1/2, 3/4 down to the tip | 0.077, 0.063, 0.059, 0.071 (waist 0.057 at 0.44 of the way down; waist / tip-defining points 0.92) |
| outer line half-width at the same levels | 0.333, 0.235, 0.204, 0.212; at mid-bridge 0.75 of the alar half-width |
| brow line height above the pupils at 0.2, 0.3, 0.4 unit out | 0.21, 0.31, 0.38 (it falls in toward the nose and runs into its line: join gap 0.003) |
| polygon sides at 1 mm, sharpest turn over 1.5 mm | brow-tip 7 sides, 36 degrees; outer 9.5 sides, 51 degrees |

## 3. Our faces, before (same camera, same unit)

Unit the pupils' distance (each face's own: 60-63.5 mm). Gap = ours minus the photo's. From `face_flow.py`, also in
`face-flow-data/before.md`.

| Measure | Reference | Ranger (gap) | Rogue (gap) | Warden (gap) | Sorceress (gap) | Knight (gap) |
|---|---|---|---|---|---|---|
| brow-tip half-width, pupil line | 0.077 | 0.168 (+0.090) | 0.070 (-0.007) | 0.107 (+0.030) | 0.098 (+0.021) | 0.104 (+0.026) |
| outer line half-width, pupil line | 0.333 | 0.317 (-0.016) | 0.398 (+0.065) | 0.316 (-0.017) | 0.401 (+0.068) | 0.319 (-0.015) |
| brow-tip half-width, a quarter down to the tip | 0.063 | 0.025 (-0.038) | 0.022 (-0.040) | 0.024 (-0.039) | 0.022 (-0.040) | 0.026 (-0.036) |
| outer line half-width, a quarter down to the tip | 0.235 | 0.078 (-0.157) | 0.431 (+0.196) | 0.075 (-0.160) | 0.440 (+0.205) | 0.443 (+0.208) |
| brow-tip half-width, half way down | 0.059 | 0.031 (-0.028) | 0.029 (-0.030) | 0.029 (-0.030) | 0.028 (-0.031) | 0.032 (-0.027) |
| outer line half-width, half way down | 0.204 | 0.123 (-0.081) | 0.134 (-0.070) | 0.118 (-0.086) | 0.120 (-0.084) | 0.134 (-0.070) |
| brow-tip half-width, three quarters down | 0.071 | 0.038 (-0.033) | 0.036 (-0.034) | 0.036 (-0.034) | 0.035 (-0.036) | 0.038 (-0.032) |
| outer line half-width, three quarters down | 0.212 | 0.204 (-0.008) | 0.193 (-0.019) | 0.202 (-0.010) | 0.189 (-0.023) | 0.216 (+0.004) |
| brow-tip waist half-width | 0.057 | 0.024 (-0.033) | 0.022 (-0.035) | 0.024 (-0.033) | 0.021 (-0.036) | 0.025 (-0.032) |
| brow-tip waist height (down from pupils, in tip heights) | 0.437 | 0.169 (-0.268) | 0.117 (-0.320) | 0.111 (-0.326) | 0.172 (-0.265) | 0.129 (-0.308) |
| tip-defining points half-width | 0.062 | 0.059 (-0.003) | 0.052 (-0.010) | 0.053 (-0.009) | 0.051 (-0.011) | 0.055 (-0.007) |
| waist / tip-defining points (Rohrich: about 1) | 0.922 | 0.398 (-0.524) | 0.425 (-0.497) | 0.446 (-0.476) | 0.408 (-0.513) | 0.455 (-0.467) |
| brow line height at |x| 0.2 | 0.207 | 0.352 (+0.145) | 0.320 (+0.113) | 0.310 (+0.103) | 0.328 (+0.121) | 0.324 (+0.117) |
| brow line height at |x| 0.3 | 0.312 | 0.310 (-0.002) | 0.312 (-0.000) | 0.296 (-0.016) | 0.306 (-0.006) | 0.314 (+0.002) |
| brow line height at |x| 0.4 | 0.376 | 0.309 (-0.067) | 0.310 (-0.066) | 0.297 (-0.079) | 0.306 (-0.070) | 0.310 (-0.066) |
| brow-to-nose join gap | 0.003 | 0.227 (+0.224) | 0.244 (+0.241) | 0.234 (+0.231) | 0.250 (+0.247) | 0.222 (+0.218) |
| outer line at the bridge / alar half-width | 0.747 | 0.534 (-0.212) | 0.630 (-0.117) | 0.520 (-0.227) | 0.563 (-0.184) | 0.526 (-0.221) |
| alar width / intercanthal width (1.0: plasticsurgerykey) | 0.881 | 0.865 (-0.016) | 0.801 (-0.081) | 0.850 (-0.031) | 0.802 (-0.079) | 0.949 (+0.068) |
| eye width / intercanthal width | 0.765 | 0.811 (+0.047) | 0.819 (+0.055) | 0.791 (+0.026) | 0.804 (+0.040) | 0.787 (+0.022) |
| mouth width / pupil distance | 0.773 | 0.443 (-0.330) | 0.514 (-0.259) | 0.514 (-0.259) | 0.531 (-0.242) | 0.477 (-0.296) |
| mouth width / alar width | 1.417 | 0.966 (-0.451) | 1.212 (-0.205) | 1.136 (-0.281) | 1.244 (-0.173) | 0.938 (-0.478) |
| bow peaks apart / pupil distance | 0.187 | 0.176 (-0.010) | 0.161 (-0.026) | 0.224 (+0.037) | 0.152 (-0.034) | 0.016 (-0.171) |
| subnasale to stomion / pupil to subnasale | 0.551 | 0.395 (-0.156) | 0.433 (-0.117) | 0.407 (-0.144) | 0.435 (-0.116) | 0.393 (-0.158) |
| pupils to subnasale | 0.705 | 0.782 (+0.077) | 0.778 (+0.073) | 0.748 (+0.043) | 0.762 (+0.057) | 0.764 (+0.059) |
| brow_tip: sides_1mm | 7.000 | 8.000 (+1.000) | 8.000 (+1.000) | 5.000 (-2.000) | 7.000 (+0.000) | 7.000 (+0.000) |
| brow_tip: max_turn_deg_per_1.5mm | 36.3 | 133.5 (+97.179) | 109.4 (+73.095) | 105.1 (+68.831) | 108.3 (+71.985) | 99.1 (+62.845) |
| brow_tip: curvature_sign_changes | 2.500 | 7.000 (+4.500) | 3.500 (+1.000) | 5.500 (+3.000) | 5.000 (+2.500) | 3.000 (+0.500) |
| outer: sides_1mm | 9.500 | 15.0 (+5.500) | 19.0 (+9.500) | 16.0 (+6.500) | 16.0 (+6.500) | 19.0 (+9.500) |
| outer: max_turn_deg_per_1.5mm | 51.2 | 134.2 (+82.958) | 168.8 (+117.605) | 157.5 (+106.275) | 134.7 (+83.508) | 142.4 (+91.200) |
| outer: curvature_sign_changes | 4.000 | 10.0 (+6.000) | 16.0 (+12.000) | 11.0 (+7.000) | 15.0 (+11.000) | 14.0 (+10.000) |


Three gaps are shared by every face, so they are in the foundation, not in any face's sliders:

1. **The bridge is a fin.** Its lit edge sits 0.022-0.026 unit from the midline at the waist against 0.057, and half
   the tip-defining points' width against about one; the side walls fall at 2.0-2.1 (depth per width) and meet the
   cheek at 0.36 (Ranger) of the alar base against the published 0.7-0.8. The cause is the nose solid's cross-section
   (`NOSE_LIFE.side`, `ridge`, and the upper wall's turn back at 5.5 mm), not the face around it.
2. **The brow does not run into the nose.** The brow's plane break ends about 0.22-0.25 unit (14 mm) from the top of
   the nose's line, where the photo's runs on unbroken. The brow's lower edge is a soft swell, with no supraorbital
   margin to turn under.
3. **The mouth is narrow.** Its corners sit 0.44-0.53 unit apart against 0.77 (and Farkas' adult women, about 50 mm).
   The base mouth (`mouthRelief`, corners at 0.081 head heights) sets it for every face; the Ranger's was narrowed on
   purpose in an earlier round, so this is John's call.

Also seen: the profile's nasofrontal angle 140 degrees (115-125 for women; `face_lines.py` flagged it on 2026-11-20),
and fine horizontal banding across the forehead and cheeks in the raking-light renders of every face, before any
change (the ring grid's shading, not this work).

## 4. The rebuild (`tools/face-creator/apply-face-flow.py`)

Each change from a measure above; the tip, the wings and the nostrils of the nose thread are untouched (the tip's
slope `side[2]`, from which the ball is derived, is kept).

| Change | From | To | Grounded in |
| --- | --- | --- | --- |
| Side walls' slope at the radix and mid-bridge | 2.0, 2.1 | 1.2, 1.28 | radix depth about 15 mm over a half intercanthal width of about 16 mm (plasticsurgerykey, Rohrich); landed on bony base 0.70 (Rohrich 0.7-0.8) |
| Crest half-width at the radix, mid-bridge; cap against the ball (`CT.wm`) | 3.2, 3.0 mm; 0.9 | 4.2, 3.4 mm; 1.0 | dorsal lines as wide as the tip-defining points (Rohrich); wider at the radix (Park) |
| Where the upper wall turns back | 5.5 mm | 8 mm | with the gentler wall it runs on to the inner corner's side (the cut was a step) |
| Above the radix the bridge sinks into the glabella | (cut at its box, 12 mm up) | 0.35 mm per mm squared from 3 mm above the nasion | the step across the brow seen in round 1's raking render |

Rounds on the Ranger (the stopping rule's numbers):

| Round | Bony base / alar (0.70-0.80) | Dorsal lines / TDP (0.85-1.15) | Waist / TDP, 35 deg (ref 0.92) | Brow-to-nose join gap (ref 0.003) | Seen |
| --- | --- | --- | --- | --- | --- |
| before | 0.36 | 0.60 | 0.40 | 0.227 | fin bridge |
| 1: walls 1.1 / 1.15, rim 0.010 | 0.88 | 1.31 | 0.66 | 0.216 | overshoot; a step across the brow |
| 2: walls 1.3 / 1.35, glabella fade, rim 0.016 | 0.67 | 1.02 | 0.65 | 0.197 | step gone; streaks above the inner corners |
| 3: walls 1.2 / 1.28, rim 0.02, brow's lower fall 0.032 | 0.70 | 1.06 | 0.67 | 0.194 | the join barely moved; streaks |
| kept: round 3's nose, rim and brow off | 0.69 | 1.06 | 0.66 | 0.224 | clean on four faces; the Knight's radix, below |

The stopping rule fired on the brow's join: two rounds in a row moved it by 0.02 and 0.003 unit, and the streaks came
with it. A render of the rim and the brow's fall alone, on the old nose, shows the streaks are theirs: the socket's
carve (`orbitCarve`, `orbit.js` and `blender/shared/orbit.py`) re-lays the skin from the lids up toward the brow, and
relief put into `faceRelief` there fights it. So the brow's edge and its join into the nose belong in the socket's own
profile (`orbitProfile`, `brow_ahead`), kept line for line between the JavaScript and the Python. They stay in the
script, off and labelled EXPERIMENTAL, for the record.

## 5. After, all five faces

From `face_flow.py` (`face-flow-data/after.md`) and `face_lines.py`:

| Face | Bony base / alar, before → after (0.70-0.80) | Dorsal lines / TDP, before → after (0.85-1.15) | Nasofrontal (unchanged) |
| --- | --- | --- | --- |
| Ranger | 0.36 → 0.69 | 0.60 → 1.06 | 141 → 140 |
| Rogue | 0.40 → 0.64 | 0.61 → 1.07 | 140 → 139 |
| Warden | 0.35 → 0.74 | 0.63 → 1.11 | 140 → 139 |
| Sorceress | 0.37 → 0.72 | 0.63 → 1.09 | 140 → 139 |
| Knight (men's ranges) | 0.36 → 0.78 | 0.65 → 1.19 | 116 → 116 |

Tip-defining points unchanged on every face (6.45-7.2 mm).

The brow-tip waist against the tip-defining points (this gauge's 35 degree edge) went from 0.40-0.46 to 0.66-0.73
(photo 0.92); the outer line at mid-bridge from 0.52-0.63 of the alar half-width to 0.89-0.94 (photo 0.75, so the
pyramid now reaches a little further out than the photo's groove, while `face_lines`' bony base sits in the published
0.70-0.80). The brow-to-nose join gap is unchanged (0.22-0.26), as section 4 says, and the brow-tip line's sharpest
turn (93-137 degrees) is that unjoined jump, so it will only read once the join is built.

Clay and raking-key renders before and after, all five faces: `/mnt/project-files/face-flow/clay-before-after-all-faces.png`;
the Ranger's turnaround after: `ranger-turnaround-after.png`; the line sheets: `flow-sheet-before.png`,
`flow-sheet-after.png`; face_lines after: `face-lines-after.png`. The eyes' inner corners and the lids are unchanged
in every view; the tip, wings, nostrils and lips are untouched.

**The Knight's radix.** His bridge comes out from under the male brow torus with hard edges: jagged spikes and a dark
hole before this change, a flat wedge with two slivers after (the blend radius between the eyes, `KF_TOP`, tried at
0.8 and 1.0, does not change it). It is the grid's projection onto the solid under an overhanging brow, there before
this work; it needs its own fix and is listed below.

## 6. What to do next

1. ~~The brow's edge and its run into the nose, built inside the socket's carve~~: done in section 7 (the join gap
   0.22-0.26 to 0.04-0.06 on the four women). The Knight's join stays at 0.21 with item 5.
2. The mouth's width, if John wants it: the reference and the norms put the corners under the inner edge of the iris.
3. The nasofrontal angle (140 against 115-125): the glabella's swell and the radix's depth in the profile.
4. The horizontal banding in raking light (the ring grid), before any of the above is judged in a full render.
5. The Knight's radix under his brow torus (the grid's projection onto the nose solid under an overhang).

## 7. Brows round the eye, the socket and the nose's side carved to them (`tools/face-creator/apply-brow-arc.py`)

John 2026-11-24: "Give the brows more defined curves and shape the face accordingly. Thicker more curved that follow the
contours of the eyes. Carve the nose by following the contours of eye curves so it looks less conic like it does now".

What was wrong, measured in the creator's own units:

- The painted brow sat about 5 mm too high and was thin. Its middle ran 26.8 mm over the inner corners' line at the
  head and 32 mm at the peak, so its upper margin was 30-35 mm up, and it was 6 mm tall at the head and 8 mm at its
  widest. Its peak lay past the outer corner and its tail fell only 2 mm.
- Inside each socket's block the skin is a Coons patch spanned over the plain face (`orbitSkinField`). Between the
  brow, the eye and the nose that left one flat plane, so the nose's straight side wall met it along a straight line.
  That straight line is the cone.

The sources:

| Source | What it gives | Used for |
|---|---|---|
| Park, Yun and Son 2017, Arch Plast Surg 44:65 (20 women aged 20-30, gaze ahead) | Upper brow margin over the inner corners' line: 26.9 mm over the inner corner, 30.9 mm over the lateral limbus (always the peak), 28.2 mm over the outer corner, 21.8 mm at the tail's end | The upper margin's table |
| Rajabi et al. 2017, Clin Ophthalmol 11:1333 (60 women) | Medial brow height 7.5 +- 1.9 mm (4-13) | Brow thickness: about 10 mm at the head ("thicker", inside the range) |
| Yalcinkaya et al. 2016, Eur Arch Otorhinolaryngol | Head over the inner corner, tail on the line from the wing through the outer corner, peak over the lateral limbus; her brow several mm above the rim, his on it | Where the brow ends; the rim's gap |
| Proko (planes of the head); Bridgman | Sockets "angled up from the outside to the keystone"; the plane from the nose bridge to the eye | The carve deepest in the upper inner quarter, where the socket runs into the nose |

The changes:

1. **The brow's margins come from tables.** The upper margin follows Park's numbers. The lower margin runs parallel
   to the eye's arc, 10 mm under the upper at the head, 8.7 mm at the peak, and tapers to the tail. The head is rounded
   at its corners instead of cut square. Arch, height and thickness still work as before. His brow is flatter and
   lower, and heavier by 0.15. The old 0.35 on a 10 mm brow made a 15 mm bar.
2. **The brow ridge follows the arc.** Its crest rises and falls with the brow's middle line, at 0.8 of its movement.
3. **The socket is carved under the arc (`S.carve`, added to the skin field's span).** The hollow has these edges:
   - Top: 2.5 mm under the brow's lower margin for her, and on it for him.
   - Inner side: 8 mm from the midline.
   - Bottom: 15 mm under the pupil.

   It is deepest in the upper inner quarter, 3 mm times the socket slider. It is 0.7 of that down the nose's side
   past the inner corner, and a quarter of it elsewhere. The nose is added onto the span after the carve, so its side
   wall runs on down into the hollow. The foot of the wall now follows the eye's curve instead of a straight line.

The gauges, all five faces (`docs/faces/face-flow-data/brow-arc-flow-gaps.md`):

| Face | Brow-to-nose join gap, before → after (ref 0.003) | Bony base / alar | Dorsal lines / TDP |
|---|---|---|---|
| Ranger | 0.224 → 0.058 | 0.70 → 0.70 | 1.06 → 1.06 |
| Rogue | 0.249 → 0.044 | 0.64 → 0.75 | 1.07 → 1.07 |
| Warden | 0.237 → 0.044 | 0.74 → 0.74 | 1.11 → 1.11 |
| Sorceress | 0.256 → 0.045 | 0.72 → 0.72 | 1.09 → 1.09 |
| Knight | 0.249 → 0.209 | 0.78 → 0.77 | 1.19 → 1.19 |

- The nose's ratios held.
- The brow now runs into the nose on the four women.
- The Knight's join is still open. His radix under the torus is broken in the grid (section 6, item 5).

Renders are in the project files (`face-flow/`): `brows-before-after.png`, `render-brow-arc-before-after.png` and
`clay-brow-arc-before-after.png`.

## 8. Brow edges, and the nose chiselled along the brow-to-nose line (same script)

John: "Improve the edges of the brows too rod like", then "Chisel the nose even more, figure out the connection between
eye brow line and nose shape".

**Brow edges.** A real brow is not a band between two clean margins. The changes:

- **Margins:** they wander along the brow on low-frequency noise. The upper margin wanders about twice as far as the
  lower and fades over a wider band.
- **Hair roots:** they spread to 1.4 half-heights instead of 1.05. They thin out toward both margins, the upper more,
  so single hairs stray past the edge.
- **Hairs:** they are longer (0.14 instead of 0.11 of the eye's units), longest in the upper half, where they lie over
  the margin. Each has more lean of its own.
- **Powder:** it is densest along the lower body, thins toward the top, and is patchy.
- **Height:** it is full through the head and body, then eases out on a cosine to a 0.5 mm tail with no slope left at
  the end, instead of the table's blunt end.

**The connection.** The brow-tip aesthetic line runs from the brow's head down the edge of the bridge to the tip. It
is the bridge's shoulder continued up into the lower edge of the brow's head. In the creator that line is made of two
things:

- The bridge's shoulder, the dorsal line.
- The socket's rim under the brow's arc (section 7). It curves down past the inner corner and into the nose's side.

What we tried and kept:

- **The bridge chiselled (`browChisel`).** The crest used to be one hyperbola from the middle to the wall. It is now
  a nearly flat front plane (radius 4 crest half-widths) that turns into the side wall over a 0.7 mm shoulder. That
  shoulder is the dorsal line, a crisp edge of light that runs up into the socket's rim and the brow's head.
- **The socket's corner deeper.** The carve goes from 3.0 to 3.6 mm.
- **His crest 15% narrower.** On the Knight's wider bridge the flat front read too broad: dorsal lines / TDP was 1.33,
  against a norm of 0.85-1.15.

What we tried and dropped:

- **The rim brought in to 6 mm from the midline and squared under the brow's head.** The brow came loose from the
  nose again: the join gap went from 0.04 to 0.10.
- **The dorsal lines flared out above the radix into the brows' heads.** No effect, because above the radix the bridge
  sinks into the glabella. These are left in the script as MED_X, MED_P and FLARE, set off.

The gauges, all five faces (`docs/faces/face-flow-data/brow-arc-flow-gaps.md`):

| Face | Waist / tip-defining points, before → after (ref 0.92, Rohrich about 1) | Join gap | Bony base / alar | Dorsal lines / TDP |
|---|---|---|---|---|
| Ranger | 0.66 → 0.73 | 0.058 → 0.043 | 0.70 → 0.65 | 1.06 → 1.10 |
| Rogue | 0.70 → 0.77 | 0.044 → 0.044 | 0.75 → 0.62 | 1.07 → 1.09 |
| Warden | 0.72 → 0.80 | 0.044 → 0.043 | 0.74 → 0.74 | 1.11 → 1.14 |
| Sorceress | 0.69 → 0.79 | 0.045 → 0.042 | 0.72 → 0.69 | 1.09 → 1.09 |
| Knight | 0.73 → 0.76 | 0.209 → 0.211 | 0.77 → 0.76 | 1.19 → 1.15 |

- The bridge's waist moved toward the reference on every face.
- The join held.
- The bony base slipped under the 0.70-0.80 norm on the Ranger, the Rogue and the Sorceress. The flat front plane
  puts more of the width into the front, so the walls reach the cheek a little further in.

Renders are in the project files (`face-flow/`): `brows2-before-after.png`, `render-chisel-before-after.png` and
`clay-chisel-before-after.png`.

## 9. The lower nose carried on from the chiselled bridge (same script, step 6)

John: "The top surface of the nose looks good but now the bottom surface needs to be updated accordingly to the new top
traced by the eye brows".

The lower nose is one wedge (front, sides and base), with the ball as its rounded end, blended into the bridge. Two
blends were wide, so the tip stayed a ball under the bridge's new planes:

- The wedge's front and sides met over a blend of 0.9 tip radii.
- The bridge and the wedge met over 5 mm.

Both are tightened: `kball` to 0.5 and `kjoin` to 3.5 mm. The bridge's dorsal lines now run on into the tip's sides,
and the lobule reads as a front plane between two side planes. One round went further, to 0.3 and 2.5 mm. It put
spikes at the nostrils' corners, and the Ranger's tip-defining points (TDP) jumped to 10.8 mm, so it was dropped.

| Face | TDP mm, before → after | Dorsal lines / TDP | Waist / TDP (ref 0.92) |
|---|---|---|---|
| Ranger | 7.5 → 8.1 | 1.10 → 1.02 | 0.73 → 0.69 |
| Rogue | 6.75 → 6.6 | 1.09 → 1.11 | 0.77 → 0.79 |
| Warden | 7.35 → 7.35 | 1.14 → 1.14 | 0.80 → 0.81 |
| Sorceress | 6.75 → 6.45 | 1.09 → 1.14 | 0.79 → 0.81 |
| Knight | 7.35 → 7.2 | 1.15 → 1.15 | 0.76 → 0.74 |

The bony base and the join are unchanged.

Renders are in the project files (`face-flow/`): `render-tip-before-after.png` and `clay-tip-before-after.png`.

## 10. The shade from the eye to the nose, and the nose's side chiselled (`shadow_trace.py`; same script, steps 7 and the valley)

John: "Trace the shadows from eye to nose and why are ours don't have that smooth wave like curve", "Our shadow ends
before the grooves", and, circling the side of the Rogue's nose from the inner corner to the wing, "This entire
sections needs to be chiselled".

**The trace (`tools/gauges/face/shadow_trace.py`).** It follows the shade's darkest line from under the brow's head
to the wing:

- On his photo, with the yellow marks painted out.
- On ours, lit by a soft frontal key, from the same camera and in the same unit, with the same opening masked round
  each eye.

On the photo the line comes straight down beside the bridge from the brow's head. It sits 0.18 pupil distances out at
the brow's head, 0.135 at the inner corner and 0.115 on the bridge's side, then swings out to 0.18 round the wing. It
is one smooth wave: three bends, the sharpest 27 degrees per 0.05.

On all five of ours the line started out at 0.37-0.40 under the brow, in the socket's roof. It came in past the eye
to 0.20-0.22 at the inner corner, and only then did it reach the nose's side. So the side of the bridge was lit from
the brow down to the eyes (the shadow "ends before" it reaches the bridge's side), and the line kinked where the
socket's shade handed over to the nose's: 33-49 degrees per 0.05. Below the eyes it runs along the nose's side and
round the wing, as the photo's does.

**The changes:**

- **A valley beside the bridge (in the socket's carve).** It runs along the photo's line, 10.9 mm out at the brow's
  head and 8.3 mm at the eyes, 2 mm deep and 3.5 mm wide, from 26 mm above the eye line to 16 mm below it. The shade
  now comes down the bridge's side from the inner corner on four faces: 0.12-0.15 there, against the photo's 0.135.
  The Ranger stays at 0.23.
- **The nose's side chiselled.** The nose blends into the cheek over the middle of the nose across 1.6 mm, down from
  5 mm (`KF_MID`). The side plane ends in a line, the nasofacial groove, instead of a smear. This also brought the
  bony base back inside the norm on all five faces: 0.71-0.74, where it had been 0.62-0.69 since section 8.

**What is still open:**

- **The shade above the eye.** It still sits in the socket under the brow (0.37-0.40 against the photo's 0.18). The
  nose's solid ends about 9-12 mm above the nasion (its box, `zN + 9 mm`, and `D()`'s `zN + 12 mm`), so above that
  the bridge's sides are the plain face. Carrying the sinking of the bridge into the glabella higher changed nothing,
  so the limit is the solid's box. The next step is to carry the bridge's side planes up to the brow's head inside
  `noseSolid`.
- **The brow-to-nose join gauge.** It went from 0.04 to 0.13-0.16 with the valley. The gauge follows the lit front
  plane's edge, and the valley beside the bridge now separates that edge from the brow ridge. Read it with the shadow
  trace, which is what John asked about.

Before and after tables: `face-flow-data/shadow-gaps-before.md` and `face-flow-data/shadow-gaps-after.md`. Renders are
in the project files (`face-flow/`): `shadow-trace-before-after.png`, `render-nose-side-before-after.png` and
`clay-nose-side-before-after.png`.
