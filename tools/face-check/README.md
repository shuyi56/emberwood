# Facial proportion line check

The check John asked for on 2026-11-20 ("what's the check called for faces to look at proper lines"). It is a
**facial anthropometric analysis**: the soft-tissue profile analysis and the rhinoplasty nasal analysis surgeons
and portrait artists run on photographs. Named landmarks (Farkas' points) are found on the head mesh itself, the
standard reference lines are drawn through them, and each measure is compared with a published range.

It belongs to the draft stage, beside the face topology check (`rq_topo.py`): run it on the Face Creator's heads
before any face render, and attach the sheet to the draft. It is report-only and never stops a render.

    node tools/face-check/dump-heads.cjs /mnt/project-files/face-creator/face-creator.html DUMP [Ranger,Rogue,...]
    python3 tools/face-check/face_lines.py DUMP --out DIR [--faces Ranger,Rogue]

`dump-heads.cjs` runs the creator in headless Chromium (Playwright, about 20 s a face), picks each preset and writes
its head, eyes, ears and hair cap as raw meshes. `face_lines.py` needs only numpy and Pillow and takes about 20 s a
face. It writes:

| File | What it holds |
| --- | --- |
| `face-lines-sheet.png` | One row per face: clay front (thirds, eye corners, alar width), profile (nasofrontal, nasolabial, nasofacial and E-lines, Goode's alar-crease line, every landmark), nose front (gull-wing line, dorsal lines, tip-defining points, bony base), nose side (alar rim over the columella, columellar show) and nose base (nostrils and their axes, lobule line). Green in range, amber just outside, red out, blue a house ratio shown for information. |
| `face-lines.json` | Per face: landmarks in the creator's millimetres, every measure with its range, sources and verdict. |
| `face-lines.md` | Which faces fail which measures, then the table of ranges and sources. |

## What it measures

Lengths are in the creator's millimetres (its head height HH = 8.6 units is 219 mm, `NOSE_LIFE.head_h`). The Knight
is judged on the men's ranges where a source gives them.

* **Face:** the vertical thirds tr-n : n-sn : sn-me and the middle-to-lower ratio (which needs no hairline, so it
  holds under a fringe), the lower face split sn-sto : sto-me (1 : 2), the intercanthal width, the orbital canon
  (ex-en = en-en) and the orbitonasal canon (al-al = en-en).
* **Profile:** nasofrontal, nasolabial, nasofacial and nasomental angles, Goode's tip projection ratio, the lips
  against Ricketts' E-line, columellar show.
* **Nose, front:** alar width against nasal height, bony base against alar width, tip-defining points' spacing, the
  dorsal aesthetic lines against the tip-defining points, the gull-wing line (columella just below the alar rims).
* **Nose, base:** lobule : columella (1 : 2), nostril length against base length (2/3), nostril axis (45-60 degrees
  from the midline), the base triangle (equilateral).
* **The nose thread's own ratios**, shown and judged against what that thread set rather than a published norm: wing
  height against wing half-width (0.97), tip ball radius against the bridge, the floor angle and how far the tip
  stands over the floor line.

Every range is in `RANGES` in `face_lines.py` with its source and note; the sources were read on 2026-11-20:
Rohrich's 10-7-5 nasal analysis (PMC7159929), Al-Sebaei's test of the neoclassical canons (PMC4369102), Park's
principles of aesthetic rhinoplasty (e-ceo.org), Powell's aesthetic evaluation of nasal contours (the Powell and
Humphreys triangle), Becker's ideal nose (rhinoplastyarchive.com), Taub's anthropometry chapter and Spear on the
esthetic plane. Where a source gives only an ideal value, the spread round it is this check's tolerance and the note
says so.

## How the landmarks are found

The midline profile is the head cut by the plane x = 0, chained and resampled every 0.1 mm. Along it: the tip is the
most forward point, the nasion the deepest point above it, the glabella the most prominent point within 25 mm above
the nasion, the subnasale the sharpest turn of the profile below the tip, the columella's line a fit to its lower
border, the lips the forward bumps and the stomion the hollow between them, the labiomental fold and the pogonion the
farthest points behind and in front of the chin's chords (so a receding chin still has a pogonion), the menton where
the chin turns under. The trichion is the hair cap's lowest front point on the midline, which a fringe pulls down, so
read the three thirds with that in mind.

From a front z-buffer: the eye corners are the ends of the eyeball's visible part; the nose is wherever it stands
0.8 mm proud of a cheek fitted either side, row by row, which gives alare, the alar crease and the wing's height; the
dorsal lines and tip-defining points are where the bridge's and tip's cross-sections turn 45 and 30 degrees from the
viewer; the bony base is the nasofacial groove at the rhinion's height; the gull-wing line is the nose's lower edge
against the lip. From the side, the alar rim is where the visible surface steps in from the wing to the columella. From
below, with everything under the subnasale cut away, the nostrils are the hollows with a lower rim all round them.
