"""The inner corner's lid blended into the skin round it (combine thread, 2026-11-21). Run after apply-inner-corner.py.

    python3 tools/face-creator/apply-inner-corner-blend.py face-creator.html main.js orbit.js   (in place; every anchor once)

apply-inner-corner.py laid the lid on the ball at the inner corner by putting the ball's depth back into L, the lid
depth. L is used only in the lid's core and the 5e hand-over band round it, and near the corner the core's reach is
small (0.3 of the fold), so the full extra depth (up to 10 mm) was handed back to the face's surface over 5e hugging
the corner. Below the canthus that showed as a vertical notch with a light line running down the side of the nose
(the Blender port's report: the lid dropped onto the ball faster than the skin beside it followed).

Now L keeps the face-plane clamp, and the extra depth is taken only at the margin (within 2e of it), blurred by 1.5e
and added to the surface itself with a fade from 1e to 8e out of the opening. The lid still lies on the ball down to
the canthus (no flesh showing from 3/4), and the skin slopes into the medial canthus over about 8 e as a shallow
hollow, as the soft tissue does over the medial canthal tendon, instead of stepping down at the corner.
Mirror in blender/shared/orbit.py (Dm, Y).
"""
import sys

EDITS = [
    ("    L[k] += (Lb - L[k]) * wmed;      // (the inner corner: the lid apposed to the ball down to the canthus, the lacrimal lake small: apply-inner-corner.py)",
     "    Dm[k] = (Lb - L[k]) * wmed * (1 - osm(0.5 * e, 2.0 * e, dist[k]));      // (the inner corner: the lid apposed to the ball down to the canthus, the lacrimal lake small; added to the surface below: apply-inner-corner.py, apply-inner-corner-blend.py)"),
    ("var reach = new Float64Array(N), fixed = new Uint8Array(N), core = new Uint8Array(N), Dl = new Float64Array(N);",
     "var reach = new Float64Array(N), fixed = new Uint8Array(N), core = new Uint8Array(N), Dl = new Float64Array(N), Dm = new Float64Array(N);"),
    ("""    Y[k] = L[k] + (Y[k] - L[k]) * osm(reach[k] - 2.5 * e, reach[k] + 2.5 * e, dist[k]);
  }
""",
     """    Y[k] = L[k] + (Y[k] - L[k]) * osm(reach[k] - 2.5 * e, reach[k] + 2.5 * e, dist[k]);
  }
  // (the inner corner's extra depth, taken at the margin, softened and faded out over 8e, so the skin beside the corner
  // slopes down with the lid instead of stepping: apply-inner-corner-blend.py)
  var DmB = oBlur(Dm, ny, nx, Math.max(2, Math.round(1.5 * e / h)));
  for (k = 0; k < N; k++) Y[k] += DmB[k] * (1 - osm(1.0 * e, 8 * e, dist[k]));
"""),
]

for f in sys.argv[1:]:
    s = open(f).read()
    for a, b in EDITS:
        if b in s:
            continue      # (already applied)
        c = s.count(a)
        assert c == 1, (f, c, a[:90])
        s = s.replace(a, b)
    open(f, 'w').write(s)
    print('ok', f)
