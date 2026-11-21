"""The Ranger's eye make-up: a tapered liner and a plum-bronze shadow chosen for her colouring (combine thread, 2026-11-21).

    python3 tools/face-creator/apply-eye-makeup.py face-creator.html [main.js ...]   (in place; every anchor once)

Colour, from her colouring (green iris, copper-red hair #7e4c34, fair warm skin #cf9f84):
  - Green eyes: the reds and purples opposite green on the wheel make it stand out, "reddish tones like mulberry,
    claret and aubergine" (Charlotte Tilbury, best eyeshadow colours for green eyes).
  - Red hair, green eyes, fair warm skin: copper, plum, mauve, bronze in the crease; on paler skin "brownish pinks and
    medium purple shades" (LoveToKnow, best eye shadow for green eyes and red hair).
  So a muted mulberry-brown, #6e4652 (hue about 340, between plum and bronze): opposite her iris, about 40 degrees from
  her hair's hue so the two sit together, and grey 83 against her skin's 172 and her brows' 78 (the brow bone's bare
  skin keeps the two apart). washAt darkens it to 0.49 in the crease and outer V (aubergine) and lightens it 1.32 under
  the lower lashes (a soft mauve). Strength 1.15 puts the lid at about 0.7 opacity: clearly defined, as John asked for, short of the Smoky look (1.5).
Liner, from her eye's shape (open 0.78, a round, open almond) and the eye's real size (the opening is about 30 mm
across, so 1 mm is 0.05 of the eye's units, RX = 0.76 being half the width):
  - Thicker toward the outer corner, and for round eyes taken "slightly outside of the outer corner" (L'Oreal Paris,
    where to put eye makeup depending on your eye shape); thicker where a thin line does not show with the eye open
    (Bobbi Brown, MasterClass, eye types).
  - So the line is now a tapered band along the upper lash line: 0.8 mm wide in the inner third, rising to 2.4 mm at
    the outer corner at liner 1 (0.04 to 0.12 eye units), fading in from the inner corner, and the Ranger gets a short
    wing (0.15) carrying it past the corner. Other presets keep their liner amounts; at their values the new band is
    as wide as the old uniform line in the middle.
  - Its colour is new (linerTone): for the Ranger a deep plum-brown, #3a2228, the same family as her shadow. The
    page opens on DEFAULTS (the Ranger's values) and the Ranger preset; both get her make-up. Every other preset sets
    its own shadow, liner and wing, and the wing code falls back to near-black where no tone is set. Jet black reads harsh against copper lashes and fair skin, and a brown-plum line still
    deepens the lash line against the green.
"""
import sys

EDITS = [
    # the opening look (these defaults are the Ranger's): her eye make-up, and the liner's colour as a setting
    ("shadow: '#463c38', shadowStrength: 1, shadowInner: 1, smoke: 1, lower: 1, liner: 0.4, wing: 0,",
     "shadow: '#6e4652', shadowStrength: 1.15, shadowInner: 1, smoke: 1, lower: 1, liner: 0.75, wing: 0.15, linerTone: '#3a2228',      // (the Ranger's eyes: apply-eye-makeup.py)"),
    # the tapered liner band
    ("""    if (P.liner > 0) {
      tc.lineWidth = 0.016 * PXU * E.k * (0.6 + P.liner); tc.strokeStyle = 'rgba(20,13,14,' + (0.4 + 0.6 * Math.max(0, P.liner - 0.4)).toFixed(3) + ')';
      tc.beginPath();
      for (var k2 = 0; k2 <= 60; k2++) { var uu = -0.96 + 1.96 * k2 / 60, p2 = pt(uu, lidTop(uu) + 0.006); if (k2 === 0) tc.moveTo(p2[0], p2[1]); else tc.lineTo(p2[0], p2[1]); }
      tc.stroke();
    }""",
     """    // (a band along the upper lash line, tapered: 0.8 mm in the inner third to 2.4 mm at the outer corner at liner 1,
    // fading in from the inner corner; the eye's units are about 20 mm, so 1 mm is 0.05: apply-eye-makeup.py)
    if (P.liner > 0) {
      var lt = hex(P.linerTone || '#140d0e'), la = clamp(0.45 + 0.6 * P.liner, 0, 0.92);
      tc.fillStyle = 'rgba(' + lt[0] + ',' + lt[1] + ',' + lt[2] + ',' + la.toFixed(3) + ')';
      var lw = function (uu) { return Math.max(0.006, P.liner * (0.04 + 0.08 * sm(-0.3, 1.0, uu)) * sm(-1.0, -0.7, uu)); };
      tc.beginPath();
      for (var k2 = 0; k2 <= 80; k2++) { var uu = -0.97 + 1.97 * k2 / 80, p2 = pt(uu, lidTop(uu) - 0.004); if (k2 === 0) tc.moveTo(p2[0], p2[1]); else tc.lineTo(p2[0], p2[1]); }
      for (k2 = 80; k2 >= 0; k2--) { var uq = -0.97 + 1.97 * k2 / 80, p3 = pt(uq, lidTop(uq) + lw(uq)); tc.lineTo(p3[0], p3[1]); }
      tc.closePath(); tc.fill();
    }"""),
    # the wing in the liner's colour
    ("      tc.fillStyle = 'rgba(20,13,14,0.9)'; tc.beginPath();\n      var a0 = pt(0.72",
     "      var wt = hex(P.linerTone || '#140d0e'); tc.fillStyle = 'rgba(' + wt[0] + ',' + wt[1] + ',' + wt[2] + ',0.9)'; tc.beginPath();\n      var a0 = pt(0.72"),
    # the lashes' layer rebuilds when the liner's colour changes
    ("['lashUp', 'lashLen', 'lashLean', 'lashCurl', 'lashFuzz', 'lashThick', 'lashLow', 'liner', 'wing', 'seed']",
     "['lashUp', 'lashLen', 'lashLean', 'lashCurl', 'lashFuzz', 'lashThick', 'lashLow', 'liner', 'linerTone', 'wing', 'seed']"),
    # a face's liner colour goes back to the default with the rest of its make-up
    ("'shadow', 'shadowStrength', 'shadowInner', 'smoke', 'lower', 'liner', 'wing', 'lashLen', 'lashUp',\n",
     "'shadow', 'shadowStrength', 'shadowInner', 'smoke', 'lower', 'liner', 'linerTone', 'wing', 'lashLen', 'lashUp',\n"),
    # the Ranger's make-up
    ("flushAmt: 1.0, blush: 0.25, blushTone: '#e08a8c', shadowStrength: 0.55, liner: 0.2, lipGloss: 0.35 },",
     "flushAmt: 1.0, blush: 0.25, blushTone: '#e08a8c', shadowStrength: 0.55, liner: 0.2, lipGloss: 0.35,\n"
     "    // (her eyes, defined for her colouring: a mulberry-brown shadow opposite her green iris and beside her copper hair, a plum-brown\n"
     "    // liner tapering out to a short wing past the outer corner, as for a round, open eye: apply-eye-makeup.py)\n"
     "    shadow: '#6e4652', shadowStrength: 1.15, liner: 0.75, linerTone: '#3a2228', wing: 0.15 },"),
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
