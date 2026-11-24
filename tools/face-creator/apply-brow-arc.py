"""Brow arc: the brows thicker and curved round the eye, the brow ridge and the socket shaped to them, and the nose's
side carved into the socket along the eye's curve (John 2026-11-24: "Give the brows more defined curves and shape the
face accordingly. Thicker more curved that follow the contours of the eyes. Carve the nose by following the contours of
eye curves so it looks less conic like it does now").

    python3 tools/face-creator/apply-brow-arc.py <face-creator.html or main.js> [--out FILE] [--params JSON]

Measured first: the painted brow sat 5 mm too high and too thin. Its middle line ran 26.8 mm over the corner line at
the head and 32 mm at the peak, so its upper margin was 30 and 35 mm up. Park, Yun and Son 2017 (Arch Plast Surg 44:65,
20 women aged 20-30, gaze ahead) measure the upper margin over the inner corners' line at 26.9 mm over the inner corner,
30.9 over the lateral limbus (always the peak), 28.2 over the outer corner and 21.8 at the tail's end. The brow was 6 mm
tall at the head and 8 at its widest; Rajabi 2017 (Clin Ophthalmol 11:1333, 60 women) measures the medial brow's
height at 7.5 +- 1.9 mm, up to 13. Its peak lay past the outer corner, its tail fell only 2 mm. And inside each
socket's block the skin was a Coons patch spanned over the plain face (orbitSkinField), so between the brow, the eye
and the nose there was one flat plane: the nose's straight side wall met it along a straight line, the cone.

The changes, each from a measure:
1. The brow's margins from tables (BROW_ARC: t along the brow, 0 its head over the inner corner, 1 its tail on the
   line from the wing through the outer corner, Yalcinkaya 2016): the upper margin Park's, the lower margin parallel to
   the eye's arc, 10 mm under the upper at the head (the thick end), 8.7 at the peak, tapering to the tail. browArch
   scales the rise and fall about the head's height as before, browHeight and browThick as before, his brow flatter
   and lower as before, and heavier by MALE_THICK (it was 0.35 on the old 6 mm brow: on the new 10 mm one that made
   the Knight's a 15 mm bar). The brow's head rounded at its corners (HEAD_ROUND) instead of cut square.
2. The brow ridge (faceRelief) follows that arc: its crest moved up and down with the brow's middle line (RIDGE_K of
   it), so the bone under the brow curves as the brow does instead of running level.
3. The socket carved under the arc (S.carve, added to the skin field's span): a hollow inside the orbit's rim, its rim
   on top RIM_GAP mm under the brow's lower margin (hers: "several millimetres above the rim"; his on it, Yalcinkaya),
   medially MED_X mm from the midline, below LO_B mm under the pupil. Deepest (CARVE mm, times the socket slider) in the
   upper inner quarter between the brow's head and the root of the nose, where the socket's medial wall runs into the
   nose's side (Proko's planes of the head: the eye socket "angled up from the outside to the keystone"; Bridgman's
   "nose bridge to eye" plane), lighter on top, under the eye and out at the side. The nose is added onto the span
   after it, so the nose's side wall runs on down into that hollow and its foot follows the eye's curve instead of a
   straight line.

Idempotent: a file already patched is patched again from its marked lines.
"""
import argparse
import json
import re

DEFAULTS = {'RIDGE_K': 0.8, 'RIM_GAP': 2.5, 'MED_X': 8.0, 'LO_B': 15.0, 'CARVE': 3.0, 'CARVE_UP': 45.0,
            'TOP': [25.8, 26.9, 29.6, 30.9, 29.4, 26.4, 21.8], 'BOT': [17.6, 17.2, 20.6, 22.2, 22.0, 21.0, 20.2],
            'LO_W': 0.7, 'HEAD_ROUND': 0.18, 'MALE_THICK': 0.15,
            'ROOTS': 2000}

MARK = '/* brow arc (apply-brow-arc.py) */'

BROW_FN = r'''%(mark)s
var BROW_ARC = { t: [0, 0.03, 0.25, 0.47, 0.63, 0.8, 1], top: %(top)s, bot: %(bot)s };      // (mm over the inner corners' line: the upper margin Park 2017's, the lower parallel to the eye's arc; apply-brow-arc.py)
function browTab(tab, t) {      // (a cubic through the table's points)
  var T = BROW_ARC.t, n = T.length, i = 0; t = clamp(t, 0, 1);
  while (i < n - 2 && t > T[i + 1]) i++;
  function m(j) { return j <= 0 ? (tab[1] - tab[0]) / (T[1] - T[0]) : j >= n - 1 ? (tab[n - 1] - tab[n - 2]) / (T[n - 1] - T[n - 2]) : (tab[j + 1] - tab[j - 1]) / (T[j + 1] - T[j - 1]); }
  var h = T[i + 1] - T[i], s = (t - T[i]) / h, s2 = s * s, s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * tab[i] + (s3 - 2 * s2 + s) * h * m(i) + (-2 * s3 + 3 * s2) * tab[i + 1] + (s3 - s2) * h * m(i + 1);
}
/* the brow's middle line and half-height at u (the eye's units out from the pupil), in mm over the corner line, the arch
   applied, before browHeight and browThick */
function browArcMM(u) {
  var Kb = RX / 1.1, us0 = -0.6, us1 = 1.1 * P.browLen, t = (clamp(u * Kb, us0, us1) - us0) / (us1 - us0), sxb = P.sex || 0;
  var top = browTab(BROW_ARC.top, t), bot = browTab(BROW_ARC.bot, t), c0 = 0.5 * (BROW_ARC.top[0] + BROW_ARC.bot[0]);
  return [c0 + P.browArch * (1 - 0.6 * sxb) * (0.5 * (top + bot) - c0), 0.5 * (top - bot)];
}
function browCentre(u) {
  var sxb = P.sex || 0, mmu = HH / 219, a = browArcMM(u);
  var zc = a[0] * mmu + P.browHeight - 0.06 * sxb, r = a[1] * mmu * P.browThick * (1 + %(mth)g * sxb);      // (sex: his brows sit lower on the ridge, flatter, heavier)
  return [zc - u * RX * P.tilt, r];
}
/* the brow ridge's crest raised and lowered with the brow's middle line (head heights; the eye's side, x > 0) */
function browRidgeArc(x, z, es) {
  var u = (x - P.eyeSpacing) * HH / RX, dz = %(ridgek)g * (browArcMM(u)[0] - 22.0) / 219;
  return gauss(x, z, 0.15 * es, 0.62 + dz, x < 0.15 * es ? 0.075 : 0.12, 0.052, 0.03 * P.browRidge) - gauss(x, z, 0.15 * es, 0.62, x < 0.15 * es ? 0.075 : 0.12, 0.052, 0.03 * P.browRidge);
}
/* the socket carved under the brow's arc (S.carve, the skin field's span): a hollow inside the orbit's rim, deepest in its
   upper inner quarter where the socket's wall runs into the nose's side. xl, zl from the pupil (model units, x lateral);
   returns depth (y back) */
function socketCarve(S) {
  var mm = S.mm, sx = P.sex || 0, aMed = (S.cx - %(medx)g * mm) / 1.05, aLat = 20 * mm, bLo = %(lob)g * mm, gap = %(gap)g * (1 - sx) * mm;
  return function (xl, zl) {
    var u = xl / RX, a = browArcMM(u), bUp = Math.max(8 * mm, (a[0] - a[1]) * mm + (P.browHeight - 0.06 * sx) - gap - (S.cz - (0.525 * HH + EYE_LIFT)));
    var ax = xl < 0 ? aMed : aLat, bz = zl > 0 ? bUp : bLo, rho = Math.sqrt(xl * xl / (ax * ax) + zl * zl / (bz * bz));
    if (rho > 1.12) return 0;
    var th = Math.atan2(zl / bz, -xl / ax) * 180 / Math.PI, w = Math.max(0.25 + 0.75 * Math.exp(-Math.pow((th - %(up)g) / 50, 2)), %(low)g * Math.exp(-Math.pow((th + 35) / 35, 2)));      // (and in its lower inner quarter, down the nose's side past the inner corner)
    return %(carve)g * mm * P.sockets * w * osm(1.1, 0.75, rho);
  };
}
'''


def patch(src, P):
    # 1. the brow's shape (a file already patched: from the mark to paintBrows)
    if MARK in src:
        old = re.search(re.escape(MARK) + r'\n.*?\n(?=function paintBrows)', src, flags=re.S)
    else:
        old = re.search(r'function browCentre\(u\) \{\n.*?\n\}\n(?=function paintBrows)', src, flags=re.S)
    assert old, 'browCentre not found'
    blk = BROW_FN % {'mark': MARK, 'top': json.dumps(P['TOP']), 'bot': json.dumps(P['BOT']), 'ridgek': P['RIDGE_K'],
                     'medx': P['MED_X'], 'lob': P['LO_B'], 'gap': P['RIM_GAP'], 'up': P['CARVE_UP'], 'carve': P['CARVE'], 'low': P['LO_W'], 'mth': P['MALE_THICK']}
    src = src[:old.start()] + blk.lstrip('\n') + src[old.end():]
    src = re.sub(r'roots\.length < [0-9]+ \* P\.browDensity', 'roots.length < %d * P.browDensity' % P['ROOTS'], src, count=1)
    # (the head of the brow rounded at its top and bottom corners instead of cut square)
    src = re.sub(r'sm\(-1\.15, -0\.8, sd\) \* sm\(1\.2, 0\.6, sd\) \* sm\(-0\.68, -0\.35, ub[^)]*\) \*',
                 'sm(-1.15, -0.8, sd) * sm(1.2, 0.6, sd) * sm(-0.68, -0.35, ub - %g * sd * sd) *' % P['HEAD_ROUND'], src, count=1)
    # 2. the brow ridge along the arc (a line after the ridge's own, so apply-face-flow.py still finds it)
    b = 'd += gauss(x, z, s * 0.15 * es, 0.62, x * s < 0.15 * es ? 0.075 : 0.12, 0.052, 0.03 * P.browRidge);'
    assert b in src, 'brow ridge line not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': ridge \*/[^\n]*', '', src)
    src = src.replace(b, b + '\n    %s: ridge */ d += browRidgeArc(x * s, z, es);      // (the ridge curves with the brow: apply-brow-arc.py)' % MARK[:-3], 1)
    # 3. the socket carved into the span, the nose added after it
    c = 'Hh[k] = Hh[k] + (span[k] - Hh[k]) * osm(0, 6 * e, edge[k]) + (Hn ? Hn[k] : 0);'
    assert c in src, 'span line not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': carve \*/[^\n]*', '', src)
    src = src.replace(c, c + '\n  %s: carve */ if (S.carve) for (k = 0; k < N; k++) Hh[k] += S.carve(X[k] - S.cx, Z[k] - S.cz) * osm(0, 6 * e, edge[k]);      // (the socket under the brow\'s arc, the nose\'s side running into it: apply-brow-arc.py)' % MARK[:-3], 1)
    d = re.search(r'\n  var px = P\.eyeSpacing \* HH, S = orbitSpec\(HH[^\n]*', src)
    assert d, 'orbitSpec call not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': spec \*/[^\n]*', '', src)
    d = re.search(r'\n  var px = P\.eyeSpacing \* HH, S = orbitSpec\(HH[^\n]*', src)
    src = src[:d.end()] + '\n  %s: spec */ S.carve = socketCarve(S);' % MARK[:-3] + src[d.end():]
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--out')
    ap.add_argument('--params', default='{}')
    A = ap.parse_args()
    P = dict(DEFAULTS, **json.loads(A.params))
    src = open(A.file).read()
    out = patch(src, P)
    open(A.out or A.file, 'w').write(out)
    print('patched', A.out or A.file, json.dumps(P))


if __name__ == '__main__':
    main()
