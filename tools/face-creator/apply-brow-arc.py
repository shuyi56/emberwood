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

DEFAULTS = {'GULL_P': 0.6, 'GULL_A': 3.0, 'GULL_D': 0.0, 'WG_DN': 0.0, 'WG_FR': 0.5, 'AG_BACK': 3.0, 'LB_WR': 0.65, 'LB_WM': 1.0, 'LB_SL': 0.75, 'RIDGE_K': 0.8, 'RIM_GAP': 2.5, 'MED_X': 8.0, 'MED_P': 2.0, 'CH_RF': 4.0, 'CH_K': 0.7, 'CH_WLO': 1.0, 'CH_FA': 0.0, 'CH_FV': 2.0, 'CH_FJ': 6.0, 'ROOF_S': 0.0, 'ROOF_W': 6.0, 'ROOF_K': 3.0, 'PN_D': -1.5, 'PN_X0': 11.0, 'PN_X1': 24.0, 'PN_ZT': 0.47, 'W4X': {'blg': 5.0, 'blc': 0.45, 'blw': 0.6, 'ctp': 0.3, 'chf': 0.9, 'chb': 1.4}, 'GR_D': 1.0, 'GR_W': 1.6, 'GR_OFF': 1.6, 'BASE_K': 3.0, 'LEAN': 0.4, 'WB_K': 0.35, 'WB_R': 0.5, 'CH_W': 1.0, 'MALE_W': 0.15, 'FLARE': 0.0, 'LO_B': 15.0, 'CARVE': 3.6, 'CARVE_UP': 45.0,
            'TOP': [25.8, 26.9, 29.6, 30.9, 29.4, 26.4, 21.8], 'THICK': [8.4, 9.6, 0.5, 1.4],
            'VALLEY': 2.0, 'VALLEY_W': 3.5, 'VALLEY_TOP': 26.0, 'KF_MID': 4.0, 'KF_TOPF': 0.4, 'TIP_KBALL': 0.5, 'TIP_KJOIN': 3.5, 'LO_W': 0.7, 'HEAD_ROUND': 0.18, 'MALE_THICK': 0.15,
            'ROOTS': 2000}

MARK = '/* brow arc (apply-brow-arc.py) */'

BROW_FN = r'''%(mark)s
var BROW_ARC = { t: [0, 0.03, 0.25, 0.47, 0.63, 0.8, 1], top: %(top)s, thick: %(thick)s };      // (mm over the inner corners' line: the upper margin Park 2017's; the height eased out along the brow, apply-brow-arc.py)
/* the brow's height (mm) along it: full through the head and body, then easing out on a cosine to a fine tail with no
   slope left at the end, so it thins softly instead of ending in a blunt rod (John: "soft easeout in thickness") */
function browThickMM(t) { var T = BROW_ARC.thick; return (T[2] + (T[1] - T[2]) * (0.5 + 0.5 * Math.cos(Math.PI * Math.pow(t, T[3])))) * lerp(T[0] / T[1], 1, sm(0, 0.08, t)); }      // (T: the head's height, the body's, the tail's, the ease's power)
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
  var top = browTab(BROW_ARC.top, t), bot = top - browThickMM(t), c0 = BROW_ARC.top[0] - 0.5 * browThickMM(0);
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
/* the bridge chiselled (apply-brow-arc.py): a front plane, barely rounded (radius %(chrf)g crest half-widths), turning into the
   side wall over a %(chk)g mm shoulder, instead of one hyperbola from the middle to the wall. The shoulder is the dorsal
   line, the light's edge on the nose. Depth back from the crest at ax out (model units) */
function browChisel(ax, w, sl, mm) {
  var Rf = %(chrf)g * w, k = %(chk)g * mm, v = ax - w, fs = w / Rf, front = ax <= w ? ax * ax / (2 * Rf) : w * w / (2 * Rf) + fs * v;
  var sr = 0.5 * (v + Math.sqrt(v * v + k * k)) - 0.5 * k, fa = %(chfa)g, fv = %(chfv)g * mm, fj = %(chfj)g * mm;
  // (the side wall in a concave flare when fa > 0: steeper just after the dorsal line, flatter out toward the cheek, at
  // the same depth fj out, so its base stays put. plane_breaks.py: the photo's side is darkest just past the corner)
  if (fa > 0) sr = (1 - fa * fv * (1 - Math.exp(-fj / fv)) / fj) * sr + fa * fv * (1 - Math.exp(-sr / fv));
  // (the roof: the nasal bones meet at 105-112 degrees (pyramidal angle, CT: Alharethy 2017, Moon 2013), so the wall first
  // falls at about 36 degrees for the bones' width, then steepens down the maxilla's frontal process)
  var rs = %(roofs)g, rw = %(roofw)g * mm, rk = %(roofk)g * mm;
  if (rs > 0 && sl > rs) { var v2 = v - rw; return front + Math.max(0, rs - fs) * sr + (sl - rs) * (0.5 * (v2 + Math.sqrt(v2 * v2 + rk * rk)) - 0.5 * (-rw + Math.sqrt(rw * rw + rk * rk))); }
  return front + Math.max(0, sl - fs) * sr;
}
/* the socket carved under the brow's arc (S.carve, the skin field's span): a hollow inside the orbit's rim, deepest in its
   upper inner quarter where the socket's wall runs into the nose's side. xl, zl from the pupil (model units, x lateral);
   returns depth (y back) */
function socketCarve(S) {
  var mm = S.mm, sx = P.sex || 0, aMed = (S.cx - %(medx)g * mm) / 1.05, aLat = 20 * mm, bLo = %(lob)g * mm, gap = %(gap)g * (1 - sx) * mm;
  return function (xl, zl) {
    var u = xl / RX, a = browArcMM(u), bUp = Math.max(8 * mm, (a[0] - a[1]) * mm + (P.browHeight - 0.06 * sx) - gap - (S.cz - (0.525 * HH + EYE_LIFT)));
    var ax = xl < 0 ? aMed : aLat, bz = zl > 0 ? bUp : bLo, pw = xl < 0 && zl > 0 ? %(medp)g : 2, rho = Math.pow(Math.pow(Math.abs(xl / ax), pw) + Math.pow(Math.abs(zl / bz), pw), 1 / pw);      // (squarer in the upper inner quarter: the rim runs on under the brow's head, then turns down the nose's side)
    // (the valley beside the bridge, from under the brow's head down to the inner corner, where the nose's side takes
    // the shade on: the photo's shade line runs there, 10.9 mm out at the brow's head, 8.3 at the eyes; ours ran out
    // into the socket instead: shadow_trace.py)
    var xm = S.cx + xl, zm = zl / mm, xc = (8.3 + 0.087 * zm + 0.003 * zm * zm) * mm;
    var vl = %(valley)g * mm * Math.exp(-Math.pow((xm - xc) / (%(valw)g * mm), 2)) * osm(%(vtop)g, %(vtop)g - 6, zm) * osm(-16, -8, zm);
    if (rho > 1.12) return vl;
    var th = Math.atan2(zl / bz, -xl / ax) * 180 / Math.PI, w = Math.max(0.25 + 0.75 * Math.exp(-Math.pow((th - %(up)g) / 50, 2)), %(low)g * Math.exp(-Math.pow((th + 35) / 35, 2)));      // (and in its lower inner quarter, down the nose's side past the inner corner)
    return %(carve)g * mm * P.sockets * w * osm(1.1, 0.75, rho) + vl;
  };
}
'''

# 4. the brow's edges (old line, new line). A brow is not a band with two clean margins: its hairs thin out toward the
# margins, the upper margin most (the hairs there lie over it and the skin shows through), single hairs stray past it,
# the margins wander, and the powder (the skin's own shade under the hairs) is densest along the lower body and patchy.
EDGES = [
    # the outline: each margin wanders along the brow (low-frequency noise on the side coordinate), the upper one twice
    # as far, and both edges ease over a wider band
    ("var shape = sm(-1.15, -0.8, sd) * sm(1.2, 0.6, sd) * sm(-0.68, -0.35, ub - 0.18 * sd * sd) *",
     "var eT = 0.3 * fbm2(u * 4.5, 3 + s, 131, 3), eB = 0.13 * fbm2(u * 6, 9 + s, 137, 2), shape = sm(-1.3, -0.65, sd - eB) * sm(1.3, 0.25, sd + eT) * sm(-0.68, -0.35, ub - 0.18 * sd * sd) *"),
    # the powder: densest along the lower body, thinning toward the top, patchy
    ("var powder = shape * (0.5 + 0.15 * warpedFbm(u * 3, dz * 3, 111, 2)) *",
     "var powder = shape * lerp(1, 0.3, sm(-0.3, 1.0, sd)) * (0.42 + 0.28 * warpedFbm(u * 7, dz * 7, 113, 2)) *"),
    ("d[o + 3] = clamp(powder * 255 * 0.9, 0, 255);", "d[o + 3] = clamp(powder * 255 * 0.8, 0, 255);"),
    # the roots: spread past the margins, fewer toward them (the top fades out further), so the edge is hairs, not a line
    ("dz = cr[0] + (r() * 2 - 1) * cr[1] * 1.05, ub = u * Kb, side0 = (dz - cr[0]) / Math.max(cr[1], 1e-3);",
     "dz = cr[0] + (r() * 2 - 1) * cr[1] * 1.4, ub = u * Kb, side0 = (dz - cr[0]) / Math.max(cr[1], 1e-3) + 0.3 * fbm2(u * 4.5, 3 + s, 131, 3);"),
    ("keep *= 1 - 0.5 * sm(0.75 * P.browLen, 1.08 * P.browLen, ub);\n",
     "keep *= 1 - 0.5 * sm(0.75 * P.browLen, 1.08 * P.browLen, ub); keep *= side0 > 0 ? 1 - 0.93 * sm(0.35, 1.4, side0) : 1 - 0.85 * sm(-0.55, -1.4, side0);\n"),
    # the hairs longer, those of the upper half longest (they lie over the margin), each with its own lean
    ("var length = (0.11 + 0.05 * sm(-0.6, 0.3, ub) - 0.05 * sm(0.65, 1.08, ub / P.browLen))",
     "var length = (0.14 + 0.05 * sm(-0.6, 0.3, ub) - 0.06 * sm(0.65, 1.08, ub / P.browLen)) * (1 + 0.35 * sm(0, 1, side))"),
    ("jit = 0.18 * angN + 0.08 * (r() - 0.5);", "jit = 0.18 * angN + 0.24 * (r() - 0.5);"),
]


def patch(src, P):
    # 1. the brow's shape (a file already patched: from the mark to paintBrows)
    if MARK in src:
        old = re.search(re.escape(MARK) + r'\n.*?\n(?=function paintBrows)', src, flags=re.S)
    else:
        old = re.search(r'function browCentre\(u\) \{\n.*?\n\}\n(?=function paintBrows)', src, flags=re.S)
    assert old, 'browCentre not found'
    blk = BROW_FN % {'mark': MARK, 'top': json.dumps(P['TOP']), 'thick': json.dumps(P['THICK']), 'ridgek': P['RIDGE_K'],
                     'medx': P['MED_X'], 'lob': P['LO_B'], 'gap': P['RIM_GAP'], 'up': P['CARVE_UP'], 'carve': P['CARVE'], 'medp': P['MED_P'], 'low': P['LO_W'], 'mth': P['MALE_THICK'], 'valley': P['VALLEY'], 'vtop': P['VALLEY_TOP'], 'valw': P['VALLEY_W'], 'chrf': P['CH_RF'], 'chk': P['CH_K'], 'chfa': P['CH_FA'], 'chfv': P['CH_FV'], 'chfj': P['CH_FJ'], 'roofs': P['ROOF_S'], 'roofw': P['ROOF_W'], 'roofk': P['ROOF_K']}
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
    # 5. the nose chiselled, and its dorsal lines run up into the brows' heads (the brow-tip aesthetic line)
    f = '  var zRad = zN, yRad = yN - NL.radix * np * mm, zSup = zPrn + 0.14 * L, ySup = yPrn + 1.3 * mm;'
    assert f in src, 'radix line not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': head \*/[^\n]*', '', src)
    src = src.replace(f, f + '\n  %s: head */ var uHB = -0.6 * 1.1 / RX, aHB = browArcMM(uHB), xHB = P.eyeSpacing * HH + uHB * RX, zHB = 0.525 * HH + EYE_LIFT + (aHB[0] - aHB[1]) * mm + P.browHeight - 0.06 * sx;      // (the brow\'s head: its lower inner corner, where the dorsal line runs into it)' % MARK[:-3], 1)
    g = '    w = lerp(lerp(wLo, wMid, sm(0, 0.5, tc)), NL.ridge[0] * nw * mm, sm(0.5, 1, tc));'
    assert g in src, 'crest width line not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': flare \*/[^\n]*', '', src)
    src = src.replace(g, g + '\n    %s: flare */ w *= %g * (1 - %g * sx); var qF = clamp((z - zRad) / Math.max(1e-6, zHB - zRad), 0, 1), wF = %g * Math.max(0, xHB - w) * (1 - Math.sqrt(Math.max(0, 1 - qF * qF))); w += wF;      // (above the radix the dorsal lines part and curve out into the brows\' heads, a quarter ellipse: upright at the radix, level at the brow)' % (MARK[:-3], P['CH_W'], P['MALE_W'], P['FLARE']), 1)
    h0 = 'var dD = (yl + sl * (Math.sqrt(w * w + ax * ax) - w) + 0.09 * bk * bk / mm - y) / Math.sqrt(1 + sl * sl);'
    h1 = 'var dD = (yl + browChisel(ax, %s, sl, mm) + 0.09' % ('w' if P['CH_WLO'] == 1 else 'w * lerp(%g, 1, sm(0.25, 0.85, tc))' % P['CH_WLO']) + ' * Math.pow(Math.max(0, bk - wF), 2) / mm - y) / Math.sqrt(1 + sl * sl);      /* brow arc (apply-brow-arc.py): the chisel */'
    src = re.sub(r'var dD = \(yl \+ browChisel\([^\n]*the chisel \*/', lambda m: h0, src)
    if h0 in src:
        src = src.replace(h0, h1, 1)
    assert h1 in src, 'bridge section line not found'
    # 6. the lower nose chiselled to match: the wedge's front, sides and the bridge meet over tighter blends, so the dorsal
    # lines run on into the tip-defining points and the lobule reads as planes instead of a ball
    m6 = re.search(r'\n  var W4 = \{"off": 1\.0[^\n]*', src)
    assert m6, 'W4 line not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': tip \*/[^\n]*', '', src)
    m6 = re.search(r'\n  var W4 = \{"off": 1\.0[^\n]*', src)
    src = src[:m6.end()] + '\n  %s: tip */ Object.assign(W4, %s);      // (the tip in planes, carrying on the bridge\'s chisel: apply-brow-arc.py)' % (MARK[:-3], json.dumps(dict({'kball': P['TIP_KBALL'], 'kjoin': P['TIP_KJOIN'], 'kbase': P['BASE_K'], 'lean': P['LEAN']}, **P['W4X']))) + src[m6.end():]
    # 7. the nose's side met by the cheek along a crisp junction (the blend of the nose into the face over the middle of
    # the nose, 5 mm, to KF_MID): the side plane ends in a line, the nasofacial groove, instead of a smear
    src = re.sub(r'lerp\(3\.2 \* mm, [0-9.]+ \* mm, sm\(zSn \+ 4 \* mm, zSn \+ 12 \* mm, z\)\)', 'lerp(3.2 * mm, %g * mm, sm(zSn + 4 * mm, zSn + 12 * mm, z))' % P['KF_MID'], src, count=1)
    # 7b. the cheek side of the nose: soft down the bridge, crisp again up by the inner corners (plane_breaks.py: the
    # photo's shade climbs onto the cheek over 8-12 mm below the eyes, over 2-3 mm beside the inner corners)
    src = re.sub(r'\* lerp\(1, [0-9.]+, sm\(zSn \+ 0\.5 \* L, zSn \+ 0\.85 \* L, z\)\)', '* lerp(1, %g, sm(zSn + 0.5 * L, zSn + 0.85 * L, z))' % P['KF_TOPF'], src, count=1)
    # 8. the cheek beside the nose tilted toward it, so the shade on the nose's side climbs back onto the cheek over
    # 8-12 mm as in the photo instead of stopping in a crease (plane_breaks.py: rise); 0 depth leaves the face as it was
    r8 = '    /* brow arc (apply-brow-arc.py): ridge */ d += browRidgeArc(x * s, z, es);'
    assert r8 in src, 'ridge line not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': paranasal \*/[^\n]*', '', src)
    if P['PN_D'] != 0:
        i8 = src.index(r8); j8 = src.index('\n', i8)
        src = src[:j8] + '\n    %s: paranasal */ if (x * s > 0) d -= %g / 219 * sm(%g / 219, %g / 219, x * s) * sm(4 / 219, 8 / 219, x * s) * sm(0.37, 0.40, z) * sm(%g, %g, z);      // (the cheek beside the nose, toward it (+) or brought forward (-): apply-brow-arc.py)' % (MARK[:-3], P['PN_D'], P['PN_X1'], P['PN_X0'], P['PN_ZT'], P['PN_ZT'] - 0.03) + src[j8:]
    # 9. the wing's underside curved in toward the tip: the crescent is widest at its bottom edge and ends flat, so the
    # nose's outline holds its full width and then drops (John: "does not curve in at the base"); the photo's line comes
    # in from its widest point over 0.5-0.7 pupil distances. WB_K narrows the wing's foot, which rounds the bottom in.
    gv0 = 'var gv = Math.pow(Math.max(0, 1 - vc * vc), W4.cwp)'
    src = re.sub(r'var gv = Math\.pow\(Math\.max\(0, 1 - vc \* vc\), W4\.cwp\)[^;]*;', gv0 + ('' if P['WB_K'] == 0 else ' * lerp(%g, 1, sm(0, %g, vc))' % (1 - P['WB_K'], P['WB_R'])) + ';', src, count=1)
    # 10. the nasofacial groove: a trough just outside the nose's side, along the photo's shade line (x 8.5 mm at the
    # eyes' line, 7.2 at 0.25 pupil distances below, 11.2 at 0.5, quadratic through them, plus GR_OFF), GR_D deep. Our
    # side flared smoothly into the cheek, so the baked cavity shade darkened the junction 4-7% where the photo's groove
    # is 12-23% darker (apply-brow-arc.py, plane_breaks.py)
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': groove \*/[^\n]*', '', src)
    if P['GR_D'] > 0:
        r10 = '    /* brow arc (apply-brow-arc.py): ridge */ d += browRidgeArc(x * s, z, es);'
        i10 = src.index(r10); j10 = src.index('\n', i10)
        gl = ('\n    %s: groove */ if (x * s > 0) { var gy = (0.525 - z) * 219 / 63, gx = (8.5 - 15.8 * gy + 42.4 * gy * gy + %g) / 219; '
              'd -= %g / 219 * Math.exp(-Math.pow((x * s - gx) / (%g / 219), 2)) * sm(-0.05, 0.05, gy) * sm(0.58, 0.48, gy); }      // (the nasofacial groove along the photo\'s line: apply-brow-arc.py)'
              % (MARK[:-3], P['GR_OFF'], P['GR_D'], P['GR_W']))
        src = src[:j10] + gl + src[j10:]
    # 11. the lower bridge flowing into the ball (John 2026-11-25: "The bridge needs to flow down at the nostrils, right
    # now it looks flat on ours"). plane_breaks.py: the photo's lit front widens steadily down the nose (half-width 0.034
    # pupil distances at y -0.05 to 0.060 at y 0.45) while ours stays 0.025-0.034 and holds a straight slab. LB_WR sets the
    # crest's half-width at the ball (share of the wings' width), LB_WM lets the middle be that wide too, LB_SL eases the
    # side walls' slope toward the ball.
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': lowbridge \*/[^\n]*', '', src)
    k11 = 'cHs = CT.Hs * L;'
    src = src.replace(k11, k11 + '\n  %s: lowbridge */ CT.wr = %g; CT.wm = %g;' % (MARK[:-3], P['LB_WR'], P['LB_WM']), 1)
    src = re.sub(r'var sl = lerp\(lerp\(NL\.side\[2\], NL\.side\[1\], sm\(0, 0\.5, tc\)\), NL\.side\[0\], sm\(0\.5, 1, tc\)\) / \(0\.85 \+ 0\.15 \* nw\)(?: \* lerp\([^;]*\))?;',
                 lambda m: 'var sl = lerp(lerp(NL.side[2], NL.side[1], sm(0, 0.5, tc)), NL.side[0], sm(0.5, 1, tc)) / (0.85 + 0.15 * nw)' + ('' if P['LB_SL'] == 1 else ' * lerp(%g, 1, sm(0, 0.6, tc))' % P['LB_SL']) + ';', src, count=1)
    # 12. the alar groove where the bridge's side meets the wing (John 2026-11-25: "read up more on how to sculpt where
    # bridge meets wings"). Bloom, Ransom & Miller 2011: "The crescentic alar groove ... frames the ala and separates this
    # convex structure from the surrounding cosmetic subunits", "The alar groove deepens as it extends posteriorly", "The
    # lower lateral cartilage does not cross the alar groove"; Bloom & Constantinides 2011: the cartilage's lower edge is
    # "about a centimeter superior to the nostril rim". AG_BACK multiplies the groove's depth at the wing's back, easing to
    # 1 toward the tip (Proko, Keith: the crease is "much sharper at the bottom of the nose and then softens a lot as it
    # moves up and curls around the nostril").
    src = re.sub(r'var gr = W4\.cgr \* (?:lerp\([^;]*?sm\(0, 0\.7, tc\)\) \* )?mm \* Math\.exp',
                 'var gr = W4.cgr * ' + ('' if P['AG_BACK'] == 1 else 'lerp(%g, 1, sm(0, 0.7, tc)) * ' % P['AG_BACK']) + 'mm * Math.exp', src, count=1)
    # 13. the wing's foot arches down (John 2026-11-25: "Wings need to arch down they are floating at the moment"). The wing's
    # lower rim ran level with the nose's floor from the front to the back, so its back end stood off the cheek on a shelf.
    # WG_DN lowers the rim at the back by that many mm, easing to nothing a third of the way along, so the rim is an arch:
    # down where the wing is fixed to the cheek, up over the nostril (Proko: the wing "hooks around and under").
    src = re.sub(r'(var ef = \(f - wC\) / wRf, uBw = uB \+ W4\.arch[^;]*?)(?: - [0-9.]+ \* mm \* sm\([^;]*?\))?;',
                 lambda m: m.group(1) + ('' if P['WG_DN'] == 0 else ' - %g * mm * sm(%g, -0.8, ef)' % (P['WG_DN'], P['WG_FR'])) + ';', src, count=1)
    # 14. the rim as a gull in flight (John 2026-11-25: his lines on the clay front, then "Rethink and research on getting the
    # arches"). Hiding the parts one at a time showed the line under the nose is the wedge's flat floor seen edge on, not the
    # wing, so the floor itself is given the outline: lowest at the columella, rising GULL_A mm over each nostril, and
    # coming down GULL_D mm at the alar base. Brito, Avashia & Rohrich 2020: "The alar rim's ideal shape resembles a gull
    # in flight", rim and columella each 1-2 mm from the nostril's long axis; Bloom & Constantinides 2011: the alar base
    # ends about 2 mm below the columella-lip junction (Guyuron) to 2-3 mm above it (Crumley).
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': gull \*/[^\n]*', '', src)
    if P['GULL_A'] or P['GULL_D']:
        k14 = 'var uB = W4.off * mm + f * tanB;'
        assert k14 in src
        src = src.replace(k14, k14 + '\n    %s: gull */ var gC = W4.colw * alarHalf, gT = clamp((ax - gC) / (alarHalf - gC), 0, 1); uB += (%g * Math.sin(Math.PI * Math.pow(gT, %g)) - %g * sm(0.55, 1, gT)) * mm;' % (MARK[:-3], P['GULL_A'], P['GULL_P'], P['GULL_D']), 1)
    # 4. the edges: broken instead of a rod's (John: "Improve the edges of the brows too rod like")
    for a, b in EDGES:
        if a in src:
            src = src.replace(a, b, 1)
        assert b in src, 'brow edge line not found: ' + a[:60]
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
