"""Face flow: the brow runs into the nose, and the nose's side walls run out into the cheeks (John 2026-11-24: "total
facial flow ... from eyebrow down to the mouth", with his reference photo's lines in yellow).

    python3 tools/face-creator/apply-face-flow.py <face-creator.html or main.js> [--out FILE] [--params JSON]

Measured first (tools/gauges/face/face_flow.py against his photo, every preset face, same front orthographic camera,
unit = the pupils' distance): on all five faces the bridge was a fin. Its lit front edge (the dorsal aesthetic line)
was 0.022-0.026 unit from the midline at the waist against the photo's 0.057, half the width of the tip-defining points
(Rohrich's 10-7-5: about equal; the photo 0.92); the side wall met the cheek at 0.52-0.63 of the wing's half-width at
mid-bridge against the photo's 0.75 and the published 0.7-0.8 (bony base / alar base: Rohrich; plasticsurgerykey 80 %);
and the brow's plane break ended 0.22-0.25 unit (about 14 mm) away from the top of the nose's line, where the photo's
line runs on without a break (Park 2011: "subtle, unbroken, and fluid", "slightly wider at the radix, narrow at the
middle third and then widen at the tip"). So the cause is the nose solid's cross-section and the missing medial orbital
rim, not the lines drawn on them. The changes, each from a measure:

1. The side walls' slope at the radix and mid-bridge, 2.0 and 2.1 (depth per width) down to SIDE_RADIX and SIDE_MID.
   The radix's depth behind the nasion to the inner corner is about 15 mm (plasticsurgerykey) over a half intercanthal
   width of about 16 mm (31-33 mm: Rohrich), a mean slope of about 0.95 from the midline to the corner; a wall of 2.0 is
   a fin that stands on a flat face. The tip's own slope (side[2], from which the ball's size is derived) is unchanged,
   so the tip, the wings and the nostrils of the nose thread keep their shape.
2. The crest's half-width at the radix (ridge[0]) to RIDGE_RADIX and the cap on the mid-bridge's crest against the ball
   (CT.wm, 0.9) to WM, so the dorsal lines' width matches the tip-defining points' (Rohrich) and the bridge is an
   hourglass, wider at the radix (Park).
3. Where the upper wall turns back into the face (5.5 mm out at the radix) to BK_TOP: with the gentler wall it runs out
   to the inner corner's side instead of being cut off half way (that cut was the step between the brow and the nose).
4. EXPERIMENTAL, OFF (RIM_AMP 0): the medial orbital rim: the brow's lower edge carried in and down past the inner corner into the nose's side wall,
   as the bone is (the supraorbital margin running into the frontal process of the maxilla and the anterior lacrimal
   crest), a soft ridge along a curve through the photo's brow line (0.21, 0.31, 0.38 unit above the pupils at 0.2, 0.3,
   0.4 unit out) down to the bridge's side at the pupils' line. Sculptors cut the sockets "angled up from the outside to
   the keystone" (Proko, planes of the head); Bridgman: on the lid "three planes wedge together: nose bridge to eye;
   brow to cheekbone". RIM_AMP its height, RIM_W its width (head heights).
6. EXPERIMENTAL, OFF (BROW_SZ_LO 0.052, as it was): the brow ridge's fall below its crest (BROW_SZ_LO head heights against 0.052 above): the supraorbital margin is an
   edge under a sloping forehead, so the brow turns under into the socket's roof along a line (the photo's brow line)
   instead of rounding off over 11 mm.
5. Above the radix the bridge sinks into the glabella (FADE_Z mm above the nasion, then FADE_K * rise^2 behind):
   with gentler walls the solid otherwise ran on up to its bounding box and was cut there, a step across the brow.

Rounds (Ranger; face_lines' bony base / alar and dorsal lines / tip-defining points, face_flow's brow-to-nose join gap
in pupil distances): before 0.36, 0.60, 0.227; round 1 (walls 1.1) 0.88, 1.31, 0.216; round 2 (walls 1.3, glabella
fade, rim 0.016) 0.67, 1.02, 0.197; round 3 (walls 1.2, rim 0.02, brow's lower fall 0.032) 0.70, 1.06, 0.194. The
nose's numbers landed in round 2 and 3; the join barely moved and the rim and the brow's fall smeared the skin above
the inner corners in streaks (a render of them alone, on the old nose, shows it is theirs). The socket's carve
(orbitCarve, orbit.js and blender/shared/orbit.py) re-lays the skin from the lids up toward the brow, so a brow edge
put into faceRelief fights it. 4 and 6 stay here, off, for the record; the brow's join belongs in the socket's own
profile (orbitProfile, brow_ahead), the next step, which John decides.

Every value can be overridden with --params '{"SIDE_RADIX": 1.1, ...}' for the rounds; the defaults are the accepted
round. Idempotent: a file already patched is patched again from its marked original lines.
"""
import argparse
import json
import re

DEFAULTS = {'SIDE_RADIX': 1.2, 'SIDE_MID': 1.28, 'RIDGE_RADIX': 4.2, 'RIDGE_MID': 3.4, 'WM': 1.0, 'BK_TOP': 8.0,
            'RIM_AMP': 0.0, 'RIM_W': 0.02, 'FADE_Z': 3.0, 'FADE_K': 0.35, 'BROW_SZ_LO': 0.052, 'KF_TOP': 0.5}

MARK = '/* face flow (apply-face-flow.py) */'


def patch(src, P):
    # 1, 2: the bridge's walls and crest (a line after the nose thread's own overrides, so theirs stay readable)
    a = 'Object.assign(NOSE_LIFE, {"scale": 0.9, "nostril": [1.75, 5.0, 3.4]});'
    assert a in src, 'NOSE_LIFE scale line not found'
    src = re.sub(r'\n' + re.escape(MARK) + r' Object\.assign\(NOSE_LIFE[^\n]*', '', src)
    src = src.replace(a, a + '\n%s Object.assign(NOSE_LIFE, {"ridge": [%g, %g, NOSE_LIFE.ridge[2]], "side": [%g, %g, NOSE_LIFE.side[2]]});'
                      % (MARK, P['RIDGE_RADIX'], P['RIDGE_MID'], P['SIDE_RADIX'], P['SIDE_MID']) +
                      '      // (the bridge an hourglass on a pyramid: face flow, apply-face-flow.py)')
    src = re.sub(r'"wm": [0-9.]+, "ncol"', '"wm": %g, "ncol"' % P['WM'], src, count=1)
    # 3: where the upper wall turns back
    src = re.sub(r'lerp\(0\.6 \* alarHalf, [0-9.]+ \* mm, sm\(0\.2, 0\.9, tc\)\)', 'lerp(0.6 * alarHalf, %g * mm, sm(0.2, 0.9, tc))' % P['BK_TOP'], src, count=1)
    # 5: above the radix the bridge sinks into the glabella instead of running on to the solid's box and being cut
    # there (with the gentler walls that cut showed as a step across the brow)
    e = '    partsS(x, y, z, PT);'
    assert e in src, 'partsS call not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': fade \*/[^\n]*', '', src)
    src = src.replace(e, e + '\n    %s: fade */ var upF = Math.max(0, z - (zN + %g * mm)); PT[0] += %g * upF * upF / mm;      // (the bridge sinks into the glabella: apply-face-flow.py)' % (MARK[:-3], P['FADE_Z'], P['FADE_K']), 1)
    # 7: the blend of the bridge into the face between the eyes
    src = re.sub(r'lerp\(1, [0-9.]+, sm\(zSn \+ 0\.5 \* L, zSn \+ 0\.85 \* L, z\)\)', 'lerp(1, %g, sm(zSn + 0.5 * L, zSn + 0.85 * L, z))' % P['KF_TOP'], src, count=1)
    # 4: the medial orbital rim, in faceRelief after the brow ridge
    b = 'd += gauss(x, z, s * 0.15 * es, 0.62, x * s < 0.15 * es ? 0.075 : 0.12, 0.052, 0.03 * P.browRidge);'
    src = re.sub(r'd \+= gauss\(x, z, s \* 0\.15 \* es, 0\.62, x \* s < 0\.15 \* es \? 0\.075 : 0\.12, \(z < 0\.62 \? [0-9.]+ : 0\.052\), 0\.03 \* P\.browRidge\);', b, src)
    assert b in src, 'brow ridge line not found'
    src = re.sub(r'\n[^\n]*' + re.escape(MARK[:-3]) + r': rim \*/[^\n]*', '', src)
    rim = ('\n    %s: rim */ d += P.browRidge * medialRim(x * s, z, es, %g, %g);      // (the brow carried down into the nose\'s side: '
           'apply-face-flow.py)' % (MARK[:-3], P['RIM_AMP'], P['RIM_W']))
    src = re.sub(r'\n' + re.escape(MARK) + r'\n/\* the medial orbital rim:.*?\n}\n', '\n', src, flags=re.S)
    if not P['RIM_AMP'] and P['BROW_SZ_LO'] == 0.052:
        return src      # (4 and 6 are off: see the docstring)
    src = src.replace(b, b.replace('0.12, 0.052,', '0.12, (z < 0.62 ? %g : 0.052),' % P['BROW_SZ_LO']) + rim)
    fn = '''
%s
/* the medial orbital rim: the brow's lower edge running in and down past the inner corner into the side of the bridge,
   as the supraorbital margin runs into the frontal process of the maxilla. A soft ridge along a quadratic curve (head
   heights, the eye's side; es the eye spacing's scale) from the brow (0.112, 0.63) through (0.06, 0.585) to the bridge's
   side at the eye line (0.03, 0.525), fading out at both ends (face flow: apply-face-flow.py) */
function medialRim(x, z, es, amp, w) {
  if (x < -0.02 || x > 0.2 * es || z < 0.45 || z > 0.7) return 0;
  var A = [0.112 * es, 0.63], B = [0.052 * es, 0.6], C = [0.03 * es, 0.525], best = 1e9, bt = 0;
  for (var i = 0; i <= 24; i++) {
    var t = i / 24, u = 1 - t, px = u * u * A[0] + 2 * u * t * B[0] + t * t * C[0], pz = u * u * A[1] + 2 * u * t * B[1] + t * t * C[1];
    var dd = (x - px) * (x - px) + (z - pz) * (z - pz); if (dd < best) { best = dd; bt = t; }
  }
  var env = Math.sin(Math.PI * Math.min(1, 0.15 + 0.85 * bt));      // (strongest between the brow's head and the inner corner)
  return amp * env * Math.exp(-best / (w * w));
}
''' % MARK
    src = re.sub(r'\n' + re.escape(MARK) + r'\n/\* the medial orbital rim:.*?\n}\n', '\n', src, flags=re.S)
    c = 'function lipHeights()'
    assert c in src
    src = src.replace(c, fn.lstrip('\n') + c, 1)
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
