"""The colour-map check: run before a render, reads the colours a build will paint and flags pairs that touch but
do not separate, or a part that reads black against what it lies on. Report-only, like the flow check.

    python3 tools/render-review/colorcheck.py boot [--out DIR]

For the boot it reads the materials of boot.py, hose.py and the skin (female.py) from their ramps (so the check
sees exactly what the render is given, not a hex typed here). It prints a table of the touching pairs and writes
colour-map.png: every material's five ramp tones, and each pair's lightness gap, colour distance and hue gap.

The limits (CIE Lab, D65):
  - touching parts need a lightness gap of 12 or more, or a colour distance (dE) of 25 or more;
  - when the lightness gap is under 25, the hue gap must be 20 degrees or more (CLAUDE.md: a brown beside brown
    vanishes; separate by hue, not by value alone);
  - the grayscale view (relative luminance) must separate them by 6 of 100 or more;
  - a cord laid over a surface must be lighter than that surface's own shadow tone (its ramp's darkest) by 10 L, and
    not darker than its mid tone by more than 8 L: a dark cord in a dark opening reads as a black hole and
    darkens what is around it.
"""
import os, sys, math, argparse
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'blender')
for d in ('base', 'boot', 'hose', 'shared'):
    sys.path.insert(0, os.path.join(R, d))
import numpy as np


def lab(rgb):
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    x, y, z = (M @ c) / np.array([0.95047, 1.0, 1.08883])
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def gray(rgb):
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    return 100.0 * float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])      # (relative luminance, 0-100: what a grayscale view shows)


def hue(L):
    return math.degrees(math.atan2(L[2], L[1])) % 360


def hue_gap(a, b):
    d = abs(hue(a) - hue(b)) % 360
    return min(d, 360 - d)


# what touches what on the boot, and which parts are cords lying over a surface
PAIRS = {'boot': [('boot leather', 'boot lace'), ('boot leather', 'boot eyelet'), ('boot lace', 'boot eyelet'),
                  ('boot leather', 'boot thread'), ('boot leather', 'hose wool'), ('hose wool', 'skin'),
                  ('boot lace', 'hose wool')]}
CORDS = {'boot': [('boot lace', 'boot leather')]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('figure')
    ap.add_argument('--out', default='.')
    ap.add_argument('--try', dest='tries', nargs='*', default=[], help="swatches to test before changing a file: 'boot leather=0x3a2418'")
    a = ap.parse_args()
    import female, boot, hose
    mats = {m['name']: m for m in female.MATERIALS.values()}
    for t in a.tries:
        n, h = t.rsplit('=', 1)
        mats[n] = dict(mats[n], ramp=np.array(female.ramp(int(h, 16), 0.1), dtype=np.float32))
    tone = lambda n, i=2: lab(mats[n]['ramp'][i])
    rows, bad = [], 0
    for p, q in PAIRS[a.figure]:
        A, B = tone(p), tone(q)
        dL, dE, dH = abs(A[0] - B[0]), float(np.linalg.norm(A - B)), hue_gap(A, B)
        dG = abs(gray(mats[p]['ramp'][2]) - gray(mats[q]['ramp'][2]))
        why = []
        if dG < 6:
            why.append('the same grey in a grayscale view (gap %.1f of 100)' % dG)
        if dL < 12 and dE < 25:
            why.append('too close in value and colour (dL %.0f, dE %.0f)' % (dL, dE))
        if dL < 25 and dH < 20 and min(np.hypot(A[1], A[2]), np.hypot(B[1], B[2])) > 8:
            why.append('similar value and the same hue (dL %.0f, hue gap %.0f deg)' % (dL, dH))
        rows.append((p, q, dL, dE, dH, why)); bad += bool(why)
    cords = []
    for c, s in CORDS[a.figure]:
        Lc, Ls, Lm = tone(c)[0], tone(s, 0)[0], tone(s, 2)[0]
        why = []
        if Lc < Ls + 10:
            why.append('cord L %.0f is not 10 lighter than the surface\'s shadow tone (%.0f): it reads black in the opening' % (Lc, Ls))
        if Lc < Lm - 8:
            why.append('cord L %.0f is more than 8 darker than the surface\'s mid tone (%.0f)' % (Lc, Lm))
        cords.append((c, s, Lc, Ls, Lm, why)); bad += bool(why)
    for p, q, dL, dE, dH, why in rows:
        print('%-3s %-12s on %-12s dL %5.1f  dE %5.1f  hue gap %5.1f  %s' % ('OK' if not why else 'BAD', p, q, dL, dE, dH, '; '.join(why)))
    for c, s, Lc, Ls, Lm, why in cords:
        print('%-3s cord %-12s over %-12s cord L %4.1f, surface shadow L %4.1f, mid L %4.1f  %s' % ('OK' if not why else 'BAD', c, s, Lc, Ls, Lm, '; '.join(why)))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = sorted({n for pr in PAIRS[a.figure] for n in pr})
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 2.2))
    for y, n in enumerate(names):
        for i, rgb in enumerate(mats[n]['ramp']):
            ax.add_patch(plt.Rectangle((i, -y), 0.95, 0.85, color=np.asarray(rgb) / 255.0))
        ax.text(-0.1, -y + 0.4, n, ha='right', va='center', fontsize=9)
    ax.set_xlim(-2.2, 5.2); ax.set_ylim(-len(names) + 0.2, 1.2); ax.axis('off')
    txt = ['%s %s/%s: dL %.0f dE %.0f hue %.0f' % ('OK ' if not w else 'BAD', p, q, dL, dE, dH) for p, q, dL, dE, dH, w in rows]
    txt += ['%s cord %s over %s: L %.0f vs shadow %.0f / mid %.0f' % ('OK ' if not w else 'BAD', c, s, Lc, Ls, Lm) for c, s, Lc, Ls, Lm, w in cords]
    ax.set_title('Colour map: the ramps (dark to light) the build is painted with\n' + '\n'.join(txt), fontsize=7, loc='left')
    os.makedirs(a.out, exist_ok=True)
    plt.savefig(os.path.join(a.out, 'colour-map.png'), dpi=140, bbox_inches='tight')
    print('colour check: %s (%d flagged); map in %s' % ('PASS' if not bad else 'WATCH', bad, os.path.join(a.out, 'colour-map.png')))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
