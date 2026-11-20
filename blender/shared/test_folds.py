"""Folds on a flat sheet: a pressed fold and a rolled fold, each baked to its target and measured.

    python3 blender/shared/test_folds.py [path/to/fabric.py's folder]

The sheet is 12 x 8 units of cloth (a lapel's size on the 63-unit figure: about 31 x 21 cm), points 0.25 apart,
split into triangles on alternate diagonals. The near side of the fold line is pinned along its far edge (as a lapel
is held by the bodice below its roll line), the far side folds over. Checked for each fold:

  - the gradient of the hinge angle against finite differences;
  - the angle reached, within 1 degree of the target, and the steps it took;
  - crisp: the turn lies within the band it was given (a pressed fold on the one row along the line, a rolled one
    over r * angle), and the cloth either side stays flat (no hinge outside the band bent more than 3 degrees);
  - a slanted line that runs across the grid's edges rather than along them still reaches its target;
  - the fold cache: a second bake of the same piece loads the solved points and runs no step;
  - a contour sheet (contours.py) is written with every converged bake, and the cloth is not strained (< 1%)."""
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else HERE)
sys.path.insert(1, HERE)

import numpy as np  # noqa: E402

import fabric as xf  # noqa: E402
import folds as fo  # noqa: E402


def sheet(w=12.0, d=8.0, h=0.25):
    nx, ny = int(round(w / h)) + 1, int(round(d / h)) + 1
    X, Y = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, d, ny), indexing='ij')
    flat = np.column_stack([X.ravel(), Y.ravel()])
    T = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            p, q, r, s = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            T += [(p, q, r), (p, r, s)] if (i + j) % 2 == 0 else [(p, q, s), (q, r, s)]
    return flat, np.array(T), h


SHEETS = os.environ.get('FOLD_SHEETS') or tempfile.mkdtemp(prefix='fold-sheets-')
THICK = 0.08                                                  # (wool, 2 mm: the gap a folded layer lies off the one under it)
PRESS = 0.5 * THICK                                           # (a pressed fold's radius: its two layers' middles a thickness apart)


def table(s):
    """What the sheet lies on: a table at z 0 under the whole sheet (the body under a lapel), and the cloth under the
    flap (the front a lapel lies back on), so the turned side rests THICK above it instead of falling through."""
    flap = s > 0

    def body(x):
        lo = np.where(flap, THICK, 0.0)
        under = x[:, 2] < lo
        p = x.copy()
        p[under, 2] = lo[under]
        return p, under
    return body


def check_gradients():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        x = rng.normal(size=(4, 3))
        H = np.array([[0, 1, 2, 3]])
        g = fo._grads(x, H)
        for i in range(4):
            for k in range(3):
                a, b = x.copy(), x.copy()
                a[i, k] += 1e-6
                b[i, k] -= 1e-6
                num = (fo.dihedral(a, H)[0] - fo.dihedral(b, H)[0]) / 2e-6
                worst = max(worst, abs(num - g[i][0, k]))
    assert worst < 1e-5, 'hinge gradient off by %g' % worst
    print('gradient: matches finite differences (worst %.1e)' % worst)


def bake(line, angle, radius, label, flat, T, h, gravity=(0, 0, 0), supported=False):
    fab = xf.Fabric(np.column_stack([flat, np.zeros(len(flat))]), dict(gravity=gravity))
    s = fo.Crease(line, angle, radius).frame(flat)[2]
    fab.pin(np.nonzero(s < -0.75 * h)[0])                   # (the front, held on the body up to the roll line)
    fab.mesh([tuple(t) for t in T])
    F = fo.Folds(fab, flat, T)
    c = F.add(line, angle, radius=radius, name=label)
    t0 = time.time()
    rep = F.bake(body=table(s) if supported else None, log=None,
                 sheet=os.path.join(SHEETS, label.replace(' ', '_').replace(',', '') + '.png'))
    took = time.time() - t0
    # crisp: where the sheet bends (every inner edge, by its distance across the line)
    Hall = fo.hinges(T)
    ang = np.degrees(np.abs(fo.dihedral(fab.pos, Hall)))
    mid = 0.5 * (flat[Hall[:, 2]] + flat[Hall[:, 3]])
    _, _, sm = c.frame(mid)
    band = c.width(h)
    outside = (sm < -1.5 * h) | (sm > band + 1.5 * h)
    bent = ang > 1.0
    span = (sm[bent].min(), sm[bent].max()) if bent.any() else (0, 0)
    stray = float(ang[outside].max())
    r = rep.creases[0]
    print('%-26s %6.1f of %5.1f deg in %4d steps (%.1f s); bent across s = %.2f..%.2f (band %.2f), outside it %.1f deg'
          % (label, r['achieved_deg'], r['target_deg'], rep.steps, took, span[0], span[1], band, stray))
    assert rep.ok, str(rep)
    # the contour sheet (John's rule: contours inspected after every solve, before any render)
    assert rep.sheet and os.path.exists(rep.sheet) and os.path.exists(rep.sheet[:-4] + '.json'), 'no contour sheet'
    assert F.contours['strain_max_pct'] < 1.0, '%s: strained %.1f%%' % (label, F.contours['strain_max_pct'])
    assert abs(r['achieved_deg'] - r['target_deg']) <= 1.0
    assert stray < 3.0, '%s: the cloth outside the band bends %.1f deg' % (label, stray)
    return fab, F, rep, span


def main():
    check_gradients()
    flat, T, h = sheet()
    vertical = np.array([[6.0, -1.0], [6.0, 9.0]])            # along the grid's edges
    # (a line's moving side is to its left: the line runs up, so x < 6 moves; flip it so x > 6 moves)
    vertical = vertical[::-1]
    print('fabric.py from', os.path.dirname(xf.__file__))
    _, _, _, sp = bake(vertical, math.radians(175), PRESS, 'pressed 175', flat, T, h)
    assert sp[1] - sp[0] <= 2.5 * h, 'a pressed fold should turn on one row (it spreads %.2f)' % (sp[1] - sp[0])
    r = 0.6                                                   # (a lapel's roll: about 1.5 cm on the figure)
    _, _, _, sp = bake(vertical, math.radians(150), r, 'rolled 150, r 0.6', flat, T, h)
    assert sp[1] - sp[0] >= 0.7 * r * math.radians(150), 'a rolled fold should spread over r * angle'
    bake(vertical, math.radians(90), PRESS, 'pressed 90', flat, T, h)
    # a slanted line: the pattern cut with the line in it (the grid turned to it) folds as well as a straight one
    a = math.atan2(3.0, 10.0)
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    turned = (flat - [6.0, 4.0]) @ Rm.T + [6.0, 4.0]
    slant = (vertical - [6.0, 4.0]) @ Rm.T + [6.0, 4.0]
    bake(slant, math.radians(160), PRESS, 'pressed 160, slanted', turned, T, h)
    # drawn across the triangles instead, a pressed fold is refused with the reason, and a rolled one works
    across = np.array([[7.5, 9.0], [4.5, -1.0]])
    fab = xf.Fabric(np.column_stack([flat, np.zeros(len(flat))]))
    fab.mesh([tuple(t) for t in T])
    try:
        fo.Folds(fab, flat, T).add(across, math.radians(160), PRESS)
        raise AssertionError('a pressed fold across the triangles should be refused')
    except ValueError as ex:
        print('refused:', str(ex)[:90] + '...')
    bake(across, math.radians(160), 0.6, 'rolled 160, across grid', flat, T, h)

    # the fold cache
    os.environ['EMBER_CLOTH_CACHE'] = tempfile.mkdtemp()
    pos = []
    for k in range(2):
        fab = xf.Fabric(np.column_stack([flat, np.zeros(len(flat))]), dict(gravity=(0, 0, 0)))
        fab.pin(np.nonzero(flat[:, 0] < 6.0 - 0.75 * h)[0])
        fab.mesh([tuple(t) for t in T])
        t0 = time.time()
        p, rep, state = fo.bake_cached('test-sheet', fab, flat, T, [dict(line=vertical, angle=math.radians(175), radius=PRESS)],
                                       log=None)
        print('cache, run %d: %s in %.2f s (%.1f deg)' % (k + 1, state, time.time() - t0, rep['creases'][0]['achieved_deg']))
        assert state == ('solved' if k == 0 else 'cached')
        pos.append(p)
    assert np.array_equal(pos[0], pos[1])
    print('contour sheets in', SHEETS)
    print('all fold checks passed')


if __name__ == '__main__':
    main()
