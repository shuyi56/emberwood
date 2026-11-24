"""The floor under a rolled fold: bake(body_band=False) keeps a floor off the turn's own rows (solver-issues.md, S1),
and Folds.land() reports where the flap lands against its draft (S2). Plain numpy + scipy, no Blender:
    python3 blender/shared/test_folds_band.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'blender', 'shared'))  # fabric.py (folds via its shim)
import fabric as xf   # noqa: E402
import folds as fo    # noqa: E402

H = 0.05


def sheet(w=1.6, l=2.0, h=H):
    xs, ys = np.arange(0, w + 1e-9, h), np.arange(0, l + 1e-9, h)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    flat = np.column_stack([X.ravel(), Y.ravel()])
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(nx, ny)
    T = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            T += [(a, b, c), (a, c, d)]
    return flat, np.array(T)


def run(body_band, use_body=True, gap=0.28, radius=0.16, angle=160.0, line_x=0.6):
    flat, T = sheet()
    fab = xf.Fabric(np.column_stack([flat, np.zeros(len(flat))]), dict(gravity=(0, 0, -375.0), bend=1e-6))
    fab.mesh(T, rest_pos=flat)
    fab.pin(np.nonzero(flat[:, 0] < line_x - 1e-9)[0])          # (the front: held where it lies)
    floor = lambda P: _floor(P, gap)                            # (the chest under the lapel, gap off the front)
    F = fo.Folds(fab, flat, T)
    F.add([[line_x, 2.1], [line_x, -0.1]], math.radians(angle), radius=radius, side=1, name='lapel')
    rep = F.bake(body=floor if use_body else None, body_band=body_band, most=1500, log=None)
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    r = np.linalg.norm(flat[E[:, 0]] - flat[E[:, 1]], axis=1)
    L = np.linalg.norm(fab.pos[E[:, 0]] - fab.pos[E[:, 1]], axis=1)
    st = 100 * (L / r - 1)
    mov = fab.w[E[:, 0]] + fab.w[E[:, 1]] > 0
    return rep, np.percentile(st[mov], [5, 95]), np.abs(st[mov]).max(), F.land()[0]


def _floor(P, gap):
    hit = P[:, 2] < gap
    hit &= P[:, 0] < 0.6                                         # (the floor is over the front, where the flap lies)
    P[hit, 2] = gap
    return P, hit


if __name__ == '__main__':
    for bb, ub in ((True, False), (True, True), (False, True)):
        rep, p, worst, ld = run(bb, ub)
        print('body' if ub else 'no body', end=' ')
        c = rep.creases[0]
        print('body_band=%s: %.1f of %.1f deg, strain 5/95 %% %.1f/%.1f, worst %.0f %%; lands %+.3f along, %+.3f off, '
              'worst %.3f' % (bb, c['achieved_deg'], c['target_deg'], p[0], p[1], worst, ld['along'], ld['off'],
                              ld['worst']))
