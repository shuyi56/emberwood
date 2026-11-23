"""A body's crease (the armpit: the arm's inner face against the torso's side, the arm sunk 0.2 into it): what
MeshBody(planes=1) does to cloth there against MeshBody(planes=k) (solver-issues.md, S3). Numpy + scipy, no Blender:
    python3 blender/shared/test_crease_body.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabric as xf   # noqa: E402

RT, RA, SINK, GAP = 4.0, 1.6, 0.2, 0.06          # torso and arm radii (figure units), the arm sunk into the side
AX = RT + RA - SINK                               # the arm's axis, at (AX, 0)


def surface(step=0.05, zs=np.arange(-1.0, 1.01, 0.05)):
    """The two cylinders' outer surface (each without the part inside the other), points and outward normals."""
    P, N = [], []
    for r, cx, other, ro in ((RT, 0.0, AX, RA), (RA, AX, 0.0, RT)):
        th = np.arange(0, 2 * math.pi, step / r)
        c = np.column_stack([cx + r * np.cos(th), r * np.sin(th)])
        keep = np.hypot(c[:, 0] - other, c[:, 1]) > ro
        for z in zs:
            P.append(np.column_stack([c[keep], np.full(keep.sum(), z)]))
            N.append(np.column_stack([np.cos(th[keep]), np.sin(th[keep]), np.zeros(keep.sum())]))
    return np.vstack(P), np.vstack(N)


def push_field(body, h=0.02):
    """The body's push over a grid in the crease's section (z 0): how much neighbouring points' pushes differ, per
    unit of their spacing (the strain the body alone forces on cloth lying there)."""
    xs, ys = np.arange(3.3, 4.6, h), np.arange(0.0, 1.4, h)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    G = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    out = (np.hypot(G[:, 0], G[:, 1]) > RT) & (np.hypot(G[:, 0] - AX, G[:, 1]) > RA)     # (outside both)
    Pp, _ = body(G.copy())
    d = (Pp - G).reshape(len(xs), len(ys), 3)
    o = out.reshape(len(xs), len(ys))
    jx = np.linalg.norm(d[1:] - d[:-1], axis=2)[o[1:] & o[:-1]] / h
    jy = np.linalg.norm(d[:, 1:] - d[:, :-1], axis=2)[o[:, 1:] & o[:, :-1]] / h
    j = np.concatenate([jx, jy])
    return np.percentile(j, 99), j.max()


def ribbon(body, steps=400, n=81, rows=9):
    """A strip of cloth across the crease's front (y > 0), from the torso's front to the arm's front, pinned at both
    ends, cut 6 % long so it sags into the crease under a pull toward it (-y). Returns the threads' strain (1st and
    99th percentiles, worst, in %) after `steps`."""
    a0, a1 = math.radians(60), math.radians(120)          # on the torso, then on the arm (its angle from its axis)
    t = np.linspace(0, 1, n)
    # (the strip's path: round the torso from 60 degrees to the crease, then round the arm to 120)
    yc = math.sqrt(RT ** 2 - ((RT ** 2 - RA ** 2 + AX ** 2) / (2 * AX)) ** 2)
    ac_t = math.atan2(yc, (RT ** 2 - RA ** 2 + AX ** 2) / (2 * AX))
    ac_a = math.atan2(yc, (RT ** 2 - RA ** 2 + AX ** 2) / (2 * AX) - AX)
    half = n // 2
    th_t = np.linspace(a0, ac_t, half + 1)[:-1]
    th_a = np.linspace(ac_a, a1, n - half)
    c = np.vstack([np.column_stack([(RT + GAP) * np.cos(th_t), (RT + GAP) * np.sin(th_t)]),
                   np.column_stack([AX + (RA + GAP) * np.cos(th_a), (RA + GAP) * np.sin(th_a)])])
    zs = np.linspace(-0.4, 0.4, rows)
    V = np.vstack([np.column_stack([c, np.full(n, z)]) for z in zs])
    idx = np.arange(n * rows).reshape(rows, n)
    T = []
    for r in range(rows - 1):
        for i in range(n - 1):
            a, b, cc, d = idx[r, i], idx[r, i + 1], idx[r + 1, i + 1], idx[r + 1, i]
            T += [(a, b, cc), (a, cc, d)]
    T = np.array(T)
    fab = xf.Fabric(V, dict(gravity=(0.0, -40.0, 0.0), walk='gauss-seidel', iterations=30, damping=0.2))
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    rest = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    fab.mesh(T)
    fab._rest[0] = fab._rest[0] * 1.06                    # (cut long: it falls into the crease)
    fab.pin(np.concatenate([idx[:, 0], idx[:, -1]]))
    for _ in range(steps):
        fab.step(1 / 240, body)
    L = np.linalg.norm(fab.pos[E[:, 0]] - fab.pos[E[:, 1]], axis=1)
    st = 100 * (L / (rest * 1.06) - 1)
    X = fab.pos
    inside = np.maximum(RT - np.hypot(X[:, 0], X[:, 1]), RA - np.hypot(X[:, 0] - AX, X[:, 1]))
    mid = 0.5 * (X[E[:, 0]] + X[E[:, 1]])
    crease = np.hypot(mid[:, 0] - (RT ** 2 - RA ** 2 + AX ** 2) / (2 * AX), mid[:, 1] - yc) < 0.6
    return (np.percentile(st, 1), np.percentile(st, 99), np.abs(st).max(), np.abs(st[crease]).max(),
            np.abs(st[~crease]).max(), inside.max())


if __name__ == '__main__':
    P, N = surface()
    fmt = 'ribbon strain 1/99 %% %.1f/%.1f, worst %.0f %% (in the crease %.0f, away %.0f); deepest inside %.3f'
    print('no body: ' + fmt % ribbon(None))
    for k in (1, 8):
        body = xf.MeshBody(P, N, gap=GAP, reach=1.0, planes=k)
        p99, mx = push_field(body)
        print('planes=%d: body push jump p99 %.2f, worst %.1f (per unit spacing); ' % (k, p99, mx) + fmt % ribbon(body))
