"""Cloth in the air above an open sheet's rim (a collar's stand over the bodice's neckline): MeshBody without and with
tangent= (solver-issues.md, S5). Numpy + scipy, no Blender:  python3 blender/shared/test_sheet_edge.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabric as xf  # noqa: E402


def run(mb):
    # the bodice: an open sheet, the plane x = 0 below its rim z = 0, normals +x (out of the body), samples every 0.05
    ys, zs = np.arange(-1, 3, 0.05), np.arange(-3, 0.001, 0.05)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    BP = np.column_stack([np.zeros(Y.size), Y.ravel(), Z.ravel()]); BN = np.tile([1.0, 0, 0], (len(BP), 1))
    # the stand: a strip rising from just above the rim, 0.2 behind the bodice's plane (in the air over the neckline)
    h, n, m = 0.147, 14, 14
    yy, zz = np.arange(n) * h, 0.05 + np.arange(m) * h
    YY, ZZ = np.meshgrid(yy, zz, indexing='ij')
    X0 = np.column_stack([np.full(YY.size, -0.2), YY.ravel(), ZZ.ravel()])
    idx = np.arange(n * m).reshape(n, m); T = []
    for i in range(n - 1):
        for j in range(m - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]; T += [(a, b, c), (a, c, d)]
    T = np.array(T)
    body = xf.MeshBody(BP, BN, gap=0.10, reach=1.0, **mb)
    fab = xf.Fabric(X0, dict(walk='gauss-seidel', iterations=4, damping=0.35, gravity=(0, 0, -375.0)))
    fab.pin(np.concatenate([idx[:, 0], idx[:, -1]]))        # (the seam below, the roll above)
    fab.mesh([tuple(t) for t in T], stretch=1e-8, bend=1e-9)
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    r0 = np.linalg.norm(X0[E[:, 0]] - X0[E[:, 1]], axis=1)
    for k in range(1, 1001):
        fab.step(1 / 960, body, neutral=True)
        p, t = body(fab.pos.copy()); live = t & (fab.w > 0); fab.pos[live] = p[live]
    s = 100 * (np.linalg.norm(fab.pos[E[:, 0]] - fab.pos[E[:, 1]], axis=1) / r0 - 1)
    return np.percentile(s, [5, 95]), np.abs(s).max()


if __name__ == '__main__':
    for mb in ({}, {'tangent': 0.075}):
        (p5, p95), worst = run(mb)
        print('MeshBody(%s): strain 5/95 %% %.1f/%.1f, worst %.1f %%' % (mb or 'as before', p5, p95, worst))
