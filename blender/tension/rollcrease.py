"""python3 blender/tension/rollcrease.py BUILD_DIR... (each holding build.npz from dump_build.py and out.blend.collar.npz)

the crease where the collar's fall meets its roll turn: the dihedral (degrees) across each edge shared by a fall face
and a turn face; reported over the fronts (|x| > 2.0, where the stand dies away) and round the back"""
import sys, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tension as TS  # noqa: E402
def crease(p):
    B = np.load(p + '/build.npz'); nf = len(np.load(p + '/out.blend.collar.npz')['pts'])
    X, T = B['collar_X'], B['collar_T']; T = T[TS.areas(X, T) > 1e-10]
    n = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]); n /= np.linalg.norm(n, axis=1)[:, None]
    isf = (T < nf).all(1)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); key = np.sort(e, axis=1)
    tri = np.tile(np.arange(len(T)), 3); o = np.lexsort((key[:, 1], key[:, 0])); key, tri = key[o], tri[o]
    same = (key[1:] == key[:-1]).all(1); a, b = tri[:-1][same], tri[1:][same]; k = key[:-1][same]
    mix = isf[a] != isf[b]
    d = np.degrees(np.arccos(np.clip((n[a] * n[b]).sum(1), -1, 1)))[mix]; m = X[k[mix]].mean(1)
    fr = np.abs(m[:, 0]) > 2.0
    return d, m, fr
if __name__ == '__main__':
    for p in sys.argv[1:]:
        d, m, fr = crease(p)
        print('%-28s fall/turn edges %d: fronts max %.1f p90 %.1f mean %.1f | back max %.1f mean %.1f' % (p, len(d), d[fr].max(), np.percentile(d[fr], 90), d[fr].mean(), d[~fr].max(), d[~fr].mean()))
