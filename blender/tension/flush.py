"""The gorge join seen on the solidified meshes:

    python3 blender/tension/flush.py BUILD_DIR [OFF]

BUILD_DIR holds out.blend and lapel.blend.lapels.npz.
Along the lapel's top edge from the roll line to the notch: rays cast straight at the lapel's face, just inside the
lapel (s = -off) and just past its edge (s = +off). Prints the front surface's depth on each side and which piece
it is: flush means the collar's face just past the edge is at the lapel's face's depth.
"""
import bpy, sys, numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree
B = sys.argv[1]; off = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
bpy.ops.wm.open_mainfile(filepath=B + '/out.blend')
dg = bpy.context.evaluated_depsgraph_get()
def bvh(name):
    o = bpy.data.objects[name].evaluated_get(dg); m = o.to_mesh(); mw = o.matrix_world
    V = [mw @ v.co for v in m.vertices]; F = [tuple(p.vertices) for p in m.polygons]
    return BVHTree.FromPolygons(V, F)
TB, TC = bvh('Redingote bodice'), bvh('Redingote collar')
L = np.load(B + '/lapel.blend.lapels.npz')
def first(p, n):
    o = Vector(p + 1.5 * n); d = Vector(-n); r = []
    for nm, T in (('lapel', TB), ('collar', TC)):
        h = T.ray_cast(o, d, 3.0)
        if h[0] is not None: r.append((h[3] - 1.5, nm))
    return min(r) if r else (np.nan, '-')
for k in (1, 2):
    P, t, d, e = L['pts%d' % k], L['t%d' % k], L['d%d' % k], L['edge%d' % k].astype(bool)
    tp = t[np.argmax(np.where(e, d, -1))]; top = np.nonzero(e & (t <= tp + 1e-6))[0]; top = top[np.argsort(t[top])]
    Q = P[top]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    kd = cKDTree(P); rows = []
    for i in range(1, len(Q) - 1, max(1, len(Q) // 16)):
        nb = kd.query_ball_point(Q[i], 0.3); A = P[nb] - P[nb].mean(0); n = np.linalg.svd(A)[2][-1]; n = n if n[1] < 0 else -n
        tg = Q[i + 1] - Q[i - 1]; tg -= n * (n @ tg); tg /= np.linalg.norm(tg)
        b = np.cross(n, tg); b = b if (P[nb].mean(0) - Q[i]) @ b < 0 else -b          # away from the lapel's inside
        dl, wl = first(Q[i] - off * b, n); dc, wc = first(Q[i] + off * b, n)
        rows.append((s[i] * 2.62, dl * 26.2, wl, dc * 26.2, wc, (dl - dc) * 26.2))
    print('side %d (%s), depth above the lapel mid-surface, mm (positive: toward the viewer)' % (k, 'her right' if Q[:, 0].mean() < 0 else 'her left'))
    print('  cm along   inside:depth piece   past edge:depth piece   collar proud by')
    for r in rows: print('  %5.1f      %6.1f %-7s       %6.1f %-7s       %6.1f' % (r[0], -r[1], r[2], -r[3], r[4], -r[5]))
