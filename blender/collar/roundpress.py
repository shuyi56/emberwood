# roundpress.py IN.blend COLLAR.pkl OUT.blend '{"round":..., "push":...}'
import bpy, sys, pickle, json, numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
blend, pk, out, spec = sys.argv[-4], sys.argv[-3], sys.argv[-2], json.loads(sys.argv[-1])
D = pickle.load(open(pk, 'rb')); kind = np.asarray(D['kind']); T = np.asarray(D['T']); UV = np.asarray(D['UV'])
bpy.ops.wm.open_mainfile(filepath=blend)
o = bpy.data.objects['Collar']; mw = np.array(o.matrix_world); inv = np.linalg.inv(mw)
P = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
assert len(P) == len(kind)
N = len(P)
nb = [set() for _ in range(N)]
for t in T:
    for i in range(3): nb[t[i]].add(int(t[(i+1) % 3])); nb[t[i]].add(int(t[(i+2) % 3]))
E_ = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
Eu, c = np.unique(E_, axis=0, return_counts=True); rim = np.zeros(N, bool); rim[np.unique(Eu[c == 1])] = True
def lap(P, w, rounds, taubin=True):
    P = P.copy(); idx = np.nonzero(w > 1e-3)[0]
    for r in range(rounds):
        for lam in ((0.5, -0.53) if taubin else (0.5,)):
            A = np.array([P[list(nb[i])].mean(0) for i in idx])
            P[idx] += (lam * w[idx])[:, None] * (A - P[idx])
    return P
P0 = P.copy()
# 1. round the fold beside the neck: the cloth within `band` of the roll line, at the neck's side
R = spec.get('round', {})
if R:
    rr = np.nonzero(kind == 'r')[0]
    from scipy.spatial import cKDTree
    d, j = cKDTree(UV[rr]).query(UV)
    wb = np.clip(1 - d / R.get('band', 1.0), 0, 1); wb = wb * wb * (3 - 2 * wb)
    y = P[:, 1]; ws = np.clip((R.get('y1', 2.0) - y) / 1.0, 0, 1) * np.clip((y - R.get('y0', -3.2)) / 0.6, 0, 1)
    sx = R.get('sides', [1, -1]); wside = np.isin(np.sign(P[:, 0]), sx) * (np.abs(P[:, 0]) > 0.8)
    w = wb * ws * wside
    w[rim] *= R.get('rim', 0.0)
    P = lap(P, w, R.get('rounds', 40), taubin=R.get('taubin', False))
    print('round: %d points, moved up to %.3f' % ((w > 1e-3).sum(), np.linalg.norm(P - P0, axis=1).max()))
# 2. press the front down onto what lies under it
Q = spec.get('push', {})
if Q:
    dg = bpy.context.evaluated_depsgraph_get()
    b = bpy.data.objects['Redingote bodice']; bw = np.array(b.matrix_world)
    BV = np.array([v.co[:] for v in b.data.vertices]) @ bw[:3, :3].T + bw[:3, 3]
    bvh = BVHTree.FromPolygons([Vector(p) for p in BV], [tuple(p.vertices) for p in b.data.polygons])
    near = np.array([bvh.find_nearest(Vector(p))[0][:] for p in P]); cl = np.linalg.norm(P - near, axis=1)
    for sx in Q.get('sides', [1]):
        reg = (P[:, 0] * sx > Q.get('x0', 1.0)) & (P[:, 1] < Q.get('y1', -1.8)) & (P[:, 2] < Q.get('z1', 51.6))
        print('push side', sx, 'region', reg.sum(), 'clearance p50 %.2f p90 %.2f max %.2f' % tuple(np.percentile(cl[reg], [50, 90, 100])))
        keep = Q.get('keep', 0.08)
        w = reg * np.clip((cl - keep) / 0.15, 0, 1)
        # fade at the region's borders
        w = w * np.clip((P[:, 0] * sx - Q.get('x0', 1.0)) / 0.5, 0, 1) * np.clip((Q.get('y1', -1.8) - P[:, 1]) / 0.5, 0, 1) * np.clip((Q.get('z1', 51.6) - P[:, 2]) / 0.5, 0, 1)
        newc = cl - Q.get('amount', 0.4) * w * (cl - keep)
        dirn = (P - near) / (cl[:, None] + 1e-9)
        P1 = near + dirn * newc[:, None]
        mv = np.linalg.norm(P1 - P, axis=1); P = P1
        print('  moved %d points, up to %.3f, mean of moved %.3f' % ((mv > 1e-3).sum(), mv.max(), mv[mv > 1e-3].mean() if (mv > 1e-3).any() else 0))
        ws = np.clip(w * 2, 0, 1); ws[rim] *= 0.3
        P = lap(P, ws, Q.get('rounds', 8), taubin=True)
L = P @ inv[:3, :3].T + inv[:3, 3]
for i, v in enumerate(o.data.vertices): v.co = L[i]
o.data.update()
D2 = dict(D); D2['pos'] = P; pickle.dump(D2, open(out.replace('.blend', '.pkl'), 'wb'))
bpy.ops.wm.save_as_mainfile(filepath=out)
