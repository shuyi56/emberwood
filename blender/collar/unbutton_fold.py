"""Unbutton, then fold (John, 2026-11-21: "include the bodice as part of the lapel", "you have to unbutton first to fold
the bodice", with a camp-collar shirt). The lapel is no drafted piece: it is the buttoned front's own cloth above the
top button that stays done up. Each front is opened down the centre-front edge to the break, and the cloth between
the fold line (neck point N to break B), the front's edge and the neckline is turned back on the fold.

    python3 blender/collar/unbutton_fold.py BUILD.blend FOLD.pkl OUT.blend ['{"roll": 0.4, "bow": 0.3, "lap": 0.61}']

  the fronts  her right front laps over her left by `lap` (the button stand): its edge runs down x = +lap, her left
              front's down x = -lap, both from the neckline to the break.
  the lapel   in the front view, the triangle N, T (the front edge's top, on the neckline), B. Her left front is the
              under one: its fold is hidden below where her right front crosses it, so its lapel is cut off there.
  the fold    as tightfold.py: mirrored across the line N to B, the crease bowed into the V, the cloth rolled off the
              chest (full at the neck end, tight at the button) and laid on the bodice with the cloth's thickness.
The drafted lapels' faces are taken out of the bodice and the new lapels put in, sewn to the front along the crease.
"""
import bpy, sys, json, pickle, numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from scipy.spatial import cKDTree

a = sys.argv
has = a[-1].lstrip().startswith('{')
opt = json.loads(a[-1]) if has else {}
blend, fold, out = a[-4:-1] if has else a[-3:]
t0, t1, tw = opt.get('t0', 0.03), opt.get('t1', 0.16), opt.get('t_over', 0.8)
ew, bow, band = opt.get('edge', 0.08), opt.get('bow', 0.3), opt.get('band', 0.36)
# the roll (John: "it's flat it should curve outwards proper ply"): the fold is no pressed crease, the cloth turns over on
# a radius. Its height is `roll` at the neck end and nothing at the button; across the lapel it rises on a quarter
# circle of radius roll_r from the crease and settles back onto the chest over roll_over.
roll, roll_r, roll_over, roll_p = opt.get('roll', 0.4), opt.get('roll_r', 0.35), opt.get('roll_over', 1.3), opt.get('roll_p', 0.8)
def rollh(t):
    return roll * np.clip(1 - t, 0, 1) ** roll_p * np.clip(t / 0.08, 0, 1)        # (down to nothing at N too: the collar takes over there)
def rollprof(d):
    d = np.maximum(d, 0); up = np.sqrt(np.clip(1 - (1 - np.minimum(d, roll_r) / roll_r) ** 2, 0, 1))
    return np.where(d < roll_r, 0.5 + 0.5 * up, np.exp(-((d - roll_r) / roll_over) ** 2))
F = pickle.load(open(fold, 'rb'))
bpy.ops.wm.open_mainfile(filepath=blend)
b = bpy.data.objects['Redingote bodice']; mw = np.array(b.matrix_world); inv = np.linalg.inv(mw)
W = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
kd = cKDTree(W)
sk = bpy.data.objects['Skin body']; sw = np.array(sk.matrix_world)
SV = np.array([v.co[:] for v in sk.data.vertices]) @ sw[:3, :3].T + sw[:3, 3]
SP = [tuple(p.vertices) for p in sk.data.polygons]
skin = BVHTree.FromPolygons([Vector(p) for p in SV], SP)
def ysk(q):
    o = []
    for x, z in np.atleast_2d(q):
        h = skin.ray_cast(Vector((x, -40.0, z)), Vector((0, 1, 0))); o.append(h[0][1] if h[0] is not None else np.nan)
    return np.array(o)
inpatch = np.zeros(len(W), bool); maps = []
for L in F['folded']:
    d, j = kd.query(np.asarray(L['pos'], float)); ok = d < 1e-4; maps.append((j, ok)); inpatch[j[ok]] = True
# the gap the bodice keeps round the body, read off the bodice beside the patches
allflat = np.vstack([np.asarray(L['flat'], float) for L in F['folded']])
lo, hi = allflat.min(0) - 1.2, allflat.max(0) + 1.2
ref = np.nonzero(~inpatch & (W[:, 1] < 0) & (W[:, 0] > lo[0]) & (W[:, 0] < hi[0]) & (W[:, 2] > lo[1]) & (W[:, 2] < hi[1]))[0]
ref = ref[::3]; yr = ysk(W[ref][:, [0, 2]]); g_ = W[ref, 1] - yr; good = np.isfinite(g_) & (g_ < 0.05) & (g_ > -0.6)
ref, g_ = ref[good], g_[good]; kr = cKDTree(W[ref][:, [0, 2]])
print('tightfold: the bodice stands %.3f off the body beside the lapels (median of %d points)' % (-np.median(g_), len(ref)))
def yfront(q):
    q = np.atleast_2d(q); dr, jr = kr.query(q, k=8); wg = 1 / (dr + 0.05) ** 2
    return ysk(q) + (g_[jr] * wg).sum(1) / wg.sum(1)
import bmesh
lapw = opt.get('lap', 0.61); nsub = opt.get('n', 44)
lapv = np.zeros(len(W), bool); geo = []
# 1. the fronts behind the fold, laid back on the body, and their edges put on the crease
for k, L in enumerate(F['folded']):
    flat, pos, lap = np.asarray(L['flat'], float), np.asarray(L['pos'], float).copy(), np.asarray(L['lap'], bool)
    j, ok = maps[k]
    N, B = np.asarray(L['roll'], float)
    if N[1] < B[1]: N, B = B, N
    sgn = 1.0 if N[0] < 0 else -1.0                   # (her right front, N at x < 0, laps to +x)
    Bf = np.array([sgn * lapw, min(B[1], min(np.asarray(L_['roll'], float)[:, 1].min() for L_ in F['folded']))])   # the fold's true foot: the top button
    ln = np.linalg.norm(Bf - N); u = (Bf - N) / ln; nrm = np.array([-u[1], u[0]])
    sd = (flat - N) @ nrm
    if sd[lap].mean() < 0: nrm, sd = -nrm, -sd
    T_ = np.asarray(L['T']); rim = np.zeros(len(flat), bool)
    for a_, b__ in ((0, 1), (1, 2), (2, 0)):
        e0, e1 = T_[:, a_], T_[:, b__]
        rim[e1[lap[e0] & ~lap[e1]]] = True; rim[e0[lap[e1] & ~lap[e0]]] = True      # front points sewn to a lapel point
    fr = ~lap
    sdf = np.where(rim, 0.0, np.minimum(sd, -0.01))
    X = flat - (sd - sdf)[:, None] * nrm[None]
    tl = ((X - N) @ u) / ln; dev = bow * 4 * np.clip(tl, 0, 1) * (1 - np.clip(tl, 0, 1))
    X = X + (dev * np.clip(1 + sdf / band, 0, 1))[:, None] * nrm[None]
    H = rollh(tl)
    wf = np.clip(1 + sdf / band, 0, 1); wf = wf * wf * (3 - 2 * wf)
    yf = yfront(X[fr]); yf = np.where(np.isfinite(yf), yf, pos[fr, 1])
    new = pos.copy()
    new[fr, 0], new[fr, 2] = X[fr, 0], X[fr, 1]
    new[fr, 1] = (1 - wf[fr]) * pos[fr, 1] + wf[fr] * yf - 0.5 * H[fr] * wf[fr] ** 2
    W[j[ok & fr]] = new[ok & fr]
    lapv[j[ok & lap]] = True
    geo.append((N, Bf, B, u, nrm, ln, sgn, new[rim & fr]))
    print('unbutton_fold: front %d: fold from (%.2f, %.2f) to (%.2f, %.2f); %d front points laid back, %d on the crease' % (k, N[0], N[1], Bf[0], Bf[1], fr.sum(), (rim & fr).sum()))
for i in np.nonzero(~lapv)[0] if False else []: pass
# write the fronts, take the drafted lapels out
for k in range(len(F['folded'])):
    j, ok = maps[k]
for i in range(len(W)):
    b.data.vertices[i].co = W[i] @ inv[:3, :3].T + inv[:3, 3]
bm = bmesh.new(); bm.from_mesh(b.data); bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[v for v in bm.verts if lapv[v.index]], context='VERTS')
bm.to_mesh(b.data); bm.free(); b.data.update()
W2 = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
allV = np.vstack([W2, SV]); n0 = len(W2)
polys = [tuple(p.vertices) for p in b.data.polygons] + [tuple(i + n0 for i in p) for p in SP]
bvh = BVHTree.FromPolygons([Vector(p) for p in allV], polys)
zx = max(g[2][1] for g in geo)                        # where her right front crosses her left front's fold
def grid(A, Bq, Cq, n):
    P, T, idx = [], [], {}
    for i in range(n + 1):
        for jx in range(n + 1 - i):
            idx[(i, jx)] = len(P); a_, b_ = i / n, jx / n; P.append(A * (1 - a_ - b_) + Bq * a_ + Cq * b_)
    for i in range(n):
        for jx in range(n - i):
            T.append((idx[(i, jx)], idx[(i + 1, jx)], idx[(i, jx + 1)]))
            if jx < n - i - 1: T.append((idx[(i + 1, jx)], idx[(i + 1, jx + 1)], idx[(i, jx + 1)]))
    return np.array(P), np.array(T)
newV, newF = [], []
for k, (N, Bf, B, u, nrm, ln, sgn, crease3) in enumerate(geo):
    zT = N[1] + opt.get('neck_rise', 0.0)
    Tp = np.array([sgn * lapw, zT])                   # the front edge's top, on the neckline
    P, T = grid(N, Tp, Bf, nsub)
    if abs(B[1] - Bf[1]) > 0.05:                      # the under front: cut off where the upper front crosses its fold
        keep = P[:, 1] >= zx - 1e-6
        # (points below the crossing are drawn up onto its level rather than cut: the lapel ends in a clean edge there)
        P = np.where(keep[:, None], P, np.column_stack([P[:, 0], np.full(len(P), zx)]))
        a2 = np.abs(0.5 * ((P[T[:, 1]] - P[T[:, 0]])[:, 0] * (P[T[:, 2]] - P[T[:, 0]])[:, 1] - (P[T[:, 1]] - P[T[:, 0]])[:, 1] * (P[T[:, 2]] - P[T[:, 0]])[:, 0]))
        T = T[a2 > 1e-7]
    sd = (P - N) @ nrm
    if sd.mean() < 0: sd = -sd; nrm = -nrm
    sd = np.maximum(sd, 0)
    tl = ((P - N) @ u) / ln; tc = np.clip(tl, 0, 1); dev = bow * 4 * tc * (1 - tc)
    X = P - 2 * sd[:, None] * nrm[None] + (dev * np.exp(-(sd / opt.get('bow_over', 1.5)) ** 2))[:, None] * nrm[None]
    H = rollh(tl); yfx = yfront(X)
    kc = cKDTree(crease3[:, [0, 2]]) if len(crease3) else None
    Q = np.zeros((len(P), 3))
    for i in range(len(P)):
        r = np.clip(sd[i] / tw, 0, 1); r = r * r * (3 - 2 * r)
        off = t0 + (t1 - t0) * r + H[i] * float(rollprof(np.array([sd[i]]))[0])
        ya = (yfx[i] if np.isfinite(yfx[i]) else -3.5) - off
        hit = bvh.ray_cast(Vector((X[i, 0], -40.0, X[i, 1])), Vector((0, 1, 0)))
        if hit[0] is None:
            Q[i] = (X[i, 0], ya, X[i, 1]); continue
        n = np.array(hit[1][:]); n = n if n[1] < 0 else -n
        p = np.array(hit[0][:]) + n * off
        g = np.clip(sd[i] / 0.3, 0, 1)
        Q[i] = (1 - g) * np.array([X[i, 0], min(ya, p[1]), X[i, 1]]) + g * p
    # the crease's own points: sewn to the front's edge there
    cz = np.nonzero(sd < 1e-9)[0]
    if kc is not None and len(cz):
        dc, jc = kc.query(X[cz], k=2); wg = 1 / (dc + 1e-3)
        Q[cz, 1] = (crease3[jc][:, :, 1] * wg).sum(1) / wg.sum(1) - 0.5 * t0
    nb = [set() for _ in range(len(P))]
    for t_ in T:
        for q_ in range(3): nb[t_[q_]].add(int(t_[(q_ + 1) % 3])); nb[t_[q_]].add(int(t_[(q_ + 2) % 3]))
    w = np.clip(sd / 0.12, 0, 1); idx = np.nonzero((w > 1e-3) & np.array([len(x) > 0 for x in nb]))[0]
    for r in range(opt.get('fair', 40)):
        A = np.array([Q[list(nb[i])][:, 1].mean() for i in idx]); Q[idx, 1] += 0.6 * w[idx] * (A - Q[idx, 1])
    base = len(W2) + sum(len(v) for v in newV)
    newV.append(Q); newF += [tuple(int(i) + base for i in t_) for t_ in T]
    print('unbutton_fold: lapel %d: front edge at x %+.2f, its top at z %.2f; %.1f cm wide at the neckline; %d points, standing up to %.2f off the front'
          % (k, sgn * lapw, zT, 2.62 * sd.max(), len(P), (H * rollprof(sd)).max()))
bm = bmesh.new(); bm.from_mesh(b.data)
vs = [bm.verts.new(tuple(q @ inv[:3, :3].T + inv[:3, 3])) for Q in newV for q in Q]
bm.verts.ensure_lookup_table(); nold = len(W2)
allv = list(bm.verts)
for f in newF:
    try: bm.faces.new([allv[i] for i in f])
    except ValueError: pass
bmesh.ops.remove_doubles(bm, verts=vs, dist=1e-5)
bm.to_mesh(b.data); bm.free()
for p in b.data.polygons: p.use_smooth = True
b.data.update()
bpy.ops.wm.save_as_mainfile(filepath=out)
