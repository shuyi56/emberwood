"""Unbutton, then fold (John, 2026-11-21: "include the bodice as part of the lapel", "you have to unbutton first to fold
the bodice", with a camp-collar shirt). The lapel is no drafted piece: it is the buttoned front's own cloth above the
top button that stays done up. Each front is opened down the centre-front edge to the break, and the cloth between
the fold line (neck point N to break B), the front's edge and the neckline is turned back on the fold.

    python3 blender/collar/unbutton_fold.py BUILD.blend FOLD.pkl OUT.blend ['{"roll": 0.4, "bow": 0.3, "lap": 0.61, "foot": null}']

`foot` is the height the fold comes down to: the top button left done up. Left out, the front is unbuttoned fully
(John: "unbutton fully let curve fall") and the fold runs to the hem; the front buttons are then taken off (undone,
they sit on the turned-back side of the under front).

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
M = F['M']
lapw = opt.get('lap', 0.61); nsub = opt.get('n', 60)
zfoot = opt.get('foot') if opt.get('foot') is not None else float(M['z_hem_front'])
full = opt.get('foot') is None
Ns = []
for L in F['folded']:
    N, B = np.asarray(L['roll'], float); Ns.append(N if N[1] > B[1] else B)
sides = []
for N in Ns:
    sgn = 1.0 if N[0] < 0 else -1.0                   # (her right front, N at x < 0, laps to +x and lies on top)
    Bf = np.array([sgn * lapw, zfoot]); ln = np.linalg.norm(Bf - N); u = (Bf - N) / ln
    nrm = np.array([-u[1], u[0]]); nrm = nrm if nrm[0] * sgn > 0 else -nrm        # toward the V
    sides.append(dict(N=N, B=Bf, u=u, n=nrm, ln=ln, sgn=sgn))
top = [s_ for s_ in sides if s_['sgn'] > 0][0]; und = [s_ for s_ in sides if s_['sgn'] < 0][0]
zx = top['N'][1] + top['u'][1] * (0 - top['N'][0]) / top['u'][0]         # where the upper front's fold crosses the centre
print('unbutton_fold: folds from the neck points (+-%.2f, %.2f) down to z %.2f (%s); they cross at z %.2f'
      % (abs(top['N'][0]), top['N'][1], zfoot, 'the hem: fully unbuttoned' if full else 'the top button done up', zx))
def bowdev(t):
    t = np.clip(t, 0, 1); return bow * 6.75 * t * t * (1 - t)            # (falls: fullest two thirds of the way down)
# 1. the drafted lapels out; the cloth in the V out; the fronts behind each fold laid on the body, their edges on the crease
lapv = np.zeros(len(W), bool)
for k, L in enumerate(F['folded']):
    j, ok = maps[k]; lapv[j[ok & np.asarray(L['lap'], bool)]] = True
q = W[:, [0, 2]]
frontz = (W[:, 1] < M['yc']) & (W[:, 2] > zfoot - 0.2) & (W[:, 2] < top['N'][1] + 0.7)
sT, sU = (q - top['N']) @ top['n'], (q - und['N']) @ und['n']
inV = frontz & ~lapv & (sT > 0) & (sU > 0) & (W[:, 2] > zx) & (W[:, 2] < top['N'][1] + 0.05)
kill = lapv | inV
bm = bmesh.new(); bm.from_mesh(b.data); bm.verts.ensure_lookup_table()
edge_ = np.zeros(len(W), bool)
for v in bm.verts:
    if kill[v.index]:
        for e in v.link_edges:
            o_ = e.other_vert(v)
            if not kill[o_.index]: edge_[o_.index] = True
for s_, sd_, mine in ((top, sT, frontz & (sU > -1e9)), (und, sU, frontz & (W[:, 2] > zx - 0.3))):
    N, u, nrm, ln = s_['N'], s_['u'], s_['n'], s_['ln']
    sel = mine & ~kill & (sd_ > -band) & (sd_ < 0.3) & (W[:, 2] < N[1] + 0.3)
    if s_ is und: sel &= sT > -0.05                   # (her left front only: not her right front's cloth)
    else: sel &= (sU > -0.05) | (W[:, 2] < zx)
    on = sel & edge_ & (np.abs(sd_) < 0.3)
    # (the front's cut edge is tucked just behind the crease and under it: the lapel's own first row, a clean line of
    # the new mesh, is the edge that shows; the cut's staircase is hidden by it)
    sdf = np.where(on, -0.04, np.minimum(sd_, -0.05))
    X = q - (sd_ - sdf)[:, None] * nrm[None]
    tl = ((X - N) @ u) / ln
    wf = np.clip(1 + sdf / band, 0, 1); wf = wf * wf * (3 - 2 * wf)
    X = X + (bowdev(tl) * wf)[:, None] * nrm[None]
    H = rollh(tl) * 0
    ii = np.nonzero(sel)[0]
    yf = yfront(X[ii]); yf = np.where(np.isfinite(yf), yf, W[ii, 1])
    W[ii, 1] = (1 - wf[ii]) * W[ii, 1] + wf[ii] * yf - 0.5 * H[ii] * wf[ii] ** 2 + 0.03 * on[ii]
    W[ii, 0], W[ii, 2] = X[ii, 0], X[ii, 1]
    s_['crease3'] = W[np.nonzero(on)[0]].copy()
    print('unbutton_fold: %s front: %d points laid back behind the fold, %d on the crease' % ('upper' if s_ is top else 'under', len(ii), on.sum()))
for v in bm.verts: v.co = tuple(W[v.index] @ inv[:3, :3].T + inv[:3, 3])
bmesh.ops.delete(bm, geom=[v for v in bm.verts if kill[v.index]], context='VERTS')
bm.to_mesh(b.data); bm.free(); b.data.update()
print('unbutton_fold: %d drafted lapel points and %d points in the V taken out' % (lapv.sum(), inV.sum()))
if full:                                              # undone, the front buttons sit on the turned-back side: off
    for o in bpy.data.objects:
        if o.name.startswith('Redingote bodice') and o.name.endswith('buttons'):
            bb = bmesh.new(); bb.from_mesh(o.data); ow = np.array(o.matrix_world)
            dead = [v for v in bb.verts if (np.array(v.co[:]) @ ow[:3, :3].T + ow[:3, 3])[1] < M['yc']]
            bmesh.ops.delete(bb, geom=dead, context='VERTS'); bb.to_mesh(o.data); bb.free(); o.data.update()
W2 = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
allV = np.vstack([W2, SV]); n0 = len(W2)
polys = [tuple(p.vertices) for p in b.data.polygons] + [tuple(i + n0 for i in p) for p in SP]
bvh = BVHTree.FromPolygons([Vector(p) for p in allV], polys)
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
# 2. the lapels: each front's own corner (N, the front edge's top T on the neckline, the fold's foot), turned back
newV, newF = [], []
for k, s_ in enumerate((top, und)):
    N, Bf, u, nrm, ln, sgn, crease3 = s_['N'], s_['B'], s_['u'], s_['n'], s_['ln'], s_['sgn'], s_['crease3']
    Tp = np.array([sgn * lapw, N[1]])
    P, T = grid(N, Tp, Bf, nsub)
    if s_ is und:                                     # the under front: cut off where the upper front crosses its fold
        P = np.where((P[:, 1] >= zx)[:, None], P, np.column_stack([P[:, 0], np.full(len(P), zx)]))
        e1, e2 = P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
        T = T[np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-7]
    sd = np.maximum((P - N) @ nrm, 0)
    tl = ((P - N) @ u) / ln
    # The lapel as tailors describe it (Diduch, "What exactly is pad stitching"; Permanent Style, "How to correct a
    # flattened lapel"; Parisian Gentleman, "The lapel roll"): the break is never a pressed crease but a rounded
    # roll, and the lapel is not a flat flap laid on the chest: padded over the hand, row by row parallel to the roll
    # line, it "stays curled" across its whole width and so hugs the chest. A mirrored, laid-flat lapel is exactly
    # the "flattened" lapel a cleaner's press makes. So, across the lapel from the break (d the cloth's own length):
    #   the roll   a half turn of radius r: the cloth leaves the front, swells r into the V, and comes back over
    #              itself 2r above the front
    #   the curl   from there the lapel bends back toward the body on a wide arc, so that its outer edge just comes
    #              down onto the chest; narrower cloth (toward the foot) only makes part of the turn.
    # The roll stays open all the way down to the foot (the "belly" is at the buttoning point): it tapers only over
    # the last tenth.
    wloc = np.maximum((np.abs(sgn * lapw - N[0]) * abs(u[1])) * (1 - np.clip(tl, 0, 1)) * 2.0 / 2.0, 1e-3)   # the lapel's width at this height
    r0 = opt.get('roll_radius', 0.18)
    rr = r0 * np.clip((1 - tl) / 0.1, 0.25, 1) * np.clip((tl + 0.05) / 0.1, 0.4, 1)
    rr = np.minimum(rr, wloc / np.pi * 0.9 + 0.02)
    arc = np.pi * rr
    th = np.minimum(sd, arc) / rr
    ex = rr * np.sin(th)                               # toward the V (+) during the turn
    hh = rr * (1 - np.cos(th))
    rest = np.maximum(sd - arc, 0)
    Rc = np.maximum((wloc - arc) ** 2 / (4 * rr) * opt.get('curl', 1.0), 0.6)
    ph = rest / Rc
    phm = np.arccos(np.clip(1 - (2 * rr - t1) / Rc, -1, 1))             # where the curl reaches the chest
    ex = ex - Rc * np.sin(np.minimum(ph, phm)) - np.maximum(ph - phm, 0) * Rc
    hh = np.where(rest > 0, np.maximum(2 * rr - Rc * (1 - np.cos(np.minimum(ph, phm))), t1), hh)
    X = P - sd[:, None] * nrm[None] + ex[:, None] * nrm[None] + (bowdev(tl))[:, None] * nrm[None]
    H = rollh(tl) * 0; yfx = yfront(X)
    Q = np.zeros((len(P), 3))
    for i in range(len(P)):
        off = t0 + hh[i]
        ya = (yfx[i] if np.isfinite(yfx[i]) else -3.5) - off
        hit = bvh.ray_cast(Vector((X[i, 0], -40.0, X[i, 1])), Vector((0, 1, 0)))
        if hit[0] is None:
            Q[i] = (X[i, 0], ya, X[i, 1]); continue
        p = np.array(hit[0][:]); p[1] -= off
        g = np.clip(sd[i] / 0.5, 0, 1)
        Q[i] = (1 - g) * np.array([X[i, 0], ya, X[i, 1]]) + g * np.array([X[i, 0], min(ya, p[1]) if sd[i] > arc[i] else ya, X[i, 1]])
    # the lapel is one smooth sheet: it does not follow each wrinkle of what it lies on. Its depth is faired hard, then
    # brought forward wherever that would sink it into the ground; twice.
    nb = [set() for _ in range(len(P))]
    for t_ in T:
        for q_ in range(3): nb[t_[q_]].add(int(t_[(q_ + 1) % 3])); nb[t_[q_]].add(int(t_[(q_ + 2) % 3]))
    cz = np.nonzero(sd < 1e-9)[0]                     # the crease's own points: over the front's edge there
    if len(crease3) > 2 and len(cz):
        dc, jc = cKDTree(crease3[:, [0, 2]]).query(X[cz], k=2); wg = 1 / (dc + 1e-3)
        Q[cz, 1] = (crease3[jc][:, :, 1] * wg).sum(1) / wg.sum(1) - 0.03 - t0 if False else Q[cz, 1]
    ground = Q[:, 1].copy()
    w = np.clip(sd / 0.08, 0, 1); idx = np.nonzero((w > 1e-3) & np.array([len(x) > 0 for x in nb]))[0]
    for rep in range(opt.get('sheet', 3)):
        for r in range(opt.get('fair', 12)):
            A = np.array([Q[list(nb[i])][:, 1].mean() for i in idx]); Q[idx, 1] += 0.7 * w[idx] * (A - Q[idx, 1])
        Q[idx, 1] = np.minimum(Q[idx, 1], ground[idx])
    for r in range(15):
        A = np.array([Q[list(nb[i])][:, 1].mean() for i in idx]); Q[idx, 1] += 0.5 * w[idx] * (A - Q[idx, 1])
    base = len(W2) + sum(len(v) for v in newV)
    newV.append(Q); newF += [tuple(int(i) + base for i in t_) for t_ in T]
    print('unbutton_fold: %s lapel: %.1f cm wide at the neckline, %.1f cm long; the roll %.1f cm high at its fullest'
          % ('upper' if s_ is top else 'under', 2.62 * sd.max(), 2.62 * (P[:, 1].max() - P[:, 1].min()), 2.62 * 2 * rr.max()))
bm = bmesh.new(); bm.from_mesh(b.data)
vs = [bm.verts.new(tuple(q_ @ inv[:3, :3].T + inv[:3, 3])) for Q in newV for q_ in Q]
allv = list(bm.verts)
for f in newF:
    try: bm.faces.new([allv[i] for i in f])
    except ValueError: pass
bmesh.ops.remove_doubles(bm, verts=vs, dist=1e-5)
bm.to_mesh(b.data); bm.free()
for p in b.data.polygons: p.use_smooth = True
b.data.update()
bpy.ops.wm.save_as_mainfile(filepath=out)
