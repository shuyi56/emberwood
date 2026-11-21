"""The lapel as part of the bodice, turned back on one tight fold (John, 2026-11-21: "include the bodice as part of the
lapel dont solve for the whole bodice just a tight fold where it meets the buttons", with a camp-collar shirt: each
front is simply folded back from the neck point to the top button; and "it should have that beautiful curve": the
fold is no straight chord, it bows into the V and comes down onto the buttons along the front edge).

    python3 blender/collar/tightfold.py BUILD.blend FOLD.pkl OUT.blend OUT_FOLD.pkl ['{"bow": 0.45, "t1": 0.12}']

No cloth solve. In the front view the pattern is drawn in:
  the lapel   every lapel point is mirrored across its front's straight fold line (neck point N to break B), so the
              lapel keeps its cut shape, and laid on what is under it there, standing off by the cloth: t0 at the
              fold, t1 further out.
  the edge    the first ring of lapel points is put on the line: the crease is one clean line.
  the front   the band of the front behind the fold, which the old cloth solve had lifted into a soft roll, is laid
              back on the body (the body's surface plus the gap the bodice keeps around it).
  the bow     the crease and the cloth either side of it are then drawn `bow` into the V, most at the middle and
              nothing at N and B, fading out across the lapel: the fold reads as a curve into the top button.
The fold pickle is written again with the new lapels, so the collar can be solved on them (finish.py).
"""
import bpy, sys, json, pickle, numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from scipy.spatial import cKDTree

a = sys.argv
has = a[-1].lstrip().startswith('{')
opt = json.loads(a[-1]) if has else {}
blend, fold, out, outf = a[-5:-1] if has else a[-4:]
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
lapv = np.zeros(len(W), bool); geo = []
for k, L in enumerate(F['folded']):
    flat, pos, lap = np.asarray(L['flat'], float), np.asarray(L['pos'], float).copy(), np.asarray(L['lap'], bool)
    j, ok = maps[k]
    N, B = np.asarray(L['roll'], float)
    if N[1] < B[1]: N, B = B, N                       # (N the neck point, B the break)
    ln = np.linalg.norm(B - N); u = (B - N) / ln; nrm = np.array([-u[1], u[0]])
    sd = (flat - N) @ nrm
    if sd[lap].mean() < 0: nrm, sd = -nrm, -sd
    # the crease: the first ring of lapel points (those sewn to a front point) is put on the line, one clean edge.
    # Only that ring: a band of points put on the line collapses whole triangles, which render as shards.
    T_ = np.asarray(L['T']); ring = np.zeros(len(flat), bool)
    for a_, b__ in ((0, 1), (1, 2), (2, 0)):
        e0, e1 = T_[:, a_], T_[:, b__]
        ring[e0[lap[e0] & ~lap[e1]]] = True; ring[e1[lap[e1] & ~lap[e0]]] = True
    on = ring
    sd = np.where(~lap, np.minimum(sd, -0.01), sd)    # (no front point past the line)
    out_ = lap & ~on
    # in the front view: the crease on the line, the lapel mirrored past it, the front where it is
    X = flat.copy()
    X[on] = flat[on] - sd[on, None] * nrm[None]
    X[out_] = flat[out_] - (2 * sd[out_])[:, None] * nrm[None]
    fo_ = ~lap
    X[fo_] = flat[fo_] - ((flat[fo_] - N) @ nrm - sd[fo_])[:, None] * nrm[None]
    # the bow: into the V (the lapel's side as cut), most at the middle of the fold
    t = np.clip(((X - N) @ u) / ln, 0, 1); dev = bow * 4 * t * (1 - t)
    dist = np.where(out_, sd, np.where(on, 0.0, -sd))
    fall = np.where(out_, np.exp(-(dist / opt.get('bow_over', 1.5)) ** 2), np.clip(1 - dist / band, 0, 1))
    X = X + (dev * fall)[:, None] * nrm[None]
    new = pos.copy()
    fr = ~lap & ~on                                   # the front's band behind the fold: back on the body
    wf = np.clip(1 + sd[fr] / band, 0, 1); wf = wf * wf * (3 - 2 * wf)
    yf = yfront(X[fr]); yf = np.where(np.isfinite(yf), yf, pos[fr, 1])
    new[fr, 0], new[fr, 2] = X[fr, 0], X[fr, 1]
    tl = ((X - N) @ u) / ln; H = rollh(tl)
    new[fr, 1] = (1 - wf) * pos[fr, 1] + wf * yf - 0.5 * H[fr] * wf ** 2
    yo = yfront(X[on]); new[on, 0], new[on, 2], new[on, 1] = X[on, 0], X[on, 1], np.where(np.isfinite(yo), yo, pos[on, 1]) - 0.5 * t0 - 0.5 * H[on]
    W[j[ok & (fr | on)]] = new[ok & (fr | on)]
    lapv[j[ok & out_]] = True
    geo.append((flat, pos, lap, sd, on, out_, X, new, j, ok, N, B, H))
    print('tightfold: lapel %d: fold from (%.2f, %.2f) to (%.2f, %.2f), bowed %.2f into the V; %d points on the crease, %d of the front laid back'
          % (k, N[0], N[1], B[0], B[1], bow, on.sum(), fr.sum()))
# what the lapels lie on: the bodice without its lapels (the fronts up to their creases), and the body
polys = [tuple(p.vertices) for p in b.data.polygons if not any(lapv[i] for i in p.vertices)]
allV = np.vstack([W, SV]); n0 = len(W)
polys += [tuple(i + n0 for i in p) for p in SP]
bvh = BVHTree.FromPolygons([Vector(p) for p in allV], polys)
V_ = np.asarray(F['V'], float); kv = cKDTree(V_)
for k, L in enumerate(F['folded']):
    flat, pos, lap, sd, on, out_, X, new, j, ok, N, B, H = geo[k]
    miss = 0
    ya_all = yfront(X[out_]); ia = dict(zip(np.nonzero(out_)[0], ya_all))
    for i in np.nonzero(out_)[0]:
        hit = bvh.ray_cast(Vector((X[i, 0], -40.0, X[i, 1])), Vector((0, 1, 0)))
        r = np.clip(sd[i] / tw, 0, 1); r = r * r * (3 - 2 * r)
        off = t0 + (t1 - t0) * r + H[i] * float(rollprof(np.array([sd[i]]))[0])
        ya = (ia[i] if np.isfinite(ia[i]) else pos[i, 1]) - off
        if hit[0] is None:
            miss += 1; new[i] = (X[i, 0], ya, X[i, 1]); continue
        n = np.array(hit[1][:]); n = n if n[1] < 0 else -n
        p = np.array(hit[0][:]) + n * off
        g = np.clip(sd[i] / 0.3, 0, 1)         # (right at the crease by the front's own depth: the faces there are open)
        new[i] = (1 - g) * np.array([X[i, 0], min(ya, p[1]), X[i, 1]]) + g * p
    # a light fairing of the laid lapel's depth over the surface, its crease held
    nb = [set() for _ in range(len(pos))]
    for t_ in np.asarray(L['T']):
        for q_ in range(3): nb[t_[q_]].add(int(t_[(q_ + 1) % 3])); nb[t_[q_]].add(int(t_[(q_ + 2) % 3]))
    w = out_ * np.clip(sd / 0.12, 0, 1); idx = np.nonzero(w > 1e-3)[0]
    for r in range(opt.get('fair', 40)):               # (the roll's surface is one smooth sweep: the ground's small steps go)
        A = np.array([new[list(nb[i])][:, 1].mean() for i in idx]); new[idx, 1] += 0.6 * w[idx] * (A - new[idx, 1])
    mv = np.linalg.norm(new - pos, axis=1)
    print('tightfold: lapel %d: %d points laid (%d without ground); moved up to %.2f (mean %.2f)' % (k, out_.sum(), miss, mv[lap].max(), mv[lap].mean()))
    for i in np.nonzero(ok)[0]:
        b.data.vertices[int(j[i])].co = new[i] @ inv[:3, :3].T + inv[:3, 3]
    dv, jv = kv.query(pos); okv = dv < 1e-4; V_[jv[okv]] = new[okv]
    L['pos'] = new
F['V'] = V_
b.data.update()
pickle.dump(F, open(outf, 'wb'))
bpy.ops.wm.save_as_mainfile(filepath=out)
