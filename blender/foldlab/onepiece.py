"""One-piece notch collar and lapel, in the lab (John, 2026-11-21: "the notch collar needs to be designed and cut as one",
"no joins").

The front, its lapel and the collar are cut from one blank, as in Linder's coat front (US patent 3,009,160, 1961): the
collar panel is part of the front's top, the notch is a V-shaped opening in the blank's outline between the lapel's
point and the collar's, and the fold line runs straight up the front to the lapel's top and on as a curve round the
neck ("an upper continuation of the fold line"), so the lapel is seamless. There is no gorge seam; the collar's
stand is still sewn to the back neck, and the front neck edge under the stand closes as Linder's thin shaping dart.

Here the build's own two drafts are the parts: the lapel's flat (the front drafted on past its roll, fold_front) and
the tailored collar's half pattern (tailor.py). The collar's half is laid onto the lapel's flat at the old gorge:
its gorge (roll end X to notch G) onto the lapel's top edge (N to G), on the far side of it, so the two gorges are one
line inside the cloth. The union is meshed as one sheet, and its roll (the lapel's roll on into the collar's) is
folded in one solve, the front's strip and the collar's neck seam held as sewn.

    python3 blender/foldlab/onepiece.py pattern FOLD.pkl COLLAR.pkl CAP_DIR OUT_DIR       the cut sheet
    python3 blender/foldlab/onepiece.py try FOLD.pkl COLLAR.pkl CAP_DIR OUT_DIR '{"belly": 0.4}' [--name x]

EXPERIMENTAL."""
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import fabric as xf      # noqa: E402
import folds as fo       # noqa: E402

OP = dict(spacing=0.07, w=1.964,
          belly=0.0,          # the lapel's outer edge bowed out by this (units) at its middle: over the bust, not straight
          gorge_blend=1.2,    # how far into the collar its start is moved to meet the lapel's top (units)
          stretch=1e-8, bend=1e-9, steps=2000, rate=960, gap=0.12, friction=0.4)


def _rim(T):
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    e, c = np.unique(e, axis=0, return_counts=True)
    return e[c == 1]


def _loops(be):
    """Boundary edges into ordered loops of point indices."""
    nb = {}
    for a, b in be:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    seen, loops = set(), []
    for s in nb:
        if s in seen:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nx = [q for q in nb[cur] if q != prev and q not in seen]
            if not nx:
                break
            prev, cur = cur, nx[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    return loops


def _mirror(X, B, N):
    u = (N - B) / np.linalg.norm(N - B)
    d = X - B
    return B + 2 * (d @ u)[:, None] * u - d


def _bary(Q, P, T, vals):
    """vals at P (per point) carried to Q by the triangle each lies in (or the nearest one)."""
    import shapely
    from scipy.spatial import cKDTree
    cen = P[T].mean(1)
    kd = cKDTree(cen)
    out = np.zeros((len(Q), vals.shape[1]))
    _, cand = kd.query(Q, k=min(12, len(T)))
    for i, q in enumerate(Q):
        best, bw = None, None
        for t in cand[i]:
            a, b, c = P[T[t]]
            m = np.array([b - a, c - a]).T
            try:
                l1, l2 = np.linalg.solve(m, q - a)
            except np.linalg.LinAlgError:
                continue
            l0 = 1 - l1 - l2
            worst = min(l0, l1, l2)
            if best is None or worst > bw:
                best, bw = (t, np.array([l0, l1, l2])), worst
            if worst >= -1e-9:
                break
        t, l_ = best
        out[i] = l_ @ vals[T[t]]
    return out


def pattern(fold_pkl, collar_pkl, cap_dir, k, S):
    """The one-piece half pattern for lapel k. Returns a dict (flat points, triangles, kinds, design, ...)."""
    import shapely
    from scipy.spatial import Delaunay, cKDTree
    F = pickle.load(open(fold_pkl, 'rb'))
    C = pickle.load(open(collar_pkl, 'rb'))
    cap = dict(np.load(os.path.join(cap_dir, 'lapel%d.npz' % k), allow_pickle=True))
    flat, T, lap, built = cap['flat'], cap['T'], cap['lap'].astype(bool), cap['built']
    B, N, poly = cap['B'], cap['N'], cap['poly']
    sg = 1 if N[0] > 0 else -1
    X2, G2, X3 = F['gorges'][sg]
    h = S['spacing'] * S['w']
    ns = int((~lap).sum())
    u = (N - B) / np.linalg.norm(N - B)
    nperp = np.array([-u[1], u[0]])
    if nperp @ (np.array([0.0, N[1]]) - B) < 0:
        nperp = -nperp
    t_, sd = (flat - B) @ u, (flat - B) @ nperp
    # the lapel's flap as the build cut it: its outline (the flap's boundary loop), and its corners N', G', P', B'
    lapT = T[lap[T].all(1) | (lap[T].sum(1) >= 1)]
    rimv = np.unique(_rim(T))
    rl = rimv[lap[rimv]]
    Gm = _mirror(G2[None], B, N)[0]
    Pm = _mirror(np.asarray(poly)[3][None], B, N)[0]
    iN = rl[np.argmin(np.linalg.norm(flat[rl] - N, axis=1))]
    iG = rl[np.argmin(np.linalg.norm(flat[rl] - Gm, axis=1))]
    iP = rl[np.argmin(np.linalg.norm(flat[rl] - Pm, axis=1))]
    Np, Gp, Pp = flat[iN], flat[iG], flat[iP]
    flap = shapely.union_all([shapely.Polygon(flat[t]) for t in T if lap[t].all()]).buffer(1e-4).buffer(-1e-4)
    strip = shapely.union_all([shapely.Polygon(flat[t]) for t in T if not lap[t].any()]).buffer(1e-4).buffer(-1e-4)
    # the collar's half, this side: its pattern turned and moved so its gorge (X to its seam's front end) lies on N' G'
    UV, CT, kind, cpos = C['UV'], C['T'], C['kind'], C['pos']
    des = C['design']
    hs = np.sign(UV[:, 0]) == -sg
    s_ = np.nonzero((kind == 's') & hs)[0]
    jX = s_[np.argmin(np.linalg.norm(des[s_] - X3, axis=1))]
    jG = s_[np.argmax(np.abs(UV[s_, 0]))]
    best = None
    for refl in (1, -1):
        U = UV * np.array([refl, 1.0])
        v1, v2 = U[jG] - U[jX], Gp - Np
        th = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        Q = (U - U[jX]) @ R.T + Np
        n2 = np.array([-v2[1], v2[0]])
        if ((Q[hs] - Np) @ n2).mean() * ((flat[lap] - Np) @ n2).mean() < 0:
            best = Q
    Q = best
    cT = CT[hs[CT].all(1)]
    cpoly = shapely.union_all([shapely.Polygon(Q[t]) for t in cT]).buffer(1e-4).buffer(-1e-4)
    # the one piece: flap and collar, the front's strip left as it is (it is the bodice's own cloth, held)
    piece = shapely.unary_union([flap, cpoly]).buffer(0.02).buffer(-0.02).difference(strip.buffer(0.3 * h))
    if piece.geom_type != 'Polygon':
        piece = max(piece.geoms, key=lambda g: g.area)
    log = []
    # the lapel's outer edge (P' to B') bowed out over the bust
    if S['belly'] > 0:
        Bp = flat[rl[np.argmin(np.linalg.norm(flat[rl] - B, axis=1))]]
        ch = Bp - Pp
        Lc = np.linalg.norm(ch)
        nn = np.array([-ch[1], ch[0]]) / Lc
        if nn @ (Pp + 0.5 * ch - Np) < 0:
            nn = -nn
        # (out, away from the roll: the side the flap's own outline lies on)
        if nn @ ((Pp + 0.5 * ch) - (B + ((Pp + 0.5 * ch - B) @ u) * u)) < 0:
            nn = -nn
        q = np.linspace(0, 1, 40)
        arc = Pp[None] + q[:, None] * ch[None] + (4 * q * (1 - q) * S['belly'])[:, None] * nn[None]
        bow = shapely.Polygon(np.vstack([arc, (Pp + 0.5 * ch - 0.05 * nn)[None]])).buffer(0)
        bow = shapely.Polygon(np.vstack([arc, arc[::-1] - 0.02 * nn])).buffer(0).union(
            shapely.Polygon(np.vstack([Pp, arc[1:-1], Bp]))).buffer(0)
        piece = piece.union(bow).buffer(0.01).buffer(-0.01).difference(strip.buffer(0.3 * h))
        if piece.geom_type != 'Polygon':
            piece = max(piece.geoms, key=lambda g: g.area)
        log.append('outer edge bowed out %.2f over %.2f' % (S['belly'], Lc))
    # the roll: the lapel's (the strip's roll points, B to N) and on into the collar's
    rp = np.nonzero(~lap & (np.abs(sd) < 0.015))[0]
    rp = rp[np.argsort(t_[rp])]
    cr = np.nonzero((kind == 'r') & hs)[0]
    cr = cr[np.argsort(np.linalg.norm(Q[cr] - Np, axis=1))]
    roll_c = Q[cr]
    # ---- mesh: the build's own lapel mesh kept whole (the strip, the turn's close rows, the flap); the collar (and the
    # bow) meshed on where they lie past it, joined to its rim points: one sheet, welded along the old gorge
    old = shapely.union_all([shapely.Polygon(flat[t]) for t in T]).buffer(1e-4).buffer(-1e-4)
    add = piece.difference(old.buffer(0.02))
    adds = [g for g in (add.geoms if hasattr(add, 'geoms') else [add]) if g.area > 4 * h * h]
    add = shapely.unary_union(adds)
    pts = []
    for g in adds:
        ring = g.exterior
        bnd = np.array([ring.interpolate(d).coords[0] for d in np.arange(0, ring.length, 0.7 * h)])
        pts.append(bnd)
    bx = add.bounds
    gx, gy = np.meshgrid(np.arange(bx[0], bx[2], h), np.arange(bx[1], bx[3], h * 0.866))
    gx[1::2] += 0.5 * h
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid = grid[shapely.contains_xy(add.buffer(-0.5 * h), grid[:, 0], grid[:, 1])]
    rline = shapely.LineString(np.vstack([flat[rp], roll_c]))
    rc_s = np.array([rline.interpolate(d).coords[0] for d in np.arange(0, rline.length, 0.7 * h)])
    rc_s = rc_s[shapely.contains_xy(add.buffer(-0.3 * h), rc_s[:, 0], rc_s[:, 1])]
    grid = grid[shapely.distance(rline, shapely.points(grid)) > 0.45 * h]
    new = np.vstack(pts + [rc_s, grid])
    new = new[cKDTree(flat).query(new)[0] > 0.45 * h]           # (the old rim's own points stand on the shared edge)
    keep = np.ones(len(new), bool)
    for i, j in sorted(cKDTree(new).query_pairs(0.35 * h)):
        if keep[i] and keep[j]:
            keep[j] = False
    new = new[keep]
    n0 = len(flat)
    near_old = rimv[shapely.distance(add, shapely.points(flat[rimv])) < 1.5 * h]
    Pd = np.vstack([flat[near_old], new])
    tri = Delaunay(Pd).simplices
    cen = Pd[tri].mean(1)
    a, b, c = Pd[tri[:, 0]], Pd[tri[:, 1]], Pd[tri[:, 2]]
    ar = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    ok = shapely.contains_xy(add.buffer(S.get('join_buf', 0.3) * h), cen[:, 0], cen[:, 1]) & ~shapely.contains_xy(old.buffer(-0.05 * h), cen[:, 0], cen[:, 1])
    ok &= shapely.contains_xy(piece.buffer(0.02), cen[:, 0], cen[:, 1]) & (ar > 0.03 * h * h)
    gidx = np.concatenate([near_old, n0 + np.arange(len(new))])
    tri = gidx[tri[ok]]
    P = np.vstack([flat, new])
    Tall = np.vstack([T, tri])
    used = np.unique(Tall)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, Tall = P[used], remap[Tall]
    a, b, c = P[Tall[:, 0]], P[Tall[:, 1]], P[Tall[:, 2]]
    crs = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    Tall[crs < 0] = Tall[crs < 0][:, [0, 2, 1]]
    is_old = used < n0
    is_strip = is_old & ~lap[np.minimum(used, n0 - 1)]
    nonman = np.unique(np.sort(np.concatenate([Tall[:, [0, 1]], Tall[:, [1, 2]], Tall[:, [2, 0]]]), 1), axis=0, return_counts=True)[1]
    log.append('mesh: %d old points, %d new; edges in 3+ triangles: %d' % (is_old.sum(), (~is_old).sum(), int((nonman > 2).sum())))
    # ---- what each point is: the collar's part (on the collar's side of the old gorge line), its seam, its roll
    gl = shapely.LineString([Np, Gp])
    n2 = np.array([-(Gp - Np)[1], (Gp - Np)[0]])
    col_side = ((P - Np) @ n2) * ((Q[hs] - Np) @ n2).mean() > 0
    in_col = col_side & shapely.contains_xy(cpoly.buffer(0.5 * h), P[:, 0], P[:, 1]) & ~is_old
    seamQ = Q[(kind == 's') & hs]
    seamQ = seamQ[np.linalg.norm(seamQ - Np, axis=1) > 0.15]          # (not the old gorge: that is inside the cloth now)
    seamQ = seamQ[((seamQ - Np) @ n2) * ((Q[hs] - Np) @ n2).mean() > 0.0 * h]
    on_seam = in_col & (cKDTree(seamQ).query(P)[0] < 0.45 * h)
    # gorge points of the collar's seam (X to G): within the gorge's band, they are the cloth now
    on_seam &= (shapely.distance(gl, shapely.points(P)) > 0.6 * h) & (np.linalg.norm(P - Gp, axis=1) > 0.4)
    cb = in_col & (np.abs(_bary(P, Q, cT, UV[:, :1])[:, 0]) < 0.35 * h)
    on_roll = shapely.distance(rline, shapely.points(P)) < 0.4 * h
    in_band = shapely.distance(rline, shapely.points(P)) < S.get('band', 0.6)      # (the turn's band: its hinges keep the turn as solved)
    # ---- the design (the start): the lapel's cloth from the build's folded lapel, the collar's from the solved collar,
    # the collar's start near the gorge moved to meet the lapel's top
    D = np.zeros((len(P), 3))
    D[~in_col] = _bary(P[~in_col], flat, T, built)
    D[in_col] = _bary(P[in_col], Q, cT, cpos)
    D[is_old] = built[used[is_old]]
    # the gap at the gorge: where the collar's start has its gorge against where the lapel's top lies
    gq = np.linspace(0, 1, 15)
    gpts = Np[None] + gq[:, None] * (Gp - Np)[None]
    on_lap = _bary(gpts, flat, T, built)
    on_colr = _bary(gpts, Q, cT, cpos)
    gap = on_lap - on_colr
    log.append('gorge: the collar\'s start stood %.2f to %.2f from the lapel\'s top (median %.2f)' % (
        np.linalg.norm(gap, axis=1).min(), np.linalg.norm(gap, axis=1).max(), np.median(np.linalg.norm(gap, axis=1))))
    dg = shapely.distance(gl, shapely.points(P))
    if S.get('design', 'lay') == 'gap':
        wgt = np.clip(1 - dg / S['gorge_blend'], 0, 1)
        wgt = wgt * wgt * (3 - 2 * wgt)
        jj = np.clip(np.round(np.clip(((P - Np) @ (Gp - Np)) / ((Gp - Np) @ (Gp - Np)), 0, 1) * 14).astype(int), 0, 14)
        m_ = in_col & ~on_seam & ~is_old
        D[m_] += (wgt[m_])[:, None] * gap[jj[m_]]
        lay_w = np.zeros(len(P))
    else:
        # (laid as a tailor lays the collar's front on the stand: near the old gorge the collar is the lapel's cloth
        # carried on, its fall turned over on the chest as the lapel's flap is and its stand standing as the front does;
        # further round it is the solved collar. The solved collar's front was the crumpled part: it was sewn to a
        # lapel top 1 cm lower than its own gorge.)
        ru = UV[(kind == 'r') & hs]
        ru = ru[np.argsort(ru[:, 0])]
        from scipy.spatial import cKDTree as _K
        def side_uv(X):
            j = np.clip(_K(ru).query(X)[1], 0, len(ru) - 2)
            a_, b_ = ru[j], ru[j + 1]
            return np.sign((b_[:, 0] - a_[:, 0]) * (X[:, 1] - a_[:, 1]) - (b_[:, 1] - a_[:, 1]) * (X[:, 0] - a_[:, 0]))
        seam_sd = np.median(side_uv(UV[(kind == 's') & hs]))
        fall_uv = (side_uv(UV) != seam_sd).astype(float)
        fall = _bary(P, Q, cT, fall_uv[:, None])[:, 0] > 0.5
        flapT = T[lap[T].all(1)]
        stripT_ = T[~lap[T].any(1)]
        ext = np.zeros((len(P), 3))
        mf = in_col & fall
        ms = in_col & ~fall
        ext[mf] = _bary(P[mf], flat, flapT, built)
        ext[ms] = _bary(P[ms], flat, stripT_, built)
        Rb = S.get('lay_r', 2.0)
        lay_w = np.clip(1 - dg / Rb, 0, 1)
        lay_w = lay_w * lay_w * (3 - 2 * lay_w)
        lay_w[~in_col] = 0
        D[in_col] = lay_w[in_col, None] * ext[in_col] + (1 - lay_w[in_col, None]) * D[in_col]
        on_seam &= lay_w < 0.05
        log.append("design: the collar's front laid on from the lapel over %.1f (%d points)" % (Rb, int((lay_w > 0).sum())))
    if S.get('develop', True):
        # the cut taken from the shape (as a pattern is taken off the stand): the new cloth's flat points moved so each
        # edge is as long as on the design, the build's lapel cut held as it is
        ed_ = np.unique(np.sort(np.concatenate([Tall[:, [0, 1]], Tall[:, [1, 2]], Tall[:, [2, 0]]]), 1), axis=0)
        L3 = np.linalg.norm(D[ed_[:, 0]] - D[ed_[:, 1]], axis=1)
        mov = (~is_old).astype(float)
        P0 = P.copy()
        for it in range(S.get('develop_iters', 600)):
            dv = P[ed_[:, 1]] - P[ed_[:, 0]]
            ln = np.maximum(np.linalg.norm(dv, axis=1), 1e-9)
            wa, wb = mov[ed_[:, 0]], mov[ed_[:, 1]]
            ws = np.maximum(wa + wb, 1e-9)
            corr = ((ln - L3) / ln / ws)[:, None] * dv
            acc = np.zeros_like(P)
            cnt = np.zeros(len(P))
            np.add.at(acc, ed_[:, 0], wa[:, None] * corr)
            np.add.at(acc, ed_[:, 1], -wb[:, None] * corr)
            np.add.at(cnt, ed_[:, 0], wa)
            np.add.at(cnt, ed_[:, 1], wb)
            P = P + 0.8 * acc / np.maximum(cnt, 1)[:, None]
        res_ = 100 * (np.linalg.norm(P[ed_[:, 0]] - P[ed_[:, 1]], axis=1) / L3 - 1)
        a, b, c = P[Tall[:, 0]], P[Tall[:, 1]], P[Tall[:, 2]]
        flip = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
        log.append("develop: the new cloth's cut moved up to %.2f; its edges against the design 5/95 %.1f %.1f %%, worst %.1f %.1f; %d triangles turned over"
                   % (np.linalg.norm(P - P0, axis=1).max(), *np.percentile(res_[~(is_old[ed_].all(1))], [5, 95]), res_[~(is_old[ed_].all(1))].min(), res_[~(is_old[ed_].all(1))].max(), int(flip.sum())))
    corners = dict(N=Np, G=Gp, P=Pp, notch_c=Q[np.nonzero((kind == 'e') & hs)[0][np.argmax(np.linalg.norm(Q[(kind == 'e') & hs] - Gp, axis=1))]])
    return dict(P=P, T=Tall, D=D, is_strip=is_strip, is_old=is_old, in_col=in_col, on_seam=on_seam, cb=cb, on_roll=on_roll, in_band=in_band, lay_w=lay_w, roll=np.vstack([flat[rp], roll_c]),
                piece=piece, flap=flap, cpoly=cpoly, strip=strip, Q=Q, cT=cT, flat=flat, T0=T, lap=lap, corners=corners,
                F=F, C=C, sg=sg, log=log, B=B, N=N)


def body_of(F, R):
    """The bodice (its lapels left out) and the skin near the collar and the lapel."""
    from scipy.spatial import cKDTree
    V = np.asarray(F['V'])
    Fq = F['F']
    lp = np.vstack([np.asarray(L['pos'])[np.asarray(L['lap'], bool)] for L in F['folded']])
    d, _ = cKDTree(lp).query(V)
    Nv = xf.vertex_normals(V, Fq)
    M = F['M']
    rad = np.array([V[:, 0], V[:, 1] - M['yc']]).T
    Nv = Nv * np.where(((Nv[:, :2] * rad).sum(1) < 0)[:, None], -1, 1)
    keep = d > 1e-4
    lo, hi = R.min(0) - 1.0, R.max(0) + 1.0
    box = ((V > lo) & (V < hi)).all(1) & keep
    sk = np.asarray(F['skinP'])
    skF = F['skinF']
    skN = xf.vertex_normals(sk, skF)
    rr = np.array([sk[:, 0], sk[:, 1] - M['yc']]).T
    skN = skN * np.where(((skN[:, :2] * rr).sum(1) < 0)[:, None], -1, 1)
    sbox = ((sk > lo) & (sk < hi)).all(1)
    return V[box], Nv[box], sk[sbox], skN[sbox]


def solve(Pt, S, log=print):
    from scipy.spatial import cKDTree
    t0 = time.time()
    P, T, D = Pt['P'], Pt['T'], Pt['D'].copy()
    n = len(P)
    bV, bN, sV, sN = body_of(Pt['F'], D)
    cloth = xf.MeshBody(bV, bN, gap=S['gap'], reach=1.0, tangent=0.1)
    skin = xf.MeshBody(sV, sN, gap=S['gap'], reach=1.0)
    lap_past = ~Pt['in_col'] & ~Pt['is_strip'] & ~Pt['in_band']

    def body(x):
        p, t = skin(x)
        p2, t2 = cloth(p.copy())
        t = t | t2
        p2[~(lap_past | Pt['in_col'])] = x[~(lap_past | Pt['in_col'])]
        t[~(lap_past | Pt['in_col'])] = False
        return p2, t
    pins = np.nonzero(Pt['is_strip'] | Pt['on_seam'])[0]
    flat3 = np.column_stack([P, np.zeros(n)])
    Hh = fo.hinges(T)
    rl = Pt['in_band']
    Hc = Hh[(rl[Hh[:, 0]] & rl[Hh[:, 1]])]
    dh = np.abs(fo.dihedral(D, Hc))
    Hc = Hc[dh > math.radians(3)]
    fab = xf.Fabric(D.copy(), dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=S['friction']))
    fab.prev[:] = fab.pos
    fab.pin(pins)
    Hc = Hc[(fab.w[Hc] > 0).any(1)]
    rest = fo.dihedral(D, Hc)
    comp = np.zeros(len(Hc))
    soft = np.zeros(n)
    soft[np.unique(Hc)] = 1e-2
    fab.mesh([tuple(int(i) for i in t) for t in T], stretch=S['stretch'], rest_pos=flat3, bend=S['bend'], flat=S['bend'], bend_at=soft)
    Fh = fo.Folds(fab, flat3, T)
    batches = fo._colour(Hc, fab.w == 0)
    hh = 1.0 / S['rate']
    fab.settings['damping'] = 0.35
    ed = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    l0 = np.linalg.norm(P[ed[:, 0]] - P[ed[:, 1]], axis=1)

    def strain(X):
        return 100 * (np.linalg.norm(X[ed[:, 0]] - X[ed[:, 1]], axis=1) / l0 - 1)
    s0 = strain(D)
    log('start: strain 5/95 %.1f %.1f worst %.1f %.1f' % (*np.percentile(s0, [5, 95]), s0.min(), s0.max()))
    for k in range(1, S['steps'] + 1):
        fab.step(hh, body, neutral=True)
        lam = np.zeros(len(Hc))
        for _ in range(4):
            Fh._walk(Hc, rest, comp, hh, batches, lam)
        p, t = body(fab.pos.copy())
        live = t & (fab.w > 0)
        fab.pos[live] = p[live]
        if k % 500 == 0:
            sn = strain(fab.pos)
            log('step %d fastest %.5f strain 5/95 %.1f %.1f worst %.1f %.1f' % (k, np.abs(fab.pos - fab.prev).max(), *np.percentile(sn, [5, 95]), sn.min(), sn.max()))
    X = fab.pos.copy()
    sn = strain(X)
    return dict(X=X, strain=sn, ed=ed, secs=round(time.time() - t0, 1), s0=s0)


def sheet_pattern(Pt, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P, T = Pt['P'], Pt['T']
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    a = ax[0]
    for g, c, lb in ((Pt['flap'], '#c9a0a0', 'lapel as the build cut it'), (Pt['cpoly'], '#a0b8d0', 'collar as tailor.py cut it')):
        xs, ys = g.exterior.xy
        a.fill(xs, ys, color=c, alpha=0.6, label=lb)
    xs, ys = Pt['strip'].exterior.xy
    a.fill(xs, ys, color='0.85', label='front (held)')
    xs, ys = Pt['piece'].exterior.xy
    a.plot(xs, ys, 'k-', lw=1.5, label='one-piece cut line')
    a.plot(*Pt['roll'].T, 'b--', lw=1.2, label='roll (fold) line')
    cn = Pt['corners']
    a.plot(*np.array([cn['N'], cn['G']]).T, 'r:', lw=2, label='old gorge seam (now inside the cloth)')
    for k_, v in cn.items():
        a.annotate(k_, v, fontsize=11)
    a.set_aspect('equal')
    a.legend(loc='lower left', fontsize=9)
    a.set_title('one-piece pattern, flat')
    b = ax[1]
    b.triplot(P[:, 0], P[:, 1], T, lw=0.15, color='0.5')
    for m, c in ((Pt['on_seam'], 'r'), (Pt['on_roll'], 'b'), (Pt['is_strip'], '0.7'), (Pt['cb'], 'g')):
        b.plot(P[m, 0], P[m, 1], '.', ms=2, color=c)
    b.set_aspect('equal')
    b.set_title('mesh: one sheet (red: neck seam held, blue: roll, grey: front held, green: CB)')
    fig.suptitle(title)
    plt.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def main():
    cmd = sys.argv[1]
    fold_pkl, collar_pkl, cap_dir, out = sys.argv[2:6]
    os.makedirs(out, exist_ok=True)
    S = dict(OP)
    if cmd == 'try' and len(sys.argv) > 6 and not sys.argv[6].startswith('--'):
        S.update(json.loads(sys.argv[6]))
    name = sys.argv[sys.argv.index('--name') + 1] if '--name' in sys.argv else 'op'
    ks = [int(sys.argv[sys.argv.index('--lapel') + 1])] if '--lapel' in sys.argv else [0, 1]
    for k in ks:
        Pt = pattern(fold_pkl, collar_pkl, cap_dir, k, S)
        for l_ in Pt['log']:
            print('lapel %d: %s' % (k, l_))
        print('lapel %d: one piece: %d points, %d triangles, area %.2f; seam held %d, roll %d' % (
            k, len(Pt['P']), len(Pt['T']), Pt['piece'].area, Pt['on_seam'].sum(), Pt['on_roll'].sum()))
        sheet_pattern(Pt, os.path.join(out, '%s-L%d-pattern.png' % (name, k)), 'one-piece notch collar and lapel, lapel %d' % k)
        if cmd == 'try':
            R = solve(Pt, S, log=lambda s: print('lapel %d: %s' % (k, s), flush=True))
            np.savez(os.path.join(out, '%s-L%d.npz' % (name, k)), X=R['X'], P=Pt['P'], T=Pt['T'], strain=R['strain'], D=Pt['D'],
                     in_col=Pt['in_col'], is_strip=Pt['is_strip'], on_seam=Pt['on_seam'], on_roll=Pt['on_roll'])
            print('lapel %d: solved in %.0f s' % (k, R['secs']))


if __name__ == '__main__':
    main()
