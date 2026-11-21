"""The notch collar as a tailor cuts it: one flat pattern, sewn to the neck edge and the lapels' gorges, and folded on
the roll line in the solver.

The draft follows the notched collar draft of Fabrics-Store's sewing glossary (blog.fabrics-store.com/?p=13206) and
the University of Kentucky's custom tailoring sheet CT-LMH.176:

  the roll line   one straight line from the lapel's break point to a point one side-stand (1 inch: 0.49 w) up the
                  neck from the neck point at the shoulder; it crosses the bodice's neck edge at X (collar.roll_cross)
  the back neck   a line as long as the back neck edge, run on from the roll line's way and swung half an inch
                  (0.485) toward the shoulder at the centre back, so the outer edge is the longer
  the front neck  the bodice's own neck edge from the shoulder to X, laid flat with the bodice's cloth: its length and
                  its turn within the bodice's surface kept (the tailor draws the collar on the flat front; this
                  bodice is drafted on the body, so its neck edge is developed instead). The stand it leaves under
                  the roll line comes out 0.96 at the shoulder against the draft's 1 inch (0.97); a front neck taken
                  from the front view instead left 1.7 and the surplus crumpled on the body (strain -17/+30 %)
  the gorge       X to the lapel's notch, by the lapel's own cut length
  the outer edge  the stand and the fall's depth out from the back neck (2 3/4 inches at the centre back), then a
                  curve down to the collar's point, which leaves the notch's angle to the lapel

The start is not a shell: the seam, the roll line and the outer edge are set where the draft puts them, the cloth
between is filled in smoothly, the fronts near X are laid flat and turned over in the roll line, and the solver does
the rest against the body and the bodice (an open sheet: MeshBody(tangent=), solver-issues.md S5)."""
import hashlib
import json
import math
import os

import numpy as np

SETTINGS = dict(
    swing=0.485,           # the tailor's half inch at her scale (1 unit is about 2.62 cm)
    end=0.6,               # the collar's end, notch to point, in w
    notch=65,              # the notch's opening against the lapel's top edge, degrees
    lay_r=1.8,             # the fronts are laid flat and turned over within this of X
    kg_smooth=12,          # rounds of smoothing on the neck edge's measured turn
    tangent=1.5,           # no push further than this many bodice sample spacings past the bodice's edge
    rate=960, steps=2000,
    stretch=1e-8, bend=1e-9,
)


def tailor_collar(E, M, V, F, BV, BF, gorges, Pp, settings=None, log=print):
    """E: the bodice's neck edge (collar.neck_edge); V, F: the bodice with its lapels turned back; BV, BF: the body;
    gorges: {side: (X in (x, z), the notch G in (x, z), X in space)}; Pp: her right lapel's point in (x, z).
    Returns dict(UV: the flat pattern, T, pos: solved, des: the start, strain: % per edge, ed, kind: per point
    's' seam, 'r' roll line, 'o' outer edge, 'e' ends, 'i' inside, report)."""
    import shapely
    from scipy.spatial import Delaunay, cKDTree
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    import fabric as xf
    import folds as fo
    P_ = dict(SETTINGS, line_only=1)
    P_.update(settings or {})
    V, BV = np.asarray(V, float), np.asarray(BV, float)
    Pp = np.asarray(Pp, float)
    import collar as C
    R, S0, L_ = C.RATIOS, C.SETTINGS, C.LAPEL
    w = M['w']; n = len(E); up = np.array([0, 0, 1.0])
    st, ss_ = R['stand'] * w, L_['roll_side'] * w
    fb = R['fall_back'] * w
    h = R['spacing'] * w
    SWING = P_['swing']          # the tailor's half inch at her scale
    LEND = P_['end'] * w           # the collar's end, notch to point
    NANG = math.radians(P_['notch'])
    FOLD = P_.get('fold', 1.25); LEAN = 0.18
    BN = xf.vertex_normals(BV, BF)
    vt = cKDTree(V[:, [0, 2]]); sel = BV[:, 2] > 44; skt = cKDTree(BV[sel][:, [0, 2]])

    def front_y(xz, off):
        near = vt.query_ball_point(xz, 0.14)
        if near: return V[near, 1].min() - off
        near = skt.query_ball_point(xz, 0.25)
        return (BV[sel][near, 1].min() - off) if near else None

    def resample(Pn, step):
        a = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pn, axis=0), axis=1))])
        m = max(2, int(round(a[-1] / step)) + 1)
        u = np.linspace(0, a[-1], m)
        return np.column_stack([np.interp(u, a, Pn[:, k]) for k in range(Pn.shape[1])]), u

    # ---------------- the right half (sg = -1), flat on the front (x, z), unfolded
    sg = -1
    X, G, X3 = gorges[sg]
    C3 = C.side_roll(E, M, sg); Cx = C3[[0, 2]]
    rh = (Cx - X) / np.linalg.norm(Cx - X)
    nm = np.array([-rh[1], rh[0]]);  nm = -nm if nm[0] * sg > 0 else nm      # medial
    mir = lambda q: q - 2 * ((q - X) @ nm)[..., None] * nm
    # the 3D seam of this side: back (CB to A), front neck (A to X), gorge (X to G on the lapel's face)
    kX = int(np.floor(n * math.atan2(abs(X3[0]), -(X3[1] - M['yc'])) / (2 * np.pi))) + 1
    back3 = E[n // 2:n // 4 - 1:-1]                        # CB (k=60) down to A (k=30)
    front3 = np.vstack([E[n // 4:kX - 1:-1], X3])           # A to X
    gor3 = []
    for q in np.linspace(0, 1, 13)[1:]:
        xz = X + q * (G - X)
        gor3.append([xz[0], front_y(xz, S0['gap'] + 0.02), xz[1]])
    gor3 = np.vstack([X3, np.array(gor3)])
    LB = np.linalg.norm(np.diff(back3, axis=0), axis=1).sum()
    LF = np.linalg.norm(np.diff(front3, axis=0), axis=1).sum()
    LG = np.linalg.norm(np.diff(gor3, axis=0), axis=1).sum()
    seam3 = np.vstack([back3, front3[1:], gor3[1:]])
    a3 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seam3, axis=0), axis=1))])
    seam3_at = lambda s: np.column_stack([np.interp(s, a3, seam3[:, k]) for k in range(3)])
    # flat front neck, X back to A: the neck edge developed with the bodice's own cloth, i.e. laid flat keeping its length
    # and its turn within the bodice's surface (its geodesic curvature), as the tailor has it on the flat front
    Nb = xf.vertex_normals(V, F); radb = np.array([V[:, 0], V[:, 1] - M['yc']]).T
    Nb = Nb * (-1 if (Nb[:, :2] * radb).sum(1).mean() < 0 else 1)
    fr = front3[::-1]
    af = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(fr, axis=0), axis=1))])
    uf = np.linspace(0, af[-1], 60)
    Pc = np.column_stack([np.interp(uf, af, fr[:, k]) for k in range(3)])
    for _ in range(3): Pc[1:-1] = 0.25 * Pc[:-2] + 0.5 * Pc[1:-1] + 0.25 * Pc[2:]
    kdb = cKDTree(V)
    Nn = np.array([Nb[kdb.query(p_, k=12)[1]].mean(0) for p_ in Pc]); Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    Tc = np.gradient(Pc, axis=0); dsc = np.linalg.norm(Tc, axis=1); Tc /= dsc[:, None]
    kg = ((np.gradient(Tc, axis=0) / dsc[:, None]) * np.cross(Nn, Tc)).sum(1)
    for _ in range(P_.get('kg_smooth', 12)): kg[1:-1] = 0.25 * kg[:-2] + 0.5 * kg[1:-1] + 0.25 * kg[2:]
    kg *= P_.get('kg_scale', 1.0)
    d0 = (fr[1] - fr[0])[[0, 2]]; th0 = math.atan2(d0[1], d0[0])
    # (the turn's sense in the flat: toward the neck, as the front view shows it)
    d1v = (fr[-1] - fr[-2])[[0, 2]]; sense = np.sign(np.arctan2(d0[0] * d1v[1] - d0[1] * d1v[0], d0 @ d1v))
    turn = np.concatenate([[0], np.cumsum(0.5 * (np.abs(kg[1:]) * 0 + kg[1:] + kg[:-1]) * np.diff(uf))])
    th = th0 + sense * np.abs(turn) * 0 + sense * np.abs(np.cumsum(np.concatenate([[0], 0.5 * (np.abs(kg[1:]) + np.abs(kg[:-1])) * np.diff(uf)]))) * 0 + sense * np.abs(turn)
    fl = np.vstack([X, X + np.cumsum(np.column_stack([np.cos(th[:-1]), np.sin(th[:-1])]) * np.diff(uf)[:, None], axis=0)])
    log('collar: tailor: ' + 'front neck developed: turns %.0f deg over %.2f (the front view alone: %.0f deg)' % (math.degrees(abs(turn[-1])), uf[-1], math.degrees(abs(np.arctan2(d0[0] * d1v[1] - d0[1] * d1v[0], d0 @ d1v)))))
    A2 = fl[-1]
    bh = rh.copy()                                          # the back neck: up from A along the roll line's way, swung
    for _ in range(3):
        nb0 = np.array([-bh[1], bh[0]]); nb0 = nb0 if nb0 @ nm > 0 else -nb0
        c2_ = A2 + ss_ * nb0
        bh = (c2_ - X) / np.linalg.norm(c2_ - X)
    lat = -nm
    u = np.linspace(0, LB, 40)
    back2 = A2[None] + u[:, None] * bh[None] + (SWING * (u / LB) ** 2)[:, None] * lat[None]
    tb = np.gradient(back2, axis=0); tb /= np.linalg.norm(tb, axis=1, keepdims=True)
    nb = np.column_stack([-tb[:, 1], tb[:, 0]]); nb = np.where((nb @ nm > 0)[:, None], nb, -nb)
    s_of_u = LB - u                                         # arc from CB
    stp = ss_ + (st - ss_) * (1 - (s_of_u / LB) ** 2)
    roll_back2 = back2 + stp[:, None] * nb
    depth = stp + fb
    out_back2 = back2 + depth[:, None] * nb
    Gm = mir(G)
    v = (Pp - G) / np.linalg.norm(Pp - G)
    cands = [np.array([v[0] * math.cos(t_) - v[1] * math.sin(t_), v[0] * math.sin(t_) + v[1] * math.cos(t_)]) for t_ in (NANG, -NANG)]
    Jf = G + LEND * max(cands, key=lambda q: q[1])
    J = mir(Jf)
    # outer edge: the back's offset, then a curve down to J
    P0 = out_back2[0]; ctrl = P0 - bh * 0.5 * np.linalg.norm(P0 - J)
    bq = np.linspace(0, 1, 30)[1:]
    bez = ((1 - bq) ** 2)[:, None] * P0 + (2 * bq * (1 - bq))[:, None] * ctrl + (bq ** 2)[:, None] * J
    H2, I2 = back2[-1], out_back2[-1]
    seam2 = np.vstack([back2[::-1], fl[::-1][1:], Gm[None]])   # CB(H) -> A -> X -> G'
    half = np.vstack([seam2, J[None], bez[::-1][1:], out_back2[1:]])
    poly = shapely.Polygon(half)
    log('collar: tailor: ' + 'stand between the front neck and the roll line: widest %.2f' % max(abs(((q_ - X)[0] * bh[1] - (q_ - X)[1] * bh[0])) for q_ in fl))
    log('collar: tailor: ' + 'half pattern: valid %s, area %.2f; back neck %.2f, front neck %.2f, gorge %.2f; A2 %s, H %s, I %s, J %s' % (poly.is_valid, poly.area, LB, LF, LG, A2.round(2), H2.round(2), I2.round(2), J.round(2)))
    roll2 = np.vstack([roll_back2[::-1], X[None]])            # Hr -> C2 -> X
    # pattern frame: CB line as the y axis
    e2 = (I2 - H2) / np.linalg.norm(I2 - H2); e1 = np.array([e2[1], -e2[0]])
    if (A2 - H2) @ e1 < 0: e1 = -e1
    toP = lambda q: np.column_stack([(q - H2) @ e1, (q - H2) @ e2])
    # boundary and feature samples of the right half, with what each is
    seam_s, su = resample(seam2, 0.7 * h)                    # su = arc from CB (pattern), equals the 3D arc by construction
    a2 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seam2, axis=0), axis=1))])
    log('collar: tailor: ' + 'pattern seam %.3f, cloth seam %.3f' % (a2[-1], a3[-1]))
    kA, kXv = len(back2) - 1, len(back2) + len(fl) - 2
    br2 = np.array([0, a2[kA], a2[kXv], a2[-1]]); br3 = np.array([0, LB, LB + LF, LB + LF + LG])
    su3 = np.interp(su, br2, br3)
    log('collar: tailor: ' + 'pattern back/front/gorge %.2f %.2f %.2f; cloth %.2f %.2f %.2f' % (*np.diff(br2), *np.diff(br3)))
    roll_s, _ = resample(roll2, 0.7 * h)
    end_s, eu = resample(np.vstack([Gm, J]), 0.7 * h)
    outer = np.vstack([J[None], bez[::-1][1:], out_back2[1:]])
    out_s, _ = resample(outer, 0.7 * h)
    ring = shapely.Polygon(half).buffer(-0.5 * h)
    bx = poly.bounds
    gx, gy = np.meshgrid(np.arange(bx[0], bx[2], h), np.arange(bx[1], bx[3], h * 0.866))
    gx[1::2] += 0.5 * h
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid = grid[shapely.contains_xy(ring, grid[:, 0], grid[:, 1])]
    rl = shapely.LineString(roll2)
    grid = grid[shapely.distance(rl, shapely.points(grid)) > 0.45 * h]
    roll_in = roll_s[1:-1][shapely.contains_xy(ring, roll_s[1:-1, 0], roll_s[1:-1, 1])]
    # ---------------- 3D targets (right side)
    def neck3(s):                                           # the round-the-neck frame at arc s from CB (s <= LB)
        # the pattern's swing makes each line across the band longer than the seam by kap per unit of height: the stand
        # as cut stands off the neck by that much, and the fall hangs outside it (the shape the flat pattern can take)
        Eo = seam3_at(s)
        o = np.column_stack([Eo[:, 0], Eo[:, 1] - M['yc'], 0 * s]); rr = np.linalg.norm(o, axis=1); o /= rr[:, None]
        g = np.clip(2 * SWING / LB ** 2 * rr, 0, 0.5)[:, None]
        d1 = up[None] * np.sqrt(1 - g * g) + o * g
        sp = ss_ + (st - ss_) * (1 - np.clip(s / LB, 0, 1) ** 2)
        return Eo, Eo + sp[:, None] * d1, o * g - up[None] * np.sqrt(1 - g * g), sp
    skin = xf.MeshBody(BV, BN, gap=0.2, reach=1.0)
    def roll3(q2):                                          # a point of the pattern's roll line, in space
        out = np.zeros((len(q2), 3))
        d = shapely.line_locate_point(rl, shapely.points(q2))     # from Hr
        Lr = np.linalg.norm(np.diff(roll_back2[::-1], axis=0), axis=1).sum()
        bk = d <= Lr
        out[bk] = neck3(d[bk] / Lr * LB)[1]
        Ca = neck3(np.array([LB]))[1][0]
        q = np.clip((d[~bk] - Lr) / max(rl.length - Lr, 1e-9), 0, 1)[:, None]
        out[~bk] = Ca + q * (X3 - Ca)
        pr, hit = skin(out.copy()); hit &= np.linalg.norm(out - X3, axis=1) > 0.4
        out[hit] = pr[hit]
        return out
    def lay(q2):                                            # a point of the fall at the front: turned over in the roll line, on the cloth
        qf = mir(q2)
        y = np.array([front_y(z_, S0['gap'] + 0.05) or -3.0 for z_ in qf])
        return np.column_stack([qf[:, 0], y, qf[:, 1]])
    def outer3(q2):
        out = np.zeros((len(q2), 3))
        ol = shapely.LineString(outer)
        d = shapely.line_locate_point(ol, shapely.points(q2))     # from J
        Lz = np.linalg.norm(np.diff(np.vstack([J[None], bez[::-1][1:], out_back2[:1]]), axis=0), axis=1).sum()
        Lb_ = ol.length - Lz
        bk = d >= Lz
        s_ = LB * (1 - np.clip((d[bk] - Lz) / Lb_, 0, 1))
        Eo, Ro, bd, sp = neck3(s_)
        out[bk] = Ro + fb * bd
        beta = np.clip(1 - d[~bk] / Lz, 0, 1); beta = (beta * beta * (3 - 2 * beta))[:, None]     # 0 at J, 1 at the back's offset
        Eo, Ro, bd, sp = neck3(np.array([LB])); sh = (Ro + fb * bd)[0]
        out[~bk] = (1 - beta) * lay(q2[~bk]) + beta * sh
        return out
    tg_seam = seam3_at(su3)
    gm_ = su > a2[kXv] + 1e-9                               # (the gorge: by its own length on the lapel's face, as the lapel's edge is cut)
    qg_ = (su[gm_] - a2[kXv]) / (a2[-1] - a2[kXv])
    xz_ = X[None] + qg_[:, None] * (G - X)[None]
    tg_seam[gm_] = np.column_stack([xz_[:, 0], [front_y(z_, S0['gap'] + 0.02) for z_ in xz_], xz_[:, 1]])
    tg_seam[(su3 > LB + LF + 0.02) & (su3 < LB + LF + 0.3 * LG)] = np.nan      # (the gorge's first third is free: the seam steps there from the neck edge onto the lapel's face)
    tg_roll = roll3(roll_in)
    tg_out = outer3(out_s)
    e0, e1_ = np.array([G[0], front_y(G, S0['gap'] + 0.02), G[1]]), lay(J[None])[0]
    tg_end = e0[None] + (eu / eu[-1])[:, None] * (e1_ - e0)[None]
    # ---------------- both halves
    def both(P2, T3, tag):
        Pr = toP(P2); Pl = Pr * np.array([-1, 1]); Tl = T3 * np.array([-1, 1, 1])
        return np.vstack([Pr, Pl]), np.vstack([T3, Tl]), [tag] * (2 * len(P2))
    pts, tgt, kind = [], [], []
    for P2, T3, tag in ((seam_s, tg_seam, 's'), (roll_in, tg_roll, 'r'), (out_s, tg_out, 'o'), (end_s, tg_end, 'e')):
        a, b, c = both(P2, T3, tag); pts.append(a); tgt.append(b); kind += c
    gp = toP(grid); gp = np.vstack([gp, gp * np.array([-1, 1])])
    pts.append(gp); tgt.append(np.full((len(gp), 3), np.nan)); kind += ['i'] * len(gp)
    pts, tgt, kind = np.vstack(pts), np.vstack(tgt), np.array(kind)
    # merge duplicates (the CB line, shared corners)
    key = np.round(pts / (0.2 * h)).astype(int)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = {'s': 0, 'e': 1, 'o': 2, 'r': 3, 'i': 4}
    best = {}
    for i, g in enumerate(inv.ravel()):
        if g not in best or order[kind[i]] < order[kind[best[g]]]: best[g] = i
    keep = np.array([best[g] for g in range(len(first))])
    UV, tgt, kind = pts[keep], tgt[keep], kind[keep]
    # on the CB line the two halves' targets are averaged to x = 0
    cb = np.abs(UV[:, 0]) < 0.1 * h
    tgt[cb & ~np.isnan(tgt[:, 0]), 0] = 0.0
    full = shapely.unary_union([shapely.Polygon(toP(half)), shapely.Polygon(toP(half) * np.array([-1, 1]))]).buffer(1e-3)
    tri = Delaunay(UV).simplices
    cen = UV[tri].mean(1)
    ar = 0.5 * np.abs((UV[tri[:, 1], 0] - UV[tri[:, 0], 0]) * (UV[tri[:, 2], 1] - UV[tri[:, 0], 1]) - (UV[tri[:, 1], 1] - UV[tri[:, 0], 1]) * (UV[tri[:, 2], 0] - UV[tri[:, 0], 0]))
    T = tri[shapely.contains_xy(full, cen[:, 0], cen[:, 1]) & (ar > 0.02 * h * h)]
    used = np.unique(T); remap = -np.ones(len(UV), int); remap[used] = np.arange(len(used))
    UV, tgt, kind, T = UV[used], tgt[used], kind[used], remap[T]
    a_, b_, c_ = UV[T[:, 0]], UV[T[:, 1]], UV[T[:, 2]]
    cr = (b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1]) - (b_[:, 1] - a_[:, 1]) * (c_[:, 0] - a_[:, 0])
    T[cr < 0] = T[cr < 0][:, [0, 2, 1]]
    ed = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    N_ = len(UV)
    log('collar: tailor: ' + 'pattern: %d points, %d triangles; seam %d, roll %d, outer %d, ends %d' % (N_, len(T), (kind == 's').sum(), (kind == 'r').sum(), (kind == 'o').sum(), (kind == 'e').sum()))
    # the left half's own seam (its gorge on its own lapel): mirrored right targets are a start; seam points snap to the left's cloth
    left = (UV[:, 0] < -0.1 * h)
    for i in np.nonzero(left & (kind == 's'))[0]:
        p = tgt[i]
        if abs(p[0]) < abs(gorges[1][2][0]) - 0.05 or p[2] < gorges[1][2][2] - 0.05:
            y = front_y(p[[0, 2]], S0['gap'] + 0.02)
            if y is not None: tgt[i, 1] = y
    # ---------------- the start: harmonic between the seam, the roll and the outer edge
    known = ~np.isnan(tgt[:, 0])
    Lm = lil_matrix((N_, N_))
    for a, b in ed:
        Lm[a, b] = Lm[b, a] = -1.0
    Lm = csr_matrix(Lm); deg = -np.asarray(Lm.sum(1)).ravel()
    Lm = Lm + csr_matrix((deg, (np.arange(N_), np.arange(N_))), shape=(N_, N_))
    fr_, kn_ = np.nonzero(~known)[0], np.nonzero(known)[0]
    des = tgt.copy()
    des[fr_] = np.column_stack([spsolve(Lm[fr_][:, fr_].tocsc(), -Lm[fr_][:, kn_] @ tgt[kn_, k]) for k in range(3)])
    # the fronts near X as the tailor has them: flat on the front, and what lies past the roll line turned over in it
    RL = P_.get('lay_r', 1.8)
    q2 = H2[None] + np.abs(UV[:, :1]) * e1[None] + UV[:, 1:2] * e2[None]
    rgt = UV[:, 0] >= 0
    med = ((q2 - X) @ nm > 0.02) & (np.linalg.norm(q2 - X, axis=1) < RL) & (kind == 'i')
    stv = np.nonzero(((q2 - X) @ nm <= 0.02) & rgt)[0]
    kst = cKDTree(q2[stv])
    solve_h = lambda kn: np.column_stack([spsolve(Lm[~kn][:, ~kn].tocsc(), -Lm[~kn][:, kn] @ des[kn, k]) for k in range(3)])
    if med.any():
        iv = np.nonzero(med)[0]
        qf = mir(q2[iv])
        dn, jn = cKDTree(des[stv][:, [0, 2]]).query(qf)
        y_st = np.where(dn < 0.25, des[stv][jn, 1] - 0.15, 1e9)
        y_cl = np.array([front_y(z_, S0['gap'] + 0.05) or 1e9 for z_ in qf])
        y = np.minimum(y_st, y_cl); y = np.where(y > 1e8, des[iv, 1], y)
        des[iv] = np.column_stack([np.where(rgt[iv], qf[:, 0], -qf[:, 0]), y, qf[:, 1]])
        kn2 = known | med
        des[~kn2] = solve_h(kn2)
        log('collar: tailor: ' + 'laid flat and turned over near X: %d points' % len(iv))
    flat3 = np.column_stack([UV, np.zeros(N_)])
    strain = lambda Xx: 100 * (np.linalg.norm(Xx[ed[:, 0]] - Xx[ed[:, 1]], axis=1) / np.linalg.norm(UV[ed[:, 0]] - UV[ed[:, 1]], axis=1) - 1)
    s0 = strain(des)
    log('collar: tailor: ' + 'start shape against the pattern: strain 5/95 %.1f %.1f, worst %.1f %.1f' % (*np.percentile(s0, [5, 95]), s0.min(), s0.max()))
    # ---------------- the body and the solve
    Nv = xf.vertex_normals(V, F); rad = np.array([V[:, 0], V[:, 1] - M['yc']]).T
    Nv = Nv * (-1 if (Nv[:, :2] * rad).sum(1).mean() < 0 else 1)     # (one sign for the whole mesh: a per-point flip is a coin toss where the surface faces up; solver-issues.md S5)
    bodyP, bodyN = np.vstack([V, BV[sel]]), np.vstack([Nv, BN[sel]])
    box = (bodyP[:, 2] > des[:, 2].min() - 2.5) & (bodyP[:, 2] < des[:, 2].max() + 1.5)
    nv_ = len(V)
    bx_c, bx_s = box[:nv_], box[nv_:]
    sp_ = float(np.median(cKDTree(V[bx_c]).query(V[bx_c], k=2)[0][:, 1]))
    tan_ = P_.get('tangent', 1.5) * sp_
    cloth_b = xf.MeshBody(V[bx_c], Nv[bx_c], gap=S0['gap'], reach=1.0, tangent=tan_)     # (the bodice: an open sheet, no wall past its edges: PR #78)
    skin_b = xf.MeshBody(BV[sel][bx_s], BN[sel][bx_s], gap=S0['gap'], reach=1.0)
    log('collar: tailor: ' + 'bodice samples %.3f apart: no push further than %.3f past its edge' % (sp_, tan_))
    def body(x):
        p, t = skin_b(x) if not P_.get('no_skin') else (x.copy(), np.zeros(len(x), bool))
        if P_.get('no_cloth'): return p, t
        p2, t2 = cloth_b(p.copy())
        return p2, t | t2
    # The stand lies between the neck edge and the neck: it meets the body only. The bodice's cloth is a wall for the fall
    # alone (a stand point just above the bodice's rim is 'behind' the rim's face and was thrown forward every step).
    if P_.get('stand_skin', False):
        stand_poly = shapely.Polygon(np.vstack([seam2[:kXv + 1], roll2[::-1][1:]])).buffer(0.02)
        q2b = H2[None] + np.abs(UV[:, :1]) * e1[None] + UV[:, 1:2] * e2[None]
        in_stand = shapely.contains_xy(stand_poly, q2b[:, 0], q2b[:, 1])
        skin_only = xf.MeshBody(BV[sel], BN[sel], gap=S0['gap'], reach=1.0)
        whole_body = body
        def body(x):
            p, t = x.copy(), np.zeros(len(x), bool)
            p[in_stand], t[in_stand] = skin_only(x[in_stand])
            p[~in_stand], t[~in_stand] = whole_body(x[~in_stand])
            return p, t
        log('collar: tailor: ' + 'stand: %d points meet the body only' % in_stand.sum())
    # (the seam sits on the cloth, not in it: every set point is put the cloth's gap off whatever is under it)
    pq, hq = body(des.copy())
    mv_ = np.where(hq[:, None], pq - des, 0.0)
    sv_ = np.nonzero(kind == 's')[0]                      # (the seam moved evenly: each point by the mean of its neighbours along it)
    jn_ = cKDTree(UV[sv_]).query(UV[sv_], k=9)[1]
    mv_[sv_] = mv_[sv_][jn_].mean(1)
    des = des + mv_
    Hh = fo.hinges(T)
    rollv = kind == 'r'
    on = rollv[Hh[:, 0]] & rollv[Hh[:, 1]] if P_.get('line_only') else (rollv[Hh[:, :2]].any(1))
    if P_.get('nohinge'): on = on & False
    pins0 = np.nonzero((kind == 's') & known)[0]
    front_out = np.nonzero((kind == 'o') & (np.linalg.norm(des[:, [0, 2]] - np.where(des[:, :1] < 0, X, X * np.array([-1, 1])), axis=1) < 3.0) | (kind == 'e'))[0]
    def run(start, pins, stretch, bend, steps, tag, nobody=False, guide=0.0):
        fab = xf.Fabric(start.copy(), dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=P_.get('friction', 0.4)))
        fab.prev[:] = fab.pos; fab.pin(pins)
        Hc = Hh[on & (fab.w[Hh] > 0).any(1)]
        rest = fo.dihedral(des, Hc); comp = np.zeros(len(Hc))
        soft = np.zeros(N_); soft[np.unique(Hc)] = 1e-2
        fab.mesh([tuple(int(i) for i in t) for t in T], stretch=stretch, rest_pos=flat3, bend=bend, flat=bend, bend_at=soft)
        Fh = fo.Folds(fab, flat3, T); batches = fo._colour(Hc, fab.w == 0)
        hh = 1.0 / P_.get('rate', 960); fab.settings['damping'] = 0.35
        for k in range(1, steps + 1):
            nb_ = nobody or P_.get('nobody')
            fab.step(hh, None if nb_ else body, neutral=True)
            if guide:                                       # (the hand: the free cloth drawn toward the set shape, less and less)
                gk = guide * (1 - k / steps)
                fr = fab.w > 0
                fab.pos[fr] += gk * (des[fr] - fab.pos[fr])
            lam = np.zeros(len(Hc))
            for _ in range(4): Fh._walk(Hc, rest, comp, hh, batches, lam)
            if not nb_:
                p, t = body(fab.pos.copy()); live = t & (fab.w > 0); fab.pos[live] = p[live]
            if k % 500 == 0:
                sn = strain(fab.pos)
                log('collar: tailor: ' + '%s step %d fastest %.5f strain 5/95 %.1f %.1f worst %.1f %.1f' % (tag, k, np.abs(fab.pos - fab.prev).max(), *np.percentile(sn, [5, 95]), sn.min(), sn.max()))
        return fab.pos.copy()
    hh_ = hashlib.sha1()
    for arr in (UV, T, des, V[bx_c], BV[sel][bx_s]):
        hh_.update(np.ascontiguousarray(np.round(np.asarray(arr, float), 5)).tobytes())
    hh_.update(json.dumps(P_, sort_keys=True, default=str).encode())
    cfile = os.path.join(fo.cache_dir(), 'tailor-collar-%s.npz' % hh_.hexdigest()[:20])
    if os.path.exists(cfile) and not os.environ.get('COLLAR_NO_CACHE'):
        got = np.load(cfile, allow_pickle=True)
        pos, rep = got['pos'], str(got['report']) + ' (from the fold cache)'
    else:
        pos = run(des, pins0, P_['stretch'], P_['bend'], P_['steps'], 'sewn')
        sn = strain(pos)
        j = cKDTree(UV).query(UV * np.array([-1, 1]))[1]
        asym = np.linalg.norm(pos - pos[j] * np.array([-1, 1, 1]), axis=1)
        rep = ('%d points; strain against the pattern 5/95 %% %.1f %.1f, worst %.1f %.1f; left/right difference median %.3f, max %.3f'
               % (N_, *np.percentile(sn, [5, 95]), sn.min(), sn.max(), np.median(asym), asym.max()))
        np.savez(cfile, pos=pos, report=rep)
    log('collar: tailor: ' + rep)
    return dict(UV=UV, T=T, pos=pos, des=des, strain=strain(pos), ed=ed, kind=kind, st_pt=np.zeros(N_), report=rep)
