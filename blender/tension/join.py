"""The join between each lapel and the collar (the gorge seam and the notch), measured on a build:

    python3 blender/tension/join.py BUILD_DIR [OUT.json]

BUILD_DIR holds foldfront's LAPEL.blend.lapels.npz / .strain.npz and notchcollar's OUT.blend.collar.npz
(lapel.blend.*, out.blend.collar.npz, as blender/tension's sweeps name them). Per side:

  seam      the collar's boundary points that lie on the lapel's top edge (within `near` of it): the gorge
  gap       their distance to the lapel's top edge (the collar sits on the lapel: about a cloth's thickness, 4.4 mm)
  step      their height off the lapel's surface along its normal (the collar should sit on the lapel, a cloth's
            thickness at most: the bottom-edge step at the collar points of the lapel retrospective)
  turn      the angle between the collar's surface and the lapel's across the seam (a smooth join turns little;
            the pinch at the collar tip is a sharp turn at the seam's top)
  kink      the seam line's own sharpest bend, degrees per step (a smooth seam has no corner but the notch)
  load      the strain on each side of the seam, in a band `band` wide: the lapel's (foldfront's edge strain
            against the cut) and the collar's (its best flat cut, tension.flatten), the mean on each, their ratio,
            and the evenness along the seam: the seam cut into eight stretches, the coefficient of variation of each
            side's eight means, averaged over the two sides (0 is perfectly even; cv_pooled pools both sides, and so
            mostly measures their ratio), and peak_lapel, the lapel's worst stretch over its mean
  normals   for step and turn, a plane fitted to each side's cloth within `fit` of the seam point
  tip_fold  the collar's sharpest dihedral within `band` of the seam's top end (the collar-tip pinch)
"""
import sys, os, json
import numpy as np
from scipy.spatial import cKDTree, Delaunay
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tension as TS  # noqa: E402


def lapel_mesh(P, t, d, ln):
    cut = np.c_[t * ln, d]
    T = Delaunay(cut).simplices
    e = np.max([np.linalg.norm(cut[T[:, a]] - cut[T[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    return T[e < 0.25]


def boundary_vertices(T, n):
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    b = np.zeros(n, bool); b[u[c == 1].ravel()] = True
    return b, u[c == 1]


def face_normals(X, T):
    n = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


def plane_normals(X, Q, r):
    kd = cKDTree(X); out = np.zeros((len(Q), 3))
    for i, q in enumerate(Q):
        nb = kd.query_ball_point(q, r)
        if len(nb) < 4:
            nb = kd.query(q, 8)[1]
        A = X[nb] - X[nb].mean(0)
        out[i] = np.linalg.svd(A, full_matrices=False)[2][-1]
    return out


def measure(bdir, near=0.3, band=0.4, collar_flat=None, profile=False, fit=0.3):
    L = np.load(os.path.join(bdir, 'lapel.blend.lapels.npz'))
    S = np.load(os.path.join(bdir, 'lapel.blend.strain.npz'))
    C = np.load(os.path.join(bdir, 'out.blend.collar.npz'))
    CX, CT = C['pts'], C['tri'].astype(np.int64)
    keep = TS.areas(CX, CT) > 1e-9
    CT = CT[keep]
    if collar_flat is None:
        Xs, Ts, ids = TS.submesh(CX, CT, np.ones(len(CT), bool))
        U = TS.flatten(Xs, Ts, iters=200)
        s1, s2, _, _ = TS.principal(Xs, Ts, U)
        cst = np.where(s1 - 1 > 1 - s2, s1 - 1, s2 - 1)            # signed: the larger of the two departures
        ccen = Xs[Ts].mean(1)
    else:
        ccen, cst = collar_flat
    cfn = face_normals(CX, CT); ccen_all = CX[CT].mean(1)
    cb, cbe = boundary_vertices(CT, len(CX))
    out = {}
    for k in (1, 2):
        P = L['pts%d' % k]; t = L['t%d' % k]; d = L['d%d' % k]; ed = L['edge%d' % k].astype(bool); ln = float(L['ln%d' % k])
        LT = lapel_mesh(P, t, d, ln); lfn = face_normals(P, LT); lcen = P[LT].mean(1)
        # orient the lapel's normals outward (away from the body: the front is -y)
        lfn *= np.where(lfn[:, 1] < 0, 1, -1)[:, None]
        tp = t[np.argmax(d)]
        top = np.nonzero(ed & (t <= tp + 1e-6))[0]                 # the lapel's top edge: roll line to its point
        if len(top) < 3:
            continue
        kd = cKDTree(P[top])
        side = (CX[:, 0] < 0) if P[:, 0].mean() < 0 else (CX[:, 0] > 0)
        dist, j = kd.query(CX)
        seam = np.nonzero(cb & side & (dist < near))[0]
        if len(seam) < 3:
            out[k] = dict(seam_points=int(len(seam))); continue
        # the seam is one run of the collar's boundary: keep the longest connected run of boundary edges among the
        # points found (stray boundary points elsewhere, such as the collar's end over the other lapel, drop out)
        import scipy.sparse as sp_
        ins = np.zeros(len(CX), bool); ins[seam] = True
        ee = cbe[ins[cbe[:, 0]] & ins[cbe[:, 1]]]
        A = sp_.coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(len(CX), len(CX)))
        _, lab = sp_.csgraph.connected_components(A, directed=False)
        big = np.bincount(lab[seam]).argmax(); seam = seam[lab[seam] == big]
        # order it by walking the run, from the end nearest the roll line
        keep_s = set(seam.tolist()); nb_ = {v: [] for v in keep_s}
        for a_, b_ in ee:
            if a_ in keep_s and b_ in keep_s:
                nb_[a_].append(b_); nb_[b_].append(a_)
        ends = [v for v in keep_s if len(nb_[v]) == 1] or list(keep_s)
        start = min(ends, key=lambda v: t[top][j[v]])
        path, prev = [start], None
        while True:
            nx_ = [w for w in nb_[path[-1]] if w != prev and w not in path[-2:]]
            if not nx_ or len(path) > len(keep_s):
                break
            prev = path[-1]; path.append(nx_[0])
        seam = np.array(path)
        s_arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(CX[seam], axis=0), axis=1))])
        # step and turn
        # normals by a plane fitted to the cloth within `fit` of each seam point (the lapel's chart triangulation is
        # too uneven for face normals: neighbouring faces of the approved build differ by up to 180 degrees)
        _, lf = cKDTree(lcen).query(CX[seam])
        nl = plane_normals(P, CX[seam], fit); nl *= np.where(nl[:, 1] < 0, 1, -1)[:, None]
        step = ((CX[seam] - lcen[lf]) * nl).sum(1)
        nc = plane_normals(CX, CX[seam], fit)
        nc *= np.where((nc * nl).sum(1) < 0, -1, 1)[:, None]
        turn = np.degrees(np.arccos(np.clip((nc * nl).sum(1), -1, 1)))
        # kink of the seam line (after a light resampling to even steps)
        q = CX[seam]
        if len(q) > 4:
            ss = np.linspace(0, s_arc[-1], max(6, int(s_arc[-1] / 0.25)))   # (6.5 mm steps: finer reads the chart's triangle steps)
            q = np.column_stack([np.interp(ss, s_arc, CX[seam][:, a]) for a in range(3)])
            v = np.diff(q, axis=0); v /= np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
            kink = np.degrees(np.arccos(np.clip((v[1:] * v[:-1]).sum(1), -1, 1)))
        else:
            kink = np.zeros(1)
        # load on each side, binned along the seam
        st, mid = S['st%d' % k], S['mid%d' % k]
        dl, jl = cKDTree(CX[seam]).query(mid); inl = dl < band
        dc, jc = cKDTree(CX[seam]).query(ccen); inc = (dc < band) & ((ccen[:, 0] < 0) == (P[:, 0].mean() < 0))
        nb = 8
        def bins(jj, val):
            pos = s_arc[jj] / max(s_arc[-1], 1e-9)
            b_ = np.minimum((pos * nb).astype(int), nb - 1)
            return np.array([np.abs(val[b_ == i]).mean() if (b_ == i).any() else np.nan for i in range(nb)])
        bl = bins(jl[inl], st[inl]); bc = bins(jc[inc], cst[inc])
        both = np.concatenate([bl, bc]); both = both[np.isfinite(both)]
        cvl = np.nanstd(bl) / max(np.nanmean(bl), 1e-9); cvc = np.nanstd(bc) / max(np.nanmean(bc), 1e-9)
        tip = CX[seam[np.argmin(t[top][j[seam]])]]
        near_tip = np.nonzero(np.linalg.norm(ccen_all - tip, axis=1) < band)[0]
        import gauge_fold
        tf = gauge_fold.max_fold(CX, CT[near_tip]) if len(near_tip) > 2 else 0.0
        out[k] = dict(seam_points=int(len(seam)), seam_len_cm=float(2.62 * s_arc[-1]),
                      gap_mean_mm=float(26.2 * dist[seam].mean()), gap_max_mm=float(26.2 * dist[seam].max()),
                      step_mean_mm=float(26.2 * np.abs(step).mean()), step_max_mm=float(26.2 * np.abs(step).max()),
                      turn_mean=float(turn.mean()), turn_max=float(turn.max()),
                      kink_max=float(kink.max()), kink_p90=float(np.percentile(kink, 90)),
                      lapel_load=float(np.nanmean(bl) * 100), collar_load=float(np.nanmean(bc) * 100),
                      load_ratio=float(np.nanmean(bl) / max(np.nanmean(bc), 1e-9)),
                      evenness_cv=float((cvl + cvc) / 2), cv_lapel=float(cvl), cv_collar=float(cvc),
                      peak_lapel=float(np.nanmax(bl) / max(np.nanmean(bl), 1e-9)), turn_p90=float(np.percentile(turn, 90)),
                      cv_pooled=float(both.std() / max(both.mean(), 1e-9)), tip_fold=float(tf))
        if profile:
            out[k]['_prof'] = dict(s=s_arc / s_arc[-1], turn=turn, gap=dist[seam], kink=kink, tpos=t[top][j[seam]] / t[top].max(),
                                   bl=bl, bc=bc)
    return out


if __name__ == '__main__':
    cf = None
    g = os.path.join(sys.argv[1], 'collar_gauge.npz')
    if os.path.exists(g):                                     # (the collar's flat cut, already solved by the sweep)
        G_ = np.load(g); s1_, s2_ = G_['s1'], G_['s2']
        cf = (G_['X'][G_['T']].mean(1), np.where(s1_ - 1 > 1 - s2_, s1_ - 1, s2_ - 1))
    r = measure(sys.argv[1], collar_flat=cf)
    s = json.dumps({k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in r.items()})
    print(s)
    if len(sys.argv) > 2:
        open(sys.argv[2], 'w').write(s)
