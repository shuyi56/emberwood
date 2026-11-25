"""The compression (tension) solver: what a flat cloth is asked to do to take a garment's 3D shape, as a field.

A tailor cuts flat cloth. Wherever the shape it must take is doubly curved, the best flat cut still has to stretch
the cloth in some places (tension) and gather it in others (compression: the excess a tailor shrinks away with the
iron or takes up in a tape or a dart). This module finds that best flat cut and reads the field off it:

  flatten     the least-distortion flat pattern of a 3D piece: the as-rigid-as-possible parameterisation of
              Liu, Zhang, Xu, Gotsman & Gortler 2008 ("A local/global approach to mesh parameterization"): the local
              step fits each triangle's nearest rotation (SVD of its Jacobian), the global step solves the
              cotangent-Laplacian system, factored once. It is the energy-model flattening used for garment patterns
              (Wang, Smith & Yuen 2002, "Surface flattening based on energy model", CAD 34), with each triangle's
              squared distance from an isometry as the energy.
  principal   each triangle's two principal stretches (singular values of the flat-to-3D Jacobian) and their
              directions on the 3D surface. s > 1: the cloth is pulled longer than it was cut (tension);
              s < 1: there is more cloth than room (compression, the bunching a fold or a shrink takes up).
              By tension field theory (Wong & Pellegrino 2006, "Wrinkled membranes" part II) wrinkles run along the
              major (tension) direction where the minor stress is compressive; a turned fold or roll runs the same way.
  release     the same piece re-flattened with a cut (a free edge) along a line: what opening the front there
              gives back.
  turn        a strip folded over a curve on the flat pattern and laid on the piece's own surface one cloth's thickness
              out (the lapel turned back on its front): the strain the turned cloth must take.

Pure numpy / scipy. No Blender (see read_bodice.py for the reader). Meant to move into the shared solvers folder;
nothing here depends on the collar scripts.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def triangles(F):
    T = []
    for f in F:
        f = [int(v) for v in f if v >= 0]
        for i in range(1, len(f) - 1):
            T.append((f[0], f[i], f[i + 1]))
    return np.array(T, dtype=np.int64)


def submesh(X, T, keep_tri):
    """The triangles kept, reindexed; (X, T, original vertex ids). Only the largest edge-connected piece."""
    T = T[keep_tri]
    # largest connected component
    n = len(X)
    E = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    nc, lab = sp.csgraph.connected_components(A, directed=False)
    used = np.unique(T)
    cnt = np.bincount(lab[used], minlength=nc)
    big = np.argmax(cnt)
    T = T[lab[T[:, 0]] == big]
    ids = np.unique(T)
    remap = -np.ones(n, np.int64); remap[ids] = np.arange(len(ids))
    return X[ids], remap[T], ids


def areas(X, T):
    return 0.5 * np.linalg.norm(np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), axis=1)


def local_frames(X, T):
    """Each 3D triangle laid flat in its own frame: (P (n,3,2) the corners, ex, ey the frame's axes in 3D)."""
    e1 = X[T[:, 1]] - X[T[:, 0]]; e2 = X[T[:, 2]] - X[T[:, 0]]
    ex = e1 / np.linalg.norm(e1, axis=1)[:, None]
    nr = np.cross(e1, e2); nr /= np.maximum(np.linalg.norm(nr, axis=1), 1e-15)[:, None]
    ey = np.cross(nr, ex)
    P = np.zeros((len(T), 3, 2))
    P[:, 1, 0] = np.linalg.norm(e1, axis=1)
    P[:, 2, 0] = (e2 * ex).sum(1); P[:, 2, 1] = (e2 * ey).sum(1)
    return P, ex, ey, nr


def cot_weights(P):
    """cot of the angle opposite each edge (i, i+1) of each triangle, from its flat frame."""
    C = np.zeros((len(P), 3))
    for k in range(3):
        a, b, c = P[:, k], P[:, (k + 1) % 3], P[:, (k + 2) % 3]      # edge a-b, opposite c
        u, v = a - c, b - c
        cr = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        C[:, k] = (u * v).sum(1) / np.maximum(cr, 1e-12)
    return np.clip(C, -10, 1e3)


def flatten(X, T, init=None, iters=60, pin=None, tol=1e-7, log=None):
    """ARAP flattening (Liu et al. 2008). Returns U (n,2)."""
    n = len(X)
    P, _, _, _ = local_frames(X, T)
    C = cot_weights(P)
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j = T[:, k], T[:, (k + 1) % 3]; w = C[:, k]
        rows += [i, j, i, j]; cols += [i, j, j, i]; vals += [w, w, -w, -w]
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    pin = int(np.argmin(X[:, 2])) if pin is None else pin
    L = L.tolil(); L[pin, :] = 0; L[pin, pin] = 1.0; L = L.tocsc()
    solve = spla.factorized(L)
    if init is None:
        init = lscm(X, T)
    U = np.array(init, float).copy()
    last = None
    for it in range(iters):
        # local: the nearest rotation of each triangle's Jacobian (flat 3D frame -> U)
        Q = U[T]                                       # (m,3,2)
        S = np.zeros((len(T), 2, 2))
        for k in range(3):
            du = Q[:, k] - Q[:, (k + 1) % 3]; dx = P[:, k] - P[:, (k + 1) % 3]
            S += C[:, k, None, None] * du[:, :, None] * dx[:, None, :]
        Us, _, Vts = np.linalg.svd(S)
        R = Us @ Vts
        bad = np.linalg.det(R) < 0
        if bad.any():
            Us[bad, :, 1] *= -1; R[bad] = Us[bad] @ Vts[bad]
        # global
        b = np.zeros((n, 2))
        for k in range(3):
            i, j = T[:, k], T[:, (k + 1) % 3]
            d = np.einsum('mab,mb->ma', R, P[:, k] - P[:, (k + 1) % 3]) * C[:, k, None]
            np.add.at(b, i, d); np.add.at(b, j, -d)
        b[pin] = U[pin]
        U = np.column_stack([solve(b[:, 0]), solve(b[:, 1])])
        e = energy(X, T, U)
        if log and (it % 10 == 0 or it == iters - 1):
            log('  arap %d: energy %.6f' % (it, e))
        if last is not None and abs(last - e) < tol * max(e, 1e-12):
            break
        last = e
    return U


def lscm(X, T):
    """Least-squares conformal map (Levy, Petitjean, Ray & Maillot 2002), two far points pinned: the start for ARAP."""
    n, m = len(X), len(T)
    P, _, _, _ = local_frames(X, T)
    A = np.sqrt(np.maximum(areas(X, T), 1e-16))
    # for each triangle: sum_k W_k (u_k + i v_k) = 0 with W_k = (P_{k+2} - P_{k+1}) as complex, over sqrt(2A)
    Wr = np.zeros((m, 3)); Wi = np.zeros((m, 3))
    for k in range(3):
        d = P[:, (k + 2) % 3] - P[:, (k + 1) % 3]
        Wr[:, k] = d[:, 0] / A; Wi[:, k] = d[:, 1] / A
    i0 = int(np.argmin(X[:, 2])); i1 = int(np.argmax(np.linalg.norm(X - X[i0], axis=1)))
    pinned = {i0: (0.0, 0.0), i1: (np.linalg.norm(X[i1] - X[i0]), 0.0)}
    free = np.setdiff1d(np.arange(n), list(pinned)); col = -np.ones(n, np.int64); col[free] = np.arange(len(free))
    nf = len(free)
    rows, cols, vals = [], [], []
    rhs = np.zeros(2 * m)
    r = np.arange(m)
    for k in range(3):
        v = T[:, k]
        for (rr, cu, cv) in ((r, Wr[:, k], -Wi[:, k]), (r + m, Wi[:, k], Wr[:, k])):
            for (cc, off) in ((cu, 0), (cv, nf)):
                fr = col[v] >= 0
                rows.append(rr[fr]); cols.append(col[v][fr] + off); vals.append(cc[fr])
            for vi, (pu, pv) in pinned.items():
                hit = v == vi
                rhs[rr[hit]] -= cu[hit] * pu + cv[hit] * pv
    Mx = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * m, 2 * nf))
    Mt = Mx.T.tocsr()
    sol = spla.spsolve((Mt @ Mx).tocsc(), Mt @ rhs)
    U = np.zeros((n, 2)); U[free, 0] = sol[:nf]; U[free, 1] = sol[nf:]
    for vi, p in pinned.items(): U[vi] = p
    return U


def principal(X, T, U):
    """Per triangle: (s1, s2, d1, d2) the larger and smaller stretch from the flat U to 3D X, and their directions
    on the 3D surface (unit vectors)."""
    P, ex, ey, nr = local_frames(X, T)
    Q = U[T]
    # J maps flat (U) to the 3D frame: J @ (Q1-Q0, Q2-Q0) = (P1-P0, P2-P0)
    Du = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], axis=2)    # (m,2,2) columns
    Dx = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)
    J = Dx @ np.linalg.inv(Du)
    Uj, s, Vt = np.linalg.svd(J)
    d1 = Uj[:, 0, 0, None] * ex + Uj[:, 1, 0, None] * ey
    d2 = Uj[:, 0, 1, None] * ex + Uj[:, 1, 1, None] * ey
    return s[:, 0], s[:, 1], d1, d2


def energy(X, T, U, per_tri=False):
    s1, s2, _, _ = principal(X, T, U)
    A = areas(X, T)
    e = (s1 - 1) ** 2 + (s2 - 1) ** 2
    return e if per_tri else float((e * A).sum() / A.sum())


def stretch_along(X, T, U, tri, dirs):
    """The stretch of the cloth along given 3D directions in given triangles (|J^+ d|^-1 ... the 3D length of a
    unit of 3D direction against its flat length): >1 pulled, <1 gathered."""
    P, ex, ey, nr = local_frames(X, T[tri])
    Q = U[T[tri]]
    Du = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], axis=2)
    Dx = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)
    Jinv = Du @ np.linalg.inv(Dx)                       # 3D frame -> flat
    dl = np.stack([(dirs * ex[:]).sum(1), (dirs * ey[:]).sum(1)], axis=1)
    dl /= np.maximum(np.linalg.norm(dl, axis=1), 1e-12)[:, None]
    return 1.0 / np.linalg.norm(np.einsum('mab,mb->ma', Jinv, dl), axis=1)


def vertex_normals(X, T):
    N = np.zeros_like(X)
    fn = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]


class Locator:
    """Points on a 2D map U(T) (a flat pattern, or the front view) carried to the surface X by barycentric
    interpolation. Robust to overlaps (the front view folds over itself at the sides): of the triangles that
    contain a point, the one nearest the camera (least y) is taken when `front` is set."""
    def __init__(self, U, T, X, front=True, k=16, snap=0.08):
        from scipy.spatial import cKDTree
        self.U, self.T, self.X, self.front, self.k, self.snap = U, T, X, front, k, snap
        self.kd = cKDTree(U[T].mean(1))

    def bary(self, q):
        _, cand = self.kd.query(q, k=self.k)
        tt = self.T[cand]                                   # (m,k,3)
        A, B, C = self.U[tt[..., 0]], self.U[tt[..., 1]], self.U[tt[..., 2]]
        v0, v1, v2 = B - A, C - A, q[:, None, :] - A
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        den = np.where(np.abs(den) < 1e-14, 1e-14, den)
        l1 = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
        l2 = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
        l0 = 1 - l1 - l2
        L = np.stack([l0, l1, l2], -1)
        inside = (L > -1e-6).all(-1)
        if self.front:
            depth = (self.X[tt][..., 1] * L).sum(-1)
            key = np.where(inside, depth, np.inf)
        else:
            key = np.where(inside, np.arange(self.k)[None, :], np.inf)
        pick = np.argmin(key, axis=1)
        ok = np.isfinite(key[np.arange(len(q)), pick])
        r = np.arange(len(q))
        tri, Lp = tt[r, pick], L[r, pick]
        if (~ok).any():                                    # just off the edge: the nearest triangle, clamped
            bad = np.nonzero(~ok)[0]
            j = np.argmax(L[bad].min(-1), axis=1)
            Lc = np.clip(L[bad, j], 0, None); Lc /= Lc.sum(-1, keepdims=True)
            tri[bad], Lp[bad] = tt[bad, j], Lc
            P = (Lc[..., None] * self.U[tt[bad, j]]).sum(1)
            ok[bad] = np.linalg.norm(P - q[bad], axis=1) < self.snap
        return tri, Lp, ok

    def __call__(self, q, X=None):
        X = self.X if X is None else X
        tri, L, ok = self.bary(np.asarray(q, float))
        out = (L[..., None] * X[tri]).sum(1)
        out[~ok] = np.nan
        return out, ok


# ---------------------------------------------------------------------------------------------------------------
# The lapel: a roll line on the front, the front released along it, and the cloth past it turned back.
# The roll line and the outline are defined exactly as blender/collar/foldfront.py defines them, so a derived design
# is passed to it as options: the line from N (on the neck edge, neck_x out from the middle) to B (the front's edge at
# the break, height `foot`), swung `belly` * 6.75 t^2 (1-t) off the chord away from the V; the outline as
# (t down the line, d out from it) in shares of the chord's length.

def roll_curve(N, B, belly, n_pts=200):
    ln = np.linalg.norm(B - N); u = (B - N) / ln
    n = np.array([-u[1], u[0]])
    n = n if n[0] * np.sign(B[0] - N[0]) > 0 else -n            # toward the V (the front's edge side)
    ts = np.linspace(0, 1, n_pts)
    C = N[None] + (ts * ln)[:, None] * u[None] - (belly * 6.75 * ts * ts * (1 - ts))[:, None] * n[None]
    return ts, C, n, ln


def outline_curve(OT, rf=0.05, sw=0.025):
    """foldfront's outline_round: a three-point outline (top, point, foot) with the point rounded (fillet rf) and the
    long edge swelling sw (negative: hollowed, an S-curve), widths rescaled to keep the point's reach."""
    OT = np.array(OT, float)
    if len(OT) != 3 or (rf == 0 and sw == 0):
        return OT
    A_, P_, B_ = OT
    def bez(p0, p1, p2, n):
        u = np.linspace(0, 1, n + 1)[:, None]; return (1 - u) ** 2 * p0 + 2 * u * (1 - u) * p1 + u ** 2 * p2
    e1 = (A_ - P_) / np.linalg.norm(A_ - P_); e2 = (B_ - P_) / np.linalg.norm(B_ - P_)
    mid = 0.5 * (P_ + B_); nrm = np.array([-e2[1], e2[0]]); nrm = nrm * np.sign(nrm[1] + 1e-12)
    ctl = mid + 2 * sw * nrm
    q1 = P_ + rf * e1; t2 = (ctl - P_) / np.linalg.norm(ctl - P_); q2 = P_ + rf * t2
    long_ = bez(P_, ctl, B_, 24); long_ = long_[np.linalg.norm(long_ - P_, axis=1) > rf * 1.05]
    O = np.vstack([A_[None], bez(q1, P_ + 0.15 * rf * (e1 + t2), q2, 10), long_])
    O = O[np.concatenate([[True], np.diff(O[:, 0]) > 1e-5])]
    O[:, 1] *= P_[1] / O[:, 1].max(); O[-1] = B_
    return O


def principal2(U, X, T):
    """Principal stretches from flat U (n,2) to 3D X for triangles T (no shared mesh needed)."""
    return principal(X, T, U)


class Front:
    """One front of the bare bodice: its 3D mesh (X, T), its best flat cut U (closed), and its front view."""
    def __init__(self, X, T, U, side, M, thick):
        self.X, self.T, self.U, self.side, self.M, self.th = X, T, U, side, M, thick
        self.A = areas(X, T)
        self.e_closed = energy(X, T, U, per_tri=True)
        self.cen = X[T].mean(1)
        self.fv = Locator(X[:, [0, 2]], T, X)          # front view (x, z) -> the surface
        Nv = vertex_normals(X, T)
        Nv *= np.where(Nv[:, 1] < 0, 1, -1)[:, None]      # outward, away from the body (front is -y)
        self.Nv = Nv

    def to_surface(self, xz, U=None):
        """front-view points -> (3D points, flat-pattern points in U, ok)."""
        U = self.U if U is None else U
        P, ok = self.fv(xz)
        Q, _ = self.fv(xz, X=np.c_[U, np.zeros(len(U))])
        return P, Q[:, :2], ok

    def split(self, N, B, belly, foot):
        """The triangles past the roll line that turn back (foldfront's rule), and the curve."""
        ts, C, n, ln = roll_curve(N, B, belly)
        from scipy.spatial import cKDTree
        q = self.cen[:, [0, 2]]
        tg = np.gradient(C, axis=0); nd = np.column_stack([-tg[:, 1], tg[:, 0]])
        nd /= np.linalg.norm(nd, axis=1)[:, None]; nd = np.where(((nd @ n) > 0)[:, None], nd, -nd)
        _, jn = cKDTree(C).query(q)
        sd = ((q - C[jn]) * nd[jn]).sum(1)
        chord = ((q - N) @ ((B - N) / ln)) / ln
        lap = (sd > 0) & (q[:, 1] > foot - 1e-3) & (chord > -0.15)
        return lap, ts, C, ln

    def evaluate(self, neck_x, foot, belly, outline, rf=0.05, sw=0.025, zone=None, iters=40, nt=60, nd_=14,
                 want=False, area=None):
        M = self.M; sg = self.side
        # N: the neck edge at neck_x (the highest point of the front's cloth there); B: the front's edge at the break
        xN = sg * neck_x
        near = np.abs(self.X[:, 0] - xN) < 0.15
        zN = self.X[near, 2].max()
        N = np.array([xN, zN - 0.02]); B = np.array([-sg * M['lap'], foot])
        lap, ts, C, ln = self.split(N, B, belly, foot)
        keepS = ~lap
        Xs, Ts, ids = submesh(self.X, self.T, keepS)
        remap = -np.ones(len(self.X), np.int64); remap[ids] = np.arange(len(ids))
        US = flatten(Xs, Ts, init=self.U[ids], iters=iters, tol=1e-6)
        eS = energy(Xs, Ts, US, per_tri=True); AS = areas(Xs, Ts); cS = Xs[Ts].mean(1)
        zone_m = np.ones(len(Ts), bool) if zone is None else zone(cS)
        # the lapel, cut past the line on S's own flat pattern and turned onto S
        S = Front.__new__(Front); S.X, S.T, S.U = Xs, Ts, US
        S.fv = Locator(Xs[:, [0, 2]], Ts, Xs)
        Cs3, Cu, ok = S.to_surface(C, U=US)
        if (~ok).sum() > 3:
            return dict(ok=False)
        Cu = Cu[ok]; tsk = ts[ok]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(Cu, axis=0), axis=1) > 1e-6])
        Cu, tsk = Cu[keep], tsk[keep]
        # the line on the pattern, faired: the cut follows triangle edges, so its points jitter by a triangle
        cf = [np.polyfit(tsk, Cu[:, k], 5) for k in (0, 1)]
        tsk = np.linspace(0, 1, 400); Cu = np.column_stack([np.polyval(c_, tsk) for c_ in cf])
        # arc length along the line on the flat pattern, its tangent and normal there (toward the V side)
        tg = np.gradient(Cu, axis=0); tg /= np.linalg.norm(tg, axis=1)[:, None]
        nu = np.column_stack([-tg[:, 1], tg[:, 0]])
        # which side is the V: nu must point away from the body side (a point just off the line, away from the V)
        _, _, nfv, _ = roll_curve(N, B, belly)
        k = len(C) // 2
        test = self._fv_to_u((C[k] - 0.3 * nfv)[None], US, S)
        if test is not None and np.dot(test - Cu[min(k, len(Cu) - 1)], nu[min(k, len(nu) - 1)]) > 0:
            nu = -nu
        O = outline_curve(outline, rf, sw)
        tt = np.linspace(O[0, 0], 1.0, nt)
        D = np.interp(tt, O[:, 0], O[:, 1]) * ln
        cut_area = float(np.trapezoid(D, tt * ln))           # the lapel's area as drafted (outline on the chord)
        wscale = 1.0
        if area is not None:                                # the design held: the same lapel area, the shape free
            wscale = area / cut_area; D = D * wscale; cut_area = area
        rows = []
        for i, t_ in enumerate(tt):
            for j in range(nd_ + 1):
                rows.append((t_, D[i] * j / nd_))
        R = np.array(rows)
        cu = np.column_stack([np.interp(R[:, 0], tsk, Cu[:, k]) for k in (0, 1)])
        nn = np.column_stack([np.interp(R[:, 0], tsk, nu[:, k]) for k in (0, 1)])
        flat = cu + R[:, 1:2] * nn                       # the lapel as cut (on the front's own pattern, past the line)
        turned_u = cu - R[:, 1:2] * nn                   # where it lands, turned over the line, on the front's pattern
        locS = Locator(US, Ts, Xs)
        Pt, okt = locS(turned_u)
        NvS = vertex_normals(Xs, Ts); NvS *= np.where(NvS[:, 1] < 0, 1, -1)[:, None]
        Nt, _ = locS(turned_u, X=NvS)
        # quads of the (t, j) grid -> triangles
        TT = []
        W = nd_ + 1
        for i in range(nt - 1):
            for j in range(nd_):
                a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
                TT += [(a, c, b), (b, c, d)]
        TT = np.array(TT)
        good = okt[TT].all(1)
        lost = 1 - good.mean()
        Pt = np.where(okt[:, None], Pt, 0.0); Nt = np.where(okt[:, None], Nt, 0.0)
        Pt = Pt + self.th * Nt / np.maximum(np.linalg.norm(Nt, axis=1), 1e-9)[:, None]
        AL = areas(np.c_[flat, np.zeros(len(flat))], TT)
        good &= AL > 1e-9
        s1, s2, _, _ = principal(Pt, TT[good], flat)
        eL = (s1 - 1) ** 2 + (s2 - 1) ** 2
        AL = AL[good]
        # the score: mean strain energy over the cloth that is there, in the upper-front zone and on the lapel
        eSz, ASz = eS[zone_m], AS[zone_m]
        tot = (eSz * ASz).sum() + (eL * AL).sum()
        out = dict(ok=True, lost=float(lost), neck_x=neck_x, foot=foot, belly=belly,
                   body=float((eSz * ASz).sum() / ASz.sum()), lapel=float((eL * AL).sum() / AL.sum()),
                   score=float(tot / (ASz.sum() + AL.sum())), lapel_area=float(AL.sum()), cut_area=cut_area, wscale=float(wscale), roll_len_fv=float(ln),
                   lap_s1_95=float(np.percentile(s1, 95)), lap_s2_5=float(np.percentile(s2, 5)),
                   body_s1_95=float(np.percentile(principal(Xs, Ts[zone_m], US)[0], 95)),
                   body_s2_5=float(np.percentile(principal(Xs, Ts[zone_m], US)[1], 5)))
        # the closed front's strain along the roll line (what the roll-line tape takes up), on the closed pattern
        C3, _, okc = self.to_surface(C)
        L3 = np.linalg.norm(np.diff(C3[okc], axis=0), axis=1).sum()
        Cuc = self.to_surface(C)[1][okc]
        Lu = np.linalg.norm(np.diff(Cuc, axis=0), axis=1).sum()
        out.update(roll_len_3d=float(L3), roll_len_flat=float(Lu), roll_stretch=float(L3 / Lu))
        if want:
            out.update(Xs=Xs, Ts=Ts, US=US, eS=eS, flat=flat, Pt=Pt, TT=TT[good], eL=eL, s1=s1, s2=s2, C=C, N=N, B=B,
                       ids=ids, R=R[:, :], O=O, ln=ln)
        return out

    def _fv_to_u(self, xz, U, S):
        Q, ok = S.fv(xz, X=np.c_[U, np.zeros(len(U))])
        return Q[0, :2] if ok[0] else None


# ---------------------------------------------------------------------------------------------------------------
# The bare front as the tailor cuts it: each front from the closing edge to the side, waist to neck, with its two
# darts slit open (an 1880s bodice's two front darts each side, as blender/redingote/bodice.py places them).

DARTS = ((0.0, 0.34, 0.45, 0.80), (0.75, 0.62, 0.90, 0.65))     # (copied from redingote/bodice.py DARTS)


def dart_lines(P, M, half=0.05):
    """True for points on a dart's line: the cloth the dart takes out, slit open in the flat cut."""
    w = M['w']; z_w = M['z_waist']; x_ap, z_ap = M['apex_x'], M['apex_z']

    def run(z, z_top, z_foot, top, mid, foot):
        up = np.clip((z - z_w) / max(z_top - z_w, 1e-6), 0.0, 1.0)
        dn = np.clip((z_w - z) / max(z_w - z_foot, 1e-6), 0.0, 1.0)
        return np.where(z >= z_w, mid + (top - mid) * up ** 1.4, mid + (foot - mid) * dn ** 1.2)
    z = P[:, 2]; on = np.zeros(len(P), bool)
    for out, waist_k, drop, foot in DARTS:
        zt, zf = z_ap - drop * w, z_w - foot * w
        x_top, x_w = x_ap + out * w, waist_k * M['waist_half']
        xs = run(z, zt, zf, x_top, x_w, x_w + 0.3 * (x_top - x_w))
        on |= (np.abs(np.abs(P[:, 0]) - xs) < half) & (z < zt) & (z > zf) & (P[:, 1] < M['yc'])
    return on


def bare_front(X, T, front_layer, M, thick, layer, x_side=3.5, iters=150, log=None):
    """One front of the bare bodice, flattened: a Front. x_side bounds it short of the armhole's turn (the front view
    folds over itself past it and the lapel never reaches it)."""
    cen = X[T].mean(1)
    keep = ((front_layer[T] == layer).all(1) & (cen[:, 1] < M['yc']) & (cen[:, 2] > M['z_waist'])
            & ~dart_lines(cen, M) & (np.abs(cen[:, 0]) < x_side))
    Xs, Ts, ids = submesh(X, T, keep)
    U = flatten(Xs, Ts, iters=iters, log=log)
    F = Front(Xs, Ts, U, -1.0 if layer == 1 else 1.0, M, thick)
    F.ids = ids
    return F
