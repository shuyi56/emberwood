"""Sheet flow: a styled piece of hair solved as ONE SURFACE in 3D, its hairs drawn as lines on it.

Solving each hair alone (endpoint_flow.py) leaves every line free to find its own way round the head: neighbours wobble
apart, and a shape that bends two ways needs every line told so. A fringe is a sheet: its hairs lie side by side and
move together. Here the sheet is a grid, u across the hairs and v along them. Its two edges are set: the ROOT edge (a
point and the direction it leaves the skin, per column) and the END edge (a point and the direction it arrives). The
surface between is the one of least bending BOTH ways, along the hairs and across them, kept off the body by a soft
active set that holds a node along the body's normal only and lets it go when it lifts, and drawn through any set curves on the way (a waist). All three coordinates of
every node are solved together in one system, so a bend in depth and a bend across the face are one surface's.

solve(P0, T0, P1, T1, gap, Nv, across, via) -> sheet [Nu, Nv, 3], report
  P0, T0, P1, T1 [Nu, 3]   each column's start point and direction, end point and direction
  gap(X [k, 3])            (how far each point stands above the body plus its clearance (negative: inside), the outward normal)
  across                   the weight of bending across the hairs against bending along them (1: alike; 0: lines alone)
  via                      set curves on the way: (share along, points [Nu, 3], weight); held at each column's nearest node
lines(sheet, t, ...) reads a hair off the sheet at its place t across it (0 to 1).
Report: passes, pushed, worst_inside, bend radius along (min, 1st percentile, median) and across (median).
"""
import numpy as np
import endpoint_flow as EF


def solve(P0, T0, P1, T1, gap, Nv=48, across=1.0, passes=40, hold=40.0, release=2e-4, via=(), log=None):
    Nu = len(P0); n = Nu * Nv
    chord = np.linalg.norm(P1 - P0, axis=1)
    X = EF.hermite(P0, T0, P1, T1, chord, chord, Nv)                                          # [Nu, Nv, 3]
    Kv = EF._bend(Nv); Ku = EF._bend(Nu)
    K = np.kron(np.kron(np.eye(Nu), Kv) + across * np.kron(Ku, np.eye(Nv)), np.eye(3))       # (x, y, z of a node together)
    fixed = np.zeros((Nu, Nv), bool); fixed[:, [0, 1, Nv - 2, Nv - 1]] = True
    fx = np.repeat(fixed.ravel(), 3)
    active = np.zeros(n, bool); idx = np.arange(n)
    rep = {}
    for it in range(passes):
        # the two nodes pinned at each edge: the edge's point, and one step along its direction at the column's own
        # spacing (a step shorter or longer than its neighbours' is a kink in a solve that measures bending per node)
        step = (np.linalg.norm(np.diff(X, axis=1), axis=2).sum(1) / (Nv - 1))[:, None]
        X[:, 0] = P0; X[:, 1] = P0 + T0 * step; X[:, -1] = P1; X[:, -2] = P1 - T1 * step
        Xf = X.reshape(-1, 3)
        g, nrm = gap(Xf)
        inside = (g < 0) & ~fixed.ravel()
        active = inside | (active & (g < release))
        rep = dict(passes=it + 1, pushed=int(inside.sum()), on_body=int(active.sum()),
                   worst_inside=float(max(0.0, -g[~fixed.ravel()].min())))
        A = K.copy(); B = np.zeros(3 * n)
        # the body holds a node only ALONG ITS NORMAL: n.(x - X) + g = 0. Along the skin the node is free, so the
        # line's way across the body is still the one of least bending (held in all three it kept a corner at a waist)
        ka = idx[active]
        for c in range(3):
            for e in range(3):
                A[3 * ka + c, 3 * ka + e] += hold * nrm[ka, c] * nrm[ka, e]
            B[3 * ka + c] += hold * nrm[ka, c] * ((nrm[ka] * Xf[ka]).sum(1) - g[ka])
        for frac, Wp, wv in via:
            if it == 0:
                kv = np.full(Nu, int(round(frac * (Nv - 1))))
            else:
                dv = np.linalg.norm(X - Wp[:, None], axis=2); dv[:, :3] = 1e9; dv[:, -3:] = 1e9
                kv = dv.argmin(1)
            k = np.arange(Nu) * Nv + kv
            for c in range(3):
                A[3 * k + c, 3 * k + c] += wv; B[3 * k + c] += wv * Wp[:, c]
        A[fx] = 0.0; A[fx, fx] = 1.0; B[fx] = Xf.ravel()[fx]
        X = np.linalg.solve(A, B).reshape(Nu, Nv, 3)

    def radius(Y):
        d1 = np.diff(Y, axis=1); d2 = np.diff(Y, 2, axis=1)
        k = np.linalg.norm(np.cross(d1[:, :-1], d2), axis=2) / np.maximum(np.linalg.norm(d1[:, :-1], axis=2) ** 3, 1e-12)
        return 1.0 / np.maximum(k, 1e-9)
    Ra = radius(X); Rc = radius(X.transpose(1, 0, 2))
    rep.update(columns=Nu, rows=Nv, along_radius_min=float(Ra.min()), along_radius_p1=float(np.percentile(Ra, 1)),
               along_radius_median=float(np.median(Ra)), across_radius_median=float(np.median(Rc)))
    if log:
        log('sheet_flow %s' % rep)
    return X, rep


def lines(sheet, t):
    """Each hair's line on the sheet at its place t (0..1) across it: [n, Nv, 3], linear between columns."""
    Nu = sheet.shape[0]
    f = np.clip(np.asarray(t, float), 0, 1) * (Nu - 1)
    i0 = np.minimum(f.astype(int), Nu - 2); w = (f - i0)[:, None, None]
    return sheet[i0] * (1 - w) + sheet[i0 + 1] * w
