"""Endpoint flow: the smoothest line between a set start and a set end, kept off a body.

Each hair is given where it starts (a point and a direction: its root and the way it leaves the skin) and where it ends
(a point and a direction: the place its tip lies and the way it arrives). The solver finds the line of least bending
between the two, the minimiser of the sum of squared second differences with both ends' points and directions held,
and keeps it at least a clearance off the body: nodes inside are pushed out, held softly where they were pushed to,
and the line is solved again around them (an active set), so the line wraps the body as a combed hair wraps a head
instead of cutting through it. Nothing between the ends is authored: no heading by height, no traced middle.

solve(P0, T0, P1, T1, floor, ...) -> lines [n, N, 3], report
  P0, P1 [n, 3]   start and end points          T0, T1 [n, 3]   unit directions there
  m0, m1 [n]      how far the line keeps each direction (a length, the Hermite tangent's size): a larger m0 stands the
                  root up longer
  floor(X [k, 3]) -> X pushed out to the body plus its clearance (points already outside are returned unchanged)
Report: iterations, nodes pushed out on the last pass, the worst depth left inside (should be ~0), bend radius min/p1.
"""
import numpy as np


def hermite(P0, T0, P1, T1, m0, m1, N):
    t = np.linspace(0, 1, N)[None, :, None]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
    return h00 * P0[:, None] + h10 * (m0[:, None] * T0)[:, None] + h01 * P1[:, None] + h11 * (m1[:, None] * T1)[:, None]


def _bend(N):
    D = np.zeros((N - 2, N))
    for i in range(N - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D.T @ D


def solve(P0, T0, P1, T1, floor, m0=None, m1=None, N=48, passes=30, hold=6.0, tol=2e-4, log=None):
    n = len(P0)
    chord = np.linalg.norm(P1 - P0, axis=1)
    m0 = chord if m0 is None else np.broadcast_to(m0, (n,)).astype(float)
    m1 = chord if m1 is None else np.broadcast_to(m1, (n,)).astype(float)
    X = hermite(P0, T0, P1, T1, m0, m1, N)
    fixed = np.zeros(N, bool); fixed[[0, 1, N - 2, N - 1]] = True      # (two nodes an end: its point and its direction)
    K = _bend(N)
    pin = X.copy()                                                     # (the ends as the Hermite set them)
    target = X.copy(); active = np.zeros((n, N), bool)
    rep = {}
    for it in range(passes):
        F = floor(X.reshape(-1, 3)).reshape(n, N, 3)
        depth = np.linalg.norm(F - X, axis=2)
        inside = (depth > 1e-6) & ~fixed[None]
        active |= inside
        # (a held node's target is where the body's surface is under it NOW: it slides along the body from pass to
        # pass. Held to where it was first pushed, the line kept the first pass's dents: bend radius 0.1 mm.)
        target = np.where(active[..., None], F, target)
        rep = dict(passes=it + 1, pushed=int(inside.sum()), worst_inside=float(depth[:, ~fixed].max()))
        if it and depth[:, ~fixed].max() < tol:
            break
        # least bending with the held nodes: (K + hold * A) x = hold * A target, the end rows replaced by their pins
        A = np.broadcast_to(K, (n, N, N)).copy()
        idx = np.arange(N)
        A[:, idx, idx] += hold * active
        B = hold * active[..., None] * target
        A[:, fixed] = 0.0
        A[:, np.where(fixed)[0], np.where(fixed)[0]] = 1.0
        B[:, fixed] = pin[:, fixed]
        X = np.linalg.solve(A, B)
    # (the hold is soft on purpose: what it leaves inside the clearance, 'worst_inside', is not snapped out. A hard snap
    # copies every bump of a sampled body into the line: bend radius 0.03 mm on a voxel head. Give the floor a clearance
    # that can spare 'worst_inside'.)
    d1 = np.diff(X, axis=1); d2 = np.diff(X, 2, axis=1)
    ds = np.linalg.norm(d1, axis=2)
    k = np.linalg.norm(np.cross(d1[:, :-1], d2), axis=2) / np.maximum(ds[:, :-1] ** 3, 1e-12)
    R = 1.0 / np.maximum(k, 1e-9)
    rep.update(lines=n, nodes=N, bend_radius_min=float(R.min()), bend_radius_p1=float(np.percentile(R, 1)),
               bend_radius_median=float(np.median(R)))
    if log:
        log('endpoint_flow %s' % rep)
    return X, rep
