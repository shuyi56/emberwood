"""The armhole the redingote's bodice and sleeves are sewn along: one seam line, measured on the body, shared by both.

The bodice is cut along it and the sleeve's head is gathered into it, so the two meet at one line however the body
changes. Nothing here is placed by hand: every size is a share of the vest's tailor's measurements (vest/fit.measure:
w the neck's half-width, x_sh/z_sh the shoulder point, z_pit_arm the armpit).

The seam lies on a ruled surface seen from the front as a tailor draws the armscye: plumb at the underarm, curving out
to the shoulder point at the top, and turned a little forward in plan (the arm hangs ahead of the side seam, so the
front of an armhole is scooped deeper than the back). Where that surface meets the skin is the armhole; where the skin
is hollow (the front and back folds of the armpit) the line is drawn taut across, as cloth is.

The loop is on the body's rest shape, with no ease: each garment adds its own.

    armhole(P, aw, M)       {'L': side, 'R': side}: each side's closed seam loop and its measurements
    armhole_cut(M, loops)   f(p): the bodice's trim field (positive on the cloth that stays, negative in the
                            armhole), for fit.trim
"""
import math

import numpy as np

RATIOS = {
    'seam_out': 0.95,        # the seam's top: this share of the way out to the shoulder point (an 1880s tailored
                             # bodice sets the sleeve at the shoulder point; a gathered head sits a little inside it)
    'drop': 0.35,            # the underarm: this far below the armpit, in w (a close tailored armhole)
    'under_in': 0.12,        # and this far out from the armpit's inner line, in w
    'curve': 1.6,            # how the line bows from plumb at the underarm out to the shoulder point (1 straight)
    'tilt': 0.15,            # the turn forward in plan: x moves in by this much for each unit forward
    'smooth': 9,             # harmonics kept round the loop (rounds the bridged underarm into a curve)
}
N = 96                       # points round a loop


def _surface(M, x_pit, ratios):
    """The seam surface for the left side: x_seam(y, z)."""
    w = M['w']
    z_u = M['z_pit_arm'] - ratios['drop'] * w
    x_u = x_pit + ratios['under_in'] * w
    x_s, z_s = ratios['seam_out'] * M['x_sh'], M['z_sh']
    y_ref = 0.5 * (M['scye_y'][0] + M['scye_y'][1])

    def x_seam(y, z):
        t = np.clip((np.asarray(z) - z_u) / (z_s - z_u), 0.0, 1.0)
        return x_u + (x_s - x_u) * t ** ratios['curve'] + ratios['tilt'] * (np.asarray(y) - y_ref)
    return x_seam, z_u, x_u, x_s, z_s, y_ref


def _x_pit(P, aw):
    band = P[(aw > 0.4) & (aw < 0.6)]
    low = band[band[:, 2] <= np.percentile(band[:, 2], 3)]
    return float(np.abs(low[:, 0]).mean())


def _hull(pts):
    """The convex hull of 2D points, counter-clockwise."""
    pts = sorted(map(tuple, pts))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(list(reversed(pts)))
    return np.array(lo[:-1] + up[:-1])


def _resample(loop, n):
    seg = np.linalg.norm(np.roll(loop, -1, 0) - loop, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, s[-1], n, endpoint=False)
    closed = np.vstack([loop, loop[:1]])
    return np.stack([np.interp(u, s, closed[:, k]) for k in range(loop.shape[1])], 1)


def armhole(P, aw, M, T=None, ratios=RATIOS):
    """The seam loop on each side. P, aw: the body's rest points and arm shares (fit.body_arrays); T its triangles
    (needed: the loop is where the skin crosses the seam surface)."""
    if T is None:
        raise ValueError('armhole: the body\'s triangles are needed')
    w = M['w']
    x_pit = _x_pit(P, aw)
    x_seam, z_u, x_u, x_s, z_s, y_ref = _surface(M, x_pit, ratios)
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = np.unique(np.sort(E, 1), axis=0)
    out = {}
    for name, side in (('L', 1.0), ('R', -1.0)):
        g = x_seam(P[:, 1], P[:, 2]) - side * P[:, 0]          # (positive inside the seam: the bodice's side)
        a, b = E[:, 0], E[:, 1]
        near = (side * P[a, 0] > 0.5 * x_u) & (P[a, 2] > z_u - 0.5 * w) & (P[a, 2] < z_s + 2.0 * w)
        cross = near & ((g[a] > 0) != (g[b] > 0))
        t = g[a[cross]] / (g[a[cross]] - g[b[cross]])
        X = P[a[cross]] + t[:, None] * (P[b[cross]] - P[a[cross]])
        X = X[X[:, 2] >= z_u]
        # (taut: the hull of the crossing seen along the seam surface, i.e. by front-back and height)
        H = _hull(X[:, 1:3])
        H = _resample(H, 4 * N)
        # rounded: the loop's low harmonics only, which turns the straight bridge under the arm into a curve
        c = H[:, 0] + 1j * H[:, 1]
        F = np.fft.fft(c)
        k = ratios['smooth']
        F[k + 1:-k] = 0
        c = np.fft.ifft(F)
        yz = np.stack([c.real, c.imag], 1)
        yz = _resample(yz, N)
        # start at the top, go forward (toward -y) first
        i0 = int(np.argmax(yz[:, 1]))
        yz = np.roll(yz, -i0, 0)
        if yz[1, 0] > yz[0, 0]:
            yz = np.roll(yz[::-1], 1, 0)
        loop = np.stack([side * x_seam(yz[:, 0], yz[:, 1]), yz[:, 0], yz[:, 1]], 1)
        seg = np.linalg.norm(np.roll(loop, -1, 0) - loop, axis=1)
        out[name] = dict(loop=loop, top=loop[0], under=loop[int(np.argmin(loop[:, 2]))], girth=float(seg.sum()),
                         depth=float(loop[:, 2].max() - loop[:, 2].min()), width=float(np.ptp(loop[:, 1])),
                         side=side, z_under=z_u, x_under=x_u, x_top=x_s, y_ref=y_ref)
    return out


def armhole_cut(M, loops, ratios=RATIOS):
    """The bodice's armhole trim, vectorised: f(p) for points p [n, 3], positive on the cloth that stays, negative in
    the armhole. It is the seam surface's own distance (in x), and below the underarm every point stays."""
    L = loops['L']
    x_u, x_s, z_u, y_ref = L['x_under'], L['x_top'], L['z_under'], L['y_ref']
    z_s = M['z_sh']

    def f(p):
        p = np.asarray(p, dtype=float)
        t = np.clip((p[:, 2] - z_u) / (z_s - z_u), 0.0, 1.0)
        xs = x_u + (x_s - x_u) * t ** ratios['curve'] + ratios['tilt'] * (p[:, 1] - y_ref)
        return np.maximum(xs - np.abs(p[:, 0]), z_u - p[:, 2])
    return f
