"""The redingote's sleeves sewn from their flat pattern (pattern.py), without the solver (John's rule, 2026-11-19:
every garment drafted flat, then joined, then draped; a new layer is shown first without the solver).

Each piece is cut as cloth: a mesh laid out over its own flat stitch line, its points keeping their place on the
pattern (the UVs are the pattern's own coordinates, so the grain runs as it was cut). The pieces are then sewn edge to
edge, each seam's two edges brought onto one line round the arm:

  forearm seam   the topsleeve's front edge to the undersleeve's, wrist to the armhole, welded into one cloth
  hindarm seam   the same at the back, down to the vent; below it the topsleeve's edge laps over the undersleeve's
  sleeve head    the topsleeve's head to the armhole (armhole.py's loop, the bodice's own seam) between the sleeve
                 seams, plain beside them and gathered between the notches: its extra length stands in small gathers
                 at the stitching, as deep as the cloth's length over each one asks, opening out into the head
  underarm       the undersleeve's head to the armhole under the arm, plain
  cuff           the band round the wrist below the sleeve's edge, its ends lapped at the vent and buttoned

Between its sewn edges a piece takes the shape it is sewn into (a Coons patch over its four sewn edges, round the arm's
axis), then lies on the sleeve's drafted surface (sleeve.Draft.field: the tube round the arm and the head standing on
its support), which is what the head's net support and the pressing would give it. At each seam the stitching draws the
cloth in a little (the allowances pressed open behind it), and the cuff and the vent carry rows of topstitching.

build_sewn() returns ([sleeve.L, sleeve.R], [buttons.L, buttons.R], [stitches.L, stitches.R], the Draft, a report)."""
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('vest', 'chemise'):
    sys.path.insert(0, os.path.join(HERE, '..', _d))
sys.path.insert(0, HERE)
import fit  # noqa: E402
import armhole  # noqa: E402
import pattern  # noqa: E402
import sleeve  # noqa: E402

CM = pattern.CM
RATIOS = {
    'gather_pitch': 0.9 / CM,    # the gathers' spacing along the stitching, sewn (a running stitch drawn up: small,
                                 # close gathers; the bench's puff stood its folds 3 mm deep)
    'gather_open': 1.2,          # how far the gathers run down into the head before they open out, in w
    'stitch_draw': 1.4 / CM,     # the gathers rise to their depth this far from the stitching
    'seam_w': 0.45 / CM,         # each side of a seam drawn in by the stitching (to the allowances pressed under it)
    'seam_depth': 0.12 / CM,     # how far the stitching draws the cloth in
    'lap_proud': 1.2,            # the vent's lap and the cuff's lap stand out by this many cloth thicknesses
    'stitch_len': 0.30 / CM,     # a topstitch's length
    'stitch_gap': 0.12 / CM,     # and the gap between two
    'stitch_in': 0.30 / CM,      # topstitching this far in from the edge
    'cols_mm': 0.4,              # the mesh's spacing along the head's stitching (sewn), in mm per gather pitch share
}
ROWS = 150                       # rows from the head to the wrist
COLS_UNDER = 64                  # columns across the undersleeve


def _resample(c, n):
    """A polyline resampled to n points evenly by length."""
    c = np.asarray(c, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, c[:, k]) for k in range(c.shape[1])], 1)


def _coons(H, W, F, B, s, v):
    """The patch over four edges: H (v 0) and W (v 1) by s, F (s 0) and B (s 1) by v, sampled on the grid s x v."""
    S, Vv = s[None, :, None], v[:, None, None]
    return ((1 - Vv) * H[None] + Vv * W[None] + (1 - S) * F[:, None] + S * B[:, None]
            - ((1 - S) * (1 - Vv) * H[0] + S * (1 - Vv) * H[-1] + (1 - S) * Vv * W[0] + S * Vv * W[-1]))


class Sewing:
    """The sleeve's seams on the body (left side): the armhole loop by its length, the arm's axis and surface."""

    def __init__(self, D, m, loops):
        self.D, self.m = D, m
        loop = loops['L']['loop']
        fine = armhole._resample(loop, 16 * len(loop))
        seg = np.linalg.norm(np.roll(fine, -1, 0) - fine, axis=1)
        self.fine, self.s, self.G = fine, np.concatenate([[0.0], np.cumsum(seg)]), float(seg.sum())
        # the cloth at the stitching sits on the skin let out by the head's ease
        w = D.M['w']
        off = []
        for p in fine:
            loc, nrm, _, _ = D.b_all.find_nearest(Vector(p))
            off.append(np.array(nrm) if loc is not None else np.zeros(3))
        self.seam_pts = fine + D.r['ease_head'] * w * np.array(off)

    def at(self, sig):
        """The seam's point (on the cloth) at sig along the loop from its top, forward first (wrapped)."""
        sig = np.mod(np.asarray(sig, dtype=float), self.G)
        cl = np.vstack([self.seam_pts, self.seam_pts[:1]])
        return np.stack([np.interp(sig, self.s, cl[:, k]) for k in range(3)], -1)

    def cyl(self, X):
        """(angle round the arm, height, distance from the axis)."""
        al, rho = self.D.polar(X)
        return np.stack([al, X[:, 2], rho], 1)

    def cart(self, C):
        cx, cy = self.D.axis(np.minimum(C[..., 1], self.D.zs[-1]))
        return np.stack([cx + C[..., 2] * np.cos(C[..., 0]), cy + C[..., 2] * np.sin(C[..., 0]), C[..., 1]], -1)

    def ring(self, z, a0, a1, n, out=0.0):
        """The sleeve's surface round the arm at height z from angle a0 to a1 (a1 may run past pi), n points evenly by
        length, in (angle, z, rho)."""
        a = np.linspace(a0, a1, 400)
        R = self.D._look(self.D.R, np.full(len(a), z), np.mod(a + math.pi, 2 * math.pi) - math.pi) + out
        C = np.stack([a, np.full(len(a), z), R], 1)
        X = self.cart(C)
        sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
        t = np.linspace(0.0, sl[-1], n)
        return np.stack([np.interp(t, sl, C[:, k]) for k in range(3)], 1), float(sl[-1])

    def arc_to(self, z, a0, length, sign, out=0.0):
        """The angle reached from a0 going round (sign +1: toward the back over the outside) by this length."""
        a = a0 + sign * np.linspace(0.0, 2 * math.pi, 1200)
        R = self.D._look(self.D.R, np.full(len(a), z), np.mod(a + math.pi, 2 * math.pi) - math.pi) + out
        X = self.cart(np.stack([a, np.full(len(a), z), R], 1))
        sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
        return float(np.interp(length, sl, a)), float(sl[-1])


def _unwrap_from(a, ref):
    """Angles unwrapped into one run, starting as near ref as may be."""
    a = np.unwrap(np.asarray(a, dtype=float))
    return a + 2 * math.pi * round((ref - a[0]) / (2 * math.pi))


def _project_star(D, X, keep=None, steps=28, rounds=16):
    """Points brought onto the sleeve's drafted surface along rays from inside the arm: straight out from the axis
    round the tube, and fanned out from a point inside the shoulder over the head. Neighbours stay in order along
    their rays, so the cloth cannot fold over itself; keep: points held."""
    X = X.copy()
    idx = np.arange(len(X)) if keep is None else np.nonzero(~keep)[0]
    Y = X[idx]
    z0 = D.z_under - 0.5 * D.M['w']
    zc = np.minimum(Y[:, 2], z0)
    cx, cy = D.axis(zc)
    C = np.stack([cx, cy, zc], 1)
    d = Y - C
    n = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    ts = np.linspace(0.3, 5.0, steps)
    f = np.stack([D.field(C + t * n, gathers=False) for t in ts], 1)
    cross = (f[:, :-1] <= 0) & (f[:, 1:] > 0)
    pick = np.where(cross.any(1), np.argmax(cross, axis=1), -1)
    a = np.where(pick >= 0, ts[np.maximum(pick, 0)], 0.0)
    b = np.where(pick >= 0, ts[np.maximum(pick, 0) + 1], 0.0)
    for _ in range(rounds):
        mid = 0.5 * (a + b)
        fm = D.field(C + mid[:, None] * n, gathers=False)
        a, b = np.where(fm <= 0, mid, a), np.where(fm <= 0, b, mid)
    t = 0.5 * (a + b)
    # (never past the armhole's seam surface onto the bodice's side: where the sleeve runs from the seam round to the
    # arm under the front and back of the armhole, it lies along the seam's own surface, not out over the body)
    P_ = C + t[:, None] * n
    e = D.r['ease_head'] * D.M['w']
    over = (P_[:, 0] < D.seam_x(P_[:, 1], P_[:, 2]) + e) & (n[:, 0] < -1e-3)
    for _ in range(3):
        P_ = C + t[:, None] * n
        gap = D.seam_x(P_[:, 1], P_[:, 2]) + e - P_[:, 0]
        t = np.where(over, np.maximum(t - np.maximum(gap, 0.0) / -np.minimum(n[:, 0], -1e-3), 0.3), t)
    ok = pick >= 0
    X[idx[ok]] = C[ok] + t[ok, None] * n[ok]
    return X


def _taubin(G, rounds=20, lam=0.5, mu=-0.53, hold_rows=1, hold_cols=True):
    """A grid of points [rows, cols, 3] smoothed without shrinking (Taubin), its first rows (and its side columns)
    held."""
    G = G.copy()
    for _ in range(rounds):
        for f in (lam, mu):
            L = np.zeros_like(G)
            L[1:-1, 1:-1] = 0.25 * (G[:-2, 1:-1] + G[2:, 1:-1] + G[1:-1, :-2] + G[1:-1, 2:]) - G[1:-1, 1:-1]
            if not hold_cols:
                L[1:-1, 0] = 0.5 * (G[:-2, 0] + G[2:, 0]) - G[1:-1, 0]
                L[1:-1, -1] = 0.5 * (G[:-2, -1] + G[2:, -1]) - G[1:-1, -1]
            L[:hold_rows] = 0.0
            G += f * L
    return G


def _grid_normals(G, D):
    """The grid's own normals, turned out from the arm."""
    a = np.gradient(G, axis=0)
    b = np.gradient(G, axis=1)
    n = np.cross(a, b)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    cx, cy = D.axis(np.minimum(G[..., 2], D.zs[-1]))
    out = np.stack([G[..., 0] - cx, G[..., 1] - cy, 0.3 * np.ones_like(cx)], -1)
    flip = (n * out).sum(-1) < 0
    n[flip] *= -1
    return n.reshape(-1, 3)


def _sine_amp(ratio):
    """The sine's depth, against its period, whose length over a period is ratio times the period."""
    x = np.linspace(0.0, 1.0, 400)

    def L(k):
        y = k * np.sin(2 * math.pi * x)
        return float(np.hypot(np.diff(x), np.diff(y)).sum())
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if L(mid) < ratio else (lo, mid)
    return 0.5 * (lo + hi)


def _dist_to(poly, Q):
    """Each point's distance (flat) to a polyline."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip((((Q[:, None, :] - a[None]) * ab[None]).sum(-1)) / np.maximum((ab ** 2).sum(-1), 1e-12)[None], 0.0, 1.0)
    p = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(Q[:, None, :] - p, axis=-1).min(1)


def cut_and_sew(D, m, loops, pieces, ratios=RATIOS, log=print):
    """The sleeve's cloth (left side), sewn: points, quads, each point's flat pattern place, the seam and edge
    weights, the topstitching's lines and the buttons' places, and the report of how the cloth was eased."""
    r, pr, w = ratios, pattern.RATIOS, D.M['w']
    sw = Sewing(D, m, loops)
    top, und = pieces['topsleeve'], pieces['undersleeve']
    # -- the seams' places round the arm: at the armhole (from the pattern's measurements on the same loop) and at
    # the wrist (the sleeve's forearm seam where the draft puts it, the hindarm where the topsleeve's width ends)
    s_f, s_h = m['fore'], m['hind']
    Pf, Ph = sw.at(s_f), sw.at(s_h)
    Cf, Ch = sw.cyl(Pf[None])[0], sw.cyl(Ph[None])[0]
    z_w = D.z_cuff
    tw, uw = top.length('wrist'), und.length('wrist') - pr['underlap']
    a_fw = sleeve.RATIOS['seam_fore']
    _, ring_w = sw.arc_to(z_w, a_fw, 0.0, 1.0)
    a_hw, _ = sw.arc_to(z_w, a_fw, ring_w * tw / (tw + uw), 1.0)
    log('sewn: forearm seam at the armhole %.2f rad (z %.2f), wrist %.2f; hindarm %.2f (z %.2f), wrist %.2f; the '
        'wrist round %.1f cm, the pattern %.1f' % (Cf[0], Cf[1], a_fw, Ch[0], Ch[1], a_hw, ring_w * CM, (tw + uw) * CM))
    # -- the topsleeve: its head over the crown, gathered between the notches
    L_head = top.length('head')
    n_in = pr['notch_in']
    gathered_flat = L_head - 2 * n_in
    gathered_sewn = (m['armhole'] - m['under_head']) - 2 * n_in
    full = gathered_flat / gathered_sewn
    pitch = r['gather_pitch']
    ncols = int(round(L_head / (pitch * full) * 10))         # (ten points over each gather's cloth)
    s_t = np.linspace(0.0, 1.0, ncols)
    t_flat = s_t * L_head
    sewn = np.minimum(t_flat, n_in) + np.clip(t_flat - n_in, 0.0, gathered_flat) / full + np.clip(t_flat - L_head + n_in, 0.0, None)
    H3 = sw.at(s_f - sewn)                                     # (from the forearm seam back over the crown)
    v = np.linspace(0.0, 1.0, ROWS)

    def sides(Cs, a_w, n):
        """A seam's line from the armhole down to the wrist, in (angle, z, rho)."""
        Rw = float(D._look(D.R, np.array([z_w]), np.array([np.mod(a_w + math.pi, 2 * math.pi) - math.pi]))[0])
        aw_ = Cs[0] + (np.mod(a_w - Cs[0] + math.pi, 2 * math.pi) - math.pi)
        return np.stack([np.linspace(Cs[0], aw_, n), np.linspace(Cs[1], z_w, n), np.linspace(Cs[2], Rw, n)], 1)
    F3, B3 = sides(Cf, a_fw, ROWS), sides(Ch, a_hw, ROWS)
    # the flat pieces, sampled on the same grid (for the UVs, the gathers' places and the seams' distances)
    Hf = _resample(top.edges['head'], ncols)
    Ff = _resample(top.edges['forearm'][::-1], ROWS)
    Bf = _resample(np.vstack([top.edges['hindarm'], top.edges['vent'][1:]]), ROWS)
    Wf = _resample(top.edges['wrist'][::-1], ncols)
    Xt_f = _coons(Hf, Wf, Ff, Bf, s_t, v)
    ncu = COLS_UNDER
    s_u = np.linspace(0.0, 1.0, ncu)
    hu = und.edges['hindarm']
    lap = und.edges['underlap']
    Huf = _resample(und.edges['head'], ncu)
    Fuf = _resample(und.edges['forearm'][::-1], ROWS)
    Buf = _resample(np.vstack([hu, lap[1:] - [pr['underlap'], 0.0]]), ROWS)
    wu = und.edges['wrist'][::-1]
    Wuf = _resample(np.vstack([wu[wu[:, 0] <= Buf[-1, 0] + 1e-6], Buf[-1:]]), ncu)
    Xu_f = _coons(Huf, Wuf, Fuf, Buf, s_u, v)
    # the hindarm seam down the arm where the topsleeve's own width ends, row by row (each row's cloth round the arm
    # as it was cut), run into the seam's point at the armhole over the head
    wt = np.linalg.norm(np.diff(Xt_f, axis=1), axis=-1).sum(1)
    wu_ = np.linalg.norm(np.diff(Xu_f, axis=1), axis=-1).sum(1)
    a_row = np.empty(ROWS)
    for i in range(ROWS):
        zz = 0.5 * (F3[i, 1] + B3[i, 1])
        _, rg = sw.arc_to(zz, F3[i, 0], 0.0, 1.0)
        a_row[i], _ = sw.arc_to(zz, F3[i, 0], rg * wt[i] / (wt[i] + wu_[i]), 1.0)
    a_row = B3[:, 0] + (np.mod(a_row - B3[:, 0] + math.pi, 2 * math.pi) - math.pi)
    bl = sleeve._ss(0.08, 0.35, v)
    B3[:, 0] = B3[:, 0] * (1 - bl) + a_row * bl
    for i in range(ROWS):
        B3[i, 2] = float(D._look(D.R, np.array([B3[i, 1]]), np.array([np.mod(B3[i, 0] + math.pi, 2 * math.pi) - math.pi]))[0]) * bl[i] + B3[i, 2] * (1 - bl[i])
    Hc = sw.cyl(H3)
    Hc[:, 0] = _unwrap_from(Hc[:, 0], F3[0, 0])
    B3[:, 0] += 2 * math.pi * round((Hc[-1, 0] - B3[0, 0]) / (2 * math.pi))
    Wc, _ = sw.ring(z_w, F3[-1, 0], B3[-1, 0], ncols)
    Xt_c = _coons(Hc, Wc, F3, B3, s_t, v)
    # -- the undersleeve: its head plain under the arm
    Hu3 = sw.at(s_f + s_u * m['under_head'])
    Huc = sw.cyl(Hu3)
    Huc[:, 0] = _unwrap_from(Huc[:, 0], F3[0, 0])
    B3u = B3.copy()
    B3u[:, 0] += 2 * math.pi * round((Huc[-1, 0] - B3u[0, 0]) / (2 * math.pi))
    Wuc, _ = sw.ring(z_w, F3[-1, 0], B3u[-1, 0], ncu)
    Xu_c = _coons(Huc, Wuc, F3, B3u, s_u, v)
    # -- onto the drafted surface (the head on its support, the tube round the arm); the stitching rows stay on the seam
    Xt = sw.cart(Xt_c).reshape(-1, 3)
    Xu = sw.cart(Xu_c).reshape(-1, 3)
    Xt[:ncols] = H3
    Xu[:ncu] = Hu3
    held_t = np.zeros(len(Xt), bool)
    held_t[:ncols] = True
    held_u = np.zeros(len(Xu), bool)
    held_u[:ncu] = True
    Xt = _project_star(D, Xt, keep=held_t)
    Xu = _project_star(D, Xu, keep=held_u)
    # (smoothed as pressed cloth: the drafted surface is found point by point, and shows its facets)
    Xt = _taubin(Xt.reshape(ROWS, ncols, 3), hold_cols=False).reshape(-1, 3)
    # (the seams' lines: the same in both pieces, row by row; the topsleeve's, as projected, is the seam's)
    Xu.reshape(ROWS, ncu, 3)[:, 0] = Xt.reshape(ROWS, ncols, 3)[:, 0]
    Xu.reshape(ROWS, ncu, 3)[:, -1] = Xt.reshape(ROWS, ncols, 3)[:, -1]
    # -- the gathers: the head's extra length, drawn up at the stitching into small folds standing off the surface
    Qt = Xt_f.reshape(-1, 2)
    u = _dist_to(Hf, Qt)
    # (down its own column from the head: the gathers hang from the stitching, they do not run down the seams)
    u = np.maximum(u, (Xt_f[:, :, 1] - Hf[None, :, 1]).ravel())
    tt = np.repeat(t_flat[None], ROWS, 0).ravel()
    inside = sleeve._ss(n_in, n_in + 0.4 * pitch * full, tt) * (1.0 - sleeve._ss(L_head - n_in - 0.4 * pitch * full, L_head - n_in, tt))
    k = _sine_amp(full)
    A = k * pitch                                              # (at the stitching: all the extra length in the folds)
    fade = sleeve._ss(0.0, r['stitch_draw'], u) * np.exp(-u / (r['gather_open'] * w))
    ph = 2 * math.pi * (tt - n_in) / (pitch * full)
    wave = np.sin(ph + 0.35 * np.sin(0.23 * ph))
    Xu = _taubin(Xu.reshape(ROWS, ncu, 3)).reshape(-1, 3)
    nt = _grid_normals(Xt.reshape(ROWS, ncols, 3), D)
    Xt = Xt + (0.5 * A * inside * fade * wave)[:, None] * nt
    # -- the stitching: each seam draws the cloth in along its line (allowances pressed open behind it)
    thick = sleeve.RATIOS['thick'] * w
    v_vent = 1.0 - pr['vent'] / _len(Bf)
    vv_t = np.repeat(v[:, None], ncols, 1).ravel()
    vv_u = np.repeat(v[:, None], ncu, 1).ravel()
    sw_, sd = r['seam_w'], r['seam_depth']
    Qu = Xu_f.reshape(-1, 2)
    d_fore_t, d_hind_t, d_head_t = _dist_to(Ff, Qt), _dist_to(Bf, Qt), u
    d_fore_u, d_hind_u, d_head_u = _dist_to(Fuf, Qu), _dist_to(Buf, Qu), _dist_to(Huf, Qu)
    open_t = vv_t > v_vent
    open_u = vv_u > v_vent

    def draw_in(d):
        return 1.0 - sleeve._ss(0.0, sw_, d)
    seam_t = np.maximum.reduce([draw_in(d_fore_t), draw_in(d_hind_t) * ~open_t, draw_in(d_head_t)])
    seam_u = np.maximum.reduce([draw_in(d_fore_u), draw_in(d_hind_u) * ~open_u, draw_in(d_head_u)])
    nu = _grid_normals(Xu.reshape(ROWS, ncu, 3), D)
    nt = _grid_normals(Xt.reshape(ROWS, ncols, 3), D)
    Xt -= (sd * seam_t)[:, None] * nt
    Xu -= (sd * seam_u)[:, None] * nu
    # -- the vent: the topsleeve's edge lapped over the undersleeve's below the seam
    lap_t = sleeve._ss(v_vent - 0.02, v_vent + 0.01, vv_t) * (1.0 - sleeve._ss(0.0, 0.6 * pr['underlap'], d_hind_t))
    Xt += (r['lap_proud'] * thick * lap_t)[:, None] * nt
    # -- one cloth: the welded seams (the forearm's whole length, the hindarm's down to the vent)
    nT = len(Xt)
    V = np.vstack([Xt, Xu])
    UV = np.vstack([Qt, Qu + [top.outline()[:, 0].max() - und.outline()[:, 0].min() + 4.0 / CM, 0.0]])
    seam = np.concatenate([seam_t, seam_u])
    weld = {}
    for i in range(ROWS):
        weld[nT + i * ncu] = i * ncols                         # (forearm)
        if v[i] <= v_vent:
            weld[nT + i * ncu + ncu - 1] = i * ncols + ncols - 1   # (hindarm)

    def grid_faces(n0, rows, cols, flip):
        F = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                a, b = n0 + i * cols + j, n0 + i * cols + j + 1
                c, d = n0 + (i + 1) * cols + j + 1, n0 + (i + 1) * cols + j
                F.append((a, b, c, d) if not flip else (a, d, c, b))
        return F
    faces = grid_faces(0, ROWS, ncols, False) + grid_faces(nT, ROWS, ncu, True)
    faces = [tuple(weld.get(i, i) for i in f) for f in faces]
    # the cut edges: the sleeve's own (its wrist under the cuff, the vent's edges), fulled
    edge = np.zeros(len(V))
    edge[(ROWS - 1) * ncols:nT] = 1.0
    edge[nT + (ROWS - 1) * ncu:] = 1.0
    # -- the report: how each piece was eased onto the arm (the cloth's length in the sewn shape against the cut)
    rep = []

    def strain(X, Q, rows, cols, name):
        X, Q = X.reshape(rows, cols, 3), Q.reshape(rows, cols, 2)
        dv = np.linalg.norm(np.diff(X, axis=0), axis=-1) / np.maximum(np.linalg.norm(np.diff(Q, axis=0), axis=-1), 1e-9)
        ds = np.linalg.norm(np.diff(X, axis=1), axis=-1) / np.maximum(np.linalg.norm(np.diff(Q, axis=1), axis=-1), 1e-9)
        body_ = slice(rows // 5, rows - 2)
        rep.append('%s: down the grain %.0f%% to %.0f%% of its cut length (median %.0f%%); across %.0f%% to %.0f%% '
                   '(median %.0f%%) below the head' % (name, 100 * np.percentile(dv[body_], 5), 100 * np.percentile(dv[body_], 95),
                                                       100 * np.median(dv[body_]), 100 * np.percentile(ds[body_], 5),
                                                       100 * np.percentile(ds[body_], 95), 100 * np.median(ds[body_])))
    strain(Xt, Qt, ROWS, ncols, 'topsleeve')
    strain(Xu, Qu, ROWS, ncu, 'undersleeve')
    rep.append('head: %.1f cm gathered into %.1f cm (x%.2f), %d gathers %.1f mm deep at the stitching' % (
        gathered_flat * CM, gathered_sewn * CM, full, int(gathered_sewn / pitch), A * CM * 10))
    rep.append('wrist: the sleeve round %.1f cm on the arm against %.1f cm cut' % (ring_w * CM, (tw + uw) * CM))
    # -- the cuff: the band round the wrist from the vent, its ends lapped
    cuff_p = pieces['cuff']
    Lc = cuff_p.length('top')
    dep = pr['cuff']
    z0, z1 = D.z_cuff + 0.05, D.z_hem
    proud = sleeve.RATIOS['cuff_proud'] * w
    nc, nr = int(Lc / (0.15 / CM)), 14
    _, ring_c = sw.arc_to(0.5 * (z0 + z1), a_hw, 0.0, -1.0, proud)
    lap_c = pr['lap']
    xs, ys = np.linspace(0.0, Lc, nc), np.linspace(0.0, dep, nr)
    CU = np.empty((nr, nc, 3))
    for i, yy in enumerate(ys):
        z = z0 + (z1 - z0) * yy / dep
        for j, xx in enumerate(xs):
            # (from the vent's back edge round the front and under, to lap over its start: the closed cuff's round is
            # the band less its lap)
            frac = xx / (Lc - lap_c)
            a_ = a_hw - 2 * math.pi * frac
            Rr = float(D._look(D.R, np.array([z]), np.array([np.mod(a_ + math.pi, 2 * math.pi) - math.pi]))[0]) + proud
            Rr += (r['lap_proud'] * thick) * sleeve._ss(Lc - lap_c - 0.3 * lap_c, Lc - lap_c, xx)
            # (its edges rounded under: the band's fold at the bottom, the seam at the top)
            yn = yy / dep
            Rr -= 0.35 * thick * (np.exp(-(yn / 0.08) ** 2) + np.exp(-((1 - yn) / 0.08) ** 2))
            CU[i, j] = sw.cart(np.array([a_, z, Rr]))
    nV = len(V)
    V = np.vstack([V, CU.reshape(-1, 3)])
    UVc = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2) + [UV[:, 0].max() + 4.0 / CM, 0.0]
    UV = np.vstack([UV, UVc])
    faces += grid_faces(nV, nr, nc, True)
    seam = np.concatenate([seam, np.zeros(nr * nc)])
    ce = np.zeros((nr, nc))
    ce[0], ce[-1], ce[:, 0], ce[:, -1] = 1.0, 1.0, 1.0, 1.0
    edge = np.concatenate([edge, ce.ravel()])
    rep.append('cuff: %.1f cm band, closed round %.1f cm on the arm (lap %.1f)' % (Lc * CM, ring_c * CM, lap_c * CM))
    # -- the topstitching: the cuff's two rows and the vent's edge
    lines = []
    si = r['stitch_in']
    for yy in (si, dep - si):
        i = int(round(yy / dep * (nr - 1)))
        lines.append(CU[i, :, :])
    vt = Xt.reshape(ROWS, ncols, 3)
    col = int(np.argmin(np.abs(np.linalg.norm(np.diff(Xt_f.reshape(ROWS, ncols, 2)[-1], axis=0), axis=1).cumsum()[::-1] - si)))
    lines.append(vt[v > v_vent - 0.01, col])
    # the buttons: on the cuff's lap, up its middle
    j = int(round((Lc - 0.5 * lap_c) / Lc * (nc - 1)))
    btn = [CU[int(round(f * (nr - 1))), j] for f in (0.22, 0.5, 0.78)]
    return dict(V=V, faces=faces, UV=UV, seam=seam, edge=edge, lines=lines, buttons=btn, report=rep, z_tube=D.z_cap)


def _len(c):
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def _stitches(lines, w, side, ratios=RATIOS, material=None, name='Sleeve stitches'):
    """The topstitching as thread: small stitches laid along each line, sunk half into the cloth."""
    bm = bmesh.new()
    L, g = ratios['stitch_len'], ratios['stitch_gap']
    rad = 0.012 * w
    for ln in lines:
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ln, axis=0), axis=1))])
        t = 0.0
        while t + L < s[-1]:
            a = np.array([np.interp(t, s, ln[:, k]) for k in range(3)])
            b = np.array([np.interp(t + L, s, ln[:, k]) for k in range(3)])
            ret = bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=4, radius=1.0)
            d = b - a
            ln_ = np.linalg.norm(d)
            rot = Vector((1, 0, 0)).rotation_difference(Vector((d / max(ln_, 1e-9)).tolist())).to_matrix()
            c = 0.5 * (a + b)
            for vv in ret['verts']:
                p = Vector((vv.co.x * 0.5 * ln_, vv.co.y * rad, vv.co.z * rad))
                p = rot @ p
                vv.co = Vector(((c[0] + p.x) * side, c[1] + p.y, c[2] + p.z))
            t += L + g
    if side < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    return o


def _buttons_at(pts, D, side, w, material=None, name='Sleeve buttons'):
    bm = bmesh.new()
    rb = sleeve.RATIOS['button'] * w
    for c in pts:
        cx, cy = D.axis(c[2])
        n = np.array([c[0] - cx, c[1] - cy, 0.0])
        n /= np.linalg.norm(n)
        ret = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=rb)
        rot = Vector((0, 0, 1)).rotation_difference(Vector(n.tolist())).to_matrix()
        for v in ret['verts']:
            p = rot @ Vector((v.co.x, v.co.y, v.co.z * 0.45 + 0.15 * rb + sleeve.RATIOS['thick'] * w))
            v.co = Vector(((c[0] + p.x) * side, c[1] + p.y, c[2] + p.z))
    if side < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    return o


def build_sewn(body, M, rig=None, material=None, button_material=None, log=print, name='Redingote sleeve'):
    """Cut both sleeves from the flat pattern drafted on this body, sew them into the shared armhole and to their
    cuffs, and hand them to the body's bones (no solver). Returns (sleeves, buttons, stitches, Draft, report)."""
    P, T, aw = fit.body_arrays(body)
    w = M['w']
    loops = armhole.armhole(P, aw, M, T)
    if rig is not None and 'Forearm.L' in rig.data.bones:
        b = rig.data.bones['Forearm.L']
        z_elbow, z_wrist = float((rig.matrix_world @ b.head_local).z), float((rig.matrix_world @ b.tail_local).z)
    else:
        gi = {g.name: g.index for g in body.vertex_groups}
        fore = [v.index for v in body.data.vertices if any(g.group == gi.get('Forearm.L') and g.weight > 0.5 for g in v.groups)]
        z_elbow, z_wrist = float(P[fore, 2].max()), float(P[fore, 2].min())
    D = sleeve.Draft(P, T, aw, M, loops, z_elbow, z_wrist, sleeve.RATIOS, log)
    m, _ = pattern.measure(P, T, aw, M, z_elbow, z_wrist)
    pieces = pattern.draft(m)
    for line in pattern.report(pieces, m):
        log('pattern: ' + line)
    cloth = cut_and_sew(D, m, loops, pieces, log=log)
    for line in cloth['report']:
        log('sewn: ' + line)
    arm_skin = sleeve._arm_skin(body, aw, rig) if rig is not None else None
    sleeves, buttons, stitches = [], [], []
    for side, sfx in ((1.0, 'L'), (-1.0, 'R')):
        V = cloth['V'] * np.array([side, 1.0, 1.0])
        me = bpy.data.meshes.new('%s.%s' % (name, sfx))
        bm = bmesh.new()
        lay = bm.loops.layers.uv.new('UVMap')
        used = sorted({i for f in cloth['faces'] for i in f})
        vs = {i: bm.verts.new(V[i]) for i in used}
        for f in cloth['faces']:
            if len(set(f)) < 3:
                continue
            ff = f if side > 0 else f[::-1]
            try:
                face = bm.faces.new([vs[i] for i in ff])
            except ValueError:
                continue
            for lp, i in zip(face.loops, ff):
                lp[lay].uv = (float(cloth['UV'][i][0]), float(-cloth['UV'][i][1]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.verts.index_update()
        order = np.array(used)
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
        rest.data.foreach_set('vector', [c for v in me.vertices for c in v.co])
        ed = me.attributes.new('edge', 'FLOAT', 'POINT')
        ed.data.foreach_set('value', cloth['edge'][order].tolist())
        sm = me.attributes.new('seam', 'FLOAT', 'POINT')
        sm.data.foreach_set('value', cloth['seam'][order].tolist())
        o = bpy.data.objects.new('%s.%s' % (name, sfx), me)
        bpy.context.scene.collection.objects.link(o)
        if material is not None:
            me.materials.append(material)
        sleeve._skin(o, body, P, aw, D, side)
        if rig is not None:
            mo = o.modifiers.new('rig', 'ARMATURE')
            mo.object = rig
            mo.use_deform_preserve_volume = True
            g = o.vertex_groups.new(name='keep off')
            zr = np.array([v.co.z for v in me.vertices])
            kw = 1.0 - sleeve._ss(D.z_cap, D.z_under, zr)
            for i in np.nonzero(kw > 1e-3)[0]:
                g.add([int(i)], float(kw[i]), 'REPLACE')
            mo = o.modifiers.new('keep off', 'SHRINKWRAP')
            mo.target = arm_skin
            mo.wrap_method = 'NEAREST_SURFACEPOINT'
            mo.wrap_mode = 'OUTSIDE_SURFACE'
            mo.offset = sleeve.KEEP_OFF * w
            mo.vertex_group = 'keep off'
        mo = o.modifiers.new('smooth', 'SUBSURF')
        mo.levels = mo.render_levels = 1
        mo = o.modifiers.new('thickness', 'SOLIDIFY')
        mo.thickness = sleeve.RATIOS['thick'] * w
        mo.offset = 1.0
        mo.use_rim = True
        bt = _buttons_at(cloth['buttons'], D, side, w, button_material, name='%s buttons.%s' % (name, sfx))
        st = _stitches(cloth['lines'], w, side, material=material, name='%s stitches.%s' % (name, sfx))
        for extra in (bt, st):
            sleeve._copy_weights(extra, o)
            if rig is not None:
                mo = extra.modifiers.new('rig', 'ARMATURE')
                mo.object = rig
        sleeves.append(o)
        buttons.append(bt)
        stitches.append(st)
    return sleeves, buttons, stitches, D, cloth['report']
