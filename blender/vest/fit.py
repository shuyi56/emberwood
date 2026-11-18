"""A fitted vest with a stand collar, drafted on whatever body it is given.

Nothing here is placed by hand. The body is measured first (measure()), the way a tailor takes a client's
measurements, and every line of the garment is a ratio of those measurements (RATIOS), so the vest follows the
body when the body changes: a longer neck gives a taller collar, a fuller bust a deeper front, wider shoulders a
wider yoke.

The garment is made as one piece and then cut:

  1. the whole piece  a shell lying on the torso and running on up the neck, without a break. Below the neckline
                      it is the body's own surface, bridged straight across the hollows cloth cannot follow (between
                      the breasts, down the spine) and let out by the ease. Above the neckline it is the collar's
                      stand, drafted as a tailor drafts it: a straight band from the neckline to the neck's girth at
                      the collar's top plus its ease, so it leans in with the neck. The two run into each other over
                      a rounded fillet instead of meeting at a seam.
  2. the trims        the hem, the armholes, and the front (the V and the collar's top edge as one line), each a
                      cut through the finished shell along a measured line. A cut only removes: the surface that
                      is left is the shell's, so the edges need no seams or facings.

build_vest() returns the one mesh, skinned to the body's own bones. Given a plate material, that mesh is the lining
and the pattern: the plates are raised over it (brass_layer()), cut from the same shell, so they sit on it and re-fit
with it."""
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

# Every size as a share of a measurement. w is the neck's half-width (the garment's unit), L the neck's length in
# front (pit of the neck to the chin).
RATIOS = {
    'ease': 0.06,            # the gap between body and vest, in w (form fitting: about 3 mm on a 165 cm woman)
    'thickness': 0.05,       # the leather's thickness, in w
    'neck_base': 1.12,       # the side neck point: where the neck's base is this many times the neck's width
    'back_drop': 0.25,       # the back neck point below the side neck point, in w (a tailor's back neck rise)
    'collar_top': 0.95,      # the collar's top edge at the side: this share of L above the pit of the neck
    'collar_rise': 0.10,     # how much higher the top edge is at the back than at the side, in L
    'collar_gap': 0.15,      # the gap between the neck and the collar's top edge, in w
    'collar_fillet': 0.42,   # the radius of the rounded run from shoulder into collar, in w
    'head_clear': 0.12,      # the least gap between the collar and the head (jaw, skull), in w
    'v_point': 0.72,         # the bottom of the V: this share of the way from the bust line up to the pit of the neck
    'v_top': 1.05,           # half the opening at the collar's top edge, in w
    'v_bow': 2.00,           # the V's edge: 1 straight, below 1 bowed outward, above 1 drawn in to a cusp at the bottom
    'corner': 0.08,          # the radius of the collar's front corners, in w
    'hem_drop': 0.60,        # the hem below the waist, in w
    'hem_point': 0.50,       # the front point of the hem below that, in w
    'shoulder': 0.94,        # the vest's shoulder ends at this share of the way out to the shoulder point: on the
                             # shoulder's flat top, short of where it rolls over into the arm
    'brass_thick': 0.045,    # the brass plates' thickness, in w
    'brass_band': 0.26,      # the border of leather left showing along the front line, in w (0.7 of it at the
                             # armholes and the hem)
    'belt_z': 0.42,          # the belt under the bust: this share of the way from the bust line down to the waist
    'belt_half': 0.24,       # half its height, in w
    'belt_dip': 0.30,        # how far it dips to a point in front, in w
    'roll': 0.9,             # the height of the plate's rolled edge, in plate thicknesses
    'dome': 0.7,             # how far a lame is raised along its middle, in plate thicknesses
    'lame': 0.85,            # the height of one lame showing, in w
    'lame_lap': 0.22,        # how far a lame laps over the one below, in w
    'lame_slope': 0.80,      # how steeply the lames rise from the middle outward
    'stud': 0.085,           # a stud's radius, in w
    'stud_step': 0.62,       # the studs' spacing along the belt, in w
    'scye_drop': 0.30,       # the armhole's bottom below the armpit, in w
    'scye_ease': 0.10,
    'scoop_front': 0.30,     # how far the armhole is hollowed in front of the arm, in w (a tailor's front scye curve)
    'scoop_back': 0.15,      # and behind it       # the armhole's ease round the arm, in w
}
STEP = 0.1                   # the shell's grid, in units
COLS = 360


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _blur(a, sigma, axis, wrap):
    """A Gaussian blur along one axis (wrapped round for the angle, held at the ends for the height)."""
    if sigma <= 0:
        return a
    n = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.moveaxis(a, axis, 0)
    pad = np.concatenate([a[-n:], a, a[:n]]) if wrap else np.concatenate([np.repeat(a[:1], n, 0), a, np.repeat(a[-1:], n, 0)])
    out = sum(k[i] * pad[i:i + len(a)] for i in range(2 * n + 1))
    return np.moveaxis(out, 0, axis)


def body_arrays(body):
    """The body's rest shape: points, triangles, and each point's share carried by the arm's bones."""
    me = body.data
    P = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', P)
    P = P.reshape(-1, 3)
    mw = np.array(body.matrix_world)
    P = P @ mw[:3, :3].T + mw[:3, 3]
    me.calc_loop_triangles()
    T = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get('vertices', T)
    T = T.reshape(-1, 3)
    arm_groups = {g.index for g in body.vertex_groups if g.name.startswith(('UpperArm', 'Forearm', 'Hand'))}
    aw = np.zeros(len(P))
    for v in me.vertices:
        for g in v.groups:
            if g.group in arm_groups:
                aw[v.index] += g.weight
    return P, T, aw


def measure(P, aw, head=None, ratios=RATIOS):
    """The tailor's measurements, read off the body's rest shape (and the head's points, for the chin).

    w, z_col     the neck's half-width and the height it is narrowest at
    yc           the neck's axis, front to back
    z_side       the side neck point: where the neck's base has widened to neck_base * w
    z_pit        the front neck point: the pit of the neck, the notch between the collarbones' inner ends
    z_back       the back neck point
    z_chin, L    the chin, and the neck's length in front
    x_sh, z_sh   the shoulder point, and scye_*: the ring where the arm leaves the body (from the skin's own bone
                 weights: the points carried half by the arm), its lowest point the armpit
    z_waist, z_bust, half-widths and depths there"""
    M = {}
    band = (aw > 0.4) & (aw < 0.6)
    if band.sum() < 20:
        raise RuntimeError('vest: the body has no arm bones to find the armholes by')
    B = P[band]
    hi = B[B[:, 2] >= np.percentile(B[:, 2], 97)]
    M['x_sh'], M['z_sh'] = float(np.abs(hi[:, 0]).mean()), float(hi[:, 2].mean())
    M['z_pit_arm'] = float(np.percentile(B[:, 2], 1))
    M['scye_y'] = (float(np.percentile(B[:, 1], 1)), float(np.percentile(B[:, 1], 99)))
    # the neck: the narrowest section above the shoulder point
    top = P[:, 2].max()
    zs = np.arange(M['z_sh'] + 0.5, M['z_sh'] + 0.75 * (top - M['z_sh']), 0.1)     # (the skin's top end is a stub inside the head)
    idx = np.round((P[:, 2] - zs[0]) / 0.1).astype(int)
    ok = (idx >= 0) & (idx < len(zs))
    W = np.zeros(len(zs))
    np.maximum.at(W, idx[ok], np.abs(P[ok, 0]))
    W = _blur(W, 1.0, 0, False)
    w = float(W.min())
    col = int(np.argmax(W <= 1.03 * w))             # the lowest height the neck is a column at
    M['w'], M['z_col'] = w, float(zs[col])
    sec = P[np.abs(P[:, 2] - M['z_col']) < 0.1]
    M['yc'] = float(0.5 * (sec[:, 1].min() + sec[:, 1].max()))
    M['girth_neck'] = float(math.pi * (3 * (w + 0.5 * np.ptp(sec[:, 1])) - math.sqrt((3 * w + 0.5 * np.ptp(sec[:, 1])) * (w + 1.5 * np.ptp(sec[:, 1])))))
    wide = np.nonzero(W[:col + 1] >= ratios['neck_base'] * w)[0]
    M['z_side'] = float(zs[wide[-1]]) if len(wide) else M['z_col'] - 0.25 * w
    M['z_back'] = M['z_side'] - ratios['back_drop'] * w
    # the pit of the neck: going up the front, the first height the middle lies behind the collarbones' inner ends
    front = P[:, 1] < M['yc']
    z_pit = None
    for z in np.arange(M['z_side'] - 1.5 * w, M['z_side'], 0.05):
        row = front & (np.abs(P[:, 2] - z) < 0.06)
        mid = row & (np.abs(P[:, 0]) < 0.08 * w)
        off = row & (np.abs(np.abs(P[:, 0]) - 0.5 * w) < 0.08 * w)
        if mid.any() and off.any() and P[mid, 1].min() - P[off, 1].min() > 0.06 * w:
            z_pit = float(z)
            break
    M['z_pit'] = z_pit if z_pit is not None else M['z_side'] - 0.62 * w
    M['pit_found'] = z_pit is not None
    # the chin: the lowest point of the head in front of the throat
    z_chin = None
    if head is not None and len(head):
        thr = sec[:, 1].min()
        f = head[head[:, 1] < thr - 0.2 * w]
        if len(f):
            z_chin = float(f[:, 2].min())
    M['z_chin'] = z_chin if z_chin is not None else M['z_pit'] + 1.6 * w
    M['L'] = M['z_chin'] - M['z_pit']
    # the torso (the points the arms do not carry): the waist its narrowest, the bust its deepest in front
    T = P[aw < 0.5]
    zt = np.arange(M['z_pit_arm'] - 5.0 * w, M['z_pit_arm'], 0.1)
    it = np.round((T[:, 2] - zt[0]) / 0.1).astype(int)
    ok = (it >= 0) & (it < len(zt))
    Wt, Yf, Yb = np.zeros(len(zt)), np.full(len(zt), 1e9), np.full(len(zt), -1e9)
    np.maximum.at(Wt, it[ok], np.abs(T[ok, 0]))
    np.minimum.at(Yf, it[ok], T[ok, 1])
    np.maximum.at(Yb, it[ok], T[ok, 1])
    Wt = _blur(Wt, 2.0, 0, False)
    lo, hi_ = 0, int(len(zt) * 0.7)
    iw = lo + int(np.argmin(Wt[lo:hi_]))
    ib = iw + int(np.argmin(Yf[iw:]))
    M['z_waist'], M['waist_half'] = float(zt[iw]), float(Wt[iw])
    M['z_bust'], M['bust_front'], M['bust_half'] = float(zt[ib]), float(Yf[ib]), float(Wt[ib])
    M['bust_depth'] = float(Yb[ib] - Yf[ib])
    M['waist_depth'] = float(Yb[iw] - Yf[iw])
    return M


def scye(P, aw, M, ratios=RATIOS):
    """The armhole's line, seen from the front, as a tailor cuts it: from the shoulder point down and in to the
    armpit, then out again under the arm to the side of the chest, scye_drop below the armpit. Returns x(z): how far
    out from the middle the cloth may reach at a height (no limit below the armhole)."""
    w = M['w']
    z_b = M['z_pit_arm'] - ratios['scye_drop'] * w
    side = P[(aw < 0.5) & (np.abs(P[:, 2] - z_b) < 0.1)]
    x_side = float(np.abs(side[:, 0]).max()) + (ratios['ease'] + 0.2) * w       # past the chest's side at the bottom
    band = P[(aw > 0.4) & (aw < 0.6)]
    low = band[band[:, 2] <= np.percentile(band[:, 2], 3)]
    x_pit = float(np.abs(low[:, 0]).mean()) - ratios['scye_ease'] * w

    x_end = ratios['shoulder'] * M['x_sh']

    def x_at(z):
        z = np.asarray(z, dtype=float)
        t = np.clip((z - z_b) / (M['z_sh'] - z_b), 0.0, 1.0)
        t_pit = (M['z_pit_arm'] - z_b) / (M['z_sh'] - z_b)
        # (out under the arm only below the armpit: above it the line runs straight up to the shoulder's end)
        x = x_pit + (x_end - x_pit) * t ** 1.3 + (x_side - x_pit) * np.clip(1.0 - t / t_pit, 0.0, 1.0) ** 2
        return np.where(z < z_b, 1e3, x)
    return x_at, z_b


def _hull(pts):
    """The convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return pts

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(reversed(pts))
    return lo[:-1] + up[:-1]


def _taut(r, sn, cs):
    """One level section as cloth pulled taut round it: the radii of the section's convex hull."""
    x, y = -r * sn, -r * cs
    h = np.array(_hull(np.stack([x, y], 1)))
    a, b = h, np.roll(h, -1, 0)
    e = b - a
    dx, dy = -sn[:, None], -cs[:, None]
    den = dx * e[None, :, 1] - dy * e[None, :, 0]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
    s = (a[None, :, 0] * dy - a[None, :, 1] * dx) / den
    t = np.where((s >= -1e-6) & (s <= 1 + 1e-6) & (t > 0), t, -1.0)
    out = t.max(1)
    return np.where(out > 0, out, r)


def whole_piece(P, T, aw, M, head_bvh=None, ratios=RATIOS, log=print):
    """Step 1: the uncut shell, a grid of points [row, column]: rows from below the hem up the body to the neckline
    and on up the collar's stand, columns round from the front (-Y) toward the figure's right (-X).
    Returns the grid, the row the neckline is at, and the collar's top edge (a height for each column)."""
    w, yc = M['w'], M['yc']
    ease = ratios['ease'] * w
    phi = 2 * math.pi * np.arange(COLS) / COLS
    sn, cs = np.sin(phi), np.cos(phi)
    # the body as rays see it from the neck's axis. The torso first; where a ray leaves through an arm's socket,
    # the shoulder's own surface above the armpit, and never further out than just past the shoulder point (the
    # armhole is cut from that part, so the arm itself is never dressed)
    torso = (aw[T] < 0.5).all(1)
    cap = ~torso & (P[T][:, :, 2] > M['z_pit_arm']).all(1)
    bt = BVHTree.FromPolygons([Vector(p) for p in P], [tuple(int(i) for i in t) for t in T[torso]])
    bc = BVHTree.FromPolygons([Vector(p) for p in P], [tuple(int(i) for i in t) for t in T[cap]]) if cap.any() else None
    x_scye, _ = scye(P, aw, M, ratios)
    z_lo = M['z_waist'] - (ratios['hem_drop'] + ratios['hem_point'] + 0.5) * w
    z_hi = min(P[:, 2].max() - 0.15, M['z_chin'] + ratios['collar_rise'] * M['L'] + 0.6 * w)
    dz = STEP / 2
    zs = np.arange(z_lo, z_hi, dz)
    R = np.empty((len(zs), COLS))
    for i, z in enumerate(zs):
        o = Vector((0.0, yc, z))
        x_clip = float(x_scye(z)) + 0.3 * w          # (a little past the armhole's line, for the cut to remove)
        for j in range(COLS):
            d = Vector((-sn[j], -cs[j], 0.0))
            hit = bt.ray_cast(o, d)[0]
            r = (hit - o).length if hit is not None else 1e9
            lim = x_clip / abs(sn[j]) if abs(sn[j]) > 1e-6 else 1e9
            if r > lim and bc is not None and z > M['z_pit_arm']:
                hit = bc.ray_cast(o, d)[0]
                r = (hit - o).length if hit is not None else 1e9
            R[i, j] = min(r, lim)
    if not np.isfinite(R).all() or R.max() > 1e8:
        raise RuntimeError('vest: the neck axis is not inside the body at every height')
    body_r = R.copy()
    for i in range(len(zs)):
        R[i] = _taut(R[i], sn, cs)
    R = np.maximum(_blur(_blur(R, 0.12 / dz, 0, False), 2.0, 1, True), R - 0.25 * ease)

    def at(z, j):
        """The taut body's radius at a height, for a column (or all of them)."""
        f = np.clip((z - zs[0]) / dz, 0, len(zs) - 1.001)
        i = np.floor(f).astype(int)
        t = f - i
        return R[i, j] * (1 - t) + R[i + 1, j] * t

    # the neckline: through the pit of the neck, the side neck points and the back neck point
    a = 0.25 * (M['z_pit'] + M['z_back']) + 0.5 * M['z_side']
    c = a - M['z_side']
    b = 0.5 * (M['z_pit'] - M['z_back'])
    z_neck = a + b * cs + c * np.cos(2 * phi)
    # each column's line up the body, hem to neckline, stepped evenly along its own length
    n_body = int(math.ceil((M['z_side'] - z_lo + M['x_sh']) / STEP))
    G = np.empty((n_body, COLS, 3))
    for j in range(COLS):
        m = zs <= z_neck[j]
        zz = np.append(zs[m], z_neck[j])
        rr = np.append(R[m, j], at(z_neck[j], j))
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rr), np.diff(zz)))])
        u = np.linspace(0.0, s[-1], n_body)
        r_, z_ = np.interp(u, s, rr), np.interp(u, s, zz)
        G[:, j] = np.stack([-r_ * sn[j], yc - r_ * cs[j], z_], 1)
    # let out by the ease, square to the surface (a radial ease would vanish on the shoulders' flat tops)
    di = np.gradient(G, axis=0)
    dj = (np.roll(G, -1, 1) - np.roll(G, 1, 1)) * 0.5
    N = np.cross(dj, di)
    N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-9)
    out = np.stack([-sn, -cs, np.zeros(COLS)], 1)[None]
    N = np.where((N * out).sum(2, keepdims=True) < -0.2, -N, N)
    G = G + ease * N
    # the collar's stand, as a tailor drafts it: its top edge is the neck's girth there plus the ease, its bottom
    # edge the neckline, and the band between them straight
    z_top = M['z_pit'] + ratios['collar_top'] * M['L'] - ratios['collar_rise'] * M['L'] * cs
    r_top = at(z_top, np.arange(COLS))
    F = np.fft.rfft(r_top)
    F[5:] = 0                                   # (a stiff band's edge is a smooth oval; it does not follow the tendons)
    r_top = np.fft.irfft(F, COLS) + ratios['collar_gap'] * w
    tip = np.stack([-r_top * sn, yc - r_top * cs, z_top], 1)
    base = G[-1]
    run = tip - base
    over = 1.0 + (0.6 * w) / np.maximum(run[:, 2], 0.2 * w)        # (the shell runs on past the top edge, to be cut)
    n_stand = int(math.ceil(float((np.linalg.norm(run, axis=1) * over).max()) / STEP))
    u = (np.arange(1, n_stand + 1) / n_stand)[:, None, None] * over[None, :, None]
    S = base[None] + u * run[None]
    # (never inside the neck: the band stands at least the ease off it)
    rs = np.hypot(S[:, :, 0], S[:, :, 1] - yc)
    need = at(S[:, :, 2], np.arange(COLS)[None, :].repeat(n_stand, 0)) + ease
    k = np.maximum(need / np.maximum(rs, 1e-6), 1.0)
    S[:, :, 0] *= k
    S[:, :, 1] = yc + (S[:, :, 1] - yc) * k
    G = np.concatenate([G, S])
    # the shoulder runs into the collar over a fillet, not a seam: the corner at the neckline rounded off
    rows = np.arange(len(G))
    span = ratios['collar_fillet'] * w / STEP
    bump = np.exp(-((rows - (n_body - 1)) / span) ** 2)[:, None, None]
    for _ in range(int(2 * span * span)):
        avg = G.copy()
        avg[1:-1] = 0.25 * G[:-2] + 0.5 * G[1:-1] + 0.25 * G[2:]
        G = G + bump * (avg - G)
    # the top edge kept clear of the head: lowered wherever the jaw or the skull comes within head_clear of the band
    if head_bvh is not None:
        clear = ratios['head_clear'] * w
        cap_z = z_top.copy()
        for j in range(COLS):
            for i in range(n_body, len(G)):
                if G[i, j, 2] > z_top[j]:
                    break
                near = head_bvh.find_nearest(Vector(G[i, j]), clear + ratios['thickness'] * w)
                if near[0] is not None:
                    cap_z[j] = G[i, j, 2] - STEP
                    break
        low = cap_z < z_top - 1e-6
        if low.any():
            m = cap_z.copy()
            for s in range(1, 17):
                m = np.minimum(m, np.minimum(np.roll(cap_z, s), np.roll(cap_z, -s)))
            z_top = np.minimum(z_top, _blur(m, 8.0, 0, True) - 0.02 * w)
            log('vest: collar top lowered for the head at %d of %d columns, by up to %.2f' % (low.sum(), COLS, float((M['z_pit'] + ratios['collar_top'] * M['L'] - ratios['collar_rise'] * M['L'] * cs - z_top).max())))
    return G, n_body - 1, z_top, body_r, zs


def _smin(a, b, k):
    """The smaller of two, with the corner between them rounded to radius k."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1 - h)


def trim_lines(M, z_top, x_scye, ratios=RATIOS):
    """Step 2's cutting lines, each a function of a point that is positive on the cloth that stays."""
    w, yc = M['w'], M['yc']

    def ang(p):
        return np.arctan2(-p[:, 0], -(p[:, 1] - yc))

    def top_at(p):
        return np.interp(np.mod(ang(p), 2 * math.pi), 2 * math.pi * np.arange(COLS + 1) / COLS, np.append(z_top, z_top[0]))

    def hem(p):
        a = ang(p)
        z = M['z_waist'] - ratios['hem_drop'] * w - ratios['hem_point'] * w * np.exp(-(a / 0.45) ** 2)
        return p[:, 2] - z

    y0, y1 = M['scye_y']
    ya, ay = 0.5 * (y0 + y1), 0.5 * (y1 - y0)

    def armholes(p):
        # (seen from above the line is a curve too: furthest out over the top of the arm, hollowed in front of it
        # and behind it, so the edge rounds the shoulder instead of turning a corner at its front)
        u = (p[:, 1] - ya) / ay
        scoop = np.where(u < 0, ratios['scoop_front'], ratios['scoop_back']) * w * np.minimum(u * u, 1.0)
        scoop = scoop * _ss(M['z_pit_arm'], 0.5 * (M['z_pit_arm'] + M['z_sh']), p[:, 2])     # (none under the arm)
        return x_scye(p[:, 2]) - scoop - np.abs(p[:, 0])

    z_v = M['z_bust'] + ratios['v_point'] * (M['z_pit'] - M['z_bust'])
    z_f = float(z_top[0])

    def front(p):
        # (the V and the collar's top edge are one line: down the top edge, round the front corner, down the V)
        s = np.clip((p[:, 2] - z_v) / (z_f - z_v), 0.0, None)
        half = ratios['v_top'] * w * s ** ratios['v_bow']
        v = np.where(p[:, 1] < yc, np.abs(p[:, 0]) - half, 1e3)
        return _smin(top_at(p) - p[:, 2], v, ratios['corner'] * w)

    return [('hem', hem), ('armholes', armholes), ('front and collar top', front)], {'z_v': z_v}


def trim(V, faces, f, snap=0.012):
    """Cut a mesh along f = 0 and keep the side where f > 0. Faces the line crosses are cut exactly along it."""
    d = f(V)
    d[np.abs(d) < snap] = 0.0
    new, made, out = [], {}, []
    n0 = len(V)
    for poly in faces:
        dd = d[poly]
        if dd.min() >= 0:
            if dd.max() > 0:
                out.append(poly)
            continue
        if dd.max() <= 0:
            continue
        res, n = [], len(poly)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            da, db = d[a], d[b]
            if da >= 0:
                res.append(a)
            if da * db < 0:
                key = (a, b) if a < b else (b, a)
                if key not in made:
                    t = da / (da - db)
                    made[key] = n0 + len(new)
                    new.append(V[a] + t * (V[b] - V[a]))
                res.append(made[key])
        if len(res) >= 3:
            out.append(res)
    if new:
        V = np.concatenate([V, np.array(new)])
    return V, out


def build_vest(body, head_objs=(), rig=None, material=None, name='Vest', ratios=RATIOS, log=print, steps=None,
               brass=None):
    """Measure the body, make the whole piece, cut it, and hand it to the body's bones. Returns (object, measures).
    steps, if a list, is filled with (name, points, faces) after the whole piece and after each cut."""
    P, T, aw = body_arrays(body)
    hp, hb = None, None
    if head_objs:
        pts, tris = [], []
        for o in head_objs:
            base = len(pts)
            pts += [o.matrix_world @ v.co for v in o.data.vertices]
            o.data.calc_loop_triangles()
            tris += [tuple(base + i for i in t.vertices) for t in o.data.loop_triangles]
        hp = np.array([p[:] for p in pts])
        hb = BVHTree.FromPolygons(pts, tris)
    M = measure(P, aw, hp, ratios)
    log('vest measures: ' + ', '.join('%s %s' % (k, ('%.2f' % v) if isinstance(v, float) else v) for k, v in M.items()))
    G, neck_row, z_top, body_r, zs = whole_piece(P, T, aw, M, hb, ratios, log)
    rows, cols = G.shape[:2]
    V = G.reshape(-1, 3)
    idx = np.arange(rows * cols).reshape(rows, cols)
    faces = np.stack([idx[:-1], np.roll(idx[:-1], -1, 1), np.roll(idx[1:], -1, 1), idx[1:]], 2).reshape(-1, 4)
    faces = [list(q) for q in faces]
    if steps is not None:
        steps.append(('whole piece', V.copy(), faces))
    lines, extra = trim_lines(M, z_top, scye(P, aw, M, ratios)[0], ratios)
    M.update(extra)
    for nm, f in lines:
        V, faces = trim(V, faces, f)
        log('vest: trimmed %s, %d faces left' % (nm, len(faces)))
        if steps is not None:
            steps.append((nm, V.copy(), faces))
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert(Vector(p), i)
    kd.balance()
    th = ratios['thickness'] * M['w']
    o = _sheet(name, [(V, faces)], M, material, body, kd, rig, lift=0.0, thickness=th)
    if brass is not None:
        pieces, studs, binding = brass_layer(V, faces, M, lines, ratios, log)
        _sheet(name + ' plates', pieces, M, brass, body, kd, rig, lift=0.0, thickness=ratios['brass_thick'] * M['w'])
        if studs:
            _sheet(name + ' studs', studs, M, brass, body, kd, rig, lift=0.0, thickness=0.0)
        if binding:
            _sheet(name + ' binding', binding, M, material, body, kd, rig, lift=0.0, thickness=0.6 * th)
    return o, M


def _sheet(name, pieces, M, material, body, kd, rig, lift, thickness):
    """A mesh object from cut pieces [(points, faces)]: lifted off the shell square to it, given its thickness
    outward, and handed to the body's bones (each point takes the weights of the skin under it)."""
    bm = bmesh.new()
    for piece in pieces:
        V, faces = piece[0], piece[1]
        up = piece[2] if len(piece) > 2 else lift          # (a piece's own lift: a number, or a function of its points)
        coord = piece[3] if len(piece) > 3 else None       # (and its own paint coordinate: a function of its points)
        pb = bmesh.new()
        vmap = {i: pb.verts.new(V[i]) for i in sorted({i for f in faces for i in f})}
        for f in faces:
            try:
                pb.faces.new([vmap[i] for i in f])
            except ValueError:
                pass
        bmesh.ops.remove_doubles(pb, verts=pb.verts, dist=0.004)
        bmesh.ops.recalc_face_normals(pb, faces=pb.faces)
        pb.normal_update()
        if thickness:
            # (outward is away from the neck's axis and up: an open sheet has no inside of its own to tell it by)
            flip = []
            for f in pb.faces:
                c = f.calc_center_median()
                if f.normal.dot(Vector((c.x, c.y - M['yc'], 0.6 * abs(f.normal.z) + 0.05)).normalized()) < 0:
                    flip.append(f)
            if flip:
                bmesh.ops.reverse_faces(pb, faces=flip)
                pb.normal_update()
        lay = pb.verts.layers.float.new('paint_u')          # (where each point lies across its plate, for the paint)
        if coord is not None:
            for v, c in zip(pb.verts, coord(np.array([v.co[:] for v in pb.verts]))):
                v[lay] = float(c)
        if callable(up):
            hs = up(np.array([v.co[:] for v in pb.verts]))
            for v, h in zip(pb.verts, hs):
                v.co += v.normal * float(h)
        elif up:
            for v in pb.verts:
                v.co += v.normal * up
        tmp = bpy.data.meshes.new('piece')
        pb.to_mesh(tmp)
        pb.free()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', [c for v in me.vertices for c in v.co])
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    groups = {g.index: o.vertex_groups.new(name=g.name) for g in body.vertex_groups
              if g.name.split('.')[0] in ('Pelvis', 'Chest', 'Neck', 'UpperArm')}
    bverts = body.data.vertices
    for v in me.vertices:
        near = kd.find_n(v.co, 4)
        acc, tot = {}, 0.0
        for _, bi, dist in near:
            k = 1.0 / (dist + 0.05)
            for g in bverts[bi].groups:
                if g.group in groups:
                    acc[g.group] = acc.get(g.group, 0.0) + k * g.weight
                    tot += k * g.weight
        for gi, wt in acc.items():
            if wt / tot > 1e-3:
                groups[gi].add([v.index], wt / tot, 'REPLACE')
    if rig is not None:
        m = o.modifiers.new('rig', 'ARMATURE')
        m.object = rig
        m.use_deform_preserve_volume = True
    if thickness:
        m = o.modifiers.new('thickness', 'SOLIDIFY')
        m.thickness = thickness
        m.offset = 1.0
        m.use_rim = True
        m = o.modifiers.new('edge', 'BEVEL')
        m.width = 0.45 * thickness
        m.segments = 2
        m.limit_method = 'ANGLE'
        m.angle_limit = math.radians(50)
    return o


def brass_layer(V, faces, M, lines, ratios=RATIOS, log=print):
    """The plates, made as a smith makes them: the leather vest is the pattern, and the plate is raised over it.
    Every plate is the vest's own surface cut again along measured lines (so it lies on the leather exactly and
    follows the body with it), set in from the leather's edges so the lining shows as a border all round, its own
    edges there rolled into a rounded bead.

      the lames    bands of plate round the whole body from the hem up, rising from the middle of the chest (and of
                   the back) out toward the shoulders, lapping like armour: each is tucked under the lame above at
                   its top edge and stands a plate's thickness proud at its bottom edge, where it laps the one below,
                   and each is raised into a shallow round across its width. The yoke and the collar are one plate,
                   over the highest lame
      the belt     a band of plate under the bust, dipping to a point in front, over the lames and set with studs

    Returns the plates, each (points, faces, lift), the studs, and the leather binding's pieces."""
    w, yc = M['w'], M['yc']
    f_arm, f_front = dict(lines)['armholes'], dict(lines)['front and collar top']
    band = ratios['brass_band'] * w
    th, pl = ratios['thickness'] * w, ratios['brass_thick'] * w
    lo = th + 0.15 * pl                 # a plate lying on the leather
    hi = lo + 1.5 * pl                  # and lying on another plate

    def ang(p):
        return np.arctan2(-p[:, 0], -(p[:, 1] - yc))

    def cut(piece, *fields):
        v, f = piece
        for fn in fields:
            if not f:
                break
            v, f = trim(v, f, fn)
        return v, f

    whole = (V, faces)
    z_belt = M['z_bust'] - ratios['belt_z'] * (M['z_bust'] - M['z_waist'])
    half = ratios['belt_half'] * w

    def belt_mid(a):
        return z_belt - ratios['belt_dip'] * w * np.exp(-(a / 0.3) ** 2)

    # the lames: t runs up the chest and falls away to the sides, so a line of equal t is a chevron
    period, slope, lap = ratios['lame'] * w, ratios['lame_slope'], ratios['lame_lap'] * w

    def t_of(p):
        return p[:, 2] - slope * np.abs(p[:, 0])

    # the plate stops short of the leather's edges, so the lining shows as a border all round it, and its own
    # edges there are rolled: turned up into a rounded bead, as a smith finishes an edge that lies against the body
    f_hem = dict(lines)['hem']

    def inset(p):
        return np.minimum(np.minimum(f_front(p) - band, f_arm(p) - 0.7 * band), f_hem(p) - 0.7 * band)

    def roll(p):
        return ratios['roll'] * pl * np.exp(-(inset(p) / (0.3 * band)) ** 2)

    def dome(u):
        # (a lame is not flat across: it is raised into a shallow round, highest along its middle)
        return ratios['dome'] * pl * np.sin(math.pi * np.clip(u, 0.0, 1.0))

    field = cut(whole, lambda p: f_front(p) - band, lambda p: f_arm(p) - 0.7 * band, lambda p: f_hem(p) - 0.7 * band)
    tv = t_of(field[0][sorted({i for f in field[1] for i in f})])
    t1 = M['z_pit'] - 0.5 * w                           # (above this the yoke and the collar are one plate)
    n = int(math.ceil((t1 - float(tv.min())) / period))
    t0 = t1 - n * period
    plates = []
    for k in range(n):
        a, b = t0 + k * period - lap, t0 + (k + 1) * period
        v, f = cut(field, lambda p, a=a: t_of(p) - a, lambda p, b=b: b - t_of(p))
        if f:
            plates.append((v, f, lambda p, a=a, b=b: lo + (hi - lo) * np.clip((b - t_of(p)) / (b - a), 0.0, 1.0)
                           + dome((t_of(p) - a) / (b - a)) + roll(p), lambda p, a=a, b=b: (t_of(p) - a) / (b - a)))
    v, f = cut(field, lambda p: t_of(p) - (t1 - lap))
    if f:
        plates.append((v, f, lambda p: lo + (hi - lo) * np.clip((t1 + period - t_of(p)) / (period + lap), 0.0, 1.0) + roll(p),
                       lambda p: (t_of(p) - (t1 - lap)) / (period + lap)))
    top = hi + (1.2 + ratios['dome']) * pl
    belt = cut(field, lambda p: half - np.abs(p[:, 2] - belt_mid(ang(p))))
    log('vest plates: %d lames, belt %d faces' % (len(plates), len(belt[1])))
    if belt[1]:
        plates.append((belt[0], belt[1], lambda p: top + dome(0.5 + 0.5 * (p[:, 2] - belt_mid(ang(p))) / half),
                       lambda p: 0.5 + 0.5 * (p[:, 2] - belt_mid(ang(p))) / half))
    binding = []
    # the studs: half-round heads along the belt's middle, evenly spaced along its length
    tris = []
    for f in faces:
        tris += [(f[0], f[i], f[i + 1]) for i in range(1, len(f) - 1)]
    tree = BVHTree.FromPolygons([Vector(p) for p in V], tris)
    out = top + (1.0 + ratios['dome']) * pl
    studs, a = [], 0.0
    ball = bmesh.new()
    bmesh.ops.create_uvsphere(ball, u_segments=12, v_segments=8, radius=ratios['stud'] * w)
    bv = np.array([v.co[:] for v in ball.verts])
    bf = [[v.index for v in f.verts] for f in ball.faces]
    ball.free()
    while a < 2 * math.pi - 1e-6:
        d = Vector((-math.sin(a), -math.cos(a), 0.0))
        o = Vector((0.0, yc, float(belt_mid(np.array([a if a < math.pi else a - 2 * math.pi]))[0])))
        hit, nrm = tree.ray_cast(o, d)[:2]
        if hit is None:
            a += 0.05
            continue
        if nrm.dot(d) < 0:
            nrm = -nrm
        studs.append((bv + np.array((hit + nrm * out)[:]), bf))
        a += ratios['stud_step'] * w / max((hit - o).length, 0.5 * w)
    return plates, studs, binding


def clearance(vest, body, head_objs=(), log=print):
    """How the vest sits on the posed body at the current frame: the least and the mean gap to the torso's skin and
    how many of its points are inside it (the arm lying against the vest's side is counted apart: that is the arm
    pressing on the cloth, not the cloth cutting into the body), and the least gap to the head."""
    dg = bpy.context.evaluated_depsgraph_get()

    def tree(o):
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        me.calc_loop_triangles()
        pts = [o.matrix_world @ v.co for v in me.vertices]
        tris = [tuple(t.vertices) for t in me.loop_triangles]
        e.to_mesh_clear()
        return BVHTree.FromPolygons(pts, tris), tris
    tb, tris = tree(body)
    aw = body_arrays(body)[2]
    ve = vest.evaluated_get(dg)
    vm = ve.to_mesh()
    pts = [vest.matrix_world @ v.co for v in vm.vertices]
    ve.to_mesh_clear()
    gaps, inside, arm = [], 0, 0
    for p in pts:
        loc, nrm, fi, dist = tb.find_nearest(p)
        s = dist if (p - loc).dot(nrm) >= 0 else -dist
        on_arm = len(aw) > max(tris[fi]) and aw[list(tris[fi])].mean() > 0.5
        if on_arm:
            arm += s < 0
            continue
        gaps.append(s)
        inside += s < 0
    out = {'min': float(min(gaps)), 'mean': float(np.mean(gaps)), 'inside': int(inside), 'arm': int(arm), 'points': len(pts)}
    if head_objs:
        hd = 1e9
        for o in head_objs:
            th = tree(o)[0]
            for p in pts[::3]:
                near = th.find_nearest(p, 2.0)
                if near[0] is not None:
                    hd = min(hd, near[3])
        out['head'] = float(hd)
    log('vest clearance: torso gap min %.3f mean %.3f, %d of %d points inside the torso, %d against the arm; head gap %s' % (
        out['min'], out['mean'], out['inside'], out['points'], out['arm'], ('%.3f' % out['head']) if 'head' in out else 'n/a'))
    return out
