"""The eye socket (orbit), the lids and the eyeball, derived from the measured head and carved into its mesh.

One shared construction for every head the project builds from the ring grid (the Face Creator's head, the game
heads exported from it, and the Blender head in blender/base/female.py). The Face Creator carries a JavaScript
port of this file (orbitCarve in face-creator/main.js); the two are kept line for line, so a change here goes
there too. Pure numpy: no Blender, so the topology check (rq_topo.py) and the draft views run on it directly.

Units are the model's (the head HEAD_H tall, the face toward -y, z up, x across; the eye's side s is -1 for the
eye at x < 0). Every size comes from the head's measure (spec): the head height sets the millimetre, the
eye spacing places the pupils, the brow and the cheek set how far forward the eye stands.

The order of the work, as John asked: the socket is carved first, then the eye is set in it.

  1. spec()      measure the head and derive the eye from it (life sizes scaled to the head)
  2. bed()       the surface the lids rest on: the eyeball (sclera with the corneal bulge) and, past its edge,
                 the soft tissue of the socket that carries on from it
  3. carve()     cut a block of the head's ring grid round each eye and fill it with a patch of concentric edge
                 loops: the skin of the lids and the folds round them blending out into the head, the lid's
                 margin (lash line, a rounded margin, the wet line on the ball), and the conjunctiva running back
                 under the lids round the ball and closed behind it, so the head has no hole
  4. globe()     the eyeball mesh, the same surface the lids were laid on, with the iris and pupil sized to it
  5. caruncle()  the caruncle and the tear meniscus in the inner corner

Topology (the face topology check, rq_topo.py): every ring of the patch is one closed edge loop round the eye,
with a spoke through each of its vertices, so every vertex in the eye's zone has four edges. The four 5-edge
poles that joining a ring patch to a grid needs sit at the block's corners, outside the eye's zone; the four
3-edge poles that closing the sac needs sit behind the ball's equator, where nothing sees them or bends them.
"""
import math

import numpy as np

# ------------------------------------------------------------------------------------------------ life sizes (mm)
# Adult female averages (oculoplastic and anthropometric references: Farkas 1994, Bergin, Gray's Anatomy):
LIFE = {
    'head_h': 219.0,         # chin to crown (vertex), the scale of the millimetre on the head
    'globe_r': 12.0,         # eyeball: 24 mm across
    'cornea_r': 7.8,         # the cornea's radius of curvature
    'limbus_r': 5.85,        # the cornea's (and the visible iris's) radius: 11.7 mm across
    'iris_back': 0.5,        # the iris lies this far behind the limbus's plane
    'film': 0.06,            # the tear film between the lids and the ball
    'margin_t': 1.8,         # the lid's thickness at its margin
    'lid_t': 1.4,            # the skin and muscle of the lid over the ball, away from the margin
    'cover_up': 1.5,         # the upper lid covers this much of the top of the iris
    'cover_lo': 0.3,         # and the lower lid this much of its foot
    'lift': 1.3,             # the line between the corners passes this far below the pupil
    'lateral': 1.1,          # the outer corner sits this far out from the pupil, in eyeball radii (13 mm: with the
                             # inner corner at IPD / 4 from the midline the opening is 28.5 mm, the pupil near its middle)
    'medial': 0.25,          # the inner corner sits this far from the midline, in IPDs (intercanthal = IPD / 2)
    'crease': 8.0,           # the upper lid's crease above the lash line
    'crease_d': 0.5,         # and how deep it folds
    'lower_fold': 9.0,       # the fold where the lower lid meets the cheek, below the lash line
    'brow_ahead': 10.0,      # the cornea's apex stands this far behind the brow, in profile
    'cheek_behind': 2.0,     # and this far behind the cheek's front below it (Hertel's figures put the lateral
                             # rim 16-18 mm behind the apex; the soft cheek over the zygoma stands about level)
    'caruncle_w': 5.0,       # the caruncle, in the inner corner
}
# the lids' shape (the traced photograph, female.py's EYE_*): heights in half-widths over the corner line
LID = {'top_peak': (-0.18, 0.62), 'bot_low': (0.30, 0.32), 'inner_angles': (50.0, 13.0),
       'outer_angles': (44.0, 30.0), 'inner_run': 0.05, 'inner_tangent': 0.8, 'peak_full': (1.25, 1.2),
       'outer_tangent': 0.5}


def smooth(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def facing(P, F, ref, toward=False):
    """Wind the faces F so their normals point away from ref (a point or [n, 3] of points), or toward it."""
    P = np.asarray(P, float)
    out = []
    for f in F:
        a, b, c = P[f[0]], P[f[1]], P[f[-1]]
        n = np.cross(P[f[1]] - a, P[f[2]] - a)
        cen = P[list(f)].mean(0)
        d = float(np.dot(n, cen - np.asarray(ref, float)))
        out.append(tuple(f) if (d >= 0) != toward else tuple(f)[::-1])
    return out


def smax(a, b, k):
    """A smooth maximum (polynomial, k the width of the blend)."""
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return b + (a - b) * h + k * h * (1 - h)


# ------------------------------------------------------------------------------------------------ 1. the measure

def lid_curve(top, L=LID):
    """One lid's edge as (u, height) samples, u -1 at the inner corner .. 1 at the outer, height in half-widths
    over the line between the corners, before scaling (female.py's _lid_curve, without EYE_OPEN)."""
    i = 0 if top else 1
    pu, ph = L['top_peak'] if top else L['bot_low']
    ai, ao = math.radians(L['inner_angles'][i]), math.radians(L['outer_angles'][i])
    dx, dy = math.cos(ai), math.sin(ai)
    run = L['inner_run']

    def hermite(p0, t0, p1, t1, n=240):
        s = np.arange(n) / n
        h00, h10, h01, h11 = 2 * s ** 3 - 3 * s * s + 1, s ** 3 - 2 * s * s + s, -2 * s ** 3 + 3 * s * s, s ** 3 - s * s
        return np.stack([h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0],
                         h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]], 1)
    q = (-1.0 + run * dx, run * dy)
    k = np.arange(20) / 20
    pts = [np.stack([-1.0 + run * dx * k, run * dy * k], 1)]
    c0 = math.hypot(pu - q[0], ph - q[1])
    c1 = math.hypot(1 - pu, ph)
    pf = L['peak_full'][i]
    pts.append(hermite(q, (L['inner_tangent'] * c0 * dx, L['inner_tangent'] * c0 * dy), (pu, ph), (pf * c0, 0.0)))
    pts.append(hermite((pu, ph), (pf * c1, 0.0), (1.0, 0.0),
                       (L['outer_tangent'] * c1 * math.cos(ao), -L['outer_tangent'] * c1 * math.sin(ao))))
    pts.append(np.array([[1.0, 0.0]]))
    P = np.concatenate(pts)
    P = P[np.argsort(P[:, 0], kind='stable')]
    return P[:, 0], np.maximum(0.0, P[:, 1])


def spec(H, ipd, pupil_z, front_y, scale=1.0, tilt=0.092, life=None, almond=None):
    """The eye derived from the head. H: chin to crown; ipd: between the pupils; pupil_z: their height;
    front_y(x, z): the depth of the head's front (before any eye) at a point, in model units. scale: the eye's
    size against life (1 is life size; the face's shape and the eye spacing stay as measured).
    Returns a dict of model-unit sizes, positions (for the eye at x > 0; the other is its mirror) and the
    measures they came from.
    almond: {'top': heights, 'bot': heights, 'lift_ratio': r}: the opening takes the Face Creator's own almond for its
    shape: the lids' curves (401 samples over u -1..1, in half-widths) at their own heights, and the corner line
    lift_ratio half-widths below the pupil. The corners and the ball stay as measured."""
    Lf = dict(LIFE, **(life or {}))
    mm = H / Lf['head_h']
    e = mm * scale
    r = Lf['globe_r'] * e
    Rc, lim = Lf['cornea_r'] * e, Lf['limbus_r'] * e
    zl = math.sqrt(r * r - lim * lim)             # the limbus's plane, from the ball's centre
    cc = zl - math.sqrt(Rc * Rc - lim * lim)      # the cornea's centre of curvature, from the ball's centre
    apex = cc + Rc                                # the cornea's apex, from the ball's centre
    px = ipd / 2
    # how far forward: the brow (the most forward point of the front over the pupil, above it) and the cheek
    # (the same, below it) set the cornea's apex, between 10 mm behind the brow and 2 mm behind the cheek
    zs_b = pupil_z + H * np.linspace(0.06, 0.14, 17)
    zs_c = pupil_z - H * np.linspace(0.06, 0.13, 15)
    brow = min(front_y(px, z) for z in zs_b)
    cheek = min(front_y(px, z) for z in zs_c)
    apex_y = 0.5 * ((brow + Lf['brow_ahead'] * mm) + (cheek + Lf['cheek_behind'] * mm))
    # the opening: the corners from the eye spacing and the ball, the lids' heights from the iris
    med = px - Lf['medial'] * ipd * scale        # the inner corner, from the pupil (toward the nose)
    lat = Lf['lateral'] * r                       # the outer corner, from the pupil
    a = 0.5 * (med + lat)                         # the opening's half-width
    xc = 0.5 * (lat - med)                        # its centre, out from the pupil
    lift = Lf['lift'] * e
    iris_r = lim
    us, ht = lid_curve(True)
    _, hb = lid_curve(False)
    if almond:
        us = np.linspace(-1, 1, len(almond['top']))
        ht, hb = np.asarray(almond['top'], float), np.asarray(almond['bot'], float)
    u0 = -xc / a
    kt = ((iris_r - Lf['cover_up'] * e) + lift) / (a * float(np.interp(u0, us, ht)))
    kb = ((iris_r - Lf['cover_lo'] * e) - lift) / (a * float(np.interp(u0, us, hb)))
    if almond:
        kt = kb = 1.0
        lift = almond['lift_ratio'] * a
    S = {'mm': mm, 'e': e, 'H': H, 'ipd': ipd, 'scale': scale,
         'centre': (px, apex_y + apex, pupil_z),          # the ball's centre (the eye at x > 0)
         'r': r, 'cornea_R': Rc, 'cornea_c': cc, 'apex': apex, 'limbus': lim, 'limbus_z': zl,
         'iris_r': iris_r, 'iris_back': Lf['iris_back'] * e, 'film': Lf['film'] * e,
         'margin_t': Lf['margin_t'] * e, 'lid_t': Lf['lid_t'] * e,
         'med': med, 'lat': lat, 'a': a, 'xc': xc, 'lift': lift, 'tilt': tilt, 'kt': kt, 'kb': kb,
         'crease': Lf['crease'] * e, 'crease_d': Lf['crease_d'] * e, 'lower_fold': Lf['lower_fold'] * e,
         'caruncle_w': Lf['caruncle_w'] * e,
         'lid_u': us, 'lid_top': ht, 'lid_bot': hb,
         'measured': {'brow_y': brow, 'cheek_y': cheek, 'apex_y': apex_y, 'pupil_z': pupil_z,
                      'fissure_mm': 2 * a / mm, 'aperture_mm': None}}
    up, lo = margin_point(S, 0.0, True), margin_point(S, 0.0, False)
    S['measured']['aperture_mm'] = (up[1] - lo[1]) / mm
    return S


def margin_point(S, xl, top):
    """The wet line (where the lid meets the ball) in the front view, at xl out from the pupil (lateral +):
    (xl, z) for the eye at x > 0, z in model units."""
    u = np.clip((np.asarray(xl, float) - S['xc']) / S['a'], -1, 1)
    zline = S['centre'][2] - S['lift'] + S['tilt'] * np.asarray(xl, float)
    if top:
        h = S['kt'] * S['a'] * np.interp(u, S['lid_u'], S['lid_top'])
        return xl, zline + h
    h = S['kb'] * S['a'] * np.interp(u, S['lid_u'], S['lid_bot'])
    return xl, zline - h


# ------------------------------------------------------------------------------------------------ 2. the bed

def globe_radius(S, th, k=None):
    """The ball's radius at th radians from its front (the sclera's sphere with the cornea's bulge, joined by a
    smooth limbal step)."""
    th = np.asarray(th, float)
    r, Rc, cc = S['r'], S['cornea_R'], S['cornea_c']
    sn = np.sin(np.minimum(th, math.pi / 2))
    tc = cc * np.cos(th) + np.sqrt(np.maximum(0.0, Rc * Rc - (cc * sn) ** 2))
    tc = np.where(th < math.pi / 2, tc, 0.0)
    return smax(np.full_like(th, r), tc, (0.35 * S['e']) if k is None else k)


def _profile(S, soft=False):
    """The bed's profile: depth from the ball's centre (y, + back) against the distance d from its axis in the
    front view. Two surfaces joined by a smooth minimum (so the joint is a soft fillet, always in front of both):
    the ball itself (its cornea's bulge softened, as the lid drapes it), and the soft tissue of the socket round
    it, a shallow bowl that meets the ball about 70 degrees round from its front (the tarsal plates follow the ball that far) and carries on gently past its
    edge to the corners. Being in front of the ball everywhere, nothing laid on it can cut into the ball."""
    key = '_prof_soft' if soft else '_prof'
    if key in S:
        return S[key]
    r = S['r']
    D = np.linspace(0, 1.75 * r, 1200)
    th = np.linspace(0, math.pi / 2, 4000)
    # (soft: the cornea's bulge smoothed out further, for the lids' outer skin: the tarsal plate bridges the
    # bulge rather than following it; the wet line itself lies on the true ball)
    R = globe_radius(S, th, k=(2.5 if soft else 0.9) * S['e'])
    dd, yy = R * np.sin(th), -R * np.cos(th)
    o = np.argsort(dd)
    ball = np.where(D < dd.max(), np.interp(D, dd[o], yy[o]), 0.0)
    tissue = -0.55 * r + 0.22 * D * D / r
    Y = -smax(-ball, -tissue, 0.1 * r)
    g = np.gradient(Y, D)
    S[key] = (D, Y)
    S[key + '_slope'] = g
    return D, Y


def bed(S, x, z, soft=False):
    """The depth (y) of the bed at (x, z) for the eye at x > 0: the ball where it shows, the socket's tissue past
    its edge. Vectorised."""
    D, Y = _profile(S, soft)
    g = S['_prof_soft_slope' if soft else '_prof_slope']
    cx, cy, cz = S['centre']
    d = np.hypot(np.asarray(x, float) - cx, np.asarray(z, float) - cz)
    return cy + np.interp(d, D, Y) + np.where(d > D[-1], g[-1] * (d - D[-1]), 0.0)


def bed_slope(S, x, z):
    """|grad bed| at (x, z): the lids' thickness is measured along the bed's normal, not along y."""
    D, Y = _profile(S)
    cx, _, cz = S['centre']
    d = np.hypot(np.asarray(x, float) - cx, np.asarray(z, float) - cz)
    return np.interp(d, D, S['_prof_slope'])



# ------------------------------------------------------------------------------------------------ the skin's surface

def _relax(Y, fixed, iters):
    """Jacobi steps of Laplace's equation on the free cells of Y (fixed cells kept)."""
    for _ in range(iters):
        A = 0.25 * (np.roll(Y, 1, 0) + np.roll(Y, -1, 0) + np.roll(Y, 1, 1) + np.roll(Y, -1, 1))
        Y = np.where(fixed, Y, A)
    return Y


def _harmonic(Y, fixed, levels=5):
    """Laplace's equation over the free cells, coarse to fine (each level solved from the one below it), so the
    fill converges in a few hundred steps whatever the size."""
    if levels == 0 or min(Y.shape) < 12:
        return _relax(Y, fixed, 600)
    # the coarser level: 2 x 2 cells into one, fixed where any of them is
    h, w = (Y.shape[0] // 2) * 2, (Y.shape[1] // 2) * 2
    Yc = Y[:h, :w].reshape(h // 2, 2, w // 2, 2)
    Fc = fixed[:h, :w].reshape(h // 2, 2, w // 2, 2)
    cnt = Fc.sum((1, 3))
    yc = np.where(cnt > 0, (Yc * Fc).sum((1, 3)) / np.maximum(cnt, 1), Yc.mean((1, 3)))
    yc = _harmonic(yc, cnt > 0, levels - 1)
    up = np.repeat(np.repeat(yc, 2, 0), 2, 1)
    Y2 = Y.copy()
    Y2[:h, :w] = np.where(fixed[:h, :w], Y[:h, :w], up)
    return _relax(Y2, fixed, 120)


def _blur(Y, n):
    k = np.exp(-np.linspace(-2.5, 2.5, 2 * n + 1) ** 2)
    k /= k.sum()
    P = np.pad(Y, n, mode='edge')
    P = np.apply_along_axis(lambda r: np.convolve(r, k, mode='valid'), 1, P)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode='valid'), 0, P)


def skin_field(S, head_y, box, M, nh, step=None):
    """The skin round the eye as one smooth surface y(x, z) over the box (x0, x1, z0, z1) (the eye at x > 0):
    the lid on its bed near the opening, the head's own surface outside the box, and between them the smooth
    (harmonic, then eased) surface that spans from one to the other, with the upper lid's crease and the fold
    under the lower lid pressed into it. M [nb, 2]: the wet line (corners at 0 and nh). Returns a function
    (x, z) -> y."""
    e = S['e']
    h = step or 0.3 * e
    x0, x1, z0, z1 = box
    pad = 6 * h
    xs = np.arange(x0 - pad, x1 + pad + h, h)
    zs = np.arange(z0 - pad, z1 + pad + h, h)
    X, Z = np.meshgrid(xs, zs)
    # distance to the wet line, and which lid (above or below the line between the corners)
    dense = np.concatenate([_resample(M[:nh + 1], 400), _resample(np.concatenate([M[nh:], M[:1]]), 400)])
    from_up = np.arange(len(dense)) < 400
    dd = np.full(X.shape, np.inf)
    near_up = np.zeros(X.shape, bool)
    along = np.zeros(X.shape)            # where along its lid the nearest point is: 0 at a corner, 1 midway
    frac = np.abs(np.sin(math.pi * (np.arange(len(dense)) % 400) / 399.0))
    for k in range(0, len(dense), 40):
        q = dense[k:k + 40]
        d = np.hypot(X[..., None] - q[:, 0], Z[..., None] - q[:, 1])
        m = d.min(-1)
        better = m < dd
        dd = np.where(better, m, dd)
        near_up = np.where(better, from_up[k + d.argmin(-1)], near_up)
        along = np.where(better, frac[k + d.argmin(-1)], along)
    # inside the opening: between the lids' curves at that x
    cx, cy, cz = S['centre']
    xl = X - cx
    zt = margin_point(S, xl, True)[1]
    zb = margin_point(S, xl, False)[1]
    inside = (xl > -S['med']) & (xl < S['lat']) & (Z < zt) & (Z > zb)
    dist = np.where(inside, -dd, dd)
    # the lid on its bed
    nrm = np.minimum(1.5, np.sqrt(1 + bed_slope(S, X, Z) ** 2))
    thick = S['lid_t'] + (S['margin_t'] - S['lid_t']) * np.exp(-(np.maximum(dist, 0) / (1.2 * e)) ** 2)
    L = bed(S, X, Z, soft=True) - thick * nrm
    Hh = np.array([[head_y(xx, zz) for xx in xs] for zz in zs])
    # (the head arrives with its own sculpted eye hollow; the carve replaces it, so the skin is built over the
    # head with that hollow filled: the head where it stands proud of the smooth span across the box, the span
    # where the head falls back from it. Built over the hollow, the skin dipped into a trough round the lids)
    outside0 = (X < x0) | (X > x1) | (Z < z0) | (Z > z1)
    # (the span is a Coons patch over the box, cubic along both rows and columns: it leaves the head on all four
    # sides with the head's own depth and slope, the face's plane over a shut eye. A harmonic membrane sagged back
    # toward the temple and left the lids standing proud of the face; columns alone missed the nose's slope and
    # drew the box's sides as lines)
    def hermite_fill(V, coord, lo, hi, axis):
        # cubic across [lo, hi] along the axis (0 rows: z; 1 columns: x), from the values and slopes just outside
        n = V.shape[axis]
        a_ = max(1, int(np.searchsorted(coord, lo, 'left')) - 1)
        b_ = min(n - 2, int(np.searchsorted(coord, hi, 'right')))
        take = lambda k: np.take(V, k, axis=axis)
        va, vb = take(a_), take(b_)
        sa, sb = (take(a_) - take(a_ - 1)) / h, (take(b_ + 1) - take(b_)) / h
        Ln = coord[b_] - coord[a_]
        t = np.clip((coord - coord[a_]) / Ln, 0, 1)
        sh = [1, 1]
        sh[axis] = -1
        t = t.reshape(sh)
        if axis == 0:
            va, vb, sa, sb = va[None], vb[None], sa[None], sb[None]
        else:
            va, vb, sa, sb = va[:, None], vb[:, None], sa[:, None], sb[:, None]
        h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
        out = h00 * va + h10 * Ln * sa + h01 * vb + h11 * Ln * sb
        inside_ = (coord >= coord[a_]) & (coord <= coord[b_])
        return np.where(inside_.reshape(sh), out, V)
    Lv = hermite_fill(Hh, zs, z0, z1, 0)
    Lu = hermite_fill(Hh, xs, x0, x1, 1)
    span = Lu + Lv - hermite_fill(Lv, xs, x0, x1, 1)
    span = _blur(span, max(2, int(round(2.5 * e / h))))
    edge0 = np.minimum.reduce([X - x0, x1 - X, Z - z0, z1 - Z])
    # (the span replaces the head inside the box outright, eased in over 6 mm: it meets the head with its depth and
    # slope, so the hand-over is smooth. Kept wherever it stood proud of the span, the head's own hollow drew its
    # rim round the eye as a ring, and a smooth minimum's offset where the two agreed drew the box's outline)
    Hh = Hh + (span - Hh) * smooth(0.0, 6.0 * e, edge0)
    # (the lids' tissue is tallest over the middle of the eye and narrows to almost nothing at the corners: a
    # full reach carried round a corner spread the lid's deep bed onto the side of the nose)
    # (the lid lies on the ball only as far as the face lets it: past that, the fat under the skin fills the
    # socket, so the lid stands no deeper behind the face's plane than a shallow sulcus above and a faint trough
    # below. Followed round the ball to the crease, it fell 10 mm behind the brow into a ring-shaped moat)
    # (which lid, as a weight that turns over smoothly across the line out of each corner: a hard switch made a
    # step along that line wherever the lids' reach or depth differ, a bright streak out of both canthi)
    wu = _blur(near_up.astype(float), max(2, int(round(2.0 * e / h))))
    # (deeper toward the corners: the canthi are tied back to the bone, the inner one deepest, so there the lid
    # follows the ball down instead of bridging to the face's plane. Held near the plane, the inner third of the
    # opening stood off the ball and showed the socket's lining)
    sulcus = (1.5 + 1.0 * wu) * e + (2.5 + 2.5 * smooth(0.5 * S['a'], -0.5 * S['a'], xl - S['xc'])) * e * (1 - smooth(0.0, 0.85, along))
    L_bed = L
    L = -smax(-L, -(Hh + sulcus), 4.0 * e)
    # (the inner corner: the lid apposed to the ball down to the canthus, which only the small lacrimal lake and the
    # caruncle separate from the eyeball (StatPearls, Lateral Canthoplasty; Elsevier, Medial angle of eye). Held to the
    # face's plane there, the margin stood 10 mm off the ball and the skin showed through it from 3/4: the clamp is
    # eased out on the corner's own weight. tools/face-creator/apply-inner-corner.py)
    # (the extra depth is taken only at the margin and added to the surface after the hand-over, faded out over 8e:
    # put back into L, it was handed to the face over the corner's narrow 5e band and stepped down beside the nose.
    # tools/face-creator/apply-inner-corner-blend.py)
    w_med = smooth(0.5 * S['a'], -0.5 * S['a'], xl - S['xc']) * (1 - smooth(0.0, 0.85, along))
    Dm = (L_bed - L) * w_med * (1 - smooth(0.5 * e, 2.0 * e, dist))
    reach = (wu * S['crease'] + (1 - wu) * 0.8 * S['lower_fold']) * (0.3 + 0.7 * along ** 0.6)
    # the fill is held to the lid well inside the lid's reach, then the lid hands over to it across a band
    # 3 mm wide (a hard edge to the held cells drew the reach's outline as a ridge)
    core = dist <= reach - 2.5 * e
    outside = (X < x0) | (X > x1) | (Z < z0) | (Z > z1)
    fixed = core | outside
    # (solved for the change to the head, not for the surface itself: the change is held to the lid in the core
    # and to nothing outside the box, and spreads smoothly between, so the head keeps its own curvature round
    # the eye; a harmonic surface flattened it into a membrane that met the head at an angle along the border)
    Dl = np.where(core, L - Hh, 0.0)
    Dl = _harmonic(Dl, fixed)
    nb_ = max(2, int(round(2.5 * e / h)))
    Dl = Dl + (_blur(Dl, nb_) - Dl) * (1 - outside) * smooth(reach - 3.5 * e, reach, dist)
    Y = Hh + Dl
    hand = smooth(reach - 2.5 * e, reach + 2.5 * e, dist)
    Y = L + (Y - L) * hand
    Y = Y + _blur(Dm, max(2, int(round(1.5 * e / h)))) * (1 - smooth(1.0 * e, 8 * e, dist))
    Y = Y + (_blur(Y, nb_) - Y) * (1 - outside) * smooth(1.5 * e, 3.5 * e, dist) * (1 - smooth(0, 6 * h, np.minimum.reduce([X - x0, x1 - X, Z - z0, z1 - Z])) * 0)
    # the crease of the upper lid and the soft fold under the lower lid, fading toward the corners and the border
    corner = smooth(0.0, 0.3, np.clip(1 - np.abs((xl - S['xc']) / S['a']), 0, 1))
    toward_border = smooth(0.0, 4 * e, np.minimum.reduce([X - x0, x1 - X, Z - z0, z1 - Z]))
    cr = wu * S['crease_d'] * np.exp(-((dist - S['crease']) / (1.1 * e)) ** 2) * corner
    fo = (1 - wu) * 0.35 * S['crease_d'] * np.exp(-((dist - S['lower_fold']) / (1.8 * e)) ** 2) * corner
    Y = Y + (cr + fo) * toward_border
    # never closer to the ball and its socket than a thin lid
    # (over the ball itself only: the socket's tissue past its edge is only what the lids rest on at the corners;
    # held under the skin there it pushed the temple out into a bump wherever the head lies further back)
    r = S['r']
    dc = np.hypot(X - cx, Z - cz)
    ball = cy - np.sqrt(np.maximum(0.0, r * r - np.minimum(dc, r) ** 2))
    floor = ball - 0.6 * S['lid_t'] * nrm - 3 * S['film']
    Y = Y + (-smax(-Y, -floor, 0.4 * S['lid_t']) - Y) * smooth(1.02 * r, 0.9 * r, dc)

    # into the head's own surface over the last 6 mm inside the box, so depth and slope both carry across the
    # border (held only at the border, the fill met the head at an angle: a line along the block's top and foot)
    edge = np.minimum.reduce([X - x0, x1 - X, Z - z0, z1 - Z])
    Y = Hh + (Y - Hh) * smooth(0.0, 6.0 * e, edge)

    def at(x, z):
        fx = np.clip((np.asarray(x) - xs[0]) / h, 0, len(xs) - 1.001)
        fz = np.clip((np.asarray(z) - zs[0]) / h, 0, len(zs) - 1.001)
        i, j = fx.astype(int), fz.astype(int)
        tx, tz = fx - i, fz - j
        return (Y[j, i] * (1 - tx) * (1 - tz) + Y[j, i + 1] * tx * (1 - tz) + Y[j + 1, i] * (1 - tx) * tz +
                Y[j + 1, i + 1] * tx * tz)
    at.grid = (xs, zs, Y)
    return at

# ------------------------------------------------------------------------------------------------ 3. the carve

def _cycle(i0, i1, j0, j1, jc):
    """The block's border as grid (i, j), from the outer (i1) side's middle row jc, up, across the top to the
    inner side, down, across the bottom and back up: its upper half and lower half the same length."""
    c = [(i1, j) for j in range(jc, j1)] + [(i, j1) for i in range(i1, i0, -1)] + \
        [(i0, j) for j in range(j1, j0, -1)] + [(i, j0) for i in range(i0, i1)] + [(i1, j) for j in range(j0, jc)]
    return c


def _resample(P, n):
    """n points evenly spaced by arc length along the polyline P [m, 2] (both ends kept)."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)


def carve(G, Z, TH, block, S, head_y, out_rings=26, in_rings=9, side=1):
    """Carve one eye's socket into the head's ring grid.

    G [NR, NS, 3]: the grid's points (rows up the head, NS segments round it, periodic). block: (i0, i1, j0, j1)
    in grid indices for the eye at x > 0 (the eye at x < 0 mirrors it: side -1). head_y(x, z): the head's
    front depth (the surface the block's border lies on). Returns (P [m, 3] the patch's new points, the border's
    grid (i, j) in order, the patch's quads as rows of 4 indices into [border points, new points], and the indices
    into P of: the lash line, the wet line, the cap's corners)."""
    i0, i1, j0, j1 = block
    jc = (j0 + j1) // 2
    assert (j1 - jc) == (jc - j0), 'the block needs an odd number of rows, its corner row in the middle'
    cyc = _cycle(i0, i1, j0, j1, jc)
    if side < 0:                       # mirrored: the outer side is i0 (x < 0)
        cyc = [(i0 + i1 - i, j) for i, j in cyc]
    nb = len(cyc)
    nh = nb // 2
    Bxyz = np.array([G[j, i] for i, j in cyc])
    B = np.stack([side * Bxyz[:, 0], Bxyz[:, 2]], 1)           # the border in the front view, as the eye at x > 0
    cx, cy, cz = S['centre']
    # the wet line, in the front view: the upper lid from the outer corner to the inner, the lower lid back; each
    # half resampled by arc length to the border's half, so each spoke runs from the border to the lid
    xs = np.linspace(S['lat'], -S['med'], 800)
    up = np.stack(margin_point(S, xs, True), 1)
    lo = np.stack(margin_point(S, xs[::-1], False), 1)
    # (by the border's own arc length, so the spokes run roughly straight in from it)
    def by_border(L, Bh):
        sb = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Bh, axis=0), axis=1))])
        sb /= sb[-1]
        # ease the lid's sampling toward the border's angle round the eye's centre, so spokes run radially
        sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))])
        sl /= sl[-1]
        ang_b = np.unwrap(np.arctan2(Bh[:, 1] - cz, Bh[:, 0] - cx))
        ang_l = np.unwrap(np.arctan2(L[:, 1] - cz, (L[:, 0] - cx) * 0.5))
        fa = (ang_b - ang_b[0]) / (ang_b[-1] - ang_b[0])
        fl = (ang_l - ang_l[0]) / (ang_l[-1] - ang_l[0])
        f = 0.5 * sb + 0.5 * np.interp(fa, fl, sl)
        return np.stack([np.interp(f, sl, L[:, 0]), np.interp(f, sl, L[:, 1])], 1)
    Mu = by_border(np.stack([cx + up[:, 0], up[:, 1]], 1), B[:nh + 1])
    Ml = by_border(np.stack([cx + lo[:, 0], lo[:, 1]], 1), np.concatenate([B[nh:], B[:1]]))
    M = np.concatenate([Mu[:-1], Ml[:-1]])                     # [nb, 2], corners at 0 and nh
    # (each corner rounded over a few spokes, about a millimetre across: a corner in one vertex is a fold where
    # the two lids' margins meet face to face)
    tt_ = np.arange(nb)
    frac_ = np.minimum(tt_ % nh, nh - tt_ % nh)
    wr = np.exp(-(frac_ / 4.0) ** 2)[:, None]
    for _ in range(40):
        M = M + 0.5 * wr * (0.5 * (np.roll(M, 1, 0) + np.roll(M, -1, 0)) - M)
    upper = np.zeros(nb, bool)
    upper[1:nh] = True
    # ---- the skin, from the lash line out to the border: rings eased from the lid's shape to the block's
    rho = (np.arange(out_rings + 1) / out_rings)
    f = rho ** 1.35
    rings_xz = [M + (B - M) * fk for fk in f]
    box = (B[:, 0].min(), B[:, 0].max(), B[:, 1].min(), B[:, 1].max())
    field = skin_field(S, lambda x, z: head_y(side * x, z), box, M, nh)
    pts = []
    for k, (rk, Pk) in enumerate(zip(rho, rings_xz)):
        if k == out_rings:
            break                                               # (the border: the grid's own points)
        x, z = Pk[:, 0], Pk[:, 1]
        y = field(x, z)
        # (into the head's exact surface over the last rings, which the field only samples)
        wb = smooth(0.94, 1.0, rk)
        if wb.any():
            y = y + (np.array([head_y(side * a_, b_) for a_, b_ in zip(x, z)]) - y) * wb
        pts.append(np.stack([side * x, y, z], 1))
    skin = pts[::-1]                                            # (outer to inner: ring 0 is next to the border)
    # ---- the margin: the lash line (ring 0 of the skin above), a rounded margin, the wet line on the ball
    bx, bz = M[:, 0], M[:, 1]
    b0 = bed(S, bx, bz)
    nrm0 = np.minimum(1.5, np.sqrt(1 + bed_slope(S, bx, bz) ** 2))
    # the direction into the opening, in the front view (from the margin toward the corner line)
    tang = np.roll(M, -1, axis=0) - np.roll(M, 1, axis=0)
    inward = np.stack([-tang[:, 1], tang[:, 0]], 1)
    inward /= np.maximum(1e-9, np.linalg.norm(inward, axis=1))[:, None]
    toward = np.stack([cx - bx, S['centre'][2] - bz], 1)
    inward *= np.sign(np.sum(inward * toward, axis=1))[:, None]
    # (toward the corners the direction turns to point at the opening's centre: there the two lids' margins
    # face each other across the canthus, and pushed straight in from each lid they crossed and folded; pointed
    # at one centre they cannot cross, and the margin rounds the canthus in a small arc)
    tt = np.arange(nb)
    frac = np.minimum(tt % nh, nh - tt % nh) / nh
    cen = np.array([cx + S['xc'], S['centre'][2] - S['lift'] + S['tilt'] * S['xc']])
    tc = cen[None, :] - M
    tc /= np.maximum(1e-9, np.linalg.norm(tc, axis=1))[:, None]
    wc = (1 - smooth(0.0, 0.25, frac))[:, None]
    inward = inward * (1 - wc) + tc * wc
    inward /= np.maximum(1e-9, np.linalg.norm(inward, axis=1))[:, None]
    mt = S['margin_t']
    # the margin's cross-section, from the lash line (the skin's last ring, A) round to the wet line on the ball
    # (W), as (depth from W toward A in margin thicknesses, push toward the opening): a rounded edge at the lash
    # line, the margin's face bowed slightly into the opening, a softer edge at the wet line. No step between
    # rings turns by more than about 50 degrees (the lash line's edge, the sharpest, is a right angle in life)
    A_y = field(bx, bz)
    W_y = b0 - S['film']
    taper = 0.5 + 0.5 * smooth(0.0, 0.18, frac)
    PROF = ((0.88, 0.10), (0.65, 0.16), (0.45, 0.17), (0.25, 0.14), (0.08, 0.07))
    rim = []
    for dn, push in PROF:
        xx, zz = bx + push * mt * taper * inward[:, 0], bz + push * mt * taper * inward[:, 1]
        yy = W_y + (A_y - W_y) * dn
        yy = np.minimum(yy, bed(S, xx, zz) - 2 * S['film'])
        rim.append(np.stack([side * xx, yy, zz], 1))
    wet = np.stack([side * bx, b0 - S['film'], bz], 1)
    # ---- the conjunctiva: from the wet line back round the ball, under the lids, to behind its equator
    C = np.array([side * cx, cy, cz])
    v = wet - C
    rad0 = np.linalg.norm(v, axis=1)
    th0 = np.arccos(np.clip(-v[:, 1] / rad0, -1, 1))
    ph0 = np.arctan2(v[:, 2], side * v[:, 0])
    # the cap's angles round the back pole come from the block's own (i, j) (its square mapped to a disc), so
    # the cap's grid fills it exactly; the last rings ease the spokes' angles round to them
    ii = np.array([i for i, j in cyc], float)
    jj = np.array([j for i, j in cyc], float)
    if side < 0:
        ii = i0 + i1 - ii
    ua = 2 * (ii - i0) / (i1 - i0) - 1
    va = 2 * (jj - j0) / (j1 - j0) - 1
    phc = np.arctan2(va * np.sqrt(1 - ua * ua / 2), ua * np.sqrt(1 - va * va / 2))
    phc = ph0 + np.angle(np.exp(1j * (phc - ph0)))
    TH_END = math.radians(95)
    sac = []
    for k in range(1, in_rings + 1):
        t = k / in_rings
        th = th0 + (TH_END - th0) * smooth(0, 1, t) ** 0.8
        ph = ph0 + (phc - ph0) * smooth(0.35, 1.0, t)
        rr = rad0 + (S['r'] + 3 * S['film'] - rad0) * smooth(0, 1, t)
        rr = np.maximum(rr, globe_radius(S, th) + S['film'])
        d = np.stack([np.sin(th) * np.cos(ph), -np.cos(th), np.sin(th) * np.sin(ph)], 1)
        d[:, 0] *= side
        sac.append(C + rr[:, None] * d)
    # ---- the cap behind the ball: the block's interior grid, its square mapped onto the back of the sphere
    ni, nj = i1 - i0, j1 - j0
    cap_ij = [(i, j) for j in range(j0 + 1, j1) for i in range(i0 + 1, i1)]
    cap = []
    for i, j in cap_ij:
        ua_, va_ = 2 * (i - i0) / ni - 1, 2 * (j - j0) / nj - 1
        du, dv = ua_ * math.sqrt(1 - va_ * va_ / 2), va_ * math.sqrt(1 - ua_ * ua_ / 2)
        rr_ = math.hypot(du, dv)
        th = math.pi - (math.pi - TH_END) * rr_
        ph = math.atan2(dv, du)
        d = np.array([side * math.sin(th) * math.cos(ph), -math.cos(th), math.sin(th) * math.sin(ph)])
        cap.append(C + (S['r'] + 3 * S['film']) * d)
    cap = np.array(cap)
    # ---- the points and the quads
    rings = skin + rim + [wet] + sac                           # each [nb, 3], outer to inner
    P = np.concatenate(rings + [cap])
    # indices: -1 - t for the border's point t; else into P
    ring_idx = [-1 - np.arange(nb)] + [np.arange(nb) + k * nb for k in range(len(rings))]
    quads = []
    for a_, b_ in zip(ring_idx[:-1], ring_idx[1:]):
        for t in range(nb):
            t2 = (t + 1) % nb
            quads.append((a_[t], a_[t2], b_[t2], b_[t]))
    # the cap: its border is the last ring, by the block's (i, j)
    last = ring_idx[-1]
    base = len(rings) * nb
    where = {ij: base + k for k, ij in enumerate(cap_ij)}
    for t, ij in enumerate(cyc if side > 0 else [(i0 + i1 - i, j) for i, j in cyc]):
        where[ij] = last[t]
    for j in range(j0, j1):
        for i in range(i0, i1):
            quads.append((where[(i, j)], where[(i + 1, j)], where[(i + 1, j + 1)], where[(i, j + 1)]))
    nsk = len(skin)
    info = {'lash': np.arange(nb) + (nsk - 1) * nb, 'wet': np.arange(nb) + (nsk + len(rim)) * nb,
            'rings': len(rings), 'nb': nb, 'cap_corners': [where[(i0, j0)], where[(i1, j0)], where[(i0, j1)], where[(i1, j1)]]}
    return P, cyc, np.array(quads, np.int64), info


# ------------------------------------------------------------------------------------------------ 4. the eyeball

def globe(S, side=1, seg=96, rings=64):
    """The eyeball: a sphere on the eye's axis (its pole at the cornea's apex, so its rings run round the limbus),
    shaped by globe_radius: the sclera, the limbal step and the corneal bulge. Returns (points [n, 3], quads and
    the two pole fans as triangles, uv: (radius from the axis in limbus radii / 4, angle) so a painting of the
    iris centres on the cornea, and the iris and pupil radii in that uv)."""
    cx, cy, cz = S['centre']
    C = np.array([side * cx, cy, cz])
    th = np.linspace(0, math.pi, rings + 1)[1:-1]
    # (rings closer over the cornea and the limbus, where the shape and the painting change)
    th = math.pi * (th / math.pi) ** 1.35
    ph = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    R = globe_radius(S, th)
    T, Ph = np.meshgrid(th, ph, indexing='ij')
    RR = R[:, None]
    P = np.stack([side * RR * np.sin(T) * np.cos(Ph), -RR * np.cos(T), RR * np.sin(T) * np.sin(Ph)], -1).reshape(-1, 3)
    front = np.array([[0.0, -globe_radius(S, np.array([0.0]))[0], 0.0]])
    back = np.array([[0.0, S['r'], 0.0]])
    P = np.concatenate([front, P, back]) + C
    n = len(th)
    idx = lambda r, k: 1 + r * seg + (k % seg)
    F = []
    for k in range(seg):
        F.append((0, idx(0, k + 1), idx(0, k)) if side > 0 else (0, idx(0, k), idx(0, k + 1)))
    for r in range(n - 1):
        for k in range(seg):
            q = (idx(r, k), idx(r, k + 1), idx(r + 1, k + 1), idx(r + 1, k))
            F.append(q if side > 0 else q[::-1])
    last = len(P) - 1
    for k in range(seg):
        F.append((last, idx(n - 1, k), idx(n - 1, k + 1)) if side > 0 else (last, idx(n - 1, k + 1), idx(n - 1, k)))
    F = facing(P, F, C)
    rad = np.concatenate([[0.0], (RR * np.sin(T)).ravel() * np.where(T.ravel() < math.pi / 2, 1, -1), [0.0]])
    uv = np.stack([rad, np.concatenate([[0.0], Ph.ravel(), [0.0]])], 1)
    return P, F, uv


def iris_disc(S, side=1, seg=96, rings=12):
    """The iris: a shallow cone just behind the limbus's plane (it bows slightly forward at the pupil, as the
    iris rests on the lens), radius the limbus's, with the pupil as its inner ring. Points, quads, and each
    point's radius in iris radii (0 the pupil's centre .. 1 the iris's root), for the iris painting."""
    cx, cy, cz = S['centre']
    C = np.array([side * cx, cy, cz])
    depth = S['limbus_z'] - S['iris_back']
    rr = np.linspace(0.0, 1.0, rings + 1)[1:]
    ph = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    Rr, Ph = np.meshgrid(rr, ph, indexing='ij')
    bow = 0.06 * S['iris_r'] * (1 - Rr ** 2)
    P = np.stack([side * S['iris_r'] * Rr * np.cos(Ph), -(depth + bow), S['iris_r'] * Rr * np.sin(Ph)], -1).reshape(-1, 3)
    P = np.concatenate([[[0, -(depth + 0.06 * S['iris_r']), 0]], P]) + C
    idx = lambda r, k: 1 + r * seg + (k % seg)
    F = [(0, idx(0, k), idx(0, k + 1)) if side < 0 else (0, idx(0, k + 1), idx(0, k)) for k in range(seg)]
    for r in range(rings - 1):
        for k in range(seg):
            q = (idx(r, k), idx(r, k + 1), idx(r + 1, k + 1), idx(r + 1, k))
            F.append(q if side > 0 else q[::-1])
    F = facing(P, F, C)
    rad = np.concatenate([[0.0], Rr.ravel()])
    return P, F, rad


# ------------------------------------------------------------------------------------------------ 5. the caruncle

def caruncle(S, side=1, n=14, m=8):
    """The caruncle in the inner corner, with the plica's fold on its outer side: a small soft mound filling the
    angle between the lids, its edges tucked under their wet lines and its outer edge laid on the ball. And the
    tear meniscus: a thin bead along the lower lid's wet line. Points and quads."""
    cx, cy, cz = S['centre']
    w = S['caruncle_w']
    # across the corner: from the inner corner (u 0) out to the plica (u 1); round it: from the upper lid (v 0)
    # to the lower (v 1)
    P = []
    for i in range(n + 1):
        u = i / n
        for j in range(m + 1):
            v = j / m
            # (its outer edge, the plica, bows out toward the cornea in a crescent and lies flat on the ball:
            # a straight edge standing off the ball read as a cut-out wedge)
            xl = -S['med'] + w * u * (0.8 + 0.4 * math.sin(math.pi * v) ** 0.8)
            zt = float(margin_point(S, xl, True)[1])
            zb = float(margin_point(S, xl, False)[1])
            z = zt + (zb - zt) * v
            # tucked under the lids at its top and bottom (v 0, 1), full a third of the way out, nothing at the plica
            swell = math.sin(math.pi * v) * math.sin(math.pi * min(1.0, 0.2 + 0.8 * u)) ** 0.8 * (1 - u ** 3)
            # (on the lids' own soft bed and never forward of their wet lines' margin: on the hard bed it stood
            # out of the corner in front of the skin when seen from the nose's side)
            y = float(bed(S, cx + xl, z, soft=True)) - S['film'] - min(0.3 * w, 0.6 * S['margin_t']) * swell
            P.append((side * (cx + xl), y, z))
    P = np.array(P)
    F = []
    for i in range(n):
        for j in range(m):
            a = i * (m + 1) + j
            F.append((a, a + m + 1, a + m + 2, a + 1))
    return P, facing(P, F, (side * cx, cy, cz))


def meniscus(S, side=1, n=60):
    """The tear meniscus: a thin wet bead along the lower lid's wet line, from the caruncle to the outer corner."""
    cx, cy, cz = S['centre']
    xs = np.linspace(-S['med'] + S['caruncle_w'] * 0.8, S['lat'] * 0.97, n)
    _, z = margin_point(S, xs, False)
    r = 0.25 * S['e']
    P, F = [], []
    for k in range(n):
        x, zz = cx + xs[k], float(z[k])
        b = float(bed(S, x, zz))
        for j, (dz, dy) in enumerate(((r * 0.2, -S['film']), (r, -0.6 * r), (r * 1.6, -S['film']))):
            P.append((side * x, b + dy, zz + dz))
    for k in range(n - 1):
        for j in range(2):
            a = k * 3 + j
            F.append((a, a + 3, a + 4, a + 1))
    P = np.array(P)
    return P, facing(P, F, (side * cx, cy, cz))
