"""The riding habit's shared numbers: what the corset does to the torso, as every garment over it must read it.

A period (1885-95) long-line corset is laced to a waist well under the body's own. The body does not stay the shape
it was: the soft trunk is drawn in at the waist, and what is drawn in goes up under the bust and down over the
abdomen and hips. The corset, the petticoat band, the fitted bodice and the skirt all sit on that corseted body,
so they read it from here and never place an offset by hand:

    M = habit.measure(body)                 # the body's own sections, measured from its mesh (rest shape)
    habit.corseted_section(M, z, theta)     # the corseted torso's radius from the body's axis, at height z, angle theta
    habit.compress_body(body, M)            # move the skin to that shape (no clipping: the flesh is drawn in)
    habit.build_corset(body, M)             # the corset itself, on the corseted body (corset_garment.py)

Everything is a share of H (the head's height, chin to crown, which the body is built from) or of the body's own
measured sections, so a body with a different waist, hips or head gives a different corset and a different
compression, with no number changed.

Conventions: lengths in scene units (H is about 9.4 of them); theta is the figure's (female.torso_point's): 0 at the
front, growing toward her right (-x), pi at the back; r is measured from the body's axis (x = 0, y = M.cy).
"""
import math

import numpy as np

# ---------------------------------------------------------------- the numbers other garments share
# The waist is drawn in: corseted girth over natural girth, at the waist. 0.88 is about 8 cm off a woman of 165 cm
# (period tight-lacing took 5 to 10 cm; the reference habit's bodice is at the tight end).
WAIST_REDUCTION = 0.88
# How much of that the depth takes against the width: the trunk is drawn in far more from the sides than front to
# back (the spine and the breastbone hold their place). 0.45 of the width's reduction is the depth's.
DEPTH_SHARE = 0.45
# The corset's thickness (coutil, two layers, and the bone channels), in H.
CORSET_THICK = 0.012
# Where the corset ends, in H from the natural waist line (down is negative). The lower edge is longest at the busk
# (it ends on the belly in a point) and shortest at the back, where it clears the seat; at the sides it ends
# CORSET_HIP_DROP below the waist, over the hip bones. The upper edge is lowest at the front centre (a dip between
# the breasts), runs up over the bust, and is highest at the back.
CORSET_HIP_DROP = 0.42
CORSET_BOTTOM_FRONT = 0.60
CORSET_BOTTOM_BACK = 0.34
CORSET_TOP_SIDE = 0.585
CORSET_TOP_FRONT_SWING = -0.045        # (the top edge rises toward the back by this much per cos(theta))
CORSET_TOP_CARRY = 1.0                 # (how much of the bust's lift the top edge follows)
CORSET_TOP_DIP = 0.10                  # the dip between the breasts
# The back opening, laced: its width at the waist, in H (parallel edges; a hand's breadth is tight, and the laces
# bridge it).
CORSET_GAP = 0.09
# The compression's reach: how far above and below the waist the trunk is drawn in, in H, easing to nothing
# (the lower ribs and the soft belly give; the bust's ribs and the pelvis hardly do).
COMPRESS_UP = 0.62
COMPRESS_DOWN = 0.50
# The tissue drawn in at the waist goes somewhere: a share of it up under the bust, the rest down over the belly and
# hips, as soft swellings centred at these heights (H from the waist) and this wide (sigma, in H).
VOLUME_UP = 0.42
BULGE_UP = (0.50, 0.28)
BULGE_DOWN = (-0.38, 0.28)
# Flesh is compressible (fat, the lower ribs flexing, organs shifting): only this share of the displaced volume has
# to reappear as swelling; the rest is the trunk's own compression.
VOLUME_KEPT = 0.40

# The corset lifts the bust: the breast tissue is carried up and forward by the cups and the busk, so the apex stands
# higher over a firmer shelf (a share of H, up) and a little fuller (the radius, a share), the lift easing in above
# the waist's reach and out again before the collarbones.
BUST_LIFT = 0.10
BUST_PUSH = 0.03
LIFT_FROM, LIFT_FULL = 0.20, 0.45       # (H above the waist: where the lift begins and is whole)
LIFT_OUT, LIFT_GONE = 0.48, 0.80        # (and where it begins to fade and is gone)

# the grid the body is measured on
Z_STEP = 0.25
N_THETA = 180


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- measuring
class Measure:
    """The body's sections, as measured: R[i, j] the radius of the skin at height zs[i], angle th[j], from the axis."""

    def __init__(self, H, cy, zs, R, z_waist):
        self.H, self.cy, self.zs, self.R = H, cy, zs, R
        self.th = np.arange(N_THETA) * (2 * math.pi / N_THETA)
        self.z_waist = z_waist
        self._cr = None
        self._key = None

    # --- natural body
    def natural_section(self, z, theta):
        return _bilerp(self.zs, self.R, z, theta)

    def girth(self, R_row):
        return _girth(R_row, self.th)

    def girth_at(self, z):
        return self.girth(_row(self.zs, self.R, z))

    @property
    def natural_waist(self):
        return self.girth_at(self.z_waist)

    # --- corseted body
    def _params(self):
        return (WAIST_REDUCTION, DEPTH_SHARE, COMPRESS_UP, COMPRESS_DOWN, VOLUME_UP, BULGE_UP, BULGE_DOWN, VOLUME_KEPT)

    def table(self):
        """R after the corset: same grid as R."""
        if self._cr is None or self._key != self._params():
            self._build()
        return self._cr

    def _build(self):
        H = self.H
        zw = self.z_waist
        a = compress_profile(self, self.zs)
        g0 = self.girth_at(zw)
        up = np.exp(-((self.zs - (zw + BULGE_UP[0] * H)) / (BULGE_UP[1] * H)) ** 2)
        dn = np.exp(-((self.zs - (zw + BULGE_DOWN[0] * H)) / (BULGE_DOWN[1] * H)) ** 2)
        shape = VOLUME_UP * up + (1.0 - VOLUME_UP) * dn
        A0 = 0.5 * (self.R ** 2).sum(1)
        bulge = 0.0
        for _ in range(4):
            # the width factor at the waist that leaves the corseted girth (after the swelling) at the laced size
            sw = 1.0 + bulge * float(np.interp(zw, self.zs, shape))
            lo, hi = 0.3, 1.0
            for _ in range(40):
                kx = 0.5 * (lo + hi)
                ky = 1.0 - (1.0 - kx) * DEPTH_SHARE
                g = sw * self.girth(_squeeze_row(self, _row(self.zs, self.R, zw), kx, ky))
                if g > WAIST_REDUCTION * g0:
                    hi = kx
                else:
                    lo = kx
            self.kx_waist = 0.5 * (lo + hi)
            k = 1.0 - (1.0 - self.kx_waist) * a
            kyv = 1.0 - (1.0 - k) * DEPTH_SHARE
            rows = np.array([_squeeze_row(self, self.R[i], k[i], kyv[i]) for i in range(len(self.zs))])
            # the displaced volume that has to reappear (a share of it: flesh is compressible) is given back as
            # a swelling, the section scaled by 1 + G * shape
            dV = VOLUME_KEPT * float(np.sum(A0 - 0.5 * (rows ** 2).sum(1)))
            lo, hi = 0.0, 0.5
            for _ in range(40):
                G = 0.5 * (lo + hi)
                s_ = 1.0 + G * shape
                gain = float(np.sum(0.5 * (rows ** 2).sum(1) * (s_ * s_ - 1.0)))
                if gain < dV:
                    lo = G
                else:
                    hi = G
            bulge = 0.5 * (lo + hi)
        self.bulge = bulge
        rows = rows * (1.0 + bulge * shape)[:, None]
        # the bust lift: the surface at height z' is carried up to z' + L(z', theta) and swells; the table is read
        # back at each height, so it is the inverse of that carry
        Lz, push = lift_field(self, self.zs[:, None], self.th[None, :])
        lifted = np.empty_like(rows)
        for j in range(rows.shape[1]):
            zp = np.interp(self.zs, self.zs + Lz[:, j], self.zs)
            lifted[:, j] = np.interp(zp, self.zs, rows[:, j] * push[:, j])
        rows = lifted
        self._cr = rows
        self._key = self._params()
        self.shape_k = k
        self.volume_change = float(np.sum(0.5 * (rows ** 2).sum(1)) - np.sum(A0)) / float(np.sum(A0))
        self.girth_waist_corseted = self.girth(_row(self.zs, rows, zw))

    def corseted_section(self, z, theta):
        return _bilerp(self.zs, self.table(), z, theta)

    def corseted_girth_at(self, z):
        return self.girth(_row(self.zs, self.table(), z))

    # --- the corset's edges
    def top(self, theta):
        """Height of the corset's upper edge at angle theta."""
        c = np.cos(theta)
        dip = CORSET_TOP_DIP * np.exp(-(_wrap(theta) / 0.28) ** 2)
        # (the cups rise with the bust they carry)
        carry = BUST_LIFT * CORSET_TOP_CARRY * np.maximum(c, 0.0) ** 1.5
        return self.z_waist + self.H * (CORSET_TOP_SIDE + CORSET_TOP_FRONT_SWING * c - dip + carry)

    def bottom(self, theta):
        """Height of the corset's lower edge at angle theta."""
        f = (1.0 + np.cos(theta)) / 2.0
        side = (CORSET_BOTTOM_FRONT - CORSET_HIP_DROP) / (CORSET_BOTTOM_FRONT - CORSET_BOTTOM_BACK)
        p = math.log(side) / math.log(0.5)
        return self.z_waist - self.H * (CORSET_BOTTOM_BACK + (CORSET_BOTTOM_FRONT - CORSET_BOTTOM_BACK) * f ** p)

    def report(self):
        t = self.table()
        out = {
            'H': self.H, 'z_waist': self.z_waist,
            'waist_natural': self.natural_waist, 'waist_corseted': self.girth_waist_corseted,
            'waist_ratio': self.girth_waist_corseted / self.natural_waist,
            'bust_natural': self.girth_at(self.z_waist + 0.5 * self.H),
            'bust_corseted': self.corseted_girth_at(self.z_waist + 0.5 * self.H),
            'hip_natural': max(self.girth_at(z) for z in np.arange(self.z_waist - 1.0 * self.H, self.z_waist - 0.5 * self.H, 0.25)),
            'length_side': self.top(math.pi / 2) - self.bottom(math.pi / 2),
            'length_front': self.top(0.0) - self.bottom(0.0),
            'length_back': self.top(math.pi) - self.bottom(math.pi),
            'kx_waist': self.kx_waist, 'bulge': self.bulge, 'volume_change': self.volume_change,
        }
        out['hip_corseted'] = max(self.corseted_girth_at(z) for z in np.arange(self.z_waist - 1.0 * self.H, self.z_waist - 0.5 * self.H, 0.25))
        return out


def _wrap(t):
    return (np.asarray(t) + math.pi) % (2 * math.pi) - math.pi


def _row(zs, R, z):
    i = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
    t = np.clip((z - zs[i]) / (zs[i + 1] - zs[i]), 0.0, 1.0)
    if np.ndim(t):
        t = t[..., None]
    return R[i] * (1 - t) + R[i + 1] * t


def _bilerp(zs, R, z, theta):
    """R (rows at zs, columns at equal angles round the circle) at height z and angle theta (arrays broadcast)."""
    n = R.shape[1]
    z, theta = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(theta, dtype=float))
    i = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
    t = np.clip((z - zs[i]) / (zs[i + 1] - zs[i]), 0.0, 1.0)
    u = (theta % (2 * math.pi)) / (2 * math.pi) * n
    j = np.floor(u).astype(int)
    f = u - j
    j0, j1 = j % n, (j + 1) % n
    lo = R[i, j0] * (1 - f) + R[i, j1] * f
    hi = R[i + 1, j0] * (1 - f) + R[i + 1, j1] * f
    return lo * (1 - t) + hi * t


def _points(row, th):
    # (x, y) from the axis: x = -r sin(theta), y = -r cos(theta)
    return -row * np.sin(th), -row * np.cos(th)


def _girth(row, th):
    x, y = _points(row, th)
    return float(np.sum(np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)))


def _squeeze_row(M, row, kx, ky):
    """The section `row` with its width scaled by kx and its depth by ky about the axis, resampled on theta."""
    x, y = _points(row, M.th)
    x, y = x * kx, y * ky
    th2 = np.arctan2(-x, -y) % (2 * math.pi)
    r2 = np.hypot(x, y)
    order = np.argsort(th2)
    t, r = th2[order], r2[order]
    return np.interp(M.th, np.concatenate([t - 2 * math.pi, t, t + 2 * math.pi]), np.concatenate([r, r, r]))


def lift_field(M, z, theta):
    """(how far a point of the natural trunk at height z and (squeezed) angle theta is carried up, how much its radius
    swells) for the corset's bust lift: strongest on the front, nothing on the back or at the sides."""
    H, zw = M.H, M.z_waist
    z = np.asarray(z, dtype=float)
    w = np.maximum(np.cos(theta), 0.0) ** 1.5
    g = _ss(zw + LIFT_FROM * H, zw + LIFT_FULL * H, z) * (1.0 - _ss(zw + LIFT_OUT * H, zw + LIFT_GONE * H, z))
    return BUST_LIFT * H * g * w, 1.0 + BUST_PUSH * g * w


def compress_profile(M, z):
    """How much of the full squeeze applies at height z (1 at the waist, 0 above and below its reach)."""
    z = np.asarray(z, dtype=float)
    u = np.where(z >= M.z_waist, (z - M.z_waist) / (COMPRESS_UP * M.H), (M.z_waist - z) / (COMPRESS_DOWN * M.H))
    return 1.0 - _ss(0.0, 1.0, np.abs(u))


# ---------------------------------------------------------------- the public functions
def measure(body, head_h=None, z_waist=None, z_lo=None, z_hi=47.0):
    """Measure a body mesh object (the rest shape, no pose): ray-cast its skin outward from the trunk's axis at each
    height and angle. The axis is x = 0 and the front/back middle of the natural waist; H is the head's height (read
    from anatomy.HEAD when not given)."""
    import bpy
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    if head_h is None:
        import anatomy
        head_h = anatomy.HEAD
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.transform(body.matrix_world)
    # the trunk alone: in the rest shape the arms hang through the hips (the poses swing them out), so every vertex
    # the arm bones carry is left out, and a ray from the axis meets the trunk's own skin
    bm.verts.ensure_lookup_table()
    w_arm = arm_weights(body)
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(w_arm >= 0.5)[0]], context='VERTS')
    tree = BVHTree.FromBMesh(bm)
    # the axis: the middle, front to back, of the trunk where it is narrowest between the ribs and the hips
    mids = []
    for z in np.arange(36.0, 42.0, 0.25):
        ys = []
        for sgn in (1.0, -1.0):
            hit = tree.ray_cast(Vector((0.0, 0.0, z)), Vector((0.0, sgn, 0.0)), 20.0)
            ys.append(hit[0].y if hit[0] is not None else None)
        if None not in ys:
            mids.append(0.5 * (ys[0] + ys[1]))
    cy = float(np.median(mids))
    zw = 39.5 if z_waist is None else z_waist       # (refined below from the girths when not given)
    z_lo = zw - 1.05 * head_h if z_lo is None else z_lo
    zs = np.arange(z_lo, z_hi + 1e-6, Z_STEP)
    th = np.arange(N_THETA) * (2 * math.pi / N_THETA)
    R = np.zeros((len(zs), N_THETA))
    for i, z in enumerate(zs):
        for j, t in enumerate(th):
            d = Vector((-math.sin(t), -math.cos(t), 0.0))
            hit = tree.ray_cast(Vector((0.0, cy, z)), d, 30.0)
            R[i, j] = hit[0].xy.__sub__(Vector((0.0, cy))).length if hit[0] is not None else (R[i - 1, j] if i else 4.0)
    bm.free()
    # (a light smoothing along z and theta, so the mesh's own jitter does not reach the garments)
    R = _smooth(R)
    if z_waist is None:
        g = np.array([_girth(R[i], th) for i in range(len(zs))])
        # the narrowest girth between the lower ribs and the hips
        sel = (zs > 35.0) & (zs < 42.5)
        zw = float(zs[sel][np.argmin(g[sel])])
    return Measure(float(head_h), float(cy), zs, R, zw)


def _smooth(R):
    k = np.array([0.25, 0.5, 0.25])
    out = R.copy()
    out[1:-1] = k[0] * R[:-2] + k[1] * R[1:-1] + k[2] * R[2:]
    out = k[0] * np.roll(out, 1, axis=1) + k[1] * out + k[2] * np.roll(out, -1, axis=1)
    return out


def natural_section(M, z, theta):
    return M.natural_section(z, theta)


def corseted_section(M, z, theta):
    """The torso's radius from its axis after the corset has drawn it in, at height z and angle theta. The corset's
    own outer surface is this plus CORSET_THICK * H; a garment over the corset adds its own ease to that."""
    return M.corseted_section(z, theta)


def corset_thick(M):
    return CORSET_THICK * M.H


def arm_weights(body):
    """How much of each vertex of the body mesh is carried by the arm bones (0 to 1)."""
    me = body.data
    w = np.zeros(len(me.vertices))
    groups = {g.index for g in body.vertex_groups if g.name.startswith(('UpperArm', 'Forearm', 'Hand'))}
    for i, v in enumerate(me.vertices):
        for g in v.groups:
            if g.group in groups:
                w[i] += g.weight
    return np.clip(w, 0, 1)


def compress_body(body, M, blend_by_arm=True):
    """Draw the skin of the body object in to the corseted shape: every vertex of the trunk is moved by the same
    squeeze and swelling corseted_section is made from (width and depth scaled about the axis by the waist's profile,
    then the displaced volume given back above and below), so the flesh is compressed, not clipped through. Points on
    the arms keep their place, blended by the arm bones' weights. The 'rest' pattern coordinates move with them.
    Returns the largest move."""
    me = body.data
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    w_arm = arm_weights(body) if blend_by_arm else np.zeros(n)
    M.table()
    z = co[:, 2]
    a = compress_profile(M, z)
    k = 1.0 - (1.0 - M.kx_waist) * a
    ky = 1.0 - (1.0 - k) * DEPTH_SHARE
    H, zw = M.H, M.z_waist
    up = np.exp(-((z - (zw + BULGE_UP[0] * H)) / (BULGE_UP[1] * H)) ** 2)
    dn = np.exp(-((z - (zw + BULGE_DOWN[0] * H)) / (BULGE_DOWN[1] * H)) ** 2)
    s = 1.0 + M.bulge * (VOLUME_UP * up + (1.0 - VOLUME_UP) * dn)
    new = co.copy()
    new[:, 0] = co[:, 0] * k * s
    new[:, 1] = M.cy + (co[:, 1] - M.cy) * ky * s
    # (the squeeze reaches the lower ribs and the pelvis only; the shoulders and the legs, outside the measured
    # range, are left exactly where they are)
    inside = (z > M.zs[0]) & (z < M.zs[-1])
    f = (1.0 - w_arm) * inside
    # the bust lift, by each point's own angle after the squeeze
    th = np.arctan2(-new[:, 0], -(new[:, 1] - M.cy)) % (2 * math.pi)
    Lz, push = lift_field(M, z, th)
    new[:, 2] = z + Lz
    new[:, 0] *= push
    new[:, 1] = M.cy + (new[:, 1] - M.cy) * push
    new = co + (new - co) * f[:, None]
    mv = np.abs(new - co).max()
    me.vertices.foreach_set('co', new.ravel())
    if 'rest' in me.attributes:
        rest = np.empty(n * 3)
        me.attributes['rest'].data.foreach_get('vector', rest)
        rest = rest.reshape(-1, 3)
        rest += (new - co)
        me.attributes['rest'].data.foreach_set('vector', rest.ravel())
    me.update()
    return float(mv)


def build_corset(body, M, rig=None, material=None):
    """Make the corset on the corseted body. Returns the objects built (shell, busk, eyelets, laces)."""
    import corset_garment
    return corset_garment.build(body, M, rig=rig)
