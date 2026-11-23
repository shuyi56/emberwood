"""The corset: a period (1885-95) long-line corset in coutil, drafted from the corseted body (habit.py).

Nothing here is placed by hand. Every height is a share of H from the waist line, every width a share of the body's
own girth at that height, and the surface is the corseted body's (habit.corseted_section) plus the corset's
thickness, bridged across the hollows cloth cannot follow (between the breasts, down the spine).

The pattern (a 12-panel long-line corset, the busk and the lacing at opposite ends of one strip):
  * the busk at the front centre: two steel halves, stud heads in a row down the closing, standing a little proud
  * five seams each side, each between two boning channels (flat steel, in raised channels that stop short of the
    edges), so the panels read as a boned shell
  * a bust gusset (a wedge let in from the top edge) and a hip gusset (from the lower edge), each two seams that
    meet in a point, as real gores do
  * the back opening, laced: a stiffened edge each side with a row of eyelets, a cord criss-crossing the opening
    from the top to the bottom, and its two tails left hanging from the waist
  * a bias binding rolled round the top and lower edges

The seams run at fixed shares of the half-girth, so they spread with the bust and the hips and close in at the waist,
as the pattern pieces of a real corset do. The garment is one shell (coutil, with the binding as a second material)
thickened outward by CORSET_THICK, skinned to the body's trunk bones."""
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector

import female
import habit

M_COUTIL = female.mat_def(90, 'corset coutil', female.ramp(0xd8ccb4, 0.10), 0.0, closeup='cloth', form=True)
M_BINDING = female.mat_def(91, 'corset binding', female.ramp(0xb7a88d, 0.10), 0.0, closeup='cloth', form=True)
M_STEEL = female.mat_def(92, 'corset steel', [[40, 42, 46], [78, 82, 88], [116, 120, 126], [164, 168, 174], [214, 218, 224]], 0.0)
M_CORD = female.mat_def(93, 'corset lace', female.ramp(0xe6dcc6, 0.10), 0.0, closeup='cloth')

# ---- the draft, as shares (of H, of the half-girth, of the corset's thickness)
SEAMS = (0.16, 0.34, 0.52, 0.70, 0.86)           # fractions of the half-girth from the front centre (0) to the back edge (1)
BONE_OFFSET = 0.17                               # a boning channel's distance from its seam, in units (about 4.5 mm)
BONE_W = 0.07                                    # and its half-width
BONE_H = 0.028                                   # and its height
SEAM_W = 0.040                                   # a seam's groove: half-width and depth
SEAM_D = 0.034
CLEAR = 0.04                                      # the corset's inner face off the skin, in units (1 mm)
BONE_STOP = 0.55                                 # the channels stop this far from the edges
BINDING_W = 0.26                                 # the bias binding along the top and lower edges, in units
BEAD_H = 0.030                                   # the binding's roll
GUSSET_BUST = (0.34, 0.30, 0.050)                # (seam fraction, how far down from the top it reaches in H, half width at the edge as a fraction)
GUSSET_HIP = (0.52, 0.22, 0.050)                 # (seam fraction, how far up from the lower edge in H, half width at the edge)
BUSK_W = 0.42                                    # half the busk's width, in units
BUSK_H = 0.034
STUD_R = 0.070
N_STUDS = 9
EYELET_PAIRS = 13
EYELET_R, EYELET_TUBE = 0.075, 0.024
EYELET_OFF = 0.26                                # eyelets this far in from the back edge
CORD_R = 0.022
N_S, N_V = 720, 150                              # the shell's grid: across (the whole strip) and up


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _hull_polar(x, y, th):
    """The convex hull of a section's points as a radius at each angle th (the taut cloth over the hollows)."""
    pts = sorted(set(zip(np.round(x, 5), np.round(y, 5))))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    poly = lo[:-1] + hi[:-1]
    r = np.zeros(len(th))
    dx, dy = -np.sin(th), -np.cos(th)
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        ex, ey = x1 - x0, y1 - y0
        den = dx * ey - dy * ex
        ok = np.abs(den) > 1e-9
        t = np.where(ok, (x0 * ey - y0 * ex) / np.where(ok, den, 1), 0)
        u = np.where(ok, (x0 * dy - y0 * dx) / np.where(ok, den, 1), -1)
        r = np.where((t > 0) & (u >= -1e-6) & (u <= 1 + 1e-6), np.maximum(r, t), r)
    return r


class Draft:
    """The corset's surface and its pattern lines on a measured body."""

    BRIDGE = 0.85          # how far the cloth is bridged across the hollows (1: the hull itself)

    def __init__(self, M):
        self.M = M
        self.H = M.H
        self.T = habit.CORSET_THICK * M.H
        self.gap = habit.CORSET_GAP * M.H
        zs = np.arange(M.zs[0], M.zs[-1] + 1e-6, 0.1)
        self.zs = zs
        th = M.th
        rows = []
        for z in zs:
            r = habit.corseted_section(M, z, th)
            x, y = -r * np.sin(th), -r * np.cos(th)
            h = _hull_polar(x, y, th)
            rows.append(r + self.BRIDGE * np.maximum(h - r, 0.0))
        self.surf = np.array(rows)                        # the body's corseted surface, bridged (no thickness)
        self.build_arcs()

    def r(self, z, theta):
        """The bridged, corseted surface at z, theta."""
        return habit._bilerp(self.zs, self.surf, z, theta)

    def build_arcs(self):
        """For each height, the signed arc length along the surface from the front centre (positive toward her
        right, theta growing) at the measuring angles, so a share of the half-girth can be turned into an angle."""
        M = self.M
        n = len(M.th)
        # angles in (-pi, pi]: index j runs theta = 0 ... 2pi; negative side is theta - 2pi
        ths = np.concatenate([M.th[n // 2 + 1:] - 2 * math.pi, M.th[:n // 2 + 1]])     # -pi+d ... pi
        self.ths = ths
        arcs = []
        for z in self.zs:
            r = self.r(z, ths % (2 * math.pi)) + self.T
            x, y = -r * np.sin(ths), -r * np.cos(ths)
            seg = np.hypot(np.diff(x), np.diff(y))
            a = np.concatenate([[0.0], np.cumsum(seg)])
            a -= np.interp(0.0, ths, a)
            arcs.append(a)
        self.arcs = np.array(arcs)

    def theta_of(self, s, z):
        """The angle at strip coordinate s in [-1, 1] (0 the front centre, +1 the back edge on her right) and height z."""
        z = np.clip(z, self.zs[0], self.zs[-1] - 1e-6)
        i = np.clip(np.floor((z - self.zs[0]) / 0.1).astype(int), 0, len(self.zs) - 2)
        t = (z - self.zs[i]) / 0.1
        s = np.asarray(s, dtype=float)
        s_b, i_b, t_b = np.broadcast_arrays(s, i, t)
        th_lo = np.empty(s_b.shape)
        th_hi = np.empty(s_b.shape)
        for k in np.unique(i_b):
            m = i_b == k
            for off, dst in ((0, th_lo), (1, th_hi)):
                a = self.arcs[k + off]
                aR, aL = a[-1] - self.gap / 2, -a[0] - self.gap / 2
                target = np.where(s_b[m] >= 0, s_b[m] * aR, s_b[m] * aL)
                dst[m] = np.interp(target, a, self.ths)
        return th_lo * (1 - t_b) + th_hi * t_b

    def half_girth(self, z):
        i = int(np.clip(round((z - self.zs[0]) / 0.1), 0, len(self.zs) - 1))
        a = self.arcs[i]
        return 0.5 * (a[-1] - self.gap / 2 + (-a[0] - self.gap / 2))

    def zrange(self, theta):
        return self.M.bottom(theta), self.M.top(theta)


def build(body, M, rig=None, name='Corset'):
    D = Draft(M)
    H, T = M.H, D.T
    zw = M.z_waist
    ns, nv = N_S, N_V
    S = np.linspace(-1.0, 1.0, ns + 1)
    V = np.linspace(0.0, 1.0, nv + 1)
    # ---- the grid: height at each (v, s) from the edges at that angle, the angle from the height at that strip
    # coordinate: a few passes settle the pair
    th = np.zeros((nv + 1, ns + 1))
    Z = np.zeros_like(th)
    th0 = D.theta_of(S, np.full_like(S, zw))
    for i, v in enumerate(V):
        th[i] = th0
    for _ in range(4):
        zb, zt = M.bottom(th), M.top(th)
        Z = zb + (zt - zb) * V[:, None]
        th = D.theta_of(S[None, :] + 0 * Z, Z)
    zb, zt = M.bottom(th), M.top(th)
    # distance from the edges, in units
    e_bot = Z - zb
    e_top = zt - Z
    e = np.minimum(e_bot, e_top)
    # across: arc distance from each seam, in units (using the half-girth at this height)
    Ls = np.vectorize(D.half_girth)(Z)                                                                  # (the half-girth at each height)
    arc_s = np.abs(S)[None, :] * Ls                                                                    # the arc from the front centre
    sgn = np.sign(S)[None, :]
    # ---- displacement along the surface's normal
    d = np.zeros_like(th)
    mat_ix = np.zeros(th.shape, dtype=int)
    # the binding rolled round both edges
    bead = BEAD_H * np.exp(-((e - 0.5 * BINDING_W) / (0.4 * BINDING_W)) ** 2) * _smooth(-0.02, 0.06, e)
    d += bead
    # the seams: grooves, each between two boning channels
    def groove(dist, w, depth):
        return -depth * np.exp(-(dist / w) ** 2)

    def ridge(dist, off, w, h):
        return h * np.exp(-((np.abs(dist) - off) / w) ** 2)
    stop = _smooth(BONE_STOP * 0.6, BONE_STOP * 1.4, e)
    # plain seams (fractions of the half-girth from the front centre)
    for f in SEAMS:
        if abs(f - GUSSET_BUST[0]) < 1e-9 or abs(f - GUSSET_HIP[0]) < 1e-9:
            continue
        dist = arc_s - f * Ls
        d += groove(dist, SEAM_W, SEAM_D) + ridge(dist, BONE_OFFSET, BONE_W, BONE_H) * stop
    # the gussets: two seams that meet in a point (a wedge let in from the edge), the boning on their outer sides
    for (f, reach, w), from_top in ((GUSSET_BUST, True), (GUSSET_HIP, False)):
        edge_d = e_top if from_top else e_bot
        t = np.clip(1.0 - edge_d / (reach * H), 0.0, 1.0)
        wedge = w * Ls * t ** 0.9
        live = _smooth(0.0, 0.25, t)
        for side in (-1, 1):
            dist = arc_s - f * Ls - side * wedge
            d += (groove(dist, SEAM_W, SEAM_D) + ridge(dist, BONE_OFFSET, BONE_W, BONE_H) * stop * (0.4 if side < 0 else 1.0)) * live
        # (the plain seam carries on above the wedge's point)
        base = groove(arc_s - f * Ls, SEAM_W, SEAM_D) + ridge(arc_s - f * Ls, BONE_OFFSET, BONE_W, BONE_H) * stop
        d += base * (1.0 - live)
    # the busk: a flat steel strip at the front centre, stood a little proud with a hairline between its halves
    busk = BUSK_H * (1.0 - _smooth(BUSK_W * 0.8, BUSK_W * 1.1, arc_s)) * _smooth(-0.05, 0.22, e)
    d += busk + groove(arc_s, 0.018, 0.020) * _smooth(-0.05, 0.22, e)
    # the stiffened back edge (the lacing's reinforcing strip, one more channel at the eyelets' edge)
    dist_back = (1.0 - np.abs(S))[None, :] * Ls
    d += 0.022 * _smooth(EYELET_OFF + 0.30, EYELET_OFF + 0.12, dist_back) * _smooth(-0.02, 0.12, e)
    d += groove(dist_back - (EYELET_OFF + 0.34), 0.03, 0.018) * _smooth(0.0, 0.2, e)
    # ---- points: the surface at (z, theta) plus the displacement along its normal
    R = D.r(Z, th % (2 * math.pi)) + CLEAR          # (clear of the skin by more than a seam is deep; the solidify thickens outward)
    X = -R * np.sin(th)
    Y = D.M.cy - R * np.cos(th)
    P = np.stack([X, Y, Z], axis=-1)
    # normals by differences over the grid
    dS = np.gradient(P, axis=1)
    dV = np.gradient(P, axis=0)
    N = np.cross(dS, dV)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    # (make them point out of the body)
    out = np.stack([X, Y - D.M.cy, 0 * Z], axis=-1)
    flip = np.sign((N * out).sum(-1))
    N *= flip[..., None]
    P = P + N * d[..., None]
    # ---- the mesh: a grid shell (open at the back, between the two eyelet edges, and at both edges)
    bm = bmesh.new()
    verts = [[bm.verts.new(tuple(P[i, j])) for j in range(ns + 1)] for i in range(nv + 1)]
    bind = (e < BINDING_W)
    for i in range(nv):
        for j in range(ns):
            f = bm.faces.new((verts[i][j], verts[i][j + 1], verts[i + 1][j + 1], verts[i + 1][j]))
            f.material_index = 1 if (bind[i, j] and bind[i + 1, j + 1]) else 0
            f.smooth = True
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # outward = away from the axis
    f0 = bm.faces[len(bm.faces) // 2]
    c = f0.calc_center_median()
    if f0.normal.dot(Vector((c.x, c.y - D.M.cy, 0.0))) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    skin = female.trunk_weights if getattr(female, 'RIG', None) and 'arm' in female.RIG else None
    obj = female.mesh_obj(name, bm, 90, skin=skin, subdiv=0, solid=T)
    obj.data.materials.append(female.mat(91))
    objs = [obj]
    # ---- the hardware: busk studs, eyelets, the lace
    objs += _hardware(D, S, th, Z, P, N, e, Ls, skin)
    D.grid = dict(S=S, V=V, th=th, Z=Z, P=P, N=N, e=e)
    bpy.context.view_layer.update()
    return objs, D


def _frame(n):
    """Two unit vectors perpendicular to n."""
    a = Vector((0, 0, 1)) if abs(n.z) < 0.9 else Vector((1, 0, 0))
    u = n.cross(a).normalized()
    return u, n.cross(u).normalized()


def _torus(bm, c, n, R, tube, seg=14, sides=6):
    u, w = _frame(n)
    ring = []
    for a in range(seg):
        ang = 2 * math.pi * a / seg
        cc = c + (u * math.cos(ang) + w * math.sin(ang)) * R
        radial = (u * math.cos(ang) + w * math.sin(ang))
        loop = []
        for b in range(sides):
            bb = 2 * math.pi * b / sides
            loop.append(bm.verts.new(cc + radial * math.cos(bb) * tube + n * math.sin(bb) * tube))
        ring.append(loop)
    for a in range(seg):
        for b in range(sides):
            bm.faces.new((ring[a][b], ring[a][(b + 1) % sides], ring[(a + 1) % seg][(b + 1) % sides], ring[(a + 1) % seg][b]))


def _dome(bm, c, n, r, h, seg=12, rings=4):
    u, w = _frame(n)
    prev = None
    rows = []
    for k in range(rings + 1):
        t = k / rings
        rr = r * math.cos(t * math.pi / 2)
        hh = h * math.sin(t * math.pi / 2)
        if k == rings:
            rows.append([bm.verts.new(c + n * hh)])
        else:
            rows.append([bm.verts.new(c + (u * math.cos(2 * math.pi * a / seg) + w * math.sin(2 * math.pi * a / seg)) * rr + n * hh) for a in range(seg)])
    for k in range(rings - 1):
        for a in range(seg):
            bm.faces.new((rows[k][a], rows[k][(a + 1) % seg], rows[k + 1][(a + 1) % seg], rows[k + 1][a]))
    for a in range(seg):
        bm.faces.new((rows[rings - 1][a], rows[rings - 1][(a + 1) % seg], rows[rings][0]))


def _tube(bm, pts, r, sides=6):
    """A tube along a polyline."""
    pts = [Vector(p) for p in pts]
    rings = []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        u, w = _frame(t)
        rings.append([bm.verts.new(p + (u * math.cos(2 * math.pi * b / sides) + w * math.sin(2 * math.pi * b / sides)) * r) for b in range(sides)])
    for a in range(len(pts) - 1):
        for b in range(sides):
            bm.faces.new((rings[a][b], rings[a][(b + 1) % sides], rings[a + 1][(b + 1) % sides], rings[a + 1][b]))


def _at(G, s, z):
    """Surface point and normal at strip coordinate s and height z (bilinear on the grid, searched by height)."""
    S, Z, P, N = G
    j = int(np.clip(round((s + 1.0) / 2.0 * (len(S) - 1)), 0, len(S) - 1))
    col = Z[:, j]
    i = int(np.clip(np.searchsorted(col, z) - 1, 0, len(col) - 2))
    t = float(np.clip((z - col[i]) / (col[i + 1] - col[i]), 0, 1))
    p = P[i, j] * (1 - t) + P[i + 1, j] * t
    n = N[i, j] * (1 - t) + N[i + 1, j] * t
    return Vector(p), Vector(n).normalized()


def _hardware(D, S, th, Z, P, N, e, Ls, skin):
    M = D.M
    zw = M.z_waist
    T = D.T
    G = (S, Z, P, N)
    out = []
    # busk studs: a row of domed heads down the front centre
    bm = bmesh.new()
    zb, zt = M.bottom(0.0), M.top(0.0)
    for k in range(N_STUDS):
        z = zb + 0.55 + (zt - zb - 1.0) * k / (N_STUDS - 1)
        p, n = _at(G, 0.0, z)
        _dome(bm, p + n * (T + BUSK_H), n, STUD_R, 0.05)
    out.append(female.mesh_obj('Corset busk studs', bm, 92, skin=skin, subdiv=0))
    # eyelets, one pair at each height down the back edges, closer at the waist
    zs_e = []
    zb, zt = M.bottom(math.pi), M.top(math.pi)
    n_e = EYELET_PAIRS
    for k in range(n_e):
        zs_e.append(zt - 0.45 - (zt - zb - 0.9) * k / (n_e - 1))
    # (the waist pair, set closer: the waist tape's two eyelets, a share of the spacing above the waist line)
    bm = bmesh.new()
    pts = {1: [], -1: []}
    for side in (-1, 1):
        for z in zs_e:
            # the eyelet's strip coordinate: EYELET_OFF in from the back edge
            L = D.half_girth(z)
            s = side * (1.0 - EYELET_OFF / L)
            p, n = _at(G, s, z)
            _torus(bm, p + n * (T + 0.01), n, EYELET_R, EYELET_TUBE)
            pts[side].append((p + n * (T + 0.01 + CORD_R + 0.01), n, z))
    out.append(female.mesh_obj('Corset eyelets', bm, 92, skin=skin, subdiv=0))
    # the lace: a cord criss-crossing from the top, in at each eyelet from the outside, out at the waist in two tails
    bm = bmesh.new()
    L_, R_ = pts[-1], pts[1]

    def span(a, b, lift=0.0):
        pa, na, za = a
        pb, nb, zb_ = b
        path = []
        for t in np.linspace(0.0, 1.0, 14):
            z = za + (zb_ - za) * t
            # across the opening at this height: the angle between the two eyelets' edges
            ta = math.atan2(-pa.x, -(pa.y - M.cy)) % (2 * math.pi)      # (both edges lie about the back, theta = pi)
            tb = math.atan2(-pb.x, -(pb.y - M.cy)) % (2 * math.pi)
            tt = ta + (tb - ta) * t
            r = D.r(z, tt % (2 * math.pi)) + T + 0.02 + CORD_R * 2.0 + 0.04 * math.sin(math.pi * t) * 0.0
            path.append((-r * math.sin(tt), M.cy - r * math.cos(tt), z))
        return path
    kw = int(np.argmin([abs(z - zw) for (_, _, z) in L_]))     # (the eyelet pair nearest the waist: where the tails come out)
    for k in range(len(L_) - 1):
        _tube(bm, span(L_[k], R_[k + 1]), CORD_R)
        _tube(bm, span(R_[k], L_[k + 1]), CORD_R)
    # the tails: from the waist pair, down the back over the corset, a hand's breadth apart, loose at the ends
    for pair, side in ((L_[kw], -1), (R_[kw], 1)):
        p, n, z = pair
        path = []
        for t in np.linspace(0.0, 1.0, 22):
            zz = z - 3.4 * t
            tt = math.atan2(-p.x, -(p.y - M.cy)) + side * 0.16 * math.sin(t * math.pi * 0.9) * 0.0 + (-side) * 0.05 * t
            r = D.r(zz, tt % (2 * math.pi)) + T + 0.02 + CORD_R * 2.0 + 0.05 * t
            path.append((-r * math.sin(tt), M.cy - r * math.cos(tt), zz))
        _tube(bm, path, CORD_R)
        # the aglet
        _tube(bm, path[-3:], CORD_R * 1.5)
    out.append(female.mesh_obj('Corset lace', bm, 93, skin=skin, subdiv=0))
    return out


def clearance(body, corset, M, n_theta=240, step=0.25):
    """The gap between the skin and the corset's inner face, measured by casting rays out from the body's axis (the
    evaluated meshes, as posed): returns (least, mean, count of rays that hit the skin but found the corset behind it,
    i.e. clipped). A negative least gap means the skin comes through the corset."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    tb = BVHTree.FromObject(body, dg)
    tc = BVHTree.FromObject(corset, dg)
    gaps = []
    zs = np.arange(M.bottom(math.pi) + 0.5, M.top(0.0) - 0.5, step)
    for z in zs:
        for t in np.linspace(0, 2 * math.pi, n_theta, endpoint=False):
            if abs(((t - math.pi + math.pi) % (2 * math.pi)) - math.pi) < 0.12:
                continue                                    # (the back opening: no cloth there)
            if not (M.bottom(t) + 0.3 < z < M.top(t) - 0.3):
                continue
            o = Vector((0.0, M.cy, z))
            d = Vector((-math.sin(t), -math.cos(t), 0.0))
            hb = tb.ray_cast(o, d, 30.0)
            hc = tc.ray_cast(o, d, 30.0)
            if hb[0] is None or hc[0] is None:
                continue
            gaps.append((hc[1] and (hc[0] - o).length) - (hb[0] - o).length)
    g = np.array(gaps)
    return float(g.min()), float(g.mean()), int((g < 0).sum()), len(g)
