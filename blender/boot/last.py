"""A tall leather boot made as a bootmaker makes it: sheets of leather cut to a pattern, lapped one over the other
and stitched, on a sole. Drafted on whatever legs it is given, from the sole to just under the knee, over the hose.

Nothing here is placed by hand. measure_legs() (the hose's own tailor's tape, blender/hose/draft.py) reads the heel,
toe, ankle, calf and knee off the skin, and every size of the boot is a share of the head's height H, the figure's
unit, so the boot re-fits when the body or the head changes.

Leather is not hose. It does not stretch and it does not follow the leg. So first the last is made, the shape the
leather is worked over, and then the pieces are cut from it:

  the last     the shaft: level sections of the leg, each pulled taut (its convex hull), and a surface round them
               that closes in on the leg only as fast as a stiff hide can (slope: it runs straight from the calf
               down past the ankle's hollow), is never narrower at the ankle than the heel needs to pass
               (ankle_girth), and stands off by the ease. The foot: the skin of the foot let out by its room, with
               a toe box (the toe region carried a little forward from the ball of the foot, and raised). The two
               meet over the ankle.

  the pieces   the vamp      the front of the foot, from the toe back to the side seams.
               two quarters  the inner and the outer, round the heel from the side seams to the heel's back seam;
                             their front edges lap over the vamp, and the outer laps over the inner at the back.
               the shaft     one sheet wrapped round the leg from the ankle seam to the top. Its two edges meet
                             down the back of the leg, one lapped over the other (the back seam), and its lower
                             edge laps over the vamp and the quarters (the ankle seam).
               the binding   a strip round the top edge.
               the sole      a slab under the foot, wider than the upper all round by the welt.

Each piece is its own sheet with its own cut edges and its own thickness; a piece that laps over another lies on
it, a thickness further out. Every lapped edge carries a row of stitches ('seam_d', each point's distance from its
piece's stitched edge, and 'seam_u', the distance along that edge: the leather's shader draws the stitches from
them, stitch_in in from the edge), and the welt carries the row that holds the upper to the sole.

The pieces are kept as they were cut (PIECES, for each side: every piece's points on the last, its faces, and its
seam edges by letter, each edge paired with the edge of the same letter on the piece it is sewn to), so pattern.py
can lay them flat. The seams: A the vamp to each quarter, B the quarters' back seam, C the ankle seam, D the
shaft's back seam, E the upper to the sole, F the binding to the top.

The figure stands on the sole: its under face is M['boot_floor']."""
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('hose', 'vest'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import draft  # noqa: E402   (the hose's: the tape, the body's data)
import fit  # noqa: E402     (the vest's cutter and hull)

# Every size as a share of the head's height H (but the plain ratios, PLAIN).
RATIOS = {
    'leather': 0.014,        # the leather's thickness (about 3.5 mm)
    'ease': 0.045,           # the shaft's room over the leg's taut section (a finger's width all round)
    'slope': 0.06,           # the most the shaft closes in on the leg for each unit it goes down
    'ankle_girth': 1.42,     # the shaft's least girth, as a multiple of the ankle's (the heel goes in through it)
    'flare': 0.025,          # the top's flare
    'flare_span': 0.30,
    'room': 0.022,           # the foot's room over the hose
    'gap': 0.024,            # the least room between the skin and the leather anywhere (the hose is 0.0105 of it)
    'toe_room': 0.045,       # the toe box's height over the toe
    'toe_long': 0.070,       # how much longer the toe region is than the foot's (a share of its length)
    'bridge': 0.16,          # the most the leather stands off a hollow of the foot it bridges
    'top_below_knee': 0.12,  # the top edge below the knee's joint
    'join': 0.16,            # the span over the ankle where the last's shaft meets its foot
    'lap': 0.040,            # how far one piece laps over the next at a seam (about 1 cm)
    'vamp_back': 0.20,       # the vamp's side seam reaches the sole this share of the foot's length ahead of the ankle
    'binding': 0.050,        # the top binding's depth
    'under': 0.035,          # how far the upper comes down past the foot's under face, to the sole
    'sole': 0.035,           # the sole's thickness (two layers of sole leather, about 9 mm)
    'welt': 0.022,           # how far the sole stands out past the upper
    'stitch_in': 0.012,      # the stitches' row, this far in from the edge
}
PLAIN = ('slope', 'ankle_girth', 'toe_long', 'vamp_back')
ROW = 0.15                   # the shaft's sections, this far apart
COLS = 96                    # and this many directions round each
GRAIN = 0.10                 # the sheets' own mesh, about this far between points
FAR = 9.0                    # 'seam_d' where a point has no stitched edge near
PIECES = {}                  # the pieces as cut, by side (see the top of this file)
NEAR = 0.015                 # a point this near a cut line is on it


def _ss(a, b, x):
    return draft._ss(a, b, x)


def _radial(h, th):
    """The radius of the convex polygon h (round the origin) in each direction th (from -y, turning toward +x)."""
    sn, cs = np.sin(th), np.cos(th)
    dx, dy = sn[:, None], -cs[:, None]
    a, b = h, np.roll(h, -1, 0)
    e = b - a
    den = dx * e[None, :, 1] - dy * e[None, :, 0]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
    s = (a[None, :, 0] * dy - a[None, :, 1] * dx) / den
    t = np.where((s >= -1e-6) & (s <= 1 + 1e-6) & (t > 0), t, -1.0)
    return t.max(1)


def shaft(P, leg, M, z_top, r, log=print):
    """The last's shaft as a table of radii [section, direction] round a centre line, from the ankle's joint to over
    the top. Returns (zs, centres [section, 2], radii [section, direction])."""
    zs = np.arange(M['ankle_z'], z_top + 1.0, ROW)
    th = np.linspace(-np.pi, np.pi, COLS, endpoint=False)
    C, R = np.zeros((len(zs), 2)), np.zeros((len(zs), COLS))
    for i, z in enumerate(zs):
        q = P[leg & (np.abs(P[:, 2] - z) < 0.6 * ROW)][:, :2]
        C[i] = np.array(fit._hull(q)).mean(0)
    C = fit._blur(C, 4.0, 0, False)
    for i, z in enumerate(zs):
        q = P[leg & (np.abs(P[:, 2] - z) < 0.6 * ROW)][:, :2]
        R[i] = _radial(np.array(fit._hull(q - C[i])), th)
    taut = R.copy()
    # the hide closes in on the leg no faster than the slope: each section stands at least as far out as every other
    # one, less the slope over the height between them
    dz = np.abs(zs[:, None] - zs[None, :])
    R = (R[None, :, :] - r['slope'] * dz[:, :, None]).max(1)
    R = R + r['ease'] + r['flare'] * _ss(z_top - r['flare_span'], z_top, zs)[:, None] ** 2
    girth = (R * (2 * np.pi / COLS)).sum(1)
    R *= np.maximum(1.0, r['ankle_girth'] * M['ankle'] / girth)[:, None]       # (the heel has to pass)
    R = fit._blur(fit._blur(R, 2.0, 1, True), 1.5, 0, False)
    g = (R * (2 * np.pi / COLS)).sum(1)
    cm = 165.0 / 63.0
    for z0, lab in ((M['ankle_min_z'], 'ankle'), (M['calf_z'], 'calf'), (z_top, 'top')):
        i = int(np.argmin(np.abs(zs - z0)))
        tg = (taut[i] * (2 * np.pi / COLS)).sum()
        log('boot: the shaft at the %s (z %.2f) is %.0f cm round, over a leg of %.0f cm: it stands off %.2f on average'
            % (lab, zs[i], g[i] * cm, tg * cm, (R[i] - taut[i]).mean()))
    return zs, C, R


def _at(T, z, a):
    """The shaft table T at heights z and directions a: the centres [n, 2] and the radii [n]."""
    zs, C, R = T
    zi = np.clip((z - zs[0]) / ROW, 0, len(zs) - 1.001)
    i0 = zi.astype(int)
    f = zi - i0
    c = C[i0] * (1 - f)[:, None] + C[i0 + 1] * f[:, None]
    ai = (np.mod(a + np.pi, 2 * np.pi)) / (2 * np.pi) * COLS
    j0 = np.floor(ai).astype(int) % COLS
    j1 = (j0 + 1) % COLS
    g = ai - np.floor(ai)
    rad = ((R[i0, j0] * (1 - g) + R[i0, j1] * g) * (1 - f) + (R[i0 + 1, j0] * (1 - g) + R[i0 + 1, j1] * g) * f)
    return c, rad


def _on_shaft(X, T):
    """Each point carried out along its direction from the centre line onto the shaft."""
    c, _ = _at(T, X[:, 2], np.zeros(len(X)))
    d = X[:, :2] - c
    a = np.arctan2(d[:, 0], -d[:, 1])
    c, rad = _at(T, X[:, 2], a)
    out = X.copy()
    out[:, :2] = c + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9) * rad[:, None]
    return out


def _sheet(T, z0, z1, a0, turn, sgn, lift):
    """A sheet of leather wrapped on the shaft: from height z0 to z1, from the direction a0 round by turn (radians;
    sgn, the leg's side, sets which way), each point lifted off the last by lift(z, t) (t the way round from a0).
    Returns (points [rows * cols, 3], faces, z and t of each point, the sheet's mean radius)."""
    nr = max(2, int(round((z1 - z0) / GRAIN)) + 1)
    z = np.linspace(z0, z1, nr)
    _, rad = _at(T, np.full(8, 0.5 * (z0 + z1)), np.linspace(-np.pi, np.pi, 8, endpoint=False))
    rm = float(rad.mean())
    nc = max(2, int(round(turn * rm / GRAIN)) + 1)
    t = np.linspace(0.0, turn, nc)
    zz, tt = (g.ravel() for g in np.meshgrid(z, t, indexing='ij'))
    a = a0 + sgn * tt
    c, rad = _at(T, zz, a)
    rad = rad + lift(zz, tt)
    X = np.stack([c[:, 0] + np.sin(a) * rad, c[:, 1] - np.cos(a) * rad, zz], 1)
    faces = []
    for j in range(nr - 1):
        for i in range(nc - 1):
            q = [j * nc + i, j * nc + i + 1, (j + 1) * nc + i + 1, (j + 1) * nc + i]
            faces.append(q if sgn > 0 else q[::-1])                          # (the grain side out)
    return X, faces, zz, tt, rm


def _weights(Pskin, Wskin, X, k=4):
    """The skin's bone weights at the points X: its nearest few points' weights, the nearer the more."""
    kd = KDTree(len(Pskin))
    for i, p in enumerate(Pskin):
        kd.insert(p, i)
    kd.balance()
    out = np.zeros((len(X), Wskin.shape[1]), dtype=np.float32)
    for i, p in enumerate(X):
        hits = kd.find_n(p, k)
        w = np.array([1.0 / max(h[2], 1e-4) for h in hits])
        out[i] = (Wskin[[h[1] for h in hits]] * (w / w.sum())[:, None]).sum(0)
    return out


def _object(name, V, faces, Wk, names, body, rig, material, seam_d, seam_u, thickness=None):
    """A piece (or several, as islands) as an object on the body's bones, with the attributes the leather's shader
    reads: 'rest', 'seam_d', 'seam_u'."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], faces)
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    a = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    a.data.foreach_set('vector', V.ravel().astype(np.float32))
    for nm, val in (('seam_d', seam_d), ('seam_u', seam_u)):
        a = me.attributes.new(nm, 'FLOAT', 'POINT')
        a.data.foreach_set('value', val.astype(np.float32))
    for gi, nm in enumerate(names):
        if nm.split('.')[0] in draft.ARM_GROUPS or nm == 'shoulder smooth':
            continue
        w = Wk[:, gi]
        if (w > 1e-4).any():
            g = o.vertex_groups.new(name=nm)
            for val in np.unique(np.round(w[w > 1e-4], 3)):
                g.add([int(i) for i in np.nonzero(np.abs(np.round(w, 3) - val) < 1e-6)[0]], float(val), 'REPLACE')
    if rig is not None:
        src = next((m for m in body.modifiers if m.type == 'ARMATURE'), None)
        m = o.modifiers.new('rig', 'ARMATURE')
        m.object = rig
        if src is not None:
            m.use_deform_preserve_volume = src.use_deform_preserve_volume
    if thickness:
        m = o.modifiers.new('thickness', 'SOLIDIFY')
        m.thickness = thickness
        m.offset = 1.0
        m.use_rim = True
        m.use_even_offset = False
        m.use_quality_normals = True
    return o


def _compact(D, fs):
    used = sorted({i for f in fs for i in f})
    km = -np.ones(len(D), dtype=np.int64)
    km[used] = np.arange(len(used))
    return D[used], [[int(km[i]) for i in f] for f in fs]


def _sole(edge, z_welt, r):
    """The sole: a slab under the upper's cut edge (its points, edge), its upper face at z_welt, wider than the upper
    all round by the welt, its rims eased. Returns (points, faces, seam_d, seam_u): the welt's stitches run midway
    between the upper and the rim."""
    c = 0.5 * (edge[:, :2].min(0) + edge[:, :2].max(0))
    na = 144
    ang = np.arctan2(edge[:, 0] - c[0], -(edge[:, 1] - c[1]))
    rr = np.linalg.norm(edge[:, :2] - c, axis=1)
    bins = np.floor((ang + np.pi) / (2 * np.pi) * na).astype(int) % na
    have = np.unique(bins)
    rad = np.array([rr[bins == b].max() for b in have])
    rad = np.interp(np.arange(na), np.concatenate([have - na, have, have + na]), np.tile(rad, 3))
    rad = fit._blur(rad, 4.0, 0, True) + r['leather']                        # (the upper's outer face)
    aa = (np.arange(na) + 0.5) / na * 2 * np.pi - np.pi
    bev = 0.18 * r['sole']
    out, zb = rad + r['welt'], z_welt - r['sole']
    rings = [(rad - 2.0 * r['welt'], z_welt), (out - bev, z_welt), (out, z_welt - bev), (out, zb + bev), (out - bev, zb)]
    V = [np.stack([c[0] + np.sin(aa) * rk, c[1] - np.cos(aa) * rk, np.full(na, zk)], 1) for rk, zk in rings]
    V.append(np.array([[c[0], c[1], z_welt], [c[0], c[1], zb]]))
    V = np.concatenate(V)
    sd = np.full(len(V), FAR)
    sd[:na] = r['stitch_in'] - 2.5 * r['welt']                               # (measured from the middle of the welt)
    sd[na:2 * na] = r['stitch_in'] + 0.5 * r['welt'] - bev
    su = np.zeros(len(V))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(V[na:2 * na, :2], axis=0), axis=1))])
    su[:na], su[na:2 * na] = arc, arc
    F = []
    for k in range(4):
        for i in range(na):
            j = (i + 1) % na
            F.append([k * na + i, k * na + j, (k + 1) * na + j, (k + 1) * na + i])
    for i in range(na):
        j = (i + 1) % na
        F.append([5 * na, j, i])
        F.append([5 * na + 1, 4 * na + i, 4 * na + j])
    return V, F, sd, su


def build_boots(body, M, rig=None, material=None, sole_material=None, name='Boot', ratios=RATIOS, log=print):
    """Make one boot for each leg from the measures M (draft.measure_legs) and hand them to the body's bones.
    Returns {'R': upper, 'L': upper, 'sole R': sole, 'sole L': sole}. M gains 'boot_top' (the top edge's height) for
    whatever goes over it, and 'boot_floor' (the sole's under face)."""
    B = draft.body_data(body)
    P, Nn, W, names = B['P'], B['N'], B['W'], B['groups']
    H = M['H']
    r = {k: v * H if k not in PLAIN else v for k, v in ratios.items()}
    tl = r['leather']
    col = {n: i for i, n in enumerate(names)}
    arm = sum(W[:, col[n]] for n in names if n.split('.')[0] in draft.ARM_GROUPS)
    z_top = M['knee_z'] - r['top_below_knee']
    M['boot_top'] = z_top
    L = M['heel_y'] - M['toe_y']
    y_ball = M['toe_y'] + 0.30 * L                     # the ball of the foot: 70 percent of the way from the heel
    z_low = M['ankle_z'] + r['join']                   # from here up the last is all shaft
    z_seam = z_low + r['lap']                          # the ankle seam: the vamp and quarter end here, under the shaft

    region = (P[:, 2] < z_seam + 0.9) & (arm < 0.02)
    idx = np.nonzero(region)[0]
    remap = -np.ones(len(P), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    faces = [[int(remap[i]) for i in f] for f in B['faces'] if all(region[i] for i in f)]
    X0, Nr, Wr = P[idx], Nn[idx], W[idx]
    E = remap[B['E'][region[B['E'][:, 0]] & region[B['E'][:, 1]]]]
    n = len(X0)
    z, y = X0[:, 2], X0[:, 1]
    z_foot = float(z.min())                            # the foot's under face
    z_welt = z_foot - 0.35 * r['under']                # the sole's upper face: the upper is cut off here
    legs = (arm < 0.02) & (P[:, 2] < z_top + 1.5)
    Pl, Wl = P[legs], W[legs]                          # (the skin the sheets take their bones from)
    M['boot_floor'] = z_welt - r['sole']

    # ---- the last's foot: the room, the toe box, and down past the foot to the sole
    down = np.clip(-Nr[:, 2], 0.0, 1.0) ** 2 * _ss(1.4, 0.3, z)
    ease = np.full(n, r['room'])
    toe = _ss(y_ball, M['toe_y'], y) * (z < M['ankle_z'])
    ease += r['toe_room'] * toe * np.clip(Nr[:, 2], 0.0, 1.0)             # (the toe box stands up over the toes)
    ease *= 1.0 - 0.75 * down
    Xf = X0 + Nr * ease[:, None]
    fwd = Xf[:, 1] < y_ball
    Xf[fwd, 1] = y_ball + (Xf[fwd, 1] - y_ball) * (1.0 + r['toe_long'])
    Xf[:, 2] -= r['under'] * down

    # ---- the last's shaft, for each leg, and where it meets the foot
    X = Xf.copy()
    tables = {}
    for sgn in (-1, 1):
        leg = (P[:, 0] * sgn > 0) & (arm < 0.02)
        tables[sgn] = shaft(P, leg, M, z_top, r, log=log if sgn < 0 else (lambda *a: None))
        mine = X0[:, 0] * sgn > 0
        w = _ss(M['ankle_z'] - 0.5 * r['join'], z_low, z[mine])
        X[mine] = _on_shaft(X0[mine], tables[sgn]) * w[:, None] + Xf[mine] * (1 - w)[:, None]
    # the foot's hollows bridged
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    lift = np.zeros(n)
    for _ in range(160):
        acc = np.zeros_like(X)
        np.add.at(acc, E[:, 0], X[E[:, 1]])
        np.add.at(acc, E[:, 1], X[E[:, 0]])
        lap = acc / np.maximum(deg, 1)[:, None] - X
        out = np.minimum(np.maximum((lap * Nr).sum(1), 0.0) * 0.6, r['bridge'] - lift)
        lift += out
        X += Nr * out[:, None]
    # and never nearer the skin than the gap
    tree = BVHTree.FromPolygons([tuple(p) for p in P], B['faces'])
    pushed = 0
    for i in range(n):
        loc, nrm, _, d = tree.find_nearest(X[i])
        if loc is None:
            continue
        side = float(np.dot(X[i] - np.array(loc), np.array(nrm)))
        if side < r['gap']:
            X[i] += np.array(nrm) * (r['gap'] - side)
            pushed += 1
    log('boot: the last\'s foot, %d points; %d bridged over hollows (by %.3f at most), %d put out to the gap'
        % (n, int((lift > 0.01).sum()), lift.max(), pushed))

    # ---- the foot's pieces, cut from the last: columns 0:3 the skin, 3:6 the last, 6:9 the normal, then the weights
    nw = len(names)
    D0 = np.concatenate([X0, X, Nr, Wr], 1)
    D0, f0 = fit.trim(D0, faces, lambda A: z_seam - A[:, 5])                 # (the ankle seam)
    D0, f0 = fit.trim(D0, f0, lambda A: A[:, 5] - z_welt)                    # (the sole)
    res = {}
    for side, sgn in (('R', -1), ('L', 1)):
        T = tables[sgn]
        yc = float(T[1][0][1])                                              # (the ankle's middle)
        m_seam = r['vamp_back'] * L / (z_seam - z_welt)                     # (the side seam runs forward as it goes down)
        kc = 1.0 / np.sqrt(1.0 + m_seam ** 2)

        def side_seam(A):                                                   # (ahead of the seam: below zero)
            return A[:, 4] - (yc - m_seam * (z_seam - A[:, 5]))
        mine = [f for f in f0 if sgn * D0[f, 0].mean() > 0]
        V, F, SD, SU, WW = [], [], [], [], []

        def add(Vp, Fp, sd, su, Wp):
            n0 = sum(len(v) for v in V)
            V.append(Vp)
            F.extend([[n0 + i for i in f] for f in Fp])
            SD.append(sd)
            SU.append(su)
            WW.append(Wp)
        xc = float(T[1][0][0])

        def outer(A):                                                       # (the outer side of the foot: above zero)
            return (A[:, 3] - xc) * sgn

        def foot_edges(D, **more):
            e = {'C': np.abs(D[:, 5] - z_seam) < NEAR, 'E': np.abs(D[:, 5] - z_welt) < NEAR}
            e.update(more)
            return e
        pieces = []
        # the vamp: ahead of the side seams, under everything
        Dv, fv = _compact(*fit.trim(D0, mine, lambda A: -side_seam(A)))
        add(Dv[:, 3:6], fv, np.full(len(Dv), FAR), np.zeros(len(Dv)), Dv[:, 9:9 + nw])
        on = np.abs(side_seam(Dv)) < NEAR
        pieces.append(dict(name='vamp', V=V[-1], F=fv, up=(0, -1, 0.35),
                           edges=foot_edges(Dv, **{'A inner': on & (outer(Dv) < 0), 'A outer': on & (outer(Dv) > 0)})))
        # the quarters: behind the side seams, their front edges lapped over the vamp; the outer one laps over the
        # inner at the heel's back seam
        for q, cut in (('inner', lambda A: -outer(A)), ('outer', lambda A: outer(A) + r['lap'])):
            Dq, fq = _compact(*fit.trim(*fit.trim(D0, mine, lambda A: side_seam(A) + r['lap']), cut))
            nq = Dq[:, 6:9] / np.maximum(np.linalg.norm(Dq[:, 6:9], axis=1, keepdims=True), 1e-9)
            d_front = (side_seam(Dq) + r['lap']) * kc
            d_back = cut(Dq) if q == 'outer' else np.full(len(Dq), FAR)
            up = 1.1 * tl * (1.0 + (_ss(4.0 * r['lap'], 0.0, d_back) if q == 'outer' else 0.0))
            add(Dq[:, 3:6] + nq * (up * np.ones(len(Dq)))[:, None], fq,
                np.minimum(d_front, d_back), np.where(d_front < d_back, Dq[:, 5] / kc, Dq[:, 5]), Dq[:, 9:9 + nw])
            pieces.append(dict(name=q + ' quarter', V=V[-1], F=fq, up=(0, 0, 1), edges=foot_edges(
                Dq, **{'A ' + q: np.abs(d_front) < NEAR, 'B': np.abs(cut(Dq)) < NEAR})))
        n_foot = sum(len(v) for v in V)
        # the shaft: one sheet from the ankle seam (lapped over the foot's pieces) to the top, its edges meeting down
        # the back of the leg, the last of it lapped over the first
        a0 = -sgn * np.pi
        _, rad0 = _at(T, np.array([0.5 * (z_low + z_top)]), np.array([a0]))
        over = r['lap'] / float(rad0[0])
        turn = 2 * np.pi + over

        def lift_shaft(zz, tt):
            return (3.3 * tl * _ss(z_seam + 4.0 * r['lap'], z_seam, zz)       # (out over the vamp and the quarters)
                    + 1.1 * tl * _ss(turn - 6.0 * over, turn - over, tt))     # (and over its own first edge)
        G, gf, zz, tt, rm = _sheet(T, z_low, z_top, a0, turn, sgn, lift_shaft)
        d_side, d_low = rm * (turn - tt), zz - z_low
        add(G, gf, np.minimum(d_side, d_low), np.where(d_side < d_low, zz, rm * tt), _weights(Pl, Wl, G))
        pieces.append(dict(name='shaft', V=G, F=gf, up=(0, 0, 1), edges={
            'C': zz < z_low + 1e-6, 'D': (tt < 1e-6) | (tt > turn - 1e-6), 'F': zz > z_top - 1e-6}))
        # the binding: a strip round the top edge, over the shaft, stitched along its lower edge
        zb = z_top - r['binding']
        G, gf, zz, tt, rm = _sheet(T, zb, z_top + 0.3 * tl, a0 + sgn * 0.5 * over, turn, sgn,
                                   lambda zz, tt: 1.15 * tl + 1.1 * tl * _ss(turn - 6.0 * over, turn - over, tt))
        add(G, gf, zz - zb, rm * tt, _weights(Pl, Wl, G))
        pieces.append(dict(name='binding', V=G, F=gf, up=(0, 0, 1), edges={'F': zz < zb + 1e-6}))
        Vall = np.concatenate(V)
        o = _object('%s.%s' % (name, side), Vall, F, np.concatenate(WW), names, body, rig, material,
                    np.concatenate(SD), np.concatenate(SU), thickness=tl)
        res[side] = o
        log('boot: %s, %d points: %s' % (o.name, len(Vall), ', '.join('%s %d' % (p['name'], len(p['V'])) for p in pieces)))
        # the sole, under the upper's cut edge
        edge = Vall[:n_foot]
        Sv, Sf, sd, su = _sole(edge[np.abs(edge[:, 2] - z_welt) < 0.12], z_welt, r)
        so = _object('%s sole.%s' % (name, side), Sv, Sf, _weights(Pl, Wl, Sv), names, body, rig,
                     sole_material or material, sd, su)
        bm = bmesh.new()
        bm.from_mesh(so.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(so.data)
        bm.free()
        so.data.polygons.foreach_set('use_smooth', [False] * len(so.data.polygons))
        res['sole ' + side] = so
        pieces.append(dict(name='sole', V=Sv, F=Sf, up=(0, -1, 0), flat=np.arange(144, 288), edges={}))
        PIECES[side] = pieces
    log('boot: the sole %.2f thick, its under face at z %.2f (%.2f under the foot)' % (r['sole'], M['boot_floor'],
                                                                                   z_foot - M['boot_floor']))
    return res


def clearance(boot, hose, log=print):
    """The least gap between the boot's inner face and the hose's outer face, at the current frame, over every point
    of the hose under the boot (judged by where it lies at rest: on the boot's own leg, between the sole and the
    top): negative where the hose comes through the leather. Both are measured on their own surface, without
    thickness."""
    def surf(o):
        for m in o.modifiers:
            if m.type == 'SOLIDIFY':
                m.show_viewport = False
        e = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = e.to_mesh()
        me.calc_loop_triangles()
        pts = [o.matrix_world @ v.co for v in me.vertices]
        tris = [tuple(t.vertices) for t in me.loop_triangles]
        rest = np.zeros(len(me.vertices) * 3)
        me.attributes['rest'].data.foreach_get('vector', rest)
        e.to_mesh_clear()
        for m in o.modifiers:
            if m.type == 'SOLIDIFY':
                m.show_viewport = True
        return pts, tris, rest.reshape(-1, 3)
    bp, bt, br = surf(boot)
    tree = BVHTree.FromPolygons(bp, bt)
    hp, _, hr = surf(hose)
    side = -1 if boot.name.endswith('.R') else 1
    z_top, z_low = float(br[:, 2].max()) - 0.3, float(br[:, 2].min()) + 0.3
    gaps = []
    for p, q in zip(hp, hr):
        if q[0] * side <= 0 or q[2] > z_top or q[2] < z_low:
            continue
        loc, nrm, _, d = tree.find_nearest(p)
        if loc is None:
            continue
        gaps.append(-d if (p - loc).dot(nrm) > 0 else d)       # (outside the boot's surface: through the leather)
    g = np.array(gaps)
    out = {'min': float(g.min()), 'mean': float(g.mean()), 'through': int((g < 0).sum()), 'n': len(g)}
    log('boot clearance %s: hose to leather min %.3f mean %.3f, %d of %d hose points through the leather' % (
        boot.name, out['min'], out['mean'], out['through'], out['n']))
    return out
