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

  the pieces   the toe cap   the toe, ahead of a seam curved across the foot just short of the ball; it laps
                             back over the vamp. (Leather will not lie flat over the toe's dome in one piece with
                             the instep: the cap takes the dome, and is blocked to it wet.)
               the vamp      the instep, from the toe cap back to the side seams.
               two quarters  the inner and the outer, round the heel from the side seams to the heel's back seam;
                             their front edges lap over the vamp, and the outer laps over the inner at the back.
               the shaft     two long panels, the inner and the outer, each from the back of the leg round to the
                             front, from the ankle seam to the top, cut to the calf. The outer laps over the inner
                             down the back (the back seam), their lower edges lap over the vamp and the quarters (the
                             ankle seam), and at the front they stop short of each other by the lacing's opening
                             (lace_gap). Laced, the shaft does not have to pass the heel closed, so it can follow the
                             calf and close in at the ankle.
               the tongue    a long strip under the opening, its sides sewn under the panels.
               the bindings  a strip along each panel's top edge.
               the sole      a slab under the foot, wider than the upper all round by the welt.

Each piece is its own sheet with its own cut edges and its own thickness; a piece that laps over another lies on
it, a thickness further out. Every lapped edge carries a row of stitches ('seam_d', each point's distance from its
piece's stitched edge, and 'seam_u', the distance along that edge: the leather's shader draws the stitches from
them, stitch_in in from the edge), and the welt carries the row that holds the upper to the sole.

The pieces are kept as they were cut (PIECES, for each side: every piece's points on the last, its faces, and its
seam edges by letter, each edge paired with the edge of the same letter on the piece it is sewn to), so pattern.py
can lay them flat. The seams: A the vamp to each quarter, B the quarters' back seam, C the ankle seam, D the
shaft's back seam, E the upper to the sole, F the binding to the top, G the toe cap to the vamp, H the tongue under
the panels; L is each panel's front edge, the lacing's, with its eyelets (each panel's 'eyelets').

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
    'ease': 0.035,           # the shaft's room over the leg's taut section (a finger's width all round)
    'slope': 0.20,           # the most the shaft closes in on the leg for each unit it goes down
    'ankle_girth': 1.12,     # the shaft's least girth, as a multiple of the ankle's (laced: the heel goes in open)
    'flare': 0.0,          # the top's flare
    'flare_span': 0.30,
    'calf_swell': 0.015,     # the shaft's own swell over the calf, past the ease (as a laced boot is cut)
    'calf_span': 0.25,       # over this far either way of the calf's fullest
    'room': 0.022,           # the foot's room over the hose
    'instep_room': 0.040,    # the extra room over the instep and the front of the ankle, where a stride presses the foot up
    'gap': 0.024,            # the least room between the skin and the leather anywhere (the hose is 0.0105 of it)
    'toe_room': 0.045,       # the toe box's height over the toe
    'toe_long': 0.070,       # how much longer the toe region is than the foot's (a share of its length)
    'bridge': 0.16,          # the most the leather stands off a hollow of the foot it bridges
    'top_below_knee': 0.12,  # the top edge below the knee's joint
    'join': 0.16,            # the span over the ankle where the last's shaft meets its foot
    'lap': 0.040,            # how far one piece laps over the next at a seam (about 1 cm)
    'throat': 0.40,          # the vamp's throat (the lacing's foot) this share of the foot's length behind the toe
    'vamp_sole': 0.62,       # and its seam reaches the sole this share of the length behind the toe
    'throat_round': 0.25,    # how fast the seam sweeps back either side of the throat (per unit across, squared)
    'cap_at': 0.24,          # the toe cap's seam crosses the middle of the foot this share of its length behind the toe
    'cap_sweep': 0.08,       # and sweeps back this share of the length further at the sides of the foot
    'binding': 0.050,        # the top binding's depth
    'under': 0.035,          # how far the upper comes down past the foot's under face, to the sole
    'sole': 0.035,           # the sole's thickness (two layers of sole leather, about 9 mm)
    'welt': 0.022,           # how far the sole stands out past the upper
    'stitch_in': 0.012,      # the stitches' row, this far in from the edge
    'lace_gap': 0.035,       # the lacing's opening down the front, drawn tight (about 9 mm between the edges)
    'binding_ring': 0.0,     # 1 sews a binding band round the top (0: none, the panels' top edges bare)
    'top_fit': 0.85,         # how far the top is drawn in from the calf's width to the leg's own (1: all the way)
    'top_snug': 0.65,        # how much of the ease is taken out at the top (laced close to the leg)
    'top_back': 0.045,       # the panels' top edge stands this much higher at the back than at the front
    'tongue_rise': 0.140,    # the last carried on this far above the boot's top, for the tongue (the most it can stand)
    'tongue': 0.050,         # (the vamp's throat: how far it is cut under each panel past its front edge)
    'eyelet_in': 0.028,      # the eyelets' row, this far in from the front edge
    'eyelet_pitch': 0.085,   # and this far apart up it (about 2.2 cm)
    'eyelet_first': 0.045,   # the lowest, this far over the ankle seam
    'eyelet': 0.018,         # an eyelet's outer radius (its ring about 9 mm across)
    'lace': 0.0042,          # the lace's radius (a round cord about 2.7 mm across)
}
PLAIN = ('top_snug', 'top_fit', 'slope', 'ankle_girth', 'toe_long', 'throat', 'vamp_sole', 'throat_round', 'cap_at', 'cap_sweep')
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
    # over the calf the lacing draws the leather in to the leg: the calf's width is not carried on up to the top
    # (which left the top a wide ring standing off the leg all round), the leg's own section is, nearly
    w = (r['top_fit'] * _ss(M['calf_z'], z_top - 0.5, zs))[:, None]
    R = R * (1.0 - w) + taut * w
    # (laced, the leather lies closer to the leg the higher it goes: the ease is let down over the calf to the top)
    R = R + (r['ease'] * (1.0 - r['top_snug'] * _ss(M['calf_z'], z_top - 0.3, zs)))[:, None] \
        + r['flare'] * _ss(z_top - r['flare_span'], z_top, zs)[:, None] ** 2
    R = R + r['calf_swell'] * np.exp(-((zs - M['calf_z']) / r['calf_span']) ** 2)[:, None]
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


def _eyelet_line(D, dist, front, at, r):
    """Where a panel's eyelets go: along the line at distance at from the middle (dist, each point's), in front,
    eyelet_pitch apart by its length, the lowest eyelet_first up it from the throat and the top one short of the top
    by half a pitch. Returns the indices of the panel's points nearest each."""
    on = np.nonzero(front & (np.abs(dist - at) < 0.6 * GRAIN))[0]
    if len(on) < 2:
        return np.zeros(0, dtype=np.int64)
    on = on[np.argsort(D[on, 5])]
    P = D[on, 3:6]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    out = []
    for s_ in np.arange(r['eyelet_first'], arc[-1] - 0.5 * r['eyelet_pitch'], r['eyelet_pitch']):
        out.append(int(on[np.argmin(np.abs(arc - s_))]))
    return np.array(out, dtype=np.int64)


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


def _object(name, V, faces, Wk, names, body, rig, material, seam_d, seam_u, thickness=None, skive=None):
    """A piece (or several, as islands) as an object on the body's bones, with the attributes the leather's shader
    reads: 'rest', 'seam_d', 'seam_u'. thickness: solidified outward by it, each lapped edge thinned over skive."""
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
        # each lapped edge skived down to a feather, as a currier thins it, so a lap reads as a line, not a ledge
        if skive:
            sk = np.clip(seam_d / skive, 0.0, 1.0)
            g = o.vertex_groups.new(name='skive')
            for val in np.unique(np.round(sk, 2)):
                g.add([int(i) for i in np.nonzero(np.abs(np.round(sk, 2) - val) < 1e-6)[0]], float(val), 'REPLACE')
            m.vertex_group = 'skive'
            m.thickness_vertex_group = 0.05
    return o


def _compact(D, fs):
    used = sorted({i for f in fs for i in f})
    km = -np.ones(len(D), dtype=np.int64)
    km[used] = np.arange(len(used))
    return D[used], [[int(km[i]) for i in f] for f in fs]


def _sole(edge, z_welt, r):
    """The sole: a slab under the upper's cut edge (its points, edge), its upper face at z_welt, wider than the upper
    all round by the welt, its rims eased. Returns (points, faces, seam_d, seam_u, the welt's row of stitches: a
    closed line midway between the upper and the rim)."""
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
    path = np.stack([c[0] + np.sin(aa) * (rad + 0.5 * r['welt']), c[1] - np.cos(aa) * (rad + 0.5 * r['welt']),
                     np.full(na, z_welt)], 1)
    return V, F, sd, su, path


TONGUE_UNDER, TONGUE_OVER, TONGUE_CORNER = 0.34, 0.06, 0.40     # (shares of the shin's width: see cut_lace)


def cut_lace(D, faces, f, r, weights, lift_on):
    """The lace-up shaft (SHAFTS['lace']): everything of the upper behind the vamp's seam, cut from the last D (its
    faces) along the fields f (the side, front and vamp-seam fields of build_boots). Two long panels, the inner and
    the outer, each from the back of the leg round to the front and from the sole to the top: the outer laps over the
    inner down the back (D), both lap over the vamp down its seam (A), and at the front they stop short of each other
    by the lacing's opening (L), from the vamp's throat to the top, with a tongue under it (H) that meets the vamp at
    the throat (C). A binding runs along each panel's top (F). Returns the pieces."""
    tl = r['leather']
    half = 0.5 * r['lace_gap']
    # the tongue is sized from the shin: at each height it reaches under each panel by TONGUE_UNDER of the shin's
    # width there (so it is narrow at the ankle and wider at the calf), and its top stands TONGUE_OVER of the shin's
    # width at the boot's top above the panels' edge, square across with its corners rounded by TONGUE_CORNER of
    # its own half width
    pts = D[np.unique(np.concatenate([np.asarray(fc) for fc in faces]))]
    zz = np.arange(pts[:, 5].min(), pts[:, 5].max() + 0.3, 0.3)
    ww = np.array([np.ptp(pts[np.abs(pts[:, 5] - z_) < 0.3, 3]) if (np.abs(pts[:, 5] - z_) < 0.3).any() else 0.0 for z_ in zz])
    ww = np.convolve(np.pad(ww, 2, mode='edge'), np.ones(5) / 5.0, mode='valid')

    def shin(z_):
        return np.interp(z_, zz, ww)

    def tw(A):                                         # (the tongue's half width at each point's height)
        return half + TONGUE_UNDER * shin(A[:, 5])
    side, front, vs, zb = f['side'], f['front'], f['vamp'], f['z_top'] - r['binding']
    lap = r['lap']

    def topz(A):                                       # (the panels' top edge: level at the front, rising to the back)
        return f['z_top'] + r['top_back'] * (1.0 - _ss(-0.6, 0.6, front(A)))
    pieces = []
    for q in ('inner', 'outer'):
        k = -1.0 if q == 'inner' else 1.0
        # (on its side of the middle: short of it by half the opening at the front, past it by a lap at the back
        # for the outer, which laps over the inner there)
        def mine(A, k=k):
            fr = front(A) > 0
            return k * side(A) - np.where(fr, half, -lap if k > 0 else 0.0)
        Dp, fp = _compact(*fit.trim(*fit.trim(*fit.trim(D, faces, mine), lambda A: lap - vs(A)),
                                    lambda A: topz(A) - A[:, 5]))             # (the panel's top edge)
        back = (front(Dp) <= 0)
        d_back = np.where(back, k * side(Dp) + lap, FAR) if k > 0 else np.full(len(Dp), FAR)
        d_a = (lap - vs(Dp)) * f['kc']
        lift = 1.1 * tl * (1.0 + (_ss(4.0 * lap, 0.0, d_back) * (1.0 + _ss(zb - 2.0 * lap, zb, Dp[:, 5])) if k > 0
                                  else 0.0))
        sd = np.minimum(d_back, d_a)
        add_p = lift_on(Dp, lift)
        e = {'A ' + q: np.abs(vs(Dp) - lap) < NEAR, 'D': (np.abs(mine(Dp)) < NEAR) & back,
             'L': (np.abs(mine(Dp)) < NEAR) & ~back, 'E': Dp[:, 5] < f['z_welt'] + NEAR, 'F': Dp[:, 5] > topz(Dp) - NEAR,
             'H': (np.abs(np.abs(side(Dp)) - tw(Dp)) < 0.5 * GRAIN) & ~back & (vs(Dp) < 0)}
        pieces.append(dict(name=q + ' shaft', V=add_p, F=fp, up=(0, 0, 1), lift=lift, edges=e, sd=sd,
                           su=np.where(d_back < d_a, Dp[:, 5], Dp[:, 4]), W=Dp[:, 9:],
                           eyelets=_eyelet_line(Dp, np.abs(side(Dp)), ~back, half + r['eyelet_in'], r)))
        # its binding (BINDING: 0 leaves it off, the panel's top edge bare): the panel's top, over it
        if not r['binding_ring']:
            continue
        Db, fb = _compact(*fit.trim(*fit.trim(*fit.trim(D, faces, mine), lambda A: A[:, 5] - zb),
                                    lambda A: f['z_top'] - A[:, 5]))
        bl = 1.1 * tl * (1.0 + (_ss(4.0 * lap, 0.0, np.where(front(Db) <= 0, side(Db) + lap, FAR)) * 2.0 if k > 0 else 0.0)) + 1.15 * tl
        pieces.append(dict(name=q + ' binding', V=lift_on(Db, bl), F=fb, up=(0, 0, 1), lift=bl,
                           edges={'F': np.abs(Db[:, 5] - zb) < NEAR, 'top': Db[:, 5] > f['z_top'] - NEAR},
                           sd=Db[:, 5] - zb, su=Db[:, 3], W=Db[:, 9:]))
    # the tongue: under the opening, from the vamp's throat to the top
    # (drafted on the same last as the panels and a leather's thickness under them all the way up: it is cut from the
    # last carried on above the boot's top, so it stands above the panels' edge there, and it ends in a round top,
    # high in the middle and cut back at its sides)
    zt = f['z_top'] + min(TONGUE_OVER * float(shin(f['z_top'])), r['tongue_rise'])
    Dt, ft = _compact(*fit.trim(*fit.trim(*fit.trim(*fit.trim(D, faces, lambda A: tw(A) - np.abs(side(A))),
                                                    lambda A: front(A)), lambda A: -vs(A)),
                                lambda A: zt - A[:, 5] - TONGUE_CORNER * tw(A) * (np.abs(side(A)) / tw(A)) ** 6))
    lt = np.zeros(len(Dt))
    Vt = lift_on(Dt, lt)
    pieces.append(dict(name='tongue', V=Vt, F=ft, up=(0, 0, 1), lift=lt,
                       edges={'C': np.abs(vs(Dt)) < NEAR, 'H': np.abs(np.abs(side(Dt)) - tw(Dt)) < NEAR},
                       sd=np.full(len(Dt), FAR), su=np.zeros(len(Dt)), W=Dt[:, 9:]))
    return pieces


# The shafts, by name: each cuts everything of the upper behind the vamp's seam (the foot, which every shaft shares,
# is the toe cap, the vamp and the sole), with the numbers its last is drafted with over the foot's.
SHAFTS = {'lace': dict(cut=cut_lace, ratios={})}


def build_boots(body, M, rig=None, material=None, sole_material=None, name='Boot', ratios=RATIOS, shaft_name='lace',
                log=print):
    """Make one boot for each leg from the measures M (draft.measure_legs) and hand them to the body's bones: the
    foot, and over it the shaft SHAFTS[shaft_name], sewn on at the ankle seam (the foot is cut the same whatever the
    shaft).
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

    z_cut = z_top + r['tongue_rise']                   # (the last carries on above the top, for the tongue)
    region = (P[:, 2] < z_cut + 0.3) & (arm < 0.02)     # (the whole upper is cut from one last)
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
    M['boot_welt'] = z_welt

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
    tables, shafts = {}, {}
    for sgn in (-1, 1):
        leg = (P[:, 0] * sgn > 0) & (arm < 0.02)
        tables[sgn] = shaft(P, leg, M, z_top, r, log=log if sgn < 0 else (lambda *a: None))
        more = SHAFTS[shaft_name]['ratios']
        if more:                                   # (the shaft's own last, over the foot's from the ankle seam up)
            rs = dict(r, **{k: v * H if k not in PLAIN else v for k, v in more.items()})
            zs, C, Rf = tables[sgn]
            _, _, Rs = shaft(P, leg, M, z_top, rs, log=lambda *a: None)
            w = _ss(M['ankle_z'] + r['join'] + r['lap'], M['ankle_z'] + r['join'] + 9.0 * r['lap'], zs)[:, None]
            shafts[sgn] = (zs, C, Rf * (1 - w) + Rs * w)
        else:
            shafts[sgn] = tables[sgn]
        mine = X0[:, 0] * sgn > 0
        w = _ss(M['ankle_z'] - r['join'], M['ankle_z'] + 0.4 * r['join'], z[mine])   # (full width under its edge)
        X[mine] = _on_shaft(X0[mine], shafts[sgn]) * w[:, None] + Xf[mine] * (1 - w)[:, None]
    # room over the instep and the front of the ankle, to flex into: the last above is the leg's own, which would
    # leave the leather no more than the gap off the instep, and a stride (the foot's toe-off) presses the foot up
    # into it (the shaft's table, carried down over the foot, cuts in there)
    X += Nr * (r['instep_room'] * _ss(1.2, 2.8, z) * (1.0 - _ss(M['ankle_z'] + r['join'], M['ankle_z'] + 3.0 * r['join'], z))
               * np.clip(-Nr[:, 1], 0.0, 1.0))[:, None]
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
    D0, f0 = fit.trim(D0, faces, lambda A: z_cut - A[:, 5])                  # (the top)
    D0, f0 = fit.trim(D0, f0, lambda A: A[:, 5] - z_welt)                    # (the sole)
    res = {}
    for side, sgn in (('R', -1), ('L', 1)):
        T = shafts[sgn]
        zs, C, _ = T

        def cx(A):
            return np.interp(A[:, 5], zs, C[:, 0])

        def side_f(A):                                                      # (the outer side: above zero)
            return (A[:, 3] - cx(A)) * sgn

        def front_f(A):                                                     # (in front of the leg's middle: above zero)
            return np.interp(A[:, 5], zs, C[:, 1]) - A[:, 4]
        mine = [f for f in f0 if sgn * D0[f, 0].mean() > 0]
        V, F, SD, SU, WW = [], [], [], [], []

        def add(Vp, Fp, sd, su, Wp):
            n0 = sum(len(v) for v in V)
            V.append(Vp)
            F.extend([[n0 + i for i in f] for f in Fp])
            SD.append(sd)
            SU.append(su)
            WW.append(Wp)

        def lift_on(D, lift):                                               # (a piece's points, lifted off the last)
            nrm = D[:, 6:9] / np.maximum(np.linalg.norm(D[:, 6:9], axis=1, keepdims=True), 1e-9)
            return D[:, 3:6] + nrm * np.broadcast_to(lift, (len(D),))[:, None]
        pieces = []
        used = D0[np.unique(np.concatenate([f for f in mine]))]
        xc = float(C[0, 0])
        # the vamp's seam, where every shaft is sewn to the foot: across the instep at the throat, its middle furthest
        # forward, and back and down from there to the sole at either side of the foot
        y_th, y_so = M['toe_y'] + r['throat'] * L, M['toe_y'] + r['vamp_sole'] * L
        near = (np.abs(used[:, 4] - y_th) < 0.3) & (np.abs(used[:, 3] - xc) < 0.6)
        z_th = float(used[near, 5].max()) if near.any() else M['ankle_z']
        m_v = (y_so - y_th) / max(z_th - z_welt, 1e-3)
        kc = 1.0 / np.sqrt(1.0 + m_v ** 2)

        def vamp_f(A):                                                      # (the vamp's side: above zero)
            return y_th + m_v * (z_th - A[:, 5]) + r['throat_round'] * (A[:, 3] - cx(A)) ** 2 - A[:, 4]
        # the toe cap's seam: an arc across the foot, its middle nearest the toe
        fx = float(np.median(used[used[:, 4] < M['toe_y'] + r['cap_at'] * L, 3])) if len(used) else xc
        half = 0.5 * float(np.ptp(used[np.abs(used[:, 4] - (M['toe_y'] + r['cap_at'] * L)) < 0.2, 3]))
        y_cap = M['toe_y'] + r['cap_at'] * L
        k_cap = r['cap_sweep'] * L / max(half, 1e-3) ** 2

        def toe_seam(A):                                                    # (ahead of the seam: above zero)
            return y_cap + k_cap * (A[:, 3] - fx) ** 2 - A[:, 4]
        tongue_w = 0.5 * r['lace_gap'] + r['tongue']
        # the vamp: ahead of its seam and behind the toe cap's, under everything
        Dv, fv = _compact(*fit.trim(*fit.trim(D0, mine, vamp_f), lambda A: -toe_seam(A)))
        add(Dv[:, 3:6], fv, np.full(len(Dv), FAR), np.zeros(len(Dv)), Dv[:, 9:9 + nw])
        on = np.abs(vamp_f(Dv)) < NEAR
        throat = np.abs(side_f(Dv)) < tongue_w
        pieces.append(dict(name='vamp', V=V[-1], F=fv, up=(0, -1, 0.35), lift=np.zeros(len(Dv)), lasted=True, edges={
            'A inner': on & ~throat & (side_f(Dv) < 0), 'A outer': on & ~throat & (side_f(Dv) > 0), 'C': on & throat,
            'E': np.abs(Dv[:, 5] - z_welt) < NEAR, 'G': np.abs(toe_seam(Dv)) < NEAR}))
        # the toe cap: ahead of its seam, its back edge lapped over the vamp
        Dc, fc = _compact(*fit.trim(D0, mine, lambda A: toe_seam(A) + r['lap']))
        add(lift_on(Dc, 1.1 * tl), fc, toe_seam(Dc) + r['lap'], Dc[:, 3], Dc[:, 9:9 + nw])
        pieces.append(dict(name='toe cap', V=V[-1], F=fc, up=(0, -1, 0.2), lift=np.full(len(Dc), 1.1 * tl), lasted=True,
                           edges={'E': np.abs(Dc[:, 5] - z_welt) < NEAR, 'G': np.abs(toe_seam(Dc) + r['lap']) < NEAR}))
        # the shaft: everything behind the vamp's seam
        f = dict(side=side_f, front=front_f, vamp=vamp_f, kc=kc, z_welt=z_welt, z_top=z_top)
        for p in SHAFTS[shaft_name]['cut'](D0, mine, f, r, lambda G: _weights(Pl, Wl, G), lift_on):
            add(p.pop('V'), p['F'], p.pop('sd'), p.pop('su'), p.pop('W')[:, :nw])
            p['V'] = V[-1]
            pieces.append(p)
        k0 = 0
        for p in pieces:
            p['start'] = k0
            k0 += len(p['V'])
        Vall = np.concatenate(V)
        o = _object('%s.%s' % (name, side), Vall, F, np.concatenate(WW), names, body, rig, material,
                    np.concatenate(SD), np.concatenate(SU), thickness=tl, skive=3.0 * r['lap'])
        res[side] = o
        log('boot: %s, %d points: %s' % (o.name, len(Vall), ', '.join('%s %d' % (p['name'], len(p['V'])) for p in pieces)))
        # the sole, under the upper's cut edge
        edge = Vall                                                     # (the foot's pieces, and a shaft's run to the sole)
        Sv, Sf, sd, su, path = _sole(edge[np.abs(edge[:, 2] - z_welt) < 0.12], z_welt, r)
        so = _object('%s sole.%s' % (name, side), Sv, Sf, _weights(Pl, Wl, Sv), names, body, rig,
                     sole_material or material, sd, su)
        bm = bmesh.new()
        bm.from_mesh(so.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(so.data)
        bm.free()
        so.data.polygons.foreach_set('use_smooth', [False] * len(so.data.polygons))
        res['sole ' + side] = so
        pieces.append(dict(name='sole', V=Sv, F=Sf, up=(0, -1, 0), flat=np.arange(144, 288), edges={}, stitch=path))
        PIECES[side] = pieces
    log('boot: the sole %.2f thick, its under face at z %.2f (%.2f under the foot)' % (r['sole'], M['boot_floor'],
                                                                                   z_foot - M['boot_floor']))
    return res


def reweight(boot, hose, k=4):
    """The boot's bone weights taken from the hose beneath it, at the hose's rest shape: each point of the leather
    takes the weights of the nearest few points of the hose, the nearer the more. The leather then moves as the hose
    does through a stride, wherever the body's own weights and the hose's part ways (the instep and the ankle)."""
    names = [g.name for g in hose.vertex_groups]
    kd = KDTree(len(hose.data.vertices))
    for v in hose.data.vertices:
        kd.insert(hose.matrix_world @ v.co, v.index)
    kd.balance()
    W = np.zeros((len(hose.data.vertices), len(names)), dtype=np.float32)
    for v in hose.data.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    for g in list(boot.vertex_groups):
        boot.vertex_groups.remove(g)
    groups = [boot.vertex_groups.new(name=n) for n in names]
    for v in boot.data.vertices:
        hits = kd.find_n(boot.matrix_world @ v.co, k)
        w = np.array([1.0 / max(h[2], 1e-4) for h in hits])
        ws = (W[[h[1] for h in hits]] * (w / w.sum())[:, None]).sum(0)
        for gi in np.nonzero(ws > 1e-4)[0]:
            groups[gi].add([v.index], float(ws[gi]), 'REPLACE')


def relieve(boot, hose, frames, margin=0.08, radius=0.8, log=print):
    """Let the leather out where a stride presses the hose through it. At each of the frames the hose points that are
    through the leather are found (clearance); the boot's points within radius of where each lies at rest are pushed
    out along their own normals by how far it was through, plus the margin, less with distance. The boot is the
    object's rest shape, so the leather is as it was cut, with room where the foot needs it."""
    scene = bpy.context.scene
    me = boot.data
    n = len(me.vertices)
    rest = np.empty(n * 3)
    me.vertices.foreach_get('co', rest)
    rest = rest.reshape(-1, 3)
    nrm = np.empty(n * 3)
    me.vertices.foreach_get('normal', nrm)
    nrm = nrm.reshape(-1, 3)
    kd = KDTree(n)
    for i, q in enumerate(rest):
        kd.insert(q, i)
    kd.balance()
    push = np.zeros(n)
    worst = 0.0
    for f in frames:
        scene.frame_set(f)
        pts = clearance(boot, hose, log=lambda *a: None)['points']
        for q, depth in pts:
            worst = max(worst, depth)
            for _, i, d in kd.find_range(q, radius):
                push[i] = max(push[i], (depth + margin) * (1.0 - d / radius))
    scene.frame_set(0)
    out = rest + nrm * push[:, None]
    me.vertices.foreach_set('co', out.ravel())
    if 'rest' in me.attributes:
        me.attributes['rest'].data.foreach_set('vector', out.ravel())
    me.update()
    log('boot relief %s: the stride pressed the hose up to %.2f through the leather; %d points let out by %.2f at most'
        % (boot.name, worst, int((push > 0).sum()), push.max()))
    return worst


def clearance(boot, hose, log=print):
    """The gap between the hose and the boot's inner face, at the current frame: from every point of the hose under
    the boot (judged by where it lies at rest: on the boot's own leg, between the sole and the top), a ray straight
    out from the hose finds the leather. Met from inside, the gap is how far; met from outside (the face turned
    toward the hose's point, so the point is on the face's outer side), the hose is through the leather there, by
    as much. Both are measured on their own surface, without thickness. (The nearest face alone misjudges where
    the leather stands well off the leg, as the shaft does at the ankle: there the nearest thing is another piece's
    cut edge, and which side of it the hose is on means nothing.)"""
    def surf(o):
        for m in o.modifiers:
            if m.type == 'SOLIDIFY':
                m.show_viewport = False
        e = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = e.to_mesh()
        me.calc_loop_triangles()
        pts = [o.matrix_world @ v.co for v in me.vertices]
        nrm = [(o.matrix_world.to_3x3() @ v.normal).normalized() for v in me.vertices]
        tris = [tuple(t.vertices) for t in me.loop_triangles]
        rest = np.zeros(len(me.vertices) * 3)
        me.attributes['rest'].data.foreach_get('vector', rest)
        e.to_mesh_clear()
        for m in o.modifiers:
            if m.type == 'SOLIDIFY':
                m.show_viewport = True
        return pts, nrm, tris, rest.reshape(-1, 3)
    bp, _, bt, br = surf(boot)
    tree = BVHTree.FromPolygons(bp, bt)
    hp, hn, _, hr = surf(hose)
    side = -1 if boot.name.endswith('.R') else 1
    z_top, z_low = float(br[:, 2].max()) - 0.3, float(br[:, 2].min()) + 0.3
    gaps = []
    where, through = [], []
    for p, nv, q in zip(hp, hn, hr):
        if q[0] * side <= 0 or q[2] > z_top or q[2] < z_low:
            continue
        loc, fn, ti, d = tree.ray_cast(p - nv * 1e-4, nv, 3.0)
        if loc is None:
            continue
        gaps.append(d if fn.dot(nv) > 0 else -d)
        if fn.dot(nv) <= 0:
            where.append((p[2], q[2], q[1], bt[ti][0]))
            through.append((q.copy(), d))                   # (how far through)      # (where it is through: posed height, rest height, rest y, the vertex it met)
    g = np.array(gaps)
    out = {'min': float(g.min()), 'mean': float(g.mean()), 'through': int((g < 0).sum()), 'n': len(g), 'points': through}
    log('boot clearance %s: hose to leather min %.3f mean %.3f, %d of %d hose points through the leather' % (
        boot.name, out['min'], out['mean'], out['through'], out['n']))
    if where:
        w = np.array(where)
        sides = PIECES.get(boot.name.rsplit('.', 1)[-1], [])
        names = [next((pc['name'] for pc in sides if pc['name'] != 'sole' and pc['start'] <= int(v) < pc['start'] + len(pc['V'])), '?') for v in w[:, 3]]
        log('boot clearance %s: met on the pieces %s' % (boot.name, {k: names.count(k) for k in sorted(set(names))}))
        log('boot clearance %s: through at rest heights %.2f to %.2f (the ankle is at 4.9), rest y %.2f to %.2f'
            % (boot.name, w[:, 1].min(), w[:, 1].max(), w[:, 2].min(), w[:, 2].max()))
    return out
