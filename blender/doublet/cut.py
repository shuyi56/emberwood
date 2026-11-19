"""An arming doublet, drafted on whatever body it is given: the padded, quilted, close-fitting jacket worn under
plate, which the harness is tied to.

It is the vest's own draft (vest/fit.py: the body measured, one whole piece wrapped taut round it, the edges cut
along measured lines), let out for the padding and the shirt beneath, and cut as an arming doublet is:

  the body      close to the chest and waist, and on below the waist seam as a short skirt flaring over the hips
  the collar    a low stand, open in front
  the front     the two edges meet down the middle and are laced shut through worked eyelets, pit of the neck to hem
  the quilting  the padding is stitched down in upright channels (level rows round the collar): each channel a
                pillow of the padding's depth between two lines of stitching, flat where a seam crosses it (the
                waist, the collar's foot) and along every edge, which is bound
  the points    pairs of eyelets with a waxed lace through them, tied, its ends tipped with aiglets: on each
                shoulder for the arm's harness and round the skirt for the leg's

Every size is a share of w, the neck's half-width, as in the vest; the number of channels comes from the bust's
girth, so they keep their width on a larger body."""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vest'))
import fit  # noqa: E402

RATIOS = dict(
    fit.RATIOS,
    ease=0.30,               # the gap between body and doublet, in w: room for the shirt, and a padded coat is
                             # not worn tight (at 0.16 it sat like a corset)
    thickness=0.06,          # the cloth's own thickness (shell and lining), in w
    pad=0.13,                # the padding's depth at a channel's crown, in w
    channel=0.46,            # a channel's width at the bust, in w (about 2.4 cm)
    collar_top=0.50,         # a low stand: this share of the neck's length above the pit of the neck
    collar_rise=0.08,
    collar_gap=0.20,
    collar_pad=0.35,         # the collar's padding, as a share of the body's
    collar_rows=3,           # the rows of stitching round the collar
    v_point=1.0,             # the collar stands open from the pit of the neck up
    v_top=0.34,              # by this much each side at its top edge, in w
    v_bow=1.0,
    corner=0.12,
    hem_drop=2.1,            # the skirt's hem below the waist, in w
    hem_point=0.0,           # (level all round)
    span=0.60,               # how much of the waist's hollow the padded cloth spans instead of following:
                             # this share of the bust's half-width over the waist's
    lay=0.16,                # the shoulder's end lies down onto the shoulder by this much, in w, so the
                             # looser coat does not stand out over the arm as a wing
    flare=0.22,              # how far the skirt stands out at the hem beyond the hips, in w
    shoulder=0.86,           # the shoulder ends on the shoulder's flat top, short of where it rolls into the arm
    scye_drop=0.38,          # a deep armhole: the arm must lift a sword
    border=0.30,             # the flat bound border along every edge, in w
    bind=0.030,              # how far the binding stands proud, in w
    seam=0.13,               # the half-width of the flat a seam leaves across the channels, in w
    eyelet=0.060,            # a worked eyelet's radius, in w
    cord=0.026,              # the lace's radius, in w
    lace_in=0.26,            # the front eyelets' distance from the edge, in w
    lace_step=0.62,          # and their spacing down the front, in w
    point_gap=0.28,          # the two eyelets of an arming point this far apart, in w
    point_tail=1.25,         # the length of a tied point's hanging ends, in w
    aiglet=0.30,             # the length of the metal tip on a lace's end, in w
    point_drop=0.45,         # the skirt's points this far below the waist seam, in w
)
SKIRT_POINTS = (32.0, 96.0, 150.0)       # where the skirt's points are, in degrees round from the front, each side


def _neckline(M):
    """The neckline's height round the body (fit.whole_piece's own line: pit, side neck points, back neck point)."""
    a = 0.25 * (M['z_pit'] + M['z_back']) + 0.5 * M['z_side']
    c, b = a - M['z_side'], 0.5 * (M['z_pit'] - M['z_back'])
    return lambda ang: a + b * np.cos(ang) + c * np.cos(2 * ang)


def quilting(M, lines, n, z_top, ratios):
    """How far each point of the cut shell is raised: the padding in its channels, flat at the seams and the bound
    border, the binding's own roll, and the skirt's flare."""
    w, yc = M['w'], M['yc']
    neck = _neckline(M)
    z_hem = M['z_waist'] - ratios['hem_drop'] * w
    row = (float(z_top.mean()) - float(neck(np.array([math.pi / 2]))[0])) / ratios['collar_rows']

    arm = dict(lines)['armholes']

    def raised(p):
        ang = np.arctan2(-p[:, 0], -(p[:, 1] - yc))
        edge = np.min([f(p) for _, f in lines], axis=0)
        above = p[:, 2] - neck(ang)
        pillow = np.sin(math.pi * ((ang * n / (2 * math.pi)) % 1.0)) ** 0.55
        rows = np.abs(np.sin(math.pi * above / max(row, 1e-6))) ** 0.55
        on_collar = fit._ss(-0.05 * w, 0.12 * w, above)
        # (the collar is thinly padded: a full pillow there crowds the hollow where the stand rises off the shoulder)
        pad = pillow * (1.0 - on_collar) + ratios['collar_pad'] * rows * on_collar
        seams = fit._ss(0.0, ratios['seam'] * w, np.abs(p[:, 2] - M['z_waist'])) * fit._ss(0.0, 2.5 * ratios['seam'] * w, np.abs(above))
        inside = fit._ss(0.5 * ratios['border'] * w, ratios['border'] * w, edge)
        bind = ratios['bind'] * w * np.exp(-((edge - 0.2 * ratios['border'] * w) / (0.22 * ratios['border'] * w)) ** 2)
        # (a padded coat hangs from the chest and the hips and bridges the waist between them: it is not cinched)
        span = ratios['span'] * max(M['bust_half'] - M['waist_half'], 0.0) * np.exp(-((p[:, 2] - M['z_waist']) / (1.3 * w)) ** 2)
        lay = ratios['lay'] * w * np.exp(-(np.clip(arm(p), 0.0, None) / (0.7 * w)) ** 2) * fit._ss(M['z_pit_arm'], M['z_sh'], p[:, 2])
        skirt = ratios['flare'] * w * np.clip((M['z_waist'] - p[:, 2]) / (M['z_waist'] - z_hem), 0.0, 1.0) ** 1.5
        return ratios['pad'] * w * pad * seams * inside + bind + skirt + span - lay
    return raised


class Surface:
    """The finished doublet's outer face, to set the trimmings on."""

    def __init__(self, o, M, thick):
        me = o.data
        self.tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        self.M, self.thick = M, thick

    def _out(self, loc, nrm):
        """A face's normal turned to the outside (away from the neck's axis, and up)."""
        ref = Vector((loc.x, loc.y - self.M['yc'], 0.6 * abs(nrm.z) + 0.05))
        return nrm if nrm.dot(ref) >= 0 else -nrm

    def at(self, ang, z):
        """The point of the outer face at an angle round from the front and a height, and the way out there."""
        o = Vector((-40.0 * math.sin(ang), self.M['yc'] - 40.0 * math.cos(ang), z))
        return self.cast(o, Vector((math.sin(ang), math.cos(ang), 0.0)))

    def cast(self, origin, direction):
        loc, nrm, _, _ = self.tree.ray_cast(origin, direction)
        if loc is None:
            return None, None
        nrm = self._out(loc, nrm)
        return loc + nrm * self.thick, nrm

    def press(self, p, r):
        """p kept at least r clear of the outer face."""
        loc, nrm, _, _ = self.tree.find_nearest(p)
        nrm = self._out(loc, nrm)
        d = (p - loc).dot(nrm)
        return p if d >= self.thick + r else p + nrm * (self.thick + r - d)


def _frame(d):
    d = d.normalized()
    a = d.cross(Vector((0.0, 0.0, 1.0)) if abs(d.z) < 0.9 else Vector((1.0, 0.0, 0.0))).normalized()
    return a, d.cross(a)


def tube(pts, radius, sides=7):
    """A round cord along the points (radius: one, or one for each point), closed at both ends."""
    pts = [Vector(p) for p in pts]
    rad = [radius] * len(pts) if np.isscalar(radius) else list(radius)
    V, F = [], []
    a, b = _frame(pts[1] - pts[0])
    for i, p in enumerate(pts):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        a = (a - d * a.dot(d)).normalized()             # (the ring carried along the cord without twisting)
        b = d.cross(a)
        V += [p + rad[i] * (math.cos(t) * a + math.sin(t) * b) for t in (2 * math.pi * k / sides for k in range(sides))]
        if i:
            F += [((i - 1) * sides + k, (i - 1) * sides + (k + 1) % sides, i * sides + (k + 1) % sides, i * sides + k) for k in range(sides)]
    F.append(tuple(range(sides - 1, -1, -1)))
    F.append(tuple((len(pts) - 1) * sides + k for k in range(sides)))
    return np.array([v[:] for v in V]), [list(f) for f in F]


def ring(centre, normal, R, r, around=12, sides=6):
    """A worked eyelet: a ring of thread lying on the cloth."""
    a, b = _frame(normal)
    n = normal.normalized()
    V, F = [], []
    for i in range(around):
        t = 2 * math.pi * i / around
        out = math.cos(t) * a + math.sin(t) * b
        for k in range(sides):
            s = 2 * math.pi * k / sides
            V.append(centre + out * (R + r * math.cos(s)) + n * (0.6 * r * math.sin(s)))
            F.append((i * sides + k, ((i + 1) % around) * sides + k, ((i + 1) % around) * sides + (k + 1) % sides, i * sides + (k + 1) % sides))
    return np.array([v[:] for v in V]), [list(f) for f in F]


def _over(S, a, b, r, steps=9, lift=0.0):
    """A lace's run from a to b lying on the cloth (lift: raised by that much at its middle, to cross another)."""
    out = []
    for k in range(steps + 1):
        t = k / steps
        p = S.press(a.lerp(b, t), r)
        loc, nrm, _, _ = S.tree.find_nearest(p)
        out.append(p + S._out(loc, nrm) * (lift * math.sin(math.pi * t)))
    return out


def _hang(S, start, lead, length, r, w):
    """A lace's loose end from start: led off along `lead`, then falling, lying on the cloth where it meets it."""
    pts, p, s, ds = [Vector(start)], Vector(start), 0.0, 0.07
    while s < length:
        k = math.exp(-s / (0.35 * w))
        d = (lead * k + Vector((0.0, 0.0, -1.0)) * (1.0 - k)).normalized()
        p = S.press(p + d * ds, r)
        pts.append(p.copy())
        s += ds
    return pts


def trimmings(S, M, ratios, log=print):
    """The front's lacing and the arming points: (thread pieces, lace pieces, metal pieces), each [(points, faces)]."""
    w, yc = M['w'], M['yc']
    r, R = ratios['cord'] * w, ratios['eyelet'] * w
    thread, lace, metal = [], [], []

    def eyelet(p, n):
        thread.append(ring(p + n * (0.4 * r), n, R, 0.45 * R))

    def end(pts):
        """A hanging end: the lace, and the aiglet on its last stretch."""
        k = max(2, int(round(ratios['aiglet'] * w / 0.07)))
        lace.append(tube(pts[:-k + 1], r))
        tip = pts[-k:]
        metal.append(tube(tip, [1.25 * r * (1.0 - 0.55 * i / (len(tip) - 1)) for i in range(len(tip))]))

    # the front: pairs of eyelets from the hem's border to the pit of the neck, laced across and across
    z0 = M['z_waist'] - ratios['hem_drop'] * w + ratios['border'] * w + 0.3 * w
    z1 = M['z_pit'] - 0.25 * w
    n = max(2, int(round((z1 - z0) / (ratios['lace_step'] * w))) + 1)
    pairs = []
    for z in np.linspace(z0, z1, n):
        side = []
        for sx in (-1.0, 1.0):
            p, nr = S.cast(Vector((sx * ratios['lace_in'] * w, yc - 40.0, float(z))), Vector((0.0, 1.0, 0.0)))
            if p is None:
                break
            eyelet(p, nr)
            side.append(p + nr * r)
        if len(side) == 2:
            pairs.append(side)
    for i in range(len(pairs) - 1):
        (l0, r0), (l1, r1) = pairs[i], pairs[i + 1]
        lace.append(tube(_over(S, l0, r1, r), r))
        lace.append(tube(_over(S, r0, l1, r, lift=2.2 * r), r))
    if pairs:
        l, rr = pairs[-1]
        lace.append(tube(_over(S, l, rr, r), r))                    # (tied across the top pair, the ends left hanging)
        mid = l.lerp(rr, 0.5)
        for sx in (-1.0, 1.0):
            end(_hang(S, mid, Vector((0.9 * sx, -0.3, 0.2)), (0.8 + 0.1 * sx) * ratios['point_tail'] * w, r, w))
    log('doublet: laced through %d pairs of eyelets down the front' % len(pairs))

    def point(c, nr, along, lead):
        """An arming point at c: two eyelets `along` the cloth, the lace tied between them, its ends led off."""
        along = (along - nr * along.dot(nr)).normalized()
        half = 0.5 * ratios['point_gap'] * w
        a, b = c + along * half, c - along * half
        for p in (a, b):
            eyelet(S.press(p, 0.0), nr)
        knot = c + nr * (2.0 * r)
        lace.append(tube([S.press(a, r), knot + along * (0.3 * half), knot - along * (0.3 * half), S.press(b, r)], [r, 1.7 * r, 1.7 * r, r]))
        for sgn, ln in ((1.0, 1.0), (-1.0, 0.85)):
            end(_hang(S, knot, (lead + along * (0.7 * sgn)).normalized(), ln * ratios['point_tail'] * w, r, w))

    made = 0
    z_pt = M['z_waist'] - ratios['point_drop'] * w
    for deg in SKIRT_POINTS:
        for sgn in (-1.0, 1.0):
            ang = sgn * math.radians(deg)
            c, nr = S.at(ang, z_pt)
            if c is not None:
                point(c, nr, Vector((math.cos(ang), -math.sin(ang), 0.0)), Vector((0.0, 0.0, -1.0)))
                made += 1
    y_sh = 0.5 * (M['scye_y'][0] + M['scye_y'][1])
    for sgn in (-1.0, 1.0):
        c, nr = S.cast(Vector((sgn * 0.78 * ratios['shoulder'] * M['x_sh'], y_sh, M['z_sh'] + 20.0)), Vector((0.0, 0.0, -1.0)))
        if c is not None:
            point(c, nr, Vector((0.0, 1.0, 0.0)), Vector((0.5 * sgn, -0.3, -0.4)))
            made += 1
    log('doublet: %d arming points tied' % made)
    return thread, lace, metal


def _cloth_uv(o, M, R0):
    """Each point's place on the flat cloth, in units: round the body at the bust's girth, by height."""
    me = o.data
    lay = me.uv_layers.new(name='UVMap')
    co = np.array([v.co[:] for v in me.vertices])
    ang = np.arctan2(-co[:, 0], -(co[:, 1] - M['yc']))
    for p in me.polygons:
        a = ang[list(p.vertices)]
        if np.ptp(a) > math.pi:                     # (a face across the back's middle: carried round)
            a = np.where(a < 0, a + 2 * math.pi, a)
        for li, a_, vi in zip(p.loop_indices, a, p.vertices):
            lay.data[li].uv = (float(a_) * R0, float(co[vi, 2]))


def build_doublet(body, head_objs=(), rig=None, cloth=None, thread=None, lace=None, metal=None, name='Doublet',
                  ratios=RATIOS, log=print):
    """Measure the body, make the whole piece, cut it, quilt it, and trim it. Returns (object, measures, covered: how far inside the
    doublet's edges a point lies, for whatever is worn under it)."""
    P, T, aw = fit.body_arrays(body)
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
    M = fit.measure(P, aw, hp, ratios)
    w = M['w']
    G, _, z_top, body_r, zs = fit.whole_piece(P, T, aw, M, hb, ratios, log)
    rows, cols = G.shape[:2]
    V = G.reshape(-1, 3)
    idx = np.arange(rows * cols).reshape(rows, cols)
    faces = [list(q) for q in np.stack([idx[:-1], np.roll(idx[:-1], -1, 1), np.roll(idx[1:], -1, 1), idx[1:]], 2).reshape(-1, 4)]
    lines, extra = fit.trim_lines(M, z_top, fit.scye(P, aw, M, ratios)[0], ratios)
    M.update(extra)
    for nm, f in lines:
        V, faces = fit.trim(V, faces, f)
        log('doublet: trimmed %s, %d faces left' % (nm, len(faces)))
    # the channels: as many as the bust's girth holds at their width, an even number so one line of stitching runs
    # down the middle of the back and the front's edges meet on another
    girth = float(body_r[int(np.argmin(np.abs(zs - M['z_bust'])))].mean()) * 2 * math.pi
    n = 2 * max(4, int(round(girth / (2 * ratios['channel'] * w))))
    log('doublet: bust girth %.1f, quilted in %d channels' % (girth, n))
    # (it follows the trunk's skin only: a point at the waist's side lies nearer the hanging arm's elbow than the
    # flank, and would be carried off by the arm)
    trunk = np.nonzero(aw < 0.5)[0]
    kd = KDTree(len(trunk))
    for i in trunk:
        kd.insert(Vector(P[i]), int(i))
    kd.balance()
    th = ratios['thickness'] * w
    o = fit._sheet(name, [(V, faces, quilting(M, lines, n, z_top, ratios))], M, cloth, body, kd, rig, lift=0.0, thickness=th)
    _cloth_uv(o, M, girth / (2 * math.pi))
    M['z_hem'] = M['z_waist'] - ratios['hem_drop'] * w
    if lace is not None:
        t_, l_, m_ = trimmings(Surface(o, M, th), M, ratios, log)
        for nm, pieces, mat in (('eyelets', t_, thread or lace), ('laces', l_, lace), ('aiglets', m_, metal or lace)):
            if pieces:
                fit._sheet('%s %s' % (name, nm), pieces, M, mat, body, kd, rig, lift=0.0, thickness=0.0)
    def covered(p):
        """How far inside the doublet's edges a point of the body lies (negative where it is not covered). The
        front's two edges are laced shut below the collar's opening, so they are no edge here."""
        out = np.minimum(lines[0][1](p), lines[1][1](p))
        return np.minimum(out, np.where(p[:, 2] < M['z_v'], 1e3, lines[2][1](p)))
    return o, M, covered
