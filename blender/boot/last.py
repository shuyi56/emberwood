"""A tall leather boot drafted on whatever legs it is given: from the sole to just under the knee, over the hose.

Nothing here is placed by hand. measure_legs() (the hose's own tailor's tape, blender/hose/draft.py) reads the heel,
toe, ankle, calf and knee off the skin, and every size of the boot is a share of the head's height H, the figure's
unit, so the boot re-fits when the body or the head changes.

Leather is not hose. It does not stretch and it does not follow the leg: a boot's shaft is a stiff tube that stands
off the calf, falls nearly straight past the ankle (the foot has to go in through it) and creases where the ankle
bends. So the boot is made in five steps:

  1. the shaft     level sections of the leg, each pulled taut (its convex hull), then a leather tube round them: it
                   may close in on the leg only as fast as a stiff hide can (SLOPE: so it runs straight from the calf
                   down past the ankle's hollow instead of curving into it), it is never narrower at the ankle than
                   the heel needs to pass (ankle_girth), it stands off by the ease and flares at the collar.
  2. the foot      the skin of the foot let out by the ease: a toe box (the toe region carried a little forward from
                   the ball of the foot, and raised), a welt round the sole's edge, the sole's thickness under it.
                   The shaft and the foot meet over the ankle.
  3. the trims     the top edge, level, a little under the knee: the line the greave overlaps (M['boot_top']).
  4. the rig       the skin's own bone weights, the same armature settings, so the boot bends as the leg bends.
  5. the creases   the leather is cut a little longer over the ankle than the place it is held in (slack), and the
                   cloth solver (blender/chemise/solver.py), given leather's numbers (LEATHER: it hardly stretches
                   and bends hard), lets the shaft take up that length. Held at the collar and the foot, it buckles
                   into a few broad, rounded creases round the ankle, as a worn boot does, and lies on the hose.

The sole is a thickness added under the foot, so the figure stands on it: the figure's floor is the sole's under
face, SOLE below the foot's."""
import os
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('hose', 'vest', 'chemise'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import draft  # noqa: E402   (the hose's: the tape, the body's data)
import fit  # noqa: E402     (the vest's cutter and hull)
import solver  # noqa: E402  (the chemise's cloth solver)

# Every size as a share of the head's height H (but the plain ratios marked so).
RATIOS = {
    'leather': 0.014,        # the leather's thickness (about 3.5 mm)
    'ease': 0.045,           # the shaft's room over the leg's taut section (a finger's width all round)
    'slope': 0.06,           # the most the shaft closes in on the leg for each unit it goes down (a plain ratio)
    'ankle_girth': 1.42,     # the shaft's least girth, as a multiple of the ankle's (the heel goes in through it)
    'flare': 0.025,          # the collar's flare at the top
    'flare_span': 0.30,
    'room': 0.022,           # the foot's room over the hose
    'gap': 0.024,            # the least room between the skin and the leather anywhere (the hose is 0.0105 of it)
    'toe_room': 0.045,       # the toe box's height over the toe
    'toe_long': 0.070,       # how much longer the toe region is than the foot's (a plain share of its length)
    'welt': 0.030,           # the welt's width round the sole
    'sole': 0.026,           # the sole's thickness under the foot
    'bridge': 0.16,          # the most the leather stands off a hollow of the foot it bridges
    'top_below_knee': 0.12,  # the top edge below the knee's joint
    'bead': 0.05,            # the rolled top edge's depth
    'join': 0.16,            # the span over the ankle where the shaft meets the foot
    'slack': 0.07,           # how much longer the leather is cut over the ankle than it is held (a plain ratio)
    'slack_at': 0.22,        # the slack's middle, above the ankle's joint
    'slack_span': 0.30,      # and its half span
    'slack_front': 0.6,      # how much more slack the front (the instep's bend) takes than the back (a plain ratio)
}
PLAIN = ('slope', 'ankle_girth', 'toe_long', 'slack', 'slack_front')

# Leather for the solver, at the figure's scale (63 units to 1.65 m, the solver taking a unit for a metre): against
# linen (blender/chemise/solver.py LINEN) it is heavier, stretches a third as much and bends twenty times as hard, so
# it takes up slack in two or three broad rounded creases, not in many small folds.
LEATHER = dict(
    mass=0.6,
    tension=400.0,
    compression=400.0,
    shear=150.0,
    bending=20.0,
    damping=(12.0, 12.0, 10.0, 6.0),
    air=1.0,
    friction=4.0,            # leather on wool
    gap=0.03,
    self_gap=0.015,
    quality=10,
    frames=60,
)
SETTLE = False               # True: let the solver crease the leather (only after the drafted boot has been seen: a new layer is shown first without the solver)
ROW = 0.15                   # the shaft's sections, this far apart
COLS = 96                    # and this many directions round each


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
    """Step 1: the shaft as a table of radii [section, direction] round a centre line, from the ankle's joint to over
    the top. Returns (zs, centres [section, 2], radii [section, direction])."""
    zs = np.arange(M['ankle_z'], z_top + 1.0, ROW)
    th = np.linspace(-np.pi, np.pi, COLS, endpoint=False)
    C, R = np.zeros((len(zs), 2)), np.zeros((len(zs), COLS))
    for i, z in enumerate(zs):
        q = P[leg & (np.abs(P[:, 2] - z) < 0.6 * ROW)][:, :2]
        h = np.array(fit._hull(q))
        C[i] = h.mean(0)
    C = fit._blur(C, 4.0, 0, False)
    for i, z in enumerate(zs):
        q = P[leg & (np.abs(P[:, 2] - z) < 0.6 * ROW)][:, :2]
        R[i] = _radial(np.array(fit._hull(q - C[i])), th)
    taut = R.copy()
    # the hide closes in on the leg no faster than SLOPE: each section stands at least as far out as every other one,
    # less the slope over the height between them
    dz = np.abs(zs[:, None] - zs[None, :])
    R = (R[None, :, :] - r['slope'] * dz[:, :, None]).max(1)
    R = R + r['ease'] + r['flare'] * _ss(z_top - r['flare_span'], z_top, zs)[:, None] ** 2
    # never narrower than the heel needs to pass
    girth = (R * (2 * np.pi / COLS)).sum(1)
    R *= np.maximum(1.0, r['ankle_girth'] * M['ankle'] / girth)[:, None]
    R = fit._blur(fit._blur(R, 2.0, 1, True), 1.5, 0, False)
    g = (R * (2 * np.pi / COLS)).sum(1)
    cm = 165.0 / 63.0
    for z0, lab in ((M['ankle_min_z'], 'ankle'), (M['calf_z'], 'calf'), (z_top, 'top')):
        i = int(np.argmin(np.abs(zs - z0)))
        tg = (taut[i] * (2 * np.pi / COLS)).sum()
        log('boot: the shaft at the %s (z %.2f) is %.0f cm round, over a leg of %.0f cm: it stands off %.2f on average'
            % (lab, zs[i], g[i] * cm, tg * cm, (R[i] - taut[i]).mean()))
    return zs, C, R, th


def _on_shaft(X, zs, C, R):
    """Each point carried out along its direction from the centre line onto the shaft."""
    zi = np.clip((X[:, 2] - zs[0]) / ROW, 0, len(zs) - 1.001)
    i0 = zi.astype(int)
    f = zi - i0
    c = C[i0] * (1 - f)[:, None] + C[i0 + 1] * f[:, None]
    d = X[:, :2] - c
    a = np.arctan2(d[:, 0], -d[:, 1])
    ai = (a + np.pi) / (2 * np.pi) * COLS
    j0 = np.floor(ai).astype(int) % COLS
    j1 = (j0 + 1) % COLS
    g = ai - np.floor(ai)
    rad = ((R[i0, j0] * (1 - g) + R[i0, j1] * g) * (1 - f)
           + (R[i0 + 1, j0] * (1 - g) + R[i0 + 1, j1] * g) * f)
    out = X.copy()
    u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    out[:, :2] = c + u * rad[:, None]
    return out, a


def build_boots(body, M, rig=None, material=None, name='Boot', ratios=RATIOS, colliders=(), settle=None, log=print):
    """Draft one boot for each leg from the measures M (draft.measure_legs), hand them to the body's bones and (with
    settle, SETTLE by default) let the solver crease them on the colliders (the hose). Returns {'R': object, 'L':
    object}. M gains 'boot_top' (the edge's height) for whatever goes over it."""
    settle = SETTLE if settle is None else settle
    B = draft.body_data(body)
    P, Nn, W, names = B['P'], B['N'], B['W'], B['groups']
    H = M['H']
    r = {k: v * H if k not in PLAIN else v for k, v in ratios.items()}
    col = {n: i for i, n in enumerate(names)}
    arm = sum(W[:, col[n]] for n in names if n.split('.')[0] in draft.ARM_GROUPS)
    z_top = M['knee_z'] - r['top_below_knee']
    M['boot_top'] = z_top
    L = M['heel_y'] - M['toe_y']
    y_ball = M['toe_y'] + 0.30 * L                     # the ball of the foot: 70 percent of the way from the heel

    region = (P[:, 2] < z_top + 0.9) & (arm < 0.02)
    idx = np.nonzero(region)[0]
    remap = -np.ones(len(P), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    faces = [[int(remap[i]) for i in f] for f in B['faces'] if all(region[i] for i in f)]
    X0, Nr, Wr = P[idx], Nn[idx], W[idx]
    E = remap[B['E'][region[B['E'][:, 0]] & region[B['E'][:, 1]]]]
    n = len(X0)
    z, y = X0[:, 2], X0[:, 1]

    # ---- 2. the foot: the room, the toe box, the welt, the sole
    ease = np.full(n, r['room'])
    toe = _ss(y_ball, M['toe_y'], y) * (z < M['ankle_z'])
    ease += r['toe_room'] * toe * np.clip(Nr[:, 2], 0.0, 1.0)             # (the toe box stands up over the toes)
    ease += r['welt'] * _ss(0.9, 0.25, z) * (np.abs(Nr[:, 2]) < 0.8)         # (the welt: the sole's edge stands out)
    ease *= 1.0 - 0.75 * np.clip(-Nr[:, 2], 0.0, 1.0) ** 2 * _ss(1.4, 0.3, z)       # (under the foot the sole is the room)
    Xf = X0 + Nr * ease[:, None]
    fwd = Xf[:, 1] < y_ball
    Xf[fwd, 1] = y_ball + (Xf[fwd, 1] - y_ball) * (1.0 + r['toe_long'])
    down = np.clip(-Nr[:, 2], 0.0, 1.0) ** 2 * _ss(1.4, 0.3, z)
    Xf[:, 2] -= r['sole'] * down

    # ---- 1. the shaft, for each leg, and where it meets the foot
    X = Xf.copy()
    ang = np.zeros(n)
    for sgn in (-1, 1):
        leg = (P[:, 0] * sgn > 0) & (arm < 0.02)
        zs, C, R, th = shaft(P, leg, M, z_top, r, log=log)
        mine = X0[:, 0] * sgn > 0
        Xs, a = _on_shaft(X0[mine], zs, C, R)
        w = _ss(M['ankle_z'] - 0.5 * r['join'], M['ankle_z'] + r['join'], z[mine])
        X[mine] = Xs * w[:, None] + Xf[mine] * (1 - w)[:, None]
        ang[mine] = a
    # the foot's hollows bridged (the shaft is convex already)
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
    log('boot: the last, %d points; %d bridged over the foot\'s hollows (by %.3f at most), %d put out to the gap'
        % (n, int((lift > 0.01).sum()), lift.max(), pushed))

    # the leather as cut: longer over the ankle than where it is held, more so in front (the instep's bend)
    front = np.clip(np.cos(ang), 0.0, 1.0) ** 2
    zc = M['ankle_z'] + r['slack_at']
    Xc = X.copy()
    Xc[:, 2] += (r['slack'] * r['slack_span'] * (1.0 + r['slack_front'] * front)
                 * 2.0 * _ss(zc - r['slack_span'], zc + r['slack_span'], X[:, 2]))

    # ---- 3. the trims
    D = np.concatenate([X0, X, Nr, Wr, Xc], 1)
    D, fs = fit.trim(D, faces, lambda A: z_top - A[:, 2])
    nw = len(names)
    Vs, V, Nr, Wr, Vc = D[:, 0:3], D[:, 3:6], D[:, 6:9], D[:, 9:9 + nw], D[:, 9 + nw:12 + nw]
    Nr = Nr / np.maximum(np.linalg.norm(Nr, axis=1, keepdims=True), 1e-9)
    log('boot: cut the top edge at %.2f (%.2f below the knee), %d faces left' % (z_top, r['top_below_knee'], len(fs)))
    top_d = z_top - Vs[:, 2]
    bead = (_ss(r['bead'] * 2.0, 0.0, top_d) * 0.7 * r['leather'])[:, None]
    V, Vc = V + Nr * bead, Vc + Nr * bead            # (the rolled edge stands proud)

    out = {}
    for side, sgn in (('R', -1), ('L', 1)):
        keep = [f for f in fs if sgn * Vs[f, 0].mean() > 0]
        used = sorted({i for f in keep for i in f})
        km = -np.ones(len(V), dtype=np.int64)
        km[used] = np.arange(len(used))
        me = bpy.data.meshes.new('%s.%s' % (name, side))
        me.from_pydata(V[used].tolist(), [], [[int(km[i]) for i in f] for f in keep])
        me.validate()
        for p in me.polygons:
            p.use_smooth = True
        o = bpy.data.objects.new('%s.%s' % (name, side), me)
        bpy.context.scene.collection.objects.link(o)
        if material is not None:
            me.materials.append(material)
        a = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
        a.data.foreach_set('vector', V[used].ravel().astype(np.float32))
        a = me.attributes.new('top_d', 'FLOAT', 'POINT')
        a.data.foreach_set('value', top_d[used].astype(np.float32))
        Wk = Wr[used]
        for gi, nm in enumerate(names):
            if nm.split('.')[0] in draft.ARM_GROUPS or nm == 'shoulder smooth':
                continue
            w = Wk[:, gi]
            if (w > 1e-4).any():
                g = o.vertex_groups.new(name=nm)
                for val in np.unique(np.round(w[w > 1e-4], 3)):
                    g.add([int(i) for i in np.nonzero(np.abs(np.round(w, 3) - val) < 1e-6)[0]], float(val), 'REPLACE')
        a = me.attributes.new('cut', 'FLOAT_VECTOR', 'POINT')
        a.data.foreach_set('vector', Vc[used].ravel().astype(np.float32))
        a = me.attributes.new('last_z', 'FLOAT', 'POINT')
        a.data.foreach_set('value', Vs[used, 2].astype(np.float32))
        if rig is not None:
            src = next((m for m in body.modifiers if m.type == 'ARMATURE'), None)
            m = o.modifiers.new('rig', 'ARMATURE')
            m.object = rig
            if src is not None:
                m.use_deform_preserve_volume = src.use_deform_preserve_volume
            if settle:
                crease(o, rig, list(colliders) or [body], M, ratios, log=log)
        m = o.modifiers.new('thickness', 'SOLIDIFY')
        m.thickness = r['leather']
        m.offset = 1.0
        m.use_rim = True
        m.use_even_offset = False
        m.use_quality_normals = True
        out[side] = o
        log('boot: %s, %d points' % (o.name, len(me.vertices)))
    return out


def crease(o, rig, colliders, M, ratios=RATIOS, P=LEATHER, log=print):
    """Step 5: held at the collar and over the foot, the shaft takes up the slack it was cut with (the 'cut'
    attribute) and buckles on the colliders. The boot's rest shape becomes what it settles to."""
    me = o.data
    n = len(me.vertices)
    cut = np.zeros(n * 3)
    me.attributes['cut'].data.foreach_get('vector', cut)
    top_d, zr = np.zeros(n), np.zeros(n)
    me.attributes['top_d'].data.foreach_get('value', top_d)
    me.attributes['last_z'].data.foreach_get('value', zr)
    join = ratios['join'] * M['H']
    pin = np.maximum(_ss(0.5, 0.0, top_d), _ss(M['ankle_z'] - 0.2 * join, M['ankle_z'] - 1.2 * join, zr))
    return solver.settle(o, rig, colliders, pinned=pin, P=P, log=log, cut=cut.reshape(-1, 3), label='boot ' + o.name[-1])


def clearance(boot, hose, log=print):
    """The least gap between the boot's inner face and the hose's outer face, at the current frame, over every point
    of the hose under the boot (judged by where it lies at rest: below the boot's top, on the boot's own leg):
    negative where the hose comes through the leather. Both are measured on their own surface, without thickness."""
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
    z_top = float(br[:, 2].max())
    gaps = []
    for p, q in zip(hp, hr):
        if q[0] * side <= 0 or q[2] > z_top:
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
