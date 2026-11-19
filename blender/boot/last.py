"""A tall leather boot drafted on whatever legs it is given: from the sole to just under the knee, over the hose.

Nothing here is placed by hand. The boot is cut on a last, and the last is the leg: measure_legs() (the hose's own
tailor's tape, imported from blender/hose/draft.py) reads the heel, toe, ankle, calf and knee off the skin, and every
size of the boot is a share of the head's height H, the figure's unit, so the boot re-fits when the body or the head
changes. The Warden's old boot was a table of hand-set rings for her old legs; on this body its toe stopped 2.5 units
short of the foot.

A boot is not a close garment like hose: leather does not stretch, so it is made on the leg's shape but with the
room a foot needs to go in and a stride needs to bend. The shell is made in four steps:

  1. the last      the skin of the lower leg and foot let out by the ease: a snug shaft, more room at the ankle (the
                   foot has to pass through it), a toe box (the toe region stretched a little forward from the ball
                   of the foot, and raised), a welt round the sole's edge and a flared collar at the top.
  2. the stiffness the leg's hollows (the Achilles' hollow, the instep, beside the shin) are bridged over by a taut
                   membrane, harder than the hose's: leather cannot follow a hollow, so it is cut straight across.
  3. the trims     the top edge, level, a little under the knee: the line the greave overlaps (M['boot_top']).
  4. the rig       the skin's own bone weights, the same armature settings, so the boot bends as the leg bends.

The sole is a thickness added under the foot, so the figure stands on it: the figure's floor is the sole's under
face, SOLE below the foot's."""
import os
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('hose', 'vest'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import draft  # noqa: E402  (the hose's: the tape, the body's data, the fit check)
import fit  # noqa: E402    (the vest's cutter)

# Every size as a share of the head's height H.
RATIOS = {
    'leather': 0.0125,       # the leather's thickness (about 3 mm)
    'ease': 0.020,           # the gap between hose and shaft's inner face (the hose is 0.011 thick)
    'ankle_room': 0.030,     # the extra room round the ankle, for the foot to pass
    'ankle_span': 0.17,
    'flare': 0.022,          # the collar's flare at the top
    'flare_span': 0.17,
    'toe_room': 0.040,       # the toe box's height over the toe
    'toe_long': 0.070,       # how much longer the toe region is than the foot's (a share of its length)
    'welt': 0.030,           # the welt's width round the sole
    'sole': 0.026,           # the sole's thickness under the foot
    'bridge': 0.16,          # the most the leather stands off a hollow it bridges
    'top_below_knee': 0.12,  # the top edge below the knee's joint
    'bead': 0.05,            # the rolled top edge's depth
}


def _ss(a, b, x):
    return draft._ss(a, b, x)


def build_boots(body, M, rig=None, material=None, name='Boot', ratios=RATIOS, log=print):
    """Draft one boot for each leg from the measures M (draft.measure_legs) and hand them to the body's bones.
    Returns {'R': object, 'L': object}. M gains 'boot_top' (the edge's height) for whatever goes over it."""
    B = draft.body_data(body)
    P, Nn, W, names = B['P'], B['N'], B['W'], B['groups']
    H = M['H']
    r = {k: v * H if k not in ('toe_long',) else v for k, v in ratios.items()}
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

    # ---- 1. the last: the ease along the leg, the toe box, the welt, the collar
    ease = np.full(n, r['ease'])
    ease += r['ankle_room'] * np.exp(-((z - (M['ankle_z'] + 0.02 * H)) / r['ankle_span']) ** 2)
    ease += r['flare'] * _ss(z_top - r['flare_span'], z_top, z) ** 2
    toe = _ss(y_ball, M['toe_y'], y) * (z < M['ankle_z'])
    ease += r['toe_room'] * toe * np.clip(Nr[:, 2], 0.0, 1.0)             # (the toe box stands up over the toes)
    ease += r['welt'] * _ss(0.9, 0.25, z) * (np.abs(Nr[:, 2]) < 0.8)         # (the welt: the sole's edge stands out)
    ease *= 1.0 - 0.75 * np.clip(-Nr[:, 2], 0.0, 1.0) ** 2 * _ss(1.4, 0.3, z)       # (under the foot the sole is the room)
    X = X0 + Nr * ease[:, None]
    # the toe region carried forward from the ball: the toe is longer in the boot than in the foot
    fwd = X[:, 1] < y_ball
    X[fwd, 1] = y_ball + (X[fwd, 1] - y_ball) * (1.0 + ratios['toe_long'])
    # the sole's thickness under the foot
    down = np.clip(-Nr[:, 2], 0.0, 1.0) ** 2 * _ss(1.4, 0.3, z)
    X[:, 2] -= r['sole'] * down

    # ---- 2. the stiffness: bridge over the hollows, harder than the hose's
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    lift = np.zeros(n)
    for _ in range(260):
        acc = np.zeros_like(X)
        np.add.at(acc, E[:, 0], X[E[:, 1]])
        np.add.at(acc, E[:, 1], X[E[:, 0]])
        lap = acc / np.maximum(deg, 1)[:, None] - X
        out = np.minimum(np.maximum((lap * Nr).sum(1), 0.0) * 0.6, r['bridge'] - lift)
        lift += out
        X += Nr * out[:, None]
    log('boot: the last, %d points; bridged %d of them over hollows, by %.3f at most' % (n, int((lift > 0.01).sum()), lift.max()))

    # ---- 3. the trims
    D = np.concatenate([X0, X, Nr, Wr], 1)
    D, fs = fit.trim(D, faces, lambda A: z_top - A[:, 2])
    Vs, V, Nr, Wr = D[:, 0:3], D[:, 3:6], D[:, 6:9], D[:, 9:9 + len(names)]
    Nr = Nr / np.maximum(np.linalg.norm(Nr, axis=1, keepdims=True), 1e-9)
    log('boot: cut the top edge at %.2f (%.2f below the knee), %d faces left' % (z_top, r['top_below_knee'], len(fs)))
    top_d = z_top - Vs[:, 2]
    V = V + Nr * (_ss(r['bead'] * 2.0, 0.0, top_d) * 0.7 * r['leather'])[:, None]   # (the rolled edge stands proud)

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
        if rig is not None:
            src = next((m for m in body.modifiers if m.type == 'ARMATURE'), None)
            m = o.modifiers.new('rig', 'ARMATURE')
            m.object = rig
            if src is not None:
                m.use_deform_preserve_volume = src.use_deform_preserve_volume
        m = o.modifiers.new('thickness', 'SOLIDIFY')
        m.thickness = r['leather']
        m.offset = 1.0
        m.use_rim = True
        m.use_even_offset = False
        m.use_quality_normals = True
        out[side] = o
        log('boot: %s, %d points' % (o.name, len(me.vertices)))
    return out


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
