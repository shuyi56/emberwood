"""The sleeve's cloth on the swatch bench (chemise/swatch.py's strip and curtain, and a puff of its own): seconds a
trial, where settling both sleeves on the figure takes minutes, so the wool and the head are found here and only
confirmed on the garment.

  the strip    the cantilever: the bending length (Peirce's), the one number for how a cloth hangs. A riding habit's
               broadcloth is a heavy, firm wool: about 3 to 4 cm (shirt linen 2.5 to 3.5, canvas 4 and over).
  the curtain  a piece gathered into a band by the head's own fullness (sleeve.RATIOS['full']): how wide and deep its
               folds are, and how crumpled. Wool's folds are broad and rounded: 8 cm and more, crumple near 0.
  the puff     the head itself, in small: a dome of cloth on its support (a cap of a sphere as wide as the head),
               pinned round its rim, which is the seam, and let out by the fullness in a band along the rim: the cloth
               there is longer than the seam it is sewn to, so it stands up in gathers. Read off it: the crown's height against the support's (does the head
               stand, or slump off its net), how many gathers stand round the rim per unit, and how deep.

    python3 blender/redingote/bench.py                          sleeve.WOOL, the head HEAD_STIFF times stiffer
    python3 blender/redingote/bench.py bending=1.5,2.5,4 stiff=2,4    every pairing, a row each"""
import itertools
import math
import os
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'chemise'))
sys.path.insert(0, HERE)
import solver  # noqa: E402
import swatch  # noqa: E402

CM = swatch.CM
GRID = swatch.GRID


def _weave_with(stiff_group, factor):
    """solver.weave, and the bending stiffness raised by factor over the group (the head's interlining)."""
    plain = solver.weave

    def weave(s, P):
        plain(s, P)
        if stiff_group and factor > 1.0:
            s.vertex_group_bending = stiff_group
            s.bending_stiffness_max = P['bending'] * factor
    return weave


def dome(radius, rise, grid=GRID):
    """A cap of a sphere: its rim a circle of `radius` at z 0, its crown `rise` above. Points and four-sided faces
    on rings GRID apart, and each point's distance from the rim along the surface."""
    R = (radius ** 2 + rise ** 2) / (2 * rise)           # the sphere's radius
    th1 = math.asin(radius / R)
    n_r = max(3, int(round(R * th1 / grid)))
    V, F, u, ring_start = [], [], [], []
    for i in range(n_r + 1):
        th = th1 * (1.0 - i / n_r)
        r = R * math.sin(th)
        n = max(6, int(round(2 * math.pi * r / grid))) if i < n_r else 1
        ring_start.append((len(V), n))
        for j in range(n):
            a = 2 * math.pi * j / n
            V.append((r * math.cos(a), r * math.sin(a), R * math.cos(th) - (R - rise)))
            u.append(R * (th1 - th))
    for i in range(n_r):
        (s0, n0), (s1, n1) = ring_start[i], ring_start[i + 1]
        for j in range(n0):
            a, b = s0 + j, s0 + (j + 1) % n0
            c = s1 + int(round((j + 1) * n1 / n0)) % n1
            d = s1 + int(round(j * n1 / n0)) % n1
            F.append((a, b, c, d) if c != d else (a, b, c))
    return np.array(V), [tuple(dict.fromkeys(f)) for f in F], np.array(u), ring_start[0][1]


def puff(P, stiff, full, radius=2.6, rise=1.1, band=0.6):
    """The head in small. Returns (the crown's height against the support's, gathers per unit round the rim, their
    depth in units, how far the cloth sank into the support at worst)."""
    sc = bpy.context.scene
    V, F, u, n_rim = dome(radius, rise)
    # the support: the same dome a little lower, as a collider
    Vs, Fs, _, _ = dome(radius * 0.97, rise * 0.85)
    me_s = bpy.data.meshes.new('support')
    me_s.from_pydata([tuple(p) for p in Vs], [], Fs)
    sup = bpy.data.objects.new('support', me_s)
    sc.collection.objects.link(sup)
    sup.modifiers.new('collision', 'COLLISION')
    sup.collision.thickness_outer = P['gap']
    sup.collision.cloth_friction = P['friction']
    me = bpy.data.meshes.new('puff')
    me.from_pydata([tuple(p) for p in V], [], F)
    o = bpy.data.objects.new('puff', me)
    sc.collection.objects.link(o)
    g = o.vertex_groups.new(name='pin')
    g.add(list(range(n_rim)), 1.0, 'REPLACE')
    gs = o.vertex_groups.new(name='stiff')
    for i in range(len(V)):
        gs.add([i], 1.0, 'REPLACE')
    m = o.modifiers.new('cloth', 'CLOTH')
    s, c = m.settings, m.collision_settings
    _weave_with('stiff', stiff)(s, P)
    s.vertex_group_mass, s.pin_stiffness = 'pin', 1.0
    # (the gathers: the cloth along the rim is fuller than the rim it is sewn to, fading out over the band; the
    # crown is drafted at its height and not let out, or it would only balloon)
    gather = 1.0 - np.clip((u - 0.3 * band) / band, 0.0, 1.0)
    solver.let_out(o, s, full * gather, P)
    c.use_collision = True
    c.collision_quality, c.distance_min = 4, 0.02
    c.use_self_collision, c.self_distance_min, c.self_friction = True, P['self_gap'], 8.0
    m.point_cache.frame_start, m.point_cache.frame_end = 1, P['frames']
    sc.frame_end = max(sc.frame_end, P['frames'])
    for f in range(1, P['frames'] + 1):
        sc.frame_set(f)
    ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    X = np.empty(len(V) * 3)
    ev.data.vertices.foreach_get('co', X)
    X = X.reshape(-1, 3)
    crown = float(X[:, 2].max()) / (rise * 0.85 + P['gap'])
    # the gathers: the cloth's height round a ring just inside the band
    k = (u > 0.8 * band) & (u < 1.3 * band)
    a = np.arctan2(X[k, 1], X[k, 0])
    rr = np.hypot(X[k, 0], X[k, 1]) * 0 + X[k, 2] + 0.25 * np.hypot(X[k, 0], X[k, 1])
    order = np.argsort(a)
    h = rr[order]
    h = h - np.convolve(np.concatenate([h[-12:], h, h[:12]]), np.ones(25) / 25.0, 'same')[12:-12]
    h = np.convolve(np.concatenate([h[-1:], h, h[:1]]), np.ones(3) / 3.0, 'same')[1:-1]
    up = np.nonzero((h > np.roll(h, 1)) & (h >= np.roll(h, -1)) & (h > 0.02))[0]
    ring = 2 * math.pi * radius * 0.85
    depth = float(np.percentile(h, 95) - np.percentile(h, 5))
    # how far into the support the cloth went (it should rest on it, not sink through)
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    t = BVHTree.FromPolygons([Vector(p) for p in Vs], Fs)
    sink = 0.0
    for p in X:
        loc, nrm, _, dd = t.find_nearest(Vector(p))
        if loc is not None and (Vector(p) - loc).dot(nrm) < 0:
            sink = max(sink, dd)
    for ob in (o, sup):
        d = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(d)
    sc.frame_set(1)
    return crown, len(up) / ring, depth, sink


def bench(P, stiff, full):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    t0 = time.time()
    bend, over = swatch.strip(P)
    per, fdepth, crumple, stretch = swatch.curtain(P)
    crown, gathers, gdepth, sink = puff(P, stiff, full)
    return dict(bend_cm=bend * CM, limp=over < 0, fold_cm=CM / max(per, 1e-9), fdepth_cm=fdepth * CM, crumple=crumple,
                crown=crown, gathers_cm=CM / max(gathers, 1e-9), gdepth_cm=gdepth * CM, sink_cm=sink * CM,
                seconds=time.time() - t0)


def main(argv):
    import sleeve
    sweep = {k: [float(x) for x in v.split(',')] for k, v in (a.split('=') for a in argv if '=' in a)}
    stiffs = sweep.pop('stiff', [sleeve.HEAD_STIFF])
    fulls = sweep.pop('full', [sleeve.RATIOS['full']])
    keys = list(sweep)
    swatch.FULL = fulls[0]
    print(' '.join('%9s' % k for k in keys) + '     stiff  full   bend cm  fold cm depth cm crumple  crown  gather cm  depth cm  sink cm    s')
    for vals in itertools.product(*sweep.values()):
        for st in stiffs:
            for fu in fulls:
                swatch.FULL = fu
                P = dict(sleeve.WOOL, **{k: type(sleeve.WOOL[k])(v) for k, v in zip(keys, vals)})
                r = bench(P, st, fu)
                print(' '.join('%9g' % v for v in vals) + '  %8g %5.2f  %s%7.2f  %7.2f  %7.2f  %6.2f  %5.2f  %9.2f  %8.2f  %7.2f %4.0f' % (
                    st, fu, '<' if r['limp'] else ' ', r['bend_cm'], r['fold_cm'], r['fdepth_cm'], r['crumple'], r['crown'],
                    r['gathers_cm'], r['gdepth_cm'], r['sink_cm'], r['seconds']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
