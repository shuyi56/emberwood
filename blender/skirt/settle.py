"""The skirt's drape, found by Blender's cloth solver instead of drawn.

draft.py drafts the skirt as the pattern: the cloth as cut, hung on the hips with its flare and its train, and only
soft flutes in it. Here it is put on the posed body and let go. The legs (and the hands, which hang by the hips) are
the collider, with the floor under them; the top, down to the hips, is held where it was drafted, as the waistband
holds it; the wool is given a broadcloth's behaviour (it hardly stretches, shears a little, and bends stiffly for its
weight, so the flare keeps its line and breaks into a few deep, round folds at the hem), and gravity does the rest.
What it settles to is carried back to the skirt's rest shape through the bones, so the skirt still follows the rig.

The figure is 63 units tall where a woman is 1.65 m, and the solver takes a unit for a metre, so cloth of real stiffness
would hang like tissue. WOOL's numbers are for this scale: found by trial against how a broadcloth skirt hangs
(folds a few centimetres across at the hem, the flare held), not copied from a preset."""
import time

import bpy
import numpy as np

WOOL = dict(
    mass=0.5,                # of one point of the cloth
    tension=400.0,           # broadcloth hardly stretches
    compression=400.0,
    shear=80.0,
    bending=14.0,            # stiff for its weight: the flare holds, and the hem rounds into a few deep folds
    damping=(10.0, 10.0, 8.0, 3.0),      # tension, compression, shear, bending
    air=1.0,
    friction=8.0,            # wool on the skin and on the floor
    gap=0.08,                # how far the cloth is kept off the body, in units
    self_gap=0.03,           # and off itself
    quality=10,
    frames=60,
    pin_from=0.0,            # set by the caller: the cloth above this height is held
    pin_to=0.0,              # ... fully, from here up; between the two the hold eases off
)


def _posed(objs):
    """One mesh of the objects as they stand at this frame (the collider)."""
    dg = bpy.context.evaluated_depsgraph_get()
    pts, polys = [], []
    for o in objs:
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        n0 = len(pts)
        pts += [tuple(o.matrix_world @ v.co) for v in me.vertices]
        polys += [tuple(n0 + i for i in p.vertices) for p in me.polygons]
        e.to_mesh_clear()
    return pts, polys


def _bone_matrices(o, rig):
    """Each point's own blend of its bones' moves at this frame (as the armature modifier makes it), [n, 4, 4]."""
    mw, mwi = np.array(rig.matrix_world), np.array(rig.matrix_world.inverted())
    move = {}
    for g in o.vertex_groups:
        pb = rig.pose.bones.get(g.name)
        if pb is not None:
            move[g.index] = mw @ np.array(pb.matrix @ pb.bone.matrix_local.inverted()) @ mwi
    out = np.zeros((len(o.data.vertices), 4, 4))
    tot = np.zeros(len(o.data.vertices))
    for v in o.data.vertices:
        for g in v.groups:
            if g.group in move and g.weight > 0:
                out[v.index] += g.weight * move[g.group]
                tot[v.index] += g.weight
    out[tot == 0] = np.eye(4)
    tot[tot == 0] = 1.0
    return out / tot[:, None, None]


# the local solver's numbers (units of this figure: 1 unit is about 2.6 cm, so gravity is 9.8 m/s^2 over 0.026 m)
XPBD = dict(
    gravity=-650.0,          # (heavy broadcloth: about 1.7 times the weight of the first, light pass)
    h=1.0 / 120.0,           # the step
    steps=150,
    iterations=20,
    damping=0.06,
    stretch=0.0,             # compliance of the threads: broadcloth keeps its length
    shear=1e-5,
    bend=3e-5,               # stiff for its weight: the flare holds
    friction=0.5,
    gap=0.08,
)


def _ties(posed, faces, edges):
    """The constraints of the sewn cloth, from its mesh: every edge (a thread), the diagonals of its quads (the bias),
    and the ties between the two neighbours of a point that lie opposite each other (the bend: a thread resists
    folding, as in fabric-core.js's `bend`)."""
    stretch = [tuple(e) for e in edges]
    shear = []
    for f in faces:
        if len(f) == 4:
            shear += [(f[0], f[2]), (f[1], f[3])]
    nb = {}
    for a, b in edges:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    bend = []
    for v, ns in nb.items():
        d = [posed[n] - posed[v] for n in ns]
        L = [np.linalg.norm(x) + 1e-12 for x in d]
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                if np.dot(d[i], d[j]) / (L[i] * L[j]) < -0.75:
                    bend.append((ns[i], ns[j]))
    return np.array(stretch), np.array(shear), np.array(bend)


def _colour(c, n):
    """The constraints in groups that share no point, so a group can be solved at once and the groups one after the
    other (Gauss-Seidel: the solver's stiffness, which a plain average of the corrections does not have)."""
    used = [0] * n
    col = np.zeros(len(c), dtype=np.int32)
    for k, (i, j) in enumerate(c):
        m = used[i] | used[j]
        b = 0
        while m >> b & 1:
            b += 1
        col[k] = b
        used[i] |= 1 << b
        used[j] |= 1 << b
    return [np.nonzero(col == b)[0] for b in range(int(col.max()) + 1)]


def _hem_stitches(o, posed):
    """Pairs (point of the hem allowance, nearest point of the outer layer): the allowance's points are the ones the
    'hem_d' attribute gives under the band, the rest of the cloth carries 50."""
    from mathutils.kdtree import KDTree
    hd = np.empty(len(o.data.vertices))
    o.data.attributes['hem_d'].data.foreach_get('value', hd)
    outer = np.nonzero(hd >= 49.0)[0]
    inner = np.nonzero(hd < 49.0)[0]
    if not len(inner):
        return np.zeros((0, 2), dtype=int)
    kd = KDTree(len(outer))
    for k, i in enumerate(outer):
        kd.insert(tuple(posed[i]), int(i))
    kd.balance()
    pairs = []
    for i in inner:
        _, j, d = kd.find(tuple(posed[i]))
        if d < 0.6:
            pairs.append((int(i), int(j)))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def settle_xpbd(o, rig, colliders, floor_z, P=XPBD, pin=(0.0, 1.0), gap=None, log=print):
    """The same job as settle(), by the local XPBD solver (fabric-core.js's method: small steps, constraints with a
    compliance, the body pushes the points out) on the sewn mesh's own edges, in numpy. The skirt o is put on the
    posed body and let go: the top (from pin[1] up) is held as the waistband holds it, easing off to pin[0]."""
    from mathutils.bvhtree import BVHTree
    n = len(o.data.vertices)
    arm = next(m for m in o.modifiers if m.type == 'ARMATURE')
    arm.use_deform_preserve_volume = False
    B = _bone_matrices(o, rig)
    bpy.context.view_layer.update()
    ev0 = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    posed = np.empty(n * 3)
    ev0.data.vertices.foreach_get('co', posed)
    posed = posed.reshape(-1, 3)
    start = posed.copy()
    pts, polys = _posed(colliders)
    tree = BVHTree.FromPolygons(pts, polys)
    faces = [tuple(p.vertices) for p in o.data.polygons]
    e = np.empty(len(o.data.edges) * 2, dtype=np.int32)
    o.data.edges.foreach_get('vertices', e)
    e = e.reshape(-1, 2)
    st, sh, bd = _ties(posed, faces, e)
    cons = [(st, P['stretch']), (sh, P['shear']), (bd, P['bend'])]
    # the hem's stitching: the turned allowance is sewn down to the cloth it lies against (as the stitch line holds it),
    # so it cannot fall open
    stitched = _hem_stitches(o, posed)
    if len(stitched):
        cons.append((stitched, 1e-6))
    cons = [(c, k, np.linalg.norm(posed[c[:, 0]] - posed[c[:, 1]], axis=1), _colour(c, n)) for c, k in cons if len(c)]
    hold = np.clip((posed[:, 2] - pin[0]) / max(pin[1] - pin[0], 1e-6), 0.0, 1.0)
    hold = hold * hold * (3 - 2 * hold)
    w = 1.0 - hold
    w[hold > 0.999] = 0.0
    gaps = np.full(n, P['gap']) if gap is None else np.array([gap(p) for p in posed])      # (the cloth's stand-off from the skin, by height)
    x, prev = posed.copy(), posed.copy()
    pushed = np.zeros(n, dtype=int)
    h = P['h']
    keep = (1.0 - P['damping']) ** (h * 60)
    free = np.nonzero(w > 0)[0]
    low = np.minimum(floor_z + 0.02, start[:, 2])        # (the floor; the hem's fold already on it stays)
    t0 = time.time()
    for step in range(P['steps']):
        v = (x - prev) * keep
        prev = x.copy()
        x = x + v * (w > 0)[:, None]
        x[:, 2] += P['gravity'] * h * h * (w > 0)
        for it in range(P['iterations']):
            for c, k, L, groups in cons:
                alpha = k / (h * h)
                for g in (groups if it % 2 == 0 else groups[::-1]):
                    i, j, Lg = c[g, 0], c[g, 1], L[g]
                    d = x[j] - x[i]
                    ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                    wi, wj = w[i], w[j]
                    q = (-(ln - Lg) / (wi + wj + alpha + 1e-12) / ln)[:, None] * d
                    x[i] -= wi[:, None] * q
                    x[j] += wj[:, None] * q
            np.maximum(x[:, 2], low, out=x[:, 2])           # (the floor holds the points every pass, not once a step)
        x = x * (w > 0)[:, None] + start * (w == 0)[:, None]          # (the held points stay where they were drafted)
        # the body pushes the points out (every other step: the legs are still)
        if step % 2 == 0:
            for i in free:
                loc, nrm, _, d = tree.find_nearest(tuple(x[i]), 2.0 * gaps[i] + 0.5)
                if loc is None:
                    continue
                p = np.array(x[i]) - np.array(loc)
                side = float(np.dot(p, np.array(nrm)))
                if side < gaps[i]:
                    x[i] = np.array(loc) + np.array(nrm) * gaps[i]
                    pushed[i] += 1
                    # (friction: the push takes away the slide along the body, and the speed it gave)
                    nn = np.array(nrm)
                    slide = x[i] - prev[i]
                    slide -= np.dot(slide, nn) * nn
                    prev[i] = x[i] - (1.0 - P['friction']) * slide
    out = x
    l0 = np.linalg.norm(start[e[:, 0]] - start[e[:, 1]], axis=1)
    l1 = np.linalg.norm(out[e[:, 0]] - out[e[:, 1]], axis=1)
    mv = np.linalg.norm(out - start, axis=1)
    ratio = l1 / np.maximum(l0, 1e-9)
    worst = np.argsort(-np.abs(ratio - 1.0))[:8]
    log('skirt (local solver): the worst edges (z, rest -> settled): ' + '; '.join(
        'z %.1f: %.2f -> %.2f' % (0.5 * (start[e[k, 0], 2] + start[e[k, 1], 2]), l0[k], l1[k]) for k in worst))
    log('   those edges: pushed by the body %s times; start z %s; end z %s' % (
        [(int(pushed[e[k, 0]]), int(pushed[e[k, 1]])) for k in worst],
        [(round(float(start[e[k, 0], 2]), 2), round(float(start[e[k, 1], 2]), 2)) for k in worst],
        [(round(float(out[e[k, 0], 2]), 2), round(float(out[e[k, 1], 2]), 2)) for k in worst]))
    log('skirt (local solver): %d steps in %.0f s: points moved %.2f on average, %.2f at most; threads %.1f%% longer on '
        'average, %.1f%% at most, %.1f%% shorter at most' % (P['steps'], time.time() - t0, float(mv.mean()), float(mv.max()),
                                                            100.0 * float(ratio.mean() - 1.0), 100.0 * float(ratio.max() - 1.0),
                                                            100.0 * float(1.0 - ratio.min())))
    Bi = np.linalg.inv(B)
    back = np.einsum('nij,nj->ni', Bi[:, :3, :3], out) + Bi[:, :3, 3]
    o.data.vertices.foreach_set('co', back.reshape(-1))
    o.data.update()
    bpy.context.scene.frame_set(1)
    return out


def settle(o, rig, colliders, floor_z, P=WOOL, log=print):
    """Let the skirt o (skinned to rig, in its drafted rest shape) fall on the posed colliders and the floor, and set
    its rest shape to what it settles to. Returns the settled points as posed."""
    sc = bpy.context.scene
    n = len(o.data.vertices)
    arm = next(m for m in o.modifiers if m.type == 'ARMATURE')
    arm.use_deform_preserve_volume = False        # (plain blending, so the settled shape can be carried back exactly)
    B = _bone_matrices(o, rig)
    rest = np.empty(n * 3)
    o.data.vertices.foreach_get('co', rest)
    rest = rest.reshape(-1, 3)
    # where it stands at this frame, the push off the legs included: the cloth starts outside them
    bpy.context.view_layer.update()
    ev0 = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    posed = np.empty(n * 3)
    ev0.data.vertices.foreach_get('co', posed)
    posed = posed.reshape(-1, 3)
    # the cloth, as it stands on the posed body, and the body to fall on
    me = bpy.data.meshes.new('cloth sim')
    me.from_pydata([tuple(p) for p in posed], [], [tuple(p.vertices) for p in o.data.polygons])
    cloth = bpy.data.objects.new('cloth sim', me)
    sc.collection.objects.link(cloth)
    pts, polys = _posed(colliders)
    r = float(np.abs(posed[:, :2]).max()) + 20.0
    n0 = len(pts)
    pts += [(-r, -r, floor_z), (r, -r, floor_z), (r, r, floor_z), (-r, r, floor_z)]
    polys += [(n0, n0 + 1, n0 + 2, n0 + 3)]
    cme = bpy.data.meshes.new('cloth collider')
    cme.from_pydata(pts, [], polys)
    body = bpy.data.objects.new('cloth collider', cme)
    sc.collection.objects.link(body)
    body.modifiers.new('collision', 'COLLISION')
    body.collision.thickness_outer = P['gap']
    body.collision.thickness_inner = 0.25
    body.collision.cloth_friction = P['friction']
    body.collision.damping = 0.5
    # the waistband's hold: full from the top to pin_to, easing off to nothing at pin_from
    z = posed[:, 2]
    hold = np.clip((z - P['pin_from']) / max(P['pin_to'] - P['pin_from'], 1e-6), 0.0, 1.0)
    hold = hold * hold * (3 - 2 * hold)
    g = cloth.vertex_groups.new(name='pin')
    for i in np.nonzero(hold > 0)[0]:
        g.add([int(i)], float(hold[i]), 'REPLACE')
    m = cloth.modifiers.new('cloth', 'CLOTH')
    s, c = m.settings, m.collision_settings
    s.quality = P['quality']
    s.mass = P['mass']
    s.tension_stiffness, s.compression_stiffness = P['tension'], P['compression']
    s.shear_stiffness, s.bending_stiffness = P['shear'], P['bending']
    s.tension_damping, s.compression_damping, s.shear_damping, s.bending_damping = P['damping']
    s.air_damping = P['air']
    s.vertex_group_mass = 'pin'
    s.pin_stiffness = 1.0
    c.use_collision, c.collision_quality, c.distance_min = True, 4, 0.03
    c.use_self_collision, c.self_distance_min, c.self_friction = True, P['self_gap'], 8.0
    m.point_cache.frame_start, m.point_cache.frame_end = 1, P['frames']
    end0 = sc.frame_end
    sc.frame_end = max(sc.frame_end, P['frames'])
    t0 = time.time()
    for f in range(1, P['frames'] + 1):
        sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cloth.evaluated_get(dg)
    out = np.empty(n * 3)
    ev.data.vertices.foreach_get('co', out)
    out = out.reshape(-1, 3)
    e = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get('vertices', e)
    e = e.reshape(-1, 2)
    l0 = np.linalg.norm(posed[e[:, 0]] - posed[e[:, 1]], axis=1)
    l1 = np.linalg.norm(out[e[:, 0]] - out[e[:, 1]], axis=1)
    mv = np.linalg.norm(out - posed, axis=1)
    worst = np.argsort(-(l1 / np.maximum(l0, 1e-9)))[:6]
    log('skirt: the most stretched edges (at rest): ' + '; '.join(
        'z %.1f, %.2f -> %.2f long' % (0.5 * (rest[e[k, 0], 2] + rest[e[k, 1], 2]), l0[k], l1[k]) for k in worst))
    log('skirt: settled in %d frames (%.0f s): points moved %.2f on average, %.2f at most; the cloth is %.1f%% longer '
        'along its threads on average, %.1f%% at most' % (P['frames'], time.time() - t0, float(mv.mean()), float(mv.max()),
                                                          100.0 * float((l1 / np.maximum(l0, 1e-9)).mean() - 1.0),
                                                          100.0 * float((l1 / np.maximum(l0, 1e-9)).max() - 1.0)))
    # carried back through the bones to the rest shape
    Bi = np.linalg.inv(B)
    back = np.einsum('nij,nj->ni', Bi[:, :3, :3], out) + Bi[:, :3, 3]
    o.data.vertices.foreach_set('co', back.reshape(-1))
    o.data.update()
    for ob in (cloth, body):
        d = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(d)
    sc.frame_end = end0
    sc.frame_set(1)
    return out
