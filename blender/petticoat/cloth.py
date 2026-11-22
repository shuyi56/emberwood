"""The petticoat's hang, found by Blender's cloth solver.

draft.hang() puts the cloth on the body with every row at the pattern's girth and its spare cloth in flutes; here it
is let go on the posed body (and the hands) and the floor, held by its band, and settles under its own weight. What
it settles to is carried back to the garment's rest shape through its bones, so it still follows the rig.

The figure is 63 units tall where a woman is 1.65 m, and the solver takes a unit for a metre, so cloth of real
stiffness would hang like tissue. COTTON's numbers are for this scale, found on swatch.py's bench (the strip's
bending length and the gathered curtain's folds) against a petticoat cotton: a firm longcloth or cambric, starched,
bending length about 3 to 4 cm, falling in a few round flutes rather than many fine folds."""
import time

import bpy
import numpy as np

COTTON = dict(
    mass=0.45,               # of one point of the cloth (with bending 40: Peirce's bending length 3.7 cm on swatch.py's strip)
    tension=90.0,            # cotton barely stretches along its threads
    compression=12.0,        # but gives at once when pushed: it folds instead of standing
    shear=12.0,              # a little give on the bias
    bending=40.0,            # starched longcloth: stiffer than shirt linen, so the flutes stay round and few
    damping=(8.0, 8.0, 6.0, 2.0),
    air=0.8,
    friction=4.0,            # cotton on the corset's coutil and on skin
    floor_friction=1.0,      # and on a floor: the train slides out into its folds instead of sticking where it lands
    gap=0.06,                # how far the cloth is kept off the body, in units
    self_gap=0.02,
    quality=8,
    frames=120,
    self_collide=False,
    sew_force=0.0,           # the sewing threads' pull (0: as hard as the solver allows)
    sew_frames=8,            # frames the pieces are sewn in without their weight (the seams close in about 10; longer and
                             # the pulled-in cloth flies up past the band)
    sew_ramp=12,             # and the frames over which the weight comes in
)


def weave(s, P):
    s.quality = P['quality']
    s.mass = P['mass']
    s.tension_stiffness, s.compression_stiffness = P['tension'], P['compression']
    s.shear_stiffness, s.bending_stiffness = P['shear'], P['bending']
    s.tension_damping, s.compression_damping, s.shear_damping, s.bending_damping = P['damping']
    s.air_damping = P['air']
    s.bending_model = 'ANGULAR'


def _posed(objs, floor=None, floor_r=60.0):
    """One mesh of the objects as they stand at this frame (the collider), or with floor, a floor square alone."""
    dg = bpy.context.evaluated_depsgraph_get()
    pts, polys = [], []
    for o in objs or ():
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        n0 = len(pts)
        mw = o.matrix_world
        pts += [tuple(mw @ v.co) for v in me.vertices]
        polys += [tuple(n0 + i for i in p.vertices) for p in me.polygons]
        e.to_mesh_clear()
    if floor is not None:
        n0 = len(pts)
        pts += [(-floor_r, -floor_r, floor), (floor_r, -floor_r, floor), (floor_r, floor_r, floor), (-floor_r, floor_r, floor)]
        polys.append((n0, n0 + 1, n0 + 2, n0 + 3))
    me = bpy.data.meshes.new('petticoat collider')
    me.from_pydata(pts, [], polys)
    return me


def bone_matrices(o, rig):
    """Each point's blend of its bones' moves at this frame (as a plain armature modifier makes it), [n, 4, 4]."""
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


def settle(o, rig, colliders, pinned, floor, P=COTTON, log=print, sew=None, decimate=None):
    """Let the garment o (skinned to rig, in its drafted rest shape) fall on the posed colliders and the floor, held
    where pinned (a weight per point, 1 held), and set its rest shape to what it settles to. Returns the settled
    points (posed).

    sew: pairs of points (cut pieces laid apart) tied by the solver's sewing threads, which pull each pair together;
    the cloth is sewn weightless for P['sew_frames'] frames, then its weight comes in. decimate: a ratio for the
    colliders (a fine corset mesh makes every frame slow)."""
    sc = bpy.context.scene
    n = len(o.data.vertices)
    # (a settle is kept by its inputs: the drafted points, the cloth's numbers and the colliders' points, so a run
    # that only changes the flounce or the look reuses it)
    import hashlib
    import os
    rest0 = np.empty(n * 3)
    o.data.vertices.foreach_get('co', rest0)
    h = hashlib.sha1(np.round(rest0, 4).tobytes() + repr(sorted(P.items())).encode() + np.round(pinned, 3).tobytes())
    if sew is not None:
        h.update(np.asarray(sew, dtype=np.int64).tobytes() + repr(decimate).encode())
    dg = bpy.context.evaluated_depsgraph_get()
    for c_ in colliders:
        e = c_.evaluated_get(dg)
        me_ = e.to_mesh()
        co_ = np.empty(len(me_.vertices) * 3)
        me_.vertices.foreach_get('co', co_)
        h.update(np.round(co_, 3).tobytes())
        e.to_mesh_clear()
    cache = os.path.join(os.environ.get('PETTI_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'petticoat')),
                         h.hexdigest()[:16] + '.npy')
    if os.path.exists(cache) and not os.environ.get('PETTI_NOCACHE'):
        back = np.load(cache)
        o.data.vertices.foreach_set('co', back.reshape(-1))
        if 'rest' in o.data.attributes:
            o.data.attributes['rest'].data.foreach_set('vector', back.reshape(-1))
        o.data.update()
        log('petticoat cloth: settled shape from the cache (%s)' % cache)
        return None
    arm = next(m for m in o.modifiers if m.type == 'ARMATURE')
    arm.use_deform_preserve_volume = False
    B = bone_matrices(o, rig)
    rest = np.empty(n * 3)
    o.data.vertices.foreach_get('co', rest)
    rest = rest.reshape(-1, 3)
    posed = np.einsum('nij,nj->ni', B[:, :3, :3], rest) + B[:, :3, 3]
    me = bpy.data.meshes.new('petticoat sim')
    loose = [] if sew is None else [tuple(int(i) for i in e) for e in sew if e[0] != e[1]]
    me.from_pydata([tuple(p) for p in posed], loose, [tuple(p.vertices) for p in o.data.polygons])
    cloth = bpy.data.objects.new('petticoat sim', me)
    sc.collection.objects.link(cloth)
    body = bpy.data.objects.new('petticoat collider', _posed(colliders))
    ground = bpy.data.objects.new('petticoat floor', _posed(None, floor))
    for ob, fr in ((body, P['friction']), (ground, P['floor_friction'])):
        sc.collection.objects.link(ob)
        if decimate and ob is body:
            dm = ob.modifiers.new('fewer', 'DECIMATE')
            dm.ratio = decimate
        ob.modifiers.new('collision', 'COLLISION')
        ob.collision.thickness_outer = P['gap']
        ob.collision.thickness_inner = 0.3
        ob.collision.cloth_friction = fr
        ob.collision.damping = 0.6
    g = cloth.vertex_groups.new(name='pin')
    for i in np.nonzero(pinned > 0)[0]:
        g.add([int(i)], float(pinned[i]), 'REPLACE')
    m = cloth.modifiers.new('cloth', 'CLOTH')
    s, c = m.settings, m.collision_settings
    weave(s, P)
    s.vertex_group_mass, s.pin_stiffness = 'pin', 1.0
    c.use_collision, c.collision_quality, c.distance_min = True, 3, 0.02
    c.use_self_collision = P['self_collide']
    c.self_distance_min, c.self_friction = P['self_gap'], 5.0
    m.point_cache.frame_start, m.point_cache.frame_end = 1, P['frames']
    if sew is not None and len(loose):
        s.use_sewing_springs = True
        s.sewing_force_max = P.get('sew_force', 0.0)
        ew = s.effector_weights
        f0, f1 = P.get('sew_frames', 45), P.get('sew_frames', 45) + P.get('sew_ramp', 25)
        for f, g_ in ((1, 0.0), (f0, 0.0), (f1, 1.0)):
            ew.gravity = g_
            ew.keyframe_insert('gravity', frame=f)
    end0 = sc.frame_end
    sc.frame_end = max(sc.frame_end, P['frames'])
    t0 = time.time()
    prev = posed
    for f in range(1, P['frames'] + 1):
        sc.frame_set(f)
        if f % 10 == 0:
            ev = cloth.evaluated_get(bpy.context.evaluated_depsgraph_get())
            cur = np.empty(n * 3)
            ev.data.vertices.foreach_get('co', cur)
            cur = cur.reshape(-1, 3)
            log('petticoat cloth: frame %d (%.0f s), points moved %.3f on average in the last 10 frames' % (
                f, time.time() - t0, float(np.linalg.norm(cur - prev, axis=1).mean())))
            prev = cur
            if os.environ.get('PETTI_DUMP'):
                np.save('%s_%03d.npy' % (os.environ['PETTI_DUMP'], f), cur.astype(np.float32))
    ev = cloth.evaluated_get(bpy.context.evaluated_depsgraph_get())
    out = np.empty(n * 3)
    ev.data.vertices.foreach_get('co', out)
    out = out.reshape(-1, 3)
    e = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get('vertices', e)
    e = e.reshape(-1, 2)
    if sew is not None and len(loose):
        lz = np.array(loose)
        gap = np.linalg.norm(out[lz[:, 0]] - out[lz[:, 1]], axis=1)
        log('petticoat cloth: %d sewn pairs, apart %.3f on average, %.3f at most, %d still over 0.1' % (
            len(lz), float(gap.mean()), float(gap.max()), int((gap > 0.1).sum())))
        ls = set(map(tuple, np.sort(lz, 1)))
        isf = np.array([tuple(sorted(x)) not in ls for x in e])
        e = e[isf]
    l0 = np.linalg.norm(posed[e[:, 0]] - posed[e[:, 1]], axis=1)
    l1 = np.linalg.norm(out[e[:, 0]] - out[e[:, 1]], axis=1)
    log('petticoat cloth: settled in %d frames (%.0f s): points moved %.2f on average, %.2f at most; edges %.1f%% '
        'longer on average, %.1f%% at the 99th percentile' % (
            P['frames'], time.time() - t0, float(np.linalg.norm(out - posed, axis=1).mean()),
            float(np.linalg.norm(out - posed, axis=1).max()), 100.0 * float((l1 / np.maximum(l0, 1e-9)).mean() - 1.0),
            100.0 * float(np.percentile(l1 / np.maximum(l0, 1e-9), 99) - 1.0)))
    Bi = np.linalg.inv(B)
    back = np.einsum('nij,nj->ni', Bi[:, :3, :3], out) + Bi[:, :3, 3]
    o.data.vertices.foreach_set('co', back.reshape(-1))
    if 'rest' in o.data.attributes:
        o.data.attributes['rest'].data.foreach_set('vector', back.reshape(-1))
    o.data.update()
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, back)
    except OSError:
        pass
    for ob in (cloth, body, ground):
        d = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(d)
    sc.frame_end = end0
    sc.frame_set(1)
    return out
