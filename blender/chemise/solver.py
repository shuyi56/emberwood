"""The chemise's drape, found by Blender's cloth solver instead of drawn.

drape.py drafts the garment as a plain loose piece with no folds in it: that is the pattern, the cloth as cut. Here
it is put on the posed body and let go. The body (and the hands) are the collider; the cloth is given linen's
behaviour (it barely stretches, shears a little, and bends stiffly for its weight, so it breaks into a few firm
folds and not into many limp ones); gravity does the rest. What it settles to is carried back to the garment's rest
shape through the bones, so the garment still follows the rig.

The figure is 63 units tall where a woman is 1.65 m, and the solver takes a unit for a metre, so cloth of real
stiffness would hang like tissue. LINEN's numbers are for this scale: found by trial against how a linen sleeve
breaks (folds a few centimetres across, 1 to 2 units), not copied from a preset."""
import bpy
import numpy as np

LINEN = dict(
    mass=0.25,               # of one point of the cloth
    tension=120.0,            # linen barely stretches
    compression=120.0,
    shear=20.0,              # and gives a little on the bias
    bending=1.0,             # crisp for its weight: it folds, it does not crumple (0.4 crumpled like paper, 6 stood
                             # like a shell)
    damping=(8.0, 8.0, 6.0, 1.5),      # tension, compression, shear, bending
    air=0.6,
    friction=6.0,            # linen on skin
    gap=0.07,                # how far the cloth is kept off the skin, in units
    self_gap=0.015,          # and off itself (small: at 0.045 the cloth's own close points pushed each other apart,
                             # and the whole garment swelled and drew its hems in)
    quality=9,
    frames=80,
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
    me = bpy.data.meshes.new('collider')
    me.from_pydata(pts, [], polys)
    return me


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


def settle(o, rig, colliders, pinned=None, P=LINEN, log=print, cut=None, label='chemise'):
    """Let the garment o (skinned to rig, in its drafted rest shape) fall on the posed colliders, and set its rest
    shape to what it settles to. pinned: a weight for each point, 1 held where it was drafted (or None). cut: the
    piece as it was cut, a point for each point in the same rest space, when that differs from where it is put on
    (a piece cut longer than the place it is held in has to take up the length: it buckles); None, as drafted."""
    sc = bpy.context.scene
    n = len(o.data.vertices)
    arm = next(m for m in o.modifiers if m.type == 'ARMATURE')
    arm.use_deform_preserve_volume = False        # (plain blending, so the settled shape can be carried back exactly)
    B = _bone_matrices(o, rig)
    rest = np.empty(n * 3)
    o.data.vertices.foreach_get('co', rest)
    rest = rest.reshape(-1, 3)
    posed = np.einsum('nij,nj->ni', B[:, :3, :3], rest) + B[:, :3, 3]
    # the cloth, as it stands on the posed body, and the body to fall on
    me = o.data.copy()
    me.vertices.foreach_set('co', posed.reshape(-1))
    cloth = bpy.data.objects.new('cloth sim', me)
    sc.collection.objects.link(cloth)
    if cut is not None:
        cloth.shape_key_add(name='Basis')
        k = cloth.shape_key_add(name='cut')
        k.data.foreach_set('co', (np.einsum('nij,nj->ni', B[:, :3, :3], cut) + B[:, :3, 3]).reshape(-1))
    body = bpy.data.objects.new('cloth collider', _posed(colliders))
    sc.collection.objects.link(body)
    cm = body.modifiers.new('collision', 'COLLISION')
    body.collision.thickness_outer = P['gap']
    body.collision.thickness_inner = 0.25
    body.collision.cloth_friction = P['friction']
    body.collision.damping = 0.5
    del cm
    if pinned is not None and np.any(pinned > 0):
        g = cloth.vertex_groups.new(name='pin')
        for i in np.nonzero(pinned > 0)[0]:
            g.add([int(i)], float(pinned[i]), 'REPLACE')
    m = cloth.modifiers.new('cloth', 'CLOTH')
    s, c = m.settings, m.collision_settings
    s.quality = P['quality']
    s.mass = P['mass']
    s.tension_stiffness, s.compression_stiffness = P['tension'], P['compression']
    s.shear_stiffness, s.bending_stiffness = P['shear'], P['bending']
    s.tension_damping, s.compression_damping, s.shear_damping, s.bending_damping = P['damping']
    s.air_damping = P['air']
    if cut is not None:
        s.rest_shape_key = cloth.data.shape_keys.key_blocks['cut']
    if pinned is not None and np.any(pinned > 0):
        s.vertex_group_mass = 'pin'
        s.pin_stiffness = P.get('pin', 1.0)
    c.use_collision, c.collision_quality, c.distance_min = True, 4, 0.02
    c.use_self_collision, c.self_distance_min, c.self_friction = True, P['self_gap'], 8.0
    m.point_cache.frame_start, m.point_cache.frame_end = 1, P['frames']
    end0 = sc.frame_end
    sc.frame_end = max(sc.frame_end, P['frames'])
    import time
    t0 = time.time()
    for f in range(1, P['frames'] + 1):
        sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cloth.evaluated_get(dg)
    out = np.empty(n * 3)
    ev.data.vertices.foreach_get('co', out)
    out = out.reshape(-1, 3)
    # how far it fell, and whether it stretched (linen should not)
    e = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get('vertices', e)
    e = e.reshape(-1, 2)
    l0 = np.linalg.norm(posed[e[:, 0]] - posed[e[:, 1]], axis=1)
    l1 = np.linalg.norm(out[e[:, 0]] - out[e[:, 1]], axis=1)
    log(label + ': settled in %d frames (%.0f s): points moved %.2f on average, %.2f at most; the cloth is %.1f%% longer '
        'along its threads on average' % (P['frames'], time.time() - t0, float(np.linalg.norm(out - posed, axis=1).mean()),
                                         float(np.linalg.norm(out - posed, axis=1).max()), 100.0 * float((l1 / np.maximum(l0, 1e-9)).mean() - 1.0)))
    # carried back through the bones to the rest shape
    Bi = np.linalg.inv(B)
    back = np.einsum('nij,nj->ni', Bi[:, :3, :3], out) + Bi[:, :3, 3]
    o.data.vertices.foreach_set('co', back.reshape(-1))
    if 'rest' in o.data.attributes:
        o.data.attributes['rest'].data.foreach_set('vector', back.reshape(-1))
    o.data.update()
    for ob in (cloth, body):
        d = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(d)
    sc.frame_end = end0
    sc.frame_set(1)
    return out

