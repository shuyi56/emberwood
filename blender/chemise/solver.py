"""The chemise's drape, found by Blender's cloth solver instead of drawn.

drape.py drafts the garment as a plain loose piece with no folds in it: that is the pattern, the cloth as cut. Here
it is put on the posed body and let go. The body (and the hands) are the collider; the cloth is given linen's
behaviour (it barely stretches, shears a little, and bends stiffly for its weight, so it breaks into a few firm
folds and not into many limp ones); gravity does the rest. What it settles to is carried back to the garment's rest
shape through the bones, so the garment still follows the rig.

The figure is 63 units tall where a woman is 1.65 m, and the solver takes a unit for a metre, so cloth of real
stiffness would hang like tissue. LINEN's numbers are for this scale: found by trial on swatch.py's bench against how a linen
sleeve breaks (folds a few centimetres across, 1 to 2 units), not copied from a preset."""
import os

import bpy
import numpy as np

LINEN = dict(
    mass=0.5,                # of one point of the cloth
    tension=120.0,           # linen barely stretches
    compression=15.0,        # but gives way at once when pushed along its threads: it folds where a stiffer sheet
                             # would stand and crumple (at 120 the gathered cloth rolled into thick, heavy folds)
    shear=20.0,              # and gives a little on the bias
    bending=0.35,            # a fine linen: soft enough to fall into narrow folds under its own weight
    damping=(8.0, 8.0, 6.0, 1.5),      # tension, compression, shear, bending
    air=0.6,
    friction=6.0,            # linen on skin
    gap=0.07,                # how far the cloth is kept off the skin, in units
    self_gap=0.015,          # and off itself (small: at 0.045 the cloth's own close points pushed each other apart,
                             # and the whole garment swelled and drew its hems in)
    quality=9,
    frames=120,
    let_out=50,              # the frames over which a fuller cloth is let out to its fullness
)
# (These were found on swatch.py's bench, not on the garment. Its gathered curtain, by the weight of a point at
# bending 0.35 and compression 15: 0.3 folds 8.7 cm wide and a little crumpled (0.18); 0.5 folds 6.5 cm wide and
# clean (0.03); 0.8 folds 5.6 cm wide but shallow. A stiffer bend (3, 10) widens the folds again.)


def weave(s, P):
    """Give a cloth modifier's settings s the cloth P's behaviour (the garment's and swatch.py's swatches' alike)."""
    s.quality = P['quality']
    s.mass = P['mass']
    s.tension_stiffness, s.compression_stiffness = P['tension'], P['compression']
    s.shear_stiffness, s.bending_stiffness = P['shear'], P['bending']
    s.tension_damping, s.compression_damping, s.shear_damping, s.bending_damping = P['damping']
    s.air_damping = P['air']


def let_out(cloth, s, full, P):
    """Let the cloth out (or draw it in) point by point: full, a share for each point (see settle)."""
    if full is None or np.ptp(full) == 0:
        return
    # (the solver's 'shrink': weight 0 the fullest cloth, weight 1 the most drawn in)
    lo, hi = -float(full.max()), -float(full.min())
    g = cloth.vertex_groups.new(name='full')
    wt = (-full - lo) / (hi - lo)
    for i in range(len(full)):
        g.add([i], float(wt[i]), 'REPLACE')
    s.vertex_group_shrink = 'full'
    # (little by little, so the cloth is not thrown: keyed, since a setting changed by hand between frames starts
    # the solver over)
    for f, k in ((1, 0.0), (P['let_out'], 1.0)):
        s.shrink_min, s.shrink_max = k * lo, k * hi
        s.keyframe_insert('shrink_min', frame=f)
        s.keyframe_insert('shrink_max', frame=f)


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


def settle(o, rig, colliders, pinned=None, full=None, P=LINEN, log=print):
    """Let the garment o (skinned to rig, in its drafted rest shape) fall on the posed colliders, and set its rest
    shape to what it settles to. pinned: a weight for each point, 1 held where it was drafted (or None). full: how
    much more cloth there is at each point than was drafted (0.3: three tenths more each way; negative: less, a
    band drawn in), or None: the cloth let out or taken in there over the first frames, as a gathered piece is
    fuller than the band it is sewn to."""
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
    weave(s, P)
    if pinned is not None and np.any(pinned > 0):
        s.vertex_group_mass = 'pin'
        s.pin_stiffness = 1.0
    let_out(cloth, s, full, P)
    c.use_collision, c.collision_quality, c.distance_min = True, 4, 0.02
    c.use_self_collision, c.self_distance_min, c.self_friction = True, P['self_gap'], 8.0
    m.point_cache.frame_start, m.point_cache.frame_end = 1, P['frames']
    end0 = sc.frame_end
    sc.frame_end = max(sc.frame_end, P['frames'])
    import time
    t0 = time.time()
    trace = os.environ.get('SOLVER_TRACE') is not None
    prev, told = posed.copy(), False
    for f in range(1, P['frames'] + 1):
        sc.frame_set(f)
        if trace and not told:
            # (where the cloth first jumps: the points that move furthest in one frame, where they started)
            cur = np.empty(n * 3)
            cloth.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices.foreach_get('co', cur)
            cur = cur.reshape(-1, 3)
            step = np.linalg.norm(cur - prev, axis=1)
            if step.max() > 0.5:
                worst = np.argsort(-step)[:25]
                log('solver trace: frame %d, step %.2f at most; worst points started at (pin):' % (f, float(step.max())))
                for i in worst:
                    log('   %6d  (%.2f, %.2f, %.2f)  pin %.2f  step %.2f' % (i, *posed[i], float(pinned[i]) if pinned is not None else 0.0, float(step[i])))
                told = True
            prev = cur
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
    log('chemise: settled in %d frames (%.0f s): points moved %.2f on average, %.2f at most; the cloth is %.1f%% longer '
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

