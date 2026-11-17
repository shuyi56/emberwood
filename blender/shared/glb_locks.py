"""For the GLB exports: keep the lock-built hair on top of the hair and head once they are thinned out.

The exports decimate the fused hair (to 11%) and the head (to 40%). A decimated surface no longer passes where the
full one did: over the round of the skull it rises, and the strands that lay a few hundredths over it in the render
(the fringe's lower layer, the veils, the strand ends falling back to the skin) end up under it, so in the model
whole layers are gone and the skin shows between the strands that are left. Each lock vertex that was outside the
full surface is moved out by as much as the thinned surface rose under it, so it keeps its clearance. (Blender only.)
"""
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HC = Vector((0.0, 0.5, 4.8))


def _tree(names, dg):
    vs, ps = [], []
    for n in names:
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        me = o.evaluated_get(dg).to_mesh()
        base = len(vs)
        vs += [o.matrix_world @ v.co for v in me.vertices]
        ps += [[base + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    return BVHTree.FromPolygons(vs, ps)


def _radius(tree, c, d):
    hit = tree.ray_cast(c + d * 30.0, -d, 30.0)
    return (hit[0] - c).length if hit[0] is not None else None


def lift_locks(lock_names, surface_names, decimate='lighter', margin=0.006):
    """Call once the decimate modifiers (named `decimate`) are on the surfaces, before exporting."""
    mods = [o.modifiers[decimate] for n in surface_names for o in [bpy.data.objects.get(n)] if o and decimate in o.modifiers]
    for m in mods:
        m.show_viewport = False
    bpy.context.view_layer.update()
    full = _tree(surface_names, bpy.context.evaluated_depsgraph_get())
    for m in mods:
        m.show_viewport = True
    bpy.context.view_layer.update()
    thin = _tree(surface_names, bpy.context.evaluated_depsgraph_get())
    for n in lock_names:
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        mw, inv = o.matrix_world, o.matrix_world.inverted()
        c = mw @ HC          # (the locks are built in head-local coordinates, where HC is the skull's centre)
        moved = 0
        for v in o.data.vertices:
            p = mw @ v.co
            d = (p - c).normalized()
            r = (p - c).length
            r0, r1 = _radius(full, c, d), _radius(thin, c, d)
            if r0 is None or r1 is None or r < r0 - 0.01:      # (roots tucked under the hair stay there)
                continue
            if r1 + margin > r:
                v.co = inv @ (c + d * (r1 + margin + max(0.0, r - r0)))
                moved += 1
        o.data.update()
        print(f'glb: {n}: lifted {moved} of {len(o.data.vertices)} vertices over the thinned hair and head', flush=True)
