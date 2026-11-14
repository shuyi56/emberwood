"""Export the ranger as a glTF model for the 3D world (public/models/ranger.glb).

Builds the same figure as ranger.py, then keys a smooth walk cycle (24 frames, 1-25, the last equal to the first)
and the idle stance (frame 40) onto one timeline, swaps the render materials for plain ones a real-time renderer
understands (each part in the middle tone of its palette ramp), and writes a binary glTF with its skinning and the
sampled animation. The world plays the walk by looping frames 1-25 and holds frame 40 when she stands.

Run with:  python blender/ranger/export_glb.py [--out path/to/ranger.glb]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import ranger  # noqa: E402

WALK_FRAMES = 24
IDLE_FRAME = 40
FPS = 24


def key_pose(sc, frame):
    for o in sc.objects:
        if o.type == 'EMPTY' and o.name != 'Ranger':
            o.keyframe_insert('rotation_euler', frame=frame)
            o.keyframe_insert('location', frame=frame)


def plain_materials():
    """Principled materials in each part's middle palette tone (sRGB -> linear), matte, with the hair a touch
    glossier: the procedural textures and AOVs of the sprite renderer mean nothing to glTF."""
    for mid, d in ranger.MATERIALS.items():
        m = bpy.data.materials.get(d['name'])
        if m is None:
            continue
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        p = nt.nodes.new('ShaderNodeBsdfPrincipled')
        rgb = d['ramp'][2] / 255.0
        lin = [((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in rgb]
        p.inputs['Base Color'].default_value = (*lin, 1.0)
        p.inputs['Roughness'].default_value = 0.9
        if mid == ranger.M_HAIR:
            # hair: each strand its own tone (the 'strands' colour, exported as the vertex colour, with the shade
            # where the hair is shut in baked into it), a soft warm sheen instead of a white highlight, and only a
            # little gloss, so it reads as hair and not as a moulded, glossy shell
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = 'strands'
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
            p.inputs['Roughness'].default_value = 0.72
            p.inputs['Specular IOR Level'].default_value = 0.25
            p.inputs['Sheen Weight'].default_value = 1.0
            p.inputs['Sheen Roughness'].default_value = 0.5
            p.inputs['Sheen Tint'].default_value = (0.42, 0.22, 0.13, 1.0)
        if mid == ranger.M_MANTLE:
            # the wool: its felted patches, the fluffed-up hem and neck and the shade where it is shut in, baked
            # into the 'wool' colour (see bake_wool), fully matte, with a soft sheen at the edges where the
            # renderer has one
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = 'wool'
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
            p.inputs['Roughness'].default_value = 1.0
            p.inputs['Specular IOR Level'].default_value = 0.1
            p.inputs['Sheen Weight'].default_value = 1.0
            p.inputs['Sheen Roughness'].default_value = 0.6
            p.inputs['Sheen Tint'].default_value = (0.2, 0.25, 0.13, 1.0)
        if mid in ranger.FORM_SHADED:
            # the sleeves and the jerkin: their folds, the shade where they are shut in and the leather's mottling,
            # baked into the 'form' colour (see bake_cloth_form)
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = 'form'
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
            p.inputs['Roughness'].default_value = 0.95 if mid == ranger.M_SLEEVE else 0.7
            p.inputs['Specular IOR Level'].default_value = 0.15 if mid == ranger.M_SLEEVE else 0.35
            if mid == ranger.M_SLEEVE:
                p.inputs['Sheen Weight'].default_value = 0.6
                p.inputs['Sheen Roughness'].default_value = 0.5
                p.inputs['Sheen Tint'].default_value = (0.2, 0.24, 0.15, 1.0)
        if mid in (ranger.M_STEEL, ranger.M_GOLD, ranger.M_MAIL):
            p.inputs['Metallic'].default_value = 0.7
            p.inputs['Roughness'].default_value = 0.45
        if mid == ranger.M_GOLD:
            # the brooch's engraving, as dark lines in its colour (the 'gold' colour; see bake_gold)
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = 'gold'
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])


def bake_hair_shade(rays=10, dist=1.6):
    """The world's renderer has no ambient occlusion of its own, so the shade where the hair is shut in (between
    locks, under the fall, where it meets her back and the mantle) is baked into its strand colours: from each
    point, how many rays out over its surface meet hair or body within a short distance."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get()
    vs, ps = [], []
    for o in bpy.context.scene.objects:
        if o.type != 'MESH':
            continue
        me = o.evaluated_get(dg).to_mesh()
        base = len(vs)
        vs += [o.matrix_world @ v.co for v in me.vertices]
        ps += [[base + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    tree = BVHTree.FromPolygons(vs, ps)
    hair = bpy.data.objects['Hair']
    mw = hair.matrix_world
    nm = mw.to_3x3().inverted().transposed()
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    attr = hair.data.color_attributes['strands']
    cols = np.zeros(len(attr.data) * 4, dtype=np.float32)
    attr.data.foreach_get('color', cols)
    cols = cols.reshape(-1, 4)
    for i, v in enumerate(hair.data.vertices):
        p = mw @ v.co
        n = (nm @ v.normal).normalized()
        hit = 0
        for d in dirs:
            d = Vector(d)
            if d.dot(n) < 0:
                d = -d
            d = (d + n).normalized()       # (cosine-leaning: rays nearer the normal count most)
            if tree.ray_cast(p + n * 0.05, d, dist)[0] is not None:
                hit += 1
        cols[i, :3] *= 1.0 - 0.55 * hit / rays
    attr.data.foreach_set('color', cols.ravel())


def bake_gold():
    """The brooch's engraving for the world: its 'etch' depth turned into a vertex colour, the gold's middle
    tone darkening to tarnish in the cut lines."""
    import numpy as np
    o = bpy.data.objects['Leaf clasp']
    me = o.data
    rgb = ranger.MATERIALS[ranger.M_GOLD]['ramp'][2] / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    dark = np.array([0.05, 0.03, 0.012])
    e = np.zeros(len(me.vertices), dtype=np.float32)
    me.attributes['etch'].data.foreach_get('value', e)
    k = np.clip((e - 0.2) / 0.4, 0, 1)[:, None] * 0.85
    cols = np.concatenate([lin * (1 - k) + dark * k, np.ones((len(e), 1))], axis=1).astype(np.float32)
    me.color_attributes.new('gold', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', cols.ravel())


def bake_wool(rays=12, dist=1.4):
    """The mantle's wool for the world, baked into its vertex colour 'wool' (linear RGB): soft felted patches
    between the darker and lighter tones of its ramp, the cut edges (hem and neck) a shade lighter where the
    wool is fulled up, and the shade where it is shut in (round the brooch, in the creases, against the neck)
    from rays cast over the outside of the cloth, as for the hair."""
    from mathutils import Vector, noise
    from mathutils.bvhtree import BVHTree
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get()
    vs, ps = [], []
    for o in bpy.context.scene.objects:
        if o.type != 'MESH':
            continue
        me = o.evaluated_get(dg).to_mesh()
        base = len(vs)
        vs += [o.matrix_world @ v.co for v in me.vertices]
        ps += [[base + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    tree = BVHTree.FromPolygons(vs, ps)
    mantle = bpy.data.objects['Mantle']
    me = mantle.data
    rmp = ranger.MATERIALS[ranger.M_MANTLE]['ramp'] / 255.0
    lin = np.where(rmp > 0.04045, ((rmp + 0.055) / 1.055) ** 2.4, rmp / 12.92)
    dark, mid, light = lin[1], lin[2], lin[3]
    rest = me.attributes['rest'].data
    edge = me.attributes['edge'].data
    mw = mantle.matrix_world
    nm = mw.to_3x3().inverted().transposed()
    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    attr = me.color_attributes.new('wool', 'FLOAT_COLOR', 'POINT')
    cols = []
    for v, r, e in zip(me.vertices, rest, edge):
        q = r.vector
        t = 0.7 * noise.noise(q * 0.55) + 0.3 * noise.noise(q * 1.7 + Vector((3.1, 0.4, 7.7)))
        t = min(1.0, max(-1.0, 1.5 * t))
        c = mid + (light - mid) * t * 0.6 if t > 0 else mid + (mid - dark) * t * 0.6
        c = c + (light - c) * 0.4 * e.value
        p = mw @ v.co
        n = (nm @ v.normal).normalized()
        p = p + n * 0.5          # (the outside of the cloth: the thickness is laid on along the normal)
        hit = 0
        for d in dirs:
            d = Vector(d)
            if d.dot(n) < 0:
                d = -d
            d = (d + n).normalized()
            if tree.ray_cast(p + n * 0.05, d, dist)[0] is not None:
                hit += 1
        c = c * (1.0 - 0.45 * hit / rays)
        cols += [*c, 1.0]
    attr.data.foreach_set('color', cols)


def bake_cloth_form(rays=16, dist=2.4):
    """The sleeves and the jerkin for the world, baked into the vertex colour 'form' (linear RGB). The world has
    no ambient occlusion and no crease shading of its own, so the cloth would read as flat tubes: here each point
    takes the shade of how shut in it is (rays cast over its surface that meet the body within a short distance:
    under the mantle and the belt, in the armpits, between the arms and the body, in the folds), a little more in
    the creases of its folds (the 'crease' attribute), a lift on the crests, and, on the leather, broad soft patches
    of tone. Cast against the figure in her idle stance (the arms held a little off the body), which is how she
    stands in the world and close to how she passes through the walk."""
    from mathutils import Vector, noise
    from mathutils.bvhtree import BVHTree
    import numpy as np
    sc = bpy.context.scene
    sc.frame_set(IDLE_FRAME)
    dg = bpy.context.evaluated_depsgraph_get()
    vs, ps = [], []
    for o in sc.objects:
        if o.type != 'MESH':
            continue
        me = o.evaluated_get(dg).to_mesh()
        base = len(vs)
        vs += [o.matrix_world @ v.co for v in me.vertices]
        ps += [[base + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(dg).to_mesh_clear()
    tree = BVHTree.FromPolygons(vs, ps)
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    targets = [o for o in sc.objects if o.type == 'MESH' and o.name == 'Jerkin' or o.name.startswith('Sleeve.')]
    for o in targets:
        # each point where it stands in the pose: the mesh deformed by the rig alone, so its points still match
        # the mesh's own (the smoothing and thickness modifiers are held off while it is read)
        held = [m for m in o.modifiers if m.type in ('SUBSURF', 'SOLIDIFY')]
        for m in held:
            m.show_viewport = False
        dg = bpy.context.evaluated_depsgraph_get()
        pm = o.evaluated_get(dg).to_mesh()
        posed = [(o.matrix_world @ v.co, (o.matrix_world.to_3x3().inverted().transposed() @ v.normal).normalized())
                 for v in pm.vertices]
        o.evaluated_get(dg).to_mesh_clear()
        for m in held:
            m.show_viewport = True
        me = o.data
        mid = ranger.M_JERKIN if o.name == 'Jerkin' else ranger.M_SLEEVE
        rmp = ranger.MATERIALS[mid]['ramp'] / 255.0
        lin = np.where(rmp > 0.04045, ((rmp + 0.055) / 1.055) ** 2.4, rmp / 12.92)
        dark, midc, light = lin[1], lin[2], lin[3]
        crease = np.zeros(len(me.vertices), dtype=np.float32)
        me.attributes['crease'].data.foreach_get('value', crease)
        rest = me.attributes['rest'].data
        solid = o.modifiers.get('thickness')
        lift = solid.thickness if solid else 0.0
        cols = []
        for (p, n), r, cr in zip(posed, rest, crease):
            q = r.vector
            if mid == ranger.M_JERKIN:
                t = 0.65 * noise.noise(q * 0.32) + 0.35 * noise.noise(q * 1.1 + Vector((2.3, 5.1, 0.7)))
                t = min(1.0, max(-1.0, 1.6 * t))
                c = midc + (light - midc) * t * 0.5 if t > 0 else midc + (midc - dark) * t * 0.5
            else:
                t = noise.noise(q * 0.4)
                c = midc + (light - midc) * 0.25 * t
            p = p + n * (lift + 0.02)
            hit = 0.0
            for d in dirs:
                d = Vector(d)
                if d.dot(n) < 0:
                    d = -d
                d = (d + n).normalized()
                h = tree.ray_cast(p + n * 0.05, d, dist)
                if h[0] is not None:
                    hit += 1.0 - 0.5 * h[3] / dist          # (nearer cover shades more)
            ao = hit / rays
            c = c * (1.0 - 0.6 * ao) * (1.0 - 0.3 * cr)
            cols += [*c, 1.0]
        me.color_attributes.new('form', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', cols)
    sc.frame_set(1)


def main():
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else os.path.join(HERE, '..', '..', 'public', 'models', 'ranger.glb')
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ranger.RES = 10000            # no snapping of the bob to sheet pixels: the model moves smoothly
    ranger.build()
    sc = bpy.context.scene
    for o in sc.objects:
        if o.animation_data:
            o.animation_data_clear()
    # the walk: one full cycle (pose() reads a column; a whole cycle past the first walk column keeps it walking)
    for k in range(WALK_FRAMES + 1):
        ranger.pose(2 - 0.5 + ranger.WALK + k * ranger.WALK / WALK_FRAMES)
        key_pose(sc, 1 + k)
    ranger.pose(0)
    key_pose(sc, IDLE_FRAME)
    sc.frame_start, sc.frame_end = 1, IDLE_FRAME
    sc.render.fps = FPS
    sc.frame_set(1)
    # the loose fibres standing off the wool are see-through shells for the close-up renders: not for the world
    for o in [o for o in sc.objects if o.name.startswith('Wool fuzz')]:
        bpy.data.objects.remove(o)
    plain_materials()
    bake_hair_shade()
    bake_wool()
    bake_cloth_form()
    bake_gold()
    # the fused hair is a fine voxel surface, far denser than the world needs: thin it out
    for name, ratio in (('Hair', 0.11),):
        d = bpy.data.objects[name].modifiers.new('lighter', 'DECIMATE')
        d.ratio = ratio
    # the rig's armature is hidden for the sprite renders; the exporter needs it in the scene to write the skin
    bpy.data.objects['Rig'].hide_render = False

    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_apply=True, export_yup=True,
        export_animations=True, export_force_sampling=True, export_frame_range=True,
        export_animation_mode='SCENE', export_skins=True, export_attributes=False,
        export_texcoords=False, export_materials='EXPORT', export_cameras=False, export_lights=False)
    print('wrote', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
