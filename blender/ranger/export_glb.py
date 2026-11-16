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
        if mid in (ranger.M_SLEEVE, ranger.M_LEGS):
            # the wool's weave: a twill normal map (see twill_image), laid round each sleeve by its 'UVMap'
            img = nt.nodes.new('ShaderNodeTexImage')
            img.image = twill_image()
            img.image.colorspace_settings.name = 'Non-Color'
            uv = nt.nodes.new('ShaderNodeUVMap')
            uv.uv_map = 'UVMap'
            nt.links.new(uv.outputs['UV'], img.inputs['Vector'])
            nm_ = nt.nodes.new('ShaderNodeNormalMap')
            nm_.inputs['Strength'].default_value = 0.8
            nt.links.new(img.outputs['Color'], nm_.inputs['Color'])
            nt.links.new(nm_.outputs['Normal'], p.inputs['Normal'])
        if mid in ranger.FORM_SHADED or mid == ranger.M_BOOTS:
            # the sleeves and the jerkin: their folds, the shade where they are shut in and the leather's mottling,
            # baked into the 'form' colour (see bake_cloth_form)
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = 'form'
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
            cloth = mid in (ranger.M_SLEEVE, ranger.M_LEGS)
            p.inputs['Roughness'].default_value = 0.95 if cloth else (0.6 if mid == ranger.M_BOOTS else 0.7)
            p.inputs['Specular IOR Level'].default_value = 0.15 if cloth else 0.35
            if cloth:
                p.inputs['Sheen Weight'].default_value = 0.6
                p.inputs['Sheen Roughness'].default_value = 0.5
                p.inputs['Sheen Tint'].default_value = (0.2, 0.24, 0.15, 1.0)
        if mid in (ranger.M_EYE, ranger.M_SCLERA):
            # the eye stays wet and glossy, so the world's lights leave a highlight on it; the iris and the
            # white keep their paintings (iris_image, sclera_image) as textures
            p.inputs['Roughness'].default_value = 0.06 if mid == ranger.M_EYE else 0.32
            if mid == ranger.M_EYE:
                tex = nt.nodes.new('ShaderNodeTexImage')
                tex.image = ranger.iris_image()
                nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
            else:
                tex = nt.nodes.new('ShaderNodeTexImage')
                tex.image = ranger.sclera_image()
                nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
        if mid in (ranger.M_STEEL, ranger.M_GOLD, ranger.M_MAIL):
            p.inputs['Metallic'].default_value = 0.7
            p.inputs['Roughness'].default_value = 0.45
        if mid == ranger.M_GOLD:
            # the brooch's engraving, as dark lines in its colour (the 'gold' colour; see bake_gold)
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = 'gold'
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])


def twill_image(size=256):
    """A tiling normal map of a soft wool twill: diagonal ribs (four across the tile, so at the world's distance
    each is a pixel or two wide rather than a shimmer), each rib broken into slightly uneven threads, over a gentle
    waviness of the cloth. Packed into the model."""
    import numpy as np
    im = bpy.data.images.get('twill')
    if im is not None:
        return im
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:size, 0:size] / size
    t = 4 * (x + y)
    rib = np.abs(np.sin(np.pi * t)) ** 0.7
    # threads: each rib cut into short lumps along its length, and every lump a little different in height
    lump = 0.5 + 0.5 * np.cos(2 * np.pi * 12 * (x - y))
    jitter = rng.random((48, 48))
    jitter = np.kron(jitter, np.ones((size // 48 + 1, size // 48 + 1)))[:size, :size]
    h = rib * (0.75 + 0.25 * lump) * (0.85 + 0.3 * jitter)
    # the cloth's waviness: two broad tiling swells
    h += 0.35 * np.sin(2 * np.pi * (x + 2 * y)) * np.sin(2 * np.pi * (2 * x - y))
    k = 1.6
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * k
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * k
    n = np.dstack([-gx, gy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    rgba = np.concatenate([n * 0.5 + 0.5, np.ones((size, size, 1))], axis=2).astype(np.float32)
    im = bpy.data.images.new('twill', size, size, alpha=False)
    im.pixels.foreach_set(rgba[::-1].ravel())
    im.file_format = 'PNG'
    im.pack()
    return im


def sleeve_uvs():
    """Lay a 'UVMap' round each sleeve from its rest position: u runs round the arm (six tiles round, with the
    seam on the inside of the arm, against the body), v down it."""
    for name, side, s in [(n, sd, s_) for n in ('Sleeve', 'Leg') for sd, s_ in (('R', 1), ('L', -1))]:
        o = bpy.data.objects.get(f'{name}.{side}')
        if o is None:
            continue
        me = o.data
        cx, cy = (-s * ranger.SH_X, 0.2) if name == 'Sleeve' else (-s * ranger.LEG_X, 0.0)
        rest = me.attributes['rest'].data
        uvl = me.uv_layers.new(name='UVMap')
        for loop in me.loops:
            q = rest[loop.vertex_index].vector
            a = math.atan2(q.y - cy, -(q.x - cx) * s)
            uvl.data[loop.index].uv = ((a / (2 * math.pi) + 0.5) * 6.0, q.z / 2.0)


def bake_hair_shade(rays=10, dist=1.6):
    """The world's renderer has no ambient occlusion of its own, so the shade where the hair is shut in (between
    locks, under the fall, where it meets her back and the mantle) is baked into its strand colours: from each
    point, how many rays out over its surface meet hair or body within a short distance."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    import numpy as np
    bpy.context.scene.frame_set(IDLE_FRAME)    # (the points below are the rest shape, which is how she stands at idle)
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
    bpy.context.scene.frame_set(IDLE_FRAME)    # (the points below are the rest shape, which is how she stands at idle)
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
    targets = [o for o in sc.objects if o.type == 'MESH' and (o.name == 'Jerkin' or o.name.split('.')[0] in ('Sleeve', 'Leg', 'Boot'))]
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
        mid = {'Jerkin': ranger.M_JERKIN, 'Leg': ranger.M_LEGS, 'Boot': ranger.M_BOOTS}.get(o.name.split('.')[0], ranger.M_SLEEVE)
        rmp = ranger.MATERIALS[mid]['ramp'] / 255.0
        lin = np.where(rmp > 0.04045, ((rmp + 0.055) / 1.055) ** 2.4, rmp / 12.92)
        dark, midc, light = lin[1], lin[2], lin[3]
        crease = np.zeros(len(me.vertices), dtype=np.float32)
        if 'crease' in me.attributes:
            me.attributes['crease'].data.foreach_get('value', crease)
        rest = me.attributes['rest'].data
        solid = o.modifiers.get('thickness')
        lift = solid.thickness if solid else 0.0
        cols = []
        for (p, n), r, cr in zip(posed, rest, crease):
            q = r.vector
            if mid in (ranger.M_JERKIN, ranger.M_BOOTS):
                t = 0.65 * noise.noise(q * 0.32) + 0.35 * noise.noise(q * 1.1 + Vector((2.3, 5.1, 0.7)))
                t = min(1.0, max(-1.0, 1.6 * t))
                c = midc + (light - midc) * t * 0.5 if t > 0 else midc + (midc - dark) * t * 0.5
            else:
                t = 0.7 * noise.noise(q * 0.45) + 0.3 * noise.noise(q * 1.3 + Vector((4.1, 0.3, 2.2)))
                t = min(1.0, max(-1.0, 1.5 * t))
                c = midc + (light - midc) * 0.4 * t if t > 0 else midc + (midc - dark) * 0.4 * t
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
            if mid == ranger.M_BOOTS:
                # the foot darkened by mud, low on it above the welt, with a clean edge
                tide = q.z + 0.5 * (0.5 + 0.5 * noise.noise(q * 1.1))
                c = c * (1.0 - 0.3 * min(1.0, max(0.0, (1.75 - tide) / 0.2)))
            cols += [*c, 1.0]
        me.color_attributes.new('form', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', cols)
    sc.frame_set(1)


def bake_hands(rays=24, dist=1.3):
    """The hands for the world, which would otherwise be one flat skin tone: a material of their own ('hand skin')
    coloured by the vertex colour 'handskin' (linear RGB). Each point takes the shade of how shut in it is (rays
    over its surface that meet the hand, the sleeve or the body within a short distance: between the fingers, in
    the curl of the palm, under the cuff), the blood-warm flush of the knuckles and the fingertips, and the paler
    skin over the back of the hand. Cast in her idle stance, as for the cloth."""
    from mathutils import Vector
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
    rng = np.random.default_rng(13)
    dirs = rng.normal(size=(rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]

    def lin(c):
        c = np.asarray(c, dtype=np.float64) / 255.0
        return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    S = ranger.SKIN
    dark, mid, light = lin(S[1]), lin(S[2]), lin(S[3])
    flush = lin([214, 140, 118])
    skin_mat = bpy.data.materials['skin']
    hand_mat = skin_mat.copy()
    hand_mat.name = 'hand skin'
    nt = hand_mat.node_tree
    p = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    col = nt.nodes.new('ShaderNodeVertexColor')
    col.layer_name = 'handskin'
    nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
    p.inputs['Roughness'].default_value = 0.62
    p.inputs['Specular IOR Level'].default_value = 0.3
    p.inputs['Sheen Weight'].default_value = 0.5
    p.inputs['Sheen Roughness'].default_value = 0.4
    p.inputs['Sheen Tint'].default_value = (0.6, 0.3, 0.25, 1.0)
    for side in ('R', 'L'):
        o = bpy.data.objects[f'Hand.{side}']
        me = o.data
        me.materials[0] = hand_mat
        pm = o.evaluated_get(dg).to_mesh()
        nmx = o.matrix_world.to_3x3().inverted().transposed()
        posed = [(o.matrix_world @ v.co, (nmx @ v.normal).normalized()) for v in pm.vertices]
        o.evaluated_get(dg).to_mesh_clear()
        cols = []
        for v, (pp, n) in zip(me.vertices, posed):
            q = v.co                                  # (the hand's own frame: wrist at the top, fingers down)
            hit = 0.0
            for d in dirs:
                d = Vector(d)
                if d.dot(n) < 0:
                    d = -d
                d = (d + n).normalized()
                h = tree.ray_cast(pp + n * 0.04, d, dist)
                if h[0] is not None:
                    hit += 1.0 - 0.6 * h[3] / dist
            ao = hit / rays
            c = mid.copy()
            # the back of the hand a shade paler, the palm side a shade warmer
            back = max(0.0, min(1.0, 0.5 + 0.8 * v.normal.x * (1 if side == 'R' else -1) * -1))
            c = c + (light - c) * 0.35 * back
            # flushed knuckles and fingertips
            knuckle = math.exp(-((q.z + 2.55) / 0.45) ** 2)
            tip = min(1.0, max(0.0, (-q.z - 3.2) / 0.8))
            c = c + (flush - c) * min(1.0, 0.7 * knuckle + 0.8 * tip)
            # the clefts between the fingers (the fingers are one fused surface, so the shade marks them apart),
            # from the knuckles to the tips
            fz = min(1.0, max(0.0, (-q.z - 2.4) / 0.5))
            gap = max(math.exp(-((q.y - g) / 0.11) ** 2) for g in (-0.56, 0.0, 0.5))
            c = c + (dark * 0.75 - c) * 0.6 * gap * fz
            # the palm side, turned to the body and away from the light, a shade deeper; up at the wrist the skin
            # falls into the shade of the cuff
            c = c + (dark - c) * (0.35 * (1.0 - back) + 0.5 * min(1.0, max(0.0, (q.z + 0.2) / 1.0)))
            c = c * (1.0 - 0.75 * ao)
            c = c + (dark - c) * 0.4 * ao
            cols += [*c, 1.0]
        me.color_attributes.new('handskin', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', cols)
    sc.frame_set(1)


def main():
    argv = sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else os.path.join(HERE, '..', '..', 'public', 'models', 'ranger.glb')
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ranger.RES = 10000            # no snapping of the bob to sheet pixels: the model moves smoothly
    # in the world she covers ground at a good pace, so the walk takes a longer stride than the sprite sheet's and
    # keeps the arm swing small (it has to read from the zoomed-out game camera). WORLD_CYCLE in src/world/main.js is the ground one cycle covers (see stride_length).
    ranger.GAIT, ranger.ARM_SWING = 1.3, 0.24
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
    sleeve_uvs()
    plain_materials()
    bake_hair_shade()
    bake_wool()
    bake_cloth_form()
    bake_hands()
    bake_gold()
    # the fused hair is a fine voxel surface, far denser than the world needs: thin it out
    # and the head is cut very fine round the eyes for the sprites' sculpted lids: the model needs far less
    for name, ratio in (('Hair', 0.11), ('Head shape', 0.4)):
        d = bpy.data.objects[name].modifiers.new('lighter', 'DECIMATE')
        d.ratio = ratio
    # the solid brows are for the sprites only: the model has the painted ones
    for o in [o for o in bpy.data.objects if o.name.startswith('Brow tube')]:
        bpy.data.objects.remove(o, do_unlink=True)
    # and the solid line between the lips likewise: the model's lip paint draws it
    bpy.data.objects.remove(bpy.data.objects['Mouth'], do_unlink=True)
    # the rig's armature is hidden for the sprite renders; the exporter needs it in the scene to write the skin
    bpy.data.objects['Rig'].hide_render = False

    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_apply=True, export_yup=True,
        export_animations=True, export_force_sampling=True, export_frame_range=True,
        export_animation_mode='SCENE', export_skins=True, export_attributes=False,
        export_texcoords=True, export_materials='EXPORT', export_cameras=False, export_lights=False)
    print('wrote', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
