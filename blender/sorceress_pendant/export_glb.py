"""Export the Sorceress (sorceress_pendant.py: the gown with the gold chain and garnet pendant) as a glTF model for the
3D world, the way blender/warden/export_glb.py exports the Warden: 24-frame walk (frames 1-25) and the idle (frame 40)
on one timeline, plain materials, skinning and sampled animation. Her head is the Face Creator's, merged in by
tools/creator-head/merge-head.mjs, so no head is built here (HEAD_GLB='').

    NECK_DROP=1.6 CLAV=2.0 python3 export_glb.py --out sorceress-blender.glb
"""
import os
import sys

os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NECK_DROP', '1.4')
os.environ.setdefault('CLAV', '1.0')
os.environ.setdefault('LENGTH', '17.5')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
sys.path.insert(0, os.path.join(HERE, '..', 'gown'))
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as S  # noqa: E402

WALK_FRAMES, IDLE_FRAME, FPS = 24, 40, 24


def key_pose(sc, frame):
    for o in sc.objects:
        if o.type == 'EMPTY' and o.name != 'Gown':
            o.keyframe_insert('rotation_euler', frame=frame)
            o.keyframe_insert('location', frame=frame)


def lin(c):
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def plain_materials():
    for mid, d in female.MATERIALS.items():
        m = bpy.data.materials.get(d['name'])
        if m is None or not m.use_nodes:
            continue
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        p = nt.nodes.new('ShaderNodeBsdfPrincipled')
        rgb = d['ramp'][2] / 255.0
        if d['name'] == 'skin':
            # (her skin in the creator face's own tone, as the preview has it, so the merge leaves the face as it is)
            hx = os.environ.get('BODY_SKIN', 'f0d1cb')
            rgb = [int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)]
            # and, as the ranger's world model has it, the baked 'skin' tone as the vertex colour: the flush on the
            # knuckles, shoulders and chest, the cooler shade under the jaw and the warm shade where the skin is
            # shut in, which stand in for the close-up skin's scattering (the world has none)
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = 'skin'
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
            p.inputs['Specular IOR Level'].default_value = 0.35
        p.inputs['Base Color'].default_value = (*[lin(c) for c in rgb], 1.0)
        name = d['name']
        metal = name.startswith('jewel') or name in ('plate', 'steel', 'gold', 'collar')
        p.inputs['Metallic'].default_value = 0.8 if metal else 0.0
        p.inputs['Roughness'].default_value = 0.45 if metal else (0.6 if name == 'skin' else (0.35 if name == 'satin' else 0.88))
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out = os.path.abspath(argv[argv.index('--out') + 1] if '--out' in argv else 'sorceress-blender.glb')
    female.RES = 10000
    female.LOCKED_WALK, female.ARM_SWING = True, 0.24
    S.build()
    sc = bpy.context.scene
    for o in sc.objects:
        if o.animation_data:
            o.animation_data_clear()
    for k in range(WALK_FRAMES + 1):
        female.pose(2 - 0.5 + female.WALK + k * female.WALK / WALK_FRAMES)
        key_pose(sc, 1 + k)
    female.pose(0)
    key_pose(sc, IDLE_FRAME)
    sc.frame_start, sc.frame_end = 1, IDLE_FRAME
    sc.render.fps = FPS
    # (the idle, not the walk's first frame: export_apply bakes the modifiers as they stand, and at a walk frame the
    # legs' proxies had the skirt pushed out and the panel wrapped onto that, up to 2.7 off the skirt at rest; in the
    # world it hung in front of the skirt like a loose strap)
    sc.frame_set(IDLE_FRAME)
    plain_materials()
    for o in sc.objects:
        if o.type == 'MESH':
            for a in list(o.data.color_attributes):
                if a.name == 'skin' and o.data.materials and o.data.materials[0] and o.data.materials[0].name == 'skin':
                    continue
                o.data.color_attributes.remove(a)
    for o in [o for o in sc.objects if o.type == 'MESH' and o.parent and o.parent.name == 'Head']:
        d = o.modifiers.new('lighter', 'DECIMATE')
        d.ratio = 0.1
    # (her legs are left out of the game model: the floor-length skirt covers them, and a glb has no "unseen by the
    # camera", so in the world's longer stride a thigh showed through the satin; the shoes stay)
    for nm in ('Leg.L', 'Leg.R', 'Shoe.L', 'Shoe.R'):
        if nm in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[nm])
    if 'Rig' in bpy.data.objects:
        bpy.data.objects['Rig'].hide_render = False
    # (the panel on the skirt: in Blender a Shrinkwrap lays it on the skirt wherever the walk has moved it, which a
    # glb does not carry, and baked by export_apply it came out up to 2.7 off the skirt. Laid on the skirt's rest
    # shape here, a little above it; with the skirt's own weights it then rides on it through the walk)
    pn, sk = bpy.data.objects.get('Panel'), bpy.data.objects.get('Skirt')
    if pn is not None and sk is not None:
        from mathutils.bvhtree import BVHTree
        from mathutils import Vector
        for m in [m for m in pn.modifiers if m.type == 'SHRINKWRAP']:
            pn.modifiers.remove(m)
        me = sk.data
        tree = BVHTree.FromPolygons([sk.matrix_world @ v.co for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        inv = pn.matrix_world.inverted()
        for v in pn.data.vertices:
            p = pn.matrix_world @ v.co
            hit, nrm, _i, _d = tree.find_nearest(p)
            if hit is None:
                continue
            away = Vector((hit.x, hit.y - 0.2, 0.0))
            if nrm.dot(away) < 0:
                nrm = -nrm
            v.co = inv @ (hit + nrm * float(os.environ.get('PANEL_LIFT', '0.14')))
        pn.data.update()
    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_apply=True, export_yup=True,
        export_animations=True, export_force_sampling=True, export_frame_range=True,
        export_animation_mode='SCENE', export_skins=True, export_attributes=False,
        export_texcoords=True, export_materials='EXPORT', export_cameras=False, export_lights=False)
    print('wrote', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
