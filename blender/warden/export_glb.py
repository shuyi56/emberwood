"""Export the Warden as a glTF model for the 3D world (public/models/warden.glb), the way blender/ranger/export_glb.py
exports the ranger: the same figure as warden.py, a 24-frame walk cycle (frames 1-25) and the idle stance (frame 40)
on one timeline, plain materials (each part in the middle tone of its palette ramp), skinning and sampled animation.
Her head, face and hair are replaced in the world by the Face Creator's (tools/creator-head/merge-head.mjs), so the
head is exported plainly and lightly.

Run with:  python blender/warden/export_glb.py [--out path/to/warden.glb]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import warden  # noqa: E402
import female  # noqa: E402

WALK_FRAMES, IDLE_FRAME, FPS = 24, 40, 24
BAKED = ('bronze',)   # (materials whose close-up colour is baked into the model, see bake_closeup_colours)


def key_pose(sc, frame):
    for o in sc.objects:
        if o.type == 'EMPTY' and o.name != 'Warden':
            o.keyframe_insert('rotation_euler', frame=frame)
            o.keyframe_insert('location', frame=frame)


def lin(c):
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def bake_closeup_colours(mats=('bronze',), samples=16):
    """Bake the close-up shader's colour of these materials into a colour attribute ('baked') on each mesh that wears
    them, with its smoothing and thickness applied first, so the world model carries the mottle, grime and rubbed
    edges as vertex colours instead of one flat tone."""
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.render.bake.target = 'VERTEX_COLORS'
    sc.render.bake.use_pass_direct = False
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.use_pass_color = True
    # (the close-up shader is fully metallic, which has no diffuse colour to bake: feed its base colour to an
    # emission shader and bake that instead)
    for name in mats:
        m = bpy.data.materials.get(name)
        nt = m.node_tree
        pb = nt.nodes.get('close bsdf')
        out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
        src = pb.inputs['Base Color'].links[0].from_socket
        em = nt.nodes.new('ShaderNodeEmission')
        nt.links.new(src, em.inputs['Color'])
        nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    done = []
    for o in list(sc.objects):
        if o.type != 'MESH' or not any(s.material and s.material.name in mats for s in o.material_slots):
            continue
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        arm = next((m for m in o.modifiers if m.type == 'ARMATURE'), None)
        if arm:
            arm.show_viewport = arm.show_render = False
        for m in [m for m in o.modifiers if m.type in ('SOLIDIFY', 'SUBSURF')]:
            bpy.ops.object.modifier_apply(modifier=m.name)
        if arm:
            arm.show_viewport = arm.show_render = True
        ca = o.data.color_attributes.new('baked', 'BYTE_COLOR', 'POINT')
        o.data.color_attributes.active_color = ca
        t = time.time()
        bpy.ops.object.bake(type='EMIT')
        done.append((o.name, len(o.data.vertices), round(time.time() - t, 1)))
    return done


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
        p.inputs['Base Color'].default_value = (*[lin(c) for c in rgb], 1.0)
        name = d['name']
        if name in BAKED:
            vc = nt.nodes.new('ShaderNodeVertexColor')
            vc.layer_name = 'baked'
            nt.links.new(vc.outputs['Color'], p.inputs['Base Color'])
        metal = name in ('plate', 'steel', 'gold', 'collar', 'bronze')
        p.inputs['Metallic'].default_value = 0.8 if metal else 0.0
        p.inputs['Roughness'].default_value = 0.55 if name in BAKED else 0.45 if metal else (0.6 if name == 'skin' else 0.88)
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out = os.path.abspath(argv[argv.index('--out') + 1] if '--out' in argv else os.path.join(HERE, '..', '..', 'public', 'models', 'warden.glb'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    female.RES = 10000
    female.LOCKED_WALK, female.ARM_SWING = True, 0.24
    warden.build()
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
    print('baked', bake_closeup_colours(BAKED))
    sc.frame_set(1)
    plain_materials()
    for o in sc.objects:
        if o.type == 'MESH':
            for a in list(o.data.color_attributes):
                if a.name != 'baked':
                    o.data.color_attributes.remove(a)
    # (the head's own parts are replaced in the world: thin them right down)
    for o in [o for o in sc.objects if o.type == 'MESH' and o.parent and o.parent.name == 'Head']:
        d = o.modifiers.new('lighter', 'DECIMATE'); d.ratio = 0.1
    bpy.data.objects['Rig'].hide_render = False
    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_apply=True, export_yup=True,
        export_animations=True, export_force_sampling=True, export_frame_range=True,
        export_animation_mode='SCENE', export_skins=True, export_attributes=False,
        export_texcoords=True, export_materials='EXPORT', export_cameras=False, export_lights=False)
    print('wrote', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
