"""Export the rogue as a glTF model for the 3D world (public/models/rogue.glb).

The same scheme as blender/ranger/export_glb.py: build the figure, key one smooth walk cycle (24 frames, 1-25, the
last equal to the first) and the idle stance (frame 40), swap the render materials for plain ones a real-time
renderer understands, and write a binary glTF with its skin and the sampled animation. The bakes are lighter
than the ranger's: the skin keeps its baked 'skin' colour and the hair its strand colours, and the cloth takes the
middle tone of its ramp.

Run with:  python blender/rogue/export_glb.py [--out path/to/rogue.glb]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import rogue  # noqa: E402
import female  # noqa: E402  (the base figure, on the path once rogue is imported)

WALK_FRAMES = 24
IDLE_FRAME = 40
FPS = 24

# (roughness, specular, metallic) by surface; anything not listed is matte cloth
FINISH = {
    rogue.M_SKIN: (0.6, 0.35, 0.0), rogue.M_HAIR: (0.72, 0.25, 0.0), rogue.M_STEEL: (0.45, 0.5, 0.7),
    rogue.M_BODICE: (0.6, 0.35, 0.0), rogue.M_BOOTS: (0.6, 0.35, 0.0), rogue.M_BELT: (0.6, 0.35, 0.0),
    rogue.M_BIND: (0.65, 0.3, 0.0), rogue.M_BRACER: (0.75, 0.25, 0.0), rogue.M_EYE: (0.06, 0.5, 0.0),
    rogue.M_SCLERA: (0.32, 0.5, 0.0),
}
VERTEX_COLOUR = {rogue.M_SKIN: 'skin', rogue.M_HAIR: 'strands'}


def key_pose(sc, frame):
    for o in sc.objects:
        if o.type == 'EMPTY' and o.name != 'Rogue':
            o.keyframe_insert('rotation_euler', frame=frame)
            o.keyframe_insert('location', frame=frame)


def _lin(rgb):
    return [((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in rgb]


def plain_materials():
    """Principled materials in each part's middle palette tone (sRGB -> linear); the skin and the hair coloured by
    their baked vertex colours, the eyes by their paintings."""
    for mid, d in rogue.MATERIALS.items():
        m = bpy.data.materials.get(d['name'])
        if m is None:
            continue
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        p = nt.nodes.new('ShaderNodeBsdfPrincipled')
        p.inputs['Base Color'].default_value = (*_lin(d['ramp'][2] / 255.0), 1.0)
        rough, spec, metal = FINISH.get(mid, (0.92, 0.15, 0.0))
        p.inputs['Roughness'].default_value = rough
        p.inputs['Specular IOR Level'].default_value = spec
        p.inputs['Metallic'].default_value = metal
        if mid in VERTEX_COLOUR:
            col = nt.nodes.new('ShaderNodeVertexColor')
            col.layer_name = VERTEX_COLOUR[mid]
            nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
        if mid == rogue.M_HAIR:
            p.inputs['Sheen Weight'].default_value = 1.0
            p.inputs['Sheen Roughness'].default_value = 0.5
            p.inputs['Sheen Tint'].default_value = (0.36, 0.22, 0.16, 1.0)
        if mid in (rogue.M_EYE, rogue.M_SCLERA):
            tex = nt.nodes.new('ShaderNodeTexImage')
            tex.image = rogue.iris_image() if mid == rogue.M_EYE else rogue.sclera_image()
            nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out = argv[argv.index('--out') + 1] if '--out' in argv else os.path.join(HERE, '..', '..', 'public', 'models', 'rogue.glb')
    out = os.path.abspath(out)
    if not out.endswith('.glb'):            # (an --out directory, as the render queue's views pass it)
        out = os.path.join(out, 'rogue.glb')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # the world's walk, as for the ranger: no snapping to sheet pixels, feet planted, a small arm swing
    female.RES = 10000
    female.LOCKED_WALK, female.ARM_SWING = True, 0.24
    rogue.build()
    sc = bpy.context.scene
    for o in sc.objects:
        if o.animation_data:
            o.animation_data_clear()
    for k in range(WALK_FRAMES + 1):
        rogue.pose(2 - 0.5 + rogue.WALK + k * rogue.WALK / WALK_FRAMES)
        key_pose(sc, 1 + k)
    rogue.pose(0)
    key_pose(sc, IDLE_FRAME)
    sc.frame_start, sc.frame_end = 1, IDLE_FRAME
    sc.render.fps = FPS
    sc.frame_set(1)

    # what is only for the close-up and sprite renders: the see-through wool fuzz shells, the solid brows (the
    # model has the painted ones) and the solid line between the lips (the lip paint draws it)
    for o in list(sc.objects):
        mats = [s.material.name for s in getattr(o, 'material_slots', []) if s.material]
        if 'wool fuzz' in mats or o.name.startswith('Brow tube') or o.name == 'Mouth':
            bpy.data.objects.remove(o, do_unlink=True)
    plain_materials()
    for o in sc.objects:
        for name in ('skin fx', 'skin tex'):
            if o.type == 'MESH' and name in o.data.color_attributes:
                o.data.color_attributes.remove(o.data.color_attributes[name])
    # the fused hair is a fine voxel surface and the head is cut very fine round the eyes: thin both out
    for name, ratio in (('Hair', 0.11), ('Head shape', 0.4)):
        o = bpy.data.objects.get(name)
        if o is not None:
            o.modifiers.new('lighter', 'DECIMATE').ratio = ratio
    bpy.data.objects['Rig'].hide_render = False

    bpy.ops.export_scene.gltf(
        filepath=out, export_format='GLB', export_apply=True, export_yup=True,
        export_animations=True, export_force_sampling=True, export_frame_range=True,
        export_animation_mode='SCENE', export_skins=True, export_attributes=False,
        export_texcoords=True, export_materials='EXPORT', export_cameras=False, export_lights=False)
    print('wrote', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
