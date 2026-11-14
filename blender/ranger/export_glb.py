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
        p.inputs['Roughness'].default_value = 0.6 if mid == ranger.M_HAIR else 0.9
        if mid in (ranger.M_EYE, ranger.M_SCLERA):
            # the eye stays wet and glossy, so the world's lights leave a highlight on it; the iris keeps its
            # painted shading as vertex colours, the white its painting (shading and veins) as a texture
            p.inputs['Roughness'].default_value = 0.12 if mid == ranger.M_EYE else 0.3
            if mid == ranger.M_EYE:
                col = nt.nodes.new('ShaderNodeVertexColor')
                col.layer_name = 'eye_col'
                nt.links.new(col.outputs['Color'], p.inputs['Base Color'])
            else:
                tex = nt.nodes.new('ShaderNodeTexImage')
                tex.image = ranger.sclera_image()
                nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
        if mid in (ranger.M_STEEL, ranger.M_GOLD, ranger.M_MAIL):
            p.inputs['Metallic'].default_value = 0.7
            p.inputs['Roughness'].default_value = 0.45
        nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])


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
    plain_materials()
    # the fused hair is a fine voxel surface, far denser than the world needs: thin it out
    for name, ratio in (('Hair', 0.25),):
        d = bpy.data.objects[name].modifiers.new('lighter', 'DECIMATE')
        d.ratio = ratio
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
