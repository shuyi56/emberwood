"""Close-up views of the ranger's head for judging the sculpt: the face, skull, jaw and neck from several angles.

Builds the figure with ranger.py, then renders the head with a perspective portrait camera, a key, fill and rim
light, in two looks: 'clay' (every surface one matte grey, the hair hidden, so only the forms show) and 'full'
(the figure's own close-up materials). Writes one PNG per look, a row of views each:
front, front 3/4, profile, back 3/4, plus a low 3/4 looking up under the jaw.

    python blender/ranger/sculpt_views.py [--out DIR] [--size PX] [--samples N] [--looks clay,full]
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranger as R  # noqa: E402

VIEWS = [('front', 0, 0.0), ('3/4', 45, 0.0), ('profile', 90, 0.0), ('back 3/4', 135, 0.0), ('low 3/4', 35, -0.35)]


def setup(size, samples):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 4
    sc.cycles.pixel_filter_type = 'GAUSSIAN'
    sc.cycles.filter_width = 1.5
    sc.render.film_transparent = False
    sc.render.resolution_x = sc.render.resolution_y = size
    if hasattr(sc.render.image_settings, 'media_type'):
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_depth = '8'
    sc.view_settings.view_transform = 'AgX' if 'AgX' in [i.identifier for i in sc.view_settings.bl_rna.properties['view_transform'].enum_items] else 'Filmic'
    bg = sc.world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.18, 0.19, 0.21, 1)
    bg.inputs['Strength'].default_value = 0.35
    for o in list(bpy.data.objects):
        if o.type == 'LIGHT':
            bpy.data.objects.remove(o)

    def area(name, loc, energy, size_, colour=(1, 1, 1)):
        d = bpy.data.lights.new(name, 'AREA')
        d.energy, d.size, d.color = energy, size_, colour
        o = bpy.data.objects.new(name, d)
        sc.collection.objects.link(o)
        o.location = loc
        o.rotation_euler = (Vector((0, 0, 58.5)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return o
    rig = bpy.data.objects.new('Lights', None)
    sc.collection.objects.link(rig)
    for o in (area('Key', (-14, -16, 68), 5200, 8, (1, 0.96, 0.9)), area('Fill', (16, -12, 57), 1100, 12, (0.85, 0.9, 1)),
              area('Rim', (8, 20, 66), 2600, 6)):
        o.parent = rig
    cam_d = bpy.data.cameras.new('Portrait')
    cam_d.lens = 85
    cam_d.clip_start = 1
    cam_d.clip_end = 1000
    cam = bpy.data.objects.new('Portrait', cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam, rig


def clay(on, saved):
    if on:
        m = bpy.data.materials.new('clay')
        m.use_nodes = True
        b = m.node_tree.nodes.get('Principled BSDF')
        b.inputs['Base Color'].default_value = (0.42, 0.40, 0.38, 1)
        b.inputs['Roughness'].default_value = 0.55
        for o in bpy.data.objects:
            if o.type != 'MESH':
                continue
            saved[o.name] = ([s.material for s in o.material_slots], o.hide_render)
            if o.name.startswith(('Hair', 'Bangs', 'Side lock', 'Wool fuzz')) or 'hair' in o.name.lower():
                o.hide_render = True
            for s in o.material_slots:
                s.material = m
    else:
        for name, (mats, hide) in saved.items():
            o = bpy.data.objects[name]
            for s, mm in zip(o.material_slots, mats):
                s.material = mm
            o.hide_render = hide


def render_row(cam, rig, out, tmp, dist=30.0):
    from PIL import Image
    sc = bpy.context.scene
    root = bpy.data.objects['Ranger']
    tiles = []
    target = Vector((0, 0, 57.6))
    for i, (name, yaw, pitch) in enumerate(VIEWS):
        root.rotation_euler = (0, 0, math.radians(yaw))
        d = Vector((0, -math.cos(pitch), math.sin(pitch)))
        cam.location = target + d * dist
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        p = os.path.join(tmp, f'v{i}.png')
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        tiles.append(np.array(Image.open(p).convert('RGB')))
        print('rendered', name, flush=True)
    root.rotation_euler = (0, 0, 0)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
    print('wrote', out)


def main():
    import tempfile
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    out_dir = opt('--out', os.path.join(tempfile.gettempdir(), 'ranger-views'))
    size, samples = int(opt('--size', 420)), int(opt('--samples', 48))
    looks = opt('--looks', 'clay,full').split(',')
    if '--views' in argv:
        keep = opt('--views', '').split(',')
        VIEWS[:] = [v for v in VIEWS if v[0].replace(' ', '') in keep]
    os.makedirs(out_dir, exist_ok=True)
    R.build()
    bpy.context.scene.frame_set(1)
    R.sprite_mode(False)
    cam, rig = setup(size, samples)
    saved = {}
    with tempfile.TemporaryDirectory() as tmp:
        for look in looks:
            clay(look == 'clay', saved)
            render_row(cam, rig, os.path.join(out_dir, f'head-{look}.png'), tmp)
            if look == 'clay':
                clay(False, saved)


if __name__ == '__main__':
    main()
