"""Close-ups of the ranger's face and the whole figure in the 3-D look (sprite_mode off), for checking eye, brow
and face work. Run as python3 face_views.py --out DIR [--angles 0,1] [--zoom 3.4] [--hz 0.6]
[--full] [--game] [--hide Eyeshadow,Lashes] [--on mouth]. Angles are in eighths of a turn (0 faces the camera, 1 is the 3/4 view)."""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import ranger  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--angles', default='0,1')
ap.add_argument('--zoom', type=float, default=3.4)
ap.add_argument('--hz', type=float, default=0.6)
ap.add_argument('--full', action='store_true')
ap.add_argument('--game', action='store_true', help='also the world\'s close follow camera: perspective, 38 degree field, 27 degrees down')
ap.add_argument('--on', default='eye', choices=('eye', 'mouth'), help='what the turned close-ups centre on: the nearer eye, or the mouth')
ap.add_argument('--hide', default='', help='comma-separated object name prefixes to leave out, e.g. Eyeshadow,Lashes')
args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
os.makedirs(args.out, exist_ok=True)

ranger.build()
sc = bpy.context.scene
sc.frame_set(1)
ranger.sprite_mode(False)
for o in bpy.data.objects:
    if args.hide and o.name.startswith(tuple(h for h in args.hide.split(',') if h)):
        o.hide_render = True
sc.render.engine = 'CYCLES'
sc.cycles.samples = 48
sc.cycles.max_bounces = 3
sc.cycles.use_denoising = False
sc.render.film_transparent = False
bg = sc.world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.09, 0.14, 0.07, 1)
bg.inputs['Strength'].default_value = 1.6
if hasattr(sc.render.image_settings, 'media_type'):
    sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_depth = '8'
cam = sc.camera
cam.rotation_euler = (math.pi / 2, 0, 0)
root = bpy.data.objects['Ranger']
angles = [float(a) for a in args.angles.split(',')]


def shoot(name, scale, z, rx, ry):
    cam.data.ortho_scale = scale
    cam.location = (0, cam.location.y, z)
    sc.render.resolution_x, sc.render.resolution_y = rx, ry
    tiles = []
    for a in angles:
        root.rotation_euler = (0, 0, a * math.pi / 4)
        cx = 0.0
        if name == 'face' and a % 8:
            # turned, the face is centred on the eye nearer the camera, wherever the head's shape puts it
            bpy.context.view_layer.update()
            def centre(o):
                return o.matrix_world @ (sum((Vector(c) for c in o.bound_box), Vector()) / 8)
            if args.on == 'mouth':
                # (the lip paint, not the solid mouth line: that is hidden up close, so its transform is never updated)
                cx = centre(bpy.data.objects.get('Lip paint') or bpy.data.objects['Mouth']).x
            else:
                near = min((centre(o) for o in bpy.data.objects if o.name.startswith('Eye white')), key=lambda c: c.y)
                cx = near.x
        cam.location = (cx, cam.location.y, z)
        sc.render.filepath = os.path.join(args.out, f'_{name}_{a:g}.png')
        bpy.ops.render.render(write_still=True)
        tiles.append(Image.open(sc.render.filepath).convert('RGB'))
        os.remove(sc.render.filepath)
    sheet = Image.new('RGB', (sum(t.width for t in tiles), tiles[0].height))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    path = os.path.join(args.out, f'{name}.png')
    sheet.save(path)
    print('wrote', path)


shoot('face', args.zoom, ranger.HEAD_Z + args.hz * ranger.HEAD_H, 600, 540)
if args.full:
    top = ranger.HEAD_Z + ranger.HEAD_H
    shoot('full', top * 1.08, top * 0.5, 400, 640)
if args.game:
    # the world's camera at its closest (src/world/main.js: distance 7 world units at a 27 degree pitch, aimed 0.9
    # above the ground, field 38 degrees; the model is scaled by 2.2 / 80 * 1.2), facing her, then the same view
    # narrowed onto the head
    scale = 2.2 / 80 * 1.2
    dist, pitch, aim = 7 / scale, math.radians(27), 0.9 / scale
    root.rotation_euler = (0, 0, 0)
    cam.data.type = 'PERSP'
    cam.data.sensor_fit = 'VERTICAL'
    cam.location = (0, -dist * math.cos(pitch), aim + dist * math.sin(pitch))
    cam.rotation_euler = (math.pi / 2 - pitch, 0, 0)
    tiles = []
    for name, fov, rx, ry in (('wide', 38, 760, 600), ('head', 5.5, 600, 600)):
        cam.data.angle = math.radians(fov)
        if name == 'head':
            # aim at the head from the same spot
            head = Vector((0, 0, ranger.HEAD_Z + 0.5 * ranger.HEAD_H))
            cam.rotation_euler = (head - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.resolution_x, sc.render.resolution_y = rx, ry
        sc.render.filepath = os.path.join(args.out, f'_game_{name}.png')
        bpy.ops.render.render(write_still=True)
        tiles.append(Image.open(sc.render.filepath).convert('RGB'))
        os.remove(sc.render.filepath)
    sheet = Image.new('RGB', (sum(t.width for t in tiles), max(t.height for t in tiles)))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    sheet.save(os.path.join(args.out, 'game.png'))
    print('wrote', os.path.join(args.out, 'game.png'))
root.rotation_euler = (0, 0, 0)
