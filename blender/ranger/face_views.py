"""Close-ups of the ranger's face and the whole figure in the 3-D look (sprite_mode off), for checking eye, brow
and face work. Run as python3 face_views.py --out DIR [--angles 0,1] [--zoom 3.4] [--hz 0.6]
[--full]. Angles are in eighths of a turn (0 faces the camera, 1 is the 3/4 view)."""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
import ranger  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--angles', default='0,1')
ap.add_argument('--zoom', type=float, default=3.4)
ap.add_argument('--hz', type=float, default=0.6)
ap.add_argument('--full', action='store_true')
args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
os.makedirs(args.out, exist_ok=True)

ranger.build()
sc = bpy.context.scene
sc.frame_set(1)
ranger.sprite_mode(False)
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
root.rotation_euler = (0, 0, 0)
