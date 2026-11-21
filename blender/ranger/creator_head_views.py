"""Cycles renders of the Face Creator head as it ships in the game: face, three-quarter, side and nose close-ups.
Imports a merged game GLB (the creator head on the game body), so the nose, eyes and mouth are the creator's own.
Run: blender -b -P creator_head_views.py -- --glb PATH --out DIR [--size 1000] [--samples 96] [--denoise]
The glTF faces +Z, which Blender imports as -Y, so the camera sits on -Y and turns about the head."""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument('--glb', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--size', type=int, default=1000)
ap.add_argument('--samples', type=int, default=96)
ap.add_argument('--denoise', action='store_true')
args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
os.makedirs(args.out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=args.glb)
sc = bpy.context.scene
bpy.context.view_layer.update()

heads = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.lower().startswith('head part')]
if not heads:
    sys.exit('no "head part" meshes in ' + args.glb)
pts = [o.matrix_world @ v.co for o in heads for v in o.data.vertices]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
H = hi.z - lo.z
cx = (lo.x + hi.x) / 2
# the nose tip: the most forward (-Y) vertex between 35% and 65% of the head's height
mid = [p for p in pts if lo.z + 0.35 * H < p.z < lo.z + 0.65 * H]
tip = min(mid, key=lambda p: p.y)
print('head height', H, 'nose tip', tip)

# lights: warm key above-left of the camera, a soft fill, a faint rim (the creator's own rig)
def sun(name, euler, energy, color):
    d = bpy.data.lights.new(name, 'SUN')
    d.energy, d.color = energy, color
    o = bpy.data.objects.new(name, d)
    o.rotation_euler = [math.radians(a) for a in euler]
    sc.collection.objects.link(o)
    return o

sun('key', (55, 0, -30), 3.2, (1.0, 0.97, 0.93))
sun('fill', (70, 0, 40), 0.9, (1.0, 0.89, 0.82))
sun('rim', (60, 0, 160), 0.6, (0.96, 0.94, 0.92))
sc.world = bpy.data.worlds.new('w')
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.96, 0.92, 0.89, 1)
bg.inputs['Strength'].default_value = 0.5

sc.render.engine = 'CYCLES'
sc.cycles.samples = args.samples
sc.cycles.use_denoising = bool(args.denoise)
sc.render.resolution_x = sc.render.resolution_y = args.size
sc.render.image_settings.file_format = 'PNG'
sc.view_settings.view_transform = 'Standard'

cam_d = bpy.data.cameras.new('cam')
cam_d.type = 'ORTHO'
cam = bpy.data.objects.new('cam', cam_d)
sc.collection.objects.link(cam)
sc.camera = cam


def shoot(name, target, scale, turn):
    """Orthographic shot of target, turned about the vertical axis through the head's centre by turn degrees."""
    cam_d.ortho_scale = scale
    t = math.radians(turn)
    # orthographic: only the direction matters, so sit on the target and step back along the view
    cam.location = target + Vector((math.sin(t), -math.cos(t), 0)) * (4 * H)
    cam.rotation_euler = (math.pi / 2, 0, t)
    sc.render.filepath = os.path.join(args.out, name + '.png')
    bpy.ops.render.render(write_still=True)


face = Vector((cx, tip.y, lo.z + 0.5 * H))
for turn, nm in ((0, 'face'), (-45, 'three-quarter'), (-90, 'side')):
    shoot(nm, face, 1.25 * H, turn)
nose = Vector((tip.x, tip.y, tip.z))
shoot('nose-front', nose, 0.32 * H, 0)
shoot('nose-three-quarter', nose, 0.32 * H, -45)
shoot('nose-side', nose, 0.32 * H, -90)
# from below: the nostril shapes and the wings
cam_d.ortho_scale = 0.32 * H
cam.location = nose + Vector((0, -math.cos(math.radians(30)), -math.sin(math.radians(30)))) * (4 * H)
cam.rotation_euler = (math.radians(120), 0, 0)
sc.render.filepath = os.path.join(args.out, 'nose-below.png')
bpy.ops.render.render(write_still=True)
print('done', args.out)
