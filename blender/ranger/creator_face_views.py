"""Cycles views of the Face Creator's face in Blender, on fixed orthographic cameras, for drafts and before/afters.

    blender -b -P creator_face_views.py -- --dump DIR [--source dump|glb] [--glb PATH]
        [--looks clay,zones,full] [--views face,three-quarter,side,eyes,lips,lips-three-quarter]
        [--size 640] [--samples 64] [--denoise] --out DIR

--source dump (the default): the creator's face as creator_face.py ports it, from a dump-face.cjs dump.
--source glb: what Blender got from the creator before the port, the creator head of a merged game glb (its skin and
  eyes, through glTF's plain materials; the hair is left out, as in the dump). It is laid on the dump's head by the
  skin's bounding box, so both stand in the same frame and the cameras are the same.
The cameras depend only on the dump's head (its height, its nose tip and its eyes), so a before and an after on the
same dump are on the same camera. Writes <look>-<view>.png. The light is creator_head_views.py's: a warm key above
and left of the camera, a soft fill and a faint rim, the creator's own rig.
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import creator_face as CF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dump', required=True)
ap.add_argument('--source', default='dump', choices=('dump', 'glb'))
ap.add_argument('--glb', default='')
ap.add_argument('--looks', default='clay,full')
ap.add_argument('--views', default='face,three-quarter,side,eyes,lips')
ap.add_argument('--size', type=int, default=640)
ap.add_argument('--samples', type=int, default=64)
ap.add_argument('--denoise', action='store_true')
ap.add_argument('--cavity', action='store_true', help="multiply in the creator's baked cavity shade (counted twice in Cycles)")
ap.add_argument('--out', default='creator-face-views')
args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
os.makedirs(args.out, exist_ok=True)
CF.CAVITY = args.cavity

D = CF.load(args.dump)
# the frame, from the dump's head alone
V = D['head']['verts']
xs, ys, zs = V[0::3], V[1::3], V[2::3]
lo, hi = Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))
H = hi.z - lo.z
cx = 0.5 * (lo.x + hi.x)
mid = [k for k in range(len(zs)) if lo.z + 0.35 * H < zs[k] < lo.z + 0.65 * H]
kt = min(mid, key=lambda k: ys[k])
tip = Vector((xs[kt], ys[kt], zs[kt]))
eye_c = [Vector([sum(e['verts'][i::3]) / (len(e['verts']) // 3) for i in range(3)]) for e in D['eyes']]
near_eye = min(eye_c, key=lambda c: c.x)        # (the eye on -x, the one the turned cameras see first)
# the mouth: the lip map's middle row is not in model space, so take the head's front at the height of the lips
# (the lips' parting sits at about 0.21 of the head's height above the chin: skin_layers' LIP_Z0..LIP_Z1)
lip_z = lo.z + 0.235 * H
lips_pts = [k for k in range(len(zs)) if abs(zs[k] - lip_z) < 0.02 * H and abs(xs[k] - cx) < 0.05 * H]
lip_y = min(ys[k] for k in lips_pts)
mouth = Vector((cx, lip_y, lip_z))
print('head height', H, 'nose tip', tip, 'eyes', eye_c, 'mouth', mouth)


def scene_reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def sun(name, euler, energy, color):
        d = bpy.data.lights.new(name, 'SUN')
        d.energy, d.color = energy, color
        d.angle = math.radians(3.0)
        o = bpy.data.objects.new(name, d)
        o.rotation_euler = [math.radians(a) for a in euler]
        sc.collection.objects.link(o)
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
    if hasattr(sc.render.image_settings, 'media_type'):      # (newer builds pick the kind of file before its format)
        sc.render.image_settings.media_type = 'IMAGE'
    sc.render.image_settings.file_format = 'PNG'
    sc.view_settings.view_transform = 'Standard'
    cam_d = bpy.data.cameras.new('cam')
    cam_d.type = 'ORTHO'
    cam = bpy.data.objects.new('cam', cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return sc, cam, cam_d


def load_glb(look):
    """the creator head of a merged game glb (its skin and its eyes), laid on the dump's head by the skin's box"""
    bpy.ops.import_scene.gltf(filepath=args.glb)
    bpy.context.view_layer.update()
    ch = bpy.data.objects.get('Creator head')
    keep = []
    for o in list(bpy.data.objects):
        if o.type != 'MESH':
            continue
        p, inside = o.parent, False
        while p is not None:
            inside = inside or p == ch
            p = p.parent
        alpha = any(n.type == 'BSDF_PRINCIPLED' and n.inputs['Alpha'].is_linked
                    for ms in o.material_slots if ms.material and ms.material.node_tree for n in ms.material.node_tree.nodes)
        if (ch is None or inside) and not alpha:
            keep.append(o)
    keep = [o for o in keep if len(o.data.vertices) > 5000]      # (the skin and the eyes)
    for o in keep:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    for o in list(bpy.data.objects):
        if o not in keep and o.type != 'LIGHT' and o.type != 'CAMERA':
            bpy.data.objects.remove(o)
    skin = max(keep, key=lambda o: len(o.data.vertices))
    bpy.context.view_layer.update()
    pts = [skin.matrix_world @ v.co for v in skin.data.vertices]
    blo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bhi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    s = H / (bhi.z - blo.z)
    T = Matrix.Translation(Vector((cx, lo.y, lo.z))) @ Matrix.Scale(s, 4) @ Matrix.Translation(-Vector((0.5 * (blo.x + bhi.x), blo.y, blo.z)))
    for o in keep:
        o.matrix_world = T @ o.matrix_world
        if look != 'full':
            o.data.materials.clear()
            o.data.materials.append(CF._flat('before ' + look, (0.42, 0.42, 0.42) if look == 'clay' else (0.78, 0.78, 0.78))[0])
    return keep


VIEWS = {      # name: (target, ortho scale in head heights, turn about z in degrees, tilt down in degrees)
    'face': ('face', 1.15, 0, 0),
    'three-quarter': ('face', 1.15, -45, 0),
    'side': ('face', 1.15, -90, 0),
    'eyes': ('eyes', 0.42, 0, 0),
    'eye-three-quarter': ('eye', 0.22, -35, 0),
    'lips': ('mouth', 0.26, 0, 0),
    'lips-three-quarter': ('mouth', 0.26, -40, 8),
}


def target(name):
    if name == 'face':
        return Vector((cx, tip.y, lo.z + 0.52 * H))
    if name == 'eyes':
        return Vector((cx, near_eye.y, 0.5 * (eye_c[0].z + eye_c[1].z)))
    if name == 'eye':
        return near_eye
    return mouth


for look in args.looks.split(','):
    sc, cam, cam_d = scene_reset()
    if args.source == 'dump':
        CF.build(D, look)
    else:
        load_glb(look)
    for v in args.views.split(','):
        tname, scale, turn, tilt = VIEWS[v]
        cam_d.ortho_scale = scale * H
        t, a = math.radians(turn), math.radians(tilt)
        d = Vector((math.sin(t) * math.cos(a), -math.cos(t) * math.cos(a), math.sin(a)))
        cam.location = target(tname) + d * (4 * H)
        cam.rotation_euler = (math.pi / 2 - a, 0, t)
        cam_d.clip_start, cam_d.clip_end = 0.01, 10 * H
        sc.render.filepath = os.path.join(args.out, '%s-%s.png' % (look, v))
        bpy.ops.render.render(write_still=True)
        print('wrote', sc.render.filepath, flush=True)
print('done', args.out)
