"""Clay and colour-coded views of the lace study's solved strips (tools/lace-study): the points study.mjs solved
and inspected, read from its run files, built into meshes and rendered. Nothing here simulates, so the pictures
are the folds the contour sheets describe (and the fold cache holds).

    blender -b --python blender/lace-study/views.py -- --runs /mnt/project-files/lace-study/runs \\
        --cases cuff-knife,cuff-gather --looks clay,parts --views front,side,back,q --size 360 --samples 8 --out DIR

One image per case, look and view: DIR/<case>-<look>-<view>.png. Orthographic cameras framed on each strip.
clay: matte clay, one colour, to judge the form. parts: the colour code (net, motif, cordonnet and scallop,
stitched heading, pressed folds), flat colours. Cycles at 8 samples, undenoised (the draft rule); --engine
workbench for a quicker look.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


RUNS = opt('--runs', '/mnt/project-files/lace-study/runs')
OUT = opt('--out', '/tmp/lace-views')
SIZE = int(opt('--size', '360'))
SAMPLES = int(opt('--samples', '8'))
ENGINE = opt('--engine', 'cycles')
LOOKS = opt('--looks', 'clay,parts').split(',')
VIEWS = opt('--views', 'front,side,back,q').split(',')
CASES = opt('--cases', '')

CLAY = (0.80, 0.77, 0.72, 1)
BODY = (0.40, 0.41, 0.44, 1)
PARTS = [(0.56, 0.70, 0.85, 1), (0.88, 0.70, 0.35, 1), (0.37, 0.19, 0.44, 1), (0.18, 0.42, 0.31, 1), (0.82, 0.32, 0.23, 1)]


def srgb_to_linear(c):
    return tuple((v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) if k < 3 else v for k, v in enumerate(c))


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.resolution_x = SIZE
    sc.render.resolution_y = SIZE
    if ENGINE == 'cycles':
        sc.render.engine = 'CYCLES'
        sc.cycles.samples = SAMPLES
        sc.cycles.use_denoising = False
    else:
        sc.render.engine = 'BLENDER_WORKBENCH'
        sh = sc.display.shading
        sh.light = 'STUDIO'
        sh.color_type = 'VERTEX' if 'parts' in LOOKS else 'OBJECT'
        sh.show_shadows = True
        sh.show_cavity = True
    world = bpy.data.worlds.new('w')
    world.color = (0.9, 0.9, 0.9)
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1)
        bg.inputs['Strength'].default_value = 1.0
    sc.world = world
    return sc


def material(name, rgba, vertex=False):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value = 0.9
    if vertex:
        attr = nt.nodes.new('ShaderNodeVertexColor')
        attr.layer_name = 'parts'
        nt.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        bsdf.inputs['Base Color'].default_value = rgba
    return m


def build(d, look):
    P = d['pos']
    verts = [(P[3 * i], -P[3 * i + 2], P[3 * i + 1]) for i in range(len(P) // 3)]       # y up -> z up
    T = d['tris']
    faces = [(T[k], T[k + 1], T[k + 2]) for k in range(0, len(T), 3)]
    me = bpy.data.meshes.new(d['name'])
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(d['name'], me)
    bpy.context.collection.objects.link(ob)
    if look == 'parts':
        layer = me.color_attributes.new('parts', 'FLOAT_COLOR', 'POINT')
        for i, k in enumerate(d['parts']):
            layer.data[i].color = srgb_to_linear(PARTS[k])
        ob.data.materials.append(material('parts', CLAY, vertex=True))
    else:
        ob.data.materials.append(material('clay', CLAY))
    # the lace is a sheet a third of a millimetre thick: shown with a little thickness so both sides shade
    mod = ob.modifiers.new('thick', 'SOLIDIFY')
    mod.thickness = 0.05 if d['mat'].get('plain') else 0.03
    mod.offset = 0
    # what it hangs on: the forearm in a cuff, the neck in a collar, the hips in a skirt, the chest behind a jabot
    o = d['o']
    if o['mount'] in ('cuff', 'collar'):
        r = o.get('cuffRadius', o['finished'] / (2 * math.pi)) * (0.7 if o['mount'] == 'cuff' else 0.82)
        h = o['depth'] + 8
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=(0, 0, -h / 2 + 3 if o['mount'] == 'cuff' else 0))
        bpy.context.object.data.materials.append(material('body', BODY))
    elif o['mount'] == 'skirt':
        # the hips under the yoke, then the thighs taken together
        for r, z0, z1 in ((o['hipRadius'], -15, 3), (o['hipRadius'] * 0.93, -40, -15)):
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=z1 - z0, location=(0, 0, (z0 + z1) / 2))
            bpy.context.object.data.materials.append(material('body', BODY))
    elif o['mount'] == 'jabot':
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0.05, -o['finished'] / 2))
        pl = bpy.context.object
        pl.scale = (o['depth'] * 2.5, o['finished'] + 4, 1)   # (local y is the height once turned upright)
        pl.rotation_euler = (math.pi / 2, 0, 0)
        pl.data.materials.append(material('body', BODY))
    return ob, verts


def camera(sc, verts, view):
    xs, ys, zs = zip(*verts)
    c = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
    size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) * 1.15
    d = {'front': Vector((0, -1, 0)), 'back': Vector((0, 1, 0)), 'side': Vector((1, 0, 0)), 'q': Vector((0.7, -0.75, 0.35))}[view].normalized()
    cam = bpy.data.cameras.new('cam')
    cam.type = 'ORTHO'
    cam.ortho_scale = size
    ob = bpy.data.objects.new('cam', cam)
    sc.collection.objects.link(ob)
    ob.location = c + d * size * 3
    ob.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    cam.clip_end = size * 10
    sc.camera = ob
    # a key light from the camera's upper left, and a fill
    for name, offs, energy in (('key', Vector((-0.5, -0.3, 0.8)), 4.0), ('fill', Vector((0.6, 0.2, 0.3)), 1.5)):
        L = bpy.data.lights.new(name, 'SUN')
        L.energy = energy
        lo = bpy.data.objects.new(name, L)
        sc.collection.objects.link(lo)
        lo.rotation_euler = (-(d + offs)).normalized().to_track_quat('-Z', 'Y').to_euler()
    return ob


def main():
    os.makedirs(OUT, exist_ok=True)
    names = CASES.split(',') if CASES else sorted(f[:-5] for f in os.listdir(RUNS) if f.endswith('.json') and 'summary' not in f and 'sweep' not in f)
    for name in names:
        with open(os.path.join(RUNS, name + '.json')) as fh:
            d = json.load(fh)
        for look in LOOKS:
            for view in VIEWS:
                sc = clear()
                ob, verts = build(d, look)
                camera(sc, verts, view)
                sc.render.filepath = os.path.join(OUT, f'{name}-{look}-{view}.png')
                bpy.ops.render.render(write_still=True)
                print('wrote', sc.render.filepath)


main()
