"""Clay views of the fabric bench (tools/fabric-bench): the cloth the node solver left, on the body it lay on.

    node tools/fabric-bench/bench.mjs --scenes boot,tasset --out /tmp/fb      # the frames, as JSON
    python3 blender/fabric-bench/views.py --frames /tmp/fb --out DIR [--scenes boot,tasset]
           [--solvers v1,final] [--size 360] [--engine cycles|workbench] [--samples 8] [--no-set]

One picture a scene: a row a solver, a column a kept frame (for the run scenes, four frames over her last stride;
for the others the last frame), all from the scene's fixed orthographic camera. The cloth is matte clay, the body
a darker clay, so the form reads without any material; where leather has taken a set (creased) the clay is
darkened toward brown (`--no-set` for plain clay). Cycles at 8 samples, undenoised, renders a cell in about a
quarter of a second at 280 px (`--engine workbench` where a GPU is there).

Nothing here simulates: it builds meshes from the positions the bench wrote, so it is the same cloth the numbers
describe.
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


FRAMES = opt('--frames', '/tmp/fb')
OUT = opt('--out', '/tmp/fb-views')
SIZE = int(opt('--size', '360'))
ENGINE = opt('--engine', 'cycles')
SAMPLES = int(opt('--samples', '8'))
SCENES = opt('--scenes', 'hang,drape,fold,boot,tasset').split(',')
SOLVERS = opt('--solvers', 'v1,draft,final').split(',')

CLOTH = (0.78, 0.74, 0.68, 1)
BODY = (0.42, 0.42, 0.44, 1)
SET = (0.45, 0.2, 0.1, 1)


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.resolution_x = SIZE
    sc.render.resolution_y = SIZE
    sc.render.film_transparent = False
    if ENGINE == 'cycles':
        sc.render.engine = 'CYCLES'
        sc.cycles.samples = SAMPLES
        sc.cycles.use_denoising = False
    else:
        sc.render.engine = 'BLENDER_WORKBENCH'
        sh = sc.display.shading
        sh.light = 'STUDIO'
        sh.color_type = 'OBJECT'
        sh.show_shadows = True
        sh.show_cavity = True
        sh.cavity_type = 'WORLD'
    world = bpy.data.worlds.new('w')
    world.color = (0.9, 0.9, 0.9)
    sc.world = world
    return sc


def material(name, rgba):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = rgba
        bsdf.inputs['Roughness'].default_value = 0.9
    return m


def mesh(name, verts, faces, rgba, offset):
    me = bpy.data.meshes.new(name)
    me.from_pydata([Vector(v) + offset for v in verts], [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    ob.color = rgba
    ob.data.materials.append(material(name, rgba))
    bpy.context.scene.collection.objects.link(ob)
    return ob


def capsule(a, b, ra, rb, offset, rgba, name):
    # a tapered tube capped with spheres, as the bench's capsules
    a, b = Vector(a), Vector(b)
    ax = b - a
    L = ax.length or 1e-6
    z = ax / L
    x = z.orthogonal().normalized()
    y = z.cross(x)
    verts, faces, seg, rings = [], [], 20, 12
    prof = []
    for k in range(rings // 2 + 1):
        t = math.pi / 2 * (1 - k / (rings // 2))
        prof.append((-math.sin(t) * ra, math.cos(t) * ra))
    for k in range(rings // 2 + 1):
        t = math.pi / 2 * k / (rings // 2)
        prof.append((L + math.sin(t) * rb, math.cos(t) * rb))
    for h, r in prof:
        for s in range(seg):
            ang = 2 * math.pi * s / seg
            verts.append(a + z * h + (x * math.cos(ang) + y * math.sin(ang)) * r)
    for k in range(len(prof) - 1):
        for s in range(seg):
            i0, i1 = k * seg + s, k * seg + (s + 1) % seg
            faces.append((i0, i1, i1 + seg, i0 + seg))
    return mesh(name, [tuple(v) for v in verts], faces, rgba, offset)


def sphere(c, r, offset, rgba, name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=Vector(c) + offset, segments=32, ring_count=16)
    ob = bpy.context.active_object
    ob.name = name
    ob.color = rgba
    ob.data.materials.append(material(name, rgba))
    bpy.ops.object.shade_smooth()
    return ob


def camera(cam, sc):
    at, frm = Vector(cam['at']), Vector(cam['from'])
    cd = bpy.data.cameras.new('cam')
    cd.type = 'ORTHO'
    cd.ortho_scale = cam['size']
    ob = bpy.data.objects.new('cam', cd)
    sc.collection.objects.link(ob)
    ob.location = frm
    ob.rotation_euler = (at - frm).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = ob
    if ENGINE == 'cycles':
        ld = bpy.data.lights.new('sun', 'SUN')
        ld.energy = 3
        lo = bpy.data.objects.new('sun', ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = (math.radians(50), 0, math.radians(30))


def cell(data, frame, path):
    sc = clear()
    zero = Vector((0, 0, 0))
    # (the bench is y up, Blender z up: x, y, z -> x, -z, y)
    conv = lambda p: (p[0], -p[2], p[1])
    pos = frame['pos']
    verts = [conv(pos[3 * i:3 * i + 3]) for i in range(len(pos) // 3)]
    t = data['tris']
    faces = [tuple(t[k:k + 3]) for k in range(0, len(t), 3)]
    cloth = mesh('cloth', verts, faces, CLOTH, zero)
    if frame.get('set') and '--no-set' not in argv:
        # (where the leather has taken a set, the clay is darkened toward a warm brown: the creases it keeps)
        attr = cloth.data.color_attributes.new('set', 'FLOAT_COLOR', 'POINT')
        for i, v in enumerate(frame['set']):
            k = min(1.0, 2.5 * v)
            attr.data[i].color = tuple(CLOTH[c] * (1 - k) + SET[c] * k for c in range(3)) + (1,)
        m = cloth.data.materials[0]
        nt = m.node_tree
        node = nt.nodes.new('ShaderNodeAttribute')
        node.attribute_name = 'set'
        nt.links.new(node.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    sol = cloth.modifiers.new('thick', 'SOLIDIFY')
    sol.thickness = 0.12
    for k, b in enumerate(frame['body']):
        if 'capsule' in b:
            a, bb, ra, rb = b['capsule']
            capsule(conv(a), conv(bb), ra, rb, zero, BODY, f'c{k}')
        elif 'sphere' in b:
            c, r = b['sphere']
            sphere(conv(c), r, zero, BODY, f's{k}')
        elif 'floor' in b:
            mesh('floor', [(-40, -40, b['floor']), (40, -40, b['floor']), (40, 40, b['floor']), (-40, 40, b['floor'])], [(0, 1, 2, 3)], (0.8, 0.8, 0.8, 1), zero)
    cam = dict(data['camera'])
    cam['at'] = conv(cam['at'])
    cam['from'] = conv(cam['from'])
    if FIT:
        # (the scene's camera direction, framed on everything any solver's cloth and the body reach)
        d = Vector(cam['from']) - Vector(cam['at'])
        cam['at'] = tuple(FIT['center'])
        cam['from'] = tuple(Vector(FIT['center']) + d)
        cam['size'] = FIT['size']
    camera(cam, sc)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def sheet(paths, cols, out, labels):
    # the cells side by side, a row a solver (Blender's own image API, so nothing outside Blender is needed)
    rows = len(paths) // cols
    W, H = SIZE * cols, SIZE * rows
    img = bpy.data.images.new('sheet', W, H)
    px = [0.0] * (W * H * 4)
    for k, p in enumerate(paths):
        im = bpy.data.images.load(p)
        src = list(im.pixels)
        r, c = k // cols, k % cols
        oy = (rows - 1 - r) * SIZE
        for y in range(SIZE):
            row = src[y * SIZE * 4:(y + 1) * SIZE * 4]
            base = ((oy + y) * W + c * SIZE) * 4
            px[base:base + SIZE * 4] = row
    img.pixels = px
    img.filepath_raw = out
    img.file_format = 'PNG'
    img.save()


FIT = None


def fit(datas):
    # the frame round every point of every kept frame, seen from the scene's camera
    conv = lambda p: Vector((p[0], -p[2], p[1]))
    cam = datas[0]['camera']
    d = (conv(cam['from']) - conv(cam['at'])).normalized()
    up = Vector((0, 0, 1))
    right = d.cross(up).normalized()
    up2 = right.cross(d)
    pts = []
    for data in datas:
        for fr in data['frames']:
            pos = fr['pos']
            pts += [conv(pos[3 * i:3 * i + 3]) for i in range(0, len(pos) // 3, 3)]
            for b in fr['body']:
                if 'capsule' in b:
                    pts += [conv(b['capsule'][0]), conv(b['capsule'][1])]
    xs = [p.dot(right) for p in pts]
    ys = [p.dot(up2) for p in pts]
    zs = [p.dot(d) for p in pts]
    c = right * (min(xs) + max(xs)) / 2 + up2 * (min(ys) + max(ys)) / 2 + d * (min(zs) + max(zs)) / 2
    return {'center': c, 'size': 1.12 * max(max(xs) - min(xs), max(ys) - min(ys)) + 2}


def main():
    global FIT
    os.makedirs(OUT, exist_ok=True)
    for scn in SCENES:
        paths, cols = [], 0
        files = [os.path.join(FRAMES, f'{scn}-{sol}.json') for sol in SOLVERS]
        datas = [json.load(open(f)) for f in files if os.path.exists(f)]
        FIT = fit(datas) if datas and '--no-fit' not in argv else None
        for sol in SOLVERS:
            f = os.path.join(FRAMES, f'{scn}-{sol}.json')
            if not os.path.exists(f):
                continue
            data = json.load(open(f))
            cols = len(data['frames'])
            for fr in data['frames']:
                p = os.path.join(OUT, f'{scn}-{sol}-f{fr["frame"]:03d}.png')
                cell(data, fr, p)
                paths.append(p)
        if paths:
            sheet(paths, cols, os.path.join(OUT, f'{scn}-sheet.png'), SOLVERS)
            print('wrote', os.path.join(OUT, f'{scn}-sheet.png'))


main()
