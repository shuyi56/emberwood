"""Clay turnaround of a bun hold solve (tools/render-queue/rq_bun.py, packed by rq_bun_pack.py): the head's surface
and the hair's strands as drawn, settled at rest and at the worst frame of each motion, orthographic, from the front,
3/4, profile, back 3/4 and back, plus a close-up of the knot from behind and above. Workbench (flat clay or the
colour-coded design pass), so a view costs a second or two.

    blender -b --python bun_views.py -- --data blender/hair/data/bun-after.npz --out DIR
        [--states drawn,rest,run,spin,twist] [--angles front,3/4,profile,back3/4,back,knot] [--size 480]
        [--look clay|parts|both]

Every image is named <tag>-<state>-<angle>-<look>.png. The design pass colours each part of the bun its own flat
colour (cap and gather, core, the four wraps, the escaping end, the short ends and flyaways), with the points the
pins and the tie hold in red, so a part that slipped or a wrap that pulled loose shows by its colour."""
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def opt(k, d=None):
    return argv[argv.index(k) + 1] if k in argv else d


DATA = opt('--data', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bun-after.npz'))
OUT = opt('--out', 'out')
SIZE = int(opt('--size', '480'))
LOOK = opt('--look', 'both')
ANGLES = opt('--angles', 'front,3/4,profile,back3/4,back,knot').split(',')
# (the part colours: flat and far apart in hue, each a design part of the bun, not a material)
PART_RGB = [(0.62, 0.60, 0.57), (0.30, 0.25, 0.55), (0.90, 0.62, 0.15), (0.20, 0.62, 0.85), (0.35, 0.75, 0.35),
            (0.85, 0.35, 0.65), (0.95, 0.90, 0.25), (0.55, 0.85, 0.85), (0.85, 0.10, 0.10)]
CLAY = (0.72, 0.70, 0.67)

z = np.load(DATA)
STATES = opt('--states', ','.join(str(s) for s in z['names'])).split(',')
TAG = str(z['tag']) or 'bun'
counts = z['counts']
starts = np.r_[0, np.cumsum(counts)[:-1]]
thick = z['thick'].astype(float)
part = z['part'].astype(int)
held = z['held'].astype(bool)
os.makedirs(OUT, exist_ok=True)

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = scene.render.resolution_y = SIZE
scene.render.film_transparent = False
sh = scene.display.shading
sh.light = 'STUDIO'
sh.show_shadows = True
sh.show_cavity = True
sh.cavity_type = 'WORLD'
scene.display.shadow_shift = 0.1
scene.world = scene.world or bpy.data.worlds.new('w')
sh.background_type = 'VIEWPORT'
sh.background_color = (0.96, 0.95, 0.93)


def material(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1.0)
    return m


def head():
    me = bpy.data.meshes.new('head')
    me.from_pydata(z['head_v'].astype(float).tolist(), [], z['head_f'].tolist())
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new('head', me)
    ob.data.materials.append(material('skin_clay', (0.80, 0.76, 0.72)))
    scene.collection.objects.link(ob)
    return ob


def strands(X, look):
    objs = []
    mats = [material(f'part{k}', PART_RGB[k]) for k in range(len(PART_RGB))]
    clay = material('hair_clay', CLAY)
    cu = bpy.data.curves.new('hair', 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 1.0
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    cu.materials.append(clay)
    for m in mats:
        cu.materials.append(m)
    for k, (s, n) in enumerate(zip(starts, counts)):
        P = X[s:s + n].astype(float)
        sp = cu.splines.new('POLY')
        sp.points.add(n - 1)
        for i in range(n):
            sp.points[i].co = (*P[i], 1.0)
            sp.points[i].radius = float(np.clip(1.3 * thick[s + i], 0.004, 0.03))
        sp.material_index = 0 if look == 'clay' else 1 + part[k]
    ob = bpy.data.objects.new('hair', cu)
    scene.collection.objects.link(ob)
    objs.append(ob)
    if look == 'parts' and held.any():
        # (the held points, as small red beads: where the pins, the tie and the tucks grip)
        me = bpy.data.meshes.new('held')
        me.from_pydata(X[held].astype(float).tolist(), [], [])
        pts = bpy.data.objects.new('held', me)
        scene.collection.objects.link(pts)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.006, segments=8, ring_count=6)
        bead = bpy.context.active_object
        bead.data.materials.append(mats[8])
        bead.parent = pts
        pts.instance_type = 'VERTS'
        objs += [pts, bead]
    return objs


CAMS = {   # azimuth from the front (degrees, toward the figure's left), elevation, look-at, ortho scale
    'front': (0, 5, (0, 0.1, 0.6), 1.5), '3/4': (45, 8, (0, 0.1, 0.6), 1.5), 'profile': (90, 5, (0, 0.1, 0.6), 1.5),
    'back3/4': (135, 10, (0, 0.1, 0.6), 1.5), 'back': (180, 10, (0, 0.1, 0.6), 1.5),
    'knot': (160, 35, (0, 0.43, 0.89), 0.7),
}


def camera(name):
    az, el, at, scale = CAMS[name]
    cam = bpy.data.cameras.new('cam')
    cam.type = 'ORTHO'
    cam.ortho_scale = scale
    ob = bpy.data.objects.new('cam', cam)
    scene.collection.objects.link(ob)
    a, e = math.radians(az), math.radians(el)
    d = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])      # (the face looks to -y)
    ob.location = tuple(np.array(at) + 4.0 * d)
    ob.rotation_euler = (math.pi / 2 - e, 0.0, a)      # (a camera looks down its -z: turned up 90 degrees it looks to +y)
    scene.camera = ob
    return ob


hd = head()
looks = ['clay', 'parts'] if LOOK == 'both' else [LOOK]
for st in STATES:
    X = z['s_' + st]
    for look in looks:
        objs = strands(X, look)
        for ang in ANGLES:
            cam = camera(ang)
            sh.color_type = 'MATERIAL'
            scene.render.filepath = os.path.join(OUT, f"{TAG}-{st}-{ang.replace('/', '')}-{look}.png")
            bpy.ops.render.render(write_still=True)
            bpy.data.objects.remove(cam, do_unlink=True)
        for o in objs:
            bpy.data.objects.remove(o, do_unlink=True)
print('bun_views: done', len(STATES), 'states')
