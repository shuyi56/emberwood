"""Blender's own cloth on the fabric bench's sheets, timed, to set against the node solver.

    python3 tools/fabric-bench/blender_cloth_timing.py [--scenes hang,drape] [--quality 5] [--frames 180]

The same sheet as tools/fabric-bench/scenes.mjs (1.02 m square, 25 x 25 points, in the figures' units), the same
body (hang: held at its two top corners; drape: dropped on a sphere of radius 9 over a floor), the same 180 frames
at 60 fps. Blender's cloth with its default settings (quality 5 steps a frame, collision quality 2, a cotton-ish
material) and the bench's gravity. Prints seconds a frame and in all, and the sheet's longest thread stretch, as
JSON lines.
"""
import json
import math
import sys
import time

import bpy

argv = sys.argv[1:]


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


SCENES = opt('--scenes', 'hang,drape').split(',')
QUALITY = int(opt('--quality', '5'))
FRAMES = int(opt('--frames', '180'))
U = 39.4
RES, SIZE = 25, 40.0


def sheet(z):
    d = SIZE / (RES - 1)
    verts, faces = [], []
    for r in range(RES):
        for c in range(RES):
            verts.append(((c - (RES - 1) / 2) * d, (r - (RES - 1) / 2) * d, z))
    for r in range(RES - 1):
        for c in range(RES - 1):
            i = r * RES + c
            faces.append((i, i + 1, i + RES + 1, i + RES))
    me = bpy.data.meshes.new('sheet')
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new('sheet', me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def run(name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = 60
    sc.frame_start, sc.frame_end = 1, FRAMES
    sc.gravity = (0, 0, -9.81 * U)
    if name == 'hang':
        # (the bench's hang: the sheet upright, held at its two top corners; here lying flat at 60, held at two
        # corners of one edge, as the bench drops it from flat)
        ob = sheet(60)
        g = ob.vertex_groups.new(name='pin')
        g.add([0, RES - 1], 1.0, 'REPLACE')
    else:
        ob = sheet(40)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=9, location=(0, 0, 30), segments=32, ring_count=16)
        ball = bpy.context.active_object
        ball.modifiers.new('col', 'COLLISION')
        bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
        floor = bpy.context.active_object
        floor.modifiers.new('col', 'COLLISION')
    cl = ob.modifiers.new('cloth', 'CLOTH')
    s = cl.settings
    s.quality = QUALITY
    s.mass = 0.18 * (SIZE / U) ** 2 / (RES * RES)       # linen, 180 g/m2, per point
    if name == 'hang':
        s.vertex_group_mass = 'pin'
    cl.collision_settings.collision_quality = 2
    cl.point_cache.frame_start, cl.point_cache.frame_end = 1, FRAMES
    rest = [(e.vertices[0], e.vertices[1], (ob.data.vertices[e.vertices[0]].co - ob.data.vertices[e.vertices[1]].co).length) for e in ob.data.edges]
    dg = bpy.context.evaluated_depsgraph_get()
    t0 = time.perf_counter()
    worst = 0.0
    for f in range(1, FRAMES + 1):
        sc.frame_set(f)
        if f % 30 == 0 or f == FRAMES:
            ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
            v = ev.data.vertices
            for i, j, l in rest:
                worst = max(worst, (v[i].co - v[j].co).length / l - 1)
    dt = time.perf_counter() - t0
    print(json.dumps({'scene': name, 'solver': f'blender cloth (quality {QUALITY})', 'frames': FRAMES,
                      'seconds': round(dt, 3), 'msPerFrame': round(1000 * dt / FRAMES, 2), 'maxStretchPct': round(100 * worst, 2)}))


for n in SCENES:
    run(n)
