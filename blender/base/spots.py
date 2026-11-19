"""Check a layered figure by measurement, and render only the places that need looking at.

A full turn of the figure is five renders, and most of every one shows nothing new. This reads a scene saved by
views.py (--save), finds by measurement where a garment worn under another shows through it, and renders one small
crop square on to each such place, and on to any place named by hand. Nothing found and nothing named: nothing
rendered.

    python3 blender/base/spots.py SCENE.blend --under Chemise --over Doublet --out DIR
    python3 blender/base/spots.py SCENE.blend --out DIR --at armhole:5.0,44.5,0,8 --at lacing:0,38,0,7

--under/--over  the pair to check. The under garment's points marked 'held' (drape.build_chemise marks the points it
                tucked under the outer garment) must lie inside the outer garment's cloth; those that do not are
                grouped into spots, each rendered from straight in front of it.
--at            a place to render whatever the check says: name:x,z,yaw,height (x and z of the view's middle once
                the figure is turned by yaw degrees, and the view's height in units).
--size, --samples   a crop's pixels (360) and samples (32)."""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _posed(o, without=('SUBSURF', 'SOLIDIFY', 'BEVEL')):
    """An object's own points as they stand at this frame (its sheet: without the modifiers that add points)."""
    off = [m for m in o.modifiers if m.type in without and m.show_viewport]
    for m in off:
        m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    pts = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
    polys = [tuple(p.vertices) for p in me.polygons]
    e.to_mesh_clear()
    for m in off:
        m.show_viewport = True
    return pts, polys


def showing(under, over, log=print):
    """The held points of `under` that are through `over`'s cloth: their places, how far out they are, and the way
    out there. (Outward is told from the sheet's own facing, which fit._sheet turns outward.)"""
    if 'held' not in under.data.attributes:
        log('spots: %s has no held points marked: nothing to check' % under.name)
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    held = np.empty(len(under.data.vertices), dtype=np.float32)
    under.data.attributes['held'].data.foreach_get('value', held)
    P, _ = _posed(under)
    Q, polys = _posed(over)
    tree = BVHTree.FromPolygons([Vector(q) for q in Q], polys)
    thick = next((m.thickness for m in over.modifiers if m.type == 'SOLIDIFY'), 0.0)
    at, depth, way = [], [], []
    for i in np.nonzero(held > 0.5)[0]:
        loc, nrm, _, _ = tree.find_nearest(Vector(P[i]))
        d = (Vector(P[i]) - loc).dot(nrm)
        if d > thick:
            at.append(P[i])
            depth.append(d - thick)
            way.append(nrm[:])
    log('spots: %d of %d held points of %s show through %s' % (len(at), int((held > 0.5).sum()), under.name, over.name))
    return np.array(at).reshape(-1, 3), np.array(depth), np.array(way).reshape(-1, 3)


def group(at, depth, way, reach=1.5):
    """The showing points gathered into spots: (middle, the way out, how many points, the furthest out)."""
    left = np.ones(len(at), dtype=bool)
    spots = []
    while left.any():
        i = int(np.argmax(np.where(left, depth, -1.0)))
        m = left & (np.linalg.norm(at - at[i], axis=1) < reach)
        n = way[m].mean(0)
        spots.append((at[m].mean(0), n / max(np.linalg.norm(n), 1e-9), int(m.sum()), float(depth[m].max())))
        left &= ~m
    return spots


def crop(root, cam, path, yaw, x, z, height, size, samples):
    sc = bpy.context.scene
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x = sc.render.resolution_y = size
    cam.data.type, cam.data.ortho_scale, cam.data.sensor_fit = 'ORTHO', height, 'VERTICAL'
    root.rotation_euler = (0, 0, math.radians(yaw))
    cam.location, cam.rotation_euler = Vector((x, -200.0, z)), (math.pi / 2, 0, 0)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('scene')
    ap.add_argument('--under')
    ap.add_argument('--over')
    ap.add_argument('--out', required=True)
    ap.add_argument('--at', action='append', default=[])
    ap.add_argument('--size', type=int, default=360)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--root', default='Gown')
    a = ap.parse_args(argv)
    bpy.ops.wm.open_mainfile(filepath=a.scene)
    sc = bpy.context.scene
    sc.frame_set(1)
    root, cam = bpy.data.objects[a.root], bpy.data.objects['View']
    sc.camera = cam
    views = []
    if a.under and a.over:
        for k, (c, n, count, deep) in enumerate(group(*showing(bpy.data.objects[a.under], bpy.data.objects[a.over]))):
            yaw = math.atan2(-n[0], -n[1])              # (turned so the way out of the spot faces the camera)
            x = c[0] * math.cos(yaw) - c[1] * math.sin(yaw)
            print('spots: %d points through at (%.1f, %.1f, %.1f), up to %.2f out' % (count, c[0], c[1], c[2], deep))
            views.append(('through-%d' % k, math.degrees(yaw), x, c[2], 6.0))
    for spec in a.at:
        name, nums = spec.split(':')
        x, z, yaw, h = (float(v) for v in nums.split(','))
        views.append((name, yaw, x, z, h))
    os.makedirs(a.out, exist_ok=True)
    for name, yaw, x, z, h in views:
        crop(root, cam, os.path.join(a.out, name + '.png'), yaw, x, z, h, a.size, a.samples)
    print('spots: %d crops rendered' % len(views))


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
