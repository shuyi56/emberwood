"""Dump the meshes of a built scene to an .npz for the join check outside Blender.

    python3 tools/joins/dump.py SCENE.blend OUT.npz [OBJECT ...]

Inside a build (rq_views.py, joins_scene) the same function, dump_scene(), reads the evaluated meshes directly.
Each object is stored as NAME/V (world coordinates, model units) and NAME/F (triangles)."""
import sys

import numpy as np


def mesh_arrays(o, dg):
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    n = len(me.vertices)
    V = np.empty(n * 3)
    me.vertices.foreach_get('co', V)
    V = V.reshape(n, 3)
    M = np.array(ev.matrix_world)
    V = V @ M[:3, :3].T + M[:3, 3]
    T = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get('vertices', T)
    seam = np.zeros(0, np.int64)
    if len(me.edges):
        sm = np.zeros(len(me.edges), bool)
        me.edges.foreach_get('use_seam', sm)
        if sm.any():
            E = np.empty(len(me.edges) * 2, np.int64)
            me.edges.foreach_get('vertices', E)
            seam = E.reshape(-1, 2)[sm]
    ev.to_mesh_clear()
    return V, T.reshape(-1, 3), seam


def dump_scene(names=None):
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for o in bpy.data.objects:
        if o.type != 'MESH' or (names and o.name not in names) or (not names and o.hide_render):
            continue
        V, F, S = mesh_arrays(o, dg)
        if len(F):
            out[o.name + '/V'], out[o.name + '/F'] = V, F
            if len(S):
                out[o.name + '/seam_edges'] = S
    return out


if __name__ == '__main__':
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    import bpy
    bpy.ops.wm.open_mainfile(filepath=a[0])
    d = dump_scene(a[2:] or None)
    np.savez_compressed(a[1], **d)
    for k in d:
        if k.endswith('/F'):
            print(k[:-2], len(d[k[:-2] + '/V']), 'verts', len(d[k]), 'tris')
