"""Dump a built bodice (after foldfront and notchcollar) for the contour sheet and the gauges, no Blender needed after:

    python3 blender/tension/dump_build.py OUT.blend OUT.npz

Saves the bodice's and the collar's points (the meshes' own points, world space, no Solidify) and triangles."""
import sys
import numpy as np
import bpy

blend, out = sys.argv[-2:]
bpy.ops.wm.open_mainfile(filepath=blend)
D = {}
for key, name in (('body', 'Redingote bodice'), ('collar', 'Redingote collar')):
    o = bpy.data.objects.get(name)
    if o is None:
        continue
    mw = np.array(o.matrix_world)
    D[key + '_X'] = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
    D[key + '_T'] = np.array([(p.vertices[0], p.vertices[j], p.vertices[j + 1]) for p in o.data.polygons
                              for j in range(1, len(p.vertices) - 1)])
b = bpy.data.objects['Redingote bodice']
D['M'] = np.array([repr({k[7:]: float(b[k]) for k in b.keys() if k.startswith('bodice_')})])
np.savez_compressed(out, **D)
print('dump_build: saved', out)
