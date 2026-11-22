"""Read the bare bodice out of a cut .blend (bodice_cut's CUT.blend: no lapels, no collar) into an .npz for the
tension solver, which needs no Blender:

    python3 blender/tension/read_bodice.py CUT.blend OUT.npz

Saves the bodice's points as foldfront.py folds them (world space, the mesh's own points: `X`) and as worn (the
armature applied, without Solidify or Bevel: `worn`; within 2.5 mm of X on the cut of 2026-11-22), its faces, its
`front_layer` (1 the right front, 2 the left front and its underlap, 0 the rest), its measures (the `bodice_*`
properties) and its cloth thickness (the Solidify's)."""
import sys
import numpy as np
import bpy

blend, out = sys.argv[-2:]
bpy.ops.wm.open_mainfile(filepath=blend)
b = bpy.data.objects['Redingote bodice']; me = b.data
mw = np.array(b.matrix_world)
for m in b.modifiers:
    if m.type in ('SOLIDIFY', 'BEVEL'):
        m.show_viewport = False
dg = bpy.context.evaluated_depsgraph_get()
mee = b.evaluated_get(dg).to_mesh()
worn = np.array([v.co[:] for v in mee.vertices]) @ mw[:3, :3].T + mw[:3, 3]
X = np.array([v.co[:] for v in me.vertices]) @ mw[:3, :3].T + mw[:3, 3]
F = [list(p.vertices) for p in me.polygons]
FF = np.full((len(F), max(len(f) for f in F)), -1)
for i, f in enumerate(F):
    FF[i, :len(f)] = f
fl = np.array([d.value for d in me.attributes['front_layer'].data])
M = {k[7:]: float(b[k]) for k in b.keys() if k.startswith('bodice_')}
th = [m for m in b.modifiers if m.type == 'SOLIDIFY'][0].thickness
np.savez_compressed(out, X=X, worn=worn, F=FF, front_layer=fl, thick=th, M=np.array([repr(M)]))
print('read_bodice: %d points, %d faces, cloth %.3f thick -> %s' % (len(X), len(F), th, out))
