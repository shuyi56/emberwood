import os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'blender')
for d in ('boot', 'hose', 'vest', 'chemise', 'sorceress_pendant', 'shared', 'gown', 'base'): sys.path.insert(0, os.path.join(R, d))
import bpy, female, numpy as np
female.MANNEQUIN = True
import boot, last, sew, pattern
boot.CREASE = False
boot.build()
import json

o = bpy.data.objects['Boot.L']
X = sew._co(o); Nv = sew._normals(o)
F = [tuple(p.vertices) for p in o.data.polygons]
out = dict(X=X, N=Nv)
names = []
for p in last.PIECES['L']:
    if p['name'] == 'sole': continue
    nm = p['name'].replace(' ', '_'); names.append(nm)
    out['V_' + nm] = np.asarray(p['V'], float); out['start_' + nm] = p['start']
    out['F_' + nm] = np.array([f for f in p['F'] if len(f) == 3] or [[0, 0, 0]])
    out['Q_' + nm] = np.array([f for f in p['F'] if len(f) == 4] or [[0, 0, 0, 0]])
    uv, give = pattern.flatten(p); out['uv_' + nm] = uv
    out['lift_' + nm] = np.broadcast_to(np.asarray(p['lift'], float), (len(p['V']),)).copy()
h = bpy.data.objects['Hose']
rig = [m.object for m in h.modifiers if m.type == 'ARMATURE']
for r_ in rig: r_.data.pose_position = 'REST'
bpy.context.view_layer.update()
e = h.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = e.to_mesh()
out['hose'] = np.array([h.matrix_world @ v.co for v in me.vertices])
out['names'] = np.array(names)
np.savez('' + os.environ.get('CONTOUR_OUT', '/tmp/boot-contours.npz') + '', **out)
print('DUMP DONE', names)
