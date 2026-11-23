"""foldcollar_blend.py LAPEL.blend COLLAR.npz OUT.blend: foldcollar.py's collar put in the scene as 'Redingote collar'
(any collar already there is removed), the coat's own cloth thickness given by a Solidify modifier."""
import bpy, sys, numpy as np
src, npz, out = sys.argv[-3:]
bpy.ops.wm.open_mainfile(filepath=src)
old = bpy.data.objects.get('Redingote collar')
if old: bpy.data.objects.remove(old, do_unlink=True)
D = np.load(npz); X, T = D['pts'], D['tri']
body = bpy.data.objects['Redingote bodice']; inv = np.linalg.inv(np.array(body.matrix_world))
me = bpy.data.meshes.new('Redingote collar'); me.from_pydata([tuple(inv[:3, :3] @ p + inv[:3, 3]) for p in X], [], [tuple(int(i) for i in t) for t in T]); me.update()
ob = bpy.data.objects.new('Redingote collar', me); ob.matrix_world = body.matrix_world.copy()
for c in body.users_collection: c.objects.link(ob)
for m in body.data.materials: me.materials.append(m)
bpy.context.view_layer.objects.active = ob
bm_ = ob.modifiers.new('Cloth thickness', 'SOLIDIFY'); sol = [m for m in body.modifiers if m.type == 'SOLIDIFY']
# (the coat's own cloth, centred on the collar's surface: 0.06 was a third of it, a collar of paper, 2026-11-24)
bm_.thickness = sol[0].thickness if sol else 0.1669; bm_.offset = 0.0; bm_.use_rim = True
for p in me.polygons: p.use_smooth = True
bpy.ops.wm.save_as_mainfile(filepath=out)
print('foldcollar_blend: saved', out)
