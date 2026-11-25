"""The right front let down above the folds' crossing, on a cut that is already built.

    python3 blender/bodice_cut/over_top.py CUT.blend OUT.blend Z0,Z1 [RUN]

The same change as BODICE_OVER_TOP in bodice_cut.py (see there), applied to a saved cut so a lapel study does not have
to rebuild the whole bodice: the right front's points (front_layer 1) lose the cloth's thickness of lift between Z0
(the height the two folds meet at, all of the lift kept) and Z1 (the roll lines' top, none kept), along the mesh's own
outward normals. RUN is BODICE_OVER_RUN (0.6): the lift was already faded out sideways past the underlap's edge."""
import sys
import numpy as np
import bpy

src, dst, zz = sys.argv[1:4]; run = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
z0, z1 = [float(x) for x in zz.split(',')]
bpy.ops.wm.open_mainfile(filepath=src)
b = bpy.data.objects['Redingote bodice']; me = b.data
th = next(m.thickness for m in b.modifiers if m.type == 'SOLIDIFY'); lap = float(b['bodice_lap'])
w = th / 0.085 if 'bodice_w' not in b else float(b['bodice_w'])
mw = np.array(b.matrix_world); R = mw[:3, :3]
V = np.array([v.co[:] for v in me.vertices]); W = V @ R.T + mw[:3, 3]
N = np.array([v.normal[:] for v in me.vertices]) @ np.linalg.inv(R)
N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
lay = np.zeros(len(V), int); me.attributes['front_layer'].data.foreach_get('value', lay)
ss = lambda a, c, x: (lambda t: t * t * (3 - 2 * t))(np.clip((x - a) / (c - a), 0, 1))
keep = ss(-lap - run * w, -lap, W[:, 0]) * (W[:, 0] < lap + 0.05)
d = th * keep * ss(z0, z1, W[:, 2]) * (lay == 1) * (N[:, 1] < 0)
Wn = W - d[:, None] * N
L = (Wn - mw[:3, 3]) @ np.linalg.inv(R).T
for i in np.nonzero(d > 1e-6)[0]: me.vertices[i].co = L[i]
me.update()
print('over_top: the right front let down from z %.2f to %.2f: %d points, %.3f at most (%.1f mm)' % (z0, z1, int((d > 1e-6).sum()), d.max(), 26.2 * d.max()), flush=True)
bpy.ops.wm.save_as_mainfile(filepath=dst)
