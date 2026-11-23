# pushout.py IN.blend OUT.blend '{"amt_r":0.3,"amt_l":0.1,"y0":0.2,"z0":51.2}'
import bpy, sys, json, numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from scipy.spatial import cKDTree
blend, out, s = sys.argv[-3], sys.argv[-2], json.loads(sys.argv[-1])
bpy.ops.wm.open_mainfile(filepath=blend)
o = bpy.data.objects['Collar']; mw = np.array(o.matrix_world); inv = np.linalg.inv(mw)
P = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3,:3].T + mw[:3,3]; N = len(P)
b = bpy.data.objects['Redingote bodice']; bw = np.array(b.matrix_world)
BV = np.array([v.co[:] for v in b.data.vertices]) @ bw[:3,:3].T + bw[:3,3]
bvh = BVHTree.FromPolygons([Vector(p) for p in BV], [tuple(p.vertices) for p in b.data.polygons])
# the neck's own surface (the body), so the collar stands off the neck, not off the bodice
bo = [x for x in bpy.data.objects if x.type=='MESH' and x.name.lower().startswith(('body','skin','mannequin'))]
print('body objects', [x.name for x in bo])
near = np.array([bvh.find_nearest(Vector(p))[0][:] for p in P]); cl = np.linalg.norm(P-near,axis=1)
# a region: the back of the neck, collar's fold and fall, from the centre back round to the side
y0, z0 = s.get('y0',0.2), s.get('z0',51.2)
wy = np.clip((P[:,1]-y0)/0.8,0,1); wz = np.clip((P[:,2]-z0)/0.6,0,1)
wy, wz = wy*wy*(3-2*wy), wz*wz*(3-2*wz)
t = np.clip((P[:,0]+1.0)/2.0,0,1); t = t*t*(3-2*t)            # her right (x<0) -> her left
amt = s.get('amt_r',0.3)*(1-t) + s.get('amt_l',0.1)*t
w = wy*wz
# direction: away from the neck's axis, horizontally (and a little up), not along the bodice normal
yc = s.get('yc', 0.0); ax = np.array([0.0, yc]); dh = P[:,[0,1]] - ax; dn = np.linalg.norm(dh,axis=1)[:,None]+1e-9
dirn = np.column_stack([dh/dn, 0.15*np.ones(N)]); dirn /= np.linalg.norm(dirn,axis=1)[:,None]
P0 = P.copy(); P = P + (w*amt)[:,None]*dirn
print('pushed up to %.3f, %d points' % (np.linalg.norm(P-P0,axis=1).max(), (w*amt>1e-3).sum()))
L = P@inv[:3,:3].T+inv[:3,3]
for i,v in enumerate(o.data.vertices): v.co = L[i]
o.data.update(); bpy.ops.wm.save_as_mainfile(filepath=out)
