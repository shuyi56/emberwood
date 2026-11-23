# edgein.py IN.blend COLLAR.pkl OUT.blend '{"x0":1.0,"y1":1.5,"in":0.12,"rounds":120}'
import bpy, sys, pickle, json, numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector
blend, pk, out, s = sys.argv[-4], sys.argv[-3], sys.argv[-2], json.loads(sys.argv[-1])
D = pickle.load(open(pk,'rb')); T = np.asarray(D['T']); kind = np.asarray(D['kind'])
bpy.ops.wm.open_mainfile(filepath=blend)
o = bpy.data.objects['Collar']; mw = np.array(o.matrix_world); inv = np.linalg.inv(mw)
P = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3,:3].T + mw[:3,3]; N = len(P)
nb = [set() for _ in range(N)]
for t in T:
    for i in range(3): nb[t[i]].add(int(t[(i+1)%3])); nb[t[i]].add(int(t[(i+2)%3]))
E_ = np.sort(np.concatenate([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]]),1); Eu,c = np.unique(E_,axis=0,return_counts=True)
rimE = Eu[c==1]; rn = {}
for a,b in rimE: rn.setdefault(a,[]).append(b); rn.setdefault(b,[]).append(a)
sx = s.get('side',1)
reg = (P[:,0]*sx > s.get('x0',1.0)) & (P[:,1] < s.get('y1',1.8)) & (P[:,1] > s.get('y0',-1.2))
fade = np.clip((P[:,0]*sx - s.get('x0',1.0))/0.5,0,1)*np.clip((s.get('y1',1.8)-P[:,1])/0.6,0,1)*np.clip((P[:,1]-s.get('y0',-1.2))/0.6,0,1)
P0 = P.copy()
# 1. fair the rim chain hard (Taubin) with the cloth behind it following
ri = np.array([i for i,v in rn.items() if len(v)==2 and reg[i] and kind[i] in ('o','e')])
for r in range(s.get('rounds',120)):
    for lam in (0.5,-0.53):
        A = np.array([P[rn[i]].mean(0) for i in ri]); P[ri] += (lam*fade[ri])[:,None]*(A-P[ri])
# band behind the rim: surface Taubin, fading from the rim inward
from scipy.spatial import cKDTree
d,_ = cKDTree(P[ri]).query(P); wb = np.clip(1-d/s.get('band',0.9),0,1)*fade*reg
bi = np.nonzero(wb>1e-3)[0]
for r in range(s.get('brounds',40)):
    for lam in (0.5,-0.53):
        A = np.array([P[list(nb[i])].mean(0) for i in bi]); P[bi] += (lam*wb[bi])[:,None]*(A-P[bi])
print('rim points faired',len(ri),'moved up to %.3f'%np.linalg.norm(P-P0,axis=1).max())
# 2. move the collar in toward the neck by `in`, never closer than `keep` to the body
b = bpy.data.objects['Redingote bodice']; bw = np.array(b.matrix_world)
BV = np.array([v.co[:] for v in b.data.vertices])@bw[:3,:3].T+bw[:3,3]
bvh = BVHTree.FromPolygons([Vector(p) for p in BV],[tuple(p.vertices) for p in b.data.polygons])
near = np.array([bvh.find_nearest(Vector(p))[0][:] for p in P]); cl = np.linalg.norm(P-near,axis=1)
step = np.minimum(s.get('in',0.12), np.maximum(cl - s.get('keep',0.06),0))*fade*reg
P1 = P + (near-P)/(cl[:,None]+1e-9)*step[:,None]
print('moved in: up to %.3f, %d points'%(step.max(),(step>1e-3).sum())); P = P1
L = P@inv[:3,:3].T+inv[:3,3]
for i,v in enumerate(o.data.vertices): v.co = L[i]
o.data.update(); D2 = dict(D); D2['pos']=P; pickle.dump(D2,open(out.replace('.blend','.pkl'),'wb'))
bpy.ops.wm.save_as_mainfile(filepath=out)
