import bpy, sys, pickle, numpy as np
blend, pk, out = sys.argv[-3], sys.argv[-2], sys.argv[-1]
D = pickle.load(open(pk,'rb')); T = np.asarray(D['T']); UV = np.asarray(D['UV'])
bpy.ops.wm.open_mainfile(filepath=blend)
o = bpy.data.objects['Collar']; mw = np.array(o.matrix_world); inv = np.linalg.inv(mw)
P = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3,:3].T + mw[:3,3]
E = np.sort(np.concatenate([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]]),1); Eu,c = np.unique(E,axis=0,return_counts=True); R = Eu[c==1]
# the centre-back cut: rim edges of each half whose both ends lie at the half's inner border
n0 = len(P)//2
cb = []
for h, sel in ((0, np.arange(len(P)) < 0), ):
    pass
rimv = np.unique(R)
# rim vertices close to another half's rim vertex are the cut
from scipy.spatial import cKDTree
A = rimv[P[rimv,0] < 0.35]; A = A[np.abs(UV[A,0]) < 0.4]
neg = A[UV[A,0] <= np.median(UV[A,0])] if False else None
comp = None
# split by which connected half the vertex is in
import collections
par = list(range(len(P)))
def f(i):
    while par[i]!=i: par[i]=par[par[i]]; i=par[i]
    return i
for a,b in Eu: par[f(a)]=f(b)
root = np.array([f(i) for i in range(len(P))]); roots = np.unique(root)
print('pieces', len(roots))
if len(roots) == 2:
    c0 = rimv[(root[rimv]==roots[0]) & (np.abs(UV[rimv,0])<0.4)]; c1 = rimv[(root[rimv]==roots[1]) & (np.abs(UV[rimv,0])<0.4)]
    d,j = cKDTree(P[c1]).query(P[c0]); ok = d < 0.5
    m = (P[c0[ok]] + P[c1[j[ok]]])/2; m[:,0] = 0
    mv = np.linalg.norm(P[c0[ok]]-m,axis=1).max()
    P[c0[ok]] = m; P[c1[j[ok]]] = m
    d2,j2 = cKDTree(P[c0]).query(P[c1]); ok2 = d2 < 0.5; P[c1[ok2]] = P[c0[j2[ok2]]]
    print('centre back: %d + %d points joined, moved up to %.3f' % (ok.sum(), ok2.sum(), mv))
L = P@inv[:3,:3].T+inv[:3,3]
for i,v in enumerate(o.data.vertices): v.co = L[i]
o.data.update()
import bmesh
bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4); bm.to_mesh(o.data); bm.free()
for p in o.data.polygons: p.use_smooth = True
bpy.ops.wm.save_as_mainfile(filepath=out)
