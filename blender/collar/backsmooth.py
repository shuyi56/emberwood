# backsmooth.py IN.blend COLLAR.pkl OUT.blend '{"y0":0.3,"chain":300,"surf":20,"band":1.0}'
# The back's edges faired: the fold's top line and the fall's outer edge behind the neck, each as one curve through the
# centre back (the two halves tied there), and the cloth beside them carried smooth. Then the halves are welded.
import bpy, sys, pickle, json, numpy as np, bmesh
from scipy.spatial import cKDTree
blend, pk, out, s = sys.argv[-4], sys.argv[-3], sys.argv[-2], json.loads(sys.argv[-1])
D = pickle.load(open(pk,'rb')); T = np.asarray(D['T']); UV = np.asarray(D['UV']); kind = np.asarray(D['kind'])
bpy.ops.wm.open_mainfile(filepath=blend)
o = bpy.data.objects['Collar']; mw = np.array(o.matrix_world); inv = np.linalg.inv(mw)
P = np.array([v.co[:] for v in o.data.vertices]) @ mw[:3,:3].T + mw[:3,3]; N = len(P); assert N == len(kind)
E_ = np.sort(np.concatenate([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]]),1); Eu,c = np.unique(E_,axis=0,return_counts=True); R = Eu[c==1]
nb = [set() for _ in range(N)]
for a,b in Eu: nb[a].add(int(b)); nb[b].add(int(a))
rimn = [set() for _ in range(N)]
for a,b in R: rimn[a].add(int(b)); rimn[b].add(int(a))
# the two halves and their centre-back pairs
par = list(range(N))
def f(i):
    while par[i]!=i: par[i]=par[par[i]]; i=par[i]
    return i
for a,b in Eu: par[f(a)]=f(b)
root = np.array([f(i) for i in range(N)]); roots = np.unique(root); assert len(roots)==2
rimv = np.unique(R)
c0 = rimv[(root[rimv]==roots[0])&(np.abs(UV[rimv,0])<0.4)]; c1 = rimv[(root[rimv]==roots[1])&(np.abs(UV[rimv,0])<0.4)]
d,j = cKDTree(P[c1]).query(P[c0]); ok = d<0.5; pa, pb = c0[ok], c1[j[ok]]
d2,j2 = cKDTree(P[c0]).query(P[c1]); ok2 = (d2<0.5)&~np.isin(c1,pb); pa = np.concatenate([pa,c0[j2[ok2]]]); pb = np.concatenate([pb,c1[ok2]])
is_cb = np.zeros(N,bool); is_cb[pa]=True; is_cb[pb]=True
def tie(P):
    m = (P[pa]+P[pb])/2; m[:,0]=0; P[pa]=m; P[pb]=m
tie(P)
for a,b in zip(pa,pb):
    u = nb[a]|nb[b]; nb[a]=set(u)-{int(a)}; nb[b]=set(u)-{int(b)}
print('centre back tied at %d pairs'%len(pa))
y0 = s.get('y0',0.3); wy = np.clip((P[:,1]-y0)/0.8,0,1); wy = wy*wy*(3-2*wy)
P0 = P.copy()
def chain_ids(sel):
    ids = np.nonzero(sel&(wy>0))[0]; out_ = []
    S = set(int(i) for i in np.nonzero(sel)[0])
    for i in ids:
        n_ = [q for q in nb[i] if q in S and q != i]
        # across the centre back the partner's chain neighbour stands in
        if len(n_) >= 2: out_.append((i, n_))
    return out_
free_cb_rim = lambda i: True
roll = kind=='r'
outer = np.zeros(N,bool); outer[rimv]=True; outer &= (kind=='o')
# (at the centre back the cut's own rim is no free edge any more: leave only the fold line and the outer edge as curves)
for name, sel in (('fold', roll), ('outer edge', outer)):
    al = np.nonzero(sel)[0]; al = al[np.argsort(UV[al, 0], kind='stable')]
    if len(al) < 5: continue
    pos_in = {int(v): q for q, v in enumerate(al)}
    idx = np.array([i for i in al[1:-1] if wy[i] > 0])
    if not len(idx): continue
    J = [(al[pos_in[int(i)] - 1], al[pos_in[int(i)] + 1]) for i in idx]
    J = np.array(J); before = P[idx].copy()
    for r in range(s.get('chain',300)):
        for lam in (0.5,-0.53):
            P[idx] += (lam*wy[idx])[:,None]*(0.5*(P[J[:,0]]+P[J[:,1]])-P[idx]); tie(P)
    mv = P[idx]-before
    # the cloth beside it carried along, fading over the band
    dq,jq = cKDTree(before).query(P0); wb = np.clip(1-dq/s.get('band',1.0),0,1); wb = wb*wb*(3-2*wb)
    oth = np.ones(N,bool); oth[idx]=False
    P[oth] += (wb[oth]*wy[oth])[:,None]*mv[jq[oth]]; tie(P)
    print('%s: %d points faired, moved up to %.3f'%(name,len(idx),np.linalg.norm(mv,axis=1).max()))
    P0 = P.copy()
# the surface between: a few rounds, the two curves held
hold = roll|outer|(kind=='s'); w = wy*(~hold); si = np.nonzero(w>1e-3)[0]
for r in range(s.get('surf',20)):
    for lam in (0.5,-0.53):
        A = np.array([P[list(nb[i])].mean(0) for i in si]); P[si] += (lam*w[si])[:,None]*(A-P[si]); tie(P)
# the centre back itself: the join's own corner eased (the two halves' corners make a tab at the fold's top)
wc = np.clip(1-np.abs(P[:,0])/s.get('cb_w',0.6),0,1)*wy*(kind!='s'); ci = np.nonzero(wc>1e-3)[0]
for r in range(s.get('cb',25)):
    A = np.array([P[list(nb[i])].mean(0) for i in ci]); P[ci] += (0.5*wc[ci])[:,None]*(A-P[ci]); tie(P)
L = P@inv[:3,:3].T+inv[:3,3]
for i,v in enumerate(o.data.vertices): v.co = L[i]
o.data.update()
bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4); bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary and abs(e.verts[0].co.x) < 0.02 and abs(e.verts[1].co.x) < 0.02], sides=6); bmesh.ops.triangulate(bm, faces=bm.faces); bm.to_mesh(o.data); bm.free()
for p in o.data.polygons: p.use_smooth = True
bpy.ops.wm.save_as_mainfile(filepath=out)
