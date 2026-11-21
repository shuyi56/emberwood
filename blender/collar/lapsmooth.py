# lapsmooth.py BUILD.blend OUT.blend (env COLLAR_FOLD_PKL = the build's fold pickle): fairs the lapels' free edges in the bodice mesh.
import bpy, sys, pickle, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collarlab as CL
from scipy.spatial import cKDTree
blend, out = sys.argv[-2], sys.argv[-1]
F = pickle.load(open(os.environ['COLLAR_FOLD_PKL'],'rb'))
bpy.ops.wm.open_mainfile(filepath=blend)
b = bpy.data.objects['Redingote bodice']; mw = np.array(b.matrix_world); inv = np.linalg.inv(mw)
W = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3,:3].T + mw[:3,3]
kd = cKDTree(W)
for k, L in enumerate(F['folded']):
    pos, T, lap, flat = np.asarray(L['pos'],float), np.asarray(L['T']), np.asarray(L['lap'],bool), np.asarray(L['flat'])
    rim = CL._rim(T)
    kind = np.where(lap, 'i', 's').astype(object)
    kind[rim[lap[rim]]] = 'o'
    # (the turn's own band stays: its rim points near the roll are left as sewn)
    B, N, _ = F['outlines'][k]; u = (N-B)/np.linalg.norm(N-B)
    sd = np.abs((flat-B)[:,0]*u[1]-(flat-B)[:,1]*u[0])
    kind[(sd < 0.45)] = 's'
    D = dict(pos=pos, T=T, UV=flat, kind=np.array(kind, dtype=str))
    new = CL.smooth_edges_pass(D, rounds=150, band=1.0, inner_rounds=60, roll=False, corner_deg=35)
    d, j = kd.query(pos)
    ok = d < 1e-4
    for i in np.nonzero(ok & (np.linalg.norm(new-pos,axis=1) > 1e-7))[0]:
        b.data.vertices[int(j[i])].co = new[i] @ inv[:3,:3].T + inv[:3,3]
    print('lapel %d: %d of %d points matched in the bodice' % (k, ok.sum(), len(pos)))
b.data.update()
bpy.ops.wm.save_as_mainfile(filepath=out)
