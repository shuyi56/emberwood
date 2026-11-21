# swapcollar.py BUILD.blend COLLAR.pkl OUT.blend: replaces the build's Collar mesh with a solved collar (edge points faired first).
import bpy, sys, pickle, numpy as np
bpy.ops.wm.open_mainfile(filepath=sys.argv[-3])
D = pickle.load(open(sys.argv[-2], 'rb'))
def smooth_edges(pos, T, rounds=4, k=0.5):
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    E, c = np.unique(E, axis=0, return_counts=True); B = E[c == 1]; nb = {}
    for a, b in B: nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    idx = np.array([i for i, v in nb.items() if len(v) == 2]); J = np.array([nb[i] for i in idx]); P = pos.copy()
    for _ in range(rounds): P[idx] += k * (0.5 * (P[J[:, 0]] + P[J[:, 1]]) - P[idx])
    return P
o = bpy.data.objects['Collar']; inv = np.linalg.inv(np.array(o.matrix_world))
P = smooth_edges(np.asarray(D['pos']), np.asarray(D['T']))
L = P @ inv[:3, :3].T + inv[:3, 3]
me = bpy.data.meshes.new('Collar2')
me.from_pydata([tuple(p) for p in L], [], [tuple(int(i) for i in t) for t in D['T']]); me.update()
for m in o.data.materials: me.materials.append(m)
for p in me.polygons: p.use_smooth = True
o.data = me
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1])
