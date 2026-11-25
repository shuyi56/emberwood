"""flatview.py BUILD OUT.png: the flat cut beside the outline from the photo's camera, for the collar and the lapel,
each side marked as a cut edge or as a fold / occlusion edge (made by the roll, the neck or the view, not by the cut)"""
import sys, numpy as np, math
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polycheck as pc
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B, out = sys.argv[1], sys.argv[2]; az = 30.0
def loop(T):
    ed = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1); u, c = np.unique(ed, axis=0, return_counts=True); E = u[c == 1]
    nb = {}
    for a, b in E: nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    loops, seen = [], set()
    for s in nb:
        if s in seen: continue
        L = [s]; seen.add(s); p = None; cur = s
        while True:
            nx = [q for q in nb[cur] if q != p and q not in seen]
            if not nx: break
            p, cur = cur, nx[0]; L.append(cur); seen.add(cur)
        loops.append(L)
    return max(loops, key=len)
fig, axs = plt.subplots(1, 4, figsize=(22, 6))
# collar, flat
G = np.load(B + '/collar_gauge.npz'); U = G['U']; lp = loop(G['T']); Pf = U[lp]; Pf = Pf / np.ptp(Pf[:, 0])
cf = pc.corners(Pf, True, eps_frac=0.02)
ax = axs[0]; ax.plot(*np.vstack([Pf, Pf[:1]]).T, color='tab:blue'); ax.plot(*Pf[cf].T, 'ko', ms=5)
ax.set_aspect('equal'); ax.set_title('collar, flat cut (whole collar): %d corners' % len(cf))
# collar, from the camera: each outline point is a cut edge if it lies on the collar mesh's own edge in the view
res = pc.check(B, az); cp = [p for p in res['pieces'] if p['piece'] == 'collar front'][0]
Pc = np.array(res['collar_outline']); _, unit = pc.lapel_outline(B, az)
M = np.load(B + '/build.npz', allow_pickle=True); R, D = pc.camera(az)
bl = loop(M['collar_T']); Xb = M['collar_X'][bl]; Xb2 = pc.to_unit(np.column_stack([Xb @ R, Xb[:, 2]]), unit)
from scipy.spatial import cKDTree
# (densify the cut edge before the distance test)
Xd = np.vstack([Xb2[i] + f * (Xb2[(i + 1) % len(Xb2)] - Xb2[i]) for i in range(len(Xb2)) for f in (0, 0.25, 0.5, 0.75)])
cut = cKDTree(Xd).query(Pc)[0] < 0.02
ax = axs[1]
for msk, col, lab in ((cut, 'tab:blue', 'cut edge'), (~cut, 'tab:red', 'fold / occlusion edge')):
    ax.plot(*Pc[msk].T, '.', color=col, ms=2.5, label=lab)
C = np.array(cp['corners']); ax.plot(*np.vstack([C, C[:1]]).T, 'k-o', ms=5, lw=0.8)
ax.set_aspect('equal'); ax.legend(loc='lower right', fontsize=8)
ax.set_title('collar front from the photo\'s side: %d sides (photo 4)' % cp['model_sides'])
# lapel, flat (t, d) and from the camera
L = np.load(B + '/lapel.blend.lapels.npz'); t, d, e = L['t2'], L['d2'], L['edge2']; pts = L['pts2']
ch = [np.array([[0.0, 0.0]])]
if 'inner2' in L.files: mi = L['inner2'] & ~e; ch.append(np.column_stack([t[mi], d[mi]])[np.argsort(d[mi])])
ch.append(np.column_stack([t[e], d[e]])[np.argsort(t[e])]); ch.append(np.array([[1.0, 0.0]]))
Fl = np.vstack(ch); Lc = np.linalg.norm(L['crease2'][-1] - L['crease2'][0]); Fl2 = np.column_stack([Fl[:, 1], -Fl[:, 0] * Lc])
cl = pc.corners(Fl2, True)
ax = axs[2]; ax.plot(*np.vstack([Fl2, Fl2[:1]]).T, color='tab:blue'); ax.plot([0, 0], [0, -Lc], color='tab:red', lw=2, label='roll line: a fold')
ax.plot(*Fl2[cl].T, 'ko', ms=5); ax.set_aspect('equal'); ax.legend(fontsize=8); ax.set_title('lapel, flat (across the roll line, down it): %d corners' % len(cl))
lpc = [p for p in res['pieces'] if p['piece'] == 'lapel'][0]; Pl = np.array(res['lapel_outline'])
ax = axs[3]; ax.plot(*np.vstack([Pl, Pl[:1]]).T, color='tab:blue'); C = np.array(lpc['corners']); ax.plot(*np.vstack([C, C[:1]]).T, 'k-o', ms=5, lw=0.8)
ax.set_aspect('equal'); ax.set_title('lapel from the photo\'s side: %d sides (photo 4)' % lpc['model_sides'])
plt.suptitle(B.split('/')[-1] + ': blue = cut edge, red = an edge made by the fold or the view'); plt.tight_layout(); plt.savefig(out, dpi=70)
print('collar flat corners', len(cf), 'collar front sides', cp['model_sides'], 'cut share %.2f' % cut.mean(), 'lapel flat corners', len(cl), 'lapel sides', lpc['model_sides'])
# which model collar-front sides are cut / fold
ci = [int(np.argmin(np.linalg.norm(Pc - q, axis=1))) for q in np.array(cp['corners'])]
for i in range(len(ci)):
    a, b = ci[i], ci[(i + 1) % len(ci)]; n = len(Pc); idx = [(a + q) % n for q in range((b - a) % n + 1)]
    print('  side %d from (%.2f,%.2f) to (%.2f,%.2f): %.0f%% cut edge' % (i, *Pc[a], *Pc[b], 100 * cut[idx].mean()))
