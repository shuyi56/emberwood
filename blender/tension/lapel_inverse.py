"""lapel_inverse.py LAPELDIR: the inverse of the lapel's bend. On a solved lapel (after it settles), fit the map from
the flat cut (t down the roll line, d out from it, in fold lengths) to the view from the photo's camera (az 30), then
put the photo's corners G and P through the inverse map: the cut corners (outline [[tG, dG], [tP, dP], [1, 0]],
outline_inner [[tG, dG], [0, 0]]) that land on the photo. One rebuild took the polygon error from 0.07 to 0.001.
The roll line (foot, neck_x) is set first, to the photo's roll angle, which this prints."""
import sys, math, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import polycheck as pc
B = sys.argv[1]; az = 30
R, _ = pc.camera(az); L = np.load(B + '/lapel.blend.lapels.npz')
pts, t, d, e, cr = L['pts2'], L['t2'], L['d2'], L['edge2'], L['crease2']
Lc = np.linalg.norm(cr[-1] - cr[0])
X = np.column_stack([pts @ R, pts[:, 2]])
sel = (t < 0.3)          # (the upper lapel, where G and P lie)
A = np.column_stack([t[sel], d[sel] / Lc, np.ones(sel.sum())]); M = np.linalg.lstsq(A, X[sel], rcond=None)[0]
print('affine residual', np.sqrt(np.mean((A @ M - X[sel]) ** 2)))
N2 = np.array([cr[0] @ R, cr[0][2]]); L02 = X[e][np.argmax(t[e])]
ref = {k: np.array([p[0], -p[1]], float) for k, p in zip(['L0', 'N', 'G', 'P'], pc.REF['lapel']['pts'])}
# similarity from the photo to the model taking the photo's L0, N onto the model's
def sim(a0, a1, b0, b1):
    va, vb = a1 - a0, b1 - b0; s = np.linalg.norm(vb) / np.linalg.norm(va); th = math.atan2(vb[1], vb[0]) - math.atan2(va[1], va[0])
    Rm = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]); return lambda p: b0 + Rm @ (p - a0)
f = sim(ref['L0'], ref['N'], L02, N2)
print('roll angle: model %.0f, photo %.0f' % (math.degrees(math.atan2(*(N2 - L02)[::-1])), math.degrees(math.atan2(*(ref['N'] - ref['L0'])[::-1]))))
Minv = M[:2]
for k in ('G', 'P'):
    q = f(ref[k]); td = np.linalg.solve(Minv.T, q - M[2]); print(k, 't %.3f d %.3f' % (td[0], td[1]))
