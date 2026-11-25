"""fitref.py BUILD...: the lapel/collar polygon of her left side seen from the reference's side (az 30, ortho), in
units of the lapel's lower edge's span (visible break L0 to lapel point P, across), against the riding-habit photo"""
import sys, math, numpy as np
REF = dict(L0=(590, 302), P=(693, 292), G=(675, 263), C=(705, 272), N=(667, 250))   # photo px, her left side (traced on the gamma-lifted close-up)
def norm(D):
    L0 = np.array(D['L0'], float); s = D['P'][0] - L0[0]
    return {k: np.array([(v[0] - L0[0]) / s, -(v[1] - L0[1]) / s]) for k, v in D.items()}
def model(B, az=30.0):
    th = math.radians(az); R = np.array([math.cos(th), math.sin(th), 0])
    Pj = lambda p: np.array([p @ R, p[2]])
    L = np.load(B + '/lapel.blend.lapels.npz'); C = np.load(B + '/out.blend.collar.npz')
    pts, t, d, e = L['pts2'], L['t2'], L['d2'], L['edge2']
    P = pts[np.argmax(np.where(e, d, -1))]; ee = pts[e]; L0 = ee[np.argmax(t[e])]; N = L['crease2'][0]
    tri = C['tri']; ed = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    u, c = np.unique(ed, axis=0, return_counts=True); bv = np.unique(u[c == 1]); X = C['pts'][bv]; X = X[X[:, 0] > 0]
    from scipy.spatial import cKDTree
    dg = cKDTree(ee).query(X)[0]; g = X[dg < 0.3]; G = g[np.argmax(np.linalg.norm(g - N, axis=1))]
    cand = X[(dg > 0.3) & (np.linalg.norm(X - G, axis=1) < 2.0)]; Cp = cand[np.argmax(np.linalg.norm(cand - N, axis=1))] if len(cand) else G   # (the collar's point: off the gorge, near the notch, farthest from the roll top)
    import re, glob
    lg = open(B + '/collar.log').read(); m = re.search(r'front 2: notch G at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\), collar point C at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', lg)
    if m: v_ = [float(x) for x in m.groups()]; G = np.array(v_[:3]); Cp = np.array(v_[3:])     # (as notchcollar cut them)
    D = {k: Pj(v) * np.array([1, -1]) for k, v in dict(L0=L0, P=P, G=G, C=Cp, N=N).items()}   # (y down, as the photo)
    return norm(D)
def angle(a, b): return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
def lines(D):
    G, P, C, N, L0 = D['G'], D['P'], D['C'], D['N'], D['L0']
    na = abs(((angle(G, P) - angle(G, C)) + 180) % 360 - 180)
    return dict(lower_edge=angle(L0, P), end_edge=angle(P, G), gorge=np.linalg.norm(G - N), lapel_end=np.linalg.norm(P - G),
                collar_end=np.linalg.norm(C - G), collar_dir=angle(G, C), notch=na, roll_rise=N[1])
if __name__ == '__main__':
    R_ = norm(REF); Lr = lines(R_)
    for B in sys.argv[1:]:
        M = model(B); Lm = lines(M); err = sum(np.sum((M[k] - R_[k]) ** 2) for k in R_)
        print('%-40s polygon error %.3f' % (B, err))
        for k in Lr: print('   %-11s ref %7.2f  model %7.2f' % (k, Lr[k], Lm[k]))
        for k in R_: print('   %-3s ref (%.2f, %.2f) model (%.2f, %.2f)' % (k, *R_[k], *M[k]))
        # and the polygon check after every fit (John, 2026-11-23: "Add a proper polygon check with side count")
        import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import polycheck
        print(polycheck.report(polycheck.check(B.rstrip('/'))))
