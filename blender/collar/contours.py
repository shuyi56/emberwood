"""The contour sheet for the bodice front's fold: looked at after the solver and before any render.

    python3 blender/collar/contours.py fold.pkl sheet.png

fold.pkl (written by the build under COLLAR_FOLD_OUT, or by the fold rig): V, F (the bodice's mesh after the fold, world),
skinP, skinF (the body), M, outlines, folded (collar.FOLDED). The sheet: level sections up the V, sections square
across each roll line (the fold's own profile: it should be one smooth hairpin, the lapel lying parallel to the front),
side sections, the front outline of the cut edges, and the bend of each hinge drawn on the flat pattern."""
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def tris_of(F):
    return np.array([(f[0], f[j], f[j + 1]) for f in F for j in range(1, len(f) - 1)])


def section(V, T, n, c):
    """The mesh cut by the plane n.x = c: segments [m, 2, 3]."""
    d = V @ n - c
    D = d[T]
    hit = (D.min(1) < 0) & (D.max(1) > 0)
    segs = []
    for t, dd in zip(T[hit], D[hit]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (dd[a] < 0) != (dd[b] < 0):
                k = dd[a] / (dd[a] - dd[b])
                pts.append(V[t[a]] + k * (V[t[b]] - V[t[a]]))
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs).reshape(-1, 2, 3)


def hinge_angles(X, T):
    """Per inner edge: (i, j, the fold across it in degrees)."""
    T = np.asarray(T)
    nrm = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    seen, out = {}, []
    for fi, t in enumerate(T):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (min(t[a], t[b]), max(t[a], t[b]))
            if key in seen:
                out.append((key[0], key[1], np.degrees(np.arccos(np.clip(nrm[fi] @ nrm[seen[key]], -1, 1)))))
            else:
                seen[key] = fi
    return np.array(out)


def sheet(D, out):
    V, T = np.asarray(D['V']), tris_of(D['F'])
    S, ST = np.asarray(D['skinP']), tris_of(D['skinF'])
    M, outlines, folded = D['M'], D['outlines'], D['folded']
    yc = M['yc']
    zb = min(B[1] for B, _, _ in outlines)
    zn = max(N[1] for _, N, _ in outlines)
    keep = (V[T][:, :, 1].max(1) < yc) & (V[T][:, :, 2].min(1) > zb - 1.5)
    T = T[keep]
    ks = (S[ST][:, :, 1].max(1) < yc + 1) & (S[ST][:, :, 2].min(1) > zb - 1.5) & (S[ST][:, :, 2].max(1) < zn + 2)
    ST = ST[ks]
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))

    def draw(a, n, c, u, v, title, lim=None):
        for (P_, T_, col, lw) in ((S, ST, '0.7', 1.0), (V, T, 'k', 1.2)):
            sg = section(P_, T_, n, c)
            if len(sg):
                a.add_collection(LineCollection(np.stack([sg @ u, sg @ v], -1), colors=col, linewidths=lw))
        a.set_aspect('equal')
        a.autoscale()
        if lim:
            a.set_xlim(*lim[0])
            a.set_ylim(*lim[1])
        a.set_title(title, fontsize=10)
    ex, ey, ez = np.eye(3)
    for k, f in enumerate((0.15, 0.4, 0.65, 0.9)):
        z = zb + f * (zn - zb)
        draw(ax[0, k], ez, z, ex, -ey, 'level section at z %.2f (%.0f%% up the V); front is up' % (z, 100 * f),
             ((-4.5, 4.5), (-yc - 0.3, -yc + 4.2)))
    col = 0
    for k, (B, N, _) in enumerate(outlines):                 # (square across each roll: the fold's profile)
        e = np.array([N[0] - B[0], 0, N[1] - B[1]])
        e /= np.linalg.norm(e)
        acr = np.cross(e, ey)
        for f in (0.35, 0.7):
            p = np.array([B[0], 0, B[1]]) + f * np.linalg.norm(N - B) * e
            c0 = p @ acr
            draw(ax[1, col], e, p @ e, acr, -ey, 'across roll %d at %.0f%% of its length (front is up)' % (k, 100 * f),
                 ((c0 - 2.0, c0 + 2.0), (-yc + 0.3, -yc + 4.3)))
            col += 1
    for k, x in enumerate((-1.2,)):
        draw(ax[2, k], ex, x, -ey, ez, 'side section at x %.1f (front is right)' % x, ((-yc + 0.5, -yc + 4.5), (zb - 1, zn + 1.5)))
    # the strain on the flat pattern: each edge's length folded against its length as drafted
    a = ax[2, 1]
    stx = []
    for k, L in enumerate(folded):
        T_ = np.asarray(L['T'])
        E = np.unique(np.sort(np.vstack([T_[:, [0, 1]], T_[:, [1, 2]], T_[:, [2, 0]]]), 1), axis=0)
        E = E[L['lap'][E].any(1)]
        P2, X = L['flat'], L['pos']
        st = 100 * (np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / np.linalg.norm(P2[E[:, 0]] - P2[E[:, 1]], axis=1) - 1)
        lc = LineCollection(P2[E], array=st, cmap='coolwarm', linewidths=1.0)
        lc.set_clim(-10, 10)
        a.add_collection(lc)
        stx.append('lapel %d: strain 5%% %.1f, median %.1f, 95%% %.1f, worst %.1f / %.1f %%' % (
            k, np.percentile(st, 5), np.median(st), np.percentile(st, 95), st.min(), st.max()))
    plt.colorbar(lc, ax=a, label='strain, % (clipped at 10)')
    a.set_aspect('equal')
    a.autoscale()
    a.set_title('strain on the flat pattern\n' + '\n'.join(stx), fontsize=9)
    print('\n'.join(stx))
    # the front outline: the cut edges (boundary edges of the cloth), from the front
    a = ax[2, 2]
    cnt = {}
    for t in T:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = (min(t[i], t[j]), max(t[i], t[j]))
            cnt[key] = cnt.get(key, 0) + 1
    bd = np.array([k for k, c in cnt.items() if c == 1])
    a.add_collection(LineCollection(V[bd][:, :, [0, 2]], colors='k', linewidths=1.0))
    for B, N, poly in outlines:
        a.plot([B[0], N[0]], [B[1], N[1]], 'r--', lw=0.8)
        a.plot(*np.vstack([poly, poly[:1]]).T, color='tab:blue', lw=0.6)
    a.set_aspect('equal')
    a.set_xlim(-4.5, 4.5)
    a.set_ylim(zb - 1, zn + 1.5)
    a.set_title('front: cut edges (black), roll lines (red), lapels as drafted (blue)', fontsize=10)
    # the bend on the flat pattern
    a = ax[2, 3]
    txt = []
    for k, L in enumerate(folded):
        H = hinge_angles(L['pos'], L['T'])
        P2 = L['flat']
        sg = np.stack([P2[H[:, 0].astype(int)], P2[H[:, 1].astype(int)]], 1)
        lc = LineCollection(sg, array=H[:, 2], cmap='viridis', linewidths=1.0)
        lc.set_clim(0, 60)
        a.add_collection(lc)
        a.plot(*L['roll'].T, 'r--', lw=0.6)
        r0, r1 = L['roll']
        q = P2[H[:, 0].astype(int)] - r0
        sd = np.abs((r1 - r0)[0] * q[:, 1] - (r1 - r0)[1] * q[:, 0]) / np.linalg.norm(r1 - r0)
        far = H[sd > 0.6, 2]
        txt.append('lapel %d: bend away from the roll band: median %.1f, 95%% %.1f, max %.1f deg' % (k, np.median(far), np.percentile(far, 95), far.max()))
    plt.colorbar(lc, ax=a, label='fold across the edge, degrees (clipped at 60)')
    a.set_aspect('equal')
    a.autoscale()
    a.set_title('bend on the flat pattern\n' + '\n'.join(txt), fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=80)
    print('\n'.join(txt))
    print('wrote', out)


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as fh:
        sheet(pickle.load(fh), sys.argv[2])
