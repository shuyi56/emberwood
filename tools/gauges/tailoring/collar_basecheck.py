"""Base check: the surface and the joins a lapel and collar are built on, measured before any fold is tuned on top.

    python3 blender/collar/basecheck.py FOLD.pkl COLLAR.pkl BUILD.log [--json OUT]

FOLD.pkl and COLLAR.pkl are the build's COLLAR_FOLD_OUT and COLLAR_OUT; BUILD.log its log. Each check prints its
number, its limit, where the limit comes from, and PASS or FAIL; the exit status is 1 if any fails.

  roll bend      the roll line's bend out of the chest (normal curvature) along each lapel's roll, on the folded
                 front. The turned lapel is a double layer 2 x fold_radius thick folded round the roll; bent along
                 the crease at radius R its inner layer is squeezed by about t / (2R). Held to 10 %, R >= 1.6 units
                 (4.2 cm): |kn| <= 0.6 per unit. (The modern draft before the chest canvas: 1.1, fold opened to 144.)
  roll sideways  its bend within the surface (geodesic curvature): a straight roll on the flat pattern is a geodesic
                 on the garment, so it should not turn: |kg| <= 0.6 per unit, by the same strain limit.
  hollow         the bodice front under each lapel's outline: how far it sinks below a string pulled taut over it
                 from the bust line up, column by column (hair canvas cannot follow a hollow: UKY CT-LMH.176, ASG
                 chest shield). Limit 0.1 units (2.6 mm, about the cloth's thickness).
  gorge held     the share of each gorge's length the lapel's edge is held along (sewn to the collar). Limit 60 %.
  gorge length   the collar's gorge as cut against the lapel's edge it is sewn to (seams pair edge for edge). 5 %.
"""
import json
import math
import pickle
import re
import sys

import numpy as np
from scipy.spatial import cKDTree


def _upper(z, r):
    o = np.argsort(z)
    z, r = z[o], r[o]
    H = []
    for p in zip(z, r):
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (p[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) >= 0:
            H.pop()
        H.append(p)
    H = np.array(H)
    out = np.empty_like(r)
    out[o] = np.interp(z, H[:, 0], H[:, 1])
    return out


def roll_curvature(L, outline, bodyP, bodyN):
    """(max |normal curvature|, max |geodesic curvature|) along one lapel's roll, from the folded front."""
    lap = np.asarray(L['lap'], bool)
    flat, pos = np.asarray(L['flat'])[:, :2], np.asarray(L['pos'])
    B, N, _ = outline
    u = (N - B) / np.linalg.norm(N - B)
    nperp = np.array([-u[1], u[0]])
    t, sd = (flat - B) @ u, (flat - B) @ nperp
    rp = np.nonzero(~lap & (np.abs(sd) < 0.015))[0]
    rp = rp[np.argsort(t[rp])]
    P = pos[rp]
    a = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    s = np.linspace(0, a[-1], 40)
    P = np.column_stack([np.interp(s, a, P[:, k]) for k in range(3)])
    for _ in range(3):
        P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
    kd = cKDTree(bodyP)
    n = np.array([bodyN[kd.query(p, k=12)[1]].mean(0) for p in P])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    T = np.gradient(P, axis=0)
    ds = np.linalg.norm(T, axis=1)
    T /= ds[:, None]
    dT = np.gradient(T, axis=0) / ds[:, None]
    kn, kg = (dT * n).sum(1), (dT * np.cross(n, T)).sum(1)
    i = int(np.argmax(np.abs(kn)))
    return float(np.abs(kn[2:-2]).max()), float(np.abs(kg[2:-2]).max()), float(P[i, 2])


def hollow(V, outline, M):
    """The deepest the front sinks below a taut string, inside the lapel's outline (front view), bust line up."""
    import shapely
    B, N, poly = outline
    pg = shapely.Polygon(poly).buffer(0.2)
    zl = M.get('apex_z', M['z_bust'])
    m = (V[:, 1] < M['yc']) & (V[:, 2] > zl - 0.5) & shapely.contains_xy(pg, V[:, 0], V[:, 2])
    worst, at = 0.0, None
    P = V[m]
    for x0 in np.arange(P[:, 0].min(), P[:, 0].max(), 0.1):
        c = np.abs(P[:, 0] - x0) < 0.05
        if c.sum() < 6:
            continue
        Q = P[c]
        r = -(Q[:, 1] - M['yc'])
        d = _upper(Q[:, 2], r) - r
        j = int(np.argmax(d))
        if d[j] > worst:
            worst, at = float(d[j]), (round(float(x0), 2), round(float(Q[j, 2]), 2))
    return worst, at


def main():
    fold, col, logp = sys.argv[1:4]
    F = pickle.load(open(fold, 'rb'))
    log = open(logp).read()
    M = F['M']
    res = []

    def check(name, val, lim, ok, src, where=''):
        res.append(dict(check=name, value=round(val, 3) if isinstance(val, float) else val, limit=lim, ok=bool(ok), source=src, where=where))
    for k, (L, out) in enumerate(zip(F['folded'], F['outlines'])):
        bP, bN = L['body']
        kn, kg, z = roll_curvature(L, out, np.asarray(bP), np.asarray(bN))
        check('lapel %d roll bend (|kn|)' % k, kn, 0.6, kn <= 0.6, 'double layer t/2R <= 10 %', 'sharpest at z %.2f' % z)
        check('lapel %d roll sideways (|kg|)' % k, kg, 0.6, kg <= 0.6, 'roll a geodesic; same strain limit')
        hw, at = hollow(np.asarray(F['V']), out, M)
        check('lapel %d hollow under it' % k, hw, 0.1, hw <= 0.1, 'canvas bridges hollows (UKY, ASG)', 'at x, z %s' % (at,))
        B, N, poly = out
        i0 = int(np.argmin(np.linalg.norm(np.asarray(poly) - N, axis=1)))
        G = min((poly[i0 - 1], poly[(i0 + 1) % len(poly)]), key=lambda q: abs((q - N) @ ((B - N) / np.linalg.norm(B - N))) / max(np.linalg.norm(q - N), 1e-9))
        glen = float(np.linalg.norm(np.asarray(G) - N))
        mm = re.findall(r'front %d: (\d+) points of the gorge held' % k, log)
        held = int(mm[-1]) if mm else 0
        share = min(1.0, held * 0.07 / max(glen, 1e-6))
        check('lapel %d gorge held' % k, share, 0.6, share >= 0.6, 'the gorge seam is sewn its whole length', '%d points on a gorge %.2f long' % (held, glen))
    mm = re.findall(r'pattern back/front/gorge ([\d.]+) ([\d.]+) ([\d.]+); cloth ([\d.]+) ([\d.]+) ([\d.]+)', log)
    if mm:
        g_pat, g_cloth = float(mm[-1][2]), float(mm[-1][5])
        d = abs(g_pat / g_cloth - 1)
        check('collar gorge vs lapel edge length', d, 0.05, d <= 0.05, 'seam edges pair edge for edge', 'cut %.2f, lapel edge %.2f' % (g_pat, g_cloth))
    bad = 0
    for r in res:
        print('%-36s %8s  limit %-5s %s   (%s) %s' % (r['check'], r['value'], r['limit'], 'PASS' if r['ok'] else 'FAIL', r['source'], r['where']))
        bad += not r['ok']
    if '--json' in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
