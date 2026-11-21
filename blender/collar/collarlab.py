"""Collar lab: the tailored top collar alone, on the solver, outside the build (John, 2026-11-21: "build the standalone
on a solver"). A build's COLLAR_FOLD_OUT pickle has everything the collar meets (the bodice with its lapels turned
back, the body, the neck edge, the gorges); tailor.tailor_collar is run on it with a try's settings, in about 3
minutes, and the result saved for a clay render (swap into the build's .blend) and its numbers printed.

    COLLAR_STYLE=modern python3 blender/collar/collarlab.py FOLD.pkl OUT.pkl '{"swing": 0.0, "gorge_ext": 0.5}'

A try's JSON is tailor.SETTINGS' keys (swing, end, notch, lay_r, crease_on_lapel, ...) and the lab's own:
  fit_fold    true: each side's collar end is drafted against its own lapel as it lies folded (John, 2026-11-21:
              "redraft the left and right side to account for the fold of the lapel so the same angle is preserved").
              The draft set the notch against the lapel's top edge as drawn flat (falling 25 deg); turned back over the
              chest that edge falls 46 deg on her right and 36 on her left, so the same collar end opened 45 deg on one
              side and 38 on the other. Here the lapel's notch and point are measured on the folded front per side,
              the collar is solved once for each side (the left in a mirrored world), and the two halves joined at the
              centre back. 'notch' is then the opening as seen.
  gorge_ext   the gorge seam carried on along the lapel's top edge by this share of it (notch toward the lapel's
              point): the collar's front reaches further down the chest and its end grows with it, as on the 1890s
              habit, whose collar takes the top 40 % of the V. 0 is the build's notch.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('collar', 'shared', 'base', 'redingote_bodice', 'redingote', 'vest', 'chemise', 'sorceress_pendant'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
import tailor      # noqa: E402


def _rim(T):
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    e, c = np.unique(e, axis=0, return_counts=True)
    return np.unique(e[c == 1])


def folded_notch(f):
    """{side: (G, P)}: each lapel's notch and point in front view (x, z), where the fold laid them."""
    r_ = [o for o in f['outlines'] if o[1][0] < 0][0]
    out = {}
    for k, (B, N, poly) in enumerate(f['outlines']):
        sg = 1 if N[0] > 0 else -1
        L = f['folded'][k]
        P, T, lap, X = np.asarray(L['flat']), np.asarray(L['T']), np.asarray(L['lap'], bool), np.asarray(L['pos'])
        u = (N - B) / np.linalg.norm(N - B)
        mir = lambda q: B + 2 * ((q - B) @ u) * u - (q - B)
        rl = _rim(T)
        rl = rl[lap[rl]]
        Gd = np.asarray(f['gorges'][sg][1])
        Pd = np.array([abs(r_[2][3][0]) * sg, r_[2][3][1]])      # (the left's outline is clipped under the right: the right's point mirrored)
        iG = rl[np.argmin(np.linalg.norm(P[rl] - mir(Gd), axis=1))]
        iP = rl[np.argmin(np.linalg.norm(P[rl] - mir(Pd), axis=1))]
        out[sg] = (X[iG][[0, 2]], X[iP][[0, 2]])
    return out


def _half(D, mirror):
    """The half of a solved collar on the side it was drafted for (the pattern's x >= 0), as (UV, pos, T, kind)."""
    UV, T = np.asarray(D['UV']), np.asarray(D['T'])
    keepT = T[(UV[T][:, :, 0] > -1e-6).all(1)]
    used = np.unique(keepT)
    re = -np.ones(len(UV), int)
    re[used] = np.arange(len(used))
    pos, uv, T2 = np.asarray(D['pos'])[used].copy(), UV[used].copy(), re[keepT]
    if mirror:
        pos[:, 0] *= -1
        uv[:, 0] *= -1
        T2 = T2[:, [0, 2, 1]]
    return uv, pos, T2, np.asarray(D['kind'])[used]


def fit_fold(f, st, ext):
    from multiprocessing import Pool
    fn = folded_notch(f)
    jobs = []
    mode = st.pop('fit_mode', 'angle')
    st = dict(st)
    gain_ = st.pop('fit_gain', 0.65)
    r_ = [o for o in f['outlines'] if o[1][0] < 0][0]
    fall = {sg_: float(np.degrees(np.arctan2(-(fn[sg_][1] - fn[sg_][0])[1], abs((fn[sg_][1] - fn[sg_][0])[0])))) for sg_ in (-1, 1)}
    if mode == 'angle':
        # (tried first, 2026-11-21: the whole end drafted on the folded edge, notch 45 as seen: both ends curled into rolls,
        # worse. Kept as fit_mode 'edge'. Here the draft stays the one that came out right on her right, and only the
        # notch's drafted angle is turned per side by how much less (or more) that side's lapel top fell in the fold.)
        for sg_ in (-1, 1):
            Gd = np.asarray(f['gorges'][sg_][1])
            Pd = np.array([abs(r_[2][3][0]) * sg_, r_[2][3][1]])
            fn[sg_] = (Gd, Pd)
    for sg in (-1, 1):
        m = np.array([-sg, 1.0])                                 # (the left's world mirrored, so it is drafted as a right)
        m3 = np.array([-sg, 1.0, 1.0])
        G, P = fn[sg]
        G2 = G + ext * (P - G)
        X, _, X3 = f['gorges'][sg]
        Xo, _, X3o = f['gorges'][-sg]
        Go, Po = fn[-sg]
        gor = {-1: (np.asarray(X) * m, G2 * m, np.asarray(X3) * m3), 1: (np.asarray(Xo) * m, (Go + ext * (Po - Go)) * m, np.asarray(X3o) * m3)}
        E = np.asarray(f['E'])
        n = len(E)
        Em = E if sg == -1 else (E * m3)[(n - np.arange(n)) % n]
        print('collarlab: side %+d: lapel top as folded falls %.0f deg; notch at %s, lapel point %s, gorge %.2f'
              % (sg, np.degrees(np.arctan2(-(P - G)[1], abs((P - G)[0]))), G2.round(2), P.round(2), np.linalg.norm(G2 - np.asarray(X))))
        flip = (lambda Fq: Fq) if sg == -1 else (lambda Fq: [list(q)[::-1] for q in Fq])     # (a mirrored mesh's faces wind the other way: its normals would point in)
        st_s = dict(st)
        if mode == 'angle':
            # (turned the other way first, 85 - 10 on her left: its opening as seen went from 38 to 27 deg, about 1.1 deg per
            # degree drafted. Her left's lapel top lies flatter, so its collar end must be turned up more to open the same.)
            st_s['notch'] = st.get('notch', 65) + gain_ * (fall[-1] - fall[sg])
            print('collarlab: side %+d: notch drafted at %.0f deg (lapel top fell %.0f deg in the fold, her right %.0f)' % (sg, st_s['notch'], fall[sg], fall[-1]))
        jobs.append((Em, f['M'], np.asarray(f['V']) * m3, flip(f['F']), np.asarray(f['skinP']) * m3, flip(f['skinF']), gor, P * m, st_s))
    with Pool(2) as pool:
        Ds = pool.map(_solve, jobs)
    parts = [_half(Ds[0], False), _half(Ds[1], True)]
    n0 = len(parts[0][0])
    D = dict(UV=np.vstack([parts[0][0], parts[1][0]]), pos=np.vstack([parts[0][1], parts[1][1]]),
             T=np.vstack([parts[0][2], parts[1][2] + n0]), kind=np.concatenate([parts[0][3], parts[1][3]]),
             report='her right: %s | her left: %s' % (Ds[0]['report'], Ds[1]['report']))
    return D


def _solve(job):
    E, M, V, F, BV, BF, gor, Pp, st = job
    return tailor.tailor_collar(E, M, V, F, BV, BF, gor, Pp, settings=dict(st), log=lambda *a: None)


def main():
    f = pickle.load(open(sys.argv[1], 'rb'))
    st = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    ext = float(st.pop('gorge_ext', 0.0))
    if st.pop('fit_fold', False):
        D = fit_fold(f, st, ext)
        pickle.dump(D, open(sys.argv[2], 'wb'))
        print('collarlab: ' + D['report'])
        return
    gorges = dict(f['gorges'])
    outs = {(1 if o[1][0] > 0 else -1): o for o in f['outlines']}
    Pp = np.asarray(outs[-1][2][3], float)
    if ext:
        for sg, (X, G, X3) in list(gorges.items()):
            # (the lapel's own point on that side: the left lapel's outline is clipped where the right lies over it, so
            # the right's is mirrored)
            P_ = Pp * np.array([-sg, 1.0]) if sg == 1 else Pp
            P_ = np.array([abs(Pp[0]) * sg, Pp[1]])
            G2 = np.asarray(G) + ext * (P_ - np.asarray(G))
            gorges[sg] = (X, G2, X3)
        print('collarlab: notch moved %.0f %% along the lapel\'s top edge: gorge %.2f -> %.2f'
              % (100 * ext, np.linalg.norm(np.asarray(f['gorges'][-1][1]) - f['gorges'][-1][0]), np.linalg.norm(gorges[-1][1] - gorges[-1][0])))
    D = tailor.tailor_collar(f['E'], f['M'], f['V'], f['F'], f['skinP'], f['skinF'], gorges, Pp, settings=st)
    pickle.dump(D, open(sys.argv[2], 'wb'))
    print('collarlab: ' + D['report'])


if __name__ == '__main__':
    main()
