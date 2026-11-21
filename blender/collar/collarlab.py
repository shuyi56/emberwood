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


def smooth_edges_pass(D, rounds=60, band=0.6, corner_deg=40.0, inner_rounds=30, front_y=None, roll=True, roll_rounds=80, roll_band=0.5, log=print):
    """The smoothing pass (John, 2026-11-21: "create a smoothing pass that gets rid of all creases at the edges"): the
    collar's free edges (outer edge and ends; not the neck seam, which is sewn and hidden) and the cloth within `band` of
    them, faired after the solve.

      the edge    each free edge is a curve: it is faired along itself with Taubin's two-step smoothing (Taubin 1995:
                  a shrinking step then a slightly larger swelling step, so wrinkles go and the curve keeps its
                  length and place), its corners held (the collar's points and the notch: where the cut pattern
                  turns more than corner_deg), so the cut shape stays.
      the band    the cloth within `band` of a free edge is smoothed the same way over the surface, fading to nothing
                  at the band's inner side, so the edge's wrinkles do not stay as dents behind it.

      the seam    at the fronts the seam's edge shows (the gorge, on the lapel): with front_y set, seam points in front of
                  that y are faired as a free edge too. The seam round the neck stays where it is sewn.
      the roll    the fold's own line is a curve too, and its waves show as a wavy top edge: each half's roll is faired
                  along itself the same way, and the cloth within roll_band of it is carried with it (each point by its
                  nearest roll point's move, fading out), so the fold keeps its section and only loses its waves.

    Returns the new positions; logs how far points moved and the edge's roughness before and after (the mean turn
    between successive edge segments, degrees)."""
    from scipy.spatial import cKDTree
    pos, T, UV, kind = np.asarray(D['pos'], float).copy(), np.asarray(D['T']), np.asarray(D['UV']), np.asarray(D['kind'])
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    E, c = np.unique(E, axis=0, return_counts=True)
    Bd = E[c == 1]
    free_pt = np.isin(kind, ('o', 'e'))
    if front_y is not None:
        free_pt |= (kind == 's') & (pos[:, 1] < front_y)
    Bd = Bd[free_pt[Bd].all(1)]
    nb = {}
    for a_, b_ in Bd:
        nb.setdefault(a_, []).append(b_)
        nb.setdefault(b_, []).append(a_)
    idx = np.array([i for i, v in nb.items() if len(v) == 2])
    J = np.array([nb[i] for i in idx])
    # corners: where the pattern's own outline turns
    v1, v2 = UV[idx] - UV[J[:, 0]], UV[J[:, 1]] - UV[idx]
    turn = np.degrees(np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0], (v1 * v2).sum(1))))
    mov = turn < corner_deg
    idx, J = idx[mov], J[mov]

    def rough(P):
        a1, a2 = P[idx] - P[J[:, 0]], P[J[:, 1]] - P[idx]
        cs = (a1 * a2).sum(1) / np.maximum(np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1), 1e-12)
        return float(np.degrees(np.arccos(np.clip(cs, -1, 1))).mean())
    r0 = rough(pos)
    lam, mu = 0.5, -0.53
    for _ in range(rounds):
        for f_ in (lam, mu):
            pos[idx] += f_ * (0.5 * (pos[J[:, 0]] + pos[J[:, 1]]) - pos[idx])
    # the band behind the edge
    edge_pts = np.unique(Bd)
    d = cKDTree(UV[edge_pts]).query(UV)[0]
    wgt = np.clip(1 - d / band, 0, 1)
    wgt = wgt * wgt * (3 - 2 * wgt)
    wgt[np.isin(np.arange(len(UV)), edge_pts)] = 0          # (the edge itself is set)
    wgt[(kind == 's') & ~free_pt] = 0
    nbr = [[] for _ in range(len(UV))]
    for a_, b_ in E:
        nbr[a_].append(b_)
        nbr[b_].append(a_)
    act = np.nonzero(wgt > 0)[0]
    cnt = np.array([len(nbr[i]) for i in act])
    ii = np.repeat(act, cnt)
    jj = np.concatenate([nbr[i] for i in act])
    slot = np.repeat(np.arange(len(act)), cnt)
    for _ in range(inner_rounds):
        for f_ in (lam, mu):
            acc = np.zeros((len(act), 3))
            np.add.at(acc, slot, pos[jj])
            pos[act] += (f_ * wgt[act])[:, None] * (acc / cnt[:, None] - pos[act])
    if roll:
        before = pos.copy()
        rmov = np.zeros(len(pos), bool)
        for sx in (1, -1):
            rr = np.nonzero((kind == 'r') & (pos[:, 0] * sx > 0))[0]
            if len(rr) < 5:
                continue
            # (in order along the roll: by the angle round the neck, then down the front)
            rr = rr[np.argsort(np.abs(UV[rr, 0]))]
            ends = [0, len(rr) - 1]
            Pr = pos[rr].copy()
            for _ in range(roll_rounds):
                for f_ in (lam, mu):
                    Pr[1:-1] += f_ * (0.5 * (Pr[:-2] + Pr[2:]) - Pr[1:-1])
            pos[rr] = Pr
            rmov[rr] = True
        ri = np.nonzero(rmov)[0]
        if len(ri):
            dr, jr = cKDTree(UV[ri]).query(UV)
            wr = np.clip(1 - dr / roll_band, 0, 1)
            wr = wr * wr * (3 - 2 * wr)
            wr[rmov] = 0
            wr[(kind == 's') & ~free_pt] = 0
            disp = (pos - before)[ri][jr]
            pos += wr[:, None] * disp
        log('collarlab: roll faired: %d points, moved up to %.3f' % (len(ri), np.linalg.norm(pos - before, axis=1).max()))
    mv = np.linalg.norm(pos - np.asarray(D['pos']), axis=1)
    log('collarlab: edges smoothed: %d edge points faired (%d corners held), %d points in the band; moved up to %.3f (median of moved %.3f); edge roughness %.1f -> %.1f deg'
        % (len(idx), int((~mov).sum()), len(act), mv.max(), np.median(mv[mv > 1e-6]) if (mv > 1e-6).any() else 0.0, r0, rough(pos)))
    return pos


def main():
    if sys.argv[1] == 'smooth':                       # collarlab.py smooth IN.pkl OUT.pkl ['{"rounds": 60, "band": 0.6}']
        D = pickle.load(open(sys.argv[2], 'rb'))
        D['pos'] = smooth_edges_pass(D, **(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}))
        pickle.dump(D, open(sys.argv[3], 'wb'))
        return
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
