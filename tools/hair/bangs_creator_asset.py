"""The ponytail field's hair with these bangs, as a Face Creator groom asset (the hair fidelity mode's "strands"):
creator_asset.py's format, with the strands taken from the ponytail renderer's fibres (bangs_render.py's lines: the
bangs and their joined side hair) instead of a guide groom. There are no cards yet, so the asset's LOD list is empty
and the page offers only the strands for it.

    python3 tools/hair/bangs_creator_asset.py HEAD.json.gz BANGS_DIR OUT.json --field-cache CACHE [--strands 14000]
    python3 tools/face-creator/apply-hair-fidelity.py IN.html OUT.html ponytail=OUT.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ponytail_flow as PF
import ponytail_render as PR
import bangs_flow as BF
import creator_asset as CA
import cards as CD


TIE_FROM_MM = 20.0          # the tie is looked for past the hand-over from the head's hair (r90 peaks at 20 mm)
TIE_SEARCH_MM = 80.0        # and this far along the tail from the bundle's start


TAIL_LOCKS = 36             # DRAFT: locks the tail's hair gathers into
TAIL_STATION_MM = 150.0     # where along the tail a hair's lock is read off
TAIL_CLUMP = 0.15           # DRAFT: how far a hair is drawn to its lock's middle (XGen Clumping modifier's Clump amount)
TAIL_CLUMP_OVER_MM = (20.0, 110.0)      # DRAFT: the locks form over this run past the tie
TAIL_ROD_FROM_MM = 30.0     # the locks are free rods from here along the tail (just past the tie, 27 mm)
TAIL_LOCK_PACKING = 0.1     # DRAFT: a loose lock's packing, for its radius (lock contact keeps locks this thick apart)
TAIL_RIBBON = 0.5           # DRAFT: a lock is a ribbon, this thick against its width
TAIL_ROLL_MM = 300.0        # DRAFT: the ribbon rolls once round in this length
TAIL_TIP_CLUMP = 0.15       # DRAFT: toward its end a lock draws in to a point (TAIL_CLUMP at its start)
TAIL_SUBLOCKS = 3           # DRAFT: a lock parts into this many sub-locks as it falls (a river's distributaries)
TAIL_SUB_SPREAD = 1.4       # DRAFT: a sub-lock's middle stands this much farther from its lock's line at the tip
TAIL_HAIR_SPREAD = 0.7      # DRAFT: and a hair this much farther from its sub-lock's middle. Both grow as ds/dt = k(t) s, k
                            # rising smoothly from 0 at the tie: a lock is a clump of hairs, not a rod, so it opens as it
                            # falls, every hair keeping its place in the order across the lock (no crossings)
TAIL_SPREAD_FROM = 0.12     # the share of a lock's length where the opening starts
TAIL_JOIN_MM = 45.0         # a hair passes from the tail's field line to its lock's rod over this length, eased with a
                            # quintic (zero slope and curvature change at both ends: the line stays G2 across the join)
TAIL_TIE_SQUEEZE = 0.45     # DRAFT: under the band a lock is pressed to this share of its free radius (the tie's packing)
TAIL_LOCK_ROUND = 0.8       # DRAFT: how much a tail hair's "out" is its lock's own (each lock lit as a round bundle)


def tail_locks(P, sid, n_strands, FF, hairs, shell=None, seed=9):
    """A tail hangs in locks, not as one rope. Each strand joins the lock (of TAIL_LOCKS, uneven: nearest of random
    centres in the tail's section) it lies in at TAIL_STATION_MM. Each lock is then solved in 3D as an elastic rod
    hanging from the tie (tools/solvers/hair/tail_lock_rods.py: bending, natural curl, gravity, contact with the other
    locks and the head), and its hairs ride it: a hair keeps its place across its lock, in the rod's own frame, as a
    flat ribbon (RIBBON) that rolls (ROLL_MM a turn) and draws in toward the tip. Returns the moved points, each
    strand's lock (-1: none), each point's unit "out" on the tail (0 off it) and the solver's report."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'solvers', 'hair'))
    import tail_lock_rods as TR
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    mm = BF.MM
    cl, sc = np.asarray(FF['cl']), np.asarray(FF['s'], float)
    tg = np.gradient(cl, axis=0); tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    e1 = np.asarray(FF['e1'], float); e1 = e1 - (e1 * tg).sum(1, keepdims=True) * tg
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12); e2 = np.cross(tg, e1)
    jn = cKDTree(cl).query(P)[1]
    v = P - cl[jn]
    a, b = (v * e1[jn]).sum(1), (v * e2[jn]).sum(1)
    scm = np.abs(sc - sc[0]) * (1.0 if sc.max() > 5 else 1 / mm)
    sj = scm[jn]
    Rj = np.asarray(FF['R'], float)[jn] * mm
    on = (np.hypot(a, b) < Rj + 10 * mm) & (sj > 1.0)
    d = np.where(on, np.abs(sj - TAIL_STATION_MM), 1e9)
    order = np.lexsort((d, sid)); first = order[np.r_[True, sid[order][1:] != sid[order][:-1]]]
    ok = d[first] < 15.0
    sa, sb = a[first], b[first]
    rad = np.percentile(np.hypot(sa[ok], sb[ok]), 97)
    th, rr = rng.random(TAIL_LOCKS) * 2 * np.pi, rad * np.sqrt(rng.random(TAIL_LOCKS))
    lock = np.full(n_strands, -1)
    lock[sid[first][ok]] = cKDTree(np.c_[rr * np.cos(th), rr * np.sin(th)]).query(np.c_[sa[ok], sb[ok]])[1]
    used, lock_c = np.unique(lock[lock >= 0], return_inverse=True)
    lock[lock >= 0] = lock_c                                           # (compact: 0..n-1)
    nl = len(used)
    lp = lock[sid]
    m = on & (lp >= 0) & (sj >= TAIL_ROD_FROM_MM)
    j0 = int(np.argmin(np.abs(scm - TAIL_ROD_FROM_MM)))
    # how far along its lock each point lies: by the hair's own length past where it enters the tail (hair does not
    # stretch; read off the centre line's stations instead, a hair's nodes landed up to half a station off along the
    # rod, which showed as a jagged line)
    seg_ = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)] / mm
    seg_[np.r_[0, np.where(np.diff(sid) != 0)[0] + 1]] = 0.0
    arc = np.cumsum(seg_)
    idx_m = np.where(m)[0]
    firstm = np.full(n_strands, -1); firstm[sid[idx_m][::-1]] = idx_m[::-1]
    f = firstm[sid[idx_m]]
    cont = scm[jn] + ((P - cl[jn]) * tg[jn]).sum(1) / mm * np.sign(scm[-1] - scm[0])     # (the station, read continuously)
    srel = np.maximum(arc[idx_m] - arc[f] + cont[f] - TAIL_ROD_FROM_MM, 0.0)
    # each lock's middle along the old tail (per station), for a hair's place across its lock
    key = lp[m] * len(cl) + jn[m]
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    ca, cb = np.bincount(inv, a[m]) / cnt, np.bincount(inv, b[m]) / cnt
    def along(keys, val, groups):
        """A per-(group, station) mean, eased along the stations (a mean over few hairs jumps station to station)."""
        G = np.full((groups, len(cl)), np.nan); G[keys // len(cl), keys % len(cl)] = val
        idx = np.arange(len(cl))
        for r in range(groups):
            ok = ~np.isnan(G[r])
            if ok.sum() < 2:
                G[r] = np.nan_to_num(G[r]); continue
            f = np.interp(idx, idx[ok], G[r, ok])
            for _ in range(3):
                f = np.convolve(np.pad(f, 6, mode='edge'), np.ones(13) / 13, mode='valid')
            G[r] = f
        return G[keys // len(cl), keys % len(cl)]
    ca, cb = along(uk, ca, nl), along(uk, cb, nl)
    oa, ob = a[m] - ca[inv], b[m] - cb[inv]
    # the rods' roots: each lock's middle at the tie, heading as the lock heads there
    roots = np.zeros((nl, 3)); head_to = np.zeros((nl, 3)); length = np.zeros(nl); nh = np.zeros(nl)
    hs = hairs[sid]
    smax = np.zeros(n_strands); np.maximum.at(smax, sid[idx_m], srel)
    for k in range(nl):
        mine = np.where(uk // len(cl) == k)[0]                         # (this lock's stations, in order along the tail)
        q0, q1 = mine[:6], mine[10:16]
        c0 = cl[j0] + ca[q0].mean() * e1[j0] + cb[q0].mean() * e2[j0]
        c1 = cl[j0 + 10] + ca[q1].mean() * e1[j0 + 10] + cb[q1].mean() * e2[j0 + 10]
        hd = 0.7 * tg[j0] * np.sign(tg[j0] @ (c1 - c0)) + 0.3 * (c1 - c0) / np.linalg.norm(c1 - c0)     # (mostly the tail's own heading)
        roots[k], head_to[k] = c0, hd / np.linalg.norm(hd)
        mem = np.where(lock == k)[0]
        length[k] = np.percentile(smax[mem], 98)       # (the rod runs to the lock's longest hairs: no hair runs on past it)
        nh[k] = hairs[mem].sum()
    d1 = e1[j0][None] - (head_to @ e1[j0])[:, None] * head_to; d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    frames = np.stack([d1, np.cross(head_to, d1), head_to], 2)
    lock_r = np.sqrt(nh * np.pi * 0.035 ** 2 / TAIL_LOCK_PACKING / np.pi)

    def head(pts, rd):
        if shell is None:
            return np.zeros(len(pts)), np.zeros((len(pts), 3))
        U = pts * mm - BF.SF.CENTRE
        r = np.linalg.norm(U, axis=1); U = U / np.maximum(r[:, None], 1e-12)
        return shell.radius(U) / mm + rd - r / mm, U
    out = TR.solve(roots / mm, frames, length, nh, lock_r, head=head, seed=seed, squeeze=TAIL_TIE_SQUEEZE, log=lambda t: print('  rods:', t, file=sys.stderr))
    X, Rr, rs = out['x'] * mm, out['R'], out['s']
    # the hairs ride their lock's rod
    lk = lp[m]
    seg = rs[lk, 1]                                                    # the lock's segment length, mm
    M = Rr.shape[1]
    i = np.clip((srel / seg).astype(int), 0, M - 1)
    fr = srel / seg - i                                                # (past the rod's end: on along its last segment)
    base = X[lk, i] + (fr * seg * mm)[:, None] * Rr[lk, i][:, :, 2]
    t01 = np.clip(srel / np.maximum(length[lk], 1.0), 0, 1)
    w = BF.sm(0.0, 80.0, srel) * (TAIL_CLUMP + (TAIL_TIP_CLUMP - TAIL_CLUMP) * BF.sm(0.5, 1.0, t01))
    psi = rng.random(nl)[lk] * 2 * np.pi + 2 * np.pi * srel / TAIL_ROLL_MM * BF.sm(0.0, 60.0, srel)
    flat = 1 - (1 - TAIL_RIBBON) * BF.sm(0.0, 80.0, srel)
    # the lock opens as it falls: its hairs part into sub-locks (by where a hair lies round its lock's middle), the
    # sub-locks' middles move apart, and the hairs of each spread about their own middle
    ns = np.bincount(sid[idx_m], minlength=n_strands).astype(float)
    ma = np.bincount(sid[idx_m], oa, minlength=n_strands) / np.maximum(ns, 1)
    mb = np.bincount(sid[idx_m], ob, minlength=n_strands) / np.maximum(ns, 1)
    sub_s = (((np.arctan2(mb, ma) + rng.random(nl)[np.maximum(lock, 0)] * 2 * np.pi) % (2 * np.pi)) / (2 * np.pi / TAIL_SUBLOCKS)).astype(int) % TAIL_SUBLOCKS
    sub = sub_s[sid[idx_m]]
    key2 = (lk * TAIL_SUBLOCKS + sub) * len(cl) + jn[m]
    uk2, inv2 = np.unique(key2, return_inverse=True)
    c2 = np.bincount(inv2)
    qa, qb = np.bincount(inv2, oa) / c2, np.bincount(inv2, ob) / c2
    qa, qb = along(uk2, qa, nl * TAIL_SUBLOCKS), along(uk2, qb, nl * TAIL_SUBLOCKS)
    g = BF.sm(TAIL_SPREAD_FROM, 1.0, t01)
    oa2 = qa[inv2] * (1 + TAIL_SUB_SPREAD * g) + (oa - qa[inv2]) * (1 + TAIL_HAIR_SPREAD * g)
    ob2 = qb[inv2] * (1 + TAIL_SUB_SPREAD * g) + (ob - qb[inv2]) * (1 + TAIL_HAIR_SPREAD * g)
    # (the lock's section pressed flat along its own axis at angle psi, the axis rolling along the lock)
    cu, su = np.cos(psi), np.sin(psi)
    pu, pv = oa2 * cu + ob2 * su, (-oa2 * su + ob2 * cu) * flat
    na, nb = (pu * cu - pv * su) * (1 - w), (pu * su + pv * cu) * (1 - w)
    i2 = np.clip(i + np.where(fr > 0.5, 1, -1), 0, M - 1)            # (frames eased between segments: no step in a hair's place)
    wf = np.abs(fr - 0.5)[:, None]
    D1 = Rr[lk, i][:, :, 0] * (1 - wf) + Rr[lk, i2][:, :, 0] * wf
    D2 = Rr[lk, i][:, :, 1] * (1 - wf) + Rr[lk, i2][:, :, 1] * wf
    Q = P.copy()
    rod = base + na[:, None] * D1 + nb[:, None] * D2
    kj = np.clip(srel / TAIL_JOIN_MM, 0, 1); kj = kj ** 3 * (10 - 15 * kj + 6 * kj * kj)
    Q[m] = P[m] + kj[:, None] * (rod - P[m])
    half = np.degrees(np.arctan(np.percentile(np.hypot(oa2, ob2)[t01 > 0.9], 95) / mm / np.maximum(np.median(length), 1.0)))
    print('lock opening: half angle %.1f deg at the tip (limit 7)' % half, file=sys.stderr)
    # which way is out: from the lock's own middle, and from the bundle's (the rods' mean at that length)
    bundle = X.mean(0)                                                 # (M+1, 3): rods share the segment count
    o_lock = na[:, None] * D1 + nb[:, None] * D2
    o_lock /= np.maximum(np.linalg.norm(o_lock, axis=1, keepdims=True), 1e-9)
    o_b = Q[m] - bundle[np.clip(i, 0, M)]
    o_b /= np.maximum(np.linalg.norm(o_b, axis=1, keepdims=True), 1e-9)
    o = TAIL_LOCK_ROUND * o_lock + (1 - TAIL_LOCK_ROUND) * o_b
    tail_out = np.zeros_like(P); tail_out[m] = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-9)
    rep = dict(out['report'], strands=int((lock >= 0).sum()), lock_radius_mm_median=round(float(np.median(lock_r)), 1),
               length_mm_median=round(float(np.median(length)), 0))
    print('tail locks', rep, file=sys.stderr)
    region = np.zeros(len(P), int); region[idx_m] = np.where(srel < TAIL_JOIN_MM, 1, 2)      # 0 head, 1 join, 2 tail
    return Q, lock, tail_out, rep, region


def kink_gauge(P, counts, region=None):
    """How sharply hair lines turn: the turn at each node between its two segments, degrees. Per region (an int per
    point, or None for all): the 99.9th percentile, the largest, and the share of nodes turning over 20 degrees.
    A smooth line at this sampling (2 to 3 mm a segment) turns a few degrees a node."""
    e = np.diff(P, axis=0); ln = np.linalg.norm(e, axis=1)
    c = (e[1:] * e[:-1]).sum(1) / np.maximum(ln[1:] * ln[:-1], 1e-18)
    turn = np.degrees(np.arccos(np.clip(c, -1, 1)))                   # at points 1..n-2
    ends = np.cumsum(counts)
    ok = np.ones(len(P), bool); ok[ends - 1] = False; ok[np.r_[0, ends[:-1]]] = False
    ok[1:-1] &= (ln[1:] > 0.2 * BF.MM) & (ln[:-1] > 0.2 * BF.MM)
    t = np.zeros(len(P)); t[1:-1] = turn
    out = {}
    for r in ([None] if region is None else np.unique(region)):
        q = t[ok if r is None else ok & (region == r)]
        if len(q):
            out['all' if r is None else int(r)] = (round(float(np.percentile(q, 99.9)), 1), round(float(q.max()), 1), round(float((q > 20).mean()), 5))
    return out


FAIR_DEG = 8.0              # a node turning more than this is faired (a smooth line here turns a few degrees a node)
FAIR_ITERS = 60


def fair(P, counts):
    """Fairing: where a hair line turns sharply at a node (where two constructions meet: the head's field and the tail,
    the tail and a lock's rod, a lift's end, a clump's start), the node and its two neighbours are drawn to the mean
    of their neighbours, again and again until no node turns more than FAIR_DEG. Roots and tips stay. Only kinked
    nodes move, so a smooth line is left exactly as it was (plain smoothing would shrink every wave)."""
    Q = P.copy()
    ends = np.cumsum(counts)
    fixed = np.zeros(len(P), bool); fixed[ends - 1] = True; fixed[np.r_[0, ends[:-1]]] = True
    moved = np.zeros(len(P), bool)
    for _ in range(FAIR_ITERS):
        e = np.diff(Q, axis=0); ln = np.linalg.norm(e, axis=1)
        c = (e[1:] * e[:-1]).sum(1) / np.maximum(ln[1:] * ln[:-1], 1e-18)
        bad = np.zeros(len(P), bool); bad[1:-1] = np.degrees(np.arccos(np.clip(c, -1, 1))) > FAIR_DEG
        bad &= ~fixed
        if not bad.any():
            break
        sel = bad.copy(); sel[1:] |= bad[:-1]; sel[:-1] |= bad[1:]; sel &= ~fixed
        k = np.where(sel)[0]
        Q[k] = 0.25 * Q[k - 1] + 0.5 * Q[k] + 0.25 * Q[k + 1]
        moved |= sel
    d = np.linalg.norm(Q - P, axis=1) / BF.MM
    return Q, dict(nodes_moved_share=round(float(moved.mean()), 4), move_mm_p99=round(float(np.percentile(d[moved], 99)) if moved.any() else 0.0, 2),
                   move_mm_max=round(float(d.max()), 2))


def main():
    a = sys.argv[1:]
    head, bdir, out = a[:3]
    nmax = int(a[a.index('--strands') + 1]) if '--strands' in a else 14000
    BF.reuse_field_cache(a[a.index('--field-cache') + 1])
    PF.HEAD = head
    z = np.load(os.path.join(bdir, 'bangs-lines.npz'))
    mine = {int(r): l for r, l in zip(z['roots'], z['lines'])}
    PF.bang_line = lambda B, i, k=32: mine[int(i)]
    side = {}
    if 'tie_roots' in z.files:
        side.update({int(r): l for r, l in zip(z['tie_roots'], z['tie_lines'])})
        side.update({int(r): l for r, l in zip(z['lock_roots'], z['lock_lines'])})
    build = PF.build
    PF.build = lambda h: (lambda B: (BF.apply_side(B, side), B)[1])(build(h))
    paths, hairs, basin, B = PR.fibres(head, per_root=1)
    tiltv = comb = layv = None
    if '--volume' in a:                                               # the side hair in layers off the bed (hair_volume.py)
        import hair_volume as HV
        FF0 = PF.free_fall(B)
        paths, tiltv, comb, layv, vol = HV.layer(paths, basin == 0, BF.Shell(B['surf']), np.asarray(FF0['cl'])[0], float(B['r_tie']),
                                     np.asarray(FF0['tang'])[0])
        print('volume', vol, file=sys.stderr)
    fun_u = fun_tie = None
    if '--funnel' in a:                                               # the side hair's last stretch gathered to the tie (tie_funnel.py)
        import tie_funnel as TF
        FFf = PF.free_fall(B)
        clf, scf = np.asarray(FFf['cl']), np.asarray(FFf['s'], float)
        smf = np.abs(scf - scf[0]) * (1.0 if scf.max() > 5 else 1 / BF.MM)
        jf = int(np.argmin(np.abs(smf - 27.0)))                       # (the tie: 27 mm along the tail, as the tie search below finds)
        paths, fun_u, fu = TF.funnel(paths, basin == 0, clf[jf], clf[jf + 2] - clf[jf - 2], float(B['r_tie']), BF.Shell(B['surf']), float(hairs[basin == 0].sum()))
        print('funnel', fu, file=sys.stderr)
        fun_tie = (clf[jf], (clf[jf + 2] - clf[jf - 2]) / np.linalg.norm(clf[jf + 2] - clf[jf - 2]))
    tier = None
    if '--mid-layer' in a:                                            # a mid layer of thick clumps under the loose top (hair_layers.py)
        import hair_layers as HLy
        paths, tier, ml = HLy.mid_layer(paths)
        print('mid layer', ml, file=sys.stderr)
    noise = None
    if '--trim' in a:                                                 # the fringe and the loose locks cut to a line (hair_trim.py)
        import hair_trim as HT
        paths, tr = HT.trim(paths, basin == 1, basin == 2)
        print('trim', tr, file=sys.stderr)
    if '--noise' in a:                                                # the noising round (noise_round.py)
        import noise_round as NR
        paths, noise = NR.apply(paths, surf=B['surf'], scale=float(a[a.index('--noise') + 1]))
        print('noise', noise, file=sys.stderr)
    print('fibres', paths.shape, file=sys.stderr)
    rng = np.random.default_rng(5)
    keep = np.arange(len(paths))
    if len(keep) > nmax:                               # (a share of the fibres, each standing for more hairs)
        keep = np.sort(rng.choice(keep, nmax, replace=False))
    ps = [p[~np.isnan(p[:, 0])] for p in paths[keep]]
    ok = [k for k, p in enumerate(ps) if len(p) >= 2]
    ps = [ps[k] for k in ok]
    keep = keep[ok]
    P = np.concatenate(ps)
    share = float(hairs.sum()) / len(ps)
    counts = np.array([len(p) for p in ps])
    q = 2.5 / 32767
    tlock = lock_out = None
    if '--tail-locks' in a:
        P_before = P
        sid = np.repeat(np.arange(len(ps)), counts)
        P, tlock, lock_out, rods, region = tail_locks(P, sid, len(ps), PF.free_fall(B), hairs[keep] * len(paths) / len(keep), BF.Shell(B['surf']))
        ps = np.split(P, np.cumsum(counts)[:-1])
        P_solved = P
        P, fr_ = fair(P, counts)
        ps = np.split(P, np.cumsum(counts)[:-1])
        print('fairing', fr_, file=sys.stderr)
        print('kinks {region: (99.9th pct deg, max deg, share over 20 deg)}; 0 head, 1 join, 2 tail\n  before', kink_gauge(P_before, counts, region),
              '\n  solved', kink_gauge(P_solved, counts, region), '\n  faired', kink_gauge(P, counts, region), file=sys.stderr)
    from scipy.spatial import cKDTree
    tree = cKDTree(P[::2])
    centre = np.array([0.0, 0.0, 0.0])
    cap_dir = os.path.join(os.path.dirname(os.path.abspath(out)), 'cap')
    os.makedirs(cap_dir, exist_ok=True)
    CD.scalp_cap(head, cap_dir)
    print('cap done', file=sys.stderr)
    V, _, Fc = CA.read_obj(os.path.join(cap_dir, 'scalp-cap.obj'))
    centre = V.mean(0) - np.array([0, 0, 0.15])
    outS = P - np.where(P[:, 2:3] < centre[2], np.array([centre[0], centre[1], 0]) + np.array([0, 0, 1]) * P[:, 2:3], centre)
    outS /= np.maximum(np.linalg.norm(outS, axis=1, keepdims=True), 1e-6)
    cs = sum(tree.query_ball_point(P + outS * d, CA.AO_R, return_length=True) for d in CA.AO_STEPS)
    sao = np.clip(np.exp(-cs / (np.median(cs) / -np.log(CA.SAO_MID))), CA.SAO_MIN, 1.0)
    hsh = lambda v: ((v.astype(np.int64) * 2654435761 + 12345) >> 7) % 251
    lock_id = basin[keep] * 7919 + keep // 40
    if comb is not None:                               # (a comb lock's hair shares its tone)
        lock_id = np.where(comb[keep] >= 0, 104729 + comb[keep], lock_id)
    clump_id = keep if tlock is None else np.where(tlock >= 0, 15485863 + tlock, keep)      # (a tail lock's hair shares its tone)
    ids = np.stack([hsh(lock_id), hsh(clump_id)], 1).astype('u1')
    # light through the hair from its flow and density (hair_light.py): per creator light, and ambient
    import hair_light as HL
    lt = HL.light(ps, hairs[keep] * len(paths) / len(keep))      # (each drawn strand: its fibre's hairs, scaled up)
    occ = np.stack([lt['key'], lt['rim'], lt['fill'], lt['ambient']], 1)
    print('light: key T median %.2f, 10th pct %.2f; ambient median %.2f, 10th pct %.2f' % (
        np.median(lt['key']), np.percentile(lt['key'], 10), np.median(lt['ambient']), np.percentile(lt['ambient'], 10)))
    # which way is out, per point: on the head, out from the skull (as the page had it); along the tail, out from the
    # tail's own centre line (the free-fall bundle's, FF['cl']). Lit by the head's "out", every hair of the tail faced
    # the same way and the tail shaded as one flat sheet; lit round its own axis it reads as a round bundle.
    FF = PF.free_fall(B)
    cl, sc = np.asarray(FF['cl']), np.asarray(FF['s'], float)
    jn = cKDTree(cl).query(P)[1]
    tg = np.gradient(cl, axis=0); tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    v = P - cl[jn]
    v -= (v * tg[jn]).sum(1, keepdims=True) * tg[jn]
    dcl = np.linalg.norm(v, axis=1)
    out_tail = v / np.maximum(dcl[:, None], 1e-9)
    sj = np.abs(sc[jn] - sc[0]) * (1.0 if sc.max() > 5 else 1 / BF.MM)       # mm along the tail from the tie
    Rj = np.asarray(FF['R'], float)[jn]
    wt = BF.sm(0.0, 25.0, sj) * (dcl / BF.MM < Rj + 10.0)
    oh = P - np.c_[np.full(len(P), centre[0]), np.minimum(P[:, 1], centre[1]), np.full(len(P), centre[2])]
    oh /= np.maximum(np.linalg.norm(oh, axis=1, keepdims=True), 1e-9)
    if tiltv is not None:                              # the comb furrows' own slopes (hair_volume.py), so they are lit
        oh += np.concatenate([tiltv[i][~np.isnan(paths[i][:, 0])] for i in keep])
        oh /= np.maximum(np.linalg.norm(oh, axis=1, keepdims=True), 1e-9)
    if fun_u is not None:                              # (in the funnel "out" turns from the head's to the tail's with the hair)
        wt = np.maximum(wt, np.concatenate([fun_u[i][~np.isnan(paths[i][:, 0])] for i in keep]))
    outv = oh * (1 - wt)[:, None] + out_tail * wt[:, None]
    if lock_out is not None:                           # (the solved tail: out from each lock's own middle)
        tw = np.linalg.norm(lock_out, axis=1, keepdims=True)
        outv = np.where(tw > 0.5, lock_out, outv)
    outv /= np.maximum(np.linalg.norm(outv, axis=1, keepdims=True), 1e-9)
    print('tail points %.0f%%' % (100 * (wt > 0.5).mean()), 's range', sc.min(), sc.max(), 'R', Rj.min(), Rj.max())
    # the tie: where the field gathers the hair (the bundle's first section), for the page to wind its thong round
    # (not the bundle's first section, cl[0], which lies inside the hair on the head and where the centre line still
    # rises: the section within TIE_SEARCH_MM along the tail where the drawn hair is gathered tightest)
    best = None
    sm_ = np.abs(sc - sc[0]) * (1.0 if sc.max() > 5 else 1 / BF.MM)
    for j in np.where((sm_ >= TIE_FROM_MM) & (sm_ < TIE_SEARCH_MM))[0]:
        m = (jn == j) & (dcl / BF.MM < Rj + 10.0)
        if m.sum() < 30:
            continue
        r90 = float(np.percentile(dcl[m], 90))
        if best is None or r90 < best[1]:
            best = (j, r90)
    jt, rt = best if best else (0, float(B['r_tie']))
    print('tie at %.0f mm along the tail, radius %.1f mm' % (sm_[jt], rt / BF.MM))
    tie = dict(c=[float(x) for x in cl[jt]], t=[float(x) for x in tg[jt]], r=rt, s_mm=float(abs(sc[jt] - sc[0])))
    if fun_tie is not None:
        # with the funnel, the band goes where the funnel gathers the hair, and is sized and centred from the hair
        # drawn there (not from the tail's centre line and its free radius: a band wider than the neck it holds
        # reads as a loose ring)
        cq, tq = fun_tie
        cen, rq = fu['centre'], fu['seen_radius_mm'] * BF.MM             # (where the funnel's own hairs cross the band's plane)
        tie = dict(c=[float(x) for x in cen], t=[float(x) for x in tq], r=rq, s_mm=27.0, fit='hair')
        print('band on the funnel neck: radius %.1f mm, centre %.1f mm off the tail line' % (rq / BF.MM, np.linalg.norm(cen - cq) / BF.MM))
    rep = dict(lods=[], guides=0, tie=tie, cap=dict(pos=CA.b64(V, '<f4'), idx=CA.b64(Fc, '<u4')),
               strands=dict(n=int(len(ps)), counts=CA.b64(counts, '<u2'), pos=CA.b64(np.round(P / q), '<i2'), q=q,
                            hairs_per_strand=round(share, 2), ao=CA.b64(np.round(sao * 255), 'u1'), ids=CA.b64(ids, 'u1'),
                            light=CA.b64(np.round(np.clip(occ, 0, 1) * 255), 'u1'), light_order='key,rim,fill,ambient',
                            out=CA.b64(np.round(outv * 127), 'i1'),
                            **({'layer': CA.b64(np.round(layv[keep] * 255), 'u1')} if layv is not None else {}),
                            **({'tier': CA.b64(tier[keep], 'u1')} if tier is not None else {})),
               source=dict(bangs=os.path.abspath(bdir), fibres=int(len(paths))))
    json.dump(rep, open(out, 'w'))
    print(out, os.path.getsize(out) // 1024, 'KB', len(ps), 'strands of', len(paths), 'fibres, %.1f hairs each' % share)


if __name__ == '__main__':
    main()
