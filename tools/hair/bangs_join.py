"""The join between the bangs and the side hair, made continuous (John, 2026-11-25: "Get the interface correct where
they meet at the joins add a continuous constraint").

Two families meet at the bangs section's side edge: the bangs (streamlines of t = normalize(w B + (1 - w) D)) and
the side hair (the ponytail's lines to the tie, and the temple lock). Left alone they meet at an angle (82 deg at the
curtain's edge) and at different heights off the skin: a ledge.

The constraint. Every root within JOIN_BAND_MM (b) of the edge has two candidate lines: its own family's, and the
other family's carried over from the nearest roots of that family on the edge (moved along the skin by the roots'
offset, at that line's own height off the skin), which is what a hair of that family standing there would do. Its
hair is walked over the bed with
    direction T(s) = slerp(T_D(s), T_B(s), Lambda),   height h(s) = (1 - Lambda) h_D + Lambda h_B,
    Lambda(d, s) = smoothstep(0, TURN_MM, s - onset(d)),   onset(d) = L - smoothstep(-b, b, d) (L + TURN_MM),
d the root's signed distance from the edge along the skin (< 0 inside the bangs), s the length along the hair, L the
bangs' length there. Lambda is smooth in d and in s, so two roots side by side across the edge get the same blend:
the join is G0 and G1 by construction. Deep in the bangs a hair never turns (pure bangs); deep in the side hair it is
side hair from its root; between, it leaves as a bang and turns into the side flow, the outer hair first, and a side
hair once turned is pulled back onto its own line (PULL_MM), so it still reaches the tie. The corners turn into the
side hair ([BTC]: the corners softened into the face-framing layers; [MBG]: the next front sections cut on the same
diagonal).

The back edge of the section (60 mm behind the hairline) is a part, where hair divides by definition: there the
gauge checks only that the scalp is covered (hairline gauge), not the tangent.

Gauge (join_gauge): one-sided fits each side of the edge, compared on it, at 5, 10, 20, 30 and 40 mm along the
hairs (see the function).
Limits from the join diagnostics (blender/shared/continuity.py, PR #86, matte): G1 pass 0.57 deg (1% luminance
contrast under a 45 deg key), fail 4 deg (Norman et al. 2006, Vision Res 46:1057, orientation thresholds 4 to 10
deg); G0 fail 0.17 mm (a quarter pixel at the queue's face crop). The opening's bound is DRAFT: 2 mm, about the
spacing of neighbouring roots across the edge (the root rows are 1 to 2 mm apart here), for the gap along the skin."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF
import ponytail_flow as PF
import bangs_flow as BF

MM = SF.MM
sm = SF.sm
JOIN_BAND_MM = 25.0             # DRAFT. The photo's lean runs from the bangs' 15 deg to the side's 25 deg over u 61 to
                                # 78 mm (strands 3 to 6), about 15 mm each side of the edge; 15 mm gave G1 5.5 deg at
                                # the edge (the side hair's own floor there 5.1), 25 mm gives 3.2 deg (floor 3.5) with
                                # 6 rank swaps of 3,816 against 1 (rounds 3 and 5, README)
CARRY_K = 6                     # edge roots averaged for a carried-over line
CARRY_SIGMA_MM = 3.0            # their weights' width
EDGE_ROW_MM = 3.0               # the roots on the edge, whose lines the other family's roots carry over
FIT_MM = 8.0                    # the one-sided fits of the join gauge take the roots within this of the edge
DEPTH_BIN_MM = 6.0              # and cut them into rows this deep behind the hairline
G1_LIM = (0.57, 4.0)            # deg, pass and fail [PR #86 continuity.py, matte]
G0_LIM = 0.17                   # mm, fail [PR #86 continuity.py]
OPEN_LIM = 2.0                  # mm, DRAFT
AT_MM = (5.0, 10.0, 20.0, 30.0, 40.0)


def edge_d(B, idx):
    """Signed distance (mm along the skin, round the head) of roots idx from the bangs section's side edge; < 0 inside
    the bangs. Only roots in front of the section's back edge have a side join; others get +inf."""
    P = B['P'][idx]
    r = np.hypot(P[:, 0] - SF.AXIS[0], P[:, 1] - SF.AXIS[1])
    d = (np.abs(B['th'][idx]) - PF.BANG_TH) * r / MM
    front = (B['inside_mm'][idx] < PF.BANG_DEPTH_MM) & (P[:, 1] < 0)
    return np.where(front, d, np.inf)


def arclen(p):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] / MM


def at_s(p, s):
    """Points of polyline p at lengths s (mm); nan past its end."""
    a = arclen(p)
    out = np.c_[[np.interp(s, a, p[:, k]) for k in range(3)]].T
    out[s > a[-1] + 1e-9] = np.nan
    return out


def side_line(B, i):
    """A side root's own line on the scalp: the ponytail's line (to the handover ring) or the temple lock's."""
    if B['basin'][i] == 0:
        return B['X'][i]
    k = B['basin'][i] - 2
    nm, c = list(PF.LOOSE.items())[k]
    return PF.loose_line(B, i, c['len'])


TURN_MM = 90.0                  # the length over which a hair in the band turns from the bangs' way to the side's:
                                # the two are up to 110 deg apart at the edge, and a smoothstep turn of 110 deg over
                                # 90 mm has its tightest radius at 90 / (1.5 x 1.92 rad) = 31 mm, inside the 19 mm
                                # curl bound (BF.BRUSH_R_MM) with room for the bangs' own bend (70 mm left 3 hairs
                                # over it at 16 mm)
PULL_MM = 60.0                  # a side hair, once turned, returns to its own line over this


def onset(d, L, b):
    """Where along the hair (mm) the turn from the bangs' way to the side's starts, for a root d from the edge: past
    the hair's end deep in the bangs (it never turns), before its root deep in the side hair (it is side hair from the
    root), and earlier the further out the root. The outer hair turning first keeps the band's order: the hairs leave
    side by side, the outer one turns back first and the inner one turns back below it, so their paths nest like Ls
    instead of crossing (a turn that varied only across the band, the same at every length, closed the fan: the inner
    hairs fell over the outer roots, 51 crossings against 17)."""
    sig = sm(-b, b, d)
    return L - sig * (L + TURN_MM)


def lambda_(d, s, L, b):
    """Lambda(d, s) = smoothstep over TURN_MM from onset(d): continuous, with its derivative, in d and in s."""
    return sm(0.0, TURN_MM, s - onset(d, L, b))


def walk(shell, jobs, b, ds=1.0):
    """Walk every job's hair over the bed at once. A job: its root, its D and B candidates (lists of (line, weight),
    mixed by direction and height), d, L (the bangs' length there), its own length, and (for a side hair) its own
    line, onto which it is pulled back once turned (PULL_MM)."""
    n = len(jobs)
    Ls = np.array([j['length'] for j in jobs])
    S = np.arange(0, Ls.max() + ds, ds)
    TD = np.zeros((n, len(S), 3)); TB = np.zeros_like(TD); HD = np.zeros((n, len(S))); HB = np.zeros_like(HD)
    LAM = np.zeros((n, len(S))); OWN = np.zeros_like(TD); PULL = np.zeros((n, len(S)))

    def tang_h(p, s):
        a = arclen(p)
        sc = np.minimum(s, a[-1] - ds)
        q0, q1 = at_s(p, sc), at_s(p, sc + ds)
        U0 = BF.unit(q0 - SF.CENTRE)
        h = ((q0 - shell.point(U0)) * shell.normals(U0)).sum(1)      # (signed: the root can sit under the bed)
        return BF.unit(q1 - q0), h

    def mix(cands):
        T = 0; h = 0
        for p, w in cands:
            if w <= 0:
                continue
            t_, h_ = tang_h(p, S)
            T = T + w * t_; h = h + w * h_
        return BF.unit(T), h

    for k, jb in enumerate(jobs):
        TD[k], HD[k] = mix(jb['D'])
        TB[k], HB[k] = mix(jb['B'])
        LAM[k] = lambda_(jb['d'], S, jb['L'], b)
        if jb['own'] is not None:
            s1 = onset(jb['d'], jb['L'], b) + TURN_MM  # (turned: all side hair from here)
            PULL[k] = sm(s1, s1 + PULL_MM, S)
            a = arclen(jb['own'])
            OWN[k] = at_s(jb['own'], np.minimum(S, a[-1]))
    X = np.array([j['P'] for j in jobs])
    base = shell.point(BF.unit(X - SF.CENTRE))
    out = np.zeros((n, len(S), 3))
    for t in range(len(S)):
        Nn = shell.normals(BF.unit(base - SF.CENTRE))
        a = TD[:, t] - (TD[:, t] * Nn).sum(1, keepdims=True) * Nn
        c = TB[:, t] - (TB[:, t] * Nn).sum(1, keepdims=True) * Nn
        a, c = BF.unit(a), BF.unit(c)
        th = np.arccos(np.clip((a * c).sum(1), -1, 1))[:, None]
        sth = np.maximum(np.sin(th), 1e-6)
        lam = LAM[:, t][:, None]
        T = np.where(th > 1e-3, (np.sin((1 - lam) * th) * a + np.sin(lam * th) * c) / sth, a)
        h = (1 - LAM[:, t]) * HD[:, t] + LAM[:, t] * HB[:, t]
        pt = base + Nn * h[:, None]
        pl = PULL[:, t][:, None]                       # (applied to the free walk, once: not compounded per step)
        out[:, t] = pt + pl * (OWN[:, t] - pt)
        base = shell.point(BF.unit(base + T * ds * MM - SF.CENTRE))
    return [out[k, :int(np.searchsorted(S, Ls[k], 'right'))] for k in range(n)]


def join(B, SFL, roots, lines, prm, b=JOIN_BAND_MM):
    """The joined lines. Returns (new bang lines list, {side root: new line}, info)."""
    d_b = edge_d(B, roots)
    side = np.where((B['basin'] != 1) & (B['basin'] >= 0))[0]
    d_s = edge_d(B, side)
    sb = side[(d_s >= 0) & (d_s < b)]
    d_sb = d_s[(d_s >= 0) & (d_s < b)]
    inb = np.where(d_b > -b)[0]                       # bang roots in the band
    # the candidate lines. Each root takes its candidate of the other family from the nearest root of that family
    # on the edge, carried over to its own root (moved along the skin by the roots' offset, at that line's own height
    # off the skin). Two roots side by side across the edge then hold the same pair of lines, shifted by the
    # distance between them, so the blend gives them the same shape.
    from scipy.spatial import cKDTree
    e_s = sb[d_sb < EDGE_ROW_MM]                       # the side roots on the edge
    e_b = np.where((d_b < 0) & (d_b > -EDGE_ROW_MM))[0]  # the bang roots on the edge
    tr_s, tr_b = cKDTree(B['P'][e_s]), cKDTree(B['P'][roots[e_b]])
    shell = B.get('shell') or B['surf']

    def carry(p, off):
        """Line p moved by off along the skin, keeping its own height off the skin."""
        U = BF.unit(p - SF.CENTRE)
        base = shell.point(U)
        h = ((p - base) * shell.normals(U)).sum(1)
        U2 = BF.unit(base + off[None] - SF.CENTRE)
        return shell.point(U2) + shell.normals(U2) * h[:, None]

    def carried(tree, edge_P, edge_line, P):
        """The other family's line at P: the lines of its CARRY_K nearest edge roots, each carried over to P, averaged
        node by node at equal lengths, weighted by distance (Gaussian, CARRY_SIGMA_MM)."""
        dist, kk = tree.query(P, k=CARRY_K)
        w = np.exp(-0.5 * (dist / MM / CARRY_SIGMA_MM) ** 2) + 1e-9
        cs = [carry(edge_line(k), P - edge_P[k]) for k in kk]
        L = min(arclen(c)[-1] for c in cs)
        sg = np.arange(0, L, 1.0)
        acc = sum(wi * at_s(c, sg) for wi, c in zip(w, cs)) / w.sum()
        return acc

    cB = lambda P: carried(tr_s, B['P'][e_s], lambda k: side_line(B, int(e_s[k])), P)
    cD = lambda P: carried(tr_b, B['P'][roots[e_b]], lambda k: lines[int(e_b[k])], P)
    # each root's two candidates: its own family's line from it, and the other family's carried over to it. (Taking
    # the carried pair for both families on the edge, mixed into the own lines across the band, was tried: G1 4.6
    # deg against 2.3, and 61 crossings against 30. The averaged carried lines and a root's own line part by more
    # than the blend closes.)
    # the blend, made in the hair's direction and height, not its position: X'(s) = slerp(T_D(s), T_B(s), Lambda),
    # h(s) = (1 - Lambda) h_D + Lambda h_B, walked over the hair's bed. Blending positions instead turns a hair by
    # dLambda/ds times the two lines' separation, which is 10 cm by the bangs' ends (curl 6 mm); blending directions
    # turns it by dLambda/ds times their angle (see TURN_MM).
    new = list(lines)
    jobs = []
    for j in inb:
        P, pd = B['P'][roots[j]], lines[j]
        jobs.append(dict(P=pd[0], D=[(pd, 1.0)], B=[(cB(P), 1.0)], d=float(d_b[j]), L=arclen(pd)[-1],
                         length=arclen(pd)[-1], own=None))
    for i, d in zip(sb, d_sb):
        P, ps = B['P'][i], side_line(B, i)
        cd = cD(P)
        jobs.append(dict(P=ps[0], D=[(cd, 1.0)], B=[(ps, 1.0)], d=float(d), L=arclen(cd)[-1],
                         length=arclen(ps)[-1], own=ps))
    paths = walk(shell, jobs, b)
    def like(q, p):
        """q at p's nodes, by share of length (the walk's length differs from its own line's by the height's change)"""
        a = arclen(p)
        return at_s(q, a / max(a[-1], 1e-9) * arclen(q)[-1] * (1 - 1e-9))
    for j, q in zip(inb, paths[:len(inb)]):
        new[j] = like(q, lines[j])
    side_new = {}
    for i, q, jb in zip(sb, paths[len(inb):], jobs[len(inb):]):
        side_new[int(i)] = like(q, jb['own'])
    info = dict(band_mm=b, bang_roots_in_band=int(len(inb)), side_roots_in_band=int(len(sb)),
                tie_roots=int((B['basin'][sb] == 0).sum()), lock_roots=int((B['basin'][sb] >= 2).sum()))
    return new, side_new, info


def join_gauge(B, roots, lines, side_lines=None, reach=FIT_MM, at_d=0.0):
    """G0 and G1 at the join, as the join diagnostics read a seam (PR #86 continuity.py: one-sided fits compared at
    the join). The roots near the edge are cut into rows (each side of the head, DEPTH_BIN_MM of depth behind the
    hairline). In each row, at each length s along the hairs, the hairs' points and tangents are fitted as quadratics
    in d (the roots' distance from the edge), one fit per side from the roots within `reach`, and the two
    fits are compared at d = 0:
        G0  the gap between the two sides' points there (mm), split into the ledge (along the skin's normal) and
            the rest (along the skin);
        G1  the angle between the two sides' tangents there (deg).
    A blend that changes smoothly across the edge gives G0 and G1 near 0 whatever its gradient; a jump between the
    families shows as their size."""
    surf = B['surf']
    d_b = edge_d(B, roots)
    side = np.where(B['basin'] != 1)[0]
    d_s = edge_d(B, side)
    # at_d != 0 is the gauge's control: the same fits about a line at_d from the edge, where both sides are one
    # family and one construction, read the gauge's own floor on hair of that shape
    kb = np.where((d_b < 0) & (np.abs(d_b - at_d) < reach))[0]
    ks = np.where((d_s >= 0) & (np.abs(d_s - at_d) < reach))[0]
    items = [(float(d_b[j]) - at_d, B['P'][roots[j]], lines[j]) for j in kb]
    for k in ks:
        i = int(side[k])
        ps = side_lines[i] if side_lines and i in side_lines else side_line(B, i)
        items.append((float(d_s[k]) - at_d, B['P'][i], ps))
    dd = np.array([it[0] for it in items])
    P0 = np.array([it[1] for it in items])
    deep = np.array([float(np.interp(0, [0], [0])) for _ in items])
    U = BF.unit(P0 - SF.CENTRE)
    # depth of each root behind the hairline
    from scipy.spatial import cKDTree
    tr = cKDTree(B['P'])
    deep = B['inside_mm'][tr.query(P0)[1]]
    rows = []
    for sx in (-1, 1):
        for lo in np.arange(0, PF.BANG_DEPTH_MM, DEPTH_BIN_MM):
            k = np.where((np.sign(P0[:, 0]) == sx) & (deep >= lo) & (deep < lo + DEPTH_BIN_MM))[0]
            kL, kR = k[dd[k] < 0], k[dd[k] >= 0]
            if len(kL) < 5 or len(kR) < 5:
                continue
            for smm in AT_MM:
                ends = {}
                for side_k, lab in ((kL, 'L'), (kR, 'R')):
                    A = np.array([at_s(items[m][2], np.array([smm, smm + 2.0])) for m in side_k])   # (n, 2, 3)
                    ok = ~np.isnan(A[:, 0, 0]) & ~np.isnan(A[:, 1, 0])
                    if ok.sum() < 5:
                        break
                    x = dd[side_k][ok]
                    X = A[ok, 0]
                    T = BF.unit(A[ok, 1] - A[ok, 0])
                    M = np.c_[np.ones_like(x), x, x * x]        # (quadratic, as continuity.compare's order 2:
                                                                # the blend's own curvature in d is not read as a gap)
                    cX = np.linalg.lstsq(M, X, rcond=None)[0]
                    cT = np.linalg.lstsq(M, T, rcond=None)[0]
                    ends[lab] = (cX[0], BF.unit(cT[0][None])[0])
                if len(ends) < 2:
                    continue
                (xa, ta), (xb, tb) = ends['L'], ends['R']
                g1 = math.degrees(math.acos(float(np.clip(ta @ tb, -1, 1))))
                n = surf.normals(BF.unit((0.5 * (xa + xb) - SF.CENTRE)[None]))[0]
                v = (xb - xa) / MM
                rows.append((smm, g1, abs(float(v @ n)), float(np.linalg.norm(v - (v @ n) * n))))
    if not rows:
        return {}
    R = np.array(rows)
    out = dict(rows=int(len(R) // len(AT_MM)), reach_mm=reach,
               limits=dict(g1_deg=G1_LIM, g0_mm=G0_LIM, along_skin_mm=OPEN_LIM))
    by = []
    for smm in AT_MM:
        k = R[:, 0] == smm
        if k.any():
            by.append(dict(at_mm=smm, g1_deg_median=round(float(np.median(R[k, 1])), 2),
                           g1_deg_max=round(float(R[k, 1].max()), 2),
                           ledge_mm_median=round(float(np.median(R[k, 2])), 2), ledge_mm_max=round(float(R[k, 2].max()), 2),
                           along_skin_mm_median=round(float(np.median(R[k, 3])), 2),
                           along_skin_mm_max=round(float(R[k, 3].max()), 2)))
    out['along'] = by
    out['g1_deg_median_all'] = round(float(np.median(R[:, 1])), 2)
    out['g1_ok'] = bool(np.median(R[:, 1]) <= G1_LIM[1])
    out['g1_pass'] = bool(np.median(R[:, 1]) <= G1_LIM[0])
    out['g0_ledge_ok'] = bool(np.median(R[:, 2]) <= G0_LIM)
    out['along_skin_ok'] = bool(np.median(R[:, 3]) <= OPEN_LIM)
    return out
