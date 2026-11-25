"""One-piece notch collar and lapel, drafted smooth (John, 2026-11-21: "the cut needs to be smooth so you can
redistribute some of the force to the lapels", "optimize the shape of the top so it can smoothly transfer force down
to the lapels").

onepiece.py laid the tailored collar's own pattern onto the lapel's at the old gorge. That leaves the roll a 67 degree
corner at the neck point (the gorge's angle), and turning the collar toward the roll's line at all runs its fall into
the lapel: a collar cut that way can only be joined at a corner, and it crumpled there. Here the top is drafted as a
tailor drafts a collar cut on with the front (a shawl collar's draft, with a notch cut in its edge; Mueller und Sohn,
the stand-up lapel collar: "extend the lapel break upwards", the back neck's length laid off along it, the collar
swung at the back; Linder, US 3,009,160: the notch a V-shaped opening in one blank):

  the roll       the lapel's roll line run straight on past the neck point for the front neck's length (to the
                 shoulder), then turned smoothly by the swing over the back neck's length to the centre back. One
                 line, no corner: the pull along the collar's roll runs straight down the lapel's.
  the swing      the tailor's half inch at the centre back (UKY CT-LMH.176; tailor.py's 0.485), toward the neck side,
                 so the outer edge is the longer and lies over the shoulders.
  the stand      the neck seam, the roll offset toward the neck by the stand: nothing at the neck point (the lapel's
                 roll), the collar's stand from the shoulder back. The front's neckline under it closes as a dart
                 (Linder's thin shaping dart): both are held on the neck edge.
  the fall       the outer edge, the roll offset outward by the fall: at the centre back the collar's fall_back, at the
                 front the notch's collar point, blended along the roll with no corner.
  the notch      a V cut into that edge at the lapel's notch G: its lower leg the lapel's top edge to the lapel's
                 point P (as the build cut it), its upper leg to the collar's point, the notch's angle opening.

    python3 blender/foldlab/onepiece2.py try FOLD.pkl COLLAR.pkl CAP_DIR OUT_DIR '{}' [--name x] [--lapel k]

EXPERIMENTAL."""
import json
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import onepiece as op    # noqa: E402

OP2 = dict(op.OP, renotch=True, bridge=4.0, notch_open=85, end_ratio=1.2, top_drop=1.0, keep_band=0.6, swing=0.485, stand=0.65, stand_side=0.12, fall_back=0.75, notch_deg=65, notch_end=0.6,
           bend_len=1.6, lay_r=2.0, back_neck=4.02, front_neck=4.49, design='lay', develop=False)


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def pattern(fold_pkl, collar_pkl, cap_dir, k, S):
    import shapely
    from scipy.spatial import Delaunay, cKDTree
    F = pickle.load(open(fold_pkl, 'rb'))
    C = pickle.load(open(collar_pkl, 'rb'))
    cap = dict(np.load(os.path.join(cap_dir, 'lapel%d.npz' % k), allow_pickle=True))
    flat, T, lap, built = cap['flat'], cap['T'], cap['lap'].astype(bool), cap['built']
    B, N, poly = cap['B'], cap['N'], cap['poly']
    M = F['M']
    w = M['w']
    sg = 1 if N[0] > 0 else -1
    X2, G2, X3 = F['gorges'][sg]
    h = S['spacing'] * S['w']
    u = (N - B) / np.linalg.norm(N - B)
    nperp = np.array([-u[1], u[0]])
    if nperp @ (np.array([0.0, N[1]]) - B) < 0:
        nperp = -nperp
    t_, sd = (flat - B) @ u, (flat - B) @ nperp
    rimv = np.unique(op._rim(T))
    rl = rimv[lap[rimv]]
    T_all0 = T
    Gm = op._mirror(G2[None], B, N)[0]
    Pm = op._mirror(np.asarray(poly)[3][None], B, N)[0]
    iG = rl[np.argmin(np.linalg.norm(flat[rl] - Gm, axis=1))]
    iP = rl[np.argmin(np.linalg.norm(flat[rl] - Pm, axis=1))]
    Gp, Pp = flat[iG], flat[iP]
    rp = np.nonzero(~lap & (np.abs(sd) < 0.015))[0]
    rp = rp[np.argsort(t_[rp])]
    Rt = flat[rp[-1]]                                              # the roll's top: the neck point
    iN = rl[np.argmin(np.linalg.norm(flat[rl] - Rt, axis=1))]
    Np = flat[iN]
    old = shapely.union_all([shapely.Polygon(flat[t]) for t in T]).buffer(1e-4).buffer(-1e-4)
    flap = shapely.union_all([shapely.Polygon(flat[t]) for t in T if lap[t].all()]).buffer(1e-4).buffer(-1e-4)
    strip = shapely.union_all([shapely.Polygon(flat[t]) for t in T if not lap[t].any()]).buffer(1e-4).buffer(-1e-4)
    # ---- the notch redrafted (John, 2026-11-21: "why our current with such a thin notch fails"): the build's notch opened
    # 30 deg, its collar end 1.9 cm against a lapel top edge of 6.4 cm (1 : 3.4), with 2.8 cm of cloth between the notch
    # and the roll to carry the collar's pull into an 8.9 cm lapel. Tailors cut the notch 60 to 90 deg (Paul Smith; Enzo
    # Custom: 75 to 90; Bespoke Unit: 90 classic); the riding habit John sent has collar end and lapel top about equal.
    # Here: the notch's tip moved out along the gorge to `bridge`, the lapel's point kept (its 3.5 in width), the collar's
    # end set from the tip at `notch_open` to the lapel's top edge, `end_ratio` of its length.
    log0 = []
    kept_T = T
    lap_top = None
    if S.get('renotch', True):
        Bt, Nt, Gt, Pt_ = np.asarray(B), np.asarray(N), np.asarray(poly[2]), np.asarray(poly[3])
        src = op._mirror(np.array([Nt, Gt, Pt_, Bt]), Bt, Nt)
        dst = np.array([Np, Gp, Pp, flat[rp[0]]])
        A_ = np.column_stack([src, np.ones(4)])
        aff = np.linalg.lstsq(A_, dst, rcond=None)[0]
        to_flat = lambda X: np.column_stack([np.atleast_2d(X), np.ones(len(np.atleast_2d(X)))]) @ aff
        sgo = np.sign(Gt[0] - Nt[0])                                  # (outward, in front view)
        if S.get('gorge_deg') is not None:
            # (the gorge set at this angle below level, outward: the notch's height. The build's fell at 43 deg, which put
            # the notch and the lapel's point low on the chest; the habit's notch sits high, near the shoulder's line)
            ga = math.radians(S['gorge_deg'])
            Gt2 = Nt + S['bridge'] / 2.62 * np.array([sgo * math.cos(ga), -math.sin(ga)])
        else:
            Gt2 = Nt + S['bridge'] / 2.62 * (Gt - Nt) / np.linalg.norm(Gt - Nt)
        if S.get('top_deg') is not None:
            # (the lapel's point moved along its top edge, set at this angle below level from the notch, to the lapel's
            # drafted width from the roll)
            ta = math.radians(S['top_deg'])
            dt_ = np.array([sgo * math.cos(ta), -math.sin(ta)])
            rn_ = np.array([-(Nt - Bt)[1], (Nt - Bt)[0]]) / np.linalg.norm(Nt - Bt)
            rn_ *= np.sign(rn_ @ (Pt_ - Bt))
            wl_ = (Pt_ - Bt) @ rn_
            k_ = (wl_ - (Gt2 - Bt) @ rn_) / (dt_ @ rn_)
            Pt_ = Gt2 + k_ * dt_
        eP = (Pt_ - Gt2) / np.linalg.norm(Pt_ - Gt2)
        Lc = np.linalg.norm(Pt_ - Gt2) / S['end_ratio']
        cands = [Gt2 + Lc * _rot(eP, sgn * math.radians(S['notch_open'])) for sgn in (1, -1)]
        Ct2 = max(cands, key=lambda c: c[1])
        Gp2, Cp2 = to_flat(op._mirror(Gt2[None], Bt, Nt))[0], to_flat(op._mirror(Ct2[None], Bt, Nt))[0]
        Pp_old = Pp
        Pp = to_flat(op._mirror(Pt_[None], Bt, Nt))[0]
        log0.append('notch: opening %.0f deg, cloth between notch and roll %.1f cm, collar end %.1f cm, lapel top %.1f cm (1 : %.2f), lapel width kept %.1f cm'
                    % (S['notch_open'], S['bridge'], Lc * 2.62, np.linalg.norm(Pt_ - Gt2) * 2.62, S['end_ratio'],
                       abs((Pt_ - Bt) @ np.array([-(Nt - Bt)[1], (Nt - Bt)[0]]) / np.linalg.norm(Nt - Bt)) * 2.62))
        # the old flap's top taken off (its turn's band along the roll kept, as solved), the new top meshed in
        tP = max((Pp - B) @ u, (Pp_old - B) @ u)
        t_cut = min((Pp - B) @ u, (Pp_old - B) @ u) - S.get('top_drop', 1.0)
        s_band = S.get('keep_band', 0.6)
        cen_t = flat[T].mean(1)
        ct, cs = (cen_t - B) @ u, (cen_t - B) @ nperp
        drop = lap[T].all(1) & (ct > t_cut) & (cs > s_band)
        kept_T = T[~drop]
        # the outer edge between the cut and the point, from the old flap's boundary
        ring_ = np.array(flap.exterior.coords)
        rt_, rs_ = (ring_ - B) @ u, (ring_ - B) @ nperp
        oe = ring_[(rt_ > t_cut - 0.05) & (rt_ < tP + 0.01) & (rs_ > s_band + 0.3)]
        oe = oe[np.argsort(-((oe - B) @ u))]
        if S.get('top_deg') is not None:
            # (the outer edge from the moved point down to the old edge at the cut: a soft curve, tangent to the old edge there)
            Ec = oe[-1]
            ring_d = ring_[(rt_ < t_cut) & (rt_ > t_cut - 0.6) & (rs_ > s_band + 0.3)]
            tan_ = (Ec - ring_d[np.argmin((ring_d - B) @ u)]) if len(ring_d) else (Pp - Ec)
            tan_ /= np.linalg.norm(tan_)
            L_ = np.linalg.norm(Pp - Ec)
            # (a gentle belly out over the bust, belly units at the middle; not tangent to the old edge, which bulged into a lobe)
            ch_ = Pp - Ec
            no_ = np.array([-ch_[1], ch_[0]]) / max(np.linalg.norm(ch_), 1e-9)
            no_ *= np.sign(no_ @ nperp) if abs(no_ @ nperp) > 1e-6 else 1.0
            ctl = 0.5 * (Ec + Pp) + 2.0 * S.get('belly_top', 0.2) * no_
            qq = np.linspace(0, 1, 30)[:, None]
            oe = ((1 - qq) ** 2 * Ec + 2 * qq * (1 - qq) * ctl + qq ** 2 * Pp)[::-1][1:]
        pc = lambda tt, ss: B + tt * u + ss * nperp
        lap_top = shapely.Polygon(np.vstack([pc(t_cut, s_band)[None], pc((Np - B) @ u, s_band)[None], Np[None], Gp2[None], Pp[None], oe])).buffer(0)
        if lap_top.geom_type != 'Polygon':
            lap_top = max(lap_top.geoms, key=lambda g: g.area)
        Gp, Cp_new = Gp2, Cp2
    else:
        Cp_new = None
    # the fall's side: where the flap lies (+s)
    # ---- the roll on past the neck point: straight for the front neck, then turned by the swing toward the neck side
    LF, LB = S['front_neck'], S['back_neck']
    alpha = math.asin(min(S['swing'] / LB, 0.9))
    Lt = LF + LB
    a = np.arange(0.0, Lt + 1e-9, 0.25 * h)
    b0, b1 = LF - 0.5 * S['bend_len'], LF + 0.5 * S['bend_len']
    # heading: 0 up to b0, then turning (smoothly: a smoothstep in the heading) to -alpha toward the neck side by b1
    # (the neck side is -s; turning toward -s from +u is a turn by -sign)
    turn_sign = -1.0 if (_rot(u, 0.1) @ nperp) > 0 else 1.0       # (a positive turn of u leans it toward +s: turn the other way)
    q = np.clip((a - b0) / (b1 - b0), 0, 1)
    q = q * q * (3 - 2 * q)
    # (over the bend the heading turns by alpha in all, and the line past it keeps that heading: the swing measured at
    # the centre back is then LB * sin(alpha) only if the turn is at the shoulder; the bend is centred there)
    head = turn_sign * alpha * q
    dirs = np.column_stack([np.cos(head) * u[0] - np.sin(head) * u[1], np.sin(head) * u[0] + np.cos(head) * u[1]])
    roll = Rt[None] + np.vstack([[0, 0], np.cumsum(0.5 * (dirs[1:] + dirs[:-1]) * np.diff(a)[:, None], axis=0)])
    nrm = np.column_stack([-dirs[:, 1], dirs[:, 0]])
    nrm *= np.sign(nrm[0] @ nperp)
    # the stand and the fall along it
    st, ss = S['stand'] * w, S['stand_side'] * w
    qs = np.clip(a / LF, 0, 1)
    stand = st * qs * qs * (3 - 2 * qs)                              # (nothing at the neck point, the full stand by the shoulder)
    seam = roll - stand[:, None] * nrm
    # the notch: its collar point C on the lapel's notch angle from the top edge, the end's length
    e_ = (Pp - Gp) / np.linalg.norm(Pp - Gp)
    cands = [Gp + S['notch_end'] * w * _rot(e_, sgn * math.radians(S['notch_deg'])) for sgn in (1, -1)]
    Cp = max(cands, key=lambda c: (c - Rt) @ u)
    if Cp_new is not None:
        Cp = Cp_new
    aC = float(np.clip((Cp - Rt) @ u, 0.05, Lt))
    sC = float((Cp - roll[np.argmin(np.abs(a - aC))]) @ nrm[np.argmin(np.abs(a - aC))])
    fb = S['fall_back'] * w + st                # (the outer edge's offset from the roll's line on the pattern at the back: the
    # fall must reach over the stand and the neck seam it hides)
    qf = np.clip((a - aC) / max(Lt - aC, 1e-6), 0, 1)
    qf = qf * qf * (3 - 2 * qf)
    fall = sC + (fb - sC) * qf
    outer = roll + fall[:, None] * nrm
    ok = a >= aC
    # the polygon: N' up the neck seam to the CB, out along the CB, back along the outer edge to C, the notch's upper
    # leg to G, and the old gorge (inside the cloth now) back to N'
    ring = np.vstack([Np[None], seam[1:], outer[::-1][outer[::-1, 0] * 0 == 0][: int(ok.sum())], Cp[None], Gp[None]])
    cpoly = shapely.Polygon(ring).buffer(0)
    if cpoly.geom_type != 'Polygon':
        cpoly = max(cpoly.geoms, key=lambda g: g.area)
    old = shapely.union_all([shapely.Polygon(flat[t]) for t in kept_T]).buffer(1e-4).buffer(-1e-4)
    if lap_top is not None:
        cpoly = shapely.unary_union([cpoly, lap_top]).buffer(0.01).buffer(-0.01)
        if cpoly.geom_type != 'Polygon':
            cpoly = max(cpoly.geoms, key=lambda g: g.area)
    piece = shapely.unary_union([old, cpoly]).buffer(0.01).buffer(-0.01)
    if piece.geom_type != 'Polygon':
        piece = max(piece.geoms, key=lambda g: g.area)
    log = log0 + ['shawl draft: roll on %.2f straight past the neck point, then swung %.1f deg over %.2f to the CB; stand %.2f, fall at the CB %.2f, collar point %.2f out from the roll'
           % (LF, math.degrees(alpha), LB, st, fb, sC)]
    # ---- mesh: the build's lapel mesh whole, the collar meshed on, joined at the old gorge's rim points
    add = cpoly.difference(old.buffer(0.02))
    adds = [g for g in (add.geoms if hasattr(add, 'geoms') else [add]) if g.area > 4 * h * h]
    add = shapely.unary_union(adds)
    pts = []
    for g in adds:
        rg = g.exterior
        pts.append(np.array([rg.interpolate(d).coords[0] for d in np.arange(0, rg.length, 0.7 * h)]))
    bx = add.bounds
    gx, gy = np.meshgrid(np.arange(bx[0], bx[2], h), np.arange(bx[1], bx[3], h * 0.866))
    gx[1::2] += 0.5 * h
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid = grid[shapely.contains_xy(add.buffer(-0.5 * h), grid[:, 0], grid[:, 1])]
    rline = shapely.LineString(np.vstack([flat[rp], roll[1:]]))
    rc_s = np.array([rline.interpolate(d).coords[0] for d in np.arange(0, rline.length, 0.7 * h)])
    rc_s = rc_s[shapely.contains_xy(add.buffer(-0.3 * h), rc_s[:, 0], rc_s[:, 1])]
    grid = grid[shapely.distance(rline, shapely.points(grid)) > 0.45 * h]
    new = np.vstack(pts + [rc_s, grid])
    new = new[cKDTree(flat[np.unique(kept_T)]).query(new)[0] > 0.45 * h]
    keep = np.ones(len(new), bool)
    for i, j in sorted(cKDTree(new).query_pairs(0.35 * h)):
        if keep[i] and keep[j]:
            keep[j] = False
    new = new[keep]
    n0 = len(flat)
    rim_k = np.unique(op._rim(kept_T))
    near_old = rim_k[shapely.distance(add, shapely.points(flat[rim_k])) < 1.5 * h]
    Pd = np.vstack([flat[near_old], new])
    tri = Delaunay(Pd).simplices
    cen = Pd[tri].mean(1)
    A_, B_, C_ = Pd[tri[:, 0]], Pd[tri[:, 1]], Pd[tri[:, 2]]
    ar = 0.5 * np.abs((B_[:, 0] - A_[:, 0]) * (C_[:, 1] - A_[:, 1]) - (B_[:, 1] - A_[:, 1]) * (C_[:, 0] - A_[:, 0]))
    okt = shapely.contains_xy(add.buffer(S.get('join_buf', 0.35) * h), cen[:, 0], cen[:, 1]) & ~shapely.contains_xy(old.buffer(-0.05 * h), cen[:, 0], cen[:, 1])
    okt &= shapely.contains_xy(piece.buffer(0.02), cen[:, 0], cen[:, 1]) & (ar > 0.03 * h * h)
    gidx = np.concatenate([near_old, n0 + np.arange(len(new))])
    tri = gidx[tri[okt]]
    P = np.vstack([flat, new])
    Tall = np.vstack([kept_T, tri])
    used = np.unique(Tall)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, Tall = P[used], remap[Tall]
    A_, B_, C_ = P[Tall[:, 0]], P[Tall[:, 1]], P[Tall[:, 2]]
    crs = (B_[:, 0] - A_[:, 0]) * (C_[:, 1] - A_[:, 1]) - (B_[:, 1] - A_[:, 1]) * (C_[:, 0] - A_[:, 0])
    Tall[crs < 0] = Tall[crs < 0][:, [0, 2, 1]]
    # (holes left where the new cloth meets the old mesh's rim: every boundary loop but the outline is filled)
    nfill = 0
    for _ in range(3):
        be = op._rim(Tall)
        loops = op._loops(be)
        if len(loops) <= 1:
            break
        loops.sort(key=len)
        extra = []
        for lp in loops[:-1]:
            if len(lp) < 3:
                continue
            pg = shapely.Polygon(P[lp]).buffer(0)
            if pg.area > 6 * h * h * max(1, len(lp) / 6):
                continue
            tr_ = Delaunay(P[lp]).simplices
            cc_ = P[lp][tr_].mean(1)
            tr_ = tr_[shapely.contains_xy(pg.buffer(1e-6), cc_[:, 0], cc_[:, 1])]
            extra.append(np.asarray(lp)[tr_])
        if not extra:
            break
        extra = np.vstack(extra)
        nfill += len(extra)
        Tall = np.vstack([Tall, extra])
    A_, B_, C_ = P[Tall[:, 0]], P[Tall[:, 1]], P[Tall[:, 2]]
    crs = (B_[:, 0] - A_[:, 0]) * (C_[:, 1] - A_[:, 1]) - (B_[:, 1] - A_[:, 1]) * (C_[:, 0] - A_[:, 0])
    Tall[crs < 0] = Tall[crs < 0][:, [0, 2, 1]]
    Tall = Tall[np.abs(crs) > 1e-9]
    log.append('mesh: %d triangles filled into gaps at the join; boundary loops left %d' % (nfill, len(op._loops(op._rim(Tall)))))
    is_old = used < n0
    is_strip = is_old & ~lap[np.minimum(used, n0 - 1)]
    in_col = ~is_old
    in_lap_new = in_col & (shapely.contains_xy(lap_top.buffer(0.02), P[:, 0], P[:, 1]) if lap_top is not None else np.zeros(len(P), bool))
    # each new point's place on the draft: along the roll (a) and off it (d, + toward the fall)
    kr = cKDTree(roll)
    _, jr = kr.query(P)
    ap, dp = a[jr], ((P - roll[jr]) * nrm[jr]).sum(1)
    seam_line = shapely.LineString(seam)
    on_seam = in_col & (shapely.distance(seam_line, shapely.points(P)) < 0.3 * h) & (ap > 0.3)
    on_cb = in_col & (ap > Lt - 0.3 * h)
    dist_roll = shapely.distance(rline, shapely.points(P))
    on_roll = dist_roll < 0.4 * h
    in_band = dist_roll < S.get('band', 0.6)
    # ---- the start: the old collar's solved shape, read at the same place along its own roll and off it; near the
    # neck point the lapel's cloth carried on (laid as the lapel lies), blended over lay_r
    UV, kind, cpos = C['UV'], C['kind'], C['pos']
    hs = np.sign(UV[:, 0]) == -sg
    ru = UV[(kind == 'r') & hs]
    ru = ru[np.argsort(-np.abs(ru[:, 0]))]                          # (front end first)
    ra = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ru, axis=0), axis=1))])
    rt = np.gradient(ru, axis=0)
    rt /= np.linalg.norm(rt, axis=1, keepdims=True)
    rn = np.column_stack([-rt[:, 1], rt[:, 0]])
    ou = UV[(kind == 'o') & hs].mean(0)
    rn *= np.sign(((ou - ru) * rn).sum(1).mean())
    aa = np.clip(ap / Lt * ra[-1], 0, ra[-1])
    base = np.column_stack([np.interp(aa, ra, ru[:, 0]), np.interp(aa, ra, ru[:, 1])])
    nn_ = np.column_stack([np.interp(aa, ra, rn[:, 0]), np.interp(aa, ra, rn[:, 1])])
    # (the old collar's own stand and fall at that place, so a point at the same share of the stand or fall lands
    # where the old collar's did)
    quv = base + dp[:, None] * nn_
    CT = C['T']
    cT = CT[hs[CT].all(1)]
    D = np.zeros((len(P), 3))
    D[is_old] = built[used[is_old]]
    D[in_col] = op._bary(quv[in_col], UV, cT, cpos)
    # near the neck point: the lapel carried on
    fall_pt = dp > 0
    flapT = T[lap[T].all(1)]
    stripT = T[~lap[T].any(1)]
    ext = np.zeros((len(P), 3))
    mf, ms = in_col & fall_pt, in_col & ~fall_pt
    ext[mf] = op._bary(P[mf], flat, flapT, built)
    ext[ms] = op._bary(P[ms], flat, stripT, built)
    dN = np.linalg.norm(P - Np, axis=1)
    lay_w = np.clip(1 - dN / S['lay_r'], 0, 1)
    lay_w = lay_w * lay_w * (3 - 2 * lay_w)
    lay_w[~in_col] = 0
    D[in_col] = lay_w[in_col, None] * ext[in_col] + (1 - lay_w[in_col, None]) * D[in_col]
    # (the lapel's new top is the lapel's cloth: laid as the lapel lies)
    lt_ = in_lap_new & ~(shapely.contains_xy(cpoly.difference(lap_top).buffer(-0.01), P[:, 0], P[:, 1]) if lap_top is not None else False)
    D[lt_] = op._bary(P[lt_], flat, flapT, built)
    # the neck seam's targets: the bodice's neck edge, from the neck point round to the CB, by length
    E = np.asarray(F['E'])
    Es = E[np.sign(E[:, 0]) == sg]
    X3p = np.asarray(X3)
    # order from the neck point (front) round to the back
    ang = np.arctan2(Es[:, 0] * sg, -(Es[:, 1] - M['yc']))
    Es = Es[np.argsort(ang)]
    j0 = int(np.argmin(np.linalg.norm(Es - X3p, axis=1)))
    Es = np.vstack([Es[j0:], [[0.0, Es[-1, 1], Es[-1, 2]]]])
    ea = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Es, axis=0), axis=1))])
    sa = ap[on_seam] / Lt * ea[-1]
    D[on_seam] = np.column_stack([np.interp(sa, ea, Es[:, c]) for c in range(3)])
    on_seam &= lay_w < 0.5
    log.append('neck edge from the neck point to the CB %.2f (pattern %.2f)' % (ea[-1], Lt))
    corners = dict(N=Np, G=Gp, P=Pp, C=Cp)
    return dict(P=P, T=Tall, D=D, is_strip=is_strip, is_old=is_old, in_col=in_col, on_seam=on_seam, cb=on_cb, on_roll=on_roll,
                in_band=in_band, lay_w=lay_w, roll=np.vstack([flat[rp], roll[1:]]), piece=piece, flap=flap, cpoly=cpoly,
                strip=strip, corners=corners, F=F, C=C, sg=sg, log=log, B=B, N=N, seam=seam, outer=outer[ok])


def main():
    cmd = sys.argv[1]
    fold_pkl, collar_pkl, cap_dir, out = sys.argv[2:6]
    os.makedirs(out, exist_ok=True)
    S = dict(OP2)
    if len(sys.argv) > 6 and not sys.argv[6].startswith('--'):
        S.update(json.loads(sys.argv[6]))
    name = sys.argv[sys.argv.index('--name') + 1] if '--name' in sys.argv else 'op2'
    ks = [int(sys.argv[sys.argv.index('--lapel') + 1])] if '--lapel' in sys.argv else [0, 1]
    for k in ks:
        Pt = pattern(fold_pkl, collar_pkl, cap_dir, k, S)
        for l_ in Pt['log']:
            print('lapel %d: %s' % (k, l_))
        print('lapel %d: one piece: %d points, %d triangles; seam held %d, CB %d' % (k, len(Pt['P']), len(Pt['T']), Pt['on_seam'].sum(), Pt['cb'].sum()))
        op.sheet_pattern(Pt, os.path.join(out, '%s-L%d-pattern.png' % (name, k)), 'one-piece notch collar and lapel, smooth draft, lapel %d' % k)
        if cmd == 'try':
            R = op.solve(Pt, S, log=lambda s: print('lapel %d: %s' % (k, s), flush=True))
            np.savez(os.path.join(out, '%s-L%d.npz' % (name, k)), X=R['X'], P=Pt['P'], T=Pt['T'], strain=R['strain'], D=Pt['D'],
                     in_col=Pt['in_col'], is_strip=Pt['is_strip'], on_seam=Pt['on_seam'], on_roll=Pt['on_roll'])
            print('lapel %d: solved in %.0f s' % (k, R['secs']))


if __name__ == '__main__':
    main()
