"""The funnel into the tie: the side hair's last stretch, built as a contraction so the lines, and the shape they
make, converge on the tie.

The field's side lines run over the head to the tail's first section, which lies inside the hair on the head, and
only then out to the tie: seen from the side the hair ended in a cylinder stub with the tie on its end, and no line
pointed at the tie. Hair gathered into a tie is a funnel: from where the hand gathers it, each hair runs nearly
straight to its own place in the tie, so the surface is a cone that follows the lines (a nozzle's contraction).

For every hair that reaches the tie: the stretch from FUNNEL_MM of its own length before the tie's plane, to its own
crossing point of that plane, is replaced by a cubic Hermite curve that leaves the old line with its direction there
and arrives at the crossing point with the direction the hair has just past the tie. So position and direction
match at both ends (G1), each hair keeps its own target (no hair changes place with a neighbour that did not already),
and the tail picks up exactly where the sides stop. The curve is kept outside the hair's bed on the head.

Measured on the first build: the lines already close on the tail's axis as a wide cone over the head (57 degrees
half angle), then turn up off the head and run 15 to 27 mm as a cylinder of the free bundle's radius (23 mm) to the
tie: that cylinder is the stub. Two changes remove it. (1) The tie is tight: a band packs hair to about half
(TIE_PACKING), so its radius is sqrt(hairs x section / (pi x packing)), about 16 mm here, and each hair's crossing
point is drawn in toward the axis by that ratio; past the tie the bundle flares back to its free radius over
FLARE_MM. (2) A hair arrives heading mostly along its own chord (ARRIVE_CHORD), not along the tail's axis, so the
funnel stays a cone right up to the band, and the hair turns inside the band, as it does under a real tie.

Gauges returned: the turn at the tie's node (deg; under the band), and the smallest bend radius in the funnel before
the band against the 19 mm curl bound. Whether hairs cross inside the funnel is not gauged (angular rank round the
axis means nothing for a hair that ends inside the disc, not on its rim).
FUNNEL_MM is DRAFT (a hand's width of gathered hair; the coordinator's contraction rule starts its blend 30 to 40 mm out).
"""
import numpy as np
import scalp_field as SF

MM = SF.MM
FUNNEL_MM = 60.0        # DRAFT
TIE_PACKING = 0.5       # a tie packs hair to about half (hair_volume.py's note); DRAFT
FLARE_MM = 45.0         # DRAFT: the bundle is back at its free radius this far past the tie
ARRIVE_CHORD = 0.6      # DRAFT: a hair's heading at the tie: this much its own chord's, the rest the tail's
BED_GAP_MM = 1.0
CURL_MIN_MM = 19.0      # the field's curl bound


def funnel(paths, sel, c_tie, t_tie, r_tie, shell, n_hairs=None):
    """paths (n, m, 3) nan-padded; sel: fibres to gather; c_tie, t_tie: the tie plane's centre and the tail's
    direction there; r_tie: the bundle's radius (head heights). Returns (paths, u per node (0 off the funnel, rising
    to 1 at the tie), report)."""
    Q = np.array(paths, float)
    n, m, _ = Q.shape
    Uu = np.zeros((n, m))
    t_tie = np.asarray(t_tie, float) / np.linalg.norm(t_tie)
    turn, rmin, a0, a1, done, ends_ = [], [], [], [], 0, []
    e1 = np.cross(t_tie, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(t_tie, e1)
    # the tie's own radius from the hair it holds, against the radius the lines cross its plane at now
    rc = []
    for i in np.where(sel)[0]:
        p = Q[i][~np.isnan(Q[i][:, 0])]
        d = (p - c_tie) @ t_tie
        rr = np.linalg.norm(p - c_tie - d[:, None] * t_tie, axis=1)
        cr = np.where((d[:-1] < 0) & (d[1:] >= 0) & (rr[1:] < 3 * r_tie))[0]
        if len(cr):
            rc.append(rr[cr[0] + 1])
    r_now = float(np.percentile(rc, 90))
    r_tight = np.sqrt(n_hairs * (0.035 * MM) ** 2 / TIE_PACKING) if n_hairs else r_now
    kq = min(1.0, r_tight / r_now)
    for i in np.where(sel)[0]:
        p = Q[i]
        good = ~np.isnan(p[:, 0]); k = int(good.sum())
        if k < 8:
            continue
        p = p[:k]
        d = (p - c_tie) @ t_tie
        rr = np.linalg.norm(p - c_tie - d[:, None] * t_tie, axis=1)
        cross = np.where((d[:-1] < 0) & (d[1:] >= 0) & (rr[1:] < 3 * r_tie))[0]
        if not len(cross):
            continue
        i1 = int(cross[0]) + 1
        if i1 >= k - 2 or i1 < 4:
            continue
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(seg)] / MM
        i0 = int(np.searchsorted(arc, arc[i1] - FUNNEL_MM))
        i0 = max(i0, 2)
        if i1 - i0 < 3:
            continue
        # the tie drawn tight, and the flare back out past it
        kk = kq + (1 - kq) * SF.sm(0.0, FLARE_MM, arc[i1:] - arc[i1])      # (by the hair's own length past the band:
        ax = c_tie + d[i1:, None] * t_tie                                  # the tail soon hangs away from the tie's axis)
        p[i1:] = ax + (p[i1:] - ax) * kk[:, None]
        Q[i, i1:k] = p[i1:]
        P0, P1 = p[i0], p[i1]
        L = np.linalg.norm(P1 - P0)
        T0 = p[i0] - p[i0 - 1]; T0 *= L / np.linalg.norm(T0)
        T1 = p[i1 + 1] - p[i1]; T1 /= np.linalg.norm(T1)
        T1 = ARRIVE_CHORD * (P1 - P0) / L + (1 - ARRIVE_CHORD) * T1; T1 *= L / np.linalg.norm(T1)
        t = ((arc[i0:i1 + 1] - arc[i0]) / (arc[i1] - arc[i0]))[:, None]
        h = (2 * t ** 3 - 3 * t ** 2 + 1) * P0 + (t ** 3 - 2 * t ** 2 + t) * T0 + (-2 * t ** 3 + 3 * t ** 2) * P1 + (t ** 3 - t ** 2) * T1
        # outside the bed, the push easing to nothing at both ends
        U = h - SF.CENTRE; r = np.linalg.norm(U, axis=1); U /= r[:, None]
        need = np.maximum(shell.radius(U) + BED_GAP_MM * MM - r, 0.0) * (4 * t[:, 0] * (1 - t[:, 0])) ** 0.5
        h = h + U * need[:, None]
        h[0], h[-1] = P0, P1
        Q[i, i0:i1 + 1] = h
        Uu[i, i0:i1 + 1] = t[:, 0] * t[:, 0] * (3 - 2 * t[:, 0])
        Uu[i, i1 + 1:k] = 1.0
        q = Q[i, :k]
        ea, eb = q[i1] - q[i1 - 1], q[i1 + 1] - q[i1]
        turn.append(np.degrees(np.arccos(np.clip(ea @ eb / np.linalg.norm(ea) / np.linalg.norm(eb), -1, 1))))
        e = np.diff(q[i0 - 1:i1 + 1], axis=0); ln = np.linalg.norm(e, axis=1)
        ang = np.arccos(np.clip((e[1:] * e[:-1]).sum(1) / (ln[1:] * ln[:-1]), -1, 1))
        rmin.append(float((0.5 * (ln[1:] + ln[:-1]) / np.maximum(ang, 1e-6)).min() / MM))
        v0, v1 = P0 - c_tie, P1 - c_tie
        a0.append(np.arctan2(v0 @ e2, v0 @ e1)); a1.append(np.arctan2(v1 @ e2, v1 @ e1)); ends_.append(P1)
        done += 1
    a0, a1 = np.array(a0), np.array(a1)
    o = np.argsort(a0); rk = np.argsort(np.argsort(a1))[o]
    dr = np.abs(rk - np.arange(len(rk))); dr = np.minimum(dr, len(rk) - dr)
    turn, rmin = np.array(turn), np.array(rmin)
    ends_ = np.array(ends_)
    cen = ends_.mean(0)
    r_seen = float(np.percentile(np.linalg.norm(ends_ - cen - ((ends_ - cen) @ t_tie)[:, None] * t_tie, axis=1), 97))
    rep = dict(centre=cen, seen_radius_mm=round(r_seen / MM, 1), tie_radius_mm=round(r_tight / MM, 1), was_mm=round(r_now / MM, 1), fibres=int(done), of=int(sel.sum()), tie_turn_deg_median=round(float(np.median(turn)), 1), tie_turn_deg_p95=round(float(np.percentile(turn, 95)), 1),
               bend_radius_mm_min_p05=round(float(np.percentile(rmin, 5)), 1), under_curl_bound_share=round(float((rmin < CURL_MIN_MM).mean()), 4))
    return Q, Uu, rep
