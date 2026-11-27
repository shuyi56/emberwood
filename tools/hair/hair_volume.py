"""Volume and layering for hair combed over the head: the field's side lines lie on the skin itself, so the hair
drawn along them is a film that shows the head's own surface (its ridges and creases) and has no depth to shade.

Real combed hair is a stack, and three things give it its depth here:

1. Layers. A hair that has run s mm from its root lies over every hair rooted along its path so far, so the hair
   under it is the roots it has passed: root density x s hairs per mm of width, each taking its section's area,
   packed at PACKING. Its height off the bed is therefore

       h(s) = ROOT_PER_MM2 x A_hair x s / PACKING,

   rising along the hair: the hair from the hairline is the top layer by the time it reaches the tie, the hair
   rooted near the tie the bottom. ROOT_PER_MM2 2.0 (scalp density 175 to 300 per cm2, Robbins ch. 1), A_hair the
   section of a 70 um hair, PACKING 0.12 (DRAFT: combed back but not slicked; a tie packs to about 0.5).
2. Root stand. A hair leaves the scalp at an angle before it is combed flat, so the whole stack stands ROOT_MM off
   the skin within ROOT_OVER_MM of the root (DRAFT).
3. Comb furrows. Combed to a tie, hair gathers into locks that run along the flow and converge on the tie, with a
   furrow between each two: the relief the eye reads as "hair" on the side of a ponytail. A lock is the hair that
   reaches the tie at the same angle round the tail's axis (hair arriving together has run together), LOCKS of them
   of uneven width; across each the surface is a rounded ridge FURROW_MM high at most and never steeper than
   FURROW_SLOPE (so furrows flatten as the locks narrow toward the tie), and each lock stands a little higher or
   lower than its neighbours (LOCK_STEP_MM). DRAFT amounts.

The height is capped (CAP_MM), fades back to the field's own line at the tie, where the band presses the stack flat,
and is laid on the hair's bed (bangs_flow.Shell: the skin with its hollows closed and its noise eased), not the skin,
so the head's creases do not carry through; the bed is taken up over the first BED_OVER_MM from the root and by
BED_MAX_MM at most, so roots stay in the skin. Each root lock (a 7 mm cell) is LOCK_VAR thicker or thinner.

layer() also returns the tilt of the hair surface's normal that the furrows make (per node, a vector to add to the
unit outward before normalising), for the page to shade by: without it the relief is lit as the smooth head; and each fibre's comb lock (-1: none), so a
lock can take one tone.
"""
import math
import numpy as np
import scalp_field as SF

MM = SF.MM
ROOT_PER_MM2 = 2.0
HAIR_DIAM_MM = 0.07
PACKING = 0.12          # DRAFT
CAP_MM = 9.0            # DRAFT
LOCK_MM = 7.0
LOCK_VAR = 0.18         # DRAFT (0.35 made 7 mm cells of unequal height: a bristly crest on the top outline from the front)
ROOT_MM = 2.0           # DRAFT
ROOT_OVER_MM = 8.0      # DRAFT
BED_OVER_MM = 12.0      # DRAFT
BED_MAX_MM = 5.0        # DRAFT
LOCKS = 46              # DRAFT: comb locks round the tie
FURROW_MM = 1.3         # DRAFT: a lock's ridge over its furrows, at most (2.6 read as ridges on the head)
FURROW_SLOPE = 0.8      # DRAFT: the steepest side of a ridge (rise over run)
LOCK_STEP_MM = 0.6      # DRAFT (was 1.3): a lock stands this much higher or lower than the mean
FURROW_FROM_MM = (25.0, 70.0)   # DRAFT: furrows grow in over this run from the root (later than 10..40, so the
                                # hair just behind the fringe stays one smooth surface)
TOP_EL_DEG = 35.0       # the top section: roots above this elevation from the skull's centre (above the parietal ridge,
                        # where a stylist's horseshoe parting runs; Milady Standard Cosmetology, sectioning)
NAPE_EL_DEG = -5.0      # the nape section: roots at the back below this
LAYER_GAP_MM = 1.2      # DRAFT: the top section lies this much clear of the sections it is combed over (2.5 stacked
                        # with the root stand and furrows to a 12 mm roll 60 mm behind the fringe; the hairs' own
                        # packed thickness there is 3.8 mm)
LAYER_FROM_MM = (15.0, 50.0)    # DRAFT: it comes clear over this run from the root
JUNCTION_MM = 5.0       # DRAFT: where swept-back hair is rooted beside a fringe, its root stands this high: the fringe
                        # stands 4 to 6 mm within 10 to 15 mm of its own roots (measured on this groom), and the
                        # swept-back hair's 1 to 3 mm there left a trough across the head with a roll behind it
JUNCTION_NEAR_MM = (30.0, 6.0)  # DRAFT: none of it past the first distance from the fringe's roots, all within the second
JUNCTION_OVER_MM = 10.0 # DRAFT: reached over this run from the root
TIE_FADE_MM = (4.0, 40.0)   # DRAFT: no lift within the first of the tie's rim, all of it past the second
RATE = ROOT_PER_MM2 * math.pi * (HAIR_DIAM_MM / 2) ** 2 / PACKING       # mm of height per mm of hair


def layer(paths, sel, shell, tie_c, tie_r, tie_axis=None, seed=3, junction=None):
    """Lift fibres paths[sel] (n, m, 3) off the bed in layers. tie_c, tie_r: the tie's centre and radius; tie_axis:
    the tail's direction there (for the comb locks; without it, no furrows). Returns (paths, tilt, comb lock per fibre, layer per fibre, report)."""
    rng = np.random.default_rng(seed)
    P = np.array(paths[sel], float)
    n, m, _ = P.shape
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    S = np.c_[np.zeros(n), np.cumsum(seg, 1)] / MM
    cell = np.floor(P[:, 0] / (LOCK_MM * MM)).astype(np.int64)
    key = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    uk, lock = np.unique(key, return_inverse=True)
    var = 1 + LOCK_VAR * (2 * rng.random(len(uk)) - 1)
    flat = P.reshape(-1, 3)
    U = flat - SF.CENTRE
    r = np.linalg.norm(U, axis=1); U /= np.maximum(r[:, None], 1e-12)
    rb = shell.radius(U)
    d_tie = (np.linalg.norm(flat - tie_c, axis=1) / MM - tie_r / MM).reshape(n, m)
    k_tie = np.argmin(d_tie, axis=1)                   # the part combed over the head: up to the hair's nearest
    s_tie = S[np.arange(n), k_tie]                     # point to the tie; the lift eases out toward it, by the
    win = SF.sm(0.0, TIE_FADE_MM[1], s_tie[:, None] - S) * (np.arange(m)[None, :] <= k_tie[:, None])   # hair's own length
    win = np.minimum(win, SF.sm(TIE_FADE_MM[0], TIE_FADE_MM[1], d_tie))
    win = (win.reshape(-1) * SF.sm(14.0, 9.0, (r - rb) / MM))          # (and only near the head)
    Sf = S.reshape(-1)
    h = np.minimum(RATE * Sf * np.repeat(var[lock], m), CAP_MM) + ROOT_MM * SF.sm(0.0, ROOT_OVER_MM, Sf)
    # sections, as hair is parted to be dressed: the top (inside the horseshoe) is combed over the sides, the sides over
    # the nape. Each root cell goes whole to one section (its edge is a parting between locks, not a ruled line).
    u0 = P[:, 0] - SF.CENTRE; u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    el = np.degrees(np.arcsin(u0[:, 2])) + 4.0 * (2 * rng.random(len(uk)) - 1)[lock]
    top = el > TOP_EL_DEG
    nape = (el < NAPE_EL_DEG) & (P[:, 0, 1] > SF.CENTRE[1])
    section = np.where(top, 1.0, np.where(nape, 0.3, 0.6))             # how far out the section lies (1 outermost)
    h = h + np.repeat(top * LAYER_GAP_MM, m) * SF.sm(LAYER_FROM_MM[0], LAYER_FROM_MM[1], Sf)
    hj = np.zeros(n * m); rep_j = None
    if junction is not None and len(junction):         # junction: the roots of the hair combed the other way (the fringe)
        from scipy.spatial import cKDTree
        dj = cKDTree(np.asarray(junction, float)).query(P[:, 0])[0] / MM
        near = SF.sm(JUNCTION_NEAR_MM[0], JUNCTION_NEAR_MM[1], dj)
        hj = np.repeat(JUNCTION_MM * near, m) * SF.sm(0.0, JUNCTION_OVER_MM, Sf)
        rep_j = int((near > 0.05).sum())
    Nn = shell.normals(U)
    tilt = np.zeros((n * m, 3))
    furrow = np.zeros(n * m)
    comb = np.full(n, -1)
    if tie_axis is not None:
        ax = np.asarray(tie_axis, float); ax /= np.linalg.norm(ax)
        e1 = np.cross(ax, [0.0, 1.0, 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        vt = P[np.arange(n), k_tie] - tie_c
        th = np.arctan2(vt @ e2, vt @ e1)                              # each hair's angle of arrival round the tail
        cuts = np.sort(rng.random(LOCKS)) * 2 * np.pi - np.pi          # uneven locks
        cuts = np.r_[cuts, cuts[0] + 2 * np.pi]
        j = np.searchsorted(cuts, th) - 1
        th2 = np.where(j < 0, th + 2 * np.pi, th); j = np.where(j < 0, LOCKS - 1, j)
        half = 0.5 * (cuts[j + 1] - cuts[j])                           # the lock's half width, rad
        x = (th2 - cuts[j]) / (2 * half) * 2 - 1                       # -1..1 across the lock
        comb = j
        step = LOCK_STEP_MM * (2 * rng.random(LOCKS) - 1)
        v = flat - tie_c
        rho = np.linalg.norm(v - np.outer(v @ ax, ax), axis=1) / MM    # mm from the tail's axis
        rho = np.maximum(rho, 15.0)
        hw = np.repeat(half, m) * rho                                  # the lock's half width here, mm
        amp = np.minimum(FURROW_MM, FURROW_SLOPE * hw / 2)             # (a parabola's steepest slope is 2 amp / hw)
        xr = np.repeat(x, m)
        grow = SF.sm(FURROW_FROM_MM[0], FURROW_FROM_MM[1], Sf)
        furrow = grow * (amp * (1 - xr * xr) - 0.5 * amp + np.repeat(step[j], m))
        slope = grow * amp * (-2 * xr) / hw                            # d height / d width, toward growing angle
        eth = np.cross(ax, v); eth /= np.maximum(np.linalg.norm(eth, axis=1, keepdims=True), 1e-12)
        eth -= (eth * Nn).sum(1, keepdims=True) * Nn
        eth /= np.maximum(np.linalg.norm(eth, axis=1, keepdims=True), 1e-12)
        tilt = -(slope * win)[:, None] * eth
    h = np.maximum(np.maximum(h + furrow, 0.3 * h), hj) * win
    bed = np.clip((rb - r) / MM, 0.0, BED_MAX_MM) * SF.sm(0.0, BED_OVER_MM, Sf) * np.minimum(1.0, win * 4)
    D = (U * (bed * MM)[:, None] + Nn * (h * MM)[:, None]).reshape(n, m, 3)
    for _ in range(4):                                 # (eased along the hair, the root held: no kinks where a window ends)
        D[:, 1:-1] = 0.25 * D[:, :-2] + 0.5 * D[:, 1:-1] + 0.25 * D[:, 2:]
    out = P + D
    Q = np.array(paths, float)
    Q[sel] = out
    T = np.zeros_like(Q); T[sel] = tilt.reshape(n, m, 3)
    C = np.full(len(Q), -1); C[sel] = comb
    Lay = np.ones(len(Q)); Lay[sel] = section
    hm = h.reshape(n, m).max(1)
    jump = np.abs(np.diff(np.linalg.norm(out - P, axis=2), axis=1)).max() / MM
    rep = dict(fibres=int(n), rate_mm_per_mm=round(RATE, 4), lift_mm_median_max=round(float(np.median(hm)), 1),
               lift_mm_p95=round(float(np.percentile(hm, 95)), 1), root_move_mm_max=round(float(np.linalg.norm(out[:, 0] - P[:, 0], axis=1).max() / MM), 3),
               node_jump_mm_max=round(float(jump), 2), tilt_p95=round(float(np.percentile(np.linalg.norm(tilt, axis=1), 95)), 2))
    rep['junction_fibres'] = rep_j
    rep['top_section_share'] = round(float(top.mean()), 2); rep['nape_section_share'] = round(float(nape.mean()), 2)
    return Q, T, C, Lay, rep
