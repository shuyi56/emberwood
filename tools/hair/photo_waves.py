"""Long layered waves with a centre part and a curtain fringe, grown from scratch from a reference photo's measurements
(John 2026-11-28: "Abandon this style and redo from scratch from reference photo"; the photo is
hair-rebuild/wavy-photo/reference.jpg). No guide of an older groom and no creator strand is read: every fibre is
combed here from its own root.

The photo, measured by hand on a 20 px grid (head height 240 px, skull top to chin; z in head heights above the chin,
x from the part). MEASURED, one photo, front view only, the head a little turned:
  part            on the midline, from the front hairline to the crown
  fringe          a curtain: thin, open at the part, its inner ends at the brow (z 0.54 at |x| 0.12), lengthening
                  outward to the cheekbone (z 0.40 at |x| 0.30), where it runs into the face-framing layers
  face framing    shorter pieces beside the face, ending between the chin and the collarbone (z 0.08 to -0.25)
  lengths         ends at z -0.76, layered (the top layers shorter), the front lengths in front of the shoulders
                  with their inner edge 0.2 from the midline on the chest
  volume          hair's half-width 0.41 at the top of the forehead, 0.45 at the eyes (the skull's is about 0.34),
                  0.54 at the mouth, 0.63 at the shoulders; 0.04 above the skull at the top
  wave            straight to the cheek (z about 0.38), then loose S waves about 0.33 long and 0.04 high, in ribbons
                  0.10 to 0.16 wide (22 to 35 mm: a stylist's 1 to 1.5 in section), the ends flicked and parted

How a fibre is combed (a head line; the lengths are then hung and waved by wavy_flow.hang_locks):
  - it starts at its root and runs over the scalp along gravity projected on the skin, pushed away from the part where
    the skin is flat (the top), and stands off the skin by a lift that grows along it (hair lies on the hair under
    it): this is the volume above;
  - where the skin turns under (below the skull's widest height) it leaves the head and hangs;
  - the fringe section (a triangle behind the front hairline, deepest at the part) is combed down and out over the
    forehead and cut on the curtain line; the hair just behind the hairline at the temples is cut as face framing.

    build(F, surf, n) -> paths (list of (k, 3) at SEG mm), hairs per fibre, lock key, kind (0 length, 1 fringe, 2 framing)
"""
import math
import numpy as np
from scipy.spatial import cKDTree

import scalp_field as SF

MM = SF.MM
SEG_MM = 2.5
N_HAIRS = 90000.0           # DRAFT: hairs on the scalp (the older grooms' budget; a scalp carries about 100,000, Robbins)
RELEASE_Z = SF.CENTRE[2]    # the skin turns under below the skull's centre height: the hair leaves the head there
PART_PUSH = 0.45            # DRAFT: the push away from the part against gravity along the skin (1: equal)
FRINGE_ARCH_MM = (12.0, 5.0, 7.0)   # DRAFT: the arch's height off the root, how far it comes in again over the forehead, and the
                            # flick at the end (a 35 mm brush's radius is 17.5)
FRINGE_PUSH = (0.45, 2.20)  # the same for the fringe, at the hairline and from the brow down: its inner edge runs from
                            # the part to the brow at |x| 0.12 and on out to the cheekbone at 0.30 (MEASURED)
VOLUME_MM = 30.0            # DRAFT: a hanging lock keeps its stand-off from the skin and body up to this (hair lies on the
                            # hair under it); wavy_flow.hang_locks' `volume`
LIFT_MM = (1.0, 28.0)       # MEASURED (photo): hair stands 9 mm off the skull at the top and about 24 mm at the eyes
LIFT_PER_MM = 0.12          # DRAFT: lift gained per mm combed
FRINGE_LIFT_MM = 12.0       # DRAFT: the fringe stands off the forehead (ghd: 'lift at the base'; the photo shows its shadow
                            # under it). At 5 mm it lay on the skin
FRINGE_HANG_Z = 0.66        # DRAFT: below the forehead's bulge the fringe hangs free of the skin
FRINGE_RAD = 0.70           # DRAFT: the fringe section's half-angle round the head
FRINGE_DEPTH = 0.10         # DRAFT (HH, 22 mm: a thin curtain; at 44 mm it hung as a dark slab over the eyes): its depth behind the hairline at the part, falling to 0 at FRINGE_RAD
FRAME_RAD, FRAME_DEPTH = 1.45, 0.12     # DRAFT: face framing: in front of the ears, this far behind the hairline
FRAME_END = (-0.25, 0.08)   # MEASURED: where the face-framing pieces end
HEM_Z = -0.76               # MEASURED: the ends
TOP_SHORTER, GRADE, RAGGED = 0.22, 0.25, 0.08   # DRAFT (HH): the top layer ends this much higher (layers), the pieces by the
                            # face up to this much higher again (Sam Villa: 'short in the front and long in the back'), and
                            # each hair up to this much short of its layer's line
TAKE = 1.3                  # CALIBRATED (first solve): the waves and the lie on the shoulder take up a quarter of the
                            # hanging length: ends cut for z -0.76 hung at -0.38 to -0.47; at 1.5 the locks splayed over the shoulders
FRINGE_FEATHER = 0.05       # DRAFT (HH): the fringe's ends are point cut over this much, not a ruled line
LOCKS_PER_SIDE = 50         # a lock is a stylist's section, 22 to 35 mm of the perimeter (MEASURED ribbon width)
WAVE = dict(start_z=0.38, full_z=0.12, lam_hh=0.33, amp_hh=0.04)     # MEASURED


def curtain_z(ax):
    """The fringe's cut line: where a fringe hair at |x| ends (MEASURED: brow at 0.12, cheekbone at 0.30)."""
    return np.clip(0.59 - (ax - 0.17) / 0.10 * 0.12, 0.47, 0.62)       # (re-MEASURED on the face crop: the inner ends at the
                                                                       #  brow's outer half, |x| 0.17, z 0.59; the outer at the
                                                                       #  temple, |x| 0.27, z 0.47, outside the eye's corner)


def wave(th, rng=None):
    """wavy_flow's wave by angle: (wavelength HH, phase, section). One phase per side (a brushed-out set is one wave
    pattern), the two sides mirrored; the caller jitters it per lock."""
    th = np.atleast_1d(th)
    jit = np.random.default_rng(4).normal(0.0, 0.5, len(th))             # DRAFT: a lock's own phase, about a twelfth of a wave
    return np.full(len(th), WAVE['lam_hh']), np.where(th > 0, 0.6, 0.0) + jit, np.zeros(len(th), int)


def build(F, surf, n=24000, seed=11, log=print):
    rng = np.random.default_rng(seed)
    per = SF.point_area(F)
    p = F['density'] * per
    p = p / p.sum()
    idx = rng.choice(len(p), size=n, p=p)
    root = F['P'][idx] + (rng.random((n, 3)) - 0.5) * 0.004
    hairs = np.full(n, N_HAIRS / n)
    th = SF.theta(root)
    # the front hairline: the lowest scalp point in each slice of angle, in front of the ears
    scalp = F['P'][F['density'] > 0]
    ths = SF.theta(scalp)
    edge = []
    for a in np.linspace(-1.9, 1.9, 96):
        m = np.abs(ths - a) < 0.03
        if m.any():
            edge.append(scalp[m][np.argmin(scalp[m][:, 2])])
    d_hl = cKDTree(np.array(edge)).query(root)[0]
    kind = np.zeros(n, int)
    fr = (np.abs(th) < FRINGE_RAD) & (d_hl < FRINGE_DEPTH * (1 - (np.abs(th) / FRINGE_RAD) ** 2))
    kind[fr] = 1
    kind[(kind == 0) & (np.abs(th) < FRAME_RAD) & (d_hl < FRAME_DEPTH)] = 2
    high = root[:, 2] > np.median(root[(kind == 0) & (root[:, 2] > RELEASE_Z), 2])     # the top layer: rooted above the under layer
    z_end = HEM_Z + TOP_SHORTER * high + RAGGED * rng.random(n)
    z_end[kind == 2] = rng.uniform(*FRAME_END, (kind == 2).sum())
    z_end = RELEASE_Z - (RELEASE_Z - z_end) * TAKE
    feather = FRINGE_FEATHER * rng.random(n)
    z_end[kind == 1] = -9.0                                            # (cut on the curtain line below)
    layer = np.where(high, 0.6, 0.2) + 0.4 * rng.random(n)             # (where in the hair's thickness the fibre lies)
    hand = np.full(n, -1)                                              # (the node where a fibre leaves the head)
    lift_cap = np.where(kind == 1, FRINGE_LIFT_MM, LIFT_MM[1]) * layer + LIFT_MM[0]
    side = np.where(root[:, 0] >= 0, 1.0, -1.0)
    crown_y = SF.CENTRE[1] + 0.10                                      # DRAFT: the part ends at the crown
    pos = root.copy()
    live = np.ones(n, bool); free = np.zeros(n, bool)
    out = [[r] for r in root]
    step = SEG_MM * MM
    g = np.array([0.0, 0.0, -1.0])
    for it in range(int(3.4 / step)):
        I = np.where(live)[0]
        if not len(I):
            break
        P = pos[I]
        q = P - SF.CENTRE
        r = np.linalg.norm(q, axis=1)
        U = q / r[:, None]
        on = ~free[I]
        new = P + step * g                                              # hanging: straight down
        if on.any():
            J = I[on]; Uo = U[on]
            # along the skin: gravity, and the push away from the part (sideways in front of the crown, outward from
            # the crown behind it), both laid on the skin (the radius from the skull's centre stands for its normal)
            b = np.zeros((len(J), 3)); b[:, 0] = side[J]
            behind = pos[J, 1] > crown_y
            rb = pos[J][:, :2] - np.array([0.0, crown_y]); rb /= np.maximum(np.linalg.norm(rb, axis=1, keepdims=True), 1e-9)
            b[behind, 0] = rb[behind, 0]; b[behind, 1] = rb[behind, 1]
            push = np.where(kind[J] == 1, FRINGE_PUSH[0] + (FRINGE_PUSH[1] - FRINGE_PUSH[0]) * SF.sm(0.80, 0.56, pos[J, 2]), PART_PUSH)
            d = g[None] + push[:, None] * b
            d[kind[J] == 1, 1] -= 1.0                                   # (the fringe is combed forward over the hairline)
            d = d - (d * Uo).sum(1, keepdims=True) * Uo
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            Q = pos[J] + step * d
            qq = Q - SF.CENTRE
            Un = qq / np.linalg.norm(qq, axis=1, keepdims=True)
            s_mm = it * SEG_MM
            lift = np.minimum(LIFT_MM[0] + LIFT_PER_MM * s_mm * layer[J], lift_cap[J]) * MM
            # the fringe bends in two planes (John 2026-11-28: "bangs should also be 3d twisting along 2 planes like the
            # picture"). Out of the forehead: an arch up off the root over the hairline (ghd: "lift at the base", on a
            # 35 mm brush), in again over the forehead, and the end flicked out ("flick the brush away from the face").
            # Round the head: down first, then out to the temple (the push below). The two together are a twisting S.
            fz = Q[:, 2] - (curtain_z(np.abs(Q[:, 0])) + feather[J])
            lf = (1.0 + FRINGE_ARCH_MM[0] * SF.sm(0.0, 25.0, s_mm) - FRINGE_ARCH_MM[1] * SF.sm(30.0, 75.0, s_mm)
                  + FRINGE_ARCH_MM[2] * SF.sm(0.07, 0.0, fz)) * (0.8 + 0.4 * layer[J]) * MM
            lift = np.where(kind[J] == 1, lf, lift)
            Qs = SF.CENTRE + Un * (surf.radius(Un) + lift)[:, None]
            # (below the forehead the fringe hangs as a curtain over the brow and the eye's corner: it does not follow
            # the skin into the socket, it only stays clear of it)
            hangs = (kind[J] == 1) & (pos[J, 2] < FRINGE_HANG_Z)
            r_was = np.linalg.norm(pos[J] - SF.CENTRE, axis=1)
            Qh = SF.CENTRE + Un * np.maximum(r_was - 0.2 * step, surf.radius(Un) + lift)[:, None]
            Q = np.where(hangs[:, None], Qh, Qs)
            new[on] = Q
            rel = J[(Q[:, 2] < RELEASE_Z) & (kind[J] != 1)]                 # (the fringe lies on the forehead to its cut)
            free[rel] = True
            for i in rel:
                hand[i] = len(out[i])
            lg = rel[kind[rel] == 0]                                        # (graded shorter toward the face)
            z_end[lg] += GRADE * TAKE * SF.sm(1.4, 0.5, np.abs(SF.theta(pos[lg])))
        pos[I] = new
        for i, x in zip(I, new):
            out[i].append(x)
        ax = np.abs(new[:, 0])
        done = (new[:, 2] <= z_end[I]) | ((kind[I] == 1) & (new[:, 2] <= curtain_z(ax) + feather[I]))
        live[I[done]] = False
    paths = [np.array(o) for o in out]
    # the locks: the fibres that hang past the wave's start, gathered by where they pass it, each side on its own
    lockkey = np.full(n, -1)
    H = np.full((n, 3), np.nan)
    for i, pth in enumerate(paths):
        j = np.where(pth[:, 2] < WAVE['start_z'])[0]
        if len(j) and (len(pth) - j[0]) * SEG_MM > 40.0:
            H[i] = pth[j[0]]
    from scipy.cluster.vq import kmeans2
    nk = 0
    for sg in (1, -1):
        m = np.where(~np.isnan(H[:, 0]) & (np.sign(H[:, 0]) == sg))[0]
        _, lab = kmeans2(H[m], LOCKS_PER_SIDE, seed=3 + sg, minit='++')
        lockkey[m] = nk + lab; nk += LOCKS_PER_SIDE
    log('photo waves: fibres %d (fringe %d, framing %d), hanging %d in %d locks; lengths mm median %.0f'
        % (n, (kind == 1).sum(), (kind == 2).sum(), (lockkey >= 0).sum(), nk, np.median([len(p_) for p_ in paths]) * SEG_MM))
    return paths, hairs, lockkey, kind, dict(hand=hand, high=high, edge=np.array(edge), root=root, layer=layer, z_end=z_end)


def aim(coll):
    """Where a front lock is headed (wavy_flow.hang_locks' `creator` hook as a function of the lock's root and angle):
    in front of the shoulder onto the chest, the locks nearest the face innermost (MEASURED: inner edge 0.2 from the
    midline, outer 0.6, at z -0.15), 8 mm clear of the body. Locks behind the ears hang as they fall."""
    def f(root, th):
        if abs(th) > 1.75:
            return None
        x = np.sign(th) * (0.22 + 0.36 * abs(th) / 1.75)
        z = -0.15
        y = -0.8
        for _ in range(200):                                            # (in from the front until 8 mm off the body)
            d, _n = coll(np.array([[x, y, z]]) / MM, np.array([8.0]))
            if d[0] > 0:
                break
            y += 0.005
        return np.array([x, y - 0.005, z]), None
    return f


# ---------------------------------------------------------------------------------------------------------------
# The flow field (John 2026-11-28: "the hair should be 3d it should bend and twist so it falls on the front while
# framing the sides and providing depth ... look at the pony tail code for how to do proper locks for falling hair ...
# first show me the flow field before anything"). The ponytail's recipe (ponytail_flow.py, bangs_creator_asset.py
# tail_locks, solvers/hair/tail_lock_rods.py) carried over to loose hair:
#   ponytail: one basin, every root funnels over the scalp to ONE outlet (the tie), is handed over onto the bundle,
#             and below it the hair is 36 thick locks, each its own 3D rod, a ribbon that rolls and opens.
#   here:     the scalp is parted into SECTIONS as a stylist takes them (rows from the bottom up, each row in sections
#             25 to 38 mm wide, the upper row's sections laid over the joints of the lower row's: bricklay). Each
#             section is a basin with its own outlet on the line where the hair leaves the head, and below its outlet
#             it is one thick lock: a 3D rod with its own curl, its own turn, kept a lock's thickness off its neighbours.
# Sources: Sam Villa (sections 1 to 1.5 in; "short in the front and long in the back"; the face frame "flowing away
# from the face"), Red Seal study guide (bricklay), ghd (a curtain fringe: lift at the root on a 35 mm round brush,
# "flick the brush away from the face"; the pieces beside it curled "away from the face"), Wella (the next section
# curled the other way: "continue this alternating pattern"; the tong held vertical, starting below the root),
# 80.lv Brown 2025 (layers with space left between them "so shadows create depth"), Miller et al. 2014 PRL 112 068103
# (a naturally curved hair under its weight: planar below w/(B k^3) = 0.239... at this curl, 0.15, it is a 3D helix).
SECTORS = 9                 # sections a side in a row: 31 mm of the perimeter each (SOURCED: 25 to 38 mm)
GATHER = 0.75               # DRAFT: a section's hair at its outlet takes this share of the section's own width
OUT_LIFT_MM = {0: 5.0, 1: 17.0, 2: 9.0, 3: 6.0}   # DRAFT: an outlet's stand-off by row: under, top (it lies on the under
                            # row: the space between the layers), face framing, fringe
CURL_ON_MM = (20.0, 110.0)  # MEASURED: straight to the cheek, the full wave by the jaw
FRINGE_K = (1 / 70.0, 1 / 35.0)   # per mm, root to tip. The tip's is a 35 mm brush's twice over (SOURCED ghd; the set relaxes: DRAFT)
PACKING = 0.1               # the ponytail's TAIL_LOCK_PACKING
LAYER_NAME = {0: 'under row', 1: 'top row', 2: 'face framing', 3: 'fringe'}


def field(F, surf, n=24000, seed=11, log=print):
    """Sections, outlets and the lines over the scalp. Returns a dict: root, hairs, kind, basin (per fibre, -1: none),
    line (list of (k, 3): root to outlet over the skin), hang_mm (the hair's length below its outlet), u (its place
    across the outlet, mm), and per basin: side, layer, sector, centre, th (angle), half_mm (the outlet's half-width)."""
    paths, hairs, _, kind, x = build(F, surf, n, seed, log=lambda t: None)
    root, hand, high = x['root'], x['hand'].copy(), x['high']
    # the fringe leaves the head at the hairline
    eth = SF.theta(x['edge']); o = np.argsort(eth)
    for i in np.where(kind == 1)[0]:
        p = paths[i]
        below = np.where(p[:, 2] < np.interp(SF.theta(p), eth[o], x['edge'][o, 2]) - 0.01)[0]
        hand[i] = below[0] if len(below) else -1
    ok = np.array([hand[i] >= 0 and (len(paths[i]) - hand[i]) * SEG_MM > 30.0 for i in range(n)])
    rel = np.array([paths[i][min(hand[i], len(paths[i]) - 1)] if ok[i] else root[i] for i in range(n)])
    th = SF.theta(rel)
    side = np.where(th >= 0, 1, 0)
    layer = np.where(kind == 1, 3, np.where(kind == 2, 2, np.where(high, 1, 0)))
    u = np.abs(th) / (np.pi / SECTORS)
    sec = np.clip(np.where(layer == 1, np.floor(u + 0.5), np.floor(u)), 0, SECTORS).astype(int)   # (the top row bricklaid)
    for lay in (2, 3):                                                   # (two pieces a side: inner and outer)
        for sd in (0, 1):
            m = ok & (layer == lay) & (side == sd)
            if m.any():
                sec[m] = (np.abs(th[m]) > np.median(np.abs(th[m]))).astype(int)
    key = np.where(ok, side * 1000 + layer * 100 + sec, -1)
    keys = np.unique(key[key >= 0])
    basin = np.full(n, -1)
    B = dict(side=[], layer=[], sector=[], centre=[], th=[], half_mm=[], hairs=[])
    tgt = rel.copy()
    upos = np.zeros(n)
    for b, k in enumerate(keys):
        m = np.where(key == k)[0]
        basin[m] = b
        thc = float(np.median(th[m]))
        lay = int(layer[m[0]])
        # each hair's place on the outlet: its own angle drawn in to the section's middle (rank kept), its own height
        tt = thc + GATHER * (th[m] - thc)
        r = np.hypot(rel[m, 0] - SF.AXIS[0], rel[m, 1] - SF.AXIS[1])
        q = np.c_[SF.AXIS[0] + r * np.sin(tt), SF.AXIS[1] - r * np.cos(tt), rel[m, 2]]
        U = q - SF.CENTRE; U /= np.linalg.norm(U, axis=1, keepdims=True)
        depth = OUT_LIFT_MM[lay] + 3.0 * (np.random.default_rng(int(k)).random(len(m)) - 0.5)
        tgt[m] = SF.CENTRE + U * (surf.radius(U) + depth * MM)[:, None]
        rm = float(np.median(r))
        upos[m] = (tt - thc) * rm / MM
        B['side'].append(int(side[m[0]])); B['layer'].append(lay); B['sector'].append(int(sec[m[0]]))
        B['centre'].append(tgt[m].mean(0)); B['th'].append(thc)
        B['half_mm'].append(float(np.percentile(np.abs(upos[m]), 90))); B['hairs'].append(float(hairs[m].sum()))
    # the lines over the scalp: the shortest way over the skull from the root to its place on the outlet (the
    # ponytail's funnel: arcs to one outlet do not cross), lifting to the outlet's stand-off
    lines, hang = [], np.zeros(n)
    for i in range(n):
        if basin[i] < 0:
            lines.append(paths[i]); continue
        U0 = root[i] - SF.CENTRE; r0 = np.linalg.norm(U0); U0 /= r0
        U1 = tgt[i] - SF.CENTRE; r1 = np.linalg.norm(U1); U1 /= r1
        om = math.acos(float(np.clip(U0 @ U1, -1, 1)))
        k = max(2, int(om * 0.5 * (r0 + r1) / (SEG_MM * MM)) + 1)
        t = np.linspace(0, 1, k)
        if om < 1e-4:
            Us = np.repeat(U0[None], k, 0)
        else:
            Us = (np.sin((1 - t) * om)[:, None] * U0 + np.sin(t * om)[:, None] * U1) / math.sin(om)
        e = t * t * (3 - 2 * t)
        skin = surf.radius(Us)
        rr = np.maximum(skin + 1.0 * MM, r0 + (r1 - r0) * e)             # (never under the skin on the way)
        lines.append(SF.CENTRE + Us * rr[:, None])
        hang[i] = (len(paths[i]) - hand[i]) * SEG_MM
    for k_ in B:
        B[k_] = np.array(B[k_])
    log('field: %d fibres, %d sections (%s); hairs per section median %.0f; outlet half-width median %.0f mm'
        % (n, len(keys), ', '.join('%s %d' % (LAYER_NAME[l], (B['layer'] == l).sum()) for l in range(4)),
           np.median(B['hairs']), np.median(B['half_mm'])))
    return dict(root=root, hairs=hairs, kind=kind, basin=basin, line=lines, hang_mm=hang, u=upos, tgt=tgt, B=B)


def locks(Fd, coll, log=print, seed=9, gain=1.0):
    """Each section's lock below its outlet as a 3D rod (tail_lock_rods.solve): its launch, its natural curl (a helix
    for the lengths, turning the other way in the next section; a C away from the face for the fringe), gravity,
    contact with the other locks and with the head and body. Returns the rods (x HH, R, s mm), each lock's ribbon turn
    psi(s) and the report."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'solvers', 'hair'))
    import tail_lock_rods as TR
    rng = np.random.default_rng(seed)
    B = Fd['B']; nl = len(B['th'])
    side = np.where(B['side'] > 0, 1.0, -1.0)
    roots = B['centre'].copy()
    length = np.array([np.percentile(Fd['hang_mm'][Fd['basin'] == k], 98) for k in range(nl)])
    nh = B['hairs']
    aimf = aim(coll)
    tang = np.zeros((nl, 3)); d1 = np.zeros((nl, 3))
    out_h = np.c_[roots[:, 0] - SF.AXIS[0], roots[:, 1] - SF.AXIS[1], np.zeros(nl)]
    out_h /= np.linalg.norm(out_h, axis=1, keepdims=True)
    for k in range(nl):
        if B['layer'][k] == 3:        # the fringe: lifted forward off the hairline, then down and out
            tv = np.array([0.45 * side[k], -0.6, -0.65]); ref = np.array([0.0, 0.0, 1.0])
        else:
            tv = np.array([0.0, 0.0, -1.0]) + 0.15 * out_h[k]
            a_ = aimf(roots[k], B['th'][k]) if abs(B['th'][k]) < 1.2 else None
            if a_ is not None:        # in front of the ears: headed for the chest in front of the shoulder
                to = a_[0] - roots[k]
                tv = 0.5 * tv / np.linalg.norm(tv) + 0.5 * to / np.linalg.norm(to)
            ref = out_h[k]
        tang[k] = tv / np.linalg.norm(tv)
        d = ref - (ref @ tang[k]) * tang[k]
        d1[k] = d / np.linalg.norm(d)
    d2 = np.cross(tang, d1)
    frames = np.stack([d1, d2, tang], 2)
    lock_r = np.sqrt(nh * 0.035 ** 2 / PACKING)
    A = WAVE['amp_hh'] / MM; c = WAVE['lam_hh'] / MM / (2 * np.pi)
    fringe = B['layer'] == 3
    hand = np.where(B['sector'] % 2 == 0, 1.0, -1.0) * side * np.where(B['layer'] == 1, -1.0, 1.0)   # (SOURCED Wella: alternating;
    tau = np.where(fringe, 0.0, hand / math.sqrt(A * A + c * c))                                     #  mirrored left and right)
    psi0 = np.pi / 2 + np.where(fringe, 0.0, 0.4 * rng.standard_normal(nl))    # (the first bend is out from the head)
    phi = np.where(fringe, np.where(side > 0, 0.0, np.pi), psi0)
    Lb3 = TR.EI_HAIR / TR.W_HAIR

    def rest(s):
        grow = SF.sm(CURL_ON_MM[0], CURL_ON_MM[1], s)
        h = np.maximum(length[:, None] - s, 0.0)
        pre = 1.0 + h / (Lb3 * (2 * np.pi / (WAVE['lam_hh'] / MM)) ** 2)   # (wavy_flow's preload against the weight below)
        kh = gain * A / (A * A + c * c) * grow * pre
        kf = FRINGE_K[0] + (FRINGE_K[1] - FRINGE_K[0]) * np.clip(s / length[:, None], 0, 1)
        return np.where(fringe[:, None], kf, kh), tau, phi
    out = TR.solve(roots / MM, frames, length, nh, lock_r, head=coll, seed=seed, rest=rest, log=lambda t: log('  rods: ' + t))
    rep = dict(out['report'], lock_radius_mm_median=round(float(np.median(lock_r)), 1), length_mm_median=round(float(np.median(length))))
    log('locks %s' % rep)
    return dict(X=out['x'] * MM, R=out['R'], s=out['s'], tau=tau, psi0=psi0, length=length, lock_r=lock_r, report=rep)


COLS = {0: [(47, 111, 179), (29, 74, 128), (111, 163, 216)], 1: [(44, 156, 90), (122, 79, 201), (20, 120, 110)],
        2: [(194, 51, 122), (150, 30, 90)], 3: [(217, 138, 31), (180, 100, 10)]}


def draw(Fd, rods, head_pts, depth, path, views=((0, 4, 'front'), (40, 6, 'three quarter'), (90, 4, 'side'), (180, 4, 'back'), (0, 80, 'top')),
         every=6, px=330):
    """The field as lines over a clay head (the ponytail's sheet): each section's lines over the scalp in its colour,
    its outlet in red, and its lock below the outlet as a ribbon (its middle line and its two edges, so the turn shows)."""
    from PIL import Image, ImageDraw
    W, H = int(1.9 * px), int(2.5 * px)
    im = Image.new('RGB', (W * len(views), H + 40), (250, 249, 247))
    dr = ImageDraw.Draw(im)
    B = Fd['B']
    pc = head_pts[head_pts[:, 2] > -0.9]
    for v, (az, el, name) in enumerate(views):
        a, e = math.radians(az), math.radians(el)
        fwd = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)

        def P(X):
            return np.c_[v * W + W / 2 + (X @ right) * px, 40 + H * 0.36 - (X @ up - 0.5) * px]

        def hid(X):
            t = np.linspace(0.015, 1.3, 24)[None, :, None]
            Q = (X[:, None] + fwd[None, None] * t).reshape(-1, 3)
            return (depth(Q) > 0.003).reshape(len(X), -1).any(1)
        nrm = pc - SF.CENTRE; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        sh = np.clip(0.74 + 0.26 * (nrm @ (fwd + 0.5 * up - 0.4 * right) / 1.2), 0.5, 1.0)
        vis = ~hid(pc + nrm * 0.004)
        for (x, y), s_ in zip(P(pc)[vis], sh[vis]):
            g = int(225 * s_); dr.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=(g, int(g * 0.98), int(g * 0.95)))
        items = []

        def add(X, col, w):
            h = hid(X); q = P(X); z = X @ fwd
            for j in range(len(X) - 1):
                if not (h[j] or h[j + 1]):
                    items.append((0.5 * (z[j] + z[j + 1]), tuple(q[j]), tuple(q[j + 1]), col, w))
        for i in range(0, len(Fd['line']), every):
            b = Fd['basin'][i]
            if b < 0:
                add(Fd['line'][i][::2], (150, 150, 150), 1); continue
            cs = COLS[int(B['layer'][b])]
            add(Fd['line'][i][::2] if len(Fd['line'][i]) > 4 else Fd['line'][i], cs[int(B['sector'][b]) % len(cs)], 1)
        for k in range(len(B['th'])):
            cs = COLS[int(B['layer'][k])]; col = cs[int(B['sector'][k]) % len(cs)]
            X, R, s = rods['X'][k], rods['R'][k], rods['s'][k]
            M = R.shape[0]
            last = int(min(M, math.ceil(rods['length'][k] / max(s[1], 1e-6))))
            psi = rods['psi0'][k] + rods['tau'][k] * s[:M]
            ew = np.cos(psi)[:, None] * R[:, :, 0] + np.sin(psi)[:, None] * R[:, :, 1]
            hw = B['half_mm'][k] * MM * (1 - 0.5 * (s[:M] / max(s[M], 1.0)) ** 2)[:, None]     # (drawing in to a point)
            add(X[:last + 1], col, 3)
            add((X[:M] + ew * hw)[:last], col, 1); add((X[:M] - ew * hw)[:last], col, 1)
            for j in range(0, last, 4):                                     # (rungs: the ribbon's width, turning)
                add(np.stack([X[j] - ew[j] * hw[j], X[j] + ew[j] * hw[j]]), col, 1)
            o = Fd['tgt'][Fd['basin'] == k]
            o = o[np.argsort(Fd['u'][Fd['basin'] == k])][::max(1, len(o) // 12)]
            add(o, (224, 32, 32), 3)
        for z, p0, p1, col, w in sorted(items, key=lambda t: t[0]):
            dr.line([p0, p1], fill=col, width=w)
        dr.text((v * W + 12, 12), name, fill=(0, 0, 0))
    im.save(path)


# ---------------------------------------------------------------------------------------------------------------
# Field v2 (John 2026-11-28 on v1: "read the recipe and do it again the hair is falling all over the place"). The
# recipe (docs/hair/ponytail-recipe.md): one sink per basin, strands keep rank, no line turns back, the fall is ONE
# shape with the S wave "laid on the middle line" so that it "moves every strand alike"; locks come later, inside it.
# v1 gave each of 36 sections its own free rod. Here the hair below the head is two falls a side, built so the rules
# hold by construction:
#   front fall  hair that leaves the head in front of FALL_SPLIT: it frames the face to the jaw, closes over the side
#               of the neck and lies on the chest in front of the shoulder (MEASURED: inner edge 0.2, outer 0.6)
#   back fall   the rest: over the back of the shoulder and down the back
# A line is (angle round the body, height): its angle moves from where it leaves the head to its place in the fall by
# a map that is monotonic in the angle (rank kept), its radius is the tent from the head to the body and then the body
# itself plus its depth in the stack, and the S wave is one pattern a side whose phase changes slowly round the head.
FALL_SPLIT = 1.75           # DRAFT: |theta| behind the ear where the hair parts over the shoulder
FRONT_BAND = (0.42, 0.85)   # MEASURED: the front fall on the chest (x 0.2 to 0.6 from the midline at the collarbone)
BACK_BAND = (2.30, np.pi)   # DRAFT: the back fall, down the back inside the shoulder blades (at 1.95 it draped over the
                            # shoulders' tops)
FAIR_MM = 25.0              # DRAFT: the course is faired over this length (half a bending length)
GATHER_Z = (0.30, -0.45)    # DRAFT: the falls take their places between the jaw and the collarbone
FACE_KEEP = 1.15            # MEASURED (the photo's face crop): beside the face the hair's front edge is at the face's own
                            # outline, back by the ear (an earring shows): no hair lies on the cheek. The angle round the
                            # head of the cheekbone's back edge, just in front of the ear (the ear's middle is at 1.57)
FACE_KEEP_Z = (0.56, 0.20)  # the front edge swings back to it between the temple and the cheekbone (ghd, Sam Villa: the
                            # pieces by the face are dried and cut 'flowing away from the face')
STACK_MM = (2.0, 18.0)      # DRAFT: the fall's thickness: the innermost hair's stand-off and the outermost's
BODY_AT_Z = -0.15           # where the tent from the head lands (the photo's shoulder line)


BANG_END = ((0.17, 0.59), (0.27, 0.47))   # MEASURED (face crop): where the bang's inner hair ends (|x|, z: the brow's outer
                            # half) and its outer hair (the temple, outside the eye's corner)
BANG_S = (0.50, 0.40)       # DRAFT: the S's two handles, as shares of the hair's drop: it leaves the root heading down and
                            # arrives heading down, and between the two it runs out to the side (the photo's S)


def bang_sheet(root, surf, seed=3, k=44):
    """One side's bang as ONE smooth surface (John 2026-11-28: "build the bangs as one smooth shape that produces that s
    curve in the picture"). The surface is S(u, t): u across it (0 the hair nearest the part, 1 the outermost, by the
    roots' order, so the order is kept) and t along the hair. Seen from the front every hair is a cubic that leaves its
    root heading down, runs out to the side and arrives at its end heading down again: an S. Its ends lie on the measured
    line from the brow to the temple. Out of the face it follows the skin at a stand-off that is the same smooth
    function of t for every hair: the arch off the root, in over the forehead, the flick at the end (FRINGE_ARCH_MM).
    root (m, 3), all on one side. Returns a list of (k, 3)."""
    rng = np.random.default_rng(seed)
    m = len(root)
    sgn = 1.0 if np.median(root[:, 0]) >= 0 else -1.0
    u = np.argsort(np.argsort(np.abs(root[:, 0]))) / max(m - 1, 1)
    x3 = sgn * (BANG_END[0][0] + (BANG_END[1][0] - BANG_END[0][0]) * u)
    z3 = BANG_END[0][1] + (BANG_END[1][1] - BANG_END[0][1]) * u
    t = np.linspace(0, 1, k)[None, :]
    drop = (root[:, 2] - z3)[:, None]
    bx = [root[:, 0:1], root[:, 0:1], x3[:, None], x3[:, None]]
    bz = [root[:, 2:3], root[:, 2:3] - BANG_S[0] * drop, z3[:, None] + BANG_S[1] * drop, z3[:, None]]
    w = [(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3]
    X = sum(a * b for a, b in zip(w, bx)); Z = sum(a * b for a, b in zip(w, bz))
    # the skin in front at each (x, z): bisection on y
    lo = np.full(X.shape, -0.75); hi = np.full(X.shape, SF.CENTRE[1])
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        q = np.stack([X, mid, Z], -1).reshape(-1, 3) - SF.CENTRE
        r = np.linalg.norm(q, axis=1)
        inside = (r < surf.radius(q / r[:, None])).reshape(X.shape)
        hi = np.where(inside, mid, hi); lo = np.where(inside, lo, mid)
    P = np.stack([X, 0.5 * (lo + hi), Z], -1)
    # (the stand-off is measured from a skin eased along the hair, so the sheet bridges the brow and does not dip)
    from scipy.ndimage import gaussian_filter1d
    q = P - SF.CENTRE
    from scipy.ndimage import maximum_filter1d
    r = gaussian_filter1d(maximum_filter1d(np.linalg.norm(q, axis=2), 9, axis=1, mode='nearest'), 4.0, axis=1, mode='nearest')
    U = q / np.linalg.norm(q, axis=2, keepdims=True)
    tt = t[0]
    lift = (1.0 + FRINGE_ARCH_MM[0] * SF.sm(0.0, 0.40, tt) - FRINGE_ARCH_MM[1] * SF.sm(0.35, 0.85, tt)
            + FRINGE_ARCH_MM[2] * SF.sm(0.65, 1.0, tt))[None, :] * MM
    lift = lift + (3.0 * rng.random(m))[:, None] * MM * SF.sm(0.0, 0.3, tt)[None, :]     # (its thickness: 3 mm)
    P = SF.CENTRE + U * (r + lift)[:, :, None]
    cut = (k * (1 - FRINGE_FEATHER * 2 * rng.random(m))).astype(int)                     # (point cut)
    return [P[i, :max(cut[i], 4)] for i in range(m)]


def body_radius(depth, nth=128, zs=None):
    """The head's and body's radius from the vertical axis by angle and height (the outermost solid along each ray),
    eased round the body so that hair bridges hollows. Returns (th, z, R)."""
    from scipy.ndimage import maximum_filter1d, gaussian_filter
    zs = np.arange(0.62, -1.0, -0.02) if zs is None else zs
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    rr = np.arange(0.95, 0.02, -0.005)
    T, Z, Rr = np.meshgrid(th, zs, rr, indexing='ij')
    X = np.stack([SF.AXIS[0] + Rr * np.sin(T), SF.AXIS[1] - Rr * np.cos(T), Z], -1).reshape(-1, 3)
    ins = (depth(X) > 0).reshape(T.shape)
    first = np.argmax(ins, axis=2)
    R = np.where(ins.any(2), rr[first], 0.05)
    R = maximum_filter1d(R, 5, axis=0, mode='wrap')
    R = gaussian_filter(R, (2.0, 1.0), mode=('wrap', 'nearest'))
    return th, zs, R


def _join(p, j, half=14, sig=4.0):
    """The line faired where it leaves the head (node j): the nodes within `half` of it are drawn to their smoothed
    places, most at j, none at the window's ends, so the scalp line runs into the fall with no corner."""
    from scipy.ndimage import gaussian_filter1d
    a, b = max(j - half, 1), min(j + half, len(p) - 1)
    if b - a < 4:
        return p
    sm_ = gaussian_filter1d(p, sig, axis=0, mode='nearest')
    w = np.clip(1 - np.abs(np.arange(a, b) - j) / half, 0, 1)[:, None]
    q = p.copy(); q[a:b] = p[a:b] + w * w * (3 - 2 * w) * (sm_[a:b] - p[a:b])
    return q


def field2(F, surf, depth, n=24000, seed=11, log=print):
    """Returns dict(line: list of (k, 3), basin: per line (0 fringe, 1 face framing, 2 front fall, 3 back fall, -1 lies
    on the head), hairs, root, sec (the line's section, for colour), release (per line: node where it leaves the head),
    gauges)."""
    from scipy.interpolate import RegularGridInterpolator
    paths, hairs, _, kind, x = build(F, surf, n, seed, log=lambda t: None)
    hand, layer, z_end = x['hand'], x['layer'], x['z_end']
    bt, bz, bR = body_radius(depth)
    Rb = RegularGridInterpolator((np.r_[bt, np.pi], bz[::-1]), np.c_[bR.T, bR[0]].T[:, ::-1], bounds_error=False, fill_value=None)
    hang = np.where((hand >= 0) & (kind != 1))[0]
    rel = np.array([paths[i][min(hand[i], len(paths[i]) - 1)] for i in hang])
    th0 = SF.theta(rel); a0 = np.abs(th0); sg = np.where(th0 >= 0, 1.0, -1.0)
    r0 = np.hypot(rel[:, 0] - SF.AXIS[0], rel[:, 1] - SF.AXIS[1])
    lo = float(np.percentile(a0, 1))
    front = a0 < FALL_SPLIT
    # each line's place in its fall: linear in its angle, so the order round the head is the order in the fall
    a1 = np.where(front, FRONT_BAND[0] + (FRONT_BAND[1] - FRONT_BAND[0]) * np.clip((a0 - lo) / (FALL_SPLIT - lo), 0, 1),
                  BACK_BAND[0] + (BACK_BAND[1] - BACK_BAND[0]) * (a0 - FALL_SPLIT) / (np.pi - FALL_SPLIT))
    d = np.clip((layer[hang] - 0.2) / 0.8, 0, 1)
    stack = (STACK_MM[0] + (STACK_MM[1] - STACK_MM[0]) * d) * MM
    land = Rb(np.c_[np.clip(sg * a1, -np.pi, np.pi), np.full(len(hang), BODY_AT_Z)]) + stack
    lam = WAVE['lam_hh']
    phase = 0.5 * np.sin(2 * a0) + np.where(sg > 0, 0.6, 0.0)           # (one pattern a side, drifting round the head)
    from scipy.ndimage import gaussian_filter1d
    lines = [p for p in paths]
    base = {}
    basin = np.where(kind == 1, 0, -1)
    for sd in (1.0, -1.0):                                              # (each side's bang: one smooth sheet)
        fm = np.where((kind == 1) & (x['root'][:, 0] * sd > 0))[0]
        if len(fm) > 3:
            for i, l in zip(fm, bang_sheet(x['root'][fm], surf, seed=3 + int(sd))):
                lines[i] = l
    sec = np.zeros(n, int)
    step = SEG_MM * MM
    for q, i in enumerate(hang):
        z = np.arange(rel[q, 2], RELEASE_Z - (RELEASE_Z - z_end[i]) / TAKE, -step)   # (cut by height here: no length is taken up)
        if len(z) < 3:
            continue
        g = SF.sm(GATHER_Z[0], GATHER_Z[1], z)
        ah = a0[q]
        if front[q]:                                                     # (beside the face: the whole front fall drawn back
            keep = lo + (max(FACE_KEEP, lo) - lo) * SF.sm(FACE_KEEP_Z[0], FACE_KEEP_Z[1], z)   # behind the cheek, order kept)
            ah = keep + (a0[q] - lo) * (FALL_SPLIT - keep) / (FALL_SPLIT - lo)
        a = ah + (a1[q] - ah) * g
        t = np.clip((rel[q, 2] - z) / max(rel[q, 2] - BODY_AT_Z, 1e-6), 0, 1)
        tent = r0[q] + (land[q] - r0[q]) * t
        floor = Rb(np.c_[sg[q] * a, z]) + stack[q]
        over0 = max(float(floor[0] - r0[q]), 0.0)                        # (the floor meets the head where the hair leaves it)
        R = np.maximum(floor - over0 * np.exp(-np.arange(len(z)) / (FAIR_MM / SEG_MM)), tent)
        # one eased course (John 2026-11-28: "needs to be smoother"): the eases above meet in corners (behind the cheek,
        # at the collarbone), so the course is faired over FAIR_MM along the hair, held at the head end, and kept off
        # the body. gauges/hair/flow_smooth.py reads the bend radius of this course against a hair's bending length.
        nf = FAIR_MM / SEG_MM
        fade = np.exp(-np.arange(len(z)) / nf)
        for _ in range(2):
            a_s = gaussian_filter1d(a, nf, mode='nearest'); a = a_s + (a[0] - a_s[0]) * fade
            R_s = gaussian_filter1d(R, nf, mode='nearest'); R_s = R_s + (R[0] - R_s[0]) * fade
            R = np.maximum(R_s, gaussian_filter1d(Rb(np.c_[sg[q] * a, z]), nf, mode='nearest') + stack[q] - over0 * fade)
        base[i] = _join(np.r_[paths[i][:hand[i]], np.c_[SF.AXIS[0] + R * np.sin(sg[q] * a), SF.AXIS[1] - R * np.cos(sg[q] * a), z]], hand[i])
        Wv = SF.sm(WAVE['start_z'], WAVE['full_z'], z) * WAVE['amp_hh']
        psi = 2 * np.pi * (RELEASE_Z - z) / lam + phase[q]               # (by height, not by each hair's own start:
                                                                          #  neighbours swing together)
        a = a + Wv * np.sin(psi) / np.maximum(R - stack[q] + np.mean(STACK_MM) * MM, 0.05) * SF.sm(np.pi, np.pi - 0.35, a)   # (the S across the fall; none at the
                                                                          #  back's middle, where the two sides meet)
        R = R + 0.5 * Wv * np.cos(psi)                                    # (and half as much out of it, a quarter wave on:
        th = sg[q] * a                                                    #  the ribbon turns as it swings)
        low = np.c_[SF.AXIS[0] + R * np.sin(th), SF.AXIS[1] - R * np.cos(th), z]
        lines[i] = _join(np.r_[paths[i][:hand[i]], low], hand[i])
        basin[i] = 1 if kind[i] == 2 else (2 if front[q] else 3)
        sec[i] = int(a0[q] / (np.pi / SECTORS))
    # gauges (the recipe's): rank swaps between neighbours, lines turning back, one sink per basin, the photo's widths
    g = {}
    for nm, b in (('front', 2), ('back', 3)):
        for s_ in (1.0, -1.0):
            m = np.where((basin == b) & (np.array([l[0, 0] for l in lines]) * s_ > 0))[0]
            if len(m) < 10:
                continue
            aa = {}
            for zq in (0.4, 0.0, -0.4):
                v = [(abs(SF.theta(lines[i][np.argmin(np.abs(lines[i][:, 2] - zq))][None])[0]), i) for i in m
                     if lines[i][-1, 2] < zq - 0.01]
                aa[zq] = dict((i, t_) for t_, i in v)
            both = [i for i in aa[0.0] if i in aa[-0.4]]                 # (between two heights below every root)
            o = sorted(both, key=lambda i: aa[0.0][i])
            r2 = np.argsort(np.argsort([aa[-0.4][i] for i in o]))
            g['rank_moves_over_3pct_%s_%s' % (nm, 'L' if s_ > 0 else 'R')] = '%d of %d' % (int((np.abs(r2 - np.arange(len(o))) > 0.03 * len(o)).sum()), len(o))
    g['lines_turning_up'] = int(sum((np.diff(lines[i][hand[i]:, 2]) > 1e-6).any() for i in hang if basin[i] > 0))
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gauges', 'hair'))
    import flow_smooth as FS
    bi = sorted(base)
    g['smooth_falls'] = FS.gauge([base[i] for i in bi], [hand[i] for i in bi], 'free')
    g['smooth_fringe'] = FS.gauge([lines[i] for i in np.where(basin == 0)[0]], None, 'set')
    tot = hairs.sum()
    g['basin_share_pct'] = {nm: round(100 * float(hairs[basin == b].sum() / tot), 1) for nm, b in
                            (('fringe', 0), ('face framing', 1), ('front fall', 2), ('back fall', 3), ('on the head', -1))}
    A = [l for l, b in zip(lines, basin) if b > 0]
    g['half_width'] = {str(zq): round(float(np.percentile([abs(l[np.argmin(np.abs(l[:, 2] - zq)), 0]) for l in A if l[-1, 2] < zq], 98)), 3)
                       for zq in (0.52, 0.2, -0.15)}
    g['half_width_photo'] = {'0.52': 0.45, '0.2': 0.54, '-0.15': 0.63}
    g['front_inner_edge_at_-0.15'] = round(float(np.percentile([abs(l[np.argmin(np.abs(l[:, 2] + 0.15)), 0]) for l, b in zip(lines, basin)
                                                                if b == 2 and l[-1, 2] < -0.15], 2)), 3)
    g['ends_z_p5'] = round(float(np.percentile([l[-1, 2] for l in A], 5)), 3)
    log('field2 %s' % g)
    return dict(line=lines, basin=basin, hairs=hairs, root=x['root'], sec=sec, release=hand, kind=kind, gauges=g)


FCOL = {0: [(217, 138, 31), (180, 100, 10)], 1: [(194, 51, 122), (150, 30, 90)],
        2: [(47, 111, 179), (111, 163, 216), (29, 74, 128)], 3: [(44, 156, 90), (122, 79, 201), (20, 120, 110)], -1: [(150, 150, 150)]}


def draw2(Fd, head_pts, depth, path, views=((0, 4, 'front'), (40, 6, 'three quarter'), (90, 4, 'side'), (180, 4, 'back'), (0, 80, 'top')),
          every=5, px=330):
    """The field as lines over a clay head and body: fringe orange, face framing magenta, the front fall in blues and
    the back fall in greens and purples (the shade changes every section, so the order across a fall can be followed),
    and in red the line where the hair leaves the head."""
    from PIL import Image, ImageDraw
    W, H = int(1.9 * px), int(2.5 * px)
    im = Image.new('RGB', (W * len(views), H + 40), (250, 249, 247))
    dr = ImageDraw.Draw(im)
    pc = head_pts[head_pts[:, 2] > -0.95]
    for v, (az, el, name) in enumerate(views):
        a, e = math.radians(az), math.radians(el)
        fwd = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        P = lambda X: np.c_[v * W + W / 2 + (X @ right) * px, 40 + H * 0.36 - (X @ up - 0.5) * px]

        def hid(X):
            t = np.linspace(0.02, 1.3, 22)[None, :, None]
            Q = (X[:, None] + fwd[None, None] * t).reshape(-1, 3)
            return (depth(Q) > 0.004).reshape(len(X), -1).any(1)
        nrm = pc - np.c_[np.zeros(len(pc)), np.full(len(pc), 0.06), np.minimum(pc[:, 2], SF.CENTRE[2])]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        sh = np.clip(0.74 + 0.26 * (nrm @ (fwd + 0.5 * up - 0.4 * right) / 1.2), 0.5, 1.0)
        vis = ~hid(pc + nrm * 0.006)
        for (x_, y_), s_ in zip(P(pc)[vis], sh[vis]):
            g = int(225 * s_); dr.ellipse((x_ - 2.5, y_ - 2.5, x_ + 2.5, y_ + 2.5), fill=(g, int(g * 0.98), int(g * 0.95)))
        items = []
        for i in range(0, len(Fd['line']), every):
            X = Fd['line'][i][::2]
            if len(X) < 2:
                continue
            cs = FCOL[int(Fd['basin'][i])]; col = cs[int(Fd['sec'][i]) % len(cs)]
            h = hid(X); q = P(X); z = X @ fwd
            for j in range(len(X) - 1):
                if not (h[j] or h[j + 1]):
                    items.append((0.5 * (z[j] + z[j + 1]), tuple(q[j]), tuple(q[j + 1]), col, 1))
            r = Fd['release'][i]
            if Fd['basin'][i] > 0 and 0 < r < len(Fd['line'][i]) - 1:
                Y = Fd['line'][i][r - 1:r + 1]
                if not hid(Y).any():
                    items.append((float(Y[0] @ fwd) + 0.01, tuple(P(Y)[0]), tuple(P(Y)[1]), (224, 32, 32), 3))
        for z, p0, p1, col, w in sorted(items, key=lambda t: t[0]):
            dr.line([p0, p1], fill=col, width=w)
        dr.text((v * W + 12, 12), name, fill=(0, 0, 0))
    im.save(path)
