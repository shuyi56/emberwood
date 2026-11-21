"""The bare foot, built from measured proportions of the adult female foot instead of a hand-set table.

Every size is a share of the foot's length FL (the canon's 1.05 head heights: anatomy.CANON['foot']), from these
sources (docs/heel-study.md lists them with links):

  Lee, Lin & Wang 2014, J Foot Ankle Res 7:44 (3D scans, 130 adults):
      ball of foot length (heel to the 1st metatarsal head) 181.6 / 249.3 = 0.728 FL;
      outside ball length (heel to the 5th) 164.1 / 249.3 = 0.658 FL;
      heel breadth, taken at 16 % of FL, 63.2 / 249.3 = 0.254 FL.
  Hajaghazadeh et al. 2018, Health Scope 7(3):e14063 (290 women), as shares of FL:
      foot breadth 0.38; 2nd-5th toe lengths 0.98, 0.94, 0.90, 0.84; ball girth 0.93; instep girth 0.95;
      ankle girth 0.91; lateral malleolus height 57.95 / 234.35 = 0.247; medial 64.53 / 234.35 = 0.275;
      heel to lateral malleolus 43.22 / 234.35 = 0.184; to medial 53.29 / 234.35 = 0.227;
      midfoot breadth 76.43 / 234.35 = 0.326.
  Rosende-Bautista et al. 2021, IJERPH 18:9815 (266 women): dorsal height at 50 % FL 5.54 / 24.06 = 0.230 FL;
      navicular tuberosity height 3.89 / 24.06 = 0.162 FL.
  Xiong et al. 2010, JAPMA 100(1) (women): footprint arch index 0.235 (the middle third of the toeless print's
      area over the whole).
  Golding, The Manufacture of Boots and Shoes, ch. IV: the tread (ball) 4/9 inside and 5/9 outside the foot's
      centre line; the seat centred on it.
  Limanou et al. 2026, J Foot Ankle Res 19(4):e70234 (3D scans, 20 women without hallux valgus, FL 243.2 mm): the
      instep's highest point at 50 % FL 64.2 mm (0.264, used: it replaces a caliper dorsal height of 0.230), the top
      of the ball girth 37.4 mm (0.154, a check: the ball here is solved from its girth), toe 1 21.6 mm (0.089) and
      toe 5 17.1 mm (0.070) high.
  Raychoudhury, Hu & Ren 2014: the 1st-5th metatarsal line about 73 deg to the cross axis (a check on 0.728/0.658).

EXPERIMENTAL (no source found yet): the toes' widths and heights, the gaps between them, how high the arch's plantar
skin rises under the navicular, the plantar edge's rounding, and the split of the midfoot's breadth about the centre
line (interpolated between Golding's heel and ball).

Coordinates: forward is -y, up +z, the floor at z = FLOOR_Z. The ankle joint's place in plan is the rig's (x = the
leg's ankle centre, y = 0), and the foot is set so its joint lies between the malleoli: the heel's back (pternion)
0.206 FL behind it (the mean of Hajaghazadeh's heel-to-malleolus distances).

build(s, FL, floor, cx) returns a closed bmesh of the foot and its toes; MARKS[s] holds its landmarks (world
coordinates, model units) for the gauge and the boots."""
import math

import bmesh
import numpy as np

R = dict(
    ball_len=0.728, out_ball_len=0.658, heel_w=0.254, heel_w_at=0.16, foot_w=0.38, mid_w=0.326,
    toes=(1.0, 0.98, 0.94, 0.90, 0.84), ball_girth=0.93, instep_girth=0.95, lat_mall=0.247, med_mall=0.275,
    heel_lat_mall=0.184, heel_med_mall=0.227, dorsum50=0.264, navicular=0.162, arch_index=0.235,
    ball_in=4.0 / 9.0, toe1_h=0.089, toe5_h=0.070,
)
EXP = dict(
    toe_w=(1.55, 1.0, 0.93, 0.87, 0.82),    # the toes' widths, relative (the hallux about half as wide again)
    toes_span=0.94,                          # the toes together, a share of the foot's breadth
    toe_gap=0.10,                            # the gap between toes, a share of each toe's width
    toe_root=0.07,
    knuckle_run=0.12,                        # how far the dorsum takes to ease down from a knuckle to its toe, share of FL                           # how far behind its metatarsal head a toe's root starts, share of FL
    arch_peak=0.45,                          # where the arch's plantar skin stands highest (t, under the navicular)
    arch_span=0.13,                          # and how far along it the lift reaches (t, a gaussian's width)
    edge=0.40,                               # the sole's edge rounding (the bottom of the section's exponent)
)
MARKS = {}
ANKLE_FRONT = 0.335          # (the shin's front at the ankle, share of FL from the heel: read off the leg, EXPERIMENTAL)
ANKLE_ROUND = 0.14           # (the round where the shin meets the instep, share of FL: EXPERIMENTAL)
SMOOTH = 0.05                # (the top line's smoothing length, share of FL: EXPERIMENTAL)
FRONT = 0.84                 # (where the forefoot's body starts to round off under the toes, share of FL: EXPERIMENTAL)
TOE2 = 0.084                 # (the 2nd toe's top over the floor, share of FL: between Limanou 2026's 1st and 5th)
CAP = 0.035                  # (the forefoot's rounded front under the toes, share of FL: EXPERIMENTAL)
WEB = 1.0                    # (the toes' webs this far ahead of the heads, in units of the heads' own slant: EXPERIMENTAL)
CONTACT = 0.08               # (model units, 2 mm: skin this near the floor is on it, pressed)


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _girth(hw_in, hw_out, top, bot, n=96, p=EXP['edge']):
    """The girth of a section (its outline's length) and its points, for half widths and heights."""
    pts = _ring(hw_in, hw_out, top, bot, n, p, 0.0)
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _ring(hw_in, hw_out, top, bot, n, p, arch, inner_sign=1.0):
    """A section of the foot: x across (+ toward the inside), z up. The top a half ellipse, the sole flat with
    rounded edges, the inner half of the sole lifted by arch (the medial arch's skin) toward the inner edge."""
    mid = bot + min(1.0, 0.5 * (top - bot))
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        ca, sa = math.cos(a), math.sin(a)
        hw = hw_in if ca >= 0 else hw_out
        if sa >= 0:
            x, z = hw * ca, mid + (top - mid) * sa
        else:
            e = abs(sa) ** p
            x = hw * math.copysign(abs(ca) ** 0.7, ca)
            z = mid - (mid - bot) * e
            # (the arch: the sole lifted from a band along the outer border, rising toward the inner edge)
            xn = x / max(hw, 1e-9)
            u = min(1.0, max(0.0, (xn + 0.55) / 1.45))
            z += arch * (u * u * (3 - 2 * u)) * (1.0 - 0.6 * (1.0 - abs(sa)) ** 4)
        out.append((x * inner_sign, z))
    return np.array(out)


def contact_width(wi, wo, top, bot, arch, eps):
    """How wide the section stands on the floor: the run of its sole within eps of the floor."""
    r = _ring(wi, wo, top, bot, 192, EXP['edge'], arch)
    sole = r[(r[:, 1] < bot + eps)]
    return float(np.ptp(sole[:, 0])) if len(sole) > 1 else 0.0


def arch_index(rows, eps):
    """The footprint's arch index (Cavanagh & Rodgers): the toeless print's length in thirds, the middle third's
    area over the whole."""
    t = rows[:, 0]
    w = np.array([contact_width(*r[1:], eps) for r in rows])
    on = (t <= 0.80) & (w > 0)
    t, w = t[on], w[on]
    a, b = t.min(), t.max()
    tt = np.linspace(a, b, 600)
    ww = np.interp(tt, t, w)
    third = (tt > a + (b - a) / 3) & (tt < a + 2 * (b - a) / 3)
    return float(ww[third].sum() / ww.sum())


def dorsal(FL, ball_top):
    """The top line of the foot along its middle, heel to toe tip, as one smooth curve: the measured heights are
    knots of a broken line, which is smoothed (a Gaussian along the length) and its knots corrected until the smooth
    curve passes through the measured ones (the dorsum at 50 %, the ball). No corner anywhere, so the curvature
    changes gradually. Returns D(t) in model units over the floor."""
    # (in front of the ankle the line keeps rising into the shin, the way the tendons carry the instep up the front of
    # the leg, so the union with the leg meets it at a shallow angle instead of a crease)
    kt = np.array([0.0, 0.03, 0.08, 0.20, 0.31, 0.5, R['ball_len'], FRONT + 0.02, 1.0])
    kh = np.array([0.12 * FL, 0.17 * FL, 0.24 * FL, 0.46 * FL, 0.355 * FL, R['dorsum50'] * FL, ball_top,
                   TOE2 * FL, TOE2 * FL])
    target = kh.copy()
    tg = np.linspace(-0.2, 1.2, 1401)
    dt = tg[1] - tg[0]
    g = np.exp(-0.5 * (np.arange(-120, 121) * dt / SMOOTH) ** 2)
    g /= g.sum()
    fix = np.array([5, 6])                              # (the knots the smooth curve must pass: dorsum, ball)
    for _ in range(8):
        line = np.interp(tg, kt, kh)
        sm = np.convolve(np.pad(line, 120, mode='edge'), g, mode='valid')
        got = np.interp(kt, tg, sm)
        kh[fix] += target[fix] - got[fix]
    # the front of the ankle: the line comes up off the instep in a round of radius ANKLE_ROUND and stands vertical
    # where the shin's front comes down (ANKLE_FRONT), so the shin runs into the instep as one concave curve
    r = ANKLE_ROUND * FL
    tf = ANKLE_FRONT
    tc = tf + r / FL
    zc = float(np.interp(tc, tg, sm)) + r
    arc = np.where(tg >= tc, -np.inf,
                   np.where(tg >= tf, zc - np.sqrt(np.clip(r * r - ((tc - tg) * FL) ** 2, 0, None)), zc + 3.0 * (tf - tg) * FL))
    arc = np.where(tg < tf - 0.08, -np.inf, arc)                 # (behind that the leg covers it)
    sm = np.maximum(sm, arc)
    g2 = np.exp(-0.5 * (np.arange(-60, 61) * dt / (0.3 * SMOOTH)) ** 2)
    sm = np.convolve(np.pad(sm, 60, mode='edge'), g2 / g2.sum(), mode='valid')
    return lambda t: np.interp(t, tg, sm)


def profile(FL):
    """The foot's sections along its length, t = share of FL from the heel's back: (t, inner half width, outer half
    width, top, bottom, arch lift), in model units over the floor. The ball's height is solved so its girth is
    R['ball_girth'] FL; the instep girth is reported as a check."""
    hw_heel = 0.5 * R['heel_w'] * FL
    rb = hw_heel                                         # (the heel's back: a round of the heel's half breadth)
    fw = R['foot_w'] * FL
    ball_in, ball_out = R['ball_in'] * fw, (1 - R['ball_in']) * fw
    ts = np.concatenate([np.linspace(0.0, 0.16, 9)[1:], np.linspace(0.18, FRONT, 36), FRONT + CAP * np.array([0.4, 0.7, 0.9, 0.98])])
    rows = []
    # the ball's height from its girth (the section at the 1st head, both half widths at the ball's)
    lo, hi = 0.05 * FL, 0.35 * FL
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if _girth(ball_in, ball_out, m, 0.0) < R['ball_girth'] * FL:
            lo = m
        else:
            hi = m
    ball_top = 0.5 * (lo + hi)
    D = dorsal(FL, ball_top)
    for t in ts:
        y = t * FL
        # half widths: round at the back, the heel's breadth by 16 %, the midfoot's at 50 %, the ball's at the heads
        if y < rb:
            back = math.sqrt(max(rb * rb - (rb - y) ** 2, 0.0))
            wi = wo = back
        else:
            split_mid = 0.5 + (R['ball_in'] - 0.5) * (0.5 - R['heel_w_at']) / (R['ball_len'] - R['heel_w_at'])
            mid_in, mid_out = split_mid * R['mid_w'] * FL, (1 - split_mid) * R['mid_w'] * FL
            wi = np.interp(t, [R['heel_w_at'], 0.5, R['ball_len'], 0.80], [hw_heel, mid_in, ball_in, 0.92 * ball_in])
            wo = np.interp(t, [R['heel_w_at'], 0.5, R['out_ball_len'], 0.80], [hw_heel, mid_out, ball_out, 0.80 * ball_out])
            # (the forefoot ends along the toes' webs, which follow the metatarsal heads' slant: the outer side gives
            # way to the small toes from just past the 5th head, the inner side runs on past the 1st; the body's
            # outer half recedes across the toes' bases, so its front is a slant, not a square)
            web = WEB * (R['ball_len'] - R['out_ball_len'])
            recede = _ss(R['out_ball_len'] + 0.02, R['ball_len'] + web, t)
            wo = wo * (1 - recede) + (-0.35 * ball_in) * recede
        # the top: up the heel's back into the leg, over the instep (the dorsum at 50 %), down to the ball
        # (the heel's back is low and the Achilles recedes over it, so the back slopes into the leg, no shelf)
        top = float(D(t))                                   # (the one smooth top line)
        if t > FRONT:
            # (the forefoot's front: a rounded cap under the toes, not a cut face)
            c = math.sqrt(max(1.0 - ((t - FRONT) / CAP) ** 2, 0.0))
            wi, wo = wi * (0.3 + 0.7 * c), wo - (1 - c) * 0.4 * ball_in
            # (a nose at the toes' mid height, the bottom rising as the top falls: no plate on the floor under the toes)
            nose = 0.9 * TOE2 * FL
            top = nose + (top - nose) * c
            cap_bot = nose * (1 - c)
        bot = 0.035 * FL * (1 - _ss(0.0, 0.06, t)) + 0.0          # (the heel's back curls up off the floor)
        if t > FRONT:
            bot = max(bot, cap_bot)
        arch = math.exp(-((t - EXP['arch_peak']) / EXP['arch_span']) ** 2)
        # (the outer half never crosses the middle: a negative half width folds the section over itself)
        wo = max(wo, 0.2 * wi)
        rows.append((t, wi, wo, top, bot, arch))
    rows = np.array(rows)
    # the arch's height under the navicular, solved so the footprint's arch index is the measured one (Xiong 2010)
    shape = rows[:, 5].copy()
    eps = CONTACT
    lo, hi = 0.0, 0.12 * FL
    for _ in range(30):
        m = 0.5 * (lo + hi)
        rows[:, 5] = m * shape
        if arch_index(rows, eps) > R['arch_index']:
            lo = m
        else:
            hi = m
    rows[:, 5] = 0.5 * (lo + hi) * shape
    arch_peak = 0.5 * (lo + hi)
    # the instep girth at 50 %, for the check
    i50 = int(np.argmin(np.abs(rows[:, 0] - 0.5)))
    g50 = _girth(rows[i50, 1], rows[i50, 2], rows[i50, 3], rows[i50, 4])
    return rows, dict(ball_top=ball_top, instep_girth=g50 / FL, ball_girth=_girth(ball_in, ball_out, ball_top, 0.0) / FL,
                      arch_skin=arch_peak / FL, arch_index=arch_index(rows, eps))


def toes(FL, s, cx, y0, floor, rows, D):
    """The five toes: base at each metatarsal head (on the 1st-5th line), tip at its measured length. Returns a list
    of (centre line [k, 3], widths [k], heights [k]) and their tips."""
    fw = R['foot_w'] * FL
    span = EXP['toes_span'] * fw
    w = np.array(EXP['toe_w'])
    w = w / w.sum() * span
    inner_edge = R['ball_in'] * fw * 0.96
    xs = inner_edge - np.concatenate([[0], np.cumsum(w)[:-1]]) - 0.5 * w     # (+ inside, toward the midline)
    t_mt = np.interp(xs, [xs[-1], xs[0]], [R['out_ball_len'], R['ball_len']])
    out = []

    def body_h(t, x):
        """The forefoot's top over the floor at length share t, x across (+ inside)."""
        wi_, wo_, top_, bot_ = (np.interp(t, rows[:, 0], rows[:, k]) for k in (1, 2, 3, 4))
        hw = wi_ if x >= 0 else wo_
        if hw <= 1e-6 or abs(x) >= hw:
            return bot_
        mid = bot_ + min(1.0, 0.5 * (top_ - bot_))
        return mid + (top_ - mid) * math.sqrt(1 - (x / hw) ** 2)
    st = np.concatenate([np.linspace(0.0, 0.7, 12), 0.7 + 0.3 * np.sin(np.linspace(0, 0.5 * np.pi, 9)[1:])])
    for i in range(5):
        wi = w[i] * (1 - EXP['toe_gap'])
        hi = np.interp(i, [0, 4], [R['toe1_h'], R['toe5_h']]) * FL       # (the toe's top over the floor: measured
        # for the 1st and 5th, Limanou 2026; between them interpolated, EXPERIMENTAL)
        t_tip = R['toes'][i]
        # (the toe starts inside the forefoot, behind its metatarsal head, and its top runs out of the forefoot's
        # top there: the dorsum flows into each toe instead of dropping into a hollow before it)
        t_root = t_mt[i] - EXP['toe_root']
        tt = t_root + st * (t_tip - t_root)
        own = hi * (1.0 - 0.12 * np.clip((tt - t_mt[i]) / (t_tip - t_mt[i]), 0, 1))
        # (each toe carries its own knuckle: the forefoot's height over its metatarsal head, easing down in an S to
        # the toe's own height, so the dorsum runs into every toe as a rounded ridge, whatever the forefoot's edge
        # does around it)
        # (the toe's top is the foot's one top line, taken across to the toe's place over the ball's breadth, until
        # that line comes down to the toe's own height: a smooth maximum, so there is no corner where they meet)
        hw = (R['ball_in'] if xs[i] >= 0 else 1 - R['ball_in']) * fw
        across = math.sqrt(max(1 - (xs[i] / hw) ** 2, 0.0))
        line = np.array([min(1.0, 0.5 * float(D(t))) + (float(D(t)) - min(1.0, 0.5 * float(D(t)))) * across for t in tt])
        line -= 0.04 * FL / 9.86
        k_ = 0.12 * hi
        top = 0.5 * (own + line + np.sqrt((own - line) ** 2 + k_ * k_))
        span_ = t_tip - t_mt[i]
        bot = hi * np.interp(tt, [t_root, t_mt[i], t_mt[i] + 0.3 * span_, t_tip], [0.45, 0.15, 0.0, 0.08])
        ww = wi * np.interp(tt, [t_root, t_mt[i], t_mt[i] + 0.5 * EXP['knuckle_run'], t_tip], [1.15, 1.12, 1.0, 0.92])
        # the tip: a rounded dome over its last half width, in plan and in profile
        Rt = 0.55 * wi
        d_tip = (t_tip - tt) * FL
        r = np.sqrt(np.clip(1 - ((Rt - np.minimum(d_tip, Rt)) / Rt) ** 2, 0.0, 1.0))
        r = np.maximum(r, 0.12)
        c = bot + 0.45 * (top - bot)
        top, bot, ww = c + (top - c) * r, c - (c - bot) * r, ww * r
        yy = y0 - tt * FL
        xc = cx + s * xs[i]
        # (the toes lean a little toward the 2nd's line as they go: the hallux in, the small toes in)
        lean = np.interp(tt, [t_mt[i], t_tip], [0, 0.04 * FL]) * np.sign(xs[i] - xs[1]) * (i != 1)
        xc = xc + s * lean
        C = np.stack([np.broadcast_to(xc, tt.shape), yy, np.full(tt.shape, floor)], 1)
        out.append((C, ww, floor + top, floor + bot))
    return out, xs, t_mt


def _loft(bm, rings):
    loops = [[bm.verts.new(tuple(p)) for p in r] for r in rings]
    n = len(rings[0])
    for a, b in zip(loops, loops[1:]):
        for k in range(n):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    bm.faces.new(list(reversed(loops[0])))
    bm.faces.new(loops[-1])


def build(s, FL, floor, cx, y_ankle=0.0, n=40):
    """The foot on side s (1 her right, at -x), FL long, its sole on the floor, the heel centred under x = cx (the
    leg's ankle centre), the ankle joint at y = y_ankle. Returns a closed bmesh; fills MARKS[s]."""
    rows, chk = profile(FL)
    y0 = y_ankle + 0.5 * (R['heel_lat_mall'] + R['heel_med_mall']) * FL      # (the pternion's y)
    bm = bmesh.new()
    rings = []
    for t, wi, wo, top, bot, arch in rows:
        sec = _ring(wi, wo, top, bot, n, EXP['edge'], arch)
        # (+x of the section is the inside: toward the midline, which is +x for her right foot, s = 1)
        X = cx + s * sec[:, 0]
        rings.append(np.stack([X, np.full(n, y0 - t * FL), floor + sec[:, 1]], 1))
    _loft(bm, rings)
    tl, xs, t_mt = toes(FL, s, cx, y0, floor, rows, dorsal(FL, chk['ball_top']))
    m = 16
    for C, ww, zt, zb in tl:
        rr = []
        for c, w_, t_, b_ in zip(C, ww, zt, zb):
            a = np.linspace(0, 2 * np.pi, m, endpoint=False)
            # (a toe's section: a rounder top over a flatter pad)
            zz = np.where(np.sin(a) >= 0, b_ + 0.45 * (t_ - b_) + 0.55 * (t_ - b_) * np.sin(a),
                          b_ + 0.45 * (t_ - b_) * (1 + np.sign(np.sin(a)) * np.abs(np.sin(a)) ** 0.7))
            rr.append(np.stack([c[0] + 0.5 * w_ * np.cos(a), np.full(m, c[1]), zz], 1))
        _loft(bm, rr)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # (outward normals checked by the signed volume: a closed shape with inward normals reads inside-out to the
    # anatomy's distance field and wrecks the ankle's fillet)
    bm.faces.ensure_lookup_table()
    vol = 0.0
    for f_ in bm.faces:
        c = f_.calc_center_median()
        vol += c.dot(f_.normal) * f_.calc_area()
    if vol < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    mt1 = (cx + s * (R['ball_in'] * R['foot_w'] * FL), y0 - R['ball_len'] * FL, floor)
    mt5 = (cx - s * ((1 - R['ball_in']) * R['foot_w'] * FL), y0 - R['out_ball_len'] * FL, floor)
    MARKS[s] = dict(
        FL=FL, pternion=(cx, y0, floor), mt1=mt1, mt5=mt5,
        toe_tips=[(float(C[-1, 0]), float(C[-1, 1]), floor) for C, _, _, _ in tl],
        seat=(cx, y0 - FL / 6.0, floor),
        heel_breadth_at=(cx, y0 - R['heel_w_at'] * FL, floor),
        lat_malleolus=(cx - s * 0.5 * 0.274 * FL, y0 - R['heel_lat_mall'] * FL, floor + R['lat_mall'] * FL),
        med_malleolus=(cx + s * 0.5 * 0.274 * FL, y0 - R['heel_med_mall'] * FL, floor + R['med_mall'] * FL),
        navicular=(cx + s * 0.5 * R['mid_w'] * FL * 0.9, y0 - 0.43 * FL, floor + R['navicular'] * FL),
        dorsum50=(cx, y0 - 0.5 * FL, floor + R['dorsum50'] * FL),
        regions=dict(heel=(0.0, 0.30), midfoot=(0.30, 0.60), forefoot=(0.60, 0.80), toes=(0.80, 1.0)),
        checks=chk,
    )
    return bm
