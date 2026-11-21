"""The contour sheet of a join (continuity.py): one PNG per join, numpy only (labels when PIL is there).

Top row, the sections across the join at six stations: side A blue, side B orange, the join a black dot, and the
curvature comb (a tooth along the normal at every sample, its length the curvature; a smooth G2 join's comb runs
through the dot with no step in its outline, a G1 join's comb jumps there, a G0 or crease shows as a gap or a
corner).
Middle row, along the join (station by station): the tangent angle G1 (deg) against the limits, the curvature
deviation G2, and the two sides' curvatures kA, kB (1/cm). Light grey: the same measure away from the join (the
control), so a step that is only at the join stands out.
Bottom row, three cheap views of the surface round the join, from one fixed orthographic camera looking at it:
the normal-deviation heat map (the largest angle to a neighbour's normal: grey none, red 4 deg or more), zebra
stripes (reflection lines: a G0 break cuts them, a G1 break kinks them, a G2 break bends them sharply; a G2-smooth
join keeps them smooth through it) and a matte clay matcap. The join line is drawn in black on all three."""
import math

import numpy as np

import continuity as C
from contours import Canvas

PW, PH, PAD = 300, 240, 16
BLUE, ORANGE, BLACK, GREY, RED, GREEN = (40, 90, 200), (230, 120, 20), (20, 20, 20), (175, 175, 175), (205, 40, 35), \
    (40, 150, 70)


def _box(P, x0, y0, w=PW, h=PH, pad=PAD, box=None):
    P = np.asarray(P, float)
    lo, hi = (P.min(0), P.max(0)) if box is None else box
    sc = min((w - 2 * pad) / max(hi[0] - lo[0], 1e-9), (h - 2 * pad - 12) / max(hi[1] - lo[1], 1e-9))
    cx, cy = (lo + hi) / 2

    def f(Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        return np.column_stack([x0 + w / 2 + (Q[:, 0] - cx) * sc, y0 + 12 + (h - 12) / 2 - (Q[:, 1] - cy) * sc])
    return f, sc


def _frame(cv, x0, y0, title):
    cv.poly(np.array([[x0, y0], [x0 + PW - 1, y0], [x0 + PW - 1, y0 + PH - 1], [x0, y0 + PH - 1]]), (215, 215, 210),
            closed=True)
    cv.text(x0 + 4, y0 + 2, title)


def section_panel(cv, x0, y0, r, comb_scale=None):
    X, s, at, _ = r['_section']
    win = 3.0 * (r.get('_win') or 1.0)
    m = (s > at - win) & (s < at + win)
    if m.sum() < 4:
        m = np.ones(len(s), bool)
    P, S = X[m], s[m]
    j = int(np.argmin(np.abs(S - at)))
    f, sc = _box(P, x0, y0)
    k = _discrete_k(P)
    nrm = _normals2(P)
    cs = comb_scale or 0.25 * (PH - 40) / max(np.percentile(np.abs(k), 95), 1e-6) / sc
    tips = P + nrm * (k * cs)[:, None]
    for i in range(len(P)):
        col = (150, 180, 235) if S[i] < at else (245, 190, 140)
        cv.line(f(P[i])[0], f(tips[i])[0], col)
    cv.poly(f(tips[S < at]), BLUE)
    cv.poly(f(tips[S >= at]), ORANGE)
    cv.poly(f(P[S < at]), BLUE, 2)
    cv.poly(f(P[S >= at]), ORANGE, 2)
    p = f(P[j])[0]
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx * dx + dy * dy <= 9:
                cv.dots(np.array([[p[0] + dx, p[1] + dy]]), BLACK)
    g2 = '-' if r['g2_dev'] is None else '%.2f' % r['g2_dev']
    _frame(cv, x0, y0, 'st %d  G0 %.3fmm G1 %.2fd G2 %s' % (r['station'], r['g0_mm'], r['g1_deg'], g2))


def _discrete_k(P):
    d1 = np.gradient(P, axis=0)
    d2 = np.gradient(d1, axis=0)
    k = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)
    w = np.ones(5) / 5
    return np.convolve(np.pad(k, 2, mode='edge'), w, mode='valid')


def _normals2(P):
    t = np.gradient(P, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.column_stack([-t[:, 1], t[:, 0]])


def chart(cv, x0, y0, xs, series, title, limits=(), ylab=''):
    """Line chart: series = [(values, colour), ...]; limits = [(value, colour), ...] as horizontal lines."""
    allv = np.concatenate([np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for v, _ in series] +
                          [np.array([l for l, _ in limits], float)] + [np.zeros(1)])
    lo, hi = float(allv.min()), float(allv.max())
    if hi - lo < 1e-9:
        hi = lo + 1
    f, _ = _box(np.array([[min(xs), lo], [max(xs), hi]]), x0, y0, box=(np.array([min(xs), lo]),
                                                                        np.array([max(xs), hi])))
    for l, col in limits:
        cv.line(f([min(xs), l])[0], f([max(xs), l])[0], col)
    cv.line(f([min(xs), 0])[0], f([max(xs), 0])[0], (225, 225, 225))
    for v, col in series:
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        pts = f(np.column_stack([np.asarray(xs)[ok], v[ok]]))
        cv.poly(pts, col, 2 if col != GREY else 1)
        cv.dots(pts, col)
    _frame(cv, x0, y0, title + ('  [%s %.3g..%.3g]' % (ylab, lo, hi) if ylab else ''))


# ------------------------------------------------------------------ the three views

def _camera(V, look):
    look = np.asarray(look, float) / np.linalg.norm(look)
    up0 = np.array([0, 0, 1.0]) if abs(look[2]) < 0.9 else np.array([0, 1.0, 0])
    right = np.cross(look, up0)
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    return right, up, look


def raster(V, F, N, look, w=PW, h=PH - 14, vals=None, box=None):
    """Orthographic z-buffer: per pixel the interpolated normal (and a per-vertex value). Returns (normal image,
    value image, mask, projector)."""
    right, up, fwd = _camera(V, look)
    P = np.column_stack([V @ right, V @ up, -(V @ fwd)])
    used = np.unique(F)
    lo, hi = (P[used, :2].min(0), P[used, :2].max(0)) if box is None else box
    sc = min((w - 8) / max(hi[0] - lo[0], 1e-9), (h - 8) / max(hi[1] - lo[1], 1e-9))
    c = (lo + hi) / 2
    X = np.column_stack([w / 2 + (P[:, 0] - c[0]) * sc, h / 2 - (P[:, 1] - c[1]) * sc, P[:, 2]])
    zb = np.full((h, w), -np.inf)
    nimg = np.zeros((h, w, 3))
    vimg = np.zeros((h, w))
    for t in F:
        a, b, cc = X[t[0]], X[t[1]], X[t[2]]
        x0, x1 = int(max(0, math.floor(min(a[0], b[0], cc[0])))), int(min(w - 1, math.ceil(max(a[0], b[0], cc[0]))))
        y0, y1 = int(max(0, math.floor(min(a[1], b[1], cc[1])))), int(min(h - 1, math.ceil(max(a[1], b[1], cc[1]))))
        if x1 < x0 or y1 < y0:
            continue
        den = (b[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - b[0]) * (a[1] - cc[1])
        if abs(den) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l1 = ((b[1] - cc[1]) * (xs - cc[0]) + (cc[0] - b[0]) * (ys - cc[1])) / den
        l2 = ((cc[1] - a[1]) * (xs - cc[0]) + (a[0] - cc[0]) * (ys - cc[1])) / den
        l3 = 1 - l1 - l2
        m = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not m.any():
            continue
        z = l1 * a[2] + l2 * b[2] + l3 * cc[2]
        yy, xx = (ys[m] - 0.5).astype(int), (xs[m] - 0.5).astype(int)
        zz = z[m]
        closer = zz > zb[yy, xx]
        if not closer.any():
            continue
        yy, xx = yy[closer], xx[closer]
        L1, L2, L3 = l1[m][closer], l2[m][closer], l3[m][closer]
        zb[yy, xx] = zz[closer]
        n = L1[:, None] * N[t[0]] + L2[:, None] * N[t[1]] + L3[:, None] * N[t[2]]
        nimg[yy, xx] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        if vals is not None:
            vimg[yy, xx] = L1 * vals[t[0]] + L2 * vals[t[1]] + L3 * vals[t[2]]
    mask = np.isfinite(zb)

    def proj(Q):
        Q = np.atleast_2d(Q)
        return np.column_stack([w / 2 + (Q @ right - c[0]) * sc, h / 2 - (Q @ up - c[1]) * sc])
    return nimg, vimg, mask, proj, (right, up, fwd)


def zebra(nimg, mask, frame, axis, stripes=14):
    """Reflection lines: the view reflected in the surface, against an environment of parallel light bars round
    `axis` (bars every 180/stripes deg)."""
    right, up, fwd = frame
    v = -fwd
    r = 2 * (nimg @ v)[..., None] * nimg - v
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    b = np.cross(a, v)
    if np.linalg.norm(b) < 1e-6:
        b = np.cross(a, up)
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    th = np.arctan2(r @ b, r @ c)
    z = (np.floor(th / math.pi * stripes) % 2).astype(float)
    out = np.empty(nimg.shape[:2] + (3,), np.uint8)
    out[:] = (250, 250, 248)
    out[mask & (z > 0)] = (240, 240, 240)
    out[mask & (z <= 0)] = (25, 25, 30)
    return out


def matcap(nimg, mask, frame):
    right, up, fwd = frame
    key = -fwd * 0.6 + up * 0.55 - right * 0.45
    key /= np.linalg.norm(key)
    lam = np.clip(nimg @ key, 0, 1)
    rim = np.clip(1 - np.abs(nimg @ fwd), 0, 1) ** 3
    v = 0.18 + 0.72 * lam + 0.12 * rim
    out = np.empty(nimg.shape[:2] + (3,), np.uint8)
    out[:] = (250, 250, 248)
    col = np.array([196, 170, 150]) * v[..., None]
    out[mask] = np.clip(col[mask], 0, 255).astype(np.uint8)
    return out


def heat(vimg, mask, lo=0.0, hi=4.0):
    t = np.clip((vimg - lo) / (hi - lo), 0, 1)
    a, b = np.array([215, 215, 210.0]), np.array([205, 40, 35.0])
    out = np.empty(vimg.shape + (3,), np.uint8)
    out[:] = (250, 250, 248)
    col = a + (b - a) * t[..., None]
    out[mask] = col[mask].astype(np.uint8)
    return out


def views(cv, x0, y0, A, B, seam, look, axis, extra_lines=(), seam_cols=None):
    """The heat map, the zebra and the matcap of the surface round the join, side by side from x0."""
    parts = [A] + ([B] if B is not None else [])
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    V, F = np.vstack(Vs), np.vstack(Fs)
    N = np.vstack([C.vertex_normals(Vp, Fp) for Vp, Fp in parts])
    nd = np.concatenate([C.normal_deviation(Vp, Fp, C.vertex_normals(Vp, Fp)) for Vp, Fp in parts])
    # (the camera looks at the join from in front: keep only faces facing it, so the far side does not show through)
    nimg, vimg, mask, proj, frame = raster(V, F, N, look, vals=nd)
    imgs = [heat(vimg, mask), zebra(nimg, mask, frame, axis), matcap(nimg, mask, frame)]
    titles = ['normal deviation (grey 0, red 4 deg+); join by grade', 'zebra (reflection lines)', 'clay matcap']
    for q, (im, tt) in enumerate(zip(imgs, titles)):
        X0 = x0 + q * PW
        cv.a[y0 + 14:y0 + 14 + im.shape[0], X0:X0 + im.shape[1]] = im
        sp = proj(seam) + np.array([X0, y0 + 14])
        if seam_cols is None:
            cv.poly(sp, (255, 210, 0) if q == 1 else BLACK, 1)
        else:
            for i in range(len(sp) - 1):
                cv.line(sp[i], sp[i + 1], seam_cols[i], 2)
        for L, col in extra_lines:
            cv.poly(proj(L) + np.array([X0, y0 + 14]), col, 1)
        _frame(cv, X0, y0, tt)


def _near(M, p, r, look):
    """The faces of M within r of p that face the camera."""
    V, F = M
    fn = C.face_normals(V, F)[0]
    keep = (np.linalg.norm(V[F].mean(1) - p, axis=1) < r) & (fn @ np.asarray(look) < 0.15)
    return V, F[keep]


def _outside(A, B):
    V, F = A
    _, _, sd = C.nearest_on_mesh(B[0], B[1], V)
    keep = (sd[F] > -1e-4).all(1)
    return V, F[keep]


def sheet(rep, rows, path, A=None, B=None, seam=None):
    cols = 3
    H = 3 * PH + 30
    cv = Canvas(cols * PW, H)
    cv.text(6, 4, '%s  [%s, %s]  %s   G0 max %.3f mm  G1 p90 %.2f deg  G2 p90 %.2f   window %.2f cm' % (
        rep['name'], rep.get('mode'), rep['kind'], rep['verdict'].upper(), rep['summary']['g0_mm_max'],
        rep['summary']['g1_deg_p90'], rep['summary']['g2_dev_p90'], rep['window_cm']))
    y = 20
    pick = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, 3).astype(int))]
    worst = max(rows, key=lambda r: r['g1_deg'] / 4 + (r['g2_dev'] or 0))
    if worst not in pick:
        pick[-1] = worst
    for q, r in enumerate(pick[:3]):
        section_panel(cv, q * PW, y, r)
    y += PH
    st = [r['station'] for r in rows]
    lim = rep['limits']
    chart(cv, 0, y, st, [([r['ctl_g1_deg'] for r in rows], GREY), ([r['g1_deg'] for r in rows], BLUE)],
          'G1 tangent angle', [(lim['g1_deg'][0], GREEN), (lim['g1_deg'][1], RED)], 'deg')
    chart(cv, PW, y, st, [([r['ctl_g2_dev'] for r in rows], GREY), ([r['g2_dev'] for r in rows], BLUE)],
          'G2 curvature deviation', [(lim['g2_dev'][0], GREEN), (lim['g2_dev'][1], RED)], '')
    chart(cv, 2 * PW, y, st, [([r['kA'] for r in rows], BLUE), ([r['kB'] for r in rows], ORANGE)],
          'kA blue, kB orange', (), '1/cm')
    y += PH
    if A is not None and seam is not None:
        # (the camera looks at the worst station from outside, square to the surface there)
        p = np.asarray(worst['p'])
        n = 0
        for M in [A] + ([B] if B is not None else []):
            i = int(np.argmin(np.linalg.norm(M[0] - p, axis=1)))
            n = n + C.vertex_normals(*M)[i]
        look = -n / max(np.linalg.norm(n), 1e-9)
        Tn = C._seam_frame(seam, False)[int(np.argmin(np.linalg.norm(seam - p, axis=1)))]
        r = 4.0 * rep['window_cm'] * 10.0 / rep['mm_per_u']        # (framed on the worst station, 4 windows round)
        A = _near(A, p, r, look)
        B = _near(B, p, r, look) if B is not None else None
        sm = seam[np.linalg.norm(seam - p, axis=1) < r]
        G = {'pass': GREEN, 'warn': (240, 180, 0), 'fail': RED}
        P = np.array([q['p'] for q in rows])
        cols = [G[rows[int(np.argmin(np.linalg.norm(P - x, axis=1)))]['grade']] for x in sm]
        views(cv, 0, y, A, B, sm, look, np.cross(Tn, look), seam_cols=cols)
    cv.save(path)
    return path


def profile_sheet(name, prof, rows, steps, path, A=None, line3=None, look=None, joins=(), rep=None):
    """The sheet of a profile line with no marked join (the foot's top line): the line with its curvature comb,
    k(s), the one-sided G1 and G2 along it with the steps found, and the three views round it."""
    cv = Canvas(3 * PW, 3 * PH + 30)
    rk = rep['rms_k_per_cm'] if rep else float(np.sqrt((prof['k'] ** 2).mean()))
    cv.text(6, 4, '%s   %s   length %.1f cm   rms k %.3f /cm (past %.1f cm)   G1 max %s deg   steps %d' % (
        name, (rep or {}).get('verdict', '').upper(), prof['s_cm'][-1], rk, (rep or {}).get('skip_ends_cm', 0),
        (rep or {}).get('g1_deg_max'), len(steps)))
    y = 20
    P, k, s = prof['xy'], prof['k'], prof['s_cm']
    f, sc = _box(P, 0, y, w=2 * PW)
    nrm = _normals2(P)
    cs = 0.3 * (PH - 40) / max(np.percentile(np.abs(k), 95), 1e-6) / sc
    tips = P + nrm * (k * cs)[:, None]
    for i in range(0, len(P), 2):
        cv.line(f(P[i])[0], f(tips[i])[0], (160, 185, 235) if k[i] >= 0 else (240, 175, 160))
    cv.poly(f(tips), BLUE)
    cv.poly(f(P), BLACK, 2)
    for st in steps:
        i = int(np.argmin(np.abs(s - st['s_cm'])))
        p = f(P[i])[0]
        col = RED if st['kind'] == 'tangent' else ORANGE
        cv.poly(np.array([[p[0] - 5, p[1] - 5], [p[0] + 5, p[1] - 5], [p[0] + 5, p[1] + 5], [p[0] - 5, p[1] + 5]]),
                col, 2, closed=True)
    for js in joins:
        i = int(np.argmin(np.abs(s - js)))
        p = f(P[i])[0]
        cv.line([p[0], p[1] - 12], [p[0], p[1] + 12], GREEN, 2)
    _frame(cv, 0, y, 'the line and its curvature comb (red box: tangent step, orange: curvature step, green: join)')
    cv.poly(np.array([[2 * PW, y], [2 * PW, y + PH]]), (215, 215, 210))
    chart(cv, 2 * PW, y, s, [(k, BLUE)], 'curvature k(s)', (), '1/cm')
    y += PH
    rs = [r['s_cm'] for r in rows]
    chart(cv, 0, y, rs, [([r['g1_deg'] for r in rows], BLUE)], 'G1 one-sided tangent jump along s',
          [(C.LIMITS['matte']['g1_deg'][0], GREEN), (C.LIMITS['matte']['g1_deg'][1], RED)], 'deg')
    chart(cv, PW, y, rs, [([r['g2_dev'] for r in rows], BLUE)], 'G2 curvature deviation along s',
          [(C.LIMITS['matte']['g2_dev'][0], GREEN), (C.LIMITS['matte']['g2_dev'][1], RED)], '')
    chart(cv, 2 * PW, y, s, [(prof['dk'], BLUE)], 'dk/ds', (), '1/cm2')
    y += PH
    if A is not None and line3 is not None:
        views(cv, 0, y, A, None, line3, look, np.array([1.0, 0, 0]))
    cv.save(path)
    return path
