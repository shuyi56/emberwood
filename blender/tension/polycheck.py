"""polycheck.py BUILD... [--az 30] [--png OUT.png] [--json OUT.json]: the polygon check against the riding-habit photo.

John, 2026-11-23: "Your lapel is a triangle there's has 4 sides", "Properly get the number of sides right", and then
"Add a proper polygon check with side count". The earlier fit (fitref.py) compared five chosen points, so a triangle
lapel could score well while having the wrong number of sides. This gauge does not choose the corners: it takes each
piece's outline from the model as seen from the photo's side, finds the corners by itself, and only then compares.

Reference (photo px of collar-reference.jpeg, her left side, traced by John's lines on the gamma-lifted close-up):
  lapel, 4 sides: break L0, roll top R, corner G, point P (roll line, level top edge, end edge, lower edge): the top
  and lower edges run level, so it is a rectangle with a triangle on its end ("a combination of a rectangle and a
  triangle", John 2026-11-23, and his markup of trace-overlay.png)
  collar front, 4 sides: notch G, top of the front at the neck T, the shoulder corner K, the rounded corner C: the
  front edge up the neck, the outer edge down, the end edge, and the bottom edge lying almost level on the lapel
  (John's markup draws it as a tilted rectangle with a rounded bottom corner)

The model's pieces, from the same ortho camera (az 30, toward her left):
  lapel: the cut outline of her left lapel (lapels.npz: the roll line, the top edge, the outer edge), projected
  collar front: what is visible of the collar, by a depth buffer of body and collar together; the outline of the
  visible patch that holds the notch. The photo's scarf fills the neck opening and hides the collar there, so the
  collar inside her left roll top (across the throat, the far side) only occludes and is not counted

Corners: Douglas-Peucker at 4% of the outline's size, then a vertex turning less than 25 degrees is not a corner,
and corners closer than 8% of the perimeter are one corner (a rounded point counts once). Units: the lapel's
lower edge span (L0 to P across), as fitref.py.

The check fails when a piece's side count differs from the photo's. Past that it lists every edge whose direction is
off by more than 12 degrees or whose share of the perimeter is off by more than a third, and every corner whose
angle is off by more than 15 degrees.
"""
import sys, os, json, math
import numpy as np

REF = {
    'lapel': dict(closed=True, names=['L0', 'R', 'G', 'P'],
                  pts=[(588, 301), (612, 269), (668, 262), (693, 290)],
                  sides=['roll line', 'top edge', 'end edge', 'lower edge']),
    'collar front': dict(closed=True, names=['G', 'T', 'K', 'C'],
                         pts=[(670, 264), (687, 206), (722, 254), (707, 273)],
                         sides=['front edge up the neck', 'outer edge', 'end edge', 'bottom edge on the lapel']),
}
SPAN = (588, 693)          # the photo's L0 and P, across: the unit
TOL = dict(dir=12.0, share=1 / 3, corner=15.0)


def photo_norm(p):
    s = SPAN[1] - SPAN[0]
    return np.array([(p[0] - SPAN[0]) / s, -(p[1] - 301) / s])


# ---------------------------------------------------------------- corners
def _dp(P, eps):
    if len(P) < 3:
        return list(range(len(P)))
    a, b = P[0], P[-1]; ab = b - a; L = np.linalg.norm(ab)
    d = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L if L > 1e-12 else np.linalg.norm(P - a, axis=1)
    k = int(np.argmax(d))
    if d[k] <= eps:
        return [0, len(P) - 1]
    l = _dp(P[:k + 1], eps); r = _dp(P[k:], eps)
    return l[:-1] + [i + k for i in r]


def corners(P, closed=True, eps_frac=0.04, turn_min=25.0, merge_frac=0.08):
    """Corner indices of the outline P (n x 2), found by itself."""
    P = np.asarray(P, float)
    size = np.linalg.norm(P.max(0) - P.min(0))
    if closed:
        # start the walk at the point farthest from the centroid (always a corner or near one)
        s = int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1))); P2 = np.vstack([P[s:], P[:s], P[s:s + 1]])
        idx = [(i + s) % len(P) for i in _dp(P2, eps_frac * size)[:-1]]
    else:
        idx = _dp(P, eps_frac * size)
    per = np.sum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1))
    def turn(i, j, k):
        u, v = P[j] - P[i], P[k] - P[j]
        return abs(math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v)))
    changed = True
    while changed and len(idx) > 2:
        changed = False
        n = len(idx)
        rng = range(n) if closed else range(1, n - 1)
        tv = [(turn(idx[(m - 1) % n], idx[m], idx[(m + 1) % n]), m) for m in rng]
        t, m = min(tv)
        if t < turn_min:
            idx.pop(m); changed = True; continue
        # a rounded corner: two corners close together along the outline become one, at the middle of the run
        for m in (range(n) if closed else range(n - 1)):
            a, b = idx[m], idx[(m + 1) % n]
            run = (b - a) % len(P) if closed else b - a
            seg = np.sum(np.linalg.norm(np.diff(P[[(a + q) % len(P) for q in range(run + 1)]], axis=0), axis=1))
            if seg < merge_frac * per and (closed or (m > 0 and m + 1 < n - 1)):
                idx[m] = (a + run // 2) % len(P); idx.pop((m + 1) % n); changed = True; break
    return idx


def describe(C, closed=True):
    """Sides of the polygon with corners C (k x 2): direction (deg, y up), share of the perimeter; corner angles."""
    C = np.asarray(C, float); k = len(C)
    E = [(C[i], C[(i + 1) % k]) for i in range(k if closed else k - 1)]
    L = np.array([np.linalg.norm(b - a) for a, b in E]); per = L.sum()
    dirs = [math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) for a, b in E]
    ang = []
    for i in range(k):
        if not closed and i in (0, k - 1):
            ang.append(float('nan')); continue
        u, v = C[(i - 1) % k] - C[i], C[(i + 1) % k] - C[i]
        ang.append(math.degrees(math.acos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))
    return dict(n=len(E), dirs=dirs, share=list(L / per), angles=ang)


def ddir(a, b):
    """Difference of two edge directions, as lines (an edge walked the other way is the same edge)."""
    d = (a - b + 180) % 360 - 180
    return d


# ---------------------------------------------------------------- curvature (John, 2026-11-23: "Is it a pentagon or
# just a highly curved top that creates the edge"). A corner is where the outline turns on a small radius; a curved
# edge turns on a large one all along. Each side's bow (its deepest point off its chord, as a share of the chord;
# + and - for the two sides of the chord, an S has both) and each corner's rounding radius, in lapel spans.
PHOTO = os.path.expanduser('~/.claude/uploads/b11c8793-1872-50e5-b6c9-17f5ea11d431/d1845277-collar-reference.jpeg')
# measured on the photo with photo_collar() below (the dark collar against the scarf and the sky; its end and bottom
# edges lie on the dark coat and cannot be measured): the front edge straight, the outer edge one long S-curve, a sharp
# top corner T and a rounded shoulder corner K
PHOTO_CURVE = {'front edge up the neck': dict(bow=(0.060, -0.025)), 'outer edge': dict(bow=(0.041, -0.074)),
               'T': dict(radius=0.037), 'K': dict(radius=0.115)}      # (polycheck.py --photo, 2026-11-23)


def resample_curve(P, step):
    P = np.asarray(P, float); sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    u = np.arange(0, sl[-1], step); return np.column_stack([np.interp(u, sl, P[:, 0]), np.interp(u, sl, P[:, 1])])


def curvature(P, sigma):
    from scipy.ndimage import gaussian_filter1d
    x = gaussian_filter1d(P[:, 0], sigma, mode='wrap'); y = gaussian_filter1d(P[:, 1], sigma, mode='wrap')
    dx, dy = np.gradient(x), np.gradient(y); ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)


def side_bow(C):
    a, b = C[0], C[-1]; ab = b - a; L = np.linalg.norm(ab); n = np.array([-ab[1], ab[0]]) / L
    sd = (C - a) @ n / L; return float(sd.max()), float(sd.min())


def curve_measure(P, ci, step=0.01, sigma_span=0.035, win=0.06):
    """Bows of the sides between corners ci of the closed outline P (lapel spans), and each corner's radius."""
    P = np.asarray(P, float); n = len(P)
    bows = []
    for i in range(len(ci)):
        a, b = ci[i], ci[(i + 1) % len(ci)]; seg = P[[(a + q) % n for q in range((b - a) % n + 1)]]
        bows.append(side_bow(seg))
    R = resample_curve(np.vstack([P, P[:1]]), step); k = curvature(R, sigma_span / step)
    radii = []
    for i in ci:
        j = int(np.argmin(np.linalg.norm(R - P[i], axis=1))); w = int(win / step)
        kk = np.abs(k[[(j + q) % len(R) for q in range(-w, w + 1)]]); radii.append(float(1 / max(kk.max(), 1e-6)))
    return bows, radii


def photo_collar():
    """The collar's outline on the photo (photo px), from the dark cloth against the scarf and the sky."""
    from PIL import Image
    from scipy import ndimage
    import contourpy
    im = np.asarray(Image.open(PHOTO).convert('L'), float); x0, x1, y0, y1 = 640, 760, 190, 285
    a = ndimage.gaussian_filter(im[y0:y1, x0:x1], 0.8); lab, _ = ndimage.label(a < 110)
    m = ndimage.binary_opening(lab == lab[230 - y0, 705 - x0], iterations=1)
    L = max(contourpy.contour_generator(z=np.pad(m.astype(float), 1)).lines(0.5), key=len) - 1
    return np.column_stack([L[:, 0] + x0, L[:, 1] + y0])


# ---------------------------------------------------------------- the model
def camera(az):
    th = math.radians(az)
    R = np.array([math.cos(th), math.sin(th), 0.0]); D = np.array([math.sin(th), -math.cos(th), 0.0])   # (D: toward the camera)
    return R, D


def lapel_outline(B, az):
    """Her left lapel's cut outline, projected (x right, y up), in walking order: roll top, the top edge (if any), the
    outer edge down to the break, the roll line back up."""
    R, _ = camera(az)
    L = np.load(B + '/lapel.blend.lapels.npz')
    pts, t, d, e, cr = L['pts2'], L['t2'], L['d2'], L['edge2'], L['crease2']
    chain = [cr[:1]]
    if 'inner2' in L.files:
        mi = L['inner2'] & ~e; chain.append(pts[mi][np.argsort(d[mi])])
    chain.append(pts[e][np.argsort(t[e])])
    chain.append(cr[::-1])
    W = np.vstack(chain)
    P2 = np.column_stack([W @ R, W[:, 2]])
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P2, axis=0), axis=1) > 1e-4]); P2 = P2[keep]
    kP = np.argmax(np.where(e, d, -1)); L0 = pts[e][np.argmax(t[e])]
    unit = dict(L0=np.array([L0 @ R, L0[2]]), P=np.array([pts[kP] @ R, pts[kP][2]]))
    return P2, unit


def zbuffer(parts, az, box, res):
    """Visible part ids on a grid over box (x0, x1, y0, y1) in projected units."""
    R, D = camera(az)
    x0, x1, y0, y1 = box; W = int(res); H = int(round(res * (y1 - y0) / (x1 - x0)))
    depth = np.full((H, W), -np.inf); ids = np.full((H, W), -1, int)
    for pid, (V, T) in enumerate(parts):
        X = (V @ R - x0) / (x1 - x0) * W; Y = (y1 - V[:, 2]) / (y1 - y0) * H; Z = V @ D
        tx, ty = X[T], Y[T]
        ok = (tx.max(1) >= 0) & (tx.min(1) < W) & (ty.max(1) >= 0) & (ty.min(1) < H)
        for tri in T[ok]:
            xs, ys, zs = X[tri], Y[tri], Z[tri]
            i0, i1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), W - 1)
            j0, j1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), H - 1)
            if i1 < i0 or j1 < j0:
                continue
            gx, gy = np.meshgrid(np.arange(i0, i1 + 1) + 0.5, np.arange(j0, j1 + 1) + 0.5)
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-12:
                continue
            a = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
            b = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
            c = 1 - a - b
            ins = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
            if not ins.any():
                continue
            z = a * zs[0] + b * zs[1] + c * zs[2]
            sub = depth[j0:j1 + 1, i0:i1 + 1]; win = ins & (z > sub)
            sub[win] = z[win]; ids[j0:j1 + 1, i0:i1 + 1][win] = pid
    return ids


def collar_outline(B, az, unit):
    """The outline of the visible collar patch that holds the notch, projected (x right, y up)."""
    import re
    from scipy import ndimage
    import contourpy
    M = np.load(B + '/build.npz', allow_pickle=True)
    R, _ = camera(az)
    lg = open(B + '/collar.log').read()
    m = re.search(r'front 2: notch G at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', lg)
    G3 = np.array([float(v) for v in m.groups()]); G2 = np.array([G3 @ R, G3[2]])
    s = unit['P'][0] - unit['L0'][0]
    box = (unit['L0'][0] - 0.6 * s, unit['P'][0] + 1.0 * s, unit['L0'][1] - 0.6 * s, unit['L0'][1] + 1.6 * s)
    res = 480
    # the photo's neck opening is filled by the scarf, which hides the collar there: the collar inside her left roll
    # top (x below it, across the throat and the far side) is drawn as an occluder of its own, not as collar front
    L = np.load(B + '/lapel.blend.lapels.npz'); xcut = float(L['crease2'][0][0])
    CX, CT = M['collar_X'], M['collar_T']; far = CX[CT].mean(1)[:, 0] < xcut
    ids = zbuffer([(M['body_X'], M['body_T']), (CX, CT[~far]), (CX, CT[far])], az, box, res)
    H, W = ids.shape
    lab, n = ndimage.label(ids == 1)
    gi = (G2[0] - box[0]) / (box[1] - box[0]) * W; gj = (box[3] - G2[1]) / (box[3] - box[2]) * H
    # the patch nearest the notch (the notch lies on its edge)
    yy, xx = np.nonzero(lab > 0)
    if len(yy) == 0:
        return None, box, ids
    k = np.argmin((xx + 0.5 - gi) ** 2 + (yy + 0.5 - gj) ** 2); mask = lab == lab[yy[k], xx[k]]
    pad = np.pad(mask.astype(float), 1)
    lines = contourpy.contour_generator(z=pad).lines(0.5)
    Ln = max(lines, key=len) - 1.0          # (the outline of the patch; px, x right, y down)
    on_border = (Ln[:, 0] < 1) | (Ln[:, 0] > W - 2) | (Ln[:, 1] < 1) | (Ln[:, 1] > H - 2)
    P2 = np.column_stack([box[0] + (Ln[:, 0] + 0.5) / W * (box[1] - box[0]), box[3] - (Ln[:, 1] + 0.5) / H * (box[3] - box[2])])
    return dict(P=P2, border=on_border, G=G2), box, ids


def to_unit(P, unit):
    s = unit['P'][0] - unit['L0'][0]
    return (np.asarray(P) - unit['L0']) / s


# ---------------------------------------------------------------- the check
def compare(name, ref, Pm, closed=True):
    Rn = np.array([photo_norm(p) for p in ref['pts']]); dr = describe(Rn, True)
    ci = corners(Pm, closed)
    Cm = Pm[ci]
    dm = describe(Cm, closed)
    out = dict(piece=name, ref_sides=dr['n'], model_sides=dm['n'], corners=Cm.tolist(), ref_corners=Rn.tolist(),
               ok=dr['n'] == dm['n'], mismatches=[])
    if dr['n'] != dm['n']:
        out['mismatches'].append('%s: %d sides in the model, %d in the photo' % (name, dm['n'], dr['n']))
        bows, radii = curve_measure(Pm, ci); out['curve'] = dict(bows=bows, radii=radii)
        for j, (cc, rr) in enumerate(zip(Cm, radii)):
            out['mismatches'].append('%s: corner %d at (%.2f, %.2f) turns %.0f deg on radius %.3f span' % (name, j, cc[0], cc[1], 180 - dm['angles'][j], rr))
        return out
    # match the model's corners to the photo's in order: the rotation (and direction) with the least corner distance
    k = len(Rn); best = None
    for sgn in (1, -1):
        for r in range(k):
            order = [(r + sgn * i) % k for i in range(k)]
            err = np.sum(np.linalg.norm(Cm[order] - Rn, axis=1))
            if best is None or err < best[0]:
                best = (err, order)
    Cm = Cm[best[1]]; dm = describe(Cm, True); out['corners'] = Cm.tolist()
    out['edges'] = []
    for i in range(k):
        dd = ddir(dm['dirs'][i], dr['dirs'][i]); sr = dm['share'][i] / dr['share'][i]
        out['edges'].append(dict(side=ref['sides'][i], dir_ref=dr['dirs'][i], dir_model=dm['dirs'][i], dir_off=dd,
                                 share_ref=dr['share'][i], share_model=dm['share'][i]))
        if abs(dd) > TOL['dir']:
            out['mismatches'].append('%s, %s: runs at %.0f deg, the photo %.0f (%+.0f)' % (name, ref['sides'][i], dm['dirs'][i], dr['dirs'][i], dd))
        if abs(sr - 1) > TOL['share']:
            out['mismatches'].append('%s, %s: %.0f%% of the outline, the photo %.0f%%' % (name, ref['sides'][i], 100 * dm['share'][i], 100 * dr['share'][i]))
    out['corner_angles'] = []
    for i in range(k):
        da = dm['angles'][i] - dr['angles'][i]
        out['corner_angles'].append(dict(corner=ref['names'][i], ref=dr['angles'][i], model=dm['angles'][i]))
        if abs(da) > TOL['corner']:
            out['mismatches'].append('%s, corner %s: %.0f deg, the photo %.0f (%+.0f)' % (name, ref['names'][i], dm['angles'][i], dr['angles'][i], da))
    # curvature: the model's sides' bows and corners' radii against what the photo shows where it can be measured
    ci_m = [ci[j] for j in best[1]]
    bows, radii = curve_measure(Pm, ci_m)
    out['curve'] = dict(bows=bows, radii=radii)
    for i in range(k):
        sd = ref['sides'][i]
        if sd in PHOTO_CURVE:
            pb = PHOTO_CURVE[sd]['bow']; mb = bows[i]
            if abs(mb[0] - pb[0]) > 0.04 or abs(mb[1] - pb[1]) > 0.04:
                out['mismatches'].append('%s, %s: bows %+.3f / %+.3f of its chord, the photo %+.3f / %+.3f' % (name, sd, mb[0], mb[1], pb[0], pb[1]))
        cn = ref['names'][i]
        if cn in PHOTO_CURVE:
            pr = PHOTO_CURVE[cn]['radius']
            if not (0.5 < radii[i] / pr < 2.0):
                out['mismatches'].append('%s, corner %s: rounded on radius %.3f span, the photo %.3f' % (name, cn, radii[i], pr))
    return out


def check(B, az=30.0):
    Pl, unit = lapel_outline(B, az)
    res = {'build': B, 'pieces': []}
    res['pieces'].append(compare('lapel', REF['lapel'], to_unit(Pl, unit)))
    res['lapel_outline'] = to_unit(Pl, unit).tolist()
    if os.path.exists(B + '/build.npz') and os.path.exists(B + '/collar.log'):
        co, box, _ = collar_outline(B, az, unit)
        if co is not None:
            Pc = to_unit(co['P'], unit)
            res['pieces'].append(compare('collar front', REF['collar front'], Pc))
            res['collar_outline'] = Pc.tolist()
            if co['border'].any():
                res['pieces'][-1]['mismatches'].append('collar front: the patch runs off the check window (%d of %d outline points)' % (co['border'].sum(), len(co['border'])))
    res['pass'] = all(p['ok'] for p in res['pieces'])
    return res


def draw(ax, res):
    """The photo's polygons (thin, orange) against the model's outline (grey) and its own corners (black)."""
    for name, ref in REF.items():
        Rn = np.array([photo_norm(p) for p in ref['pts']]); Rn = np.vstack([Rn, Rn[:1]])
        ax.plot(Rn[:, 0], Rn[:, 1], color='tab:orange', lw=2.5, alpha=0.8)
    for key in ('lapel_outline', 'collar_outline'):
        if key in res:
            P = np.array(res[key]); ax.plot(P[:, 0], P[:, 1], color='0.6', lw=1)
    for p in res['pieces']:
        C = np.array(p['corners']); C = np.vstack([C, C[:1]])
        ax.plot(C[:, 0], C[:, 1], 'k-o', ms=4, lw=1.2)
    ax.set_aspect('equal'); ax.set_xlim(-0.6, 1.8); ax.set_ylim(-0.6, 1.7)
    t = ['%s %d/%d sides' % (p['piece'], p['model_sides'], p['ref_sides']) for p in res['pieces']]
    ax.set_title('polygon check %s: %s' % ('PASS' if res['pass'] else 'FAIL', ', '.join(t)), fontsize=10)


def report(res):
    lines = ['polycheck %s: %s' % (res['build'], 'PASS' if res['pass'] else 'FAIL')]
    for p in res['pieces']:
        lines.append('   %-13s sides: model %d, photo %d%s' % (p['piece'], p['model_sides'], p['ref_sides'], '' if p['ok'] else '   <- side count differs'))
        for e in p.get('edges', []):
            lines.append('      %-24s dir %6.0f (photo %6.0f, %+4.0f)   share %3.0f%% (photo %3.0f%%)' % (e['side'], e['dir_model'], e['dir_ref'], e['dir_off'], 100 * e['share_model'], 100 * e['share_ref']))
        if 'curve' in p and 'edges' in p:
            for e, b in zip(p['edges'], p['curve']['bows']):
                lines.append('      %-24s bow %+.3f / %+.3f of its chord' % (e['side'], b[0], b[1]))
            for c, r in zip(p['corner_angles'], p['curve']['radii']):
                c['radius'] = r
        for c in p.get('corner_angles', []):
            lines.append('      corner %-3s %5.0f deg (photo %5.0f)%s' % (c['corner'], c['model'], c['ref'], '   radius %.3f span' % c['radius'] if 'radius' in c else ''))
    mm = [m for p in res['pieces'] for m in p['mismatches']]
    lines += ['   mismatch: ' + m for m in mm] if mm else ['   no mismatched edge']
    return '\n'.join(lines)


if __name__ == '__main__':
    args = sys.argv[1:]; az = 30.0; png = js = None
    if '--az' in args: i = args.index('--az'); az = float(args[i + 1]); del args[i:i + 2]
    if '--png' in args: i = args.index('--png'); png = args[i + 1]; del args[i:i + 2]
    if '--json' in args: i = args.index('--json'); js = args[i + 1]; del args[i:i + 2]
    if '--photo' in args:      # remeasure PHOTO_CURVE on the photo
        P = np.array([photo_norm(p) for p in photo_collar()]); Rn = np.array([photo_norm(p) for p in REF['collar front']['pts']])
        iG, iT, iK = [int(np.argmin(np.linalg.norm(P - q, axis=1))) for q in Rn[:3]]       # (C lies on the dark coat)
        if not (iK < iT < iG): P = P[::-1]; iG, iT, iK = len(P) - 1 - iG, len(P) - 1 - iT, len(P) - 1 - iK
        bows, radii = curve_measure(P, [iK, iT, iG])        # (the outline runs K, T, G: the sides K-T and T-G)
        bo = (-bows[0][1], -bows[0][0])                      # (the outer edge walked T to K, as the check walks it)
        bf = (-bows[1][1], -bows[1][0])
        print('photo collar: front edge G-T bow %+.3f / %+.3f, outer edge T-K bow %+.3f / %+.3f; radius at T %.3f, at K %.3f span'
              % (*bf, *bo, radii[1], radii[0])); sys.exit(0)
    allres = []
    for B in args:
        r = check(B.rstrip('/'), az); allres.append(r); print(report(r))
    if js:
        json.dump(allres if len(allres) > 1 else allres[0], open(js, 'w'), indent=1)
    if png:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(allres), figsize=(6 * len(allres), 6), squeeze=False)
        for ax, r in zip(axs.flat, allres):
            draw(ax, r); ax.set_xlabel(r['build'].split('/')[-1])
        plt.tight_layout(); plt.savefig(png, dpi=80)
    sys.exit(0 if all(r['pass'] for r in allres) else 1)
