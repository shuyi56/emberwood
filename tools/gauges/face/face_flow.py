"""Face flow gauge: the brow-to-mouth lines a sculptor reads, traced on a reference photo and on a Face Creator head
from the same (front, orthographic) camera, in one unit, with the gaps listed.

What it is for
    John's ask (2026-11-24): "total facial flow", brow down to mouth, as sculptors build it. On his reference he drew
    the lines in yellow. They are the standard plane breaks of that region:

    1. the brow-tip line (the dorsal aesthetic line): the lit front plane's edge, from the brow's underside, curving in
       past the inner corner of the eye and down the side of the bridge to the tip-defining point. Rhinoplasty texts
       call it "two gently diverging curves" from the medial brow to the tip-defining points, an "hourglass", "slightly
       wider at the radix, narrow at the middle third and then widen at the tip", "subtle, unbroken, and fluid"
       (Park 2011; Cangello; Hilinski; Rohrich's 10-7-5 analysis: its width should match the tip-defining points').
    2. the outer line: the upper lid's crease, running in past the inner corner and down where the nose's side wall
       meets the cheek (the nasofacial groove), round the wing's crease. Bridgman: on the lid "three planes wedge
       together: nose bridge to eye; brow to cheekbone"; Proko's head planes: the eye sockets "angled up from the
       outside to the keystone".
    3. the philtrum's column and the upper lip's vermilion line out to the corner.

    Line 1 is the edge of the region whose surface faces the viewer within THETA (45 degrees: where face_lines.py
    (PR #75) puts the dorsal aesthetic lines). Line 2 is the concave valley next to it (the deepest bend of the depth
    across the row, or down the column for the lid crease). Line 3 is the convex ridge between the nose's sill and the
    lip, and the vermilion's top edge.

In and out
    ref   <photo> <sheet.measure.json> --out DIR    the photo's yellow strokes (auto-traced) and its landmarks (a
          refmeasure sheet: pupil_l/r, en_l/r, ex_l/r, sn, tdp_l/r, al_l/r, peak_l/r, ch_l/r, sto)
    model <dump.json> --out DIR                      a head dumped by tools/face-check/dump-heads.cjs (PR #75)
    compare <ref.flow.json> <model.flow.json> [more models] --out DIR
    Each writes <name>.flow.json (lines and landmarks in the unit frame); compare writes flow-gaps.md, flow-gaps.json
    and flow-sheet.png (the model's clay front with both sets of lines, and the photo with both).

    The unit frame: origin midway between the pupils, x along the pupil line (to image right), y down, one unit the
    distance between the pupils. The photo is levelled on its pupils (John's reference has a 5.6 degree roll). Both
    sides are measured and their mean is compared, which halves the effect of a small turn of the head.

Run
    python3 tools/gauges/face/face_flow.py ref ref.png ref.measure.json --out flow
    python3 tools/gauges/face/face_flow.py model dump/Ranger.json --out flow
    python3 tools/gauges/face/face_flow.py compare flow/ref.flow.json flow/Ranger.flow.json --out flow
    Needs numpy, scipy, Pillow; the photo tracing also scikit-image.

Trouble looks like
    - the brow-tip line pinched in to a knife-edge bridge and then flaring to a wide tip (waist / tip ratio far
      under 1), or no waist at all;
    - the outer line's half-width at the bridge's middle far inside the photo's (a fin of a nose standing on the face:
      the side walls do not run out into the cheek);
    - a line with many polygon sides at the 1 mm tolerance or a sharp turn (kink): the plane change is a step, not a
      flow;
    - mouth corners, wings or philtrum columns off their plumb lines (corners under the medial limbus, wings under the
      inner corners: Hilinski, Cangello, plasticsurgerykey).

Use it when
    Any change to the face between brow and mouth (the nose solid, the sockets, the brow, the cheeks, the lips), before
    and after, with the face_lines check (PR #75) for the profile angles this front view cannot see.

Grounded in
    Park 2011, Fundamental Principles in Aesthetic Rhinoplasty (e-ceo.org, doi 10.3342/ceo.2011.4.2.55); Brito,
    Avashia, Rohrich 2020 (PMC7159929); Cangello and Hilinski nasal analysis pages; plasticsurgerykey "Nasal analysis
    and anatomy" (alar base = intercanthal width, bony base 80 % of it, radix between lash line and lid crease);
    Bridgman, Constructive Anatomy (the head's planes); Proko, Planes and forms of the head; Anatomy for Sculptors,
    Forming the head. The rasteriser is face_lines.py's (PR #75), copied so this runs on main.
"""
import argparse
import base64
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

THETA = 35.0          # degrees from facing the viewer: the edge of the lit front plane (EXPERIMENTAL: between face_lines'
                      # 30 (tip-defining points) and 45 (dorsal lines); 45 never reaches the brow's underside on our heads)
MM_PER_UNIT = 219.0 / 8.6      # the creator's head: HH 8.6 units is 219 mm (NOSE_LIFE.head_h)


# ------------------------------------------------------------------------------------------------ shared

def level_frame(pl, pr):
    """The unit frame from two pupils (image coords, y down): returns f(points) -> unit coords."""
    pl, pr = np.asarray(pl, float), np.asarray(pr, float)
    o = 0.5 * (pl + pr)
    d = pr - pl
    u = np.linalg.norm(d)
    c, s = d / u

    def f(P):
        P = np.asarray(P, float) - o
        return np.stack([c * P[..., 0] + s * P[..., 1], -s * P[..., 0] + c * P[..., 1]], -1) / u
    return f, math.degrees(math.atan2(d[1], d[0])), u


def order_path(pts):
    """Order a thin stroke's pixels into one polyline: the longest shortest-path through the 8-neighbour graph."""
    from collections import deque
    S = {tuple(p) for p in pts}
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def bfs(s):
        prev = {s: None}
        q = deque([s])
        last = s
        while q:
            p = q.popleft()
            last = p
            for d in nb:
                n = (p[0] + d[0], p[1] + d[1])
                if n in S and n not in prev:
                    prev[n] = p
                    q.append(n)
        return last, prev
    a, _ = bfs(next(iter(S)))
    b, prev = bfs(a)
    path = []
    p = b
    while p is not None:
        path.append(p)
        p = prev[p]
    return np.array(path, float)


def resample(C, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if d[-1] < step:
        return C
    s = np.arange(0, d[-1], step)
    return np.stack([np.interp(s, d, C[:, j]) for j in range(C.shape[1])], 1)


def smooth_line(C, sig):
    if len(C) < 5:
        return C
    return np.stack([ndimage.gaussian_filter1d(C[:, j], sig, mode='nearest') for j in range(2)], 1)


def dp_sides(C, tol):
    """Douglas-Peucker: how many straight sides the line needs to stay within tol (the polygon side count)."""
    def rec(i, j):
        a, b = C[i], C[j]
        ab = b - a
        L = np.linalg.norm(ab)
        if j <= i + 1:
            return [i, j]
        P = C[i + 1:j] - a
        dist = np.abs(ab[0] * P[:, 1] - ab[1] * P[:, 0]) / L if L > 1e-12 else np.linalg.norm(P, axis=1)
        k = int(np.argmax(dist))
        if dist[k] <= tol:
            return [i, j]
        m = i + 1 + k
        return rec(i, m)[:-1] + rec(m, j)
    return len(rec(0, len(C) - 1)) - 1


def flow_stats(C, unit_mm):
    """Curvature checks on a line in unit coords: sides at 1 mm, the sharpest turn over 1.5 mm, wiggles."""
    if C is None or len(C) < 8:
        return None
    step = 0.25 / unit_mm      # 0.25 mm
    R = smooth_line(resample(C, step), 2)
    T = np.diff(R, axis=0)
    ang = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    w = max(1, int(round(1.5 / 0.25)))
    turn = np.abs(ang[w:] - ang[:-w]) if len(ang) > w else np.array([0.0])
    k = ndimage.gaussian_filter1d(np.diff(ang), 6)
    big = np.abs(k) > np.radians(0.5)
    sg = np.sign(k[big])
    wig = int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0
    return {'length_mm': float(np.sum(np.linalg.norm(T, axis=1)) * unit_mm),
            'sides_1mm': dp_sides(R, 1.0 / unit_mm),
            'max_turn_deg_per_1.5mm': float(np.degrees(turn.max())),
            'curvature_sign_changes': wig}


def half_at_rows(lines, ys):
    """Mean |x| of a pair of side lines at each level y (the first crossing, coming down), unit coords."""
    out = []
    for y in ys:
        xs = []
        for C in lines:
            if C is None:
                continue
            i = np.nonzero(np.diff(np.sign(C[:, 1] - y)) != 0)[0]
            if not len(i):
                continue
            # the crossing nearest the midline (the nose's line, not the brow's far end)
            best = None
            for k in i:
                t = (y - C[k, 1]) / (C[k + 1, 1] - C[k, 1])
                x = abs(C[k, 0] + t * (C[k + 1, 0] - C[k, 0]))
                best = x if best is None else min(best, x)
            xs.append(best)
        out.append(float(np.mean(xs)) if xs else None)
    return out


def y_at_cols(lines, xs, upper=True):
    """Mean y of a pair of side lines at |x| = each value (the highest crossing if upper)."""
    out = []
    for x in xs:
        ys = []
        for C in lines:
            if C is None:
                continue
            ax = np.abs(C[:, 0])
            i = np.nonzero(np.diff(np.sign(ax - x)) != 0)[0]
            if not len(i):
                continue
            cand = []
            for k in i:
                t = (x - ax[k]) / (ax[k + 1] - ax[k])
                cand.append(C[k, 1] + t * (C[k + 1, 1] - C[k, 1]))
            ys.append(min(cand) if upper else max(cand))
        out.append(float(np.mean(ys)) if ys else None)
    return out


# ------------------------------------------------------------------------------------------------ the photo

def trace_ref(photo, sheet_path, out):
    from skimage.measure import label
    from skimage.morphology import skeletonize, dilation, disk
    sh = json.load(open(sheet_path))
    Lm = sh['landmarks']
    f, roll, ipd_px = level_frame(Lm['pupil_l'], Lm['pupil_r'])
    a = np.asarray(Image.open(photo).convert('RGB')).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (R > 200) & (G > 150) & (B < 90) & (R - B > 140)      # John's yellow marker
    m = dilation(m, disk(1))
    sk = skeletonize(m)
    lab = label(sk, connectivity=2)
    strokes = []
    for k in range(1, lab.max() + 1):
        yy, xx = np.nonzero(lab == k)
        if len(xx) < 40:
            continue
        P = order_path(np.stack([xx, yy], 1))
        strokes.append(f(P))
    # name the strokes: the brow-tip lines start above the pupils (y < -0.25) and end at the tip; the outer lines start
    # at the lid crease near the pupil line; the philtrum and lip strokes lie below the subnasale
    sn = f(Lm['sn'])
    lines = {'brow_tip': [], 'outer': [], 'philtrum_lip': []}
    for C in strokes:
        top = C[:, 1].min()
        if C[:, 1].max() > sn[1] + 0.05 and top > sn[1] - 0.15:
            lines['philtrum_lip'].append(C)
        elif top < -0.25:
            lines['brow_tip'].append(C)
        else:
            # (the left outer line was drawn into the philtrum stroke on John's photo: cut it at the subnasale)
            lines['outer'].append(C[C[:, 1] < sn[1] + 0.02])
            rest = C[C[:, 1] >= sn[1] + 0.02]
            if len(rest) > 20 and np.abs(rest[:, 0]).min() < 0.2:
                lines['philtrum_lip'].append(rest)
    lm = {k: f(v).tolist() for k, v in Lm.items()}
    res = {'kind': 'photo', 'name': os.path.splitext(os.path.basename(photo))[0], 'image': os.path.abspath(photo),
           'roll_deg': roll, 'unit_px': ipd_px, 'unit_mm': None, 'landmarks': lm,
           'lines': {k: [C.tolist() for C in v] for k, v in lines.items()}}
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, res['name'] + '.flow.json')
    json.dump(res, open(p, 'w'))
    print(p, {k: len(v) for k, v in lines.items()}, 'roll %.1f deg' % roll)
    return res


# ------------------------------------------------------------------------------------------------ the model

def load_dump(path):
    d = json.load(open(path))
    out = {'face': d['face'], 'head': None, 'eyes': [], 'params': d.get('P', {})}
    for m in d['meshes']:
        V = np.frombuffer(base64.b64decode(m['P']), np.float32).reshape(-1, 3).astype(np.float64)
        F = np.frombuffer(base64.b64decode(m['I']), np.uint32).reshape(-1, 3).astype(np.int64)
        if m['role'] == 'head':
            out['head'] = (V, F)
        elif m['role'] == 'eye':
            out['eyes'].append((V, F))
    return out


def zbuffer(V, F, lo, hi, res):
    """Orthographic z-buffer looking down -z (three.js: x right, y up, z toward the viewer). Image rows go down."""
    W = int(math.ceil((hi[0] - lo[0]) / res))
    H = int(math.ceil((hi[1] - lo[1]) / res))
    zb = np.full(W * H, -np.inf)
    px = (V[:, 0] - lo[0]) / res
    py = (hi[1] - V[:, 1]) / res
    x, y, z = px[F], py[F], V[F, 2]
    x0 = np.floor(x.min(1)).astype(np.int64)
    y0 = np.floor(y.min(1)).astype(np.int64)
    x1 = np.ceil(x.max(1)).astype(np.int64)
    y1 = np.ceil(y.max(1)).astype(np.int64)
    keep = (x1 >= 0) & (y1 >= 0) & (x0 < W) & (y0 < H) & (z.max(1) > -1e9)
    x, y, z, x0, y0, x1, y1 = x[keep], y[keep], z[keep], x0[keep], y0[keep], x1[keep], y1[keep]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    ok = np.abs(det) > 1e-12
    size = np.maximum(x1 - x0, y1 - y0)
    groups = [np.nonzero(ok & (size >= a) & (size < b))[0] for a, b in ((0, 4), (4, 12), (12, 40))]
    groups += [np.array([k]) for k in np.nonzero(ok & (size >= 40))[0]]      # (a few long ones, one at a time)
    for sel in groups:
        if not len(sel):
            continue
        K = int(size[sel].max()) + 1
        for oy in range(K):
            for ox in range(K):
                cx = x0[sel] + ox
                cy = y0[sel] + oy
                v = (cx <= x1[sel]) & (cy <= y1[sel]) & (cx >= 0) & (cy >= 0) & (cx < W) & (cy < H)
                g = sel[v]
                if not len(g):
                    continue
                qx, qy = cx[v] + 0.5, cy[v] + 0.5
                xa, ya, d = x[g], y[g], det[g]
                w1 = ((qx - xa[:, 0]) * (ya[:, 2] - ya[:, 0]) - (xa[:, 2] - xa[:, 0]) * (qy - ya[:, 0])) / d
                w2 = ((xa[:, 1] - xa[:, 0]) * (qy - ya[:, 0]) - (qx - xa[:, 0]) * (ya[:, 1] - ya[:, 0])) / d
                w0 = 1 - w1 - w2
                ins = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
                g = g[ins]
                zz = w0[ins] * z[g, 0] + w1[ins] * z[g, 1] + w2[ins] * z[g, 2]
                flat = cy[v][ins] * W + cx[v][ins]
                o = np.argsort(zz)
                flat, zz = flat[o], zz[o]
                b = zz > zb[flat]
                zb[flat[b]] = zz[b]
    return zb.reshape(H, W)


def contour_through(A, level, seed, mask=None):
    """The iso-line of A at level that passes nearest the seed (row, col): one polyline of (row, col)."""
    from skimage.measure import find_contours
    Cs = find_contours(A, level, mask=mask)
    best, bd = None, 1e18
    for C in Cs:
        d = np.min(np.sum((C - seed) ** 2, 1))
        if d < bd:
            best, bd = C, d
    return best


def trace_model(dump, out, theta=THETA):
    face = load_dump(dump)
    V, F = face['head']
    # pupils: each eyeball's front-most point
    pup = []
    for (Ve, Fe) in face['eyes']:
        k = np.argmax(Ve[:, 2])
        pup.append(Ve[k])
    pup.sort(key=lambda p: p[0])
    ipd = float(np.linalg.norm(pup[1][:2] - pup[0][:2]))
    unit_mm = ipd * MM_PER_UNIT
    res = ipd / 700.0
    cx, cy = 0.5 * (pup[0][0] + pup[1][0]), 0.5 * (pup[0][1] + pup[1][1])
    lo = (cx - 1.0 * ipd, cy - 1.25 * ipd)
    hi = (cx + 1.0 * ipd, cy + 1.05 * ipd)
    zb = zbuffer(V, F, lo, hi, res)
    allE = [zbuffer(Ve, Fe, lo, hi, res) for (Ve, Fe) in face['eyes']]
    ze = np.maximum.reduce(allE)
    eye_vis = ze > zb + 1e-6
    H, W = zb.shape
    fin = np.isfinite(zb)
    zf = np.where(fin, zb, np.nanmin(np.where(fin, zb, np.nan)))
    mm = 1.0 / MM_PER_UNIT      # a millimetre in model units
    zs = ndimage.gaussian_filter(zf, 0.5 * mm / res)       # (0.5 mm: the photo's softness at this scale)
    gy, gx = np.gradient(zs, res)
    # (rows go down the image, so the surface's slope up is -gy)
    nz = 1.0 / np.sqrt(1 + gx * gx + gy * gy)
    A = np.degrees(np.arccos(nz))
    Ax = np.degrees(np.arctan(np.abs(gx)))      # (the turn across: the side walls; the bridge's slope in profile left out)
    lap = ndimage.gaussian_filter(zf, 1.0 * mm / res)
    K = ndimage.laplace(lap) / (res * res)           # > 0 concave (a valley seen from the front)

    def to_unit(rc):
        x = lo[0] + (rc[:, 1] + 0.5) * res
        y = hi[1] - (rc[:, 0] + 0.5) * res
        return np.stack([(x - cx) / ipd, -(y - cy) / ipd], 1)

    def rc_of(u):
        return np.array([(hi[1] - cy + u[1] * ipd) / res - 0.5, (u[0] * ipd + cx - lo[0]) / res - 0.5])

    # landmarks from the front buffer: inner and outer eye corners (the visible ball's ends)
    lm = {'pupil_l': [-0.5, 0.0], 'pupil_r': [0.5, 0.0]}
    yy, xx = np.nonzero(eye_vis)
    U = to_unit(np.stack([yy, xx], 1))
    for sgn, nm in ((-1, 'l'), (1, 'r')):
        s = U[np.sign(U[:, 0]) == sgn]
        if len(s):
            lm['en_' + nm] = s[np.argmin(np.abs(s[:, 0]))].tolist()
            lm['ex_' + nm] = s[np.argmax(np.abs(s[:, 0]))].tolist()
    # the midline profile: tip (most forward), subnasale (the turn under the nose), the lips
    mid = int(round((cx - lo[0]) / res - 0.5))
    col = zs[:, mid]
    rows_u = to_unit(np.stack([np.arange(H), np.full(H, mid)], 1))[:, 1]
    band = (rows_u > 0.1) & (rows_u < 0.9)
    r_prn = int(np.nonzero(band)[0][np.argmax(col[band])])
    # down the midline from the tip: the subnasale (the first hollow), the upper lip (the next bump), the stomion (the
    # next hollow) and the lower lip (the next bump), on the profile smoothed over 0.2 mm
    cs_ = ndimage.gaussian_filter1d(col, 0.2 * mm / res)
    seg = np.arange(r_prn + 2, min(H - 2, r_prn + int(0.9 * ipd / res)))
    ext = []
    for r in seg:
        if cs_[r] < cs_[r - 1] and cs_[r] <= cs_[r + 1]:
            ext.append(('min', r))
        elif cs_[r] > cs_[r - 1] and cs_[r] >= cs_[r + 1]:
            ext.append(('max', r))
    # (drop wiggles under 0.15 mm)
    clean = []
    for e in ext:
        if clean and abs(cs_[e[1]] - cs_[clean[-1][1]]) < 0.15 * mm:
            continue
        if clean and clean[-1][0] == e[0]:
            if (e[0] == 'min') == (cs_[e[1]] < cs_[clean[-1][1]]):
                clean[-1] = e
            continue
        clean.append(e)
    if os.environ.get('FLOW_DEBUG'):
        print([(k, round(float(rows_u[r]), 3), round(float(cs_[r] / mm), 2)) for k, r in clean])
    mins = [r for k, r in clean if k == 'min']
    maxs = [r for k, r in clean if k == 'max']
    r_ls = [r for r in maxs if r > mins[0]][0]
    # (the subnasale is where the columella turns into the lip: the sharpest bend between the tip and the upper lip,
    # not the deepest point, which on a concave upper lip lies lower)
    k_ = max(1, int(0.5 * mm / res))
    seg = np.arange(r_prn + k_, r_ls - k_)
    r_sn = int(seg[np.argmax(cs_[seg + k_] - 2 * cs_[seg] + cs_[seg - k_])])
    r_sto = [r for r in mins if r > r_ls][0]
    r_li = [r for r in maxs if r > r_sto][0]
    lm['prn'] = to_unit(np.array([[r_prn, mid]]))[0].tolist()
    lm['sn'] = to_unit(np.array([[r_sn, mid]]))[0].tolist()
    if True:
        lm['ls'] = to_unit(np.array([[r_ls, mid]]))[0].tolist()
        lm['li'] = to_unit(np.array([[r_li, mid]]))[0].tolist()
        lm['sto'] = to_unit(np.array([[r_sto, mid]]))[0].tolist()
        # the mouth's corners: going out along the parting line (its row followed column by column), the last column
        # where the parting is still a groove at least 0.3 mm deep against the lips above and below
        r = r_sto
        best = {}
        for sgn, nm in ((-1, 'l'), (1, 'r')):
            r = r_sto
            last = None
            for cc in range(mid, mid + sgn * int(0.6 * ipd / res), sgn):
                win = np.arange(max(0, r - 6), min(H, r + 7))
                rr = int(win[np.argmin(zs[win, cc])])
                w2 = int(1.6 * mm / res)
                up = zs[max(0, rr - w2):rr, cc].max() if rr > 0 else zs[rr, cc]
                dn = zs[rr + 1:rr + w2 + 1, cc].max() if rr + 1 < H else zs[rr, cc]
                if min(up, dn) - zs[rr, cc] < 0.3 * mm:
                    break
                last = (rr, cc)
                r = rr
            if last:
                lm['ch_' + nm] = to_unit(np.array([last]))[0].tolist()
    # 1. the brow-tip line: the edge of the front-facing plane through the bridge's side, on each side
    lines = {'brow_tip': [], 'outer': [], 'philtrum_lip': []}
    zb15 = ndimage.gaussian_filter(zf, 1.5 * mm / res)
    Kyy = -np.gradient(np.gradient(zb15, res, axis=0), res, axis=0)      # > 0 a convex bend down a column
    brow_k = []
    # (row by row from the tip's level up to above the inner corners, from the midline out: the first point that
    # turns THETA from the viewer; then column by column out along the brow: coming down from the forehead, the first
    # point whose surface turns THETA and faces down, the brow's underside)
    for sgn in (-1, 1):
        pts = []
        for yu in np.arange(lm['prn'][1], -0.12, -0.003):
            r = int(round(rc_of([0, yu])[0]))
            cs = np.arange(mid, mid + sgn * int(0.45 * ipd / res), sgn)
            hit = np.nonzero((Ax[r, cs] > theta) | eye_vis[r, cs])[0]
            if not len(hit):
                break
            pts.append([(cs[hit[0]] - mid) * res / ipd, yu])
        brow = []
        for xu in np.arange(0.1, 0.8, 0.004):
            c_ = int(round(rc_of([sgn * xu, 0])[1]))
            rs = np.array([int(round(rc_of([0, yu])[0])) for yu in np.arange(-0.6, -0.16, 0.002)])
            ok = ~eye_vis[rs, c_]
            # (the brow's plane break: the sharpest convex bend down the column, where the brow's front turns under
            # into the socket's roof; no threshold, so a soft brow still has one, only weaker)
            kk = np.where(ok, Kyy[rs, c_], -np.inf)
            q = int(np.argmax(kk))
            if not np.isfinite(kk[q]) or kk[q] <= 0:
                continue
            brow.append([sgn * xu, -0.6 + 0.002 * q])
            brow_k.append(float(kk[q]) * mm)      # (1/mm: the bend's sharpness)
        # (top to bottom: the brow from its outer end in, then down the nose)
        line = brow[::-1] + pts[::-1]
        lines['brow_tip'].append(np.array(line) if len(line) > 4 else None)
    # 2. the outer line: the valley beside the nose, row by row from the inner corner down to the wing's crease, and
    # the lid crease, column by column from the inner corner out
    for sgn in (-1, 1):
        bt = lines['brow_tip'][0 if sgn < 0 else 1]
        pts = []
        y_en = np.mean([lm['en_l'][1], lm['en_r'][1]])
        for yu in np.arange(y_en, lm['sn'][1] + 0.005, 0.004):
            r = int(round(rc_of([0, yu])[0]))
            w_in = half_at_rows([bt], [yu])[0] if bt is not None else 0.05
            w_in = (w_in or 0.05) + 0.6 * mm / ipd
            cs = np.array([int(round(rc_of([sgn * xu, yu])[1])) for xu in np.arange(w_in, 0.5, 0.002)])
            cs = cs[(cs >= 0) & (cs < W)]
            if not len(cs):
                continue
            kk = K[r, cs]
            q = int(np.argmax(kk))
            if kk[q] <= 0:
                continue
            pts.append([sgn * (w_in + 0.002 * q), yu])
        crease = []
        en_x = abs(lm['en_' + ('l' if sgn < 0 else 'r')][0])
        ex_x = abs(lm['ex_' + ('l' if sgn < 0 else 'r')][0])
        for xu in np.arange(en_x, ex_x, 0.004):
            c_ = int(round(rc_of([sgn * xu, 0])[1]))
            rs = np.array([int(round(rc_of([0, yu])[0])) for yu in np.arange(-0.36, -0.07, 0.002)])
            ok = ~eye_vis[rs, c_] & ~eye_vis[np.minimum(rs + 2, H - 1), c_]
            if not ok.any():
                continue
            kk = np.where(ok, K[rs, c_], -np.inf)
            q = int(np.argmax(kk))
            if kk[q] <= 0:
                continue
            crease.append([sgn * xu, -0.36 + 0.002 * q])
        line = (crease[::-1] if crease else []) + pts
        lines['outer'].append(np.array(line) if len(line) > 4 else None)
    # 3. the philtrum's columns (the ridge beside the groove, from the sill to the bow) and the vermilion's top edge
    for sgn in (-1, 1):
        pts = []
        if 'ls' in lm:
            for yu in np.arange(lm['sn'][1] + 0.02, lm['ls'][1], 0.004):
                r = int(round(rc_of([0, yu])[0]))
                cs = np.array([int(round(rc_of([sgn * xu, yu])[1])) for xu in np.arange(0.5 * mm / ipd, 9 * mm / ipd, 0.002)])
                kk = -K[r, cs]
                q = int(np.argmax(kk))
                pts.append([sgn * (0.5 * mm / ipd + 0.002 * q), yu])
            # the vermilion border: column by column out to the corner, the sharpest convex turn above the parting
            ch = lm.get('ch_' + ('l' if sgn < 0 else 'r'))
            if ch:
                x0 = abs(pts[-1][0]) if pts else 0.08
                for xu in np.arange(x0, abs(ch[0]) + 0.004, 0.004):
                    c_ = int(round(rc_of([sgn * xu, 0])[1]))
                    rs = np.array([int(round(rc_of([0, yu])[0])) for yu in np.arange(lm['ls'][1] - 0.08, lm['sto'][1] - 0.005, 0.002)])
                    kk = -K[rs, c_]
                    q = int(np.argmax(kk))
                    pts.append([sgn * xu, lm['ls'][1] - 0.08 + 0.002 * q])
        lines['philtrum_lip'].append(np.array(pts) if len(pts) > 4 else None)
    # the peaks of the bow: where the philtrum's line meets the vermilion's
    for sgn, nm in ((-1, 'l'), (1, 'r')):
        C = lines['philtrum_lip'][0 if sgn < 0 else 1]
        if C is not None and 'ls' in lm:
            k = int(np.argmin(np.abs(C[:, 1] - lm['ls'][1])))
            lm['peak_' + nm] = C[k].tolist()
    # the wings (alare): the nose's widest point between the tip and the subnasale, over a cheek fitted either side
    if os.environ.get('FLOW_DEBUG'):
        print({k: np.round(v, 3).tolist() for k, v in lm.items()})
    rws = [int(round(rc_of([0, yu])[0])) for yu in np.arange(lm['prn'][1] - 0.05, lm['sn'][1] + 0.01, 0.003)]
    best = 0
    for r in rws:
        x_ = (np.arange(W) + 0.5) * res + lo[0] - cx
        ax = np.abs(x_)
        cheek = (ax > 0.42 * ipd) & (ax < 0.62 * ipd)
        cfit = np.polyfit(x_[cheek] ** 2, zs[r, cheek], 1)
        ex = zs[r] - np.polyval(cfit, x_ ** 2)
        m = ex > 0.8 * mm
        c0 = mid
        rr = c0
        while rr < W - 1 and m[rr + 1]:
            rr += 1
        ll = c0
        while ll > 0 and m[ll - 1]:
            ll -= 1
        hw = 0.5 * (rr - ll) * res / ipd
        if hw > best:
            best, r_al = hw, r
    yal = to_unit(np.array([[r_al, mid]]))[0][1]
    lm['al_l'], lm['al_r'] = [-best, yal], [best, yal]
    # the tip-defining points: on the tip's row, where the cross-section turns 30 degrees (face_lines' rule)
    r = int(round(rc_of([0, lm['prn'][1]])[0]))
    for sgn, nm in ((-1, 'l'), (1, 'r')):
        cs = np.arange(mid, mid + sgn * int(0.2 * ipd / res), sgn)
        k = cs[np.argmax(A[r, cs] > 30)]
        lm['tdp_' + nm] = [float((k - mid) * res / ipd), lm['prn'][1]]
    res_ = {'kind': 'model', 'name': face['face'], 'dump': os.path.abspath(dump), 'unit_mm': unit_mm, 'landmarks': lm,
            'lines': {k: [None if C is None else np.asarray(C).tolist() for C in v] for k, v in lines.items()},
            'theta': theta, 'brow_bend_per_mm_mean': float(np.mean(brow_k)) if brow_k else None}
    os.makedirs(out, exist_ok=True)
    # the clay front, for the sheet: lit from upper left like the photo, the eyes darker
    nx, ny = -gx / np.sqrt(1 + gx * gx + gy * gy), gy / np.sqrt(1 + gx * gx + gy * gy)
    l = np.array([-0.35, 0.45, 0.82])
    l /= np.linalg.norm(l)
    sh = np.clip(nx * l[0] + ny * l[1] + nz * l[2], 0, 1)
    v = 0.16 + 0.7 * sh + 0.14 * nz ** 4
    img = np.stack([v * 0.86, v * 0.8, v * 0.74], -1)
    img[eye_vis] *= 0.55
    img[~fin] = [0.12, 0.13, 0.15]
    png = os.path.join(out, face['face'] + '.clay.png')
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(png)
    res_['clay'] = os.path.abspath(png)
    res_['clay_frame'] = {'cx': cx, 'cy': cy, 'ipd': ipd, 'lo': lo, 'hi': hi, 'res': res}
    if os.environ.get('FLOW_DEBUG'):
        Image.fromarray(np.clip(A * 255 / 90, 0, 255).astype(np.uint8)).save(os.path.join(out, face['face'] + '.angle.png'))
        Image.fromarray(np.clip(128 + K * 0.02, 0, 255).astype(np.uint8)).save(os.path.join(out, face['face'] + '.K.png'))
    p = os.path.join(out, face['face'] + '.flow.json')
    json.dump(res_, open(p, 'w'))
    print(p, 'unit %.1f mm' % unit_mm)
    return res_


# ------------------------------------------------------------------------------------------------ compare

Y0 = 0.8      # (the sheet's panels show unit x -1..1, y -Y0..2-Y0)
LEVELS = [('pupil line', 0.0), ('a quarter down to the tip', 0.25), ('half way down', 0.5),
          ('three quarters down', 0.75)]


def measures(R):
    """The numbers compared, from one flow.json (unit = pupil distance unless named)."""
    lm = {k: np.array(v) for k, v in R['landmarks'].items()}
    L = {k: [None if C is None else np.array(C) for C in v] for k, v in R['lines'].items()}
    M = {}
    y_tip = np.mean([lm['tdp_l'][1], lm['tdp_r'][1]])
    for name, t in LEVELS:
        y = t * y_tip
        M['brow-tip half-width, ' + name] = half_at_rows(L['brow_tip'], [y])[0]
        M['outer line half-width, ' + name] = half_at_rows(L['outer'], [y])[0]
    # the hourglass: the brow-tip line's narrowest half-width between the pupils' line and the tip, and its height
    ys = np.linspace(-0.1, y_tip, 60)
    hw = half_at_rows(L['brow_tip'], ys)
    ok = [(h, y) for h, y in zip(hw, ys) if h is not None]
    if ok:
        h, y = min(ok)
        M['brow-tip waist half-width'] = h
        M['brow-tip waist height (down from pupils, in tip heights)'] = y / y_tip
        M['tip-defining points half-width'] = 0.5 * (abs(lm['tdp_l'][0]) + abs(lm['tdp_r'][0]))
        M['waist / tip-defining points (Rohrich: about 1)'] = h / M['tip-defining points half-width']
    # the brow's underside: the brow-tip line's height above the pupils at |x| 0.2, 0.3, 0.4
    for x in (0.2, 0.3, 0.4):
        v = y_at_cols(L['brow_tip'], [x])[0]
        M['brow line height at |x| %.1f' % x] = None if v is None else -v
    # the join: how far the brow's line ends from the top of the nose's line (0 when they run into each other)
    gaps = []
    for C in L['brow_tip']:
        if C is None:
            continue
        top = C[C[:, 1] > -0.13]
        brow = C[C[:, 1] <= -0.13]
        if len(top) and len(brow):
            a = top[np.argmin(top[:, 1])]
            gaps.append(float(np.min(np.linalg.norm(brow - a, axis=1))))
    if gaps:
        M['brow-to-nose join gap'] = float(np.mean(gaps))
    if R.get('brow_bend_per_mm_mean') is not None:
        M["brow's plane break, bend per mm (model only)"] = R['brow_bend_per_mm_mean']
    M['outer line at the bridge / alar half-width'] = (M['outer line half-width, half way down'] or 0) / (0.5 * (abs(lm['al_l'][0]) + abs(lm['al_r'][0])))
    M['alar width / intercanthal width (1.0: plasticsurgerykey)'] = (abs(lm['al_l'][0]) + abs(lm['al_r'][0])) / (abs(lm['en_l'][0]) + abs(lm['en_r'][0]))
    M['eye width / intercanthal width'] = 0.5 * (abs(lm['ex_l'][0] - lm['en_l'][0]) + abs(lm['ex_r'][0] - lm['en_r'][0])) / (abs(lm['en_l'][0]) + abs(lm['en_r'][0]))
    if 'ch_l' in lm and 'ch_r' in lm:
        M['mouth width / pupil distance'] = abs(lm['ch_r'][0] - lm['ch_l'][0])
        M['mouth width / alar width'] = abs(lm['ch_r'][0] - lm['ch_l'][0]) / (abs(lm['al_l'][0]) + abs(lm['al_r'][0]))
    if 'peak_l' in lm and 'peak_r' in lm:
        M['bow peaks apart / pupil distance'] = abs(lm['peak_r'][0] - lm['peak_l'][0])
    if 'sto' in lm:
        M['subnasale to stomion / pupil to subnasale'] = (lm['sto'][1] - lm['sn'][1]) / lm['sn'][1]
    M['pupils to subnasale'] = float(lm['sn'][1])
    # flow of each line (mean of the two sides)
    for k in ('brow_tip', 'outer'):
        st = [flow_stats(C, R['unit_mm'] or 62.0) for C in L[k] if C is not None]
        st = [s for s in st if s]
        if st:
            for q in ('sides_1mm', 'max_turn_deg_per_1.5mm', 'curvature_sign_changes'):
                M['%s: %s' % (k, q)] = float(np.mean([s[q] for s in st]))
    return M


def draw_lines(im, R, frame, col, w=3):
    d = ImageDraw.Draw(im)
    for k, Cs in R['lines'].items():
        for C in Cs:
            if C is None or len(C) < 2:
                continue
            P = [frame(np.array(p)) for p in C]
            d.line([tuple(p) for p in P], fill=col, width=w)
    for k, p in R['landmarks'].items():
        q = frame(np.array(p))
        d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], outline=col, width=2)


def compare(ref_path, model_paths, out):
    ref = json.load(open(ref_path))
    Mr = measures(ref)
    models = [json.load(open(p)) for p in model_paths]
    Ms = [measures(m) for m in models]
    os.makedirs(out, exist_ok=True)
    rows = []
    for k in Mr:
        rv = Mr[k]
        row = {'measure': k, 'reference': rv}
        for m, M in zip(models, Ms):
            mv = M.get(k)
            row[m['name']] = mv
            row[m['name'] + ' gap'] = None if (mv is None or rv is None) else mv - rv
        rows.append(row)
    json.dump({'reference': ref['name'], 'models': [m['name'] for m in models], 'rows': rows}, open(os.path.join(out, 'flow-gaps.json'), 'w'), indent=1)
    md = ['| Measure | Reference | ' + ' | '.join('%s (gap)' % m['name'] for m in models) + ' |',
          '|---|---|' + '---|' * len(models)]
    f3 = lambda v: '' if v is None else ('%.3f' % v if abs(v) < 10 else '%.1f' % v)
    for r in rows:
        md.append('| %s | %s | %s |' % (r['measure'], f3(r['reference']), ' | '.join(
            '%s (%s)' % (f3(r[m['name']]), ('%+.3f' % r[m['name'] + ' gap']) if r[m['name'] + ' gap'] is not None else '') for m in models)))
    open(os.path.join(out, 'flow-gaps.md'), 'w').write('\n'.join(md) + '\n')
    # the sheet: per model its clay front with its own lines (cyan) and the photo's (yellow), aligned on the pupils;
    # and the photo with its lines and the first model's
    S = 620
    sc = S / 2.0      # pixels per unit
    panels = []
    ph = Image.open(ref['image']).convert('RGB')
    # the photo, levelled and scaled to the unit frame
    for m in models:
        fr = m['clay_frame']
        clay = Image.open(m['clay'])
        # (the clay is in its own frame: map unit -> clay pixel, then resample to S x S centred on the pupils)
        def clay_px(u, fr=fr):
            x = u[0] * fr['ipd'] + fr['cx']
            y = -u[1] * fr['ipd'] + fr['cy']
            return np.array([(x - fr['lo'][0]) / fr['res'], (fr['hi'][1] - y) / fr['res']])
        a = clay_px(np.array([-1.0, -Y0]))
        b = clay_px(np.array([1.0, 2.0 - Y0]))
        crop = clay.crop((int(a[0]), int(a[1]), int(b[0]), int(b[1]))).resize((S, S), Image.LANCZOS)
        fr_u = lambda u: np.array([(u[0] + 1.0) * sc, (u[1] + Y0) * sc])
        draw_lines(crop, ref, fr_u, (255, 205, 40), 3)
        draw_lines(crop, m, fr_u, (60, 220, 255), 2)
        ImageDraw.Draw(crop).text((8, 8), '%s clay, front ortho: ours cyan, reference yellow' % m['name'], fill=(255, 255, 255))
        panels.append(crop)
    # the photo in the unit frame (affine: unit -> photo pixel)
    roll = math.radians(ref['roll_deg'])
    u = ref['unit_px']
    # the original pixel positions are not stored in unit form; rebuild the frame from the sheet the trace read
    o_px = None
    try:
        sheet = json.load(open(os.path.splitext(ref['image'])[0] + '.measure.json'))
        a_ = np.array(sheet['landmarks']['pupil_l'], float)
        b_ = np.array(sheet['landmarks']['pupil_r'], float)
        o_px = 0.5 * (a_ + b_)
    except Exception:
        pass
    if o_px is not None:
        c, s = math.cos(roll), math.sin(roll)
        # unit (X, Y) -> pixel: o + u * R(roll) [X, Y]
        out_im = Image.new('RGB', (S, S))
        A = np.array([[c, -s], [s, c]]) * u / sc
        off = o_px - A @ np.array([sc, Y0 * sc])
        out_im = ph.transform((S, S), Image.AFFINE, (A[0, 0], A[0, 1], off[0], A[1, 0], A[1, 1], off[1]), Image.BICUBIC)
        fr_u = lambda q: np.array([(q[0] + 1.0) * sc, (q[1] + Y0) * sc])
        draw_lines(out_im, ref, fr_u, (255, 205, 40), 3)
        if models:
            draw_lines(out_im, models[0], fr_u, (60, 220, 255), 2)
        ImageDraw.Draw(out_im).text((8, 8), 'reference, levelled on the pupils: yellow its lines, cyan %s' % (models[0]['name'] if models else ''), fill=(255, 255, 255))
        panels = [out_im] + panels
    W_ = S * min(3, len(panels))
    H_ = S * int(math.ceil(len(panels) / 3))
    sheet_im = Image.new('RGB', (W_, H_), (20, 20, 24))
    for i, p in enumerate(panels):
        sheet_im.paste(p, ((i % 3) * S, (i // 3) * S))
    sheet_im.save(os.path.join(out, 'flow-sheet.png'))
    print('\n'.join(md))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sp = ap.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('ref')
    a.add_argument('photo')
    a.add_argument('sheet')
    a.add_argument('--out', default='.')
    a = sp.add_parser('model')
    a.add_argument('dump')
    a.add_argument('--out', default='.')
    a.add_argument('--theta', type=float, default=THETA)
    a = sp.add_parser('compare')
    a.add_argument('ref')
    a.add_argument('models', nargs='+')
    a.add_argument('--out', default='.')
    A = ap.parse_args()
    if A.cmd == 'ref':
        trace_ref(A.photo, A.sheet, A.out)
    elif A.cmd == 'model':
        trace_model(A.dump, A.out, A.theta)
    else:
        compare(A.ref, A.models, A.out)


if __name__ == '__main__':
    main()
