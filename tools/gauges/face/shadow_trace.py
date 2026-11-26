"""Shadow trace: the edge of the shade from the brow, past the inner corner of the eye, down the side of the nose and
round its wing, on a reference photo and on Face Creator heads, from the same front camera in the same unit (the
pupils' distance, levelled on the pupils, as face_flow.py). John 2026-11-24: "Trace the shadows from eye to nose and
why are ours don't have that smooth wave like curve".

    python3 tools/gauges/face/shadow_trace.py ref <photo> <ref.measure.json> --out DIR
    python3 tools/gauges/face/shadow_trace.py model <dump.json> --out DIR          (a dump-heads.cjs dump)
    python3 tools/gauges/face/shadow_trace.py compare DIR/<ref>.shadow.json DIR/<face>.shadow.json ... --out DIR

The photo: John's yellow marks are painted out (biharmonic inpainting over the strokes), the image is levelled on the
pupils and resampled to the unit frame, and its luminance is the shading (the photo's light is a soft frontal key, so
its shade is where the surface turns away). The heads: a z-buffer of the head and the eyes from the front (face_flow's
frame), normals from it, lit by a soft frontal key from a little above (the photo's) with Lambert's law and a wrap.

The line: the iso-shade through the side of the nose at the pupils' line, at LEVEL of the brightest of the bridge's
front between the eyes and the tip (EXPERIMENTAL: 0.82; 0.75-0.9 trace the same course on the photo), followed up to
the brow and down round the wing, on each side; the two sides are averaged (mirrored). Its measures, x out from the
midline at heights y (down from the pupils' line, all in pupil distances):
  - x at the brow's head (y -0.25), at the inner corner (0), on the bridge's side (0.25), round the wing (0.5);
  - the wave: inset = x(-0.25) - x at its narrowest above the wing, outset = x(0.5) - that narrowest; the line is a
    wave when both are well above 0 (the photo: in at the radix, out at the wing), a straight cone when the inset is 0;
  - its turns: the sign changes of its curvature (smoothed), and the largest turn per 0.05 unit (a kink).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_flow import level_frame, load_dump, zbuffer  # noqa: E402

RES = 1 / 300.0                       # the unit grid's step (pupil distances)
X0, X1, Y0, Y1 = -0.7, 0.7, -0.6, 0.8  # the unit grid (x right, y down)
LEVEL = None      # (unused: the line is the shade's core, below)
Y_TOP, Y_BOT = -0.28, 0.55      # the brow's head to the wing's foot
X_IN, X_OUT = 0.04, 0.40
LAM, STEP = 0.02, 6      # (bend penalty per (0.01 across per 0.01 down)^2, and the most columns a row may step)
LIGHT = (0.0, 0.25, 1.0)              # the key, from the front and a little above (x right, y up, z to the viewer)
WRAP = 0.25


def eye_mask(X, Y, pl, pr):
    """the same opening round each pupil on photo and head (half-width 0.26, half-height 0.11 pupil distances): the
    lids' own dark (lashes, liner, the canthus) is not the face's shade"""
    m = np.zeros_like(X, bool)
    for p in (pl, pr):
        m |= ((X - p[0]) / 0.26) ** 2 + ((Y - p[1]) / 0.11) ** 2 < 1
    return m


def grid():
    xs = np.arange(X0, X1, RES)
    ys = np.arange(Y0, Y1, RES)
    return xs, ys


def shade_photo(photo, sheet):
    from skimage.restoration import inpaint_biharmonic
    from skimage.morphology import dilation, disk
    Lm = json.load(open(sheet))['landmarks']
    pl, pr = np.asarray(Lm['pupil_l'], float), np.asarray(Lm['pupil_r'], float)
    a = np.asarray(Image.open(photo).convert('RGB')).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (R > 170) & (G > 120) & (B < 120) & (R - B > 100)      # John's yellow marker
    m = dilation(m, disk(4))
    # (inpaint on a crop round the face's middle, scaled so the pupils are 300 px apart)
    ipd = np.linalg.norm(pr - pl)
    o = 0.5 * (pl + pr)
    s = 300.0 / ipd
    x0, x1 = int(max(0, o[0] - 0.8 * ipd)), int(min(a.shape[1], o[0] + 0.8 * ipd))
    y0, y1 = int(max(0, o[1] - 0.7 * ipd)), int(min(a.shape[0], o[1] + 0.9 * ipd))
    crop = a[y0:y1, x0:x1] / 255.0
    mc = m[y0:y1, x0:x1]
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((int((x1 - x0) * s), int((y1 - y0) * s)), Image.LANCZOS)).astype(float) / 255
    ms = np.asarray(Image.fromarray(mc.astype(np.uint8) * 255).resize(small.shape[1::-1], Image.NEAREST)) > 0
    fill = inpaint_biharmonic(small, ms, channel_axis=-1)
    L = 0.299 * fill[..., 0] + 0.587 * fill[..., 1] + 0.114 * fill[..., 2]
    # resample to the unit grid: unit (x, y) -> photo px (levelled on the pupils, y down)
    d = (pr - pl) / ipd
    xs, ys = grid()
    X, Y = np.meshgrid(xs, ys)
    px = o[0] + ipd * (X * d[0] - Y * d[1])
    py = o[1] + ipd * (X * d[1] + Y * d[0])
    S = ndimage.map_coordinates(L, [(py - y0) * s, (px - x0) * s], order=1, mode='nearest')
    f, _, _ = level_frame(pl, pr)
    eyes = eye_mask(X, Y, f(Lm['pupil_l']), f(Lm['pupil_r']))      # (the eyes and the brows' hair are not shade)
    # (the brows' hair: dark against the skin above the eyes)
    Sm = ndimage.gaussian_filter(S, 2.0)
    hair = (Y < -0.12) & (Sm < 0.8 * np.percentile(Sm[(Y < -0.4) & (np.abs(X) < 0.2)], 50))
    hair = ndimage.binary_dilation(hair, iterations=4)
    return S, eyes | hair, {k: f(v).tolist() for k, v in Lm.items()}


def shade_model(dump):
    face = load_dump(dump)
    V, F = face['head']
    pup = sorted([Ve[np.argmax(Ve[:, 2])] for (Ve, Fe) in face['eyes']], key=lambda p: p[0])
    ipd = float(np.linalg.norm(pup[1][:2] - pup[0][:2]))
    cx, cy = 0.5 * (pup[0][0] + pup[1][0]), 0.5 * (pup[0][1] + pup[1][1])
    res = ipd * RES
    lo = (cx + X0 * ipd, cy - Y1 * ipd)
    hi = (cx + X1 * ipd, cy - Y0 * ipd)
    zb = zbuffer(V, F, lo, hi, res)
    ze = np.maximum.reduce([zbuffer(Ve, Fe, lo, hi, res) for (Ve, Fe) in face['eyes']])
    eyes = ze > zb + 1e-6
    z = np.where(np.isfinite(zb), zb, np.nan)
    z = ndimage.gaussian_filter(np.nan_to_num(z, nan=np.nanmin(z)), 1.0)
    gy, gx = np.gradient(z, res)      # (rows go down: dz/dY up = -gy)
    n = np.stack([-gx, gy, np.ones_like(z)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Lv = np.asarray(LIGHT, float) / np.linalg.norm(LIGHT)
    S = np.clip((n @ Lv + WRAP) / (1 + WRAP), 0, 1)
    H = len(grid()[1])
    W = len(grid()[0])
    S, eyes = S[:H, :W], eyes[:H, :W]
    xs, ys = grid()
    X, Y = np.meshgrid(xs, ys)
    eyes = ndimage.binary_dilation(eyes, iterations=6) | eye_mask(X, Y, [-0.5, 0], [0.5, 0])
    lm = {'pupil_l': [-0.5, 0.0], 'pupil_r': [0.5, 0.0]}
    return S, eyes, lm, face['face']


def trace(S, eyes, name, hair=None):
    """The shade's core from the brow's head to the wing on each side: the darkest smooth path down the side of the
    nose (dynamic programming over rows 0.01 apart: the shade at each point plus a bend penalty), the eyes and the
    brows' hair counted as lit so the path runs beside them, not through them"""
    xs, ys = grid()
    Sm = ndimage.gaussian_filter(S, 2.0)
    mid = np.abs(xs) < 0.03
    rows = (ys > -0.05) & (ys < 0.45)
    ref = np.percentile(Sm[np.ix_(rows, mid)], 90)
    A = Sm / ref
    B = np.where(eyes, 1.3, A)
    if hair is not None:
        B = np.where(hair, 1.3, B)
    out = {}
    yr = np.arange(Y_TOP, Y_BOT + 1e-9, 0.01)
    ri = np.array([np.argmin(np.abs(ys - y)) for y in yr])
    for side, sg in (('l', -1), ('r', 1)):
        xo = np.arange(X_IN, X_OUT, RES)
        ci = np.array([np.argmin(np.abs(xs - sg * x)) for x in xo])
        C = B[np.ix_(ri, ci)]
        n, m = C.shape
        D = np.full((n, m), np.inf)
        bk = np.zeros((n, m), int)
        D[0] = C[0]
        jj = np.arange(m)
        for r in range(1, n):
            # (a step of k columns between rows costs LAM (k RES / 0.01)^2: the line may lean, not jump)
            best = np.full(m, np.inf)
            arg = np.zeros(m, int)
            for k in range(-STEP, STEP + 1):
                src = np.clip(jj - k, 0, m - 1)
                v = D[r - 1, src] + LAM * (k * RES / 0.01) ** 2
                bt = v < best
                best[bt] = v[bt]
                arg[bt] = src[bt]
            D[r] = best + C[r]
            bk[r] = arg
        j = int(np.argmin(D[-1]))
        path = [j]
        for r in range(n - 1, 0, -1):
            j = bk[r, j]
            path.append(j)
        path = path[::-1]
        out[side] = np.stack([xo[path], yr], 1)
    return A, out


def line_at(Q, y):
    """x of the line at height y (the outermost crossing, or None)"""
    xs = []
    for i in range(len(Q) - 1):
        (xa, ya), (xb, yb) = Q[i], Q[i + 1]
        if (ya - y) * (yb - y) <= 0 and ya != yb:
            xs.append(xa + (xb - xa) * (y - ya) / (yb - ya))
    return max(xs) if xs else None


def measures(lines):
    def both(y):
        v = [line_at(lines[s], y) for s in ('l', 'r')]
        v = [q for q in v if q is not None]
        return float(np.mean(v)) if v else None
    M = {'x_brow_head': both(-0.25), 'x_inner_corner': both(0.0), 'x_bridge_side': both(0.25), 'x_wing': both(0.5)}
    ysamp = np.arange(-0.25, 0.46, 0.01)
    xv = [both(y) for y in ysamp]
    xv = [q for q in xv if q is not None]
    nar = min(xv) if xv else None
    M['x_narrowest'] = nar
    M['inset'] = (M['x_brow_head'] - nar) if (nar is not None and M['x_brow_head'] is not None) else None
    M['outset'] = (M['x_wing'] - nar) if (nar is not None and M['x_wing'] is not None) else None
    # turns: along the line resampled by arc length 0.01, smoothed
    turns, kink = [], []
    for s in ('l', 'r'):
        Q = lines[s]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        if seg[-1] < 0.1:
            continue
        t = np.arange(0, seg[-1], 0.01)
        R = np.stack([np.interp(t, seg, Q[:, 0]), np.interp(t, seg, Q[:, 1])], 1)
        R = ndimage.gaussian_filter1d(R, 3, axis=0, mode='nearest')
        a = np.unwrap(np.arctan2(np.diff(R[:, 1]), np.diff(R[:, 0])))
        da = np.diff(a)
        sig = ndimage.gaussian_filter1d(da, 4)
        sc = np.sign(sig[np.abs(sig) > 0.004])
        turns.append(int(np.sum(sc[1:] != sc[:-1])))
        kink.append(float(np.degrees(np.max(np.abs(a[5:] - a[:-5]))) if len(a) > 5 else 0))
    M['curvature_sign_changes'] = float(np.mean(turns)) if turns else None
    M['max_turn_deg_per_0.05'] = float(np.mean(kink)) if kink else None
    M['length'] = float(np.mean([np.sum(np.linalg.norm(np.diff(lines[s], axis=0), axis=1)) for s in ('l', 'r')]))
    return M


def save(kind, name, A, lines, lm, out, extra=None):
    os.makedirs(out, exist_ok=True)
    res = {'kind': kind, 'name': name, 'level': LEVEL, 'light': LIGHT, 'landmarks': lm,
           'lines': {k: v.tolist() for k, v in lines.items()}, 'measures': measures(lines)}
    if extra:
        res.update(extra)
    json.dump(res, open(os.path.join(out, name + '.shadow.json'), 'w'), indent=1)
    im = Image.fromarray((np.clip(A, 0, 1.2) / 1.2 * 255).astype(np.uint8)).convert('RGB')
    im.save(os.path.join(out, name + '.shade.png'))
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['measures'].items()}))


def to_px(P, sg):
    xs, ys = grid()
    return [((sg * x - X0) / RES, (y - Y0) / RES) for x, y in P]


def compare(ref_path, model_paths, out):
    ref = json.load(open(ref_path))
    rows = []
    tiles = []
    for p in [ref_path] + model_paths:
        R = json.load(open(p))
        base = Image.open(p.replace('.shadow.json', '.shade.png')).convert('RGB')
        if R['kind'] == 'photo':
            base = Image.open(p.replace('.shadow.json', '.photo.png')).convert('RGB') if os.path.exists(p.replace('.shadow.json', '.photo.png')) else base
        d = ImageDraw.Draw(base)
        for s, sg in (('l', -1), ('r', 1)):
            if R is not ref:
                d.line(to_px(np.asarray(ref['lines'][s]), sg), fill=(255, 200, 40), width=3)
            d.line(to_px(np.asarray(R['lines'][s]), sg), fill=(40, 200, 255) if R['kind'] != 'photo' else (255, 60, 60), width=3)
        d.text((8, 8), R['name'], fill=(255, 230, 90))
        tiles.append(base.crop((int((-0.55 - X0) / RES), int((-0.5 - Y0) / RES), int((0.55 - X0) / RES), int((0.7 - Y0) / RES))))
        rows.append((R['name'], R['measures']))
    W = sum(t.width for t in tiles)
    sheet = Image.new('RGB', (W, tiles[0].height))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    os.makedirs(out, exist_ok=True)
    sheet.save(os.path.join(out, 'shadow-sheet.png'))
    keys = ['x_brow_head', 'x_inner_corner', 'x_bridge_side', 'x_wing', 'x_narrowest', 'inset', 'outset', 'curvature_sign_changes', 'max_turn_deg_per_0.05']
    md = ['| Measure | ' + ' | '.join(n for n, _ in rows) + ' |', '|---' * (len(rows) + 1) + '|']
    for k in keys:
        cells = []
        for n, M in rows:
            v = M.get(k)
            cells.append('-' if v is None else ('%.3f' % v if isinstance(v, float) else str(v)))
        md.append('| %s | %s |' % (k, ' | '.join(cells)))
    open(os.path.join(out, 'shadow-gaps.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('ref')
    a.add_argument('photo')
    a.add_argument('sheet')
    a.add_argument('--out', required=True)
    b = sub.add_parser('model')
    b.add_argument('dump')
    b.add_argument('--out', required=True)
    c = sub.add_parser('compare')
    c.add_argument('ref')
    c.add_argument('models', nargs='+')
    c.add_argument('--out', required=True)
    A = ap.parse_args()
    if A.cmd == 'ref':
        S, eyes, lm = shade_photo(A.photo, A.sheet)
        Aa, lines = trace(S, eyes, 'ref')
        name = os.path.splitext(os.path.basename(A.photo))[0]
        save('photo', name, Aa, lines, lm, A.out)
    elif A.cmd == 'model':
        S, eyes, lm, nm = shade_model(A.dump)
        Aa, lines = trace(S, eyes, nm)
        save('model', nm, Aa, lines, lm, A.out)
    else:
        compare(A.ref, A.models, A.out)


if __name__ == '__main__':
    main()
