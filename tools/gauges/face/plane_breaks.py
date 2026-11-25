"""Plane breaks: where the front of the nose turns into its side, how far it turns and how sharp the turn is, read from
the shading across the shadow line, on a reference photo and on Face Creator heads, the same way on both.
John 2026-11-24: "Read how to derive the right chisel properly from the lines".

    python3 tools/gauges/face/plane_breaks.py ref <photo> <ref.measure.json> --out DIR
    python3 tools/gauges/face/plane_breaks.py model <dump.json> --out DIR
    python3 tools/gauges/face/plane_breaks.py compare DIR/<ref>.breaks.json DIR/<face>.breaks.json ... --out DIR

The method is the draughtsman's, put into numbers:
  - A plane break is where the tone steps. Its place is where the light front starts to fall (Hampton, "Shading After
    Construction": the box of the nose "drops back in shadow and comes forward").
  - The tonal step gives the turn. On a matt surface the light reflected goes with the cosine of the angle to the
    light (Lambert; Horn, "Understanding Image Intensities" 2.1), so the side's tone over the front's tone gives how
    far the side has turned away from the front.
  - The width of the step gives the sharpness. "A sharp edge suggests a sudden plane change. A firm edge suggests a
    rounded corner. A soft edge suggests a large round form" (Prokopenko, "Intro to Edges"). A turn of phi spread over
    a projected width w is a corner of radius about w / sin(phi).
  - Few edges are hard: "Look for the one or two HARD edges and let the others all be softer" (Weistling, Proko).

Each height y down the line (pupil distances below the pupils' line, the shadow_trace line on the same image), on each
row from the midline out:
  front   the bright front's level (the row's top between the midline and the line), 1 = the bridge's front
  core    the tone at the shade's core (the line), over the front's
  turn    the side's turn from the front, degrees, from core by Lambert with shadow_trace's key and wrap
  x_break where the tone has fallen 10% of the way from front to core (the corner's start)
  width   the 10-90% fall, in mm (pupil distance 63 mm, Dodgson 2004 adult mean)
  radius  width / sin(turn), mm: the corner's radius
  rise    the 10-90% climb from the core out to the cheek or socket's lit level, mm (the far edge of the shade band)
On a head the true normals are known, so the gauge also reports the turn and the radius from the geometry (turn_geo,
radius_geo): the check that the shading reading is right where it can be checked.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shadow_trace as st  # noqa: E402

IPD_MM = 63.0
YS = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45]


def turn_from_core(core, cf):
    """the side's turn from the front (deg) when its tone is core x the front's, with the front's cosine to the key cf,
    under shadow_trace's wrapped Lambert"""
    W = st.WRAP
    sf = (cf + W) / (1 + W)
    cs = core * sf * (1 + W) - W
    return float(np.degrees(np.arccos(np.clip(cs / cf, -1, 1))))


def row_profile(A, xs, ys, y, xl, sg):
    i = int(np.argmin(np.abs(ys - y)))
    xo = np.arange(0.0, 0.45, st.RES)
    ci = np.array([np.argmin(np.abs(xs - sg * x)) for x in xo])
    return xo, A[i, ci]


def crossing(xo, p, level, j0, j1):
    """first x from j0 toward j1 where p crosses level"""
    step = 1 if j1 >= j0 else -1
    for j in range(j0, j1, step):
        a, b = p[j], p[j + step]
        if (a - level) * (b - level) <= 0 and a != b:
            return xo[j] + (xo[j + step] - xo[j]) * (level - a) / (b - a)
    return None


def breaks(A, lines, eyes, geo=None):
    xs, ys = st.grid()
    Am = ndimage.gaussian_filter(np.where(eyes, np.nan, A), 1.5)
    cf = np.asarray(st.LIGHT, float)[2] / np.linalg.norm(st.LIGHT)
    out = {}
    for y in YS:
        recs = []
        for side, sg in (('l', -1), ('r', 1)):
            xc = st.line_at(np.asarray(lines[side]), y)
            if xc is None:
                continue
            xo, p = row_profile(Am, xs, ys, y, 0, sg)
            jc = int(np.argmin(np.abs(xo - xc)))
            if not np.all(np.isfinite(p[:jc + 1])):
                continue
            jf = int(np.argmax(p[:jc + 1]))
            front, core = float(p[jf]), float(p[jc])
            if front - core < 0.03:
                continue
            x90 = crossing(xo, p, front - 0.1 * (front - core), jf, jc)
            x10 = crossing(xo, p, core + 0.1 * (front - core), jf, jc)
            r = {'front': front, 'core': core / front, 'turn': turn_from_core(core / front, cf),
                 'x_break': x90, 'x_core': xc}
            if x90 is not None and x10 is not None:
                w = (x10 - x90) * IPD_MM
                r['width'] = w
                r['radius'] = w / max(np.sin(np.radians(r['turn'])), 0.05)
            # the far edge: climb from the core out to the next lit level (cheek, socket, canthus)
            tail = p[jc:]
            ok = np.isfinite(tail)
            if ok.sum() > 10:
                k = np.where(ok)[0]
                tail = tail[: k[np.argmax(np.diff(np.r_[k, 10 ** 6]) > 1)] + 1]
                if len(tail) > 10:
                    lit = float(np.max(tail))
                    a90 = crossing(xo[jc:], tail, core + 0.9 * (lit - core), 0, len(tail) - 1)
                    a10 = crossing(xo[jc:], tail, core + 0.1 * (lit - core), 0, len(tail) - 1)
                    r['far_lit'] = lit / front
                    if a90 is not None and a10 is not None:
                        r['rise'] = (a90 - a10) * IPD_MM
            if geo is not None:
                ang = geo(y, sg, xo)      # the true horizontal turn of the normal (deg) along the row
                if ang is not None:
                    r['turn_geo'] = float(ang[jc] - ang[jf])
                    if x90 is not None and x10 is not None:
                        g = np.gradient(np.radians(ang), xo * IPD_MM)
                        seg = (xo >= x90) & (xo <= x10)
                        if seg.any() and np.max(np.abs(g[seg])) > 1e-6:
                            r['radius_geo'] = float(1.0 / np.max(np.abs(g[seg])))
            recs.append(r)
        if recs:
            keys = set.intersection(*[set(r) for r in recs])
            out['%.2f' % y] = {k: float(np.mean([r[k] for r in recs])) for k in sorted(keys)}
    return out


def model_geo(dump):
    """the head's own normals from its z-buffer, as a function giving the horizontal turn along a row"""
    face = st.load_dump(dump)
    V, F = face['head']
    pup = sorted([Ve[np.argmax(Ve[:, 2])] for (Ve, Fe) in face['eyes']], key=lambda q: q[0])
    ipd = float(np.linalg.norm(pup[1][:2] - pup[0][:2]))
    cx, cy = 0.5 * (pup[0][0] + pup[1][0]), 0.5 * (pup[0][1] + pup[1][1])
    res = ipd * st.RES
    zb = st.zbuffer(V, F, (cx + st.X0 * ipd, cy - st.Y1 * ipd), (cx + st.X1 * ipd, cy - st.Y0 * ipd), res)
    z = np.where(np.isfinite(zb), zb, np.nan)
    z = ndimage.gaussian_filter(np.nan_to_num(z, nan=np.nanmin(z)), 1.0)
    gy, gx = np.gradient(z, res)
    xs, ys = st.grid()
    gx = gx[:len(ys), :len(xs)]

    def f(y, sg, xo):
        i = int(np.argmin(np.abs(ys - y)))
        ci = np.array([np.argmin(np.abs(xs - sg * x)) for x in xo])
        return np.degrees(np.arctan(-sg * gx[i, ci]))       # (turn away from the viewer, outward positive)
    return f


def save(name, kind, R, out):
    os.makedirs(out, exist_ok=True)
    json.dump({'name': name, 'kind': kind, 'breaks': R}, open(os.path.join(out, name + '.breaks.json'), 'w'), indent=1)
    print(name)
    for y, r in R.items():
        print('  y %s ' % y + ' '.join('%s %.3g' % (k, v) for k, v in r.items()))


def compare(paths, out):
    Rs = [json.load(open(p)) for p in paths]
    keys = ['x_break', 'core', 'turn', 'width', 'radius', 'rise', 'turn_geo', 'radius_geo']
    md = []
    for k in keys:
        md.append('\n**%s**\n' % k)
        md.append('| y | ' + ' | '.join(R['name'] for R in Rs) + ' |')
        md.append('|---' * (len(Rs) + 1) + '|')
        for y in ['%.2f' % v for v in YS]:
            cells = []
            for R in Rs:
                v = R['breaks'].get(y, {}).get(k)
                cells.append('-' if v is None else '%.3g' % v)
            md.append('| %s | %s |' % (y, ' | '.join(cells)))
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'plane-breaks.md'), 'w').write('\n'.join(md) + '\n')
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
    c.add_argument('files', nargs='+')
    c.add_argument('--out', required=True)
    A = ap.parse_args()
    if A.cmd == 'ref':
        S, eyes, lm = st.shade_photo(A.photo, A.sheet)
        Aa, lines = st.trace(S, eyes, 'ref')
        save(os.path.splitext(os.path.basename(A.photo))[0], 'photo', breaks(Aa, lines, eyes), A.out)
    elif A.cmd == 'model':
        S, eyes, lm, nm = st.shade_model(A.dump)
        Aa, lines = st.trace(S, eyes, nm)
        save(nm, 'model', breaks(Aa, lines, eyes, model_geo(A.dump)), A.out)
    else:
        compare(A.files, A.out)


if __name__ == '__main__':
    main()
