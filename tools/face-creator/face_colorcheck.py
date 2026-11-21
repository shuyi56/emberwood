"""The colour-map check for a dumped Face Creator face (dump-face.cjs), before a textured Blender render: the face's
touching colours read from the maps the render is given, with tools/render-review/colorcheck.py's limits (lightness
gap 12 or colour distance 25; under 25 of lightness the hue must differ by 20 degrees; a grayscale gap of 6 of 100).

    python3 tools/face-creator/face_colorcheck.py DUMP [--out swatches.png] [--hair 7e4c34]

Regions, from the dump's own maps: skin (the cheek, under the eye), eye shadow (the lid's mulberry wash), liner
(the darkest band along the upper lash line), brows and lips (the mask and the lip map), the eye white and the iris
(the eye's painting). Hair is a hex, the hairstyle's base tone."""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw


def lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def lab(rgb):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    x, y, z = (M @ lin(rgb)) / np.array([0.95047, 1.0, 1.08883])
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return np.array([116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))])


def gray(rgb):
    return 100 * float(np.dot(lin(rgb), [0.2126, 0.7152, 0.0722]))


ap = argparse.ArgumentParser()
ap.add_argument('dump')
ap.add_argument('--out', default='')
ap.add_argument('--hair', default='7e4c34')
a = ap.parse_args()
D = json.load(open(os.path.join(a.dump, 'face.json')))
face = np.array(Image.open(os.path.join(a.dump, 'face.png')).convert('RGB')).astype(float)
mask = np.array(Image.open(os.path.join(a.dump, 'mask.png')).convert('RGB')).astype(float) / 255
lipm = np.array(Image.open(os.path.join(a.dump, 'lip.png')).convert('RGB')).astype(float) / 255
eye = np.array(Image.open(os.path.join(a.dump, D['eyes'][0]['map'])).convert('RGB')).astype(float)
H, W = face.shape[:2]
yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
lum = face @ [0.2126, 0.7152, 0.0722]
# the eyes' band, from the mask's lash-and-waterline blobs (green, below the brows)
g = mask[..., 1] > 0.5
rows = np.where(g.any(1))[0]
brow_rows = rows[rows < rows.min() + 0.06 * H]
eye_band = (yy > (rows.min() + 0.07 * H) / H) & (yy < 0.42)
reg = {}
reg['brow'] = face[g & (yy < (brow_rows.max() + 2) / H)].mean(0)
reg['lips'] = face[lipm[..., 0] > 0.6].mean(0)
purple = face[..., 2] - face[..., 1]
lid = eye_band & ~g
reg['eye shadow'] = face[lid & (purple > np.percentile(purple[lid], 97))].mean(0)
band = eye_band & g
reg['liner'] = face[band & (lum < np.percentile(lum[band], 8))].mean(0)
cheek = (yy > 0.45) & (yy < 0.55) & (np.abs(xx - 0.5) > 0.12) & (np.abs(xx - 0.5) < 0.2) & ~g
reg['skin'] = np.median(face[cheek], 0)
el = eye @ [0.2126, 0.7152, 0.0722]
sat = eye.max(2) - eye.min(2)
top = np.arange(eye.shape[0])[:, None] < eye.shape[0] * 0.6      # (the ball; the lid's flesh is painted below it)
reg['eye white'] = np.median(eye[top & (el > np.percentile(el[top[:, 0]], 75)) & (sat < np.percentile(sat[top[:, 0]], 40))], 0)
cx = eye.shape[1] * 0.53
ir = (np.abs(np.arange(eye.shape[1]) - cx) < 20)[None, :] & (np.abs(np.arange(eye.shape[0]) - eye.shape[0] * 0.35) < 20)[:, None]
reg['iris'] = np.median(eye[ir & (el > 30)], 0)
reg['hair'] = np.array([int(a.hair[i:i + 2], 16) for i in (0, 2, 4)], float)
# (the pairs that touch on the face; the brows are meant to match the hair, and the brow bone's bare skin keeps the
# shadow off the brows, so neither is a pair)
PAIRS = [('skin', 'eye shadow'), ('eye shadow', 'liner'), ('skin', 'lips'), ('skin', 'brow'),
         ('eye white', 'iris'), ('eye white', 'eye shadow'), ('eye white', 'liner'), ('hair', 'skin')]
print('%-12s %s' % ('region', 'sRGB'))
for k, v in reg.items():
    print('%-12s #%02x%02x%02x  L %.0f  grey %.0f' % (k, *np.round(v).astype(int), lab(v)[0], gray(v)))
print()
flags = []
rows_out = []
for p, q in PAIRS:
    A, B = lab(reg[p]), lab(reg[q])
    dL, dE = abs(A[0] - B[0]), float(np.linalg.norm(A - B))
    hA, hB = math.degrees(math.atan2(A[2], A[1])) % 360, math.degrees(math.atan2(B[2], B[1])) % 360
    dh = min(abs(hA - hB), 360 - abs(hA - hB))
    dg = abs(gray(reg[p]) - gray(reg[q]))
    bad = []
    if dL < 12 and dE < 25:
        bad.append('too close')
    if dL < 25 and dh < 20:
        bad.append('hue %.0f' % dh)
    if dg < 6:
        bad.append('grey %.1f' % dg)
    rows_out.append((p, q, dL, dE, dh, dg, bad))
    print('%-11s / %-11s  dL %5.1f  dE %5.1f  hue %5.1f  grey %5.1f  %s' % (p, q, dL, dE, dh, dg, ', '.join(bad) or 'ok'))
if a.out:
    im = Image.new('RGB', (900, 60 + 40 * len(rows_out)), (245, 242, 238))
    d = ImageDraw.Draw(im)
    x = 10
    for k, v in reg.items():
        c = tuple(int(t) for t in np.round(v))
        d.rectangle([x, 10, x + 80, 40], fill=c)
        gv = int(round(255 * (gray(v) / 100) ** (1 / 2.2)))
        d.rectangle([x, 40, x + 80, 50], fill=(gv, gv, gv))
        d.text((x, 0), k, fill=(0, 0, 0))
        x += 88
    for i, (p, q, dL, dE, dh, dg, bad) in enumerate(rows_out):
        y = 60 + 40 * i
        d.rectangle([10, y, 50, y + 30], fill=tuple(int(t) for t in reg[p]))
        d.rectangle([50, y, 90, y + 30], fill=tuple(int(t) for t in reg[q]))
        d.text((100, y + 8), '%s / %s   dL %.1f  dE %.1f  hue %.0f  grey %.1f   %s' % (p, q, dL, dE, dh, dg, ', '.join(bad) or 'ok'),
               fill=(160, 0, 0) if bad else (0, 0, 0))
    im.save(a.out)
