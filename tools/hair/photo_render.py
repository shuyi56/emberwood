"""The wavy photo field (photo_waves.field2), rendered as hair: every root's field line becomes fibres, drawn by
render.py's fibre renderer (clay and shaded turnarounds and close-ups). No groom and no solve: this is the ponytail
recipe's step 3, the field as it stands seen as hair, so its faults show before any locks are built.

    python3 tools/hair/photo_render.py HEAD.json.gz OUT_DIR [--lines 16000] [--per-root 2] [--size 750]
        [--tone 7e4c34] [--photo reference.jpg] [--no-locks]

Each line is drawn --per-root times; the copies sit a little apart across the hair (square to it, COPY_MM) and are
cut a little apart at the tip, and share the line's hairs, so the coverage on screen is the field's own hair count.
With --photo the front close-up is saved beside the reference.
"""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF  # noqa: E402
import photo_waves as PW  # noqa: E402
import wavy_flow as WF  # noqa: E402
import render as RD  # noqa: E402

NODES = 160                 # nodes per drawn fibre, even by length
COPY_MM = 1.5               # DRAFT: a copy's offset from its line, square to it (a field line stands for a few hundred hairs)
CUT_JIT = (0.08, 0.04)      # tips end within the last 8% of the length (falls), 4% (fringe)


def resample(p, n=NODES, keep=1.0):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, s[-1] * keep, n)
    return np.c_[[np.interp(t, s, p[:, k]) for k in range(3)]].T


def fibres(Fd, per_root=2, seed=7):
    rng = np.random.default_rng(seed)
    paths, hairs, basin = [], [], []
    for i, l in enumerate(Fd['line']):
        if len(l) < 4:
            continue
        b = int(Fd['basin'][i])
        for c in range(per_root):
            p = resample(l, keep=1 - CUT_JIT[b == 0] * rng.random())
            if c:
                t = np.gradient(p, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
                o = rng.normal(0, 1, 3); o = o - (t @ o)[:, None] * t
                o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-9)
                grow = np.clip(np.arange(NODES) / 12.0, 0, 1)[:, None]          # (all copies share the root)
                p = p + o * COPY_MM * PW.MM * grow
            paths.append(p); hairs.append(Fd['hairs'][i] / per_root); basin.append(b)
    return np.array(paths), np.array(hairs), np.array(basin)


def main():
    from PIL import Image
    a = sys.argv[1:]
    head, out = a[0], a[1]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    n, per, size = opt('--lines', 16000), opt('--per-root', 2), opt('--size', 750)
    os.makedirs(out, exist_ok=True)
    if '--tone' in a:
        h = a[a.index('--tone') + 1].lstrip('#')
        RD.FILL_SRGB = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    pts = SF.load_head(head)
    surf = SF.Surface(pts); F = SF.build(surf)
    body = json.load(gzip.open(head))['bodies']['mannequin']
    coll, depth = WF.colliders(surf, body)
    Fd = PW.field2(F, surf, depth, n)
    if '--no-locks' not in a:
        Fd, lrep = PW.lock_fall(Fd)
    paths, hairs, basin = fibres(Fd, per)
    RD.HEAD_CLOUD = pts
    RD.SCALP_F = F
    rep = dict(locks=None if '--no-locks' in a else lrep, lines=n, fibres=len(paths), hairs=round(float(hairs.sum())), per_root=per, gauges=Fd['gauges'])
    for mode in ('clay', 'full'):
        row = []
        for az in (0, 45, 90, 135, 180):
            cv = RD.Canvas(size, az)
            bg, zb = RD.clay_backdrop(cv, surf, body, mode)
            row.append((RD.to_srgb(RD.render_hair(cv, paths, mode, bg, zb, hairs)) * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(row, 1)).save(os.path.join(out, 'turn-%s.png' % mode))
    for name, az, box in (('closeup-front', 0, (-0.75, 0.75, -0.55, 1.15)), ('closeup-side', 90, (-0.8, 0.8, -0.55, 1.15)),
                          ('closeup-34', 40, (-0.8, 0.8, -0.55, 1.15)), ('closeup-back', 180, (-0.75, 0.75, -0.55, 1.15))):
        imgs = []
        for mode in ('clay', 'full'):
            cv = RD.Canvas(size, az, box)
            bg, zb = RD.clay_backdrop(cv, surf, body, mode)
            imgs.append((RD.to_srgb(RD.render_hair(cv, paths, mode, bg, zb, hairs)) * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(imgs, 1)).save(os.path.join(out, name + '.png'))
        if name == 'closeup-front' and '--photo' in a:
            ph = Image.open(a[a.index('--photo') + 1]).convert('RGB')
            ph = ph.resize((int(ph.width * imgs[1].shape[0] / ph.height), imgs[1].shape[0]))
            Image.fromarray(np.concatenate([np.asarray(ph), imgs[1]], 1)).save(os.path.join(out, 'beside-photo.png'))
    json.dump(rep, open(os.path.join(out, 'render.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != 'gauges'}))


if __name__ == '__main__':
    main()
