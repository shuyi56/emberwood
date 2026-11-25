"""Ponytail flow field, rendered as hair: every root of the field becomes fibres along its own flow line, drawn by
render.py's fibre renderer (clay and shaded turnarounds). No groom and no solve: the hair is the field's lines as
they stand, so the render shows what the field gives before any real groom is built from it.

    python3 tools/hair/ponytail_render.py HEAD.json.gz OUT_DIR [--per-root 3] [--size 750] [--tone 7e4c34]

Each root's line is drawn --per-root times. The copies differ only in their place in the bundle's section (a small
jitter of the angle and the radius, DRAFT) and in the cut of their tips (render.py's 10% jitter), and share the root's
hairs between them, so the coverage on screen is the field's own hair count.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF  # noqa: E402
import ponytail_flow as PF  # noqa: E402
import render as RD  # noqa: E402

NODES = 150                 # nodes per drawn fibre, even by length (3 mm on the longest)
PSI_JIT = 0.10              # DRAFT, rad: a copy's shift round the bundle's section
RR_JIT = 0.06               # DRAFT: and along its radius (share of the bundle's radius)
FRONT_COPIES = 4            # the fringe and the lock are drawn finer, as render.py's FRONT_BOOST draws the front
FRONT_SPREAD = 0.012        # HH: a front copy's offset from its root's line (about 1.3 mm each way)
CUT_JIT = 0.10              # render.py's cut jitter: tips end within the last 10% of the length


def resample(p, n=NODES, keep=1.0):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, s[-1] * keep, n)
    return np.c_[[np.interp(t, s, p[:, k]) for k in range(3)]].T


def fibres(head, per_root=3, seed=7):
    """(paths [n, NODES, 3], hairs per path, basin per path) for every root of the field."""
    B = PF.build(head)
    FF = PF.free_fall(B)
    psi, rr = PF.bundle_places(B)
    rng = np.random.default_rng(seed)
    paths, hairs, basin = [], [], []
    for i in np.where(B['basin'] == 0)[0]:
        p0, r0 = psi[i], rr[i]
        for c in range(per_root):
            if c:
                psi[i] = p0 + rng.normal(0, PSI_JIT); rr[i] = float(np.clip(r0 + rng.normal(0, RR_JIT), 0.0, 1.0))
            pts, _ = PF.strand(B, FF, psi, rr, i)
            paths.append(resample(pts, keep=1 - CUT_JIT * rng.random())); hairs.append(B['hairs'][i] / per_root)
            basin.append(0)
        psi[i], rr[i] = p0, r0
    for i in np.where(B['basin'] == 1)[0]:
        pts = PF.bang_line(B, i, k=32)
        for c in range(FRONT_COPIES):
            off = (rng.random(3) - 0.5) * FRONT_SPREAD * (c > 0)
            paths.append(resample(pts + off, keep=1 - 0.04 * rng.random())); hairs.append(B['hairs'][i] / FRONT_COPIES)
            basin.append(1)
    for k, (nm, cfg) in enumerate(PF.LOOSE.items()):
        for i in np.where(B['basin'] == 2 + k)[0]:
            pts = PF.loose_line(B, i, cfg['len'])
            for c in range(FRONT_COPIES):
                off = (rng.random(3) - 0.5) * FRONT_SPREAD * (c > 0)
                paths.append(resample(pts + off, keep=1 - 0.04 * rng.random())); hairs.append(B['hairs'][i] / FRONT_COPIES)
                basin.append(2 + k)
    return np.array(paths), np.array(hairs), np.array(basin), B


def main():
    import gzip
    from PIL import Image
    a = sys.argv[1:]
    head, out = a[0], a[1]
    per = int(a[a.index('--per-root') + 1]) if '--per-root' in a else 3
    size = int(a[a.index('--size') + 1]) if '--size' in a else 750
    os.makedirs(out, exist_ok=True)
    PF.HEAD = head
    PF.CACHE = os.path.join(os.path.dirname(os.path.abspath(out)), 'cache')
    os.makedirs(PF.CACHE, exist_ok=True)
    if '--tone' in a:
        h = a[a.index('--tone') + 1].lstrip('#')
        RD.FILL_SRGB = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if '--melanin' in a:
        RD.CE, RD.CP = (float(v) for v in a[a.index('--melanin') + 1].split(','))
    paths, hairs, basin, B = fibres(head, per)
    RD.HEAD_CLOUD = SF.load_head(head)
    surf = B['surf']
    RD.SCALP_F = B['F']
    body = json.load(gzip.open(head))['bodies']['mannequin']
    np.savez_compressed(os.path.join(out, 'fibres.npz'), paths=paths.astype(np.float32), hairs=hairs, basin=basin)
    rep = dict(fibres=len(paths), hairs=round(float(hairs.sum())), hairs_per_fibre=round(float(hairs.mean()), 2),
               per_root=per)
    for mode in ('clay', 'full'):
        row = []
        for az in (0, 45, 90, 135, 180):
            cv = RD.Canvas(size, az)
            bg, zb = RD.clay_backdrop(cv, surf, body, mode)
            row.append((RD.to_srgb(RD.render_hair(cv, paths, mode, bg, zb, hairs)) * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(row, 1)).save(os.path.join(out, 'turn-%s.png' % mode))
    for name, az, box in (('closeup-side', 90, (-0.55, 0.75, 0.1, 1.12)), ('closeup-front', 0, (-0.5, 0.5, 0.25, 1.1)),
                          ('closeup-back', 180, (-0.55, 0.55, -0.2, 1.12))):
        imgs = []
        for mode in ('clay', 'full'):
            cv = RD.Canvas(size, az, box)
            bg, zb = RD.clay_backdrop(cv, surf, body, mode)
            imgs.append((RD.to_srgb(RD.render_hair(cv, paths, mode, bg, zb, hairs)) * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(imgs, 1)).save(os.path.join(out, name + '.png'))
    json.dump(rep, open(os.path.join(out, 'render.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
