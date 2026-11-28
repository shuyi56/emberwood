"""The wavy photo field (photo_waves.field2), rendered as hair: every root's field line becomes fibres, drawn by
render.py's fibre renderer (clay and shaded turnarounds and close-ups). No groom and no solve: this is the ponytail
recipe's step 3, the field as it stands seen as hair, so its faults show before any locks are built.

    python3 tools/hair/photo_render.py HEAD.json.gz OUT_DIR [--lines 16000] [--per-root 2] [--size 750]
        [--tone 7e4c34] [--photo reference.jpg] [--no-locks] [--quick] [--draft] [--crown] [--cache DIR | --no-cache]

Each line is drawn --per-root times; the copies sit a little apart across the hair (square to it, COPY_MM) and are
cut a little apart at the tip, and share the line's hairs, so the coverage on screen is the field's own hair count.
With --photo the front close-up is saved beside the reference.

Cached under --cache (default OUT/../photo-cache): the head's scalp field, the comb over the head, the body's radius
table. (The clay backdrops are not cached: 3 s each to build, 20 MB each to keep. First timing, 2026-11-28: the
field, locks and fibres take 44 s; the 18 hair renders take 610 s, 30 to 40 s each, so --quick is the saving.) Each is rebuilt only when the head file or the code and constants it depends on
change, so a change to the falls, the cut or the locks skips them. --quick draws only the front, side and back
close-ups in colour (quick.png). render.json's 'seconds' gives the time per stage.
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
CROWN_SIGN = 1.0            # which way the crown views tip (up toward the camera)
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


def _md5(path):
    import hashlib
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:12]


def main():
    import time
    from PIL import Image
    a = sys.argv[1:]
    head, out = a[0], a[1]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    n, per, size = opt('--lines', 16000), opt('--per-root', 2), opt('--size', 750)
    if '--draft' in a:                                                    # (a look, not a picture: half the fibres, smaller)
        per, size = 1, 560
    cache = None if '--no-cache' in a else opt('--cache', os.path.join(os.path.dirname(os.path.abspath(out)), 'photo-cache'))
    quick = '--quick' in a
    os.makedirs(out, exist_ok=True)
    T = [time.time()]; times = {}

    def lap(name):
        T.append(time.time()); times[name] = round(T[-1] - T[-2], 1)
    if '--tone' in a:
        h = a[a.index('--tone') + 1].lstrip('#')
        RD.FILL_SRGB = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    hk = _md5(head)
    src = lambda m: _md5(m.__file__)
    pts = SF.load_head(head)
    surf, F = PW._cached(cache, 'head', (hk, src(SF)), lambda: (lambda sf: (sf, SF.build(sf)))(SF.Surface(pts)))
    body = json.load(gzip.open(head))['bodies']['mannequin']
    coll, depth = WF.colliders(surf, body)
    lap('head')
    Fd = PW.field2(F, surf, depth, n, cache=cache, head_key=hk)
    lap('field')
    lrep = None
    if '--no-locks' not in a:
        Fd, lrep = PW.lock_fall(Fd)
    lap('locks')
    paths, hairs, basin = fibres(Fd, per)
    lap('fibres')
    RD.HEAD_CLOUD = pts
    RD.SCALP_F = F

    def backdrop(cv, key, mode):
        # (not cached: it builds in 3 s and is 20 MB a view; the time is in the hair, see render.json 'seconds')
        return RD.clay_backdrop(cv, surf, body, mode)
    rep = dict(locks=lrep, lines=n, fibres=len(paths), hairs=round(float(hairs.sum())), per_root=per, gauges=Fd['gauges'])
    if not quick:
        for mode in ('clay', 'full'):
            row = []
            for az in (0, 45, 90, 135, 180):
                cv = RD.Canvas(size, az)
                bg, zb = backdrop(cv, ('turn', az), mode)
                row.append((RD.to_srgb(RD.render_hair(cv, paths, mode, bg, zb, hairs)) * 255).astype(np.uint8))
            Image.fromarray(np.concatenate(row, 1)).save(os.path.join(out, 'turn-%s.png' % mode))
        lap('turnarounds')
    BOX = (-0.8, 0.8, -0.55, 1.15)                                         # (one box for all: the strip needs equal heights)
    views = (('closeup-front', 0, BOX), ('closeup-side', 90, BOX), ('closeup-34', 40, BOX), ('closeup-back', 180, BOX))
    full = {}
    for name, az, box in views:
        if quick and name == 'closeup-34':
            continue
        imgs = []
        for mode in (('full',) if quick else ('clay', 'full')):
            cv = RD.Canvas(size, az, box)
            bg, zb = backdrop(cv, (az, box), mode)
            imgs.append((RD.to_srgb(RD.render_hair(cv, paths, mode, bg, zb, hairs)) * 255).astype(np.uint8))
        full[name] = imgs[-1]
        if quick:
            Image.fromarray(imgs[-1]).save(os.path.join(out, name + '.png'))
        if not quick:
            Image.fromarray(np.concatenate(imgs, 1)).save(os.path.join(out, name + '.png'))
    if quick:                                                             # (front, side and back in one strip)
        Image.fromarray(np.concatenate([full[k] for k in ('closeup-front', 'closeup-side', 'closeup-back')], 1)).save(
            os.path.join(out, 'quick.png'))
    if '--crown' in a:
        # (the crown from above, in clay and colour: ridges and seams between sections do not show from the front)
        import math
        row = []
        for az, el in ((0, 60), (40, 45)):
            for mode in ('clay', 'full'):
                cv = RD.Canvas(size, az, (-0.75, 0.75, -0.75, 0.75))
                e = math.radians(el); up, fw = cv.up.copy(), cv.fwd.copy()
                cv.up = up * math.cos(e) + fw * math.sin(e) * CROWN_SIGN; cv.fwd = fw * math.cos(e) - up * math.sin(e) * CROWN_SIGN
                c0 = np.array([0.0, 0.06, 0.95]); o = (c0 @ cv.right, c0 @ cv.up)      # (centred on the top of the head)
                cv.box = (o[0] - 0.75, o[0] + 0.75, o[1] - 0.75, o[1] + 0.75)
                bg, zb = backdrop(cv, ('crown', az, el), mode)
                row.append((RD.to_srgb(RD.render_hair(cv, paths, mode, bg, zb, hairs)) * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(row, 1)).save(os.path.join(out, 'crown.png'))
        lap('crown')
    if '--photo' in a:
        fr = full['closeup-front']
        ph = Image.open(a[a.index('--photo') + 1]).convert('RGB')
        ph = ph.resize((int(ph.width * fr.shape[0] / ph.height), fr.shape[0]))
        Image.fromarray(np.concatenate([np.asarray(ph), fr], 1)).save(os.path.join(out, 'beside-photo.png'))
    lap('closeups')
    rep['seconds'] = dict(times, total=round(T[-1] - T[0], 1))
    json.dump(rep, open(os.path.join(out, 'render.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != 'gauges'}))


if __name__ == '__main__':
    main()
