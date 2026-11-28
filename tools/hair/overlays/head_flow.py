"""Flow diagram of the hair ON THE HEAD only: head_flow.py HEAD CACHE OUT.png [--every 6] [--blue-only] ['PW.NAME = value' ...].
Every line from its root to where it leaves the head (the bangs to their ends), nothing below. Four panels: front, side
(face left), above (face down the page), back. Only the half of the head facing the viewer is drawn. Skull grey; the dot
is the root; fringe orange, face framing magenta, front fall blue, back fall green."""
import sys, os, json, gzip, hashlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import scalp_field as SF, photo_waves as PW, wavy_flow as WF
from PIL import Image, ImageDraw
head, cache, out = sys.argv[1:4]
every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 6
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
hk = md5(head)
for a_ in sys.argv:
    if a_.startswith('PW.'):
        exec(a_)
BLUE_ONLY = '--blue-only' in sys.argv; FRINGE_BLUE = PW.FRINGE_FROM_BLUE
pts = SF.load_head(head)
surf, F = PW._cached(cache, 'head', (hk, md5(SF.__file__)), lambda: (lambda sf: (sf, SF.build(sf)))(SF.Surface(pts)))
coll, depth = WF.colliders(surf, json.load(gzip.open(head))['bodies']['mannequin'])
Fd = PW.field2(F, surf, depth, 16000, cache=cache, head_key=hk, log=lambda *a: print(*[str(v)[:400] for v in a]) if '_flow' in str(a[0]) else None)
L, B, H = Fd['line'], Fd['basin'], Fd['release']
col = {0: (235, 130, 20), 1: (215, 50, 160), 2: (40, 110, 230), 3: (30, 160, 80), -1: (120, 120, 120)}
R = []
for i in range(0, len(L), every):
    l = L[i] if int(B[i]) == 0 or H[i] < 0 else L[i][:H[i] + 1]
    if len(l) > 2 and (not BLUE_ONLY or int(B[i]) == 2 or (int(B[i]) == 0 and FRINGE_BLUE)):
        R.append((l, col[2] if BLUE_ONLY else col[int(B[i])]))
W, Hh = 900, 900; C = SF.CENTRE
im = Image.new('RGB', (2 * W, 2 * Hh), (250, 250, 248))
# view: (title, screen x axis and sign, screen y axis and sign, facing test on a point)
views = [('FRONT', (0, 1), (2, 1), lambda q: q[:, 1] < C[1] + 0.02, (0.0, 0.70)),
         ('SIDE (face left)', (1, 1), (2, 1), lambda q: q[:, 0] > 0.0, (0.08, 0.70)),
         ('ABOVE (face down the page)', (0, 1), (1, 1), lambda q: q[:, 2] > 0.70, (0.0, 0.08)),
         ('BACK', (0, -1), (2, 1), lambda q: q[:, 1] > C[1] - 0.02, (0.0, 0.70))]
k = 950
for n, (title, ax, ay, face, (cx, cy)) in enumerate(views):
    pim = Image.new('RGB', (W, Hh), (250, 250, 248)); d = ImageDraw.Draw(pim)
    P = lambda q: (W / 2 + ax[1] * (q[ax[0]] - cx) * k, Hh / 2 - (q[ay[0]] - cy) * k)
    sk = pts[face(pts)][::4]
    for q in sk:
        d.point(P(q), fill=(185, 185, 185))
    for l, c in R:
        m = face(l)
        if m.sum() < 3 or not m[0]:
            continue
        seg = l[:int(np.argmin(m)) if not m.all() else len(l)]            # (from the root until it passes out of view)
        if len(seg) > 2:
            d.line([P(q) for q in seg], fill=c, width=1)
            x_, y_ = P(seg[0]); d.ellipse((x_ - 1.5, y_ - 1.5, x_ + 1.5, y_ + 1.5), fill=(20, 20, 20))
    d.text((14, 10), title, fill=(0, 0, 0)); d.rectangle((0, 0, W - 1, Hh - 1), outline=(120, 120, 120))
    im.paste(pim, ((n % 2) * W, (n // 2) * Hh))
im.save(out)
