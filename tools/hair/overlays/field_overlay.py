"""The whole flow field drawn over the reference photo, front view: field_overlay.py HEAD CACHE OUT.png [--locks].
Fringe orange, face framing magenta, front fall blue, back fall green. Stages come from the render's cache."""
import sys, os, json, gzip, hashlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import scalp_field as SF, photo_waves as PW, wavy_flow as WF
from PIL import Image, ImageDraw
head, cache, out = sys.argv[1:4]
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
hk = md5(head)
pts = SF.load_head(head)
surf, F = PW._cached(cache, 'head', (hk, md5(SF.__file__)), lambda: (lambda sf: (sf, SF.build(sf)))(SF.Surface(pts)))
coll, depth = WF.colliders(surf, json.load(gzip.open(head))['bodies']['mannequin'])
Fd = PW.field2(F, surf, depth, 16000, cache=cache, head_key=hk, log=lambda *a: None)
if '--locks' in sys.argv:
    Fd, _ = PW.lock_fall(Fd, log=lambda *a: None)
k = 3
im = Image.open('/mnt/project-files/hair-rebuild/wavy-photo/reference.jpg').convert('RGB')
im = im.resize((im.width * k, im.height * k))
d = ImageDraw.Draw(im)
P = lambda q: ((262 + 232 * q[0]) * k, (348 - 232 * q[2]) * k)
col = {0: (255, 150, 30), 1: (230, 60, 170), 2: (60, 140, 255), 3: (60, 220, 110), -1: (200, 200, 200)}
L, B = Fd['line'], Fd['basin']
order = sorted(range(len(L)), key=lambda i: -np.mean(L[i][:, 1]))          # (back to front)
for i in order[::12]:
    l = L[i]
    if int(B[i]) == 3:                                                   # (the back fall shows only outside the face and neck)
        l = l[(np.abs(l[:, 0]) > 0.30) | (l[:, 2] < -0.35)]
    if len(l) > 1:
        d.line([P(q) for q in l], fill=col[int(B[i])], width=1)
print(json.dumps(Fd['gauges'].get('half_width')), json.dumps(Fd['gauges'].get('smooth_falls', {}).get('ok')))
im.save(out)
