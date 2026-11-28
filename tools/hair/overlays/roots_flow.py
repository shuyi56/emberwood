"""Flow diagrams of the ROOTS ONLY: roots_flow.py HEAD CACHE OUT.png [--mm 40].
The first --mm of every line of the field, nothing else. Four panels: a cross-section across the head through the top
(looking from the front), a cross-section along the head beside the part (looking from the side), the roots from above,
and the roots from the front. Skull grey, each root a line from a dot, coloured by how steeply it leaves the skin
(red 90 degrees, blue flat). Prints the leaving angle and the height reached."""
import sys, os, json, gzip, hashlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import scalp_field as SF, photo_waves as PW, wavy_flow as WF
from PIL import Image, ImageDraw
head, cache, out = sys.argv[1:4]
mm = float(sys.argv[sys.argv.index('--mm') + 1]) if '--mm' in sys.argv else 40.0
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
hk = md5(head)
for a_ in sys.argv:                                                     # (--set 'PW.NAME = value': a trial, not saved)
    if a_.startswith('PW.'):
        exec(a_)
pts = SF.load_head(head)
surf, F = PW._cached(cache, 'head', (hk, md5(SF.__file__)), lambda: (lambda sf: (sf, SF.build(sf)))(SF.Surface(pts)))
coll, depth = WF.colliders(surf, json.load(gzip.open(head))['bodies']['mannequin'])
Fd = PW.field2(F, surf, depth, 16000, cache=cache, head_key=hk, log=lambda *a: None)
MM = PW.MM
R = []; ANG = []; HT = []
for l in Fd['line']:
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(l, axis=0), axis=1))] / MM
    q = l[s <= mm]
    if len(q) < 4:
        continue
    n = q[0] - SF.CENTRE; n /= np.linalg.norm(n)
    t = q[2] - q[0]; t /= max(np.linalg.norm(t), 1e-9)
    ang = np.degrees(np.arcsin(np.clip(t @ n, -1, 1)))
    U = q - SF.CENTRE; r = np.linalg.norm(U, axis=1); U /= r[:, None]
    R.append(q); ANG.append(ang); HT.append(float((r - surf.radius(U)).max() / MM))
ANG = np.array(ANG); HT = np.array(HT)
top = np.array([q[0, 2] > 0.80 for q in R])
print(json.dumps(dict(roots=len(R), leaving_angle_deg_median=round(float(np.median(ANG)), 1), leaving_angle_top_median=round(float(np.median(ANG[top])), 1),
                      height_mm_top_median=round(float(np.median(HT[top])), 1), height_mm_top_p90=round(float(np.percentile(HT[top], 90)), 1))))
W, H = 900, 760
im = Image.new('RGB', (2 * W, 2 * H), (250, 250, 248))
col = lambda a: (int(40 + 200 * np.clip(a / 90, 0, 1)), 60, int(230 - 190 * np.clip(a / 90, 0, 1)))


def panel(ox, oy, title, ax, sel, skull, k, cx, cy, every=1):
    pim = Image.new('RGB', (W, H), (250, 250, 248)); d = ImageDraw.Draw(pim)
    ox_, oy_ = ox, oy; ox = oy = 0
    P = lambda q: (ox + W / 2 + (q[ax[0]] - cx) * k, oy + H / 2 - (q[ax[1]] - cy) * abs(k))
    for q in skull:
        x_, y_ = P(q); d.point((x_, y_), fill=(170, 170, 170))
    idx = [i for i in range(len(R)) if sel(R[i])][::every]
    for i in idx:
        d.line([P(q) for q in R[i]], fill=col(ANG[i]), width=2)
        x_, y_ = P(R[i][0]); d.ellipse((x_ - 2, y_ - 2, x_ + 2, y_ + 2), fill=(20, 20, 20))
    d.text((ox + 14, oy + 10), title + '   (%d roots)' % len(idx), fill=(0, 0, 0))
    d.rectangle((ox, oy, ox + W - 1, oy + H - 1), outline=(120, 120, 120))
    im.paste(pim, (ox_, oy_))


y0 = -0.02; x0 = 0.12; sl = 3.0 * MM
panel(0, 0, 'ACROSS THE HEAD, through the top, seen from the front: slice 6 mm thick', (0, 2), lambda q: abs(q[0, 1] - y0) < sl,
      pts[np.abs(pts[:, 1] - y0) < sl], 1500, 0.0, 0.86)
panel(W, 0, 'ALONG THE HEAD, 26 mm beside the part, seen from the side (face left): slice 6 mm thick', (1, 2), lambda q: abs(q[0, 0] - x0) < sl,
      pts[np.abs(pts[:, 0] - x0) < sl], 1100, 0.10, 0.80)
panel(0, H, 'FROM ABOVE (face down the page): every 12th root', (0, 1), lambda q: q[0, 2] > 0.55, pts[pts[:, 2] > 0.55][::6], 1100, 0.0, 0.08, 12)
panel(W, H, 'FROM THE FRONT: every 12th root in front of the ears', (0, 2), lambda q: q[0, 1] < 0.08, pts[pts[:, 1] < 0.0][::6], 1100, 0.0, 0.80, 12)
d = ImageDraw.Draw(im)
for i, a_ in enumerate((0, 30, 60, 90)):
    d.rectangle((W - 60 + 0, H - 110 + 24 * i, W - 40, H - 94 + 24 * i), fill=col(a_)); d.text((W - 34, H - 108 + 24 * i), '%d deg off the skin' % a_, fill=(0, 0, 0))
im.save(out)
