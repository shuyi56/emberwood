import sys, pickle, glob, os, numpy as np
sys.path.insert(0, 'tools/hair')
import scalp_field as SF, photo_waves as PW
from PIL import Image, ImageDraw
S = os.path.dirname(os.path.abspath(__file__))
f = sorted(glob.glob(S + '/photo-cache/comb-*.pkl'), key=os.path.getmtime)[-1]
paths, hairs, _, kind, x = pickle.load(open(f, 'rb'))
pts = SF.load_head(sys.argv[1]); surf = SF.Surface(pts)
paths = [PW._stand_at_part(PW._fair_head(p, surf)) for p in paths]
paths = PW._head_clumps(paths, x['hand'], kind, x['root'], surf)
k = 5; x0, y0, x1, y1 = 100, 60, 420, 300
im = Image.open('/mnt/project-files/hair-rebuild/wavy-photo/reference.jpg').convert('RGB').crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k))
d = ImageDraw.Draw(im)
P = lambda X, Z: ((262 + 232 * X - x0) * k, (348 - 232 * Z - y0) * k)
# skull and hair outlines in the front view: max z per x column, and max |x| per z row
A = np.concatenate([p[:max(h, 2) + 1] for p, h in zip(paths, x['hand'])])
def outline(Q, col):
    xs = np.linspace(-0.6, 0.6, 121); top = []
    for a, b in zip(xs[:-1], xs[1:]):
        m = (Q[:, 0] >= a) & (Q[:, 0] < b)
        if m.any(): top.append(((a + b) / 2, Q[m, 2].max()))
    d.line([P(*t) for t in top], fill=col, width=3)
    return dict((round(t[0], 2), t[1]) for t in top)
sk = outline(pts[pts[:, 2] > 0.5], (0, 255, 255)); hr = outline(A, (255, 230, 0))
for xx in (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
    kx = min(sk, key=lambda v: abs(v - xx)); print(xx, 'skull z %.3f hair z %.3f lift mm %.0f' % (sk[kx], hr[kx], (hr[kx] - sk[kx]) * 219))
im.save(sys.argv[2])
