import sys, pickle, glob, os, numpy as np
sys.path.insert(0, 'tools/hair')
import scalp_field as SF, photo_waves as PW
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
f = sorted(glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/photo-cache/comb-*.pkl'), key=os.path.getmtime)[-1]
paths, hairs, _, kind, x = pickle.load(open(f, 'rb'))
pts = SF.load_head(sys.argv[1]); surf = SF.Surface(pts)
paths = [PW._fair_head(p, surf) for p in paths]; hand = x['hand']
k = 5; x0, y0, x1, y1 = 120, 80, 400, 300
im = Image.open('/mnt/project-files/hair-rebuild/wavy-photo/reference.jpg').convert('RGB').crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k))
d = ImageDraw.Draw(im)
P = lambda X, Z: ((262 + 232 * X - x0) * k, (348 - 232 * Z - y0) * k)
for sd, col in ((1.0, (255, 40, 40)), (-1.0, (40, 120, 255))):
    fm = np.where((kind == 1) & (x['root'][:, 0] * sd > 0))[0]
    sm_ = [paths[i][:max(hand[i], 2) + 1] for i in np.where((kind != 1) & (x['root'][:, 0] * sd > 0) & (hand >= 0))[0] if abs(SF.theta(paths[i][:1])[0]) < PW.FALL_SPLIT]
    sp = np.concatenate([p_[2 * len(p_) // 3:] for p_ in sm_]); st = np.concatenate([np.gradient(p_, axis=0)[2 * len(p_) // 3:] for p_ in sm_])
    st /= np.maximum(np.linalg.norm(st, axis=1, keepdims=True), 1e-9)
    L = PW.bang_sheet(x['root'][fm], surf, seed=3, side=(cKDTree(sp), sp, st))
    for l in L[::4]:
        d.line([P(q[0], q[2]) for q in l], fill=col, width=2)
    fr_ = [p_ for p_ in sm_ if p_[0, 1] < SF.CENTRE[1] + 0.02]
    for p_ in fr_[::max(len(fr_) // 60, 1)]:
        q_ = p_[p_[:, 1] < SF.CENTRE[1] + 0.10]
        if len(q_) > 2:
            d.line([P(q[0], q[2]) for q in q_], fill=(255, 220, 0), width=2)
im.save(sys.argv[2])
