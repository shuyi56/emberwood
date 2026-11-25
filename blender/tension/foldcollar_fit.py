"""foldcollar_fit.py BUILD OUTDIR [--rounds N] [--fixed '{opts}'] [--start '{opts}']: fit foldcollar.py's controls to the photo's collar front.

Score: the mean distance, both ways, between the photo's collar-front polygon (G, T, K, C; polycheck.REF) and the
outline of the collar visible from the photo's camera (depth buffer with the coat, the scarf's cover laid over it), in
lapel spans, plus 0.2 x the 95th percentile strain of the laid cloth past 15% (a shape that needs the cloth stretched
is not a cut), plus 0.3 x the gap between the outer corner K's radius and the photo's (polycheck.PHOTO_CURVE).
--fixed: options passed on unchanged; --start: the controls' starting values. Coordinate search: each control tried a step up and down, the best kept, steps halved when none helps."""
import sys, os, json, math, numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'collar'))
import polycheck as pc, foldcollar as fc
from scipy import ndimage
from scipy.spatial import cKDTree
import contourpy
CTRL = [('width', 2.2, 0.3, (1.2, 3.6)), ('stand', 0.9, 0.2, (0.3, 1.8)), ('fold_angle', 45.0, 8.0, (20.0, 75.0)),
        ('open', 25.0, 10.0, (5.0, 70.0)), ('corner', 0.5, 0.2, (0.1, 1.2)), ('seam_from', 0.0, 8.0, (-40.0, 40.0)),
        ('fold_k', 3.0, 1.0, (1.0, 8.0))]
FIXED = {}


def outline(B, X, T, az=30.0):
    _, unit = pc.lapel_outline(B, az); M = np.load(B + '/build.npz', allow_pickle=True)
    s = unit['P'][0] - unit['L0'][0]
    box = (unit['L0'][0] - 0.6 * s, unit['P'][0] + 1.0 * s, unit['L0'][1] - 0.6 * s, unit['L0'][1] + 1.6 * s)
    L = np.load(B + '/lapel.blend.lapels.npz'); xcut = float(L['crease2'][0][0]); far = X[T].mean(1)[:, 0] < xcut
    ids = pc.zbuffer([(M['body_X'], M['body_T']), (X, T[~far]), (X, T[far])], az, box, 480); H, W = ids.shape
    vis = ids == 1
    jj, ii = np.mgrid[0:H, 0:W]
    U_ = pc.to_unit(np.column_stack([box[0] + (ii.ravel() + 0.5) / W * (box[1] - box[0]), box[3] - (jj.ravel() + 0.5) / H * (box[3] - box[2])]), unit)
    g, t_ = pc.photo_norm(pc.REF['collar front']['pts'][0]), pc.photo_norm(pc.REF['collar front']['pts'][1]); r_ = pc.photo_norm(pc.REF['lapel']['pts'][1])
    side = lambda a, b, q: (b[0] - a[0]) * (q[:, 1] - a[1]) - (b[1] - a[1]) * (q[:, 0] - a[0])
    vis &= ~((side(g, t_, U_) > 0) & (side(r_, g, U_) > 0)).reshape(H, W)          # (the scarf)
    lab, n = ndimage.label(vis)
    if n == 0: return None
    big = 1 + int(np.argmax(ndimage.sum(vis, lab, range(1, n + 1))))
    Ln = max(contourpy.contour_generator(z=np.pad((lab == big).astype(float), 1)).lines(0.5), key=len) - 1.0
    P2 = np.column_stack([box[0] + (Ln[:, 0] + 0.5) / W * (box[1] - box[0]), box[3] - (Ln[:, 1] + 0.5) / H * (box[3] - box[2])])
    return pc.to_unit(P2, unit)


def ref_samples(n=40):
    R = np.array([pc.photo_norm(p) for p in pc.REF['collar front']['pts']]); o = []
    for i in range(len(R)):
        a, b = R[i], R[(i + 1) % len(R)]; o += [a + (b - a) * k / n for k in range(n)]
    return np.array(o)


def evaluate(args):
    B, x, fixed = args; opt = dict(fixed); opt.update({k: float(v) for (k, *_), v in zip(CTRL, x)})
    try:
        R, info = fc.build(B, opt); X, st = fc.relax(R, B, opt)
        if opt.get('shape_cut'): R['pts'] = X; _, st, _ = fc.shape_cut(R, opt)        # (the stretch against the shaped cut, when that is the cut)
        O = outline(B, X, R['tri'])
        if O is None: return dict(score=9.0, opt=opt)
        Rs = ref_samples(); gap = 0.5 * (cKDTree(O).query(Rs)[0].mean() + cKDTree(Rs).query(O)[0].mean())
        p95 = float(np.percentile(np.abs(st), 95)); ci = pc.corners(O, True)
        kref = np.array(pc.photo_norm(pc.REF['collar front']['pts'][2])); rK = None; pen = 0.0
        if len(ci):
            _, rad = pc.curve_measure(O, ci); dK = np.linalg.norm(O[ci] - kref, axis=1); iK = int(np.argmin(dK))
            if dK[iK] < 0.2: rK = float(rad[iK]); pen = 0.3 * abs(rK - pc.PHOTO_CURVE['K']['radius'])
        return dict(score=float(gap + 0.2 * max(p95 - 0.15, 0) + pen), gap=float(gap), strain_p95=p95, sides=len(ci), K_radius=rK, opt=opt)
    except Exception as e:
        return dict(score=9.0, err=repr(e), opt=opt)


if __name__ == '__main__':
    B = os.path.abspath(sys.argv[1]); outd = os.path.abspath(sys.argv[2]); os.makedirs(outd, exist_ok=True)
    rounds = int(sys.argv[sys.argv.index('--rounds') + 1]) if '--rounds' in sys.argv else 5
    FIXED = json.loads(sys.argv[sys.argv.index('--fixed') + 1]) if '--fixed' in sys.argv else {}
    st0 = json.loads(sys.argv[sys.argv.index('--start') + 1]) if '--start' in sys.argv else {}
    x = np.array([st0.get(c[0], c[1]) for c in CTRL], float); h = np.array([c[2] for c in CTRL]); lo = np.array([c[3][0] for c in CTRL]); hi = np.array([c[3][1] for c in CTRL])
    log = open(outd + '/fit.log', 'a')
    def P(*a):
        print(*a, flush=True); print(*a, file=log); log.flush()
    with ProcessPoolExecutor(4) as pool:
        best = evaluate((B, x, FIXED)); P('start', json.dumps(best))
        for rd in range(rounds):
            cand = []
            for i in range(len(x)):
                for sg in (1, -1):
                    y = x.copy(); y[i] = np.clip(y[i] + sg * h[i], lo[i], hi[i])
                    if abs(y[i] - x[i]) > 1e-9: cand.append(y)
            res = list(pool.map(evaluate, [(B, y, FIXED) for y in cand]))
            gains = sorted([(r['score'], k) for k, r in enumerate(res)])
            imp = [(sc, k) for sc, k in gains if sc < best['score'] - 1e-4]
            if not imp: h *= 0.5; P('round %d: no step helps, steps halved' % (rd + 1)); continue
            # the best single step, and all the helping steps taken together
            y = x.copy()
            for sc, k in imp:
                d = cand[k] - x; i = int(np.argmax(np.abs(d)))
                if y[i] == x[i]: y[i] = cand[k][i]
            both = evaluate((B, y, FIXED)); one = res[imp[0][1]]
            new = both if both['score'] < one['score'] else one
            x = np.array([new['opt'][c[0]] for c in CTRL]); best = new
            P('round %d' % (rd + 1), json.dumps(best))
    json.dump(best, open(outd + '/best.json', 'w'), indent=1); P('best', json.dumps(best))
