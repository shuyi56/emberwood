"""foldcollar_gauge.py BUILD COLLAR.npz [OUT.json]: what the folded collar's fit must not give away.

John, 2026-11-24: "Why does every adjustment we make reduce creasing and makes the collar thinner". Every gauge until
now (stretch, the join's kink and turn, the outline gap from one camera) rewarded a smoother, flatter, closer collar,
and none measured the crease or the collar's body, so each round traded them off. These do:

  crease    along the roll line, per 0.5 of its length: the turn of the cloth's normal from the stand (0.2 to 0.35
            below the line) to the fall (0.2 to 0.35 past it), and the share of that turn made within 0.08 of the line
            (1: a pressed crease; 0.3: a soft roll). Photo: a pressed crease round the back, softening to the ends.
  body      how far the fall's cloth stands off the stand's (mid-surface to mid-surface, round the back: a cloth's
            thickness is touching, more is a collar with a roll in it), and off the neck's skin if BUILD has skin.npz
  on coat   the fall's mid-surface off the coat's own (BUILD/build.npz, the bare mesh). The coat's cloth stands
            `th` outside its mesh and the collar's is centred on its own, so anything under 1.5 th is sunk into the coat
  thickness the collar's cloth as it will be drawn, against the coat's
BUILD needs build.npz; `th` is the coat's Solidify thickness (0.1669 = 4.4 mm) unless given as --th."""
import sys, json, numpy as np
from scipy.spatial import cKDTree


def gauge(B, npz, th=0.1669):
    D = np.load(npz); X, T, fall, F = D['pts'], D['tri'], D['fall'], D['fold']
    uv = D['uv_rect'] if 'uv_rect' in D.files else D['uv']; Ls = uv[:, 0].max()
    n = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]); n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    c = uv[T].mean(1); dq, jf = cKDTree(F).query(c); sgn = np.where(c[:, 1] > np.interp(c[:, 0], F[:, 0], F[:, 1]), 1.0, -1.0); s = sgn * dq
    turn, sharp, xs = [], [], []
    for x0 in np.arange(0, Ls, 0.5):
        m = (c[:, 0] >= x0) & (c[:, 0] < x0 + 0.5)
        g = lambda lo, hi: n[m & (s >= lo) & (s < hi)]
        a, b, a1, b1 = g(-0.35, -0.2), g(0.2, 0.35), g(-0.08, 0), g(0, 0.08)
        if min(len(a), len(b), len(a1), len(b1)) < 3: continue
        u = lambda v: v.mean(0) / np.linalg.norm(v.mean(0)); ang = lambda p, q: float(np.degrees(np.arccos(np.clip(u(p) @ u(q), -1, 1))))
        tt = ang(a, b); turn.append(tt); sharp.append(ang(a1, b1) / max(tt, 1e-6)); xs.append(x0 + 0.25)
    xs, turn, sharp = np.array(xs), np.array(turn), np.array(sharp); xe = np.minimum(xs, Ls - xs); back = xe > 3.0
    out = dict(crease_turn_back=float(np.median(turn[back])), crease_sharp_back=float(np.median(sharp[back])),
               crease_turn_ends=float(np.median(turn[~back])), crease_sharp_ends=float(np.median(sharp[~back])))
    # body: the fall off the stand, round the back
    xv = np.minimum(uv[:, 0], Ls - uv[:, 0]); st_ = (~fall) & (xv > 3.0); fl_ = fall & (xv > 3.0)
    hx = np.interp(uv[:, 0], F[:, 0], F[:, 1]); far = fl_ & (uv[:, 1] - hx > 0.3)
    d = cKDTree(X[st_]).query(X[far])[0]; out.update(body_off_stand=float(np.median(d)), body_off_stand_th=float(np.median(d) / th))
    # on the coat
    M = np.load(B + '/build.npz', allow_pickle=True); BX, BT = M['body_X'], M['body_T']; fn = np.cross(BX[BT[:, 1]] - BX[BT[:, 0]], BX[BT[:, 2]] - BX[BT[:, 0]])
    vn = np.zeros_like(BX)
    for k in range(3): np.add.at(vn, BT[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]
    dd, j = cKDTree(BX).query(X[fall]); sd = ((X[fall] - BX[j]) * vn[j]).sum(1); nr = dd < 0.45
    out.update(on_coat_median_th=float(np.median(sd[nr]) / th), sunk_share=float((sd[nr] < 1.5 * th).mean()), over_coat_points=int(nr.sum()))
    js = json.load(open(npz + '.json')) if __import__('os').path.exists(npz + '.json') else {}
    out['thickness_th'] = float(js.get('opt', {}).get('thickness', 0.06) / th)
    return out


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]; th = float(sys.argv[sys.argv.index('--th') + 1]) if '--th' in sys.argv else 0.1669
    if '--th' in sys.argv: a.remove(sys.argv[sys.argv.index('--th') + 1])
    g = gauge(a[0].rstrip('/'), a[1], th)
    if len(a) > 2: json.dump(g, open(a[2], 'w'), indent=1)
    print('foldcollar_gauge: ' + ', '.join('%s %s' % (k, ('%.2f' % v) if isinstance(v, float) else v) for k, v in g.items()))
