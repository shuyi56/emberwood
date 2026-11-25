"""collar_inverse.py LAPELDIR OUTDIR [--rounds 3]: solve the collar's controls for the photo's collar front.

John, 2026-11-24: "Solve for the proper collar". The collar's visible corners are not cut corners (T and K come from
its roll and its turn round the neck, see polycheck.py), so there is no closed form from cut to view. The controls are
nudged one at a time, the outline from the photo's camera is measured on each build, and the step goes down the
measured slope (finite differences, then a line search over three step lengths), all builds in parallel.

Score: the outline gap, the mean distance from the photo's collar-front polygon (G, T, K, C, with the scarf over its
front, sampled along its edges) to the model's visible collar outline and back, in lapel spans; plus a penalty past
the approved limit on the collar's tip fold (15.6 deg, feasible.py), since a shape that hairpins is not a collar."""
import sys, os, json, subprocess, math, numpy as np
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import polycheck as pc
REPO = os.path.dirname(os.path.dirname(HERE))
CTRL = [('stand', 0.4, 0.15, (0.15, 1.5)), ('fall_over', 0.6, 0.2, (0.2, 2.0)), ('point_len', 0.9, 0.15, (0.3, 1.6)),
        ('notch_angle', 40.0, 10.0, (15.0, 80.0)), ('rise', 1.3, 0.25, (0.4, 2.5)), ('fall_side', 0.0, 0.2, (-0.6, 1.2))]
FIXED = {"roll_r": 0.16, "join_h": 0.08, "join_run": 2}
TIP_LIM = 15.6


def ref_samples(n=40):
    R = np.array([pc.photo_norm(p) for p in pc.REF['collar front']['pts']]); out = []
    for i in range(len(R)):
        a, b = R[i], R[(i + 1) % len(R)]; out += [a + (b - a) * k / n for k in range(n)]
    return np.array(out)


def build(lap, out, x):
    os.makedirs(out, exist_ok=True)
    for f in ('lapel.blend.lapels.npz', 'lapel.blend.strain.npz'):
        if not os.path.exists(out + '/' + f): os.symlink(os.path.abspath(lap + '/' + f), out + '/' + f)
    opt = dict(FIXED, **{k: float(v) for (k, *_), v in zip(CTRL, x)})
    with open(out + '/collar.log', 'w') as lg:
        subprocess.run([sys.executable, '-I', REPO + '/blender/collar/notchcollar.py', lap + '/lapel.blend', out + '/out.blend', json.dumps(opt)], stdout=lg, stderr=subprocess.STDOUT, cwd=REPO)
    subprocess.run([sys.executable, '-I', REPO + '/blender/tension/dump_build.py', out + '/out.blend', out + '/build.npz'], capture_output=True, cwd=REPO)
    g = subprocess.run([sys.executable, HERE + '/collar_gauge.py', out + '/out.blend.collar.npz', out + '/collar_gauge.npz'], capture_output=True)
    subprocess.run([sys.executable, HERE + '/join.py', out, out + '/join.json'], capture_output=True)
    if os.path.exists(out + '/out.blend'): os.remove(out + '/out.blend')
    json.dump(opt, open(out + '/opt.json', 'w'))
    return score(out)


def score(B):
    try:
        _, unit = pc.lapel_outline(B, 30.0); co, _, _ = pc.collar_outline(B, 30.0, unit)
        M = pc.to_unit(co['P'], unit); Rs = ref_samples()
        from scipy.spatial import cKDTree
        g1 = cKDTree(M).query(Rs)[0].mean(); g2 = cKDTree(Rs).query(M)[0].mean()   # (the photo to the model, the model to the photo)
        J = json.load(open(B + '/join.json')); tip = max(J['1']['tip_fold'], J['2']['tip_fold'])
        r = pc.check(B); sides = [p['model_sides'] for p in r['pieces'] if p['piece'] == 'collar front'][0]
        s = 0.5 * (g1 + g2) + 0.005 * max(tip - TIP_LIM, 0)
        return dict(score=float(s), gap=float(0.5 * (g1 + g2)), tip=float(tip), sides=int(sides))
    except Exception as e:
        return dict(score=9.0, gap=9.0, tip=999.0, sides=0, err=str(e))


if __name__ == '__main__':
    lap, outd = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); rounds = int(sys.argv[sys.argv.index('--rounds') + 1]) if '--rounds' in sys.argv else 3
    x = np.array([c[1] for c in CTRL], float); h = np.array([c[2] for c in CTRL]); lo = np.array([c[3][0] for c in CTRL]); hi = np.array([c[3][1] for c in CTRL])
    log = open(outd + '/solve.log', 'a') if os.path.isdir(outd) else (os.makedirs(outd) or open(outd + '/solve.log', 'a'))
    def P(*a):
        print(*a); print(*a, file=log); log.flush()
    pool = ThreadPoolExecutor(4); n = [0]
    def run(xx):
        n[0] += 1; d = '%s/b%03d' % (outd, n[0]); return d, xx, build(lap, d, xx)
    d0, _, f0 = run(x); P('start', dict(zip([c[0] for c in CTRL], x.round(3))), f0, d0)
    best = (f0['score'], x.copy(), d0, f0)
    for rd in range(rounds):
        xs = [np.clip(x + h * np.eye(len(x))[i], lo, hi) for i in range(len(x))]
        res = list(pool.map(run, xs))
        gr = np.array([(r[2]['score'] - best[0]) / max(abs(xx[i] - x[i]), 1e-9) for i, (r, xx) in enumerate(zip(res, xs))])
        for r in res:
            if r[2]['score'] < best[0]: best = (r[2]['score'], r[1].copy(), r[0], r[2])
        P('round %d slopes' % (rd + 1), dict(zip([c[0] for c in CTRL], (gr * h).round(4))))
        dirn = -gr * h * h; dirn /= max(np.linalg.norm(dirn / h), 1e-9)          # (down the slope, in steps of each control's own size)
        cand = [np.clip(best[1] + a * dirn, lo, hi) for a in (0.75, 1.5, 3.0)]
        for r in pool.map(run, cand):
            P('   step', dict(zip([c[0] for c in CTRL], r[1].round(3))), r[2], r[0])
            if r[2]['score'] < best[0]: best = (r[2]['score'], r[1].copy(), r[0], r[2])
        P('round %d best' % (rd + 1), best[0], dict(zip([c[0] for c in CTRL], best[1].round(3))), best[3], best[2])
        if np.allclose(best[1], x): h *= 0.5
        x = best[1].copy()
    json.dump(dict(opt=dict(FIXED, **{c[0]: float(v) for c, v in zip(CTRL, best[1])}), result=best[3], build=best[2]), open(outd + '/best.json', 'w'), indent=1)
    P('best', best[2], best[3])
