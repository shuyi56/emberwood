"""The heel's shape and height found by gradient descent on the support gauge (blender/heel/study.py).

    python3 tools/heel/optimize.py DUMP.npz [--out OUT.json] [--plot OUT.png] [--steps 60] [--shapes]

DUMP.npz is the feet a build saved (heel.py with HEEL_DUMP=PATH). Runs outside Blender (numpy; matplotlib for --plot).

What is varied (the design vector):
    h        the heel's height, cm: the rise of the sole of the foot at the seat point
    width    the tread's width, a share of the seat's (1: straight sides, a block; less: the sides taper in)
    length   the tread's length, a share of the block's (the block's tread runs from the breast as far behind the seat
             point as in front of it; less: a shorter tread)
    rake     how far the tread's middle sits in front of the seat point, a share of the block's tread length (0: under
             the seat point, a straight back; more: the back raked in under the foot, as a dogger's or a packer's)
    arch     the footbed's fill through the waist (1: the insole follows the arch; 0: a straight shank line)
    cup      the footbed's cup round the heel (1: follows the heel; 0: a flat seat)

What is minimised: a sum of terms, each a gauge reading over the size it should stay under, squared. The gauge's own
physics is the study's; which readings count and how much is this file's (WEIGHTS), and is EXPERIMENTAL unless the
term names a source.

The gradient is taken by central differences (the gauge is a contact solve, not a formula) and the step is Adam's,
each variable kept inside its bounds."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'blender', 'heel'))
import study as S  # noqa: E402

NAMES = ('h', 'width', 'length', 'rake', 'arch', 'cup')
LO = np.array([0.5, 0.45, 0.45, 0.0, 0.0, 0.0])
HI = np.array([5.0, 1.00, 1.00, 0.35, 1.0, 1.0])
START = np.array([2.5, 0.85, 1.00, 0.0, 0.7, 0.7])
EPS = np.array([0.04, 0.02, 0.02, 0.01, 0.03, 0.03])

# Each term: what the gauge reads, over the size it is judged against, squared, times its weight.
WEIGHTS = dict(
    centred=1.0,     # the rear foot's load on the tread's middle, front to back (Golding: the heel stands on the seat
                     # point's perpendicular), over half the tread's length
    shares=1.0,      # the load's shares under the heel, midfoot, forefoot and toes, against standing barefoot
                     # (Cavanagh 1987): a heel that throws the load forward leaves them
    peak=0.5,        # the highest pressure under the foot, over barefoot's heel peak (Cavanagh 1987, 139 kPa)
    slide=1.0,       # the push toward the toes the lacing must hold at the walking peak, over body weight
    tip=1.0,         # how far short the heel is of standing EXP['tip_min'] degrees of sideways tilt (EXPERIMENTAL limit)
    sag=1.0,         # the waist's sag at the walking peak past EXP['waist_sag'] mm (EXPERIMENTAL limit)
    stack=0.25,      # the stack's highest compression at the tread, the rear load at the walking peak over the tread's
                     # area and raised by how far off its middle the load stands (a short column: F/A (1 + 6 e/l)),
                     # over barefoot's heel peak. EXPERIMENTAL
    strike=0.25,     # at heel strike the foot lands on the tread's back edge: how far that edge lies behind the ankle,
                     # the lever the strike's force turns the foot down with, over the foot's length. EXPERIMENTAL: no
                     # source gives a limit; it is what a raked back is for
    leather=0.05,    # the stack's leather, tread area times height, over the block's at 2 cm. EXPERIMENTAL
)


def design(f, x):
    """The study's design for the vector x (see the top)."""
    h, width, length, rake, arch, cup = x
    d = S.derive(f, float(h), arch_fill=float(arch), cup=float(cup), top_of_seat=float(width))
    block = d['top_len']
    d['top_len'] = float(length) * block
    d['top_y'] = d['seat_y'] - float(rake) * block          # (forward is -y)
    d['top_back_y'] = d['top_y'] + 0.5 * d['top_len']
    d['block_len'] = block
    return d


def terms(Pl, step, f, k, e, x):
    d = design(f, x)
    o = S.support(Pl, step, f, d, k, e)
    half_len = 0.5 * d['top_len'] * S.MM
    W = o['W']
    t = {}
    t['centred'] = (o['rear_cop_off_mm'] / half_len) ** 2
    t['shares'] = sum((o['share'][n] - S.CAVANAGH[n]) ** 2 for n in ('heel', 'midfoot', 'forefoot', 'toes')) / 0.05
    t['peak'] = (o['peak_all'] / S.CAVANAGH['heel_peak']) ** 2
    t['slide'] = (o['vamp_N_walk'] / (2.0 * W)) ** 2
    t['tip'] = (max(0.0, S.EXP['tip_min'] - o['tip_deg']) / S.EXP['tip_min']) ** 2
    t['sag'] = max(0.0, o['shank_sag_walk'] - S.EXP['waist_sag']) ** 2
    area = d['top_w'] * d['top_len'] * S.MM ** 2 * (np.pi / 4)          # mm^2 (the tread a rounded plan)
    Fr = o['rear_N'] * S.GAIT_PEAK * 2.0
    sigma = Fr / area * (1.0 + 6.0 * abs(o['rear_cop_off_mm']) / (d['top_len'] * S.MM)) * 1e3   # kPa
    t['stack'] = (sigma / S.CAVANAGH['heel_peak']) ** 2
    lever = max(0.0, d['top_back_y'] - f['ankle'][1])
    t['strike'] = (lever / (0.1 * f['L'])) ** 2
    block_area = 0.85 * d['seat_w'] * d['block_len']
    t['leather'] = (d['top_w'] * d['top_len'] * x[0]) / (block_area * 2.0)
    return t, d, o, dict(sigma_kpa=float(sigma), lever_cm=float(lever * S.CM))


def loss(Pl, step, f, k, e, x):
    t = terms(Pl, step, f, k, e, x)[0]
    return float(sum(WEIGHTS[n] * v for n, v in t.items()))


def descend(Pl, step, f, k, e, x0=START, steps=60, lr=0.05, free=None, log=print):
    """Adam on the loss, the gradient by central differences, x kept in [LO, HI]. free: the variables let move (names);
    the rest stay at x0. Returns (x, the path [(step, loss, x)])."""
    x = np.clip(np.array(x0, float), LO, HI)
    idx = [i for i, n in enumerate(NAMES) if free is None or n in free]
    scale = HI - LO
    m, v = np.zeros(len(x)), np.zeros(len(x))
    path, best = [], (np.inf, x.copy())
    for it in range(steps):
        L = loss(Pl, step, f, k, e, x)
        path.append((it, L, x.copy()))
        if L < best[0]:
            best = (L, x.copy())
        g = np.zeros(len(x))
        for i in idx:
            a, b = x.copy(), x.copy()
            a[i], b[i] = min(HI[i], x[i] + EPS[i]), max(LO[i], x[i] - EPS[i])
            g[i] = (loss(Pl, step, f, k, e, a) - loss(Pl, step, f, k, e, b)) / (a[i] - b[i]) * scale[i]
        m = 0.8 * m + 0.2 * g
        v = 0.95 * v + 0.05 * g * g
        mh, vh = m / (1 - 0.8 ** (it + 1)), v / (1 - 0.95 ** (it + 1))
        rate = lr * (0.5 + 0.5 * np.cos(np.pi * it / steps))          # (eased to rest)
        x = np.clip(x - rate * scale * mh / (np.sqrt(vh) + 1e-8), LO, HI)
        if it % 5 == 0:
            log('step %2d  loss %.4f  %s' % (it, L, '  '.join('%s %.3f' % (n, val) for n, val in zip(NAMES, path[-1][2]))))
    L = loss(Pl, step, f, k, e, x)
    path.append((steps, L, x.copy()))
    if L < best[0]:
        best = (L, x.copy())
    return best[1], path


# The named shapes, as this file's variables. The shapes are the bootmakers' (Stitchdown, Nick's, Frank's: block 90
# degrees all round with the larger tread set back; dogger about 1.25 in, back angled, sides nearly straight; logger
# 1.5 to 2 in, concave, tapering in; packer or riding about 2 in, its back 120 to 130 degrees from the ground, tapering
# in; Cuban, back and sides tapering in, breast straight). The numbers given to each here are a reading of those
# descriptions, EXPERIMENTAL where the sources give none (only the heights and the packer's angle are theirs).
SHAPES = {
    'block':  dict(h=2.0, width=1.00, length=1.00, rake=0.00),
    'cuban':  dict(h=3.2, width=0.80, length=0.80, rake=0.10),
    'dogger': dict(h=3.2, width=0.92, length=0.80, rake=0.12),
    'logger': dict(h=4.4, width=0.80, length=0.78, rake=0.10),
    'packer': dict(h=5.0, width=0.75, length=0.60, rake=0.22),
}


def report(Pl, step, f, k, e, x):
    t, d, o, extra = terms(Pl, step, f, k, e, x)
    s = o['share']
    return dict(
        x={n: float(v) for n, v in zip(NAMES, x)}, loss=float(sum(WEIGHTS[n] * v for n, v in t.items())),
        terms={n: float(WEIGHTS[n] * v) for n, v in t.items()},
        h_breast_cm=o['h_breast_cm'], pitch_deg=o['theta_deg'],
        tread_cm=(d['top_w'] * S.CM, d['top_len'] * S.CM), seat_w_cm=d['seat_w'] * S.CM,
        back_set_in_cm=float((d['seat_y'] + 0.5 * d['block_len'] - d['top_back_y']) * S.CM),
        shares={n: float(v) for n, v in s.items()}, peak_kpa=o['peak_all'], rear_off_mm=o['rear_cop_off_mm'],
        tip_deg=o['tip_deg'], vamp_walk_N=o['vamp_N_walk'], sag_walk_mm=o['shank_sag_walk'],
        shank_need=o['shank_S_need'], shank_has=o['shank_S'], **extra)


def main():
    arg = lambda kk, dd=None: sys.argv[sys.argv.index(kk) + 1] if kk in sys.argv else dd  # noqa: E731
    D = np.load(sys.argv[1])
    P, N, mask, ankle = D['P'], D['N'], D['maskR'], D['ankleR']
    f = S.measure(P, N, mask, ankle)
    Pl, step = S.plantar(P, N, mask)
    k, e, bare = S.calibrate(Pl, step, f)
    out = {}
    print('--- gradient descent, every variable free')
    steps = int(arg('--steps', 60))
    runs = []
    for x0 in (START, np.array([1.2, 1.0, 1.0, 0.0, 1.0, 1.0]), np.array([4.0, 0.7, 0.7, 0.2, 0.4, 0.4])):
        x, path = descend(Pl, step, f, k, e, x0=x0, steps=steps)
        runs.append((loss(Pl, step, f, k, e, x), x, path))
        print('from %s: loss %.4f at %s' % (np.round(x0, 2).tolist(), runs[-1][0], np.round(x, 3).tolist()))
    runs.sort(key=lambda r_: r_[0])
    print('--- the named shapes on this foot (on the footbed the descent found)')
    for name, sh in SHAPES.items():
        x = np.array([sh['h'], sh['width'], sh['length'], sh['rake'], runs[0][1][4], runs[0][1][5]])
        out[name] = r = report(Pl, step, f, k, e, x)
        print('%-7s loss %.3f  %s' % (name, r['loss'], '  '.join('%s %.3f' % kv for kv in r['terms'].items())))
    out['optimum'] = report(Pl, step, f, k, e, runs[0][1])
    out['starts'] = [dict(loss=float(L), x=[float(v) for v in x]) for L, x, _ in runs]
    out['path'] = [dict(step=int(i), loss=float(L), x=[float(v) for v in x]) for i, L, x in runs[0][2]]
    print(json.dumps({kk: vv for kk, vv in out['optimum'].items()}, indent=1, default=float))
    if arg('--out'):
        json.dump(out, open(arg('--out'), 'w'), indent=1, default=float)
    if arg('--plot'):
        plot(out, runs, Pl, step, f, k, e, arg('--plot'))


def plot(out, runs, Pl, step, f, k, e, dst):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for L, x, path in runs:
        ax[0].plot([p[0] for p in path], [p[1] for p in path], lw=2, label='from h %.1f cm' % path[0][2][0])
    ax[0].set_yscale('log')
    ax[0].set_xlabel('step')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Gradient descent: the loss from three starts')
    ax[0].legend()
    hs = np.linspace(0.5, 5.0, 19)
    best = runs[0][1]
    parts = {n: [] for n in WEIGHTS}
    for h in hs:
        x = best.copy()
        x[0] = h
        t = terms(Pl, step, f, k, e, x)[0]
        for n in WEIGHTS:
            parts[n].append(WEIGHTS[n] * t[n])
    ax[1].stackplot(hs, [parts[n] for n in WEIGHTS], labels=list(WEIGHTS))
    ax[1].axvline(best[0], color='k', ls='--')
    ax[1].set_xlabel('heel height at the seat point, cm (other variables at the optimum)')
    ax[1].set_title('What the loss is made of, by height')
    ax[1].legend(fontsize=8)
    names = list(SHAPES) + ['optimum']
    bottom = np.zeros(len(names))
    for n in WEIGHTS:
        vals = np.array([out[nm]['terms'][n] for nm in names])
        ax[2].bar(names, vals, bottom=bottom, label=n)
        bottom += vals
    ax[2].set_title('The named shapes and the optimum on this foot')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)


if __name__ == '__main__':
    main()
