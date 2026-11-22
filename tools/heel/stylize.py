"""The stylized heel: the optimizer's seat, height and footbed kept, and the heel's body given a flare and a waist,
scored against the straight block with the same support gauge, plus one reading the gauge does not take (the neck).

    python3 tools/heel/stylize.py DUMP.npz [--out OUT.json] [--plot OUT.png]

DUMP.npz as for optimize.py (heel.py with HEEL_DUMP=PATH). Runs outside Blender (numpy; matplotlib for --plot).

The body between the seat (u = 0, under the sole) and the tread (u = 1, on the floor) is a stack of sections, each a
rounded plan w(u) wide and l(u) long:

    s(u) = tread + (seat - tread) * (1 - u) ** flare  -  waist * seat * 4 u (1 - u)

flare   how the taper from the seat to the tread is spent: 1 a straight taper (the block's, the optimizer's); more
        than 1 spends it high, so the heel stands wide under the seat and narrows quickly below it (the flared,
        buttressed top of a Ferragamo-style heel); a straight block (tread = seat) has no taper to spend
waist   a pinch at mid-height as a share of the seat's size, with the tread flaring back out below it (the curved
        waist and flared foot of a Louis or French heel; American Duchess, "Concerning Historical Heels", 2024:
        "a curved 'waist' and flared bottom"; SATRA, heel styles: a spool "narrows in the middle and then widens
        again at the bottom")

The same profile is applied to the width and to the length, so the back and the breast curve as the sides do.

What the style costs, read on the gauge (optimize.terms: the load's centring and shares, the peak pressure, the push
the upper holds, the sideways tip, the waist's sag, the stack's compression at the tread, the strike's lever, the
leather), and one term the gauge does not have:

neck    the highest stress in any section of the body at heel strike: the vertical load over the section's area
        plus the braking force's bending over its section modulus (the force acts at the tread, so the moment
        grows with the height above it). The section is an ellipse w by l: area pi w l / 4, modulus about the
        axis across the foot pi w l^2 / 32 (Roark, an elliptical section). The loads: the vertical peak
        study.GAIT_PEAK x body weight (Perry and Burnfield) all on the heel at strike, and a braking peak of
        BRAKE x body weight (Winter, Biomechanics and Motor Control of Human Movement: the fore-aft force in
        walking peaks near 0.2 body weight). The limit it is judged against is the straight block's own neck
        stress at the optimizer's height: EXPERIMENTAL (a ratio, not a strength: no source gives a breaking stress
        for a stacked leather heel's body, and a real waisted heel is built round a stiffener or cut in wood)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimize as O  # noqa: E402

S = O.S
BRAKE = 0.20          # Winter: the fore-aft (braking) ground force's peak in walking, x body weight
NECK_W = 0.25         # the neck term's weight in the loss, as the stack's (optimize.WEIGHTS['stack']). EXPERIMENTAL

# The variants. 'pump' is measured on John's cream pump (uploads 0145bd9b; refmeasure sheet
# /mnt/project-files/zip-ankle-boot/ref/flared-heel-pump.measure.json): the tread 0.70 of the seat's width, half
# the taper spent by 0.47 of the way down (flare 1.2), the back bowed in 3% of the height (waist 0.03).
VARIANTS = {
    'block':     dict(width=None, flare=1.0, waist=0.00, note='the optimizer\'s straight block (PR 81)'),
    'waisted':   dict(width=None, flare=1.0, waist=0.12, note='the same tread and seat, a curved waist between'),
    'pump':      dict(width=0.70, flare=1.2, waist=0.03, note='measured on the reference pump'),
    'ferragamo': dict(width=0.70, flare=2.2, waist=0.06, note='the pump pushed: a buttressed top, a quick narrowing'),
}


def section(u, seat, tread, flare, waist):
    return tread + (seat - tread) * (1.0 - u) ** flare - waist * seat * 4.0 * u * (1.0 - u)


def neck(d, h_cm, flare, waist, W, n=41):
    """The body's sections and the highest stress in them at heel strike (kPa): returns (kPa, u at the max, the
    sections [(u, w_mm, l_mm, kPa)])."""
    BW = 2.0 * W                                       # (the gauge's W is the load on one foot standing)
    Fv, Fh = S.GAIT_PEAK * BW, BRAKE * BW
    rows = []
    for u in np.linspace(0.0, 1.0, n):
        w = section(u, d['seat_w'], d['top_w'], flare, waist) * S.MM
        l_ = section(u, d['block_len'], d['top_len'], flare, waist) * S.MM
        z = (1.0 - u) * h_cm * 10.0                    # mm above the tread
        A = np.pi * w * l_ / 4.0
        Z = np.pi * w * l_ ** 2 / 32.0
        sig = (Fv / A + Fh * z / Z) * 1e3              # N/mm^2 -> kPa
        rows.append((float(u), float(w), float(l_), float(sig)))
    k = int(np.argmax([r[3] for r in rows]))
    return rows[k][3], rows[k][0], rows


def score(Pl, step, f, k, e, x, v, ref_sig=None):
    x = np.array(x, float)
    if v['width'] is not None:
        x[1] = v['width']
    t, d, o, extra = O.terms(Pl, step, f, k, e, x)
    sig, u_at, rows = neck(d, x[0], v['flare'], v['waist'], o['W'])
    terms = {n: float(O.WEIGHTS[n] * val) for n, val in t.items()}
    # (the leather: the body's volume against the block's, not the tread's area times the height)
    vol = float(np.mean([r[1] * r[2] for r in rows]))
    blk = d['seat_w'] * d['block_len'] * S.MM ** 2
    terms['leather'] = O.WEIGHTS['leather'] * vol / (0.85 * blk * 2.0) * (x[0] / 1.0)
    if ref_sig is not None:
        terms['neck'] = NECK_W * (sig / ref_sig) ** 2
    return dict(x={n: float(val) for n, val in zip(O.NAMES, x)}, flare=v['flare'], waist=v['waist'], note=v['note'],
                loss=float(sum(terms.values())), terms=terms, neck_kpa=sig, neck_at_u=u_at,
                sections=rows[::5], tip_deg=o['tip_deg'], peak_kpa=o['peak_all'], shares=o['share'],
                vamp_walk_N=o['vamp_N_walk'], tread_cm=(d['top_w'] * S.CM, d['top_len'] * S.CM),
                seat_cm=(d['seat_w'] * S.CM, d['block_len'] * S.CM), **extra)


def main():
    arg = lambda kk, dd=None: sys.argv[sys.argv.index(kk) + 1] if kk in sys.argv else dd  # noqa: E731
    D = np.load(sys.argv[1])
    f = S.measure(D['P'], D['N'], D['maskR'], D['ankleR'])
    Pl, step = S.plantar(D['P'], D['N'], D['maskR'])
    k, e, bare = S.calibrate(Pl, step, f)
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'blender', 'heelboot'))
    base = json.load(open(arg('--base'))) if arg('--base') else None
    x0 = [1.7675569, 1.0, 0.8102475, 0.0, 0.3905357, 1.0] if base is None else [base['optimum']['x'][n] for n in O.NAMES]
    out = {}
    ref = score(Pl, step, f, k, e, x0, VARIANTS['block'])
    for name, v in VARIANTS.items():
        out[name] = r = score(Pl, step, f, k, e, x0, v, ref_sig=ref['neck_kpa'])
        print('%-10s loss %.3f  neck %5.0f kPa at u %.2f (x%.2f the block)  tip %.1f deg  peak %.0f kPa  shares %s'
              % (name, r['loss'], r['neck_kpa'], r['neck_at_u'], r['neck_kpa'] / ref['neck_kpa'], r['tip_deg'],
                 r['peak_kpa'], ' '.join('%s %.0f' % (n, 100 * s) for n, s in r['shares'].items())))
        print('           ' + '  '.join('%s %.3f' % kv for kv in r['terms'].items()))
    # what the photo's height would cost (the reference boot: 3.9 cm at the back)
    for h in (2.5, 3.0, 3.9):
        x = list(x0)
        x[0] = h
        for name in VARIANTS:
            r = score(Pl, step, f, k, e, x, VARIANTS[name], ref_sig=ref['neck_kpa'])
            out['%s@%.1f' % (name, h)] = r
            print('%-10s at %.1f cm: loss %.3f  neck x%.2f  tip %.1f deg  shares %s'
                  % (name, h, r['loss'], r['neck_kpa'] / ref['neck_kpa'], r['tip_deg'],
                     ' '.join('%s %.0f' % (n, 100 * s) for n, s in r['shares'].items())))
    if arg('--out'):
        json.dump(out, open(arg('--out'), 'w'), indent=1, default=float)
    if arg('--plot'):
        plot(out, arg('--plot'))


def plot(out, dst):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(VARIANTS)
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 1, 1])
    for row, (sfx, title) in enumerate((('', 'at the optimizer\'s height'), ('@3.0', 'at 3.0 cm'))):
        ax = fig.add_subplot(gs[row, 0])
        for i, n in enumerate(names):            # side (the breast to the left) and back, at true size, mm
            key = n + sfx if n + sfx in out else None
            if key is None:
                v = VARIANTS[n]
                r = out[n]
                h = (3.0 if sfx else r['x']['h']) * 10.0
            else:
                r, v, h = out[key], VARIANTS[n], out[key]['x']['h'] * 10.0
            U = np.linspace(0, 1, 61)
            ls = np.array([section(u, r['seat_cm'][1] * 10, r['tread_cm'][1] * 10, v['flare'], v['waist']) for u in U])
            ws = np.array([section(u, r['seat_cm'][0] * 10, r['tread_cm'][0] * 10, v['flare'], v['waist']) for u in U])
            z = (1 - U) * h
            x0 = i * 200.0
            back = x0 + 80.0
            ax.fill_betweenx(z, back - ls, back, color='C%d' % i, alpha=0.75, lw=0)
            ax.fill_betweenx(z, x0 + 140 - ws / 2, x0 + 140 + ws / 2, color='C%d' % i, alpha=0.4, lw=0)
            ax.plot([x0, x0 + 185], [0, 0], color='k', lw=0.8)
            ax.text(x0 + 5, h + 3, n, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 200 * len(names))
        ax.set_ylim(-3, 45)
        ax.set_title('Side (breast left, dark) and back (light) of each heel, true size in mm, ' + title, fontsize=10)
    ax1 = fig.add_subplot(gs[:, 1])
    bottom = np.zeros(len(names))
    for t in out['block']['terms']:
        vals = np.array([out[n]['terms'].get(t, 0.0) for n in names])
        ax1.bar(names, vals, bottom=bottom, label=t)
        bottom += vals
    ax1.set_title('The gauge\'s loss by term, at the optimizer\'s height (lower is better)')
    ax1.legend(fontsize=7)
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.bar(names, [out[n]['neck_kpa'] for n in names], color=['C%d' % i for i in range(len(names))])
    ax2.set_title('Neck stress at heel strike, kPa (highest section)')
    fig.tight_layout()
    fig.savefig(dst, dpi=110)


if __name__ == '__main__':
    main()
