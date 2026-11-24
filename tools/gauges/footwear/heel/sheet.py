"""The heel's support and tension sheet: one page that says how a foot is carried on a heel, and where the shoe is
loaded. Runs outside Blender (numpy and matplotlib) on the feet a build saved (heel.py with HEEL_DUMP=PATH), and
on the built parts if they were saved too (fig.py style parts.npz: every 'Heel ...' mesh and the posed 'Skin body').

    python3 tools/heel/sheet.py DUMP.npz OUT.png [--parts PARTS.npz] [--h CM] [--arch 1] [--cup 1]

Panels: the side and back sections of the foot on its heel (the parts in their design colours); the pressure under
the foot in plan with the heel's tread and the centres of pressure; the load shares against Cavanagh 1987
(the validation: the gauge is calibrated on the heel's share and peak only, so the midfoot, forefoot and toes and the
forefoot's peak are the test); the sweep over heel height (shares, peak pressure, rear load against the tread's
middle, tip angle, vamp load, shank safety); the waist's bending moment and sag; and the numbers, each marked
SOURCED or EXPERIMENTAL. Writes OUT.png and OUT.json."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # study.py, the support gauge, beside this file
import study as S  # noqa: E402

COL = {'insole': '#33a0a0', 'shank': '#d92626', 'sole': '#f28c1a', 'lift_a': '#8c6bbf', 'lift_b': '#598cd9',
       'wedge': '#e6cc33', 'top_lift': '#33b34d', 'Skin_body': '#9a9590'}


def arg(k, d=None):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def main():
    src, dst = sys.argv[1], sys.argv[2]
    D = np.load(src)
    P, N = D['P'], D['N']
    mask, ankle = D['maskR'], D['ankleR']
    f = S.measure(P, N, mask, ankle)
    Pl, step = S.plantar(P, N, mask)
    k, e, bare = S.calibrate(Pl, step, f)
    af, cup = float(arg('--arch', 1)), float(arg('--cup', 1))
    h = float(arg('--h')) if arg('--h') else S.best_height(Pl, step, f, k, e, arch_fill=af, cup=cup)
    d = S.derive(f, h, arch_fill=af, cup=cup)
    o = S.support(Pl, step, f, d, k, e)
    dflat = S.derive(f, h, arch_fill=0, cup=0)
    of = S.support(Pl, step, f, dflat, k, e)
    # the boots as they are (blender/boot: no heel, a flat sole): the foot flat on a flat bed
    boot = S.support(Pl, step, f, S.derive(f, 0.0, arch_fill=0, cup=0), k, e, th=0.0)
    hs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
    sweep = {'full': [], 'flat': []}
    for hh in hs:
        for key, (a_, c_) in (('full', (af, cup)), ('flat', (0, 0))):
            dd = S.derive(f, hh, arch_fill=a_, cup=c_)
            sweep[key].append(S.support(Pl, step, f, dd, k, e, th=0.0 if hh == 0 else None))

    fig = plt.figure(figsize=(20, 13), dpi=100)
    fig.suptitle('Heel derived from the foot: support and tension gauge (right foot; heel %.1f cm at the seat, %.1f cm '
                 'at the breast)' % (h, o['h_breast_cm']), x=0.01, ha='left', fontsize=15)
    gs = fig.add_gridspec(3, 4, left=0.04, right=0.99, top=0.93, bottom=0.04, wspace=0.28, hspace=0.32)
    cm = S.CM

    parts = None
    if arg('--parts') and os.path.exists(arg('--parts')):
        parts = np.load(arg('--parts'))
    xc = float(Pl[:, 0].mean())

    # 1 the side section, through the foot's middle
    ax = fig.add_subplot(gs[0, 0:2])
    ax.set_title('Side section through the middle of the foot (cm); weight line, ankle, rear centre of pressure')
    if parts is not None:
        for kname in parts.files:
            if not kname.endswith('.R') and kname != 'Skin_body':
                continue
            V = parts[kname]
            base = kname.split('.')[0].replace('Heel_', '')
            base = '_'.join(w for w in base.split('_') if not w.isdigit())
            m = np.abs(V[:, 0] - xc) < (0.12 if kname != 'Skin_body' else 0.08)
            if kname == 'Skin_body':
                m &= (V[:, 2] < f['ankle'][2] + 3) & (V[:, 0] < 0)
            ax.scatter(-V[m, 1] * cm, V[m, 2] * cm, s=2, c=COL.get(base, '#555'), label=base if kname != 'Skin_body' else 'skin')
    else:
        ax.scatter(-o['Pf'][:, 1] * cm, o['Pf'][:, 2] * cm, s=2, c='#999')
    ax.axhline(f['floor'] * cm, color='k', lw=0.8)
    ax.axvline(-o['y_w'] * cm, color='#c00', ls='--', lw=1, label='weight line')
    ax.axvline(-o['rear_cop'][1] * cm, color='#060', ls=':', lw=1.4, label='rear centre of pressure')
    ax.axvline(-d['top_y'] * cm, color='#33b34d', lw=0.8, label='tread middle (Golding seat point)')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left', ncol=3, markerscale=4)
    ax.set_xlabel('forward (cm)')

    # 2 the back section, across the seat point
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title('Section across at the seat point (cm), from behind')
    if parts is not None:
        for kname in parts.files:
            if not kname.endswith('.R') and kname != 'Skin_body':
                continue
            V = parts[kname]
            base = '_'.join(w for w in kname.split('.')[0].replace('Heel_', '').split('_') if not w.isdigit())
            m = np.abs(V[:, 1] - d['top_y']) < 0.12
            if kname == 'Skin_body':
                m &= (V[:, 2] < 4.0) & (V[:, 0] < 0)
            ax.scatter(-V[m, 0] * cm, V[m, 2] * cm, s=3, c=COL.get(base, '#555'))
    ax.axhline(f['floor'] * cm, color='k', lw=0.8)
    ax.set_aspect('equal')

    # 3 pressure in plan
    ax = fig.add_subplot(gs[0, 3])
    ax.set_title('Pressure under the foot on the heel (kPa, standing)')
    sc = ax.scatter(-Pl[:, 0] * cm, -Pl[:, 1] * cm, c=o['p'], s=9, cmap='magma_r', vmin=0, vmax=max(60, o['p'].max()))
    plt.colorbar(sc, ax=ax, fraction=0.04)
    for a_, b_, nm in S.REGIONS[1:]:
        ax.axhline(-(f['heel_y'] - a_ * f['L']) * cm, color='#888', lw=0.5)
    ty = np.array([d['breast_y'], d['top_y'] + 0.5 * d['top_len']])
    cxr = o['rear_cop'][0]
    ax.add_patch(plt.Rectangle((-(cxr + 0.5 * d['top_w']) * cm, -ty[1] * cm), d['top_w'] * cm, d['top_len'] * cm,
                               fill=False, ec='#33b34d', lw=1.5, label='heel tread'))
    ax.plot(-o['rear_cop'][0] * cm, -o['rear_cop'][1] * cm, 'g+', ms=14, mew=2, label='rear centre of pressure')
    ax.plot(-o['x_w'] * cm, -o['y_w'] * cm, 'rx', ms=10, mew=2, label='weight line')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xticks([])

    # 4 shares against Cavanagh
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title('Load shares vs Cavanagh 1987 (SOURCED)')
    names = ['heel', 'midfoot', 'forefoot', 'toes']
    cols = [('Cavanagh barefoot', [S.CAVANAGH[n] for n in names], '#444'),
            ('gauge barefoot', [bare['share'][n] for n in names], '#999'),
            ('boot as built (flat)', [boot['share'][n] for n in names], '#c8a27a'),
            ('derived heel, flat bed', [of['share'][n] for n in names], '#f28c1a'),
            ('derived heel, fitted bed', [o['share'][n] for n in names], '#33a0a0')]
    w = 0.16
    for i, (lab, v, c) in enumerate(cols):
        ax.bar(np.arange(4) + (i - 2) * w, np.array(v) * 100, w, label=lab, color=c)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylabel('% of the load on one foot')
    ax.legend(fontsize=7)

    # 5 sweep: shares and peak
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title('Shares over heel height (cm at seat)\nfitted bed solid, flat dashed')
    for n_, c in zip(names, ('#7a3', '#39c', '#c63', '#888')):
        ax.plot(hs, [r['share'][n_] * 100 for r in sweep['full']], '-', color=c, label=n_)
        ax.plot(hs, [r['share'][n_] * 100 for r in sweep['flat']], '--', color=c)
    ax.axvline(h, color='k', lw=0.8)
    off = o['h_breast_cm'] - h                      # (the breast stands this much over the seat's rise)
    ax.axvspan(2.54 - off, 3.175 - off, color='#ddd', alpha=0.6, label='Dooley 1-1.25 in (at the breast)')
    ax.axvline(3.9 - off, color='#c00', lw=0.8, ls=':', label='Chen 2020: control worse from 3.9 cm')
    ax.legend(fontsize=7)
    ax.set_ylabel('%')
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.set_title('Peak kPa, and rear load off\nthe tread middle (mm, green)')
    ax2.plot(hs, [r['peak_all'] for r in sweep['full']], '-', color='#33a0a0', label='peak, fitted bed')
    ax2.plot(hs, [r['peak_all'] for r in sweep['flat']], '--', color='#f28c1a', label='peak, flat bed')
    ax2.axhline(S.CAVANAGH['heel_peak'], color='#444', lw=0.6, ls=':', label='barefoot heel peak (Cavanagh)')
    ax3 = ax2.twinx()
    ax3.plot(hs[1:], [r['rear_cop_off_mm'] for r in sweep['full'][1:]], '-', color='#060', label='rear COP off (fitted)')
    ax3.axhline(0, color='#060', lw=0.5)
    ax2.axvline(h, color='k', lw=0.8)
    ax2.legend(fontsize=7, loc='upper left')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.set_ylabel('mm behind (+) the tread middle')

    # 6 sweep: stability and tension
    ax = fig.add_subplot(gs[1, 3])
    ax.set_title('Tip angle (deg), vamp load walking (N),\nshank safety (x10), over heel height')
    ax.plot(hs[1:], [r['tip_deg'] for r in sweep['full'][1:]], '-', color='#7a3', label='tip angle')
    ax.axhline(S.EXP['tip_min'], color='#7a3', ls=':', lw=0.8, label='tip floor %d deg (EXPERIMENTAL)' % S.EXP['tip_min'])
    ax.plot(hs[1:], [r['vamp_N_walk'] for r in sweep['full'][1:]], '-', color='#c33', label='vamp / lacing (N)')
    ax.plot(hs[1:], [10 * r['shank_SF_walk'] for r in sweep['full'][1:]], '-', color='#d92626', alpha=0.5,
            label='shank safety x10')
    ax.axvline(h, color='k', lw=0.8)
    ax.legend(fontsize=7)

    # 7 the waist
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title('Waist: moment N mm, sag mm (standing)')
    ax.plot(o['shank_x'], o['shank_M'], '-', color='#d92626', label='moment (fitted)')
    ax.plot(of['shank_x'], of['shank_M'], '--', color='#f28c1a', label='moment (flat)')
    ax4 = ax.twinx()
    ax4.plot(o['shank_x'], o['shank_defl'], '-', color='#333', label='sag (fitted)')
    ax.set_xlabel('mm forward of the heel breast')
    ax.legend(fontsize=7, loc='upper right')
    ax4.legend(fontsize=7, loc='lower right')

    # 8 the heel's sections at several heights (plan): each lift's outline, the seat's over the tread's
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('Heel: sections every lift (plan, cm)')
    if parts is not None:
        lifts = sorted((kk for kk in parts.files if kk.endswith('.R') and any(t in kk for t in ('lift', 'wedge'))),
                       key=lambda kk: parts[kk][:, 2].mean())
        for i, kk in enumerate(lifts):
            V = parts[kk]
            V = V[V[:, 2] > V[:, 2].min() + 0.01]                       # (the lift's upper face)
            ang = np.arctan2(V[:, 1] - V[:, 1].mean(), V[:, 0] - V[:, 0].mean())
            q = []
            for a0 in np.linspace(-np.pi, np.pi, 73)[:-1]:
                m = np.abs(np.angle(np.exp(1j * (ang - a0)))) < 0.09
                if m.any():
                    r_ = np.hypot(V[m, 0] - V[:, 0].mean(), V[m, 1] - V[:, 1].mean())
                    q.append(V[m][np.argmax(r_), :2])
            q = np.array(q + q[:1])
            base = '_'.join(w for w in kk.split('.')[0].replace('Heel_', '').split('_') if not w.isdigit())
            ax.plot(-q[:, 0] * cm, -q[:, 1] * cm, '-', color=COL.get(base, '#555'), lw=1.2)
    ax.plot(-Pl[:, 0] * cm, -Pl[:, 1] * cm, '.', ms=1, color='#ccc')
    ax.set_ylim(-(f['heel_y'] + 0.3) * cm, -(d['breast_y'] - 1.0) * cm)
    ax.set_aspect('equal')

    # 9 the numbers
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    s = o['share']
    lines = [
        'FOOT (measured on the mannequin): length %.1f cm, ball %.1f cm, heel at the seat point %.1f cm, ankle %.0f%% of '
        'the length from the heel, metatarsal line %.0f deg off square (Raychoudhury 2014: 17)' % (
            f['L'] * cm, f['ball_w'] * cm, f['heel_w'] * cm, 100 * f['ankle_share'], f['mtp_deg_measured']),
        'VALIDATION (barefoot, Cavanagh 1987; fitted: heel share 60%% and heel peak 139 kPa): midfoot %.0f%% (8), '
        'forefoot %.0f%% (28), toes %.0f%% (about 4), forefoot peak %.0f kPa (53)' % (
            100 * bare['share']['midfoot'], 100 * bare['share']['forefoot'], 100 * bare['share']['toes'],
            bare['peak']['forefoot']),
        '   trend checks: heel share falls and forefoot rises with height (Broch 2004) %s; midfoot rises from flat to '
        '2-4 cm (Wibowo 2017) %s' % (
            'PASS' if sweep['flat'][-1]['share']['heel'] < sweep['flat'][1]['share']['heel'] and
            sweep['flat'][-1]['share']['forefoot'] + sweep['flat'][-1]['share']['toes'] >
            sweep['flat'][1]['share']['forefoot'] + sweep['flat'][1]['share']['toes'] else 'FAIL',
            'PASS' if sweep['flat'][5]['share']['midfoot'] > sweep['flat'][0]['share']['midfoot'] else 'FAIL'),
        'HEEL (derived): %.1f cm at the seat, %.1f cm at the breast (AmeFird measure; Dooley 2.5-3.2; Chen <3.9), '
        'pitch %.0f deg, %d lifts of %.0f mm (atelier rule)' % (h, o['h_breast_cm'], o['theta_deg'], d['lifts'], S.LIFT_MM),
        '   seat %.1f cm = 3/4 of the ball (Golding), room over the heel %.1f mm; tread %.1f x %.1f cm centred on the seat '
        'point (Golding), breast %.1f cm from the back (1/4 of the last, One & Awl)' % (
            d['seat_w'] * cm, o['seat_fit_mm'], d['top_w'] * cm, d['top_len'] * cm, (f['heel_y'] - d['breast_y']) * cm),
        'SUPPORT: loads heel %.0f%% midfoot %.0f%% forefoot %.0f%% toes %.0f%%; peak %.0f kPa (flat bed %.0f, boot as '
        'built %.0f); contact %.0f cm2 (flat %.0f)' % (
            100 * s['heel'], 100 * s['midfoot'], 100 * s['forefoot'], 100 * s['toes'], o['peak_all'], of['peak_all'],
            boot['peak_all'], o['contact_cm2'], of['contact_cm2']),
        '   rear load %.1f mm off the tread middle; tips at %.0f deg (floor %d, EXPERIMENTAL)' % (
            o['rear_cop_off_mm'], o['tip_deg'], S.EXP['tip_min']),
        'TENSION: forward push %.0f N standing; friction %.2f (EXPERIMENTAL) holds the rest; vamp/lacing %.0f N walking '
        '(x%.1f, Perry), %.2f N/mm along the throat' % (
            o['forward_N'], S.EXP['friction'], o['vamp_N_walk'], 2 * S.GAIT_PEAK, o['vamp_tension_N_mm']),
        '   waist moment %.0f N mm walking; steel shank %gx%g mm (EXPERIMENTAL) safety %.1f, sag %.2f mm; to sag <= %.1f '
        'mm it needs S = %.0f kN mm2 (ISO 18896 measure; has %.0f)' % (
            o['shank_M_walk'], S.EXP['shank_w'], S.EXP['shank_t'], o['shank_SF_walk'], o['shank_sag_walk'],
            S.EXP['waist_sag'], o['shank_S_need'], o['shank_S']),
    ]
    ax.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=8.6, family='monospace', wrap=True)
    fig.savefig(dst)

    def clean(r):
        return {kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in r.items()
                if isinstance(v, (float, int, str, bool, dict, tuple, np.floating))}
    nums = dict(h_seat_cm=h, k=k, e_cm=e * cm, foot={kk: (v.tolist() if hasattr(v, 'tolist') else v) for kk, v in f.items()
                                                    if kk != 'profile'},
                design={kk: float(v) for kk, v in d.items()}, fitted=clean(o), flat=clean(of), boot=clean(boot),
                barefoot=clean(bare), sweep={kk: [clean(r) for r in v] for kk, v in sweep.items()}, heights=hs.tolist(),
                experimental=S.EXP)
    with open(os.path.splitext(dst)[0] + '.json', 'w') as fh:
        json.dump(nums, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print('wrote', dst)
    print(S.summary(f, d, o))


if __name__ == '__main__':
    main()
