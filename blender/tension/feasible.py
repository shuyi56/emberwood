"""Stage 1 of the coupled solve: collect a sweep of lapel and collar builds, apply the limits, and map the feasible
region. Stage 2 (pick) chooses inside it the joins that carry the load most evenly with the smoothest turn.

    python3 blender/tension/feasible.py SWEEP_DIR OUT_DIR [--field preL.jsonl]

SWEEP_DIR/<lapel>/ holds a foldfront build (lapel.blend.*, fold.log) and SWEEP_DIR/<lapel>/c_<collar>/ the
notchcollar builds on it, each with collar.log, out.blend.collar.npz, gauge.json (the collar's flat cut) and
join.json (join.py). Lapel names are w<width*100>_p<point_t*100>; collar names a<angle>_g<notch_at>_p<point_len>.

The limits. No published strain limit for a fulled wool was found (Kawabata's KES-F gives the measures, EMT, G,
B, not tailoring thresholds; Leung et al. 2000 define formability as extensibility under low load times bending
rigidity, without limits in the abstract). So every limit is the approved build's own number: a design is feasible
when it is nowhere worse than what John accepted (notch_lapel with the notch collar, 2026-11-22) and has no flaw:

  lapel   foldfront's 95% edge strain under 15.0% on both fronts, the worst visible edge (her right front) under
          +39%, no torn faces (no dihedral over 120 degrees in the turned cloth), and the field's turned-lapel energy
          under the approved line's (0.0169)
  collar  its flat cut's 95% stretch under 2.5%, no sharp fold in the fall (neighbouring faces over 60 degrees apart,
          the lapel retrospective's count), the notch as laid within 2.5 degrees of the angle asked on both sides
  join    the seam closed (the collar's gorge within 8 mm of the lapel's top edge everywhere: a cloth's thickness
          and a half), the collar's tip fold under the approved build's +10%
"""
import sys, os, re, json, glob, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tension as TS  # noqa: E402
import gauge_fold  # noqa: E402

# builds seen torn in close-up renders, which no gauge here flags (render-queue rejections)
REJECTED = {'w42_p24_a02_f09': 'torn edge (rej. 119)', 'w42_p24_a04_f09': 'torn edge (rej. 119)'}
LIM = dict(lap95=15.0, lapworst=39.0, lapfold=120.0, field=0.0169, col95=2.5, sharp=60.0, notch=2.5, gap=8.0, tip=15.6)


def fold_log(path):
    s = open(path).read()
    r = {}
    for k in (1, 2):
        m = re.search(r'front %d: strain over \d+ edges: mean \|([\d.]+)%%\|, 95%% under \|([\d.]+)%%\|, worst ([-\d.]+)%% / \+([\d.]+)%%' % k, s)
        if m:
            r[k] = dict(mean=float(m.group(1)), p95=float(m.group(2)), worst_neg=float(m.group(3)), worst_pos=float(m.group(4)))
    return r


def notch(path):
    s = open(path).read()
    return [(float(a), float(b), float(c)) for a, b, c in re.findall(r'the notch opens ([\d.]+) degrees as cut, ([\d.]+) as laid \(asked ([\d.]+)\)', s)]


def sharp_folds(npz):
    C = np.load(npz); X, T = C['pts'], C['tri'].astype(np.int64)
    T = T[TS.areas(X, T) > 1e-9]
    n = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]); n /= np.linalg.norm(n, axis=1)[:, None]
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); key = np.sort(e, axis=1)
    tri = np.tile(np.arange(len(T)), 3); o = np.lexsort((key[:, 1], key[:, 0])); key, tri = key[o], tri[o]
    same = (key[1:] == key[:-1]).all(1)
    return int(((n[tri[:-1][same]] * n[tri[1:][same]]).sum(1) < 0.5).sum())


def lapel_torn(lapnpz):
    import join as J
    L = np.load(lapnpz); out = 0.0
    for k in (1, 2):
        P = L['pts%d' % k]; T = J.lapel_mesh(P, L['t%d' % k], L['d%d' % k], float(L['ln%d' % k]))
        T = T[TS.areas(P, T) > 1e-10]
        out = max(out, gauge_fold.max_fold(P, T))
    return out


def collect(sweep, field=None, stage2=False):
    F = {}
    if field and os.path.exists(field):
        for l in open(field):
            r = json.loads(l)
            if r['belly'] < 0.01:
                F[(round(r['width'], 2), round(r['point_t'], 2))] = r['lapel']
    rows = []
    for ld in sorted(glob.glob(os.path.join(sweep, '*/'))):
        name = os.path.basename(ld.rstrip('/'))
        m = re.match(r'w(\d+)_p(\d+)', name)
        if not m or not os.path.exists(os.path.join(ld, 'fold.log')):
            continue
        w, pt = int(m.group(1)) / 100, int(m.group(2)) / 100
        fl = fold_log(os.path.join(ld, 'fold.log'))
        if len(fl) < 2:
            continue
        torn = lapel_torn(os.path.join(ld, 'lapel.blend.lapels.npz'))
        for cd in sorted(glob.glob(os.path.join(ld, 'c_*/'))):
            cn = os.path.basename(cd.rstrip('/'))[2:]
            if ('_sr' in cn) != stage2:      # (stage 2's join-shape builds: see pick.py)
                continue
            mc = re.match(r'a(\d+)_g([\d.]+)_p([\d.]+)', cn)
            try:
                g = json.load(open(os.path.join(cd, 'gauge.json')))
                j = json.load(open(os.path.join(cd, 'join.json')))
                nt = notch(os.path.join(cd, 'collar.log'))
                sh = sharp_folds(os.path.join(cd, 'out.blend.collar.npz'))
            except Exception:
                continue
            if len(nt) < 2 or '1' not in j or '2' not in j or 'gap_max_mm' not in j['1']:
                continue
            m2 = re.search(r'_sr([\d.]+)_rr([\d.]+)_et([\d.]+)', cn)
            r = dict(lapel=name, width=w, point_t=pt, collar=cn, angle=float(mc.group(1)), notch_at=float(mc.group(2)),
                     point_len=float(mc.group(3)), lap95=max(fl[1]['p95'], fl[2]['p95']), lapmean=(fl[1]['mean'] + fl[2]['mean']) / 2,
                     lapworst=fl[1]['worst_pos'], lapfold=torn, field=F.get((w, pt)), col95=100 * (g['s1_95'] - 1),
                     sharp=sh, notch_err=max(abs(b - c) for a, b, c in nt), notch_laid=[b for a, b, c in nt],
                     gap=max(j['1']['gap_max_mm'], j['2']['gap_max_mm']), tip=max(j['1']['tip_fold'], j['2']['tip_fold']),
                     turn_max=max(j['1']['turn_max'], j['2']['turn_max']), turn_mean=(j['1']['turn_mean'] + j['2']['turn_mean']) / 2,
                     kink=max(j['1']['kink_max'], j['2']['kink_max']),
                     cv=(j['1']['evenness_cv'] + j['2']['evenness_cv']) / 2,
                     turn_p90=max(j['1']['turn_p90'], j['2']['turn_p90']),
                     peak=max(j['1']['peak_lapel'], j['2']['peak_lapel']),
                     ratio=(j['1']['load_ratio'] + j['2']['load_ratio']) / 2,
                     lap_load=(j['1']['lapel_load'] + j['2']['lapel_load']) / 2, col_load=(j['1']['collar_load'] + j['2']['collar_load']) / 2,
                     sym_tip=abs(j['1']['tip_fold'] - j['2']['tip_fold']))
            mt = re.search(r'_t(\d+)', cn); r['tip_round'] = int(mt.group(1)) / 100 if mt else 0.15
            ml = re.search(r'_a(\d+)_f(\d+)', name)
            r['gorge_from'], r['fillet'] = (int(ml.group(1)) / 100, int(ml.group(2)) / 100) if ml else (0.04, 0.05)
            if m2:
                r.update(stand_run=float(m2.group(1)), roll_r=float(m2.group(2)), end_taper=float(m2.group(3)))
            fails = []
            if r['lap95'] > LIM['lap95']: fails.append('lapel 95%')
            if r['lapworst'] > LIM['lapworst']: fails.append('lapel worst')
            # (lapfold is reported, not applied: on the lapel's chart triangulation the approved build itself reads
            # 180 degrees, so it cannot tell a tear from a sliver; the worst-edge limit is what caught the swell tear)
            if r['field'] is not None and r['field'] > LIM['field']: fails.append('field')
            if r['col95'] > LIM['col95']: fails.append('collar 95%')
            if r['sharp'] > 0: fails.append('sharp folds %d' % r['sharp'])
            if r['notch_err'] > LIM['notch']: fails.append('notch %.1f off' % r['notch_err'])
            if r['gap'] > LIM['gap']: fails.append('seam open %.1f mm' % r['gap'])
            if r['tip'] > LIM['tip']: fails.append('tip fold %.1f' % r['tip'])
            if name in REJECTED: fails.append(REJECTED[name])
            r['fails'] = fails; r['feasible'] = not fails
            rows.append(r)
    return rows


# the approved build's own join (notch_lapel + notch collar, 2026-11-22), the yardstick for each term
BASE = dict(cv=0.52, turn_p90=53.3, tip=13.6)


def objective(r):
    """Stage 2: the even, smooth join. Three terms, each over the approved build's own value, so the approved build
    scores 3.0: the load's unevenness along the seam (cv, each side's eight stretches), the turn across the seam at
    its 90th percentile (the sharpest turn without one stray point), and the collar-tip fold. The lapel-to-collar load
    ratio is reported, not scored: lowering it by loading the collar more would add tension, not spread it."""
    return r['cv'] / BASE['cv'] + r['turn_p90'] / BASE['turn_p90'] + r['tip'] / BASE['tip']


def maps(rows, path):
    """One sheet: top, the lapel plane (width x point), each lapel's best feasible join and its edge strain; bottom, the
    full collar grid on the derived lapel, one panel per collar point length. Grey: infeasible (the reason printed)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    nrm = Normalize(2.8, 3.6); cm = plt.get_cmap('viridis_r')
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    rows = [r for r in rows if re.fullmatch(r'w\d+_p\d+', r['lapel'])]      # (the lapel plane: round 2's gorge variants are in sweep.json)
    laps = sorted({(r['width'], r['point_t']) for r in rows})
    W = sorted({w for w, _ in laps}); PT = sorted({p for _, p in laps})
    for k, (key, ttl) in enumerate((('objective', 'best feasible join on each lapel (objective, approved = 3.0)'),
                                    ('lap95', 'lapel edge strain, 95th percentile % (limit 15.0)'))):
        a = ax[0, k]
        for (w, pt) in laps:
            R = [r for r in rows if r['width'] == w and r['point_t'] == pt]
            F = [r for r in R if r['feasible']]
            x, y = W.index(w), PT.index(pt)
            if key == 'objective':
                if F:
                    b_ = min(F, key=lambda r: r['objective'])
                    a.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color=cm(nrm(b_['objective']))))
                    a.text(x, y, '%.2f\n%s\n%d of %d feasible' % (b_['objective'], b_['collar'], len(F), len(R)), ha='center', va='center', fontsize=8)
                else:
                    a.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color='0.8')); a.text(x, y, 'none feasible', ha='center', va='center', fontsize=8)
            else:
                v = R[0]['lap95']
                a.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color=plt.get_cmap('magma_r')((v - 9) / 7)))
                a.text(x, y, '%.1f%%\nworst +%.1f%%' % (v, R[0]['lapworst']), ha='center', va='center', fontsize=8,
                       color='w' if v > 12 else 'k')
        a.set_xlim(-.5, len(W) - .5); a.set_ylim(-.5, len(PT) - .5)
        a.set_xticks(range(len(W))); a.set_xticklabels(['%.2f' % w for w in W]); a.set_yticks(range(len(PT))); a.set_yticklabels(['%.2f' % p for p in PT])
        a.set_xlabel('lapel width (outline peak)'); a.set_ylabel('lapel point (share of roll line)'); a.set_title(ttl, fontsize=9)
    a = ax[0, 2]
    for r in rows:
        a.scatter(r['turn_p90'], r['cv'], s=18, c=[cm(nrm(r['objective']))] if r['feasible'] else ['0.75'], edgecolors='k' if r['feasible'] else 'none', linewidths=.4)
    a.scatter([BASE['turn_p90']], [BASE['cv']], marker='*', s=160, c='crimson', label='approved build')
    a.set_xlabel('turn across the seam, 90th percentile (deg)'); a.set_ylabel('load unevenness along the seam (cv)')
    a.set_title('every build: grey infeasible, colour = objective', fontsize=9); a.legend(fontsize=8)
    dl = max(laps, key=lambda l: sum(1 for r in rows if (r['width'], r['point_t']) == l))
    for jx, pl in enumerate(sorted({r['point_len'] for r in rows if (r['width'], r['point_t']) == dl})[:3]):
        a = ax[1, jx]
        R = [r for r in rows if (r['width'], r['point_t']) == dl and r['point_len'] == pl]
        A = sorted({r['angle'] for r in R}); G = sorted({r['notch_at'] for r in R})
        for r in R:
            x, y = A.index(r['angle']), G.index(r['notch_at'])
            a.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color=cm(nrm(r['objective'])) if r['feasible'] else '0.85'))
            a.text(x, y, '%.2f' % r['objective'] + ('' if r['feasible'] else '\n' + '\n'.join(r['fails'])), ha='center', va='center',
                   fontsize=7.5, color='k' if r['feasible'] else '0.35')
        a.set_xlim(-.5, len(A) - .5); a.set_ylim(-.5, len(G) - .5)
        a.set_xticks(range(len(A))); a.set_xticklabels(['%g' % x for x in A]); a.set_yticks(range(len(G))); a.set_yticklabels(['%g' % x for x in G])
        a.set_xlabel('notch angle (deg)'); a.set_ylabel('notch position (share of lapel top edge)')
        a.set_title('collar grid on lapel w %.2f p %.2f, collar point %.2f' % (dl[0], dl[1], pl), fontsize=9)
    plt.subplots_adjust(hspace=0.38, wspace=0.32, right=0.86)
    sm = plt.cm.ScalarMappable(norm=nrm, cmap=cm); fig.colorbar(sm, ax=ax[:, :], shrink=0.5, label='join objective (lower = smoother, more even)')
    fig.suptitle('Stage 1: the feasible region of the coupled lapel-collar sweep')
    plt.savefig(path, dpi=70); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('sweep'); ap.add_argument('out'); ap.add_argument('--field')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = collect(a.sweep, a.field)
    for r in rows:
        r['objective'] = float(objective(r))
    json.dump(rows, open(os.path.join(a.out, 'sweep.json'), 'w'), indent=1)
    feas = [r for r in rows if r['feasible']]
    print('feasible: %d of %d' % (len(feas), len(rows)))
    for r in sorted(feas, key=lambda r: r['objective'])[:10]:
        print('  %s %s objective %.3f cv %.2f turn %.0f tip %.1f ratio %.1f lap95 %.1f col95 %.2f notch %s'
              % (r['lapel'], r['collar'], r['objective'], r['cv'], r['turn_max'], r['tip'], r['ratio'], r['lap95'], r['col95'], r['notch_laid']))
    maps(rows, os.path.join(a.out, 'feasible.png'))
