"""Read the bare bodice's tension field and derive the lapel from it.

    python3 blender/tension/read_bodice.py CUT.blend BARE.npz          # (Blender: the bare cut, no lapels, no collar)
    python3 blender/tension/derive.py BARE.npz OUTDIR [--fronts 1,2] [--quick]

Steps, each written to OUTDIR:
  1. each front flattened (tension.bare_front): the best flat cut with the darts slit, and its field:
     field.png (tension, compression with their directions, Gaussian curvature), field.json (percentiles).
  2. the roll line searched: the top on the neck (neck_x), the break (foot over the bust line) and the swing off the
     chord (belly), the lapel's area held at the current design's, scored by the strain the turned lapel must take
     lying on its front (Front.evaluate). search_roll.jsonl, search_roll.png.
  3. the outline searched on the chosen line: where the point sits down the line (point_t) and the long edge's
     swell (the S-curve: negative hollows it), the area held. search_outline.jsonl.
  4. the choice, within the design's own limits (below), written as foldfront.py options: derived.json.

The limits are the design, not the field's: the field alone always prefers a smaller lapel, set higher. So:
  - the lapel's area is held at the approved design's (notch_lapel), so only its shape moves;
  - the roll line's top stays within the tailoring rule's window: the roll line meets the neck where the collar's
    stand carries it on, about one stand height (1.0 cm, 0.38) out from the side neck point (the neck's half width
    w = 1.96), so neck_x 2.1 to 2.3. The field prefers it lower (1.5 to 1.8), but built (foldfront + notchcollar)
    a top at 1.9 or 1.6 gained no lapel strain at all (3.1% / 2.9% mean against 3.1%) and pulled the notch off
    true (37.5 and 43 degrees laid for 40);
  - the point stays within the reference's band: outline point_t 0.27 traced off the photo, built at 0.30 to 0.31.
    0.35 measured 0.38 built, with the reach down to 0.56 of the half shoulder (photo 0.62): a different lapel, so the
    window is 0.25 to 0.30.
  - the break stays where John approved it (1.35 under the bust line): the field is nearly flat in it, and its slight
    preference for a lower break comes only with a narrower lapel at the same area (width down 12 to 14%);
  - within the evaluator's noise (NOISE of the best), the design nearest the approved one is taken: the straightest
    line, the top nearest 2.2, the point nearest 0.27, the swell nearest 0.025;
  - the long edge's swell stays at or under the approved 0.025: at 0.05 foldfront's edge mesher tore a patch of
    inverted faces at the lapel's edge by the point (logged rejection, build E), for a gain within noise.
"""
import sys, os, json, itertools, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tension as TS  # noqa: E402

CURRENT = dict(neck_x=2.2, foot_over_bust=-1.35, belly=0.25, outline=[[0.04, 0.0], [0.27, 0.40], [1.0, 0.0]],
               outline_round=[0.05, 0.025])                 # foldfront's notch_lapel, as approved 2026-11-22
LIMITS = dict(neck_x=(2.1, 2.3), foot_over_bust=(-1.35, -1.35), point_t=(0.25, 0.30), swell=(-0.025, 0.025))
NOISE = 0.05      # the evaluator's scatter between neighbouring designs (the triangles the line crosses): about 5 to 10%


def field_png(fronts, path, lines=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(HERE, '..', 'zipboot'))
    import gauge as G
    fig, ax = plt.subplots(len(fronts), 3, figsize=(16, 7.2 * len(fronts)), squeeze=False)
    for r, (k, F) in enumerate(fronts.items()):
        s1, s2, d1, d2 = TS.principal(F.X, F.T, F.U)
        c = F.cen; sel = np.arange(0, len(F.T), 45)
        K, _ = G.gauss(F.X, F.T)
        for j, (q, nm, cm, dd, vm) in enumerate(((s1 - 1, 'tension: s1 - 1 (pulled longer than cut)', 'Reds', d1, 0.04),
                                                 (1 - s2, 'compression: 1 - s2 (more cloth than room)', 'Blues', d2, 0.04),
                                                 (None, 'Gaussian curvature K (1/unit^2)', 'PuOr', None, 0.08))):
            a = ax[r, j]
            if q is None:
                sc = a.tripcolor(F.X[:, 0], F.X[:, 2], F.T, K, cmap=cm, vmin=-vm, vmax=vm, shading='gouraud')
            else:
                sc = a.tripcolor(F.X[:, 0], F.X[:, 2], F.T, facecolors=q, cmap=cm, vmin=0, vmax=vm)
                a.quiver(c[sel, 0], c[sel, 2], dd[sel, 0], dd[sel, 2], angles='xy', headwidth=0, headlength=0,
                         headaxislength=0, pivot='mid', scale=45, width=0.0018, color='0.25')
            for nm_, (C, st) in (lines or {}).get(k, {}).items():
                a.plot(C[:, 0], C[:, 1], st, lw=1.6, label=nm_)
            a.set_aspect('equal'); a.set_title('front %d: %s' % (k, nm), fontsize=9); plt.colorbar(sc, ax=a, shrink=0.5)
            if lines and j == 0:
                a.legend(fontsize=8, loc='lower left')
    plt.tight_layout(); plt.savefig(path, dpi=60); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bare'); ap.add_argument('out')
    ap.add_argument('--fronts', default='1,2'); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--reuse', action='store_true', help='read the searches already in OUTDIR, only choose and draw')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = np.load(a.bare); M = eval(str(D['M'][0])); th = float(D['thick'])
    X = D['X']; T = TS.triangles(D['F']); fl = D['front_layer']
    fronts = {}
    stats = {}
    for k in [int(x) for x in a.fronts.split(',')]:
        F = TS.bare_front(X, T, fl, M, th, k)
        s1, s2, _, _ = TS.principal(F.X, F.T, F.U)
        stats[k] = dict(points=len(F.X), energy=TS.energy(F.X, F.T, F.U),
                        s1=np.percentile(s1, [50, 95, 99]).round(4).tolist(), s2=np.percentile(s2, [1, 5, 50]).round(4).tolist())
        print('derive: front %d flattened: %d points, tension 95%% %.1f%%, compression 5%% %.1f%%'
              % (k, len(F.X), 100 * (stats[k]['s1'][1] - 1), 100 * (1 - stats[k]['s2'][1])), flush=True)
        fronts[k] = F
    json.dump(stats, open(os.path.join(a.out, 'field.json'), 'w'), indent=1)
    zone = lambda c: c[:, 2] > M['z_bust'] - 2.0
    # the current design, as the yardstick
    base = {}
    for k, F in fronts.items():
        base[k] = F.evaluate(neck_x=CURRENT['neck_x'], foot=M['z_bust'] + CURRENT['foot_over_bust'], belly=CURRENT['belly'],
                             outline=CURRENT['outline'], sw=CURRENT['outline_round'][1], zone=zone, iters=30)
    area = float(np.mean([b['cut_area'] for b in base.values()]))
    # 2. the roll line
    NX = [1.6, 1.8, 2.0, 2.2, 2.4] if a.quick else [1.4, 1.6, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4]
    FO = [-1.35] if a.quick else [-2.0, -1.35, -0.7]
    BE = [-0.1, 0.0, 0.25] if a.quick else [-0.1, -0.05, 0.0, 0.05, 0.1, 0.25]
    rows = []
    if a.reuse:
        rows = [json.loads(l) for l in open(os.path.join(a.out, 'search_roll.jsonl'))]; NX = FO = BE = []
    with open(os.path.join(a.out, 'search_roll.jsonl'), 'a' if a.reuse else 'w') as fo:
        for nx, fob, be in itertools.product(NX, FO, BE):
            r = {}
            for k, F in fronts.items():
                e = F.evaluate(neck_x=nx, foot=M['z_bust'] + fob, belly=be if be else 0.001, outline=CURRENT['outline'],
                               sw=CURRENT['outline_round'][1], zone=zone, iters=30, area=area)
                r[k] = {q: e[q] for q in ('lapel', 'body', 'score', 'lap_s1_95', 'lap_s2_5', 'lost', 'wscale', 'roll_stretch') if q in e}
            row = dict(neck_x=nx, foot_over_bust=fob, belly=be, fronts=r,
                       lapel=float(np.mean([v['lapel'] for v in r.values()])))
            rows.append(row); fo.write(json.dumps(row) + '\n'); fo.flush()
            print('derive: roll line neck_x %.2f break %+.2f belly %+.2f: turned lapel energy %.5f' % (nx, fob, be, row['lapel']), flush=True)
    ok = [r for r in rows if LIMITS['neck_x'][0] - 1e-9 <= r['neck_x'] <= LIMITS['neck_x'][1] + 1e-9]
    ok = [r for r in ok if LIMITS['foot_over_bust'][0] - 1e-9 <= r['foot_over_bust'] <= LIMITS['foot_over_bust'][1] + 1e-9]
    lo = min(r['lapel'] for r in ok)
    best = min([r for r in ok if r['lapel'] <= (1 + NOISE) * lo],
               key=lambda r: (abs(r['belly']), abs(r['neck_x'] - CURRENT['neck_x'])))
    # 3. the outline on the chosen line
    PT = [0.21, 0.27, 0.35, 0.45] if a.quick else [0.15, 0.21, 0.25, 0.27, 0.30, 0.35, 0.45]
    SW = [-0.025, 0.0, 0.025, 0.05] if a.quick else [-0.025, 0.0, 0.025, 0.05, 0.075]
    orows = []
    if a.reuse:
        orows = [json.loads(l) for l in open(os.path.join(a.out, 'search_outline.jsonl'))]; PT = SW = []
    with open(os.path.join(a.out, 'search_outline.jsonl'), 'a' if a.reuse else 'w') as fo:
        for pt, sw in itertools.product(PT, SW):
            r = {}
            for k, F in fronts.items():
                e = F.evaluate(neck_x=best['neck_x'], foot=M['z_bust'] + best['foot_over_bust'], belly=best['belly'] or 0.001,
                               outline=[[0.04, 0.0], [pt, 0.40], [1.0, 0.0]], sw=sw, zone=zone, iters=30, area=area)
                r[k] = {q: e[q] for q in ('lapel', 'lap_s1_95', 'lap_s2_5', 'lost', 'wscale') if q in e}
            row = dict(point_t=pt, swell=sw, fronts=r, lapel=float(np.mean([v['lapel'] for v in r.values()])))
            orows.append(row); fo.write(json.dumps(row) + '\n'); fo.flush()
            print('derive: outline point_t %.2f swell %+.3f: turned lapel energy %.5f' % (pt, sw, row['lapel']), flush=True)
    ook = [r for r in orows if LIMITS['point_t'][0] - 1e-9 <= r['point_t'] <= LIMITS['point_t'][1] + 1e-9
           and LIMITS['swell'][0] - 1e-9 <= r['swell'] <= LIMITS['swell'][1] + 1e-9]
    olo = min(r['lapel'] for r in ook)
    obest = min([r for r in ook if r['lapel'] <= (1 + NOISE) * olo],
                key=lambda r: (abs(r['point_t'] - 0.27) + abs(r['swell'] - 0.025)))
    ws = float(np.mean([v['wscale'] for v in obest['fronts'].values()]))
    derived = dict(neck_x=best['neck_x'], foot_over_bust=best['foot_over_bust'], belly=best['belly'] or 0.001,
                   outline=[[0.04, 0.0], [obest['point_t'], round(0.40 * ws, 4)], [1.0, 0.0]],
                   outline_round=[0.05, obest['swell']])
    res = dict(current=CURRENT, derived=derived, limits=LIMITS, area=area,
               current_score={k: {q: base[k][q] for q in ('lapel', 'lap_s1_95', 'lap_s2_5', 'roll_stretch')} for k in base},
               best_roll=best, best_outline=obest,
               foldfront_options=dict(recipe='notch_lapel', **{q: derived[q] for q in ('neck_x', 'foot_over_bust', 'belly', 'outline', 'outline_round')}))
    json.dump(res, open(os.path.join(a.out, 'derived.json'), 'w'), indent=1)
    # 4. the field with both roll lines
    lines = {}
    for k, F in fronts.items():
        sg = F.side
        def curve(nx, fob, be):
            zN = F.X[np.abs(F.X[:, 0] - sg * nx) < 0.15, 2].max()
            return TS.roll_curve(np.array([sg * nx, zN - 0.02]), np.array([-sg * M['lap'], M['z_bust'] + fob]), be)[1]
        lines[k] = {'current roll line': (curve(CURRENT['neck_x'], CURRENT['foot_over_bust'], CURRENT['belly']), 'k--'),
                    'derived roll line': (curve(derived['neck_x'], derived['foot_over_bust'], derived['belly']), 'g-')}
    field_png(fronts, os.path.join(a.out, 'field.png'), lines)
    search_png(rows, orows, os.path.join(a.out, 'search.png'))
    print('derive: chosen', json.dumps(res['foldfront_options']), flush=True)


def search_png(rows, orows, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fobs = sorted({r['foot_over_bust'] for r in rows}); fob = -1.35 if -1.35 in fobs else fobs[0]
    for be in sorted({r['belly'] for r in rows}):
        R = sorted([r for r in rows if r['belly'] == be and r['foot_over_bust'] == fob], key=lambda r: r['neck_x'])
        ax[0].plot([r['neck_x'] for r in R], [r['lapel'] for r in R], 'o-', label='belly %+.2f' % be)
    ax[0].axvspan(*LIMITS['neck_x'], color='0.9', label='tailoring window'); ax[0].set_xlabel('roll line top, neck_x')
    ax[0].set_ylabel('turned lapel strain energy (mean (s1-1)^2+(s2-1)^2)'); ax[0].legend(fontsize=8)
    ax[0].set_title('roll line (break %+.2f over the bust line, lapel area held)' % fob)
    PT = sorted({r['point_t'] for r in orows}); SW = sorted({r['swell'] for r in orows})
    G = np.array([[next(r['lapel'] for r in orows if r['point_t'] == p and r['swell'] == s) for s in SW] for p in PT])
    im = ax[1].imshow(G, origin='lower', cmap='viridis', aspect='auto')
    ax[1].set_xticks(range(len(SW))); ax[1].set_xticklabels(['%+.3f' % s for s in SW]); ax[1].set_xlabel('long edge swell (S-curve < 0)')
    ax[1].set_yticks(range(len(PT))); ax[1].set_yticklabels(['%.2f' % p for p in PT]); ax[1].set_ylabel('point_t (point down the line)')
    for i in range(len(PT)):
        for j in range(len(SW)):
            ax[1].text(j, i, '%.4f' % G[i, j], ha='center', va='center', fontsize=7, color='w')
    plt.colorbar(im, ax=ax[1]); ax[1].set_title('outline on the chosen roll line')
    plt.tight_layout(); plt.savefig(path, dpi=70); plt.close(fig)


if __name__ == '__main__':
    main()
