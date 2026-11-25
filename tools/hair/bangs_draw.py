"""Drawings for bangs_flow.py: the flow field over the clay head (side, front, 3/4 and close-ups), the w profile
against the photo's strands, the field over the photo with its gap list, and the hairline gauge on the whole head.
Colours mark design elements only: the ponytail's funnel (blue), its re-solved ear and nape lines (green, purple),
the temple lock (magenta), the bangs' centre panel (orange, roots inside w0, pure fall) and the bangs' wings (yellow,
roots past w0, blended with the side flow)."""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF
import ponytail_flow as PF
import bangs_flow as BF

PF.COL.update({'bangs centre': '#e07b12', 'bangs wings': '#d9c21a'})


def tie_lines(B, n=1400, seed=3):
    """The ponytail's drawn lines, picked as ponytail_flow.main picks them (same seed), so the field reads the same."""
    FF = PF.free_fall(B)
    psi, rr = PF.bundle_places(B)
    rng = np.random.default_rng(seed)
    tie = np.where(B['basin'] == 0)[0]
    pick = rng.choice(tie, size=n, replace=False, p=B['hairs'][tie] / B['hairs'][tie].sum())
    for rs_ in (1, 2):
        rr_ = np.where(B['resolve'] == rs_)[0]
        pick = np.r_[pick, rng.choice(rr_, size=min(60, len(rr_)), replace=False)]
    lines = {}
    for i in np.unique(pick):
        pts, part = PF.strand(B, FF, psi, rr, i)
        lines[int(i)] = (pts, part, None)
    return FF, lines


def extras(B, roots, lines, prm, every=4):
    u0 = BF.u_of(B['P'][roots], prm)
    centre = [lines[j] for j in range(0, len(lines), every) if abs(u0[j]) < prm['w0_mm']]
    wings = [lines[j] for j in range(0, len(lines), every) if abs(u0[j]) >= prm['w0_mm']]
    lock = []
    for k_, (nm, c) in enumerate(PF.LOOSE.items()):
        li = np.where(B['basin'] == 2 + k_)[0]
        lock += [PF.loose_line(B, i, c['len']) for i in li[::6]]
    return [('temple lock', lock), ('bangs wings', wings), ('bangs centre', centre)]


def legend(fig, extra_label=''):
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=PF.COL[k], lw=3, label=l) for k, l in (
        ('tie', 'side flow B: to the tie'), ('ear wisp', 'B: re-solved ear lines'), ('temple lock', 'B: temple lock'),
        ('bangs centre', 'bangs, centre panel (|u| < w0)'), ('bangs wings', 'bangs, wings (|u| >= w0)'))]
    fig.legend(handles=hs, loc='lower center', ncol=5, fontsize=11, frameon=False)


def draw_all(B, SFL, roots, lines, prm, G, out, head, sheet, preset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PF.HEAD = head
    FF, tl = tie_lines(B)
    ex = extras(B, roots, lines, prm)
    # ---- the views
    fig, axs = plt.subplots(1, 4, figsize=(26, 9.6), facecolor='white')
    for ax, (az, el, t) in zip(axs, ((90, 0, "side (the photo's camera)"), (0, 0, 'front'), (35, 8, '3/4 front'),
                                     (180, 0, 'back'))):
        PF.draw_view(ax, B, FF, tl, ex, az, el, t)
    legend(fig)
    fig.suptitle('Bangs (%s): t = normalize(w(u) B + (1 - w(u)) D), w = smoothstep(%.1f, %.1f mm), D %.1f deg off '
                 'the part. Streamlines only; colours mark design elements' % (preset, prm['w0_mm'], prm['w1_mm'],
                                                                             prm['d_deg']), fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.02)
    fig.savefig(os.path.join(out, 'bangs-field-views.png'), dpi=100)
    plt.close(fig)
    # ---- close-ups of the face (more lines)
    ex2 = extras(B, roots, lines, prm, every=2)
    fig, axs = plt.subplots(1, 3, figsize=(24, 9.5), facecolor='white')
    for ax, (az, el, t) in zip(axs, ((0, 0, 'front, close'), (35, 8, '3/4 front, close'), (90, 0, 'side, close'))):
        PF.draw_view(ax, B, FF, tl, ex2, az, el, t)
        ax.set_xlim(-0.5, 0.5) if az == 0 else ax.set_xlim(-0.55, 0.45)
        ax.set_ylim(0.3, 1.08)
    legend(fig)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.07, wspace=0.02)
    fig.savefig(os.path.join(out, 'bangs-field-closeups.png'), dpi=100)
    plt.close(fig)
    PF.HIDDEN = None


def draw_profile(fitj, prm, out):
    """The w profile: the photo's strands read as shares of B, the fitted smoothstep, and the preset's."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white')
    u = np.linspace(0, 190, 400)
    ax.plot(u, SF.sm(fitj['w0_mm'], fitj['w1_mm'], u), color='#222', lw=2,
            label='fitted to the photo: w0 %.1f, w1 %.1f mm (rms %.3f)' % (fitj['w0_mm'], fitj['w1_mm'], fitj['rms_w']))
    if (prm['w0_mm'], prm['w1_mm']) != (fitj['w0_mm'], fitj['w1_mm']):
        ax.plot(u, SF.sm(prm['w0_mm'], prm['w1_mm'], u), '--', color='#c2337a', lw=2,
                label='this preset: w0 %.1f, w1 %.1f mm' % (prm['w0_mm'], prm['w1_mm']))
    for r in fitj['table']:
        c = '#2f6fb3' if r['side_flow_only'] else '#e07b12'
        ax.plot(r['u_mm'], r['w_photo'], 'o', color=c, ms=8)
        ax.annotate('%s %.0f°' % (r['name'].replace('strand', 's'), r['photo_deg']), (r['u_mm'], r['w_photo']),
                    textcoords='offset points', xytext=(5, -12), fontsize=8)
    for x, lab in ((66, 'bangs section edge'), (73.4, 'temporal point')):
        ax.axvline(x, color='#888', ls=':', lw=1); ax.text(x + 1, 0.02, lab, rotation=90, fontsize=8, color='#555')
    ax.set_xlabel('u: distance from the part along the skin, mm')
    ax.set_ylabel('w: share of the side flow B')
    ax.set_title('Back-flow strength w(u). Dots: the photo strands, w read off each lean between D (%.1f°) and B (%.1f°);'
                 '\norange: fitted, blue: behind the ear front (B alone)' % (fitj['phi_D_deg'], fitj['phi_B_deg']),
                 fontsize=10)
    ax.set_ylim(-0.05, 1.08); ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'bangs-w-profile.png'), dpi=110)
    plt.close(fig)


def draw_photo(B, SFL, roots, lines, prm, sheet, photo_png, out):
    """The field over the photo through photo_fit, the measured strands (white), the field's lean at each (cyan),
    and the gap list."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(photo_png).convert('RGB')
    sc, tx, ty = BF.photo_fit(sheet, B)
    to_px = lambda P_: np.c_[sc * P_[:, 1] + tx, -sc * P_[:, 2] + ty]
    smp = BF.photo_samples(sheet)
    Xs = BF.sample_points(B, smp, (sc, tx, ty))
    t, mag, w, u = BF.t_at(B, SFL, Xs, prm)
    Nn = B['surf'].normals(BF.unit(Xs - SF.CENTRE))
    Bv = BF.B_at(SFL, Xs, Nn)
    phm, phb = BF.side_deg(t), BF.side_deg(Bv)
    fig, ax = plt.subplots(figsize=(11, 11), facecolor='white')
    ax.imshow(img)
    u0 = BF.u_of(B['P'][roots], prm)
    for j in range(0, len(lines), 3):
        p = lines[j]
        if p[0, 0] < -0.01:
            continue
        q = to_px(p)
        ax.plot(q[:, 0], q[:, 1], color=PF.COL['bangs centre' if abs(u0[j]) < prm['w0_mm'] else 'bangs wings'],
                lw=0.7, alpha=0.9)
    for k_, (nm, c) in enumerate(PF.LOOSE.items()):
        for i in np.where(B['basin'] == 2 + k_)[0][::4]:
            p = PF.loose_line(B, i, c['len'])
            if p[0, 0] > 0:
                q = to_px(p); ax.plot(q[:, 0], q[:, 1], color=PF.COL['temple lock'], lw=0.7)
    # the lean of the built bang lines where they pass the photo strand (their lift included), when any pass within
    # 12 px of its middle; else the field's own lean on the skin there
    # (every built family is read, outermost first: the bangs, then the temple lock, then the ponytail's scalp lines,
    # so each photo strand is compared with the model's hair that lies on top there)
    fams = [('bang lines', list(lines))]
    lk = []
    for k_, (nm, c) in enumerate(PF.LOOSE.items()):
        lk += [PF.loose_line(B, i, c['len']) for i in np.where(B['basin'] == 2 + k_)[0]]
    fams.append(('temple lock', lk))
    ring = B['r_tie'] / BF.MM + PF.RING_MM
    fams.append(('side flow lines', [B['X'][i][B['rho'][i] >= ring] for i in np.where(B['basin'] == 0)[0][::3]]))
    fam_segs = []
    for nm, pl in fams:
        sg = [np.c_[to_px(p)[:-1], to_px(p)[1:]] for p in pl if len(p) > 2 and p[len(p) // 2, 0] > 0]
        sg = np.concatenate(sg)
        fam_segs.append((nm, sg, 0.5 * (sg[:, :2] + sg[:, 2:])))
    src = []
    for j, s in enumerate(smp):
        for nm, segs, mids in fam_segs:
            k = np.linalg.norm(mids - s['mid'][None], axis=1) < 12
            if k.sum() >= 3:
                break
        if k.sum() >= 3:
            d = segs[k, 2:] - segs[k, :2]
            a2 = 2 * np.arctan2(d[:, 0], d[:, 1])                 # an axial mean: a line has no head or tail
            m = math.atan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2
            deg = math.degrees(m)
            deg = (deg + 90) % 180 - 90                          # as refmeasure reads a lean: -90 to 90 from plumb
            phm[j] = deg; src.append(nm)
        else:
            src.append('field')
    rows = []
    for s, x, pm, pb, ww, uu, sr in zip(smp, Xs, phm, phb, w, u, src):
        ax.plot([s['a'][0], s['b'][0]], [s['a'][1], s['b'][1]], color='white', lw=2.2)
        L = 0.5 * np.linalg.norm(s['b'] - s['a'])
        d = np.array([math.sin(math.radians(pm)), math.cos(math.radians(pm))])
        ax.plot([s['mid'][0] - d[0] * L, s['mid'][0] + d[0] * L], [s['mid'][1] - d[1] * L, s['mid'][1] + d[1] * L],
                color='#35c8ff', lw=2)
        ax.text(s['b'][0] + 3, s['b'][1], s['name'].replace('strand', 's'), color='w', fontsize=8)
        gp = (pm - s['deg'] + 90) % 180 - 90                     # (axial: a lean of -80 is a rise of 100)
        rows.append(dict(name=s['name'], photo_deg=round(s['deg'], 1), field_deg=round(float(pm), 1),
                         gap_deg=round(float(gp), 1), side_flow_B_deg=round(float(pb), 1),
                         u_mm=round(float(uu), 1), w=round(float(ww), 3), read_from=sr,
                         flagged=bool(abs(gp) > 10)))
    # end heights against the photo, by the eye
    lm = sheet['landmarks']
    eye_px = lm['eye'][1]
    L_ = B['F']['landmarks']
    def pz(z):
        return (eye_px - (z - L_['pupil'][2]) * sc)
    ends = np.array([p[-1] for p in lines])
    cen = np.abs(u0) < 10
    side = (np.abs(u0) > 55)
    for nm, k, ph in (('fringe edge (|u|<10)', cen, lm['fringe_edge']), ('side ends (|u|>55)', side, lm['side_end'])):
        zf = float(np.median(ends[k, 2]))
        rows.append(dict(name=nm, photo_px_y=ph[1], field_px_y=round(float(pz(zf)), 1),
                         gap_heads=round(float((pz(zf) - ph[1]) / abs(lm['menton'][1] - lm['head_top'][1])), 3),
                         flagged=bool(abs(pz(zf) - ph[1]) / 616 > 0.05)))
        ax.axhline(pz(zf), color='#e07b12', lw=0.8, ls='--', xmax=0.4)
        ax.axhline(ph[1], color='white', lw=0.8, ls=':', xmax=0.4)
    ax.set_xlim(100, 760); ax.set_ylim(700, 0)
    y = 470
    ax.text(110, y - 18, 'white: photo strand; cyan: the field\'s lean there; dashed: field end height, dotted: photo\'s',
            color='w', fontsize=8, bbox=dict(facecolor='k', alpha=0.6))
    for r in rows:
        if 'photo_deg' in r:
            txt = '%-9s photo %5.1f  model %5.1f  gap %+5.1f  (B %5.1f) u %5.1f w %.2f %s' % (
                r['name'], r['photo_deg'], r['field_deg'], r['gap_deg'], r['side_flow_B_deg'], r['u_mm'], r['w'],
                {'bang lines': 'bangs', 'temple lock': 'lock', 'side flow lines': 'side'}.get(r['read_from'], 'field'))
        else:
            txt = '%-22s gap %+.3f head' % (r['name'], r['gap_heads'])
        ax.text(110, y, txt, color='w', fontsize=7.5, family='monospace',
                bbox=dict(facecolor='#a01010' if r['flagged'] else 'k', alpha=0.6, pad=1))
        y += 15
    ax.axis('off')
    fig.savefig(os.path.join(out, 'bangs-over-photo.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)
    return rows


def hairline(B, roots, lines, out, cache_dir):
    """hairline_gap.py on the whole head: the ponytail's lines (every root, as ponytail_flow.main gauges them) with
    these bangs in place of the old ones."""
    import hairline_gap as HG
    tie = np.where(B['basin'] == 0)[0]
    ring = B['r_tie'] / BF.MM + PF.RING_MM
    paths = [B['X'][i][B['rho'][i] >= ring] for i in tie]
    w = list(B['hairs'][tie])
    paths += list(lines); w += list(B['hairs'][roots])
    for k, (nm, c) in enumerate(PF.LOOSE.items()):
        li = np.where(B['basin'] == 2 + k)[0]
        paths += [PF.loose_line(B, i, c['len'], k=30) for i in li]; w += list(B['hairs'][li])
    keep = [j for j, p in enumerate(paths) if len(p) >= 2]
    PF.CACHE = cache_dir
    memo = PF.Memo('hairline', ('bangs', PF.file_key(SF.__file__), PF.file_key(HG.__file__)))
    H, Wc = HG.measure(B['F'], B['surf'], [paths[j] for j in keep], np.array(w)[keep], memo=memo)
    memo.save()
    PF.CACHE = None
    rep = HG.report(H, Wc)
    HG.draw(H, Wc, os.path.join(out, 'hairline-gauge.png'))
    return {k: dict(cells=v['cells'], bare=len(v['bare']), longest_gap_mm=v['longest_gap_mm'],
                    thinnest_per_mm=v['thinnest'], bare_at=v['bare']) for k, v in rep.items()}
