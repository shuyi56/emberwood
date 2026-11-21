"""The draft topology check (John, 2026-11-21): the topology checks of blender/shared/drafttopo.py run at the drafting
stage, before anything is sewn or solved, on a real draft. Report-only, like the flow check and the face topology
check: it never stops a render; it writes what it found and a sheet of cheap diagrams.

    draft-topo.json         every check: pass, fail or watch, the value against its limit, why, and its source
    draft-topo-sheet.png    a tile per piece (its edge loops by seam label, holes red, singular points ringed, the
                            curvature as dots and the dart spot), the seam graph, the parts-and-holds graph and each
                            closure's buttons against its break, pressure points and strain

The adapters read a draft from what the garment modules already keep:

    boot      blender/boot/last.py's PIECES (each piece as cut on the last: its points, faces and seam letters, the
              eyelets), the sole's rim labelled by the upper piece over it, the lace threaded through the eyelets
              as sew.lacing threads it (main: the ends tucked in the top holes; --lace bow: tied in a bow, as on the
              bow branch)
    bodice    the redingote bodice's closing edge (its buttons as bodice.py sets them, from the neck down at
              button_step), the lapel's break point (collar.py's LAPEL drop for the style), the bust and waist lines
              from the measures, and the ease's strain round the body at each height
    hair      a bun or French twist hold test's packed npz (rq_bun_pack.py): every strand, its root against the
              scalp, and the holds rq_bun.py builds (tie, tucks, U-pins, net, core), as a parts graph

Run it:

    in Blender           python3 tools/render-queue/rq_run.py -- blender/shared/draft_topo_views.py --figure boot --out DIR
                         (builds the figure as drafted, no solver; --clay adds cheap orthographic clay views)
    on a dumped draft    python3 rq_drafttopo.py --draft DRAFT.pkl --out DIR
    on a hair test       python3 rq_drafttopo.py --hair blender/hair/data/bun-after.npz [--bun-tools DIR] --out DIR
    self test            python3 rq_drafttopo.py --selftest"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', '..', 'blender', 'shared'), os.path.join(HERE, 'shared'), HERE):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
import drafttopo as D  # noqa: E402

CM = 165.0 / 63.0            # a model unit in cm (a 165 cm figure is 63 units tall)


# ------------------------------------------------------------------------------------------------ the boot

def _runs_of(p, label):
    """The boundary runs of one label on a piece's main part (vertex lists; its stray islands left out)."""
    r = D.piece_report(dict(D.main_component(p)[0], flat=True), flatten=False)
    return [x['verts'] for lp in r['loops'] for x in lp['runs'] if x['label'] == label]


def boot_draft(pieces, ratios=None, lace='tucked', side='R'):
    """The boot as drafted: the pieces, their seams (each run paired with the run nearest it on the partner piece),
    the parts graph of the lacing, and what the sewn boot should be."""
    P = []
    for p in pieces:
        q = dict(name=p['name'], V=np.asarray(p['V'], float), F=[list(f) for f in p['F']],
                 edges={k: np.array(v, copy=True) for k, v in (p.get('edges') or {}).items()}, material='leather',
                 visible=True, allowance=0.0, solid=p['name'] == 'sole')
        P.append(q)
    by = {p['name']: p for p in P}
    # 1. split labels with several runs on one piece into their own labels (the vamp's E runs down each side)
    for p in P:
        for lab in list(p['edges']):
            rs = _runs_of(p, lab)
            if len(rs) > 1:
                m = p['edges'].pop(lab)
                m = D._mask(m, len(p['V']))
                for k, vs in enumerate(sorted(rs, key=len, reverse=True)):
                    mk = np.zeros(len(p['V']), bool)
                    mk[vs] = True
                    p['edges']['%s%d' % (lab, k + 1)] = mk
                    m[vs] = False
                if m.any():
                    p['edges'][lab + '~'] = m          # (what is left: interior or stray points of the label)
    # 2. the sole: a slab, a closed solid (chi 2), not a sheet; the upper's lower edge (E) is stitched onto its welt,
    #    a lap onto its face (the welt seam), not an edge to an edge
    Es = [(p['name'], lab) for p in P if p['name'] != 'sole' for lab in p['edges']
          if lab.startswith('E') and not lab.endswith('~')]
    seams = [[lab, nm, 'sole', None, 'lap'] for nm, lab in Es]
    # 3. the other seams: a label on two pieces, each one run (A inner, A outer, C, D, G); H the tongue's sides lapped
    #    onto the panels' faces (the panels carry H as a band, not an edge)
    seen = {}
    for p in P:
        if p['name'] == 'sole':
            continue
        for lab in p['edges']:
            if lab.startswith('E') or lab.endswith('~') or lab in ('L', 'F', 't'):
                continue
            seen.setdefault(lab.rstrip('0123456789'), []).append((p['name'], lab))
    for base, where in sorted(seen.items()):
        edge_on = [(nm, lab) for nm, lab in where if _runs_of(by[nm], lab)]
        if base == 'H' or (len(edge_on) == 1 and len(where) > 1):
            # (a lapped seam: each edge run onto the face of the piece whose band of the label lies nearest it)
            for nm, lab in edge_on:
                others = [(o, ol) for o, ol in where if o != nm]
                if others:
                    o, ol = min(others, key=lambda x: _run_gap(by, (nm, lab), x))
                    seams.append([lab, nm, o, None, 'lap'])
            continue
        pieces_with = sorted({nm for nm, _ in edge_on})
        if len(pieces_with) == 2:
            a = [x for x in edge_on if x[0] == pieces_with[0]]
            b = [x for x in edge_on if x[0] == pieces_with[1]]
            # (several runs: pair by nearness)
            for la in a:
                best = min(b, key=lambda lb: _run_gap(by, la, lb))
                seams.append([la[1], la[0], best[0], best[1]])
        else:
            for nm, lab in edge_on:
                seams.append([lab, nm, nm])          # (reported: a label on one piece only)
    for p in P:
        p['openings'] = [l for l in p['edges'] if l in ('L', 'F', 't') or l.startswith(('L', 'F'))]
        # (the tongue's top stands free above the panels: an edge meant to stay open)
        if p['name'] == 'tongue':
            # (the tongue's free top, standing above the panels: its highest unlabelled run, named 't'; any other
            # unlabelled run of its edge is an edge left unsewn)
            r = D.piece_report(dict(D.main_component(p)[0], flat=True), flatten=False)
            free = [x['verts'] for lp in r['loops'] for x in lp['runs'] if x['label'] is None]
            if free:
                top = max(free, key=lambda vs: p['V'][vs, 2].mean())
                mk = np.zeros(len(p['V']), bool)
                mk[top] = True
                p['edges']['t'] = mk
            p['openings'].append('t')
    # 4. the lacing: eyelets set in the panels, the lace through them, crossing the opening, as sew.lacing does
    nodes, holds = ['foot'], []
    for p in P:
        nodes.append(p['name'])
    for p in P:                 # (the leg is in the boot: every piece that wraps the foot or the shin lies on it)
        holds.append(['foot', p['name'], 'worn on'])
    for s in seams:
        holds.append([s[1], s[2], 'seam ' + str(s[0]).rstrip('0123456789').split(' ')[0]])
    ein = list(pieces_by(pieces, 'inner shaft').get('eyelets', []))
    eout = list(pieces_by(pieces, 'outer shaft').get('eyelets', []))
    expect = []
    for q, E in (('inner', ein), ('outer', eout)):
        for k in range(len(E)):
            e = 'eyelet %s %d' % (q, k + 1)
            nodes.append(e)
            expect.append(e)
            holds.append([e, q + ' shaft', 'set'])
    npair = min(len(ein), len(eout))
    captured = []
    for k in range(npair - 1):
        for j, (qa, qb) in enumerate((('inner', 'outer'), ('outer', 'inner'))):
            seg = 'lace %d%s' % (k + 1, 'ab'[j])
            nodes.append(seg)
            holds.append([seg, 'eyelet %s %d' % (qa, k + 1), 'through eyelet', 'sliding'])
            holds.append([seg, 'eyelet %s %d' % (qb, k + 2), 'through eyelet', 'sliding'])
    segs = [x for x in nodes if x.startswith('lace ')]
    # (the cord is one: each crossing runs on into the next through the hole they share: already in the graph)
    loaded = list(segs)
    if lace == 'bow':
        # the two ends out of the top holes tied over the opening: a knot, two loops and two tails, all pinned at the knot
        for x in ('knot', 'loop L', 'loop R', 'tail L', 'tail R'):
            nodes.append(x)
        top = ['lace %d%s' % (npair - 1, c) for c in 'ab']
        for t in top:
            holds.append([t, 'knot', 'tied'])
        for x in ('loop L', 'loop R', 'tail L', 'tail R'):
            holds.append([x, 'knot', 'tied'])
        captured = segs + ['knot', 'loop L', 'loop R', 'tail L', 'tail R']
    parts = dict(nodes=nodes, anchors=['foot'], holds=holds, loaded=loaded, min_holds=2, captured=captured,
                 expect_used=expect)
    # the boot sewn: the vamp, toe cap and two shafts sewn to each other are one sheet round the foot, open at the top
    # (the top edge and the lacing slit) and along the welt: an annulus, chi 0. Its welt is stitched all round onto
    # the sole, a closed solid (a sphere, chi 2), along a circle (chi 0): 2. The tongue (a disc, 1) is joined on three
    # arcs, its bottom to the vamp's throat and each side lapped under a panel (each -1): 2 + 1 - 3 = 0. Two edges stay
    # free: the leg's opening (the panels' top edges and the lacing slit) and the tongue's top, an arc from under one
    # panel to under the other.
    expected = dict(pieces_chi={p['name']: (2 if p['name'] == 'sole' else 1) for p in P}, chi=0, genus=0,
                    openings=['top and lacing', 'tongue top'])
    return dict(name='boot %s (%s lace)' % (side, lace), unit_cm=CM, pieces=P, seams=seams, parts=parts,
                expected=expected, seam_ease=0.05, reach=0.5)


def pieces_by(pieces, name):
    return next(p for p in pieces if p['name'] == name)


def _run_gap(by, la, lb):
    from scipy.spatial import cKDTree
    A = by[la[0]]['V'][np.nonzero(D._mask(by[la[0]]['edges'][la[1]], len(by[la[0]]['V'])))[0]]
    B = by[lb[0]]['V'][np.nonzero(D._mask(by[lb[0]]['edges'][lb[1]], len(by[lb[0]]['V'])))[0]]
    return float(cKDTree(B).query(A)[0].mean())


# ------------------------------------------------------------------------------------------------ the bodice

# the lapel styles' break, in w below the neck edge's front (blender/collar/collar.py STYLES on the lapel branch), and
# how far under the break collar.open_front cuts the buttons away (0.15 w)
LAPEL_DROP = {'buttoned to the neck': None, 'turned': 1.60, 'notch': 1.35, 'modern': 3.6}
LAPEL_CLEAR = 0.15


def bodice_draft(M, ratios, buttons_z, edge, style='buttoned to the neck', strain=None, break_z=None):
    """The redingote bodice's closing: the front edge from the neck to the front's point (edge: points top to bottom),
    the buttons on it (their heights, as bodice.py set them), the lapel's break for the style (the buttons above it,
    and within LAPEL_CLEAR w under it, cut away as collar.open_front does), the bust and waist lines, and the strain
    along the edge where a solve has measured one. Arc lengths are measured down the edge from the neck. break_z: the
    build's own break rule (collar.break_z(z_nominal, M), which sets the break on the top button), called with the
    drafted drop's height; without it the drafted drop is the break."""
    w = M['w']
    E = np.asarray(edge, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(E, axis=0), axis=1))]

    def s_at(z):
        return float(np.interp(-z, -E[:, 2], s))
    drop = LAPEL_DROP[style]
    zb = None if drop is None else M['z_neck_front'] - drop * w
    if zb is not None and break_z is not None:
        zb = float(break_z(zb, M))
    bz = [z for z in buttons_z if zb is None or z <= zb - LAPEL_CLEAR * w]
    clos = dict(name='front closing (%s)' % style, edge=E, buttons=[s_at(z) for z in bz],
                break_s=None if zb is None else s_at(zb), neck_s=0.0 if zb is None else None,
                end_s=s_at(M['z_hem_front']), pressure=dict(bust=s_at(M['z_bust']), waist=s_at(M['z_waist'])),
                button_d=2 * ratios['button'] * w, extension=ratios['lap'] * w, hole_len=None, strain=strain, _unit=CM)
    return dict(name='redingote bodice, %s' % style, unit_cm=CM, pieces=[], closures=[clos])


# ------------------------------------------------------------------------------------------------ hair

def hair_draft(npz, bun_tools=None, state='drawn', name=None):
    """A bun or French twist as drafted: each strand a part; the scalp the anchor. A strand's root is a hold only when
    it lies on the scalp (the stress test pins every strand's first two points, wherever they are: a root in the air
    is a pin to nothing, a part that only the solver's own pin holds up). The other holds are those rq_bun.py builds
    for the style's BUN settings (the tie and the core, the gather's ends, the tucked ends, each U-pin, the net)."""
    d = np.load(npz, allow_pickle=True)
    counts = d['counts'].astype(int)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    runs = [np.arange(a, a + c) for a, c in zip(starts, counts)]
    X = np.asarray(d['s_' + state], float)
    thick = np.asarray(d['thick'], float)
    part = d['part'].astype(int)
    pnames = [str(x) for x in d['parts']]
    bun = json.loads(str(d['bun'])) if 'bun' in d.files else {}
    HV, HF = np.asarray(d['head_v'], float), np.asarray(d['head_f'], int)
    from scipy.spatial import cKDTree
    # the scalp: distance from each root to the head's surface (its points, densely enough: the head mesh is fine)
    tree = cKDTree(HV)
    el = np.median(np.linalg.norm(HV[HF[:, 0]] - HV[HF[:, 1]], axis=1))
    root_d = np.array([tree.query(X[r[0]])[0] for r in runs])
    on_scalp = root_d < max(0.02, 2.5 * el)                  # (HH: within a lock's half width of the skin)
    nodes = ['scalp'] + ['strand %d' % k for k in range(len(runs))]
    holds = []
    for k, r in enumerate(runs):
        if on_scalp[k]:
            holds.append(['strand %d' % k, 'scalp', 'root'])
    held_pts = np.asarray(d['held'], bool) if 'held' in d.files else np.zeros(len(X), bool)
    K = None
    if bun_tools and any(bun.get(k) for k in ('tie', 'core', 'pins', 'net', 'tucks')):
        sys.path.insert(0, bun_tools)
        import rq_bun as B
        B.BUN.update(bun)
        strands = [dict(w=thick[r]) for r in runs]
        K = B.knot_of(strands, X, runs)
        sid = np.concatenate([np.full(len(r), k) for k, r in enumerate(runs)])
        if bun.get('tie'):
            nodes.append('tie')
            holds.append(['tie', 'scalp', 'elastic'])
            holds.append(['strand %d' % K['core'], 'tie', 'wound on'])
            for k, r in enumerate(runs):
                if k in K['strands']:
                    continue
                if B._near_core(X[r[-1]][None], K, 1.25, 0.05)[0]:
                    holds.append(['strand %d' % k, 'tie', 'gathered'])
        if bun.get('tucks'):
            starts_f = np.array([X[runs[k][0]] for k in K['free']]) if K['free'] else np.zeros((0, 3))
            for k in K['body']:
                if k == K['core']:
                    continue
                e = X[runs[k][-1]]
                if len(starts_f) and np.linalg.norm(starts_f - e, axis=1).min() < 0.03:
                    continue
                if B._near_core(e[None], K, 1.4, 0.0)[0]:
                    holds.append(['strand %d' % k, 'strand %d' % K['core'], 'tucked'])
        if bun.get('pins'):
            body = np.concatenate([runs[k] for k in K['body']])
            lines = [(p0, dd, 2.0 * K['core_r']) for p0, dd in B._pin_lines(K)] if not bun.get('roll') else B._pin_lines_roll(K)
            for pi, (p0, dd, depth) in enumerate(lines):
                pn = 'pin %d' % (pi + 1)
                nodes.append(pn)
                v = X[body] - p0
                a = v @ dd
                dist = np.linalg.norm(v - np.outer(a, dd), axis=1)
                caught = body[(dist < bun.get('pin_catch', 0.02)) & (a > 0) & (a < depth)]
                for k in sorted(set(sid[caught].tolist())):
                    holds.append(['strand %d' % k, pn, 'U-pin'])
                # (a pin is driven through the knot into the base: it holds to the scalp's hair, the gathered tail
                # under it; here to the scalp itself, as the solve pins its points to the head)
                holds.append([pn, 'scalp', 'driven in'])
        if bun.get('net'):
            import rq_hair as R
            idx = K['idx']
            a, b = R.neighbours(X[idx], 0.6 * R.NEAR * 1.0)
            o = sid[idx][a] != sid[idx][b]
            pairs = {(min(x, y), max(x, y)) for x, y in zip(sid[idx][a][o], sid[idx][b][o])}
            for x, y in sorted(pairs):
                holds.append(['strand %d' % x, 'strand %d' % y, 'net'])
    # the knot's strands (and the core) are the loaded parts: they carry the bun's weight
    if K is not None:
        loaded = ['strand %d' % k for k in K['body']]
    else:
        # (no hold settings: the knot's parts by their part number, everything but the cap and gather)
        loaded = ['strand %d' % k for k in range(len(runs)) if part[k] not in (0,)]
    info = dict(roots_on_scalp=int(on_scalp.sum()), roots_off_scalp=int((~on_scalp).sum()),
                off_scalp_by_part={pnames[p]: int(((part == p) & ~on_scalp).sum()) for p in np.unique(part)},
                root_gap_max=float(root_d.max()), held_points=int(held_pts.sum()), bun=bun)
    geom = {('strand %d' % k): X[r] for k, r in enumerate(runs)}
    if K is not None:
        geom['tie'] = X[runs[K['core']]].mean(0)
        if bun.get('pins'):
            for pi, L_ in enumerate(lines):
                geom['pin %d' % (pi + 1)] = L_[0]
    parts = dict(nodes=nodes, anchors=['scalp'], holds=holds, loaded=loaded, min_holds=2,
                 part_of={('strand %d' % k): pnames[part[k]] for k in range(len(runs))}, geom=geom, backdrop=HV)
    nm = name or os.path.splitext(os.path.basename(npz))[0]
    return dict(name=nm, unit_cm=22.5, pieces=[], parts=parts, info=info)


def hair_summary(rep, draft):
    """The parts check by the bun's own design parts: how many strands of each part float or are short of holds."""
    P = rep['parts']
    po = draft['parts']['part_of']
    out = {}
    fl = set(P['floating'])
    short = {x['part'] for x in P['loaded'] if x['paths'] < x['need']}
    one = {x['part'] for x in P['loaded'] if x['paths'] == 1}
    for s, pn in po.items():
        o = out.setdefault(pn, dict(strands=0, floating=0, short=0, single=0))
        o['strands'] += 1
        o['floating'] += s in fl
        o['short'] += s in short
        o['single'] += s in one
    return out


# ------------------------------------------------------------------------------------------------ run and write

def write(draft, out, stem='draft-topo', title=None):
    os.makedirs(out, exist_ok=True)
    rep = D.run(draft)
    if draft.get('info'):
        rep['info'] = draft['info']
    if draft.get('parts', {}).get('part_of'):
        rep['by_part'] = hair_summary(rep, draft)
    js = os.path.join(out, stem + '.json')
    open(js, 'w').write(D.to_json(rep))
    png = D.sheet(rep, draft, os.path.join(out, stem + '-sheet.png'), title=title)
    print(D.summary(rep), flush=True)
    print('rq drafttopo: %s -> %s, %s' % (draft['name'], js, png), flush=True)
    return rep


def main(argv):
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d      # noqa: E731
    out = opt('--out', 'draft-topo')
    if '--selftest' in argv:
        res = D.selftest()
        for nm, ok in res:
            print('%s  %s' % ('ok  ' if ok else 'FAIL', nm))
        print('%d of %d pass' % (sum(ok for _, ok in res), len(res)))
        return 0 if all(ok for _, ok in res) else 1
    if opt('--hair'):
        for path in opt('--hair').split(','):
            dr = hair_draft(path, opt('--bun-tools'), state=opt('--state', 'drawn'))
            write(dr, out, stem='draft-topo-' + os.path.splitext(os.path.basename(path))[0])
        return 0
    if opt('--draft'):
        p = opt('--draft')
        dr = pickle.load(open(p, 'rb')) if p.endswith('.pkl') else json.load(open(p))
        write(dr, out)
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
