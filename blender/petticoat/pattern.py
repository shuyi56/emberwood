"""The petticoat as a tailor cuts it: flat pattern pieces drafted from the corseted body's measurements, each seam
edge paired with the edge it is sewn to. Nothing here is a shell round the body: draft.pattern() only measures (the
line the cloth takes over the corset and hips, and its length from the band to the floor at every angle), and the
pieces are drawn flat from those numbers, as a cutter drafts from a measuring tape.

Pieces (a gored petticoat of about 1890):

  front gore      cut 1 on the fold (straight grain at the centre front); a hip dart each side
  side gore       cut 2; straight grain on its front edge, the flare all on its back edge (bias); a hip dart
  back breadth    cut 2; straight grain on the centre back; wider at the top, gathered into the back of the band, and
                  longer at the centre back for the train; the centre-back seam is left open at the top (the placket)
  waistband       cut 1; a straight strip folded double, lapped at the centre back for a button
  flounce strip   cut 3; straight strips joined end to end into a ring, gathered GATHER times onto the skirt

Seams (pairs of edges sewn together; every pair's lengths are checked equal, or in its gather ratio):

  front gore side edge   = side gore front edge          (left and right)
  side gore back edge    = back breadth side edge        (left and right)
  back breadth CB edge   = the other back breadth's CB edge, below the placket
  dart legs              each dart's two legs
  skirt top edges        = waistband lower edge          (flat over the front and sides, gathered over the back)
  flounce strip ends     = the next strip's start        (three seams, a ring)
  flounce top edge       = the flounce line on the skirt (gathered GATHER times)
  hems                   the skirt's and the flounce's, turned up by the hem allowance and stitched (no pair)

Tucks: three tucks above the flounce line; their cloth (two depths each) is let into every gore's length, and their
fold and stitch lines are marked on each piece.

Everything is in model units (1 unit is about 2.6 cm); report() and the layout print centimetres."""
import json
import math

import numpy as np

import draft

CM = 165.0 / 63.0
# Seam allowances, in H (a period dressmaker's: 1.5 cm seams, 1 cm at the band, a 2.5 cm turned hem).
ALLOW = {'seam': 0.061, 'waist': 0.041, 'hem': 0.10, 'fold': 0.0, 'placket': 0.061}
# Where the seams fall round the body, as angles from the centre front (half the figure; the other half mirrors):
# the front gore to here, the side gore to here, the back breadth to the centre back.
SEAM_AT = (0.25 * math.pi, 0.62 * math.pi)
DART_SHARE = 0.5             # of each gore's waist suppression (its hip arc less its waist arc), taken in its dart(s)
DART_DEPTH = 0.62            # a dart's length, as a share of the waist-to-hip depth
GATHER_BACK = 2.0            # the back breadths' top against the back of the band (the gathered back)
GATHER = 1.6                 # the flounce strips against the flounce line
FLOUNCE_STRIPS = 3
TUCKS = 3
TUCK = 0.02                  # a tuck's depth, in H (about 5 mm: each takes 1 cm of cloth)
TUCK_GAP = 0.05              # between tucks' stitch lines, in H (about 1.2 cm)
HEADING = 0.045              # the flounce's heading above its seam, in H
LAP = 0.25                   # the band's lap at the centre back, in H


# ---------------------------------------------------------------- measures read off draft.pattern

def _arc(r_row, th, a, b, n=200):
    t = np.linspace(a, b, n)
    r = np.interp(t, th, r_row, period=2 * np.pi)
    x, y = r * np.sin(t), -r * np.cos(t)
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def measures(M, pat):
    """The cutter's numbers at each seam angle and over each gore: arcs at the waist, the hips and the hem, the depth
    from the waist to the hips, and the length from the band to the hem (the train at the back)."""
    th, z, r, s = pat['th'], pat['z'], pat['r'], pat['s']
    i_band = 0          # (the band's top row: the band is straight, its girth the corseted waist's with its ease)
    i_hip = int(np.argmin(np.abs(z - M['hips_z'])))
    i_hem = int(np.argmin(np.abs(z - pat['hem_z'])))
    a1, a2 = SEAM_AT
    spans = {'front': (0.0, a1), 'side': (a1, a2), 'back': (a2, math.pi)}
    out = {'H': M['H'], 'spans': spans}
    for k, (a, b) in spans.items():
        out[k] = {'waist': _arc(r[i_band], th, a, b), 'hips': _arc(r[i_hip], th, a, b), 'hem': _arc(r[i_hem], th, a, b)}

    def at(arr, ang):
        return float(np.interp(ang, th, arr, period=2 * np.pi))
    s_top = s[int(np.argmin(np.abs(z - pat['band_bot'])))]
    out['length'] = {ang: at(pat['length'] - s_top, ang) for ang in (0.0, a1, a2, math.pi)}
    out['hip_depth'] = {ang: at(s[i_hip] - s_top, ang) for ang in (0.0, a1, a2, math.pi)}
    out['band'] = 2.0 * sum(out[k]['waist'] for k in spans)
    out['back_band'] = 2.0 * out['back']['waist']
    return out


# ---------------------------------------------------------------- drawing flat pieces

def _curve(p0, p1, t0, t1, n=24):
    """A cubic from p0 to p1 leaving along t0 and arriving along t1 (unit-ish tangents scaled by the span)."""
    p0, p1, t0, t1 = (np.asarray(v, float) for v in (p0, p1, t0, t1))
    d = np.linalg.norm(p1 - p0)
    c0, c1 = p0 + t0 * d / 3, p1 - t1 * d / 3
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c0 + 3 * (1 - u) * u ** 2 * c1 + u ** 3 * p1


def _length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _edge(top, hip, hem_x, length, hip_depth):
    """A gore's side edge from its waist point down over the hip to the hem, as cut: a curve from the waist out to the
    hip, then straight to the hem; the hem point's height solved so the edge is length long. Returns the polyline."""
    top = np.asarray(top, float)
    hipp = np.array([hip, -hip_depth])

    def build(hy):
        hem = np.array([hem_x, hy])
        down = (hem - hipp) / np.linalg.norm(hem - hipp)
        upper = _curve(top, hipp, np.array([0.0, -1.0]) + 0.0 * down, down, 20)
        lower = np.linspace(hipp, hem, 30)
        return np.concatenate([upper, lower[1:]])
    lo, hi = -hip_depth - 1e-3, -4.0 * length
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _length(build(mid)) < length:
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))


def _straight(x, length):
    return np.array([[x, 0.0], [x, -length]])


def _hem(a, b, ta, tb):
    """The hem from point a to point b, square to the edges it leaves and meets (ta, tb: the edges' directions down)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na = np.array([-ta[1], ta[0]])
    nb = np.array([-tb[1], tb[0]])
    if na.dot(b - a) < 0:
        na = -na
    if nb.dot(b - a) < 0:
        nb = -nb
    return _curve(a, b, 0.75 * na, 0.75 * nb, 40)


def _dir(P, end=True):
    d = P[-1] - P[-2] if end else P[1] - P[0]
    return d / np.linalg.norm(d)


class Piece:
    """A flat pattern piece: its seam line as named edges in order round it (each a polyline, all run the same way
    round), the allowance on each, its grain, its marks (darts, tuck and flounce lines, notches), and how many to cut."""

    def __init__(self, name, cut, grain, fold=None):
        self.name, self.cut, self.grain, self.fold = name, cut, grain, fold
        self.edges = []          # (name, kind, points)
        self.darts = []          # (centre x, width at the top, depth)
        self.lines = []          # (name, points)

    def add(self, name, kind, pts):
        self.edges.append((name, kind, np.asarray(pts, float)))

    def edge(self, name):
        return next(p for n, k, p in self.edges if n == name)

    def outline(self):
        pts = [self.edges[0][2][0]]
        for _, _, p in self.edges:
            pts += list(p[1:])
        return np.array(pts)

    def cut_line(self):
        """The cutting line: the seam line let out by each edge's allowance (mitred at the corners)."""
        from shapely.geometry import Polygon
        poly = Polygon(self.outline())
        if not poly.is_valid:
            poly = poly.buffer(0)
        out = poly
        for name, kind, p in self.edges:
            a = ALLOW[kind] * H_UNIT[0]
            if a <= 0:
                continue
            from shapely.geometry import LineString
            out = out.union(LineString(p).buffer(a, cap_style='flat', join_style='mitre'))
        out = out.buffer(0.0)
        return np.array(out.exterior.coords)

    def to_dict(self):
        return dict(name=self.name, cut=self.cut, grain=self.grain, fold=self.fold,
                    edges=[dict(name=n, kind=k, allowance=ALLOW[k] * H_UNIT[0], points=p.round(4).tolist())
                           for n, k, p in self.edges],
                    darts=[dict(x=x, width=w, depth=d) for x, w, d in self.darts],
                    lines=[dict(name=n, points=np.asarray(p).round(4).tolist()) for n, p in self.lines])


H_UNIT = [1.0]


def _tuck_lines(piece, H, depth_flounce):
    """The flounce line and the tucks' fold and stitch lines, each the hem's line set up the piece by its distance
    (so they run parallel to the hem, and round the skirt at one height above the floor), cut to the piece."""
    from shapely.geometry import LineString, Polygon
    hem = LineString(piece.edge('hem'))
    body = Polygon(piece.outline()).buffer(0)

    def up(d):
        # (the hem's line moved into the piece: of the two sides, the one that lies inside it; extended past the
        # edges first, so the cut reaches them)
        best = None
        for sgn in (1, -1):
            g = hem.offset_curve(sgn * d, join_style='round')
            if g.is_empty:
                continue
            if g.geom_type == 'MultiLineString':
                g = max(g.geoms, key=lambda q: q.length)
            c = np.array(g.coords)
            e0 = c[0] + (c[0] - c[1]) / np.linalg.norm(c[0] - c[1]) * 20
            e1 = c[-1] + (c[-1] - c[-2]) / np.linalg.norm(c[-1] - c[-2]) * 20
            g = LineString(np.vstack([e0, c, e1])).intersection(body)
            if g.geom_type == 'MultiLineString':
                g = max(g.geoms, key=lambda q: q.length)
            if g.is_empty or g.geom_type != 'LineString':
                continue
            if best is None or g.length > best.length:
                best = g
        return None if best is None else np.array(best.coords)
    lines = [('flounce line', depth_flounce)]
    for k in range(TUCKS):
        d = depth_flounce + TUCK_GAP * H * (1.2 + k) + 2 * TUCK * H * k
        lines += [('tuck %d stitch' % (k + 1), d), ('tuck %d fold' % (k + 1), d + TUCK * H)]
    for name, d in lines:
        pts = up(d)
        if pts is not None:
            piece.lines.append((name, pts))


def draft_pieces(M, pat, log=print):
    """Every piece, drafted flat from the measures. Returns (pieces, seams, measures)."""
    H = M['H']
    H_UNIT[0] = H
    m = measures(M, pat)
    a1, a2 = SEAM_AT
    L, D = m['length'], m['hip_depth']
    tuck_cloth = TUCKS * 2 * TUCK * H
    Df = draft.RATIOS['flounce'] * H
    pieces = {}

    # front gore: half of it, the centre front on the fold at x = 0, the side edge to +x
    f = m['front']
    sup = max(0.0, f['hips'] - f['waist'])
    dart = DART_SHARE * sup
    fg = Piece('front gore', 'cut 1 on the fold', 'centre front, straight grain', fold='centre front')
    cf = _straight(0.0, L[0.0] + tuck_cloth)
    side = _edge((f['waist'] + dart, 0.0), f['hips'], f['hem'], L[a1] + tuck_cloth, D[a1])
    hem = _hem(side[-1], cf[-1], _dir(side), _dir(cf))
    fg.add('waist', 'waist', [[0.0, 0.0], side[0]])
    fg.add('side', 'seam', side)
    fg.add('hem', 'hem', hem)
    fg.add('centre front', 'fold', cf[::-1])
    fg.darts.append((0.55 * f['waist'], dart, DART_DEPTH * D[0.0]))
    _tuck_lines(fg, H, Df)
    pieces['front gore'] = fg

    # side gore: its front edge straight grain at x = 0, the back edge to +x takes the flare
    sg_m = m['side']
    sup = max(0.0, sg_m['hips'] - sg_m['waist'])
    dart = DART_SHARE * sup
    sg = Piece('side gore', 'cut 2 (a pair)', 'front edge, straight grain')
    fr = _straight(0.0, L[a1] + tuck_cloth)
    bk = _edge((sg_m['waist'] + dart, 0.0), sg_m['hips'], sg_m['hem'], L[a2] + tuck_cloth, D[a2])
    hem = _hem(bk[-1], fr[-1], _dir(bk), _dir(fr))
    sg.add('waist', 'waist', [[0.0, 0.0], bk[0]])
    sg.add('back', 'seam', bk)
    sg.add('hem', 'hem', hem)
    sg.add('front', 'seam', fr[::-1])
    sg.darts.append((0.5 * (sg_m['waist'] + dart), dart, DART_DEPTH * D[a1]))
    _tuck_lines(sg, H, Df)
    pieces['side gore'] = sg

    # back breadth: the centre back straight grain at x = 0, the side edge to +x; its top GATHER_BACK times the back
    # of the band, gathered in; straight from the top to the hem (no dart: the gathers take the shaping)
    b = m['back']
    top_w = GATHER_BACK * b['waist']
    bb = Piece('back breadth', 'cut 2 (a pair)', 'centre back, straight grain')
    cb = _straight(0.0, L[math.pi] + tuck_cloth)
    hem_x = max(b['hem'], top_w)
    hem_pt_len = L[a2] + tuck_cloth
    dx = hem_x - top_w
    sd = np.array([[top_w, 0.0], [hem_x, -math.sqrt(max(hem_pt_len ** 2 - dx ** 2, 1e-6))]])
    sd = np.linspace(sd[0], sd[1], 40)
    hem = _hem(sd[-1], cb[-1], _dir(sd), _dir(cb))
    bb.add('waist', 'waist', [[0.0, 0.0], sd[0]])
    bb.add('side', 'seam', sd)
    bb.add('hem', 'hem', hem)
    plk = 0.9 * H
    bb.add('centre back', 'seam', cb[::-1])
    bb.lines.append(('placket end', np.array([[-0.3, -plk], [0.3, -plk]])))
    bb.lines.append(('gathers', np.array([[0.0, -0.15], [top_w, -0.15]])))
    _tuck_lines(bb, H, Df)
    pieces['back breadth'] = bb

    # waistband: a straight strip, folded double along its length; the lap at the centre back
    band_len = m['band'] + LAP * H
    bd = draft.RATIOS['band'] * H
    wb = Piece('waistband', 'cut 1', 'along its length, straight grain')
    wb.add('lower', 'waist', [[0.0, 0.0], [band_len, 0.0]])
    wb.add('end', 'seam', [[band_len, 0.0], [band_len, 2 * bd]])
    wb.add('upper', 'waist', [[band_len, 2 * bd], [0.0, 2 * bd]])
    wb.add('start', 'seam', [[0.0, 2 * bd], [0.0, 0.0]])
    wb.lines.append(('fold', np.array([[0.0, bd], [band_len, bd]])))
    x = LAP * H
    for k, w in (('back', m['back']['waist']), ('side', m['side']['waist']), ('front', 2 * m['front']['waist']),
                 ('side', m['side']['waist']), ('back', m['back']['waist'])):
        x += w
        wb.lines.append(('notch: end of %s' % k, np.array([[x, -0.1], [x, 0.3]])))
    pieces['waistband'] = wb

    # flounce strips: the flounce line's girth on the skirt, GATHER times, in FLOUNCE_STRIPS lengths
    def width_at(piece, y):
        out = piece.outline()
        xs = []
        for p, q in zip(out, np.roll(out, -1, 0)):
            if (p[1] - y) * (q[1] - y) <= 0 and p[1] != q[1]:
                xs.append(p[0] + (y - p[1]) * (q[0] - p[0]) / (q[1] - p[1]))
        return (max(xs) - min(xs)) if xs else 0.0
    fl_line = 0.0
    for k, n in (('front gore', 2), ('side gore', 2), ('back breadth', 2)):
        fl_line += n * _length(dict(pieces[k].lines)['flounce line'])
    strip_len = GATHER * fl_line / FLOUNCE_STRIPS
    depth = Df + HEADING * H
    fs = Piece('flounce strip', 'cut %d' % FLOUNCE_STRIPS, 'along its length, straight grain')
    fs.add('top', 'waist', [[0.0, 0.0], [strip_len, 0.0]])
    fs.add('end', 'seam', [[strip_len, 0.0], [strip_len, -depth]])
    fs.add('hem', 'hem', [[strip_len, -depth], [0.0, -depth]])
    fs.add('start', 'seam', [[0.0, -depth], [0.0, 0.0]])
    fs.lines.append(('gathering line', np.array([[0.0, -HEADING * H], [strip_len, -HEADING * H]])))
    pieces['flounce strip'] = fs

    seams = seam_pairs(pieces, m, fl_line)
    return pieces, seams, m


def seam_pairs(pieces, m, fl_line):
    """Every seam: (piece, edge) sewn to (piece, edge), its kind, and the lengths. 'equal' seams must match to the
    millimetre; 'gather' seams are in their ratio."""
    P = pieces

    def ln(p, e):
        return _length(P[p].edge(e))
    plk = 0.9 * m['H']
    back_top = P['back breadth'].edge('waist')
    out = [
        dict(a=('front gore', 'side'), b=('side gore', 'front'), kind='equal', note='left and right',
             la=ln('front gore', 'side'), lb=ln('side gore', 'front')),
        dict(a=('side gore', 'back'), b=('back breadth', 'side'), kind='equal', note='left and right',
             la=ln('side gore', 'back'), lb=ln('back breadth', 'side')),
        dict(a=('back breadth', 'centre back'), b=('back breadth', 'centre back'), kind='equal',
             note='left to right, below the placket (%.1f cm open)' % (plk * CM),
             la=ln('back breadth', 'centre back') - plk, lb=ln('back breadth', 'centre back') - plk),
    ]
    for name in ('front gore', 'side gore'):
        for x, w, d in P[name].darts:
            leg = math.hypot(0.5 * w, d)
            out.append(dict(a=(name, 'dart left leg'), b=(name, 'dart right leg'), kind='equal',
                            note='a %.1f cm dart, %.1f cm long' % (w * CM, d * CM), la=leg, lb=leg))
    fw = _length(P['front gore'].edge('waist')) - P['front gore'].darts[0][1]
    sw = _length(P['side gore'].edge('waist')) - P['side gore'].darts[0][1]
    out += [
        dict(a=('front gore', 'waist'), b=('waistband', 'lower: front'), kind='equal', note='darts closed; both halves',
             la=2 * fw, lb=2 * m['front']['waist']),
        dict(a=('side gore', 'waist'), b=('waistband', 'lower: sides'), kind='equal', note='dart closed; each side',
             la=sw, lb=m['side']['waist']),
        dict(a=('back breadth', 'waist'), b=('waistband', 'lower: back'), kind='gather', note='each side, gathered',
             la=_length(back_top), lb=m['back']['waist']),
        dict(a=('flounce strip', 'end'), b=('flounce strip', 'start'), kind='equal', note='%d seams, a ring' % FLOUNCE_STRIPS,
             la=ln('flounce strip', 'end'), lb=ln('flounce strip', 'start')),
        dict(a=('flounce strip', 'top'), b=('skirt', 'flounce line'), kind='gather', note='the strips joined, round the skirt',
             la=FLOUNCE_STRIPS * ln('flounce strip', 'top'), lb=fl_line),
    ]
    for s in out:
        s['ratio'] = s['la'] / max(s['lb'], 1e-9)
        s['ok'] = abs(s['la'] - s['lb']) * CM < 0.1 if s['kind'] == 'equal' else s['ratio'] > 1.0
    return out


def report(pieces, seams, m, log=print):
    log('petticoat pattern pieces (cm):')
    for p in pieces.values():
        o = p.outline()
        log('  %-14s %-18s %5.1f wide, %5.1f long, grain: %s' % (p.name, p.cut, np.ptp(o[:, 0]) * CM, np.ptp(o[:, 1]) * CM, p.grain))
    log('petticoat seams:')
    for s in seams:
        log('  %-30s to %-30s %-6s %6.1f / %6.1f cm  x%.2f  %s  (%s)' % (
            ' '.join(s['a']), ' '.join(s['b']), s['kind'], s['la'] * CM, s['lb'] * CM, s['ratio'],
            'ok' if s['ok'] else 'MISMATCH', s['note']))
    log('petticoat measures: band %.1f cm, back of band %.1f, lengths CF %.1f side %.1f back side %.1f CB %.1f cm' % (
        m['band'] * CM, m['back_band'] * CM, m['length'][0.0] * CM, m['length'][SEAM_AT[0]] * CM,
        m['length'][SEAM_AT[1]] * CM, m['length'][math.pi] * CM))


def save(pieces, seams, m, path):
    data = dict(units='model units (1 unit = %.3f cm)' % CM, H=m['H'],
                pieces=[p.to_dict() for p in pieces.values()],
                seams=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()} for s in seams])
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=1, default=float)


def layout(pieces, seams, m, path):
    """The pattern laid out flat, in centimetres: seam lines solid, cutting lines dashed, grain arrows, darts, the
    flounce and tuck lines, notches, and each seam edge labelled with what it is sewn to."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = ['front gore', 'side gore', 'back breadth', 'flounce strip', 'waistband']
    tall = [pieces[k] for k in order[:3]]
    fig = plt.figure(figsize=(17, 12))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
    x0 = 0.0
    place = {}
    for p in tall:
        o = p.outline()
        place[p.name] = (x0 - o[:, 0].min() + 4, 0.0)
        x0 += np.ptp(o[:, 0]) * CM + 14
    ymin = min(p.outline()[:, 1].min() for p in tall) * CM
    place['flounce strip'] = (4.0, ymin - 45)
    place['waistband'] = (4.0, ymin - 100)
    pair_of = {}
    for s in seams:
        pair_of.setdefault(s['a'], s['b'])
        pair_of.setdefault(s['b'], s['a'])
    for p in pieces.values():
        dx, dy = place[p.name]
        T = lambda P: np.asarray(P) * CM + [dx * (1 if p.name in ('flounce strip', 'waistband') else 1), dy]  # noqa
        if p.name not in ('flounce strip', 'waistband'):
            T = lambda P, dx=dx: np.asarray(P) * CM + [dx, 0.0]  # noqa
        o = T(p.outline())
        ax.fill(o[:, 0], o[:, 1], color='#f3efe4', zorder=0)
        c = T(p.cut_line())
        ax.plot(c[:, 0], c[:, 1], '--', color='#8a8070', lw=0.8)
        for name, kind, e in p.edges:
            e = T(e)
            ax.plot(e[:, 0], e[:, 1], '-', color='#2a2a2a' if kind != 'fold' else '#3a6ea5', lw=1.4)
            mid = e[len(e) // 2]
            other = pair_of.get((p.name, name))
            lab = name + (' (fold)' if kind == 'fold' else '') + ('\n= ' + ' '.join(other) if other else '')
            ax.text(mid[0], mid[1], lab, fontsize=6.5, color='#444', ha='center', va='center',
                    bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
        for x, w, d in p.darts:
            pts = T([[x - w / 2, 0], [x, -d], [x + w / 2, 0]])
            ax.plot(pts[:, 0], pts[:, 1], '-', color='#a33', lw=1.0)
        for name, ln in p.lines:
            ln = T(ln)
            style = ':' if 'fold' in name else '-.'
            col = '#b07a2a' if 'tuck' in name else ('#2a7a4a' if 'flounce' in name or 'gather' in name else '#666')
            ax.plot(ln[:, 0], ln[:, 1], style, color=col, lw=0.8)
            if not name.startswith(('notch', 'tuck 2', 'tuck 3')) and 'fold' not in name:
                lab = 'tucks x%d (stitch and fold lines)' % TUCKS if name.startswith('tuck') else name
                ax.text(ln[-1, 0] + 1, ln[-1, 1], lab, fontsize=5.5, color=col, va='center')
        o2 = p.outline()
        gx = T([[o2[:, 0].min() + 0.35 * np.ptp(o2[:, 0]), o2[:, 1].max() - 0.25 * np.ptp(o2[:, 1])]])[0]
        if p.name in ('flounce strip', 'waistband'):
            ax.annotate('', xy=(gx[0] + 30, gx[1]), xytext=(gx[0], gx[1]), arrowprops=dict(arrowstyle='<->', color='#3a6ea5'))
        else:
            ax.annotate('', xy=(gx[0], gx[1] - 40), xytext=(gx[0], gx[1]), arrowprops=dict(arrowstyle='<->', color='#3a6ea5'))
        ax.text(gx[0] + 1.5, gx[1] - 3, 'grain', fontsize=6, color='#3a6ea5')
        tl = T([[o2[:, 0].mean(), o2[:, 1].max() - 0.12 * np.ptp(o2[:, 1])]])[0]
        ax.text(tl[0], tl[1], '%s\n%s' % (p.name.upper(), p.cut), fontsize=9, ha='center', weight='bold', color='#222')
    ax.set_aspect('equal')
    ax.grid(True, color='#e4e0d6', lw=0.4)
    ax.set_xlabel('cm')
    bad = [s for s in seams if not s['ok']]
    ax.set_title('Petticoat, c.1890: flat pattern drafted from the corseted mannequin (H %.2f units = %.1f cm). '
                 'Seam lines solid, cutting lines dashed (allowances 1.5 cm seams, 1 cm band, 2.5 cm hem). %d seams paired, %s.'
                 % (m['H'], m['H'] * CM, len(seams), 'all lengths match' if not bad else '%d MISMATCHED' % len(bad)),
                 fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    d = np.load(sys.argv[1])
    M = draft.measure(d['P'], d['W'], list(d['names']), float(d['H']), extra=d['extra'])
    draft.report(M)
    pat = draft.pattern(M)
    pieces, seams, m = draft_pieces(M, pat)
    report(pieces, seams, m)
    if len(sys.argv) > 2:
        layout(pieces, seams, m, sys.argv[2] + '.png')
        save(pieces, seams, m, sys.argv[2] + '.json')
