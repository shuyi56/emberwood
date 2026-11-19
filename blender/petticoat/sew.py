"""Cutting and sewing the petticoat: the flat pieces of pattern.py cut as cloth, laid round the body, sewn along their
paired edges in the solver, and draped.

  cut     each piece (the front gore unfolded, the side gores and back breadths as left and right pairs) is meshed flat
          inside its seam line, with the darts cut out of it and every marked line (the flounce line, the tucks'
          stitch, fold and upper lines) as a line of the mesh. Each edge that is sewn has its points spaced so that
          it and its partner have the same number, point for point down the seam (a gathered edge keeps its own, and
          is sewn in proportion).
  lay     the pieces are rolled, flat, onto a cone round the body (a cone is a flat sheet rolled: no cloth is
          stretched), each in its place round the waist; what would reach below the floor at the back is laid out
          along it, the angle drawn in so the cloth is not widened.
  sew     the paired points are tied by the solver's sewing threads, which pull to nothing: the side seams, the
          centre back below the placket, the dart legs, the tucks (each stitch line to its upper line, folding the
          cloth at the fold line between), and the skirt's top to the band (flat over the front and sides, gathered at
          the back). The band is held where it sits on the corset. After the drape the tied points are joined into
          one cloth.
  flounce the three strips, cut flat and joined in a ring, are set on the skirt's flounce line after the skirt has
          settled (the layer above, on the layer below): the strip's top is laid along the line in even pleats (a
          pleated strip is still unstretched), held there as sewn, and the rest let fall over the skirt.

Every count is in model units; the cloth's sampling is SPACING."""
import math
import os
import time

import numpy as np

import draft
import pattern
import fabric as xf

SPACING = 0.42               # the cloth's points, in units (about 1.1 cm)
FLOUNCE_SPACING = 0.32
LIFT = 0.25                  # the cone stood off the body's widest, in units, so no piece starts inside it


# ---------------------------------------------------------------- cut

def _resample(P, ts):
    """Points along polyline P at arc lengths ts (from its start)."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.empty((len(ts), 2))
    for c in range(2):
        out[:, c] = np.interp(ts, cum, P[:, c])
    return out


def _len(P):
    return float(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1).sum())


def _cross_params(edge, lines):
    """Where the marked lines meet an edge: arc lengths along it."""
    from shapely.geometry import LineString, Point
    E = LineString(edge)
    out = []
    for _, ln in lines:
        q = LineString(ln)
        for end in (ln[0], ln[-1]):
            if E.distance(Point(end)) < 0.05:
                out.append(E.project(Point(end)))
        del q
    return out


class Cut:
    """A piece as cut, in its flat coordinates: boundary edges (name -> point indices, top to bottom or along), the
    marked lines (name -> indices), and the triangles."""

    def __init__(self, name, side, V, tris, edges, lines, grain_x):
        self.name, self.side, self.V, self.tris, self.edges, self.lines = name, side, V, tris, edges, lines
        self.grain_x = grain_x


def _full_outline(piece):
    """A pattern piece as one whole flat shape: (named edges in order round it, each its points; and the lines).
    The front gore is unfolded about its centre front; darts are cut out of the waist as V notches."""
    edges = [(n, k, np.asarray(p, float)) for n, k, p in piece.edges]
    lines = [(n, np.asarray(p, float)) for n, p in piece.lines if not n.startswith('notch')]
    darts = list(piece.darts)
    if piece.fold:
        # (the half drawn has: waist (0 -> side top), side (down), hem (to CF), centre front (up). Unfold about x = 0.)
        E = {n: p for n, _, p in edges}
        side, hem, waist = E['side'], E['hem'], E['waist']
        m = np.array([-1.0, 1.0])
        edges = [('waist', 'waist', np.vstack([(waist * m)[::-1], waist[1:]])),
                 ('side L', 'seam', side),
                 ('hem', 'hem', np.vstack([hem, (hem * m)[::-1][1:]])),
                 ('side R', 'seam', (side * m)[::-1])]
        # (the lines drawn again on the whole piece, each the whole hem's line set up it: a half's lines mirrored
        # and joined at the fold can cross each other near it)
        whole = pattern.Piece(piece.name, 1, piece.grain)
        for n, k, p in edges:
            whole.add(n, k, p)
        pattern._tuck_lines(whole, pattern.H_UNIT[0], draft.RATIOS['flounce'] * pattern.H_UNIT[0])
        names = {n for n, _ in lines}
        lines = [(n, np.asarray(p, float)) for n, p in whole.lines if n in names]
        darts = [(-x, w, d) for x, w, d in darts] + darts
    return edges, lines, darts


# Edges sewn point for point are sampled together: the same count, at the same shares of their length (with the
# points where either's marked lines meet it), so a seam's points pair one to one.
SEWN_ALIKE = ([('front gore', 'side L'), ('front gore', 'side R'), ('side gore', 'front')],
              [('side gore', 'back'), ('back breadth', 'side')])


def _shared_params(pieces, spacing):
    out = {}
    full = {n: _full_outline(pieces[n]) for n in ('front gore', 'side gore', 'back breadth')}
    for group in SEWN_ALIKE:
        polys = []
        for pn, en in group:
            edges, lines, _ = full[pn]
            P = next(p for n, k, p in edges if n == en)
            down = P[0][1] < P[-1][1]          # (measured from the top whichever way the outline runs)
            polys.append((pn, en, P, lines, down))
        L = np.mean([_len(p[2]) for p in polys])
        fr = list(np.linspace(0, 1, max(2, int(math.ceil(L / spacing))) + 1))
        for pn, en, P, lines, down in polys:
            Lp = _len(P)
            for t in _cross_params(P, lines):
                fr.append(t / Lp if not down else 1.0 - t / Lp)
        fr = np.unique(np.round(np.sort(fr), 4))
        keep = [fr[0]]
        for f in fr[1:]:
            if (f - keep[-1]) * L > 0.12 * spacing:
                keep.append(f)
        keep[-1] = 1.0
        fr = np.array(keep)
        for pn, en, P, lines, down in polys:
            Lp = _len(P)
            out[(pn, en)] = (1.0 - fr[::-1]) * Lp if down else fr * Lp
    return out


def cut(pieces, spacing=SPACING):
    """Mesh every piece. Returns {(piece, side): Cut}."""
    import triangle
    out = {}
    shared = _shared_params(pieces, spacing)
    plan = [('front gore', 'C'), ('side gore', 'L'), ('side gore', 'R'), ('back breadth', 'L'), ('back breadth', 'R'),
            ('waistband', 'C')]
    for name, side in plan:
        piece = pieces[name]
        edges, lines, darts = _full_outline(piece)
        if side == 'R' and name != 'front gore':
            pass                     # (the right is the left's mirror, made when it is laid: same cloth, turned over)
        verts, segs, named = [], [], {}
        # the boundary, edge by edge; darts cut into the waist edge
        for en, kind, P in edges:
            L = _len(P)
            if (name, en) in shared:
                ts = shared[(name, en)]
                pts = _resample(P, ts)
                idx = list(range(len(verts), len(verts) + len(pts) - 1))
                verts += [tuple(p) for p in pts[:-1]]
                named[en] = idx + [None]
                continue
            ts = list(np.linspace(0.0, L, max(2, int(math.ceil(L / spacing))) + 1))
            ts += _cross_params(P, lines)
            ts = np.unique(np.round(np.sort(ts), 4))
            keep = [ts[0]]
            for t in ts[1:]:
                if t - keep[-1] > 0.12 * spacing:
                    keep.append(t)
                else:
                    keep[-1] = t if t in (ts[-1],) else keep[-1]
            ts = np.array(keep)
            ts[-1] = L
            pts = _resample(P, ts)
            if en == 'waist' and darts:
                pts = _cut_darts(pts, darts, spacing, named, len(verts))
            idx = list(range(len(verts), len(verts) + len(pts) - 1))
            verts += [tuple(p) for p in pts[:-1]]
            named[en] = idx + [None]             # (the last point is the next edge's first: filled below)
        n_b = len(verts)
        # close the edges' ends onto the next edge's first point
        order = [e[0] for e in edges]
        for k, en in enumerate(order):
            nxt = named[order[(k + 1) % len(order)]][0]
            named[en][-1] = nxt
        for i in range(n_b):
            segs.append((i, (i + 1) % n_b))
        # the marked lines, snapped to the boundary at their ends
        bv = np.array(verts)
        line_idx = {}
        for ln_name, P in lines:
            L = _len(P)
            ts = np.linspace(0.0, L, max(2, int(math.ceil(L / spacing))) + 1)
            pts = _resample(P, ts)
            ia = int(np.argmin(np.linalg.norm(bv - pts[0], axis=1)))
            ib = int(np.argmin(np.linalg.norm(bv - pts[-1], axis=1)))
            inner = list(range(len(verts), len(verts) + len(pts) - 2))
            verts += [tuple(p) for p in pts[1:-1]]
            chain = [ia] + inner + [ib]
            segs += list(zip(chain[:-1], chain[1:]))
            line_idx[ln_name] = chain
        tri = triangle.triangulate({'vertices': np.array(verts), 'segments': np.array(segs)},
                                   'pq28Ya%.4f' % (0.5 * spacing * spacing))
        V = tri['vertices']
        # (Y: no points added on the segments, so the boundary and lines keep exactly their points)
        out[(name, side)] = Cut(name, side, V, tri['triangles'], {k: v for k, v in named.items()}, line_idx, 0.0)
        if 'dart legs' in named:
            out[(name, side)].darts = named['dart legs']
    return out


def _cut_darts(pts, darts, spacing, named, base):
    """Cut V notches into a waist edge (points pts, along x): each dart's two legs down to its point and back.
    Records the legs' point indices (relative to base) for sewing."""
    out = []
    legs = []
    pts = np.asarray(pts)
    xs = pts[:, 0]
    i = 0
    for x, w, d in sorted(darts):
        a, b = x - 0.5 * w, x + 0.5 * w
        while i < len(pts) and xs[i] < a - 1e-9:
            out.append(pts[i])
            i += 1
        while i < len(pts) and xs[i] <= b + 1e-9:
            i += 1
        top_a, top_b = np.array([a, pts[0][1]]), np.array([b, pts[0][1]])
        apex = np.array([x, pts[0][1] - d])
        n = max(2, int(math.ceil(math.hypot(0.5 * w, d) / spacing)))
        left = [top_a + (apex - top_a) * k / n for k in range(n + 1)]
        right = [apex + (top_b - apex) * k / n for k in range(1, n + 1)]
        start = base + len(out)
        out += left + right
        # (the legs, top to point: left is start..start+n, right runs from the point back up)
        legs.append((list(range(start, start + n + 1)), list(range(start + 2 * n, start + n - 1, -1))))
    out += list(pts[i:])
    named.setdefault('dart legs', []).extend(legs)
    return np.array(out)


# ---------------------------------------------------------------- lay

PLEAT = 1.0                  # a gathered or darted top laid in pleats this long (in the cloth) before it is sewn


TUCK_LAYER = 0.05            # a layer of the cloth, in units, as the tucks' folds are laid


def hem_distance(c):
    """Each point's distance on the pattern from its piece's hem (the tucks' and the flounce's lines are the hem's
    line set up the piece by such a distance)."""
    from shapely.geometry import LineString, Point
    hem = LineString(c.V[[i for i in c.edges['hem'] if i is not None]])
    return np.array([hem.distance(Point(p)) for p in c.V])


def tuck_zones(H):
    """(stitch line's distance from the hem, the tuck's cloth) for each tuck, as pattern._tuck_lines draws them."""
    Df = draft.RATIOS['flounce'] * H
    t = pattern.TUCK * H
    return [(Df + pattern.TUCK_GAP * H * (1.2 + k) + 2 * t * k, 2 * t) for k in range(pattern.TUCKS)]


def _tucked(c, down, H):
    """The tucks laid as they are pressed: each folded at its fold line and its two lines brought together, the
    fold hanging down over the stitching, the outer layer off the skirt by two layers and the inner by one; the
    cloth below each tuck drawn up by the cloth it takes. Returns (each point's length down from the band as laid,
    and how far out)."""
    dh = hem_distance(c)
    d2 = down.copy()
    out = np.zeros(len(down))
    for d0, w in tuck_zones(H):
        below = dh < d0
        zone = (dh >= d0) & (dh <= d0 + w)
        v = d0 + w - dh                                    # down from the upper line, 0 .. w
        inner = zone & (v > 0.5 * w)
        d2[inner] -= 2 * v[inner] - w
        out[zone] += np.where(v[zone] <= 0.5 * w, 2.0, 1.0) * TUCK_LAYER
        d2[below] -= w
    return d2, out


def lay(cuts, pieces, m, M, pat, log=print):
    """Put each cut piece where it is worn, unstretched: every point at its own length down the pattern's line of
    cloth from the band (pattern.s), round between its piece's seam angles (SEAM_AT) in proportion across the piece
    at that height. Where the piece is wider than its place round the garment (the gathered back, the darts' opening
    at the top) the spare is laid in radial pleats, each PLEAT long, as deep as it takes for the cloth to keep its own
    width: the gathering thread drawn up before the top is sewn. What lies below the floor at the back turns out along
    it, its angle drawn in toward the centre back so the cloth is not widened. Returns {key: points [n, 3]}."""
    a1, a2 = pattern.SEAM_AT
    th, zr, R, sl = pat['th'], pat['z'], pat['r'], pat['s']
    nth = len(th)
    floor = M['floor'] + 0.25
    i_bot = int(np.argmin(np.abs(zr - pat['band_bot'])))
    s_bot = sl[i_bot]
    spans = {('front gore', 'C'): (-a1, a1, 1.0), ('side gore', 'L'): (a1, a2, 1.0),
             ('side gore', 'R'): (-a2, -a1, -1.0), ('back breadth', 'L'): (a2, math.pi, -1.0),
             ('back breadth', 'R'): (-math.pi, -a2, 1.0)}

    def col(t):
        return (np.round((np.asarray(t) % (2 * math.pi)) / (2 * math.pi) * nth).astype(int)) % nth

    def at(t, down):
        """height and radius of the line of cloth at angle t, down (from the band's foot) along it"""
        j = col(t)
        z = np.empty(len(j))
        r = np.empty(len(j))
        for jj in np.unique(j):
            sel = j == jj
            ss = sl[i_bot:, jj] - s_bot[jj]
            z[sel] = np.interp(down[sel], ss, zr[i_bot:])
            r[sel] = np.interp(down[sel], ss, R[i_bot:, jj])
        return z, r

    def arc(lo, hi, z):
        """the garment's length round between angles lo and hi at height z"""
        i = int(np.argmin(np.abs(zr - z)))
        tt = np.linspace(lo, hi, 64)
        rr = np.interp(tt % (2 * math.pi), th, R[i], period=2 * math.pi)
        return float(np.sum(np.hypot(np.diff(rr), 0.5 * (rr[1:] + rr[:-1]) * np.diff(tt))))
    out = {}
    worst = 0.0
    for k, c in cuts.items():
        if k[0] == 'waistband':
            continue
        lo, hi, d = spans[k]
        x = c.V[:, 0] * d
        down = -c.V[:, 1]
        # the piece's width at each height: between its two long edges (the seams, the centre lines), each carried
        # on straight down past its end (the dart notches' opening is laid in the pleats)
        sides = []
        for en, idx in c.edges.items():
            if en in ('waist', 'hem', 'dart legs'):
                continue
            idx = [i for i in idx if i is not None]
            yy = down[idx]
            o = np.argsort(yy)
            sides.append((float(x[idx].mean()), yy[o], x[idx][o]))
        sides.sort(key=lambda e: e[0])
        (_, ya, xa), (_, yb_, xb) = sides[0], sides[-1]
        x0 = np.interp(down, ya, xa)
        x1 = np.interp(down, yb_, xb)
        f = np.clip((x - x0) / np.maximum(x1 - x0, 1e-6), 0.0, 1.0)
        t = lo + f * (hi - lo)
        down_t, out_t = _tucked(c, down, M['H'])
        z, r = at(t, down_t)
        # the spare round at each height, in pleats
        k_ = np.ones(len(down))
        for y in np.unique(np.round(down, 1)):
            sel = np.abs(down - y) < 0.06
            zz = float(np.median(z[sel]))
            L = arc(lo, hi, max(zz, floor))
            k_[sel] = float(np.median(x1[sel] - x0[sel])) / max(L, 1e-6)
        # (narrower than its place: drawn in to fit, as a gore hangs inside the pattern's widest flare)
        fit = np.minimum(k_, 1.0)
        r = r * fit
        k_ = np.maximum(k_, 1.0)
        lay.debug[k] = (k_.copy(), fit.copy(), z.copy())
        worst = max(worst, float(k_.max()))
        amp = 0.5 * PLEAT * np.sqrt(np.maximum(0.0, 1.0 - 1.0 / k_ ** 2))
        u = (x - x0) / PLEAT
        tri = 1.0 - np.abs(2.0 * (u - np.floor(u)) - 1.0)     # 0 at a fold inside, 1 at a fold outside
        r = r + LIFT + amp * tri + out_t
        # the train: below the floor the cloth turns out along it
        down = down_t
        below = z < floor
        if below.any():
            zf, rf = at(t[below], np.array([float(np.interp(floor, zr[::-1], (sl[:, jj] - s_bot[jj])[::-1]))
                                            for jj in col(t[below])]))
            extra = np.array([float(np.interp(floor, zr[::-1], (sl[:, jj] - s_bot[jj])[::-1])) for jj in col(t[below])])
            extra = down[below] - extra
            rf = rf * fit[below] + LIFT
            R_ = rf + extra
            dt = (t[below] - math.pi + math.pi) % (2 * math.pi) - math.pi
            t[below] = math.pi + dt * rf / R_
            r[below] = R_
            z[below] = floor
        out[k] = np.stack([r * np.sin(t), -r * np.cos(t), z], 1)
    log('petticoat lay: each piece at its place on the pattern, the spare round in pleats (up to %.2f times the '
        'place round)' % worst)
    return out


lay.debug = {}


def band_ring(cut_band, m, M, pat):
    """The waistband laid where it is worn: round the corset at the waist, its lap over the centre back. Its points'
    angle from the centre back by their place along the band, their radius the band's own (pattern row 0)."""
    V = cut_band.V
    L = m['band']
    lap = pattern.LAP * M['H']
    bd = draft.RATIOS['band'] * M['H']
    th = pat['th']
    # (along the band from its start: the lap, then the girth from the centre back round by her right)
    s = V[:, 0] - lap
    ang = math.pi + 2 * math.pi * s / L
    r = np.interp(ang % (2 * math.pi), th, pat['r'][0], period=2 * math.pi) + GAP + 0.03 + 0.04 * (s < 0)
    # (the strip folded double: its lower half on the outside, its upper half folded down inside, so the band is
    # bd deep with the fold at the top)
    up = V[:, 1]
    zz = np.where(up <= bd, pat['band_bot'] + up, pat['band_bot'] + 2 * bd - up)
    r = r - 0.05 * (up > bd)
    return np.stack([r * np.sin(ang), -r * np.cos(ang), zz], 1), ang


# ---------------------------------------------------------------- sew

def _pairs_equal(a, b):
    """Points of two edges sewn point for point (both top to bottom, the same count)."""
    n = min(len(a), len(b))
    if len(a) != len(b):
        ia = np.round(np.linspace(0, len(a) - 1, max(len(a), len(b)))).astype(int)
        ib = np.round(np.linspace(0, len(b) - 1, max(len(a), len(b)))).astype(int)
        return list(zip(np.asarray(a)[ia], np.asarray(b)[ib]))
    return list(zip(a[:n], b[:n]))


def _pairs_gather(long, short):
    """A longer edge gathered onto a shorter one: each point of the long edge to the short edge's point at its
    share of the way along."""
    long, short = np.asarray(long), np.asarray(short)
    j = np.round(np.linspace(0, len(short) - 1, len(long))).astype(int)
    return list(zip(long, short[j]))


def _down(c, en):
    """An edge's points ordered top to bottom (by their height in the flat piece)."""
    idx = [i for i in c.edges[en] if i is not None]
    return idx if c.V[idx[0], 1] >= c.V[idx[-1], 1] else idx[::-1]


def assemble(cuts, placed, band_pts, band_ang, m, M):
    """One cloth of every piece: points, triangles, the pairs to sew, the band's points (held), and per point its
    piece, its flat (pattern) coordinates and its piece's grain. Returns a dict."""
    keys = [k for k in cuts if k[0] != 'waistband'] + [('waistband', 'C')]
    off, P, T, UV, piece_of = {}, [], [], [], []
    n = 0
    for k in keys:
        c = cuts[k]
        off[k] = n
        P.append(placed[k] if k[0] != 'waistband' else band_pts)
        tri = c.tris
        if k[1] == 'R' or (k[0] == 'back breadth' and k[1] == 'L'):
            tri = tri[:, ::-1]           # (a mirrored piece: its faces turned so the cloth's face is outward)
        T.append(tri + n)
        UV.append(c.V)
        piece_of += [k] * len(c.V)
        n += len(c.V)
    P = np.concatenate(P)
    T = np.concatenate(T)
    UV = np.concatenate(UV)
    g = lambda k, en: [off[k] + i for i in _down(cuts[k], en)]  # noqa: E731
    sew = []
    # side seams
    for s in ('L', 'R'):
        sew += _pairs_equal(g(('front gore', 'C'), 'side ' + s), g(('side gore', s), 'front'))
        sew += _pairs_equal(g(('side gore', s), 'back'), g(('back breadth', s), 'side'))
    # centre back, below the placket
    plk = 0.9 * M['H']
    a = g(('back breadth', 'L'), 'centre back')
    b = g(('back breadth', 'R'), 'centre back')
    cL = cuts[('back breadth', 'L')]
    a_ok = [i for i in a if -cL.V[i - off[('back breadth', 'L')], 1] > plk]
    b_ok = [i for i in b if -cuts[('back breadth', 'R')].V[i - off[('back breadth', 'R')], 1] > plk]
    sew += _pairs_equal(a_ok, b_ok)
    # darts
    for k in (('front gore', 'C'), ('side gore', 'L'), ('side gore', 'R')):
        for la, lb in getattr(cuts[k], 'darts', []):
            sew += _pairs_equal([off[k] + i for i in la], [off[k] + i for i in lb])
    # tucks: each stitch line to its upper line, point for point
    for k in keys[:-1]:
        c = cuts[k]
        for t in range(pattern.TUCKS):
            lo, up = c.lines.get('tuck %d stitch' % (t + 1)), c.lines.get('tuck %d upper' % (t + 1))
            if lo and up:
                sew += _pairs_equal([off[k] + i for i in lo], [off[k] + i for i in up])
    # the skirt's tops to the band's lower edge, by angle: each piece's top to the stretch of band over its place
    kb = ('waistband', 'C')
    cb_ = cuts[kb]
    # (the band's lower edge round her once: not its lap under the far end at the centre back)
    lap = pattern.LAP * M['H']
    low = [off[kb] + i for i in range(len(cb_.V)) if cb_.V[i, 1] < 1e-6 and cb_.V[i, 0] >= lap - 1e-6]
    low_ang = np.array([band_ang[i - off[kb]] for i in low])
    low = np.array(low)[np.argsort(low_ang)]
    low_ang = np.sort(low_ang)
    a1, a2 = pattern.SEAM_AT
    spans = {('front gore', 'C'): (-a1, a1), ('side gore', 'L'): (a1, a2), ('side gore', 'R'): (-a2, -a1),
             ('back breadth', 'L'): (a2, math.pi), ('back breadth', 'R'): (-math.pi, -a2)}
    for k, (lo, hi) in spans.items():
        top = [i for i in cuts[k].edges['waist'] if i is not None]
        # (not the dart legs' points, and of each dart's two corners at the top only the first: the dart closed,
        # they are one point of the top)
        shut = {lb[0] for la, lb in getattr(cuts[k], 'darts', [])}
        top = [off[k] + i for i in top if abs(cuts[k].V[i, 1]) < 1e-6 and i not in shut]
        top = sorted(top, key=lambda i: math.atan2(P[i, 0], -P[i, 1]))
        if k[0] == 'back breadth' and k[1] == 'R':
            top = sorted(top, key=lambda i: math.atan2(P[i, 0], -P[i, 1]) % (2 * math.pi))
        aw = np.arctan2(np.sin(low_ang), np.cos(low_ang))
        if hi > math.pi - 1e-6:
            sel = (aw >= lo - 1e-6) | (aw <= -math.pi + 1e-6)
        else:
            sel = (aw >= lo - 1e-6) & (aw <= hi + 1e-6)
        band_part = list(low[sel][np.argsort(aw[sel] % (2 * math.pi) if k[1] == 'R' and k[0] == 'back breadth' else aw[sel])])
        if len(band_part) < 2:
            continue
        sew += _pairs_gather(top, band_part) if len(top) >= len(band_part) else _pairs_equal(top, band_part)
    pinned = np.zeros(len(P))
    pinned[off[kb]:] = 1.0
    return dict(P=P, T=T, UV=UV, piece=piece_of, sew=np.array(sew, dtype=np.int64), pinned=pinned, off=off, H=M['H'])


def weld(P, T, sew):
    """Join the sewn points: each pair made one point (where several are sewn to one, all of them), the faces kept.
    Returns (points, faces, map old -> new)."""
    n = len(P)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in sew:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(n)])
    uniq, new = np.unique(root, return_inverse=True)
    Q = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(Q, new, P)
    np.add.at(cnt, new, 1)
    Q /= cnt[:, None]
    F = new[T]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return Q, F[ok], new


def weights(P, M):
    """The pelvis above the fullest hip; below it each side's thigh takes over toward the hem, a share of 0.45 at
    the floor, on the side the point lies (through the middle, both alike)."""
    down = draft._ss(M['hips_z'], M['floor'], P[:, 2])
    k = 0.45 * down
    side = draft._ss(-1.2, 1.2, P[:, 0])
    return {'Pelvis': 1.0 - k, 'Thigh.L': k * side, 'Thigh.R': k * (1.0 - side)}


# ---------------------------------------------------------------- drape (the local solver)

# The cotton in the local solver's terms (xpbd_fabric, fabric-core.js's model): threads that keep their length, a
# starched longcloth's bend (stiffer than the cape's 2e-4), its slide on the corset's coutil and on skin.
FABRIC = dict(walk='gauss-seidel', iterations=4, stretch=0.0, bend=2e-4, damping=0.03, friction=0.4)
# (the flattening stencils' compliance: on the swatch strip (8 cm, 0.42 spacing, 4 walks at 1/240 s) 1e-6 gives
# Peirce's bending length 3.7 cm, a starched longcloth; 1e-5 gives 2.6, 1e-4 1.7)
FLAT = 1e-6
STEP = 1.0 / 240             # s
SEW_SECONDS = 0.6            # the seams drawn up and the cloth let fall
GAP = 0.06                   # off the body, in units


def tethers(A, cuts):
    """Each skirt point held within its length of cloth from the band: tied to the band point its piece's top is
    sewn to nearest above it on the pattern, at its flat distance from that top point (and 2% of give)."""
    UV, S, off = A['UV'], A['sew'], A['off']
    band = A['pinned'] > 0
    partner = {}
    for a, b in S:
        if band[b] and not band[a]:
            partner[int(a)] = int(b)
        elif band[a] and not band[b]:
            partner[int(b)] = int(a)
    I, J, R = [], [], []
    for k, c in cuts.items():
        if k[0] == 'waistband':
            continue
        o = off[k]
        tops = np.array([o + i for i in range(len(c.V)) if (o + i) in partner])
        if not len(tops):
            continue
        idx = np.arange(o, o + len(c.V))
        j = np.argmin(np.abs(UV[idx, 0][:, None] - UV[tops, 0][None, :]), axis=1)
        t = tops[j]
        I.append(idx)
        J.append(np.array([partner[int(q)] for q in t]))
        R.append(1.02 * np.linalg.norm(UV[idx] - UV[t], axis=1) + 0.05)
    return np.concatenate(I), np.concatenate(J), np.concatenate(R)


def drape(A, cuts, pts, nrm, floor, log=print):
    """The local solver's work, apart from Blender (petticoat.py's offline bench runs it on a saved body)."""
    T, S = A['T'], A['sew']
    S, extra, weld = gathers(A)
    S = one_hold(S, A['pinned'] > 0, len(A['P']))
    n0 = len(A['P'])
    fab = xf.Fabric(np.vstack([A['P'], extra]), FABRIC)
    fab.pin(np.nonzero(A['pinned'] > 0)[0])
    fab.pin(np.arange(n0, n0 + len(extra)))
    soft = np.zeros(len(A['P']))
    for k, c in cuts.items():
        if k[0] == 'waistband':
            continue
        dh = hem_distance(c)
        z = np.zeros(len(c.V), bool)
        for d0, w in tuck_zones(A['H']):
            z |= (dh >= d0 - 0.15) & (dh <= d0 + w + 0.15)
        soft[A['off'][k]:A['off'][k] + len(c.V)][z] = TUCK_BEND
    fab.mesh(T, rest_pos=A['UV'], bend_at=soft, flat=FLAT)
    fab.sew(S[:, 0], S[:, 1], loose=1e-3, firm=0.0, seconds=0.25)
    I, J, R = tethers(A, cuts)
    fab.tether(I, J, R)
    body_ = xf.MeshBody(pts, nrm, gap=GAP, floor=floor)
    t0 = time.time()
    n = int(round(SEW_SECONDS / STEP))
    for k in range(n):
        fab.step(STEP, body_)
        if (k + 1) % 24 == 0:
            g = np.linalg.norm(fab.pos[S[:, 0]] - fab.pos[S[:, 1]], axis=1)
            log('petticoat fabric: %.2f s (%.0f s), seams apart %.3f on average, %.3f at most' % (
                fab.t, time.time() - t0, float(g.mean()), float(g.max())))
    k = fab.settle(body_, h=STEP, max_steps=720, tol=2e-4, log=log)
    x1 = fab.pos.copy()
    fab.step(STEP, body_, neutral=True)
    mv = np.linalg.norm(fab.pos - x1, axis=1)
    fab.pos[:], fab.prev[:] = x1, x1
    bad = np.argsort(-mv)[:5]
    log('petticoat fabric: still moving: ' + ', '.join('%d at %s by %.3f' % (i, np.round(x1[i], 1), mv[i]) for i in bad))
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    E = np.unique(E, axis=0)
    st = fab.strain(E[:, 0], E[:, 1], np.linalg.norm(A['UV'][E[:, 0]] - A['UV'][E[:, 1]], axis=1))
    g = np.linalg.norm(fab.pos[S[:, 0]] - fab.pos[S[:, 1]], axis=1)
    log('petticoat fabric: settled in %d more steps (%.0f s in all): threads %.1f%% long at the 99th percentile, '
        '%.1f%% at most; seams apart %.3f at most' % (k, time.time() - t0, 100 * (np.percentile(st, 99) - 1),
                                                     100 * (st.max() - 1), float(g.max())))
    if os.environ.get('PETTI_DRAPE_OUT'):
        np.save(os.environ['PETTI_DRAPE_OUT'], fab.pos[:n0])
    return fab.pos[:n0].copy(), weld


TUCK_BEND = 5e-3             # the bend ties across a pressed tuck's folds: the press holds it, not the cloth


def one_hold(S, fixed, n0):
    """Where points sewn together are each sewn to a different held point (a dart's two legs both to the band),
    only the first of those holds is kept: the rest could only pull the seam apart."""
    parent = {}

    def find(i):
        while parent.get(i, i) != i:
            i = parent[i]
        return i
    isfix = lambda i: i >= n0 or fixed[i]  # noqa: E731
    for a, b in S:
        if not isfix(a) and not isfix(b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    held = {}
    out = []
    for a, b in S:
        if isfix(b) != isfix(a):
            free, fx = (a, b) if isfix(b) else (b, a)
            r = find(free)
            if r in held and held[r] != fx:
                continue
            held[r] = fx
        out.append((a, b))
    return np.array(out, np.int64)


def gathers(A):
    """The skirt's top sewn to the band: where several of its points go to one point of the band (the gathered
    back), each is sewn instead to its own place along the band's edge from that point toward the next (a gathered
    edge is drawn up evenly along the band, its cloth standing in small folds between), those places held where the
    band is. Returns (pairs for the solver, the places, pairs to join after: the gathered points other than the
    first stay where they are sewn, on the band's edge)."""
    P, UV, S = A['P'], A['UV'], A['sew']
    band = A['pinned'] > 0
    n0 = len(P)
    groups = {}
    for a, b in S:
        if band[b] and not band[a]:
            groups.setdefault(int(b), []).append(int(a))
    low = np.nonzero(band & (UV[:, 1] < 1e-6))[0]
    ang = np.arctan2(P[low, 0], -P[low, 1])
    low = low[np.argsort(ang)]
    ang = np.sort(ang)
    nxt = {int(low[i]): int(low[(i + 1) % len(low)]) for i in range(len(low))}
    out, extra, weld = [], [], []
    for a, b in S:
        if band[b] and not band[a] and len(groups[int(b)]) > 1:
            continue
        out.append((a, b))
        weld.append((a, b))
    for b, part in groups.items():
        if len(part) < 2:
            continue
        part = sorted(part, key=lambda i: math.atan2(P[i, 0], -P[i, 1]))
        d = P[nxt[b]] - P[b]
        if np.linalg.norm(d) > 1.5:          # (the band's end at the centre back: toward the point before instead)
            d = -d
        k = len(part)
        weld.append((part[0], b))
        for j, a in enumerate(part):
            if j == 0:
                out.append((a, b))
                continue
            extra.append(P[b] + d * (j / k))
            out.append((a, n0 + len(extra) - 1))
    return np.array(out, np.int64), np.array(extra).reshape(-1, 3), np.array(weld, np.int64)
