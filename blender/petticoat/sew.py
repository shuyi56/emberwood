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
        # (the waist's own points kept off the dart's corners by a third of the spacing, so no sliver of cloth is
        # left beside a corner)
        while i < len(pts) and xs[i] < a - 0.35 * spacing:
            out.append(pts[i])
            i += 1
        while i < len(pts) and xs[i] <= b + 0.35 * spacing:
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
    lp = P[low]
    la_ = np.concatenate([low_ang - 2 * math.pi, low_ang, low_ang + 2 * math.pi])
    lpp = np.vstack([lp, lp, lp])

    def ring_at(t):
        """the band's lower edge at angle t (the points between its own by angle)"""
        t = (t + math.pi) % (2 * math.pi) - math.pi
        return np.array([np.interp(t, la_, lpp[:, a]) for a in range(3)])
    targets = []
    for k, (lo, hi) in spans.items():
        top = [i for i in cuts[k].edges['waist'] if i is not None]
        # (not the dart legs' points, and of each dart's two corners at the top only the first: the dart closed,
        # they are one point of the top)
        shut = {lb[0] for la, lb in getattr(cuts[k], 'darts', [])}
        top = [off[k] + i for i in top if abs(cuts[k].V[i, 1]) < 1e-6 and i not in shut]
        top = sorted(top, key=lambda i: math.atan2(P[i, 0], -P[i, 1]))
        if k[0] == 'back breadth' and k[1] == 'R':
            top = sorted(top, key=lambda i: math.atan2(P[i, 0], -P[i, 1]) % (2 * math.pi))
        # each top point to its own place along the band's edge over the piece's span, in proportion to its
        # length along the top with the darts closed: flat over the front and sides, gathered evenly at the back
        c = cuts[k]
        corner = {la[0]: lb[0] for la, lb in getattr(c, 'darts', [])}
        u = [0.0]
        for p_, q_ in zip(top[:-1], top[1:]):
            i_, j_ = p_ - off[k], q_ - off[k]
            # (across a dart: from its corner on the far side, the dart shut, whichever way the top runs)
            if i_ in corner and np.linalg.norm(c.V[j_] - c.V[corner[i_]]) < np.linalg.norm(c.V[j_] - c.V[i_]):
                i_ = corner[i_]
            elif j_ in corner and np.linalg.norm(c.V[i_] - c.V[corner[j_]]) < np.linalg.norm(c.V[i_] - c.V[j_]):
                j_ = corner[j_]
            u.append(u[-1] + float(np.linalg.norm(c.V[j_] - c.V[i_])))
        u = np.array(u) / max(u[-1], 1e-9)
        th_ = lo + u * (hi - lo)
        for i_, t_ in zip(top, th_):
            targets.append((i_, ring_at(t_)))
    pinned = np.zeros(len(P))
    pinned[off[kb]:] = 1.0
    return dict(P=P, T=T, UV=UV, piece=piece_of, sew=np.array(sew, dtype=np.int64), pinned=pinned, off=off, H=M['H'],
                to_band=(np.array([t[0] for t in targets], np.int64), np.array([t[1] for t in targets]).reshape(-1, 3)),
                band_low=low)


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
    """Each skirt point held within its length of cloth from the band: tied to the place on the band that its
    piece's top point nearest above it on the pattern is sewn to (numbered after the cloth's points, as gathers()
    numbers them), at its flat distance from that top point (and 2% of give)."""
    UV, off = A['UV'], A['off']
    n0 = len(A['P'])
    partner = {int(i): n0 + k for k, i in enumerate(A['to_band'][0])}
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
        # (round a dart's legs the cloth is shaped into a cone when the dart is sewn: the starch gives there)
        for la, lb in getattr(c, 'darts', []):
            from shapely.geometry import LineString, Point
            ln = LineString(np.vstack([c.V[la], c.V[lb][::-1]]))
            near = np.array([ln.distance(Point(p)) < DART_EASE for p in c.V])
            seg = soft[A['off'][k]:A['off'][k] + len(c.V)]
            seg[near] = np.maximum(seg[near], DART_BEND)
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


DART_BEND = 2e-4             # and round the darts, within DART_EASE of their legs, where the sewn dart shapes a cone
DART_EASE = 0.8
TUCK_BEND = 0                # the tuck zone is simulated flat: a 5 mm tuck is thinner than the cloth's points (1.1 cm), so
                             # folding it in the solver only throws spikes; its ridge and stitching are drawn as geometry


FLOUNCE_FLAT = None           # the flounce is lighter than the skirt and gathered, not starched to stand: it falls in
FLOUNCE_BEND = 2e-4           # many small folds from its gathers
FLOUNCE_PLEAT = 2.4           # its gathers laid in folds this long in the cloth (several points across, so they show)
FLOUNCE_ITERATIONS = 12       # gathered 1.6 times, its threads need the passes to stay their length and fold


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
    """The skirt's top sewn to the band: each top point to its own place on the band's edge (assemble's to_band),
    those places held where the band is. Returns (pairs for the solver: the seams and darts and tucks, then each top
    point to its place, numbered after the cloth's points; the places; the pairs to join after: the seams, and each
    band point with the top point whose place is nearest it, if near)."""
    P = A['P']
    n0 = len(P)
    idx, tg = A['to_band']
    S = [tuple(p) for p in A['sew']]
    out = S + [(int(i), n0 + k) for k, i in enumerate(idx)]
    weld = list(S)
    low = A['band_low']
    if len(idx):
        d = np.linalg.norm(tg[None, :, :] - P[low][:, None, :], axis=2)
        j = np.argmin(d, axis=1)
        sp = np.median(np.linalg.norm(np.diff(P[low], axis=0), axis=1))
        used = set()
        for b, jj in zip(low, j):
            if d[list(low).index(b), jj] < 0.35 * sp and jj not in used:
                weld.append((int(idx[jj]), int(b)))
                used.add(jj)
    return np.array(out, np.int64), tg, np.array(weld, np.int64)


# ---------------------------------------------------------------- the solved folds, kept

CACHE = os.environ.get('PETTI_CACHE', '/mnt/project-files/petticoat/cache')


def settled_key(*arrays, extra=''):
    """What the solved folds depend on: the shape the solver was given (its arrays), every solver and lay setting
    in this file, and the solver's own code (shared/fabric.py). A material or texture run changes none of these,
    so it finds its folds already solved."""
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes())
    sets = {k: v for k, v in sorted(globals().items())
            if k.isupper() and isinstance(v, (int, float, str, tuple, dict, type(None))) and k != 'CACHE'}
    h.update(repr(sets).encode())
    h.update(extra.encode())
    for f in (xf.__file__, __file__):
        with open(f, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()[:20]


def settled_load(kind, key, log=print):
    path = os.path.join(CACHE, '%s-%s.npz' % (kind, key))
    try:
        d = np.load(path, allow_pickle=False)
        out = {k: d[k] for k in d.files}
        log('petticoat cache: %s folds loaded (%s), the solver not run' % (kind, key))
        return out
    except (OSError, ValueError):
        return None


def settled_save(kind, key, log=print, **arrays):
    try:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(os.path.join(CACHE, '%s-%s.npz' % (kind, key)), **arrays)
        log('petticoat cache: %s folds saved (%s)' % (kind, key))
    except OSError as e:
        log('petticoat cache: not saved (%s)' % e)


# ---------------------------------------------------------------- the flounce (the layer above, on the settled skirt)

def flounce_ring(piece, spacing=FLOUNCE_SPACING):
    """The flounce strips as cut, joined end to start into a ring: a grid of points on the flat strips laid end to end
    (u along the ring, v down from the top edge), its triangles, and the gathering line's points. The grid is open at
    the centre front: its last column lies on its first and the two are sewn point for point (the ring's closing seam;
    the other joins between strips fall on grid columns and are meshed through)."""
    top = piece.edge('top')
    L = float(top[-1][0] - top[0][0])
    depth = float(-piece.edge('end')[-1][1])
    head = float(-dict(piece.lines)['gathering line'][0][1])
    G = pattern.FLOUNCE_STRIPS * L
    nu = int(math.ceil(G / spacing))
    nu += (-nu) % pattern.FLOUNCE_STRIPS
    us = np.arange(nu + 1) * (G / nu)
    vs = np.unique(np.concatenate([np.linspace(0.0, depth, int(math.ceil(depth / spacing)) + 1), [head]]))
    nc, nv = nu + 1, len(vs)
    UV = np.array([(u, -v) for v in vs for u in us])
    idx = lambda i, j: j * nc + i  # noqa: E731
    T = []
    for j in range(nv - 1):
        for i in range(nu):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            T += [(a, c, b), (b, c, d)] if (i + j) % 2 else [(a, c, d), (a, d, b)]
    jg = int(np.argmin(np.abs(vs - head)))
    close = np.array([(idx(nu, j), idx(0, j)) for j in range(nv)])
    return dict(UV=UV, T=np.array(T), gather=np.array([idx(i, jg) for i in range(nc)]), close=close,
                nu=nu, nc=nc, nv=nv, G=G, us=us, vs=vs, head=head, depth=depth,
                seams=[k * nu // pattern.FLOUNCE_STRIPS for k in range(pattern.FLOUNCE_STRIPS)])


def _outward_normals(X, T, piece):
    """A settled cloth's point normals by its faces' winding, each piece's turned outward as a whole."""
    X, T = np.asarray(X, float), np.asarray(T)
    nf = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    lab = np.asarray([hash(p) for p in piece])
    mid = X[T].mean(1)
    for p_ in np.unique(lab[T[:, 0]]):
        on = lab[T[:, 0]] == p_
        if np.einsum('ij,ij->i', nf[on, :2], mid[on, :2]).sum() < 0:
            nf[on] *= -1
    N = np.zeros_like(X)
    for c_ in range(3):
        np.add.at(N, T[:, c_], nf)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def cloth_surface(X, T, piece, n=3, smooth=1.2):
    """A settled cloth as a body for the layer above: points sampled across its faces (n to an edge, so a point
    slipping between its own points still meets it) with the faces' normals, each piece's turned outward as a whole
    (a fold's sides keep their true sense, which a point-by-point radial test gets wrong where the cloth faces round
    her)."""
    X, T = np.asarray(X, float), np.asarray(T)
    nf = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-12)
    lab = np.asarray([hash(p) for p in piece])[T[:, 0]]
    mid = X[T].mean(1)
    for p_ in np.unique(lab):
        on = lab == p_
        if np.einsum('ij,ij->i', nf[on, :2], mid[on, :2]).sum() < 0:
            nf[on] *= -1
    # (pushed out along the cloth's lie, not its folds: a fold's side would shove the layer above round her, where
    # it gathers into itself; the faces' normals averaged over a fold's width)
    from scipy.spatial import cKDTree
    tree = cKDTree(mid)
    sm = np.zeros_like(nf)
    for f_, nb in enumerate(tree.query_ball_point(mid, smooth)):
        sm[f_] = nf[nb].sum(0)
    nf = sm / np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-12)
    bc = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
    pts = np.einsum('kc,fcd->fkd', bc, X[T]).reshape(-1, 3)
    return pts, np.repeat(nf, len(bc), axis=0)


def flounce(A, cuts, X, pieces, M, pts, nrm, floor, log=print):
    """Set the flounce on the settled skirt X (the layer below, not moved again): its gathering line sewn evenly
    round the skirt's flounce line, the strips laid down over the skirt in pleats, then let fall on the skirt, her and
    the floor in the local solver. Returns (points, triangles, flat UVs, the ring)."""
    F = flounce_ring(pieces['flounce strip'])
    UV, T, nu, nc, nv = F['UV'], F['T'], F['nu'], F['nc'], F['nv']
    # the skirt's flounce line, round her from the centre front: each piece's stretch of it in its cut order, turned
    # to run round the same way, the pieces by angle; its length the cloth's (flat), so the gathers fall evenly in the
    # cloth however the settled skirt folds
    runs = []
    for k, c in cuts.items():
        if k[0] == 'waistband' or 'flounce line' not in c.lines:
            continue
        li = np.asarray(c.lines['flounce line'])
        g = A['off'][k] + li
        th = np.unwrap(np.arctan2(X[g, 0], -X[g, 1]))
        if th[-1] < th[0]:
            li, g, th = li[::-1], g[::-1], th[::-1]
        ds = np.linalg.norm(np.diff(c.V[li], axis=0), axis=1)
        runs.append((float(np.mean(th)), g, ds, th, k, li))
    runs.sort(key=lambda r: (r[0] + math.pi) % (2 * math.pi))
    g = np.concatenate([r[1] for r in runs])
    s = np.concatenate([[0.0], np.cumsum(np.concatenate([np.append(r[2], 0.0) for r in runs]))[:-1]])
    Ls = float(np.sum([r[2].sum() for r in runs]))
    # (the centre front: where the front gore's stretch crosses angle 0)
    fr = min(runs, key=lambda r: abs(r[0]))
    i0 = int(np.nonzero(np.isin(g, fr[1]))[0][0])
    th0 = fr[3] - 2 * math.pi * round(fr[0] / (2 * math.pi))
    s0 = s[i0] + float(np.interp(0.0, th0, np.concatenate([[0.0], np.cumsum(fr[2])])))
    qq = np.vstack([X[g], X[g[:1]]])
    ss = np.append(s, Ls)

    def on_line(frac):
        t = (np.asarray(frac) * Ls + s0) % Ls
        return np.stack([np.interp(t, ss, qq[:, a_]) for a_ in range(3)], -1)
    # the gathering line's points: evenly round the line (their share of the ring), from the centre front
    frac = F['us'] / F['G']
    tg = on_line(frac)
    # (the seam lies on the skirt's face: the flounce's side of it a gap out, where the body lets it be)
    rr = np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-9)
    tg[:, :2] *= ((rr + GAP + 0.02) / rr)[:, None]
    log('petticoat flounce: %d points round, %d down; the line round the skirt %.1f cm, the strips %.1f cm '
        '(gathered %.2f times)' % (nu, nv, Ls * pattern.CM, F['G'] * pattern.CM, F['G'] / Ls))
    # laid where it hangs: each column of the flounce down the skirt's own cloth from its place on the line (to the
    # line's nearest hem point, on the flat piece, then on along the floor), a gap out from the skirt's face, with
    # its spare round the ring laid in folds
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    from shapely.geometry import LineString, Point
    k_ = F['G'] / Ls
    amp = 0.5 * FLOUNCE_PLEAT * math.sqrt(max(0.0, 1.0 - 1.0 / k_ ** 2))
    nX = _outward_normals(X, A['T'], A['piece'])
    t_ = (frac * Ls + s0) % Ls
    run_at = np.searchsorted(np.cumsum([r[2].sum() for r in runs]), t_, side='right').clip(0, len(runs) - 1)
    starts = np.concatenate([[0.0], np.cumsum([r[2].sum() for r in runs])[:-1]])
    P = np.empty((nc * nv, 3))
    u = F['us'] / FLOUNCE_PLEAT
    fold = amp * (1.0 - np.abs(2.0 * (u - np.floor(u)) - 1.0))
    dzs = F['vs'] - F['head']                       # (down from the gathering line; the heading stands above it)
    for r_ in np.unique(run_at):
        _, g_, ds_, _, k, li = runs[r_]
        c = cuts[k]
        cols = np.nonzero(run_at == r_)[0]
        cs = np.concatenate([[0.0], np.cumsum(ds_)])
        tt = np.clip(t_[cols] - starts[r_], 0.0, cs[-1])
        p0 = np.stack([np.interp(tt, cs, c.V[li, a_]) for a_ in range(2)], -1)
        hem = LineString(c.V[[i for i in c.edges['hem'] if i is not None]])
        q0 = np.array([hem.interpolate(hem.project(Point(p))).coords[0] for p in p0])
        dn = q0 - p0
        Df_ = np.maximum(np.linalg.norm(dn, axis=1), 1e-9)
        dn /= Df_[:, None]
        tri_ = Triangulation(c.V[:, 0], c.V[:, 1], c.tris)
        o = A['off'][k] + np.arange(len(c.V))
        f3 = [LinearTriInterpolator(tri_, X[o, a_]) for a_ in range(3)]
        fn = [LinearTriInterpolator(tri_, nX[o, a_]) for a_ in range(3)]
        for j, dz in enumerate(dzs):
            d_ = np.minimum(dz, Df_ - 1e-3)
            fp = p0 + dn * d_[:, None]
            at = np.stack([np.ma.filled(f(fp[:, 0], fp[:, 1]), np.nan) for f in f3], -1)
            nn = np.stack([np.ma.filled(f(fp[:, 0], fp[:, 1]), np.nan) for f in fn], -1)
            bad = ~np.isfinite(at).all(1)
            if bad.any():                           # (off the piece's mesh: its nearest point)
                jn = np.argmin(np.linalg.norm(c.V[None, :, :] - fp[bad][:, None, :], axis=2), axis=1)
                at[bad], nn[bad] = X[o[jn]], nX[o[jn]]
            nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
            q = at + nn * (GAP + 0.08 + fold[cols] * min(1.0, 0.4 + abs(dz) / 3.0))[:, None]
            past = np.maximum(dz - Df_, 0.0)        # (below the hem: on out along the floor)
            rad = np.column_stack([q[:, 0], q[:, 1]])
            rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
            q[:, :2] += rad * past[:, None]
            q[:, 2] = np.maximum(q[:, 2], floor + 0.08)
            P[j * nc + cols] = q
    # the solver: the flounce's points, then the places on the line (held)
    n0 = len(P)
    fab = xf.Fabric(np.vstack([P, tg]), dict(FABRIC, iterations=FLOUNCE_ITERATIONS))
    fab.pin(np.arange(n0, n0 + nc))
    fab.mesh(T, rest_pos=np.column_stack([UV, np.zeros(len(UV))]), bend=FLOUNCE_BEND, flat=FLOUNCE_FLAT)
    fab.sew(F['gather'], np.arange(n0, n0 + nc), loose=1e-3, firm=0.0, seconds=0.2)
    fab.sew(F['close'][:, 0], F['close'][:, 1], loose=1e-3, firm=0.0, seconds=0.2)
    col = np.arange(n0) % nc
    fab.tether(np.arange(n0), n0 + col, 1.02 * np.abs(UV[:, 1] + F['head']) + 0.05)
    # the body: the settled skirt (outer side), her, the floor
    sp_, sn_ = cloth_surface(X, A['T'], A['piece'])
    body_ = xf.MeshBody(np.vstack([sp_, pts]), np.vstack([sn_, nrm]), gap=GAP, floor=floor, reach=1.0)
    t0 = time.time()
    for k in range(int(round(0.4 / STEP))):
        fab.step(STEP, body_)
    k = fab.settle(body_, h=STEP, most=480, tol=2e-4, log=log)
    log('petticoat flounce: settled in %d more steps (%.0f s)' % (k, time.time() - t0))
    return fab.pos[:n0].copy(), T, UV, F
