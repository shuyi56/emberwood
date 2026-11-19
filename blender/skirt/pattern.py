"""The skirt as the tailor cuts it: flat pattern pieces, each edge paired with the edge it is sewn to.

The finished skirt (draft.py) is a surface. Here the same surface is read back as cloth: it is cut at the gore seams
into seven gores (a front gore, and on each side a side-front, a side-back and a back gore, the two back gores meeting
at the middle of the back), and a straight waistband, and each piece is laid flat. Nothing is sewn: no seam is joined,
no stitch is built as geometry and the skirt that renders is still the one drafted shell. This module only drafts
(John, 2026-11-19: always draft flat, and do the stitching only when he says).

A piece is laid flat by unfolding its surface grid (the gore's rows down and columns across) point by point: each
point sits at its true distance from the two already placed beside and above it, so every length along the grid is
kept, and the edges that will meet in a seam are the very polylines the finished skirt has. A gore is nearly a
developable cone, so little is stretched; `stretch` reports how much (the cell diagonals, flat over solid: the edges
themselves are kept exactly).

Each piece carries
    stitch   the finished outline, the line the seams are sewn on (closed polygon, flat units)
    cut      the cut outline: the stitch line let out by the seam allowance at the seams and the waist, and by the hem
             allowance at the hem (a turned hem is folded up inside and sewn down)
    grain    the straight grain, a line down the middle of the piece, parallel to the warp
    edges    left, right, top and bottom: each edge's polyline on the stitch line and its length
    notches  balance marks at the hip and the knee, the same height on both edges of a seam
and the pairs list says which edge is sewn to which, with both lengths and the difference (the ease: a seam whose two
sides differ is eased in, here none do, and the check is that they match).

Units are model units (a head is about 9.4); `flat_sheet` lays the pieces out for a picture."""
import math

import numpy as np

# the seams of the skirt, as angles from the front (phi 0, growing toward her right): from draft.SEAMS
SEAM_PHIS = [0.55, 1.45, 2.4, math.pi, 2 * math.pi - 2.4, 2 * math.pi - 1.45, 2 * math.pi - 0.55]
GORES = ['right side-front gore', 'right side-back gore', 'right back gore', 'left back gore',
         'left side-back gore', 'left side-front gore', 'front gore']
COL_STEP = 0.05          # how far apart a gore's columns run (radians)
HEM_ALLOWANCE_TURN = 1.0   # the hem allowance, in hem bands: the turned-up depth, and the fold's own


def _circle_hit(A, ra, B, rb, away):
    """The point at distance ra from A and rb from B, on the side of AB that `away` (a direction) points to."""
    d = np.linalg.norm(B - A)
    if d < 1e-9:
        return None
    a = (ra * ra - rb * rb + d * d) / (2 * d)
    h2 = ra * ra - a * a
    if h2 < 0:
        h2 = 0.0
    u = (B - A) / d
    n = np.array([-u[1], u[0]])
    m = A + a * u
    p1, p2 = m + math.sqrt(h2) * n, m - math.sqrt(h2) * n
    return p1 if np.dot(p1 - m, away) >= np.dot(p2 - m, away) else p2


def unfold(G):
    """Lay a grid of 3-D points G (rows down, columns across) flat, keeping every edge's length where it can be kept:
    returns F (rows, cols, 2) with the middle column straight down."""
    R, C = G.shape[:2]
    F = np.zeros((R, C, 2))
    c = C // 2
    down = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(G[:, c], axis=0), axis=1))])
    F[:, c, 1] = down
    for sgn, cols in ((1, range(c + 1, C)), (-1, range(c - 1, -1, -1))):
        for j in cols:
            jp = j - sgn
            F[0, j] = F[0, jp] + np.array([sgn * np.linalg.norm(G[0, j] - G[0, jp]), 0.0])
            for i in range(1, R):
                A, B = F[i, jp], F[i - 1, j]
                hit = _circle_hit(A, np.linalg.norm(G[i, j] - G[i, jp]), B, np.linalg.norm(G[i, j] - G[i - 1, j]),
                                  np.array([sgn * 1.0, 0.3]))
                F[i, j] = hit if hit is not None else B + np.array([0.0, np.linalg.norm(G[i, j] - G[i - 1, j])])
    return F


def _length(poly):
    return float(np.linalg.norm(np.diff(np.asarray(poly), axis=0), axis=1).sum())


def _gore(name, a, b, column, rows, allow, hem_allow, notch_z):
    n = max(8, int(round((b - a) / COL_STEP)))
    phis = np.linspace(a, b, n + 1)
    G = np.stack([column(p)[0] for p in phis], axis=1)            # (rows, cols, 3)
    F = unfold(G)               # (x grows with the angle: the right edge is the one toward the next gore round)
    left, right = F[:, 0], F[:, -1]
    top, bottom = F[0, :], F[-1, :]
    stitch = np.concatenate([left, bottom[1:], right[-2::-1], top[-2:0:-1]])
    # the cut line: out by the allowances (the left edge's outward is -x, the right's +x, the top's -y, the hem's +y)
    cl = left + np.array([-allow, 0.0])
    cr = right + np.array([allow, 0.0])
    ct = top + np.array([0.0, -allow])
    cb = bottom + np.array([0.0, hem_allow])
    cut = np.concatenate([[[cl[0, 0], ct[0, 1]]], cl, [[cl[-1, 0], cb[0, 1]]], cb[1:-1], [[cr[-1, 0], cb[-1, 1]]],
                          cr[::-1], [[cr[0, 0], ct[-1, 1]]], ct[::-1][1:-1]])
    mid = n // 2
    grain = np.array([[F[:, mid, 0].mean(), F[0, mid, 1] + 0.3], [F[:, mid, 0].mean(), F[-1, mid, 1] - 0.3]])
    # solid versus flat: the unfolding keeps every edge of the grid (rows and columns), so what it can stretch is the
    # shear: the cell diagonals, flat over solid
    ratios = []
    for i in range(rows - 1):
        for j in range(n):
            for (a_, b_) in (((i, j), (i + 1, j + 1)), ((i, j + 1), (i + 1, j))):
                ratios.append(np.linalg.norm(F[a_] - F[b_]) / max(np.linalg.norm(G[a_] - G[b_]), 1e-9))
    notches = []
    for nz in notch_z:
        k = int(np.argmin(np.abs(G[:, 0, 2] - nz)))
        notches.append((float(nz), (float(left[k, 0]), float(left[k, 1])), (float(right[k, 0]), float(right[k, 1]))))
    return dict(name=name, kind='gore', phi=(float(a), float(b)), stitch=stitch, cut=cut, grain=grain,
                edges={'left': dict(poly=left, length=_length(left)), 'right': dict(poly=right, length=_length(right)),
                       'top': dict(poly=top, length=_length(top)), 'bottom': dict(poly=bottom, length=_length(bottom))},
                notches=notches, stretch=(float(min(ratios)), float(max(ratios))), solid=G)


def _waistband(top_lengths, height, allow):
    """The waistband: a straight strip, twice the finished height (folded on its middle line, the facing inside), as
    long as the gores' top edges laid end to end; its lower long edge is sewn to them in order."""
    L = float(sum(top_lengths))
    xs = np.cumsum([0.0] + list(top_lengths))
    stitch = np.array([[0.0, 0.0], [L, 0.0], [L, 2 * height], [0.0, 2 * height]])
    cut = np.array([[-allow, -allow], [L + allow, -allow], [L + allow, 2 * height + allow], [-allow, 2 * height + allow]])
    return dict(name='waistband', kind='band', stitch=stitch, cut=cut,
                grain=np.array([[0.3, height], [L - 0.3, height]]),     # (the straight grain runs along the strip)
                fold=np.array([[0.0, height], [L, height]]),
                edges={'bottom': dict(poly=np.array([[0.0, 0.0], [L, 0.0]]), length=L,
                                      segments=[(float(a), float(b)) for a, b in zip(xs[:-1], xs[1:])]),
                       'top': dict(poly=np.array([[0.0, 2 * height], [L, 2 * height]]), length=L),
                       'left': dict(poly=np.array([[0.0, 0.0], [0.0, 2 * height]]), length=2 * height),
                       'right': dict(poly=np.array([[L, 0.0], [L, 2 * height]]), length=2 * height)},
                notches=[], stretch=(1.0, 1.0))


def draft_pieces(D, M, log=print):
    """The skirt's flat pieces and the pairs of edges that are sewn together. Returns (pieces, pairs)."""
    r, column = D['r'], D['column']
    rows = D['rows']
    allow = r['allowance']
    hem_allow = HEM_ALLOWANCE_TURN * r['hem_band'] + r['wool']
    notch_z = [M['hips_z'], M['knee_z']]
    pieces = []
    bounds = SEAM_PHIS + [SEAM_PHIS[0] + 2 * math.pi]
    # (the front gore wraps through phi 0: it runs from the last seam to the first plus a turn)
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(SEAM_PHIS) - 1)] + [(SEAM_PHIS[-1] - 2 * math.pi, SEAM_PHIS[0])]
    for name, (a, b) in zip(GORES, spans):
        # (column() takes any phi: it is periodic, and the front gore's columns run through zero)
        pieces.append(_gore(name, a, b, column, rows, allow, hem_allow, notch_z))
    order = [6, 0, 1, 2, 3, 4, 5]                    # (round the waist from her left front, as the waistband is laid)
    # the strip's bottom edge is the gores' top edges in the order they stand round the waist, from the front gore
    ring = [pieces[i] for i in order]
    band = _waistband([p['edges']['top']['length'] for p in ring], r['band'], allow)
    pieces.append(band)
    # the pairs: each gore's right edge (toward her right, the next gore round) to the next gore's left edge
    pairs = []
    ids = {p['name']: k for k, p in enumerate(pieces)}
    for k, p in enumerate(ring):
        q = ring[(k + 1) % len(ring)]
        la, lb = p['edges']['right']['length'], q['edges']['left']['length']
        pairs.append(dict(a=(p['name'], 'right'), b=(q['name'], 'left'), length_a=la, length_b=lb, ease=lb - la,
                          kind='seam'))
    # (the ring starts at the front gore, whose left edge pairs with the left side-front's right edge: pairs above run
    # her-left to her-right, so the first, front gore -> right side-front, is the front gore's right edge)
    xs = band['edges']['bottom']['segments']
    for p, (x0, x1) in zip(ring, xs):
        pairs.append(dict(a=(p['name'], 'top'), b=('waistband', 'bottom[%.2f..%.2f]' % (x0, x1)),
                          length_a=p['edges']['top']['length'], length_b=x1 - x0, ease=(x1 - x0) - p['edges']['top']['length'],
                          kind='waist'))
    pairs.append(dict(a=('waistband', 'left'), b=('waistband', 'right'), length_a=band['edges']['left']['length'],
                      length_b=band['edges']['right']['length'], ease=0.0, kind='closing'))
    for p in pieces:
        if p['kind'] == 'gore':
            pairs.append(dict(a=(p['name'], 'bottom'), b=(p['name'], 'hem allowance, folded up inside'),
                              length_a=p['edges']['bottom']['length'], length_b=p['edges']['bottom']['length'],
                              ease=0.0, kind='hem'))
    return pieces, pairs, dict(allowance=allow, hem_allowance=hem_allow)


def report(pieces, pairs, log=print):
    log('skirt pattern: %d pieces' % len(pieces))
    for p in pieces:
        if p['kind'] == 'gore':
            e = p['edges']
            log('  %-22s top %.2f, hem %.2f, sides %.2f / %.2f; stretch %.3f..%.3f' % (
                p['name'], e['top']['length'], e['bottom']['length'], e['left']['length'], e['right']['length'],
                p['stretch'][0], p['stretch'][1]))
        else:
            log('  %-22s %.2f long, %.2f wide (folded to %.2f)' % (p['name'], p['edges']['bottom']['length'],
                                                                 p['edges']['left']['length'], 0.5 * p['edges']['left']['length']))
    worst = max(abs(q['ease']) for q in pairs)
    log('skirt pattern: %d paired edges, the worst pair differs by %.4f' % (len(pairs), worst))
    for q in pairs:
        log('  %-7s %-30s <-> %-34s %.2f / %.2f' % (q['kind'], '%s %s' % q['a'], '%s %s' % q['b'], q['length_a'], q['length_b']))


def flat_sheet(pieces, path, scale=14.0, gap=3.0):
    """The pieces laid out side by side, drawn: the cut line black, the stitch line red, the grain blue, the notches
    green, each seam edge lettered with its pair's letter (a pair shares a letter)."""
    from PIL import Image, ImageDraw
    x0 = 0.0
    placed = []
    for p in pieces:
        pts = np.asarray(p['cut'])
        lo, hi = pts.min(0), pts.max(0)
        placed.append((p, np.array([x0 - lo[0], -lo[1]])))
        x0 += hi[0] - lo[0] + gap
    H = max((np.asarray(p['cut'])[:, 1].max() - np.asarray(p['cut'])[:, 1].min()) for p in pieces) + 2 * gap
    img = Image.new('RGB', (int((x0 + gap) * scale), int(H * scale)), (246, 244, 238))
    dr = ImageDraw.Draw(img)

    def px(P, off):
        P = np.asarray(P) + off + np.array([gap * 0.5, gap])
        return [(float(a * scale), float(b * scale)) for a, b in P]
    for p, off in placed:
        dr.polygon(px(p['cut'], off), outline=(20, 20, 20))
        dr.line(px(np.concatenate([p['stitch'], p['stitch'][:1]]), off), fill=(200, 40, 40), width=1)
        dr.line(px(p['grain'], off), fill=(40, 80, 200), width=2)
        for z, a, b in p.get('notches', []):
            for q in (a, b):
                x, y = px([q], off)[0]
                dr.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(30, 150, 60))
        c = np.asarray(p['cut']).mean(0)
        dr.text(px([c], off)[0], p['name'], fill=(20, 20, 20))
    img.save(path)
    return path
