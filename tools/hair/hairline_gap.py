"""The hairline gauge: is the scalp behind the hairline covered all the way along it?

    python3 tools/hair/hairline_gap.py HEAD.json.gz GROOM_DIR [--strands 8000] [--map OUT.png] [--check]

The invariant (John 2026-11-25: "the hairline cannot have any gaps"): from one ear, over the temples and the forehead,
to the other ear, no cell of scalp behind the hairline is bare. The strip of scalp HL_DEPTH_MM deep behind the hairline
(scalp_field's own hairline and its distance inside it) is cut into cells HL_CELL_RAD round the head by HL_BAND_MM
deep. The drawn hair (children as render.py grows them, which is what the page and the cards draw) is laid on the
skin along the skull's radius, as the eye sees it from outside; a cell is covered when HL_COVER or more hairs per mm
of its width pass over it. Prints, per side, the bare cells, the longest bare run along the hairline in mm, and the
thinnest cell. --check exits 1 when any cell is bare, so a build can stop on it."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF
import render as RD

HL_TH = 1.75             # rad: the hairline gauged, the centre part round to over the ear (the ear's back about 1.6)
HL_CELL_RAD = 0.06       # rad round the head: about 4.5 mm of hairline at the temple
HL_DEPTH_MM = 21.0       # the strip's depth behind the hairline: the transition zone, 2 cm [W25] (scalp_field TRANSITION)
HL_BAND_MM = 7.0         # each band's depth
HL_OFF_MM = 20.0         # hair lying up to this far off the skin still hides it (the bangs' root lift is 6 mm)
HL_COVER = 4.0           # DRAFT: hairs per mm of a cell's width for it to read as covered: side_fill's FILL_HAIRS (40
                         # over a 1 cm cell), the count at which the page's strand view stops showing skin
HL_SUB = 4               # each 1 cm segment of a path is read at this many points


def measure(F, surf, paths, hairs, memo=None):
    """(H, width_mm): hairs over each cell [side, angle bin, band] and each cell's width along the hairline.
    memo: an optional object with .d (dict), .new and .h(array): the cells each path touches are kept by the path's
    content, so a rerun reads only the paths that changed."""
    from scipy.spatial import cKDTree
    nb, nd = int(round(HL_TH / HL_CELL_RAD)), int(round(HL_DEPTH_MM / HL_BAND_MM))
    P, th, ins = F['P'], F['theta'], F['inside'] / SF.MM
    strip = F['mask'] & (np.abs(th) < HL_TH) & (ins < HL_DEPTH_MM)
    sp = np.where(strip)[0]
    side = (th[sp] > 0).astype(int)
    cell = (side * nb + np.minimum((np.abs(th[sp]) / HL_CELL_RAD).astype(int), nb - 1)) * nd \
        + np.minimum((ins[sp] / HL_BAND_MM).astype(int), nd - 1)
    # near the strip but not in it (the forehead below, the scalp further in): a hair nearest those covers no cell
    near = np.where(~strip & (np.abs(th) < HL_TH + 0.3) & (P[:, 2] > 0.3))[0]
    allp = np.r_[sp, near]
    tree = cKDTree(SF.unit(P[allp] - SF.CENTRE))
    H = np.zeros(2 * nb * nd)
    t = (np.arange(HL_SUB) / HL_SUB)[None, :, None]
    for p, h in zip(paths, hairs):
        p = p[~np.isnan(p[:, 0])]
        if len(p) < 2:
            continue
        mk = memo.h(p) if memo is not None else None
        if mk is not None and mk in memo.d:
            H[memo.d[mk]] += h
            continue
        q = (p[:-1, None] + (p[1:] - p[:-1])[:, None] * t).reshape(-1, 3)
        q = q[(q[:, 2] > 0.3) & (-surf.depth(q) < HL_OFF_MM * SF.MM)]
        cells = np.zeros(0, int)
        if len(q):
            k = tree.query(SF.unit(q - SF.CENTRE))[1]
            cells = np.unique(cell[k[k < len(sp)]])
        if mk is not None:
            memo.d[mk] = cells; memo.new = True
        H[cells] += h
    # each cell's width along the hairline: its angle bin's arc at the skin's distance from the axis
    r = np.hypot(P[sp, 0] - SF.AXIS[0], P[sp, 1] - SF.AXIS[1])
    W = np.zeros(2 * nb * nd); N = np.zeros(2 * nb * nd)
    np.add.at(W, cell, r); np.add.at(N, cell, 1)
    width = np.where(N > 0, W / np.maximum(N, 1), np.nan) * HL_CELL_RAD / SF.MM
    return H.reshape(2, nb, nd), width.reshape(2, nb, nd)


def report(H, width):
    """Per side: the cover (hairs per mm) by cell, the bare cells, and the longest bare run along the hairline."""
    cov = H / width
    out = {}
    for s, lab in ((0, 'right'), (1, 'left')):
        ok = ~(cov[s] < HL_COVER)                      # (a cell with no skin samples has no width: not counted)
        bare = [(round(float((i + 0.5) * HL_CELL_RAD), 2), int(j)) for i, j in zip(*np.where(~ok))]
        col = ~ok.all(1)                               # an angle bin with any bare band
        run = best = 0.0
        for i in range(len(col)):
            run = run + np.nanmean(width[s, i]) if col[i] else 0.0
            best = max(best, run)
        out[lab] = dict(cells=int(np.isfinite(cov[s]).sum()), bare=bare, longest_gap_mm=round(float(best), 1),
                        thinnest=round(float(np.nanmin(cov[s])), 1))
    return out


def gauge(head, groom_dir, n=8000):
    surf = SF.Surface(SF.load_head(head))
    F = SF.build(surf)
    Gm = dict(np.load(os.path.join(groom_dir, 'groom.npz')))
    paths, share = RD.grow_children(F, Gm, n)
    H, width = measure(F, surf, paths, share * Gm['n_hairs'].sum())
    return H, width, report(H, width)


def draw(H, width, path):
    """The strip unrolled: angle round the head across (right ear, forehead, left ear), depth behind the hairline up.
    Green covered, red bare, the number the hairs per mm."""
    from PIL import Image, ImageDraw
    cov = H / width
    nb, nd = cov.shape[1:]
    c = 34
    im = Image.new('RGB', (2 * nb * c + 20, nd * c + 50), (28, 30, 30)); d = ImageDraw.Draw(im)
    for s in (0, 1):
        for i in range(nb):
            for j in range(nd):
                v = cov[s, i, j]
                if not np.isfinite(v):
                    continue
                x = 10 + ((nb - 1 - i) if s == 0 else (nb + i)) * c
                y = 10 + (nd - 1 - j) * c
                k = min(v / (10 * HL_COVER), 1.0)
                col = (200, 50, 50) if v < HL_COVER else (int(150 * (1 - k)), int(110 + 60 * k), 60)
                d.rectangle([x, y, x + c - 2, y + c - 2], fill=col)
                d.text((x + 4, y + 10), '%d' % round(v), fill=(255, 255, 255))
    d.text((10, nd * c + 16), 'right ear', fill=(200, 200, 200))
    d.text((nb * c - 12, nd * c + 16), 'part', fill=(200, 200, 200))
    d.text((2 * nb * c - 40, nd * c + 16), 'left ear', fill=(200, 200, 200))
    d.text((10, nd * c + 32), 'bottom row: the hairline. hairs per mm of hairline; red under %.0f' % HL_COVER,
           fill=(200, 200, 200))
    im.save(path)


def main():
    a = sys.argv[1:]
    n = int(a[a.index('--strands') + 1]) if '--strands' in a else 8000
    H, width, rep = gauge(a[0], a[1], n)
    for lab, r in rep.items():
        print('hairline %-5s %d cells, %d bare, longest gap %.1f mm, thinnest %.1f hairs/mm; bare at (|theta|, band): %s'
              % (lab, r['cells'], len(r['bare']), r['longest_gap_mm'], r['thinnest'], r['bare'] if r['bare'] else 'none'))
    if '--map' in a:
        draw(H, width, a[a.index('--map') + 1])
    if '--check' in a and any(r['bare'] for r in rep.values()):
        print('HAIRLINE INVARIANT BROKEN: the hairline has a gap')
        sys.exit(1)


if __name__ == '__main__':
    main()
