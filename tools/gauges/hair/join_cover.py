"""Join gauge: the interface between two sets of hair combed opposite ways (a fringe and the hair swept back from it).

    python3 tools/gauges/hair/join_cover.py HEAD.json.gz ASSET.json [--cap-mm 0.4] [--band-mm 12]

For a Face Creator strand asset (bangs_creator_asset.py). Strands are split by the way they leave the root (forward or
back). Rows run across the join, front to back, in three slabs (left, middle, right). Each row gives

  cover   layers of hair over the scalp: strand length per mm2 x the strand's drawn width (hairs_per_strand x 0.07 mm).
          Under 1 the scalp cap shows through.
  outer   the hair surface's height over the scalp, mm (90th percentile).
  buried  the share of the strand length that lies under the scalp cap (--cap-mm): buried hair is not seen at all,
          and the bare cap shows as a flat dark band.

and the summary gives the worst cover and the buried share within --band-mm of the join. Trouble: cover under 1 on
more than a few mm, buried over 0.1, or a step in `outer` over 2 mm between neighbouring rows.
"""
import os, sys, json, base64, zlib, gzip
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hair'))
import scalp_field as SF

HAIR_MM = 0.07


def _dec(s, dt):
    b = base64.b64decode(s)
    for f in (zlib.decompress, gzip.decompress):
        try:
            b = f(b); break
        except Exception:
            pass
    return np.frombuffer(b, dt)


def gauge(head, asset, cap_mm=0.4, band_mm=12.0, log=print):
    MM = SF.MM
    a = json.load(open(asset))['strands']
    cnt = _dec(a['counts'], '<u2').astype(int)
    pos = _dec(a['pos'], '<i2').reshape(-1, 3) * a['q']
    width = a['hairs_per_strand'] * HAIR_MM
    st = np.r_[0, np.cumsum(cnt)[:-1]]
    sid = np.repeat(np.arange(len(cnt)), cnt)
    surf = SF.Surface(SF.load_head(head))
    R = pos[st]
    fwd_s = (pos[np.minimum(st + 8, st + cnt - 1)] - R)[:, 1] < 0
    fwd = fwd_s[sid]
    U = pos - SF.CENTRE
    r = np.linalg.norm(U, axis=1)
    h = (r - np.linalg.norm(surf.point(U / r[:, None]) - SF.CENTRE, axis=1)) / MM
    x, y = pos[:, 0] / MM, pos[:, 1] / MM
    top = (pos[:, 2] > np.percentile(R[fwd_s, 2], 5)) & (h < 25)
    seg = np.r_[np.linalg.norm(np.diff(pos, axis=0), axis=1) / MM, 0.0]
    seg[np.cumsum(cnt) - 1] = 0
    rep = dict(slabs={})
    worst, bur_len, all_len = 9.0, 0.0, 0.0
    for xs in (-30, 0, 30):
        sl = top & (np.abs(x - xs) < 10)
        fr = R[fwd_s & (np.abs(R[:, 0] / MM - xs) < 10)]
        bk = R[~fwd_s & (np.abs(R[:, 0] / MM - xs) < 10) & (R[:, 2] > np.percentile(R[fwd_s, 2], 5))]
        yj = 0.5 * (np.percentile(fr[:, 1], 90) + np.percentile(bk[:, 1], 10)) / MM     # the join: where the root sets meet
        rows = []
        log('slab x %+d mm, join at y %.0f mm\n   y-join  cover  fwd/back  outer  buried' % (xs, yj))
        for dy in range(-24, 25, 4):
            k = sl & (np.abs(y - yj - dy) < 2)
            L = seg[k].sum()
            cover = L / 80.0 * width
            bur = seg[k & (h < cap_mm)].sum() / max(L, 1e-9)
            outer = float(np.percentile(h[k], 90)) if k.sum() > 10 else float('nan')
            rows.append((dy, round(cover, 2), round(outer, 1), round(bur, 2)))
            log('   %+5d  %5.2f  %4.2f/%4.2f  %5.1f  %5.2f' % (dy, cover, seg[k & fwd].sum() / 80 * width, seg[k & ~fwd].sum() / 80 * width, outer, bur))
            if abs(dy) <= band_mm:
                worst = min(worst, cover); bur_len += seg[k & (h < cap_mm)].sum(); all_len += L
        rep['slabs'][xs] = rows
    rep['cover_min_in_band'] = round(worst, 2)
    rep['buried_share_in_band'] = round(bur_len / max(all_len, 1e-9), 3)
    log(json.dumps({k: v for k, v in rep.items() if k != 'slabs'}))
    return rep


if __name__ == '__main__':
    a = sys.argv[1:]
    f = lambda k, d: float(a[a.index(k) + 1]) if k in a else d
    gauge(a[0], a[1], f('--cap-mm', 0.4), f('--band-mm', 12.0))
