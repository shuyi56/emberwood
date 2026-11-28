"""How alone each point of a drawn strand is: 0 inside a mass of hair, 1 for a lone hair over skin or air.

    python3 tools/hair/strand_iso.py ASSET.json [OUT.json]      # adds strands.iso (u1 per point) to a strand asset

A lone hair is not a thin slice of the mass. It is seen at its own width against what lies behind it, with no hair
under it, and its colour is one fibre's (light passes through it: the TT lobe, Marschner et al. 2003), not the dark
of many absorptions (Zinke et al. 2008). The page draws by this value: no coverage layer under a lone hair, its true
width, a lighter tone.
Measured as the number of other strands that pass within R_MM of the point.
"""
import sys, json, base64, zlib, gzip
import numpy as np
from scipy.spatial import cKDTree

K = 96                      # neighbours looked at per point
R_MM = 3.0                  # the neighbourhood: about one lock's width
LONE_N, MASS_N = 6, 22      # DRAFT: other strands within R_MM. Up to LONE_N is a loose wisp (a few hairs running
                            # together still show the skin between them), from MASS_N it is a mass. Set on the Warden's
                            # groom: the fringe's face has a median of 40, the strays over the forehead 1 to 5.


def iso(P_mm, counts, k=K):
    """P_mm (n, 3) in mm, counts: points per strand -> (n,) 0..1."""
    counts = np.asarray(counts)
    sid = np.repeat(np.arange(len(counts)), counts)
    d, j = cKDTree(P_mm).query(P_mm, k=k, distance_upper_bound=R_MM, workers=-1)
    ok = np.isfinite(d)
    s = np.where(ok, sid[np.minimum(j, len(sid) - 1)], -1)
    s[s == sid[:, None]] = -1
    s.sort(axis=1)
    n = ((s[:, 1:] != s[:, :-1]) & (s[:, 1:] >= 0)).sum(1) + (s[:, 0] >= 0)   # distinct other strands in reach
    n = np.where(ok[:, -1], MASS_N, n)                                         # (k points all in reach: a mass)
    t = np.clip((MASS_N - n) / float(MASS_N - LONE_N), 0, 1)
    v = t * t * (3 - 2 * t)
    st = np.r_[0, np.cumsum(counts)[:-1]]              # eased along each strand: no on/off flicker point to point
    for _ in range(6):
        w = v.copy(); w[1:-1] = 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:]
        w[st] = v[st]; w[st + counts - 1] = v[st + counts - 1]; v = w
    return v, n


def _dec(s, dt):
    b = base64.b64decode(s)
    for f in (zlib.decompress, gzip.decompress):
        try:
            return np.frombuffer(f(b), dt), f
        except Exception:
            pass
    return np.frombuffer(b, dt), None


if __name__ == '__main__':
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scalp_field as SF
    src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else src
    a = json.load(open(src)); s = a['strands']
    raw, comp = _dec(s['pos'], '<i2')
    cnt, _ = _dec(s['counts'], '<u2')
    v, _ = iso(raw.reshape(-1, 3) * s['q'] / SF.MM, cnt.astype(int))
    b = np.round(v * 255).astype('u1').tobytes()
    if comp is zlib.decompress:
        b = zlib.compress(b)
    elif comp is gzip.decompress:
        b = gzip.compress(b)
    s['iso'] = base64.b64encode(b).decode()
    json.dump(a, open(out, 'w'))
    print('iso', dict(points=len(v), lone_share=round(float((v > 0.5).mean()), 4), mean=round(float(v.mean()), 3)))
