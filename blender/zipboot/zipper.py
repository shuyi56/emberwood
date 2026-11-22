"""The zip of the zip ankle boot as real parts, built along the seam last.cut_zip cut and sew.sew closed: its two
tapes under the leather's folded edges, the coil's teeth between them, the bottom stop, the slider closed at the top
and its pull hanging from it. Each part follows the leather nearest it (sew._on_boot), so the zip moves with the boot.

Sizes: a #3 nylon coil (last.ZIP: the teeth 4.3 mm across closed, the tape 22.7 mm both sides; UCAN Zippers). The
slider and pull are measured on the reference photo (refmeasure: the slider and pull together 0.060 of the foot's
length, about 1.4 cm, the pull about 0.022 L wide). Where no source gives a size the number is marked EXPERIMENTAL."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'boot'))
import last  # noqa: E402
import sew  # noqa: E402

PARTS = {
    'pitch_mm': 1.9,        # the coil's element pitch along the zip. EXPERIMENTAL (no source found for a #3 coil's
                            # pitch; about 0.45 of the teeth's width, as a coil looks)
    'teeth_t_mm': 2.2,      # the coil's thickness: Rockywoods gives a #5 coil 0.103 in (2.6 mm), scaled to the #3's
                            # width (4.3 / 5.0)
    'tape_t_mm': 0.6,       # the tape's thickness. EXPERIMENTAL
    'slider_l': 0.024,      # the slider's body, x the foot's length L (photo: slider and pull 0.060 L together)
    'slider_w_mm': 7.0,     # and its width: the teeth and a flange either side (photo, about 0.03 L). EXPERIMENTAL split
    'slider_t_mm': 3.2,     # its thickness over the teeth. EXPERIMENTAL
    'pull_l': 0.040,        # the pull's length, x L (photo: the rest of the 0.060 L, lapping the slider)
    'pull_w': 0.022,        # and its width, x L (photo)
    'pull_t_mm': 1.0,       # its thickness. EXPERIMENTAL
    'stop_mm': 3.0,         # the bottom stop's length along the zip. EXPERIMENTAL
}


def _box(c, t, a, n, l, w, h, taper=1.0):
    """A box centred at c, l long along t, w across along a, h thick along n (its top face, along +t, taper x as
    wide). Returns (points [8, 3], faces)."""
    V = []
    for st in (-1, 1):
        ww = w * (taper if st > 0 else 1.0)
        for sa in (-1, 1):
            for sn in (-1, 1):
                V.append(c + t * st * 0.5 * l + a * sa * 0.5 * ww + n * sn * 0.5 * h)
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return np.array(V), F


def _line(o, pieces):
    """The zip's middle line up the closed seam, from the stop to the top: points [m, 3], the leather's normal there,
    the direction across the zip (from the front piece's edge to the inner quarter's), and the gap between them."""
    X, Nv = sew._co(o), sew._normals(o)
    by = {p['name']: p for p in pieces}
    fr, inn = by['vamp and outer quarter'], by['inner quarter']
    zi = fr['zip']

    def edge(p):
        idx = np.nonzero(p['edges']['Z'])[0] + p['start']
        return X[idx], Nv[idx]
    A, NA = edge(fr)
    B, NB = edge(inn)
    if len(A) < 4 or len(B) < 4:
        raise RuntimeError('zipper: the zip edges were not found on %s (%d and %d points)' % (o.name, len(A), len(B)))
    zs = np.arange(zi['stop'] + 0.02, zi['top'] - 0.02, 0.04)

    def at(P, Nn, z):
        w = np.exp(-((P[:, 2] - z) / 0.05) ** 2)
        if w.sum() < 1e-6:
            k = np.argmin(np.abs(P[:, 2] - z))
            return P[k], Nn[k]
        return (P * w[:, None]).sum(0) / w.sum(), (Nn * w[:, None]).sum(0) / w.sum()
    M, N, Ax, G = [], [], [], []
    for z in zs:
        a, na = at(A, NA, z)
        b, nb = at(B, NB, z)
        M.append(0.5 * (a + b))
        n = na + nb
        N.append(n / max(np.linalg.norm(n), 1e-9))
        Ax.append(b - a)
        G.append(np.linalg.norm(b - a))
    M, N, Ax = np.array(M), np.array(N), np.array(Ax)
    for _ in range(3):                                   # (a smooth line: the edges' points are a mesh's, not a curve)
        M[1:-1] = 0.25 * M[:-2] + 0.5 * M[1:-1] + 0.25 * M[2:]
        N[1:-1] = 0.25 * N[:-2] + 0.5 * N[1:-1] + 0.25 * N[2:]
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    Ax -= N * (Ax * N).sum(1, keepdims=True)
    Ax /= np.maximum(np.linalg.norm(Ax, axis=1, keepdims=True), 1e-9)
    return M, N, Ax, np.array(G)


def build(o, pieces, rig, L, mats, log=print):
    """The zip's parts on the boot o. mats: {'teeth', 'tape', 'metal'}. Returns {part: object}."""
    zi = pieces[0]['zip']
    mm, tl = zi['mm'], o.modifiers['thickness'].thickness if 'thickness' in o.modifiers else 0.0
    Z, Pr = last.ZIP, PARTS
    M, N, Ax, G = _line(o, pieces)
    seg = np.linalg.norm(np.diff(M, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    def frame(s):
        k = int(np.clip(np.searchsorted(arc, s) - 1, 0, len(seg) - 1))
        f = (s - arc[k]) / max(seg[k], 1e-9)
        c = M[k] + (M[k + 1] - M[k]) * f
        n = N[k] + (N[k + 1] - N[k]) * f
        n /= np.linalg.norm(n)
        t = M[k + 1] - M[k]
        t -= n * np.dot(t, n)
        t /= max(np.linalg.norm(t), 1e-9)
        a = np.cross(n, t)
        if np.dot(a, Ax[k]) < 0:
            a = -a
        return c, t, a, n
    out = {}
    tw, th = Z['teeth_mm'] * mm, Pr['teeth_t_mm'] * mm
    # the teeth: the coil's elements, each side's in turn, each reaching a little past the middle into the other's
    V, F = [], []
    p = Pr['pitch_mm'] * mm
    for k, s in enumerate(np.arange(0.5 * p, arc[-1] - 0.5 * p, 0.5 * p)):
        c, t, a, n = frame(s)
        sd = 1.0 if k % 2 else -1.0
        cc = c + a * sd * 0.19 * tw - n * 0.5 * tl          # (set in the gap, its face a little under the leather's)
        bv, bf = _box(cc, t, a, n, 0.55 * p, 0.62 * tw, th)
        F += [[i + len(V) for i in f_] for f_ in bf]
        V += list(bv)
    out['teeth'] = sew._on_boot(o, np.array(V), F, o.name.replace('Boot', 'Zip teeth'), mats.get('teeth'), rig)
    # the tapes: either side from the teeth out under the leather
    V, F = [], []
    half = 0.5 * Z['tape_mm'] * mm
    for sd in (-1.0, 1.0):
        n0 = len(V)
        rows = np.arange(0.0, arc[-1] + 1e-6, 0.06)
        for s in rows:
            c, t, a, n = frame(s)
            for x in (0.35 * tw, half):
                base = c + a * sd * x - n * (tl + 0.5 * Pr['tape_t_mm'] * mm)
                V += [base + n * 0.5 * Pr['tape_t_mm'] * mm, base - n * 0.5 * Pr['tape_t_mm'] * mm]
        for i in range(len(rows) - 1):
            q0, q1 = n0 + 4 * i, n0 + 4 * (i + 1)
            F += [[q0, q0 + 2, q1 + 2, q1], [q0 + 1, q1 + 1, q1 + 3, q0 + 3],
                  [q0, q1, q1 + 1, q0 + 1], [q0 + 2, q0 + 3, q1 + 3, q1 + 2]]
    out['tape'] = sew._on_boot(o, np.array(V), F, o.name.replace('Boot', 'Zip tape'), mats.get('tape'), rig)
    # the bottom stop, across both sides' teeth at the zip's foot
    c, t, a, n = frame(0.0)
    sv, sf = _box(c + t * 0.5 * Pr['stop_mm'] * mm - n * 0.3 * tl, t, a, n, Pr['stop_mm'] * mm, 1.15 * tw, 1.1 * th)
    # the slider, closed at the top: a body wider at its mouth (the top) than its tail, standing on the teeth
    sl, sw, st_ = Pr['slider_l'] * L, Pr['slider_w_mm'] * mm, Pr['slider_t_mm'] * mm
    c, t, a, n = frame(arc[-1] - 0.5 * sl)
    bv, bf = _box(c + n * (0.5 * st_ - 0.3 * tl), t, a, n, sl, 0.8 * sw, st_, taper=1.25)
    # the pull: a flat rounded tab hinged on the slider's back, lying down over the slider and past its tail
    pl, pw, pt = Pr['pull_l'] * L, Pr['pull_w'] * L, Pr['pull_t_mm'] * mm
    hinge = arc[-1] - 0.35 * sl
    hp = pl - 0.5 * pw
    ring = [(-0.5 * pw, 0.0), (0.5 * pw, 0.0), (0.5 * pw, -hp)]
    ring += [(0.5 * pw * np.cos(g_), -hp - 0.5 * pw * np.sin(g_)) for g_ in np.linspace(0, np.pi, 11)[1:-1]]
    ring += [(-0.5 * pw, -hp)]
    on_slider, on_teeth = st_ + 0.5 * pt - 0.3 * tl, th + 0.5 * pt - 0.5 * tl
    PV = []
    for sgn in (1, -1):
        for x, y in ring:
            cc, tt, aa, nn = frame(max(hinge + y, 0.0))
            k_ = np.clip((-y - 0.65 * sl) / (0.5 * sl), 0.0, 1.0)          # (off the slider's tail, down onto the teeth)
            PV.append(cc + aa * x + nn * (on_slider * (1 - k_) + on_teeth * k_ + sgn * 0.5 * pt))
    m = len(ring)
    PF = [list(range(m))[::-1], list(range(m, 2 * m))] + [[i, (i + 1) % m, m + (i + 1) % m, m + i] for i in range(m)]
    V = list(sv) + list(bv) + PV
    F = sf + [[i + 8 for i in f_] for f_ in bf] + [[i + 16 for i in f_] for f_ in PF]
    out['metal'] = sew._on_boot(o, np.array(V), F, o.name.replace('Boot', 'Zip slider'), mats.get('metal'), rig)
    log('zipper %s: %.1f cm of #3 coil (%d elements at %.1f mm), tapes %.1f mm wide, the slider %.1f by %.1f mm, the '
        'pull %.1f mm long; the leather edges stand %.1f mm apart on average (%.1f..%.1f)'
        % (o.name, arc[-1] * 165 / 63, int(arc[-1] / (0.5 * p)), Pr['pitch_mm'], Z['tape_mm'], sl / mm, sw / mm, pl / mm,
           G.mean() / mm, G.min() / mm, G.max() / mm), flush=True)
    return out
