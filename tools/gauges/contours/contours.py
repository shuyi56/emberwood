"""Contours: the inspection of a solved sheet before it is rendered (John's rule, 2026-11-20: after any solver run and
before any render, the thread looks at the result's contours itself). One sheet of drawings and one file of numbers:

  - sections across each fold, at five stations along its line, the solved cloth (dark) over the target (grey, the
    fold as pressed from the pattern, or the caller's own target), in the crease's own frame;
  - sections at five heights (top view), and the side and front profiles, the solved cloth over the target;
  - the angle along each crease, hinge by hinge, against its target, and the angle measured across the fold;
  - on the flat pattern: the strain (each triangle's worst edge against its cut length, blue none to red 3% or more)
    and the bend (each edge's angle, grey flat to red 30 degrees or more; the creases' own edges in black).

contour_sheet(folds, path, target=None) writes path (.png) and path with .json; it needs numpy only (labels are
drawn when PIL is there, as outside Blender; Blender's Python has no PIL, so its sheets go unlabelled, the panels in
the order above). Folds.bake writes one every time it converges (see its sheet argument)."""
import json
import math
import struct
import zlib

import numpy as np

W, H = 300, 240            # one panel
PAD = 18


# ------------------------------------------------------------------ a small canvas (no dependencies)

class Canvas:
    def __init__(self, w, h, bg=(250, 250, 248)):
        self.a = np.empty((h, w, 3), np.uint8)
        self.a[:] = bg
        self.texts = []

    def line(self, p, q, col, width=1):
        p, q = np.asarray(p, float), np.asarray(q, float)
        n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1]), 1)) + 1
        t = np.linspace(0, 1, n)[:, None]
        pts = np.rint(p + (q - p) * t).astype(int)
        for dx in range(-(width // 2), width - width // 2):
            for dy in range(-(width // 2), width - width // 2):
                self._put(pts[:, 0] + dx, pts[:, 1] + dy, col)

    def poly(self, P, col, width=1, closed=False):
        for k in range(len(P) - 1 + (1 if closed else 0)):
            self.line(P[k], P[(k + 1) % len(P)], col, width)

    def dots(self, P, col):
        P = np.rint(np.asarray(P)).astype(int)
        self._put(P[:, 0], P[:, 1], col)

    def tri(self, A, B, C, col):
        lo = np.floor(np.minimum(np.minimum(A, B), C)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(A, B), C)).astype(int)
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        p = np.column_stack([xs.ravel() + 0.5, ys.ravel() + 0.5])
        v0, v1, v2 = B - A, C - A, p - A
        d = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(d) < 1e-9:
            return
        b1 = (v2[:, 0] * v1[1] - v2[:, 1] * v1[0]) / d
        b2 = (v0[0] * v2[:, 1] - v0[1] * v2[:, 0]) / d
        m = (b1 >= -1e-6) & (b2 >= -1e-6) & (b1 + b2 <= 1 + 1e-6)
        self._put(xs.ravel()[m], ys.ravel()[m], col)

    def text(self, x, y, s, col=(30, 30, 30)):
        self.texts.append((x, y, s, col))

    def _put(self, x, y, col):
        h, w = self.a.shape[:2]
        m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        self.a[y[m], x[m]] = col

    def save(self, path):
        a = self.a
        try:
            from PIL import Image, ImageDraw
            im = Image.fromarray(a)
            d = ImageDraw.Draw(im)
            for x, y, s, col in self.texts:
                d.text((x, y), s, fill=col)
            im.save(path)
            return
        except ImportError:
            pass
        h, w = a.shape[:2]
        raw = b''.join(b'\x00' + a[r].tobytes() for r in range(h))

        def chunk(t, d):
            c = struct.pack('>I', len(d)) + t + d
            return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
        with open(path, 'wb') as fh:
            fh.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def _fit(P, x0, y0, w=W, h=H, pad=PAD, box=None):
    """A map of 2D points into a panel (y up), the same scale on both axes. box: (lo, hi) to fit instead of P's."""
    lo, hi = (P.min(0), P.max(0)) if box is None else box
    s = min((w - 2 * pad) / max(hi[0] - lo[0], 1e-9), (h - 2 * pad - 12) / max(hi[1] - lo[1], 1e-9))
    cx, cy = 0.5 * (lo + hi)

    def f(Q):
        Q = np.asarray(Q, float)
        return np.column_stack([x0 + w / 2 + (Q[:, 0] - cx) * s, y0 + 12 + (h - 12) / 2 - (Q[:, 1] - cy) * s])
    return f, s


def _ramp(v, lo, hi, a=(60, 110, 200), b=(215, 45, 35), mid=(235, 235, 225)):
    t = float(np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1))
    c0, c1, u = (a, mid, t * 2) if t < 0.5 else (mid, b, t * 2 - 1)
    return tuple(int(c0[k] + (c1[k] - c0[k]) * u) for k in range(3))


def _edges(T):
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    return np.unique(E, axis=0)


def _cut(X, E, val, level):
    """Where the mesh's edges cross val == level: the points (interpolated on X) and, per crossing, its edge."""
    a, b = val[E[:, 0]], val[E[:, 1]]
    m = (a - level) * (b - level) < 0
    t = ((level - a[m]) / (b[m] - a[m]))[:, None]
    return X[E[m, 0]] + t * (X[E[m, 1]] - X[E[m, 0]])


# ------------------------------------------------------------------ the sheet

def contour_sheet(F, path, target=None, x=None, stations=5, heights=5):
    """Draw folds F's sheet (x, else its fabric's points) over target ([n, 3] world points: the shape it should take;
    if None, the creases folded from the pattern about the solved fixed side cannot be placed in the world, so only
    the hinges' targets are drawn). Writes path and path.json (the numbers), returns the numbers."""
    X = F.fab.pos if x is None else np.asarray(x, float)
    Tg = None if target is None else np.asarray(target, float)
    T = F.T
    P = F.flat[:, :2]
    E = _edges(T)
    nums = dict(creases=[], sections=[], heights=[])
    cols = 3
    rows = 2 + len(F.creases)
    cv = Canvas(W * cols, H * rows)
    dark, grey, red = (25, 25, 30), (175, 175, 180), (200, 40, 30)

    # row 1: side profile, front profile, sections at heights (top view)
    for ci, (ax, name) in enumerate((((1, 2), 'side profile (y, z)'), ((0, 2), 'front profile (x, z)'))):
        pts = np.vstack([X[:, ax]] + ([Tg[:, ax]] if Tg is not None else []))
        f, _ = _fit(pts, ci * W, 0)
        for Y, col in ([(Tg, grey)] if Tg is not None else []) + [(X, dark)]:
            Q = f(Y[:, ax])
            for a, b in E[::1]:
                cv.line(Q[a], Q[b], col)
        cv.text(ci * W + 6, 4, name + ('  grey: target' if Tg is not None else ''))
    zs = np.linspace(X[:, 2].min(), X[:, 2].max(), heights + 2)[1:-1] + 1e-4 * np.ptp(X[:, 2])
    f, _ = _fit(np.vstack([X[:, :2]] + ([Tg[:, :2]] if Tg is not None else [])), 2 * W, 0)
    for z in zs:
        for Y, col in ([(Tg, grey)] if Tg is not None else []) + [(X, dark)]:
            C = _cut(Y, E, Y[:, 2], z)
            if len(C):
                cv.dots(f(C[:, :2]), col)
        if Tg is not None:
            a, b = _cut(X, E, X[:, 2], z), _cut(Tg, E, Tg[:, 2], z)
            if len(a) and len(b):
                d = np.min(np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2), 1)
                nums['heights'].append(dict(z=round(float(z), 4), off_median=round(float(np.median(d)), 4),
                                            off_max=round(float(d.max()), 4)))
    cv.text(2 * W + 6, 4, 'sections at %d heights (top view)' % heights)

    # row 2: the pattern's strain and bend
    L0 = np.linalg.norm(F.flat[E[:, 0]] - F.flat[E[:, 1]], axis=1)
    L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    strain_e = L / np.maximum(L0, 1e-12) - 1
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(E)}
    tri_e = np.array([[key[tuple(sorted((int(t[i]), int(t[(i + 1) % 3]))))] for i in range(3)] for t in T])
    worst = strain_e[tri_e][np.arange(len(T)), np.argmax(np.abs(strain_e[tri_e]), 1)]
    f, _ = _fit(P, 0, H)
    Q = f(P)
    for k, t in enumerate(T):
        cv.tri(Q[t[0]], Q[t[1]], Q[t[2]], _ramp(abs(worst[k]), 0.0, 0.03, a=(60, 110, 200), mid=(240, 240, 232)))
    cv.text(6, H + 4, 'strain on the pattern (blue 0, red 3%%+): worst %.1f%%' % (100 * np.abs(worst).max()))
    Hall = F.H_all
    ang = np.degrees(np.abs(_dih(X, Hall)))
    mine = set()
    for Hc in F._H:
        mine |= {tuple(sorted(r)) for r in Hc[:, 2:].tolist()}
    is_c = np.array([tuple(sorted(r)) in mine for r in Hall[:, 2:].tolist()])
    f, _ = _fit(P, W, H)
    Q = f(P)
    for k in np.argsort(ang):
        a, b = Hall[k, 2], Hall[k, 3]
        cv.line(Q[a], Q[b], dark if is_c[k] else _ramp(ang[k], 0, 30, a=(205, 205, 210), mid=(235, 170, 60)),
                2 if is_c[k] else 1)
    stray = float(ang[~is_c].max()) if (~is_c).any() else 0.0
    cv.text(W + 6, H + 4, 'bend on the pattern (creases black): elsewhere at most %.0f deg' % stray)
    nums.update(strain_max_pct=round(100 * float(np.abs(worst).max()), 2),
                strain_p99_pct=round(100 * float(np.percentile(np.abs(worst), 99)), 2),
                bend_outside_creases_max_deg=round(stray, 1))

    # a panel of notes (the numbers)
    m = F.measure(X)
    for i, (c, (a, _)) in enumerate(zip(F.creases, m)):
        cv.text(2 * W + 6, H + 4 + 14 * i, '%s: %.1f of %.1f deg' % (c.name, a, math.degrees(abs(c.angle))))

    # a row per crease: sections across it at stations, and the angle along it
    for ri, (c, Hc, rest) in enumerate(zip(F.creases, F._H, F._rest)):
        y0 = H * (2 + ri)
        _, t, s = c.frame(P)
        # (stations along the crease's own edges, where the sheet is)
        mids = 0.5 * (P[Hc[:, 2]] + P[Hc[:, 3]])
        _, tm, _ = c.frame(mids)
        tt = np.linspace(tm.min(), tm.max(), stations + 2)[1:-1] + 0.137 * F.h     # (off the rows of points)
        sub = (2 * W) // stations
        reach = 6 * F.h + c.width(F.h)
        sec_nums = []
        for j, t0 in enumerate(tt):
            ox = j * sub
            Cx = _cut(X, E, t, t0)
            sv = _cut(s[:, None].repeat(3, 1), E, t, t0)[:, 0] if len(Cx) else np.zeros(0)
            keep = np.abs(sv) < reach
            # the crease's frame at the station: origin where s = 0, the fixed side's direction and normal
            if keep.sum() < 4:
                continue
            Cx, sv = Cx[keep], sv[keep]
            o = Cx[np.argmin(np.abs(sv))]
            fixed = Cx[sv < 0]
            if len(fixed) < 2:
                continue
            u = fixed[np.argmin(sv[sv < 0])] - o
            u /= max(np.linalg.norm(u), 1e-12)
            # the axis: the crease's points near the station
            on = np.unique(Hc[:, 2:])
            first = on[np.abs(s[on]) < 0.5 * F.h]                  # (where the turn begins: a rolled band's
            on = first if len(first) >= 2 else on                    # other rows run along it too, but further out)
            _, ton, _ = c.frame(P[on])
            near = on[np.argsort(np.abs(ton - t0))[:4]]
            if len(near) < 2:
                continue
            ax = np.linalg.svd(X[near] - X[near].mean(0))[2][0]
            tn = ton[np.argsort(np.abs(ton - t0))[:4]]
            if ((X[near] - X[near].mean(0)) @ ax) @ (tn - tn.mean()) < 0:
                ax = -ax                                             # (as the line runs, so every station reads alike)
            n = np.cross(ax, u)
            n /= max(np.linalg.norm(n), 1e-12)

            def plane(Y):
                return np.column_stack([(Y - o) @ (-u), (Y - o) @ n])
            pts = [plane(Cx)]
            Ct = None
            if Tg is not None:
                Ct = _cut(Tg, E, t, t0)
                st = _cut(s[:, None].repeat(3, 1), E, t, t0)[:, 0]
                kt = np.abs(st) < reach
                Ct, st = Ct[kt], st[kt]
                if len(Ct):
                    ot = Ct[np.argmin(np.abs(st))]
                    Ct = (Ct - ot + o)[np.argsort(st)]       # (in order across the fold, as cut)
                    pts.append(plane(Ct))
            f, _ = _fit(np.vstack(pts), ox, y0, w=sub)
            if Ct is not None and len(Ct):
                cv.poly(f(plane(Ct)), grey, 2)
            cv.poly(f(plane(Cx[np.argsort(sv)])), dark, 2)
            cv.dots(f(np.zeros((1, 2))), red)
            if Ct is not None and len(Ct):
                d = np.min(np.linalg.norm(Cx[:, None] - Ct[None], axis=2), 1)
                sec_nums.append(dict(station=round(float(t0), 4), off_median=round(float(np.median(d)), 4),
                                     off_max=round(float(d.max()), 4)))
        cv.text(6, y0 + 4, '%s: sections across it at %d stations (red: the crease)' % (c.name, stations))
        # the angle along it
        _, th, _ = c.frame(0.5 * (P[Hc[:, 2]] + P[Hc[:, 3]]))
        got = np.degrees(np.abs(_dih(X, Hc)))
        want = np.degrees(np.abs(rest)) * float(getattr(F, 'gain', np.ones(len(F.creases)))[ri])
        err = got - want
        tg = math.degrees(abs(c.angle))
        # the angle measured across the fold, segment by segment along it, against the target; and each hinge's
        # miss of its rest (a rolled band's hinges each hold a share, so their own angles are not the fold's)
        span = max(10.0, float(np.abs(err).max()) * 1.1)
        y_mid = y0 + 12 + (H - 12) / 2

        def f2(tv, v, scale):
            return np.column_stack([2 * W + PAD + (tv - th.min()) / max(np.ptp(th), 1e-9) * (W - 2 * PAD),
                                    y_mid - v / scale * ((H - 12) / 2 - PAD)])
        cv.line(f2(np.array([th.min()]), np.array([0.0]), 1)[0], f2(np.array([th.max()]), np.array([0.0]), 1)[0], red)
        cv.dots(f2(th, err, span), dark)
        seg = F.measure_segments(X)[ri] if hasattr(F, 'measure_segments') else []
        for (t_a, t_b, a_deg) in seg:
            q = f2(np.array([t_a, t_b]), np.array([a_deg - tg] * 2), span)
            cv.line(q[0], q[1], (40, 120, 60), 2)
        cv.text(2 * W + 6, y0 + 4, 'along it, deg off: hinges (dots), across the fold (green); +-%.0f' % span)
        a = m[ri][0]
        nums['creases'].append(dict(name=c.name, target_deg=round(tg, 2), measured_deg=round(a, 2),
                                    hinge_deg_min=round(float(got.min()), 1), hinge_deg_max=round(float(got.max()), 1)))
        nums['sections'].append(dict(name=c.name, stations=sec_nums))
    cv.save(path)
    with open(path.rsplit('.', 1)[0] + '.json', 'w') as fh:
        json.dump(nums, fh, indent=1)
    return nums


def _dih(X, H):
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'solvers', 'folds'))  # (appended)
    from folds import dihedral
    return dihedral(X, H)
