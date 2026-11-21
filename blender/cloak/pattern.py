"""The cloak as a tailor cuts it: flat pattern pieces drafted from the figure's measures (measure.py), each seam edge
paired with the edge it is sewn to, and the cloth of each piece laid out from its flat shape.

A yoked circular cloak (the cut of a riding or travelling cloak):

  front yoke     cut 2 in wool, 2 in lining; straight grain on the centre front. Fitted over the collarbone and
                 the front of the shoulder, from the neck to the yoke line.
  back yoke      cut 1 on the fold (centre back), 1 in lining. Fitted over the top of the back and the shoulders.
  front skirt    cut 2 in wool, 2 in lining; straight grain on the centre-front edge, the flare thrown to the side
                 seam (which comes out on the bias). Slashed and spread from the hem to the yoke line, so it is
                 plain where it is sewn to the yoke and full at the hem.
  back skirt     cut 1 on the fold (centre back, straight grain), 1 in lining; spread as the fronts.
  stand collar   cut 1 on the fold in wool, 1 collar facing; interfaced. A curved band: its neck edge is the cloak's
                 neckline, its top edge the neck's girth that much higher, with ease.

Seams (each a pair of edges of the same length; report() checks every pair):

  shoulder       front yoke shoulder edge  = back yoke shoulder edge          (left and right)
  side           front skirt side edge     = back skirt side edge             (left and right; the shoulder
                                                                               seam's line, carried down the arm)
  yoke           front yoke lower edge     = front skirt top edge             (left and right)
                 back yoke lower edge      = back skirt top edge
  neck           collar neck edge          = the yokes' neck edges, CF to CF
  collar         collar top edge and ends  = collar facing's (bagged and turned)
  front edges    shell centre-front edges  = lining's (bagged: the lining is sewn to the shell at the fronts and
                                              the neck, and its hem is turned and slip-stitched to the shell's hem)
  hem            turned up by the hem allowance and stitched; the yoke seam, the front edges and the hem are
                 topstitched (the stitches are built as geometry, see sew.py)

The draft: the line the cloth takes down from the neck at every angle is measured (measure.meridian), the yoke
and skirt are laid on those lines, and each piece is flattened (developed) from that surface with its threads'
lengths kept (develop). The skirt pieces are then slashed and spread below the yoke line (spread) for fullness.
Everything is in model units; the sheet prints centimetres."""
import json
import math

import numpy as np

import measure as MS

CM = MS.CM
# Seam allowances, in H: 1.5 cm seams, 1 cm at the neck and the collar, 1 cm on the bagged front edges, a 4 cm
# turned hem (a heavy wool cloak's hem is deep, to hang well); none on the folds.
ALLOW = {'seam': 0.061, 'neck': 0.041, 'edge': 0.041, 'hem': 0.16, 'fold': 0.0}
FULLNESS = 1.55              # the skirt's hem, cut, over what it would be hanging plain (the folds take the rest)
FOLD_PITCH = 0.95            # H between hanging folds at the hem (where each panel's folds are counted from)
COLLAR_H = 0.22              # the stand collar's height, in H (about 5.5 cm)
COLLAR_EASE = 0.02           # its top stands off the neck by this much, in H
CF_GAP = 0.012               # the fronts meet edge to edge at the neck; this much apart, in radians, either side
SMOOTH = 6                   # passes of smoothing across the measured lines
STEP = 0.40                  # the cloth's mesh: about this far between threads, in units (about 1 cm)


# ---------------------------------------------------------------- the surface the cloak is drafted on
class Panel:
    """One piece's patch of the drafting surface: a grid of points S[i, j] (rows i down from its top edge, columns j
    across in angle round the neck), and once developed, its flat layout F[i, j]."""

    def __init__(self, name, cut, grain, edges, S, phis, rows_s):
        self.name, self.cut, self.grain = name, cut, grain
        self.edges = edges          # {'top': name, 'bottom': name, 'left': name, 'right': name} (left = column 0)
        self.S, self.phis, self.s = S, phis, rows_s
        self.F = None               # developed (before any spread)
        self.Fs = None              # as cut (after the spread)
        self.strain = None
        self.spread = 0.0
        self.folds = 0.0

    def edge_pts(self, side, flat=True):
        A = self.Fs if flat else self.S
        return {'top': A[0], 'bottom': A[-1], 'left': A[:, 0], 'right': A[:, -1]}[side]

    def outline(self):
        F = self.Fs
        return np.concatenate([F[0], F[1:, -1], F[-1, ::-1][1:], F[::-1, 0][1:]])


def _row_len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=-1).sum())


def surface(M, phi0, phi1, part, step=STEP, ncols=None, nrows=None):
    """The drafting surface between angles phi0 and phi1: the yoke (from the neck down YOKE along every meridian)
    or the skirt (from there to the hem). Returns (S, phis, s): points [rows, cols, 3], the columns' angles, and the
    distance down the meridian of every point."""
    H = M.H
    sy = MS.YOKE * H
    if ncols is None:
        # (enough columns that the widest row, the skirt's hem cut full, has threads STEP apart)
        ph = np.linspace(phi0, phi1, 64)
        hem = np.array([M.point(p, *M.meridian(p)[0][-1]) for p in ph])
        ncols = max(3, int(math.ceil(FULLNESS * _row_len(hem) / step)))
    phis = np.linspace(phi0, phi1, ncols + 1)
    spans = []
    for p in phis:
        P, s = M.meridian(p)
        a, b = (0.0, min(sy, s[-1] - 1.0)) if part == 'yoke' else (min(sy, s[-1] - 1.0), s[-1])
        spans.append((P, s, a, b))
    if nrows is None:
        nrows = max(3, int(math.ceil(max(b - a for _, _, a, b in spans) / (0.6 * step if part == 'yoke' else step))))
    S = np.zeros((nrows + 1, ncols + 1, 3))
    T = np.zeros((nrows + 1, ncols + 1))
    for j, (p, (P, s, a, b)) in enumerate(zip(phis, spans)):
        t = np.linspace(a, b, nrows + 1)
        r = np.interp(t, s, P[:, 0])
        z = np.interp(t, s, P[:, 1])
        d = MS._dir(p)
        S[:, j, 0] = M.axis[0] + r * d[0]
        S[:, j, 1] = M.axis[1] + r * d[1]
        S[:, j, 2] = z
        T[:, j] = t
    # (the lines are measured one by one; the cloth between them is smooth: a few passes of smoothing across,
    # the panel's edge lines kept where they are so the next panel meets them)
    for _ in range(SMOOTH):
        S[:, 1:-1] = 0.25 * S[:, :-2] + 0.5 * S[:, 1:-1] + 0.25 * S[:, 2:]
    return S, phis, T


# ---------------------------------------------------------------- developing a piece flat
def _grid_edges(nr, nc):
    idx = np.arange(nr * nc).reshape(nr, nc)
    E = [np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),        # across
         np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1),        # down (the meridians)
         np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], 1),     # the two bias lines
         np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], 1)]
    w = [1.0, 1.0, 0.35, 0.35]
    return E, w


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def unwrap(S, method='MINIMUM_STRETCH'):
    """Blender's own unwrap of the patch (SLIM, minimum stretch, where there is one), scaled to its true size:
    the first layout develop() relaxes from. None outside Blender."""
    try:
        import bpy
        import bmesh
    except ImportError:
        return None
    nr, nc = S.shape[:2]
    idx = np.arange(nr * nc).reshape(nr, nc)
    faces = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    me = bpy.data.meshes.new('cloak develop')
    me.from_pydata(S.reshape(-1, 3).tolist(), [], faces.tolist())
    ob = bpy.data.objects.new('cloak develop', me)
    bpy.context.scene.collection.objects.link(ob)
    prev = bpy.context.view_layer.objects.active
    for o in bpy.context.view_layer.objects.selected:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    me.uv_layers.new(name='flat')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    try:
        bpy.ops.uv.unwrap(method=method, margin=0.0)
    except TypeError:
        bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    uv = np.zeros((nr * nc, 2))
    for loop in me.loops:
        uv[loop.vertex_index] = me.uv_layers['flat'].data[loop.index].uv
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    if prev is not None:
        bpy.context.view_layer.objects.active = prev
    X = S.reshape(-1, 3)
    E, _ = _grid_edges(nr, nc)
    l3 = sum(np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1).sum() for e in E[:2])
    l2 = sum(np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1).sum() for e in E[:2])
    return (uv * (l3 / max(l2, 1e-12))).reshape(nr, nc, 2)


def develop(S, iters=1500):
    """Lay the patch S flat keeping the length of every thread across and down it (and, more loosely, on the
    bias): what a piece of cloth does when it is taken off the body and laid on the table. A doubly curved patch
    (over the shoulder) cannot keep them all; the strain left is reported, and is what the cloth's give and the
    yoke's shaping take up. Returns (F, strain): the flat points [rows, cols, 2] and each thread's (flat - 3D) / 3D."""
    nr, nc = S.shape[:2]
    X = S.reshape(-1, 3)
    E, w = _grid_edges(nr, nc)
    L = [np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1) for e in E]
    # a first layout, marched row by row as a cutter unrolls a cone: the top row laid out straight along its own
    # length, then every point of the next row placed from the two above it by its distances to them (exact on
    # a developable patch; on a curved one the error is left for the relaxing below to spread out)
    F0 = unwrap(S)
    if F0 is not None:
        F = F0.reshape(-1, 2)
        return _relax(F, X, E, w, nr, nc, iters)
    F = np.zeros((nr, nc, 2))
    # (the first strip, between the top two rows, unfolded triangle by triangle: each new point from the last two,
    # on the far side of them from the one before)
    order = [(0, 0), (1, 0)] + [(r, j) for j in range(1, nc) for r in (0, 1)]
    F[1, 0] = (0.0, -np.linalg.norm(S[1, 0] - S[0, 0]))
    for t in range(2, len(order)):
        (ra, ja), (rb, jb), (rc, jc) = order[t - 2], order[t - 1], order[t]
        A, B = F[ra, ja], F[rb, jb]
        a = np.linalg.norm(S[rc, jc] - S[ra, ja])
        b = np.linalg.norm(S[rc, jc] - S[rb, jb])
        AB = B - A
        dd = np.linalg.norm(AB) + 1e-12
        u = AB / dd
        x_ = (a * a - b * b + dd * dd) / (2 * dd)
        y_ = math.sqrt(max(a * a - x_ * x_, 1e-12))
        n = np.array([-u[1], u[0]])
        P1, P2 = A + x_ * u + y_ * n, A + x_ * u - y_ * n
        if t >= 3:
            Cp = F[order[t - 3]]
            # (the far side from the point before the last two)
            side = np.sign(_cross(AB, Cp - A))
            F[rc, jc] = P1 if np.sign(_cross(AB, P1 - A)) != side else P2
        else:
            F[rc, jc] = P1 if P1[0] > P2[0] else P2
    for i in range(1, nr - 1):
        for j in range(nc):
            k = j + 1 if j + 1 < nc else j - 1
            A, B = F[i, j], F[i, k]
            a = np.linalg.norm(S[i + 1, j] - S[i, j])
            b = np.linalg.norm(S[i + 1, j] - S[i, k])
            AB = B - A
            dd = np.linalg.norm(AB) + 1e-12
            u = AB / dd
            x_ = (a * a - b * b + dd * dd) / (2 * dd)
            y_ = math.sqrt(max(a * a - x_ * x_, 1e-12))
            n = np.array([u[1], -u[0]]) if k > j else np.array([-u[1], u[0]])
            F[i + 1, j] = A + x_ * u + y_ * n
    F = F.reshape(-1, 2)
    return _relax(F, X, E, w, nr, nc, iters)


def _relax(F, X, E, w, nr, nc, iters):
    L = [np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1) for e in E]
    # then relaxed: every thread pulled to its length (projected, a few thousand sweeps; the bias lines softer)
    deg = np.zeros(len(F))
    for e, ww in zip(E, w):
        np.add.at(deg, e.ravel(), ww)
    for it in range(iters):
        D = np.zeros_like(F)
        for e, l0, ww in zip(E, L, w):
            v = F[e[:, 1]] - F[e[:, 0]]
            l = np.linalg.norm(v, axis=1) + 1e-12
            c = (ww * 0.5 * (l - l0) / l)[:, None] * v
            np.add.at(D, e[:, 0], c)
            np.add.at(D, e[:, 1], -c)
        F += 1.6 * D / deg[:, None]
    strain = []
    for e, l0 in zip(E[:2], L[:2]):
        l = np.linalg.norm(F[e[:, 1]] - F[e[:, 0]], axis=1)
        strain.append((l - l0) / l0)
    return F.reshape(nr, nc, 2), np.concatenate(strain)


def _orient(F, grain_col):
    """Turn the flat piece so its grain (the given column, top to bottom) runs straight down the page."""
    v = F[-1, grain_col] - F[0, grain_col]
    a = math.atan2(v[0], -v[1])
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, s], [-s, c]])
    G = (F - F[0, grain_col]) @ R.T
    return G


def spread(F, grain_col, k):
    """Slash and spread: every column below the top row turned about its top point, by k radians for each column
    it lies from the grain column, so the piece keeps its top edge (sewn to the yoke) and its length down every
    line, and gains k times the length below for each column of width at the hem. The grain column stays put."""
    nr, nc = F.shape[:2]
    G = F.copy()
    for j in range(nc):
        a = k * (j - grain_col)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        G[1:, j] = (F[1:, j] - F[0, j]) @ R.T + F[0, j]
    return G


# ---------------------------------------------------------------- the draft
def draft(M, log=print):
    """Draft the cloak on the figure measured as M. Returns (panels, seams): the pieces of her right side and the
    back halves (the left side is their mirror; the back pieces are cut on the fold), and the seam pairs."""
    H = M.H
    ps = M.phi_s
    cf = CF_GAP
    parts = {}
    # (her right half: the fronts from the centre front to the shoulder seam, the backs from it to the centre back)
    for name, a, b, part in (('front yoke', cf, ps, 'yoke'), ('back yoke', ps, math.pi, 'yoke'),
                             ('front skirt', cf, ps, 'skirt'), ('back skirt', ps, math.pi, 'skirt')):
        parts[name] = (a, b, part)
    # the yoke and the skirt below it share their columns, so the yoke seam pairs thread for thread
    # (and front and back the same number of rows, so the shoulder and side seams pair too)
    sy = MS.YOKE * H
    longest = max(M.meridian(p)[1][-1] for p in np.linspace(cf, math.pi, 61))
    rows_skirt = max(3, int(math.ceil((longest - sy) / STEP)))
    rows_yoke = max(3, int(math.ceil(sy / (0.6 * STEP))))
    cols = {}
    for side in ('front', 'back'):
        a, b, _ = parts[side + ' skirt']
        S, phis, T = surface(M, a, b, 'skirt', nrows=rows_skirt)
        cols[side] = len(phis) - 1
        parts[side + ' skirt'] = (S, phis, T)
        Sy, phy, Ty = surface(M, a, b, 'yoke', ncols=cols[side], nrows=rows_yoke)
        parts[side + ' yoke'] = (Sy, phy, Ty)
    E = {'front yoke': dict(top='neck', bottom='yoke', left='cf', right='shoulder'),
         'back yoke': dict(top='neck', bottom='yoke', left='shoulder', right='cb'),
         'front skirt': dict(top='yoke', bottom='hem', left='cf', right='side'),
         'back skirt': dict(top='yoke', bottom='hem', left='side', right='cb')}
    cut = {'front yoke': '2 wool, 2 lining', 'back yoke': '1 wool, 1 lining, on the fold',
           'front skirt': '2 wool, 2 lining', 'back skirt': '1 wool, 1 lining, on the fold'}
    panels = {}
    for name, (S, phis, T) in parts.items():
        grain = 0 if name.startswith('front') else S.shape[1] - 1
        p = Panel(name, cut[name], grain, E[name], S, phis, T)
        F, st = develop(S)
        p.F = _orient(F, grain)
        p.strain = st
        p.Fs = p.F
        panels[name] = p
    # the fullness: each skirt spread so its hem is FULLNESS times its plain hang, the same share front and back
    for side in ('front', 'back'):
        p = panels[side + ' skirt']
        nc = p.F.shape[1]
        w0 = _row_len(p.F[-1])
        lo, hi = 0.0, 0.05
        sign = 1.0
        for s_ in (1.0, -1.0):
            if _row_len(spread(p.F, p.grain, s_ * 0.002)[-1]) > w0:
                sign = s_
                break
        for _ in range(50):
            k = 0.5 * (lo + hi)
            if _row_len(spread(p.F, p.grain, sign * k)[-1]) < FULLNESS * w0:
                lo = k
            else:
                hi = k
        p.spread = sign * 0.5 * (lo + hi)
        p.Fs = spread(p.F, p.grain, p.spread)
        hem3 = _row_len(p.S[-1])
        # its hanging folds, counted from the hem as cut
        n = _row_len(p.Fs[-1]) / (FOLD_PITCH * H)
        p.folds = max(1.0, round(n)) if side == 'front' else max(0.5, round(2 * n) / 2.0)
        p.hem_plain = hem3
    # the collar: a band round the neck from one front edge to the other, on the neckline's columns
    yk = {s: panels[s + ' yoke'] for s in ('front', 'back')}
    neck = np.concatenate([yk['front'].S[0], yk['back'].S[0][1:]])
    phis = np.concatenate([yk['front'].phis, yk['back'].phis[1:]])
    nrow = max(3, int(round(COLLAR_H * H / (0.6 * STEP))))
    C = np.zeros((nrow + 1, len(phis), 3))
    for j, (p, n0) in enumerate(zip(phis, neck)):
        d = MS._dir(p)
        z0 = n0[2]
        r0 = float(np.hypot(*(n0[:2] - M.axis)))
        zt = z0 + COLLAR_H * H
        rt = M.radius(p, zt) + (MS.NECK_EASE + COLLAR_EASE) * H
        for i in range(nrow + 1):
            u = i / nrow
            z = z0 + COLLAR_H * H * u
            # (a cone from the neckline up to its top edge, never closer to the neck than its ease)
            r = max(r0 + (rt - r0) * u, M.radius(p, z) + (MS.NECK_EASE + COLLAR_EASE * u) * H)
            C[i, j] = (M.axis[0] + r * d[0], M.axis[1] + r * d[1], z)
        C[0, j] = n0
    cp = Panel('collar', '1 wool on the fold, 1 facing on the fold, 1 interfacing', len(phis) - 1,
               dict(top='neck', bottom='collar top', left='cf', right='cb'), C, phis, None)
    F, st = develop(C)
    cp.F = cp.Fs = _orient(F, len(phis) - 1)
    cp.strain = st
    panels['collar'] = cp
    seams = [
        ('shoulder', ('front yoke', 'right'), ('back yoke', 'left')),
        ('side', ('front skirt', 'right'), ('back skirt', 'left')),
        ('yoke (front)', ('front yoke', 'bottom'), ('front skirt', 'top')),
        ('yoke (back)', ('back yoke', 'bottom'), ('back skirt', 'top')),
        ('neck (front)', ('collar', 'top'), ('front yoke', 'top')),
        ('neck (back)', ('collar', 'top'), ('back yoke', 'top')),
    ]
    return panels, seams


def report(panels, seams, M, log=print):
    H = M.H
    log('cloak pattern (her right side; the back pieces on the fold), in cm:')
    out = {'pieces': {}, 'seams': {}}
    for p in panels.values():
        o = p.outline()
        w, h = np.ptp(o[:, 0]) * CM, np.ptp(o[:, 1]) * CM
        st = np.abs(p.strain)
        line = '  %-12s %-44s %5.1f x %5.1f  strain mean %.1f%% max %.1f%%' % (
            p.name, p.cut, w, h, 100 * st.mean(), 100 * st.max())
        if p.name.endswith('skirt'):
            line += '  spread %.2f deg a column, hem %.1f cut / %.1f plain, %.1f folds' % (
                math.degrees(p.spread), _row_len(p.Fs[-1]) * CM, p.hem_plain * CM, p.folds)
        log(line)
        out['pieces'][p.name] = dict(w_cm=w, h_cm=h, strain_mean=float(st.mean()), strain_max=float(st.max()))
    # every seam: its two edges' lengths, flat (as cut)
    for nm, (pa, ea), (pb, eb) in seams:
        A = panels[pa].edge_pts(ea)
        B = panels[pb].edge_pts(eb)
        if pa == 'collar':
            # (the collar's neck edge runs the whole neckline: the front part, then the back)
            nf = panels['front yoke'].S.shape[1]
            A = A[:nf] if 'front' in nm else A[nf - 1:]
        la, lb = _row_len(A) * CM, _row_len(B) * CM
        log('  seam %-13s %s %s %.1f = %s %s %.1f  (%+.1f%%)' % (nm, pa, ea, la, pb, eb, lb, 100 * (la - lb) / lb))
        out['seams'][nm] = (la, lb)
    return out


# ---------------------------------------------------------------- the sheet
def _offset(P, d, closed=True):
    """A polygon let out by d (miters at the corners; P runs either way round)."""
    P = np.asarray(P, float)
    if closed:
        A = np.roll(P, 1, 0)
        B = np.roll(P, -1, 0)
    else:
        A = np.vstack([P[:1] * 2 - P[1:2], P[:-1]])
        B = np.vstack([P[1:], P[-1:] * 2 - P[-2:-1]])
    t1 = P - A
    t2 = B - P
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12
    n1 = np.stack([t1[:, 1], -t1[:, 0]], 1)
    n2 = np.stack([t2[:, 1], -t2[:, 0]], 1)
    n = n1 + n2
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    cosh = np.clip((n * n1).sum(1), 0.3, 1.0)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    sgn = 1.0 if area < 0 else -1.0
    return P + sgn * (d / cosh)[:, None] * n


def cut_line(p, H):
    """The cutting line round a piece: each edge let out by its own allowance (the folds by none)."""
    F = p.Fs
    kinds = {'neck': 'neck', 'yoke': 'seam', 'shoulder': 'seam', 'side': 'seam', 'cf': 'edge', 'cb': 'fold',
             'hem': 'hem', 'collar top': 'neck'}
    # (each side of the outline, offset by its allowance; the corners where two sides meet are joined)
    sides = [('top', F[0]), ('right', F[:, -1]), ('bottom', F[-1, ::-1]), ('left', F[::-1, 0])]
    pts = []
    out = p.outline()
    area = 0.5 * np.sum(out[:, 0] * np.roll(out[:, 1], -1) - np.roll(out[:, 0], -1) * out[:, 1])
    for side, P in sides:
        d = ALLOW[kinds[p.edges[side]]] * H
        t = np.gradient(P, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        n = np.stack([t[:, 1], -t[:, 0]], 1) * (1.0 if area > 0 else -1.0)
        pts.append(P + d * n)
    return np.concatenate(pts + [pts[0][:1]])


def layout(panels, seams, M, path, title='Cloak'):
    """The pattern laid out flat, in centimetres, as a cutting sheet: seam lines solid, cutting lines dashed, the
    grain arrow, fold edges marked, each edge labelled with what it is sewn to, and the yoke line on the skirt."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H = M.H
    pair = {}
    for nm, (pa, ea), (pb, eb) in seams:
        pair[(pa, ea)] = '%s (%s)' % (pb, nm)
        pair[(pb, eb)] = '%s (%s)' % (pa, nm)
    order = ['back skirt', 'front skirt', 'back yoke', 'front yoke', 'collar']
    fig = plt.figure(figsize=(22, 13))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.9])
    x0 = 0.0
    y_top = {'back skirt': 0, 'front skirt': 0, 'back yoke': 0, 'front yoke': -70, 'collar': -140}
    col = {'back skirt': 0, 'front skirt': 1, 'back yoke': 2, 'front yoke': 2, 'collar': 2}
    xs = {}
    for nm in order:
        p = panels[nm]
        c = col[nm]
        if c not in xs:
            xs[c] = x0
            w = max(np.ptp(panels[k].Fs[..., 0].ravel()) for k in order if col[k] == c) * CM
            x0 += w + 25
        o = p.outline() * CM
        dx = xs[c] - o[:, 0].min() + 5
        dy = y_top[nm] - o[:, 1].max()
        T = lambda P: np.asarray(P) * CM + [dx, dy]  # noqa: E731
        oo = T(p.outline())
        ax.fill(oo[:, 0], oo[:, 1], color='#efe9f2', zorder=0)
        ct = T(cut_line(p, H))
        ax.plot(ct[:, 0], ct[:, 1], '--', color='#7a6a80', lw=0.9)
        F = p.Fs
        for side, P in (('top', F[0]), ('bottom', F[-1]), ('left', F[:, 0]), ('right', F[:, -1])):
            e = p.edges[side]
            PP = T(P)
            fold = e == 'cb'
            ax.plot(PP[:, 0], PP[:, 1], '-', color='#3c2a46' if not fold else '#a0522d', lw=2.0 if not fold else 2.6)
            m = PP[len(PP) // 2]
            lab = e + (' (fold)' if fold else '')
            if (p.name, side) in pair:
                lab += '\n= ' + pair[(p.name, side)]
            if p.name == 'collar' and side == 'top':
                lab = 'neck edge\n= yokes\' neck edges (neck)'
            ax.text(m[0], m[1], lab, fontsize=7.5, ha='center', va='center', color='#2a2030',
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=1))
        # the grain: along the grain column
        g = T(F[:, p.grain])
        a, b = g[int(0.2 * len(g))], g[int(0.8 * len(g))]
        if p.name == 'collar':
            a, b = T(F[0, p.grain]), T(F[-1, p.grain])
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='<->', color='#444', lw=1.2))
        mid = T(np.mean(F.reshape(-1, 2), 0))
        st = np.abs(p.strain)
        info = '%s\ncut %s\n%.0f x %.0f cm' % (p.name.upper(), p.cut, np.ptp(oo[:, 0]), np.ptp(oo[:, 1]))
        if p.name.endswith('skirt'):
            info += '\nspread %.1f%% at the hem, %.1f folds' % (100 * (FULLNESS - 1), p.folds)
        info += '\nthreads kept to %.1f%% (max %.1f%%)' % (100 * st.mean(), 100 * st.max())
        ax.text(mid[0], mid[1], info, fontsize=8.5, ha='center', va='center', color='#1a1420')
    ax.set_aspect('equal')
    ax.grid(True, color='#e4e0e8', lw=0.5)
    ax.set_xlabel('cm')
    allow = ', '.join('%s %.1f cm' % (k, v * H * CM) for k, v in ALLOW.items() if v > 0)
    ax.set_title('%s: flat pattern drafted from the measured body (her right side; backs on the fold). '
                 'Solid: seam line. Dashed: cutting line (allowances: %s). Arrows: straight grain. Brown: fold.'
                 % (title, allow), fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save(panels, seams, M, path, extra=None):
    d = dict(H=M.H, units_to_cm=CM, allowances_H=ALLOW, fullness=FULLNESS, pieces={}, seams=[])
    for p in panels.values():
        d['pieces'][p.name] = dict(cut=p.cut, edges=p.edges, grain_column=int(p.grain),
                                   seam_line=(p.outline() * CM).round(2).tolist(),
                                   cut_line=(cut_line(p, M.H) * CM).round(2).tolist(),
                                   spread_deg_per_column=math.degrees(p.spread), folds=p.folds)
    d['seams'] = [dict(name=n, a=list(a), b=list(b)) for n, a, b in seams]
    if extra:
        d.update(extra)
    with open(path, 'w') as fh:
        json.dump(d, fh, indent=1)
