"""The petticoat's flounce: a straight strip of the same cotton, GATHER times as long as the line it is set on,
gathered onto the skirt FLOUNCE (draft.RATIOS) above the hem with a narrow heading standing above the seam. It is what
holds a petticoat's hem out and gives the habit over it its round fall.

It is drafted on the settled skirt, not solved again: each line of the flounce runs down the skirt's own surface
(from the seam to the hem, at that angle), stood off it by the cloth's thickness and by the gathers. A gathered
strip is pleated finely at the seam, where the thread draws it up, and opens lower down into fewer, rounder folds,
so the gathers are two waves round the skirt, the fine one fading and the broad one growing down the flounce, and at
every row their depth is solved so that the row is the strip's own length (GATHER times the seam's girth). They are
only ever outward of the skirt, so the flounce cannot pass through it, and it follows the skirt's bones point by point.

UVs are the strip's pattern: u along it (its full gathered length), v down it from the seam, in model units."""

import bpy
import numpy as np

import draft

GATHER = 1.6                 # the strip's length against the line it is set on (cotton flounces: 1.5 to 2)
GATHER_W = 0.085             # a gather at the seam, in H (about 2 cm)
OPEN = 4                     # the broad folds lower down: one to this many gathers
HEADING = 0.045              # the heading above the seam, in H (about 1 cm)
STAND = 0.6                  # the flounce stood off the skirt, in the cloth's thicknesses
BRIDGE_PASSES = 120          # the smoothing round the ring the flounce is read off (about 1.5 units of arc)
STEP = 0.12                  # the strip's sampling along its length, in units (a gather wants a few points)


def _rings(V, info):
    """For each row of the skirt: its points' angles (ascending, wrapped) and positions, to read a row at any angle."""
    out = []
    for idx in info['rows']:
        th = info['th'][idx] % (2 * np.pi)
        o = np.argsort(th)
        th, P = th[o], V[idx][o]
        out.append((np.concatenate([th[-1:] - 2 * np.pi, th, th[:1] + 2 * np.pi]),
                    np.concatenate([P[-1:], P, P[:1]])))
    return out


def _read(rings, ts, t, th):
    """Points on the skirt at fractions t down it and angles th (arrays of one shape)."""
    t = np.clip(t, 0.0, 1.0)
    i = np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2)
    f = (t - ts[i]) / (ts[i + 1] - ts[i])
    out = np.zeros(t.shape + (3,))
    for k in np.unique(i):
        m = i == k
        a = np.stack([np.interp(th[m], rings[k][0], rings[k][1][:, c]) for c in range(3)], -1)
        b = np.stack([np.interp(th[m], rings[k + 1][0], rings[k + 1][1][:, c]) for c in range(3)], -1)
        out[m] = a + (b - a) * f[m][:, None]
    return out


def _ring_length(P):
    return float(np.linalg.norm(P - np.roll(P, -1, 0), axis=1).sum())


def draft_flounce(V, info, M, pat, seed=11):
    """The flounce's points [rows, cols, 3], its UVs, and per row the fraction down it (q: 0 the seam, 1 the hem)."""
    H = M['H']
    D = draft.RATIOS['flounce'] * H
    ts = info['ts']
    rings = _rings(V, info)
    L = pat['length']
    th_tab = pat['th']
    rng = np.random.default_rng(seed)
    # the seam's girth on the skirt, and from it the strip's length and the number of gathers
    def t_at(q, th):
        Lq = np.interp(th, th_tab, L, period=2 * np.pi)
        return 1.0 - (1.0 - q) * D / Lq
    probe = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    seam = _read(rings, ts, t_at(0.0, probe), probe)
    G = _ring_length(seam)
    strip = GATHER * G
    n_g = OPEN * int(round(G / (GATHER_W * H) / OPEN))     # (whole broad folds round the ring: no break at the back)
    n_cols = int(round(strip / STEP))
    n_cols += (-n_cols) % 8
    th = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)
    # (each gather a little off its even place, as a hand draws them up)
    ph = n_g * th / (2 * np.pi) + 0.22 * np.sin(7 * th + rng.uniform(0, 6)) + 0.14 * np.sin(17 * th + rng.uniform(0, 6))
    # (each gather drawn up a little more or less than its neighbours)
    vary = 1.0 + 0.35 * np.sin(2 * np.pi * (ph * round(n_g / 2.7) / n_g + rng.uniform())) * np.sin(5 * th + rng.uniform(0, 6))
    fine = 0.5 * (1.0 + np.sin(2 * np.pi * ph)) * vary
    broad = 0.5 * (1.0 + np.sin(2 * np.pi * ph / OPEN + 0.9 * np.sin(3 * th + rng.uniform(0, 6))))
    stand = STAND * draft.RATIOS['thickness'] * H + 0.02
    qs = np.concatenate([np.linspace(-HEADING * H / D, 0.0, 3)[:-1], np.linspace(0.0, 1.0, int(round(D / 0.25)) + 1)])
    rows, bases, normals, waves, amps = [], [], [], [], []
    def bridged(P):
        # (the flounce is the outer, fuller layer: it bridges the skirt's deep valleys instead of diving into each,
        # so it is read off the skirt smoothed round the ring, and never inside the skirt's own surface)
        S = P.copy()
        for _ in range(BRIDGE_PASSES):
            S = 0.25 * np.roll(S, 1, 0) + 0.5 * S + 0.25 * np.roll(S, -1, 0)
        r0 = np.hypot(P[:, 0], P[:, 1])
        rs = np.maximum(np.hypot(S[:, 0], S[:, 1]), 1e-6)
        S[:, :2] *= (np.maximum(rs, r0) / rs)[:, None]
        return S

    for q in qs:
        t = t_at(max(q, 0.0), th)
        base = bridged(_read(rings, ts, t, th))
        if q < 0:            # (the heading: straight up from the seam, along the skirt's line above it)
            above = bridged(_read(rings, ts, t_at(0.0, th) - 0.02, th))
            d = above - base
            base = base + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9) * (-q * D)
        # the skirt's outward normal there
        e1 = np.roll(base, -1, 0) - np.roll(base, 1, 0)
        nxt = bridged(_read(rings, ts, np.clip(t + 0.01, 0, 1), th))
        prv = bridged(_read(rings, ts, np.clip(t - 0.01, 0, 1), th))
        e2 = nxt - prv
        n = np.cross(e2, e1)
        radial = np.stack([base[:, 0], base[:, 1], np.zeros(len(base))], 1)
        n *= np.sign((n * radial).sum(1, keepdims=True) + 1e-12)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        w = draft._ss(0.0, 0.7, max(q, 0.0))
        wave = (1.0 - w) * fine + w * broad
        lo, hi = 0.0, 0.05
        while _ring_length(base + n * (stand + hi * wave)[:, None]) < strip and hi < 20:
            hi *= 2
        for _ in range(40):
            A = 0.5 * (lo + hi)
            if _ring_length(base + n * (stand + A * wave)[:, None]) < strip:
                lo = A
            else:
                hi = A
        bases.append(base)
        normals.append(n)
        waves.append(wave)
        amps.append(0.5 * (lo + hi))
    # (the depth eased down the flounce, so the rows do not stand out as bands)
    a = np.array(amps)
    k = np.array([1, 2, 3, 2, 1], float)
    k /= k.sum()
    a = np.convolve(np.pad(a, 2, mode='edge'), k, mode='valid')
    for base, n, wave, A in zip(bases, normals, waves, a):
        P = base + n * (stand + A * wave)[:, None]
        P[:, 2] = np.maximum(P[:, 2], M['floor'] + 0.03)
        rows.append(P)
    R = np.array(rows)
    # (and the points eased across rows and round them, the gathers' tubes kept)
    for _ in range(2):
        R[1:-1] = 0.5 * R[1:-1] + 0.25 * (R[:-2] + R[2:])
    return R, qs, strip, D, n_g


def build(skirt, M, pat, info, rig, material, log=print):
    me = skirt.data
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', V)
    V = V.reshape(-1, 3)
    R, qs, strip, D, n_g = draft_flounce(V, info, M, pat)
    nr, nc, _ = R.shape
    verts = R.reshape(-1, 3)
    faces = [(r * nc + c, r * nc + (c + 1) % nc, (r + 1) * nc + (c + 1) % nc, (r + 1) * nc + c)
             for r in range(nr - 1) for c in range(nc)]
    fm = bpy.data.meshes.new('Petticoat flounce')
    fm.from_pydata([tuple(p) for p in verts], [], faces)
    for p in fm.polygons:
        p.use_smooth = True
    uv = fm.uv_layers.new(name='pattern')
    lo = np.empty(len(fm.loops), dtype=np.int64)
    fm.loops.foreach_get('vertex_index', lo)
    r_, c_ = lo // nc, lo % nc
    u = c_ / nc * strip
    # (the face that closes the ring at the centre back takes its far corners on round by the strip's length)
    fu = u.reshape(-1, 4)
    fu = np.where((fu.max(1, keepdims=True) - fu) > 0.5 * strip, fu + strip, fu)
    uv.data.foreach_set('uv', np.stack([fu.reshape(-1), qs[r_] * D], 1).reshape(-1))
    rest = fm.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', verts.reshape(-1))
    a = fm.attributes.new('down', 'FLOAT', 'POINT')
    a.data.foreach_set('value', np.repeat(qs, nc))
    o = bpy.data.objects.new('Petticoat flounce', fm)
    bpy.context.scene.collection.objects.link(o)
    fm.materials.append(material)
    # each point follows the skirt's bones where it lies on the skirt
    from mathutils.kdtree import KDTree
    kd = KDTree(len(V))
    for i, p in enumerate(V):
        kd.insert(p, i)
    kd.balance()
    near = np.array([kd.find(p)[1] for p in verts])
    gw = {g.index: g.name for g in skirt.vertex_groups}
    W = {n: np.zeros(len(V)) for n in gw.values()}
    for v in me.vertices:
        for g in v.groups:
            W[gw[g.group]][v.index] = g.weight
    for name, w in W.items():
        g = o.vertex_groups.new(name=name)
        ww = w[near]
        for i in np.nonzero(ww > 1e-4)[0]:
            g.add([int(i)], float(ww[i]), 'REPLACE')
    m = o.modifiers.new('rig', 'ARMATURE')
    m.object = rig
    log('petticoat flounce: %.2f deep (%.0f cm), strip %.1f long (%.1f m) on a seam %.1f round, %d gathers, %d points' % (
        D, D * 165 / 63, strip, strip * 165 / 63 / 100, strip / GATHER, n_g, len(verts)))
    return o
