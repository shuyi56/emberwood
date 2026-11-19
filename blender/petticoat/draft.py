"""The petticoat drafted on whatever body it is given: a white cotton underskirt of about 1890, worn over the corset
under a closed riding-habit skirt, that gives the habit its A-line and its hang.

Nothing is placed by hand. The body is measured first (measure()): the head's height H, the waist, the hips, the
seat and the floor, read off the skin's rest shape, and every size of the petticoat is a share of H (RATIOS), so it
re-fits when the body or the head changes. The waist is not the bare body's: the band sits on the corset, whose
compressed section comes from the shared module habit.py (owned by the corset thread): the body is drawn in by
habit.compress_body, the corset is put on it, and the petticoat is measured over both, so the band sits on the
corset's own surface and moves when habit's numbers do.

How a petticoat of 1885-1895 is cut, and how that is followed here:

  the band     a straight band about 3.5 cm deep at the corseted waist, closed at the centre back.
  the yoke     the front and side gores are darted to lie smooth over the corset and the hips; the back breadth is
               gathered into the back of the band (the last of the bustle's fullness). Here: the corset's outer
               section, or the body below it, let out by the ease, and pulled taut over hollows (a concave envelope
               down each line of the cloth), so it bridges the small of the back and lies on every convex form.
  the gores    from the fullest hip straight down to the hem, flaring little at the front, more at the sides and most
               at the back, to a hem of about 2.7 m (HEM, a share of the length): the pattern of the gored skirts of the time. The
               back runs on into a short train that lies on the floor (TRAIN); the front clears the floor.
  the flounce  a deep gathered flounce set on above the hem (flounce.py), headed by tucks, that holds the hem out.

The garment is made in three steps:

  1. the pattern  r(t, theta): the radius from the body's axis of each line of the cloth (theta round from the front,
                  t from the band to the hem), with the girth of every row as cut. pattern() returns it.
  2. the hang     the same rows hung from the hip plumb, each drawn in to where cloth that full would hang, its spare
                  girth put into a few deep flutes (more of them, closer, at the back where the breadth is gathered),
                  so that every row keeps the pattern's girth: the cloth's rest lengths are the pattern's. hang()
                  builds that mesh; the solver (cloth.py) then lets it fall on the body and the floor.
  3. the rig      the band and yoke follow the pelvis; lower down each side follows its thigh a little, more toward
                  the hem, so the skirt moves with the legs without being cut by them.

Rows are rings of their own length (more points round the hem than round the band, about GRID apart), joined by
strips of triangles, so the cloth is evenly sampled from the band to the hem. The pattern is the UV map: u the
distance round the row as cut, v the distance down the cloth, both in model units, so the weave runs true."""
import os

import numpy as np

# Every size as a share of the head's height H (9.39 units on the mannequin; one unit is about 2.6 cm).
RATIOS = {
    'band': 0.145,           # the waistband's depth (about 3.5 cm)
    'band_ease': 0.010,      # the band over the corset (about 2.5 mm)
    'ease': 0.018,           # the yoke over the corset and hips (about 4 mm)
    'thickness': 0.004,      # the cotton's thickness (about 1 mm)
    'flare_from': 0.10,      # the gores start to flare this far above the fullest hip
    'clear': 0.09,           # the hem clears the floor at the front by this much (about 2 cm: under the habit's hem)
    'train': 0.7,            # the back is this much longer than it needs to reach the floor (about 17 cm)
    'flounce': 1.05,         # the flounce's depth (about 26 cm), set on this far above the hem
    'flute': 0.42,           # a flute's width at the hem, as the cloth hangs (about 10 cm)
}
# The hem's girth as cut, as a share of the skirt's length (the corseted waist to the floor): the gores' angle, so a
# taller or shorter body keeps the same line. A petticoat of 1890 was 2.5 to 3 m round on a skirt about 1.05 m long,
# a little less than the dress over it.
HEM = 2.55
# How the gores' flare is shared round the skirt (front, side, back): the front falls almost plumb, the back breadth
# is widest. The share at an angle theta from the front is interpolated between these.
FLARE_SHARE = (0.45, 1.0, 1.55)
# How far the cloth that full hangs out from the hip plumb, as a share of the pattern's flare: the rest of the girth
# goes into flutes. (Cotton of this weight carries about a third of its cut flare; the solver settles the rest.)
HANG = 0.38
GRID = float(os.environ.get('PETTI_GRID', '0.36'))   # the cloth's sampling, in units (about 1 cm)
NTH = 720                    # angles the profile is tabled at
SEED = 7


def _ss(a, b, x):
    t = np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- the body

def _hull(pts):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    p = np.unique(np.round(np.asarray(pts, dtype=float), 6), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def half(q):
        h = []
        for x in q:
            while len(h) >= 2 and cross(h[-1] - h[-2], x - h[-2]) <= 0:
                h.pop()
            h.append(x)
        return h
    lo, hi = half(p), half(p[::-1])
    return np.array(lo[:-1] + hi[:-1])


def ray_radius(poly, th):
    """The distance from the axis (0, 0) to a closed polygon, along the directions at angles th from the front (-y),
    toward her left (+x) first. The farthest crossing, so a hull that misses the axis still reads sensibly."""
    d = np.stack([np.sin(th), -np.cos(th)], 1)
    a = poly
    b = np.roll(poly, -1, 0)
    e = b - a
    # solve a + u e = s d   for every (direction, edge)
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    s = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
    u = (a[None, :, 0] * d[:, None, 1] - a[None, :, 1] * d[:, None, 0]) / den
    ok = (u >= -1e-9) & (u <= 1 + 1e-9) & (s > 0)
    return np.where(ok, s, 0.0).max(1)


def girth(r, th=None):
    """The length round a closed ring given as radii at evenly spaced angles."""
    n = len(r)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) if th is None else th
    x, y = r * np.sin(th), -r * np.cos(th)
    return float(np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0]))).sum())


def measure(P, W, groups, H, floor=None, extra=None):
    """The tailor's measurements of the body for a skirt, from the skin's rest points P, its bone weights W [point,
    group] and their group names, in model units:

    H            the head's height, the unit
    floor        the floor (the lowest point of the feet)
    waist_z, waist   the narrowest section of the trunk above the hips, and its girth (natural, uncorseted)
    hips_z, hips     the fullest section of the hips (and seat), and its girth
    seat_z       where the seat stands furthest back
    section(z)   the trunk's convex section at z as a polygon (the arms left out; below the crotch both legs)
    r_body(z, th)    the radius of that section along the angles th"""
    col = {n: i for i, n in enumerate(groups)}
    arm = sum(W[:, col[n]] for n in groups if n.split('.')[0] in ('UpperArm', 'Forearm', 'Hand'))
    T = P[arm < 0.3]
    if extra is not None and len(extra):
        T = np.concatenate([T, extra])          # (what is worn under it: the corset)
    M = {'H': float(H), 'floor': float(P[:, 2].min() if floor is None else floor)}
    cache = {}

    def section(z, dz=0.09):
        k = round(float(z), 3)
        if k not in cache:
            m = np.abs(T[:, 2] - z) < dz
            cache[k] = _hull(T[m][:, :2]) if m.sum() >= 3 else np.zeros((0, 2))
        return cache[k]

    def r_body(z, th):
        poly = section(z)
        return ray_radius(poly, th) if len(poly) >= 3 else np.zeros_like(th)
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    zs = np.arange(M['floor'] + 0.3 * (T[:, 2].max() - M['floor']), T[:, 2].max() - 1.2 * H, 0.2)
    g = np.array([girth(r_body(z, th), th) for z in zs])
    # the fullest hips: the widest section in the lower trunk; the waist: the narrowest above it, below the bust
    lo = zs < zs[0] + 0.55 * (zs[-1] - zs[0])
    i = int(np.argmax(np.where(lo, g, -1)))
    M['hips_z'], M['hips'] = float(zs[i]), float(g[i])
    up = (zs > M['hips_z'] + 0.4 * H) & (zs < M['hips_z'] + 1.6 * H)
    j = int(np.argmin(np.where(up, g, np.inf)))
    M['waist_z'], M['waist'] = float(zs[j]), float(g[j])
    back = [r_body(z, np.array([np.pi]))[0] for z in zs[lo]]
    M['seat_z'] = float(zs[lo][int(np.argmax(back))])
    M['section'], M['r_body'] = section, r_body
    return M


def report(M, log=print):
    cm = 165.0 / 63.0
    H = M['H']
    log('petticoat measures: H %.2f, waist %.2f H (%.0f cm) at %.2f, hips %.2f H (%.0f cm) at %.2f, seat at %.2f, '
        'floor %.2f, waist to floor %.2f H (%.0f cm)' % (
            H, M['waist'] / H, M['waist'] * cm, M['waist_z'], M['hips'] / H, M['hips'] * cm, M['hips_z'],
            M['seat_z'], M['floor'], (M['waist_z'] - M['floor']) / H, (M['waist_z'] - M['floor']) * cm))


# ---------------------------------------------------------------- 1. the pattern

def _share(th, share=FLARE_SHARE):
    """The flare's share at angles th from the front: front, side, back, eased between."""
    a = np.abs(((th + np.pi) % (2 * np.pi)) - np.pi) / np.pi          # 0 front, 0.5 side, 1 back
    f, s, b = share
    return np.where(a < 0.5, f + (s - f) * _ss(0.0, 0.5, a), s + (b - s) * _ss(0.5, 1.0, a))


def _concave_down(r, z):
    """Each line of cloth pulled taut: the least concave profile r(z) over the body's (z descending), so the cloth
    bridges hollows and lies on convex forms. r [nz, nth], z [nz] descending."""
    out = r.copy()
    for j in range(r.shape[1]):
        # upper hull of points (z, r) seen from outside: a concave envelope by monotone chain
        pts = list(zip(z, r[:, j]))
        h = []
        for p in pts:
            while len(h) >= 2:
                (z1, r1), (z2, r2) = h[-2], h[-1]
                # keep h[-1] only if it lies above the chord from h[-2] to p
                t = (z2 - z1) / (p[0] - z1) if p[0] != z1 else 0.0
                if r2 >= r1 + t * (p[1] - r1):
                    break
                h.pop()
            h.append(p)
        hz = np.array([q[0] for q in h])
        hr = np.array([q[1] for q in h])
        out[:, j] = np.interp(-z, -hz, hr)
    return out


def pattern(M, ratios=RATIOS, hem=HEM, share=FLARE_SHARE, log=print):
    """The petticoat as cut, as a table of radii: rows down the cloth (z for the yoke, then on down the gores), angles
    round it. M is measured on what the petticoat goes over: the corseted body with the corset on it, so the band and
    the yoke sit on the corset's own surface. Returns a dict:

    th          the angles [NTH]
    z           the heights of the table's rows [nz], from the band's top down to below the floor (the gores run on)
    r           the radius of the cloth at each [nz, NTH] (the pattern: with the gores' full flare)
    plumb       the radius the cloth would hang at with no flare at all (from the fullest hip straight down)
    flare       the flare's slope at each angle (radius gained per unit of drop below flare_z)
    band_top, band_bot, flare_z, hem_z   the band, where the flare starts, the front hem's height
    length      the cloth's length from the band's top to the hem at each angle (the train at the back)"""
    H = M['H']
    r_ = {k: v * H for k, v in ratios.items()}
    th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
    band_top = M['waist_z'] + 0.5 * r_['band']
    band_bot = M['waist_z'] - 0.5 * r_['band']
    hem_z = M['floor'] + r_['clear']
    z = np.arange(band_top, M['floor'] - 2.5 * H, -0.1)
    nz = len(z)
    r = np.zeros((nz, NTH))
    ankle = M['floor'] + 0.5 * H                  # (the feet stand out under the hem: the solver meets them, not the cut)
    for i, zz in enumerate(z):
        body = M['r_body'](zz, th) if zz > ankle else np.zeros(NTH)
        r[i] = body + (r_['band_ease'] if zz >= band_bot else r_['ease'])
    # (the band is straight: as deep as it is, its girth is the waist's)
    r[z >= band_bot] = r[int(np.argmin(np.abs(z - M['waist_z'])))]
    # (below the crotch the section is both legs: the cloth only ever hangs plumb from above, never in)
    hip_i = int(np.argmin(np.abs(z - M['hips_z'])))
    seat_i = int(np.argmin(np.abs(z - M['seat_z'])))
    low = max(hip_i, seat_i)
    r[low:] = np.maximum.accumulate(r[low:], axis=0)
    r[:low + 1] = _concave_down(r[:low + 1], z[:low + 1])
    r[low:] = np.maximum.accumulate(r[low:], axis=0)
    plumb = r.copy()
    # the gores' flare: from flare_z down, a share at each angle, scaled so the hem's girth is HEM
    flare_z = M['hips_z'] + r_['flare_from']
    drop = np.clip(flare_z - z, 0.0, None)
    w = _share(th, share)
    hem_i = int(np.argmin(np.abs(z - hem_z)))

    def at_hem(c):
        return girth(plumb[hem_i] + c * w * drop[hem_i], th)
    want = hem * (M['waist_z'] - M['floor'])
    lo, hi = 0.0, 2.0
    for _ in range(50):
        c = 0.5 * (lo + hi)
        lo, hi = (c, hi) if at_hem(c) < want else (lo, c)
    c = 0.5 * (lo + hi)
    slope = c * w
    r = plumb + slope[None, :] * drop[:, None]
    # the length of each line of cloth from the band's top: to the floor at the front's clearance, more toward the
    # back by the train (the hem curve a smooth rise from the back to the front)
    dr = np.diff(r, axis=0)
    ds = np.sqrt(dr ** 2 + 0.01)
    s = np.vstack([np.zeros((1, NTH)), np.cumsum(ds, axis=0)])
    to_hem = np.array([np.interp(-hem_z, -z, s[:, j]) for j in range(NTH)])
    a = np.abs(((th + np.pi) % (2 * np.pi)) - np.pi) / np.pi
    length = to_hem + r_['train'] * _ss(0.35, 1.0, a) ** 1.5
    log('petticoat pattern: band %.2f round (%.0f cm), hips %.2f, hem %.2f (%.2f H, %.0f cm); flare slopes front %.3f '
        'side %.3f back %.3f; length front %.2f back %.2f' % (
            girth(r[0], th), girth(r[0], th) * 165 / 63, girth(r[hip_i], th), girth(r[hem_i], th), girth(r[hem_i], th) / H,
            girth(r[hem_i], th) * 165 / 63, slope[0], slope[NTH // 4], slope[NTH // 2], length[0], length[NTH // 2]))
    return dict(th=th, z=z, r=r, plumb=plumb, flare=slope, s=s, length=length, band_top=band_top, band_bot=band_bot,
                flare_z=flare_z, hem_z=hem_z)


# ---------------------------------------------------------------- 2. the hang

def _at(pat, key, t):
    """A pattern table read at a fraction t of each line's length: [NTH] values (r or z) for the rows at t."""
    s = pat['s']
    L = pat['length']
    out = np.empty(NTH)
    for j in range(NTH):
        sj = t * L[j]
        col = pat[key][:, j] if key != 'z' else pat['z']
        out[j] = np.interp(sj, s[:, j], col)
    return out


def _flutes(th, n, rng):
    """The flutes' phase round the skirt: n of them, closer where the flare's share is larger (the gathered back),
    each a little off its even place. Returns phase(th) in turns."""
    dens = _share(th) ** 1.6
    cdf = np.cumsum(dens)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0] + (cdf[-1] - cdf[0]) / len(th))
    jitter = 0.12 * np.sin(2 * np.pi * (3 * cdf + rng.uniform())) + 0.07 * np.sin(2 * np.pi * (5 * cdf + rng.uniform()))
    return n * cdf + jitter + 0.25


def hang(M, pat, ratios=RATIOS, hang_share=HANG, grid=GRID, seed=SEED, log=print):
    """The petticoat's mesh as it is put on the body before the solver: rows at fractions of each line's length,
    each ring drawn in toward the hip plumb and fluted so that its girth stays the pattern's. Returns (V, F, info)
    with info holding per point: t (down the cloth, 0 at the band's top, 1 at the hem), u, v (the pattern, in units),
    th, row index, and the list of rows (index arrays)."""
    H = M['H']
    th = pat['th']
    rng = np.random.default_rng(seed)
    Lmax = float(pat['length'].max())
    nrows = max(8, int(round(Lmax / grid)) + 1)
    ts = np.linspace(0.0, 1.0, nrows)
    floor = M['floor']
    gap = 0.03 * H
    # the flutes: count from the hem's spare girth and the flute width
    hem_hang = pat['plumb'][int(np.argmin(np.abs(pat['z'] - pat['hem_z'])))]
    n_flutes = int(round(girth(hem_hang, th) / (ratios['flute'] * H)))
    n_flutes += n_flutes % 2
    phase = _flutes(th, n_flutes, rng)
    wave = np.sin(2 * np.pi * phase)
    # (a flute is rounder at its crest outside than in its valley: the cloth turns over the crest, folds in the valley)
    wave = np.where(wave > 0, np.abs(wave) ** 0.8, -np.abs(wave) ** 1.25)
    V, U, Vv, T, TH, rows, Gs, R = [], [], [], [], [], [], [], []
    for i, t in enumerate(ts):
        r_pat = _at(pat, 'r', t)
        z_pat = _at(pat, 'z', t)
        plumb = _at(pat, 'plumb', t)
        G = girth(r_pat, th)
        hang_r = plumb + hang_share * (r_pat - plumb)
        # the flutes: as deep as it takes for the ring to keep the pattern's girth
        lo, hi = 0.0, 0.5 * float(np.max(r_pat - plumb)) + 1e-3
        if girth(hang_r, th) < G - 1e-3:
            for _ in range(40):
                A = 0.5 * (lo + hi)
                if girth(hang_r + A * wave, th) < G:
                    lo = A
                else:
                    hi = A
                if girth(hang_r + hi * wave, th) < G:
                    hi *= 2
            A = 0.5 * (lo + hi)
        else:
            A = 0.0
        rr = hang_r + A * wave
        zz = z_pat.copy()
        # the train: what would go through the floor turns out along it
        below = zz < floor + gap
        if below.any():
            rr = np.where(below, rr + (floor + gap - zz), rr)
            zz = np.where(below, floor + gap, zz)
        # resample the ring to points about grid apart, evenly by its own length
        x, y = rr * np.sin(th), -rr * np.cos(th)
        seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(24, int(round(cum[-1] / grid)))
        n += (-n) % 8
        want = np.linspace(0.0, cum[-1], n, endpoint=False)
        k = np.searchsorted(cum, want, side='right') - 1
        f = (want - cum[k]) / np.maximum(seg[k], 1e-9)
        th_k = th[k] + f * (2 * np.pi / NTH)
        x_ = np.interp(th_k, np.append(th, 2 * np.pi), np.append(x, x[0]))
        y_ = np.interp(th_k, np.append(th, 2 * np.pi), np.append(y, y[0]))
        z_ = np.interp(th_k, np.append(th, 2 * np.pi), np.append(zz, zz[0]))
        # the pattern's u: round the row as cut (its girth G), v: down the line of cloth
        Lk = np.interp(th_k, np.append(th, 2 * np.pi), np.append(pat['length'], pat['length'][0]))
        idx = np.arange(len(V), len(V) + n)
        rows.append(idx)
        V += list(np.stack([x_, y_, z_], 1))
        U += list(want / cum[-1] * G)
        Vv += list(t * Lk)
        T += [t] * n
        TH += list(th_k)
        Gs.append(G)
        R += [i] * n
    V = np.array(V)
    F = []
    for a, b in zip(rows[:-1], rows[1:]):
        F += _zip(a, np.array(TH)[a], b, np.array(TH)[b])
    info = dict(t=np.array(T), u=np.array(U), v=np.array(Vv), th=np.array(TH), rows=rows, n_flutes=n_flutes, ts=ts,
                G=np.array(Gs), row=np.array(R))
    log('petticoat hang: %d rows, %d points, %d faces, %d flutes' % (len(rows), len(V), len(F), n_flutes))
    return V, F, info


def _zip(a, ta, b, tb):
    """Triangles joining ring a (angles ta, ascending) to ring b (angles tb): walk both rings round by angle."""
    na, nb = len(a), len(b)
    i = j = 0
    out = []
    while i < na or j < nb:
        ai, aj = a[i % na], a[(i + 1) % na]
        bi, bj = b[j % nb], b[(j + 1) % nb]
        ta_next = ta[(i + 1) % na] + (2 * np.pi if i + 1 >= na else 0.0)
        tb_next = tb[(j + 1) % nb] + (2 * np.pi if j + 1 >= nb else 0.0)
        if j >= nb or (i < na and ta_next <= tb_next):
            out.append((ai, aj, bi))
            i += 1
        else:
            out.append((ai, bj, bi))
            j += 1
    return out


# ---------------------------------------------------------------- 3. the rig

def weights(V, info, M, pat):
    """Each point's bones: the pelvis above the fullest hip; below it, its own side's thigh takes over toward the hem,
    up to THIGH_SHARE, on the side it lies (through the middle, both alike)."""
    t = info['t']
    z = V[:, 2]
    hips = M['hips_z']
    down = _ss(hips, M['floor'], z) * (t > 0.05)
    k = 0.45 * down
    side = _ss(-1.2, 1.2, V[:, 0])           # 1 on her left (+x), 0 on her right
    return {'Pelvis': 1.0 - k, 'Thigh.L': k * side, 'Thigh.R': k * (1.0 - side)}
