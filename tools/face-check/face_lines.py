"""The facial proportion and profile line check (John, 2026-11-20: "what's the check called for faces to look at
proper lines and how do we add it").

It is a facial anthropometric analysis: the soft-tissue profile analysis and the rhinoplasty nasal analysis that
surgeons and portrait artists run on photographs. Named landmarks (Farkas' points) are found on the head mesh itself,
the standard reference lines are drawn through them on clay front, profile and base views, and each measure is
compared with a published range. It runs on the Face Creator's heads before any face render, in the draft stage,
beside the topology check (render-queue rq_topo.py). Report-only: it never stops a render.

    node tools/face-check/dump-heads.cjs face-creator.html DUMP [Ranger,Rogue,...]     (headless Chromium, ~20 s a face)
    python3 tools/face-check/face_lines.py DUMP --out DIR [--faces Ranger,Rogue]

Writes DIR/face-lines.json (per face: landmarks in mm, every measure with its range, source and verdict),
DIR/face-lines-sheet.png (one row per face: front, profile, base and a nose close-up with the lines and numbers) and
DIR/face-lines.md (which faces fail which measures). Needs numpy and Pillow only.

Frame (the creator's, as the dump writes it): x across the face, y up, z toward the viewer, chin near y = 0. The
creator's head height HH = 8.6 units is 219 mm (NOSE_LIFE.head_h, Farkas' adult female v-gn), so every length is
reported in the creator's own millimetres. The Knight is male (sex 1): the measures with a sex split use the men's range.

Landmarks (Farkas' names; soft tissue):
    tr  trichion       the hairline on the midline: the hair cap's lowest front point on the midline section (a fringe
                       hides the scalp's own hairline, so the three thirds lean on the hairstyle; mid_low does not)
    g   glabella       the most prominent point of the forehead's midline above the nasion
    n   nasion         the deepest point of the midline between the glabella and the nose tip
    prn pronasale      the most forward point of the nose (the tip)
    cm  columella      the point where the columella's lower tangent from sn touches it
    sn  subnasale      where the columella meets the upper lip: the sharpest turn of the profile below the tip
    ls  labrale sup.   the most forward point of the upper lip
    sto stomion        the deepest point between the lips on the midline
    li  labrale inf.   the most forward point of the lower lip
    sm  supramentale   the deepest point of the fold between the lower lip and the chin
    pg  pogonion       the most forward point of the chin
    me  menton         where the chin's front turns under (its tangent 45 degrees from vertical, the gnathion's
                       construction: the mesh has no bony menton)
    en, ex             the eye's inner and outer corner: the ends of the eyeball's visible part from the front
    al  alare          the wing's most lateral point: the widest point where the nose stands proud of the cheek
                       (more than 0.8 mm over a cheek fitted either side)
    ac  alar crease    the deepest (most posterior) point of the wing's crease against the cheek
    tdp tip-defining points, dal dorsal aesthetic lines: where the tip's and the bridge's cross-section turn 30
                       degrees from facing forward (the light reflex a portrait shows as two highlights)

Ranges and their sources are in RANGES below; every one was read from the source on 2026-11-20 (none from memory).
Where a source gives only an ideal value, the range around it is this check's tolerance and says so.
"""
import argparse
import base64
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HH = 8.6
MM = HH / 219.0          # one of the creator's millimetres in model units

SRC = {
    'P1': 'Brito, Avashia, Rohrich 2020, Evidence-based Nasal Analysis for Rhinoplasty: the 10-7-5 method, '
          'https://pmc.ncbi.nlm.nih.gov/articles/PMC7159929',
    'P2': 'Al-Sebaei 2015, The validity of three neo-classical facial canons, '
          'https://pmc.ncbi.nlm.nih.gov/articles/PMC4369102/',
    'E': 'Park 2011, Fundamental Principles in Aesthetic Rhinoplasty, https://www.e-ceo.org/journal/view.php?number=157',
    'S': 'Spear Education, Evaluating facial esthetics: the esthetic plane, '
         'https://speareducation.com/2013/08/evaluating-facial-esthetics-the-esthetic-plane',
    'PW': 'Powell, Aesthetic Evaluation of Nasal Contours (Cummings ch. 38; Powell and Humphreys aesthetic triangle), '
          'https://famona.tripod.com/ent/cummings/cumm038.pdf',
    'B': 'Becker, The Ideal Nose, https://www.rhinoplastyarchive.com/articles/rhinoplasty-fundamentals/the-ideal-nose',
    'T': 'Taub et al., Anthropometry, cephalometry and orthognathic surgery, '
         'https://plasticsurgerykey.com/anthropometry-cephalometry-and-orthognathic-surgery/',
}

# id: (label, unit, female range, male range or None, tolerance outside the range still read as 'near', sources, note)
RANGES = {
    'third_upper': ('Upper third tr-n / tr-me', '', (0.30, 0.367), None, 0.02, ['B', 'T', 'P2'],
                    'canon 1/3 each (P2 on tr-n-sn-gn; B, T on the glabella); range spans the canon and P2\'s measured women (0.367 / 0.298 / 0.338); '
                    'the spread is this check\'s'),
    'third_mid': ('Middle third n-sn / tr-me', '', (0.30, 0.367), None, 0.02, ['B', 'T', 'P2'], 'as above'),
    'third_low': ('Lower third sn-me / tr-me', '', (0.30, 0.367), None, 0.02, ['B', 'T', 'P2'], 'as above'),
    'mid_low': ('Middle / lower third n-sn / sn-me', '', (0.85, 1.05), None, 0.05, ['P2', 'B', 'T'],
                'canon 1.0; P2 women 53.2 / 60.3 = 0.88 (sn-gn); needs no hairline, so it holds under a fringe'),
    'lower_split': ('Lower face sn-sto / sto-me', '', (0.45, 0.55), None, 0.05, ['T'],
                    'T: "should be 1 : 2"; +-10% is this check\'s'),
    'en_en_mm': ('Intercanthal width en-en', 'mm', (31, 33), (31, 33), 1.5, ['P1', 'P2'],
                 'P1 normal 31-33 mm; P2 women 30.3 +- 2.4'),
    'orbital': ('Orbital canon ex-en / en-en', '', (0.95, 1.15), None, 0.05, ['P2'],
                'canon 1.0; P2 women mean 32.4 / 30.3 = 1.07; spread is this check\'s'),
    'orbitonasal': ('Orbitonasal canon al-al / en-en', '', (0.95, 1.10), (1.0, 1.25), 0.05, ['P1', 'P2', 'T'],
                    'P1 "alar base width ideally approximates the intercanthal distance"; P2 women 32.9 / 30.3 = 1.08, '
                    'men 36.9 / 30.3 = 1.22'),
    'alar_vs_height': ('Alar width al-al / nasal height n-sn', '', (0.63, 0.77), None, 0.04, ['P1', 'PW'],
                       'P1 "70% of nasal height"; +-10% is this check\'s'),
    'bony_base': ('Bony base / alar width', '', (0.70, 0.80), None, 0.04, ['P1', 'E'],
                  'P1 "Bony base width should be 70%-80% of the alar base"; E dorsum ~80% of base'),
    'tdp_mm': ('Tip-defining points apart', 'mm', (5.0, 6.0), None, 1.0, ['P1'], 'P1 "5-6 mm"'),
    'dal_vs_tdp': ('Dorsal lines width / tip-defining points', '', (0.85, 1.15), None, 0.1, ['P1'],
                   'P1: the dorsal aesthetic lines "should match ... the tip-defining points width"; +-15% is mine'),
    'nasofrontal': ('Nasofrontal angle g-n-prn', 'deg', (115, 135), (115, 130), 5, ['B', 'PW', 'T'],
                    'B 115-130; PW, T 125-135; PW men 115-130'),
    'nasolabial': ('Nasolabial angle cm-sn-ls', 'deg', (95, 110), (90, 100), 5, ['P1', 'E', 'PW', 'B'],
                   'P1 women 95-110, men 90-95; E women 100-105, men 90-100; PW, B overall 90-120'),
    'nasofacial': ('Nasofacial angle (g-pg vs n-prn)', 'deg', (30, 36), (34, 40), 3, ['PW', 'B', 'T'],
                   'PW 36-40 preferred, women at the lower end of 30-40; B 30-40; T 35-40'),
    'nasomental': ('Nasomental angle n-prn-pg', 'deg', (120, 132), (120, 132), 4, ['PW', 'B'], 'PW, B 120-132'),
    'goode': ('Goode ratio ac-prn / n-prn', '', (0.55, 0.60), (0.57, 0.62), 0.03, ['PW', 'B', 'T'],
              'PW "0.55 to 0.60", 0.55 women, 0.60 men'),
    'eline_upper': ('Upper lip to E-line (prn-pg)', 'mm', (-5.0, -3.0), None, 1.5, ['S', 'B'],
                    'S, B: upper lip 4 mm behind (negative = behind); +-1 mm is mine'),
    'eline_lower': ('Lower lip to E-line (prn-pg)', 'mm', (-3.0, -1.0), None, 1.5, ['S', 'B'],
                    'S, B: lower lip 2 mm behind; +-1 mm is mine'),
    'col_show': ('Columellar show (side)', 'mm', (2.0, 4.0), None, 0.5, ['B', 'P1', 'E'],
                 'B 2-4 mm normal; P1, E 2-3 mm'),
    'gull_drop': ('Gull wing: columella below alar rims (front)', 'mm', (0.3, 3.0), None, 0.3, ['PW', 'P1'],
                  'PW: columella "slightly lower" than the alar line, a gull in flight; P1 "resembles a gull in '
                  'flight"; the numbers are mine (slightly = 0.3-3 mm)'),
    'lobule_frac': ('Base: lobule / (lobule + columella)', '', (0.30, 0.37), None, 0.03, ['P1', 'T'],
                    'P1 columella-to-lobule 2:1; T nostrils two thirds, tip one third; spread is mine'),
    'nostril_len': ('Base: nostril length / base length', '', (0.60, 0.72), None, 0.04, ['E', 'T'],
                    'E nostrils "two thirds the length of the base"'),
    'nostril_axis': ('Base: nostril axis from midline', 'deg', (45, 60), None, 5, ['PW'],
                     'PW "approximately 45 to 60 degrees from the midline" (white noses)'),
    'base_triangle': ('Base: length / width (equilateral 0.87)', '', (0.80, 0.95), None, 0.05, ['P1'],
                      'P1 "the nose should create an equilateral triangle" (height/side 0.866); spread is mine'),
}
# the nose thread's own ratios (house rules derived on the Face Creator, 2026-11-20): measured and shown, judged against
# the value that thread set, not against a published norm
HOUSE = {
    'wing_h_vs_w': ('Wing height / wing half-width (house 0.97)', '', (0.87, 1.07), None, 0.05, [],
                    'nose thread: wing height 0.97 of its width; here width = al-al / 2'),
    'ball_vs_bridge': ('Tip ball radius / bridge half-width at mid-dorsum', '', None, None, 0, [],
                       'information: the nose thread fitted the bridge to the ball'),
    'floor_angle': ('Floor angle: columella tangent above horizontal', 'deg', None, None, 0, [],
                    'information: the nose thread set ~7 deg (Ranger) to ~12 deg (Rogue) from tip position and ball'),
    'tip_over_floor': ('Tip above the floor line sn-cm', 'mm', None, None, 0, [], 'information'),
}
MALE = {'Knight'}


# ------------------------------------------------------------------------------------------------ loading

def load(path):
    d = json.load(open(path))
    out = {'face': d['face'], 'head': None, 'eyes': [], 'ears': [], 'cap': [], 'params': d.get('P', {})}
    for m in d['meshes']:
        V = np.frombuffer(base64.b64decode(m['P']), np.float32).reshape(-1, 3).astype(np.float64)
        F = np.frombuffer(base64.b64decode(m['I']), np.uint32).reshape(-1, 3).astype(np.int64)
        if m['role'] == 'head':
            out['head'] = (V, F)
        elif m['role'] == 'eye':
            out['eyes'].append((V, F))
        elif m['role'] == 'hair' and m.get('cap'):
            out['cap'].append((V, F))
        elif m['role'] == 'other' and len(V) > 2000 and np.abs(V[:, 0]).mean() > 2.2:
            out['ears'].append((V, F))
    return out


# ------------------------------------------------------------------------------------------------ geometry

def section(V, F, axis, c):
    """The plane axis = c through the mesh: ordered polylines (lists of points)."""
    s = V[:, axis] - c
    s = np.where(s == 0, 1e-12, s)
    sg = s[F] > 0
    T = F[~(sg.all(1) | (~sg).all(1))]
    E = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1)
    sa, sb = s[E[..., 0]], s[E[..., 1]]
    cr = (sa > 0) != (sb > 0)
    idx = np.argsort(~cr, axis=1, kind='stable')[:, :2]
    r = np.arange(len(E))[:, None]
    Es = E[r, idx]
    ta, tb = sa[r, idx], sb[r, idx]
    t = ta / (ta - tb)
    P = V[Es[..., 0]] + (V[Es[..., 1]] - V[Es[..., 0]]) * t[..., None]
    key = np.sort(Es, axis=2)
    keys = [(int(a), int(b)) for a, b in key.reshape(-1, 2)]
    pts = {}
    adj = {}
    for i in range(len(T)):
        k0, k1 = keys[2 * i], keys[2 * i + 1]
        pts[k0] = P[i, 0]
        pts[k1] = P[i, 1]
        adj.setdefault(k0, []).append(k1)
        adj.setdefault(k1, []).append(k0)
    seen = set()
    chains = []
    for k in adj:
        if k in seen:
            continue
        # walk to one end (or round a loop), then collect
        start, prev = k, None
        cur = k
        while True:
            nb = [q for q in adj[cur] if q != prev]
            if not nb or nb[0] == start:
                break
            prev, cur = cur, nb[0]
            if len(adj[cur]) == 1:
                break
        start = cur
        chain, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nb = [q for q in adj[cur] if q != prev and q not in seen]
            if not nb:
                break
            prev, cur = cur, nb[0]
            seen.add(cur)
            chain.append(cur)
        chains.append(np.array([pts[q] for q in chain]))
    return chains


def resample(C, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    return np.stack([np.interp(s, d, C[:, j]) for j in range(C.shape[1])], 1)


def raster(V, F, ids, uvd, lo, hi, res, keep=None):
    """Orthographic z-buffer. uvd(V) -> (u, v, depth) with larger depth nearer the viewer. lo, hi: the (u, v) window
    in model units, res: units a pixel. Returns depth, the triangle's id per pixel (-1 empty), and the window."""
    W = int(math.ceil((hi[0] - lo[0]) / res))
    H = int(math.ceil((hi[1] - lo[1]) / res))
    zb = np.full(W * H, -np.inf)
    tb = np.full(W * H, -1, np.int64)
    U = uvd(V)
    px = (U[:, 0] - lo[0]) / res
    py = (hi[1] - U[:, 1]) / res
    dz = U[:, 2]
    T = F if keep is None else F[keep]
    I = ids if keep is None else ids[keep]
    x = px[T]
    y = py[T]
    z = dz[T]
    x0 = np.floor(x.min(1)).astype(np.int64)
    y0 = np.floor(y.min(1)).astype(np.int64)
    x1 = np.ceil(x.max(1)).astype(np.int64)
    y1 = np.ceil(y.max(1)).astype(np.int64)
    inw = (x1 >= 0) & (y1 >= 0) & (x0 < W) & (y0 < H)
    x, y, z, x0, y0, x1, y1, I = x[inw], y[inw], z[inw], x0[inw], y0[inw], x1[inw], y1[inw], I[inw]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    ok = np.abs(det) > 1e-12
    size = np.maximum(x1 - x0, y1 - y0)
    for lo_s, hi_s in ((0, 4), (4, 10), (10, 30), (30, 100000)):
        m = ok & (size >= lo_s) & (size < hi_s)
        if not m.any():
            continue
        sel = np.nonzero(m)[0]
        if hi_s > 30:      # a few big triangles, one at a time
            groups = [[k] for k in sel]
        else:
            groups = [sel]
        for g in groups:
            g = np.asarray(g)
            K = int(size[g].max()) + 1
            for oy in range(K):
                for ox in range(K):
                    cx = x0[g] + ox
                    cy = y0[g] + oy
                    v = (cx <= x1[g]) & (cy <= y1[g]) & (cx >= 0) & (cy >= 0) & (cx < W) & (cy < H)
                    if not v.any():
                        continue
                    gg = g[v]
                    qx = cx[v] + 0.5
                    qy = cy[v] + 0.5
                    xa, ya = x[gg], y[gg]
                    d = det[gg]
                    w1 = ((qx - xa[:, 0]) * (ya[:, 2] - ya[:, 0]) - (xa[:, 2] - xa[:, 0]) * (qy - ya[:, 0])) / d
                    w2 = ((xa[:, 1] - xa[:, 0]) * (qy - ya[:, 0]) - (qx - xa[:, 0]) * (ya[:, 1] - ya[:, 0])) / d
                    w0 = 1 - w1 - w2
                    ins = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
                    if not ins.any():
                        continue
                    gg = gg[ins]
                    zz = (w0[ins] * z[gg, 0] + w1[ins] * z[gg, 1] + w2[ins] * z[gg, 2])
                    flat = cy[v][ins] * W + cx[v][ins]
                    order = np.argsort(zz)
                    flat, zz, gg = flat[order], zz[order], gg[order]
                    better = zz > zb[flat]
                    flat, zz, gg = flat[better], zz[better], gg[better]
                    zb[flat] = zz      # (sorted ascending, so the nearest of repeated pixels is written last)
                    tb[flat] = I[gg]
    return zb.reshape(H, W), tb.reshape(H, W), (lo, hi, res, W, H)


def tri_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


# ------------------------------------------------------------------------------------------------ landmarks

def profile(V, F):
    """The midline profile, front part, ordered top to bottom, resampled every 0.1 mm: columns (y, z)."""
    chains = section(V, F, 0, 0.0)
    C = max(chains, key=len)
    yz = C[:, 1:3]
    i = int(np.argmax(yz[:, 1]))
    # walk from the tip both ways while on the front of the head
    a = i
    while a > 0 and yz[a - 1, 1] > 0.15 and yz[a - 1, 0] < 0.92 * HH:
        a -= 1
    b = i
    while b < len(yz) - 1 and yz[b + 1, 1] > 0.15 and yz[b + 1, 0] > -0.08 * HH:
        b += 1
    seg = yz[a:b + 1]
    if seg[0, 0] < seg[-1, 0]:
        seg = seg[::-1]
    return resample(seg, 0.1 * MM)


def tangent_angle(Pyz, k=8):
    """Per point: the direction of travel (down the profile) as degrees: 90 straight down, 0 straight back, 180 forward."""
    d = np.zeros_like(Pyz)
    d[k:-k] = Pyz[2 * k:] - Pyz[:-2 * k]
    d[:k] = d[k]
    d[-k:] = d[-k - 1]
    return np.degrees(np.arctan2(-d[:, 0], -d[:, 1]))


def first(cond, start=0):
    idx = np.nonzero(cond[start:])[0]
    return start + int(idx[0]) if len(idx) else None


def landmarks(face):
    V, F = face['head']
    L = {}
    eyes = [E[0].mean(0) for E in face['eyes']]
    eye_y = float(np.mean([e[1] for e in eyes])) if eyes else 0.525 * HH
    Pr = profile(V, F)
    Y, Z = Pr[:, 0], Pr[:, 1]
    ang = tangent_angle(Pr)

    def pt(i):
        return np.array([0.0, Y[i], Z[i]])
    # the tip
    m = (Y > 0.28 * HH) & (Y < eye_y)
    iprn = int(np.nonzero(m)[0][np.argmax(Z[m])])
    # nasion: the deepest point above the tip, below the brow
    m = (np.arange(len(Y)) < iprn) & (Y < eye_y + 0.09 * HH) & (Y > Y[iprn] + 8 * MM)
    idx = np.nonzero(m)[0]
    i_n = int(idx[np.argmin(Z[idx])])
    m = (np.arange(len(Y)) < i_n) & (Y < Y[i_n] + 25 * MM)
    idx = np.nonzero(m)[0]
    ig = int(idx[np.argmax(Z[idx])])
    # subnasale: past the tip, the profile turns under (going back along the columella), then turns down the lip
    iu = first(ang < 45, iprn + 5)
    idown = first(ang > 75, iu)
    seg = np.arange(iu, idown + 1)
    curv = np.gradient(ang)[seg]
    isn = int(seg[np.argmax(curv)])
    # the columella's tangent: a line fitted to its lower border, 20-60% of the way from sn to the tip; cm is where
    # that line passes under the border's middle
    cand = np.arange(iprn, isn)
    t = (Z[cand] - Z[isn]) / (Z[iprn] - Z[isn])
    cand = cand[(t > 0.2) & (t < 0.6)]
    A = np.stack([Z[cand] - Z[isn], np.ones(len(cand))], 1)
    slope, icpt = np.linalg.lstsq(A, Y[cand], rcond=None)[0]
    icm = int(cand[len(cand) // 2])
    cm_pt = np.array([0.0, Y[isn] + slope * (Z[icm] - Z[isn]), Z[icm]])
    # the lips and chin, by the profile's forward bumps and hollows below sn

    def argext(lo, hi, fn, ylo=None):
        r = np.arange(lo, hi)
        if ylo is not None:
            r = r[Y[r] > ylo]
        return int(r[fn(Z[r])])
    ils = argext(isn + 5, len(Y), np.argmax, Y[isn] - 0.075 * HH)
    # the lower lip: the next forward bump after a hollow under the upper lip
    r = np.arange(ils + 5, len(Y))
    r = r[Y[r] > Y[ils] - 0.10 * HH]
    # local maxima of z
    zz = Z[r]
    lm = [k for k in range(3, len(zz) - 3) if zz[k] >= zz[k - 3:k + 4].max()]
    ili = int(r[max(lm, key=lambda k: zz[k])]) if lm else int(r[np.argmax(zz)])
    isto = argext(ils, ili + 1, np.argmin)
    # the chin: menton where its front turns under; then the labiomental fold the farthest point behind the chord li-me,
    # and the pogonion the farthest in front of the chord sm-me (so a receding chin with no forward bump still has one)
    ime = first((ang < 45) & (Y < Y[ili] - 15 * MM), ili + 5) or len(Y) - 1

    def off_chord(i0, i1):
        a, b = np.array([Y[i0], Z[i0]]), np.array([Y[i1], Z[i1]])
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        if n[1] < 0:
            n = -n
        r = np.arange(i0 + 1, i1)
        return r, (np.stack([Y[r], Z[r]], 1) - a) @ n
    r, o = off_chord(ili, ime)
    ism = int(r[np.argmin(o)])
    r, o = off_chord(ism, ime)
    ipg = int(r[np.argmax(o)])
    L['cm'] = cm_pt
    for nm, i in (('g', ig), ('n', i_n), ('prn', iprn), ('sn', isn), ('ls', ils), ('sto', isto),
                  ('li', ili), ('sm', ism), ('pg', ipg), ('me', ime)):
        L[nm] = pt(i)
    # trichion: the hair cap's lowest front point on the midline
    L['tr'] = None
    for Vc, Fc in face['cap']:
        for C in section(Vc, Fc, 0, 0.0):
            m = (C[:, 2] > L['g'][2] - 30 * MM) & (C[:, 1] > L['g'][1] + 5 * MM)
            if m.any():
                c = C[m][np.argmin(C[m][:, 1])]
                if L['tr'] is None or c[1] < L['tr'][1]:
                    L['tr'] = np.array([0.0, c[1], c[2]])
    return L, Pr, eyes


def front_measures(face, L, eyes):
    """From a front z-buffer of the head and eyes: the eye corners, the nose's outline over the cheek (alare, the
    alar crease, bony base, dorsal lines and tip-defining points) and the lower edge of the nose (the gull wing)."""
    V, F = face['head']
    res = 0.15 * MM
    lo = (-48 * MM, L['sn'][1] - 14 * MM)
    hi = (48 * MM, L['n'][1] + 30 * MM)
    meshes = [(V, F, 0)] + [(E[0], E[1], 1 + k) for k, E in enumerate(face['eyes'])]
    allV = np.concatenate([m[0] for m in meshes])
    offs = np.cumsum([0] + [len(m[0]) for m in meshes])
    allF = np.concatenate([m[1] + offs[k] for k, m in enumerate(meshes)])
    owner = np.concatenate([np.full(len(m[1]), m[2]) for m in meshes])
    zb, tb, win = raster(allV, allF, np.arange(len(allF)), lambda P: P[:, [0, 1, 2]], lo, hi, res)
    Hh, Ww = zb.shape
    own = np.where(tb >= 0, owner[np.maximum(tb, 0)], -1)
    xs = lo[0] + (np.arange(Ww) + 0.5) * res
    ys = hi[1] - (np.arange(Hh) + 0.5) * res
    out = {'front': (zb, tb, own, win, allV, allF)}
    # eye corners: the visible eyeball's ends
    for k in range(len(face['eyes'])):
        yy, xx = np.nonzero(own == 1 + k)
        if not len(xx):
            continue
        X = xs[xx]
        side = np.sign(X.mean())
        ax = np.abs(X)
        i_en = np.argmin(ax)
        i_ex = np.argmax(ax)
        nm = 'R' if side < 0 else 'L'
        out['en_' + nm] = np.array([X[i_en], ys[yy[i_en]], zb[yy[i_en], xx[i_en]]])
        out['ex_' + nm] = np.array([X[i_ex], ys[yy[i_ex]], zb[yy[i_ex], xx[i_ex]]])
    # the head alone, for the nose
    zh, th, _ = raster(V, F, np.arange(len(F)), lambda P: P[:, [0, 1, 2]], lo, hi, res)
    out['head_front'] = (zh, th, win)
    # the nose over a cheek fitted either side, row by row: z = a + b x^2 on |x| 24-38 mm
    ax = np.abs(xs)
    cheek = (ax > 24 * MM) & (ax < 38 * MM)
    excess = np.full_like(zh, np.nan)
    for r in range(Hh):
        z = zh[r]
        m = cheek & np.isfinite(z)
        if m.sum() < 20:
            continue
        A = np.stack([np.ones(m.sum()), xs[m] ** 2], 1)
        c, *_ = np.linalg.lstsq(A, z[m], rcond=None)
        excess[r] = z - (c[0] + c[1] * xs ** 2)
    out['excess'] = excess
    thr = 0.8 * MM
    mid = Ww // 2
    halfw = np.zeros(Hh)
    for r in range(Hh):
        e = excess[r]
        if not np.isfinite(e[mid]) or e[mid] < thr:
            continue
        rr = mid
        while rr < Ww - 1 and np.isfinite(e[rr + 1]) and e[rr + 1] > thr:
            rr += 1
        ll = mid
        while ll > 0 and np.isfinite(e[ll - 1]) and e[ll - 1] > thr:
            ll -= 1
        halfw[r] = 0.5 * (xs[rr] - xs[ll])
    out['halfw'] = halfw
    nasal_h = L['n'][1] - L['sn'][1]
    rows_al = (ys > L['sn'][1] - 3 * MM) & (ys < L['sn'][1] + 0.45 * nasal_h)
    r_al = int(np.nonzero(rows_al)[0][np.argmax(halfw[rows_al])])
    al = halfw[r_al]
    out['al_R'] = np.array([-al, ys[r_al], zh[r_al, np.argmin(np.abs(xs + al))]])
    out['al_L'] = np.array([al, ys[r_al], zh[r_al, np.argmin(np.abs(xs - al))]])
    # the alar crease: the deepest point of the outline in the wing's rows
    best = None
    rows_ac = (ys > L['sn'][1] - 3 * MM) & (ys < ys[r_al] + 6 * MM)
    for r in np.nonzero(rows_ac & (halfw > 0.7 * al))[0]:
        for sgn in (-1, 1):
            c = int(np.argmin(np.abs(xs - sgn * halfw[r])))
            z = zh[r, c]
            if np.isfinite(z) and (best is None or z < best[2]):
                best = (xs[c], ys[r], z)
    out['ac'] = np.array(best)
    # bony base: the nose's width over the cheek at the level halfway down from the nasion toward the eye line's...
    # here at the rhinion's height, 0.35 of the way from n to sn (where the nasal bones end)
    # (the nasofacial groove: where the bridge's side wall turns into the face, the sharpest concave bend of the smoothed
    # cross-section 3-16 mm either side of the midline)
    y_bb = L['n'][1] - 0.35 * nasal_h
    r_bb = int(np.argmin(np.abs(ys - y_bb)))
    zrow = np.nanmean(zh[max(0, r_bb - 3):r_bb + 4], 0)
    k = int(round(1.0 * MM / res))
    zr = np.convolve(np.nan_to_num(zrow), np.ones(2 * k + 1) / (2 * k + 1), 'same')
    c2 = np.zeros_like(zr)
    c2[k:-k] = zr[2 * k:] - 2 * zr[k:-k] + zr[:-2 * k]
    hb = []
    for sg in (1, -1):
        idx = np.nonzero((np.sign(xs) == sg) & (np.abs(xs) > 3 * MM) & (np.abs(xs) < 16 * MM))[0]
        hb.append(abs(xs[idx[np.argmax(c2[idx])]]))
    out['bony_half'] = float(np.mean(hb)) if hb else float('nan')
    out['y_bb'] = ys[r_bb]
    # the light reflex lines: where the cross-section turns 30 degrees from facing forward, at mid-dorsum and at the tip
    def reflex(yv, lim, slope, groove=False):
        # the light reflex: the front-lit band of the bridge or tip, out to where its cross-section turns from facing the
        # viewer (45 degrees on the bridge: the dorsal aesthetic lines; 30 on the tip: the tip-defining points, and on
        # the tip never past the groove between the tip and the wing, the first hollow going out)
        r = int(np.argmin(np.abs(ys - yv)))
        z = np.nanmean(zh[max(0, r - 3):r + 4], 0)
        k = int(round(0.4 * MM / res))
        zs_ = np.convolve(np.nan_to_num(z), np.ones(2 * k + 1) / (2 * k + 1), 'same')
        dz = np.gradient(zs_, res)
        h = []
        for sg in (1, -1):
            idx = np.nonzero((np.sign(xs) == sg) & (np.abs(xs) < lim))[0]
            idx = idx[np.argsort(np.abs(xs[idx]))]
            past = list(np.nonzero(np.abs(dz[idx]) > slope)[0])
            if groove:
                zz = zs_[idx]
                hol = [q for q in range(2, len(idx) - 2) if zz[q] < zz[q - 2] and zz[q] < zz[q + 2]]
                past += hol[:1]
            h.append(abs(xs[idx[min(past)]]) if past else lim)
        return 0.5 * sum(h), ys[r]
    y_md = 0.5 * (L['n'][1] + L['prn'][1])
    out['dal_half'], out['y_dal'] = reflex(y_md, 8 * MM, 1.0)
    out['tdp_half'], out['y_tdp'] = reflex(L['prn'][1], 9 * MM, math.tan(math.radians(30)), groove=True)
    # the tip's ball: a circle fitted to the tip row's cross-section within 3 mm of the midline
    r = int(np.argmin(np.abs(ys - L['prn'][1])))
    m = np.abs(xs) < 3 * MM
    x_, z_ = xs[m], zh[r, m]
    A = np.stack([x_, z_, np.ones_like(x_)], 1)
    c, *_ = np.linalg.lstsq(A, -(x_ ** 2 + z_ ** 2), rcond=None)
    out['ball_r'] = math.sqrt(max(1e-12, c[0] ** 2 / 4 + c[1] ** 2 / 4 - c[2]))
    # the nose's lower edge seen from the front (the gull wing): per column, going down from the tip, the first row
    # where the visible surface jumps back by more than 1.5 mm (from the nose to the lip behind it)
    edge = np.full(Ww, np.nan)
    r0 = int(np.argmin(np.abs(ys - (L['prn'][1] + 2 * MM))))
    for cidx in range(Ww):
        if abs(xs[cidx]) > al:
            continue
        z = zh[:, cidx]
        for rr in range(r0, Hh - 1):
            if np.isfinite(z[rr]) and np.isfinite(z[rr + 1]) and z[rr] - z[rr + 1] > 1.5 * MM and excess[rr, cidx] > thr:
                edge[cidx] = ys[rr]
                break
    out['edge'] = edge
    out['xs'], out['ys'] = xs, ys
    # the wing's height: the nose's outline's vertical extent in the column through the wing (0.75 of its half-width,
    # outside the bridge's base, so the run is the wing alone)
    hts = []
    for sgn in (-1, 1):
        cidx = int(np.argmin(np.abs(xs - sgn * 0.75 * al)))
        rows = np.nonzero(np.isfinite(excess[:, cidx]) & (excess[:, cidx] > thr) & (ys < L['prn'][1] + 0.3 * nasal_h))[0]
        if len(rows):
            # (the run of rows that holds the wing)
            runs = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
            run = max(runs, key=len)
            hts.append(ys[run[0]] - ys[run[-1]])
    out['wing_h'] = float(np.mean(hts)) if hts else float('nan')
    return out


def side_measures(face, L, al_half):
    """From the side, z-buffered from the viewer's right (+x): the alar rim's lower edge over the columella."""
    V, F = face['head']
    res = 0.1 * MM
    lo = (L['sn'][2] - 26 * MM, L['sn'][1] - 6 * MM)
    hi = (L['prn'][2] + 4 * MM, L['prn'][1] + 10 * MM)
    zb, tb, win = raster(V, F, np.arange(len(F)), lambda P: np.stack([P[:, 2], P[:, 1], P[:, 0]], 1), lo, hi, res)
    Hh, Ww = zb.shape
    us = lo[0] + (np.arange(Ww) + 0.5) * res
    ys = hi[1] - (np.arange(Hh) + 0.5) * res
    # per column: going down from the tip's height, the alar rim is where the visible surface steps in from the wing
    # (x near al) to the columella or the nostril (x near the midline)
    rim = np.full(Ww, np.nan)
    for c in range(Ww):
        x = zb[:, c]
        on = np.isfinite(x) & (x > 0.55 * al_half)
        rows = np.nonzero(on & (ys < L['prn'][1]))[0]
        if not len(rows):
            continue
        rr = rows[-1]
        if rr + 1 < Hh and np.isfinite(x[rr + 1]) and x[rr + 1] < 0.45 * al_half:
            rim[c] = ys[rr]
    return {'rim': rim, 'us': us, 'side': (zb, tb, win)}


def base_measures(face, L, al_half):
    """From below, with everything under the subnasale cut away: the nostrils, the lobule and the base triangle."""
    V, F = face['head']
    res = 0.1 * MM
    lo = (-24 * MM, L['sn'][2] - 16 * MM)
    hi = (24 * MM, L['prn'][2] + 3 * MM)
    cut = L['sn'][1] - 0.6 * MM
    keep = (V[F][:, :, 1] > cut).all(1)
    zb, tb, win = raster(V, F, np.arange(len(F)), lambda P: np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1), lo, hi, res,
                         keep=keep)
    Hh, Ww = zb.shape
    xs = lo[0] + (np.arange(Ww) + 0.5) * res
    zs = hi[1] - (np.arange(Hh) + 0.5) * res
    height = -zb      # the visible surface's height y (looking up)
    # the nostrils: holes that run well up into the nose, inside the wings, behind the tip
    up = height - L['sn'][1]
    # a nostril is a hollow that runs up into the nose with lower rim all round it: an h-dome of the height seen from
    # below (the height less its reconstruction by dilation from 10 mm under it), with empty space counted as very
    # high, so the tip and the cheeks, open to it, are no domes
    upf = np.where(np.isfinite(up), up, 40 * MM)
    rec = upf - 10 * MM
    for _ in range(2000):
        d = rec.copy()
        d[1:] = np.maximum(d[1:], rec[:-1])
        d[:-1] = np.maximum(d[:-1], rec[1:])
        d[:, 1:] = np.maximum(d[:, 1:], rec[:, :-1])
        d[:, :-1] = np.maximum(d[:, :-1], rec[:, 1:])
        d = np.minimum(d, upf)
        if np.array_equal(d, rec):
            break
        rec = d
    nos = (upf - rec) > 0.8 * MM
    nos &= (np.abs(xs)[None, :] < al_half) & (np.abs(xs)[None, :] > 0.3 * MM)
    nos &= (zs[:, None] < L['prn'][2] - 1.0 * MM) & (zs[:, None] > L['sn'][2] - 12 * MM)
    # (keep, each side, the connected piece nearest where a nostril sits: mid-way along the base, mid-way out the wing;
    # the rest is the lip and cheek rising behind the wings' feet, seen from below)
    lab = label(nos)
    pick = np.zeros_like(nos)
    zc = L['sn'][2] + 0.4 * (L['prn'][2] - L['sn'][2])
    for sgn in (-1, 1):
        cx = int(np.argmin(np.abs(xs - sgn * 0.45 * al_half)))
        cz = int(np.argmin(np.abs(zs - zc)))
        yy, xx = np.nonzero(lab > 0)
        if not len(xx):
            continue
        side = np.sign(xs[xx]) == sgn
        if not side.any():
            continue
        dd = (yy[side] - cz) ** 2 + (xx[side] - cx) ** 2
        k = lab[yy[side][np.argmin(dd)], xx[side][np.argmin(dd)]]
        pick |= lab == k
    nos = pick
    out = {'base': (zb, tb, win, keep), 'nostril': nos, 'xs': xs, 'zs': zs}
    res_ax = []
    for sgn in (-1, 1):
        m = nos & ((np.sign(xs)[None, :] == sgn))
        yy, xx = np.nonzero(m)
        if len(xx) < 30:
            continue
        P2 = np.stack([xs[xx], zs[yy]], 1)
        c = P2.mean(0)
        w, U = np.linalg.eigh(np.cov((P2 - c).T))
        a = U[:, np.argmax(w)]
        axis = math.degrees(math.atan2(abs(a[0]), abs(a[1])))      # from the midline (the z axis)
        proj = (P2 - c) @ a
        res_ax.append({'side': sgn, 'axis': axis, 'len': proj.max() - proj.min(), 'apex_z': P2[:, 1].max(),
                       'back_z': P2[:, 1].min(), 'c': c, 'a': a, 'ends': (c + a * proj.min(), c + a * proj.max())})
    out['nostrils'] = res_ax
    return out


def label(mask):
    """4-connected components of a boolean image (no scipy): an int image, 0 for background."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        st = [(y0, x0)]
        lab[y0, x0] = n
        while st:
            y, x = st.pop()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    st.append((yy, xx))
    return lab


# ------------------------------------------------------------------------------------------------ measures

def angle_at(a, b, c):
    """The angle at b between b->a and b->c, in the profile plane (y, z), degrees."""
    u = np.array([a[1] - b[1], a[2] - b[2]])
    v = np.array([c[1] - b[1], c[2] - b[2]])
    return math.degrees(math.acos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def line_angle(a, b, c, d):
    u = np.array([b[1] - a[1], b[2] - a[2]])
    v = np.array([d[1] - c[1], d[2] - c[2]])
    return math.degrees(math.acos(np.clip(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def signed_to_line(p, a, b):
    """Distance of p from the line a-b in the profile, mm; negative behind it (toward the face)."""
    u = np.array([b[1] - a[1], b[2] - a[2]])
    u /= np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n      # (pointing forward, +z)
    return float(np.array([p[1] - a[1], p[2] - a[2]]) @ n) / MM


def measure(face):
    L, Pr, eyes = landmarks(face)
    Fm = front_measures(face, L, eyes)
    al_half = Fm['al_L'][0]
    Sm = side_measures(face, L, al_half)
    Bm = base_measures(face, L, al_half)
    M = {}
    d = lambda a, b: float(np.linalg.norm(a - b)) / MM
    # thirds
    if L['tr'] is not None:
        tot = (L['tr'][1] - L['me'][1])
        M['third_upper'] = (L['tr'][1] - L['n'][1]) / tot
        M['third_mid'] = (L['n'][1] - L['sn'][1]) / tot
        M['third_low'] = (L['sn'][1] - L['me'][1]) / tot
    M['mid_low'] = (L['n'][1] - L['sn'][1]) / (L['sn'][1] - L['me'][1])
    M['lower_split'] = (L['sn'][1] - L['sto'][1]) / (L['sto'][1] - L['me'][1])
    # eyes and nose widths (the front)
    if 'en_R' in Fm and 'en_L' in Fm:
        en_en = (Fm['en_L'][0] - Fm['en_R'][0]) / MM
        ex_en = 0.5 * ((Fm['ex_L'][0] - Fm['en_L'][0]) + (Fm['en_R'][0] - Fm['ex_R'][0])) / MM
        M['en_en_mm'] = en_en
        M['orbital'] = ex_en / en_en
        M['orbitonasal'] = 2 * al_half / MM / en_en
    nasal_h = (L['n'][1] - L['sn'][1]) / MM
    M['alar_vs_height'] = 2 * al_half / MM / nasal_h
    M['bony_base'] = Fm['bony_half'] / al_half
    M['tdp_mm'] = 2 * Fm['tdp_half'] / MM
    M['dal_vs_tdp'] = Fm['dal_half'] / Fm['tdp_half']
    # the profile's angles
    M['nasofrontal'] = angle_at(L['g'], L['n'], L['prn'])
    M['nasolabial'] = angle_at(L['cm'], L['sn'], L['ls'])
    M['nasofacial'] = line_angle(L['g'], L['pg'], L['n'], L['prn'])
    M['nasomental'] = angle_at(L['n'], L['prn'], L['pg'])
    ac = Fm['ac']
    M['goode'] = (L['prn'][2] - ac[2]) / math.hypot(L['prn'][1] - L['n'][1], L['prn'][2] - L['n'][2])
    M['eline_upper'] = signed_to_line(L['ls'], L['prn'], L['pg'])
    M['eline_lower'] = signed_to_line(L['li'], L['prn'], L['pg'])
    # columellar show: in the side view, the widest gap between the alar rim and the columella's lower edge, over the
    # middle of the span from the alar crease to the tip
    Y, Z = Pr[:, 0], Pr[:, 1]
    i_prn = int(np.argmax(np.where((Y > L['sn'][1]) & (Y < L['prn'][1] + 1), Z, -np.inf)))
    i_sn = int(np.argmin(np.abs(Y - L['sn'][1]) + np.abs(Z - L['sn'][2])))
    colY = Y[i_prn:i_sn + 1]
    colZ = Z[i_prn:i_sn + 1]
    gaps = []
    for c, u in enumerate(Sm['us']):
        if not np.isfinite(Sm['rim'][c]):
            continue
        t = (u - ac[2]) / (L['prn'][2] - ac[2])
        if t < 0.3 or t > 0.85:
            continue
        m = np.abs(colZ - u) < 0.15 * MM
        if not m.any():
            continue
        gaps.append(((Sm['rim'][c] - colY[m].min()) / MM, u, Sm['rim'][c], colY[m].min()))
    if gaps:
        g = max(gaps)
        M['col_show'] = g[0]
        Sm['show_at'] = g[1:]
    # gull wing: the columella's low point against the wing rims' highest point, from the front
    xs, edge = Fm['xs'], Fm['edge']
    cm_ = np.abs(xs) < 1.0 * MM
    wing = (np.abs(xs) > 0.3 * al_half) & (np.abs(xs) < 0.8 * al_half) & np.isfinite(edge)
    if cm_.any() and wing.any() and np.isfinite(edge[cm_]).any():
        col_low = np.nanmin(edge[cm_])
        wing_peak = np.nanmax(edge[wing])
        M['gull_drop'] = (wing_peak - col_low) / MM
        Fm['gull'] = (col_low, wing_peak)
    # the base
    nos = Bm['nostrils']
    if nos:
        apex = max(n['apex_z'] for n in nos)
        base_back = ac[2]
        col_len = apex - L['sn'][2]
        lob = L['prn'][2] - apex
        M['lobule_frac'] = lob / (lob + col_len)
        base_len = L['prn'][2] - base_back
        M['nostril_len'] = float(np.mean([n['len'] for n in nos])) / base_len
        M['nostril_axis'] = float(np.mean([n['axis'] for n in nos]))
        M['base_triangle'] = base_len / (2 * al_half)
        Bm['apex_z'] = apex
    # the nose thread's ratios
    M['wing_h_vs_w'] = Fm['wing_h'] / al_half
    M['ball_vs_bridge'] = Fm['ball_r'] / Fm['dal_half']
    M['floor_angle'] = math.degrees(math.atan2(L['cm'][1] - L['sn'][1], L['cm'][2] - L['sn'][2]))
    M['tip_over_floor'] = float(((L['prn'][1] - L['sn'][1]) - (L['prn'][2] - L['sn'][2]) * math.tan(math.radians(M['floor_angle'])))
              * math.cos(math.radians(M['floor_angle'])) / MM)
    lm = {k: (None if v is None else [round(float(q) / MM, 2) for q in v]) for k, v in L.items()}
    for k in ('en_R', 'en_L', 'ex_R', 'ex_L', 'al_R', 'al_L', 'ac'):
        if k in Fm:
            lm[k] = [round(float(q) / MM, 2) for q in Fm[k]]
    extra = {'nasal_height_mm': nasal_h, 'alar_width_mm': 2 * al_half / MM, 'ball_r_mm': Fm['ball_r'] / MM,
             'dal_mm': 2 * Fm['dal_half'] / MM, 'bony_base_mm': 2 * Fm['bony_half'] / MM,
             'wing_h_mm': Fm['wing_h'] / MM, 'nasal_length_mm': d(L['n'], L['prn'])}
    return {'L': L, 'profile': Pr, 'front': Fm, 'side': Sm, 'base': Bm, 'M': M, 'landmarks_mm': lm, 'extra': extra}


def judge(face_name, M):
    male = face_name in MALE
    rows = []
    for table, kind in ((RANGES, 'published'), (HOUSE, 'house')):
        for k, (label, unit, fr, mr, tol, src, note) in table.items():
            v = M.get(k)
            rng = (mr if (male and mr) else fr)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                verdict = 'not measured'
            elif rng is None:
                verdict = 'info'
            elif rng[0] <= v <= rng[1]:
                verdict = 'ok'
            elif rng[0] - tol <= v <= rng[1] + tol:
                verdict = 'near'
            else:
                verdict = 'out'
            rows.append({'id': k, 'label': label, 'unit': unit, 'value': None if v is None else round(float(v), 3),
                         'range': rng, 'verdict': verdict, 'kind': kind, 'sources': src, 'note': note,
                         'sex': 'male' if male else 'female'})
    return rows


# ------------------------------------------------------------------------------------------------ drawing

def clay(V, F, zb, tb, light=(-0.45, 0.55, 0.7), axes=(0, 1, 2), signs=(1, 1, 1)):
    """Matcap-like clay shading from the per-pixel triangle normals (no materials)."""
    N = tri_normals(V, F)
    N = N[:, list(axes)] * np.array(signs)
    l = np.array(light, float)
    l /= np.linalg.norm(l)
    m = tb >= 0
    n = N[np.maximum(tb, 0)]
    lam = np.clip(n @ l, 0, 1)
    rim = np.clip(n[..., 2], 0, 1)
    v = 0.18 + 0.62 * lam + 0.2 * rim ** 2
    img = np.zeros(tb.shape + (3,))
    base = np.array([0.80, 0.74, 0.68])
    img[:] = np.array([0.13, 0.14, 0.16])
    img[m] = (v[m, None] * base)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


COL = {'ok': (90, 200, 110), 'near': (240, 190, 60), 'out': (235, 80, 70), 'info': (150, 170, 220),
       'not measured': (150, 150, 150)}


def font(sz):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw_face(R, rows, title):
    V, F = R['face']['head']
    L = R['L']
    v = {r['id']: r for r in rows}
    S = 2.2      # px per mm in the panels
    f = font(15)
    fs = font(13)

    def col(k):
        return COL[v[k]['verdict']] if k in v else (255, 255, 255)

    def lab(k):
        r = v[k]
        return '%s %s' % (('%.2f' if abs(r['value']) < 10 else '%.0f') % r['value'] if r['value'] is not None else '-',
                          r['unit'])
    panels = []
    # 1. front: full face
    res = 1.0 / S * MM
    top = (L['tr'][1] if L['tr'] is not None else L['g'][1] + 60 * MM) + 12 * MM
    lo, hi = (-80 * MM, L['me'][1] - 14 * MM), (80 * MM, top)
    zb, tb, win = raster(V, F, np.arange(len(F)), lambda P: P[:, [0, 1, 2]], lo, hi, res)
    im = clay(V, F, zb, tb).convert('RGB')
    dr = ImageDraw.Draw(im)
    W = im.width
    P = lambda x, y: ((x - lo[0]) / res, (hi[1] - y) / res)
    Fm = R['front']
    for k, key in (('tr', 'third_upper'), ('n', 'third_mid'), ('sn', 'third_low'), ('me', None)):
        if L[k] is None:
            continue
        y = P(0, L[k][1])[1]
        dr.line([(0, y), (W, y)], fill=(120, 200, 255), width=1)
        dr.text((4, y - 15), k, fill=(120, 200, 255), font=fs)
    # thirds' values beside their bands
    pairs = [('tr', 'n', 'third_upper'), ('n', 'sn', 'third_mid'), ('sn', 'me', 'third_low')]
    for a, b, k in pairs:
        if L[a] is None or k not in v:
            continue
        ym = P(0, 0.5 * (L[a][1] + L[b][1]))[1]
        dr.text((W - 92, ym - 8), lab(k), fill=col(k), font=f)
    # sto line
    y = P(0, L['sto'][1])[1]
    dr.line([(W * 0.3, y), (W * 0.7, y)], fill=col('lower_split'), width=1)
    dr.text((W * 0.7 + 4, y - 8), 'sto ' + lab('lower_split'), fill=col('lower_split'), font=fs)
    # eyes and alae
    for nm in ('en_R', 'ex_R', 'en_L', 'ex_L', 'al_R', 'al_L'):
        if nm in Fm:
            x, y = P(Fm[nm][0], Fm[nm][1])
            c = (255, 120, 220) if nm.startswith('al') else (255, 230, 120)
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=c, width=2)
            dr.line([(x, y - 60 if nm.startswith('en') else y - 18), (x, y + 18)], fill=c, width=1)
    if 'en_R' in Fm:
        y = P(0, Fm['en_R'][1])[1] - 40
        dr.text((P(Fm['en_R'][0], 0)[0] + 2, y - 16), 'en-en ' + lab('en_en_mm'), fill=col('en_en_mm'), font=fs)
        dr.text((4, y + 90), 'orbital ' + lab('orbital'), fill=col('orbital'), font=fs)
    y = P(0, Fm['al_R'][1])[1]
    dr.line([P(Fm['al_R'][0], Fm['al_R'][1]), P(Fm['al_L'][0], Fm['al_L'][1])], fill=(255, 120, 220), width=1)
    dr.text((P(Fm['al_L'][0], 0)[0] + 6, y + 2), 'al/en ' + lab('orbitonasal'), fill=col('orbitonasal'), font=fs)
    dr.text((4, 4), title + ' - front', fill=(255, 255, 255), font=f)
    panels.append(im)
    # 2. profile, with the lines
    res = 1.0 / S * MM
    lo, hi = (L['me'][2] - 70 * MM, L['me'][1] - 14 * MM), (L['prn'][2] + 22 * MM, top)
    zb, tb, win = raster(V, F, np.arange(len(F)), lambda Q: np.stack([Q[:, 2], Q[:, 1], Q[:, 0]], 1), lo, hi, res)
    im = clay(V, F, zb, tb, light=(0.5, 0.55, 0.65), axes=(2, 1, 0)).convert('RGB')
    dr = ImageDraw.Draw(im)
    Q = lambda p: ((p[2] - lo[0]) / res, (hi[1] - p[1]) / res)
    pr = R['profile']
    dr.line([((z - lo[0]) / res, (hi[1] - y) / res) for y, z in pr[::3]], fill=(250, 250, 250), width=1)

    def ray(a, b, ext=1.25, c=(255, 255, 255)):
        pa, pb = Q(a), Q(b)
        pe = (pa[0] + (pb[0] - pa[0]) * ext, pa[1] + (pb[1] - pa[1]) * ext)
        dr.line([pa, pe], fill=c, width=2)
    ray(L['n'], L['g'], 1.3, col('nasofrontal'))
    ray(L['n'], L['prn'], 1.15, col('nasofrontal'))
    ray(L['sn'], L['cm'], 1.6, col('nasolabial'))
    ray(L['sn'], L['ls'], 1.5, col('nasolabial'))
    ray(L['g'], L['pg'], 1.0, col('nasofacial'))
    ray(L['prn'], L['pg'], 1.08, col('eline_lower'))
    ac = Fm['ac']
    xa = Q(ac)[0]
    dr.line([(xa, Q(ac)[1] - 40), (xa, Q(ac)[1] + 30)], fill=col('goode'), width=1)
    for k in ('tr', 'g', 'n', 'prn', 'cm', 'sn', 'ls', 'sto', 'li', 'sm', 'pg', 'me'):
        if L[k] is None:
            continue
        x, y = Q(L[k])
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 230, 120))
        dr.text((x + 5, y - 7), k, fill=(255, 230, 120), font=fs)
    txt = [('nasofrontal', 'NF'), ('nasolabial', 'NL'), ('nasofacial', 'NFac'), ('nasomental', 'NM'), ('goode', 'Goode'),
           ('eline_upper', 'E-line ls'), ('eline_lower', 'E-line li'), ('col_show', 'col. show')]
    for i, (k, short) in enumerate(txt):
        dr.text((4, 26 + 17 * i), '%s %s' % (short, lab(k)), fill=col(k), font=f)
    dr.text((4, 4), title + ' - profile', fill=(255, 255, 255), font=f)
    panels.append(im)
    # 3. nose close-ups: front (gull wing, reflex lines), side (columellar show), base (nostrils)
    S2 = 7.0
    res = 1.0 / S2 * MM
    lo, hi = (-22 * MM, L['sn'][1] - 8 * MM), (22 * MM, L['n'][1] + 4 * MM)
    zb, tb, win = raster(V, F, np.arange(len(F)), lambda P_: P_[:, [0, 1, 2]], lo, hi, res)
    im = clay(V, F, zb, tb).convert('RGB')
    dr = ImageDraw.Draw(im)
    P = lambda x, y: ((x - lo[0]) / res, (hi[1] - y) / res)
    xs, edge = Fm['xs'], Fm['edge']
    pts = [P(x, e) for x, e in zip(xs, edge) if np.isfinite(e)]
    if len(pts) > 2:
        dr.line(pts, fill=col('gull_drop'), width=2)
    for half, yv, k in ((Fm['dal_half'], Fm['y_dal'], 'dal_vs_tdp'), (Fm['tdp_half'], Fm['y_tdp'], 'tdp_mm'),
                        (Fm['bony_half'], Fm['y_bb'], 'bony_base')):
        for s_ in (-1, 1):
            x, y = P(s_ * half, yv)
            dr.line([(x, y - 6), (x, y + 6)], fill=col(k), width=2)
        dr.line([P(-half, yv), P(half, yv)], fill=col(k), width=1)
    for s_ in (-1, 1):
        x, y = P(s_ * Fm['al_L'][0], Fm['al_L'][1])
        dr.line([(x, y - 30), (x, y + 30)], fill=col('alar_vs_height'), width=1)
    dr.text((4, 4), 'nose front', fill=(255, 255, 255), font=f)
    for i, (k, short) in enumerate([('gull_drop', 'gull'), ('tdp_mm', 'TDP'), ('dal_vs_tdp', 'DAL/TDP'),
                                    ('bony_base', 'bony/al'), ('alar_vs_height', 'al/n-sn'), ('wing_h_vs_w', 'wing h/w'),
                                    ('ball_vs_bridge', 'ball/bridge')]):
        dr.text((4, 24 + 16 * i), '%s %s' % (short, lab(k)), fill=col(k), font=fs)
    panels.append(im)
    # side close-up
    lo, hi = (L['sn'][2] - 22 * MM, L['sn'][1] - 8 * MM), (L['prn'][2] + 4 * MM, L['prn'][1] + 14 * MM)
    zb, tb, win = raster(V, F, np.arange(len(F)), lambda Q_: np.stack([Q_[:, 2], Q_[:, 1], Q_[:, 0]], 1), lo, hi, res)
    im = clay(V, F, zb, tb, light=(0.5, 0.55, 0.65), axes=(2, 1, 0)).convert('RGB')
    dr = ImageDraw.Draw(im)
    Q = lambda p: ((p[2] - lo[0]) / res, (hi[1] - p[1]) / res)
    Sm = R['side']
    pts = [((u - lo[0]) / res, (hi[1] - r) / res) for u, r in zip(Sm['us'], Sm['rim']) if np.isfinite(r)]
    if len(pts) > 2:
        dr.line(pts, fill=(255, 120, 220), width=2)
    if 'show_at' in Sm:
        u, a, b = Sm['show_at']
        dr.line([((u - lo[0]) / res, (hi[1] - a) / res), ((u - lo[0]) / res, (hi[1] - b) / res)], fill=col('col_show'),
                width=3)
    ray(L['sn'], L['cm'], 1.6, col('nasolabial'))
    for k in ('prn', 'cm', 'sn'):
        x, y = Q(L[k])
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 230, 120))
        dr.text((x + 5, y - 7), k, fill=(255, 230, 120), font=fs)
    dr.text((4, 4), 'nose side', fill=(255, 255, 255), font=f)
    dr.text((4, 24), 'col. show ' + lab('col_show'), fill=col('col_show'), font=fs)
    dr.text((4, 40), 'floor ' + lab('floor_angle'), fill=col('floor_angle'), font=fs)
    panels.append(im)
    # base close-up
    Bm = R['base']
    zb, tb, win, keep = Bm['base']
    lo_b, hi_b, res_b = win[0], win[1], win[2]
    im = clay(V, F, zb, tb, light=(-0.3, 0.4, 0.8), axes=(0, 2, 1), signs=(1, 1, -1)).convert('RGB')
    im = im.resize((int(im.width * (S2 * res_b / MM)), int(im.height * (S2 * res_b / MM))))
    dr = ImageDraw.Draw(im)
    sc = 1.0 / (MM / S2)
    B = lambda x, z: ((x - lo_b[0]) * sc, (hi_b[1] - z) * sc)
    for n in Bm['nostrils']:
        a, b = n['ends']
        dr.line([B(*a), B(*b)], fill=col('nostril_axis'), width=2)
    if 'apex_z' in Bm:
        y = B(0, Bm['apex_z'])[1]
        dr.line([(0, y), (im.width, y)], fill=col('lobule_frac'), width=1)
    for z_, c in ((L['prn'][2], (255, 230, 120)), (L['sn'][2], (255, 230, 120)), (R['front']['ac'][2], (255, 120, 220))):
        y = B(0, z_)[1]
        dr.line([(im.width * 0.35, y), (im.width * 0.65, y)], fill=c, width=1)
    dr.text((4, 4), 'nose base', fill=(255, 255, 255), font=f)
    for i, (k, short) in enumerate([('lobule_frac', 'lobule'), ('nostril_len', 'nostril/base'),
                                    ('nostril_axis', 'nostril axis'), ('base_triangle', 'len/width')]):
        dr.text((4, 24 + 16 * i), '%s %s' % (short, lab(k)), fill=col(k), font=fs)
    panels.append(im)
    # one row
    Hh = max(p.height for p in panels)
    row = Image.new('RGB', (sum(p.width for p in panels) + 6 * len(panels), Hh), (24, 25, 28))
    x = 0
    for p in panels:
        row.paste(p, (x, (Hh - p.height) // 2))
        x += p.width + 6
    return row


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dump')
    ap.add_argument('--out', required=True)
    ap.add_argument('--faces', default='Ranger,Rogue,Warden,Sorceress,Knight')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {'version': 1, 'mm_per_unit': 1 / MM, 'sources': SRC, 'faces': {}}
    rows_img = []
    for nm in a.faces.split(','):
        p = os.path.join(a.dump, nm + '.json')
        if not os.path.exists(p):
            print('no dump for', nm)
            continue
        face = load(p)
        R = measure(face)
        R['face'] = face
        rows = judge(nm, R['M'])
        report['faces'][nm] = {'landmarks_mm': R['landmarks_mm'], 'extra': {k: round(float(v), 2)
                               for k, v in R['extra'].items()}, 'measures': rows}
        fails = [r['label'] for r in rows if r['verdict'] == 'out']
        print('%-10s out: %s' % (nm, '; '.join(fails) or 'none'))
        rows_img.append(draw_face(R, rows, nm))
    json.dump(report, open(os.path.join(a.out, 'face-lines.json'), 'w'), indent=1)
    if rows_img:
        Wd = max(r.width for r in rows_img)
        key = Image.new('RGB', (Wd, 30), (24, 25, 28))
        dk = ImageDraw.Draw(key)
        x = 8
        for k in ('ok', 'near', 'out', 'info'):
            dk.rectangle([x, 9, x + 14, 23], fill=COL[k])
            dk.text((x + 20, 7), {'ok': 'in range', 'near': 'just outside', 'out': 'out of range',
                                  'info': 'house ratio, shown'}[k], fill=(230, 230, 230), font=font(15))
            x += 170
        sheet = Image.new('RGB', (Wd, 30 + sum(r.height + 8 for r in rows_img)), (24, 25, 28))
        sheet.paste(key, (0, 0))
        y = 30
        for r in rows_img:
            sheet.paste(r, (0, y))
            y += r.height + 8
        sheet.save(os.path.join(a.out, 'face-lines-sheet.png'))
    # the short list
    md = ['# Facial proportion check', '', 'Measures out of range (just outside in brackets):', '']
    for nm, F_ in report['faces'].items():
        out = [r for r in F_['measures'] if r['verdict'] == 'out']
        near = [r for r in F_['measures'] if r['verdict'] == 'near']
        fmt = lambda r: '%s %s%s (range %s-%s)' % (r['label'], r['value'], (' ' + r['unit']) if r['unit'] else '',
                                                     r['range'][0], r['range'][1])
        md.append('- **%s**: %s%s' % (nm, '; '.join(fmt(r) for r in out) or 'none',
                                      (' [' + '; '.join(fmt(r) for r in near) + ']') if near else ''))
    md += ['', '## Ranges and sources', '', '| Measure | Women | Men | Sources | Note |', '|---|---|---|---|---|']
    for k, (label, unit, fr, mr, tol, src, note) in RANGES.items():
        md.append('| %s | %s | %s | %s | %s |' % (label, '%s-%s %s' % (fr[0], fr[1], unit),
                                                 ('%s-%s %s' % (mr[0], mr[1], unit)) if mr else 'same',
                                                 ', '.join(src), note))
    md += [''] + ['- %s: %s' % (k, v) for k, v in SRC.items()]
    open(os.path.join(a.out, 'face-lines.md'), 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
