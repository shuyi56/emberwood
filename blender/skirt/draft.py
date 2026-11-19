"""The riding habit's outer skirt, drafted on whatever body it is given: a long, closed A-line skirt of navy wool, smooth
over the hips, with a deep hem and a modest train, joined at the waist under the bodice's basque (ca. 1885-1895).

Nothing here is placed by hand. The body is measured first (measure(): the waist, the hips, the seat, the crotch, the
knee and the floor, read off the body's rest shape and its bones), and every size of the skirt is a share of the head's
height H, the figure's own unit (anatomy.py builds the body from it), so the skirt re-fits when the body or the head
changes:

  1. the body's sections   at every height the trunk and thighs are cut level and wrapped in their convex hull: the
                           outline cloth takes over a body (it bridges the waist's hollow and the cleft of the seat
                           and cannot follow them), as the distance from the body's axis at each angle.
  2. the pattern           from the waist to the hips the skirt is the hull let out by the ease (the petticoat's bulk
                           and the wool's thickness); below the hips it flares from that outline to the hem's own
                           shape (front, side and back radii, the back's longer for the train), the flare growing
                           slowly at first and faster toward the hem, as a gored skirt hangs.
  3. the hem               a lifted hem in front (the toes clear it), on the floor behind, with the train lying out
                           behind on the ground; a turned and stitched deep hem, and the seams of the gores.
  4. the rig               the pelvis carries the top, and each thigh a share of the cloth rising to the hem, as the
                           gown's skirt does, and a push modifier keeps the legs out of the cloth.

The pattern is laid out as the tailor cuts it: the UV map ('pattern') is the cloth's own length down and its girth
round, in model units, so the weave runs true; the seams, the hem and the waist reach the shader as attributes."""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

# Every size as a share of the head's height H (9.4 units on the mannequin; one unit is about 2.6 cm).
RATIOS = {
    'ease_waist': 0.02,     # the cloth's stand-off from the skin at the waist: close, the wool's thickness only
    'ease_hips': 0.05,      # over the hips: the petticoat's bulk under it (the petticoat thread's published value
    'ease_hem': 0.10,        # at the hem, and the petticoat's own flare below that, replace these when it lands)
    'top_above': 0.045,      # the top edge above the waist, under the jacket's basque
    'hem_front': 1.45,       # the hem's radius in front, at the sides and at the back, from the axis (the train's)
    'hem_side': 1.5,
    'hem_back': 1.85,
    'hem_lift': 0.045,       # the hem's height above the floor in front, at the sides; it lies on the floor behind
    'hem_lift_side': 0.035,
    'flare_from': 0.0,       # the flare starts at the hips
    'hem_band': 0.105,       # the deep hem's depth (the turned and stitched edge)
    'wool': 0.0085,          # the wool's thickness (about 2 mm); the hem's turn doubles it
    'nap_roll': 0.012,       # the soft roll of the turned hem
    'fold_depth': 0.11,     # the soft flutes the flare falls into at the hem, before the cloth settles
}
FLARE = 1.1                  # how the flare grows toward the hem (1 is straight from the hips)
GORES = 7                    # seams at +-0.55, +-1.45, +-2.4 rad from the front and one up the middle of the back
SEAMS = [-2.4, -1.45, -0.55, 0.55, 1.45, 2.4, math.pi]
ARM_GROUPS = ('UpperArm', 'Forearm', 'Hand')
NZ, NPHI = 96, 241           # the sections' table: heights and angles


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _hull(pts):
    """The convex hull of 2-D points (monotone chain), counter-clockwise."""
    p = sorted(set((round(float(a), 5), round(float(b), 5)) for a, b in pts))
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return lo[:-1] + up[:-1]


def _girth(pts):
    if len(pts) < 3:
        return 0.0
    h = np.array(_hull(pts))
    return float(np.linalg.norm(h - np.roll(h, 1, 0), axis=1).sum())


def _polar(hull, phis):
    """The hull's distance from the origin along each direction phi (0 the front, -y; her right -x, as the gown's)."""
    h = np.array(hull)
    out = np.zeros(len(phis))
    dx, dy = -np.sin(phis), -np.cos(phis)
    A, Bv = h, np.roll(h, -1, 0)
    e = Bv - A
    for k in range(len(h)):
        # solve A + u e = r d  (u in 0..1, r > 0)
        den = dx * e[k, 1] - dy * e[k, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = (A[k, 0] * e[k, 1] - A[k, 1] * e[k, 0]) / den
            u = (A[k, 0] * dy - A[k, 1] * dx) / den
        ok = (np.abs(den) > 1e-12) & (u >= -1e-9) & (u <= 1 + 1e-9) & (r > 0)
        out = np.where(ok, np.maximum(out, r), out)
    return out


def body_data(body):
    """The body's rest shape: points in the world, and its bone weights as a table [point, group]."""
    me = body.data
    n = len(me.vertices)
    P = np.empty(n * 3)
    me.vertices.foreach_get('co', P)
    P = P.reshape(-1, 3) @ np.array(body.matrix_world)[:3, :3].T + np.array(body.matrix_world)[:3, 3]
    names = [g.name for g in body.vertex_groups]
    W = np.zeros((n, len(names)), dtype=np.float32)
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    return dict(P=P, W=W, groups=names)


def measure(B, rig, H):
    """The tailor's measurements, from the skin's rest shape and the bones (rest), in model units:

    H                      the head's height, the unit every ratio is a share of
    floor_z                the soles
    hip_z, knee_z          the hip joint and the knee (the bones' heads)
    crotch_z               the lowest point of the trunk in the middle
    waist_z, waist         the narrowest section of the trunk between the hips and the bust, and its girth
    hips_z, hips           the widest section of the hips, and its girth
    seat_z                 where the seat stands furthest back"""
    P, W, names = B['P'], B['W'], B['groups']
    col = {n: i for i, n in enumerate(names)}
    arm = sum(W[:, col[n]] for n in names if n.split('.')[0] in ARM_GROUPS)
    bones = rig.data.bones
    M = {'H': float(H), 'floor_z': float(P[:, 2].min())}
    M['hip_z'] = float(bones['Thigh.R'].head_local.z)
    M['knee_z'] = float(bones['Shin.R'].head_local.z)
    trunk = arm < 0.3
    mid = trunk & (np.abs(P[:, 0]) < 0.25)
    M['crotch_z'] = float(P[mid & (P[:, 2] < M['hip_z']) & (P[:, 2] > M['knee_z']), 2].min())

    def section(z, dz=0.07):
        m = trunk & (np.abs(P[:, 2] - z) < dz)
        return P[m][:, :2]
    zs = np.arange(M['hip_z'], M['hip_z'] + 1.4 * H, 0.2)
    g = [_girth(section(z)) for z in zs]
    i = int(np.argmin(g))
    M['waist_z'], M['waist'] = float(zs[i]), g[i]
    zs = np.arange(M['crotch_z'] + 0.3, M['waist_z'], 0.2)
    g = [_girth(section(z)) for z in zs]
    i = int(np.argmax(g))
    M['hips_z'], M['hips'] = float(zs[i]), g[i]
    back = [section(z)[:, 1].max() for z in zs]
    M['seat_z'] = float(zs[int(np.argmax(back))])
    return M


def report(M, log=print):
    cm = 165.0 / 63.0
    H = M['H']
    log('skirt measures: ' + ', '.join('%s %.2f H (%.0f cm)' % (k, M[k] / H, M[k] * cm) for k in ('waist', 'hips')))
    log('skirt heights: ' + ', '.join('%s %.2f' % (k, M[k]) for k in
                                      ('waist_z', 'hips_z', 'seat_z', 'hip_z', 'crotch_z', 'knee_z', 'floor_z')))


def sections(B, M, zs):
    """The body's outline at each height in zs, as the distance from the axis at each angle (shape [len(zs), NPHI]):
    the convex hull of the trunk and legs cut level there, arms left out."""
    P, W, names = B['P'], B['W'], B['groups']
    col = {n: i for i, n in enumerate(names)}
    arm = sum(W[:, col[n]] for n in names if n.split('.')[0] in ARM_GROUPS)
    keep = arm < 0.3
    phis = np.linspace(0.0, 2 * math.pi, NPHI)
    R = np.zeros((len(zs), NPHI))
    dz = 0.35 * (zs[0] - zs[-1]) / max(1, len(zs) - 1) + 0.05
    for i, z in enumerate(zs):
        m = keep & (np.abs(P[:, 2] - z) < dz)
        if m.sum() < 8:
            R[i] = R[i - 1] if i else 1.0
            continue
        R[i] = _polar(_hull(P[m][:, :2]), phis)
    return phis, R


def hem_shape(phi, M, r):
    """The hem's xy at the angle phi (as the gown's: x = -r sin, y = -r cos; the front -y), the train out behind."""
    c, s = np.cos(phi), np.sin(phi)
    # (the back's radius swells toward the middle of the back, so the train is a tongue and not a wider circle)
    k = np.where(c >= 0, 0.0, (-c) ** 2)
    rf = r['hem_front'] + (r['hem_back'] - r['hem_front']) * k
    return -r['hem_side'] * s, -rf * c


def hem_height(phi, M, r):
    c = np.cos(phi)
    front = np.maximum(c, 0.0) ** 2
    back = np.maximum(-c, 0.0) ** 2
    side = np.sin(phi) ** 2
    return M['floor_z'] + r['hem_lift'] * front + r['hem_lift_side'] * side + 0.0 * back


def under_body(points):
    """A body-like record for the sections() of whatever lies under the skirt (the petticoat's evaluated points): the
    skirt's inner surface is its outer one plus the ease."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    return dict(P=P, W=np.zeros((len(P), 1), dtype=np.float32), groups=['none'])


def draft(B, M, ratios=RATIOS, under=None, log=print):
    """The skirt's surface (under: under_body() of the petticoat, if it is worn): a sheet of rows down and columns round, its points (world), its pattern coordinates and its
    attributes. Returns a dict."""
    H = M['H']
    r = {k: v * H if k in ('ease_waist', 'ease_hips', 'ease_hem', 'top_above', 'hem_front', 'hem_side', 'hem_back',
                           'hem_lift', 'hem_lift_side', 'hem_band', 'wool', 'nap_roll', 'fold_depth') else v
         for k, v in ratios.items()}
    z_top = M['waist_z'] + r['top_above']
    z_hem = M['floor_z']
    zs = np.linspace(z_top + 0.3, z_hem, NZ)
    phis, R = sections(B, M, zs)
    if under is not None:
        # (the petticoat's outline, from its own points, and the skirt stands the ease off it: the larger of the two)
        _, Ru = sections(under, M, zs)
        R = np.maximum(R, Ru)
    # the outline settled: smoothed down the height (the body's sections are not perfectly regular) and round
    for _ in range(3):
        R[1:-1] = 0.25 * R[:-2] + 0.5 * R[1:-1] + 0.25 * R[2:]
        R[:, 1:-1] = 0.25 * R[:, :-2] + 0.5 * R[:, 1:-1] + 0.25 * R[:, 2:]
    R[:, -1] = R[:, 0]

    def body_r(z, phi):
        zi = np.clip((zs[0] - z) / (zs[0] - zs[-1]) * (NZ - 1), 0, NZ - 1.0001)
        i = int(zi)
        f = zi - i
        row = (1 - f) * R[i] + f * R[i + 1]
        return np.interp(phi, phis, row)

    zh = M['hips_z']
    rows, cols = 76, 160

    def ease(z):
        e = r['ease_waist'] + (r['ease_hips'] - r['ease_waist']) * _ss(M['waist_z'], zh, z)
        return e + (r['ease_hem'] - r['ease_hips']) * _ss(zh, z_hem, z) ** 1.3 * (z < zh)

    P = np.zeros((rows, cols, 3))
    UV = np.zeros((rows, cols, 2))
    seam = np.zeros((rows, cols))
    hem_d = np.zeros((rows, cols))
    top_d = np.zeros((rows, cols))
    phi_c = 2 * math.pi * np.arange(cols) / cols
    hem_z_c = hem_height(phi_c, M, r)
    hx, hy = hem_shape(phi_c, M, r)
    fold_n = 6
    for j, phi in enumerate(phi_c):
        rmax = 0.0
        run = 0.0
        for i in range(rows):
            v = i / (rows - 1)
            z = z_top + (hem_z_c[j] - z_top) * v
            br = float(body_r(z, phi)) + float(ease(z))
            s = float(np.clip((zh - z) / (zh - hem_z_c[j]), 0.0, 1.0))
            t = s ** FLARE
            x0, y0 = -br * math.sin(phi), -br * math.cos(phi)
            x1, y1 = hx[j], hy[j]
            # (the flare in xy, as the gown's: the hips' outline let out toward the hem's)
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            rr = math.hypot(x, y)
            # soft flutes where the flare is free of the legs: deepest toward the hem and at the back, none in front
            fl = r['fold_depth'] * t ** 1.6 * (0.35 + 0.65 * min(1.0, max(0.0, -math.cos(phi) + 0.35)))
            rr2 = rr + fl * math.cos(fold_n * phi + 0.4 * math.sin(3 * phi) + 6.0 * s)
            x, y = x * rr2 / rr, y * rr2 / rr
            # (the cloth never turns back in on itself below the hips)
            if z < zh:
                rr3 = math.hypot(x, y)
                if rr3 < rmax:
                    x, y = x * rmax / rr3, y * rmax / rr3
            rmax = max(rmax, math.hypot(x, y))
            P[i, j] = (x, y, z)
        # the cloth's own lengths, down the column
        d = np.linalg.norm(np.diff(P[:, j], axis=0), axis=1)
        down = np.concatenate([[0.0], np.cumsum(d)])
        UV[:, j, 1] = down
        hem_d[:, j] = down[-1] - down
        top_d[:, j] = down
    # the cloth's girth round: the length of the row between the neighbouring columns (u in model units)
    for i in range(rows):
        seg = np.linalg.norm(np.diff(np.concatenate([P[i, :, :2], P[i, :1, :2]]), axis=0), axis=1)
        UV[i, :, 0] = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    # the seams: the distance along the row from the nearest gore seam
    for i in range(rows):
        rad = np.hypot(P[i, :, 0], P[i, :, 1])
        ds = np.min([np.abs((phi_c - s_ + math.pi) % (2 * math.pi) - math.pi) for s_ in SEAMS], axis=0)
        seam[i] = ds * rad
    log('skirt: %d x %d points; top %.2f, hem %.2f..%.2f above the floor; radius at the hips %.2f..%.2f, at the hem '
        '%.2f front, %.2f side, %.2f back' % (
            rows, cols, z_top, float(hem_z_c.min() - z_hem), float(hem_z_c.max() - z_hem),
            float(np.hypot(P[:, :, 0], P[:, :, 1])[np.argmin(abs(P[:, 0, 2] - zh))].min()),
            float(np.hypot(P[:, :, 0], P[:, :, 1])[np.argmin(abs(P[:, 0, 2] - zh))].max()),
            float(np.hypot(P[-1, 0, 0], P[-1, 0, 1])), float(np.hypot(P[-1, cols // 4, 0], P[-1, cols // 4, 1])),
            float(np.hypot(P[-1, cols // 2, 0], P[-1, cols // 2, 1]))))
    return dict(P=P, UV=UV, seam=seam, hem_d=hem_d, top_d=top_d, rows=rows, cols=cols, r=r, z_top=z_top)


def sheet_mesh(D):
    """The sheet as a closed ring of quads, normals out; the pattern UVs and attributes ride on it."""
    rows, cols = D['rows'], D['cols']
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new('pattern')
    la = {k: bm.verts.layers.float.new(k) for k in ('seam', 'hem_d', 'top_d')}
    V = [[bm.verts.new(tuple(D['P'][i, j])) for j in range(cols)] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for k in la:
                V[i][j][la[k]] = float(D[k][i, j])
    for i in range(rows - 1):
        for j in range(cols):
            j2 = (j + 1) % cols
            f = bm.faces.new((V[i][j], V[i][j2], V[i + 1][j2], V[i + 1][j]))
            for lp, (ii, jj, wrap) in zip(f.loops, ((i, j, 0), (i, j + 1, 1), (i + 1, j + 1, 1), (i + 1, j, 0))):
                jj_ = jj if jj < cols else cols - 1
                u = D['UV'][ii, jj_, 0]
                if jj == cols:          # (the last column's far edge: the girth's end, a whole round on)
                    u = D['UV'][ii, cols - 1, 0] + float(np.linalg.norm(D['P'][ii, cols - 1, :2] - D['P'][ii, 0, :2]))
                lp[uvl].uv = (u, D['UV'][ii, jj_, 1])
    bm.normal_update()
    # outward: a face's normal should point away from the axis
    bm.faces.ensure_lookup_table()
    f0 = bm.faces[len(bm.faces) // 3]
    c = f0.calc_center_median()
    if f0.normal.x * c.x + f0.normal.y * c.y < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    return bm
