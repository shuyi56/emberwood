"""The skirt sewn: the flat pieces (pattern.py) cut as cloth, laid on the body, and joined along their paired edges.

John said go (2026-11-19: stitch the skirt). What this module does, and in what order:

  1. cut      each gore is its own piece of cloth: a grid of points down and across, its outer layer from the waist
              seam to the hem line, then the hem: the cloth turns inward in a round fold (radius `fold_r`), and the
              hem allowance goes back up the inside, a hem band high, where it is sewn down. The pieces carry the
              pattern's own coordinates (the 'pattern' UVs are the flat piece, so the weave runs along each gore's
              grain), each cut at its own place on the cloth.
  2. waistband  a ring of straight cloth, folded double over its top edge, its outer layer standing on the gores'
              top edge, its facing hanging down inside it.
  3. sew      the paired edges are joined: the seam edges of neighbouring gores, the gores' top edges to the
              waistband's lower edge, the waistband's two ends. A seam is one line of points the two pieces share (the
              points are welded at the stitch line, as a seam draws the two edges together); the seam allowances are
              cut beyond it and pressed to one side inside, as separate strips of the wool.
  4. stitches the stitch lines are geometry: dashed thread, a stitch long and a gap, lying on the inside of every seam,
              along the waist join and along the hem's inner edge.

Everything reads the pieces and the draft, so it follows the body and the pattern: a different body gives a different
pattern, and the sewing follows."""
import math

import bmesh
import numpy as np

STITCH = 0.26             # a stitch's length
GAP = 0.12                # the gap between stitches
THREAD = 0.028            # the thread's width
INSET = 0.035             # how far inside the cloth the stitches and the pressed allowances lie
COLUMN_ROWS_UP = 6        # the hem allowance's rows back up the inside


def _inward(P):
    """The horizontal unit vector from a point toward the skirt's axis."""
    r = math.hypot(P[0], P[1]) or 1.0
    return np.array([-P[0] / r, -P[1] / r, 0.0])


def _at_height(col, z):
    """The point of an outer column (rows from the top down) at height z, by interpolation."""
    zs = col[:, 2]
    i = int(np.clip(np.searchsorted(-zs, -z) - 1, 0, len(zs) - 2))
    f = (z - zs[i]) / (zs[i + 1] - zs[i] if zs[i + 1] != zs[i] else 1.0)
    return col[i] + (col[i + 1] - col[i]) * f


def _hem_column(col, hz, rho, band, rows_up=COLUMN_ROWS_UP):
    """What follows an outer column at its hem: the fold's points (round the turn) and the hem allowance's, back up the
    inside. Returns fold (3, 3) and inner (rows_up + 1, 3), and the heights above the fold's lowest point."""
    Pe = col[-1]
    u = _inward(Pe)
    fold = [Pe + u * rho * (1 - math.cos(th)) + np.array([0.0, 0.0, -rho * math.sin(th)])
            for th in (math.pi / 4, math.pi / 2, 3 * math.pi / 4)]
    inner = []
    hs = []
    for k in range(rows_up + 1):
        z = hz + band * k / rows_up
        Q = _at_height(col, z)
        inner.append(Q + _inward(Q) * 2 * rho)
        hs.append(band * k / rows_up)
    return np.array(fold), np.array(inner), hs


def _flat_extend(F, rows_extra_y):
    """The flat coordinates of the points below a gore's last row: straight on from it, along the grain."""
    return [F[-1] + np.array([0.0, y]) for y in rows_extra_y]


def sew(D, M, pieces, log=print):
    """Cut the pieces as cloth, join them, and return the sewn skirt as a bmesh (plus the data the stitches need)."""
    r = D['r']
    column = D['column']
    rho = r['fold_r']
    band = r['hem_band']
    z_top = D['z_top']
    wb_h = r['band']
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new('pattern')
    la = {k: bm.verts.layers.float.new(k) for k in ('seam', 'hem_d', 'top_d')}
    gores = [p for p in pieces if p['kind'] == 'gore']
    allphis = []
    seam_cols = {}              # the seam angle -> its outer column (the stitch line)
    face_sets = []
    inner_edges = []            # (gore's phis, [inner layer's top edge points]) for the hem stitch line
    # ---- 1. cut: every gore a piece of cloth with its hem
    for k, g in enumerate(gores):
        phis = g['phis']
        n = len(phis)
        cols_outer, folds, inners, hs = [], [], [], None
        for ph in phis:
            col, hz = column(ph)
            f, inn, hs = _hem_column(col, hz, rho, band)
            cols_outer.append(col)
            folds.append(f)
            inners.append(inn)
        rows = len(cols_outer[0])
        Fl = g['flat']
        shift = np.array([37.0 * k, 13.0 * k])                      # (each gore cut at its own place on the cloth)
        # the flat coordinates of the fold and the hem allowance: straight on down from the last row, along the grain
        y_fold = [0.5 * math.pi * rho * s for s in (0.5, 1.0, 1.5)]
        y_in = [math.pi * rho + band * kk / COLUMN_ROWS_UP for kk in range(COLUMN_ROWS_UP + 1)]
        V = []
        for j in range(n):
            col = []
            for i in range(rows):
                col.append((cols_outer[j][i], Fl[i, j], 50.0))
            for s in range(3):
                col.append((folds[j][s], Fl[-1, j] + np.array([0.0, y_fold[s]]), 50.0))
            for kk in range(COLUMN_ROWS_UP + 1):
                col.append((inners[j][kk], Fl[-1, j] + np.array([0.0, y_in[kk]]), hs[kk]))
            V.append(col)
        total = len(V[0])
        verts = [[bm.verts.new(tuple(p)) for p, _, _ in V[j]] for j in range(n)]
        for j in range(n):
            for i in range(total):
                verts[j][i][la['hem_d']] = float(V[j][i][2])
                verts[j][i][la['top_d']] = float(abs(V[j][i][0][2] - z_top))
        faces = []
        for j in range(n - 1):
            for i in range(total - 1):
                f = bm.faces.new((verts[j][i], verts[j + 1][i], verts[j + 1][i + 1], verts[j][i + 1]))
                for lp, (jj, ii) in zip(f.loops, ((j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1))):
                    lp[uvl].uv = tuple(V[jj][ii][1] + shift)
                faces.append(f)
        # outward: the outer layer's faces point away from the axis (the hem's inner layer faces the other way)
        bm.normal_update()
        f0 = faces[len(faces) // 3 if len(faces) > 3 else 0]
        c = f0.calc_center_median()
        if f0.normal.x * c.x + f0.normal.y * c.y < 0:
            bmesh.ops.reverse_faces(bm, faces=faces)
        face_sets.append(faces)
        allphis += [float(p) % (2 * math.pi) for p in phis]
        seam_cols[(k, 'left')] = cols_outer[0]
        seam_cols[(k, 'right')] = cols_outer[-1]
        inner_edges.append([inners[j][-1] for j in range(n)])
    # ---- 2. the waistband: outer layer on the gores' top edge, over the top, the facing back down inside
    phis_u = sorted({round(p, 9) for p in allphis})
    phis_u = [p for p in phis_u]
    ring = []
    for ph in phis_u + [phis_u[0] + 2 * math.pi]:
        col, hz = column(ph)
        P0, P1 = col[0], col[1]
        up = (P0 - P1) / max(abs(P0[2] - P1[2]), 1e-9)            # (the skirt's line, carried up past its top edge)
        up = up / np.linalg.norm(up) if up[2] > 0 else np.array([0.0, 0.0, 1.0])
        top_o = P0 + up * (wb_h - rho) / up[2]
        u = _inward(P0)
        outer = [P0 + (top_o - P0) * t for t in np.linspace(0.0, 1.0, 5)]
        arc = [top_o + u * rho * (1 - math.cos(th)) + np.array([0.0, 0.0, rho * math.sin(th)])
               for th in (math.pi / 4, math.pi / 2, 3 * math.pi / 4)]
        facing = [top_o + u * 2 * rho + (P0 - top_o) * t for t in np.linspace(0.0, 1.0, 5)]
        ring.append(outer + arc + facing)
    total = len(ring[0])
    vw = [[bm.verts.new(tuple(p)) for p in col] for col in ring]
    # (the ring's last column is its first: the waistband's two ends are one line of points)
    flat_len = [0.0]
    for j in range(1, len(ring)):
        flat_len.append(flat_len[-1] + float(np.linalg.norm(ring[j][0] - ring[j - 1][0])))
    wfaces = []
    hts = [0.0] + [float(np.linalg.norm(ring[0][i] - ring[0][i - 1])) for i in range(1, total)]
    hcum = np.cumsum(hts)
    for j in range(len(ring)):
        for i in range(total):
            vw[j][i][la['hem_d']] = 50.0
            vw[j][i][la['top_d']] = float(abs(ring[j][i][2] - z_top))
    for j in range(len(ring) - 1):
        for i in range(total - 1):
            f = bm.faces.new((vw[j][i], vw[j][i + 1], vw[j + 1][i + 1], vw[j + 1][i]))
            for lp, (jj, ii) in zip(f.loops, ((j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i))):
                lp[uvl].uv = (flat_len[jj] + 500.0, hcum[ii])      # (the strip: along its grain, cut elsewhere on the bolt)
            wfaces.append(f)
    bm.normal_update()
    f0 = wfaces[len(wfaces) // 2]
    c = f0.calc_center_median()
    if f0.normal.x * c.x + f0.normal.y * c.y < 0:
        bmesh.ops.reverse_faces(bm, faces=wfaces)
    # ---- 3. sew: the paired edges' points are drawn together (welded at the stitch line)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=2e-4)
    log('skirt sewn: %d gores and a waistband, %d points joined along the seams (%d to %d points)' % (
        len(gores), before - len(bm.verts), before, len(bm.verts)))
    # the seam attribute: the distance across the cloth to the nearest seam
    seam_phis = sorted({round(float(g['phi'][0]) % (2 * math.pi), 9) for g in gores})
    for v in bm.verts:
        ph = math.atan2(-v.co.x, -v.co.y) % (2 * math.pi)
        rad = math.hypot(v.co.x, v.co.y)
        ds = min(abs((ph - s + math.pi) % (2 * math.pi) - math.pi) for s in seam_phis)
        v[la['seam']] = ds * rad
    sew_data = dict(seam_cols=seam_cols, gores=gores, ring=ring, inner_edges=inner_edges, phis=phis_u,
                    seam_phis=seam_phis)
    return bm, sew_data


# ---------------------------------------------------------------- the stitches and the pressed allowances
def _resample(poly, step):
    """Points along a polyline every `step`, with the cumulative length at each."""
    P = np.asarray(poly, dtype=float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1], step)
    s = np.append(s, cum[-1])
    return np.stack([np.interp(s, cum, P[:, i]) for i in range(3)], axis=1), s


def stitch_mesh(lines):
    """Dashed thread along each (polyline, offset direction at each point, width direction at each point): quads of
    THREAD width, a stitch long with a gap. Lies INSET inside the cloth along the offset direction."""
    bm = bmesh.new()
    for pts, off, wid in lines:
        P, s = _resample(pts, 0.065)
        O = np.stack([np.interp(s, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1))]),
                                np.asarray(off)[:, i]) for i in range(3)], axis=1)
        W = np.stack([np.interp(s, np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1))]),
                                np.asarray(wid)[:, i]) for i in range(3)], axis=1)
        period = STITCH + GAP
        i = 0
        while i < len(P) - 1:
            phase = s[i] % period
            if phase < STITCH:
                # a stitch: run the quad strip until the stitch ends
                j = i
                while j < len(P) - 1 and (s[j] % period) < STITCH and (s[j] - s[i]) < STITCH:
                    j += 1
                if j > i:
                    strip = [(P[q] + O[q] * INSET, W[q]) for q in range(i, j + 1)]
                    vs = []
                    for p, w in strip:
                        w = w / (np.linalg.norm(w) or 1.0)
                        vs.append((bm.verts.new(tuple(p - w * THREAD * 0.5)), bm.verts.new(tuple(p + w * THREAD * 0.5))))
                    for a, b in zip(vs[:-1], vs[1:]):
                        bm.faces.new((a[0], a[1], b[1], b[0]))
                i = max(j, i + 1)
                # (skip the gap)
                while i < len(P) - 1 and (s[i] % period) >= STITCH:
                    i += 1
            else:
                i += 1
    return bm


def allowance_mesh(D, sd, r):
    """The seam allowances, cut beyond each seam's stitch line and pressed to one side inside: strips of the wool a
    seam allowance wide, lying on the inside of the next gore toward the back (both gores' allowances, one over the
    other). The back-middle seam presses to her right."""
    column = D['column']
    A = r['allowance']
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new('pattern')
    for s in sd['seam_phis']:
        direction = 1.0 if s <= math.pi + 1e-6 else -1.0
        base, _ = column(s)
        lim = np.nonzero(base[:, 2] >= base[-1, 2] + r['hem_band'])[0]
        top, bot = 0, int(lim[-1]) if len(lim) else len(base) - 1
        for layer in (0, 1):
            far = []
            for i in range(top, bot + 1):
                rad = math.hypot(base[i, 0], base[i, 1])
                ph2 = s + direction * A / rad
                col2, _ = column(ph2)
                far.append(col2[i])
            near = [base[i] for i in range(top, bot + 1)]
            inset = INSET * (1.0 + layer * 1.6)
            rows = []
            for p, q in zip(near, far):
                u = _inward(p)
                rows.append((bm.verts.new(tuple(p + u * inset)), bm.verts.new(tuple(q + _inward(q) * inset))))
            for k in range(len(rows) - 1):
                a, b = rows[k], rows[k + 1]
                f = bm.faces.new((a[0], a[1], b[1], b[0]) if direction > 0 else (a[1], a[0], b[0], b[1]))
                for lp, (cc, rr) in zip(f.loops, ((0, k), (1, k), (1, k + 1), (0, k + 1))):
                    lp[uvl].uv = (cc * A, rr * 0.3)
    return bm


def stitch_lines(D, sd, r):
    """The stitch lines as (points, offset direction, width direction) for stitch_mesh: every gore seam (up the
    stitch line, on the inside), the waist join (round the gores' top edge) and the hem's inner edge."""
    lines = []
    column = D['column']
    for s in sd['seam_phis']:
        col, _ = column(s)
        lim = np.nonzero(col[:, 2] >= col[-1, 2] + r['hem_band'])[0]
        col = col[:int(lim[-1]) + 1]
        off = np.array([_inward(p) for p in col])
        t = np.array([[math.cos(s), -math.sin(s), 0.0]] * len(col))          # (across the seam, along the angle)
        lines.append((col, off, t))
    # the waist join: the gores' top edge, all the way round
    ph = list(sd['phis'])
    ph = ph + [ph[0] + 2 * math.pi]
    top = np.array([column(p)[0][0] for p in ph])
    lines.append((top, np.array([_inward(p) for p in top]), np.array([[0.0, 0.0, 1.0]] * len(top))))
    # the hem's inner edge: where the hem allowance is sewn down
    hemline = []
    for g_edge in sd['inner_edges']:
        hemline += list(g_edge)
    hemline = np.array(hemline)
    lines.append((hemline, np.zeros((len(hemline), 3)),
                  np.array([[0.0, 0.0, 1.0]] * len(hemline))))
    return lines
