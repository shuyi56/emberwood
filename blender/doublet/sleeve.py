"""The arming doublet's sleeves, cut from the flat draft (sleeve_draft.py: the tape's measures, then a one-piece
set-in sleeve) and sewn to the doublet's own armhole.

The draft's cloth is laid on the arm point for point:

  the seam      the cap's edge lies on the armhole's edge, length for length: the shoulder point on the armhole's
                top, the underarm seam on its lowest point, the front of the cap along the front of the armhole
                and the back along the back
  below it      from the bicep line down, each level line of the draft is a ring round the arm: the arm's own
                section drawn taut and let out evenly until it is as long as the draft is wide there
  the cap       between the seam and the bicep line the cloth runs straight from one to the other, and is then
                lifted clear of the shoulder by the sleeve's room, so it rounds over the top of the arm

It is quilted as the body is, in channels down its length (the seam under the arm on a line of stitching), flat at
the armhole seam, the elbow and the cuff, which is bound."""
import math

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import fit
import sleeve_draft

RATIOS = dict(
    sleeve_draft.RATIOS,
    sleeve_channel=0.46,     # a channel's width at the bicep line, in w
    sleeve_pad=0.13,         # the padding's depth at a channel's crown, in w
    sleeve_seam=0.13,        # the half-width of the flat the elbow's seam leaves, in w
    sleeve_let_out=1.6,      # the sleeve's room comes in over this far from the armhole seam, in w
    sleeve_hang=1.3,         # the seam's hold on the trunk gives way to the arm's over this far from it, in w
)
STEP = 0.1                   # the rings' spacing along the arm, in units
NA = 96                      # directions round the arm


def _arm_table(P, T, aw, M, z_lo, z_hi):
    """The arm's skin as radii from its axis: (axes [height, (x, y)], heights, angles, radii [height, angle]). The left arm
    (x > 0); alpha 0 is out to the side."""
    w = M['w']
    torso = (aw[T] < 0.5).all(1)
    pts = [Vector(p) for p in P]
    ba = BVHTree.FromPolygons(pts, [tuple(int(i) for i in t) for t in T[~torso]])
    band = P[(aw > 0.4) & (aw < 0.6)]
    low = band[band[:, 2] <= np.percentile(band[:, 2], 3)]
    x_pit = float(np.abs(low[:, 0]).mean())
    arm = P[(aw > 0.5) & (P[:, 0] > 0) & (P[:, 2] > z_lo) & (P[:, 2] < M['z_pit_arm'])]
    cx = 0.5 * float(arm[:, 0].min() + arm[:, 0].max())
    cy = 0.5 * float(arm[:, 1].min() + arm[:, 1].max())
    # (the arm's axis follows the arm: the forearm leans out from the elbow, and a plumb axis would miss it)
    zs = np.arange(z_lo, z_hi, STEP)
    arm_all = (aw > 0.5) & (P[:, 0] > 0)
    ax = np.full((len(zs), 2), np.nan)
    for i, z in enumerate(zs):
        if z > M['z_pit_arm'] - 0.3 * w:
            continue
        m = arm_all & (np.abs(P[:, 2] - z) < 0.3)
        if m.sum() > 8:
            ax[i] = 0.5 * (P[m][:, :2].min(0) + P[m][:, :2].max(0))
    for i in range(len(zs)):
        if np.isnan(ax[i, 0]):
            ax[i] = (cx, cy)
    k = int(round(1.0 / STEP))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    pad = np.concatenate([np.repeat(ax[:1], k, 0), ax, np.repeat(ax[-1:], k, 0)])
    axs = np.stack([np.convolve(pad[:, c], ker, 'valid') for c in (0, 1)], 1)
    al = -math.pi + 2 * math.pi * np.arange(NA) / NA
    ca, sa = np.cos(al), np.sin(al)
    reach = (cx - x_pit + 0.2 * w) / np.maximum(-ca, 1e-6)
    Ra = np.full((len(zs), NA), np.nan)
    for i, z in enumerate(zs):
        o = Vector((float(axs[i, 0]), float(axs[i, 1]), float(z)))
        r = np.empty(NA)
        for j in range(NA):
            hit = ba.ray_cast(o, Vector((ca[j], sa[j], 0.0)))[0]
            r[j] = (hit - o).length if hit is not None else 1e9
        if z > M['z_pit_arm'] - 0.3 * w:
            r = np.where(ca < -0.05, np.minimum(r, reach), r)
        ok = r < 1e8
        if ok.mean() < 0.6:
            continue
        r = np.minimum(r, 1.5 * float(np.median(r[ok])))
        Ra[i] = fit._taut(r, -ca, -sa)
    have = np.nonzero(~np.isnan(Ra[:, 0]))[0]
    i0, i1 = int(have.min()), int(have.max())
    for i in range(i0):
        Ra[i] = Ra[i0]
    for i in range(i1 - 1, i0, -1):
        if np.isnan(Ra[i, 0]):
            Ra[i] = Ra[i + 1]
    i_cap = max(i0, i1 - int(round(0.45 * w / STEP)))
    Ra = Ra[:i_cap + 1]
    Ra = np.maximum(fit._blur(fit._blur(Ra, 1.2, 0, False), 1.5, 1, True), Ra - 0.5 * STEP)
    return axs[:i_cap + 1], zs[:i_cap + 1], al, Ra


def _resample(ring, n, start_back=True):
    """A closed ring of points [m, 2] as n points evenly along its length, from its first point."""
    seg = np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    at = cum[-1] * np.arange(n) / n
    closed = np.concatenate([ring, ring[:1]])
    return np.stack([np.interp(at, cum, closed[:, c]) for c in (0, 1)], 1), float(cum[-1])


def _path(pts):
    """A line of points with its length along it: (points, cumulative length from 0 to 1)."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, cum / cum[-1]


def _along(path, u):
    pts, cum = path
    return np.stack([np.interp(u, cum, pts[:, c]) for c in range(pts.shape[1])], 1)


def _weights(body, kd, groups, p):
    acc, tot = {}, 0.0
    for _, bi, dist in kd.find_n(p, 4):
        k = 1.0 / (dist + 0.05)
        for g in body.data.vertices[bi].groups:
            if g.group in groups:
                acc[g.group] = acc.get(g.group, 0.0) + k * g.weight
                tot += k * g.weight
    return {gi: v / tot for gi, v in acc.items()} if tot else {}


def build_sleeves(body, M, rig, cloth, doublet, name='Doublet sleeve', ratios=RATIOS, base=None, log=print):
    """A sleeve on each arm, sewn to the doublet's armholes. base: the doublet's ratios (thickness, border, bind).
    Returns the objects."""
    base = base or {}
    w = M['w']
    P, T, aw = fit.body_arrays(body)
    co = np.array([v.co[:] for v in doublet.data.vertices])
    hole = sleeve_draft.armhole(co, [list(p.vertices) for p in doublet.data.polygons])
    b = {n: rig.data.bones[n] for n in ('UpperArm.L', 'Forearm.L')}
    head, mid, tail = (rig.matrix_world @ v for v in (b['UpperArm.L'].head_local, b['Forearm.L'].head_local, b['Forearm.L'].tail_local))
    m = sleeve_draft.measure(P, T, aw, hole, ((head.z, mid.z, tail.z), (head.x, head.y)), None, w, ratios)
    d = sleeve_draft.draft(m, w, ratios)
    h, wf, wb = d['cap'], d['front'], d['back']
    log('doublet: sleeve from the draft: armhole %.2f, cap %.2f high, bicep line %.2f, elbow %.2f, cuff %.2f' % (
        m['armhole'], h, d['bicep'], d['elbow'], d['cuff']))
    z_bot, z_cuff, z_elbow = float(m['bottom'][2]), m['z_cuff'], m['z_elbow']
    # --- the rings, from the cuff up to the bicep line (the underarm point's height)
    axs, zs, al, Ra = _arm_table(P, T, aw, M, z_cuff, z_bot + 0.45 * w + 3.0 * STEP)
    NS = 96
    s = np.arange(NS) / NS                       # (round the sleeve from the seam: the back, the shoulder line, the front)
    lev = np.arange(z_cuff, z_bot + 1e-6, 2 * STEP)[::-1]            # (from the bicep line down)
    lev[0] = z_bot
    y_of = h + (z_bot - lev)
    width = np.where(y_of <= d['y_elbow'], np.interp(y_of, [h, d['y_elbow']], [d['bicep'], d['elbow']]),
                     np.interp(y_of, [d['y_elbow'], d['y_cuff']], [d['elbow'], d['cuff']]))
    ca, sa = np.cos(al), np.sin(al)
    tube = np.empty((len(lev), NS, 3))
    for i, z in enumerate(lev):
        k = int(np.clip(round((z - zs[0]) / STEP), 0, len(zs) - 1))
        r = Ra[k]
        off = 0.0
        for _ in range(4):                       # (let out evenly until the ring is as long as the draft is wide)
            ring = np.stack([(r + off) * ca, (r + off) * sa], 1)
            L = float(np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1).sum())
            off += (width[i] - L) / (2 * math.pi)
        ring = np.stack([(r + off) * ca, (r + off) * sa], 1)[::-1]   # (from the body's side round by the back)
        ring = np.roll(ring, 1, 0)                                   # (its first point at alpha = -pi: against the body)
        q, _ = _resample(ring, NS)
        tube[i, :, 0], tube[i, :, 1], tube[i, :, 2] = axs[k, 0] + q[:, 0], axs[k, 1] + q[:, 1], z
    room = (d['bicep'] - m['bicep']) / (2 * math.pi)
    # --- the seam: each column's place on the armhole, by its length along the cap's curve
    i_top, i_bot = int(hole[:, 2].argmax()), int(hole[:, 2].argmin())
    hole = np.roll(hole, -i_bot, 0)
    i_top = (i_top - i_bot) % len(hole)
    one, two = hole[:i_top + 1], np.concatenate([hole[i_top:], hole[:1]])[::-1]        # (both from the bottom up to the top)
    back_p, front_p = (one, two) if one[:, 1].mean() > two[:, 1].mean() else (two, one)
    back_c = sleeve_draft._half(wb, h, ratios['back_scoop'], ratios['crown'])           # (from the shoulder point out)
    front_c = sleeve_draft._half(wf, h, ratios['front_scoop'], ratios['crown'])
    x_col = -wb + s * (wb + wf)
    A = np.empty((NS, 3))
    y_cap = np.empty(NS)
    for side, curve, path, cols in ((-1.0, back_c, _path(back_p), x_col < 0), (1.0, front_c, _path(front_p), x_col >= 0)):
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
        cum /= cum[-1]
        ax_ = np.abs(x_col[cols])
        u = np.interp(ax_, curve[:, 0], cum)                # (0 at the shoulder point, 1 at the underarm)
        y_cap[cols] = np.interp(ax_, curve[:, 0], curve[:, 1])
        A[cols] = _along(path, 1.0 - u)
    # --- the cap: from the seam straight to the bicep line, then lifted clear of the shoulder
    KC = 12
    v = (np.arange(KC) / KC)[:, None, None]
    cap = A[None] * (1 - v) + tube[0][None] * v
    skin = BVHTree.FromPolygons([Vector(p) for p in P], [tuple(int(i) for i in t) for t in T])
    grid = np.concatenate([cap, tube])                    # [row, column, 3]
    fixed = np.zeros(len(grid), bool)
    fixed[0] = True
    n_free = KC + 6                                       # (the cap's rows and the first few rings settle together)

    # (the room comes in gradually from the seam, where the cloth is as close to the shoulder as the doublet's own
    # armhole edge is: let out at once, the sleeve's head stood up off the shoulder as a puff above the seam)
    def off_skin(p):
        loc, nrm, _, dist = skin.find_nearest(Vector(p))
        return loc, nrm, (dist if (Vector(p) - loc).dot(nrm) >= 0 else -dist)
    at_seam = np.array([max(off_skin(A[j])[2], 0.05) for j in range(NS)])
    run = np.concatenate([np.zeros((1, NS)), np.cumsum(np.linalg.norm(np.diff(grid[:n_free], axis=0), axis=2), 0)])
    need = at_seam[None, :] + (room - at_seam[None, :]) * fit._ss(0.0, ratios['sleeve_let_out'] * w, run)

    def clear(G):
        for i in range(1, n_free):
            for j in range(NS):
                loc, nrm, sd = off_skin(G[i, j])
                # (cloth hangs: on the top of the shoulder it rests, as close as the seam is; its room is round
                # the arm's sides and under it)
                rests = float(fit._ss(0.15, 0.75, nrm.z))
                nd = need[i, j] + (at_seam[j] - need[i, j]) * rests
                if sd < nd:
                    G[i, j] = loc + nrm * nd
    clear(grid)
    for _ in range(30):
        g = grid.copy()
        mid_ = 0.25 * (g[:-2] + g[2:] + np.roll(g[1:-1], 1, 1) + np.roll(g[1:-1], -1, 1))
        grid[1:n_free] = 0.5 * g[1:n_free] + 0.5 * mid_[:n_free - 1]
        clear(grid)
    rows = len(grid)
    # --- the quilting, on the cloth's own lines: the channels follow the columns
    n = 2 * max(3, int(round(d['bicep'] / (2 * ratios['sleeve_channel'] * w))))
    border, bind_r, th = base.get('border', 0.30) * w, base.get('bind', 0.03) * w, base.get('thickness', 0.06) * w
    du = np.roll(grid, -1, 1) - np.roll(grid, 1, 1)
    dv = np.gradient(grid, axis=0)
    nrm = np.cross(du, dv)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-9)
    c_ax = np.array([axs[-1, 0], axs[-1, 1]])
    if float(((grid[KC + 4, :, :2] - c_ax) * nrm[KC + 4, :, :2]).sum()) < 0:
        nrm = -nrm
    step = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    from_seam = np.concatenate([np.zeros((1, NS)), np.cumsum(step, 0)])
    z_g = grid[:, :, 2]
    pillow = np.sin(math.pi * ((s[None, :] * n) % 1.0)) ** 0.55
    inside = fit._ss(0.5 * border, border, from_seam) * fit._ss(0.5 * border, border, z_g - z_cuff)
    seam = fit._ss(0.0, ratios['sleeve_seam'] * w, np.abs(z_g - z_elbow))
    edge = z_g - z_cuff
    bind = bind_r * np.exp(-((edge - 0.2 * border) / (0.22 * border)) ** 2)
    lift = ratios['sleeve_pad'] * w * pillow * inside * seam + bind
    lift[0] = 0.0
    G = grid + nrm * lift[:, :, None]
    idx = np.arange(rows * NS).reshape(rows, NS)
    faces = [list(q) for q in np.stack([idx[:-1], np.roll(idx[:-1], -1, 1), np.roll(idx[1:], -1, 1), idx[1:]], 2).reshape(-1, 4)]
    kd_all, kd_arm, kd_trunk = (KDTree(len(P)) for _ in range(3))
    for i, p in enumerate(P):
        kd_all.insert(Vector(p), i)
        (kd_arm if aw[i] > 0.5 else kd_trunk).insert(Vector(p), i)
    for k in (kd_all, kd_arm, kd_trunk):
        k.balance()
    kd_hole = KDTree(len(hole))
    for i, p in enumerate(hole):
        kd_hole.insert(Vector(p), i)
    kd_hole.balance()
    out = []
    for sgn, tag in ((1.0, 'L'), (-1.0, 'R')):
        V = G.reshape(-1, 3).copy()
        F = [list(f) for f in faces]
        if sgn < 0:
            V[:, 0] *= -1.0
            F = [f[::-1] for f in F]

        def away(c, sgn=sgn):
            k = int(np.clip(round((c.z - zs[0]) / STEP), 0, len(zs) - 1))
            up = max(c.z - z_bot, 0.0) / max(h, 1e-6)
            return Vector((c.x - sgn * axs[k, 0], c.y - axs[k, 1], 1.5 * up)).normalized()

        o = fit._sheet('%s.%s' % (name, tag), [(V, F, 0.0, None, away)], M, cloth, body, kd_arm, rig, lift=0.0, thickness=th)
        # (the seam is held as the doublet's armhole is, by the trunk; from there the arm takes the sleeve over)
        groups = {g.index: o.vertex_groups[g.name] for g in body.vertex_groups if g.name in o.vertex_groups}
        hang = ratios['sleeve_hang'] * w
        for vtx in o.data.vertices:
            p = Vector((sgn * vtx.co.x, vtx.co.y, vtx.co.z))
            dist = kd_hole.find(p)[2]
            if dist >= hang:
                continue
            f = float(fit._ss(0.0, hang, dist))
            real = vtx.co.copy()
            wt, wa = _weights(body, kd_trunk, groups, real), _weights(body, kd_arm, groups, real)
            for gi, grp in groups.items():
                val = (1 - f) * wt.get(gi, 0.0) + f * wa.get(gi, 0.0)
                if val > 1e-3:
                    grp.add([vtx.index], val, 'REPLACE')
                else:
                    grp.remove([vtx.index])
        _tube_uv(o, s, from_seam, d['bicep'])
        out.append(o)
    return out


def _tube_uv(o, s, from_seam, width):
    """Each point's place on the flat cloth: across the sleeve, by its length down from the seam."""
    me = o.data
    lay = me.uv_layers.new(name='UVMap')
    rows, NS = from_seam.shape
    # (the sheet's points are the grid's, in order, where none were merged; otherwise the nearest column's)
    if len(me.vertices) != rows * NS:
        return
    for p in me.polygons:
        cols = np.array([vi % NS for vi in p.vertices])
        u = s[cols].copy()
        if np.ptp(u) > 0.5:
            u = np.where(u < 0.5, u + 1.0, u)
        for li, u_, vi in zip(p.loop_indices, u, p.vertices):
            lay.data[li].uv = (float(u_) * width, -float(from_seam[vi // NS, vi % NS]))
