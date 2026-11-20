"""The arming doublet's sleeves: a padded sleeve for each arm, drafted on the body's own arm as cut.py drafts the
body (every size a share of w, the neck's half-width).

Each is a tube round the arm's axis, the arm's skin (as the rays from the axis find it, drawn taut) let out for the
padding and the shirt beneath. Quilted as the body is: upright channels, each a pillow of the padding between two
lines of stitching; flat along the cuff, which is bound, and across the elbow, where a seam lets the arm bend. Over
the shoulder it is closed with a rounded cap, so the cloth falls over the top of the arm and the body's own shoulder
end lies under it. It is made apart from the body and meets it where the two cross: the body's armhole is open,
and the sleeve fills it."""
import math

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import fit

RATIOS = dict(
    sleeve_ease=0.30,        # the gap between arm and sleeve at the upper arm, in w
    sleeve_ease_cuff=0.20,   # and at the cuff
    sleeve_cuff=0.25,        # the cuff ends this far above the wrist, in w
    sleeve_channel=0.46,     # a channel's width, in w
    sleeve_pad=0.13,         # the padding's depth at a channel's crown, in w
    sleeve_cap=0.60,         # the cap's height over the arm's top ring, as a share of the ring's radius
    sleeve_over=0.25,        # the crown stands this far above the shoulder's top, in w
    sleeve_seam=0.13,        # the half-width of the flat the elbow's seam leaves, in w
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


def build_sleeves(body, M, rig, cloth, name='Doublet sleeve', ratios=RATIOS, base=None, log=print):
    """A sleeve on each arm. base: the doublet's ratios (thickness, border, bind). Returns the objects."""
    base = base or {}
    w = M['w']
    P, T, aw = fit.body_arrays(body)
    gi = {g.name: g.index for g in body.vertex_groups}
    fore = [v.index for v in body.data.vertices if any(g.group == gi.get('Forearm.L') and g.weight > 0.5 for g in v.groups)]
    z_elbow, z_wrist = float(P[fore, 2].max()), float(P[fore, 2].min())
    z_cuff = z_wrist + ratios['sleeve_cuff'] * w
    axs, zs, al, Ra = _arm_table(P, T, aw, M, z_cuff, M['z_side'] + 0.8 * w)
    keep = zs >= z_cuff
    zs, Ra, axs = zs[keep], Ra[keep], axs[keep]
    cx, cy = float(axs[-1, 0]), float(axs[-1, 1])
    t = np.clip((zs - z_cuff) / max(M['z_sh'] - z_cuff, 1e-6), 0.0, 1.0)
    ease = (ratios['sleeve_ease_cuff'] + (ratios['sleeve_ease'] - ratios['sleeve_ease_cuff']) * fit._ss(0.0, 0.7, t)) * w
    R = Ra + ease[:, None]
    # the cap: a half-ellipse over the top ring
    R_top = R[-1]
    # (its crown stands over the shoulder's skin wherever that rises above the top ring: toward the neck the
    # shoulder climbs, and a round cap would leave it bare between the cap and the body's own shoulder end)
    ca, sa = np.cos(al), np.sin(al)
    h_min = ratios['sleeve_cap'] * float(R_top.mean())
    top_xy = P[(P[:, 2] > float(zs[-1]) - 2.0) & (P[:, 2] < M['z_side'] + 1.0) & (P[:, 0] > 1.0)]
    h_a = np.empty(NA)
    for j in range(NA):
        c = np.array([cx + 0.55 * R_top[j] * ca[j], cy + 0.55 * R_top[j] * sa[j]])
        near = top_xy[np.hypot(top_xy[:, 0] - c[0], top_xy[:, 1] - c[1]) < 0.9]
        z_skin = float(near[:, 2].max()) if len(near) else float(zs[-1])
        h_a[j] = max(h_min, z_skin + ratios['sleeve_over'] * w - float(zs[-1]))
    h_a = fit._blur(h_a[None, :], 3.0, 1, True)[0]
    h = float(h_a.max())
    K = 6
    rings = [(float(z), R[i]) for i, z in enumerate(zs)]
    for k in range(1, K + 1):
        a = 0.5 * math.pi * k / K
        # (a boxy crown: a superellipse, so it keeps its height out toward the rim before turning down)
        rings.append((float(zs[-1]) + h_a * math.sin(a) ** 0.8, R_top * math.cos(a) ** 0.8))
    z_dome = float(zs[-1])
    ca, sa = np.cos(al), np.sin(al)
    axr = np.concatenate([axs, np.repeat(axs[-1:], K, 0)])
    G = np.array([[(axr[i, 0] + r[j] * ca[j], axr[i, 1] + r[j] * sa[j], z[j] if np.ndim(z) else z) for j in range(NA)] for i, (z, r) in enumerate(rings)])
    rows = len(rings)
    idx = np.arange(rows * NA).reshape(rows, NA)
    faces = [list(q) for q in np.stack([idx[:-1], np.roll(idx[:-1], -1, 1), np.roll(idx[1:], -1, 1), idx[1:]], 2).reshape(-1, 4)]
    circ = 2 * math.pi * float(Ra[len(Ra) // 2].mean() + ease[len(Ra) // 2])
    n = 2 * max(3, int(round(circ / (2 * ratios['sleeve_channel'] * w))))
    log('doublet: sleeve %d channels, cuff at %.1f, elbow at %.1f, cap %.1f over the arm\'s top ring' % (n, z_cuff, z_elbow, h))
    border = base.get('border', 0.30) * w
    bind_r = base.get('bind', 0.03) * w
    th = base.get('thickness', 0.06) * w
    trunk_arm = np.nonzero(aw > 0.5)[0]
    kd = KDTree(len(trunk_arm))
    for i in trunk_arm:
        kd.insert(Vector(P[i]), int(i))
    kd.balance()
    out = []
    for sgn, tag in ((1.0, 'L'), (-1.0, 'R')):
        V = G.reshape(-1, 3).copy()
        F = [list(f) for f in faces]
        if sgn < 0:
            V[:, 0] *= -1.0
            F = [f[::-1] for f in F]

        def raised(p, sgn=sgn):
            x = sgn * p[:, 0]
            ang = np.arctan2(p[:, 1] - np.interp(p[:, 2], zs, axs[:, 1]), x - np.interp(p[:, 2], zs, axs[:, 0]))
            pillow = np.sin(math.pi * ((ang * n / (2 * math.pi)) % 1.0)) ** 0.55
            above = p[:, 2] - z_dome
            capfade = 1.0 - fit._ss(0.0, h, above)
            edge = p[:, 2] - z_cuff
            seam = fit._ss(0.0, ratios['sleeve_seam'] * w, np.abs(p[:, 2] - z_elbow))
            inside = fit._ss(0.5 * border, border, edge)
            bind = bind_r * np.exp(-((edge - 0.2 * border) / (0.22 * border)) ** 2)
            return ratios['sleeve_pad'] * w * pillow * seam * inside * capfade + bind

        def away(c, sgn=sgn):
            ax_, ay_ = float(np.interp(c.z, zs, axs[:, 0])), float(np.interp(c.z, zs, axs[:, 1]))
            up = max(c.z - z_dome, 0.0) / max(h, 1e-6)
            return Vector((c.x - sgn * ax_, c.y - ay_, up * 2.0)).normalized()

        o = fit._sheet('%s.%s' % (name, tag), [(V, F, raised, None, away)], M, cloth, body, kd, rig, lift=0.0, thickness=th)
        _tube_uv(o, sgn, axs, zs, circ / (2 * math.pi))
        out.append(o)
    return out


def _tube_uv(o, sgn, axs, zs, R0):
    """Each point's place on the flat cloth, in units: round the arm at its girth, by height."""
    me = o.data
    lay = me.uv_layers.new(name='UVMap')
    co = np.array([v.co[:] for v in me.vertices])
    ang = np.arctan2(co[:, 1] - np.interp(co[:, 2], zs, axs[:, 1]), sgn * co[:, 0] - np.interp(co[:, 2], zs, axs[:, 0]))
    for p in me.polygons:
        a = ang[list(p.vertices)]
        if np.ptp(a) > math.pi:
            a = np.where(a < 0, a + 2 * math.pi, a)
        for li, a_, vi in zip(p.loop_indices, a, p.vertices):
            lay.data[li].uv = (float(a_) * R0, float(co[vi, 2]))
