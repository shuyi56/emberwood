"""A loose linen chemise, drafted on whatever body it is given, from the body's own measurements (the vest's
fit.measure: the same tailor's tape).

It is not fitted. Linen does not stretch or cling: a shift rests on the body only where the body holds it up (the
shoulders, the bust, the shoulder blades, the top of the upper arm) and everywhere else hangs plumb from those places,
standing off the waist and the arm, its spare width falling into creases. So the garment is drafted as cloth hangs:

  1. the whole piece  one surface, made as the outside of three hanging tubes run together: the body's (round the
                      neck's axis) and each sleeve's (round the arm's). Each tube's radius at a height is the widest
                      the body has been above it (the plumb line) plus the looseness, less the creases. The tubes are
                      joined smoothly over the shoulders and parted below the armpit by the gap between arm and side,
                      so the sleeves run into the body without a seam, as a peasant blouse cut in one.
  2. the trims        the neckline (a wide scoop lying on the collarbones), the hem at the hip and the sleeves' open
                      ends, each a cut through the finished piece (fit.trim).

Every size is a share of w, the neck's half-width, as in the vest."""
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vest'))
import fit  # noqa: E402
import solver  # noqa: E402

RATIOS = {
    'ease': 0.05,            # the gap between skin and linen where the linen rests on the body, in w
    'loose': 0.50,           # how far the body's cloth stands out from its plumb line, in w
    'flare': 0.18,           # and how much further by the hem, in w
    'neck_side': 1.60,       # the neckline: half its width, in w
    'neck_front': 1.55,      # how far forward of the neck's axis it scoops, in w
    'neck_back': 1.20,       # and how far back
    'hem': 2.6,              # the hem below the waist, in w
    'fold': 0.80,            # the depth of the body's creases at the hem, in w
    'folds': 9,              # how many round the body
    'gather': 0.075,         # the depth of the gathers at the neckline, in w
    'gathers': 38,           # how many round the neckline
    'gather_len': 1.4,       # how far down they run before they have opened out, in w
    'sleeve_full': 0.55,     # how far the sleeve stands off the arm's plumb line, in w
    'sleeve_bell': 0.35,     # and how much further by its open end, in w
    'sleeve_fold': 0.75,     # the depth of the sleeve's creases, in w
    'sleeve_folds': 7,       # how many round the sleeve
    'cuff': 1.3,             # the sleeve ends this far above the wrist, in w
    'blend': 0.85,           # the radius of the run from sleeve into body over the shoulder, in w
    'swell': 0.22,           # the slow in-and-out of the hanging cloth round the body and the sleeve, by the hem, in w
    'slit': 0.13,            # the gap left between the sleeve's cloth and the body's under the arm, in w
    'slit_cut': 0.40,        # the width that gap is cut at on the grid (then closed up to 'slit'), in w: wide enough
                             # for the grid to draw both its edges cleanly
    'underarm': 0.18,        # the radius the cut's edges are rounded to, in w
    'thick': 0.008,          # the linen's thickness, in w
}
VOXEL = 0.2                  # the grid the whole piece is drawn on, in units (the cloth's own grain for the solver:
                             # the render smooths between its points)
NB, NA = 180, 96             # directions round the body's axis and round an arm's


def _folds(rng, count, width, low=0.5, drift=0.3, wander=0.7):
    """`count` creases round a tube at uneven places: their centres, half-widths, depths, how far each drifts round
    on its way down, and how its depth comes and goes along its length."""
    gaps = rng.uniform(0.6, 1.4, count)
    c = np.cumsum(gaps) / gaps.sum() * 2 * math.pi + rng.uniform(0, 2 * math.pi)
    return (c, width * math.pi * gaps / gaps.sum(), rng.uniform(low, 1.0, count), rng.uniform(-drift, drift, count),
            rng.uniform(0.6, 1.6, count), rng.uniform(0, 2 * math.pi, count), wander)


def _creases(folds, ang, t, sharp=1.6):
    """The creases' depth (0 to 1) at angles ang [1, n] and t [rows, 1], the way down the tube (0 to 1)."""
    c, hw, h, drift, freq, phase, wander = folds
    out = np.zeros(np.broadcast(ang, t).shape)
    for ci, wi, hi, di, fi, pi_ in zip(c, hw, h, drift, freq, phase):
        d = (ang - ci - di * t + math.pi) % (2 * math.pi) - math.pi
        live = 1.0 - wander * (0.5 + 0.5 * np.sin(2 * math.pi * fi * t + pi_))
        out = np.maximum(out, hi * live * (0.5 + 0.5 * np.cos(np.clip(d / wi, -1.0, 1.0) * math.pi)) ** sharp)
    return out


def _hang(R):
    """Rows from the top down: each radius at least the one above (cloth hangs plumb from the widest place)."""
    out = R.copy()
    for i in range(len(R) - 2, -1, -1):          # (row 0 is the lowest)
        out[i] = np.maximum(out[i], out[i + 1])
    return out


class Draft:
    """The measured tubes: the body's and the arms' skin, and the cloth hung on them, as radii by height and angle."""

    def __init__(self, P, T, aw, M, arm_z, ratios=RATIOS, log=print):
        self.M, self.r = M, ratios
        w, yc = M['w'], M['yc']
        ease = ratios['ease'] * w
        z_elbow, z_wrist = arm_z
        self.z_hem = M['z_waist'] - ratios['hem'] * w
        self.z_cuff = z_wrist + ratios['cuff'] * w
        self.zs = zs = np.arange(min(self.z_hem, self.z_cuff) - 1.0, M['z_side'] + 0.8 * w, VOXEL)
        nz = len(zs)
        torso = (aw[T] < 0.5).all(1)
        pts = [Vector(p) for p in P]
        bt = BVHTree.FromPolygons(pts, [tuple(int(i) for i in t) for t in T[torso]])
        ba = BVHTree.FromPolygons(pts, [tuple(int(i) for i in t) for t in T[~torso]])
        # --- the body, round the neck's axis: the torso's skin, and at the arm's socket the armhole's own line
        phi = self.phi = 2 * math.pi * np.arange(NB) / NB
        sn, cs = np.sin(phi), np.cos(phi)
        x_scye, z_b = fit.scye(P, aw, M, fit.RATIOS)
        band = P[(aw > 0.4) & (aw < 0.6)]
        low = band[band[:, 2] <= np.percentile(band[:, 2], 3)]
        self.x_pit = x_pit = float(np.abs(low[:, 0]).mean())
        z_floor = self.z_hem - 0.5              # (below the hem the legs part: the table is held at this height)
        Rb = np.empty((nz, NB))
        for i, z in enumerate(zs):
            o = Vector((0.0, yc, max(float(z), z_floor)))
            lim_x = float(x_scye(z)) if z > z_b else 1e9
            for j in range(NB):
                hit = bt.ray_cast(o, Vector((-sn[j], -cs[j], 0.0)))[0]
                r = (hit - o).length if hit is not None else 1e9
                R_lim = lim_x / abs(sn[j]) if abs(sn[j]) > 1e-6 else 1e9
                Rb[i, j] = min(r, R_lim)
            if Rb[i].max() > 1e8:
                Rb[i] = Rb[i - 1] if i else 3.0 * w
            Rb[i] = fit._taut(Rb[i], sn, cs)
        Rb = np.maximum(fit._blur(fit._blur(Rb, 1.2, 0, False), 1.5, 1, True), Rb - 0.5 * ease)
        self.Rb_skin = Rb
        rest = Rb + ease
        # the body's cloth: plumb from the widest place above, but at the sides never from further out than the
        # armpit (the shoulder's end is the sleeve's to hang from, not the body's)
        cap = (x_pit + 0.1 * w) / np.maximum(np.abs(sn), 1e-6)
        plumb = _hang(np.minimum(rest, cap[None, :]))
        # (and wide enough all the way up to pass over the widest place below it, down to the hem: the hips. A shift
        # is a straight tube; cut narrower than the hip its hem would sit on the hip and the cloth pouch over it)
        i_hem = int(np.argmin(np.abs(zs - self.z_hem)))
        over = np.minimum(rest, cap[None, :])[i_hem:].copy()
        for i in range(1, len(over)):
            over[i] = np.maximum(over[i], over[i - 1])
        plumb[i_hem:] = np.maximum(plumb[i_hem:], np.where((zs[i_hem:] < M['z_bust'])[:, None], over, 0.0))
        # the neckline's height in each direction: where the resting cloth is as far out as the neckline's oval
        a, bf, bk = ratios['neck_side'] * w, ratios['neck_front'] * w, ratios['neck_back'] * w
        r_neck = 1.0 / np.sqrt((sn / a) ** 2 + (cs / np.where(cs > 0, bf, bk)) ** 2)
        self.z_neck = z_neck = np.array([zs[np.nonzero(rest[:, j] >= r_neck[j])[0].max()] for j in range(NB)])
        t_b = np.clip((z_neck[None, :] - zs[:, None]) / np.maximum(z_neck[None, :] - self.z_hem, 1e-6), 0.0, 1.0)
        below = np.clip(z_neck[None, :] - zs[:, None], 0.0, None)
        out = ratios['loose'] * w * fit._ss(0.0, 1.6 * w, below) + ratios['flare'] * w * t_b ** 1.5
        rng = np.random.default_rng(7)
        creases = _creases(_folds(rng, ratios['folds'], 0.55, wander=0.5), phi[None, :], t_b)
        swell = ratios['swell'] * w * t_b * np.sin(3 * phi[None, :] + 1.3 + 1.1 * t_b) * np.sin(2 * phi[None, :] + 0.4)
        gathers = _creases(_folds(rng, ratios['gathers'], 1.2, low=0.6, drift=0.1, wander=0.0), phi[None, :], t_b, sharp=1.2)
        a_f = ratios['fold'] * w * fit._ss(0.12, 0.85, t_b)
        a_g = ratios['gather'] * w * fit._ss(0.0, 0.1 * w, below) * np.exp(-below / (ratios['gather_len'] * w))
        self.Rb = np.maximum(np.maximum(plumb + out, rest) + swell - a_f * creases + a_g * gathers, rest)
        self.R0 = float(self.Rb[int(np.argmin(np.abs(zs - self.z_hem)))].mean())       # (the flat cloth's width over 2 pi)
        # --- an arm, round its own axis (the left one; the right is its mirror)
        arm = P[(aw > 0.5) & (P[:, 0] > 0) & (P[:, 2] > z_elbow) & (P[:, 2] < M['z_pit_arm'])]
        self.cx = cx = 0.5 * float(arm[:, 0].min() + arm[:, 0].max())
        self.cy = cy = 0.5 * float(arm[:, 1].min() + arm[:, 1].max())
        al = self.alpha = -math.pi + 2 * math.pi * np.arange(NA) / NA        # (0 out to the side, +-pi toward the body)
        ca, sa = np.cos(al), np.sin(al)
        reach = (cx - x_pit + 0.2 * w) / np.maximum(-ca, 1e-6)                # (toward the body, never past the armpit)
        Ra = np.full((nz, NA), np.nan)
        for i, z in enumerate(zs):
            o = Vector((cx, cy, float(z)))
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
        for i in range(i0):                       # (below the last ring found, the wrist's)
            Ra[i] = Ra[i0]
        # above the highest ring the shoulder is closed with a dome
        # (above the cap's base the shoulder is closed with a rounded cap, drawn in parts(): the table is held there)
        i_cap = max(i0, i1 - int(round(0.45 * w / VOXEL)))
        self.z_dome = float(zs[i_cap])
        self.h_dome = float(zs[i1] - zs[i_cap]) + 0.30 * w
        for i in range(i1 - 1, i0, -1):
            if np.isnan(Ra[i, 0]):
                Ra[i] = Ra[i + 1]
        for i in range(i_cap + 1, nz):
            Ra[i] = Ra[i_cap]
        Ra = np.maximum(fit._blur(fit._blur(Ra, 1.2, 0, False), 1.5, 1, True), Ra - 0.5 * ease)
        self.Ra_skin = Ra
        rest_a = Ra + ease
        t_a = np.clip((M['z_sh'] - zs) / (M['z_sh'] - self.z_cuff), 0.0, 1.0)[:, None]
        down = np.clip(M['z_sh'] - zs, 0.0, None)[:, None]
        out_a = ratios['sleeve_full'] * w * fit._ss(0.5 * w, 3.5 * w, down) + ratios['sleeve_bell'] * w * t_a ** 1.5
        a_fa = ratios['sleeve_fold'] * w * fit._ss(0.15, 0.6, t_a)
        self.Ra = []
        for seed in (3, 4):                       # (each sleeve its own creases)
            cr = _creases(_folds(np.random.default_rng(seed), ratios['sleeve_folds'], 0.55, wander=0.5), al[None, :], t_a)
            sw = ratios['swell'] * w * t_a * np.sin(2 * al[None, :] + seed + 1.5 * t_a)
            self.Ra.append(np.maximum(_hang(rest_a) + out_a + sw - a_fa * cr, rest_a))
        self.Ra0 = float(self.Ra[0][int(np.argmin(np.abs(zs - self.z_cuff)))].mean())
        # --- the gap between arm and side, below the armpit: midway between the body's side and the arm's inner face
        side = (Rb * np.abs(sn)[None, :]).max(1)
        inner = cx - Ra[:, 0]                      # (alpha = -pi is column 0)
        self.z_slit = M['z_pit_arm'] - 0.25 * w
        # (cut as a tailor cuts a side seam: one straight line, the best through that midway from the armpit down)
        mid = 0.5 * (side + inner)
        m = (zs > self.z_cuff) & (zs < self.z_slit)
        self.xc = np.polyval(np.polyfit(zs[m], mid[m], 1), zs)
        log('chemise: neckline at %.2f front, %.2f side, %.2f back; hem %.2f; sleeve to %.2f; arm axis x %.2f; the body\'s cloth '
            '%.2f off the waist in front' % (z_neck[0], z_neck[NB // 4], z_neck[NB // 2], self.z_hem, self.z_cuff, cx,
                                            float(self.Rb[int(np.argmin(np.abs(zs - M['z_waist']))), 0] - Rb[int(np.argmin(np.abs(zs - M['z_waist']))), 0])))

    def _look(self, table, z, ang, n):
        """A table [height, direction] read at heights z and angles ang (0 to 2 pi), between its entries."""
        f = np.clip((z - self.zs[0]) / VOXEL, 0.0, len(self.zs) - 1.001)
        i = np.floor(f).astype(int)
        t = f - i
        g = (ang % (2 * math.pi)) / (2 * math.pi) * n
        j = np.floor(g).astype(int) % n
        s = g - np.floor(g)
        j1 = (j + 1) % n
        return ((table[i, j] * (1 - s) + table[i, j1] * s) * (1 - t) + (table[i + 1, j] * (1 - s) + table[i + 1, j1] * s) * t)

    def parts(self, X):
        """How far outside each tube's cloth the points X [n, 3] are (negative inside): (body, left arm, right arm),
        and each tube's own angle at them."""
        yc = self.M['yc']
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        phi = np.arctan2(-x, -(y - yc))
        d_b = np.hypot(x, y - yc) - self._look(self.Rb, z, phi, NB)
        d_b = np.maximum(d_b, z - self.zs[-1])
        out = [d_b]
        angs = [phi]
        for side, tab in ((1.0, self.Ra[0]), (-1.0, self.Ra[1])):
            al = np.arctan2(y - self.cy, side * x - self.cx)
            R = self._look(tab, np.minimum(z, self.z_dome), al + math.pi, NA)
            # (above the cap's base: an ellipsoid on that ring, h_dome high)
            up = np.clip(z - self.z_dome, 0.0, None) * R / self.h_dome
            out.append(np.hypot(np.hypot(side * x - self.cx, y - self.cy), up) - R)
            angs.append(al)
        return out, angs

    def field(self, X):
        """The whole piece: negative inside the cloth's surface."""
        w = self.M['w']
        (d_b, d_l, d_r), _ = self.parts(X)
        k = self.r['blend'] * w
        d = fit._smin(fit._smin(d_b, d_l, k), d_r, k)
        # parted under the arm
        xc = np.interp(X[:, 2], self.zs, self.xc)
        # (the cut ends in a round at the underarm, and its edges are rounded: a slot, not a slash)
        half = 0.5 * self.r['slit_cut'] * w
        gap = np.hypot(np.abs(X[:, 0]) - xc, np.clip(X[:, 2] - (self.z_slit - half), 0.0, None)) - half
        return -fit._smin(-d, gap, self.r['underarm'] * w)

    def close_slit(self, V):
        """Draw the cut under the arm up from the width it was made at to the gap the cloth really has there: the
        points near the cut's line moved toward it, those further off less, none beyond three widths."""
        w = self.M['w']
        a, r = 0.5 * self.r['slit_cut'] * w + VOXEL, self.r['slit'] / self.r['slit_cut']
        b = 3.0 * a
        xc = np.interp(V[:, 2], self.zs, self.xc)
        u = np.abs(V[:, 0]) - xc
        au = np.abs(u)
        g = np.where(au <= a, au * r, np.where(au < b, a * r + (au - a) * (b - a * r) / (b - a), au))
        k = 1.0 - fit._ss(self.z_slit - a, self.z_slit + 0.6 * w, V[:, 2])       # (fading out above the underarm)
        out = V.copy()
        out[:, 0] = np.sign(V[:, 0]) * (xc + np.sign(u) * (au + k * (g - au)))
        return out

    def sleeve_share(self, X):
        """How much of a point is the sleeve's cloth (1) and how much the body's (0)."""
        (d_b, d_l, d_r), _ = self.parts(X)
        xc = np.interp(X[:, 2], self.zs, self.xc)
        d_a = np.minimum(d_l, d_r)
        return np.where(X[:, 2] < self.z_slit, (np.abs(X[:, 0]) > xc).astype(float), fit._ss(-0.3, 0.3, d_b - d_a))

    def cloth_uv(self, X, sleeve):
        """A point's place on the flat cloth, in units: the body's (round its axis, by height) or a sleeve's."""
        (_, _, _), (phi, al_l, al_r) = self.parts(X)
        if sleeve:
            ang = np.where(X[:, 0] > 0, al_l, al_r)
            return ang, self.Ra0
        return phi, self.R0


def surface_nets(D, origin, h):
    """The surface D = 0 of a grid of values, as points and four-sided faces: one point in every grid cell the
    surface passes through (the mean of where it crosses the cell's edges), one face across every grid edge it
    crosses. Faces are wound so their outside is the positive side."""
    shape = np.array(D.shape)
    cells = shape - 1
    acc = np.zeros((int(cells.prod()), 3))
    cnt = np.zeros(int(cells.prod()))
    S = D < 0
    quads = []

    def flat(c):
        return (c[0] * cells[1] + c[1]) * cells[2] + c[2]
    for ax in range(3):
        o1, o2 = [a for a in range(3) if a != ax]
        a, s = np.moveaxis(D, ax, 0), np.moveaxis(S, ax, 0)
        cross = s[:-1] != s[1:]
        i, j, k = np.nonzero(cross)
        t = a[:-1][i, j, k] / (a[:-1][i, j, k] - a[1:][i, j, k])
        p = np.zeros((len(i), 3))
        p[:, ax], p[:, o1], p[:, o2] = i + t, j, k
        ids = []
        for dj, dk in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            cj, ck = j + dj, k + dk
            ok = (cj >= 0) & (cj < cells[o1]) & (ck >= 0) & (ck < cells[o2])
            c = [None, None, None]
            c[ax], c[o1], c[o2] = i, np.clip(cj, 0, cells[o1] - 1), np.clip(ck, 0, cells[o2] - 1)
            f = flat(c)
            np.add.at(acc, f[ok], p[ok])
            np.add.at(cnt, f[ok], 1.0)
            ids.append(np.where(ok, f, -1))
        q = np.stack(ids, 1)
        ok = (q >= 0).all(1)
        q, inside_first = q[ok], s[:-1][i, j, k][ok]
        # (the four cells go round the edge; which way round is outward depends on the axis and on which end is inside)
        flip = inside_first if ax != 1 else ~inside_first
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    used = np.nonzero(cnt > 0)[0]
    remap = np.full(len(cnt), -1)
    remap[used] = np.arange(len(used))
    V = acc[used] / cnt[used, None] * h + np.asarray(origin)[None, :]
    return V, remap[np.concatenate(quads)]


def _relax(V, faces, rounds, keep=None):
    """Smooth the points toward their neighbours' mean (the grid's facets taken out)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 3]], faces[:, [3, 0]]])
    for _ in range(rounds):
        acc = np.zeros_like(V)
        cnt = np.zeros(len(V))
        for a, b in ((0, 1), (1, 0)):
            np.add.at(acc, e[:, a], V[e[:, b]])
            np.add.at(cnt, e[:, a], 1.0)
        V = V + 0.5 * (acc / np.maximum(cnt, 1.0)[:, None] - V)
    return V


# The pattern for the solver: the same loose piece cut plain, with no folds drawn into it (the solver makes them) and
# standing a little further off the skin to start from, so it begins clear of the body and falls onto it.
PLAIN = {'fold': 0.0, 'sleeve_fold': 0.0, 'swell': 0.0, 'gather': 0.0, 'ease': 0.14}


def build_chemise(body, M, rig=None, material=None, name='Chemise', ratios=RATIOS, log=print, settle=True,
                  colliders=()):
    """Draft the chemise on the body (M: fit.measure's measurements) and hand it to the body's bones. settle: let
    the cloth solver hang it on the posed body (and the colliders with it); without it the folds are drawn."""
    if settle:
        ratios = dict(ratios, **PLAIN)
    P, T, aw = fit.body_arrays(body)
    w = M['w']
    gi = {g.name: g.index for g in body.vertex_groups}
    fore = [v.index for v in body.data.vertices if any(g.group == gi.get('Forearm.L') and g.weight > 0.5 for g in v.groups)]
    if not fore:
        raise RuntimeError('chemise: the body has no forearm to end the sleeve at')
    D = Draft(P, T, aw, M, (float(P[fore, 2].max()), float(P[fore, 2].min())), ratios, log)
    # 1. the whole piece
    reach = D.cx + float(max(t.max() for t in D.Ra)) + 0.4
    depth = float(D.Rb.max()) + 0.4
    lo = np.array([-reach, M['yc'] - depth, D.zs[0] + 0.2])
    hi = np.array([reach, M['yc'] + depth, D.zs[-1] - 0.1])
    n = np.ceil((hi - lo) / VOXEL).astype(int) + 1
    gx, gy, gz = (lo[a] + VOXEL * np.arange(n[a]) for a in range(3))
    X = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), 3).reshape(-1, 3)
    F = np.concatenate([D.field(X[a:a + 400000]) for a in range(0, len(X), 400000)]).reshape(n)
    V, faces = surface_nets(F, lo, VOXEL)
    V = D.close_slit(_relax(V, faces, 5))
    log('chemise: the whole piece, %d points on a %d x %d x %d grid' % (len(V), *n))
    # 2. the trims
    yc = M['yc']
    a, bf, bk = ratios['neck_side'] * w, ratios['neck_front'] * w, ratios['neck_back'] * w

    def neckline(p):
        b = np.where(p[:, 1] < yc, bf, bk)
        oval = np.sqrt((p[:, 0] / a) ** 2 + ((p[:, 1] - yc) / b) ** 2) - 1.0
        return np.maximum(oval * a, M['z_bust'] - p[:, 2])        # (only the top is cut: below the bust all is kept)

    def hems(p):
        xc = np.interp(p[:, 2], D.zs, D.xc)
        return p[:, 2] - np.where(np.abs(p[:, 0]) > xc, D.z_cuff, D.z_hem)
    faces = [list(f) for f in faces]
    for nm, f in (('neckline', neckline), ('hems', hems)):
        V, faces = fit.trim(V, faces, f)
        log('chemise: cut at the %s, %d faces left' % (nm, len(faces)))
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    V, faces = V[used], [[int(remap[i]) for i in f] for f in faces]
    # off the skin: the body's cloth off the torso, the sleeves' off the arms
    share = D.sleeve_share(V)
    pts = [Vector(p) for p in P]
    torso = (aw[T] < 0.5).all(1)
    trees = [BVHTree.FromPolygons(pts, [tuple(int(i) for i in t) for t in T[m]]) for m in (torso, ~torso)]
    gap, moved = 0.5 * ratios['ease'] * w, 0
    for i in range(len(V)):
        p = Vector(V[i])
        for tree, on in ((trees[0], share[i] < 0.5), (trees[1], share[i] >= 0.5)):
            if not on:
                continue
            loc, nrm, _, dist = tree.find_nearest(p)
            if loc is not None and (dist if (p - loc).dot(nrm) >= 0 else -dist) < gap:
                V[i] = loc + nrm * gap
                moved += 1
    log('chemise: %d of %d points eased off the skin' % (moved, len(V)))
    # the mesh, its flat-cloth coordinates, its bones
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    lay = bm.loops.layers.uv.new('UVMap')
    vs = [bm.verts.new(p) for p in V]
    ang_b, R_b = D.cloth_uv(V, False)
    ang_s, R_s = D.cloth_uv(V, True)
    for f in faces:
        try:
            face = bm.faces.new([vs[i] for i in f])
        except ValueError:
            continue
        sl = np.mean(share[f]) >= 0.5
        an = np.array((ang_s if sl else ang_b)[f])
        if np.ptp(an) > math.pi:                    # (a face across the cloth's seam: carried round)
            an = np.where(an < 0, an + 2 * math.pi, an)
        for lp, i, a_ in zip(face.loops, f, an):
            lp[lay].uv = (float(a_) * (R_s if sl else R_b), float(V[i][2]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', [c for v in me.vertices for c in v.co])
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    groups = {g.index: o.vertex_groups.new(name=g.name) for g in body.vertex_groups
              if g.name.split('.')[0] in ('Pelvis', 'Chest', 'Neck', 'UpperArm', 'Forearm', 'Hand')}
    bverts = body.data.vertices
    kds = []
    for m in (aw < 0.5, aw >= 0.5):                 # (the body's cloth follows the trunk's skin, the sleeves' the arms')
        ids = [int(i) for i in np.nonzero(m)[0]]
        kd = KDTree(len(ids))
        for k_, i in enumerate(ids):
            kd.insert(Vector(P[i]), k_)
        kd.balance()
        kds.append((kd, ids))
    for v in me.vertices:
        s = float(share[v.index])
        acc, tot = {}, 0.0
        for (kd, ids), part, part_is_body in zip(kds, (1.0 - s, s), (True, False)):
            if part <= 1e-3:
                continue
            got, sub = {}, 0.0
            # (the sleeve's bones by height along the arm's axis, averaged over a span of it: a loose sleeve bends
            # as one soft tube at the elbow, where the skin's own bones change over within a finger's width)
            ats = [v.co] if part_is_body else [Vector((math.copysign(D.cx, v.co.x), D.cy, v.co.z + dz_)) for dz_ in (-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4)]
            for at in ats:
                for _, bi, dist in kd.find_n(at, 4):
                    k_ = 1.0 / (dist + 0.05)
                    for g in bverts[ids[bi]].groups:
                        if g.group in groups:
                            got[g.group] = got.get(g.group, 0.0) + k_ * g.weight
                            sub += k_ * g.weight
            for gi_, wt in got.items():
                acc[gi_] = acc.get(gi_, 0.0) + part * wt / max(sub, 1e-9)
                tot += part * wt / max(sub, 1e-9)
        for gi_, wt in acc.items():
            if wt / tot > 1e-3:
                groups[gi_].add([v.index], wt / tot, 'REPLACE')
    if rig is not None:
        m = o.modifiers.new('rig', 'ARMATURE')
        m.object = rig
        m.use_deform_preserve_volume = True
        if settle:
            solver.settle(o, rig, [body, *colliders], log=log)
    m = o.modifiers.new('smooth', 'SUBSURF')
    m.levels = m.render_levels = 1
    m = o.modifiers.new('thickness', 'SOLIDIFY')
    m.thickness = ratios['thick'] * w
    m.offset = 1.0
    m.use_rim = True
    return o, D
