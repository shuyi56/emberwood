"""The cloak's measures, taken off the built figure as a tailor takes them off a client: the neck base, the shoulder
points, the profile of the body along every line the cloth falls down from the neck, the knee and the floor.

Nothing here is a hand-placed height. Every landmark is found on the mesh, and the few choices a cutter makes (how
low the front neck sits, how deep the yoke is, how far below the knee the hem falls) are shares of H, the head's
height (chin to crown), which the whole figure is built from. A body with other shoulders, another neck or another
head gives other measures and so another pattern, with no number changed.

    M = measure.take(objs, H, rig)          # objs: the body's meshes (skin, neck, hands) and anything worn under
    M.neck(phi)                             # the cloak's neckline point at angle phi (3D)
    M.meridian(phi)                         # the line the cloth takes from the neckline down to the hem at phi

Conventions (the figure's, as habit.py and female.torso_point use them): z up, the front is -y, the figure's own
right is -x. phi is measured round the neck's axis from the centre front (0), growing toward her right, pi at the
centre back; the direction at phi is (-sin phi, -cos phi). Lengths are in model units (1 unit is about 2.6 cm)."""
import math

import numpy as np

CM = 165.0 / 63.0               # (the figure is 63 units tall for a 165 cm woman)

# ---------------------------------------------------------------- the cutter's choices, in H
# The neckline. The side neck point (SNP) and the back neck point (C7) are found on the mesh (where the neck turns
# into the shoulder and into the back). The front neck point is the pit of the neck, which the skin does not model
# as a hollow: it is set below the side neck point by a dressmaker's block's front-neck drop, a share of the head.
FRONT_DROP = 0.24               # (about 6 cm on her: the block's 7 cm, raised a little so the stand collar hugs)
NECK_EASE = 0.012               # the cloak's neck edge stands this far off the skin (cloth and a finger's ease)
# Everything the cloth falls over is let out by this much (the cloth's own thickness and a little air)
EASE = 0.012
# The yoke: the fitted piece over the shoulders, this deep along every line down from the neck. 0.62 H is about
# 15 cm: just past the shoulder point at the sides, over the top of the shoulder blades behind, above the bust.
YOKE = 0.62
# The hem: level all round, this far below the knee joint (a cloak to just under the knee, as the Warden's)
BELOW_KNEE = 0.30
# How far out the cloth swings for each unit it falls once it leaves the body, before any fullness is added:
# the drape of a cut cloak with no flare of its own (the flare comes from the pattern, see pattern.py)
HANG = math.radians(3.0)
# A sheet held out by an arm falls away from it across, no steeper than this (radius lost per unit round)
TENT = 0.45

LEAN = math.tan(math.radians(30))   # where the neck's skin leans this far out of the vertical, it has met the trunk
N_PHI = 120                     # meridians round the neck (3 degrees apart)
WEDGE = 0.45                    # half-thickness of the slice of the body each meridian is measured in


def _dir(phi):
    return np.array([-math.sin(phi), -math.cos(phi)])


class Measures:
    """What the cloak is cut from. See the module's docstring."""

    def __init__(self, H, axis, floor, knee, snp, c7, fnp, acromion, phi_s, P, log):
        self.H, self.axis, self.floor, self.knee = H, axis, floor, knee
        self.snp, self.c7, self.fnp, self.acromion, self.phi_s = snp, c7, fnp, acromion, phi_s
        self.P = P
        self.z_hem = knee - BELOW_KNEE * H
        self._mer = {}
        self.log = log
    # --- the neckline
    def neck_z(self, phi):
        """The neckline's height at phi: a smooth curve round the neck through the front neck point (phi 0), the
        side neck points (pi/2) and C7 (pi), as a tailor draws the neckline through the three."""
        return self.neck_rz(phi)[1]

    def neck_rz(self, phi):
        key = round(phi, 6)
        c = getattr(self, '_neck', {})
        self._neck = c
        if key in c:
            return c[key]
        # (a cosine series through the three: z = a0 + a1 cos phi + a2 cos 2 phi, smooth all round)
        zf, zs, zb = self.fnp[2], self.snp[2], self.c7[2]
        a1 = 0.5 * (zf - zb)
        a02 = 0.5 * (zf + zb)
        a0, a2 = 0.5 * (zs + a02), 0.5 * (a02 - zs)
        z = a0 + a1 * math.cos(phi) + a2 * math.cos(2 * phi)
        r = self.radius(phi, z)
        c[key] = (r, z)
        return r, z

    def neck(self, phi):
        r, z = self.neck_rz(phi)
        r += NECK_EASE * self.H
        d = _dir(phi)
        return np.array([self.axis[0] + r * d[0], self.axis[1] + r * d[1], z])

    def radius(self, phi, z, dz=0.18):
        """The skin's distance from the neck's axis at angle phi and height z (the outermost point of the slice)."""
        rz = self._slice(phi)
        m = np.abs(rz[:, 1] - z) < dz
        while not m.any():
            dz *= 1.6
            m = np.abs(rz[:, 1] - z) < dz
        # (the outermost point near the axis's own line: the neck's or the chest's skin, not an arm)
        rr = rz[m, 0]
        return float(rr.max())

    def _slice(self, phi):
        key = round(phi, 6)
        s = getattr(self, '_slices', {})
        self._slices = s
        if key not in s:
            d = _dir(phi)
            n = np.array([-d[1], d[0]])
            Q = self.P[:, :2] - self.axis
            r = Q @ d
            m = (np.abs(Q @ n) < WEDGE) & (r > 0)
            s[key] = np.stack([r[m], self.P[m, 2]], 1)
        return s[key]

    # --- the line the cloth takes down from the neck
    def meridian(self, phi):
        """The line the cloth falls along at angle phi, bridged across from its neighbours (see tent): (pts, s), the
        points as (r, z) in the slice and the length along it to each."""
        key = round(phi, 6)
        if key not in self._mer:
            self.tent()
            if key not in self._mer:
                self._mer[key] = self._tented(phi)
        return self._mer[key]

    def tent(self):
        """The cloth hangs as one sheet, not as separate tapes: where one line down catches on something that
        stands out (an arm, a hand) the lines beside it are held out too, the sheet falling away from it no
        steeper than TENT across. Each line's radius at every height is the most of its own and its neighbours'
        less TENT times the distance round to them."""
        if getattr(self, '_tent', None) is not None:
            return
        phis = np.linspace(0.0, 2 * math.pi, N_PHI, endpoint=False)
        zg = np.arange(self.z_hem, self.snp[2] + 1.0 * self.H, 0.1)
        R = np.full((N_PHI, len(zg)), np.nan)
        for i, p in enumerate(phis):
            P, _ = self._rope(p)
            m = (zg <= P[0, 1]) & (zg >= P[-1, 1])
            R[i, m] = np.interp(zg[m], P[::-1, 1], P[::-1, 0])
        Rt = R.copy()
        for i in range(N_PHI):
            dphi = np.abs(((phis - phis[i]) + math.pi) % (2 * math.pi) - math.pi)
            with np.errstate(invalid='ignore'):
                cand = R - TENT * R * dphi[:, None]
            ok = ~np.isnan(R[i])
            c = np.nanmax(np.where(np.isnan(cand), -1.0, cand), axis=0)
            Rt[i, ok] = np.maximum(R[i, ok], c[ok])
        self._tent = (phis, zg, Rt)

    def _tented(self, phi):
        phis, zg, Rt = self._tent
        P0, _ = self._rope(phi)
        # (between the two measured lines either side, by angle)
        f = (phi % (2 * math.pi)) / (2 * math.pi) * N_PHI
        i0 = int(math.floor(f)) % N_PHI
        i1 = (i0 + 1) % N_PHI
        w = f - math.floor(f)
        top = P0[0, 1]
        m = zg <= top
        zz = np.concatenate([[top], zg[m][::-1]])
        r0 = np.interp(zz, zg, np.nan_to_num(Rt[i0], nan=0.0))
        r1 = np.interp(zz, zg, np.nan_to_num(Rt[i1], nan=0.0))
        rr = np.maximum(np.interp(zz, P0[::-1, 1], P0[::-1, 0]), (1 - w) * r0 + w * r1)
        rr[0] = P0[0, 0]
        P = _resample(np.stack([rr, zz], 1), 0.25)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        return P, s

    def _rope(self, phi):
        """The line the cloth falls along at angle phi, from the neckline to the hem's height: drawn over the body
        like a tape laid from the neck point outward and down (the upper convex hull of the body's slice, let out
        by EASE), leaving the body where the hull falls steeper than HANG and going on straight at HANG to the hem.
        Returns (pts, s): the points as (r, z) in the slice, and the length along it to each."""
        key = round(phi, 6)
        c_ = getattr(self, '_ropes', {})
        self._ropes = c_
        if key in c_:
            return c_[key]
        n = self.neck(phi)
        d = _dir(phi)
        r0 = float((n[:2] - self.axis) @ d)
        z0 = n[2]
        rz = self._slice(phi)
        e = EASE * self.H
        # (only what lies below the neck point and out from it: not the neck above it)
        Q = rz[(rz[:, 1] < z0 - 0.05) & (rz[:, 0] > r0 - 0.5)]
        # let the body out by the ease: each point carried away from the axis and up by e (a coarse offset
        # that is exact where the surface faces outward or up, which is where the cloth lies on it)
        Q = Q + np.array([e, e * 0.6])
        pts = [(r0, z0)]
        c = np.array([r0, z0])
        for _ in range(400):
            v = Q - c
            ok = (v[:, 1] < -1e-4) & (np.hypot(v[:, 0], v[:, 1]) > 0.08)
            if not ok.any():
                break
            # the angle of each point from straight down (0) toward outward (pi/2): the tape takes the largest
            ang = np.arctan2(v[ok, 0], -v[ok, 1])
            k = int(np.argmax(ang))
            if ang[k] <= HANG:
                break
            # (of points at nearly that angle, the farthest, so the tape steps along the hull, not through it)
            near = np.nonzero(ang > ang[k] - 1e-3)[0]
            vv = v[ok][near]
            k = near[int(np.argmax(np.hypot(vv[:, 0], vv[:, 1])))]
            c = Q[ok][k].copy()
            if c[1] <= self.z_hem:
                p0 = np.array(pts[-1])
                t = (p0[1] - self.z_hem) / (p0[1] - c[1])
                pts.append((float(p0[0] + t * (c[0] - p0[0])), float(self.z_hem)))
                break
            pts.append((float(c[0]), float(c[1])))
        last = np.array(pts[-1])
        if last[1] > self.z_hem:
            pts.append((float(last[0] + math.tan(HANG) * (last[1] - self.z_hem)), float(self.z_hem)))
        P = _resample(np.array(pts), 0.25)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        c_[key] = (P, s)
        return P, s

    def point(self, phi, r, z):
        d = _dir(phi)
        return np.array([self.axis[0] + r * d[0], self.axis[1] + r * d[1], z])

    def report(self, log=None):
        log = log or self.log
        H = self.H
        neck = sum(np.linalg.norm(self.neck(a) - self.neck(b))
                   for a, b in zip(np.linspace(0, 2 * math.pi, 241)[:-1], np.linspace(0, 2 * math.pi, 241)[1:]))
        back = self.meridian(math.pi)[1][-1]
        side = self.meridian(self.phi_s)[1][-1]
        front = self.meridian(0.0)[1][-1]
        log('cloak measures (units; cm in brackets): H %.2f (%.1f)' % (H, H * CM))
        log('  neck base girth %.2f (%.1f), side neck point z %.2f, C7 z %.2f, front neck z %.2f' % (
            neck, neck * CM, self.snp[2], self.c7[2], self.fnp[2]))
        log('  shoulder points at x %.2f z %.2f (across %.1f cm), shoulder seam at %.0f deg' % (
            self.acromion[0], self.acromion[2], 2 * abs(self.acromion[0]) * CM, math.degrees(self.phi_s)))
        log('  floor %.2f, knee %.2f, hem z %.2f (%.1f cm off the floor)' % (
            self.floor, self.knee, self.z_hem, (self.z_hem - self.floor) * CM))
        log('  length neck to hem: back %.2f (%.1f), over the shoulder %.2f (%.1f), front %.2f (%.1f)' % (
            back, back * CM, side, side * CM, front, front * CM))
        return dict(H=H, neck=neck, back=back, side=side, front=front, z_hem=self.z_hem, floor=self.floor,
                    knee=self.knee, snp_z=self.snp[2], c7_z=self.c7[2], fnp_z=self.fnp[2],
                    shoulders=2 * abs(self.acromion[0]), phi_s=self.phi_s)


def _resample(P, step):
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)


def _smooth1(a, n):
    k = np.ones(n) / n
    p = np.pad(a, n // 2, mode='edge')
    return np.convolve(p, k, mode='valid')


def world_points(objs):
    """The evaluated (posed) points of the given mesh objects, in world space."""
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in objs:
        if o is None or o.type != 'MESH':
            continue
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        mw = np.array(o.matrix_world)
        out.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        e.to_mesh_clear()
    return np.concatenate(out)


def take(objs, H, rig=None, log=print):
    """Measure the figure (objs: its body meshes and whatever is worn under the cloak, posed as it stands)."""
    P = world_points(objs)
    floor = float(P[:, 2].min())
    # the knee: the shin's joint when there is a rig, else a quarter of the way up from the floor to the crown
    knee = None
    if rig is not None:
        for nm in ('Shin.L', 'Shin.R'):
            b = rig.pose.bones.get(nm)
            if b is not None:
                knee = float((rig.matrix_world @ b.head).z) if knee is None else 0.5 * (knee + float((rig.matrix_world @ b.head).z))
    top = float(P[:, 2].max())
    if knee is None:
        knee = floor + 0.27 * (top - floor)

    # the neck: its narrowest section between the shoulders and the jaw (searched from the top of the trunk up)
    def section(z, dz=0.12):
        m = np.abs(P[:, 2] - z) < dz
        return P[m]
    zs = np.arange(top - 0.1 * H, top - 1.2 * H, -0.1)
    widths = []
    for z in zs:
        S = section(z)
        S = S[np.abs(S[:, 0]) < 0.6 * H]
        widths.append(np.ptp(S[:, 0]) if len(S) > 10 else 1e9)
    widths = np.array(widths)
    zn = float(zs[int(np.argmin(widths))])
    S = section(zn)
    S = S[np.abs(S[:, 0]) < 0.6 * H]
    axis = np.array([0.0, 0.5 * (S[:, 1].min() + S[:, 1].max())])

    # the side neck point: on each side, the highest point where the neck's outline (front view) leans out 30
    # degrees from the vertical, turning into the shoulder (the same lean as C7's behind)
    def outline_x(z, side):
        m = (np.abs(P[:, 2] - z) < 0.1) & (np.abs(P[:, 1] - axis[1]) < 0.35 * H) & (side * P[:, 0] > 0)
        return float(np.abs(P[m, 0]).max()) if m.any() else 0.0
    zz = np.arange(zn, zn - 0.9 * H, -0.05)
    snps = []
    for side in (1, -1):
        xs = _smooth1(np.array([outline_x(z, side) for z in zz]), 7)
        dx = np.gradient(xs, 0.05)
        k = int(np.argmax(dx >= LEAN))
        snps.append((side * xs[k], zz[k]))
    snp = np.array([0.5 * (abs(snps[0][0]) + abs(snps[1][0])), axis[1], 0.5 * (snps[0][1] + snps[1][1])])

    # C7, the back neck point: where the back of the neck (side view, on the middle line) turns into the back's
    # slope, at 30 degrees from the vertical (the back of the neck leans; it meets the back at a softer corner)
    def back_y(z):
        m = (np.abs(P[:, 2] - z) < 0.1) & (np.abs(P[:, 0]) < 0.3)
        return float(P[m, 1].max()) if m.any() else axis[1]
    by = _smooth1(np.array([back_y(z) for z in zz]), 7)
    dby = _smooth1(np.gradient(by, 0.05), 5)
    # (the first height, going down the back of the neck, where it leans back 30 degrees or more)
    k = int(np.argmax(dby >= LEAN))
    c7 = np.array([0.0, by[k], zz[k]])
    # the front neck point: below the side neck point by the block's drop
    zf = snp[2] - FRONT_DROP * H

    def front_y(z):
        m = (np.abs(P[:, 2] - z) < 0.1) & (np.abs(P[:, 0]) < 0.3)
        return float(P[m, 1].min()) if m.any() else axis[1]
    fnp = np.array([0.0, front_y(zf), zf])

    # the shoulder points (acromion): on the shoulder's top outline (front view), where it turns down at 45 degrees
    def top_z(x):
        m = (np.abs(P[:, 0] - x) < 0.1) & (P[:, 2] < snp[2] + 0.2) & (P[:, 2] > snp[2] - 1.2 * H)
        return float(P[m, 2].max()) if m.any() else None
    acs = []
    for side in (1, -1):
        xs = np.arange(snp[0] + 0.15 * H, snp[0] + 1.0 * H, 0.05)
        tz = _smooth1(np.array([top_z(side * x) or np.nan for x in xs]), 7)
        g = np.gradient(tz, 0.05)
        k = int(np.argmax(-g >= 1.0))
        x = side * xs[k]
        m = (np.abs(P[:, 0] - x) < 0.12) & (np.abs(P[:, 2] - tz[k]) < 0.25)
        acs.append(np.array([x, float(np.median(P[m, 1])), tz[k]]))
    ac = np.array([0.5 * (abs(acs[0][0]) + abs(acs[1][0])), 0.5 * (acs[0][1] + acs[1][1]), 0.5 * (acs[0][2] + acs[1][2])])
    # the shoulder seam's angle round the neck: the line from the neck's axis through the shoulder point
    phi_s = math.atan2(ac[0], -(ac[1] - axis[1]))
    M = Measures(H, axis, floor, knee, snp, c7, fnp, ac, phi_s, P, log)
    return M
