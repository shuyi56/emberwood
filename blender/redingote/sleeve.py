"""The redingote's sleeves, ca. 1885-1895: navy wool, a gathered head standing up off the shoulder over a close
two-piece tailored sleeve, ending in a narrow cuff at the wrist closed with small buttons.

Drafted on whatever body it is given, from the body's own measurements (vest/fit.measure, w the neck's half-width)
and sewn into the armhole the bodice is cut along (armhole.py: the same seam surface for both, so they meet at one
line however the body changes). Nothing is placed by hand. A sleeve is made as the chemise's is:

  1. the whole piece  one surface, the outside of a tube round the arm's own axis: each ring the arm's skin pulled
                      taut, let out by the ease. Over the shoulder the tube rises past the seam, and on its upper
                      side it is let out by the puff (a gathered head stands up off the arm on its interlining).
  2. the trims        the armhole (the seam surface: what the bodice keeps, the sleeve loses, and the other way round)
                      and the hem at the wrist, each a cut through the finished piece (fit.trim).
  3. the cloth        the solver lets the head out by its fullness, pinned along the seam, so the extra cloth gathers
                      into the armhole and stands on its support; the tailored sleeve below is held close.
  4. the tailoring    the two seams of a two-piece sleeve (the forearm seam in front, toward the thumb, and the
                      hindarm seam behind, over the elbow), the ease let in behind the elbow, and the cuff: a band
                      standing a little proud, with a short vent and its buttons on the hindarm seam.

build_sleeves() returns the two sleeve objects (and their buttons), skinned to the body's own bones. The sleeves'
bones are blended over a span at the elbow, so the tube bends as one round piece in the walk and the run instead of
creasing where the skin's own weights change over."""
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('vest', 'chemise'):
    sys.path.insert(0, os.path.join(HERE, '..', _d))
sys.path.insert(0, HERE)
import fit  # noqa: E402
import solver  # noqa: E402
import drape  # noqa: E402
import armhole  # noqa: E402

RATIOS = {
    'ease': 0.10,            # the gap between arm and sleeve below the head, in w (a close tailored sleeve: 5 mm)
    'ease_head': 0.06,       # and under the head, where it lies on the shoulder before the puff, in w
    'cap_base': 0.80,        # the head's gathers end this far below the underarm, in w (on the deltoid's lower edge)
    'puff': 0.55,            # how far the head stands off the shoulder at its crown, in w (ca. 1889-1891: a modest
                             # puff, about 2.5 cm, not yet the leg-of-mutton of 1895)
    'puff_in': 0.25,         # the head rises to its full height this far from the seam, in w (the gathers stand up
                             # off the seam at once: the 'kick-up' of ca. 1890)
    'puff_hold': 0.75,       # stays there to this far from the seam, in w
    'puff_out': 2.30,        # and has run down into the tailored sleeve by this far, in w (about 6 cm: the head
                             # covers the deltoid's top, the reference's upper quarter of the upper arm)
    'puff_round': 0.55,      # how far round the arm the puff runs from its crown: cos of the angle it fades out by
                             # (0: to the front and back of the arm, 1: only on top). The underarm stays plain.
    'full': 0.55,            # how much more cloth the head has than the seam it is gathered into, each way
    'gather': 0.12,          # the drawn gathers' depth at the seam, in w (deeper than the bench's 3 mm: the grid and the
                             # smoothing take about half of it)
    'gather_step': 0.90,     # and their spacing along the seam, in w (the bench's puff: 4.6 cm)
    'gather_len': 1.10,      # how far down from the seam they run before they open out, in w
    'band': 0.30,            # the width of the band along the seam that is held to the seam's length, in w
    'elbow_ease': 0.26,      # the ease let into the hindarm seam behind the elbow, in w (room for the elbow's point
                             # when the arm bends: a run bends it to 90 degrees)
    'elbow_span': 1.10,      # over this much of the arm's length, in w
    'hem': 0.12,             # the sleeve ends this far below the wrist joint, in w (over the wrist bone)
    'cuff': 0.85,            # the cuff's height, in w (4.5 cm)
    'cuff_ease': 0.18,       # the cuff's ease round the wrist, in w (enough to pass the hand once unbuttoned)
    'cuff_proud': 0.045,     # how far the cuff stands out from the sleeve, in w
    'seam_fore': -2.10,      # the forearm seam's angle round the arm (0 out to the side, -pi/2 straight forward)
    'seam_hind': 1.25,       # the hindarm seam's angle (+pi/2 straight back)
    'seam_w': 0.10,          # a seam's half-width, in w (the pressed-open allowances either side of the stitching)
    'seam_depth': 0.030,     # how far the stitching line sinks, in w
    'vent': 0.70,            # the vent at the cuff, up the hindarm seam from the hem, in w
    'buttons': 3,
    'button': 0.10,          # a button's radius, in w (1 cm across)
    'button_step': 0.30,     # the buttons' spacing up the vent, in w
    'thick': 0.022,          # the wool's thickness, in w (a riding habit's broadcloth: about 1.1 mm)
}
VOXEL = 0.2                  # the grid the piece is drawn on (the cloth's grain to the solver: drape.VOXEL)
NA = 96                      # directions round the arm

# Wool broadcloth for a riding habit, at the figure's scale (63 units for 1.65 m), found on bench.py: a heavy, firm,
# fulled wool, bending length 3.2 cm on the strip (bending 10: 0.8 cm, 40: 2.0, 150: 2.7, 300: 3.2, 500: 3.2), so it
# falls in broad rounded folds (the curtain's 9 cm); the gathered head's cloth twice as stiff for its interlining. The
# head's gathers on the bench's puff: fullness 0.40 stood them 4.6 cm apart and 3 mm deep, 0.55 the same but deeper
# folds below; compression 8 (it gives at once along its threads) let them stand where 40 spread them 18 cm apart.
WOOL = dict(solver.LINEN,
            mass=0.9,
            tension=150.0,
            compression=8.0,
            shear=35.0,
            bending=300.0,
            friction=8.0,
            gap=0.06,
            frames=110,
            let_out=45)
KEEP_OFF = 0.07               # the least gap the cloth keeps off the skin in any pose, in w
SPAN = 0.8                   # the span the tube's bones are averaged over, either side, in w
HEAD_FRAMES = 60             # the frames the head is let settle over (it is cut close to its shape: it has little to do)
HEAD_STIFF = 2.0             # the head's bending stiffness against the sleeve's (wool on a net interlining)


def _ss(a, b, x):
    return fit._ss(a, b, x)


class Draft:
    """One sleeve (the left; the right is drafted on its mirror): the arm's axis, the cloth's radius round it by
    height and direction, the seam surface and the cuts."""

    def __init__(self, P, T, aw, M, loops, z_elbow, z_wrist, ratios=RATIOS, log=print):
        self.M, self.r = M, ratios
        w = M['w']
        L = self.L = loops['L']
        self.z_under = L['z_under']
        self.z_cap = L['z_under'] - ratios['cap_base'] * w
        self.z_elbow, self.z_wrist = z_elbow, z_wrist
        self.z_hem = z_wrist - ratios['hem'] * w
        self.z_cuff = self.z_hem + ratios['cuff'] * w
        self.seam_cut = armhole.sleeve_cut(M, loops)
        left = P[:, 0] > 0
        # the arm's axis: the middle of each level section of the arm below the armpit, a smooth line through them,
        # carried on up through the shoulder as the upper arm's own line
        arm = P[left & (aw > 0.9) & (P[:, 2] > z_wrist - 0.3) & (P[:, 2] < self.z_under)]
        zb = np.arange(z_wrist, self.z_under - 0.2, 0.25)
        mid = []
        for z in zb:
            s = arm[np.abs(arm[:, 2] - z) < 0.15]
            mid.append((0.5 * (s[:, 0].min() + s[:, 0].max()), 0.5 * (s[:, 1].min() + s[:, 1].max())) if len(s) > 8 else (np.nan, np.nan))
        mid = np.array(mid)
        ok = ~np.isnan(mid[:, 0])
        self.ax = np.polyfit(zb[ok], mid[ok, 0], 3), np.polyfit(zb[ok], mid[ok, 1], 3)
        up = ok & (zb > z_elbow + 0.5 * (self.z_under - z_elbow))
        self.ax_up = np.polyfit(zb[up], mid[up, 0], 1), np.polyfit(zb[up], mid[up, 1], 1)
        self.z_ax = float(zb[up].max())
        # the cloth's rings, round the axis
        zs = self.zs = np.arange(self.z_hem - 0.6, L['loop'][:, 2].max() + (ratios['puff'] + 1.2) * w, VOXEL)
        al = self.alpha = -math.pi + 2 * math.pi * np.arange(NA) / NA        # (0 out to the side, +-pi toward the body)
        ca, sa = np.cos(al), np.sin(al)
        pts = [Vector(p) for p in P]
        tri_arm = (aw[T] > 0.5).all(1) & (P[T][:, :, 0] > 0).all(1)
        tri_top = (aw[T] > 0.2).any(1) & (P[T][:, :, 0] > 0).all(1)
        b_arm = BVHTree.FromPolygons(pts, [tuple(int(i) for i in t) for t in T[tri_arm]])
        b_top = BVHTree.FromPolygons(pts, [tuple(int(i) for i in t) for t in T[tri_top]])
        self.b_all = BVHTree.FromPolygons(pts, [tuple(int(i) for i in t) for t in T[(P[T][:, :, 0] > -0.5).all(1)]])
        loop = L['loop']
        Rs = np.full((len(zs), NA), np.nan)
        for i, z in enumerate(zs):
            cx, cy = self.axis(z)
            o = Vector((cx, cy, float(z)))
            tree = b_arm if z <= self.z_under else b_top
            # (toward the body, never further than just past the seam: the cut there takes the rest)
            x_in = float(self.seam_x(cy, z)) - 0.6
            r = np.empty(NA)
            for j in range(NA):
                hit = tree.ray_cast(o, Vector((ca[j], sa[j], 0.0)))[0]
                r[j] = (hit - o).length if hit is not None else 1e9
                if ca[j] < -0.05 and z > self.z_cap:
                    r[j] = min(r[j], (cx - x_in) / -ca[j])
            hit_ok = r < 1e8
            if hit_ok.mean() < 0.6:
                continue
            r = np.minimum(r, 1.6 * float(np.median(r[hit_ok])))
            r[~hit_ok] = float(np.median(r[hit_ok]))
            Rs[i] = fit._taut(r, -ca, -sa)
        have = np.nonzero(~np.isnan(Rs[:, 0]))[0]
        i0, i1 = int(have.min()), int(have.max())
        Rs[:i0] = Rs[i0]
        for i in range(i0 + 1, i1):
            if np.isnan(Rs[i, 0]):
                Rs[i] = Rs[i - 1]
        # above the highest ring found the shoulder's top: closed with a dome, as the chemise's sleeve is
        self.z_dome = float(zs[i1])
        Rs[i1:] = Rs[i1]
        Rs = np.maximum(fit._blur(fit._blur(Rs, 1.5, 0, False), 1.5, 1, True), Rs - 0.3 * ratios['ease'] * w)
        self.R_skin = Rs
        # the ease: close on the arm, closer still where the head lies on the shoulder before it puffs
        e = ratios['ease'] * w + (ratios['ease_head'] - ratios['ease']) * w * _ss(self.z_cap, self.z_under, zs)
        R = Rs + e[:, None]
        # the elbow's ease, let into the hindarm seam behind the joint
        span = ratios['elbow_span'] * w
        back = np.clip(np.cos(al - 0.5 * math.pi), 0.0, None) ** 2
        R += ratios['elbow_ease'] * w * np.exp(-0.5 * ((zs - z_elbow) / (0.5 * span)) ** 2)[:, None] * back[None, :]
        self.R = R
        # the head (see field): an offset of the shoulder's own skin, standing off it by the puff on its upper side
        self.round_k = ratios['puff_round']
        self.loop_fine = armhole._resample(loop, 16 * len(loop))
        kd = KDTree(len(self.loop_fine))
        for i, p in enumerate(self.loop_fine):
            kd.insert(Vector(p), i)
        kd.balance()
        self.kd_loop = kd
        seg = np.linalg.norm(np.roll(self.loop_fine, -1, 0) - self.loop_fine, axis=1)
        self.loop_s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])          # (each fine point's way along the seam)
        self.h_dome = float(loop[:, 2].max() - self.z_dome) + ratios['ease_head'] * w + 0.2
        self.R0 = float(self.R[int(np.argmin(np.abs(zs - z_elbow)))].mean())
        log('sleeve: axis x %.2f at the elbow, %.2f at the wrist; head from %.2f (underarm %.2f, seam top %.2f); hem '
            '%.2f, cuff to %.2f; skin round the elbow %.2f, cloth %.2f' % (
                self.axis(z_elbow)[0], self.axis(z_wrist)[0], self.z_cap, self.z_under, loop[:, 2].max(), self.z_hem,
                self.z_cuff, float(Rs[int(np.argmin(np.abs(zs - z_elbow)))].mean()), self.R0))

    def axis(self, z):
        z = np.asarray(z, dtype=float)
        lo = (np.polyval(self.ax[0], z), np.polyval(self.ax[1], z))
        hi = (np.polyval(self.ax_up[0], z), np.polyval(self.ax_up[1], z))
        t = _ss(self.z_ax - 0.5, self.z_ax + 0.5, z)
        return lo[0] * (1 - t) + hi[0] * t, lo[1] * (1 - t) + hi[1] * t

    def seam_x(self, y, z):
        """The seam surface's x at (y, z) on the left side."""
        L = self.L
        t = np.clip((np.asarray(z) - L['z_under']) / (self.M['z_sh'] - L['z_under']), 0.0, 1.0)
        return L['x_under'] + (L['x_top'] - L['x_under']) * t ** armhole.RATIOS['curve'] + armhole.RATIOS['tilt'] * (np.asarray(y) - L['y_ref'])

    def _look(self, table, z, ang):
        f = np.clip((z - self.zs[0]) / VOXEL, 0.0, len(self.zs) - 1.001)
        i = np.floor(f).astype(int)
        t = f - i
        g = ((ang + math.pi) % (2 * math.pi)) / (2 * math.pi) * NA
        j = np.floor(g).astype(int) % NA
        s = g - np.floor(g)
        j1 = (j + 1) % NA
        return (table[i, j] * (1 - s) + table[i, j1] * s) * (1 - t) + (table[i + 1, j] * (1 - s) + table[i + 1, j1] * s) * t

    def polar(self, X):
        """Each point's angle round the arm and its distance from the axis (left side: x as given)."""
        cx, cy = self.axis(np.minimum(X[:, 2], self.zs[-1]))
        return np.arctan2(X[:, 1] - cy, X[:, 0] - cx), np.hypot(X[:, 0] - cx, X[:, 1] - cy)

    def top(self, al, z):
        """How much of the head a direction round the arm is (1 at the crown, 0 under the arm), by height."""
        k = self.round_k
        return np.clip((np.cos(al) + k) / (1.0 + k), 0.0, 1.0) ** 0.8 * _ss(self.z_cap - 0.4 * self.M['w'], self.z_cap + 0.8 * self.M['w'], z)

    def puff_at(self, u):
        """The head's height off the skin by the distance u from the seam: up at once off the gathers, held over the
        shoulder's top, and run down into the sleeve."""
        r, w = self.r, self.M['w']
        return r['puff'] * w * _ss(0.0, r['puff_in'] * w, u) * (1.0 - _ss(r['puff_hold'] * w, r['puff_out'] * w, u))

    def field(self, X, puff=1.0, gathers=True):
        """The whole piece: negative inside the cloth's surface (left side). Below and under the arm it is the
        tailored tube; over the shoulder it is the shoulder's own skin let out by the head's ease and, by how far
        from the seam a point is, the puff (puff: a share of it, for the head's support). The two are blended by
        how much of the head a place is, so the head runs into the sleeve without a step."""
        w = self.M['w']
        al, rho = self.polar(X)
        R = self._look(self.R, np.minimum(X[:, 2], self.z_dome), al)
        up = np.clip(X[:, 2] - self.z_dome, 0.0, None) * R / self.h_dome
        d = np.hypot(rho, up) - R
        hw = self.top(al, X[:, 2])
        on = np.nonzero((hw > 1e-3) & (d < 3.0 * w) & (d > -2.0 * w))[0]
        if len(on):
            ds, u, sl = np.empty(len(on)), np.empty(len(on)), np.empty(len(on))
            for n_, i in enumerate(on):
                p = Vector(X[i])
                loc, nrm, _, dist = self.b_all.find_nearest(p)
                ds[n_] = (dist if (p - loc).dot(nrm) >= 0 else -dist) if loc is not None else d[i]
                        # (how far from the seam along the skin: from the skin's point under it, so that straight over the
                # seam the head is still at the seam, and the puff rises as the cloth runs away from it; over the
                # bodice's side of the seam there is no puff, or the head would stand there as a wall to be cut)
                q = loc if loc is not None else p
                inside = q.x < float(self.seam_x(q.y, q.z))        # (skin on the bodice's side of the seam: none)
                if inside:
                    u[n_], sl[n_] = 0.0, 0.0
                else:
                    _, j, u[n_] = self.kd_loop.find(q)
                    sl[n_] = self.loop_s[j]
            dh = ds - self.r['ease_head'] * w - puff * self.puff_at(u) * hw[on]
            if gathers:
                # drawn: the head's fullness standing in soft gathers off the seam, a little uneven, running down
                # from it and opening out (the solver's own gathers replace these when it is run)
                r = self.r
                ph = 2 * math.pi * sl / (r['gather_step'] * w)
                wave = 0.5 + 0.5 * np.cos(ph + 0.6 * np.sin(0.37 * ph))
                fade = _ss(0.0, 0.15 * w, u) * np.exp(-u / (r['gather_len'] * w))
                dh += r['gather'] * w * wave * fade * hw[on] * (self.puff_at(u + 0.5 * w) > 0)
            d[on] = d[on] * (1.0 - hw[on]) + dh * hw[on]
        return d

    def cuts(self, p):
        """The trims: positive on the cloth that stays (outside the seam, above the hem)."""
        return np.minimum(self.seam_cut(p), p[:, 2] - self.z_hem)


def _mesh_from_field(D, log):
    reach = float(D.R.max()) + 0.6
    cx, cy = D.axis(D.zs)
    lo = np.array([min(cx.min(), D.seam_x(0.0, D.z_under)) - reach, cy.min() - reach, D.zs[0] + 0.2])
    hi = np.array([cx.max() + reach, cy.max() + reach, D.zs[-1] - 0.1])
    n = np.ceil((hi - lo) / VOXEL).astype(int) + 1
    gx, gy, gz = (lo[a] + VOXEL * np.arange(n[a]) for a in range(3))
    X = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), 3).reshape(-1, 3)
    F = np.concatenate([D.field(X[a:a + 400000]) for a in range(0, len(X), 400000)]).reshape(n)
    V, faces = drape.surface_nets(F, lo, VOXEL)
    V = drape._relax(V, faces, 4)
    log('sleeve: the whole piece, %d points on a %d x %d x %d grid' % (len(V), *n))
    faces = [list(f) for f in faces]
    V, faces = fit.trim(V, faces, D.cuts)
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    V, faces = V[used], [[int(remap[i]) for i in f] for f in faces]
    # (only the sleeve's own piece: the field's tube also runs past the seam a little inside the body; the cut leaves
    # bits of it there, which are not joined to the sleeve)
    V, faces = _largest(V, faces)
    log('sleeve: cut at the armhole and the hem, %d points, %d faces' % (len(V), len(faces)))
    return V, faces


def _sew(D, V, faces, w, ratios, log):
    """Set the head's cut edge exactly on the armhole's seam, let out by the head's ease (the bodice's edge lies on
    the same line by its own), and carry the cloth near it along by as much, fading out over the band: the cut is
    made on the grid, so without this the edge wanders off the seam by up to a grid step and more where the puff
    leans over it."""
    loop = D.L['loop']
    # (the loop drawn finely, each point let out along the skin's normal there)
    fine = armhole._resample(loop, 16 * len(loop))
    out = np.empty_like(fine)
    for i, p in enumerate(fine):
        loc, nrm, _, _ = D.b_all.find_nearest(Vector(p))
        out[i] = p + ratios['ease_head'] * w * (np.array(nrm) if loc is not None else 0.0)
    bm = bmesh.new()
    for p in V:
        bm.verts.new(p)
    bm.verts.ensure_lookup_table()
    for f in faces:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except ValueError:
            pass
    bm.verts.index_update()
    edge = np.array([v.index for v in bm.verts if v.is_boundary and v.link_faces])
    bm.free()
    edge = edge[V[edge, 2] > D.z_hem + 2 * VOXEL]
    d2 = ((V[edge, None, :] - out[None, :, :]) ** 2).sum(2)
    delta = out[d2.argmin(1)] - V[edge]
    band = ratios['band'] * w * 3.0
    dist = np.concatenate([np.linalg.norm(V[i:i + 3000, None, :] - V[edge][None, :, :], axis=2) for i in range(0, len(V), 3000)])
    near = dist.argmin(1)
    # (a long move is carried further, so the cloth is not dragged into a crease: over three times its length)
    reach = np.maximum(band, 3.0 * np.linalg.norm(delta, axis=1))
    rel = dist / reach[None, :]
    near = rel.argmin(1)
    k = 1.0 - _ss(0.0, 1.0, rel.min(1))
    if os.environ.get("SLEEVE_DEBUG"):
        for i_ in np.argsort(-np.linalg.norm(delta, axis=1))[:12]:
            log("  sew %s -> %s" % (np.round(V[edge[i_]], 2), np.round(delta[i_], 2)))
    log('sleeve: sewn to the seam: the edge moved %.2f on average, %.2f at most' % (
        float(np.linalg.norm(delta, axis=1).mean()), float(np.linalg.norm(delta, axis=1).max())))
    V = V + k[:, None] * delta[near]
    # (and the band smoothed, so a long move leaves no crease where it was carried: the edge itself held)
    S = _relax_polys(V, faces, 8)
    k[edge] = 0.0
    return V + np.clip(2.0 * k, 0.0, 1.0)[:, None] * (S - V)


def _relax_polys(V, faces, rounds):
    """drape._relax for faces of any number of sides (the cut leaves some three- and five-sided)."""
    e = np.array([(f[i], f[(i + 1) % len(f)]) for f in faces for i in range(len(f))])
    for _ in range(rounds):
        acc = np.zeros_like(V)
        cnt = np.zeros(len(V))
        for a, b in ((0, 1), (1, 0)):
            np.add.at(acc, e[:, a], V[e[:, b]])
            np.add.at(cnt, e[:, a], 1.0)
        V = V + 0.5 * (acc / np.maximum(cnt, 1.0)[:, None] - V)
    return V


def _largest(V, faces):
    """The largest connected piece of a mesh."""
    parent = list(range(len(V)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for f in faces:
        r0 = find(f[0])
        for i in f[1:]:
            ri = find(i)
            if ri != r0:
                parent[ri] = r0
    roots = np.array([find(i) for i in range(len(V))])
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    vals, counts = np.unique(roots[used], return_counts=True)
    keep_root = vals[np.argmax(counts)]
    faces = [f for f in faces if roots[f[0]] == keep_root]
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    return V[used], [[int(remap[i]) for i in f] for f in faces]


def _tailor(D, V, edge_pts, w, ratios):
    """The two seams and the cuff, laid into the cloth's surface (left side). Returns the points moved, and for each
    point: how far into the cuff it is (0 to 1) and how near a seam's stitching (0 to 1)."""
    al, rho = D.polar(V)
    out_dir = np.stack([np.cos(al), np.sin(al), np.zeros(len(V))], 1)
    # the seams: a shallow trough at the stitching between the allowances' two slight ridges
    sw = ratios['seam_w'] * w
    seam = np.zeros(len(V))
    lift = np.zeros(len(V))
    for a0 in (ratios['seam_fore'], ratios['seam_hind']):
        d = ((al - a0 + math.pi) % (2 * math.pi) - math.pi) * rho        # (along the surface, round the arm)
        u = d / sw
        on = (V[:, 2] < D.z_cap).astype(float) * (1.0 - _ss(D.z_cap - 0.6 * w, D.z_cap, V[:, 2]))
        seam = np.maximum(seam, on * np.exp(-u ** 2 * 6.0))
        lift += on * ratios['seam_depth'] * w * (-np.exp(-u ** 2 * 10.0) + 0.45 * np.exp(-(np.abs(u) - 0.7) ** 2 * 12.0))
    # the cuff: a band standing proud of the sleeve, its top edge rounded
    cuff = 1.0 - _ss(D.z_cuff - 0.04 * w, D.z_cuff + 0.10 * w, V[:, 2])
    lift += ratios['cuff_proud'] * w * cuff
    return V + lift[:, None] * out_dir, cuff, seam


def _buttons(D, side, w, ratios, material=None, name='Sleeve buttons'):
    """Small domed buttons up the vent on the hindarm seam (left side, mirrored by side)."""
    bm = bmesh.new()
    a0 = ratios['seam_hind'] + 0.10           # (just behind the seam: on the overlap of the vent)
    for k in range(ratios['buttons']):
        z = D.z_hem + 0.24 * w + k * ratios['button_step'] * w
        R = float(D._look(D.R, np.array([z]), np.array([a0]))[0]) + ratios['cuff_proud'] * w + ratios['thick'] * w
        cx, cy = D.axis(z)
        n = np.array([math.cos(a0), math.sin(a0), 0.0])
        c = np.array([cx, cy, z]) + R * n
        rb = ratios['button'] * w
        ret = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=rb)
        vs = ret['verts']
        # (flattened to a dome, its back sunk into the cloth, turned to face out from the arm)
        rot = Vector((0, 0, 1)).rotation_difference(Vector(n.tolist())).to_matrix()
        for v in vs:
            p = Vector((v.co.x, v.co.y, v.co.z * 0.45 + 0.15 * rb))
            p = rot @ p
            v.co = Vector(((c[0] + p.x) * side, c[1] + p.y, c[2] + p.z))
    if side < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if material is not None:
        me.materials.append(material)
    return o


def _skin(o, body, P, aw, D, side, groups_wanted=('Chest', 'Neck', 'UpperArm', 'Forearm', 'Hand')):
    """Hand the sleeve to the body's bones. The tailored sleeve takes the arm's bones by height along the axis,
    averaged over a span (a sleeve bends at the elbow as one round tube); the head takes the skin's own blend of the
    chest and the upper arm, from the skin under it."""
    me = o.data
    groups = {g.index: o.vertex_groups.new(name=g.name) for g in body.vertex_groups
              if g.name.split('.')[0] in groups_wanted}
    bverts = body.data.vertices

    def tree(pick):
        ids = [int(i) for i in np.nonzero(pick)[0]]
        kd = KDTree(len(ids))
        for k_, i in enumerate(ids):
            kd.insert(Vector(P[i]), k_)
        kd.balance()
        return kd, ids
    # (the tube from the arm's own skin only; the head from all the skin round the shoulder, chest and arm alike)
    kd_arm, ids_arm = tree((side * P[:, 0] > 0) & (aw > 0.5))
    kd_top, ids_top = tree((side * P[:, 0] > 0) & (P[:, 2] > D.z_cap - D.M['w']))
    w = D.M['w']
    for v in me.vertices:
        z = v.co.z
        cx, cy = D.axis(min(z, D.zs[-1]))
        head = float(_ss(D.z_cap - 0.4 * w, D.z_under, z))
        got, sub = {}, 0.0
        if head < 1.0:
            for dz_ in np.linspace(-SPAN * w, SPAN * w, 7):
                for _, bi, dist in kd_arm.find_n(Vector((side * float(cx), float(cy), z + dz_)), 4):
                    k_ = (1.0 - head) / (dist + 0.05)
                    for g in bverts[ids_arm[bi]].groups:
                        if g.group in groups:
                            got[g.group] = got.get(g.group, 0.0) + k_ * g.weight
                            sub += k_ * g.weight
        if head > 0.0:
            for _, bi, dist in kd_top.find_n(v.co, 8):
                k_ = head / (dist + 0.05)
                for g in bverts[ids_top[bi]].groups:
                    if g.group in groups:
                        got[g.group] = got.get(g.group, 0.0) + k_ * g.weight
                        sub += k_ * g.weight
        for gi_, wt in got.items():
            if wt / sub > 1e-3:
                groups[gi_].add([v.index], wt / sub, 'REPLACE')
    return groups


def _copy_weights(dst, src):
    """Each of dst's points takes the bones of src's nearest point (the buttons ride on the cuff)."""
    kd = KDTree(len(src.data.vertices))
    for v in src.data.vertices:
        kd.insert(v.co, v.index)
    kd.balance()
    groups = {g.index: dst.vertex_groups.new(name=g.name) for g in src.vertex_groups}
    for v in dst.data.vertices:
        _, i, _ = kd.find(v.co)
        for g in src.data.vertices[i].groups:
            groups[g.group].add([v.index], g.weight, 'REPLACE')


def build_sleeves(body, M, rig=None, material=None, button_material=None, ratios=RATIOS, log=print, settle=False,
                  colliders=(), name='Redingote sleeve'):
    """Draft both sleeves on the body (M: fit.measure's measurements), cut them at the shared armhole and the wrist,
    let the solver gather the heads, lay in the seams and the cuffs, and hand them to the body's bones. Returns
    ([sleeve.L, sleeve.R], [buttons.L, buttons.R], the left Draft)."""
    P, T, aw = fit.body_arrays(body)
    w = M['w']
    loops = armhole.armhole(P, aw, M, T)
    gi = {g.name: g.index for g in body.vertex_groups}
    fore = [v.index for v in body.data.vertices if any(g.group == gi.get('Forearm.L') and g.weight > 0.5 for g in v.groups)]
    if not fore:
        raise RuntimeError('sleeve: the body has no forearm to end the sleeve at')
    if rig is not None and 'Forearm.L' in rig.data.bones:
        b = rig.data.bones['Forearm.L']
        z_elbow, z_wrist = float((rig.matrix_world @ b.head_local).z), float((rig.matrix_world @ b.tail_local).z)
    else:
        z_elbow, z_wrist = float(P[fore, 2].max()), float(P[fore, 2].min())
    D = Draft(P, T, aw, M, loops, z_elbow, z_wrist, ratios, log)
    V0, faces0 = _mesh_from_field(D, log)
    V0 = _sew(D, V0, faces0, w, ratios, log)
    arm_skin = _arm_skin(body, aw, rig) if rig is not None else None
    sleeves, buttons = [], []
    for side, sfx in ((1.0, 'L'), (-1.0, 'R')):
        V = V0.copy()
        faces = [list(f) for f in faces0]
        V, cuff, seam = _tailor(D, V, None, w, ratios)
        al, rho = D.polar(V)
        V[:, 0] *= side
        me = bpy.data.meshes.new('%s.%s' % (name, sfx))
        bm = bmesh.new()
        lay = bm.loops.layers.uv.new('UVMap')
        vs = [bm.verts.new(p) for p in V]
        for f in faces:
            ff = f if side > 0 else f[::-1]
            try:
                face = bm.faces.new([vs[i] for i in ff])
            except ValueError:
                continue
            an = al[ff]
            if np.ptp(an) > math.pi:
                an = np.where(an < 0, an + 2 * math.pi, an)
            for lp, i, a_ in zip(face.loops, ff, an):
                lp[lay].uv = (float(a_) * D.R0, float(V[i][2]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        edge = np.array([v.index for v in bm.verts if v.is_boundary and v.link_faces])
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
        rest.data.foreach_set('vector', [c for v in me.vertices for c in v.co])
        # the cut edges (the hem and the seam), for the wool's fulled edge
        Ve = V[edge]
        d_e = np.concatenate([np.linalg.norm(V[i:i + 3000, None, :] - Ve[None, :, :], axis=2).min(1) for i in range(0, len(V), 3000)])
        ed = me.attributes.new('edge', 'FLOAT', 'POINT')
        ed.data.foreach_set('value', (1.0 - _ss(0.0, 0.25, d_e)).tolist())
        sm = me.attributes.new('seam', 'FLOAT', 'POINT')
        sm.data.foreach_set('value', seam.tolist())
        o = bpy.data.objects.new('%s.%s' % (name, sfx), me)
        bpy.context.scene.collection.objects.link(o)
        if material is not None:
            me.materials.append(material)
        _skin(o, body, P, aw, D, side)
        if rig is not None:
            m = o.modifiers.new('rig', 'ARMATURE')
            m.object = rig
            m.use_deform_preserve_volume = True
            if settle:
                _settle(o, rig, body, colliders, D, V * np.array([side, 1.0, 1.0]), edge, cuff, side, w, ratios, log, aw=aw)
            # kept off the skin in every pose: where a bent joint brings the arm's flesh up under the cloth (the
            # forearm's muscle into the inside of the elbow, the elbow's point into the back of it), the cloth is
            # carried out over it instead of the arm showing through. Only the tailored sleeve: the head stands off.
            g = o.vertex_groups.new(name='keep off')
            zr = np.array([v.co.z for v in me.vertices])
            kw = 1.0 - _ss(D.z_cap, D.z_under, zr)
            for i in np.nonzero(kw > 1e-3)[0]:
                g.add([int(i)], float(kw[i]), 'REPLACE')
            m = o.modifiers.new('keep off', 'SHRINKWRAP')
            m.target = arm_skin
            m.wrap_method = 'NEAREST_SURFACEPOINT'
            m.wrap_mode = 'OUTSIDE_SURFACE'
            m.offset = KEEP_OFF * w
            m.vertex_group = 'keep off'
        m = o.modifiers.new('smooth', 'SUBSURF')
        m.levels = m.render_levels = 1
        m = o.modifiers.new('thickness', 'SOLIDIFY')
        m.thickness = ratios['thick'] * w
        m.offset = 1.0
        m.use_rim = True
        bt = _buttons(D, side, w, ratios, button_material, name='%s buttons.%s' % (name, sfx))
        _copy_weights(bt, o)
        if rig is not None:
            m = bt.modifiers.new('rig', 'ARMATURE')
            m.object = rig
        sleeves.append(o)
        buttons.append(bt)
    return sleeves, buttons, D


def _arm_skin(body, aw, rig, name='Sleeve arm skin'):
    """The arms' own skin, posed exactly as the body's is (the body itself, with all its modifiers, its torso masked
    away): the sleeves are kept off it, and not off the torso, since where an arm hangs against the hip, keeping the
    cloth off the hip's skin would push it into the arm. It shares the body's mesh, so it costs nothing to keep."""
    o = body.copy()
    o.name = name
    bpy.context.scene.collection.objects.link(o)
    g = o.vertex_groups.get('sleeve arm') or o.vertex_groups.new(name='sleeve arm')
    g.add([int(i) for i in np.nonzero(aw > 0.5)[0]], 1.0, 'REPLACE')
    m = o.modifiers.new('arm only', 'MASK')
    m.vertex_group = 'sleeve arm'
    m.threshold = 0.5
    o.hide_render = True
    o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    o.display_type = 'WIRE'
    return o


def _settle(o, rig, body, colliders, D, V, edge, cuff, side, w, ratios, log, aw=None):
    """The head's gathers: its cloth let out by the fullness, held along the seam (and the cuff and the tailored
    sleeve below held where they were drafted), so the extra cloth gathers into the armhole and stands on the head's
    support, a hidden shell just inside the drafted puff (a riding habit's sleeve head is mounted on a stiffened net)."""
    zl = V[:, 2]
    head = _ss(D.z_cap - 0.3 * w, D.z_cap + 0.5 * w, zl) * D.top(D.polar(V)[0], zl)
    Ve = V[edge]
    at_seam_e = Ve[:, 2] > D.z_hem + 0.5 * VOXEL
    d_seam = np.concatenate([np.linalg.norm(V[i:i + 3000, None, :] - Ve[at_seam_e][None, :, :], axis=2).min(1)
                             for i in range(0, len(V), 3000)])
    band = ratios['band'] * w
    # (the gathers: along the seam the head's cloth is longer than the seam it is sewn to, fading out over the band;
    # the crown is drafted at its height and is not let out, or it would only balloon)
    full = ratios['full'] * head * (1.0 - _ss(0.3 * band, 1.6 * band, d_seam))
    # held: the seam itself, the tailored sleeve (lightly, so it can settle on the arm), the cuff firmly
    pinned = np.maximum.reduce([(d_seam < 0.5 * VOXEL).astype(float), 0.25 * (1.0 - head), cuff])
    _settle_head(o, rig, body, colliders, D, V, head, full, pinned, side, w, log, aw_body=aw)


def _settle_head(o, rig, body, colliders, D, V, head, full, pinned, side, w, log, margin=3, aw_body=None):
    """solver.settle on the head alone: the head's cloth and a few rows of the sleeve below it, cut out as a piece of
    its own, on the posed shoulder (only the skin round it as the collider) and on the head's support, a hidden shell
    just inside the drafted puff (a riding habit's sleeve head is mounted on a stiffened net). The tailored sleeve
    below is drafted close and needs no solving; settling the whole garment on the whole body took ten minutes.
    The piece is let go with the head's own stiffness (wool on its interlining), and what it settles to is carried
    back to the rest shape through the bones, as solver.settle does."""
    import time
    sc = bpy.context.scene
    me = o.data
    n = len(me.vertices)
    arm = next(m for m in o.modifiers if m.type == 'ARMATURE')
    keep_pv = arm.use_deform_preserve_volume
    arm.use_deform_preserve_volume = False
    B = solver._bone_matrices(o, rig)
    rest = np.empty(n * 3)
    me.vertices.foreach_get('co', rest)
    rest = rest.reshape(-1, 3)
    posed = np.einsum('nij,nj->ni', B[:, :3, :3], rest) + B[:, :3, 3]
    # the piece: the head, grown by a few rings of faces into the sleeve below
    polys = [tuple(p.vertices) for p in me.polygons]
    inside = head > 1e-3
    grown = inside.copy()
    for _ in range(margin):
        nxt = grown.copy()
        for f in polys:
            if any(grown[i] for i in f):
                nxt[list(f)] = True
        grown = nxt
    ids = np.nonzero(grown)[0]
    remap = np.full(n, -1)
    remap[ids] = np.arange(len(ids))
    sub_f = [tuple(int(remap[i]) for i in f) for f in polys if all(grown[i] for i in f)]
    sm = bpy.data.meshes.new('head sim')
    sm.from_pydata([tuple(p) for p in posed[ids]], [], sub_f)
    cloth = bpy.data.objects.new('head sim', sm)
    sc.collection.objects.link(cloth)
    # held: the seam, the piece's own lower edge (it is sewn on to the sleeve there), and as given
    bm = bmesh.new()
    bm.from_mesh(sm)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    rim = np.array([v.is_boundary for v in bm.verts], dtype=bool)
    bm.free()
    pin = np.maximum(pinned[ids], (rim & ~inside[ids]).astype(float))
    pin = np.maximum(pin, (~inside[ids]).astype(float) * 0.6)
    g = cloth.vertex_groups.new(name='pin')
    for i in np.nonzero(pin > 0)[0]:
        g.add([int(i)], float(pin[i]), 'REPLACE')
    # the collider: the posed skin near the piece, and the support
    lo, hi = posed[ids].min(0) - 1.0, posed[ids].max(0) + 1.0
    dg = bpy.context.evaluated_depsgraph_get()
    pts, tris = [], []
    for ob in (body, *colliders):
        e = ob.evaluated_get(dg)
        em = e.to_mesh()
        em.calc_loop_triangles()
        X = np.array([(ob.matrix_world @ v.co)[:] for v in em.vertices])
        Tt = np.array([t.vertices[:] for t in em.loop_triangles])
        c = X[Tt].mean(1)
        k = ((c > lo) & (c < hi)).all(1)
        if ob is body:
            # (the arm's and the shoulder's skin only: where the arm hangs against the chest the two skins meet, and
            # cloth started between them is thrown out by both at once)
            k &= (aw_body[Tt] > 0.25).any(1)
        pts.append(X)
        tris.append(Tt[k] + sum(len(p_) for p_ in pts[:-1]))
        e.to_mesh_clear()
    sup = _support(D, side, w)
    S = np.array([v.co[:] for v in sup.data.vertices])
    pts.append(S)
    tris.append(np.array([t for p_ in sup.data.polygons for t in _fan(p_.vertices)]) + sum(len(p_) for p_ in pts[:-1]))
    bpy.data.objects.remove(sup)
    cm = bpy.data.meshes.new('head collider')
    cm.from_pydata([tuple(p) for p in np.concatenate(pts)], [], [tuple(int(i) for i in t) for t in np.concatenate(tris)])
    col = bpy.data.objects.new('head collider', cm)
    sc.collection.objects.link(col)
    col.modifiers.new('collision', 'COLLISION')
    col.collision.thickness_outer = WOOL['gap']
    col.collision.thickness_inner = 0.2
    col.collision.cloth_friction = WOOL['friction']
    P = dict(WOOL, bending=WOOL['bending'] * HEAD_STIFF, quality=int(os.environ.get('SLEEVE_Q', 9)), frames=HEAD_FRAMES)
    m = cloth.modifiers.new('cloth', 'CLOTH')
    st, cs = m.settings, m.collision_settings
    solver.weave(st, P)
    st.vertex_group_mass, st.pin_stiffness = 'pin', 1.0
    solver.let_out(cloth, st, full[ids], P)
    cs.use_collision, cs.collision_quality, cs.distance_min = True, 4, 0.02
    cs.use_self_collision, cs.self_distance_min, cs.self_friction = True, P['self_gap'], 8.0
    m.point_cache.frame_start, m.point_cache.frame_end = 1, P['frames']
    end0, f0 = sc.frame_end, sc.frame_current
    sc.frame_end = max(sc.frame_end, P['frames'])
    t0 = time.time()
    # (the rig stays at the pose the piece was cut in: the body is not in the collider, only its posed copy)
    for f in range(1, P['frames'] + 1):
        sc.frame_set(f)
    out = np.empty(len(ids) * 3)
    cloth.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices.foreach_get('co', out)
    out = out.reshape(-1, 3)
    moved = np.linalg.norm(out - posed[ids], axis=1)
    if os.environ.get('SLEEVE_DEBUG'):
        for i_ in np.argsort(-moved)[:6]:
            log('  flew %d: from %s by %.2f, pin %.2f full %.2f rim %s inside %s' % (
                i_, np.round(posed[ids][i_], 2), moved[i_], pin[i_], full[ids][i_], rim[i_], inside[ids][i_]))
    log('sleeve: the head settled in %d frames (%.0f s), %d points: moved %.2f on average, %.2f at most' % (
        P['frames'], time.time() - t0, len(ids), float(moved.mean()), float(moved.max())))
    Bi = np.linalg.inv(B[ids])
    back = np.einsum('nij,nj->ni', Bi[:, :3, :3], out) + Bi[:, :3, 3]
    rest[ids] = back
    me.vertices.foreach_set('co', rest.reshape(-1))
    me.attributes['rest'].data.foreach_set('vector', rest.reshape(-1))
    me.update()
    for ob in (cloth, col):
        d = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(d)
    sc.frame_end = end0
    sc.frame_set(f0)
    arm.use_deform_preserve_volume = keep_pv


def _fan(vs):
    vs = list(vs)
    return [(vs[0], vs[i], vs[i + 1]) for i in range(1, len(vs) - 1)]


def _support(D, side, w):
    """The head's net support: the drafted head's surface, a little inside it, from the cap's base up to the seam."""
    reach = float(D.R.max()) + 0.6
    cx, cy = D.axis(D.zs)
    lo = np.array([cx.min() - reach, cy.min() - reach, D.z_cap - 0.2])
    hi = np.array([cx.max() + reach, cy.max() + reach, D.zs[-1] - 0.1])
    n = np.ceil((hi - lo) / VOXEL).astype(int) + 1
    gx, gy, gz = (lo[a] + VOXEL * np.arange(n[a]) for a in range(3))
    X = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), 3).reshape(-1, 3)
    # (a little inside the cloth everywhere, the tube too, so the cloth does not start inside its own support)
    F = np.concatenate([D.field(X[a:a + 400000], puff=0.65, gathers=False) for a in range(0, len(X), 400000)]).reshape(n) + 0.15
    V, faces = drape.surface_nets(F, lo, VOXEL)
    keep = np.ones(len(V), bool)
    V[:, 0] *= side
    me = bpy.data.meshes.new('sleeve support')
    me.from_pydata([tuple(p) for p in V], [], [tuple(int(i) for i in f) for f in faces if keep[f].all()])
    o = bpy.data.objects.new('sleeve support', me)
    bpy.context.scene.collection.objects.link(o)
    return o


def clearance(sleeve, body, D, side, log=print):
    """How the sleeve sits on the posed body at the current frame: how much of it is inside the arm's skin, the least
    and the mean gap to the skin (from the sleeve's inner face)."""
    dg = bpy.context.evaluated_depsgraph_get()

    def pts(o, mod_off=('thickness', 'smooth')):
        saved = []
        for m in o.modifiers:
            if m.name in mod_off:
                saved.append((m, m.show_viewport))
                m.show_viewport = False
        dg.update()
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        X = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
        tris = []
        me.calc_loop_triangles()
        tris = [tuple(t.vertices) for t in me.loop_triangles]
        e.to_mesh_clear()
        for m, s in saved:
            m.show_viewport = s
        dg.update()
        return X, tris
    S, _ = pts(sleeve)
    B, BT = pts(body)
    tree = BVHTree.FromPolygons([Vector(p) for p in B], BT)
    gaps = []
    for p in S:
        loc, nrm, _, dist = tree.find_nearest(Vector(p))
        if loc is None:
            continue
        gaps.append(dist if (Vector(p) - loc).dot(nrm) >= 0 else -dist)
    g = np.array(gaps)
    return dict(inside=float((g < 0).mean()), least=float(g.min()), mean=float(np.median(g)), n=len(g))
