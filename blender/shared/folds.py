"""Folds: authored folds (a lapel turned back over its roll line, a collar turned down at its roll, a pressed hem or
pleat) baked into a sheet in the local fabric solver (fabric.py), quickly and to a measured angle.

The fabric's own bending (ties two points along, and the petticoat's flattening stencils) only ever pulls a sheet
back toward the shape it was cut in, so it cannot hold a fold: a sheet laid out folded springs open, and a sheet
asked to fold by moving its rest lengths creeps there slowly and stops wherever its points slow below settle()'s tol.
That is how a lapel ended 19 degrees short of its target. Here the fold is a constraint of its own, as in
src/world/fabric-core.js (crease, fold; PR #61):

  - a hinge on each inner edge of the triangle mesh keeps the angle between its two triangles at a rest angle
    (Bridson, Marino and Fedkiw 2003; as an XPBD constraint, Mueller et al. 2007 and Macklin et al. 2016): its
    gradient moves the two wing points along their triangles' normals and the edge's two points against them;
  - the rest angles are not guessed per edge: they are measured on the fold as a tailor makes it, the flat pattern
    turned about the fold line through a cylinder of the fold's radius. A pressed fold has the radius of the cloth's
    thickness, so its whole turn falls on the one row of edges along the line (crisp); a rolled fold (a collar's or a
    lapel's roll) has a radius of its own, so the turn spreads over a band r * angle wide, a share on each row. Only
    the hinges whose measured rest is not flat are made: the rest of the cloth keeps its soft bending;
  - the cloth's own bend ties and flattening stencils across the band are softened (they would pull it flat);
  - the angle is eased in (ramp) so the flap swings over instead of snapping and throwing itself through the body,
    then the sheet is held until the angle measured across the fold (the sheet's normals either side of the band) is
    within tol of the target. The stop is the angle, never the points' speed alone and never a step count: if the
    run reaches `most` steps first, the report says it stopped short, by how much, and the result is not cached.

    import folds as fo
    F = fo.Folds(fab, flat, T)                         flat: the pattern [n, 2] (or [n, 3] in its plane) as cut
    fo.soft_at(flat, T, [crease, ...])                 per-point compliance for fab.mesh(bend_at=...) (PR #59)
    c = F.add(line, angle, radius=0.0, side=+1)        line: points on the pattern [k, 2]; angle in radians
    report = F.bake(body=None, h=1/240, tol_deg=1.0, ...)
    report.ok, report.achieved_deg, report.target_deg, report.steps
    pos, report = fo.bake_cached(key_parts, fab, flat, T, creases, ...)   the fold cache (see bake_cached)

A crease's angle is how far the far side turns from flat: 0 flat, about 175 degrees for a pressed lapel lying back on
the front (180 would put the two layers through each other; the cloth's thickness keeps it short of that), 90 for a
stand collar's turn. side (+1 or -1) is which way it turns: toward the pattern's +z (its right side up) or away. The
side of the line that moves is the side to the left of the line's direction (the line's in-plane normal, -dy, dx).

Units are the caller's. The module needs numpy only, so it runs in Blender and outside it (test_folds.py)."""
import hashlib
import json
import math
import os
import tempfile

import numpy as np

CACHE_VERSION = 1


# ------------------------------------------------------------------ geometry

def _flat3(flat):
    flat = np.asarray(flat, float)
    return flat if flat.shape[1] == 3 else np.hstack([flat, np.zeros((len(flat), 1))])


def hinges(T):
    """Every inner edge of a consistently wound triangle mesh as (a, b, c, d): the edge c -> d, a the third point of
    the triangle (a, c, d), b of (b, d, c)."""
    T = np.asarray(T, np.int64)
    seen = {}
    out = []
    for p, q, r in T:
        for u, v, o in ((p, q, r), (q, r, p), (r, p, q)):
            other = seen.pop((v, u), None)
            if other is not None:
                out.append((other[1], o, other[0][0], other[0][1]))      # (a: o of the first, b: this o, c, d)
            else:
                seen[(u, v)] = ((u, v), o)
    return np.array(out, np.int64).reshape(-1, 4)


def dihedral(x, H):
    """The signed angle at each hinge (0 flat; positive when the wings turn toward the side the triangles face)."""
    a, b, c, d = x[H[:, 0]], x[H[:, 1]], x[H[:, 2]], x[H[:, 3]]
    N1 = np.cross(c - a, d - a)
    N2 = np.cross(d - b, c - b)
    e = d - c
    n1 = N1 / np.maximum(np.linalg.norm(N1, axis=1), 1e-12)[:, None]
    n2 = N2 / np.maximum(np.linalg.norm(N2, axis=1), 1e-12)[:, None]
    eh = e / np.maximum(np.linalg.norm(e, axis=1), 1e-12)[:, None]
    return np.arctan2(np.einsum('ij,ij->i', np.cross(n1, n2), eh), np.einsum('ij,ij->i', n1, n2))


def _grads(x, H):
    """The angle's gradient at the four points of each hinge (checked against finite differences in test_folds)."""
    a, b, c, d = x[H[:, 0]], x[H[:, 1]], x[H[:, 2]], x[H[:, 3]]
    N1 = np.cross(c - a, d - a)
    N2 = np.cross(d - b, c - b)
    e = d - c
    L = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
    eh = e / L[:, None]
    q1 = N1 / np.maximum(np.einsum('ij,ij->i', N1, N1), 1e-18)[:, None]
    q2 = N2 / np.maximum(np.einsum('ij,ij->i', N2, N2), 1e-18)[:, None]
    ga = -L[:, None] * q1
    gb = -L[:, None] * q2
    gc = -(np.einsum('ij,ij->i', a - d, eh)[:, None] * q1 + np.einsum('ij,ij->i', b - d, eh)[:, None] * q2)
    gd = np.einsum('ij,ij->i', a - c, eh)[:, None] * q1 + np.einsum('ij,ij->i', b - c, eh)[:, None] * q2
    return ga, gb, gc, gd


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _colour(H, fixed):
    """Batches of hinges, no two in a batch sharing a moving point (so a batch is solved at once)."""
    col = np.full(len(H), -1)
    used = {}
    for k, h in enumerate(H):
        pts = [int(p) for p in h if not fixed[p]]
        c = 0
        while any(c in used.get(p, ()) for p in pts):
            c += 1
        col[k] = c
        for p in pts:
            used.setdefault(p, set()).add(c)
    return [np.nonzero(col == c)[0] for c in range(col.max() + 1)] if len(H) else []


# ------------------------------------------------------------------ the fold as a tailor makes it

class Crease:
    """A fold along a line on the pattern. angle: radians (how far the far side turns from flat); radius: the fold's
    inside radius, in the pattern's units (0 or the cloth's thickness: pressed; larger: rolled); side: +1 turns the
    moving side toward the pattern's +z, -1 away; compliance: how firmly the hinges hold their rest (0 rigid)."""

    def __init__(self, line, angle, radius=0.0, side=1, compliance=0.0, name='fold'):
        self.line = np.asarray(line, float)[:, :2]
        if len(self.line) < 2:
            raise ValueError('a crease needs a line of two points or more')
        self.angle = float(angle)
        self.radius = float(radius)
        self.side = 1 if side >= 0 else -1
        self.compliance = float(compliance)
        self.name = name

    def spec(self):
        return dict(line=np.round(self.line, 6).tolist(), angle=round(self.angle, 6), radius=round(self.radius, 6),
                    side=self.side, compliance=self.compliance, name=self.name)

    def width(self, h):
        """The band the turn spreads over: the arc of the fold's radius through its angle; none if that is narrower
        than the mesh's rows h (a pressed fold: its whole turn falls on the edges along its line)."""
        w = abs(self.angle) * self.radius
        return w if w >= h else 0.0

    def frame(self, P):
        """For pattern points P [m, 2]: the nearest segment of the line, the distance along it, and the signed
        distance across it (positive on the moving side)."""
        A, B = self.line[:-1], self.line[1:]
        D = B - A
        Ln = np.maximum(np.linalg.norm(D, axis=1), 1e-12)
        U = D / Ln[:, None]
        rel = P[:, None, :] - A[None]
        t = np.einsum('mkj,kj->mk', rel, U)
        tc = np.clip(t, 0, Ln[None])
        foot = A[None] + tc[..., None] * U[None]
        dist = np.linalg.norm(P[:, None, :] - foot, axis=2)
        k = np.argmin(dist, axis=1)
        m = np.arange(len(P))
        Nrm = np.column_stack([-U[:, 1], U[:, 0]])
        s = np.einsum('mj,mj->m', rel[m, k], Nrm[k])
        return k, t[m, k], s

    def folded(self, P2, k):
        """The pattern points P2 turned about segment k of the line as the fold is made (a cylinder of the fold's
        radius, then straight on), in 3D (z: the pattern's normal)."""
        A, B = self.line[k], self.line[k + 1]
        u = (B - A) / max(np.linalg.norm(B - A), 1e-12)
        nrm = np.array([-u[1], u[0]])
        rel = P2 - A
        t = rel @ u
        s = rel @ nrm
        th = abs(self.angle)
        r = self.radius
        W = th * r
        z = self.side * (1.0 if self.angle >= 0 else -1.0)
        out_s = np.where(s <= 0, s, 0.0)
        out_z = np.zeros(len(P2))
        if r > 0:
            arc = (s > 0) & (s < W)
            phi = np.where(arc, s / r, 0.0)
            out_s = np.where(arc, r * np.sin(phi), out_s)
            out_z = np.where(arc, z * r * (1 - np.cos(phi)), out_z)
        past = s >= W
        e_s, e_z = (r * math.sin(th), z * r * (1 - math.cos(th))) if r > 0 else (0.0, 0.0)
        out_s = np.where(past, e_s + (s - W) * math.cos(th), out_s)
        out_z = np.where(past, e_z + z * (s - W) * math.sin(th), out_z)
        xy = A[None] + t[:, None] * u[None] + out_s[:, None] * nrm[None]
        return np.column_stack([xy, out_z])


def _spacing(flat3, T):
    T = np.asarray(T, np.int64)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    return float(np.median(np.linalg.norm(flat3[e[:, 0]] - flat3[e[:, 1]], axis=1)))


def rest_angles(flat, T, H, crease):
    """Each hinge's rest angle for the crease, measured on the fold made from the flat pattern: the hinge's four
    points turned about the line's segment nearest the hinge's edge."""
    flat3 = _flat3(flat)
    P = flat3[:, :2]
    mid = 0.5 * (P[H[:, 2]] + P[H[:, 3]])
    k, _, _ = crease.frame(mid)
    if crease.width(0) < _spacing(flat3, T):
        # (a fold tighter than the mesh's rows can only turn on the one row of edges along its line: its arc would
        # bend the triangles themselves)
        crease = Crease(crease.line, crease.angle, 0.0, crease.side)
    rest = np.zeros(len(H))
    for seg in np.unique(k):
        sel = np.nonzero(k == seg)[0]
        pts = np.unique(H[sel])
        X = np.zeros((len(flat3), 3))
        X[pts] = crease.folded(P[pts], seg)
        rest[sel] = dihedral(X, H[sel])
    # (the pattern's own winding: a mesh wound clockwise in the pattern faces -z, so its angles come out turned)
    f = np.asarray(T, np.int64)[0]
    up = np.cross(flat3[f[1]] - flat3[f[0]], flat3[f[2]] - flat3[f[0]])[2]
    return rest if up >= 0 else -rest


def soft_at(flat, T, creases, compliance=1e-2):
    """A compliance per point, soft on the points of every crease's hinges and 0 elsewhere: for fabric.py's
    mesh(bend_at=...) (PR #59), so the cloth's own bend ties and flattening stencils give way at the fold."""
    H = hinges(T)
    out = np.zeros(len(flat))
    for c in creases:
        r = rest_angles(flat, T, H, c)
        out[np.unique(H[np.abs(r) > 1e-3])] = compliance
    return out


def soften(fab, H, compliance=1e-2):
    """Soften, on a Fabric already meshed, the bend ties across the hinges H (the ties between their two wings) and
    the flattening stencils over their edges (PR #59's). Works with main's fabric.py and with PR #59's."""
    wings = {(int(min(a, b)), int(max(a, b))) for a, b in H[:, :2]}
    edges = {(int(min(c, d)), int(max(c, d))) for c, d in H[:, 2:]}
    n = 0
    if hasattr(fab, '_I'):                                  # PR #59: ties in lists of arrays, built on demand
        for I, J, C in zip(fab._I, fab._J, fab._comp):
            for k in range(len(I)):
                if (min(I[k], J[k]), max(I[k], J[k])) in wings:
                    C[k] = max(C[k], compliance)
                    n += 1
        for q in getattr(fab, '_quads', []):
            A, B, comp = q[0], q[1], q[5]
            for k in range(len(A)):
                if (min(A[k], B[k]), max(A[k], B[k])) in edges:
                    comp[k] = max(comp[k], compliance)
        fab._built = None
    else:                                                   # main: flat arrays
        for k in range(len(fab.ci)):
            if (min(fab.ci[k], fab.cj[k]), max(fab.ci[k], fab.cj[k])) in wings:
                fab.comp[k] = max(fab.comp[k], compliance)
                n += 1
    return n


# ------------------------------------------------------------------ the bake

class Report:
    def __init__(self):
        self.creases = []          # per crease: dict(name, target_deg, achieved_deg, worst_deg, band, hinges)
        self.steps = 0
        self.ok = False
        self.why = ''

    @property
    def achieved_deg(self):
        return [c['achieved_deg'] for c in self.creases]

    @property
    def target_deg(self):
        return [c['target_deg'] for c in self.creases]

    def __str__(self):
        head = 'folds: %s in %d steps' % ('reached' if self.ok else 'STOPPED SHORT (%s)' % self.why, self.steps)
        rows = ['  %s: %.1f of %.1f deg (worst along the line %.1f), %d hinges over a band %.3f wide'
                % (c['name'], c['achieved_deg'], c['target_deg'], c['worst_deg'], c['hinges'], c['band'])
                for c in self.creases]
        return '\n'.join([head] + rows)

    def as_dict(self):
        return dict(ok=self.ok, why=self.why, steps=self.steps, creases=self.creases)


class Folds:
    """Authored folds on a Fabric. flat: the pattern as cut, [n, 2] (or [n, 3] in its plane, z 0); T: its triangles,
    wound alike (the same T given to fab.mesh)."""

    def __init__(self, fab, flat, T, soften_compliance=1e-2):
        self.fab = fab
        self.flat = _flat3(flat)
        self.T = np.asarray(T, np.int64)
        self.H_all = hinges(self.T)
        self.h = _spacing(self.flat, self.T)
        self.creases = []
        self._H, self._rest, self._comp, self._owner = [], [], [], []
        self.soften_compliance = soften_compliance

    def add(self, line, angle, radius=0.0, side=1, compliance=0.0, name=None):
        """A crease (see Crease). Returns it."""
        c = Crease(line, angle, radius, side, compliance, name or 'fold %d' % (len(self.creases) + 1))
        r = rest_angles(self.flat, self.T, self.H_all, c)
        on = np.abs(r) > 1e-3
        if not on.any():
            raise ValueError('%s: no edge of the mesh lies on its line' % c.name)
        if c.width(self.h) == 0:
            # a pressed fold turns on the edges along its line, so the mesh must have them: a line drawn across the
            # triangles would have to bend the triangles themselves (cut the pattern with the line in it, as a
            # tailor marks the roll line: points along it before triangulating)
            _, _, sp = c.frame(self.flat[:, :2])
            e = self.H_all[on][:, 2:]
            along = (np.abs(sp[e[:, 0]]) < 0.05 * self.h) & (np.abs(sp[e[:, 1]]) < 0.05 * self.h)
            if along.mean() < 0.9:
                raise ValueError('%s: a pressed fold needs the mesh to have edges along its line (%.0f%% of the '
                                 'crossing edges do); put points on the line before triangulating, or give the fold '
                                 'a radius of two rows or more' % (c.name, 100 * along.mean()))
        self._H.append(self.H_all[on])
        self._rest.append(r[on])
        self._comp.append(np.full(on.sum(), c.compliance))
        self._owner.append(np.full(on.sum(), len(self.creases)))
        soften(self.fab, self.H_all[on], self.soften_compliance)
        self.creases.append(c)
        return c

    # -- measuring

    def measure(self, x=None):
        """Per crease: the angle the sheet turns through across it, measured from the triangles' normals just
        outside the band on either side, segment by segment along the line: (mean, worst) in degrees, signed as the
        target."""
        x = self.fab.pos if x is None else x
        P = self.flat[:, :2]
        cen = P[self.T].mean(1)
        N = np.cross(x[self.T[:, 1]] - x[self.T[:, 0]], x[self.T[:, 2]] - x[self.T[:, 0]])
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
        out = []
        for c in self.creases:
            k, t, s = c.frame(cen)
            W = c.width(self.h)
            lo = (s < -0.5 * self.h) & (s > -2.5 * self.h)
            hi = (s > W + 0.5 * self.h) & (s < W + 2.5 * self.h)
            angs = []
            for seg in range(len(c.line) - 1):
                a, b = lo & (k == seg), hi & (k == seg)
                if not a.any() or not b.any():
                    continue
                n0, n1 = N[a].mean(0), N[b].mean(0)
                n0 /= np.linalg.norm(n0)
                n1 /= np.linalg.norm(n1)
                ang = math.degrees(math.acos(np.clip(n0 @ n1, -1, 1)))
                angs.append(ang)
            tgt = math.degrees(abs(c.angle))
            if not angs:
                out.append((float('nan'), float('nan')))
                continue
            angs = np.array(angs)
            out.append((float(angs.mean()), float(angs[np.argmax(np.abs(angs - tgt))])))
        return out

    # -- solving

    def _walk(self, H, rest, comp, h, batches, lam, w=None):
        x = self.fab.pos
        w = self.fab.w if w is None else w
        for k in batches:
            Hk = H[k]
            C = _wrap(dihedral(x, Hk) - rest[k])
            g = _grads(x, Hk)
            W = [w[Hk[:, j]] for j in range(4)]
            den = sum(W[j] * np.einsum('ij,ij->i', g[j], g[j]) for j in range(4))
            al = comp[k] / (h * h)
            dl = (-C - al * lam[k]) / np.maximum(den + al, 1e-12)
            ok = den > 1e-12
            dl = np.where(ok, dl, 0.0)
            lam[k] += dl
            for j in range(4):
                x[Hk[:, j]] += (W[j] * dl)[:, None] * g[j]

    def _sides(self):
        """Per crease: the points on its moving side (past the band) and the triangles just before the line (whose
        normal is the fixed side's), each with its nearest segment; and the pattern's winding (+1 faces +z)."""
        P = self.flat[:, :2]
        cen = P[self.T].mean(1)
        f = self.T[0]
        up = 1.0 if np.cross(self.flat[f[1]] - self.flat[f[0]], self.flat[f[2]] - self.flat[f[0]])[2] >= 0 else -1.0
        out = []
        for c, Hc in zip(self.creases, self._H):
            k, _, s = c.frame(P)
            kt, _, st = c.frame(cen)
            near = (st < 0) & (st > -2.5 * self.h)
            on = np.unique(Hc[:, 2:])                                   # (the crease's own edges' points ...)
            first = on[np.abs(s[on]) < 0.5 * self.h]                    # (... where the turn begins: the axis)
            on = first if len(first) >= 2 else on
            out.append(dict(move=np.nonzero(s > 1e-9)[0], seg=k, near=near, tseg=kt, on=on, onseg=k[on]))
        return out, up

    def _lay(self, x, sides, up, f):
        """Lay each crease's moving side over to f of its fold, in the frame of the crease as it lies now (its axis
        through the points where the turn begins, the fixed side's direction and normal): the fold as made from the
        pattern (see Crease.folded), so the band takes the fold's arc and the cloth past it is turned whole, every
        length on the moving side kept. The hand that lays the flap over while the iron goes down."""
        N = np.cross(x[self.T[:, 1]] - x[self.T[:, 0]], x[self.T[:, 2]] - x[self.T[:, 0]]) * up
        w = self.fab.w
        P = self.flat[:, :2]
        for c, sd in zip(self.creases, sides):
            r = c.radius if c.width(self.h) > 0 else 0.0
            part = Crease(c.line, c.angle * f, r, c.side)
            for seg in range(len(c.line) - 1):
                pts = sd['on'][sd['onseg'] == seg]
                sel = (sd['seg'][sd['move']] == seg) & (w[sd['move']] > 0)
                mv = sd['move'][sel]
                tri = sd['near'] & (sd['tseg'] == seg)
                if len(pts) < 2 or not len(mv) or not tri.any():
                    continue
                A = c.line[seg]
                u = c.line[seg + 1] - A
                u /= max(np.linalg.norm(u), 1e-12)
                nrm = np.array([-u[1], u[0]])
                Q = x[pts]
                o = Q.mean(0)
                ax = np.linalg.svd(Q - o)[2][0]                          # (the line through the crease's points)
                tq = (P[pts] - A) @ u
                if np.corrcoef(tq, (Q - o) @ ax)[0, 1] < 0 if len(pts) > 2 else ((Q[1] - Q[0]) @ ax) * (tq[1] - tq[0]) < 0:
                    ax = -ax                                             # (oriented as the line runs on the pattern)
                n = N[tri].sum(0)
                n -= ax * (n @ ax)
                n /= max(np.linalg.norm(n), 1e-12)
                pa = np.cross(n, ax)                                     # (across, into the moving side: z x u = nrm)
                O = o - tq.mean() * ax - ((P[pts] - A) @ nrm).mean() * pa
                F3 = part.folded(P[mv], seg)
                t = (F3[:, :2] - A) @ u
                sa = (F3[:, :2] - A) @ nrm
                x[mv] = O + t[:, None] * ax + sa[:, None] * pa + F3[:, 2:3] * n

    def bake(self, body=None, h=1.0 / 240, ramp=None, speed=6.0, tol_deg=1.0, tol=2e-4, most=3000, passes=4,
             damping=0.35, guide=True, log=print):
        """Fold the sheet:

          1. over ramp seconds (default: the largest angle over speed, radians a second) the creases' rest angles
             are eased in from flat, and (guide) the moving side of each is laid over with them in the crease's frame
             as it lies (see _lay). A cloth's own bending is far weaker
             than a crease's hinges, so left to the hinges alone the crease turns only the row beside it and the
             cloth beyond bends back (a kink, not a fold), and the turn creeps across the flap one row a walk;
             a tailor lays the flap over by hand as the iron goes down. The turn keeps every length on the moving
             side, so the cloth there is not strained by it;
          2. then held, the hinges keeping the crease and the cloth draping by its own softness under gravity and on
             the body, until every crease is within tol_deg of its target (measured: see measure) and the fastest
             point moves less than tol a step.

        passes: hinge walks a step. Returns a Report; report.ok is False if `most` steps ran out first, and
        report.why says by how much and where."""
        fab = self.fab
        H = np.concatenate(self._H)
        rest = np.concatenate(self._rest)
        comp = np.concatenate(self._comp)
        batches = _colour(H, fab.w == 0)
        if ramp is None:
            ramp = max(abs(c.angle) for c in self.creases) / speed
        nramp = max(int(round(ramp / h)), 1)
        sides, up = self._sides()
        tgts = np.array([abs(c.angle) for c in self.creases])
        s = fab.s
        was = s['damping']
        s['damping'] = damping
        rep = Report()
        k = 0
        try:
            for k in range(1, most + 1):
                f = min(k / nramp, 1.0)
                f = f * f * (3 - 2 * f)                        # (smoothstep: no jolt at the start or the end)
                if guide and k <= nramp:
                    self._lay(fab.pos, sides, up, f)
                    fab.prev[:] = fab.pos                       # (the hand's turn is not kept as speed)
                fab.step(h, body, neutral=True)
                lam = np.zeros(len(H))
                for _ in range(passes):
                    self._walk(H, rest * f, comp, h, batches, lam)
                if body is not None:
                    p, touched = body(fab.pos.copy())
                    live = touched & (fab.w > 0)
                    fab.pos[live] = p[live]
                if k >= nramp and k % 10 == 0:
                    m = self.measure()
                    far = float(np.abs(fab.pos - fab.prev).max())
                    close = all(abs(a - math.degrees(t)) <= tol_deg for (a, _), t in zip(m, tgts))
                    if log and k % 200 == 0:
                        log('folds: step %d, %s, fastest point %.5f' % (
                            k, ', '.join('%.1f/%.1f' % (a, math.degrees(t)) for (a, _), t in zip(m, tgts)), far))
                    if close and far < tol:
                        rep.ok = True
                        break
        finally:
            s['damping'] = was
            fab.prev[:] = fab.pos
        rep.steps = k
        m = self.measure()
        for (a, worst), c, Hc in zip(m, self.creases, self._H):
            rep.creases.append(dict(name=c.name, target_deg=round(math.degrees(abs(c.angle)), 2),
                                    achieved_deg=round(a, 2), worst_deg=round(worst, 2), hinges=int(len(Hc)),
                                    band=round(max(c.width(self.h), self.h), 4)))
        if not rep.ok:
            short = [c for c in rep.creases if not abs(c['achieved_deg'] - c['target_deg']) <= tol_deg]
            rep.why = ('%s %.1f deg off' % (short[0]['name'], short[0]['achieved_deg'] - short[0]['target_deg'])
                       if short else 'still moving') + ' after %d steps' % k
        if log:
            log(str(rep))
        return rep


# ------------------------------------------------------------------ the fold cache

def cache_dir():
    """The fold cache: EMBER_CLOTH_CACHE/folds (the layers' settle memory's folder, PR #58), else the temp folder."""
    d = os.path.join(os.environ.get('EMBER_CLOTH_CACHE') or os.path.join(tempfile.gettempdir(), 'emberwood-cloth'),
                     'folds')
    os.makedirs(d, exist_ok=True)
    return d


def cache_key(fab, flat, T, creases, bake_args, extra=()):
    """A key of everything the folded result depends on: this module's and fabric.py's code, the cloth's points as
    they start, its pins and pattern and triangles, the creases, the fabric's settings, the bake's arguments, and
    whatever the caller adds in extra (the body's surface, for one: a body is a function, so pass its points)."""
    hs = hashlib.sha256()
    for mod in (__file__, getattr(__import__(type(fab).__module__), '__file__', None)):
        if mod and os.path.exists(mod):
            with open(mod, 'rb') as fh:
                hs.update(fh.read())
    for a in (fab.pos, fab.w, _flat3(flat), np.asarray(T, np.int64)):
        hs.update(np.ascontiguousarray(np.round(np.asarray(a, float), 6)).tobytes())
    sett = {k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v) for k, v in fab.s.items()}
    hs.update(json.dumps([CACHE_VERSION, [c.spec() for c in creases], sett, bake_args], sort_keys=True,
                         default=str).encode())
    for e in extra:
        hs.update(np.ascontiguousarray(np.round(np.asarray(e, float), 6)).tobytes())
    return hs.hexdigest()[:24]


def bake_cached(name, fab, flat, T, creases, body=None, extra=(), log=print, **bake_args):
    """Folds.bake through the fold cache: the same piece, creases, settings and body (extra) load the solved points
    and skip the solver; anything else solves and, if the folds reached their targets, saves. creases: Crease
    objects (or dicts of Crease's arguments). Returns (points, report as a dict, 'cached' | 'solved')."""
    creases = [c if isinstance(c, Crease) else Crease(**c) for c in creases]
    key = cache_key(fab, flat, T, creases, bake_args, extra)
    path = os.path.join(cache_dir(), '%s-%s.npz' % (name, key))
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        fab.pos[:] = z['pos']
        fab.prev[:] = z['pos']
        rep = json.loads(str(z['report']))
        if log:
            log('folds: %s remembered (%s)' % (name, key))
        return fab.pos.copy(), rep, 'cached'
    F = Folds(fab, flat, T)
    for c in creases:
        F.add(c.line, c.angle, c.radius, c.side, c.compliance, c.name)
    rep = F.bake(body=body, log=log, **bake_args)
    if rep.ok:
        np.savez(path, pos=fab.pos, report=json.dumps(rep.as_dict()))
    elif log:
        log('folds: %s not cached (it stopped short)' % name)
    return fab.pos.copy(), rep.as_dict(), 'solved'
