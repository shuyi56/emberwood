"""Fabric: the game's particle solver for cloth (src/world/fabric-core.js), for the Blender scripts. It folds, sews
and settles sheets here as it does in the game (John's rule, 2026-11-19: fold with the local solver; Blender's own
cloth solver is kept for final renders and for structured pieces such as the chemise). One module for every garment:
the hose thread's port (Jacobi walk, bvh_body) and the petticoat's (Gauss-Seidel walk, sewing, tethers, flattening)
joined.

From fabric-core.js, the same model:

  - every tie keeps its rest length, as stiff as its `compliance` lets it (0: rigid), within its `slack` (a share of
    the rest it may change freely before it resists);
  - XPBD (Macklin, Mueller and Chentanez 2016) in small fixed steps (Macklin et al. 2019): each step moves every
    point by its speed (Verlet: from the last step's place, kept by `damping` per 1/60 s) and by gravity, then walks
    the ties `iterations` times; a tie's compliance is divided by the step squared, so the stiffness belongs to the
    fabric, not to the step;
  - after the ties each point is pushed out of the body, and the body takes away its slide along it by Coulomb
    friction (a slide smaller than friction times the push is stopped outright);
  - held chains keep their segments' exact lengths, follow the leader with the point above taking `lead` of the
    move the other way (Mueller et al. 2012);
  - settle(): run to rest, damped hard, the body's pushes not kept as speed.

The walk (`walk`):

  - 'jacobi' (the default, the hose's): all ties at once, each point taking the mean of the moves its ties ask of it,
    over-relaxed by `relax`; the same answer as the game's at the cost of more walks a step;
  - 'gauss-seidel': as fabric-core.js walks them, one after another, forward one walk and backward the next, here in
    colour batches (no two ties of a batch share a moving point, so a batch moves at once in numpy). Fewer walks
    for the same stiffness; what the petticoat uses.

Additions to fabric-core.js's model, each documented where it is defined:

  - mesh(): a triangle or quad mesh's sheet (edges the threads, quads' diagonals the shear, the far corners across
    each edge the bend ties), its rest lengths optionally from the flat pattern (rest_pos), so the cloth comes back
    to the cloth as cut whatever it was laid in;
  - flatten(): stencils that resist a fold in proportion to its angle (bend ties two along resist only as its
    square, and cannot make a starched cloth: the swatch strip reached 1.7 cm of bending length at any compliance);
  - tether(): ties that only stop a point going further than its rest from another (a long-range attachment, Kim,
    Chentanez and Mueller 2012), so a long skirt hung from its band does not sag through a hundred threads;
  - sew(): ties drawn to nothing between a seam's paired points, their compliance eased from loose to firm;
  - bodies: a callable on all points at once, body(P) -> (pushed P, touched mask). MeshBody (scipy, the nearest
    surface point and its normal, with a floor) and bvh_body (a mathutils BVHTree) are two.

The API (kept stable: the boot, and the skirt, sleeves and petticoat scripts use it):

    f = Fabric(points [n, 3], settings)    settings: FABRIC_DEFAULTS' names, the game's where it has them
    f.pin(indices, weight=None)            hold points (weight 1, or partly: an inverse mass of 1 - weight)
    f.tie(i, j, compliance, rest=None)     ties between point arrays, at their present lengths or rest (slack=)
    f.mesh(faces)                          a mesh's threads, quad diagonals and bend ties; returns their counts
                                           (rest_pos=, bend_at=, flat=: see mesh)
    f.step(h, body, neutral=False)         one step; body(points) -> (points pushed out, touched mask)
    f.settle(body, h, most, tol, damping)  run to rest; returns the steps taken
    f.pos                                  the points as they are now
    bvh_body(tree, gap)                    a body from a mathutils BVHTree (normals outward)
  added with the petticoat:
    f.tether(i, j, rest)                   ties that only stop i going further than rest from j
    f.sew(i, j, loose, firm, seconds)      seam ties of no length, eased from loose to firm compliance
    f.flatten(a, b, c, d, rest, comp)      fold-resisting stencils (an angle bend; mesh(flat=) makes them)
    f.hold_chain(points)                   a chain kept at its exact segment lengths (fabric-core.js's hold)
    MeshBody(points, normals, gap, floor)  a body from a surface's points and normals (needs scipy; nothing else does);
                                           planes=8 keeps each point off both walls of a crease at once (armpits)

It follows src/world/fabric-core.js: what is added there (per-point mass, angle bends, creases that set) is to be
added here under the same names.

Units are the caller's; positions are an array [n, 3], z up (fabric-core.js's y), so GRAVITY is along -z: 9.81 m/s^2
in the figure's units (63 units to 1.65 m)."""
import time

import numpy as np

UNIT = 1.65 / 63
GRAVITY = (0.0, 0.0, -375.0)     # 9.8 m/s^2 on a figure 63 units to 1.65 m (UNIT), as the hose's port has it

FABRIC_DEFAULTS = dict(
    gravity=GRAVITY,         # units / s^2
    damping=0.02,            # share of each point's speed lost per 1/60 s
    walk='jacobi',           # or 'gauss-seidel' (see above)
    iterations=12,           # walks over the ties per step (Jacobi needs more than the game's 2)
    relax=1.6,               # Jacobi's over-relaxation of the mean move
    stretch=0.0,             # compliance of the threads (0: they keep their length)
    shear=2e-5,              # compliance of the diagonals
    bend=2e-4,               # compliance of the bend ties
    friction=0.5,            # how well the body holds what lies on it: the slide it stops, as a share of its push
    lead=0.9,                # held chains: the share of a point's correction the point above takes the other way
)


def _colour(I, J, n, fixed=None, more=()):
    """Batches of ties (or of stencils: I, J and the more columns), no two in a batch sharing a moving point
    (greedy, in their order; a pinned point never moves, so any number of a batch's ties may share it)."""
    cols = [np.asarray(I), np.asarray(J)] + [np.asarray(c) for c in more]
    m = len(cols[0])
    col = np.full(m, -1)
    used = [set() for _ in range(n)]
    for k in range(m):
        pts = [int(c[k]) for c in cols]
        sets = [used[p] for p in pts if fixed is None or not fixed[p]]
        c = 0
        while any(c in u for u in sets):
            c += 1
        col[k] = c
        for u in sets:
            u.add(c)
    return [np.nonzero(col == c)[0] for c in range(col.max() + 1)] if m else []


class Fabric:
    def __init__(self, pos, settings=None):
        """pos: [n, 3] (copied)."""
        self.pos = np.array(pos, dtype=float)
        self.n = len(self.pos)
        self.prev = self.pos.copy()
        self.w = np.ones(self.n)                 # inverse mass: 0 pins a point
        self.settings = dict(FABRIC_DEFAULTS, **(settings or {}))
        self.s = self.settings
        self._I, self._J, self._rest, self._comp, self._slack, self._kind = [], [], [], [], [], []
        self.chains = []
        self.touch = np.zeros(self.n, bool)
        self._quads = []                         # (a, b, c, d, rest, compliance): flattening stencils (see flatten)
        self._built = None
        self._sew = None                         # (first tie, last tie, compliance from, to, seconds)
        self.t = 0.0

    # ------------------------------------------------------------------ ties

    def pin(self, idx, weight=None):
        """Hold the points idx where they are (weight 1), or partly (an inverse mass of 1 - weight)."""
        idx = np.asarray(idx, dtype=np.int64)
        self.w[idx] = 0.0 if weight is None else np.clip(1.0 - np.asarray(weight, dtype=float), 0.0, 1.0)
        self._built = None

    def ties(self, I, J, compliance=0.0, slack=0.0, rest=None, kind=0):
        """Tie points I[k] and J[k] (arrays) at their present distance or rest; kind 1: a tether (only resists going
        further than rest). Returns the ties' index range."""
        I, J = np.asarray(I, np.int64).ravel(), np.asarray(J, np.int64).ravel()
        keep = ~((self.w[I] == 0) & (self.w[J] == 0))
        if rest is None:
            rest = np.linalg.norm(self.pos[I] - self.pos[J], axis=1)
        rest = np.broadcast_to(np.asarray(rest, float), I.shape)[keep]
        I, J = I[keep], J[keep]
        k0 = sum(len(a) for a in self._I)
        self._I.append(I)
        self._J.append(J)
        self._rest.append(rest)
        self._comp.append(np.full(len(I), float(compliance)))
        self._slack.append(np.full(len(I), float(slack)))
        self._kind.append(np.full(len(I), kind, np.int8))
        self._built = None
        return k0, k0 + len(I)

    def tie(self, i, j, compliance=0.0, rest=None, slack=0.0):
        """Tie points i to points j (single points or arrays) at the distance they are at now (or rest)."""
        i, j = np.atleast_1d(i), np.atleast_1d(j)
        keep = i != j
        if rest is not None:
            rest = np.broadcast_to(np.asarray(rest, float), i.shape)[keep]
        return self.ties(i[keep], j[keep], compliance, slack, rest)

    def tether(self, I, J, rest, compliance=0.0):
        return self.ties(I, J, compliance, 0.0, rest, kind=1)

    def sew(self, I, J, loose=1e-2, firm=0.0, seconds=0.3):
        """Draw the paired points together: ties to nothing, their compliance eased from loose to firm over
        seconds (a seam pulled up at once throws the cloth)."""
        a, b = self.ties(I, J, loose, 0.0, 0.0)
        self._sew = (a, b, float(loose), float(firm), float(seconds))
        return a, b

    def flatten(self, A, B, C, D, rest, compliance):
        """Flattening stencils over each inner edge (A, B) with its two far corners (C, D): the edge's midpoint is
        kept at its rest distance from the corners' midpoint (on a flat pattern, about none). This resists a fold
        in proportion to its angle, where a tie between the far corners resists only as the angle squared and so
        lets a starched cloth hang like a soft one (the swatch strip: a bending length of 1.7 cm at any bend
        compliance, against 3.7 for starched longcloth). An addition to fabric-core.js's model."""
        keep = ~((self.w[A] == 0) & (self.w[B] == 0) & (self.w[C] == 0) & (self.w[D] == 0))
        rest = np.broadcast_to(np.asarray(rest, float), np.shape(A))
        comp = np.broadcast_to(np.asarray(compliance, float), np.shape(A))
        self._quads.append(tuple(np.asarray(v)[keep] for v in (A, B, C, D, rest, comp)))
        self._built = None

    def mesh(self, faces, stretch=None, shear=None, bend=None, rest_pos=None, bend_at=None, flat=None):
        """Tie a mesh's points as a sheet: its edges (the threads), its quads' diagonals (the shear), and across each
        edge the two points on either side of it (the bend ties). rest_pos: where the rest lengths are measured
        (the flat pattern, [n, 2] or [n, 3]); else the points as they are now. Of a triangle mesh also: bend_at, a
        compliance per point, the softer of a tie's two kept (a pressed fold); flat, a compliance for flattening
        stencils over the same edges (see flatten), with bend_at likewise. Returns (threads, diagonals, bend ties)."""
        s = self.settings
        if any(len(f) != 3 for f in faces):
            return self._mesh_any(faces, stretch, shear, bend, rest_pos)
        T = np.asarray(faces, np.int64)
        E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
        E, inv = np.unique(E, axis=0, return_inverse=True)
        te = inv.reshape(3, -1).T
        R = self.pos if rest_pos is None else np.asarray(rest_pos, float)
        self.ties(E[:, 0], E[:, 1], s['stretch'] if stretch is None else stretch, 0.0,
                  np.linalg.norm(R[E[:, 0]] - R[E[:, 1]], axis=1))
        # bend: the far corners across each inner edge
        order = np.argsort(te.ravel(), kind='stable')
        tri_of = np.repeat(np.arange(len(T)), 3)[order]
        e_sorted = te.ravel()[order]
        first = np.searchsorted(e_sorted, np.arange(len(E)))
        cnt = np.bincount(e_sorted, minlength=len(E))
        inner = np.nonzero(cnt == 2)[0]
        t0, t1 = tri_of[first[inner]], tri_of[first[inner] + 1]
        a, b = E[inner, 0], E[inner, 1]

        def far(t):
            tt = T[t]
            return np.where((tt[:, 0] != a) & (tt[:, 0] != b), tt[:, 0],
                            np.where((tt[:, 1] != a) & (tt[:, 1] != b), tt[:, 1], tt[:, 2]))
        c0, c1 = far(t0), far(t1)
        bc = np.full(len(c0), s['bend'] if bend is None else bend)
        if bend_at is not None:
            bend_at = np.asarray(bend_at, float)
            bc = np.maximum(bc, np.maximum(bend_at[c0], bend_at[c1]))
        self.ties(c0, c1, 0.0, 0.0, np.linalg.norm(R[c0] - R[c1], axis=1))
        self._comp[-1] = bc[~((self.w[c0] == 0) & (self.w[c1] == 0))]
        if flat is not None:
            fc = np.full(len(c0), float(flat))
            if bend_at is not None:
                fc = np.maximum(fc, np.maximum(bend_at[c0], bend_at[c1]))
            R3 = R if R.shape[1] == 3 else np.hstack([R, np.zeros((len(R), 1))])
            rest = np.linalg.norm(0.5 * (R3[a] + R3[b]) - 0.5 * (R3[c0] + R3[c1]), axis=1)
            self.flatten(a, b, c0, c1, rest, fc)
        return len(E), 0, len(c0)

    def _mesh_any(self, faces, stretch, shear, bend, rest_pos):
        """mesh() for faces of any size (the hose's quads): the same ties, walked face by face."""
        s = self.settings
        R = self.pos if rest_pos is None else np.asarray(rest_pos, float)
        edges, diag, opp = set(), set(), {}
        for f in faces:
            k = len(f)
            for a in range(k):
                u, v = f[a], f[(a + 1) % k]
                e = (min(u, v), max(u, v))
                edges.add(e)
                opp.setdefault(e, []).append(f[(a + 2) % k])
            if k == 4:
                diag.add((min(f[0], f[2]), max(f[0], f[2])))
                diag.add((min(f[1], f[3]), max(f[1], f[3])))

        def tie(pairs, comp):
            p = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
            if len(p):
                self.ties(p[:, 0], p[:, 1], comp, 0.0, np.linalg.norm(R[p[:, 0]] - R[p[:, 1]], axis=1))
            return len(p)
        ne = tie(edges, s['stretch'] if stretch is None else stretch)
        nd = tie(diag, s['shear'] if shear is None else shear)
        nb = tie({(o[0], o[1]) for o in opp.values() if len(o) == 2 and o[0] != o[1]}, s['bend'] if bend is None else bend)
        return ne, nd, nb

    def hold_chain(self, chain):
        chain = np.asarray(chain, np.int64)
        self.chains.append((chain, np.linalg.norm(np.diff(self.pos[chain], axis=0), axis=1)))

    def _build(self):
        I, J = np.concatenate(self._I), np.concatenate(self._J)
        self._built = dict(I=I, J=J, rest=np.concatenate(self._rest), comp=np.concatenate(self._comp),
                           slack=np.concatenate(self._slack), kind=np.concatenate(self._kind),
                           batches=_colour(I, J, self.n, self.w == 0) if self.settings.get('walk') == 'gauss-seidel'
                           else [])
        if self._quads:
            Q = [np.concatenate([q[k] for q in self._quads]) for k in range(6)]
            self._built['quads'] = Q
            self._built['qbatches'] = _colour(Q[0], Q[1], self.n, self.w == 0, more=(Q[2], Q[3]))

    # ------------------------------------------------------------------ steps

    def step(self, h, body=None, neutral=False):
        s = self.settings
        if self._built is None:
            self._build()
        B = self._built
        x, pv, w = self.pos, self.prev, self.w
        keep = (1.0 - s['damping']) ** (h * 60.0)
        live = w > 0
        v = (x[live] - pv[live]) * keep
        pv[live] = x[live]
        x[live] += v + np.asarray(s['gravity']) * h * h
        comp = B['comp']
        if self._sew is not None:
            a, b, lo, hi, secs = self._sew
            f = min(1.0, self.t / max(secs, 1e-9))
            comp = comp.copy()
            comp[a:b] = lo + (hi - lo) * f
        alpha = comp / (h * h)
        lam = np.zeros(len(B['I']))
        lamq = np.zeros(len(B['quads'][0])) if 'quads' in B else None
        I, J, rest, slack, kind = B['I'], B['J'], B['rest'], B['slack'], B['kind']
        nb = len(B['batches'])
        jac = s.get('walk', 'jacobi') == 'jacobi'
        if jac:
            cnt = np.maximum(np.bincount(np.concatenate([I, J]), minlength=self.n).astype(float), 1.0)
        for it in range(s['iterations']):
            if jac:
                self._jacobi_walk(I, J, rest, slack, kind, alpha, lam, cnt)
                if 'quads' in B:
                    self._flat_walk(B, h, lamq, back=bool(it & 1))
                continue
            for q in (range(nb) if not it & 1 else range(nb - 1, -1, -1)):
                k = B['batches'][q]
                i, j = I[k], J[k]
                d = x[j] - x[i]
                L = np.sqrt(np.einsum('ij,ij->i', d, d))
                tgt = rest[k]
                sl = slack[k]
                if sl.any():
                    tgt = np.where(sl > 0, np.minimum(tgt * (1 + sl), np.maximum(tgt * (1 - sl), L)), tgt)
                C = L - tgt
                kd = kind[k]
                if kd.any():
                    C = np.where(kd == 1, np.maximum(C, 0.0), C)
                wi, wj = w[i], w[j]
                al = alpha[k]
                den = wi + wj + al
                # (a tie between two held points: none is made by ties(), but a pin() after mesh() leaves them, and
                # 0 / 0 here turned the whole sheet to NaN)
                dl = np.where((L > 1e-9) & (den > 0), (-C - al * lam[k]) / np.where(den > 0, den, 1.0), 0.0)
                lam[k] += dl
                qv = (dl / np.maximum(L, 1e-9))[:, None] * d
                x[i] -= wi[:, None] * qv
                x[j] += wj[:, None] * qv
            if 'quads' in B:
                self._flat_walk(B, h, lamq, back=bool(it & 1))
        self.hold()
        if body is not None:
            p, touched = body(x.copy())
            touched &= live
            self.touch = touched
            t = np.nonzero(touched)[0]
            if len(t):
                push = p[t] - x[t]
                if neutral:
                    pv[t] += push
                mu = s['friction']
                nl = np.linalg.norm(push, axis=1)
                if mu:
                    ok = nl > 1e-12
                    u = np.where(ok[:, None], push / np.maximum(nl, 1e-12)[:, None], 0.0)
                    sv = p[t] - pv[t]
                    sv -= np.einsum('ij,ij->i', sv, u)[:, None] * u
                    sll = np.linalg.norm(sv, axis=1)
                    hold = mu * nl
                    kk = np.where(sll <= hold, 1.0, hold / np.maximum(sll, 1e-12))
                    kk = np.where(ok, kk, 0.0)
                    p[t] -= kk[:, None] * sv
                x[t] = p[t]
                if neutral:          # (and what was running into the body stops: no speed kept along the push)
                    u = push / np.maximum(nl, 1e-12)[:, None]
                    vn = np.einsum('ij,ij->i', x[t] - pv[t], u)
                    pv[t] += np.minimum(vn, 0.0)[:, None] * u
            if self.chains:
                self.hold(body, neutral)
        self.t += h

    def _jacobi_walk(self, I, J, rest, slack, kind, alpha, lam, cnt):
        """every tie at once, each point moved by the mean of its ties' moves, over-relaxed"""
        x, w = self.pos, self.w
        d = x[J] - x[I]
        L = np.sqrt(np.einsum('ij,ij->i', d, d))
        tgt = rest
        if slack.any():
            tgt = np.where(slack > 0, np.minimum(rest * (1 + slack), np.maximum(rest * (1 - slack), L)), rest)
        C = L - tgt
        if kind.any():
            C = np.where(kind == 1, np.maximum(C, 0.0), C)
        wi, wj = w[I], w[J]
        dl = np.where(L > 1e-9, (-C - alpha * lam) / np.maximum(wi + wj + alpha, 1e-12), 0.0)
        lam += dl
        q = (dl / np.maximum(L, 1e-9))[:, None] * d
        acc = np.zeros_like(x)
        for a in range(3):
            acc[:, a] = np.bincount(J, weights=wj * q[:, a], minlength=self.n) - np.bincount(I, weights=wi * q[:, a],
                                                                                          minlength=self.n)
        x += self.settings['relax'] * acc / cnt[:, None]

    def _flat_walk(self, B, h, lam, back=False):
        x, w = self.pos, self.w
        A_, B_, C_, D_, rest, comp = B['quads']
        nb = len(B['qbatches'])
        for q in (range(nb - 1, -1, -1) if back else range(nb)):
            k = B['qbatches'][q]
            a, b, c, d = A_[k], B_[k], C_[k], D_[k]
            v = 0.5 * (x[a] + x[b]) - 0.5 * (x[c] + x[d])
            L = np.sqrt(np.einsum('ij,ij->i', v, v))
            C = L - rest[k]
            al = comp[k] / (h * h)
            W = 0.25 * (w[a] + w[b] + w[c] + w[d])
            dl = (-C - al * lam[k]) / np.maximum(W + al, 1e-12)
            dl = np.where(L > 1e-9, dl, 0.0)
            lam[k] += dl
            nv = (dl / np.maximum(L, 1e-9))[:, None] * v * 0.5
            x[a] += w[a][:, None] * nv
            x[b] += w[b][:, None] * nv
            x[c] -= w[c][:, None] * nv
            x[d] -= w[d][:, None] * nv

    def hold(self, body=None, neutral=False, still=False):
        x, pv, w, lead = self.pos, self.prev, self.w, self.settings['lead']
        for ch, ln in self.chains:
            for k in range(1, len(ch)):
                i, b = ch[k], ch[k - 1]
                if w[i] == 0:
                    continue
                d = x[i] - x[b]
                L = np.linalg.norm(d) or 1.0
                p = x[b] + d * (ln[k - 1] / L)
                if still:
                    pv[i] += p - x[i]
                elif w[b] != 0:
                    pv[b] += lead * (p - x[i])
                if body is not None:
                    q, t = body(p[None].copy())
                    if t[0]:
                        if neutral:
                            pv[i] += q[0] - p
                        p = q[0]
                x[i] = p

    def settle(self, body=None, h=1.0 / 120, most=600, tol=1e-4, damping=0.35, log=None, every=60, max_steps=None):
        """Run to rest: steps until the fastest point moves less than tol a step (or max_steps), damped hard.
        Returns the steps taken."""
        s = self.settings
        was = s['damping']
        s['damping'] = damping
        k = 0
        t0 = time.time()
        for k in range(most if max_steps is None else max_steps):
            self.step(h, body, neutral=True)
            if k % 10 == 9:
                far = float(np.abs(self.pos - self.prev).max())
                if log and (k + 1) % every == 0:
                    log('fabric: step %d (%.0f s), fastest point %.4f a step' % (k + 1, time.time() - t0, far))
                if far < tol:
                    break
        s['damping'] = was
        self.prev[:] = self.pos
        return k + 1

    def shift(self, i, d):
        self.pos[i] += d
        self.prev[i] += d

    def strain(self, I, J, rest):
        return np.linalg.norm(self.pos[I] - self.pos[J], axis=1) / np.maximum(rest, 1e-9)


class MeshBody:
    """A body for Fabric.step: the points must stay gap off a surface (given as its points and outward normals,
    dense enough that the nearest point stands for the nearest surface) and above a floor.

    planes: how many nearby surface points are searched for a crease's second wall (1: the nearest only, as
    before; 8 is enough). In a crease, where two surfaces meet (the armpit: the torso's side and the arm's inner face, or the arm
    sunk a little into the side), the nearest point flips from one surface to the other between neighbouring cloth
    points, so neighbours are pushed in opposite directions, across the crease, and the cloth between them tears
    (solver-issues.md, S3). With planes > 1 each point is pushed out of all the nearby planes at once (cyclic
    projection onto both walls' planes, Dykstra's corrections), which in a crease is out along the crease's
    bisector: neighbours get nearly the same push. The second wall is the nearest sample facing more than 60
    degrees away from the nearest one's (turn) within `band` of its distance."""

    def __init__(self, pts, normals, gap=0.06, floor=None, floor_gap=0.03, reach=4.0, planes=1, band=None,
                 sweeps=6, turn=0.5, tangent=None):
        from scipy.spatial import cKDTree
        self.p = np.asarray(pts, float)
        n = np.asarray(normals, float)
        self.n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.tree = cKDTree(self.p)
        self.gap, self.floor, self.floor_gap, self.reach = gap, floor, floor_gap, reach
        self.planes, self.sweeps = int(planes), int(sweeps)
        self.band = 2.0 * gap + 0.05 if band is None else band
        self.turn = float(turn)                  # (cos of the angle past which a sample is another wall: 0.5, 60 deg)
        self.tangent = tangent                   # (an open sheet's edge: see _past_edge)

    def __call__(self, P):
        if self.planes > 1:
            return self._many(P)
        d, j = self.tree.query(P, k=1, distance_upper_bound=self.reach)
        near = np.isfinite(d)
        j = np.where(near, j, 0)
        nrm = self.n[j]
        depth = np.einsum('ij,ij->i', P - self.p[j], nrm)
        k = self._edge_weight(P, j, nrm, depth)
        hit = near & (depth < self.gap) & (k > 0)
        P[hit] += (k[hit] * (self.gap - depth[hit]))[:, None] * nrm[hit]
        return self._floor(P, hit)

    def _edge_weight(self, P, j, nrm, depth):
        """tangent: for a body that is an open sheet (cloth as a collider: a bodice under a collar). Its nearest
        sample's plane stands for the surface only over the sheet itself; past the sheet's edge it would carry on as
        an endless wall, and cloth in the air beyond the rim (a collar's stand over the neckline) is pushed out of
        that wall every step, held stretched and never relaxing (solver-issues.md, S5). With tangent set, the push
        is full while the point's foot on the nearest sample's plane lies within tangent of the sample, and fades to
        nothing at twice that (past the sheet's edge). A hard cut there made points on the line chatter in and out
        of the push every step. About 1.5 times the samples' spacing; None (off) for closed bodies."""
        if self.tangent is None:
            return np.ones(len(P))
        lat = np.linalg.norm(P - self.p[j] - depth[:, None] * nrm, axis=1)
        return np.clip(2.0 - lat / self.tangent, 0.0, 1.0)

    def _floor(self, P, hit):
        if self.floor is not None:
            fl = self.floor + self.floor_gap
            low = P[:, 2] < fl
            P[low, 2] = fl
            hit |= low
        return P, hit

    def _many(self, P):
        # Each point is kept off the nearest surface point's plane and, if there is one within band, off the nearest
        # point of a surface facing another way (normals more than `turn` apart: the other wall of a crease). Not off
        # every nearby plane: on a convex part (a limb's round) the point is outside if it is in front of ONE plane,
        # and in front of all of them puts it on a polygon's corner, further out. In a crease it must be outside
        # BOTH walls. The two are solved together (cyclic projection with Dykstra's corrections).
        k = self.planes
        d, j = self.tree.query(P, k=k, distance_upper_bound=self.reach)
        ok = np.isfinite(d)
        near = ok[:, 0]
        j = np.where(ok, j, 0)
        Nn = self.n[j]                                                   # [m, k, 3]
        other = ok & (np.einsum('mj,mkj->mk', Nn[:, 0], Nn) < self.turn) & (d <= d[:, :1] + self.band)
        has = other.any(1)
        j2 = j[np.arange(len(P)), np.argmax(other, 1)]
        Q = np.stack([self.p[j[:, 0]], self.p[j2]], 1)
        Nq = np.stack([Nn[:, 0], self.n[j2]], 1)
        ke = self._edge_weight(P, j[:, 0], Nn[:, 0], np.einsum('mj,mj->m', P - self.p[j[:, 0]], Nn[:, 0]))
        near &= ke > 0
        use = np.column_stack([near, near & has])
        off = np.einsum('mkj,mkj->mk', Q, Nq) + self.gap                 # (plane c: n . x >= off)
        X = P.copy()
        corr = np.zeros(Q.shape)
        hit = np.zeros(len(P), bool)
        for _ in range(self.sweeps if has.any() else 1):
            for c in range(2):
                y = X + corr[:, c]
                viol = off[:, c] - np.einsum('mj,mj->m', y, Nq[:, c])
                viol = np.where(use[:, c], np.maximum(viol, 0.0), 0.0)
                xn = y + viol[:, None] * Nq[:, c]
                corr[:, c] = y - xn
                X = xn
                hit |= viol > 0
        P[hit] = P[hit] + ke[hit, None] * (X[hit] - P[hit])
        return self._floor(P, hit)


def vertex_normals(V, F):
    """Area-weighted point normals of a mesh (faces as index lists), for a MeshBody."""
    V = np.asarray(V, float)
    N = np.zeros_like(V)
    for f in F:
        for k in range(1, len(f) - 1):
            a, b, c = f[0], f[k], f[k + 1]
            nrm = np.cross(V[b] - V[a], V[c] - V[a])
            N[a] += nrm
            N[b] += nrm
            N[c] += nrm
    return N


def bvh_body(tree, gap):
    """A body for Fabric.step from a mathutils BVHTree of a surface with its normals outward: points nearer it than
    gap, or behind it, are set out to gap in front of it."""
    def body(x):
        out = x.copy()
        touch = np.zeros(len(x), dtype=bool)
        for i in range(len(x)):
            loc, nrm, _, d = tree.find_nearest(x[i], 4.0 * gap + 0.5)
            if loc is None:
                continue
            side = (x[i, 0] - loc[0]) * nrm[0] + (x[i, 1] - loc[1]) * nrm[1] + (x[i, 2] - loc[2]) * nrm[2]
            if side < gap:
                out[i] = (loc[0] + nrm[0] * gap, loc[1] + nrm[1] * gap, loc[2] + nrm[2] * gap)
                touch[i] = True
        return out, touch
    return body
