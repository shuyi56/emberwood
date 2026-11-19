"""Fabric: the game's particle solver for cloth (src/world/fabric-core.js), for the Blender scripts. It folds and
settles sheets here as it does in the game; Blender's own cloth solver is kept for final renders and for structured
pieces such as the chemise.

The sheet is points held together by ties, each keeping its length: the threads (`stretch`), the diagonals
(`shear`), and ties two points apart (`bend`), so a sheet rolls over an edge in a curve instead of creasing at
every row. Seams are ties of no length between the two pieces' points. A tie's compliance (how soft it is) is
divided by the step squared, so the sheet is as stiff in small steps as in large ones (XPBD: Macklin, Mueller and
Chentanez 2016). After the ties each point is pushed out of the body, and the body's hold on it takes away part of
its slide along it (`friction`).

The game's solver walks the ties one by one (Gauss-Seidel). Here they are solved all at once and each point takes the
mean of the moves its ties ask of it (Jacobi, with over-relaxation `relax`): the same answer, walked in numpy, at the
cost of more walks a step (`iterations`). Units are the caller's; positions are an array [n, 3], z up."""
import numpy as np

FABRIC_DEFAULTS = dict(
    gravity=(0.0, 0.0, -375.0),   # units / s^2 (9.8 m/s^2 on a figure 63 units to 1.65 m)
    damping=0.02,                 # share of each point's speed lost per 1/60 s
    iterations=12,                # walks over the ties a step
    relax=1.6,                    # over-relaxation of the mean move
    stretch=0.0,                  # compliance of the threads (0: they keep their length)
    shear=2e-5,                   # of the diagonals
    bend=2e-4,                    # of the ties two points along
    friction=0.5,                 # how well the body holds what lies on it: the slide it stops, as a share of its push
)


class Fabric:
    def __init__(self, pos, settings=None):
        self.pos = np.array(pos, dtype=float)
        self.n = len(self.pos)
        self.prev = self.pos.copy()
        self.w = np.ones(self.n)                     # inverse mass: 0 pins a point
        self.s = dict(FABRIC_DEFAULTS, **(settings or {}))
        self.ci, self.cj, self.rest, self.comp = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                                                  np.zeros(0), np.zeros(0))

    def pin(self, idx, weight=None):
        """Hold the points idx where they are (weight 1), or partly (an inverse mass of 1 - weight)."""
        idx = np.asarray(idx, dtype=np.int64)
        self.w[idx] = 0.0 if weight is None else np.clip(1.0 - np.asarray(weight, dtype=float), 0.0, 1.0)

    def tie(self, i, j, compliance=0.0, rest=None):
        """Tie points i to points j (arrays) at the distance they are at now (or rest)."""
        i, j = np.asarray(i, dtype=np.int64).ravel(), np.asarray(j, dtype=np.int64).ravel()
        r = np.linalg.norm(self.pos[i] - self.pos[j], axis=1) if rest is None else np.broadcast_to(rest, i.shape)
        keep = ~((self.w[i] == 0) & (self.w[j] == 0)) & (i != j)
        self.ci = np.concatenate([self.ci, i[keep]])
        self.cj = np.concatenate([self.cj, j[keep]])
        self.rest = np.concatenate([self.rest, np.asarray(r, dtype=float)[keep]])
        self.comp = np.concatenate([self.comp, np.broadcast_to(np.asarray(compliance, dtype=float), i.shape)[keep]])

    def mesh(self, faces, stretch=None, shear=None, bend=None):
        """Tie a mesh's points as a sheet: its edges (the threads), its quads' diagonals, and across each edge the
        two points on either side of it (the bend ties), at the lengths they have now."""
        s = self.s
        edges, diag, opp = set(), set(), {}
        for f in faces:
            k = len(f)
            for a in range(k):
                u, v = f[a], f[(a + 1) % k]
                e = (min(u, v), max(u, v))
                edges.add(e)
                opp.setdefault(e, []).append(f[(a + 2) % k] if k == 3 else f[(a + 2) % k])
            if k == 4:
                diag.add((min(f[0], f[2]), max(f[0], f[2])))
                diag.add((min(f[1], f[3]), max(f[1], f[3])))
        e = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
        self.tie(e[:, 0], e[:, 1], s['stretch'] if stretch is None else stretch)
        if diag:
            d = np.array(sorted(diag), dtype=np.int64)
            self.tie(d[:, 0], d[:, 1], s['shear'] if shear is None else shear)
        b = np.array([(o[0], o[1]) for o in opp.values() if len(o) == 2 and o[0] != o[1]], dtype=np.int64).reshape(-1, 2)
        if len(b):
            self.tie(b[:, 0], b[:, 1], s['bend'] if bend is None else bend)
        return len(e), len(diag), len(b)

    def step(self, h, body=None, neutral=False):
        """One step of h seconds. body(points) -> (points pushed out of what they must stay out of, touched mask)."""
        s, x, pv, w = self.s, self.pos, self.prev, self.w
        free = w > 0
        keep = (1.0 - s['damping']) ** (h * 60.0)
        v = (x - pv) * keep
        pv[:] = x
        x[free] += v[free] + np.asarray(s['gravity']) * h * h
        ci, cj, rest = self.ci, self.cj, self.rest
        alpha = self.comp / (h * h)
        lam = np.zeros(len(ci))
        wi, wj = w[ci], w[cj]
        count = np.bincount(np.concatenate([ci, cj]), minlength=self.n).astype(float)
        count = np.maximum(count, 1.0)
        for _ in range(s['iterations']):
            d = x[cj] - x[ci]
            L = np.linalg.norm(d, axis=1)
            ok = L > 1e-9
            C = L - rest
            dl = np.where(ok, (-C - alpha * lam) / np.maximum(wi + wj + alpha, 1e-12), 0.0)
            lam += dl
            q = (dl / np.maximum(L, 1e-9))[:, None] * d
            acc = np.zeros_like(x)
            np.add.at(acc, ci, -wi[:, None] * q)
            np.add.at(acc, cj, wj[:, None] * q)
            x += s['relax'] * acc / count[:, None]
        if body is not None:
            p, touch = body(x)
            push = p - x
            if s['friction']:
                nl = np.linalg.norm(push, axis=1)
                u = push / np.maximum(nl, 1e-12)[:, None]
                sl = p - pv
                sl -= u * (sl * u).sum(1)[:, None]
                sn = np.linalg.norm(sl, axis=1)
                k = np.where(sn <= s['friction'] * nl, 1.0, s['friction'] * nl / np.maximum(sn, 1e-12))
                k = np.where(touch & free, k, 0.0)
                p = p - sl * k[:, None]
            if neutral:
                pv[touch] += (p - x)[touch]
            x[free] = p[free]

    def settle(self, body=None, h=1.0 / 120.0, most=600, tol=1e-4, damping=0.35):
        """Run to rest: steps until the fastest point moves less than tol a step (or most steps), its speed damped
        hard so it settles instead of swinging. Returns the steps taken."""
        was = self.s['damping']
        self.s['damping'] = damping
        k = 0
        for k in range(1, most + 1):
            self.step(h, body, neutral=True)
            if k % 10 == 0 and np.abs(self.pos - self.prev).max() < tol:
                break
        self.s['damping'] = was
        self.prev[:] = self.pos
        return k


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
