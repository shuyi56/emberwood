"""The groom (hair rebuild, foundations 3 to 6): real strands on the measured scalp, settled under their own weight by
a rod model, with a mass and volume budget, and a style written as operations on the natural fall.

    python3 tools/hair/groom.py HEAD.json.gz OUT_DIR [--guides 2400] [--style ranger]
    python3 tools/hair/groom.py --selftest OUT_DIR         the cantilever check of the rod model

Units: head heights (HH) for geometry, as scalp_field.py; SI inside the energy (1 HH = 0.219 m).

Foundation 3, the strand (one hair):
  [LS20] diameter 80 um at every scalp site (Leerunyakul & Suchonwanit 2020, PMC7035527).
  [A22]  Young's modulus 2.9 +/- 0.2 GPa for human hair (do Amaral 2022, Front Ecol Evol, PMC9197501, Table 1).
  [S20]  density 1.312 +/- 0.043 g/cm3 for human hair ("Porosity at different structural levels in human and yak belly hair", 2020, PMC7248950).
  [DLM07] curl: the geometric classification into eight groups (De la Mettrie et al. 2007, Hum Biol 79:265; method
         in PMC5453415): groups I-IV by the curve diameter of a 6 cm length, V-VIII by a 0.98 cm curl meter and the
         number of waves. The Ranger's hair is straight (group I): no intrinsic curvature. The CD cut-offs of groups
         I-IV are not in the open text, so curly groups wait for them.
  Cross-section taken round (ellipticity is what makes hair curl; group I is near round).
Foundation 4, the mechanics: a Kirchhoff rod, inextensible, bending energy EI/2 * integral of curvature squared,
  gravity, the head and the body as colliders, solved for rest (static equilibrium) by minimising the total energy.
  The straight, round rod's twist decouples from its centreline at rest, so twist is not carried for group I.
  The root's first segment is clamped along the follicle (the exit angle of scalp_field). Checked against the
  cantilever formula: a clamped strand held level sags w L^4 / (8 EI) at its tip (--selftest).
Foundation 5, the budget: the hair count from the scalp's density over its area; each guide stands for its share.
  A guide of n hairs has n times the bending stiffness and n times the weight (hairs side by side, free to slide),
  so it holds its shape over the same elastogravity length as one hair, (EI / w)^(1/3) = 4.5 cm.
  The hair's thickness off the scalp: the cross-section of the hairs that lie over a place, divided by the packing
  fraction of loose hair. PACKING is CALIBRATED to the approved creator Ranger's depth (about 10 mm over the back of
  the head), not measured: no open measurement of a loose hair mass's volume fraction was found.
  Friction between hairs is not modelled: no human hair-on-hair coefficient was found (do Amaral 2008, PMC2270361,
  notes there is none in the literature).
Foundation 6, the style: operations on the natural fall, in order. The Ranger (long straight fall):
  part (a centre part from the trichion back), comb (away from the part at right angles, then down), settle, cut
  (the Ranger's approved length line, the creator's hairEdge), settle again.
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF  # noqa: E402

HH_M = SF.HH_MM / 1000.0
G = 9.81
HAIR_E = 2.9e9                     # Pa [A22]
HAIR_D = SF.SHAFT_UM * 1e-6        # m [LS20]
HAIR_RHO = 1312.0                  # kg/m3 [S20]
HAIR_I = math.pi * HAIR_D ** 4 / 64
HAIR_EI = HAIR_E * HAIR_I          # N m2, one hair
HAIR_W = HAIR_RHO * math.pi * HAIR_D ** 2 / 4 * G     # N/m, one hair's weight per length
ELASTOGRAVITY_M = (HAIR_EI / HAIR_W) ** (1 / 3)
PACKING = 0.165                    # CALIBRATED: volume fraction of loose hair over the head (see above)
SEG_M = 0.01                       # m: the rod's segment length (well under the elastogravity length)


# ---------------------------------------------------------------------------------------------------- colliders
class Colliders:
    """The head (scalp_field.Surface's radius by direction, as a grid) and the bare mannequin's body (the creator's
    MANNEQUIN table, anatomy/mannequin-v14.blend), as differentiable depth functions for torch."""
    def __init__(self, surf, body):
        import torch
        self.torch = torch
        lat = np.linspace(-np.pi / 2, np.pi / 2, 181)
        lon = np.linspace(-np.pi, np.pi, 361)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        U = np.c_[(np.cos(LA) * np.sin(LO)).ravel(), (-np.cos(LA) * np.cos(LO)).ravel(), np.sin(LA).ravel()]
        R = surf.radius(U).reshape(LA.shape)
        self.R = torch.tensor(R, dtype=torch.float64)[None, None]
        self.centre = torch.tensor(SF.CENTRE)
        na = body['na']
        r = np.array(body['r'], float).reshape(-1, na) * 0.001
        r = np.c_[r, r[:, :1]]                                  # wrap the angle
        self.B = torch.tensor(r)[None, None]
        self.bz0, self.bdz, self.bnz = body['z0'], body['dz'], r.shape[0]

    def _grid(self, img, gx, gy):
        g = self.torch.stack([gx, gy], -1)[None, None]
        return self.torch.nn.functional.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)[0, 0, 0]

    def head_depth(self, X):
        t = self.torch
        q = X - self.centre
        r = t.linalg.norm(q, dim=-1)
        lat = t.asin(t.clamp(q[:, 2] / r, -1, 1))
        lon = t.atan2(q[:, 0], -q[:, 1])
        R = self._grid(self.R, lon / math.pi, lat / (math.pi / 2))
        d = t.where(X[:, 2] > 0.2, R - r, t.full_like(r, -1.0))
        return t.maximum(d, self.ear_depth(X))

    def ear_depth(self, X):
        """The ears, which the head export leaves out: an ellipsoid each side over the creator's ear (EAR.TOP 0.60,
        EAR.BOTTOM 0.365, depth EY), standing EAR_OUT off the head. EXPERIMENTAL shape (the creator's ear mesh is not
        exported yet)."""
        t = self.torch
        out = []
        for sx in (-1.0, 1.0):
            c = t.tensor([sx * (EAR_X + 0.5 * EAR_OUT), SF.EAR_CANAL[1], 0.5 * (SF.EAR_TOP + SF.EAR_BOTTOM)], dtype=X.dtype)
            ax = t.tensor(EAR_AXES, dtype=X.dtype)
            q = (X - c) / ax
            rr = t.linalg.norm(q, dim=-1)
            out.append((1 - rr) * ax.min())
        return t.maximum(out[0], out[1])

    def body_depth(self, X):
        t = self.torch
        th = t.atan2(X[:, 0] - SF.AXIS[0], -(X[:, 1] - SF.AXIS[1]))
        rr = t.hypot(X[:, 0] - SF.AXIS[0], X[:, 1] - SF.AXIS[1])
        fz = (self.bz0 - X[:, 2]) / self.bdz
        R = self._grid(self.B, th / math.pi, 2 * fz / (self.bnz - 1) - 1)
        return t.where((X[:, 2] < 0.2) & (X[:, 2] > -2.4), R - rr, t.full_like(rr, -1.0))


# ---------------------------------------------------------------------------------------------------- the rod
def rod_energy(torch, D, roots, d0, n_hairs, seg_hh, col, k_contact, mask=None, layer=None):
    """Total energy (J) of the guides. D: free segment directions (G, S-1, 3), unnormalised; d0: the clamped first
    segment along the follicle (G, 3). Positions are built by walking the unit directions, so the length is exact.
    mask (G, S): 1 for segments the guide still has after the cut, 0 past its end (they weigh and bend nothing).
    layer: the guide's layer offset off the head and body at each node (foundation 5), a function of the nodes."""
    d = torch.cat([d0[:, None], D], 1)
    d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
    X = roots[:, None] + torch.cumsum(d * seg_hh, 1)           # nodes 1..S (HH)
    if mask is None:
        mask = torch.ones(X.shape[:2], dtype=X.dtype)
    ds = seg_hh * HH_M
    n = n_hairs[:, None]
    cos = (d[:, 1:] * d[:, :-1]).sum(-1)
    E_b = (n * HAIR_EI / ds * (1 - cos) * mask[:, 1:]).sum()
    E_g = (n * HAIR_W * ds * X[..., 2] * HH_M * mask).sum()
    P = X.reshape(-1, 3)
    o = layer(X).reshape(-1) if layer is not None else 0.0
    pen = torch.relu(col.head_depth(P) + o) + torch.relu(col.body_depth(P) + o)
    E_c = (k_contact * (n * mask).reshape(-1) * (pen * HH_M) ** 2).sum()
    return E_b + E_g + E_c, X, (E_b.item(), E_g.item(), E_c.item())


def settle(roots, d0, D_init, n_hairs, seg_hh, col, iters=400, k_contact=(1e3, 1e4, 1e5), mask=None, layer=None,
           log=None):
    """Static equilibrium by minimising the energy (L-BFGS), the contact stiffened in steps (continuation) so the
    first steps can move hair out of the colliders without the penalty freezing the solve. Returns the settled
    nodes (G, S+1, 3) in HH and the directions."""
    import torch
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    R, d0t, nt = t(roots), t(d0), t(n_hairs)
    mk = t(mask) if mask is not None else None
    D = t(D_init).requires_grad_(True)
    scale = 1.0 / (HAIR_W * SEG_M * 1e-3 * max(1.0, float(np.mean(n_hairs))))   # energies of order 1 for the optimiser
    for k in np.atleast_1d(k_contact):
        opt = torch.optim.LBFGS([D], lr=1, max_iter=iters, history_size=30, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-12, tolerance_change=1e-15)

        def closure():
            opt.zero_grad()
            E, X, parts = rod_energy(torch, D, R, d0t, nt, seg_hh, col, k, mk, layer)
            (E * scale).backward()
            return E
        opt.step(closure)
        with torch.no_grad():
            E, X, parts = rod_energy(torch, D, R, d0t, nt, seg_hh, col, k, mk, layer)
        if log is not None:
            log.append(dict(k=float(k), E=E.item(), bend=parts[0], grav=parts[1], contact=parts[2]))
    X = X.numpy()
    return np.concatenate([roots[:, None], X], 1), D.detach().numpy()


def elastica_sag(L):
    """The exact (large-deflection) tip sag of an inextensible cantilever under its own weight, by shooting on
    theta'' = -(w/EI)(L-s) cos(theta), theta(0) = 0, theta'(L) = 0."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    f = lambda s, y: [y[1], -(HAIR_W / HAIR_EI) * (L - s) * math.cos(y[0]), math.sin(y[0])]
    end = lambda k0: solve_ivp(f, [0, L], [0, k0, 0], rtol=1e-10, atol=1e-12).y[1, -1]
    k0 = brentq(end, -1e3, 1e3)
    return abs(solve_ivp(f, [0, L], [0, k0, 0], rtol=1e-10, atol=1e-12).y[2, -1])


def selftest(out):
    """A strand clamped level at its root, settled: the tip's sag against w L^4 / (8 EI), small deflection."""
    import torch

    class NoCol:
        def head_depth(self, X):
            return torch.zeros(X.shape[0], dtype=torch.float64)
        body_depth = head_depth
    res = []
    for L_cm in (1.0, 2.0, 3.0, 4.0):
        S = 60
        seg_hh = L_cm / 100 / S / HH_M
        roots = np.array([[0.0, 0.0, 0.0]])
        d0 = np.array([[1.0, 0, 0]])
        D = np.tile([1.0, 0, 0], (1, S - 1, 1))
        X, _ = settle(roots, d0, D, np.array([1.0]), seg_hh, NoCol(), iters=10000, k_contact=(0.0,))
        sag = -X[0, -1, 2] * HH_M
        L = L_cm / 100
        theory = HAIR_W * L ** 4 / (8 * HAIR_EI)
        res.append(dict(L_cm=L_cm, sag_mm=round(sag * 1000, 4), theory_mm=round(theory * 1000, 4),
                        elastica_mm=round(elastica_sag(L) * 1000, 4),
                        ratio=round(sag / theory, 3), sag_over_L=round(sag / L, 4)))
    rep = dict(elastogravity_cm=round(ELASTOGRAVITY_M * 100, 2), EI_Nm2=HAIR_EI, w_N_per_m=HAIR_W, cantilever=res)
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, 'rod_selftest.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------------------------------------------- the style
# The Ranger's length line: the creator's hairEdge (main.js), the end height by angle round the head, as approved.
def ranger_cut_z(th):
    a = np.abs(th)
    side = -0.5 - 0.25 * SF.sm(1.0, 1.5, a)
    front = -0.5                                   # the face-framing front locks (the bangs come last: no fringe yet)
    e = front + (side - front) * SF.sm(0.86, 1.08, a)
    u = np.clip((a - 1.7) / (np.pi - 1.7), 0, 1)
    back = -1.42 - 0.1 * u ** 1.5
    return e + (back - e) * SF.sm(1.5, 2.1, a)


PART_BACK_Z = 0.97        # STYLE: the centre part runs from the trichion back to the top of the head
PART_HALF = 0.004         # STYLE: the part's half-width, HH (about 1 mm), as the knotted bun's pe(z) at the hairline


def comb_ranger(F, roots_idx):
    """part + comb: hair on the top and front is combed away from a centre part at right angles, then down; the rest
    keeps its natural growth direction (the back of the head and the nape fall as they grow)."""
    P, N, v = F['P'][roots_idx], F['N'][roots_idx], F['dir'][roots_idx]
    side = np.where(P[:, 0] >= 0, 1.0, -1.0)
    away = np.c_[side, np.zeros(len(P)), -0.35 * np.ones(len(P))]       # out from the part and a little down
    away -= N * (N * away).sum(1, keepdims=True)
    away /= np.linalg.norm(away, axis=1, keepdims=True)
    front = P[:, 1] < 0.25                                               # ahead of the crown
    top = SF.sm(0.6, 0.85, P[:, 2])
    w = top * front * (1 - SF.sm(0.15, 0.35, np.abs(P[:, 0])) * 0.6)
    out = v * (1 - w)[:, None] + away * w[:, None]
    # tuck behind the ear: hair rooted between the face-framing front and the ear is combed back and down over it
    th = SF.theta(P)
    tuck = SF.sm(SPLIT - 0.1, SPLIT + 0.1, np.abs(th)) * (1 - SF.sm(1.55, 1.8, np.abs(th))) * (1 - SF.sm(0.75, 0.9, P[:, 2]))
    back = np.c_[np.zeros(len(P)), np.ones(len(P)), -0.5 * np.ones(len(P))]
    back -= N * (N * back).sum(1, keepdims=True)
    back /= np.linalg.norm(back, axis=1, keepdims=True)
    out = out * (1 - tuck)[:, None] + back * tuck[:, None]
    # behind the crown the whorl's flow turns forward on one side; a long fall is brushed back there, so no hair
    # rooted behind the crown crosses it and drops over the face (two guides did before this)
    fwd = ~front & (out[:, 1] < 0.1)
    out[fwd, 1] = 0.1
    out[fwd] += back[fwd] * 0.3
    out -= N * (N * out).sum(1, keepdims=True)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    # roots inside the part itself are bare scalp
    keep = ~((np.abs(P[:, 0]) < PART_HALF) & (P[:, 2] > F['landmarks']['trichion'][2] - 0.01) & (P[:, 2] < PART_BACK_Z) & front)
    return out, keep


# ---------------------------------------------------------------------------------------------------- the groom
def pick_roots(F, n, seed=3):
    """n guide roots spread over the scalp by density (each guide then stands for an equal share of the hair)."""
    rng = np.random.default_rng(seed)
    per = SF.point_area(F)
    hairs = F['density'] * per * (SF.HH_MM / 10) ** 2              # hairs at each sample
    p = hairs / hairs.sum()
    # blue-noise-ish: oversample, then keep the farthest-apart by a greedy pass on a grid
    idx = rng.choice(len(p), size=n * 4, p=p)
    from scipy.spatial import cKDTree
    keep, t = [], None
    pts = F['P'][idx]
    order = rng.permutation(len(idx))
    rmin = math.sqrt((per * F['mask']).sum() / n) * 0.55
    grid = {}
    for k in order:
        q = pts[k]; c = tuple((q / rmin).astype(int))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        if np.linalg.norm(pts[j] - q) < rmin:
                            ok = False; break
        if ok:
            grid.setdefault(c, []).append(k); keep.append(k)
        if len(keep) >= n:
            break
    return idx[np.array(keep)], float(hairs.sum())


def follicle_dirs(F, idx, comb):
    """The first segment's direction: leaving the skin at the exit angle, along the (combed) growth direction."""
    N = F['N'][idx]
    a = np.radians(F['exit_deg'][idx])[:, None]
    return np.cos(a) * comb + np.sin(a) * N


def clear_follicles(roots, d0, N, seg_hh, col, layer, step_deg=5.0):
    """The first segment is clamped, so the solver cannot move its end. Where the skin curves away under a shallow
    exit angle (the sideburn in front of the ear), that end lands inside the head; raise the exit angle there, a few
    degrees at a time, until the end clears the skin and the layer above it."""
    import torch
    d0 = d0.copy()
    for _ in range(18):
        X = torch.tensor((roots + d0 * seg_hh)[:, None], dtype=torch.float64)
        o = layer(X).reshape(-1)
        bad = (col.head_depth(X.reshape(-1, 3)) + o > 0).numpy()
        if not bad.any():
            break
        a = np.radians(step_deg)
        t = d0[bad] - (d0[bad] * N[bad]).sum(1, keepdims=True) * N[bad]
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        ang = np.arcsin(np.clip((d0[bad] * N[bad]).sum(1), -1, 1))[:, None] + a
        d0[bad] = np.cos(ang) * t + np.sin(ang) * N[bad]
    return d0


EAR_X = 0.33                       # HH: the head's half-width at the ear (measured on the export: 0.326-0.343 at z .4-.5)
EAR_OUT = 0.08                     # HH: EXPERIMENTAL, how far the ear stands off the head (about 18 mm)
EAR_AXES = (0.5 * EAR_OUT + 0.005, 0.06, 0.5 * (SF.EAR_TOP - SF.EAR_BOTTOM))


def ear_depth_np(X):
    out = -np.ones(len(X))
    for sx in (-1.0, 1.0):
        c = np.array([sx * (EAR_X + 0.5 * EAR_OUT), SF.EAR_CANAL[1], 0.5 * (SF.EAR_TOP + SF.EAR_BOTTOM)])
        ax = np.array(EAR_AXES)
        out = np.maximum(out, (1 - np.linalg.norm((X - c) / ax, axis=1)) * ax.min())
    return out


LOOKAHEAD = 0.35                   # HH: the first guess moves hair to its side of the shoulder this far above it
SPLIT = 1.0                        # STYLE (Ranger): hair rooted in front of this angle frames the face and falls in front
                                   # of the shoulder; the rest is tucked behind the ears (the creator's Ranger)


def body_depth_np(body, X):
    """numpy twin of Colliders.body_depth (the creator's mannequinR)."""
    na = body['na']
    r = np.array(body['r'], float).reshape(-1, na) * 0.001
    th = np.arctan2(X[:, 0] - SF.AXIS[0], -(X[:, 1] - SF.AXIS[1]))
    rr = np.hypot(X[:, 0] - SF.AXIS[0], X[:, 1] - SF.AXIS[1])
    fz = np.clip((body['z0'] - X[:, 2]) / body['dz'], 0, r.shape[0] - 1.001)
    fa = (th + np.pi) / (2 * np.pi) * na
    r0, v = np.floor(fz).astype(int), fz - np.floor(fz)
    a0, u = np.floor(fa).astype(int) % na, fa - np.floor(fa)
    a1 = (a0 + 1) % na
    R = (r[r0, a0] * (1 - u) + r[r0, a1] * u) * (1 - v) + (r[r0 + 1, a0] * (1 - u) + r[r0 + 1, a1] * u) * v
    return np.where((X[:, 2] < 0.2) & (X[:, 2] > -2.4), R - rr, -1.0)


def body_y_extent(body, x, z, side, ny=321):
    """The body's front (side -1: least y) or back (side +1: greatest y) surface at each (x, z), from the mannequin
    table; nan where the line at that x misses the body."""
    ys = np.linspace(-0.9, 1.0, ny)
    G = len(x)
    Q = np.c_[np.repeat(x, ny), np.tile(ys, G), np.repeat(z, ny)]
    ins = (body_depth_np(body, Q) > 0).reshape(G, ny)
    any_ = ins.any(1)
    first = np.argmax(ins, 1)
    last = ny - 1 - np.argmax(ins[:, ::-1], 1)
    y = np.where(side < 0, ys[first], ys[last])
    return np.where(any_, y, np.nan)


def initial_paths(surf, body, roots, d0, comb, seg_hh, S, lift):
    """A first guess outside the colliders: out along the follicle, over the head along the comb while turning down
    (pushed out of the head, plus the layer lift), then plumb down; below the chin it is kept in front of the body
    for hair rooted in front of the ear and behind it for the rest (the creator's S.SPLIT), never laid out over the
    shoulder, where frictionless hair would slide down the arm (the cape fault in hair-building-guide.md)."""
    G = len(roots)
    side = np.where(np.abs(SF.theta(roots)) < SPLIT, -1.0, 1.0)
    X = np.zeros((G, S + 1, 3)); X[:, 0] = roots
    X[:, 1] = roots + d0 * seg_hh
    d = d0.copy()
    for k in range(2, S + 1):
        s = k * seg_hh
        turn = SF.sm(0.0, 0.3, s)
        want = comb * (1 - turn) + np.array([0, 0, -1.0]) * turn
        d = 0.6 * d + 0.4 * want
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        x = X[:, k - 1] + d * seg_hh
        for _ in range(3):
            q = x - SF.CENTRE
            r = np.linalg.norm(q, axis=1)
            hd = np.where(x[:, 2] > 0.2, surf.depth(x) + lift, -1.0)
            x = np.where((hd > 0)[:, None], x + q / r[:, None] * np.maximum(hd, 0)[:, None], x)
            ed = ear_depth_np(x) + lift
            out = np.c_[np.sign(x[:, 0]), np.zeros(G), np.zeros(G)]
            x = np.where((ed > 0)[:, None], x + out * np.maximum(ed, 0)[:, None] * 1.2, x)
        low = x[:, 2] < 0.2
        if low.any():
            # look ahead to the shoulder below, so the hair moves to its side as it falls instead of stepping over
            ye = np.fmax(body_y_extent(body, x[low, 0], x[low, 2], side[low]) * side[low],
                         body_y_extent(body, x[low, 0], x[low, 2] - LOOKAHEAD, side[low]) * side[low]) * side[low]
            y = x[low, 1]
            y = np.where(np.isnan(ye), y, np.where(side[low] < 0, np.minimum(y, ye - lift), np.maximum(y, ye + lift)))
            x[low, 1] = y
        dd = x - X[:, k - 1]
        d = dd / np.linalg.norm(dd, axis=1, keepdims=True)
        X[:, k] = X[:, k - 1] + d * seg_hh
    D = np.diff(X[:, 1:], axis=1)
    return D / np.linalg.norm(D, axis=-1, keepdims=True)


class Layers:
    """Foundation 5, the thickness of the hair under each guide. In each column round the head, the hair at height z
    is the hair rooted above z (it hangs down from its root); a guide rooted at z_r lies on the hairs rooted between z
    and z_r. Its distance off the head there is their cross-section over the column's width, divided by the packing
    fraction of loose hair. Below the lowest root it lies on the whole column under it."""
    NB, NZ, Z0, Z1 = 48, 64, 0.2, 1.05

    def __init__(self, roots, n_hairs):
        import torch
        self.torch = torch
        th = SF.theta(roots)
        self.col = ((th + np.pi) / (2 * np.pi) * self.NB).astype(int) % self.NB
        zg = np.linspace(self.Z0, self.Z1, self.NZ)
        area1 = math.pi * HAIR_D ** 2 / 4
        width_m = 2 * np.pi * 0.35 * HH_M / self.NB
        k = area1 / (PACKING * width_m) / HH_M                   # HH of thickness per hair in the column
        # cum[c, i]: thickness (HH) of the hair in column c rooted above zg[i]
        cum = np.zeros((self.NB, self.NZ))
        for c in range(self.NB):
            z = roots[self.col == c, 2]; n = n_hairs[self.col == c]
            cum[c] = [(n * (z > q)).sum() * k for q in zg]
        self.cum = cum
        self.root_cum = np.array([np.interp(z, zg, cum[c]) for z, c in zip(roots[:, 2], self.col)])
        self.T = torch.tensor(cum, dtype=torch.float64)[None, None]
        self.rc = torch.tensor(self.root_cum, dtype=torch.float64)
        self.cy = torch.tensor(2 * (self.col + 0.0) / (self.NB - 1) - 1, dtype=torch.float64)

    def __call__(self, X):
        t = self.torch
        G, S = X.shape[:2]
        gx = 2 * (X[..., 2] - self.Z0) / (self.Z1 - self.Z0) - 1
        gy = self.cy[:, None].expand(G, S)
        g = t.stack([gx, gy], -1).reshape(1, 1, -1, 2)
        under = t.nn.functional.grid_sample(self.T, g, mode='bilinear', padding_mode='border', align_corners=True)
        under = under.reshape(G, S)
        return t.relu(under - self.rc[:, None])          # the hair rooted between this node's height and the root

    def max_mm(self):
        return float(self.cum.max()) * SF.HH_MM


def build_groom(surf, F, body, n_guides=2400, style="ranger", iters=1500, log=None):
    idx, total = pick_roots(F, n_guides)
    comb, keep = comb_ranger(F, idx)
    keep &= ear_depth_np(F["P"][idx]) < -0.025          # no hair grows on the ear or the bare skin right round it
    idx, comb = idx[keep], comb[keep]
    n_hairs = np.full(len(idx), total / len(idx))
    roots = F['P'][idx] + F['N'][idx] * 0.0005
    d0 = follicle_dirs(F, idx, comb)
    seg_hh = SEG_M / HH_M
    layer = Layers(roots, n_hairs)
    # grow everything long enough for the cut, settle, cut on the settled hair, settle again
    need = roots[:, 2] - ranger_cut_z(SF.theta(roots)) + 0.6
    S = int(math.ceil(need.max() / seg_hh)) + 1
    lift0 = np.array([float(np.interp(z, np.linspace(layer.Z0, layer.Z1, layer.NZ), layer.cum[c])) for z, c in
                      zip(roots[:, 2], layer.col)])
    D = initial_paths(surf, body, roots, d0, comb, seg_hh, S, 0.5 * lift0.mean())
    col = Colliders(surf, body)
    d0 = clear_follicles(roots, d0, F['N'][idx], seg_hh, col, layer)
    t0 = time.time()
    X, D = settle(roots, d0, D, n_hairs, seg_hh, col, iters=iters, layer=layer, log=log)
    t1 = time.time()
    endz = ranger_cut_z(SF.theta(X[:, -1]))
    lengths = np.full(len(idx), S)
    for g in range(len(idx)):
        below = np.where(X[g, 1:, 2] < endz[g])[0]
        if len(below):
            lengths[g] = max(3, below[0])
    mask = (np.arange(S)[None] < lengths[:, None]).astype(float)
    X2, D = settle(roots, d0, D, n_hairs, seg_hh, col, iters=iters, layer=layer, mask=mask, log=log)
    t2 = time.time()
    X2[:, 1:][mask == 0] = np.nan
    return dict(roots=roots, idx=idx, X=X2, X_uncut=X, lengths=lengths, n_hairs=n_hairs, total=total,
                layer_max_mm=layer.max_mm(), seg_hh=seg_hh, d0=d0, comb=comb, times=(t1 - t0, t2 - t1))


def main():
    a = sys.argv[1:]
    if a and a[0] == '--selftest':
        selftest(a[1])
        return
    head, out = a[0], a[1]
    n = int(a[a.index('--guides') + 1]) if '--guides' in a else 2400
    import gzip
    os.makedirs(out, exist_ok=True)
    surf = SF.Surface(SF.load_head(head))
    F = SF.build(surf)
    body = json.load(gzip.open(head))['bodies']['mannequin']
    log = []
    Gm = build_groom(surf, F, body, n_guides=n, log=log)
    np.savez_compressed(os.path.join(out, 'groom.npz'), **{k: v for k, v in Gm.items() if isinstance(v, np.ndarray)})
    rep = dict(guides=int(len(Gm['idx'])), hairs=int(Gm['total']), hairs_per_guide=round(float(Gm['n_hairs'][0]), 1),
               elastogravity_cm=round(ELASTOGRAVITY_M * 100, 2), layer_max_mm=round(Gm['layer_max_mm'], 2),
               mean_length_cm=round(float(np.mean(Gm['lengths']) * SEG_M * 100), 1), solve_s=[round(x, 1) for x in Gm['times']],
               energy=log)
    json.dump(rep, open(os.path.join(out, 'groom.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
