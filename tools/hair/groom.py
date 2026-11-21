"""The groom (hair rebuild, foundations 3 to 6): real strands on the measured scalp, settled under their own weight by
a rod model, with a mass and volume budget, and a style written as operations on the natural fall.

    python3 tools/hair/groom.py HEAD.json.gz OUT_DIR [--guides 2400] [--style ranger] [--reuse OLD/groom.npz]
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
def rod_energy(torch, D, roots, d0, n_hairs, seg_hh, col, k_contact, mask=None, layer=None, kb0=None):
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
    if kb0 is None:
        e_b = 1 - cos
    else:
        # set hair (the bangs): bending measured from a rest curvature, with the discrete curvature binormal
        # kb = 2 d_(i-1) x d_i / (1 + d_(i-1).d_i) of [BER08]; |kb|^2 / 2 is 1 - cos to second order, so a guide with
        # no set (kb0 = 0) has the same energy as the others
        kb = 2 * torch.cross(d[:, :-1], d[:, 1:], dim=-1) / (1 + cos)[..., None]
        set_ = (kb0.abs().sum(-1) > 0).to(kb.dtype)
        e_b = set_ * 0.5 * ((kb - kb0) ** 2).sum(-1) + (1 - set_) * (1 - cos)
    E_b = (n * HAIR_EI / ds * e_b * mask[:, 1:]).sum()
    E_g = (n * HAIR_W * ds * X[..., 2] * HH_M * mask).sum()
    P = X.reshape(-1, 3)
    o = layer(X).reshape(-1) if layer is not None else 0.0
    pen = torch.relu(col.head_depth(P) + o) + torch.relu(col.body_depth(P) + o)
    E_c = (k_contact * (n * mask).reshape(-1) * (pen * HH_M) ** 2).sum()
    return E_b + E_g + E_c, X, (E_b.item(), E_g.item(), E_c.item())


def settle(roots, d0, D_init, n_hairs, seg_hh, col, iters=400, k_contact=(1e3, 1e4, 1e5), mask=None, layer=None,
           log=None, kb0=None):
    """Static equilibrium by minimising the energy (L-BFGS), the contact stiffened in steps (continuation) so the
    first steps can move hair out of the colliders without the penalty freezing the solve. Returns the settled
    nodes (G, S+1, 3) in HH and the directions."""
    import torch
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    R, d0t, nt = t(roots), t(d0), t(n_hairs)
    mk = t(mask) if mask is not None else None
    kb = t(kb0) if kb0 is not None else None
    D = t(D_init).requires_grad_(True)
    scale = 1.0 / (HAIR_W * SEG_M * 1e-3 * max(1.0, float(np.mean(n_hairs))))   # energies of order 1 for the optimiser
    for k in np.atleast_1d(k_contact):
        opt = torch.optim.LBFGS([D], lr=1, max_iter=iters, history_size=30, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-12, tolerance_change=1e-15)

        def closure():
            opt.zero_grad()
            E, X, parts = rod_energy(torch, D, R, d0t, nt, seg_hh, col, k, mk, layer, kb)
            (E * scale).backward()
            return E
        opt.step(closure)
        with torch.no_grad():
            E, X, parts = rod_energy(torch, D, R, d0t, nt, seg_hh, col, k, mk, layer, kb)
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


# ---------------------------------------------------------------------------------------------------- curtain bangs
# [BTC] behindthechair.com, "The cheekbone guide for cutting soft custom curtain bangs": a triangular section split
#       down the middle, overdirected across the face and cut to follow the cheekbone, opening the bridge of the nose,
#       finished with a round brush.
# [CC]  curlcentric.com, "How to cut curtain bangs": the section's base from the outer corner of one eye to the other
#       (never wider than the brows), its apex where the head begins to slope forward to the forehead; straight hair
#       cut to "cheekbone height or just below".
# [MBG] mindbodygreen.com, "How to style curtain bangs": 1 to 1.5 inches of hair each side of the part; the shortest
#       piece at or just below the cheekbones; round brush, dried away from the face.
# [OGL] ogleschool.edu, "A step-by-step guide for curtain bangs": the longest pieces no shorter than the chin; dried
#       from the roots for volume, each side away from the centre, the brush rolled under and outward.
BANG_EX = 0.213                    # HH: the outer canthus. Farkas' women: intercanthal 31.8 mm (the creator's alar
                                   # note) and a 30.7 mm eye fissure put the canthus 46.6 mm out; the fissure's centre,
                                   # 31.3 mm, sits on the creator's pupil (30.7 mm), so the two agree
BANG_DEPTH_MM = 60.0               # the section's depth on the part behind the hairline, from the ground truth: the
                                   # Face Creator's Ranger roots its curtain bangs 60 mm back along the part (at z 0.98,
                                   # y -0.125). [MBG]'s 1 to 1.5 in (25-38 mm) gave too little hair to cover the brow
                                   # ([CC]: start further back for fuller bangs)
BANG_GUIDE_X = 4                   # guide density in the bangs against the rest
Z_CHEEK = 0.415                    # HH: the cheekbone's height in the creator (main.js, the cheekbones' gaussian)
Z_CHIN_LOCK = 0.0                  # HH: the chin, [OGL]'s shortest length for the outer pieces
BANG_EXIT = 45.0                   # deg: the roots dried up off the forehead ([OGL] "from the roots for volume").
                                   # EXPERIMENTAL: no measured root angle after a blow-dry
BANG_SWEEP_M = 0.10               # m: the length over which the dried bang turns from the comb to its end direction,
                                   # about the brush's circumference (EXPERIMENTAL)
BANG_OUT_M = 0.04                  # m: the first move, out toward the temple (EXPERIMENTAL)
BRUSH_R_MM = 19.0                  # the round brush's radius (a 1.5 inch brush), EXPERIMENTAL: common size, unmeasured
SET_KEEP = 0.5                     # share of the brush's curvature the hair keeps once cool (the water / hydrogen-bond
                                   # set), EXPERIMENTAL
# [BER08] Bergou, Wardetzky, Robinson, Audoly & Grinspun 2008, "Discrete elastic rods": the curvature binormal.


def bang_section(F, idx, surf):
    """The bangs' roots: [CC]'s triangle, its base on the hairline between the outer canthi, its apex on the part.
    Returns (in the section, t: 0 at the part to 1 at the triangle's edge, the side -1 / +1)."""
    P = F['P'][idx]
    tr = F['landmarks']['trichion']
    # the apex: BANG_DEPTH_MM up the midline from the trichion, measured along the scalp
    el = np.radians(np.linspace(5, 85, 801))
    U = np.c_[np.zeros_like(el), -np.cos(el), np.sin(el)]
    Pm = surf.point(U)
    i0 = np.argmin(np.linalg.norm(Pm - tr, axis=1))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pm[i0:], axis=0), axis=1))] * SF.HH_MM
    apex = Pm[i0 + np.argmin(np.abs(arc - BANG_DEPTH_MM))]
    # elevation of each root seen from the skull's centre, on the face side
    elev = lambda X: np.arctan2(X[:, 2] - SF.CENTRE[2], -(X[:, 1] - SF.CENTRE[1]))
    e_h, e_a = elev(tr[None])[0], elev(apex[None])[0]
    f = np.clip((e_a - elev(P)) / (e_a - e_h), 0, None)          # 0 at the apex, 1 at the hairline (and below)
    half = BANG_EX * np.minimum(f, 1.0)
    inside = (np.abs(P[:, 0]) <= half) & (P[:, 1] < SF.CENTRE[1]) & (elev(P) <= e_a) & (F['mask'][idx] > 0.5)
    t = np.clip(np.abs(P[:, 0]) / BANG_EX, 0, 1)
    return inside, t, np.where(P[:, 0] >= 0, 1.0, -1.0), apex


def bang_comb(F, idx, t, side):
    """Dried each side away from the centre: forward over the brow and down, turned outward more the further the
    root sits from the part, so the curtain opens on the bridge of the nose ([BTC], [OGL])."""
    N = F['N'][idx]
    v = np.c_[side * (0.25 + 0.9 * t), -np.ones(len(t)), -0.6 * np.ones(len(t))]
    v -= N * (N * v).sum(1, keepdims=True)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def bang_cut_z(t):
    """Shortest beside the part at the cheekbone ([CC], [MBG]), longer outward, the outer edge at the chin ([OGL])."""
    return Z_CHEEK + (Z_CHIN_LOCK - Z_CHEEK) * t ** 1.3


def bang_sweep(t, side):
    """Where the dried bang points at its end: down the side of the face, a little out, to the cheekbone ([BTC],
    [OGL]); pointed further out it leaves the face and stands in the air."""
    v = np.c_[side * (0.25 + 0.15 * t), -0.1 * np.ones(len(t)), -np.ones(len(t))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


BANG_ROOT_LIFT_MM = 6.0            # the dried root's volume off the forehead, eased to the skin over the first 3 cm
                                   # (EXPERIMENTAL; [OGL] dries "from the roots for volume")
BANG_LIE_MM = 1.0                  # where the combed bang lies off the skin below that


def creator_field(style_path, front=True):
    """The Face Creator's own strands (its export) as a flow field to comb along: points, unit tangents, and the
    length of strand left from each point to its tip. front: only the strands rooted at the front of the head (the
    curtain bangs, their cover locks and the face-framing locks)."""
    import gzip
    d = json.load(gzip.open(style_path))
    P, T, R = [], [], []
    for st in d['strands']:
        p = np.array(st['p'], float).reshape(-1, 3)
        if front and not (p[0, 1] < -0.1 and p[0, 2] > 0.7):
            continue
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        rem = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]
        t = np.gradient(p, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        P.append(p); T.append(t); R.append(rem)
    from scipy.spatial import cKDTree
    P, T, R = np.concatenate(P), np.concatenate(T), np.concatenate(R)
    return dict(P=P, T=T, rem=R, tree=cKDTree(P))


TRACE_PULL = 0.5                   # how far each combed step is drawn onto the creator's strands (0 none, 1 onto them)
TRACE_K = 8
FRAME_TH = 0.9                     # rad: the face-framing hair combed with the creator's framing locks (in front of the
                                   # temples, |theta| under about 52 deg)
FRAME_PULL = 0.3
FRAME_ON_Z = 0.15                  # HH: the framing hair is laid on the face down to the jaw, then falls


def comb_bangs(surf, col, roots, d0, sweep, comb, seg_hh, S, field=None, out=None, pull=None, lie_hh=None):
    """The combing pass: the bangs are combed into their style and laid on the face, as a brush lays them, instead of
    being left to gravity (a lone hair holds a shape only over its 4.5 cm elastogravity length, so a solve lets a
    swept bang fall straight). Each step heads from the comb out toward the temple, then down beside the eye, and is
    laid on the skin at the root lift easing to BANG_LIE_MM."""
    import torch
    G = len(roots)
    X = np.zeros((G, S + 1, 3)); X[:, 0] = roots; X[:, 1] = roots + d0 * seg_hh
    d = d0.copy()
    side = np.sign(sweep[:, 0])
    if out is None:
        out = np.c_[side * 0.55, -0.2 * np.ones(G), -0.8 * np.ones(G)]  # (down over the brow's corner as much as out)
    pull = TRACE_PULL if pull is None else pull
    target = None
    if field is not None:                 # the length: what is left of the creator's nearest strand past the root
        target = field['rem'][field['tree'].query(roots)[1]]
    for k in range(2, S + 1):
        s_m = k * seg_hh * HH_M
        t1 = SF.sm(0.0, BANG_OUT_M, s_m); t2 = SF.sm(BANG_OUT_M, BANG_SWEEP_M, s_m)
        want = (comb * (1 - t1) + out * t1) * (1 - t2) + sweep * t2
        if field is not None:             # traced: along the creator's strands where the comb meets them
            dist, nb = field['tree'].query(X[:, k - 1], k=TRACE_K)
            w = 1.0 / np.maximum(dist, 1e-3) ** 2
            tf = (field['T'][nb] * w[..., None]).sum(1); tf /= np.maximum(np.linalg.norm(tf, axis=1, keepdims=True), 1e-9)
            near = (dist[:, 0] < 0.1)[:, None]
            want = np.where(near, tf, want)
        d = 0.5 * d + 0.5 * want
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        x = X[:, k - 1] + d * seg_hh
        if field is not None:
            dist, nb = field['tree'].query(x, k=TRACE_K)
            w = 1.0 / np.maximum(dist, 1e-3) ** 2
            pf = (field['P'][nb] * w[..., None]).sum(1) / w.sum(1)[:, None]
            x = x + pull * (pf - x) * (dist[:, :1] < 0.1)
        lie = (BANG_LIE_MM + (BANG_ROOT_LIFT_MM - BANG_LIE_MM) * (1 - SF.sm(0.0, 0.03, s_m))) / SF.HH_MM
        on = x[:, 2] > 0.2
        if lie_hh is not None:            # laid on the skin at its own depth in the lock (the face-framing hair: the
            lie = lie_hh                   # creator's framing locks lie on the cheek)
            on = x[:, 2] > FRAME_ON_Z
        elif field is not None:
            on[:] = False                  # (traced hair lies where the creator's does: only kept out of the head)
            pen = col.head_depth(torch.tensor(x, dtype=torch.float64)).numpy()
            q = x - SF.CENTRE; u = q / np.linalg.norm(q, axis=1, keepdims=True)
            x = x + u * np.maximum(pen + lie, 0)[:, None]
        if on.any():                      # laid on the skin: moved along the radius to the lie height, in or out,
            # then out of the collider where it stands over the skin (it bridges the eye socket, as a comb does)
            import torch
            q = x[on] - SF.CENTRE; u = q / np.linalg.norm(q, axis=1, keepdims=True)
            lo = lie[on] if np.ndim(lie) else lie
            x[on] = SF.CENTRE + u * (surf.radius(u) + lo)[:, None]
            pen = col.head_depth(torch.tensor(x[on], dtype=torch.float64)).numpy()
            x[on] = x[on] + u * np.maximum(pen + lo, 0)[:, None]
        dd = x - X[:, k - 1]
        d = dd / np.linalg.norm(dd, axis=1, keepdims=True)
        X[:, k] = X[:, k - 1] + d * seg_hh                  # (each step kept one segment long)
        on = X[:, k, 2] > 0.2
        if on.any():                      # (and never left inside: over the socket the step may stretch a little)
            q = X[on, k] - SF.CENTRE; u = q / np.linalg.norm(q, axis=1, keepdims=True)
            pen = col.head_depth(torch.tensor(X[on, k], dtype=torch.float64)).numpy()
            X[on, k] += u * np.maximum(pen + 0.2 / SF.HH_MM, 0)[:, None]
        low = X[:, k, 2] <= 0.2
        if low.any():                     # (below the chin: kept off the body, out along the level radius)
            q = X[low, k] - np.array([SF.AXIS[0], SF.AXIS[1], 0.0]); q[:, 2] = 0
            u = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
            pen = col.body_depth(torch.tensor(X[low, k], dtype=torch.float64)).numpy()
            X[low, k] += u * np.maximum(pen + 1.0 / SF.HH_MM, 0)[:, None]
    if field is not None:
        return X, target
    return X


def bang_set(X, bang):
    """The set: each bang is dried round the brush into the styled shape (the first guess: over the brow, round and
    out to the cheekbone); once cool it keeps SET_KEEP of that shape's curvature, never tighter than the brush.
    Returned as rest curvature binormals [BER08]."""
    kb0 = np.zeros((X.shape[0], X.shape[1] - 2, 3))
    cap = 2 * math.tan(0.5 * SEG_M / (BRUSH_R_MM / 1000.0))
    for g in np.where(bang)[0]:
        T = np.diff(X[g], axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)     # S segments, S - 1 joints
        c = (T[:-1] * T[1:]).sum(1)
        kb = 2 * np.cross(T[:-1], T[1:]) / (1 + c)[:, None] * SET_KEEP
        m = np.linalg.norm(kb, axis=1, keepdims=True)
        kb0[g] = kb * np.minimum(1, cap / np.maximum(m, 1e-12))
        kb0[g][m[:, 0] < 1e-9] = 1e-9                 # (marks the guide as set even where it is straight)
    return kb0


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
EAR_OUT = 0.035                    # HH: how far the ear stands off the head: the creator's EAR.EY (main.js), the
                                   # ground truth. 0.08 (EXPERIMENTAL before) held the side hair 13 mm wider than the creator's
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


def initial_paths(surf, body, roots, d0, comb, seg_hh, S, lift, sweep=None):
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
        if sweep is not None:              # styled hair (the bangs): from the comb round to its own end direction
            # (dried in two moves: out toward the temple over the first part of the length, then down beside the
            # eye, so the curtain opens on the face and still lies on it)
            sw = ~np.isnan(sweep[:, 0])
            L1 = BANG_OUT_M / HH_M
            t1 = SF.sm(0.0, L1, s); t2 = SF.sm(L1, BANG_SWEEP_M / HH_M, s)
            out = np.c_[np.sign(sweep[sw, 0]) * 0.85, -0.2 * np.ones(sw.sum()), -0.5 * np.ones(sw.sum())]
            want[sw] = (comb[sw] * (1 - t1) + out * t1) * (1 - t2[None] if np.ndim(t2) else 1 - t2) + sweep[sw] * t2
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

    def subset(self, sel):
        """The same layers for some of the guides (the per-guide column and root depth taken with them)."""
        import copy
        o = copy.copy(self)
        o.cy, o.rc = self.cy[self.torch.tensor(sel)], self.rc[self.torch.tensor(sel)]
        return o

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


def build_groom(surf, F, body, n_guides=2400, style="ranger", iters=1500, log=None, bangs=True, reuse=None, trace=None):
    """reuse: a previous groom.npz. The guides do not act on each other (no hair-hair contact; the layer is a fixed
    function of the roots), so a guide with the same root and the same style as in that groom settles to the same
    place: it is copied, and only the rest (here the bangs, while they are being styled) are solved."""
    idx, total = pick_roots(F, n_guides)
    comb, keep = comb_ranger(F, idx)
    keep &= ear_depth_np(F["P"][idx]) < -0.025          # no hair grows on the ear or the bare skin right round it
    idx, comb = idx[keep], comb[keep]
    n_hairs = np.full(len(idx), total / len(idx))
    bang = np.zeros(len(idx), bool)
    if bangs:
        # the bangs get BANG_GUIDE_X times the guide density (their shape is in fewer, shorter hairs), each guide
        # standing for proportionally fewer hairs, so the bangs' hair count is the scalp's
        b0 = bang_section(F, idx, surf)[0]
        allb = np.where(bang_section(F, np.arange(len(F['P'])), surf)[0])[0]
        allb = allb[~np.isin(allb, idx)]
        allb = allb[comb_ranger(F, allb)[1]]                    # (not in the part itself)
        extra = np.random.default_rng(11).choice(allb, size=min(len(allb), (BANG_GUIDE_X - 1) * int(b0.sum())), replace=False)
        hb = b0.sum() * total / len(idx)
        idx = np.r_[idx, extra]
        comb = np.r_[comb, comb_ranger(F, extra)[0]]
        bang = bang_section(F, idx, surf)[0]
        n_hairs = np.full(len(idx), total / (len(idx) - (len(extra))))
        n_hairs[bang] = hb / bang.sum()
    roots = F['P'][idx] + F['N'][idx] * 0.0005
    _, bt, bside, _ = bang_section(F, idx, surf)
    if bangs:
        comb[bang] = bang_comb(F, idx[bang], bt[bang], bside[bang])
    d0 = follicle_dirs(F, idx, comb)
    if bangs:
        a = math.radians(BANG_EXIT)
        d0[bang] = math.cos(a) * comb[bang] + math.sin(a) * F['N'][idx[bang]]
    seg_hh = SEG_M / HH_M
    layer = Layers(roots, n_hairs)
    # grow everything long enough for the cut, settle, cut on the settled hair, settle again
    need = roots[:, 2] - ranger_cut_z(SF.theta(roots)) + 0.6
    S = int(math.ceil(need.max() / seg_hh)) + 1
    lift0 = np.array([float(np.interp(z, np.linspace(layer.Z0, layer.Z1, layer.NZ), layer.cum[c])) for z, c in
                      zip(roots[:, 2], layer.col)])
    sweep = None
    if bangs:
        sweep = np.full((len(idx), 3), np.nan)
        sweep[bang] = bang_sweep(bt[bang], bside[bang])
    D = initial_paths(surf, body, roots, d0, comb, seg_hh, S, 0.5 * lift0.mean(), sweep)
    col = Colliders(surf, body)
    d0 = clear_follicles(roots, d0, F['N'][idx], seg_hh, col, layer)
    kb0 = None
    if bangs:
        dd = np.concatenate([d0[:, None], D], 1)
        X0 = np.concatenate([roots[:, None], roots[:, None] + np.cumsum(dd * seg_hh, 1)], 1)
        kb0 = bang_set(X0, bang)
    solve = np.ones(len(idx), bool)
    frame = np.zeros(len(idx), bool)
    same = False
    if reuse is not None:
        from scipy.spatial import cKDTree
        P0 = np.load(reuse)
        same = P0['X_uncut'].shape[1] == S + 1
        if same:                          # guides matched by their roots (the bang section may have grown or shrunk)
            dist, j = cKDTree(P0['roots']).query(roots)
            hit = dist < 1e-9
            solve = bang | ~hit
            P = {k: np.zeros((len(idx),) + P0[k].shape[1:], P0[k].dtype) for k in ('X_uncut', 'X', 'lengths')}
            for k in P:
                P[k][hit] = P0[k][j[hit]]
        print('reuse: %d of %d guides copied' % ((~solve).sum(), len(idx)) if same else 'reuse: groom differs, full solve')

    def settle_some(D, mask=None):
        X = np.zeros((len(idx), S + 1, 3)); Dn = D.copy()
        if (~solve).any():
            X[~solve] = P['X_uncut'][~solve] if mask is None else np.nan_to_num(P['X'][~solve])
        if not solve.any():
            return X, Dn
        m = None if mask is None else mask[solve]
        Xs, Ds = settle(roots[solve], d0[solve], D[solve], n_hairs[solve], seg_hh, col, iters=iters, layer=layer.subset(solve),
                        log=log, kb0=None if kb0 is None else kb0[solve], mask=m)
        X[solve] = Xs; Dn[solve] = Ds
        return X, Dn
    if bangs:
        # the bangs are combed, not solved: their paths come from the combing pass and are never settled
        field = creator_field(trace) if trace else None
        Xb = comb_bangs(surf, col, roots[bang], d0[bang], sweep[bang], comb[bang], seg_hh, S, field)
        if field is not None:
            Xb, bang_len = Xb
        solve &= ~bang
        if not same:
            P = {'X_uncut': np.zeros((len(idx), S + 1, 3)), 'X': np.zeros((len(idx), S + 1, 3)),
                 'lengths': np.zeros(len(idx), int)}
        P = {k: np.array(P[k]) for k in ('X_uncut', 'X', 'lengths')}
        P['X_uncut'][bang] = Xb
        if field is not None:
            # the face-framing hair (in front of the temples) is combed too, along the creator's framing locks onto
            # the cheek, then straight down in front of the shoulder; cut by the style's line like the rest
            thr = SF.theta(roots)
            frame = ~bang & (np.abs(thr) < FRAME_TH) & (roots[:, 2] > 0.55)
            down = np.tile([0.0, 0.0, -1.0], (frame.sum(), 1))
            lie_f = np.clip(layer.root_cum[frame], 1.0 / SF.HH_MM, 10.0 / SF.HH_MM)
            Xf, _ = comb_bangs(surf, col, roots[frame], d0[frame], np.c_[np.sign(roots[frame, 0]) * 1e-3, down[:, 1:]],
                           comb[frame], seg_hh, S, field, out=down, pull=FRAME_PULL, lie_hh=lie_f)
            solve &= ~frame
            P['X_uncut'][frame] = Xf
    t0 = time.time()
    X, D = settle_some(D)
    if (~solve).any():
        Xu = P['X_uncut'][~solve]
        dd = np.diff(Xu[:, 1:], axis=1)
        D[~solve] = dd / np.linalg.norm(dd, axis=-1, keepdims=True)
    t1 = time.time()
    endz = ranger_cut_z(SF.theta(X[:, -1]))
    if bangs:
        endz[bang] = bang_cut_z(bt[bang])
    trace_len = None
    if bangs and trace:
        trace_len = np.maximum(3, np.round(bang_len * HH_M / SEG_M)).astype(int)
    lengths = np.full(len(idx), S)
    for g in range(len(idx)):
        below = np.where(X[g, 1:, 2] < endz[g])[0]
        if len(below):
            lengths[g] = max(3, below[0])
    mask = (np.arange(S)[None] < lengths[:, None]).astype(float)
    if (~solve).any():
        reused = ~solve & ~bang & ~frame
        lengths[reused] = P['lengths'][reused]
        mask = (np.arange(S)[None] < lengths[:, None]).astype(float)
    if trace_len is not None:             # traced bangs end where the creator's do
        lengths[bang] = trace_len
        mask = (np.arange(S)[None] < lengths[:, None]).astype(float)
    if bangs:
        P['X'][bang] = Xb                 # (the combed bangs, cut below by the mask)
        if frame.any():
            P['X'][frame] = P['X_uncut'][frame]
    X2, D = settle_some(D, mask)
    t2 = time.time()
    X2[:, 1:][mask == 0] = np.nan
    # the locks the children blend within: 0 the tucked back, 1 / 2 the face-framing hair right / left of the part,
    # 3 / 4 the bangs right / left
    th = SF.theta(roots)
    lock = np.where(np.abs(th) < SPLIT, np.where(roots[:, 0] < SF.AXIS[0], 1, 2), 0)
    lock[bang] = np.where(roots[bang, 0] < SF.AXIS[0], 3, 4)
    return dict(roots=roots, idx=idx, X=X2, lock=lock, bang=bang.astype(np.int8), X_uncut=X, lengths=lengths, n_hairs=n_hairs, total=total,
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
    reuse = a[a.index('--reuse') + 1] if '--reuse' in a else None
    trace = a[a.index('--trace') + 1] if '--trace' in a else None
    Gm = build_groom(surf, F, body, n_guides=n, log=log, reuse=reuse, trace=trace)
    np.savez_compressed(os.path.join(out, 'groom.npz'), **{k: v for k, v in Gm.items() if isinstance(v, np.ndarray)})
    rep = dict(guides=int(len(Gm['idx'])), hairs=int(Gm['total']), hairs_per_guide=round(float(Gm['n_hairs'][0]), 1),
               elastogravity_cm=round(ELASTOGRAVITY_M * 100, 2), layer_max_mm=round(Gm['layer_max_mm'], 2),
               mean_length_cm=round(float(np.mean(Gm['lengths']) * SEG_M * 100), 1), solve_s=[round(x, 1) for x in Gm['times']],
               energy=log)
    json.dump(rep, open(os.path.join(out, 'groom.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
