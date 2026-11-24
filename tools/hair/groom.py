"""The groom (hair rebuild, foundations 3 to 6): real strands on the measured scalp, settled under their own weight by
a rod model, with a mass and volume budget, and a style written as operations on the natural fall.

    python3 tools/hair/groom.py HEAD.json.gz OUT_DIR [--guides 2400] [--style ranger|warden] [--reuse OLD/groom.npz]
        [--trace CREATOR_STYLE.json.gz] [--body ranger|warden|mannequin]   (default: the style's own character)      (warden: the natural wave, cut on the creator's length line; needs --trace)
    python3 tools/hair/groom.py --selftest OUT_DIR         the cantilever check of the rod model

Units: head heights (HH) for geometry, as scalp_field.py; SI inside the energy (1 HH = 0.219 m).

Foundation 3, the strand (one hair):
  [LS20] diameter 80 um at every scalp site (Leerunyakul & Suchonwanit 2020, PMC7035527).
  [A22]  Young's modulus 2.9 +/- 0.2 GPa for human hair (do Amaral 2022, Front Ecol Evol, PMC9197501, Table 1).
  [S20]  density 1.312 +/- 0.043 g/cm3 for human hair ("Porosity at different structural levels in human and yak belly hair", 2020, PMC7248950).
  [DLM07] curl: the geometric classification into eight groups (De la Mettrie et al. 2007, Hum Biol 79:265; method
         in PMC5453415): groups I-IV by the curve diameter of a 6 cm length, V-VIII by a 0.98 cm curl meter and the
         number of waves. The Ranger's hair is straight (group I): no intrinsic curvature. The CD cut-offs of groups
         I-IV are not in the open text, so curly groups wait for them. The Warden's wave is taken from the creator
         instead (see "natural waves" below): a rest curvature solved under gravity.
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
def rod_energy(torch, D, roots, d0, n_hairs, seg_hh, col, k_contact, mask=None, layer=None, kb0=None, pin=None):
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
    if pin is not None:
        # a styled hold (the hair set behind the ear, laid in at the temple): each node drawn to its styled place
        # with stiffness k per hair, weighted w (G, S) (pin = (targets (G, S, 3), w, k))
        Xt, w, k = pin
        E_c = E_c + (k * n * w * mask * (((X - Xt) * HH_M) ** 2).sum(-1)).sum()
    return E_b + E_g + E_c, X, (E_b.item(), E_g.item(), E_c.item())


def settle(roots, d0, D_init, n_hairs, seg_hh, col, iters=400, k_contact=(1e3, 1e4, 1e5), mask=None, layer=None,
           log=None, kb0=None, pin=None):
    """Static equilibrium by minimising the energy (L-BFGS), the contact stiffened in steps (continuation) so the
    first steps can move hair out of the colliders without the penalty freezing the solve. Returns the settled
    nodes (G, S+1, 3) in HH and the directions."""
    import torch
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    R, d0t, nt = t(roots), t(d0), t(n_hairs)
    mk = t(mask) if mask is not None else None
    kb = t(kb0) if kb0 is not None else None
    D = t(D_init).requires_grad_(True)
    pn = None if pin is None else (t(pin[0]), t(pin[1]), float(pin[2]))
    scale = 1.0 / (HAIR_W * SEG_M * 1e-3 * max(1.0, float(np.mean(n_hairs))))   # energies of order 1 for the optimiser
    for k in np.atleast_1d(k_contact):
        opt = torch.optim.LBFGS([D], lr=1, max_iter=iters, history_size=30, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-12, tolerance_change=1e-15)

        def closure():
            opt.zero_grad()
            E, X, parts = rod_energy(torch, D, R, d0t, nt, seg_hh, col, k, mk, layer, kb, pn)
            (E * scale).backward()
            return E
        opt.step(closure)
        with torch.no_grad():
            E, X, parts = rod_energy(torch, D, R, d0t, nt, seg_hh, col, k, mk, layer, kb, pn)
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
RANGER_LEN_Z = -1.2      # HH: the Ranger's one length, sides and back: the creator's back length (creator_cut, z -1.2).
                         # STYLE: John, 2026-11-23, "get rid of side cut": the sides (cut at -0.5 to -0.75, the
                         # approved line) now fall to the back's length, past the shoulders, where ends that stop on the
                         # shoulder flip out [CC2][BRIT]. Overrides the creator's shorter, layered sides.


def ranger_cut_z(th):
    a = np.abs(th)
    front = -0.5                                   # the face-framing front locks (the bangs come last: no fringe yet)
    return front + (RANGER_LEN_Z - front) * SF.sm(0.86, 1.3, a)


PART_BACK_Z = 0.97        # STYLE: the centre part runs from the trichion back to the top of the head
PART_HALF = 0.004         # STYLE: the part's half-width, HH (about 1 mm), as the knotted bun's pe(z) at the hairline
PART_END_Y = 0.02          # HH: where the side combing away from the part ends: the top of the head. The creator
                          # (wavyFull depth) thins its cap for the part only on the face side, up to the vertex (y 0);
                          # PART_BACK_Z alone ran it on to y 0.18, over the crown


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
BANG_DEPTH_MM = 90.0               # the section's depth on the part behind the hairline. The Face Creator roots its
                                   # curtain bangs 60 mm back along the part (at z 0.98, y -0.125), which gave 2,970
                                   # hairs and read as thin (John); 90 mm gives 4,900 (EXPERIMENTAL: deeper adds no
                                   # more, the section reaches the end of the part). [MBG]'s 1 to 1.5 in (25-38 mm)
                                   # gave too little hair to cover the brow ([CC]: start further back for fuller bangs)
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
FRAME_OUTER = 0.4                  # the outer share of the creator's bangs the Warden's framing hair is laid along
FRAME_PULL = 0.3
FRAME_ON_Z = 0.15                  # HH: the framing hair is laid on the face down to the jaw, then falls


TRACE_JOIN_M = 0.03                # m: a traced bang leaves its own root and joins its creator strand over this length


def creator_bangs(style_path):
    """The creator's curtain bangs: its strands rooted on the part at the front of the head (all from the part's
    front, z > 0.95), each side ordered across the fan by how far out it lies at the brow (z 0.85)."""
    import gzip
    import gzip
    d = json.load(gzip.open(style_path))
    out = {-1.0: [], 1.0: []}
    for st in d['strands']:
        p = np.array(st['p'], float).reshape(-1, 3)
        if not (p[0, 1] < -0.1 and p[0, 2] > 0.95):
            continue
        k = np.argmin(np.abs(p[:, 2] - 0.85))
        if p[:, 2].min() > 0.84:
            continue
        out[1.0 if p[k, 0] >= 0 else -1.0].append((abs(p[k, 0]), p))
    return {s: [q for _, q in sorted(v, key=lambda a: a[0])] for s, v in out.items()}


def resample(p, seg_hh, n):
    """A polyline at the rod's segment length, n nodes, carried on straight down past its end (as it would fall: a
    strand that ends turning out, carried on along its last tangent, shot out sideways); and its length (HH)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    u = np.arange(n) * seg_hh
    q = np.stack([np.interp(u, s, p[:, k]) for k in range(3)], 1)
    past = u > s[-1]
    if past.any():
        q[past] = p[-1] + (u[past] - s[-1])[:, None] * np.array([0.0, 0.0, -1.0])
    return q, s[-1]


def trace_bangs(roots, t, side, bangs, seg_hh, S, rng_seed=17):
    """Each bang laid along the creator's own bangs, in order across the fan: the guides of a side ranked by how far
    from the part they are rooted, the creator's strands ranked by how far out they lie at the brow, and each guide
    following the blend of the two strands at its rank from where it passes nearest the root. It leaves its own root
    and joins that path over TRACE_JOIN_M, so the curtain fans out as evenly as the creator's (pulling each bang onto its nearest creator
    strand gathered many onto the same few and left gaps between them). Returns the paths and their lengths."""
    X = np.zeros((len(roots), S + 1, 3)); L = np.zeros(len(roots))
    rng = np.random.default_rng(rng_seed)
    s = np.arange(S + 1) * seg_hh
    w = np.clip(1 - s * HH_M / TRACE_JOIN_M, 0, 1)
    w = w * w * (3 - 2 * w)
    for sd in (-1.0, 1.0):
        gi = np.where(side == sd)[0]
        if not len(gi):
            continue
        n2 = 2 * S + 2
        C = [resample(p, seg_hh, n2) for p in bangs[sd]]
        order = gi[np.argsort(t[gi] + 0.02 * rng.random(len(gi)))]
        q = (np.arange(len(order)) + 0.5) / len(order) * (len(C) - 1)
        for g, qq in zip(order, q):
            a = int(np.floor(qq)); b = min(a + 1, len(C) - 1); f = qq - a
            path = (1 - f) * C[a][0] + f * C[b][0]
            # the bang takes up its strand where the strand passes nearest its root (a root out on the hairline joins
            # its strand part way down, one at the part joins it at the start), the rest of the strand from there
            k0 = int(np.argmin(np.linalg.norm(path[:S] - roots[g], axis=1)))
            # a root back on the part first lies forward over the scalp to where its strand starts, at the rod's
            # segment length (laid on the scalp by push_out afterwards), then follows the strand
            gap = np.linalg.norm(path[k0] - roots[g])
            npre = int(np.ceil(gap / seg_hh))
            pre = roots[g] + (path[k0] - roots[g]) * (np.arange(npre) / max(npre, 1))[:, None]
            full = np.concatenate([pre, path[k0:]], 0)[:S + 1]
            X[g] = full + (roots[g] - full[0])[None] * w[:, None]
            L[g] = max(3 * seg_hh, (1 - f) * C[a][1] + f * C[b][1] - k0 * seg_hh + npre * seg_hh)
    return X, L


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


# ---------------------------------------------------------------------------------------------------- natural waves
# The Warden's waves (the creator's wavyFull, main.js wavyMass) as the hair's own shape: a natural wave is in the fibre
# (its rest curvature), not set by a brush, so unlike the bangs it is kept in the rod and solved under gravity.
# The weight of the hair below a point pulls the wave out there. For a small wave of wavenumber k = 2 pi / lambda
# under tension T, the rod's balance B (y - y0)'''' = T y'' gives the settled amplitude
#     A = A0 / (1 + T / (B k^2)),  T = w h   (h: the length of hair hanging below)
# so a wave is half pulled out where h = Lg^3 k^2 (Lg the elastogravity length, 4.48 cm): 17.6 cm for a 142 mm wave.
# The root of a long wavy fall hangs straighter and the ends wave, as the creator draws it (its wave grows in below
# z 0.3, sm(0.3, -0.05, z)). Checked on a free-hanging rod: 2.4 / 3.5 / 5.2 mm settled against 2.6 / 3.7 / 5.4 mm
# from the formula at 50 / 30 / 15 cm of hang (10 mm rest wave).
# The solver: L-BFGS crawls from either the rest shape or the straight fall (both stall well above the minimum: the
# tension mode is stiff), so it starts from the formula's shape, the lowest energy of the three by far.
WAVE_LAM0, WAVE_LAM_STEP = 0.5, 0.08   # HH: the creator's wavelength by section round the head (wavyMass: 0.5 + 0.08 sec)
WAVE_PHASE = (0.0, 1.9, 3.6, 5.1, 0.8)  # rad: the creator's phase by section (+0.6 on its left side), wavyMass
WAVE_SEC = 0.75                    # rad: the creator's section width round the head, sec = floor((|th| + 0.2) / 0.75)
WAVE_LOOSE = 0.6                   # rad: each lock's own phase jitter within its section (the creator's 0.6 r)
WAVE_RADIAL = 0.85                 # the wave's out-of-scalp axis against its sideways axis (the creator's 0.034 / 0.04)
WAVE_Z_ON = 0.3                    # HH: the wave's phase counted from where the hair crosses this height (the creator's
                                   # depth = 0.3 - z), so its crests fall where the creator's do
WAVE_Z_RAMP = (0.3, -0.05)         # HH: the rest wave grows in down the fall between these heights, the creator's
                                   # W = sm(0.3, -0.05, z) (its top is sleek: the crown and the sides above the jaw lie
                                   # combed flat). Laid by the straight fall's height, so it is fixed along each hair.
WAVE_A_MM = 30.0                   # the rest wave's amplitude, CALIBRATED: the settled wave below the shoulders matches
                                   # the creator's (about 11 mm, measured on its strands as on the groom's)
WAVE_END_JIT = 0.2                 # HH: the wave locks' ends spread up from the creator's longest (its lock tips, measured:
                                   # 0.2 HH between its longest and its median ends over the back)
WAVE_ITERS = 300                   # L-BFGS steps per contact stage for the wave settle
WAVE_LOCKS = 88                    # the wave locks (one phase each, their children blended only within them): the
                                   # creator's wavyFull has 88 strand locks


def wave_locks(roots, group, n=WAVE_LOCKS, seed=4):
    """The locks a wave is shared in: k-means on the roots within each group (the front each side and the back), so
    neighbouring guides wave together and the children, blended within a lock, keep the wave instead of averaging
    two phases flat."""
    from scipy.cluster.vq import kmeans2
    lock = np.zeros(len(roots), int); nxt = 0
    for g in np.unique(group):
        gi = np.where(group == g)[0]
        k = max(1, int(round(n * len(gi) / len(roots))))
        _, lab = kmeans2(roots[gi], k, minit='++', seed=seed)
        lock[gi] = nxt + lab; nxt += k
    return lock


def wave_frame(X):
    """Along each path: the tangent, the out-of-scalp axis (horizontal, away from the head's axis, normal to the
    tangent), and the sideways axis (round the head)."""
    T = np.gradient(X, axis=1); T /= np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-12)
    r = X.copy(); r[..., 2] = 0; r[..., 0] -= SF.AXIS[0]; r[..., 1] -= SF.AXIS[1]
    r -= T * (r * T).sum(-1, keepdims=True); r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    return T, r, np.cross(T, r)


def wave_params(roots, lock):
    """Wavelength (HH) and phase per guide: the creator's section by the root's angle, one phase per lock."""
    th = SF.theta(roots)
    sec = np.minimum(4, np.floor((np.abs(th) + 0.2) / WAVE_SEC)).astype(int)
    lam = WAVE_LAM0 + WAVE_LAM_STEP * sec
    jit = np.random.default_rng(9).random(lock.max() + 1)[lock]
    ph = np.array(WAVE_PHASE)[sec] + np.where(th > 0, 0.6, 0.0) + WAVE_LOOSE * jit
    return lam, ph


def wave_paths(X, lam, ph, A_hh, seg_hh):
    """The straight fall X (G, S+1, 3) with a wave of amplitude A_hh (G, S+1) laid along it; returns the nodes (at the
    rod's exact segment length) and the segment directions."""
    G, S1 = X.shape[:2]
    s = np.arange(S1) * seg_hh
    T, r, b = wave_frame(X)
    side = np.where(X[:, :1, 0] >= SF.AXIS[0], 1.0, -1.0)
    cross = np.argmax(X[..., 2] < WAVE_Z_ON, axis=1)          # (0 for hair rooted below it: counted from the root)
    phi = 2 * np.pi * (s[None] - cross[:, None] * seg_hh) / lam[:, None] + ph[:, None]
    off = A_hh[..., None] * (WAVE_RADIAL * np.sin(phi)[..., None] * r + side[..., None] * np.cos(phi)[..., None] * b)
    dd = np.diff(X + off, axis=1); dd /= np.maximum(np.linalg.norm(dd, axis=-1, keepdims=True), 1e-12)
    return np.concatenate([X[:, :1], X[:, :1] + np.cumsum(dd * seg_hh, 1)], 1), dd


def wave_rest(X, lengths, lam, ph, seg_hh):
    """The natural wave on the straight falls X: rest curvature binormals [BER08] of the rest wave, and the
    formula's settled shape (each node's amplitude cut by the hair hanging below it) to start the solve from."""
    G, S1 = X.shape[:2]
    s = np.arange(S1) * seg_hh
    A0 = WAVE_A_MM / SF.HH_MM * SF.sm(WAVE_Z_RAMP[0], WAVE_Z_RAMP[1], X[..., 2])
    Y, _ = wave_paths(X, lam, ph, A0, seg_hh)
    T = np.diff(Y, axis=1); T /= np.linalg.norm(T, axis=-1, keepdims=True)
    c = (T[:, :-1] * T[:, 1:]).sum(-1)
    kb0 = 2 * np.cross(T[:, :-1], T[:, 1:]) / (1 + c)[..., None]
    kb0[np.linalg.norm(kb0, axis=-1) < 1e-9] = 1e-9                    # (marks the guide as shaped where it is straight)
    hang_m = np.maximum(lengths[:, None] - np.arange(S1)[None], 0) * SEG_M
    k = 2 * np.pi / (lam * HH_M)
    f = 1.0 / (1.0 + hang_m / (ELASTOGRAVITY_M ** 3 * k[:, None] ** 2))
    Ys, ds = wave_paths(X, lam, ph, A0 * f, seg_hh)
    return kb0, Ys, ds


# Waves that drape, from stylists [WS-*]: curl last, after the brush, and let it set [WS-AS]; start below the root, at
# the cheekbone or eye level [WS-AS][WS-WWW] (WAVE_Z_RAMP); leave the last inch straight [WS-TYMO][WS-BTC]; weight
# stretches the wave near the top [WS-HF] (wave_rest's hang). [WS-AS] americansalon.com, natural beach waves;
# [WS-WWW] whowhatwear.com, S-waves; [WS-TYMO] tymobeauty.com wave guide; [WS-BTC] behindthechair.com, Hollywood and
# beachy waves; [WS-HF] hairfinder.com, weighed-down hair. CG: Pixar (Merida) and Disney keep the curl's rest shape
# against gravity (fxguide, Brave; Disney, gravity preloading): here the wave is laid again after the passes.
WAVE_SET_GAIN = 0.8                # the set wave's share of the rest wave, CALIBRATED: the sides' median wave 8 mm, the
                                   # creator's 8.2 (at 1.0: 10.3, and the neighbours' waves crossed)
WAVE_END_STRAIGHT = 2.5            # nodes (2.5 cm, an inch): the ends left straight
WAVE_SET_TH = (2.0, 2.6)           # |theta| of the roots: the wave set fades from the sides to none at the back
WAVE_SET_ITERS = 60                # Laplacian steps that take the wave out of a path to find its middle line (a 14 cm
                                   # wave falls to 5%, the shoulder's 60 cm curve keeps 85%)


def middle_line(X):
    """Each strand's middle line: the wave smoothed out of it below WAVE_Z_RAMP's top (above it the tuck behind the ear
    and the combed top keep their line; smoothed there, the tucked sides swung forward over the cheek), the root held,
    carried on past the strand's end in its last direction. Returns the line, the valid nodes and their count."""
    G, S1 = X.shape[:2]
    ok = ~np.isnan(X[..., 0])
    n = ok.sum(1)
    C = X.copy()
    for g in range(G):
        k = n[g]
        if k < S1:
            d = C[g, k - 1] - C[g, k - 2]
            C[g, k:] = C[g, k - 1] + d * np.arange(1, S1 - k + 1)[:, None]
    wz = SF.sm(WAVE_Z_RAMP[0], WAVE_Z_RAMP[1], C[..., 2])[..., None]
    for _ in range(WAVE_SET_ITERS):
        L = np.zeros_like(C)
        L[:, 1:-1] = 0.5 * (C[:, :-2] + C[:, 2:]) - C[:, 1:-1]
        L[:, -1] = C[:, -2] - C[:, -1] + (C[:, -2] - C[:, -3])          # (the end: toward its line's continuation)
        C = C + 0.5 * L * wz
    return C, ok, n


def wave_on(C, n, lam, ph, seg_hh, gain=1.0):
    """The wave laid along middle lines C (strands n nodes long), as wave_rest lays it on the straight falls: the rest
    curvature binormals of the rest wave (kb0, in the frame of the line it lies along now) and the settled shape (each
    node's amplitude cut by the hair hanging below it). The last inch is left straight [WS-TYMO][WS-BTC]."""
    G, S1 = C.shape[:2]
    A0 = gain * WAVE_A_MM / SF.HH_MM * SF.sm(WAVE_Z_RAMP[0], WAVE_Z_RAMP[1], C[..., 2])
    A0 = A0 * np.clip(((n - 1)[:, None] - np.arange(S1)[None]) / WAVE_END_STRAIGHT, 0, 1)
    Y, _ = wave_paths(C, lam, ph, A0, seg_hh)
    T = np.diff(Y, axis=1); T /= np.linalg.norm(T, axis=-1, keepdims=True)
    c = (T[:, :-1] * T[:, 1:]).sum(-1)
    kb0 = 2 * np.cross(T[:, :-1], T[:, 1:]) / (1 + c)[..., None]
    kb0[np.linalg.norm(kb0, axis=-1) < 1e-9] = 1e-9
    hang_m = np.maximum((n - 1)[:, None] - np.arange(S1)[None], 0) * SEG_M
    kw = 2 * np.pi / (lam * HH_M)
    f = 1.0 / (1.0 + hang_m / (ELASTOGRAVITY_M ** 3 * kw[:, None] ** 2))
    Ys, _ = wave_paths(C, lam, ph, A0 * f, seg_hh)
    return kb0, Ys


def wave_set(X, roots, keep, lam, ph, seg_hh):
    """The wave set again as the last step, as a stylist curls last, after the brush [see WAVE_SET sources in
    build_groom]: each strand's middle line gets the formula's settled wave laid back along it, at the rod's segment
    length, on the sides only (fading out over WAVE_SET_TH). keep: left as they are."""
    C, ok, n = middle_line(X)
    _, Y = wave_on(C, n, lam, ph, seg_hh, WAVE_SET_GAIN)
    Y[~ok] = np.nan
    w = (1.0 - SF.sm(WAVE_SET_TH[0], WAVE_SET_TH[1], np.abs(SF.theta(roots)))) * ~keep
    return X + w[:, None, None] * (Y - X)


def push_out(col, X, layer=None, margin_hh=1.0 / SF.HH_MM, steps=8):
    """Nodes of a placed (combed, not solved) path moved out of the head and the body along the depth's gradient,
    to the hair's layer off them plus a margin; the root stays."""
    import torch
    G, S1 = X.shape[:2]
    X = X.copy()
    for _ in range(steps):
        moved = False
        for fn in (col.head_depth, col.body_depth):
            P = torch.tensor(X[:, 1:].reshape(-1, 3), dtype=torch.float64, requires_grad=True)
            d = fn(P)
            d.sum().backward()
            n = P.grad.numpy(); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            o = layer(torch.tensor(X[:, 1:], dtype=torch.float64)).numpy().reshape(-1) if layer is not None else 0.0
            pen = np.maximum(d.detach().numpy() + o + margin_hh, 0)
            if (pen > 0).any():
                moved = True
                X[:, 1:] -= (n * pen[:, None]).reshape(G, S1 - 1, 3)       # (the depth grows inward)
        if not moved:
            break
    return X


LAYER_GAP_MM = 2.0                 # STYLE: the top layer's middle lies this far over the outer surface of the hair under it
LAYER_EASE_M = 0.02                # m: from the root, the top layer rises onto the hair under it over this length
LAYER_BIN = (0.08, 0.03)           # (rad round the axis, HH up): the cells the under layer's outer surface is read in


def layer_over(X, top, seg_hh, under=None):
    """Lay the `top` guides (the bangs and the face-framing hair) over the rest. A haircut's shorter, front layers lie
    on top of the longer hair under them: "the hair on top ... blends with layers beneath" (Wikipedia, Layered hair),
    as the Face Creator draws the Ranger's side bangs over the hair at her sides. Combed or traced on their own, they
    lay 20-30 mm inside the side hair (layerchk: the Ranger's bangs 100% under it, the framing hair 94%). Each point of a
    top guide is moved out from the head's axis to the under layer's outer surface (the furthest under-layer node in
    its cell of angle and height) plus LAYER_GAP_MM, eased in from the root over LAYER_EASE_M; a guide that clears the
    hair under it is left where it is."""
    X = X.copy()
    under = X[~top if under is None else under].reshape(-1, 3)
    under = under[~np.isnan(under[:, 0])]
    d = under[:, :2] - np.asarray(SF.AXIS)
    th, r = np.arctan2(d[:, 0], -d[:, 1]), np.hypot(d[:, 0], d[:, 1])
    key = lambda t, z: np.round(t / LAYER_BIN[0]).astype(int) * 100000 + np.round(z / LAYER_BIN[1]).astype(int)
    env = {}
    for k, rr in zip(key(th, under[:, 2]), r):
        if rr > env.get(k, 0.0):
            env[k] = rr
    gap = LAYER_GAP_MM / SF.HH_MM
    ease_n = max(1, int(round(LAYER_EASE_M / HH_M / seg_hh)))
    w = np.clip(np.arange(X.shape[1]) / ease_n, 0, 1)
    w = w * w * (3 - 2 * w)
    for g in np.where(top)[0]:
        n = int((~np.isnan(X[g, :, 0])).sum())          # (a guide is its first n nodes; the cut rest is NaN)
        P = X[g, :n]
        dd = P[:, :2] - np.asarray(SF.AXIS)
        tg, rg = np.arctan2(dd[:, 0], -dd[:, 1]), np.hypot(dd[:, 0], dd[:, 1])
        need = np.array([env.get(k, 0.0) for k in key(tg, P[:, 2])]) + gap
        lift = np.maximum(need - rg, 0)
        # (spread along the strand, so a cell edge does not kink it, and eased in from the root)
        lift = np.convolve(np.pad(lift, 3, mode='edge'), np.ones(7) / 7, mode='valid') * w[:n]
        P[:, :2] += dd / np.maximum(rg, 1e-9)[:, None] * lift[:, None]
    return X


FALL_Z = 0.3                       # HH: below the jaw the face-framing hair only falls
FALL_WIN = 9                       # nodes (9 cm): the depth (y) of the fall is smoothed over this window


def smooth_fall(X, sel, body=None, margin_hh=3.0 / SF.HH_MM):
    """The face-framing hair below the jaw falls; it does not swing back in. Combed down the cheek to the jaw and let
    fall from there, the Ranger's framing guides bowed in toward the neck below the chin and out again (a third of
    them 20-37 degrees off the fall over their last 5 cm), which from below read as rows of hooks inside the hair.
    Below FALL_Z each selected guide keeps the widest |x| it has reached (it never comes back in toward the neck)
    and its depth is smoothed over FALL_WIN nodes; nodes then in the body are set back out from the axis."""
    X = X.copy()
    for g in np.where(sel)[0]:
        n = int((~np.isnan(X[g, :, 0])).sum())
        P = X[g, :n]
        low = np.where(P[:, 2] < FALL_Z)[0]
        if len(low) < 3:
            continue
        a = low[0]
        sx = np.sign(P[a, 0] - SF.AXIS[0]) or 1.0
        ax = sx * (P[a:, 0] - SF.AXIS[0])
        P[a:, 0] = SF.AXIS[0] + sx * np.maximum.accumulate(ax)
        h = FALL_WIN // 2
        y = np.pad(P[max(0, a - h):, 1], (0, h), mode='edge')
        ys = np.convolve(np.pad(y, (h, 0), mode='edge'), np.ones(FALL_WIN) / FALL_WIN, mode='valid')
        P[a:, 1] = ys[-(n - a) - h:len(ys) - h] if h else ys[-(n - a):]
        if body is not None:
            for _ in range(4):
                dep = body_depth_np(body, P[a:]) + margin_hh
                bad = dep > 0
                if not bad.any():
                    break
                q = P[a:][bad, :2] - np.asarray(SF.AXIS)
                r = np.linalg.norm(q, axis=1)
                Q = P[a:]
                Q[bad, :2] = np.asarray(SF.AXIS) + q * ((r + dep[bad]) / np.maximum(r, 1e-9))[:, None]
                P[a:] = Q
    return X


FLARE_TOP = 0.7                    # HH: below here the sides are fitted to the creator's outline
FLARE_IN = 0.25                    # HH: the hair's inner side at the jaw and neck (the neck's half-width is about
                                   # 0.2): the fit widens or narrows the hair beyond this, so the hair next to the
                                   # head and neck stays where it is
FLARE_BIN = 0.05                   # HH: the heights the outlines are compared in
FLARE_PCT = 98                     # the outline: this percentile of the hair's distance out to the side


def creator_flare(X, style_path, surf=None):
    """The Ranger's sides flow outward as the creator draws them: from the cheek down her outline widens from 95 to
    121 mm (half-width, at -0.25 HH) while the groom's fell as a straight column at 109 mm (14 mm too wide at the jaw,
    8 mm too narrow at the shoulders). Each side is fitted height by height: the hair beyond FLARE_IN from the axis is
    scaled so its outline (FLARE_PCT) lands on the creator's, eased in below FLARE_TOP. A fit to the creator, not
    physics: it stretches or packs the outer hair by up to about a quarter."""
    import creator_truth as CT
    P, A, strands = CT.load(style_path)
    Q = CT.creator_points(P, A, strands)
    X = X.copy()
    pts = X.reshape(-1, 3)
    ok = ~np.isnan(pts[:, 0])
    zs = np.arange(FLARE_TOP + FLARE_BIN, pts[ok, 2].min() - FLARE_BIN, -FLARE_BIN)
    for sgn in (1.0, -1.0):
        rat = []
        for z in zs:
            c = Q[(np.abs(Q[:, 2] - z) < FLARE_BIN) & (sgn * Q[:, 0] > 0)]
            g = pts[ok & (np.abs(pts[:, 2] - z) < FLARE_BIN) & (sgn * pts[:, 0] > 0)]
            rat.append((np.percentile(sgn * c[:, 0], FLARE_PCT) - FLARE_IN) / (np.percentile(sgn * g[:, 0], FLARE_PCT) - FLARE_IN)
                       if len(c) > 20 and len(g) > 20 else np.nan)
        rat = np.array(rat)
        rat = np.where(np.isnan(rat), np.interp(np.arange(len(rat)), np.where(~np.isnan(rat))[0], rat[~np.isnan(rat)]), rat)
        rat = np.convolve(np.pad(rat, 1, mode='edge'), np.ones(3) / 3, mode='valid')
        rat = np.clip(rat, 0.6, 1.6)
        side = ok & (sgn * (pts[:, 0] - SF.AXIS[0]) > FLARE_IN)
        z = pts[side, 2]
        k = np.interp(-z, -zs, rat)                                    # (zs falls)
        e = np.clip((FLARE_TOP - z) / (2 * FLARE_BIN), 0, 1)
        k = 1 + (k - 1) * e * e * (3 - 2 * e)
        pts[side, 0] = SF.AXIS[0] + sgn * (FLARE_IN + (sgn * (pts[side, 0] - SF.AXIS[0]) - FLARE_IN) * k)
    if surf is not None:
        # (packing the hair at the ear drew some of its inner nodes into the skull: set them back out over the skin)
        up = ok & (pts[:, 2] > 0.2)
        q = pts[up] - SF.CENTRE
        r = np.linalg.norm(q, axis=1)
        need = surf.radius(q / r[:, None]) + 0.5 / SF.HH_MM
        inside = r < need
        q[inside] *= (need[inside] / r[inside])[:, None]
        pts[up] = SF.CENTRE + q
    return pts.reshape(X.shape)


def creator_cut(style_path, nb=48):
    """The end height by angle round the head from the creator's own hair: in each bin of angle (where the hair
    hangs), how low its strands and its cap's hem reach (CUT_PCT percentile of their lowest points), smoothed round
    the head. Returns a function of the angle."""
    import gzip
    import gzip
    d = json.load(gzip.open(style_path))
    pts = []
    for st in d['strands']:
        p = np.array(st['p'], float).reshape(-1, 3)
        pts.append(p[np.argmin(p[:, 2])])
    c = d['cap']
    P = np.array(c['p'], float).reshape(c['cols'], c['rows'], 3)      # (the cap's points run down each column)
    A = np.array(c['a'], float).reshape(c['cols'], c['rows'])
    for j in range(c['cols']):
        ok = np.where(A[j] > 0.5)[0]
        if len(ok):
            pts.append(P[j, ok[-1]])
    pts = np.array(pts)
    b = ((SF.theta(pts) + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    z = np.array([-np.percentile(-pts[b == i, 2], CUT_PCT) if (b == i).any() else np.nan for i in range(nb)])
    ctr = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
    # made even, the longer side kept (the creator's two sides differ by a lock or two), and in front of the face,
    # where no hair hangs (the points there are the cap's top and the bangs), held at the framing locks' length
    # beside it, before the smoothing, so the face does not shorten the framing locks
    z = np.fmin(z, z[::-1])
    front = np.abs(ctr) < CUT_FRONT_TH
    z[front] = np.nan
    edge = np.argmin(np.where(front, 9, np.abs(np.abs(ctr) - CUT_FRONT_TH)))
    z[front] = z[edge]
    z = np.where(np.isnan(z), np.nanmean(z), z)
    z = np.convolve(np.r_[z[-1:], z, z[:1]], np.ones(3) / 3, 'valid')
    return lambda q: np.interp(np.asarray(q), ctr, z, period=2 * np.pi)


CUT_FRONT_TH = 0.6                 # rad: in front of this the creator's cut is held at its value here
CUT_PCT = 99                       # the creator's longest ends make its outline (its tips are cut uneven, not on one line)


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
    # (only in front of the part's end: behind the vertex the hair is brushed back, so the part does not run on over
    # the crown and open a bare patch there, John's "no bald spots")
    w = top * front * (1 - SF.sm(0.15, 0.35, np.abs(P[:, 0])) * 0.6) * (1 - SF.sm(PART_END_Y - 0.06, PART_END_Y + 0.04, P[:, 1]))
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
    fwd = (~front | (P[:, 1] > PART_END_Y - 0.06)) & (out[:, 1] < 0.1)     # (and behind the part's end)
    out[fwd, 1] = 0.1
    out[fwd] += back[fwd] * 0.3
    out -= N * (N * out).sum(1, keepdims=True)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    # roots inside the part itself are bare scalp
    keep = ~((np.abs(P[:, 0]) < PART_HALF) & (P[:, 2] > F['landmarks']['trichion'][2] - 0.01) & (P[:, 2] < PART_BACK_Z) & front & (P[:, 1] < PART_END_Y + 0.04))
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
RANGER_SPLIT = 1.0                        # STYLE (Ranger): hair rooted in front of this angle frames the face and falls in front
WARDEN_SPLIT = 1.25                # rad: the creator's Warden: its strands that fall in front of the shoulder are rooted
                                   # up to 1.21 rad round (90th percentile), so more of her hair frames the face
SPLIT = RANGER_SPLIT
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


# ---------------------------------------------------------------------------------------------------- the stress test
# The hair stress test (tools/render-queue/rq_hair.py, the game's fabric solver) found four faults in the first
# grooms (rejection 113): the hair hung through the character's own body (it was solved against the bare
# mannequin), the parting and the jaw had sharp bends, and the roots stood off the scalp under the creator's cap.
# The fixes below take the stress test's own colliders and limits as the measure.

def character_body(sections, mannequin, na=96):
    """A character's body (the creator's sections: z, half-width, front depth, back depth, as makeDrape reads them;
    her collar, cape and back) as a mannequin-style table, so the solver's colliders take it unchanged: the radius
    from SF.AXIS where each ray first leaves the section's superellipse (the stress test's body_field, exactly)."""
    S = np.array(sections, float)
    CY = 0.04
    z0, dz = mannequin['z0'], mannequin['dz']
    nz = int(math.ceil((z0 - S[-1, 0]) / dz)) + 1
    zz = z0 - dz * np.arange(nz)
    th = np.arange(na) / na * 2 * np.pi - np.pi
    Z, TH = np.meshgrid(zz, th, indexing='ij')

    def F(r):
        x = SF.AXIS[0] + r * np.sin(TH)
        y = SF.AXIS[1] - r * np.cos(TH) - CY
        z = Z
        k = np.clip(np.searchsorted(-S[:, 0], -z) - 1, 0, len(S) - 2)
        t = np.clip((S[k, 0] - z) / (S[k, 0] - S[k + 1, 0]), 0, 1)
        t = t * t * (3 - 2 * t)
        D = S[k, 1:] + (S[k + 1, 1:] - S[k, 1:]) * t[..., None]
        ry = np.where(y < 0, D[..., 1], D[..., 2])
        return (np.abs(x / D[..., 0]) ** 3 + np.abs(y / ry) ** 3) ** (1 / 3) - 1
    lo, hi = np.zeros(Z.shape), np.full(Z.shape, 2.0)
    for _ in range(40):                    # (bisection: inside at lo, outside at hi)
        mid = 0.5 * (lo + hi)
        ins = F(mid) < 0
        lo, hi = np.where(ins, mid, lo), np.where(ins, hi, mid)
    R = np.where(Z <= S[0, 0], lo, 0.0)    # (above the top section: no body)
    return dict(z0=z0, dz=dz, na=na, r=np.round(R * 1000).astype(int).ravel().tolist())


class CreatorCap:
    """The creator's hair cap (the export's cap grid, where its alpha shows) as a radius by direction from the skull's
    centre: the creator roots its strands on this shell, and the stress test counts points under it as buried."""
    def __init__(self, style_path):
        import gzip
        from scipy.spatial import cKDTree
        cap = json.load(gzip.open(style_path))['cap']
        P = np.array(cap['p'], float).reshape(-1, 3)
        a = np.array(cap['a'], float)
        P = P[a > 0.2]
        q = P - SF.CENTRE
        self.r = np.linalg.norm(q, axis=1)
        self.tree = cKDTree(q / self.r[:, None])

    def radius(self, U):
        d, i = self.tree.query(U, k=6)
        w = 1.0 / np.maximum(d, 1e-4) ** 2
        R = (self.r[i] * w).sum(1) / w.sum(1)
        return np.where(d[:, 0] < 0.08, R, 0.0)          # (no cap in that direction: nothing to clear)


CAP_CLEAR_MM = 1.0     # STYLE: the hair lies this far over the creator's cap (the stress test's thickness margin is 2 mm)
CAP_WIN = 3            # nodes (3 cm): the window the push off the cap is smoothed over
CAP_ROOT_M = 0.03      # m: below the root the hair rises onto the cap over this length (a lock's root fans out of it)


def onto_cap(X, cap, keep=None):
    """Each guide moved out of the creator's cap: its root to the cap's surface (where the creator roots its strands),
    the rest pushed out along the radius to CAP_CLEAR_MM over it, eased in over CAP_ROOT_M so the root leaves the cap
    along it instead of standing up off the skin. keep: guides left as they are."""
    X = X.copy()
    G, S1, _ = X.shape
    q = X - SF.CENTRE
    r = np.linalg.norm(q, axis=-1)
    U = q / np.maximum(r, 1e-9)[..., None]
    ok = ~np.isnan(X[..., 0])
    R = np.zeros((G, S1))
    R[ok] = cap.radius(U[ok])
    s = np.arange(S1) * SEG_M
    clear = np.minimum(1.0, s / CAP_ROOT_M) * CAP_CLEAR_MM / SF.HH_MM
    need = np.where((R > 0) & ok, R + clear[None] - r, 0.0)
    need = np.maximum(need, 0.0)
    # (smoothed along the strand, a max over CAP_WIN nodes then a mean: pushed node by node, the hair steps wherever
    # the cap's edge or its radius changes under it)
    W = CAP_WIN
    pad = np.pad(need, ((0, 0), (W, W)), mode='edge')
    mx = np.max(np.stack([pad[:, i:i + S1] for i in range(2 * W + 1)]), 0)
    pad = np.pad(mx, ((0, 0), (W, W)), mode='edge')
    need = np.mean(np.stack([pad[:, i:i + S1] for i in range(2 * W + 1)]), 0)
    need[:, 0] = np.maximum(R[:, 0] - r[:, 0], 0.0)
    if keep is not None:
        need[keep, KEEP_ROOT_NODES:] = 0.0
    X[ok] = (SF.CENTRE + U * (r + need)[..., None])[ok]
    return X


KINK_MAX_DEG = 20.0    # STYLE: no single 1 cm segment turns further than this (the stress test flags 35 at 9 mm)
RELAX_ITERS = 200
KEEP_ROOT_NODES = 99  # nodes: a kept (combed) guide is relaxed this far from its root (all of it: only its sharp points move)


def relax_kinks(X, col, cap=None, keep=None, iters=RELAX_ITERS):
    """Sharp bends taken out: where a node turns more than KINK_MAX_DEG, it and its neighbours are moved toward the
    mean of theirs (a local Laplacian step), then the chain is set back to its segment length from the root and
    pushed out of the head, the body and the cap again. The root and its first node stay."""
    import torch
    X = X.copy()
    seg = SEG_M / HH_M
    for _ in range(iters):
        d = np.diff(X, axis=1)
        u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
        c = np.einsum('gsk,gsk->gs', u[:, :-1], u[:, 1:])
        ang = np.degrees(np.arccos(np.clip(np.nan_to_num(c, nan=1.0), -1, 1)))     # at nodes 1 .. S-1
        nb = np.fmax(np.pad(ang, ((0, 0), (1, 0)))[:, :-1], np.pad(ang, ((0, 0), (0, 1)))[:, 1:])
        bad = np.zeros(X.shape[:2], bool)
        # (a sharp point, sharper than the line either side of it, or any very sharp turn; a wave keeps its curves)
        bad[:, 1:-1] = ((ang > KINK_MAX_DEG) & (ang > 1.5 * nb)) | (ang > 2 * KINK_MAX_DEG)
        if keep is not None:              # (combed guides kept: only their roots, where they leave the scalp)
            bad[keep, KEEP_ROOT_NODES:] = False
        if not bad.any():
            break
        sm = bad.copy()
        sm[:, 1:] |= bad[:, :-1]
        sm[:, :-1] |= bad[:, 1:]
        sm[:, :1] = False
        L = X.copy()
        L[:, 1:-1] = 0.5 * (X[:, :-2] + X[:, 2:])
        last = (~np.isnan(X[..., 0])).sum(1) - 1                # (a guide's last node: its own mean is itself)
        L[np.arange(len(X)), last] = X[np.arange(len(X)), last]
        X = np.where(sm[..., None] & ~np.isnan(L), 0.5 * X + 0.5 * L, X)
        # back to the segment length, from the root out
        for k in range(2, X.shape[1]):
            v = X[:, k] - X[:, k - 1]
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            X[:, k] = X[:, k - 1] + v / np.maximum(n, 1e-12) * seg
        g, k = np.where(sm & ~np.isnan(X[..., 0]))
        if len(g):
            P = torch.tensor(X[g, k], dtype=torch.float64)
            for fn in (col.head_depth, col.body_depth):
                for _ in range(4):
                    P.requires_grad_(True)
                    dep = fn(P)
                    gr, = torch.autograd.grad(dep.clamp(min=0).sum(), P)
                    with torch.no_grad():
                        n = gr / gr.norm(dim=1, keepdim=True).clamp(min=1e-12)
                        P = P - n * (dep + 1.0 / SF.HH_MM).clamp(min=0)[:, None]
            X[g, k] = P.detach().numpy()
    if cap is not None:
        X = onto_cap(X, cap, keep)
    # (a guide the smoothing folded back on itself, a few in a thousand, ends where it folds)
    d = np.diff(X, axis=1)
    u = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    c = np.nan_to_num(np.einsum('gsk,gsk->gs', u[:, :-1], u[:, 1:]), nan=1.0)
    for g in np.where((c < -0.5).any(1))[0]:
        k = int(np.argmax(c[g] < -0.5))
        X[g, max(3, k + 1):] = np.nan
    return X


CLIMB_HH = 0.05        # HH: a guide that climbs this far back up after it has fallen is cut where it turns (hair
                       # under gravity does not climb: a few framing guides combed onto the cape's steep shoulder looped
                       # back up over the head)
CLIMB_FROM = 6         # nodes: a bang's root lift (it leaves the forehead upward) is not a climb


COMB_K = 8             # the guides a guide is combed with (its nearest in its own group, by root and path)
COMB_PULL = 0.5        # STYLE: per pass, how far each segment turns toward its neighbours' mean direction
COMB_PASSES = 4
ROOT_LIE_DEG = 25.0    # STYLE: the steepest a root's first segments leave the scalp (the creator's p99 is 29)
ROOT_LIE_N = 3         # segments from the root held to that
COMB_ROOT_M = 0.0      # m: the comb takes hold from the root (a follicle grows along the flow round it: a root left
                       # standing across the hair over it was most of the crossings the stress test found)


def comb_pass(X, group, col, cap=None, passes=COMB_PASSES, pull=COMB_PULL):
    """A comb run through the settled hair: each guide's segments turned toward the mean direction of its COMB_K
    nearest guides of the same group (the same lock: the back, each side's face-framing hair, each side's bangs, or
    for wavy hair each wave lock, so a comb never drags one lock into the next), node by node down the strand, then
    the strand rebuilt from its root at its own segment length and kept out of the head, the body and the cap. Strays
    come back into their lock, crossing neighbours lie parallel, and the silhouette stays: the mean of a lock's
    directions is the lock's own direction. (The usual groom tool's comb: Houdini's guide groom, XGen's comb brush.)"""
    from scipy.spatial import cKDTree
    import torch
    X = X.copy()
    G, S1, _ = X.shape
    seg = SEG_M / HH_M
    ok = ~np.isnan(X[..., 0])
    s = np.arange(S1 - 1) * SEG_M
    hold = (np.clip(s / COMB_ROOT_M, 0, 1) if COMB_ROOT_M > 0 else np.ones_like(s))[None, :, None]
    for _ in range(passes):
        D = np.diff(X, axis=1)
        D /= np.maximum(np.linalg.norm(D, axis=-1, keepdims=True), 1e-12)
        valid = ~np.isnan(D[..., 0])
        Dn = D.copy()
        for gr in np.unique(group):
            ids = np.where(group == gr)[0]
            if len(ids) < 3:
                continue
            feat = []
            for k in (0, 5, 10, 20):
                Y = X[ids, min(k, S1 - 1)]
                last = X[ids, np.maximum(ok[ids].sum(1) - 1, 0)]
                feat.append(np.where(np.isnan(Y), last, Y))
            F = np.concatenate(feat, 1)
            kk = min(COMB_K + 1, len(ids))
            dist, nb = cKDTree(F).query(F, k=kk)
            nb = ids[nb[:, 1:]]
            w = np.exp(-(dist[:, 1:] / max(np.median(dist[:, 1:]), 1e-9)) ** 2)        # (nearer guides comb harder)
            Dm = np.zeros((len(ids), S1 - 1, 3))
            W = np.zeros((len(ids), S1 - 1))
            for j in range(nb.shape[1]):
                v = valid[nb[:, j]]
                Dm += np.where(v[..., None], np.nan_to_num(D[nb[:, j]]), 0.0) * w[:, j, None, None]
                W += v * w[:, j, None]
            has = (W > 1e-6) & valid[ids]
            Dm = Dm / np.maximum(W, 1e-9)[..., None]
            a = pull * hold[0, :, 0][None] * has
            mix = (1 - a)[..., None] * D[ids] + a[..., None] * Dm
            mix /= np.maximum(np.linalg.norm(mix, axis=-1, keepdims=True), 1e-12)
            Dn[ids] = np.where(has[..., None], mix, D[ids])
        # the first segments laid down along the scalp: no steeper than ROOT_LIE_DEG off it (a root that stands up
        # crosses the hair lying over it; the creator's roots leave at 29 degrees, p99)
        nr = X[:, :ROOT_LIE_N] - SF.CENTRE
        nr /= np.maximum(np.linalg.norm(nr, axis=-1, keepdims=True), 1e-12)
        d = Dn[:, :ROOT_LIE_N]
        up = np.einsum('gsk,gsk->gs', d, nr)
        tan = d - up[..., None] * nr
        tl = np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-12)
        lim = math.sin(math.radians(ROOT_LIE_DEG))
        upc = np.clip(up, -lim, lim)
        dl = tan / tl * np.sqrt(1 - upc ** 2)[..., None] + upc[..., None] * nr
        Dn[:, :ROOT_LIE_N] = np.where(np.isnan(d), d, dl)
        Y = X.copy()
        for k in range(S1 - 1):
            Y[:, k + 1] = np.where(valid[:, k, None], Y[:, k] + Dn[:, k] * seg, np.nan)
        # kept out of the head and the body, as relax_kinks keeps them
        g, k = np.where(ok)
        P = torch.tensor(Y[g, k], dtype=torch.float64)
        for fn in (col.head_depth, col.body_depth):
            for _ in range(4):
                P.requires_grad_(True)
                dep = fn(P)
                gr_, = torch.autograd.grad(dep.clamp(min=0).sum(), P)
                with torch.no_grad():
                    n = gr_ / gr_.norm(dim=1, keepdim=True).clamp(min=1e-12)
                    P = P - n * (dep + 1.0 / SF.HH_MM).clamp(min=0)[:, None]
        Y[g, k] = P.detach().numpy()
        X = Y
    if cap is not None:
        X = onto_cap(X, cap)
    return X


FRONT_X = 0.22         # HH: the side hair (outside the face's width); the bangs in front of the face are left alone
FRONT_TOP = 0.55       # HH: below the temple the side hair's front edge is held to the creator's
FRONT_BIN = 0.05       # HH: the heights the front edges are compared in
FRONT_PCT = 5          # the front edge: this percentile of the hair's depth (y) at that height
FRONT_WIN = 2          # nodes: the push back is smoothed along the strand over this window
FRONT_SLOPE = 1.0      # HH back per HH down: the steepest the held edge turns back below the creator's front ends


def creator_front(X, style_path, col=None, keep=None):
    """The side hair's front edge held to the creator's: below the temple no side hair lies further forward (less y)
    than the creator's side hair at that height, left and right. The hair is swung back round the body's axis at its
    own distance from it (round the jaw and over the shoulder, not into them: moved straight back it went into the
    body, and the body pushed it forward again), by the angle it reaches past the edge, smoothed along the strand (a
    max then a mean over FRONT_WIN nodes) and eased in from FRONT_TOP. Below the creator's front ends the edge turns
    back no faster than FRONT_SLOPE."""
    import gzip
    import torch
    import gzip
    d = json.load(gzip.open(style_path))
    C = np.concatenate([np.array(st['p'], float).reshape(-1, 3) for st in d['strands']])
    zb = np.arange(FRONT_TOP + FRONT_BIN, -1.6, -FRONT_BIN)
    X = X.copy()
    G, S1, _ = X.shape
    ax, ay = SF.AXIS
    th = np.arctan2(X[..., 0] - ax, -(X[..., 1] - ay))
    r = np.hypot(X[..., 0] - ax, X[..., 1] - ay)
    turn = np.zeros((G, S1))
    for sd in (-1.0, 1.0):
        edge = np.full(len(zb), np.nan)
        side = C[np.sign(C[:, 0]) == sd]
        for i, z in enumerate(zb):
            m = (np.abs(side[:, 0]) > FRONT_X) & (np.abs(side[:, 2] - z) < FRONT_BIN)
            if m.sum() > 10:
                edge[i] = np.percentile(side[m, 1], FRONT_PCT)
        ok = ~np.isnan(edge)
        if not ok.any():
            continue
        last = np.where(ok)[0].max()
        edge[last:] = edge[last]
        edge = np.interp(np.arange(len(zb)), np.where(~np.isnan(edge))[0], edge[~np.isnan(edge)])
        edge = np.convolve(np.pad(edge, 2, mode='edge'), np.ones(5) / 5, mode='valid')
        # (going down, the edge moves back no faster than FRONT_SLOPE: the creator's front locks end at the chin and its
        # edge then jumps behind the shoulder, and a strand held to that jump was swung back off the jaw)
        for i in range(1, len(edge)):
            edge[i] = min(edge[i], edge[i - 1] + FRONT_SLOPE * FRONT_BIN)
        e = np.interp(-np.nan_to_num(X[..., 2]), -zb, edge)
        on = (np.sign(X[..., 0]) == sd) & (np.abs(X[..., 0]) > FRONT_X) & ~np.isnan(X[..., 0])
        ease = np.clip((FRONT_TOP - X[..., 2]) / 0.05, 0, 1)
        want = np.arccos(np.clip((ay - e) / np.maximum(r, 1e-9), -1, 1))         # |th| at which y reaches the edge
        need = np.nan_to_num(np.maximum(want - np.abs(th), 0.0) * ease)
        turn = np.where(on, np.maximum(turn, need), turn)
    W = FRONT_WIN
    pad = np.pad(turn, ((0, 0), (W, W)), mode='edge')
    mx = np.max(np.stack([pad[:, i:i + S1] for i in range(2 * W + 1)]), 0)
    pad = np.pad(mx, ((0, 0), (W, W)), mode='edge')
    turn = np.mean(np.stack([pad[:, i:i + S1] for i in range(2 * W + 1)]), 0)
    turn[:, 0] = 0.0
    if keep is not None:
        turn[keep] = 0.0
    th2 = th + np.sign(th) * turn
    X[..., 0] = np.where(np.isnan(X[..., 0]), np.nan, ax + r * np.sin(th2))
    X[..., 1] = np.where(np.isnan(X[..., 1]), np.nan, ay - r * np.cos(th2))
    if col is not None:
        g, k = np.where(~np.isnan(X[..., 0]) & (turn > 0))
        if len(g):
            P = torch.tensor(X[g, k], dtype=torch.float64)
            for fn in (col.head_depth, col.body_depth):
                for _ in range(4):
                    P.requires_grad_(True)
                    dep = fn(P)
                    gr_, = torch.autograd.grad(dep.clamp(min=0).sum(), P)
                    with torch.no_grad():
                        n = gr_ / gr_.norm(dim=1, keepdim=True).clamp(min=1e-12)
                        P = P - n * (dep + 1.0 / SF.HH_MM).clamp(min=0)[:, None]
            X[g, k] = P.detach().numpy()
    return X


# ---------------------------------------------------------------------------------------------------- the styling
# How a stylist cuts and finishes long hair (sources: Sam Villa, "Long Layering and Face Framing" [SV]; Red Seal
# hairdressing study guide, "Layering, Over-Direction & Guidelines" [RS]; Salon Today, "10 tips for cutting and blow
# drying long layered hair" [ST]; Drybar, "Round Brush Hair Styling Technique" [DB]; Roz, "How to do a blowout" [RZ]):
#   layers   cut at 90 degrees or more of elevation [RS]: the hair high on the head ends higher than the perimeter,
#            only slightly shorter for long layers [RS], so the ends stack instead of ending on one blunt line
#   framing  the face frame starts between the cheekbone and the jaw [SV] and steps longer away from the face [AS:
#            American Salon, "How to cut face-framing layers"], cut and blown away from the face [SV][RZ]
#   ends     rolled round the brush [RZ]: under at the back and sides, away from the face in front [RZ][DB]
# Both grooms were cut one length (every root in a sector ended at the same height); the creator's own hair is
# layered (its crown strands end 0.3 to 0.55 HH above its nape strands) and face framed (its front strands end at
# the cheek, z 0.25, Ranger).
STYLES = {
    #            frame_z: the face frame's shortest piece (HH); frame_th: |theta| over which it steps to the perimeter;
    #            layer_k: end rise per HH of root height above layer_z0; roll_deg: the ends' roll (0: none)
    'ranger': dict(frame_z=0.25, frame_th=(0.4, 1.0), layer_k=1.2, layer_z0=0.72, roll_deg=15.0, tuck=(1.0,)),   # CALIBRATED:
              # creator Ranger front ends 0.26 (roots |th| .6-1), crown vs mid roots 1.2 HH per HH (sector 1.5-2.2)
    'warden': dict(frame_z=-0.2, frame_th=(0.4, 1.4), layer_k=1.2, layer_z0=0.72, roll_deg=0.0, tuck=(1.0, -1.0)),   # CALIBRATED:
              # creator Warden front ends -0.2 to -0.3; layer_k kept at the Ranger's (its crown's 2.4 is a top
              # layer of short strands over its shell); the waves are their own finish (no roll)
}
# tuck: the sides (+1 her left, -1 her right) tucked behind the ear, STYLE (John's pick from the side-style sheet,
# 2026-11-22: the Ranger's left side, both of the Warden's). Every side is also laid in to the creator's width.
TUCK_TH = (0.85, 1.6)   # |theta| of the roots tucked: in front of the ear, back to just behind it (not the bangs)
TUCK_PAST = 0.32        # rad: tucked hair lies this far behind the ear's |theta| (about 1.5 cm at the ear), STYLE
TUCK_FROM_Z = 0.72      # HH: the tuck takes hold from the temple, over 0.12 HH, STYLE
SIDE_TH = (0.9, 2.2)    # |theta| of the side hair laid in to the creator's width
SIDE_CLEAR_MM = 2.0     # mm: the side hair's width over the creator's (its 95th percentile per 0.05 HH of height), CALIBRATED
SIDE_OPEN = 1.5         # HH of width per HH of drop below the creator's side hair (wider than the shoulder's slope)
SIDE_ROLL_DEG = 20.0    # the side ends rolled under after the tuck, STYLE (the stylists' smooth finish [RZ])
CUT_FALL = 1.5          # the plumb's weight against the line above, for a cut end
CUT_TAIL = 5            # nodes (5 cm): a cut end straightened onto the line above it
POINT_CUT_HH = 0.1      # HH: the stagger of the point-cut ends (about 2 cm), STYLE
FRAME_MIN_Y = -0.12    # HH: the face frame is the hair rooted in front of this (the front hairline and temples)
ROLL_FREE_HH = 0.06     # HH: an end rolls only where it hangs this far clear of the body and the head
ROLL_M = 0.04          # m: the length of the end rolled round the brush (a 1.5 inch brush's half turn is ~6 cm)


def style_cut(X, lengths, roots, keep, st):
    """The layered, face-framed cut on the hair as it lies: each guide's end raised to the highest of its perimeter
    (where it ends now), its face-frame line and its layer line, and cut where it reaches that height (only ever
    shorter). Returns X, lengths."""
    X = X.copy()
    lengths = lengths.copy()
    th = np.abs(SF.theta(roots))
    end = np.array([x[~np.isnan(x[:, 0])][-1, 2] for x in X])
    a, b = st['frame_th']
    u = np.clip((th - a) / (b - a), 0, 1)
    u = u * u * (3 - 2 * u)
    frame = np.where(th < b, st['frame_z'] + (end - st['frame_z']) * u, -9.0)
    # (the interior layers behind the face frame: in front, the frame's own line is the cut)
    layer = end + st['layer_k'] * np.maximum(roots[:, 2] - st['layer_z0'], 0.0) * u
    target = np.maximum.reduce([end, frame, layer])
    # point cut [SV]: the shears into the ends at 45 degrees, so the ends of a section stagger instead of ending on
    # one line (cut straight across, the face frame read as a shelf at the cheek)
    jit = np.random.default_rng(23).random(len(X)) * POINT_CUT_HH
    target = np.where(target > end + 1e-6, target - jit, target)
    for g in range(len(X)):
        if keep is not None and keep[g]:
            continue
        if target[g] <= end[g] + 1e-6:
            continue
        below = np.where(X[g, 1:, 2] < target[g])[0]
        if len(below):
            n = max(3, below[0])
            X[g, n + 1:] = np.nan
            lengths[g] = min(lengths[g], n)
            # the cut end hangs on in the line above it: it was bent by the length below (swung back round the
            # jaw), and cut at that bend it hooked out
            T = min(CUT_TAIL, n - 3)
            if T >= 2:
                k0 = n - T
                ref = X[g, k0] - X[g, k0 - 3]
                ref /= max(np.linalg.norm(ref), 1e-12)
                ref = ref + CUT_FALL * np.array([0.0, 0.0, -1.0])         # (and falls toward plumb, as cut hair does
                ref /= max(np.linalg.norm(ref), 1e-12)                    # over its 4.5 cm elastogravity length)
                for j in range(k0 + 1, n + 1):
                    v = X[g, j] - X[g, j - 1]
                    L = np.linalg.norm(v)
                    X[g, j] = X[g, j - 1] + ref * L
    return X, lengths


def end_roll(X, roots, keep, deg, col=None):
    """The ends rolled as a round brush leaves them: over the last ROLL_M each segment turns a little further, up to deg
    at the tip, under (toward the body's axis) at the back and sides, away from the face for the face frame."""
    if deg <= 0:
        return X
    X = X.copy()
    G, S1, _ = X.shape
    seg = SEG_M / HH_M
    nroll = max(2, int(round(ROLL_M / SEG_M)))
    th = np.abs(SF.theta(roots))
    free = np.ones(G, bool)
    if col is not None:
        # (an end lying on the body or the cape cannot roll under: rolled, the push out stretched it down the back)
        import torch
        nn_ = (~np.isnan(X[..., 0])).sum(1)
        tips = torch.tensor(X[np.arange(G), np.maximum(nn_ - 1, 0)], dtype=torch.float64)
        dep = torch.maximum(col.body_depth(tips), col.head_depth(tips)).numpy()
        free = dep < -ROLL_FREE_HH
    tail = np.zeros((G, S1), bool)
    seglen = np.zeros((G, S1 - 1))
    for g in range(G):
        if keep is not None and keep[g]:
            continue
        if not free[g]:
            continue
        n = int((~np.isnan(X[g, :, 0])).sum())
        if n < nroll + 3:
            continue
        P = X[g, :n].copy()
        D = np.diff(P, axis=0)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        tip = P[-1]
        inward = np.array([SF.AXIS[0] - tip[0], SF.AXIS[1] - tip[1], 0.0])
        inward /= max(np.linalg.norm(inward), 1e-9)
        # (the face frame rolls away from the face: back, as it is blown [RZ]; rolled out sideways it flicked into a
        # wing at the jaw)
        if th[g] < 1.4:                  # (the face frame's ends are point cut and left: rolled, short pieces hooked)
            continue
        aim = inward
        for j in range(nroll):
            k = n - 1 - nroll + j
            ang = math.radians(deg) * (j + 1) / nroll
            d = D[k]
            t = aim - (aim @ d) * d
            tl = np.linalg.norm(t)
            if tl < 1e-6:
                continue
            D[k] = math.cos(ang) * d + math.sin(ang) * t / tl
        L = np.linalg.norm(np.diff(X[g, :n], axis=0), axis=1)     # (each segment keeps its own length)
        k0 = n - 1 - nroll
        P[k0 + 1:] = P[k0] + np.cumsum(D[k0:] * L[k0:, None], axis=0)
        X[g, :n] = P
        tail[g, k0 + 1:n] = True
        seglen[g, :n - 1] = L
    if col is not None and tail.any():
        X = push_valid(col, X, mask=tail)
        for g in np.where(tail.any(1))[0]:          # (pushed out, each rolled segment back to its own length)
            k = np.where(tail[g])[0]
            for j in k:
                v = X[g, j] - X[g, j - 1]
                X[g, j] = X[g, j - 1] + v / max(np.linalg.norm(v), 1e-12) * seglen[g, j - 1]
    return X


def push_valid(col, X, mask=None, margin_hh=1.0 / SF.HH_MM, steps=4):
    """The nodes that exist (and are in mask) moved out of the head and the body along the depth's gradient (push_out
    for cut guides: the colliders' grid lookups take no NaN)."""
    import torch
    X = X.copy()
    m = ~np.isnan(X[..., 0])
    m[:, 0] = False
    if mask is not None:
        m &= mask
    g, k = np.where(m)
    if not len(g):
        return X
    P = torch.tensor(X[g, k], dtype=torch.float64)
    for fn in (col.head_depth, col.body_depth):
        for _ in range(steps):
            P.requires_grad_(True)
            dep = fn(P)
            gr_, = torch.autograd.grad(dep.clamp(min=0).sum(), P)
            with torch.no_grad():
                n = gr_ / gr_.norm(dim=1, keepdim=True).clamp(min=1e-12)
                P = P - n * (dep + margin_hh).clamp(min=0)[:, None]
    X[g, k] = P.detach().numpy()
    return X


def _polar(X):
    ax, ay = SF.AXIS
    return np.arctan2(X[..., 0] - ax, -(X[..., 1] - ay)), np.hypot(X[..., 0] - ax, X[..., 1] - ay)


def _from_polar(X, th, r):
    ax, ay = SF.AXIS
    X = X.copy()
    X[..., 0] = ax + r * np.sin(th)
    X[..., 1] = ay - r * np.cos(th)
    return X


def _smooth_along(a, W=2):
    """a running max then a running mean along each strand (a push that rises and settles without a kink)."""
    S1 = a.shape[1]
    pad = np.pad(a, ((0, 0), (W, W)), mode='edge')
    mx = np.max(np.stack([pad[:, i:i + S1] for i in range(2 * W + 1)]), 0)
    pad = np.pad(mx, ((0, 0), (W, W)), mode='edge')
    return np.mean(np.stack([pad[:, i:i + S1] for i in range(2 * W + 1)]), 0)


def side_tuck(X, roots, keep, sides):
    """The hair rooted in front of the ear (|theta| in TUCK_TH, not the bangs) taken over the top of the ear and
    tucked behind it: from the temple down, swung back round the body axis (at constant r) until it lies TUCK_PAST
    behind the ear, and held there below it (the swing is carried down the strand)."""
    ok = ~np.isnan(X[..., 0])
    th, r = _polar(X)
    thr = SF.theta(roots)
    ear_th = math.atan2(EAR_X, -(SF.EAR_CANAL[1] - SF.AXIS[1])) + TUCK_PAST
    w = np.clip((TUCK_FROM_Z - X[..., 2]) / 0.12, 0, 1)
    for sd in sides:
        g = (np.sign(thr) == sd) & (np.abs(thr) > TUCK_TH[0]) & (np.abs(thr) < TUCK_TH[1]) & ~keep
        m = g[:, None] & ok
        need = np.where(m, np.maximum(ear_th - np.abs(th), 0) * w, 0.0)
        need = np.maximum.accumulate(np.nan_to_num(need), axis=1)
        need[:, 0] = 0
        Y = _from_polar(X, th + sd * need, r)
        X = np.where(m[..., None], Y, X)
        th, r = _polar(X)
    return X


def side_debulk(X, keep, style_path):
    """The side hair from the ear down laid in to the creator's side width (+SIDE_CLEAR_MM), the push smoothed along
    each strand: our sides stood 6 to 11 mm wider than the creator's at the jaw (hair-rebuild/sides/README.md)."""
    ok = ~np.isnan(X[..., 0])
    th, _ = _polar(X)
    side = (np.abs(th) > SIDE_TH[0]) & (np.abs(th) < SIDE_TH[1]) & ok & ~keep[:, None]
    import gzip
    d = json.load(gzip.open(style_path))
    C = np.concatenate([np.array(s['p']).reshape(-1, 3) for s in d['strands']])
    tc, _ = _polar(C)
    zb = np.arange(0.7, -1.6, -0.05)
    W = np.full(len(zb), np.nan)
    for i, z in enumerate(zb):
        m = (np.abs(tc) > SIDE_TH[0]) & (np.abs(tc) < SIDE_TH[1]) & (np.abs(C[:, 2] - z) < 0.05)
        if m.sum() > 10:
            W[i] = np.percentile(np.abs(C[m, 0]), 95)
    k = np.where(~np.isnan(W))[0]
    # below the creator's own side hair there is nothing to match: the limit opens out (SIDE_OPEN HH per HH of drop)
    # so hair longer than the creator's lies out over the shoulder instead of being pulled in against the neck
    W[k[-1]:] = W[k[-1]] + SIDE_OPEN * (zb[k[-1]] - zb[k[-1]:])
    k = np.where(~np.isnan(W))[0]
    W = np.interp(np.arange(len(zb)), k, W[k]) + SIDE_CLEAR_MM / SF.HH_MM
    lim = np.interp(-np.nan_to_num(X[..., 2]), -zb, W)
    ease = np.clip((0.6 - X[..., 2]) / 0.1, 0, 1)
    over = np.where(side, np.maximum(np.abs(X[..., 0]) - lim, 0) * ease, 0.0)
    over = _smooth_along(np.nan_to_num(over))
    over[:, 0] = 0
    X = X.copy()
    X[..., 0] -= np.sign(X[..., 0]) * over
    return X


# ------------------------------------------------------------------------------------------- hair over the shoulder
# [CC2] curlcentric.com, "Why does my hair flick out at the ends": ends that reach the shoulder are pushed up by it
#       ("Your shoulders push the ends up every time you move"): grow past the shoulders or cut above them; roll the
#       round brush under at the ends; weight keeps ends down.
# [BRIT] brit.co, "How to stop your lob from flipping out": "A flip at the ends can be the result of your hair hitting
#       your shoulders"; a blunt line, rounded under with a 1.25 inch round brush.
# [HF]  hairfinder.com, "Inverse layering": bulk underneath pushes the hair out along the curve of the neck; the under
#       layers cut shorter so the top layer lies in (side_debulk lays the sides in to the creator's width).
# [MAYA] Autodesk Maya nHair styling tips: passive colliders used as grooming tools (here the body is the comb's bed).
# The hair that passes over the shoulder lies on it: it follows the shoulder's curve without a bend where it meets it,
# and falls on in front of or behind it; the ends hang past it and turn under (SIDE_ROLL_DEG).
SHOULDER_TH = (0.8, 2.6)     # |theta| of the roots whose hair can pass over a shoulder (the sides, not the back's middle)
SHOULDER_TOP_Z = 0.15        # HH: the smoothing takes hold below the jaw, over 0.15 HH (the shoulder starts at z ~0)
SHOULDER_ITERS = 80
SHOULDER_K = 0.35            # the Laplacian step's weight (the strand's own curve, toward the mean of its neighbours)
SHOULDER_G = 0.004           # HH per step: gravity on each node, so the smoothing settles the strand down onto the
                             # shoulder instead of pulling it taut over it, CALIBRATED (no lift off the cape at 0.004)


def shoulder_flow(X, roots, keep, col, keep_shape=False):
    """Each side strand laid over the shoulder as a comb lays it: below the jaw, its nodes moved toward the mean of
    their neighbours and a little down, set back to their segment lengths from the root, and pushed out of the body
    and the head, SHOULDER_ITERS times. The strand bends where the body bends it, smoothly, and nowhere else.
    keep_shape (the Warden's waves): the smoothing acts on each node's move from where it was, not on the strand's
    curve, so the push off the shoulder spreads along the strand and the waves stay."""
    X = X.copy()
    th = np.abs(SF.theta(roots))
    g = np.where((th > SHOULDER_TH[0]) & (th < SHOULDER_TH[1]) & ~keep)[0]
    if not len(g):
        return X
    Y = X[g]
    ok = ~np.isnan(Y[..., 0])
    L0 = np.linalg.norm(np.diff(Y, axis=1), axis=-1)
    Y0 = Y.copy()
    for _ in range(SHOULDER_ITERS):
        w = np.clip((SHOULDER_TOP_Z - Y[..., 2]) / 0.15, 0, 1) * ok
        w[:, :2] = 0
        D = Y - Y0 if keep_shape else Y
        lap = np.zeros_like(Y)
        lap[:, 1:-1] = 0.5 * (D[:, :-2] + D[:, 2:]) - D[:, 1:-1]
        lap = np.nan_to_num(lap)                           # (a strand's last node: no neighbour below, no step)
        Y = Y + (SHOULDER_K * lap + np.array([0, 0, -SHOULDER_G])) * w[..., None]
        for j in range(1, Y.shape[1]):                     # (the chain set back to its lengths, from the root)
            d = Y[:, j] - Y[:, j - 1]
            n = np.linalg.norm(d, axis=-1, keepdims=True)
            Y[:, j] = np.where(ok[:, j, None], Y[:, j - 1] + d / np.maximum(n, 1e-12) * np.nan_to_num(L0[:, j - 1, None]), np.nan)
        Y = push_valid(col, Y, mask=w > 0)
    X[g] = push_valid(col, Y, steps=8)
    return X


# The Warden's sides solved again, as the back is: the shape passes (the swing back, the tuck, the width, the comb)
# set where her side hair sits at the temple and the ear, and the solver hangs the rest, with her rest wave, against
# her head, shoulders and collar. Measured before (2026-11-24): the passes left the side guides 20 mm (median) and
# 85 mm (90th pct) from the solve, against the back's 14 and 19, and neighbouring side strands 6 to 10 degrees apart
# 30 cm down (the back: 4.4); the back, solved and barely touched, is the part John called perfect.
SIDE_SOLVE_TH = 2.4          # |theta| of the roots solved again (the back beyond is left as it settled)
SIDE_PIN_Z = (0.15, 0.45)    # HH: the styled hold fades from full above 0.45 (the ear's middle) to none below 0.15 (the
                             # jaw): the tuck and the temple are held, the fall below hangs free, STYLE
SIDE_PIN_K = 1.0e-2          # N/m^2 per hair per node: the hold, CALIBRATED (holds the tucked hair within ~1 mm of
                             # its styled place against the weight of 40 cm hanging below it)


def side_resolve(X, roots, keep, d0, n_hairs, seg_hh, col, layer, kb0, iters, log=None):
    """The side guides (|theta| < SIDE_SOLVE_TH, not keep) settled again from their styled shape: held to it near the
    root (SIDE_PIN_Z), free below, with their rest wave (kb0), cut where they are cut now."""
    th = np.abs(SF.theta(roots))
    sel = (th < SIDE_SOLVE_TH) & ~keep
    if not sel.any():
        return X
    Xs = X[sel]
    G, S1 = Xs.shape[:2]
    ok = ~np.isnan(Xs[..., 0])
    n = ok.sum(1)
    F = Xs.copy()
    for g in range(G):                                     # (past the end: the last direction carried on)
        k = n[g]
        if k < S1:
            F[g, k:] = F[g, k - 1] + (F[g, k - 1] - F[g, k - 2]) * np.arange(1, S1 - k + 1)[:, None]
    D = np.diff(F, axis=1)[:, 1:]
    mask = (np.arange(S1 - 1)[None] < (n - 1)[:, None]).astype(float)
    w = SF.sm(SIDE_PIN_Z[0], SIDE_PIN_Z[1], F[:, 1:, 2])
    Y, _ = settle(roots[sel], d0[sel], D, n_hairs[sel], seg_hh, col, iters=iters, layer=layer.subset(sel), log=log,
                  kb0=kb0[sel], mask=mask, pin=(F[:, 1:], w, SIDE_PIN_K))
    Y[~ok] = np.nan
    X = X.copy()
    X[sel] = Y
    return X


SIDE_FORM_PASSES = 4          # comb passes within each wave lock, CALIBRATED (2: 4.6 degrees, 4: 3.9)


def side_form(X, roots, group, keep, col):
    """The side guides combed within their wave locks (comb_pass, its group the lock), a light relax, the back left as
    it is (the comb fading out between |theta| 2.2 and SIDE_SOLVE_TH + 0.2)."""
    Y = comb_pass(X, group, col, None, passes=SIDE_FORM_PASSES)
    Y = relax_kinks(push_valid(col, Y), col, None, keep=keep)
    w = (1.0 - SF.sm(2.2, SIDE_SOLVE_TH + 0.2, np.abs(SF.theta(roots))))[:, None, None]
    out = X + w * (np.nan_to_num(Y) - np.nan_to_num(X))
    out[np.isnan(X[..., 0])] = np.nan
    return out


# ------------------------------------------------------------------------------------------- the face-framing locks
# [FF-MS] modernsalon.com, "The S-wave step by steps": the front in its own sections, each wave worked "away from the
#         cheekbone to frame the face". [FF-MBG] mindbodygreen.com, clip-and-twist waves: "the two front sections",
#         twisted "away from your face", for "extra definition in the front pieces". [FF-DAV] davines.com, face-framing
#         layers: "two or three curls facing away from each side of your face". [FF-GLAM] glam.com, three-wrap
#         face-framing: away from the eyes, back in round the cheeks, out again round the jaw. [FF-BOU] boucleme.co.uk:
#         broken groupings read as "frizz, flyaways, and ... stringiness"; hair that holds together reads as a curl.
#         [FF-XG] Autodesk XGen Clump: clumping scaled along the hair from the root (a ramp), pulling the hairs to
#         the clump's centre; sub-clumps inside the primary ones.
# Her framing hair (the guides that pass in front of the ear at the cheekbone) hung as about 27 loose, straight guides
# a side: see-through wisps. Now each side's framing hair is gathered into FRAME_LOCKS locks: each lock one shape (the
# mean of its guides, waved away from the face from the cheekbone down), its guides drawn in to it from the cheekbone
# (loose there, a ribbon from mid-length), so the lock reads as one defined curl.
FRAME_LOCKS = 3              # locks a side [FF-DAV][FF-MBG]: 2-3 front sections
FRAME_Z0 = 0.4               # HH: where a lock is gathered and its wave starts: the cheekbone [FF-MS] (the eyes ~0.5)
FRAME_TH = 1.15              # |theta| at FRAME_Z0 in front of which hair is framing hair (the ear's front ~1.3)
FRAME_A_MM = 12.0            # mm: the lock's wave, STYLE (the creator's side waves 8 median, 14 at the 90th pct: the
                             # front pieces curled more defined than the mass)
FRAME_LAM = 0.45             # HH (10 cm): two or three bends over the lock's fall to the collar [FF-DAV][FF-GLAM]
FRAME_IN = 0.05             # HH (1 cm): the lock eases in toward the neck by its end, to lie on the collar, STYLE
FRAME_BACK = 0.08           # HH (1.8 cm): and back by its end: framing pieces fall back over the collar, STYLE
FRAME_TWIST = 0.7           # the curl's turn round the lock against its bend out (0: a flat S, 1: a round spiral)
                            # [FF-GHD] ghdhair.com: twist "the hair as you wrap"
FRAME_VARY = 0.15           # each lock's wavelength and height within +-15% [WS-GOOP]: alternate sizes "so the waves
                            # look less uniform"
FRAME_PHASE_JIT = 0.6       # rad: each lock's phase within +-0.6 (the creator's own lock jitter, WAVE_LOOSE)
FRAME_CLUMP = (0.35, 0.9)    # the pull of a guide to its lock's shape: at the cheekbone, and from mid-length on [FF-XG]


def frame_locks(X, roots, sel_from, seg_hh, col, n_locks=FRAME_LOCKS):
    """Gather the face-framing hair into defined locks (see FRAME_*). sel_from: the guides that may be framing hair.
    Returns the new X and a lock id per guide (-1: not framing hair)."""
    from scipy.cluster.vq import kmeans2
    X = X.copy()
    G, S1 = X.shape[:2]
    ok = ~np.isnan(X[..., 0])
    n = ok.sum(1)
    k0 = np.full(G, -1)
    for g in np.where(sel_from)[0]:
        below = np.where(ok[g] & (X[g, :, 2] < FRAME_Z0))[0]
        if len(below) and below[0] >= 2 and n[g] - below[0] >= 4:
            k0[g] = below[0]
    P0 = np.full((G, 3), np.nan)
    has = k0 >= 0
    P0[has] = X[has, k0[has]]
    fr = has & (np.abs(SF.theta(P0)) < FRAME_TH)
    fl = np.full(G, -1)
    nxt = 0
    for sd in (-1, 1):
        gi = np.where(fr & (np.sign(P0[:, 0] - SF.AXIS[0]) == sd))[0]
        if len(gi) < n_locks:
            continue
        _, lab = kmeans2(P0[gi][:, :2], n_locks, minit='++', seed=7)
        for k in range(n_locks):
            m = gi[lab == k]
            if not len(m):
                continue
            fl[m] = nxt
            # the lock's shape: its guides' falls from the cheekbone, averaged node by node, carried on past each end
            L = int(np.median(n[m] - k0[m]))
            F = np.zeros((len(m), L, 3))
            for j, g in enumerate(m):
                seg = X[g, k0[g]:n[g]]
                if len(seg) < L:
                    seg = np.vstack([seg, seg[-1] + (seg[-1] - seg[-2]) * np.arange(1, L - len(seg) + 1)[:, None]])
                F[j] = seg[:L] - seg[0]
            c0 = P0[m].mean(0)
            Cl = c0 + F.mean(0)
            for _ in range(30):                              # (its own wiggles smoothed out of it)
                Cl[1:-1] += 0.25 * (Cl[:-2] + Cl[2:] - 2 * Cl[1:-1])
            # waved away from the face [FF-MS][FF-GLAM]: out from the head's axis, then back in by the cheek, out at
            # the jaw; grown in over the first quarter wave, the last inch left looser [WS-TYMO]
            sarc = np.arange(L) * seg_hh
            u = sarc / max(sarc[-1], 1e-9)
            # it drapes: from the cheekbone the lock eases in toward the neck and back, so it lies on the collar and
            # falls behind the collarbone line instead of hanging plumb off the jaw (stiff, in profile)
            inw = Cl - np.array([SF.AXIS[0], SF.AXIS[1], 0.0]); inw[:, 2] = 0
            inw /= -np.maximum(np.linalg.norm(inw, axis=1, keepdims=True), 1e-9)
            Cl = Cl + (inw * FRAME_IN + np.array([0.0, FRAME_BACK, 0.0])) * SF.sm(0.0, 1.0, u)[:, None]
            out = -inw
            side = np.cross(np.array([0.0, 0.0, 1.0]), out)
            # each lock its own: its wavelength, height and phase a little apart, so the locks are not three copies
            rng = np.random.default_rng(31 + nxt)
            lam = FRAME_LAM * (1 + FRAME_VARY * (2 * rng.random() - 1))
            a_mm = FRAME_A_MM * (1 + FRAME_VARY * (2 * rng.random() - 1))
            ph = FRAME_PHASE_JIT * (2 * rng.random() - 1)
            amp = a_mm / SF.HH_MM * SF.sm(0.0, lam / 4, sarc) * np.clip((L - 1 - np.arange(L)) / 2.5, 0.4, 1)
            phi = 2 * np.pi * sarc / lam + ph
            # twisted as it is curled [FF-MBG][FF-GHD]: the bend turns round the lock (out, then round toward the
            # back), a loose spiral that reads from the side as well as from the front
            Cl = Cl + amp[:, None] * (np.sin(phi)[:, None] * out + FRAME_TWIST * (1 - np.cos(phi))[:, None] * side
                                      * np.sign(Cl[:1, 0] - SF.AXIS[0]) * -1)
            for j, g in enumerate(m):
                Lg = n[g] - k0[g]
                t = np.arange(Lg) / max(Lg - 1, 1)
                pull = FRAME_CLUMP[0] + (FRAME_CLUMP[1] - FRAME_CLUMP[0]) * SF.sm(0.0, 0.5, t)
                tgt = Cl[np.minimum(np.arange(Lg), L - 1)]
                if Lg > L:                                    # (a guide longer than its lock: it carries on its line)
                    tgt[L:] = Cl[-1] + (Cl[-1] - Cl[-2]) * np.arange(1, Lg - L + 1)[:, None]
                own = X[g, k0[g]:n[g]]
                Y = own + (tgt + (P0[g] - c0) * (1 - pull)[:, None] - own) * pull[:, None]
                # (back to the rod's segment length, from the cheekbone node)
                d = np.diff(np.vstack([X[g, k0[g] - 1], Y]), axis=0)
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                X[g, k0[g]:n[g]] = X[g, k0[g] - 1] + np.cumsum(d * seg_hh, 0)
            nxt += 1
    X = push_valid(col, X)
    return X, fl


BANG_OUT_Z = (0.9, 0.3)      # HH: the band (brow to cheekbone) over which the bangs are brushed out to the creator's
BANG_OUT_PCT = (10, 50)      # the bangs' spread matched at these percentiles of |x| per height (inner edge and middle)
BANG_OUT_GAIN = 1.0          # CALIBRATED: the full gap to the creator's sweep
BANG_OUT_ITERS = 3           # (the forehead and the rebuilt length take back part of each move)


def bang_out(X, bang, style_path, seg_hh, col):
    """The bangs brushed further out to the sides, off the eyes: at the eyes ours hung 1 to 1.5 cm in from the
    creator's sweep (|x| median 0.27 against 0.34 HH at z 0.6) and the tips covered the outer half of the eye, with
    the face showing between them (John, 2026-11-24: "brush more to the side so less gaps"). Each bang node is moved
    out by the gap at its height (the larger of the gaps at BANG_OUT_PCT), and the strand rebuilt at the rod's
    length from its root."""
    B = creator_bangs(style_path)
    C = np.concatenate([q for sd in B for q in B[sd]])
    ok = ~np.isnan(X[..., 0])
    P = X[bang][ok[bang]]
    zb = np.arange(BANG_OUT_Z[0], BANG_OUT_Z[1] - 0.01, -0.05)
    gap = np.zeros(len(zb))
    for i, z in enumerate(zb):
        mc = (np.abs(C[:, 2] - z) < 0.025) & (C[:, 1] < SF.AXIS[1])
        mo = (np.abs(P[:, 2] - z) < 0.025) & (P[:, 1] < SF.AXIS[1])
        if mc.sum() > 10 and mo.sum() > 10:
            ac, ao = np.abs(C[mc, 0] - SF.AXIS[0]), np.abs(P[mo, 0] - SF.AXIS[0])
            gap[i] = np.mean([np.percentile(ac, q) - np.percentile(ao, q) for q in BANG_OUT_PCT])   # (in or out)
    for _ in range(2):
        gap[1:-1] = 0.25 * gap[:-2] + 0.5 * gap[1:-1] + 0.25 * gap[2:]
    X = X.copy()
    for g in np.where(bang)[0]:
        n = ok[g].sum()
        z = X[g, :n, 2]
        lo = np.minimum.accumulate(z)                         # (held at its lowest so far: no swing back in)
        dx = BANG_OUT_GAIN * np.interp(-lo, -zb, gap) * np.sign(X[g, :n, 0] - SF.AXIS[0])
        dx = np.where(lo < zb[-1], np.interp(-zb[-1], -zb, gap) * np.sign(X[g, :n, 0] - SF.AXIS[0]), dx)
        Y = X[g, :n].copy(); Y[:, 0] += dx
        d = np.diff(Y, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        X[g, 1:n] = X[g, 0] + np.cumsum(d * seg_hh, 0)
    return push_valid(col, X, mask=bang[:, None] & ok)


BANG_BLEND_NODES = 6         # nodes (6 cm): the bang's end turned into the side hair's flow over this length
BANG_BLEND_W = 0.85          # how far the tip's direction turns to the side hair's (0 at the start of the blend), STYLE
BANG_BLEND_R = 0.12          # HH (2.6 cm): the side hair the bang's end is read against


def bang_blend(X, bang, col):
    """Curtain bangs flow into the side hair: brushed away from the face, the fringe's ends carry on into the hair at
    the temple rather than stopping on an edge over the eye (John, 2026-11-24: "the flow from the bangs to the sides
    should be smooth"). Over each bang's last BANG_BLEND_NODES nodes, its direction turns, more toward the tip, to the
    mean direction of the hair that is not bangs within BANG_BLEND_R of it; the end is rebuilt at the rod's length and
    pushed out of the head."""
    from scipy.spatial import cKDTree
    X = X.copy()
    ok = ~np.isnan(X[..., 0])
    n = ok.sum(1)
    d = np.diff(X, axis=1)
    o = ~bang
    P = 0.5 * (X[o, 1:] + X[o, :-1]).reshape(-1, 3)
    D = d[o].reshape(-1, 3)
    v = ~np.isnan(P[:, 0])
    P, D = P[v], D[v] / np.linalg.norm(D[v], axis=1, keepdims=True)
    tree = cKDTree(P)
    for g in np.where(bang & (n > BANG_BLEND_NODES + 2))[0]:
        k0 = n[g] - BANG_BLEND_NODES
        for j in range(k0, n[g]):
            seg = X[g, j] - X[g, j - 1]
            L = np.linalg.norm(seg)
            near = tree.query_ball_point(X[g, j - 1], BANG_BLEND_R)
            if not near:
                X[g, j] = X[g, j - 1] + seg
                continue
            flow = D[near].mean(0)
            flow /= max(np.linalg.norm(flow), 1e-12)
            w = BANG_BLEND_W * SF.sm(0.0, 1.0, (j - k0 + 1) / BANG_BLEND_NODES)
            u = (1 - w) * seg / L + w * flow
            X[g, j] = X[g, j - 1] + u / max(np.linalg.norm(u), 1e-12) * L
    return push_valid(col, X, mask=bang[:, None] & ok)


EDGE_Z = (0.55, 0.42, -0.1, -0.35)   # HH: the face edge filled in from below the bang's end (0.55 to 0.42) down past
                                     # the jaw (-0.1, out by -0.35: the shoulder drape stays as John approved it)
EDGE_REACH = 0.45                    # rad: the forward swing fades out over this much behind the face edge
EDGE_PCT = 2                         # the face edge: the 2nd percentile of |theta| of the hair at a height, per side


def _face_edge(P, zb):
    th, _ = _polar(P)
    E = np.full((2, len(zb)), np.nan)
    for i, z in enumerate(zb):
        m = np.abs(P[:, 2] - z) < 0.03
        for j, sd in enumerate((-1, 1)):
            mm = m & (np.sign(th) == sd)
            if mm.sum() > 20:
                E[j, i] = np.percentile(np.abs(th[mm]), EDGE_PCT)
    return E


def face_edge(X, sel, style_path, col):
    """The hair at the side of the face brought forward to the creator's face edge: under the bangs' ends the
    creator's hair falls in an unbroken sheet at |theta| 0.75 from the cheekbone to the jaw, ours stood back at 0.93
    to 1.1, and the temple and cheek showed between the bangs and the side hair (John, 2026-11-24: "there should be
    no gaps like these"). Each guide in sel is swung forward round the body axis (at constant r) by the gap at its
    height, in full at the edge and less further back (EDGE_REACH), so the sheet stretches forward instead of sliding."""
    import gzip
    d = json.load(gzip.open(style_path))
    C = np.concatenate([np.array(s['p']).reshape(-1, 3) for s in d['strands']])
    zb = np.arange(EDGE_Z[0], EDGE_Z[3] - 0.05, -0.05)
    ok = ~np.isnan(X[..., 0])
    Ec = _face_edge(C, zb)
    Eo = _face_edge(X[sel & ok.any(1)].reshape(-1, 3)[ok[sel & ok.any(1)].ravel()], zb)
    gap = np.nan_to_num(np.maximum(Eo - Ec, 0))
    for _ in range(3):
        gap[:, 1:-1] = 0.25 * gap[:, :-2] + 0.5 * gap[:, 1:-1] + 0.25 * gap[:, 2:]
    Eo = np.where(np.isnan(Eo), np.nanmax(Eo), Eo)
    th, r = _polar(X)
    z = np.nan_to_num(X[..., 2])
    w = SF.sm(EDGE_Z[0], EDGE_Z[1], z) * (1 - SF.sm(EDGE_Z[2], EDGE_Z[3], z))
    Y = X.copy()
    for j, sd in enumerate((-1, 1)):
        g = sel[:, None] & ok & (np.sign(th) == sd)
        zz, at = zb[::-1], (gap[j][::-1], Eo[j][::-1])
        dth = np.interp(z, zz, at[0])
        e = np.interp(z, zz, at[1])
        fall = np.clip(1 - (np.abs(th) - e) / EDGE_REACH, 0, 1)
        sw = np.where(g, dth * w * fall, 0.0)
        Y = np.where(g[..., None], _from_polar(X, th - sd * sw, r), Y)
    return push_valid(col, Y)


# Side-swept / curtain bangs as stylists finish them (John, 2026-11-24: "read more about side swept bangs"):
# [KENRA] kenraprofessional.com, "Fringe 101": the centre shortest, the sides "longer as they feed into the hair", a
#         vertical slide cut; "the deeper the cut, the more piecey it will be".
# [BTC-RW] behindthechair.com, "The cheekbone guide for cutting soft custom curtain bangs": slide cut, then softened
#         "creating no lines".
# [DAV] davines.com, "How to style curtain bangs": the fringe blends "gradually into the rest of the hair"; a round
#         brush under the bangs at the root, "curving the ends away from your face".
# Ours read stringy: the ends staggered 4.4 cm (p90, against the bangs beside them) as a deep point cut leaves them,
# and the strands drawn finer at the front showed the forehead between them.
BANG_LINE_K = 10             # bangs a guide's cut line is read from (its nearest by root)
BANG_LINE_TOL = 0.025        # HH (5 mm): a slide-cut line's own unevenness; a bang longer than its line by more is cut
BANG_FAN_SIGMA = 0.035       # HH (8 mm, in x): each bang's shape blended with the bangs rooted around it, so the fan opens
                             # evenly from the part instead of in two groups with the forehead between (the inner
                             # bangs fell to the brow, the outer swept off to the temple)
BANG_FAN_PASSES = 1
BANG_SHEET_PASSES = 3        # comb passes run through each side's bangs alone, so the side lies as one sheet [BTC-RW]
BANG_EDGE_HH = 0.08          # HH (1.8 cm): the hair rooted this close outside the bangs' section is combed with them,
                             # so the section blends into the hair beside it [DAV] (the forehead showed in a wedge at
                             # each side of the section, where the bangs swept out and the hair beside them went back)


def bang_finish(X, lengths, bang, roots, col, free=None, comb=True, cut=True):
    """The bangs finished as one swept sheet (comb): each side's fan evened out from the part and combed through, with
    the hair rooted just outside the section (free: the guides that may be combed with them); then (cut) slide-cut on
    one smooth line, a bang hanging past the line its neighbours make cut back to it. Returns X and lengths."""
    from scipy.spatial import cKDTree
    side = np.where(roots[:, 0] < SF.AXIS[0], 1, 2)
    d, _ = cKDTree(roots[bang]).query(roots)
    edge = ~bang & (d < BANG_EDGE_HH) & (roots[:, 1] < SF.AXIS[1])
    if free is not None:
        edge &= free
    group = np.where(bang | edge, side, -np.arange(len(roots)) - 10)
    # the fan evened out in root space: each guide takes the Gaussian-weighted mean shape (from its root) of the
    # guides of its side rooted round it
    ok = ~np.isnan(X[..., 0])
    for _ in range(BANG_FAN_PASSES if comb else 0):
        Y = X.copy()
        for sd in (1, 2):
            gi = np.where(bang & (side == sd))[0]          # (the bangs only: the hair beside them goes back)
            if len(gi) < 3:
                continue
            F = X[gi] - roots[gi][:, None]
            last = ok[gi].sum(1) - 1
            for j in range(len(gi)):                       # (each carried on past its end, to blend over any length)
                F[j, last[j] + 1:] = F[j, last[j]]
            D2 = (roots[gi][:, None, 0] - roots[gi][None, :, 0]) ** 2   # (across the fan only: its x)
            W = np.exp(-D2 / (2 * BANG_FAN_SIGMA ** 2))
            W /= W.sum(1, keepdims=True)
            M = np.einsum('ij,jsk->isk', W, F)
            for j, g in enumerate(gi):
                Y[g, :last[j] + 1] = roots[g] + M[j, :last[j] + 1]
                d = np.diff(Y[g, :last[j] + 1], axis=0)    # (back to the rod's segment length)
                L = np.linalg.norm(np.diff(X[g, :last[j] + 1], axis=0), axis=1, keepdims=True)
                d *= L / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                Y[g, 1:last[j] + 1] = X[g, 0] + np.cumsum(d, 0)
        X = push_valid(col, Y)
    if comb:
        X = comb_pass(X, group, col, passes=BANG_SHEET_PASSES)
    ok = ~np.isnan(X[..., 0])
    n = ok.sum(1)
    lengths = lengths.copy()
    if not cut:
        return X, lengths
    for sd in (1, 2):
        gi = np.where(bang & (side == sd) & (n > 4))[0]
        if len(gi) <= BANG_LINE_K:
            continue
        E = X[gi, n[gi] - 1, 2]
        _, nn = cKDTree(roots[gi]).query(roots[gi], BANG_LINE_K + 1)
        line = np.median(E[nn], 1)
        for j, g in enumerate(gi):
            if E[j] < line[j] - BANG_LINE_TOL:
                below = np.where(ok[g] & (X[g, :, 2] < line[j]))[0]
                k = max(4, below[0])
                X[g, k + 1:] = np.nan
                lengths[g] = min(lengths[g], k)
    return X, lengths


TOP_SMOOTH_Z = (0.1, 0.35)   # HH: the side hair smoothed in full above 0.35 (the cheekbone), none below 0.1: the waves
                             # start at the jaw, as the creator's do (its bends above z 0.2 are 1.9 to 2.6 mm, ours
                             # were 4 to 6; John, 2026-11-24: "reduce curl at the tops, too bumpy at the temples")
TOP_SMOOTH_ITERS = 30
TOP_SMOOTH_K = 0.3


def top_smooth(X, sel, col):
    """The bumps taken out of the side hair over the temples and down to the cheekbone: Laplacian smoothing along
    each strand in sel, weighted by height (TOP_SMOOTH_Z), the root held; then rebuilt at the rod's length from the
    root and kept out of the head."""
    X = X.copy()
    ok = ~np.isnan(X[..., 0])
    for g in np.where(sel)[0]:
        n = ok[g].sum()
        if n < 5:
            continue
        P = X[g, :n].copy()
        L = np.linalg.norm(np.diff(P, axis=0), axis=1, keepdims=True)
        w = SF.sm(TOP_SMOOTH_Z[0], TOP_SMOOTH_Z[1], P[:, 2])[1:-1, None]
        for _ in range(TOP_SMOOTH_ITERS):
            P[1:-1] += TOP_SMOOTH_K * w * (P[:-2] + P[2:] - 2 * P[1:-1])
        d = np.diff(P, axis=0)
        d *= L / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        X[g, 1:n] = X[g, 0] + np.cumsum(d, 0)
    return push_valid(col, X, mask=sel[:, None] & ok)


def recut(X, lengths, cut_z=None, keep=None):
    """Cut again on the hair as it finally lies: the cut is made on the first settle, and the hair the body then holds
    out (the cape over the shoulders) hangs past the length line by the time it rests. And any guide that climbs back
    up is cut where it turns."""
    X = X.copy()
    lengths = lengths.copy()
    for g in range(len(X)):
        z = X[g, :, 2]
        ok = ~np.isnan(z)
        if ok.sum() <= CLIMB_FROM:
            continue
        lo = np.minimum.accumulate(np.where(ok, z, np.inf))
        up = np.where(ok & (z - lo > CLIMB_HH) & (np.arange(len(z)) >= CLIMB_FROM))[0]
        if len(up):
            k = int(np.argmin(z[:up[0]]))                 # (the turn: the lowest point before the climb)
            n = max(3, k)
            X[g, n + 1:] = np.nan
            lengths[g] = min(lengths[g], n)
    if cut_z is None:
        return X, lengths
    endz = cut_z(SF.theta(np.nan_to_num(X[:, 0])))
    for g in range(len(X)):
        if keep is not None and keep[g]:
            continue
        below = np.where(X[g, 1:, 2] < endz[g])[0]
        if len(below) and below[0] + 1 < lengths[g] + 1:
            n = max(3, below[0])
            X[g, n + 1:] = np.nan
            lengths[g] = min(lengths[g], n)
    return X, lengths


def build_groom(surf, F, body, n_guides=2400, style="ranger", iters=1500, log=None, bangs=True, reuse=None, trace=None):
    """style: "ranger" (straight, curtain bangs, the approved length line) or "warden" (the natural wave, no bangs,
    cut on the creator's own length line, which needs trace). reuse: a previous groom.npz. The guides do not act on each other (no hair-hair contact; the layer is a fixed
    function of the roots), so a guide with the same root and the same style as in that groom settles to the same
    place: it is copied, and only the rest (here the bangs, while they are being styled) are solved."""
    wavy = style == "warden"
    global SPLIT
    SPLIT = WARDEN_SPLIT if wavy else RANGER_SPLIT
    if wavy:
        assert trace, "the Warden is cut on the creator's length line: pass --trace"
    cut_z = creator_cut(trace) if wavy else ranger_cut_z
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
    need = roots[:, 2] - cut_z(SF.theta(roots)) + 0.6
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
            if 'combed' in P0.files:      # (a guide combed in that groom was placed, not settled: never copy it)
                hit &= P0['combed'][j] == 0
            solve = bang | ~hit
            P = {k: np.zeros((len(idx),) + P0[k].shape[1:], P0[k].dtype) for k in ('X_uncut', 'X', 'lengths')}
            for k in P:
                P[k][hit] = P0[k][j[hit]]
        print('reuse: %d of %d guides copied' % ((~solve).sum(), len(idx)) if same else 'reuse: groom differs, full solve')

    def settle_some(D, mask=None, its=None):
        X = np.zeros((len(idx), S + 1, 3)); Dn = D.copy()
        if (~solve).any():
            X[~solve] = P['X_uncut'][~solve] if mask is None else np.nan_to_num(P['X'][~solve])
        if not solve.any():
            return X, Dn
        m = None if mask is None else mask[solve]
        Xs, Ds = settle(roots[solve], d0[solve], D[solve], n_hairs[solve], seg_hh, col, iters=its or iters, layer=layer.subset(solve),
                        log=log, kb0=None if kb0 is None else kb0[solve], mask=m)
        X[solve] = Xs; Dn[solve] = Ds
        return X, Dn
    field = creator_field(trace) if trace else None
    if bangs or field is not None:
        if not same:
            P = {'X_uncut': np.zeros((len(idx), S + 1, 3)), 'X': np.zeros((len(idx), S + 1, 3)),
                 'lengths': np.zeros(len(idx), int)}
        P = {k: np.array(P[k]) for k in ('X_uncut', 'X', 'lengths')}
    if bangs:
        # the bangs are combed, not solved: their paths come from the combing pass and are never settled
        if field is not None:
            # traced: laid along the creator's bangs in order across the fan, kept clear of the head
            Xb, bang_len = trace_bangs(roots[bang], bt[bang], bside[bang], creator_bangs(trace), seg_hh, S)
            Xb = push_out(col, Xb, layer.subset(bang))
        else:
            Xb = comb_bangs(surf, col, roots[bang], d0[bang], sweep[bang], comb[bang], seg_hh, S, field)
        solve &= ~bang
        P['X_uncut'][bang] = Xb
    if field is not None:
        # the face-framing hair (in front of the temples) is combed too, along the creator's framing locks onto
        # the cheek, then straight down in front of the shoulder; cut by the style's line like the rest
        thr = SF.theta(roots)
        frame = ~bang & (np.abs(thr) < FRAME_TH) & (roots[:, 2] > 0.55)
        down = np.tile([0.0, 0.0, -1.0], (frame.sum(), 1))
        lie_f = np.clip(layer.root_cum[frame], 1.0 / SF.HH_MM, 10.0 / SF.HH_MM)
        if wavy:
            # the Warden's framing hair continues the curtain outward: laid in rank order (by its angle round the
            # head) along the outer part of the creator's bangs, which run down the edge of the face to the cheek.
            # Combed on its own it lay out over the side of the head and left the face's edge bare and sheer.
            cb = creator_bangs(trace)
            outer = {k: v[int(len(v) * (1 - FRAME_OUTER)):] for k, v in cb.items()}
            fside = np.where(roots[frame, 0] >= SF.AXIS[0], 1.0, -1.0)
            Xf, _ = trace_bangs(roots[frame], np.abs(thr[frame]), fside, outer, seg_hh, S)
            Xf = push_out(col, Xf, layer.subset(frame))
        else:
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
    endz = cut_z(SF.theta(X[:, -1]))
    th = SF.theta(roots)
    lock = np.where(np.abs(th) < SPLIT, np.where(roots[:, 0] < SF.AXIS[0], 1, 2), 0)
    clump = np.zeros(len(idx), int)
    if wavy:
        # the ends: each wave lock ends at its own length, up to WAVE_END_JIT above the creator's longest, so the hem is
        # the creator's uneven, pointed lock ends instead of one blunt line
        clump = wave_locks(roots, lock)
        endz = endz + WAVE_END_JIT * np.random.default_rng(13).random(clump.max() + 1)[clump]
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
    if wavy:
        # cut as wavy hair is cut, dry: the wave takes up length, so each guide is cut where its waved shape (the
        # formula's, for its hang) reaches the line, not where the straight fall does (twice: the hang moves with it)
        lam, ph = wave_params(roots, clump)
        for _ in range(2):
            Yw = wave_rest(X, lengths, lam, ph, seg_hh)[1]
            for g in range(len(idx)):
                below = np.where(Yw[g, 1:, 2] < endz[g])[0]
                lengths[g] = max(3, below[0]) if len(below) else S
    mask = (np.arange(S)[None] < lengths[:, None]).astype(float)
    if (~solve).any() and not wavy:        # (the wavy cut is recomputed: every wavy guide is solved again anyway)
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
    if wavy:
        # the natural wave: every guide not combed is solved again with its rest wave (none is copied: the wave is
        # new to each), started from the formula's settled shape; the combed framing hair gets that shape laid on
        kb0, Ys, ds = wave_rest(X, lengths, lam, ph, seg_hh)
        solve = ~bang & ~frame
        D = ds[:, 1:].copy()
        if frame.any():
            P['X'][frame] = push_out(col, Ys[frame], layer.subset(frame))
    # (the wave starts at the formula's equilibrium, so its solve only polishes it and clears the contacts)
    X2, D = settle_some(D, mask, its=WAVE_ITERS if wavy else None)
    t2 = time.time()
    X2[:, 1:][mask == 0] = np.nan
    if bangs or frame.any():
        # layered as cut: the face-framing hair over the hair at the sides, the bangs over both
        X2 = layer_over(X2, frame, seg_hh, under=~bang & ~frame)
        X2 = layer_over(X2, bang, seg_hh)
    if frame.any() and not wavy:
        # below the jaw the face-framing hair falls, without swinging back in toward the neck
        X2 = smooth_fall(X2, frame, body)
    if trace and not wavy:
        # the Ranger's sides flow outward below the cheek, as the creator's do
        X2 = creator_flare(X2, trace, surf)
    if trace:
        # the stress test's faults: off the creator's cap, then no sharp bends (the combed bangs keep their shape)
        cap = CreatorCap(trace)
        X2 = onto_cap(X2, cap)
        X2 = relax_kinks(X2, col, cap, keep=bang)
        # the comb: strays back into their lock, crossing neighbours laid parallel, roots along the scalp
        fl = np.full(len(roots), -1)                 # (the face-framing locks: frame_locks, the Warden)
        lk = np.where(bang, np.where(roots[:, 0] < SF.AXIS[0], 3, 4), lock)      # (the bangs combed among themselves)
        X2 = comb_pass(X2, lk * 1000 + clump, col, cap)
        # the front of the side hair held back to the creator's (combed forward over the shoulders, it hid the jaw)
        X2 = creator_front(X2, trace, col, keep=bang)
        X2 = comb_pass(X2, lk * 1000 + clump, col, cap, passes=2)        # (a light comb over the swung strands)
        X2 = relax_kinks(X2, col, cap, keep=bang)
        # the styling: layered and face framed, point cut, the ends rolled (see STYLES)
        st = STYLES['warden' if wavy else 'ranger']
        X2, lengths = style_cut(X2, lengths, roots, bang, st)
        X2 = push_valid(col, X2)
        X2 = relax_kinks(X2, col, cap, keep=bang)
        X2 = end_roll(X2, roots, bang, st['roll_deg'], col)
        # the sides: tucked behind the ear (STYLES tuck), laid in to the creator's width, the ends rolled under
        X2 = side_tuck(X2, roots, bang, st['tuck'])
        X2 = side_debulk(X2, bang, trace)
        X2 = push_valid(col, X2)
        # the sides combed and laid over the shoulder (see SHOULDER_*): a light comb, then each side strand smoothed
        # along the body as it passes over the shoulder, then the ends turned under
        X2 = comb_pass(X2, lk * 1000 + clump, col, cap, passes=2)
        if wavy:
            # the Warden's sides hung by the solver, as her back is (see SIDE_SOLVE_TH): styled at the top, free below
            X2 = side_resolve(X2, roots, bang, d0, n_hairs, seg_hh, col, layer, kb0, WAVE_ITERS, log)
        else:
            X2 = shoulder_flow(X2, roots, bang, col)
            X2 = relax_kinks(X2, col, cap, keep=bang)
        X2 = end_roll(X2, roots, bang, SIDE_ROLL_DEG, col)        # (the ends under, the back's as John approved them)
        # (relax ends on the cap, which runs down through her shoulders and cape: the last word is the body's, and a
        # relax without the cap takes out the bends that push leaves)
        X2 = push_valid(col, X2)
        X2 = relax_kinks(X2, col, None, keep=bang)
        if wavy:
            # then formed: the side hair combed within its wave locks, as a stylist rakes waves into ribbons with the
            # fingers [WS-AS] (re-solved, neighbouring side strands sat 6.6 degrees apart 30 cm down, the back 4.4,
            # and read as messy; combed 4 passes: 3.9, crossings 114 to 92, the creator's 90)
            X2 = side_form(X2, roots, lk * 1000 + clump, bang, col)
            # the face-framing hair gathered into a few defined locks, waved away from the face (see FRAME_*); its
            # children blend within their lock
            X2, fl = frame_locks(X2, roots, np.abs(SF.theta(roots)) < SIDE_SOLVE_TH, seg_hh, col)
            X2 = relax_kinks(X2, col, None, keep=bang & (fl < 0))
            clump = np.where(fl >= 0, 900 + fl, clump)
            # the side of the face filled to the creator's edge under the bangs (no gap at the temple and cheek)
            X2 = face_edge(X2, ~bang & (np.abs(SF.theta(roots)) < SIDE_SOLVE_TH), trace, col)
            X2 = relax_kinks(X2, col, None, keep=bang & (fl < 0))
            # the bangs combed into one fan with the hair beside them (see bang_finish), then brushed to the creator's
            # sweep, off the eyes and over the brow
            X2, _ = bang_finish(X2, lengths, bang & (fl < 0), roots, col, free=fl < 0, cut=False)
            for _ in range(BANG_OUT_ITERS):
                X2 = bang_out(X2, bang & (fl < 0), trace, seg_hh, col)
            # and the bangs' ends carried on into the side hair, so the fringe flows into the sides
            X2 = bang_blend(X2, bang & (fl < 0), col)
            # and finished as one sheet on a smooth slide-cut line (see bang_finish)
            X2, lengths = bang_finish(X2, lengths, bang & (fl < 0), roots, col, comb=False)
            # the tops smoothed: no waves over the temples, they start at the jaw (see TOP_SMOOTH_Z)
            X2 = top_smooth(X2, ~bang & (np.abs(SF.theta(roots)) > 0.6) & (np.abs(SF.theta(roots)) < SIDE_SOLVE_TH), col)
    if trace and not wavy:
        # one length round the sides and back, on the creator's back length (RANGER_LEN_Z): no side cut
        X2, lengths = recut(X2, lengths, ranger_cut_z, keep=bang)
    elif trace:
        X2, lengths = recut(X2, lengths)
    # the locks the children blend within: 0 the tucked back, 1 / 2 the face-framing hair right / left of the part,
    # 3 / 4 the bangs right / left
    lock[bang] = np.where(roots[bang, 0] < SF.AXIS[0], 3, 4)
    if trace:
        # (a framing lock blends its children within itself, its bang and framing guides alike)
        lock = np.where(fl >= 0, np.where(roots[:, 0] < SF.AXIS[0], 1, 2), lock)
    return dict(roots=roots, idx=idx, X=X2, lock=lock, clump=clump, combed=(bang | frame).astype(np.int8), bang=bang.astype(np.int8), X_uncut=X, lengths=lengths, n_hairs=n_hairs, total=total,
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
    bodies = json.load(gzip.open(head))['bodies']
    style0 = a[a.index('--style') + 1] if '--style' in a else 'ranger'
    who = a[a.index('--body') + 1] if '--body' in a else style0
    # the character's own body (her collar, cape and back), as the game and the stress test drape her hair on;
    # --body mannequin for the bare mannequin
    body = bodies['mannequin'] if who == 'mannequin' else character_body(bodies[who], bodies['mannequin'])
    log = []
    reuse = a[a.index('--reuse') + 1] if '--reuse' in a else None
    trace = a[a.index('--trace') + 1] if '--trace' in a else None
    style = a[a.index('--style') + 1] if '--style' in a else 'ranger'
    Gm = build_groom(surf, F, body, n_guides=n, style=style, log=log, reuse=reuse, trace=trace)
    np.savez_compressed(os.path.join(out, 'groom.npz'), **{k: v for k, v in Gm.items() if isinstance(v, np.ndarray)})
    rep = dict(guides=int(len(Gm['idx'])), hairs=int(Gm['total']), hairs_per_guide=round(float(Gm['n_hairs'][0]), 1),
               elastogravity_cm=round(ELASTOGRAVITY_M * 100, 2), layer_max_mm=round(Gm['layer_max_mm'], 2),
               mean_length_cm=round(float(np.mean(Gm['lengths']) * SEG_M * 100), 1), solve_s=[round(x, 1) for x in Gm['times']],
               energy=log)
    json.dump(rep, open(os.path.join(out, 'groom.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
