"""A 3D static solve for the locks of a ponytail: each lock is one Kirchhoff rod, clamped at the tie, free at the tip.

Why: a tail drawn by offsetting hairs from one centre line is one rope with a texture on it. Real hair below a tie is
a bundle of locks, each hanging and curling on its own, pushed apart by its neighbours, so the locks part as they fall.

The model (Kirchhoff's rod; Bertails et al. 2006, "Super-Helices for Predicting the Dynamics of Natural Hair": hair as
an inextensible rod with natural curvature and twist, bent by its weight; Bergou et al. 2008, "Discrete Elastic Rods":
the twist-free Bishop frame carried along the centre line):

  - A lock is M straight segments of fixed length l (inextensible). Its frame R_i = (d1, d2, t) is carried from the
    clamped root: R_{i+1} = exp(l [w_i]) R_i, x_{i+1} = x_i + l t_i.
  - Constitutive law at node i (isotropic section): the bend is the rest bend plus the moment over the stiffness,
        w_i = R_i k0_i + M_perp / EI + (M . t) t / GJ.
    k0 is the natural curl in the lock's own frame, k0 (cos(tau0 s + phi), sin(tau0 s + phi), 0): with no load the
    lock is a helix of curvature k0 and torsion tau0. The curl grows in from zero over CURL_FROM_MM below the tie.
  - Statics of a rod with a free tip: the moment at a node is the moment of every force below it,
        M_i = sum_{j>i} (x_j - x_i) x F_j,
    so no shooting is needed. F is gravity (w per mm), contact between locks (a penalty when two locks' nodes are
    nearer than the sum of their radii: locks keep a lock's thickness apart) and contact with the head.
  - Solved by fixed-point iteration on w with under-relaxation, the load stepped in from zero and each joint's step scaled
    by its elastic stiffness over the stiffness of the load below it (a long lock is far past its bending length
    (EI / w)^(1/3), about 45 mm, so the plain iteration overshoots and never settles).

Stiffness and weight (per lock of n hairs; hairs in a lock slide on each other, so stiffnesses add):
  E 3.9 GPa and density 1.3 g/cm3 (Robbins, Chemical and Physical Behavior of Human Hair, ch. 9), 70 um diameter:
  EI = n x 4.6e-3 N mm2 ... per hair 4.6e-9 N m2; w = n x 4.9e-8 N/mm. GJ = 0.7 EI (DRAFT; hair's torsional modulus
  is well under its Young's, Robbins ch. 9).
Natural curl: helix radius 12 mm, pitch 100 mm (DRAFT, read off the reference photo's loose waves against its tie).

    rods = solve(roots, frames, lengths, n_hairs, radius, head=None, seed=0)   ->  dict(x, R, s, report)

Trouble looks like: report['residual_mm'] over 0.5 (not converged: raise ITERS), 'min_gap_ratio' under 0.5 (locks
inside each other: raise CONTACT_K), 'rank_swaps' high (locks crossed round the axis).
"""
import numpy as np
from scipy.spatial import cKDTree

E_GPA, RHO, DIAM_MM = 3.9, 1.3e-6, 0.07                 # kg/mm3
EI_HAIR = E_GPA * 1e3 * np.pi * (DIAM_MM / 2) ** 4 / 4   # N mm2 (GPa = 1e3 N/mm2)
W_HAIR = RHO * np.pi * (DIAM_MM / 2) ** 2 * 9.81         # N per mm
GJ_OVER_EI = 0.7        # DRAFT
CURL_R_MM, CURL_PITCH_MM = 12.0, 100.0                  # DRAFT (reference photo)
CURL_FROM_MM = 80.0     # DRAFT: no curl at the tie, all of it this far below
SEG_MM = 5.0
ITERS, RAMP, RELAX, STEP_MAX = 4000, 1000, 0.02, 0.35
CONTACT_K = 3.0e-4      # N per mm of overlap (about a segment's weight per mm)
HEAD_K = 6.0e-4
G_DIR = np.array([0.0, 0.0, -1.0])


def _rot(v):
    """Rodrigues: rotation matrices for rotation vectors v (n, 3)."""
    th = np.linalg.norm(v, axis=1)
    k = v / np.maximum(th, 1e-12)[:, None]
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def _build(x0, R0, W, l):
    n, M, _ = W.shape
    X = np.zeros((n, M + 1, 3)); Rs = np.zeros((n, M, 3, 3))
    X[:, 0] = x0; R = R0.copy()
    for i in range(M):
        R = _rot(W[:, i]) @ R                           # (the bend at node i turns the frame, then the segment runs)
        Rs[:, i] = R
        X[:, i + 1] = X[:, i] + l[:, None] * R[:, :, 2]
    return X, Rs


def solve(roots, frames, lengths, n_hairs, radius, head=None, seed=0, curl=(CURL_R_MM, CURL_PITCH_MM), log=None):
    """roots (n, 3) mm; frames (n, 3, 3) columns d1, d2, t at the tie; lengths (n,) mm; n_hairs (n,); radius (n,) mm,
    a lock's radius at the tie; head: f(points mm) -> (depth inside mm, outward normal) or None."""
    rng = np.random.default_rng(seed)
    n = len(roots)
    M = int(np.ceil(lengths.max() / SEG_MM))
    l = lengths / M
    s = (np.arange(M)[None, :] + 0.5) * l[:, None]
    EI, w = n_hairs * EI_HAIR, n_hairs * W_HAIR
    Rh, c = curl[0], curl[1] / (2 * np.pi)
    k0 = Rh / (Rh * Rh + c * c) * (0.85 + 0.3 * rng.random(n))
    tau0 = c / (Rh * Rh + c * c) * (0.9 + 0.2 * rng.random(n))
    phi = 0.8 * rng.standard_normal(n)                  # (waves set together run near in step; DRAFT)
    grow = np.clip(s / CURL_FROM_MM, 0, 1); grow = grow * grow * (3 - 2 * grow)
    K0 = (k0[:, None] * grow)[:, :, None] * np.stack([np.cos(tau0[:, None] * s + phi[:, None]), np.sin(tau0[:, None] * s + phi[:, None]), np.zeros_like(s)], 2)
    rad = radius[:, None] * np.sqrt(np.clip(1 - 0.7 * (np.arange(M + 1)[None, :] / M) ** 2, 0.1, 1))    # (thinning to the tip)
    W = np.zeros((n, M, 3))
    # each joint's own step: its elastic stiffness over that plus the stiffness the load below it adds (the weight's
    # and the contacts' levers), so the heavy top of a lock moves slowly and the light tip freely
    below = (lengths[:, None] - s + 0.5 * l[:, None])
    kel = (EI / l)[:, None]
    # (RELAX is small because every joint steps at once: each alone would correct the whole imbalance)
    alpha = (RELAX * kel / (kel + 0.5 * w[:, None] * below ** 2 + 0.3 * CONTACT_K * below ** 2))[:, :, None]
    X, Rs = _build(roots, frames, W, l)
    lock_of = np.repeat(np.arange(n), M + 1)
    res = 0.0
    for it in range(ITERS):
        lam = min(1.0, (it + 1) / RAMP)
        F = np.zeros((n, M + 1, 3))
        F[:, 1:] += lam * (w * l)[:, None, None] * G_DIR
        flat = X.reshape(-1, 3)
        pairs = cKDTree(flat).query_pairs(2 * rad.max(), output_type='ndarray')
        if len(pairs):
            pairs = pairs[lock_of[pairs[:, 0]] != lock_of[pairs[:, 1]]]
            d = flat[pairs[:, 0]] - flat[pairs[:, 1]]
            dist = np.linalg.norm(d, axis=1)
            over = rad.reshape(-1)[pairs[:, 0]] + rad.reshape(-1)[pairs[:, 1]] - dist
            m = over > 0
            f = (CONTACT_K * over[m] / np.maximum(dist[m], 1e-6))[:, None] * d[m]
            Ff = F.reshape(-1, 3)
            np.add.at(Ff, pairs[m, 0], f); np.add.at(Ff, pairs[m, 1], -f)
        if head is not None:
            dep, nrm = head(flat, rad.reshape(-1))
            F += (HEAD_K * np.maximum(dep, 0))[:, None].reshape(n, M + 1, 1) * nrm.reshape(n, M + 1, 3)
        F[:, 0] = 0
        # the moment at node i (the joint at X_i, i = 0..M-1) of the forces on the nodes below it
        S0 = np.cumsum(F[:, ::-1], 1)[:, ::-1]                              # sum_{j>=i} F_j
        S1 = np.cumsum(np.cross(X, F)[:, ::-1], 1)[:, ::-1]
        Mo = (S1[:, 1:] - np.cross(X[:, :-1], S0[:, 1:]))                   # (n, M, 3)
        t = Rs[:, :, :, 2]
        Mt = (Mo * t).sum(2, keepdims=True) * t
        Wt = l[:, None, None] * (np.einsum('nmij,nmj->nmi', Rs, K0) + (Mo - Mt) / EI[:, None, None] + Mt / (GJ_OVER_EI * EI[:, None, None]))
        wn = np.linalg.norm(Wt, axis=2, keepdims=True)
        Wt *= np.minimum(1.0, STEP_MAX / np.maximum(wn, 1e-12))
        eq = float(np.linalg.norm(Wt - W, axis=2).max())                     # how far from balance, rad per joint
        cool = 1.0 if it < RAMP else 0.05 + 0.95 * (1 - (it - RAMP) / max(1, ITERS - RAMP)) ** 2     # (settling: steps shrink)
        W = W + cool * alpha * (Wt - W)
        Xn, Rs = _build(roots, frames, W, l)
        res = float(np.abs(Xn - X).max()); X = Xn
        if log and it % 300 == 0:
            log('it %d load %.2f move %.3f mm' % (it, lam, res))
    # gauges
    ax = frames[:, :, 2].mean(0); ax /= np.linalg.norm(ax)
    c0 = roots.mean(0)
    e1 = np.cross(ax, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)

    def rank(P):
        v = P - P.mean(0)
        return np.argsort(np.argsort(np.arctan2(v @ e2, v @ e1)))
    r0 = rank(roots)
    outer = np.linalg.norm(roots - c0, axis=1) > np.median(np.linalg.norm(roots - c0, axis=1))
    swaps = {}
    for fr in (0.25, 0.5, 0.75):
        j = int(fr * M)
        a, b = r0[outer], rank(X[:, j])[outer]
        oa = np.argsort(a); bb = np.argsort(np.argsort(b[oa]))
        d = np.abs(bb - np.arange(len(bb))); d = np.minimum(d, len(bb) - d)
        swaps[fr] = int((d > 2).sum())
    flat = X.reshape(-1, 3)
    pairs = cKDTree(flat).query_pairs(2 * rad.max(), output_type='ndarray')
    pairs = pairs[lock_of[pairs[:, 0]] != lock_of[pairs[:, 1]]]
    gap = np.linalg.norm(flat[pairs[:, 0]] - flat[pairs[:, 1]], axis=1) / (rad.reshape(-1)[pairs[:, 0]] + rad.reshape(-1)[pairs[:, 1]])
    width = [float(np.ptp((X[:, int(fr * M)] - X[:, int(fr * M)].mean(0)) @ e1)) for fr in (0.0, 0.33, 0.66, 1.0)]
    rep = dict(locks=int(n), segments=int(M), residual_mm=round(res, 4), imbalance_rad_max=round(eq, 4), min_gap_ratio=round(float(gap.min()) if len(gap) else 9.9, 2),
               gap_under_half_share=round(float((gap < 0.5).mean()) if len(gap) else 0.0, 4),
               rank_swaps_outer={str(k): v for k, v in swaps.items()}, outer_locks=int(outer.sum()),
               width_mm_at_0_33_66_100=[round(x, 1) for x in width],
               bending_length_mm=round(float(np.median((EI / w) ** (1 / 3))), 1), hairs_per_lock_median=float(np.median(n_hairs)))
    return dict(x=X, R=Rs, s=np.c_[np.zeros(n), np.cumsum(np.repeat(l[:, None], M, 1), 1)], rad=rad, report=rep)
