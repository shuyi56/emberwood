"""A noising round over finished hair lines: the field's lines are smooth by construction (streamlines, smoothing
passes, blended joins), and every hair laid exactly along them reads as a moulded shell. This breaks them up the way
a groom's last modifiers do (Autodesk Maya XGen, Noise modifier: Magnitude, the farthest a point moves; Frequency,
"cycles per unit length of the primitive"; Magnitude Scale, the amount along the hair from root to tip; Correlation,
how far neighbouring hairs share the noise; Preserve Length. Clumping modifier: noise per clump, so locks move
together):

  1. Lock noise: hairs whose roots share a lock (a cell LOCK_MM across on the scalp) share one smooth wander, so the
     hair breaks into locks that part from each other (XGen's correlation, clump noise).
  2. Strand noise: each hair adds its own, smaller and shorter, so hairs within a lock do not lie as one ribbon.
  3. Flyaways: a few hairs (FLY_SHARE) lift off over their last part.

Every displacement is square to the hair, grows from 0 at the root (the root and the hairline cover stay where they
were) toward the tip along a smoothstep over the first RAMP share of the length, and is kept outside the skin. Each
node moves only square to the hair (XGen's Preserve Length 0): keeping the length (100) pulled the tips back along
their hairs by 30 to 45 mm, which raised the fringe's edge off its measured height. All amounts are DRAFT
(XGen documents the controls, not values; these were set by eye). Round 1 (lock 3 mm over 45 mm, hair 0.9 mm over
14 mm, a third harmonic) read as frizzy curl, not straight hair (rejection logged): straight hair's own bend is slow,
so the noise here is long and quiet.

    import noise_round as NR; paths = NR.apply(paths, surf=..., seed=11)
"""
import numpy as np
import scalp_field as SF

MM = SF.MM
LOCK_MM = 7.0           # DRAFT: a lock's root cell on the scalp
LOCK_AMP_MM = 1.6       # DRAFT: lock noise at the tip
LOCK_WAVE_MM = 90.0     # DRAFT: its wavelength along the hair (XGen Frequency 1 / 90 mm)
STRAND_AMP_MM = 0.35    # DRAFT: per-hair noise at the tip
STRAND_WAVE_MM = 60.0   # DRAFT: its wavelength
RAMP = 0.6              # DRAFT: the noise reaches full size at this share of the length (Magnitude Scale)
FLY_SHARE = 0.005       # DRAFT: share of hairs that fly away
FLY_AMP_MM = 5.0        # DRAFT: a flyaway's lift at its tip
FLY_FROM = 0.55         # DRAFT: a flyaway leaves its lock past this share of its length
SKIN_GAP_MM = 0.6       # a hair keeps this far off the skin


def _smooth_noise(rng, n, s_mm, wave_mm, k=2):
    """k-channel smooth 1D noise at lengths s_mm (two random-phase sines per channel, wavelengths wave and wave/1.7,
    amplitudes 1 and 0.4; a third at wave/2.9 read as curl, round 1), unit RMS-ish. Shape (n, len(s), k)."""
    out = np.zeros((n, len(s_mm), k))
    for f, a in ((1.0, 1.0), (1.7, 0.4)):
        ph = rng.random((n, 1, k)) * 2 * np.pi
        w = 2 * np.pi * f / wave_mm * (0.8 + 0.4 * rng.random((n, 1, k)))
        out += a * np.sin(w * s_mm[None, :, None] + ph)
    return out / 0.8


def apply(paths, surf=None, seed=11, scale=1.0):
    """paths (n, m, 3) in head heights, nan-padded; returns the noised paths and a report."""
    rng = np.random.default_rng(seed)
    P = np.array(paths, dtype=float)
    n, m, _ = P.shape
    good = ~np.isnan(P[:, :, 0])
    seg = np.linalg.norm(np.diff(np.nan_to_num(P), axis=1), axis=2) * (good[:, 1:] & good[:, :-1])
    S = np.c_[np.zeros(n), np.cumsum(seg, 1)] / MM                    # length along each hair, mm
    L = S.max(1)
    t = S / np.maximum(L[:, None], 1e-9)
    # the frame along each hair: tangent, the side (tangent x outward) and the out (side x tangent)
    T = np.gradient(np.nan_to_num(P), axis=1)
    T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-12)
    O = np.nan_to_num(P) - SF.CENTRE
    side = np.cross(T, O); side /= np.maximum(np.linalg.norm(side, axis=2, keepdims=True), 1e-12)
    out = np.cross(side, T); out /= np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-12)
    # locks: root cells
    cell = np.floor(P[:, 0] / (LOCK_MM * MM)).astype(np.int64)
    key = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    uk, lock = np.unique(key, return_inverse=True)
    s_grid = np.arange(0, L.max() + 1.0, 1.0)
    lock_n = _smooth_noise(rng, len(uk), s_grid, LOCK_WAVE_MM)        # (locks, s, 2)
    ramp = SF.sm(0.1, RAMP, t)[:, :, None]                            # (nothing over the first tenth: round 1 frizzed the crown)
    idx = np.clip(np.round(S).astype(int), 0, len(s_grid) - 1)
    ln = lock_n[lock[:, None], idx]                                    # (n, m, 2)
    sn = np.zeros((n, m, 2))
    for f, a in ((1.0, 1.0), (1.7, 0.5)):                              # (per hair, evaluated directly)
        ph = rng.random((n, 1, 2)) * 2 * np.pi
        sn += a * np.sin(2 * np.pi * f / STRAND_WAVE_MM * S[:, :, None] + ph)
    sn /= 0.8
    d = ramp * (LOCK_AMP_MM * ln + STRAND_AMP_MM * sn) * scale
    fly = rng.random(n) < FLY_SHARE
    if fly.any():
        lift = SF.sm(FLY_FROM, 1.0, t[fly]) * FLY_AMP_MM * scale * (0.5 + rng.random((int(fly.sum()), 1)))
        d[fly, :, 1] += lift
    D = (d[:, :, :1] * side + d[:, :, 1:] * out) * MM
    Q = P + D
    if surf is not None:                                               # outside the skin
        flat = Q[good]
        U = flat - SF.CENTRE
        r = np.linalg.norm(U, axis=1)
        U /= np.maximum(r[:, None], 1e-12)
        rs = surf.radius(U) + SKIN_GAP_MM * MM
        flat = np.where((r < rs)[:, None], SF.CENTRE + U * rs[:, None], flat)
        Q[good] = flat
    tip = np.linalg.norm(Q[:, :, :] - P, axis=2)[np.arange(n), good.sum(1) - 1] / MM
    rep = dict(locks=int(len(uk)), flyaways=int(fly.sum()), tip_move_mm_median=round(float(np.median(tip)), 2),
               tip_move_mm_p95=round(float(np.percentile(tip, 95)), 2),
               root_move_mm_max=round(float(np.nanmax(np.linalg.norm(Q[:, 0] - P[:, 0], axis=1)) / MM), 3))
    return Q, rep
