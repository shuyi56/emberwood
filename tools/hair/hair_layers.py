"""The mid layer: hair is built from the inside out in layers with different jobs (Unreal's Hair Card Generator:
Coverage, "the core layer"; Mid Layer, "meant for building the overall shapes"; Top Layer; Flyaways. Giovannini and
Verhoolen, 80.lv: an opaque base, then layers of ever less opaque hair, with gaps left "to give the hair depth").
One population of hairs drawn one way gives one texture; the layers have to differ in how they clump.

Here MID_SHARE of the fibres become the mid layer: gathered into thick clumps (the hair of one CLUMP_MM root cell
drawn MID_CLUMP of the way to the cell's mean line, the clumping of XGen's first, coarse Clumping modifier: "low
values for big clumps", Hindsbol Hansen, CGMA), so between two clumps there is a gap down to the coverage layer. The
rest stays the top layer: loose, fine, lying over the clumps. The page draws the mid layer wider, opaque, a little
darker and deeper (hair-fidelity.js). Shares and amounts DRAFT; Unreal's suggested clump counts are 200 mid to 500
top, so the mid clumps are the coarser.
"""
import numpy as np
import scalp_field as SF

MM = SF.MM
MID_SHARE = 0.4         # DRAFT
CLUMP_MM = 11.0         # DRAFT: a mid clump's root cell
MID_CLUMP = 0.7         # DRAFT: how far a hair is drawn to its clump's line
MOVE_MAX_MM = 4.0       # DRAFT: no hair is drawn farther than this (where a cell's hairs part ways, they are no clump)
CLUMP_OVER = (0.03, 0.2)    # DRAFT: the clump forms over this share of the length from the root


def mid_layer(paths, seed=21):
    """paths (n, m, 3), nan-padded. Returns (paths, tier per fibre: 1 mid, 0 top, report)."""
    rng = np.random.default_rng(seed)
    Q = np.array(paths, float)
    n, m, _ = Q.shape
    tier = (rng.random(n) < MID_SHARE).astype(int)
    mid = np.where(tier == 1)[0]
    cell = np.floor(Q[mid, 0] / (CLUMP_MM * MM) + rng.random(3)).astype(np.int64)
    key = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    uk, cl = np.unique(key, return_inverse=True)
    P = Q[mid]
    good = ~np.isnan(P[:, :, 0])
    P0 = np.where(good[:, :, None], P, 0.0)
    cnt = np.zeros((len(uk), m)); np.add.at(cnt, cl, good.astype(float))
    mean = np.zeros((len(uk), m, 3)); np.add.at(mean, cl, P0)
    mean /= np.maximum(cnt[:, :, None], 1.0)
    full = cnt >= np.maximum(2, 0.6 * cnt[:, :1])            # (where most of the clump still runs: not past a trim)
    t = np.linspace(0, 1, m)[None, :]
    w = MID_CLUMP * SF.sm(CLUMP_OVER[0], CLUMP_OVER[1], t) * full[cl] * good
    d = w[:, :, None] * (np.where(good[:, :, None], mean[cl], 0.0) - P0)
    dl = np.linalg.norm(d, axis=2, keepdims=True)
    P = P + d * np.minimum(1.0, MOVE_MAX_MM * MM / np.maximum(dl, 1e-12))
    P[~good] = np.nan
    move = np.linalg.norm(np.nan_to_num(P - Q[mid]), axis=2).max(1) / MM
    Q[mid] = P
    rep = dict(mid_fibres=int(len(mid)), clumps=int(len(uk)), hairs_per_clump=round(float(len(mid) / len(uk)), 1),
               move_mm_median=round(float(np.median(move)), 1), move_mm_p95=round(float(np.percentile(move, 95)), 1))
    return Q, tier, rep
