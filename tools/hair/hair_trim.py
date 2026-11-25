"""The trim: the fringe and the loose locks cut to a line, as a stylist finishes a cut. The field grows every hair to
its preset length along its own path, so hairs that took a longer way round hang past the rest and the fringe's edge
is ragged and long. A cut ignores each hair's length: everything below the line goes.

The line (z, up, against the pupil's height, across the face by |x|): a curtain fringe is cut shortest in the middle,
"at the bridge of the nose or the brows", and longer toward the sides, down to the cheekbone (bangs_flow's [CC] and
[BS]). So: BROW_MM above the pupil within CENTRE_HH of the middle, falling to SIDE_MM below it at SIDE_HH and beyond.
The loose locks are cut LOCK_MM below the pupil. The edge is point-cut, not blunt: each lock (a 7 mm root cell) ends
up to POINT_MM above the line (point cutting softens a fringe's edge; amount DRAFT). Heights are DRAFT, set from the
person's "trim the bangs" against the photo-measured edge they replace.
"""
import numpy as np
import scalp_field as SF

MM = SF.MM
BROW_MM = 9.0           # DRAFT: the middle of the fringe ends this far above the pupil (on the brow)
SIDE_MM = 10.0          # DRAFT: its sides end this far below the pupil (it was 24: the cheekbone)
CENTRE_HH, SIDE_HH = 0.08, 0.33
LOCK_MM = 12.0          # DRAFT: the loose locks end this far below the pupil (it was 25)
POINT_MM = 3.0          # DRAFT
CELL_MM = 7.0


def line(x, lock=False):
    zp = SF.PUPIL[1]
    if lock:
        return np.full(np.shape(x), zp - LOCK_MM * MM)
    return zp + (BROW_MM - (BROW_MM + SIDE_MM) * SF.sm(CENTRE_HH, SIDE_HH, np.abs(x))) * MM


def trim(paths, fringe, locks, seed=7):
    """paths (n, m, 3), nan-padded; fringe, locks: masks of the fibres to cut. Returns (paths, report)."""
    rng = np.random.default_rng(seed)
    Q = np.array(paths, float)
    cut_mm = []
    for sel, is_lock in ((fringe, False), (locks, True)):
        for i in np.where(sel)[0]:
            p = Q[i]
            good = ~np.isnan(p[:, 0])
            c = np.floor(p[0] / (CELL_MM * MM)).astype(np.int64)
            j = np.random.default_rng(abs(int(c[0] * 73856093 ^ c[1] * 19349663 ^ c[2] * 83492791)) + seed).random()
            zl = line(p[:, 0], is_lock) + j * POINT_MM * MM
            below = good & (p[:, 2] < zl) & (p[:, 1] < SF.CENTRE[1])        # (in front of the head only)
            below[:3] = False
            if not below.any():
                continue
            k = int(np.argmax(below))
            a, b = p[k - 1], p[k]
            fa, fb = a[2] - zl[k - 1], b[2] - zl[k]
            t = fa / (fa - fb) if fa != fb else 0.0
            seg = np.linalg.norm(np.diff(p[good], axis=0), axis=1)
            cut_mm.append(float(seg[k - 1:].sum() - t * seg[k - 1]) / MM)
            p[k] = a + t * (b - a)
            p[k + 1:] = np.nan
    cut_mm = np.array(cut_mm) if cut_mm else np.zeros(1)
    rep = dict(cut_fibres=int(len(cut_mm)), of=int(fringe.sum() + locks.sum()), cut_mm_median=round(float(np.median(cut_mm)), 1),
               cut_mm_p95=round(float(np.percentile(cut_mm, 95)), 1))
    return Q, rep
