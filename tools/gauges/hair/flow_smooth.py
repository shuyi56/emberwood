"""How smooth a hair flow is: the bend radius of its lines, before any wave or curl is laid on them.

Why: a flow built from separate eases (swing back behind the cheek, then forward to the chest, then onto the body) can
join them in corners. Drawn as lines it reads as a zigzag (John 2026-11-28, the photo style's front fall: "fix this flow
add the proper gauge so you never flow like this again needs to be smoother"). Nothing in the field's other gauges (rank,
turning back, flux) sees a corner.

The bound: hair that hangs free cannot hold a bend tighter than its bending length, l = (EI / w)^(1/3), about 45 to
50 mm (Goldstein, Warren, Ball 2012, PRL 108 078101: "about 5 cm"; tail_lock_rods.py computes 45.4 mm from Robbins'
modulus and density). A tighter bend needs a contact to hold it (the skull, an ear, a tie). So:
  - free hair (off the head): radius >= FREE_R_MM (45). The share of nodes under it must be under FREE_SHARE (1%), and
    no node under FREE_MIN_MM (25).
  - styled hair held by its set (a fringe dried on a round brush): radius >= SET_R_MM (17.5, a 35 mm brush; ghd).
Also counted: reversals, the times a line's sideways heading (round the body) changes sign over more than REV_MM of
travel each way: a fall that swings one way and back is an S; more than one S in the flow itself (not the wave) is a zigzag.

    gauge(lines, release=None, kind='free') -> dict;  python3 tools/gauges/hair/flow_smooth.py FIELD.npz

lines: list of (k, 3) in head heights; release: per line, the node where it leaves the head (the part before it is not
free and is skipped). Trouble looks like: 'ok' False; 'worst' lists the lines to look at (index, radius mm, height z).
"""
import sys
import numpy as np

MM = 1 / 219.0
STEP_MM = 10.0              # the length a bend is read over (a fifth of the bending length)
FREE_R_MM, FREE_SHARE, FREE_MIN_MM = 45.0, 0.01, 25.0
SET_R_MM = 17.5
REV_MM = 15.0


def _resample(p, step):
    L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if L[-1] < 3 * step:
        return None
    s = np.arange(0, L[-1], step)
    return np.stack([np.interp(s, L, p[:, c]) for c in range(3)], 1)


def gauge(lines, release=None, kind='free', axis=(0.0, 0.04)):
    rad, where, rev = [], [], []
    for i, p in enumerate(lines):
        j = 0 if release is None else max(int(release[i]) - 2, 0)
        q = _resample(np.asarray(p)[j:], STEP_MM * MM)
        if q is None:
            continue
        d = np.diff(q, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        ang = np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1, 1))
        r = STEP_MM / np.maximum(ang, 1e-6)
        rad.append(r); where.append(np.c_[np.full(len(r), i), q[1:-1, 2]])
        th = np.unwrap(np.arctan2(q[:, 0] - axis[0], -(q[:, 1] - axis[1])))
        arc = np.diff(th) * np.hypot(q[1:, 0] - axis[0], q[1:, 1] - axis[1]) / MM      # mm travelled round the body per step
        runs, cur = [], 0.0
        for a in arc:                                                    # (runs of one sign, by the distance they cover)
            if cur == 0 or (a >= 0) == (cur >= 0):
                cur += a
            else:
                runs.append(cur); cur = a
        runs.append(cur)
        big = [x for x in runs if abs(x) > REV_MM]
        rev.append(max(len(big) - 1, 0))
    if not rad:
        return dict(ok=True, lines=0)
    R = np.concatenate(rad); Wh = np.concatenate(where)
    lim = FREE_R_MM if kind == 'free' else SET_R_MM
    under = float((R < lim).mean())
    o = np.argsort(R)[:5]
    out = dict(kind=kind, lines=len(rad), bound_mm=lim, radius_mm_min=round(float(R.min()), 1),
               radius_mm_p1=round(float(np.percentile(R, 1)), 1), radius_mm_median=round(float(np.median(R)), 0),
               share_under_bound=round(under, 4), reversals_max=int(max(rev)), lines_with_over_1_reversal=int((np.array(rev) > 1).sum()),
               worst=[dict(line=int(Wh[k, 0]), radius_mm=round(float(R[k]), 1), z=round(float(Wh[k, 1]), 2)) for k in o])
    if kind == 'free':
        out['ok'] = bool(under < FREE_SHARE and R.min() >= FREE_MIN_MM and out['lines_with_over_1_reversal'] == 0)
    else:
        out['ok'] = bool(under < FREE_SHARE)
    return out


if __name__ == '__main__':
    z = np.load(sys.argv[1], allow_pickle=True)
    print(gauge(list(z['lines']), z['release'] if 'release' in z else None))
