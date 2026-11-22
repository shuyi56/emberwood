"""The bun hold test (John, 2026-11-21): does a bun keep its shape and its place on the head through running, walking,
turning and twisting? It runs the hair stress test (rq_hair.py) on a bun style, with one more motion (`twist`: the
head turned on the neck, the body still), and then measures the knot itself in every solved frame, in the head's
frame:

    sag_max        how far the knot's centre of mass moves from where it was drawn (HH)
    drop_max       the downward part of that, along the head's own down (HH)
    swivel_max     how far the knot turns as one piece on the head (degrees: the rigid turn that best fits it)
    deform_p99     what is left after that rigid fit: how far the knot's own shape gives (HH, the worst 1% of points)
    core_pierce    how deep knot strands sink into the knot's core (HH): wraps collapsing into the bun
    head_pierce    how deep knot strands go into the scalp or neck (HH)

Limits: half a sprite pixel. A hair feature of 10 pixels is about 0.35 HH (CLAUDE.md), so one pixel is 0.035 HH and
the knot may move or give at most 0.017 HH before it shows; at the knot's rim (0.17 HH out) that is a turn of 5.7
degrees. Pierce limits are the stress test's own depth limits.

    python3 rq_bun.py STYLES_DIR [--style bun] [--out OUT] [--poses walk,run,turn,startstop,spin,twist]
        [--held | --bun JSON]          the engineered hold (HELD below), or any mix of its parts; neither: as drawn
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rq_hair as R  # noqa: E402

PX = 0.035                     # HH: one sprite pixel (a 10-pixel hair feature is 0.35 HH)
LIM = {'sag_max': 0.5 * PX, 'drop_max': 0.5 * PX, 'swivel_max': 5.7, 'deform_p99': 0.5 * PX,
       'core_pierce': 0.01, 'head_pierce': 0.03}

# ---------------------------------------------------------------------------------------------------- twist
# The head turned on the neck with the body still: 70 degrees to each side (cervical axial rotation, most of it at
# C1-C2), the sweep from one side to the other at a peak of 350 deg/s, a quick look round well under the 25 rad/s
# (1430 deg/s) peak of a voluntary turn (PMC6444911). The axis is vertical through the skull's centre line.
TWIST_DEG, TWIST = 70.0, [(0.35, 1.0), (0.15, 1.0), (0.6, -1.0), (0.15, -1.0), (0.35, 0.0), (0.6, 0.0)]
_motion = R.motion


def motion(pose, walk):
    if pose != 'twist':
        return _motion(pose, walk)
    ax = np.array([0.0, 0.04, 0.0])
    a = math.radians(TWIST_DEG)
    dur = sum(d for d, _ in TWIST)

    def ang(t):
        t0, prev = 0.0, 0.0
        for d, tgt in TWIST:
            if t <= t0 + d:
                return a * (prev + (tgt - prev) * R._smooth((t - t0) / d))
            t0, prev = t0 + d, tgt
        return a * prev

    def f(t):
        Rh = R._yaw(ang(t))
        return Rh, ax - Rh @ ax, np.eye(3), np.zeros(3)
    return dur, 0.0, f


# ---------------------------------------------------------------------------------------------------- walk and run
# The stress test reads the game's walk cycle from its keys with straight lines between them (np.interp): the speed
# jumps at every key, so the head is shaken at 4.9 g in the walk and 14.8 g in the run (the run is the walk at 1.7x
# the rate and 1.8x the swing), where a runner's head peaks at 1.55 g (15.2 m/s^2 at the forehead at 2.83 m/s,
# Macdermid et al. 2015, J Sport Health Sci, PMC6189242; the body takes out about 80% of the tibia's 7.7 g on the
# way up). Here the cycle is a periodic cubic through the same keys (smooth speed, same poses), and the run's swing
# is scaled so the knot's peak acceleration is that 1.55 g (SMOOTH below; RUN_G the target).
SMOOTH = False      # (the stress test's own walk and run now do this: kept to reproduce the fix on an older copy)
RUN_G, RUN_RATE = 1.55, 1.7
_RUN_AMP = {}


def _cycle_fn(W):
    from scipy.interpolate import CubicSpline
    wt, wb, wq = np.array(W['t']), np.array(W['body']), np.array(W['chest'])
    qq = wq.copy()
    for k in range(1, len(qq)):            # (one hemisphere, so the quaternion keys interpolate the short way)
        if qq[k] @ qq[k - 1] < 0:
            qq[k] = -qq[k]
    bs = CubicSpline(wt, np.r_[wb[:-1], wb[:1]], bc_type='periodic')
    qs = CubicSpline(wt, np.r_[qq[:-1], qq[:1]], bc_type='periodic')
    # (the keys start one frame in, at 1/24 s, and the last repeats the first: the period is their span, not the last
    # key's time. Taken as the last key's time, as the stress test does, every cycle has a 1/24 s jump)
    t0, T = float(wt[0]), float(wt[-1] - wt[0])

    def cycle(t, rate=1.0, amp=1.0):
        u = t0 + (t * rate) % T
        b = bs(u) * amp
        q = qs(u).copy()
        q[:3] *= amp
        q /= np.linalg.norm(q)
        return b, R._qmat(q)
    return cycle, T


def _peak_g(f, dur, point=(0, 0.428, 0.892)):
    h = 1 / 120
    c = np.array(point)
    P = np.array([f(t)[0] @ c + f(t)[1] for t in np.arange(0, dur, h)]) * R.HEAD_CM / 100
    return float(np.linalg.norm(np.diff(P, 2, axis=0) / h / h, axis=1).max() / 9.81)


def _walk_run(pose, walk):
    cycle, T = _cycle_fn(walk)
    piv = np.array([0, 0.04, walk['chest_z']])
    I = np.eye(3)
    if pose == 'walk':
        def f(t):
            b, Rm = cycle(t)
            return I, b, Rm, b + piv - Rm @ piv
        return 2 * T, T, f
    lean = R._pitch(math.radians(-8))

    def make(amp):
        def f(t):
            b, Rm = cycle(t, RUN_RATE, amp)
            Rb = lean @ Rm
            return lean, b, Rb, b + piv - Rb @ piv
        return f
    if 'amp' not in _RUN_AMP:
        lo, hi = 0.1, 3.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if _peak_g(make(mid), 2 * T / RUN_RATE) > RUN_G:
                hi = mid
            else:
                lo = mid
        _RUN_AMP['amp'] = round(0.5 * (lo + hi), 3)
    return 2 * T / RUN_RATE + 0.6, T / RUN_RATE + 0.6, make(_RUN_AMP['amp'])


def motion2(pose, walk):
    if SMOOTH and walk and pose in ('walk', 'run'):
        return _walk_run(pose, walk)
    return motion(pose, walk)


R.motion = motion2
R.POSES = R.POSES + ('twist',)


# ---------------------------------------------------------------------------------------------------- the hold
# How a real bun is held, built into the solve (each part switched on in BUN, and the settings in every cache key):
#   tie     the elastic round the base of the tail: the gathered hair is gripped where it meets the knot, so the
#           gather strands' ends are held to the head there, and the core (the tail's own base, wound tight round
#           the tie) is one solid piece with the head.
#   core    what the wraps are wound on: a solid ball the wraps lie on and cannot fall into (the creator's dark
#           core, shrunk to where the inner wraps sit).
#   pins    U-pins round the rim, prongs aimed at the centre of the coil, catching the wraps they pass through
#           and pinning them to the base (Bloch: 3-4 pins round the bun, pushed into the coil's centre).
#   tucks   each wrap's end tucked under the band before it or into the elastic (makeup.com, Babes in Hairland:
#           "tuck tails into the elastic or under the first coiled layer; pin"; Bloch: "tuck the ends under"): the
#           last two points of every wrap that ends inside the knot are held. The wrap whose end escapes (the
#           creator's 'free' band, its end carried on by a short free clump) is left loose.
#   shape   a tight bun is one firm piece on the head: wound under tension onto its core, pinned and netted, every
#           wrap pressed on the one below it. Each step every point of the knot's body is drawn part of the way
#           (`shape`) back toward where it was drawn, carried in the head's frame: the global shape constraint game
#           hair solvers use to keep an updo's style (Han and Harada 2012, "Real-time hair simulation with efficient
#           hair style preservation"; AMD TressFX's global shape matching). The free end and flyaways are left out.
#   net     the strands of the knot tied to one another more firmly than loose hair is (a hairnet over the bun).
# The engineered hold (round 5, 2026-11-21): passes every motion on the fixed solver. `pins` and `pin_catch` from the
# Bloch ballet bun (3-4 U-pins round the rim; a U-pin's prongs about 1 cm apart catch 0.035 HH round them); `shape`
# 0.3 and `damp` 0.2 are tuned on this solver (0.1 let the run drop 0.086 HH), `net` 10x firmer than loose
# cohesion: tuned values, not measured ones.
HELD = {'tie': True, 'core': True, 'tucks': True, 'pins': 4, 'pin_catch': 0.035, 'net': 2e-4, 'shape': 0.3, 'damp': 0.2}
BUN = {'tie': False, 'core': False, 'core_scale': 0.8, 'pins': 0, 'pin_lat': 0.25, 'pin_catch': 0.02,
       'pin_phase': 0.0, 'net': None, 'sticks': 0, 'tucks': False, 'shape': 0.0, 'damp': 0.0,
       'roll': False, 'pin_seam': [-0.88, 0.48, 0.0], 'pin_span': [0.12, 0.88], 'pin_cross': False, 'slack': 0.0, 'ref': 'drawn', 'free_by': 'length', 'tuck_r': 1.4}
# The French twist (round 0, 2026-11-21): the same hold on a roll instead of a ball. `roll` swaps the core's ellipsoid
# for a capsule along the core strand's own centre line (the roll follows the curve of the back of the head, which a
# straight ellipsoid cannot), the gather ends and tucks are tested against it, and the pins are driven in along the
# seam (`pin_seam`, the direction from the roll's axis to its seam, `pin_span` the share of the roll's length the
# first and last pin sit at), as a French twist is pinned (Marie Claire: pins nearly vertical, one every 1-2 inches;
# L'Oreal: tuck the ends into the tunnel of the twist).
# Round 3 (2026-11-21) set the roll's two values apart from the ball's: `core_scale` 1.0 (the roll's strands lie on its
# surface, drawn at the core's own radius; at the ball's 0.8 they sank into the gap and the start-stop deform stayed
# at 0.025 HH over three rounds) and `shape` 0.4 (0.3 left the core pierce at 0.013 with the full-size core; the roll
# is twice the knot's length). Five pins span the roll, 1.4-1.5 inches apart on a 15 cm roll.
HELD_ROLL = dict(HELD, roll=True, pins=5, pin_catch=0.035, core_scale=1.0, shape=0.4)


def configure(**kw):
    BUN.update(kw)
    R.SOLVE_VERSION = '7+bun:' + json.dumps({k: v for k, v in BUN.items() if k != 'ref'}, sort_keys=True) + (f'+smooth{RUN_G}' if SMOOTH else '')


_build = R.build_solver
_collider = R.collider
_STATE = {}


def _pin_lines(K):
    """The U-pins: lines from the knot's rim at even turns round its axis (at latitude pin_lat up from the
    foot), aimed through the knot's centre."""
    n = K['axis']
    e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    L = []
    for k in range(BUN['pins']):
        a = BUN['pin_phase'] + 2 * math.pi * k / BUN['pins']
        d = math.cos(BUN['pin_lat']) * (math.cos(a) * e1 + math.sin(a) * e2) + math.sin(BUN['pin_lat']) * n
        L.append((K['centre'] + 1.3 * K['core_r'] * d, -d))
    return L


def _pin_lines_roll(K):
    """The pins of a French twist: driven in along the seam, each from a point on the roll's surface on the seam side
    toward its axis, evenly spread over the roll's length."""
    P, pr = K['path'], K['pr']
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = BUN['pin_seam']
    s = np.array(s, float) / np.linalg.norm(s)
    lo, hi = BUN['pin_span']
    L = []
    for k in range(BUN['pins']):
        f = lo + (hi - lo) * (k / max(1, BUN['pins'] - 1))
        j = int(np.searchsorted(seg, f * seg[-1]))
        j = min(j, len(P) - 1)
        sk = s * np.array([-1.0, 1.0, 1.0]) if BUN.get('pin_cross') and k % 2 else s      # (crossed: every second pin from the other side)
        L.append((P[j] + 1.3 * pr[j] * sk, -sk, 2.0 * pr[j]))
    return L


def _near_core(Y, K, k, add):
    """Which points of Y lie within k core radii (plus add) of the core: the ball's centre, or the roll's centre line."""
    if 'path' in K:
        D, Rr, _ = path_near(Y, K['path'], K['pr'])
        return D < k * Rr + add
    return np.linalg.norm(Y - K['centre'], axis=1) < k * K['core_r'] + add


def build_solver(strands, W):
    f, runs, roots, X, thick = _build(strands, W)
    if not any([BUN['tie'], BUN['core'], BUN['pins'], BUN['net'], BUN['sticks'], BUN['tucks'], BUN['shape']]):
        return f, runs, roots, X, thick
    K = knot_of(strands, X, runs)
    hold = []
    if BUN['tie']:
        # the core: held whole
        hold.append(runs[K['core']])
        # the gather: every strand off the knot whose end reaches the knot's foot is gripped at its last two points
        for k, r in enumerate(runs):
            if k in K['strands']:
                continue
            if _near_core(X[r[-1]][None], K, 1.25, 0.05)[0]:
                hold.append(r[-2:])
    if BUN['tucks']:
        starts = np.array([X[runs[k][0]] for k in K['free']]) if K['free'] else np.zeros((0, 3))
        for k in K['body']:
            if k == K['core']:
                continue
            e = X[runs[k][-1]]
            if len(starts) and np.linalg.norm(starts - e, axis=1).min() < 0.03:
                continue                       # (its end carries on as an escaping end)
            if _near_core(e[None], K, BUN['tuck_r'], 0.0)[0]:
                hold.append(runs[k][-2:])
    if BUN['pins']:
        body = np.concatenate([runs[k] for k in K['body']])
        lines = [(p0, d, 2.0 * K['core_r']) for p0, d in _pin_lines(K)] if not BUN['roll'] else _pin_lines_roll(K)
        for p0, d, depth in lines:
            v = X[body] - p0
            a = v @ d
            dist = np.linalg.norm(v - np.outer(a, d), axis=1)
            hold.append(body[(dist < BUN['pin_catch']) & (a > 0) & (a < depth)])
    hold = np.unique(np.concatenate(hold)) if hold else np.zeros(0, int)
    f.pin(hold)
    if BUN['net']:
        idx = K['idx']
        a, b = R.neighbours(X[idx], 0.6 * R.NEAR * W.m['height'])
        sid = np.concatenate([np.full(len(r), k) for k, r in enumerate(runs)])[idx]
        o = sid[a] != sid[b]
        f.tie(idx[a[o]], idx[b[o]], BUN['net'])
    inside = np.zeros(len(X), bool)
    if BUN['core']:
        cd = core_depth(X, dict(K, core_r=K['core_r'] * BUN['core_scale'], core_half=K['core_half'] * BUN['core_scale'], cscale=BUN['core_scale']), 1.0)
        inside = cd > 0
        inside[runs[K['core']]] = True
    _STATE.update(K=K, inside=inside, hold=hold, rest=X.copy(), tgt_prev=None)
    if BUN['shape']:
        f._shape_idx = np.setdiff1d(K['body_idx'], hold)
    return f, runs, np.unique(np.r_[roots, hold]).astype(int), X, thick


def collider(W, thick, hh, frame=None):
    body, state = _collider(W, thick, hh, frame)
    _STATE['frame'] = state
    if not BUN['core'] or 'K' not in _STATE:
        return body, state
    K, inside = _STATE['K'], _STATE['inside']
    c, n = K['centre'], K['axis']
    ra, rh = K['core_r'] * BUN['core_scale'], K['core_half'] * BUN['core_scale']
    gap = np.maximum(thick, 0.004 * hh)

    def core_push(Y, gp):
        """Points of Y (head heights, the head's frame) inside the core, set out onto its surface plus gp."""
        if 'path' in K:
            Z = Y.copy()
            near = np.linalg.norm(Y - c, axis=1) < K['core_half'] * 1.4 + 2 * K['core_r'] + 0.1
            m = np.zeros(len(Y), bool)
            if near.any():
                D, Rr, C = path_near(Y[near], K['path'], K['pr'] * BUN['core_scale'])
                g = gp[near] if np.ndim(gp) else gp
                rr = Rr + g
                inn = D < rr
                Zn = Y[near].copy()
                dd = np.maximum(D, 1e-9)[:, None]
                Zn[inn] = (C + (Y[near] - C) * (rr[:, None] / dd))[inn]
                Z[near] = Zn
                m[np.where(near)[0][inn]] = True
            return Z, m
        d = Y - c
        a = d @ n
        rv = d - np.outer(a, n)
        rr = np.linalg.norm(rv, axis=1)
        q = np.sqrt((rr / (ra + gp)) ** 2 + (a / (rh + gp)) ** 2)
        m = q < 1
        Z = Y.copy()
        Z[m] = c + d[m] / q[m, None]
        return Z, m

    def wrapped(x, which=None):
        p, touch = body(x, which)
        idx = np.arange(len(x)) if which is None else which
        ok = ~inside[idx]
        if not ok.any():
            return p, touch
        Rh, oh = state['Rh'], state['oh']
        Y = (p / R.UNITS - oh) @ Rh
        Z, m = core_push(Y, gap[idx])
        m &= ok
        if m.any():
            p = p.copy()
            p[m] = (Z[m] @ Rh.T + oh) * R.UNITS
            touch = touch | m
        return p, touch
    return wrapped, state


_step = R.HairFabric.step


def step(self, h, body=None, neutral=False):
    _step(self, h, body, neutral)
    idx = getattr(self, '_shape_idx', None)
    if idx is None or not BUN['shape'] or 'frame' not in _STATE:
        return
    fr = _STATE['frame']
    tgt = (_STATE['rest'][idx] @ fr['Rh'].T + fr['oh']) * R.UNITS
    self.pos[idx] += BUN['shape'] * (tgt - self.pos[idx])
    if BUN['damp']:
        # (the knot's own give is damped: hair packed into a bun rubs on itself, so a point's speed relative to the
        # head's motion there is cut by `damp` a step; the head's own motion is kept)
        prev = _STATE.get('tgt_prev')
        if prev is not None and len(prev) == len(tgt):
            v_head = tgt - prev
            v_rel = (self.pos[idx] - self.prev[idx]) - v_head
            self.prev[idx] = self.pos[idx] - v_head - (1 - BUN['damp']) * v_rel
        _STATE['tgt_prev'] = tgt


R.HairFabric.step = step
R.build_solver = build_solver
R.collider = collider

# ---------------------------------------------------------------------------------------------------- the knot

def knot_of(strands, X0, runs):
    """The knot: its core (the longest strand running along the knot's axis through the middle of it, the dark
    ball) and every strand whose points mostly lie within the knot's reach of the core's middle."""
    lens = [np.linalg.norm(np.diff(X0[r], axis=0), axis=1).sum() for r in runs]
    # the core: the broadest strand at the back of the crown (the dark ball under the knot, drawn as one tube along
    # the knot's axis, far wider than any lock)
    best, core = -1, None
    roll = BUN['roll']
    for k, r in enumerate(runs):
        P = X0[r]
        wm = float(np.mean(strands[k]['w']))
        if P[:, 1].mean() > 0.25 and P[:, 2].mean() > (0.45 if roll else 0.6) and wm > best:
            best, core = wm, k
    rad = best
    P = X0[runs[core]]
    c = 0.5 * (P[0] + P[-1])
    n = (P[-1] - P[0]) / np.linalg.norm(P[-1] - P[0])
    reach = 0.25
    knot = []
    if roll:
        path = X0[runs[core]]
        w0 = np.abs(np.asarray(strands[core]['w'], float))
        pr = np.interp(np.linspace(0, 1, len(path)), np.linspace(0, 1, len(w0)), w0)
        pr = np.maximum(pr, 0.35 * pr.max())      # (the tube's ends are closed round: a capsule keeps a radius there)
        # (the roll's strands lie on its surface, a radius off its centre line: the gather's locks that run up beside it
        # on the cap, and the combed sections ending at its top, are not part of it: the roll's strands run its length)
        for k, r in enumerate(runs):
            d, rr, _ = path_near(X0[r], path, pr)
            if (d < 1.15 * rr + 0.012 + BUN['slack']).mean() > 0.85 and (lens[k] > 0.5 or k == core):
                knot.append(k)
    else:
        for k, r in enumerate(runs):
            d = np.linalg.norm(X0[r] - c, axis=1)
            if (d < reach).mean() > 0.7:
                knot.append(k)
    idx = np.concatenate([runs[k] for k in knot])
    # (its body: the wraps and the core; the short pieces, the escaping end and the flyaways, are left free to move
    # as they are meant to and measured apart)
    if BUN['free_by'] == 'extent':
        # (a curl's path is several times as long as the strand it winds along: a curly style tells a free end or flyaway
        # from a wrap by how far it reaches from its root, not by the length of its coil)
        lens = [float(np.linalg.norm(X0[r] - X0[r][0], axis=1).max()) * (1.25 / 0.8) if float(np.mean(strands[k]['w'])) >= 0.006 else 0.0
                for k, r in enumerate(runs)]      # (reach 0.2 HH or more is a wrap (1.25/0.8 maps it past 0.25); the thin flyaways are free whatever their reach)
    bodyk = [k for k in knot if lens[k] > 0.25 or k == core]
    free = [k for k in knot if lens[k] <= 0.25 and k != core]
    if BUN['free_by'] == 'extent' and len(_STATE.get('hold', [])):
        # (a loop or curl the hold grips at only its end is a loose piece, measured with the free ends: the hold
        # is not what keeps it in place, so it is not the hold's to pass)
        hs = set(int(q) for q in _STATE['hold'])
        loose = [k for k in bodyk if k != core and sum(int(q) in hs for q in runs[k]) < 0.05 * len(runs[k])]      # (gripped at under 5 percent of its points: its two tucked end points only)
        bodyk = [k for k in bodyk if k not in loose]
        free = free + loose
    half = 0.5 * float(np.linalg.norm(P[-1] - P[0]))
    Kd = {'body': bodyk, 'free': free, 'body_idx': np.concatenate([runs[k] for k in bodyk]),'centre': c, 'axis': n, 'core': core, 'core_r': float(strands[core]['w'].max()), 'core_half': half,
          'strands': knot, 'idx': idx, 'runs': runs}
    if roll:
        Kd.update(path=path, pr=pr, core_r=float(pr.max()), core_half=0.5 * float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
    return Kd


def kabsch(A, B):
    """The rigid turn and shift that best carries A onto B (rows), and what is left over per point."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    fit = (A - ca) @ Rm.T + cb
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rm) - 1) / 2))))
    return ang, np.linalg.norm(B - fit, axis=1)


def path_near(Y, P, pr, chunk=1500):
    """For each point of Y, its distance to the polyline P, the radius there (pr interpolated along P) and the nearest
    point on it."""
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    D = np.empty(len(Y))
    Rr = np.empty(len(Y))
    C = np.empty((len(Y), 3))
    for s0 in range(0, len(Y), chunk):
        y = Y[s0:s0 + chunk]
        t = np.clip(((y[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        c = A[None] + t[..., None] * AB[None]
        d = np.linalg.norm(y[:, None, :] - c, axis=2)
        j = np.argmin(d, axis=1)
        r = np.arange(len(y))
        D[s0:s0 + chunk] = d[r, j]
        C[s0:s0 + chunk] = c[r, j]
        Rr[s0:s0 + chunk] = pr[j] + t[r, j] * (pr[j + 1] - pr[j])
    return D, Rr, C


def core_depth(Y, K, hh):
    """How deep points sit inside the core (an ellipsoid round the knot's axis, radius core_r, half-length
    core_half; for a roll a capsule along the core strand's centre line), as a share of HH (0 outside)."""
    if 'path' in K:
        sc = K.get('cscale', 1.0)
        D, Rr, _ = path_near(Y, K['path'], K['pr'] * sc)
        return np.maximum(0.0, 1 - D / Rr) * Rr / hh
    d = Y - K['centre']
    a = d @ K['axis']
    rr = np.linalg.norm(d - np.outer(a, K['axis']), axis=1)
    q = np.sqrt((rr / K['core_r']) ** 2 + (a / K['core_half']) ** 2)
    return np.maximum(0.0, (1 - q)) * K['core_r'] / hh


def knot_metrics(X0, Y, K, W, thick, own_core=True):
    hh = W.m['height']
    i = K['body_idx']
    A, B = X0[i], Y[i]
    shift = B.mean(0) - A.mean(0)
    ang, res = kabsch(A, B)
    out = {'sag': float(np.linalg.norm(shift) / hh), 'drop': float(max(0.0, -shift[2]) / hh), 'swivel': ang,
           'deform': float(np.percentile(res, 99) / hh)}
    # (the core's own points and the knot's innermost collar are the core: only the other knot strands can sink in)
    cd = core_depth(B, K, hh)
    cd0 = core_depth(A, K, hh)
    out['core'] = float(np.percentile(np.maximum(0, cd - cd0), 99.5))
    # (the core's foot is drawn sunk into the scalp, under the gather, where it is hidden: only the wraps count)
    w = ~np.isin(i, K['runs'][K['core']])
    out['head'] = float(np.max(np.maximum(0, -(W.head_d(B[w]) + 0.25 * thick[i][w])) / hh))
    j = np.concatenate([np.arange(0)] + [K['runs'][k] for k in K['free']])
    out['free'] = float(np.percentile(np.linalg.norm(Y[j] - X0[j], axis=1), 99) / hh) if len(j) else 0.0
    return out


# ---------------------------------------------------------------------------------------------------- the knot's contours

def knot_sheet(states, K, W, path, size=220):
    """The knot's contour sheet (the contour rule: inspected after every solve). Per state a row: the knot from the
    side and from the back (its points over the head's outline, the drawn knot in grey under it), and its sections
    at three heights along its axis, the edge's radius against the angle round the axis (blue) over the drawn one
    (grey). A held bun keeps its rings where they were; a slipping one shows them shifted, a sagging one drops."""
    from PIL import Image, ImageDraw
    n = K['axis']
    e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    i = K['body_idx']
    X0 = states[0][1]
    c = K['centre']
    span = max(0.42, 2.4 * K['core_half']) if 'path' in K else 0.42
    H = W._head_pts if hasattr(W, '_head_pts') else None
    rows = len(states)
    img = Image.new('RGB', (size * 2 + 360 + 40, rows * (size + 30) + 10), (247, 245, 240))
    d = ImageDraw.Draw(img)

    def px(u, v, ox, oy):
        return ox + size / 2 + u / span * size / 2, oy + size / 2 - v / span * size / 2

    def ring(P, hgt):
        q = P - c
        a = q @ n
        m = np.abs(a - hgt) < 0.03
        if m.sum() < 8:
            return None
        ang = np.arctan2(q[m] @ e2, q[m] @ e1)
        r = np.linalg.norm(q[m] - np.outer(a[m], n), axis=1)
        b = np.linspace(-np.pi, np.pi, 37)
        out = np.full(36, np.nan)
        for k in range(36):
            mm = (ang >= b[k]) & (ang < b[k + 1])
            if mm.any():
                out[k] = r[mm].max()
        return out
    nums = []
    for row, (name, X) in enumerate(states):
        oy = 10 + row * (size + 30)
        d.text((10, oy), name, fill=(40, 40, 40))
        oy += 14
        for col, (ax_u, ax_v) in enumerate(((1, 2), (0, 2))):
            ox = 10 + col * (size + 10)
            d.rectangle([ox, oy, ox + size, oy + size], outline=(200, 200, 200))
            if H is not None:
                hp = H[(np.abs(H[:, ax_u] - c[ax_u]) < span) & (np.abs(H[:, 2] - c[2]) < span)]
                for p in hp[::3]:
                    x, y = px(p[ax_u] - c[ax_u], p[2] - c[2], ox, oy)
                    d.point((x, y), fill=(215, 205, 195))
            for P, colr in ((X0[i], (190, 190, 190)), (X[i], (30, 90, 170))):
                for p in P:
                    x, y = px(p[ax_u] - c[ax_u], p[2] - c[2], ox, oy)
                    d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=colr)
            d.text((ox + 4, oy + size - 14), 'side' if col == 0 else 'back', fill=(80, 80, 80))
        ox = 20 + 2 * (size + 10)
        shift = []
        for k, hgt in enumerate((-0.5 * K['core_half'], 0.0, 0.5 * K['core_half']) if 'path' in K else (-0.03, 0.05, 0.13)):
            y0 = oy + 10 + k * (size // 3)
            d.text((ox, y0 - 10), f'section {hgt:+.2f} HH along the axis', fill=(80, 80, 80))
            for P, colr in ((X0[i], (190, 190, 190)), (X[i], (30, 90, 170))):
                r = ring(P, hgt)
                if r is None:
                    continue
                pts = [(ox + k2 * 9, y0 + 50 - rr * 200) for k2, rr in enumerate(r) if not np.isnan(rr)]
                if len(pts) > 1:
                    d.line(pts, fill=colr, width=2)
            r0, r1 = ring(X0[i], hgt), ring(X[i], hgt)
            if r0 is not None and r1 is not None:
                m = ~np.isnan(r0) & ~np.isnan(r1)
                if m.any():
                    shift.append(float(np.abs(r1[m] - r0[m]).mean()))
        nums.append(round(max(shift), 4) if shift else None)
        d.text((ox, oy + size - 14), f'worst section shift {nums[-1]} HH', fill=(40, 40, 40))
    img.save(path)
    return nums


def run(src, style='bun', out=None, poses=('walk', 'run', 'turn', 'startstop', 'spin', 'twist'), heat=True):
    world = R.load_world(os.path.join(src, 'head.json.gz'))
    st = R.load_style(os.path.join(src, style + '.json.gz'))
    cache = R.cache_dir()
    rep = R.analyse(st, world, poses=('rest',) + tuple(poses), solve=True, out=out, heat=heat, cache=cache)
    W = R.World(world, st, cache)
    X0, Xs, runs, roots, thick, steps, _ = R.settle(st['strands'], W, cache)
    K = knot_of(st['strands'], X0, runs)
    hh = W.m['height']
    res = {'bun': dict(BUN), 'held_points': int(len(_STATE.get('hold', []))),
           'knot': {'strands': len(K['strands']), 'body': len(K['body']), 'free': len(K['free']), 'points': int(len(K['idx'])),
                    'centre': [round(float(v), 3) for v in K['centre']], 'axis': [round(float(v), 3) for v in K['axis']]}}
    m = knot_metrics(X0, Xs, K, W, thick)
    res['rest'] = {'sag_max': m['sag'], 'drop_max': m['drop'], 'swivel_max': m['swivel'], 'deform_p99': m['deform'],
                   'core_pierce': m['core'], 'head_pierce': m['head'], 'free_p99': m['free']}
    walk = R.walk_keys()
    states = [('as drawn', X0), ('settled at rest', Xs)]
    W._head_pts = world['head']
    for p in poses:
        _, Fp, _ = R.check_pose(p, st['strands'], W, (X0, Xs, runs, roots, thick), walk, cache)
        ref = Xs if BUN['ref'] == 'rest' else X0      # (curls: the drawn coil is not the solver's rest, so a curly style measures its motion from its settled rest, and reports the settle apart)
        rows = [knot_metrics(ref, Y, K, W, thick) for Y in Fp['frames']]
        states.append((f'{p}, worst frame', Fp['frames'][int(np.argmax([r['sag'] + r['deform'] for r in rows]))]))
        res[p] = {'sag_max': max(r['sag'] for r in rows), 'drop_max': max(r['drop'] for r in rows),
                  'swivel_max': max(r['swivel'] for r in rows), 'deform_p99': max(r['deform'] for r in rows),
                  'core_pierce': max(r['core'] for r in rows), 'head_pierce': max(r['head'] for r in rows),
                  'free_p99': max(r['free'] for r in rows),
                  'worst_frame': int(np.argmax([r['sag'] + r['deform'] for r in rows]))}
    for p, v in res.items():
        if p in ('knot', 'bun', 'held_points'):
            continue
        for k in list(v):
            if k in LIM:
                v[k] = round(float(v[k]), 4)
        v['fails'] = [k for k in LIM if v.get(k, 0) > LIM[k]]
    res['limits'] = LIM
    res['stress'] = {'verdict': rep['verdict'], 'settle': rep.get('settle'), 'poses': rep['poses'],
                     'drawn': {k: rep['drawn'][k] for k in ('kinks', 'spikes', 'crossings', 'depth_head', 'buried_pct') if k in rep['drawn']}}
    if out:
        os.makedirs(out, exist_ok=True)
        res['contours'] = dict(zip([n for n, _ in states], knot_sheet(states, K, W, os.path.join(out, 'bun-contours.png'))))
        json.dump(res, open(os.path.join(out, 'bun-hold.json'), 'w'), indent=1)
    return res


def table(res):
    keys = list(LIM)
    s = f"{'':<10}" + ''.join(f'{k:>12}' for k in keys) + '\n'
    for p, v in res.items():
        if p in ('knot', 'limits', 'stress', 'bun', 'held_points', 'contours'):
            continue
        s += f'{p:<10}' + ''.join(f"{v[k]:>11.4f}{'*' if k in v['fails'] else ' '}" for k in keys) + '\n'
    s += f"{'limit':<10}" + ''.join(f'{LIM[k]:>12.4f}' for k in keys) + '\n'
    return s


if __name__ == '__main__':
    a = sys.argv[1:]

    def opt(k, d=None):
        return a[a.index(k) + 1] if k in a else d
    if '--rough' in a:
        globals()['SMOOTH'] = False
    if opt('--bun'):
        configure(**json.loads(opt('--bun')))
    elif '--held-roll' in a:
        configure(**HELD_ROLL)
    elif '--held' in a:
        configure(**HELD)
    r = run(a[0], opt('--style', 'bun'), opt('--out'), tuple(opt('--poses', 'walk,run,turn,startstop,spin,twist').split(',')))
    print(json.dumps(r['knot']))
    print(table(r))
