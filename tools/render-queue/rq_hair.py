"""The hair stress test (John, 2026-11-20): does a hairstyle's flow make sense, before anything is rendered? It reads
every lock of a face creator hairstyle as the strands the creator built it from, measures them against the head
they grow on, then runs them through our own solver (blender/shared/fabric.py, the game's fabric, the one the
creator drapes long hair with) at rest and in five motions, and reports where the hair would fail. It is the flow
check (rq_flow.py) for hair, and it works the same way: report-only, a band calibrated on approved hair, the worst
spots ringed and numbered on a heatmap, results cached so a re-run costs nothing.

    node rq_hair_export.cjs face-creator.html DIR [styles]     the input: DIR/<style>.json.gz and DIR/head.json.gz
    python3 rq_hair.py DIR [--styles a,b] [--out OUT] [--poses rest,walk,run,turn,startstop,spin] [--no-solve]
    python3 rq_hair.py DIR --calibrate wavyFull,ranger,ponytail      writes hair_baseline.json from approved hair
    python3 rq_hair.py --selftest OUT                                 a clean head of hair passes, broken ones fail

Everything is measured in head heights, from the head itself: the creator's skin is read, its chin, crown, width
and depth measured, and every distance below is a share of that measured height (HH), so a different head or scale
moves the limits with it. Per style, written to OUT/<style>/hair.json and hair-heat.png:

  the strands as drawn
    kink_radius_p1     the tightest 1% of bends along the strands (radius of curvature, HH; lower is worse)
    kinks              sharp single-point bends (over KINK_DEG between two straight neighbours), one per place
    spikes             zigzags: two neighbouring bends turning opposite ways, a jag in the line, one per place
    root_stand_p99     how far each root leaves the scalp's surface, degrees (a root should leave along it)
    bare_roots         roots off the cap (on bare skin: fuzz in the hairline)
    flow_dev_p99       how far a strand runs from the strands round it (degrees, area of a lock's width): a smooth
                       flow is near 0; crossing locks, swirls and strands combed against the rest stand out
    crossings          places where strands of neighbouring locks cross each other at over CROSS_DEG (tangles)
    depth_head, depth_body   the deepest a strand goes into the skin, the neck and shoulders (HH)
    buried_pct         share of strand points sunk into the hair's own cap (the cap's surface cuts through them)
  settled by the solver, at rest (roots held, gravity, the head, cap and body as colliders, friction)
    drift_p99, drift_max   how far the strands move to rest (HH). Hair drawn where its weight would keep it barely
                       moves; a lock held up in the air (a visor, a flap, a strand floating off the shoulder) drops
    stretch_p99        segment length after the solve against as drawn (the chain is held: stretch shows a strand
                       caught between colliders)
    fold_radius_p1     the tightest bends after the solve (HH)
  in motion (walk and run from the game's walk cycle, a 90 degree turn, start and stop, a full spin)
    jitter_p99         the shiver left after the motion's own swing is taken out (HH per frame, at 30 fps)
    depth_body_max, depth_head_max   the deepest any strand goes into the body or head in any frame
    stretch_p99, fold_radius_p1      as above, the worst frame
    crossings          the most crossing places in any frame

The band (hair_baseline.json beside this file): each limit is the worst approved value with a 25% margin, as the
flow check's; one approved style far worse than the rest on a metric (over 3x) is listed as an outlier instead.
Nothing blocks. The settled points are cached (the fold cache rule: keyed on the strands, colliders and solver
settings), and so is each style's whole result."""
import gzip
import hashlib
import json
import math
import os
import sys

import numpy as np

VERSION = 1
SOLVE_VERSION = 7         # (bumped whenever the solve itself changes: its cached settles and motions are dropped)
MOTION_VERSION = 3        # (bumped when the motions change (not the solve): only the cached motions are dropped)
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..', '..', 'blender', 'shared'), os.path.join(HERE, 'blender', 'shared'), HERE):
    if os.path.exists(os.path.join(_p, 'fabric.py')):
        sys.path.insert(0, _p)
        break
from fabric import Fabric  # noqa: E402

UNITS = 8.6               # the creator's head height in model units (the figure: 63 units, 165 cm)
GRAVITY = -375.0          # units / s^2 (fabric.py's: 9.8 m/s^2 at that scale)
SOLVER = dict(damping=0.02, iterations=4, relax=1.6, stretch=0.0, shear=2e-5, bend=2e-4, friction=0.5,
              gravity=(0.0, 0.0, GRAVITY))
SHAPE = 0.712              # how hard a strand keeps its drawn shape, per step (see HairFabric): tune_stiffness()
                          # finds it from hair's own stiffness (python3 rq_hair.py --tune)
COHESION = 2e-3          # compliance of the ties between neighbouring strands (loose: they slide, but hold together)
STEP_JUMP = 0.12          # HH: an outline jumping sideways this far between neighbouring rows (160 over the hair) is a ledge
SEG = 0.04                # HH: the strands are resampled to about this for the solver (a lock is about 0.05 across)
KINK_DEG = 35.0           # a bend sharper than this at one point, between straighter neighbours, is a kink
ZIG_DEG = 20.0            # each of two neighbouring bends turning opposite ways past this is a spike (a zigzag)
CROSS_DEG = 45.0          # strands of different locks this close and this far apart in direction cross
NEAR = 0.035              # HH: the flow's neighbourhood (about a lock's width) and the crossing distance
PLACE = 0.08              # HH: faults closer than this are one place
FPS = 30
POSES = ('rest', 'walk', 'run', 'turn', 'startstop', 'spin')
DEFAULT_LIMITS = {
    'kink_radius_p1': 0.02, 'kinks': 20, 'spikes': 20, 'root_stand_p99': 60.0, 'bare_roots': 10,
    'flow_dev_p99': 60.0, 'crossings': 30, 'depth_head': 0.03, 'depth_body': 0.05, 'buried_pct': 10.0,
    'drift_p99': 0.3, 'drift_max': 0.6, 'stretch_p99': 1.1, 'fold_radius_p1': 0.02,
}
MOTION_LIMITS = {'jitter_p99': 0.01, 'depth_body_max': 0.08, 'depth_head_max': 0.05, 'stretch_p99': 1.1,
                 'fold_radius_p1': 0.02, 'crossings': 40}
LOWER_IS_WORSE = ('kink_radius_p1', 'fold_radius_p1')
COUNTS = ('kinks', 'spikes', 'bare_roots', 'crossings')
MARGIN = 1.25


def limits():
    """The band: hair_baseline.json beside this file ({'limits': {...}, 'poses': {pose: {...}}}), else the loose
    defaults (nothing is flagged on guesses until it is calibrated)."""
    try:
        b = json.load(open(os.path.join(HERE, 'hair_baseline.json')))
        return dict(DEFAULT_LIMITS, **b['limits']), {p: dict(MOTION_LIMITS, **v) for p, v in b['poses'].items()}, b
    except (OSError, ValueError, KeyError):
        return dict(DEFAULT_LIMITS), {p: dict(MOTION_LIMITS) for p in POSES[1:]}, None


# ---------------------------------------------------------------------------------------------------- the input

def load_style(path):
    d = json.load(gzip.open(path))
    strands = []
    for s in d['strands']:
        P = np.array(s['p'], float).reshape(-1, 3)
        if len(P) < 3:
            continue
        w = np.array(s.get('w', [0.01] * len(P)), float)
        w = np.linalg.norm(w.reshape(-1, 3), axis=1) if len(w) == 3 * len(P) else w
        lock = int(s.get('lock', -1))
        # (strands drawn outside hairClump carry no lock: each is its own, so the strands of one clump may meet)
        strands.append({'p': P, 'w': w, 't': np.array(s.get('t', [0.005] * len(P)), float),
                        'lock': lock if lock >= 0 else 100000 + len(strands), 'layer': int(s.get('layer', 0))})
    cap = d['cap']
    return {'style': d['style'], 'name': d.get('name', d['style']), 'strands': strands,
            'cap': np.array(cap['p'], float).reshape(-1, 3), 'cap_a': np.array(cap['a'], float)}


def load_world(path):
    d = json.load(gzip.open(path))
    return {'head': np.array(d['head']['p'], float).reshape(-1, 3), 'head_n': np.array(d['head']['n'], float).reshape(-1, 3),
            'bodies': d['bodies']}


# ---------------------------------------------------------------------------------------------------- the head

def measure_head(H):
    """The head, measured from its skin (creator axes: x across, y back, z up from the chin; the face toward -y):
    chin, crown, height, width at the temples, depth from the brow to the back of the skull, and the skull's centre.
    Every limit is a share of the measured height."""
    face = H[H[:, 1] < -0.15]
    chin = float(np.percentile(face[:, 2], 0.5)) if len(face) else float(H[:, 2].min())
    crown = float(H[:, 2].max())
    hh = crown - chin
    band = H[(H[:, 2] > chin + 0.55 * hh) & (H[:, 2] < chin + 0.75 * hh)]
    width = float(band[:, 0].max() - band[:, 0].min())
    depth = float(band[:, 1].max() - band[:, 1].min())
    skull = H[H[:, 2] > chin + 0.5 * hh]
    c = np.array([0.0, 0.5 * (skull[:, 1].max() + skull[:, 1].min()), chin + 0.58 * hh])
    return {'chin': chin, 'crown': crown, 'height': hh, 'width': width, 'depth': depth, 'centre': c.tolist(),
            'width_ratio': width / hh, 'depth_ratio': depth / hh}


class Grid:
    """A signed distance field on a regular grid (trilinear), from points with outward normals."""

    def __init__(self, P, N, lo, hi, cell):
        self.lo, self.cell = np.asarray(lo, float), cell
        self.shape = tuple(int(math.ceil((hi[k] - lo[k]) / cell)) + 1 for k in range(3))
        g = np.stack(np.meshgrid(*[self.lo[k] + cell * np.arange(self.shape[k]) for k in range(3)], indexing='ij'), -1)
        Q = g.reshape(-1, 3)
        try:
            from scipy.spatial import cKDTree
            d, i = cKDTree(P).query(Q, k=4)
        except ImportError:                  # (plain numpy: slower, the same answer)
            d, i = np.zeros((len(Q), 4)), np.zeros((len(Q), 4), dtype=np.int64)
            for a in range(0, len(Q), 2048):
                D = ((Q[a:a + 2048, None, :] - P[None, ::1, :]) ** 2).sum(-1)
                k = np.argpartition(D, 4, axis=1)[:, :4]
                d[a:a + 2048] = np.sqrt(np.take_along_axis(D, k, 1))
                i[a:a + 2048] = k
        side = np.einsum('qkj,qkj->qk', Q[:, None, :] - P[i], N[i])
        sgn = np.where(side.mean(1) < 0, -1.0, 1.0)
        self.v = (sgn * d[:, 0]).reshape(self.shape)

    def __call__(self, X):
        f = (np.asarray(X) - self.lo) / self.cell
        out = np.any((f < 0) | (f > np.array(self.shape) - 1.001), axis=1)
        f = np.clip(f, 0, np.array(self.shape) - 1.001)
        i = np.floor(f).astype(int)
        t = f - i
        v = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wgt = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    v = v + wgt * self.v[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return np.where(out, 1.0, v)


class CapMap:
    """The hair's own cap as a radius round the skull's centre in each direction (where the cap is opaque)."""
    NA, NP = 96, 48

    def __init__(self, cap, alpha, c):
        self.c = np.asarray(c)
        a, p, r = self._dir(cap)
        keep = alpha > 0.5
        acc = np.zeros(self.NA * self.NP)
        cnt = np.zeros(self.NA * self.NP)
        k = (p * self.NA + a)[keep]
        np.add.at(acc, k, r[keep])
        np.add.at(cnt, k, 1)
        R = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).reshape(self.NP, self.NA)
        for _ in range(2):                    # (close the pinholes between the cap's columns)
            nb = np.stack([np.roll(R, 1, 1), np.roll(R, -1, 1), np.roll(R, 1, 0), np.roll(R, -1, 0)])
            fill = np.isnan(R) & (np.sum(~np.isnan(nb), 0) >= 3)
            with np.errstate(all='ignore'):
                R[fill] = (np.nansum(nb, 0) / np.maximum(np.sum(~np.isnan(nb), 0), 1))[fill]
        self.R = R
        A = np.zeros(self.NA * self.NP)
        np.maximum.at(A, p * self.NA + a, alpha)
        A = A.reshape(self.NP, self.NA)
        self.A = np.maximum(A, np.maximum(np.roll(A, 1, 1), np.roll(A, -1, 1)))

    def alpha(self, X):
        """How opaque the cap is in a point's direction (0: no hair there, bare skin)."""
        a, p, _ = self._dir(X)
        return self.A[p, a]

    def _dir(self, X):
        d = np.asarray(X) - self.c
        r = np.linalg.norm(d, axis=1)
        az = np.arctan2(d[:, 0], -d[:, 1])
        po = np.arccos(np.clip(d[:, 2] / np.maximum(r, 1e-9), -1, 1))
        a = np.clip(((az + np.pi) / (2 * np.pi) * self.NA).astype(int), 0, self.NA - 1)
        p = np.clip((po / np.pi * self.NP).astype(int), 0, self.NP - 1)
        return a, p, r

    def __call__(self, X):
        """Distance outside the cap along the ray from the centre (nan where there is no cap)."""
        a, p, r = self._dir(X)
        return r - self.R[p, a]

    def normal(self, X):
        d = np.asarray(X) - self.c
        return d / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]


def body_field(B, kind):
    """The body the hair falls on, in head heights from the chin: the creator's MANNEQUIN table (the new mannequin's
    neck, shoulders and back) or a character's sections (her collar, cape and back), as the creator's makeDrape reads
    them. Returns f(X) -> signed distance (HH, < 0 inside)."""
    CY = 0.04
    if kind == 'mannequin':
        M = B
        z0, dz, na = M['z0'], M['dz'], M['na']
        r = np.array(M['r'], float).reshape(-1, na) * 0.001
        nz = r.shape[0]

        def f(X):
            x, y, z = X[:, 0], X[:, 1] - CY, X[:, 2]
            fz = (z0 - z) / dz
            top = np.where(fz < 0, np.clip((fz + 4) / 4, 0, 1) ** 2 * (3 - 2 * np.clip((fz + 4) / 4, 0, 1)), 1.0)
            fz = np.clip(fz, 0, nz - 1.001)
            th = np.arctan2(x, -y)
            fa = (th + np.pi) / (2 * np.pi) * na
            a0 = np.floor(fa).astype(int) % na
            u = fa - np.floor(fa)
            a1 = (a0 + 1) % na
            r0 = np.floor(fz).astype(int)
            v = fz - r0
            R = ((1 - u) * r[r0, a0] + u * r[r0, a1]) * (1 - v) + ((1 - u) * r[r0 + 1, a0] + u * r[r0 + 1, a1]) * v
            return np.hypot(x, y) - top * R
        return f
    S = np.array(B, float)

    def f(X):
        x, y, z = X[:, 0], X[:, 1] - CY, X[:, 2]
        # (between sections a smoothstep, as the creator's dims(): linear steps put a crease in the surface at every
        # section, and a strand lying across one is pushed two ways in turn)
        k = np.clip(np.searchsorted(-S[:, 0], -z) - 1, 0, len(S) - 2)
        t = np.clip((S[k, 0] - z) / (S[k, 0] - S[k + 1, 0]), 0, 1)
        t = t * t * (3 - 2 * t)
        D = S[k, 1:] + (S[k + 1, 1:] - S[k, 1:]) * t[:, None]
        t = np.clip((S[0, 0] - z) / 0.05, 0, 1)        # (above the top section: nothing)
        ry = np.where(y < 0, D[:, 1], D[:, 2])
        F = (np.abs(x / D[:, 0]) ** 3 + np.abs(y / ry) ** 3) ** (1 / 3) - 1
        return np.where(t > 0, F * np.minimum(D[:, 0], ry), 1.0)
    return f


class World:
    """The head (skin), the hair's cap over it and the body below, as distance fields in the head's frame."""

    def __init__(self, world, style, cache=None):
        H, N = world['head'], world['head_n']
        self.m = measure_head(H)
        hh = self.m['height']
        lo, hi = H.min(0) - 0.12 * hh, H.max(0) + 0.12 * hh
        key = hashlib.sha1(np.round(H, 4).tobytes() + b'grid1').hexdigest()[:16]
        p = os.path.join(cache, 'head-' + key + '.npz') if cache else None
        if p and os.path.exists(p):
            z = np.load(p)
            self.head = Grid.__new__(Grid)
            self.head.lo, self.head.cell, self.head.v = z['lo'], float(z['cell']), z['v']
            self.head.shape = self.head.v.shape
        else:
            self.head = Grid(H, N, lo, hi, 0.012 * hh)
            if p:
                np.savez_compressed(p, lo=self.head.lo, cell=self.head.cell, v=self.head.v)
        self.neck_z = float(H[:, 2].min()) + 0.06 * hh      # (the skin's open edge at the neck: the body takes over)
        self.cap = CapMap(style['cap'], style['cap_a'], self.m['centre'])
        # (the head and the cap as one field on the head's grid, for the solver: one lookup a point instead of two)
        G = self.head
        g = np.stack(np.meshgrid(*[G.lo[k] + G.cell * np.arange(G.shape[k]) for k in range(3)], indexing='ij'), -1)
        g = g.reshape(-1, 3)
        self.shell = Grid.__new__(Grid)
        self.shell.lo, self.shell.cell, self.shell.shape = G.lo, G.cell, G.shape
        self.shell.v = np.minimum(np.where(g[:, 2] > self.neck_z, G.v.ravel(), 1.0), self.cap_d(g)).reshape(G.shape)
        kind = world['bodies'].get('styles', {}).get(style['style'], 'mannequin')
        self.body_kind = kind
        self.body = body_field(world['bodies'][kind], kind)

    def head_d(self, X):
        return np.where(X[:, 2] > self.neck_z, self.head(X), 1.0)

    def cap_d(self, X):
        d = self.cap(X)
        return np.where(np.isnan(d), 1.0, d)


# ---------------------------------------------------------------------------------------------------- poses

def bake_walk(glb, out):
    """The game's walk cycle from a figure's .glb (its Body translation and Chest rotation, 40 keys over 1.67 s), in
    the creator's frame and head heights: hair_poses.json beside this file."""
    import struct
    b = open(glb, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + n])
    bin_ = b[20 + n + 8:]

    def acc(i):
        a = j['accessors'][i]
        v = j['bufferViews'][a['bufferView']]
        k = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[a['type']]
        return np.frombuffer(bin_, np.float32, a['count'] * k, v.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(a['count'], k)
    tr = rot = t = None
    for a in j['animations']:
        for c in a['channels']:
            nm = j['nodes'][c['target']['node']].get('name')
            s = a['samplers'][c['sampler']]
            if a['name'] == 'Body' and nm == 'Body' and c['target']['path'] == 'translation':
                t, tr = acc(s['input'])[:, 0], acc(s['output'])
            if a['name'] == 'Chest' and nm == 'Chest' and c['target']['path'] == 'rotation':
                rot = acc(s['output'])
    # (the clip is one cycle, two steps, then an ease back to standing: keep the cycle, up to the first key that
    # repeats the first, so the loop has no jump)
    for k in range(4, len(t)):
        if np.abs(tr[k] - tr[0]).max() < 1e-3 and np.abs(rot[k] - rot[0]).max() < 1e-4:
            t, tr, rot = t[:k + 1], tr[:k + 1], rot[:k + 1]
            break
    # glTF (x, y up, z front) to the creator's (x, y back, z up), units to head heights
    tr_c = np.stack([tr[:, 0], -tr[:, 2], tr[:, 1]], -1) / UNITS
    q_c = np.stack([rot[:, 0], -rot[:, 2], rot[:, 1], rot[:, 3]], -1)
    json.dump({'walk': {'t': np.round(t, 4).tolist(), 'body': np.round(tr_c, 5).tolist(), 'chest': np.round(q_c, 6).tolist(),
                        'chest_z': (41.0 - 54.2) / UNITS, 'from': os.path.basename(glb)}}, open(out, 'w'))


def _qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _yaw(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _pitch(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def motion(pose, walk):
    """A pose's motion: (duration s, warm-up s, f(t) -> (head R, head origin, body R, body origin)), each a rigid
    move of the head's and the chest's frames from rest, in head heights. The figure faces -y, so forward is -y."""
    W = walk or {'t': [0, 1.667], 'body': [[0, 0, 0]] * 2, 'chest': [[0, 0, 0, 1]] * 2, 'chest_z': -1.535}
    wt, wb, wq = np.array(W['t']), np.array(W['body']), np.array(W['chest'])
    # (the clip's keys start at t0 and its last key repeats the first: the period is the span from the first key to
    # the repeat, not the last key's time, and a periodic cubic spline through the keys gives a smooth speed where
    # np.interp made it jump at every key. John/bun thread, 2026-11-21: the head peaked at 4.9 g in the walk and
    # 14.8 g in the run; a runner's forehead peaks at about 1.55 g, 15.2 m/s^2 (Macdermid et al. 2015))
    from scipy.interpolate import CubicSpline
    T = float(wt[-1] - wt[0])
    n = len(wt) - 1 if np.allclose(wb[0], wb[-1]) and np.allclose(wq[0], wq[-1]) else len(wt)
    kt = np.r_[wt[:n], wt[0] + T]
    csb = CubicSpline(kt, np.vstack([wb[:n], wb[:1]]), bc_type='periodic')
    csq = CubicSpline(kt, np.vstack([wq[:n], wq[:1]]), bc_type='periodic')
    piv = np.array([0, 0.04, W['chest_z']])
    I = np.eye(3)
    G = 9.81 / (HEAD_CM / 100.0)               # gravity in head heights / s^2

    def cycle(t, rate=1.0, amp=1.0):
        u = wt[0] + (t * rate) % T
        b = csb(u) * amp
        q = csq(u)
        q[:3] = q[:3] * amp
        q /= np.linalg.norm(q)
        return b, _qmat(q)

    def peak_g(rate, amp):
        """The head's largest acceleration over the cycle, in g (its origin's second difference)."""
        ts = np.linspace(0, T / rate, 400, endpoint=False)
        b = np.array([cycle(t, rate, amp)[0] for t in ts])
        dt = ts[1] - ts[0]
        a = np.diff(np.vstack([b[-1:], b, b[:1]]), 2, axis=0) / dt ** 2
        return float(np.linalg.norm(a, axis=1).max() / G)

    def run_amp(rate=1.7, g=1.55):
        """The swing of the run cycle (a share of the walk's) that gives the head a runner's 1.55 g peak."""
        lo, hi = 0.05, 4.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if peak_g(rate, mid) < g else (lo, mid)
        return 0.5 * (lo + hi)
    if pose == 'walk':
        def f(t):
            b, R = cycle(t)
            return I, b, R, b + piv - R @ piv
        return 2 * T, T, f
    if pose == 'run':
        RUN_AMP = run_amp()
        lean = _pitch(math.radians(-8))
        def f(t):
            b, R = cycle(t, 1.7, RUN_AMP)
            Rb = lean @ R
            return lean, b, Rb, b + piv - Rb @ piv
        return 2 * T / 1.7 + 0.6, T / 1.7 + 0.6, f
    if pose in ('turn', 'spin'):
        ang, dur = (math.pi / 2, 0.6) if pose == 'turn' else (2 * math.pi, 1.2)
        ax = np.array([0, 0.04, 0])
        def f(t):
            R = _yaw(ang * _smooth(t / dur))
            o = ax - R @ ax
            return R, o, R, o
        return dur + 0.8, 0.0, f
    if pose == 'startstop':
        v, ramp, hold = 53.0 / UNITS, 0.3, 0.6         # (a walk's 1.4 m/s, reached and lost in 0.3 s)
        def pos(t):
            seg = [(ramp, 'up'), (hold, 'go'), (ramp, 'down'), (0.8, 'still')]
            x, t0 = 0.0, 0.0
            for d, k in seg:
                u = min(max(t - t0, 0.0), d)
                x += v * (u * u / (2 * ramp) if k == 'up' else u if k == 'go' else (u - u * u / (2 * ramp)) if k == 'down' else 0.0)
                t0 += d
            return np.array([0, -x, 0])
        def f(t):
            o = pos(t)
            return I, o, I, o
        return 2 * ramp + hold + 0.8, 0.0, f
    raise ValueError(pose)


# ---------------------------------------------------------------------------------------------------- strands

def resample(P, seg):
    """A polyline resampled to about seg apart (its ends kept)."""
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(3, int(round(L[-1] / seg)) + 1)
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, P[:, k]) for k in range(3)], -1)


def smooth_line(P, width):
    """The strand's body without its waves: a Gaussian along its length (width in points), the ends held."""
    if width < 1 or len(P) < 5:
        return P.copy()
    k = np.arange(-int(3 * width), int(3 * width) + 1)
    g = np.exp(-0.5 * (k / width) ** 2)
    Q = np.empty_like(P)
    n = len(P)
    for i in range(n):
        j = np.clip(i + k, 0, n - 1)
        Q[i] = (g[:, None] * P[j]).sum(0) / g.sum()
    Q[0] = P[0]
    return Q


def bends(P):
    """Turning angle (degrees) and radius of curvature at each inner point, and each bend's turning axis."""
    d = np.diff(P, axis=0)
    l = np.linalg.norm(d, axis=1)
    u = d / np.maximum(l, 1e-9)[:, None]
    c = np.clip((u[:-1] * u[1:]).sum(1), -1, 1)
    ang = np.degrees(np.arccos(c))
    rad = 0.5 * (l[:-1] + l[1:]) / np.maximum(np.radians(ang), 1e-6)
    ax = np.cross(u[:-1], u[1:])
    return ang, rad, ax, u, l


def places(pts, scores, spacing=PLACE, most=None):
    """Faults grouped into places: the worst first, each place the faults within spacing of it."""
    order = np.argsort(-np.asarray(scores))
    kept = []
    for i in order:
        if all(np.linalg.norm(pts[i] - pts[j]) > spacing for j in kept):
            kept.append(i)
            if most and len(kept) >= most:
                break
    return kept


def neighbours(X, r):
    """Pairs of points closer than r (a grid hash; numpy only)."""
    cell = np.floor(X / r).astype(np.int64)
    key = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    order = np.argsort(key, kind='stable')
    sk = key[order]
    A, B = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c2 = cell + np.array([dx, dy, dz])
                k2 = (c2[:, 0] * 73856093) ^ (c2[:, 1] * 19349663) ^ (c2[:, 2] * 83492791)
                lo = np.searchsorted(sk, k2, 'left')
                hi = np.searchsorted(sk, k2, 'right')
                cnt = hi - lo
                if cnt.sum() == 0:
                    continue
                a = np.repeat(np.arange(len(X)), cnt)
                b = order[np.concatenate([np.arange(l0, h0) for l0, h0 in zip(lo[cnt > 0], hi[cnt > 0])])] \
                    if cnt.max() > 0 else np.zeros(0, int)
                ok = (a < b) & (np.linalg.norm(X[a] - X[b], axis=1) < r)
                A.append(a[ok])
                B.append(b[ok])
    if not A:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(A), np.concatenate(B)


def flow_field(X, T, lock, layer, r, hh):
    """Per point, how far its direction is from the strands round it (degrees), and the crossings: pairs of points of
    different locks on one layer, nearer than r, running more than CROSS_DEG apart."""
    a, b = neighbours(X, r * hh)
    other = lock[a] != lock[b]
    same = layer[a] == layer[b]
    dev = np.zeros(len(X))
    acc = np.zeros_like(X)
    cnt = np.zeros(len(X))
    o = other
    np.add.at(acc, a[o], T[b[o]])
    np.add.at(acc, b[o], T[a[o]])
    np.add.at(cnt, a[o], 1)
    np.add.at(cnt, b[o], 1)
    m = cnt > 0
    mean = acc[m] / np.maximum(np.linalg.norm(acc[m], axis=1), 1e-9)[:, None]
    dev[m] = np.degrees(np.arccos(np.clip((T[m] * mean).sum(1), -1, 1)))
    c = np.degrees(np.arccos(np.clip(np.abs((T[a] * T[b]).sum(1)), 0, 1)))
    x = other & same & (c > CROSS_DEG)
    return dev, a[x], b[x], c[x]


def flatten(strands, key='p'):
    X = np.concatenate([s[key] for s in strands])
    sid = np.concatenate([np.full(len(s[key]), k) for k, s in enumerate(strands)])
    first = np.r_[0, np.cumsum([len(s[key]) for s in strands])[:-1]]
    return X, sid, first


def tangents(strands, key='p'):
    out = []
    for s in strands:
        P = s[key]
        d = np.gradient(P, axis=0)
        out.append(d / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None])
    return np.concatenate(out)


def place_name(p, m):
    """Where on the head or body a point is: 'left temple', 'nape', 'right shoulder, back'. The face is toward -y and
    her left is +x."""
    hh, chin = m['height'], m['chin']
    x, y, z = p
    h = (z - chin) / hh
    side = 'left' if x > 0.03 * hh else 'right' if x < -0.03 * hh else ''
    front = y < 0.0
    if h > 0.92:
        r = 'crown'
    elif h > 0.55:
        r = ('fringe' if abs(x) < 0.12 * hh else 'temple') if front and y < -0.2 * hh else \
            'side of the head' if abs(x) > 0.25 * hh else 'back of the head'
    elif h > 0.2:
        r = ('cheek' if front and y < -0.1 * hh else 'ear' if abs(x) > 0.3 * hh else 'nape' if y > 0.15 * hh else 'jaw')
    elif h > -0.15:
        r = 'jaw' if front else 'nape' if abs(x) < 0.2 * hh else 'neck'
    elif h > -0.45:
        r = 'shoulder' if abs(x) > 0.3 * hh else 'collar' if front else 'upper back'
    else:
        r = 'chest' if front else 'back'
    face = '' if r in ('crown', 'fringe', 'temple', 'cheek', 'jaw', 'nape', 'back of the head', 'collar',
                       'upper back', 'chest', 'back') else (', front' if front else ', back')
    return (side + ' ' + r).strip() + face if r not in ('fringe', 'crown', 'nape') or not side else (side + ' ' + r)


# ---------------------------------------------------------------------------------------------------- the checks

def check_drawn(strands, W):
    """The strands as drawn: bends, zigzags, roots, the flow between strands, and how deep they go into the skin,
    the body and the hair's own cap. Returns (summary, per-point score fields, fault list)."""
    m = W.m
    hh = m['height']
    X, sid, first = flatten(strands)
    n = len(X)
    T = tangents(strands)
    lock = np.array([strands[k]['lock'] for k in sid])
    layer = np.array([strands[k]['layer'] for k in sid])
    thick = np.concatenate([np.maximum(s['t'], 0.002) for s in strands])
    kink = np.zeros(n)
    rad_all = []
    faults = []
    for k, s in enumerate(strands):
        P = s['p']
        if len(P) < 4:
            continue
        ang, rad, ax, u, l = bends(P)
        rad_all.append(rad / hh)
        o = first[k] + 1
        # a kink: a sharp bend at one point with straighter neighbours either side
        nb = np.maximum(np.r_[0, ang[:-1]], np.r_[ang[1:], 0])
        kk = np.where((ang > KINK_DEG) & (ang > 2 * nb))[0]
        for i in kk:
            kink[o + i] = max(kink[o + i], ang[i] / KINK_DEG)
            faults.append(('kink', o + i, float(ang[i]), f'bends {ang[i]:.0f} degrees at one point'))
        # a spike: two neighbouring bends turning opposite ways (a jag)
        opp = (ax[:-1] * ax[1:]).sum(1) < 0
        zz = np.where(opp & (ang[:-1] > ZIG_DEG) & (ang[1:] > ZIG_DEG))[0]
        for i in zz:
            faults.append(('spike', o + i, float(min(ang[i], ang[i + 1])), f'zigzags {ang[i]:.0f}/{ang[i + 1]:.0f} degrees'))
    rad_all = np.concatenate(rad_all) if rad_all else np.array([1.0])
    # roots: the first point, its cap and its direction against the scalp
    R0 = X[first]
    T0 = T[first]
    nrm = W.cap.normal(R0)
    stand = np.degrees(np.arcsin(np.clip((T0 * nrm).sum(1), -1, 1)))
    bare = (W.cap.alpha(R0) < 0.2) & (R0[:, 2] > m['chin'] + 0.3 * hh)
    for k in np.where(bare)[0]:
        faults.append(('bare root', first[k], 1.0, 'rooted off the cap, on bare skin'))
    # the flow between strands
    dev, ca, cb, cang = flow_field(X, T, lock, layer, NEAR, hh)
    # depths: into the skin, the body, the hair's own cap
    dh = W.head_d(X)
    db = W.body(X)
    dc = W.cap_d(X)
    in_head = np.maximum(0, -(dh + 0.25 * thick)) / hh
    in_body = np.maximum(0, -(db + 0.25 * thick)) / hh
    buried = (dc < -np.maximum(2.5 * thick, 0.012 * hh)) & (np.arange(n) != np.repeat(first, np.diff(np.r_[first, n])))
    root_zone = np.zeros(n, bool)
    for k, f0 in enumerate(first):                        # (a strand's first few points grow out of the cap)
        root_zone[f0:f0 + max(2, len(strands[k]['p']) // 8)] = True
    buried &= ~root_zone
    for i in places(X, in_head, most=6):
        if in_head[i] > 0.004:
            faults.append(('into the head', i, float(in_head[i]), f'{in_head[i]:.3f} HH into the skin'))
    for i in places(X, in_body, most=6):
        if in_body[i] > 0.004:
            faults.append(('into the body', i, float(in_body[i]), f'{in_body[i]:.3f} HH into the neck or shoulders'))
    cross_pts = 0.5 * (X[ca] + X[cb])
    cross_keep = places(cross_pts, cang) if len(ca) else []
    for i in cross_keep[:12]:
        faults.append(('crossing', int(ca[i]), float(cang[i]), f'locks {lock[ca[i]]} and {lock[cb[i]]} cross at {cang[i]:.0f} degrees'))
    kink_pts = [f[1] for f in faults if f[0] == 'kink']
    spike_pts = [f[1] for f in faults if f[0] == 'spike']
    S = {
        'strands': len(strands), 'points': int(n), 'locks': int(len(set(lock.tolist()))),
        'kink_radius_p1': round(float(np.percentile(rad_all, 1)), 4),
        'kinks': len(places(X[kink_pts], np.ones(len(kink_pts)))) if kink_pts else 0,
        'spikes': len(places(X[spike_pts], np.ones(len(spike_pts)))) if spike_pts else 0,
        'root_stand_p99': round(float(np.percentile(np.abs(stand), 99)), 2),
        'bare_roots': int(bare.sum()),
        'flow_dev_p99': round(float(np.percentile(dev, 99)), 2),
        'crossings': len(cross_keep),
        'depth_head': round(float(in_head.max()), 4), 'depth_body': round(float(in_body.max()), 4),
        'buried_pct': round(100.0 * float(buried.mean()), 2),
    }
    root_score = np.zeros(n)
    root_score[first] = np.abs(stand) / 60.0
    F = {'X': X, 'kink': kink, 'flow': dev, 'head': in_head, 'body': in_body, 'buried': buried.astype(float),
         'root': root_score, 'cross': np.bincount(np.r_[ca, cb], minlength=n).astype(float)}
    return S, F, faults


def _rotate_min(a, b, v):
    """v turned by the smallest rotation that takes unit a onto unit b (row-wise, Rodrigues)."""
    c = np.clip((a * b).sum(1), -1.0, 1.0)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax, axis=1)
    k = ax / np.maximum(s, 1e-12)[:, None]
    kv = np.cross(k, v)
    kd = (k * v).sum(1)[:, None]
    out = v * c[:, None] + kv * s[:, None] + k * kd * (1 - c)[:, None]
    return np.where((s > 1e-9)[:, None], out, v)


class HairFabric(Fabric):
    """fabric.py's solver with a hair strand's bending stiffness. Distance ties have no stiffness against a small
    bend of a straight run, so the held chains (follow the leader, root to tip) also keep each strand's drawn shape:
    every segment is turned part of the way (`shape`, per step) toward its drawn direction carried along by its parent
    segment (a local shape constraint, as TressFX's and other game hair solvers keep a strand's style), then set at
    its length. tune_stiffness() sets `shape` from hair's own stiffness against its weight."""
    shape = 0.712
    head_R = np.eye(3)       # the head's turn from rest: the drawn directions turn with it (see hold)

    def step(self, h, body=None, neutral=False):
        """fabric.py's step with the body left to the chains: each strand is walked root to tip and every point,
        once set at its length from the one above, is pushed out of the body there and then (as the creator's hold
        does), so the points below follow the pushed one. Pushed after the walk instead, the walk pulled a strand
        lying on the shoulders back in and the push set it out again, every step: it shivered. The contact is
        inelastic (the speed into or out of the surface is taken away) and slowed along it by friction."""
        self._body = body
        super().step(h, None, neutral)

    def hold(self, body=None, neutral=False):
        C, L, lead = self.chains
        x, pv, w = self.pos, self.prev, self.w
        # (the drawn directions turn with the head: the shape target is the strand's drawn shape on the head as it is
        # now. Left at rest, a head yaw is lost by the minimal rotation from parent to child, which has no twist about
        # the parent's axis, and a curl round the vertical is pulled back toward its unturned shape. Found by the bun
        # thread: 0.20 HH of shape error after a 90 degree turn, 0.012 with this.)
        R = self._rest_dir @ self.head_R.T
        for k in range(1, C.shape[1]):
            m = C[:, k] >= 0
            i, j, r = C[m, k], C[m, k - 1], L[m, k - 1]
            d = x[i] - x[j]
            ln = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            u = d / ln[:, None]
            if k >= 2:
                jj = C[m, k - 2]
                pc = x[j] - x[jj]
                pc /= np.maximum(np.linalg.norm(pc, axis=1), 1e-12)[:, None]
                tgt = _rotate_min(R[m, k - 2], pc, R[m, k - 1])
                u = (1 - self.shape) * u + self.shape * tgt
                u /= np.maximum(np.linalg.norm(u, axis=1), 1e-12)[:, None]
            p = x[j] + u * r[:, None]
            f = w[i] > 0
            up = f & (w[j] > 0)
            pv[j[up]] += lead * (p[up] - x[i[up]])
            bd = getattr(self, '_body', None)
            if bd is not None:
                q, touch = bd(p, i)
                t = touch & f
                if t.any():
                    push = q[t] - p[t]
                    nl = np.linalg.norm(push, axis=1)
                    nn = push / np.maximum(nl, 1e-12)[:, None]
                    v = q[t] - pv[i[t]]
                    v -= nn * (v * nn).sum(1)[:, None]
                    sn = np.linalg.norm(v, axis=1)
                    mu = self.s['friction']
                    kf = np.where(sn <= mu * nl, 0.0, 1.0 - mu * nl / np.maximum(sn, 1e-12))
                    pv[i[t]] = q[t] - v * kf[:, None]
                    p[t] = q[t]
            x[i[f]] = p[f]

    def chain(self, runs, lead=0.9):
        super().chain(runs, lead)
        C, L, _ = self.chains
        D = np.zeros(C[:, 1:].shape + (3,))
        ok = C[:, 1:] >= 0
        d = self.pos[C[:, 1:][ok]] - self.pos[C[:, :-1][ok]]
        D[ok] = d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
        self._rest_dir = D


HEAD_CM = 165.0 / 63.0 * UNITS     # the creator's head height in centimetres (22.5: the figure 165 cm, 63 units)
# A hair: Young's modulus 4 GPa, 70 micrometres across, 1300 kg/m^3. A lock is many hairs side by side: its bending
# stiffness and its weight both grow with the count, so it bends under itself over the same length as one hair.
HAIR_E, HAIR_D, HAIR_RHO = 4e9, 70e-6, 1300.0


def elastogravity_hh(head_cm=HEAD_CM):
    """The length over which a hair holds its shape against its own weight, (E I / weight per length)^(1/3), in
    head heights: about 4.6 cm, 0.2 of the creator's head."""
    I = math.pi * HAIR_D ** 4 / 64
    lam = HAIR_RHO * math.pi * HAIR_D ** 2 / 4 * 9.81
    return (HAIR_E * I / lam) ** (1 / 3) * 100.0 / head_cm


def cantilever_sag(shape, L, seg=SEG):
    """A straight strand of length L (HH) held level at its root, settled: how far its tip drops, as a share of L."""
    n = int(round(L / seg)) + 1
    X = np.c_[np.linspace(0, L, n), np.zeros(n), np.zeros(n)] * UNITS
    f = HairFabric(X, SOLVER)
    f.pin([0, 1])
    r = np.arange(n)
    f.tie(r[:-2], r[2:], SOLVER['bend'])
    f.chain([r])
    f.shape = shape
    f.settle(None, h=1 / 120, most=3000, tol=1e-5)
    return float(-f.pos[-1, 2] / UNITS / L)


def tune_stiffness(head_cm=HEAD_CM):
    """SHAPE from hair's own stiffness: the strand held level at its root sags an eighth of its length at the
    elastogravity length (a cantilever of that length under its own weight, w L^4 / 8 E I = L / 8). Bisection on
    the solver itself, so it holds for these steps and segments. (The solver's sag grows about with the length, not
    its cube as a beam's does: short strands are a little softer than real hair, long ones stiffer. Fine for a
    check of where hair is held up; not a groom simulation.)"""
    L = elastogravity_hh(head_cm)
    lo, hi = 0.5, 0.99
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if cantilever_sag(mid, L) > 0.125:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 3), L


def build_solver(strands, W):
    """The strands as the solver's chains, each a strand of real hair rather than a string: resampled to SEG, the
    waves taken out (the solver moves the strand's body; the waves ride on it, as the creator's drape does), held at
    the root along its first segment (a follicle sets the direction a hair leaves the scalp), kept to its length
    exactly (the held chain), stiff against bending over a few spans (ties 2, 3 and 5 points along, as groom solvers
    keep a guide's rest shape), and held loosely to the strands round it (the hair's own mass holds a lock where it
    lies; the creator's drape ties neighbouring strands the same way)."""
    hh = W.m['height']
    runs, P, roots, thick = [], [], [], []
    n = 0
    for s in strands:
        Q = resample(s['p'], SEG * hh)
        Q = smooth_line(Q, 1.5)
        t = np.interp(np.linspace(0, 1, len(Q)), np.linspace(0, 1, len(s['t'])), s['t'])
        runs.append(np.arange(n, n + len(Q)))
        roots += [n, n + 1]
        n += len(Q)
        P.append(Q)
        thick.append(t)
    X = np.concatenate(P)
    f = HairFabric(X * UNITS, SOLVER)
    f.pin(np.array(roots))
    for r in runs:
        if len(r) > 2:
            f.tie(r[:-2], r[2:], SOLVER['bend'])
    f.shape = SHAPE
    sid = np.concatenate([np.full(len(r), k) for k, r in enumerate(runs)])
    a, b = neighbours(X, 0.6 * NEAR * hh)
    o = sid[a] != sid[b]
    if o.any():
        f.tie(a[o], b[o], COHESION)
    f.chain(runs)
    return f, runs, np.array(roots), X, np.concatenate(thick)


def collider(W, thick, hh, frame=None):
    """Fabric's body: points pushed out of the head, the cap and the body to the strand's own half-thickness, along
    the fields' gradient. frame(t) gives the head's and the body's moves; None: at rest."""
    gap = np.maximum(thick, 0.004 * hh)
    e = 0.004 * hh
    state = {'Rh': np.eye(3), 'oh': np.zeros(3), 'Rb': np.eye(3), 'ob': np.zeros(3)}

    def field(Y):
        Yh = (Y - state['oh']) @ state['Rh']           # (into the head's frame)
        Yb = (Y - state['ob']) @ state['Rb']           # (and the body's)
        return np.minimum(W.shell(Yh), W.body(Yb))

    def body(x, which=None):
        gp = gap if which is None else gap[which]
        Y = x / UNITS
        d = field(Y) - gp
        touch = d < 0
        if not touch.any():
            return x, touch
        Y = Y.copy()
        idx = np.where(touch)[0]
        for _ in range(3):                    # (a few Newton steps: the fields are distances, nearly)
            Z = Y[idx]
            dz = field(Z) - gp[idx]
            m = dz < 0
            if not m.any():
                break
            idx, Z, dz = idx[m], Z[m], dz[m]
            g = np.stack([(field(Z + e * np.eye(3)[k]) - field(Z - e * np.eye(3)[k])) / (2 * e) for k in range(3)], -1)
            g /= np.maximum(np.linalg.norm(g, axis=1), 1e-9)[:, None]
            Y[idx] = Z - dz[:, None] * g
        return Y * UNITS, touch
    return body, state


def solver_metrics(X0, X, runs, hh):
    """Stretch and the tightest bends of the solved chains against their rest."""
    st, rad = [], []
    for r in runs:
        a, b = X0[r], X[r]
        l0 = np.linalg.norm(np.diff(a, axis=0), axis=1)
        l1 = np.linalg.norm(np.diff(b, axis=0), axis=1)
        st.append(l1 / np.maximum(l0, 1e-9))
        if len(r) > 3:
            rad.append(bends(b)[1] / hh)
    st = np.concatenate(st)
    rad = np.concatenate(rad) if rad else np.array([1.0])
    return float(np.percentile(st, 99)), float(np.percentile(rad, 1)), st


def settle(strands, W, cache=None):
    """The strands run to rest on the head and body by the solver. Returns (rest points, settled points, runs,
    roots, thick, steps). Cached on the strands, the colliders and the solver settings (the fold cache rule)."""
    f, runs, roots, X0, thick = build_solver(strands, W)
    hh = W.m['height']
    h = hashlib.sha1(np.round(X0, 4).tobytes() + repr((VERSION, SOLVE_VERSION, SOLVER, SEG, COHESION, SHAPE, W.body_kind)).encode()
                     + np.round(W.head.v[::3, ::3, ::3], 3).tobytes() + np.nan_to_num(np.round(W.cap.R, 3)).tobytes())
    key = h.hexdigest()[:20]
    p = os.path.join(cache, 'settle-' + key + '.npz') if cache else None
    if p and os.path.exists(p):
        z = np.load(p)
        return X0, z['X'], runs, roots, thick, int(z['steps']), True
    body, _ = collider(W, thick, hh)
    steps = f.settle(body, h=1 / 120, most=1500, tol=2e-4 * UNITS)
    X = f.pos / UNITS
    if p:
        # (the drape and the check share this: the settled strands, each segment's strain against as drawn, and the
        # rest shape they were solved from; the renders load the drape written from it, see write_drape)
        st = np.concatenate([np.linalg.norm(np.diff(X[r], axis=0), axis=1) /
                             np.maximum(np.linalg.norm(np.diff(X0[r], axis=0), axis=1), 1e-9) for r in runs])
        np.savez_compressed(p, X=X, X0=X0, steps=steps, strain=st.astype(np.float32),
                            runs=np.array([len(r) for r in runs]))
    return X0, X, runs, roots, thick, steps, False


def _arc(P):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return L / max(L[-1], 1e-12)


def _at(P, s, u):
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], -1)


def drape_strands(strands_out, X0, X, runs):
    """The settle carried onto strands as another export draws them (export-hair.cjs's, which the Blender figures
    build from: the same strands in the same order, sampled differently): each point of a strand is found on the
    solved strand by its share of the length, moved with the strand's body, and its wave (how far it stands off
    the body as drawn) turned with the body's direction there and put back. Points in head heights, the creator's
    axes. Returns the moved points per strand."""
    out = []
    for P, r in zip(strands_out, runs):
        P = np.asarray(P, float)
        u = _arc(P)
        A, B = X0[r], X[r]
        sa = _arc(A)
        a, b = _at(A, sa, u), _at(B, sa, u)
        ta = np.gradient(A, axis=0)
        tb = np.gradient(B, axis=0)
        ta, tb = _at(ta, sa, u), _at(tb, sa, u)
        ta /= np.maximum(np.linalg.norm(ta, axis=1), 1e-12)[:, None]
        tb /= np.maximum(np.linalg.norm(tb, axis=1), 1e-12)[:, None]
        out.append(b + _rotate_min(ta, tb, P - a))
    return out


def write_drape(export_path, solve_src, out_path, world, cache):
    """The drape for the renders: export_path (an export-hair.cjs file, as blender/warden/hair_creator.json.gz) with
    its strands moved to where the cached settle put them. The settle is read from the cache (solved once if it is
    not there), so the renders show the strands the check measured."""
    import copy
    d = json.load(gzip.open(export_path))
    st = load_style(solve_src)
    W = World(world, st, cache)
    X0, X, runs, roots, thick, steps, cached = settle(st['strands'], W, cache)
    keep = [k for k, s in enumerate(d['strands']) if len(s['p']) >= 3 * 3]
    if len(keep) != len(runs):
        raise SystemExit(f'{export_path}: {len(keep)} strands, the solve has {len(runs)}: not the same style')
    moved = drape_strands([np.array(d['strands'][k]['p'], float).reshape(-1, 3) for k in keep], X0, X, runs)
    o = copy.deepcopy(d)
    for k, P in zip(keep, moved):
        o['strands'][k]['p'] = [round(float(v), 4) for v in P.ravel()]
    o['drape'] = {'solver': 'tools/render-queue/rq_hair.py', 'version': VERSION, 'solve': SOLVE_VERSION,
                  'body': W.body_kind, 'steps': int(steps), 'cached': bool(cached)}
    with gzip.open(out_path, 'wt') as f:
        json.dump(o, f)
    return o['drape']


def write_motion(export_path, solve_src, pose, out_path, world, cache):
    """The solved motion for playback: every cached frame of a pose (the strands as the solver moved them, in the
    head's frame, 30 fps) carried onto the export's strands the way write_drape does the rest settle. Written as
    an .npz: `frames` [frame, point, 3] in head heights in the creator's axes, the export's points in its own order
    (`counts` per strand), `fps`, and what it was solved from. A Blender figure sets it as a point cache or shape
    keys on its hair; the frames are in the head's frame, so the hair follows the head bone. Read from the same
    cache as the check: not solved again unless the hair, the settings or the motion changed."""
    d = json.load(gzip.open(export_path))
    st = load_style(solve_src)
    W = World(world, st, cache)
    settled = settle(st['strands'], W, cache)
    X0, X, runs, roots, thick, steps, _ = settled
    keep = [k for k, s in enumerate(d['strands']) if len(s['p']) >= 9]
    if len(keep) != len(runs):
        raise SystemExit(f'{export_path}: {len(keep)} strands, the solve has {len(runs)}: not the same style')
    S, Fp, _ = check_pose(pose, st["strands"], W, settled[:5], walk_keys(), cache)
    P0 = [np.array(d['strands'][k]['p'], float).reshape(-1, 3) for k in keep]
    frames = []
    for Y in Fp['frames']:
        # (each frame's strands, with their waves carried as in the rest drape)
        frames.append(np.concatenate(drape_strands(P0, X0, Y, runs)))
    np.savez_compressed(out_path, frames=np.array(frames, np.float32), counts=np.array([len(p) for p in P0]),
                        fps=FPS, pose=pose, version=VERSION, solve=SOLVE_VERSION, shape=SHAPE,
                        solver=json.dumps(SOLVER, default=list), frame='head, head heights, creator axes')
    return {'pose': pose, 'frames': len(frames), 'strands': len(P0), 'cached': S['cached']}


def check_settle(strands, W, cache=None):
    X0, X, runs, roots, thick, steps, cached = settle(strands, W, cache)
    hh = W.m['height']
    drift = np.linalg.norm(X - X0, axis=1) / hh
    sp99, rad1, st = solver_metrics(X0, X, runs, hh)
    faults = []
    for i in places(X0, drift, spacing=PLACE * hh, most=8):
        if drift[i] > 0.08:
            faults.append(('drops', i, float(drift[i]), f'moves {drift[i]:.2f} HH to rest under its own weight'))
    S = {'drift_p99': round(float(np.percentile(drift, 99)), 4), 'drift_max': round(float(drift.max()), 4),
         'stretch_p99': round(sp99, 4), 'fold_radius_p1': round(rad1, 4), 'steps': steps, 'cached': cached}
    return S, {'X0': X0, 'X': X, 'drift': drift, 'runs': runs}, faults, (X0, X, runs, roots, thick)


def check_pose(pose, strands, W, settled, walk, cache=None):
    """One motion: the settled hair carried by the head (roots held to it), the body moving under it, solved step by
    step at 120 Hz and read at 30 fps in the head's frame. Cached like the settle."""
    X0, Xs, runs, roots, thick = settled
    hh = W.m['height']
    dur, warm, mv = motion(pose, walk)
    key = hashlib.sha1(np.round(Xs, 4).tobytes() + repr((VERSION, SOLVE_VERSION, MOTION_VERSION, pose, SOLVER, SHAPE, COHESION, json.dumps(walk, sort_keys=True))).encode()).hexdigest()[:20]
    p = os.path.join(cache, f'pose-{key}.npz') if cache else None
    if p and os.path.exists(p):
        frames = np.load(p)['frames']
        cached = True
    else:
        f, runs2, roots2, _, _ = build_solver(strands, W)
        f.pos[:] = Xs * UNITS
        f.prev[:] = f.pos
        body, state = collider(W, thick, hh)
        root_home = Xs[roots] * UNITS
        h = 1 / 120
        frames = []
        n = int(round((dur + warm) / h))
        for k in range(n + 1):
            t = k * h
            Rh, oh, Rb, ob = mv(t)
            state.update(Rh=Rh, oh=oh, Rb=Rb, ob=ob)
            f.pos[roots] = root_home @ Rh.T + oh * UNITS
            f.head_R = Rh
            f.step(h, body)
            if t >= warm and k % 4 == 0:
                frames.append(((f.pos / UNITS - oh) @ Rh))
        frames = np.array(frames)
        cached = False
        if p:
            np.savez_compressed(p, frames=frames.astype(np.float32))
    frames = frames.astype(float)
    # jitter: what is left of each point's path after a smooth (3-frame) pass, where it flips side every frame
    # (a smooth swing leaves the same side of its path frame after frame; a shiver flips side every frame)
    r = frames[1:-1] - 0.25 * (frames[:-2] + 2 * frames[1:-1] + frames[2:])
    flip = np.maximum(0.0, -(r[:-1] * r[1:]).sum(-1))
    jit = np.sqrt(flip.mean(0)) / hh
    jit_max = np.sqrt(flip.max(0)) / hh
    # depths, stretch, bends and crossings per frame (in the head's frame the body moves: carry it in)
    dur_, warm_, mv_ = motion(pose, walk)
    db_max = np.zeros(len(Xs))
    dh_max = np.zeros(len(Xs))
    st_w, rad_w, cross_w = 1.0, 1.0, 0
    worst_frame = 0
    lock = np.concatenate([np.full(len(rr), strands[k]['lock']) for k, rr in enumerate(runs)])
    layer = np.concatenate([np.full(len(rr), strands[k]['layer']) for k, rr in enumerate(runs)])
    for fi, Y in enumerate(frames):
        t = warm_ + fi * 4 / 120
        Rh, oh, Rb, ob = mv_(t)
        Yw = Y @ Rh.T + oh
        Yb = (Yw - ob) @ Rb
        db = np.maximum(0, -(W.body(Yb) + 0.25 * thick)) / hh
        dh = np.maximum(0, -(W.head_d(Y) + 0.25 * thick)) / hh
        db_max = np.maximum(db_max, db)
        dh_max = np.maximum(dh_max, dh)
        if fi % 3 == 0:
            s99, r1, _ = solver_metrics(X0, Y, runs, hh)
            if s99 > st_w:
                st_w = s99
            rad_w = min(rad_w, r1)
            T = np.concatenate([np.gradient(Y[rr], axis=0) for rr in runs])
            T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
            _, ca, cb, cang = flow_field(Y, T, lock, layer, NEAR, hh)
            c = len(places(0.5 * (Y[ca] + Y[cb]), cang)) if len(ca) else 0
            if c > cross_w:
                cross_w, worst_frame = c, fi
    S = {'jitter_p99': round(float(np.percentile(jit, 99)), 5), 'jitter_max': round(float(jit_max.max()), 4),
         'depth_body_max': round(float(db_max.max()), 4), 'depth_head_max': round(float(dh_max.max()), 4),
         'stretch_p99': round(st_w, 4), 'fold_radius_p1': round(rad_w, 4), 'crossings': int(cross_w),
         'swing_max': round(float(np.linalg.norm(frames - Xs[None], axis=-1).max() / hh), 3),
         'frames': int(len(frames)), 'cached': cached}
    faults = []
    for i in places(Xs, jit, spacing=PLACE * hh, most=4):
        if jit[i] > 0.004:
            faults.append(('jitter', i, float(jit[i]), f'shivers {jit[i]:.3f} HH a frame in the {pose}'))
    for i in places(Xs, db_max, spacing=PLACE * hh, most=4):
        if db_max[i] > 0.01:
            faults.append(('through the body', i, float(db_max[i]), f'goes {db_max[i]:.3f} HH into the body in the {pose}'))
    return S, {'jit': jit, 'db': db_max, 'dh': dh_max, 'frames': frames, 'worst': worst_frame}, faults


# ---------------------------------------------------------------------------------------------------- verdicts

def judge(S, lim):
    flags, worst = [], 'ok'
    for k, v in lim.items():
        x = S.get(k)
        if x is None or v is None:
            continue
        low = k in LOWER_IS_WORSE
        if not ((x < v) if low else (x > v)):
            continue
        far = (x < v / 1.5) if low else (x > v + max(3, v)) if k in COUNTS else (x > v * 1.5)
        lvl = 'likely rejection' if far else 'watch'
        flags.append({'metric': k, 'value': x, 'limit': v, 'level': lvl})
        if lvl == 'likely rejection' or worst == 'ok':
            worst = lvl
    flags.sort(key=lambda f: (f['level'] != 'likely rejection',))
    return flags, worst


def badness(k, v):
    return v - 1 if k == 'stretch_p99' else 1 / max(v, 1e-9) if k in LOWER_IS_WORSE else v


FLOORS = {'kink_radius_p1': 0.004, 'kinks': 2, 'spikes': 2, 'root_stand_p99': 15.0, 'bare_roots': 1,
          'flow_dev_p99': 20.0, 'crossings': 2, 'depth_head': 0.005, 'depth_body': 0.005, 'buried_pct': 1.0,
          'drift_p99': 0.03, 'drift_max': 0.06, 'stretch_p99': 1.01, 'fold_radius_p1': 0.004,
          'jitter_p99': 0.001, 'depth_body_max': 0.01, 'depth_head_max': 0.01}


# what no approved hair excuses: a strand into the skin or the body past these (HH: 0.01 is about 2 mm) is through it,
# whatever the approved styles do; an approved style past one is listed with the outliers
FIXED = {'depth_head': 0.01, 'depth_body': 0.01, 'depth_head_max': 0.015, 'depth_body_max': 0.02}


def band(summaries, keys, outliers, where):
    out = {}
    for k in keys:
        if k in FIXED:
            out[k] = FIXED[k]
            for n, S in summaries.items():
                if S.get(k) is not None and S[k] > FIXED[k]:
                    outliers.append({'metric': k, 'where': where, 'style': n, 'value': S[k], 'past': 'the fixed limit'})
            continue
        vals = sorted(((badness(k, S[k]), n, S[k]) for n, S in summaries.items() if S.get(k) is not None), reverse=True)
        if not vals:
            continue
        if len(vals) >= 3 and vals[0][0] > 3 * max(vals[1][0], 1e-6) and k not in COUNTS:
            outliers.append({'metric': k, 'where': where, 'style': vals[0][1], 'value': vals[0][2]})
            vals = vals[1:]
        w = vals[0][0]
        v = w + 2 if k in COUNTS else 1 + w * MARGIN if k == 'stretch_p99' else 1 / (w * MARGIN) if k in LOWER_IS_WORSE \
            else w * MARGIN
        f = FLOORS.get(k)
        if f is not None:
            v = min(v, f) if k in LOWER_IS_WORSE else max(v, f)
        out[k] = round(float(v), 4) if k not in COUNTS else int(math.ceil(v))
    return out


def calibrate(reports, path):
    outliers = []
    lim = band({n: dict(r['drawn'], **{('settle_' + k if k in ('stretch_p99', 'fold_radius_p1') else k): v
                                      for k, v in r['settle'].items()}) for n, r in reports.items()},
               [k for k in DEFAULT_LIMITS if k not in ('stretch_p99', 'fold_radius_p1')], outliers, 'drawn')
    lim2 = band({n: r['settle'] for n, r in reports.items()}, ['drift_p99', 'drift_max', 'stretch_p99', 'fold_radius_p1'],
                outliers, 'rest')
    lim.update(lim2)
    poses = {}
    for pose in POSES[1:]:
        S = {n: r['poses'][pose] for n, r in reports.items() if pose in r.get('poses', {})}
        if S:
            poses[pose] = band(S, list(MOTION_LIMITS), outliers, pose)
    b = {'from': sorted(reports), 'version': VERSION, 'margin': MARGIN, 'limits': lim, 'poses': poses,
         'outliers': outliers, 'head': next(iter(reports.values()))['head']}
    json.dump(b, open(path, 'w'), indent=1)
    return b


# ---------------------------------------------------------------------------------------------------- heatmap

BG, INK, RING, CLAY, BODY = (246, 245, 242), (40, 40, 44), (205, 48, 40), (200, 198, 192), (226, 224, 219)
RAMP = [(0.0, (196, 194, 188)), (0.35, (158, 197, 244)), (0.7, (85, 152, 231)), (1.0, (37, 106, 191)),
        (1.5, (13, 54, 107))]
VIEWS = [('front', 0.0), ('3/4', 45.0), ('profile', 90.0), ('back', 180.0)]


def ramp(t):
    t = np.clip(np.nan_to_num(t), 0, RAMP[-1][0])
    xs = np.array([a for a, _ in RAMP])
    cs = np.array([c for _, c in RAMP], float)
    return np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], -1)


def _view(X, az):
    """Orthographic: the camera turns round the figure by az degrees (0 the front, from -y)."""
    a = math.radians(az)
    u = X[:, 0] * math.cos(a) + X[:, 1] * math.sin(a)
    depth = -X[:, 0] * math.sin(a) + X[:, 1] * math.cos(a)     # (larger is farther)
    return u, X[:, 2], depth


def panel(draw, ox, oy, size, W, lines, cols, widths, az, rings, box):
    """One view: the head and body in clay, the strands as lines coloured by score, far to near, the worst ringed."""
    x0, x1, z0, z1 = box
    s = size / max(x1 - x0, z1 - z0)

    def px(u, z):
        return ox + (u - x0) * s, oy + (z1 - z) * s
    # the body and head in clay (their surface points, splatted) and the strands, drawn together far to near
    items = []
    for P, col in ((W._body_pts, BODY), (W._head_pts, CLAY)):
        u, z, d = _view(P, az)
        items += [(d[k], 0, px(u[k], z[k]), None, col, 0) for k in range(0, len(P), 2)]
    for L, C, Wd in zip(lines, cols, widths):
        u, z, d = _view(L, az)
        for i in range(len(L) - 1):
            items.append((0.5 * (d[i] + d[i + 1]), 1, px(u[i], z[i]), px(u[i + 1], z[i + 1]),
                          tuple(int(c) for c in C[i]), Wd[i]))
    items.sort(key=lambda q: -q[0])
    for _, kind, a, b, c, w in items:
        if kind == 0:
            draw.rectangle([a[0] - 1.3, a[1] - 1.3, a[0] + 1.3, a[1] + 1.3], fill=c)
        else:
            draw.line([a, b], fill=c, width=max(1, int(round(w * s))))
    for num, p in rings:
        u, z, _ = _view(p[None], az)
        x, y = px(u[0], z[0])
        r = 9
        draw.ellipse([x - r, y - r, x + r, y + r], outline=RING, width=2)
        draw.text((x + r + 1, y - r - 2), str(num), fill=RING)


# ------------------------------------------------------------------------------------------------ the contours
# John's rule (2026-11-20): after any solver run and before any render, the contours of the result are inspected:
# profiles and sections of the draped hair mass. A contour that is perfectly regular (a ring as true as a circle, a
# silhouette as smooth as a drawn curve) or stepped (a ledge round the mass at one height) is a fault to fix first:
# real hair has a ragged edge.

def contours(X, W, rows=160, rings=(0.8, 0.5, 0.2, -0.1, -0.4, -0.7), band=0.025):
    """The outline of the hair mass (points X, head heights): per height row the extent to each side in the front
    view (x) and in the side view (y, front and back), and the sections at the given heights above the chin: each a
    ring of the points within a band of that height, as radius against angle round the head's axis.
    Returns {'rows': z, 'x': (lo, hi), 'y': (lo, hi), 'sections': {h: (angle, radius)}} and the numbers:
      steps         rows where the outline jumps sideways by over STEP_JUMP between neighbouring rows, more than
                    a smooth outline does (a ledge)
      regularity    for each section, how far the ring's radius wanders from the smooth ring fitted to it (rms, as a
                    share of the radius): hair's own edge is ragged (above 0.01), a computed ring that is
                    perfectly regular is near 0
    """
    hh = W.m['height']
    z0, z1 = float(X[:, 2].min()), W.m['crown']
    zs = np.linspace(z0, z1, rows)
    lo = np.full((2, rows), np.nan)
    hi = np.full((2, rows), np.nan)
    for i, z in enumerate(zs):
        m = np.abs(X[:, 2] - z) < 0.5 * (zs[1] - zs[0]) + 0.01
        if m.sum() > 5:
            lo[0, i], hi[0, i] = X[m, 0].min(), X[m, 0].max()
            lo[1, i], hi[1, i] = X[m, 1].min(), X[m, 1].max()
    c = np.array(W.m['centre'])
    sec, reg = {}, {}
    for h in rings:
        z = W.m['chin'] + h * hh
        m = np.abs(X[:, 2] - z) < band * hh
        if m.sum() < 60:
            continue
        d = X[m] - c
        a = np.arctan2(d[:, 0], d[:, 1])
        r = np.hypot(d[:, 0], d[:, 1])
        order = np.argsort(a)
        a, r = a[order], r[order]
        # the outer edge: the farthest point in each of 72 sectors
        k = np.clip(((a + np.pi) / (2 * np.pi) * 72).astype(int), 0, 71)
        edge = np.full(72, np.nan)
        for j in range(72):
            if (k == j).any():
                edge[j] = r[k == j].max()
        ok = ~np.isnan(edge)
        if ok.sum() < 36:
            continue
        ang = (np.arange(72) + 0.5) / 72 * 2 * np.pi - np.pi
        e = edge.copy()
        e[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], edge[ok], period=72)
        # smooth ring: the first 3 harmonics (a shifted ellipse), what a computed ring would be
        F = np.fft.rfft(e)
        G = np.zeros_like(F)
        G[:4] = F[:4]
        smooth = np.fft.irfft(G, 72)
        reg[h] = float(np.sqrt(np.mean((e - smooth) ** 2)) / max(e.mean(), 1e-9))
        sec[h] = (ang, e, smooth)
    # steps: the extent of the outline in the side view, back edge, from row to row
    steps = 0
    for arr in (lo[0], hi[0], hi[1]):
        a = arr[~np.isnan(arr)][4:-4]       # (not the first and last rows: where the mass starts and ends)
        d = np.abs(np.diff(a))
        if len(d) > 4:
            steps += int((d > STEP_JUMP * hh).sum())
    return {'rows': zs, 'lo': lo, 'hi': hi, 'sections': sec}, \
        {'outline_steps': steps, 'ring_irregularity_min': round(min(reg.values()), 4) if reg else None,
         'rings': {str(h): round(v, 4) for h, v in reg.items()}}


def contour_sheet(styles, path, size=300):
    """One sheet: for each state ('drawn', 'settled', and each motion's worst frame) the front and side outlines
    of the hair mass over the head, and the sections as radius against angle with the smooth ring fitted to each.
    styles: [(title, X, W)]."""
    from PIL import Image, ImageDraw
    rows = []
    for title, X, W in styles:
        C, N = contours(X, W)
        rows.append((title, X, W, C, N))
    img = Image.new('RGB', (3 * size + 20, len(rows) * (size + 34) + 10), BG)
    d = ImageDraw.Draw(img)
    for ri, (title, X, W, C, N) in enumerate(rows):
        oy = 10 + ri * (size + 34)
        d.text((10, oy), f"{title}: {N['outline_steps']} outline steps, smallest ring irregularity "
                         f"{N['ring_irregularity_min']} (hair's own edge is ragged; near 0 is a computed ring)", fill=INK)
        oy += 18
        hh = W.m['height']
        zs = C['rows']
        z1, z0 = W.m['crown'] + 0.05 * hh, float(zs.min()) - 0.05 * hh
        s = size / max(1.6 * hh, z1 - z0)
        for pi, (u, name) in enumerate(((0, 'front'), (1, 'side'))):
            ox = 10 + pi * size
            cx = ox + size / 2
            for P, col in ((W._head_pts, CLAY), (W._body_pts, BODY)):
                for q in P[::2]:
                    x = q[0] if u == 0 else q[1]
                    d.rectangle([cx + x * s - 1, oy + (z1 - q[2]) * s - 1, cx + x * s + 1, oy + (z1 - q[2]) * s + 1], fill=col)
            for arr, col in ((C['lo'][u], (37, 106, 191)), (C['hi'][u], (37, 106, 191))):
                pts = [(cx + arr[i] * s, oy + (z1 - zs[i]) * s) for i in range(len(zs)) if not np.isnan(arr[i])]
                if len(pts) > 1:
                    d.line(pts, fill=col, width=2)
            d.text((ox + 4, oy + size - 14), name + ' outline', fill=INK)
        ox = 10 + 2 * size
        d.text((ox + 4, oy), 'sections: edge radius vs angle (blue), smooth ring (grey)', fill=INK)
        for k, (h, (ang, e, sm)) in enumerate(list(C['sections'].items())[:6]):
            by = oy + 24 + k * (size - 30) / 6
            hgt = (size - 30) / 6 - 4
            top = max(e.max(), sm.max())
            bot = min(e.min(), sm.min())
            for arr, col in ((sm, (170, 170, 168)), (e, (37, 106, 191))):
                pts = [(ox + 4 + j / 72 * (size - 12), by + hgt - (arr[j] - bot) / max(top - bot, 1e-6) * hgt) for j in range(72)]
                d.line(pts, fill=col, width=2)
            d.text((ox + 4, by - 2), f'{h:+.1f} H', fill=INK)
    img.save(path)
    return rows[0][4], [r[4] for r in rows]


def heatmap(style, W, F, Fs, poses, rings, lim, path, size=230):
    from PIL import Image, ImageDraw
    hh = W.m['height']
    strands = style['strands']
    X, sid, first = flatten(strands)
    score = np.max(np.stack([
        F['kink'], F['flow'] / max(lim['flow_dev_p99'], 1e-6), F['head'] / max(lim['depth_head'], 1e-6),
        F['body'] / max(lim['depth_body'], 1e-6), F['root'] * 60.0 / max(lim['root_stand_p99'], 1e-6),
        np.minimum(F['cross'], 1.5)]), 0)
    rows = [('as drawn: worst of bends, flow, roots, depth and crossings (1.0 = the band)', None)]
    if Fs is not None:
        rows.append(('settled by the solver at rest: how far it moved (1.0 = the band)', 'settle'))
    for p in poses:
        rows.append((f'{p}: jitter and depth into the body, worst over the motion (1.0 = the band)', p))
    box = (-0.75 * hh, 0.75 * hh, max(X[:, 2].min(), W.m['chin'] - 1.4 * hh) - 0.08 * hh, W.m['crown'] + 0.12 * hh)
    head_h = 28
    img = Image.new('RGB', (len(VIEWS) * size + 20, len(rows) * (size + head_h) + 50), BG)
    d = ImageDraw.Draw(img)
    d.text((10, 8), f"{style['name']} ({style['style']}): {len(strands)} strands, body: {W.body_kind}; "
                    f"head {hh:.3f} tall, {W.m['width_ratio']:.2f} wide, {W.m['depth_ratio']:.2f} deep (HH)", fill=INK)
    for ri, (title, kind) in enumerate(rows):
        oy = 40 + ri * (size + head_h)
        d.text((10, oy), title, fill=INK)
        oy += 16
        if kind is None:
            lines = [s['p'] for s in strands]
            sc = [score[first[k]:first[k] + len(s['p'])] for k, s in enumerate(strands)]
            widths = [np.maximum(s['w'], 0.004) * 1.2 for s in strands]
            fl = rings.get('drawn', [])
            ptsrc = X
        else:
            runs = Fs['runs']
            src = Fs['X'] if kind == 'settle' else poses[kind][1]['frames'][poses[kind][1]['worst']]
            lines = [src[r] for r in runs]
            if kind == 'settle':
                v = Fs['drift'] / max(lim['drift_p99'], 1e-6)
            else:
                pl = poses[kind][2]
                v = np.maximum(poses[kind][1]['jit'] / max(pl['jitter_p99'], 1e-6),
                               poses[kind][1]['db'] / max(pl['depth_body_max'], 1e-6))
            sc = [v[r] for r in runs]
            widths = [np.full(len(r), 0.012 * hh) for r in runs]
            fl = rings.get(kind, [])
            ptsrc = src
        rg = [(n, ptsrc[i]) for n, i in fl]
        for vi, (vn, az) in enumerate(VIEWS):
            panel(d, 10 + vi * size, oy, size - 8, W, lines, [ramp(c) for c in sc], widths, az, rg, box)
            d.text((14 + vi * size, oy + size - 22), vn, fill=INK)
    img.save(path)


# ---------------------------------------------------------------------------------------------------- run

SHARED_HOME = '/mnt/project-files/render-queue'


def cache_dir():
    """The shared hair cache: RQ_HAIR_CACHE, else hair-cache in RQ_HOME, else in the queue's shared folder where it
    exists (so every thread and the renders read the same solves), else the user's own cache."""
    home = os.environ.get('RQ_HOME') or (SHARED_HOME if os.path.isdir(SHARED_HOME) else None)
    d = os.environ.get('RQ_HAIR_CACHE') or (os.path.join(home, 'hair-cache') if home
                                            else os.path.join(os.path.expanduser('~'), '.cache', 'emberwood-hair'))
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def walk_keys():
    try:
        return json.load(open(os.path.join(HERE, 'hair_poses.json')))['walk']
    except (OSError, ValueError, KeyError):
        return None


def analyse(style, world, poses=POSES, solve=True, out=None, heat=True, cache=None):
    """The whole test on one style. Returns its report (and writes hair.json and hair-heat.png into out)."""
    lim, plim, _ = limits()
    W = World(world, style, cache)
    hh = W.m['height']
    W._head_pts = world['head'][::4]
    zz = np.linspace(W.m['chin'] - 1.6 * hh, W.neck_z + 0.05, 60)
    th = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    Z, TH = np.meshgrid(zz, th, indexing='ij')
    ring = []
    for r in np.linspace(0.02, 1.3, 120):           # (the body's surface, found along rays from its axis)
        ring.append(np.stack([r * np.sin(TH), 0.04 - r * np.cos(TH), Z], -1).reshape(-1, 3))
    ring = np.stack(ring, 1)
    inside = W.body(ring.reshape(-1, 3)).reshape(ring.shape[:2]) < 0
    k = np.where(inside.any(1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], 1), -1)
    W._body_pts = ring[np.arange(len(ring)), np.maximum(k, 0)][k >= 0]
    S, F, faults = check_drawn(style['strands'], W)
    faults = [f + ('drawn',) for f in faults]
    rep = {'style': style['style'], 'name': style['name'], 'version': VERSION, 'body': W.body_kind,
           'head': {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in W.m.items()},
           'drawn': S, 'poses': {}}
    Fs, pose_out = None, {}
    if solve and style['strands']:
        Ss, Fs, fs, settled = check_settle(style['strands'], W, cache)
        rep['settle'] = Ss
        faults += [f + ('settle',) for f in fs]
        walk = walk_keys()
        for p in poses:
            if p == 'rest':
                continue
            Sp, Fp, fp = check_pose(p, style['strands'], W, settled, walk, cache)
            rep['poses'][p] = Sp
            pose_out[p] = (Sp, Fp, plim.get(p, MOTION_LIMITS))
            faults += [f + (p,) for f in fp]
    # verdict
    flat = dict(S)
    if 'settle' in rep:
        flat.update({k: rep['settle'][k] for k in ('drift_p99', 'drift_max', 'stretch_p99', 'fold_radius_p1')})
    flags, worst = judge(flat, lim)
    for p, Sp in rep['poses'].items():
        fl, w = judge(Sp, plim.get(p, MOTION_LIMITS))
        for f in fl:
            f['pose'] = p
        flags += fl
        if w == 'likely rejection' or (w == 'watch' and worst == 'ok'):
            worst = w
    rep['flags'], rep['verdict'] = flags, worst
    # the worst spots, numbered as on the heatmap
    spots = []
    pts_of = {'drawn': F['X']}
    if Fs is not None:
        pts_of['settle'] = Fs['X0']
        for p in pose_out:
            pts_of[p] = Fs['X0']
    faults.sort(key=lambda f: -f[2])
    seen = []
    for f in faults:
        if len(spots) >= 12:
            break
        P = pts_of[f[4]][f[1]]
        if any(np.linalg.norm(P - q) < PLACE * hh and g == f[0] for q, g in seen):
            continue
        seen.append((P, f[0]))
        spots.append({'n': len(spots) + 1, 'fault': f[0], 'where': place_name(P, W.m), 'when': f[4],
                      'note': f[3], 'at': [round(float(x), 3) for x in P], '_i': int(f[1])})
    rep['spots'] = [{k: v for k, v in sp.items() if not k.startswith('_')} for sp in spots]
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(rep, open(os.path.join(out, 'hair.json'), 'w'), indent=1)
        if heat:
            rings = {}
            for sp in spots:
                rings.setdefault(sp['when'], []).append((sp['n'], sp['_i']))
            heatmap(style, W, F, Fs, pose_out, rings, lim, os.path.join(out, 'hair-heat.png'))
            if Fs is not None:
                states = [('as drawn', F['X'], W), ('settled at rest', Fs['X'], W)]
                for pn, (Sp, Fp, _) in pose_out.items():
                    states.append((f'{pn}, worst frame', Fp['frames'][Fp['worst']], W))
                _, nums = contour_sheet(states, os.path.join(out, 'hair-contours.png'))
                rep['contours'] = {t[0]: n for t, n in zip(states, nums)}
                json.dump(rep, open(os.path.join(out, 'hair.json'), 'w'), indent=1)
    return rep


def line(rep):
    s = f"{rep['style']:<14} {rep['verdict']:<17} strands {rep['drawn']['strands']:>4}"
    for f in rep['flags'][:5]:
        s += f"  {f['metric']}{'@' + f['pose'] if 'pose' in f else ''}={f['value']} (>{f['limit']})"
    return s


# ---------------------------------------------------------------------------------------------------- self test

def _synthetic(kind='clean'):
    """A sphere head on the creator's axes with straight locks hanging off the back of it, and broken versions."""
    hh = 1.0
    c = np.array([0, 0.05, 0.55])
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 120, endpoint=False), np.linspace(0.02, np.pi - 0.02, 60))
    S = np.stack([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)], -1).reshape(-1, 3)
    head = c + 0.45 * S * np.array([0.8, 1.0, 1.0])
    head[:, 2] = np.maximum(head[:, 2], -0.1)
    nrm = (head - c) / np.array([0.8, 1.0, 1.0]) ** 2
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    face = np.array([[0, -0.4, 0.0], [0.05, -0.4, 0.0]])
    head = np.r_[head, face]
    nrm = np.r_[nrm, [[0, -1, 0], [0, -1, 0]]]
    cap = c + 0.47 * S * np.array([0.8, 1.0, 1.0])
    cap_a = (S[:, 2] > -0.2).astype(float)
    strands = []
    for k, a in enumerate(np.linspace(-1.2, 1.2, 25)):
        for j in range(3):
            aa = a + 0.02 * j
            p0 = c + 0.475 * np.array([0.8 * math.sin(aa) * 0.7, math.cos(aa) * 0.7, 0.7])
            # (over the skull and down the back, along the cap, then hanging)
            pts = []
            for t in np.linspace(0, 1, 40):
                ph = 0.8 + 1.6 * min(t, 0.5) / 0.5 * 0.6
                q = c + 0.48 * np.array([0.8 * math.sin(aa) * math.sin(ph), math.cos(aa) * math.sin(ph), math.cos(ph)])
                if t > 0.5:
                    q = q.copy()
                    q[2] -= (t - 0.5) * 1.2
                pts.append(q)
            P = np.array(pts)
            if kind == 'kink' and k == 12 and j == 1:
                P[20] += np.array([0.06, 0, 0])
            if kind == 'cross' and k == 12:
                P[25:] = P[25:] + np.linspace(0, 1, 15)[:, None] * np.array([0.5, 0, 0])
            if kind == 'float' and k in (11, 12, 13):
                P[20:] = P[20] + (P[20:] - P[20]) * np.array([1, 0, 0]) + np.linspace(0, 0.6, 20)[:, None] * np.array([0, 1, 0])
            strands.append({'p': P, 'w': np.full(len(P), 0.01), 't': np.full(len(P), 0.005), 'lock': k, 'layer': 0})
    style = {'style': 'synthetic-' + kind, 'name': 'synthetic ' + kind, 'strands': strands, 'cap': cap, 'cap_a': cap_a}
    M = {'z0': 0.075, 'dz': 0.025, 'na': 8, 'r': ([60] * 8 + [70] * 8 + [80] * 8) + [200] * 8 * 20 + [250] * 8 * 70}
    world = {'head': head, 'head_n': nrm, 'bodies': {'mannequin': M, 'styles': {}}}
    return style, world


def selftest(out):
    import tempfile
    cache = tempfile.mkdtemp()
    res = {}
    for kind in ('clean', 'kink', 'cross', 'float'):
        st, w = _synthetic(kind)
        res[kind] = analyse(st, w, poses=('rest', 'turn'), out=os.path.join(out, kind), cache=cache)
        print(line(res[kind]))
    ok = (res['kink']['drawn']['kinks'] + res['kink']['drawn']['spikes'] >
          res['clean']['drawn']['kinks'] + res['clean']['drawn']['spikes'] and
          res['cross']['drawn']['crossings'] > res['clean']['drawn']['crossings'] and
          res['float']['settle']['drift_max'] > 2 * res['clean']['settle']['drift_max'])
    print('selftest', 'ok' if ok else 'FAILED')
    return ok


def main():
    a = sys.argv[1:]

    def opt(k, d=None):
        return a[a.index(k) + 1] if k in a else d
    if '--selftest' in a:
        sys.exit(0 if selftest(opt('--selftest')) else 1)
    if '--drape' in a:
        # python3 rq_hair.py DIR --drape STYLE --into EXPORT.json.gz --out DRAPED.json.gz
        n = opt('--drape')
        print(write_drape(opt('--into'), os.path.join(a[0], n + '.json.gz'), opt('--out'),
                          load_world(os.path.join(a[0], 'head.json.gz')), cache_dir()))
        return
    if '--motion' in a:
        # python3 rq_hair.py DIR --motion STYLE:POSE --into EXPORT.json.gz --out MOTION.npz
        n, pose = opt('--motion').split(':')
        print(write_motion(opt('--into'), os.path.join(a[0], n + '.json.gz'), pose, opt('--out'),
                           load_world(os.path.join(a[0], 'head.json.gz')), cache_dir()))
        return
    if '--tune' in a:
        sh, L = tune_stiffness()
        print(f'elastogravity length {L:.3f} HH ({L * HEAD_CM:.1f} cm): SHAPE {sh} (now {SHAPE})')
        return
    if '--bake-walk' in a:
        bake_walk(opt('--bake-walk'), os.path.join(HERE, 'hair_poses.json'))
        return
    src = a[0]
    out = opt('--out', os.path.join(src, 'report'))
    world = load_world(os.path.join(src, 'head.json.gz'))
    names = opt('--styles') or opt('--calibrate')
    names = names.split(',') if names else sorted(f[:-8] for f in os.listdir(src) if f.endswith('.json.gz') and f != 'head.json.gz')
    poses = tuple(opt('--poses', ','.join(POSES)).split(','))
    cache = cache_dir()
    reps = {}
    for n in names:
        st = load_style(os.path.join(src, n + '.json.gz'))
        if not st['strands']:
            print(f'{n:<14} no locks (the cap only): nothing to test')
            continue
        reps[n] = analyse(st, world, poses=poses, solve='--no-solve' not in a, out=os.path.join(out, n), cache=cache)
        print(line(reps[n]), flush=True)
    if '--calibrate' in a:
        b = calibrate(reps, os.path.join(HERE, 'hair_baseline.json'))
        print(json.dumps(b['limits']), json.dumps(b['outliers']))
    json.dump({n: {'verdict': r['verdict'], 'flags': r['flags'], 'spots': r['spots']} for n, r in reps.items()},
              open(os.path.join(out, 'hair-summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
