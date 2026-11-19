"""Lamed shoulder plates (spaulders), raised over whatever the shoulder already wears.

A spaulder is a cop, a shallow dome over the point of the shoulder, and below it a fall of lames: narrow curved
plates round the outside of the upper arm, each lapped by the one above so the arm can lift, all riveted to leathers
inside. It hangs from the arming point on the doublet's shoulder.

As the armourer works, flat first. Every plate is a flat blank cut from sheet (blanks(): the cop a disc, a lame the
strip of a cone unrolled, an arc whose radius comes from where the lame sits on the shoulder's round) and is then
raised to the shoulder. The shoulder is measured for it as it is dressed: the surface the plates are raised to is the
outermost of the arm and everything worn on it (shirt sleeve, doublet, vest), seen from the shoulder's own centre,
smoothed (plate does not follow a fold of linen) and stood off by a gap.

Every size is a share of w, the neck's half-width, or an angle about the shoulder's centre."""
import math

import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

import fit

RATIOS = dict(
    gap=0.10,                # the room between the plates and what is under them, in w
    plate=0.055,             # the plate's thickness, in w
    cop=58.0,                # the cop reaches this far down from the top of the shoulder, in degrees
    curl=11.0,               # below that the cop's rim runs on this far, bent in toward the first lame, in degrees
    lame=15.5,               # the height of one lame showing, in degrees
    lames=4,
    lap=6.0,                 # how far a plate runs up under the one above, in degrees
    round_cop=100.0,         # how far round from straight out the cop reaches, front and back, in degrees
    round_lame=94.0,         # and the lames (the inside of the arm is left bare)
    inner=0.80,              # toward the neck the cop stops at this share of the way out to the shoulder point
    corner=0.35,             # the radius of a plate's corners, in w
    corner_cop=2.4,          # the cop's corners are rounder by this many times
    dome=0.8,                # how far a lame is raised along its middle, in plate thicknesses
    step=2.6,                # a lame's bottom edge stands this far proud of the next lame's top, in plate
                             # thicknesses: the groove between two lames
    flare_cop=0.34,          # the cop's lower edge is turned out by this much, in w: the top plate flares, a
                             # brim over the lames
    cop_sit=1.0,             # the cop stands this far off the shoulder's smoothed round, in plate thicknesses
    cop_ride=1.4,            # and this far proud of the first lame where it laps it
    curl_in=0.70,            # the rim comes back in by this share of the flare
    flare_lame=0.16,         # and each lame's lower edge, by this much
    flare_last=0.30,         # the lowest lame's the most of the lames: the fall ends in a kick
    box=3.2,                 # the outline's squareness seen from the front: 2 follows the shoulder's round,
                             # more makes a corner of it (a superellipse's exponent)
    box_out=0.10,            # and the whole set stands this much further out at the corner, in w
    rivet=0.075,             # a rivet head's radius, in w
)
D_TH, D_AL = 2.0, 3.0        # the measuring grid about the shoulder's centre, in degrees


def _dirs(th, al, side):
    """Unit vectors at the angles th (down from straight up) and al (round from straight out to the side)."""
    return np.stack([side * np.sin(th) * np.cos(al), np.sin(th) * np.sin(al), np.cos(th)], -1)


class Shoulder:
    """One dressed shoulder, measured about its centre: the radius of the outermost surface in every direction."""

    def __init__(self, P, aw, M, tree, side, ratios):
        w = M['w']
        self.side, self.M, self.r = side, M, ratios
        arm = P[(aw > 0.5) & (side * P[:, 0] > 0) & (P[:, 2] > M['z_pit_arm'] - 1.0 * w) & (P[:, 2] < M['z_pit_arm'])]
        half = 0.5 * float(np.ptp(arm[:, 0]))
        self.C = np.array([0.5 * float(arm[:, 0].min() + arm[:, 0].max()), 0.5 * float(arm[:, 1].min() + arm[:, 1].max()),
                           M['z_sh'] - half])
        self.th = np.radians(np.arange(0.0, 150.0 + D_TH, D_TH))
        self.al = np.radians(np.arange(-180.0, 180.0, D_AL))
        TH, AL = np.meshgrid(self.th, self.al, indexing='ij')
        D = _dirs(TH, AL, side)
        R = np.full(TH.shape, np.nan)
        c = Vector(self.C)
        for i in range(TH.shape[0]):
            for j in range(TH.shape[1]):
                d = Vector(D[i, j])
                hit = tree.ray_cast(c + d * 30.0, -d, 30.0)[0]
                if hit is not None:
                    R[i, j] = (hit - c).length
        R = np.where(np.isnan(R), np.nanmedian(R), R)
        # plate does not follow a fold: the surface smoothed, but never inside what it was measured from
        smooth = fit._blur(fit._blur(R, 2.5, 0, False), 4.0, 1, True)
        for _ in range(3):
            smooth = np.maximum(smooth, fit._blur(fit._blur(np.maximum(R, smooth), 2.0, 0, False), 3.0, 1, True))
        self.R = np.maximum(smooth, R) + ratios['gap'] * w
        # the cop is raised from a smoother shoulder still: one even dome, with none of the lumps of what is under it
        # (stood off by the most the dressed shoulder comes through it, within the cop's reach)
        # (measured over the arm's side of the shoulder only: toward the neck the rays find the collar, far off)
        side_r = float(self.R[int(round(90.0 / D_TH)), int(round(180.0 / D_AL))])
        # (clamped only toward the neck, past where the cop reaches round: within its reach, front and back, it has
        # to clear the breastplate's own chest and back, which stand well out from the shoulder's centre)
        neckward = (np.abs(self.al) > math.radians(ratios['round_cop'] + 12.0))[None, :]
        near = np.where(neckward, np.minimum(self.R, 1.35 * side_r), self.R)
        # (a smooth envelope over it: blurred, lifted to everything it was blurred from, and blurred again, until it
        # lies over the dressed shoulder everywhere and follows none of its lumps)
        even = near.copy()
        for _ in range(12):
            even = fit._blur(fit._blur(np.maximum(near, even), 5.0, 0, False), 7.0, 1, True)
        reach = (self.th <= math.radians(ratios['cop'] + ratios['curl']))[:, None] & (np.abs(self.al) <= math.radians(ratios['round_cop']))[None, :]
        # (and lifted only where something still comes through it, locally, not the whole dome by its worst point)
        rest = np.maximum(near - even, 0.0)
        for _ in range(4):
            rest = np.maximum(rest, fit._blur(fit._blur(rest, 2.5, 0, False), 3.5, 1, True))
        self.R_cop = even + rest
        self.R_side = float(self.R[int(round(90.0 / D_TH)), int(round(180.0 / D_AL))])

    def radius(self, th, al, cop=False):
        """The measured radius at angles th and al (between the grid's entries); cop: the cop's even dome."""
        R = self.R_cop if cop else self.R
        f = np.clip(th / math.radians(D_TH), 0.0, len(self.th) - 1.001)
        i = np.floor(f).astype(int)
        t = f - i
        g = ((al + math.pi) % (2 * math.pi)) / math.radians(D_AL)
        j = np.floor(g).astype(int) % len(self.al)
        s = g - np.floor(g)
        j1 = (j + 1) % len(self.al)
        return (R[i, j] * (1 - s) + R[i, j1] * s) * (1 - t) + (R[i + 1, j] * (1 - s) + R[i + 1, j1] * s) * t

    def angles(self, p):
        q = p - self.C[None, :]
        r = np.linalg.norm(q, axis=1)
        return np.arccos(np.clip(q[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0)), np.arctan2(q[:, 1], self.side * q[:, 0]), r


def fit_smin(a, b, k):
    return fit._smin(a, b, k)


def _lift(S, name, u, th):
    """How far a plate stands off the measured shoulder at u across it (0 its top edge, 1 its bottom), at th down
    from straight up. Tucked under the plate above at its top, proud at its bottom where it laps the next, raised
    along its middle, its lower edge turned out; and the whole made squarer than the shoulder's round."""
    r, w = S.r, S.M['w']
    t_pl = r['plate'] * w
    n = r['box']
    square = (np.abs(np.sin(th)) ** n + np.abs(np.cos(th)) ** n) ** (-1.0 / n) - 1.0      # (0 at the top and the side)
    box = (r['box_out'] * w + S.R_side * 1.0) * square * (th < 0.5 * math.pi)
    if name == 'cop':
        # The cop sits close on the shoulder: a plate's thickness off its smoothed round, with none of the lames'
        # squaring or flare. Only where it laps the first lame does it stand out, just enough to ride over it.
        fit = t_pl * r['cop_sit']
        a1 = math.radians(r['cop'] - 2.0 * r['lap'])
        b1 = math.radians(r['cop'] + r['lame'] + r['curl'])
        u1 = np.clip((th - a1) / (b1 - a1), 0.0, 1.0)
        over = (_lift(S, 'lame 1', u1, th) + t_pl * r['cop_ride']) * (th > a1)
        return -fit_smin(-fit, -over, 3.0 * t_pl)
    k = r['flare_last'] if name == 'lame %d' % r['lames'] else r['flare_lame']
    return box + t_pl * (r['step'] * u + r['dome'] * np.sin(math.pi * u)) + k * w * u ** 2


def plates(S):
    """The plates' places on the shoulder: (name, top, bottom, how far round), angles in radians, the cop first."""
    r = S.r
    curl = math.radians(r['curl'])
    out = [('cop', 0.0, math.radians(r['cop']) + curl, math.radians(r['round_cop']))]
    a = math.radians(r['cop'])
    for k in range(r['lames']):
        # (the first lame runs twice as far up under the cop: the cop's edge is turned out, and the shirt showed
        # in the gap under it)
        lap = math.radians(r['lap']) * (2.0 if k == 0 else 1.0)
        # (and it is taller by the cop's curled rim, which covers that much of it)
        b = a + math.radians(r['lame']) + (curl if k == 0 else 0.0)
        out.append(('lame %d' % (k + 1), a - lap, b, math.radians(r['round_lame'])))
        a = b
    return out


def blanks(S):
    """The flat blanks the plates are raised from, each an outline in units [(x, y)]. The cop: a disc as wide as
    the dome is over its top. A lame: the strip of the cone that touches the shoulder's round along the lame's
    middle, unrolled: an arc, of radius R tan(th) and turning through (the angle it goes round) cos(th)."""
    out = []
    for name, a, b, half in plates(S):
        mid = 0.5 * (a + b)
        R = float(S.radius(np.array([mid]), np.array([0.0]))[0])
        if name == 'cop':
            rad = R * b                                          # (the dome's own length from its top to its edge)
            out.append((name, [(rad * math.cos(t), rad * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 49)]))
            continue
        rho, turn, wid = R * math.tan(mid) if abs(math.cos(mid)) > 0.05 else 1e3, 2 * half * math.cos(mid), R * (b - a)
        if abs(turn) < 1e-3:                                     # (at the shoulder's widest the cone is a cylinder: a straight strip)
            ln = R * 2 * half
            out.append((name, [(-0.5 * ln, 0.0), (0.5 * ln, 0.0), (0.5 * ln, wid), (-0.5 * ln, wid)]))
            continue
        ts = np.linspace(-0.5 * turn, 0.5 * turn, 25)
        r0, r1 = abs(rho) - 0.5 * wid, abs(rho) + 0.5 * wid
        out.append((name, [(r0 * math.sin(t), r0 * math.cos(t) - abs(rho)) for t in ts] + [(r1 * math.sin(t), r1 * math.cos(t) - abs(rho)) for t in ts[::-1]]))
    return out


def draw_blanks(S, path, scale=60.0):
    """The blanks laid out side by side as a cutting sheet (an image), with their sizes in centimetres."""
    from PIL import Image, ImageDraw
    cm = 165.0 / 63.0
    bl = blanks(S)
    boxes = [(np.array(o).min(0), np.array(o).max(0)) for _, o in bl]
    W = int(sum(b[1][0] - b[0][0] for b in boxes) * scale + 40 * (len(bl) + 1))
    H = int(max(b[1][1] - b[0][1] for b in boxes) * scale + 110)
    im = Image.new('RGB', (W, H), (244, 240, 230))
    dr = ImageDraw.Draw(im)
    x = 40
    for (name, o), (lo, hi) in zip(bl, boxes):
        pts = [(x + (px - lo[0]) * scale, 40 + (hi[1] - py) * scale) for px, py in o]
        dr.polygon(pts, outline=(60, 40, 30), fill=(205, 170, 120))
        dr.text((x, H - 60), '%s\n%.1f x %.1f cm' % (name, (hi[0] - lo[0]) * cm, (hi[1] - lo[1]) * cm), fill=(40, 30, 20))
        x += int((hi[0] - lo[0]) * scale) + 40
    im.save(path)


def _rivet(c, n, r, sides=10, rings=3):
    """A rivet's domed head on the plate at c, facing n."""
    a = n.cross(Vector((0, 0, 1)) if abs(n.z) < 0.9 else Vector((1, 0, 0))).normalized()
    b = n.cross(a)
    V, F = [c + n * (0.6 * r)], []
    for k in range(1, rings + 1):
        ph = 0.5 * math.pi * k / rings
        for i in range(sides):
            t = 2 * math.pi * i / sides
            V.append(c + (a * math.cos(t) + b * math.sin(t)) * (r * math.sin(ph)) + n * (0.6 * r * math.cos(ph)))
    for i in range(sides):
        F.append([0, 1 + i, 1 + (i + 1) % sides])
        for k in range(rings - 1):
            p, q = 1 + k * sides, 1 + (k + 1) * sides
            F.append([p + i, q + i, q + (i + 1) % sides, p + (i + 1) % sides])
    return np.array([v[:] for v in V]), F


def raise_plates(S, log=print):
    """The plates raised to the measured shoulder: (plate pieces, rivet pieces), each [(points, faces)]."""
    w, r, M = S.M['w'], S.r, S.M
    t_pl = r['plate'] * w
    x_in = r['inner'] * M['x_sh']
    pieces, rivets = [], []
    for name, a, b, half in plates(S):
        th = np.arange(max(a - math.radians(3), 0.0), b + math.radians(3), math.radians(1.0))
        al = np.arange(-half - math.radians(6), half + math.radians(6), math.radians(1.5))
        TH, AL = np.meshgrid(th, al, indexing='ij')
        u = np.clip((TH - a) / (b - a), 0.0, 1.0)                 # (0 at the plate's top edge, 1 at its bottom)
        R = S.radius(TH.reshape(-1), AL.reshape(-1), cop=name == 'cop').reshape(TH.shape) + _lift(S, name, u, TH)
        V = (S.C[None, None, :] + R[:, :, None] * _dirs(TH, AL, S.side)).reshape(-1, 3)
        n0, n1 = TH.shape
        idx = np.arange(n0 * n1).reshape(n0, n1)
        faces = [list(q) for q in np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], 2).reshape(-1, 4)]

        def edge(p, a=a, b=b, half=half, name=name):
            th_, al_, rr = S.angles(p)
            k = r['corner'] * w * (r['corner_cop'] if name == 'cop' else 1.0)
            across = np.minimum(th_ - a, b - th_) * rr if name != 'cop' else (b - th_) * rr
            round_ = (half - np.abs(al_)) * rr * np.maximum(np.sin(th_), 0.15)
            f = fit._smin(across, round_, k)
            if name == 'cop':                                      # (and toward the neck it stops short of the collar)
                f = fit._smin(f, S.side * p[:, 0] - x_in, k)
            return f
        V, faces = fit.trim(V, faces, edge)
        used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        pieces.append((V[used], [[int(remap[i]) for i in f] for f in faces]))
        # a rivet at each end of a lame, through to the leathers inside
        if name != 'cop':
            mid = 0.5 * (a + b) + 0.15 * (b - a)
            for sgn in (-1.0, 1.0):
                al_r = sgn * (half - 0.16)
                uu = (mid - a) / (b - a)
                rad = float(S.radius(np.array([mid]), np.array([al_r]))[0]) + float(_lift(S, name, np.array(uu), np.array(mid))) + t_pl
                n = Vector(_dirs(np.array(mid), np.array(al_r), S.side))
                rivets.append(_rivet(Vector(S.C) + n * rad, n, r['rivet'] * w))
    return pieces, rivets


def build_spaulders(body, M, worn, rig=None, plate=None, name='Spaulder', ratios=RATIOS, log=print, sheet=None):
    """A spaulder on each shoulder of the body, over the garments `worn` (objects). sheet: a path to draw the flat
    blanks to (the left shoulder's)."""
    P, T, aw = fit.body_arrays(body)
    tree = fit.rest_tree([body, *worn])[0]
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert(Vector(p), i)
    kd.balance()
    out = []
    for side, tag in ((1.0, 'L'), (-1.0, 'R')):
        S = Shoulder(P, aw, M, tree, side, ratios)
        pieces, rivets = raise_plates(S, log)
        t = ratios['plate'] * M['w']
        out.append(fit._sheet('%s.%s' % (name, tag), pieces, M, plate, body, kd, rig, lift=0.0, thickness=t))
        fit._sheet('%s rivets.%s' % (name, tag), rivets, M, plate, body, kd, rig, lift=0.0, thickness=0.0)
        bl = blanks(S)
        log('spaulder %s: centre (%.2f, %.2f, %.2f), radius %.2f at the side; blanks: %s' % (
            tag, *S.C, float(S.radius(np.array([math.pi / 2]), np.array([0.0]))[0]),
            ', '.join('%s %.1f x %.1f' % (n, *np.ptp(np.array(o), axis=0)) for n, o in bl)))
        if sheet and side > 0:
            draw_blanks(S, sheet)
    return out
