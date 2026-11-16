"""The ear, sculpted as a surface, for any figure's head. Pure Python and numpy (no Blender), so the face check
(blender/facecheck) draws exactly what Blender builds.

How an ear is built (as an anatomy-minded sculptor blocks it in): a flap of cartilage shaped like a C opening
toward the face, about as tall as from the brow to the base of the nose and a little over half as wide, its
long axis tilted back about 15 degrees, set just behind the jaw's back edge. On its outer face:
- the helix: the rolled rim round the top and back, which starts inside the bowl (the crus of the helix) and
  fades into the lobe at the bottom;
- the scapha: the long groove just inside the rim;
- the antihelix: a Y-shaped ridge inside that, its stem running down the back of the bowl and splitting into
  two crura at the top, with a shallow hollow (the triangular fossa) between them;
- the concha: the deep bowl round the ear's opening;
- the tragus: the small flap in front of the opening, and the antitragus across from it, over the lobe;
- the lobe: soft and full, with no cartilage ridges.
The flap stands off the skull toward its back edge (about 20 degrees), joined to the head along its front.

ear_grids(head_point, H) gives, for each side, one grid of points in the head's own coordinates (model units,
face toward -y): rows run round the outline, columns out over the front face from the bowl to the rim and back
over the rear face to inside the head. ear_tones gives its colour, side_x_finder the head_point for a figure.
"""
import math

import numpy as np

# the ear's layout in its own plane: u back along the head (from the ear's opening), v up, in head heights
EAR_TOP, EAR_BOTTOM = 0.625, 0.36      # the top of the helix about level with the brow, the lobe with the nose's base
EAR_TILT = math.radians(15)            # the long axis leans back
EAR_FLARE = math.radians(20)           # how far the back edge stands off the skull
EAR_Y = 0.035                          # where the opening sits front to back (head heights; 0 is the head's middle)

# the outline, round from the top of the front edge (where the helix leaves the head), back over the top, down
# the back, round the lobe and up the front past the tragus (u, v in head heights from the opening)
OUTLINE = [(0.000, 0.105), (0.030, 0.135), (0.075, 0.150), (0.115, 0.138), (0.140, 0.105), (0.150, 0.055),
           (0.145, 0.000), (0.128, -0.055), (0.100, -0.100), (0.070, -0.135), (0.040, -0.160), (0.012, -0.170),
           (-0.008, -0.158), (-0.014, -0.130), (-0.008, -0.095), (-0.012, -0.040), (-0.020, 0.020), (-0.014, 0.070)]
CENTRE = (0.055, -0.005)


def _closed_spline(pts, n):
    """A smooth closed curve through pts (Catmull-Rom), n points evenly in parameter."""
    P = np.array(pts, dtype=np.float64)
    m = len(P)
    out = []
    for k in range(n):
        t = k / n * m
        i = int(t)
        f = t - i
        p0, p1, p2, p3 = P[(i - 1) % m], P[i % m], P[(i + 1) % m], P[(i + 2) % m]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3))
    return np.array(out)


def _sm(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _g(d, w):
    return np.exp(-(d / w) ** 2)


def _seg_dist(P, A, B):
    """Distance from points P (..., 2) to the polyline through A..B points (list)."""
    best = np.full(P.shape[:-1], 1e9)
    for a, b in zip(A, B):
        a, b = np.array(a), np.array(b)
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (a + t[..., None] * ab), axis=-1))
    return best


def _poly(pts):
    return pts[:-1], pts[1:]


# the antihelix's stem and its two crura, and the helix's crus (inside the bowl), in head heights
ANTIHELIX = [(0.040, -0.115), (0.075, -0.080), (0.098, -0.030), (0.100, 0.020), (0.090, 0.060)]
CRUS_UPPER = [(0.090, 0.060), (0.080, 0.095), (0.060, 0.112)]
CRUS_LOWER = [(0.090, 0.060), (0.055, 0.070), (0.022, 0.068)]
HELIX_CRUS = [(-0.005, 0.040), (0.025, 0.030), (0.050, 0.020)]


def ear_relief(u, v, edge, outline_t):
    """How far the ear's outer face stands out of its plane (head heights) at (u, v); edge is the distance in
    from the outline, outline_t where round the outline the point lies (0..1 from the front-top)."""
    P = np.stack([u, v], -1)
    front = _sm(0.80, 0.9, outline_t) + _sm(0.06, 0.0, outline_t)      # the front edge, joined to the head
    lobe = _sm(-0.10, -0.13, v)
    rimmed = (1 - front) * (1 - lobe)
    h = np.zeros_like(u)
    h += 0.016 * _g(edge - 0.010, 0.010) * rimmed                    # the helix's rolled rim
    h -= 0.008 * _g(edge - 0.026, 0.009) * rimmed                    # the scapha, just inside it
    a = _seg_dist(P, *_poly(ANTIHELIX))
    c1 = _seg_dist(P, *_poly(CRUS_UPPER))
    c2 = _seg_dist(P, *_poly(CRUS_LOWER))
    h += 0.008 * _g(np.minimum(a, np.minimum(c1 * 1.15, c2 * 1.3)), 0.016) * (1 - lobe)
    fossa = _g(np.hypot(u - 0.070, v - 0.088), 0.018)
    h -= 0.005 * fossa                                                # the triangular fossa between the crura
    concha = _g(np.hypot((u - 0.040) / 1.1, (v + 0.02) / 1.25), 0.042)
    h -= 0.024 * concha * (1 - lobe)                                  # the bowl
    h -= 0.012 * _g(np.hypot(u - 0.006, v + 0.005), 0.012)            # the opening, deep in the bowl
    h += 0.008 * _g(_seg_dist(P, *_poly(HELIX_CRUS)), 0.008)          # the helix's root crossing the bowl
    h += 0.012 * _g(np.hypot((u + 0.006) / 0.8, v + 0.012), 0.016)    # the tragus
    h += 0.009 * _g(np.hypot(u - 0.052, v + 0.078), 0.015)            # the antitragus
    h += 0.006 * lobe * _g(edge - 0.03, 0.04)                         # the lobe's soft fullness
    return h


def ear_grids(head_point, H, NT=160, NF=36, NB=16):
    """The two ears as closed grids in the head's coordinates (model units). head_point(z, side, y) gives the
    head's surface x at height z (head heights) and depth y on that side, so each ear is set on its own skull.
    Returns, for each side, a dict: 'side', 'P' (NT, NF + NB, 3) points (rows round the outline, which wrap;
    columns from the bowl's centre out to the rim, then back under the flap into the head), 'h' the relief of the
    outer face (0 on the rear), 'rear' (the rear face's share, 0 on the outer face, 1 where it meets the skull),
    'edge' (head heights in from the rim) and 'lobe' (1 on the lobe)."""
    ring = _closed_spline(OUTLINE, NT)
    C = np.array(CENTRE)
    ts = np.arange(NT) / NT
    sf = np.linspace(0, 1, NF) ** 0.8          # out from the bowl's centre to the outline (the outer face)
    sb = np.linspace(1, 0, NB + 1)[1:]          # and back from the outline toward the skull (the rear face)
    ct, st = math.cos(EAR_TILT), math.sin(EAR_TILT)
    zc = 0.5 * (EAR_TOP + EAR_BOTTOM) - 0.5 * (max(v for _, v in OUTLINE) + min(v for _, v in OUTLINE))
    out = []
    for side in (-1, 1):
        P, Hr, R, E, Lb = [], [], [], [], []
        for i in range(NT):
            O = ring[i]
            L = np.linalg.norm(O - C)
            uv = C + sf[:, None] * (O - C)
            u, v = uv[:, 0], uv[:, 1]
            edge = (1 - sf) * L
            h = ear_relief(u, v, edge, np.full(NF, ts[i]))
            # the flap stands off the skull toward its back edge, the rim rounding over onto the rear face
            # (and the whole flap a little proud of it, back from where the front edge joins the cheek, so the bowl
            # stays a bowl and only the opening goes into the head)
            w_front = np.maximum(0.0, u + 0.01) * math.tan(EAR_FLARE) + h + 0.004 + 0.016 * _sm(-0.015, 0.04, u)
            ub, vb = C[0] + sb * (O[0] - C[0]), C[1] + sb * (O[1] - C[1])
            thick = 0.010 + 0.010 * _sm(-0.10, -0.15, vb)     # (the lobe is thicker)
            # (the back of the flap comes in to the skull in a smooth fold over the concha's bulge, no gap)
            w_back = ((np.maximum(0.0, ub + 0.01) * math.tan(EAR_FLARE) + 0.016 - thick) * sb ** 0.6
                      - 0.012 * (1 - sb) ** 2)
            uu, vv, ww = np.concatenate([u, ub]), np.concatenate([v, vb]), np.concatenate([w_front, w_back])
            # the ear's plane, tilted back about the opening, set on the head's side
            y = EAR_Y + uu * ct + vv * st
            z = zc + (-uu * st + vv * ct)
            xs = np.array([head_point(zz, side, yy) for zz, yy in zip(z, y)])
            P.append(np.stack([side * (xs + ww), y, z], -1) * H)
            Hr.append(np.concatenate([h, np.zeros(NB)]))
            R.append(np.concatenate([np.zeros(NF), 1 - sb]))
            E.append(np.concatenate([edge, (1 - sb) * L]))
            Lb.append(_sm(-0.10, -0.13, vv))
        out.append(dict(side=side, P=np.array(P), h=np.array(Hr), rear=np.array(R), edge=np.array(E),
                        lobe=np.array(Lb)))
    return out


def ear_tones(e):
    """The ear's colour, as the head's skin_tones paints it: (flush, shade). Ears are rosier than the face, most
    of all round the rim and in the lobe, where the light comes through thin skin; the bowl, the groove inside
    the rim and the fold where the flap meets the skull are shut in and dark."""
    rim = np.exp(-(e['edge'] / 0.02) ** 2) * (1 - e['rear'])
    flush = np.clip(0.45 + 0.35 * rim + 0.3 * e['lobe'], 0, 1)
    shade = np.clip(-e['h'] / 0.03, 0, 1) + _sm(0.5, 1.0, e['rear'])
    return flush, np.clip(shade, 0, 1)


def side_x_finder(head_section, jaw_relief=None):
    """head_point for ear_grids from a figure's head_section: the head's half-width at height z and depth y,
    found by bisection round the section (front to back is monotonic in y on the side)."""
    def head_point(z, side, y):
        lo, hi = 0.15, math.pi - 0.15
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            xm, ym = head_section(z, math.cos(mid), math.sin(mid))
            if ym < y:
                lo = mid
            else:
                hi = mid
        xm, ym = head_section(z, math.cos(lo), math.sin(lo))
        d = jaw_relief(ym, z) if jaw_relief else 0.0
        return xm + d
    return head_point
