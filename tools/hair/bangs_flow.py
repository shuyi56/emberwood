"""DRAFT: bangs as a blend of two flow fields on the forehead, read off the ponytail field's cached side flow
(John, 2026-11-25: "How do we model proper bangs based of the field strength of the sides, hair closer to the sides
follow the back flow more strongly than those at the center").

    python3 tools/hair/bangs_flow.py HEAD.json.gz OUT_DIR --field-cache CACHE_DIR --photo photo.measure.json
                                     [--preset curtain|straight|side-swept] [--fit]

The model. Two unit direction fields on the skin:
  B(x)  the back-flow: the direction the side hair already takes toward the tie, read off the ponytail field's own
        lines (the blue funnel lines, the re-solved ear lines and the temple lock) near x, weighted by distance.
  D(x)  the fall of the bangs: gravity laid on the skin (forward over the crown's front slope, down the forehead),
        turned by D_DEG away from the part toward the cheekbone.
The bang hair's direction is t = normalize(w(u) B + (1 - w(u)) D), u the distance along the skin from the part,
w = smoothstep(w0, w1, |u|): 0 at the part, 1 from w1 on. At w = 1 a bang hair takes the side flow's own direction,
so the bangs join the sides with the same tangent (C1) by construction. Every bang root's hair is a streamline of t,
traced on the skin at the lift the fringe's root has, to its length L(u), set by where its end should hang (the
fringe edge at the part, the cheekbone at the side lock).

Constraints gauged on the built lines (bangs.json):
  (1) divergence: the fan opens away from the part (dm/du > 0); the hair through each cut across the fringe, per mm of
      the cut's width, stays >= F_MIN over the first BODY share of the length. The tips taper below it on purpose.
  (2) monotone progress, no stagnation: |w B + (1 - w) D| >= STAG_MIN everywhere a hair passes, and each hair moves
      away from its root at every step.
  (3) bounded curl: curvature <= K_MAX (read over CURL_STEP_MM) along the sweep from D to B.
  (4) lengths taper to points, and the outer ends meet the side lock's S line (distance of each end to the lock).
  (5) hairline_gap.py on the whole head (ponytail lines + these bangs): no bare cell; rank swaps between neighbouring
      bang roots: 0.
The ear is not judged (the model has none).

Parameters (PRESETS): part offset, sweep (split or one side), D angle, w0, w1, end heights. Curtain, straight and
side-swept bangs are three parameter sets over the same two fields.

Sources.
  [CC]  curlcentric.com, How to cut curtain bangs: base of the triangle "from the outer corner of each of your eyes";
        part "down the middle so that it falls equally on either side of your nose"; length "cheekbone height or just
        below"; sides "taper".
  [BS]  bangstyle.com, How to pick a fringe: width "from the end of brow to end of brow"; centre triangle "a depth of
        one inch"; part "based off the middle"; "elevate and point cut to blend together".
  [BTC] behindthechair.com, The cheekbone guide for cutting curtain bangs: "a clean triangular section", the sides
        overdirected "toward the opposite cheekbone", corners softened with light layering.
  [MBG] mindbodygreen.com, How to cut your own bangs: a middle part "best for curtain bangs"; the next front sections
        trimmed a little shorter than the longest layer, diagonally.
  [F94] Farkas 1994, Anthropometry of the head and face: endocanthion and exocanthion spans of adult women.
  [GWB12] Goldstein, Warren, Ball 2012 (via ponytail_flow): a hair's bending length about 5 cm.
  Diffuser analogy for the fan (duct design): a conical diffuser stays attached up to a half angle of about 7 deg
        (rule of thumb quoted in eng-tips thread 133014, "7.5 degree half-angle ... close to lossless"). Used only to
        name the fan's opening rate; the bound the gauge applies is the hair count per mm (F_MIN), not an angle.
Every constant is flagged SOURCED, MEASURED (refmeasure sheet, head units), FITTED (to the measured strand angles),
DERIVED or DRAFT."""
import sys, os, json, math, glob, pickle, hashlib, time, inspect
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF
import ponytail_flow as PF

MM = SF.MM
unit = PF.unit
sm = SF.sm

STEP_MM = 1.5                   # the streamline's step on the skin
STAG_MIN = 0.35                 # DRAFT: the smallest |w B + (1 - w) D| allowed (1 = the two agree, 0 = they cancel)
F_MIN = PF.F_MIN                # hairs per mm: the no-gap floor (hairline_gap.py HL_COVER, DRAFT there)
BODY = 0.7                      # DRAFT: the fan must hold F_MIN over this share of each hair's length; the rest tapers
CURL_STEP_MM = PF.CURL_STEP_MM
BRUSH_R_MM = 19.0               # DRAFT: the tightest sweep a bang is dried to, a 1.5 inch round brush's radius
K_MAX = 1.0 / BRUSH_R_MM        # per mm: the curl bound (|dm/ds| <= K)
ROOT_EXIT_MM = 8.0              # the curl bound is read from here on: the first mm are the hair leaving its follicle
                                # (scalp_field's exit angle, 15 deg [P23][W25]) and lifting off the skin, not the sweep
ROOT_EXIT_STEPS = int(ROOT_EXIT_MM / PF.CURL_STEP_MM)
LIFT_MM = (1.0, 20.0)           # the root's and the tips' lift, and the most the fringe stands off the forehead:
                                # MEASURED, the photo's fringe front against the forehead (fitted through the nose
                                # tip): 26, 19.5, 13 and 2.6 mm at 230, 260, 282 and 306 px, the tips on the brow
BANG_EDGE_MM = 66.0             # the bangs section's outer edge, u at ponytail_flow's BANG_TH on this head
B_K = 64                        # the side-flow nodes read per point
B_SIGMA_MM = 10.0               # DRAFT: their Gaussian weight's width (wider than the lines' spacing, so the field
                                # reads the flow, not the nearest single line)
CAP_MM = 110.0                  # DERIVED: the longest a bang runs past the hairline: the curtain's side ends at the
                                # cheekbone [CC], 85 mm from the trichion on this head, times 1.3 for its curve (DRAFT)
SHELL_RAD = 0.12                # rad seen from the skull's centre (about 12 mm at the brow): the hollows bridged
SHELL_GRID = 0.008              # rad: the shell's grid (about 0.8 mm)
SHELL_BLUR = 0.035              # rad: the shell's blur (about 3.5 mm), so the bed carries no noise of the export
RANK_BAND_MM = 3.0              # the depth of a row of roots for the rank gauge (a wider band pairs roots one behind
                                # the other, whose hairs lie at different heights of the sheet)
PART_FADE_MM = 8.0              # DRAFT: D's turn grows in over this distance either side of the part
SMOOTH = 6                      # passes easing each traced line (the skin read off the export is noisy at 1 mm)
TAPER = 0.18                    # DRAFT: the deepest-rooted hairs end up to this share shorter (the tips taper to points,
                                # as a point cut leaves them [CC][BS])

# The presets. w0, w1 and D_DEG of the curtain are FITTED to the photo's strand angles (see fit()); the end heights are
# MEASURED off the photo against the eye (refmeasure: fringe_edge, side_end, eye), the part is SOURCED [CC][BS][MBG].
PRESETS = {
    'curtain': dict(part_mm=0.0, sweep=0, d_deg=27.7, w0_mm=37.5, w1_mm=145.0, edge_above_eye=0.087,
                    side_below_eye=0.097, side_u_mm=70.0, side_end_behind_nose=0.263),
    # the coordinator's reading of John's model: w = 1 at the bangs section's outer edge (66 mm, just inside the
    # temple), so the outer bang roots leave with the side flow's own tangent. Not what the photo measures (fit.json).
    'curtain-c1': dict(part_mm=0.0, sweep=0, d_deg=27.7, w0_mm=17.0, w1_mm=66.0, edge_above_eye=0.087,
                       side_below_eye=0.097, side_u_mm=70.0, side_end_behind_nose=0.263),
    'straight': dict(part_mm=0.0, sweep=0, d_deg=0.0, w0_mm=52.0, w1_mm=80.0, edge_above_eye=0.087,
                     side_below_eye=-0.087, side_u_mm=70.0),
    'side-swept': dict(part_mm=-25.0, sweep=1, d_deg=55.0, w0_mm=40.0, w1_mm=105.0, edge_above_eye=0.06,
                       side_below_eye=0.05, side_u_mm=80.0, corner=0.0),
}


# --------------------------------------------------------------------------------------------- the ponytail field
def reuse_field_cache(cache_dir):
    """Read the ponytail field's stages from its cache folder as they are (scalp, blue lines, free fall). Their keys
    hash file mtimes, which a fresh checkout changes, so a stage with exactly one cached file is taken by name: the
    field is the merged one (PR #88) and is not solved again here. Nothing is written to that folder."""
    def cached(name, key, fn):
        f = glob.glob(os.path.join(cache_dir, name + '-*.pkl'))
        if len(f) == 1:
            return pickle.load(open(f[0], 'rb'))
        print('field cache: %s not found, solving' % name, file=sys.stderr)
        return fn()
    PF.cached = cached
    PF.CACHE = None


def side_flow(B):
    """The side flow's nodes and tangents: every non-bang line on the scalp (the funnel's and the re-solved lines up
    to the handover ring, and the temple lock), front half of the head only."""
    from scipy.spatial import cKDTree
    pts, tan = [], []
    ring = B['r_tie'] / MM + PF.RING_MM
    for i in np.where(B['basin'] == 0)[0]:
        X = B['X'][i][B['rho'][i] >= ring]
        if len(X) < 3:
            continue
        X = X[::2]
        T = unit(np.gradient(X, axis=0))
        k = X[:, 1] < 0.25
        pts.append(X[k]); tan.append(T[k])
    for k_, (nm, c) in enumerate(PF.LOOSE.items()):
        for i in np.where(B['basin'] == 2 + k_)[0]:
            X = PF.loose_line(B, i, c['len'])
            T = unit(np.gradient(X, axis=0))
            pts.append(X); tan.append(T)
    P = np.concatenate(pts); T = np.concatenate(tan)
    return dict(P=P, T=T, tree=cKDTree(P))


def B_at(SFL, X, N):
    """The back-flow's unit direction at X (n, 3), in the skin's tangent plane (normals N)."""
    d, j = SFL['tree'].query(X, k=B_K)
    w = np.exp(-0.5 * (d / MM / B_SIGMA_MM) ** 2) + 1e-9
    T = (SFL['T'][j] * w[..., None]).sum(1)
    T = T - (T * N).sum(1, keepdims=True) * N
    return unit(T)


def u_of(X, prm):
    """Signed distance along the skin from the part, mm (+ toward +x, the character's left)."""
    th = SF.theta(X)
    r = np.hypot(X[:, 0] - SF.AXIS[0], X[:, 1] - SF.AXIS[1])
    return th * r / MM - prm['part_mm']


def w_of(u, prm):
    return sm(prm['w0_mm'], prm['w1_mm'], np.abs(u))


def D_at(X, N, u, prm):
    """The bangs' fall, forward and down from the part: down and forward (0, -1, -1) laid on the skin, which is
    straight down the forehead and straight forward over the crown's top, where gravity alone lies flat on the skin
    and has no direction; then turned d_deg away from the part (sweep 0: each side away from the part; +-1: every
    hair toward that side)."""
    f = unit(np.array([0, -1.0, -1.0]))[None]
    g = unit(f - (f * N).sum(1, keepdims=True) * N)
    sgn = np.sign(u) if prm['sweep'] == 0 else np.full(len(u), float(prm['sweep']))
    sgn[sgn == 0] = 1.0
    s = sgn[:, None] * np.array([1.0, 0, 0])[None]
    s = s - (s * N).sum(1, keepdims=True) * N
    s = s - (s * g).sum(1, keepdims=True) * g
    s = unit(s)
    a = math.radians(prm['d_deg'])
    # the turn fades in over the first PART_FADE_MM from the part, so the two sides do not split at a corner
    a = a * sm(0.0, PART_FADE_MM, np.abs(u)) if prm['sweep'] == 0 else np.full(len(u), a)
    D = unit(np.cos(a)[:, None] * g + np.sin(a)[:, None] * s)
    # the corners: from w0 out to the section's edge the fall turns toward the side end, the point in front of the ear
    # where the photo's outer pieces end (MEASURED, side_end), so the corners sweep over the temple to it, as the
    # photo's outer fringe is swept to the front of the ear [README field: bang_at_ear]. [BTC]: the corners softened
    # into the face-framing layers.
    E = side_end_point(prm)
    if E is not None:
        Es = E[None] * np.c_[np.where(X[:, 0] >= 0, 1.0, -1.0), np.ones(len(X)), np.ones(len(X))]
        v = Es - X
        v = unit(v - (v * N).sum(1, keepdims=True) * N)
        c = sm(prm['w0_mm'], prm.get('edge_mm', BANG_EDGE_MM), np.abs(u)) * prm.get('corner', 1.0)
        D = unit((1 - c)[:, None] * D + c[:, None] * v)
    return D


_E = {}


def side_end_point(prm):
    """The side end on the skin (+x side), from the preset's measured place: behind the nose tip and below the eye."""
    if 'side_end_behind_nose' not in prm or _SURF.get('s') is None:
        return None
    key = (prm['side_end_behind_nose'], prm['side_below_eye'])
    if key not in _E:
        surf, L = _SURF['s'], _SURF['L']
        y = L['pronasale'][1] + prm['side_end_behind_nose']
        z = L['pupil'][2] - prm['side_below_eye']
        xs = np.linspace(0.0, 0.62, 250)
        ins = surf.depth(np.c_[xs, np.full_like(xs, y), np.full_like(xs, z)]) > 0
        _E[key] = np.array([xs[len(xs) - 1 - np.argmax(ins[::-1])], y, z])
    return _E[key]


_SURF = {}


def t_at(B, SFL, X, prm):
    """(t, |w B + (1 - w) D|, w, u) at points X on (or just off) the skin."""
    surf = B.get('shell') or B['surf']
    U = unit(X - SF.CENTRE)
    N = surf.normals(U)
    u = u_of(X, prm)
    w = w_of(u, prm)
    Bv = B_at(SFL, X, N)
    Dv = D_at(X, N, u, prm)
    v = w[:, None] * Bv + (1 - w)[:, None] * Dv
    return unit(v), np.linalg.norm(v, axis=1), w, u


class Shell(SF.Surface):
    """The skin with its hollows filled and its noise eased: the hair's bed. Hair hanging from the brow does not follow
    the skin into the eye socket or the temple's dip; it bridges them, as a sheet laid over the head does. The skin's
    radius is read once on a grid of directions (longitude, latitude, SHELL_GRID apart), closed (a grey-scale
    closing, a dilation then an erosion, over a disc SHELL_RAD across: hollows narrower than the disc fill, convex
    parts keep their shape) and blurred by SHELL_BLUR, then read back by interpolation."""
    def __init__(self, surf):
        from scipy import ndimage
        from scipy.interpolate import RegularGridInterpolator
        self.u, self.r, self.tree = surf.u, surf.r, surf.tree
        lon = np.arange(-math.pi, math.pi + 1e-9, SHELL_GRID)
        lat = np.arange(-0.7, 1.5708 + 1e-9, SHELL_GRID)
        LO, LA = np.meshgrid(lon, lat, indexing='ij')
        U = np.c_[(np.cos(LA) * np.sin(LO)).ravel(), (-np.cos(LA) * np.cos(LO)).ravel(), np.sin(LA).ravel()]
        R = SF.Surface.radius(self, U).reshape(LO.shape)
        k = int(round(SHELL_RAD / SHELL_GRID))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disc = (xx ** 2 + yy ** 2) <= k * k
        R = ndimage.grey_closing(R, footprint=disc, mode='wrap')
        R = ndimage.gaussian_filter(R, SHELL_BLUR / SHELL_GRID, mode='wrap')
        self.f = RegularGridInterpolator((lon, lat), R, bounds_error=False, fill_value=None)

    def radius(self, U):
        lon = np.arctan2(U[:, 0], -U[:, 1])
        lat = np.arcsin(np.clip(U[:, 2], -1, 1))
        return self.f(np.c_[lon, lat])


# --------------------------------------------------------------------------------------------- the end heights
def end_z(B, u, prm):
    """Where a hair rooted at u should end (HH above the chin): the fringe edge at the part, the side pieces at the
    cheekbone, measured against the eye so the creator's face, not hers, sets the height."""
    eye = B['F']['landmarks']['pupil'][2]
    zc = eye + prm['edge_above_eye']
    zs = eye - prm['side_below_eye']
    return zc + (zs - zc) * sm(0.0, prm['side_u_mm'], np.abs(u))


# --------------------------------------------------------------------------------------------- the streamlines
def trace(B, SFL, roots, prm, max_mm=170.0):
    """Every root's streamline of t, all roots stepped together, on the shell. Returns a list of (k, 3) polylines and
    the per-hair stagnation minimum."""
    surf = Shell(B['surf'])
    P = B['P'][roots]
    n = len(P)
    u0 = u_of(P, prm)
    zend = end_z(B, u0, prm)
    deep = np.clip(B['inside_mm'][roots] / PF.BANG_DEPTH_MM, 0, 1)
    rnd = (np.sin(P[:, 0] * 9973.0 + P[:, 1] * 7919.0 + P[:, 2] * 6151.0) * 43758.5453) % 1.0
    S = [P.copy()]
    alive = np.ones(n, bool)
    length = np.zeros(n)
    stag = np.ones(n)
    Xs = P.copy()
    steps = int(max_mm / STEP_MM)
    stop_len = np.full(n, max_mm)
    cap = np.minimum(B['inside_mm'][roots] + CAP_MM, max_mm)
    for k in range(steps):
        t, mag, w, u = t_at(B, SFL, Xs, prm)
        stag = np.where(alive, np.minimum(stag, mag), stag)
        Xn = Xs + t * (STEP_MM * MM)
        Un = unit(Xn - SF.CENTRE)
        Xn = surf.point(Un)                                   # back onto the skin (the lift is laid on after)
        length = length + alive * STEP_MM
        done = alive & ((Xn[:, 2] <= zend) | (length >= cap))
        stop_len[done] = length[done]
        alive = alive & ~done
        Xs = np.where(alive[:, None] | done[:, None], Xn, Xs)
        S.append(Xs.copy())
        if not alive.any():
            break
    S = np.stack(S, 1)                                        # (n, K, 3) on the skin
    out = []
    for i in range(n):
        L = stop_len[i] * (1 - TAPER * deep[i] * rnd[i])      # the deeper roots end shorter: the tips taper
        kk = max(int(round(L / STEP_MM)), 2)
        p = S[i, :kk + 1].copy()
        for _ in range(SMOOTH):
            p[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
        s = np.linspace(0, 1, len(p))
        U = unit(p - SF.CENTRE)
        p = surf.point(U)
        Nn = surf.normals(U)
        # the lift. Over the scalp: the dried root's lift (groom.py, 6 mm), grown in over the first 15 mm. Below the
        # hairline: the fringe stands off the forehead most at its middle and its tips come back onto the brow (the
        # photo's C curve: MEASURED, LIFT_MM). The corners, over the temple, stand off half as much (DRAFT).
        lmax = LIFT_MM[1] * (1 - 0.5 * sm(prm['w0_mm'], BANG_EDGE_MM, abs(u0[i])))
        smm = s * L
        z_hl = float(B['hl'](np.array([B['th'][roots[i]]]))[0])
        v = np.clip((z_hl - p[:, 2]) / max(z_hl - zend[i], 1e-3), 0, 1)
        base = 1.0 + 5.0 * sm(0.0, 15.0, smm) * (1 - v)
        lift = (base + (lmax - base) * np.sin(np.pi * v ** 0.7) ** 2) * MM
        out.append(p + Nn * lift[:, None])
    return out, stag, stop_len


# --------------------------------------------------------------------------------------------- the photo
def photo_fit(sheet, B):
    """Model (y, z) -> photo px: one scale and a shift (no turn, the photo's plumb is trusted), fitted on the nose tip
    (both axes), the menton and the eye (heights). The scale is the unit's: 616 px for the head top to the menton,
    and the photo's head top includes PF.PH['head_top_hair'] of hair over the crown."""
    lm = sheet['landmarks']
    L = B['F']['landmarks']
    unit_px = abs(lm['menton'][1] - lm['head_top'][1])
    sc = unit_px / (1 + PF.PH['head_top_hair'])
    tx = lm['nose_tip'][0] - sc * L['pronasale'][1]
    # heights: nose tip and menton (z 0), equally weighted
    ty = np.mean([lm['nose_tip'][1] + sc * L['pronasale'][2], lm['menton'][1] + sc * 0.0])
    return sc, tx, ty


def photo_samples(sheet):
    out = []
    k = 1
    while 's%da' % k in sheet['landmarks']:
        a, b = np.array(sheet['landmarks']['s%da' % k]), np.array(sheet['landmarks']['s%db' % k])
        if b[1] < a[1]:
            a, b = b, a
        out.append(dict(name='strand%d' % k, mid=0.5 * (a + b), a=a, b=b,
                        deg=math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))))
        k += 1
    return out


def sample_points(B, samples, fit):
    """Each photo strand's middle as a point on the model's skin, on the side the camera sees (+x)."""
    sc, tx, ty = fit
    surf = B['surf']
    pts = []
    for s in samples:
        y, z = (s['mid'][0] - tx) / sc, (ty - s['mid'][1]) / sc
        xs = np.linspace(0.0, 0.62, 125)
        Q = np.c_[xs, np.full_like(xs, y), np.full_like(xs, z)]
        ins = surf.depth(Q) > 0
        x = xs[len(xs) - 1 - np.argmax(ins[::-1])] if ins.any() else 0.0
        # (in front of the skin, the fringe's front: the midline point at that height)
        pts.append(np.array([x, y, z]) if ins.any() else SF.on_skin_at(surf, 0.0, z))
    return np.array(pts)


def side_deg(t):
    """A direction's lean from plumb as the side camera sees it (+ toward the back), as refmeasure reads a strand."""
    return np.degrees(np.arctan2(t[:, 1], -t[:, 2]))


def fit(B, SFL, samples, Xs, prm, sheet):
    """w0 and w1 fitted to the photo's own strands, in w, not in angle. The photo gives three things: D's lean (the
    strand at the part, strand1), B's lean (the strands behind the ear's front edge, which are side flow alone), and
    every strand between. Each strand's share of B is read off its lean as the blend of two unit directions:
        w = sin(phi - phi_D) / (sin(phi - phi_D) + sin(phi_B - phi)),
    and its u is the distance from the part of the skin point under it. The fit does not use the model's own B, so a
    gap between the model's side flow and the photo's (gauged in photo_gaps) does not bend the bangs' profile.
    Returns the fit and the table."""
    ph = np.array([s['deg'] for s in samples])
    ear_x = sheet['landmarks']['ear_top'][0]
    isB = np.array([s['mid'][0] > ear_x for s in samples])
    phD, phB = ph[0], float(ph[isB].mean())
    r = np.radians
    w_ph = np.sin(r(ph - phD)) / np.maximum(np.sin(r(ph - phD)) + np.sin(r(phB - ph)), 1e-9)
    w_ph = np.clip(w_ph, 0, 1)
    u = np.abs(u_of(Xs, prm))
    use = ~isB
    best = None
    for w0 in np.arange(0, 100, 2.5):
        for w1 in np.arange(w0 + 10, 260, 2.5):
            e = sm(w0, w1, u[use]) - w_ph[use]
            err = float(np.sqrt(np.mean(e ** 2)))
            if best is None or err < best[0]:
                best = (err, w0, w1)
    table = [dict(name=s['name'], photo_deg=round(s['deg'], 1), u_mm=round(float(uu), 1), w_photo=round(float(wp), 3),
                  w_fit=round(float(sm(best[1], best[2], uu)), 3), side_flow_only=bool(b))
             for s, uu, wp, b in zip(samples, u, w_ph, isB)]
    return dict(rms_w=round(best[0], 3), w0_mm=best[1], w1_mm=best[2], phi_D_deg=round(float(phD), 1),
                phi_B_deg=round(phB, 1), table=table)


EX_EX_MM = 88.0                 # SOURCED [F94]: the outer eye corners' span, adult women (Farkas 1994, about 87 to 89 mm)
CHEEK_BELOW_EYE_MM = 15.0       # DRAFT: the cheekbone's height below the pupils


def derive_d_deg(B):
    """D's turn away from the part, DERIVED from the cutting rules: the section's base runs to the outer corners of
    the eyes [CC] and its sides are directed toward the cheekbone [BTC][CC]. So D is the line, seen from the front,
    from the trichion at the part to the cheekbone under the outer eye corner, measured from plumb."""
    L = B['F']['landmarks']
    tr = L['trichion']
    x = 0.5 * EX_EX_MM * MM
    z = L['pupil'][2] - CHEEK_BELOW_EYE_MM * MM
    return float(np.degrees(np.arctan2(x - tr[0], tr[2] - z))), np.array([x, np.nan, z])


# --------------------------------------------------------------------------------------------- the gauges
def gauges(B, SFL, roots, lines, stag, prm, lock_lines):
    g = {}
    surf = B['surf']
    hairs = B['hairs'][roots]
    P = B['P'][roots]
    u0 = u_of(P, prm)
    # (2) stagnation and monotone progress
    back = 0
    for p in lines:
        d = np.linalg.norm(p - p[0], axis=1)
        back += int((np.diff(d) < -0.05 * STEP_MM * MM).any())
    g['stagnation'] = dict(min_magnitude=round(float(stag.min()), 3), bound=STAG_MIN, ok=bool(stag.min() >= STAG_MIN),
                           hairs_turning_back=back, of=len(lines))
    # (3) curl: the sweep's turn within the skin (geodesic curvature, the |dm/ds| of the model), read over CURL_STEP_MM,
    # against K_MAX; the bend over the head's own shape is reported beside it, not bounded
    kg, kt = [], []
    for p in lines:
        q = resample_mm(p, CURL_STEP_MM)
        if len(q) < 4:
            kg.append(0.0); kt.append(0.0); continue
        T1, T2 = unit(q[1:-1] - q[:-2]), unit(q[2:] - q[1:-1])
        dT = (T2 - T1) / (CURL_STEP_MM * MM)
        Nn = (B.get('shell') or surf).normals(unit(q[1:-1] - SF.CENTRE))      # (the hair's bed: the skin's normal is noisy)
        dTg = dT - (dT * Nn).sum(1, keepdims=True) * Nn
        ks = np.linalg.norm(dTg, axis=1)[ROOT_EXIT_STEPS:]                  # (past the root's exit: see ROOT_EXIT_MM)
        kg.append(float(ks.max()) * MM if len(ks) else 0.0); kt.append(float(np.linalg.norm(dT, axis=1).max()) * MM)
    kg, kt = np.array(kg), np.array(kt)
    g['curl'] = dict(sweep_tightest_radius_mm=round(float(1 / max(kg.max(), 1e-9)), 1), bound_radius_mm=BRUSH_R_MM,
                     sweep_over_bound_pct=round(100 * float((kg > K_MAX).mean()), 2), ok=bool((kg <= K_MAX).all()),
                     bend_over_head_tightest_radius_mm=round(float(1 / max(kt.max(), 1e-9)), 1))
    # (1) divergence: hairs per mm across each cut (share f of each hair's length), by 4 mm bins of the cut
    fan = []
    for f in (0.25, 0.5, BODY, 0.9):
        ends = np.array([p[min(int(f * (len(p) - 1)), len(p) - 1)] for p in lines])
        uu = u_of(ends, prm)
        o = np.argsort(uu)
        bins = np.arange(np.floor(uu.min() / 4) * 4, uu.max() + 4, 4.0)
        h, _ = np.histogram(uu, bins, weights=hairs)
        inside = (bins[:-1] > uu.min() + 4) & (bins[1:] < uu.max() - 4)
        cover = h[inside] / 4.0
        mids = 0.5 * (bins[:-1] + bins[1:])[inside]
        fan.append(dict(at=f, width_mm=round(float(uu.max() - uu.min()), 1),
                        thinnest_per_mm=round(float(cover.min()), 1) if len(cover) else None,
                        bins_under_floor=int((cover < F_MIN).sum()), bins=int(inside.sum()),
                        under_floor_at_u_mm=[round(float(m), 0) for m in mids[cover < F_MIN]]))
    # dm/du: how fast the fan opens, per depth band at 40 mm from the root, from the part outward (> 0: it opens)
    deep_ = B['inside_mm'][roots]
    dms = []
    for lo in range(0, 60, 10):
        k = np.where((deep_ >= lo) & (deep_ < lo + 10) & (np.abs(u0) > 3))[0]
        if len(k) < 4:
            continue
        pts = np.array([at_mm(lines[j], 40.0) for j in k]); ok_ = ~np.isnan(pts[:, 0])
        k, pts = k[ok_], pts[ok_]
        for sg in (-1, 1):
            m = np.sign(u0[k]) == sg
            if m.sum() < 4:
                continue
            o = np.argsort(np.abs(u0[k][m]))
            a_, b_ = np.abs(u0[k][m])[o], np.abs(u_of(pts[m], prm))[o]
            c = np.polyfit(a_, b_, 1)[0]
            dms.append(c)
    dm = np.array(dms)
    g['divergence'] = dict(cuts=fan, dm_du_by_band=[round(float(v), 2) for v in dm],
                           opens=bool((dm > 0).all()),
                           ok=bool(all(c['bins_under_floor'] == 0 for c in fan if c['at'] <= BODY)))
    # (5b) rank swaps: within each row of roots (RANK_BAND_MM deep behind the hairline), neighbours by root u keep
    # their side of each other at 20, 40 and 60 mm from the root, against their side at 6 mm. The side is read along
    # the sheet's own across-direction (the hair's tangent crossed with the skin's normal), so a fringe wrapping past
    # the temple onto the cheek is not read as a swap. A swap: the side reverses by more than 1 mm.
    deep = B['inside_mm'][roots]
    swaps = pairs = 0
    swap_at = []
    for lo in np.arange(0, 60, RANK_BAND_MM):
        k = np.where((deep >= lo) & (deep < lo + RANK_BAND_MM))[0]
        if len(k) < 3:
            continue
        k = k[np.argsort(u0[k])]
        ref = None
        for smm in (6.0, 20.0, 40.0, 60.0):
            pa = np.array([at_mm(lines[j], smm) for j in k])
            pn = np.array([at_mm(lines[j], smm + 2.0) for j in k])
            T = unit(pn - pa)
            Nn = surf.normals(unit(np.nan_to_num(pa) - SF.CENTRE + 1e-9))
            lat = unit(np.cross(Nn, T))
            s1 = ((pa[1:] - pa[:-1]) * lat[:-1]).sum(1) / MM     # neighbour's side, mm (+: toward + u)
            if ref is None:
                ref = s1                                           # as they leave the roots
                continue
            ok_ = ~np.isnan(s1) & ~np.isnan(ref)
            bad = np.where(ok_ & (np.sign(s1) != np.sign(ref)) & (np.abs(s1 - ref) > 1.0))[0]
            swaps += len(bad); pairs += int(ok_.sum())
            swap_at += [round(float(u0[k[b_]]), 1) for b_ in bad]
    g['rank_swaps'] = dict(swaps=swaps, pairs=pairs, ok=swaps == 0,
                           at_root_u_mm=sorted(set(swap_at))[:20])
    # C1 join: at the section's outer roots, the angle between the bang hair's first step and the side flow there
    outer = np.abs(B['th'][roots]) > PF.BANG_TH - 0.08
    if outer.any():
        X0 = np.array([lines[j][min(3, len(lines[j]) - 1)] for j in np.where(outer)[0]])
        Nn = surf.normals(unit(X0 - SF.CENTRE))
        Bv = B_at(SFL, X0, Nn)
        T0 = unit(np.array([lines[j][min(4, len(lines[j]) - 1)] - lines[j][min(2, len(lines[j]) - 2)]
                            for j in np.where(outer)[0]]))
        ang = np.degrees(np.arccos(np.clip((Bv * T0).sum(1), -1, 1)))
        g['join_first_step'] = dict(roots=int(outer.sum()), w_mean=round(float(w_of(u0[outer], prm).mean()), 3),
                         angle_to_side_flow_deg_median=round(float(np.median(ang)), 1),
                         angle_to_side_flow_deg_max=round(float(ang.max()), 1))
    # (4) lengths and the ends against the side lock
    Ls = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() / MM for p in lines])
    ub = np.arange(0, np.abs(u0).max() + 10, 10.0)
    prof = []
    for a, b in zip(ub[:-1], ub[1:]):
        k = (np.abs(u0) >= a) & (np.abs(u0) < b)
        if k.any():
            prof.append(dict(u_mm='%d-%d' % (a, b), roots=int(k.sum()), length_mm_median=round(float(np.median(Ls[k])), 1),
                             length_mm_min=round(float(Ls[k].min()), 1), w_root=round(float(w_of(0.5 * (a + b), prm)), 2),
                             end_z=round(float(np.median(ends[k, 2])), 3)))
    g['length_profile'] = prof
    if lock_lines:
        from scipy.spatial import cKDTree
        LP = np.concatenate(lock_lines)
        tr = cKDTree(LP)
        side = w_of(u0, prm) > 0.9
        if side.any():
            d = tr.query(ends[side])[0] / MM
            g['ends_to_side_lock_mm'] = dict(hairs=int(side.sum()), median=round(float(np.median(d)), 1),
                                             max=round(float(d.max()), 1))
    return g


def at_mm(p, smm):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] / MM
    if s[-1] < smm:
        return np.full(3, np.nan)
    return np.array([np.interp(smm, s, p[:, k]) for k in range(3)])


def resample_mm(p, step):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] / MM
    t = np.arange(0, s[-1], step)
    if len(t) < 3:
        return p
    return np.c_[[np.interp(t, s, p[:, k]) for k in range(3)]].T


# --------------------------------------------------------------------------------------------- main
def lines_memo(path, key):
    """A content cache for the bang lines: keyed on the roots, the parameters and the side flow's nodes, so a change
    to the bangs re-solves only the bangs and a rerun solves nothing."""
    p = os.path.join(path, 'bangs-lines-%s.pkl' % hashlib.sha1(repr(key).encode()).hexdigest()[:16])
    return p


def build_all(head, field_cache, out, prm, preset):
    reuse_field_cache(field_cache)
    PF.HEAD = head
    t0 = time.time()
    B = PF.build(head)
    roots = np.where(B['basin'] == 1)[0]
    SFL = side_flow(B)
    B['shell'] = Shell(B['surf'])
    _SURF.update(s=B['surf'], L=B['F']['landmarks'])
    print('time: field and side flow %.1f s' % (time.time() - t0), file=sys.stderr)
    mycache = os.path.join(os.path.dirname(os.path.abspath(out)), 'cache-bangs')
    os.makedirs(mycache, exist_ok=True)
    key = (preset, sorted(prm.items()), PF.Memo.h(B['P'][roots]), PF.Memo.h(SFL['P'][::50], SFL['T'][::50]),
           STEP_MM, LIFT_MM, BANG_EDGE_MM, TAPER, B_K, B_SIGMA_MM, CAP_MM, SMOOTH, PART_FADE_MM, SHELL_RAD, SHELL_GRID, SHELL_BLUR,
           hashlib.sha1(''.join(inspect.getsource(f) for f in (trace, t_at, B_at, D_at, u_of, w_of, end_z, side_flow, Shell, side_end_point))
                        .encode()).hexdigest())
    p = lines_memo(mycache, key)
    if os.path.exists(p):
        lines, stag, stop = pickle.load(open(p, 'rb'))
        print('bangs: lines reused from %s' % os.path.basename(p), file=sys.stderr)
    else:
        lines, stag, stop = trace(B, SFL, roots, prm)
        pickle.dump((lines, stag, stop), open(p, 'wb'), protocol=4)
        print('bangs: %d lines solved (%.1f s)' % (len(lines), time.time() - t0), file=sys.stderr)
    return B, SFL, roots, lines, stag


def main():
    a = sys.argv[1:]
    head, out = a[0], a[1]
    os.makedirs(out, exist_ok=True)
    field_cache = a[a.index('--field-cache') + 1]
    preset = a[a.index('--preset') + 1] if '--preset' in a else 'curtain'
    prm = dict(PRESETS[preset])
    photo = a[a.index('--photo') + 1] if '--photo' in a else None
    sheet = json.load(open(photo)) if photo else None
    import bangs_draw as BD
    t0 = time.time()
    B, SFL, roots, lines, stag = build_all(head, field_cache, out, prm, preset)
    G = dict(preset=preset, parameters=prm)
    # the join at the section's side edge, made continuous (bangs_join.py). Gauged before and after.
    import bangs_join as BJ
    G['join_before'] = BJ.join_gauge(B, roots, lines)
    side_new = {}
    if prm.get('join_band_mm', BJ.JOIN_BAND_MM) > 0:
        jb = prm.get('join_band_mm', BJ.JOIN_BAND_MM)
        jkey = (jb, PF.Memo.h(*[np.asarray(p) for p in lines[::7]]), PF.Memo.h(B['X'][::97]),
                hashlib.sha1(inspect.getsource(BJ).encode()).hexdigest())
        jp = os.path.join(os.path.dirname(os.path.abspath(out)), 'cache-bangs',
                          'bangs-join-%s.pkl' % hashlib.sha1(repr(jkey).encode()).hexdigest()[:16])
        if os.path.exists(jp):
            lines, side_new, G['join_info'] = pickle.load(open(jp, 'rb'))
        else:
            lines, side_new, G['join_info'] = BJ.join(B, SFL, roots, lines, prm, jb)
            pickle.dump((lines, side_new, G['join_info']), open(jp, 'wb'), protocol=4)
        G['join_after'] = BJ.join_gauge(B, roots, lines, side_new)
        G['join_control'] = {'%+g mm' % c: BJ.join_gauge(B, roots, lines, side_new, at_d=c)['g1_deg_median_all']
                             for c in (-9.0, 9.0)}     # (the same fits inside each family: the gauge's floor there)
        apply_side(B, side_new)
    if sheet is not None:
        fitp = photo_fit(sheet, B)
        smp = photo_samples(sheet)
        Xs = sample_points(B, smp, fitp)
        fj = fit(B, SFL, smp, Xs, prm, sheet)
        dd, zyg = derive_d_deg(B)
        fj['d_deg_derived'] = round(dd, 1)
        json.dump(fj, open(os.path.join(out, 'fit.json'), 'w'), indent=1, default=float)
        G['fit'] = {k: v for k, v in fj.items() if k != 'table'}
        BD.draw_profile(fj, prm, out)
        G['photo_gaps'] = BD.draw_photo(B, SFL, roots, lines, prm, sheet,
                                        os.path.join(os.path.dirname(photo), sheet['image']), out)
    lock = []
    for k_, (nm, c) in enumerate(PF.LOOSE.items()):
        lock += [PF.loose_line(B, i, c['len']) for i in np.where(B['basin'] == 2 + k_)[0]]
    G.update(gauges(B, SFL, roots, lines, stag, prm, lock))
    G['hairline'] = BD.hairline(B, roots, lines, out, os.path.join(os.path.dirname(os.path.abspath(out)), 'cache-bangs'))
    print('time: gauges %.1f s' % (time.time() - t0), file=sys.stderr)
    BD.draw_all(B, SFL, roots, lines, prm, G, out, head, sheet, preset)
    print('time: drawings %.1f s' % (time.time() - t0), file=sys.stderr)
    json.dump(G, open(os.path.join(out, 'bangs.json'), 'w'), indent=1, default=float)
    tie = [i for i in side_new if B['basin'][i] == 0]
    lk = [i for i in side_new if B['basin'][i] >= 2]
    np.savez_compressed(os.path.join(out, 'bangs-lines.npz'), roots=roots,
                        lines=np.array([PF_pad(p) for p in lines]),
                        tie_roots=np.array(tie, int), tie_lines=np.array([side_new[i] for i in tie]).reshape(len(tie), -1, 3),
                        lock_roots=np.array(lk, int), lock_lines=np.array([PF_pad(side_new[i], 51) for i in lk]).reshape(len(lk), -1, 3))
    print(json.dumps(G, indent=1, default=float))


_LOCK_OVERRIDE = {}
_loose_line = PF.loose_line


def _joined_loose_line(B, i, length=None, k=50):
    if int(i) in _LOCK_OVERRIDE:
        return PF_pad(_LOCK_OVERRIDE[int(i)], k + 1)
    return _loose_line(B, i, length, k)


def apply_side(B, side_new):
    """Put the joined side lines in place of the field's own: the tie lines in B['X'] (same nodes, so rho still
    holds), the temple lock's through PF.loose_line. The ponytail's files and caches are not touched."""
    for i, p in side_new.items():
        if B['basin'][i] == 0:
            B['X'][i] = p
        else:
            _LOCK_OVERRIDE[int(i)] = p
    PF.loose_line = _joined_loose_line


def PF_pad(p, n=120):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.c_[[np.interp(t, s, p[:, k]) for k in range(3)]].T


if __name__ == '__main__':
    main()
