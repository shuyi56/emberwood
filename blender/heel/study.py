"""The heel, derived from the foot: a study of how high a heel stands, how it is pitched, how wide its seat and its
tread are, where its breast falls and how long and stiff its shank is, so that the foot is carried as well as it can
be; and the gauge that measures it (support and tension). numpy only: it runs inside Blender (heel.py) and outside it
on the foot's points (tools/heel/sheet.py draws the sheet).

Every number below is taken from a named source, or labelled EXPERIMENTAL where none was found (docs/heel-study.md
lists the sources and what each number is derived from).

  Golding, "The Manufacture of Boots and Shoes" (c. 1902), ch. IV, sole shapes:
      the seat point (the middle of the heel's tread) 1/6 of the foot's length from the back of the heel;
      the inside joint (1st metatarsal head) 5/7 of the foot's length from the back of the heel;
      the seat's width 3/4 of the tread's width at the joint;
      the heel is placed on a perpendicular from the seat point (its tread's middle stands under the seat point).
  One & Awl, "Last measurements" (2024): the heel's length about 1/4 of the last's length.
  AmeFird, Footwear glossary: heel height measured on a vertical at the heel's breast; 1 in low, 2 in medium, 3 in high.
  Dooley, "A Manual of Shoemaking" (1912): the average foot carries a 1 to 1 1/4 in heel without strain.
  Chen, Li & Wang, PeerJ 8:e10239 (2020): dynamic postural control worsens from a 3.9 cm heel (0.8, 3.9, 7.0, 10.1
      tested); directional control 82 % -> 78 % -> 69 % (inexperienced) at 3.9 -> 7.0 -> 10.1 cm.
  Cavanagh, Rodgers & Iiboshi, Foot & Ankle 7(5) (1987), 107 feet, barefoot standing: heel 60 %, midfoot 8 %,
      forefoot 28 % of the load (toes the rest, "minimally involved"); peak pressure heel 139 kPa, forefoot 53 kPa.
  Broch, Wyller & Steen, JAPMA 94(5) (2004): forefoot to heel about 2:3 standing; with posture unchanged the forefoot's
      load rises and the heel's falls as the heel is raised.
  Wibowo et al., J Phys Ther Sci 29 (2017): at 2-4 cm heels the midfoot takes 10-11 % of body weight against 1 % flat,
      and the central heel's load falls from 14 % to about 1 %.
  Raychoudhury, Hu & Ren, Front Bioeng Biotechnol 2:73 (2014): the metatarsal heads' line (1st to 5th) at about 73 deg
      to the foot's cross axis in plan (so it slants back to the outside).
  ISO 18896 / EN 12959 (shank longitudinal stiffness): the shank clamped 32 mm at its heel end, loaded 2-8 N at its
      toe end, flexural rigidity S = W L^3 / (3 a); no limits given (SATRA TM58 likewise).
  Perry & Burnfield, "Gait Analysis" (2010): the vertical ground force peaks at about 110 % of body weight in walking.

Units: model units (the figure 63 units for 165 cm, so 1 unit = 2.619 cm) for geometry, newtons and kPa for loads.
Forward is -y, up is +z, her right foot at -x."""
import math

import numpy as np

CM = 165.0 / 63.0               # cm per model unit
MM = 10.0 * CM                  # mm per model unit

# ---- the sourced proportions (see the top)
SEAT_POINT = 1.0 / 6.0          # Golding: the seat point, a share of the foot's length from the back of the heel
JOINT = 5.0 / 7.0               # Golding: the inside joint (1st metatarsal head)
SEAT_OF_TREAD = 0.75            # Golding: the seat's width, a share of the sole's width at the joint (his 'tread')
HEEL_OF_LAST = 0.25             # One & Awl: the heel's length, a share of the last's length
LAST_ALLOW = 2.5 * 0.847        # Golding: the last 2.5 sizes (1/3 in each) longer than the foot, in cm
MTP_AXIS_DEG = 90.0 - 73.0      # Raychoudhury 2014: the metatarsal heads' line slants back this far from square
CAVANAGH = dict(heel=0.60, midfoot=0.08, forefoot=0.28, toes=0.04, heel_peak=139.0, fore_peak=53.0)
GAIT_PEAK = 1.10                # Perry & Burnfield: walking's vertical force peak, x body weight
STEEL_E = 200e3                 # N/mm^2, steel's Young's modulus (any engineering handbook)
LIFT_MM = 5.0                   # the atelier note: a heel lift about 5 mm of sole leather, one spare for compression

# ---- EXPERIMENTAL (no source found; each is a placeholder the gauge reports against, not a rule)
EXP = dict(
    mass=60.0,            # kg, the figure's body mass (165 cm, BMI 22); shares do not depend on it, kPa do
    friction=0.40,        # the foot (in hose) on a leather insole: how much forward push friction alone holds
    shank_w=12.0,         # mm, a steel shank's width
    shank_t=1.2,          # mm, and its thickness
    shank_yield=1100.0,   # N/mm^2, a tempered spring steel's yield
    waist_sag=1.0,        # mm, the most the waist may give under the walking peak before the arch is unsupported
    tip_min=20.0,         # deg, the least sideways tilt the loaded heel should stand before it tips onto its edge
    leather_uts=20.0,     # N/mm^2, upper leather's tensile strength (for the vamp's tension margin)
    leather_t=2.0,        # mm, the upper's thickness at the vamp
    sole_E=150.0,         # N/mm^2, sole leather's bending modulus (the waist's sole and insole bend with the shank)
    waist_w_of_seat=0.75, # Golding: the waist 3/4 of the seat (sourced; kept here beside the other waist numbers)
)
REGIONS = ((0.0, 0.30, 'heel'), (0.30, 0.60, 'midfoot'), (0.60, 0.80, 'forefoot'), (0.80, 1.01, 'toes'))


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ measuring the foot

def plantar(P, N, foot, step=0.15):
    """The sole of the foot as a height field: cells step across (model units), each the lowest downward-facing skin
    point in it. P, N the skin's points and normals, foot a mask of one foot's points. Returns [n, 3] points."""
    sel = foot & (N[:, 2] < -0.7) & (P[:, 2] < P[foot, 2].min() + 0.8)       # (inside the feather line)
    Q = P[sel]
    ij = np.floor(Q[:, :2] / step).astype(int)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.lexsort((Q[:, 2], key))
    key, Q = key[order], Q[order]
    first = np.concatenate([[True], key[1:] != key[:-1]])
    return Q[first].copy(), step


def measure(P, N, foot, ankle):
    """The foot's measures off its skin: length, the heel's back, the toe, widths along it, the 1st and 5th metatarsal
    heads (the medial and lateral extremes of the ball), the heel's width at the seat point, and the ankle joint's
    place (ankle: the Foot bone's head). Returns a dict (model units)."""
    F = P[foot]
    hy, ty = float(F[:, 1].max()), float(F[:, 1].min())
    L = hy - ty
    side = -1.0 if F[:, 0].mean() < 0 else 1.0          # (her right foot at -x: its inside is toward +x)
    f = dict(heel_y=hy, toe_y=ty, L=L, side=side, floor=float(F[:, 2].min()))
    prof = []
    for s in np.linspace(0.02, 0.98, 49):
        q = F[np.abs(F[:, 1] - (hy - s * L)) < 0.06]
        prof.append((s, q[:, 0].min(), q[:, 0].max()) if len(q) > 2 else (s, np.nan, np.nan))
    prof = np.array(prof)
    f['profile'] = prof
    inner = prof[:, 2] if side < 0 else -prof[:, 1]     # (how far each section reaches to the inside)
    outer = -prof[:, 1] if side < 0 else prof[:, 2]
    band = (prof[:, 0] > 0.55) & (prof[:, 0] < 0.90)
    i1 = np.nanargmax(np.where(band, inner, -np.inf))
    i5 = np.nanargmax(np.where((prof[:, 0] > 0.50) & (prof[:, 0] < 0.85), outer, -np.inf))
    f['mt1'] = np.array([prof[i1, 2] if side < 0 else prof[i1, 1], hy - prof[i1, 0] * L])
    f['mt5'] = np.array([prof[i5, 1] if side < 0 else prof[i5, 2], hy - prof[i5, 0] * L])
    f['ball_w'] = float(np.nanmax(prof[band, 2] - prof[band, 1]))
    # (the heel's breadth at the seat point, under the ankle bones: the malleoli are wider than the heel and stand
    # over the seat, not in it)
    q = F[(np.abs(F[:, 1] - (hy - SEAT_POINT * L)) < 0.06) & (F[:, 2] < F[:, 2].min() + 0.1 * L)]
    f['heel_w'] = float(np.ptp(q[:, 0]))
    f['seat_y'] = hy - SEAT_POINT * L
    f['joint_y'] = hy - JOINT * L
    f['ankle'] = np.asarray(ankle, float)
    f['ankle_share'] = (hy - ankle[1]) / L
    d = f['mt1'] - f['mt5']
    f['mtp_deg_measured'] = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
    return f


# ------------------------------------------------------------------ posing the foot on a heel

def mtp_axis(f):
    """The line the foot bends about on a heel: through the metatarsal heads at the floor, slanting back to the outside
    by MTP_AXIS_DEG (Raychoudhury 2014), through the middle of the measured 1st and 5th heads. Returns (point, dir)."""
    c = 0.5 * (f['mt1'] + f['mt5'])
    a = math.radians(MTP_AXIS_DEG)
    # (a cross line, turned so its outer end lies further back: at -x for her right foot the back is +y)
    d = np.array([1.0, f['side'] * math.tan(a), 0.0])
    return np.array([c[0], c[1], f['floor']]), d / np.linalg.norm(d)


def _rot(X, p, d, th):
    """Points X turned by th about the line through p along unit d (Rodrigues)."""
    v = X - p
    c, s = math.cos(th), math.sin(th)
    return p + v * c + np.cross(d, v) * s + np.outer(v @ d, d) * (1 - c)


def pose(X, f, th, band=0.55, ankle_span=(-0.25, 0.9)):
    """The foot raised onto a heel: the rear of the foot turned by th about the metatarsal axis (the toes stay on the
    floor), blended across a band (model units) about the axis; above the ankle the leg is carried up and forward
    with the ankle (so the shin stays upright and the ankle plantarflexes by th). X [n, 3]; returns the moved
    points. th > 0 raises the heel."""
    p, d = mtp_axis(f)
    side = np.cross(d, [0, 0, 1.0])                     # (the axis's normal in plan, pointing back toward the heel)
    if side[1] < 0:
        side = -side
    back = (X[:, :2] - p[:2]) @ side[:2]                # (how far behind the axis)
    w = _ss(-0.5 * band, 0.5 * band, back)
    turned = _rot(X, p, d, th)
    a = f['ankle']
    a2 = _rot(a[None], p, d, th)[0]
    up = _ss(a[2] + ankle_span[0], a[2] + ankle_span[1], X[:, 2])
    foot = X + w[:, None] * (turned - X)
    return foot * (1 - up)[:, None] + (X + (a2 - a)) * up[:, None]


def theta_for(f, Pl, h_seat):
    """The turn that lifts the sole of the foot at the seat point by h_seat (model units)."""
    near = np.abs(Pl[:, 1] - f['seat_y']) < 0.2
    z0 = Pl[near, 2].min()
    lo, hi = 0.0, 0.8
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if pose(Pl[near], f, m)[:, 2].min() - z0 < h_seat:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ the heel, derived

def derive(f, h_seat_cm, arch_fill=1.0, cup=1.0, top_of_seat=0.85, insole_mm=3.0, sole_mm=9.0):
    """The heel's design from the foot's measures f and a heel height (the rise of the sole of the foot at the seat
    point, cm). Returns a dict (model units unless named _cm/_mm/_deg). arch_fill and cup are the footbed's fit to the
    foot (1: the insole follows the posed sole of the foot through the waist / round the heel; 0: a straight shank
    line and a flat seat); top_of_seat (EXPERIMENTAL) is the tread's width as a share of the seat's."""
    L = f['L']
    last_L = L + LAST_ALLOW / CM
    d = dict(h_seat_cm=h_seat_cm, arch_fill=arch_fill, cup=cup)
    d['seat_y'] = f['seat_y']                                       # Golding's seat point
    d['top_y'] = f['seat_y']                                      # Golding: the heel on the seat point's perpendicular
    d['breast_y'] = f['heel_y'] - HEEL_OF_LAST * last_L              # One & Awl: the heel a quarter of the last
    d['seat_w'] = SEAT_OF_TREAD * f['ball_w']                       # Golding
    d['top_w'] = top_of_seat * d['seat_w']                       # EXPERIMENTAL share
    d['top_len'] = 2.0 * (d['top_y'] - d['breast_y'])           # (centred on the seat point, its front at the breast)
    d['shank_from'] = 0.5 * (d['seat_y'] + d['breast_y'])           # EXPERIMENTAL: half way into the seat
    d['shank_to'] = f['joint_y'] + 0.4                              # EXPERIMENTAL: 1 cm behind the joint line
    d['lifts'] = int(math.ceil(h_seat_cm * 10.0 / LIFT_MM)) + 1      # the atelier rule: N = A / B, one spare
    d['insole'] = insole_mm / MM
    d['sole'] = sole_mm / MM
    return d


# ------------------------------------------------------------------ the gauge: support

def footbed(Pf, f, d, P0):
    """The insole's top under each posed sole point Pf (z), for the design d. P0 the same points unposed. Ahead of
    the metatarsal axis the forepart lies flat on the floor; behind it the bed follows the posed foot by cup (across)
    and arch_fill (through the waist), else a straight line from the seat to the ball, flat across."""
    zb = np.full(len(Pf), f['floor'])
    p, dd = mtp_axis(f)
    yb = p[1]
    rear = Pf[:, 1] > yb
    # the straight shank line: from the seat's lowest posed point to the ball on the floor
    near = np.abs(Pf[:, 1] - d['seat_y']) < 0.25
    zs = Pf[near, 2].min()
    line = f['floor'] + (zs - f['floor']) * np.clip((Pf[:, 1] - yb) / (d['seat_y'] - yb), 0, None)
    # the foot's own surface along each line across (its lowest point at that y): a flat-across bed
    across = np.empty(len(Pf))
    for i, y in enumerate(Pf[:, 1]):
        across[i] = Pf[np.abs(Pf[:, 1] - y) < 0.12, 2].min()
    waist = _ss(yb, yb + 0.6, Pf[:, 1]) * (1 - _ss(d['seat_y'] - 0.3, d['seat_y'] + 0.3, Pf[:, 1]))
    heel = _ss(d['breast_y'] - 0.3, d['breast_y'] + 0.3, Pf[:, 1])
    flat_bed = np.minimum(line, across)                              # (a straight line, never above the foot)
    fit_w = np.clip(np.where(heel > 0.5, d['cup'], np.maximum(d['arch_fill'] * waist, heel * d['cup'])), 0, 1)
    rear_bed = flat_bed * (1 - fit_w) + Pf[:, 2] * fit_w
    zb[rear] = np.minimum(rear_bed[rear], Pf[rear, 2])
    return zb


def solve_contact(Pf, gap, area, W, y_w, x_w, k, iters=60):
    """A rigid foot pressed onto a Winkler bed: each sole cell pushes back k * compression (kPa per model unit of
    compression); the foot sinks and rocks (down, pitch, roll) until the bed's force equals W (N) and its centre of
    pressure lies on the weight line (x_w, y_w). area per cell in model units^2. Returns pressures (kPa)."""
    A = area * (MM ** 2) * 1e-6           # m^2
    yc, xc = Pf[:, 1].mean(), Pf[:, 0].mean()
    u = np.array([gap.min() + 0.02, 0.0, 0.0])
    for _ in range(iters):
        comp = u[0] + u[1] * (Pf[:, 1] - yc) + u[2] * (Pf[:, 0] - xc) - gap
        on = comp > 0
        p = k * np.where(on, comp, 0.0)                     # kPa
        Fz = p * A * 1e3                                    # N
        r = np.array([Fz.sum() - W, (Fz * (Pf[:, 1] - y_w)).sum(), (Fz * (Pf[:, 0] - x_w)).sum()])
        if abs(r[0]) < 1e-3 * W and abs(r[1]) < 1e-4 * W and abs(r[2]) < 1e-4 * W:
            break
        g = k * A * 1e3 * on
        B = np.stack([np.ones(len(Pf)), Pf[:, 1] - yc, Pf[:, 0] - xc], 1)
        C = np.stack([np.ones(len(Pf)), Pf[:, 1] - y_w, Pf[:, 0] - x_w], 1)
        J = (C * g[:, None]).T @ B
        try:
            du = np.linalg.solve(J + 1e-9 * np.eye(3), -r)
        except np.linalg.LinAlgError:
            du = -r * 1e-6
        u = u + np.clip(du, -0.2, 0.2)
    comp = u[0] + u[1] * (Pf[:, 1] - yc) + u[2] * (Pf[:, 0] - xc) - gap
    return k * np.clip(comp, 0, None), u


def regions(P0, f):
    s = (f['heel_y'] - P0[:, 1]) / f['L']
    return {name: (s >= a) & (s < b) for a, b, name in REGIONS}


def support(Pl, step, f, d, k, e, th=None, W=None):
    """The support gauge for the foot f on the design d: the foot posed on the heel, pressed on its bed, and every
    number of how it is carried. Pl the unposed sole points (plantar), k the bed's stiffness (kPa per unit), e the
    weight line's distance in front of the ankle joint (units). Returns a dict."""
    W = W if W is not None else 0.5 * EXP['mass'] * 9.81
    th = theta_for(f, Pl, d['h_seat_cm'] / CM) if th is None else th
    Pf = pose(Pl, f, th)
    a2 = pose(f['ankle'][None], f, th, ankle_span=(-10, -9))[0]      # (the ankle joint carried with the foot)
    zb = footbed(Pf, f, d, Pl) if d is not None else np.full(len(Pf), f['floor'])
    gap = Pf[:, 2] - zb
    area = np.full(len(Pf), step * step)
    y_w = a2[1] - e
    x_w = float((Pl[:, 0]).mean())
    A = area * (MM ** 2) * 1e-6
    sag = np.zeros(len(Pf))
    for _ in range(12 if d is not None else 1):
        # the foot pressed on its bed; the waist then gives under what it carries (standing), and the foot is pressed
        # again on the bed as it has given, until the two agree (the shank's support of the arch)
        p, u = solve_contact(Pf, gap + sag, area, W, y_w, x_w, k)
        Fz = p * A * 1e3
        if d is None:
            break
        sh = shank(Pf, Fz, zb, f, d)
        sag = 0.5 * sag + 0.5 * sh['cell_sag'] / MM
    reg = regions(Pl, f)
    out = dict(theta_deg=math.degrees(th), y_w=y_w, x_w=x_w, p=p, Pf=Pf, zb=zb, gap=gap, Fz=Fz, W=W)
    out['share'] = {n: float(Fz[m].sum() / W) for n, m in reg.items()}
    out['peak'] = {n: float(p[m].max()) if m.any() else 0.0 for n, m in reg.items()}
    out['peak_all'] = float(p.max())
    out['contact_cm2'] = float((p > 1.0).sum() * step * step * CM * CM)
    # the bed's slope under each cell (down toward the toes): the forward push it gives the foot
    yb_ = np.round(Pf[:, 1] / 0.15).astype(int)
    keys, inv = np.unique(yb_, return_inverse=True)
    prof = np.array([zb[inv == i].min() for i in range(len(keys))])
    sl_k = np.gradient(prof, keys * 0.15) if len(keys) > 2 else np.zeros(len(keys))
    sl = sl_k[inv]
    out['bed_profile'] = np.stack([keys * 0.15, prof], 1)
    out['forward_N'] = float((Fz * np.clip(sl, 0, None)).sum())       # tan(angle) * normal: the push toward the toes
    out['pitch_deg'] = math.degrees(math.atan(np.median(sl[(Pf[:, 1] > f['seat_y'] - 0.4) & (Pf[:, 1] < f['seat_y'] + 0.4)]))) if d else 0.0
    if d is not None:
        rear = Pf[:, 1] > d['breast_y']
        Fr = Fz[rear].sum()
        out['rear_N'] = float(Fr)
        cy = float((Fz[rear] * Pf[rear, 1]).sum() / max(Fr, 1e-9))
        cx = float((Fz[rear] * Pf[rear, 0]).sum() / max(Fr, 1e-9))
        out['rear_cop'] = (cx, cy)
        out['rear_cop_off_mm'] = (cy - d['top_y']) * MM               # + behind the tread's middle
        out['rear_cop_on_top'] = abs(cy - d['top_y']) < 0.5 * d['top_len']
        h = (d['h_seat_cm'] / CM) + d['insole'] + d['sole']
        half = 0.5 * d['top_w'] - abs(cx - float(Pl[Pl[:, 1] > d['breast_y'], 0].mean()))
        out['tip_deg'] = math.degrees(math.atan2(half, h))             # tilt before the load line leaves the tread
        out['h_breast_cm'] = float((zb[np.abs(Pf[:, 1] - d['breast_y']) < 0.2].min() - f['floor']) * CM + d['sole'] * CM)
        # the waist as a beam: the moment along it between the heel's breast and the forepart's ground
        out.update(shank(Pf, Fz, zb, f, d))
        walk = shank(Pf, Fz, zb, f, d, scale=GAIT_PEAK * 2.0)
        out.update(shank_M_walk=walk['shank_M_peak'], shank_sigma_walk=walk['shank_sigma'],
                   shank_SF_walk=walk['shank_SF'], shank_sag_walk=walk['shank_sag_mm'],
                   shank_S_need=walk['shank_S_need'])
        # the forward push the footbed's slope gives, less what friction holds: carried by the vamp and the lacing
        slide = max(0.0, out['forward_N'] - EXP['friction'] * Fr)
        out['vamp_N'] = slide
        out['vamp_N_walk'] = slide * GAIT_PEAK * 2.0                    # (one foot, the walking peak)
        throat = 0.9 * f['ball_w'] * MM                                 # EXPERIMENTAL: the vamp's throat line, mm
        out['vamp_tension_N_mm'] = out['vamp_N_walk'] / throat
        out['vamp_stress'] = out['vamp_tension_N_mm'] / EXP['leather_t']
        out['seat_fit_mm'] = (d['seat_w'] - f['heel_w']) * MM            # the seat's room over the foot's heel
        out['heel_kpa_mean'] = float(Fr / max((p[rear] > 1).sum() * (step * MM) ** 2 * 1e-6, 1e-9) / 1e3)
    return out


def ground_y(Pf, zb, f, d):
    """Where the sole's under face first meets the floor, going forward from the heel's breast: the waist is bridged
    from the breast to here (the bed's height above the floor, less nothing: insole and sole are the same thickness
    under the whole foot, so the sole touches where the bed comes down to the floor)."""
    ys = np.linspace(d['breast_y'], f['toe_y'], 120)
    for y in ys:
        m = np.abs(Pf[:, 1] - y) < 0.1
        if m.any() and zb[m].min() - f['floor'] < 0.02:
            return float(y)
    return float(f['joint_y'])


def shank(Pf, Fz, zb, f, d, scale=1.0):
    """The waist as a beam from the heel's breast to where the sole meets the floor (ground_y): held fast at the heel
    (the heel block is nailed through the seat; SATRA TM58 and ISO 18896 clamp a shank at its heel end the same
    way) and resting on the floor at its front end (a propped cantilever). The sole of the foot presses down on it
    (Fz on each cell over the span, times scale: 1 standing, GAIT_PEAK * 2 the walking peak on one foot). Its
    bending stiffness is the steel shank's and the sole's and insole's leather together. Returns the moment along it,
    the steel's stress and safety, the sag, and the sag at each cell of the sole (mm) for the contact's next round."""
    yg = ground_y(Pf, zb, f, d)
    ys = np.linspace(d['breast_y'], yg, 48)
    x = (d['breast_y'] - ys) * MM                                   # mm from the heel's breast
    span = max(x[-1], 1e-6)
    on = (Pf[:, 1] < d['breast_y']) & (Pf[:, 1] >= yg)
    xi = (d['breast_y'] - Pf[on, 1]) * MM
    Fi = Fz[on] * scale
    I = EXP['shank_w'] * EXP['shank_t'] ** 3 / 12.0
    bw = EXP['waist_w_of_seat'] * d['seat_w'] * MM
    t_l = (d['sole'] + d['insole']) * MM
    EI_leather = EXP['sole_E'] * bw * t_l ** 3 / 12.0             # (the sole and insole as one slab over the shank)
    EI = STEEL_E * I + EI_leather

    def bend(M):
        curv = M / EI
        sl = np.concatenate([[0], np.cumsum(0.5 * (curv[1:] + curv[:-1]) * np.diff(x))])
        return np.concatenate([[0], np.cumsum(0.5 * (sl[1:] + sl[:-1]) * np.diff(x))])
    # the loads alone, as a cantilever from the heel (sagging down: positive)
    Ml = np.array([(Fi[xi > xx] * (xi[xi > xx] - xx)).sum() for xx in x])
    dl = bend(Ml)
    # the floor's prop at the front end, R, so the end does not sag: R span^3 / (3 EI) = dl[-1]
    R = dl[-1] * 3.0 * EI / span ** 3 if span > 1 else 0.0
    M = Ml - R * (span - x)
    defl = bend(M)
    sigma = np.abs(M).max() * (STEEL_E * I / EI) * (0.5 * EXP['shank_t']) / I     # (the steel's share of the moment)
    sag = float(defl.max())
    need = max(0.0, EI * max(sag, 0) / EXP['waist_sag'] - EI_leather) / 1e3
    cell = np.zeros(len(Pf))
    cell[on] = np.interp(xi, x, defl)
    return dict(ground_y=yg, span_mm=float(span), shank_x=x, shank_M=M, shank_defl=defl,
                shank_M_peak=float(np.abs(M).max()), shank_sigma=float(sigma),
                shank_SF=EXP['shank_yield'] / max(sigma, 1e-9), shank_sag_mm=sag, shank_S=float(STEEL_E * I / 1e3),
                waist_EI=float(EI / 1e3), shank_S_need=float(need), cell_sag=cell,
                shank_len_mm=float((d['shank_from'] - d['shank_to']) * MM))


# ------------------------------------------------------------------ calibration on the known-good case

def calibrate(Pl, step, f, rounds=6):
    """The bed's stiffness k and the weight line e from Cavanagh 1987 (barefoot standing): e so the heel takes 60 % of
    the load, k so its peak is 139 kPa. Returns (k, e, the barefoot result). The midfoot, forefoot and toes' shares
    and the forefoot's peak are not fitted: they are the validation."""
    k, e = 4000.0, 1.5
    for _ in range(rounds):
        lo, hi = -2.0, 6.0
        for _ in range(30):
            e = 0.5 * (lo + hi)
            r = support(Pl, step, f, None, k, e, th=0.0)
            if r['share']['heel'] > CAVANAGH['heel']:
                lo = e
            else:
                hi = e
        k *= CAVANAGH['heel_peak'] / max(r['peak']['heel'], 1e-6)
    r = support(Pl, step, f, None, k, e, th=0.0)
    return k, e, r


# ------------------------------------------------------------------ the derived height, and a summary

def best_height(Pl, step, f, k, e, arch_fill=1.0, cup=1.0, lo=0.3, hi=5.0, **kw):
    """The heel's height (cm, the rise at the seat point) at which the rear foot's load centres on the heel's tread:
    the heel stands under the load (Golding's perpendicular from the seat point) and the tread is loaded evenly
    front to back. Lower heels put the rear load behind the tread's middle, toward the back edge; higher ones in
    front, toward the breast. Found by halving the interval (the offset falls as the heel rises)."""
    def off(h):
        return support(Pl, step, f, derive(f, h, arch_fill=arch_fill, cup=cup, **kw), k, e)['rear_cop_off_mm']
    a, b = off(lo), off(hi)
    if a * b > 0:
        return lo if abs(a) < abs(b) else hi
    for _ in range(18):
        m = 0.5 * (lo + hi)
        if off(m) > 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def summary(f, d, o):
    """One line of the numbers that matter."""
    s = o['share']
    return ('heel %.1f cm at the seat (%.1f at the breast), pitch %.0f deg; seat %.1f cm wide over a heel of %.1f '
            '(room %.1f mm); tread %.1f x %.1f cm; loads heel %.0f%% midfoot %.0f%% forefoot %.0f%% toes %.0f%%; '
            'peak %.0f kPa; rear load %.1f mm off the tread middle; tips at %.0f deg; forward push %.0f N, vamp %.0f N '
            '(walking %.0f N, %.2f N/mm); waist moment %.0f N mm walking, shank safety %.1f, sag %.2f mm (needs S %.0f, '
            'has %.0f kN mm2)'
            % (d['h_seat_cm'], o['h_breast_cm'], o['theta_deg'], d['seat_w'] * CM, f['heel_w'] * CM, o['seat_fit_mm'],
               d['top_w'] * CM, d['top_len'] * CM, 100 * s['heel'], 100 * s['midfoot'], 100 * s['forefoot'],
               100 * s['toes'], o['peak_all'], o['rear_cop_off_mm'], o['tip_deg'], o['forward_N'], o['vamp_N'],
               o['vamp_N_walk'], o['vamp_tension_N_mm'], o['shank_M_walk'], o['shank_SF_walk'], o['shank_sag_walk'],
               o['shank_S_need'], o['shank_S']))
