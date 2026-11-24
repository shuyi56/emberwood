"""DRAFT: a ponytail as a flow field, before any real hair (John, 2026-11-25: "build the proper initial conditions that
creates the look in the picture and show me the flow field before working on the real hair").

    python3 tools/hair/ponytail_flow.py HEAD.json.gz OUT_DIR [--photo reference.measure.json]

Nothing here grooms, solves or renders strands. It lays out where every hair goes as streamlines, measures the field
against John's invariants, and draws it as lines over a clay head.

The model (John's, from the project chat). s runs from a root to its outlet, u across the flow; F(s, u) is strands per
unit width. dF/ds + d(F m)/du = r with F >= F_min, |m| <= m_max, |dm/ds| <= K, r >= 0, flux conserved, one sink per
basin, progress monotonic, strands keep rank.

How it is built.
  * Scalp (the funnel): every root's hair runs to the tie along the shortest way over the skull: the great arc from
    its direction to the tie's, seen from the skull's centre, laid on the skin. Arcs to one point never cross and
    never turn back, so rank and monotonic progress hold by construction; the gauges re-measure both on the built
    lines anyway. Chart: rho (mm along the skin to the tie) and alpha (the bearing the hair arrives from at the tie).
  * Lift: hair rooted further out lies over hair rooted nearer the tie. The stack's thickness is the flux through a
    cut times a hair's section over the packing, so the lift grows toward the tie with no free profile.
  * Handover: at the ring round the tie each line turns out of the scalp into the bundle's axis on a quintic curve,
    matched in position and direction at both ends. Its place in the bundle keeps its bearing (in order) and its depth
    in the stack (inside to outside).
  * Free fall: the bundle's middle line is a heavy elastica launched at the tie's angle; its envelope is the ponytail
    shape equation [GWB12]; the S wave is laid on the middle line, none at the tie, in full a quarter head below it.
  * Bangs and loose locks: separate basins with their own outlets, drawn as design elements.

Sources.
  [GWB12] Goldstein, Warren, Ball 2012, "Shape of a ponytail and the statistical physics of hair fiber bundles",
          PRL 108, 078101 (arXiv 1204.0371). Envelope l^3 R'''' - (L - s) R'' + R' - Pi(R) = 0, Pi = Pi0 (1 - R/R*),
          Pi0 0.85, R* 6 cm; bending length l = (A / (lambda g))^(1/3) about 5 cm; launch angle of the outer fibres
          17 deg; packing at the clamp 0.95 (9500 fibres, clamp radius 4 mm); fibre density uniform across the bundle,
          self-similar down it; a hair's own curvature is "reduced in the vicinity of the clamp".
  [SP]    StatPearls, Traction alopecia (NBK470434): loss starts "in the temporal and preauricular regions and above
          the ears"; the treatment is a looser style. No force threshold.
  [FS11]  Samrao et al. 2011, Dermatol Online J 17(11):1, the fringe sign: the rim hairs "may be shorter and thus
          'fall out' of a tight ponytail"; too tight is tenting (the scalp raised by the pull), pain, stinging.
          No force, slack or lift number.
  [LO][NB][SV][BIO] L'Oreal, NewBeauty, Sam Villa, Biolage ponytail guides: tease or backcomb the crown for lift,
          "pull out a few strands near the hairline", drape the sides over the ears, curl with a 1.5 inch wand.
          None gives a number for tension, slack, lift or the loose strands' thickness.
Every constant is flagged SOURCED, MEASURED (off the photo, in heads), CALIBRATED or DRAFT (no source found)."""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF

MM = SF.MM                      # one mm in heads (HH, chin to crown, 219 mm)
HAIR_D_MM = SF.SHAFT_UM / 1000  # SOURCED [LS20] via scalp_field: 80 micron
HAIR_A = math.pi * HAIR_D_MM ** 2 / 4          # mm^2, one hair's section

# ---- the photo, in heads (reference.measure.json: unit head_top -> chin, 620 px; plumb the ear canal)
PH = dict(tie_above_chin=0.545, tie_behind_ear=0.471, tie_w=0.210, apex_rise=0.116, apex_behind_tie=0.158,
          pony_len_y=1.694, tail_behind_tie=0.31, pony_max_w=0.524, fall_deg=2.2, wave_len=0.548, wave_swing=0.097, bang_edge_h=0.594,
          ear_wisp=0.453, head_top_hair=0.03)
# ---- initial and boundary conditions
TIE_Z = PH['tie_above_chin'] * (1 + PH['head_top_hair'])    # MEASURED: the tie's height above the chin, HH (the photo's
                                                            # unit includes about 3% of hair over the crown)
TIE_W = PH['tie_w']             # MEASURED: the bundle's width at the tie, HH (46 mm: a loose tie, see the gauge)
TIE_OFF = 0.07                  # MEASURED: the tie's middle stands this far off the skin (0.471 behind the ear canal
                                # against the skull's 0.40 at that height)
LAUNCH_DEG = math.degrees(math.atan2(PH['apex_rise'], PH['apex_behind_tie']))   # MEASURED: 36 deg above level, back
LAUNCH_FIBRE_DEG = 17.0         # SOURCED [GWB12]: the outer fibres' angle to the axis at the tie
PHI_TIE_MAX = 0.95              # SOURCED [GWB12]: the packing a tied bundle reaches. (John's 0.6 to 0.7 was a guess at
                                # this number; the measured clamp is nearly close packed. The gauge reports both)
PHI_LOOSE = 0.165               # CALIBRATED (groom.py PACKING): loose hair over the head
PHI_NEAR_MM = 60.0              # DRAFT: the packing rises from loose to the tie's over the last 6 cm of scalp
F_MIN = 4.0                     # hairs per mm: hairline_gap.py's HL_COVER, the no-gap floor (DRAFT there)
EAR_DELTA_MM = 3.0              # DRAFT: the hair's offset over the ear
EAR_RIM_MM = 5.0                # DRAFT: the ear rim's radius; K >= 1 / (rim + delta)
K_MAX = 1.0 / (EAR_RIM_MM + EAR_DELTA_MM)       # per mm: the bound on curvature
CURL_STEP_MM = 4.0              # the length curvature is read over (half the tightest radius allowed)
THETA_MAX_DEG = 45.0            # DRAFT: the steepest a strand converges on the bundle's axis (m_max = tan of it)
SLACK_EPS = 0.005               # DRAFT: arc >= (1 + eps) x the way along the skin. No source gives a number
                                # ([SP][FS11][LO][NB]); 0.5% is the slack of a 9 mm rise over the crown on a 20 cm sweep, the
                                # photo's volume there (3% of a head)
RING_MM = 34.0                  # DRAFT: the handover starts this far outside the tie's own radius
HAND_AXIS_MM = 30.0             # DRAFT: and ends this far down the bundle's axis
LAUNCH_MAX_DEG = 84.0           # the steepest launch tried (the photo's tie-to-apex line is 36 deg; its bundle leaves
                                # the tie steeply and folds over, so the fit below is free to go well past that)
HAIR_LEN = 2.0                  # CALIBRATED, HH: one length for every hair, set so the longest tail matches the photo
BEND_L_MM = 50.0                # SOURCED [GWB12]: a hair's bending length (groom.py's own: 44.8)
PI0, RSTAR_MM = 0.85, 47.0      # Pi0 SOURCED [GWB12]. R*, the radius where the swelling ends: CALIBRATED to the photo's
                                # widest tail (0.52 head with its waves); [GWB12]'s 60 mm is for a 9500-hair switch
WAVE_LEN = PH['wave_len']       # MEASURED
WAVE_A = 0.5 * PH['wave_swing'] # MEASURED: half the crest-to-trough swing
WAVE_ON = 0.25                  # DRAFT, HH: the wave grows in over this length below the tie ([GWB12]: reduced there)
# ---- the basins that do not go to the tie
BANG_TH = 0.75                  # |theta|: the fringe's section, corner to corner (groom.py's bang section)
BANG_DEPTH_MM = 60.0            # the section's depth behind the hairline: the creator's (groom.py BANG_DEPTH note)
BANG_EDGE_Z = PH['bang_edge_h'] * (1 + PH['head_top_hair'])   # MEASURED: the fringe's edge, HH above the chin
LOOSE = {                       # DRAFT sizes (no source gives a thickness); places from [LO][BIO] and the photo
    'temple lock': dict(th=(0.75, 0.95), depth_mm=22.0, z=(0.6, 1.0), len=0.62),
}
# (John, 2026-11-25, lines drawn on the overlay from the temple over the ear's top to the tie: "the green and purple
# should follow those lines". The ear wisp and the nape wisps are gone: those roots go to the tie, see SAG.)
# What the first field lacked: gravity along the scalp. Every hair took the shortest way to the tie, which is the
# line of a hair pulled taut. A hair with slack hangs between its root and the tie: on the side of the head, where
# the skin is steep, it sags below the shortest way, over the ear's top, and comes up into the tie from under it.
SAG_G = 0.9                     # CALIBRATED to John's drawn lines: gravity's pull across the hair's line against the
                                # pull toward the tie, where the skin is vertical (the dip it gives is in the gauges)
SAG_FADE = (0.15, 0.55)         # rad from the tie (12 to 45 mm of skin): the pull fades, and the hair turns up into it
RESOLVE = {                     # the roots re-solved to the tie along John's lines (the first field's ear wisp and
                                # nape wisps, same patches): drawn in their old colours so they can be followed
    'ear wisp': dict(th=(1.05, 1.5), depth_mm=16.0, z=(0.0, 0.66)),
    'nape wisps': dict(th=(2.3, 3.2), depth_mm=13.0, z=(0.0, 1.0)),
}
LOOSE_FRACTION = (0.01, 0.05)   # John's bound on the loose locks' share of all hair
CUTS_MM = (150, 120, 90, 60, 40, 25)     # the cuts the balance sheet is taken on (mm of skin from the tie)


def sm(a, b, x):
    return SF.sm(a, b, x)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def ear_depth(X):
    """Depth inside the ear's ellipsoid (HH, > 0 inside), as groom.ear_depth_np."""
    EAR_X, EAR_OUT = 0.33, 0.035
    ax = np.array([0.5 * EAR_OUT + 0.005, 0.06, 0.5 * (SF.EAR_TOP - SF.EAR_BOTTOM)])
    out = -np.ones(len(X))
    for sx in (-1.0, 1.0):
        c = np.array([sx * (EAR_X + 0.5 * EAR_OUT), SF.EAR_CANAL[1], 0.5 * (SF.EAR_TOP + SF.EAR_BOTTOM)])
        out = np.maximum(out, (1 - np.linalg.norm((X - c) / ax, axis=1)) * ax.min())
    return out


def slerp(U0, U1, t):
    """Great arcs from each U0 (n, 3) to U1 (3,), at t (k,): (n, k, 3)."""
    c = np.clip(U0 @ U1, -1, 1)
    om = np.arccos(c)[:, None]
    so = np.maximum(np.sin(om), 1e-9)
    a = np.sin((1 - t[None]) * om) / so
    b = np.sin(t[None] * om) / so
    return a[..., None] * U0[:, None] + b[..., None] * U1[None, None]


# --------------------------------------------------------------------------------------------- the field
CACHE = None                    # the cache folder, set in main


def cached(name, key, fn):
    """fn()'s result, stored under name and reused while key (anything hashable to text) is the same. Each stage
    names what it depends on, so changing a constant redoes that stage and the ones after it, not the ones before."""
    import pickle, hashlib, time
    if CACHE is None:
        return fn()
    hk = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    p = os.path.join(CACHE, '%s-%s.pkl' % (name, hk))
    t0 = time.time()
    if os.path.exists(p):
        out = pickle.load(open(p, 'rb'))
        print('cache: %-12s reused (%.1f s)' % (name, time.time() - t0), file=sys.stderr)
        return out
    out = fn()
    for old in os.listdir(CACHE):
        if old.startswith(name + '-'):
            os.remove(os.path.join(CACHE, old))
    pickle.dump(out, open(p, 'wb'), protocol=4)
    print('cache: %-12s solved (%.1f s)' % (name, time.time() - t0), file=sys.stderr)
    return out


def file_key(path):
    st = os.stat(path)
    return (os.path.basename(path), st.st_size, int(st.st_mtime))


def scalp(head):
    """Stage 1: the skin and the scalp field (depends on the head and scalp_field.py only)."""
    def solve():
        surf = SF.Surface(SF.load_head(head))
        return surf.u, surf.r, SF.build(surf)
    u, r, F = cached('scalp', (file_key(head), file_key(SF.__file__)), solve)
    from scipy.spatial import cKDTree
    surf = SF.Surface.__new__(SF.Surface)
    surf.u, surf.r, surf.tree = u, r, cKDTree(u)
    return surf, F


def build(head):
    surf, F = scalp(head)
    hl, _ = SF.make_hairline(F['landmarks'])
    m = F['mask'] & (F['density'] > 0)
    P, N, th = F['P'][m], F['N'][m], F['theta'][m]
    hairs = (F['density'] * SF.point_area(F))[m] * (SF.HH_MM / 10) ** 2
    inside_mm = F['inside'][m] / MM
    grow = F['dir'][m]
    n = len(P)

    # --- basins
    basin = np.zeros(n, int)                                  # 0: the tie
    names = ['tie', 'bangs']
    basin[(np.abs(th) < BANG_TH) & (inside_mm < BANG_DEPTH_MM) & (P[:, 1] < 0)] = 1
    for k, (nm, c) in enumerate(LOOSE.items()):
        a = np.abs(th)
        sel = (basin == 0) & (a > c['th'][0]) & (a < c['th'][1]) & (inside_mm < c['depth_mm']) \
            & (P[:, 2] > c['z'][0]) & (P[:, 2] < c['z'][1])
        basin[sel] = 2 + k
        names.append(nm)

    resolve = np.zeros(n, int)                                # 0: as first solved (blue); 1, 2: re-solved sets
    for k, (nm, c) in enumerate(RESOLVE.items()):
        a = np.abs(th)
        resolve[(basin == 0) & (a > c['th'][0]) & (a < c['th'][1]) & (inside_mm < c['depth_mm'])
                & (P[:, 2] > c['z'][0]) & (P[:, 2] < c['z'][1])] = 1 + k

    # --- the tie: its foot on the skin, its axis, its middle
    el = np.linspace(-0.6, 0.9, 3000)
    Pm = surf.point(np.c_[np.zeros_like(el), np.cos(el), np.sin(el)])
    T = Pm[np.argmin(np.abs(Pm[:, 2] - TIE_Z))]
    UT = unit(T - SF.CENTRE)
    nT = surf.normals(UT[None])[0]
    la = math.radians(LAUNCH_DEG)
    axis = np.array([0.0, math.cos(la), math.sin(la)])
    C = T + nT * TIE_OFF
    r_tie = 0.5 * TIE_W

    # --- the funnel: great arcs to the tie, on the skin
    tie = basin == 0
    U0 = unit(P - SF.CENTRE)
    K = 120
    t = np.linspace(0, 1, K)
    # Stage: the blue lines, the shortest way over the skull to the tie (great arcs). Cached: they depend on the
    # head and the tie's place only, and are not solved again when the other lines change.
    def blue():
        Ub = slerp(U0, UT, t)
        return (Ub.astype(np.float32), surf.point(Ub.reshape(-1, 3)).reshape(n, K, 3),
                surf.normals(Ub.reshape(-1, 3)).reshape(n, K, 3).astype(np.float32))
    U, S, NL = cached('blue-lines', (file_key(head), file_key(SF.__file__), TIE_Z, K, n, 'with normals'), blue)
    U, S, NL = U.astype(float), S.copy(), NL.astype(float)
    # Stage: the re-solved lines only. Their direction is toward the tie along the skin plus gravity's pull across
    # that line (SAG_G), so they sag below the shortest way, over the ear's top, and come up into the tie from under
    # it. Each step closes the same share of the angle left to the tie, so every line arrives.
    ri = np.where(resolve > 0)[0]
    if len(ri):
        Uc = U0[ri].copy()
        U[ri, 0] = Uc
        for k in range(1, K):
            om = np.arccos(np.clip(Uc @ UT, -1, 1))
            d = unit(UT[None] - (Uc @ UT)[:, None] * Uc)
            g = np.array([0, 0, -1.0])[None] - (-Uc[:, 2:3]) * Uc
            gp = g - (g * d).sum(1, keepdims=True) * d
            v = d + SAG_G * sm(SAG_FADE[0], SAG_FADE[1], om)[:, None] * gp
            h = om / (K - k) * np.linalg.norm(v, axis=1)
            Uc = unit(Uc + unit(v) * h[:, None])
            U[ri, k] = Uc
        U[ri, -1] = UT
        S[ri] = surf.point(U[ri].reshape(-1, 3)).reshape(len(ri), K, 3)
        NL[ri] = surf.normals(U[ri].reshape(-1, 3)).reshape(len(ri), K, 3)
    S[:, 0] = P
    dist = np.arccos(np.clip(U @ UT, -1, 1)) * (float(np.mean(np.linalg.norm(P - SF.CENTRE, axis=1))) / MM)
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2)
    rho = np.c_[seg[:, ::-1].cumsum(1)[:, ::-1], np.zeros(n)] / MM           # mm of skin left to the tie
    rho0 = rho[:, 0]
    # bearing at the tie: the direction the hair arrives from, in the tie's tangent plane (0: from straight above)
    e_up = unit(np.array([0, 0, 1.0]) - nT[2] * nT)
    e_sd = np.cross(e_up, nT)
    d_in = S[:, -8] - T
    alpha = np.arctan2(d_in @ e_sd, d_in @ e_up)

    # --- F on the chart: flux through each cut, per mm of the cut's width
    Rs = float(np.mean(np.linalg.norm(P - SF.CENTRE, axis=1))) / MM          # the skull's mean radius, mm
    na = 48
    ab = np.minimum(((alpha + np.pi) / (2 * np.pi) * na).astype(int), na - 1)
    rho_ax = np.arange(10.0, 260.0, 5.0)
    width = lambda r: Rs * np.sin(np.clip(r / Rs, 0, np.pi)) * (2 * np.pi / na)   # mm of cut per bearing bin
    flux = np.zeros((na, len(rho_ax)))
    for b in range(na):
        r0 = np.sort(rho0[tie & (ab == b)])
        w = hairs[tie & (ab == b)][np.argsort(rho0[tie & (ab == b)])]
        cs = np.r_[0, np.cumsum(w)]
        flux[b] = cs[-1] - cs[np.searchsorted(r0, rho_ax)]                   # hairs rooted beyond the cut
    Fmap = flux / width(rho_ax)[None]
    # where the scalp has roots on the far side of the cut at all (inside the hairline): the domain of the no-gap rule
    reach = np.array([[(tie & (ab == b) & (rho0 > r + 10.0)).any() for r in rho_ax] for b in range(na)])
    # (10 mm inside the last root: the hairline's own rim is gauged on the built lines, as hairline_gap.py does)

    # --- lift: the stack of hair rooted between here and my root, at the local packing
    phi = lambda r: PHI_LOOSE + (PHI_TIE_MAX * 0.7 - PHI_LOOSE) * (1 - sm(r_tie / MM, PHI_NEAR_MM, r))
    lift = np.zeros((n, K))
    order = {}
    for b in range(na):
        ii = np.where(tie & (ab == b))[0]
        if not len(ii):
            continue
        ii = ii[np.argsort(rho0[ii])]
        cs = np.r_[0, np.cumsum(hairs[ii])]
        pos = np.searchsorted(rho0[ii], rho[ii], side='left')                # (m, K): roots nearer than this node
        mine = np.arange(len(ii))[:, None]
        under = cs[mine] - cs[np.minimum(pos, mine)]                         # hairs rooted between the node and me
        lift[ii] = under * HAIR_A / phi(rho[ii]) / np.maximum(width(rho[ii]), 0.3)
        order[b] = ii
    lift = np.minimum(lift, 0.5 * TIE_W / MM)                                # (no deeper than the tie is wide)
    # eased from the root, and over the ear at its offset
    for _ in range(6):                                                       # (the stack is read in bins: eased along)
        lift[:, 1:-1] = 0.25 * lift[:, :-2] + 0.5 * lift[:, 1:-1] + 0.25 * lift[:, 2:]
    lift_mm = lift * sm(0.0, 12.0, (rho0[:, None] - rho)) + 0.5
    X = S + NL * (lift_mm * MM)[..., None]
    ed = ear_depth(X.reshape(-1, 3)).reshape(n, K)
    over_ear = (ed > -EAR_DELTA_MM * MM)
    side = np.sign(X[..., 0])[..., None] * np.array([1.0, 0, 0])
    push = np.maximum(ed + EAR_DELTA_MM * MM, 0)
    for _ in range(40):                                                      # (onto the ear and off it without a corner)
        push[:, 1:-1] = np.maximum(push[:, 1:-1], 0.25 * push[:, :-2] + 0.5 * push[:, 1:-1] + 0.25 * push[:, 2:])
    for _ in range(6):
        push[:, 1:-1] = 0.25 * push[:, :-2] + 0.5 * push[:, 1:-1] + 0.25 * push[:, 2:]
    push = push * sm(0.0, 10.0, rho0[:, None] - rho)                         # (the root stays on the skin)
    X = X + side * push[..., None]
    for _ in range(3):
        X[:, 1:-1] = 0.25 * X[:, :-2] + 0.5 * X[:, 1:-1] + 0.25 * X[:, 2:]

    # slack: a line shorter than (1 + eps) x its way along the skin is raised off the head in one soft arc (the
    # crown's volume: "tease the roots at crown for lift" [NB][BIO]) until it has it
    Ss = S.copy()                              # (the skin's way, eased as the line is: the skin's read is noisy)
    for _ in range(3):
        Ss[:, 1:-1] = 0.25 * Ss[:, :-2] + 0.5 * Ss[:, 1:-1] + 0.25 * Ss[:, 2:]
    on = (0.5 * (rho[:, 1:] + rho[:, :-1]) >= r_tie / MM + RING_MM)
    arc = (np.linalg.norm(np.diff(X, axis=1), axis=2) * on).sum(1)
    skin = (np.linalg.norm(np.diff(Ss, axis=1), axis=2) * on).sum(1)
    need = np.maximum(1.3 * SLACK_EPS - (arc / np.maximum(skin, 1e-9) - 1), 0)
    amp = (2 / np.pi) * np.sqrt(need) * skin
    uu = np.clip((rho0[:, None] - rho) / np.maximum(rho0 - (r_tie / MM + RING_MM), 1.0)[:, None], 0, 1)
    bump = amp[:, None] * np.sin(np.pi * uu) ** 2
    X = X + NL * bump[..., None]
    lift_mm = lift_mm + bump / MM
    return dict(surf=surf, F=F, P=P, N=N, th=th, hairs=hairs, basin=basin, names=names, T=T, nT=nT, axis=axis, C=C,
                r_tie=r_tie, X=X, S=S, Ss=Ss, resolve=resolve, dist=dist, rho=rho, rho0=rho0, alpha=alpha, ab=ab, na=na, rho_ax=rho_ax, Fmap=Fmap,
                reach=reach, flux=flux, lift_mm=lift_mm, over_ear=over_ear, grow=grow, e_up=e_up, e_sd=e_sd,
                inside_mm=inside_mm, hl=hl, Rs=Rs, width=width)


# --------------------------------------------------------------------------------------------- the free fall
def centreline(L_mm, launch_deg, ell_mm, n=400):
    """The bundle's middle line: a heavy elastica clamped at the launch angle. The tail is long against its bending
    length (L / ell about 5 to 9: "dominated by gravity" [GWB12]), so the weight below each point is the tension that
    straightens it, and the elastica's first integral gives the turn directly:
    theta' = -sqrt(2 (L - s) (1 + sin theta) / ell^3), from the launch angle down to plumb.
    Returns s (mm), back (mm), up (mm), theta."""
    s = np.linspace(0, L_mm, n)
    ds = s[1] - s[0]
    th = np.zeros(n); th[0] = math.radians(launch_deg)
    for k in range(n - 1):
        sub = 8
        t = th[k]
        for q in range(sub):
            w = max(L_mm - (s[k] + q * ds / sub), 0.0)
            t = max(t - math.sqrt(2 * w * (1 + math.sin(t)) / ell_mm ** 3) * ds / sub, -0.5 * math.pi)
        th[k + 1] = t
    back = np.r_[0, np.cumsum(np.cos(th[:-1]) * ds)]
    up = np.r_[0, np.cumsum(np.sin(th[:-1]) * ds)]
    return s, back, up, th


def envelope(L_mm, Rc_mm, n=400):
    """[GWB12] eq. 3 with eq. 4's pressure: l^3 R'''' - (L - s) R'' + R' - Pi0 (1 - R / R*) = 0,
    R(0) = Rc, R'(0) = tan 17 deg, R''(L) = R'''(L) = 0."""
    from scipy.integrate import solve_bvp
    l3 = BEND_L_MM ** 3
    s = np.linspace(0, L_mm, n)
    f = lambda s_, y: np.vstack([y[1], y[2], y[3], ((L_mm - s_) * y[2] - y[1] + PI0 * (1 - y[0] / RSTAR_MM)) / l3])
    bc = lambda a, b: np.array([a[0] - Rc_mm, a[1] - math.tan(math.radians(LAUNCH_FIBRE_DEG)), b[2], b[3]])
    y0 = np.vstack([Rc_mm + (RSTAR_MM - Rc_mm) * (1 - np.exp(-s / 80)), np.zeros(n), np.zeros(n), np.zeros(n)])
    sol = solve_bvp(f, bc, s, y0, max_nodes=50000, tol=1e-4)
    return s, sol.sol(s)[0], sol.status


def free_fall(B):
    L0 = round(float((HAIR_LEN - B['rho0'][B['basin'] == 0].min() * MM) / MM), 2)
    key = (L0, PH, LAUNCH_MAX_DEG, B['r_tie'], tuple(B['C']), BEND_L_MM, PI0, RSTAR_MM, WAVE_A, WAVE_LEN, WAVE_ON, 'v2')
    return cached('free-fall', key, lambda: _free_fall(B))


def _free_fall(B):
    """The bundle: the middle line, the envelope, and each tie strand's place in it."""
    L = (HAIR_LEN - B['rho0'][B['basin'] == 0].min() * MM) / MM          # the longest tail, mm
    # the middle line's bending length CALIBRATED so the arc rises as the photo's does (a tied, curled bundle is
    # stiffer than one hair: the report gives both)
    best = None
    for la in np.arange(30.0, LAUNCH_MAX_DEG + 1, 3.0):
        for ell in np.arange(30.0, 260.0, 5.0):
            s, back, up, th = centreline(L, la, ell)
            k = int(np.argmax(up))
            # (the arc's top, and how far behind the tie the tail hangs: MEASURED tail_behind_tie, 0.31 head)
            e = (up[k] * MM - PH['apex_rise']) ** 2 + (back[k] * MM - PH['apex_behind_tie']) ** 2 \
                + (back[int(0.6 * len(back))] * MM - PH['tail_behind_tie']) ** 2
            if best is None or e < best[0]:
                best = (e, ell, la, s, back, up, th)
    _, ell, la, s, back, up, th = best
    sR, R, status = envelope(L, B['r_tie'] / MM)
    # the S wave: every strand moved by the same amount, across the middle line, in the side plane. (The section
    # does not tilt with the wave: tilted, the strands inside each bend fold, since the wave bends tighter than the
    # bundle is wide)
    A = WAVE_A / MM * sm(0, WAVE_ON / MM, s)
    w = A * np.sin(2 * np.pi * s / (WAVE_LEN / MM))
    nb, nu = -np.sin(th), np.cos(th)                                      # the normal (back, up)
    base = B['C'][None] + np.c_[np.zeros_like(s), back, up] * MM
    cl = base + np.c_[np.zeros_like(s), w * nb, w * nu] * MM
    tang = unit(np.gradient(base, axis=0))
    e1 = unit(np.c_[np.zeros_like(s), -tang[:, 2], tang[:, 1]])           # in the side plane, across the bundle
    e2 = np.array([1.0, 0, 0])
    axis = np.array([0.0, math.cos(math.radians(la)), math.sin(math.radians(la))])
    return dict(L=L, ell=ell, launch=la, axis=axis, s=s, cl=cl, tang=tang, e1=e1, e2=e2, R=R, bvp=status, back=back, up=up)


def bundle_places(B):
    """Each tie strand's place in the bundle's section: the angle keeps the order of the bearings (spread by flux, so
    the section is evenly filled), the radius keeps the order of the stack (near roots inside, far roots outside)."""
    tie = np.where(B['basin'] == 0)[0]
    a = B['alpha'][tie]
    o = np.argsort(a)
    cdf = np.cumsum(B['hairs'][tie][o]) / B['hairs'][tie].sum()
    psi = np.zeros(len(tie)); psi[o] = 2 * np.pi * (cdf - 0.5)
    # anchor: the hair from straight above keeps its side (psi 0 at alpha 0)
    psi = psi - psi[np.argmin(np.abs(a))]
    rr = np.zeros(len(tie))
    for b in range(B['na']):
        k = np.where(B['ab'][tie] == b)[0]
        if len(k):
            kk = k[np.argsort(B['rho0'][tie][k])]
            c = np.cumsum(B['hairs'][tie][kk]) / B['hairs'][tie][kk].sum()
            rr[kk] = np.sqrt(c)                                          # uniform over the disc [GWB12]
    out_psi = np.full(len(B['P']), np.nan); out_r = np.full(len(B['P']), np.nan)
    out_psi[tie] = psi; out_r[tie] = rr
    return out_psi, out_r


def strand(B, FF, psi, rr, i, n_scalp=60, n_hand=24, step_mm=6.0):
    """One tie strand as a polyline: scalp, handover, bundle. Returns (points, part ids 0/1/2)."""
    X, rho = B['X'][i], B['rho'][i]
    ring = (B['r_tie'] / MM) + RING_MM                                       # mm: the handover starts this far out
    k1 = np.searchsorted(-rho, -ring)                                     # first node inside the ring
    k1 = max(min(k1, len(rho) - 2), 2)
    A = X[:k1]
    # the bundle side: a point s_h down the axis, at this strand's place
    s_h = HAND_AXIS_MM
    js = np.searchsorted(FF['s'], s_h)
    def place(j):
        return FF['cl'][j] + (FF['R'][j] * MM) * rr[i] * (math.cos(psi[i]) * FF['e1'][j] + math.sin(psi[i]) * FF['e2'])
    P0, P1 = A[-1], place(js)
    d0 = unit(A[-1] - A[-2]); d1 = unit(place(js + 1) - place(js))
    span = np.linalg.norm(P1 - P0)
    u = np.linspace(0, 1, n_hand)[:, None]
    # a cubic from the scalp's direction to the bundle's, matched in position and direction at both ends; of the
    # handle lengths tried, the one whose tightest bend is gentlest
    h0, h1, h2, h3 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    Hc, kbest = None, None
    for f0 in (0.6, 1.0, 1.5, 2.2):
        for f1 in (0.6, 1.0, 1.5, 2.2):
            H_ = h0 * P0 + h1 * (f0 * span * d0) + h2 * P1 + h3 * (f1 * span * d1)
            kv = curvature(np.r_[A[-2:-1], H_, [place(js + 1)]]).max()
            if kbest is None or kv < kbest:
                Hc, kbest = H_, kv
    # the tail, to this hair's own length
    tail_mm = HAIR_LEN / MM - B['rho0'][i] - s_h
    Hc = np.r_[Hc[:1], Hc[1:-1], Hc[-1:]]
    je = min(np.searchsorted(FF['s'], s_h + max(tail_mm, 5.0)), len(FF['s']) - 1)
    jj = np.arange(js, je + 1, max(int(step_mm / (FF['s'][1] - FF['s'][0])), 1))
    Tl = np.array([place(j) for j in jj])
    pts = np.r_[A, Hc[1:-1], Tl]
    part = np.r_[np.zeros(len(A), int), np.ones(len(Hc) - 2, int), np.full(len(Tl), 2)]
    return pts, part


def bang_line(B, i, k=40):
    """A fringe hair (design element): forward over the scalp to the hairline, then down the forehead to the edge,
    swept to its own side as the photo's is."""
    surf = B['surf']
    P = B['P'][i]
    sx = 1.0 if P[0] >= 0 else -1.0
    a0 = abs(float(SF.theta(P[None])[0]))
    # the middle of the fringe falls to the brow; its outer part is swept to the side, to the front of the ear's top
    # (MEASURED: the photo's bang_at_ear, 0.145 head in front of the ear canal at the ear top's height)
    wout = float(sm(0.3, BANG_TH, a0))
    th1 = sx * (1.25 * a0 + 0.05 + 0.3 * wout)
    z1 = BANG_EDGE_Z + (SF.EAR_TOP + 0.02 - BANG_EDGE_Z) * wout + 0.02 * math.cos(140 * P[0])
    end = SF.on_skin_at(surf, th1, z1)
    U = slerp(unit(P - SF.CENTRE)[None], unit(end - SF.CENTRE), np.linspace(0, 1, k))[0]
    S = surf.point(U)
    lift = (1.0 + 5.0 * np.sin(np.linspace(0, np.pi, k)) ** 2) * MM          # the dried root's lift (groom.py: 6 mm)
    return S + surf.normals(U) * lift[:, None]


def loose_line(B, i, length, k=50):
    """A loose lock's hair (design element): it leaves its root along the main flow (tangent matched at the peel),
    turns to plumb over a bending length, and hangs, kept off the head."""
    surf = B['surf']
    d = unit(B['X'][i, 3] - B['X'][i, 0])
    p = B['P'][i] + B['N'][i] * 1.0 * MM
    step = length / k
    out = [p]
    down = np.array([0, 0, -1.0])
    for j in range(k):
        w = 1 - math.exp(-(j * step / MM) / BEND_L_MM * 2.5)
        v = unit((1 - w) * d + w * down)
        p = p + v * step
        if p[2] > 0.2:
            dep = surf.depth(p[None])[0] + 2.0 * MM
            if dep > 0:
                p = p + unit(p - SF.CENTRE) * dep
        e = ear_depth(p[None])[0] + EAR_DELTA_MM * MM
        if e > 0:
            p = p + np.array([np.sign(p[0]), 0, 0]) * e
        out.append(p)
    return np.array(out)


# --------------------------------------------------------------------------------------------- the gauges
def curvature(pts):
    """Curvature per mm at the inner nodes of a polyline (HH in)."""
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    t1, t2 = unit(b - a), unit(c - b)
    ds = 0.5 * (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - b, axis=1)) / MM
    return np.linalg.norm(t2 - t1, axis=1) / np.maximum(ds, 1e-9)


def gauges(B, FF, psi, rr, lines):
    g = {}
    tie = B['basin'] == 0
    tot = B['hairs'].sum()
    g['hairs_total'] = round(float(tot))
    g['basins'] = {nm: dict(hairs=round(float(B['hairs'][B['basin'] == k].sum())),
                            share_pct=round(100 * float(B['hairs'][B['basin'] == k].sum() / tot), 2))
                   for k, nm in enumerate(B['names'])}
    loose = sum(v['share_pct'] for k, v in g['basins'].items() if k not in ('tie', 'bangs'))
    g['loose_fraction_pct'] = dict(value=round(loose, 2), bound=[100 * LOOSE_FRACTION[0], 100 * LOOSE_FRACTION[1]],
                                   ok=bool(LOOSE_FRACTION[0] * 100 <= loose <= LOOSE_FRACTION[1] * 100))
    n_tie = float(B['hairs'][tie].sum())
    # balance sheet: on each cut, the strands counted crossing it on the BUILT lines, against the roots beyond it
    sheet = []
    prev = None
    for r in CUTS_MM:
        beyond = float(B['hairs'][tie & (B['dist'][:, 0] > r)].sum())
        crossing = 0.0
        for i, (pts, part, rho_line) in lines.items():
            if B['dist'][i, 0] > r and (rho_line[:-1] > r).any() and (rho_line <= r).any():
                crossing += lines_w[i]
        row = dict(cut_mm=r, roots_beyond=round(beyond), crossing_built=round(crossing),
                   source_since_last=None if prev is None else round(beyond - prev))
        row['error_pct'] = round(100 * (crossing - beyond) / max(beyond, 1), 2)
        sheet.append(row); prev = beyond
    g['balance_sheet'] = sheet
    g['through_tie'] = dict(strands=round(n_tie), of_total_pct=round(100 * n_tie / tot, 1))
    # F: the floor inside the hairline (where the cut has roots beyond it), and where it is lowest
    Fm = np.where(B['reach'], B['Fmap'], np.nan)
    # (the outermost 7 mm is the hairline's own transition row, which hairline_gap.py gauges on the built hair)
    k = np.unravel_index(np.nanargmin(Fm), Fm.shape)
    low = np.argwhere(Fm < F_MIN)
    g['F_min'] = dict(value_per_mm=round(float(np.nanmin(Fm)), 2), floor=F_MIN, at_bearing_deg=round(math.degrees(
        (k[0] + 0.5) / B['na'] * 2 * np.pi - np.pi)), at_rho_mm=float(B['rho_ax'][k[1]]),
        cells_under_floor=int(len(low)), cells=int(np.isfinite(Fm).sum()))
    # the tie
    area = math.pi * (B['r_tie'] / MM) ** 2
    phi_tie = n_tie * HAIR_A / area
    w_min = 2 * math.sqrt(n_tie * HAIR_A / PHI_TIE_MAX / math.pi)
    g['tie'] = dict(width_mm=round(2 * B['r_tie'] / MM, 1), packing=round(phi_tie, 3), packing_max=PHI_TIE_MAX,
                    johns_range=[0.6, 0.7], narrowest_width_mm=round(w_min, 1),
                    width_at_0p65_mm=round(2 * math.sqrt(n_tie * HAIR_A / 0.65 / math.pi), 1),
                    check_100k_hairs_mm=round(2 * math.sqrt(1e5 * HAIR_A / 0.65 / math.pi), 1),
                    ok=bool(phi_tie <= PHI_TIE_MAX))
    # density away from the tie: packing of the stack at 40 mm out, against 0.7 of the ceiling
    # funnel: the flow front's width W where the handover starts against the tie's w
    ring = B['r_tie'] / MM + RING_MM
    W = 2 * ring
    need = (W - 2 * B['r_tie'] / MM) / (2 * math.tan(math.radians(THETA_MAX_DEG)))
    # the handover's length along the axis (18 mm) plus the rise off the skin
    g['funnel'] = dict(front_width_mm=round(W, 1), tie_width_mm=round(2 * B['r_tie'] / MM, 1),
                       min_length_mm=round(need, 1), theta_max_deg=THETA_MAX_DEG)
    # per built line: curvature, convergence angle, slack, handover mismatch
    kmax, conv, slack, mis_dir, mis_pos, where = [], [], [], [], [], []
    hand_len = []
    for i, (pts, part, rho_line) in lines.items():
        # curvature read at CURL_STEP_MM along the line (the skin is read to about a mm: finer than that the bend
        # between two nodes is the reading's noise, not the hair's)
        sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))] / MM
        st = np.arange(0, sl[-1], CURL_STEP_MM)
        if len(st) < 3:
            continue
        rs = np.c_[[np.interp(st, sl, pts[:, c]) for c in range(3)]].T
        kv = curvature(rs)
        j = int(np.argmax(kv)); kmax.append(kv[j])
        where.append(part[min(int(np.searchsorted(sl, st[j + 1])), len(part) - 1)])
        h = pts[part == 1]
        if len(h) > 2:
            t = unit(np.diff(pts, axis=0))
            ph = part[:-1]
            # the angle to the bundle's axis where the strand joins it
            conv.append(np.degrees(np.arccos(np.clip(t[ph == 1] @ FF['axis'], -1, 1))).max())
            a = np.where(part == 1)[0]
            hand_len.append(np.linalg.norm(np.diff(pts[a[0] - 1:a[-1] + 2], axis=0), axis=1).sum() / MM)
            # mismatch at both joins: the turn between the segments either side of each join
            for q in (a[0] - 1, a[-1] + 1):
                # (the turn across the join less the curve's own turn over one segment there)
                own = np.degrees(np.arccos(np.clip(t[q] @ t[min(q + 1, len(t) - 1)], -1, 1)))
                mis_dir.append(max(np.degrees(np.arccos(np.clip(t[q - 1] @ t[q], -1, 1))) - own, 0.0))
            # density handover: checked below on the sectors
        sc = pts[part == 0]
        arc = np.linalg.norm(np.diff(sc, axis=0), axis=1).sum()
        skin = np.linalg.norm(np.diff(B['Ss'][i][:len(sc)], axis=0), axis=1).sum()
        if skin > 30 * MM:
            slack.append(arc / max(skin, 1e-9) - 1)
    kmax = np.array(kmax); where = np.array(where)
    g['curl'] = dict(max_per_mm=round(float(kmax.max()), 4), K_per_mm=round(K_MAX, 4),
                     tightest_radius_mm=round(float(1 / kmax.max()), 1), p99_per_mm=round(float(np.percentile(kmax, 99)), 4),
                     where=['scalp', 'handover', 'bundle'][int(where[np.argmax(kmax)])],
                     over_K_pct=round(100 * float((kmax > K_MAX).mean()), 2), ok=bool(kmax.max() <= K_MAX))
    g['funnel']['handover_length_mm_median'] = round(float(np.median(hand_len)), 1)
    g['funnel']['ok'] = bool(np.median(hand_len) >= need)
    g['turn_into_bundle_deg'] = dict(median=round(float(np.median(conv)), 1), max=round(float(np.max(conv)), 1),
                                     note='the angle between a strand and the bundle axis during the handover; hair '
                                          'from the crown comes down onto the tie and turns furthest')
    slack = np.array(slack)
    g['slack'] = dict(median_pct=round(100 * float(np.median(slack)), 2), min_pct=round(100 * float(slack.min()), 2),
                      eps_pct=100 * SLACK_EPS, under_eps_pct=round(100 * float((slack < SLACK_EPS).mean()), 1),
                      ok=bool(np.median(slack) >= SLACK_EPS), status='DRAFT bound, no source')
    g['handover'] = dict(direction_mismatch_deg_max=round(float(np.max(mis_dir)), 2),
                         direction_mismatch_deg_median=round(float(np.median(mis_dir)), 2), position_gap_mm=0.0)
    # density across the handover: the flux in each sector of the bundle against an even share
    tie_i = np.where(tie)[0]
    sec = np.minimum(((psi[tie_i] % (2 * np.pi)) / (2 * np.pi) * 12).astype(int), 11)
    share = np.array([B['hairs'][tie_i][sec == k].sum() for k in range(12)]) / n_tie * 12
    g['handover']['sector_density_min_max'] = [round(float(share.min()), 3), round(float(share.max()), 3)]
    g['handover']['twist_deg_max'] = round(float(np.degrees(np.abs(np.angle(np.exp(1j * (psi[tie_i] - B['alpha'][tie_i]))))).max()), 1)
    # rank: neighbours in bearing at the 150..25 mm cuts, and round the bundle at three stations
    ids = np.array(sorted(lines.keys()))
    swaps = 0; pairs = 0
    stations = []
    for r in CUTS_MM:
        ang = {}
        for i in ids:
            pts, part, rho_line = lines[i]
            if B['dist'][i, 0] > r + 5:
                j = int(np.argmin(np.abs(np.where(rho_line >= 0, rho_line, 1e9) - r)))
                d = pts[j] - B['T']
                ang[i] = math.atan2(d @ B['e_sd'], d @ B['e_up'])
        stations.append(ang)
    for smm in (30.0, 150.0, 300.0):
        j = int(np.searchsorted(FF['s'], smm))
        ang = {}
        for i in ids:
            pts, part, rho_line = lines[i]
            tl = pts[part == 2]
            if len(tl) and HAIR_LEN / MM - B['rho0'][i] > smm + 10:
                q = tl[int(np.argmin(np.linalg.norm(tl - FF['cl'][j], axis=1)))] - FF['cl'][j]
                if np.linalg.norm(q) > 1.5 * MM:
                    ang[i] = math.atan2(q @ FF['e2'], q @ FF['e1'][j])
        stations.append(ang)
    for A_, B_ in zip(stations[:-1], stations[1:]):
        common = [i for i in A_ if i in B_]
        if len(common) < 3:
            continue
        oa = sorted(common, key=lambda i: A_[i])
        # cyclic order: the number of descents in B's angles read in A's order (one is the wrap itself)
        seq = np.unwrap(np.array([B_[i] for i in oa]))
        dsc = int((np.diff(seq) < -0.35).sum())
        swaps += dsc; pairs += len(common) - 1
    g['rank'] = dict(swaps=swaps, neighbour_pairs_checked=pairs, ok=bool(swaps == 0),
                     note='a swap: two neighbours change order by more than 20 deg of bearing between two stations')
    # monotonic progress: rho never rises along a built line
    back = sum(int((np.diff(rl[:(pt == 0).sum()]) > 0.5).any()) for (p_, pt, rl) in lines.values())
    g['monotonic'] = dict(lines_turning_back=back, lines=len(lines), ok=bool(back == 0))
    # against the grain: the turn from the follicle's own direction to the combed direction at the root [SP][FS11]
    d = unit(B['X'][:, 3] - B['X'][:, 0])
    turn = np.degrees(np.arccos(np.clip((d * unit(B['grow'])).sum(1), -1, 1)))
    w = B['hairs'] * tie
    g['root_turn_deg'] = dict(median=round(float(np.median(turn[tie])), 1),
                              over_120_pct=round(100 * float(w[turn > 120].sum() / w.sum()), 1),
                              note='pull at the root, read as the turn from the follicle to the comb: no force '
                                   'ceiling is published [SP][FS11], so this is reported, not bounded')
    # the ear
    oe = B['over_ear'][tie].any(1)
    g['ear'] = dict(flux_over_ear_pct=round(100 * float(B['hairs'][tie][oe].sum() / n_tie), 2), offset_mm=EAR_DELTA_MM,
                    K_per_mm=round(K_MAX, 3))
    # lift
    lm = B['lift_mm'][tie]
    g['lift_mm'] = dict(at_roots=0.5, median_at_60mm=round(float(np.median(lm[np.abs(B['rho'][tie] - 60) < 3])), 1),
                        p95_at_30mm=round(float(np.percentile(lm[np.abs(B['rho'][tie] - 30) < 3], 95)), 1),
                        note='from the stack (flux x hair section / packing), not a free profile')
    # free fall
    cl = FF['cl']
    g['free_fall'] = dict(tail_longest_heads=round(FF['L'] * MM, 2), bending_length_mm=FF['ell'], launch_deg=FF['launch'],
                          single_hair_bending_mm=BEND_L_MM, apex_rise_heads=round(float(FF['up'].max() * MM), 3),
                          envelope_max_width_heads=round(float(2 * FF['R'].max() * MM), 3),
                          envelope_solved=bool(FF['bvp'] == 0),
                          fall_deg=round(float(np.degrees(np.arctan2(cl[-1, 1] - cl[len(cl) // 3, 1],
                                                                     cl[len(cl) // 3, 2] - cl[-1, 2]))), 1))
    return g


lines_w = {}


# --------------------------------------------------------------------------------------------- the drawings
RES_COL = {1: '#2c9c5a', 2: '#7a4fc9'}
COL = {'tie': '#2f6fb3', 'bangs': '#d98a1f', 'temple lock': '#c2337a', 'ear wisp': '#2c9c5a', 'nape wisps': '#7a4fc9',
       'handover': '#111111', 'bundle': '#2f6fb3'}


def camera(az_deg, el_deg=0.0):
    a, e = math.radians(az_deg), math.radians(el_deg)
    fwd = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])   # from the head to the camera
    right = unit(np.cross(np.array([0, 0, 1.0]), fwd))
    up = np.cross(fwd, right)
    return right, up, fwd


def hidden(surf, P, fwd, steps=14, reach=1.2):
    """True where the head lies between the point and the camera."""
    t = np.linspace(0.02, reach, steps)[None, :, None]
    Q = (P[:, None] + fwd[None, None] * t).reshape(-1, 3)
    neck = (Q[:, 2] < 0.25) & (Q[:, 2] > -0.3) & (np.hypot(Q[:, 0], Q[:, 1] - 0.08) < 0.2)     # (a plain neck)
    ins = ((surf.depth(Q) > 0.004) & (Q[:, 2] > 0.2)) | (ear_depth(Q) > 0) | neck
    return ins.reshape(len(P), steps).any(1)


def draw_view(ax, B, FF, lines, extra, az, el, title, body=None):
    from matplotlib.collections import LineCollection
    surf = B['surf']
    right, up, fwd = camera(az, el)
    Pc = SF.load_head(HEAD)
    Pc = Pc[Pc[:, 2] > -0.25]
    # the clay head: the export's points, the nearest drawn last, lit from the camera's upper left
    c0 = Pc - SF.CENTRE
    nrm = unit(c0)
    dpt = Pc @ fwd
    o = np.argsort(dpt)
    L = unit(fwd + 0.5 * up - 0.4 * right)
    shade = np.clip(0.72 + 0.28 * (nrm @ L), 0.5, 1.0)
    vis = (nrm @ fwd) > -0.15
    ax.scatter((Pc @ right)[o][vis[o]], (Pc @ up)[o][vis[o]], c=np.c_[shade, shade, shade][o][vis[o]] * [0.86, 0.84, 0.82],
               s=16, linewidths=0, zorder=1)
    segs, cols, lws, zs = [], [], [], []

    def add(pts, col, lw):
        hid = hidden(surf, pts, fwd)
        x, y = pts @ right, pts @ up
        for j in range(len(pts) - 1):
            if not (hid[j] or hid[j + 1]):
                segs.append([(x[j], y[j]), (x[j + 1], y[j + 1])]); cols.append(col); lws.append(lw)
                zs.append(0.5 * (pts[j] + pts[j + 1]) @ fwd)
    for q, (i, (pts, part, rl)) in enumerate(lines.items()):
        for p_, nm in ((0, 'tie'), (1, 'handover'), (2, 'bundle')):
            k = np.where(part == p_)[0]
            if len(k) > 1 and (p_ == 0 or q % 4 == 0 or B['resolve'][i]):                   # (every 4th line through the tie and tail)
                k = np.r_[max(k[0] - 1, 0), k]
                rs_ = int(B['resolve'][i])
                if rs_ and p_ == 0:
                    add(pts[k], RES_COL[rs_], 1.3)
                else:
                    add(pts[k], COL[nm] if p_ != 2 else ('#2f6fb3', '#1d4a80', '#6fa3d8')[(q // 4) % 3], 0.55 if p_ != 1 else 0.6)
    for nm, pl in extra:
        for pts in pl:
            add(pts, COL[nm], 0.8 if nm == 'bangs' else 1.3)
    o = np.argsort(zs)
    ax.add_collection(LineCollection([segs[j] for j in o], colors=[cols[j] for j in o], linewidths=[lws[j] for j in o],
                                     zorder=3))
    # the tie
    th = np.linspace(0, 2 * np.pi, 60)
    ring = FF['cl'][0][None] + B['r_tie'] * (np.cos(th)[:, None] * FF['e1'][0] + np.sin(th)[:, None] * FF['e2'])
    ax.plot(ring @ right, ring @ up, color='#e02020', lw=2.0, zorder=5)
    ax.set_aspect('equal'); ax.set_xlim(-0.95, 1.25); ax.set_ylim(-1.55, 1.15); ax.axis('off')
    ax.set_title(title, fontsize=11)


def main():
    global HEAD
    a = sys.argv[1:]
    HEAD, out = a[0], a[1]
    os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    global CACHE
    CACHE = os.path.join(os.path.dirname(os.path.abspath(out)), 'cache')
    if '--no-cache' in a:
        CACHE = None
    else:
        os.makedirs(CACHE, exist_ok=True)
    import time, inspect
    t_all = time.time()
    B = build(HEAD)
    print('time: %-16s %.1f s' % ('field', time.time() - t_all), file=sys.stderr)
    FF = free_fall(B)
    psi, rr = bundle_places(B)
    rng = np.random.default_rng(3)
    tie = np.where(B['basin'] == 0)[0]
    # the drawn lines: an even sample of the tie basin by flux, each standing for an equal share
    pick = rng.choice(tie, size=1400, replace=False, p=B['hairs'][tie] / B['hairs'][tie].sum())
    for rs_ in (1, 2):                                          # (and 60 of each re-solved set, so they can be seen)
        rr_ = np.where(B['resolve'] == rs_)[0]
        pick = np.r_[pick, rng.choice(rr_, size=min(60, len(rr_)), replace=False)]
    pick = np.unique(pick)
    w_blue = B['hairs'][tie].sum() / len(pick)
    lines = {}
    for i in pick:
        pts, part = strand(B, FF, psi, rr, i)
        nsc = int((part == 0).sum())
        rl = np.r_[B['dist'][i][:nsc], np.full(len(pts) - nsc, -1.0)]
        lines[int(i)] = (pts, part, rl)
        cls = B['resolve'][pick] == B['resolve'][i]
        lines_w[int(i)] = B['hairs'][tie][B['resolve'][tie] == B['resolve'][i]].sum() / cls.sum()
    extra = []
    bi = np.where(B['basin'] == 1)[0]
    # stage: the fringe's lines (depends on the head, the fringe's constants and bang_line itself)
    bkey = (file_key(HEAD), BANG_TH, BANG_DEPTH_MM, BANG_EDGE_Z, len(bi), inspect.getsource(bang_line))
    bang_all = cached('bangs', bkey, lambda: [bang_line(B, i, k=32) for i in bi])
    extra.append(('bangs', [bang_all[j] for j in rng.choice(len(bi), size=min(260, len(bi)), replace=False)]))
    for k, (nm, c) in enumerate(LOOSE.items()):
        li = np.where(B['basin'] == 2 + k)[0]
        extra.append((nm, [loose_line(B, i, c['len'] * (0.8 + 0.2 * rng.random())) for i in
                           rng.choice(li, size=min(40, len(li)), replace=False)]))
    print('time: %-16s %.1f s' % ('lines', time.time() - t_all), file=sys.stderr)
    G = gauges(B, FF, psi, rr, lines)
    # the hairline, by hairline_gap.py's own coverage rule, on every root's line (not the drawn sample)
    import hairline_gap as HG
    ring = B['r_tie'] / MM + RING_MM
    paths = [B['X'][i][B['rho'][i] >= ring] for i in tie]
    w = list(B['hairs'][tie])
    paths += bang_all; w += list(B['hairs'][bi])
    for k, (nm, c) in enumerate(LOOSE.items()):
        li = np.where(B['basin'] == 2 + k)[0]
        paths += [loose_line(B, i, c['len'], k=30) for i in li]; w += list(B['hairs'][li])
    keep = [j for j, p in enumerate(paths) if len(p) >= 2]
    Hc, Wc = HG.measure(B['F'], B['surf'], [paths[j] for j in keep], np.array(w)[keep])
    hl = HG.report(Hc, Wc)
    HG.draw(Hc, Wc, os.path.join(out, 'hairline-gauge.png'))
    G['hairline'] = {k: dict(cells=v['cells'], bare=len(v['bare']), longest_gap_mm=v['longest_gap_mm'],
                             thinnest_per_mm=v['thinnest'], bare_at=v['bare']) for k, v in hl.items()}
    G['photo_ratios_heads'] = PH
    G['conditions'] = dict(tie_z=round(TIE_Z, 3), tie_width=TIE_W, tie_off_skin=TIE_OFF, launch_deg=FF['launch'],
                           launch_fibre_deg=LAUNCH_FIBRE_DEG, hair_length_heads=HAIR_LEN, wave_len=WAVE_LEN,
                           wave_amp=WAVE_A, bang_edge_z=round(BANG_EDGE_Z, 3), F_min=F_MIN, K_per_mm=round(K_MAX, 3),
                           theta_max_deg=THETA_MAX_DEG, slack_eps=SLACK_EPS)
    json.dump(G, open(os.path.join(out, 'gauges.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out, 'field.npz'), P=B['P'], basin=B['basin'], alpha=B['alpha'], rho0=B['rho0'],
                        hairs=B['hairs'], psi=psi, rr=rr, Fmap=B['Fmap'], rho_ax=B['rho_ax'], centre=FF['cl'], R=FF['R'],
                        tie=B['C'], axis=B['axis'])

    print('time: %-16s %.1f s' % ('gauges+hairline', time.time() - t_all), file=sys.stderr)
    # ---- the four views
    fig, axs = plt.subplots(1, 4, figsize=(26, 9.6), facecolor='white')
    for ax, (az, el, t) in zip(axs, ((90, 0, "side (the photo's camera)"), (0, 0, 'front'), (180, 0, 'back'),
                                     (45, 8, '3/4'))):
        draw_view(ax, B, FF, lines, extra, az, el, t)
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=COL[k], lw=3, label=l) for k, l in (
        ('tie', 'to the tie (funnel, then free fall)'), ('handover', 'handover into the bundle'),
        ('bangs', 'bangs (own outlet)'), ('temple lock', 'loose: temple lock'), ('ear wisp', 're-solved to the tie: ear roots'),
        ('nape wisps', 're-solved to the tie: nape roots'))] + [Line2D([0], [0], color='#e02020', lw=3, label='the tie')]
    fig.legend(handles=hs, loc='lower center', ncol=7, fontsize=11, frameon=False)
    fig.suptitle('Ponytail flow field (draft): streamlines only, colours mark design elements', fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.02)
    fig.savefig(os.path.join(out, 'flow-field-views.png'), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 12.5), facecolor='white')
    draw_view(ax, B, FF, lines, extra, 90, 0, "side (the photo's camera)")
    fig.savefig(os.path.join(out, 'flow-field-side.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    print('time: %-16s %.1f s' % ('views', time.time() - t_all), file=sys.stderr)
    # ---- the flat F map and the balance sheet
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 7.5), facecolor='white', gridspec_kw=dict(width_ratios=[1.5, 1]))
    Fm = np.where(B['reach'], B['Fmap'], np.nan)
    deg = (np.arange(B['na']) + 0.5) / B['na'] * 360 - 180
    im = a1.pcolormesh(deg, B['rho_ax'], np.log10(np.maximum(Fm.T, 0.1)), cmap='viridis', shading='nearest', vmin=0, vmax=3.3)
    cb = fig.colorbar(im, ax=a1); cb.set_label('F, hairs per mm of cut (log10)')
    lowm = np.where((Fm < F_MIN).T)
    a1.scatter(deg[lowm[1]], B['rho_ax'][lowm[0]], s=14, c='#e02020', label='under the floor (%g per mm)' % F_MIN)
    a1.set_xlabel('bearing at the tie, deg (0: from straight above; +-180: from the nape; + the left side)')
    a1.set_ylabel('distance from the tie along the skin, mm (top: the hairline)')
    a1.set_title('Flat map of F: the scalp unrolled round the tie. Grey: outside the hairline')
    a1.set_facecolor('#bbbbbb'); a1.legend(loc='upper right')
    for r in CUTS_MM:
        a1.axhline(r, color='w', lw=0.6, ls=':')
    a2.axis('off')
    rows = [['cut, mm from tie', 'roots beyond', 'counted crossing', 'error %', 'sourced since last cut']]
    for r in G['balance_sheet']:
        rows.append([r['cut_mm'], r['roots_beyond'], r['crossing_built'], r['error_pct'],
                     '' if r['source_since_last'] is None else r['source_since_last']])
    rows.append(['the tie', G['through_tie']['strands'], G['through_tie']['strands'], 0.0, ''])
    tb = a2.table(cellText=rows, loc='upper center', cellLoc='center'); tb.auto_set_font_size(False); tb.set_fontsize(10)
    tb.scale(1, 1.7)
    txt = 'Basins (hairs, share):\n' + '\n'.join('  %-12s %6d  %5.2f%%' % (k, v['hairs'], v['share_pct'])
                                                 for k, v in G['basins'].items())
    a2.text(0.02, 0.30, txt, family='monospace', fontsize=11, va='top', transform=a2.transAxes)
    a2.set_title('Balance sheet: flux through each cut against the roots beyond it')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'flow-F-map.png'), dpi=110)
    plt.close(fig)

    # ---- the overlay on the photo
    if '--photo' in a:
        sheet = json.load(open(a[a.index('--photo') + 1]))
        from PIL import Image
        img = Image.open(os.path.join(os.path.dirname(a[a.index('--photo') + 1]), sheet['image'])).convert('RGB')
        lm = sheet['landmarks']
        L_ = B['F']['landmarks']
        # fit: scale and shift (no turn: the photo's plumb is trusted) on chin, nose tip, skull top, skull back, tie
        Pm = SF.load_head(HEAD)
        chin = Pm[(np.abs(Pm[:, 0]) < 0.03) & (Pm[:, 2] < 0.05) & (Pm[:, 2] > -0.02)]
        chin = chin[np.argmin(chin[:, 1])]
        back = Pm[(np.abs(Pm[:, 0]) < 0.03) & (np.abs(Pm[:, 2] - 0.62) < 0.02)]
        back = back[np.argmax(back[:, 1])]
        model = {'chin': np.array([0, chin[1] + 0.06, 0.0]), 'nose_tip': L_['pronasale'], 'head_back': back}
        A_, b_ = [], []
        for k, p in model.items():
            A_ += [[p[1], 1, 0], [-p[2], 0, 1]]; b_ += [lm[k][0], lm[k][1]]
        sc, tx, ty = np.linalg.lstsq(np.array(A_), np.array(b_), rcond=None)[0]
        to_px = lambda P_: np.c_[sc * P_[:, 1] + tx, -sc * P_[:, 2] + ty]
        fig, ax = plt.subplots(figsize=(11, 14.6), facecolor='white')
        ax.imshow(img)
        fwd = camera(90)[2]
        for i, (pts, part, rl) in [kv for q_, kv in enumerate(lines.items()) if q_ % 2 == 0 or B['resolve'][kv[0]]]:
            if pts[0, 0] < -0.02:
                continue
            q = to_px(pts)
            if B['resolve'][i]:
                ks = part == 0
                ax.plot(q[ks, 0], q[ks, 1], color=RES_COL[int(B['resolve'][i])], lw=1.3)
            ax.plot(q[:, 0], q[:, 1], color='#35c8ff', lw=0.5, alpha=0.75)
        for nm, pl in extra:
            for pts in pl[::2]:
                if pts[0, 0] < -0.02:
                    continue
                q = to_px(pts); ax.plot(q[:, 0], q[:, 1], color=COL[nm], lw=0.9)
        # the bundle's outline
        for sg in (-1, 1):
            q = to_px(FF['cl'] + sg * (FF['R'] * MM)[:, None] * FF['e1'])
            ax.plot(q[:, 0], q[:, 1], color='yellow', lw=2)
        q = to_px(np.array([B['C']])); ax.plot(q[:, 0], q[:, 1], 'o', color='#e02020', ms=9)
        # the gap list: the same quantities read off the field, in heads of the photo's unit
        hp = sheet['computed']['unit_px'] if 'computed' in sheet else 620.0
        pv = lambda P_: to_px(np.atleast_2d(P_))[0]
        cl = FF['cl']
        mod = dict(
            tie_above_chin=(lm['chin'][1] - pv(B['C'])[1]) / hp,
            tie_behind_ear=(pv(B['C'])[0] - lm['ear_canal'][0]) / hp,
            tie_w=TIE_W * sc / hp,
            apex_rise=(pv(B['C'])[1] - to_px(cl)[:, 1].min()) / hp,
            pony_len_y=(to_px(cl)[-1, 1] - pv(B['C'])[1]) / hp,
            pony_max_w=2 * FF['R'].max() * MM * sc / hp + PH['wave_swing'],
            fall_deg=G['free_fall']['fall_deg'],
            tail_behind_tie=(to_px(cl)[int(0.6 * len(cl)), 0] - pv(B['C'])[0]) / hp,
            bang_edge_h=(lm['chin'][1] - pv(np.array([0, -0.4, BANG_EDGE_Z]))[1]) / hp,
            ear_top_to_tie_bottom_level=(pv(np.array([0.33, 0.04, SF.EAR_TOP]))[1]
                                         - pv(B['C'] - FF['e1'][0] * B['r_tie'])[1]) / hp)
        ref = dict(PH); ref['ear_top_to_tie_bottom_level'] = (lm['ear_top'][1] - lm['tie_bottom'][1]) / hp
        gaps = []
        for k, v in mod.items():
            gp = v - ref[k]
            flag = abs(gp) > (5.0 if k == "fall_deg" else 0.05)   # (the fall is read through the S wave: 5 deg)
            gaps.append(dict(name=k, photo=round(ref[k], 3), field=round(float(v), 3), gap=round(float(gp), 3),
                             flagged=bool(flag)))
        G['photo_gaps'] = gaps
        G['photo_fit'] = dict(px_per_head_model=round(float(sc), 1), px_per_head_photo=hp)
        ytxt = 1030
        ax.text(15, ytxt, 'field over the photo. blue: to the tie; yellow: the bundle envelope [GWB12]; red: the tie',
                color='w', fontsize=10, bbox=dict(facecolor='k', alpha=0.6))
        for gp in gaps:
            ytxt += 34
            ax.text(15, ytxt, '%-28s photo %6.3f  field %6.3f  gap %+6.3f %s' % (
                gp['name'], gp['photo'], gp['field'], gp['gap'], 'GAP' if gp['flagged'] else ''), color='w',
                fontsize=9, family='monospace', bbox=dict(facecolor='#a01010' if gp['flagged'] else 'k', alpha=0.6))
        ax.axis('off')
        fig.savefig(os.path.join(out, 'flow-over-photo.png'), dpi=110, bbox_inches='tight')
        plt.close(fig)
        json.dump(G, open(os.path.join(out, 'gauges.json'), 'w'), indent=1)
    print('time: %-16s %.1f s' % ('map+overlay', time.time() - t_all), file=sys.stderr)
    print(json.dumps(G, indent=1))
    print('total %.1f s' % (time.time() - t_all), file=sys.stderr)


if __name__ == '__main__':
    main()
