"""A notch collar on the turned lapels (John, 2026-11-22: "go back to the other lapel and add collar so it's 30 degrees
notch with new lapels").

    python3 blender/collar/foldfront.py CUT.blend LAPEL.blend '{"recipe": "notch_lapel"}'
    python3 blender/collar/notchcollar.py LAPEL.blend OUT.blend ['{"notch_angle": 30}']

The collar is a separate piece, as a tailor's is: sewn to the bodice's neckline round the back and to the top edge of
each lapel (the gorge) from the roll line out to the notch. Its own roll line runs on from the lapels' roll lines, up the
sides of the neck and round the back; below the roll is the stand (against the neck), above it the fall, turned over
and lying down over the stand and the shoulders. At the front the collar's end runs from the notch G (on the lapel's top
edge) out to the collar's point C, cut so that the notch opens 30 degrees between the collar's end and the lapel's edge
beyond G.

How it is laid. The stand is a band from the neckline up to the roll line, leaning in a little, kept off the neck. The
fall lies on whatever is under it (the stand, the bodice, the turned lapels), the cloth's thickness off it: every point
of it is found by a ray cast in toward one point inside the chest, below the neck (F), so the fall is a single sheet over
a chart of directions seen from F (azimuth round the neck, elevation) and can never fold through itself. (Walking lines
across the collar one by one, the first way tried, laid the back well and crumpled the fronts, where the lines ran off
the lapel's edge and crossed.) The outline is cut on that chart: the roll line, the gorge along each lapel's top edge out
to the notch G, the end G to the collar's point C, and the outer edge, whose depth below the roll is measured along the
cloth and rises from the back to each point. Near the roll the fall is lifted on a quarter round so that it turns over
instead of creasing.
"""
import bpy, bmesh, sys, json, math, numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import Delaunay, cKDTree
from matplotlib.path import Path

a = sys.argv
has = a[-1].lstrip().startswith('{')
opt = json.loads(a[-1]) if has else {}
blend, out = a[-3:-1] if has else a[-2:]
LP = np.load(blend + '.lapels.npz')

notch_angle = opt.get('notch_angle', 40.0)  # the notch's opening, degrees (swept 20 to 90: 30 to 50 clean, 40 chosen; 60 and up the end folds over)
notch_at = opt.get('notch_at', 0.5)         # the notch on the lapel's top edge: this share of the way from the roll line to the lapel's point
point_len = opt.get('point_len', 0.7)       # the collar's point (G to C) this share of the lapel's (G to P)
stand = opt.get('stand', 0.4)               # the stand at the back (1.0 cm: John, "press down the collar"; was 2.5, then 1.6)
fall_over = opt.get('fall_over', 0.6)       # the fall reaches this far past the stand at the back (it covers the seam)
stand_run = opt.get('stand_run', 3.2)       # the stand dies away over this length toward the gorge (2.2 creased the fall)
neck_clear = opt.get('neck_clear', 0.05)    # the stand's inside this far off the neck
lean = opt.get('lean', 0.35)                # the stand leans in toward the neck (share of its height)
roll_r = opt.get('roll_r', 0.08)           # the roll line's soft turn, radius (0: a hard fold between two pieces)
end_taper = opt.get('end_taper', 0.25)     # below this stand height the turn tapers onto the fall's roll point: the collar's ends closed (0: left open)
# the collar's roll run into the lapel's (John, 2026-11-23, for the high lapel: "Collar roll runs into the lapel"): at
# each roll top the collar's turn starts as wide as the lapel's own turn there (join_h, the stand's height at the roll
# top) and as round, so the two rolls meet at the gorge as one roll instead of the collar's standing off the lapel's
join_h = opt.get("join_h", 0.0)        # (0: off, the stand dies to nothing at each roll top as before)
join_run = opt.get('join_run', 1.0)       # (the turn is kept round over this length of roll line from each roll top)
if join_h > 0 and 'end_taper' not in opt: end_taper = 0.0     # (the collar's turn meets the lapel's open, as one roll: not closed onto the fall)
rise = opt.get('rise', 1.3)                 # the outer edge rises to each collar point over this much azimuth (radians)
gap = opt.get('gap', 0.04)
hm = opt.get('mesh', 0.06)
tip_r = opt.get('tip_round', 0.15)          # the collar's point rounded, radius (a share of G to C)
zF = opt.get('focus_z', 48.5)               # the point the fall is seen from (on the neck's axis)

bpy.ops.wm.open_mainfile(filepath=blend)
b = bpy.data.objects['Redingote bodice']
M = {k[7:]: b[k] for k in b.keys() if k.startswith('bodice_')}
yc = M['yc']
sol = [m for m in b.modifiers if m.type == 'SOLIDIFY'][0]
th = sol.thickness
mw = np.array(b.matrix_world)
W = np.array([v.co[:] for v in b.data.vertices]) @ mw[:3, :3].T + mw[:3, 3]
BP = [tuple(p.vertices) for p in b.data.polygons]
BT = BVHTree.FromPolygons([Vector(p) for p in W], BP)
sk = bpy.data.objects['Skin body']; sw = np.array(sk.matrix_world)
SV = np.array([v.co[:] for v in sk.data.vertices]) @ sw[:3, :3].T + sw[:3, 3]
SP = [tuple(p.vertices) for p in sk.data.polygons]
ST = BVHTree.FromPolygons([Vector(p) for p in SV], SP)
Z = np.array([0, 0, 1.0])
F0 = np.array([0.0, yc, zF])


def radial(p):
    r = np.array([p[0], p[1] - yc, 0.0]); return r / max(np.linalg.norm(r), 1e-9)


def push(p, layers):
    p = np.array(p, float)
    for tree, c in layers:
        loc, nor, _, dist = tree.find_nearest(Vector(p))
        if loc is None or dist >= c: continue
        d = p - np.array(loc)
        n = d / dist if dist > 1e-7 else np.array(nor)
        p = np.array(loc) + n * c
    return p


def smooth_curve(P, k, ends=True):
    Q = P.copy()
    for _ in range(k):
        Q[1:-1] = 0.25 * Q[:-2] + 0.5 * Q[1:-1] + 0.25 * Q[2:]
    return Q


def resample(P, ds):
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    s = np.linspace(0, L[-1], int(np.ceil(L[-1] / ds)) + 1)
    return np.column_stack([np.interp(s, L, P[:, k]) for k in range(P.shape[1])])


def ss(a_, b_, x):
    t = np.clip((np.asarray(x, float) - a_) / (b_ - a_), 0, 1); return t * t * (3 - 2 * t)


# 1. the neckline round the back: the bodice's top edge from one lapel's roll line to the other's
bm = bmesh.new(); bm.from_mesh(b.data); bm.verts.ensure_lookup_table()
bd = np.array([v.is_boundary for v in bm.verts]); bm.free()
lap_all = np.vstack([LP['pts1'], LP['pts2'], LP['crease1'], LP['crease2']])
near_lap = cKDTree(lap_all).query(W)[0] < 0.08
rr_ = np.hypot(W[:, 0], W[:, 1] - yc)
psiW = np.mod(np.arctan2(W[:, 0], -(W[:, 1] - yc)), 2 * np.pi)   # 0 at the front, round her left (+x) to pi at the back
NA, NB = LP['crease1'][0], LP['crease2'][0]           # the roll lines' tops: front 1 is her right (x < 0)
pA = np.mod(np.arctan2(NA[0], -(NA[1] - yc)), 2 * np.pi); pB = np.mod(np.arctan2(NB[0], -(NB[1] - yc)), 2 * np.pi)
sel = bd & ~near_lap & (W[:, 2] > M['z_neck_front']) & (rr_ < 4.0) & (psiW > pB + 0.02) & (psiW < pA - 0.02)
bins = np.linspace(pB, pA, 64)
E = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = sel & (psiW >= lo) & (psiW < hi)
    if m.any(): E.append(W[np.nonzero(m)[0][np.argmax(W[m, 2])]])
E = np.vstack([NB, np.array(E), NA])                   # from her left's roll top (B) round the back to her right's (A)
E = smooth_curve(E, 6); E[0], E[-1] = NB, NA
E = resample(E, 0.04)
eL = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(E, axis=0), axis=1))])
d_end = np.minimum(eL, eL[-1] - eL)
st = stand * ss(0.0, stand_run, d_end)
# the stand drafted from the neckline (John: "think about how the shape changes as a function of the neckline"): the
# collar's roll line is the lapel's roll line carried on round the neck, so at each lapel's roll top N the stand rises
# off the neckline at the slope that keeps the roll straight on from the lapel's crease, and it reaches its full
# height at the side neck point, where the neckline turns from the front to the back
psE = np.mod(np.arctan2(E[:, 0], -(E[:, 1] - yc)), 2 * np.pi)
ks = {'B': int(np.argmin(np.abs(psE - 0.5 * np.pi))), 'A': int(np.argmin(np.abs(psE - 1.5 * np.pi)))}
upE = np.array([(Z - lean * radial(e)) / np.linalg.norm(Z - lean * radial(e)) for e in E])
def _rise(k0, k1, crease):
    # from E[k0] (the roll top N) toward E[k1] (the side neck point): the slope a there, and the cubic h(s)
    sgn = 1 if k1 > k0 else -1
    tE = E[k0 + sgn] - E[k0]; tE /= np.linalg.norm(tE)
    c_ = crease[min(8, len(crease) - 1)] - crease[0]; c_ = -c_ / np.linalg.norm(c_)     # (the crease carried on up past N)
    S = abs(eL[k1] - eL[k0])
    best = (9, 0.0)
    for a_ in np.linspace(0, 4.0, 400):
        v = tE + a_ * upE[k0]; ang = math.degrees(math.acos(np.clip(v @ c_ / np.linalg.norm(v), -1, 1)))
        if ang < best[0] or best[0] == 9: best = (ang, a_) if ang < best[0] or best[0] == 9 else best
    a_ = best[1] * opt.get('rise_gain', 1.0)      # (rise_gain: the stand rises off the roll top this much steeper than the crease asks)
    # (a rise with that slope at N that levels out smoothly to the full stand by the side neck point)
    ss_ = np.abs(eL - eL[k0]); k_ = max(a_ / stand, 3.0 / S)
    h = stand * (1 - np.exp(-k_ * ss_)) / (1 - np.exp(-k_ * S)); h = np.where(ss_ <= S, np.minimum(h, stand), stand)
    if join_h > 0: h = join_h + (stand - join_h) * h / stand      # (the roll joined to the lapel's: the stand starts at the lapel's turn, not at nothing)
    ang0 = math.degrees(math.acos(np.clip(tE @ c_, -1, 1)))
    return h, best[0], ang0, a_, S
if opt.get('from_neck', True):
    hB, angB, angB0, aB, SB = _rise(0, ks['B'], LP['crease2'])
    hA, angA, angA0, aA, SA = _rise(len(E) - 1, ks['A'], LP['crease1'])
    st = np.minimum(hA, hB)
    print('notchcollar: the stand rises off the roll tops at slope %.2f and %.2f' % (aA, aB))
    print('notchcollar: the stand drafted from the neckline: it rises off each roll top at the lapel crease\'s slope (the roll turns %.0f and %.0f degrees off the crease, was %.0f and %.0f), full at the side neck points %.1f and %.1f cm round'
          % (angA, angB, angA0, angB0, 2.62 * SA, 2.62 * SB))
# the stand's own shape round the neck (Mueller & Sohn, women's lapel collar: the back stand "approx. 0.5 cm more than
# at the shoulder"): full at the centre back, stand_shoulder of it from each side neck point on toward the front
stand_sh = opt.get('stand_shoulder', 1.0)
if stand_sh != 1.0:
    psE_ = np.mod(np.arctan2(E[:, 0], -(E[:, 1] - yc)), 2 * np.pi)
    w_ = np.clip(np.abs(psE_ - np.pi) / (0.5 * np.pi), 0, 1)          # 0 at the back, 1 at the side neck points
    st = np.minimum(st, stand * (1 - (1 - stand_sh) * ss(0.0, 1.0, w_)))     # (the rise off each roll top kept)
    print('notchcollar: the stand %.2f cm at the back, %.2f cm at the side neck points' % (2.62 * stand, 2.62 * stand * stand_sh))
print('notchcollar: the neckline round the back %.1f cm, from z %.2f at the roll lines to %.2f at the back' % (2.62 * eL[-1], NA[2], E[:, 2].max()))

# 2. the stand: from the neckline up to the roll line, leaning in, off the neck. Its top is the roll line.
nS = 12
up = np.array([(Z - lean * radial(e)) / np.linalg.norm(Z - lean * radial(e)) for e in E])
Rr = E + st[:, None] * up
# the ring (John: "the ring of the collar, reduce that length"): the roll is drawn in toward the neck until it lies
# hug off it, as much as the stand stands there (where the stand dies away at the fronts it keeps to the neckline)
hug = opt.get('hug')            # (off: the bodice's neckline is now drafted to the collar, bodice_cut.neck_fit; was 0.15)
E0 = E.copy()
wh = ss(0.0, opt.get('hug_run', 1.0) * stand_run, d_end)      # (drawn in gently, over twice the stand's own run, so the fall finds no step)
if hug and opt.get('hug_base', True):     # (the collar's neck edge too: the collar is cut shorter round the neck and sewn
    for k in range(len(E)):               #  on inside the bodice's neckline, which its fall covers)
        q = np.array(ST.find_nearest(Vector(E[k]))[0]); dv = E[k] - q; dv[2] = 0.0; dh = np.linalg.norm(dv)
        if dh > hug: E[k] = E[k] - wh[k] * (1 - hug / dh) * dv
    E = smooth_curve(E, 30); E[0], E[-1] = E0[0], E0[-1]
    Rr = E + st[:, None] * up
if hug:
    for k in range(len(Rr)):
        q = np.array(ST.find_nearest(Vector(Rr[k]))[0]); dv = Rr[k] - q; dv[2] = 0.0; dh = np.linalg.norm(dv)
        if dh > hug: Rr[k] = Rr[k] - wh[k] * (1 - hug / dh) * dv
Rr = smooth_curve(Rr, 20); Rr[0], Rr[-1] = E[0], E[-1]
SX = np.zeros((len(E), nS + 1, 3))
for k in range(len(E)):
    for j in range(nS + 1):
        SX[k, j] = push(E[k] + (j / nS) * (Rr[k] - E[k]), [(ST, neck_clear + th)])     # (the stand's thickness lies toward the neck)
for _ in range(opt.get('stand_smooth', 25)):                # (pushed point by point off the neck: evened out round the neck)
    SX[1:-1] = 0.25 * SX[:-2] + 0.5 * SX[1:-1] + 0.25 * SX[2:]
    SX[:, 1:-1] = 0.75 * SX[:, 1:-1] + 0.125 * (SX[:, :-2] + SX[:, 2:])
    SX[:, 0] = E
    for k in range(len(E)):
        for j in range(1, nS + 1): SX[k, j] = push(SX[k, j], [(ST, neck_clear + th - 0.01)])
Rr = SX[:, -1].copy()
if opt.get('from_neck', True):
    for nm, k0, sg, cr in (('A', len(Rr) - 1, -1, LP['crease1']), ('B', 0, 1, LP['crease2'])):
        j_ = next(j for j in range(len(Rr)) if np.linalg.norm(Rr[k0 + sg * j] - Rr[k0]) > 0.3)
        v_ = Rr[k0 + sg * j_] - Rr[k0]; c_ = cr[0] - cr[min(8, len(cr) - 1)]
        print('notchcollar: the roll as laid turns %.0f degrees off lapel %s\'s crease at its top' % (math.degrees(math.acos(np.clip(v_ @ c_ / np.linalg.norm(v_) / np.linalg.norm(c_), -1, 1))), nm))
print('notchcollar: the roll round the neck %.1f cm (the neckline %.1f cm), %.2f cm off the neck at most, %.2f on average'
      % (2.62 * np.linalg.norm(np.diff(Rr, axis=0), axis=1).sum(), 2.62 * np.linalg.norm(np.diff(E, axis=0), axis=1).sum(),
         2.62 * max(ST.find_nearest(Vector(p))[3] for p in Rr), 2.62 * np.mean([ST.find_nearest(Vector(p))[3] for p in Rr])))
keep_s = st > 0.03
sq = []
ids = np.arange(len(E) * (nS + 1)).reshape(len(E), nS + 1)
for k in range(len(E) - 1):
    if not (keep_s[k] and keep_s[k + 1]): continue
    for j in range(nS):
        sq.append((ids[k, j], ids[k + 1, j], ids[k + 1, j + 1], ids[k, j + 1]))
SXf = SX.reshape(-1, 3)
# (for what the fall lies on, the stand runs on a little below the neckline, so no ray slips between the two)
SK = np.array([[push(E[k] + (j / 3.0) * ((E0[k] - 0.35 * up[k]) - E[k]), [(ST, 0.06)]) for j in range(4)] for k in range(len(E))])   # (under the bodice's own edge, so it makes no ledge; with the ring drawn in it bridges back out to the bodice's neckline)
skq = [(len(SXf) + k * 4 + j, len(SXf) + (k + 1) * 4 + j, len(SXf) + (k + 1) * 4 + j + 1, len(SXf) + k * 4 + j + 1)
       for k in range(len(E) - 1) if keep_s[k] and keep_s[k + 1] for j in range(3)]
print('notchcollar: the stand %.1f cm at the back, dying away over %.1f cm toward each gorge; %d faces' % (2.62 * stand, 2.62 * stand_run, len(sq)))

# 3. what the fall lies on: the stand, the bodice (turned lapels and all) and the skin, each with its clearance
GV = np.vstack([W, SV, SXf, SK.reshape(-1, 3)]); n1, n2 = len(W), len(W) + len(SV)
GP = BP + [tuple(i + n1 for i in p) for p in SP] + [tuple(i + n2 for i in q) for q in sq + skq]
clr = np.concatenate([np.full(len(BP), th + gap), np.full(len(SP), 0.10), np.full(len(sq) + len(skq), th + gap)])
GT = BVHTree.FromPolygons([Vector(p) for p in GV], GP)


def dirn(ps, be):
    return np.array([math.sin(ps) * math.cos(be), -math.cos(ps) * math.cos(be), math.sin(be)])


def ground(ps, be):
    d = dirn(ps, be)
    h = GT.ray_cast(Vector(F0 + 9.0 * d), Vector(-d), 9.5)
    if h[0] is None: return None
    n = np.array(h[1]); n = n if n @ d > 0 else -n
    return np.array(h[0]) + clr[h[2]] * n


def chart(p):
    d = np.asarray(p, float) - F0
    return np.array([np.mod(math.atan2(d[0], -d[1]), 2 * np.pi), math.asin(d[2] / np.linalg.norm(d))])


# the roll line on the chart (azimuth increasing from her left's roll top B round the back to her right's A)
Rc = np.array([chart(p) for p in Rr])
assert np.all(np.diff(Rc[:, 0]) > 0), 'the roll line must run round the neck one way'
betaR = lambda ps: np.interp(ps, Rc[:, 0], Rc[:, 1])
Rof = lambda ps: np.column_stack([np.interp(ps, Rc[:, 0], Rr[:, k]) for k in range(3)])

# 4. the notch on each lapel: G on the lapel's top edge, C turned off the edge beyond G by the notch's angle
def notch(L):
    pts, t, d, e = LP['pts%d' % L], LP['t%d' % L], LP['d%d' % L], LP['edge%d' % L]
    kP = np.argmax(np.where(e, d, -1))
    m = e & (t <= t[kP] + 1e-6); top3 = np.vstack([LP['crease%d' % L][0], pts[m][np.argsort(t[m])]])
    four = ('inner%d' % L) in LP.files       # a four-sided lapel (foldfront's outline_inner): its top edge runs level along
    if four:                                 # the neckline from the roll top to the corner, then the end edge down to the point
        mi = LP['inner%d' % L] & ~m; top_ = pts[mi][np.argsort(d[mi])]; end_ = pts[m][np.argsort(t[m])]
        top3 = np.vstack([LP['crease%d' % L][0], top_, end_])
        Lc_ = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(top3, axis=0), axis=1))])
        f_corner = Lc_[len(top_)] / Lc_[-1]  # (where along the top line the corner lies)
    top3 = smooth_curve(resample(top3, 0.04), 3) if not four else resample(top3, 0.04)
    Lt = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(top3, axis=0), axis=1))])
    kG = int(np.searchsorted(Lt, (f_corner if four and 'notch_at' not in opt else notch_at) * Lt[-1]))
    if four: kG = min(kG, len(top3) - 2)
    G3, P3 = top3[kG], top3[-1]
    nG = np.array(BT.find_nearest(Vector(G3))[1]); nG = nG if nG @ radial(G3) > 0 or nG[1] < 0 else -nG
    u = P3 - G3; u -= nG * (u @ nG); lu = np.linalg.norm(u); u /= lu
    inner = pts[np.argmin(np.abs(t - t[kP]) + np.abs(d - 0.3 * d[kP]))]     # a point inside the lapel, below its top edge
    best = None
    for sg in (1, -1):
        a_ = math.radians(notch_angle) * sg
        v = u * math.cos(a_) + np.cross(nG, u) * math.sin(a_)
        if (v - (v @ u) * u) @ (inner - G3) < 0: best = v
    C3 = G3 + point_len * lu * best
    lift = lambda p: (p + (th + gap) * nG)
    print('notchcollar: front %d: gorge %.1f cm from the roll line to the notch, the lapel\'s point %.1f cm past it, the collar\'s %.1f cm'
          % (L, 2.62 * Lt[kG], 2.62 * lu, 2.62 * point_len * lu))
    print('notchcollar: front %d: notch G at (%.3f, %.3f, %.3f), collar point C at (%.3f, %.3f, %.3f), lapel point P at (%.3f, %.3f, %.3f)' % (L, *G3, *C3, *P3))
    return dict(G3=lift(G3), P3=lift(P3), C3=C3, gorge=np.array([lift(p) for p in top3[:kG + 1]]), nG=nG)
NtA = notch(1); NtB = notch(2)
for Nt in (NtA, NtB):
    Nt['Gc'], Nt['Cc'] = chart(Nt['G3']), chart(Nt['C3'])
    Nt['gc'] = np.array([chart(p) for p in Nt['gorge']])
    seg = np.linspace(0, 1, 30)[:, None]
    Nt['end'] = np.array([chart(p) for p in Nt['G3'] + seg * (Nt['C3'] - Nt['G3'])])

# 5. the fall's chart. Each meridian (a line of constant azimuth seen from F) is walked down from the collar's top,
# and its length along the cloth is the chart's second coordinate: the chart is (azimuth, length down the cloth), so
# the mesh drawn on it comes out even on the cloth, steep over the stand as well as flat on the shoulders. The top is
# the roll line, and past each roll line's end the gorge.
Fb = stand + fall_over
dB = opt.get('meridian_step', 0.003)
topc = [Rc]; top3 = [Rr]
for Nt, NN in ((NtA, NA), (NtB, NB)):
    pN = chart(NN)[0]
    beyond = (Nt['gc'][:, 0] > Rc[-1, 0] + 1e-4) if Nt is NtA else (Nt['gc'][:, 0] < Rc[0, 0] - 1e-4)
    if beyond.any():
        topc.append(Nt['gc'][beyond]); top3.append(Nt['gorge'][beyond])
topc = np.vstack(topc); top3 = np.vstack(top3); o_ = np.argsort(topc[:, 0]); topc, top3 = topc[o_], top3[o_]
k_ = np.concatenate([[True], np.diff(topc[:, 0]) > 1e-5]); topc, top3 = topc[k_], top3[k_]
dpsi = opt.get('meridian_gap', 0.005)
psis = np.arange(topc[0, 0], topc[-1, 0] + 1e-9, dpsi)
vmax = opt.get('vmax', 4.6)
# what the fall lies on, as seen from F: the distance out along each direction, on a grid, with the crevices closed (the
# gap behind a lapel's roll, the slit at the lapel's edge: cloth bridges them, a ray does not) and evened out
from scipy import ndimage
bgrid = np.arange(-0.8, 1.35, dB)
Hraw = np.full((len(psis), len(bgrid)), np.nan)
for k, ps in enumerate(psis):
    for j, be in enumerate(bgrid):
        q = ground(ps, be)
        if q is not None: Hraw[k, j] = np.linalg.norm(q - F0)
Hf = np.where(np.isfinite(Hraw), Hraw, np.nanmin(Hraw))
cl = opt.get('close', 9)
Hc = ndimage.grey_closing(Hf, size=(cl, cl))
Hs = np.maximum(ndimage.gaussian_filter(Hc, opt.get('h_sigma', 1.5)), Hf)
# (only under the roll: above it lies the gap between the stand and the neck, which must stay open)
bR_ = np.interp(psis, Rc[:, 0], Rc[:, 1], left=np.inf, right=np.inf)
Ec0 = np.array([chart(p) for p in E]); sfp = np.interp(psis, Ec0[:, 0], st / stand, left=0.0, right=0.0)
below = (bgrid[None, :] < (bR_[:, None] - opt.get('close_from', 0.06))) & (sfp[:, None] < opt.get('close_where', 0.5))   # (toward the fronts only, where the collar crosses the lapels)
Hs = np.where(below, Hs, Hf)
ground_ray = ground
def ground(ps, be):
    k = (ps - psis[0]) / dpsi; j = (be - bgrid[0]) / dB
    if not (0 <= k <= len(psis) - 1 and 0 <= j <= len(bgrid) - 1): return None
    if not below[int(round(k)), int(round(j))]: return ground_ray(ps, be)     # (where nothing was closed: the ray itself)
    r = ndimage.map_coordinates(Hs, [[k], [j]], order=1)[0]
    return F0 + r * dirn(ps, be)
print('notchcollar: what the fall lies on, %d x %d directions, crevices closed' % Hs.shape)
MER = []
for ps in psis:
    be = float(np.interp(ps, topc[:, 0], topc[:, 1])); p = np.array([np.interp(ps, topc[:, 0], top3[:, c_]) for c_ in range(3)])
    q0 = ground(ps, be - 1e-4)
    if q0 is not None and np.linalg.norm(q0 - p) < 0.3: p = q0          # (start on the cloth under the top, unless that is far off)
    bs, vs, Ps = [be], [0.0], [p]; acc = 0.0
    while be > -0.7 and acc < vmax:
        be -= dB; q = ground(ps, be)
        if q is None: continue
        acc += np.linalg.norm(q - p); p = q
        bs.append(be); vs.append(acc); Ps.append(q)
    MER.append((np.array(bs), np.array(vs), np.array(Ps)))
MER0 = None
# the meridians are walked one by one and the cloth under them has steps (a lapel's edge, its roll): the walked sheet is
# put on a regular grid (azimuth, length down) and evened out across the meridians and along them, kept off everything
MERW = list(MER)
VG = np.arange(0.0, vmax, 0.02)
GR = np.array([[np.interp(VG, vs, Ps[:, c_]) for c_ in range(3)] for _, vs, Ps in MER]).transpose(0, 2, 1)
def push_n(p):
    loc, nor, fi, dist = GT.find_nearest(Vector(p))
    if loc is None or dist >= clr[fi]: return p
    d = p - np.array(loc); n = d / dist if dist > 1e-7 else np.array(nor)
    return np.array(loc) + n * clr[fi]
def push_h(p):
    # (out along the line from F only, onto what the fall lies on: a nearest-point push near a sharp edge, the lapel's
    # roll or the stand's dying end, sends neighbours different ways and crumples the cloth)
    c = chart(p); q = ground(c[0], c[1])
    if q is None: return push_n(p)
    r = np.linalg.norm(q - F0); d = p - F0
    return p if np.linalg.norm(d) >= r else F0 + r * d / np.linalg.norm(d)
push_g = push_h if opt.get('push', 'h') == 'h' else push_n
for it in range(opt.get('grid_smooth', 10)):
    G_ = GR.copy()
    G_[1:-1] = 0.25 * GR[:-2] + 0.5 * GR[1:-1] + 0.25 * GR[2:]
    G_[:, 1:-1] = 0.5 * G_[:, 1:-1] + 0.25 * (G_[:, :-2] + G_[:, 2:])
    G_[:, 0] = GR[:, 0]
    GR = G_
    if it % 2 == 1 or it == opt.get('grid_smooth', 10) - 1:
        for k in range(GR.shape[0]):
            for j in range(1, GR.shape[1]):
                GR[k, j] = push_g(GR[k, j])
MER = [(None, VG, GR[k]) for k in range(GR.shape[0])]
print('notchcollar: %d meridians walked down the cloth, evened out on a %d x %d grid' % (len(MER), GR.shape[0], GR.shape[1]))
def _mer(ps):
    f = np.clip((ps - psis[0]) / dpsi, 0, len(psis) - 1.0001); k = int(f); return k, f - k
def to_v(ps, be):
    k, a_ = _mer(ps); r = []
    for kk in (k, k + 1):
        bs, vs, _ = MERW[kk]; r.append(np.interp(-be, -bs, vs))
    return (1 - a_) * r[0] + a_ * r[1]
def lay(ps, v):
    k, a_ = _mer(ps); r = []
    for kk in (k, k + 1):
        _, vs, Ps = MER[kk]; r.append(np.array([np.interp(v, vs, Ps[:, c_]) for c_ in range(3)]))
    return (1 - a_) * r[0] + a_ * r[1]
def chv(p):
    c = chart(p); return np.array([c[0], to_v(c[0], c[1])])
for Nt in (NtA, NtB):
    # the gorge on the chart: its azimuth as seen from F, its length down measured along the seam itself (what of each
    # step along it is not across the meridians is down them). Looked up through the walked meridians instead, the seam
    # (which runs nearly along a meridian, on the lapel's steep edge) jumps up and down the chart and the cloth folds
    g3 = Nt['gorge']; gps = np.unwrap(np.array([chart(p)[0] for p in g3]))
    gps = smooth_curve(np.column_stack([gps, np.zeros_like(gps)]), 4)[:, 0]
    Rh = np.linalg.norm((g3 - F0)[:, :2], axis=1)
    ds = np.linalg.norm(np.diff(g3, axis=0), axis=1); dps = np.diff(gps) * 0.5 * (Rh[1:] + Rh[:-1])
    gvv = np.concatenate([[0.0], np.cumsum(np.sqrt(np.maximum(ds ** 2 - dps ** 2, (0.2 * ds) ** 2)))])
    Nt['gv'] = np.column_stack([np.mod(gps, 2 * np.pi), gvv]); Nt['Gv'] = Nt['gv'][-1].copy()
    # the collar's end: the lapel's edge carried on past G (the seam's own direction there) turned by the notch's angle
    # toward the back, measured on the chart scaled to lengths at G, so the notch opens the same on the cloth both sides
    RG = Rh[-1]; k0 = max(0, len(g3) - 4)
    # the collar's end as cut (G to C, turned off the lapel's edge by the notch's angle on the cloth, step 4), put on the
    # chart the same way as the seam: across the meridians and down them
    gc_ = Nt['C3'] - g3[-1]; lgc = np.linalg.norm(gc_)
    dpC = (np.mod(chart(Nt['C3'])[0] - gps[-1] + np.pi, 2 * np.pi) - np.pi) * RG
    dzC = math.sqrt(max(lgc ** 2 - dpC ** 2, (0.2 * lgc) ** 2)) * (1.0 if gc_[2] <= 0 else -1.0)
    Nt['Cv'] = Nt['Gv'] + np.array([dpC / RG, dzC])
    u_e = np.linspace(0, 1, 30)[:, None]
    Nt['ev'] = Nt['Gv'] + u_e * (Nt['Cv'] - Nt['Gv'])
    eb = opt.get('end_bow', 0.0)        # the collar's end G to C bowed by this share of its length (positive: toward the fall's depth; negative hollows it toward the gorge)
    if eb:
        dv_ = (Nt['Cv'] - Nt['Gv']) * np.array([3.0, 1.0]); nv_ = np.array([-dv_[1], dv_[0]]) / np.linalg.norm(dv_)
        nv_ = nv_ if nv_[1] > 0 else -nv_                        # (toward the fall's depth: away from the gorge's side)
        Nt['ev'] = Nt['ev'] + (eb * np.linalg.norm(dv_) * 4 * u_e * (1 - u_e)) * (nv_ / np.array([3.0, 1.0]))[None]
FcA, FcB = NtA['Cv'][1], NtB['Cv'][1]
fall_side = opt.get('fall_side', 0.0)    # the fall this much deeper (or shallower) at the side neck, faded to nothing at the back and the points
def Flen(ps):
    base = Fb + (FcA - Fb) * ss(NtA['Cv'][0] - rise, NtA['Cv'][0], ps) + (FcB - Fb) * ss(NtB['Cv'][0] + rise, NtB['Cv'][0], ps)
    if not fall_side: return base
    uA = np.clip((ps - np.pi) / (NtA['Cv'][0] - np.pi), 0, 1); uB = np.clip((np.pi - ps) / (np.pi - NtB['Cv'][0]), 0, 1)
    u_ = np.maximum(uA, uB)
    return base + fall_side * np.sin(np.pi * u_) ** 2
pss = np.linspace(NtB['Cv'][0], NtA['Cv'][0], 200)
outer = np.column_stack([pss, Flen(pss)])
print('notchcollar: the fall %.1f cm deep at the back, %.1f and %.1f cm at the points (along the cloth from the roll)'
      % (2.62 * Fb, 2.62 * FcA, 2.62 * FcB))

# 6. the outline on the chart (azimuth scaled to about a length), the points rounded, meshed
SC = np.array([3.0, 1.0])
def fillet(Pa, Pc, Pb, r, n=9):
    ua = (Pa - Pc) / np.linalg.norm(Pa - Pc); ub = (Pb - Pc) / np.linalg.norm(Pb - Pc)
    q0, q1 = Pc + r * ua, Pc + r * ub; u_ = np.linspace(0, 1, n)[:, None]
    return (1 - u_) ** 2 * q0 + 2 * u_ * (1 - u_) * Pc + u_ ** 2 * q1
roll_c = np.column_stack([Rc[:, 0], np.zeros(len(Rc))]) * SC
gB = NtB['gv'] * SC; gA = NtA['gv'] * SC
eB = NtB['ev'] * SC; eA = NtA['ev'] * SC
oc = outer * SC
rB = tip_r * np.linalg.norm(eB[-1] - eB[0]); rA = tip_r * np.linalg.norm(eA[-1] - eA[0])
poly = np.vstack([roll_c, gA[1:], eA[1:-2], fillet(eA[-3], eA[-1], oc[-4], rA), oc[-4:3:-1],
                  fillet(oc[3], eB[-1], eB[-3], rB), eB[-3:0:-1], gB[::-1][:-1]])
path = Path(poly)
def densify(P, h):
    o_ = []
    for p0, p1 in zip(P, np.vstack([P[1:], P[:1]])):
        n_ = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / h)))
        o_ += [p0 + (p1 - p0) * i / n_ for i in range(n_)]
    o_ = np.array(o_); keep = [0]
    for i in range(1, len(o_)):
        if np.linalg.norm(o_[i] - o_[keep[-1]]) > 0.6 * h: keep.append(i)
    return o_[keep]
bnd = densify(poly, hm)
gx, gy = np.meshgrid(np.arange(poly[:, 0].min(), poly[:, 0].max(), hm), np.arange(poly[:, 1].min(), poly[:, 1].max(), hm * 0.866))
gx = gx + (np.arange(gx.shape[0])[:, None] % 2) * 0.5 * hm
G2 = np.column_stack([gx.ravel(), gy.ravel()])
G2 = G2[path.contains_points(G2)]
G2 = G2[cKDTree(bnd).query(G2)[0] > 0.6 * hm]
UV = np.vstack([bnd, G2]); nb_ = len(bnd)
tri = Delaunay(UV).simplices
tri = tri[path.contains_points(UV[tri].mean(1))]
A2 = UV[tri]; ar = 0.5 * np.abs((A2[:, 1, 0] - A2[:, 0, 0]) * (A2[:, 2, 1] - A2[:, 0, 1]) - (A2[:, 2, 0] - A2[:, 0, 0]) * (A2[:, 1, 1] - A2[:, 0, 1]))
tri = tri[ar > 0.03 * hm * hm]
used = np.unique(tri); rm = -np.ones(len(UV), int); rm[used] = np.arange(len(used)); UV = UV[used]; tri = rm[tri]
onb = np.zeros(len(UV), bool); onb[rm[np.arange(nb_)][rm[np.arange(nb_)] >= 0]] = True

# 7. laid: every point of the chart carried to its place on its meridians
ch = UV / SC
P3 = np.array([lay(ps, v) for ps, v in ch])
on_roll = onb & (ch[:, 1] < 0.3 * hm) & (ch[:, 0] >= Rc[0, 0]) & (ch[:, 0] <= Rc[-1, 0])
P3[on_roll] += np.array([0.5 * (th + gap) * radial(p) + 0.03 * Z for p in P3[on_roll]])   # (the fall's top edge stands just over and outside the stand's)
# the gorge is sewn to the lapel's top edge: its points sit on that edge (lifted the cloth's thickness)
on_gorge = np.zeros(len(UV), bool)
for Nt in (NtA, NtB):
    gd = resample(np.column_stack([Nt['gv'] * SC, Nt['gorge']]), 0.01)
    dg, ig = cKDTree(gd[:, :2]).query(UV)
    m = onb & (dg < 0.3 * hm) & ~on_roll
    if opt.get('gorge_fix', True): P3[m] = gd[ig[m], 2:]
    on_gorge |= m
# a light smoothing, kept off everything under it; boundary points move along the boundary only
nb = [set() for _ in range(len(UV))]
for a_, b_, c_ in tri:
    nb[a_] |= {b_, c_}; nb[b_] |= {a_, c_}; nb[c_] |= {a_, b_}
bnb = [[j for j in n_ if onb[j]] if onb[i] else list(n_) for i, n_ in enumerate(nb)]
# the collar's end, from the notch to where its point begins to round, is laid straight on the cloth as cut, so the notch
# opens on the cloth the angle it was cut to
on_end = np.zeros(len(UV), bool)
for Nt in (NtA, NtB):
    evs = Nt['ev'] * SC; de, ie = cKDTree(evs).query(UV)
    m = onb & (de < 0.4 * hm) & (ie <= int(0.8 * (len(evs) - 1))) & ~on_gorge & ~on_roll
    for i in np.nonzero(m)[0]:
        t_ = ie[i] / (len(evs) - 1.0); P3[i] = push_g(Nt['G3'] + t_ * (Nt['C3'] - Nt['G3']))
    on_end |= m; Nt['end_idx'] = np.nonzero(m)[0]; Nt['end_t'] = ie[m] / (len(evs) - 1.0)
fixed = on_roll | on_gorge | on_end
def push_n(p):
    loc, nor, fi, dist = GT.find_nearest(Vector(p))
    if loc is None or dist >= clr[fi]: return p
    d = p - np.array(loc); n = d / dist if dist > 1e-7 else np.array(nor)
    return np.array(loc) + n * clr[fi]
def push_h(p):
    # (out along the line from F only, onto what the fall lies on: a nearest-point push near a sharp edge, the lapel's
    # roll or the stand's dying end, sends neighbours different ways and crumples the cloth)
    c = chart(p); q = ground(c[0], c[1])
    if q is None: return push_n(p)
    r = np.linalg.norm(q - F0); d = p - F0
    return p if np.linalg.norm(d) >= r else F0 + r * d / np.linalg.norm(d)
push_g = push_h if opt.get('push', 'h') == 'h' else push_n
for _ in range(opt.get('smooth', 6)):
    A_ = np.array([P3[n_].mean(0) if n_ else P3[i] for i, n_ in enumerate(bnb)])
    for i in np.nonzero(~fixed)[0]:
        P3[i] = push_g(P3[i] + 0.5 * (A_[i] - P3[i]))
# toward the fronts, where the collar's ends cross the lapels' rolls, what the fall lies on is steep and stepped, seen
# from F: there the fall is let hang as a membrane, smoothed hard and only ever pushed out, so it bridges the steps
sfv = np.interp(ch[:, 0], psis, sfp)
zone = np.nonzero(~fixed)[0]
wz = (opt.get('back_hang', 0.6) + (1 - opt.get('back_hang', 0.6)) * ss(0.95, 0.4, sfv[zone]))[:, None]   # (all over, hardest at the fronts)
from scipy.sparse import coo_matrix
rr = np.concatenate([[i] * len(n_) for i, n_ in enumerate(bnb)]).astype(int); cc = np.concatenate([list(n_) for n_ in bnb]).astype(int)
Wm = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(UV), len(UV))).tocsr()
deg = np.maximum(np.asarray(Wm.sum(1)).ravel(), 1)
# (next to the gorge the cloth is sewn to the lapel's edge, a step seen from F: it is not pushed there, it hangs from the seam)
gall = np.vstack([resample(Nt['gv'] * SC, 0.01) for Nt in (NtA, NtB)])
dgz = cKDTree(gall).query(UV[zone])[0]
zone_p = zone[dgz > opt.get('seam_free', 0.3)]
gz_i = np.nonzero(on_gorge)[0] if opt.get('seam_slide', True) else np.zeros(0, int)
g3s = [(g_, cKDTree(g_)) for g_ in (resample(Nt['gorge'], 0.005) for Nt in (NtA, NtB))]
# the roll edge is evened out too, along itself, most toward its ends: there it meets the seam to the lapel, which
# sits at the lapel's edge, not over the stand, and a fixed roll made a step (and a nick in the fall's top) at the join
ri = np.nonzero(on_roll)[0]
dre = np.minimum(ch[ri, 0] - Rc[0, 0], Rc[-1, 0] - ch[ri, 0])
wr = (opt.get('roll_even', 0.15) + (1 - opt.get('roll_even', 0.15)) * ss(opt.get('roll_free', 0.45), 0.0, dre))[:, None]
for it in range(opt.get('front_smooth', 250)):
    A_ = (Wm @ P3) / deg[:, None]
    P3[zone] += 0.6 * wz * (A_[zone] - P3[zone])
    P3[ri] += 0.6 * wr * (A_[ri] - P3[ri])
    if len(gz_i):     # (the seam to the lapel slides along the lapel's edge, evening its points out, and stays on it)
        P3[gz_i] += 0.6 * (A_[gz_i] - P3[gz_i])
        for g3d_, g3t_ in g3s:      # (in order along the seam, evenly spaced: two points on one spot make a spike)
            dd, ii = g3t_.query(P3[gz_i]); mine = gz_i[dd < 0.05]
            if len(mine) < 2: continue
            si = g3t_.query(P3[mine])[1]; o_ = np.argsort(si, kind='stable')
            P3[mine[o_]] = g3d_[np.round(np.linspace(si.min(), si.max(), len(mine))).astype(int)]
    if it % 3 == 2 or it == opt.get('front_smooth', 250) - 1:
        for i in zone_p: P3[i] = push_g(P3[i])
        for i in ri: P3[i] = push_n(P3[i])
# a last polish: what the line from F cannot see (the bodice's own neckline edge at the sides of the neck, seen edge on,
# under the fall) still props the cloth up from the side and creases it; the fall is evened once more and kept off
# everything by its nearest point too, so it rounds over that edge instead of folding on it
for it in range(opt.get('polish', 40)):
    A_ = (Wm @ P3) / deg[:, None]
    P3[zone] += 0.5 * (A_[zone] - P3[zone])
    if it % 2 == 1 or it == opt.get('polish', 40) - 1:
        for i in zone_p: P3[i] = push_n(push_g(P3[i]))
# pressed (John: "press down the collar"): the fall is drawn in onto what it lies on, its ends onto the lapels, its
# back onto the stand and shoulders, then evened again and kept off everything
def gap_of(p):
    c = chart(p); q = ground(c[0], c[1])
    return None if q is None else np.linalg.norm(p - F0) - np.linalg.norm(q - F0)
gz0 = np.array([g_ for g_ in (gap_of(P3[i]) for i in zone_p) if g_ is not None])
print('notchcollar: before pressing, the fall floats %.2f cm over what it lies on on average, %.2f at most (95%%)' % (2.62 * gz0.mean(), 2.62 * np.percentile(gz0, 95)))
press = opt.get('press', 0.0)
if press:
    # (drawn in by an evened-out share of the gap, not onto the ground point by point: that copies every step under it)
    wpr = ss(0.0, opt.get('press_from', 0.8), ch[:, 1])     # (not at the roll: the fall turns over the stand there)
    for it in range(opt.get('press_rounds', 4)):
        g = np.zeros(len(UV)); ok = np.zeros(len(UV), bool)
        for i in zone_p:
            v_ = gap_of(P3[i])
            if v_ is not None: g[i] = max(v_, 0.0); ok[i] = True
        for _ in range(60):
            gs = (Wm @ g) / deg; g = np.where(ok, 0.5 * g + 0.5 * gs, gs)
        for i in zone_p:
            d = P3[i] - F0; r = np.linalg.norm(d); P3[i] = F0 + (r - press * wpr[i] * g[i]) * d / r
        for k_ in range(10):
            A_ = (Wm @ P3) / deg[:, None]
            P3[zone] += 0.5 * (A_[zone] - P3[zone])
            if k_ % 2 == 1:
                for i in zone_p: P3[i] = push_n(push_g(P3[i]))
    gz1 = np.array([g_ for g_ in (gap_of(P3[i]) for i in zone_p) if g_ is not None])
    print('notchcollar: pressed, the fall floats %.2f cm on average, %.2f at most (95%%)' % (2.62 * gz1.mean(), 2.62 * np.percentile(gz1, 95)))
# settled onto the lapels (John: "the left collar needs more adjustment to its position to compensate for the folding of
# the lapel, it is too far up"): the fall is laid on a ground with its crevices closed and hung as a membrane between
# its edges, so over the folded lapels its middle and its points hovered up to 6 mm above the cloth they should lie on.
# Each point over the bodice is drawn down toward its own nearest point of it by an evened share of how far it hovers
# past the clearance, then evened and kept off everything by the nearest point (not by the closed ground).
settle = opt.get('settle', 0.9)
if settle:
    nBP = len(BP); zs = np.union1d(zone_p, np.nonzero(on_end)[0])      # (the collar's ends too: laid straight from the notch to the point, they hovered most)
    wst = ss(0.0, opt.get('settle_from', 0.5), ch[:, 1])       # (not at the roll: the fall turns over the stand there)
    h0 = []
    for it in range(opt.get('settle_rounds', 4)):
        g = np.zeros(len(UV)); N_ = np.zeros((len(UV), 3))
        for i in zs:
            loc, nor, fi, dist = GT.find_nearest(Vector(P3[i]))
            if loc is None or fi >= nBP or dist > 0.8 or dist < 1e-6: continue
            g[i] = max(dist - clr[fi], 0.0); N_[i] = (P3[i] - np.array(loc)) / dist
        if it == 0: h0 = g[zs][g[zs] > 0]
        D_ = (settle * wst * g)[:, None] * N_           # (the move itself evened, not only its size: neighbours go the same way, so the edges stay fair)
        for _ in range(opt.get('settle_even', 40)):
            D_ = 0.5 * D_ + 0.5 * (Wm @ D_) / deg[:, None]
        mv = np.zeros(len(UV), bool); mv[zs] = True; mv &= ~(on_roll | on_gorge)
        P3[mv] -= D_[mv]
        for Nt in (NtA, NtB):                          # (each end laid straight again, from the notch to its lowered point)
            ix, tt = Nt['end_idx'], Nt['end_t']
            if len(ix) > 2:
                o_ = np.argsort(tt); a0, a1 = P3[ix[o_[0]]].copy(), P3[ix[o_[-1]]].copy(); u_ = (tt[o_] - tt[o_[0]]) / max(tt[o_[-1]] - tt[o_[0]], 1e-9)
                for q_, uu in zip(ix[o_], u_): P3[q_] = push_n(a0 + uu * (a1 - a0))
        for k_ in range(8):
            A_ = (Wm @ P3) / deg[:, None]
            P3[zone] += 0.5 * (A_[zone] - P3[zone])
            for i in zone_p: P3[i] = push_n(P3[i])
    h1 = []
    for i in zs:
        loc, nor, fi, dist = GT.find_nearest(Vector(P3[i]))
        if loc is not None and fi < nBP and dist < 0.8: h1.append(max(dist - clr[fi], 0.0))
    print('notchcollar: settled onto the lapels: the fall hovered %.1f mm over them on average, %.1f at most; now %.1f, %.1f at most'
          % (26.2 * np.mean(h0), 26.2 * np.max(h0), 26.2 * np.mean(h1), 26.2 * np.max(h1)))
# a last smoothing round over the pressed fall (John: "add a smoothing round to the collar"), kept off everything
for it in range(opt.get('final_smooth', 30) if (press or settle) else 0):
    A_ = (Wm @ P3) / deg[:, None]
    P3[zone] += 0.5 * (A_[zone] - P3[zone])
    if it % 2 == 1 or it == opt.get('final_smooth', 30) - 1:
        for i in zone_p: P3[i] = push_n(P3[i]) if settle else push_n(push_g(P3[i]))
if settle:
    # the fall's free edge faired along itself: where the collar's point crosses the lapel's edge the ground under it
    # steps down a cloth's thickness, and the edge, kept off it point by point, took the step as small bumps
    from collections import defaultdict as _dd
    ec_ = _dd(int)
    for t in tri:
        for a_, b_ in ((0, 1), (1, 2), (2, 0)): ec_[tuple(sorted((int(t[a_]), int(t[b_]))))] += 1
    bn_ = _dd(list)
    for (a_, b_), c_ in ec_.items():
        if c_ == 1: bn_[a_].append(b_); bn_[b_].append(a_)
    fb_ = np.array([i for i in bn_ if len(bn_[i]) == 2 and not fixed[i]])
    for _ in range(opt.get('edge_fair', 12)):
        A_ = np.array([0.5 * (P3[bn_[i][0]] + P3[bn_[i][1]]) for i in fb_])
        P3[fb_] += 0.5 * (A_ - P3[fb_])
    for _ in range(opt.get('edge_in', 20)):           # (and the cloth just inside it brought along)
        A_ = (Wm @ P3) / deg[:, None]; keep_ = P3[fb_].copy()
        P3[zone] += 0.5 * (A_[zone] - P3[zone]); P3[fb_] = keep_
print('notchcollar: the fall hung as a membrane over %d points toward the fronts' % len(zone))

# 8. what was built, measured
ed = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
le = np.linalg.norm(P3[ed[:, 0]] - P3[ed[:, 1]], axis=1); lc = np.linalg.norm(UV[ed[:, 0]] - UV[ed[:, 1]], axis=1)
print('notchcollar: %d points, %d faces; edges %.3f mean, %.3f at most on the cloth (%.3f on the chart); strain against the chart mean |%.1f%%|, 95%% under |%.1f%%|'
      % (len(UV), len(tri), le.mean(), le.max(), lc.mean(), 100 * np.abs(le / lc - 1).mean(), 100 * np.percentile(np.abs(le / lc - 1), 95)))
for L, Nt in ((1, NtA), (2, NtB)):
    u1, u2 = Nt['C3'] - Nt['G3'], Nt['P3'] - Nt['G3']
    # (as laid: the cloth's own points on the end, from G a third of the way to C, the point itself being rounded)
    evs = Nt['ev'] * SC; de, ie = cKDTree(evs).query(UV); me = onb & (de < 0.5 * hm) & (ie > 3) & (ie < 12)
    gi = np.argmin(np.linalg.norm(UV - Nt['Gv'] * SC, axis=1))
    u1b = P3[me].mean(0) - P3[gi] if me.any() else u1
    u2 = Nt['P3'] - P3[gi]
    ang = lambda x, y: math.degrees(math.acos(np.clip(x @ y / np.linalg.norm(x) / np.linalg.norm(y), -1, 1)))
    print('notchcollar: front %d: the notch opens %.1f degrees as cut, %.1f as laid (asked %.0f)' % (L, ang(u1, u2), ang(u1b, u2), notch_angle))
dmin = min(BT.find_nearest(Vector(p))[3] for p in P3[~on_gorge]); smin = min(ST.find_nearest(Vector(p))[3] for p in P3)
print('notchcollar: the fall at least %.3f off the bodice (the cloth is %.3f), %.3f off the skin' % (dmin, th, smin))

# the fall's corners on the roll tops evened (the lapels cross, one over the other, so the two corners lie on different
# ground: the over lapel's side kept a small fold there that the under side did not): the fall's inner points within
# end_even of each roll top relaxed toward their neighbours, the edges left where they are
end_even = opt.get('end_even', 0.6)
if end_even > 0:
    from collections import defaultdict
    ec_ = defaultdict(int); nb_ = defaultdict(set)
    for t in tri:
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            ec_[tuple(sorted((int(t[a_]), int(t[b_]))))] += 1; nb_[int(t[a_])].add(int(t[b_])); nb_[int(t[b_])].add(int(t[a_]))
    onb_ = np.zeros(len(P3), bool)
    for (a_, b_), c_ in ec_.items():
        if c_ == 1: onb_[a_] = onb_[b_] = True
    ir_ = np.nonzero(on_roll)[0]; ir_ = ir_[np.argsort(ch[ir_, 0])]
    for F_ in (P3[ir_[0]].copy(), P3[ir_[-1]].copy()):
        dd_ = np.linalg.norm(P3 - F_, axis=1); idx_ = np.nonzero((dd_ < end_even) & ~onb_)[0]
        w_ = 0.5 * (1 - ss(0.5 * end_even, end_even, dd_[idx_]))
        for _ in range(opt.get('end_even_rounds', 30)):
            avg_ = np.array([P3[list(nb_[int(q)])].mean(axis=0) for q in idx_])
            P3[idx_] += w_[:, None] * (avg_ - P3[idx_])
        print('notchcollar: the fall\'s corner at (%.2f, %.2f, %.2f) evened over %d points' % (F_[0], F_[1], F_[2], len(idx_)))

# 9. the collar: the stand and the fall as one piece, in the bodice's wool, the same thickness. The stand is hung from
# the fall's own points on the roll line (so the two are one cloth there, no seam and no ragged edge) down to the neckline
Ec = np.array([chart(p) for p in E])
ir = np.nonzero(on_roll)[0]; ir = ir[np.argsort(ch[ir, 0])]
sv, sf_ = [], []
nS2 = 10; nA = 7; lines = []; HH = []; thin = {}
for i in ir:
    ps = ch[i, 0]; e = np.array([np.interp(ps, Ec[:, 0], E[:, k]) for k in range(3)]); h_ = np.interp(ps, Ec[:, 0], st)
    if h_ < opt.get('stand_min', -1.0 if end_taper > 0 else 0.03): lines.append(None); continue
    n0_ = len(sv)
    ln_ = []; fk = np.interp(ps, Ec[:, 0], np.arange(len(E))); k0 = min(int(fk), len(E) - 2); a_ = fk - k0
    col = (1 - a_) * SX[k0] + a_ * SX[k0 + 1]                       # (the stand as laid, at this point of the roll)
    for j in range(nS2):
        ln_.append(len(P3) + len(sv)); sv.append(np.array([np.interp(j / nS2, np.linspace(0, 1, nS + 1), col[:, c_]) for c_ in range(3)]))
    S_ = np.array([np.interp(0.97, np.linspace(0, 1, nS + 1), col[:, c_]) for c_ in range(3)])
    ln_.append(len(P3) + len(sv)); sv.append(S_)
    if roll_r > 0:
        # the roll line pressed into a soft roll, not a fold (tailors: the roll line "should be pressed into shape
        # rather than stitched flat"; the top collar is cut larger by the turn of the cloth so it rolls over the
        # under collar): the stand turns over into the fall on a half round of at least roll_r, one cloth, with the
        # cloth's thickness on the outside of the turn (toward the neck on the stand, outward on the fall)
        F_ = P3[i]; Cc = 0.5 * (S_ + F_); hh = 0.5 * (S_ - F_); HH.append(2 * np.linalg.norm(hh))
        u_ = Z - lean * radial(S_); u_ = u_ - (u_ @ hh) / max(hh @ hh, 1e-12) * hh; u_ /= np.linalg.norm(u_)
        bb = roll_r * min(1.0, h_ / stand) ** 2     # (a flat half round over however wide the turn is; fading out fast where the stand dies away toward the lapels, so no lump or curl at its end)
        if join_h > 0:     # (near each roll top the turn is a true half round, as the lapel's roll is, fading to the collar's own flat turn)
            dN_ = min(abs(ps - Ec[0, 0]), abs(Ec[-1, 0] - ps)) * np.linalg.norm(S_[:2] - np.array([0.0, yc]))
            bb = max(bb, np.linalg.norm(hh) * (1 - ss(0.0, join_run, dN_)))
        for t_ in np.linspace(0, 1, nA + 2)[1:-1]:
            ln_.append(len(P3) + len(sv)); sv.append(Cc + math.cos(math.pi * t_) * hh + math.sin(math.pi * t_) * bb * u_)
        ln_.append(int(i))
    if end_taper > 0 and roll_r > 0 and h_ < end_taper:
        # the collar's ends closed: where the stand dies away beside each lapel the turn is drawn in onto the fall's
        # own roll point, so the roll tapers to nothing at the fall's corner on the gorge and no open end shows
        t_ = 0.12 + 0.88 * ss(0.0, end_taper, h_)       # (not quite to a point: a point pinches the cloth's thickness)
        for q_ in range(n0_, len(sv)): sv[q_] = P3[i] + t_ * (sv[q_] - P3[i])
        for q_ in range(n0_, len(sv)): thin[len(P3) + q_] = t_      # (and the cloth thinned with it: a turn tighter than the cloth is thick would cut through itself)
    lines.append(ln_)        # (without roll_r the stand stops just under the fall's top edge: two pieces meeting in a hard fold)
for l0_, l1_ in zip(lines[:-1], lines[1:]):
    if l0_ is None or l1_ is None: continue
    for j in range(min(len(l0_), len(l1_)) - 1):
        sf_.append((l0_[j], l1_[j], l1_[j + 1], l0_[j + 1]))
if HH: print('notchcollar: the roll turned over a soft half round of %.2f; the turn %.2f wide on average, %.2f at most' % (roll_r, np.mean(HH), np.max(HH)))
verts = np.vstack([P3, np.array(sv).reshape(-1, 3)])
faces = [tuple(int(i) for i in t) for t in tri] + sf_
sq = sf_
me = bpy.data.meshes.new('Redingote collar'); me.from_pydata([tuple(p) for p in verts], [], faces); me.update()
# one cloth, one winding: across the roll the stand's outside turns into the fall's underside, so the faces are wound
# alike over the whole piece and turned so the fall faces out (its thickness lies outward, the stand's toward the neck)
cbm = bmesh.new(); cbm.from_mesh(me); cbm.faces.ensure_lookup_table()
bmesh.ops.recalc_face_normals(cbm, faces=cbm.faces[:]); cbm.normal_update()
vote = sum(np.sign(np.array(f.normal) @ dirn(*chart(np.array(f.calc_center_median())))) for f in cbm.faces[:len(tri)])
if vote < 0: bmesh.ops.reverse_faces(cbm, faces=cbm.faces[:len(tri)])
if roll_r <= 0:       # (one cloth over the roll: the stand's side follows the fall's through the turn)
    vote = sum(np.sign(np.array(f.normal) @ -radial(np.array(f.calc_center_median()))) for f in cbm.faces[len(tri):])
    if vote < 0: bmesh.ops.reverse_faces(cbm, faces=cbm.faces[len(tri):])      # (the stand's thickness toward the neck)
cbm.to_mesh(me); cbm.free()
for p in me.polygons: p.use_smooth = True
old = bpy.data.objects.get('Redingote collar')
if old: bpy.data.objects.remove(old, do_unlink=True)
o = bpy.data.objects.new('Redingote collar', me); bpy.context.scene.collection.objects.link(o)
for m in b.data.materials: me.materials.append(m)
vg_ = o.vertex_groups.new(name='thick'); vg_.add(list(range(len(verts))), 1.0, 'REPLACE')
for q_, t_ in thin.items(): vg_.add([int(q_)], float(t_), 'REPLACE')
s2 = o.modifiers.new('Solidify', 'SOLIDIFY'); s2.thickness = th; s2.offset = sol.offset; s2.use_rim = True; s2.use_even_offset = False; s2.vertex_group = 'thick'; s2.thickness_vertex_group = 0.1
if b.parent: o.parent = b.parent; o.matrix_parent_inverse = b.parent.matrix_world.inverted()
np.savez(out + '.collar.npz', uv=UV, pts=P3, tri=tri)
bpy.ops.wm.save_as_mainfile(filepath=out)
print('notchcollar: saved', out)
