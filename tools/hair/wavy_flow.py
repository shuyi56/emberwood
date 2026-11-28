"""The Warden's long wavy hair, rebuilt with the new hair generation (the ponytail and bangs work, PR #97), as a Face
Creator groom asset.

    python3 tools/hair/wavy_flow.py HEAD.json.gz GROOM_DIR CREATOR_STYLE.json.gz OUT_DIR [--strands 16000]
            [--fibres 24000] [--noise 1.0] [--no-mid-layer] [--rest-gain G]

Writes OUT_DIR/asset-wavyFull.json (the page's strands asset: tools/face-creator/apply-hair-fidelity.py
wavyFull=...), OUT_DIR/fibres.npz (every fibre, nan-padded, head heights, with its share of the hair: the input of
creator_truth.py --fibres and of the gauges) and OUT_DIR/wavy.json (every gauge).

What is new against the old Warden asset (creator_asset.py from groom.py's guides):

1. On the head the hair is the approved groom's (GROOM_DIR, gp2_w: the part, the curtain bangs combed along the
   creator's own strands, the sweep behind the ear, the side piece), grown as fibres from every part of the scalp by
   the scalp map's density (render.grow_children).
2. Below HANDOVER_Z the hair hangs in wave locks, and each lock is solved in 3D as one Kirchhoff rod with its own
   natural wave (tools/solvers/hair/tail_lock_rods.py, the ponytail tail's solver): bending, gravity, lock against
   lock, the head, the ears and her collar, cape and back. The hairs ride their lock (each keeps its place across
   it), the lock draws in a little and then opens into sub-locks as it falls, and a hair passes from its head line to
   its lock over JOIN_MM with a quintic ease (G2 at both ends). The old build waved each guide on its own after five
   geometric passes; here the wave is the lock's rest shape and gravity takes out what it takes out.
3. The wave is the creator's (ground truth, project hair rule): its mass wave (main.js wavyMass) is an elliptical
   helix of 0.034 HH out and 0.040 HH sideways, wavelength 0.5 + 0.08 x section HH (110 to 180 mm, five sections
   round the head), each section its own phase, grown in from z 0.3 to -0.05. Each lock takes its section's
   wavelength and phase. Gravity pulls a wave out where much hair hangs below it: a planar wave of wavenumber k under
   tension w h (h the hair hanging below) settles at A0 / (1 + h / (Lb^3 k^2)), Lb = (EI / w)^(1/3) (Kirchhoff rod,
   small amplitude; the Warden waves round, hair-rebuild/ranger-foundations.md, checked against a free rod to 0.2 mm).
   So the rest wave is preloaded by that factor (gravity preloading: Disney's and Pixar's long-hair rigs keep a curl's
   rest shape against gravity, fxguide on Merida; the Warden round's stylist sources). REST_GAIN is CALIBRATED so the
   settled wave height matches the creator's, measured the same way on both (wave_height). The last END_STRAIGHT_MM
   of each lock is straight (stylists leave about an inch of the ends: tymobeauty.com, behindthechair.com, as cited
   in the Warden round).
4. The finish is the ponytail's: a mid layer of clumps (hair_layers.py), the noising round (noise_round.py), fairing
   of kinks where constructions meet, light through the hair per creator lamp (hair_light.py), how alone each strand
   is (strand_iso.py), the out direction for lighting taken from each lock's own middle, and the scalp cap at
   CAP_LIFT_MM under the roots.
"""
import gzip
import json
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'solvers', 'hair'))
import scalp_field as SF  # noqa: E402
import render as RD  # noqa: E402
import groom as GR  # noqa: E402
import creator_asset as CA  # noqa: E402
import tail_lock_rods as TR  # noqa: E402

MM = SF.MM

HANDOVER_Z = 0.30           # SOURCED (creator): wavyMass's wave grows in from z 0.3 (W = sm(0.3, -0.05, z)); its phase
WAVE_FULL_Z = -0.05         # counts depth from there, so a lock's s = 0 is the creator's depth 0
WAVE_OUT_HH, WAVE_SIDE_HH = 0.034, 0.040     # SOURCED (creator): wavyMass's two axes
MIN_HANG_MM = 40.0          # a hair hanging less than this below the hand-over stays on its head line (no lock)
LOCK_SPAN_MM = 30.0         # DRAFT: a wave lock wider than this at the hand-over is split (k-means); a lock is 1-2 cm
SEG_OUT_MM = 2.5            # the strands' point spacing (the ponytail's field lines run 2-3 mm)
PACKING = 0.1               # DRAFT, the tail's TAIL_LOCK_PACKING: a loose lock's packing, for its contact radius
JOIN_MM = 45.0              # the tail's TAIL_JOIN_MM: head line to lock, quintic ease
CLUMP0, CLUMP_TIP = 0.12, 0.3   # DRAFT: a hair drawn this far to its lock's middle, at the top and toward the tip
SUBLOCKS = 3                # the tail's: a lock parts into sub-locks as it falls
SUB_SPREAD, HAIR_SPREAD = 0.8, 0.4   # DRAFT (the tail's 1.4 / 0.7 halved: loose wavy hair hangs closer than a tail)
SPREAD_FROM = 0.15          # the share of a lock's length where the opening starts
END_STRAIGHT_MM = 25.0      # stylists: about an inch of the ends left out of the wave
REST_GAIN = 1.0             # CALIBRATED below (the settled wave against the creator's)
CAP_LIFT_MM = 0.4           # the scalp cap under the roots (PR #97: at 1 mm the roots beside a join lay under it)
LOCK_ROUND = 0.8            # the tail's TAIL_LOCK_ROUND: a hanging hair's "out" is mostly its lock's own


def creator_wave(th):
    """The creator's wave for a direction round the head (wavyMass): wavelength (HH) and phase by section."""
    a = np.abs(th)
    sec = np.minimum(4, np.floor((a + 0.2) / 0.75)).astype(int)
    ph = np.array([0.0, 1.9, 3.6, 5.1, 0.8])[sec] + np.where(th > 0, 0.6, 0.0)
    return 0.5 + 0.08 * sec, ph, sec


def resample(p, step):
    L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if L[-1] < step:
        return p
    from scipy.interpolate import CubicSpline
    keep = np.r_[True, np.diff(L) > 1e-9]
    cs = CubicSpline(L[keep], p[keep], bc_type='natural')
    return cs(np.arange(0, L[-1] + 1e-9, step))


def colliders(surf, body):
    """The head (the skin's radius map), the ears (groom.py's ellipsoids) and her body (collar, cape, back), as one
    depth in mm (positive inside) with its outward normal, for the rods."""
    def depth(X):
        d = np.where(X[:, 2] > 0.2, surf.depth(X), -1.0)              # (the skin's radius map is the head's above z 0.2,
        d = np.maximum(d, GR.ear_depth_np(X))                          #  as groom.Colliders reads it; below that, the body)
        return np.maximum(d, GR.body_depth_np(body, X))

    def f(pts_mm, rad_mm):
        X = pts_mm * MM
        d0 = depth(X)
        e = 0.5 * MM
        g = np.stack([(depth(X + e * np.eye(3)[k]) - depth(X - e * np.eye(3)[k])) / (2 * e) for k in range(3)], 1)
        n = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        return d0 / MM + rad_mm, n
    return f, depth


def wave_height(lines, zbands=((0.3, 0.0), (0.0, -0.3), (-0.3, -0.6), (-0.6, -0.9)), win_mm=130.0, step_mm=2.0):
    """The wave's height, measured the same way on the creator and the groom (as the Warden round did): each line
    resampled every step_mm, its horizontal deviation from its own moving average over about one wavelength, the
    median and 90th percentile by height band, mm."""
    k = max(3, int(win_mm / step_mm) | 1)
    dev, zz = [], []
    for p in lines:
        q = resample(p, step_mm * MM)
        if len(q) < k + 2:
            continue
        ker = np.ones(k) / k
        avg = np.stack([np.convolve(q[:, c], ker, mode='valid') for c in range(3)], 1)
        mid = q[k // 2: k // 2 + len(avg)]
        dev.append(np.hypot(*(mid - avg)[:, :2].T) / MM); zz.append(mid[:, 2])
    if not dev:
        return {}
    dev, zz = np.concatenate(dev), np.concatenate(zz)
    out = {}
    for hi, lo in zbands:
        m = (zz <= hi) & (zz > lo)
        if m.sum() > 20:
            out['%.1f..%.1f' % (hi, lo)] = (round(float(np.median(dev[m])), 1), round(float(np.percentile(dev[m], 90)), 1))
    return out


def hang_locks(paths, lockkey, hairs, coll, rest_gain, log, seed=9):
    """The hair below the hand-over as wave locks solved in 3D (see the module's note 2 and 3). paths: list of (k, 3)
    head-height lines at SEG_OUT_MM. Returns the new lines, each line's lock (-1: none), each point's lock "out"
    (zero off a lock) and the report."""
    rng = np.random.default_rng(seed)
    n = len(paths)
    h_i = np.full(n, -1)
    for i, p in enumerate(paths):
        below = np.where(p[:, 2] < HANDOVER_Z)[0]
        if len(below) and (len(p) - below[0]) * SEG_OUT_MM > MIN_HANG_MM:
            h_i[i] = below[0]
    on = np.where(h_i >= 0)[0]
    H = np.array([paths[i][h_i[i]] for i in on])
    # the locks: the groom's wave locks (the creator's 88), split where one is wider than a lock at the hand-over
    key = lockkey[on]
    lock = np.full(n, -1)
    nl = 0
    for k in np.unique(key):
        mem = np.where(key == k)[0]
        span = np.ptp(H[mem], axis=0).max() / MM if len(mem) > 1 else 0.0
        kk = max(1, int(math.ceil(span / LOCK_SPAN_MM)))
        if kk > 1 and len(mem) > kk:
            from scipy.cluster.vq import kmeans2
            _, lab = kmeans2(H[mem], kk, seed=int(k) % 1000 + 1, minit='++')
        else:
            lab = np.zeros(len(mem), int)
        for c in np.unique(lab):
            lock[on[mem[lab == c]]] = nl; nl += 1
    # each lock's rod: its root the members' middle at the hand-over, heading as they head over their next 30 mm
    roots = np.zeros((nl, 3)); tang = np.zeros((nl, 3)); length = np.zeros(nl); nh = np.zeros(nl)
    srel_max = {}
    for k in range(nl):
        mem = np.where(lock == k)[0]
        roots[k] = np.mean([paths[i][h_i[i]] for i in mem], 0)
        j = int(30.0 / SEG_OUT_MM)
        tv = np.mean([paths[i][min(len(paths[i]) - 1, h_i[i] + j)] - paths[i][h_i[i]] for i in mem], 0)
        tang[k] = tv / max(np.linalg.norm(tv), 1e-12)
        ls = np.array([(len(paths[i]) - 1 - h_i[i]) * SEG_OUT_MM for i in mem])
        length[k] = np.percentile(ls, 98)
        nh[k] = hairs[mem].sum()
    out_h = roots.copy(); out_h[:, 2] = 0; out_h[:, 0] -= SF.AXIS[0]; out_h[:, 1] -= SF.AXIS[1]
    d1 = out_h - (out_h * tang).sum(1, keepdims=True) * tang
    d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-12)
    d2 = np.cross(tang, d1)
    frames = np.stack([d1, d2, tang], 2)
    lock_r = np.sqrt(nh * np.pi * 0.035 ** 2 / PACKING / np.pi)
    # the creator's wave per lock: wavelength and phase by section; the handedness so that the lock's offset runs
    # out = sin(psi), sideways = side x cos(psi) as wavyMass draws it (see the module's note 3)
    th = SF.theta(roots)
    lam_hh, ph, sec = creator_wave(th)
    lam = lam_hh / MM
    side = np.where(th < 0, -1.0, 1.0)
    t_th = np.c_[np.cos(th), np.sin(th), np.zeros(nl)]                 # (d theta direction: theta = atan2(x, -y))
    sig = np.sign((d2 * (side[:, None] * t_th)).sum(1)); sig[sig == 0] = 1
    tsign = -sig
    phi = tsign * (ph + np.pi / 2)
    EI, w = nh * TR.EI_HAIR, nh * TR.W_HAIR
    Lb3 = EI / w                                                       # mm^3: (EI / w), the bending length cubed
    A_mm = 0.5 * (WAVE_OUT_HH + WAVE_SIDE_HH) / MM                     # the creator's wave, mean of its two axes

    def rest(s):
        c = lam[:, None] / (2 * np.pi)
        z = HANDOVER_Z - s * MM                                         # (the creator's W by depth below the hand-over)
        Wc = SF.sm(HANDOVER_Z, WAVE_FULL_Z, z)
        end = SF.sm(length[:, None], length[:, None] - END_STRAIGHT_MM, s)
        h = np.maximum(length[:, None] - s, 0.0)                         # hair hanging below, mm
        pre = 1.0 + h / (Lb3[:, None] * (2 * np.pi / lam[:, None]) ** 2)
        # the settled helix (the creator's radius A at this lock's pitch) has curvature A / (A^2 + c^2); the rest
        # curvature is that, grown in as the creator's W, straight over the last inch, and preloaded against the
        # tension (in the small-amplitude rod the settled curvature is the rest curvature over `pre`; preloading the
        # radius instead saturates, since a helix's curvature peaks at radius c)
        kap = A_mm / (A_mm ** 2 + c * c)
        # (the turn rate along the hair: one turn per vertical wavelength on a helix of the settled radius, whose
        # arc per turn is 2 pi sqrt(A^2 + c^2))
        return rest_gain * kap * Wc * end * pre, tsign / np.sqrt(A_mm ** 2 + c[:, 0] ** 2), phi
    log('locks %d, hairs per lock median %.0f, lock radius median %.1f mm, length median %.0f mm, bending length %.0f mm'
        % (nl, np.median(nh), np.median(lock_r), np.median(length), np.median(Lb3 ** (1 / 3))))
    out = TR.solve(roots / MM, frames, length, nh, lock_r, head=coll, seed=seed, rest=rest,
                   log=lambda t: log('  rods: ' + t))
    X, Rr, rs = out['x'] * MM, out['R'], out['s']
    M = Rr.shape[1]
    # the hairs ride their locks
    new = list(paths)
    lock_out = [np.zeros_like(p) for p in paths]
    sub_ph = rng.random(nl) * 2 * np.pi
    for k in range(nl):
        mem = np.where(lock == k)[0]
        R0 = frames[k]
        offs = np.array([paths[i][h_i[i]] - roots[k] for i in mem])
        a0, b0 = offs @ R0[:, 0], offs @ R0[:, 1]
        sub = (((np.arctan2(b0 - b0.mean(), a0 - a0.mean()) + sub_ph[k]) % (2 * np.pi)) / (2 * np.pi / SUBLOCKS)).astype(int) % SUBLOCKS
        for si in range(SUBLOCKS):
            ms = sub == si
            if not ms.any():
                continue
            qa, qb = a0[ms].mean(), b0[ms].mean()
            for i, a, b in zip(mem[ms], a0[ms], b0[ms]):
                p = paths[i]; j0 = h_i[i]
                srel = np.arange(len(p) - j0) * SEG_OUT_MM
                seg = rs[k, 1]
                ii = np.clip((srel / seg).astype(int), 0, M - 1)
                fr = srel / seg - ii
                base = X[k, ii] + (fr * seg * MM)[:, None] * Rr[k, ii][:, :, 2]
                t01 = np.clip(srel / max(length[k], 1.0), 0, 1)
                g = SF.sm(SPREAD_FROM, 1.0, t01)
                wcl = SF.sm(0.0, 80.0, srel) * (CLUMP0 + (CLUMP_TIP - CLUMP0) * SF.sm(0.5, 1.0, t01))
                oa = (qa * (1 + SUB_SPREAD * g) + (a - qa) * (1 + HAIR_SPREAD * g)) * (1 - wcl)
                ob = (qb * (1 + SUB_SPREAD * g) + (b - qb) * (1 + HAIR_SPREAD * g)) * (1 - wcl)
                i2 = np.clip(ii + np.where(fr > 0.5, 1, -1), 0, M - 1)
                wf = np.abs(fr - 0.5)[:, None]
                D1 = Rr[k, ii][:, :, 0] * (1 - wf) + Rr[k, i2][:, :, 0] * wf
                D2 = Rr[k, ii][:, :, 1] * (1 - wf) + Rr[k, i2][:, :, 1] * wf
                rod = base + oa[:, None] * D1 + ob[:, None] * D2
                kj = np.clip(srel / JOIN_MM, 0, 1); kj = kj ** 3 * (10 - 15 * kj + 6 * kj * kj)
                q = p.copy()
                q[j0:] = p[j0:] + kj[:, None] * (rod - p[j0:])
                new[i] = q
                o = oa[:, None] * D1 + ob[:, None] * D2
                o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-12)
                lo = np.zeros_like(p); lo[j0:] = o * SF.sm(0.0, 25.0, srel)[:, None]
                lock_out[i] = lo
    rep = dict(out['report'], hairs_on_locks=int((lock >= 0).sum()), lock_radius_mm_median=round(float(np.median(lock_r)), 1),
               length_mm_median=round(float(np.median(length)), 0), rest_gain=rest_gain)
    return new, lock, lock_out, rep, dict(X=X, s=rs, nh=nh, length=length, sec=sec)


def main():
    a = sys.argv[1:]
    head, gdir, style, out = a[:4]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    nfib, nstr = opt('--fibres', 24000), opt('--strands', 16000)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'build.log'), 'w')

    def log(t):
        print(t, file=sys.stderr); logf.write(t + '\n'); logf.flush()
    rep = {}
    surf = SF.Surface(SF.load_head(head))
    F = SF.build(surf)
    bodies = json.load(gzip.open(head))['bodies']
    body = GR.character_body(bodies['warden'], bodies['mannequin'])
    Gm = dict(np.load(os.path.join(gdir, 'groom.npz')))
    paths0, share = RD.grow_children(F, Gm, nfib)
    near = cKDTree(Gm['roots']).query(paths0[:, 0])[1]
    lockkey = Gm['lock'][near].astype(int) * 1000 + Gm['clump'][near].astype(int)
    hairs = share * float(Gm['n_hairs'].sum())
    paths = [resample(p[~np.isnan(p[:, 0])], SEG_OUT_MM * MM) for p in paths0]
    log('fibres %d, %.1f hairs each (median)' % (len(paths), np.median(hairs)))
    # the creator's wave, measured
    import creator_truth as CT
    _, _, cstr = CT.load(style)
    rep['wave_creator_mm'] = wave_height([s['p'] for s in cstr if s['p'][:, 2].min() < 0.0])
    rep['wave_groom_before_mm'] = wave_height([p for p in paths if p[:, 2].min() < 0.0][::4])
    log('wave height, creator %s; old groom %s' % (rep['wave_creator_mm'], rep['wave_groom_before_mm']))
    coll, depth = colliders(surf, body)
    gain = opt('--rest-gain', REST_GAIN)
    paths, lock, lock_out, rr, rods = hang_locks(paths, lockkey, hairs, coll, gain, log)
    rep['rods'] = rr
    rep['wave_locks_mm'] = wave_height([x for x in rods['X']])
    rep['wave_groom_after_mm'] = wave_height([p for p in paths if p[:, 2].min() < 0.0][::4])
    log('rods %s' % rr)
    log('wave height: rods %s; hairs %s' % (rep['wave_locks_mm'], rep['wave_groom_after_mm']))
    # into one nan-padded array for the finishing passes
    m = max(len(p) for p in paths)
    P = np.full((len(paths), m, 3), np.nan)
    LO = np.zeros((len(paths), m, 3))
    for i, p in enumerate(paths):
        P[i, :len(p)] = p; LO[i, :len(p)] = lock_out[i]
    tier = None
    if '--no-mid-layer' not in a:
        import hair_layers as HLy
        P, tier, ml = HLy.mid_layer(P)
        rep['mid_layer'] = ml; log('mid layer %s' % ml)
    if '--noise' in a:
        import noise_round as NR
        P, nz = NR.apply(P, surf=surf, scale=opt('--noise', 1.0))
        rep['noise'] = nz; log('noise %s' % nz)
    # nothing inside the head or body after the passes: such a node is set back out along the collider's normal
    good = ~np.isnan(P[:, :, 0])
    flat = P[good]
    dd, nn = coll(flat / MM, np.full(len(flat), 0.6))
    ins = dd > 0
    flat[ins] += nn[ins] * (dd[ins] * MM)[:, None]
    P[good] = flat
    rep['pushed_out_nodes'] = int(ins.sum())
    np.savez_compressed(os.path.join(out, 'fibres.npz'), paths=P.astype(np.float32), hairs=hairs, lock=lock,
                        tier=tier if tier is not None else np.zeros(len(P), int))
    # the strands drawn: a share of the fibres, each standing for more hair
    rng = np.random.default_rng(5)
    keep = np.sort(rng.choice(len(P), min(nstr, len(P)), replace=False))
    ps = [P[k][~np.isnan(P[k][:, 0])] for k in keep]
    los = [LO[k][~np.isnan(P[k][:, 0])] for k in keep]
    ok = [j for j, p in enumerate(ps) if len(p) >= 2]
    ps, los, keep = [ps[j] for j in ok], [los[j] for j in ok], keep[ok]
    counts = np.array([len(p) for p in ps])
    import bangs_creator_asset as BCA
    Pc = np.concatenate(ps)
    region = np.concatenate([np.where(np.linalg.norm(l, axis=1) > 0.5, 2, 0) for l in los])
    k_before = BCA.kink_gauge(Pc, counts, region)
    Pc, fr_ = BCA.fair(Pc, counts)
    rep['kinks'] = dict(before=k_before, faired=BCA.kink_gauge(Pc, counts, region), fairing=fr_)
    log('kinks %s' % rep['kinks'])
    LOc = np.concatenate(los)
    share_d = float(hairs.sum()) / len(ps)
    q = 2.5 / 32767
    # the scalp cap, under the roots
    import cards as CD
    cap_dir = os.path.join(out, 'cap'); os.makedirs(cap_dir, exist_ok=True)
    CD.scalp_cap(head, cap_dir, lift_mm=CAP_LIFT_MM)
    V, _, Fc = CA.read_obj(os.path.join(cap_dir, 'scalp-cap.obj'))
    centre = V.mean(0) - np.array([0, 0, 0.15])
    tree = cKDTree(Pc[::2])
    outS = Pc - np.where(Pc[:, 2:3] < centre[2], np.array([centre[0], centre[1], 0]) + np.array([0, 0, 1]) * Pc[:, 2:3], centre)
    outS /= np.maximum(np.linalg.norm(outS, axis=1, keepdims=True), 1e-6)
    cs = sum(tree.query_ball_point(Pc + outS * d, CA.AO_R, return_length=True) for d in CA.AO_STEPS)
    sao = np.clip(np.exp(-cs / (np.median(cs) / -np.log(CA.SAO_MID))), CA.SAO_MIN, 1.0)
    hsh = lambda v: ((v.astype(np.int64) * 2654435761 + 12345) >> 7) % 251
    lid = np.where(lock[keep] >= 0, 104729 + lock[keep], lockkey[keep])
    ids = np.stack([hsh(lid), hsh(keep)], 1).astype('u1')
    import hair_light as HL
    lt = HL.light(np.split(Pc, np.cumsum(counts)[:-1]), hairs[keep] * len(P) / len(keep))
    occ = np.stack([lt['key'], lt['rim'], lt['fill'], lt['ambient']], 1)
    rep['light'] = dict(key_median=round(float(np.median(lt['key'])), 2), key_p10=round(float(np.percentile(lt['key'], 10)), 2),
                        ambient_median=round(float(np.median(lt['ambient'])), 2))
    log('light %s' % rep['light'])
    # out: from the skull on the head, from the lock's own middle where it hangs (blended in over its first 25 mm)
    oh = Pc - np.c_[np.full(len(Pc), centre[0]), np.minimum(Pc[:, 1], centre[1]), np.full(len(Pc), centre[2])]
    oh /= np.maximum(np.linalg.norm(oh, axis=1, keepdims=True), 1e-9)
    wl = np.linalg.norm(LOc, axis=1)
    lo_u = LOc / np.maximum(wl[:, None], 1e-9)
    ol = LOCK_ROUND * lo_u + (1 - LOCK_ROUND) * oh
    outv = oh * (1 - wl)[:, None] + ol * wl[:, None]
    outv /= np.maximum(np.linalg.norm(outv, axis=1, keepdims=True), 1e-9)
    import strand_iso as SI
    iso = SI.iso(Pc / MM, counts)[0]
    asset = dict(lods=[], guides=0, cap=dict(pos=CA.b64(V, '<f4'), idx=CA.b64(Fc, '<u4')),
                 strands=dict(n=int(len(ps)), counts=CA.b64(counts, '<u2'), pos=CA.b64(np.round(Pc / q), '<i2'), q=q,
                              hairs_per_strand=round(share_d, 2), ao=CA.b64(np.round(sao * 255), 'u1'), ids=CA.b64(ids, 'u1'),
                              light=CA.b64(np.round(np.clip(occ, 0, 1) * 255), 'u1'), light_order='key,rim,fill,ambient',
                              out=CA.b64(np.round(outv * 127), 'i1'), iso=CA.b64(np.round(iso * 255), 'u1'),
                              **({'tier': CA.b64(tier[keep], 'u1')} if tier is not None else {})),
                 body=dict(name='warden', sections=bodies['warden'], cy=0.04, p=3),
                 source=dict(groom=os.path.abspath(gdir), fibres=int(len(P)), builder='tools/hair/wavy_flow.py'))
    json.dump(asset, open(os.path.join(out, 'asset-wavyFull.json'), 'w'))
    rep['strands'] = dict(n=int(len(ps)), hairs_each=round(share_d, 1), loose_points_share=round(float((iso > 0.5).mean()), 4))
    # the gauges on the finished hair: the hairline invariant (hairline_gap.py) and the join between the bangs and the
    # hair behind them (join_cover.py, PR #97)
    import hairline_gap as HG
    Hc, Wc = HG.measure(F, surf, P, hairs)
    hl = HG.report(Hc, Wc)
    rep['hairline'] = {k: dict(bare=len(v['bare']), longest_gap_mm=v['longest_gap_mm'], thinnest_per_mm=v['thinnest']) for k, v in hl.items()}
    HG.draw(Hc, Wc, os.path.join(out, 'hairline_gap.png'))
    sys.path.insert(0, os.path.join(HERE, '..', 'gauges', 'hair'))
    import join_cover as JC
    jc = JC.gauge(head, os.path.join(out, 'asset-wavyFull.json'), cap_mm=CAP_LIFT_MM, log=lambda t: logf.write(t + '\n'))
    rep['join'] = {k: v for k, v in jc.items() if k != 'slabs'}
    json.dump(rep, open(os.path.join(out, 'wavy.json'), 'w'), indent=1)
    log(json.dumps(rep))


if __name__ == '__main__':
    main()
