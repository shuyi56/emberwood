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
5. Sectioning (John 2026-11-28: the shoulder-length hair "divided into ugly clumps"). How it is done on a head: a
   wave set is taken in sections 1 to 1.5 in wide (Sam Villa, samvilla.com 3D waves; "the larger the section, the
   looser the curl", saloncentric.com), no wider than the tool (Red Seal study guide, roller sets), in rows from the
   bottom up, laid like bricks "eliminating visible partings" (same guide), and then brushed or combed out "to form
   a uniform wave pattern" (saloncentric.com; behindthechair.com: "gently run a wide tooth comb throughout all of the
   curls"). So the sections are a way of making the wave, and none is left standing as a rope. In a groom: clumps are
   blended into their neighbours (XGen Clump modifier, Copy and Copy Variance; Clump Variance "to breakup uniform
   looking clumps"), come in levels (primary, secondary, tertiary: MetaHuman groom tools; smaller clumps a third of
   the medium: P. Zielinski, 80.lv) and flatten toward the tips in long hair (XGen Flatness). Here: a hair takes the
   smooth lines of the locks within one section's width of it and its own lock's wave (hang_locks), a lock is not
   drawn tighter toward its tip, and it flattens as it falls. Gauged by curtain_gauge.
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
CLUMP0, CLUMP_TIP = 0.12, 0.12  # DRAFT: a hair drawn this far to its lock's middle, at the top and toward the tip
                            # (0.3 at the tip drew every lock to a point: a hem of fingers. Brushed-out waves end open)
SECTION_BLEND = 0.8         # DRAFT: how far a hair takes its neighbours' lines with its own lock's (0: separate ropes)
BLEND_R_MM = 40.0           # SOURCED: a wave section is 1 to 1.5 in wide (Sam Villa; Red Seal roller sets: no wider than
                            # the roller); a hair is shared between the locks within one section's width of it
FLATNESS = 0.5              # DRAFT: a lock's depth (out from the body) at its tip against its width (XGen Flatness)
SUBLOCKS = 3                # the tail's: a lock parts into sub-locks as it falls
SUB_SPREAD, HAIR_SPREAD = 0.3, 0.4   # DRAFT (the tail's 1.4 / 0.7; sub-locks at 0.8 parted every lock into three fingers at the hem)
SPREAD_FROM = 0.15          # the share of a lock's length where the opening starts
END_STRAIGHT_MM = 25.0      # stylists: about an inch of the ends left out of the wave
REST_GAIN = 1.0             # CALIBRATED below (the settled wave against the creator's)
CAP_LIFT_MM = 0.4           # the scalp cap under the roots (PR #97: at 1 mm the roots beside a join lay under it)
FRONT_SHARE = 0.5           # a lock is a front lock where more than this share of the creator's strands at its angle
                            # (hand-over theta within FRONT_DTH, same side) hang in front of the shoulder
FRONT_DTH = 0.12            # rad
FRONT_AT_Z = -0.15          # where a front lock is aimed: the creator's strands' median point at this height
WAVE_TAKE = 1.15            # DRAFT: hair length per unit of hang along a waved lock (the Warden round measured about 20%
                            # taken up by the wave; the hang here also runs forward over the shoulder)
LOOK_HUE = 0.8              # STYLE: the highlight and the glints this far toward the hair's hue (hair-fidelity.js asset.look:
                            # glints are TRT light and carry the pigment, Marschner et al. 2003; white ones read ashy)
LOOK_LIGHT_FLOOR, LOOK_INNER_DARK, LOOK_MID_DARK = 0.25, 0.85, 0.92   # STYLE (swatches, John 2026-11-28): the least light
                            # under other hair and the two inner-layer tones (hair-fidelity.js asset.look); at the
                            # shader's 0, 0.65 and 0.8 every inner lock read as a hard shadow on this fall
LOOK_TONE = dict(strand_light=0.4, glint=0.15, spec1=0.05, spec2=0.15, scatter=3.0, kk_gain=1.5)   # CALIBRATED against
                            # the creator's own hair from the same cameras (median sRGB 45/29/22 front; the shader's
                            # defaults gave 109/73/55, these 71/46/36), John 2026-11-28: "too light". Light floor 0.4 to 0.25.
MIN_LOCK_SHARE = 0.25       # DRAFT: a lock with under this share of the median lock's hair joins its nearest neighbour
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


def side_gauge(lines, hairs=None):
    """Left against right below the jaw: per side (her left is +x), the hair in front of the shoulder line (y under the
    head's axis) at z -0.3 and -0.5 (hairs crossing), the front tips' height (median, 5th pct) and the half-width of
    all the hair at z -0.3 (98th pct of |x|). A style that is the same on both sides should match within a lock."""
    hairs = np.ones(len(lines)) if hairs is None else hairs
    out = {}
    for name, sg in (('left', 1.0), ('right', -1.0)):
        tips, wt, cross, half = [], [], {-0.3: 0.0, -0.5: 0.0}, []
        for p, h in zip(lines, hairs):
            p = p[~np.isnan(p[:, 0])]
            if len(p) < 2 or p[-1, 2] > 0.0 or p[-1, 0] * sg <= 0:
                continue
            j = int(np.argmin(np.abs(p[:, 2] + 0.3)))
            if abs(p[j, 2] + 0.3) < 0.02:
                half.append(abs(p[j, 0]))
            if p[-1, 1] < SF.AXIS[1]:
                tips.append(p[-1, 2]); wt.append(h)
                for z in cross:
                    if p[-1, 2] < z:
                        cross[z] += h
        out[name] = dict(front_hairs=int(sum(wt)), tip_z_median=round(float(np.median(tips)), 3) if tips else None,
                         tip_z_p5=round(float(np.percentile(tips, 5)), 3) if tips else None,
                         front_hairs_at={str(z): int(v) for z, v in cross.items()},
                         half_width_at_m03=round(float(np.percentile(half, 98)), 3) if half else None)
    return out


def curtain_gauge(lines, hairs=None, zs=(-0.1, -0.2, -0.3), cell_mm=4.0, hem_mm=20.0):
    """How far the hanging hair is one curtain or separate clumps. At each height: the hairs crossing it laid out by
    their angle round the body (arc mm), in cells of cell_mm; `empty` is the share of empty cells inside the span the
    hair covers on each side, `cv` the spread of the hair per cell over its mean (a comb of ropes is high on both).
    `hem_step_mm`: the median difference in tip height between hairs that are neighbours round the body within hem_mm
    (a hem of fingers steps; a brushed-out hem runs on)."""
    hairs = np.ones(len(lines)) if hairs is None else hairs
    out = {}
    L = [p[~np.isnan(p[:, 0])] for p in lines]
    for z in zs:
        th, r, w = [], [], []
        for p, h in zip(L, hairs):
            if len(p) < 2 or p[-1, 2] > z or p[0, 2] < z:
                continue
            j = int(np.argmin(np.abs(p[:, 2] - z)))
            th.append(SF.theta(p[j:j + 1])[0]); r.append(np.hypot(p[j, 0] - SF.AXIS[0], p[j, 1] - SF.AXIS[1])); w.append(h)
        th, r, w = np.array(th), np.array(r), np.array(w)
        emp, cv = [], []
        for sg in (1, -1):
            m = np.sign(th) == sg
            if m.sum() < 20:
                continue
            arc = np.abs(th[m]) * np.median(r[m]) / MM
            lo, hi = np.percentile(arc, [2, 98])
            c = np.histogram(arc, np.arange(lo, hi + cell_mm, cell_mm), weights=w[m])[0]
            emp.append(float((c == 0).mean())); cv.append(float(c.std() / max(c.mean(), 1e-9)))
        out[str(z)] = dict(empty=round(float(np.mean(emp)), 3), cv=round(float(np.mean(cv)), 2))
    tip = np.array([p[-1] for p in L if len(p) > 1 and p[-1, 2] < 0.0])
    if len(tip) > 10:
        tr = cKDTree(np.c_[tip[:, 0], tip[:, 1]])
        d, j = tr.query(np.c_[tip[:, 0], tip[:, 1]], 2)
        ok = d[:, 1] < hem_mm * MM
        out['hem_step_mm'] = round(float(np.median(np.abs(tip[ok, 2] - tip[j[ok, 1], 2])) / MM), 1)
    return out


def creator_front(strands):
    """The creator's strands that hang below its hand-over: each one's angle round the head where it passes z
    HANDOVER_Z, whether it hangs in front of the shoulder, its point at FRONT_AT_Z and its lowest height."""
    th, fr, at, end = [], [], [], []
    for p in strands:
        if p[:, 2].min() > FRONT_AT_Z or p[0, 2] < HANDOVER_Z:
            continue
        i = np.where(p[:, 2] < HANDOVER_Z)[0][0]
        j = int(np.argmin(np.abs(p[:, 2] - FRONT_AT_Z)))
        th.append(SF.theta(p[i:i + 1])[0]); fr.append(p[j, 1] < SF.AXIS[1]); at.append(p[j]); end.append(p[:, 2].min())
    return np.array(th), np.array(fr), np.array(at), np.array(end)


def hang_locks(paths, lockkey, hairs, coll, rest_gain, log, seed=9, creator=None, blend=None):
    """The hair below the hand-over as wave locks solved in 3D (see the module's note 2 and 3). paths: list of (k, 3)
    head-height lines at SEG_OUT_MM. Returns the new lines, each line's lock (-1: none), each point's lock "out"
    (zero off a lock) and the report."""
    rng = np.random.default_rng(seed)
    blend = SECTION_BLEND if blend is None else blend
    n = len(paths)
    h_i = np.full(n, -1)
    for i, p in enumerate(paths):
        below = np.where(p[:, 2] < HANDOVER_Z)[0]
        if len(below) and (len(p) - below[0]) * SEG_OUT_MM > MIN_HANG_MM:
            h_i[i] = below[0]
    on = np.where(h_i >= 0)[0]
    pull = np.zeros((n, 3))                                            # (a merged hair's shift onto its new lock)
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
    # a lock of a few hairs is no lock: on its own rod it hangs alone, clear of the mass (the dangling lock over the
    # collar, 13 fibres; John, 2026-11-28). It joins the nearest full lock on its side, and its hairs are drawn in to
    # that lock's own spread.
    Hall = np.full((n, 3), np.nan); Hall[on] = H
    cnt = np.array([hairs[lock == k].sum() for k in range(nl)])
    small = cnt < MIN_LOCK_SHARE * np.median(cnt)
    if small.any() and (~small).any():
        cen = np.array([Hall[lock == k].mean(0) for k in range(nl)])
        big = np.where(~small)[0]
        for k in np.where(small)[0]:
            same = big[np.sign(cen[big, 0]) == np.sign(cen[k, 0])]
            cand = same if len(same) else big
            j = cand[np.argmin(np.linalg.norm(cen[cand] - cen[k], axis=1))]
            host = np.where(lock == j)[0]
            rad = np.percentile(np.linalg.norm(Hall[host] - cen[j], axis=1), 90)
            for i in np.where(lock == k)[0]:
                d = Hall[i] - cen[j]; dn = np.linalg.norm(d)
                pull[i] = cen[j] + d * min(1.0, rad / max(dn, 1e-9)) - Hall[i]
                lock[i] = j
        remap = np.cumsum(~small) - 1
        lock[lock >= 0] = remap[lock[lock >= 0]]
        nl = int((~small).sum())
        log('small locks merged %d (under %.0f hairs), locks now %d' % (small.sum(), MIN_LOCK_SHARE * np.median(cnt), nl))
    # each lock's rod: its root the members' middle at the hand-over, heading as they head over their next 30 mm
    roots = np.zeros((nl, 3)); tang = np.zeros((nl, 3)); length = np.zeros(nl); nh = np.zeros(nl)
    srel_max = {}
    for k in range(nl):
        mem = np.where(lock == k)[0]
        roots[k] = np.mean([paths[i][h_i[i]] + pull[i] for i in mem], 0)
        j = int(30.0 / SEG_OUT_MM)
        tv = np.mean([paths[i][min(len(paths[i]) - 1, h_i[i] + j)] - paths[i][h_i[i]] for i in mem], 0)
        tang[k] = tv / max(np.linalg.norm(tv), 1e-12)
        ls = np.array([(len(paths[i]) - 1 - h_i[i]) * SEG_OUT_MM for i in mem])
        length[k] = np.percentile(ls, 98)
        nh[k] = hairs[mem].sum()
    # front locks (John 2026-11-28, "yes" to bringing them back): where the creator's hair at a lock's angle hangs in
    # front of the shoulder, the lock heads for the creator's own place there and is as long as the creator's
    front = np.zeros(nl, bool); grow = np.ones(n)
    frac = 0.85 + 0.15 * rng.random(n)                                 # (each hair's share of a front lock's length)
    if creator is not None:
        cth, cfr, cat, cend = creator
        thk = SF.theta(roots)
        for k in range(nl):
            # (the creator's two sides pooled, the far side mirrored: its wavy style is the same left and right below
            # the jaw (half-widths 0.60 and 0.61 HH at z -0.3), and taking each side's own strands gave the two sides
            # different targets and lengths: front tips at z -0.39 on her right against -0.27 on her left, John 2026-11-28)
            c = np.abs(np.abs(cth) - abs(thk[k])) < FRONT_DTH
            if c.sum() >= 3 and cfr[c].mean() > FRONT_SHARE:
                c &= cfr
                sgn = 1.0 if thk[k] >= 0 else -1.0
                m = cat[c].copy(); m[:, 0] = sgn * np.abs(m[:, 0])
                tgt = np.median(m, 0)
                tv = tgt - roots[k]; tv /= np.linalg.norm(tv)
                tang[k] = tv
                L = (np.linalg.norm(tgt - roots[k]) + (FRONT_AT_Z - np.median(cend[c]))) / MM * WAVE_TAKE
                mem = np.where(lock == k)[0]
                for i in mem:                                           # (each hair to its share of the new length)
                    own = (len(paths[i]) - 1 - h_i[i]) * SEG_OUT_MM
                    grow[i] = L * frac[i] / max(own, 1.0)               # (cut as well as grown: a side
                length[k] = L                                           #  left at its own longer length hung lower)
                front[k] = True
        log('front locks %d of %d (%d left, %d right; %d hairs lengthened, %d cut)' % (front.sum(), nl, (front & (thk > 0)).sum(),
            (front & (thk < 0)).sum(), int((grow > 1.0).sum()), int((grow < 1.0).sum())))
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
    # the hairs ride their locks. Sectioning (John 2026-11-28: "divided into ugly clumps"; research in the module's
    # note 5): a set of waves is taken in sections 25 to 38 mm wide, laid like bricks so that no parting runs through,
    # and then brushed out, so the sections merge into one waved surface. So a hair does not ride its own lock alone:
    # it takes the smooth line (the lock's line averaged over one wavelength) of the up to three locks nearest to it
    # at the hand-over, weighted by nearness (XGen's Copy: "blend the clumping effect between neighboring hair
    # clumps", by a different amount per hair, Copy Variance), and on that the wave of its own lock, so the wave keeps
    # the creator's phase. Its place across the lock is kept from the blended root, flattened toward the tip
    # ("the natural tendency of hair clumps to flatten out towards the tips in long hair", XGen Flatness).
    from scipy.ndimage import uniform_filter1d
    new = list(paths)
    lock_out = [np.zeros_like(p) for p in paths]
    sub_ph = rng.random(nl) * 2 * np.pi
    Xs = np.stack([uniform_filter1d(X[k], max(3, int(round(lam[k] / rs[k, 1]))) | 1, axis=0, mode='nearest') for k in range(nl)])
    Xw = X - Xs

    def samp(A, t):                                                     # (a line at fractional nodes, run on straight past its end)
        i = np.clip(np.floor(t).astype(int), 0, len(A) - 2)
        return A[i] + (t - i)[:, None] * (A[i + 1] - A[i])
    nn = min(3, nl)
    dn, jn = cKDTree(roots).query(Hall[on] + pull[on], nn)
    dn, jn = dn.reshape(len(on), nn), jn.reshape(len(on), nn)
    nbr = {}
    for r, i in enumerate(on):
        w = np.maximum(0.0, 1.0 - dn[r] / (BLEND_R_MM * MM)) ** 2
        own = (jn[r] == lock[i]).astype(float)
        if not own.any() or blend <= 0:
            nbr[i] = (np.array([lock[i]]), np.array([1.0])); continue
        u = blend * (0.5 + 0.5 * rng.random())                           # (per hair: Copy Variance)
        w = (1 - u) * own + u * w / max(w.sum(), 1e-9)
        nbr[i] = (jn[r][w > 1e-3], w[w > 1e-3] / w[w > 1e-3].sum())
    # the length: a hair between a front lock and its neighbour takes its share of each, so the hem runs on
    tgt_len = {}
    for i in on:
        own = (len(paths[i]) - 1 - h_i[i]) * SEG_OUT_MM
        js, ws = nbr[i]
        tgt_len[i] = float(sum(w * (length[j] * frac[i] if front[j] else own) for j, w in zip(js, ws)))
    for k in range(nl):
        mem = np.where(lock == k)[0]
        R0 = frames[k]
        rb = np.array([sum(w * roots[j] for j, w in zip(*nbr[i])) for i in mem])
        offs = np.array([paths[i][h_i[i]] + pull[i] for i in mem]) - rb
        own_o = np.array([paths[i][h_i[i]] + pull[i] - roots[k] for i in mem])
        a0, b0 = offs @ R0[:, 0], offs @ R0[:, 1]
        ao, bo = own_o @ R0[:, 0], own_o @ R0[:, 1]
        sub = (((np.arctan2(bo - bo.mean(), ao - ao.mean()) + sub_ph[k]) % (2 * np.pi)) / (2 * np.pi / SUBLOCKS)).astype(int) % SUBLOCKS
        for si in range(SUBLOCKS):
            ms = sub == si
            if not ms.any():
                continue
            qa, qb = a0[ms].mean(), b0[ms].mean()
            for i, a, b in zip(mem[ms], a0[ms], b0[ms]):
                p = paths[i]; j0 = h_i[i]
                srel = np.arange(max(2, int(round(tgt_len[i] / SEG_OUT_MM)) + 1)) * SEG_OUT_MM
                if len(srel) < len(p) - j0:                             # (a cut hair)
                    p = p[:j0 + len(srel)]
                if len(srel) > len(p) - j0:                             # (a lengthened hair: its head line runs on
                    p = np.r_[p, p[-1] + (p[-1] - p[-2])[None] * np.arange(1, len(srel) - (len(p) - j0) + 1)[:, None]]   # straight,
                                                                        #  then rides its lock like the rest)
                seg = rs[k, 1]
                ii = np.clip((srel / seg).astype(int), 0, M - 1)
                fr = srel / seg - ii
                js, ws = nbr[i]
                base = samp(Xw[k], srel / seg) + sum(w * samp(Xs[j], srel / rs[j, 1]) for j, w in zip(js, ws))
                t01 = np.clip(srel / max(srel[-1], 1.0), 0, 1)
                g = SF.sm(SPREAD_FROM, 1.0, t01)
                wcl = SF.sm(0.0, 80.0, srel) * (CLUMP0 + (CLUMP_TIP - CLUMP0) * SF.sm(0.5, 1.0, t01))
                oa = (qa * (1 + SUB_SPREAD * g) + (a - qa) * (1 + HAIR_SPREAD * g)) * (1 - wcl) * (1 - FLATNESS * g)
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
                o = D1 / np.maximum(np.linalg.norm(D1, axis=1, keepdims=True), 1e-12)   # (out: the lock's own outward side;
                lo = np.zeros_like(p); lo[j0:] = o * SF.sm(0.0, 25.0, srel)[:, None]    #  the merged hair is a sheet, not a rope)
                lock_out[i] = lo
    rep = dict(out['report'], hairs_on_locks=int((lock >= 0).sum()), lock_radius_mm_median=round(float(np.median(lock_r)), 1),
               length_mm_median=round(float(np.median(length)), 0), rest_gain=rest_gain, front_locks=int(front.sum()))
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
    cf = creator_front([s_['p'] for s_ in cstr]) if '--no-front' not in a else None
    paths, lock, lock_out, rr, rods = hang_locks(paths, lockkey, hairs, coll, gain, log, creator=cf, blend=opt('--blend', SECTION_BLEND))
    rep['rods'] = rr
    rep['wave_locks_mm'] = wave_height([x for x in rods['X']])
    rep['wave_groom_after_mm'] = wave_height([p for p in paths if p[:, 2].min() < 0.0][::4])
    rep['sides'] = dict(creator=side_gauge([s_['p'] for s_ in cstr]), hair=side_gauge(paths, hairs))
    log('sides %s' % rep['sides'])
    rep['curtain'] = curtain_gauge(paths, hairs)
    log('curtain %s' % rep['curtain'])
    nlk = lock.max() + 1
    sz = np.array([hairs[lock == k].sum() for k in range(nlk)])
    rep['locks'] = dict(n=int(nlk), hairs_median=int(np.median(sz)), hairs_min=int(sz.min()))
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
    # (a lock's far half is seen through the hair in front of it, between the next locks: lit by the lock's own "out"
    # it faced away from every lamp and showed as a black inner layer (John, 2026-11-28). A fibre is lit from all round
    # its axis (Kajiya & Kay 1989) and what darkens the far half is the hair in front, which the baked transmittance
    # holds: so the far half's "out" is mirrored to the outside. The ponytail, one bundle, never shows its far side.)
    inward = np.minimum((lo_u * oh).sum(1, keepdims=True), 0.0)
    lo_u = lo_u - 2 * inward * oh
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
                 look=dict(spec1_hue=LOOK_HUE, glint_hue=LOOK_HUE, light_floor=LOOK_LIGHT_FLOOR, inner_dark=LOOK_INNER_DARK,
                           mid_dark=LOOK_MID_DARK, view_side=True, cast_shadow=True, **LOOK_TONE),
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
