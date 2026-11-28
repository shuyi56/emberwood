"""Long layered waves with a centre part and a curtain fringe, grown from scratch from a reference photo's measurements
(John 2026-11-28: "Abandon this style and redo from scratch from reference photo"; the photo is
hair-rebuild/wavy-photo/reference.jpg). No guide of an older groom and no creator strand is read: every fibre is
combed here from its own root.

The photo, measured by hand on a 20 px grid (head height 240 px, skull top to chin; z in head heights above the chin,
x from the part). MEASURED, one photo, front view only, the head a little turned:
  part            on the midline, from the front hairline to the crown
  fringe          a curtain: thin, open at the part, its inner ends at the brow (z 0.54 at |x| 0.12), lengthening
                  outward to the cheekbone (z 0.40 at |x| 0.30), where it runs into the face-framing layers
  face framing    shorter pieces beside the face, ending between the chin and the collarbone (z 0.08 to -0.25)
  lengths         ends at z -0.76, layered (the top layers shorter), the front lengths in front of the shoulders
                  with their inner edge 0.2 from the midline on the chest
  volume          hair's half-width 0.41 at the top of the forehead, 0.45 at the eyes (the skull's is about 0.34),
                  0.54 at the mouth, 0.63 at the shoulders; 0.04 above the skull at the top
  wave            straight to the cheek (z about 0.38), then loose S waves about 0.33 long and 0.04 high, in ribbons
                  0.10 to 0.16 wide (22 to 35 mm: a stylist's 1 to 1.5 in section), the ends flicked and parted

How a fibre is combed (a head line; the lengths are then hung and waved by wavy_flow.hang_locks):
  - it starts at its root and runs over the scalp along gravity projected on the skin, pushed away from the part where
    the skin is flat (the top), and stands off the skin by a lift that grows along it (hair lies on the hair under
    it): this is the volume above;
  - where the skin turns under (below the skull's widest height) it leaves the head and hangs;
  - the fringe section (a triangle behind the front hairline, deepest at the part) is combed down and out over the
    forehead and cut on the curtain line; the hair just behind the hairline at the temples is cut as face framing.

    build(F, surf, n) -> paths (list of (k, 3) at SEG mm), hairs per fibre, lock key, kind (0 length, 1 fringe, 2 framing)
"""
import math
import os
import numpy as np
from scipy.spatial import cKDTree

import scalp_field as SF

MM = SF.MM
SEG_MM = 2.5
N_HAIRS = 90000.0           # DRAFT: hairs on the scalp (the older grooms' budget; a scalp carries about 100,000, Robbins)
RELEASE_Z = SF.CENTRE[2]    # the skin turns under below the skull's centre height: the hair leaves the head there
PART_PUSH = 0.45            # DRAFT: the push away from the part against gravity along the skin (1: equal)
FRINGE_ARCH_MM = (22.0, 8.0, 7.0)   # DRAFT: the arch's height off the root, how far it comes in again over the forehead, and the
                            # flick at the end (a 35 mm brush's radius is 17.5)
FRINGE_PUSH = (0.45, 2.20)  # the same for the fringe, at the hairline and from the brow down: its inner edge runs from
                            # the part to the brow at |x| 0.12 and on out to the cheekbone at 0.30 (MEASURED)
VOLUME_MM = 30.0            # DRAFT: a hanging lock keeps its stand-off from the skin and body up to this (hair lies on the
                            # hair under it); wavy_flow.hang_locks' `volume`
LIFT_MM = (1.0, 28.0)       # MEASURED (photo): hair stands 9 mm off the skull at the top and about 24 mm at the eyes
LIFT_PER_MM = 0.12          # DRAFT: lift gained per mm combed
FRINGE_LIFT_MM = 12.0       # DRAFT: the fringe stands off the forehead (ghd: 'lift at the base'; the photo shows its shadow
                            # under it). At 5 mm it lay on the skin
# Sectioning (hairfinder, the seven-section parting: top, two sides, two crown, two nape, parted along the parietal ridge
# and from ear to ear; Red Seal study guide 6.4: sides 'from the top of the ear forward to the face frame', top 'from
# the parietal ridge up to the front hairline', a pivot section at the crown with 'a radial or pivoting parting', nape
# 'below the occipital bone'). Direction: a long layered cut's sides are over-directed 'slightly backward' (NY study
# guide 8.5), each bevel brought 'around the head shape' (Sam Villa), the face frame 'flowing away from the face'.
# No source says the side hair is combed straight down; straight down is only the uncut 'natural fall'.
TOP_LIFT_MM, ROOT_RISE_MM = 27.0, 14.0   # the hair's height over the skull at the top, and the length from the root over
                            # which it rises to it. John 2026-11-28: "raise the hair more it's still too flat it should have
                            # more arching at the roots". This overrides the measurement (9 mm, 0.04 HH, read off the photo's
                            # top outline); the rise is longer (30 to 42 mm) so the taller arch bends no tighter
# THE ARCH (John 2026-11-28, the crown circled in the side view: "get the proper arch research how to do it properly in
# procedural hair"). What the sources do:
#   lift is a ROTATION AT THE ROOT, not a stand-off: Houdini's Lift Guides "applies a rotation, similar to the Set
#     Direction tool" and its brush has "a target Angle for the lift" (SideFX docs, Guide Process: Lift; Guide Groom)
#   the hair is a VOLUME of layers stacked on the scalp, each strand traced through corresponding points of every layer
#     and joined with "C1 Catmull-Rom splines"; the artist shapes only "the external surface of the mesh volume"
#     (Yuksel, Schaefer, Keyser 2009, Hair Meshes)
#   the stylist's root lift: "overdirect each section to a 90-degree angle", dried over "a large round brush"
#     (Behind the Chair, 9 blowout tips); "lift sections of hair upwards as you dry" (Hershesons)
# So a hair leaves its root AT AN ANGLE and turns over at one constant bend onto the hair's outer surface, landing
# tangent to it: height T * (1 - (1 - s/L)^2) over the run L. Leaving angle atan(2T/L) = 47 degrees, bend radius
# L^2/(2T) = 56 mm (a large round brush and the hair's own spring). Before, the rise was a smoothstep: flat on the skin
# at the root, then a steep ramp, then flat again, three times the bend, so the top read as spikes, not arches.
ENV_Z = (1.00, 0.75)        # the surface eases from the top's height to the side's over this height, ending at the head's widest (at 0.62:
                            # round 4's outline had two lobes on top and pinched in at the ears)
OUTLINE_Z, OUTLINE_MAX_MM = 0.75, 12.0   # the side outline is held out from the head's widest (z 0.75) down, by at most this
PART_DIP, PART_WIDE_MM = 1.0, 45.0   # DRAFT: the outer surface's height at the part (a share), and where it is full
ENV_MM = (25.0, 14.0)       # the hair's outer surface over the skull: at the top and by the temple (John 2026-11-28: "raise the
                            # hair more", front and back alike; the photo's own top outline reads 9 mm, overridden). By the
                            # temple 12 (it was 23): "sides are too thick and not smooth curves" (John 2026-11-28)
SWEEP_IN_MM = 70.0          # DRAFT: the back sweep grows in over this length from the root
FLARE = (0.055, 0.08)        # MEASURED (the photo's half-widths 0.45 at the eyes, 0.54 at the mouth, 0.63 at the shoulders,
                            # and John's line stepping out at eye level): the fall stands out this much (HH) below the
                            # temple, and this much again by the shoulder; the outer layer most
SIDE_FLOW = dict(w=1.0, min=0.45, max=1.2, fwd=3.0)   # DRAFT: how far the sides' comb takes the bangs' flow, the back sweep kept, the slope's cap
BACK_SWEEP = 0.9            # DRAFT: the pull back along the skin against gravity, at the face frame
FALL_CARRY = 0.12           # DRAFT (HH): below the head the hair keeps turning the way it left for about this drop
FRINGE_HANG_Z = 0.66        # DRAFT: below the forehead's bulge the fringe hangs free of the skin
FRINGE_RAD = 0.70           # DRAFT: the fringe section's half-angle round the head
FRINGE_DEPTH = 0.30         # DRAFT (HH): the bang section's depth behind the hairline at the part, falling to 0 at FRINGE_RAD. 66 mm (John 2026-11-28: "begin the bangs farther back"; at 22 mm the bang was a thin strip at the hairline, at 44 mm it still started in front of the lift)
FRAME_RAD, FRAME_DEPTH = 1.45, 0.20     # DRAFT: face framing: in front of the ears, this far behind the hairline
LAYER_END = (0.02, -0.30)   # MEASURED on John's drawing over the photo (2026-11-29): the first layer hooks in at the jaw, the
                            # second ends at the shoulder line (z from the chin, HH)
LAYER_PHASE = (0.6, 2.4)    # DRAFT: each layer's wave phase (rad): the second half a wave on, so it bulges where the first hooks in
LAYER_HOOK = (12.0, 0.35)    # DRAFT: mm a layer's end turns in, and the share of its fall it turns over
LAYER_RAG, LAYER_BLEND, LAYER_NEXT = 0.06, 0.10, -0.50   # DRAFT: a tier's ragged ends, and the grade behind the tiers into the lengths
FRAME_END = (-0.25, 0.08)   # MEASURED: where the face-framing pieces end
HEM_Z = -0.76               # MEASURED: the ends
TOP_SHORTER, GRADE, RAGGED = 0.22, 0.25, 0.08   # DRAFT (HH): the top layer ends this much higher (layers), the pieces by the
                            # face up to this much higher again (Sam Villa: 'short in the front and long in the back'), and
                            # each hair up to this much short of its layer's line
TAKE = 1.3                  # CALIBRATED (first solve): the waves and the lie on the shoulder take up a quarter of the
                            # hanging length: ends cut for z -0.76 hung at -0.38 to -0.47; at 1.5 the locks splayed over the shoulders
FRINGE_FEATHER = 0.05       # DRAFT (HH): the fringe's ends are point cut over this much, not a ruled line
LOCKS_PER_SIDE = 50         # a lock is a stylist's section, 22 to 35 mm of the perimeter (MEASURED ribbon width)
WAVE = dict(start_z=0.56, full_z=0.42, lam_hh=0.43, amp_hh=0.045)    # TRACED on the photo's two big locks (bulges 100 px apart
                            # at 234 px a head: 0.43 HH; swing 13 px: 0.055 HH peak, 0.045 kept). Before: 0.33, 0.04


def curtain_z(ax):
    """The fringe's cut line: where a fringe hair at |x| ends (MEASURED: brow at 0.12, cheekbone at 0.30)."""
    return np.clip(0.59 - (ax - 0.17) / 0.10 * 0.12, 0.47, 0.62)       # (re-MEASURED on the face crop: the inner ends at the
                                                                       #  brow's outer half, |x| 0.17, z 0.59; the outer at the
                                                                       #  temple, |x| 0.27, z 0.47, outside the eye's corner)


def wave(th, rng=None):
    """wavy_flow's wave by angle: (wavelength HH, phase, section). One phase per side (a brushed-out set is one wave
    pattern), the two sides mirrored; the caller jitters it per lock."""
    th = np.atleast_1d(th)
    jit = np.random.default_rng(4).normal(0.0, 0.5, len(th))             # DRAFT: a lock's own phase, about a twelfth of a wave
    return np.full(len(th), WAVE['lam_hh']), np.where(th > 0, 0.6, 0.0) + jit, np.zeros(len(th), int)


def build(F, surf, n=24000, seed=11, log=print):
    rng = np.random.default_rng(seed)
    per = SF.point_area(F)
    p = F['density'] * per
    p = p / p.sum()
    idx = rng.choice(len(p), size=n, p=p)
    root = F['P'][idx] + (rng.random((n, 3)) - 0.5) * 0.004
    hairs = np.full(n, N_HAIRS / n)
    th = SF.theta(root)
    # the front hairline: the lowest scalp point in each slice of angle, in front of the ears
    scalp = F['P'][F['density'] > 0]
    ths = SF.theta(scalp)
    edge = []
    for a in np.linspace(-1.9, 1.9, 96):
        m = np.abs(ths - a) < 0.03
        if m.any():
            edge.append(scalp[m][np.argmin(scalp[m][:, 2])])
    d_hl = cKDTree(np.array(edge)).query(root)[0]
    kind = np.zeros(n, int)
    fr = (np.abs(th) < FRINGE_RAD) & (d_hl < FRINGE_DEPTH * (1 - (np.abs(th) / FRINGE_RAD) ** 2))
    kind[fr] = 1
    kind[(kind == 0) & (np.abs(th) < FRAME_RAD) & (d_hl < FRAME_DEPTH)] = 2
    high = root[:, 2] > np.median(root[(kind == 0) & (root[:, 2] > RELEASE_Z), 2])     # the top layer: rooted above the under layer
    z_end = HEM_Z + TOP_SHORTER * high + RAGGED * rng.random(n)
    # (John 2026-11-29: "read up more on layering", with the photo's layers drawn: a bulge to the jaw that hooks in, a
    # second to the shoulder. A layered cut is over-directed forward: "the hair pulled from gets longer", so the lengths
    # grow going back from the face. The face frame starts "between the cheekbone and the jawline" and the section
    # behind it is cut "to about the length of the base of the neck" (Sam Villa, long layering and face framing); the
    # centre back keeps the perimeter. So the ends are tiers by the root's distance behind the front hairline, each
    # tier ragged by LAYER_RAG, and the hair behind the tiers grades into the full length over LAYER_BLEND.)
    m2 = kind == 2
    tier = np.minimum((d_hl[m2] / FRAME_DEPTH * len(LAYER_END)).astype(int), len(LAYER_END) - 1)
    z_end[m2] = np.array(LAYER_END)[tier] + LAYER_RAG * (rng.random(m2.sum()) - 0.5)
    m0 = (kind == 0) & (np.abs(th) < FRAME_RAD) & (d_hl < FRAME_DEPTH + LAYER_BLEND)
    w_ = SF.sm(FRAME_DEPTH, FRAME_DEPTH + LAYER_BLEND, d_hl[m0])
    z_end[m0] = LAYER_NEXT + (z_end[m0] - LAYER_NEXT) * w_
    z_end = RELEASE_Z - (RELEASE_Z - z_end) * TAKE
    feather = FRINGE_FEATHER * rng.random(n)
    z_end[kind == 1] = -9.0                                            # (cut on the curtain line below)
    layer = np.where(high, 0.6, 0.2) + 0.4 * rng.random(n)             # (where in the hair's thickness the fibre lies)
    hand = np.full(n, -1)                                              # (the node where a fibre leaves the head)
    lift_cap = np.where(kind == 1, FRINGE_LIFT_MM, LIFT_MM[1]) * layer + LIFT_MM[0]
    side = np.where(root[:, 0] >= 0, 1.0, -1.0)
    crown_y = SF.CENTRE[1] + 0.10                                      # DRAFT: the part ends at the crown
    pos = root.copy()
    live = np.ones(n, bool); free = np.zeros(n, bool)
    out = [[r] for r in root]
    step = SEG_MM * MM
    g = np.array([0.0, 0.0, -1.0])
    for it in range(int(3.4 / step)):
        s_mm = it * SEG_MM
        I = np.where(live)[0]
        if not len(I):
            break
        P = pos[I]
        q = P - SF.CENTRE
        r = np.linalg.norm(q, axis=1)
        U = q / r[:, None]
        on = ~free[I]
        new = P + step * g                                              # hanging: straight down
        if on.any():
            J = I[on]; Uo = U[on]
            # along the skin: gravity, and the push away from the part (sideways in front of the crown, outward from
            # the crown behind it), both laid on the skin (the radius from the skull's centre stands for its normal)
            b = np.zeros((len(J), 3)); b[:, 0] = side[J]
            behind = pos[J, 1] > crown_y
            rb = pos[J][:, :2] - np.array([0.0, crown_y]); rb /= np.maximum(np.linalg.norm(rb, axis=1, keepdims=True), 1e-9)
            b[behind, 0] = rb[behind, 0]; b[behind, 1] = rb[behind, 1]
            push = np.where(kind[J] == 1, FRINGE_PUSH[0] + (FRINGE_PUSH[1] - FRINGE_PUSH[0]) * SF.sm(0.80, 0.56, pos[J, 2]), PART_PUSH)
            d = g[None] + push[:, None] * b
            d[kind[J] == 1, 1] -= 1.0                                   # (the fringe is combed forward over the hairline)
            # the top and the sides are directed diagonally back, away from the face (John 2026-11-28: "research how to
            # section properly, it shouldn't be straight here"; sections and directions in the note above BACK_SWEEP):
            # most at the face frame, none at the ear-to-ear parting, where the crown's pivot takes over
            wf = SF.sm(SF.AXIS[1] + 0.04, -0.24, pos[J, 1]) * (kind[J] != 1) * ~behind
            d[:, 1] += BACK_SWEEP * wf * SF.sm(0.0, SWEEP_IN_MM, s_mm)       # ("more curved from top": the line leaves the
                                                                        #  part square to it and bows back as it runs)
            # (John 2026-11-29, flow lines drawn on the photo from the part over the sides of the head: "try flow lines on
            # the sides, they follow the same flow as the bangs". In front of the ears the comb's heading in the front view
            # is the bangs' own: the slope of the traced outer bang line at that height (TRACE_OUT), steep by the part and
            # turning out toward the temple, laid on the skin. The back sweep there is kept at SIDE_FLOW['back'] of
            # itself, so the side hair runs beside the bangs instead of crossing under them.)
            if SIDE_FLOW['w'] > 0:
                tz = np.array([t_[1] for t_ in TRACE_OUT]); tx = np.array([t_[0] for t_ in TRACE_OUT])
                zm = 0.5 * (tz[1:] + tz[:-1]); sl = (tx[1:] - tx[:-1]) / (tz[:-1] - tz[1:])
                slope = np.minimum(np.interp(pos[J, 2], zm[::-1], sl[::-1]), SIDE_FLOW['max'])
                # (the heading is set in the FRONT view, as the bangs' is, and its depth solved so the step stays on the
                # skin: over the top of the head a line can only come down in the front view by running forward, so the
                # sides come forward over the temple beside the bangs. A plain projection sent them sideways off the
                # part and straight down the side of the head: an inverted U in the front view, drawn in round 23's
                # first overlay.)
                slope = np.maximum(slope, SIDE_FLOW['min'])
                uy = np.minimum(Uo[:, 1], -0.08)
                dy = np.clip(-(side[J] * slope * Uo[:, 0] - Uo[:, 2]) / uy, -SIDE_FLOW['fwd'], 0.3)
                df = np.c_[side[J] * slope, dy, -np.ones(len(J))]
                df /= np.linalg.norm(df, axis=1, keepdims=True)
                dn_ = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                wq = (SIDE_FLOW['w'] * wf)[:, None]
                d = dn_ * (1 - wq) + df * wq
            d = d - (d * Uo).sum(1, keepdims=True) * Uo
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            Q = pos[J] + step * d
            qq = Q - SF.CENTRE
            Un = qq / np.linalg.norm(qq, axis=1, keepdims=True)
            s_mm = (it + 1) * SEG_MM                                     # (the new node's own length: the first step rises too)
            lift = np.minimum(LIFT_MM[0] + LIFT_PER_MM * s_mm * layer[J], lift_cap[J]) * MM
            # (John 2026-11-28, lines drawn on the photo: "hair from roots shouldn't lie flat on the scalp also should be
            # more curved from top". The stand-off rises from the root as an arc over ROOT_RISE_MM, the lift a round
            # brush gives at the base, to the photo's height: TOP_LIFT_MM over the top, LIFT_MM[1] by the temple. Before,
            # it grew LIFT_PER_MM per mm combed, so the first 50 mm lay on the skin.)
            # ONE outer surface over the whole head (John 2026-11-28: "why is the light blue lifted so much more than dark
            # blue": the lift was a base and two patches, front +12 mm and back +8 mm, and the section over the ear sat in the
            # gap between them). Its height depends on the height on the head alone: ENV_MM[0] over the top, ENV_MM[1] by the
            # temple, the same in front, over the ears and behind. A hair's layer sets how deep under it the hair lies.
            capz = (ENV_MM[0] + (ENV_MM[1] - ENV_MM[0]) * SF.sm(*ENV_Z, Q[:, 2])) * (0.45 + 0.55 * layer[J])
            # (John 2026-11-29, circling the top of the head from the front and from behind: "fix these ridges". The
            # outer surface stood at its full height right up to the part, where the hair starts on the skin: two humps
            # and a cleft 17 mm deep within 12 mm of the part (measured, tools/gauges/hair/part_profile.py; a shorter
            # root rise barely changed it, 36 to 14 to 7 mm, because a hair cannot rise faster than straight up).
            # The surface now comes down toward the part by itself: PART_DIP of its height at the part, full height
            # PART_WIDE_MM to the side, one shallow saddle and no step.)
            capz = capz * (PART_DIP + (1 - PART_DIP) * SF.sm(0.0, PART_WIDE_MM, np.abs(Q[:, 0]) / MM))
            # (one smooth outline down the sides: below the head's widest the skull narrows to the ear, and the hair
            # laid on it at a set height followed it in, so the outline pinched at ear height (round 4: 4.6 mm in from
            # z 0.75 to 0.55). The hair over the sides is lifted by that narrowing, so the outline falls straight.)
            Hq = Q[:, 2] < OUTLINE_Z
            if Hq.any():
                th_ = SF.theta(Q[Hq])
                ref = np.c_[SF.AXIS[0] + np.sin(th_), SF.AXIS[1] - np.cos(th_), np.full(Hq.sum(), OUTLINE_Z)] - SF.CENTRE
                ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
                pr = SF.CENTRE + ref * surf.radius(ref)[:, None]
                ps = SF.CENTRE + Un[Hq] * surf.radius(Un[Hq])[:, None]
                rc = lambda X: np.hypot(X[:, 0] - SF.AXIS[0], X[:, 1] - SF.AXIS[1])
                horiz = np.maximum(np.hypot(Un[Hq, 0], Un[Hq, 1]), 0.5)
                fill = np.clip(rc(pr) - rc(ps), 0, OUTLINE_MAX_MM * MM) / horiz * np.abs(np.sin(th_))
                capz = capz.copy(); capz[Hq] = capz[Hq] + fill / MM * SF.sm(OUTLINE_Z, OUTLINE_Z - 0.03, Q[Hq, 2])
            lift = (LIFT_MM[0] + capz * (1 - (1 - min(s_mm / ROOT_RISE_MM, 1.0)) ** 2)) * MM
            # the fringe bends in two planes (John 2026-11-28: "bangs should also be 3d twisting along 2 planes like the
            # picture"). Out of the forehead: an arch up off the root over the hairline (ghd: "lift at the base", on a
            # 35 mm brush), in again over the forehead, and the end flicked out ("flick the brush away from the face").
            # Round the head: down first, then out to the temple (the push below). The two together are a twisting S.
            fz = Q[:, 2] - (curtain_z(np.abs(Q[:, 0])) + feather[J])
            lf = (1.0 + FRINGE_ARCH_MM[0] * SF.sm(0.0, 25.0, s_mm) - FRINGE_ARCH_MM[1] * SF.sm(30.0, 75.0, s_mm)
                  + FRINGE_ARCH_MM[2] * SF.sm(0.07, 0.0, fz)) * (0.8 + 0.4 * layer[J]) * MM
            lift = np.where(kind[J] == 1, lf, lift)
            Qs = SF.CENTRE + Un * (surf.radius(Un) + lift)[:, None]
            # (below the forehead the fringe hangs as a curtain over the brow and the eye's corner: it does not follow
            # the skin into the socket, it only stays clear of it)
            hangs = (kind[J] == 1) & (pos[J, 2] < FRINGE_HANG_Z)
            r_was = np.linalg.norm(pos[J] - SF.CENTRE, axis=1)
            Qh = SF.CENTRE + Un * np.maximum(r_was - 0.2 * step, surf.radius(Un) + lift)[:, None]
            Q = np.where(hangs[:, None], Qh, Qs)
            new[on] = Q
            rel = J[(Q[:, 2] < RELEASE_Z) & (kind[J] != 1)]                 # (the fringe lies on the forehead to its cut)
            free[rel] = True
            for i in rel:
                hand[i] = len(out[i])
            lg = rel[kind[rel] == 0]                                        # (graded shorter toward the face)
            z_end[lg] += GRADE * TAKE * SF.sm(1.4, 0.5, np.abs(SF.theta(pos[lg])))
        pos[I] = new
        for i, x in zip(I, new):
            out[i].append(x)
        ax = np.abs(new[:, 0])
        done = (new[:, 2] <= z_end[I]) | ((kind[I] == 1) & (new[:, 2] <= curtain_z(ax) + feather[I]))
        live[I[done]] = False
    paths = [np.array(o) for o in out]
    # the locks: the fibres that hang past the wave's start, gathered by where they pass it, each side on its own
    lockkey = np.full(n, -1)
    H = np.full((n, 3), np.nan)
    for i, pth in enumerate(paths):
        j = np.where(pth[:, 2] < WAVE['start_z'])[0]
        if len(j) and (len(pth) - j[0]) * SEG_MM > 40.0:
            H[i] = pth[j[0]]
    from scipy.cluster.vq import kmeans2
    nk = 0
    for sg in (1, -1):
        m = np.where(~np.isnan(H[:, 0]) & (np.sign(H[:, 0]) == sg))[0]
        _, lab = kmeans2(H[m], LOCKS_PER_SIDE, seed=3 + sg, minit='++')
        lockkey[m] = nk + lab; nk += LOCKS_PER_SIDE
    log('photo waves: fibres %d (fringe %d, framing %d), hanging %d in %d locks; lengths mm median %.0f'
        % (n, (kind == 1).sum(), (kind == 2).sum(), (lockkey >= 0).sum(), nk, np.median([len(p_) for p_ in paths]) * SEG_MM))
    return paths, hairs, lockkey, kind, dict(hand=hand, high=high, edge=np.array(edge), root=root, layer=layer, z_end=z_end)


def aim(coll):
    """Where a front lock is headed (wavy_flow.hang_locks' `creator` hook as a function of the lock's root and angle):
    in front of the shoulder onto the chest, the locks nearest the face innermost (MEASURED: inner edge 0.2 from the
    midline, outer 0.6, at z -0.15), 8 mm clear of the body. Locks behind the ears hang as they fall."""
    def f(root, th):
        if abs(th) > 1.75:
            return None
        x = np.sign(th) * (0.22 + 0.36 * abs(th) / 1.75)
        z = -0.15
        y = -0.8
        for _ in range(200):                                            # (in from the front until 8 mm off the body)
            d, _n = coll(np.array([[x, y, z]]) / MM, np.array([8.0]))
            if d[0] > 0:
                break
            y += 0.005
        return np.array([x, y - 0.005, z]), None
    return f


# ---------------------------------------------------------------------------------------------------------------
# The flow field (John 2026-11-28: "the hair should be 3d it should bend and twist so it falls on the front while
# framing the sides and providing depth ... look at the pony tail code for how to do proper locks for falling hair ...
# first show me the flow field before anything"). The ponytail's recipe (ponytail_flow.py, bangs_creator_asset.py
# tail_locks, solvers/hair/tail_lock_rods.py) carried over to loose hair:
#   ponytail: one basin, every root funnels over the scalp to ONE outlet (the tie), is handed over onto the bundle,
#             and below it the hair is 36 thick locks, each its own 3D rod, a ribbon that rolls and opens.
#   here:     the scalp is parted into SECTIONS as a stylist takes them (rows from the bottom up, each row in sections
#             25 to 38 mm wide, the upper row's sections laid over the joints of the lower row's: bricklay). Each
#             section is a basin with its own outlet on the line where the hair leaves the head, and below its outlet
#             it is one thick lock: a 3D rod with its own curl, its own turn, kept a lock's thickness off its neighbours.
# Sources: Sam Villa (sections 1 to 1.5 in; "short in the front and long in the back"; the face frame "flowing away
# from the face"), Red Seal study guide (bricklay), ghd (a curtain fringe: lift at the root on a 35 mm round brush,
# "flick the brush away from the face"; the pieces beside it curled "away from the face"), Wella (the next section
# curled the other way: "continue this alternating pattern"; the tong held vertical, starting below the root),
# 80.lv Brown 2025 (layers with space left between them "so shadows create depth"), Miller et al. 2014 PRL 112 068103
# (a naturally curved hair under its weight: planar below w/(B k^3) = 0.239... at this curl, 0.15, it is a 3D helix).
SECTORS = 9                 # sections a side in a row: 31 mm of the perimeter each (SOURCED: 25 to 38 mm)
GATHER = 0.75               # DRAFT: a section's hair at its outlet takes this share of the section's own width
OUT_LIFT_MM = {0: 5.0, 1: 17.0, 2: 9.0, 3: 6.0}   # DRAFT: an outlet's stand-off by row: under, top (it lies on the under
                            # row: the space between the layers), face framing, fringe
CURL_ON_MM = (20.0, 110.0)  # MEASURED: straight to the cheek, the full wave by the jaw
FRINGE_K = (1 / 70.0, 1 / 35.0)   # per mm, root to tip. The tip's is a 35 mm brush's twice over (SOURCED ghd; the set relaxes: DRAFT)
PACKING = 0.1               # the ponytail's TAIL_LOCK_PACKING
LAYER_NAME = {0: 'under row', 1: 'top row', 2: 'face framing', 3: 'fringe'}


def field(F, surf, n=24000, seed=11, log=print):
    """Sections, outlets and the lines over the scalp. Returns a dict: root, hairs, kind, basin (per fibre, -1: none),
    line (list of (k, 3): root to outlet over the skin), hang_mm (the hair's length below its outlet), u (its place
    across the outlet, mm), and per basin: side, layer, sector, centre, th (angle), half_mm (the outlet's half-width)."""
    paths, hairs, _, kind, x = build(F, surf, n, seed, log=lambda t: None)
    root, hand, high = x['root'], x['hand'].copy(), x['high']
    # the fringe leaves the head at the hairline
    eth = SF.theta(x['edge']); o = np.argsort(eth)
    for i in np.where(kind == 1)[0]:
        p = paths[i]
        below = np.where(p[:, 2] < np.interp(SF.theta(p), eth[o], x['edge'][o, 2]) - 0.01)[0]
        hand[i] = below[0] if len(below) else -1
    ok = np.array([hand[i] >= 0 and (len(paths[i]) - hand[i]) * SEG_MM > 30.0 for i in range(n)])
    rel = np.array([paths[i][min(hand[i], len(paths[i]) - 1)] if ok[i] else root[i] for i in range(n)])
    th = SF.theta(rel)
    side = np.where(th >= 0, 1, 0)
    layer = np.where(kind == 1, 3, np.where(kind == 2, 2, np.where(high, 1, 0)))
    u = np.abs(th) / (np.pi / SECTORS)
    sec = np.clip(np.where(layer == 1, np.floor(u + 0.5), np.floor(u)), 0, SECTORS).astype(int)   # (the top row bricklaid)
    for lay in (2, 3):                                                   # (two pieces a side: inner and outer)
        for sd in (0, 1):
            m = ok & (layer == lay) & (side == sd)
            if m.any():
                sec[m] = (np.abs(th[m]) > np.median(np.abs(th[m]))).astype(int)
    key = np.where(ok, side * 1000 + layer * 100 + sec, -1)
    keys = np.unique(key[key >= 0])
    basin = np.full(n, -1)
    B = dict(side=[], layer=[], sector=[], centre=[], th=[], half_mm=[], hairs=[])
    tgt = rel.copy()
    upos = np.zeros(n)
    for b, k in enumerate(keys):
        m = np.where(key == k)[0]
        basin[m] = b
        thc = float(np.median(th[m]))
        lay = int(layer[m[0]])
        # each hair's place on the outlet: its own angle drawn in to the section's middle (rank kept), its own height
        tt = thc + GATHER * (th[m] - thc)
        r = np.hypot(rel[m, 0] - SF.AXIS[0], rel[m, 1] - SF.AXIS[1])
        q = np.c_[SF.AXIS[0] + r * np.sin(tt), SF.AXIS[1] - r * np.cos(tt), rel[m, 2]]
        U = q - SF.CENTRE; U /= np.linalg.norm(U, axis=1, keepdims=True)
        depth = OUT_LIFT_MM[lay] + 3.0 * (np.random.default_rng(int(k)).random(len(m)) - 0.5)
        tgt[m] = SF.CENTRE + U * (surf.radius(U) + depth * MM)[:, None]
        rm = float(np.median(r))
        upos[m] = (tt - thc) * rm / MM
        B['side'].append(int(side[m[0]])); B['layer'].append(lay); B['sector'].append(int(sec[m[0]]))
        B['centre'].append(tgt[m].mean(0)); B['th'].append(thc)
        B['half_mm'].append(float(np.percentile(np.abs(upos[m]), 90))); B['hairs'].append(float(hairs[m].sum()))
    # the lines over the scalp: the shortest way over the skull from the root to its place on the outlet (the
    # ponytail's funnel: arcs to one outlet do not cross), lifting to the outlet's stand-off
    lines, hang = [], np.zeros(n)
    for i in range(n):
        if basin[i] < 0:
            lines.append(paths[i]); continue
        U0 = root[i] - SF.CENTRE; r0 = np.linalg.norm(U0); U0 /= r0
        U1 = tgt[i] - SF.CENTRE; r1 = np.linalg.norm(U1); U1 /= r1
        om = math.acos(float(np.clip(U0 @ U1, -1, 1)))
        k = max(2, int(om * 0.5 * (r0 + r1) / (SEG_MM * MM)) + 1)
        t = np.linspace(0, 1, k)
        if om < 1e-4:
            Us = np.repeat(U0[None], k, 0)
        else:
            Us = (np.sin((1 - t) * om)[:, None] * U0 + np.sin(t * om)[:, None] * U1) / math.sin(om)
        e = t * t * (3 - 2 * t)
        skin = surf.radius(Us)
        rr = np.maximum(skin + 1.0 * MM, r0 + (r1 - r0) * e)             # (never under the skin on the way)
        lines.append(SF.CENTRE + Us * rr[:, None])
        hang[i] = (len(paths[i]) - hand[i]) * SEG_MM
    for k_ in B:
        B[k_] = np.array(B[k_])
    log('field: %d fibres, %d sections (%s); hairs per section median %.0f; outlet half-width median %.0f mm'
        % (n, len(keys), ', '.join('%s %d' % (LAYER_NAME[l], (B['layer'] == l).sum()) for l in range(4)),
           np.median(B['hairs']), np.median(B['half_mm'])))
    return dict(root=root, hairs=hairs, kind=kind, basin=basin, line=lines, hang_mm=hang, u=upos, tgt=tgt, B=B)


def locks(Fd, coll, log=print, seed=9, gain=1.0):
    """Each section's lock below its outlet as a 3D rod (tail_lock_rods.solve): its launch, its natural curl (a helix
    for the lengths, turning the other way in the next section; a C away from the face for the fringe), gravity,
    contact with the other locks and with the head and body. Returns the rods (x HH, R, s mm), each lock's ribbon turn
    psi(s) and the report."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'solvers', 'hair'))
    import tail_lock_rods as TR
    rng = np.random.default_rng(seed)
    B = Fd['B']; nl = len(B['th'])
    side = np.where(B['side'] > 0, 1.0, -1.0)
    roots = B['centre'].copy()
    length = np.array([np.percentile(Fd['hang_mm'][Fd['basin'] == k], 98) for k in range(nl)])
    nh = B['hairs']
    aimf = aim(coll)
    tang = np.zeros((nl, 3)); d1 = np.zeros((nl, 3))
    out_h = np.c_[roots[:, 0] - SF.AXIS[0], roots[:, 1] - SF.AXIS[1], np.zeros(nl)]
    out_h /= np.linalg.norm(out_h, axis=1, keepdims=True)
    for k in range(nl):
        if B['layer'][k] == 3:        # the fringe: lifted forward off the hairline, then down and out
            tv = np.array([0.45 * side[k], -0.6, -0.65]); ref = np.array([0.0, 0.0, 1.0])
        else:
            tv = np.array([0.0, 0.0, -1.0]) + 0.15 * out_h[k]
            a_ = aimf(roots[k], B['th'][k]) if abs(B['th'][k]) < 1.2 else None
            if a_ is not None:        # in front of the ears: headed for the chest in front of the shoulder
                to = a_[0] - roots[k]
                tv = 0.5 * tv / np.linalg.norm(tv) + 0.5 * to / np.linalg.norm(to)
            ref = out_h[k]
        tang[k] = tv / np.linalg.norm(tv)
        d = ref - (ref @ tang[k]) * tang[k]
        d1[k] = d / np.linalg.norm(d)
    d2 = np.cross(tang, d1)
    frames = np.stack([d1, d2, tang], 2)
    lock_r = np.sqrt(nh * 0.035 ** 2 / PACKING)
    A = WAVE['amp_hh'] / MM; c = WAVE['lam_hh'] / MM / (2 * np.pi)
    fringe = B['layer'] == 3
    hand = np.where(B['sector'] % 2 == 0, 1.0, -1.0) * side * np.where(B['layer'] == 1, -1.0, 1.0)   # (SOURCED Wella: alternating;
    tau = np.where(fringe, 0.0, hand / math.sqrt(A * A + c * c))                                     #  mirrored left and right)
    psi0 = np.pi / 2 + np.where(fringe, 0.0, 0.4 * rng.standard_normal(nl))    # (the first bend is out from the head)
    phi = np.where(fringe, np.where(side > 0, 0.0, np.pi), psi0)
    Lb3 = TR.EI_HAIR / TR.W_HAIR

    def rest(s):
        grow = SF.sm(CURL_ON_MM[0], CURL_ON_MM[1], s)
        h = np.maximum(length[:, None] - s, 0.0)
        pre = 1.0 + h / (Lb3 * (2 * np.pi / (WAVE['lam_hh'] / MM)) ** 2)   # (wavy_flow's preload against the weight below)
        kh = gain * A / (A * A + c * c) * grow * pre
        kf = FRINGE_K[0] + (FRINGE_K[1] - FRINGE_K[0]) * np.clip(s / length[:, None], 0, 1)
        return np.where(fringe[:, None], kf, kh), tau, phi
    out = TR.solve(roots / MM, frames, length, nh, lock_r, head=coll, seed=seed, rest=rest, log=lambda t: log('  rods: ' + t))
    rep = dict(out['report'], lock_radius_mm_median=round(float(np.median(lock_r)), 1), length_mm_median=round(float(np.median(length))))
    log('locks %s' % rep)
    return dict(X=out['x'] * MM, R=out['R'], s=out['s'], tau=tau, psi0=psi0, length=length, lock_r=lock_r, report=rep)


COLS = {0: [(47, 111, 179), (29, 74, 128), (111, 163, 216)], 1: [(44, 156, 90), (122, 79, 201), (20, 120, 110)],
        2: [(194, 51, 122), (150, 30, 90)], 3: [(217, 138, 31), (180, 100, 10)]}


def draw(Fd, rods, head_pts, depth, path, views=((0, 4, 'front'), (40, 6, 'three quarter'), (90, 4, 'side'), (180, 4, 'back'), (0, 80, 'top')),
         every=6, px=330):
    """The field as lines over a clay head (the ponytail's sheet): each section's lines over the scalp in its colour,
    its outlet in red, and its lock below the outlet as a ribbon (its middle line and its two edges, so the turn shows)."""
    from PIL import Image, ImageDraw
    W, H = int(1.9 * px), int(2.5 * px)
    im = Image.new('RGB', (W * len(views), H + 40), (250, 249, 247))
    dr = ImageDraw.Draw(im)
    B = Fd['B']
    pc = head_pts[head_pts[:, 2] > -0.9]
    for v, (az, el, name) in enumerate(views):
        a, e = math.radians(az), math.radians(el)
        fwd = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)

        def P(X):
            return np.c_[v * W + W / 2 + (X @ right) * px, 40 + H * 0.36 - (X @ up - 0.5) * px]

        def hid(X):
            t = np.linspace(0.015, 1.3, 24)[None, :, None]
            Q = (X[:, None] + fwd[None, None] * t).reshape(-1, 3)
            return (depth(Q) > 0.003).reshape(len(X), -1).any(1)
        nrm = pc - SF.CENTRE; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        sh = np.clip(0.74 + 0.26 * (nrm @ (fwd + 0.5 * up - 0.4 * right) / 1.2), 0.5, 1.0)
        vis = ~hid(pc + nrm * 0.004)
        for (x, y), s_ in zip(P(pc)[vis], sh[vis]):
            g = int(225 * s_); dr.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=(g, int(g * 0.98), int(g * 0.95)))
        items = []

        def add(X, col, w):
            h = hid(X); q = P(X); z = X @ fwd
            for j in range(len(X) - 1):
                if not (h[j] or h[j + 1]):
                    items.append((0.5 * (z[j] + z[j + 1]), tuple(q[j]), tuple(q[j + 1]), col, w))
        for i in range(0, len(Fd['line']), every):
            b = Fd['basin'][i]
            if b < 0:
                add(Fd['line'][i][::2], (150, 150, 150), 1); continue
            cs = COLS[int(B['layer'][b])]
            add(Fd['line'][i][::2] if len(Fd['line'][i]) > 4 else Fd['line'][i], cs[int(B['sector'][b]) % len(cs)], 1)
        for k in range(len(B['th'])):
            cs = COLS[int(B['layer'][k])]; col = cs[int(B['sector'][k]) % len(cs)]
            X, R, s = rods['X'][k], rods['R'][k], rods['s'][k]
            M = R.shape[0]
            last = int(min(M, math.ceil(rods['length'][k] / max(s[1], 1e-6))))
            psi = rods['psi0'][k] + rods['tau'][k] * s[:M]
            ew = np.cos(psi)[:, None] * R[:, :, 0] + np.sin(psi)[:, None] * R[:, :, 1]
            hw = B['half_mm'][k] * MM * (1 - 0.5 * (s[:M] / max(s[M], 1.0)) ** 2)[:, None]     # (drawing in to a point)
            add(X[:last + 1], col, 3)
            add((X[:M] + ew * hw)[:last], col, 1); add((X[:M] - ew * hw)[:last], col, 1)
            for j in range(0, last, 4):                                     # (rungs: the ribbon's width, turning)
                add(np.stack([X[j] - ew[j] * hw[j], X[j] + ew[j] * hw[j]]), col, 1)
            o = Fd['tgt'][Fd['basin'] == k]
            o = o[np.argsort(Fd['u'][Fd['basin'] == k])][::max(1, len(o) // 12)]
            add(o, (224, 32, 32), 3)
        for z, p0, p1, col, w in sorted(items, key=lambda t: t[0]):
            dr.line([p0, p1], fill=col, width=w)
        dr.text((v * W + 12, 12), name, fill=(0, 0, 0))
    im.save(path)


# ---------------------------------------------------------------------------------------------------------------
# Field v2 (John 2026-11-28 on v1: "read the recipe and do it again the hair is falling all over the place"). The
# recipe (docs/hair/ponytail-recipe.md): one sink per basin, strands keep rank, no line turns back, the fall is ONE
# shape with the S wave "laid on the middle line" so that it "moves every strand alike"; locks come later, inside it.
# v1 gave each of 36 sections its own free rod. Here the hair below the head is two falls a side, built so the rules
# hold by construction:
#   front fall  hair that leaves the head in front of FALL_SPLIT: it frames the face to the jaw, closes over the side
#               of the neck and lies on the chest in front of the shoulder (MEASURED: inner edge 0.2, outer 0.6)
#   back fall   the rest: over the back of the shoulder and down the back
# A line is (angle round the body, height): its angle moves from where it leaves the head to its place in the fall by
# a map that is monotonic in the angle (rank kept), its radius is the tent from the head to the body and then the body
# itself plus its depth in the stack, and the S wave is one pattern a side whose phase changes slowly round the head.
FALL_SPLIT = 2.20           # DRAFT: |theta| where the hair parts over the shoulder: well behind the ear, so that some of
                            # the back hair runs diagonally forward over the shoulder (John 2026-11-28; at 1.75 only the
                            # hair in front of the ear came forward, 38% of it)
FRONT_BAND = (0.55, 1.30)   # MEASURED: the front fall on the chest (x 0.2 to 0.6 from the midline at the collarbone)
BACK_BAND = (2.30, np.pi)   # DRAFT: the back fall, down the back inside the shoulder blades (at 1.95 it draped over the
                            # shoulders' tops)
SIDE_CUT_Z = -0.45          # the shoulder's height on the mannequin: where the side hair is cut
SIDE_CUT_A = (0.80, 1.20, 2.40, 2.75)   # DRAFT: the falls' places (angle round the body) cut to it: rising in over the front
                            # fall's outer part, full over the shoulder, out again behind it (the middle front and back long)
FAIR_MM = 25.0              # DRAFT: the course is faired over this length (half a bending length)
GATHER_Z = (0.50, -0.50)    # the falls start for their places at the cheek and ear and are there at the collarbone (John
                            # 2026-11-28: "twist more at the cheeks and ear section so it's less of a sharp turn in the chin
                            # and neck area"; from 0.30 the hair dropped straight past the ear and cut forward under the jaw)
FACE_KEEP = 1.15            # MEASURED (the photo's face crop): beside the face the hair's front edge is at the face's own
                            # outline, back by the ear (an earring shows): no hair lies on the cheek. The angle round the
                            # head of the cheekbone's back edge, just in front of the ear (the ear's middle is at 1.57)
FACE_KEEP_Z = (0.56, 0.20)  # the front edge swings back to it between the temple and the cheekbone (ghd, Sam Villa: the
                            # pieces by the face are dried and cut 'flowing away from the face')
STACK_MM = (1.5, 10.0)      # DRAFT: the fall's thickness: the innermost hair's stand-off and the outermost's
WAVE_OUT_Z = 0.46           # John 2026-11-29: "begin the curve of the sides earlier, first curve begins around the temples". The
                            # height at which the sides' wave is first fully out (it was at the release, z 0.56, where the wave
                            # has no size yet, so the first curve seen was a whole wave lower, at the chin); it grows in from
                            # the release (WAVE start_z) to full by the cheekbone (full_z)
BODY_AT_Z = -0.40           # where the tent from the head lands: the mannequin's shoulder (its neck is bare from the chin
                            # to z -0.3; the Warden's collar put the 'shoulder' at -0.15)


BANG_END = ((0.11, 0.60), (0.30, 0.44))   # where the bang's inner hair ends: |x| 0.11 over the brow's inner end (the first render
                            # at 0.17 left the middle of the forehead bare against the photo's narrow arch), and the outer
                            # MEASURED (face crop): where the bang's inner hair ends (|x|, z: the brow's outer
                            # half) and its outer hair (the temple, outside the eye's corner)
BANG_S = (0.06, 0.45, 0.05) # DRAFT: the S's handles: out from the root square to the part (HH), up from the end (a share of
                            # the hair's drop) and in from the end (HH): out, round and down, then out again
TRACE_IN = ((0.085, 0.757), (0.13, 0.612), (0.20, 0.50), (0.30, 0.42), (0.41, 0.385))   # (John's drawn line, 2026-11-29: 'the lines you drew are correct') # (John's line, 2026-11-29) # TRACED on the photo (|x| HH, z HH,
                            # the bang's inner edge from the hairline down to the brow's inner end; photo px (x - 262) / 232,
                            # (348 - y) / 232, both sides averaged)
TRACE_OUT = ((0.27, 0.72), (0.30, 0.62), (0.35, 0.52), (0.41, 0.455), (0.46, 0.43))   # (the old V's outer edge, bent out at the end) # the outer edge: out over the temple and down to the outer end
BANG_INNER = (0.4, 0.25)    # DRAFT: the innermost hair's first handle: this share of the sideways reach, and this share of its
                            # drop downward (the outermost: all sideways, none down)
BANG_SPAN = ((0.0, 0.45), (0.3, 0.95), (0.6, 1.0))   # where along the hair the arch rises, dips and flicks. GAUGED
                            # (flow_smooth, set hair, 35 mm brush): a 12 mm arch risen over the first quarter bent at 8.8 mm
                            # with 12% of the nodes under 17.5 mm; over the first half, 0.8% (and BANG_S[0] 0.10 to 0.06)


BANG_FLOW = False              # John 2026-11-28 'redraft bangs go back to flow': the fringe is the comb's own flow from its
                               # roots (lifted with the rest), not the traced sheet (bang_sheet, kept: BANG_FLOW = False)
BANG_BACK = 0.20            # HH
BANG_BEND = (0.035, 0.06, 0.25)   # DRAFT (HH out, HH up, the outermost lock's share)
BANG_RIBS, BANG_RIB_MM, BANG_FLICK_MM = 4, 4.0, 4.0   # DRAFT: locks across one side's bang, their depth, the ends' lift off the cheek
BANG_TAPER, BANG_RAG = 0.10, 0.22   # the lower edge's fall-off toward the sheet's two edges, and the ragged end (shares of the length)
BANG_FAIR = 3.0                # nodes (of k) the bang is faired over. GAUGED: rooted 66 mm back, unfaired, 2.2% of nodes
                               # bent tighter than the 35 mm brush (at the root and at the brow); faired over 3, 0.3%


BANG_3D = True              # John 2026-11-29: "get the bangs correct by using 3d flow solver where it's being pulled by the side
                            # along multiple axis". False: the two traced edges blended in the front view (TRACE_IN, TRACE_OUT)
BANG_AXES = dict(side0=0.10, side0u=0.55, side1=3.2, up=0.95, field=0.3, to=0.8, along=0.2)   # DRAFT: the pulls on a bang hair, as weights against gravity's 1:
                            # forward off the crown (dying over the first third), to the side (side0 from the root, growing
                            # by side1 toward the tip), and up at the tip (gravity cancelled by this share)
BANG_HOLD = (0.30, 0.55)    # DRAFT: the share of a bang hair's length held on the head, and where it is fully free
BANG_OVER_MM = 20.0
BANG_FULL, BANG_THICK_MM, BANG_ENV = 0.55, 6.0, 0.45   # DRAFT
BANG_DROP = (0.0, 0.12)      # DRAFT (HH): the bang leaves the outer surface from the hairline to this far below it
BANG_END_AT = 0.86           # the locks' tips: this share along the traced lines (past it the tips stood out beyond the side hair as spikes, 2026-11-29)
BANG_TRACED_FLOW = True      # the 3D flow steered along the traced lines in the front view (False: by BANG_AXES' weights)
BANG_CURL = (2.5, 60.0, 0.8)   # DRAFT: a lock's wave (mm), its wavelength (mm), and each hair's own wave (mm)
BANG_LAYER = (1.0, 1.0, 1.0, 1.0)   # DRAFT: each lock's length as a share of the flow's, from the part outward
BANG_LAYER_MM = 3.0                    # DRAFT: each lock lies this far over the next one out
BANG_LEN = (0.80, 0.66)     # DRAFT (HH): a bang hair's length, nearest the part and outermost (from the traced lines)


def _bang_march(root, surf, u, uc, sgn, k, side=None, front=None):
    """The bang as a flow in 3D. Every hair is marched from its root over the head: at each step its heading is the
    sum of the pulls on it (gravity down; forward, off the crown; to the side, growing along the hair, as a round brush
    draws a curtain bang away from the part; up at the tip, where the blow-dry turns the end), with the part normal to
    the head taken out so that the hair slides over the skin, and the step is then put back on the skin. The locks
    nearest the part are pulled most (BANG_BEND[2] the outermost's share). The sheet's stand-off, fairing and cut
    are bang_sheet's. Returns (m, k, 3)."""
    m = len(root)
    # (John 2026-11-29: "carve a few locks from the flow and force the ends to meet tips and layer bangs like the
    # picture". The flow is cut into BANG_RIBS locks by the roots' order; every hair of a lock has the lock's length
    # (its middle hair's), times the lock's layer (BANG_LAYER, the lock nearest the part shortest), so the locks end
    # one below the other; and over its last half each lock closes onto its mean line, fully, so its ends meet.)
    lk = np.minimum((u * BANG_RIBS).astype(int), BANG_RIBS - 1)
    lay = np.interp(lk / max(BANG_RIBS - 1, 1), np.linspace(0, 1, len(BANG_LAYER)), BANG_LAYER)
    ds = ((BANG_LEN[0] + (BANG_LEN[1] - BANG_LEN[0]) * uc) * lay / (k - 1))[:, None]
    inner = 1 - (1 - BANG_BEND[2]) * uc
    P = np.zeros((m, k, 3)); p = root.astype(float).copy(); P[:, 0] = p
    lkh = np.arange(m)
    for j in range(1, k):
        t = (j - 0.5) / (k - 1)
        q = p - SF.CENTRE; n = q / np.linalg.norm(q, axis=1, keepdims=True)
        # the pulls seen from the front: gravity down (let go at the tip by 'up') and the pull to the side, which
        # grows along the hair. The hair lies on the head, so its heading front to back is whatever keeps it on the
        # skin: over the crown that is forward, on the forehead down, at the temple back.
        down = 1 - BANG_AXES['up'] * inner * SF.sm(0.72, 1.0, t) ** 2
        # (John 2026-11-29: "research how to get the bell shape forehead, right now it's a triangle". The cut: a
        # triangle section, "the middle is the shortest, and either end is longer resembling an umbrella effect" (Sam
        # Villa, twist cut fringe); the dry: a small round brush, "twist your wrist and curl the ends" (same), "that
        # perfect sweeping motion on the ends" (Brad Mondo, Stylist). So the hair at the part is short, comes down the
        # forehead nearly straight and turns out only in its curled end, at the brow: the bare forehead between the
        # two sides is narrow above and flares at the brow, a bell. A steady pull to the side from the root gave
        # straight diagonals: a triangle. side0: the pull on the forehead, small at the part and growing outward
        # (side0u); side1: the curl of the end, over the last 40%.)
        sx = sgn * ((BANG_AXES['side0'] + BANG_AXES['side0u'] * uc) * SF.sm(0.0, 0.2, t)
                    + BANG_AXES['side1'] * inner * SF.sm(0.58, 0.97, t) ** 1.3)
        ny = np.where(np.abs(n[:, 1]) < 0.15, -0.15, n[:, 1])
        d = np.c_[sx, np.clip(-(sx * n[:, 0] - down * n[:, 2]) / ny, -3.0, 3.0), -down]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        step = ds
        if front is not None:
            # (John 2026-11-29, of round 12's traced sheet: "you got it right here". Seen from the front the flow
            # is steered along the traced lines (TRACE_IN, TRACE_OUT, blended across the sheet): each step advances by
            # the traced line's own step in x and z, and its depth is still solved on the head. The pulls above are
            # then the traced field's, not weights.)
            fx = sgn * (front[lkh, j, 0] - front[lkh, j - 1, 0]); fz = front[lkh, j, 1] - front[lkh, j - 1, 1]
            fl = np.hypot(fx, fz)
            d = np.c_[fx, np.clip(-(fx * n[:, 0] + fz * n[:, 2]) / ny, -3.0 * fl, 3.0 * fl), fz]
            step = np.linalg.norm(d, axis=1, keepdims=True)
            d /= np.maximum(step, 1e-9)
        if side is not None:
            # (the side field: the hair beside the bang where it leaves the head at the temple. Toward its end the
            # bang is drawn to that hair (BANG_AXES['to']) and along its heading (BANG_AXES['along']), in all three
            # axes, so the ends turn into the side hair's flow.)
            g = BANG_AXES['field'] * SF.sm(0.5, 1.0, t)
            dist, ix = side[0].query(p, k=12)
            to = side[1][ix].mean(1) - p
            to /= np.maximum(np.linalg.norm(to, axis=1, keepdims=True), 1e-6)
            al = side[2][ix].mean(1); al /= np.maximum(np.linalg.norm(al, axis=1, keepdims=True), 1e-6)
            al[:, 2] *= down[:, None][:, 0]                                # (the tip's lift is kept)
            d = d + g * (BANG_AXES['to'] * to + BANG_AXES['along'] * al)
        d = d - (d * n).sum(1, keepdims=True) * n
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        p = p + d * step
        q = p - SF.CENTRE; r = np.linalg.norm(q, axis=1, keepdims=True); U = q / r
        # (toward its end the bang rides over the side hair, BANG_OVER_MM off the skin: on the bare skin it could not
        # reach the photo's width, the head being narrower than the hair)
        p = SF.CENTRE + U * (surf.radius(U)[:, None] + (1.0 + BANG_OVER_MM * SF.sm(0.45, 0.9, t)) * MM)
        P[:, j] = p
    # (John 2026-11-29: "you need natural hair curl as well". Hair is not drawn with a ruler: each lock carries its own
    # slow wave across its length (BANG_CURL: mm, wavelength mm) and each hair a smaller one of its own, from the root
    # out. Radius of the lock's wave: wavelength^2 / (4 pi^2 amplitude), kept over the 17.5 mm of a set fringe.)
    rc = np.random.default_rng(17)
    seg = np.diff(P, axis=1); sarc = np.c_[np.zeros(m), np.cumsum(np.linalg.norm(seg, axis=2), axis=1)] / MM
    tan = np.gradient(P, axis=1); tan /= np.maximum(np.linalg.norm(tan, axis=2, keepdims=True), 1e-9)
    nrm = P - SF.CENTRE; nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    lat = np.cross(tan, nrm); lat /= np.maximum(np.linalg.norm(lat, axis=2, keepdims=True), 1e-9)
    tt_ = np.linspace(0, 1, k)[None, :]
    phl = rc.uniform(0, 2 * np.pi, BANG_RIBS)[lk][:, None]; lml = (BANG_CURL[1] * rc.uniform(0.85, 1.15, BANG_RIBS))[lk][:, None]
    wv = BANG_CURL[0] * np.sin(2 * np.pi * sarc / lml + phl) * SF.sm(0.1, 0.4, tt_)
    wv = wv + BANG_CURL[2] * np.sin(2 * np.pi * sarc / (0.6 * lml) + rc.uniform(0, 2 * np.pi, (m, 1))) * SF.sm(0.05, 0.3, tt_)
    P = P + lat * (wv * MM)[..., None]
    # (each lock draws to a tip: its hairs move onto the lock's mean line over the last 45%)
    # (John 2026-11-29: "keep the ends same, increase thickness overall of each lock": the lock keeps its full width to
    # BANG_FULL of its length and closes to the same tip from there)
    w = (BANG_TIP * SF.sm(BANG_FULL, BANG_END_AT, np.linspace(0, 1, k)))[None, :, None]
    for j in range(BANG_RIBS):
        mm = lk == j
        if mm.sum() > 1:
            P[mm] = P[mm] + (P[mm].mean(0, keepdims=True) - P[mm]) * w
    return P


def bang_sheet(root, surf, seed=3, k=44, side=None):
    """One side's bang as ONE smooth surface (John 2026-11-28: "build the bangs as one smooth shape that produces that s
    curve in the picture"). The surface is S(u, t): u across it (0 the hair nearest the part, 1 the outermost, by the
    roots' order, so the order is kept) and t along the hair. Seen from the front every hair is a cubic that leaves its
    root heading down, runs out to the side and arrives at its end heading down again: an S. Its ends lie on the measured
    line from the brow to the temple. Out of the face it follows the skin at a stand-off that is the same smooth
    function of t for every hair: the arch off the root, in over the forehead, the flick at the end (FRINGE_ARCH_MM).
    root (m, 3), all on one side. Returns a list of (k, 3)."""
    rng = np.random.default_rng(seed)
    m = len(root)
    sgn = 1.0 if np.median(root[:, 0]) >= 0 else -1.0
    u = np.argsort(np.argsort(np.abs(root[:, 0]))) / max(m - 1, 1)
    t = np.linspace(0, 1, k)[None, :]
    # (John 2026-11-28: "trace the smooth curve compared to our trapezoidal". The bang's inner edge and its outer edge are
    # traced on the photo (TRACE_IN, TRACE_OUT: the bare forehead is an arch, narrow and rounded at the hairline and
    # then straight down, not the V the cubic gave) and every hair is the blend of the two curves by its place across
    # the sheet, held at its own root and eased onto the blend over the first 40% of its length. The ends then lie on
    # the line between the two traced ends, the photo's lower edge, which is nearly straight.)
    from scipy.interpolate import CubicSpline

    def curve(pts, tt_):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        return CubicSpline(d / d[-1], pts, bc_type='natural')(tt_)
    a_ = np.abs(root[:, 0]); r_in = root[np.argmin(a_)]; r_out = root[np.argmax(a_)]
    cin = curve(np.r_[[[abs(r_in[0]), r_in[2]]], np.array(TRACE_IN)], t[0])
    cout = curve(np.r_[[[abs(r_out[0]), r_out[2]]], np.array(TRACE_OUT)], t[0])
    # (each of the BANG_RIBS locks draws to a tip: its hairs move onto the lock's centre line over the last 45%)
    vr = (u * BANG_RIBS) % 1.0 - 0.5
    uc = np.clip((np.floor(u * BANG_RIBS) + 0.5) / BANG_RIBS, 0, 1)
    u2 = (u[:, None] + (uc - u)[:, None] * BANG_TIP * SF.sm(0.55, 1.0, t[0])[None, :])[..., None]
    B = (1 - u2) * cin[None] + u2 * cout[None]
    # (John 2026-11-29: "inside locks bend more than outside ones". Each lock's end turns out and up by BANG_BEND
    # (HH), in full for the lock nearest the part and falling to BANG_BEND[2] of it for the outermost.)
    bend = ((1 - (1 - BANG_BEND[2]) * uc)[:, None] * SF.sm(0.55, 1.0, t[0])[None, :] ** 2)
    B = B + bend[..., None] * np.array([BANG_BEND[0], BANG_BEND[1]])[None, None, :]
    B = B + (np.c_[np.abs(root[:, 0]), root[:, 2]] - B[:, 0])[:, None, :] * (1 - SF.sm(0.0, 0.4, t[0]))[None, :, None]
    X = sgn * B[..., 0]; Z = B[..., 1]
    # the skin in front at each (x, z): bisection on y
    lo = np.full(X.shape, -0.75); hi = np.full(X.shape, SF.CENTRE[1])
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        q = np.stack([X, mid, Z], -1).reshape(-1, 3) - SF.CENTRE
        r = np.linalg.norm(q, axis=1)
        inside = (r < surf.radius(q / r[:, None])).reshape(X.shape)
        hi = np.where(inside, mid, hi); lo = np.where(inside, lo, mid)
    # (where a traced line runs out past the head's outline there is no skin in front of it and the search fell back
    # to the head's centre: the ends wrapped round to the ear and showed from behind. Seen 2026-11-29 in the side and
    # back views. The sheet stays in front of the temple: no farther back than BANG_BACK in front of the centre.)
    P = np.stack([X, np.minimum(0.5 * (lo + hi), SF.CENTRE[1] - BANG_BACK), Z], -1)
    if BANG_3D:
        B0 = (1 - u)[:, None, None] * cin[None] + u[:, None, None] * cout[None]
        B0 = B0 + (np.c_[np.abs(root[:, 0]), root[:, 2]] - B0[:, 0])[:, None, :] * (1 - SF.sm(0.0, 0.4, t[0]))[None, :, None]
        P = _bang_march(root, surf, u, uc, sgn, k, side, front=B0 if BANG_TRACED_FLOW else None)
        P = np.array([_stand_at_part(p_) for p_ in P])                  # (stood up at the part, as the hair beside it)
    # (the stand-off is measured from a skin eased along the hair, so the sheet bridges the brow and does not dip)
    from scipy.ndimage import gaussian_filter1d
    q = P - SF.CENTRE
    from scipy.ndimage import maximum_filter1d
    r = gaussian_filter1d(maximum_filter1d(np.linalg.norm(q, axis=2), 9, axis=1, mode='nearest'), 4.0, axis=1, mode='nearest')
    U = q / np.linalg.norm(q, axis=2, keepdims=True)
    tt = t[0]
    # (John 2026-11-28: "bangs need to begin from roots and be lifted as well". The bang is part of the same hair: it
    # leaves its root on the same arc as the hair behind it, at an angle and over at one bend, up to FRINGE_ARCH_MM[0]
    # under the outer surface, and from there comes down over the forehead to FRINGE_ARCH_MM[1] off the brow.)
    rise = 1 - (1 - np.clip(tt / BANG_SPAN[0][1], 0, 1)) ** 2
    down = SF.sm(*BANG_SPAN[1], tt)
    lift = (1.0 + FRINGE_ARCH_MM[0] * rise * (1 - down) + FRINGE_ARCH_MM[1] * down)[None, :] * MM
    if BANG_3D:
        # (John 2026-11-29: "fix the crown, the root lift does not agree with the bangs' root lift, it causes ridges on
        # the head". The bang rose to 22 mm over the first 45% of its length, about 80 mm; the hair beside it rises to
        # the one outer surface (ENV_MM by height) over ROOT_RISE_MM, 36 mm: two surfaces side by side with a step
        # between. The bang now leaves its root on the same arc to the same surface (at BANG_ENV of its height, the
        # middle of the layers beside it), and comes down from it to the forehead as before.)
        s_mm = np.c_[np.zeros(m), np.cumsum(np.linalg.norm(np.diff(P, axis=1), axis=2), axis=1)] / MM
        capz = (ENV_MM[0] + (ENV_MM[1] - ENV_MM[0]) * SF.sm(*ENV_Z, P[..., 2])) * BANG_ENV
        capz = capz * (PART_DIP + (1 - PART_DIP) * SF.sm(0.0, PART_WIDE_MM, np.abs(P[..., 0]) / MM))
        scalp_lift = (LIFT_MM[0] + capz * (1 - (1 - np.minimum(s_mm / ROOT_RISE_MM, 1.0)) ** 2)) * MM
        lift = np.full((m, k), FRINGE_ARCH_MM[1] * MM)
    lift = lift + (FRINGE_ARCH_MM[2] * SF.sm(*BANG_SPAN[2], tt))[None, :] * (1 - 0.85 * u)[:, None] * MM
    lift = lift - (0.6 * FRINGE_ARCH_MM[1] * SF.sm(0.5, 1.0, tt))[None, :] * u[:, None] * MM
    lift = lift + ((BANG_THICK_MM if BANG_3D else 3.0) * rng.random(m))[:, None] * MM * (SF.sm(0.0, 0.3, tt) * (1 - SF.sm(0.6, 1.0, tt) * (1.0 if BANG_3D else 0.0)))[None, :]     # (its thickness: 3 mm, closing at a 3D lock's tip)
    if BANG_3D:                                                          # (the layers: each lock over the next one out)
        lk_ = np.minimum((u * BANG_RIBS).astype(int), BANG_RIBS - 1)
        lift = lift + (BANG_LAYER_MM * (BANG_RIBS - 1 - lk_))[:, None] * MM * SF.sm(0.2, 0.5, tt)[None, :]
    # (John 2026-11-29: "the lines you drew are correct but the render is flat ... the inflection near the ends and
    # it's bending up". Seen from the front the lines were right, but the sheet lay at one stand-off, a plate: now it is
    # ribbed across into BANG_RIBS locks (each a lit crest and a dark crease, BANG_RIB_MM deep, growing from the root),
    # and past the inflection the ends lift off the cheek by BANG_FLICK_MM, the outward bend carried into depth.)
    rib = 0.5 - 0.5 * np.cos(2 * np.pi * BANG_RIBS * u)
    lift = lift + (BANG_RIB_MM * rib)[:, None] * MM * SF.sm(0.05, 0.45, tt)[None, :]
    lift = lift + (BANG_FLICK_MM * SF.sm(0.62, 1.0, tt) ** 2)[None, :] * MM
    if BANG_3D:
        # (John 2026-11-29, circling where the bang meets the side hair: "get the joint correct". The bang came down
        # off the outer surface by its length, from 30% along, while still over the scalp, so along the joint it lay
        # under the hair beside it: 9.5 mm at the median (tools/gauges/hair/crown_step.py), a crease with a shadow in
        # it. It now stays on the outer surface, at the depth of its place in the hair (BANG_ENV to 1 of the height,
        # as the layers beside it), for as long as it is over the scalp, and comes down to the forehead only past the
        # hairline: over BANG_DROP below the lowest root at its distance from the part.)
        ax_ = np.abs(root[:, 0]); o_ = np.argsort(ax_)
        nb = max(4, m // 25)
        hx = np.array([ax_[o_[i:i + nb]].mean() for i in range(0, m - nb + 1, nb)])
        hz = np.array([root[o_[i:i + nb], 2].min() for i in range(0, m - nb + 1, nb)])
        hl = np.interp(np.abs(P[..., 0]), hx, hz)
        dn2 = np.maximum.accumulate(SF.sm(hl + BANG_DROP[0], hl - BANG_DROP[1], P[..., 2]), axis=1)
        dep = BANG_ENV + (1 - BANG_ENV) * rng.random(m)[:, None]
        lift = scalp_lift * dep / BANG_ENV * (1 - dn2) + lift * dn2
    P = SF.CENTRE + U * (r + lift)[:, :, None]
    # (faired along the hair over BANG_FAIR nodes, the root held, as the falls are: a bang rooted far back runs over the
    # top of the head, where the front projection's cubic and the stand-off's turns meet in corners tighter than the
    # brush. Then kept a millimetre off the skin.)
    if BANG_FAIR > 0:
        Ps = gaussian_filter1d(P, BANG_FAIR, axis=1, mode='nearest')
        hold = np.exp(-np.arange(k) / max(BANG_FAIR, 1.0))[None, :, None]
        P = Ps + (P[:, :1] - Ps[:, :1]) * hold
        q = P - SF.CENTRE; rr = np.linalg.norm(q, axis=2)
        Uq = q / rr[:, :, None]
        sk = surf.radius(Uq.reshape(-1, 3)).reshape(rr.shape) + 1.0 * MM
        P = SF.CENTRE + Uq * np.maximum(rr, sk)[:, :, None]
        if BANG_3D:
            # (John 2026-11-29: "the bangs should begin with the same roots as everywhere else". The fairing along
            # the hair had flattened the root's arc: the bang reached 10 mm over the skull 20 mm from its root, the hair
            # beside it 15 mm (measured). Over the scalp the bang's height is put back to the falls' own law after
            # the fairing: the same arc off the root (ROOT_RISE_MM) to the same surface (ENV_MM), at the same spread of
            # depths under it (0.45 to 1, as build's layers).)
            P = SF.CENTRE + Uq * (np.maximum(rr, sk) * dn2 + (sk - 1.0 * MM + lift) * (1 - dn2))[:, :, None]
    # (John 2026-11-28: "edges are not curved, the ends should taper off". The lower edge is a curve, longest in the
    # middle of the sheet and shorter toward both edges (BANG_TAPER), the ends cut ragged so they thin out.)
    cut = (k * (1 - BANG_TAPER * (2 * u - 1) ** 2 - BANG_RAG * np.abs(2 * vr) * (0.6 + 0.4 * rng.random(m)))).astype(int)
    if BANG_3D:                                                          # (a lock's hairs end together, at its tip)
        cut = (np.full(m, k) * BANG_END_AT).astype(int) + 1 - (2 * rng.random(m)).astype(int)
    return [P[i, :max(cut[i], 4)] for i in range(m)]


def body_radius(depth, nth=128, zs=None):
    """The head's and body's radius from the vertical axis by angle and height (the outermost solid along each ray),
    eased round the body so that hair bridges hollows. Returns (th, z, R)."""
    from scipy.ndimage import maximum_filter1d, gaussian_filter
    zs = np.arange(0.62, -1.0, -0.02) if zs is None else zs
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    rr = np.arange(0.95, 0.02, -0.005)
    T, Z, Rr = np.meshgrid(th, zs, rr, indexing='ij')
    X = np.stack([SF.AXIS[0] + Rr * np.sin(T), SF.AXIS[1] - Rr * np.cos(T), Z], -1).reshape(-1, 3)
    ins = (depth(X) > 0).reshape(T.shape)
    first = np.argmax(ins, axis=2)
    R = np.where(ins.any(2), rr[first], 0.05)
    R = maximum_filter1d(R, 5, axis=0, mode='wrap')
    R = gaussian_filter(R, (2.0, 1.0), mode=('wrap', 'nearest'))
    return th, zs, R


def _join(p, j, half=14, sig=4.0):
    """The line faired where it leaves the head (node j): the nodes within `half` of it are drawn to their smoothed
    places, most at j, none at the window's ends, so the scalp line runs into the fall with no corner."""
    from scipy.ndimage import gaussian_filter1d
    a, b = max(j - half, 1), min(j + half, len(p) - 1)
    if b - a < 4:
        return p
    sm_ = gaussian_filter1d(p, sig, axis=0, mode='nearest')
    w = np.clip(1 - np.abs(np.arange(a, b) - j) / half, 0, 1)[:, None]
    q = p.copy(); q[a:b] = p[a:b] + w * w * (3 - 2 * w) * (sm_[a:b] - p[a:b])
    return q


HEAD_FAIR = 2.5             # nodes (2.5 mm each) the comb on the head is faired over


def _fair_head(p, surf):
    from scipy.ndimage import gaussian_filter1d
    if len(p) < 5:
        return p
    # (the root is pinned and nothing else: a hold that fades over a few nodes keeps the corner at the root, where the
    # first step runs along the skin and the lift then turns it out. The line is reflected through the root before
    # filtering, so the faired line still starts at it with the direction of the lines after it.)
    ext = np.r_[2 * p[0] - p[1:8][::-1], p]
    q = gaussian_filter1d(ext, HEAD_FAIR, axis=0, mode='nearest')[7:]
    q = q + (p[0] - q[0]) * np.exp(-np.arange(len(p)) / 1.0)[:, None]
    d = q - SF.CENTRE; r = np.linalg.norm(d, axis=1); U = d / r[:, None]
    floor = surf.radius(U) + LIFT_MM[0] * MM * (np.arange(len(p)) > 0)
    return SF.CENTRE + U * np.maximum(r, floor)[:, None]


def _cached(cache, name, key, make, log=print):
    """make() once per key: the result is pickled under cache/name-<key>.pkl. cache None: no caching."""
    import hashlib, pickle, time
    if cache is None:
        return make()
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, '%s-%s.pkl' % (name, hashlib.md5(repr(key).encode()).hexdigest()[:16]))
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    # (keep the newest two of a stage: every edit to a constant the stage reads makes a new key, and a comb is 58 MB)
    old = sorted([g for g in os.listdir(cache) if g.startswith(name + '-') and g.endswith('.pkl')],
                 key=lambda g: os.path.getmtime(os.path.join(cache, g)))
    for g in old[:-1]:
        os.remove(os.path.join(cache, g))
    t = time.time(); v = make()
    pickle.dump(v, open(f + '.tmp', 'wb'), protocol=4); os.replace(f + '.tmp', f)
    log('cache: built %s in %.0f s' % (name, time.time() - t))
    return v


PART_STAND = (12.0, 14.0, 60.0)   # DRAFT (mm): how far the hair by the part stands in toward it, the length over which it
                                  # stands up, and the length by which it has leaned back out


def _stand_at_part(p):
    """The roots by the part stand up before they lean out (the blow-dry's lift at the base: the section is
    overdirected, dried standing, and falls back over). Each line on top of the head is held in toward the part over
    its first PART_STAND[1] mm and let out again by PART_STAND[2], never across the part. Without it the outer surface
    was two humps with a cleft between them 17 mm deep (a hair that leaves the part sideways is 14 mm away from it by
    the time it has risen); lowering the surface toward the part (PART_DIP) only made the cleft a wide valley, seen
    2026-11-29."""
    if len(p) < 3 or p[0, 2] < 0.78:
        return p
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] / MM
    b = SF.sm(0.0, PART_STAND[1], s) * SF.sm(PART_STAND[2], PART_STAND[1], s) * SF.sm(0.78, 0.88, p[0, 2])
    sg = 1.0 if p[min(len(p) - 1, 8), 0] >= 0 else -1.0
    xa = np.abs(p[:, 0])
    q = p.copy()
    q[:, 0] = sg * np.maximum(xa - PART_STAND[0] * MM * b, np.minimum(xa, 0.5 * MM))
    return q


def _code_key(*fns):
    """What a function's result depends on in this module: its own source and the module constants it names."""
    import inspect
    g = globals(); out = []
    for fn in fns:
        out.append(inspect.getsource(fn))
        out += [(k, repr(g[k])) for k in sorted(set(fn.__code__.co_names)) if k.isupper() and k in g]
    return out


def field2(F, surf, depth, n=24000, seed=11, log=print, cache=None, head_key=None):
    """cache: a directory. The comb over the head (build) and the body's radius table are saved there and reused
    while the head (head_key), n, seed, and the code and constants they depend on are unchanged, so a change to the
    falls, the bangs' sheet, the cut or the locks does not re-comb the head."""
    """Returns dict(line: list of (k, 3), basin: per line (0 fringe, 1 face framing, 2 front fall, 3 back fall, -1 lies
    on the head), hairs, root, sec (the line's section, for colour), release (per line: node where it leaves the head),
    gauges)."""
    from scipy.interpolate import RegularGridInterpolator
    paths, hairs, _, kind, x = _cached(cache, 'comb', (head_key, n, seed, _code_key(build, curtain_z)),
                                       lambda: build(F, surf, n, seed, log=lambda t: None), log)
    hand, layer, z_end = x['hand'], x['layer'], x['z_end']
    bt, bz, bR = _cached(cache, 'bodyradius', (head_key, _code_key(body_radius)), lambda: body_radius(depth), log)
    Rb = RegularGridInterpolator((np.r_[bt, np.pi], bz[::-1]), np.c_[bR.T, bR[0]].T[:, ::-1], bounds_error=False, fill_value=None)
    hang = np.where((hand >= 0) & (kind != 1))[0]
    rel = np.array([paths[i][min(hand[i], len(paths[i]) - 1)] for i in hang])
    th0 = SF.theta(rel); a0 = np.abs(th0); sg = np.where(th0 >= 0, 1.0, -1.0)
    r0 = np.hypot(rel[:, 0] - SF.AXIS[0], rel[:, 1] - SF.AXIS[1])
    lo = float(np.percentile(a0, 1))
    front = a0 < FALL_SPLIT
    # each line's place in its fall: linear in its angle, so the order round the head is the order in the fall
    a1 = BACK_BAND[0] + (BACK_BAND[1] - BACK_BAND[0]) * (a0 - FALL_SPLIT) / (np.pi - FALL_SPLIT)
    # (the front fall fills its band by the hairs' order, not by their angle: the lengths, swept back on the head, all
    # leave it behind the temple, and placed by angle they left the band's inner part to the short face-framing
    # pieces alone: an inner edge 0.42 from the midline at the collarbone against the photo's 0.2)
    kh = kind[hang]
    for s_ in (1.0, -1.0):
        for kk, band in ((0, FRONT_BAND), (2, (FRONT_BAND[0], FRONT_BAND[0] + 0.25))):
            m = np.where(front & (sg == s_) & (kh == kk))[0]
            if len(m) > 1:
                a1[m] = band[0] + (band[1] - band[0]) * np.argsort(np.argsort(a0[m])) / (len(m) - 1)
    d = np.clip((layer[hang] - 0.2) / 0.8, 0, 1)
    stack = (STACK_MM[0] + (STACK_MM[1] - STACK_MM[0]) * d) * MM
    land = Rb(np.c_[np.clip(sg * a1, -np.pi, np.pi), np.full(len(hang), BODY_AT_Z)]) + stack
    lam = WAVE['lam_hh']
    phase = 0.5 * np.sin(2 * a0) + np.where(sg > 0, 0.6, 0.0)           # (one pattern a side, drifting round the head)
    from scipy.ndimage import gaussian_filter1d
    # (the comb on the head is faired before anything is hung from it: each step is pushed back out to the skin plus
    # its lift, and the skin's small bumps turned the line by over 20 degrees a node on 8,000 of 7,500 lines' heads,
    # never gauged because the smoothness gauge starts at the release. Faired over HEAD_FAIR nodes, root held, kept at
    # least its first stand-off off the skin.)
    paths = [_stand_at_part(_fair_head(p, surf)) for p in paths]
    lines = [p for p in paths]
    base = {}
    basin = np.where(kind == 1, 0, -1)
    for sd in (1.0, -1.0):                                              # (each side's bang: one smooth sheet)
        fm = np.where((kind == 1) & (x['root'][:, 0] * sd > 0))[0]
        if len(fm) > 3 and not BANG_FLOW:
            # the side field for this side's bang: the combed lines of the hair beside it (face framing and front fall)
            from scipy.spatial import cKDTree
            sm_ = [paths[i][:max(hand[i], 2) + 1] for i in np.where((kind != 1) & (x['root'][:, 0] * sd > 0) & (hand >= 0))[0]
                   if abs(SF.theta(paths[i][:1])[0]) < FALL_SPLIT]
            sp = np.concatenate([p_[2 * len(p_) // 3:] for p_ in sm_]); st = np.concatenate([np.gradient(p_, axis=0)[2 * len(p_) // 3:] for p_ in sm_])
            st /= np.maximum(np.linalg.norm(st, axis=1, keepdims=True), 1e-9)
            for i, l in zip(fm, bang_sheet(x['root'][fm], surf, seed=3, side=(cKDTree(sp), sp, st))):
                # (stood up at the part like the hair beside it, then faired again: the bang's nodes are 4 mm apart and
                # the stand alone bent 6% of them tighter than the brush)
                lines[i] = l                                             # (stood at the part inside bang_sheet)
    sec = np.zeros(n, int)
    step = SEG_MM * MM
    for q, i in enumerate(hang):
        ze = RELEASE_Z - (RELEASE_Z - z_end[i]) / TAKE
        # (the hair that falls over the shoulders is cut at the shoulder, graded into the long front and back: John
        # 2026-11-28, "cut sides to shoulder length")
        ws = SF.sm(SIDE_CUT_A[0], SIDE_CUT_A[1], a1[q]) * SF.sm(SIDE_CUT_A[3], SIDE_CUT_A[2], a1[q])
        ze = ze + ws * (max(ze, SIDE_CUT_Z + RAGGED * ((i * 0.6180339) % 1.0)) - ze)       # (point cut: ragged by RAGGED)
        z = np.arange(rel[q, 2], ze, -step)   # (cut by height here: no length is taken up)
        if len(z) < 3:
            continue
        g = SF.sm(GATHER_Z[0], GATHER_Z[1], z)
        # (the hair leaves the head still turning back: that turn dies away over FALL_CARRY, so the course off the head
        # carries on the course on it: no straight drop under a swept side)
        pj = paths[i][max(hand[i] - 7, 0):hand[i] + 1]
        w0 = 0.0
        if len(pj) > 3 and pj[0, 2] - pj[-1, 2] > 1e-3:
            w0 = float(np.clip((abs(SF.theta(pj[-1:])[0]) - abs(SF.theta(pj[:1])[0])) / (pj[0, 2] - pj[-1, 2]), 0.0, 2.0))
        carry = w0 * FALL_CARRY * (1 - np.exp(-(rel[q, 2] - z) / FALL_CARRY))
        ah = a0[q]
        if front[q]:                                                     # (beside the face: the whole front fall drawn back
            keep = lo + (max(FACE_KEEP, lo) - lo) * SF.sm(FACE_KEEP_Z[0], FACE_KEEP_Z[1], z)   # behind the cheek, order kept)
            ah = keep + (a0[q] - lo) * (FALL_SPLIT - keep) / (FALL_SPLIT - lo)
        ah = ah + carry * np.clip((np.pi - ah) / 0.5, 0, 1)
        a = ah + (a1[q] - ah) * g
        t = np.clip((rel[q, 2] - z) / max(rel[q, 2] - BODY_AT_Z, 1e-6), 0, 1)
        tent = r0[q] + (land[q] - r0[q]) * t
        flare = (FLARE[0] * SF.sm(RELEASE_Z, RELEASE_Z - 0.16, z) + FLARE[1] * SF.sm(0.25, -0.15, z)) * (0.35 + 0.65 * d[q]) \
            * (1 - 0.6 * SF.sm(-0.2, -0.6, z)) * np.clip(np.sin(a), 0, 1) ** 2       # (at the sides: not on the chest, not down the back)
        floor = Rb(np.c_[sg[q] * a, z]) + stack[q]
        over0 = max(float(floor[0] - r0[q]), 0.0)                        # (the floor meets the head where the hair leaves it)
        R = np.maximum(floor - over0 * np.exp(-np.arange(len(z)) / (FAIR_MM / SEG_MM)), tent + flare)
        # one eased course (John 2026-11-28: "needs to be smoother"): the eases above meet in corners (behind the cheek,
        # at the collarbone), so the course is faired over FAIR_MM along the hair, held at the head end, and kept off
        # the body. gauges/hair/flow_smooth.py reads the bend radius of this course against a hair's bending length.
        nf = FAIR_MM / SEG_MM
        fade = np.exp(-np.arange(len(z)) / nf)
        for _ in range(2):
            a_s = gaussian_filter1d(a, nf, mode='nearest'); a = a_s + (a[0] - a_s[0]) * fade
            R_s = gaussian_filter1d(R, nf, mode='nearest'); R_s = R_s + (R[0] - R_s[0]) * fade
            R = np.maximum(R_s, gaussian_filter1d(Rb(np.c_[sg[q] * a, z]), nf, mode='nearest') + stack[q] - over0 * fade)
        base[i] = _join(np.r_[paths[i][:hand[i]], np.c_[SF.AXIS[0] + R * np.sin(sg[q] * a), SF.AXIS[1] - R * np.cos(sg[q] * a), z]], hand[i])
        Wv = SF.sm(WAVE['start_z'], WAVE['full_z'], z) * WAVE['amp_hh']
        psi = 2 * np.pi * (WAVE_OUT_Z - z) / lam + phase[q]               # (by height, not by each hair's own start:
                                                                          #  neighbours swing together)
        a = a + Wv * np.sin(psi) / np.maximum(R - stack[q] + np.mean(STACK_MM) * MM, 0.05) * SF.sm(np.pi, np.pi - 0.35, a)   # (the S across the fall; none at the
                                                                          #  back's middle, where the two sides meet)
        R = R + 0.5 * Wv * np.cos(psi)                                    # (and half as much out of it, a quarter wave on:
        th = sg[q] * a                                                    #  the ribbon turns as it swings)
        low = np.c_[SF.AXIS[0] + R * np.sin(th), SF.AXIS[1] - R * np.cos(th), z]
        lines[i] = _join(np.r_[paths[i][:hand[i]], low], hand[i])
        basin[i] = 1 if kind[i] == 2 else (2 if front[q] else 3)
        sec[i] = int(a0[q] / (np.pi / SECTORS))
    # gauges (the recipe's): rank swaps between neighbours, lines turning back, one sink per basin, the photo's widths
    g = {}
    for nm, b in (('front', 2), ('back', 3)):
        for s_ in (1.0, -1.0):
            m = np.where((basin == b) & (np.array([l[0, 0] for l in lines]) * s_ > 0))[0]
            if len(m) < 10:
                continue
            aa = {}
            for zq in (0.4, 0.0, -0.4):
                v = [(abs(SF.theta(lines[i][np.argmin(np.abs(lines[i][:, 2] - zq))][None])[0]), i) for i in m
                     if lines[i][-1, 2] < zq - 0.01]
                aa[zq] = dict((i, t_) for t_, i in v)
            both = [i for i in aa[0.0] if i in aa[-0.4]]                 # (between two heights below every root)
            o = sorted(both, key=lambda i: aa[0.0][i])
            r2 = np.argsort(np.argsort([aa[-0.4][i] for i in o]))
            g['rank_moves_over_3pct_%s_%s' % (nm, 'L' if s_ > 0 else 'R')] = '%d of %d' % (int((np.abs(r2 - np.arange(len(o))) > 0.03 * len(o)).sum()), len(o))
    g['lines_turning_up'] = int(sum((np.diff(lines[i][hand[i]:, 2]) > 1e-6).any() for i in hang if basin[i] > 0))
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gauges', 'hair'))
    import flow_smooth as FS
    bi = sorted(base)
    g['smooth_falls'] = FS.gauge([base[i] for i in bi], [hand[i] for i in bi], 'free')
    # (the fringe is gauged in two parts since it took the falls' root: past the root's arc against the brush, as before,
    # and the arc itself reported apart: a root stood up by the dryer leaves the skin over ROOT_RISE_MM, tighter than
    # any brush, and the falls' roots, the same arc, were never in this gauge: theirs starts where they leave the head)
    def _past(l):
        s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(l, axis=0), axis=1))] / MM
        return l[s_ >= ROOT_RISE_MM + 8.0]
    fr_ = [lines[i] for i in np.where(basin == 0)[0]]
    g['smooth_fringe'] = FS.gauge([p_ for p_ in map(_past, fr_) if len(p_) > 4], None, 'set')
    g['smooth_fringe_with_root'] = {k_: v_ for k_, v_ in FS.gauge(fr_, None, 'set').items() if k_ in ('share_under_bound', 'radius_mm_min', 'ok')}
    tot = hairs.sum()
    g['basin_share_pct'] = {nm: round(100 * float(hairs[basin == b].sum() / tot), 1) for nm, b in
                            (('fringe', 0), ('face framing', 1), ('front fall', 2), ('back fall', 3), ('on the head', -1))}
    A = [l for l, b in zip(lines, basin) if b > 0]
    g['half_width'] = {str(zq): round(float(np.percentile([abs(l[np.argmin(np.abs(l[:, 2] - zq)), 0]) for l in A if l[-1, 2] < zq], 98)), 3)
                       for zq in (0.52, 0.2, -0.15)}
    g['half_width_photo'] = {'0.52': 0.45, '0.2': 0.54, '-0.15': 0.63}
    g['front_inner_edge_at_-0.15'] = round(float(np.percentile([abs(l[np.argmin(np.abs(l[:, 2] + 0.15)), 0]) for l, b in zip(lines, basin)
                                                                if b == 2 and l[-1, 2] < -0.15], 2)), 3)
    g['ends_z_p5'] = round(float(np.percentile([l[-1, 2] for l in A], 5)), 3)
    log('field2 %s' % g)
    return dict(line=lines, basin=basin, hairs=hairs, root=x['root'], sec=sec, release=hand, kind=kind, gauges=g)



# ---- locks inside the falls (the ponytail recipe's step 6, for hair that is not tied) ----
# A fall drawn as offsets from one field is one sheet with a texture. Real hair is a bundle of locks: neighbours on the
# head gather into a lock below it, each lock carries its own share of the wave (a little ahead of or behind its
# neighbours, so the locks part in depth), and toward the ends a lock opens into sub-locks. The field stays the shape:
# a lock's axis is the mean of its own field lines, so the locks fill the field's fall and keep its order.
LOCK_SIZE = 160             # DRAFT: field lines per lock (16,000 lines: about 45 locks a side; the recipe's tail took 33)
LOCK_SCALE = {1: 0.25, 2: 2.5, 3: 0.7}   # lock size by fall, as a share of LOCK_SIZE (John 2026-11-28: "use thicker locks and
                            # fewer like the picture for the hair that rests on the chest and thinner ones for the shoulder
                            # locks"): face framing finest, the front fall on the chest 2.5x (the photo shows about 4 or 5
                            # broad locks a side there), the back fall over the shoulders 0.7x
FRONT_TRACE = {1: (0.6,), 2: (2.6, 0.0)}   # TRACED on the photo (hair-rebuild/wavy-photo/photo-lock-trace.png): the front of
                            # each side is two big locks, the inner one beside the face and down the chest and the outer
                            # one over the shoulder's front, waving about half a wave apart (the inner bulges where the
                            # outer draws in); the face-framing pieces ride as one lock. Phase (rad) per lock, inner first
CLUMP_B = {3: 0.85, 2: 0.7, 1: 0.7}         # the back fall barely gathers, and LOCK_PLAY scales its locks' own wave and opening: the back is
LOCK_PLAY = {3: 0.3}        # one continuous sheet (John 2026-11-28: "back should be continuous"; gathered, the back's locks
                            # left gaps and a step where they began under the head)
CLUMP = 0.45                # recipe step 6: a lock gathers to this share of its spread on the head
CLUMP_Z = 0.10              # HH below the release over which it gathers
SUBLOCKS, SUB_OPEN, HAIR_OPEN = 3, 1.4, 0.7   # recipe step 6: 3 sub-locks, middles 1.4x farther out at the tip, hairs 0.7x
END_FADE = 0.08              # HH: a member is weighed out of its lock's axis over this much of its fall's end
OPEN_FROM = 0.0            # recipe step 6: opening starts 12% down the lock
LOCK_IN = 0.20              # HH below the release over which a lock's own wave grows in (John 2026-11-28: "separate locks
                            # earlier"; it grew in with the side's wave, from z 0.38 to 0.12, so the locks parted only low)
LOCK_SWING = 0.5            # DRAFT: a lock's swing across the fall off its side's, as a share of the wave (at 1.0, with the
                            # phase sd 0.9 and the wavelength 12% off, round 4 read as crimp, not loose waves)
LOCK_LAM = 0.04            # DRAFT: a lock's wavelength within this share of its side's
LOCK_PHASE = 0.5            # rad: a lock's own wave phase against its side's (wavy_flow's jitter: a twelfth of a wave)
LOCK_DEPTH_MM = 3.0         # DRAFT: and the lock's own swing out of the fall with it (the locks part in depth)


TIP_T, TIP_K, TIP_CUT = 0.22, 0.85, 0.7   # DRAFT: the share of a lock's fall over which it draws to a tip, how far in, and how much
                                          # earlier its outermost hairs end
BANG_TIP = 1.0                            # DRAFT: how far a bang lock's hairs draw onto its centre line at the end


TEX = dict(amp=0.25, lam=0.10, tip_amp=0.35, tip_lam=0.15, sub_amp=0.22, sub_lam=0.55, hair_mm=1.2, layer_lam=0.8)   # DRAFT
LOCK_LONG = 1.2             # HH: the longest locks' fall (release to tip), against which a lock is 'short'


def lock_fall(Fd, seed=5, log=print):
    """Groups the hanging lines (face framing, front and back falls) into locks and rebuilds each line below the head
    as its lock's axis plus its own offset, gathered and then opened. Returns (Fd with new lines and 'lock', gauges)."""
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    lines = list(Fd['line']); basin = Fd['basin']; rel = Fd['release']
    lock = np.full(len(lines), -1)
    nl = 0
    fixed = {}
    zg = -0.05                                                        # (where the locks are read: below the jaw, in the fall)
    for b in (1, 2, 3):
        for sd in (1.0, -1.0):
            m = np.where((basin == b) & (np.array([l[0, 0] for l in lines]) * sd > 0))[0]
            if len(m) < 4:
                continue
            feat = []
            for i in m:
                l = lines[i]; r = l[min(rel[i], len(l) - 1)]
                q = l[np.argmin(np.abs(l[:, 2] - zg))] if l[-1, 2] < zg else l[-1]
                feat.append(np.r_[r, q])
            feat = np.array(feat)
            k = max(1, int(round(len(m) / (LOCK_SIZE * LOCK_SCALE[b]))))
            if b in FRONT_TRACE:
                # (traced on the photo, John 2026-11-28: "there are two big ones otherwise it's continuous on the
                # front": the fall is split by its order round the body into the traced locks, each with the traced
                # phase, and nothing finer)
                tr = FRONT_TRACE[b]
                rk = np.argsort(np.argsort(np.abs(SF.theta(feat[:, 3:6])))) / max(len(m) - 1, 1)
                lab = np.minimum((rk * len(tr)).astype(int), len(tr) - 1)
                if b == 1 and len(LAYER_END) > 1:
                    # (the face frame's layers are locks by where they end, not by their place round the body: each
                    # layer draws to its own tip at its own height, so the layers' ends show one above the other)
                    tr = LAYER_PHASE
                    ez = np.array([lines[i][-1, 2] for i in m])
                    cutz = RELEASE_Z - (RELEASE_Z - 0.5 * (LAYER_END[0] + LAYER_END[1])) * TAKE
                    cutz = RELEASE_Z - (RELEASE_Z - cutz) / TAKE
                    lab = (ez < cutz).astype(int)
                for j, ph_ in enumerate(tr):
                    fixed[nl + j] = ph_
                lock[m] = nl + lab
                nl += len(tr)
                continue
            cen = feat[rng.choice(len(m), k, replace=False)]
            for _ in range(6):                                            # (a few Lloyd steps: even, compact locks)
                lab = cKDTree(cen).query(feat)[1]
                cen = np.array([feat[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
            lock[m] = nl + lab
            nl += k
    rep = dict(locks=int(nl), lines_per_lock=round(float((lock >= 0).sum() / max(nl, 1)), 1))
    step = SEG_MM * MM
    for j in range(nl):
        m = np.where(lock == j)[0]
        if len(m) == 0:
            continue
        top = max(lines[i][min(rel[i], len(lines[i]) - 1), 2] for i in m)
        bot = min(lines[i][-1, 2] for i in m)
        zz = np.arange(top, bot - step, -step)
        # each member's fall as x(z), y(z): monotonic in z below its release (gauged: no line turns up)
        X = np.full((len(m), len(zz), 2), np.nan)
        for a_, i in enumerate(m):
            f = lines[i][rel[i]:]
            if len(f) < 2:
                continue
            o = np.argsort(-f[:, 2]); zf = -f[o, 2]
            ok = (zz <= f[0, 2] + 1e-9) & (zz >= f[-1, 2] - 1e-9)
            X[a_, ok, 0] = np.interp(-zz[ok], zf, f[o, 0]); X[a_, ok, 1] = np.interp(-zz[ok], zf, f[o, 1])
        # (each member is held at its last place past its end and weighed out over the last END_FADE of its fall, so
        # the axis does not jump where a shorter hair ends: a mean that loses a member in one step turns a corner)
        Xh = X.copy(); Wt = np.zeros(X.shape[:2])
        for a_ in range(len(m)):
            ok = ~np.isnan(X[a_, :, 0])
            if not ok.any():
                continue
            last = np.where(ok)[0][-1]; first = np.where(ok)[0][0]
            Xh[a_, last + 1:] = X[a_, last]; Xh[a_, :first] = X[a_, first]
            zl = zz[last]
            Wt[a_] = SF.sm(zl, zl + END_FADE, zz) * (np.arange(len(zz)) >= first)
        Wt = Wt + 1e-9
        ax = (Xh * Wt[..., None]).sum(0) / Wt.sum(0)[:, None]
        for c in range(2):
            ax[:, c] = gaussian_filter1d(ax[:, c], 4.0, mode='nearest')
        # the lock's own wave: a phase ahead of or behind its side's, out of the fall (radially from the body's axis)
        # (the side's wave is replaced by the lock's own: its phase and wavelength a little off its side's, the swing
        # across the fall as large as the side's and LOCK_DEPTH_MM out of it. With the side's wave alone every lock
        # waved in step and the fall read as one rippled sheet in the first render.)
        bl = int(basin[m[0]])
        en = SF.sm(RELEASE_Z, RELEASE_Z - LOCK_IN, zz) * LOCK_PLAY.get(bl, 1.0)                  # (from the release: the locks part high)
        ph = rng.normal(0, LOCK_PHASE); lam_l = WAVE['lam_hh'] * (1 + LOCK_LAM * rng.uniform(-1, 1))
        if j in fixed:
            ph, lam_l = fixed[j], WAVE['lam_hh']
        psi = 2 * np.pi * (WAVE_OUT_Z - zz) / WAVE['lam_hh']
        # (John 2026-11-29: "read how to vary frequency, amplitude and wave length to get layering and texture". What the
        # grooming tools do, and what is done here:
        #  - a clump has its own noise, "Noise Frequency" in "cycles per-clump", and "Noise Correlation" keeps
        #    "neighboring hairs" together (Maya XGen Clump modifier): each lock has its own wavelength and size
        #    (TEX['lam'], TEX['amp']: a share either way), its hairs waving with it;
        #  - "Noise Scale" is a ramp from root to tip and a curl's radius runs from "Factor Start" to "Factor End", its
        #    "Frequency" free to "vary for different points of the same curve" (XGen; Blender Curl Hair Curves): the
        #    wave grows toward the tip (TEX['tip_amp']) and its wavelength shortens (TEX['tip_lam']), as a curl
        #    tightens where the hair is lighter;
        #  - clumps are stacked, a secondary Clump at a higher "Density" under the primary (XGen workflow tips), and a
        #    curl takes a "Random Offset per curve": each sub-lock carries a smaller, shorter wave of its own
        #    (TEX['sub_amp'], TEX['sub_lam']) and each hair a small one about its sub-lock's (TEX['hair_mm']);
        #  - layers: a shorter lock's wave is also shorter (its wavelength scaled toward TEX['layer_lam'] by how much
        #    shorter than the longest it is), so the layers do not wave in step and each reads apart.)
        fr = np.clip((top - zz) / max(top - bot, 1e-6), 0, 1)
        short = float(np.clip(1 - (top - bot) / LOCK_LONG, 0, 1))
        amp_l = (1 + TEX['amp'] * rng.uniform(-1, 1)) * (1 + TEX['tip_amp'] * fr)
        lam_l = lam_l * (1 + TEX['lam'] * rng.uniform(-1, 1)) * (1 - (1 - TEX['layer_lam']) * short)
        if j in fixed:
            amp_l, lam_l = 1 + TEX['tip_amp'] * fr, WAVE['lam_hh']
        lam_z = lam_l * (1 - TEX['tip_lam'] * fr)
        psl = 2 * np.pi * (WAVE_OUT_Z - top) / lam_l + ph + np.cumsum(2 * np.pi * step / lam_z)
        ph_sub = rng.uniform(0, 2 * np.pi, SUBLOCKS)
        out = ax - np.array(SF.AXIS)[None]
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        tan = np.c_[-out[:, 1], out[:, 0]]
        ax2 = ax + out * (en * amp_l * LOCK_DEPTH_MM * MM * np.sin(psl + 0.5 * np.pi))[:, None] \
            + tan * (en * LOCK_SWING * WAVE['amp_hh'] * (amp_l * np.sin(psl) - np.sin(psi)))[:, None] * np.sign(np.mean(ax[:, 0]) + 1e-9)
        # sub-locks: by the side of the axis each member lies on at mid-fall
        mid = len(zz) // 2
        om = np.nan_to_num(X[:, mid] - ax[mid])
        ang = np.arctan2(om[:, 1], om[:, 0])
        far = np.linalg.norm(om, axis=1); far = np.argsort(np.argsort(far)) / max(len(far) - 1, 1)
        sub = np.floor(((ang - ang.min()) / max(np.ptp(ang), 1e-9)) * SUBLOCKS * 0.999).astype(int)
        for a_, i in enumerate(m):
            f = lines[i][rel[i]:]
            if len(f) < 2:
                continue
            ok = ~np.isnan(X[a_, :, 0])
            if ok.sum() < 3:
                continue
            zi = zz[ok]; Xi = X[a_, ok]; A = ax[ok]; A2 = ax2[ok]
            t = np.clip((zi[0] - zi) / max(zi[0] - zi[-1], 1e-6), 0, 1)                  # (0 at the release, 1 at the tip)
            gather = 1 - (1 - CLUMP_B.get(bl, CLUMP)) * SF.sm(zi[0], zi[0] - CLUMP_Z, zi)
            opn = SF.sm(OPEN_FROM, 1.0, t) * LOCK_PLAY.get(bl, 1.0)
            ss = sub == sub[a_]
            sc = (((Xh[ss] - ax[None]) * Wt[ss][..., None]).sum(0) / Wt[ss].sum(0)[:, None])[ok]
            sc = np.c_[[gaussian_filter1d(sc[:, c], 4.0, mode='nearest') for c in range(2)]].T
            own = (Xi - A) - sc
            # (John 2026-11-29: "make all locks to a tip like the picture". Over the last TIP_T of the lock's fall its
            # hairs draw in onto the axis (by TIP_K), and the hairs farthest from the axis end first (up to TIP_CUT of
            # that stretch earlier), so the lock ends in a point, not a cut edge.)
            tl = np.clip((top - zi) / max(top - bot, 1e-6), 0, 1)
            conv = 1 - TIP_K * SF.sm(1 - TIP_T, 1.0, tl)
            new = A2 + (sc * (gather + (SUB_OPEN - 1) * opn * gather)[:, None] + own * (gather * (1 + (HAIR_OPEN) * opn))[:, None]) * conv[:, None]
            # (the sub-lock's own wave and the hair's, across the fall and out of it a quarter wave on)
            ps_ = psl[ok] * (1.0 / TEX['sub_lam']) + ph_sub[sub[a_]]
            ph_ = ps_ * 1.3 + 2 * np.pi * ((i * 0.6180339) % 1.0)
            wv_ = (TEX['sub_amp'] * WAVE['amp_hh'] * np.sin(ps_) + TEX['hair_mm'] * MM * np.sin(ph_)) * (en[ok] * (1 + TEX['tip_amp'] * fr[ok]))
            wo_ = 0.5 * (TEX['sub_amp'] * WAVE['amp_hh'] * np.cos(ps_) + TEX['hair_mm'] * MM * np.cos(ph_)) * en[ok]
            new = new + (tan[ok] * wv_[:, None] + out[ok] * wo_[:, None]) * conv[:, None]
            if bl == 1 and LAYER_HOOK[0] > 0:
                # (a layer's end turns in toward the neck, the hook John drew at the jaw on the photo: the round
                # brush's turn at the ends, over the last LAYER_HOOK[1] of the layer's fall)
                new = new.copy()
                new[:, 0] -= np.sign(np.mean(ax[:, 0]) + 1e-9) * LAYER_HOOK[0] * MM * SF.sm(1 - LAYER_HOOK[1], 1.0, tl) ** 2
            p = np.c_[new, zi]
            keep = zi >= bot + TIP_CUT * TIP_T * (top - bot) * far[a_]
            if keep.sum() >= 4:
                p = p[keep]
            p[:, :2] = p[:, :2] + (lines[i][rel[i], :2] - p[0, :2]) * np.exp(-np.arange(len(p)) / 6.0)[:, None]
            lines[i] = _join(np.r_[lines[i][:rel[i]], p], rel[i])
    out_ = dict(Fd); out_['line'] = lines; out_['lock'] = lock
    log('lock_fall %s' % rep)
    return out_, rep


FCOL = {0: [(217, 138, 31), (180, 100, 10)], 1: [(194, 51, 122), (150, 30, 90)],
        2: [(47, 111, 179), (111, 163, 216), (29, 74, 128)], 3: [(44, 156, 90), (122, 79, 201), (20, 120, 110)], -1: [(150, 150, 150)]}


def draw2(Fd, head_pts, depth, path, views=((0, 4, 'front'), (40, 6, 'three quarter'), (90, 4, 'side'), (180, 4, 'back'), (0, 80, 'top')),
          every=5, px=330):
    """The field as lines over a clay head and body: fringe orange, face framing magenta, the front fall in blues and
    the back fall in greens and purples (the shade changes every section, so the order across a fall can be followed),
    and in red the line where the hair leaves the head."""
    from PIL import Image, ImageDraw
    W, H = int(1.9 * px), int(2.5 * px)
    im = Image.new('RGB', (W * len(views), H + 40), (250, 249, 247))
    dr = ImageDraw.Draw(im)
    pc = head_pts[head_pts[:, 2] > -0.95]
    for v, (az, el, name) in enumerate(views):
        a, e = math.radians(az), math.radians(el)
        fwd = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        right = np.cross([0, 0, 1.0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        P = lambda X: np.c_[v * W + W / 2 + (X @ right) * px, 40 + H * 0.36 - (X @ up - 0.5) * px]

        def hid(X):
            t = np.linspace(0.02, 1.3, 22)[None, :, None]
            Q = (X[:, None] + fwd[None, None] * t).reshape(-1, 3)
            return (depth(Q) > 0.004).reshape(len(X), -1).any(1)
        nrm = pc - np.c_[np.zeros(len(pc)), np.full(len(pc), 0.06), np.minimum(pc[:, 2], SF.CENTRE[2])]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        sh = np.clip(0.74 + 0.26 * (nrm @ (fwd + 0.5 * up - 0.4 * right) / 1.2), 0.5, 1.0)
        vis = ~hid(pc + nrm * 0.006)
        for (x_, y_), s_ in zip(P(pc)[vis], sh[vis]):
            g = int(225 * s_); dr.ellipse((x_ - 2.5, y_ - 2.5, x_ + 2.5, y_ + 2.5), fill=(g, int(g * 0.98), int(g * 0.95)))
        items = []
        for i in range(0, len(Fd['line']), every):
            X = Fd['line'][i][::2]
            if len(X) < 2:
                continue
            cs = FCOL[int(Fd['basin'][i])]; col = cs[int(Fd['sec'][i]) % len(cs)]
            h = hid(X); q = P(X); z = X @ fwd
            for j in range(len(X) - 1):
                if not (h[j] or h[j + 1]):
                    items.append((0.5 * (z[j] + z[j + 1]), tuple(q[j]), tuple(q[j + 1]), col, 1))
            r = Fd['release'][i]
            if Fd['basin'][i] > 0 and 0 < r < len(Fd['line'][i]) - 1:
                Y = Fd['line'][i][r - 1:r + 1]
                if not hid(Y).any():
                    items.append((float(Y[0] @ fwd) + 0.01, tuple(P(Y)[0]), tuple(P(Y)[1]), (224, 32, 32), 3))
        for z, p0, p1, col, w in sorted(items, key=lambda t: t[0]):
            dr.line([p0, p1], fill=col, width=w)
        dr.text((v * W + 12, 12), name, fill=(0, 0, 0))
    im.save(path)
