"""The bare figure's anatomy below the approved neck and trapezius: the deltoid and the arm's muscles, the pelvis and
hips, the legs and bare feet, made one skin with the upper body.

Proportions (head height H = 8.6, figure 63 tall, so 1 unit is about 2.6 cm on a 165 cm woman):
  * deltoid: from the collarbone, acromion and scapular spine, round over the top of the arm and down to its
    insertion about halfway down the upper arm, widest a little below the shoulder point, so the line from the
    neck runs down over a curve into the arm. It stands about 1.7 past the shoulder point.
  * upper arm about 28 cm round (radius 1.7), biceps in front and triceps behind; forearm fullest just below the
    elbow (24 cm, radius 1.5), drawn in to a 15 cm wrist.
  * waist about one head across, the iliac crest just above the hip joint, the widest point of the hips at the
    great trochanters (about 1.5 heads across), the glutes fullest at the level of the crotch.
  * thigh about 51 cm round at the top (radius 3.1), knee 1.95, calf fullest a third of the way down the shin
    (outer head higher, inner head lower), ankle about 20 cm round, the inner ankle bone higher than the outer.
  * foot about 1.07 heads long (15% of the height) and 37% as wide as it is long.

Everything below the waist is built here as closed shapes and joined to the upper body in one volume (a voxel
remesh), with a rounded fillet (a smooth union) where the thighs meet the pelvis and the feet the ankles. In the rest
pose the arms hang straight down through the hips (the poses hold them 10 degrees out), so the join is made with the
arms swung out by ARM_SPREAD and the skin is put back afterwards through its own bone weights."""
import math
import os

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import female

ON = os.environ.get('ANATOMY', '1') != '0'
# The foot: 'measured' builds it from the measured proportions of the female foot (foot.py), with the ankle joint
# at its measured height (John, 2026-11-21); 'old' keeps the first hand-set foot and the rig's ankle at 3.4.
FOOT_MODEL = os.environ.get('FOOT_MODEL', 'measured')
# ---------------------------------------------------------------- the canon: every size from the head
# The head as it sits on the body (the Face Creator's head, and the mannequin's fitted to the same size) is the
# unit. H is its height, chin to crown; F the face, hairline to chin, 5/6 of H (Richer's and the classical canon).
# Both are measured on the head the figure is given (configure(), called by the figure before the body is built);
# the numbers here are only the mannequin head's, for a module loaded on its own.
SEED = globals().get('SEED', {'H': 9.39, 'chin': 53.42})
HEAD = float(os.environ.get('CANON_H', SEED['H']))
FACE = HEAD * 5.0 / 6.0
CHIN = SEED['chin']
CANON = {
    'height': 7.5 * HEAD,                 # the figure, crown to floor
    'shoulders': 1.68 * HEAD,             # across the deltoids (a woman's: 1 2/3 H; a man's 2 H)
    'waist': 1.0 * HEAD,
    'hips': float(os.environ.get('HIPS_H', '1.42')) * HEAD,                   # across the great trochanters, the widest point
    'crotch': CHIN - 3.0 * HEAD,          # chin to nipples, navel and crotch: 1 H each
    'thigh': 2.0 * HEAD,                  # top of the femur to the knee's joint, and the knee to the floor
    'hand': FACE,                         # the hand is as long as the face
    'foot': 1.05 * HEAD,                  # the foot a little longer than the head
}
# John's choice (2026-11-18): the figure stays 63 tall, 6.7 heads, on its own rig; only the widths, the hands and
# the feet follow the head. CANON_LEGS=1 builds the 7.5-head legs instead (a shape study, off the walk's rig).
LONG = bool(os.environ.get('CANON_LEGS'))
FLOOR_Z = CHIN + HEAD - CANON['height'] if LONG else 0.0
HIP_JOINT = FLOOR_Z + 4.0 * HEAD if LONG else female.HIP_Z
if not LONG:
    CANON['crotch'] = 26.2
    CANON['thigh'] = female.THIGH
MEASURED_FOOT = FOOT_MODEL == 'measured'
if MEASURED_FOOT:
    import foot as _foot
    # the ankle joint's centre at the middle of the two malleoli (Wu et al. 2002, the ISB's joint centre), at their
    # measured heights (Hajaghazadeh 2018: 0.247 and 0.275 of the foot's length)
    ANKLE_H = 0.5 * (_foot.R['lat_mall'] + _foot.R['med_mall']) * CANON['foot']
    # the ball's reach in front of the ankle (the rocker the walk lifts the heel about): the metatarsal heads'
    # middle, Lee 2014's 0.728 and 0.658 of the length from the heel, less the heel's 0.206 behind the joint
    FOOT_BALL_M = (0.5 * (_foot.R['ball_len'] + _foot.R['out_ball_len'])
                   - 0.5 * (_foot.R['heel_lat_mall'] + _foot.R['heel_med_mall'])) * CANON['foot']
else:
    ANKLE_H = 3.4
print('anatomy canon: H %.2f F %.2f floor %.2f hip joint %.2f knee %.2f crotch %.2f' % (
    HEAD, FACE, FLOOR_Z, HIP_JOINT, HIP_JOINT - 2 * HEAD, CANON['crotch']), flush=True)
VOXEL = float(os.environ.get('ANAT_VOXEL', '0.14'))
ARM_SPREAD = 0.42
K_HIP = float(os.environ.get('K_HIP', '2.0'))
K_ANKLE = float(os.environ.get('K_ANKLE', '1.2'))   # (the leg into the foot: a wide fillet, the front of the ankle one curve with the instep)
ANKLE_FRONT_ITERS = int(os.environ.get('ANKLE_FRONT_ITERS', '0'))
FOOT_SMOOTH = int(os.environ.get('FOOT_SMOOTH', '60'))
CUT_Z = 46.0           # the upper body's skin is kept above this; the pelvis is built from here down
KEEP_Z = 47.0          # above this the new skin is put back exactly on the upper body's


def _ss(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _ssn(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _table(rows, z):
    """A smooth curve down a table of (z, values...) rows, z descending: a cubic through the rows with slopes from
    their neighbours (Catmull-Rom on uneven spacing), so the outline has no flat steps at the rows."""
    if z >= rows[0][0]:
        return rows[0][1:]
    if z <= rows[-1][0]:
        return rows[-1][1:]
    n = len(rows)
    for i in range(n - 1):
        a, b = rows[i], rows[i + 1]
        if b[0] <= z <= a[0]:
            h = a[0] - b[0]
            t = (a[0] - z) / h
            p = rows[i - 1] if i > 0 else a
            q = rows[i + 2] if i + 2 < n else b
            out = []
            for k in range(1, len(a)):
                m0 = (b[k] - p[k]) / max(p[0] - b[0], 1e-6) * h if i > 0 else (b[k] - a[k])
                m1 = (q[k] - a[k]) / max(a[0] - q[0], 1e-6) * h if i + 2 < n else (b[k] - a[k])
                t2, t3 = t * t, t * t * t
                out.append((2 * t3 - 3 * t2 + 1) * a[k] + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * b[k]
                           + (t3 - t2) * m1)
            return tuple(out)
    return rows[-1][1:]


# ---------------------------------------------------------------- arms
U, F = female.UPPER, female.FORE
# The arm is not one tube: each part has its own section, wider one way than the other, and every size is a share of
# the shoulder's (S, the arm's radius at the shoulder joint, under the deltoid), so the arm follows the shoulder it
# hangs from (John, 2026-11-18). Rows: local height (0 at the shoulder joint), half width across, depth in front of
# the bone, depth behind it.
#   the deltoid   the widest point of the arm and of the figure, round over the joint (kept as approved above -1)
#   the triceps   behind the upper arm, fullest two fifths of the way down, a little deeper than the shoulder itself,
#                 then flat into its tendon over the lower third, so the back of the arm runs in to the elbow
#   the biceps    in front, fullest lower, three fifths of the way down
#   the elbow     the narrowest depth, wide across (the two knobs of the humerus), the point of the elbow behind
#   the forearm   its muscle in the upper half: fullest a fifth of the way down (0.86 of the shoulder across, 0.8
#                 deep), still 0.73 at the middle, then running into tendon and the wrist's oval over the lower
#                 half, so the depth seen from the side falls in one line (0.8, 0.73, 0.7, the wrist's 0.67) and
#                 the width seen from the front tapers faster. With the wider wrist the old forearm (0.62 deep at
#                 its middle, less than the wrist) read as bone (John, 2026-11-18). Over that, two masses
#                 (forearm()): the brachioradialis and the extensors, from above the elbow down the outside and
#                 front, and the flexors on the inside, a little lower.
#   the wrist     an oval lying the other way: thin across and deep front to back, as the palm faces the thigh
S_ARM = 1.82
# The wrist comes from the hand, and the hand from the head (John, 2026-11-18: the hand looked too big for the wrist
# from the side). The hand is as long as the face; the palm is 0.4 of that across; the wrist is a flat oval 0.78 of
# the palm wide, as the palm lies (front to back, with the palm to the thigh), and 0.65 of its own width thick. The
# forearm runs down into that oval and the heel of the hand opens out of it, so from the side the arm widens into
# the hand in one line instead of a thin stalk carrying a paddle. (It was 1.45 wide under a palm of 3.1.)
PALM_W = 0.2 * CANON['hand']           # half widths
WRIST_W = 0.78 * PALM_W
WRIST_T = 0.65 * WRIST_W
ARM_SEC = [(1.5, 0.3, 0.276, 0.276), (1.25, 1.2, 1.104, 1.104), (0.8, 1.72, 1.58, 1.58), (0.0, S_ARM, 0.92 * S_ARM, 0.92 * S_ARM),
           (-1.0, 0.972 * S_ARM, 0.895 * S_ARM, 0.9 * S_ARM),
           (-0.23 * U, 0.90 * S_ARM, 0.86 * S_ARM, 0.96 * S_ARM), (-0.41 * U, 0.80 * S_ARM, 0.85 * S_ARM, 1.0 * S_ARM),
           (-0.61 * U, 0.74 * S_ARM, 0.84 * S_ARM, 0.86 * S_ARM), (-0.82 * U, 0.70 * S_ARM, 0.74 * S_ARM, 0.73 * S_ARM),
           (-U, 0.74 * S_ARM, 0.64 * S_ARM, 0.71 * S_ARM),
           (-U - 0.22 * F, 0.86 * S_ARM, 0.80 * S_ARM, 0.78 * S_ARM), (-U - 0.5 * F, 0.74 * S_ARM, 0.73 * S_ARM, 0.71 * S_ARM),
           (-U - 0.8 * F, 0.52 * S_ARM, 0.5 * (0.73 * S_ARM + WRIST_W), 0.5 * (0.71 * S_ARM + WRIST_W)),
           (-U - F, WRIST_T, WRIST_W, WRIST_W), (-U - F - 0.4, WRIST_T, WRIST_W, WRIST_W)]
# (the one radius other code asks for, the hole a sleeve needs and the ribs' clearance: the section's widest)
ARM_R = [(r[0], max(r[1:])) for r in ARM_SEC]


def arm_half(z):
    """The arm's half width across at local height z."""
    return _table(ARM_SEC, z)[0]


def deltoid(z, phi, s):
    """The deltoid over the arm's section: its round cap over the top of the joint (the base figure's, where the
    trapezius was fitted to it) and below it the muscle's belly, wide over the front, outside and back near the top
    and drawn to a point on the outside at its insertion."""
    out = s * math.sin(phi)
    wrap = max(0.0, 0.45 + 0.55 * out) ** 1.6
    r = 0.42 * wrap * math.exp(-((z + 1.4) / 2.1) ** 2) * (1.0 if z < 0.6 else math.exp(-((z - 0.6) / 0.7) ** 2))
    # (the belly: its edge closes from round the front and back of the arm to the outside as it runs down)
    c = -0.35 + 0.95 * _ss(-1.0, -4.6, z)
    ang = max(0.0, (out - c) / (1.0 - c)) ** 0.8
    r += 0.12 * ang * math.exp(-((z + 2.5) / 2.0) ** 2) * (1.0 - _ss(-4.2, -6.0, z)) * _ss(0.6, -1.0, z)
    return r


def forearm(z, phi, s):
    """The forearm's two muscle masses over its section, as shares of the shoulder's radius: the brachioradialis
    and extensors on the outside and front, rising above the elbow's outer side and fullest just below it; the
    flexors on the inside, fullest a quarter of the way down. Both run out before the wrist."""
    o_, fr = s * math.sin(phi), math.cos(phi)
    k = float(os.environ.get('FOREARM_K', '1.0'))
    r = 0.09 * S_ARM * math.exp(-((z + U + 0.16 * F) / (0.2 * F)) ** 2) * max(0.0, 0.75 * o_ + 0.55 * fr) ** 1.5
    r += 0.08 * S_ARM * math.exp(-((z + U + 0.27 * F) / (0.2 * F)) ** 2) * max(0.0, -0.85 * o_ + 0.3 * fr) ** 1.5
    return k * r


def arm_xy(z, phi, s, out=0.0):
    """The bare arm's surface at local height z and phi (0 the front), pushed out by `out`."""
    a, bf, bb = _table(ARM_SEC, z)
    sn, c = math.sin(phi), math.cos(phi)
    d = deltoid(z, phi, s) + forearm(z, phi, s) + out
    return -(a + d) * sn, -((bf if c >= 0 else bb) + 0.92 * d) * c


def patch():
    if not ON:
        return
    female.BARE_ARM_R[:] = ARM_R
    female.deltoid = deltoid
    female.arm_xy = arm_xy
    female.ARM_OUT, female.ARM_CARRY, female.IDLE_ARM = ARM_OUT, ARM_CARRY, dict(IDLE_ARM)
    if LONG:
        # (the rig's legs are the canon's: the knee and the ankle joint where the body's are, the floor under them)
        female.THIGH, female.SHIN, female.ANKLE_Z = TH, SH, FLOOR_Z + 3.4
        # (and the world walk's stride follows the longer legs: a cycle of 52 instead of the 46 the shorter-legged
        # figures take, so the steps keep the same length against the leg and the standing knee stays nearly
        # straight; a world model of this figure needs its own walk cycle in src/world/main.js to match)
        female.STRIDE = 52.0
        # (the bare foot is longer than a boot and has no toe joint, so it rolls up nearly over the tips of its toes
        # to push off: rolling over a boot's ball put the toes 1.9 into the floor; at 5.8 they stay within 0.1 of it)
        female.FOOT_BALL = 5.8
    elif MEASURED_FOOT:
        # (the ankle joint at its measured height: the shin longer by what the ankle comes down, the slack kept)
        female.ANKLE_Z = FLOOR_Z + ANKLE_H
        female.SHIN = 13.8 + (3.4 - ANKLE_H)
        female.FOOT_BALL = FOOT_BALL_M


# ---------------------------------------------------------------- pelvis
# (z, half width, front depth, back depth) of the trunk from the waist down: the iliac crest just above the hip
# joint, the widest point at the trochanters, then the glutes and the crotch
def waist_wide(z):
    """How much the trunk is widened about the waist (the base trunk's waist was 0.78 H across; the canon's is 1 H)."""
    return 1.0 + (CANON['waist'] / 2 / 3.65 - 1.0) * math.exp(-((z - 39.4) / 3.0) ** 2)


HW = CANON['hips'] / 2
WW = float(os.environ.get('WAIST_W', '4.05'))
# The trunk's outline from under the arm to the crotch is one S-curve (Bammes, Richer): the ribcage holds its width
# down to the tenth rib, turns in to the waist (the narrowest point, a little above the navel, in the soft gap
# between the ribs and the pelvis), then swells out over the iliac crest and on in one convex sweep to the
# widest point at the great trochanters, level with the crotch's top. Waist to hips 0.62 across (about 0.7 round).
KNEE_Z = HIP_JOINT - CANON['thigh']
# The figure's outline from the waist to the knee, one curve (half widths): the pelvis and the thigh both take
# their outer side from it, so the thigh cannot stand out from the hip or fall inside it where they join.
OUTLINE = [(39.6, WW), (38.6, WW + 0.15), (37.2, WW + 0.7), (35.4, WW + 1.65), (33.6, HW - 0.2), (31.8, HW),
           (30.0, HW), (28.0, HW - 0.15), (CANON['crotch'], HW - 0.45),
           (CANON['crotch'] + (KNEE_Z - CANON['crotch']) * 0.23, HW - 0.8),
           (CANON['crotch'] + (KNEE_Z - CANON['crotch']) * 0.52, HW - 1.3),
           (CANON['crotch'] + (KNEE_Z - CANON['crotch']) * 0.8, HW - 1.75), (KNEE_Z, HW - 1.95)]


def out_w(z):
    return _table(OUTLINE, z)[0]


# The back, seen from the side, as one line from the neck to the thigh (John, 2026-11-18: the arch of the back is set
# by the bust, the waist, the hips and the neck; the seat by the waist, the hips and the thighs). Its depth behind
# the body's axis at each landmark is a share of the body's width there, and the line swings between them in single
# half waves with no flat run and no corner:
#   the neck   where the approved neck and trapezius end (left as they are above it), leaving at their own slope
#   the bust   the upper back is fullest over the shoulder blades, just above the bust: 0.75 of the ribs' half width
#   the waist  the hollow of the lower back is deepest at the waist: 0.62 of its half width
#   the hips   the seat is fullest a third of the way up from the crotch to the waist: 0.64 of the hips' half width
#   the thigh  as deep as it is wide, at the fold
THIGH_W = (out_w(CANON['crotch']) - 0.35) / 2          # the thigh's half width at the crotch (0.35: the gap's half)
SEAT_Z = CANON['crotch'] + 0.32 * (39.6 - CANON['crotch'])        # the fullest point, a third of the way up to the waist
BUST_Z = 44.3
NECK_Z, NECK_D, NECK_SLOPE = 50.0, 2.97, 0.25         # the back of the neck's base on the approved skin (measured)
BLADE_Z = BUST_Z + 0.4 * (NECK_Z - BUST_Z)            # the shoulder blades: two fifths of the way from bust to neck
WAIST_Z = 39.6
D_SEAT = 0.64 * HW
D_WAIST = 0.62 * WW
_SEAT = [(SEAT_Z + 1.6, D_SEAT - 0.12), (SEAT_Z, D_SEAT), (CANON['crotch'] + 1.6, D_SEAT - 0.12),
         (CANON['crotch'], 0.5 * (D_SEAT + THIGH_W) + 0.05),
         (CANON['crotch'] + (KNEE_Z - CANON['crotch']) * 0.1, THIGH_W + 0.1),
         (CANON['crotch'] + (KNEE_Z - CANON['crotch']) * 0.23, THIGH_W - 0.15)]


def _wave(a, b, t):
    return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, t))))


def back_d(z):
    d_blade = float(os.environ.get('BLADE_K', '0.75')) * BUST_HALF
    if z >= NECK_Z:
        return NECK_D
    if z >= BLADE_Z:                 # out of the neck's own slope, level over the blades (a Hermite span)
        L = NECK_Z - BLADE_Z
        t = (NECK_Z - z) / L
        h00, h10, h01 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2
        return h00 * NECK_D + h10 * L * NECK_SLOPE + h01 * d_blade
    if z >= WAIST_Z:
        return _wave(d_blade, D_WAIST, (BLADE_Z - z) / (BLADE_Z - WAIST_Z))
    if z >= SEAT_Z:
        return _wave(D_WAIST, D_SEAT, (WAIST_Z - z) / (WAIST_Z - SEAT_Z))
    return _table(_SEAT, z)[0]


# The arm at rest hangs from the shoulder, not out from it: the deltoid is the figure's widest point and the upper arm
# falls from under it almost straight down, against the ribs (4 degrees out), so the elbow lies in the waist's hollow
# inside the line of the shoulder. The forearm then angles out from the elbow (the carrying angle, larger in women)
# just far enough for the hand to pass the hip, and the elbow is never locked: the upper arm hangs a touch behind the
# shoulder and the forearm comes forward, so the hand rests at the front of the thigh's side.
ARM_OUT = float(os.environ.get('ARM_OUT', '0.07'))
ARM_CARRY = float(os.environ.get('ARM_CARRY', '0.07'))
IDLE_ARM = {1: (-0.02, 0.36), -1: (-0.07, 0.26)}


def rib_w(z, gap=0.05):
    """The ribcage's half width under the arm: out to where the hanging arm's inner side lies, so the arm rests
    against the ribs instead of in a channel cut for it or sunk into them. (The ribcage is an egg, narrow at the
    armpit and widest at the eighth to tenth ribs, where the arm has swung clear of it.)"""
    return female.SH_X + (female.SH_Z - z) * math.sin(ARM_OUT) - arm_half(z - female.SH_Z) - gap


BUST_HALF = min(rib_w(44.5), WW + 0.62)        # the ribs' half width at the bust

# (the top two rows lie inside the upper body, so the join has no ledge: the upper body's own skin shows there)
PELVIS = [(48.0, 3.6, 2.0, 2.2), (47.0, 4.2, 2.65, 2.8), (46.0, rib_w(46.0), 2.95, back_d(46.0)), (44.5, BUST_HALF, 3.0, back_d(44.5)),
          (43.0, WW + 0.6, 2.95, back_d(43.0)), (41.6, WW + 0.38, 2.8, back_d(41.6)), (40.4, WW + 0.15, 2.6, back_d(40.4)), (39.6, WW, 2.5, back_d(39.6)), (38.6, out_w(38.6), 2.55, back_d(38.6)),
          (37.2, out_w(37.2), 2.75, back_d(37.2)), (35.4, out_w(35.4), 3.0, back_d(35.4)),
          (33.6, out_w(33.6), 3.25, back_d(33.6)), (31.6, out_w(31.6), 3.25, back_d(31.6)),
          (30.0, out_w(30.0), 3.1, back_d(30.0)), (28.6, out_w(28.6), 2.8, back_d(28.6)),
          (CANON['crotch'] + 1.9, 5.2, 2.3, back_d(CANON['crotch'] + 1.9) - 0.3), (CANON['crotch'] + 0.9, 3.4, 1.7, 3.3), (CANON['crotch'] + 0.3, 1.8, 1.0, 1.3),
          (CANON['crotch'], 0.3, 0.3, 0.3)]
# (the widths and depths above were set on a head 9.39 tall; they are kept as ratios of the head, eased back to
# the upper body's own sizes where the two join under the arms)
UNIT = HEAD / 9.39
PELVIS = [(r[0],) + tuple(v * (1.0 + (UNIT - 1.0) * _ss(46.0, 42.0, r[0])) for v in r[1:]) for r in PELVIS]


def pelvis_point(z, phi):
    rx, fy, by = _table(PELVIS, z)
    sx, c = math.sin(phi), math.cos(phi)
    # under the arm the chest is not an oval: the chest's muscle in front and the back's behind run out to the arm as
    # the two walls of the armpit, so the section is squarer there (as the body above the join is), easing to the
    # waist's oval. An oval under a square section left a ledge across the chest and the back at the join.
    e = 2.0 / (2.0 + float(os.environ.get('PIT_SQ', '1.0')) * _ss(42.5, 46.5, z))
    sx, c = math.copysign(abs(sx) ** e, sx), math.copysign(abs(c) ** e, c)
    x = -rx * sx
    if c >= 0:
        y = -fy * c
        # (the lower belly: a soft round below the navel)
        y -= 0.12 * math.exp(-(x / 2.6) ** 2 - ((z - 34.5) / 2.0) ** 2) * c
        if z > 40.0:
            for s in (-1, 1):    # (as torso_point: the bust's foot, where it sits on the ribs)
                sz = 2.3 if z > 44.3 else 2.0     # (a softer underside: the tighter one read as a ledge)
                y -= 1.25 * math.exp(-((x - s * 2.2) / 2.2) ** 2 - ((z - 44.3) / sz) ** 2) * c
    else:
        y = -by * c
        # the glutes: two rounds either side of the cleft, fullest low, at the crotch
        for s in (-1, 1):
            y -= 0.0 * c
        y += 0.18 * math.exp(-(x / 0.8) ** 2) * _ss(31.5, 29.0, z) * (-c)    # (the cleft)
    return x, y


# ---------------------------------------------------------------- legs
HZ = female.HIP_Z
TH = HZ - (HIP_JOINT - CANON['thigh'])            # the rig's hip pivot down to the canon's knee
SH = (HIP_JOINT - CANON['thigh']) - (FLOOR_Z + 3.4) if LONG else (13.8 + (3.4 - ANKLE_H) if MEASURED_FOOT else female.SHIN)
# (local z from the hip pivot, outer, inner, front and back radius, centre x out from the midline)
_CR = CANON['crotch'] - HZ          # (the crotch, local)


def _tz(f):
    """A height down the thigh: 0 at the crotch, 1 at the knee."""
    return _CR + (-TH - _CR) * f


def _sz(f):
    """A height down the shin: 0 at the knee, 1 at the ankle joint."""
    return -TH - SH * f


def _ro(lz, cx, inset=0.0):
    """The thigh's outer radius at local height lz, from the figure's outline."""
    return out_w(HZ + lz) / UNIT_ - cx - inset


UNIT_ = HEAD / 9.39


def _rb(lz, inset=0.0):
    """The thigh's back radius at local height lz, from the seat's curve."""
    return back_d(HZ + lz) / UNIT_ - inset


LEG = [(1.8, 1.2, 1.2, 1.2, 1.4, 3.6), (1.0, 2.0, 2.0, 1.9, 2.2, 3.65), (0.0, _ro(0.0, 3.7, 0.25), 2.6, 2.5, _rb(0.0, 0.9), 3.7),
       (-1.5, _ro(-1.5, 3.75, 0.05), 3.0, 2.9, _rb(-1.5, 0.35), 3.75), (-3.3, _ro(-3.3, 3.75), 3.25, 3.15, _rb(-3.3), 3.75),
       (_CR + 0.9, _ro(_CR + 0.9, 3.7), 3.3, 3.2, _rb(_CR + 0.9), 3.7), (_CR, _ro(_CR, 3.65), 3.3, 3.2, _rb(_CR), 3.65),
       (_tz(0.1), _ro(_tz(0.1), 3.55), 3.1, 3.15, _rb(_tz(0.1)), 3.55),
       (_tz(0.23), _ro(_tz(0.23), 3.45), 2.85, 3.1, _rb(_tz(0.23)), 3.45), (_tz(0.52), _ro(_tz(0.52), 3.2), 2.35, 2.7, 2.45, 3.2),
       (_tz(0.8), _ro(_tz(0.8), 3.0), 1.95, 2.2, 2.0, 3.0),
       (-TH, _ro(-TH, 2.9), 1.8, 2.0, 1.8, 2.9), (_sz(0.13), 1.7, 1.7, 1.8, 1.95, 2.85), (_sz(0.31), 1.9, 1.85, 1.7, 2.4, 2.8),
       (_sz(0.45), 1.75, 1.9, 1.6, 2.3, 2.8), (_sz(0.68), 1.35, 1.4, 1.4, 1.65, 2.8),
       (_sz(0.88), 1.1, 1.1, 1.25, 1.3, 2.8), (-TH - SH + 0.6, 1.05, 1.1, 1.25, 1.4, 2.8),
       (-TH - SH - 1.2, 1.05, 1.1, 1.25, 1.4, 2.8)]
LEG = [(r[0],) + tuple(v * UNIT for v in r[1:]) for r in LEG]


def leg_point(z, phi, s):
    """The leg's surface at local height z (from the hip joint) and phi (from the front toward her right), for the
    leg on side s (1 her right). Returns world x, y (the hip joint at x = -s * LEG_X)."""
    ro, ri, rf, rb, cx = _table(LEG, z)
    sp, c = math.sin(phi), math.cos(phi)
    out = s * sp
    rx = ro if out > 0 else ri
    ry = rf if c > 0 else rb
    x = -s * cx - rx * sp
    y = -ry * c
    # the knee: the kneecap over the front, the hollows either side of its tendon below it
    y -= 0.22 * math.exp(-((z + TH + 0.1) / 0.85) ** 2) * max(0.0, c) ** 3
    # the inner thigh's teardrop above the knee (vastus medialis)
    x += s * 0.18 * math.exp(-((z + TH - 1.8) / 1.3) ** 2) * max(0.0, -out) * max(0.0, c + 0.3)
    # the ankle bones: the inner one higher and further forward than the outer
    # (measured: each half the malleoli's height difference either side of the joint, Hajaghazadeh 2018)
    if MEASURED_FOOT:
        dm = 0.5 * (_foot.R['med_mall'] - _foot.R['lat_mall']) * CANON['foot']
        z_in, z_out = z + TH + SH - 0.2 - dm, z + TH + SH - 0.2 + dm        # (0.2: the shin's slack past the joint)
    else:
        z_in, z_out = z + TH + SH - 0.1, z + TH + SH + 0.4
    ab = 0.16 * math.exp(-(z_in / 0.5) ** 2) * max(0.0, -out) ** 4
    ab += 0.16 * math.exp(-(z_out / 0.5) ** 2) * max(0.0, out) ** 4
    x -= s * ab * (1 if out > 0 else -1)
    return x, y


def leg_bm(s):
    zs = [1.8 - 0.2 * i for i in range(int((1.8 + TH + SH + 1.2) / 0.2) + 1)]
    bm = female.loft_fn(zs, lambda z, p: leg_point(z, p, s), n=48)
    bmesh.ops.translate(bm, vec=(0, 0, female.HIP_Z), verts=bm.verts)
    return bm


# ---------------------------------------------------------------- feet
# rings along the foot, heel to toe: (y, half width, top, bottom, centre x shift toward the midline)
FOOT = [(2.35, 0.35, 1.5, 0.55, 0.0), (2.1, 0.85, 2.4, 0.15, 0.0), (1.4, 1.1, 3.6, 0.0, 0.0), (0.4, 1.18, 4.6, 0.02, 0.0),
        (-0.8, 1.32, 3.7, 0.12, 0.05), (-2.4, 1.52, 2.75, 0.2, 0.1), (-4.0, 1.7, 2.0, 0.0, 0.2),
        (-5.3, 1.66, 1.4, 0.0, 0.3), (-6.2, 1.45, 1.05, 0.05, 0.35), (-6.75, 0.95, 0.8, 0.15, 0.4),
        (-6.95, 0.35, 0.62, 0.3, 0.4)]
FOOT_K = CANON['foot'] / 9.3


def foot_bm(s):
    cx0 = -s * 2.8
    bm = bmesh.new()
    loops = []
    n = 40
    ys = []
    for (ya, *_), (yb, *_) in zip(FOOT, FOOT[1:]):
        for i in range(6):
            ys.append(ya + (yb - ya) * i / 6)
    ys.append(FOOT[-1][0])
    rows = [(y,) + tuple(_table(FOOT, y)) for y in ys]
    for y, hw, top, bot, sh in rows:
        mid = 0.5 * (top + bot) if top - bot < 2.0 else bot + 1.0
        loop = []
        for k in range(n):
            a = 2 * math.pi * k / n
            ca, sa = math.cos(a), math.sin(a)
            if sa >= 0:
                z = mid + (top - mid) * sa
                xx = hw * ca
            else:
                e = abs(sa) ** 0.35          # (a flat sole, rounded at its edges)
                z = mid - (mid - bot) * e
                xx = hw * math.copysign(abs(ca) ** 0.6, ca)
            # (the arch: the inner side of the sole raised under the instep)
            inner = max(0.0, (s * xx) / max(hw, 1e-6))
            z += 0.45 * inner * math.exp(-((y + 2.0) / 1.4) ** 2) * (sa < 0) * (1 - abs(sa)) ** 0.3
            loop.append(bm.verts.new((cx0 + s * sh + xx * 1.1, y * FOOT_K, FLOOR_Z + z)))
        loops.append(loop)
    for a, b in zip(loops, loops[1:]):
        for k in range(n):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    bm.faces.new(list(reversed(loops[0])))
    bm.faces.new(loops[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


# ---------------------------------------------------------------- the join
def _closed(bm):
    edges = [e for e in bm.edges if e.is_boundary]
    if edges:
        bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _tree(bm):
    return BVHTree.FromBMesh(bm)


def _sdf(tree, p):
    loc, nrm, _, d = tree.find_nearest(p)
    if loc is None:
        return 1e3, Vector((0, 0, 1))
    v = p - loc
    sg = 1.0 if v.dot(nrm) >= 0.0 else -1.0
    g = v / d * sg if d > 1e-6 else nrm
    return sg * d, g


def _smin(a, ga, b, gb, k):
    if k <= 1e-4:
        return (a, ga) if a < b else (b, gb)
    h = min(1.0, max(0.0, 0.5 + 0.5 * (b - a) / k))
    f = b + (a - b) * h - k * h * (1.0 - h)
    return f, gb.lerp(ga, h)


def _arm_mats():
    """Each side's swing of the arm out from the body, about its shoulder joint."""
    mats = {}
    for side, s in (('R', 1), ('L', -1)):
        piv = Vector((-s * female.SH_X, 0.2, female.SH_Z))
        R = Matrix.Rotation(s * ARM_SPREAD, 4, 'Y')
        mats[side] = Matrix.Translation(piv) @ R @ Matrix.Translation(-piv)
    return mats


def _blend(w, mats):
    """The weighted sum of the bones' matrices for a point's weights (the arms' swing, the rest still)."""
    M = np.zeros((4, 4))
    tot = 0.0
    for b, x in w.items():
        side = b.split('.')[-1]
        if b.split('.')[0] in ('UpperArm', 'Forearm', 'Hand'):
            M += x * np.array(mats[side])
        else:
            M += x * np.eye(4)
        tot += x
    if tot < 1e-6:
        return np.eye(4)
    return M / tot


# How far the thigh's bone carries the skin round the groin: nothing within GROIN[0] of the groin's line (the
# midline, from the crotch up), all of it GROIN[1] away, measured across the body and (times GROIN[2]) down the
# inside of the thigh below the crotch.
GROIN = [0.6, 5.0, 0.8]


def lower_weights(p, lw):
    """The bones' shares of a point of the skin from the waist down (lw: the two legs' weight functions, by side,
    1 her right). The thigh takes the skin from the hip joint down, eased over a long band so the leg's swing
    stretches the skin over the hip instead of folding a step into it. Round the groin the pelvis keeps it: the
    two thighs meet there a hair apart and swing opposite ways, so skin that followed its own thigh right up to the
    midline was pulled out to several times its length across the gap in a long stride or a run. The thigh's share
    grows with the distance from the groin instead, so that pull is spread down the top of the inner thigh. On the
    midline itself what the thighs have is shared between the two by the distance from it."""
    tw = female.trunk_weights(p)
    s = 1 if p.x < 0 else -1
    lg = lw[s](p)
    cz = CANON['crotch']
    band = 0.9 * _ss(cz - 1.6, cz + 0.4, p.z)
    if band > 0.05 and abs(p.x) < band:
        mL = _ss(-band, band, p.x)
        lgR, lgL = lw[1](p), lw[-1](p)
        lg = {k: v * (1.0 - mL) for k, v in lgR.items()}
        for k, v in lgL.items():
            lg[k] = lg.get(k, 0.0) + v * mL
    a = _ss(female.HIP_Z + 1.5, female.HIP_Z - 3.5, p.z)
    a *= _ss(GROIN[0], GROIN[1], math.hypot(p.x, GROIN[2] * max(0.0, cz - p.z)))
    w_ = {k: v * (1 - a) for k, v in tw.items()}
    for k, v in lg.items():
        w_[k] = w_.get(k, 0.0) + v * a
    return w_


def fuse_lower():
    """Join the pelvis, legs and feet to the upper body's skin. Called after the gown's build, before anything else
    reads 'Skin body'."""
    if not ON:
        return
    sb = bpy.data.objects['Skin body']
    me = sb.data
    n0 = len(me.vertices)
    names = [g.name for g in sb.vertex_groups]
    W0 = [dict() for _ in range(n0)]
    for v in me.vertices:
        for ge in v.groups:
            if ge.weight > 1e-5:
                W0[v.index][names[ge.group]] = ge.weight
    mats = _arm_mats()
    # the upper body's skin with the arms swung out (rest data: Skin body is unparented at the origin)
    co0 = np.array([v.co[:] for v in me.vertices])
    sp0 = np.empty_like(co0)
    for i in range(n0):
        M = _blend(W0[i], mats)
        sp0[i] = (M @ np.append(co0[i], 1.0))[:3]
    # which points are the arm's (by their weights) and which the trunk's
    armw = np.array([sum(x for b, x in W0[i].items() if b.split('.')[0] in ('UpperArm', 'Forearm', 'Hand'))
                     for i in range(n0)])
    # the upper back, on the same line as the rest (back_d): the old skin's back is flat from the shoulders to the
    # cut, so its rear half is scaled to the line's depth at each height, from the cut up to the base of the neck
    # (above that it is the approved neck and trapezius, untouched; the arms are left alone)
    zs = np.arange(CUT_Z - 3.0, NECK_Z + 0.26, 0.25)
    mid = []
    for zi in zs:
        m = (np.abs(sp0[:, 2] - zi) < 0.2) & (np.abs(sp0[:, 0]) < 0.6) & (sp0[:, 1] > 0)
        mid.append(sp0[m, 1].max() if m.any() else np.nan)
    mid = np.array(mid)
    ok = ~np.isnan(mid)
    ratio = np.array([back_d(zi) for zi in zs[ok]]) / mid[ok]
    print('anatomy: back line, old skin to new', ' '.join(f'{a:.1f}:{b:.2f}>{back_d(a):.2f}' for a, b in zip(zs[ok][::2], mid[ok][::2])), flush=True)
    r = np.interp(sp0[:, 2], zs[ok], ratio, left=ratio[0], right=1.0)
    r = 1.0 + (r - 1.0) * (1.0 - _ssn(NECK_Z - 0.6, NECK_Z, sp0[:, 2])) * (1.0 - np.clip(armw * 2.0, 0, 1))
    sp0[:, 1] = np.where(sp0[:, 1] > 0, sp0[:, 1] * r, sp0[:, 1])
    up = bmesh.new()
    up.from_mesh(me)
    up.verts.ensure_lookup_table()
    for i, v in enumerate(up.verts):
        v.co = sp0[i]
    full_tree = _tree(_closed(up.copy()))
    # (cut the old trunk below the waist: its hip block goes, the arms stay whole)
    kill = [v for v in up.verts if v.co.z < CUT_Z and armw[v.index] < 0.5]
    print('anatomy: cut', len(kill), 'of', len(up.verts), 'armw hist', np.histogram(armw, bins=5)[0].tolist(), flush=True)
    bmesh.ops.delete(up, geom=kill, context='VERTS')
    up = _closed(up)
    pv = female.loft_fn([48.0 - 0.2 * i for i in range(int((48.0 - CANON['crotch']) / 0.2) + 1)], pelvis_point, n=64)
    pv = _closed(pv)
    legs = [_closed(leg_bm(s)) for s in (1, -1)]
    if MEASURED_FOOT:
        feet = [_closed(_foot.build(s, CANON['foot'], FLOOR_Z, -s * 2.8 * UNIT, 0.0)) for s in (1, -1)]
    else:
        feet = [_closed(foot_bm(s)) for s in (1, -1)]
    # trees for the fillet
    t_body = _tree(_merge([up, pv]))
    t_leg = _tree(_merge(legs))
    t_foot = _tree(_merge(feet))
    # the union, as one volume
    allb = _merge([up, pv] + legs + feet)
    tmp = bpy.data.meshes.new('anat union')
    allb.to_mesh(tmp)
    allb.free()
    ob = bpy.data.objects.new('anat union', tmp)
    bpy.context.scene.collection.objects.link(ob)
    rm = ob.modifiers.new('remesh', 'REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = VOXEL
    rm.adaptivity = 0.0
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    em = ob.evaluated_get(dg).to_mesh()
    out = bmesh.new()
    out.from_mesh(em)
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(tmp)
    out.verts.ensure_lookup_table()
    n = len(out.verts)
    co = np.array([v.co[:] for v in out.verts])
    ed = np.array([[e.verts[0].index, e.verts[1].index] for e in out.edges])
    deg = np.maximum(np.bincount(ed.ravel(), minlength=n), 1).astype(float)[:, None]

    def relax(c, w, iters):
        for _ in range(iters):
            acc = np.zeros_like(c)
            np.add.at(acc, ed[:, 0], c[ed[:, 1]])
            np.add.at(acc, ed[:, 1], c[ed[:, 0]])
            c = c + w[:, None] * 0.5 * (acc / deg - c)
        return c
    co = relax(co, np.full(n, 0.35), 3)
    # the fillets: the zero set of smin(body, smin(leg, foot, k_ankle), k_hip), for points near two of the parts
    dB = np.array([t_body.find_nearest(Vector(p))[3] for p in co])
    dL = np.array([t_leg.find_nearest(Vector(p))[3] for p in co])
    dF = np.array([t_foot.find_nearest(Vector(p))[3] for p in co])
    z = co[:, 2]
    ax = np.abs(co[:, 0])
    kh = K_HIP * _ssn(33.0, 30.5, z) * _ssn(CANON['crotch'] - 4.7, CANON['crotch'] - 1.7, z) * np.maximum(1.0 - _ssn(2.2, 4.2, ax), _ssn(1.0, 2.5, co[:, 1]) * (1.0 - _ssn(5.3, 6.1, ax)))
    # (only between the legs and over the groin: at the hip the thigh and the pelvis share one outline, and a smooth
    # union of two surfaces lying on each other swells out by a quarter of its width, a saddlebag round the hip)
    ka = K_ANKLE * _ssn(FLOOR_Z + 7.5, FLOOR_Z + 5.5, z)
    near_h = (kh > 1e-3) & (np.abs(dB - dL) < kh) & (np.minimum(dB, dL) < 0.3)
    near_a = (ka > 1e-3) & (np.abs(dL - dF) < ka) & (np.minimum(dL, dF) < 0.3)
    cand = np.nonzero(near_h | near_a)[0]

    def project(c, idx, steps=6):
        for i in idx:
            p = Vector(c[i])
            for _ in range(steps):
                b, gb = _sdf(t_body, p)
                l_, gl = _sdf(t_leg, p)
                f_, gf = _sdf(t_foot, p)
                lf, glf = _smin(l_, gl, f_, gf, ka[i])
                f, g = _smin(b, gb, lf, glf, kh[i])
                if g.length < 1e-6:
                    break
                g.normalize()
                p -= g * f
                if abs(f) < 1e-3:
                    break
            c[i] = p
        return c
    co = project(co, cand)
    w = np.zeros(n)
    w[cand] = 1.0
    for _ in range(3):
        w2 = relax(w[:, None].repeat(3, 1), np.full(n, 1.0), 2)[:, 0]
        co = relax(co, 0.8 * np.clip(w2 * 1.5, 0, 1), 4)
        co = project(co, cand, steps=4)
    # (the feet: the voxels and the fillet's projection onto the faceted parts leave ripples on the gently sloping
    # instep, seen in its curvature; a shrink-free (Taubin) smoothing over the feet takes them out without thinning
    # the toes)
    if MEASURED_FOOT:
        wf = _ssn(FLOOR_Z + 5.0, FLOOR_Z + 3.5, co[:, 2])
        for _ in range(FOOT_SMOOTH):
            co = relax(co, 1.0 * wf, 1)
            co = relax(co, -1.06 * wf, 1)
        # (the front of the ankle: where the shin comes down onto the instep the union leaves a crease; the skin
        # there is relaxed as one patch, like the seat, so the shin runs into the instep as one concave line)
        az, ay, ax_ = co[:, 2] - FLOOR_Z, co[:, 1], np.abs(np.abs(co[:, 0]) - 2.8 * UNIT)
        wa = _ssn(2.2, 2.9, az) * (1.0 - _ssn(4.6, 5.4, az)) * _ssn(-0.3, -0.9, ay) * (1.0 - _ssn(-3.6, -4.3, ay)) \
            * (1.0 - _ssn(1.1, 1.7, ax_))
        co = relax(co, 0.9 * wa, ANKLE_FRONT_ITERS)
    # the seat smoothed as one surface: where the thigh's back comes out of the pelvis, the cleft and the fold
    # under the seat are eased by relaxing the skin over the whole back of the hips (a soft-edged patch)
    seat = _ssn(0.3, 1.8, co[:, 1]) * _ssn(CANON['crotch'] - 4.0, CANON['crotch'] - 1.0, z) * (1.0 - _ssn(34.0, 37.0, z))
    co = relax(co, float(os.environ.get('SEAT_SMOOTH', '0.9')) * seat, int(os.environ.get('SEAT_ITERS', '60')))
    # (the voxels' steps along the old waist cut, eased)
    seam = np.exp(-((co[:, 2] - CUT_Z) / 0.8) ** 2) * (1.0 - _ssn(4.6, 5.4, np.abs(co[:, 0])))
    co = relax(co, 0.6 * seam, 6)
    # above the waist the skin goes back exactly onto the upper body's (the approved neck, shoulders and arms)
    keep = _ssn(KEEP_Z - 1.0, KEEP_Z + 1.8, co[:, 2])       # (eased in over a long band: a short one left a step)
    for i in np.nonzero(keep > 0)[0]:
        loc = full_tree.find_nearest(Vector(co[i]))[0]
        if loc is not None:
            co[i] = co[i] + keep[i] * (np.array(loc[:]) - co[i])
    band_ = np.exp(-((co[:, 2] - (KEEP_Z + 0.2)) / 1.2) ** 2) * (1.0 - _ssn(4.4, 5.2, np.abs(co[:, 0])))
    co = relax(co, 0.7 * band_, 10)
    # the weights: from the nearest point of the old skin above the hips, the legs' own below, blended over the hip
    kd = KDTree(n0)
    for i in range(n0):
        kd.insert(Vector(sp0[i]), i)
    kd.balance()
    lw = {1: female.leg_weights('R'), -1: female.leg_weights('L')}
    Wn = []
    src = np.empty(n, dtype=np.int64)
    rest = np.empty_like(co)
    for i in range(n):
        p = Vector(co[i])
        _, j, dist = kd.find(p)
        src[i] = j
        db_, dl_ = t_body.find_nearest(p)[3], t_leg.find_nearest(p)[3]
        df_ = t_foot.find_nearest(p)[3]
        dl_ = min(dl_, df_)
        if (p.z >= CUT_Z or armw[j] > 0.5) and (dist < 0.35 or p.z > KEEP_Z + 1.0):
            w_ = dict(W0[j])
        else:
            w_ = lower_weights(p, lw)
        Wn.append(w_)
    # (under the arm the new trunk meets the old arm: the weights there eased across neighbours, so the skin of the
    # armpit is shared between the chest and the arm instead of tearing along the join)
    bn = sorted({k for w_ in Wn for k in w_})
    bi = {k: j for j, k in enumerate(bn)}
    WM = np.zeros((n, len(bn)))
    for i, w_ in enumerate(Wn):
        for k, v in w_.items():
            WM[i, bi[k]] = v
    reg = _ssn(42.5, 44.0, co[:, 2]) * (1.0 - _ssn(47.0, 48.0, co[:, 2])) * _ssn(3.2, 4.2, np.abs(co[:, 0]))
    ri = np.nonzero(reg > 0)[0]
    for _ in range(30):
        acc = np.zeros_like(WM)
        np.add.at(acc, ed[:, 0], WM[ed[:, 1]])
        np.add.at(acc, ed[:, 1], WM[ed[:, 0]])
        WM[ri] += (acc[ri] / deg[ri] - WM[ri]) * (0.7 * reg[ri])[:, None]
    WM /= np.maximum(WM.sum(1), 1e-9)[:, None]
    Wn = [{bn[j]: float(WM[i, j]) for j in np.nonzero(WM[i] > 1e-4)[0]} for i in range(n)]
    for i in range(n):
        M = _blend(Wn[i], mats)
        rest[i] = (np.linalg.inv(M) @ np.append(co[i], 1.0))[:3]
    bad = np.nonzero((rest[:, 2] < 12) & (np.abs(rest[:, 0]) > 4.8))[0]
    print('anatomy: low outer points', len(bad), rest[bad[:5]].round(2).tolist() if len(bad) else '', 'their spread co', co[bad[:5]].round(2).tolist() if len(bad) else '', [Wn[i] for i in bad[:3]], flush=True)
    rest = round_shoulders(rest)
    # the joints eased as one skin (John, 2026-11-18): the point of the shoulder, where the trapezius's ridge meets
    # the deltoid; the root of the neck, where its column meets the slope; and the fold of the armpit, front and
    # back, where the chest's skin meets the arm's. Each is a soft-edged patch relaxed over its neighbours.
    az, rz = np.abs(rest[:, 0]), rest[:, 2]
    w_sh = _ssn(4.6, 5.6, az) * _ssn(47.6, 48.6, rz)
    w_nk = _ssn(48.6, 49.4, rz) * (1.0 - _ssn(52.2, 53.0, rz)) * (1.0 - 0.6 * _ssn(4.6, 5.6, az))
    w_pit = np.exp(-((rz - 46.6) / 1.3) ** 2) * _ssn(2.2, 3.4, az) * (1.0 - _ssn(6.6, 7.4, az))
    k = float(os.environ.get('JOINT_SMOOTH', '1.0'))
    rest = relax(rest, np.clip(k * (0.9 * w_sh + 0.6 * w_nk), 0, 1), 14)
    rest = relax(rest, np.clip(k * 0.9 * w_pit, 0, 1), 18)
    # a second hand on each wrist, closed (for the run: use_fist() shows one or the other), with the open hand's
    # weights and its colour layers
    for side, s_ in (('R', 1), ('L', -1)):
        h = bpy.data.objects.get(f'Hand.{side}')
        if h is None or bpy.data.objects.get(f'Fist.{side}') is not None:
            continue
        fo = female.mesh_obj(f'Fist.{side}', female.hand_mesh(s_, fist=True), female.M_SKIN, loc=tuple(h.location),
                             subdiv=0, skin=female.arm_weights(side))
        for a_ in h.data.color_attributes:
            arr_ = np.zeros(len(h.data.vertices) * 4)
            a_.data.foreach_get('color', arr_)
            b_ = fo.data.color_attributes.new(a_.name, a_.data_type, a_.domain)
            b_.data.foreach_set('color', np.tile(arr_.reshape(-1, 4).mean(0), len(fo.data.vertices)))
        fo.hide_render = fo.hide_viewport = True
    for side in ('L', 'R'):       # (the hand as long as the face; the fist on the open hand's scale)
        o = bpy.data.objects.get(f'Hand.{side}')
        if o is not None:
            zs_ = [v.co.z for v in o.data.vertices]
            k = CANON['hand'] / (max(zs_) - min(zs_))
            for q in (o, bpy.data.objects.get(f'Fist.{side}')):
                if q is None:
                    continue
                top = max(v.co.z for v in q.data.vertices)
                for v in q.data.vertices:
                    v.co = Vector((v.co.x * k, v.co.y * k, top + (v.co.z - top) * k))
                q.data.update()
    # the wrist: the forearm ends in the wrist's oval (ARM_SEC), and the root of the hand is drawn out to the same
    # oval. The hand's own root was a narrow dome that flared to the palm inside one unit; now its section runs
    # from the wrist's oval at the joint to the palm's own a fifth of the hand lower, centred on the arm at the
    # wrist. The two skins cross just under the joint, the forearm's end tucked inside the heel of the hand.
    bpy.context.view_layer.update()
    zw = female.SH_Z - U - F
    fore = np.array([sum(v for b_, v in w_.items() if b_.split('.')[0] in ('Forearm', 'Hand')) for w_ in Wn]) > 0.5
    for nm, sgn in (('Hand.R', -1.0), ('Hand.L', 1.0), ('Fist.R', -1.0), ('Fist.L', 1.0)):
        o = bpy.data.objects.get(nm)
        if o is None:
            continue
        mw_ = Matrix.Translation(o.location)      # (a skinned mesh, unparented: and a hidden one's matrix_world is stale)
        H_ = np.array([(mw_ @ v.co)[:] for v in o.data.vertices])
        axx, axy = sgn * female.SH_X, 0.2
        hx, hy, dz_ = H_[:, 0] - axx, H_[:, 1] - axy, H_[:, 2] - zw
        # (the hand's section as built, by height: its middle and half sizes across and along the palm)
        zb = np.arange(-2.6, dz_.max() + 0.11, 0.1)
        ext = np.full((len(zb), 4), np.nan)
        for q, z_ in enumerate(zb):
            mm_ = np.abs(dz_ - z_) < 0.1
            if mm_.sum() > 3:
                ext[q] = (hx[mm_].min(), hx[mm_].max(), hy[mm_].min(), hy[mm_].max())
        okq = ~np.isnan(ext[:, 0])
        ext = np.stack([np.interp(zb, zb[okq], ext[okq, c_]) for c_ in range(4)], 1)
        for _ in range(2):
            ext[1:-1] = (ext[:-2] + 2 * ext[1:-1] + ext[2:]) / 4
        cxc, hxc = np.interp(dz_, zb, 0.5 * (ext[:, 0] + ext[:, 1])), np.maximum(np.interp(dz_, zb, 0.5 * (ext[:, 1] - ext[:, 0])), 0.08)
        cyc, hyc = np.interp(dz_, zb, 0.5 * (ext[:, 2] + ext[:, 3])), np.maximum(np.interp(dz_, zb, 0.5 * (ext[:, 3] - ext[:, 2])), 0.08)
        qp = int(np.argmin(np.abs(zb + 2.0)))
        cxp, hxp = 0.5 * (ext[qp, 0] + ext[qp, 1]), 0.5 * (ext[qp, 1] - ext[qp, 0])
        cyp, hyp = 0.5 * (ext[qp, 2] + ext[qp, 3]), 0.5 * (ext[qp, 3] - ext[qp, 2])
        u = _ssn(-2.0, -0.3, dz_)
        inner = 1.0 - 0.1 * _ssn(-0.25, 0.3, dz_)          # (a little inside the forearm above the joint)
        tx = cxp * (1 - u) + (hx - cxc) / hxc * (hxp + u * (WRIST_T - hxp)) * inner
        ty = cyp * (1 - u) + (hy - cyc) / hyc * (hyp + u * (WRIST_W - hyp)) * inner
        nx_, ny_ = hx + u * (tx - hx), hy + u * (ty - hy)
        print(f'anatomy: wrist {nm} hand root at the joint was {2 * np.interp(0.0, zb, 0.5 * (ext[:, 1] - ext[:, 0])):.2f} thick, '
              f'{2 * np.interp(0.0, zb, 0.5 * (ext[:, 3] - ext[:, 2])):.2f} wide; now {2 * WRIST_T:.2f} by {2 * WRIST_W:.2f}; '
              f'palm {2 * hxp:.2f} by {2 * hyp:.2f}', flush=True)
        inv_ = mw_.inverted()
        for q, v in enumerate(o.data.vertices):
            v.co = inv_ @ Vector((axx + nx_[q], axy + ny_[q], H_[q, 2]))
        o.data.update()
        if nm.startswith('Fist'):
            continue
        # (the forearm's end, drawn in under the joint so its edge lies inside the heel of the hand)
        px, py = rest[:, 0] - axx, rest[:, 1] - axy
        ii = np.nonzero(fore & (np.hypot(px, py) < 2.4) & (rest[:, 2] < zw + 0.2) & (rest[:, 2] > zw - 1.0))[0]
        k_ = 1.0 - 0.14 * (1.0 - _ssn(zw - 0.38, zw - 0.1, rest[ii, 2]))
        rest[ii, 0] = axx + px[ii] * k_
        rest[ii, 1] = axy + py[ii] * k_
    L = np.linalg.norm(rest[ed[:, 0]] - rest[ed[:, 1]], axis=1)
    lg_ = np.nonzero(L > 0.8)[0]
    print('anatomy: long edges', len(lg_), [(rest[ed[e, 0]].round(1).tolist(), rest[ed[e, 1]].round(1).tolist(), Wn[ed[e, 0]], Wn[ed[e, 1]]) for e in lg_[:4]], flush=True)
    for i, v in enumerate(out.verts):
        v.co = rest[i]
    bmesh.ops.recalc_face_normals(out, faces=out.faces[:])
    # the old skin's baked colour layers, from the nearest old point
    layers = [(a.name, a.data_type) for a in me.color_attributes if a.domain == 'POINT']
    cols = {}
    for nm, _ in layers:
        a = me.color_attributes[nm]
        arr = np.zeros(n0 * 4)
        a.data.foreach_get('color', arr)
        cols[nm] = arr.reshape(-1, 4)
    mats_ = list(me.materials)
    mods = [(m.name, m.type) for m in sb.modifiers]
    bpy.data.objects.remove(sb)
    for nm in ('Leg.L', 'Leg.R', 'Shoe.L', 'Shoe.R'):
        o = bpy.data.objects.get(nm)
        if o is not None:
            bpy.data.objects.remove(o)
    idx = {}

    def weights(p, _c=[0]):
        i = _c[0]
        _c[0] += 1
        return Wn[i]
    nb = female.mesh_obj('Skin body', out, female.M_SKIN, subdiv=0, skin=weights)
    nb.data.materials.clear()
    for m in mats_:
        nb.data.materials.append(m)
    for nm, dt in layers:
        a = nb.data.color_attributes.new(nm, dt, 'POINT')
        a.data.foreach_set('color', cols[nm][src].ravel())
    R_ = rest
    prof = []
    for zz in np.arange(45.0, 52.01, 0.5):
        m = np.abs(R_[:, 2] - zz) < 0.15
        prof.append((round(float(zz), 1), round(float(np.abs(R_[m, 0]).max()), 2) if m.any() else None))
    tops = []
    for xx in np.arange(4.0, 8.61, 0.4):
        m = np.abs(np.abs(R_[:, 0]) - xx) < 0.12
        tops.append((round(float(xx), 1), round(float(R_[m, 2][R_[m, 2] > 44].max()), 2) if (m & (R_[:, 2] > 44)).any() else None))
    fr = []
    for zz in np.arange(44.0, 49.6, 0.25):
        m = (np.abs(R_[:, 2] - zz) < 0.1) & (np.abs(R_[:, 0] - 1.0) < 0.15)
        fr.append((round(float(zz), 2), round(float(R_[m, 1].min()), 2), round(float(R_[m, 1].max()), 2)) if m.any() else None)
    print('anatomy: front/back y at x=1 by z', fr, flush=True)
    print('anatomy: half width by z', prof, 'top z by x', tops, flush=True)
    print(f'anatomy: skin {n0} -> {n} points, {len(cand)} on the fillets, modifiers before {mods}', flush=True)
    return nb


SH_SLOPE = math.radians(float(os.environ.get('SH_SLOPE', '21')))
SH_TAN = (float(os.environ.get('SH_TAN_X', '6.0')), float(os.environ.get('SH_TAN_Z', '49.54')))
SH_OUT = float(os.environ.get('SH_OUT', str(CANON['shoulders'] / 2 + 0.1)))


def use_fist(on=True, side=None):
    """Show the closed fists in place of the open hands (or the open hands again), on one side ('L' or 'R') or
    both. The two are separate meshes on the same wrist; a pose picks one."""
    for sd in (side,) if side else ('L', 'R'):
        for nm, show in ((f'Hand.{sd}', not on), (f'Fist.{sd}', on)):
            o = bpy.data.objects.get(nm)
            if o is not None:
                o.hide_render = o.hide_viewport = not show


def round_shoulders(co, b=0.35):
    """The corner where the trapezius's slope met the side of the deltoid, rounded: seen from the front the
    shoulder's outline follows one arc, tangent to the slope where the trapezius ends (SH_TAN, at SH_SLOPE) and
    to the side of the arm at its widest (SH_OUT). Points outside that arc (a cylinder along the depth) are drawn
    onto it with a soft clamp; the slope and the neck inside it are left as they are."""
    nx, nz = -math.sin(SH_SLOPE), -math.cos(SH_SLOPE)
    R = (SH_OUT - SH_TAN[0]) / (1.0 + nx)
    ox, oz = SH_TAN[0] + nx * R, SH_TAN[1] + nz * R
    co = co.copy()
    ax = np.abs(co[:, 0])
    dx, dz = ax - ox, co[:, 2] - oz
    d = np.hypot(dx, dz)
    m = (dx > 0) & (dz > 0) & (d > R - b)
    t = d[m] - (R - b)
    d2 = (R - b) + b * np.tanh(t / b)
    # (only past the tangent point, eased in over a few degrees, so the trapezius's slope is left alone)
    ang = np.arcsin(np.clip(dx[m] / d[m], -1, 1))
    a = _ssn(SH_SLOPE - 0.02, SH_SLOPE + 0.2, ang)
    d2 = d[m] + a * (d2 - d[m])
    k = d2 / d[m]
    co[m, 0] = np.sign(co[m, 0]) * (ox + dx[m] * k)
    co[m, 2] = oz + dz[m] * k
    print(f'anatomy: shoulders rounded, arc r {R:.2f} about ({ox:.2f}, {oz:.2f}), {int(m.sum())} points', flush=True)
    return co


def _merge(bms):
    out = bmesh.new()
    for b in bms:
        tmp = bpy.data.meshes.new('part')
        b.to_mesh(tmp)
        out.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    return out


def head_stub(head):
    """The plain mannequin head's sculpt runs on below the jaw as a block (inside the collar on a dressed figure);
    bare, it stood out behind the neck. Below the jaw line its points are drawn in to inside the neck."""
    if not ON:
        return
    for o in head.children:
        if o.type != 'MESH' or not o.name.startswith('Head shape'):
            continue
        for v in o.data.vertices:
            c = v.co
            if c.z < 0.9 and c.y > -0.5:
                k = _ss(0.9, -0.3, c.z) * _ss(-0.5, 1.0, c.y)
                c.y -= k * max(0.0, c.y - 0.2) * 0.85
                c.x *= 1.0 - 0.35 * k
        o.data.update()


def measure():
    """The head's own sizes and the body's, in the head's units, for checking the body against the head's canon."""
    import collections
    hp = []
    for o in bpy.data.objects:
        if o.type == 'MESH' and (o.name.startswith('Head shape') or o.name.startswith('head part 0')):
            hp += [o.matrix_world @ v.co for v in o.data.vertices]
    sb = bpy.data.objects['Skin body']
    B = np.array([v.co[:] for v in sb.data.vertices])
    out = {}
    if hp:
        H = np.array([p[:] for p in hp])
        chin = min(p[2] for p in H if abs(p[0]) < 0.5 and p[1] < -1.9)
        top = H[:, 2].max()
        hh = top - chin
        out.update(chin=chin, top=top, head_h=hh, head_w=2 * np.abs(H[:, 0]).max(), head_d=H[:, 1].max() - H[:, 1].min())
        for nm, f in (('cheek', 0.42), ('jaw', 0.17), ('temple', 0.72)):
            m = np.abs(H[:, 2] - (chin + f * hh)) < 0.12
            out['w_' + nm] = 2 * np.abs(H[m, 0]).max()
    def w(z, xmax=99, dz=0.15):
        m = (np.abs(B[:, 2] - z) < dz) & (np.abs(B[:, 0]) < xmax)
        return 2 * np.abs(B[m, 0]).max() if m.any() else 0
    def d(z, xmax=1.0):
        m = (np.abs(B[:, 2] - z) < 0.15) & (np.abs(B[:, 0]) < xmax)
        return B[m, 1].max() - B[m, 1].min() if m.any() else 0
    out['floor'] = B[:, 2].min()
    out['shoulders'] = max(w(z) for z in np.arange(45, 49.5, 0.25))
    for z in (52.0, 51.0, 48.0, 46.0, 44.3, 42.0, 39.4, 37.0, 34.0, 32.0, 30.5, 29.0):
        out[f'trunk w{z}'] = w(z, 6.6)
        out[f'trunk d{z}'] = d(z, 2.5)
    for z in (28, 24, 20, 15, 11.2, 6, 2, -2, -4.2):
        m = (np.abs(B[:, 2] - z) < 0.15) & (B[:, 0] < 0)
        out[f'leg w{z}'] = B[m, 0].max() - B[m, 0].min()
        out[f'leg d{z}'] = B[m, 1].max() - B[m, 1].min()
        out[f'legs across{z}'] = w(z)
    m = (B[:, 2] < FLOOR_Z + 2.0)
    out['foot_len'] = B[m, 1].max() - B[m, 1].min()
    m = (B[:, 2] < FLOOR_Z + 1.0) & (B[:, 0] < 0)
    out['foot_w'] = B[m, 0].max() - B[m, 0].min()
    for zl in (-2, -5, -8, -9.8, -12, -15, -19):
        m = (np.abs(B[:, 2] - (female.SH_Z + zl)) < 0.15) & (B[:, 0] < -6.0 if zl > -3 else B[:, 0] < -4.2) & (np.abs(B[:, 0] + female.SH_X) < 2.6)
        if m.any():
            out[f'arm w{zl}'] = B[m, 0].max() - B[m, 0].min()
    hd = bpy.data.objects.get('Hand.R')
    if hd is not None:
        P = np.array([(hd.matrix_world @ v.co)[:] for v in hd.data.vertices])
        out['hand_len'] = P[:, 2].max() - P[:, 2].min()
        out['hand_bottom_z'] = P[:, 2].min()
    out['neck w'] = min(w(z, 4) for z in np.arange(51.0, 53.0, 0.25))
    out['crotch_z'] = max(B[(np.abs(B[:, 0]) < 0.15) & (B[:, 2] < 32), 2].min(), 0) if True else 0
    m = (np.abs(B[:, 0]) < 0.2) & (B[:, 2] < 34) & (B[:, 2] > 20)
    out['crotch_z'] = B[m, 2].min()
    print('MEASURE ' + ', '.join(f'{k} {v:.2f}' for k, v in out.items()), flush=True)
    dg = bpy.context.evaluated_depsgraph_get()
    cz = CANON['crotch']
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        ev = o.evaluated_get(dg)
        em = ev.to_mesh()
        P = np.array([(o.matrix_world @ v.co)[:] for v in em.vertices])
        if len(P) == 0:
            continue
        box = (np.abs(P[:, 0]) < 3.5) & (np.abs(P[:, 2] - cz) < 2.5) & (P[:, 1] < 1.0)
        print(f'CROTCH {o.name} mat {o.active_material.name if o.active_material else None} points in box {int(box.sum())} hide {o.hide_render}', flush=True)
        if o.name == 'Skin body':
            R = np.array([v.co[:] for v in o.data.vertices])
            E = np.array([e.vertices[:] for e in em.edges])
            if len(R) == len(P):
                lp = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
                lr = np.linalg.norm(R[E[:, 0]] - R[E[:, 1]], axis=1)
                st = lp / np.maximum(lr, 1e-6)
                eb = box[E[:, 0]]
                bad = np.nonzero(eb & ((st > 2.0) | (st < 0.4)))[0]
                print(f'CROTCH stretched edges {len(bad)} of {int(eb.sum())}; max {st[eb].max():.2f} min {st[eb].min():.2f}', flush=True)
                for e in bad[:: max(1, len(bad) // 12)][:12]:
                    i0 = E[e, 0]
                    g = {o.vertex_groups[x.group].name: round(x.weight, 2) for x in o.data.vertices[i0].groups if x.weight > 0.02}
                    print('   posed', P[i0].round(2).tolist(), 'rest', R[i0].round(2).tolist(), f'stretch {st[e]:.2f}', g, flush=True)
                # (folded faces: posed normal against rest normal)
                flips = 0
                samp = []
                for f in em.polygons:
                    vs = f.vertices[:]
                    if not box[vs[0]]:
                        continue
                    nr = np.cross(R[vs[1]] - R[vs[0]], R[vs[2]] - R[vs[0]])
                    npz = np.cross(P[vs[1]] - P[vs[0]], P[vs[2]] - P[vs[0]])
                    if np.dot(nr, npz) < 0:
                        flips += 1
                        if len(samp) < 8:
                            samp.append((P[vs[0]].round(2).tolist(), R[vs[0]].round(2).tolist()))
                print('CROTCH flipped faces', flips, samp, flush=True)
        ev.to_mesh_clear()
    for z in np.arange(50.0, 43.4, -0.25):
        m = np.abs(B[:, 2] - z) < 0.13
        r = []
        for x0 in (2.0, 3.5, 4.3):
            mm = m & (np.abs(B[:, 0] + x0) < 0.2)
            r.append(f'{B[mm, 1].min():.2f}/{B[mm, 1].max():.2f}' if mm.any() else '-')
        mm = m & (np.abs(B[:, 1]) < 0.3) & (B[:, 0] < 0) & (B[:, 0] > -5.6)
        print(f'JOINT z {z:.2f} front/back at x2 {r[0]} x3.5 {r[1]} x4.3 {r[2]} side {B[mm, 0].min() if mm.any() else 0:.2f}', flush=True)
    for z in np.arange(54.0, 26.0, -0.5):
        m = np.abs(B[:, 2] - z) < 0.15
        mm = m & (np.abs(B[:, 0]) < 0.5)
        mt = m & (np.abs(B[:, 0]) < 5.0)
        if mm.any():
            print(f'BACKP z {z:.1f} mid {B[mm, 1].max():.2f} trunk {B[mt, 1].max():.2f} front {B[mt, 1].min():.2f}', flush=True)


def configure(head_h, chin_z):
    """Set the unit: the height of the head the figure wears (chin to crown, as it sits on the body) and where its
    chin is. Every table in this module is worked out again from them."""
    import importlib
    import sys
    SEED.update(H=float(head_h), chin=float(chin_z))
    mod = importlib.reload(sys.modules[__name__])
    mod.patch()
    return mod


def head_height(scale=1.0):
    """The mannequin head's height as built, chin to crown (the chin: the lowest point at the front near the
    middle, as the figure's seat_head finds it), times the scale the figure gives it. None if there is no head."""
    pts = []
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name.startswith('Head shape'):
            pts += [o.matrix_world @ v.co for v in o.data.vertices]
    if not pts:
        return None
    chin = min(p.z for p in pts if abs(p.x) < 0.5 and p.y < -1.9)
    return (max(p.z for p in pts) - chin) * scale
