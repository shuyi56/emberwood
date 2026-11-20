"""The arming doublet's sleeve, begun where a tailor begins: the tape, then the flat draft.

  1. the measures   taken on the body's rest shape and on the doublet as it is cut:
                      the armhole     the doublet's own armhole edge: its length, front and back of the shoulder
                                      point, and its depth (shoulder point to underarm point)
                      the arm         its girth at the bicep, the elbow, the widest of the forearm and the wrist (a
                                      tape drawn taut round a level section), and the hand's, which has to pass
                      the lengths     shoulder point to elbow and to the cuff over the outside of the arm, and
                                      underarm point to the cuff down the inside
  2. the draft      a one-piece set-in sleeve, its seam under the arm:
                      the cap's height   the over-arm length less the underarm length: what the cloth has to
                                         climb from the underarm to the shoulder point with the arm hanging
                      the cap's curve    front and back each as long as its share of the armhole (plus the cap's
                                         ease): the bicep line's width is whatever makes them so, and is checked
                                         against the arm's girth and the room the shirt and the padding need
                      the taper          the elbow and the cuff, each the arm's girth there plus its room; where
                                         the hand is wider than the cuff the seam is left open and laced

    python3 blender/doublet/sleeve_draft.py FIGURE.blend --out SHEET.png

FIGURE.blend: the doublet figure as saved by base/views.py (--save). Sizes are in units; the sheet gives centimetres
too (165 cm to 63 units, as the spaulders' cutting sheet does)."""
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vest'))
import fit  # noqa: E402

CM = 165.0 / 63.0
RATIOS = dict(
    room=0.30,               # the room between arm and sleeve at the bicep and the elbow, in w: the body's own ease
    room_cuff=0.16,          # and at the cuff
    cuff=0.25,               # the cuff ends this far above the wrist, in w
    cap_ease=0.015,          # the cap's curve is this much longer than the armhole (eased in over the shoulder)
    front_scoop=0.45,        # how far in from the underarm the front of the cap stays low, as a share of its width
    back_scoop=0.30,         # and the back (the back of the arm is fuller: the curve rises sooner)
    crown=0.45,              # how far out from the shoulder point the cap stays high, as a share of each side's width
)


def _perimeter(pts):
    h = np.array(fit._hull(pts))
    return float(np.linalg.norm(np.roll(h, -1, 0) - h, axis=1).sum())


def _chain(pts):
    """The length of a tape drawn taut over the outside (+x) of points [(x, z)] from the first to the last."""
    pts = np.asarray(pts)
    keep = [0]
    i = 0
    while i < len(pts) - 1:
        # (the next point the tape touches: the one it turns least toward the arm to reach)
        d = pts[i + 1:] - pts[i]
        ang = np.arctan2(d[:, 0], -d[:, 1])
        i = i + 1 + int(np.argmax(ang))
        keep.append(i)
    q = pts[keep]
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def armhole(co, polygons, side=1.0):
    """The garment's armhole on one side: its edge as a closed line of points, from its open edges."""
    cnt = defaultdict(int)
    for vs in polygons:
        for a, b in zip(vs, vs[1:] + vs[:1]):
            cnt[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in cnt.items():
        if c == 1:
            adj[a].append(b)
            adj[b].append(a)
    seen, best = set(), None
    for s in adj:
        if s in seen:
            continue
        loop, cur = [s], s
        seen.add(s)
        while True:
            nx = [n for n in adj[cur] if n not in seen]
            if not nx:
                break
            cur = nx[0]
            seen.add(cur)
            loop.append(cur)
        c = co[loop]
        cen = c.mean(0)
        # (the armhole: the open edge that lies off to the side and wholly above the waist)
        if side * cen[0] > 0.25 * np.abs(co[:, 0]).max() and (best is None or len(loop) > len(best)) and np.ptp(c[:, 0]) < 0.5 * np.ptp(co[:, 0]):
            best = loop
    return co[best]


def measure(P, T, aw, hole, joints, hand=None, w=1.0, ratios=RATIOS):
    """The tape's measures. hole: the armhole's edge [n, 3]; joints: the heights of (shoulder, elbow, wrist) and the
    arm's axis (x, y); hand: the hand's points, if it is a piece of its own."""
    (z_s, z_e, z_w), (ax, ay) = joints
    m = {}
    seg = np.linalg.norm(np.roll(hole, -1, 0) - hole, axis=1)
    i_top, i_bot = int(hole[:, 2].argmax()), int(hole[:, 2].argmin())
    a, b = sorted((i_top, i_bot))
    one, L = float(seg[a:b].sum()), float(seg.sum())
    front_first = hole[a:b][:, 1].mean() < hole[:, 1].mean()          # (the front is toward -Y)
    m['armhole'], m['armhole_front'], m['armhole_back'] = L, (one if front_first else L - one), (L - one if front_first else one)
    m['top'], m['bottom'] = hole[i_top], hole[i_bot]
    m['depth'] = float(hole[i_top, 2] - hole[i_bot, 2])
    # the arm's girths: a level section of the arm's skin, the tape taut round it
    tri = P[T[~(aw[T] < 0.5).all(1)]]

    def section(z):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            A, B = tri[:, i], tri[:, j]
            k = (A[:, 2] - z) * (B[:, 2] - z) < 0
            t = (z - A[k, 2]) / (B[k, 2] - A[k, 2])
            pts.append(A[k] + t[:, None] * (B[k] - A[k]))
        q = np.concatenate(pts)
        return q[(q[:, 0] > 0) & (np.hypot(q[:, 0] - ax, q[:, 1] - ay) < 2.2 * w)]
    z_cuff = z_w + ratios['cuff'] * w
    zs = np.arange(z_cuff, m['bottom'][2], 0.2)
    girth, outer, inner = [], [], []
    for z in zs:
        q = section(float(z))
        girth.append(_perimeter(q[:, :2]))
        outer.append(float(q[:, 0].max()))
        inner.append(float(q[:, 0].min()))
    girth = np.array(girth)
    m['zs'], m['girths'] = zs, girth
    up = zs > z_e + 0.25 * (z_s - z_e)
    m['bicep'] = float(girth[up].max())
    m['elbow'] = float(girth[np.argmin(np.abs(zs - z_e))])
    fore = zs < z_e
    m['forearm'] = float(girth[fore].max())
    m['wrist'] = float(girth[0])
    m['z_elbow'], m['z_cuff'] = z_e, z_cuff
    if hand is not None:
        hz = np.arange(hand[:, 2].min() + 0.1, hand[:, 2].max(), 0.1)
        m['hand'] = max(_perimeter(hand[np.abs(hand[:, 2] - z) < 0.12][:, :2]) for z in hz if (np.abs(hand[:, 2] - z) < 0.12).sum() > 8)
    # the lengths, with the sleeve's room on the arm: over the outside from the shoulder point, down the inside
    # from the underarm point
    room = ratios['room'] * w
    over = [(float(m['top'][0]), float(m['top'][2]))]
    zz = np.arange(m['top'][2] - 0.2, z_cuff, -0.2)
    for z in zz:
        q = section(float(z))
        if len(q) > 8:
            over.append((float(q[:, 0].max()) + room, float(z)))
    over.append((over[-1][0], z_cuff))
    m['over_arm'] = _chain(over)
    k = int(np.argmin([abs(p[1] - z_e) for p in over]))
    m['to_elbow'] = _chain(over[:k + 1])
    m['under_arm'] = float(np.hypot(m['bottom'][0] - (inner[0] - ratios['room_cuff'] * w), m['bottom'][2] - z_cuff))
    return m


def _bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _half(width, h, scoop, crown):
    """One side of the cap, from the shoulder point (0, 0) out and down to the underarm (width, h): level at both
    ends, an S between."""
    return _bezier(np.array([0.0, 0.0]), np.array([crown * width, 0.0]), np.array([(1.0 - scoop) * width, h]), np.array([width, h]))


def _width_for(length, h, scoop, crown):
    lo, hi = 0.0, 3.0 * length
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        c = _half(mid, h, scoop, crown)
        if float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) < length:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def draft(m, w, ratios=RATIOS):
    """The flat sleeve from the measures: dict(outline [n, 2] (x across, front to +x; y down from the shoulder
    point), and the draft's own sizes)."""
    d = {}
    h = m['over_arm'] - m['under_arm']
    # (never taller than a cap can be cut for this armhole, nor so low that the sleeve stands out as a wing)
    d['cap'] = h = float(np.clip(h, m['armhole'] / 4.5, m['armhole'] / 3.0))
    ease = 1.0 + ratios['cap_ease']
    d['front'] = wf = _width_for(m['armhole_front'] * ease, h, ratios['front_scoop'], ratios['crown'])
    d['back'] = wb = _width_for(m['armhole_back'] * ease, h, ratios['back_scoop'], ratios['crown'])
    d['bicep'] = wf + wb
    room = 2 * math.pi * ratios['room'] * w
    d['bicep_room'] = d['bicep'] - m['bicep']
    d['elbow'] = max(m['elbow'], m['forearm'] - 0.0) + room
    d['cuff'] = m['wrist'] + 2 * math.pi * ratios['room_cuff'] * w
    d['y_elbow'] = h + (m['bottom'][2] - m['z_elbow'])
    d['y_cuff'] = h + m['under_arm']
    # (a hand wider than the cuff: the seam is left open above the cuff, as far up as the sleeve is narrower than
    # the hand, and laced shut)
    need = m.get('hand', 0.0) * 1.03
    d['slit'] = float(np.clip((need - d['cuff']) / max(d['elbow'] - d['cuff'], 1e-6), 0.0, 1.0) * (d['y_cuff'] - d['y_elbow']))
    share = wf / (wf + wb)
    front = _half(wf, h, ratios['front_scoop'], ratios['crown'])
    back = _half(wb, h, ratios['back_scoop'], ratios['crown']) * np.array([-1.0, 1.0])
    side_f = [(share * d['elbow'], d['y_elbow']), (share * d['cuff'], d['y_cuff'])]
    side_b = [(-(1 - share) * d['cuff'], d['y_cuff']), (-(1 - share) * d['elbow'], d['y_elbow'])]
    d['outline'] = np.array([*front, *side_f, *side_b, *back[::-1]])
    d['cap_len'] = float(np.linalg.norm(np.diff(front, axis=0), axis=1).sum() + np.linalg.norm(np.diff(back, axis=0), axis=1).sum())
    return d


def draw(m, d, w, path, scale=42.0):
    """The cutting sheet: the sleeve laid flat, its lines named, and the tape's measures beside it."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 19)
    except OSError:
        f = fb = ImageFont.load_default()
    o = d['outline']
    lo, hi = o.min(0), o.max(0)
    W, H = int((hi[0] - lo[0]) * scale) + 620, int((hi[1] - lo[1]) * scale) + 150
    im = Image.new('RGB', (W, H), (244, 240, 230))
    dr = ImageDraw.Draw(im)
    ink, red, pale = (50, 36, 28), (150, 40, 34), (150, 135, 120)

    def at(x, y):
        return (60 + (x - lo[0]) * scale, 70 + (y - lo[1]) * scale)
    dr.polygon([at(*p) for p in o], fill=(214, 150, 140), outline=ink)
    dr.line([at(*p) for p in o] + [at(*o[0])], fill=ink, width=3)
    share = d['front'] / d['bicep']
    for name, y, half_b, half_f in (('bicep line', d['cap'], d['back'], d['front']),
                                    ('elbow line', d['y_elbow'], (1 - share) * d['elbow'], share * d['elbow'])):
        dr.line([at(-half_b, y), at(half_f, y)], fill=pale, width=2)
        dr.text((at(-half_b, y)[0] + 8, at(0, y)[1] - 24), name, fill=ink, font=f)
    dr.line([at(0, 0), at(0, d['y_cuff'])], fill=pale, width=2)                    # (the grain, down from the shoulder point)
    dr.line([at(0, 0)[0], at(0, 0)[1] - 12, at(0, 0)[0], at(0, 0)[1] + 12], fill=red, width=4)
    dr.text((at(0, 0)[0] - 52, at(0, 0)[1] - 40), 'shoulder point', fill=red, font=f)
    dr.text((at(-d['back'], d['cap'])[0], at(0, d['cap'])[1] - 62), 'BACK', fill=ink, font=fb)
    dr.text((at(d['front'], d['cap'])[0] - 62, at(0, d['cap'])[1] - 62), 'FRONT', fill=ink, font=fb)
    if d['slit'] > 0:
        t = d['slit'] / (d['y_cuff'] - d['y_elbow'])
        for sx, a, b in ((1.0, share * d['cuff'], share * d['elbow']), (-1.0, (1 - share) * d['cuff'], (1 - share) * d['elbow'])):
            dr.line([at(sx * a, d['y_cuff']), at(sx * (a + t * (b - a)), d['y_cuff'] - d['slit'])], fill=red, width=6)
        dr.text((at(share * d['cuff'], 0)[0] + 30, at(0, d['y_cuff'] - 0.5 * d['slit'])[1] - 10), 'slit, laced', fill=red, font=f)
    dr.text((at(-0.5 * d['cuff'], 0)[0], at(0, d['y_cuff'])[1] + 8), 'cuff   (seam under the arm, at both edges)', fill=ink, font=f)

    def row(name, v):
        return '%-34s %5.2f   %5.1f cm' % (name, v, v * CM)
    lines = [
        ('THE TAPE', None),
        ('armhole, as the doublet is cut', m['armhole']), ('   its front', m['armhole_front']), ('   its back', m['armhole_back']),
        ('   its depth', m['depth']),
        ('arm at the bicep', m['bicep']), ('arm at the elbow', m['elbow']), ('forearm at its widest', m['forearm']),
        ('wrist, at the cuff', m['wrist']),
        ('hand', m.get('hand')),
        ('shoulder point to elbow, over', m['to_elbow']), ('shoulder point to cuff, over', m['over_arm']),
        ('underarm point to cuff', m['under_arm']),
        ('', None), ('THE DRAFT', None),
        ('cap height (over less under)', d['cap']),
        ('cap curve (armhole + %.1f%%)' % (100 * RATIOS['cap_ease']), d['cap_len']),
        ('bicep line', d['bicep']), ('   front of the shoulder point', d['front']), ('   back of it', d['back']),
        ('   room over the arm', d['bicep_room']),
        ('elbow line', d['elbow']), ('cuff', d['cuff']), ('underarm seam', d['y_cuff'] - d['cap']),
        ('length, shoulder point to cuff', d['y_cuff']),
        ('cuff slit, laced (the hand passes)', d['slit']),
    ]
    fm = f
    try:
        fm = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 16)
    except OSError:
        pass
    x0, y0 = int((hi[0] - lo[0]) * scale) + 110, 60
    for name, v in lines:
        if v is None and name:
            dr.text((x0, y0), name + '                         units', fill=red, font=fb)
        elif v is not None:
            dr.text((x0, y0), row(name, v), fill=ink, font=fm)
        y0 += 25
    im.save(path)


def from_figure(log=print):
    """The measures and the draft from the doublet figure open in Blender."""
    import bpy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cut
    body = bpy.data.objects['Skin body']
    P, T, aw = fit.body_arrays(body)
    M = fit.measure(P, aw, None, cut.RATIOS)
    w = M['w']
    o = bpy.data.objects['Doublet']
    co = np.array([v.co[:] for v in o.data.vertices])
    hole = armhole(co, [list(p.vertices) for p in o.data.polygons])
    rig = bpy.data.objects['Rig']
    b = {n: rig.data.bones[n] for n in ('UpperArm.L', 'Forearm.L')}
    head, mid, tail = (rig.matrix_world @ v for v in (b['UpperArm.L'].head_local, b['Forearm.L'].head_local, b['Forearm.L'].tail_local))
    ho = bpy.data.objects.get('Hand.L')
    hand = np.array([(ho.matrix_world @ v.co)[:] for v in ho.data.vertices]) if ho else None
    m = measure(P, T, aw, hole, ((head.z, mid.z, tail.z), (head.x, head.y)), hand, w)
    d = draft(m, w)
    for k in ('armhole', 'armhole_front', 'armhole_back', 'depth', 'bicep', 'elbow', 'forearm', 'wrist', 'hand', 'to_elbow', 'over_arm', 'under_arm'):
        if k in m:
            log('tape: %-14s %6.2f  (%.1f cm)' % (k, m[k], m[k] * CM))
    for k in ('cap', 'cap_len', 'bicep', 'front', 'back', 'bicep_room', 'elbow', 'cuff', 'slit', 'y_elbow', 'y_cuff'):
        log('draft: %-13s %6.2f  (%.1f cm)' % (k, d[k], d[k] * CM))
    return m, d, w


if __name__ == '__main__':
    import bpy
    src = sys.argv[1]
    out = sys.argv[sys.argv.index('--out') + 1]
    bpy.ops.wm.open_mainfile(filepath=src)
    m, d, w = from_figure()
    draw(m, d, w, out)
    print('sheet:', out)
