"""The sorceress's gown (gown.py, the gown thread's) with the accessory module's gold chain and garnet pendant, and the
Face Creator's Sorceress head and hair hung on the Head empty (HEAD_GLB). For views.py --mannequin."""
import math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gown'))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
os.environ.setdefault('GOWN_NO_HAIR', '1')
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import female  # noqa: E402
# (her head set lower on the body: at the base figure's height the creator's head stood on a long, bare column of
# neck above the off-shoulder neckline. NECK_DROP units lower, set before the gown and the module read it)
NECK_DROP = float(os.environ.get('NECK_DROP', '0'))
female.HEAD_Z -= NECK_DROP
NECK_ANAT = os.environ.get('NECK_ANAT', '1') != '0'
if NECK_ANAT:
    # (read by gown.py as it loads: the shoulder points 0.6 higher, so the collarbones run nearer level, and the
    # collarbones and the hollows above them more than twice as strong)
    os.environ.setdefault('TRAP_DROP', '0.8')
    os.environ.setdefault('CLAV', '1.4')
import anatomy  # noqa: E402
anatomy.patch()
import gown as G  # noqa: E402
import accessories as acc  # noqa: E402

# (the body's close-up skin: the skin thread's new look (female.SKIN_LOOKS), 'final' unless SKIN_LOOK says,
# with its baked tone moved to the creator face's, BODY_SKIN, so the face and the body are one skin)
FLAT_BODY = bool(os.environ.get('FLAT_BODY'))
if hasattr(female, 'SKIN_LOOKS'):
    female.SKIN_LOOK = os.environ.get('SKIN_LOOK', 'final')
_tones = female.skin_tones


def neck_tone():
    """The gown's neck column is its own mesh, under the body's skin, and above h 52.5 it is the wider of the two, so
    it is what shows between the chin and the top of the chest. Its baked skin layers ('skin', 'skin fx', 'skin
    tex') came out darker and warmer than the body's, a tan band under the chin. Each is matched to the body's skin
    at the same height (the mean of the body's points within 0.5 of it), with a soft shade kept just under the jaw."""
    nk, sb = bpy.data.objects.get('Neck'), bpy.data.objects.get('Skin body')
    if nk is None or sb is None or os.environ.get('NECK_TONE', '1') == '0':
        return
    zb = [(sb.matrix_world @ v.co).z for v in sb.data.vertices]
    zn = [(nk.matrix_world @ v.co).z for v in nk.data.vertices]
    chin = CHIN_Z
    for nm in ('skin', 'skin fx', 'skin tex'):
        an, ab = nk.data.color_attributes.get(nm), sb.data.color_attributes.get(nm)
        if an is None or ab is None or an.domain != 'POINT' or ab.domain != 'POINT':
            continue
        # (the body's mean at each half unit of height, over its neck and upper chest)
        bins = {}
        for i, d in enumerate(ab.data):
            if zb[i] > 48.5:
                k = round(zb[i] * 2)
                acc_ = bins.setdefault(k, [0.0, 0.0, 0.0, 0])
                for c in range(3):
                    acc_[c] += d.color[c]
                acc_[3] += 1
        if not bins:
            continue
        top = max(bins)
        mean_n = [sum(d.color[c] for d in an.data) / len(an.data) for c in range(3)]
        for i, d in enumerate(an.data):
            k = min(top, round(zn[i] * 2))
            while k not in bins and k > min(bins):
                k -= 1
            ref = bins[k]
            tgt = [ref[c] / ref[3] for c in range(3)]
            sh = 1.0 - (0.12 if nm == 'skin' else 0.0) * _ss(chin - 1.2, chin - 0.1, zn[i])
            col = d.color
            d.color = tuple(col[c] * (tgt[c] / max(1e-6, mean_n[c])) * sh for c in range(3)) + (col[3],)


def _skin_tones():
    hexv = os.environ.get('BODY_SKIN', 'f0d1cb')
    face = [int(hexv[i:i + 2], 16) for i in (0, 2, 4)]
    mid = female.SKIN[2]
    # (the whole ramp scaled by how far the face's tone is from its middle step, so the shade under the jaw and
    # the flush keep their steps relative to it)
    female.SKIN = [[min(255, round(c * f / m)) for c, f, m in zip(step, face, mid)] for step in female.SKIN]
    res = _tones()
    neck_tone()        # (after the bake: it writes the 'skin' colours this reads)
    return res


female.skin_tones = _skin_tones

LENGTH = float(os.environ.get('LENGTH', 17.5))
HEAD_GLB = os.environ.get('HEAD_GLB', '/tmp/claude-0/head/sorceress-head.glb')
HEAD_FIT_S, HEAD_FIT_DZ = '1.1', '-0.6'     # the head's size on the gown's neck (the seat is measured: seat_head)
PIT_Z = 50.3                                # the pit of the neck, the notch between the collarbones
# (the chin stands above the pit of the neck by a third of the head, 8.6 tall before it is sized; NECK_LEN is that
# fraction)
CHIN_Z = PIT_Z + float(os.environ.get('NECK_LEN', '0.33')) * 8.6 * float(os.environ.get('HEAD_FIT_S', HEAD_FIT_S))


def seat_head(objs, unit=1.0):
    """Move the head's parts (objs, the roots to move) so its chin stands at CHIN_Z. The chin is measured, the lowest
    point at the front of the head near the middle: the creator head's chin and the mannequin head's sit at
    different heights above the head joint, and a seat worked out for one left the other's neck 0.9 too long."""
    bpy.context.view_layer.update()
    lo = None

    def walk(o):
        nonlocal lo
        if o.type == 'MESH':
            for v in o.data.vertices:
                q = o.matrix_world @ v.co
                if abs(q.x) < 0.5 and q.y < -1.9 and (lo is None or q.z < lo):
                    lo = q.z
        for c in o.children:
            walk(c)
    for o in objs:
        walk(o)
    if lo is None:
        return
    for o in objs:
        o.location.z += (CHIN_Z - lo) / unit      # (unit: the parent's scale, when objs are its children)
    print('seat_head: chin was', round(lo, 2), 'now', round(CHIN_Z, 2), flush=True)
_fuse = G.fuse_body
SLIM = float(os.environ.get('NECK_SLIM', '0'))     # (how much narrower the neck is made, as a fraction)
NECK_CY = 0.42                                        # (the neck's axis, front to back)


def _ss(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def slim_at(x, y, z):
    """The factor the neck is drawn in by toward its axis at a point: full from the base of the neck (h 50.9) up
    to just under the jaw, eased out into the shoulders and chest below and into the jaw above, and only near the
    axis, so the shoulders and the face keep their shape."""
    hz = female.HEAD_Z
    w = _ss(49.4, 50.6, z) * (1.0 - _ss(hz - 0.15, hz + 0.45, z))
    # (fading out by the distance to the side, not round the axis: the neck's sides stand well behind its axis,
    # where a radial fall-off left them at full width)
    w *= 1.0 - _ss(2.8, 4.0, abs(x))
    return 1.0 - SLIM * w


NECK_W = float(os.environ.get('NECK_W', '0'))     # the neck column's half-width, front view (0: the old slim)
_WCUR = {}


def measure_neck(o):
    """The body's half-width in the front view at each height (0.1 apart), as fused, before any shaping."""
    _WCUR.clear()
    M = o.matrix_world
    for v in o.data.vertices:
        p = M @ v.co
        if 48.5 < p.z < 54.0:
            k = round(p.z * 10)
            _WCUR[k] = max(_WCUR.get(k, 0.0), abs(p.x))


def _wtable():
    """The measured half-widths as a smooth curve: gaps filled, smoothed over about half a unit (the remesh's
    slices are ragged, and a factor read off each one rippled the neck into rings), and never wider going up."""
    import numpy as np
    if '_curve' in _WCUR:
        return _WCUR['_curve']
    ks = sorted(k for k in _WCUR if isinstance(k, int))
    zs = np.arange(ks[0], ks[-1] + 1)
    raw = np.array([_WCUR.get(int(k), np.nan) for k in zs])
    ok = ~np.isnan(raw)
    raw = np.interp(zs, zs[ok], raw[ok])
    g = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    g /= g.sum()
    sm = np.convolve(np.pad(raw, 9, mode='edge'), g, mode='valid')
    sm = np.minimum.accumulate(sm)          # (from the bottom up: the neck only narrows toward the head)
    _WCUR['_curve'] = (zs / 10.0, sm)
    return _WCUR['_curve']


def column_at(z):
    """The neck as a column: from the base of the neck up, each height's half-width brought to NECK_W (never
    widened), eased in over h 49.5 to 50.5, where the shoulders spread out from it. Below that the body is as
    fused. Returns the factor x is drawn in by at height z."""
    import numpy as np
    zs, sm = _wtable()
    w = float(np.interp(z, zs, sm))
    t = _ss(49.2, 51.0, z) * (1.0 - _ss(female.HEAD_Z - 0.1, female.HEAD_Z + 0.4, z))
    # (a little wider toward its base, as the neck's muscles spread to the collarbones: NECK_W at mid-neck)
    wn = NECK_W + 0.2 * (1.0 - _ss(50.0, 52.0, z)) - 0.1
    wt = w + (min(w, wn) - w) * t
    return wt / w if w > 1e-6 else 1.0


def slim_mesh(o, frame=None):
    M = frame if frame is not None else o.matrix_world
    inv = M.inverted()
    for v in o.data.vertices:
        p = M @ v.co
        k = column_at(p.z) if NECK_W > 0 and _WCUR else slim_at(p.x, p.y, p.z)
        if k < 1.0:
            p.x *= k
            # (front to back a good deal less: the column is narrower seen from the front, and drawing the nape
            # and the throat in as far dented the top of the back)
            ky = 1.0 - (1.0 - k) * (0.4 if NECK_W > 0 and _WCUR else 1.0)
            p.y = NECK_CY + (p.y - NECK_CY) * ky
            v.co = inv @ p
    o.data.update()


TRAP_CARVE = float(os.environ.get('TRAP_CARVE', '0'))     # (off: on, it squared the neck and pulled the chain up in front)


def carve_trapezius(o):
    """With the head set lower, the shoulders' slope ran from right under the jaw straight out to the arm, a cone
    with no neck in it. A neck is a column that stands up out of the shoulders: the top of the trapezius is let
    down beside it (nothing on the column itself, inside |x| 2, the full TRAP_CARVE from |x| 3.2 out), only above
    h 49 so the shoulder points and the bodice line stay put. The neck's sides then meet a gently falling
    shoulder line at a soft angle."""
    if TRAP_CARVE <= 0:
        return
    M = o.matrix_world
    inv = M.inverted()
    for v in o.data.vertices:
        p = M @ v.co
        ax = abs(p.x)
        if ax < 2.0 or p.z < 49.0:
            continue
        w = _ss(2.0, 3.2, ax) * _ss(49.0, 50.6, p.z)
        if w > 0:
            p.z -= TRAP_CARVE * w
            v.co = inv @ p
    o.data.update()


SHOULDER_ROUND = float(os.environ.get('SHOULDER_ROUND', '0.5'))


def round_shoulder_profile(o):
    """From the side the bare shoulder was a block: the back stood up as a flat wall to the top of the shoulder and
    met a level top at a corner, with the arm hanging from the front of it. The top of the shoulder is rounded over
    from the side: above h 47.6 the back is drawn forward more and more toward the top (1.5 at the shoulder line),
    and the front a little back, away from the neck and spine (nothing inside |x| 2, in full from |x| 4.5), so the
    trapezius rolls over into the back and the deltoid instead of ending in an edge."""
    if SHOULDER_ROUND <= 0:
        return
    M = o.matrix_world
    inv = M.inverted()
    for v in o.data.vertices:
        p = M @ v.co
        ax = abs(p.x)
        if ax < 2.0 or p.z < 47.6:
            continue
        w = _ss(2.0, 4.5, ax) * SHOULDER_ROUND
        if p.y > 0:
            k = 1.0 - w * (1.5 / 2.8) * _ss(47.6, 50.0, p.z) ** 1.5
        else:
            k = 1.0 - w * (0.8 / 2.6) * _ss(48.4, 50.2, p.z) ** 1.5
        p.y *= k
        v.co = inv @ p
    o.data.update()


NECK_STAND = os.environ.get('NECK_STAND', '0') != '0'   # (off: John asked for the gown's own neck and shoulders back)
TRAP_ANGLE = float(os.environ.get('TRAP_ANGLE', '19'))     # the shoulder line's fall from the neck, degrees
NECK_HALF = float(os.environ.get('NECK_HALF', '1.3'))     # the neck column's half-width where the shoulders leave it
ACROMION_X = 7.2


def stand_neck(objs):
    """The neck standing up out of the shoulders. Measured, her shoulder line ran from her chin straight out to the
    arm at 32 degrees, a cone with no neck in it: the trapezius met the neck 0.3 under the jaw. On a woman the neck
    is a column about 0.7 of the head's breadth across; the trapezius leaves it about halfway down (the height of
    the 7th vertebra, a little above the pit of the neck) and falls to the point of the shoulder at 15 to 25
    degrees, sagging slightly; the chin stands above the pit of the neck by the chin-to-nose distance (a third of
    the head). The points of the shoulders and everything below h 47.6 stay where the gown has them: between the
    neck and the shoulder point the surface above that height is let down in proportion, so the top of the shoulder
    comes down to the new line (TRAP_ANGLE from the shoulder point, ACROMION_X) and the neck's side is left
    standing, with a short fillet where the two meet (NECK_HALF to NECK_HALF + 0.65)."""
    if not NECK_STAND:
        return
    z0, step = 47.6, 0.25
    top = {}
    for o in objs:
        M = o.matrix_world
        for v in o.data.vertices:
            p = M @ v.co
            k = int(abs(p.x) / step)
            if p.z > top.get(k, -1e9):
                top[k] = p.z
    n = max(top) + 1
    H = [top.get(k, top[max(top)]) for k in range(n)]
    for _ in range(3):                                   # (smoothed: the bins' tops step with the mesh)
        H = [H[0]] + [(H[i - 1] + 2 * H[i] + H[i + 1]) / 4 for i in range(1, n - 1)] + [H[-1]]

    def h_at(x):
        f = min(n - 1.001, max(0.0, x / step - 0.5))
        i = int(f)
        return H[i] + (H[i + 1] - H[i]) * (f - i)
    za, x1 = h_at(ACROMION_X), NECK_HALF + 0.65
    tan = math.tan(math.radians(TRAP_ANGLE))

    def t_at(x):
        h = h_at(x)
        if x >= ACROMION_X or x <= NECK_HALF:
            return h
        line = za + tan * (ACROMION_X - x) - 0.12 * math.sin(math.pi * max(0.0, x - x1) / (ACROMION_X - x1))
        return min(h, h + (line - h) * _ss(NECK_HALF, x1, x))
    if os.environ.get('PROBE_NECK'):
        print('shoulder line was/now', [(x / 2, round(h_at(x / 2), 2), round(t_at(x / 2), 2)) for x in range(0, 17)], flush=True)
    for o in objs:
        M = o.matrix_world
        inv = M.inverted()
        for v in o.data.vertices:
            p = M @ v.co
            ax = abs(p.x)
            if p.z <= z0 or ax <= NECK_HALF or ax >= ACROMION_X:
                continue
            h, t = h_at(ax), t_at(ax)
            if h - t < 1e-4 or h <= z0 + 0.5:
                continue
            p.z = z0 + (min(p.z, h) - z0) * (t - z0) / (h - z0)
            v.co = inv @ p
        o.data.update()
    # (the top of the neck. The body's skin closes to a point inside the head, narrower under the jaw than the
    # creator head's own neck (1.35 to 1.65 half across), which then showed outside it as a pale collar; and the
    # 'Neck' core under the skin, a darker tone, stood out of the lowered skin as a band. The skin is widened to a
    # column of NECK_TOP half-width from h 51.6 up into the jaw, so the neck is the body's one skin all the way
    # up, and the core is drawn in under it)
    wt = float(os.environ.get('NECK_TOP', '1.7'))
    sb = next((o for o in objs if o.name == 'Skin body'), None)
    if sb is not None and wt > 0:
        M = sb.matrix_world
        inv = M.inverted()
        cur = {}
        for v in sb.data.vertices:
            p = M @ v.co
            if p.z > 51.3 and abs(p.x) < 3.0:
                k = round(p.z * 10)
                cur[k] = max(cur.get(k, 0.0), abs(p.x))
        ks = sorted(cur)
        w = [cur[k] for k in ks]
        for _ in range(4):
            w = [w[0]] + [(w[i - 1] + 2 * w[i] + w[i + 1]) / 4 for i in range(1, len(w) - 1)] + [w[-1]]
        cur = dict(zip(ks, w))
        ztop = max(ks) / 10.0
        for v in sb.data.vertices:
            p = M @ v.co
            if p.z <= 51.5 or abs(p.x) >= 3.0:
                continue
            wc = cur.get(round(p.z * 10))
            if not wc:
                continue
            f = 1.0 + (min(2.4, max(1.0, wt / wc)) - 1.0) * _ss(51.5, 52.0, p.z)
            if f > 1.0:
                p.x *= f
                p.y = NECK_CY + (p.y - NECK_CY) * (1.0 + (f - 1.0) * 0.8)
            # (and carried up into the head, a unit past the chin (about h 54.7 less NECK_DROP), so the column's
            # closed top is out of sight above the floor of the jaw instead of ending in a rim under it)
            if p.z > 51.6:
                chin = 54.7 - NECK_DROP
                p.z = 51.6 + (p.z - 51.6) * max(1.0, (chin + 0.75 - 51.6) / (ztop - 51.6))
                # (narrowing as it passes up behind the jawline, which is 1.2 to 1.5 half across there: at full
                # width its top corners showed beside the jaw)
                k = 1.0 - 0.3 * _ss(chin - 0.3, chin + 0.75, p.z)
                p.x = max(-wt * k, min(wt * k, p.x * k))     # (and its ragged closing rim held to the column)
                p.y = NECK_CY + (p.y - NECK_CY) * k
            v.co = inv @ p
        sb.data.update()
    nk = next((o for o in objs if o.name == 'Neck'), None)
    if nk is not None:
        M = nk.matrix_world
        inv = M.inverted()
        for v in nk.data.vertices:
            p = M @ v.co
            p.x *= 0.7
            p.y = NECK_CY + (p.y - NECK_CY) * 0.7
            v.co = inv @ p
        nk.data.update()


def neck_anatomy(sb, core):
    """The neck, the line from it to the shoulder, and the muscles and bones that show at the base of the throat,
    worked from surface anatomy (and a photograph of a bare neck and shoulders John gave as the reference):

    - the neck is a column, near round, about 0.6 of the head's breadth across, leaning a little forward as it
      rises; it does not end at the chin but runs on up behind the jaw to the skull under the ears;
    - its side does not meet the shoulder at an angle: the trapezius leaves the neck about halfway down and sweeps
      out to the point of the shoulder in one concave curve, steep beside the neck and near level by the shoulder;
    - the two sternocleidomastoids run from behind the ears down and forward to the notch between the collarbones,
      a V on the front of the neck, with the hollow of the notch between their ends.

    The body's skin closed to a point inside the head, a cone, with a separate darker neck core standing out of
    it. The cone is drawn up into the column (the core is tucked inside it, so the neck is the body's one skin),
    the tops of the shoulders are let down to the concave line, and the muscles and the notch are raised and sunk
    on the column."""
    import bmesh
    chin = CHIN_Z
    wn = float(os.environ.get('NECK_HALF_W', '2.0'))
    M = sb.matrix_world
    inv = M.inverted()
    # ---- finer skin over the neck and the base of the throat, for the muscles
    bm = bmesh.new()
    bm.from_mesh(sb.data)
    for zmin in (49.6, 50.9):
        es = [e for e in bm.edges if all((M @ v.co).z > zmin and abs((M @ v.co).x) < 3.7 for v in e.verts)]
        bmesh.ops.subdivide_edges(bm, edges=es, cuts=1, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(sb.data)
    bm.free()
    sb.data.update()
    V = sb.data.vertices
    P = [M @ v.co for v in V]
    # ---- the cone's section at each height: half-width, front and back
    zj = 51.0
    tab = {}
    for q in P:
        if q.z > zj - 0.2:
            k = round(q.z * 20)
            t = tab.setdefault(k, [0.0, 0.0, 0.0])
            t[0] = max(t[0], abs(q.x))
            if abs(q.x) < 0.5:
                t[1], t[2] = min(t[1], q.y), max(t[2], q.y)
    ks = sorted(tab)
    cols = [[tab[k][i] for k in ks] for i in range(3)]
    for c in cols:
        for _ in range(6):
            c[:] = [c[0]] + [(c[i - 1] + 2 * c[i] + c[i + 1]) / 4 for i in range(1, len(c) - 1)] + [c[-1]]
    ztop = ks[-1] / 20.0
    zlim = ztop - 0.25                       # (the closing tip's own section is ragged: read just under it)

    def sect(z):
        f = (min(z, zlim) * 20 - ks[0])
        i = max(0, min(len(ks) - 2, int(f)))
        a = max(0.0, min(1.0, f - i))
        return [c[i] + (c[i + 1] - c[i]) * a for c in cols]
    top = chin + 1.6

    def ncy(z):
        # (the column's axis, front to back: it leans forward as it rises, about 15 degrees above the shoulders,
        # so the nape runs up into the back of the skull and the throat comes forward under the jaw)
        return 0.145 - 0.1 * (min(z, 52.0) - 51.5) - 0.27 * max(0.0, z - 52.0)
    kz = (top - zj) / (ztop - zj)
    for i, q in enumerate(P):
        if q.z <= zj:
            continue
        w0, f0, b0 = sect(q.z)
        z1 = zj + (q.z - zj) * kz
        t = _ss(zj, zj + 0.6, q.z)
        wt = wn - 0.25 * _ss(chin + 0.4, top, z1)
        dt = 2.0 - 0.2 * _ss(51.5, 53.0, z1)
        cy0, d0 = (f0 + b0) / 2, max(0.3, (b0 - f0) / 2)
        cy1 = cy0 + (ncy(z1) - cy0) * t
        fx = 1.0 + (min(3.0, wt / max(0.3, w0)) - 1.0) * t
        fy = 1.0 + (min(3.0, dt / d0) - 1.0) * t
        q.x = max(-wt - 0.6 * (1 - t), min(wt + 0.6 * (1 - t), q.x * fx)) if t > 0.98 else q.x * fx
        q.y = cy1 + (q.y - cy0) * fy
        q.z = z1
    # ---- the line from the neck to the shoulder
    x_a, z0, step = ACROMION_X, 47.6, 0.25
    tops = {}
    for q in P:
        if q.z < chin - 0.6 or abs(q.x) > wn + 0.05:      # (not the column itself)
            k = int(abs(q.x) / step)
            if abs(q.x) > wn - 0.3 and q.z > tops.get(k, -1e9):
                tops[k] = q.z
    n = max(tops) + 1
    H = [tops.get(k, 0.0) for k in range(n)]
    first = min(tops)
    H = [H[first] if k < first else h for k, h in enumerate(H)]
    for _ in range(3):
        H = [H[0]] + [(H[i - 1] + 2 * H[i] + H[i + 1]) / 4 for i in range(1, n - 1)] + [H[-1]]

    def h_at(x):
        f = min(n - 1.001, max(0.0, x / step - 0.5))
        i = int(f)
        return H[i] + (H[i + 1] - H[i]) * (f - i)
    za = h_at(x_a)
    zn = chin - 0.33 * (chin - PIT_Z)            # (where the curve leaves the neck's side: a third down the neck)
    TA, TB = float(os.environ.get('TRAP_A', '0.6')), float(os.environ.get('TRAP_B', '1.6'))

    ang = math.tan(math.radians(float(os.environ.get('TRAP_ANGLE2', '24'))))
    sag = float(os.environ.get('TRAP_SAG', '0.15'))

    def t_at(x):
        # (the trapezius is a muscle with a body, thickening as it runs out to the collarbone and the shoulder
        # blade: its top is a soft slope, near straight, about 24 degrees from the shoulder point up to the neck,
        # sagging only a few hundredths of its span. A deep curve here emptied it, and the shoulder read as bone.
        # It turns up into the neck's side over a short fillet, not at a corner)
        u = (x - wn) / (x_a - wn)
        if u <= 0 or u >= 1:
            return h_at(x)
        line = za + ang * (x_a - x) - sag * math.sin(math.pi * u)
        line += (zn - (za + ang * (x_a - wn))) * (1.0 - _ss(wn, wn + 0.9, x)) ** 2
        line += (h_at(x) - line) * _ss(0.8, 1.0, u)
        return min(h_at(x), line)
    if os.environ.get('PROBE_NECK'):
        print('chin', round(chin, 2), 'junction', round(zn, 2), 'shoulder point', round(za, 2), 'line was/now',
              [(x / 2, round(h_at(x / 2), 2), round(t_at(x / 2), 2)) for x in range(4, 17)], flush=True)
    for q in P:
        ax = abs(q.x)
        if q.z <= z0 or ax <= wn + 0.02 or ax >= x_a:
            continue
        h, t = h_at(ax), t_at(ax)
        if h - t > 1e-4 and h > z0 + 0.5 and q.z <= h + 0.05:
            q.z = z0 + (q.z - z0) * (t - z0) / (h - z0)
    # (the drawn-up skin relaxed: stretched to 2.6 times its height, the remesh's uneven rows showed as creases)
    for v, q in zip(V, P):
        v.co = inv @ q
    bm = bmesh.new()
    bm.from_mesh(sb.data)
    sel = [v for v in bm.verts if (M @ v.co).z > 50.9 and abs((M @ v.co).x) < 3.0]
    for _ in range(int(os.environ.get('NECK_RELAX', '10'))):
        bmesh.ops.smooth_vert(bm, verts=sel, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(sb.data)
    bm.free()
    V = sb.data.vertices
    P = [M @ v.co for v in V]
    # (the shoulder point: where the trapezius ends and the deltoid begins the outline turns from one curve into
    # the other; the arm's cap stood up there as a knob. Relaxed over the top of the shoulder only)
    bm = bmesh.new()
    bm.from_mesh(sb.data)
    sel = [v for v in bm.verts if 5.6 < abs((M @ v.co).x) < 8.3 and (M @ v.co).z > 48.2]
    for _ in range(int(os.environ.get('ACROMION_RELAX', '6'))):
        bmesh.ops.smooth_vert(bm, verts=sel, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(sb.data)
    bm.free()
    V = sb.data.vertices
    P = [M @ v.co for v in V]
    # ---- the sternocleidomastoids and the notch between the collarbones
    A = float(os.environ.get('SCM', '0.2'))
    z_m, z_s = chin + 2.2, 50.55
    for q in P:
        if not (z_s - 0.9 < q.z < z_m + 0.3) or abs(q.x) > 3.2:
            continue
        cy = ncy(max(q.z, 51.0))
        r = math.hypot(q.x, q.y - cy)
        if r < 0.3:
            continue
        phi = math.atan2(abs(q.x), -(q.y - cy))             # (0 at the front, pi at the nape)
        if phi > 2.3:
            continue
        t = (z_m - q.z) / (z_m - z_s)
        tc = min(1.0, max(0.0, t))
        phi_c = math.radians(105.0 + (13.0 - 105.0) * tc ** 0.85)
        d = math.hypot(2.0 * (phi - phi_c), min(0.0, q.z - z_s) * 1.6 + max(0.0, q.z - z_m))
        sig = 0.6 + (0.36 - 0.6) * tc
        up = A * _ss(0.0, 0.2, t) * (1.0 - 0.35 * tc) * math.exp(-(d / sig) ** 2)
        # (the notch: a hollow between the two tendons, above the breastbone)
        up -= 1.4 * A * math.exp(-(q.x / 0.5) ** 2 - ((q.z - 50.55) / 0.45) ** 2) * (1.0 if q.y < cy else 0.0)
        if abs(up) > 1e-4:
            q.x += q.x / r * up
            q.y += (q.y - cy) / r * up
    for v, q in zip(V, P):
        v.co = inv @ q
    sb.data.update()
    if core is not None:                       # (the darker core, drawn in under the skin and out of sight)
        Mc = core.matrix_world
        ic = Mc.inverted()
        for v in core.data.vertices:
            q = Mc @ v.co
            q.x *= 0.6
            q.y = NECK_CY + (q.y - NECK_CY) * 0.6
            v.co = ic @ q
        core.data.update()


def fuse_body():
    # (after the trunk and the bare arms are made one skin: the chain lies on that skin)
    _fuse()
    bpy.context.view_layer.update()     # (the world matrices, which the shaping below reads)
    measure_neck(bpy.data.objects['Skin body'])
    # her neck slimmer, before the chain is let fall on it (the body's skin and the base of its neck)
    for nm in ('Neck', 'Skin body'):
        if nm in bpy.data.objects:
            slim_mesh(bpy.data.objects[nm])
            carve_trapezius(bpy.data.objects[nm])
    stand_neck([bpy.data.objects[nm] for nm in ('Neck', 'Skin body') if nm in bpy.data.objects])
    if NECK_ANAT:
        neck_anatomy(bpy.data.objects['Skin body'], bpy.data.objects.get('Neck'))
    round_shoulder_profile(bpy.data.objects['Skin body'])
    if os.environ.get('WIDTHS'):
        sb = bpy.data.objects['Skin body']
        best = {}
        for v in sb.data.vertices:
            p = sb.matrix_world @ v.co
            k = round(p.z * 2) / 2
            if 49.5 <= k <= 51.5 and abs(p.x) > best.get(k, (0,))[0]:
                best[k] = (round(abs(p.x), 2), round(p.y, 2))
        print('widths at fuse', sorted(best.items()), flush=True)
    acc.necklace('Necklace', female.FIG['body'], length=LENGTH, pendant='drop')


G.fuse_body = fuse_body
# ---- the belt: a gold girdle with a garnet, as the chain and its pendant are, and a sash that ends softly
BELT_GOLD = os.environ.get('BELT_GOLD', '1') != '0'
M_SASH = female.mat_def(86, 'sash', female.ramp(int(os.environ.get('SASH', '8a6a34'), 16), 0.12), 0.0, closeup='cloth')


def dress_belt():
    """The gown's belt, medallion and long panel were one flat grey, a strap beside the gold chain at her neck. The
    band and the medallion are the chain's gold, with a garnet cabochon set in the medallion (the pendant's stone),
    and the panel an old-gold sash. The panel came to a long point, a blade in the walk: it ends in a round tip."""
    belt, med, pn = (bpy.data.objects.get(n) for n in ('Belt', 'Belt medallion', 'Panel'))
    if BELT_GOLD:
        gold = acc._material(acc.M_ACC_GOLD)
        for o in (belt, med):
            if o is not None:
                o.data.materials[0] = gold
        if med is not None:
            l = med.location
            female.mesh_obj('Belt garnet', female.ellipsoid(0.36, 0.2, 0.36, 20, 10), acc.M_GARNET,
                            loc=(l.x, l.y - 0.1, l.z), subdiv=0, skin=G.trunk_weights)
            acc._material(acc.M_GARNET)
        if pn is not None:
            pn.data.materials[0] = female.mat(M_SASH)
    if pn is not None and os.environ.get('PANEL_TIP', '1') != '0':
        for v in pn.data.vertices:
            z = v.co.z
            if z >= 15.5:
                continue
            was = max(0.02, 0.95 * _ss(12.6, 15.5, z) ** 0.5)
            t = (13.55 - z) / 0.95       # (a half circle, as wide as the sash)
            now = 0.95 * (math.sqrt(max(0.0, 1.0 - t * t)) if t > 0 else 1.0)
            v.co.x = v.co.x / was * max(now, 0.02)
        pn.data.update()


def _finish(after_skin=None):
    dress_belt()
    return female.finish_figure(acc.follow)


G.finish_figure = _finish
# (the gown slopes its own shoulders (trapezius_slope, TRAP_DROP); the base figure's slope_shoulders, added to main
# since, let them down a second time here: thinner and lower than the gown's own renders. Off for this figure)
female.SHOULDER_SLOPE = float(os.environ.get('BASE_SLOPE', '0'))


def build():
    res = G.build()
    # (the body's sizes all come from the head it wears: measured here, before the body below the arms is built)
    bpy.context.view_layer.update()
    hh = anatomy.head_height(float(os.environ.get('HEAD_FIT_S', HEAD_FIT_S)))
    if hh:
        anatomy.configure(hh, CHIN_Z)
    anatomy.fuse_lower()
    # the layered skin's paint (the skin thread's skin_layers.py; John, 2026-11-18: pass the skin to the mannequin):
    # the feet's rosier tone and the thin-skin map where veins may show, baked into the body's point colours. Once
    # per build, on the rest shape; colours only.
    if anatomy.ON and bpy.data.objects.get('Skin body') is not None:
        import skin_layers
        _zw = female.SH_Z - female.UPPER - female.FORE
        skin_layers.paint_body(bpy.data.objects['Skin body'],
                               hands=[o for o in (bpy.data.objects.get('Hand.L'), bpy.data.objects.get('Hand.R')) if o],
                               neck=bpy.data.objects.get('Neck'),
                               wrists=[(-female.SH_X, 0.2, _zw), (female.SH_X, 0.2, _zw)])
        if hasattr(skin_layers, 'paint_face'):       # (the head's freckle and pore maps, where the module has them)
            skin_layers.paint_face(bpy.data.objects.get('Head shape'), bpy.data.objects['Skin body'],
                                   bpy.data.objects.get('Neck'))
    if os.environ.get('PROBE_CHIN'):
        for o in bpy.data.objects:
            if o.type == 'MESH' and (o.name.startswith('head part 0') or o.name in ('Skin body', 'Neck')):
                W = [o.matrix_world @ v.co for v in o.data.vertices]
                mid = [p for p in W if abs(p.x) < 0.4]
                hi = [p for p in W if p.z > 51.0 and abs(p.x) < 3.2]
                print('PROBE', o.name, 'zmin', round(min(p.z for p in W), 2), 'zmax', round(max(p.z for p in W), 2),
                      'front-mid lowest', [round(c, 2) for c in min((p for p in mid if p.y < -0.5), key=lambda p: p.z, default=Vector((0, 0, 0)))],
                      'halfW by h', [(h / 4, round(max((abs(p.x) for p in hi if abs(p.z - h / 4) < 0.07), default=0), 2)) for h in range(204, 222)], flush=True)
    acc.clearance_report()
    # (the legs: the floor-length skirt covers them but for the shoes, and in the idle her left thigh stood out
    # through the skirt at the hip; they stay in the build as the skirt's proxies, unseen)
    for nm in ('Leg.L', 'Leg.R') + (('Neck',) if NECK_STAND or NECK_ANAT else ()):
        o = bpy.data.objects.get(nm)
        if o is not None:
            o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
    # (the sleeve tops: the gown's sleeve starts 0.22 off the bare arm, and in the game it gripped the upper arm like
    # a tight cuff under the lace. Eased off the arm by SLEEVE_EASE at the top, most on the outside of the arm, and
    # back to the gown's own fit by the middle of the upper arm, where the bell starts to open)
    ease = float(os.environ.get('SLEEVE_EASE', '0.45'))
    for side in ('L', 'R'):
        for nm in (f'Sleeve.{side}', f'Sleeve lining.{side}', f'Lace shoulder.{side}'):
            o = bpy.data.objects.get(nm)
            if o is None or ease <= 0:
                continue
            sx = 1.0 if o.matrix_world.translation.x > 0 else -1.0
            for v in o.data.vertices:
                c = v.co
                r = math.hypot(c.x, c.y)
                if r < 1e-4:
                    continue
                k = ease * (1.0 - _ss(-8.5, -2.9, c.z))
                if k <= 0:
                    continue
                out = 0.5 + 0.5 * max(-1.0, min(1.0, sx * c.x / r))
                k *= 0.35 + 0.65 * out
                c.x += c.x / r * k
                c.y += c.y / r * k
            o.data.update()
    # (the tops of the shoulders: the fused skin's remesh left a row of little spikes along the outer shoulder,
    # where the arm's cap met the body; smoothed out over the top of each shoulder)
    sb = bpy.data.objects.get('Skin body')
    if sb is not None and float(os.environ.get('SHOULDER_SMOOTH', '1')) > 0:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(sb.data)
        mw = sb.matrix_world
        sel = [v for v in bm.verts if abs((mw @ v.co).x) > 4.5 and (mw @ v.co).z > 46.8]
        for _ in range(int(float(os.environ.get('SHOULDER_LAPLACE', '0')) * 12)):     # (off: it shrank the shoulders)
            bmesh.ops.smooth_vert(bm, verts=sel, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
        bm.to_mesh(sb.data)
        bm.free()
        sb.data.update()
        # (the spikes come in the pose, not the rest shape: points at the top of the shoulder carry a mix of the
        # arm's and the chest's weights and are pulled apart as the arm hangs. A corrective smooth after the rig,
        # over the shoulders only, smooths what the pose does to the surface and keeps the rest shape)
        # (and the cause itself, for the game model, which has no modifiers: the bone weights over the top of the
        # shoulder smoothed across neighbours, so no point is pulled a different way from the ones beside it)
        import numpy as np
        me = sb.data
        n = len(me.vertices)
        reg = np.zeros(n)
        for v in me.vertices:
            p = mw @ v.co
            reg[v.index] = _ss(4.2, 5.2, abs(p.x)) * _ss(46.2, 47.2, p.z)
        idx = np.nonzero(reg > 0)[0]
        bones = [g for g in sb.vertex_groups if g.name != 'shoulder smooth']
        W = np.zeros((n, len(bones)))
        gi = {g.index: k for k, g in enumerate(bones)}
        for v in me.vertices:
            for ge in v.groups:
                if ge.group in gi:
                    W[v.index, gi[ge.group]] = ge.weight
        ed = np.array([e.vertices[:] for e in me.edges])
        deg = np.bincount(ed.ravel(), minlength=n).astype(float)
        for _ in range(int(os.environ.get('WEIGHT_SMOOTH', '25'))):
            acc_ = np.zeros_like(W)
            np.add.at(acc_, ed[:, 0], W[ed[:, 1]])
            np.add.at(acc_, ed[:, 1], W[ed[:, 0]])
            avg = acc_ / np.maximum(deg, 1)[:, None]
            W[idx] += (avg[idx] - W[idx]) * (0.6 * reg[idx])[:, None]
        W[idx] /= np.maximum(W[idx].sum(1), 1e-9)[:, None]
        # (the top of the shoulder stays with the chest: carried by the upper arm, the whole cap swung forward with
        # the arm in the walk and stood out in front of the collarbone as a knob. Above the sleeve's top the arm's
        # share falls to ARM_ON_CAP of what it was by the shoulder line, the rest going to the chest)
        ci = next((k for k, g in enumerate(bones) if g.name == 'Chest'), None)
        arm = [k for k, g in enumerate(bones) if g.name.split('.')[0] in ('UpperArm', 'Forearm', 'Hand')]
        keep = float(os.environ.get('ARM_ON_CAP', '0.25'))
        if ci is not None and arm:
            for i in idx:
                z = (mw @ me.vertices[int(i)].co).z
                f = 1.0 - (1.0 - keep) * _ss(47.8, 49.0, z)
                if f < 1.0:
                    moved = W[i, arm].sum() * (1.0 - f)
                    W[i, arm] *= f
                    W[i, ci] += moved
        for i in idx:
            for k, g in enumerate(bones):
                if W[i, k] > 1e-4:
                    g.add([int(i)], float(W[i, k]), 'REPLACE')
                else:
                    g.remove([int(i)])
        vg = sb.vertex_groups.new(name='shoulder smooth')
        for v in sb.data.vertices:
            p = mw @ v.co
            w = _ss(4.2, 5.2, abs(p.x)) * _ss(46.2, 47.2, p.z)
            if w > 0:
                vg.add([v.index], w, 'REPLACE')
        cs = sb.modifiers.new('shoulder smooth', 'CORRECTIVE_SMOOTH')
        cs.vertex_group = vg.name
        cs.factor = 0.5
        cs.iterations = int(os.environ.get('SHOULDER_CS', '12'))
        cs.smooth_type = 'LENGTH_WEIGHTED'
        cs.rest_source = 'ORCO'
    if os.environ.get('PROBE'):
        _probe()
    if os.environ.get('SIDE'):
        import collections as _c
        dg = bpy.context.evaluated_depsgraph_get()
        for nm in ('Skin body', 'Sleeve.L', 'Lace shoulder.L', 'Bodice'):
            o = bpy.data.objects.get(nm)
            if o is None:
                continue
            me = o.evaluated_get(dg).to_mesh()
            R = _c.defaultdict(lambda: [9, -9, 0])
            for v in me.vertices:
                p = o.matrix_world @ v.co
                if 45.5 < p.z < 51 and p.x > 2.5:
                    k = round(p.z * 2) / 2
                    R[k][0] = min(R[k][0], p.y); R[k][1] = max(R[k][1], p.y); R[k][2] = max(R[k][2], p.x)
            print('side', nm, [(k, round(a, 2), round(b, 2), round(c, 2)) for k, (a, b, c) in sorted(R.items())], flush=True)
        print('side SH', female.SH_Z, getattr(female, 'SH_X', None), flush=True)
    if os.environ.get('WIDTHS'):
        import collections
        sb = bpy.data.objects['Skin body']
        W = collections.defaultdict(float)
        for v in sb.data.vertices:
            p = sb.matrix_world @ v.co
            if p.z > 47.5:
                W[round(p.z * 2) / 2] = max(W[round(p.z * 2) / 2], abs(p.x))
        print('widths', sorted(W.items()), 'HEAD_Z', female.HEAD_Z, flush=True)
    if os.environ.get('NO_DRESS'):
        # (the bare mannequin, for judging the body: the gown's pieces out of the camera's sight and casting nothing)
        left = []
        for o in bpy.data.objects:
            if o.type != 'MESH':
                continue
            if o.name.startswith(('Leg', 'Shoe')):          # (the legs show again: the skirt hid them)
                o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = True
                o.hide_render = False
            if o.name.startswith(('Bodice', 'Sleeve', 'Lace', 'Skirt', 'Panel', 'Belt', 'Proxy')):
                o.visible_camera = o.visible_shadow = o.visible_diffuse = o.visible_glossy = False
                o.hide_render = True
            else:
                left.append(o.name)
        print('NO_DRESS: left', left, flush=True)
        if os.environ.get('NO_ACC'):
            for o in bpy.data.objects:
                if o.name.startswith('Necklace'):
                    o.hide_render = True
    if not HEAD_GLB and 'Head' in bpy.data.objects:
        # (the mannequin's own head, fitted to the neck the same way: sized about its chin and seated lower)
        hd = bpy.data.objects['Head']
        fs, fz = float(os.environ.get('HEAD_FIT_S', HEAD_FIT_S)), float(os.environ.get('HEAD_FIT_DZ', HEAD_FIT_DZ))
        hd.scale = [c * fs for c in hd.scale]
        # (its parts, not the Head joint itself: the pose places the joint again on every frame)
        seat_head(list(hd.children), unit=fs)
        anatomy.head_stub(hd)
    if HEAD_GLB:
        head = bpy.data.objects['Head']
        # (the Blender head goes, as merge-head.mjs drops it; the creator's runs down into the gown's neck)
        if 'Head shape' in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects['Head shape'])
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=HEAD_GLB)
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            if o.parent is None:
                o.parent = head
                o.matrix_parent_inverse.identity()
        # (the creator's materials are plain glTF Principled; keep them for the 3D views, out of the sprite pipeline)
        # (the hair's veil left out, as merge-head.mjs leaves it out: a soft see-through layer the creator lays over
        # the locks beside the face for its own light. Unlit and blended, here it fell over the cheeks as a speckled
        # brown film, the dark band down both sides of her face)
        for o in list(new):
            if o.type != 'MESH':
                continue
            mats = [s_.material for s_ in o.material_slots if s_.material]
            if mats and all(not any(n.type == 'BSDF_PRINCIPLED' for n in m_.node_tree.nodes) for m_ in mats):
                new.remove(o)
                bpy.data.objects.remove(o)
        if os.environ.get('NO_HAIR'):          # (the bare mannequin: the hair hides the neck and the shoulders)
            for o in list(new):
                if o.type == 'MESH' and any(sl.material and sl.material.use_nodes and any(
                        n.type == 'BSDF_PRINCIPLED' and n.inputs['Alpha'].is_linked for n in sl.material.node_tree.nodes)
                        for sl in o.material_slots):
                    new.remove(o)
                    bpy.data.objects.remove(o)
        # (the creator's neck ends in a lip that turns in under the body's; cut it off where the two necks are the
        # same width, a little under the jaw, and let the body's neck carry on from there)
        import bmesh
        cut = float(os.environ.get('NECK_CUT', '-0.65'))
        for o in new:
            if o.type == 'MESH' and o.name.startswith('head part 0'):
                bm = bmesh.new()
                bm.from_mesh(o.data)
                M = o.matrix_basis          # (the creator's frame: the head's own, chin at 0)
                bmesh.ops.delete(bm, geom=[v for v in bm.verts if (M @ v.co).z < cut], context='VERTS')
                bm.to_mesh(o.data)
                bm.free()
                # (the creator bakes a soft shade into its vertex colours under the chin and down its neck; cut
                # there, that darker stub ended in a hard line on the body's plain neck. Faded out below the jaw,
                # so the neck is one tone across the join and only the light shades the chin's underside)
                col = o.data.attributes.get('Color')
                top = float(os.environ.get('NECK_FADE', '0.6'))
                if col is not None and col.domain == 'CORNER':
                    for li, lp in enumerate(o.data.loops):
                        z = (M @ o.data.vertices[lp.vertex_index].co).z
                        t = min(1.0, max(0.0, (top - z) / (top - cut)))
                        if t > 0:
                            t = t * t * (3 - 2 * t)
                            c = col.data[li].color
                            col.data[li].color = tuple(c[i] + (1.0 - c[i]) * t for i in range(3)) + (c[3],)
        # (the head fitted to the body, not the body to the head: the gown's neck and shoulders are as its own
        # sheet has them, and the creator head is seated on that neck. HEAD_FIT_DZ lowers it from the head joint
        # (the chin a third of a head above the pit of the neck), HEAD_FIT_S sizes it about its chin, HEAD_FIT_DY
        # moves it front to back over the neck's axis)
        fs, fz, fy = (float(os.environ.get(k, d)) for k, d in (('HEAD_FIT_S', HEAD_FIT_S), ('HEAD_FIT_DZ', HEAD_FIT_DZ), ('HEAD_FIT_DY', '0')))
        for o in new:
            if o.parent is head:
                o.scale = [c * fs for c in o.scale]
                o.location.y += fy
        seat_head([o for o in new if o.parent is head])
        if os.environ.get('PROBE_CHIN'):
            bpy.context.view_layer.update()
            for o in new:
                if o.type == 'MESH' and o.name.startswith('head part 0'):
                    W = [o.matrix_world @ v.co for v in o.data.vertices]
                    lo = min(W, key=lambda p: p.z)
                    chin = min((p for p in W if abs(p.x) < 0.3 and p.y < -1.2), key=lambda p: p.z)
                    print('PROBE head zmin', [round(c, 2) for c in lo], 'chin', [round(c, 2) for c in chin], 'zmax', round(max(p.z for p in W), 2),
                          'halfW by h', [(h / 4, round(max((abs(p.x) for p in W if abs(p.z - h / 4) < 0.07), default=0), 2),
                                          round(min((p.y for p in W if abs(p.z - h / 4) < 0.07), default=0), 2),
                                          round(max((p.y for p in W if abs(p.z - h / 4) < 0.07), default=0), 2)) for h in range(208, 226)], flush=True)
            for nm in ('Skin body', 'Neck'):
                o = bpy.data.objects[nm]
                W = [o.matrix_world @ v.co for v in o.data.vertices]
                hi = [p for p in W if p.z > 49.0 and abs(p.x) < 3.4]
                print('PROBE2', nm, 'zmax', round(max(p.z for p in W), 2), [(h / 4, round(max((abs(p.x) for p in hi if abs(p.z - h / 4) < 0.07), default=0), 2),
                      round(min((p.y for p in hi if abs(p.z - h / 4) < 0.07 and abs(p.x) < 0.4), default=0), 2),
                      round(max((p.y for p in hi if abs(p.z - h / 4) < 0.07 and abs(p.x) < 0.4), default=0), 2)) for h in range(198, 222)], flush=True)
        # (the body's skin as the game model has it: export_glb.py flattens every material to one plain tone, so
        # the close-up skin shader's pores and speckle never reach the world. Here in her own skin tone, the
        # creator preset's, so the face and the neck are one skin; BODY_SKIN, sRGB hex)
        hexv = os.environ.get('BODY_SKIN', 'f0d1cb')
        lin = lambda c: ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92  # noqa: E731
        rgb = [lin(int(hexv[i:i + 2], 16) / 255) for i in (0, 2, 4)]
        m = bpy.data.materials.get('skin') if FLAT_BODY else None
        if m is not None:
            nt = m.node_tree
            out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
            old = nt.nodes.get('close bsdf')
            if old is not None:
                old.name = 'close bsdf (pores)'
            pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
            pb.name = 'close bsdf'
            pb.inputs['Base Color'].default_value = (*rgb, 1.0)
            # (shaded as the creator shades its face, so the two skins read as one: its roughness, its low
            # specular, and the warm sheen that rosies the skin where it turns from the light)
            pb.inputs['Roughness'].default_value = 0.55
            pb.inputs['Specular IOR Level'].default_value = 0.275
            pb.inputs['Sheen Weight'].default_value = 1.0
            pb.inputs['Sheen Tint'].default_value = (1.0, 0.42, 0.32, 1.0)
            pb.inputs['Sheen Roughness'].default_value = 0.55
            nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])
        # (and the creator's neck drawn in the same, so the two necks meet at one width)
        from mathutils import Matrix
        for o in new:
            if o.type == 'MESH' and o.name.startswith('head part 0'):
                slim_mesh(o, Matrix.Translation((0, 0, female.HEAD_Z)) @ o.matrix_basis)
        # the face's skin matched to the body's, as merge-head.mjs matches it (SKIN_GAIN: linear r,g,b)
        gain = [float(v) for v in os.environ.get('SKIN_GAIN', '1,1,1').split(',')]
        for o in new:
            if o.type != 'MESH' or not o.name.startswith('head part 0'):
                continue
            for m in {s_.material for s_ in o.material_slots if s_.material}:
                nt = m.node_tree
                bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
                sock = bsdf.inputs['Base Color']
                if not sock.links:
                    continue
                src = sock.links[0].from_socket
                mul = nt.nodes.new('ShaderNodeMix')
                mul.data_type, mul.blend_type = 'RGBA', 'MULTIPLY'
                mul.inputs['Factor'].default_value = 1.0
                mul.inputs['B'].default_value = (*gain, 1.0)
                nt.links.new(src, mul.inputs['A'])
                nt.links.new(mul.outputs['Result'], sock)
        # (the hair's shadow on her face lighter: the creator lights its hair with a soft fill that lifts the shade
        # the locks cast on the cheeks; under one hard key here that shade went near black down both sides of the
        # face. Shadow rays see each lock as HAIR_SHADOW opaque, camera rays as it is)
        k = float(os.environ.get('HAIR_SHADOW', '0.0'))
        done = set()
        for o in new:
            if o.type != 'MESH' or o.name.startswith('head part 0'):
                continue
            for m in {s_.material for s_ in o.material_slots if s_.material}:
                if m.name in done:
                    continue
                done.add(m.name)
                nt = m.node_tree
                out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
                bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if out is None or bsdf is None or not bsdf.inputs['Alpha'].links:
                    continue          # (only the hair: its strands carry a cut-out; the eyes and teeth do not)
                # (soft strand tips: the creator cuts its alpha at 0.5 (a glTF MASK), which leaves the fine wisps at
                # the hairline as hard blocky steps; feed the strand alpha straight in so Cycles fades them)
                if os.environ.get('HARD_TIPS') is None:
                    cut = bsdf.inputs['Alpha'].links[0].from_node
                    if cut.type == 'MATH' and cut.operation == 'SUBTRACT':
                        lt = cut.inputs[1].links[0].from_node if cut.inputs[1].links else None
                        if lt is not None and lt.type == 'MATH' and lt.operation == 'LESS_THAN' and lt.inputs[0].links:
                            # (ramped over TIP_LO..TIP_HI: all that the cut kept stays solid, so the crown still
                            # covers the scalp, and only the wisps it dropped fade in instead of stepping)
                            ramp = nt.nodes.new('ShaderNodeMapRange')
                            ramp.inputs['From Min'].default_value = float(os.environ.get('TIP_LO', '0.0'))
                            ramp.inputs['From Max'].default_value = float(os.environ.get('TIP_HI', '0.5'))
                            nt.links.new(lt.inputs[0].links[0].from_socket, ramp.inputs['Value'])
                            nt.links.new(ramp.outputs['Result'], bsdf.inputs['Alpha'])
                src = out.inputs['Surface'].links[0].from_socket
                lp = nt.nodes.new('ShaderNodeLightPath')
                # (and the light bounced between the locks and the cheek: diffuse rays see it the same way, or the
                # shade comes back as a brown speckle where the fill is cut off and the plum hair bounces onto skin)
                any_ = nt.nodes.new('ShaderNodeMath')
                any_.operation = 'MAXIMUM'
                nt.links.new(lp.outputs['Is Shadow Ray'], any_.inputs[0])
                nt.links.new(lp.outputs['Is Diffuse Ray'], any_.inputs[1])
                # (and glossy rays: the skin's sheen at the jaw's edge mirrored the dark locks beside it, a brown
                # speckled line down both sides of the face)
                any2 = nt.nodes.new('ShaderNodeMath')
                any2.operation = 'MAXIMUM'
                nt.links.new(any_.outputs[0], any2.inputs[0])
                nt.links.new(lp.outputs['Is Glossy Ray'], any2.inputs[1])
                any_ = any2
                f = nt.nodes.new('ShaderNodeMath')
                f.operation = 'MULTIPLY'
                f.inputs[1].default_value = 1.0 - k
                nt.links.new(any_.outputs[0], f.inputs[0])
                tr = nt.nodes.new('ShaderNodeBsdfTransparent')
                mix = nt.nodes.new('ShaderNodeMixShader')
                nt.links.new(f.outputs[0], mix.inputs['Fac'])
                nt.links.new(src, mix.inputs[1])
                nt.links.new(tr.outputs['BSDF'], mix.inputs[2])
                nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
        if os.environ.get('NO_HAIR_TEST'):
            for o in new:
                if o.type == 'MESH' and any(s_.material and s_.material.name in done for s_ in o.material_slots):
                    o.hide_render = True
        print('hair shadow lightened on', len(done), 'materials', flush=True)
        print('creator head:', len(new), 'objects', flush=True)
    if os.environ.get('SKIN_LAYERS') and not FLAT_BODY and bpy.data.materials.get('skin') is not None:
        import skin_layers      # (the layered skin itself, for the 3D look only: it replaces the skin material's nodes)
        skin_layers.apply(bpy.data.materials['skin'])
    if os.environ.get('MEASURE'):
        bpy.context.view_layer.update()
        anatomy.measure()
    return res


def _probe():
    """(debug, PROBE=1: what the front camera sees first over the skirt, posed at the idle)"""
    import collections
    from mathutils import Vector
    sc = bpy.context.scene
    sc.frame_set(1)
    dg = bpy.context.evaluated_depsgraph_get()
    hits = collections.Counter()
    for zi in range(0, 60):
        z = 20 + zi * 0.33
        for xi in range(-40, 41):
            x = xi * 0.2
            ok, loc, n, i, ob, M = sc.ray_cast(dg, Vector((x, -200, z)), Vector((0, 1, 0)))
            if ok:
                hits[ob.name] += 1
                if ob.name not in ('Skirt', 'Panel', 'Belt', 'Belt medallion', 'Bodice') and not ob.name.startswith(('Sleeve', 'Arm', 'Hand')):
                    hits[(ob.name, round(x), round(z))] += 1
    for k, v in sorted(hits.items(), key=lambda kv: -kv[1])[:12]:
        print('probe', k, v, flush=True)
    print('probe objs', [(o.name, o.type, round(o.dimensions.z, 1)) for o in bpy.data.objects if any(t in o.name for t in ('Leg', 'Shoe', 'Foot', 'Skin', 'Lace'))], flush=True)
    sp = collections.Counter()
    for zi in range(0, 40):
        z = 47.5 + zi * 0.06
        for xi in range(0, 60):
            x = 5.5 + xi * 0.05
            for sx in (-1, 1):
                ok, loc, n, i, ob, M = sc.ray_cast(dg, Vector((sx * x, -200, z)), Vector((0, 1, 0)))
                if ok:
                    sp[(ob.name, sx)] += 1
    print('probe shoulder', sp.most_common(12), flush=True)
    top = collections.Counter()
    for xi in range(0, 120):
        x = 5.4 + xi * 0.02
        for yi in range(-20, 21):
            y = NECK_CY + yi * 0.08
            for sx in (-1, 1):
                ok, loc, n, i, ob, M = sc.ray_cast(dg, Vector((sx * x, y, 60)), Vector((0, 0, -1)))
                if ok:
                    top[ob.name] += 1
    print('probe top', top.most_common(12), flush=True)
    import bmesh as _bm
    sb = bpy.data.objects['Skin body']
    print('probe sb mods', [(m.type, m.name) for m in sb.modifiers], len(sb.data.vertices), flush=True)
    b = _bm.new(); b.from_mesh(sb.data)
    mw = sb.matrix_world
    devs = []
    for v in b.verts:
        p = mw @ v.co
        if abs(p.x) > 4.5 and p.z > 47.0 and v.link_edges:
            m = sum((e.other_vert(v).co for e in v.link_edges), Vector()) / len(v.link_edges)
            devs.append(((v.co - m).length, tuple(round(c, 2) for c in p), len(v.link_edges)))
    devs.sort(reverse=True)
    print('probe devs', devs[:15], flush=True)
