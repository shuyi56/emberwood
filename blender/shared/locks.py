"""Hair built as locks, the way the face creator builds it (docs/hair-building-guide.md): each lock follows a guide
curve laid on the head, and is drawn as a clump of fine, flat strands that sit together at the root, part toward the
end and stop at staggered lengths, each tapering to a point.

Pure Python: no bpy or mathutils, so the geometry can be built and checked without Blender (see check_locks). The
figure scripts pass in a `surface(d)` function that gives the distance from the skull's centre to the outermost
surface along the unit direction d (in Blender, BVH ray casts against the head and the hair cap), and turn the
strands into a bmesh with to_bmesh.

Units are the figure's head-local units (a head is 8.6 high, the chin at the origin, the face toward -Y), the same
"rogue units" the face creator's lock code is written in, so its knots carry over unchanged.
"""
import math

HC = (0.0, 0.5, 4.8)     # the skull's centre


def sm(e0, e1, x):
    if e0 == e1:
        return 1.0 if x >= e1 else 0.0
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scl(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    n = math.sqrt(dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def frac(x):
    return x - math.floor(x)


def rdir(th, el):
    """The direction from the skull's centre at angle th round from the front and elevation el, as the face
    creator's rgDir (th > 0 toward +X)."""
    return (math.sin(th) * math.cos(el), -math.cos(th) * math.cos(el), math.sin(el))


def catmull(knots, samples):
    """A Catmull-Rom curve through the knots (tuples of any length), `samples` points in all."""
    out, n = [], len(knots)
    m = max(1, math.ceil(samples / (n - 1)))
    for i in range(n - 1):
        p0, p1, p2, p3 = knots[max(0, i - 1)], knots[i], knots[i + 1], knots[min(n - 1, i + 2)]
        for k in range(m):
            t = k / m
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2
                                    + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3) for c in range(len(p1))))
    out.append(tuple(knots[-1]))
    return out


def guide(knots, surface, lift0=0.0, samples=44, ride=None, hc=HC, over=None, smooth=0):
    """A lock's guide curve. knots: (th, el, lift) with the lift in head-local units over the scalp. The knots are
    eased in (th, el, lift) and every sample is then set on the head, so the curve follows the head between knots
    instead of cutting through it.

    surface(d) -> (scalp, outer): the distances along d to the scalp and to the outermost surface (the hair cap or
    whatever the lock lies on). Past its root the lock rides on the outer surface (never closer than ride[1] above
    it once it is ride[0] along), so a thick cap does not swallow it and cut through it in steps; the root itself may
    start under the cap, which hides it. ride=False lays it from the scalp only (a lock slipping out from under the
    hair). over=u0: the lift is measured from the outer surface instead, eased in over the first u0 of the lock (the
    face creator's Rogue fringe, which adds its lift to the cap's depth)."""
    if ride is None:
        ride = (0.12, 0.05)
    pts = catmull(knots, samples)
    n = len(pts)
    ds, rs, lo = [], [], []
    for q, (th, el, lift) in enumerate(pts):
        u = q / (n - 1)
        d = rdir(th, el)
        scalp, outer = surface(d)
        r = scalp + lift0 + lift
        floor = -1e9
        if over is not None:
            r += (outer - scalp) * sm(0.0, over, u)
        elif ride:
            floor = lerp(r, outer + ride[1], sm(ride[0] * 0.4, ride[0], u))
            r = max(r, floor)
        ds.append(d)
        rs.append(r)
        lo.append(floor)
    # (smooth: eases the height along the lock, so the root's rise out from under the cap turns into its run over
    # the head in a curve, not at a corner; never below the ride)
    for _ in range(smooth):
        rs = [rs[0]] + [max(lo[i], 0.25 * rs[i - 1] + 0.5 * rs[i] + 0.25 * rs[i + 1]) for i in range(1, n - 1)] + [rs[-1]]
    path = [add(hc, scl(d, r)) for d, r in zip(ds, rs)]
    return path, out_axes(path, hc)


def out_axes(path, hc=HC):
    """For each point, the direction a flat strand faces: away from the head, square to the curve."""
    n, O = len(path), []
    for q in range(n):
        T = norm(sub(path[min(n - 1, q + 1)], path[max(0, q - 1)]))
        nr = norm(sub(path[q], hc))
        O.append(norm(cross(norm(cross(T, nr)), T)))
    return O


def clump(path, O, width_at, ns, seed, tip=0.3, stagger=0.2, spread=0.6, layer=0.004):
    """The strands of one lock (hairClump in the face creator). width_at(u) is the lock's half-width along it.
    Returns strands, each a list of (centre, across axis, out axis, half-width, half-depth, u)."""
    n0, strands = len(path), []
    for si in range(ns):
        c = (si + 0.5) / ns - 0.5
        r1 = frac(math.sin(si * 12.9898 + seed * 78.233) * 43758.5453)
        n = max(8, round(n0 * (1 - stagger * r1 * (0.4 + 1.5 * abs(c)))))
        S = []
        for q in range(n):
            T = norm(sub(path[min(n0 - 1, q + 1)], path[max(0, q - 1)]))
            ax = norm(cross(T, O[q]))
            u = q / (n0 - 1)
            w = width_at(u)
            off = c * 2 * w * (0.75 + spread * sm(0.3, 1, u))
            p = add(add(path[q], scl(ax, off)), scl(O[q], layer * si))
            ww = w * 2.4 / ns * (1 - 0.4 * sm(0.65, 1, u)) * (1 - 0.45 * (2 * abs(c)) ** 2)
            # (the end of the strand tapers to a point over its last stretch, as the face creator fades it out)
            v = q / (n - 1)
            ww *= (1 - sm(1 - tip, 1.0, v)) ** 0.7 + 0.04 * (1 - v)
            S.append((p, ax, O[q], ww, max(0.018, 0.16 * ww), u))
        strands.append(S)
    return strands


def ribbons(strands, seg=6):
    """The strands as closed flat tubes. Returns (verts, faces, normals, info) where info gives each vertex
    (strand index, u along the lock, position across the strand)."""
    verts, faces, normals, info = [], [], [], []
    for k, S in enumerate(strands):
        rings = []
        for (p, ax, o, rx, ry, u) in S:
            ring = []
            for j in range(seg):
                a = 2 * math.pi * j / seg
                ca, sa = math.cos(a), math.sin(a)
                ring.append(len(verts))
                verts.append(add(p, add(scl(ax, rx * ca), scl(o, ry * sa))))
                # (lit as part of the hair's outer surface: the normal leans out from the head, so a strand's
                # underside does not turn dark against the forehead)
                normals.append(norm(add(scl(o, 0.8), scl(add(scl(ax, ca), scl(o, sa)), 0.35))))
                info.append((k, u, ca))
            rings.append(ring)
        # (wound so every face looks out of the strand: the ring runs from the across axis toward the out axis,
        # which turns back along the strand, so the sides go ring-to-next-ring first and the start cap keeps the
        # ring's order; wound the other way they faced in and rendered black under their custom normals)
        for ra, rb in zip(rings, rings[1:]):
            for j in range(seg):
                faces.append((ra[j], rb[j], rb[(j + 1) % seg], ra[(j + 1) % seg]))
        faces.append(tuple(rings[0]))
        faces.append(tuple(reversed(rings[-1])))
    return verts, faces, normals, info


def to_bmesh(bm, verts, faces):
    """Add the ribbons to a bmesh (Blender only)."""
    vs = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([vs[i] for i in f])
    return vs


def check_locks(locks, surface, hc=HC, hairline=None):
    """The checks the face creator runs before a render (hairlint.js), on lock guide paths: locks = [(name, path)].
    surface(d) -> (scalp, outer) as for guide. Returns a list of (fault, name, note):
      buried  the lock comes out of the hair and dips back into it, so the hair's surface cuts through it in steps
      float   its last third is held off the head with nothing under it (bangs standing off the face as a visor)
      flap    its end stands off the head
      kink    it bends sharply at one point"""
    out = []
    for name, path in locks:
        n = len(path)
        bur = cnt = fl = fc = 0
        out1 = False
        kink = 0.0
        for q, p in enumerate(path):
            v = sub(p, hc)
            r = math.sqrt(dot(v, v))
            scalp, outer = surface(scl(v, 1 / r))
            clear = r - outer
            cnt += 1
            if clear > 0.02:
                out1 = True
            elif (out1 or q > n / 2) and clear < -0.035:
                bur += 1
            if q >= 0.7 * n:
                fc += 1
                if clear > 0.47:
                    fl += 1
            if 1 < q < n - 2:
                a = norm(sub(p, path[q - 2]))
                b = norm(sub(path[q + 2], p))
                kink = max(kink, math.acos(max(-1.0, min(1.0, dot(a, b)))))
        v = sub(path[-1], hc)
        r = math.sqrt(dot(v, v))
        tip_clear = r - surface(scl(v, 1 / r))[1]
        if cnt > 4 and bur / cnt > 0.15:
            out.append(('buried', name, f'{round(100 * bur / cnt)}% of it dips back inside the hair'))
        if fc > 3 and fl / fc > 0.3:
            out.append(('float', name, f'{round(100 * fl / fc)}% of its last third is held off the head'))
        if tip_clear > 0.3:
            out.append(('flap', name, f'its end stands {tip_clear:.2f} off the head'))
        if kink > 0.6:
            out.append(('kink', name, f'bends {round(math.degrees(kink))} degrees at one point'))
    return out


def ranger_bangs(surface, s, frame=True):
    """One side of the Ranger's curtain bangs (the face creator's rangerBangs): curtain locks in two staggered
    layers, rooted beside the centre parting, lifting off the brow in a C and falling back to the skin below it,
    the inner edge of the opening concave; one wide cover lock over them; and, with frame, four long locks framing
    the face down the front of the side hair. s: the side (+1 toward +X). Returns (strands, [(name, guide path)])."""
    guides = []
    strands = []
    NL = 7
    for ly in range(2):
        for i in range(NL):
            t = (i + 0.5 * ly) / (NL - 0.5)
            if t > 0.56:      # (the outer ones are left to the cover lock)
                continue
            th0, thT, elT, dl = s * (0.03 + 0.2 * t), s * (0.36 + 0.76 * t), 0.14 - 0.56 * t ** 1.2, -0.12 if ly else 0.0
            jit = 0.012 * math.sin(i * 5.1 + s * 1.3 + ly * 2.2)
            ea = 1 - t
            # (the lift is at the root and over the brow; below it the lock falls back to the skin)
            K = [(th0, 1.05, 0.08), (lerp(th0, thT, 0.05), 0.88, 0.4 + 0.3 * ea + dl), (lerp(th0, thT, 0.17) + jit, 0.7, 0.62 + 0.5 * ea + dl),
                 (lerp(th0, thT, 0.4), lerp(0.62, elT, 0.4), 0.4 + 0.25 * ea + dl),
                 (lerp(th0, thT, 0.78) + jit, lerp(0.62, elT, 0.78), 0.22 + 0.08 * ea + 0.5 * dl), (thT + jit, elT, 0.08 + 0.3 * dl)]
            path, O = guide(K, surface, 0.02 * ly + 0.01 * (i % 3), ride=(0.12, 0.03 + 0.03 * ly + 0.015 * (i % 3)))
            w0 = lerp(0.36, 0.46, t)
            guides.append((f'curtain {s} {ly}.{i}', path))
            strands += clump(path, O, lambda u, w0=w0: w0 * 1.1 * (0.55 + 0.45 * sm(0, 0.3, u)), 7,
                               900 + i * 4 + ly * 2 + (50 if s > 0 else 0), tip=0.32, stagger=0.22)
    # one wide cover lock over each side of the curtain, lying on the others, its end back on the skin
    th0, thT, elT = s * 0.14, s * 1.06, -0.45
    K = [(th0, 1.05, 0.12), (lerp(th0, thT, 0.12), 0.88, 0.62), (lerp(th0, thT, 0.35), 0.68, 0.8), (lerp(th0, thT, 0.6), lerp(0.6, elT, 0.4), 0.56),
         (lerp(th0, thT, 0.86), lerp(0.6, elT, 0.78), 0.3), (thT, elT, 0.12)]
    path, O = guide(K, surface, 0.06, ride=(0.12, 0.12))
    guides.append((f'cover {s}', path))
    strands += clump(path, O, lambda u: 0.75 * (0.55 + 0.45 * sm(0, 0.3, u)), 11, 940 + (50 if s > 0 else 0), tip=0.26)
    # the face-framing locks: long curtains down the front edge of the side hair, from the temple to the jaw, on top
    # of the side hair
    if frame:
        for j in range(4):
            u = j / 3
            r, a, end = s * (0.62 + 0.3 * u), s * (0.98 + 0.2 * u), -0.95 + 0.25 * u
            K2 = [(r, 0.9, 0.1), (lerp(r, a, 0.45), 0.55, 0.5), (a, 0.25, 0.62), (a + s * 0.06, -0.08, 0.66),
                  (a + s * 0.08, lerp(-0.08, end, 0.5), 0.66), (a + s * 0.05, end, 0.62)]
            path, O = guide(K2, surface, 0.0, samples=50, ride=(0.15, 0.08))
            guides.append((f'frame {s} {j}', path))
            strands += clump(path, O, lambda u2: 0.36 * (1 - 0.35 * sm(0.6, 1, u2)), 7, 960 + j + (50 if s > 0 else 0),
                               tip=0.2, stagger=0.22, spread=0.4)
    return strands, guides


def strand(path, O, width_at, tip=0.18, depth=0.3, min_depth=0.03):
    """One fine strand along its own path (a lock drawn strand by strand, as the hooked centre locks are)."""
    n, S = len(path), []
    for q in range(n):
        T = norm(sub(path[min(n - 1, q + 1)], path[max(0, q - 1)]))
        ax = norm(cross(T, O[q]))
        u = q / (n - 1)
        ww = width_at(u) * ((1 - sm(1 - tip, 1.0, u)) ** 0.7 + 0.04 * (1 - u))
        S.append((path[q], ax, O[q], ww, max(min_depth * (1 - 0.7 * u), depth * ww), u))
    return S


def rogue_bangs(surface):
    """The Rogue's fringe, as the face creator's default (HAIR_STYLES.ponytail): swept bangs either side, each lock
    rooted back on the crown, lifting off the forehead in a C and sweeping away from the centre as it falls, short
    and thin in the middle and longer toward the temples, laid over the cap; in the middle a close set of narrow
    locks that fall nearly straight from the crown and hook to one side over their last third, each drawn as fine
    strands that part toward the end and stop at staggered lengths; and the face-framing locks at each temple,
    slipping out from under the combed-back hair and falling to the jaw. Angles as rdir (th > 0 toward +X).
    Returns (strands, [(name, guide path)])."""
    strands, guides = [], []
    NB = 8
    BANGS = []
    for bl in range(2):
        for bi in range(NB):
            f = (bi + (0.5 if bl else 0.0)) / (NB - 0.5)
            if f > 1.02 or f < 0.25:      # (the middle is left to the hooked locks)
                continue
            BANGS.append([0.3 * f, 0.05 + 1.2 * f, 0.25 - 0.5 * f ** 1.6 + (0.03 if bl else 0.0),
                          lerp(0.3, 0.38, sm(0, 0.45, f)), 1 - 0.03 * ((bi * 3 + bl) % 3), bl])
    for side in (-1, 1):
        for j, b in enumerate(BANGS):
            if side < 0 and b[0] < 0.15:      # (the inner locks on her left of the parting are left out: they crossed the centre)
                continue
            if side > 0:
                b = list(b)
                b[2] += 0.11 * (1 - sm(0.55, 0.95, b[0] / 0.3))      # (the left-most ones end at the brow, not over the eye)
            jit = 0.01 * math.sin(j * 7.3 + side * 1.7)
            th0, th1 = side * b[0] + jit, side * b[1] + 1.5 * jit
            inner = 1 - sm(0.22, 0.6, abs(b[1]))
            e1 = b[2] + 0.6 * jit
            dl = (-0.17 if b[5] else 0.0) + 0.055 * ((j * 2 + (1 if side > 0 else 0)) % 3 - 1)
            K = [(1.02, 0.08, 0), (0.86, 0.36 + dl, 0.05), (0.68, 0.58 + dl, 0.16), (0.5, 0.44 + dl, 0.36),
                 (lerp(0.5, e1, 0.5), 0.26 + 0.6 * dl, 0.64),
                 (e1, (0.1 + 0.4 * dl) * (1 - 0.6 * sm(0.5, 0.95, abs(b[1]) / 1.25)), 1)]
            # (rgDir(-X) in the face creator, X being the Rogue's own angle: the fringe sweeps out from the parting)
            K = [(-lerp(lerp(th0, th1, q3), side * (b[1] - 0.05) - 0.03 + 0.1 * q3, inner) + 0.17 * q3 ** 1.3, el, lift)
                 for el, lift, q3 in K]
            path, O = guide(K, surface, over=0.16, smooth=12)
            guides.append((f'bang {side} {j}', path))
            strands += clump(path, O, lambda u, w=b[3]: w * 1.15 * (0.55 + 0.45 * sm(0, 0.3, u)), 6,
                               501 + j + (40 if side > 0 else 0), tip=0.3, stagger=0.18)
    # the middle: narrow locks falling nearly straight from the crown and hooking to one side over their last
    # third, the set laid each a little over the one before, so no skin shows between them
    NH = 9
    for k in range(NH):
        t = k / (NH - 1)
        h0 = -0.22 + 0.68 * t + 0.13
        h1 = 0.19 * 1.3
        h2 = 0.16 - 0.09 * t + 0.13 * sm(0.45, 1, t) + 0.025 * math.sin(k * 2.7)
        h3 = 0.2
        NS, KS = 7, 0.03 * (k % 3)      # (each lock a little over its neighbours; riding the cap does the rest)
        gl = None
        for si in range(NS):
            c = (si + 0.5) / NS - 0.5
            r1 = frac(math.sin(si * 12.9898 + k * 78.233) * 43758.5453)
            r2 = frac(math.sin(si * 4.1 + k * 9.7) * 9631.7)
            spread, short, hook = c * 0.05, 0.13 * r1 * (0.4 + abs(c)), h1 * (0.8 + 0.4 * r2)
            K = [(1.03, 0.08, 0), (0.86, 0.36, 0.0), (0.68, 0.56, 0.0), (0.5, 0.44, 0.06),
                 (lerp(0.5, h2 + short, 0.55), 0.27, 0.4), (h2 + short, 0.1, 1)]
            K = [(-(h0 + spread * (0.25 + 0.75 * q) + hook * q ** 1.5), el, lift + (KS + 0.012 * si) * (1 - 0.7 * q)) for el, lift, q in K]
            path, O = guide(K, surface, ride=(0.35, 0.02), smooth=12)
            gl = gl or path
            strands.append(strand(path, O, lambda u: h3 * 2.3 / NS * (1 - 0.45 * sm(0.6, 1, u))))
        guides.append((f'hook {k}', gl))
    # the face-framing locks at each temple: five, the front ones longest, stepping shorter and back toward the ear,
    # rooted under the combed-back hair and slipping out from under it at the hairline
    for side in (-1, 1):
        for j in range(5):
            t = j / 4
            r, a, end = 0.46 + 0.36 * t, 0.66 + 0.3 * t, -1.12 + 0.32 * t
            K = [(r, 0.82), (lerp(r, a, 0.45), 0.5), (a, 0.22), (a + 0.08, -0.08), (a + 0.1, lerp(-0.08, end, 0.4)),
                 (a + 0.08, lerp(-0.08, end, 0.75)), (a + 0.04, end)]
            path, O = guide([(side * th, el, 0.04) for th, el in K], surface, samples=40, ride=False)
            guides.append((f'wisp {side} {j}', path))
            strands += clump(path, O, lambda u: (0.1 + 0.2 * (1 - u) ** 0.8) * 0.85 * (0.55 + 0.9 * u), 6,
                               301 + j * 10 + (100 if side > 0 else 0), tip=0.14, stagger=0.22, spread=0.3, layer=0.006)
    return strands, guides


def _wob(a, b):
    """A smooth pseudo-noise in [-1, 1] (stands in for mathutils.noise, so this runs without Blender)."""
    return 0.5 * math.sin(1.7 * a + 2.3 * b) + 0.3 * math.sin(3.1 * b - 0.7 * a + 1.1) + 0.2 * math.sin(5.3 * b + 2.9 * a)


def tail_locks(path, radius, k_tie):
    """The ponytail as a hierarchy of locks, as the face creator builds it: round a slim core, seven primary locks
    leave the tie together, bloom out just past it, fall gathered and part again toward the end, each twisting a
    little round the tail, swaying its own way and ending at its own length; each primary is three secondary locks
    that lie together near the tie and split apart over the lower half, tapering to points at staggered lengths.
    path: the tail's centre line (from inside the mound, past the tie, to the end); radius(s) -> (across, through).
    Returns strands (tubes: each point (centre, across axis, through axis, rx, ry, u)) and their tones."""
    N = len(path)

    def frame(j):
        T = norm(sub(path[min(N - 1, j + 1)], path[max(0, j - 1)]))
        ax = (1.0, 0.0, 0.0)
        ax = norm(sub(ax, scl(T, dot(ax, T))))
        return ax, norm(cross(T, ax))
    out, tones = [], []
    NP = 7
    for pl in range(NP):
        rp = frac(math.sin(pl * 12.9898) * 43758.5453)
        rq = frac(math.sin(pl * 4.77 + 1.3) * 24634.6)
        a0 = pl * 2 * math.pi / NP + 0.6 * (rp - 0.5)
        endP = 0.74 + 0.26 * ((pl * 5) % NP) / (NP - 1) + 0.06 * (rq - 0.5)
        sizeP = 0.72 + 0.56 * rq
        toneP = (1.0, 0.9, 1.08, 0.95, 1.04, 0.88, 1.0)[pl]
        twist, swA, swF, swP = 0.3 + 0.7 * rp, 0.55 + 0.65 * rq, 1.4 + 1.6 * rp, 2 * math.pi * rq
        for sl in range(3):
            rs = frac(math.sin(pl * 7.31 + sl * 3.17) * 9631.7)
            subW = 0.75 + 0.5 * rs
            i_end = int(min(1.0, endP * (0.86 + 0.14 * rs) - 0.04 * sl * (pl % 2)) * (N - 1))
            S = []
            for j in range(k_tie - 1, i_end + 1):
                s2 = j / (N - 1)
                u = (j - (k_tie - 1)) / max(1, i_end - k_tie + 1)
                r, rd = radius(s2)
                ax, dp = frame(j)
                fo = lerp(0.22, 0.78, sm(0, 0.16, s2)) - 0.1 * sm(0.25, 0.55, s2) + 0.45 * sm(0.6, 1, s2) * (0.8 + 0.4 * rp)
                thc = a0 + twist * s2 + 0.35 * sm(0.6, 1, s2) * (rp - 0.5)
                wob = 0.6 * sm(0.3, 1, s2) * _wob(pl * 3.7, s2 * 4)
                sway = swA * sm(0.15, 0.6, s2) * math.sin(swF * math.pi * s2 + swP)
                P0 = add(path[j], add(scl(ax, r * fo * math.cos(thc) + wob + sway), scl(dp, rd * fo * math.sin(thc) + 0.5 * sway * math.cos(swP))))
                pr = 0.3 * sizeP
                split = lerp(0.3, 1.4 + 0.5 * rs * (1 if sl == 2 else 0), sm(0.4 - 0.1 * rs, 0.95, s2))
                ths = thc + sl * 2 * math.pi / 3 + (0.5 + 0.6 * rs) * s2
                C = add(P0, add(scl(ax, r * pr * split * math.cos(ths)), scl(dp, rd * pr * split * math.sin(ths))))
                # (bound in at the tie, full through the fall, tapering to a point over a long last stretch)
                cr = pr * 0.62 * subW * sm(0, 0.05, u + 0.02) * (1 - u) ** 0.55 + 0.012
                S.append((C, scl(ax, -1.0), dp, r * cr, rd * cr * 1.15, u))      # (across = cross(T, out), as ribbons winds it)
            out.append(S)
            tones.append(toneP * (0.95 + 0.1 * rs))
    return out, tones
