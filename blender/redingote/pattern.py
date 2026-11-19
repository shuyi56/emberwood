"""The redingote's sleeve as a tailor drafts it: flat pattern pieces from the measured arm and armhole, each seam edge
paired with the edge it is sewn to (John's rule, 2026-11-19: every garment drafted flat first; the sewing waits for
his go, so nothing here is joined or draped yet).

A two-piece sleeve of ca. 1885-1895 with a gathered head:

  topsleeve    the wide outer panel, crown to wrist. Its head is cut higher and wider than a plain sleeve between the
               front and back notches: that is the cloth gathered into the armhole for the puff. Straight grain down the
               upper arm; its back (hindarm) edge longer over the elbow by the elbow ease, eased in when sewn.
  undersleeve  the narrow inner panel, underarm to wrist, its short head plain, matching the armhole's underarm.
               An underlap at the wrist on its hindarm edge, under the vent.
  cuff         a straight-grain band and its facing, the ends lapped at the vent.
  support      the head's net support: the head's shape cut plain, to the armhole's own length.

Every size is measured on the body (vest/fit.measure, armhole.armhole) or a share of w, the neck's half-width. The
units are the figure's (63 to 1.65 m: 2.62 cm a unit); draw() prints them in cm.

    pieces(P, T, aw, M)    {name: Piece}: each piece's stitch line (the seam line, by edges), its cut line (the
                           stitch line let out by the allowances), grain line and notches
    SEAMS                  which edge is sewn to which, and how much one is eased or gathered into the other
"""
import math

import numpy as np

import armhole

CM = 165.0 / 63.0            # centimetres in a unit
RATIOS = {
    'ease': 0.10,            # round the arm, in w (sleeve.RATIOS['ease']: the drafted sleeve's own)
    'top_share': 0.60,       # the topsleeve's share of the sleeve's width (the forearm and hindarm seams lie under it)
    'under_head': 0.33,      # the undersleeve's share of the armhole, round the underarm (pitch to pitch)
    'under_dip': 0.90,       # the undersleeve's width at the cap's base against the armhole's underarm it is sewn to
    'fore_ahead': 2.5 / CM,  # the forearm seam this far ahead of the underarm's lowest point, along the armhole
    'notch_in': 2.5 / CM,    # the gathers stop this far short of each sleeve seam (plain cloth next to the seams)
    'full': 0.55,            # the head's gathers: this much more cloth than the armhole between the notches
    'crown_rise': 0.50,      # the crown raised above a plain sleeve's for the puff, in w (sleeve.RATIOS['puff'])
    'elbow_ease': 1.0 / CM,  # the topsleeve's hindarm edge longer than the undersleeve's over the elbow
    'elbow_bend': 1.2 / CM,  # the sleeve's forward curve at the elbow (a two-piece sleeve is cut bent)
    'support_chord': 0.80,   # the head support's width against the armhole length it spans (its edge a plain cap)
    'cuff': 4.5 / CM,        # the cuff's depth
    'cuff_drop': 0.6 / CM,   # the cuff's lower edge this far below the wrist joint (over the wrist bone: sleeve's hem)
    'lap': 1.5 / CM,         # the cuff's ends lapped at the vent
    'vent': 7.0 / CM,        # the vent, up the hindarm seam from the hem
    'underlap': 2.0 / CM,    # the undersleeve's extension under the vent
    'seam': 1.0 / CM,        # seam allowance
    'seam_head': 1.5 / CM,   # at the armhole
    'hem': 3.0 / CM,         # turned up at the wrist (under the cuff)
}


class Piece:
    """A flat piece: its stitch line as named edges (each an [n, 2] polyline, in order round the piece, clockwise),
    each edge's allowance, its grain line and its notches. cut() is the cutting line."""

    def __init__(self, name, edges, allowance, grain, notches=(), count=2):
        self.name, self.edges, self.allowance = name, edges, allowance
        self.grain, self.notches, self.count = grain, list(notches), count

    def outline(self):
        return np.concatenate([e[:-1] for e in self.edges.values()])

    def length(self, edge):
        e = self.edges[edge]
        return float(np.linalg.norm(np.diff(e, axis=0), axis=1).sum())

    def cut(self):
        """The cutting line: each edge moved out by its own allowance, and each corner closed where the two moved
        edges meet (their end segments extended), so the cut line turns the corner without a jog."""
        offs = []
        for nm, e in self.edges.items():
            d = np.gradient(e, axis=0)
            n = np.stack([d[:, 1], -d[:, 0]], 1)            # (outward for a clockwise outline, y down)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            offs.append(e + self.allowance[nm] * n)
        k = len(offs)
        for i in range(k):
            a, b = offs[i], offs[(i + 1) % k]
            p, r = a[-2], a[-1] - a[-2]
            q, t = b[0], b[1] - b[0]
            den = r[0] * t[1] - r[1] * t[0]
            if abs(den) > 1e-9 * np.linalg.norm(r) * np.linalg.norm(t):
                x = p + r * ((q[0] - p[0]) * t[1] - (q[1] - p[1]) * t[0]) / den
                if np.linalg.norm(x - a[-1]) < 4.0 * max(self.allowance.values()) + 1e-9:
                    a[-1] = x
                    b[0] = x
                    continue
            mid = 0.5 * (a[-1] + b[0])
            a[-1], b[0] = mid, mid
        return np.concatenate([o[:-1] for o in offs])


def _bezier(p0, p1, p2, p3, n=40):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _len(c):
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def _solve(f, target, lo, hi):
    """The x in [lo, hi] where f(x) = target (f rising)."""
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def girth(P, aw, z, side=1.0, band=0.12):
    """The arm's girth at a height on the rest shape, taut (the convex hull of its level section)."""
    s = P[(np.abs(P[:, 2] - z) < band) & (side * P[:, 0] > 0) & (aw > 0.9)][:, :2]
    h = armhole._hull(s)
    return float(np.linalg.norm(np.roll(h, -1, 0) - h, axis=1).sum())


def measure(P, T, aw, M, z_elbow, z_wrist, ratios=RATIOS):
    """The sleeve's measurements: the armhole's parts (from the shared loop), the arm's girths and lengths."""
    w = M['w']
    loops = armhole.armhole(P, aw, M, T)
    L = loops['L']
    loop = L['loop']
    seg = np.linalg.norm(np.roll(loop, -1, 0) - loop, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])            # (from the crown, forward first)
    G = s[-1]
    i_u = int(np.argmin(loop[:, 2]))
    s_u = s[i_u]
    s_fore = s_u - ratios['fore_ahead']                     # (the forearm seam, ahead of the underarm)
    s_hind = s_fore + ratios['under_head'] * G              # (the hindarm seam, behind it)
    m = dict(w=w, armhole=G, crown=0.0, fore=s_fore, hind=s_hind,
             front_head=s_fore, back_head=G - s_hind, under_head=s_hind - s_fore,
             depth=L['depth'],
             bicep=girth(P, aw, L['z_under'] - 0.8 * w), elbow=girth(P, aw, z_elbow),
             wrist=girth(P, aw, z_wrist + 0.3 * w),
             outside=float(L['top'][2] - z_wrist), inside=float(L['under'][2] - z_wrist),
             upper=float(L['top'][2] - z_elbow), z_elbow=z_elbow, z_wrist=z_wrist)
    m['rise_back'] = float(np.interp(s_hind, s[:-1], loop[:, 2]) - np.interp(s_fore, s[:-1], loop[:, 2]))
    m['notch_front'] = s_fore - ratios['notch_in']
    m['notch_back'] = s_hind + ratios['notch_in']
    m['gathered'] = m['notch_front'] + (G - m['notch_back'])    # (the armhole between the notches, over the crown)
    return m, loops


def draft(m, ratios=RATIOS):
    """The pieces, drafted flat from the measurements m. x across the sleeve (front to back), y down from the crown
    line; the topsleeve and undersleeve share the forearm and hindarm seam lines' heights."""
    w = m['w']
    ease = 2 * math.pi * ratios['ease'] * w
    Wb, We, Ww = m['bicep'] + ease, m['elbow'] + ease, m['wrist'] + ease
    # the undersleeve's width at the cap's base: a little short of the armhole's underarm, so its head dips into the
    # underarm curve and matches it; the topsleeve takes the rest of the sleeve's width
    rb = m['rise_back']
    under_w = math.sqrt(max((ratios['under_dip'] * m['under_head']) ** 2 - rb ** 2, (0.3 * m['under_head']) ** 2))
    ts = max(ratios['top_share'], 1.0 - under_w / Wb)
    # heights: the cap's base (the underarm) below the crown line, the elbow, the wrist
    cap = 0.75 * m['depth'] + ratios['crown_rise'] * w
    y_el = m['upper']
    y_wr = m['outside'] - (ratios['cuff'] - ratios['cuff_drop'])   # (the sleeve ends at the cuff's top)
    bend = ratios['elbow_bend']

    def sides(share, y):
        """The panel's front and back edge x at height y (cap base, elbow, wrist), widths by share, bent at the
        elbow (both edges move forward by the bend there)."""
        W = np.interp(y, [cap, y_el, y_wr], [Wb, We, Ww]) * share
        c = -bend * np.interp(y, [cap, y_el, y_wr], [0.0, 1.0, 0.3])
        return c - 0.5 * W, c + 0.5 * W

    ys = np.linspace(cap, y_wr, 60)
    # (the hindarm seam meets the armhole higher than the forearm seam, by the armhole's own rise between them, so
    # the back edges start higher: each edge as long as the arm is from its seam's point to the cuff)
    yb = np.linspace(cap - rb, y_wr, 60)
    fx_t, _ = sides(ts, ys)
    _, bx_t = sides(ts, yb)
    fx_u, _ = sides(1.0 - ts, ys)
    _, bx_u = sides(1.0 - ts, yb)
    # the topsleeve's hindarm edge let out over the elbow by the ease (eased into the undersleeve's when sewn)
    bulge = np.exp(-0.5 * ((yb - y_el) / (0.18 * (y_wr - cap))) ** 2)
    extra = ratios['elbow_ease']
    k = _solve(lambda a: _len(np.stack([bx_t + a * bulge, yb], 1)), _len(np.stack([bx_u, yb], 1)) + extra, 0.0, 3.0)
    bx_t2 = bx_t + k * bulge
    fore_t = np.stack([fx_t, ys], 1)[::-1]                 # (wrist up to the cap's base: clockwise, y down)
    vent0 = y_wr - ratios['vent']
    iv = int(np.searchsorted(yb, vent0))
    hind_t = np.stack([bx_t2, yb], 1)[:iv + 1]               # (sewn: the cap's base down to the vent's top)
    vent_t = np.stack([bx_t2, yb], 1)[iv:]                   # (the vent: faced, lapped over the undersleeve's underlap)
    # the topsleeve's head: from its forearm edge's top over the crown to its hindarm edge's top. Its length: the
    # plain stretches beside the seams, and the gathered armhole between the notches let out by the fullness
    target = ratios['notch_in'] * 2 + m['gathered'] * (1.0 + ratios['full'])
    p0, p3 = np.array([fx_t[0], cap]), np.array([bx_t2[0], cap - rb])

    def head(h, off):
        top = np.array([0.5 * (p0[0] + p3[0]) + off * (p3[0] - p0[0]), cap - h])
        a = _bezier(p0, p0 + [0.0, -0.55 * h], top + [-0.35 * (top[0] - p0[0]) - 0.6 * h, 0.0], top, 40)
        b = _bezier(top, top + [0.35 * (p3[0] - top[0]) + 0.6 * h, 0.0], p3 + [0.0, -0.55 * h], p3, 40)
        return np.concatenate([a, b[1:]]), len(a)
    # (the crown, where the head meets the bodice's shoulder seam, as far along the head as it is along the armhole
    # once the gathers are let out: the front's gathered stretch is the longer, so the crown stands behind the middle)
    crown_at = (ratios['notch_in'] + (m['front_head'] - ratios['notch_in']) * (1.0 + ratios['full'])) / target

    def shaped(off):
        h_ = _solve(lambda hh: _len(head(hh, off)[0]), target, 0.1, 6.0 * cap + 10.0)
        c, na = head(h_, off)
        return h_, c, _len(c[:na]) / _len(c)
    off = _solve(lambda o: shaped(o)[2], crown_at, -0.45, 0.45)
    h, head_t, _ = shaped(off)
    wrist_t = np.stack([np.linspace(bx_t2[-1], fx_t[-1], 12), np.full(12, y_wr)], 1)
    top = Piece('topsleeve', {'head': head_t, 'hindarm': hind_t, 'vent': vent_t, 'wrist': wrist_t, 'forearm': fore_t},
                {'head': ratios['seam_head'], 'hindarm': ratios['seam'], 'vent': ratios['seam'], 'wrist': ratios['hem'],
                 'forearm': ratios['seam']},
                grain=np.array([[-bend * 0.0, cap - 0.6 * h], [-bend * 0.0, y_el]]))
    # notches on the head: the gathers between them
    sh = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(head_t, axis=0), axis=1))])
    for d_ in (ratios['notch_in'], sh[-1] - ratios['notch_in']):
        top.notches.append(head_t[int(np.argmin(np.abs(sh - d_)))])
    top.notches.append(head_t[int(np.argmin(head_t[:, 1]))])          # (the crown: to the bodice's shoulder seam)
    top.notches.append(np.array([bx_t2[int(np.argmin(np.abs(yb - y_el)))], y_el]))   # (the elbow: ease between)
    # the undersleeve: its own plain head, a shallow curve dipping below the cap's base, to the armhole's underarm
    q0, q3 = np.array([fx_u[0], cap]), np.array([bx_u[0], cap - rb])

    def uhead(d):
        mid = 0.5 * (q0 + q3) + [0.0, d]
        a = _bezier(q3, q3 + [-0.3 * (q3[0] - q0[0]), d * 0.2], mid + [0.25 * (q3[0] - q0[0]), 0.0], mid, 30)
        b = _bezier(mid, mid + [-0.25 * (q3[0] - q0[0]), 0.0], q0 + [0.3 * (q3[0] - q0[0]), d * 0.2], q0, 30)
        return np.concatenate([a, b[1:]])
    d = _solve(lambda d_: _len(uhead(d_)), m['under_head'], 0.0, 4.0) if _len(uhead(0.0)) < m['under_head'] else 0.0
    head_u = uhead(d)
    fore_u = np.stack([fx_u, ys], 1)
    hind_u = np.stack([bx_u, yb], 1)[:iv + 1][::-1]          # (sewn: the vent's top up to the cap's base)
    # (the underlap under the vent: the hindarm edge stepped out by it, from the wrist up to the vent's top)
    lap_u = np.concatenate([np.stack([bx_u[iv:] + ratios['underlap'], yb[iv:]], 1)[::-1],
                            [[bx_u[iv], yb[iv]]]])
    wrist_u = np.stack([np.linspace(fx_u[-1], bx_u[-1] + ratios['underlap'], 10), np.full(10, y_wr)], 1)
    # (clockwise like the topsleeve: the head front to back, down the hindarm and underlap, the wrist back to front,
    # up the forearm)
    rev = lambda e: e[::-1].copy()                           # noqa: E731
    under = Piece('undersleeve', {'head': rev(head_u), 'hindarm': rev(hind_u), 'underlap': rev(lap_u),
                                  'wrist': rev(wrist_u), 'forearm': rev(fore_u)},
                  {'head': ratios['seam_head'], 'hindarm': ratios['seam'], 'underlap': ratios['seam'],
                   'wrist': ratios['hem'], 'forearm': ratios['seam']},
                  grain=np.array([[0.5 * (fx_u[0] + bx_u[0]), cap + d + 1.0], [0.5 * (fx_u[0] + bx_u[0]), y_el]]))
    under.notches.append(np.array([bx_u[int(np.argmin(np.abs(yb - y_el)))], y_el]))
    under.notches.append(np.array([bx_u[iv], yb[iv]]))
    # the cuff and its facing: straight-grain bands, the wrist's cloth girth and the lap
    Lc = Ww + ratios['lap']

    def band(name):
        r = np.array([[0.0, 0.0], [Lc, 0.0], [Lc, ratios['cuff']], [0.0, ratios['cuff']]])
        edges = {'top': np.stack([np.linspace(0, Lc, 10), np.zeros(10)], 1),
                 'end_back': np.stack([np.full(4, Lc), np.linspace(0, ratios['cuff'], 4)], 1),
                 'bottom': np.stack([np.linspace(Lc, 0, 10), np.full(10, ratios['cuff'])], 1),
                 'end_front': np.stack([np.zeros(4), np.linspace(ratios['cuff'], 0, 4)], 1)}
        del r
        return Piece(name, edges, {k_: ratios['seam'] for k_ in edges}, grain=np.array([[0.1 * Lc, 0.5 * ratios['cuff']],
                                                                                       [0.9 * Lc, 0.5 * ratios['cuff']]]))
    cuff, facing = band('cuff'), band('cuff facing')
    for k_ in range(3):
        cuff.notches.append(np.array([Lc - 0.5 * ratios['lap'], (0.25 + 0.25 * k_) * ratios['cuff']]))   # (buttons)
    # the head's support: the topsleeve's head cut plain to the armhole between the notches, down to the puff's end
    gat = m['gathered'] + 2 * ratios['notch_in']
    a0 = np.array([0.0, 0.0])

    chord = ratios['support_chord'] * gat

    def shead(hh):
        return _bezier(a0, a0 + [0.05 * chord, -hh], a0 + [0.95 * chord, -hh], a0 + [chord, 0.0], 50)
    # (its edge the plain cap's curve: the gathered head's armhole length on a chord the plain sleeve's width at the
    # notches, so it holds the head's line where the gathered cloth would slump)
    hs = _solve(lambda hh: _len(shead(hh)), gat, 0.0, gat)
    sup_top = shead(hs)
    depth = 2.0 * w
    sup_bot = _bezier(a0 + [chord, 0.0], a0 + [0.75 * chord, depth], a0 + [0.25 * chord, depth], a0, 50)
    support = Piece('head support (net)', {'head': sup_top, 'hem': sup_bot},
                    {'head': ratios['seam_head'], 'hem': 0.0}, grain=np.array([[0.5 * chord, 0.1 * depth], [0.5 * chord, 0.8 * depth]]))
    return {p.name: p for p in (top, under, cuff, facing, support)}


# which edge is sewn to which; 'ease' is how much longer the first is than the second, taken in as it is sewn
SEAMS = [
    ('forearm seam', ('topsleeve', 'forearm'), ('undersleeve', 'forearm'), 'plain'),
    ('hindarm seam', ('topsleeve', 'hindarm'), ('undersleeve', 'hindarm'), 'eased over the elbow, down to the vent'),
    ('vent', ('topsleeve', 'vent'), ('undersleeve', 'underlap'), 'faced, lapped over the underlap'),
    ('sleeve head', ('topsleeve', 'head'), ('bodice', 'armhole: front notch over the crown to the back notch'), 'gathered between the notches'),
    ('underarm', ('undersleeve', 'head'), ('bodice', 'armhole: the underarm, forearm seam to hindarm seam'), 'plain'),
    ('cuff', ('cuff', 'top'), ('sleeve', 'wrist (topsleeve + undersleeve)'), 'plain, ends lapped at the vent and buttoned'),
    ('cuff facing', ('cuff facing', 'top'), ('cuff', 'bottom'), 'turned in'),
    ('support', ('head support (net)', 'head'), ('topsleeve', 'head'), 'tacked in the head seam, the head gathered over it'),
]


def report(pieces, m):
    """The edges' lengths against their partners', in cm."""
    out = []
    t, u = pieces['topsleeve'], pieces['undersleeve']
    out.append('armhole %.1f cm: over the crown %.1f, underarm %.1f; gathered between notches %.1f' % (
        m['armhole'] * CM, (m['armhole'] - m['under_head']) * CM, m['under_head'] * CM, m['gathered'] * CM))
    out.append('topsleeve head %.1f cm (%.0f%% of the armhole over the crown)' % (
        t.length('head') * CM, 100 * t.length('head') / (m['armhole'] - m['under_head'])))
    out.append('undersleeve head %.1f cm against the underarm %.1f' % (u.length('head') * CM, m['under_head'] * CM))
    out.append('forearm seam: top %.1f, under %.1f cm' % (t.length('forearm') * CM, u.length('forearm') * CM))
    out.append('hindarm seam: top %.1f, under %.1f cm (eased %.1f over the elbow)' % (
        t.length('hindarm') * CM, u.length('hindarm') * CM, (t.length('hindarm') - u.length('hindarm')) * CM))
    out.append('wrist: sleeve %.1f cm (closed), cuff %.1f (lap %.1f)' % (
        (t.length('wrist') + u.length('wrist') - RATIOS['underlap']) * CM, pieces['cuff'].length('top') * CM, RATIOS['lap'] * CM))
    out.append('girths: bicep %.1f, elbow %.1f, wrist %.1f cm; lengths: outside %.1f, inside %.1f cm' % (
        m['bicep'] * CM, m['elbow'] * CM, m['wrist'] * CM, m['outside'] * CM, m['inside'] * CM))
    return out


def draw(pieces, m, path, px_cm=9.0):
    """The pattern sheet as a PNG: each piece's cut line, its stitch line dashed, grain line, notches, its name and
    count, the seams listed with their lengths."""
    from PIL import Image, ImageDraw
    s = px_cm * CM
    order = ['topsleeve', 'undersleeve', 'head support (net)', 'cuff', 'cuff facing']
    boxes, x0 = [], 30
    for nm in order:
        c = pieces[nm].cut()
        lo, hi = c.min(0), c.max(0)
        boxes.append((nm, x0 - lo[0] * s, 60 - lo[1] * s if nm in ('topsleeve', 'undersleeve') else None, lo, hi))
        x0 += (hi[0] - lo[0]) * s + 40 if nm in ('topsleeve', 'undersleeve') else 0
    W = max(1000, int(x0 + max((pieces[n].cut().max(0)[0] - pieces[n].cut().min(0)[0]) * s for n in order[2:]) + 60))
    H = int(max((pieces[n].cut().max(0)[1] - pieces[n].cut().min(0)[1]) * s for n in order[:2]) + 380)
    im = Image.new('RGB', (W, H), (250, 248, 242))
    dr = ImageDraw.Draw(im)
    y_side = 60
    for nm, ox, oy, lo, hi in boxes:
        p = pieces[nm]
        if oy is None:
            ox, oy = x0 - lo[0] * s, y_side - lo[1] * s
            y_side += (hi[1] - lo[1]) * s + 40
        T = lambda q: [(ox + a * s, oy + b * s) for a, b in q]       # noqa: E731
        cut = p.cut()
        dr.line(T(np.vstack([cut, cut[:1]])), fill=(30, 36, 60), width=2)
        for e in p.edges.values():
            q = T(e)
            for k in range(0, len(q) - 1, 2):
                dr.line([q[k], q[k + 1]], fill=(170, 60, 50), width=1)
        g = T(p.grain)
        dr.line(g, fill=(90, 90, 90), width=1)
        for end in (g[0], g[1]):
            dr.ellipse([end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3], outline=(90, 90, 90))
        for n_ in p.notches:
            a, b = T([n_])[0]
            dr.line([(a - 5, b - 5), (a + 5, b + 5)], fill=(170, 60, 50), width=2)
            dr.line([(a - 5, b + 5), (a + 5, b - 5)], fill=(170, 60, 50), width=2)
        c0 = T([0.5 * (lo + hi)])[0]
        dr.text((c0[0] - 40, c0[1]), '%s\ncut %d' % (p.name, p.count), fill=(30, 36, 60))
    yt = H - 260
    dr.text((30, yt - 30), 'Redingote sleeve, ca. 1885-1895: two-piece, gathered head. Red dashes: stitch lines. '
            'Blue: cut lines. X: notches. Grey: grain.', fill=(30, 36, 60))
    for i, line in enumerate(report(pieces, m)):
        dr.text((30, yt + 18 * i), line, fill=(30, 36, 60))
    for i, (nm, a, b, how) in enumerate(SEAMS):
        dr.text((30, yt + 18 * (8 + i) - 10), '%s: %s %s to %s %s, %s' % (nm, a[0], a[1], b[0], b[1], how), fill=(80, 50, 40))
    im.save(path)
    return path
