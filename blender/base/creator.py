"""A face from the face creator (/mnt/project-files/face-creator) on a Blender figure.

tools/face-creator/export-face.cjs writes a face as one file: every setting of the creator's, the colours it
derives from them, and the creator's own head surface (its headPoint) sampled on a grid. apply() puts that face on
the base figure before it builds:

    import creator
    creator.apply('blender/faces/warden.json.gz')      # after the figure's own settings, before its build

or, for any figure, from the command line: blender/base/views.py --figure warden --face blender/faces/rogue.json.gz

The head's shape comes from the surface, not from the sliders, so the Blender head is the creator's head (Sex, the
lips, the nostrils, every slider, and whatever the creator adds later) without porting a slider by hand. The eyes,
brows, lashes, make-up, lips and skin are the base figure's own, driven by the creator's settings (see apply).
Without a file nothing here runs and every figure builds as before.
"""
import gzip
import hashlib
import json
import math

import numpy as np

import female


class Surface:
    """The creator's head surface: point(z, th) for z from the neck up to the crown (head heights) and th the angle
    round the head (0 straight forward, toward +x positive), the same parameters female.sculpted_head and
    female._head_point use. Bicubic (Catmull-Rom) between the grid's samples, round the head periodically."""

    def __init__(self, s):
        nz, nt, u = s['nz'], s['nt'], s['unit']
        self.nz, self.nt = nz, nt
        self.z0, self.z1, self.th0 = s['z0'], s['z1'], s['th0']
        self.dz = (self.z1 - self.z0) / (nz - 1)
        self.dt = 2 * math.pi / nt
        self.P = np.stack([np.cumsum(np.array(s[k], dtype=np.int64).reshape(nz, nt), axis=1) * u
                           for k in ('x', 'y', 'z')], axis=-1)        # (nz, nt, 3)
        self.id = hashlib.sha256(self.P.tobytes()).hexdigest()[:16]

    def __repr__(self):
        return f'Surface({self.id})'

    @staticmethod
    def _w(t):
        t2, t3 = t * t, t * t * t
        return np.stack([-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t,
                         0.5 * t3 - 0.5 * t2], axis=-1)

    def points(self, z, th):
        """Points (n, 3) at arrays z and th."""
        z, th = np.atleast_1d(np.asarray(z, dtype=np.float64)), np.atleast_1d(np.asarray(th, dtype=np.float64))
        fz = np.clip((z - self.z0) / self.dz, 0, self.nz - 1)
        ft = ((th - self.th0) / self.dt) % self.nt
        iz, it = np.floor(fz).astype(int), np.floor(ft).astype(int)
        wz, wt = self._w(fz - iz), self._w(ft - it)
        out = np.zeros(z.shape + (3,))
        for a in range(4):
            rz = np.clip(iz - 1 + a, 0, self.nz - 1)
            for b in range(4):
                rt = (it - 1 + b) % self.nt
                out += (wz[..., a] * wt[..., b])[..., None] * self.P[rz, rt]
        return out

    def point(self, z, th):
        p = self.points(z, th)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def section(self, z, c, s):
        """(x, y) round the head at height z, as female.head_section gives them (here with the face on): for the
        ears' placement (ear.side_x_finder)."""
        x, y, _ = self.point(z, math.atan2(s, c))
        return x, y


def load(path):
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as fh:
        doc = json.load(fh)
    if not str(doc.get('format', '')).startswith('emberwood-creator-face/'):
        raise ValueError(f'{path}: not a face creator export (tools/face-creator/export-face.cjs)')
    return doc


def _hex(h):
    return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]


def _scaled(rows, r):
    return [[min(255, max(0, round(v * k))) for v, k in zip(c, r)] for c in rows]


def _recolour(mid, rmp, alt=None):
    m = female.MATERIALS[mid]
    m['ramp'] = np.array(rmp, dtype=np.float32)
    if alt is not None:
        m['alt'] = np.array(alt, dtype=np.float32)


# the creator's settings with no counterpart in the Blender figure yet (they are listed by apply)
UNUSED = ['contour', 'highlight', 'smoke', 'lower', 'noseFlush', 'coolUnder', 'texture', 'oil', 'sss',
          'lashUp', 'lashLen', 'lashLean', 'lashCurl', 'lashFuzz', 'lashThick', 'lashLow', 'browLen', 'browDensity',
          'irisSize', 'pupil', 'fibre', 'crypts', 'collarette', 'limbal', 'warm', 'glint', 'vessels', 'redness', 'flush',
          'tint', 'gazeX', 'lift', 'seed', 'hairStyle', 'hairEnds', 'beard', 'beardTone', 'beardLen', 'beardDensity',
          'beardGrey', 'cheekLine', 'neckLine']


def apply(src, shape=True, colours=True, hair_colour=True, quiet=False):
    """Put a creator face (an export's path, or the loaded dict) on the base figure. Call it after the figure's
    own settings (it replaces them) and before the build.

    shape        the head from the creator's surface, and the face-shape settings (FACE, the lids' construction)
    colours      skin, flush, iris, brows, lips, blush, eyeshadow and liner from the creator's settings
    hair_colour  the hair material in the creator's hair tone (the figure keeps its own hairstyle)
    Returns the loaded export."""
    doc = load(src) if isinstance(src, str) else src
    P = doc['settings']
    f = female
    if shape:
        surf = Surface(doc['surface'])
        f.HEAD_SOURCE = surf
        f.HEAD_SOURCE_ID = surf.id      # (a plain setting, so the render queue's head memo keys on the surface)
        for k in f.FACE:
            if k in P:
                f.FACE[k] = P[k]
        f.FACE['sex'] = P.get('sex', 0)
        # the lids, as the creator's "Copy ranger.py values" writes them
        f.EYE_OPEN, f.EYE_TILT = P['open'], P['tilt']
        f.EYE_TOP_PEAK = (P['peakPos'], 0.62)
        f.EYE_BOT_LOW = (-P['peakPos'] * 1.67, 0.32)
        f.EYE_INNER_ANGLES = (P['innerAngle'], P['innerAngle'] * 0.26)
        f.EYE_OUTER_ANGLES = (P['outerAngle'], P['outerAngle'] * 0.68)
        f.EYE_INNER_RUN = P['innerRun']
        f.EYE_PEAK_FULL = (P['archFull'], P['archFull'] * 0.96)
        f.UNDER_JAW_SHADE = (P['jawShade0'], P['jawShadeRange'], P['jawShadeAmt'])
    if colours:
        D = doc['derived']
        # skin: the ranger's ramp (and her flushed one) scaled to the creator's tone, as its export does
        base = [207, 159, 132]
        r = [a / b for a, b in zip(_hex(P['skinTone']), base)]
        if hasattr(f, 'SKIN_FAIR'):
            f.SKIN_FAIR = False      # (the export's own skin tone wins over the Fair swatch switch)
        f.SKIN[:] = _scaled(f.SKIN, r)
        f.SKIN_FLUSH[:] = _scaled(f.SKIN_FLUSH, r)
        _recolour(f.M_SKIN, f.SKIN, f.SKIN_FLUSH)
        f.FLUSH_TONE = (_hex(P['flushTone']), P.get('flushAmt', 1.0))
        f.SKIN_FRECKLES, f.SKIN_NOISE = P.get('freckles', 1.0), P.get('blotch', 1.0)
        f.BLUSH = (P.get('blush', 0.0), tuple(_hex(P['blushTone'])))
        f.LIP_COLOURS = dict(D['lips'])
        f.LIP_GLOSS = P.get('lipGloss', 0.0)
        _recolour(f.M_EYE, D['iris'])
        _recolour(f.M_BROW, f.ramp(int(P['browColor'][1:], 16), 0.08))
        f.BROW_TIP = _hex(P['browColor'])
        if hasattr(f, 'MAKEUP'):
            sh, sx = _hex(P['shadow']), P.get('sex', 0)
            f.MAKEUP.update({'shadow': tuple(sh), 'shadowDark': tuple(round(v * 0.49) for v in sh),
                             'shadowLow': tuple(min(255, round(v * 1.32)) for v in sh),
                             'shadowAmt': 0.78 * P['shadowStrength'], 'shadowInner': P['shadowInner'],
                             'liner': P['liner'], 'wing': P['wing'],
                             # (a male face's brows (sex) sit lower and flatter and are heavier, as the creator's)
                             'browArch': P['browArch'] * (1 - 0.6 * sx), 'browThick': P['browThick'] * (1 + 0.35 * sx),
                             'browHeight': P['browHeight'] - 0.06 * sx})
    if hair_colour:
        _recolour(f.M_HAIR, f.ramp(int(P['hairTone'][1:], 16), 0.1))
    if not quiet:
        print(f"creator face: {doc.get('face') or 'settings'} (sex {P.get('sex', 0)}, hair tone {P['hairTone']}, "
              f"skin {P['skinTone']}); not used by the Blender figure: {', '.join(UNUSED)}", flush=True)
    return doc
