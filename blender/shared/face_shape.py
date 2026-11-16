"""The face creator's shape sliders, in Python. The creator (the "Face creator" artifact) sculpts the ranger's head
with every form of face_relief, mouth_relief and jaw_relief scaled by a slider, and exports the sliders as
FACE_SHAPE (multipliers: 1 is the ranger, 0 for the widths and offsets). This module builds those same functions
from a FACE_SHAPE, so a face made in the creator can be checked with the face check (blender/facecheck,
--preset) exactly as the creator shaped it, and a figure can take it on.

shaped(S, FACE_WARP, HEAD_W) gives (face_relief, mouth_relief, jaw_relief, head_w): with S all at the ranger's
values they are the ranger's own functions, form for form.
"""
import math

RANGER = dict(jawWidth=0, jawDef=1, chinProj=1, chinWidth=1, chinHeight=1,
              cheekWidth=0, cheekbones=1, cbHeight=0, apples=1, cheekFull=1,
              noseProj=1, noseWidth=1, noseLen=0, tipBall=1, nostrils=1,
              browRidge=1, sockets=1, eyeSpacing=0.14,
              mouthWidth=1, lipUpper=1, lipLower=1, smile=1)


def gauss(x, z, x0, z0, sx, sz, amp):
    return amp * math.exp(-((x - x0) / sx) ** 2 - ((z - z0) / sz) ** 2)


def _sm(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def shaped(S, FACE_WARP, HEAD_W):
    S = dict(RANGER, **(S or {}))
    es, nw, nl, cb = S['eyeSpacing'] / 0.14, S['noseWidth'], S['noseLen'], S['cbHeight']
    hl = (0.6 + 0.4 * S['lipUpper'], 0.6 + 0.4 * S['lipLower'])

    def mouth_relief(x, z):
        ax = abs(x)
        d = 0.0
        zl = 0.258 + 0.007 * S['smile'] * (ax / 0.082) ** 2
        pinch = math.sqrt(max(0.0, 1 - (ax / 0.081) ** 2))
        bow = 0.006 * math.exp(-((ax - 0.019) / 0.011) ** 2)
        top = 0.258 + (0.038 + bow - 0.012 * (ax / 0.082) ** 2) * hl[0]
        top = zl + (top - zl) * pinch ** 0.8
        t = (z - zl) / max(0.004, top - zl)
        fall = 0.0095 * max(0.4, pinch)
        if 0.0 < t and z < top + fall:
            d += 0.016 * S['lipUpper'] * pinch ** 1.6 * math.sin(min(1.0, t) * math.pi * 0.9) ** 0.8 * (
                1.0 if t <= 1.0 else 0.5 + 0.5 * math.cos(math.pi * (z - top) / fall))
        d += gauss(x, z, 0, 0.266, 0.02, 0.01, 0.005 * S['lipUpper'])
        low = 0.034 * hl[1] * pinch ** 1.4
        d += 0.017 * S['lipLower'] * pinch ** 2 * math.exp(-((z - (zl - 0.56 * low)) / max(0.004, 0.47 * low)) ** 2)
        d -= (0.008 + 0.003 * (ax / 0.082) ** 2) * math.exp(-((z - zl) / 0.006) ** 2) * math.exp(-(x / 0.079) ** 6)
        for sx in (-1, 1):
            d += gauss(x, z, sx * 0.079, 0.262, 0.012, 0.011, -0.002)
            d += gauss(x, z, sx * 0.02, 0.31, 0.009, 0.018, 0.0035)
            for k in range(6):
                u = k / 5
                px, pz = sx * (0.07 + 0.032 * u), 0.355 - 0.1 * u
                d += gauss(x, z, px, pz, 0.014, 0.02, -0.0015 * (1 - 0.5 * u))
                d += gauss(x, z, px + sx * 0.022, pz, 0.018, 0.022, 0.002 * (1 - 0.5 * u))
        d += gauss(x, z, 0, 0.312, 0.012, 0.02, -0.0015)
        d += gauss(x, z, 0, 0.385 + nl, 0.028 * nw, 0.024, 0.006 * S['tipBall'])
        d += gauss(x, z, 0, 0.19, 0.05, 0.014, -0.008)
        return d

    def face_relief(x, z):
        z = FACE_WARP(z)
        d = 0.0
        for s in (-1, 1):
            d += gauss(x, z, s * 0.15 * es, 0.62, 0.12, 0.04, 0.03 * S['browRidge'])
            d += gauss(x, z, s * 0.14 * es, 0.525, 0.08, 0.05, -0.06 * S['sockets'])
            d += gauss(x, z, s * 0.23, 0.415 + cb, 0.09, 0.06, 0.028 * S['cheekbones'])
            d += gauss(x, z, s * 0.145, 0.375 + 0.5 * cb, 0.07, 0.06, 0.04 * S['apples'])
            d += gauss(x, z, s * 0.19, 0.28, 0.11, 0.11, 0.032 * S['cheekFull'])
            d += gauss(x, z, s * 0.18, 0.2, 0.09, 0.09, 0.016 * S['cheekFull'])
            d += gauss(x, z, s * 0.05 * nw, 0.37 + nl, 0.021 * nw, 0.02, 0.018 * S['nostrils'])
            d += gauss(x, z, s * 0.037 * nw, 0.378 + nl, 0.008, 0.02, -0.004 * S['nostrils'])
            d += gauss(x, z, s * 0.235 * es, 0.53, 0.035, 0.085, 0.016)
        zt = 0.36 + nl
        if z > zt:
            t = min(1.0, max(0.0, (0.6 - z) / (0.6 - zt)))
            h = (0.012 + 0.04 * t ** 1.6) * math.exp(-max(0.0, z - 0.58) ** 2 / 0.0008)
        else:
            h = 0.05 * math.exp(-((zt - z) / 0.022) ** 2)
        h *= S['noseProj']
        w = (0.026 + 0.01 * min(1.0, max(0.0, (0.56 - z) / 0.18)) + 0.012 * max(0.0, 1 - abs(z - (0.37 + nl)) / 0.06)) * nw
        d += h * math.exp(-(x / (1.15 * w)) ** 2)
        d += gauss(x, z, 0, 0.635, 0.07, 0.03, 0.012)
        d += gauss(x, z, 0, 0.27, 0.16, 0.1, 0.018)
        d += mouth_relief(x / S['mouthWidth'], z)
        d += gauss(x, z, 0, 0.088 * (0.6 + 0.4 * S['chinHeight']), 0.046 * S['chinWidth'], 0.044 * S['chinHeight'],
                   0.028 * S['chinProj'])
        for s in (-1, 1):
            d += gauss(x, z, s * 0.09 * S['chinWidth'], 0.11, 0.045, 0.06, -0.003)
        return d

    def jaw_relief(y, z):
        zl = 0.17 * min(1.0, max(0.0, (y + 0.34) / 0.4)) ** 1.7

        def ss(t):
            t = min(1.0, max(0.0, t))
            return t * t * (3 - 2 * t)
        return 0.02 * S['jawDef'] * ss((z - zl + 0.03) / 0.1) * ss((0.11 - y) / 0.11) * ss((zl + 0.26 - z) / 0.26)

    def head_w(z):
        jw = _sm(-0.05, 0.08, z) * _sm(0.42, 0.2, z)
        cw = math.exp(-((z - 0.42) / 0.12) ** 2)
        return HEAD_W(z) * (1 + S['jawWidth'] * jw + S['cheekWidth'] * cw)

    return face_relief, mouth_relief, jaw_relief, head_w
