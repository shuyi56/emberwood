"""Face checks without Blender: build a figure's head from its own Python, draw it, and measure it.

A face check used to mean a Blender render through the render queue. This runs the face code itself (the head's
sections, face_relief, mouth_relief, jaw_relief, the lids, eye_relief, and the paintings: iris, sclera,
eyeshadow, lashes, brows, lips) in plain Python with numpy and Pillow, with bpy and bmesh replaced by stubs. It
draws the face from the front, at 3/4, in profile, close on the eyes and the mouth, and at the game's sprite scale,
and it measures the things a face check looks for: whether the eyeball shows through the lids or leaves a gap,
how much iris shows, where the features sit, whether the lip paint stays on the sculpted lips, and symmetry.

Use it from the command line:

    python blender/facecheck/facecheck.py ranger                      # views + measurements of the ranger
    python blender/facecheck/facecheck.py rogue --before origin/main  # the same, side by side with another commit
    python blender/facecheck/facecheck.py ranger --views eyes,lips --out /tmp/check --quality fine

or from Python:

    import facecheck
    fig = facecheck.load('ranger')            # or load(path_to_figure_py)
    face = facecheck.build(fig)               # the head, the eyes and the painting, in model units
    report = facecheck.measure(face)          # a dict of measurements (see MEASURES)
    facecheck.draw(face, 'three', size=640).save('three.png')

A figure is blender/<name>/<name>.py. Its own module is run (with stubs for Blender), so a figure that builds on
blender/base/female.py gets the base figure's face with its own colours, exactly as Blender would.

What it does not show: hair (each figure builds its own), cloth, Cycles lighting, subsurface scattering, and
the sprite pipeline's palette. Check those with the render queue; check the face here.
"""
import argparse
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import types

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ------------------------------------------------------------------------------------------------ loading
class _Any:
    """Absorbs anything Blender-only that a figure touches at import time."""
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, n):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()

    def __iter__(self):
        return iter(())

    def __getitem__(self, k):
        return _Any()

    def __setitem__(self, k, v):
        pass

    def __bool__(self):
        return False


class _Pixels:
    def __init__(self, img):
        self.img = img

    def foreach_set(self, a):
        self.img.data = np.asarray(a, dtype=np.float32).reshape(self.img.h, self.img.w, 4)[::-1].copy()

    def __getitem__(self, k):
        return self.img.data[::-1].ravel()[k]


class _Image(dict):
    """bpy.data.images.new: keeps the painting's pixels (top row first, RGBA 0..1) in .data."""
    def __init__(self, name, w, h, alpha=False, **k):
        super().__init__()
        self.name, self.w, self.h, self.size = name, w, h, (w, h)
        self.data = np.zeros((h, w, 4), dtype=np.float32)
        self.pixels = _Pixels(self)

    def pack(self):
        pass


def _stub_modules():
    bpy = types.ModuleType('bpy')
    bpy.__getattr__ = lambda n: _Any()
    bpy.data = types.SimpleNamespace(images=types.SimpleNamespace(new=_Image), materials=_Any(), meshes=_Any(),
                                     objects=_Any(), actions=_Any(), collections=_Any())
    bpy.context, bpy.ops, bpy.types, bpy.app = _Any(), _Any(), _Any(), _Any()
    mods = {'bpy': bpy}
    for m in ('bmesh', 'bmesh.types', 'bmesh.ops', 'bpy.types', 'mathutils', 'mathutils.bvhtree',
              'mathutils.kdtree', 'mathutils.noise', 'OpenEXR', 'Imath'):
        mod = types.ModuleType(m)
        mod.__getattr__ = lambda n: _Any()
        mods[m] = mod
    return mods


def figure_path(name, root=REPO):
    if name.endswith('.py') and os.path.exists(name):
        return os.path.abspath(name)
    p = os.path.join(root, 'blender', name, name + '.py')
    if not os.path.exists(p):
        have = sorted(d for d in os.listdir(os.path.join(root, 'blender'))
                      if os.path.exists(os.path.join(root, 'blender', d, d + '.py')))
        raise SystemExit(f'no figure {name!r} under {root}/blender (figures: {", ".join(have)})')
    return p


class Figure:
    """A figure's module, run with Blender stubbed out. .g is where its face functions live (the figure's own
    module, or blender/base/female.py for a figure built on the base figure)."""
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        saved = {k: sys.modules.get(k) for k in list(sys.modules)}
        stubs = _stub_modules()
        sys.modules.update(stubs)
        here = os.path.dirname(path)
        base = os.path.join(here, '..', 'base')
        shared = os.path.join(here, '..', 'shared')
        sys.path[:0] = [here, base, shared]
        for k in ('female',):             # (a fresh base figure for each figure, so colours do not leak between them)
            sys.modules.pop(k, None)
        try:
            ns = {'__name__': 'figure_' + self.name, '__file__': path}
            exec(compile(open(path).read(), path, 'exec'), ns)
        finally:
            del sys.path[:3]
            for k in list(sys.modules):
                if k in stubs or k == 'female':
                    if saved.get(k) is None:
                        sys.modules.pop(k, None)
                    else:
                        sys.modules[k] = saved[k]
        self.ns = ns
        if 'face_relief' not in ns:
            raise SystemExit(f'{path} has no face_relief: not a figure this tool can build')
        self.g = ns['face_relief'].__globals__
        self.notes = []

    def __getattr__(self, k):
        if k in self.ns:
            return self.ns[k]
        if k in self.g:
            return self.g[k]
        raise AttributeError(k)

    def const(self, fns, pattern, default, label):
        """A number written inside one of the figure's functions (e.g. the iris radius in build_eye)."""
        for fn in fns.split(','):
            try:
                src = inspect.getsource(self.ns.get(fn) or self.g[fn])
            except (KeyError, OSError, TypeError):
                continue
            m = re.search(pattern, src)
            if m:
                return float(m.group(1))
        self.notes.append(f'{label}: not found in {fns}, using {default}')
        return default


def load(name_or_path, root=REPO):
    return Figure(figure_path(name_or_path, root))


# ------------------------------------------------------------------------------------------------ helpers
def sm(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def sample(img, u, v):
    """Bilinear sample of an (H, W, C) image at u across 0..1 and v up 0..1 (clamped at the edges)."""
    H, W = img.shape[:2]
    fx = np.clip(u * W - 0.5, 0, W - 1.001)
    fy = np.clip((1 - v) * H - 0.5, 0, H - 1.001)
    x0, y0 = fx.astype(int), fy.astype(int)
    tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
    a = img[y0, x0] * (1 - tx) + img[y0, x0 + 1] * tx
    b = img[y0 + 1, x0] * (1 - tx) + img[y0 + 1, x0 + 1] * tx
    return a * (1 - ty) + b * ty


def over(base, rgba):
    a = rgba[..., 3:4]
    return base * (1 - a) + rgba[..., :3] * a


# ------------------------------------------------------------------------------------------------ the head
QUALITY = {'draft': (300, 0.0034), 'normal': (420, 0.0024), 'fine': (600, 0.0017)}
_FIG = None     # (the figure, for the worker processes)


def _ring(args):
    """One ring of the sculpted head (sculpted_head, point by point): its points before the eyes' lids are
    sculpted, and how squarely each faces forward."""
    z, ths = args
    F = _FIG
    hs, fr, jr = F.head_section, F.face_relief, F.jaw_relief
    neck = F.g.get('HEAD_NECK_Z', -0.13)
    out = []
    e = 1e-3
    for th in ths:
        c, s = math.cos(th), math.sin(th)
        x, y = hs(z, c, s)
        xa, ya = hs(z, math.cos(th + e), math.sin(th + e))
        za, zb = min(1.0, z + e), max(neck, z - e)
        xz, yz = hs(za, c, s)
        xb, yb = hs(zb, c, s)
        t1 = (xa - x, ya - y, 0.0)
        t2 = (xz - xb, yz - yb, za - zb)
        n = (t1[1] * t2[2], -t1[0] * t2[2], t1[0] * t2[1] - t1[1] * t2[0])
        nl = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if nl < 1e-12:
            out.append((x, y, z, 0.0))
            continue
        n = (n[0] / nl, n[1] / nl, n[2] / nl)
        if n[0] * x + n[1] * (y - 0.04) < 0:
            n = (-n[0], -n[1], -n[2])
        f = max(0.0, -n[1])
        u = min(1.0, max(0.0, (z - 0.005) / 0.04))
        d = (fr(x, z) * f * u * u * (3 - 2 * u) if f > 0 and u > 0 else 0.0) + jr(y, z)
        out.append((x + n[0] * d, y + n[1] * d, z + n[2] * d, f))
    return out


def head_grid(F, quality='normal', procs=None):
    """The head as a grid (rings up it, segments round it), in model units, before the lids are sculpted."""
    global _FIG
    ns_, dz = QUALITY[quality]
    neck = F.g.get('HEAD_NECK_Z', -0.13)
    zs, z = [], neck
    while z < 1.0:
        zs.append(z)
        z += dz / (0.28 + 0.72 * float(sm(-0.02, 0.12, z) * sm(0.84, 0.72, z)))
    zs.append(1.0)
    tau = -1 + 2 * np.arange(ns_) / ns_
    ths = np.pi * (0.5 * tau + 0.5 * tau ** 3)
    _FIG = F
    jobs = [(z, ths) for z in zs]
    procs = procs or os.cpu_count() or 1
    if procs > 1:
        import multiprocessing as mp
        try:
            with mp.get_context('fork').Pool(procs) as pool:
                rows = pool.map(_ring, jobs, chunksize=4)
        except (ValueError, OSError):
            rows = [_ring(j) for j in jobs]
    else:
        rows = [_ring(j) for j in jobs]
    a = np.array(rows, dtype=np.float64)          # (rings, segs, 4)
    H = F.HEAD_H
    return a[..., :3] * H, a[..., 3], np.array(zs), ths


def grid_normals(P):
    """Normals of a (rings, segs, 3) grid, wrapping round, pointing out."""
    di = np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)
    dj = np.zeros_like(P)
    dj[1:-1] = P[2:] - P[:-2]
    dj[0], dj[-1] = P[1] - P[0], P[-1] - P[-2]
    n = np.cross(di, dj)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    c = P.reshape(-1, 3).mean(0)
    flip = np.sum(n[P.shape[0] // 2] * (P[P.shape[0] // 2] - c), axis=-1).mean() < 0
    return -n if flip else n


def cavity(P, n):
    """How far each point lies below its neighbours (a cheap stand-in for occlusion), 0..1."""
    occ = np.zeros(P.shape[:2])
    for s, k in ((2, 70.0), (7, 22.0)):
        nb = (np.roll(P, s, 1) + np.roll(P, -s, 1) + np.concatenate([P[:1].repeat(s, 0), P[:-s]])
              + np.concatenate([P[s:], P[-1:].repeat(s, 0)])) / 4
        occ += k * np.sum((nb - P) * n, axis=-1)
    return np.clip(occ, 0, 1)


F_SCALE = [8.6]


# ------------------------------------------------------------------------------------------------ the eyes
def eye_setup(F):
    H = F.HEAD_H
    ex = F.const('build_head,build', r'x = s \* ([0-9.]+) \* HEAD_H', 0.14, 'eye x (head heights)')
    ezh = F.const('build_head,build', r'eyes = \[\(feat\[s\]\[0\], ([0-9.]+) \* HEAD_H', 0.525, 'eye height (head heights)')
    ir = F.const('build_eye', r'IR = ([0-9.]+)', 0.31, 'iris radius')
    ilift = F.const('build_eye', r'iz = ez - EYE_LIFT \+ ([0-9.]+)', 0.06, 'iris lift')
    return dict(x=ex * H, zh=ezh, ez=ezh * H + F.EYE_LIFT, IR=ir, iz_off=-F.EYE_LIFT + ilift)


class FrontMap:
    """y of the plain head (before the lids) seen from the front, as a height map over (x, z)."""
    def __init__(self, P, x0, x1, z0, z1, step=0.01, holes=True):
        self.x0, self.z0, self.step = x0, z0, step
        W, Hh = int((x1 - x0) / step) + 1, int((z1 - z0) / step) + 1
        y = np.full((Hh, W), np.inf)
        pts = P.reshape(-1, 3)
        k = (pts[:, 0] > x0 - 0.1) & (pts[:, 0] < x1 + 0.1) & (pts[:, 2] > z0 - 0.1) & (pts[:, 2] < z1 + 0.1) & (pts[:, 1] < 0)
        pts = pts[k]
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                i = np.round((pts[:, 0] - x0) / step).astype(int) + dx
                j = np.round((pts[:, 2] - z0) / step).astype(int) + dz
                ok = (i >= 0) & (i < W) & (j >= 0) & (j < Hh)
                np.minimum.at(y, (j[ok], i[ok]), pts[ok, 1])
        # (fill any holes, and the margins past the head, from their nearest neighbours)
        for _ in range(200 if holes else 0):
            bad = ~np.isfinite(y)
            if not bad.any():
                break
            pad = np.pad(y, 1, constant_values=np.inf)
            nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
            fin = np.isfinite(nb)
            mean = np.where(fin, nb, 0).sum(0) / np.maximum(fin.sum(0), 1)
            y = np.where(bad & fin.any(0), mean, y)
        self.y = y

    def __call__(self, x, z):
        i = np.clip(np.round((np.asarray(x) - self.x0) / self.step).astype(int), 0, self.y.shape[1] - 1)
        j = np.clip(np.round((np.asarray(z) - self.z0) / self.step).astype(int), 0, self.y.shape[0] - 1)
        return self.y[j, i]


def eye_ball(F, E, s, front, seg=240, rings=140):
    """build_eye's white: an almond reshaped to the lids' opening, bent round the face, tucked under the lids at
    the corners. Returns points, its (u, w) texture coordinates and the grid shape."""
    RX, RY, RZ, K, TILT = F.EYE_RX, 0.36, F.EYE_RZ, F.EYE_K, F.EYE_TILT
    x, ez = s * E['x'], E['ez']
    ey = float(front(x, E['zh'] * F.HEAD_H))
    cy = ey + 0.24
    lt = np.vectorize(F.lid_top)
    lb = np.vectorize(F.lid_bottom)
    ph = np.linspace(0, np.pi, rings + 1)[:, None]
    az = np.linspace(0, 2 * np.pi, seg + 1)[None, :]
    vx = RX * np.sin(ph) * np.cos(az)
    vy = RY * np.sin(ph) * np.sin(az)
    vz = RZ * np.cos(ph) + 0 * az
    u = np.clip(vx / RX, -1, 1) * s
    half = RZ * np.sqrt(np.maximum(1e-6, 1 - u * u))
    zn = np.clip(vz / half, -1, 1)
    vz = zn * np.where(zn > 0, lt(u), lb(u)) * K
    vy = np.where(np.abs(u) < 0.999, vy * np.maximum(1e-3, 1 - u * u) ** -0.3, vy)
    a = -s * TILT
    rx, rz = vx * math.cos(a) + vz * math.sin(a), -vx * math.sin(a) + vz * math.cos(a)
    px, py, pz = rx + x, vy + cy, rz + ez
    py = py + front(px, pz) - ey
    uu = (px - x) * s / RX
    dz = pz - ez - uu * RX * TILT
    lim = np.where(dz > 0, lt(uu), lb(uu))
    ov = np.abs(dz) / np.maximum(lim, 1e-3)
    py = py - 0.07 * np.minimum(1, ov) ** 2 + 1.1 * sm(0.97, 1.06, ov) ** 1.5
    py = py + 0.05 * sm(0.8, 1.0, np.abs(px - x) / RX) ** 1.5
    P = np.stack([px, py, pz], -1)
    tex = np.stack([(uu + 1) / 2, (dz / (RZ * K) + 1) / 2], -1)
    return P, tex


# ------------------------------------------------------------------------------------------------ building
def build(F, quality='normal', procs=None):
    """The head, the eyes and their colours. Returns a dict of numpy arrays and the paintings."""
    F_SCALE[0] = F.HEAD_H
    P, front_f, zs, ths = head_grid(F, quality, procs)
    E = eye_setup(F)
    H = F.HEAD_H
    front = FrontMap(P, -E['x'] - 1.8, E['x'] + 1.8, E['ez'] - 1.2, E['ez'] + 1.2)
    # the lids, sculpted into the face (sculpt_eyes): pushed along y by eye_relief round both eyes
    eyes = [(s * E['x'], E['ez'], s) for s in (-1, 1)]
    flat = P.reshape(-1, 3).copy()
    near = (flat[:, 1] < 0) & (np.abs(np.abs(flat[:, 0]) - E['x']) < 1.6) & (np.abs(flat[:, 2] - E['ez']) < 1.3)
    er = F.eye_relief
    for k in np.nonzero(near)[0]:
        flat[k, 1] -= sum(er(flat[k, 0], flat[k, 2], s, ex, ez) for ex, ez, s in eyes)
    P = flat.reshape(P.shape)
    n = grid_normals(P)
    occ = cavity(P, n)
    face = dict(F=F, E=E, P=P, n=n, occ=occ, front_f=front_f, zs=zs, ths=ths, front=front, eyes=eyes)
    paint(face)
    face['balls'] = []
    for s in (-1, 1):
        bp, bt = eye_ball(F, E, s, front)
        face['balls'].append(dict(s=s, P=bp, tex=bt, n=grid_normals_open(bp), col=eye_colour(F, E, s, bp, bt)))
    face['ears'] = ears(F)
    return face


def ears(F):
    """The figure's ears (blender/shared/ear.py), if it has them: set on its own head and coloured as its
    skin_tones colours them (rosier than the face, dark in the bowl and where the flap meets the skull)."""
    eg = F.ns.get('ear_grids') or F.g.get('ear_grids')
    if eg is None:
        return []
    finder, tones = eg.__globals__['side_x_finder'], eg.__globals__['ear_tones']
    skin = np.array(F.SKIN[2], dtype=np.float64) / 255
    flushc, deep = np.array([204, 128, 112]) / 255, np.array([150, 78, 64]) / 255
    out = []
    for e in eg(finder(F.head_section, F.jaw_relief), F.HEAD_H):
        P = e['P']
        di = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
        dj = np.gradient(P, axis=1)
        n = np.cross(dj, di)
        ln = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.where(ln > 1e-9, n / np.maximum(ln, 1e-12), np.array([e['side'], 0, 0.0]))
        NFs = int((e['rear'][0] == 0).sum())
        if np.sum(n[:, 1:NFs, 0] * e['side']) < 0:
            n = -n
        flush, shade = tones(e)
        c = skin + (flushc - skin) * (0.6 * flush)[..., None]
        k = shade[..., None]
        c = c * (1 - 0.45 * k) + deep * 0.18 * k
        out.append(dict(side=e['side'], P=P, n=n, col=np.clip(c, 0, 1), gloss=np.zeros(P.shape[:2]) + 0.15))
    return out


def grid_normals_open(P):
    di = np.gradient(P, axis=1)
    dj = np.gradient(P, axis=0)
    n = np.cross(di, dj)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    c = P.reshape(-1, 3).mean(0)
    if np.sum(n * (P - c), axis=-1).mean() < 0:
        n = -n
    return n


def eye_colour(F, E, s, P, tex):
    """The white (sclera_image) with the iris (iris_image) laid on it, round the iris's centre."""
    scl = F.sclera_image().data[..., :3]
    col = sample(scl, tex[..., 0], tex[..., 1])
    iris = F.iris_image().data[..., :3]
    x, iz, IR = s * E['x'], E['ez'] + E['iz_off'], E['IR']
    dx, dz = (P[..., 0] - x) * s / IR, (P[..., 2] - iz) / IR
    r = np.hypot(dx, dz)
    ic = sample(iris, (dx + 1) / 2, (dz + 1) / 2)
    a = np.clip((1 - r) * 40, 0, 1)[..., None]
    return col * (1 - a) + ic * a


def paint(face):
    """The skin's colour (skin_tones, the head's part) with the paintings laid over it by their own patches:
    eyeshadow and lashes round each eye, the brows, and the lips."""
    F, E, P, n = face['F'], face['E'], face['P'], face['n']
    H = F.HEAD_H
    u, w = P[..., 0] / H, P[..., 2] / H
    fr = np.clip(-n[..., 1] * 1.5, 0, 1)

    def g(u0, v0, su, sv):
        return np.exp(-((u - u0) / su) ** 2 - ((w - v0) / sv) ** 2)
    skin = np.array(F.SKIN[2], dtype=np.float64) / 255
    flushc, cool = np.array([204, 128, 112]) / 255, np.array([164, 138, 126]) / 255
    flush = fr * (0.95 * (g(-0.2, 0.365, 0.105, 0.06) + g(0.2, 0.365, 0.105, 0.06)) + 0.8 * g(0, 0.35, 0.05, 0.045)
                  + 0.35 * g(0, 0.085, 0.07, 0.05) + 0.3 * g(0, 0.66, 0.12, 0.05))
    cold = fr * 0.8 * (g(-0.14, 0.47, 0.065, 0.025) + g(0.14, 0.47, 0.065, 0.025))
    cold = cold + np.clip(-n[..., 2] - 0.25, 0, 1) * np.clip((0.3 - w) / 0.2, 0, 1) * 1.1 + np.clip((0.02 - w) / 0.1, 0, 1) * 0.7
    flush, cold = np.clip(flush, 0, 1), np.clip(cold, 0, 1)
    c = skin + (flushc - skin) * (0.6 * flush)[..., None]
    c = c + (cool - c) * (0.5 * cold)[..., None]
    k = np.clip(face['occ'] * 1.3, 0, 1)[..., None]
    c = c * (1 - 0.45 * k) + np.array([150, 78, 64]) / 255 * 0.18 * k
    face['gloss'] = np.clip(fr * (g(0, 0.43, 0.04, 0.15) + 0.7 * g(0, 0.085, 0.06, 0.05) + 0.5 * g(0, 0.66, 0.15, 0.06)), 0, 1)
    # the patches round each eye: laid on straight from the front, thinned where the face turns away
    fwd = sm(0.4, 0.7, -n[..., 1])
    RX, TILT = F.EYE_RX, F.EYE_TILT
    shadow, lashes, brows = F.eyeshadow_image().data, F.lash_image().data, F.brow_image().data
    for ex, ez, s in face['eyes']:
        uu = (P[..., 0] - ex) * s / RX
        dz = P[..., 2] - ez - uu * RX * TILT
        front = P[..., 1] < 0
        for img, (U0, U1, V0, V1) in ((shadow, (F.SH_U0, F.SH_U1, F.SH_V0, F.SH_V1)), (brows, (F.BR_U0, F.BR_U1, F.BR_V0, F.BR_V1)),
                                      (lashes, (F.SH_U0, F.SH_U1, F.SH_V0, F.SH_V1))):
            inside = front & (uu > U0) & (uu < U1) & (dz > V0) & (dz < V1)
            if not inside.any():
                continue
            px = sample(img, (uu[inside] - U0) / (U1 - U0), (dz[inside] - V0) / (V1 - V0)).copy()
            px[:, 3] *= fwd[inside]
            c[inside] = over(c[inside], px)
    # the lips (lip_painting), over the mouth in real heights
    lp = F.lip_painting()
    inside = (P[..., 1] < 0) & (u > F.LIP_X0) & (u < F.LIP_X1) & (w > F.LIP_Z0) & (w < F.LIP_Z1)
    face['lip_alpha'] = np.zeros(u.shape)
    if inside.any():
        px = sample(lp, (u[inside] - F.LIP_X0) / (F.LIP_X1 - F.LIP_X0), (w[inside] - F.LIP_Z0) / (F.LIP_Z1 - F.LIP_Z0))
        c[inside] = over(c[inside], px)
        face['lip_alpha'][inside] = px[:, 3]
    face['col'] = np.clip(c, 0, 1)


# ------------------------------------------------------------------------------------------------ drawing
VIEWS = {     # azimuth (degrees, + turns her to her left), elevation, distance, look-at height (head heights)
    'front': (0, 4, 34, 0.5), 'three': (35, 6, 34, 0.5), 'profile': (88, 3, 34, 0.5),
    'eyes': (12, 3, 15, 0.53), 'lips': (16, -2, 14, 0.27), 'game': (20, 38, 34, 0.5),
    'ear': (-90, 3, 15, 0.49, 0.09), 'ear_back': (160, 6, 20, 0.5, 0.1),     # (and the look-at depth, if not 0)
}
LIGHT = np.array([-0.45, -0.75, 0.55])     # the key, from her right, in front and above (model axes)


def _camera(view, H):
    az, el, dist, ty = VIEWS[view][:4]
    tdepth = VIEWS[view][4] if len(VIEWS[view]) > 4 else 0.0
    az, el = math.radians(az), math.radians(el)
    tgt = np.array([0, tdepth * H, ty * H])
    eye = tgt + dist * np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, fwd, right, up


def _shade(P, n, col, eye, gloss, rough):
    L = LIGHT / np.linalg.norm(LIGHT)
    V = eye - P
    V /= np.linalg.norm(V, axis=-1, keepdims=True)
    lam = np.clip(np.sum(n * L, -1), 0, 1)
    wrap = np.clip((np.sum(n * L, -1) + 0.35) / 1.35, 0, 1)        # (a soft wrap, as skin's light does)
    rim = np.clip(np.sum(n * np.array([0.7, 0.5, 0.3]), -1), 0, 1) * 0.25
    Hh = L + V
    Hh /= np.linalg.norm(Hh, axis=-1, keepdims=True)
    spec = np.clip(np.sum(n * Hh, -1), 0, 1) ** (8 + 120 * (1 - rough)) * (0.08 + 0.4 * gloss)
    light = 0.28 + 0.62 * (0.5 * lam + 0.5 * wrap) + rim
    warm = np.array([1.0, 0.93, 0.88])
    return np.clip(col * light[..., None] * warm + spec[..., None], 0, 1)


def draw(face, view='front', size=640, ss=2):
    """One view of the face as a PIL image (points splatted with a depth test, at ss times the size, then
    reduced)."""
    F = face['F']
    H = F.HEAD_H
    eye, fwd, right, up = _camera(view, H)
    W = size * ss
    fov = math.radians(20)
    f = W / 2 / math.tan(fov / 2)
    parts = [(face['P'], face['n'], face['col'], face['gloss'], 0.55)]
    for b in face['balls']:
        parts.append((b['P'], b['n'], b['col'], np.ones(b['P'].shape[:2]) * 0.8, 0.1))
    for e in face.get('ears', []):
        parts.append((e['P'], e['n'], e['col'], e['gloss'], 0.6))
    xs, ys, ds, cs, rs = [], [], [], [], []
    for P, n, col, gloss, rough in parts:
        p = P.reshape(-1, 3)
        nn = n.reshape(-1, 3)
        rel = p - eye
        d = rel @ fwd
        x = rel @ right / d * f + W / 2
        y = -(rel @ up) / d * f + W / 2
        # each point covers about its grid spacing on screen
        sp = np.maximum(np.linalg.norm(np.gradient(P, axis=0), axis=-1), np.linalg.norm(np.gradient(P, axis=1), axis=-1)).ravel()
        r = np.clip(sp / d * f * 0.75, 0.6, 14)
        facing = np.sum(nn * (eye - p), -1) > 0
        k = facing & (d > 0)
        c = _shade(p[k], nn[k], col.reshape(-1, 3)[k], eye, gloss.ravel()[k], rough)
        xs.append(x[k]); ys.append(y[k]); ds.append(d[k]); cs.append(c); rs.append(r[k])
    x, y, d, c, r = (np.concatenate(a) for a in (xs, ys, ds, cs, rs))
    R = int(math.ceil(r.max()))
    zb = np.full(W * W, np.inf)
    idx_all, d_all, c_all = [], [], []
    for oy in range(-R, R + 1):
        for ox in range(-R, R + 1):
            k = (ox * ox + oy * oy) <= r * r + 0.25
            px = np.round(x[k] + ox).astype(int)
            py = np.round(y[k] + oy).astype(int)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < W)
            idx_all.append(py[ok] * W + px[ok])
            d_all.append(d[k][ok])
            c_all.append(c[k][ok])
    idx, dd, cc = np.concatenate(idx_all), np.concatenate(d_all), np.concatenate(c_all)
    order = np.lexsort((dd, idx))
    idx, cc = idx[order], cc[order]
    first = np.ones(len(idx), bool)
    first[1:] = idx[1:] != idx[:-1]
    img = np.zeros((W * W, 3)) + np.array([0.17, 0.19, 0.18])
    img[idx[first]] = cc[first]
    im = Image.fromarray((img.reshape(W, W, 3) ** (1 / 1.0) * 255).astype(np.uint8))
    if view == 'game':        # the game's camera: the head about 17 pixels tall, shown blown up
        small = im.resize((28, 28), Image.LANCZOS)
        return small.resize((size, size), Image.NEAREST)
    return im.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------------------------------------ measuring
MEASURES = """
eye.poke_through_px   white of the eye showing outside the lids' opening, in 0.01-unit pixels (her left eye; the face is symmetric; 0 is right)
eye.gap_px            pixels well inside the opening where the skin stands in front of the ball (a lid dropping over the white)
eye.open_w, open_h    the opening's width and its height at the upper lid's peak plus the lower lid's dip (units)
eye.iris_showing      the share of the iris's disc that shows between the lids (0..1)
eye.iris_top_hidden   how far the upper lid comes down over the iris (units)
eye.sprite_px         the opening's width and height at the game's scale (head 17 px tall)
features.*_z          heights in head heights (chin 0, crown 1) of the brow, the eye line, the nose tip, the mouth line, the chin
features.*_fwd        how far each stands forward of the eye line's skin (units)
widths.*              the face's half-width (units) at the cheekbones, the mouth and the jaw
lips.paint_on_sculpt  share of the painted lip (alpha > 0.5) that lies on the sculpted lips (1 is right)
lips.sculpt_painted   share of the sculpted lips that the paint covers
brows.clearance       gap between the brow's lower edge and the upper lid's crease at the eye's centre (units)
symmetry.max          the largest left/right difference in the face's relief (units; 0 is symmetric)
"""


def measure(face):
    F, E, P = face['F'], face['E'], face['P']
    H = F.HEAD_H
    RX, TILT = F.EYE_RX, F.EYE_TILT
    out = {'figure': F.name}
    # ---- the eyes, seen square from the front: which is nearer, the skin or the ball
    step = 0.01
    ex, ez = E['x'], E['ez']
    x0, x1, z0, z1 = ex - 1.3, ex + 1.3, ez - 0.7, ez + 0.8
    skin = FrontMap(P, x0, x1, z0, z1, step, holes=False)
    ball = [b for b in face['balls'] if b['s'] == 1][0]
    bm = FrontMap(ball['P'], x0, x1, z0, z1, step, holes=False)
    xs = x0 + np.arange(skin.y.shape[1]) * step
    zz = z0 + np.arange(skin.y.shape[0]) * step
    X, Z = np.meshgrid(xs, zz)
    u = (X - ex) / RX
    dz = Z - ez - u * RX * TILT
    lt, lb = np.vectorize(F.lid_top), np.vectorize(F.lid_bottom)
    T, B = lt(np.clip(u, -1, 1)), lb(np.clip(u, -1, 1))
    opening = (np.abs(u) < 1) & (dz < T) & (dz > -B)
    shows = bm.y < skin.y - 0.002
    margin = (np.abs(u) < 1.03) & (dz < T + 0.03) & (dz > -B - 0.03)
    deep = (np.abs(u) < 0.85) & (dz < T - 0.06) & (dz > -B + 0.06)     # (the lids' margins lap over the ball's edge by design)
    iz = ez + E['iz_off']
    disc = np.hypot(X - ex, Z - iz) < E['IR']
    peak_u = float(F.EYE_TOP_PEAK[0]) if 'EYE_TOP_PEAK' in F.g else -0.18
    out['eye'] = {
        'poke_through_px': int(np.sum(shows & ~margin)),
        'gap_px': int(np.sum(deep & ~shows)),
        'open_w': round(2 * RX, 3),
        'open_h': round(float(F.lid_top(peak_u) + max(F.lid_bottom(v) for v in np.linspace(-1, 1, 81))), 3),
        'iris_showing': round(float(np.sum(disc & opening & shows) / max(1, np.sum(disc))), 3),
        'iris_top_hidden': round(float(max(0.0, (iz + E['IR']) - (ez + F.lid_top(0.0)))), 3),
    }
    px = 17 / H
    out['eye']['sprite_px'] = [round(out['eye']['open_w'] * px, 2), round(out['eye']['open_h'] * px, 2)]
    # ---- the profile down the middle of the face, and the features on it
    mid = np.argmin(np.abs(face['ths']))
    prof = P[:, mid]
    zh = prof[:, 2] / H
    yf = prof[:, 1]
    eye_skin = float(skin(ex, ez))

    def fwd_at(lo, hi, pick=np.argmin):
        k = np.nonzero((zh > lo) & (zh < hi))[0]
        i = k[pick(yf[k])]
        return round(float(zh[i]), 4), round(float(eye_skin - yf[i]), 3)
    nz, nf = fwd_at(0.3, 0.45)
    # the chin: the most forward point of the ball of the chin (a local bulge below the crease under the lip)
    k = np.nonzero((zh > 0.02) & (zh < 0.18))[0]
    loc = [i for i in k[1:-1] if yf[i] <= yf[i - 1] and yf[i] <= yf[i + 1]]
    if loc:
        i = loc[0]      # (the lowest bulge: the ball of the chin, not the roll under the lower lip)
        cz, cf = round(float(zh[i]), 4), round(float(eye_skin - yf[i]), 3)
    else:
        cz, cf = None, None
    mk = np.nonzero((zh > 0.2) & (zh < 0.3))[0]
    mouth_line = mk[np.argmax(yf[mk])]
    bz = F.brow_line(0.0)[0] if 'brow_line' in F.g else None
    out['features'] = {'nose_tip_z': nz, 'nose_tip_fwd': nf, 'chin_z': cz, 'chin_fwd': cf,
                       'mouth_line_z': round(float(zh[mouth_line]), 4), 'eye_line_z': round(float(ez / H), 4),
                       'brow_mid_z': round(float((ez - F.EYE_LIFT + bz) / H), 4) if bz is not None else None}

    def half_width(zh0):
        j = int(np.argmin(np.abs(face['zs'] - zh0)))
        return round(float(np.max(np.abs(P[j, :, 0]))), 3)
    out['widths'] = {'cheekbones': half_width(0.42), 'mouth': half_width(0.25), 'jaw': half_width(0.15)}
    # ---- lips: paint against sculpt (in the warped heights mouth_relief is laid out in)
    xs = np.linspace(F.LIP_X0, F.LIP_X1, 240)
    zr = np.linspace(F.LIP_Z1, F.LIP_Z0, 140)
    lp = F.lip_painting()
    a = sample(lp, (xs[None, :] - F.LIP_X0) / (F.LIP_X1 - F.LIP_X0) + 0 * zr[:, None], (zr[:, None] - F.LIP_Z0) / (F.LIP_Z1 - F.LIP_Z0) + 0 * xs[None, :])[..., 3]
    fw = np.array([F.FACE_WARP(z) for z in zr]) if 'FACE_WARP' in F.g else zr
    rel = np.array([[F.mouth_relief(x, w) for x in xs] for w in fw])
    paint_m, sculpt_m = a > 0.5, rel > 0.006
    out['lips'] = {'paint_on_sculpt': round(float(np.sum(paint_m & sculpt_m) / max(1, paint_m.sum())), 3),
                   'sculpt_painted': round(float(np.sum(paint_m & sculpt_m) / max(1, sculpt_m.sum())), 3)}
    # ---- brows: the lower edge over the crease
    if bz is not None:
        zc, r = F.brow_line(0.0)
        crease = F.lid_top(0.0) + 0.13 + 0.035
        out['brows'] = {'clearance': round(float(zc - EYE_LIFT_OF(F) - r - crease), 3)}
    # ---- symmetry of the relief
    xs = np.linspace(0.02, 0.35, 34)
    zs = np.linspace(0.05, 0.7, 66)
    d = max(abs(F.face_relief(x, z) - F.face_relief(-x, z)) for x in xs for z in zs)
    out['symmetry'] = {'max': round(float(d * H), 4)}
    out['notes'] = F.notes
    return out


def EYE_LIFT_OF(F):
    return F.EYE_LIFT


# ------------------------------------------------------------------------------------------------ before / after
def checkout(ref, root=REPO):
    """The blender/ folder at a git ref, unpacked in a temporary folder (for a before)."""
    tmp = tempfile.mkdtemp(prefix='facecheck-')
    arch = subprocess.run(['git', '-C', root, 'archive', ref, 'blender'], capture_output=True)
    if arch.returncode:
        raise SystemExit(f'git archive {ref} failed: {arch.stderr.decode().strip()}')
    subprocess.run(['tar', '-x', '-C', tmp], input=arch.stdout, check=True)
    return tmp


def diff(a, b, path=''):
    """The measurements that changed between two reports."""
    out = []
    for k in b:
        if k in ('notes', 'figure'):
            continue
        va, vb = a.get(k) if isinstance(a, dict) else None, b[k]
        if isinstance(vb, dict):
            out += diff(va or {}, vb, path + k + '.')
        elif va != vb:
            out.append(f'{path}{k}: {va} -> {vb}')
    return out


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 20], fill=(20, 22, 21))
    d.text((5, 4), text, fill=(235, 238, 236))
    return im


def sheet(rows, size):
    W = max(len(r) for _, r in rows) * size
    out = Image.new('RGB', (W, len(rows) * size), (30, 32, 31))
    for j, (name, ims) in enumerate(rows):
        for i, (v, im) in enumerate(ims):
            out.paste(label(im.copy(), f'{name} {v}'), (i * size, j * size))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check a figure\'s face without Blender: views and measurements.')
    ap.add_argument('figure', help='a figure name (blender/<name>/<name>.py) or a path to its .py')
    ap.add_argument('--views', default='front,three,profile,eyes,lips,ear,game', help='any of ' + ','.join(VIEWS))
    ap.add_argument('--size', type=int, default=480)
    ap.add_argument('--quality', default='normal', choices=list(QUALITY))
    ap.add_argument('--before', help='a git ref to compare with (e.g. origin/main): drawn and measured the same way')
    ap.add_argument('--out', default=None, help='folder for the images and report (default: /tmp/facecheck-<figure>)')
    ap.add_argument('--procs', type=int, default=None)
    ap.add_argument('--no-images', action='store_true', help='measurements only')
    a = ap.parse_args(argv)
    out = a.out or os.path.join(tempfile.gettempdir(), 'facecheck-' + os.path.basename(a.figure).replace('.py', ''))
    os.makedirs(out, exist_ok=True)
    views = [v for v in a.views.split(',') if v]
    runs = []
    if a.before:
        root = checkout(a.before)
        runs.append(('before', load(a.figure if a.figure.endswith('.py') else a.figure, root)))
    runs.append(('now' if a.before else '', load(a.figure)))
    reports, rows = {}, []
    for tag, F in runs:
        face = build(F, a.quality, a.procs)
        rep = measure(face)
        reports[tag or 'now'] = rep
        if not a.no_images:
            ims = [(v, draw(face, v, a.size)) for v in views]
            for v, im in ims:
                im.save(os.path.join(out, f'{(tag + "-") if tag else ""}{v}.png'))
            rows.append((tag or F.name, ims))
    if rows:
        sheet(rows, a.size).save(os.path.join(out, 'sheet.png'))
    result = {'figure': runs[-1][1].name, 'reports': reports}
    if a.before:
        result['changed'] = diff(reports['before'], reports['now'])
    with open(os.path.join(out, 'report.json'), 'w') as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(reports['now'], indent=1))
    if a.before:
        print('changed since ' + a.before + ':\n  ' + ('\n  '.join(result['changed']) or 'nothing measured changed'))
    print('images and report in', out)


if __name__ == '__main__':
    main()
