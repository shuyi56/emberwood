"""Tileable linen textures: the maps a fine plain-woven linen needs, made from numpy alone (no image files in the
repository), the same every time (fixed seeds), and packed into the .blend so they travel with it.

What linen is, and what the maps carry of it:

  * A shift's linen is a balanced plain weave, one thread over and one under, of 60 to 75 threads to the inch, from
    wet-spun line flax: a smooth, strong, lustrous yarn with few hairs and only an occasional slub.
  * Nothing in it is ruler-straight. Flax has no stretch, so the loom's unevenness stays in the cloth: the warp ends
    sit in pairs and drift (the reed's dents), the weft is beaten closer here and wider there and lies in a slight
    bow, and every thread wanders a little from its place.
  * A yarn is never one thickness or one shade from end to end. It swells and thins over a few centimetres, so what
    shows from a step back is short faint streaks that fade in and out, mostly along the weft (the softer, fuller
    yarn), never lines that run across the whole cloth.
  * Washed and mangled, the yarns are pressed flat: ribbons lying shoulder to shoulder that close the cloth and give
    it a smooth, faintly polished face. It is not a net of round cords with holes between.

Two maps, at two scales, because the two things are seen at different distances:

  weave (WEAVE_THREADS threads across the tile each way), read as data:
      R  height  the woven surface: flattened yarns riding over and under, the weft riding higher than the warp
      G  tone    the yarn's own slight shade there (0.5 the mean)
      B  rub     how much of a pressed, polished flat the surface is there (the crowns of the yarns)
  streaks (a larger tile, not a multiple of the weave's, so the two never repeat together), read as data:
      R  tone    the cloth's faint streakiness from a step back (0.5 the mean): weft-wise dashes, fainter warp-wise
                 ones, and a slow cloudiness
      G  body    how full the yarn is there (0.5 the mean): the same streaks as relief

Both tiles' x runs across the warp and y along it (the straight grain)."""
import os
import tempfile

import numpy as np

WEAVE_SIZE = 2048
WEAVE_THREADS = 256         # 8 pixels a thread
STREAK_SIZE = 1024
STREAK_SPAN = 424.0         # the streak tile's width, in threads (so it is 1.66 weave tiles: they do not line up)


def _smooth2(rng, shape, sy, sx):
    """Smooth tileable noise, unit standard deviation, correlated over about sy by sx pixels."""
    ny, nx = shape
    w = np.fft.rfft2(rng.standard_normal(shape))
    fy, fx = np.fft.fftfreq(ny)[:, None], np.fft.rfftfreq(nx)[None, :]
    out = np.fft.irfft2(w * np.exp(-0.5 * (2 * np.pi) ** 2 * ((fx * sx) ** 2 + (fy * sy) ** 2)), shape)
    return out / max(out.std(), 1e-9)


def _yarns(rng, count, n, px, slubs):
    """`count` yarns, each `n` samples long (px samples a thread): their width and shade along their length
    [yarn, along]. The width swells and thins over a few centimetres (tens of threads), with a rare slub; the shade
    drifts along the yarn, and differs a little from its neighbours'."""
    along = lambda s: _smooth2(rng, (count, n), 0.0, s * px)       # noqa: E731  (each yarn its own, along its length)
    width = 1.0 + 0.085 * along(45.0) + 0.045 * along(12.0)
    slub = np.zeros((count, n))
    x = np.arange(n)
    for i in np.nonzero(rng.random(count) < slubs)[0]:
        c, half = rng.uniform(0, n), rng.uniform(6.0, 16.0) * px                # (3 to 6 mm long)
        d = (x - c + n / 2) % n - n / 2
        slub[i] = rng.uniform(0.5, 1.0) * np.exp(-(d / half) ** 2)
    tone = 0.035 * along(70.0) + 0.02 * along(18.0) + 0.008 * rng.standard_normal((count, 1)) + 0.08 * slub
    return width * (1.0 + 0.55 * slub), tone


def weave_maps(size=WEAVE_SIZE, threads=WEAVE_THREADS, seed=5):
    rng = np.random.default_rng(seed)
    n, p = size, size // threads
    assert p * threads == size
    yy, xx = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing='ij')
    # each thread's place: off the even grid, thread by thread and along its own length. The warp ends in pairs (the
    # reed), drifting a little; the weft beaten unevenly and lying in a slow bow
    pair = 0.10 * p * np.where((np.arange(threads) % 2) == 0, 1.0, -1.0)
    xw = xx + pair[(xx // p).astype(int) % threads] \
        + 0.16 * p * _smooth2(rng, (n, n), 60 * p, 0.8 * p) + 0.45 * p * _smooth2(rng, (n, n), 90 * p, 14 * p)
    yw = yy + 0.24 * p * _smooth2(rng, (n, n), 0.8 * p, 50 * p) + 0.80 * p * _smooth2(rng, (n, n), 10 * p, 80 * p)
    i, j = np.floor(xw / p).astype(int) % threads, np.floor(yw / p).astype(int) % threads
    u, v = (xw / p) % 1.0 - 0.5, (yw / p) % 1.0 - 0.5          # (across the warp yarn, across the weft yarn)
    yi, xi = yy.astype(int), xx.astype(int)
    w_wid, w_tone = _yarns(rng, threads, n, p, slubs=0.06)
    f_wid, f_tone = _yarns(rng, threads, n, p, slubs=0.14)     # (the weft: the softer, less even yarn)
    ww, wt = 0.90 * w_wid[i, yi], w_tone[i, yi]
    fw, ft = 1.00 * f_wid[j, xi], f_tone[j, xi]
    # a yarn pressed flat: a ribbon with a broad level crown and rounded shoulders
    w_sec = np.clip(1.0 - np.abs(u / (0.5 * ww)) ** 3.0, 0.0, None) ** 0.45
    f_sec = np.clip(1.0 - np.abs(v / (0.5 * fw)) ** 3.0, 0.0, None) ** 0.45
    # over and under: where the warp is up (i + j even) the weft is down; the weft, the softer yarn, bends more
    over = np.where((i + j) % 2 == 0, 1.0, -1.0)
    w_h = w_sec * (0.70 + 0.14 * (ww - 0.9)) * (0.80 + 0.20 * over * np.cos(2 * np.pi * v))
    f_h = f_sec * (0.74 + 0.14 * (fw - 1.0)) * (0.72 - 0.28 * over * np.cos(2 * np.pi * u))
    height = np.maximum(w_h, f_h)
    top = w_h >= f_h
    tone = 0.5 + np.where(top, wt, ft) - 0.06 * np.clip(0.55 - height, 0.0, None) / 0.55      # (a shade dimmer between yarns)
    height = height / height.max()
    rub = np.clip((height - 0.72) / 0.2, 0.0, 1.0)
    return height, np.clip(tone, 0.0, 1.0), rub


def streak_maps(size=STREAK_SIZE, span=STREAK_SPAN, seed=9):
    rng = np.random.default_rng(seed)
    n, t = size, size / span                # (t: pixels a thread)
    sh = (n, n)
    # weft-wise dashes: a thread or a few threads wide (y), a few centimetres long (x), of several lengths
    weft = 0.55 * _smooth2(rng, sh, 1.0 * t, 22 * t) + 0.35 * _smooth2(rng, sh, 2.2 * t, 40 * t) + 0.30 * _smooth2(rng, sh, 0.8 * t, 10 * t)
    warp = 0.55 * _smooth2(rng, sh, 18 * t, 1.0 * t) + 0.35 * _smooth2(rng, sh, 35 * t, 2.0 * t)
    cloud = _smooth2(rng, sh, 60 * t, 60 * t)
    # (a streak is a yarn that is fuller than its neighbours for a while: only the strongest show, the rest is even)
    keep = lambda a: np.sign(a) * np.clip(np.abs(a) - 0.9, 0.0, None)        # noqa: E731
    lines = 1.0 * keep(weft / weft.std()) + 0.55 * keep(warp / warp.std())
    tone = 0.5 + 0.11 * lines + 0.05 * cloud
    body = 0.5 + 0.16 * lines
    return np.clip(tone, 0.0, 1.0), np.clip(body, 0.0, 1.0), np.zeros(sh)


def _image(name, maps):
    import bpy
    from PIL import Image
    px = (np.stack(maps, 2) * 255 + 0.5).astype(np.uint8)
    path = os.path.join(tempfile.mkdtemp(), name.replace(' ', '-') + '.png')
    Image.fromarray(px[::-1]).save(path)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = 'Non-Color'
    img.pack()
    return img


def make_images(weave='linen weave', streaks='linen streaks'):
    """The two maps as packed images in the open .blend (read as data: Non-Color). Returns (weave, streaks)."""
    import bpy
    return (bpy.data.images.get(weave) or _image(weave, weave_maps()),
            bpy.data.images.get(streaks) or _image(streaks, streak_maps()))
