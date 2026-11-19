"""A tileable linen texture: the maps a fine plain-woven linen needs, made from numpy alone (no image files in the
repository), the same every time (a fixed seed), and packed into the .blend so it travels with it.

What linen is, and what of it the maps carry:
  * a plain weave, one thread over and one under, of about 60 to 75 threads to the inch in a shift's fine
    "handkerchief" linen (4 oz or less): THREADS threads cross the tile each way
  * flax yarn is uneven. A thread swells and thins along its length, and here and there carries a slub, a lump two or
    three times its width a centimetre or two long. Slubs lie in both the warp and the weft, so the cloth is streaked
    faintly both ways: that streakiness, not the weave itself, is what reads as linen from a step back
  * yarn from different lots differs a shade, so the streaks come in bands several threads wide as well
  * the fibre is smooth and long, so the cloth has little fuzz and a low dry lustre; the weave is a little open, and a
    single layer lets some light through between the threads

Channels (all 0 to 1, read as data, not colour):
    R  height     the woven surface: each thread a rounded ridge, riding up where it crosses over and down where it
                  crosses under, higher on a slub
    G  tone       how pale the cloth is there (0.5 its mean): thread by thread, band by band, paler on a slub (more
                  fibre, less light through it)
    B  open       the gaps between the threads (1 where neither thread covers)

The tile's x runs across the warp and its y along it (the straight grain)."""
import os
import tempfile

import numpy as np

SIZE = 1024
THREADS = 256           # threads across the tile, each way (4 pixels a thread)


def _smooth1(rng, n, s):
    """Smooth periodic 1D noise, unit standard deviation, correlated over about s samples."""
    f = np.fft.rfft(rng.standard_normal(n))
    out = np.fft.irfft(f * np.exp(-0.5 * (2 * np.pi * np.fft.rfftfreq(n) * s) ** 2), n)
    return out / max(out.std(), 1e-9)


def _yarns(rng, count, n):
    """`count` yarns, each `n` samples long: (width, tone) arrays [yarn, along]. A yarn's width wanders along it and
    jumps on a slub; its tone is its own shade, plus its lot's (shared with its neighbours), paler on a slub."""
    width = np.empty((count, n))
    slub = np.zeros((count, n))
    x = np.arange(n)
    for i in range(count):
        width[i] = 1.0 + 0.16 * _smooth1(rng, n, n / 14.0) + 0.07 * _smooth1(rng, n, n / 60.0)
        for _ in range(rng.poisson(1.1)):                       # (about one slub a yarn in a tile's length)
            c, half = rng.uniform(0, n), rng.uniform(0.012, 0.045) * n
            d = (x - c + n / 2) % n - n / 2
            slub[i] = np.maximum(slub[i], rng.uniform(0.5, 1.0) * np.exp(-(d / half) ** 4))
    width = width * (1.0 + 1.2 * slub)
    lot = _smooth1(rng, count, 3.5)[:, None]                    # (bands several yarns wide)
    own = rng.standard_normal(count)[:, None]
    tone = 0.5 + 0.10 * lot + 0.05 * own + 0.03 * _smooth1(rng, n, n / 8.0)[None, :] * own + 0.22 * slub
    return width, tone


def linen_maps(size=SIZE, threads=THREADS, seed=5):
    rng = np.random.default_rng(seed)
    n, p = size, size // threads
    assert p * threads == size
    w_wid, w_tone = _yarns(rng, threads, n)         # warp: yarn i runs along y at x in [i*p, (i+1)*p)
    f_wid, f_tone = _yarns(rng, threads, n)         # weft: yarn j runs along x
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    i, j = xx // p, yy // p
    u, v = (xx % p + 0.5) / p - 0.5, (yy % p + 0.5) / p - 0.5   # (across the warp yarn, across the weft yarn)
    ww, wt = w_wid[i, yy], w_tone[i, yy]
    fw, ft = f_wid[j, xx], f_tone[j, xx]
    # a yarn's section: a rounded ridge as wide as the yarn is thick there (0.42 of the pitch at its mean)
    w_sec = np.clip(1.0 - (u / (0.42 * ww)) ** 2, 0.0, None) ** 0.5
    f_sec = np.clip(1.0 - (v / (0.42 * fw)) ** 2, 0.0, None) ** 0.5
    # over and under: the warp yarn is up on the cells where i + j is even, and rides smoothly between
    over = np.where((i + j) % 2 == 0, 1.0, -1.0)
    w_h = w_sec * ww * (0.62 + 0.38 * over * np.cos(2 * np.pi * v))
    f_h = f_sec * fw * (0.62 - 0.38 * over * np.cos(2 * np.pi * u))
    height = np.maximum(w_h, f_h)
    tone = np.where(w_h >= f_h, wt, ft)
    open_ = ((w_sec <= 0) & (f_sec <= 0)).astype(float)
    height = height / height.max()
    tone = np.clip(tone * (0.80 + 0.20 * height / max(height.mean(), 1e-9) * 0.5 + 0.10), 0.0, 1.0)   # (dimmer down between the yarns)
    return height, np.clip(tone, 0.0, 1.0), open_


def make_image(name='vest linen', size=SIZE, seed=5):
    """The maps as one packed image in the open .blend (read as data: Non-Color)."""
    import bpy
    from PIL import Image
    px = (np.stack(linen_maps(size, seed=seed), 2) * 255 + 0.5).astype(np.uint8)
    path = os.path.join(tempfile.mkdtemp(), name.replace(' ', '-') + '.png')
    Image.fromarray(px[::-1]).save(path)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = 'Non-Color'
    img.pack()
    return img
