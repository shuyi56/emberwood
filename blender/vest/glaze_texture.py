"""A tileable fired-glaze texture: the maps a vitreous glaze over metal needs, made from numpy alone (no image files in
the repository), the same every time (a fixed seed), and packed into the .blend so it travels with it.

Channels (all 0 to 1, read as data, not colour):
    R  thickness  how thick the glass lies: broad pools and thin places, drawn out a little downward where it ran in
                  the kiln (the texture's -y is down), with a fine ripple in it
    G  crazing    the crackle: a net of hairline cracks, large cells broken again into small ones, as glaze crazes when
                  it cools faster than the metal under it (1 on a crack)
    B  pinholes   small pits where a bubble burst through the glass and bare metal shows (1 in a pit)"""
import os
import tempfile

import numpy as np

SIZE = 1024


def _smooth2(rng, n, sx, sy=None):
    """Smooth tileable noise, unit standard deviation, correlated over about sx by sy pixels."""
    sy = sx if sy is None else sy
    w = np.fft.rfft2(rng.standard_normal((n, n)))
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None, :]
    out = np.fft.irfft2(w * np.exp(-0.5 * (2 * np.pi) ** 2 * ((fx * sx) ** 2 + (fy * sy) ** 2)), (n, n))
    return out / max(out.std(), 1e-9)


def _crackle(rng, n, cells, warp=None):
    """A tileable net of cracks: the distance from each pixel to the border between its two nearest of `cells`
    scattered points (0 on a border). Each point only touches a window round it, so this is quick."""
    f1 = np.full((n, n), 1e9)
    f2 = np.full((n, n), 1e9)
    half = int(3.2 * n / np.sqrt(cells))
    off = np.arange(-half, half + 1)
    dy, dx = np.meshgrid(off, off, indexing='ij')
    gy, gx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    wy, wx = (gy, gx) if warp is None else (gy + warp[0], gx + warp[1])
    for py, px in rng.uniform(0, n, (cells, 2)):
        iy, ix = (int(py) + dy) % n, (int(px) + dx) % n
        # (the pixel's own, warped, place measured from the point, the short way round the tile)
        ry = (wy[iy, ix] - py + n / 2) % n - n / 2
        rx = (wx[iy, ix] - px + n / 2) % n - n / 2
        d = np.hypot(ry, rx)
        a, b = f1[iy, ix], f2[iy, ix]
        f2[iy, ix] = np.where(d < a, a, np.minimum(b, d))
        f1[iy, ix] = np.minimum(a, d)
    d = f2 - f1
    # (f2 - f1 grows at a different rate from border to border; divided by its own slope it is the true distance to
    # the crack, so every crack is drawn the same width)
    gy_, gx_ = np.gradient(np.pad(d, 1, mode='wrap'))
    slope = np.hypot(gy_, gx_)[1:-1, 1:-1]
    return d / np.maximum(slope, 0.35)


def glaze_maps(size=SIZE, seed=11):
    rng = np.random.default_rng(seed)
    n = size
    # thickness: pools, a downward run, a ripple
    t = 0.55 * _smooth2(rng, n, n / 9.0) + 0.35 * _smooth2(rng, n, n / 40.0, n / 12.0) + 0.10 * _smooth2(rng, n, n / 110.0)
    t = 0.5 + 0.5 * np.tanh(0.9 * t / t.std())
    # crazing: cracks wander (the cells' borders are bent by a slow warp), big cells then small ones inside them; the
    # small cracks are fainter and die out where the glass is thin (thin glaze does not craze)
    warp = (0.012 * n * _smooth2(rng, n, n / 30.0), 0.012 * n * _smooth2(rng, n, n / 30.0))
    big = _crackle(rng, n, 70, warp)
    small = _crackle(rng, n, 420, warp)
    line = lambda d, w: np.clip(1.0 - d / w, 0.0, 1.0) ** 1.5      # noqa: E731
    craze = np.maximum(line(big, 2.2), 0.6 * line(small, 1.5) * np.clip((t - 0.3) / 0.3, 0.0, 1.0))
    # pinholes: a few, round, in clusters
    seeds = (rng.random((n, n)) > 1 - 55.0 / (n * n) * (1.0 + 2.5 * (_smooth2(rng, n, n / 14.0) > 0.6))).astype(float)
    k = np.fft.rfft2(seeds)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None, :]
    blob = np.fft.irfft2(k * np.exp(-0.5 * (2 * np.pi * 2.6) ** 2 * (fx * fx + fy * fy)), (n, n))
    pin = np.clip(blob / 0.012, 0.0, 1.0)
    return t, np.clip(craze, 0.0, 1.0), pin


def make_image(name='vest glaze', size=SIZE, seed=11):
    """The maps as one packed image in the open .blend (read as data: Non-Color)."""
    import bpy
    from PIL import Image
    px = (np.stack(glaze_maps(size, seed), 2) * 255 + 0.5).astype(np.uint8)
    path = os.path.join(tempfile.mkdtemp(), name.replace(' ', '-') + '.png')
    Image.fromarray(px[::-1]).save(path)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = 'Non-Color'
    img.pack()
    return img
