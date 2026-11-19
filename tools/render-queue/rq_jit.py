"""Hot-swap rendering: render only the part of an image that a change touches, and take the rest from main's render.

    rq_run.py --jit-save DIR -- SCRIPT ARGS...    (main: render as usual, and keep each image and a probe of it in DIR)
    rq_run.py --jit-use DIR  -- SCRIPT ARGS...    (a branch: render only what differs from the images in DIR)

Every still the script renders (bpy.ops.render.render(write_still=True) to a PNG) is caught here, by its place in
the run: the script's n-th render on the branch is matched with the n-th render on main, and only when the camera,
its lens, the resolution, the samples and the frame are the same.

The probe. Each image is also rendered at PROBE samples, with the same seed and adaptive sampling off, to an EXR.
Cycles draws each pixel's random numbers from the pixel and the sample number alone, so where nothing a pixel's
paths reach has changed, its probe is the same to the last bit on main and on the branch; a pixel that sees the
change, directly, in a shadow it casts, in a reflection or in light bounced off it, comes out different. So the
probes' difference is the map of what the change touches, found for a fraction of a render.

The region. The map, widened by MARGIN pixels (the pixel filter, and paths too rare for the probe to find), is
split into blocks; each group of touching blocks is rendered at the script's full settings as a border render
(not cropped), and its pixels are pasted over main's image. A border render's pixels are the full render's pixels
exactly (checked: no difference at all inside the border), so the image is the one a full render would give
wherever the probe saw the change, and main's pixels, untouched by it, everywhere else.

It falls back to a plain render when there is nothing in DIR for this image, when anything about the shot differs,
when denoising is on (the denoiser mixes each pixel with its neighbours, so a border's edge would show), when the
image is not a PNG, or when the change covers more than FULL_AT of the frame.
"""
import json
import os
import shutil
import tempfile
import time

import bpy
import numpy as np

PROBE = 4          # samples for the probe
MARGIN = 10        # pixels round what the probe found
BLOCK = 24         # the map is grouped in blocks this size before regions are cut
FULL_AT = float(os.environ.get('RQ_JIT_FULL_AT', 0.55))   # above this share of the frame, render it all
EPS = float(os.environ.get('RQ_JIT_EPS', 1e-6))           # probe difference that counts as a change
PERSIST = os.environ.get('RQ_JIT_PERSIST') == '1'          # (keep the scene synced between renders: experiment)
KEEP = os.environ.get('RQ_JIT_KEEP')                       # (keep the branch's probes here, to study them)

_state = {'n': 0, 'mode': None, 'dir': None, 'saved': 0.0, 'log': []}


def _shot(sc):
    cam = sc.camera
    return {
        'res': [sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage],
        'samples': sc.cycles.samples, 'adaptive': bool(sc.cycles.use_adaptive_sampling),
        'seed': sc.cycles.seed, 'frame': sc.frame_current, 'engine': sc.render.engine,
        'cam': [round(x, 5) for row in cam.matrix_world for x in row] if cam else None,
        'lens': [cam.data.type, round(cam.data.ortho_scale, 5), round(cam.data.lens, 5)] if cam else None,
        'view': sc.view_settings.view_transform, 'look': sc.view_settings.look,
        'film_transparent': bool(sc.render.film_transparent),
    }


def _render_to(path, fmt, **over):
    """Render the scene to path in fmt ('PNG' as the script set it up, or 'EXR' for the probe), with settings
    changed only for this render."""
    sc = bpy.context.scene
    keep = {}
    for k, v in over.items():
        obj, attr = (sc.cycles, k[7:]) if k.startswith('cycles_') else (sc.render, k)
        keep[(obj, attr)] = getattr(obj, attr)
        setattr(obj, attr, v)
    imf = sc.render.image_settings
    fkeep = (imf.file_format, imf.color_depth, imf.color_mode, sc.render.filepath)
    if fmt == 'EXR':
        imf.file_format = 'OPEN_EXR'
        imf.color_depth = '32'
        imf.color_mode = 'RGBA'
    sc.render.filepath = path
    try:
        _ORIG.render(write_still=True)
    finally:
        imf.file_format, imf.color_depth = fkeep[0], fkeep[1]
        imf.color_mode = fkeep[2]
        sc.render.filepath = fkeep[3]
        for (obj, attr), v in keep.items():
            setattr(obj, attr, v)


def _probe(path):
    _render_to(path, 'EXR', cycles_samples=PROBE, cycles_use_adaptive_sampling=False, use_border=False)
    im = bpy.data.images.load(path)
    w, h = im.size
    a = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(a)
    bpy.data.images.remove(im)
    return a.reshape(h, w, 4)[::-1]          # (top row first, as in the PNG)


def _regions(mask):
    """Rectangles (x0, y0, x1, y1), top-down pixels, end exclusive, covering the mask widened by MARGIN, one per
    group of touching BLOCK-sized blocks."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    bh, bw = (h + BLOCK - 1) // BLOCK, (w + BLOCK - 1) // BLOCK
    blocks = np.zeros((bh, bw), bool)
    for y0, y1, x0, x1 in ((max(0, y - MARGIN), min(h, y + MARGIN + 1), max(0, x - MARGIN), min(w, x + MARGIN + 1))
                           for y, x in zip(ys, xs)):
        blocks[y0 // BLOCK:(y1 - 1) // BLOCK + 1, x0 // BLOCK:(x1 - 1) // BLOCK + 1] = True
    seen = np.zeros_like(blocks)
    out = []
    for by in range(bh):
        for bx in range(bw):
            if not blocks[by, bx] or seen[by, bx]:
                continue
            stack, cells = [(by, bx)], []
            seen[by, bx] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if 0 <= ny < bh and 0 <= nx < bw and blocks[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            cy0 = min(c[0] for c in cells); cy1 = max(c[0] for c in cells) + 1
            cx0 = min(c[1] for c in cells); cx1 = max(c[1] for c in cells) + 1
            out.append((cx0 * BLOCK, cy0 * BLOCK, min(w, cx1 * BLOCK), min(h, cy1 * BLOCK)))
    return out


def _log(msg):
    _state['log'].append(msg)
    print(f'rq jit: {msg}', flush=True)


def render(*args, **kw):
    sc = bpy.context.scene
    i = _state['n']
    _state['n'] += 1
    path = sc.render.filepath
    imf = sc.render.image_settings
    plain = not kw.get('write_still') or args or imf.file_format != 'PNG' or sc.render.engine != 'CYCLES' or \
        sc.cycles.use_denoising or sc.render.use_border
    if _state['mode'] == 'off':
        if PERSIST:
            sc.render.use_persistent_data = True
        return _ORIG.render(*args, **kw)
    if plain:
        _log(f'render {i}: rendered in full ({"denoised" if sc.cycles.use_denoising else "not a plain PNG still"})')
        return _ORIG.render(*args, **kw)
    d = _state['dir']
    if PERSIST:
        sc.render.use_persistent_data = True
    shot = _shot(sc)
    base_png, base_probe, base_meta = (os.path.join(d, f'r{i}{x}') for x in ('.png', '-probe.exr', '.json'))
    try:
        if _state['mode'] == 'save':
            t = time.time()
            r = _ORIG.render(*args, **kw)
            full = time.time() - t
            shutil.copyfile(path, base_png)
            _probe(base_probe)
            with open(base_meta, 'w') as f:
                json.dump({'shot': shot, 'seconds': round(full, 1), 'out': os.path.basename(path)}, f)
            _log(f'render {i}: saved as the base ({full:.0f} s)')
            return r
        try:
            meta = json.load(open(base_meta))
        except (OSError, ValueError):
            meta = None
        if meta is None or meta['shot'] != shot or not os.path.exists(base_png) or not os.path.exists(base_probe):
            why = 'no base for it' if meta is None else 'the shot differs from the base'
            _log(f'render {i}: rendered in full ({why})')
            return _ORIG.render(*args, **kw)
        t = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            pa = _probe(os.path.join(KEEP or tmp, f'probe{i}.exr' if KEEP else 'probe.exr'))
            pb = np.empty_like(pa)
            im = bpy.data.images.load(base_probe)
            im.pixels.foreach_get(pb.reshape(-1))
            bpy.data.images.remove(im)
            pb = pb[::-1]
            mask = np.abs(pa - pb).max(axis=2) > EPS
            h, w = mask.shape
            regs = _regions(mask)
            if regs:      # (one border render: each render pays the scene's sync again, which costs more than the pixels)
                regs = [(min(r[0] for r in regs), min(r[1] for r in regs), max(r[2] for r in regs), max(r[3] for r in regs))]
            area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in regs) / float(w * h)
            if not regs:
                shutil.copyfile(base_png, path)
                _log(f'render {i}: unchanged, main\'s image reused ({time.time() - t:.0f} s for the probe, '
                     f'{meta["seconds"]:.0f} s saved)')
                _state['saved'] += meta['seconds'] - (time.time() - t)
                return {'FINISHED'}
            if area > FULL_AT:
                _log(f'render {i}: the change touches {area:.0%} of the frame, rendered in full')
                return _ORIG.render(*args, **kw)
            from PIL import Image
            img = np.array(Image.open(base_png))
            for k, (x0, y0, x1, y1) in enumerate(regs):
                rp = os.path.join(tmp, f'region{k}.png')
                # (Blender's border is in 0-1, from the bottom left)
                # (2 pixels wider all round than what is pasted: Blender rounds a border's edge to whole pixels)
                bx0, by0, bx1, by1 = max(0, x0 - 2), max(0, y0 - 2), min(w, x1 + 2), min(h, y1 + 2)
                _render_to(rp, 'PNG', use_border=True, use_crop_to_border=False,
                           border_min_x=bx0 / w, border_max_x=bx1 / w, border_min_y=1 - by1 / h, border_max_y=1 - by0 / h)
                part = np.array(Image.open(rp))
                img[y0:y1, x0:x1] = part[y0:y1, x0:x1]
            Image.fromarray(img).save(path)
        took = time.time() - t
        _state['saved'] += meta['seconds'] - took
        _log(f'render {i}: {len(regs)} region(s), {area:.0%} of the frame, {took:.0f} s against {meta["seconds"]:.0f} s '
             f'in full ({int(mask.sum())} pixels changed in the probe)')
        return {'FINISHED'}
    finally:
        pass


class _Ops:
    def __getattr__(self, k):
        return getattr(_ORIG, k)

    render = staticmethod(render)


_ORIG = None


def install(mode, d):
    global _ORIG
    os.makedirs(d, exist_ok=True)
    _state['mode'], _state['dir'] = mode, d
    _ORIG = bpy.ops.render
    bpy.ops.render = _Ops()

    import atexit

    def report():
        if _state['mode'] == 'use':
            print(f'rq jit: {_state["n"]} renders, about {max(0.0, _state["saved"]):.0f} s of rendering saved', flush=True)
    atexit.register(report)
    print(f'rq jit: {"saving the base in" if mode == "save" else "hot-swapping against"} {d}', flush=True)
