"""A figure module the render queue puts in a job's worktree (as blender/outfit/outfit.py) to render several pieces of
one set together, on one body, in one build: John's rule, 2026-11-19 ("when multiple pieces of the same set piece
request renders simultaneously batch them together and do a combined render always").

    env OUTFIT=skirt,sleeves python3 blender/base/views.py --figure outfit --mannequin --turn --out DIR

OUTFIT names the pieces' figure modules (blender/<piece>/<piece>.py), in wearing order, under to over: the queue
takes the order from sets.json. Each piece's build() is run in turn on the same scene:
  * the body (sorceress_pendant.build, which every standalone garment starts with) is built once; later calls
    return the first build's result;
  * a piece that another piece builds for itself (the boots build the hose) is built once: every piece's build()
    is run at most once, whoever calls it;
  * each piece is imported with its own folders first on sys.path, and the helper modules of the pieces before it
    (pattern.py, draft.py, sew.py: several garments have one of each name) are dropped from sys.modules first, so
    each piece gets its own;
  * the --head framing (HEAD_CZ, HEAD_H) covers the close-ups of the pieces in OUTFIT_FRAME (every piece when it is
    not set), from the lowest to the highest;
  * a material id that two pieces both define is reported (the later piece's look wins for both).
Each piece is fitted to the body as its own module fits it: a combined render shows where the layers meet."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.path.normpath(os.path.join(HERE, '..'))
SHARED = tuple(os.path.join(BLENDER, d) + os.sep for d in ('base', 'shared', 'sorceress_pendant'))
for d in ('sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(BLENDER, d))
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import sorceress_pendant as SP  # noqa: E402

PIECES = [p.strip() for p in os.environ.get('OUTFIT', '').split(',') if p.strip()]
# the pieces whose own close-up (--head) the combined one frames: the queue names the ones whose requests asked for it
FRAME = [p.strip() for p in os.environ.get('OUTFIT_FRAME', '').split(',') if p.strip()]
MEASURES = {}
LIGHT_SCALE = {}     # (views.py reads the figure's light scales: every piece's, merged)
BUILT = {}       # piece -> what its build() returned


def once(name, fn):
    def build(*a, **k):
        if name not in BUILT:
            BUILT[name] = fn(*a, **k)
        return BUILT[name]
    build.__wrapped__ = fn
    return build


def _helpers():
    """The modules loaded from a garment's own folders (not the base figure's, and not a piece itself)."""
    out = []
    for k, m in list(sys.modules.items()):
        f = getattr(m, '__file__', None) or ''
        if f.startswith(BLENDER + os.sep) and not f.startswith(SHARED) and k not in PIECES and k != 'outfit':
            out.append(k)
    return out


def build():
    if not PIECES:
        raise SystemExit('outfit: set OUTFIT=piece,piece (blender/<piece>/<piece>.py), in wearing order')
    SP.build = once('_body', SP.build)
    res = None
    frames, owner = [], {}
    framed = {k: os.environ.get(k) for k in ('HEAD_CZ', 'HEAD_H')}
    for name in PIECES:
        if not os.path.exists(os.path.join(BLENDER, name, name + '.py')):
            raise SystemExit(f'outfit: no blender/{name}/{name}.py at this commit')
        for k in _helpers():
            del sys.modules[k]
        sys.path.insert(0, os.path.join(BLENDER, name))
        mats = {k: v['name'] for k, v in female.MATERIALS.items()}
        mod = sys.modules.get(name) or importlib.import_module(name)
        for k, v in female.MATERIALS.items():
            if k in owner and mats.get(k) != v['name'] and k >= 30:
                print(f'outfit: material {k} is defined by {owner[k]} and by {name}', flush=True)
            if mats.get(k) != v['name']:
                owner[k] = name
        # every piece's build() at most once, whoever calls it (the boots build the hose)
        for p in PIECES:
            m = sys.modules.get(p)
            if m is not None and hasattr(m, 'build') and not hasattr(m.build, '__wrapped__'):
                m.build = once(p, m.build)
        for k in ('HEAD_CZ', 'HEAD_H'):
            if framed[k] is None:
                os.environ.pop(k, None)
        r = mod.build()
        res = res if res is not None else r
        if framed['HEAD_CZ'] is None and os.environ.get('HEAD_CZ') and os.environ.get('HEAD_H') \
                and (not FRAME or name in FRAME):
            cz, h = float(os.environ['HEAD_CZ']), float(os.environ['HEAD_H'])
            frames.append((cz - h / 2, cz + h / 2))
        MEASURES.update(getattr(mod, 'MEASURES', {}) or {})
        LIGHT_SCALE.update(getattr(mod, 'LIGHT_SCALE', {}) or {})
        print(f'outfit: built {name}', flush=True)
    if frames:
        lo, hi = min(f[0] for f in frames), max(f[1] for f in frames)
        os.environ['HEAD_CZ'], os.environ['HEAD_H'] = '%.2f' % (0.5 * (lo + hi)), '%.2f' % (hi - lo)
    bpy.context.view_layer.update()
    return res
