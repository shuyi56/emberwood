"""Run a render script from a branch with the queue's speed-ups, then hand over to it unchanged.

    python3 rq_run.py [--quality check|final] [--memo DIR] -- SCRIPT ARGS...

--memo DIR   remember the head's surface between runs. The eyes' sculpt solves head_front_y (female.py) for
             every vertex round the eyes, and the ears solve the head's side the same way: about 70% of a figure's
             build. Both are pure functions of the head's code and settings, so their answers are stored in DIR
             under a key made of female.py, ear.py and every plain setting and function of the female module as
             they stand when the build first asks. Same key, same inputs: the stored answer is the exact float the
             function returns for them, so the head is the one a run without the memo builds. (The build itself
             cuts the eyes' patch in a slightly different vertex order from run to run, so a few hundred new
             inputs turn up each time; those are worked out and added.) Any change to the head's code or settings
             gives a new key and the answers are worked out again.
             The same DIR/cloth remembers every cloth settle (blender/base/layers.py), keyed by everything it reads.
--gpu KIND   render Cycles on the graphics card (OPTIX or CUDA), on every device of that kind it finds.
--quality    check: Cycles at 8 samples with the denoiser on, for a quick look (hair, cloth and light still
             read; fine noise and caustics do not). final (default): the script's own settings, untouched.
--adaptive T final renders of 16 samples or more stop sampling a pixel once its noise is below T (Cycles' adaptive
             sampling; the default threshold is 0.01). The sprite's one-sample cells are not touched.
--keep-scene keep Cycles' copy of the scene between renders (use_persistent_data): pixel-identical, and each
             render after the first skips the scene sync (13 s or more on a figure).
--hot DIR    hot swap in the build (rq_hot.py): a costly step (the warden's hair clearance and trim) whose code and
             every surface it can reach are unchanged is not run again; what it did is put back from DIR.
--jit-save DIR / --jit-use DIR
             hot swap (rq_jit.py): on main, keep every image and a 4-sample probe of it in DIR; on a branch, render
             only the regions where its probe differs from main's and paste them over main's image.
"""
import atexit
import hashlib
import importlib
import marshal
import os
import pickle
import runpy
import sys
import types

argv = sys.argv[1:]
if 'blender' in os.path.basename(sys.argv[0]).lower():     # run as blender -b --python rq_run.py -- [own] -- SCRIPT
    argv = argv[argv.index('--') + 1:]
cut = argv.index('--')
own, rest = argv[:cut], argv[cut + 1:]
opt = lambda k, d=None: own[own.index(k) + 1] if k in own else d  # noqa: E731
QUALITY = opt('--quality', 'final')
MEMO = opt('--memo')
if MEMO:            # (the cloth settles are remembered beside the head's surface: blender/base/layers.py)
    os.environ.setdefault('EMBER_CLOTH_CACHE', os.path.join(MEMO, 'cloth'))
GPU = opt('--gpu')          # OPTIX or CUDA: render on the graphics card (the desktop worker)
script = os.path.abspath(rest[0]).replace('\\', '/')
blender = script[:script.index('/blender/') + len('/blender/')] if '/blender/' in script else \
    (os.path.join(os.getcwd(), 'blender/') if os.path.isdir(os.path.join(os.getcwd(), 'blender')) else None)


def plain(v, depth=0):
    if depth > 6:
        return False
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return True
    if isinstance(v, (tuple, list, frozenset)):
        return all(plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(plain(k, depth + 1) and plain(x, depth + 1) for k, x in v.items())
    return False


def _strip(c):
    consts = tuple(_strip_code(x) if isinstance(x, types.CodeType) else x for x in c.co_consts)
    return marshal.dumps(c.replace(co_filename='', co_consts=consts))


def _strip_code(c):
    return c.replace(co_filename='', co_consts=tuple(_strip_code(x) if isinstance(x, types.CodeType) else x
                                                     for x in c.co_consts))


def module_key(mod, files):
    root = blender[:-len('blender/')]
    h = hashlib.sha256()
    for f in files:
        with open(f, 'rb') as fh:
            h.update(fh.read())
    for name in sorted(vars(mod)):
        v = vars(mod)[name]
        if isinstance(v, types.FunctionType):
            h.update(name.encode()); h.update(_strip(v.__code__))
            h.update(repr(v.__defaults__).replace(root, '').encode() if plain(v.__defaults__) else b'?')
        elif plain(v):
            h.update(name.encode()); h.update(repr(v).replace(root, '').encode())
    return h.hexdigest()[:20]


def install_memo():
    base = os.path.join(blender, 'base')
    if not os.path.exists(os.path.join(base, 'female.py')):
        return
    for p in (os.path.join(blender, 'shared'), base):
        if p not in sys.path:
            sys.path.insert(0, p)
    female = importlib.import_module('female')
    ear = sys.modules.get('ear')
    files = [female.__file__] + ([ear.__file__] if ear else [])
    os.makedirs(MEMO, exist_ok=True)
    state = {'key': None, 'tables': {}, 'new': set()}

    def table(name):
        if state['key'] is None:
            state['key'] = module_key(female, files)
            print(f'rq memo: head key {state["key"]}', flush=True)
        t = state['tables'].get(name)
        if t is None:
            path = os.path.join(MEMO, f'{name}-{state["key"]}.pkl')
            try:
                with open(path, 'rb') as fh:
                    t = pickle.load(fh)
                print(f'rq memo: {name} {len(t)} answers loaded', flush=True)
            except (OSError, EOFError, pickle.UnpicklingError):
                t = {}
            state['tables'][name] = t
        return t

    def save():
        for name in state['new']:
            path = os.path.join(MEMO, f'{name}-{state["key"]}.pkl')
            tmp = f'{path}.{os.getpid()}'
            with open(tmp, 'wb') as fh:
                pickle.dump(state['tables'][name], fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            print(f'rq memo: {name} {len(state["tables"][name])} answers saved', flush=True)
    atexit.register(save)

    raw_front = female.head_front_y

    def head_front_y(x, z):
        if getattr(female, 'HEAD_SOURCE', None) is not None:
            return raw_front(x, z)     # (a head from a creator's file: its surface is not in the key)
        t = table('head_front_y')
        k = (x, z)
        if k in t:
            return t[k]
        y = t[k] = raw_front(x, z)
        state['new'].add('head_front_y')
        return y
    female.head_front_y = head_front_y

    if hasattr(female, 'side_x_finder'):
        raw_finder = female.side_x_finder

        def side_x_finder(head_section, jaw_relief=None):
            raw = raw_finder(head_section, jaw_relief)
            if head_section is not getattr(female, 'head_section', None) or \
                    jaw_relief not in (None, getattr(female, 'jaw_relief', None)):
                return raw     # (a figure's own section: not covered by the key)
            name = 'side_x' + ('' if jaw_relief is None else '_jaw')

            def head_point(z, side, y):
                t = table(name)
                k = (z, side, y)
                if k in t:
                    return t[k]
                r = t[k] = raw(z, side, y)
                state['new'].add(name)
                return r
            return head_point
        female.side_x_finder = side_x_finder


def install_quality():
    import bpy

    @bpy.app.handlers.persistent
    def check(scene, *_):
        if scene.render.engine == 'CYCLES':
            scene.cycles.samples = min(scene.cycles.samples, 8)
            scene.cycles.use_denoising = True
            if hasattr(scene.cycles, 'denoiser'):
                scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.app.handlers.render_pre.append(check)
    print('rq quality: check (8 samples, denoised)', flush=True)


def install_gpu():
    import bpy
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = GPU
    prefs.get_devices()
    used = [d.name for d in prefs.devices if d.type == GPU]
    for d in prefs.devices:
        d.use = d.type == GPU
    if not used:
        print(f'rq gpu: no {GPU} device, rendering on the CPU', flush=True)
        return

    @bpy.app.handlers.persistent
    def gpu(scene, *_):
        if scene.render.engine == 'CYCLES':
            scene.cycles.device = 'GPU'
    bpy.app.handlers.render_pre.insert(0, gpu)
    print(f'rq gpu: {GPU} on {", ".join(used)}', flush=True)


if MEMO and blender:
    install_memo()
if QUALITY == 'check':
    install_quality()
if GPU:
    install_gpu()
def install_keep_scene():
    """Every render keeps Cycles' copy of the scene for the next one (use_persistent_data), so a script that
    renders five angles syncs the figure once instead of five times. Only the turned root and the camera are
    updated between renders. Checked pixel for pixel on the warden with her hair (a 3/4 view rendered fresh, and
    again after the profile from the kept scene: no pixel differs) and on the mannequin turn (180 s down to
    137 s, identical)."""
    import bpy
    real = bpy.ops.render

    class Ops:
        def __getattr__(self, k):
            return getattr(real, k)

        @staticmethod
        def render(*a, **kw):
            sc = bpy.context.scene
            if sc.render.engine == 'CYCLES':
                sc.render.use_persistent_data = True
            return real.render(*a, **kw)
    bpy.ops.render = Ops()
    print('rq keep-scene: the scene stays loaded between renders', flush=True)


def install_adaptive(threshold):
    """John, 2026-11-19: adaptive sampling for the 3D look's final renders. Cycles stops sampling each pixel once its
    noise is under the threshold, so flat backdrop and smooth cloth finish early and the samples go where the
    noise is."""
    import bpy

    @bpy.app.handlers.persistent
    def adaptive(scene, *_):
        if scene.render.engine == 'CYCLES' and scene.cycles.samples >= 16:
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = threshold
            scene.cycles.adaptive_min_samples = 0
    bpy.app.handlers.render_pre.append(adaptive)
    print(f'rq adaptive: final renders stop at noise {threshold}', flush=True)


if '--keep-scene' in own:
    install_keep_scene()
if opt('--adaptive') and QUALITY != 'check' and blender:
    install_adaptive(float(opt('--adaptive')))
if opt('--hot'):                              # build hot swap: see rq_hot.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rq_hot
    rq_hot.install(opt('--hot'))
if opt('--jit-save') or opt('--jit-use'):     # hot swap: see rq_jit.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rq_jit
    rq_jit.install('save' if opt('--jit-save') else 'use', opt('--jit-save') or opt('--jit-use'))
sys.argv = rest
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name='__main__')
