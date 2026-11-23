"""Layered clothes, cheaply: one quality tier for the whole build, and every cloth settle remembered.

Dressing a figure is a stack of layers (chemise, doublet, corset, petticoat, skirt, hose, boots), and each loose one
is hung by a cloth solver (chemise/solver.py, petticoat/cloth.py, ...) against the body and the layers under it. The
solver is the costly step of a garment, and a layered build ran it for every layer on every render, though most
renders change only the top layer. Two things here stop that:

The tier. EMBER_TIER (draft, check or final) is the one setting for how good a run is, the same names views.py's
--tier uses for the render (view_tools.TIERS): views.py sets it from --tier, and a preset is a draft. Unset is final:
every build without it is exactly what it was.

    draft   the render: 8 samples, not denoised, 320 px, clay unless --look says (view_tools). The build: no
            subdivision on the clothes or the body, and the cloth solver runs only when it is asked for (--settle):
            a settle that is not remembered leaves its garment as it was fitted (the drafted or offset-shell shape),
            and the image is labelled with each layer's state. So adding a layer costs its fit and one cheap render;
            the solver runs on it (and on the layers it re-settles under it) in a second round, once the draft is
            seen and found worth it (John's layering rule, 2026-11-19).
    check   the render: 32 samples denoised, 480 px. The build: as final, the solver at full quality.
    final   untouched.

The settle memory. Every cloth settle (any function settle(o, rig, colliders, ...) in a module under blender/) is
looked up under a key of everything it reads: its module's code, the garment's points, faces and bone weights as
drafted, the rig's pose, the world-space surface of every collider as it stands when the settle starts, the other
arguments (pinned, full, the cloth's settings P, ...), the frame and Blender's version. Same key, same answer: what
the settle left the garment as (its points and its 'rest' attribute, its armature modifier's settings) is put back
exactly as the solver left it, and the settle's own return value returned. Change the body, the garment's draft,
the cloth's settings or any layer under it and the key changes, so only that layer and the layers settled against it
run again. A layer above a changed one is re-settled because its colliders' surface changed, not because of any list
of who lies on whom.

The memory lives in EMBER_CLOTH_CACHE (rq_run.py --memo DIR puts it in DIR/cloth), else in the system's temp folder.
EMBER_SETTLE=1 (views.py --settle) runs every solver that misses, at any tier; EMBER_SETTLE=0 runs none, even at
final. Each settle prints one line (layers: ...) saying whether it was remembered, solved or left unsettled, and how
long it took, and summary() is what views.py writes on the image's label."""
import hashlib
import inspect
import os
import pickle
import sys
import tempfile
import time

import bpy
import numpy as np

TIERS = ('draft', 'check', 'final')
CACHE_VERSION = 1


def tier():
    t = os.environ.get('EMBER_TIER', 'final')
    return t if t in TIERS else 'final'


def settle_policy():
    """'run' (a miss is solved), 'skip' (a miss is left unsettled)."""
    s = os.environ.get('EMBER_SETTLE')
    if s is not None:
        return 'run' if s not in ('0', 'off', 'no', '') else 'skip'
    return 'skip' if tier() == 'draft' else 'run'


def cache_dir():
    d = os.environ.get('EMBER_CLOTH_CACHE') or os.path.join(tempfile.gettempdir(), 'emberwood-cloth')
    os.makedirs(d, exist_ok=True)
    return d


LOG = []          # one entry per settle: (garment, 'cached' | 'solved' | 'unsettled', seconds)


def _record(garment, state, k, seconds, colliders):
    """One line per settle in the memory's own log (log.jsonl), for --list's hit and miss counts."""
    LOG.append((garment, state, seconds))
    try:
        import json
        with open(os.path.join(cache_dir(), 'log.jsonl'), 'a') as fh:
            fh.write(json.dumps({'t': round(time.time()), 'garment': garment, 'state': state, 'key': k,
                                 'seconds': round(seconds, 1), 'on': colliders, 'tier': tier()}) + '\n')
    except OSError:
        pass


# ------------------------------------------------------------------ the key
def _arr(h, a):
    a = np.ascontiguousarray(a)
    h.update(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())


def _mesh_key(h, o):
    """A garment as drafted: points, faces, bone weights, modifiers' settings, the rig's pose it rides on."""
    me = o.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', co)
    _arr(h, co)
    loops = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get('vertex_index', loops)
    _arr(h, loops)
    tot = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get('loop_total', tot)
    _arr(h, tot)
    names = [g.name for g in o.vertex_groups]
    h.update(repr(names).encode())
    w = [(v.index, g.group, round(g.weight, 6)) for v in me.vertices for g in v.groups]
    _arr(h, np.array(w, np.float64).reshape(-1, 3))
    for m in o.modifiers:
        h.update(repr((m.type, m.name, getattr(m, 'use_deform_preserve_volume', None))).encode())
    for k in sorted(me.attributes.keys()):
        if k.startswith('.'):
            continue
        a = me.attributes[k]
        h.update(repr((k, a.domain, a.data_type)).encode())
    _arr(h, np.array(o.matrix_world, np.float64))


def _rig_key(h, rig):
    if rig is None:
        h.update(b'no rig')
        return
    _arr(h, np.array(rig.matrix_world, np.float64))
    for pb in rig.pose.bones:
        h.update(pb.name.encode())
        _arr(h, np.array(pb.matrix, np.float64))
        _arr(h, np.array(pb.bone.matrix_local, np.float64))


def _surface_key(h, o):
    """A collider as the solver sees it: its evaluated surface in world space at this frame."""
    dg = bpy.context.evaluated_depsgraph_get()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', co)
    loops = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get('vertex_index', loops)
    e.to_mesh_clear()
    h.update(o.name.encode())
    _arr(h, np.round(co, 5))
    _arr(h, loops)
    _arr(h, np.array(o.matrix_world, np.float64))


def _value_key(h, v, depth=0):
    if depth > 5:
        h.update(b'deep')
    elif isinstance(v, np.ndarray):
        _arr(h, v)
    elif isinstance(v, dict):
        h.update(b'{')
        for k in sorted(v, key=repr):
            h.update(repr(k).encode())
            _value_key(h, v[k], depth + 1)
        h.update(b'}')
    elif isinstance(v, (list, tuple)):
        h.update(b'[')
        for x in v:
            _value_key(h, x, depth + 1)
        h.update(b']')
    elif isinstance(v, bpy.types.Object):
        _surface_key(h, v) if v.type == 'MESH' else h.update(v.name.encode())
    elif callable(v):
        h.update(getattr(v, '__qualname__', repr(type(v))).encode())
    else:
        h.update(repr(v).encode())


def _code_key(h, fn):
    """The settle's own module, and every module of the repo's blender/ folder it imports by name."""
    mod = sys.modules.get(fn.__module__)
    files = set()
    if mod is not None and getattr(mod, '__file__', None):
        files.add(mod.__file__)
        for v in vars(mod).values():
            f = getattr(v, '__file__', None)
            if f and os.sep + 'blender' + os.sep in os.path.abspath(f) and os.path.basename(f) not in ('female.py',):
                files.add(f)
    for f in sorted(files):
        with open(f, 'rb') as fh:
            h.update(os.path.basename(f).encode())
            h.update(fh.read())


def key(fn, o, rig, colliders, args, kwargs):
    h = hashlib.sha256()
    h.update(repr((CACHE_VERSION, bpy.app.version_string, fn.__module__, fn.__name__)).encode())
    _code_key(h, fn)
    _mesh_key(h, o)
    _rig_key(h, rig)
    for c in colliders:
        _surface_key(h, c)
    _value_key(h, list(args))
    _value_key(h, kwargs)
    h.update(repr(bpy.context.scene.frame_current).encode())
    return h.hexdigest()[:24]


# ------------------------------------------------------------------ what a settle leaves
def _snapshot(o):
    me = o.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', co)
    rest = None
    if 'rest' in me.attributes:
        rest = np.empty(len(me.vertices) * 3, np.float32)
        me.attributes['rest'].data.foreach_get('vector', rest)
    mods = [(m.name, m.use_deform_preserve_volume) for m in o.modifiers if m.type == 'ARMATURE']
    return {'co': co, 'rest': rest, 'mods': mods}


def _restore(o, snap):
    me = o.data
    me.vertices.foreach_set('co', snap['co'])
    if snap['rest'] is not None and 'rest' in me.attributes:
        me.attributes['rest'].data.foreach_set('vector', snap['rest'])
    for name, keep in snap['mods']:
        m = o.modifiers.get(name)
        if m is not None:
            m.use_deform_preserve_volume = keep
    me.update()


def _posed_now(o):
    """Where the garment's points stand now on the posed rig (what a settle returns when the cloth did not move)."""
    dg = bpy.context.evaluated_depsgraph_get()
    e = o.evaluated_get(dg)
    # (the garment's own points: its armature only, nothing that adds points)
    off = [m for m in o.modifiers if m.type != 'ARMATURE' and m.show_viewport]
    for m in off:
        m.show_viewport = False
    dg.update()
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    co = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
    e.to_mesh_clear()
    for m in off:
        m.show_viewport = True
    return co


def remembered(fn):
    """fn (a settle(o, rig, colliders, ...)) looked up in the settle memory first; see the module's docstring."""
    if getattr(fn, '_layers', False):
        return fn
    sig = inspect.signature(fn)

    def settle(o, rig, colliders, *args, **kwargs):
        t0 = time.time()
        log = kwargs.get('log') or print
        b = sig.bind_partial(o, rig, colliders, *args, **kwargs)
        b.apply_defaults()
        rest = {k: v for k, v in b.arguments.items() if k not in ('o', 'rig', 'colliders', 'log')}
        k = key(fn, o, rig, list(colliders), (), rest)
        path = os.path.join(cache_dir(), f'{fn.__module__}.{fn.__name__}-{k}.pkl')
        what = f'{o.name} on {", ".join(c.name for c in colliders)}'
        # (for the label: the garment, and the clothes it is settled against, the body and the hands left out)
        over = [c.name for c in list(colliders)[1:] if not c.name.startswith('Hand')]
        name = o.name + (f' against {"+".join(over)}' if over else '')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as fh:
                    got = pickle.load(fh)
                _restore(o, got['snap'])
                _record(name, 'cached', k, time.time() - t0, [c.name for c in colliders])
                print(f'layers: {what}: settle remembered ({k}, {time.time() - t0:.1f} s; it took {got["took"]:.0f} s)',
                      flush=True)
                return got['ret']
            except Exception as ex:                      # (a torn file: solved again and stored over it)
                print(f'layers: {what}: memory unreadable ({ex}), solving', flush=True)
        if settle_policy() == 'skip':
            _record(name, 'unsettled', k, time.time() - t0, [c.name for c in colliders])
            print(f'layers: {what}: left unsettled (draft: run again with --settle for the solver)', flush=True)
            return _posed_now(o)
        ret = fn(o, rig, colliders, *args, **kwargs)
        took = time.time() - t0
        _record(name, 'solved', k, took, [c.name for c in colliders])
        tmp = path + f'.{os.getpid()}.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump({'snap': _snapshot(o), 'ret': ret, 'took': took}, fh, protocol=4)
        os.replace(tmp, path)
        print(f'layers: {what}: solved in {took:.0f} s, remembered as {k}', flush=True)
        return ret

    settle.__wrapped__ = fn
    settle.__doc__ = fn.__doc__
    settle._layers = True
    return settle


def _is_settle(fn):
    try:
        p = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return p[:3] == ['o', 'rig', 'colliders']


def install():
    """Wrap every cloth settle already imported from the repo's blender/ folder (call before the figure builds)."""
    done = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None) or ''
        if os.sep + 'blender' + os.sep not in os.path.abspath(f):
            continue
        fn = getattr(mod, 'settle', None)
        if inspect.isfunction(fn) and fn.__module__ == mod.__name__ and _is_settle(fn):
            mod.settle = remembered(fn)
            done.append(name)
    if done:
        print(f'layers: tier {tier()}, solver {settle_policy()} on a miss; remembering {", ".join(sorted(done))} '
              f'in {cache_dir()}', flush=True)
    return done


# ------------------------------------------------------------------ the draft build
def draft_scene():
    """The draft tier's build, after the figure is built: no subdivision anywhere (the clothes' and the body's).
    Returns what was turned off, for the log."""
    off = 0
    for o in bpy.data.objects:
        for m in o.modifiers:
            if m.type in ('SUBSURF', 'MULTIRES') and (m.show_render or m.show_viewport):
                m.show_render = m.show_viewport = False
                off += 1
    print(f'layers: draft: {off} subdivision modifiers off', flush=True)
    return off


def summary():
    """The settles of this run, for the image's label: 'cloth: Chemise cached, Chemise against Doublet unsettled'."""
    if not LOG:
        return ''
    return 'cloth: ' + ', '.join(f'{n} {s}' for n, s, _ in LOG)


# ------------------------------------------------------------------ looking at the memory, clearing it
def main(argv):
    """python3 blender/base/layers.py [--list] [--clear [GARMENT]]: what the settle memory holds and how often it
    was used, or clear all of it (or one garment's settles: the file names start with the settle's module)."""
    import json
    d = cache_dir()
    files = sorted(f for f in os.listdir(d) if f.endswith('.pkl'))
    if '--clear' in argv:
        i = argv.index('--clear')
        only = argv[i + 1] if len(argv) > i + 1 and not argv[i + 1].startswith('--') else None
        gone = 0
        for f in files:
            if only is None or only in f:
                os.remove(os.path.join(d, f))
                gone += 1
        if only is None and os.path.exists(os.path.join(d, 'log.jsonl')):
            os.remove(os.path.join(d, 'log.jsonl'))
        print(f'layers: cleared {gone} remembered settles from {d}' + (f' (matching {only!r})' if only else ''))
        return
    size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    print(f'{d}: {len(files)} remembered settles, {size / 1e6:.1f} MB')
    for f in files:
        print('  ', f)
    p = os.path.join(d, 'log.jsonl')
    if os.path.exists(p):
        rows = [json.loads(x) for x in open(p) if x.strip()]
        n = {s: sum(r['state'] == s for r in rows) for s in ('cached', 'solved', 'unsettled')}
        saved = sum(r['seconds'] for r in rows if r['state'] == 'solved')
        print(f"settles: {n['cached']} remembered, {n['solved']} solved ({saved:.0f} s of solver), "
              f"{n['unsettled']} left unsettled in drafts")
        for r in rows[-12:]:
            print(f"   {time.strftime('%m-%d %H:%M', time.localtime(r['t']))}  {r['tier']:5}  {r['state']:9}  "
                  f"{r['seconds']:6.1f} s  {r['garment']} on {', '.join(r['on'])}")


if __name__ == '__main__':
    main(sys.argv[1:])
