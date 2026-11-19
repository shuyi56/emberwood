"""NOT USED BY THE QUEUE (John, 2026-11-18: stopped, since the warden takes her hair from the Face Creator).

Hot swap in the build: skip a costly step of a figure's build when nothing it depends on has changed, and put
its saved result in place instead.

    rq_run.py --hot DIR -- SCRIPT ARGS...

A figure's build is code, so a branch that changes a greave still runs every line of the build. Most of it is
cheap. The steps that are not are the ones that fit the hair to the clothes in every pose of the walk: on the
warden, the clearance (twice) and the trim, about 110 s of a 190 s build. Neither step reads the greave. So each
call of such a step is looked up under a key made of everything it can read:

  code      the step's own code and, followed name by name, every function and plain setting of the figure's
            module (and the base figure's, imported into it) that the code names: the step, what it calls, what
            that calls. A change anywhere else in the module, such as the greave's lines in build(), is not in
            the key.
  inputs    the hair as it stands when the step is called (its cap and locks, vertex by vertex, and the points
            the clearance moves), and, pose by pose (the frames the step visits), the world-space surface of
            every mesh the step can reach: for the clearance, the clothes it names (CLOTHES); for the trim, every
            mesh showing whose box, in any pose, comes within REACH of the hair's box. Surfaces are compared as
            sorted, rounded points and face centres, because the base figure builds the eyes' patch in a
            slightly different vertex order from run to run.

Same key, same result: the result is what the step did, stored when it last ran with that key. For the clearance
that is where every vertex of the hair ended and where the points it moves ended; for the trim, which faces of the
cap it took out, round by round. On a hit those are put back exactly; on a miss the step runs and its result is
stored. The key's surfaces are found by evaluating the meshes in each pose, which is what the steps themselves
begin with, so a miss costs a few seconds more than a run without this.

Each step and key is printed (rq hot: ...), so a log shows what was reused.
"""
import hashlib
import marshal
import os
import pickle
import sys
import time
import types

import bpy
import numpy as np

REACH = 1.8          # (the trim looks 1.5 out from each face of the cap; and a little over)
ROUND = 4            # decimals a surface is compared to
STEPS = {'warden': ('creator_hair_clearance', 'creator_hair_trim')}

_dir = None
_frames_cache = {}


def _plain(v, depth=0):
    if depth > 6:
        return False
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return True
    if isinstance(v, (tuple, list, frozenset)):
        return all(_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(_plain(k, depth + 1) and _plain(x, depth + 1) for k, x in v.items())
    return False


def _code_bytes(c):
    consts = tuple(_code_bytes(x) if isinstance(x, types.CodeType) else x for x in c.co_consts)
    return marshal.dumps(c.replace(co_filename='', co_consts=tuple(x if not isinstance(x, bytes) else x for x in consts)))


def _names(c):
    out = set(c.co_names)
    for x in c.co_consts:
        if isinstance(x, types.CodeType):
            out |= _names(x)
    return out


def code_key(fn):
    """The code a step runs: the function and, transitively, every function and plain value of its module that its
    code names (by name, as Python looks them up when it runs)."""
    h = hashlib.sha256()
    seen = set()
    todo = [fn]
    while todo:
        f = todo.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        h.update(f.__qualname__.encode())
        h.update(marshal.dumps(_strip(f.__code__)))
        h.update(repr(f.__defaults__).encode() if _plain(f.__defaults__) else b'?')
        g = f.__globals__
        for n in sorted(_names(f.__code__)):
            if n not in g:
                continue
            v = g[n]
            if isinstance(v, types.FunctionType):
                todo.append(v)
            elif _plain(v):
                h.update(n.encode()); h.update(repr(v).encode())
            elif type(v).__module__ == 'mathutils':      # (a Vector or Matrix setting, such as the skull's centre)
                h.update(n.encode()); h.update(repr(tuple(v)).encode())
            elif isinstance(v, types.ModuleType):
                h.update(n.encode())
    return h.hexdigest()[:16]


def _strip(c):
    return c.replace(co_filename='', co_consts=tuple(_strip(x) if isinstance(x, types.CodeType) else x
                                                     for x in c.co_consts))


def _surface(o, dg):
    """(sorted rounded world points and face centres, box) of a mesh as it is now."""
    me = o.evaluated_get(dg).to_mesh()
    n = len(me.vertices)
    co = np.empty(n * 3, np.float64)
    me.vertices.foreach_get('co', co)
    co = co.reshape(n, 3)
    M = np.array(o.matrix_world)
    w = co @ M[:3, :3].T + M[:3, 3]
    nf = len(me.polygons)
    ce = np.empty(nf * 3, np.float64)
    me.polygons.foreach_get('center', ce)
    o.evaluated_get(dg).to_mesh_clear()
    ce = ce.reshape(nf, 3) @ M[:3, :3].T + M[:3, 3]
    box = (w.min(axis=0), w.max(axis=0)) if n else (np.zeros(3), np.zeros(3))
    pts = np.round(np.concatenate([w, ce]), ROUND) + 0.0      # (+0.0: no -0.0)
    pts = pts[np.lexsort(pts.T[::-1])] if len(pts) else pts
    return hashlib.sha256(pts.tobytes()).hexdigest()[:16], box


def _hair_objs():
    return [o for o in bpy.data.objects if o.type == 'MESH' and (o.name == 'Hair' or o.name.startswith('Hair locks'))]


def _poses(cols):
    """Per frame visited: {mesh name: (surface hash, box)} for every mesh except the hair, which is an input of its
    own. The figure's other meshes do not change during these steps, so this is worked out once per build."""
    key = tuple(cols)
    if key in _frames_cache:
        return _frames_cache[key]
    sc = bpy.context.scene
    hair = {o.name for o in _hair_objs()}
    out = []
    t = time.time()
    for col in cols:
        sc.frame_set(col + 1)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        fr = {}
        for o in bpy.data.objects:
            if o.type != 'MESH' or o.name in hair:
                continue
            fp, box = _surface(o, dg)
            fr[o.name] = (fp, box, bool(o.hide_render or not o.visible_get()))
            if os.environ.get('RQ_HOT_DUMPMESH') and col == cols[0]:
                me = o.evaluated_get(dg).to_mesh()
                co = np.empty(len(me.vertices) * 3, np.float64); me.vertices.foreach_get('co', co)
                o.evaluated_get(dg).to_mesh_clear()
                np.save(os.environ['RQ_HOT_DUMPMESH'] + '-' + o.name.replace(' ', '_') + '.npy', co.reshape(-1, 3))
        cap = bpy.data.objects.get('Hair')
        out.append((fr, [round(x, 6) for row in cap.matrix_world for x in row] if cap else None))
    sc.frame_set(1)
    bpy.context.view_layer.update()
    print(f'rq hot: the figure\'s surfaces in {len(cols)} poses, {time.time() - t:.0f} s', flush=True)
    _frames_cache[key] = out
    if os.environ.get('RQ_HOT_EXIT_AFTER_POSES'):
        os._exit(0)
    return out


def _hair_state(F):
    h = hashlib.sha256()
    boxes = []
    for o in sorted(_hair_objs(), key=lambda o: o.name):
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get('co', co)
        h.update(o.name.encode()); h.update(co.tobytes())
        nl = len(me.loops)
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get('vertex_index', lv)
        h.update(lv.tobytes())
        M = np.array(o.matrix_world)
        w = co.reshape(n, 3) @ M[:3, :3].T + M[:3, 3]
        if n:
            boxes.append((w.min(axis=0), w.max(axis=0)))
    P = getattr(F, 'HAIR_PTS', None)
    if P is not None:
        h.update(np.array([tuple(p) for p in P], np.float32).tobytes())
        h.update(repr(getattr(F, 'HAIR_RUNS', None)).encode())
    return h.hexdigest()[:16], boxes


def _near(box, boxes, reach):
    lo, hi = box
    return any(np.all(lo <= b[1] + reach) and np.all(hi >= b[0] - reach) for b in boxes)


def step_key(F, fn, args, kw):
    cols = kw.get('cols', fn.__defaults__[0] if fn.__defaults__ else range(10))
    hair_fp, hair_boxes = _hair_state(F)
    poses = _poses(cols)
    h = hashlib.sha256()
    h.update(code_key(fn).encode()); h.update(repr((args, sorted(kw.items()))).encode())
    h.update(hair_fp.encode())
    reads = set()
    clothes = getattr(F, 'CLOTHES', ())
    for fr, capm in poses:
        h.update(repr(capm).encode())
        # (the hair in this pose: its boxes carried by the cap's own move from the rest pose)
        for name in sorted(fr):
            fp, box, hidden = fr[name]
            if fn.__name__.endswith('clearance'):
                want = name.startswith(clothes)
            else:
                want = not hidden and _near(box, hair_boxes, REACH + 6.0)   # (6: how far the head carries the hair in the walk)
            if want:
                reads.add(name)
                h.update(name.encode()); h.update(fp.encode())
    if os.environ.get('RQ_HOT_DUMP'):          # (every part of the key, to find what differs between two builds)
        import json
        parts = {'code': code_key(fn), 'args': repr((args, sorted(kw.items()))), 'hair': _hair_parts(F),
                 'poses': [{'cap': capm, 'meshes': {n: fr[n][0] for n in sorted(fr)}} for fr, capm in poses]}
        with open(os.environ['RQ_HOT_DUMP'] + f'-{fn.__name__}-{_dump_n()}.json', 'w') as f:
            json.dump(parts, f, indent=0)
    return h.hexdigest()[:20], reads


_DN = [0]


def _dump_n():
    _DN[0] += 1
    return _DN[0]


def _hair_parts(F):
    out = {}
    for o in sorted(_hair_objs(), key=lambda o: o.name):
        n = len(o.data.vertices)
        co = np.empty(n * 3, np.float32)
        o.data.vertices.foreach_get('co', co)
        out[o.name] = {'n': n, 'co': hashlib.sha256(co.tobytes()).hexdigest()[:12],
                       'co_r3': hashlib.sha256(np.round(co, 3).tobytes()).hexdigest()[:12],
                       'loops': len(o.data.loops)}
    P = getattr(F, 'HAIR_PTS', None)
    if P is not None:
        a = np.array([tuple(p) for p in P], np.float32)
        out['HAIR_PTS'] = {'n': len(P), 'h': hashlib.sha256(a.tobytes()).hexdigest()[:12],
                           'r3': hashlib.sha256(np.round(a, 3).tobytes()).hexdigest()[:12]}
        np.save(os.environ['RQ_HOT_DUMP'] + f'-pts-{_DN[0]}.npy', a)
    return out


# ---- what each step did, and putting it back
def _hair_coords():
    out = {}
    for o in _hair_objs():
        co = np.empty(len(o.data.vertices) * 3, np.float32)
        o.data.vertices.foreach_get('co', co)
        out[o.name] = co
    return out


def run_clearance(F, fn, args, kw):
    fn(*args, **kw)
    return {'coords': _hair_coords(),
            'pts': None if F.HAIR_PTS is None else [tuple(p) for p in F.HAIR_PTS]}


def put_clearance(F, res):
    for o in _hair_objs():
        o.data.vertices.foreach_set('co', res['coords'][o.name])
        o.data.update()
    if res['pts'] is not None:
        for p, v in zip(F.HAIR_PTS, res['pts']):
            p[:] = v


def run_trim(F, fn, args, kw):
    """Run the trim, noting which faces of the cap each of its cuts took out."""
    import bmesh
    cap = bpy.data.objects['Hair']
    cuts = []
    real_delete = bmesh.ops.delete

    def delete(bm, geom=(), context='FACES'):
        if bm is not None and context == 'FACES':
            cuts.append(sorted(f.index for f in geom if isinstance(f, bmesh.types.BMFace)))
        return real_delete(bm, geom=geom, context=context)
    bmesh.ops.delete = delete
    try:
        fn(*args, **kw)
    finally:
        bmesh.ops.delete = real_delete
    return {'cuts': cuts, 'polys': len(cap.data.polygons), 'verts': len(cap.data.vertices)}


def put_trim(F, res):
    import bmesh
    cap = bpy.data.objects['Hair']
    for idx in res['cuts']:
        bm = bmesh.new()
        bm.from_mesh(cap.data)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in idx], context='FACES')
        bm.to_mesh(cap.data)
        cap.data.update()
        bm.free()
    if (len(cap.data.polygons), len(cap.data.vertices)) != (res['polys'], res['verts']):
        raise RuntimeError('rq hot: the trim put back does not give the cap it gave')


KINDS = {'creator_hair_clearance': (run_clearance, put_clearance), 'creator_hair_trim': (run_trim, put_trim)}


def wrap(F, name):
    fn = getattr(F, name)
    run_, put_ = KINDS[name]

    def hot(*args, **kw):
        if name == 'creator_hair_trim' and (len(args) > 1 and args[1] is not None or kw.get('cut') is not None):
            return fn(*args, **kw)      # (a cut also rewrites the cap's map, which is not stored)
        t = time.time()
        key, reads = step_key(F, fn, args, kw)
        path = os.path.join(_dir, f'{name}-{key}.pkl')
        try:
            with open(path, 'rb') as fh:
                res = pickle.load(fh)
        except (OSError, EOFError, pickle.UnpicklingError):
            res = None
        if res is not None:
            put_(F, res)
            print(f'rq hot: {name} reused ({key}, {len(reads)} meshes in its key, {time.time() - t:.0f} s)', flush=True)
            return None
        res = run_(F, fn, args, kw)
        tmp = f'{path}.{os.getpid()}'
        with open(tmp, 'wb') as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        print(f'rq hot: {name} ran and was stored ({key}, {len(reads)} meshes in its key, {time.time() - t:.0f} s)',
              flush=True)
    hot.__wrapped__ = fn
    setattr(F, name, hot)


def install(d):
    """Wrap the costly steps of a figure's module as soon as it is imported."""
    global _dir
    _dir = d
    os.makedirs(d, exist_ok=True)
    import builtins
    real_import = builtins.__import__
    done = set()

    def imp(name, globals=None, locals=None, fromlist=(), level=0):
        m = real_import(name, globals, locals, fromlist, level)
        for fig, steps in STEPS.items():
            mod = sys.modules.get(fig)
            if mod is not None and fig not in done and all(hasattr(mod, s) for s in steps):
                done.add(fig)
                for s in steps:
                    wrap(mod, s)
                print(f'rq hot: {fig}: {", ".join(steps)} hot-swappable, results in {d}', flush=True)
        return m
    builtins.__import__ = imp
    import importlib
    real_im = importlib.import_module

    def im(name, package=None):
        m = real_im(name, package)
        imp('importlib')
        return m
    importlib.import_module = im
