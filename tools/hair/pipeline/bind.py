"""Run one tool of the hair pipeline in its own process with the style's constants bound onto it.

    python3 bind.py JOB.json

JOB: {"paths": [dirs put first on sys.path], "set": [[module, {CONST: value}], ...] (applied in order, so a module
that copies another's constant at import sees the bound value), "derive": [modules whose derive() recomputes the
constants computed from others], "after": [[module, {CONST: value}], ...] (set after derive), "call":
"module:function", "argv": [...] (sys.argv for a tool's main) or "kwargs": {...}, "report": path}.

A dict value with "__merge__": true updates the tool's dict constant key by key instead of replacing it.
Values from YAML come as lists and dicts; each is turned back into the type of the tool's own default (tuple, dict of
tuples, float), so a bound constant is the same Python value the tool would have had. The report lists every bound
constant and whether it differs from the tool's default: a style that reproduces today's tool lists none.

Why a process per tool: the tools keep state in module globals (caches, memos, the head). A fresh process starts each
stage from the tool's own defaults plus the style, never from what an earlier stage left behind."""
import importlib
import json
import math
import os
import sys


def coerce(new, old):
    if isinstance(old, tuple) and isinstance(new, (list, tuple)):
        if len(old) == len(new):
            return tuple(coerce(n, o) for n, o in zip(new, old))
        return tuple(new)
    if isinstance(old, dict) and isinstance(new, dict):
        return {k: coerce(v, old[k]) if k in old else v for k, v in new.items()}
    if isinstance(old, float) and isinstance(new, int) and not isinstance(new, bool):
        return float(new)
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [coerce(n, o) for n, o in zip(new, old)]
    return new


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b) or (math.isnan(float(a)) and math.isnan(float(b)))
        except (TypeError, ValueError):
            return False
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def apply(mods, changed):
    for name, consts in mods:
        m = importlib.import_module(name)
        for k, v in consts.items():
            if not hasattr(m, k):
                raise AttributeError('%s has no constant %s (the binding is out of date with the tool)' % (name, k))
            old = getattr(m, k)
            if isinstance(v, dict) and v.get('__merge__') and isinstance(old, dict):
                # (a dict constant updated key by key: a style that sets one preset keeps the tool's others)
                new = dict(old)
                new.update({kk: coerce(vv, old.get(kk)) if kk in old else vv for kk, vv in v.items() if kk != '__merge__'})
            else:
                new = coerce(v, old)
            if not same(new, old):
                changed.append(dict(module=name, const=k, default=repr(old), style=repr(new)))
            setattr(m, k, new)


def main(job_path):
    job = json.load(open(job_path))
    for p in reversed(job.get('paths', [])):
        sys.path.insert(0, p)
    changed = []
    apply(job.get('set', []), changed)
    for name in job.get('derive', []):
        m = importlib.import_module(name)
        if hasattr(m, 'derive'):
            m.derive()
    apply(job.get('after', []), changed)
    if job.get('report'):
        json.dump(dict(changed_from_default=changed), open(job['report'], 'w'), indent=1)
    mod, fn = job['call'].split(':')
    if mod.endswith('.py'):                                   # (a driver file, imported by path)
        spec = importlib.util.spec_from_file_location(os.path.basename(mod)[:-3], mod)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    if 'argv' in job:
        sys.argv = [mod] + [str(x) for x in job['argv']]
        return getattr(m, fn)()
    return getattr(m, fn)(**job.get('kwargs', {}))


if __name__ == '__main__':
    import importlib.util  # noqa: F401
    main(sys.argv[1])
