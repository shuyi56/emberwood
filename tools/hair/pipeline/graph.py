"""The hair pipeline's graph: ordered stages, each a node with typed inputs and outputs, and gauges as check nodes
after the stages they judge. A gate stops the run where John has to see something first (the flow field, before any
hair: project rule and the ponytail recipe's order, "measure, field, gauge, look, then hair").

A node:
    Node(name, kind, fn, inputs=[artifact, ...], outputs={artifact: type}, params=[dotted param prefixes], doc)
kind is 'stage' (makes artifacts), 'gauge' (measures them, returns checks) or 'gate' (passes or waits).
fn(ctx, ins, d) gets the input artifacts' paths and its own folder d, and returns
    dict(outputs={artifact: path}, result={...})                      for a stage
    dict(result=dict(checks=[check(...), ...], ...))                   for a gauge or a gate

Artifacts are files or folders, named once in the graph, each with a type from TYPES (a check that the thing on disk
is what it claims). The graph's own inputs (the head, the reference sheet) come from the style's `inputs`.

Caching: a node's key is the hash of its kind and name, the parameters it reads, its inputs' keys and the code of the
tools (every .py the family names). A node whose key and outputs are already on disk is reused, not run: change one
parameter and only the nodes that read it, and the nodes after them, run again. The run writes OUT/run.json, one row
per node with what it read, what it made, how long it took and every check.

Gauge checks fail the run where they stand unless the style lists the check under gauges.allow_fail with the reason
(a known fault that is reported, not hidden: the recipe's "report it, do not tune the field to hide it")."""
import hashlib
import json
import os
import subprocess
import sys
import time

from params import digest

HERE = os.path.dirname(os.path.abspath(__file__))


def _npz_has(*keys):
    def f(p):
        import numpy as np
        z = np.load(p, allow_pickle=True)
        miss = [k for k in keys if k not in z.files]
        return 'missing %s' % miss if miss else None
    return f


def _json_has(*keys):
    def f(p):
        d = json.load(open(p))
        miss = [k for k in keys if k not in d]
        return 'missing %s' % miss if miss else None
    return f


def _dir_has(*names):
    def f(p):
        if not os.path.isdir(p):
            return 'not a folder'
        miss = [n for n in names if not os.path.exists(os.path.join(p, n))]
        return 'missing %s' % miss if miss else None
    return f


TYPES = {
    'head': lambda p: None if p.endswith('.json.gz') and os.path.exists(p) else 'not a creator head export (.json.gz)',
    'refsheet': _json_has('landmarks', 'computed', 'unit'),     # a refmeasure sheet (PR #89)
    'reference': _json_has('ratios', 'checks'),                 # the measure stage's report
    'tied_field': _dir_has('field.npz', 'gauges.json', 'flow-field-views.png'),
    'field_cache': lambda p: None if os.path.isdir(p) else 'not a folder',
    'loose_field': _dir_has('field.npz', 'field-course.npz', 'gauges.json', 'flow-field.png'),
    'lines': _npz_has('lines'),
    'bangs': _dir_has('bangs-lines.npz'),
    'asset': _json_has('strands'),
    'fibres_dir': _dir_has('fibres.npz'),
    'figure': lambda p: None if p.endswith('.png') and os.path.exists(p) else 'not a png',
    'report': lambda p: None if os.path.exists(p) else 'missing',
}


def check(name, value, ok, bound=None, note=None):
    c = dict(name=name, value=value, ok=bool(ok))
    if bound is not None:
        c['bound'] = bound
    if note:
        c['note'] = note
    return c


class Node:
    def __init__(self, name, kind, fn, inputs=(), outputs=None, params=(), doc=''):
        assert kind in ('stage', 'gauge', 'gate')
        self.name, self.kind, self.fn = name, kind, fn
        self.inputs, self.outputs, self.params, self.doc = list(inputs), dict(outputs or {}), list(params), doc


class Graph:
    def __init__(self, family, nodes, graph_inputs, code_files):
        self.family, self.nodes, self.graph_inputs, self.code_files = family, nodes, graph_inputs, code_files
        made = set(graph_inputs)
        for n in nodes:                                   # (the order is the run's: every input made before it is read)
            for a in n.inputs:
                if a not in made:
                    raise ValueError('node %s reads %s before any node makes it' % (n.name, a))
            for a, t in n.outputs.items():
                if t not in TYPES:
                    raise ValueError('node %s: unknown type %s' % (n.name, t))
                made.add(a)

    def describe(self):
        rows = []
        for i, n in enumerate(self.nodes):
            rows.append('%2d %-6s %-22s in: %-40s out: %s' % (i, n.kind, n.name, ', '.join(n.inputs) or '-',
                                                             ', '.join('%s (%s)' % kv for kv in n.outputs.items()) or '-'))
        return '\n'.join(rows)


def file_digest(p):
    h = hashlib.sha1()
    if os.path.isdir(p):
        for root, _, files in sorted(os.walk(p)):
            for f in sorted(files):
                h.update(f.encode()); h.update(file_digest(os.path.join(root, f)).encode())
        return h.hexdigest()[:16]
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


class Context:
    """What a node's function can use: the style, the folders, and tool() to run a tool with the style bound."""
    def __init__(self, style, out, code_root, roots, binder, log):
        self.style, self.out, self.code_root, self.roots, self.binder, self.log = style, out, code_root, roots, binder, log

    def path(self, p):
        return self.style.resolve_path(p, self.roots)

    def tool(self, d, call, argv=None, kwargs=None, name='tool'):
        """Run call ('module:function' or 'driver.py:function') in a fresh process with the style's constants bound.
        Its log goes to d/NAME.log; the bound-constant report to d/NAME.bound.json."""
        job = dict(self.binder(), call=call, report=os.path.join(d, name + '.bound.json'))
        if argv is not None:
            job['argv'] = argv
        else:
            job['kwargs'] = kwargs or {}
        jp = os.path.join(d, name + '.job.json')
        json.dump(job, open(jp, 'w'), indent=1)
        t0 = time.time()
        with open(os.path.join(d, name + '.log'), 'w') as lf:
            r = subprocess.run([sys.executable, os.path.join(HERE, 'bind.py'), jp], stdout=lf, stderr=subprocess.STDOUT,
                               cwd=self.code_root)
        self.log('    %s: %s (%.0f s)' % (name, call, time.time() - t0))
        if r.returncode != 0:
            tail = open(os.path.join(d, name + '.log')).read()[-3000:]
            raise RuntimeError('%s failed (exit %d); log %s:\n%s' % (call, r.returncode, os.path.join(d, name + '.log'), tail))
        return json.load(open(job['report']))


def run(graph, ctx, until=None, force=(), log=print):
    style = ctx.style
    allow = style.get('gauges.allow_fail', {}) or {}
    code_key = digest({os.path.relpath(f, ctx.code_root): file_digest(f) for f in graph.code_files})
    keys, paths = {}, {}
    for a, p in graph.graph_inputs.items():
        if p is None:                                   # (an optional input the style does not give)
            paths[a], keys[a] = None, 'none'
            continue
        err = TYPES[graph_input_type(a)](p)
        if err:
            raise ValueError('input %s (%s): %s' % (a, p, err))
        paths[a], keys[a] = p, file_digest(p)
    manifest = dict(style=style.path, style_name=style.get('style'), family=graph.family, code_key=code_key,
                    started=time.strftime('%Y-%m-%d %H:%M:%S'), nodes=[])
    status = 'done'
    for i, n in enumerate(graph.nodes):
        if any(a not in paths for a in n.inputs):
            manifest['nodes'].append(dict(name=n.name, kind=n.kind, status='skipped', why='an input was not made'))
            continue
        pslice = {p: style.get(p, None) for p in n.params}
        if isinstance(pslice.get('gauges'), dict):              # (which faults are known changes no stage's output)
            pslice['gauges'] = {k: v for k, v in pslice['gauges'].items() if k != 'allow_fail'}
        key = digest(dict(node=n.name, kind=n.kind, params=pslice, ins={a: keys[a] for a in n.inputs}, code=code_key,
                          approved=style.get('approved', None) if n.kind == 'gate' else None))
        d = os.path.join(ctx.out, '%02d-%s' % (i, n.name))
        os.makedirs(d, exist_ok=True)
        meta_p = os.path.join(d, 'node.json')
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
        reuse = (meta is not None and meta.get('key') == key and meta.get('status') in ('ok', 'known')
                 and n.name not in force and all(os.path.exists(p) for p in meta.get('outputs', {}).values()))
        t0 = time.time()
        if reuse:
            res = meta
            log('%-6s %-22s reused' % (n.kind, n.name))
        else:
            log('%-6s %-22s running' % (n.kind, n.name))
            r = n.fn(ctx, {a: paths[a] for a in n.inputs}, d) or {}
            outs = r.get('outputs', {})
            for a, t in n.outputs.items():
                if a not in outs:
                    raise RuntimeError('node %s did not make %s' % (n.name, a))
                err = TYPES[t](outs[a])
                if err:
                    raise RuntimeError('node %s made %s at %s, but it is not a %s: %s' % (n.name, a, outs[a], t, err))
            result = r.get('result', {})
            st = 'ok'
            for c in result.get('checks', []):
                if not c['ok']:
                    if c['name'] in allow:
                        c['known'] = allow[c['name']]
                    else:
                        st = 'waiting' if n.kind == 'gate' else 'failed'
            if st == 'ok' and any('known' in c for c in result.get('checks', [])):
                st = 'known'
            res = dict(key=key, name=n.name, kind=n.kind, status=st, seconds=round(time.time() - t0, 1),
                       outputs=outs, params=pslice, inputs={a: paths[a] for a in n.inputs}, result=result, doc=n.doc)
            json.dump(res, open(meta_p, 'w'), indent=1, default=str)
        for a, p in res.get('outputs', {}).items():
            paths[a] = p
            keys[a] = digest([key, a])
        row = dict(name=n.name, kind=n.kind, status=res['status'], seconds=res.get('seconds'), reused=reuse,
                   outputs=res.get('outputs', {}),
                   checks=[dict(c) for c in res.get('result', {}).get('checks', [])])
        if 'message' in res.get('result', {}):
            row['message'] = res['result']['message']
        manifest['nodes'].append(row)
        for c in row['checks']:
            flag = 'ok ' if c['ok'] else ('known' if 'known' in c else 'FAIL')
            log('         %-5s %-34s %s%s' % (flag, c['name'], json.dumps(c.get('value'), default=str)[:70],
                                             '  bound %s' % json.dumps(c['bound'], default=str) if 'bound' in c else ''))
        if res['status'] == 'failed':
            status = 'failed at %s' % n.name
            log('stopped: %s %s failed (list a known fault under gauges.allow_fail with its reason)' % (n.kind, n.name))
            break
        if res['status'] == 'waiting':
            status = 'waiting at %s' % n.name
            log('stopped at gate %s: %s' % (n.name, res['result'].get('message', '')))
            break
        if until and n.name == until:
            status = 'stopped after %s (--until)' % n.name
            break
    manifest['status'] = status
    manifest['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(manifest, open(os.path.join(ctx.out, 'run.json'), 'w'), indent=1, default=str)
    return manifest


GRAPH_INPUT_TYPES = {'head': 'head', 'refsheet': 'refsheet', 'bangs_refsheet': 'refsheet'}


def graph_input_type(a):
    return GRAPH_INPUT_TYPES.get(a, 'report')
