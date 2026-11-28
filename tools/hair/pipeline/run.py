"""One runner for every hairstyle: read a style's parameter file, build its family's graph, run it.

    python3 tools/hair/pipeline/run.py STYLE.yaml OUT [--code REPO] [--files PROJECT_FILES] [--until NODE]
                                       [--force NODE,...] [--graph] [--bindings]

  STYLE.yaml   the style (tools/hair/pipeline/styles/*.yaml); its family says which graph it runs
  OUT          the run's folder: one sub-folder per node, run.json the manifest. Rerun into the same folder and only
               nodes whose parameters, inputs or code changed run again
  --code       the checkout whose tools run (default: this one). A PR's worktree runs that PR's tools
  --files      where relative input paths are found after the style's own folder (default /mnt/project-files)
  --until      stop after this node (for example field_review, to see a field)
  --force      run these nodes even if cached
  --graph      print the graph (nodes, inputs, outputs) and stop
  --bindings   print the tool constants the style sets, and which differ from the tools' defaults, and stop

The run stops at a failing gauge (unless the style lists it under gauges.allow_fail with its reason) and at the
field_review gate until the style carries approved.field = the field's digest, which the gate prints."""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import params as PR  # noqa: E402
from graph import Context, Graph, run  # noqa: E402

FAMILIES = {'tied': 'families.tied', 'loose': 'families.loose'}


def family(name):
    import importlib
    if name not in FAMILIES:
        raise ValueError('unknown family %r (known: %s)' % (name, ', '.join(FAMILIES)))
    return importlib.import_module(FAMILIES[name])


def code_files(code_root):
    fs = []
    for pat in ('tools/hair/*.py', 'tools/solvers/hair/*.py', 'tools/gauges/hair/*.py'):
        fs += sorted(glob.glob(os.path.join(code_root, pat)))
    for pat in ('*.py', 'families/*.py'):                        # (the pipeline's own code, wherever --code points)
        fs += sorted(glob.glob(os.path.join(HERE, pat)))
    return fs


def build(style_path, code_root, files_root):
    style = PR.load(style_path)
    fam = family(style.get('family'))
    roots = [style.dir, files_root, code_root]
    for p in (os.path.join(code_root, 'tools', 'hair'), os.path.join(code_root, 'tools', 'solvers', 'hair'),
              os.path.join(code_root, 'tools', 'gauges', 'hair')):
        if p not in sys.path:
            sys.path.insert(1, p)
    inputs, nodes = fam.graph(style, code_root, roots)
    g = Graph(style.get('family'), nodes, inputs, code_files(code_root))
    return style, fam, g, roots


def bindings_report(style, fam, code_root):
    """The constants the style binds, against the tools' defaults (in a fresh process, as a stage sees them)."""
    job = dict(fam.binder(style, code_root)(), call='builtins:print', kwargs={})
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        job['report'] = os.path.join(t, 'r.json')
        jp = os.path.join(t, 'job.json')
        json.dump(job, open(jp, 'w'))
        subprocess.run([sys.executable, os.path.join(HERE, 'bind.py'), jp], check=True, cwd=code_root,
                       stdout=subprocess.DEVNULL)
        rep = json.load(open(job['report']))
    n = sum(len(c) for _, c in job['set']) + sum(len(c) for _, c in job['after'])
    return dict(bound=n, changed=rep['changed_from_default'])


def main(argv):
    a = list(argv)
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d        # noqa: E731
    style_path, out = a[0], a[1]
    code_root = os.path.abspath(opt('--code', os.path.join(HERE, '..', '..', '..')))
    files_root = opt('--files', '/mnt/project-files')
    style, fam, g, roots = build(style_path, code_root, files_root)
    if '--graph' in a:
        print('%s (family %s)\n%s' % (style.get('style'), style.get('family'), g.describe()))
        return 0
    if '--bindings' in a:
        print(json.dumps(bindings_report(style, fam, code_root), indent=1))
        return 0
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'run.log'), 'a')

    def log(t):
        print(t)
        logf.write(t + '\n'); logf.flush()
    log('style %s (family %s), code %s' % (style.get('style'), style.get('family'), code_root))
    ctx = Context(style, os.path.abspath(out), code_root, roots, fam.binder(style, code_root), log)
    force = (opt('--force', '') or '').split(',')
    m = run(g, ctx, until=opt('--until'), force=[f for f in force if f], log=log)
    log('status: %s' % m['status'])
    return 0 if m['status'] in ('done',) or m['status'].startswith(('waiting', 'stopped')) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
