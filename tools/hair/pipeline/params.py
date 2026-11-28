"""A style's parameter file: load it, keep every number's provenance, and resolve it to the tools' constants.

The file is YAML (JSON is YAML too, so a .json file loads the same way). A leaf is either a bare value or
    {v: VALUE, flag: FLAG, src: "where it comes from"}
and a mapping may carry `_flag` / `_src`, which its children inherit. FLAG is one of FLAGS. The provenance gauge
(check_provenance) fails a file with a number that has no flag, so every number in a style names its grounding
(project rule: research grounding).

A src of the form "ref:NAME" says the value is the reference sheet's ratio NAME; the measure stage checks it.

    load(path) -> Style;  Style.get('wave.mass.length_hh');  Style.flat() -> {path: Leaf}
"""
import hashlib
import json
import os

FLAGS = {
    'SOURCED': 'taken from a cited source (paper, craft or anatomy reference)',
    'MEASURED': 'read off the reference (refmeasure sheet, photo crop) or the head',
    'CREATOR': 'the Face Creator\'s own value (ground truth for hair, project rule)',
    'CALIBRATED': 'fitted so a gauge matches the reference; the gauge is named in src',
    'JOHN': 'John\'s direct instruction (applied, and still checked against research)',
    'DRAFT': 'no source found yet: a placeholder to replace (EXPERIMENTAL)',
    'STYLE': 'an artistic choice of this style (a look, not a measurement)',
    'DERIVED': 'computed from other parameters (the formula is in src)',
    'TOOL': 'a tool setting (sampling, a step, a seed): changes cost, not the look',
}


class Leaf:
    __slots__ = ('v', 'flag', 'src', 'path')

    def __init__(self, v, flag, src, path):
        self.v, self.flag, self.src, self.path = v, flag, src, path

    def __repr__(self):
        return 'Leaf(%s=%r, %s)' % (self.path, self.v, self.flag)


def _is_leaf(x):
    return isinstance(x, dict) and 'v' in x and set(x) <= {'v', 'flag', 'src'}


def _walk(node, path, flag, src, out):
    if _is_leaf(node):
        out[path] = Leaf(node['v'], node.get('flag', flag), node.get('src', src), path)
    elif isinstance(node, dict):
        f = node.get('_flag', flag)
        s = node.get('_src', src)
        for k, v in node.items():
            if k in ('_flag', '_src'):
                continue
            _walk(v, (path + '.' if path else '') + str(k), f, s, out)
    else:
        out[path] = Leaf(node, flag, src, path)


def plain(node):
    """The file's values without the provenance wrappers."""
    if _is_leaf(node):
        return plain(node['v'])
    if isinstance(node, dict):
        return {k: plain(v) for k, v in node.items() if k not in ('_flag', '_src')}
    if isinstance(node, list):
        return [plain(v) for v in node]
    return node


class Style:
    def __init__(self, raw, path):
        self.raw, self.path = raw, os.path.abspath(path)
        self.dir = os.path.dirname(self.path)
        self.leaves = {}
        _walk(raw, '', None, None, self.leaves)
        self.values = plain(raw)

    def get(self, dotted, default=KeyError):
        cur = self.values
        for k in dotted.split('.'):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                if default is KeyError:
                    raise KeyError('%s: no parameter %r' % (os.path.basename(self.path), dotted))
                return default
        return cur

    def has(self, dotted):
        return self.get(dotted, None) is not None

    def leaf(self, dotted):
        return self.leaves.get(dotted)

    def section(self, dotted):
        """A subtree's values, for hashing a node's inputs: a node sees only the parameters it reads."""
        return self.get(dotted, None)

    def resolve_path(self, p, roots):
        """A file named in the style: absolute, or relative to the first of roots (style dir, project files, repo)
        where it exists."""
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        for r in roots:
            q = os.path.join(r, p)
            if os.path.exists(q):
                return q
        return os.path.join(roots[0], p)


def load(path):
    with open(path) as f:
        text = f.read()
    try:
        import yaml
        raw = yaml.safe_load(text)
    except ImportError:                                  # (JSON needs nothing installed)
        raw = json.loads(text)
    if not isinstance(raw, dict) or raw.get('schema') != SCHEMA:
        raise ValueError('%s: not a style file (schema: %s expected)' % (path, SCHEMA))
    return Style(raw, path)


SCHEMA = 'emberwood-hair-style/1'


def digest(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def check_provenance(style, skip=('schema', 'style', 'family', 'title', 'inputs', 'approved', 'notes', 'stages')):
    """Every number in the style names its grounding: a flag from FLAGS. Strings, booleans and file paths need none.
    Returns dict(ok, numbers, by_flag, unflagged, bad_flag, draft)."""
    nums, unflagged, bad, by = 0, [], [], {}
    draft = []

    def numeric(v):
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return True
        if isinstance(v, (list, tuple)):
            return any(numeric(x) for x in v)
        if isinstance(v, dict):
            return any(numeric(x) for x in v.values())
        return False
    for p, lf in style.leaves.items():
        if p.split('.')[0] in skip or not numeric(lf.v):
            continue
        nums += 1
        if lf.flag is None:
            unflagged.append(p)
        elif lf.flag not in FLAGS:
            bad.append('%s: %s' % (p, lf.flag))
        else:
            by[lf.flag] = by.get(lf.flag, 0) + 1
            if lf.flag == 'DRAFT':
                draft.append(p)
    return dict(ok=not unflagged and not bad, numbers=nums, by_flag=by, unflagged=unflagged, bad_flag=bad,
                draft=draft)
