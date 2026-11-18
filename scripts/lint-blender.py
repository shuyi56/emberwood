#!/usr/bin/env python3
"""Lint rules for the Blender figure scripts (blender/<figure>/*.py). They are plain Python run inside Blender, so
eslint does not see them; this is run by `npm run lint`.

    python3 scripts/lint-blender.py            check every figure script
    python3 scripts/lint-blender.py --self-test run the rule against small fixtures

Rule shoulder-slope-twice
    female.finish_figure() lets every mesh on the body's shoulders down by female.SHOULDER_SLOPE (slope_shoulders).
    A figure that also slopes its own shoulders (a trapezius_slope, a TRAP_DROP setting, a call to slope_shoulders)
    is sloped twice unless it says what it means for the base slope: the shoulders come out thin, low and curved
    down, and no render of the figure alone shows why (the Sorceress's did, next to the headless gown's). So a
    figure script that slopes its own shoulders must also assign SHOULDER_SLOPE (female.SHOULDER_SLOPE = 0 when its
    own slope replaces the base one, or a smaller value when the two are meant to add up).
"""
import ast
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SKIP_DIRS = {'base', 'shared'}          # (the base figure owns the slope; shared holds helpers, not figures)
OWN_SLOPE_NAME = re.compile(r'trapezius|shoulder_?slope|slope_?shoulder', re.I)
OWN_SLOPE_ENV = {'TRAP_DROP', 'TRAP_CORNER', 'TRAP_NARROW'}


def own_slope_signals(tree):
    """(line, what) for each way a script slopes its own shoulders."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and OWN_SLOPE_NAME.search(n.name):
            out.append((n.lineno, f'defines {n.name}()'))
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', '')
            if OWN_SLOPE_NAME.search(name):
                out.append((n.lineno, f'calls {name}()'))
            # os.environ.get('TRAP_DROP', ...) and friends
            if name == 'get' and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in OWN_SLOPE_ENV:
                out.append((n.lineno, f'reads {n.args[0].value}'))
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                tn = t.id if isinstance(t, ast.Name) else getattr(t, 'attr', '')
                if tn in OWN_SLOPE_ENV:
                    out.append((n.lineno, f'sets {tn}'))
    return out


def sets_base_slope(tree):
    """True if the script assigns SHOULDER_SLOPE (female.SHOULDER_SLOPE = ..., or a bare name)."""
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []
        for t in targets:
            if (isinstance(t, ast.Attribute) and t.attr == 'SHOULDER_SLOPE') or (isinstance(t, ast.Name) and t.id == 'SHOULDER_SLOPE'):
                return True
        if isinstance(n, ast.Call) and getattr(n.func, 'id', getattr(n.func, 'attr', '')) == 'setattr' and len(n.args) > 1 \
                and isinstance(n.args[1], ast.Constant) and n.args[1].value == 'SHOULDER_SLOPE':
            return True
    return False


def check_source(src, path='<src>'):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f'{path}:{e.lineno}: syntax error: {e.msg}']
    signals = own_slope_signals(tree)
    if signals and not sets_base_slope(tree):
        line, what = signals[0]
        more = f' (and {len(signals) - 1} more)' if len(signals) > 1 else ''
        return [f'{path}:{line}: shoulder-slope-twice: this figure slopes its own shoulders ({what}{more}) but never '
                'sets SHOULDER_SLOPE, so female.finish_figure() slopes them again. Set female.SHOULDER_SLOPE = 0 '
                'if the own slope replaces the base one, or a smaller value if the two should add up.']
    return []


def figure_scripts():
    base = os.path.join(ROOT, 'blender')
    for fig in sorted(os.listdir(base)):
        d = os.path.join(base, fig)
        if fig in SKIP_DIRS or not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.py'):
                yield os.path.join(d, fn)


def self_test():
    bad = "import female\ndef trapezius_slope(co):\n    return co\n"
    bad_env = "import os\nTRAP_DROP = float(os.environ.get('TRAP_DROP', '1.4'))\n"
    bad_call = "import female\nfemale.slope_shoulders(2.0)\n"
    ok_set = bad + "female.SHOULDER_SLOPE = 0\n"
    ok_bare = bad_env + "SHOULDER_SLOPE = 0.5\n"
    plain = "import female\nx = female.finish_figure()\n"
    doc = '"""about the trapezius, TRAP_DROP and slope_shoulders in prose only"""\nimport female\n'
    cases = [(bad, 1), (bad_env, 1), (bad_call, 1), (ok_set, 0), (ok_bare, 0), (plain, 0), (doc, 0)]
    fails = [i for i, (src, want) in enumerate(cases) if len(check_source(src)) != want]
    if fails:
        print('lint-blender self-test failed on cases', fails)
        return 1
    print('lint-blender self-test: ok')
    return 0


def main():
    if '--self-test' in sys.argv:
        return self_test()
    problems = []
    for p in figure_scripts():
        with open(p, encoding='utf-8') as f:
            problems += check_source(f.read(), os.path.relpath(p, ROOT))
    for line in problems:
        print(line)
    if not problems:
        print('lint-blender: ok')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
