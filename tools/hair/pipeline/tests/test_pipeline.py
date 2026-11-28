"""Tests for the hair pipeline's own parts (no head, no tools run): parameter files, binding, the wave model and
gauge, and the graph's caching and stopping. Run: python3 -m pytest tools/hair/pipeline/tests -q"""
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import bind  # noqa: E402
import params as PR  # noqa: E402
import wave as W  # noqa: E402
from graph import Context, Graph, Node, check, run  # noqa: E402


def _style(tmp, body):
    p = os.path.join(tmp, 's.yaml')
    open(p, 'w').write(body)
    return PR.load(p)


def test_leaves_inherit_flags_and_provenance_fails_unflagged():
    with tempfile.TemporaryDirectory() as t:
        s = _style(t, """schema: emberwood-hair-style/1
style: t
family: tied
a: {_flag: MEASURED, _src: sheet, x: 0.5, y: {v: 2, flag: DRAFT, src: guess}}
b: 3
c: {v: 4, flag: NOPE}
""")
        assert s.get('a.x') == 0.5 and s.leaf('a.x').flag == 'MEASURED' and s.leaf('a.y').flag == 'DRAFT'
        r = PR.check_provenance(s)
        assert r['unflagged'] == ['b'] and r['bad_flag'] == ['c: NOPE'] and not r['ok'] and r['draft'] == ['a.y']


def test_bind_coerces_to_the_tools_types_and_merges_dicts():
    assert bind.coerce([1, 2], (0.1, 0.2)) == (1, 2) and isinstance(bind.coerce([1, 2], (0.1, 0.2)), tuple)
    assert bind.coerce({'th': [1, 2]}, {'th': (0.0, 0.0)}) == {'th': (1, 2)}
    assert isinstance(bind.coerce(3, 1.0), float)
    m = type(sys)('fake_tool'); m.P = {'a': {'x': 1.0}, 'b': {'x': 2.0}}; m.K = (1.0, 2.0)
    sys.modules['fake_tool'] = m
    ch = []
    bind.apply([['fake_tool', {'P': {'__merge__': True, 'a': {'x': 1}}, 'K': [1, 2]}]], ch)
    assert m.P == {'a': {'x': 1.0}, 'b': {'x': 2.0}} and m.K == (1.0, 2.0) and ch == []
    bind.apply([['fake_tool', {'K': [1, 3]}]], ch)
    assert len(ch) == 1 and ch[0]['const'] == 'K'


def _helix(lam, a_out, a_side, delta, n=1200, length=4.0, phi=0.3):
    s = np.linspace(0, length, n)
    o, sd = W.offsets(s, lam, a_out, a_side, phi, delta)
    base = np.c_[np.zeros(n), np.full(n, -0.3), -s]                 # a straight fall in front of the axis
    # out is radial from the axis (0, 0.04): -y here; side = t x out
    t = np.array([0, 0, -1.0]); out = np.array([0, -1.0, 0]); side = np.cross(t, out)
    return base + o[:, None] * out + sd[:, None] * side, base


def test_wave_gauge_recovers_a_planar_meander_and_a_helix():
    p, b = _helix(0.33, 0.0, 0.04, 0.0)
    m = W.measure([p], (0.2, 0.46), trends=[b])
    assert abs(m['wavelength'] - 0.33) < 0.01 and abs(m['amp_side'] - 0.04) < 0.002 and m['amp_out'] < 0.002
    p, b = _helix(0.33, 0.02, 0.04, math.pi / 2)
    m = W.measure([p], (0.2, 0.46), trends=[b])
    assert abs(m['amp_out'] - 0.02) < 0.002 and abs(m['amp_side'] - 0.04) < 0.002 and abs(m['twist_deg'] - 90) < 3
    p, _ = _helix(0.33, 0.02, 0.04, math.pi / 2)                     # (no base given: the trend is smoothed out)
    m = W.measure([p], (0.2, 0.46))
    assert abs(m['wavelength'] - 0.33) < 0.02 and abs(m['amp_side'] - 0.04) < 0.006


def test_neighbours_sharing_phase_are_measured_as_sharing_it():
    lines = [_helix(0.33, 0.0, 0.04, 0.0, phi=0.05 * j)[0] + np.array([0.01 * j, 0, 0]) for j in range(8)]
    m = W.measure(lines, (0.2, 0.46), order=list(range(8)))
    assert m['neighbour_phase_step_deg'] < 5
    lines = [_helix(0.33, 0.0, 0.04, 0.0, phi=1.7 * j)[0] for j in range(8)]
    assert W.measure(lines, (0.2, 0.46), order=list(range(8)))['neighbour_phase_step_deg'] > 60


def test_graph_reuses_unchanged_nodes_and_stops_at_a_failing_gauge_and_a_gate():
    with tempfile.TemporaryDirectory() as t:
        calls = []

        def make(ctx, ins, d):
            calls.append('make')
            p = os.path.join(d, 'x.json'); json.dump({'n': ctx.style.get('a.n')}, open(p, 'w'))
            return dict(outputs={'x': p})

        def judge(ctx, ins, d):
            n = json.load(open(ins['x']))['n']
            return dict(result=dict(checks=[check('n small', n, n < 5, 5)]))
        body = """schema: emberwood-hair-style/1
style: t
family: tied
a: {n: {v: %d, flag: TOOL}}
gauges: {allow_fail: %s}
"""
        sp = os.path.join(t, 's.yaml')
        g = lambda: Graph('tied', [Node('make', 'stage', make, outputs={'x': 'report'}, params=['a']),   # noqa: E731
                                   Node('judge', 'gauge', judge, inputs=['x'])], {}, [])

        def go(n, allow='{}'):
            open(sp, 'w').write(body % (n, allow))
            st = PR.load(sp)
            return run(g(), Context(st, os.path.join(t, 'out'), t, [t], lambda: {}, lambda s: None), log=lambda s: None)
        assert go(1)['status'] == 'done' and calls == ['make']
        assert go(1)['status'] == 'done' and calls == ['make']             # reused
        assert go(7)['status'] == 'failed at judge' and calls == ['make', 'make']
        m = go(7, '{"n small": "known: test"}')
        assert m['status'] == 'done' and m['nodes'][1]['status'] == 'known' and calls == ['make', 'make']


def test_both_styles_are_fully_flagged():
    for f in ('warden-ponytail.yaml', 'warden-photo-waves.yaml'):
        r = PR.check_provenance(PR.load(os.path.join(HERE, 'styles', f)))
        assert r['ok'], (f, r['unflagged'], r['bad_flag'])
