"""Nodes and helpers every family shares: provenance, the reference sheet, the field gate, the smoothness and wave
gauges, and the binding of style parameters onto tool constants."""
import glob
import json
import os
import pickle

import numpy as np

from graph import Node, check, file_digest
from params import check_provenance


def bindings(style, vocab):
    """The style's values as tool constants. vocab: [(param path, module, CONST, phase, transform)], phase 'set'
    (before the module's derive()) or 'after'. The style's `tools:` section adds raw {module: {CONST: value}} for
    anything the vocabulary does not name; setting one constant twice is an error."""
    sets, after, seen, order = {}, {}, {}, []
    for path, mod, const, phase, tf in vocab:
        v = style.get(path, None)
        if v is None:
            continue
        v = tf(v, style) if tf else v
        if (mod, const) in seen:
            raise ValueError('%s.%s is set by both %s and %s' % (mod, const, seen[(mod, const)], path))
        seen[(mod, const)] = path
        (sets if phase == 'set' else after).setdefault(mod, {})[const] = v
        if mod not in order:
            order.append(mod)
    for mod, consts in (style.get('tools', {}) or {}).items():
        for const, v in consts.items():
            if (mod, const) in seen:
                raise ValueError('%s.%s is set by both %s and tools:' % (mod, const, seen[(mod, const)]))
            seen[(mod, const)] = 'tools.%s.%s' % (mod, const)
            sets.setdefault(mod, {})[const] = v
            if mod not in order:
                order.append(mod)
    return sets, after, order


def provenance_node():
    def fn(ctx, ins, d):
        r = check_provenance(ctx.style)
        json.dump(r, open(os.path.join(d, 'provenance.json'), 'w'), indent=1)
        return dict(result=dict(checks=[
            check('every number flagged', len(r['unflagged']), not r['unflagged'], 0,
                  'unflagged: %s' % r['unflagged'][:8] if r['unflagged'] else None),
            check('flags known', len(r['bad_flag']), not r['bad_flag'], 0),
            check('numbers by flag', r['by_flag'], True, note='DRAFT numbers are placeholders still to be sourced')],
            **r))
    return Node('provenance', 'gauge', fn, params=[''], doc='every number in the style names its grounding')


def _sheet_value(sheet, name, alias):
    seg = sheet.get('computed', {}).get('segments', {})
    src = alias.get(name, name)
    key, _, field = src.partition('.')
    if key in seg:
        return seg[key].get(field or 'units')
    return None


def measure_node(ratios_path='reference.ratios'):
    """The reference sheet (refmeasure, PR #89) read as ratios in head units, checked against the style's
    ratios: a style may keep a value off the sheet only with a reason (reference.kept)."""
    def fn(ctx, ins, d):
        sheet = json.load(open(ins['refsheet']))
        st = ctx.style
        tol0 = st.get('reference.tolerance', 0.005)
        tols = st.get('reference.tolerance_by_ratio', {}) or {}
        kept = st.get('reference.kept', {}) or {}
        alias = st.get('reference.sheet_names', {}) or {}
        ratios = st.get(ratios_path, {}) or {}
        rows, checks = {}, []
        for k, v in ratios.items():
            sv = _sheet_value(sheet, k, alias)
            row = dict(style=v, sheet=None if sv is None else round(float(sv), 4))
            if sv is not None and isinstance(v, (int, float)):
                tol = tols.get(k, tol0)
                gap = float(v) - float(sv)
                row['gap'] = round(gap, 4)
                ok = abs(gap) <= tol
                if not ok and k in kept:
                    row['kept'] = kept[k]
                checks.append(check('ratio %s' % k, dict(style=v, sheet=round(float(sv), 4)), ok or k in kept, tol,
                                    kept.get(k)))
            rows[k] = row
        # every parameter whose src says "ref:NAME" (or "ref:NAME*0.5", "ref:NAME*crown": times 1 + head_top_hair,
        # the photo's unit includes the hair over the crown) must be that ratio
        for p, lf in st.leaves.items():
            if isinstance(lf.src, str) and lf.src.startswith('ref:'):
                expr = lf.src[4:].split()[0]
                nm, _, fac = expr.partition('*')
                want = ratios.get(nm)
                if want is None:
                    checks.append(check('ref of %s' % p, nm, False, note='no ratio %s in the reference' % nm))
                    continue
                if fac == 'crown':
                    want = want * (1 + ratios.get('head_top_hair', 0.0))
                elif fac:
                    want = want * float(fac)
                checks.append(check('%s = ref %s' % (p, expr), dict(param=lf.v, ref=round(want, 6)),
                                    abs(float(lf.v) - float(want)) < 1e-9))
        rep = dict(unit=sheet.get('unit'), plumb=sheet.get('plumb'), unit_px=sheet.get('computed', {}).get('unit_px'),
                   ratios=rows, checks=checks)
        p = os.path.join(d, 'reference.json')
        json.dump(rep, open(p, 'w'), indent=1)
        return dict(outputs={'reference': p}, result=dict(checks=checks))
    return Node('measure', 'stage', fn, inputs=['refsheet'], outputs={'reference': 'reference'},
                params=['reference'], doc='the reference photo\'s measured ratios, against the style\'s')


def gate_node(name, field_art, figures, what='the flow field'):
    """Stops the run until John has seen the field: the style's approved.<name> must be the digest of the field's
    data (field.npz). Change anything that moves the field and the digest changes, so approval never carries over
    to a field he has not seen."""
    def fn(ctx, ins, d):
        dg = file_digest(os.path.join(ins[field_art], 'field.npz'))
        ap = (ctx.style.get('approved', {}) or {}).get(name)
        figs = [os.path.join(ins[field_art], f) for f in figures if os.path.exists(os.path.join(ins[field_art], f))]
        ok = ap == dg
        msg = ('approved (%s)' % dg) if ok else (
            'show John %s first: %s. When he approves it, add to the style file:  approved: {%s: %s}'
            % (what, ', '.join(figs), name, dg))
        return dict(result=dict(checks=[check('John has seen %s' % what, dict(field=dg, approved=ap), ok)],
                                message=msg, digest=dg, figures=figs))
    return Node('%s_review' % name, 'gate', fn, inputs=[field_art], params=['approved'],
                doc='nothing is groomed until John has seen the flow field (ponytail recipe, project rule)')


def wave_checks(m, tgt, tol, prefix):
    """m: wave.measure's result; tgt: dict(length_hh, amp_out_hh, amp_side_hh, twist_deg)."""
    out = []
    if not m.get('lines'):
        return [check('%s: lines measured' % prefix, 0, False)]
    for k, mk in (('length_hh', 'wavelength'), ('amp_out_hh', 'amp_out'), ('amp_side_hh', 'amp_side')):
        if tgt.get(k) is None:
            continue
        t = float(tgt[k]); v = float(m[mk])
        ok = abs(v - t) <= max(tol * abs(t), 0.002)
        out.append(check('%s: %s' % (prefix, k), round(v, 4), ok, dict(target=t, tol_share=tol)))
    if tgt.get('twist_deg') is not None:
        ok = abs(m['twist_deg'] - tgt['twist_deg']) <= 20.0
        out.append(check('%s: twist_deg' % prefix, round(m['twist_deg'], 1), ok, dict(target=tgt['twist_deg'], tol_deg=20)))
    if m.get('neighbour_phase_step_deg') is not None and tgt.get('neighbour_phase_step_max_deg') is not None:
        out.append(check('%s: neighbours share phase' % prefix, round(m['neighbour_phase_step_deg'], 1),
                         m['neighbour_phase_step_deg'] <= tgt['neighbour_phase_step_max_deg'],
                         tgt['neighbour_phase_step_max_deg']))
    return out


def smooth_checks(g, name):
    return [check(name, dict(radius_mm_min=g.get('radius_mm_min'), share_under=g.get('share_under_bound'),
                             reversals=g.get('lines_with_over_1_reversal')), g.get('ok', False),
                  dict(kind=g.get('kind'), bound_mm=g.get('bound_mm')))]


def newest(pattern):
    f = sorted(glob.glob(pattern), key=os.path.getmtime)
    return f[-1] if f else None


def load_pickle(p):
    return pickle.load(open(p, 'rb'))


def npz(p):
    return dict(np.load(p, allow_pickle=True))
