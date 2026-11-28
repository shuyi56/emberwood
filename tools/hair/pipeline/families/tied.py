"""Family 'tied': hair gathered to a tie and then falling (a ponytail; a bun or a half-up style is the same graph with
other parameters). The tools are the ponytail recipe's (docs/hair/ponytail-recipe.md), in its order:

    provenance  gauge   every number in the style flagged with its grounding
    measure     stage   the reference sheet's ratios (refmeasure) against the style's          -> reference
    field       stage   ponytail_flow.py: funnel, gravity, ear, lift, handover, free fall       -> tied_field, field_cache
    field_gauges gauge  the recipe's bar: flux, sinks, packing, funnel, rank, hairline, curl, slack, handover, loose share
    smoothness  gauge   flow_smooth on the bundle's middle line (free) and the fringe (set), before the wave
    wave        gauge   the S wave measured on the middle line against the style's wave
    field_review gate   John sees the field before any hair
    look        stage   (stages.look) ponytail_render.py: the field as fibres, clay and shaded       -> look
    bangs       stage   bangs_flow.py: the B/D blend on the forehead, the style's preset and curve  -> bangs
    bangs_gauges gauge  the bangs' own constraints (bangs.json)
    asset       stage   bangs_creator_asset.py with the finishing steps the style lists             -> asset
    asset_gauges gauge  join cover (join_cover.py), kinks after fairing

Shading lives in the asset's look and the page (hair-fidelity.js); apply the asset with
tools/face-creator/apply-hair-fidelity.py IN.html OUT.html ponytail=ASSET.json."""
import json
import os

import numpy as np

from graph import Node, check
from families import common as C

MODS = ['scalp_field', 'ponytail_flow', 'bangs_flow', 'bangs_creator_asset']

T = lambda f: (lambda v, s: f(v))          # noqa: E731


def _curve(v, style):
    """bangs.curve (and part.offset_mm) as the preset dict bangs_flow reads."""
    pre = style.get('bangs.preset', 'curtain')
    cur = dict(style.get('bangs.curve', {}) or {})
    if style.get('part.offset_mm', None) is not None:
        cur['part_mm'] = style.get('part.offset_mm')
    return {'__merge__': True, pre: cur}


# param path -> tool constant. 'set' before ponytail_flow.derive(), 'after' overrides a derived one.
VOCAB = [
    ('reference.ratios', 'ponytail_flow', 'PH', 'set', None),
    ('tie.off_skin_hh', 'ponytail_flow', 'TIE_OFF', 'set', None),
    ('tie.packing_max', 'ponytail_flow', 'PHI_TIE_MAX', 'set', None),
    ('tie.launch_fibre_deg', 'ponytail_flow', 'LAUNCH_FIBRE_DEG', 'set', None),
    ('tie.launch_max_deg', 'ponytail_flow', 'LAUNCH_MAX_DEG', 'set', None),
    ('tie.ring_mm', 'ponytail_flow', 'RING_MM', 'set', None),
    ('tie.hand_axis_mm', 'ponytail_flow', 'HAND_AXIS_MM', 'set', None),
    ('tie.converge_max_deg', 'ponytail_flow', 'THETA_MAX_DEG', 'set', None),
    ('root_lift.packing_loose', 'ponytail_flow', 'PHI_LOOSE', 'set', None),
    ('root_lift.packing_rise_mm', 'ponytail_flow', 'PHI_NEAR_MM', 'set', None),
    ('scalp.sag_g', 'ponytail_flow', 'SAG_G', 'set', None),
    ('scalp.sag_fade_rad', 'ponytail_flow', 'SAG_FADE', 'set', None),
    ('scalp.resolve', 'ponytail_flow', 'RESOLVE', 'set', None),
    ('ear.cover_hh', 'ponytail_flow', 'EAR_COVER', 'set', None),
    ('ear.offset_mm', 'ponytail_flow', 'EAR_DELTA_MM', 'set', None),
    ('ear.rim_mm', 'ponytail_flow', 'EAR_RIM_MM', 'set', None),
    ('ear.lanes_mm', 'ponytail_flow', 'EAR_LANES_MM', 'set', None),
    ('ear.ease_passes', 'ponytail_flow', 'EAR_EASE', 'set', None),
    ('fall.hair_length_hh', 'ponytail_flow', 'HAIR_LEN', 'set', None),
    ('fall.bending_length_mm', 'ponytail_flow', 'BEND_L_MM', 'set', None),
    ('fall.pi0', 'ponytail_flow', 'PI0', 'set', None),
    ('fall.rstar_mm', 'ponytail_flow', 'RSTAR_MM', 'set', None),
    ('wave.mass.length_hh', 'ponytail_flow', 'WAVE_LEN', 'after', None),
    ('wave.mass.amp_side_hh', 'ponytail_flow', 'WAVE_A', 'after', None),
    ('wave.mass.grow_in_hh', 'ponytail_flow', 'WAVE_ON', 'set', None),
    ('wave.locks.count', 'ponytail_flow', 'CURL_LOCKS', 'set', None),
    ('wave.locks.core_share', 'ponytail_flow', 'CURL_CORE', 'set', None),
    ('wave.locks.gather', 'ponytail_flow', 'CURL_GATHER', 'set', None),
    ('wave.locks.radius_mm', 'ponytail_flow', 'CURL_A_MM', 'set', None),
    ('wave.locks.phase_spread_rad', 'ponytail_flow', 'CURL_PHASE', 'set', None),
    ('wave.locks.grow_in_mm', 'ponytail_flow', 'CURL_ON_MM', 'set', None),
    ('bangs.section_rad', 'ponytail_flow', 'BANG_TH', 'set', None),
    ('bangs.depth_mm', 'ponytail_flow', 'BANG_DEPTH_MM', 'set', None),
    ('bangs.edge_z', 'ponytail_flow', 'BANG_EDGE_Z', 'after', None),
    ('bangs.fan_rad', 'ponytail_flow', 'BANG_FAN', 'set', None),
    ('bangs.stagger_mm', 'ponytail_flow', 'BANG_STAGGER_MM', 'set', None),
    ('bangs.curve', 'bangs_flow', 'PRESETS', 'set', _curve),
    ('loose_locks.locks', 'ponytail_flow', 'LOOSE', 'set', None),
    ('loose_locks.gather', 'ponytail_flow', 'LOCK_GATHER', 'set', None),
    ('loose_locks.end_y_hh', 'ponytail_flow', 'LOCK_END_Y', 'set', None),
    ('loose_locks.lift_mm', 'ponytail_flow', 'LOCK_LIFT_MM', 'set', None),
    ('loose_locks.share_bound', 'ponytail_flow', 'LOOSE_FRACTION', 'set', None),
    ('gauges.f_min_per_mm', 'ponytail_flow', 'F_MIN', 'set', None),
    ('gauges.curl_step_mm', 'ponytail_flow', 'CURL_STEP_MM', 'set', None),
    ('gauges.slack_eps', 'ponytail_flow', 'SLACK_EPS', 'set', None),
    ('gauges.cuts_mm', 'ponytail_flow', 'CUTS_MM', 'set', None),
]


def validate(style):
    """What this family's tools can and cannot do, said before anything runs."""
    errs = []
    if style.get('root_lift.model', 'stack') != 'stack':
        errs.append('root_lift.model: a tied style\'s lift is the stack\'s thickness (recipe step 1.6); "%s" is not built'
                    % style.get('root_lift.model'))
    if style.get('wave.mass.amp_out_hh', 0) not in (0, 0.0, None):
        errs.append('wave.mass.amp_out_hh: the tied mass wave is planar (the S in the side plane, recipe 1.8)')
    if style.get('wave.locks.form', 'helix') != 'helix':
        errs.append('wave.locks.form: the tail locks wind as helices')
    return errs


def binder(style, code_root):
    sets, after, order = C.bindings(style, VOCAB)
    mods = [m for m in MODS if m in order] + [m for m in order if m not in MODS]
    paths = [os.path.join(code_root, 'tools', 'hair'), os.path.join(code_root, 'tools', 'solvers', 'hair'),
             os.path.join(code_root, 'tools', 'gauges', 'hair')]
    return lambda: dict(paths=paths, set=[[m, sets[m]] for m in mods if m in sets], derive=['ponytail_flow'],
                        after=[[m, after[m]] for m in mods if m in after])


def _field(ctx, ins, d):
    out = os.path.join(d, 'out')
    argv = [ins['head'], out]
    if ins.get('refsheet'):
        argv += ['--photo', ins['refsheet']]
    rep = ctx.tool(d, 'ponytail_flow:main', argv=argv, name='ponytail_flow')
    return dict(outputs={'tied_field': out, 'field_cache': os.path.join(d, 'cache')},
                result=dict(changed_from_default=rep['changed_from_default']))


def _field_gauges(ctx, ins, d):
    G = json.load(open(os.path.join(ins['tied_field'], 'gauges.json')))
    st = ctx.style
    b = lambda k, dflt=None: st.get('gauges.' + k, dflt)     # noqa: E731
    bare = sum(v['bare'] for v in G['hairline'].values())
    err = max(abs(r['error_pct']) for r in G['balance_sheet'])
    lf = G['loose_fraction_pct']
    cs = [
        check('flux conserved per cut', round(err, 2), err <= b('flux_error_pct'), b('flux_error_pct')),
        check('hairline bare cells', bare, bare == 0, 0),
        check('tie packing', G['tie']['packing'], G['tie']['packing'] <= G['tie']['packing_max'], G['tie']['packing_max']),
        check('funnel length', G['funnel']['handover_length_mm_median'], G['funnel']['ok'], G['funnel']['min_length_mm']),
        check('rank swaps', G['rank']['swaps'], G['rank']['swaps'] <= b('rank_swaps_max', 0), b('rank_swaps_max', 0)),
        check('lines turning back', G['monotonic']['lines_turning_back'], G['monotonic']['lines_turning_back'] == 0, 0),
        check('handover gap mm', G['handover']['position_gap_mm'], G['handover']['position_gap_mm'] <= b('handover_gap_mm'),
              b('handover_gap_mm')),
        check('handover turn deg', G['handover']['direction_mismatch_deg_max'],
              G['handover']['direction_mismatch_deg_max'] <= b('handover_deg'), b('handover_deg')),
        check('loose share pct', lf['value'], lf['ok'], lf['bound']),
        check('curl radius', G['curl']['tightest_radius_mm'], G['curl']['ok'], round(1 / G['curl']['K_per_mm'], 1)),
        check('slack', G['slack']['median_pct'], G['slack']['ok'], G['slack']['eps_pct']),
    ]
    for g in G.get('photo_gaps', []):
        cs.append(check('photo gap %s' % g['name'], dict(photo=g['photo'], field=g['field']), not g['flagged']))
    return dict(result=dict(checks=cs))


def _cache_free_fall(cache):
    return C.load_pickle(C.newest(os.path.join(cache, 'free-fall-*.pkl')))


def _smoothness(ctx, ins, d):
    import flow_smooth as FS
    FF = _cache_free_fall(ins['field_cache'])
    z = np.load(os.path.join(ins['tied_field'], 'field.npz'))
    MM = 1 / 219.0
    base = z['tie'][None] + np.c_[np.zeros(len(FF['back'])), FF['back'], FF['up']] * MM
    g1 = FS.gauge([base], None, 'free')
    cs = C.smooth_checks(g1, 'bundle middle line (free hair)')
    res = dict(middle_line=g1)
    return dict(result=dict(checks=cs, **res))


def _wave(ctx, ins, d):
    import wave as W
    FF = _cache_free_fall(ins['field_cache'])
    z = np.load(os.path.join(ins['tied_field'], 'field.npz'))
    MM = 1 / 219.0
    base = z['tie'][None] + np.c_[np.zeros(len(FF['back'])), FF['back'], FF['up']] * MM
    st = ctx.style
    lam = st.get('wave.mass.length_hh')
    on = int(np.searchsorted(FF['s'], (st.get('wave.mass.grow_in_hh') or 0) / MM))
    cl = z['centre'][on:]
    # (the tail's middle line hangs in the side plane: 'out' is back from the tie, 'side' is that plane's normal; the
    # S lies across the line in the side plane, so it reads on the 'out' axis of a frame taken from the x axis)
    m = W.measure([cl], (0.6 * lam, 1.4 * lam), trends=[base[on:]], axis=(1e3, 0.0))
    tgt = dict(length_hh=lam, amp_side_hh=st.get('wave.mass.amp_side_hh'), amp_out_hh=0.0, twist_deg=None)
    cs = C.wave_checks(m, tgt, st.get('gauges.wave_tol_share', 0.1), 'mass wave')
    return dict(result=dict(checks=cs, measured=m))


def _look(ctx, ins, d):
    out = os.path.join(d, 'out')
    os.makedirs(out, exist_ok=True)
    link = os.path.join(d, 'cache')
    if not os.path.exists(link):
        os.symlink(ins['field_cache'], link)
    st = ctx.style
    ctx.tool(d, 'ponytail_render:main', argv=[ins['head'], out, '--per-root', st.get('look.per_root', 1),
                                              '--size', st.get('look.size', 750)], name='ponytail_render')
    return dict(outputs={'look': os.path.join(out, 'turn-clay.png')})


def _bangs(ctx, ins, d):
    out = os.path.join(d, 'out')
    argv = [ins['head'], out, '--field-cache', ins['field_cache'], '--preset', ctx.style.get('bangs.preset', 'curtain')]
    if ins.get('bangs_refsheet'):
        argv += ['--photo', ins['bangs_refsheet']]
    ctx.tool(d, 'bangs_flow:main', argv=argv, name='bangs_flow')
    return dict(outputs={'bangs': out})


def _bangs_gauges(ctx, ins, d):
    import flow_smooth as FS
    G = json.load(open(os.path.join(ins['bangs'], 'bangs.json')))
    z = np.load(os.path.join(ins['bangs'], 'bangs-lines.npz'), allow_pickle=True)
    L = [l[~np.isnan(l[:, 0])] for l in z['lines']]
    g = FS.gauge(L, None, 'set')
    cs = C.smooth_checks(g, 'fringe (set hair)')
    for k, v in G.items():
        if isinstance(v, dict) and 'ok' in v:
            cs.append(check('bangs %s' % k, {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}, v['ok']))
    return dict(result=dict(checks=cs, smooth=g))


def _asset(ctx, ins, d):
    st = ctx.style
    out = os.path.join(d, 'asset.json')
    argv = [ins['head'], ins['bangs'], out, '--field-cache', ins['field_cache'], '--strands', st.get('finish.strands', 14000)]
    for step in st.get('finish.steps', []) or []:
        argv += ['--' + step]
    if st.get('finish.noise', None) is not None:
        argv += ['--noise', st.get('finish.noise')]
    ctx.tool(d, 'bangs_creator_asset:main', argv=argv, name='bangs_creator_asset')
    return dict(outputs={'asset': out})


def _asset_gauges(ctx, ins, d):
    import join_cover as JC
    jc = JC.gauge(ins['head'], ins['asset'], cap_mm=ctx.style.get('finish.cap_lift_mm', 0.4), log=lambda t: None)
    s = {k: v for k, v in jc.items() if k != 'slabs'}
    cs = [check('join cover (layers)', s.get('cover_min'), (s.get('cover_min') or 0) >= 1.0, 1.0),
          check('join buried share', s.get('buried_max'), (s.get('buried_max') or 0) <= 0.1, 0.1)]
    log_p = os.path.join(os.path.dirname(ins['asset']), 'bangs_creator_asset.log')
    return dict(result=dict(checks=cs, join=s, log=log_p))


def graph(style, code_root, roots):
    from graph import Graph
    errs = validate(style)
    if errs:
        raise ValueError('the tied family cannot build this style:\n  ' + '\n  '.join(errs))
    inputs = {'head': style.resolve_path(style.get('inputs.head'), roots),
              'refsheet': style.resolve_path(style.get('inputs.reference', None), roots),
              'bangs_refsheet': style.resolve_path(style.get('inputs.bangs_reference', None), roots)}
    nodes = [C.provenance_node(), C.measure_node(),
             Node('field', 'stage', _field, inputs=['head', 'refsheet'],
                  outputs={'tied_field': 'tied_field', 'field_cache': 'field_cache'},
                  params=['reference', 'tie', 'root_lift', 'scalp', 'ear', 'fall', 'wave', 'bangs', 'loose_locks',
                          'gauges', 'part', 'tools'], doc='ponytail_flow.py: the flow field, no hair'),
             Node('field_gauges', 'gauge', _field_gauges, inputs=['tied_field'], params=['gauges'],
                  doc='the recipe\'s bar on the field (step 2)'),
             Node('smoothness', 'gauge', _smoothness, inputs=['tied_field', 'field_cache'],
                  doc='flow_smooth: free hair no tighter than its bending length; set hair than its brush'),
             Node('wave', 'gauge', _wave, inputs=['tied_field', 'field_cache'], params=['wave', 'gauges'],
                  doc='the S wave on the middle line, against the style\'s'),
             C.gate_node('field', 'tied_field', ['flow-field-views.png', 'flow-F-map.png', 'flow-over-photo.png',
                                                 'hairline-gauge.png'])]
    if style.get('stages.look', False):
        nodes.append(Node('look', 'stage', _look, inputs=['head', 'field_cache'], outputs={'look': 'figure'},
                          params=['look', 'tools'], doc='the field as fibres (recipe step 3)'))
    nodes += [Node('bangs', 'stage', _bangs, inputs=['head', 'field_cache', 'bangs_refsheet'], outputs={'bangs': 'bangs'},
                   params=['bangs', 'part', 'tools'], doc='bangs_flow.py (recipe step 4)'),
              Node('bangs_gauges', 'gauge', _bangs_gauges, inputs=['bangs']),
              Node('asset', 'stage', _asset, inputs=['head', 'bangs', 'field_cache'], outputs={'asset': 'asset'},
                   params=['finish', 'tools'], doc='bangs_creator_asset.py (recipe steps 5 and 6)'),
              Node('asset_gauges', 'gauge', _asset_gauges, inputs=['head', 'asset'], params=['finish'])]
    return inputs, nodes
