"""Family 'loose': hair that hangs from the head, combed from its roots, with a part, a fringe, face framing and
falls that wave (the Warden's long photo waves; any loose, layered style). The tools are the photo-waves build
(tools/hair/photo_waves.py and wavy_flow.py, PR #99), in the recipe's order:

    provenance   gauge  every number flagged
    measure      stage  the reference sheet's ratios against the style's (refmeasure)            -> reference
    field        stage  photo_waves.field2: the combed head lines, the fringe sheet, the falls   -> loose_field
    field_gauges gauge  rank, lines turning up, smoothness (flow_smooth), widths and ends against the photo
    wave         gauge  the falls' wave against the style's: wavelength, two amplitudes, twist, neighbours in phase
    field_review gate   John sees the field before any hair
    hair         stage  wavy_flow.py photo mode: fibres, wave locks (Kirchhoff rods), finish, asset  -> fibres_dir, asset
    hair_gauges  gauge  hairline, join cover, kinks (wavy.json)

The hair stage follows PR #99 as it changes (lock_fall and photo_render.py are still being written there); the
field is approved (v14) and reproduces exactly."""
import json
import math
import os

import numpy as np

from graph import Node, check
from families import common as C

MODS = ['scalp_field', 'photo_waves', 'wavy_flow']


def _wave(v, style):
    w = style.get('wave')
    return dict(start_z=w['start_z'], full_z=w['full_z'], lam_hh=w['length_hh'], amp_hh=w['amp_side_hh'])


def _env(v, style):
    """The lift's outer surface over the skull (top, temple), from the heights or from the leaving angle.
    A hair leaves its root at an angle and turns over at one constant bend onto that surface, landing tangent to it:
    height T (1 - (1 - s/L)^2) over the run L (photo_waves), so the leaving angle is atan(2T / L) and the bend radius
    L^2 / 2T. Given emergence_deg and relax_mm, T = L tan(angle) / 2 at the top, and the temple keeps temple_share."""
    if 'outer_mm' in v:
        return (v['outer_mm']['top'], v['outer_mm']['temple'])
    T = v['relax_mm'] * math.tan(math.radians(v['emergence_deg'])) / 2
    return (T, T * v.get('temple_share', 23.0 / 27.0))


VOCAB = [
    ('hairs.on_scalp', 'photo_waves', 'N_HAIRS', 'set', None),
    ('part.push', 'photo_waves', 'PART_PUSH', 'set', None),
    ('root_lift', 'photo_waves', 'ENV_MM', 'set', _env),
    ('root_lift.relax_mm', 'photo_waves', 'ROOT_RISE_MM', 'set', None),
    ('root_lift.base_mm', 'photo_waves', 'LIFT_MM', 'set', None),
    ('root_lift.per_mm', 'photo_waves', 'LIFT_PER_MM', 'set', None),
    ('root_lift.top_partner_mm', 'photo_waves', 'TOP_LIFT_MM', 'set', None),
    ('fall.hem_z', 'photo_waves', 'HEM_Z', 'set', None),
    ('fall.top_shorter_hh', 'photo_waves', 'TOP_SHORTER', 'set', None),
    ('fall.grade_hh', 'photo_waves', 'GRADE', 'set', None),
    ('fall.ragged_hh', 'photo_waves', 'RAGGED', 'set', None),
    ('fall.take', 'photo_waves', 'TAKE', 'set', None),
    ('fall.flare_hh', 'photo_waves', 'FLARE', 'set', None),
    ('fall.back_sweep', 'photo_waves', 'BACK_SWEEP', 'set', None),
    ('fall.sweep_in_mm', 'photo_waves', 'SWEEP_IN_MM', 'set', None),
    ('fall.carry_hh', 'photo_waves', 'FALL_CARRY', 'set', None),
    ('fall.split_rad', 'photo_waves', 'FALL_SPLIT', 'set', None),
    ('fall.front_band_rad', 'photo_waves', 'FRONT_BAND', 'set', None),
    ('fall.back_band_rad', 'photo_waves', 'BACK_BAND', 'set', None),
    ('fall.fair_mm', 'photo_waves', 'FAIR_MM', 'set', None),
    ('fall.gather_z', 'photo_waves', 'GATHER_Z', 'set', None),
    ('fall.face_keep_rad', 'photo_waves', 'FACE_KEEP', 'set', None),
    ('fall.face_keep_z', 'photo_waves', 'FACE_KEEP_Z', 'set', None),
    ('fall.stack_mm', 'photo_waves', 'STACK_MM', 'set', None),
    ('fall.body_at_z', 'photo_waves', 'BODY_AT_Z', 'set', None),
    ('framing.section_rad', 'photo_waves', 'FRAME_RAD', 'set', None),
    ('framing.depth_hh', 'photo_waves', 'FRAME_DEPTH', 'set', None),
    ('framing.end_z', 'photo_waves', 'FRAME_END', 'set', None),
    ('wave', 'photo_waves', 'WAVE', 'set', _wave),
    ('wave.locks_per_side', 'photo_waves', 'LOCKS_PER_SIDE', 'set', None),
    ('bangs.section_rad', 'photo_waves', 'FRINGE_RAD', 'set', None),
    ('bangs.depth_hh', 'photo_waves', 'FRINGE_DEPTH', 'set', None),
    ('bangs.hang_z', 'photo_waves', 'FRINGE_HANG_Z', 'set', None),
    ('bangs.arch_mm', 'photo_waves', 'FRINGE_ARCH_MM', 'set', None),
    ('bangs.push', 'photo_waves', 'FRINGE_PUSH', 'set', None),
    ('bangs.lift_mm', 'photo_waves', 'FRINGE_LIFT_MM', 'set', None),
    ('bangs.feather_hh', 'photo_waves', 'FRINGE_FEATHER', 'set', None),
    ('bangs.curve.end', 'photo_waves', 'BANG_END', 'set', None),
    ('bangs.curve.handles', 'photo_waves', 'BANG_S', 'set', None),
    ('bangs.curve.span', 'photo_waves', 'BANG_SPAN', 'set', None),
    ('bangs.curve.fair_nodes', 'photo_waves', 'BANG_FAIR', 'set', None),
    ('locks.volume_mm', 'photo_waves', 'VOLUME_MM', 'set', None),
    ('locks.section_blend', 'wavy_flow', 'SECTION_BLEND', 'set', None),
    ('locks.blend_radius_mm', 'wavy_flow', 'BLEND_R_MM', 'set', None),
    ('locks.flatness', 'wavy_flow', 'FLATNESS', 'set', None),
    ('locks.end_straight_mm', 'wavy_flow', 'END_STRAIGHT_MM', 'set', None),
    ('locks.rest_gain', 'wavy_flow', 'REST_GAIN', 'set', None),
    ('finish.cap_lift_mm', 'wavy_flow', 'CAP_LIFT_MM', 'set', None),
    ('shading.hue', 'wavy_flow', 'LOOK_HUE', 'set', None),
    ('shading.light_floor', 'wavy_flow', 'LOOK_LIGHT_FLOOR', 'set', None),
    ('shading.inner_dark', 'wavy_flow', 'LOOK_INNER_DARK', 'set', None),
    ('shading.mid_dark', 'wavy_flow', 'LOOK_MID_DARK', 'set', None),
    ('shading.tone', 'wavy_flow', 'LOOK_TONE', 'set', None),
]


def validate(style):
    errs = []
    if style.get('part.line', 'centre') != 'centre':
        errs.append('part.line: photo_waves parts on the midline only ("%s" is not built yet)' % style.get('part.line'))
    w = style.get('wave', {})
    if w.get('amp_out_hh') is not None and abs(w['amp_out_hh'] - 0.5 * w['amp_side_hh']) > 1e-9:
        errs.append('wave.amp_out_hh: field2 lays the out wave at half the side wave (0.5 x amp_side_hh)')
    if w.get('twist_deg') not in (None, 90, 90.0):
        errs.append('wave.twist_deg: field2 lays the out wave a quarter wave on (90 deg)')
    return errs


def binder(style, code_root):
    sets, after, order = C.bindings(style, VOCAB)
    mods = [m for m in MODS if m in order] + [m for m in order if m not in MODS]
    paths = [os.path.join(code_root, 'tools', 'hair'), os.path.join(code_root, 'tools', 'solvers', 'hair'),
             os.path.join(code_root, 'tools', 'gauges', 'hair')]
    return lambda: dict(paths=paths, set=[[m, sets[m]] for m in mods if m in sets], derive=[],
                        after=[[m, after[m]] for m in mods if m in after])


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _field(ctx, ins, d):
    st = ctx.style
    out = os.path.join(d, 'out')
    rep = ctx.tool(d, os.path.join(HERE, 'drivers.py') + ':loose_field',
                   kwargs=dict(head=ins['head'], out=out, n=st.get('field.lines', 8000), seed=st.get('field.seed', 11),
                               body=st.get('field.body', 'mannequin')), name='field2')
    return dict(outputs={'loose_field': out}, result=dict(changed_from_default=rep['changed_from_default']))


def _field_gauges(ctx, ins, d):
    G = json.load(open(os.path.join(ins['loose_field'], 'gauges.json')))
    st = ctx.style
    cs = []
    lim = st.get('gauges.rank_moves_share', 0.03)
    for k, v in G.items():
        if k.startswith('rank_moves'):
            a, b = (int(x) for x in v.split(' of '))
            cs.append(check(k.replace('_over_3pct', ''), v, a <= lim * b, '%.0f%% of lines' % (100 * lim)))
    rl = st.get('root_lift')
    T, L = _env(rl, st)[0], rl['relax_mm']
    cs.append(check('root lift (outer layer)', dict(leaving_deg=round(math.degrees(math.atan2(2 * T, L)), 1),
                                                   bend_radius_mm=round(L * L / (2 * T), 1), height_mm=round(T, 1)), True,
                    note='derived from root_lift: atan(2T / L), L^2 / 2T'))
    cs.append(check('lines turning up', G['lines_turning_up'], G['lines_turning_up'] == 0, 0))
    cs += C.smooth_checks(G['smooth_falls'], 'falls (free hair)')
    cs += C.smooth_checks(G['smooth_fringe'], 'fringe (set hair)')
    tol = st.get('gauges.width_tol_hh', 0.03)
    ref = st.get('reference.ratios', {}) or {}
    for z, w in G['half_width'].items():
        k = 'half_width_at_%s' % z
        if k in ref:
            cs.append(check('half width at z %s' % z, dict(field=w, photo=ref[k]), abs(w - ref[k]) <= tol, tol))
    if 'front_inner_edge_at_-0.15' in ref:
        v = G['front_inner_edge_at_-0.15']
        cs.append(check('front fall inner edge', dict(field=v, photo=ref['front_inner_edge_at_-0.15']),
                        abs(v - ref['front_inner_edge_at_-0.15']) <= tol, tol))
    if 'ends_z' in ref:
        cs.append(check('ends (5th pct z)', dict(field=G['ends_z_p5'], photo=ref['ends_z']),
                        abs(G['ends_z_p5'] - ref['ends_z']) <= tol, tol))
    return dict(result=dict(checks=cs, basins=G['basin_share_pct']))


def _wave_gauge(ctx, ins, d):
    import wave as W
    import scalp_field as SF
    z = np.load(os.path.join(ins['loose_field'], 'field.npz'))
    st = ctx.style
    w = st.get('wave')
    lines, trends, th = [], [], []
    LL = z['lines']                                  # (once: indexing a compressed npz decompresses it every time)
    LB = np.load(os.path.join(ins['loose_field'], 'field-course.npz'))['lines']
    for i in np.where(z['basin'] >= 2)[0][::st.get('gauges.wave_every', 4)]:
        p, b = LL[i], LB[i]
        k = ~np.isnan(p[:, 0]) & ~np.isnan(b[:, 0])
        p, b = p[k], b[k]
        sel = b[:, 2] < w['full_z']                   # (where the wave is in full)
        if sel.sum() > 20:
            lines.append(p[sel]); trends.append(b[sel])
            th.append(float(SF.theta(p[sel][:1])[0]))
    order = np.argsort(th)
    lam = w['length_hh']
    m = W.measure(lines, (0.6 * lam, 1.4 * lam), axis=tuple(SF.AXIS[:2]), trends=trends, order=order, along='height')
    tgt = dict(length_hh=lam, amp_side_hh=w['amp_side_hh'], amp_out_hh=w.get('amp_out_hh'), twist_deg=w.get('twist_deg'),
               neighbour_phase_step_max_deg=st.get('gauges.neighbour_phase_step_deg', 30))
    cs = C.wave_checks(m, tgt, st.get('gauges.wave_tol_share', 0.25), 'fall wave')
    return dict(result=dict(checks=cs, measured=m))


def _hair(ctx, ins, d):
    st = ctx.style
    out = os.path.join(d, 'out')
    argv = [ins['head'], 'photo', 'none', out, '--fibres', st.get('finish.fibres', 24000),
            '--strands', st.get('finish.strands', 16000)]
    if st.get('finish.noise', None) is not None:
        argv += ['--noise', st.get('finish.noise')]
    ctx.tool(d, 'wavy_flow:main', argv=argv, name='wavy_flow')
    return dict(outputs={'fibres_dir': out, 'asset': os.path.join(out, 'asset-wavyFull.json')})


def _hair_gauges(ctx, ins, d):
    R = json.load(open(os.path.join(ins['fibres_dir'], 'wavy.json')))
    bare = sum(v['bare'] for v in R['hairline'].values())
    kk = R['kinks']['faired']
    cs = [check('hairline bare cells', bare, bare == 0, 0),
          check('join cover (layers)', R['join'].get('cover_min'), (R['join'].get('cover_min') or 0) >= 1.0, 1.0)]
    return dict(result=dict(checks=cs, kinks=kk))


def graph(style, code_root, roots):
    errs = validate(style)
    if errs:
        raise ValueError('the loose family cannot build this style:\n  ' + '\n  '.join(errs))
    inputs = {'head': style.resolve_path(style.get('inputs.head'), roots),
              'refsheet': style.resolve_path(style.get('inputs.reference', None), roots)}
    nodes = [C.provenance_node()]
    if inputs['refsheet']:
        nodes.append(C.measure_node())
    else:
        def no_sheet(ctx, ins, d):
            return dict(result=dict(checks=[check('reference measured with refmeasure', None, False,
                                                  note='no sheet: the ratios are typed into the style by hand')]))
        nodes.append(Node('measure', 'gauge', no_sheet, params=['reference']))
    nodes += [Node('field', 'stage', _field, inputs=['head'], outputs={'loose_field': 'loose_field'},
                   params=['hairs', 'part', 'root_lift', 'fall', 'framing', 'wave', 'bangs', 'locks', 'field', 'tools'],
                   doc='photo_waves.field2: the flow field, no hair'),
              Node('field_gauges', 'gauge', _field_gauges, inputs=['loose_field'], params=['gauges', 'reference']),
              Node('wave', 'gauge', _wave_gauge, inputs=['loose_field'], params=['wave', 'gauges']),
              C.gate_node('field', 'loose_field', ['flow-field.png', 'flow-field-face.png']),
              Node('hair', 'stage', _hair, inputs=['head', 'loose_field'], outputs={'fibres_dir': 'fibres_dir', 'asset': 'asset'},
                   params=['locks', 'finish', 'shading', 'tools'], doc='wavy_flow.py photo mode (PR #99)'),
              Node('hair_gauges', 'gauge', _hair_gauges, inputs=['fibres_dir'])]
    return inputs, nodes
