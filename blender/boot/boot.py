"""A standalone garment: tall leather boots over the hose on the bare mannequin. The boots are drafted in last.py from
the legs' own measurements (the foot, ankle, calf and knee) as sheets of leather lapped and stitched on a sole, so
they re-fit with the body; this module builds the hose under them (blender/hose) and supplies the leather's look and
its stitches.

    python3 blender/base/views.py --figure boot --mannequin --turn --clay --out DIR
    python3 blender/base/views.py --figure boot --mannequin --head --set VIEW=3 --out DIR

VIEW frames the --head close-up as in hose.py: 0 the legs, 1 the hips, 2 the knee, 3 the ankle; and 4 the boots, from
under the sole to over the knee."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for d in ('hose', 'vest', 'sorceress_pendant', 'shared', 'base'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
os.environ.setdefault('HEAD_GLB', '')
os.environ.setdefault('NO_DRESS', '1')
os.environ.setdefault('NO_ACC', '1')
import bpy  # noqa: E402
import female  # noqa: E402
import hose  # noqa: E402
import last  # noqa: E402

VIEW = 0
FOOT = 'footed'
MEASURES = hose.MEASURES
M_LEATHER = female.mat_def(75, 'boot leather', female.ramp(0x4a2f1f, 0.1), 0.24, surface='leather', closeup='leather')


STITCHES = False             # the stitches' rows drawn on the leather (held back until the sewing round)
STITCH = dict(
    pitch=0.15,                  # one stitch and the gap after it, in units (about 4 mm)
    length=0.62,                 # the stitch's share of the pitch
    width=0.022,                 # the thread's half width (a waxed linen thread, about 1 mm)
    colour=(0.42, 0.31, 0.17),   # waxed linen, darkened by the dubbin
)


def stitch_closeup(name, P=STITCH):
    """The stitches on the leather's 3D look: a row of waxed thread along every lapped edge, last.RATIOS['stitch_in']
    in from it, each stitch pulled a little into the leather. Drawn from the pieces' 'seam_d' (the distance from the
    stitched edge) and 'seam_u' (the distance along it)."""
    m = bpy.data.materials.get(name)
    if m is None:
        return
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    if not out.inputs['Surface'].links:
        return
    pb = out.inputs['Surface'].links[0].from_node
    if pb.type != 'BSDF_PRINCIPLED':
        return
    N = nt.nodes.new

    def math_(op, a, b=None):
        k = N('ShaderNodeMath')
        k.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                k.inputs[i].default_value = float(v)
            else:
                nt.links.new(v, k.inputs[i])
        return k.outputs[0]

    def attr(nm):
        a = N('ShaderNodeAttribute')
        a.attribute_name = nm
        return a.outputs['Fac']
    inset = last.RATIOS['stitch_in'] * float(MEASURES.get('H', 9.4))
    across = math_('DIVIDE', math_('ABSOLUTE', math_('SUBTRACT', attr('seam_d'), inset)), P['width'])
    along = math_('FRACT', math_('DIVIDE', attr('seam_u'), P['pitch']))
    # a stitch: round across the thread, and rounded off at its two ends where it goes down through the leather
    end = math_('DIVIDE', math_('ABSOLUTE', math_('SUBTRACT', along, 0.5 * P['length'])), 0.5 * P['length'])
    thread = math_('MULTIPLY', math_('LESS_THAN', across, 1.0), math_('LESS_THAN', end, 1.0))
    high = math_('MULTIPLY', thread, math_('SQRT', math_('MAXIMUM', math_('SUBTRACT', 1.0, math_('MULTIPLY', across, across)), 0.0)))
    # the groove the row is pulled into, a little wider than the thread
    groove = math_('LESS_THAN', across, 2.2)
    mix = N('ShaderNodeMix')
    mix.data_type = 'RGBA'
    nt.links.new(thread, mix.inputs['Factor'])
    base = pb.inputs['Base Color']
    if base.links:
        nt.links.new(base.links[0].from_socket, mix.inputs['A'])
    else:
        mix.inputs['A'].default_value = base.default_value
    mix.inputs['B'].default_value = (*P['colour'], 1.0)
    nt.links.new(mix.outputs['Result'], base)
    bump = N('ShaderNodeBump')
    bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = 0.9, 0.03
    nt.links.new(math_('SUBTRACT', high, math_('MULTIPLY', groove, 0.5)), bump.inputs['Height'])
    nrm = pb.inputs['Normal']
    if nrm.links:
        nt.links.new(nrm.links[0].from_socket, bump.inputs['Normal'])
    nt.links.new(bump.outputs['Normal'], nrm)


def build():
    if VIEW == 4:                # the boots alone: from under the sole to over the knee
        knee = female.HIP_Z - female.THIGH
        os.environ.setdefault('HEAD_CZ', '%.2f' % (0.5 * (knee + 1.0) - 0.8))
        os.environ.setdefault('HEAD_H', '%.2f' % (knee + 2.6))
    hose.VIEW, hose.FOOT = min(VIEW, 3), FOOT
    res = hose.build()
    body = bpy.data.objects['Skin body']
    rig = bpy.data.objects.get('Rig')
    ho = bpy.data.objects['Hose']
    leather = female.mat(M_LEATHER)
    boots = last.build_boots(body, hose.MEASURES, rig=rig, material=leather)
    if STITCHES:
        stitch_closeup(leather.name)
    if os.environ.get('BOOT_PATTERN'):       # (the flat pattern sheet, saved as an image at this path)
        import pattern
        pattern.pairs(last.PIECES['R'])
        pattern.sheet(last.PIECES['R'], os.environ['BOOT_PATTERN'], title='Tall boot, right foot: the pattern laid flat')
    if os.environ.get('BOOT_CHECK'):
        sc = bpy.context.scene
        for f in [int(x) for x in os.environ['BOOT_CHECK'].split(',')]:
            sc.frame_set(f)
            for s in 'RL':
                last.clearance(boots[s], ho)
        sc.frame_set(0)
    bpy.context.view_layer.update()
    return res
