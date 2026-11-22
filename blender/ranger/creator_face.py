"""The Face Creator's face in Blender: the creator's own head, eyes and ears, with every one of its shaders rebuilt for
Cycles from the creator's own maps and numbers, so a Blender render shows the face as the live creator builds it.

The input is a dump of the creator (tools/face-creator/dump-face.cjs): the meshes and the maps that the glTF export
loses (the pore relief, the lip layer map, the roughness zones, the non-skin mask). glTF carries only a plain material
and its painted colour, so a creator head brought in through the game's glb has none of the work below.

What is ported, and from where (the research behind each number is in the named file):
  - the head's shape: the creator's mesh as is (the eye sockets of blender/shared/orbit.py, the nose builder's
    hybrid grid, the mouth), with its own normals
  - the epidermis's colour: the creator's painted face (tone, flush, freckles, brows, lashes, the Ranger's mulberry
    shadow and tapered liner: apply-eye-makeup.py)
  - the layered skin: skin_layers.skin_layers on the creator's rest positions, which is the shader the creator itself
    ports (its SKIN_LAYERS are skin_layers.py's numbers, its noise Blender's), with its thin-skin and freckle zones
  - the pores: the creator's pore relief (apply-pores.py: 64 a square centimetre on the cheek, 0.16 mm2 each,
    Flament et al. 2015) as a bump on the skin, at the pit depth the creator measures them by (0.13 mm, Maia Campos
    et al. 2019). Blender's own sowing is turned off, and so is the scatter's hold on the pits, as in the creator:
    Cycles' random-walk scatter blurs the pit's diffuse light per colour by itself, which the creator only imitates
    (its SK_DW per-colour normals and SK_AO occlusion)
  - the roughness: the creator's zones (the oil on the nose, the wet waterline) as their difference from its plain
    skin's 0.56, on top of skin_layers' patches, as the creator's shader does
  - the lips in two layers: the creator's lip map (red: the lip, green: how wet, blue: the lip print) driving the
    lips thread's two-layer surface (skin_layers' LIP_ numbers: a smooth, short-scattering under layer and a coat
    with the grooves in its own normal), with the creator's own coat weights and roughness for this face
  - the eye whites: the creator's painting (cream base, cream and cool patches, warm shade: apply-sclera-milk.py;
    the vessels from the corners: apply-sclera-veins.py) on the eyeball, with the tear film as the coat. The creator's
    warm self-glow stands in for light scattered inside the sclera; here the white scatters for real
  - left out on purpose: the creator's baked cavity shade (the head's vertex colour; apply-shade-fix.py) and its
    wrapped light. Both stand in for occlusion and scatter, which Cycles traces; multiplied in, the shade would be
    counted twice (CAVITY = True puts it back, for a comparison)
"""
import json
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'base'))
import skin_layers as SL  # noqa: E402

UNIT_MM = 27.0                  # one model unit (the creator's HH = 8.6 units to the head's height; skin_layers.py)
# the pore relief. The creator's tile stands at about 0.75 on bare skin and falls to about 0.27 at a pit's floor (its
# own pit ramp, (0.6 - h) * 3, is full there); a pit is 0.13 mm deep (Maia Campos et al. 2019, Front Physiol 10:254,
# the creator's SK_PIT.d). So a unit of the tile is 0.13 / 0.48 mm
PORE_DEPTH_MM = 0.13
PORE_TILE_SPAN = 0.48
PORE_DIST = PORE_DEPTH_MM / PORE_TILE_SPAN / UNIT_MM
# the eye: the cornea's index (Le Grand's schematic eye, 1.376), and the tear film as the coat
CORNEA_IOR = 1.376
# the sclera's scatter. Sclera scatters far more than it absorbs, forward (g = 0.9: Hammer et al. 1995, Phys Med Biol
# 40:963). EXPERIMENTAL: the reach (about 0.3 mm, red furthest) is set to the creator's glow, not measured
SCLERA_SSS = dict(weight=1.0, radius=(1.0, 0.7, 0.5), scale=0.3 / UNIT_MM, anisotropy=0.9)
CAVITY = False
# looks: 'full'; 'clay' (one grey albedo, the relief and the lips' coat kept); 'zones' (each part of the face's
# surface in its own flat colour: layered skin, lips, painted brows and lashes, eye make-up, eye white, iris)
ZONE_COLOURS = {'skin': (0.78, 0.78, 0.78), 'lips': (0.95, 0.8, 0.1), 'paint': (0.1, 0.75, 0.85),
                'makeup': (0.85, 0.1, 0.65), 'white': (0.95, 0.95, 0.9), 'iris': (0.15, 0.55, 0.2), 'ear': (0.6, 0.6, 0.7)}


def load(dump):
    D = json.load(open(os.path.join(dump, 'face.json')))
    D['dir'] = dump
    return D


def image(D, key_or_file, colour='sRGB'):
    f = D['files'].get(key_or_file, key_or_file)
    img = bpy.data.images.load(os.path.join(D['dir'], f), check_existing=True)
    img.colorspace_settings.name = colour
    return img


def build_mesh(name, m, coll):
    V = np.array(m['verts'], dtype=np.float64).reshape(-1, 3)
    I = np.array(m['index'], dtype=np.int64).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], I.tolist())
    me.update()
    loops = np.zeros(len(me.loops), dtype=np.int64)
    me.loops.foreach_get('vertex_index', loops)
    for key, lname in (('uv', 'UVMap'), ('uv1', 'UVWrap')):
        if key in m:
            uv = np.array(m[key]).reshape(-1, 2)
            lay = me.uv_layers.new(name=lname)
            lay.data.foreach_set('uv', uv[loops].ravel())
    me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), dtype=bool))
    if 'normals' in m:
        me.normals_split_custom_set_from_vertices(np.array(m['normals']).reshape(-1, 3).tolist())
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    return o, V, me


# ------------------------------------------------------------------------------------------------ node helpers
class Nodes:
    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.L = self.nt.links.new

    def new(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def math(self, op, a, b=None, c=None):
        n = self.new('ShaderNodeMath', operation=op)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                self.L(v, n.inputs[i])
        return n.outputs[0]

    def maprange(self, v, a, b, c, d, smooth=False):
        n = self.new('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        self.L(v, n.inputs['Value'])
        for k, x in (('From Min', a), ('From Max', b), ('To Min', c), ('To Max', d)):
            n.inputs[k].default_value = x
        return n.outputs['Result']

    def tex(self, img, uv='UVMap', mapping=None, interp='Cubic', ext='EXTEND'):
        t = self.new('ShaderNodeTexImage', interpolation=interp, extension=ext)
        t.image = img
        u = self.new('ShaderNodeUVMap', uv_map=uv)
        vec = u.outputs['UV']
        if mapping:
            mp = self.new('ShaderNodeMapping', vector_type='POINT')
            self.L(vec, mp.inputs['Vector'])
            mp.inputs['Scale'].default_value = (mapping['repeat'][0], mapping['repeat'][1], 1)
            mp.inputs['Location'].default_value = (mapping['offset'][0], mapping['offset'][1], 0)
            vec = mp.outputs['Vector']
        self.L(vec, t.inputs['Vector'])
        return t

    def sep(self, socket):
        s = self.new('ShaderNodeSeparateColor')
        self.L(socket, s.inputs['Color'])
        return s.outputs

    def bump(self, height, dist, normal=None, strength=1.0):
        b = self.new('ShaderNodeBump')
        b.inputs['Strength'].default_value = strength
        b.inputs['Distance'].default_value = dist
        self.L(height, b.inputs['Height'])
        if normal is not None:
            self.L(normal, b.inputs['Normal'])
        return b.outputs['Normal']

    def mixshader(self, fac, a, b):
        m = self.new('ShaderNodeMixShader')
        self.L(fac, m.inputs['Fac'])
        self.L(a, m.inputs[1])
        self.L(b, m.inputs[2])
        return m.outputs['Shader']


def _flat(name, colour, rough=0.5):
    m = bpy.data.materials.new(name)
    N = Nodes(m)
    out = N.new('ShaderNodeOutputMaterial')
    p = N.new('ShaderNodeBsdfPrincipled')
    p.inputs['Base Color'].default_value = (*colour, 1)
    p.inputs['Roughness'].default_value = rough
    N.L(p.outputs['BSDF'], out.inputs['Surface'])
    return m, N, p, out


# ------------------------------------------------------------------------------------------------ the skin
def head_material(D, look='full'):
    """the head's one surface: the layered skin, the painted parts (brows, lashes, waterline: the mask's green) on the
    plain scattering skin, and the lips in two layers, all on the creator's maps"""
    m = bpy.data.materials.new('creator skin ' + look)
    N = Nodes(m)
    L = N.L
    out = N.new('ShaderNodeOutputMaterial')
    face = N.tex(image(D, 'face'))
    mask = N.sep(N.tex(image(D, 'mask', 'Non-Color')).outputs['Color'])
    lip = N.sep(N.tex(image(D, 'lip', 'Non-Color')).outputs['Color'])
    lip_r, wet, lf = lip[0], lip[1], lip[2]
    rough_map = N.sep(N.tex(image(D, 'rough', 'Non-Color')).outputs['Color'])[0]
    pu = D['poresUV']
    pores = N.sep(N.tex(image(D, 'pores', 'Non-Color'), 'UVWrap', mapping=pu, ext='REPEAT').outputs['Color'])[0]
    tone = face.outputs['Color']
    if CAVITY:
        occ = N.new('ShaderNodeVertexColor', layer_name='occ')
        mul = N.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        mul.inputs['Factor'].default_value = 1.0
        L(tone, mul.inputs['A'])
        L(occ.outputs['Color'], mul.inputs['B'])
        tone = mul.outputs['Result']
    if look == 'clay':
        grey = N.new('ShaderNodeRGB')
        grey.outputs[0].default_value = (0.42, 0.42, 0.42, 1)
        tone = grey.outputs[0]
    pore_n = N.bump(pores, PORE_DIST)

    # the layered skin (skin_layers writes into an output node: give it one, then take its shader). Blender's own pore
    # sowing and the pit's scatter hold are off: the pores are the creator's (above)
    keep = SL.PORES, SL.FURROWS, SL.FRECKLES
    SL.PORES, SL.FURROWS, SL.FRECKLES = 0.0, 0.0, 0.0     # (the creator's freckles are in its painted tone)
    tmp = N.new('ShaderNodeOutputMaterial', is_active_output=False)
    SL.skin_layers(N.nt, tmp, 'all', tone)
    SL.PORES, SL.FURROWS, SL.FRECKLES = keep
    layered = tmp.inputs['Surface'].links[0].from_socket
    sb = layered.node
    N.nt.nodes.remove(tmp)
    for n in list(N.nt.nodes):        # (its bump, now flat: the creator's pore relief in its place)
        if n.type == 'BUMP' and n.outputs['Normal'].links and n.outputs['Normal'].links[0].to_node == sb:
            N.nt.nodes.remove(n)
    L(pore_n, sb.inputs['Normal'])
    # (the creator's roughness zones, as their difference from its plain skin's 0.56)
    r_old = sb.inputs['Roughness'].links[0].from_socket
    L(N.math('MAXIMUM', N.math('ADD', r_old, N.math('SUBTRACT', rough_map, 0.56)), 0.05), sb.inputs['Roughness'])
    if look == 'clay':
        sb.inputs['Base Color'].links and N.nt.links.remove(sb.inputs['Base Color'].links[0])
        sb.inputs['Base Color'].default_value = (0.42, 0.42, 0.42, 1)

    # the painted parts: the creator keeps them out of the layers, on its plain skin (which still scatters)
    plain = N.new('ShaderNodeBsdfPrincipled')
    L(tone, plain.inputs['Base Color'])
    plain.subsurface_method = 'RANDOM_WALK_SKIN'
    plain.inputs['Subsurface Weight'].default_value = 1.0
    plain.inputs['Subsurface Radius'].default_value = (1.0, 0.36, 0.18)
    plain.inputs['Subsurface Scale'].default_value = SL.SSS_SCALE
    plain.inputs['IOR'].default_value = 1.4
    plain.inputs['Specular IOR Level'].default_value = SL.SPECULAR
    L(N.math('MAXIMUM', rough_map, 0.05), plain.inputs['Roughness'])
    L(pore_n, plain.inputs['Normal'])

    # the lips: the under layer and the coat (skin_layers' LIP_ numbers), on the creator's lip map
    lp = N.new('ShaderNodeBsdfPrincipled')
    L(tone, lp.inputs['Base Color'])
    L(N.maprange(wet, 0, 1, *SL.LIP_BASE_ROUGH), lp.inputs['Roughness'])
    lp.inputs['IOR'].default_value = 1.4
    lp.inputs['Specular IOR Level'].default_value = SL.SPECULAR * SL.LIP_BASE_SPEC
    lp.subsurface_method = 'BURLEY'
    lp.inputs['Subsurface Weight'].default_value = SL.LIP_SSS_WEIGHT
    lp.inputs['Subsurface Radius'].default_value = SL.LIP_SSS_RADIUS
    lp.inputs['Subsurface Scale'].default_value = SL.LIP_SSS_SCALE
    L(N.bump(N.math('MULTIPLY', lf, -SL.LIP_BASE_FURROW), SL.LIP_COAT_BUMP, pore_n), lp.inputs['Normal'])
    lc = D['uniforms']['lipCoat']      # (coat weight dry and wet, coat roughness dry and wet: this face's)
    L(N.maprange(wet, 0, 1, lc[0], lc[1]), lp.inputs['Coat Weight'])
    L(N.math('MAXIMUM', N.math('MULTIPLY_ADD', lf, 0.3, N.maprange(wet, 0, 1, lc[2], lc[3])), 0.06), lp.inputs['Coat Roughness'])
    lp.inputs['Coat IOR'].default_value = 1.45
    L(N.bump(N.math('MULTIPLY', lf, -1.0), SL.LIP_COAT_BUMP), lp.inputs['Coat Normal'])
    if look == 'clay':
        for b in (plain, lp):
            if b.inputs['Base Color'].links:
                N.nt.links.remove(b.inputs['Base Color'].links[0])
            b.inputs['Base Color'].default_value = (0.42, 0.42, 0.42, 1)

    painted = N.mixshader(lip_r, plain.outputs['BSDF'], lp.outputs['BSDF'])
    surf = N.mixshader(mask[1], layered, painted)
    L(surf, out.inputs['Surface'])
    return m


def zones_head(D, N, out, mask, lip_r):
    """the colour-coded pass: which surface each part of the head gets. Eye make-up is where the painted face is
    darker than the creator's tone round the eyes and the mask is skin (the shadow wash) or paint (the liner)"""
    L = N.L

    def rgb(c):
        n = N.new('ShaderNodeRGB')
        n.outputs[0].default_value = (*c, 1)
        return n.outputs[0]
    face = N.tex(image(D, 'face'))
    # the eye shadow's wash: the mulberry hue (#6e4652) stands apart from the skin in red minus green over blue
    fc = N.sep(face.outputs['Color'])
    purple = N.maprange(N.math('SUBTRACT', fc[2], N.math('MULTIPLY', fc[1], 1.0)), -0.02, 0.06, 0, 1)
    c = N.new('ShaderNodeMix', data_type='RGBA')
    L(purple, c.inputs['Factor'])
    L(rgb(ZONE_COLOURS['skin']), c.inputs['A'])
    L(rgb(ZONE_COLOURS['makeup']), c.inputs['B'])
    c2 = N.new('ShaderNodeMix', data_type='RGBA')
    L(mask[1], c2.inputs['Factor'])
    L(c.outputs['Result'], c2.inputs['A'])
    L(rgb(ZONE_COLOURS['paint']), c2.inputs['B'])
    c3 = N.new('ShaderNodeMix', data_type='RGBA')
    L(lip_r, c3.inputs['Factor'])
    L(c2.outputs['Result'], c3.inputs['A'])
    L(rgb(ZONE_COLOURS['lips']), c3.inputs['B'])
    p = N.new('ShaderNodeBsdfDiffuse')
    L(c3.outputs['Result'], p.inputs['Color'])
    L(p.outputs['BSDF'], out.inputs['Surface'])


# ------------------------------------------------------------------------------------------------ the eyes
def eye_material(D, k, look='full'):
    e = D['eyes'][k]
    mt = e['mat']
    m = bpy.data.materials.new('creator eye %d %s' % (k, look))
    N = Nodes(m)
    L = N.L
    out = N.new('ShaderNodeOutputMaterial')
    t = N.tex(image(D, e['map']))
    lum = N.new('ShaderNodeRGBToBW')
    L(t.outputs['Color'], lum.inputs['Color'])
    white = N.maprange(lum.outputs['Val'], 0.22, 0.6, 0, 1, smooth=True)     # (the creator's own mask of its glow)
    if look == 'zones':
        p = N.new('ShaderNodeBsdfDiffuse')
        c = N.new('ShaderNodeMix', data_type='RGBA')
        L(white, c.inputs['Factor'])
        c.inputs['A'].default_value = (*ZONE_COLOURS['iris'], 1)
        c.inputs['B'].default_value = (*ZONE_COLOURS['white'], 1)
        L(c.outputs['Result'], p.inputs['Color'])
        L(p.outputs['BSDF'], out.inputs['Surface'])
        return m
    p = N.new('ShaderNodeBsdfPrincipled')
    if look == 'clay':
        p.inputs['Base Color'].default_value = (0.42, 0.42, 0.42, 1)
    else:
        L(t.outputs['Color'], p.inputs['Base Color'])
    p.inputs['Roughness'].default_value = mt['roughness']
    p.inputs['IOR'].default_value = mt['ior']
    p.inputs['Specular IOR Level'].default_value = 0.5 * mt['specularIntensity']      # (three.js scales F0 by it; Blender's 0.5 is F0 as is)
    p.inputs['Coat Weight'].default_value = mt['clearcoat']
    p.inputs['Coat Roughness'].default_value = mt['clearcoatRoughness']
    p.inputs['Coat IOR'].default_value = CORNEA_IOR
    p.subsurface_method = 'RANDOM_WALK'
    L(N.math('MULTIPLY', white, SCLERA_SSS['weight']), p.inputs['Subsurface Weight'])
    p.inputs['Subsurface Radius'].default_value = SCLERA_SSS['radius']
    p.inputs['Subsurface Scale'].default_value = SCLERA_SSS['scale']
    p.inputs['Subsurface Anisotropy'].default_value = SCLERA_SSS['anisotropy']
    L(p.outputs['BSDF'], out.inputs['Surface'])
    return m


def ear_material(D, e, look='full'):
    if look == 'zones':
        return _flat('creator ear zones', ZONE_COLOURS['ear'])[0]
    m, N, p, out = _flat('creator ear ' + look, (0.42, 0.42, 0.42) if look == 'clay' else (1, 1, 1), e['mat']['roughness'])
    if look != 'clay':
        vc = N.new('ShaderNodeVertexColor', layer_name='occ')
        N.L(vc.outputs['Color'], p.inputs['Base Color'])
    p.subsurface_method = 'RANDOM_WALK_SKIN'
    p.inputs['Subsurface Weight'].default_value = 1.0
    p.inputs['Subsurface Radius'].default_value = (1.0, 0.36, 0.18)
    p.inputs['Subsurface Scale'].default_value = SL.SSS_SCALE
    p.inputs['IOR'].default_value = 1.4
    p.inputs['Specular IOR Level'].default_value = SL.SPECULAR
    return m


# ------------------------------------------------------------------------------------------------ the build
def build(D, look='full', coll=None):
    """the creator's head, eyes and ears as objects in coll; returns them as a dict"""
    coll = coll or bpy.context.scene.collection
    h = D['head']
    head, V, me = build_mesh('Creator head', h, coll)
    rest = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    rest.data.foreach_set('vector', np.array(D.get('restBaked') or h['skinRest'], dtype=np.float64))
    zones = np.array(h['skinZones']).reshape(-1, 2)
    # skin_layers' lookups: red the freckle zone, blue the pores (off: the creator's relief), alpha the thin skin
    st = me.color_attributes.new('skin tex', 'FLOAT_COLOR', 'POINT')
    st.data.foreach_set('color', np.stack([zones[:, 0], np.zeros(len(V)), np.zeros(len(V)), zones[:, 1]], 1).ravel())
    c = np.array(h['color']).reshape(-1, 3)
    oc = me.color_attributes.new('occ', 'FLOAT_COLOR', 'POINT')
    oc.data.foreach_set('color', np.concatenate([c, np.ones((len(c), 1))], 1).ravel())
    head.data.materials.append(_zones_mat(D) if look == 'zones' else head_material(D, look))
    eyes = []
    for k, e in enumerate(D['eyes']):
        o, _, _ = build_mesh('Creator eye %d' % k, e, coll)
        o.data.materials.append(eye_material(D, k, look))
        eyes.append(o)
    ears = []
    for k, e in enumerate(D['ears']):
        o, _, eme = build_mesh('Creator ear %d' % k, e, coll)
        if 'color' in e:
            ec = np.array(e['color']).reshape(-1, 3)
            a = eme.color_attributes.new('occ', 'FLOAT_COLOR', 'POINT')
            a.data.foreach_set('color', np.concatenate([ec, np.ones((len(ec), 1))], 1).ravel())
        o.data.materials.append(ear_material(D, e, look))
        ears.append(o)
    return {'head': head, 'eyes': eyes, 'ears': ears}


def _zones_mat(D):
    m = bpy.data.materials.new('creator skin zones')
    N = Nodes(m)
    out = N.new('ShaderNodeOutputMaterial')
    mask = N.sep(N.tex(image(D, 'mask', 'Non-Color')).outputs['Color'])
    lip_r = N.sep(N.tex(image(D, 'lip', 'Non-Color')).outputs['Color'])[0]
    zones_head(D, N, out, mask, lip_r)
    return m
