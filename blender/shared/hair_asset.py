"""The Emberwood hair asset (tools/hair-asset/FORMAT.md) in Blender: read a .hair.glb written by the face creator's
exporter and build the hairstyle exactly as the creator draws it. Every piece keeps the creator's own vertices,
normals, texture coordinates and vertex tones, and wears the creator's material rebuilt node for node: its painted
strand colours times each strand's tone times the hair's shade, its strand bump, matte (roughness 1, the F0 0.04 of
three.js's standard material), and its ends cut out of the three-layer hair-tip map as the creator's shader reads it.
The sheer veils under the clumps are unlit colour over what lies behind, as in the creator.

    import hair_asset
    A = hair_asset.load('wavyFull.hair.glb')
    objs = hair_asset.build(A, parent=head, scale=HEAD_H / 8.6)
    hair_asset.unshadow(objs)          # (the creator's hair takes no shadow; see unshadow)

Nothing here is tuned to look like the creator: the numbers all come from the asset. The one thing Blender cannot
take from it is what three.js works out per pixel from how big the hair is on screen: the tip gain (from fwidth) and
the bump's strength (a height per pixel); set_view gives both the size of a pixel on the hair, as hair_bench.py does.

Head space: the asset is in the creator's frame (Y up, +Z to the front, the chin at 0, the head 8.6 tall); built in
Blender it is (x, y, z) = (X, -Z, Y) * scale, which is the base figure's own head frame (x across, y back, z up).
"""
import json
import math
import os
import struct
import tempfile

import bpy
import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


class Asset:
    """A read hair asset: .json (the glTF), .ext (its emberwood_hair block), .acc(i) (an accessor as an array),
    .image_bytes(i), .prims (one dict of arrays per material), .materials (the creator's material entries)."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            buf = f.read()
        magic, ver, _ = struct.unpack_from('<III', buf, 0)
        if magic != 0x46546C67 or ver != 2:
            raise ValueError(path + ': not a glTF 2 binary')
        off, self.json, self.bin = 12, None, None
        while off < len(buf):
            ln, ty = struct.unpack_from('<II', buf, off)
            body = buf[off + 8:off + 8 + ln]
            if ty == 0x4E4F534A:
                self.json = json.loads(body.decode('utf8'))
            elif ty == 0x004E4942:
                self.bin = body
            off += 8 + ln
        self.path = path
        self.ext = self.json['extras']['emberwood_hair']
        if self.ext.get('schema') != 'emberwood.hair/1':
            raise ValueError(path + ': unknown schema ' + str(self.ext.get('schema')))
        self.materials = [m['extras']['emberwood'] for m in self.json['materials']]
        # one set of arrays per material: its plain-mesh pieces as stored, then its tubes rebuilt from their rings
        # (FORMAT.md, "Strands"); each piece's firstVertex/firstIndex are set to where it lands
        by_mat = {}
        for me in self.json.get('meshes', []):
            for pr in me['primitives']:
                a = pr['attributes']
                by_mat[pr['material']] = [dict(pos=self.acc(a['POSITION']), nrm=self.acc(a['NORMAL']),
                                               uv=self.acc(a['TEXCOORD_0']), col=self.acc(a['COLOR_0']),
                                               along=self.acc(a['_ALONG'])[:, 0], piece=self.acc(a['_PIECE'])[:, 0],
                                               idx=self.acc(pr['indices'])[:, 0].astype(np.int64))]
        rg = self.ext.get('rings')
        if rg:
            R = {k: self.acc(rg[k]).astype(np.float64) for k in ('centre', 'axisA', 'axisB', 'color', 'tex')}
        for pi, pc in enumerate(self.ext['pieces']):
            t = pc.get('tube')
            if not t or 'firstRing' not in t:
                continue
            parts = by_mat.setdefault(pc['material'], [])
            nv0 = sum(len(q['pos']) for q in parts)
            ni0 = sum(len(q['idx']) for q in parts)
            d = decode_tube(t, {k: v[t['firstRing']:t['firstRing'] + t['rings']] for k, v in R.items()})
            d['piece'] = np.full(len(d['pos']), float(pi))
            d['idx'] = d['idx'] + nv0
            parts.append(d)
            pc.update(firstVertex=nv0, vertexCount=len(d['pos']), firstIndex=ni0, indexCount=len(d['idx']))
        self.prims = []
        for mi in sorted(by_mat):
            parts = by_mat[mi]
            self.prims.append(dict(material=mi, **{k: np.concatenate([q[k] for q in parts]) for k in
                                                   ('pos', 'nrm', 'uv', 'col', 'along', 'piece', 'idx')}))

    def acc(self, i):
        a = self.json['accessors'][i]
        v = self.json['bufferViews'][a['bufferView']]
        n = NCOMP[a['type']]
        dt = COMP[a['componentType']]
        o = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        arr = np.frombuffer(self.bin, dtype=dt, count=a['count'] * n, offset=o).reshape(a['count'], n)
        if a.get('normalized'):
            arr = arr.astype(np.float32) / np.iinfo(dt).max
        return arr

    def image_bytes(self, tex):
        img = self.json['images'][self.json['textures'][tex]['source']]
        v = self.json['bufferViews'][img['bufferView']]
        o = v.get('byteOffset', 0)
        return self.bin[o:o + v['byteLength']]

    def pieces(self):
        return self.ext['pieces']

    def centre_line(self, piece):
        """A tube's centre line (its ring centres; creator frame, model units), or None for the other pieces."""
        t = piece.get('tube')
        if not t:
            return None
        return self.acc(self.ext['rings']['centre'])[t['firstRing']:t['firstRing'] + t['rings']]


def decode_tube(t, rg):
    """Rebuild one tube from its rings, vertex for vertex as the creator's tube() makes it (FORMAT.md, "Strands"):
    rings of S + 1 points (the last repeating the first), then the two end points; triangles in the creator's order;
    normals as three.js's computeVertexNormals. rg: that tube's rows of the ring accessors. Glb-convention uv."""
    S1, n = t['ringSize'], t['rings']
    S = S1 - 1
    w = 2.0 * np.pi * np.arange(S1) / S
    c, sn = np.cos(w), np.sin(w)
    pos = rg['centre'][:, None, :] + c[None, :, None] * rg['axisA'][:, None, :] + sn[None, :, None] * rg['axisB'][:, None, :]
    f = 0.9 + 0.1 * sn
    col = np.repeat(rg['color'][:, None, :], S1, axis=1)
    col[:, :, :3] *= f[None, :, None]
    uv = np.empty((n, S1, 2))
    uv[:, :, 0] = t['u0'] + (np.arange(S1) / S)[None, :] * rg['tex'][:, 0:1]
    uv[:, :, 1] = rg['tex'][:, 1:2]
    al = np.repeat(rg['tex'][:, 2:3], S1, axis=1)
    pos = np.concatenate([pos.reshape(-1, 3), rg['centre'][[0, n - 1]]])
    col = np.concatenate([col.reshape(-1, 4), np.asarray(t['endColor'], dtype=np.float64)])
    uv = np.concatenate([uv.reshape(-1, 2), [[t['u0'], rg['tex'][0, 1]], [t['u0'], rg['tex'][n - 1, 1]]]])
    al = np.concatenate([al.reshape(-1), t['endAlong']])
    i = np.arange(n - 1)[:, None] * S1 + np.arange(S)[None, :]
    a, b = i.reshape(-1), i.reshape(-1) + 1
    body = np.stack([a, b, a + S1, b, b + S1, a + S1], axis=1).reshape(-1)
    s = np.arange(S)
    c0, c1 = n * S1, n * S1 + 1
    e0 = np.stack([s, s + 1, np.full(S, c0)], axis=1).reshape(-1)
    a1 = (n - 1) * S1 + s
    e1 = np.stack([a1, np.full(S, c1), a1 + 1], axis=1).reshape(-1)
    idx = np.concatenate([body, e0, e1]).astype(np.int64)
    tri = idx.reshape(-1, 3)
    A, B, C = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(C - B, A - B)
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, tri[:, k], fn)
    l = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(l > 0, l, 1.0)
    return dict(pos=pos, nrm=nrm, uv=uv, col=col, along=al, idx=idx)


def load(path):
    return Asset(path)


def to_blender(p, scale):
    """Creator frame (X, Y up, Z front) to the figure's head frame (x, y back, z up)."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1) * scale


# ---------------------------------------------------------------- images
def _images(A):
    """The asset's three textures as Blender images (the strand colours sRGB, the bump and the tip map data), and the
    tip map's blue channel blurred as the creator's hairline reads it (texture2D with a mip bias of 4: a 16x box)."""
    key = os.path.basename(A.path) + ':' + A.ext['source']['sha256']
    have = bpy.data.images.get('hair tips ' + key)
    if have:
        return {k: bpy.data.images['hair %s %s' % (k, key)] for k in ('hair', 'bump', 'tips', 'tips soft')}
    d = tempfile.mkdtemp(prefix='hair_asset_')
    out = {}
    for k, t in A.ext['textures'].items():
        p = os.path.join(d, k + '.png')
        with open(p, 'wb') as f:
            f.write(A.image_bytes(t))
        img = bpy.data.images.load(p)
        img.name = 'hair %s %s' % (k, key)
        cs = A.json['textures'][t].get('extras', {}).get('colorSpace', '')
        img.colorspace_settings.name = 'sRGB' if cs == 'srgb' else 'Non-Color'
        img.pack()
        out[k] = img
    # the hairline's soft read: mip level 4 of the tip map, as WebGL would sample it (box-filtered 16 x 16)
    t = out['tips']
    W, H = t.size
    px = np.array(t.pixels[:], dtype=np.float32).reshape(H, W, 4)
    k = 16
    small = px[:H - H % k, :W - W % k].reshape(H // k, k, W // k, k, 4).mean(axis=(1, 3))
    soft = bpy.data.images.new('hair tips soft ' + key, W // k, H // k, alpha=True, float_buffer=True)
    soft.colorspace_settings.name = 'Non-Color'
    soft.pixels.foreach_set(small.ravel())
    soft.pack()
    out['tips soft'] = soft
    return out


# ---------------------------------------------------------------- materials
class _N:
    """Node helpers on one tree."""

    def __init__(self, nt):
        self.nt = nt

    def node(self, kind, **kw):
        n = self.nt.nodes.new(kind)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=0.0, c=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        for i, x in enumerate((a, b) if c is None else (a, b, c)):
            if isinstance(x, (int, float)):
                n.inputs[i].default_value = x
            else:
                self.link(x, n.inputs[i])
        return n.outputs[0]

    def smoothstep(self, e0, e1, x):
        """GLSL smoothstep (Hermite, clamped)."""
        t = self.math('MULTIPLY', self.math('SUBTRACT', x, e0), 1.0 / (e1 - e0), clamp=False)
        t = self.math('MINIMUM', self.math('MAXIMUM', t, 0.0), 1.0)
        return self.math('MULTIPLY', self.math('MULTIPLY', t, t), self.math('SUBTRACT', 3.0, self.math('MULTIPLY', t, 2.0)))

    def mix(self, a, b, t):
        """a + (b - a) t, on floats."""
        return self.math('MULTIPLY_ADD', self.math('SUBTRACT', b, a), t, a)

    def vec(self, u, v):
        n = self.node('ShaderNodeCombineXYZ')
        for i, x in enumerate((u, v)):
            if isinstance(x, (int, float)):
                n.inputs[i].default_value = x
            else:
                self.link(x, n.inputs[i])
        return n.outputs[0]

    def tex(self, img, vec, ext='REPEAT'):
        t = self.node('ShaderNodeTexImage', image=img, interpolation='Linear', extension=ext)
        self.link(vec, t.inputs['Vector'])
        s = self.node('ShaderNodeSeparateColor')
        self.link(t.outputs['Color'], s.inputs['Color'])
        return t, s

    def rgb_scale(self, col, k):
        n = self.node('ShaderNodeVectorMath', operation='SCALE')
        self.link(col, n.inputs[0])
        if isinstance(k, (int, float)):
            n.inputs['Scale'].default_value = k
        else:
            self.link(k, n.inputs['Scale'])
        return n.outputs['Vector']


BUMP = 1.0      # (a multiplier on the asset's bumpScale, for checking the bump alone; 1 is the creator's)


def _cut_material(A, e, imgs, scale, look='full', base=None, tone=1.0):
    """The creator's cut-out hair material (FORMAT.md 'cut', model emberwood-cut-1), in Cycles nodes. look 'clay'
    keeps the cut-out and every tone the cut-out shader applies, and turns the colour map and the vertex tones grey
    (as the exporter's clay reference renders do). With `base` (a figure's hair material), built inside a copy of
    it: the creator's surface replaces its 'hair bsdf' and the cut-out is its 'tip cut', so its sprite surface and
    its passes stay (see build)."""
    p = e['params']
    if base is not None:
        m = base.copy()
        m.name = base.name + ' creator'
        nt = m.node_tree
        old = nt.nodes.get('hair bsdf')
        if old is not None:
            nt.nodes.remove(old)
        N = _N(nt)
        out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    else:
        m = bpy.data.materials.new('hair cut (%s)' % look)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        N = _N(nt)
        out = N.node('ShaderNodeOutputMaterial')
    uvn = N.node('ShaderNodeUVMap', uv_map='UVMap')
    uv = N.node('ShaderNodeSeparateXYZ')
    N.link(uvn.outputs['UV'], uv.inputs['Vector'])
    u, v = uv.outputs['X'], uv.outputs['Y']
    col = N.node('ShaderNodeVertexColor', layer_name='Col')
    a = col.outputs['Alpha']
    # how far off the strands are, in tip-map texels per pixel: three.js measures it (fwidth); here it is set
    # per camera (set_tip_scale), into this value node
    tpx = N.node('ShaderNodeValue', name='tip px', label='tip texels per pixel')
    tpx.outputs[0].default_value = 3.0
    far = N.smoothstep(p['farPx0'], p['farPx1'], tpx.outputs[0])
    gain = N.math('DIVIDE', p['gainNum'], N.mix(p['gainNear'], p['gainFar'], far))
    t_in = N.math('MULTIPLY', N.math('SUBTRACT', p['inOne'], N.smoothstep(p['in0'], p['in1'], a)),
                  N.math('SUBTRACT', p['farOne'], N.math('MULTIPLY', p['inFar'], far)))
    tips = imgs['tips']
    # (three.js clamps the map up and down (ClampToEdge) and repeats it round; one Blender image node does one or the
    # other, so v is held inside the map's first and last rows by hand)
    lo, hi = 0.5 / tips.size[1], 1.0 - 0.5 / tips.size[1]

    def cv(x):
        return N.math('MINIMUM', N.math('MAXIMUM', x, lo), hi)
    _, s1 = N.tex(tips, N.vec(u, cv(a)))
    _, s2 = N.tex(tips, N.vec(N.math('ADD', u, p['du2']), cv(N.math('MAXIMUM', N.math('SUBTRACT', a, p['dv2']), p['dv2min']))))
    _, s3 = N.tex(tips, N.vec(N.math('ADD', u, p['du3']), cv(N.math('MAXIMUM', N.math('SUBTRACT', a, p['dv3']), p['dv3min']))))
    tA = N.math('MULTIPLY', s1.outputs['Green'], gain)
    tA2 = N.math('MULTIPLY', s2.outputs['Blue'], gain)
    tA3 = N.math('MULTIPLY', s3.outputs['Blue'], gain)
    if p.get('hairline', 0) > p['hairlineOn']:
        # (a hairline: the soft read of the wider hairs, and the other two layers off)
        _, sh = N.tex(imgs['tips soft'], N.vec(u, cv(a)))
        tA = N.math('MULTIPLY', sh.outputs['Blue'], p['hairlineGain'])
        tA2 = N.math('ADD', 0.0, p['hairlineA2'])
        tA3 = N.math('ADD', 0.0, p['hairlineA3'])
    l1 = N.math('GREATER_THAN', tA, p['cut1'] - 1e-6)          # (tipA >= cut1)
    l2 = N.math('MULTIPLY', N.math('SUBTRACT', 1.0, l1), N.math('GREATER_THAN', tA2, p['cut2'] - 1e-6))
    l3 = N.math('MULTIPLY', N.math('SUBTRACT', 1.0, l1), N.math('SUBTRACT', 1.0, l2))
    alpha = N.math('ADD', N.math('MULTIPLY', l1, tA), N.math('ADD', N.math('MULTIPLY', l2, tA2), N.math('MULTIPLY', l3, tA3)))
    keep = N.math('GREATER_THAN', alpha, e['alphaTest'] - 1e-6)    # (three.js discards alpha < alphaTest)
    # the tone each branch gives: the top layer its own hair's tone near its end, the layers under it darker
    tone1 = N.mix(1.0, N.math('MULTIPLY', s1.outputs['Red'], p['toneGain']), t_in)
    tone_ = N.math('ADD', N.math('MULTIPLY', l1, tone1), N.math('ADD', N.math('MULTIPLY', l2, p['tone2']), N.math('MULTIPLY', l3, p['tone3'])))
    k = N.math('MULTIPLY', tone_, p['shade'] * tone)
    # the colour: the material's colour times its map times the vertex tone
    if look == 'clay':
        rgb = N.node('ShaderNodeRGB')
        rgb.outputs[0].default_value = (0.6, 0.6, 0.6, 1.0)      # (linear, as three.js's setRGB)
        base_col = rgb.outputs[0]
        # (the exporter's clay: color 0.6 sRGB, no map, the vertex tones kept: they are the strands' own shade)
        vc = N.node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        vc.inputs['Factor'].default_value = 1.0
        N.link(base_col, vc.inputs['A'])
        N.link(col.outputs['Color'], vc.inputs['B'])
        base_col = vc.outputs['Result']
    else:
        ht, _ = N.tex(imgs['hair'], N.vec(u, N.math('MINIMUM', N.math('MAXIMUM', v, 0.5 / imgs['hair'].size[1]), 1 - 0.5 / imgs['hair'].size[1])))
        vc = N.node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        vc.inputs['Factor'].default_value = 1.0
        N.link(ht.outputs['Color'], vc.inputs['A'])
        N.link(col.outputs['Color'], vc.inputs['B'])
        tint = N.node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        tint.inputs['Factor'].default_value = 1.0
        N.link(vc.outputs['Result'], tint.inputs['A'])
        tint.inputs['B'].default_value = (*e['color'], 1.0)
        base_col = tint.outputs['Result']
    bsdf = N.node('ShaderNodeBsdfPrincipled', name='hair bsdf')
    N.link(N.rgb_scale(base_col, k), bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = e.get('roughness', 1.0)
    bsdf.inputs['Metallic'].default_value = e.get('metalness', 0.0)
    bsdf.inputs['IOR'].default_value = 1.5                    # (F0 0.04, three.js's standard material)
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    if look != 'clay' and e['maps'].get('bump') == 'bump':
        bt, _ = N.tex(imgs['bump'], N.vec(u, N.math('MINIMUM', N.math('MAXIMUM', v, 0.5 / imgs['bump'].size[1]), 1 - 0.5 / imgs['bump'].size[1])))
        bump = N.node('ShaderNodeBump')
        # three.js's bump map: a height of bumpScale x the map's value, in the model's units
        bump.inputs['Strength'].default_value = 1.0
        # three.js's bump (r151 on) is a height of bumpScale x the map's value per screen pixel, not per unit of
        # length: it measures the map's change from one pixel to the next against the surface's unit directions there.
        # So it matches a Blender bump whose distance is bumpScale x the size of a pixel on the surface; set_view sets
        # it for the camera (here: a pixel 0.034 of a model unit, the bench's 512-pixel views at 50 units)
        bump.inputs['Distance'].default_value = e.get('bumpScale', 1.0) * 0.034 * scale * BUMP
        bump.name = 'creator bump'
        bump['bumpScale'] = e.get('bumpScale', 1.0)
        N.link(bt.outputs['Color'], bump.inputs['Height'])
        N.link(bump.outputs['Normal'], bsdf.inputs['Normal'])
    # shadows are cast through the creator's depth material: all three layers at once, read generously (x 1.7)
    d1 = N.math('MAXIMUM', s1.outputs['Green'], N.math('MAXIMUM', s2.outputs['Blue'], s3.outputs['Blue']))
    dkeep = N.math('GREATER_THAN', N.math('MULTIPLY', d1, 1.7), e['alphaTest'] - 1e-6)
    lp = N.node('ShaderNodeLightPath')
    keep_all = N.mix(keep, dkeep, lp.outputs['Is Shadow Ray'])
    clear = N.node('ShaderNodeBsdfTransparent')
    mixs = N.node('ShaderNodeMixShader', name='tip cut')      # (inputs[2] the surface: female.sprite_mode swaps it)
    N.link(keep_all, mixs.inputs['Fac'])
    N.link(clear.outputs['BSDF'], mixs.inputs[1])
    N.link(bsdf.outputs['BSDF'], mixs.inputs[2])
    N.link(mixs.outputs['Shader'], out.inputs['Surface'])
    m.use_backface_culling = e.get('side') == 'front'
    if hasattr(m, 'blend_method'):
        m.blend_method = 'CLIP'
    if hasattr(m, 'surface_render_method'):
        m.surface_render_method = 'DITHERED'
    m['emberwood'] = 'cut'
    return m


def _veil_material(e, tone=1.0):
    """The sheer veil: unlit vertex colour laid over what is behind it by its alpha (three.js's SrcAlpha /
    OneMinusSrcAlpha), as the creator's MeshBasicMaterial (its colour scaled by `tone`, as the hair's)."""
    m = bpy.data.materials.new('hair veil')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N = _N(nt)
    out = N.node('ShaderNodeOutputMaterial')
    col = N.node('ShaderNodeVertexColor', layer_name='Col')
    em = N.node('ShaderNodeEmission')
    tint = N.node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    tint.inputs['Factor'].default_value = 1.0
    N.link(col.outputs['Color'], tint.inputs['A'])
    tint.inputs['B'].default_value = (*e['color'], 1.0)
    N.link(tint.outputs['Result'], em.inputs['Color'])
    em.inputs['Strength'].default_value = tone
    clear = N.node('ShaderNodeBsdfTransparent')
    mixs = N.node('ShaderNodeMixShader')
    N.link(N.math('MULTIPLY', col.outputs['Alpha'], e.get('opacity', 1.0)), mixs.inputs['Fac'])
    N.link(clear.outputs['BSDF'], mixs.inputs[1])
    N.link(em.outputs['Emission'], mixs.inputs[2])
    N.link(mixs.outputs['Shader'], out.inputs['Surface'])
    if hasattr(m, 'blend_method'):
        m.blend_method = 'BLEND'
    if hasattr(m, 'surface_render_method'):
        m.surface_render_method = 'BLENDED'
    m['emberwood'] = 'veil'
    return m


def _standard_material(e):
    m = bpy.data.materials.new(e.get('name', 'hair piece'))
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*e['color'], 1.0)
    b.inputs['Roughness'].default_value = e.get('roughness', 1.0)
    b.inputs['Metallic'].default_value = e.get('metalness', 0.0)
    b.inputs['Specular IOR Level'].default_value = 0.5
    if e.get('vertexColors'):
        vc = m.node_tree.nodes.new('ShaderNodeVertexColor')
        vc.layer_name = 'Col'
        mx = m.node_tree.nodes.new('ShaderNodeMix')
        mx.data_type = 'RGBA'
        mx.blend_type = 'MULTIPLY'
        mx.inputs['Factor'].default_value = 1.0
        mx.inputs['B'].default_value = (*e['color'], 1.0)
        m.node_tree.links.new(vc.outputs['Color'], mx.inputs['A'])
        m.node_tree.links.new(mx.outputs['Result'], b.inputs['Base Color'])
    return m


# ---------------------------------------------------------------- objects
def _mesh(name, pos, nrm, uv, col, idx, attrs):
    me = bpy.data.meshes.new(name)
    nv, nt = len(pos), len(idx) // 3
    me.vertices.add(nv)
    me.vertices.foreach_set('co', pos.astype(np.float32).ravel())
    me.loops.add(nt * 3)
    me.loops.foreach_set('vertex_index', idx.astype(np.int32))
    me.polygons.add(nt)
    me.polygons.foreach_set('loop_start', np.arange(0, nt * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set('use_smooth', np.ones(nt, dtype=bool))
    uvl = me.uv_layers.new(name='UVMap')
    uvl.data.foreach_set('uv', uv[idx].astype(np.float32).ravel())
    ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', col.astype(np.float32).ravel())
    for k, val in attrs.items():
        at = me.attributes.new(k, 'FLOAT', 'POINT')
        at.data.foreach_set('value', val.astype(np.float32))
    # the creator's own normals (its seam-free cap, its tubes' smooth rings), not Blender's recomputed ones
    me.normals_split_custom_set_from_vertices(nrm.astype(np.float32))
    me.validate(clean_customdata=False)
    return me


def _subset(pr, pcs):
    """The vertices and triangles of some pieces of one material's arrays, renumbered from 0."""
    vs = np.concatenate([np.arange(pc['firstVertex'], pc['firstVertex'] + pc['vertexCount']) for pc in pcs])
    ix = np.concatenate([pr['idx'][pc['firstIndex']:pc['firstIndex'] + pc['indexCount']] for pc in pcs])
    remap = np.full(len(pr['pos']), -1, dtype=np.int64)
    remap[vs] = np.arange(len(vs))
    out = {k: pr[k][vs] for k in ('pos', 'nrm', 'uv', 'col', 'along', 'piece')}
    out['idx'] = remap[ix]
    return out


def _group(pc, kind, split):
    """Which object a piece goes into. Not split: one per material kind. Split (a figure's): 'cap' (the opaque head
    of hair), 'locks' (the strands of the top layer, and those that have none), 'locks under' (the layer under it,
    in its gaps), 'veils'."""
    if not split:
        return kind
    if kind == 'veil':
        return 'veils'
    if pc['role'] == 'cap':
        return 'cap'
    return 'locks under' if pc.get('layer') == 1 else 'locks'


SPLIT_NAMES = {'cap': '%s', 'locks': '%s locks', 'locks under': '%s locks under', 'veils': '%s veils'}


def build(A, parent=None, scale=1.0, look='full', name='Hair', collection=None, split=False, base=None, tone=1.0):
    """Build the asset in the figure's head frame (scaled by `scale`: HEAD_H / 8.6 on a figure, 1 on the bench),
    parented to `parent` when given. Each object carries the asset's attributes: 'along' (the distance along a
    strand from its root, -1 off the strands) and 'piece' (the asset's piece index). Returns the objects.

    Not split (the bench): one object per creator material, '<name> cut', '<name> veil'. Split (a figure): '<name>'
    the cap, '<name> locks', '<name> locks under' and '<name> veils' (see _group), and the cut material is built
    inside a copy of `base` (the figure's own hair material, so its sprite surface and its passes stay), with the
    creator's surface as its 'hair bsdf' and the cut-out as its 'tip cut' (female.sprite_mode switches them). `tone`
    scales the hair's colour (a figure's views light a dark colour more than the creator's viewer does)."""
    imgs = _images(A)
    col = collection or bpy.context.scene.collection
    objs = []
    mats = {}
    tpu = tip_texels_per_unit(A)
    for pr in A.prims:
        e = A.materials[pr['material']]
        kind = e['kind']
        groups = {}
        for i in np.unique(pr['piece']).astype(int):
            pc = A.pieces()[i]
            groups.setdefault(_group(pc, kind, split), []).append(pc)
        for g, pcs in groups.items():
            sub = _subset(pr, pcs) if (split or len(groups) > 1) else pr
            oname = (SPLIT_NAMES[g] % name) if split else '%s %s' % (name, kind)
            pos = to_blender(sub['pos'], scale)
            nrm = to_blender(sub['nrm'], 1.0)
            uv = sub['uv'].copy()
            uv[:, 1] = 1.0 - uv[:, 1]          # (back from glTF's down-the-image v to Blender's, which is three.js's)
            me = _mesh(oname, pos, nrm, uv, sub['col'], sub['idx'], {'along': sub['along'], 'piece': sub['piece']})
            if pr['material'] not in mats:
                if kind == 'cut':
                    mats[pr['material']] = _cut_material(A, e, imgs, scale, look, base=base if split else None, tone=tone)
                elif kind == 'veil':
                    mats[pr['material']] = _veil_material(e, tone=tone)
                else:
                    mats[pr['material']] = _standard_material(e)
            me.materials.append(mats[pr['material']])
            o = bpy.data.objects.new(oname, me)
            col.objects.link(o)
            if parent is not None:
                o.parent = parent
            o['hair_kind'] = kind
            # (a piece of the creator's that casts no shadow: the strands and the veils; the cap casts. An object
            # casts when any piece of it does, through the material's shadow cut)
            o.visible_shadow = any(pc.get('castShadow') for pc in pcs)
            if kind == 'veil':
                # (unlit and seen only by the camera, as a MeshBasicMaterial is drawn: no light bounces off it)
                o.visible_diffuse = o.visible_glossy = o.visible_transmission = o.visible_volume_scatter = False
                o.visible_shadow = False
            o['receive_shadow'] = any(pc.get('receiveShadow') for pc in pcs)
            o['hair_scale'] = scale
            o['tip_texels_per_unit'] = tpu
            objs.append(o)
    return objs


def points(objs, A):
    """The points a figure moves the hair by (its clearance off the clothes): every vertex of the cap and the veils
    is its own point; each ring of a tube is one point, its centre, which the ring's vertices (and the tube's end
    point) follow, so a strand moves without changing its section. Sets the int attribute 'pt' (the point a vertex
    follows) and returns (points, runs): the points in each object's own frame, and the tubes' runs of them,
    (first, count), root to tip."""
    pts, runs = [], []
    for o in objs:
        me = o.data
        pc_of = np.empty(len(me.vertices))
        me.attributes['piece'].data.foreach_get('value', pc_of)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3)
        pt = np.empty(len(me.vertices), dtype=np.int64)
        v = 0
        n = len(me.vertices)
        while v < n:
            pc = A.pieces()[int(pc_of[v])]
            t = pc.get('tube')
            cnt = pc['vertexCount']
            if t:
                S1, R = t['ringSize'], t['rings']
                ring = co[v:v + R * S1].reshape(R, S1, 3)
                centre = ring[:, :-1].mean(axis=1)
                base = len(pts)
                pts.extend(centre)
                runs.append((base, R))
                pt[v:v + R * S1] = base + np.repeat(np.arange(R), S1)
                pt[v + R * S1] = base
                pt[v + R * S1 + 1] = base + R - 1
            else:
                pt[v:v + cnt] = len(pts) + np.arange(cnt)
                pts.extend(co[v:v + cnt])
            v += cnt
        at = me.attributes.get('pt') or me.attributes.new('pt', 'INT', 'POINT')
        at.data.foreach_set('value', pt.astype(np.int32))
    from mathutils import Vector
    return [Vector(p) for p in pts], runs


def set_view(objs, A, pixel, scale=1.0):
    """Set what three.js works out per pixel for the camera, from the size of one pixel on the hair (`pixel`, in the
    asset's model units: 2 d tan(fov / 2) / height at the hair's distance d): how generously the ends are read
    (tipPx, the tip-map texels a pixel covers) and the bump's strength (a height of bumpScale per pixel)."""
    tpx = tip_texels_per_unit(A) * pixel
    for o in objs:
        for m in o.data.materials:
            if not m or not m.node_tree:
                continue
            nd = m.node_tree.nodes
            if 'tip px' in nd:
                nd['tip px'].outputs[0].default_value = tpx
            if 'creator bump' in nd:
                b = nd['creator bump']
                b.inputs['Distance'].default_value = b['bumpScale'] * pixel * scale * BUMP


def tip_px_for(A, far):
    """The tip-map texels per pixel at which the creator's cut-out reads the ends as it does at `far` (0 up close, 1
    far off: its tipGain then 1 and 2.5): for a figure, whose views and sprite sheet each have one camera distance."""
    p = next(e for e in A.materials if e['kind'] == 'cut')['params']
    lo, hi = 0.0, 1.0
    for _ in range(40):          # (smoothstep's inverse)
        t = 0.5 * (lo + hi)
        if t * t * (3 - 2 * t) < far:
            lo = t
        else:
            hi = t
    return p['farPx0'] + 0.5 * (lo + hi) * (p['farPx1'] - p['farPx0'])


def set_sprite_px(objs, sprite):
    """The cut-out's tip texels per pixel for a figure's sprite sheet (female.sprite_mode puts it in, and the view's
    back, which set_pixel keeps)."""
    for o in objs:
        for m in o.data.materials:
            if m and m.node_tree and 'tip px' in m.node_tree.nodes:
                nd = m.node_tree.nodes['tip px']
                nd['sprite_px'] = sprite
                if 'view_px' not in nd:
                    nd['view_px'] = nd.outputs[0].default_value


def set_pixel(objs, pixel):
    """set_view for built objects alone, from the size of a pixel in Blender's units (an orthographic view: its
    height / its height in pixels), for a figure's views: build stored what the asset gives for it."""
    for o in objs:
        if 'tip_texels_per_unit' not in o:
            continue
        sc = o['hair_scale']
        tpx = o['tip_texels_per_unit'] * pixel / sc
        for m in o.data.materials:
            if not m or not m.node_tree:
                continue
            nd = m.node_tree.nodes
            if 'tip px' in nd:
                nd['tip px'].outputs[0].default_value = tpx
                nd['tip px']['view_px'] = tpx
            if 'creator bump' in nd:
                b = nd['creator bump']
                b.inputs['Distance'].default_value = b['bumpScale'] * pixel * BUMP


def clay_material(m, rgb, roughness):
    """A clay copy of a cut material: the same cut-out (and its shadow cut), a plain matte colour, no bump."""
    c = bpy.data.materials.get(m.name + ' clay')
    if c is not None:
        return c
    c = m.copy()
    c.name = m.name + ' clay'
    nt = c.node_tree
    b = nt.nodes.get('hair bsdf')
    if b is None:
        return c
    for k in ('Base Color', 'Normal'):
        for ln in list(b.inputs[k].links):
            nt.links.remove(ln)
    b.inputs['Base Color'].default_value = (*rgb, 1.0)
    b.inputs['Roughness'].default_value = roughness
    return c


def hair_objects():
    """The built hair objects in the scene (any asset's)."""
    return [o for o in bpy.data.objects if o.type == 'MESH' and 'hair_kind' in o]


def pixel_size(cam, dist, res):
    """The size of one pixel at `dist` from a perspective camera rendering `res` pixels high (model units)."""
    fov = cam.data.angle_y if cam.data.sensor_fit == 'VERTICAL' else cam.data.angle
    return 2.0 * dist * math.tan(fov / 2.0) / res


def tip_texels_per_unit(A):
    """How many tip-map texels one model unit of hair covers (three.js's tipPx is this times a pixel's size): the
    larger of the map's two rates, round the strands (u) and up them (the cut-out). Measured on the cap, round the
    head at mid height and down its steepest column."""
    pr = next(p for p in A.prims if A.materials[p['material']]['kind'] == 'cut')
    cap = next(pc for pc in A.pieces() if pc['role'] == 'cap')
    rows = cap['grid']['rows']
    sl = slice(cap['firstVertex'], cap['firstVertex'] + cap['vertexCount'])
    P, U, C = pr['pos'][sl], pr['uv'][sl], pr['col'][sl]
    j = rows // 2
    du = abs(U[rows + j, 0] - U[j, 0]) / (np.linalg.norm(P[rows + j] - P[j]) + 1e-9)
    seg = np.linalg.norm(np.diff(P[:rows], axis=0), axis=1) + 1e-9
    da = np.max(np.abs(np.diff(C[:rows, 3])) / seg)
    p = A.materials[pr['material']]['params']
    return max(du * p['tipW'], da * p['tipH'])


def unshadow(objs, scene=None):
    """The creator's hair takes no shadow (receiveShadow false on every piece: in its shadow map the hair went
    near-black in patches and the ends lit up in bands). Cycles has no switch for that, so each light in the scene is
    split in two with light linking: one lights everything but the hair, as before; its twin lights only the hair
    and casts no shadow from anything (an empty shadow linking set). Needs Blender 4.0 or later. The world's light
    still reaches the hair through its own occlusion; the bench, which matches the creator, keeps its world on the
    hair unoccluded (hair_bench.creator_world)."""
    scene = scene or bpy.context.scene
    objs = [o for o in objs if not o.get('receive_shadow')]
    if not objs:
        return
    excl = bpy.data.collections.new('hair (not lit by the scene lights)')
    only = bpy.data.collections.new('hair (lit unshadowed)')
    for o in objs:
        excl.objects.link(o)
        only.objects.link(o)
    for co in excl.collection_objects:
        co.light_linking.link_state = 'EXCLUDE'
    # (a shadow linking set that holds only a far-off speck: no object of the scene blocks the twin's light)
    speck = bpy.data.objects.get('hair: no shadow') or bpy.data.objects.new('hair: no shadow', None)
    nobody = bpy.data.collections.get('hair: casts on hair') or bpy.data.collections.new('hair: casts on hair')
    if speck.name not in nobody.objects:
        nobody.objects.link(speck)
    for L in [o for o in scene.objects if o.type == 'LIGHT' and not o.get('hair_twin')]:
        twin = L.copy()
        twin.data = L.data.copy()
        twin['hair_twin'] = True
        twin.name = L.name + ' (on the hair)'
        for c in L.users_collection:
            c.objects.link(twin)
        L.light_linking.receiver_collection = excl
        twin.light_linking.receiver_collection = only
        twin.light_linking.blocker_collection = nobody
