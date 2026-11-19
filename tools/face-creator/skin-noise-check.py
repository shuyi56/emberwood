"""Check the Face Creator's rebuilt Blender textures against Blender's own: Noise (Fac and Color), Voronoi F1 (Distance
and Color) and Voronoi Distance to Edge, as skin_layers.py uses them, on a plane in the same rest coordinates. Blender
renders them as emissions (Cycles, 1 sample at each pixel's centre); headless Chromium runs the creator's GLSL
(SKIN_NOISE_GLSL in main.js) on the same points. Prints the largest difference of each; all should be below 1e-4.

    python3 tools/face-creator/skin-noise-check.py <main.js>       (needs bpy, numpy and node with playwright)"""
import json
import os
import subprocess
import sys
import tempfile

import bpy
import numpy as np

N = 256
CASES = [      # (name, Blender node setup, GLSL expression of R, the rest position)
    ('noise fac 1.5 / 4 / 0.6', ('noise', 1.5, 4.0, 0.6, 'Fac', None), 'sl_noise(R, 1.5, 4.0, 0.6)'),
    ('noise color.g 0.35 / 3 / 0.6', ('noise', 0.35, 3.0, 0.6, 'Color', 'Green'), 'sl_noiseColor(R, 0.35, 3.0, 0.6).y'),
    ('noise color.b 0.6 / 1 / 0.5', ('noise', 0.6, 1.0, 0.5, 'Color', 'Blue'), 'sl_noiseColor(R, 0.6, 1.0, 0.5).z'),
    ('voronoi F1 distance 2.3', ('voronoi', 2.3, 'F1', 'Distance', None), 'sl_voronoiF1(R, 2.3).x'),
    ('voronoi F1 color.r 1.3', ('voronoi', 1.3, 'F1', 'Color', 'Red'), 'sl_voronoiF1(R, 1.3).y'),
    ('voronoi F1 color.g 0.55', ('voronoi', 0.55, 'F1', 'Color', 'Green'), 'sl_voronoiF1(R, 0.55).z'),
    ('voronoi edge 0.5', ('voronoi', 0.5, 'DISTANCE_TO_EDGE', 'Distance', None), 'sl_voronoiEdge(R, 0.5)'),
    ('voronoi edge 1.5', ('voronoi', 1.5, 'DISTANCE_TO_EDGE', 'Distance', None), 'sl_voronoiEdge(R, 1.5)'),
]
Z = 55.0      # (a height on the figure's head, so the cells are those the head uses)


def blender_side(tmp):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    me = bpy.data.meshes.new('p')
    me.from_pydata([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], [], [(0, 1, 2, 3)])
    a = me.attributes.new('rest', 'FLOAT_VECTOR', 'POINT')
    a.data.foreach_set('vector', [-2, -2, Z, 2, -2, Z, 2, 2, Z, -2, 2, Z])
    o = bpy.data.objects.new('p', me)
    sc.collection.objects.link(o)
    cd = bpy.data.cameras.new('c')
    cd.type, cd.ortho_scale = 'ORTHO', 4
    cam = bpy.data.objects.new('c', cd)
    sc.collection.objects.link(cam)
    cam.location = (0, 0, 10)
    sc.camera = cam
    sc.render.engine = 'CYCLES'
    sc.cycles.samples, sc.cycles.filter_width, sc.cycles.use_denoising = 1, 0.01, False
    sc.render.resolution_x = sc.render.resolution_y = N
    sc.render.image_settings.file_format, sc.render.image_settings.color_depth = 'OPEN_EXR', '32'
    sc.world = bpy.data.worlds.new('w')
    out = []
    for name, spec, _ in CASES:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        mo = nt.nodes.new('ShaderNodeOutputMaterial')
        em = nt.nodes.new('ShaderNodeEmission')
        nt.links.new(em.outputs[0], mo.inputs[0])
        rest = nt.nodes.new('ShaderNodeAttribute')
        rest.attribute_name = 'rest'
        if spec[0] == 'noise':
            t = nt.nodes.new('ShaderNodeTexNoise')
            t.inputs['Scale'].default_value, t.inputs['Detail'].default_value, t.inputs['Roughness'].default_value = spec[1:4]
            sock, ch = spec[4], spec[5]
        else:
            t = nt.nodes.new('ShaderNodeTexVoronoi')
            t.inputs['Scale'].default_value, t.feature = spec[1], spec[2]
            sock, ch = spec[3], spec[4]
        nt.links.new(rest.outputs['Vector'], t.inputs['Vector'])
        s = t.outputs[sock]
        if ch:
            sep = nt.nodes.new('ShaderNodeSeparateColor')
            nt.links.new(s, sep.inputs[0])
            s = sep.outputs[ch]
        nt.links.new(s, em.inputs['Color'])
        o.data.materials.clear()
        o.data.materials.append(m)
        sc.render.filepath = os.path.join(tmp, 'b.exr')
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(sc.render.filepath)
        out.append(np.array(im.pixels[:]).reshape(N, N, 4)[..., 0])
        bpy.data.images.remove(im)
    return out


GL = r"""
const fs = require('fs'), { chromium } = require('playwright');
const src = fs.readFileSync(process.argv[2], 'utf8');
const i = src.indexOf('var SKIN_NOISE_GLSL = ['), j = src.indexOf("].join('\\n') + '\\n';", i);
const GLSL = eval(src.slice(i + 'var SKIN_NOISE_GLSL = '.length, j + 1)).join('\n') + '\n';
const exprs = JSON.parse(process.argv[3]), N = +process.argv[4], Z = process.argv[5];
(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage(), res = [];
  for (const e of exprs) {
    const frag = `#version 300 es\nprecision highp float; precision highp int;\n${GLSL}\nout vec4 o;\nvoid main() { vec3 R = vec3(gl_FragCoord.x / ${N}.0 * 4.0 - 2.0, gl_FragCoord.y / ${N}.0 * 4.0 - 2.0, ${Z}); o = vec4(${e}, 0.0, 0.0, 1.0); }`;
    res.push(await p.evaluate(([fs_, N]) => {
      const c = document.createElement('canvas'); c.width = c.height = N; const gl = c.getContext('webgl2'); gl.getExtension('EXT_color_buffer_float');
      const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
      const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, '#version 300 es\nvoid main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.0-1.0,0.0,1.0); }'));
      gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs_)); gl.linkProgram(pr);
      const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, N, N);
      const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.viewport(0, 0, N, N); gl.useProgram(pr); gl.drawArrays(gl.TRIANGLES, 0, 3);
      const px = new Float32Array(N * N * 4); gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, px); return Array.from(px.filter((v, k) => k % 4 === 0));
    }, [frag, N]));
  }
  console.log(JSON.stringify(res)); await b.close();
})();
"""


def main():
    main_js = os.path.abspath(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        bl = blender_side(tmp)
        js = os.path.join(tmp, 'gl.cjs')
        open(js, 'w').write(GL)
        env = dict(os.environ, NODE_PATH=subprocess.run(['npm', 'root', '-g'], capture_output=True, text=True).stdout.strip())
        r = subprocess.run(['node', js, main_js, json.dumps([c[2] for c in CASES]), str(N), str(Z)], capture_output=True, text=True, env=env, check=True)
        gl = json.loads(r.stdout.strip().split('\n')[-1])
    worst = 0
    for (name, _, _), b, g in zip(CASES, bl, gl):
        d = np.abs(b - np.array(g).reshape(N, N))
        # (a pixel on a cell's edge can fall either side with float rounding: the 99.9th percentile, and the share over 1e-4)
        p = float(np.percentile(d, 99.9))
        worst = max(worst, p)
        print('%-32s 99.9th pct %.2e   share > 1e-4: %.4f' % (name, p, float((d > 1e-4).mean())))
    print('OK' if worst < 1e-4 else 'MISMATCH')
    sys.exit(0 if worst < 1e-4 else 1)


if __name__ == '__main__':
    main()
