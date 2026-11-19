/* ======================================================================================================
   The layered skin (blender/base/skin_layers.py, PRs #41 and #42), drawn here as Blender draws it.
   Three layers, as a painter builds skin: 1 a deep, mottled red and brown flesh; 2 the veins over it, seen only where
   the skin is thin; 3 the epidermis, this creator's painted skin (tone, flush, shade, freckles), laid over both at
   EPIDERMIS_OPACITY, more opaque where it is pigmented, with sun spots, moles and blemishes in it. Then a slow drift in
   the tone, a patchy roughness, a thin oil film, a soft down, and light that spreads under the skin.
   - Every number is skin_layers.py's own (SKIN_LAYERS below is generated from it by tools/face-creator/skin-params.mjs).
   - The patterns are Blender's own procedural textures, rebuilt bit for bit (its Perlin noise and fBm, its Voronoi F1 and
     distance to edge, on Bob Jenkins's hash as Blender uses it), read at the same point of the figure's rest space
     (model units, the chin at HEAD_Z), so a vein or a mole sits where Blender puts it. They are baked once, at a
     first visit, into three maps over the head's wrap-around coordinates (uv1), kept between visits, so the face
     costs three texture reads and a shape slider costs nothing more.
   - Where they show is skin_tones' (female.py) thin-skin and freckle maps on the head, the same gaussians in head
     heights, with paint_face's hold of the face's thin skin to 0.3.
   - The pores and freckles stay this creator's own (its pore relief and its ranked freckle patches, which the sliders
     drive). Blender's pits are a third of a millimetre and vanish under its scatter at a face's distance; given
     PORE_SHADE, the creator's larger pits read as dark pinpricks, so they keep their own warm shade (poreTint) and take
     from Blender only the roughness in the pit and the hold out of the scatter.
   - What three.js cannot do: subsurface scattering. Its closest cheap equivalent is normalised wrapped diffuse light per
     channel (red wrapping furthest, in Blender's radius ratio 1 : 0.36 : 0.18: light moved from the lit side past the
     terminator, none added), held back where pigment and pores sit, as Blender holds them out of the scatter. Blender's bounce light (views.py look_3d) is added to the skin alone, so the hair's
     look and its exported lights do not change.
   ====================================================================================================== */
/* @@SKIN_LAYERS@@ */ /* @@SKIN_LAYERS_END@@ */
var SKIN_WRAP_RADIUS = 0.8;      // (model units: the curvature the scatter is measured against, the nose's and the cheek's)
var SKIN_BAKE_W = 1024, SKIN_BAKE_H = 512;
function skinLin(c) { return c.map(function (v) { v /= 255; return v > 0.04045 ? Math.pow((v + 0.055) / 1.055, 2.4) : v / 12.92; }); }
function skinGlslConsts() {
  var S = SKIN_LAYERS, out = [];
  function f(v) { var s = String(+v); return s.indexOf('.') < 0 && s.indexOf('e') < 0 ? s + '.0' : s; }
  function v3(c) { return 'vec3(' + c.map(f).join(', ') + ')'; }
  Object.keys(S).forEach(function (k) {
    var v = S[k];
    if (typeof v === 'number') out.push('#define SL_' + k + ' ' + f(v));
    else if (Array.isArray(v) && v.length === 3 && v.some(function (q) { return q > 1; })) out.push('#define SL_' + k + ' ' + v3(skinLin(v)));
    else if (Array.isArray(v) && v.length === 2) { out.push('#define SL_' + k + '_0 ' + f(v[0])); out.push('#define SL_' + k + '_1 ' + f(v[1])); }
  });
  return out.join('\n') + '\n';
}
/* Blender's procedural textures (intern/cycles/kernel/svm and gpu_shader_material_*.glsl), as GLSL ES 3.0 */
var SKIN_NOISE_GLSL = [
  'uint sl_rot(uint x, uint k) { return (x << k) | (x >> (32u - k)); }',
  '#define SL_FINAL(a, b, c) { c ^= b; c -= sl_rot(b, 14u); a ^= c; a -= sl_rot(c, 11u); b ^= a; b -= sl_rot(a, 25u); c ^= b; c -= sl_rot(b, 16u); a ^= c; a -= sl_rot(c, 4u); b ^= a; b -= sl_rot(a, 14u); c ^= b; c -= sl_rot(b, 24u); }',
  '#define SL_MIX(a, b, c) { a -= c; a ^= sl_rot(c, 4u); c += b; b -= a; b ^= sl_rot(a, 6u); a += c; c -= b; c ^= sl_rot(b, 8u); b += a; a -= c; a ^= sl_rot(c, 16u); c += b; b -= a; b ^= sl_rot(a, 19u); a += c; c -= b; c ^= sl_rot(b, 4u); b += a; }',
  'uint sl_h2(uint kx, uint ky) { uint a, b, c; a = b = c = 0xdeadbeefu + (2u << 2u) + 13u; b += ky; a += kx; SL_FINAL(a, b, c); return c; }',
  'uint sl_h3(uint kx, uint ky, uint kz) { uint a, b, c; a = b = c = 0xdeadbeefu + (3u << 2u) + 13u; c += kz; b += ky; a += kx; SL_FINAL(a, b, c); return c; }',
  'uint sl_h4(uint kx, uint ky, uint kz, uint kw) { uint a, b, c; a = b = c = 0xdeadbeefu + (4u << 2u) + 13u; a += kx; b += ky; c += kz; SL_MIX(a, b, c); a += kw; SL_FINAL(a, b, c); return c; }',
  'float sl_uf(uint h) { return float(h) / float(0xFFFFFFFFu); }',
  'uint sl_iu(int x) { return x >= 0 ? uint(x) : ~uint(-x - 1); }',      // (an int's own bits, as C's cast keeps them)
  'vec3 sl_hash33i(ivec3 v) {',      // (Voronoi hashes the cell's integer coordinates with PCG3D, hash_pcg3d_i)
  '  v = v * 1664525 + 1013904223;',
  '  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;',
  '  v ^= (v >> 16);',
  '  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;',
  '  return vec3(v & 0x7FFFFFFF) * (1.0 / float(0x7FFFFFFF));',
  '}',
  'float sl_grad(uint hash, float x, float y, float z) {',
  '  uint h = hash & 15u; float u = h < 8u ? x : y; float vt = (h == 12u || h == 14u) ? x : z; float v = h < 4u ? y : vt;',
  '  return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);',
  '}',
  'float sl_fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }',
  'float sl_perlin(vec3 p) {',
  '  vec3 fl = floor(p), f = p - fl; int X = int(fl.x), Y = int(fl.y), Z = int(fl.z);',
  '  uint x0 = sl_iu(X), x1 = sl_iu(X + 1), y0 = sl_iu(Y), y1 = sl_iu(Y + 1), z0 = sl_iu(Z), z1 = sl_iu(Z + 1);',
  '  float u = sl_fade(f.x), v = sl_fade(f.y), w = sl_fade(f.z);',
  '  float a = mix(sl_grad(sl_h3(x0, y0, z0), f.x, f.y, f.z), sl_grad(sl_h3(x1, y0, z0), f.x - 1.0, f.y, f.z), u);',
  '  float b = mix(sl_grad(sl_h3(x0, y1, z0), f.x, f.y - 1.0, f.z), sl_grad(sl_h3(x1, y1, z0), f.x - 1.0, f.y - 1.0, f.z), u);',
  '  float c = mix(sl_grad(sl_h3(x0, y0, z1), f.x, f.y, f.z - 1.0), sl_grad(sl_h3(x1, y0, z1), f.x - 1.0, f.y, f.z - 1.0), u);',
  '  float d = mix(sl_grad(sl_h3(x0, y1, z1), f.x, f.y - 1.0, f.z - 1.0), sl_grad(sl_h3(x1, y1, z1), f.x - 1.0, f.y - 1.0, f.z - 1.0), u);',
  '  return mix(mix(a, b, v), mix(c, d, v), w);',
  '}',
  'float sl_snoise(vec3 p) { return 0.9820 * sl_perlin(p); }',
  'float sl_fbm(vec3 p, float detail, float rough) {',      // (the Noise texture's fBm, normalized, lacunarity 2)
  '  float fs = 1.0, amp = 1.0, maxamp = 0.0, sum = 0.0; int n = int(detail);',
  '  for (int i = 0; i <= 15; i++) { if (i > n) break; sum += sl_snoise(fs * p) * amp; maxamp += amp; amp *= rough; fs *= 2.0; }',
  '  float rmd = detail - floor(detail);',
  '  if (rmd != 0.0) { float sum2 = sum + sl_snoise(fs * p) * amp; return mix(0.5 * sum / maxamp + 0.5, 0.5 * sum2 / (maxamp + amp) + 0.5, rmd); }',
  '  return 0.5 * sum / maxamp + 0.5;',
  '}',
  'vec3 sl_off(float seed) { uint s = floatBitsToUint(seed); return vec3(100.0 + sl_uf(sl_h2(s, floatBitsToUint(0.0))) * 100.0, 100.0 + sl_uf(sl_h2(s, floatBitsToUint(1.0))) * 100.0, 100.0 + sl_uf(sl_h2(s, floatBitsToUint(2.0))) * 100.0); }',
  'float sl_noise(vec3 co, float scale, float detail, float rough) { return sl_fbm(co * scale, detail, rough); }',
  'vec3 sl_noiseColor(vec3 co, float scale, float detail, float rough) { vec3 p = co * scale; return vec3(sl_fbm(p, detail, rough), sl_fbm(p + sl_off(3.0), detail, rough), sl_fbm(p + sl_off(4.0), detail, rough)); }',
  'vec4 sl_voronoiF1(vec3 co, float scale) {',      // (distance, then the cell's colour)
  '  vec3 p = co * scale, cf = floor(p), loc = p - cf; ivec3 cell = ivec3(cf), tOff = ivec3(0); float md = 3.4e38;',
  '  for (int k = -1; k <= 1; k++) for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {',
  '    ivec3 o = ivec3(i, j, k); vec3 pp = vec3(o) + sl_hash33i(cell + o); float d = distance(pp, loc);',
  '    if (d < md) { md = d; tOff = o; }',
  '  }',
  '  return vec4(md, sl_hash33i(cell + tOff));',
  '}',
  'float sl_voronoiEdge(vec3 co, float scale) {',
  '  vec3 p = co * scale, cf = floor(p), loc = p - cf, vc = vec3(0.0); ivec3 cell = ivec3(cf); float md = 3.4e38;',
  '  for (int k = -1; k <= 1; k++) for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {',
  '    ivec3 o = ivec3(i, j, k); vec3 v = vec3(o) + sl_hash33i(cell + o) - loc; float d = dot(v, v);',
  '    if (d < md) { md = d; vc = v; }',
  '  }',
  '  md = 3.4e38;',
  '  for (int k = -1; k <= 1; k++) for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {',
  '    ivec3 o = ivec3(i, j, k); vec3 v = vec3(o) + sl_hash33i(cell + o) - loc; vec3 pe = v - vc;',
  '    if (dot(pe, pe) > 0.0001) md = min(md, dot((vc + v) / 2.0, normalize(pe)));',
  '  }',
  '  return md;',
  '}',
  // Map Range, smooth step and linear (both clamped), and Blender's safe divide
  'float sl_ms(float v, float a, float b, float lo, float hi) { float t = b != a ? clamp((v - a) / (b - a), 0.0, 1.0) : 0.0; t = (3.0 - 2.0 * t) * t * t; return lo + t * (hi - lo); }',
  'float sl_ml(float v, float a, float b, float lo, float hi) { float t = b != a ? clamp((v - a) / (b - a), 0.0, 1.0) : 0.0; return lo + t * (hi - lo); }',
  'float sl_div(float a, float b) { return b != 0.0 ? a / b : 0.0; }',
  'vec3 sl_warped(vec3 co, float scale, float amount) { return co + (sl_noiseColor(co, scale, 2.0, 0.5) - 0.5) * amount; }'
].join('\n') + '\n';
/* the bake: the head drawn flat over uv1, each point working out skin_layers' patterns at its rest position.
   Out: 0 layer 1's colour (sRGB-encoded) and the veins; 1 the veins' two hue ramps and the tone's red and yellow
   drift; 2 the sun spot, mole and blemish shapes, and the roughness noise */
var SKIN_BAKE_FRAG = [
  'precision highp float; precision highp int;',
  'in vec3 vRest;',
  'layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1; layout(location = 2) out vec4 o2;',
  'void main() {',
  '  vec3 R = vRest;',
  // layer 1: deep red and brown in broad mottled patches, and a fine speckle of capillaries
  '  vec3 c1 = mix(SL_MUSCLE_RED, SL_MUSCLE_BROWN, sl_ms(sl_noise(R, 1.5, 4.0, 0.6), 0.4, 0.6, 0.0, 1.0));',
  '  vec3 layer1 = c1 * sl_ml(sl_noise(R, 14.0, 3.0, 0.5), 0.3, 0.7, 0.88, 1.1);',
  // layer 2: wandering trunks and the thin branches springing off them, on twice-warped coordinates
  '  vec3 wv = sl_warped(sl_warped(R, SL_VEIN_BEND_0, SL_VEIN_BEND_1), SL_VEIN_WIGGLE_0, SL_VEIN_WIGGLE_1);',
  '  float td = sl_voronoiEdge(wv, SL_VEIN_TRUNK_SCALE), bd = sl_voronoiEdge(wv, SL_VEIN_BRANCH_SCALE);',
  '  float trunk = sl_ms(td, 0.0, SL_VEIN_TRUNK_WIDTH, 1.0, 0.0); trunk *= trunk;',
  '  float branch = sl_ms(bd, 0.0, SL_VEIN_BRANCH_WIDTH, 1.0, 0.0); branch *= branch;',
  '  float gate = sl_ms(sl_noise(R, 1.1, 1.0, 0.5), 0.4, 0.6, 0.0, 1.0);',
  '  trunk *= gate;',
  '  float near = sl_ms(td, 0.0, SL_VEIN_BRANCH_REACH, 1.0, 0.0) * gate;',
  '  branch = branch * near * (SL_VEIN_BRANCH_STRENGTH * sl_ms(sl_noise(R, 2.7, 2.0, 0.5), 0.45, 0.6, 0.0, 1.0));',
  '  float veins = max(trunk, branch) * sl_ms(sl_noise(R, 0.9, 2.0, 0.5), 0.3, 0.6, 0.45, 1.0);',
  '  float hue = sl_ms(sl_noise(R, 0.6, 1.0, 0.5), 0.45, 0.65, 0.0, 1.0), hue2 = sl_ms(sl_noise(R, 0.45, 2.0, 0.5), 0.55, 0.75, 0.0, 0.8);',
  // the melanin of layer 3 that this creator does not paint itself: sun spots, moles, blemishes
  '  vec3 bent = sl_warped(R, 9.0, 0.06);',
  '  vec4 sv = sl_voronoiF1(bent, 2.3);',
  '  float spot = sl_ms(sl_div(sv.x, sv.z * 0.2 + 0.25), 0.3, 1.0, 1.0, 0.0) * sl_ms(sv.y, 0.7, 0.8, 0.0, 1.0);',
  '  vec4 mv = sl_voronoiF1(R, 0.55);',
  '  float mole = sl_ms(sl_div(mv.x, mv.z * 0.035 + 0.03), 0.75, 1.0, 1.0, 0.0) * sl_ms(mv.y, 0.5, 0.55, 0.0, 1.0);',
  '  vec4 bv = sl_voronoiF1(R, 1.3);',
  '  float blem = sl_ms(sl_div(bv.x, 0.16), 0.4, 1.0, 1.0, 0.0) * sl_ms(bv.y, 0.82, 0.9, 0.0, 1.0);',
  // the tone's slow drift redder and yellower, and the patchy roughness
  '  vec3 dn = sl_noiseColor(R, 0.35, 3.0, 0.6);',
  '  float dR = sl_ms(dn.x, 0.45, 0.75, 0.0, SL_TONE_DRIFT), dY = sl_ms(dn.y, 0.5, 0.8, 0.0, SL_TONE_DRIFT * 0.7);',
  '  float rn = sl_ml(sl_noise(R, 2.2, 4.0, 0.5), 0.3, 0.7, 0.0, 1.0);',
  '  o0 = vec4(pow(clamp(layer1, 0.0, 1.0), vec3(1.0 / 2.2)), veins);',
  '  o1 = vec4(hue, hue2 / 0.8, dR / max(SL_TONE_DRIFT, 1e-4), dY / max(SL_TONE_DRIFT * 0.7, 1e-4));',
  '  o2 = vec4(spot, mole, blem, rn);',
  '}'
].join('\n');
var SKIN_BAKE = { done: false, pending: false, ref: null, ms: 0, cached: false };
/* bake the patterns once, and keep them: in the browser between visits (the startup cache, PC, when there is one), and
   across every shape slider and preset. The maps are over uv1, which every head shape shares, so the patterns ride on
   the skin as the face is reshaped, as a painted skin does, instead of the head moving through a fixed field. A first
   bake waits until the layers are on, and runs off the load: its shader compiles in the background (compileAsync), the
   head shows the creator's own skin meanwhile, and the layers come in when it is done. ref keeps the rest position the
   patterns were read at (the head as it was then, at a first visit the default head), for the Blender bench. */
function bakeSkinLayers(geo) {
  var B = SKIN_BAKE;
  if (B.done || B.pending) return;
  var t0 = performance.now(), key = 'skin|' + SKIN_BAKE_W + 'x' + SKIN_BAKE_H, PCs = typeof PC !== 'undefined' ? PC : null;
  var hit = PCs && PCs.get(key);
  if (hit && hit.px && hit.ref && hit.ref.length === geo.attributes.skinRest.array.length) { B.ref = hit.ref; B.cached = true; skinBakeUse(hit.px, t0); return; }
  var busy = document.getElementById('busy');
  if (!(P.skinLayers > 0) || (busy && !busy.hidden)) return;
  B.pending = true; B.ref = geo.attributes.skinRest.array.slice();
  var rt = new THREE.WebGLMultipleRenderTargets(SKIN_BAKE_W, SKIN_BAKE_H, 3, { depthBuffer: false });
  var mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, side: THREE.DoubleSide, depthTest: false, depthWrite: false,
    vertexShader: 'precision highp float;\nin vec3 position;\nin vec2 skinWrap;\nout vec3 vRest;\nvoid main() { vRest = position; gl_Position = vec4(skinWrap * 2.0 - 1.0, 0.0, 1.0); }',
    fragmentShader: 'precision highp float; precision highp int;\n' + skinGlslConsts() + SKIN_NOISE_GLSL + SKIN_BAKE_FRAG.replace('precision highp float; precision highp int;', '') });
  // (flat over the wrap-around coordinates: a triangle across the seam at the back is drawn twice, once each side,
  // so the maps, which repeat round the head, have no gap there)
  var R = B.ref, W = geo.attributes.skinWrap.array, I = geo.index.array, P_ = [], U_ = [];
  for (var t = 0; t < I.length; t += 3) {
    var a = I[t], b = I[t + 1], c = I[t + 2], ua = W[a * 2], ub = W[b * 2], uc = W[c * 2];
    var seam = Math.max(ua, ub, uc) - Math.min(ua, ub, uc) > 0.5;
    for (var side = 0; side < (seam ? 2 : 1); side++) [a, b, c].forEach(function (v) {
      var u = W[v * 2]; if (seam) u += side === 0 ? (u < 0.5 ? 1 : 0) : (u > 0.5 ? -1 : 0);
      P_.push(R[v * 3], R[v * 3 + 1], R[v * 3 + 2]); U_.push(u, W[v * 2 + 1]);
    });
  }
  var bg = new THREE.BufferGeometry();
  bg.setAttribute('position', new THREE.Float32BufferAttribute(P_, 3)); bg.setAttribute('skinWrap', new THREE.Float32BufferAttribute(U_, 2));
  var sc = new THREE.Scene(), cam = new THREE.Camera(), ms = new THREE.Mesh(bg, mat); ms.frustumCulled = false; sc.add(ms);
  function draw() {
    var prev = renderer.getRenderTarget(), cc = new THREE.Color(), ca = renderer.getClearAlpha(); renderer.getClearColor(cc);
    renderer.setRenderTarget(rt); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.render(sc, cam);
    // (read the three maps back: they become plain textures, and the startup cache keeps them)
    var gl = renderer.getContext(), n = SKIN_BAKE_W * SKIN_BAKE_H * 4, px = new Uint8Array(n * 3);
    for (var k = 0; k < 3; k++) { gl.readBuffer(gl.COLOR_ATTACHMENT0 + k); gl.readPixels(0, 0, SKIN_BAKE_W, SKIN_BAKE_H, gl.RGBA, gl.UNSIGNED_BYTE, px.subarray(n * k, n * (k + 1))); }
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    renderer.setRenderTarget(prev); renderer.setClearColor(cc, ca);
    rt.dispose(); mat.dispose(); bg.dispose();
    if (PCs) PCs.put(key, { px: px, ref: B.ref });
    B.pending = false; skinBakeUse(px, t0);
    if (typeof requestRender === 'function') requestRender();
  }
  var go = function () { setTimeout(draw, 0); };
  if (renderer.compileAsync) renderer.compileAsync(sc, cam).then(go, go); else go();
}
function skinBakeUse(px, t0) {
  var n = SKIN_BAKE_W * SKIN_BAKE_H * 4;
  [0, 1, 2].forEach(function (k) {
    var tx = new THREE.DataTexture(px.subarray(n * k, n * (k + 1)), SKIN_BAKE_W, SKIN_BAKE_H, THREE.RGBAFormat, THREE.UnsignedByteType);
    tx.wrapS = THREE.RepeatWrapping; tx.wrapT = THREE.ClampToEdgeWrapping; tx.magFilter = THREE.LinearFilter; tx.minFilter = THREE.LinearMipmapLinearFilter;
    tx.generateMipmaps = true; tx.anisotropy = 4; tx.needsUpdate = true;
    skinU['skinT' + k].value = tx;
  });
  SKIN_BAKE.done = true; SKIN_BAKE.ms = performance.now() - t0; skinU.skinBaked.value = 1;
}
/* called as the head is drawn: once the layers are on and the creator has finished loading (its busy sign is down),
   start the bake, half a second on, so the face is up and answering first */
function skinBakeTick(geo) {
  var busy = document.getElementById('busy');
  if (SKIN_BAKE.done || SKIN_BAKE.pending || SKIN_BAKE.sched || !(P.skinLayers > 0) || !geo.attributes.skinRest || (busy && !busy.hidden)) return;
  SKIN_BAKE.sched = true; setTimeout(function () { SKIN_BAKE.sched = false; bakeSkinLayers(geo); }, 500);
}
/* where the layers show on the head: skin_tones' (female.py, 'Head shape') thin skin and freckle maps, from each point
   (in head heights) and its normal, in model axes; paint_face's hold of the face's thin skin to 0.3 */
function skinZonesAt(u, w, ny, nz) {
  function g(a, b, a0, b0, sa, sb) { var p = (a - a0) / sa, q = (b - b0) / sb; return Math.exp(-p * p - q * q); }
  var front = clamp(-ny * 1.5, 0, 1);
  var cold = front * 0.8 * (g(u, w, -0.14, 0.47, 0.065, 0.025) + g(u, w, 0.14, 0.47, 0.065, 0.025));
  cold += clamp(-nz - 0.25, 0, 1) * clamp((0.3 - w) / 0.2, 0, 1) * 1.1 + clamp((0.02 - w) / 0.1, 0, 1) * 0.7;
  var thin = front * clamp(0.9 * (g(u, w, -0.215, 0.5, 0.045, 0.075) + g(u, w, 0.215, 0.5, 0.045, 0.075)) + 0.7 * g(u, w, 0, 0.5, 0.03, 0.06) + 0.8 * cold, 0, 1);
  var freck = front * clamp(g(u, w, 0, 0.42, 0.3, 0.075) + 0.3 * g(u, w, 0, 0.68, 0.2, 0.06), 0, 1);
  return [freck, Math.min(thin, 0.3)];
}
/* the mask the creator paints beside the face (its planar uv): red, its own freckles (layer 3 is opaque there and
   holds them out of the scatter); green, what is not skin (the lips, the waterline, brows and lashes) */
var skinMaskCanvas = document.createElement('canvas'); skinMaskCanvas.width = TW; skinMaskCanvas.height = TH;
var skinMaskCtx = skinMaskCanvas.getContext('2d'), skinMaskImg = skinMaskCtx.createImageData(TW, TH);
var skinMaskTmp = document.createElement('canvas'); skinMaskTmp.width = TW; skinMaskTmp.height = TH;
function skinMaskOver(layer) {      // (a brow or lash layer's coverage, added to the mask's green)
  var c = skinMaskTmp.getContext('2d'); c.globalCompositeOperation = 'copy'; c.drawImage(layer, 0, 0);
  c.globalCompositeOperation = 'source-in'; c.fillStyle = '#00ff00'; c.fillRect(0, 0, TW, TH); c.globalCompositeOperation = 'source-over';
  skinMaskCtx.globalCompositeOperation = 'lighter'; skinMaskCtx.drawImage(skinMaskTmp, 0, 0); skinMaskCtx.globalCompositeOperation = 'source-over';
}
var skinMaskTex = new THREE.CanvasTexture(skinMaskCanvas); skinMaskTex.anisotropy = 4;
var skinU = { skinT0: { value: null }, skinT1: { value: null }, skinT2: { value: null }, skinMask: { value: skinMaskTex }, skinBaked: { value: 0 },
  skinLayers: { value: 1 }, skinVeins: { value: 1 }, skinSpots: { value: 1 }, skinMoles: { value: 1 }, skinPores: { value: 1 },
  skinWrapRGB: { value: new THREE.Vector3() }, skinBounceDir: { value: new THREE.Vector3(0, -1, 0) }, skinBounceCol: { value: new THREE.Vector3() } };
/* the bounce: views.py's warm area light from below, at its share of the key's light on the head (energy over the
   distance squared, from the figure's head) times this creator's key */
var SKIN_BOUNCE_W = new THREE.Vector3();
(function () {
  var L = SKIN_LAYERS.LIGHTS, F = SKIN_LAYERS.FIGURE, hc = [0, 0, F.HEAD_Z + 0.5 * F.HEAD_H];
  function d2(p) { return Math.pow(p[0] - hc[0], 2) + Math.pow(p[1] - hc[1], 2) + Math.pow(p[2] - hc[2], 2); }
  var share = (L.Bounce.energy / d2(L.Bounce.loc)) / (L.Key.energy / d2(L.Key.loc));
  SKIN_BOUNCE_W.set(L.Bounce.loc[0] - L.Bounce.aim[0], L.Bounce.loc[2] - L.Bounce.aim[2], -(L.Bounce.loc[1] - L.Bounce.aim[1])).normalize();      // (toward the light, in three.js axes)
  SKIN_BOUNCE_W.share = share; SKIN_BOUNCE_W.col = L.Bounce.colour;
})();
function skinLayersShader(sh) {
  Object.keys(skinU).forEach(function (k) { sh.uniforms[k] = skinU[k]; });
  sh.vertexShader = 'attribute vec2 skinWrap;\nattribute vec2 skinZones;\nvarying vec2 vSkinWrap;\nvarying vec2 vSkinZones;\n' +
    sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vSkinWrap = skinWrap; vSkinZones = skinZones;');
  var pre = skinGlslConsts() + 'uniform sampler2D skinT0, skinT1, skinT2, skinMask;\nuniform float skinBaked, skinLayers, skinVeins, skinSpots, skinMoles, skinPores;\n' +
    'uniform vec3 skinWrapRGB, skinBounceDir, skinBounceCol;\nvarying vec2 vSkinWrap;\nvarying vec2 vSkinZones;\nfloat skHeld = 0.0, skPit = 0.0, skRough = 0.0, skLay = 0.0, skSh = 1.0;\n' +
    'float sk_ms(float v, float a, float b) { float t = clamp((v - a) / (b - a), 0.0, 1.0); return (3.0 - 2.0 * t) * t * t; }\n';
  var body = [
    '#include <color_fragment>',
    '  {',
    '    vec4 mk = texture2D(skinMask, vMapUv);',
    '    skLay = skinLayers * skinBaked * (1.0 - mk.g);',
    '    if (skLay > 0.0) {',      // (none of it when the layers are off or not baked yet)
    '    vec4 t0 = texture2D(skinT0, vSkinWrap), t1 = texture2D(skinT1, vSkinWrap), t2 = texture2D(skinT2, vSkinWrap);',
    '    vec3 ep = diffuseColor.rgb;',
    '#ifdef USE_BUMPMAP',
    '    skPit = clamp((0.6 - texture2D(bumpMap, vBumpMapUv).r) * 3.0, 0.0, 1.0);',      // (the creator's own pits, already shaded by poreTint)
    '#endif',
    '    vec3 layer1 = pow(t0.rgb, vec3(2.2)) * skinGain;',
    '    float shown = sk_ms(vSkinZones.y, 0.15, 0.6);',
    '    float veins = t0.a * shown * skinVeins;',
    '    vec3 vs = mix(SL_VEIN_SEEN_BLUE, SL_VEIN_SEEN_PURPLE, t1.r); vs = mix(vs, SL_VEIN_SEEN_GREEN, t1.g * 0.8);',
    '    float spot = t2.r * (vSkinZones.x * 0.8 + 0.2) * SL_SUNSPOTS * skinSpots, mole = t2.g * SL_MOLES * skinMoles, blem = t2.b, fr = mk.r;',
    '    vec3 c3 = mix(ep, SL_SUNSPOT, clamp(spot * 0.5, 0.0, 1.0));',
    '    c3 = mix(c3, SL_MOLE, clamp(mole * 0.92, 0.0, 1.0));',
    '    c3 = mix(c3, SL_BLEMISH, clamp(blem * 0.55, 0.0, 1.0));',
    '    float op = clamp(SL_EPIDERMIS_OPACITY + max(max(fr, max(spot * 0.6, mole)), blem * 0.7) * (1.0 - SL_EPIDERMIS_OPACITY), 0.0, 1.0);',
    '    vec3 comb = mix(layer1, c3, op);',
    '    comb = mix(comb, comb * vs, clamp(veins * SL_VEIN_SEEN, 0.0, 1.0));',
    '    comb = mix(comb, SL_TONE_RED, t1.b * SL_TONE_DRIFT);',
    '    comb = mix(comb, SL_TONE_YELLOW, t1.a * SL_TONE_DRIFT * 0.7);',
    '    diffuseColor.rgb = mix(diffuseColor.rgb, comb, skLay);',
    '    float pig = max(fr, max(spot * 0.5, mole));',
    '    skHeld = skLay * min(1.0, max(pig * SL_PIGMENT_HOLD, skPit * SL_PORES * skinPores * SL_PORE_HOLD));',
    '    skRough = mix(SL_ROUGH_0, SL_ROUGH_1, t2.a) + skPit * 0.18;',
    '    }',
    '  }'
  ].join('\n');
  var fs = sh.fragmentShader.replace('#include <color_fragment>', body);
  // the roughness: Blender's patches plus the pits, keeping this creator's own zones (the oil on the nose, the lips' and
  // the waterline's wet) as their difference from its plain skin (0.56)
  fs = fs.replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\n  roughnessFactor = mix(roughnessFactor, clamp(skRough + roughnessFactor - 0.56, 0.05, 1.0), skLay);');
  // the scatter's stand-in: wrapped diffuse light, red furthest; none where the skin holds pigment or a pit
  // (the shadow map's self-shadow starts at the terminator: a convex cheek shadows itself where it turns from the light.
  // The plain light is zero there, but the scatter's is not, and the shadow cut it into a hard edge across the cheek. So
  // the key's shadow is kept apart (skSh): the specular takes it whole, the diffuse lets it go as the surface turns away,
  // where only self-shadow can sit. A cast shadow (the nose's) falls on skin that faces the light and stays.)
  fs = fs.replace('#include <lights_physical_pars_fragment>', THREE.ShaderChunk.lights_physical_pars_fragment
    .replace('irradiance = dotNL * directLight.color;', 'irradiance = dotNL * directLight.color * skSh;')
    .replace('ccIrradiance = dotNLcc * directLight.color;', 'ccIrradiance = dotNLcc * directLight.color * skSh;')
    .replace('reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
    'float skNL = dot( geometryNormal, directLight.direction ); vec3 skW = skinWrapRGB * ( 1.0 - skHeld );\n' +
    '\tfloat skVis = mix( skSh, mix( 1.0, skSh, smoothstep( 0.0, 0.3, skNL ) ), skLay );\n' +
    '\treflectedLight.directDiffuse += clamp( ( vec3( skNL ) + skW ) / ( 1.0 + skW ), 0.0, 1.0 ) / ( 1.0 + skW ) * skVis * directLight.color * BRDF_Lambert( material.diffuseColor );'));
  var LFB = THREE.ShaderChunk.lights_fragment_begin, SHL = 'directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ]';
  if (LFB.indexOf(SHL) < 0) throw new Error('skin layers: three.js shadow chunk changed');
  fs = fs.replace('#include <lights_fragment_begin>', LFB
    .replace('getDirectionalLightInfo( directionalLight, directLight );', 'getDirectionalLightInfo( directionalLight, directLight ); skSh = 1.0;')
    .replace(SHL, 'skSh = ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ]'));      // (normalised: the red that reaches past the terminator is taken from the lit side, as the scatter moves it, not added)
  // the bounce from below (the skin alone)
  fs = fs.replace('#include <lights_fragment_maps>', '#include <lights_fragment_maps>\n  reflectedLight.indirectDiffuse += skinBounceCol * clamp((dot(normal, skinBounceDir) + 0.5) / 1.5, 0.0, 1.0) * BRDF_Lambert(material.diffuseColor);');
  sh.fragmentShader = pre + fs;
}
