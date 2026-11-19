// Export a Face Creator hairstyle as an Emberwood hair asset (FORMAT.md): every piece of hair exactly as the creator
// draws it (the cap, every strand, the tied pieces, the sheer veils), its materials as the creator's shader has them,
// its three textures and the creator's own viewer (lights, environment, tone mapping), in
// one .glb. Optionally renders reference views of the creator on fixed cameras for the Blender bench to match.
//
//   node tools/hair-asset/export.cjs <face-creator.html> <style> <out.hair.glb> [--face Warden]
//        [--views front,three,profile,back,back3] [--size 512] [--looks full,clay] [--probe] [--head head.glb]
//
// --views writes <out dir>/creator-<view>-<look>.png (transparent background, the head in clay for 'clay' and as the
// creator paints it for 'full') and records the cameras in the asset, so hair_bench.py renders the same shots.
// --probe measures the creator's room environment (the light a matte white surface gets from it, by normal) and
// stores it in the asset's viewer block. Needs playwright (with Chromium) and network access to cdn.jsdelivr.net.
'use strict';
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const { GlbWriter } = require('./glb.cjs');

const argv = process.argv.slice(2);
const flag = (k, d) => { const i = argv.indexOf(k); return i < 0 ? d : argv[i + 1]; };
const [page, style, out] = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--')));
if (!page || !style || !out) { console.log('usage: export.cjs <face-creator.html> <style> <out.hair.glb> [--face F] [--views ...] [--probe]'); process.exit(2); }
const face = flag('--face', null), views = (flag('--views', '') || '').split(',').filter(Boolean);
const size = +flag('--size', 512), looks = (flag('--looks', 'full,clay')).split(',');
const probe = argv.includes('--probe'), headOut = flag('--head', null);

// The creator's cut-out shader, as the creator injects it (onBeforeCompile), with every number taken out. An export
// whose shader does not have exactly this shape stops: the readers implement this model and nothing else, so a change
// to the creator's shader has to be carried into FORMAT.md and both readers first (bump the model's version).
const CUT_MODEL = 'emberwood-cut-1';
const CUT_PARAMS = ['tipW', 'tipH', 'farPx0', 'farPx1', 'gainNum', 'gainNear', 'gainFar', 'inOne', 'in0', 'in1', 'farOne',
  'inFar', 'du2', 'dv2', 'dv2min', 'du3', 'dv3', 'dv3min', 'shade', 'hairlineOn', 'hairlineBias', 'hairlineGain',
  'hairlineA2', 'hairlineA3', 'cut1', 'toneOne', 'toneGain', 'cut2', 'tone2', 'tone3'];
const NUM = /(\d+\.\d+|\d+\.(?!\w)|\.\d+)/g;

(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 400, height: 300 } });
  p.on('pageerror', e => { console.log('pageerror:', e.message); process.exit(1); });
  await p.goto('file://' + path.resolve(page));
  const idle = () => p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 600000 });
  await idle();
  if (face) { await p.locator(`[data-face="${face}"]`).click(); await p.waitForTimeout(500); await idle(); }
  const r = await p.evaluate(async (opt) => {
    const THREE = await import('three');
    const fc = window.__fc, ST = fc.HAIR_STYLES[opt.style];
    if (!ST) return { error: 'no such style: ' + opt.style };
    // the style's own pieces: what its extras() returns is what the creator adds to the scene as the hair
    let extras = null; const ex0 = ST.extras;
    if (ex0) ST.extras = function (m) { extras = ex0.call(this, m) || []; return extras; };
    // (a fresh build, never the creator's cache of built hair; with the lock log on, which also skips the cache)
    window.__hairLog = []; fc.state.hairKey = null;
    fc.P.hairStyle = opt.style; fc.P.hair = true; fc.update();
    ST.extras = ex0;
    const log = window.__hairLog || []; window.__hairLog = null;
    if (ex0 && !extras) return { error: 'the style did not rebuild (extras not called)' };
    const scene = fc.hairMesh.parent; scene.updateMatrixWorld(true);
    const meshes = [fc.hairMesh];
    (extras || []).forEach(o => o.traverse(m => { if (m.isMesh) meshes.push(m); }));
    const b64 = a => { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
    const tipTex = fc.hairCapMat.alphaMap, hairTex = fc.hairCapMat.map, bumpTex = fc.hairCapMat.bumpMap;
    // ---- materials: one entry per creator material, with what its onBeforeCompile injects
    const mats = [], matIx = new Map();
    function inject(m) {
      const sh = { uniforms: {}, vertexShader: '#include <begin_vertex>', fragmentShader: '#include <color_fragment>\n#include <alphamap_fragment>' };
      if (m.onBeforeCompile && m.onBeforeCompile.toString() !== THREE.Material.prototype.onBeforeCompile.toString()) m.onBeforeCompile(sh, fc.renderer);
      const u = {}; Object.keys(sh.uniforms).forEach(k => { u[k] = sh.uniforms[k].value; });
      return { vertex: sh.vertexShader, fragment: sh.fragmentShader, uniforms: u };
    }
    function material(m) {
      if (matIx.has(m)) return matIx.get(m);
      const e = { name: m.name || m.type, threeType: m.type, side: m.side === THREE.DoubleSide ? 'double' : m.side === THREE.BackSide ? 'back' : 'front',
        vertexColors: !!m.vertexColors, color: m.color ? [m.color.r, m.color.g, m.color.b] : [1, 1, 1], opacity: m.opacity,
        transparent: !!m.transparent, depthWrite: m.depthWrite, alphaTest: m.alphaTest || 0, toneMapped: m.toneMapped !== false };
      if (m.isMeshStandardMaterial) Object.assign(e, { roughness: m.roughness, metalness: m.metalness, envMapIntensity: m.envMapIntensity, bumpScale: m.bumpScale });
      e.maps = { map: m.map === hairTex ? 'hair' : m.map ? 'other' : null, bump: m.bumpMap === bumpTex ? 'bump' : m.bumpMap ? 'other' : null, alpha: m.alphaMap === tipTex ? 'tips' : m.alphaMap ? 'other' : null };
      if (m.blending === THREE.CustomBlending) e.blend = { src: m.blendSrc, dst: m.blendDst, srcAlpha: m.blendSrcAlpha, dstAlpha: m.blendDstAlpha };
      e.kind = e.maps.alpha === 'tips' && e.alphaTest > 0 ? 'cut' : m.isMeshBasicMaterial && m.transparent ? 'veil' : e.maps.map === 'hair' ? 'shell' : 'standard';
      e.shader = inject(m);
      mats.push(e); matIx.set(m, mats.length - 1); return mats.length - 1;
    }
    // ---- pieces: every mesh's geometry as drawn (world space: the creator's head frame). The creator's tubes (the
    // strands, the tied pieces, the veils) are stored as their rings (FORMAT.md, "Strands"): a centre, two axes, a
    // colour and the texture's place per ring, which rebuilds every vertex, normal and triangle of the tube exactly.
    // Each tube is rebuilt here and checked against the creator's own mesh; a piece that does not rebuild within
    // TOL is stored as a plain mesh instead. Everything else (the cap) is a plain mesh, packed per material.
    const TOL = 1e-4, TOL_N = 1e-3;      // (model units, colours and texture coordinates; normals: 1e-3 is 0.06 degrees)
    const packs = {}, pieces = [], R = { centre: [], axisA: [], axisB: [], color: [], tex: [] }, stats = { tubes: 0, raw: 0, worst: 0 };
    function decodeTube(t, rg) {
      // rg: { centre, axisA, axisB, color, tex } arrays for this tube's rings only (flat)
      const S1 = t.ringSize, S = S1 - 1, n = t.rings, nv = n * S1 + 2, pos = new Float64Array(nv * 3), col = new Float64Array(nv * 4), uv = new Float64Array(nv * 2), al = new Float64Array(nv), idx = [];
      for (let i = 0; i < n; i++) for (let s = 0; s < S1; s++) {
        const w = 2 * Math.PI * s / S, c = Math.cos(w), sn = Math.sin(w), f = 0.9 + 0.1 * sn, q = i * S1 + s;
        for (let d = 0; d < 3; d++) pos[q * 3 + d] = rg.centre[i * 3 + d] + c * rg.axisA[i * 3 + d] + sn * rg.axisB[i * 3 + d];
        for (let d = 0; d < 3; d++) col[q * 4 + d] = rg.color[i * 4 + d] * f;
        col[q * 4 + 3] = rg.color[i * 4 + 3];
        uv[q * 2] = t.u0 + (s / S) * rg.tex[i * 3]; uv[q * 2 + 1] = rg.tex[i * 3 + 1]; al[q] = rg.tex[i * 3 + 2];
      }
      for (let e = 0; e < 2; e++) {
        const i = e ? n - 1 : 0, q = n * S1 + e;
        for (let d = 0; d < 3; d++) pos[q * 3 + d] = rg.centre[i * 3 + d];
        for (let d = 0; d < 4; d++) col[q * 4 + d] = t.endColor[e][d];
        uv[q * 2] = t.u0; uv[q * 2 + 1] = rg.tex[i * 3 + 1]; al[q] = t.endAlong[e];
      }
      for (let i = 0; i < n - 1; i++) for (let s = 0; s < S; s++) { const a = i * S1 + s, b = a + 1; idx.push(a, b, a + S1, b, b + S1, a + S1); }
      for (let e = 0; e < 2; e++) { const i0 = e ? n - 1 : 0, c = n * S1 + e; for (let s = 0; s < S; s++) { const a = i0 * S1 + s, b = a + 1; if (e) idx.push(a, c, b); else idx.push(a, b, c); } }
      // normals as three.js's computeVertexNormals makes them: each face's (c - b) x (a - b), summed, normalised
      const nrm = new Float64Array(nv * 3);
      for (let k = 0; k < idx.length; k += 3) {
        const A = idx[k] * 3, B = idx[k + 1] * 3, C = idx[k + 2] * 3;
        const cbx = pos[C] - pos[B], cby = pos[C + 1] - pos[B + 1], cbz = pos[C + 2] - pos[B + 2], abx = pos[A] - pos[B], aby = pos[A + 1] - pos[B + 1], abz = pos[A + 2] - pos[B + 2];
        const nx = cby * abz - cbz * aby, ny = cbz * abx - cbx * abz, nz = cbx * aby - cby * abx;
        for (const V of [A, B, C]) { nrm[V] += nx; nrm[V + 1] += ny; nrm[V + 2] += nz; }
      }
      for (let q = 0; q < nv; q++) { const l = Math.hypot(nrm[q * 3], nrm[q * 3 + 1], nrm[q * 3 + 2]) || 1; nrm[q * 3] /= l; nrm[q * 3 + 1] /= l; nrm[q * 3 + 2] /= l; }
      return { pos, nrm, col, uv, al, idx };
    }
    function encodeTube(W, S1) {
      const S = S1 - 1, n = (W.n - 2) / S1, rg = { centre: [], axisA: [], axisB: [], color: [], tex: [] };
      for (let i = 0; i < n; i++) {
        const c = [0, 0, 0], A = [0, 0, 0], B = [0, 0, 0];
        for (let s = 0; s < S; s++) for (let d = 0; d < 3; d++) c[d] += W.pos[(i * S1 + s) * 3 + d] / S;
        for (let s = 0; s < S; s++) { const w = 2 * Math.PI * s / S; for (let d = 0; d < 3; d++) { const x = W.pos[(i * S1 + s) * 3 + d] - c[d]; A[d] += 2 / S * Math.cos(w) * x; B[d] += 2 / S * Math.sin(w) * x; } }
        const col = [0, 0, 0]; let ff = 0;
        for (let s = 0; s < S1; s++) { const f = 0.9 + 0.1 * Math.sin(2 * Math.PI * s / S); ff += f * f; for (let d = 0; d < 3; d++) col[d] += f * W.col[(i * S1 + s) * 4 + d]; }
        rg.centre.push(...c); rg.axisA.push(...A); rg.axisB.push(...B);
        rg.color.push(col[0] / ff, col[1] / ff, col[2] / ff, W.col[i * S1 * 4 + 3]);
        rg.tex.push(W.uv[(i * S1 + S) * 2] - W.uv[i * S1 * 2], W.uv[i * S1 * 2 + 1], W.al[i * S1]);
      }
      Object.keys(rg).forEach(k => { rg[k] = rg[k].map(Math.fround); });      // (checked as stored: float32)
      const e0 = n * S1, e1 = e0 + 1;
      const t = { ringSize: S1, rings: n, u0: Math.fround(W.uv[0]), endColor: [Array.from(W.col.slice(e0 * 4, e0 * 4 + 4), Math.fround), Array.from(W.col.slice(e1 * 4, e1 * 4 + 4), Math.fround)], endAlong: [Math.fround(W.al[e0]), Math.fround(W.al[e1])] };
      // the check: rebuild, and compare everything with the creator's mesh
      const D = decodeTube(t, rg);
      if (D.idx.length !== W.idx.length) return null;
      for (let k = 0; k < D.idx.length; k++) if (D.idx[k] !== W.idx[k]) return null;
      let err = 0; const cmp = (a, b, k0) => { for (let k = 0; k < a.length; k++) err = Math.max(err, Math.abs(a[k] - b[k]) * k0); };
      cmp(D.pos, W.pos, 1); cmp(D.nrm, W.nrm, TOL / TOL_N); cmp(D.col, W.col, 1); cmp(D.uv, W.uv, 1); cmp(D.al, W.al, 1);
      if (!(err <= TOL)) return { fail: err };      // (err: in units of TOL's scale, normals' error scaled down to it)
      return { t, rg, err };
    }
    meshes.forEach((m) => {
      if (!m.visible) return;
      const g = m.geometry, mi = material(m.material), ud = g.userData || {};
      const P = g.attributes.position, N = g.attributes.normal, U = g.attributes.uv, C = g.attributes.color, n = P.count;
      const nm = new THREE.Matrix3().getNormalMatrix(m.matrixWorld), v = new THREE.Vector3();
      const W = { n, pos: new Float64Array(n * 3), nrm: new Float64Array(n * 3), uv: new Float64Array(n * 2), col: new Float64Array(n * 4), al: new Float64Array(n), idx: null };
      for (let i = 0; i < n; i++) {
        v.fromBufferAttribute(P, i).applyMatrix4(m.matrixWorld); W.pos.set([v.x, v.y, v.z], i * 3);
        if (N) { v.fromBufferAttribute(N, i).applyMatrix3(nm).normalize(); W.nrm.set([v.x, v.y, v.z], i * 3); } else W.nrm.set([0, 1, 0], i * 3);
        W.uv.set([U ? U.getX(i) : 0, U ? 1 - U.getY(i) : 1], i * 2);      // (glTF's v runs down the image: 1 - three's)
        W.col.set(C ? [C.getX(i), C.getY(i), C.getZ(i), C.itemSize > 3 ? C.getW(i) : 1] : [1, 1, 1, 1], i * 4);
        W.al[i] = ud.along ? ud.along[i] : -1;
      }
      const I = g.index ? g.index.array : null;
      W.idx = I ? Array.from(I) : Array.from({ length: n }, (_, k) => k);
      const pc = { role: m === fc.hairMesh ? 'cap' : 'piece', material: mi, renderOrder: m.renderOrder, castShadow: m.castShadow, receiveShadow: m.receiveShadow };
      ['len', 'tipLen', 'rootLen', 'lock', 'layer', 'tipFrac', 'thin'].forEach(k => { if (ud[k] !== undefined) pc[k] = ud[k]; });
      let enc = null;
      if (m !== fc.hairMesh && I && I[0] === 0 && I[1] === 1 && I[2] > 2 && (n - 2) % I[2] === 0 && (n - 2) / I[2] >= 2) {
        // a tube (the creator's tube(): rings of S + 1 points, the last repeating the first, then its two end points)
        pc.role = m.material.isMeshBasicMaterial ? 'veil' : (ud.lock !== undefined ? 'lock-strand' : 'strand');
        enc = encodeTube(W, I[2]);
        if (enc && enc.fail !== undefined) { pc.rawReason = 'rebuild error ' + enc.fail.toExponential(2); enc = null; }
      }
      if (enc) {
        enc.t.firstRing = R.centre.length / 3;
        Object.keys(R).forEach(k => R[k].push(...enc.rg[k]));
        pc.tube = enc.t; stats.tubes++; stats.worst = Math.max(stats.worst, enc.err);
      } else {
        const pk = packs[mi] || (packs[mi] = { pos: [], nrm: [], uv: [], col: [], along: [], piece: [], idx: [], nv: 0 });
        const base = pk.nv, i0 = pk.idx.length;
        for (let i = 0; i < n; i++) {
          pk.pos.push(W.pos[i * 3], W.pos[i * 3 + 1], W.pos[i * 3 + 2]); pk.nrm.push(W.nrm[i * 3], W.nrm[i * 3 + 1], W.nrm[i * 3 + 2]);
          pk.uv.push(W.uv[i * 2], W.uv[i * 2 + 1]); pk.col.push(W.col[i * 4], W.col[i * 4 + 1], W.col[i * 4 + 2], W.col[i * 4 + 3]);
          pk.along.push(W.al[i]); pk.piece.push(pieces.length);
        }
        W.idx.forEach(k => pk.idx.push(base + k));
        pk.nv += n;
        Object.assign(pc, { firstVertex: base, vertexCount: n, firstIndex: i0, indexCount: pk.idx.length - i0 });
        if (m === fc.hairMesh) pc.grid = { rows: I[1], cols: n / I[1] };
        stats.raw++;
      }
      pieces.push(pc);
    });
    const prims = Object.keys(packs).map(mi => {
      const pk = packs[mi];
      return { material: +mi, pos: b64(new Float32Array(pk.pos)), nrm: b64(new Float32Array(pk.nrm)), uv: b64(new Float32Array(pk.uv)), col: b64(new Float32Array(pk.col)),
        along: b64(new Float32Array(pk.along)), piece: b64(new Float32Array(pk.piece)), idx: b64(new Uint32Array(pk.idx)), nv: pk.nv };
    });
    // the depth (shadow) material the cut pieces cast with
    const dm = fc.hairMesh.customDepthMaterial, dsh = { uniforms: {}, vertexShader: '#include <begin_vertex>', fragmentShader: '#include <alphamap_fragment>' };
    if (dm && dm.onBeforeCompile) dm.onBeforeCompile(dsh);
    const png = t => t.image.toDataURL('image/png').split(',')[1];
    const tex = {
      hair: { png: png(hairTex), colorSpace: hairTex.colorSpace, wrapS: hairTex.wrapS === THREE.RepeatWrapping ? 'repeat' : 'clamp', wrapT: hairTex.wrapT === THREE.RepeatWrapping ? 'repeat' : 'clamp', width: hairTex.image.width, height: hairTex.image.height },
      bump: { png: png(bumpTex), colorSpace: bumpTex.colorSpace, wrapS: bumpTex.wrapS === THREE.RepeatWrapping ? 'repeat' : 'clamp', wrapT: bumpTex.wrapT === THREE.RepeatWrapping ? 'repeat' : 'clamp', width: bumpTex.image.width, height: bumpTex.image.height },
      tips: { png: png(tipTex), colorSpace: tipTex.colorSpace, wrapS: tipTex.wrapS === THREE.RepeatWrapping ? 'repeat' : 'clamp', wrapT: tipTex.wrapT === THREE.RepeatWrapping ? 'repeat' : 'clamp', width: tipTex.image.width, height: tipTex.image.height },
    };
    // the lock guides the creator logged for its hair check while it built (window.__hairLog, filled by hairClump)
    const guides = log.map(L => ({ seed: L.seed, path: L.pth.map(q => [q[0] * 8.6, q[2] * 8.6, -q[1] * 8.6]) }));
    // ---- the creator's viewer: its lights, environment and tone mapping, so a renderer can be checked against it
    const r0 = fc.renderer, lights = [];
    scene.traverse(o => {
      if (!o.isLight || !o.visible) return;
      const L = { type: o.type, color: [o.color.r, o.color.g, o.color.b], intensity: o.intensity };
      if (o.isDirectionalLight) { const d = new THREE.Vector3().subVectors(o.position, o.target.position).normalize(); L.toLight = [d.x, d.y, d.z]; L.castShadow = o.castShadow; }
      if (o.isHemisphereLight) { L.groundColor = [o.groundColor.r, o.groundColor.g, o.groundColor.b]; L.up = [0, 1, 0]; }
      lights.push(L);
    });
    const viewer = { renderer: 'three.js r' + THREE.REVISION, toneMapping: r0.toneMapping === THREE.ACESFilmicToneMapping ? 'aces-filmic-three' : String(r0.toneMapping),
      exposure: r0.toneMappingExposure, output: 'srgb', lights: lights, environment: 'RoomEnvironment (PMREM, sigma 0.04)' };
    // optional: the creator's room environment itself, as an equirectangular map in Blender's layout (u round from
    // -X, v up from the bottom row; linear, before tone mapping and before any material's envMapIntensity), so a
    // renderer gets the same light from it, diffuse and specular alike
    if (opt.probe) {
      const { RoomEnvironment } = await import('three/addons/environments/RoomEnvironment.js');
      const room = new RoomEnvironment(), crt = new THREE.WebGLCubeRenderTarget(256, { type: THREE.FloatType, generateMipmaps: false });
      const cc = new THREE.CubeCamera(0.1, 100, crt); room.add(cc); cc.update(r0, room);
      const EW = 512, EH = 256, ert = new THREE.WebGLRenderTarget(EW, EH, { type: THREE.FloatType });
      const q = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({ uniforms: { env: { value: crt.texture } },
        vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
        fragmentShader: 'uniform samplerCube env; varying vec2 vUv; void main() { float lon = (vUv.x - 0.5) * 6.2831853, lat = (vUv.y - 0.5) * 3.1415927;' +
          ' vec3 b = vec3(-cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat));' +      // (Blender's direction for that pixel)
          ' gl_FragColor = vec4(textureCube(env, vec3(b.x, b.z, -b.y)).rgb, 1.0); }', depthTest: false }));
      const qs = new THREE.Scene(); qs.add(q);
      const tm = r0.toneMapping, cs = r0.outputColorSpace; r0.toneMapping = THREE.NoToneMapping; r0.outputColorSpace = THREE.LinearSRGBColorSpace;
      r0.setRenderTarget(ert); r0.render(qs, new THREE.Camera()); const eb = new Float32Array(EW * EH * 4); r0.readRenderTargetPixels(ert, 0, 0, EW, EH, eb); r0.setRenderTarget(null);
      r0.toneMapping = tm; r0.outputColorSpace = cs;
      const rgb = new Float32Array(EW * EH * 3); for (let k = 0; k < EW * EH; k++) { rgb[k * 3] = eb[k * 4]; rgb[k * 3 + 1] = eb[k * 4 + 1]; rgb[k * 3 + 2] = eb[k * 4 + 2]; }
      viewer.environmentMap = { layout: 'equirect-blender', width: EW, height: EH, data: b64(rgb), note: 'linear radiance; the creator scales it by each material envMapIntensity (hair 0.25)' };
      // and what three.js's own shading takes from it: a matte white (roughness 1) sphere lit by the environment alone,
      // seen from six sides, its colour binned by normal into a small map in the same layout (diffuse and the little
      // specular a matte surface has, at envMapIntensity 1). A renderer that lights a surface by this map at its normal,
      // unoccluded, gets the creator's ambient exactly.
      const vis = [];
      scene.traverse(o => { if (o.visible && (o.isMesh || o.isLight)) { vis.push(o); o.visible = false; } });
      const sph = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 48), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, metalness: 0, envMapIntensity: 1 }));
      scene.add(sph);
      // (twice: white, and black, which leaves only the specular part. three.js's shading is linear in the diffuse
      // colour, so a surface of colour c gets c x (white - black) + black: dark hair is mostly the black part)
      for (const [key, col] of [['environmentResponse', 0xffffff], ['environmentSpecular', 0x000000]]) {
      sph.material.color.set(col);
      const RS = 128, rt = new THREE.WebGLRenderTarget(RS, RS, { type: THREE.FloatType }), cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10), buf = new Float32Array(RS * RS * 4);
      const MW = 64, MH = 32, acc = new Float64Array(MW * MH * 4);
      r0.toneMapping = THREE.NoToneMapping; r0.outputColorSpace = THREE.LinearSRGBColorSpace;
      [[0, 0, 3], [0, 0, -3], [3, 0, 0], [-3, 0, 0], [0, 3, 0.001], [0, -3, 0.001]].forEach(cp => {
        cam.position.set(cp[0], cp[1], cp[2]); cam.lookAt(0, 0, 0); cam.updateMatrixWorld();
        r0.setRenderTarget(rt); r0.setClearColor(0x000000, 0); r0.clear(); r0.render(scene, cam); r0.readRenderTargetPixels(rt, 0, 0, RS, RS, buf); r0.setRenderTarget(null);
        for (let j = 0; j < RS; j++) for (let i = 0; i < RS; i++) {
          const x = (i + 0.5) / (RS / 2) - 1, y = (j + 0.5) / (RS / 2) - 1, rr = x * x + y * y; if (rr > 0.85) continue;
          const nv = new THREE.Vector3(x, y, Math.sqrt(1 - rr)).transformDirection(cam.matrixWorld), k = (j * RS + i) * 4;
          const bx = nv.x, by = -nv.z, bz = nv.y;      // (Blender's direction)
          const u = Math.atan2(by, -bx) / (2 * Math.PI) + 0.5, vv = Math.atan2(bz, Math.hypot(bx, by)) / Math.PI + 0.5;
          const q = Math.min(MH - 1, Math.floor(vv * MH)) * MW + Math.min(MW - 1, Math.floor(u * MW));
          acc[q * 4] += buf[k]; acc[q * 4 + 1] += buf[k + 1]; acc[q * 4 + 2] += buf[k + 2]; acc[q * 4 + 3] += 1;
        }
      });
      const resp = new Float32Array(MW * MH * 3); let empty = 0;
      for (let q = 0; q < MW * MH; q++) { const c = acc[q * 4 + 3]; if (!c) empty++; for (let ch = 0; ch < 3; ch++) resp[q * 3 + ch] = c ? acc[q * 4 + ch] / c : 0; }
      viewer[key] = { layout: 'equirect-blender', width: MW, height: MH, data: b64(resp), emptyBins: empty,
        note: 'outgoing radiance of a matte ' + (col ? 'white' : 'black') + ' surface (roughness 1, metalness 0) at envMapIntensity 1, by its normal, seen from the six axes; linear' };
      acc.fill(0);
      }
      scene.remove(sph); vis.forEach(o => { o.visible = true; });
      r0.toneMapping = tm; r0.outputColorSpace = cs;
    }
    // ---- reference renders on fixed cameras
    const shots = [], cams = {};
    if (opt.views.length) {
      const VW = { front: [0, 4], three: [35, 6], profile: [88, 3], back3: [145, 6], back: [180, 4] }, D = 50, TY = 1.5;      // (the whole head and the hair to the shoulders)
      const W = opt.size, H = opt.size, old = { w: r0.domElement.width, h: r0.domElement.height, pr: r0.getPixelRatio() };
      r0.setPixelRatio(1); r0.setSize(W, H, false);
      const cam = new THREE.PerspectiveCamera(20, 1, 1, 200);
      const clay = new THREE.MeshStandardMaterial({ color: 0x9a9a9a, roughness: 1, metalness: 0, envMapIntensity: 0.25 });
      const hairSet = new Set(meshes);
      const keep = new Map();
      for (const look of opt.looks) {
        if (look === 'clay') scene.traverse(o => { if (o.isMesh && !hairSet.has(o) && o.visible) { keep.set(o, o.material); o.material = clay; } });
        // (clay hair: the hair's own cut-out and shade, its colour map and vertex tones grey)
        const hairKeep = new Map();
        if (look === 'clay') meshes.forEach(m => { const mm = m.material; if (!mm.map) return; const c = mm.clone(); c.map = null; c.bumpMap = null; c.color.setRGB(0.6, 0.6, 0.6); c.onBeforeCompile = mm.onBeforeCompile; hairKeep.set(m, mm); m.material = c; });
        for (const vname of opt.views) {
          const [az, el] = VW[vname], a = az * Math.PI / 180, e = el * Math.PI / 180;
          cam.position.set(D * Math.sin(a) * Math.cos(e), TY + D * Math.sin(e), D * Math.cos(a) * Math.cos(e)); cam.lookAt(0, TY, 0); cam.updateMatrixWorld();
          cams[vname] = { position: cam.position.toArray(), target: [0, TY, 0], fovY: 20, aspect: 1 };
          r0.setClearColor(0x000000, 0); r0.render(scene, cam);
          shots.push({ name: 'creator-' + vname + '-' + look + '.png', png: r0.domElement.toDataURL('image/png').split(',')[1] });
        }
        hairKeep.forEach((mm, m) => { m.material = mm; });
        keep.forEach((mm, o) => { o.material = mm; }); keep.clear();
      }
      r0.setPixelRatio(old.pr); r0.setSize(old.w / old.pr, old.h / old.pr, false);
    }
    // the head the hair was built on, for a bench to render the hair against (positions and normals only)
    let head = null;
    if (opt.head) {
      const hp = [], hn = [], hi = []; let nv = 0; const hs = new Set(meshes), v = new THREE.Vector3();
      scene.traverse(o => {
        if (!o.isMesh || hs.has(o) || !o.visible || !o.geometry.attributes.position) return;
        const g = o.geometry, P = g.attributes.position, N = g.attributes.normal, nm = new THREE.Matrix3().getNormalMatrix(o.matrixWorld);
        for (let i = 0; i < P.count; i++) { v.fromBufferAttribute(P, i).applyMatrix4(o.matrixWorld); hp.push(v.x, v.y, v.z); if (N) { v.fromBufferAttribute(N, i).applyMatrix3(nm).normalize(); hn.push(v.x, v.y, v.z); } else hn.push(0, 1, 0); }
        const I = g.index ? g.index.array : null; if (I) for (let k = 0; k < I.length; k++) hi.push(nv + I[k]); else for (let k = 0; k < P.count; k++) hi.push(nv + k);
        nv += P.count;
      });
      head = { pos: b64(new Float32Array(hp)), nrm: b64(new Float32Array(hn)), idx: b64(new Uint32Array(hi)) };
    }
    return { head, style: opt.style, name: ST.name, hairTone: fc.P.hairTone, face: opt.face, softHairline: !!ST.softHairline, mats, pieces, prims, rings: Object.fromEntries(Object.keys(R).map(k => [k, b64(new Float32Array(R[k]))])), nRings: R.centre.length / 3, stats, tex, guides, viewer,
      depthShader: { vertex: dsh.vertexShader, fragment: dsh.fragmentShader }, shots, cams, revision: THREE.REVISION };
  }, { style, face, views, size, looks, probe, head: !!headOut });
  await b.close();
  if (r.error) { console.log(r.error); process.exit(1); }

  // ---- the cut-out shader: check its shape and take its numbers
  r.mats.forEach(m => {
    if (m.kind !== 'cut') return;
    const f = m.shader.fragment, nums = (f.match(NUM) || []).map(Number), skeleton = f.replace(NUM, '#');
    const known = fs.readFileSync(path.join(__dirname, 'cut-shader-1.glsl'), 'utf8').trim();
    if (skeleton.trim() !== known) {
      fs.writeFileSync(out + '.cut-shader.glsl', skeleton);
      console.log('the creator\'s cut-out shader has changed shape (written to ' + out + '.cut-shader.glsl): update FORMAT.md, the readers and cut-shader-1.glsl');
      process.exit(1);
    }
    if (nums.length !== CUT_PARAMS.length) { console.log('cut shader: expected ' + CUT_PARAMS.length + ' numbers, found ' + nums.length); process.exit(1); }
    m.model = CUT_MODEL; m.params = {}; CUT_PARAMS.forEach((k, i) => { m.params[k] = nums[i]; });
    m.params.hairline = m.shader.uniforms.hairline !== undefined ? m.shader.uniforms.hairline : (r.softHairline ? 1 : 0);
  });

  // ---- write the glb
  const W = new GlbWriter(), J = W.json, B = s => Buffer.from(s, 'base64'), F32 = s => { const u = B(s); return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4).slice(); };
  const imgs = {};
  J.samplers = [];
  J.textures = [];
  Object.keys(r.tex).forEach(k => {
    const t = r.tex[k];
    J.samplers.push({ magFilter: 9729, minFilter: 9987, wrapS: t.wrapS === 'repeat' ? 10497 : 33071, wrapT: t.wrapT === 'repeat' ? 10497 : 33071 });
    J.textures.push({ source: W.image(B(t.png), k), sampler: J.samplers.length - 1, name: k, extras: { colorSpace: t.colorSpace } });
    imgs[k] = J.textures.length - 1;
  });
  J.materials = r.mats.map(m => {
    const g = { name: m.name + ' (' + m.kind + ')', doubleSided: m.side === 'double', extras: { emberwood: JSON.parse(JSON.stringify(m)) } };
    if (m.shader.vertex.trim() === '#include <begin_vertex>') delete g.extras.emberwood.shader.vertex;      // (kept only when it says something)
    const pbr = { baseColorFactor: [m.color[0], m.color[1], m.color[2], m.opacity], metallicFactor: m.metalness || 0, roughnessFactor: m.roughness === undefined ? 1 : m.roughness };
    if (m.maps.map === 'hair') pbr.baseColorTexture = { index: imgs.hair };
    g.pbrMetallicRoughness = pbr;
    if (m.kind === 'cut') { g.alphaMode = 'MASK'; g.alphaCutoff = m.alphaTest; }
    if (m.kind === 'veil') { g.alphaMode = 'BLEND'; g.extensions = { KHR_materials_unlit: {} }; J.extensionsUsed = ['KHR_materials_unlit']; }
    return g;
  });
  J.meshes = []; J.nodes = []; const kids = [];
  r.prims.forEach(pr => {
    const m = r.mats[pr.material];
    const attrs = {
      POSITION: W.accessor(F32(pr.pos), 3, { minmax: true, target: 34962 }), NORMAL: W.accessor(F32(pr.nrm), 3, { target: 34962 }),
      TEXCOORD_0: W.accessor(F32(pr.uv), 2, { target: 34962 }), COLOR_0: W.accessor(F32(pr.col), 4, { target: 34962 }),
      _ALONG: W.accessor(F32(pr.along), 1, { target: 34962 }), _PIECE: W.accessor(F32(pr.piece), 1, { target: 34962 }),      // (float: a vertex attribute's elements must be 4-byte aligned)
    };
    const ib = B(pr.idx), idx = new Uint32Array(ib.buffer, ib.byteOffset, ib.byteLength / 4).slice();
    J.meshes.push({ name: m.kind, primitives: [{ attributes: attrs, indices: W.accessor(idx, 1, { target: 34963 }), material: pr.material, mode: 4 }] });
    J.nodes.push({ name: 'hair ' + m.kind, mesh: J.meshes.length - 1 }); kids.push(J.nodes.length - 1);
  });
  // the tubes' rings, one accessor per quantity (FORMAT.md, "Strands")
  const rings = r.nRings ? { count: r.nRings, centre: W.accessor(F32(r.rings.centre), 3), axisA: W.accessor(F32(r.rings.axisA), 3), axisB: W.accessor(F32(r.rings.axisB), 3),
    color: W.accessor(F32(r.rings.color), 4), tex: W.accessor(F32(r.rings.tex), 3) } : null;
  const gl = []; r.guides.forEach(g => { g.first = gl.length / 3; g.count = g.path.length; g.path.forEach(q => gl.push(...q)); delete g.path; });
  J.nodes.push({ name: 'hair', children: kids });
  J.scenes = [{ name: r.name, nodes: [J.nodes.length - 1] }]; J.scene = 0;
  const src = fs.readFileSync(page);
  ['environmentMap', 'environmentResponse', 'environmentSpecular'].forEach(k => { const em = r.viewer[k]; if (em) { em.accessor = W.accessor(F32(em.data), 3); delete em.data; } });
  J.extras = { emberwood_hair: {
    schema: 'emberwood.hair/1',
    style: r.style, name: r.name, face: r.face || null, hairTone: r.hairTone,
    source: { creator: path.basename(page), sha256: crypto.createHash('sha256').update(src).digest('hex').slice(0, 16), three: 'r' + r.revision, exported: new Date().toISOString() },
    units: { headHeight: 8.6, note: 'model units: the creator head is 8.6 from chin to crown' },
    frame: { up: '+Y', front: '+Z', origin: 'chin', blender: '(x, y, z)_blender = (X, -Z, Y) * HEAD_H / 8.6' },
    textures: { hair: imgs.hair, bump: imgs.bump, tips: imgs.tips },
    pieces: r.pieces, rings,
    guides: r.guides, guideLines: gl.length ? W.accessor(new Float32Array(gl), 3) : null,
    depthShader: r.depthShader, viewer: r.viewer, cameras: r.cams,
  } };
  fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
  fs.writeFileSync(out, W.write());
  r.shots.forEach(s => fs.writeFileSync(path.join(path.dirname(out), s.name), B(s.png)));
  if (headOut && r.head) {
    const H = new GlbWriter(), HJ = H.json, ib = B(r.head.idx);
    HJ.meshes = [{ name: 'head', primitives: [{ attributes: { POSITION: H.accessor(F32(r.head.pos), 3, { minmax: true, target: 34962 }), NORMAL: H.accessor(F32(r.head.nrm), 3, { target: 34962 }) },
      indices: H.accessor(new Uint32Array(ib.buffer, ib.byteOffset, ib.byteLength / 4).slice(), 1, { target: 34963 }), mode: 4 }] }];
    HJ.nodes = [{ name: 'creator head', mesh: 0 }]; HJ.scenes = [{ nodes: [0] }]; HJ.scene = 0;
    HJ.extras = { emberwood_reference: { what: 'the face creator head (and eyes, brows, ears) the hair was exported on', face: r.face || null, units: 'model, head 8.6' } };
    fs.writeFileSync(headOut, H.write());
  }
  const nv = r.prims.reduce((s, q) => s + q.nv, 0);
  console.log(`${r.name}: ${r.pieces.length} pieces (${r.mats.map(m => m.kind).join(', ')}), ${r.stats.tubes} as tubes (${r.nRings} rings, worst rebuild error ${r.stats.worst.toExponential(1)}), ${r.stats.raw} as meshes (${nv} vertices), ${r.guides.length} lock guides, ${(fs.statSync(out).size / 1e6).toFixed(1)} MB` + (r.shots.length ? `, ${r.shots.length} reference views` : ''));
})();
