import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import './styles.css';
import { COLS, ROWS, WALK } from '../shared/sprites.js';
import { loadSheet } from '../shared/sheet-loader.js';
import { ROSTER, loadParty, memberSpec } from '../shared/party.js';
import { V_LIT, V_END, F_LIT, V_UNLIT, F_UNLIT, V_LOCAL } from './shaders.js';

function main() {
  const statusEl = document.getElementById('status');
  const say = (t) => { if (statusEl.textContent !== t) statusEl.textContent = t; };
  if (window.innerWidth > 700) document.getElementById('help').open = true;

  const canvas = document.getElementById('view');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const clamp = THREE.MathUtils.clamp, lerp = THREE.MathUtils.lerp;
  const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

  // ---------- Renderer ----------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xcfe2e6, 40, 150);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.3, 600);

  // uniforms every custom shader shares; updated once per frame or when the theme changes
  const SHARED = {
    uTime: { value: 0 }, uWater: { value: 0 }, uViewH: { value: 800 }, uExposure: { value: 1 },
    uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSunCol: { value: new THREE.Color(1, 1, 1) },
    uSkyTop: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() }, uNight: { value: 0 },
    // the light a sprite gets in the open in the current mood (see applyMood): sprites are lit relative to it
    uSpriteRef: { value: new THREE.Vector3(1, 1, 1) }
  };

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    SHARED.uViewH.value = h * renderer.getPixelRatio();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  // ---------- Terrain shape ----------
  const SIZE = 280, SEG = 300, LIMIT = 88, WATER = -1.4;
  const CAMP = { x: 0, z: 0 }, CIRCLE = { x: 38, z: -26 }, LOOKOUT = { x: 63, z: -51 };
  // Two cliff-sided mesas stacked as tiers, each reached by a single ramp.
  const MESAS = [
    { x: CIRCLE.x, z: CIRCLE.z, r: 14, top: 11, wob: 1 },
    { x: LOOKOUT.x, z: LOOKOUT.z, r: 8, top: 20, wob: 0.45 }
  ];
  const CLEAR = [
    { x: CAMP.x, z: CAMP.z, r: 7 },
    { x: CIRCLE.x, z: CIRCLE.z, r: 7.5 },
    { x: LOOKOUT.x, z: LOOKOUT.z, r: 6 }
  ];
  function toward(from, to, dist) {
    const dx = to.x - from.x, dz = to.z - from.z, l = Math.hypot(dx, dz);
    return { x: from.x + dx / l * dist, z: from.z + dz / l * dist };
  }
  const R1a = toward(CIRCLE, CAMP, 38), R1b = toward(CIRCLE, CAMP, 10.5);      // valley floor -> first tier
  const R2a = toward(CIRCLE, LOOKOUT, 9), R2b = toward(LOOKOUT, CIRCLE, 4.5);  // first tier -> second tier
  const PATH = [[2, -2], [R1a.x, R1a.z], [R1b.x, R1b.z], [CIRCLE.x, CIRCLE.z], [R2a.x, R2a.z], [R2b.x, R2b.z]];

  function rawHeight(x, z) {
    return 3.2 * Math.sin(x * 0.045 + 1.3) * Math.cos(z * 0.05 - 0.4)
         + 1.6 * Math.sin(x * 0.11 + z * 0.07)
         + 0.6 * Math.sin(x * 0.23 - z * 0.19 + 2)
         + 0.25 * Math.sin(x * 0.5 + 1) * Math.sin(z * 0.45);
  }
  function baseHeight(x, z) {
    // rolling ground plus one broad, walkable hill
    let y = rawHeight(x, z) + 13 * Math.exp(-((x + 50) * (x + 50) + (z + 45) * (z + 45)) / (2 * 26 * 26));
    const d = Math.hypot(x - CAMP.x, z - CAMP.z);
    if (d < 16) y += (3.0 - y) * (1 - smoothstep(7, 16, d));
    return y;
  }
  function mesaApply(y, m, x, z) {
    const dx = x - m.x, dz = z - m.z, d = Math.hypot(dx, dz);
    if (d > m.r + 6) return y;
    const a = Math.atan2(dz, dx);
    const edge = m.r + m.wob * (1.5 * Math.sin(3 * a + 1) + 0.8 * Math.sin(7 * a + 2));
    const k = 1 - smoothstep(edge, edge + 2.6, d);       // 2.6 units from top to bottom: a cliff
    return y + (m.top + 0.25 * Math.sin(x * 0.4) * Math.cos(z * 0.37) - y) * k;
  }
  function rampApply(y, a, b, ha, hb, x, z) {
    const vx = b.x - a.x, vz = b.z - a.z;
    const t = clamp(((x - a.x) * vx + (z - a.z) * vz) / (vx * vx + vz * vz), 0, 1);
    const d = Math.hypot(x - a.x - vx * t, z - a.z - vz * t);
    if (d > 3.8) return y;
    return y + (lerp(ha, hb, t) - y) * (1 - smoothstep(2.2, 3.8, d));
  }
  const R1h = baseHeight(R1a.x, R1a.z);
  function height(x, z) {
    let y = baseHeight(x, z);
    y = mesaApply(y, MESAS[0], x, z);
    y = rampApply(y, R1a, R1b, R1h, MESAS[0].top, x, z);
    y = mesaApply(y, MESAS[1], x, z);
    y = rampApply(y, R2a, R2b, MESAS[0].top, MESAS[1].top, x, z);
    return y;
  }
  function tooSteep(x, z) {
    const gx = (height(x + 0.4, z) - height(x - 0.4, z)) / 0.8, gz = (height(x, z + 0.4) - height(x, z - 0.4)) / 0.8;
    return gx * gx + gz * gz > 1.2;
  }
  function pathDist(x, z) {
    let best = Infinity;
    for (let k = 0; k < PATH.length - 1; k++) {
      const ax = PATH[k][0], az = PATH[k][1], bx = PATH[k + 1][0], bz = PATH[k + 1][1];
      const vx = bx - ax, vz = bz - az;
      const t = clamp(((x - ax) * vx + (z - az) * vz) / (vx * vx + vz * vz), 0, 1);
      best = Math.min(best, Math.hypot(x - ax - vx * t, z - az - vz * t));
    }
    return best;
  }

  let seed = 7719;
  function rand() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }


  // ---------- Shader library ----------
  // Every surface is a ShaderMaterial. Textures are generated in the fragment shader from noise,
  // so there are no image files. Lit shaders read three.js's own light and shadow uniforms.
  SHARED.uWater.value = WATER;


  function makeMat(lit, vert, frag, own, opts) {
    const uniforms = THREE.UniformsUtils.merge(lit ? [THREE.UniformsLib.lights, THREE.UniformsLib.fog] : [THREE.UniformsLib.fog]);
    Object.assign(uniforms, SHARED, own || {});
    return new THREE.ShaderMaterial(Object.assign({
      uniforms, lights: lit, fog: true,
      vertexShader: (lit ? V_LIT : V_UNLIT) + vert,
      fragmentShader: (lit ? F_LIT : F_UNLIT) + frag
    }, opts || {}));
  }
  const colour = (hex) => ({ value: new THREE.Color(hex) });

  // Ground: grass with tufts, dry patches and flowers; dirt paths with pebbles; sand at the shore;
  // banded rock wherever the slope is steep, so every cliff face gets strata.
  const terrainMat = makeMat(true, `
attribute float aPath;
varying float vPath;
void main() {
  vPath = aPath;
  vec4 worldPosition = modelMatrix * vec4(position, 1.0);
  vec3 worldNormal = normalize(mat3(modelMatrix) * normal);
` + V_END + `}`, `
uniform float uWater;
varying float vPath;
vec3 bedColour(float id) {
  vec3 c = mix(lin(vec3(0.4, 0.38, 0.35)), lin(vec3(0.61, 0.58, 0.5)), n_hash12(vec2(id, 3.1)));
  c = mix(c, lin(vec3(0.56, 0.47, 0.38)), step(0.7, n_hash12(vec2(id, 9.7))) * 0.6);
  c = mix(c, lin(vec3(0.38, 0.38, 0.38)), step(0.85, n_hash12(vec2(id, 5.3))) * 0.4);
  return c;
}
// joints, mottling and run-off stains for one flat projection of a cliff face
float rockFace(vec2 q, vec3 p) {
  vec2 joint = n_cells(vec2(q.x * 0.55, q.y * 0.33) + n_noise(p.xz * 0.5) * 0.8);
  float m = 1.0 - (1.0 - smoothstep(0.0, 0.03, joint.y)) * 0.16;
  m *= 0.72 + 0.5 * n_fbm(vec2(q.x * 1.1, q.y * 1.4));
  m *= 1.0 - smoothstep(0.55, 0.85, n_noise(vec2(q.x * 2.2, q.y * 0.18))) * 0.16;
  return m;
}
void main() {
  vec3 p = vWorldPos;
  vec3 N = normalize(vNormalW);
  float slope = 1.0 - N.y;
  float large = n_fbm(p.xz * 0.045);
  float edge = n_fbm(p.xz * 0.33) - 0.5;

  vec3 grass = mix(lin(vec3(0.27, 0.4, 0.17)), lin(vec3(0.44, 0.54, 0.25)), smoothstep(0.3, 0.72, large));
  grass = mix(grass, lin(vec3(0.6, 0.56, 0.34)), smoothstep(0.55, 0.78, n_fbm(p.xz * 0.021 + 7.0)) * 0.55);
  grass *= 0.72 + 0.5 * n_fbm(p.xz * 0.4 + 2.0);
  float tuft = n_noise(p.xz * 3.1) * 0.6 + n_noise(p.xz * 11.0) * 0.4;
  grass *= 0.74 + 0.42 * tuft;
  vec2 fc = floor(p.xz * 3.0);
  float fh = n_hash12(fc);
  vec2 fo = fract(p.xz * 3.0) - 0.5 + (n_hash22(fc) - 0.5) * 0.5;
  float flower = step(0.975, fh) * (1.0 - smoothstep(0.06, 0.1, length(fo))) * smoothstep(0.35, 0.65, large);
  grass = mix(grass, mix(lin(vec3(0.96, 0.9, 0.55)), lin(vec3(0.86, 0.56, 0.78)), step(0.99, fh)), flower);

  // Trail: packed earth with damp patches, dried-mud cracks, half-buried stones, gravel and leaf litter.
  vec3 dirt = mix(lin(vec3(0.3, 0.22, 0.15)), lin(vec3(0.47, 0.36, 0.24)), n_fbm(p.xz * 0.9));
  float damp = smoothstep(0.48, 0.72, n_fbm(p.xz * 0.35 + 11.0));
  dirt = mix(dirt, lin(vec3(0.21, 0.16, 0.115)), damp * 0.75);
  dirt *= (0.78 + 0.44 * n_noise(p.xz * 14.0)) * (0.9 + 0.2 * n_noise(p.xz * 41.0));
  vec2 mud = n_cells(p.xz * 1.7 + n_noise(p.xz * 0.8) * 0.7);
  dirt *= 1.0 - (1.0 - smoothstep(0.0, 0.04, mud.y)) * (1.0 - damp) * smoothstep(0.35, 0.6, n_fbm(p.xz * 0.22 + 4.0)) * 0.5;
  vec3 st = n_cells3(p.xz * 1.7);
  float stR = 0.06 + 0.08 * fract(st.z * 3.7);
  float stone = (1.0 - smoothstep(stR * 0.75, stR, st.x)) * step(0.8, st.z);
  dirt *= 1.0 - (1.0 - smoothstep(stR, stR * 1.5, st.x)) * step(0.8, st.z) * (1.0 - stone) * 0.35;
  vec3 stoneCol = mix(lin(vec3(0.3, 0.29, 0.27)), lin(vec3(0.5, 0.47, 0.41)), fract(st.z * 7.3));
  stoneCol *= (0.75 + 0.35 * n_noise3(p * 9.0)) * (1.15 - 0.5 * st.x / stR);
  dirt = mix(dirt, stoneCol, stone);
  vec3 gr = n_cells3(p.xz * 5.5);
  float gravel = (1.0 - smoothstep(0.1, 0.2, gr.x)) * step(0.45, gr.z) * smoothstep(0.4, 0.65, n_noise(p.xz * 0.9 + 9.0));
  dirt = mix(dirt, mix(lin(vec3(0.3, 0.28, 0.25)), lin(vec3(0.56, 0.52, 0.45)), fract(gr.z * 5.1)), gravel * 0.8);
  vec3 lf = n_cells3(p.xz * 11.0);
  float litter = (1.0 - smoothstep(0.1, 0.22, lf.x)) * step(0.8, lf.z) * smoothstep(0.45, 0.7, n_noise(p.xz * 0.6 + 21.0));
  dirt = mix(dirt, mix(lin(vec3(0.42, 0.26, 0.1)), lin(vec3(0.28, 0.19, 0.08)), fract(lf.z * 9.0)), litter * 0.75);

  vec3 sand = mix(lin(vec3(0.72, 0.65, 0.47)), lin(vec3(0.82, 0.76, 0.58)), n_noise(p.xz * 1.4));

  // Rock: beds of uneven thickness that dip, bend and step across faults, cut by joints and stained by run-off.
  float layer = p.y * 0.8 + dot(p.xz, vec2(0.04, -0.025))
              + n_fbm(p.xz * 0.11) * 3.4 + n_fbm(vec2(p.x + p.z, p.y * 1.7) * 0.45) * 1.5
              + floor(n_noise(p.xz * 0.06 + 3.0) * 4.0) * 0.43;
  layer += n_noise(vec2(layer * 0.63, 3.0)) * 1.6 + n_noise(vec2(layer * 1.9, 8.0)) * 0.5;
  float bandId = floor(layer), band = fract(layer);
  float rough = n_noise3(p * 1.6);
  vec3 rock = mix(bedColour(bandId), bedColour(bandId + 1.0), smoothstep(0.0, 1.0, band + (rough - 0.5) * 0.3));
  float lip = (1.0 - smoothstep(0.0, 0.25, band)) * smoothstep(0.75, 1.0, n_hash12(vec2(bandId, 2.2)) + rough * 0.3);
  rock *= 1.0 - lip * 0.12;
  rock *= 0.94 + 0.12 * n_noise(vec2(layer * 8.0, (p.x + p.z) * 0.4));
  // blend the face detail from two side-on projections so it never stretches
  float wx = N.x * N.x / max(N.x * N.x + N.z * N.z, 0.0001);
  rock *= mix(rockFace(p.xy, p), rockFace(p.zy + 17.0, p), wx);
  rock *= 0.85 + 0.3 * n_noise3(p * 8.0);
  rock = mix(rock, lin(vec3(0.3, 0.38, 0.18)), smoothstep(0.62, 0.8, n_fbm3(p * 0.45 + 30.0)) * 0.35);

  float wPath = smoothstep(0.3, 0.62, vPath + edge * 0.45 + (n_noise(p.xz * 2.3) - 0.5) * 0.22);
  // grass creeps in from the verge as tufts
  float verge = 1.0 - smoothstep(0.6, 0.98, vPath);
  wPath *= 1.0 - smoothstep(0.55, 0.7, n_noise(p.xz * 2.7 + 5.0) * 0.6 + n_noise(p.xz * 7.0) * 0.4) * verge * 0.9;
  float wSand = 1.0 - smoothstep(uWater + 0.2, uWater + 1.1, p.y + edge * 1.2);
  float wRock = smoothstep(0.24, 0.42, slope + edge * 0.18);

  vec3 col = mix(grass, dirt, wPath);
  col = mix(col, sand, wSand);
  col = mix(col, rock, wRock);
  col *= mix(0.6, 1.0, smoothstep(uWater - 0.1, uWater + 0.4, p.y));

  // fine relief: tilt the normal by the gradient of a small height field
  float e = 0.05;
  float amp = mix(0.25, 0.75, max(wRock, wPath * 0.5));
  float b0 = n_noise3(p * 2.4) + n_noise3(p * 6.1) * 0.4;
  vec3 g = vec3(n_noise3((p + vec3(e, 0.0, 0.0)) * 2.4) + n_noise3((p + vec3(e, 0.0, 0.0)) * 6.1) * 0.4,
                n_noise3((p + vec3(0.0, e, 0.0)) * 2.4) + n_noise3((p + vec3(0.0, e, 0.0)) * 6.1) * 0.4,
                n_noise3((p + vec3(0.0, 0.0, e)) * 2.4) + n_noise3((p + vec3(0.0, 0.0, e)) * 6.1) * 0.4);
  g = (g - b0) * amp / e;
  g -= N * dot(g, N);
  N = normalize(N - g * 0.06);

  float sh;
  col *= gather(N, 0.0, sh);
  gl_FragColor = outCol(applyFog(col), 1.0);
}`);

  // Water: moving wave normals, sky reflection by Fresnel, sun glint, depth tint and shore foam.
  // A small height texture of the terrain tells each pixel how deep the water is.
  const HTEX = 256, HMIN = -6, HRANGE = 12;
  const heightTex = (function () {
    const data = new Uint8Array(HTEX * HTEX * 4);
    for (let j = 0; j < HTEX; j++) for (let i = 0; i < HTEX; i++) {
      const x = (i + 0.5) / HTEX * SIZE - SIZE / 2, z = (j + 0.5) / HTEX * SIZE - SIZE / 2;
      const v = Math.round(clamp((height(x, z) - HMIN) / HRANGE, 0, 1) * 255);
      const k = (j * HTEX + i) * 4;
      data[k] = v; data[k + 1] = v; data[k + 2] = v; data[k + 3] = 255;
    }
    const t = new THREE.DataTexture(data, HTEX, HTEX, THREE.RGBAFormat);
    t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearFilter;
    t.needsUpdate = true;
    return t;
  })();

  const waterMat = makeMat(true, `
void main() {
  vec4 worldPosition = modelMatrix * vec4(position, 1.0);
  vec3 worldNormal = vec3(0.0, 1.0, 0.0);
` + V_END + `}`, `
uniform float uWater;
uniform sampler2D uHeight;
uniform float uHMin;
uniform float uHRange;
uniform float uSize;
float waveH(vec2 q) {
  float t = uTime;
  return sin(dot(q, vec2(0.8, 0.6)) * 1.3 + t * 1.1) * 0.05
       + sin(dot(q, vec2(-0.5, 0.9)) * 2.1 + t * 1.7) * 0.03
       + n_noise(q * 1.6 + vec2(t * 0.35, t * 0.2)) * 0.06
       + n_noise(q * 3.7 - vec2(t * 0.3, -t * 0.25)) * 0.03;
}
void main() {
  vec3 p = vWorldPos;
  float ground = texture2D(uHeight, p.xz / uSize + 0.5).r * uHRange + uHMin;
  float depth = uWater - ground;
  float e = 0.04;
  float h0 = waveH(p.xz);
  vec3 N = normalize(vec3(-(waveH(p.xz + vec2(e, 0.0)) - h0) / e, 1.0, -(waveH(p.xz + vec2(0.0, e)) - h0) / e));
  vec3 V = normalize(cameraPosition - p);
  float sh;
  vec3 irr = gather(vec3(0.0, 1.0, 0.0), 0.0, sh);
  vec3 body = mix(lin(vec3(0.24, 0.52, 0.5)), lin(vec3(0.04, 0.2, 0.27)), smoothstep(0.0, 3.0, depth)) * irr;
  float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
  vec3 R = reflect(-V, N);
  vec3 col = mix(body, skyColour(R), clamp(fres, 0.0, 1.0) * 0.9);
  col += uSunCol * pow(max(dot(R, uSunDir), 0.0), 180.0) * 3.0 * sh;
  float wob = n_noise(p.xz * 2.2 + uTime * 0.4) - 0.5;
  float foam = 1.0 - smoothstep(0.0, 0.1, depth + wob * 0.06);
  float ripple = smoothstep(0.8, 1.0, sin(depth * 18.0 - uTime * 2.2 + wob * 3.0) * 0.5 + 0.5) * (1.0 - smoothstep(0.05, 0.35, depth));
  foam = clamp(foam * 0.85 + ripple * 0.4, 0.0, 1.0);
  col = mix(col, lin(vec3(0.92, 0.95, 0.93)) * (irr * 0.9 + 0.05), foam);
  float alpha = max(max(mix(0.5, 0.92, smoothstep(0.0, 2.0, depth)), foam), fres);
  gl_FragColor = outCol(applyFog(col), alpha);
}`, {
    uHeight: { value: heightTex }, uHMin: { value: HMIN }, uHRange: { value: HRANGE }, uSize: { value: SIZE }
  }, { transparent: true, depthWrite: false });

  // Foliage: crowns sway in the wind, with leaf clumps, depth darkening and light shining through.
  const foliageMat = makeMat(true, `
attribute float aShade;
varying vec3 vColor;
varying vec3 vBase;
varying float vShade;
void main() {
  mat4 m = modelOf();
  vec4 worldPosition = m * vec4(position, 1.0);
  vec3 base = (m * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  worldPosition.xz += windSway(base, max(worldPosition.y - base.y, 0.0));
  vec3 worldNormal = normalize(mat3(m) * normal);
#ifdef USE_INSTANCING_COLOR
  vColor = instanceColor;
#else
  vColor = vec3(0.25, 0.5, 0.3);
#endif
  vBase = base;
  vShade = aShade;
` + V_END + `}`, `
varying vec3 vColor;
varying vec3 vBase;
varying float vShade;
void main() {
  vec3 p = vWorldPos;
  vec3 N = normalize(vNormalW);
  vec3 radial = p - vec3(vBase.x, p.y, vBase.z);
  float rl = length(radial);
  if (rl > 0.001) N = normalize(mix(N, normalize(radial / rl + vec3(0.0, 0.45, 0.0)), 0.2));
  float clump = smoothstep(0.25, 0.8, n_noise3(p * 2.3) * 0.65 + n_noise3(p * 6.5) * 0.35);
  float sprig = n_noise3(p * 15.0);
  vec3 base = lin(vColor);
  vec3 col = base * (0.78 + 0.3 * clump);
  col *= mix(0.4, 1.08, vShade);
  col = mix(col, col * vec3(1.18, 1.12, 0.78), n_noise3(p * 0.6) * 0.45);
  col *= 0.9 + 0.2 * sprig;
  float sh;
  vec3 irr = gather(N, 0.45, sh);
  vec3 V = normalize(cameraPosition - p);
  vec3 through = base * uSunCol * pow(max(dot(-V, uSunDir), 0.0), 3.0) * 0.4 * sh;
  col = col * irr + through;
  col += base * irr * pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.3;
  gl_FragColor = outCol(applyFog(col), 1.0);
}`);

  // Wood: furrowed bark along the length, growth rings on cut ends.
  const woodMat = (hex) => makeMat(true, V_LOCAL, `
uniform vec3 uColor;
varying vec3 vLocal;
varying vec3 vLocalN;
void main() {
  vec3 N = normalize(vNormalW);
  vec3 base = lin(uColor);
  float ang = atan(vLocal.z, vLocal.x);
  float r = length(vLocal.xz);
  vec3 col;
  if (abs(vLocalN.y) > 0.8) {
    float rings = sin(r * 75.0 + n_noise(vLocal.xz * 9.0) * 3.0) * 0.5 + 0.5;
    col = mix(lin(vec3(0.6, 0.45, 0.28)), lin(vec3(0.8, 0.64, 0.43)), rings);
    col *= 1.0 - smoothstep(0.75, 1.0, r / 0.2) * 0.45;
  } else {
    float furrow = n_noise(vec2(ang * 3.0, vLocal.y * 1.2));
    float ridges = abs(sin(ang * 7.0 + furrow * 5.0 + vLocal.y * 0.5));
    float plates = n_noise(vec2(ang * 6.0, vLocal.y * 9.0));
    col = base * (0.5 + 0.5 * smoothstep(0.1, 0.9, ridges)) * (0.82 + 0.36 * plates);
  }
  float sh;
  col *= gather(N, 0.15, sh);
  gl_FragColor = outCol(applyFog(col), 1.0);
}`, { uColor: colour(hex) });

  // Bark for trees: furrows run along each trunk and branch, wrapped seamlessly around it,
  // with plates, moss and the same wind sway as the leaves.
  const barkMat = (hex, mossAmt) => makeMat(true, `
varying vec2 vBark;
void main() {
  mat4 m = modelOf();
  vec4 worldPosition = m * vec4(position, 1.0);
  vec3 base = (m * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  worldPosition.xz += windSway(base, max(worldPosition.y - base.y, 0.0));
  vec3 worldNormal = normalize(mat3(m) * normal);
  vBark = uv;
` + V_END + `}`, `
uniform vec3 uColor;
uniform float uMoss;
varying vec2 vBark;
void main() {
  vec3 N = normalize(vNormalW);
  float a = vBark.x * 6.28318, v = vBark.y;
  vec3 ring = vec3(cos(a), sin(a), 0.0);
  float furrow = n_noise3(ring * 1.4 + vec3(0.0, 0.0, v * 1.2));
  float ridges = abs(sin(a * 5.0 + furrow * 5.0 + v * 0.6));
  float plates = n_noise3(ring * 3.0 + vec3(0.0, 0.0, v * 7.0));
  vec3 col = lin(uColor) * (0.45 + 0.55 * smoothstep(0.1, 0.9, ridges)) * (0.78 + 0.44 * plates);
  float moss = smoothstep(0.55, 0.8, n_noise3(ring * 0.9 + vec3(3.0, 0.0, v * 0.7))) * (1.0 - smoothstep(0.5, 3.0, v));
  col = mix(col, lin(vec3(0.3, 0.42, 0.17)) * (0.7 + 0.5 * plates), moss * uMoss);
  float sh;
  col *= gather(N, 0.15, sh);
  gl_FragColor = outCol(applyFog(col), 1.0);
}`, { uColor: colour(hex), uMoss: { value: mossAmt } });

  // Stone: granite grain and speckle, lichen and moss on upward faces, rain streaks,
  // and optional coursed blocks with mortar for built walls.
  const stoneMat = (hex, bricks, moss) => makeMat(true, V_LOCAL, `
uniform vec3 uColor;
uniform float uBricks;
uniform float uMoss;
varying vec3 vLocal;
varying vec3 vLocalN;
void main() {
  vec3 p = vWorldPos;
  vec3 N = normalize(vNormalW);
  vec3 base = lin(uColor);
  float grain = n_noise3(p * 7.0) * 0.6 + n_noise3(p * 23.0) * 0.4;
  vec3 col = base * (0.7 + 0.45 * n_fbm3(p * 0.8)) * (0.86 + 0.24 * grain);
  col = mix(col, base * 1.4, step(0.86, n_hash13(floor(p * 30.0))) * 0.45);
  col = mix(col, base * 0.55, step(0.9, n_hash13(floor(p * 30.0) + 7.0)) * 0.5);
  float relief = grain;
  if (uBricks > 0.5 && abs(vLocalN.y) < 0.6) {
    float ang = atan(vLocal.z, vLocal.x);
    vec2 bq = vec2((ang / 6.28318 + 0.5) * 13.0, vLocal.y / 0.42);
    bq.x += mod(floor(bq.y), 2.0) * 0.5;
    vec2 f = fract(bq);
    float edgeD = min(min(f.x, 1.0 - f.x) * 0.8, min(f.y, 1.0 - f.y) * 0.42);
    float mortar = 1.0 - smoothstep(0.015, 0.045, edgeD);
    col *= 0.8 + 0.4 * n_hash12(floor(bq));
    col = mix(col, lin(vec3(0.52, 0.5, 0.44)) * (0.8 + 0.3 * grain), mortar);
    relief = mix(relief, -0.8, mortar);
  }
  float up = clamp(N.y, 0.0, 1.0);
  float lich = smoothstep(0.58, 0.72, n_fbm3(p * 1.6 + 4.0));
  col = mix(col, lin(vec3(0.8, 0.7, 0.33)) * (0.8 + 0.3 * grain), lich * 0.55 * uMoss);
  float mossy = smoothstep(0.5, 0.9, up + (n_fbm3(p * 1.2) - 0.5) * 0.9);
  col = mix(col, lin(vec3(0.3, 0.42, 0.18)) * (0.75 + 0.4 * grain), mossy * 0.75 * uMoss);
  col *= 1.0 - (1.0 - up) * smoothstep(0.55, 0.9, n_noise(vec2((p.x + p.z) * 6.0, p.y * 0.5))) * 0.3;
  float sh;
  col *= gather(N, 0.1, sh) * (0.86 + 0.14 * relief);
  gl_FragColor = outCol(applyFog(col), 1.0);
}`, { uColor: colour(hex), uBricks: { value: bricks ? 1 : 0 }, uMoss: { value: moss } });

  // Canvas: woven cloth with stitched seams, stains, mud at the hem, and sun showing through.
  const canvasMat = (hex) => makeMat(true, V_LOCAL, `
uniform vec3 uColor;
varying vec3 vLocal;
varying vec3 vLocalN;
void main() {
  vec3 p = vWorldPos;
  vec3 N = normalize(vNormalW);
  vec3 base = lin(uColor);
  float ang = atan(vLocal.z, vLocal.x);
  float weave = (sin((vLocal.x + vLocal.z) * 150.0) * sin(vLocal.y * 150.0)) * 0.5 + 0.5;
  float seam = 1.0 - smoothstep(0.0, 0.025, abs(fract(vLocal.y * 1.6) - 0.5));
  float stitch = seam * step(0.5, fract(ang * 18.0));
  float stain = smoothstep(0.45, 0.75, n_fbm(vec2(ang * 2.0, vLocal.y * 1.5) + 3.0));
  base *= (0.9 + 0.1 * weave) * (1.0 - 0.18 * stain) * (1.0 - seam * 0.18);
  base = mix(base, lin(vec3(0.95, 0.9, 0.75)), stitch * 0.5);
  float mud = 1.0 - smoothstep(-1.05, -0.55, vLocal.y + (n_noise(vec2(ang * 4.0, 0.0)) - 0.5) * 0.25);
  base = mix(base, lin(vec3(0.36, 0.28, 0.2)), mud * 0.6);
  float sh;
  vec3 col = base * gather(N, 0.35, sh);
  vec3 V = normalize(cameraPosition - p);
  col += base * uSunCol * pow(max(dot(-V, uSunDir), 0.0), 4.0) * 0.25 * (1.0 - max(dot(N, uSunDir), 0.0));
  gl_FragColor = outCol(applyFog(col), 1.0);
}`, { uColor: colour(hex) });

  // Sprites: a camera-facing card that shows one frame of a character sheet. Pixels stay crisp,
  // and the card still takes the scene's light, tree shadows, firelight and fog.
  const spriteMat = (sheet) => makeMat(true, `
uniform vec2 uSize;
varying vec2 vCard;
void main() {
  vec3 centre = (modelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  vec3 right = normalize(vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]));
  vec3 camUp = normalize(vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]));
  // lean the card most of the way toward the camera so it isn't squashed when seen from above
  vec3 up = normalize(mix(vec3(0.0, 1.0, 0.0), camUp, 0.7));
  float stretch = 1.0 / max(dot(up, camUp), 0.5);
  vec4 worldPosition = vec4(centre + right * position.x * uSize.x + up * (position.y - 0.05) * uSize.y * stretch, 1.0);
  vec3 worldNormal = vec3(0.0, 1.0, 0.0);
  vCard = vec2(position.x + 0.5, position.y);
` + V_END + `
  // The card leans toward the camera, so its bottom rows (the feet) sink into the ground and the terrain
  // hides them, cutting both feet off along one line. For the depth test only, pull the card along the view
  // ray toward the camera (the picture stays where it is), so the ground no longer covers the feet.
  vec4 pulled = projectionMatrix * vec4(mvPosition.xyz * (1.0 - 0.8 / max(length(mvPosition.xyz), 1.0)), 1.0);
  gl_Position.z = pulled.z / pulled.w * gl_Position.w;
}`, `
uniform sampler2D uSheet;
uniform vec2 uGrid;
uniform vec3 uFrame;   // column, row, mirror
uniform vec3 uSpriteRef;
varying vec2 vCard;
void main() {
  vec2 uv = vCard;
  if (uFrame.z > 0.5) uv.x = 1.0 - uv.x;
  vec4 t = texture2D(uSheet, (vec2(uFrame.x, uGrid.y - 1.0 - uFrame.y) + uv) / uGrid);
  if (t.a < 0.5) discard;
  // The sprites are painted with their own light and shade, so they skip the world's exposure and tone curve,
  // which would crush their shadow tones and push the mid tones brighter and more saturated. The world's
  // light only scales them, relative to the light in the open (uSpriteRef): out in daylight a sprite shows
  // exactly its painted colours, and it darkens and cools in shadow, warms by a fire, and dims at night.
  float sh;
  vec3 k = min(gather(vec3(0.0, 1.0, 0.0), 1.0, sh) / uSpriteRef, vec3(1.35));
  vec3 col = sRGBToLinear(vec4(t.rgb, 1.0)).rgb * k;
  // far off it fades into the fog as the world draws it (through its tone curve), so it sits in the scene
  col = mix(col, toneMap(lin(fogColor)), fogAmount());
  gl_FragColor = linearToOutputTexel(vec4(col, 1.0));
}`, { uSheet: { value: sheet }, uGrid: { value: new THREE.Vector2(COLS, ROWS) }, uFrame: { value: new THREE.Vector3(0, 0, 0) }, uSize: { value: new THREE.Vector2(1.76, 2.2) } });

  // The contact shadow and the rings lie flat on the ground, so grass tufts, the slope of the terrain and the
  // ground's own surface would cut through them. For the depth test only, pull them along the view ray toward
  // the camera (the picture stays where it is), so they draw over everything on the ground round the feet.
  // The pull is less than the sprite's own (0.8, see spriteMat), so the feet still stand on the shadow.
  const GROUND_PULL = `
  vec4 pulledG = projectionMatrix * vec4(mv.xyz * (1.0 - 0.6 / max(length(mv.xyz), 1.0)), 1.0);
  gl_Position.z = pulledG.z / pulledG.w * gl_Position.w;`;

  // Contact shadow under each sprite: a crisp, dark ellipse that is always there, drawn over the grass and the
  // ground (see GROUND_PULL) so nothing hides it. A dark core inside a slightly lighter band, each with a hard
  // edge (smoothed over a hair, so it doesn't shimmer), so it still reads on dark grass and at night. The colour
  // leans cool violet, as the sprites' own shadows do, instead of a grey that sinks into the ground.
  const blobShadowMat = makeMat(false, `
varying vec2 vDisc;
void main() {
  vDisc = position.xy;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vFogDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
` + GROUND_PULL + `
}`, `
varying vec2 vDisc;
void main() {
  float r = length(vDisc) / 0.5;
  float inside = 1.0 - smoothstep(0.96, 1.0, r);
  float core = 1.0 - smoothstep(0.6, 0.64, r);
  float a = mix(0.58, 0.8, core) * inside * (1.0 - fogAmount());
  gl_FragColor = linearToOutputTexel(vec4(mix(lin(vec3(0.08, 0.06, 0.13)), lin(vec3(0.03, 0.02, 0.06)), core), a));
}`, {}, { transparent: true, depthWrite: false });

  // Ground rings for the party and the move target; the target ring has marching dashes.
  const ringMat = (hex, inner, outer, dash) => makeMat(false, `
varying vec2 vRing;
void main() {
  vRing = position.xy;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vFogDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
` + GROUND_PULL + `
}`, `
uniform vec3 uColor;
uniform float uInner;
uniform float uOuter;
uniform float uDash;
varying vec2 vRing;
void main() {
  float t = (length(vRing) - uInner) / (uOuter - uInner);
  float band = smoothstep(0.0, 0.3, t) * (1.0 - smoothstep(0.7, 1.0, t));
  float ang = atan(vRing.y, vRing.x);
  float dash = step(0.45, fract(ang * 8.0 / 6.28318 - uTime * 0.6));
  float a = band * mix(0.9, 0.3 + 0.7 * dash, uDash) * (1.0 - fogAmount());
  gl_FragColor = outCol(lin(uColor) * 1.4, a);
}`, { uColor: colour(hex), uInner: { value: inner }, uOuter: { value: outer }, uDash: { value: dash ? 1 : 0 } },
  { transparent: true, depthWrite: false });

  // Flames: camera-facing cards with rising, turbulent noise, blended additively.
  const flameMat = (w, h, seedVal) => makeMat(false, `
uniform vec2 uSize;
varying vec2 vUv2;
void main() {
  vec3 centre = (modelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  vec3 right = normalize(vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]));
  vec4 wp = vec4(centre + right * position.x * uSize.x + vec3(0.0, position.y * uSize.y, 0.0), 1.0);
  vUv2 = position.xy;
  vec4 mv = viewMatrix * wp;
  vFogDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}`, `
uniform float uSeed;
uniform float uPower;
varying vec2 vUv2;
void main() {
  vec2 uv = vUv2;
  float n = n_fbm(vec2(uv.x * 3.2 + uSeed, uv.y * 2.4 - uTime * 2.1));
  float n2 = n_noise(vec2(uv.x * 7.0 + uSeed, uv.y * 5.0 - uTime * 3.4));
  float width = 0.42 * (1.0 - uv.y) + 0.04;
  float d = abs(uv.x + (n - 0.5) * 0.3 * uv.y) / width;
  float body = 1.0 - smoothstep(0.0, 1.0, d);
  float heat = body * (1.0 - uv.y * 0.95) * (0.55 + 0.9 * n) + n2 * 0.15 * body;
  heat *= smoothstep(0.0, 0.08, uv.y) * uPower;
  vec3 col = mix(lin(vec3(0.85, 0.2, 0.04)), lin(vec3(1.0, 0.56, 0.12)), smoothstep(0.15, 0.55, heat));
  col = mix(col, lin(vec3(1.0, 0.94, 0.72)), smoothstep(0.6, 1.0, heat));
  float a = smoothstep(0.02, 0.5, heat) * (1.0 - fogAmount());
  gl_FragColor = linearToOutputTexel(vec4(col * 1.6, a));
}`, { uSize: { value: new THREE.Vector2(w, h) }, uSeed: { value: seedVal }, uPower: { value: 1 } },
  { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });

  // Embers: points that rise from each fire, drift, cool and fade.
  const emberMat = (seedVal) => makeMat(false, `
attribute float aSeed;
uniform float uSeedOff;
varying float vLife;
void main() {
  float s = fract(aSeed + uSeedOff);
  float life = fract(uTime * (0.3 + 0.25 * fract(s * 7.13)) + s);
  vec3 c = (modelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  float a = s * 40.0 + uTime * 0.8;
  vec3 off = vec3((sin(a) * 0.3 + sin(uTime * 2.0 + s * 9.0) * 0.2) * life, 0.3 + life * 3.4, cos(a) * 0.3 * life);
  vec4 mv = viewMatrix * vec4(c + off, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(1.5, 0.08 * projectionMatrix[1][1] * uViewH * 0.5 / -mv.z);
  vLife = life;
  vFogDepth = -mv.z;
}`, `
varying float vLife;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = (1.0 - smoothstep(0.2, 0.5, d)) * (1.0 - vLife) * smoothstep(0.0, 0.08, vLife) * (1.0 - fogAmount());
  vec3 col = mix(lin(vec3(1.0, 0.75, 0.3)), lin(vec3(0.9, 0.2, 0.05)), vLife) * 2.0;
  gl_FragColor = linearToOutputTexel(vec4(col, a));
}`, { uSeedOff: { value: seedVal } }, { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });

  // Sky: gradient dome with sun or moon, drifting clouds by day and stars by night.
  const sky = new THREE.Mesh(new THREE.SphereGeometry(450, 32, 16), makeMat(false, `
varying vec3 vDir;
void main() {
  vDir = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vFogDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}`, `
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  vec3 col = skyColour(d);
  float s = dot(d, uSunDir);
  col += mix(uSunCol * 6.0, vec3(1.6, 1.65, 1.8), uNight) * smoothstep(0.9993, 0.9997, s);
  if (d.y > 0.0) {
    vec2 cuv = d.xz / (d.y + 0.15) * 1.4 + vec2(uTime * 0.008, uTime * 0.003);
    float cloud = smoothstep(0.5, 0.8, n_fbm(cuv)) * smoothstep(0.0, 0.2, d.y);
    vec3 cloudCol = mix(lin(vec3(1.0, 0.99, 0.96)), lin(vec3(0.16, 0.2, 0.28)), uNight);
    col = mix(col, cloudCol * (0.75 + 0.35 * n_fbm(cuv * 2.0 + 5.0)), cloud * 0.7);
    vec3 q = d * 160.0;
    float star = step(0.996, n_hash13(floor(q))) * (1.0 - smoothstep(0.05, 0.2, length(fract(q) - 0.5)));
    col += vec3(star) * uNight * (0.7 + 0.3 * sin(uTime * 3.0 + n_hash13(floor(q)) * 40.0)) * smoothstep(0.05, 0.3, d.y) * (1.0 - cloud);
  }
  gl_FragColor = outCol(col, 1.0);
}`, {}, { side: THREE.BackSide, depthWrite: false, fog: false }));
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  scene.add(sky);

  // ---------- Terrain mesh ----------
  (function buildTerrain() {
    const geo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) pos.setY(i, height(pos.getX(i), pos.getZ(i)));
    geo.computeVertexNormals();
    // aPath marks trodden ground: the paths, plus the camp and hilltop clearings
    const worn = new Float32Array(pos.count);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      let d = 1 - smoothstep(0.9, 2.2, pathDist(x, z));
      for (const s of CLEAR) d = Math.max(d, 0.95 * (1 - smoothstep(s.r * 0.45, s.r * 0.95, Math.hypot(x - s.x, z - s.z))));
      worn[i] = d;
    }
    geo.setAttribute('aPath', new THREE.BufferAttribute(worn, 1));
    const ground = new THREE.Mesh(geo, terrainMat);
    ground.receiveShadow = true;
    scene.add(ground);
  })();

  const water = new THREE.Mesh(new THREE.PlaneGeometry(SIZE, SIZE), waterMat);
  water.rotation.x = -Math.PI / 2;
  water.position.y = WATER;
  water.receiveShadow = true;
  water.renderOrder = 1;
  scene.add(water);

  // ---------- Scenery ----------
  const obstacles = [];   // circles the party steers around
  function solid(geo, material, x, y, z) {
    const m = new THREE.Mesh(geo, material);
    m.position.set(x, y, z);
    m.castShadow = true; m.receiveShadow = true;
    scene.add(m);
    return m;
  }
  const nearSpot = (x, z, pad) => CLEAR.some((s) => Math.hypot(x - s.x, z - s.z) < s.r + pad);

  // ---------- Trees ----------
  // Each tree is grown from code: a curved, tapering trunk tube, branches, and a crown.
  // Seven variants are built once and instanced across the map.
  const V3 = THREE.Vector3;
  const newBuf = () => ({ pos: [], nor: [], uv: [], shade: [] });
  function pushV(buf, p, n, u, v, shade) { buf.pos.push(p.x, p.y, p.z); buf.nor.push(n.x, n.y, n.z); buf.uv.push(u, v); buf.shade.push(shade); }
  function toGeometry(buf) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(buf.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(buf.nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(buf.uv, 2));
    g.setAttribute('aShade', new THREE.Float32BufferAttribute(buf.shade, 1));
    return g;
  }
  // a tube swept along a polyline; uv.x wraps around it, uv.y is distance along it
  function tube(buf, pts, radii, sides) {
    const rings = [];
    let ref = new V3(1, 0, 0), along = 0;
    for (let i = 0; i < pts.length; i++) {
      const t = new V3().subVectors(pts[Math.min(i + 1, pts.length - 1)], pts[Math.max(i - 1, 0)]).normalize();
      let nx = ref.clone().addScaledVector(t, -ref.dot(t));
      if (nx.lengthSq() < 1e-4) nx = new V3(0, 0, 1).addScaledVector(t, -t.z);
      nx.normalize(); ref = nx;
      const bz = new V3().crossVectors(t, nx);
      if (i > 0) along += pts[i].distanceTo(pts[i - 1]);
      const ring = [];
      for (let k = 0; k <= sides; k++) {
        const a = k / sides * Math.PI * 2;
        const dir = nx.clone().multiplyScalar(Math.cos(a)).addScaledVector(bz, Math.sin(a));
        ring.push({ p: pts[i].clone().addScaledVector(dir, radii[i]), n: dir, u: k / sides, v: along });
      }
      rings.push(ring);
    }
    for (let i = 0; i < rings.length - 1; i++) for (let k = 0; k < sides; k++) {
      const a = rings[i][k], b = rings[i][k + 1], c = rings[i + 1][k], d = rings[i + 1][k + 1];
      [a, b, c, b, d, c].forEach((q) => pushV(buf, q.p, q.n, q.u, q.v, 1));
    }
  }
  // one flat triangle, wound so it faces away from `inside`
  function facet(buf, a, b, c, inside, sa, sb, sc, round) {
    const n = new V3().subVectors(b, a).cross(new V3().subVectors(c, a)).normalize();
    const out = new V3().addVectors(a, b).add(c).multiplyScalar(1 / 3).sub(inside);
    if (n.dot(out) < 0) { n.negate(); let t = b; b = c; c = t; t = sb; sb = sc; sc = t; }
    // with a rounding centre, lean each corner's normal outward from it so facets shade as one curved surface
    const soft = (q) => round ? new V3().subVectors(q, round).normalize().multiplyScalar(0.7).addScaledVector(n, 0.3).normalize() : n;
    pushV(buf, a, soft(a), 0, 0, sa); pushV(buf, b, soft(b), 0, 0, sb); pushV(buf, c, soft(c), 0, 0, sc);
  }
  // ---------- Foliage textures and materials ----------
  // Leaf, needle and grass textures are painted in code into small alpha textures.
  // Cards carrying them break up the outline of every crown and cover the ground in tufts.
  function paintTexture(S, paint) {
    const data = new Uint8Array(S * S * 4);
    const put = (x, y, tone, tint) => {
      if (x < 0 || y < 0 || x >= S || y >= S) return;
      const k = (y * S + x) * 4;
      data[k] = Math.min(255, tone * tint[0] * 255); data[k + 1] = Math.min(255, tone * tint[1] * 255);
      data[k + 2] = Math.min(255, tone * tint[2] * 255); data[k + 3] = 255;
    };
    paint(put);
    const t = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
    t.generateMipmaps = true;
    t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter;
    t.needsUpdate = true;
    return t;
  }
  // a spray of pointed leaves on short stalks, each with a midrib, a folded half and a lighter edge
  const leafTex = paintTexture(128, (put) => {
    const S = 128;
    const leaves = [];
    for (let i = 0; i < 13; i++) {
      const a = rand() * Math.PI * 2, d = 0.12 + rand() * 0.2;
      leaves.push({ cx: 0.5 + Math.cos(a) * d, cy: 0.5 + Math.sin(a) * d, ang: a + (rand() - 0.5) * 1.2,
        len: 0.15 + rand() * 0.08, wid: 0.065 + rand() * 0.03, tone: 0.72 + rand() * 0.28, warm: rand() });
    }
    leaves.sort((p, q) => p.tone - q.tone);
    leaves.forEach((L) => {
      const ca = Math.cos(L.ang), sa = Math.sin(L.ang), r = L.len + 0.02;
      for (let y = Math.floor((L.cy - r) * S); y <= (L.cy + r) * S; y++) for (let x = Math.floor((L.cx - r) * S); x <= (L.cx + r) * S; x++) {
        const px = x / S - L.cx, py = y / S - L.cy;
        const u = (px * ca + py * sa) / L.len, v = (-px * sa + py * ca);
        if (u < -1 || u > 1) continue;
        const half = L.wid * Math.pow(1 - u * u, 0.7) * (u > 0 ? 1 - u * 0.35 : 1);
        if (Math.abs(v) > half) continue;
        let tone = L.tone * (v > 0 ? 1 : 0.8);
        if (Math.abs(v) < 0.006) tone *= 0.7;
        if (Math.abs(v) > half * 0.8) tone *= 1.1;
        tone *= 0.9 + 0.2 * Math.abs(Math.sin(u * 9 + v * 40));
        put(x, y, tone, [0.92 + L.warm * 0.12, 1, 0.78 - L.warm * 0.1]);
      }
      // stalk back toward the centre of the spray
      for (let t = 0; t < 1; t += 0.02) {
        const sx = L.cx - ca * L.len * (1 + t * 0.5), sy = L.cy - sa * L.len * (1 + t * 0.5);
        put(Math.round(sx * S), Math.round(sy * S), 0.5, [0.85, 0.8, 0.6]);
      }
    });
  });
  // a conifer sprig: a twig running down the card with needles angled along it
  const needleTex = paintTexture(128, (put) => {
    const S = 128;
    const line = (x0, y0, x1, y1, tone, tint) => {
      const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) * S);
      for (let i = 0; i <= n; i++) { const t = i / n, x = Math.round((x0 + (x1 - x0) * t) * S), y = Math.round((y0 + (y1 - y0) * t) * S); put(x, y, tone, tint); put(x + 1, y, tone * 0.92, tint); }
    };
    for (let side = 0; side < 3; side++) {
      const x0 = 0.3 + side * 0.2 + (rand() - 0.5) * 0.06, lean = (side - 1) * 0.12;
      for (let y = 0.95; y > 0.06; y -= 0.018) {
        const t = (0.95 - y) / 0.9, x = x0 + lean * t;
        const len = 0.06 + 0.1 * Math.sin(t * Math.PI * 0.9 + 0.2);
        [-1, 1].forEach((s) => line(x, y, x + s * len * 0.75, y - len * 0.75, 0.68 + rand() * 0.32, [0.86, 1, 0.82]));
      }
      line(x0, 0.97, x0 + lean, 0.05, 0.45, [0.85, 0.7, 0.55]);
    }
  });
  // a clump of grass blades, darker at the root, bending as they rise
  const grassTex = paintTexture(128, (put) => {
    const S = 128;
    for (let i = 0; i < 22; i++) {
      const bx = 0.12 + rand() * 0.76, h = 0.45 + rand() * 0.52, lean = (rand() - 0.5) * 0.5, w = 0.016 + rand() * 0.016;
      const tint = rand() < 0.15 ? [1.05, 1, 0.62] : [0.88, 1, 0.75];
      for (let y = 0; y < h * S; y++) {
        const t = y / (h * S);
        const cx = (bx + lean * t * t) * S, half = w * S * (1 - t) + 0.5;
        for (let x = Math.floor(cx - half); x <= cx + half; x++) put(x, y, 0.45 + 0.55 * t, tint);
      }
    }
  });

  const cardVertex = `
attribute float aShade;
varying vec3 vColor;
varying float vShade;
varying vec2 vLeafUv;
void main() {
  mat4 m = modelOf();
  vec4 worldPosition = m * vec4(position, 1.0);
  vec3 base = (m * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  worldPosition.xz += windSway(base, max(worldPosition.y - base.y, 0.0));
  float flutter = sin(uTime * 5.0 + dot(worldPosition.xyz, vec3(3.1, 1.7, 2.3))) * 0.03;
  worldPosition.xyz += vec3(flutter, flutter * 0.5, -flutter);
  vec3 worldNormal = normalize(mat3(m) * normal);
#ifdef USE_INSTANCING_COLOR
  vColor = instanceColor;
#else
  vColor = vec3(0.3, 0.5, 0.3);
#endif
  vShade = aShade;
  vLeafUv = uv;
` + V_END + `}`;
  // leaf cards are lit with the crown's rounded normal, so they shade as part of one mass
  const cardMat = (tex) => makeMat(true, cardVertex, `
uniform sampler2D uLeaf;
varying vec3 vColor;
varying float vShade;
varying vec2 vLeafUv;
void main() {
  vec4 t = texture2D(uLeaf, vLeafUv);
  if (t.a < 0.42) discard;
  vec3 p = vWorldPos;
  vec3 N = normalize(vNormalW);
  vec3 base = lin(vColor);
  vec3 col = base * (0.55 + 0.5 * t.rgb) * mix(0.4, 1.08, vShade);
  col = mix(col, col * vec3(1.18, 1.12, 0.78), n_noise3(p * 0.6) * 0.45);
  float sh;
  vec3 irr = gather(N, 0.5, sh);
  vec3 V = normalize(cameraPosition - p);
  vec3 through = base * uSunCol * pow(max(dot(-V, uSunDir), 0.0), 3.0) * 0.55 * sh * vShade;
  col = col * irr + through;
  col += base * irr * pow(1.0 - abs(dot(N, V)), 3.0) * 0.25;
  gl_FragColor = outCol(applyFog(col), 1.0);
}`, { uLeaf: { value: tex } }, { side: THREE.DoubleSide });
  const leafCards = cardMat(leafTex), needleCards = cardMat(needleTex);
  // shadows that follow the leaf shapes instead of square cards
  const cardDepth = (tex) => new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: tex, alphaTest: 0.42, side: THREE.DoubleSide });
  const leafDepth = cardDepth(leafTex), needleDepth = cardDepth(needleTex);

  // grass tufts take their colour from the same noise as the ground under them, so they sit in it
  const grassMat = makeMat(true, `
varying vec3 vColor;
varying vec3 vBase;
varying vec2 vLeafUv;
void main() {
  mat4 m = modelOf();
  vec4 worldPosition = m * vec4(position, 1.0);
  vec3 base = (m * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
  float tip = uv.y * uv.y;
  worldPosition.xz += (windSway(base, 6.0) * 0.7 + vec2(sin(uTime * 2.3 + base.x * 0.7), cos(uTime * 1.9 + base.z * 0.6)) * 0.05) * tip;
  vec3 worldNormal = vec3(0.0, 1.0, 0.0);
#ifdef USE_INSTANCING_COLOR
  vColor = instanceColor;
#else
  vColor = vec3(1.0);
#endif
  vBase = base;
  vLeafUv = uv;
` + V_END + `}`, `
uniform sampler2D uLeaf;
varying vec3 vColor;
varying vec3 vBase;
varying vec2 vLeafUv;
void main() {
  vec4 t = texture2D(uLeaf, vLeafUv);
  if (t.a < 0.42) discard;
  vec3 p = vWorldPos;
  float large = n_fbm(vBase.xz * 0.045);
  vec3 grass = mix(lin(vec3(0.27, 0.4, 0.17)), lin(vec3(0.44, 0.54, 0.25)), smoothstep(0.3, 0.72, large));
  grass = mix(grass, lin(vec3(0.6, 0.56, 0.34)), smoothstep(0.55, 0.78, n_fbm(vBase.xz * 0.021 + 7.0)) * 0.55);
  grass *= 0.72 + 0.5 * n_fbm(vBase.xz * 0.4 + 2.0);
  vec3 col = grass * vColor * (0.6 + 0.55 * t.rgb);
  float sh;
  vec3 irr = gather(vec3(0.0, 1.0, 0.0), 0.3, sh);
  vec3 V = normalize(cameraPosition - p);
  col = col * irr + grass * uSunCol * pow(max(dot(-V, uSunDir), 0.0), 3.0) * 0.4 * sh * vLeafUv.y;
  gl_FragColor = outCol(applyFog(col), 1.0);
}`, { uLeaf: { value: grassTex } }, { side: THREE.DoubleSide });

  // one quad centred on c, spanning +/-a and +/-b; n is the normal used for lighting
  function card(buf, c, a, b, n, shade) {
    const q = [c.clone().sub(a).sub(b), c.clone().add(a).sub(b), c.clone().add(a).add(b), c.clone().sub(a).add(b)];
    const uv = [[0, 0], [1, 0], [1, 1], [0, 1]];
    [0, 1, 2, 0, 2, 3].forEach((i) => pushV(buf, q[i], n, uv[i][0], uv[i][1], shade));
  }
  // scatter leaf cards over a crown surface, tilted at random so the outline is broken from any angle
  function crownCards(geo, count, size) {
    const buf = newBuf(), pos = geo.attributes.position, nor = geo.attributes.normal, sh = geo.attributes.aShade;
    for (let i = 0; i < count; i++) {
      const k = Math.floor(rand() * pos.count);
      const n = new V3(nor.getX(k), nor.getY(k), nor.getZ(k));
      const c = new V3(pos.getX(k), pos.getY(k), pos.getZ(k)).addScaledVector(n, 0.05 + rand() * 0.15);
      const t = new V3().crossVectors(n, new V3(rand() - 0.5, rand() - 0.5, rand() - 0.5)).normalize();
      const b = new V3().crossVectors(n, t).multiplyScalar(0.55).addScaledVector(n, 0.45).normalize();
      const s = size * (0.75 + rand() * 0.5);
      card(buf, c, t.multiplyScalar(s * 0.5), b.multiplyScalar(s * 0.5), n.clone().add(new V3(0, 0.25, 0)).normalize(), sh.getX(k));
    }
    return toGeometry(buf);
  }

  // Broadleaf crown: the leaf masses are fused into one continuous surface with a smooth union,
  // so neighbouring masses flow into each other instead of meeting at hard seams.
  function crown(masses) {
    const K = 0.3;
    const field = (x, y, z) => {
      let sum = 0;
      for (const m of masses) {
        const dx = x - m.c.x, dy = (y - m.c.y) / m.sq, dz = z - m.c.z;
        sum += Math.exp(-(Math.sqrt(dx * dx + dy * dy + dz * dz) - m.r) / K);
      }
      return -K * Math.log(sum);
    };
    let wsum = 0;
    const centre = new V3();
    masses.forEach((m) => { const w = m.r * m.r * m.r; centre.addScaledVector(m.c, w); wsum += w; });
    centre.multiplyScalar(1 / wsum);
    centre.y -= 0.2;
    let reach = 0;
    masses.forEach((m) => { reach = Math.max(reach, m.c.distanceTo(centre) + m.r + 0.6); });
    const s1 = rand() * 10, s2 = rand() * 10;

    // one vertex per direction on a fine sphere; march in from outside to the outermost surface
    const src = new THREE.IcosahedronGeometry(1, 4).attributes.position;
    const index = new Map(), verts = [], tris = [];
    for (let i = 0; i < src.count; i++) {
      const key = [src.getX(i), src.getY(i), src.getZ(i)].map((v) => Math.round(v * 1e4)).join(',');
      if (!index.has(key)) {
        const d = new V3(src.getX(i), src.getY(i), src.getZ(i)).normalize();
        let t = reach, prev = field(centre.x + d.x * t, centre.y + d.y * t, centre.z + d.z * t);
        while (t > 0.25) {
          const t2 = t - 0.06, f = field(centre.x + d.x * t2, centre.y + d.y * t2, centre.z + d.z * t2);
          if (f <= 0) {
            let lo = t2, hi = t;
            for (let k = 0; k < 10; k++) {
              const m = (lo + hi) / 2;
              if (field(centre.x + d.x * m, centre.y + d.y * m, centre.z + d.z * m) <= 0) lo = m; else hi = m;
            }
            t = lo; break;
          }
          t = t2; prev = f;
        }
        t *= 1 + 0.045 * Math.sin(d.x * 5.1 + s1) * Math.sin(d.z * 4.7 + s2) + 0.03 * Math.sin(d.y * 9.3 + s1) * Math.sin(d.x * 8.7 + s2) + 0.02 * Math.sin(d.z * 13.1 + s1);
        index.set(key, verts.length);
        verts.push(new V3(centre.x + d.x * t, centre.y + d.y * t, centre.z + d.z * t));
      }
      tris.push(index.get(key));
    }
    let minY = Infinity, maxY = -Infinity;
    verts.forEach((v) => { minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y); });

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(verts.flatMap((v) => [v.x, v.y, v.z]), 3));
    g.setIndex(tris);
    g.computeVertexNormals();
    const nor = g.attributes.normal;
    // shade: darker low in the crown and in the creases where masses meet
    const shade = verts.map((v, i) => {
      const n = new V3(nor.getX(i), nor.getY(i), nor.getZ(i));
      const ao = clamp(field(v.x + n.x * 0.5, v.y + n.y * 0.5, v.z + n.z * 0.5) / 0.5, 0, 1);
      const h = (v.y - minY) / Math.max(maxY - minY, 0.01);
      return (0.3 + 0.7 * smoothstep(-0.1, 0.9, h * 0.6 + (n.y * 0.5 + 0.5) * 0.4)) * (0.55 + 0.45 * ao);
    });
    g.setAttribute('aShade', new THREE.Float32BufferAttribute(shade, 1));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(verts.length * 2), 2));
    return g;
  }

  // conifer: leaning trunk with root flare, and tiers of drooping, ragged boughs
  function makePine() {
    const bark = newBuf(), leaf = newBuf(), sprigs = newBuf();
    const H = 5.6 + rand() * 1.6, lean = (rand() - 0.5) * 0.6, leanA = rand() * 6.28;
    const axis = (y) => { const u = Math.max(y, 0) / H; return new V3(Math.cos(leanA) * lean * u * u, y, Math.sin(leanA) * lean * u * u); };
    const pts = [], rad = [];
    for (let i = 0; i <= 6; i++) { pts.push(axis(-0.25 + (H + 0.05) * i / 6)); rad.push(0.23 * (1 - i / 6) + 0.03); }
    rad[0] = 0.34;
    tube(bark, pts, rad, 7);
    const tiers = 7 + Math.floor(rand() * 2);
    for (let j = 0; j < tiers; j++) {
      const f = j / (tiers - 1);
      const yRim = 1.15 + f * (H - 2.05);
      const R = lerp(1.8, 0.42, Math.pow(f, 0.85)) * (0.9 + rand() * 0.2);
      const rise = lerp(1.5, 1.0, f);
      const c = axis(yRim), apex = axis(yRim + rise);
      const n = 14, mid = [], rim = [], spin = rand() * 6.28;
      for (let k = 0; k < n; k++) {
        const a = spin + k / n * Math.PI * 2, dx = Math.cos(a), dz = Math.sin(a);
        const reach = (k % 2 ? 0.84 : 1) * (0.92 + rand() * 0.16);
        mid.push(new V3(c.x + dx * R * 0.52, yRim + rise * 0.6, c.z + dz * R * 0.52));
        rim.push(new V3(c.x + dx * R * reach, yRim - (reach - 0.8) * 0.5 - rand() * 0.08, c.z + dz * R * reach));
      }
      const below = new V3(c.x, yRim - 3, c.z), above = new V3(c.x, yRim + rise + 3, c.z);
      const under = new V3(c.x, yRim + rise * 0.35, c.z);
      const lit = lerp(0.78, 1, f), core = new V3(c.x, yRim - 0.8, c.z);
      for (let k = 0; k < n; k++) {
        const k2 = (k + 1) % n;
        facet(leaf, apex, mid[k], mid[k2], below, 0.5 * lit, 0.72 * lit, 0.72 * lit, core);
        facet(leaf, mid[k], rim[k], rim[k2], below, 0.72 * lit, lit, lit, core);
        facet(leaf, mid[k], rim[k2], mid[k2], below, 0.72 * lit, lit, 0.72 * lit, core);
        facet(leaf, under, rim[k], rim[k2], above, 0.12, 0.38, 0.38);
      }
      // sprigs hang off the rim and fan out over the tier, so its edge is feathery instead of straight
      for (let k = 0; k < n; k++) {
        const out = new V3(rim[k].x - c.x, 0, rim[k].z - c.z).normalize();
        const side = new V3(-out.z, 0, out.x);
        const nrm = out.clone().add(new V3(0, 0.6, 0)).normalize();
        const len = (0.55 + R * 0.22) * (0.85 + rand() * 0.3);
        const hang = out.clone().multiplyScalar(0.85).add(new V3(0, -0.4 - rand() * 0.2, 0)).normalize();
        card(sprigs, rim[k].clone().addScaledVector(hang, len * 0.2), side.clone().multiplyScalar(len * 0.42), hang.clone().multiplyScalar(-len * 0.5), nrm, lit);
        if (k % 2 === 0) {
          const m2 = mid[k].clone().lerp(rim[(k + 1) % n], 0.5);
          const lay = out.clone().multiplyScalar(0.8).add(new V3(0, -0.35, 0)).normalize();
          card(sprigs, m2.addScaledVector(nrm, 0.06), side.clone().multiplyScalar(len * 0.4), lay.multiplyScalar(-len * 0.45), nrm, lit * 0.9);
        }
      }
    }
    return { bark: toGeometry(bark), leaf: toGeometry(leaf), cards: toGeometry(sprigs), kind: 'pine' };
  }

  // broadleaf: bent trunk that forks into boughs, each carrying its own leaf mass
  function makeOak() {
    const bark = newBuf(), leaf = newBuf();
    const h1 = 1.7 + rand() * 0.7, bendA = rand() * 6.28, bend = 0.15 + rand() * 0.3;
    const top = new V3(Math.cos(bendA) * bend, h1, Math.sin(bendA) * bend);
    tube(bark, [new V3(0, -0.25, 0), new V3(Math.cos(bendA) * bend * 0.2, h1 * 0.45, Math.sin(bendA) * bend * 0.2), top], [0.38, 0.27, 0.22], 8);
    const masses = [];
    const boughs = 3 + Math.floor(rand() * 2), start = rand() * 6.28;
    const fork = top.clone().add(new V3(0, -0.12, 0));
    for (let b = 0; b < boughs; b++) {
      const a = start + b / boughs * Math.PI * 2 + (rand() - 0.5) * 0.7;
      const out = 1.2 + rand() * 0.7, up = 0.9 + rand() * 0.8;
      const p1 = top.clone().add(new V3(Math.cos(a) * out * 0.5, up * 0.3, Math.sin(a) * out * 0.5));
      const p2 = top.clone().add(new V3(Math.cos(a + 0.25) * out, up, Math.sin(a + 0.25) * out));
      tube(bark, [fork, p1, p2], [0.16, 0.11, 0.05], 5);
      masses.push({ c: p2.clone().add(new V3(0, 0.3, 0)), r: 1.3 + rand() * 0.45, sq: 0.92 });
      if (rand() < 0.5) masses.push({ c: p1.clone().add(new V3(Math.cos(a) * 0.5, 0.75, Math.sin(a) * 0.5)), r: 0.85 + rand() * 0.3, sq: 0.95 });
    }
    const tip = top.clone().add(new V3((rand() - 0.5) * 0.5, 1.5 + rand() * 0.4, (rand() - 0.5) * 0.5));
    tube(bark, [fork, top.clone().lerp(tip, 0.5).add(new V3(0.1, 0, 0.08)), tip], [0.18, 0.12, 0.05], 5);
    masses.push({ c: tip.clone().add(new V3(0, 0.35, 0)), r: 1.55 + rand() * 0.4, sq: 0.94 });
    const leafGeo = crown(masses);
    return { bark: toGeometry(bark), leaf: leafGeo, cards: crownCards(leafGeo, 240, 0.85), kind: 'oak' };
  }

  // dead snag: bare, crooked, with broken limbs
  function makeSnag() {
    const bark = newBuf();
    const H = 4 + rand() * 1.2, pts = [], rad = [];
    for (let i = 0; i <= 5; i++) { pts.push(new V3((rand() - 0.5) * 0.3 * i / 5, -0.25 + (H + 0.25) * i / 5, (rand() - 0.5) * 0.3 * i / 5)); rad.push(0.24 * (1 - i / 5) + 0.05); }
    rad[0] = 0.33;
    tube(bark, pts, rad, 7);
    for (let b = 0; b < 5; b++) {
      const from = pts[2 + (b % 3)], a = rand() * 6.28, len = 0.8 + rand() * 0.9;
      const p1 = from.clone().add(new V3(Math.cos(a) * len * 0.55, len * 0.25, Math.sin(a) * len * 0.55));
      const p2 = from.clone().add(new V3(Math.cos(a + 0.4) * len, len * 0.75, Math.sin(a + 0.4) * len));
      tube(bark, [from, p1, p2], [0.09, 0.06, 0.02], 4);
    }
    return { bark: toGeometry(bark), leaf: null, kind: 'snag' };
  }

  // shrub: a low cluster of leaf masses with no visible trunk
  function makeBush() {
    const masses = [], n = 2 + Math.floor(rand() * 3);
    for (let i = 0; i < n; i++) {
      const a = rand() * 6.28, d = i ? 0.35 + rand() * 0.35 : 0;
      masses.push({ c: new V3(Math.cos(a) * d, 0.35 + rand() * 0.2, Math.sin(a) * d), r: 0.45 + rand() * 0.25, sq: 0.8 });
    }
    const leafGeo = crown(masses);
    return { bark: null, leaf: leafGeo, cards: crownCards(leafGeo, 55, 0.6), kind: 'bush' };
  }

  (function buildTrees() {
    const variants = [makePine(), makePine(), makePine(), makeOak(), makeOak(), makeOak(), makeSnag(), makeBush(), makeBush(), makeBush()];
    const groups = variants.map(() => []);
    let total = 0;
    for (let tries = 0; tries < 2600 && total < 440; tries++) {
      const x = (rand() - 0.5) * 230, z = (rand() - 0.5) * 230;
      const grove = Math.sin(x * 0.08 + 1) * Math.cos(z * 0.07 - 2) + 0.35 * Math.sin(x * 0.3 + z * 0.2);
      if (grove < -0.15) continue;
      const y = height(x, z);
      if (y < WATER + 0.5 || nearSpot(x, z, 2.5) || pathDist(x, z) < 4.2) continue;
      if (Math.abs(height(x + 1, z) - y) + Math.abs(height(x, z + 1) - y) > 0.9) continue;
      // conifers keep to the high ground and their own stands; broadleaves fill the valleys
      const conifer = Math.sin(x * 0.05 + 2) * Math.cos(z * 0.045 + 1) + (y - 3) * 0.12 + (rand() - 0.5) * 0.7 > 0;
      const v = rand() < 0.04 ? 6 : (conifer ? 0 : 3) + Math.floor(rand() * 3);
      const s = variants[v].kind === 'oak' ? 0.75 + rand() * 0.6 : 0.8 + rand() * 0.8;
      groups[v].push({ x, y, z, s, rot: rand() * Math.PI * 2 });
      obstacles.push({ x, z, r: 0.42 * s });
      total++;
    }
    // shrubs gather at the woodland edges and along the paths
    for (let tries = 0, bushes = 0; tries < 3000 && bushes < 260; tries++) {
      const x = (rand() - 0.5) * 220, z = (rand() - 0.5) * 220;
      const grove = Math.sin(x * 0.08 + 1) * Math.cos(z * 0.07 - 2) + 0.35 * Math.sin(x * 0.3 + z * 0.2);
      const pd = pathDist(x, z);
      if (Math.abs(grove + 0.15) > 0.35 && !(pd > 3.4 && pd < 6)) continue;
      const y = height(x, z);
      if (y < WATER + 0.4 || nearSpot(x, z, 1.5) || pd < 3.4) continue;
      if (Math.abs(height(x + 1, z) - y) + Math.abs(height(x, z + 1) - y) > 0.8) continue;
      const v = 7 + Math.floor(rand() * 3), s = 0.7 + rand() * 0.7;
      groups[v].push({ x, y: y - 0.05, z, s, rot: rand() * Math.PI * 2 });
      obstacles.push({ x, z, r: 0.55 * s });
      bushes++;
    }
    const liveBark = barkMat(0x6b4a30, 0.6), pineBark = barkMat(0x5a3a28, 0.3), deadBark = barkMat(0x8a8378, 0.15);
    const palettes = { pine: [0x2a5f3c, 0x2f6b3f, 0x28583a, 0x3a7044], oak: [0x35692f, 0x3d7231, 0x47782f, 0x316030, 0x56762c], bush: [0x3e6e30, 0x4a7834, 0x55772c, 0x386a3a, 0x6b7a2e] };
    const d = new THREE.Object3D(), col = new THREE.Color();
    variants.forEach((variant, v) => {
      const list = groups[v];
      if (!list.length) return;
      const parts = [];
      if (variant.bark) parts.push(new THREE.InstancedMesh(variant.bark, variant.kind === 'snag' ? deadBark : variant.kind === 'pine' ? pineBark : liveBark, list.length));
      const coloured = [];
      if (variant.leaf) coloured.push(new THREE.InstancedMesh(variant.leaf, foliageMat, list.length));
      if (variant.cards) {
        const pine = variant.kind === 'pine';
        const cards = new THREE.InstancedMesh(variant.cards, pine ? needleCards : leafCards, list.length);
        cards.customDepthMaterial = pine ? needleDepth : leafDepth;
        coloured.push(cards);
      }
      const all = parts.concat(coloured);
      list.forEach((t, i) => {
        d.position.set(t.x, t.y, t.z);
        d.rotation.set(0, t.rot, 0);
        d.scale.setScalar(t.s);
        d.updateMatrix();
        all.forEach((p) => p.setMatrixAt(i, d.matrix));
        if (coloured.length) {
          const pal = palettes[variant.kind];
          col.setHex(pal[Math.floor(rand() * pal.length)]);
          coloured.forEach((p) => p.setColorAt(i, col));
        }
      });
      all.forEach((p) => { p.castShadow = true; p.receiveShadow = true; p.frustumCulled = false; scene.add(p); });
    });
  })();

  // grass tufts (instanced): three crossed cards each, on open grass only
  (function buildGrass() {
    const buf = newBuf();
    for (let k = 0; k < 3; k++) {
      const a = k / 3 * Math.PI;
      const side = new V3(Math.cos(a), 0, Math.sin(a)), n = new V3(0, 1, 0);
      card(buf, new V3(0, 0.22, 0), side.multiplyScalar(0.32), new V3(0, 0.22, 0), n, 1);
    }
    const geo = toGeometry(buf);
    const list = [];
    for (let tries = 0; tries < 80000 && list.length < 20000; tries++) {
      const x = (rand() - 0.5) * 210, z = (rand() - 0.5) * 210;
      if (Math.sin(x * 0.13) * Math.cos(z * 0.11) + Math.sin(x * 0.37 + z * 0.29) * 0.5 < -0.35) continue;
      const y = height(x, z);
      if (y < WATER + 0.8 || pathDist(x, z) < 2.3 + rand() * 0.8 || nearSpot(x, z, 0.5)) continue;
      if (Math.abs(height(x + 0.5, z) - height(x - 0.5, z)) + Math.abs(height(x, z + 0.5) - height(x, z - 0.5)) > 0.45) continue;
      list.push({ x, y, z });
    }
    const tufts = new THREE.InstancedMesh(geo, grassMat, list.length);
    const d = new THREE.Object3D(), col = new THREE.Color();
    list.forEach((t, i) => {
      d.position.set(t.x, t.y - 0.03, t.z);
      d.rotation.set(0, rand() * Math.PI, 0);
      const s = 0.6 + rand() * 0.8;
      d.scale.set(s, s * (0.7 + rand() * 0.6), s);
      d.updateMatrix();
      tufts.setMatrixAt(i, d.matrix);
      const v = 0.85 + rand() * 0.3;
      col.setRGB(v * (0.95 + rand() * 0.1), v, v * (0.9 + rand() * 0.15));
      tufts.setColorAt(i, col);
    });
    tufts.receiveShadow = true;
    tufts.frustumCulled = false;
    scene.add(tufts);
  })();

  // boulders (instanced)
  (function buildRocks() {
    const list = [];
    for (let tries = 0; tries < 700 && list.length < 110; tries++) {
      const x = (rand() - 0.5) * 220, z = (rand() - 0.5) * 220;
      const y = height(x, z);
      if (y < WATER - 0.2 || nearSpot(x, z, 1.5) || pathDist(x, z) < 4.2) continue;
      const s = 0.4 + rand() * rand() * 1.8;
      list.push({ x, y, z, s });
      obstacles.push({ x, z, r: s * 0.8 });
    }
    const rocks = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0), stoneMat(0x8b8f88, false, 1), list.length);
    const d = new THREE.Object3D();
    list.forEach((r, i) => {
      d.position.set(r.x, r.y + r.s * 0.15, r.z);
      d.rotation.set(rand() * 3, rand() * 3, rand() * 3);
      d.scale.set(r.s, r.s * (0.55 + rand() * 0.4), r.s * (0.8 + rand() * 0.4));
      d.updateMatrix();
      rocks.setMatrixAt(i, d.matrix);
    });
    rocks.castShadow = true; rocks.receiveShadow = true; rocks.frustumCulled = false;
    scene.add(rocks);
  })();

  // fires
  const fires = [];
  const flameGeo = new THREE.PlaneGeometry(1, 1); flameGeo.translate(0, 0.5, 0);
  const emberGeo = (function () {
    const n = 28, g = new THREE.BufferGeometry(), seeds = new Float32Array(n);
    for (let i = 0; i < n; i++) seeds[i] = rand();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
    return g;
  })();
  function addFire(x, z, lift) {
    const y = height(x, z) + lift;
    const spot = new THREE.Group();
    spot.position.set(x, y, z);
    scene.add(spot);
    const mats = [flameMat(1.0, 1.7, rand() * 10), flameMat(0.55, 1.0, rand() * 10)];
    mats.forEach((m) => { const f = new THREE.Mesh(flameGeo, m); f.frustumCulled = false; spot.add(f); });
    const embers = new THREE.Points(emberGeo, emberMat(rand()));
    embers.frustumCulled = false;
    spot.add(embers);
    const light = new THREE.PointLight(0xff8a3c, 1, 26, 2);
    light.position.set(x, y + 1.4, z);
    scene.add(light);
    fires.push({ mats, light, phase: rand() * 10 });
  }

  // camp: fire ring, logs, two tents
  (function buildCamp() {
    const y = height(CAMP.x, CAMP.z);
    const ringStone = stoneMat(0x6f706a, false, 0.2);
    for (let k = 0; k < 9; k++) {
      const a = k / 9 * Math.PI * 2;
      solid(new THREE.IcosahedronGeometry(0.22, 0), ringStone, CAMP.x + Math.cos(a) * 0.75, y + 0.08, CAMP.z + Math.sin(a) * 0.75);
    }
    const logG = new THREE.CylinderGeometry(0.2, 0.2, 2, 10);
    const logMat = woodMat(0x6b4a30);
    [[2.3, 0.4, 0.2], [-1.2, 2.2, 1.9]].forEach(([dx, dz, rot]) => {
      const log = solid(logG, logMat, CAMP.x + dx, y + 0.2, CAMP.z + dz);
      log.rotation.set(Math.PI / 2, 0, rot);
      obstacles.push({ x: CAMP.x + dx, z: CAMP.z + dz, r: 0.7 });
    });
    addFire(CAMP.x, CAMP.z, 0);
    obstacles.push({ x: CAMP.x, z: CAMP.z, r: 0.95 });

    const tentG = new THREE.ConeGeometry(1.9, 2.1, 4); tentG.rotateY(Math.PI / 4);
    [[-4.2, -2.4, 0xc4ae82, 0.5], [-1.4, -4.8, 0x7d8c9a, 1.1]].forEach(([dx, dz, hex, rot]) => {
      const tx = CAMP.x + dx, tz = CAMP.z + dz;
      const tent = solid(tentG, canvasMat(hex), tx, height(tx, tz) + 1.0, tz);
      tent.rotation.y = rot;
      obstacles.push({ x: tx, z: tz, r: 1.5 });
    });
  })();

  // hilltop stone circle with a brazier
  (function buildCircle() {
    const standing = stoneMat(0x8e9188, false, 1);
    const count = 11;
    const gates = [Math.atan2(R1b.z - CIRCLE.z, R1b.x - CIRCLE.x), Math.atan2(R2a.z - CIRCLE.z, R2a.x - CIRCLE.x)];
    for (let k = 0; k < count; k++) {
      const a = k / count * Math.PI * 2 + 0.2;
      // leave a gateway where each ramp meets the ring
      if (gates.some((g) => Math.abs(Math.atan2(Math.sin(a - g), Math.cos(a - g))) < 0.36)) continue;
      const x = CIRCLE.x + Math.cos(a) * 5.4, z = CIRCLE.z + Math.sin(a) * 5.4;
      const tall = 2.4 + rand() * 1.4;
      const fallen = k === 5;
      const m = solid(new THREE.BoxGeometry(0.95, tall, 0.55), standing, x, height(x, z) + (fallen ? 0.25 : tall / 2 - 0.2), z);
      m.rotation.y = -a + Math.PI / 2 + (rand() - 0.5) * 0.3;
      if (fallen) m.rotation.z = Math.PI / 2 - 0.08; else { m.rotation.x = (rand() - 0.5) * 0.14; m.rotation.z = (rand() - 0.5) * 0.14; }
      obstacles.push({ x, z, r: fallen ? 1.1 : 0.6 });
    }
    const y = height(CIRCLE.x, CIRCLE.z);
    solid(new THREE.CylinderGeometry(1.1, 1.3, 0.6, 12), stoneMat(0x8e9188, true, 0.5), CIRCLE.x, y + 0.3, CIRCLE.z);
    solid(new THREE.CylinderGeometry(0.5, 0.3, 0.35, 12), stoneMat(0x3b3a36, false, 0), CIRCLE.x, y + 0.75, CIRCLE.z);
    addFire(CIRCLE.x, CIRCLE.z, 0.85);
    obstacles.push({ x: CIRCLE.x, z: CIRCLE.z, r: 1.5 });
  })();

  // second tier: a watchtower with a beacon
  (function buildLookout() {
    const walls = stoneMat(0x9a9a90, true, 0.6);
    const y = height(LOOKOUT.x, LOOKOUT.z);
    solid(new THREE.CylinderGeometry(1.5, 1.85, 4.6, 20), walls, LOOKOUT.x, y + 2.2, LOOKOUT.z);
    solid(new THREE.CylinderGeometry(1.95, 1.95, 0.5, 20), walls, LOOKOUT.x, y + 4.7, LOOKOUT.z);
    addFire(LOOKOUT.x, LOOKOUT.z, 4.95);
    obstacles.push({ x: LOOKOUT.x, z: LOOKOUT.z, r: 2.1 });
  })();

  // ---------- Navigation: a walkability grid and A* ----------
  // Cliffs, water and scenery are blocked, so the only way up a tier is its ramp.
  const NAV = LIMIT * 2 + 1, NAV2 = NAV * NAV;
  const walk = new Uint8Array(NAV2);
  for (let iz = 0; iz < NAV; iz++) for (let ix = 0; ix < NAV; ix++) {
    const x = ix - LIMIT, z = iz - LIMIT;
    walk[iz * NAV + ix] = height(x, z) > WATER + 0.2 && !tooSteep(x, z) ? 1 : 0;
  }
  for (const o of obstacles) {
    const need = o.r + 0.4, reach = Math.ceil(need);
    const cx = Math.round(o.x) + LIMIT, cz = Math.round(o.z) + LIMIT;
    for (let iz = cz - reach; iz <= cz + reach; iz++) for (let ix = cx - reach; ix <= cx + reach; ix++) {
      if (ix < 0 || iz < 0 || ix >= NAV || iz >= NAV) continue;
      if (Math.hypot(ix - LIMIT - o.x, iz - LIMIT - o.z) < need) walk[iz * NAV + ix] = 0;
    }
  }
  const isWalk = (ix, iz) => ix >= 0 && iz >= 0 && ix < NAV && iz < NAV && walk[iz * NAV + ix] === 1;
  // true when the whole grid cell around a point is walkable
  function openAt(x, z) {
    const ix = Math.floor(x + LIMIT), iz = Math.floor(z + LIMIT);
    return isWalk(ix, iz) && isWalk(ix + 1, iz) && isWalk(ix, iz + 1) && isWalk(ix + 1, iz + 1);
  }
  function nearestNode(x, z, reach) {
    const cx = Math.round(x) + LIMIT, cz = Math.round(z) + LIMIT;
    let best = -1, bestD = Infinity;
    for (let iz = cz - reach; iz <= cz + reach; iz++) for (let ix = cx - reach; ix <= cx + reach; ix++) {
      if (!isWalk(ix, iz)) continue;
      const d = Math.hypot(ix - LIMIT - x, iz - LIMIT - z);
      if (d < bestD) { bestD = d; best = iz * NAV + ix; }
    }
    return best;
  }
  function sightLine(a, b) {
    const d = Math.hypot(b.x - a.x, b.z - a.z), n = Math.ceil(d / 0.35);
    for (let k = 1; k < n; k++) { const u = k / n; if (!openAt(a.x + (b.x - a.x) * u, a.z + (b.z - a.z) * u)) return false; }
    return true;
  }

  const gCost = new Float32Array(NAV2), fCost = new Float32Array(NAV2), cameFrom = new Int32Array(NAV2), done = new Uint8Array(NAV2);
  const STEPS = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, 1.4142], [1, -1, 1.4142], [-1, 1, 1.4142], [-1, -1, 1.4142]];
  function findPath(start, goal) {
    gCost.fill(Infinity); done.fill(0);
    const gx = goal % NAV, gz = (goal / NAV) | 0;
    const guess = (id) => { const dx = Math.abs(id % NAV - gx), dz = Math.abs(((id / NAV) | 0) - gz); return Math.max(dx, dz) + 0.4142 * Math.min(dx, dz); };
    const heap = [];
    const push = (id) => {
      heap.push(id);
      let i = heap.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (fCost[heap[p]] <= fCost[heap[i]]) break; const t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p; }
    };
    const pop = () => {
      const top = heap[0], end = heap.pop();
      if (heap.length) {
        heap[0] = end;
        let i = 0;
        for (;;) {
          const l = i * 2 + 1, r = l + 1; let m = i;
          if (l < heap.length && fCost[heap[l]] < fCost[heap[m]]) m = l;
          if (r < heap.length && fCost[heap[r]] < fCost[heap[m]]) m = r;
          if (m === i) break;
          const t = heap[m]; heap[m] = heap[i]; heap[i] = t; i = m;
        }
      }
      return top;
    };
    gCost[start] = 0; fCost[start] = guess(start); cameFrom[start] = -1; push(start);
    while (heap.length) {
      const cur = pop();
      if (cur === goal) {
        const nodes = [];
        for (let id = cur; id !== -1; id = cameFrom[id]) nodes.push({ x: id % NAV - LIMIT, z: ((id / NAV) | 0) - LIMIT });
        return nodes.reverse();
      }
      if (done[cur]) continue;
      done[cur] = 1;
      const cx = cur % NAV, cz = (cur / NAV) | 0, ch = height(cx - LIMIT, cz - LIMIT);
      for (const [dx, dz, cost] of STEPS) {
        const nx = cx + dx, nz = cz + dz;
        if (!isWalk(nx, nz)) continue;
        if (dx && dz && !(isWalk(cx + dx, cz) && isWalk(cx, cz + dz))) continue;
        const id = nz * NAV + nx;
        if (done[id]) continue;
        const g = gCost[cur] + cost + Math.abs(height(nx - LIMIT, nz - LIMIT) - ch) * 0.6;
        if (g < gCost[id]) { gCost[id] = g; fCost[id] = g + guess(id); cameFrom[id] = cur; push(id); }
      }
    }
    return null;
  }
  // Route from a point to a point: A* over the grid, then straighten wherever there is a clear line.
  function route(from, to) {
    const start = nearestNode(from.x, from.z, 3), goal = nearestNode(to.x, to.z, 4);
    if (start < 0 || goal < 0) return null;
    const nodes = findPath(start, goal);
    if (!nodes) return null;
    if (openAt(to.x, to.z) && !tooSteep(to.x, to.z)) nodes.push({ x: to.x, z: to.z });
    const pts = [];
    let i = 0;
    while (i < nodes.length - 1) {
      let j = Math.min(nodes.length - 1, i + 60);
      while (j > i + 1 && !sightLine(nodes[i], nodes[j])) j--;
      pts.push(nodes[j]);
      i = j;
    }
    if (!pts.length) pts.push(nodes[nodes.length - 1]);
    return pts;
  }

  // ---------- Party ----------
  const cardGeo = new THREE.PlaneGeometry(1, 1); cardGeo.translate(0, 0.5, 0);
  // Characters modelled in 3D (blender/<name>/export_glb.py) walk the world as the model itself rather than a
  // card. The model is in sheet units (a frame is 80 tall), scaled to the card's height in the world (and a
  // fifth more, as the cards lean toward the camera and so look taller than they are). Its one
  // timeline holds a walk cycle (frames 1-25, at 24 fps) and the idle stance (frame 40): the walk's phase
  // follows the hero's stride, so the feet keep pace with the ground they cover.
  const MODEL_SCALE = 2.2 / 80 * 1.2, WALK_SECONDS = 24 / 24, IDLE_SECONDS = 39 / 24;
  const gltf = new GLTFLoader();
  function loadModel(file, g) {
    const anim = { mixer: null, actions: [] };
    gltf.load(import.meta.env.BASE_URL + file, (res) => {
      const m = res.scene;
      m.scale.setScalar(MODEL_SCALE);
      m.traverse((o) => {
        if (!o.isMesh) return;
        o.castShadow = true;
        o.receiveShadow = true;
        o.frustumCulled = false;      // (a skinned mesh's bounds are its rest pose)
      });
      g.add(m);
      anim.mixer = new THREE.AnimationMixer(m);
      anim.actions = res.animations.map((clip) => {
        const a = anim.mixer.clipAction(clip);
        a.setLoop(THREE.LoopOnce);
        a.clampWhenFinished = true;
        a.play();
        return a;
      });
      anim.mixer.setTime(IDLE_SECONDS);
    }, undefined, (err) => console.warn('model failed to load:', file, err));
    return anim;
  }
  function poseModel(h) {
    if (!h.anim || !h.anim.mixer) return;
    const t = h.moving ? ((h.stride % (Math.PI * 2)) / (Math.PI * 2)) * WALK_SECONDS : IDLE_SECONDS;
    h.anim.mixer.setTime(t);
  }
  function makeHero(key, spec, ringColour) {
    const g = new THREE.Group();
    // the sheet is a pre-painted image (scripts/build-sprites.mjs); until it arrives the texture is empty and
    // the card draws nothing. (An image texture is flipped on upload, so its top row is the top of the card.)
    const tex = new THREE.Texture();
    tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.LinearFilter; tex.generateMipmaps = false;
    if (!ROSTER.find((r) => r.key === key).model) loadSheet(key, spec.hairStyle || null, spec).then((img) => { tex.image = img; tex.needsUpdate = true; });
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.5, 64), blobShadowMat);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = 0.05;
    // an ellipse, wider across than deep, just inside the selection ring; drawn after the see-through grass
    shadow.scale.set(0.9, 0.66, 1);
    shadow.renderOrder = 2;
    g.add(shadow);
    const model = ROSTER.find((r) => r.key === key).model;
    let material = null, anim = null;
    if (model) anim = loadModel(model, g);
    else {
      material = spriteMat(tex);
      const card = new THREE.Mesh(cardGeo, material);
      card.frustumCulled = false;
      card.receiveShadow = true;
      g.add(card);
    }
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.4, 0.62, 48), ringMat(ringColour, 0.44, 0.58, false));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.07;
    ring.renderOrder = 2;
    g.add(ring);
    scene.add(g);
    return { g, anim, x: 0, z: 0, moving: false, stride: rand() * 6, breath: Math.floor(rand() * 2), frame: material ? material.uniforms.uFrame.value : new THREE.Vector3(), facing: 0 };
  }

  // the party picked on the sprite bench (the first member leads), each in their chosen hairstyle
  const party = loadParty();
  const leader = makeHero(party.members[0], memberSpec(party.members[0], party), 0x7dff7a);
  const followers = party.members.slice(1).map((k) => makeHero(k, memberSpec(k, party), 0xd8e6d2));
  leader.x = 2.6; leader.z = 2.4;
  const START = [[3.4, 4.0], [1.2, 4.2], [2.4, 5.4]];
  followers.forEach((f, k) => { [f.x, f.z] = START[k % START.length]; });
  [leader].concat(followers).forEach((h) => h.g.position.set(h.x, height(h.x, h.z), h.z));
  leader.g.rotation.y = Math.PI * 1.15;

  let trail = [{ x: leader.x, z: leader.z }];
  let order = null, stuck = 0;

  const marker = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.74, 64), ringMat(0xffe08a, 0.46, 0.7, true));
  marker.rotation.x = -Math.PI / 2;
  marker.renderOrder = 2;
  marker.visible = false;
  scene.add(marker);

  // Move one hero toward (tx, tz); slide around obstacles. Returns distance actually covered.
  function stepHero(h, tx, tz, speed, dt) {
    const dx = tx - h.x, dz = tz - h.z, dist = Math.hypot(dx, dz);
    if (dist < 0.02) return 0;
    const step = Math.min(speed * dt, dist);
    let nx = h.x + dx / dist * step, nz = h.z + dz / dist * step;
    for (const o of obstacles) {
      const ox = nx - o.x, oz = nz - o.z, need = o.r + 0.32;
      if (Math.abs(ox) > need || Math.abs(oz) > need) continue;
      const d = Math.hypot(ox, oz);
      if (d < need && d > 0.0001) { nx = o.x + ox / d * need; nz = o.z + oz / d * need; }
    }
    if (height(nx, nz) < WATER + 0.15) return -1;   // water ahead
    const moved = Math.hypot(nx - h.x, nz - h.z);
    if (moved > 0.0005) {
      const want = Math.atan2(nx - h.x, nz - h.z);
      let diff = want - h.g.rotation.y;
      diff = Math.atan2(Math.sin(diff), Math.cos(diff));
      h.g.rotation.y += diff * Math.min(1, dt * 12);
    }
    h.x = nx; h.z = nz;
    return moved;
  }

  function updateParty(dt, now) {
    leader.moving = false;
    if (order) {
      const wp = order.pts[order.i], lastLeg = order.i === order.pts.length - 1;
      const intended = Math.min(5.6 * dt, Math.hypot(wp.x - leader.x, wp.z - leader.z));
      const moved = stepHero(leader, wp.x, wp.z, 5.6, dt);
      if (moved < 0) { order = null; say('The water is too deep to cross here.'); }
      else {
        leader.moving = true;
        stuck = moved < intended * 0.25 ? stuck + dt : 0;
        if (Math.hypot(wp.x - leader.x, wp.z - leader.z) < (lastLeg ? 0.12 : 0.4)) {
          if (lastLeg) { order = null; say('The party has arrived.'); } else order.i++;
        } else if (stuck > 0.7) { order = null; say('Something is in the way. Try a spot to one side.'); }
      }
      if (!order) { marker.visible = false; stuck = 0; }
      const last = trail[trail.length - 1];
      if (Math.hypot(leader.x - last.x, leader.z - last.z) > 0.55) {
        trail.push({ x: leader.x, z: leader.z });
        if (trail.length > 14) trail.shift();
      }
    }
    followers.forEach((f, k) => {
      const idx = trail.length - 1 - 3 * (k + 1);
      if (idx < 0) { f.moving = false; return; }
      const spot = trail[idx];
      const gap = Math.hypot(spot.x - f.x, spot.z - f.z);
      f.moving = gap > 0.2 && stepHero(f, spot.x, spot.z, gap > 3 ? 6.6 : 5.4, dt) > 0.0005;
    });
    [leader].concat(followers).forEach((h) => {
      if (h.moving) h.stride += dt * 14;
      h.g.position.set(h.x, height(h.x, h.z), h.z);
      // pick one of eight facings from the heading relative to the camera; a margin keeps it from flickering
      let rel = h.g.rotation.y - cam.yaw;
      rel = Math.atan2(Math.sin(rel), Math.cos(rel));
      let off = rel - h.facing * Math.PI / 4;
      off = Math.atan2(Math.sin(off), Math.cos(off));
      if (Math.abs(off) > Math.PI / 8 + 0.07) h.facing = ((Math.round(rel / (Math.PI / 4)) % 8) + 8) % 8;
      const step = Math.floor((h.stride % (Math.PI * 2)) / (Math.PI * 2 / WALK));
      const idle = Math.floor(now / 650 + h.breath) % 2;
      h.frame.set(h.moving ? 2 + step : idle, h.facing, 0);
      poseModel(h);
    });
    if (marker.visible && !reduceMotion) marker.scale.setScalar(1 + 0.12 * Math.sin(now / 160));
  }

  // ---------- Party camera ----------
  // Orbits a focus point on the ground. Zoom sets both distance and tilt:
  // close in the lens drops toward the horizon, far out it looks almost straight down.
  const MIN_DIST = 7, MAX_DIST = 62;
  const cam = { yaw: -0.97, yawGoal: -0.97, dist: 34, distGoal: 34, pitch: 0.8, focus: new THREE.Vector3(leader.x, height(leader.x, leader.z), leader.z), follow: true };
  const followBtn = document.getElementById('follow');
  function setFollow(on) {
    cam.follow = on;
    followBtn.setAttribute('aria-pressed', String(on));
  }

  function updateCamera(dt, instant) {
    // held keys
    const turn = (keys.has('KeyQ') ? 1 : 0) - (keys.has('KeyE') ? 1 : 0);
    if (turn) cam.yawGoal += turn * 1.9 * dt;
    const px = (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0) - (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0);
    const pz = (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0) - (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0);
    if (px || pz) {
      if (cam.follow) { setFollow(false); say('Camera is free. Press Space to return to the party.'); }
      const speed = cam.dist * 0.9 * dt, s = Math.sin(cam.yaw), c = Math.cos(cam.yaw);
      cam.focus.x = clamp(cam.focus.x + (px * c + pz * s) * speed, -LIMIT, LIMIT);
      cam.focus.z = clamp(cam.focus.z + (-px * s + pz * c) * speed, -LIMIT, LIMIT);
    }

    const ease = instant || reduceMotion ? 1 : 1 - Math.exp(-dt * 8);
    cam.yaw += (cam.yawGoal - cam.yaw) * ease;
    cam.dist += (cam.distGoal - cam.dist) * ease;
    if (cam.follow) {
      const k = instant ? 1 : 1 - Math.exp(-dt * 5);
      cam.focus.x += (leader.x - cam.focus.x) * k;
      cam.focus.z += (leader.z - cam.focus.z) * k;
    }
    const groundY = Math.max(height(cam.focus.x, cam.focus.z), WATER) + 0.9;
    cam.focus.y += (groundY - cam.focus.y) * (instant ? 1 : 1 - Math.exp(-dt * 6));

    const zoom = (cam.dist - MIN_DIST) / (MAX_DIST - MIN_DIST);
    let want = THREE.MathUtils.degToRad(lerp(27, 66, Math.pow(zoom, 0.7)));
    // If a cliff or hill sits between the lens and the party, tilt up until the view is clear.
    const sy = Math.sin(cam.yaw), cy = Math.cos(cam.yaw);
    const blocked = (p) => {
      const fl = Math.cos(p) * cam.dist, up = Math.sin(p) * cam.dist;
      for (let k = 1; k <= 12; k++) {
        const u = k / 12;
        if (Math.max(height(cam.focus.x + sy * fl * u, cam.focus.z + cy * fl * u), WATER) + 0.5 > cam.focus.y + up * u) return true;
      }
      return false;
    };
    while (want < 1.42 && blocked(want)) want += 0.04;
    cam.pitch += (want - cam.pitch) * (instant || reduceMotion ? 1 : 1 - Math.exp(-dt * 5));
    const pitch = cam.pitch;
    const flat = Math.cos(pitch) * cam.dist;
    camera.position.set(
      cam.focus.x + sy * flat,
      cam.focus.y + Math.sin(pitch) * cam.dist,
      cam.focus.z + cy * flat
    );
    const floor = Math.max(height(camera.position.x, camera.position.z), WATER) + 1.3;
    if (camera.position.y < floor) camera.position.y = floor;
    camera.lookAt(cam.focus);

    scene.fog.near = cam.dist + 22;
    scene.fog.far = cam.dist + 125;
  }

  // ---------- Light and time of day (follows the page theme) ----------
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera;
  sc.left = -42; sc.right = 42; sc.top = 42; sc.bottom = -42; sc.near = 1; sc.far = 180;
  sun.shadow.bias = -0.0008;
  scene.add(sun); scene.add(sun.target);
  const sunDir = new THREE.Vector3();
  let fireBase = 0.7;

  function isDark() {
    const t = document.documentElement.getAttribute('data-theme');
    return t ? t === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function applyMood() {
    const night = isDark();
    const horizon = night ? 0x1b2a3a : 0xcfe2e6;
    scene.fog.color.setHex(horizon);
    SHARED.uHorizon.value.setHex(horizon);
    SHARED.uSkyTop.value.setHex(night ? 0x03070f : 0x5f98c4);
    SHARED.uNight.value = night ? 1 : 0;
    SHARED.uExposure.value = night ? 0.9 : 0.62;
    hemi.color.setHex(night ? 0x4a64a0 : 0xd8ecff);
    hemi.groundColor.setHex(night ? 0x0e1610 : 0x5b6a40);
    hemi.intensity = night ? 0.16 : 0.62;
    sun.color.setHex(night ? 0x9db8ff : 0xffe4b5);
    sun.intensity = night ? 0.28 : 1.05;
    sunDir.set(night ? 0.45 : -0.5, night ? 0.8 : 0.72, night ? -0.4 : 0.48).normalize();
    SHARED.uSunDir.value.copy(sunDir);
    SHARED.uSunCol.value.copy(sun.color).multiplyScalar(sun.intensity);
    // The light a sprite gets in the open (its normal is straight up, its light wrapped all the way round, as
    // in the sprite shader): the sky from above plus the sun. By day sprites are lit relative to exactly this,
    // so in the open they show their painted colours; at night relative to a neutral grey brighter than the
    // moonlight, so they dim to about NIGHT_SPRITE of their painted brightness and keep the moonlight's blue.
    const open = new THREE.Vector3(hemi.color.r, hemi.color.g, hemi.color.b).multiplyScalar(hemi.intensity)
      .add(new THREE.Vector3(sun.color.r, sun.color.g, sun.color.b).multiplyScalar(sun.intensity * (sunDir.y + 1) / 2));
    const NIGHT_SPRITE = 0.5, lum = 0.2126 * open.x + 0.7152 * open.y + 0.0722 * open.z;
    if (night) SHARED.uSpriteRef.value.setScalar(lum / NIGHT_SPRITE);
    else SHARED.uSpriteRef.value.copy(open);
    fireBase = night ? 1.8 : 0.6;
  }
  applyMood();
  new MutationObserver(applyMood).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyMood);

  // ---------- Picking the ground ----------
  const caster = new THREE.Raycaster();
  function pickGround(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    caster.setFromCamera({ x: (clientX - r.left) / r.width * 2 - 1, y: -(clientY - r.top) / r.height * 2 + 1 }, camera);
    const o = caster.ray.origin, d = caster.ray.direction;
    const below = (t) => o.y + d.y * t < Math.max(height(o.x + d.x * t, o.z + d.z * t), WATER);
    let prev = 0;
    for (let t = 0.6; t < 420; t += 0.6) {
      if (below(t)) {
        let a = prev, b = t;
        for (let k = 0; k < 14; k++) { const m = (a + b) / 2; if (below(m)) b = m; else a = m; }
        return { x: o.x + d.x * b, z: o.z + d.z * b };
      }
      prev = t;
    }
    return null;
  }

  function giveOrder(clientX, clientY) {
    const p = pickGround(clientX, clientY);
    if (!p) return;
    if (Math.abs(p.x) > LIMIT || Math.abs(p.z) > LIMIT) { say('That is past the edge of the map.'); return; }
    if (height(p.x, p.z) < WATER + 0.15) { say('That is open water. Pick dry ground.'); return; }
    if (nearestNode(p.x, p.z, 4) < 0) { say('That cliff is too steep to stand on.'); return; }
    const pts = route(leader, p);
    if (!pts) { say('The party can’t find a way there on foot.'); return; }
    const end = pts[pts.length - 1], endY = height(end.x, end.z), rise = endY - height(leader.x, leader.z);
    order = { pts, i: 0 }; stuck = 0;
    marker.position.set(end.x, endY + 0.09, end.z);
    marker.scale.setScalar(1);
    marker.visible = true;
    say(rise > 2.5 ? 'The party climbs toward higher ground.' : rise < -2.5 ? 'The party heads downhill.' : 'The party is on the move.');
  }

  // ---------- Input ----------
  const keys = new Set();
  const pointers = new Map();
  let dragged = false, multi = false, pinchStart = 0, pinchDist = 0;

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, button: e.button });
    if (pointers.size === 1) { dragged = false; multi = false; }
    if (pointers.size === 2) {
      multi = true;
      const [a, b] = [...pointers.values()];
      pinchStart = Math.hypot(a.x - b.x, a.y - b.y); pinchDist = cam.distGoal;
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x;
    p.x = e.clientX; p.y = e.clientY;
    if (pointers.size === 1) {
      if (Math.hypot(p.x - p.sx, p.y - p.sy) > 7) dragged = true;
      if (dragged) cam.yawGoal -= dx * 0.0065;
    } else if (pointers.size === 2 && pinchStart > 0) {
      const [a, b] = [...pointers.values()];
      cam.distGoal = clamp(pinchDist * pinchStart / Math.max(20, Math.hypot(a.x - b.x, a.y - b.y)), MIN_DIST, MAX_DIST);
    }
  });
  function release(e) {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    pointers.delete(e.pointerId);
    if (e.type === 'pointerup' && !dragged && !multi && p.button === 0) giveOrder(e.clientX, e.clientY);
  }
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.distGoal = clamp(cam.distGoal * Math.exp(e.deltaY * 0.0012), MIN_DIST, MAX_DIST);
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const onButton = e.target instanceof HTMLElement && (e.target.tagName === 'BUTTON' || e.target.tagName === 'SUMMARY');
    if (e.code === 'Space' || e.code === 'Home') {
      if (onButton && e.code === 'Space') return;
      e.preventDefault(); setFollow(true); say('Camera is following the party.');
      return;
    }
    if (e.code === 'Equal' || e.code === 'NumpadAdd') cam.distGoal = clamp(cam.distGoal * 0.8, MIN_DIST, MAX_DIST);
    if (e.code === 'Minus' || e.code === 'NumpadSubtract') cam.distGoal = clamp(cam.distGoal * 1.25, MIN_DIST, MAX_DIST);
    if (e.code.startsWith('Arrow')) e.preventDefault();
    keys.add(e.code);
  });
  window.addEventListener('keyup', (e) => keys.delete(e.code));
  window.addEventListener('blur', () => keys.clear());

  document.getElementById('rotL').addEventListener('click', () => { cam.yawGoal -= Math.PI / 4; });
  document.getElementById('rotR').addEventListener('click', () => { cam.yawGoal += Math.PI / 4; });
  document.getElementById('zoomIn').addEventListener('click', () => { cam.distGoal = clamp(cam.distGoal * 0.7, MIN_DIST, MAX_DIST); });
  document.getElementById('zoomOut').addEventListener('click', () => { cam.distGoal = clamp(cam.distGoal * 1.4, MIN_DIST, MAX_DIST); });
  followBtn.addEventListener('click', () => {
    setFollow(!cam.follow);
    say(cam.follow ? 'Camera is following the party.' : 'Camera is free. Pan with W A S D.');
  });

  // ---------- Loop ----------
  const altEl = document.getElementById('alt');
  let last = performance.now(), tick = 0;
  updateCamera(0, true);

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.08);
    last = now;

    updateParty(dt, now);
    updateCamera(dt, false);

    // the shadow-casting light travels with the camera focus
    sun.position.copy(cam.focus).addScaledVector(sunDir, 80);
    sun.target.position.copy(cam.focus);

    for (const f of fires) {
      const flick = reduceMotion ? 1 : 0.82 + 0.18 * Math.sin(now / 90 + f.phase) * Math.sin(now / 37 + f.phase * 2) + 0.08 * Math.sin(now / 13);
      f.light.intensity = fireBase * flick;
      f.mats.forEach((m) => { m.uniforms.uPower.value = 0.75 + 0.3 * flick; });
    }
    SHARED.uTime.value = reduceMotion ? 12 : now / 1000;
    sky.position.copy(camera.position);

    if ((tick++ & 7) === 0) {
      const alt = Math.max(0, Math.round(height(leader.x, leader.z) - WATER)) + ' m';
      if (altEl.textContent !== alt) altEl.textContent = alt;
    }

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
main();
