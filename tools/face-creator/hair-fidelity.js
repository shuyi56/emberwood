// Hair fidelity: show a hairstyle as the creator sculpts it, or as the physics groom (tools/hair) baked for the game:
// cards at three levels of detail, or the strands themselves. Added to the page by apply-hair-fidelity.py, with each
// groomed style's asset (tools/hair/creator_asset.py) in a <script type="application/json" id="groom-asset-STYLE">.
import * as THREE from 'three';

const HH = 8.6;                                   // the creator's units per head height (rq_hair_export.cjs)
const MODES = [['creator', 'Creator (sculpted locks)'], ['lod2', 'Groom: cards, LOD 2'], ['lod1', 'Groom: cards, LOD 1'],
  ['lod0', 'Groom: cards, LOD 0'], ['strands', 'Groom: strands']];
const LOD_INDEX = { lod0: 0, lod1: 1, lod2: 2 };
const SCALP_SHADE = 0.4;                          // STYLE: the scalp cap against the hair colour: the shade map's root (0.5) in
                                                  // the shadow of the hair over it (render.py: 0.35)

function bytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
// head heights in the hair's frame (x across, y back, z up) to the page's units (x, y up, z toward the viewer)
function toPage(src, scale) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) { out[i] = src[i] * scale * HH; out[i + 1] = src[i + 2] * scale * HH; out[i + 2] = -src[i + 1] * scale * HH; }
  return out;
}

// How the cards look: the creator's own hair surface, so the groom reads as the same hair. The strand lines, the dark
// partings between locks, the single light-catching strands and the rounded relief of each lock all come from the
// creator's hair texture and bump map (hairField and paintHair in the creator; hairCapMat.map and .bumpMap here),
// mapped onto each card where the card lies round the head, so neighbouring cards carry on the creator's locks.
// The cut-out (each lock of the card ending at its own length, the soft root) is this file's own map, on uv1.
// The shade map: one value along every hair, multiplied into its colour, shared by the cards, the strands and the
// scalp. It is the game hair "root" map (UE4 Photorealistic Character: "a mask which is used to change the color of
// the hair from the root to the tip"; inZOI's DIRO map: "Hair root color and gradient starting point"), with the
// per-strand ID (each strand a little lighter or darker) and the baked AO (creator_asset.py) beside it.
// Dark at the root: the root lies under all the hair over it, so it gets the least light (self-shadowing: Yuksel &
// Keyser, Deep Opacity Maps; Scheuermann 2004 stores it as per-vertex AO). Lighter toward the ends: scalp hair grows
// about 1 cm a month (Wikipedia, Human hair growth), so the end of a 45 cm length is nearly four years old, and its
// colour has been changed by that much sunlight, mainly UVA (Nogueira & Joekes 2004, via Nogueira et al. 2006), light
// hair more than dark (Signori 2004, same review). No measured root-to-tip lightness was found: the amounts are STYLE.
const ROOT_DARK = 0.5;                            // STYLE: the root against the mid-length
const ROOT_CM = 4;                                // STYLE: how far from the root the darkening reaches
const TIP_LIGHT = 1.35;                           // STYLE: the very end against the mid-length
const TIP_FRAC = 0.45;                            // STYLE: the share of each hair's length that lightens toward its end
const sstep = x => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
// s: distance from the root (page units); t: the share of the hair's length (0 root, 1 end)
const shadeAt = (s, t) => (ROOT_DARK + (1 - ROOT_DARK) * sstep(s / (ROOT_CM * 10 * HH / HEAD_MM))) *
  (1 + (TIP_LIGHT - 1) * sstep((t - (1 - TIP_FRAC)) / TIP_FRAC));
const CLUMPS = 6;                                 // STYLE: locks across one card in the cut-out
const ROOT_FADE = 0.05;                           // STYLE: the cut-out fades in over this much of the card at the root
const TEX_V_HH = 1.6;                             // the creator's hair texture runs this many head heights down the hair
const HAIR_SHADE = 0.8;                           // the creator's own (HAIR_SHADE): its hair is drawn at 0.8 of its paint
const CENTRE = [0, 0.56 * HH, -0.06 * HH];        // the skull's centre in the page's units (scalp_field.CENTRE)
const HAIR_MM = 0.07;                             // SOURCED: a scalp hair is 50-100 um across (Robbins, Chemical and Physical
                                                  // Behavior of Human Hair, ch. 9); the middle of that range
const HEAD_MM = 219;                              // one head height in mm (scalp_field.HH_MM)
const PX_WORLD = { value: 0.002 };                // one pixel's width at unit depth: set each frame from the camera
function cutTexture() {
  const W = 256, H = 512, c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const cw = W / CLUMPS;
  for (let j = 0; j < CLUMPS; j++) {
    const cx = (j + 0.5 + (rnd() - 0.5) * 0.3) * cw, len = H * (0.72 + 0.28 * rnd()), tipX = cx + (rnd() - 0.5) * cw * 0.4;
    const edge = Math.min(cx, W - cx) / (W * 0.5);              // the card's outer locks are thinner, so its edge is soft
    for (let k = 0; k < 90; k++) {
      const off = (rnd() - 0.5) * cw * 0.95, x0 = cx + off, end = len * (0.8 + 0.2 * rnd());
      const x1 = tipX + off * 0.25 * (end / len);                 // the lock narrows to its tip
      const centre = 1 - Math.abs(off) / (cw * 0.5);
      const a = Math.min(1, 0.45 + 0.9 * centre) * Math.min(1, 0.35 + 1.3 * edge) * (0.7 + 0.3 * rnd()), f = end / H;
      const grd = g.createLinearGradient(0, 0, 0, end);
      grd.addColorStop(0, 'rgba(255,255,255,0)'); grd.addColorStop(Math.min(0.5, ROOT_FADE / f), `rgba(255,255,255,${a})`);
      grd.addColorStop(0.92, `rgba(255,255,255,${a * 0.9})`); grd.addColorStop(1, 'rgba(255,255,255,0)');
      g.strokeStyle = grd; g.lineWidth = 1 + rnd() * 1.6; g.beginPath();
      g.moveTo(x0, 0); g.quadraticCurveTo(x0 + (rnd() - 0.5) * 4, end * 0.55, x1, end); g.stroke();
    }
  }
  const t = new THREE.CanvasTexture(c); t.flipY = false;          // canvas row y is v = y / H: the root at the top
  t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping; t.anisotropy = 8; t.channel = 1;
  return t;
}

// Per card vertex: the creator's texture coordinates (round the head from the card's root, down the hair by length),
// the shading normal (out from the head, as the creator's head of hair is lit: a flat card's own normal turned its
// underside black wherever it lifted off the head) and the colour (the baked light through the hair, darker roots).
function cardAttributes(P, UV, AO) {
  const n = P.length / 3, uv0 = new Float32Array(n * 2), N = new Float32Array(n * 3), C = new Float32Array(n * 3);
  let start = 0;
  for (let i = 0; i <= n; i += 2) {
    if (i < n && !(i > start && UV[2 * i + 1] === 0)) continue;
    // a card is the rows from `start` to i (two vertices a row, v 0 at its root)
    const mx = (P[3 * start] + P[3 * start + 3]) / 2, mz = (P[3 * start + 2] + P[3 * start + 5]) / 2 - CENTRE[2];
    const u0 = Math.atan2(mx, mz) / (2 * Math.PI) + 0.5, r = Math.max(1, Math.hypot(mx, mz));
    const w0 = Math.hypot(P[3 * start] - P[3 * start + 3], P[3 * start + 1] - P[3 * start + 4], P[3 * start + 2] - P[3 * start + 5]);
    const du = w0 / (2 * Math.PI * r);
    let s = 0, px = 0, py = 0, pz = 0;
    for (let j = start; j < i; j += 2) {
      const cx = (P[3 * j] + P[3 * j + 3]) / 2, cy = (P[3 * j + 1] + P[3 * j + 4]) / 2, cz = (P[3 * j + 2] + P[3 * j + 5]) / 2;
      if (j > start) s += Math.hypot(cx - px, cy - py, cz - pz);
      px = cx; py = cy; pz = cz;
      for (const q of [j, j + 1]) {
        uv0[2 * q] = u0 + (UV[2 * q] - 0.5) * du; uv0[2 * q + 1] = Math.min(1, s / (TEX_V_HH * HH));
        const ox = P[3 * q] - CENTRE[0], oy = P[3 * q + 1] - Math.min(P[3 * q + 1], CENTRE[1]), oz = P[3 * q + 2] - CENTRE[2];
        const l = Math.hypot(ox, oy, oz) || 1; N[3 * q] = ox / l; N[3 * q + 1] = oy / l; N[3 * q + 2] = oz / l;
        C[3 * q] = C[3 * q + 1] = C[3 * q + 2] = (AO ? AO[q] : 1) * shadeAt(s, UV[2 * q + 1]);
      }
    }
    start = i;
  }
  return { uv0, N, C };
}

function buildStyle(asset, tone, creatorMat) {
  const group = new THREE.Group(), parts = {};
  const col = new THREE.Color(tone);
  const cardMat = new THREE.MeshStandardMaterial({ color: 0xffffff, map: creatorMat.map, bumpMap: creatorMat.bumpMap,
    bumpScale: creatorMat.bumpScale, alphaMap: cutTexture(), alphaTest: 0.35, side: THREE.DoubleSide, vertexColors: true,
    roughness: 1, metalness: 0, envMapIntensity: 0.25 });
  // the creator's shade, and no flipping of the normal on a card's far side: both sides are the hair's outer surface
  cardMat.onBeforeCompile = sh => {
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <color_fragment>', '#include <color_fragment>\n\tdiffuseColor.rgb *= ' + HAIR_SHADE.toFixed(3) + ';')
      .replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\n\tnormal = normalize( vNormal );');
  };
  // the scalp under the hair in the root colour, as render.py draws it (SCALP_SHADE), so no skin shows between hairs
  const capMat = new THREE.MeshStandardMaterial({ color: col.clone().multiplyScalar(SCALP_SHADE), roughness: 1, metalness: 0 });   // (fully rough, as UE's scalp: no stray highlight)
  asset.lods.forEach((L, i) => {
    const g = new THREE.BufferGeometry(), P = toPage(new Float32Array(bytes(L.pos)), 1), UV = new Float32Array(bytes(L.uv));
    const A = cardAttributes(P, UV, L.ao ? new Float32Array(bytes(L.ao)) : null);
    g.setAttribute('position', new THREE.BufferAttribute(P, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(A.uv0, 2));          // the creator's hair texture and relief
    g.setAttribute('uv1', new THREE.BufferAttribute(UV, 2));            // the card's own: the cut-out
    g.setAttribute('normal', new THREE.BufferAttribute(A.N, 3));
    g.setAttribute('color', new THREE.BufferAttribute(A.C, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(L.idx)), 1));
    const m = new THREE.Mesh(g, cardMat); m.castShadow = true; m.name = 'groom-cards-' + i;
    parts['lod' + i] = m; group.add(m);
  });
  const cg = new THREE.BufferGeometry();
  cg.setAttribute('position', new THREE.BufferAttribute(toPage(new Float32Array(bytes(asset.cap.pos)), 1), 3));
  cg.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(asset.cap.idx)), 1)); cg.computeVertexNormals();
  parts.cap = new THREE.Mesh(cg, capMat); group.add(parts.cap);
  // The strands: each drawn child stands for `hairs_per_strand` real hairs (about 9.6), so it is a ribbon that wide:
  // that many hairs of HAIR_MM side by side, turned to face the camera, and never under a pixel. (As one-pixel lines
  // they covered a ninth of what the hair covers, and seen close the skin and the ear showed through.) Lit like the
  // cards, with normals out from the head; a little lighter or darker strand to strand, dark at the root.
  const S = asset.strands, cnt = new Uint16Array(bytes(S.counts)), raw = new Int16Array(bytes(S.pos));
  const pts = new Float32Array(raw.length); for (let i = 0; i < raw.length; i++) pts[i] = raw[i] * S.q;
  const P = toPage(pts, 1), np = P.length / 3;
  const wHair = (S.hairs_per_strand || 1) * HAIR_MM * HH / HEAD_MM;
  const pos = new Float32Array(np * 6), tan = new Float32Array(np * 6), nrm = new Float32Array(np * 6);
  const side = new Float32Array(np * 2), wid = new Float32Array(np * 2), vc = new Float32Array(np * 6), idx = [];
  let k = 0, seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (const c of cnt) {
    const b = 0.8 + 0.4 * rnd();
    for (let j = 0, sj = 0; j < c; j++) {
      const q = k + j, a0 = k + Math.max(0, j - 1), a1 = k + Math.min(c - 1, j + 1), t = j / (c - 1);
      if (j) sj += Math.hypot(P[3 * q] - P[3 * q - 3], P[3 * q + 1] - P[3 * q - 2], P[3 * q + 2] - P[3 * q - 1]);
      let tx = P[3 * a1] - P[3 * a0], ty = P[3 * a1 + 1] - P[3 * a0 + 1], tz = P[3 * a1 + 2] - P[3 * a0 + 2];
      const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
      const ox = P[3 * q] - CENTRE[0], oy = P[3 * q + 1] - Math.min(P[3 * q + 1], CENTRE[1]), oz = P[3 * q + 2] - CENTRE[2];
      const ol = Math.hypot(ox, oy, oz) || 1;
      const shade = b * shadeAt(sj, t);
      for (let e = 0; e < 2; e++) {
        const v = 2 * q + e;
        pos.set(P.subarray(3 * q, 3 * q + 3), 3 * v); tan[3 * v] = tx; tan[3 * v + 1] = ty; tan[3 * v + 2] = tz;
        nrm[3 * v] = ox / ol; nrm[3 * v + 1] = oy / ol; nrm[3 * v + 2] = oz / ol;
        side[v] = e ? 1 : -1; wid[v] = wHair * (1 - 0.6 * Math.max(0, (t - 0.85) / 0.15));      // (thinning over the last 15%)
        vc[3 * v] = vc[3 * v + 1] = vc[3 * v + 2] = shade;
      }
      if (j < c - 1) idx.push(2 * q, 2 * q + 1, 2 * q + 3, 2 * q, 2 * q + 3, 2 * q + 2);
    }
    k += c;
  }
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); sg.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  sg.setAttribute('strandT', new THREE.BufferAttribute(tan, 3)); sg.setAttribute('strandSide', new THREE.BufferAttribute(side, 1));
  sg.setAttribute('strandW', new THREE.BufferAttribute(wid, 1)); sg.setAttribute('color', new THREE.BufferAttribute(vc, 3));
  sg.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
  sg.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, CENTRE[1], 0), 2 * HH);
  const strandMat = new THREE.MeshStandardMaterial({ color: col, vertexColors: true, side: THREE.DoubleSide, roughness: 0.8,
    metalness: 0, envMapIntensity: 0.25 });
  strandMat.onBeforeCompile = sh => {
    sh.uniforms.pxWorld = PX_WORLD;
    sh.vertexShader = 'attribute vec3 strandT;\nattribute float strandSide;\nattribute float strandW;\nuniform float pxWorld;\n' +
      sh.vertexShader.replace('#include <project_vertex>', [
        'vec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0 );',
        'vec3 tv = normalize( ( modelViewMatrix * vec4( strandT, 0.0 ) ).xyz );',
        'vec3 sd = normalize( cross( tv, normalize( -mvPosition.xyz ) ) );',
        'mvPosition.xyz += sd * ( 0.5 * max( strandW, -mvPosition.z * pxWorld ) * strandSide );',
        'gl_Position = projectionMatrix * mvPosition;'].join('\n'));
    sh.fragmentShader = sh.fragmentShader.replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\n\tnormal = normalize( vNormal );');
  };
  parts.strands = new THREE.Mesh(sg, strandMat); parts.strands.frustumCulled = false; parts.strands.castShadow = false;
  group.add(parts.strands);
  const segs = idx.length / 6;
  return { group, parts, cardMat, capMat, strandMat, segs, tone };
}

function start(fc) {
  const assets = {};
  document.querySelectorAll('script[id^="groom-asset-"]').forEach(s => { assets[s.id.slice(12)] = s; });
  const scene = fc.hairMesh.parent, built = {};
  let mode = 'creator';
  const ui = document.createElement('label'); ui.className = 'check'; ui.appendChild(document.createTextNode('Fidelity '));
  const sel = document.createElement('select'); sel.id = 'sel-hair-fidelity'; sel.setAttribute('aria-label', 'Hair fidelity');
  MODES.forEach(([v, t]) => { const o = document.createElement('option'); o.value = v; o.textContent = t; sel.appendChild(o); });
  sel.addEventListener('change', () => { mode = sel.value; });
  ui.appendChild(sel);
  const note = document.createElement('div'); note.style.cssText = 'font-size:11px;opacity:.7;margin:2px 0 6px';
  const place = () => { const h = document.getElementById('sel-hair'); if (h && !ui.isConnected) { h.parentElement.after(ui); ui.after(note); } };
  // the creator's hair: the head of hair, its locks, and the sheer brown veils it lays over the scalp between strands
  // (hairVeil: vertex-coloured, custom-blended); a veil left showing under the groom read as a flat red-brown patch
  const isVeil = mat => mat && mat.isMeshBasicMaterial && mat.vertexColors && mat.blending === THREE.CustomBlending;
  const isCreatorHair = m => m.isMesh && (m === fc.hairMesh || m.material === fc.hairCapMat || isVeil(m.material) ||
    (m.material && fc.hairCapMat.map && m.material.map === fc.hairCapMat.map));
  const asset_n = el => (el._n = el._n || JSON.parse(el.textContent).strands.n).toLocaleString();
  function frame() {
    place();
    const el = fc.renderer.domElement;
    PX_WORLD.value = 2 * Math.tan(fc.camera.fov * Math.PI / 360) / Math.max(1, el.clientHeight);
    const style = fc.P.hairStyle, have = !!assets[style];
    [...sel.options].forEach(o => { o.disabled = o.value !== 'creator' && !have; });
    if (!have && mode !== 'creator') { mode = 'creator'; sel.value = 'creator'; }
    note.textContent = have ? '' : 'No groom for this style yet (groomed: ' + Object.keys(assets).join(', ') + ')';
    const groom = mode !== 'creator' && fc.P.hair;
    let changed = false;
    const show = (m, v) => { if (m.visible !== v) { m.visible = v; changed = true; } };
    scene.children.forEach(m => { if (isCreatorHair(m)) show(m, fc.P.hair && !groom); });
    Object.values(built).forEach(b => { if (!(groom && b === built[style])) show(b.group, false); });
    if (groom) {
      let b = built[style];
      if (!b) { b = built[style] = buildStyle(JSON.parse(assets[style].textContent), fc.P.hairTone, fc.hairCapMat); scene.add(b.group);
        note.textContent = ''; }
      if (b.tone !== fc.P.hairTone) { const c = new THREE.Color(fc.P.hairTone); b.capMat.color.copy(c).multiplyScalar(SCALP_SHADE); b.strandMat.color.copy(c); b.tone = fc.P.hairTone; changed = true; }
      show(b.group, true);
      show(b.parts.lod0, mode === 'lod0'); show(b.parts.lod1, mode === 'lod1'); show(b.parts.lod2, mode === 'lod2');
      show(b.parts.cap, true); show(b.parts.strands, mode === 'strands');
      if (mode in LOD_INDEX) { const a = JSON.parse(assets[style].dataset.summary || '{}'); note.textContent = a[mode] || ''; }
      else note.textContent = b.segs.toLocaleString() + ' segments, ' + asset_n(assets[style]) + ' strands';
    }
    if (changed) fc.controls.dispatchEvent({ type: 'change' });      // (the page draws on demand: ask for a frame)
    requestAnimationFrame(frame);
  }
  window.__hairFidelity = { set: v => { mode = v; sel.value = v; }, get: () => mode, built };
  frame();
}

(function wait() { if (window.__fc && window.__fc.hairMesh && window.__fc.hairMesh.parent) start(window.__fc); else setTimeout(wait, 200); })();
