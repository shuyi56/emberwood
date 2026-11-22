// Hair fidelity: show a hairstyle as the creator sculpts it, or as the physics groom (tools/hair) baked for the game:
// cards at three levels of detail, or the strands themselves. Added to the page by apply-hair-fidelity.py, with each
// groomed style's asset (tools/hair/creator_asset.py) in a <script type="application/json" id="groom-asset-STYLE">.
import * as THREE from 'three';

const HH = 8.6;                                   // the creator's units per head height (rq_hair_export.cjs)
const MODES = [['creator', 'Creator (sculpted locks)'], ['lod2', 'Groom: cards, LOD 2'], ['lod1', 'Groom: cards, LOD 1'],
  ['lod0', 'Groom: cards, LOD 0'], ['strands', 'Groom: strands']];
const LOD_INDEX = { lod0: 0, lod1: 1, lod2: 2 };
const SCALP_SHADE = 0.6;                          // STYLE: the scalp cap against the hair colour. (render.py draws it at 0.35, among far denser strands; lit
                                                  // under cards, 0.35 read as dark holes between them)

function bytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
// head heights in the hair's frame (x across, y back, z up) to the page's units (x, y up, z toward the viewer)
function toPage(src, scale) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) { out[i] = src[i] * scale * HH; out[i + 1] = src[i + 2] * scale * HH; out[i + 2] = -src[i + 1] * scale * HH; }
  return out;
}

// The cards' strand texture, baked once (grey, multiplied by the hair colour). v runs root (0) to tip (1). A card
// stands for a few locks, so the texture is CLUMPS locks side by side: each a bundle of fine fibres that narrows to
// its own tip, with a dark crease between neighbours, a root-to-tip gradient (dark roots, lightest toward the tips)
// and each fibre a little lighter or darker than the next. The alpha fades in over the first ROOT_FADE of v (the
// root colour of the scalp cap shows through, not a card's hard edge) and cuts each lock off at its own length.
// (80 Level, "Tips and tricks on hair for games": clumps, root-to-tip gradient and per-strand value in the texture.)
const ROOT_DARK = 0.7;                            // STYLE: the root's value against the mid-length
const TIP_LIGHT = 1.25;                           // STYLE: the tip's value against the mid-length
const CLUMPS = 6;                                 // STYLE: locks across one card: wide enough to read at LOD 0 size
const ROOT_FADE = 0.05;                           // STYLE
function strandTexture() {
  const W = 256, H = 512, c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'); g.clearRect(0, 0, W, H);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  // v is linear (1 = the hair colour itself); the canvas holds sRGB
  const grey = v => { const x = Math.round(255 * Math.pow(Math.min(1, v), 1 / 2.2)); return `${x},${x},${x}`; };
  const cw = W / CLUMPS;
  for (let j = 0; j < CLUMPS; j++) {
    const cx = (j + 0.5 + (rnd() - 0.5) * 0.3) * cw, len = H * (0.72 + 0.28 * rnd()), tipX = cx + (rnd() - 0.5) * cw * 0.4;
    const edge = Math.min(cx, W - cx) / (W * 0.5);              // the card's outer locks are thinner, so its edge is soft
    for (let k = 0; k < 90; k++) {
      const off = (rnd() - 0.5) * cw * 0.95, x0 = cx + off, end = len * (0.8 + 0.2 * rnd());
      const x1 = tipX + off * 0.25 * (end / len);                 // the lock narrows to its tip
      const centre = 1 - Math.abs(off) / (cw * 0.5);              // 1 at the lock's middle, 0 at the crease
      const a = Math.min(1, 0.3 + 0.9 * centre) * Math.min(1, 0.35 + 1.3 * edge) * (0.6 + 0.4 * rnd());
      const b = (0.75 + 0.2 * rnd()) * (0.75 + 0.25 * centre), f = end / H;
      const grd = g.createLinearGradient(0, 0, 0, end);
      grd.addColorStop(0, `rgba(${grey(b * ROOT_DARK)},0)`);
      grd.addColorStop(Math.min(0.5, ROOT_FADE / f), `rgba(${grey(b * ROOT_DARK)},${a})`);
      grd.addColorStop(Math.min(0.9, 0.35 / f), `rgba(${grey(b)},${a})`);
      grd.addColorStop(0.92, `rgba(${grey(b * TIP_LIGHT)},${a * 0.85})`);
      grd.addColorStop(1, `rgba(${grey(b * TIP_LIGHT)},0)`);
      g.strokeStyle = grd; g.lineWidth = 1 + rnd() * 1.6; g.beginPath();
      g.moveTo(x0, 0); g.quadraticCurveTo(x0 + (rnd() - 0.5) * 4, end * 0.55, x1, end); g.stroke();
    }
  }
  const t = new THREE.CanvasTexture(c); t.flipY = false;          // canvas row y is v = y / H: the root at the top
  t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping; t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function buildStyle(asset, tone) {
  const group = new THREE.Group(), parts = {};
  const col = new THREE.Color(tone);
  const cardMat = new THREE.MeshStandardMaterial({ color: col, map: strandTexture(), alphaTest: 0.3, side: THREE.DoubleSide,
    vertexColors: true, roughness: 0.75, metalness: 0, envMapIntensity: 0.3 });
  // the scalp under the hair in the root colour, as render.py draws it (SCALP_SHADE), so no skin shows between hairs
  const capMat = new THREE.MeshStandardMaterial({ color: col.clone().multiplyScalar(SCALP_SHADE), roughness: 0.95, metalness: 0 });
  asset.lods.forEach((L, i) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(toPage(new Float32Array(bytes(L.pos)), 1), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(bytes(L.uv)), 2));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(L.idx)), 1));
    if (L.ao) {                                   // the light through the hair, baked per vertex (creator_asset.py)
      const ao = new Float32Array(bytes(L.ao)), vc = new Float32Array(ao.length * 3);
      for (let i = 0; i < ao.length; i++) vc[3 * i] = vc[3 * i + 1] = vc[3 * i + 2] = ao[i];
      g.setAttribute('color', new THREE.BufferAttribute(vc, 3));
    }
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, cardMat); m.castShadow = true; m.name = 'groom-cards-' + i;
    parts['lod' + i] = m; group.add(m);
  });
  const cg = new THREE.BufferGeometry();
  cg.setAttribute('position', new THREE.BufferAttribute(toPage(new Float32Array(bytes(asset.cap.pos)), 1), 3));
  cg.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(asset.cap.idx)), 1)); cg.computeVertexNormals();
  parts.cap = new THREE.Mesh(cg, capMat); group.add(parts.cap);
  // the strands: line segments, each strand a little lighter or darker than the next and lighter toward its tip
  const S = asset.strands, cnt = new Uint16Array(bytes(S.counts)), raw = new Int16Array(bytes(S.pos));
  const pts = new Float32Array(raw.length); for (let i = 0; i < raw.length; i++) pts[i] = raw[i] * S.q;
  const P = toPage(pts, 1);
  let nseg = 0; for (const c of cnt) nseg += c - 1;
  const pos = new Float32Array(nseg * 6), vc = new Float32Array(nseg * 6);
  let o = 0, k = 0, seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (const c of cnt) {
    const b = 0.75 + 0.5 * rnd();
    for (let j = 0; j < c - 1; j++) {
      for (const jj of [j, j + 1]) {
        const t = jj / (c - 1), s = b * (t < 0.35 ? ROOT_DARK + (1 - ROOT_DARK) * t / 0.35 : 1 + (TIP_LIGHT - 1) * (t - 0.35) / 0.65);
        pos.set(P.subarray((k + jj) * 3, (k + jj) * 3 + 3), o);
        vc[o] = col.r * s; vc[o + 1] = col.g * s; vc[o + 2] = col.b * s; o += 3;
      }
    }
    k += c;
  }
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); sg.setAttribute('color', new THREE.BufferAttribute(vc, 3));
  parts.strands = new THREE.LineSegments(sg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75 }));
  group.add(parts.strands);
  return { group, parts, cardMat, capMat, tone };
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
  function frame() {
    place();
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
      if (!b) { b = built[style] = buildStyle(JSON.parse(assets[style].textContent), fc.P.hairTone); scene.add(b.group);
        note.textContent = ''; }
      if (b.tone !== fc.P.hairTone) { const c = new THREE.Color(fc.P.hairTone); b.cardMat.color.copy(c); b.capMat.color.copy(c).multiplyScalar(SCALP_SHADE); b.tone = fc.P.hairTone; changed = true; }
      show(b.group, true);
      show(b.parts.lod0, mode === 'lod0'); show(b.parts.lod1, mode === 'lod1'); show(b.parts.lod2, mode === 'lod2');
      show(b.parts.cap, true); show(b.parts.strands, mode === 'strands');
      if (mode in LOD_INDEX) { const a = JSON.parse(assets[style].dataset.summary || '{}'); note.textContent = a[mode] || ''; }
      else note.textContent = (b.parts.strands.geometry.attributes.position.count / 2 | 0).toLocaleString() + ' segments';
    }
    if (changed) fc.controls.dispatchEvent({ type: 'change' });      // (the page draws on demand: ask for a frame)
    requestAnimationFrame(frame);
  }
  window.__hairFidelity = { set: v => { mode = v; sel.value = v; }, get: () => mode, built };
  frame();
}

(function wait() { if (window.__fc && window.__fc.hairMesh && window.__fc.hairMesh.parent) start(window.__fc); else setTimeout(wait, 200); })();
