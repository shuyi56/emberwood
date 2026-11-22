// Hair fidelity: show a hairstyle as the creator sculpts it, or as the physics groom (tools/hair) baked for the game:
// cards at three levels of detail, or the strands themselves. Added to the page by apply-hair-fidelity.py, with each
// groomed style's asset (tools/hair/creator_asset.py) in a <script type="application/json" id="groom-asset-STYLE">.
import * as THREE from 'three';

const HH = 8.6;                                   // the creator's units per head height (rq_hair_export.cjs)
const MODES = [['creator', 'Creator (sculpted locks)'], ['lod2', 'Groom: cards, LOD 2'], ['lod1', 'Groom: cards, LOD 1'],
  ['lod0', 'Groom: cards, LOD 0'], ['strands', 'Groom: strands']];
const LOD_INDEX = { lod0: 0, lod1: 1, lod2: 2 };

function bytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
// head heights in the hair's frame (x across, y back, z up) to the page's units (x, y up, z toward the viewer)
function toPage(src, scale) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) { out[i] = src[i] * scale * HH; out[i + 1] = src[i + 2] * scale * HH; out[i + 2] = -src[i + 1] * scale * HH; }
  return out;
}

// the strand texture for the cards: fine fibres along v (root to tip), cut out toward the tip and at the card's edges
function strandTexture() {
  const W = 64, H = 256, c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let k = 0; k < 160; k++) {
    const x = rnd() * W, end = H * (0.82 + 0.18 * rnd()), edge = Math.min(x, W - x) / (W * 0.5);
    const a = Math.min(1, 0.35 + edge * 1.3) * (0.6 + 0.4 * rnd());
    const grd = g.createLinearGradient(0, 0, 0, end);
    grd.addColorStop(0, `rgba(255,255,255,${a})`); grd.addColorStop(0.8, `rgba(255,255,255,${a})`); grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.strokeStyle = grd; g.lineWidth = 0.8 + rnd() * 1.4; g.beginPath(); g.moveTo(x, 0); g.lineTo(x + (rnd() - 0.5) * 2, end); g.stroke();
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}

function buildStyle(asset, tone) {
  const group = new THREE.Group(), parts = {};
  const col = new THREE.Color(tone);
  const cardMat = new THREE.MeshStandardMaterial({ color: col, alphaMap: strandTexture(), alphaTest: 0.35, side: THREE.DoubleSide,
    roughness: 0.85, metalness: 0, envMapIntensity: 0.4 });
  const capMat = new THREE.MeshStandardMaterial({ color: col.clone().multiplyScalar(0.55), roughness: 0.9, metalness: 0 });
  asset.lods.forEach((L, i) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(toPage(new Float32Array(bytes(L.pos)), 1), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(bytes(L.uv)), 2));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(L.idx)), 1));
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
        const s = b * (0.85 + 0.3 * jj / (c - 1));
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
  const isCreatorHair = m => m.isMesh && (m === fc.hairMesh || m.material === fc.hairCapMat ||
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
      if (b.tone !== fc.P.hairTone) { const c = new THREE.Color(fc.P.hairTone); b.cardMat.color.copy(c); b.capMat.color.copy(c).multiplyScalar(0.55); b.tone = fc.P.hairTone; changed = true; }
      show(b.group, true);
      show(b.parts.lod0, mode === 'lod0'); show(b.parts.lod1, mode === 'lod1'); show(b.parts.lod2, mode === 'lod2');
      show(b.parts.cap, mode !== 'strands'); show(b.parts.strands, mode === 'strands');
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
