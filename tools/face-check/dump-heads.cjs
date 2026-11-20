// Dump the Face Creator's preset heads as raw meshes for the facial proportion check (face_lines.py).
//   node dump-heads.cjs <face-creator.html> <out dir> [Ranger,Rogue,...]
// Runs the creator in headless Chromium, picks each face preset, waits for the build, and writes <out>/<face>.json:
// every visible mesh of the scene (head, eyes, ears, hair...) as world-space positions and triangles (base64
// Float32 / Uint32), tagged by role. Hair is dumped too (the hair cap gives the hairline) but never measured as skin.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const page_ = path.resolve(process.argv[2]), out = process.argv[3];
  const faces = (process.argv[4] || 'Ranger,Rogue,Warden,Sorceress,Knight').split(',');
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader'] });
  const p = await b.newPage({ viewport: { width: 900, height: 700 } });
  p.on('pageerror', (e) => console.log('page error:', e.message));
  await p.goto('file://' + page_);
  const idle = () => window.__fc && window.__fc.head.geometry.attributes.position && document.getElementById('busy').hidden;
  await p.waitForFunction(idle, null, { timeout: 600000 });
  for (const nm of faces) {
    const t0 = Date.now();
    await p.click(`button[data-face="${nm}"]`);
    await p.waitForTimeout(400);
    await p.waitForFunction(idle, null, { timeout: 600000 });
    await p.waitForTimeout(500);
    const data = await p.evaluate(() => {
      const FC = window.__fc, scene = FC.head.parent, out = [];
      function b64(a) { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); }
      scene.updateMatrixWorld(true);
      scene.traverse((m) => {
        if (!m.isMesh || !m.visible || !m.geometry.attributes.position) return;
        let v = m; while (v) { if (!v.visible) return; v = v.parent; }
        const g = m.geometry, pa = g.attributes.position, n = pa.count; if (!n) return;
        const P = new Float32Array(n * 3), e = m.matrixWorld.elements;
        for (let i = 0; i < n; i++) { const x = pa.getX(i), y = pa.getY(i), z = pa.getZ(i);
          P[i * 3] = e[0] * x + e[4] * y + e[8] * z + e[12]; P[i * 3 + 1] = e[1] * x + e[5] * y + e[9] * z + e[13]; P[i * 3 + 2] = e[2] * x + e[6] * y + e[10] * z + e[14]; }
        let I; if (g.index) I = Uint32Array.from(g.index.array); else { I = new Uint32Array(n); for (let i = 0; i < n; i++) I[i] = i; }
        const mat = Array.isArray(m.material) ? m.material[0] : m.material;
        let role = 'other';
        if (m === FC.head) role = 'head'; else if (FC.eyeMeshes.indexOf(m) >= 0) role = 'eye';
        else if (mat === FC.hairCapMat) role = 'hair';
        else if (mat && (mat === FC.hairCapMat || mat.alphaMap || /hair/i.test(m.name || ''))) role = 'hair';
        else if (FC.hairMesh && (m === FC.hairMesh)) role = 'hair';
        // (only the hair cap is kept: it gives the hairline; the strands are thousands of small meshes the check never reads)
        if (role === 'hair' && mat !== FC.hairCapMat) return;
        if (role === 'other' && n < 2000) return;
        out.push({ role, name: m.name || '', mat: mat ? (mat.name || mat.type) : '', cap: mat === FC.hairCapMat, n, P: b64(P), I: b64(I), nI: I.length });
      });
      return { meshes: out, P: { noseProj: FC.P.noseProj, noseWidth: FC.P.noseWidth, sex: FC.P.sex, eyeSpacing: FC.P.eyeSpacing } };
    });
    fs.writeFileSync(path.join(out, nm + '.json'), JSON.stringify({ face: nm, page: page_, ...data }));
    console.log(nm, data.meshes.map((m) => m.role + ':' + m.n).join(' '), ((Date.now() - t0) / 1000).toFixed(1) + 's');
  }
  await b.close();
})();
