// Export a face creator hairstyle for the Blender figures: the cap as a grid and every strand of every lock as a
// centre line with its width, thickness, shade and cut-out, in head heights in the creator's own axes (x across,
// y back, z up from the chin). blender/warden/warden.py builds the hair from it.
// usage: node export-hair.cjs <face-creator.html> <style> <out.json.gz> [head preset]
const { chromium } = require('playwright');
const zlib = require('zlib'), fs = require('fs'), path = require('path');
(async () => {
  const [page, style, out, face] = process.argv.slice(2);      // (face: a head preset to apply first, as Warden, so the
  // strand colours are painted in its hair tone)
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 320, height: 240 } });
  p.on('pageerror', e => { console.log('pageerror:', e.message); process.exit(1); });
  await p.goto('file://' + path.resolve(page));
  await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 300000 });
  if (face) {
    await p.locator(`[data-face="${face}"]`).click();
    await p.waitForTimeout(500);
    await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 300000 });
  }
  const r = await p.evaluate(style => {
    const HH = 8.6, STEP = 3, CI = 2, CJ = 2;
    const q = v => Math.round(v * 1e4) / 1e4;
    if (!__fc.HAIR_STYLES[style]) return null;
    __fc.P.hairStyle = style; __fc.update();
    const cap = __fc.hairMesh.geometry, cp = cap.attributes.position.array, cc = cap.attributes.color.array;
    const n = cap.attributes.position.count;
    // (the cap is a grid of columns round the head, each NV rows from the crown down: found from the index)
    const NV = cap.index.array[1], NTC = n / NV;
    const C = { cols: 0, rows: 0, p: [], a: [], u: [], v: [] };
    const cuv = cap.attributes.uv.array;
    const js = []; for (let j = 0; j < NV; j += CJ) js.push(j); if (js[js.length - 1] !== NV - 1) js.push(NV - 1);
    const is = []; for (let i = 0; i < NTC - 1; i += CI) is.push(i);
    C.cols = is.length; C.rows = js.length;
    is.forEach(i => js.forEach(j => { const k = i * NV + j; C.p.push(q(cp[k * 3] / HH), q(-cp[k * 3 + 2] / HH), q(cp[k * 3 + 1] / HH)); C.a.push(Math.round(cc[k * 4 + 3] * 100) / 100); C.u.push(q(cuv[k * 2])); C.v.push(q(cuv[k * 2 + 1])); }));
    const S = [];
    __fc.hairMesh.parent.children.forEach(m => {
      if (!m.isMesh || m === __fc.hairMesh || m.material !== __fc.hairCapMat || !m.geometry.userData.along) return;
      const g = m.geometry, P = g.attributes.position.array, K = g.attributes.color.array, cnt = g.attributes.position.count;
      // (a tube: rings of 9 points, the last repeating the first, then the two end points)
      const S1 = 9, rings = Math.floor((cnt - 2) / S1), st = { p: [], w: [], t: [], c: [], a: [], v: [] };
      // (where the strand sits on the hair-tip map, across: its first and last ring point; along it is the cut-out a)
      const UV = g.attributes.uv.array; st.u = [q(UV[0]), q(UV[(S1 - 1) * 2])];
      st.lock = g.userData.lock === undefined ? -1 : g.userData.lock; st.layer = g.userData.layer || 0;      // (which lock and layer: 0 the top, 1 the darker one under it)
      const pt = k => [P[k * 3] / HH, -P[k * 3 + 2] / HH, P[k * 3 + 1] / HH];
      for (let i = 0; i < rings; i += STEP) {
        const ii = Math.min(i, rings - 1), a = pt(ii * S1), o = pt(ii * S1 + 4), s2 = pt(ii * S1 + 2), s6 = pt(ii * S1 + 6);
        const c = [(a[0] + o[0]) / 2, (a[1] + o[1]) / 2, (a[2] + o[2]) / 2];
        st.p.push(q(c[0]), q(c[1]), q(c[2]));
        st.w.push(q(a[0] - c[0]), q(a[1] - c[1]), q(a[2] - c[2]));                                   // half the width, across the strand
        st.t.push(q(Math.hypot(s2[0] - s6[0], s2[1] - s6[1], s2[2] - s6[2]) / 2));                  // half its thickness
        const k = ii * S1 + 2;
        st.c.push(Math.round((K[k * 4] + K[k * 4 + 1] + K[k * 4 + 2]) / 3 * 100) / 100);        // its shade there
        st.a.push(Math.round(K[k * 4 + 3] * 100) / 100);
        st.v.push(q(UV[ii * S1 * 2 + 1]));                                                        // where it is down the strand texture                                          // the cut-out: 0 at the faded root and tip
      }
      S.push(st);
    });
    // the hair-tip map the cap and the strands wear (red: each hair's tone; green: the top layer's cut-out; blue: the
    // wider hairs of the layers under it), as the creator reads it three times over
    const tips = __fc.hairCapMat.alphaMap.image.toDataURL('image/png');
    // (and the hair's own surface: its strand colours, painted in the hair's tone, and its strand bump)
    const tex = __fc.hairCapMat.map.image.toDataURL('image/png'), bump = __fc.hairCapMat.bumpMap.image.toDataURL('image/png');
    return { style: style, name: __fc.HAIR_STYLES[style].name, cap: C, strands: S, tips: tips, tex: tex, bump: bump };
  }, style);
  await b.close();
  if (!r) { console.log('no such style: ' + style); process.exit(1); }
  // (the map goes beside the export as a PNG: hair_tips.png)
  fs.writeFileSync(path.join(path.dirname(out), 'hair_tips.png'), Buffer.from(r.tips.split(',')[1], 'base64'));
  fs.writeFileSync(path.join(path.dirname(out), 'hair_tex.png'), Buffer.from(r.tex.split(',')[1], 'base64'));
  fs.writeFileSync(path.join(path.dirname(out), 'hair_bump.png'), Buffer.from(r.bump.split(',')[1], 'base64'));
  delete r.tips; delete r.tex; delete r.bump;
  fs.writeFileSync(out, zlib.gzipSync(JSON.stringify(r), { level: 9 }));
  console.log(`${r.name}: cap ${r.cap.cols} x ${r.cap.rows}, ${r.strands.length} strands, ${fs.statSync(out).size} bytes`);
})();
