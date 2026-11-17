// Export a face creator hairstyle for the Blender figures: the cap as a grid and every strand of every lock as a
// centre line with its width, thickness, shade and cut-out, in head heights in the creator's own axes (x across,
// y back, z up from the chin). blender/warden/warden.py builds the hair from it.
// usage: node export-hair.cjs <face-creator.html> <style> <out.json.gz>
const { chromium } = require('playwright');
const zlib = require('zlib'), fs = require('fs'), path = require('path');
(async () => {
  const [page, style, out] = process.argv.slice(2);
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 320, height: 240 } });
  p.on('pageerror', e => { console.log('pageerror:', e.message); process.exit(1); });
  await p.goto('file://' + path.resolve(page));
  await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 300000 });
  const r = await p.evaluate(style => {
    const HH = 8.6, STEP = 3, CI = 2, CJ = 2;
    const q = v => Math.round(v * 1e4) / 1e4;
    if (!__fc.HAIR_STYLES[style]) return null;
    __fc.P.hairStyle = style; __fc.update();
    const cap = __fc.hairMesh.geometry, cp = cap.attributes.position.array, cc = cap.attributes.color.array;
    const n = cap.attributes.position.count;
    // (the cap is a grid of columns round the head, each NV rows from the crown down: found from the index)
    const NV = cap.index.array[1], NTC = n / NV;
    const C = { cols: 0, rows: 0, p: [], a: [] };
    const js = []; for (let j = 0; j < NV; j += CJ) js.push(j); if (js[js.length - 1] !== NV - 1) js.push(NV - 1);
    const is = []; for (let i = 0; i < NTC - 1; i += CI) is.push(i);
    C.cols = is.length; C.rows = js.length;
    is.forEach(i => js.forEach(j => { const k = i * NV + j; C.p.push(q(cp[k * 3] / HH), q(-cp[k * 3 + 2] / HH), q(cp[k * 3 + 1] / HH)); C.a.push(Math.round(cc[k * 4 + 3] * 100) / 100); }));
    const S = [];
    __fc.hairMesh.parent.children.forEach(m => {
      if (!m.isMesh || m === __fc.hairMesh || m.material !== __fc.hairCapMat || !m.geometry.userData.along) return;
      const g = m.geometry, P = g.attributes.position.array, K = g.attributes.color.array, cnt = g.attributes.position.count;
      // (a tube: rings of 9 points, the last repeating the first, then the two end points)
      const S1 = 9, rings = Math.floor((cnt - 2) / S1), st = { p: [], w: [], t: [], c: [], a: [] };
      const pt = k => [P[k * 3] / HH, -P[k * 3 + 2] / HH, P[k * 3 + 1] / HH];
      for (let i = 0; i < rings; i += STEP) {
        const ii = Math.min(i, rings - 1), a = pt(ii * S1), o = pt(ii * S1 + 4), s2 = pt(ii * S1 + 2), s6 = pt(ii * S1 + 6);
        const c = [(a[0] + o[0]) / 2, (a[1] + o[1]) / 2, (a[2] + o[2]) / 2];
        st.p.push(q(c[0]), q(c[1]), q(c[2]));
        st.w.push(q(a[0] - c[0]), q(a[1] - c[1]), q(a[2] - c[2]));                                   // half the width, across the strand
        st.t.push(q(Math.hypot(s2[0] - s6[0], s2[1] - s6[1], s2[2] - s6[2]) / 2));                  // half its thickness
        const k = ii * S1 + 2;
        st.c.push(Math.round((K[k * 4] + K[k * 4 + 1] + K[k * 4 + 2]) / 3 * 100) / 100);        // its shade there
        st.a.push(Math.round(K[k * 4 + 3] * 100) / 100);                                          // the cut-out: 0 at the faded root and tip
      }
      S.push(st);
    });
    return { style: style, name: __fc.HAIR_STYLES[style].name, cap: C, strands: S };
  }, style);
  await b.close();
  if (!r) { console.log('no such style: ' + style); process.exit(1); }
  fs.writeFileSync(out, zlib.gzipSync(JSON.stringify(r), { level: 9 }));
  console.log(`${r.name}: cap ${r.cap.cols} x ${r.cap.rows}, ${r.strands.length} strands, ${fs.statSync(out).size} bytes`);
})();
