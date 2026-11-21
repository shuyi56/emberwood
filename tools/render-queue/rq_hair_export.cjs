// The hair stress test's input: every hairstyle of the face creator, built in the page one after another and written
// as rq_hair.py reads it (the cap as a grid and every strand of every lock as a centre line with its half-width
// and half-thickness, in head heights in the creator's own axes: x across, y back, z up from the chin), with each
// style's head preset where one is given. The same strand read as tools/face-creator/export-hair.cjs (which the
// Blender figures build from), every ring instead of every third, so a kink between rings is not stepped over.
// usage: node rq_hair_export.cjs <face-creator.html> <out dir> [styles, comma separated] [head preset]
const { chromium } = require('playwright');
const zlib = require('zlib'), fs = require('fs'), path = require('path');
(async () => {
  const [page, out, only, face] = process.argv.slice(2);
  fs.mkdirSync(out, { recursive: true });
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
  // the head the hair grows on (the creator's skin, every 5th point with its normal, in the hair's frame and units),
  // and the bodies the creator drapes long hair on (its MANNEQUIN table and each character's sections, read from
  // the page's source): hair_stress.py's colliders
  const head = await p.evaluate(() => {
    const HH = 8.6, q = v => Math.round(v * 1e4) / 1e4, frame = __fc.hairMesh.parent;
    let root = frame; while (root.parent) root = root.parent;
    root.updateMatrixWorld(true);
    let skin = null;
    root.traverse(m => { if (m.isMesh && m.visible && m.material && m.material.type === 'MeshPhysicalMaterial' &&
      (!skin || m.geometry.attributes.position.count > skin.geometry.attributes.position.count)) skin = m; });
    const inv = frame.matrixWorld.clone().invert(), M = inv.multiply(skin.matrixWorld), g = skin.geometry;
    const P = g.attributes.position, N = g.attributes.normal, e = M.elements, out = { p: [], n: [] };
    for (let i = 0; i < P.count; i += 5) {
      const x = P.getX(i), y = P.getY(i), z = P.getZ(i), nx = N.getX(i), ny = N.getY(i), nz = N.getZ(i);
      const X = e[0] * x + e[4] * y + e[8] * z + e[12], Y = e[1] * x + e[5] * y + e[9] * z + e[13], Z = e[2] * x + e[6] * y + e[10] * z + e[14];
      let a = e[0] * nx + e[4] * ny + e[8] * nz, b = e[1] * nx + e[5] * ny + e[9] * nz, c = e[2] * nx + e[6] * ny + e[10] * nz;
      const l = Math.hypot(a, b, c) || 1;
      out.p.push(q(X / HH), q(-Z / HH), q(Y / HH)); out.n.push(q(a / l), q(-c / l), q(b / l));
    }
    return out;
  });
  const src = fs.readFileSync(page, 'utf8'), grab = re => { const m = src.match(re); return m ? m[1] : null; };
  const bodies = {
    mannequin: JSON.parse(grab(/var MANNEQUIN = (\{[^}]*\})/).replace(/(\w+):/g, '"$1":')),
    ranger: JSON.parse(grab(/var RANGER_DRAPE = makeDrape\((\[[\s\S]*?\]\])/)),
    warden: JSON.parse(grab(/var WARDEN_DRAPE = makeDrape\((\[[\s\S]*?\]\])/)),
    styles: { ranger: 'ranger', wavyFull: 'warden' },
  };
  fs.writeFileSync(path.join(out, 'head.json.gz'), zlib.gzipSync(JSON.stringify({ head, bodies }), { level: 9 }));
  console.log(`head: ${head.p.length / 3} points`);
  const styles = only ? only.split(',') : await p.evaluate(() => Object.keys(__fc.HAIR_STYLES));
  for (const style of styles) {
    const r = await p.evaluate(style => {
      const HH = 8.6, q = v => Math.round(v * 1e4) / 1e4;
      if (!__fc.HAIR_STYLES[style]) return null;
      __fc.P.hairStyle = style; __fc.update();
      const cap = __fc.hairMesh.geometry, cp = cap.attributes.position.array, cc = cap.attributes.color.array;
      const n = cap.attributes.position.count, NV = cap.index.array[1], NTC = n / NV;
      const C = { cols: 0, rows: 0, p: [], a: [] };
      const js = []; for (let j = 0; j < NV; j += 2) js.push(j); if (js[js.length - 1] !== NV - 1) js.push(NV - 1);
      const is = []; for (let i = 0; i < NTC - 1; i += 2) is.push(i);
      C.cols = is.length; C.rows = js.length;
      is.forEach(i => js.forEach(j => { const k = i * NV + j; C.p.push(q(cp[k * 3] / HH), q(-cp[k * 3 + 2] / HH), q(cp[k * 3 + 1] / HH)); C.a.push(Math.round(cc[k * 4 + 3] * 100) / 100); }));
      const S = [];
      __fc.hairMesh.parent.children.forEach(m => {
        if (!m.isMesh || m === __fc.hairMesh || m.material !== __fc.hairCapMat || !m.geometry.userData.along) return;
        const g = m.geometry, P = g.attributes.position.array, cnt = g.attributes.position.count;
        // (a ring has sides + 1 points, the last repeating the first, and every point of a ring has the same `along`:
        // read the ring size from that, the tube's sides are 8 for a clump and 14 or 12 for a broad piece)
        const al = g.userData.along; let S1 = 1; while (S1 < al.length && al[S1] === al[0]) S1++;
        const rings = Math.floor((cnt - 2) / S1), sd = S1 - 1, st = { p: [], w: [], t: [] };
        st.lock = g.userData.lock === undefined ? -1 : g.userData.lock; st.layer = g.userData.layer || 0;
        const pt = k => [P[k * 3] / HH, -P[k * 3 + 2] / HH, P[k * 3 + 1] / HH];
        for (let i = 0; i < rings; i++) {
          // (the centre is the mean of the ring's sides points; the half-width is the distance to the first point (the
          // section's x axis), the half-thickness the distance to the point a quarter turn round)
          const c = [0, 0, 0]; for (let j = 0; j < sd; j++) { const q = pt(i * S1 + j); c[0] += q[0] / sd; c[1] += q[1] / sd; c[2] += q[2] / sd; }
          const a = pt(i * S1), s2 = pt(i * S1 + Math.round(sd / 4));
          st.p.push(q(c[0]), q(c[1]), q(c[2]));
          st.w.push(q(Math.hypot(a[0] - c[0], a[1] - c[1], a[2] - c[2])));
          st.t.push(q(Math.hypot(s2[0] - c[0], s2[1] - c[1], s2[2] - c[2])));
        }
        if (rings > 1) S.push(st);
      });
      return { style: style, name: __fc.HAIR_STYLES[style].name, cap: C, strands: S };
    }, style);
    if (!r) { console.log('no such style: ' + style); continue; }
    const f = path.join(out, style + '.json.gz');
    fs.writeFileSync(f, zlib.gzipSync(JSON.stringify(r), { level: 9 }));
    console.log(`${style} (${r.name}): cap ${r.cap.cols} x ${r.cap.rows}, ${r.strands.length} strands`);
  }
  await b.close();
})();
