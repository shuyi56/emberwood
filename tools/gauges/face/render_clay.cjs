// Cheap clay and matcap renders of Face Creator heads: orthographic front, 3/4, side and back, plus close-ups of the
// brow-to-mouth region from the front and 3/4, the hair and everything but the head, eyes and ears hidden. No skin,
// no materials: form only (the project's draft-render rule). Two shadings per view: a soft clay matcap and a raking
// key from the upper left that shows the plane breaks.
//
//   node tools/gauges/face/render_clay.cjs <face-creator.html> <out dir> [Ranger,Rogue,...] [settings JSON]
//
// Writes <out>/<face>-clay.png: one sheet per face (rows: clay, raking key; columns: front, 3/4, side, back, close
// front, close 3/4), and <out>/<face>-<view>.png for each panel. Runs in headless Chromium (Playwright), about 20 s a face.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const [page_, out, facesArg, extra] = process.argv.slice(2);
  const faces = (facesArg || 'Ranger,Rogue,Warden,Sorceress,Knight').split(',');
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage({ viewport: { width: 900, height: 700 } });
  p.on('pageerror', (e) => console.log('page error:', e.message));
  await p.goto('file://' + path.resolve(page_));
  const idle = () => window.__fc && window.__fc.head.geometry.attributes.position && document.getElementById('busy').hidden;
  await p.waitForFunction(idle, null, { timeout: 600000 });
  for (const nm of faces) {
    await p.click(`button[data-face="${nm}"]`);
    await p.waitForTimeout(400);
    await p.waitForFunction(idle, null, { timeout: 600000 });
    if (extra) { await p.evaluate((x) => { Object.assign(__fc.P, JSON.parse(x)); __fc.update(); }, extra); await p.waitForTimeout(400); await p.waitForFunction(idle, null, { timeout: 600000 }); }
    await p.waitForTimeout(500);
    const shots = await p.evaluate(async () => {
      const FC = window.__fc, head = FC.head, scene = head.parent, R = FC.renderer;
      const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js');
      // keep the head, the eyes and the ears (the big meshes that are not hair)
      const keep = new Set([head, ...FC.eyeMeshes]);
      const hidden = [];
      scene.traverse((m) => {
        if (!m.isMesh || !m.visible) return;
        const mat = Array.isArray(m.material) ? m.material[0] : m.material;
        const n = m.geometry.attributes.position ? m.geometry.attributes.position.count : 0;
        const hair = mat === FC.hairCapMat || (mat && mat.alphaMap) || /hair/i.test(m.name || '');
        if (keep.has(m) || (!hair && n > 5000 && n < 20000)) return;
        m.visible = false; hidden.push(m);
      });
      function matcap(kind) {
        const c = document.createElement('canvas'); c.width = c.height = 256; const g = c.getContext('2d'), D = g.createImageData(256, 256);
        const L = kind === 'key' ? [-0.55, 0.6, 0.58] : [-0.25, 0.35, 0.9], l = Math.hypot(...L);
        for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
          const nx = (x + 0.5) / 128 - 1, ny = 1 - (y + 0.5) / 128, r2 = nx * nx + ny * ny, k = (y * 256 + x) * 4;
          if (r2 > 1) { D.data[k + 3] = 255; continue; }
          const nz = Math.sqrt(1 - r2), d = Math.max(0, (nx * L[0] + ny * L[1] + nz * L[2]) / l);
          const v = kind === 'key' ? 0.1 + 0.85 * Math.pow(d, 1.6) : 0.28 + 0.55 * d + 0.17 * Math.pow(nz, 3);
          D.data[k] = 255 * v * 0.86; D.data[k + 1] = 255 * v * 0.8; D.data[k + 2] = 255 * v * 0.74; D.data[k + 3] = 255;
        }
        g.putImageData(D, 0, 0); const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
      }
      const mats = { clay: new THREE.MeshMatcapMaterial({ matcap: matcap('clay') }), key: new THREE.MeshMatcapMaterial({ matcap: matcap('key') }) };
      head.geometry.computeBoundingBox();
      const bb = head.geometry.boundingBox.clone().applyMatrix4(head.matrixWorld);
      const eyes = FC.eyeMeshes.map((e) => { e.geometry.computeBoundingBox(); const q = new THREE.Vector3(); e.geometry.boundingBox.getCenter(q); return q.applyMatrix4(e.matrixWorld); });
      const ec = eyes.length ? eyes.reduce((a, q) => a.add(q), new THREE.Vector3()).multiplyScalar(1 / eyes.length) : bb.getCenter(new THREE.Vector3());
      const ipd = eyes.length > 1 ? eyes[0].distanceTo(eyes[1]) : 2.3;
      const hc = bb.getCenter(new THREE.Vector3()), hh = bb.max.y - bb.min.y;
      const W = 700, H = 760, old = new THREE.Vector2(); R.getSize(old); const oldPR = R.getPixelRatio();
      R.setPixelRatio(1); R.setSize(W, H, false);
      const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
      const views = [
        ['front', 0, hc, 0.62 * hh], ['3q', 35, hc, 0.62 * hh], ['side', 90, hc, 0.62 * hh], ['back', 180, hc, 0.62 * hh],
        ['close-front', 0, new THREE.Vector3(ec.x, ec.y - 0.32 * ipd, ec.z), 0.95 * ipd], ['close-3q', 35, new THREE.Vector3(ec.x, ec.y - 0.32 * ipd, ec.z), 0.95 * ipd]];
      const outp = {};
      const bg = scene.background; scene.background = new THREE.Color(0x1d1f24);
      for (const sh of ['clay', 'key']) {
        scene.overrideMaterial = mats[sh];
        for (const [name, az, c, half] of views) {
          const a = az * Math.PI / 180;      // (az: the camera swung round toward the face's left, as the creator's 3/4)
          cam.left = -half * W / H; cam.right = half * W / H; cam.top = half; cam.bottom = -half; cam.updateProjectionMatrix();
          cam.position.set(c.x + 30 * Math.sin(a), c.y, c.z + 30 * Math.cos(a)); cam.up.set(0, 1, 0); cam.lookAt(c);
          R.render(scene, cam);
          outp[sh + '-' + name] = R.domElement.toDataURL('image/png');
        }
      }
      scene.overrideMaterial = null; scene.background = bg; hidden.forEach((m) => { m.visible = true; });
      R.setPixelRatio(oldPR); R.setSize(old.x, old.y, false);
      return outp;
    });
    for (const k in shots) fs.writeFileSync(path.join(out, `${nm}-${k}.png`), Buffer.from(shots[k].split(',')[1], 'base64'));
    console.log(nm, Object.keys(shots).length, 'panels');
  }
  await b.close();
})();
