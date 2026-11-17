// Export the Face Creator's head and hair (as built in the body viewer) to a glb, in the creator's frame (chin at 0,
// y up, face toward +z), the frame of the Head node in ranger.glb.
//   node export-head.js [viewer.html] [out.glb] [settings.json]
// Runs the viewer in headless Chromium, waits for the head and hair, and writes every visible mesh that is not the
// body through three's GLTFExporter. (The creator's shader tweaks, onBeforeCompile, do not export: the model gets the
// plain material with its painted maps.)
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const page_ = path.resolve(process.argv[2] || path.join(__dirname, 'ranger-body-viewer.html'));
  const out = process.argv[3] || 'creator-head.glb', settings = process.argv[4] ? fs.readFileSync(process.argv[4], 'utf8') : null;
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader'] });
  const p = await b.newPage();
  p.on('pageerror', (e) => console.log('page error:', e.message));
  await p.goto('file://' + page_);
  const ready = () => window.__rangerBody && window.__rangerBody.group.children.length > 1 && document.getElementById('busy').hidden;
  await p.waitForFunction(ready, null, { timeout: 300000 });
  // The face paint for the world: the skin's fine surface texture and blotches are painted for the creator's own
  // shader (a bump map and a pore tint undo them there); exported bare, they read as a grey-green mottle in the game.
  // So the export repaints with them off, keeping the freckles, flush, make-up, brows and lips.
  // The lips for the world: flatter and painted, a clear shape of colour on an almost flat mouth (the sculpted volume,
  // lit by the game's sun, read as a puffy pout at play distance), no gloss.
  // The mouth is also a little wider with its corners turned up, and the cheeks round it slimmer, so that it reads as a
  // soft closed smile rather than a small pursed mouth set between full cheeks.
  const FOR_WORLD = { texture: 0, blotch: 0, lipUpper: 0.55, lipLower: 0.6, lipGloss: 0, lipLine: 1, smile: 2.2, mouthWidth: 1.0,
    apples: 0.3, cheekFull: 0.25, chinProj: 1.0 };
  if (process.env.WORLD) Object.assign(FOR_WORLD, JSON.parse(process.env.WORLD));
  const base = settings ? JSON.parse(settings) : await p.evaluate(() => JSON.parse(JSON.stringify(window.__fc.P)));
  await p.fill('#vw-json', JSON.stringify(Object.assign(base, FOR_WORLD))); await p.click('#vw-load');
  await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 300000 });
  await p.waitForTimeout(3000);
  const STRAND_MAX = +(process.env.STRAND_MAX || 2000), FRONT_Z = +(process.env.FRONT_Z || 1.3);
  const b64 = await p.evaluate(async ([STRAND_MAX, FRONT_Z]) => {
    const THREE = await import('three');
    const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js');
    const RB = window.__rangerBody, g = new THREE.Group(); g.name = 'Creator head';
    let k = 0;
    // glTF has no alpha map: the hair strands' alpha (its green channel, as three reads it) goes into the alpha of
    // their colour map instead, with the same cut-off (alphaMode MASK), or every strand exports as an opaque ribbon
    const baked = new Map();
    function draw(tex, w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d'); x.drawImage(tex.image, 0, 0, w, h); return x.getImageData(0, 0, w, h); }
    function bakeAlpha(m) {
      if (!m.alphaMap) return m;
      if (baked.has(m)) return baked.get(m);
      const am = m.alphaMap, mp = m.map, w = Math.min(1024, (mp || am).image.width), h = Math.min(1024, (mp || am).image.height);
      const A = draw(am, w, h), C = mp ? draw(mp, w, h) : null, c = document.createElement('canvas'); c.width = w; c.height = h;
      const x = c.getContext('2d'), D = x.createImageData(w, h);
      for (let i = 0; i < w * h; i++) { const o = i * 4; D.data[o] = C ? C.data[o] : 255; D.data[o + 1] = C ? C.data[o + 1] : 255; D.data[o + 2] = C ? C.data[o + 2] : 255; D.data[o + 3] = A.data[o + 1] * (C ? C.data[o + 3] / 255 : 1); }
      x.putImageData(D, 0, 0);
      const t = new THREE.CanvasTexture(c); t.colorSpace = mp ? mp.colorSpace : THREE.SRGBColorSpace; t.flipY = (mp || am).flipY;
      t.wrapS = (mp || am).wrapS; t.wrapT = (mp || am).wrapT; t.repeat.copy((mp || am).repeat); t.offset.copy((mp || am).offset);
      const n = m.clone(); n.map = t; n.alphaMap = null; n.alphaTest = m.alphaTest || 0.5; n.transparent = false;
      baked.set(m, n); return n;
    }
    RB.scene.children.forEach((o) => {
      if (o === RB.group || !o.visible) return;
      o.traverse((m) => {
        if (!m.isMesh || !m.visible) return;
        // (the fine strand clumps laid over each lock: hundreds of thin alpha-cut ribbons that at the world's size only
        // break into a speckle of dark dashes over the hair; the locks underneath carry the shape and the light)
        // (the bangs and the locks framing the face are built the same way, so only those behind the front of the head go)
        if (m.material && m.material.alphaMap && m.geometry.attributes.position.count < STRAND_MAX) {
          m.geometry.computeBoundingBox(); const c = m.geometry.boundingBox.getCenter(new THREE.Vector3()).applyMatrix4(m.matrixWorld);
          if (c.z < FRONT_Z) return;
        }
        m.updateWorldMatrix(true, false);
        const c = new THREE.Mesh(m.geometry, Array.isArray(m.material) ? m.material.map(bakeAlpha) : bakeAlpha(m.material));
        c.name = (m.name || 'head part') + ' ' + (k++);
        c.applyMatrix4(m.matrixWorld);
        g.add(c);
      });
    });
    const buf = await new GLTFExporter().parseAsync(g, { binary: true, onlyVisible: true, maxTextureSize: 2048 });
    let s = ''; const u = new Uint8Array(buf); for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
    return btoa(s);
  }, [STRAND_MAX, FRONT_Z]);
  fs.writeFileSync(out, Buffer.from(b64, 'base64'));
  console.log('wrote', out, fs.statSync(out).size, 'bytes');
  await b.close();
})();
