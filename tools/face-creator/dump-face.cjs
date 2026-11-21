// Dump the Face Creator's face for Blender (blender/ranger/creator_face.py): the head, eye and ear meshes in Blender's
// axes (x, y back, z up; model units), and every map and number the creator's shaders read, including those that
// GLTFExporter drops (the pore relief, the lip layer map, the roughness map, the layered skin's lookups and the lips'
// and eyes' shader constants). It is the port's single input, so Blender shows the face exactly as the creator builds it.
//
//   node tools/face-creator/dump-face.cjs <face-creator.html> <out dir> [face preset] [settings JSON]
//
// Writes <out>/face.json (meshes and numbers), the maps as PNGs beside it, and creator-front.png (the creator's own
// render of the front, for reference).
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const [page, out, face, extra] = process.argv.slice(2);
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1300, height: 1000 } });
  p.on('pageerror', (e) => { console.log('pageerror:', e.message); process.exit(1); });
  p.on('console', (m) => { if (m.type() === 'error') console.log(m.text()); });
  await p.goto('file://' + path.resolve(page));
  await p.waitForFunction(() => document.getElementById('busy').hidden && window.__skin, null, { timeout: 900000 });
  if (face) {
    await p.evaluate((f) => document.querySelector(`[data-face="${f}"]`).click(), face);
    await p.waitForTimeout(400);
    await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 900000 });
  }
  if (extra) await p.evaluate((x) => { Object.assign(__fc.P, JSON.parse(x)); __fc.update(); }, extra);
  // (the layered skin's first bake runs after the load; the lip map and the pore tile are drawn with the face)
  await p.waitForFunction(() => !(__fc.P.skinLayers > 0) || __skin.B.done, null, { timeout: 900000 });
  await p.waitForFunction(() => __skin.head.material.bumpMap && __skin.U.lipMap.value, null, { timeout: 900000 });
  await p.waitForTimeout(1500);
  await p.locator('#gl').screenshot({ path: path.join(out, 'creator-front.png') });
  const data = await p.evaluate(() => {
    const S = __skin, U = S.U, M = S.head.material;
    const bl = (a, k) => [a[k * 3], -a[k * 3 + 2], a[k * 3 + 1]];      // (three.js (x, y up, z front) to Blender (x, y back, z up))
    function mesh(m, attrs) {
      m.updateWorldMatrix(true, false);
      const g = m.geometry.clone(); g.applyMatrix4(m.matrixWorld);
      const P = g.attributes.position.array, N = g.attributes.normal && g.attributes.normal.array, n = g.attributes.position.count;
      const o = { verts: [], index: g.index ? Array.from(g.index.array) : null };
      for (let k = 0; k < n; k++) o.verts.push(...bl(P, k));
      if (N) { o.normals = []; for (let k = 0; k < n; k++) o.normals.push(...bl(N, k)); }
      if (g.attributes.uv) o.uv = Array.from(g.attributes.uv.array);
      if (g.attributes.color) { const C = g.attributes.color, s = C.itemSize; o.color = []; for (let k = 0; k < n; k++) o.color.push(C.array[k * s], C.array[k * s + 1], C.array[k * s + 2]); }
      (attrs || []).forEach((a) => { if (g.attributes[a]) o[a] = Array.from(g.attributes[a].array); });
      return o;
    }
    const png = (c) => c.toDataURL('image/png');
    function texPng(t) {      // (a canvas or image texture as a PNG, drawn at its own size)
      const im = t.image; if (im.toDataURL) return png(im);
      const c = document.createElement('canvas'); c.width = im.width; c.height = im.height; c.getContext('2d').drawImage(im, 0, 0); return png(c);
    }
    function dataPng(t) {      // (a DataTexture, its first row at the bottom as WebGL reads it back: flipped to an image's order)
      const { width: w, height: h, data: d } = t.image, c = document.createElement('canvas'); c.width = w; c.height = h;
      const x = c.getContext('2d'), D = x.createImageData(w, h);
      for (let y = 0; y < h; y++) D.data.set(d.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
      x.putImageData(D, 0, 0); return png(c);
    }
    const bump = M.bumpMap, v3 = (v) => v.toArray(), col = (c) => [c.r, c.g, c.b];
    const ears = (S.ears || []).filter((m) => m.visible && m.geometry.index);
    return {
      head: mesh(S.head, ['uv1', 'skinRest', 'skinZones', 'sssMask']),
      restBaked: S.B && S.B.ref ? Array.from(S.B.ref) : null,      // (the rest the patterns were baked at, which every shape keeps)
      eyes: S.eyes.map((m) => Object.assign(mesh(m), { map: texPng(m.material.map),
        mat: { roughness: m.material.roughness, clearcoat: m.material.clearcoat, clearcoatRoughness: m.material.clearcoatRoughness, ior: m.material.ior, specularIntensity: m.material.specularIntensity } })),
      ears: ears.map((m) => Object.assign(mesh(m), { mat: { color: col(m.material.color), roughness: m.material.roughness } })),
      maps: {
        face: png(S.face), mask: png(S.mask), lip: texPng(U.lipMap.value), rough: texPng(M.roughnessMap),
        pores: texPng(bump),
        skinT0: U.skinT0.value ? dataPng(U.skinT0.value) : null, skinT1: U.skinT1.value ? dataPng(U.skinT1.value) : null, skinT2: U.skinT2.value ? dataPng(U.skinT2.value) : null
      },
      poresUV: { channel: bump.channel, repeat: [bump.repeat.x, bump.repeat.y], offset: [bump.offset.x, bump.offset.y], flipY: bump.flipY, bumpScale: M.bumpScale },
      skinMat: { roughness: M.roughness, ior: M.ior, specularIntensity: M.specularIntensity, sheen: M.sheen, sheenRoughness: M.sheenRoughness, sheenColor: col(M.sheenColor),
        clearcoat: M.clearcoat, clearcoatRoughness: M.clearcoatRoughness },
      uniforms: { skinGain: (typeof skinGain !== 'undefined') ? null : null, lipWrap: v3(U.lipWrap.value), lipCoat: v3(U.lipCoat.value), lipCoatBump: U.lipCoatBump.value, lipBaseSpec: U.lipBaseSpec.value,
        skinWrapRGB: v3(U.skinWrapRGB.value), skinLayers: U.skinLayers.value, skinBaked: U.skinBaked.value },
      P: __fc.P
    };
  });
  const maps = data.maps; delete data.maps; data.files = {};
  for (const [k, url] of Object.entries(maps)) {
    if (!url) continue;
    const f = k + '.png'; fs.writeFileSync(path.join(out, f), Buffer.from(url.split(',')[1], 'base64')); data.files[k] = f;
  }
  data.eyes.forEach((e, k) => { const f = `eye${k}.png`; fs.writeFileSync(path.join(out, f), Buffer.from(e.map.split(',')[1], 'base64')); e.map = f; });
  fs.writeFileSync(path.join(out, 'face.json'), JSON.stringify(data));
  console.log('wrote', out, Object.keys(data.files).join(' '));
  await b.close();
})();
