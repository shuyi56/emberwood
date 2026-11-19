// Dump a Face Creator head for the Blender skin bench (blender/base/skin_creator_bench.py): the head, eye and ear
// meshes in Blender's axes (x, y back, z up; model units), the head's layered-skin lookups (rest, zones), its painted
// face and mask, the eyes' paintings, the creator's views as cameras, its lights, and its ambient light (hemisphere and
// room environment) as an equirectangular map (Blender's Environment Texture layout: u = 0.5 - atan2(y, x) / 2pi)
// of what a matte white surface gets from it at each normal. Also writes
// the creator's own render of each view, so the bench can put the two side by side.
//
//   node tools/face-creator/dump-head.cjs <face-creator.html> <out dir> [face preset] [settings JSON]
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const [page, out, face, extra] = process.argv.slice(2);
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1300, height: 1000 } });
  p.on('pageerror', e => { console.log('pageerror:', e.message); process.exit(1); });
  p.on('console', m => { if (m.type() === 'error') console.log(m.text()); });
  await p.goto('file://' + path.resolve(page));
  await p.waitForFunction(() => document.getElementById('busy').hidden && window.__skin, null, { timeout: 600000 });
  if (face) { await p.locator(`[data-face="${face}"]`).first().click(); await p.waitForTimeout(400); await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 600000 }); }
  await p.evaluate(extra => { if (extra) Object.assign(__fc.P, JSON.parse(extra)); __fc.P.hair = false; __fc.update(); }, extra || null);
  // (PLAIN=1: the head in a plain material, its painted face on a matte standard surface, to check the bench's light
  // against the creator's with nothing of either skin in the way; the bench's --plain is the same surface)
  if (process.env.PLAIN) await p.evaluate(plain => { const S = __skin, T = S.THREE; S.head.material = plain === 'white' ? new T.MeshStandardMaterial({ color: 0xbbbbbb, roughness: 0.56, metalness: 0, envMapIntensity: S.head.material.envMapIntensity }) : new T.MeshStandardMaterial({ map: S.head.material.map, vertexColors: true, roughness: 0.56, metalness: 0, envMapIntensity: S.head.material.envMapIntensity }); }, process.env.PLAIN);
  const V = { front: [0, 4, 34, 0.5], three: [35, 6, 34, 0.5], profile: [88, 3, 34, 0.5], cheek: [28, 2, 13, 0.4], nose: [8, 4, 11, 0.42], temple: [55, 4, 12, 0.55] };
  const cams = {};
  for (const [k, v] of Object.entries(V)) {
    cams[k] = await p.evaluate(([k, v]) => {
      __fc.VIEWS['_' + k] = { az: v[0], el: v[1], dist: v[2], ty: v[3] }; __fc.setView('_' + k, true);
      const c = __skin.camera; c.updateMatrixWorld();
      return { fov: c.fov, aspect: c.aspect, matrixWorld: c.matrixWorld.elements.slice() };
    }, [k, v]);
    await p.waitForTimeout(300);
    await p.locator('#gl').screenshot({ path: path.join(out, `creator-${k}.png`) });
  }
  const size = await p.evaluate(() => { const c = __fc.renderer.domElement; return [c.width, c.height]; });
  const data = await p.evaluate(() => {
    const S = __skin, THREE_ = S.head.geometry.attributes.position.constructor;
    const bl = (a, k) => [a[k * 3], -a[k * 3 + 2], a[k * 3 + 1]];      // (three.js (x, y up, z front) to Blender (x, y back, z up))
    function mesh(m, extraAttrs) {
      const g = m.geometry, P = g.attributes.position.array, n = g.attributes.position.count, o = { verts: [], index: g.index ? Array.from(g.index.array) : null };
      for (let k = 0; k < n; k++) o.verts.push(...bl(P, k));
      if (g.attributes.uv) o.uv = Array.from(g.attributes.uv.array);
      if (g.attributes.color) { const C = g.attributes.color, s = C.itemSize; o.color = []; for (let k = 0; k < n; k++) o.color.push(C.array[k * s], C.array[k * s + 1], C.array[k * s + 2]); }
      (extraAttrs || []).forEach(a => { o[a] = Array.from(g.attributes[a].array); });
      return o;
    }
    const L = S.lights, lin = c => [c.r, c.g, c.b];
    const dir = l => { const d = l.position.clone().sub(l.target.position).normalize(); return [d.x, -d.z, d.y]; };
    return {
      head: mesh(S.head, ['skinRest', 'skinZones']),
      eyes: S.eyes.map(m => Object.assign(mesh(m), { map: m.material.map.image.toDataURL('image/png') })),
      ears: S.ears.map(m => mesh(m)),
      face: S.face.toDataURL('image/png'), mask: S.mask.toDataURL('image/png'),
      lights: ['key', 'rim', 'fill'].map(k => ({ name: k, dir: dir(L[k]), color: lin(L[k].color), intensity: L[k].intensity, shadow: L[k].castShadow })),
      hemi: { sky: lin(L.hemi.color), ground: lin(L.hemi.groundColor), intensity: L.hemi.intensity },
      bounce: { dir: [S.bounce.x, -S.bounce.z, S.bounce.y], color: S.U.skinBounceCol.value.toArray() },      // (the skin's warm bounce from below, views.py's)
      skinEnv: S.head.material.envMapIntensity, P: __fc.P, toneExposure: __fc.renderer.toneMappingExposure
    };
  });
  // the ambient light as a map: a matte white sphere lit by the hemisphere and the room alone, seen from the front and
  // the back, linear; each pixel's normal and value binned into Blender's equirectangular layout
  // (two spheres, as the hair bench does: a white one, and a black one whose light is only the room's reflection off
  // the surface, which does not take its colour; both with the skin's roughness)
  data.ambient = await p.evaluate(() => {
    const S = __skin, T = S.THREE, r = __fc.renderer, W = 64, H = 32, R = 256, out = { W, H };
    const sc = new T.Scene(); sc.environment = S.scene.environment; sc.add(S.lights.hemi.clone());
    const rt = new T.WebGLRenderTarget(R, R, { type: T.FloatType });
    const cam = new T.OrthographicCamera(-1, 1, 1, -1, 0.1, 10), tm = r.toneMapping; r.toneMapping = T.NoToneMapping;
    const px = new Float32Array(R * R * 4);
    [['white', 0xffffff], ['black', 0x000000]].forEach(([name, col]) => {
      const ball = new T.Mesh(new T.SphereGeometry(1, 128, 64), new T.MeshStandardMaterial({ color: col, roughness: 0.56, metalness: 0, envMapIntensity: S.head.material.envMapIntensity }));
      sc.add(ball);
      const acc = new Float64Array(W * H * 3), cnt = new Float64Array(W * H);
      [1, -1].forEach(side => {
        cam.position.set(0, 0, 3 * side); cam.lookAt(0, 0, 0); cam.updateMatrixWorld();
        r.setRenderTarget(rt); r.setClearColor(0x000000, 0); r.clear(); r.render(sc, cam); r.readRenderTargetPixels(rt, 0, 0, R, R, px); r.setRenderTarget(null);
        for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
          const u = (x + 0.5) / R * 2 - 1, v = (y + 0.5) / R * 2 - 1, q = 1 - u * u - v * v; if (q < 0.02) continue;
          const nx = u * side, ny = v, nz = Math.sqrt(q) * side;      // (the normal, in three.js axes; seen from behind, x flips)
          const bx = nx, by = -nz, bz = ny;
          const iu = Math.min(W - 1, Math.floor((0.5 - Math.atan2(by, bx) / (2 * Math.PI)) * W)), iv = Math.min(H - 1, Math.floor((Math.atan2(bz, Math.hypot(bx, by)) / Math.PI + 0.5) * H));
          const o = (y * R + x) * 4, k = iv * W + iu;
          acc[k * 3] += px[o]; acc[k * 3 + 1] += px[o + 1]; acc[k * 3 + 2] += px[o + 2]; cnt[k]++;
        }
      });
      sc.remove(ball);
      const o = []; for (let k = 0; k < W * H; k++) for (let c = 0; c < 3; c++) o.push(cnt[k] ? acc[k * 3 + c] / cnt[k] : -1);
      out[name] = o;
    });
    r.toneMapping = tm; r.setClearColor(0x000000, 0);
    return out;
  });
  fs.writeFileSync(path.join(out, 'head.json'), JSON.stringify(Object.assign(data, { cams, size })));
  console.log('wrote', out);
  await b.close();
})();
