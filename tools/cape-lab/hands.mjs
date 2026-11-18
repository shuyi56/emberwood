// The cape on the Warden's own model, checked in the browser: in each motion, how many frames show a hand or
// bracer through the back of her cape (outside the cape's skinned surface, within 80° of straight behind, so
// the side openings where her arms come out don't count), and how far out the worst one is.
//
//   npm run dev
//   node tools/cape-lab/hands.mjs [--url http://127.0.0.1:5173/cape-lab.html?model=models/warden.glb] [--scenarios walk,turn]
//        [--walk new]
import { createRequire } from 'module';
import path from 'path';
import { execSync } from 'child_process';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  try { return require('playwright'); } catch { /* fall back to the global install */ }
  return require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
}
const { chromium } = loadPlaywright();
const arg = (name, def) => {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : def;
};
const url = arg('url', 'http://127.0.0.1:5173/cape-lab.html');
const walk = arg('walk', 'clip');   // clip (her own), new or main (baked walks)
await (async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto(url + (url.includes('?') ? '&' : '?') + 'capture'); await p.waitForFunction(() => !!window.__lab);
  for (const sc of arg('scenarios', 'walk,startstop,turn,run').split(',')) {
    await p.evaluate(([sc, walk]) => window.__lab.setup({ body: 'model', opts: { scenario: sc, springs: true, walk } }), [sc, walk]);
    const r = await p.evaluate((sc) => {
      const s = window.__lab.sim(), d = s.dummy, V = d.root.position.constructor;
      const cape = d.cape.mesh, hands = []; d.root.traverse(o => { if (o.isSkinnedMesh && /^(Hand|Bracer|Gauntlet)/.test(o.name) && o.visible) hands.push(o); });
      const CZ = -0.5, ang = (x, z) => Math.atan2(x, -(z - CZ)), NA = 48, NY = 30;
      const frames = { run: 300, turn: 420, walk: 300, startstop: 360 }[sc];
      let bad = 0, worst = 0;
      for (let f = 0; f < frames; f++) {
        s.step(); if (f % 3) continue;
        d.root.updateMatrixWorld(true); cape.skeleton.update();
        const inv = d.root.matrixWorld.clone().invert();
        // the cape's outer radius on a grid of angle and height, in her space
        const grid = new Float32Array(NA * NY).fill(-1);
        const pos = cape.geometry.attributes.position, v = new V();
        for (let i = 0; i < pos.count; i++) { v.fromBufferAttribute(pos, i); cape.boneTransform(i, v); v.applyMatrix4(cape.matrixWorld).applyMatrix4(inv);
          const ia = Math.floor((ang(v.x, v.z) + Math.PI) / (2 * Math.PI) * NA), iy = Math.floor((v.y - 15) / 1.4);
          if (iy < 0 || iy >= NY) continue; const k = iy * NA + ((ia % NA) + NA) % NA; grid[k] = Math.max(grid[k], Math.hypot(v.x, v.z - CZ)); }
        let n = 0;
        for (const h of hands) { const hp = h.geometry.attributes.position;
          for (let i = 0; i < hp.count; i += 4) { v.fromBufferAttribute(hp, i); h.boneTransform(i, v); v.applyMatrix4(h.matrixWorld).applyMatrix4(inv);
            const a = ang(v.x, v.z); if (Math.abs(a) > 1.4) continue;   // the cape's back and sides, not its front opening
            const ia = Math.floor((a + Math.PI) / (2 * Math.PI) * NA), iy = Math.floor((v.y - 15) / 1.4); if (iy < 0 || iy >= NY) continue;
            const rc = grid[iy * NA + ia]; if (rc < 0) continue;
            const o = Math.hypot(v.x, v.z - CZ) - rc; if (o > 0.3) { n++; worst = Math.max(worst, o); } } }
        if (n > 5) bad++;
      }
      return `${sc}: frames with a hand through the back of the cape ${bad}/${Math.ceil(frames / 3)}, deepest ${worst.toFixed(2)}`;
    }, sc);
    console.log(r);
  }
  await b.close();
})();
