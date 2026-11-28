// Shots of a groom in the Face Creator's hair fidelity mode (the page made by apply-hair-fidelity.py), from fixed
// cameras, so a before and an after can be laid side by side (the shots the hair threads have been sending John).
//
//   node tools/hair/creator_shots.cjs PAGE.html OUT_PREFIX STYLE [FACE] [MODE] [VIEWS]
//
// MODE: strands (default) | creator | lod0 ... ; VIEWS: comma list of name=az/el/dist/ty or a preset name
// (front, three, profile, back3q, back, drape, close3q, closeback). Writes OUT_PREFIX-<view>.png at 900 x 900.
const { chromium } = require('playwright');
const path = require('path');
const PRESET = {
  front: { az: 0, el: 4, dist: 44, ty: 0.15 }, three: { az: 40, el: 6, dist: 44, ty: 0.15 },
  profile: { az: 90, el: 3, dist: 44, ty: 0.15 }, back3q: { az: 140, el: 6, dist: 44, ty: 0.15 },
  back: { az: 180, el: 4, dist: 44, ty: 0.15 }, drape: { az: 35, el: 10, dist: 62, ty: -0.25 },
  close3q: { az: 50, el: 4, dist: 24, ty: 0.3 }, closeback: { az: 160, el: 2, dist: 26, ty: 0.05 },
};
(async () => {
  const [page, prefix, style, face, mode0, views0] = process.argv.slice(2);
  const mode = mode0 || 'strands';
  const views = (views0 || 'front,three,profile,back3q,back').split(',');
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1300, height: 950 } });
  p.on('pageerror', e => console.log('pageerror:', e.message));
  await p.goto('file://' + path.resolve(page));
  await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 600000 });
  if (face && face !== '-') {
    await p.locator(`[data-face="${face}"]`).click();
    await p.waitForTimeout(500);
    await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 600000 });
  }
  await p.evaluate(s => { __fc.P.hairStyle = s; const h = document.getElementById('sel-hair'); if (h) h.value = s; __fc.update(); }, style);
  await p.waitForFunction(() => document.getElementById('busy').hidden && window.__hairFidelity, null, { timeout: 600000 });
  await p.evaluate(m => window.__hairFidelity.set(m), mode);
  await p.waitForTimeout(4000);
  for (const v of views) {
    let name = v, cam = PRESET[v];
    if (v.includes('=')) { const [n, s] = v.split('='); const [az, el, dist, ty] = s.split('/').map(Number); name = n; cam = { az, el, dist, ty }; }
    await p.evaluate(c => { __fc.VIEWS._shot = c; __fc.setView('_shot', true); }, cam);
    await p.waitForTimeout(2500);
    await p.locator('#gl').screenshot({ path: `${prefix}-${name}.png` });
    console.log(`${prefix}-${name}.png`);
  }
  await b.close();
})();
