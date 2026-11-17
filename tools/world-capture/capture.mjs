// Capture the leader walking in the real ember world (index.html), frame by frame.
//
//   npm run dev                      (or any server for the world on --url)
//   node tools/world-capture/capture.mjs --out /tmp/walk [--views side,front3q,back3q,front,back]
//        [--frames 24] [--fps 60] [--dist 9] [--member ranger] [--size 300x340] [--scale 2]
//
// The page is opened with ?capture, which exposes window.__ember (see "Capture hook" in src/world/main.js).
// The page's clock is driven from here: requestAnimationFrame and performance.now are replaced, and each frame
// advances game time by exactly 1/fps s, so the frames are evenly spaced in game time however slow the
// software renderer is. For each view the leader is placed on open ground, the camera is set at the game's own
// zoom and tilt for --dist and locked onto her, and she walks a straight line across the shot.
// Writes <out>/<view>/fNNN.png, a contact sheet <out>/<view>.png per view, and <out>/frames.json.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  try { return require('playwright'); } catch { /* fall back to the global install */ }
  const root = execSync('npm root -g').toString().trim();
  return require(path.join(root, 'playwright'));
}
const { chromium } = loadPlaywright();

const arg = (name, def) => {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : def;
};
const out = path.resolve(arg('out', 'world-capture'));
const url = arg('url', 'http://127.0.0.1:5173/');
const views = arg('views', 'side,front3q,back3q').split(',');
const N = Number(arg('frames', 24)), FPS = Number(arg('fps', 60));
const dist = Number(arg('dist', 9));
const member = arg('member', 'ranger');
const [W, H] = arg('size', '300x340').split('x').map(Number);
const scale = Number(arg('scale', 2));
const settle = Number(arg('settle', 0.6));        // seconds of walking before the first frame
// the heading of the walk relative to the camera: 0 walks straight at it
const REL = { front: 0, front3q: Math.PI / 4, side: Math.PI / 2, back3q: Math.PI * 3 / 4, back: Math.PI };
const HEADING = 0.6;                               // the world heading of every walk (only the camera turns)

fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 960, height: 640 }, deviceScaleFactor: scale });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
await page.addInitScript((m) => {
  localStorage.setItem('emberwood.party', JSON.stringify({ members: [m], hair: {} }));
  let fake = 0; const q = [];
  performance.now = () => fake;
  window.requestAnimationFrame = (cb) => { q.push(cb); return q.length; };
  // all but the last of n steps skip the render: only the frame that is captured is drawn
  window.__step = (ms, n = 1) => {
    for (let k = 0; k < n; k++) {
      if (window.__ember) window.__ember.skipRender = k < n - 1;
      fake += ms; q.splice(0).forEach((cb) => cb(fake));
    }
  };
}, member);
await page.goto(url + (url.includes('?') ? '&' : '?') + 'capture', { waitUntil: 'load' });
await page.evaluate(() => {
  const h = document.getElementById('help'); if (h) { h.open = false; h.style.visibility = 'hidden'; }
  for (const el of document.querySelectorAll('.hud, #toolbar, #toast, nav, footer')) el.style.visibility = 'hidden';
});
// wait for the model to load
for (let k = 0; k < 120; k++) {
  await page.evaluate(() => window.__step(1000 / 30));
  if (await page.evaluate(() => window.__ember.state().loaded)) break;
  await page.waitForTimeout(1000);
}
const meta = { url, member, fps: FPS, frames: N, dist, views: {} };
const walkLen = 6 + (settle + N / FPS) * 7;
const [sx0, sz0] = arg('near', '-6,6').split(',').map(Number);
const spot = await page.evaluate(([h, len, x, z, c]) => window.__ember.findSpot(x, z, h, len, c), [HEADING, walkLen, sx0, sz0, Number(arg('clear', 5))]);
console.log('walk from', spot);
if (!spot) throw new Error('no open ground for the walk');
for (const view of views) {
  const dir = path.join(out, view);
  fs.mkdirSync(dir, { recursive: true });
  await page.evaluate(([s, h, yaw, d]) => {
    window.__ember.place(s.x, s.z, h);
    window.__ember.camera(yaw, d, true);
  }, [spot, HEADING, HEADING - REL[view], dist]);
  await page.evaluate(() => window.__step(1000 / 30, 40));      // stand, and let the camera's tilt settle
  await page.evaluate(([h, len]) => window.__ember.walk(h, len), [HEADING, walkLen]);
  await page.evaluate(([ms, n]) => window.__step(ms, n), [1000 / FPS, Math.round(settle * FPS)]);
  const frames = [];
  for (let i = 0; i < N; i++) {
    await page.evaluate((ms) => window.__step(ms), 1000 / FPS);
    const c = await page.evaluate(() => window.__ember.leaderOnScreen());
    const st = await page.evaluate(() => window.__ember.state());
    frames.push({ i, stride: st.stride, moving: st.moving });
    await page.screenshot({ path: path.join(dir, `f${String(i).padStart(3, '0')}.png`), clip: { x: c.x - W / 2, y: c.y - H * 0.55, width: W, height: H } });
  }
  meta.views[view] = { frames };
  console.log('captured', view);
}
fs.writeFileSync(path.join(out, 'frames.json'), JSON.stringify(meta, null, 1));
await browser.close();
