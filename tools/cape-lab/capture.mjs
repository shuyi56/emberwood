// Capture the cape bench (cape-lab.html) frame by frame: a strip of frames, a GIF and the loop drift for each
// motion and camera view.
//
//   npm run dev
//   node tools/cape-lab/capture.mjs --out /tmp/cape [--scenarios turn,startstop,walk] [--views back3q,side]
//        [--seconds 7] [--every 3] [--size 420x520] [--settings '{"stiffness":300}'] [--columns 5] [--rows 5] [--solver fabric]
//        [--colliders] [--no-springs] [--body model] [--walk new] [--zoom 2.6 --look-y 50]
//
// --body model hangs the cape on the Warden's own glb instead of the stand-in (the page loads models/warden.glb,
// or the file given by ?model=<path under public/> in --url). --zoom and --look-y frame a close-up (the head and
// hair at 2.6 and 50).
//
// The bench is opened with ?capture, which exposes window.__lab (see "Capture hook" in src/cape-lab/main.js):
// the sim runs in fixed frames of 1/60 s, and only the frames captured are drawn, so a run is the same every
// time however slow the headless renderer is. `--every 3` keeps every third frame (20 a second).
// Writes <out>/<scenario>-<view>/fNNN.png, <out>/<scenario>-<view>.gif, a sheet <out>/<scenario>-<view>.png
// (8 frames across) and <out>/summary.json.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { execSync, execFileSync } from 'child_process';

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
const flag = (name) => process.argv.includes('--' + name);
const out = path.resolve(arg('out', 'cape-capture'));
const url = arg('url', 'http://127.0.0.1:5173/cape-lab.html');
const scenarios = arg('scenarios', 'turn,startstop,walk').split(',');
const views = arg('views', 'back3q,side').split(',');
const seconds = Number(arg('seconds', 0));
const every = Number(arg('every', 3));
const [W, H] = arg('size', '420x520').split('x').map(Number);
const settings = JSON.parse(arg('settings', '{}'));
const columns = Number(arg('columns', 7)), rows = Number(arg('rows', 5));
const zoom = Number(arg('zoom', 1)), lookY = arg('look-y') ? Number(arg('look-y')) : undefined;
const body = arg('body', 'standin');
const walk = arg('walk', 'clip');          // clip (the body's own), new or main (baked walks, src/cape-lab/baked-walk.js)          // standin, or model: the Warden's glb (the page's ?model=)
const LENGTH = { walk: 4, startstop: 6, turn: 7, run: 5, spin: 4 };

fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text()); });
await page.goto(url + (url.includes('?') ? '&' : '?') + 'capture' + (arg('solver') ? '&solver=' + arg('solver') : ''), { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__lab, null, { timeout: 30000 });

const summary = [];
for (const sc of scenarios) {
  for (const view of views) {
    const name = `${sc}-${view}`;
    const dir = path.join(out, name);
    fs.rmSync(dir, { recursive: true, force: true });
    fs.mkdirSync(dir, { recursive: true });
    await page.evaluate((o) => window.__lab.setup(o), {
      settings, columns, rows, body,
      opts: { scenario: sc, view, showColliders: flag('colliders'), showBones: flag('bones'), springs: !flag('no-springs'), showBody: true, zoom, lookY, walk },
    });
    // the loop check settles over a few cycles before the frames are kept
    if (sc === 'walk') await page.evaluate(() => window.__lab.advance(42 * 6));
    const total = Math.round((seconds || LENGTH[sc] || 5) * 60 / every);
    const t0 = Date.now();
    for (let i = 0; i < total; i++) {
      await page.evaluate((n) => { window.__lab.advance(n); window.__lab.render(); }, every);
      await page.locator('#view').screenshot({ path: path.join(dir, `f${String(i).padStart(3, '0')}.png`) });
    }
    const st = await page.evaluate(() => window.__lab.state());
    summary.push({ name, frames: total, drift: st.drift });
    console.log(`${name}: ${total} frames in ${((Date.now() - t0) / 1000).toFixed(1)} s` + (st.drift !== null ? `, loop drift ${st.drift.toFixed(4)}` : ''));
    const fps = 60 / every;
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', path.join(dir, 'f%03d.png'),
      '-vf', `scale=${Math.round(W * 0.75)}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer`,
      path.join(out, name + '.gif')]);
    // a sheet of 8 frames spread over the run
    const pick = Array.from({ length: 8 }, (_, k) => Math.round(k * (total - 1) / 7));
    const inputs = pick.flatMap((k) => ['-i', path.join(dir, `f${String(k).padStart(3, '0')}.png`)]);
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...inputs, '-filter_complex',
      pick.map((_, k) => `[${k}:v]`).join('') + `hstack=inputs=8,scale=iw/2:-1`, path.join(out, name + '.png')]);
  }
}
fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(summary, null, 2));
await browser.close();
