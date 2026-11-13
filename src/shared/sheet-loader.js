// Loads the pre-painted sprite sheets (see sheets.js and scripts/build-sprites.mjs) in the browser.
// Each sheet comes back as an image, top row first: facing rows down, frames across. If a file is missing
// (the build step has not run) the sheet is painted here instead, with a warning, so pages still work.
import { paintSheet } from './sprites.js';
import { SHEET_DIR, sheetFile } from './sheets.js';

const BASE = import.meta.env.BASE_URL + SHEET_DIR + '/';
// the manifest's hash tags each request, so a repainted sheet is never served from a stale cache
let version = null;
const getVersion = () => version || (version = fetch(BASE + 'manifest.json', { cache: 'no-store' })
  .then((r) => (r.ok ? r.json() : null)).then((m) => (m && m.hash) || '').catch(() => ''));

// paint a sheet live into a canvas (the fallback)
function paintedCanvas(spec) {
  const sh = paintSheet(spec), cv = document.createElement('canvas');
  cv.width = sh.W; cv.height = sh.H;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(sh.W, sh.H);
  for (let y = 0; y < sh.H; y++) img.data.set(sh.data.subarray((sh.H - 1 - y) * sh.W * 4, (sh.H - y) * sh.W * 4), y * sh.W * 4);
  ctx.putImageData(img, 0, 0);
  return cv;
}

const cache = new Map();
// key and hair name the sheet's file; spec is only used to paint it if the file cannot be loaded
function loadSheet(key, hair, spec) {
  const file = sheetFile(key, hair);
  if (!cache.has(file)) {
    cache.set(file, getVersion().then((v) => new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => {
        console.warn(`sprite sheet ${file} is missing; painting it at runtime (run \`npm run sprites\`)`);
        resolve(paintedCanvas(spec));
      };
      img.src = BASE + file + (v ? '?v=' + v : '');
    })));
  }
  return cache.get(file);
}

export { loadSheet };
