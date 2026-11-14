// Paints every sprite sheet once and stores it as a PNG in public/sprites/, so the bench and the world
// load finished images instead of painting sheets (about a second each) at runtime.
//
//   node scripts/build-sprites.mjs           paint only if the sprite code changed since the last run
//   node scripts/build-sprites.mjs --force   paint everything
//
// (Painting is limited by memory more than cores: past about 8 threads it gets no faster. SPRITE_THREADS=n overrides.)
// Runs before `npm run dev` and `npm run build`; during `dev` the Vite plugin in vite.config.js reruns it
// whenever src/shared changes. A hash of the sprite sources is kept in public/sprites/manifest.json, both
// to skip unchanged runs and so pages can request the current version of each file past the cache.
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { createHash } from 'node:crypto';
import { availableParallelism } from 'node:os';
import { deflateSync } from 'node:zlib';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SHARED = path.join(ROOT, 'src', 'shared');
const OUT = path.join(ROOT, 'public', 'sprites');
const shared = (file) => pathToFileURL(path.join(SHARED, file)).href;

// ---------- PNG (8-bit RGBA, no filtering) ----------
const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc32 = (buf) => { let c = ~0; for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8); return ~c >>> 0; };
function chunk(type, data) {
  const len = Buffer.alloc(4), body = Buffer.concat([Buffer.from(type, 'ascii'), data]), crc = Buffer.alloc(4);
  len.writeUInt32BE(data.length); crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}
// rows: a function giving image row y (top first) as a Uint8Array of W * 4 bytes
function encodePNG(W, H, rows) {
  const raw = Buffer.alloc((W * 4 + 1) * H);
  for (let y = 0; y < H; y++) raw.set(rows(y), y * (W * 4 + 1) + 1);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
}

// ---------- worker: paint the sheets it is given ----------
async function paint(files) {
  const { paintSheet } = await import(shared('sprites.js'));
  const { allSheets } = await import(shared('sheets.js'));
  const wanted = new Set(files);
  for (const s of allSheets().filter((s) => wanted.has(s.file))) {
    const { data, W, H } = paintSheet(s.spec);
    // paintSheet stores rows bottom first (for a GL texture); a PNG is top first
    const png = encodePNG(W, H, (y) => data.subarray((H - 1 - y) * W * 4, (H - y) * W * 4));
    fs.writeFileSync(path.join(OUT, s.file), png);
    parentPort.postMessage(s.file);
  }
}

// ---------- main ----------
function sourceHash(sheets) {
  const h = createHash('sha256');
  const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name)).forEach((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p);
    // (the browser-side loader cannot change what is painted)
    else if (e.name.endsWith('.js') && e.name !== 'sheet-loader.js') { h.update(path.relative(SHARED, p).replace(/\\/g, '/')); h.update(fs.readFileSync(p)); }
  });
  walk(SHARED);
  h.update(fs.readFileSync(fileURLToPath(import.meta.url)));
  // sheets rendered elsewhere (the Blender ranger) count too, so a new render is copied in
  for (const s of sheets) if (s.source && fs.existsSync(path.join(ROOT, s.source))) h.update(fs.readFileSync(path.join(ROOT, s.source)));
  return h.digest('hex').slice(0, 16);
}

async function main() {
  const force = process.argv.includes('--force');
  const { allSheets } = await import(shared('sheets.js'));
  const sheets = allSheets(), hash = sourceHash(sheets), manifestPath = path.join(OUT, 'manifest.json');
  let old = null;
  try { old = JSON.parse(fs.readFileSync(manifestPath, 'utf8')); } catch { /* first run */ }
  if (!force && old && old.hash === hash && sheets.every((s) => s.source || fs.existsSync(path.join(OUT, s.file)))) {
    console.log(`sprites: up to date (${sheets.length} sheets)`);
    return;
  }
  fs.mkdirSync(OUT, { recursive: true });
  // rendered sheets are copied in; a missing render is left out, and the page paints a fallback
  for (const s of sheets.filter((s) => s.source)) {
    const src = path.join(ROOT, s.source);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(OUT, s.file));
    else console.warn(`sprites: ${s.source} is missing; ${s.file} will be painted in the page instead`);
  }
  const toPaint = sheets.filter((s) => !s.source);
  const t0 = Date.now(), n = Math.max(1, Math.min(toPaint.length, +process.env.SPRITE_THREADS || Math.min(8, availableParallelism() - 1)));
  console.log(`sprites: painting ${toPaint.length} sheets on ${n} threads...`);
  const groups = Array.from({ length: n }, () => []);
  toPaint.forEach((s, i) => groups[i % n].push(s.file));
  let done = 0;
  await Promise.all(groups.map((files) => new Promise((resolve, reject) => {
    const w = new Worker(fileURLToPath(import.meta.url), { workerData: files });
    w.on('message', () => { done++; });
    w.on('error', reject);
    w.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`sprite worker exited with ${code}`))));
  })));
  // remove sheets left over from styles or characters that no longer exist
  const keep = new Set(sheets.map((s) => s.file).concat('manifest.json'));
  fs.readdirSync(OUT).filter((f) => f.endsWith('.png') && !keep.has(f)).forEach((f) => fs.unlinkSync(path.join(OUT, f)));
  fs.writeFileSync(manifestPath, JSON.stringify({ hash, sheets: sheets.map(({ key, hair, file }) => ({ key, hair, file })) }, null, 2) + '\n');
  console.log(`sprites: painted ${done} sheets in ${((Date.now() - t0) / 1000).toFixed(1)}s -> public/sprites/`);
}

if (isMainThread) main().catch((e) => { console.error(e); process.exit(1); });
else paint(workerData).catch((e) => { console.error(e); process.exit(1); });
