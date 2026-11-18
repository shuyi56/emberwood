// Export a face from the face creator for the Blender figures: every setting (head shape, Sex, eyes, brows, lips,
// skin, make-up, hair), the colours the creator derives from them, and the creator's own head surface sampled on a
// grid, so blender/base/creator.py builds exactly the head the creator shows (no slider is ported by hand).
//
// usage: node export-face.cjs <face-creator.html> <out.json.gz> [--face NAME] [--settings FILE.json]
//   --face      a head preset of the creator's (Ranger, Rogue, Warden, Knight, ...), applied first
//   --settings  the creator's "Copy settings (JSON)" block saved to a file, applied after the preset
//
// The surface is the creator's headPoint(z, th) in head heights (x across, y back, z up from the chin; th the angle
// round the head, 0 straight forward), on NZ heights from the neck to the crown and NT angles round, stored as
// integers of 1e-5 head heights, differenced along each row (they compress to a few hundred kB).
const { chromium } = require('playwright');
const zlib = require('zlib'), fs = require('fs'), path = require('path'), os = require('os');
const NZ = 566, NT = 1440;
(async () => {
  const a = process.argv.slice(2);
  const opt = k => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : null; };
  const page = a[0], out = a[1], face = opt('--face'), settingsFile = opt('--settings');
  if (!page || !out) { console.log('usage: node export-face.cjs <face-creator.html> <out.json.gz> [--face NAME] [--settings FILE]'); process.exit(2); }
  // (the creator's head functions live in a module; a copy of the page hands them to the exporter on window.__fc)
  let html = fs.readFileSync(page, 'utf8');
  const hook = 'window.__fc = {';
  if (html.indexOf(hook) < 0) { console.log('the creator page has no window.__fc'); process.exit(1); }
  html = html.replace(hook, hook + ' headPoint: headPoint, OUT: OUT, NECK: NECK, HH: HH, applyFace: applyFace, FACES: FACES, ' +
    'DEFAULTS: DEFAULTS, lipFamily: lipFamily, rampFor: rampFor, getP: function () { return P; }, ');
  const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'fc-')), 'fc.html');
  fs.writeFileSync(tmp, html);
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 320, height: 240 } });
  p.on('pageerror', e => { console.log('pageerror:', e.message); process.exit(1); });
  await p.goto('file://' + tmp);
  await p.waitForFunction(() => document.getElementById('busy').hidden, null, { timeout: 300000 });
  const settings = settingsFile ? JSON.parse(fs.readFileSync(settingsFile, 'utf8')) : null;
  const r = await p.evaluate(([face, settings, NZ, NT]) => {
    const F = window.__fc;
    if (face) { if (!F.FACES[face]) return { error: 'no such face: ' + face + ' (have ' + Object.keys(F.FACES).join(', ') + ')' }; F.applyFace(face); }
    const P = F.getP();
    if (settings) Object.keys(F.DEFAULTS).forEach(k => { if (k in settings) P[k] = settings[k]; });
    // the colours the creator derives (as its "Copy ranger.py values"): the lips' family with the lipstick mixed in,
    // and the iris ramp
    const fam = F.lipFamily(), mix = (c, d, t) => c.map((v, i) => v + (d[i] - v) * t);
    const lips = {};
    ['MID', 'UP', 'HI', 'DEEP'].forEach(k => { lips[k] = (P.lipstickAmt > 0 ? mix(fam[k], fam.LS, P.lipstickAmt) : fam[k]).map(v => Math.round(Math.max(0, Math.min(255, v)))); });
    const iris = F.rampFor().map(c => c.map(v => Math.round(v)));
    // the head's surface
    const z0 = F.NECK, xs = [], ys = [], zs = [], O = F.OUT;
    for (let i = 0; i < NZ; i++) {
      const z = z0 + (1 - z0) * i / (NZ - 1);
      for (let j = 0; j < NT; j++) {
        F.headPoint(z, -Math.PI + 2 * Math.PI * j / NT);
        xs.push(Math.round(O[0] * 1e5)); ys.push(Math.round(O[1] * 1e5)); zs.push(Math.round(O[2] * 1e5));
      }
    }
    // and exact points off the grid, over the face and round the head, for checking the Blender side's lookup
    let sd = 12345; const rnd = () => { sd = (sd * 1103515245 + 12345) % 2147483648; return sd / 2147483648; };
    const check = [];
    for (let k = 0; k < 3000; k++) {
      const face = k < 2000, z = face ? 0.0 + 0.75 * rnd() : z0 + (1 - z0) * rnd(), th = face ? (rnd() - 0.5) * 1.6 : (rnd() * 2 - 1) * Math.PI;
      F.headPoint(z, th); check.push([z, th, O[0], O[1], O[2]]);
    }
    return { check: check, face: face, settings: JSON.parse(JSON.stringify(P)), derived: { lips: lips, iris: iris }, xs: xs, ys: ys, zs: zs, z0: z0 };
  }, [face, settings, NZ, NT]);
  await b.close();
  if (r.error) { console.log(r.error); process.exit(1); }
  const diff = v => { const o = new Array(v.length); for (let i = 0; i < v.length; i++) o[i] = (i % NT === 0) ? v[i] : v[i] - v[i - 1]; return o; };
  const doc = { format: 'emberwood-creator-face/1', face: r.face, source: path.basename(page), settings: r.settings, derived: r.derived,
    check: r.check, surface: { nz: NZ, nt: NT, z0: r.z0, z1: 1.0, th0: -Math.PI, unit: 1e-5, x: diff(r.xs), y: diff(r.ys), z: diff(r.zs) } };
  fs.writeFileSync(out, zlib.gzipSync(JSON.stringify(doc), { level: 9 }));
  console.log(`${r.face || 'settings'}: surface ${NZ} x ${NT}, ${fs.statSync(out).size} bytes -> ${out}`);
})();
