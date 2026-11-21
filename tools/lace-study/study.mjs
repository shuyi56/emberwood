// The lace and pleating study: solve each strip, inspect its contours, cache its folds, sweep the settings.
//
//   node tools/lace-study/study.mjs cases  [--out DIR] [--fresh]     the study's strips, solved (or from the cache)
//   node tools/lace-study/study.mjs sweep  [--out DIR] [--only NAME]  one setting moved at a time, numbers per run
//
// Every solve is inspected before anything renders it (the project's contour rule): sections across the strip at
// 0, 25, 50, 75 and 100% of its depth (spread, ripple, waves, the sharpest turn off a fold), the threads' stretch
// against the cut, the folds' angles near the heading and at the scallop, and the flags (perfectly round, flat,
// kinked, stretched, blown up). contours.py draws the sheet from the JSON written here.
//
// The fold cache: a solved strip's points are saved under a key made of its shape and the solver's settings
// (lace.cacheKey), in /mnt/project-files/lace-study/cache/, and loaded on the next run with the same key, so a
// render of the same strip shows exactly the folds that were inspected. --fresh solves again.
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import * as L from './lace.js';

const arg = (name, def) => { const i = process.argv.indexOf('--' + name); return i > 0 ? process.argv[i + 1] : def; };
const flag = (name) => process.argv.includes('--' + name);
const OUT = arg('out', '/mnt/project-files/lace-study/runs');
const CACHE = arg('cache', '/mnt/project-files/lace-study/cache');
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(CACHE, { recursive: true });

// The study's strips. The engageante is a frill sewn inside a sleeve's end and hanging round the forearm (a
// sleeve opening about 24 cm round, a frill 6 cm deep); the jabot is a frill down the centre front; the collar
// frill stands on a neckband 36 cm round.
export const CASES = {
  'cuff-knife':     { kind: 'knife', mount: 'cuff', pitch: 1.5, pleatDepth: 1.5, finished: 24, depth: 6, res: 0.3, press: 0.6 },
  'cuff-box':       { kind: 'box', mount: 'cuff', pitch: 1.5, pleatDepth: 0.75, finished: 24, depth: 6, res: 0.3, press: 0.6 },
  'cuff-accordion': { kind: 'accordion', mount: 'cuff', pitch: 2.0, ratio: 2.5, finished: 24, depth: 6, res: 0.3, press: 0.6, substep: 1 / 480 },
  'cuff-flute':     { kind: 'flute', mount: 'cuff', radius: 0.35, finished: 24, depth: 6, res: 0.3, press: 0.8, lace: 'starched' },
  'cuff-gather':    { kind: 'gather', mount: 'cuff', ratio: 2.0, stitch: 2.0, finished: 24, depth: 6, res: 0.3 },
  'jabot-gather':   { kind: 'gather', mount: 'jabot', ratio: 2.0, stitch: 2.0, finished: 20, depth: 6, res: 0.3 },
  'jabot-knife':    { kind: 'knife', mount: 'jabot', pitch: 1.5, pleatDepth: 1.5, finished: 20, depth: 6, res: 0.3, press: 0.6 },
  // the pleated skirt: worsted wool knife-pleated onto a hip yoke (96 cm round the hips), 50 cm from the hip line to
  // the hem (a knee-length skirt with a 20 cm yoke), pleats 3 cm wide and 3 cm deep, hard-pressed (at press 0.7 the
  // inner layers sag onto the hips and the contact tears them 26%; at 1, 0.8% worst)
  'skirt-knife':    { kind: 'knife', mount: 'skirt', lace: 'wool', pitch: 3, pleatDepth: 3, finished: 96, depth: 50, rows: 67, res: 0.75, press: 1, headingWidth: 1.0 },
  // the same skirt on the game's base figure (figure.js, from blender/base/female.py): the yoke's seam at her widest
  // hip, her trunk and legs the body; the bench's fast settings (4.5 cm pleats, a 1.5 by 2 cm mesh, 2 passes, 120 Hz)
  'skirt-figure':   { kind: 'knife', mount: 'skirt', body: 'figure', yokeZ: 32, lace: 'wool', pitch: 4.5, pleatDepth: 4.5, depth: 42, rows: 22, res: 1.5, press: 1, headingWidth: 1.0, passes: 2, substep: 1 / 120 },
  'collar-flute':   { kind: 'flute', mount: 'collar', radius: 0.35, finished: 36, depth: 4, res: 0.3, press: 0.8, lace: 'starched' },
};

const SETTLE_TOL = 2e-4;           // cm a step: the fastest point slower than this for 20 frames in a row is at rest
const MOST = 600;                  // frames (10 s): a run that has not settled by then is reported as not settled
const DYNAMIC = 180;               // frames (3 s) at the lace's own damping before the settle
const SETTLE_DAMPING = 0.3;        // the settle's damping (fabric-core settle(): 0.35)

export function solve(name, o, { fresh = false } = {}) {
  const S = L.buildStrip({ ...o, cuffRadius: (o.finished ?? 24) / (2 * Math.PI) });
  const key = crypto.createHash('sha1').update(L.cacheKey(S)).digest('hex').slice(0, 16);
  const file = path.join(CACHE, `${name}-${key}.json`);
  const curve = [];
  let frames = 0, settled = null, from = 'solve', ms = 0;
  if (!fresh && fs.existsSync(file)) {
    const c = JSON.parse(fs.readFileSync(file, 'utf8'));
    S.f.pos.set(c.pos); S.f.prev.set(c.pos);
    frames = c.frames; settled = c.settled; from = 'cache'; curve.push(...(c.curve || []));
  } else {
    let quiet = 0;
    const t0 = performance.now(), was = S.f.settings.damping;
    for (frames = 1; frames <= MOST; frames++) {
      // the first DYNAMIC frames as they fall (what the viewer shows), then run to rest with the speed damped hard,
      // as the solver's settle() does for a drape: lace at 3% damping swings for many seconds
      if (frames === DYNAMIC + 1) S.f.settings.damping = SETTLE_DAMPING;
      const fr = L.frame(S);
      const st = frames % 10 === 0 ? L.strains(S) : null;
      curve.push({ frame: frames, speed: fr.speed, strain: st ? st.max : undefined });
      if (!Number.isFinite(fr.speed)) break;
      quiet = frames > DYNAMIC && fr.speed < SETTLE_TOL ? quiet + 1 : 0;
      if (quiet >= 20) { settled = frames; break; }
    }
    S.f.settings.damping = was;
    ms = (performance.now() - t0) / Math.max(1, frames);
    frames = Math.min(frames, MOST);
  }
  const st = L.strains(S), sec = L.sections(S), ch = L.creaseHold(S), flags = L.contourFlags(S, sec, st);
  if (settled == null) flags.push(`not settled in ${MOST} frames (fastest point ${curve.length ? curve[curve.length - 1].speed.toExponential(1) : '?'} cm a step)`);
  // cache only a sound solve: settled, not blown up
  const sound = settled != null && !flags.some((f) => f.startsWith('blew up'));
  if (from === 'solve' && sound) fs.writeFileSync(file, JSON.stringify({ key, name, o, pos: Array.from(S.f.pos), frames, settled, curve }));
  const round = L.isRound(o);
  return {
    S, name, key, from, file: sound || from === 'cache' ? file : null, frames, settled, msPerFrame: ms,
    numbers: {
      points: S.f.n, cut: S.prof.cut, finished: S.prof.finished, ratio: S.prof.ratio,
      strainMaxPct: 100 * st.max, strainP99Pct: 100 * st.p99, strainMeanPct: 100 * st.mean,
      foldTopDeg: ch.topDeg, foldBottomDeg: ch.bottomDeg,
      flareAtScallop: round ? sec[4].mean / sec[0].mean : sec[4].spread / Math.max(1e-6, sec[0].spread),
      rippleAtScallopCm: sec[4].ripple, wavesAtScallop: sec[4].lobes,
      sections: sec.map((s) => ({ share: s.share, mean: s.mean, ripple: s.ripple, lobes: s.lobes, kinkDeg: s.kinkDeg, spread: s.spread })),
    },
    flags, curve, perPoint: Array.from(st.perPoint),
  };
}

// what the renders and the contour sheet read: the mesh, its parts (the colour code), the sections, the strain
function dump(R, out) {
  const S = R.S, f = S.f;
  const parts = Array.from(S.part);
  for (const c of S.creaseCols) for (let r = 0; r < S.rows; r++) { parts[S.idx(c, r)] = 4; if (S.span(c) === 2) parts[S.idx(c + 1, r)] = 4; }
  const sec = L.sections(S);
  const data = {
    name: R.name, key: R.key, from: R.from, cache: R.file, o: S.p, mat: S.mat, cols: S.cols, rows: S.rows,
    pos: Array.from(f.pos, (v) => +v.toFixed(4)), tris: Array.from(f.tris), flat: Array.from(S.flat, (v) => +v.toFixed(4)),
    parts, partNames: ['net ground', 'motif', 'cordonnet and scallop', 'heading (stitched)', 'pressed fold'],
    strain: R.perPoint.map((v) => +v.toFixed(5)), sections: sec.map((s) => ({ share: s.share, pts: s.pts.map((p) => p.map((v) => +v.toFixed(3))) })),
    numbers: R.numbers, flags: R.flags, curve: R.curve, frames: R.frames, settled: R.settled, msPerFrame: R.msPerFrame,
  };
  fs.writeFileSync(path.join(out, `${R.name}.json`), JSON.stringify(data));
}

function line(R) {
  const n = R.numbers;
  return `${R.name.padEnd(16)} ${R.from.padEnd(5)} pts ${String(n.points).padStart(5)} ratio ${n.ratio.toFixed(2)} settled ${String(R.settled ?? '-').padStart(4)} ` +
    `strain max ${n.strainMaxPct.toFixed(2)}% p99 ${n.strainP99Pct.toFixed(2)}% mean ${n.strainMeanPct.toFixed(3)}% folds ${n.foldTopDeg.toFixed(0)}/${n.foldBottomDeg.toFixed(0)} ` +
    `flare ${n.flareAtScallop.toFixed(2)} ripple ${n.rippleAtScallopCm.toFixed(2)} waves ${n.wavesAtScallop}` + (R.flags.length ? `  FLAGS: ${R.flags.join('; ')}` : '');
}

// The sweep: one setting moved at a time from a base strip, the rest held (the base is the engageante's knife
// pleat and its gather). Each run gives the shape numbers and the stability numbers.
export const SWEEPS = {
  'knife-press':     { base: 'cuff-knife', key: 'press', values: [0, 0.25, 0.5, 0.75, 1] },
  'knife-bend':      { base: 'cuff-knife', key: 'bendingLength', values: [0.5, 1.0, 1.5, 2.5] },
  'knife-pitch':     { base: 'cuff-knife', key: 'pitch', values: [1.0, 1.5, 2.0, 3.0], also: (v) => ({ pleatDepth: v }) },
  'knife-gravity':   { base: 'cuff-knife', key: 'gravity', values: [490, 981, 1962] },
  'gather-ratio':    { base: 'cuff-gather', key: 'ratio', values: [1.25, 1.5, 2.0, 2.5, 3.0] },
  'gather-bend':     { base: 'cuff-gather', key: 'bendingLength', values: [0.5, 1.0, 1.5, 2.5] },
  'gather-stitch':   { base: 'cuff-gather', key: 'stitch', values: [0.8, 1.2, 1.6, 2.0, 2.4] },
  'stab-stretchK':   { base: 'cuff-knife', key: 'stretchK', values: [0, 30, 100, 300, 600] },
  'stab-passes':     { base: 'cuff-knife', key: 'passes', values: [1, 2, 4] },
  'stab-iterations': { base: 'cuff-knife', key: 'iterations', values: [1, 2, 4, 8] },
  'stab-substep':    { base: 'cuff-knife', key: 'substep', values: [1 / 120, 1 / 240, 1 / 480] },
  'stab-res':        { base: 'cuff-knife', key: 'res', values: [0.25, 0.3, 0.4, 0.5] },
};

if (process.argv[1] && process.argv[1].endsWith('study.mjs')) {
  const mode = process.argv[2] || 'cases';
  if (mode === 'cases') {
    const only = arg('only', null);
    // (a run of some cases keeps the others' rows)
    const sumFile = path.join(OUT, 'cases-summary.json');
    const summary = only && fs.existsSync(sumFile) ? JSON.parse(fs.readFileSync(sumFile, 'utf8')) : {};
    for (const [name, o] of Object.entries(CASES)) {
      if (only && !only.split(',').includes(name)) continue;
      const R = solve(name, o, { fresh: flag('fresh') });
      dump(R, OUT);
      console.log(line(R));
      summary[name] = { key: R.key, from: R.from, settled: R.settled, frames: R.frames, msPerFrame: R.msPerFrame, ...R.numbers, sections: undefined, flags: R.flags };
    }
    fs.writeFileSync(sumFile, JSON.stringify(summary, null, 1));
  } else if (mode === 'sweep') {
    const only = arg('only', null), rows = [];
    for (const [sw, def] of Object.entries(SWEEPS)) {
      if (only && !only.split(',').includes(sw)) continue;
      for (const val of def.values) {
        const o = { ...CASES[def.base], [def.key]: val, ...(def.also ? def.also(val) : {}) };
        const name = `${sw}-${typeof val === 'number' ? +val.toFixed(4) : val}`;
        const R = solve(name, o, { fresh: flag('fresh') });
        console.log(line(R));
        rows.push({ sweep: sw, key: def.key, value: val, settled: R.settled, frames: R.frames, msPerFrame: R.msPerFrame, flags: R.flags, ...R.numbers, sections: undefined });
      }
    }
    fs.writeFileSync(path.join(OUT, `sweep${only ? '-' + only.replace(/,/g, '+') : ''}.json`), JSON.stringify(rows, null, 1));
  }
}
