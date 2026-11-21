// The solver viewer's scenes. Each preset builds a cloth on the project's fabric solver and says which of its
// settings can be moved while it runs (live) and which rebuild it. The lace strip is the study's own; the other three
// are the fabric bench's kinds of test (a drape on a ball, a pressed fold, a collar's roll held at its gorge).
//
// A scene: { f, tris, uv, parts, partNames, partColors, body, bodies (to draw), measure(), sections(), live(key, v),
// flat (the cut pattern, for the strain), edges (the threads, for the strain) }
import { Fabric, MATERIALS } from './fabric-core.js';
import * as L from './lace.js';

const DEG = Math.PI / 180;

// the colour-coded draft's parts: flat, distinct, readable in grey too (values spread from dark to light)
export const LACE_PARTS = {
  names: ['net ground', 'motif', 'cordonnet and scallop', 'heading (stitched)', 'pressed fold'],
  colors: ['#8fb3d9', '#e0b35a', '#5e3070', '#2f6b4f', '#d2523a'],
};

function threadsOf(cols, rows, idx) {
  const e = [];
  for (let c = 0; c < cols; c++) for (let r = 0; r < rows; r++) {
    if (c + 1 < cols) e.push(idx(c, r), idx(c + 1, r));
    if (r + 1 < rows) e.push(idx(c, r), idx(c, r + 1));
  }
  return Int32Array.from(e);
}

// ------------------------------------------------------------------ the lace strip
const lace = {
  id: 'lace', name: 'Pleated lace strip',
  note: 'A lace frill cut flat at the pleat\'s length ratio, folded, stitched along its heading and hung. Move the press to see pressed pleats hold or open; the bend sets how far the lace flares.',
  params: [
    { key: 'kind', label: 'Pleat', type: 'select', options: [['knife', 'Knife'], ['box', 'Box'], ['accordion', 'Accordion'], ['flute', 'Fluted (goffered)'], ['gather', 'Gathered']], value: 'knife' },
    { key: 'mount', label: 'Sewn to', type: 'select', options: [['band', 'Straight band'], ['cuff', 'Cuff (engageante)'], ['collar', 'Standing collar'], ['jabot', 'Jabot (down the front)']], value: 'cuff' },
    { key: 'lace', label: 'Lace', type: 'select', options: [['cotton', 'Fine cotton'], ['silk', 'Silk'], ['starched', 'Starched cotton']], value: 'cotton' },
    { key: 'press', label: 'Press (heat set)', min: 0, max: 1, step: 0.05, value: 0.6, live: true, fmt: (v) => v.toFixed(2) },
    { key: 'bendingLength', label: 'Bending length', unit: 'cm', min: 0.3, max: 3, step: 0.05, value: null, live: true, fmt: (v) => v.toFixed(2) },
    { key: 'gravity', label: 'Gravity', unit: 'm/s²', min: 0, max: 20, step: 0.1, value: 9.81, live: true, scale: 100, fmt: (v) => v.toFixed(1) },
    { key: 'damping', label: 'Damping', min: 0, max: 0.3, step: 0.005, value: 0.03, live: true, fmt: (v) => v.toFixed(3) },
    { key: 'pitch', label: 'Pleat width', unit: 'cm', min: 0.6, max: 3, step: 0.1, value: 1.5, fmt: (v) => v.toFixed(1), show: (v) => v.kind !== 'gather' && v.kind !== 'flute' },
    { key: 'pleatDepth', label: 'Pleat depth', unit: 'cm', min: 0.3, max: 3, step: 0.1, value: 1.5, fmt: (v) => v.toFixed(1), show: (v) => v.kind === 'knife' || v.kind === 'box' },
    { key: 'ratio', label: 'Length ratio', unit: '×', min: 1.1, max: 3.5, step: 0.05, value: 2.5, fmt: (v) => v.toFixed(2), show: (v) => v.kind === 'accordion' || v.kind === 'gather' },
    { key: 'radius', label: 'Flute radius', unit: 'cm', min: 0.2, max: 1, step: 0.05, value: 0.35, fmt: (v) => v.toFixed(2), show: (v) => v.kind === 'flute' },
    { key: 'stitch', label: 'Gathering stitch', unit: 'cm', min: 0.8, max: 3, step: 0.1, value: 2.0, fmt: (v) => v.toFixed(1), show: (v) => v.kind === 'gather' },
    { key: 'finished', label: 'Finished length', unit: 'cm', min: 8, max: 40, step: 1, value: 24, fmt: (v) => v.toFixed(0) },
    { key: 'depth', label: 'Frill depth', unit: 'cm', min: 3, max: 14, step: 0.5, value: 6, fmt: (v) => v.toFixed(1) },
    { key: 'res', label: 'Mesh spacing', unit: 'cm', min: 0.2, max: 0.6, step: 0.05, value: 0.4, fmt: (v) => v.toFixed(2) },
    { key: 'stretchK', label: 'Threads held in the bend solve', min: 0, max: 600, step: 10, value: 300, live: true, fmt: (v) => v.toFixed(0) },
    { key: 'passes', label: 'Bend solve passes', min: 1, max: 6, step: 1, value: 4, live: true, fmt: (v) => v.toFixed(0) },
    { key: 'iterations', label: 'Constraint walks a step', min: 1, max: 12, step: 1, value: 4, live: true, fmt: (v) => v.toFixed(0) },
    { key: 'substepHz', label: 'Steps a second', min: 120, max: 960, step: 60, value: 240, live: true, fmt: (v) => v.toFixed(0) },
  ],
  build(v) {
    const o = { ...v, gravity: v.gravity * 100, substep: 1 / v.substepHz, cuffRadius: v.finished / (2 * Math.PI) };
    if (v.bendingLength == null) delete o.bendingLength;
    if (v.kind === 'box' && v.pleatDepth > v.pitch / 2 + 1e-6) o.pleatDepth = v.pitch / 2;
    o.rows = Math.max(8, Math.round(v.depth / v.res) + 1);
    delete o.substepHz;
    const S = L.buildStrip(o), f = S.f, n = f.n;
    const parts = Uint8Array.from(S.part);
    for (const c of S.creaseCols) for (let r = 0; r < S.rows; r++) { parts[S.idx(c, r)] = 4; if (S.span(c) === 2) parts[S.idx(c + 1, r)] = 4; }
    const uv = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) { uv[2 * i] = S.flat[2 * i]; uv[2 * i + 1] = S.flat[2 * i + 1]; }
    const bodies = [];
    if (v.mount === 'cuff') bodies.push({ type: 'cylinder', r: S.p.cuffRadius * 0.7, y0: -v.depth - 6, y1: 4 });
    if (v.mount === 'collar') bodies.push({ type: 'cylinder', r: S.p.cuffRadius * 0.82, y0: -6, y1: v.depth * 0.3 });
    if (v.mount === 'jabot') bodies.push({ type: 'plane', z: 0, w: v.depth * 2.5, h: v.finished + 4, y: -v.finished / 2 });
    const scene = {
      S, f, tris: Uint32Array.from(f.tris), uv, parts, partNames: LACE_PARTS.names, partColors: LACE_PARTS.colors,
      body: S.body, bodies, flat: S.flat, edges: threadsOf(S.cols, S.rows, S.idx), uvScale: [S.prof.cut, v.depth],
      info: [
        ['Cut length', `${S.prof.cut.toFixed(1)} cm`], ['Finished', `${S.prof.finished.toFixed(1)} cm`],
        ['Cut ÷ finished', `${S.prof.ratio.toFixed(2)}×`], ['Points', `${n}`],
        ['Bending length', `${(S.mat.bendingLength * 100).toFixed(2)} cm`], ['Weight', `${(S.mat.weight * 1000).toFixed(0)} g/m²`],
      ],
      measure() {
        const ch = L.creaseHold(S);
        const out = [];
        if (S.creaseCols.length) out.push(['Folds near the heading', `${ch.topDeg.toFixed(0)}°`], ['Folds at the scallop', `${ch.bottomDeg.toFixed(0)}°`]);
        const sec = L.sections(S, [0.5, 1]);
        const round = v.mount === 'cuff' || v.mount === 'collar';
        out.push([round ? 'Flare at the scallop' : 'Spread at the scallop', round ? `${(sec[1].mean / S.p.cuffRadius).toFixed(2)}× the heading` : `${sec[1].spread.toFixed(1)} cm`]);
        out.push(['Ripple at the scallop', `${sec[1].ripple.toFixed(2)} cm`], ['Waves at the scallop', `${sec[1].lobes}`]);
        return out;
      },
      sections() {
        const sec = L.sections(S, [0, 0.25, 0.5, 0.75, 1]);
        return { round: v.mount === 'cuff' || v.mount === 'collar', jabot: v.mount === 'jabot', lines: sec.map((s) => ({ label: `${Math.round(s.share * 100)}%`, pts: s.pts })) };
      },
      flags(st) { return L.contourFlags(S, L.sections(S), st); },
      live(key, val) {
        const s = f.settings;
        if (key === 'press') L.setPress(f, v.kind === 'gather' ? 0 : val);
        else if (key === 'bendingLength') {
          const dens = S.mat.weight / 1e4;
          f.stiff.B = dens * Math.hypot(...s.gravity) * Math.pow(val, 3);
          f.stiff._h = null;
        } else if (key === 'gravity') s.gravity = [0, -val * 100, 0];
        else if (key === 'damping') s.damping = val;
        else if (key === 'stretchK') { f.stiff.o.stretchK = val; f.stiff._h = null; }
        else if (key === 'passes') f.stiff.o.passes = val;
        else if (key === 'iterations') s.iterations = val;
        else if (key === 'substepHz') { s.substep = 1 / val; f.stiff._h = null; }
      },
    };
    return scene;
  },
};

// ------------------------------------------------------------------ common grid sheets
function gridSheet(cols, rows, at3, o) {
  const n = cols * rows, pos = new Float64Array(3 * n), idx = (c, r) => r * cols + c;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) pos.set(at3(c, r), 3 * idx(c, r));
  const flat = new Float64Array(2 * n);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { flat[2 * idx(c, r)] = c * o.du; flat[2 * idx(c, r) + 1] = r * o.dv; }
  return { pos, idx, flat, n };
}

function materialSettings(name) {
  const m = MATERIALS[name];
  return { m, density: m.weight / 1e4 };
}

const sphere = (c, r, pad) => (p) => {
  const dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2], d = Math.hypot(dx, dy, dz), R = r + pad;
  if (d >= R) return false;
  const k = d > 1e-9 ? R / d : 0;
  p[0] = c[0] + dx * k; p[1] = c[1] + (d > 1e-9 ? dy * k : R); p[2] = c[2] + dz * k;
  return true;
};

// ------------------------------------------------------------------ a sheet dropped on a ball
const drape = {
  id: 'drape', name: 'Sheet on a ball',
  note: 'After the drape tester (Cusick: a 30 cm specimen over an 18 cm support): a square of cloth dropped centred on a ball. The bend decides how many folds it falls into.',
  params: [
    { key: 'material', label: 'Cloth', type: 'select', options: [['silk', 'Silk'], ['linen', 'Linen'], ['wool', 'Wool']], value: 'linen' },
    { key: 'size', label: 'Sheet', unit: 'cm', min: 20, max: 60, step: 2, value: 30, fmt: (v) => v.toFixed(0) },
    { key: 'res', label: 'Mesh spacing', unit: 'cm', min: 0.75, max: 3, step: 0.25, value: 1.25, fmt: (v) => v.toFixed(2) },
    { key: 'bendingLength', label: 'Bending length', unit: 'cm', min: 0.5, max: 8, step: 0.1, value: null, live: true, fmt: (v) => v.toFixed(1) },
    { key: 'gravity', label: 'Gravity', unit: 'm/s²', min: 0, max: 20, step: 0.1, value: 9.81, live: true, fmt: (v) => v.toFixed(1) },
    { key: 'damping', label: 'Damping', min: 0, max: 0.3, step: 0.005, value: 0.03, live: true, fmt: (v) => v.toFixed(3) },
    { key: 'friction', label: 'Friction', min: 0, max: 1, step: 0.05, value: 0.5, live: true, fmt: (v) => v.toFixed(2) },
    { key: 'iterations', label: 'Constraint walks a step', min: 1, max: 12, step: 1, value: 4, live: true, fmt: (v) => v.toFixed(0) },
    { key: 'substepHz', label: 'Steps a second', min: 120, max: 960, step: 60, value: 960, live: true, fmt: (v) => v.toFixed(0) },
  ],
  build(v) {
    const { m, density } = materialSettings(v.material);
    const cols = Math.round(v.size / v.res) + 1, d = v.size / (cols - 1);
    const G = gridSheet(cols, cols, (c, r) => [(c - (cols - 1) / 2) * d, 30, (r - (cols - 1) / 2) * d], { du: d, dv: d });
    // (laid on the ball's top and clamped there over a small disc, as the tester clamps the specimen's centre
    // between its discs: left free, the sheet creeps off the ball one way, the walk's order deciding which)
    const top = 15 + 9 + 0.3;
    for (let i = 0; i < G.n; i++) G.pos[3 * i + 1] = top;
    const f = new Fabric(G.pos, { gravity: [0, -v.gravity * 100, 0], damping: v.damping, iterations: v.iterations, substep: 1 / v.substepHz, shear: m.shear, friction: v.friction, contact: 'iteration', edgeContact: true });
    for (let i = 0; i < G.n; i++) if (Math.hypot(G.pos[3 * i], G.pos[3 * i + 2]) < 3.01) f.pin(i);
    const c = v.bendingLength ?? m.bendingLength * 100;
    f.sheet(cols, cols, G.idx, v.bend === 'hinge' ? { bend: null, hinge: m.hinge, shear: m.shear, flat: true } : { bend: null, hinge: null, shear: m.shear });
    f.weigh(density);
    if (v.bend !== 'hinge') f.stiffen({ bendingLength: c, density, passes: v.passes ?? 1, stretchK: v.stretchK ?? 0 });
    const ball = sphere([0, 15, 0], 9, 0.3), body = ball;
    const n = G.n, uv = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) { uv[2 * i] = G.flat[2 * i]; uv[2 * i + 1] = G.flat[2 * i + 1]; }
    const parts = new Uint8Array(n);
    return {
      f, tris: Uint32Array.from(f.tris), uv, parts, partNames: ['sheet'], partColors: ['#8fb3d9'], body,
      bodies: [{ type: 'sphere', c: [0, 15, 0], r: 9 }],
      flat: G.flat, edges: threadsOf(cols, cols, G.idx), uvScale: [v.size, v.size],
      info: [['Points', `${n}`], ['Bending length', `${c.toFixed(2)} cm`], ['Weight', `${(m.weight * 1000).toFixed(0)} g/m²`]],
      measure() {
        // the drape coefficient's stand-in: the share of the sheet's outline circle the hem still covers
        const x = f.pos;
        let r = 0, k = 0;
        for (let i = 0; i < cols; i++) for (const j of [G.idx(i, 0), G.idx(i, cols - 1), G.idx(0, i), G.idx(cols - 1, i)]) { r += Math.hypot(x[3 * j], x[3 * j + 2]); k++; }
        return [['Hem radius (mean)', `${(r / k).toFixed(1)} cm`], ['Flat half-width', `${(v.size / 2).toFixed(1)} cm`]];
      },
      live(key, val) {
        const s = f.settings;
        if (key === 'bendingLength') { if (f.stiff) { f.stiff.B = density * Math.hypot(...s.gravity) * Math.pow(val, 3); f.stiff._h = null; } }
        else if (key === 'gravity') s.gravity = [0, -val * 100, 0];
        else if (key === 'damping') s.damping = val;
        else if (key === 'friction') s.friction = val;
        else if (key === 'iterations') s.iterations = val;
        else if (key === 'substepHz') { s.substep = 1 / val; f.stiff._h = null; }
      },
    };
  },
};

// ------------------------------------------------------------------ a pressed fold (authored hinges)
const fold = {
  id: 'fold', name: 'Pressed fold',
  note: 'A wool panel held along its top with one fold pressed down its middle (a box pleat\'s edge). The fold is a line of hinges resting at the angle asked; it eases in so you can watch it form.',
  params: [
    { key: 'angle', label: 'Fold angle', unit: '°', min: 0, max: 175, step: 1, value: 70, live: true, fmt: (v) => v.toFixed(0) },
    { key: 'firm', label: 'Fold firmness', unit: 'log', min: -7, max: -1, step: 0.25, value: -4, live: true, fmt: (v) => `1e${v.toFixed(2)}` },
    { key: 'material', label: 'Cloth', type: 'select', options: [['silk', 'Silk'], ['linen', 'Linen'], ['wool', 'Wool']], value: 'wool' },
    { key: 'bendingLength', label: 'Bending length', unit: 'cm', min: 0.5, max: 8, step: 0.1, value: null, live: true, fmt: (v) => v.toFixed(1) },
    { key: 'ease', label: 'Ease in over', unit: 's', min: 0, max: 3, step: 0.1, value: 1, fmt: (v) => v.toFixed(1) },
    { key: 'gravity', label: 'Gravity', unit: 'm/s²', min: 0, max: 20, step: 0.1, value: 9.81, live: true, fmt: (v) => v.toFixed(1) },
    { key: 'damping', label: 'Damping', min: 0, max: 0.3, step: 0.005, value: 0.03, live: true, fmt: (v) => v.toFixed(3) },
    { key: 'iterations', label: 'Constraint walks a step', min: 1, max: 12, step: 1, value: 4, live: true, fmt: (v) => v.toFixed(0) },
  ],
  build(v) {
    const { m, density } = materialSettings(v.material);
    const cols = 25, rows = 15, w = 50, hgt = 30, d = w / (cols - 1), dv = hgt / (rows - 1);
    const G = gridSheet(cols, rows, (c, r) => [(c - (cols - 1) / 2) * d, 35 - r * dv, 0], { du: d, dv });
    const f = new Fabric(G.pos, { gravity: [0, -v.gravity * 100, 0], damping: v.damping, iterations: v.iterations, substep: 1 / 240, shear: m.shear, friction: 0.5 });
    for (let c = 0; c < cols; c++) f.pin(G.idx(c, 0));
    f.sheet(cols, rows, G.idx, { bend: null, hinge: m.hinge, shear: m.shear, flat: true });
    f.weigh(density);
    f.stiffen({ bendingLength: v.bendingLength ?? m.bendingLength * 100, density, passes: 1 });
    f.tether();
    const mid = (cols - 1) / 2, line = [];
    for (let r = 0; r < rows; r++) line.push(G.idx(mid, r));
    f.fold(line, 0, { compliance: 10 ** v.firm, set: false });
    f._pack();
    const hk = [];
    for (let q = 1; q < line.length; q++) hk.push(f._edgeHinge.get(f._key(line[q - 1], line[q])));
    let target = v.angle * DEG, t = 0;
    const n = G.n, uv = new Float32Array(2 * n), parts = new Uint8Array(n);
    for (let i = 0; i < n; i++) { uv[2 * i] = G.flat[2 * i]; uv[2 * i + 1] = G.flat[2 * i + 1]; }
    for (const i of line) parts[i] = 1;
    for (let c = 0; c < cols; c++) parts[G.idx(c, 0)] = 2;
    const setRest = () => { const k = v.ease > 0 ? Math.min(1, t / v.ease) : 1; for (const h of hk) f._packed.hRest[h] = target * k; };
    return {
      f, tris: Uint32Array.from(f.tris), uv, parts, partNames: ['panel', 'pressed fold', 'held edge'], partColors: ['#8fb3d9', '#d2523a', '#2f6b4f'],
      body: null, bodies: [], flat: G.flat, edges: threadsOf(cols, rows, G.idx), uvScale: [w, hgt],
      info: [['Points', `${n}`], ['Fold hinges', `${hk.length}`]],
      before(dt) { t += dt; setRest(); },
      measure() {
        let a = 0;
        for (const h of hk) { const P = f._packed; a += Math.abs(f._angle(P.ha[h], P.hb[h], P.hc[h], P.hd[h])); }
        return [['Fold angle held', `${(a / hk.length / DEG).toFixed(1)}°`], ['Asked', `${v.angle.toFixed(0)}°`]];
      },
      live(key, val) {
        const s = f.settings;
        if (key === 'angle') { target = val * DEG; v.angle = val; setRest(); }
        else if (key === 'firm') for (const h of hk) f._packed.hComp[h] = 10 ** val;
        else if (key === 'bendingLength') { f.stiff.B = density * Math.hypot(...s.gravity) * Math.pow(val, 3); f.stiff._h = null; }
        else if (key === 'gravity') s.gravity = [0, -val * 100, 0];
        else if (key === 'damping') s.damping = val;
        else if (key === 'iterations') s.iterations = val;
      },
    };
  },
};

// ------------------------------------------------------------------ a collar's roll, held at its gorge
const collar = {
  id: 'collar', name: 'Collar roll',
  note: 'A turn-down collar sewn round the neck at its gorge (held), standing up to the roll line and turned down over it. The roll is a band of hinges as wide as its radius times its angle; it eases in from flat. The collar is cut curved (its spring) so the fall can lie out over the stand.',
  params: [
    { key: 'roll', label: 'Roll angle', unit: '°', min: 0, max: 175, step: 1, value: 160, live: true, fmt: (v) => v.toFixed(0) },
    { key: 'turn', label: 'Roll turns the fall', type: 'select', options: [['out', 'Outward'], ['in', 'Inward']], value: 'out' },
    { key: 'radius', label: 'Roll radius', unit: 'cm', min: 0.2, max: 2, step: 0.1, value: 0.6, fmt: (v) => v.toFixed(1) },
    { key: 'stand', label: 'Stand height', unit: 'cm', min: 1.5, max: 5, step: 0.25, value: 3, fmt: (v) => v.toFixed(2) },
    { key: 'spring', label: 'Collar spring (cut curve)', min: 0.02, max: 0.9, step: 0.02, value: 0.5, fmt: (v) => v.toFixed(2) },
    { key: 'fall', label: 'Fall depth', unit: 'cm', min: 2, max: 8, step: 0.25, value: 4.5, fmt: (v) => v.toFixed(2) },
    { key: 'material', label: 'Cloth', type: 'select', options: [['silk', 'Silk'], ['linen', 'Linen'], ['wool', 'Wool']], value: 'linen' },
    { key: 'bendingLength', label: 'Bending length (interfaced)', unit: 'cm', min: 1, max: 12, step: 0.25, value: 6, live: true, fmt: (v) => v.toFixed(2) },
    { key: 'hingeLog', label: 'Cloth hinge softness', unit: 'log', min: -6, max: 0, step: 0.25, value: -1, live: true, fmt: (v) => `1e${v.toFixed(2)}` },
    { key: 'firm', label: 'Roll firmness', unit: 'log', min: -7, max: -1, step: 0.25, value: -4, live: true, fmt: (v) => `1e${v.toFixed(2)}` },
    { key: 'ease', label: 'Ease in over', unit: 's', min: 0, max: 3, step: 0.1, value: 1.2, fmt: (v) => v.toFixed(1) },
    { key: 'gravity', label: 'Gravity', unit: 'm/s²', min: 0, max: 20, step: 0.1, value: 9.81, live: true, fmt: (v) => v.toFixed(1) },
    { key: 'damping', label: 'Damping', min: 0, max: 0.3, step: 0.005, value: 0.04, live: true, fmt: (v) => v.toFixed(3) },
    { key: 'iterations', label: 'Constraint walks a step', min: 1, max: 12, step: 1, value: 4, live: true, fmt: (v) => v.toFixed(0) },
  ],
  build(v) {
    const { m, density } = materialSettings(v.material);
    // round the back half of the neck (36 cm round). The collar is cut as a curve, an annular sector: a straight
    // collar cannot turn down on a round neck, since its fall would have to stretch to lie further out than its
    // stand (the first trial, cut straight, held its roll at 6 of 160 degrees). Pattern drafting gives a turn-down
    // collar "spring": its fall edge cut longer than its neck edge, between a stand collar (straight, 0) and a flat
    // collar (cut to the neckline's own curve). Here spring is the sine of the cone the sector rolls into.
    const R = 36 / (2 * Math.PI), span = Math.PI, cols = 33, depth = v.stand + v.fall, rows = Math.max(10, Math.round(depth / 0.4) + 1);
    const sinA = Math.max(0.02, Math.min(0.95, v.spring)), cosA = Math.sqrt(1 - sinA * sinA);
    const R0 = R + 0.4, gorge = R0 * span, phi = span * sinA, rho0 = gorge / phi;
    const dv = depth / (rows - 1), du = gorge / (cols - 1);
    const at3 = (c, r) => { const a = -span / 2 + span * c / (cols - 1), v_ = r * dv; return [(R0 + v_ * sinA) * Math.sin(a), 40 + v_ * cosA, -(R0 + v_ * sinA) * Math.cos(a)]; };
    const G = gridSheet(cols, rows, at3, { du, dv });
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const i = r * cols + c, th = phi * c / (cols - 1), rho = rho0 + r * dv; G.flat[2 * i] = rho * Math.cos(th); G.flat[2 * i + 1] = rho * Math.sin(th); }
    const f = new Fabric(G.pos, { gravity: [0, -v.gravity * 100, 0], damping: v.damping, iterations: v.iterations, substep: 1 / 240, shear: m.shear, friction: 0.3, contact: 'iteration' });
    // (held along the gorge and the row above it: the seam allowance is caught in the neckline seam with the
    // bodice under it, so the collar cannot swing about the seam line; held on one row it flopped down at once)
    for (let c = 0; c < cols; c++) { f.pin(G.idx(c, 0)); f.pin(G.idx(c, 1)); }
    // (its hinges rest as it is laid round the neck, not flat: the neckline seam gives it that curve, and hinges resting
    // flat crushed the cone in on itself, stretching it 50 to 80%)
    f.sheet(cols, rows, G.idx, { bend: null, hinge: 10 ** v.hingeLog, shear: m.shear, flat: false });
    // The collar's own stiffness is the implicit bend at its bending length (an interfaced collar: hinges alone cap
    // out near 2 cm and a 7 cm collar fell flat from its gorge), with the threads held in the same solve. Across the
    // roll band and one row either side the implicit bend is let go (scale 0), so the roll's hinges turn it there;
    // kept at the band's edges, it pulled the turn flat and the roll held 10 of 160 degrees.
    f.weigh(density);
    f.stiffen({ bendingLength: v.bendingLength ?? 6, density, passes: 3, stretchK: 300 });
    f.tether();
    // the roll band: the rows within the roll's width (radius x angle) above the stand line, each its share
    const band = [];
    const width = Math.max(dv, v.radius * v.roll * DEG);
    const r0 = Math.round(v.stand / dv), r1 = Math.max(r0 + 1, Math.round((v.stand + width) / dv));
    for (let r = r0; r < Math.min(rows - 1, r1); r++) band.push(r);
    for (let r = Math.max(0, r0 - 1); r <= Math.min(rows - 1, r1 + 1); r++) for (let c = 0; c < cols; c++) f.stiff.scale[G.idx(c, r)] = 0;
    f.stiff._h = null;
    const hk = [];
    for (const r of band) for (let c = 0; c < cols - 1; c++) {
      const a = G.idx(c, r), b = G.idx(c + 1, r);
      if (f.crease(a, b, 0, { compliance: 10 ** v.firm, set: false })) hk.push(f._edgeHinge.get(f._key(a, b)));
    }
    f._pack();
    // which way is out: the roll turns the fall outward, away from the neck; measured once, by a trial nudge
    let sign = 1, target = v.roll * DEG, t = 0;
    {
      const P = f._packed, h = hk[Math.floor(hk.length / 2)];
      const d = P.hd[h], save = [f.pos[3 * d], f.pos[3 * d + 2]];
      const before = f._angle(P.ha[h], P.hb[h], P.hc[h], P.hd[h]);
      const bq = P.hb[h];
      const out = [f.pos[3 * bq], f.pos[3 * bq + 2]], l = Math.hypot(...out) || 1;
      f.pos[3 * bq] += 0.05 * out[0] / l; f.pos[3 * bq + 2] += 0.05 * out[1] / l;
      const after = f._angle(P.ha[h], P.hb[h], P.hc[h], P.hd[h]);
      f.pos[3 * bq] -= 0.05 * out[0] / l; f.pos[3 * bq + 2] -= 0.05 * out[1] / l;
      void save; void d;
      sign = (Math.sign(after - before) || 1) * (v.turn === 'in' ? 1 : -1);
    }
    const setRest = () => { const k = v.ease > 0 ? Math.min(1, t / v.ease) : 1; const each = sign * target * k / Math.max(1, band.length); for (const h of hk) f._packed.hRest[h] = each; };
    const neck = (p) => { const rr = Math.hypot(p[0], p[2]); if (rr >= R + 0.1 || p[1] < 30 || p[1] > 60) return false; const k = (R + 0.1) / (rr || 1); p[0] *= k; p[2] *= k; return true; };
    const n = G.n, uv = new Float32Array(2 * n), parts = new Uint8Array(n);
    for (let i = 0; i < n; i++) { uv[2 * i] = G.flat[2 * i]; uv[2 * i + 1] = G.flat[2 * i + 1]; }
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) parts[G.idx(c, r)] = r < r0 ? 0 : band.includes(r) ? 1 : 2;
    for (let c = 0; c < cols; c++) parts[G.idx(c, 0)] = 3;
    return {
      f, tris: Uint32Array.from(f.tris), uv, parts, partNames: ['stand', 'roll', 'fall', 'gorge (held)'], partColors: ['#8fb3d9', '#d2523a', '#e0b35a', '#2f6b4f'],
      body: neck, bodies: [{ type: 'cylinder', r: R, y0: 25, y1: 48 }], flat: G.flat, edges: threadsOf(cols, rows, G.idx), uvScale: [gorge, depth],
      info: [['Points', `${n}`], ['Roll rows', `${band.length}`], ['Roll width', `${width.toFixed(2)} cm`]],
      before(dt) { t += dt; setRest(); },
      measure() {
        // the roll as turned: the angle between the stand's and the fall's direction at centre back
        const c = (cols - 1) / 2, x = f.pos, p = (r) => { const i = 3 * G.idx(c, r); return [x[i], x[i + 1], x[i + 2]]; };
        const a = p(Math.max(0, r0 - 2)), b = p(r0), cc = p(Math.min(rows - 2, r1 + 1)), d = p(Math.min(rows - 1, r1 + 3));
        const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [d[0] - cc[0], d[1] - cc[1], d[2] - cc[2]];
        const cs = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / ((Math.hypot(...u) * Math.hypot(...w)) || 1);
        return [['Roll held', `${(Math.acos(Math.max(-1, Math.min(1, cs))) / DEG).toFixed(0)}°`], ['Asked', `${v.roll.toFixed(0)}°`]];
      },
      live(key, val) {
        const s = f.settings;
        if (key === 'roll') { target = val * DEG; setRest(); }
        else if (key === 'firm') for (const h of hk) f._packed.hComp[h] = 10 ** val;
        else if (key === 'bendingLength') { f.stiff.B = density * Math.hypot(...s.gravity) * Math.pow(val, 3); f.stiff._h = null; }
        else if (key === 'hingeLog') { const hs = new Set(hk); for (let k = 0; k < f._packed.hComp.length; k++) if (!hs.has(k)) f._packed.hComp[k] = 10 ** val; }
        else if (key === 'gravity') s.gravity = [0, -val * 100, 0];
        else if (key === 'damping') s.damping = val;
        else if (key === 'iterations') s.iterations = val;
      },
    };
  },
};

export const PRESETS = [lace, drape, fold, collar];

// the strain of every thread against its cut length, per point (the largest of its threads) and overall
export function strainOf(scene) {
  const x = scene.f.pos, fl = scene.flat, e = scene.edges, n = scene.f.n, per = new Float32Array(n), w = scene.f.w;
  let max = 0, sum = 0, cnt = 0;
  for (let k = 0; k < e.length; k += 2) {
    const i = e[k], j = e[k + 1];
    if (w[i] === 0 && w[j] === 0) continue;          // (a thread between two held points is the stitching's)
    const r = Math.hypot(fl[2 * i] - fl[2 * j], fl[2 * i + 1] - fl[2 * j + 1]);
    const l = Math.hypot(x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]);
    const s = Math.abs(l / r - 1);
    if (s > per[i]) per[i] = s;
    if (s > per[j]) per[j] = s;
    if (s > max) max = s;
    sum += s; cnt++;
  }
  return { per, max, mean: sum / (cnt || 1) };
}
