// The Face Creator's layered skin reads the same numbers as Blender's: every top-level constant of
// blender/base/skin_layers.py, written into the creator's main.js between its SKIN_LAYERS markers as one object.
// With them, from the same Blender files: where the head sits on the figure (female.py HEAD_H, HEAD_Z: the skin's
// patterns are laid out in the figure's rest space, so the creator's head looks them up where the figure's head is),
// and the shared 3D look's key and bounce lights (views.py look_3d), for the creator's warm bounce from below.
// Run it after any change to those; with --check it only says whether main.js is out of date (exit 1).
//
//   node tools/face-creator/skin-params.mjs <main.js> [--check]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const PY = path.join(here, '../../blender/base/skin_layers.py');
const [target, flag] = process.argv.slice(2);
if (!target) { console.error('usage: node skin-params.mjs <main.js> [--check]'); process.exit(2); }

// NAME = value  (# comment): numbers, and lists or tuples of numbers
const consts = {};
for (const line of fs.readFileSync(PY, 'utf8').split('\n')) {
  const m = /^([A-Z][A-Z0-9_]*)\s*=\s*([^#]+?)\s*(#.*)?$/.exec(line);
  if (!m) continue;
  const v = m[2].replace(/\(/g, '[').replace(/\)/g, ']');
  try { consts[m[1]] = JSON.parse(v); } catch (e) { /* not a plain number or list: not a parameter */ }
}
const FEMALE = path.join(here, '../../blender/base/female.py'), VIEWS = path.join(here, '../../blender/base/views.py');
const hz = /^HEAD_H, HEAD_Z = ([\d.]+), ([\d.]+)/m.exec(fs.readFileSync(FEMALE, 'utf8'));
if (!hz) { console.error('HEAD_H, HEAD_Z not found in female.py'); process.exit(2); }
consts.FIGURE = { HEAD_H: +hz[1], HEAD_Z: +hz[2] };
const vsrc = fs.readFileSync(VIEWS, 'utf8'), tup = t => JSON.parse('[' + t + ']');
consts.LIGHTS = {};
for (const name of ['Key', 'Bounce']) {
  const m = new RegExp("area\\('" + name + "', \\(([^)]*)\\), ([\\d.]+), ([\\d.]+)(?:, \\(([^)]*)\\))?(?:, aim=\\(([^)]*)\\))?\\)").exec(vsrc);
  if (!m) { console.error('light ' + name + ' not found in views.py'); process.exit(2); }
  consts.LIGHTS[name] = { loc: tup(m[1]), energy: +m[2], size: +m[3], colour: m[4] ? tup(m[4]) : [1, 1, 1], aim: m[5] ? tup(m[5]) : [0, 0, 40] };
}
const block = '/* @@SKIN_LAYERS@@ generated from blender/base/skin_layers.py by tools/face-creator/skin-params.mjs: do not edit */\n' +
  'var SKIN_LAYERS = ' + JSON.stringify(consts) + ';\n/* @@SKIN_LAYERS_END@@ */';
const src = fs.readFileSync(target, 'utf8');
const re = /\/\* @@SKIN_LAYERS@@[\s\S]*?@@SKIN_LAYERS_END@@ \*\//;
if (!re.test(src)) { console.error('no SKIN_LAYERS markers in ' + target); process.exit(2); }
const out = src.replace(re, block);
if (flag === '--check') {
  if (out !== src) { console.error('main.js is out of date with skin_layers.py: run node tools/face-creator/skin-params.mjs ' + target); process.exit(1); }
  console.log('main.js matches skin_layers.py (' + Object.keys(consts).length + ' parameters)');
} else {
  fs.writeFileSync(target, out);
  console.log('wrote ' + Object.keys(consts).length + ' parameters into ' + target);
}
