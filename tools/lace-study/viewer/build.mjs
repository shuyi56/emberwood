// Builds the solver viewer: one self-contained page (viewer/lace-solver-viewer.html) with the study's modules
// inlined, so it runs the very code the study runs (fabric-core.js, fabric-stiff.js, figure.js, lace.js, presets.js).
//
//   node tools/lace-study/viewer/build.mjs [--out FILE]
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url)), root = path.join(here, '..');
const order = ['fabric-stiff', 'fabric-core', 'figure', 'lace', 'presets'];
let mods = 'const __m = {};\n';
for (const name of order) {
  let src = fs.readFileSync(path.join(root, name + '.js'), 'utf8');
  const exported = [];
  // imports become reads of the modules built before
  src = src.replace(/^import\s+\*\s+as\s+(\w+)\s+from\s+'\.\/([\w-]+)\.js';\s*$/gm, (_, ns, m) => `const ${ns} = __m['${m}'];`);
  src = src.replace(/^import\s+\{([^}]*)\}\s+from\s+'\.\/([\w-]+)\.js';\s*$/gm, (_, names, m) => `const {${names}} = __m['${m}'];`);
  src = src.replace(/^export\s+(const|let|function|class)\s+(\w+)/gm, (_, kw, nm) => { exported.push(nm); return `${kw} ${nm}`; });
  if (/^\s*(import|export)\s/m.test(src)) throw new Error(`${name}: an import or export the build does not handle`);
  mods += `__m['${name}'] = (() => {\n${src}\nreturn { ${exported.join(', ')} };\n})();\n`;
}
const tpl = fs.readFileSync(path.join(here, 'viewer.html'), 'utf8');
const outArg = process.argv.indexOf('--out');
const out = outArg > 0 ? process.argv[outArg + 1] : path.join(here, 'lace-solver-viewer.html');
fs.writeFileSync(out, tpl.replace('/*@MODULES@*/', () => mods));
console.log('wrote', out, (fs.statSync(out).size / 1024).toFixed(0) + ' KB');
