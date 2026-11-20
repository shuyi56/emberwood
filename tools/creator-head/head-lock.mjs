// Thin the Face Creator's head for the game without touching the face where it moves (John 2026-11-20).
// The simplifier is given a lock on every vertex in the eye patch, the mouth and the nose, so the lid loops, the lips
// and the nostrils keep every vertex they have; the rest of the head (scalp, cheeks, jaw, ears, neck) is thinned.
//   lockedSimplify(prim, ratio, lock, error)  one primitive, the vertices flagged in `lock` held
//   faceLock(doc, prim)                       the lock for a head primitive: the creator's _EYEPATCH attribute when the
//                                             export carries one, else boxes from the eyeballs and the canon mouth/nose
import { MeshoptSimplifier } from 'meshoptimizer';

export async function ready() { await MeshoptSimplifier.ready; MeshoptSimplifier.useExperimentalFeatures = true; }

// Zones in head heights (rq_topo.py measures the same ones): eyes 0.85 x 0.62 eye widths, mouth 0.75 x 0.48 mouth
// widths, nose 0.3 x 0.42 of the IPD, each grown by GROW so the zone's edge is not where the thinning starts.
const GROW = 1.0;

export function faceLock(headDoc, pr, boxesToo = false) {
  const P = pr.getAttribute('POSITION').getArray(), n = P.length / 3, lock = new Uint8Array(n);
  // the creator's own marks: _EYEPATCH (the lids' margins and the skin round them) and _MOUTHPATCH (the parting line and the lips)
  const marks = ['_EYEPATCH', '__EYEPATCH', '_MOUTHPATCH', '__MOUTHPATCH', '_NOSEPATCH', '__NOSEPATCH'].map((m) => pr.getAttribute(m)).filter(Boolean);
  let used = 'box';
  for (const m of marks) { const A = m.getArray(); for (let v = 0; v < n; v++) if (A[v] > 0.5) lock[v] = 1; }
  if (marks.length) used = 'the creator\'s _EYEPATCH, _MOUTHPATCH';
  if (!boxesToo) return { lock, held: lock.reduce((a, b) => a + b, 0), used: marks.length ? used : 'none', n };
  let top = -1e9, bot = 1e9;
  for (let v = 0; v < n; v++) { top = Math.max(top, P[3 * v + 1]); }
  // the eyeballs: the small meshes of the export, averaged per side
  const sides = { '-1': [0, 0, 0, 0], '1': [0, 0, 0, 0] };
  for (const m of headDoc.getRoot().listMeshes()) for (const q of m.listPrimitives()) {
    if (q === pr) continue;
    const Q = q.getAttribute('POSITION').getArray(), k = Q.length / 3;
    if (k > 20000) continue;
    for (let v = 0; v < k; v++) {
      const y = Q[3 * v + 1], z = Q[3 * v + 2];
      if (y < 0.45 * top || y > 0.62 * top || z < 0) continue;
      const s = Q[3 * v] < 0 ? '-1' : '1', a = sides[s]; a[0] += Q[3 * v]; a[1] += y; a[2] += Q[3 * v + 2]; a[3]++;
    }
  }
  const e0 = sides['-1'], e1 = sides['1'];
  const H = top, boxes = [];
  if (e0[3] && e1[3]) {
    const c0 = [e0[0] / e0[3], e0[1] / e0[3]], c1 = [e1[0] / e1[3], e1[1] / e1[3]], ipd = Math.abs(c1[0] - c0[0]), ew = ipd / 2.2;
    for (const c of [c0, c1]) boxes.push([c[0], c[1], 0.85 * ew * GROW, 0.62 * ew * GROW]);
    const mw = 0.8 * ipd;
    boxes.push([0, 0.232 * H, 0.75 * mw * GROW * 0.5 * 1.1, 0.48 * mw * GROW * 0.5 * 1.1 + 0.02 * H]);
    boxes.push([0, 0.35 * H, 0.3 * ipd * GROW, 0.42 * ipd * GROW]);
    if (used === 'box') used = 'boxes from the eyeballs';
  } else {   // (no eyeballs in the export: the canon face)
    boxes.push([-0.14 * H, 0.5265 * H, 0.12 * H, 0.09 * H], [0.14 * H, 0.5265 * H, 0.12 * H, 0.09 * H],
               [0, 0.232 * H, 0.17 * H, 0.11 * H], [0, 0.35 * H, 0.09 * H, 0.12 * H]);
    if (used === 'box') used = 'canon boxes';
  }
  let held = 0;
  for (let v = 0; v < n; v++) {
    if (!lock[v] && P[3 * v + 2] > 0) for (const [cx, cy, hw, hh] of boxes) if (Math.abs(P[3 * v] - cx) < hw && Math.abs(P[3 * v + 1] - cy) < hh) { lock[v] = 1; break; }
    held += lock[v];
  }
  return { lock, held, used, n };
}

export function lockedSimplify(pr, ratio, lock, error = 0.0003, uvWeight = 0.5) {
  const P = pr.getAttribute('POSITION').getArray(), acc = pr.getIndices(), I = acc.getArray();
  const target = Math.max(3, Math.floor((I.length / 3) * ratio)) * 3;
  // (the texture coordinates ride along, lightly weighted, so the paint stays where it was drawn)
  const uvA = pr.getAttribute('TEXCOORD_0'), uv = uvA ? Float32Array.from(uvA.getArray()) : new Float32Array(P.length / 3 * 2);
  const [out, err] = MeshoptSimplifier.simplifyWithAttributes(new Uint32Array(I), P, 3, uv, 2, [uvWeight, uvWeight], lock, target, error, ['LockBorder']);
  acc.setArray(I instanceof Uint32Array ? out : (P.length / 3 > 65535 ? out : new Uint16Array(out)));
  return { from: I.length / 3, to: out.length / 3, error: err * MeshoptSimplifier.getScale(P, 3) };
}
