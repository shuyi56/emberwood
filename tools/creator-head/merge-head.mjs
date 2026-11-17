// Put the Face Creator's head and hair (export-head.js) on the game's Ranger (public/models/ranger.glb), in place of
// the head, hair and face paint from Blender. The body, clothes, skinning and the walk and idle animations are kept
// as exported; the creator's parts go under the Head node, whose frame is the creator's (chin at 0, face toward +z).
//   node merge-head.mjs ranger.glb creator-head.glb out.glb [headRatio] [hairRatio] [hairSoft]
// The creator's head is sculpted far finer than the game needs: it is simplified to headRatio of its triangles
// (default 0.12), its hair strands joined by material, and every texture capped at 1024.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { mergeDocuments, dedup, join, weld, simplifyPrimitive, prune, unpartition } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';

const [, , basePath, headPath, outPath, ratioArg] = process.argv;
const ratio = parseFloat(ratioArg || '0.25'), hairRatio = parseFloat(process.argv[6] || '0.4'), hairSoft = parseFloat(process.argv[7] || '5');
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(basePath), headDoc = await io.read(headPath);
const root = doc.getRoot();
const headNode = root.listNodes().find((n) => n.getName() === 'Head');
if (!headNode) throw new Error('no Head node in ' + basePath);

// (drop Blender's head, hair and face paint, and the eye animation channels that pointed at them)
const old = [];
(function walk(n) { n.listChildren().forEach((c) => { old.push(c); walk(c); }); })(headNode);
for (const a of root.listAnimations()) for (const ch of a.listChannels()) if (old.includes(ch.getTargetNode())) ch.dispose();
// The skin's colour, matched to the game's: the creator paints its face for its own renderer (ACES tone mapping, a
// skin gain and scattering in its shader, none of which export), so in the world the bare texture read grey and dull
// beside the Blender neck and hands. Take the Blender head's mean skin colour (its vertex colours) before it goes,
// then scale the creator skin's vertex colours so the face, sampled through its texture, averages the same.
const lin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
function meanVC(prim) {
  const c = prim.getAttribute('COLOR_0'); if (!c) return null;
  const e = [], s = [0, 0, 0]; for (let i = 0; i < c.getCount(); i++) { c.getElement(i, e); for (let k = 0; k < 3; k++) s[k] += e[k]; }
  return s.map((v) => v / c.getCount());
}
let skinTarget = null;
for (const n of old) { const m = n.getMesh(); if (m && n.getName() === 'Head shape') skinTarget = meanVC(m.listPrimitives()[0]); }
// (a model exported without the head's vertex colours, as the Warden's is: match the face to her skin material, the
// one her neck and hands wear)
if (!skinTarget) { const sm = root.listMaterials().find((mt) => mt.getName() === 'skin'); if (sm) skinTarget = sm.getBaseColorFactor().slice(0, 3); }
old.forEach((n) => { const m = n.getMesh(); n.dispose(); if (m) m.dispose(); });
// The cape's lining: a sheet just inside the cape's thickness, drawn from both sides. In the world's depth buffer
// it lay too close to the cape to tell apart, and walking (each mesh weighted on its own) pushed it through, so
// red streaks broke through the back of the cape. So the lining is turned to face in, toward her, and drawn from
// that side only: from outside its faces are back faces and are never drawn, however close it comes or wherever it
// pokes through; from inside, under the cape's open front, it is the nearest surface and shows red as before.
// (env LINING: comma-separated mesh names, default "Cape lining")
for (const name of (process.env.LINING || 'Cape lining').split(',')) {
  const n = root.listNodes().find((x) => x.getName() === name && x.getMesh());
  if (!n) continue;
  for (const pr of n.getMesh().listPrimitives()) {
    const I = pr.getIndices(), ia = I.getArray();
    for (let f = 0; f < ia.length; f += 3) { const t = ia[f + 1]; ia[f + 1] = ia[f + 2]; ia[f + 2] = t; }
    I.setArray(ia);
    const N = pr.getAttribute('NORMAL');
    if (N) { const na = N.getArray(); for (let k = 0; k < na.length; k++) na[k] = -na[k]; N.setArray(na); }
    if (pr.getMaterial()) pr.getMaterial().setDoubleSided(false);
  }
  console.log('lining faces in:', name);
}
// (the thin embroidered trims stood too close to the cloth for the world's depth buffer and flickered through it:
// TRIM_LIFT lifts the 'embroidery' material off it along its normals)
const trimLift = +(process.env.TRIM_LIFT || 0.1);
for (const m of root.listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial(); if (!mt || mt.getName() !== 'embroidery' || !trimLift) continue;
  const P = pr.getAttribute('POSITION'), N = pr.getAttribute('NORMAL'), p = P.getArray(), n = N.getArray();
  for (let k = 0; k < p.length; k++) p[k] += n[k] * trimLift;
  P.setArray(p);
}
// (a body built for close-up renders, as the Warden's is, carries far more than the world needs: env BODY_TRIS caps
// each skinned part at about that many triangles; off by default, the ranger's export is already light)
const bodyTris = +(process.env.BODY_TRIS || 0);
if (bodyTris) {
  await MeshoptSimplifier.ready;
  for (const m of root.listMeshes()) for (const pr of m.listPrimitives()) {
    const I = pr.getIndices(), t = I ? I.getCount() / 3 : 0;
    if (!pr.getAttribute('JOINTS_0') || t <= bodyTris) continue;
    simplifyPrimitive(pr, { simplifier: MeshoptSimplifier, ratio: bodyTris / t, error: +(process.env.BODY_ERR || 0.0008), lockBorder: true });
  }
}

// (merge the creator's scene and hang its nodes under the Head node)
await MeshoptSimplifier.ready;
await headDoc.transform(dedup(), weld({ tolerance: 0.0001 }));
// Two creator parts that do not carry over, both found by their materials (vertex colours, no map):
//  - the hair's veil, a soft see-through glow laid over the locks at the temples for the creator's light; it exports
//    unlit, and in the world it showed as flat, bright orange patches either side of the face;
//  - the ears (sheen): her hair is worn down over them, and their tops showed through the gaps between the locks.
const extOf = (mt, name) => mt.listExtensions().some((e) => e.extensionName === name);
for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial();
  if (!mt || mt.getBaseColorTexture() || !pr.getAttribute('COLOR_0')) continue;
  if ((mt.getAlphaMode() === 'BLEND' && extOf(mt, 'KHR_materials_unlit')) || extOf(mt, 'KHR_materials_sheen')) { m.removePrimitive(pr); pr.dispose(); }
}
// (the creator's skin: the opaque part with a painted map and vertex colours, the head)
if (skinTarget) for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial(), tex = mt && mt.getBaseColorTexture(), vc = pr.getAttribute('COLOR_0'), uv = pr.getAttribute('TEXCOORD_0');
  if (!tex || !vc || !uv || mt.getAlphaMode() !== 'OPAQUE' || pr.getAttribute('POSITION').getCount() < 20000) continue;
  const { data, info } = await sharp(Buffer.from(tex.getImage())).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const s = [0, 0, 0], e = [], u = [], N = vc.getCount();
  for (let i = 0; i < N; i++) {
    vc.getElement(i, e); uv.getElement(i, u);
    const x = Math.min(info.width - 1, Math.max(0, Math.floor(u[0] * info.width))), y = Math.min(info.height - 1, Math.max(0, Math.floor(u[1] * info.height))), o = (y * info.width + x) * 3;
    for (let k = 0; k < 3; k++) s[k] += lin(data[o + k] / 255) * e[k];
  }
  const g = s.map((v, k) => skinTarget[k] / (v / N));
  for (let i = 0; i < N; i++) { vc.getElement(i, e); vc.setElement(i, [e[0] * g[0], e[1] * g[1], e[2] * g[2], e.length > 3 ? e[3] : 1].slice(0, e.length)); }
  // (the creator's roughness map is drawn for its own lighting; in the world it left oily patches across the face)
  mt.setMetallicRoughnessTexture(null); mt.setRoughnessFactor(0.6);
  console.log('skin matched to the Blender head: gain', g.map((v) => v.toFixed(3)).join(' '));
}

// simplify only the head itself, one primitive at a time (a document-wide pass reached every piece again on each
// call, and the head, simplified twice, lost the shape of the mouth); the error is kept small so the lips, the nose
// and the lids hold their shape
for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial();
  if (pr.getAttribute('POSITION').getCount() > 100000 && mt && mt.getAlphaMode() === 'OPAQUE')
    simplifyPrimitive(pr, { simplifier: MeshoptSimplifier, ratio, error: 0.0001, lockBorder: true });
}

// The mouth's shading for the world: the sculpt round the mouth (the folds from the nose wings to the corners, the dip
// above the upper lip, the fold under the lower lip, the parting) is modelled for the creator's soft, scattering skin;
// under the game's top-down sun and plain shading every fold went dark, and together they drew a frown and a pursed
// pout round the painted lips. So the normals over the mouth are smoothed (the shape and silhouette stay as sculpted),
// fading out to the cheeks and chin: the lips read from their paint, the creases go.
const MOUTH = { x: 0, y: 2.3, z: 4.0, rx: 1.8, ry: 1.6, rz: 1.6 }, mouthIters = +(process.env.MOUTH_ITERS || 40), mouthMax = +(process.env.MOUTH_MAX || 0.9);
for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial(), P = pr.getAttribute('POSITION'), N = pr.getAttribute('NORMAL'), I = pr.getIndices();
  if (!(mt && mt.getAlphaMode() === 'OPAQUE' && mt.getBaseColorTexture() && N && I && P.getCount() > 20000)) continue;
  const n = P.getCount(), pos = P.getArray(), nor = N.getArray(), idx = I.getArray();
  const w = new Float32Array(n); let any = 0;
  for (let i = 0; i < n; i++) {
    const r = Math.hypot((pos[i * 3] - MOUTH.x) / MOUTH.rx, (pos[i * 3 + 1] - MOUTH.y) / MOUTH.ry, (pos[i * 3 + 2] - MOUTH.z) / MOUTH.rz);
    const t = Math.min(1, Math.max(0, (1 - r) / 0.5)); w[i] = mouthMax * t * t * (3 - 2 * t); if (w[i] > 0) any++;
  }
  if (any < 100) continue;
  // (neighbours from the triangles, for the vertices in the region and their rings)
  const nb = new Map(); const link = (a, b) => { if (w[a] > 0) { let s = nb.get(a); if (!s) nb.set(a, s = new Set()); s.add(b); } };
  for (let f = 0; f < idx.length; f += 3) { const a = idx[f], b = idx[f + 1], c = idx[f + 2]; link(a, b); link(a, c); link(b, a); link(b, c); link(c, a); link(c, b); }
  let cur = Float32Array.from(nor);
  for (let it = 0; it < mouthIters; it++) {
    const nx = Float32Array.from(cur);
    for (const [i, s] of nb) {
      let x = cur[i * 3], y = cur[i * 3 + 1], z = cur[i * 3 + 2];
      for (const j of s) { x += cur[j * 3]; y += cur[j * 3 + 1]; z += cur[j * 3 + 2]; }
      const l = Math.hypot(x, y, z) || 1; nx[i * 3] = x / l; nx[i * 3 + 1] = y / l; nx[i * 3 + 2] = z / l;
    }
    cur = nx;
  }
  for (const i of nb.keys()) {
    let x = nor[i * 3] + (cur[i * 3] - nor[i * 3]) * w[i], y = nor[i * 3 + 1] + (cur[i * 3 + 1] - nor[i * 3 + 1]) * w[i], z = nor[i * 3 + 2] + (cur[i * 3 + 2] - nor[i * 3 + 2]) * w[i];
    const l = Math.hypot(x, y, z) || 1; nor[i * 3] = x / l; nor[i * 3 + 1] = y / l; nor[i * 3 + 2] = z / l;
  }
  N.setArray(nor);
  console.log('mouth normals smoothed:', nb.size, 'vertices,', mouthIters, 'passes');
}

// The hair's texture for the world: in the creator a bump map, per-lock shading and its shader soften the fine strand
// stripes and the ragged cut-out at the tips; in the world, mipmapped and alpha-tested, both broke up into a speckle
// over the head. Blur the colour across the strands (they run down the texture) so the locks keep their bands but lose
// the pixel stripes, and soften the cut-out across too, so the tips end in a few clean points instead of a fringe of
// single-pixel hairs.
// (hairCut, env HAIR_CUT: where the softened cut-out starts; lower fills the gaps inside the fine strand clumps of the
// bangs, which at the world's size broke into a dotted, dark speckle)
const hairCut = +(process.env.HAIR_CUT || 0.15);
const hairTex = new Set();
for (const mt of headDoc.getRoot().listMaterials()) if (mt.getAlphaMode() === 'MASK' && mt.getBaseColorTexture()) hairTex.add(mt.getBaseColorTexture());
for (const t of hairTex) {
  const img = sharp(Buffer.from(t.getImage())).ensureAlpha();
  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
  const W = info.width, H = info.height, out = Buffer.from(data);
  const blurRow = (src, ch, sigma) => {      // a horizontal gaussian on one channel, wrapping (the texture tiles round the head)
    const r = Math.ceil(sigma * 2.5), k = []; let ks = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); ks += v; }
    const res = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let a = 0; for (let i = -r; i <= r; i++) a += k[i + r] * src[(y * W + ((x + i + W) % W)) * 4 + ch]; res[y * W + x] = a / ks; }
    return res;
  };
  for (let ch = 0; ch < 3; ch++) { const b = blurRow(data, ch, hairSoft * W / 1024); for (let i = 0; i < W * H; i++) out[i * 4 + ch] = Math.round(0.3 * data[i * 4 + ch] + 0.7 * b[i]); }
  const a = blurRow(data, 3, 2.5 * W / 1024);
  for (let i = 0; i < W * H; i++) { const v = a[i] / 255, s = Math.min(1, Math.max(0, (v - hairCut) / 0.4)); out[i * 4 + 3] = Math.round(255 * s * s * (3 - 2 * s)); }
  t.setImage(await sharp(out, { raw: { width: W, height: H, channels: 4 } }).png().toBuffer());
  console.log('softened the hair texture', W + 'x' + H);
}
// The hair's shading for the world: the creator darkens the locks between one another (their vertex colours) for its
// own soft light, and in the world, with the sun's hard shading on top, the bangs and the locks under them went dark
// and streaky. So the locks' shading is pulled most of the way to its mean (hairEven, default 0.85), and the hair gets a little of its own colour
// back as emission (hairLift, env HAIR_LIFT, default 0.45), which lifts its shadows without flattening the sunlit side.
const hairLift = +(process.env.HAIR_LIFT || 0.45), hairEven = +(process.env.HAIR_EVEN || 0.85);
for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial(), C = pr.getAttribute('COLOR_0');
  if (!mt || mt.getAlphaMode() !== 'MASK' || !mt.getBaseColorTexture()) continue;
  if (C && hairEven > 0) {
    const n = C.getCount(), e = [], mean = [0, 0, 0];
    for (let i = 0; i < n; i++) { C.getElement(i, e); for (let k = 0; k < 3; k++) mean[k] += e[k] / n; }
    for (let i = 0; i < n; i++) { C.getElement(i, e); for (let k = 0; k < 3; k++) e[k] = mean[k] + (e[k] - mean[k]) * (1 - hairEven); C.setElement(i, e); }
  }
  if (hairLift > 0) { mt.setEmissiveTexture(mt.getBaseColorTexture()); mt.setEmissiveFactor([hairLift, hairLift, hairLift]); }
}
await headDoc.transform(join({ keepNamed: false }));
// the joined hair strands (hundreds of ribbons): thinned along their length, borders free, so they keep their
// outline at the world's distance
for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial();
  if (mt && mt.getAlphaMode() === 'MASK' && pr.getAttribute('POSITION').getCount() > 20000)
    simplifyPrimitive(pr, { simplifier: MeshoptSimplifier, ratio: hairRatio, error: 0.002, lockBorder: false });
}
const map = mergeDocuments(doc, headDoc);
const hs = headDoc.getRoot().getDefaultScene() || headDoc.getRoot().listScenes()[0];
const added = hs.listChildren().map((n) => map.get(n));
for (const n of added) { n.detach && n.detach(); headNode.addChild(n); }
// (the merged document brings a scene of its own; keep only the game's)
const scenes = root.listScenes(), keep = scenes[0];
scenes.slice(1).forEach((s) => s.dispose());
root.setDefaultScene(keep);
await doc.transform(prune(), dedup(), unpartition());
for (const t of root.listTextures()) {
  const sz = t.getSize();
  if (sz && Math.max(sz[0], sz[1]) > 1024) console.warn('large texture', t.getName(), sz);
}
await io.write(outPath, doc);
const tri = (d) => d.getRoot().listMeshes().reduce((s, m) => s + m.listPrimitives().reduce((q, p) => q + (p.getIndices() ? p.getIndices().getCount() / 3 : 0), 0), 0);
console.log('wrote', outPath, 'meshes', root.listMeshes().length, 'triangles', Math.round(tri(doc)));
