// Put the Face Creator's head and hair (export-head.js) on the game's Ranger (public/models/ranger.glb), in place of
// the head, hair and face paint from Blender. The body, clothes, skinning and the walk and idle animations are kept
// as exported; the creator's parts go under the Head node, whose frame is the creator's (chin at 0, face toward +z).
//   node merge-head.mjs ranger.glb creator-head.glb out.glb [headRatio] [hairRatio]
// The creator's head is sculpted far finer than the game needs: it is simplified to headRatio of its triangles
// (default 0.12), its hair strands joined by material, and every texture capped at 1024.
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { mergeDocuments, dedup, join, weld, simplify, prune, unpartition } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';

const [, , basePath, headPath, outPath, ratioArg] = process.argv;
const ratio = parseFloat(ratioArg || '0.12'), hairRatio = parseFloat(process.argv[6] || '0.4');
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
old.forEach((n) => { const m = n.getMesh(); n.dispose(); if (m) m.dispose(); });

// (merge the creator's scene and hang its nodes under the Head node)
await MeshoptSimplifier.ready;
await headDoc.transform(dedup(), weld({ tolerance: 0.0001 }));
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

// simplify only the big pieces (the head, the eyes, the hair cap); the strands are left as built
for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const n = pr.getAttribute('POSITION').getCount();
  if (n > 20000) await headDoc.transform(simplify({ simplifier: MeshoptSimplifier, ratio, error: 0.0004, lockBorder: true }));
}
await headDoc.transform(join({ keepNamed: false }));
// the joined hair strands (hundreds of ribbons): thinned along their length, borders free, so they keep their
// outline at the world's distance
for (const m of headDoc.getRoot().listMeshes()) for (const pr of m.listPrimitives()) {
  const mt = pr.getMaterial();
  if (mt && mt.getAlphaMode() === 'MASK' && pr.getAttribute('POSITION').getCount() > 20000) {
    await headDoc.transform(simplify({ simplifier: MeshoptSimplifier, ratio: hairRatio, error: 0.002, lockBorder: false }));
  }
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
