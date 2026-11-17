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
old.forEach((n) => { const m = n.getMesh(); n.dispose(); if (m) m.dispose(); });

// (merge the creator's scene and hang its nodes under the Head node)
await MeshoptSimplifier.ready;
await headDoc.transform(dedup(), weld({ tolerance: 0.0001 }));
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
