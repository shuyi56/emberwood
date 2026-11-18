// The Warden's own model as the cape bench's body: her glb (blender/warden/export_glb.py), played the way the
// world plays it (src/world/main.js: every clip set to one time, the walk's phase following her stride).
//
// Her own cape (the cape, its lining and the two front panels) is re-skinned here to spring-bone chains hung
// from her Chest joint: the same layout and weighting the stand-in's cape has (`capeSkin` in dummy.js), with
// each bone's head on her cape's inner surface. This is what blender/warden/warden.py would do to port it, done
// in the page so the springs can be tried on her real cape and body now.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { capeSkin, CAPE } from './dummy.js';

// the world's timing for a modelled hero: a walk cycle in frames 1-25 at 24 fps, the idle stance at frame 40
const WALK_SECONDS = 24 / 24, IDLE_SECONDS = 39 / 24;
// her own cape's meshes
const OWN_CAPE = /^Cape(_|$)/;
// the centre of the ring her cape hangs round (z), and how far either side of straight behind its chains reach:
// her cape's back reaches 102° round each side, and its front panels on to about 146°
const CZ = -0.5, SPAN = 2.55;
const angleOf = (x, z) => Math.atan2(x, -(z - CZ));

export function loadModelBody(url) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(url, (res) => {
      const root = res.scene;
      root.name = 'Warden model';
      const bones = {};
      const rest = [];
      root.traverse((o) => {
        rest.push([o, o.position.clone(), o.quaternion.clone(), o.scale.clone()]);
        if (o.isBone && !bones[o.name]) bones[o.name] = o;
        if (!o.isMesh) return;
        o.castShadow = !(o.material && o.material.alphaTest > 0);
        o.receiveShadow = true;
        o.frustumCulled = false;
      });
      const mixer = new THREE.AnimationMixer(root);
      for (const clip of res.animations) {
        const a = mixer.clipAction(clip);
        a.setLoop(THREE.LoopOnce);
        a.clampWhenFinished = true;
        a.play();
      }
      const capeMeshes = [];
      root.traverse((o) => { if (o.isSkinnedMesh && OWN_CAPE.test(o.name)) capeMeshes.push(o); });
      const body = {
        root, bones, cape: null, hair: null, mixer, capeMeshes,
        // standing or walking, as the world does it: the walk at its phase, or the idle stance
        pose(ph, w) {
          mixer.setTime(w > 0.05 ? (ph / (Math.PI * 2)) * WALK_SECONDS : IDLE_SECONDS);
        },
        // back to the pose the glb was exported in, which the cape is laid out and bound against
        restPose() {
          mixer.stopAllAction();
          for (const [o, p, q, s] of rest) if (o !== root) { o.position.copy(p); o.quaternion.copy(q); o.scale.copy(s); }
          for (const a of mixer._actions) a.play();
          root.updateMatrixWorld(true);
        },
      };
      body.setCape = (o) => {
        body.restPose();
        reskinCape(body, o);
        if (!body.hair) reskinHair(body);
      };
      resolve(body);
    }, undefined, reject);
  });
}

// Hang her cape on spring-bone chains: `columns` chains of `rows + 1` bones under Chest, named cape_<c>_<r>,
// row 0 the band over the shoulders. Her rigid cape meshes are hidden and skinned copies take their place.
function reskinCape(body, { columns = 7, rows = 5 } = {}) {
  const { root, bones } = body;
  if (body.cape) {
    for (const m of body.cape.meshes) { root.remove(m); m.geometry.dispose(); }
    for (const b of body.cape.bones) if (b.name.endsWith('_0')) b.parent.remove(b);
  }
  root.updateMatrixWorld(true);
  const toRoot = new THREE.Matrix4().copy(root.matrixWorld).invert();
  // her cape's vertices in her own space, at rest
  const sources = body.capeMeshes.map((m) => {
    m.visible = false;
    const g = m.geometry.clone();
    g.applyMatrix4(new THREE.Matrix4().multiplyMatrices(toRoot, m.matrixWorld));
    return { m, g };
  });
  const pts = [];
  for (const { g } of sources) {
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) pts.push([p.getX(i), p.getY(i), p.getZ(i)]);
  }
  const top = Math.max(...pts.map((p) => p[1])), hemY = Math.min(...pts.map((p) => p[1]));
  const colA = (c) => -SPAN + 2 * SPAN * (c / (columns - 1));
  const rowY = [top - 0.3];
  for (let r = 0; r <= rows; r++) rowY.push(CAPE.shoulderY - (CAPE.shoulderY - hemY) * (r / rows));
  const layout = { colA, rowY, hemY, columns, rows };
  // a point on her cape's inner face at angle a and height y: the nearest ring of her cape's vertices there
  const surface = (a, y) => {
    for (let wa = 0.12, wy = 1.2; wa < 1.6; wa *= 1.6, wy *= 1.4) {
      let best = Infinity, n = 0;
      for (const [x, py, z] of pts) {
        if (Math.abs(py - y) > wy || Math.abs(angleOf(x, z) - a) > wa) continue;
        best = Math.min(best, Math.hypot(x, z - CZ));
        n++;
      }
      if (n >= 3) return new THREE.Vector3(Math.sin(a) * (best - 0.3), y, CZ - Math.cos(a) * (best - 0.3));
    }
    return new THREE.Vector3(Math.sin(a) * 8, y, CZ - Math.cos(a) * 8);
  };
  // the bones, from points in her space (whatever the chest bone's roll)
  const capeBones = [];
  const place = (b, parent, p) => {
    parent.add(b);
    b.position.copy(parent.worldToLocal(p.clone().applyMatrix4(root.matrixWorld)));
    b.quaternion.copy(parent.getWorldQuaternion(new THREE.Quaternion()).invert())
      .multiply(root.getWorldQuaternion(new THREE.Quaternion()));
    b.userData.rest = b.quaternion.clone();
    b.updateMatrixWorld(true);
  };
  for (let c = 0; c < columns; c++) {
    let parent = bones.Chest;
    for (let r = 0; r <= rows; r++) {
      const b = new THREE.Bone();
      b.name = `cape_${c}_${r}`;
      place(b, parent, surface(colA(c), rowY[r]));
      capeBones.push(b);
      parent = b;
    }
    const tip = new THREE.Object3D();
    tip.name = `cape_${c}_tip`;
    place(tip, parent, surface(colA(c), hemY + 0.5));
  }
  root.updateMatrixWorld(true);
  const skeleton = new THREE.Skeleton(capeBones);
  // her meshes, skinned to the chains
  const meshes = sources.map(({ m, g }) => {
    const p = g.attributes.position;
    const si = new Uint16Array(p.count * 4), sw = new Float32Array(p.count * 4);
    for (let i = 0; i < p.count; i++) {
      const k = capeSkin(angleOf(p.getX(i), p.getZ(i)), p.getY(i), layout);
      si.set(k.index, i * 4);
      sw.set(k.weight, i * 4);
    }
    g.setAttribute('skinIndex', new THREE.BufferAttribute(si, 4));
    g.setAttribute('skinWeight', new THREE.BufferAttribute(sw, 4));
    const material = Array.isArray(m.material) ? m.material.map((x) => x.clone()) : m.material.clone();
    for (const x of [].concat(material)) x.skinning = true;
    const sm = new THREE.SkinnedMesh(g, material);
    sm.name = m.name + ' (springs)';
    sm.castShadow = m.castShadow;
    sm.receiveShadow = true;
    sm.frustumCulled = false;
    sm.renderOrder = m.renderOrder;
    root.add(sm);
    sm.updateMatrixWorld(true);
    sm.bind(skeleton, sm.matrixWorld.clone());
    return sm;
  });
  body.cape = { bones: capeBones, meshes, mesh: meshes.find((m) => m.name === 'Cape (springs)') || meshes[0], lining: meshes.find((m) => /lining/.test(m.name)) || meshes[0], skeleton, columns, rows };
}

// Hang her hair on spring-bone chains from her Head (see src/world/hair-rig.js): a sheet of `columns` chains
// across her back and a lock of two chains in front of each shoulder, each `rows + 1` bones from the scalp down.
// Her hair mesh (the alpha-cut cards under the head) is hidden and a skinned copy takes its place.
const HAIR_GROUPS = [
  // [prefix, columns, first and last column's angle round the head (0 straight behind, + toward her left)]
  ['hair', 5, -1.2, 1.2],
  ['hairL', 2, 2.15, 2.85],
  ['hairR', 2, -2.85, -2.15],
];
const groupOf = (a) => (Math.abs(a) <= 1.65 ? 0 : a > 0 ? 1 : 2);
function reskinHair(body, { rows = 3 } = {}) {
  const { root } = body;
  const head = root.getObjectByName('Head');
  if (!head) return;
  let src = null;
  head.traverse((o) => {
    if (o.isMesh && !o.isSkinnedMesh && o.material && o.material.alphaTest > 0 &&
      (!src || o.geometry.attributes.position.count > src.geometry.attributes.position.count)) src = o;
  });
  if (!src) return;
  root.updateMatrixWorld(true);
  const toRoot = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const headInRoot = new THREE.Matrix4().multiplyMatrices(toRoot, head.matrixWorld);
  const toHead = headInRoot.clone().invert();
  src.visible = false;
  const g = src.geometry.clone();
  g.applyMatrix4(new THREE.Matrix4().multiplyMatrices(toRoot, src.matrixWorld));
  // her hair's vertices in the head's own axes (y up, +z her front)
  const p = g.attributes.position, v = new THREE.Vector3();
  const pts = new Float32Array(p.count * 3);
  let hemY = Infinity;
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).applyMatrix4(toHead);
    pts.set([v.x, v.y, v.z], i * 3);
    hemY = Math.min(hemY, v.y);
  }
  const ang = (x, z) => Math.atan2(x, -z);
  const rowY = [1.5, -1.5];
  for (let r = 2; r <= rows + 1; r++) rowY.push(rowY[1] - (rowY[1] - hemY) * ((r - 1) / rows));
  // where the hair is at angle a and height y: the middle of the strands there (head axes)
  const at = (a, y) => {
    for (let wa = 0.2, wy = 0.8; wa < 2; wa *= 1.6, wy *= 1.4) {
      let x = 0, z = 0, n = 0;
      for (let i = 0; i < p.count; i++) {
        const px = pts[i * 3], py = pts[i * 3 + 1], pz = pts[i * 3 + 2];
        if (Math.abs(py - y) > wy) continue;
        let d = Math.abs(ang(px, pz) - a);
        if (d > Math.PI) d = 2 * Math.PI - d;
        if (d > wa) continue;
        x += px; z += pz; n++;
      }
      if (n >= 8) return new THREE.Vector3(x / n, y, z / n);
    }
    return new THREE.Vector3(Math.sin(a) * 4, y, -Math.cos(a) * 4);
  };
  const place = (b, parent, ph) => {
    parent.add(b);
    const w = ph.clone().applyMatrix4(headInRoot).applyMatrix4(root.matrixWorld);
    b.position.copy(parent.worldToLocal(w));
    b.quaternion.copy(parent.getWorldQuaternion(new THREE.Quaternion()).invert())
      .multiply(root.getWorldQuaternion(new THREE.Quaternion()));
    b.userData.rest = b.quaternion.clone();
    b.updateMatrixWorld(true);
  };
  const bones = [], layouts = [];
  for (const [prefix, columns, a0, a1] of HAIR_GROUPS) {
    const colA = (c) => a0 + (a1 - a0) * (c / (columns - 1));
    layouts.push({ base: bones.length, layout: { colA, rowY, hemY, columns, rows } });
    for (let c = 0; c < columns; c++) {
      let parent = head;
      for (let r = 0; r <= rows; r++) {
        const b = new THREE.Bone();
        b.name = `${prefix}_${c}_${r}`;
        place(b, parent, at(colA(c), rowY[r]));
        bones.push(b);
        parent = b;
      }
      const tip = new THREE.Object3D();
      tip.name = `${prefix}_${c}_tip`;
      place(tip, parent, at(colA(c), hemY + 0.3));
    }
  }
  root.updateMatrixWorld(true);
  const skeleton = new THREE.Skeleton(bones);
  const si = new Uint16Array(p.count * 4), sw = new Float32Array(p.count * 4);
  for (let i = 0; i < p.count; i++) {
    const x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
    const a = ang(x, z);
    const { base, layout } = layouts[groupOf(a)];
    const k = capeSkin(a, y, layout);
    si.set(k.index.map((n) => n + base), i * 4);
    sw.set(k.weight, i * 4);
  }
  g.setAttribute('skinIndex', new THREE.BufferAttribute(si, 4));
  g.setAttribute('skinWeight', new THREE.BufferAttribute(sw, 4));
  const material = src.material.clone();
  material.skinning = true;
  const sm = new THREE.SkinnedMesh(g, material);
  sm.name = 'Hair (springs)';
  sm.castShadow = src.castShadow;
  sm.receiveShadow = true;
  sm.frustumCulled = false;
  sm.renderOrder = src.renderOrder;
  root.add(sm);
  sm.updateMatrixWorld(true);
  sm.bind(skeleton, sm.matrixWorld.clone());
  body.hair = { bones, mesh: sm, skeleton };
}
