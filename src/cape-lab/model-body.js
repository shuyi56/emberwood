// The Warden's own model as the cape bench's body: her glb (blender/warden/export_glb.py), played the way the
// world plays it (src/world/main.js: every clip set to one time, the walk's phase following her stride).
//
// Her own cape (the cape, its lining and the two front panels) is re-skinned here to spring-bone chains hung
// from her Chest joint: the same layout and weighting the stand-in's cape has (`capeSkin` in dummy.js), with
// each bone's head on her cape's inner surface. This is what blender/warden/warden.py would do to port it, done
// in the page so the springs can be tried on her real cape and body now.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { capeSkin, CAPE, setCape } from './dummy.js';
import { underClothes } from '../world/layering.js';
import { makeColliders, WARDEN_COLLIDERS, WARDEN_MODEL_SETTINGS } from '../world/cape-rig.js';
import { makeWalkPlayer } from './baked-walk.js';

// the world's timing for a modelled hero: a walk cycle in frames 1-25 at 24 fps, the idle stance at frame 40
const WALK_SECONDS = 24 / 24, IDLE_SECONDS = 39 / 24;
// her own cape's meshes
const OWN_CAPE = /^Cape(_|$)/;
// the centre of the ring her cape hangs round (z), and how far either side of straight behind its chains reach:
// her cape's back reaches 102° round each side, and its front panels on to about 146°
const CZ = -0.5, SPAN = 2.55;
// a cape that runs further round (the tailored cloak, blender/cloak/, closes at the centre front): ?capespan=<radians>
// sets how far either side of the centre back its outer chains run
const spanOf = () => Number(new URLSearchParams(globalThis.location ? globalThis.location.search : '').get('capespan')) || SPAN;
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
      // her arms and legs drawn a layer under her cape, so a limb swinging into it never shows through the cloth
      // (src/world/layering.js; ?layer=0 draws them as they are)
      if (new URLSearchParams(location.search).get('layer') !== '0') {
        root.traverse((o) => { if (o.isSkinnedMesh && !OWN_CAPE.test(o.name)) underClothes(o); });
      }
      const body = {
        root, bones, cape: null, hair: null, mixer, capeMeshes,
        // standing or walking, as the world does it: the walk at its phase, or the idle stance
        pose(ph, w) {
          mixer.setTime(w > 0.05 ? (ph / (Math.PI * 2)) * WALK_SECONDS : IDLE_SECONDS);
          // a baked walk (baked-walk.js) in place of her clip's
          if (w > 0.05 && body.walkPlayer) body.walkPlayer.apply(ph);
        },
        walkPlayer: null,
        // walk: baked walk data, or null for her own clip
        setWalk(walk) {
          body.restPose();
          body.walkPlayer = walk ? makeWalkPlayer(walk, root, [[root.getObjectByName('Head'), 'Neck']]) : null;
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
    m.userData.replaced = true;
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
  const span = spanOf();
  const colA = (c) => -span + 2 * span * (c / (columns - 1));
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
  // where each bone's head (and each chain's tip) goes: on her cape's inner face, then out of her body's
  // colliders as she stands. Her cape is modelled pressed against her sleeves, so those points start inside
  // her arms; left there, the arm capsules would either shove the cape out into a bell or have to be shrunk
  // to nothing, and then a swinging hand comes through. The mesh keeps its modelled shape either way: it is
  // bound to the bones where they end up.
  const spots = [];
  for (let c = 0; c < columns; c++) {
    for (let r = 0; r <= rows; r++) spots.push({ c, r, p: surface(colA(c), rowY[r]) });
    spots.push({ c, r: 'tip', p: surface(colA(c), hemY + 0.5) });
  }
  clearOfBody(body, spots.filter((s) => s.r !== 0).map((s) => s.p));
  for (let c = 0; c < columns; c++) {
    let parent = bones.Chest;
    for (let r = 0; r <= rows; r++) {
      const b = new THREE.Bone();
      b.name = `cape_${c}_${r}`;
      place(b, parent, spots.find((s) => s.c === c && s.r === r).p);
      capeBones.push(b);
      parent = b;
    }
    const tip = new THREE.Object3D();
    tip.name = `cape_${c}_tip`;
    place(tip, parent, spots.find((s) => s.c === c && s.r === 'tip').p);
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

// Move points (in her space, at the pose the glb was exported in) out of her body's colliders, grown by the
// margin the springs keep from them, as she stands idle; the points ride her chest. Leaves her at rest pose.
function clearOfBody(body, points, settings = WARDEN_MODEL_SETTINGS) {
  const { root, bones } = body;
  const chest = bones.Chest;
  root.updateMatrixWorld(true);
  const toChest = new THREE.Matrix4().copy(chest.matrixWorld).invert().multiply(root.matrixWorld);
  const local = points.map((p) => p.clone().applyMatrix4(toChest));
  body.pose(0, 0);
  root.updateMatrixWorld(true);
  const colliders = makeColliders(root, WARDEN_COLLIDERS);
  for (const c of colliders) c.refresh();
  const extra = settings.hitRadius + settings.contactSoftness / 2 + 0.15;
  const w = new THREE.Vector3();
  for (const q of local) {
    w.copy(q).applyMatrix4(chest.matrixWorld);
    for (let it = 0; it < 12; it++) {
      let moved = false;
      for (const c of colliders) moved = c.push(w, extra) || moved;
      if (!moved) break;
    }
    q.copy(w).applyMatrix4(new THREE.Matrix4().copy(chest.matrixWorld).invert());
  }
  body.restPose();
  const fromChest = new THREE.Matrix4().copy(root.matrixWorld).invert().multiply(chest.matrixWorld);
  points.forEach((p, i) => p.copy(local[i]).applyMatrix4(fromChest));
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
// Her hair is modelled lying a little inside her cape's collar band at the back (from about 48 up to the collar),
// so wherever it met the band the cloth showed through the hair, worse as her head moves on her chest while she
// walks. Lift the hair there (geometry in her space, at rest) out over the cape's outer surface by CAPE_CLEAR,
// moving every layer of the hair at that spot alike, so it keeps its depth. (A preview: the lasting fix is the
// hairstyle in the Face Creator, or the band sitting closer to her neck in the cape.)
const CAPE_CLEAR = 0.6, BACK_HAIR = 0.75;   // (radians either side of straight behind her)
function clearOfCape(body, g) {
  if (!body.cape) return;
  const CZ = -0.5, NA = 96, NY = 2;     // bins: by angle round her, and half a unit of height
  const ang = (x, z) => Math.atan2(x, -(z - CZ));
  const bin = (x, y, z) => Math.floor((ang(x, z) + Math.PI) / (2 * Math.PI) * NA) + ':' + Math.round(y * NY);
  const outer = new Map();
  for (const m of body.cape.meshes) {
    const p = m.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i), k = bin(x, y, z);
      outer.set(k, Math.max(outer.get(k) ?? 0, Math.hypot(x, z - CZ)));
    }
  }
  const p = g.attributes.position, need = new Map();
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    if (Math.abs(ang(x, z)) > BACK_HAIR + 0.25) continue;
    const k = bin(x, y, z), rc = outer.get(k);
    if (rc === undefined) continue;
    need.set(k, Math.max(need.get(k) ?? 0, rc + CAPE_CLEAR - Math.hypot(x, z - CZ)));
  }
  // (each spot takes the most its neighbours need too, so the lifted hair stays smooth)
  const lift = (x, y, z) => {
    const a0 = Math.floor((ang(x, z) + Math.PI) / (2 * Math.PI) * NA), y0 = Math.round(y * NY);
    let d = 0;
    for (let da = -2; da <= 2; da++) for (let dy = -2; dy <= 2; dy++) {
      const n = need.get(((a0 + da) % NA + NA) % NA + ':' + (y0 + dy));
      if (n > 0) d = Math.max(d, n * (1 - (Math.abs(da) + Math.abs(dy)) / 6));
    }
    return d;
  };
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    // (only the hair down her back: the hair past her shoulders is modelled tucked under the cape, and lifting
    // it spread her hair out over the cape like a shawl)
    const side = Math.min(1, Math.max(0, (Math.abs(ang(x, z)) - BACK_HAIR) / 0.25));
    if (side >= 1) continue;
    const d = lift(x, y, z) * (1 - side);
    if (!d) continue;
    const r = Math.hypot(x, z - CZ);
    p.setXYZ(i, x * (r + d) / r, y, CZ + (z - CZ) * (r + d) / r);
  }
  p.needsUpdate = true;
}

// A ponytail (the creator's `ponytail` style): the strands that fall from the tie at the back of her crown to below
// her nape, found as whole strands (the hair's connected pieces) that reach down past TAIL_BELOW behind her head
// (head axes, her chin at 0). Null when there are too few for a tail.
const TAIL_BELOW = -1;
function findTail(g, pts) {
  const n = pts.length / 3, up = new Int32Array(n).map((_, i) => i);
  const find = (i) => { while (up[i] !== i) { up[i] = up[up[i]]; i = up[i]; } return i; };
  const idx = g.index ? g.index.array : null, tris = idx ? idx.length : n;
  for (let t = 0; t + 2 < tris; t += 3) {
    const a = find(idx ? idx[t] : t), b = find(idx ? idx[t + 1] : t + 1), c = find(idx ? idx[t + 2] : t + 2);
    up[b] = a; up[find(c)] = a;
  }
  const lo = new Map(), zs = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    lo.set(r, Math.min(lo.get(r) ?? Infinity, pts[i * 3 + 1]));
    const z = zs.get(r) || [0, 0]; z[0] += pts[i * 3 + 2]; z[1]++; zs.set(r, z);
  }
  const tail = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const r = find(i), [zSum, zN] = zs.get(r);
    if (lo.get(r) < TAIL_BELOW && zSum / zN < -1.5) { tail[i] = 1; count++; }
  }
  return count > 500 ? tail : null;
}

// The ponytail on one chain, `hairT_0_<row>`, from the tie down the tail: row 0 at the tie rides her head, as
// does the rest of her hair (drawn back tight over the skull), and the rows under it swing. Each tail vertex
// follows the bone its height falls on, blended with the next over BLEND either side of a joint.
function reskinPonytail(body, { src, g, pts, tail, headInRoot }, { rows = 5 } = {}) {
  const { root } = body;
  const head = root.getObjectByName('Head');
  const n = pts.length / 3;
  let top = -Infinity, tip = Infinity;
  for (let i = 0; i < n; i++) if (tail[i]) { tip = Math.min(tip, pts[i * 3 + 1]); top = Math.max(top, pts[i * 3 + 1]); }
  // the middle of the tail at height y (head axes), from the tail's own strands
  const at = (y) => {
    for (let w = 0.6; w < 6; w *= 1.5) {
      let x = 0, z = 0, k = 0;
      for (let i = 0; i < n; i++) if (tail[i] && Math.abs(pts[i * 3 + 1] - y) < w) { x += pts[i * 3]; z += pts[i * 3 + 2]; k++; }
      if (k >= 8) return new THREE.Vector3(x / k, y, z / k);
    }
    return new THREE.Vector3(0, y, -4);
  };
  // the tie (the strand-free piece the tail is gathered by sits at about two thirds of the way up the tail's
  // loop): the chain starts just under where the tail turns down
  const start = top - 2.5;
  const rowY = [start + 1];
  for (let r = 1; r <= rows + 1; r++) rowY.push(start - (start - (tip + 0.3)) * ((r - 1) / rows));
  const place = (b, parent, ph) => {
    parent.add(b);
    b.position.copy(parent.worldToLocal(ph.clone().applyMatrix4(headInRoot).applyMatrix4(root.matrixWorld)));
    b.quaternion.copy(parent.getWorldQuaternion(new THREE.Quaternion()).invert())
      .multiply(root.getWorldQuaternion(new THREE.Quaternion()));
    b.userData.rest = b.quaternion.clone();
    b.updateMatrixWorld(true);
  };
  const bones = [];
  let parent = head;
  for (let r = 0; r <= rows; r++) {
    const b = new THREE.Bone();
    b.name = `hairT_0_${r}`;
    place(b, parent, at(rowY[r]));
    bones.push(b);
    parent = b;
  }
  const tipNode = new THREE.Object3D();
  tipNode.name = 'hairT_0_tip';
  place(tipNode, parent, at(rowY[rows + 1]));
  root.updateMatrixWorld(true);
  const skeleton = new THREE.Skeleton(bones);
  const BLEND = 0.6;
  const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const y = pts[i * 3 + 1];
    sw[i * 4] = 1;
    if (!tail[i] || y >= rowY[1]) continue;
    // the swinging bone r runs from rowY[r] down to rowY[r + 1]
    let r = 1;
    while (r < rows && y < rowY[r + 1]) r++;
    let a = r, b = r, wb = 0;
    if (rowY[r] - y < BLEND) { b = r - 1; wb = 0.5 - 0.5 * (rowY[r] - y) / BLEND; }
    else if (r < rows && y - rowY[r + 1] < BLEND) { b = r + 1; wb = 0.5 - 0.5 * (y - rowY[r + 1]) / BLEND; }
    si[i * 4] = a; si[i * 4 + 1] = b;
    sw[i * 4] = 1 - wb; sw[i * 4 + 1] = wb;
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
  root.add(sm);
  sm.updateMatrixWorld(true);
  sm.bind(skeleton, sm.matrixWorld.clone());
  body.hair = { bones, mesh: sm, skeleton, ponytail: true };
}

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
  src.userData.replaced = true;   // (re-skinned below: the body toggle must not show it again)
  const g = src.geometry.clone();
  g.applyMatrix4(new THREE.Matrix4().multiplyMatrices(toRoot, src.matrixWorld));
  clearOfCape(body, g);
  // her hair's vertices in the head's own axes (y up, +z her front)
  const p = g.attributes.position, v = new THREE.Vector3();
  const pts = new Float32Array(p.count * 3);
  let hemY = Infinity;
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).applyMatrix4(toHead);
    pts.set([v.x, v.y, v.z], i * 3);
    hemY = Math.min(hemY, v.y);
  }
  const tail = findTail(g, pts);
  if (tail) return reskinPonytail(body, { src, g, pts, tail, headInRoot });
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

// The new mannequin as the bench's body (public/models/mannequin.glb: the bare figure of anatomy/mannequin-v14.blend,
// its skin on its own rig, which has the Warden's joint names). It has no cape of its own, so it wears the
// stand-in's (dummy.js), hung from its Chest, and no walk of its own, so it plays a baked one (?walk=new).
// headUrl: a model whose Head node carries a Face Creator head and hair (the Warden's glb), to wear in place of
// the mannequin's blank head; its hair is then hung on spring-bone chains as the Warden's is (reskinHair).
export function loadMannequinBody(url, headUrl = null) {
  const load = (u) => new Promise((ok, no) => new GLTFLoader().load(u, ok, undefined, no));
  return Promise.all([load(url), headUrl ? load(headUrl) : null]).then(([res, donor]) => new Promise((resolve) => {
    (() => {
      const root = res.scene;
      root.name = 'Mannequin';
      const bones = {}, rest = [];
      const skin = new THREE.MeshStandardMaterial({ color: 0xb9a79a, roughness: 0.85 });
      root.traverse((o) => {
        rest.push([o, o.position.clone(), o.quaternion.clone(), o.scale.clone()]);
        if (o.isBone && !bones[o.name]) bones[o.name] = o;
        if (!o.isMesh) return;
        // (a skinned mesh needs its own material with skinning on, in this three.js)
        o.material = o.isSkinnedMesh ? Object.assign(skin.clone(), { skinning: true }) : skin;
        o.castShadow = o.receiveShadow = true;
        o.frustumCulled = false;
      });
      root.updateMatrixWorld(true);
      const head = root.getObjectByName('Head'), theirs = donor && donor.scene.getObjectByName('Head');
      if (head && theirs) {
        donor.scene.updateMatrixWorld(true);
        // (their head's frame in their model, set in ours: both are the creator's, chin at the Head node)
        const wear = new THREE.Group();
        wear.name = 'Creator head';
        new THREE.Matrix4().copy(head.matrixWorld).invert().multiply(theirs.matrixWorld).decompose(wear.position, wear.quaternion, wear.scale);
        for (const c of [...head.children]) c.traverse((o) => { o.visible = false; o.userData.replaced = true; });   // (the body toggle must not show it again)
        for (const c of [...theirs.children]) wear.add(c);
        wear.traverse((o) => { if (o.isMesh) { o.castShadow = !(o.material && o.material.alphaTest > 0); o.receiveShadow = true; o.frustumCulled = false; } });
        head.add(wear);
        root.updateMatrixWorld(true);
      }
      const body = {
        root, bones, cape: null, hair: null, walkPlayer: null,
        // (her legs are longer than the Warden's and her soles lie below her rig's origin: lifted to stand on the ground)
        lift: -new THREE.Box3().setFromObject(root).min.y,
        pose(ph, w) {
          if (w > 0.02 && body.walkPlayer) body.walkPlayer.apply(ph); else body.restPose();
        },
        setWalk(walk) {
          body.restPose();
          body.walkPlayer = walk ? makeWalkPlayer(walk, root, [[root.getObjectByName('Head'), 'Neck']]) : null;
          // (a run gives its own pace: its stride, scaled to her legs, which are longer than the figure it was made on)
          const leg = bones.Thigh_L || bones.ThighL, foot = bones.FootL;
          const mine = leg && foot ? leg.getWorldPosition(new THREE.Vector3()).distanceTo(foot.getWorldPosition(new THREE.Vector3())) / root.scale.y : 0;
          body.gait = walk && walk.seconds_per_cycle ? { stride: walk.stride * (mine && walk.leg ? mine / walk.leg : 1), seconds: walk.seconds_per_cycle } : null;
        },
        restPose() {
          for (const [o, p, q, sc] of rest) if (o !== root) { o.position.copy(p); o.quaternion.copy(q); o.scale.copy(sc); }
          root.updateMatrixWorld(true);
        },
      };
      body.setCape = (o) => { body.restPose(); setCape(body, o); body.cape.meshes = [body.cape.mesh, body.cape.lining]; if (!body.hair && theirs) reskinHair(body); };
      resolve(body);
    })();
  }));
}
