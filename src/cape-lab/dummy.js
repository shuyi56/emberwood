// A stand-in for the Warden: her joints (names and rest positions from public/models/warden.glb), a plain
// mannequin over them, a walk made in code, and a cape of spring-bone chains skinned like the one
// blender/warden/warden.py will make. All in her model units (80 tall, +z her front).
import * as THREE from 'three';

// rest positions of her joints, from the Warden's glb
const J = {
  Pelvis: [0, 31, 0], Chest: [0, 41, 0], Neck: [0, 51, 0],
  'UpperArm.L': [5.75, 49, -0.2], 'Forearm.L': [5.75, 39.2, -0.2], 'Hand.L': [5.75, 29.4, -0.2],
  'UpperArm.R': [-5.75, 49, -0.2], 'Forearm.R': [-5.75, 39.2, -0.2], 'Hand.R': [-5.75, 29.4, -0.2],
  'Thigh.L': [3.3, 31, 0], 'Shin.L': [3.3, 17, 0], 'Foot.L': [3.3, 3.4, 0],
  'Thigh.R': [-3.3, 31, 0], 'Shin.R': [-3.3, 17, 0], 'Foot.R': [-3.3, 3.4, 0],
};
const PARENT = {
  Pelvis: 'Rig', Chest: 'Pelvis', Neck: 'Chest',
  'UpperArm.L': 'Chest', 'Forearm.L': 'UpperArm.L', 'Hand.L': 'Forearm.L',
  'UpperArm.R': 'Chest', 'Forearm.R': 'UpperArm.R', 'Hand.R': 'Forearm.R',
  'Thigh.L': 'Pelvis', 'Shin.L': 'Thigh.L', 'Foot.L': 'Shin.L',
  'Thigh.R': 'Pelvis', 'Shin.R': 'Thigh.R', 'Foot.R': 'Shin.R',
};

// The cape's shape: an open ring around her back, an ellipse at each height. `t` runs from the collar (0) to
// the hem (1); `a` is the angle round her, 0 straight behind. The real cape reaches 52.7 at the collar,
// spreads over the shoulders to about 8.5 either side by y 47, and flares to 14.6 at the hem (y 19).
export const CAPE = {
  collarY: 53, shoulderY: 47, hemY: 19,
  span: 2.65,           // radians either side of the centre back (the front edges at about 152°)
};
function ringAt(y) {
  // half-width (x), half-depth (z) and the ring's centre z at height y
  const { collarY, shoulderY, hemY } = CAPE;
  if (y >= shoulderY) {
    const u = (collarY - y) / (collarY - shoulderY);          // 0 at the collar, 1 at the shoulders
    const e = Math.sin(u * Math.PI / 2);
    return { rx: 3.8 + (8.6 - 3.8) * e, rz: 3.6 + (5.2 - 3.6) * e, cz: -0.4 };
  }
  const u = (shoulderY - y) / (shoulderY - hemY);              // 0 at the shoulders, 1 at the hem
  return { rx: 8.6 + (14.4 - 8.6) * Math.pow(u, 1.2), rz: 5.2 + (7.2 - 5.2) * u, cz: -0.4 - 0.5 * u };
}
function capePoint(a, y, inset = 0, out = new THREE.Vector3()) {
  const r = ringAt(y);
  return out.set((r.rx - inset) * Math.sin(a), y, r.cz - (r.rz - inset) * Math.cos(a));
}

function mat(color, extra = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0, ...extra });
}
// r128 has no CapsuleGeometry: a lathe of a half circle, a straight run and a half circle, along +y from 0 to len
function capsuleGeo(r, len, seg = 14) {
  const pts = [];
  for (let i = 0; i <= seg; i++) { const t = -Math.PI / 2 + (i / seg) * Math.PI / 2; pts.push(new THREE.Vector2(Math.cos(t) * r, Math.sin(t) * r)); }
  for (let i = 0; i <= seg; i++) { const t = (i / seg) * Math.PI / 2; pts.push(new THREE.Vector2(Math.cos(t) * r, len + Math.sin(t) * r)); }
  pts[0].x = 0.0001; pts[pts.length - 1].x = 0.0001;
  return new THREE.LatheGeometry(pts, 20);
}
// a capsule mesh from a to b (in a parent's space)
function limb(a, b, r, material) {
  const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
  const len = va.distanceTo(vb);
  const m = new THREE.Mesh(capsuleGeo(r, len), material);
  m.position.copy(va);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
  m.castShadow = true;
  return m;
}

export function makeDummy({ columns = 5, rows = 5 } = {}) {
  const root = new THREE.Group();
  root.name = 'Warden stand-in';
  const rig = new THREE.Object3D();
  rig.name = 'Rig';
  root.add(rig);
  const bones = { Rig: rig };
  for (const name of Object.keys(J)) {
    const b = new THREE.Bone();
    b.name = name;
    bones[name] = b;
  }
  for (const [name, p] of Object.entries(PARENT)) {
    const b = bones[name], par = bones[p];
    const pp = p === 'Rig' ? [0, 0, 0] : J[p];
    b.position.set(J[name][0] - pp[0], J[name][1] - pp[1], J[name][2] - pp[2]);
    par.add(b);
  }
  bones.rest = Object.fromEntries(Object.keys(J).map((n) => [n, bones[n].position.clone()]));

  // the mannequin: each piece rides one joint
  const skin = mat(0x9a9488), cloth = mat(0x55604f);
  const rel = (name, p) => [p[0] - J[name][0], p[1] - J[name][1], p[2] - J[name][2]];
  const add = (name, mesh) => { bones[name].add(mesh); return mesh; };
  // torso: a squashed lathe from the hips to the shoulders, the robe's back 4 behind the spine
  const torsoPts = [[0.01, -3], [6.3, -2], [6.6, 2], [6.0, 6], [6.6, 9], [6.9, 11.5], [4.5, 12.6], [0.01, 12.8]]
    .map(([x, y]) => new THREE.Vector2(x, y));
  const torso = add('Pelvis', new THREE.Mesh(new THREE.LatheGeometry(torsoPts, 24), cloth));
  torso.scale.set(1, 1, 0.68);
  torso.castShadow = true;
  // robe skirt over the thighs (rides the pelvis; the real skirt follows the thighs a little)
  const skirtPts = [[6.3, 0], [7.2, -5], [8.0, -9]].map(([x, y]) => new THREE.Vector2(x, y));
  const skirt = add('Pelvis', new THREE.Mesh(new THREE.LatheGeometry(skirtPts, 24), mat(0x55604f, { side: THREE.DoubleSide })));
  skirt.scale.set(1, 1, 0.72);
  add('Neck', limb([0, 0, 0], [0, 3, 0], 1.9, skin));
  const head = add('Neck', new THREE.Mesh(new THREE.SphereGeometry(5, 24, 18), skin));
  head.position.set(0, 9.5, 0.6);
  head.scale.set(0.92, 1.12, 1);
  head.castShadow = true;
  for (const s of ['L', 'R']) {
    add('UpperArm.' + s, limb([0, -1.5, 0], rel('UpperArm.' + s, J['Forearm.' + s]), 2.0, cloth));
    add('Forearm.' + s, limb([0, 0, 0], rel('Forearm.' + s, J['Hand.' + s]), 1.8, cloth));
    add('Hand.' + s, limb([0, 0, 0], [0, -4, 0.3], 1.4, skin));
    add('Thigh.' + s, limb([0, 0, 0], rel('Thigh.' + s, J['Shin.' + s]), 3.0, mat(0x4b4639)));
    add('Shin.' + s, limb([0, 0, 0], rel('Shin.' + s, J['Foot.' + s]), 2.5, mat(0x4b4639)));
    const foot = add('Foot.' + s, limb([0, -1.5, -1.5], [0, -1.5, 5], 2.0, mat(0x3a2f25)));
    foot.scale.set(1.1, 0.8, 1);
  }

  const cape = makeCape(bones, { columns, rows });
  root.add(cape.mesh, cape.lining);
  return { root, rig, bones, cape };
}

// The cape: `columns` chains round her back, each of `rows + 1` bones: row 0 is the band from the collar to
// the shoulders (it rides the chest), rows 1..rows swing. Named cape_<column>_<row>, parented to Chest.
export function makeCape(bones, { columns = 5, rows = 5 } = {}) {
  const { collarY, shoulderY, hemY, span } = CAPE;
  const chest = bones.Chest;
  const colA = (c) => -span * 0.86 + (2 * span * 0.86) * (c / (columns - 1));   // outer chains just inside the edges
  // the heights of each row's head: the band, then even rows down to the hem
  const rowY = [collarY + 0.4];
  for (let r = 0; r <= rows; r++) rowY.push(shoulderY - (shoulderY - hemY) * (r / rows));
  // rowY[k] is the head of row k (k = 0..rows), rowY[rows + 1] the hem
  const capeBones = [];
  const chestW = new THREE.Vector3(...J.Chest);
  for (let c = 0; c < columns; c++) {
    let parent = chest, parentW = chestW;
    for (let r = 0; r <= rows; r++) {
      const b = new THREE.Bone();
      b.name = `cape_${c}_${r}`;
      // heads sit on the cloth's inner face
      const w = capePoint(colA(c), rowY[r], 0.3);
      b.position.copy(w).sub(parentW);
      parent.add(b);
      capeBones.push(b);
      parent = b; parentW = w;
    }
    // a tip to give the last bone its length and direction
    const tip = new THREE.Object3D();
    tip.name = `cape_${c}_tip`;
    tip.position.copy(capePoint(colA(c), hemY, 0.3)).sub(parentW);
    parent.add(tip);
  }
  const boneIndex = (c, r) => c * (rows + 1) + r;

  // the cloth: a grid round the back (U) and down (V)
  const NU = 49, NV = 46;
  const make = (inset, lining) => {
    const pos = [], idx = [], si = [], sw = [], uv = [];
    for (let iv = 0; iv <= NV; iv++) {
      const t = iv / NV;
      const y = collarY + (hemY - collarY) * t;
      for (let iu = 0; iu <= NU; iu++) {
        const a = -span + 2 * span * (iu / NU);
        const p = capePoint(a, y, inset);
        pos.push(p.x, p.y, p.z);
        uv.push(iu / NU, t);
        // columns: blend the two chains either side by angle
        const fc = Math.min(columns - 1, Math.max(0, (a - colA(0)) / (colA(1) - colA(0))));
        const c0 = Math.min(columns - 2, Math.floor(fc)), wc1 = fc - c0, wc0 = 1 - wc1;
        // rows: rigid in the middle of a bone, blended at its joints
        let fr;
        if (y >= rowY[1]) fr = (rowY[0] - y) / (rowY[0] - rowY[1]);               // in the band: 0..1
        else fr = 1 + (rowY[1] - y) / ((rowY[1] - hemY) / rows);                  // 1..rows+1
        const s = Math.min(rows, Math.max(0, fr - 0.5));
        const r0 = Math.min(rows - 1, Math.floor(s)), wr1 = Math.min(1, s - r0), wr0 = 1 - wr1;
        si.push(boneIndex(c0, r0), boneIndex(c0, r0 + 1), boneIndex(c0 + 1, r0), boneIndex(c0 + 1, r0 + 1));
        sw.push(wc0 * wr0, wc0 * wr1, wc1 * wr0, wc1 * wr1);
      }
    }
    const W = NU + 1;
    for (let iv = 0; iv < NV; iv++) for (let iu = 0; iu < NU; iu++) {
      const a = iv * W + iu, b = a + 1, c = a + W, d = c + 1;
      // outward faces: counter-clockwise seen from outside
      if (lining) idx.push(a, c, b, b, c, d); else idx.push(a, b, c, b, d, c);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4));
    g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  };
  // bind at rest: the bones' world matrices now (the stand-in stands at the origin)
  let top = chest;
  while (top.parent) top = top.parent;
  top.updateMatrixWorld(true);
  const skeleton = new THREE.Skeleton(capeBones);
  const mesh = new THREE.SkinnedMesh(make(0, false), mat(0x3f5a78, { side: THREE.FrontSide, skinning: true }));
  mesh.name = 'Cape';
  const lining = new THREE.SkinnedMesh(make(0.25, true), mat(0x8a2a24, { side: THREE.FrontSide, skinning: true }));
  lining.name = 'Cape lining';
  for (const m of [mesh, lining]) {
    m.castShadow = true;
    m.receiveShadow = true;
    m.frustumCulled = false;
  }
  mesh.bind(skeleton, new THREE.Matrix4());
  lining.bind(skeleton, new THREE.Matrix4());
  return { mesh, lining, bones: capeBones, skeleton, columns, rows };
}

// ---- the walk, in code ----
// A walk cycle of the Warden's pace: one cycle (two steps) covers 46 units (STRIDE in blender/ranger/ranger.py).
export const CYCLE = 46;

const ease = (x) => x * x * (3 - 2 * x);
// pose the rig for walk phase `ph` (radians, one cycle 2π) at walk weight `w` (0 standing, 1 full walk; above 1
// is a run: longer, faster swings) and time `t` (for breathing)
export function pose(bones, ph, w, t = 0) {
  const R = bones.rest;
  const s = Math.sin(ph), c = Math.cos(ph);
  const run = Math.max(0, w - 1);
  const wk = Math.min(1, w);
  const P = bones.Pelvis;
  P.position.copy(R.Pelvis);
  // two bobs per cycle (low as the feet pass), a sway over the standing foot
  P.position.y += wk * (-0.9 + 0.9 * Math.cos(2 * ph)) + (1 - wk) * 0.15 * Math.sin(t * 2.1);
  P.position.x += wk * 0.7 * s;
  P.rotation.set(0.05 * wk + 0.12 * run, wk * 0.1 * s, wk * -0.04 * c);
  const C = bones.Chest;
  C.position.copy(R.Chest);
  C.rotation.set(0.03 * wk + 0.1 * run + (1 - wk) * 0.01 * Math.sin(t * 2.1), wk * -0.2 * s, wk * 0.03 * c);
  bones.Neck.rotation.set(-0.04 * wk, wk * 0.1 * s, 0);
  const amp = 0.42 + 0.25 * run;
  for (const [side, off] of [['L', 0], ['R', Math.PI]]) {
    const p = ph + off, sp = Math.sin(p), cp = Math.cos(p);
    // thigh swings forward (-x rotation) on sin > 0
    bones['Thigh.' + side].rotation.set(-wk * amp * sp, 0, 0);
    // the knee folds through the swing (forward-moving leg, the half where cos < 0 leading up to contact)
    const swing = Math.max(0, Math.sin(p + 0.9));
    bones['Shin.' + side].rotation.set(wk * (0.08 + (0.75 + 0.5 * run) * ease(swing)), 0, 0);
    bones['Foot.' + side].rotation.set(wk * (-0.15 * cp), 0, 0);
    // arms swing against the legs, the elbow bending as the arm comes forward
    const sgn = side === 'L' ? 1 : -1;
    bones['UpperArm.' + side].rotation.set(wk * (0.32 + 0.3 * run) * sp, 0, sgn * 0.06);
    bones['Forearm.' + side].rotation.set(-wk * (0.18 + 0.25 * Math.max(0, sp)) - 0.08 - 0.5 * run, 0, 0);
  }
}
