// A walk baked from the base figure's Blender script (blender/base/female.py, through blender/base/bake_walk.py) to JSON, played
// on either bench body by phase. Each frame gives, per rig joint, `q`: its rotation in her space away from the
// pose the rig was bound in (xyzw), and `p`: where its pivot is (model units). So the walk can be tried on the
// cape and hair before a new warden.glb is exported. Only the root joint's pivot is taken from `p`; the others
// hang from it at this body's own proportions.
//
//   public/cape-lab/walk-new.json   the planted-feet walk as female.py walks now (PR #35: heel strike, a straighter
//                                   stance leg, the arms swinging back)
//   public/cape-lab/walk-main.json  main's walk baked the same way (it reproduces warden.glb's clip)
//   public/cape-lab/walk-run.json   the run (blender/base/female.py pose_run), baked the same way; it also gives its
//                                   stride, its time a cycle, the leg (thigh + shin) it was made on, and that it closes
//                                   the hands (`fists`: the body shows its Fist meshes)
import * as THREE from 'three';
import { PropertyBinding } from 'three';

export const WALKS = {
  clip: 'Her own walk',
  new: 'New walk (planted feet)',
  main: 'Main walk (baked)',
  run: 'Run (baked, from the walk thread)',
};

// each joint's parent in the walk's rig (a sided joint's parent is on its own side, unless it is the chest or
// pelvis; the neck hangs from its own joints, Cervical and Midneck, where the walk has them, or else from the chest)
const UP = { Chest: 'Pelvis', Cervical: 'Chest', Midneck: 'Cervical', Neck: ['Midneck', 'Chest'], Thigh: 'Pelvis', Shin: 'Thigh', Foot: 'Shin', UpperArm: 'Chest', Forearm: 'UpperArm', Hand: 'Forearm' };
const UP_SIDED = new Set(['Shin', 'Foot', 'Forearm', 'Hand']);

const cache = {};
export function loadBakedWalk(kind, base = '') {
  if (!cache[kind]) cache[kind] = fetch(`${base}cape-lab/walk-${kind}.json`).then((r) => r.json());
  return cache[kind];
}

// the node for a rig joint: GLTFLoader drops the dots (`Thigh.L` is `ThighL`) and adds `_1` when a joint shares
// its name with another node (`Neck_1`, `HandL_1`)
function findJoint(root, name) {
  const want = [name, PropertyBinding.sanitizeNodeName(name)];
  want.push(want[1] + '_1');
  let o = null;
  root.traverse((n) => { if (!o && n.isBone && want.includes(n.name)) o = n; });
  if (!o) root.traverse((n) => { if (!o && want.includes(n.name)) o = n; });
  return o;
}

// root: the body's root (her space); followers: [node, joint name] pairs for nodes outside the rig that ride a
// joint (her head rides the neck). Call it with the body in the pose the rig was bound in.
export function makeWalkPlayer(walk, root, followers = []) {
  root.updateMatrixWorld(true);
  const toRoot = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const inRoot = (o) => new THREE.Matrix4().multiplyMatrices(toRoot, o.matrixWorld);
  const joints = Object.keys(walk.bones[0]).map((name) => {
    const node = findJoint(root, name);
    if (!node) return null;
    const bind = inRoot(node), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
    bind.decompose(p, q, s);
    return { name, node, q, s, p };
  }).filter(Boolean);
  const byName = Object.fromEntries(joints.map((j) => [j.name, j]));
  // each joint's nearest driven ancestor: its pivot is carried from that one's by the walk's turn, at this body's
  // own offset, not put where the walk had it. The walk was baked on the base figure, whose neck joint sits 3
  // higher above the chest than the Warden's: taken as it was, her neck stretched and lifted her head and hair
  // off her collar
  // (her rig is flat, every joint a child of `Rig`, so the chain is by name)
  for (const j of joints) {
    const side = /\.[LR]$/.exec(j.name) && UP_SIDED.has(j.name.slice(0, -2)) ? j.name.slice(-2) : '';
    j.up = [].concat(UP[j.name.replace(/\.[LR]$/, '')] || []).map((u) => byName[u + side]).find(Boolean) || null;
    if (j.up) j.off = j.p.clone().sub(j.up.p);
  }
  const follow = followers.map(([node, of]) => node && byName[of] && { node, of: byName[of], bind: inRoot(node) })
    .filter(Boolean);
  const n = walk.bones.length;
  const q = new THREE.Quaternion(), q2 = new THREE.Quaternion(), p = new THREE.Vector3();
  const m = new THREE.Matrix4(), parentInv = new THREE.Matrix4(), T = new THREE.Matrix4();
  const toLocal = (node, mRoot) => {
    // the node's parent in her space, now
    parentInv.multiplyMatrices(toRoot, node.parent.matrixWorld).invert();
    m.multiplyMatrices(parentInv, mRoot).decompose(node.position, node.quaternion, node.scale);
    node.updateMatrixWorld(true);
  };
  // between keys, a Catmull-Rom curve through the four nearest (a straight blend between two keys turns the
  // walk's speed at every key, which the springs feel as a shake)
  const cr = (a0, a1, a2, a3, t) => 0.5 * (2 * a1 + (a2 - a0) * t + (2 * a0 - 5 * a1 + 4 * a2 - a3) * t * t
    + (3 * a1 - a0 - 3 * a2 + a3) * t * t * t);
  const curve = (arrs, t, size) => Array.from({ length: size }, (_, i) => cr(arrs[0][i], arrs[1][i], arrs[2][i], arrs[3][i], t));
  const sample = (name, f) => {
    const k = Math.floor(f), t = f - k;
    const keys = [-1, 0, 1, 2].map((d) => walk.bones[((k + d) % n + n) % n][name]);
    // (each quaternion on the same side as the one before it, so the curve doesn't swing the long way round)
    const qs = keys.map((K) => q2.fromArray(K.q).toArray());
    for (let i = 1; i < 4; i++) {
      if (qs[i].reduce((sum, x, c) => sum + x * qs[i - 1][c], 0) < 0) qs[i] = qs[i].map((x) => -x);
    }
    q.fromArray(curve(qs, t, 4)).normalize();
    p.fromArray(curve(keys.map((K) => K.p), t, 3));
  };
  return {
    // phase in radians, one cycle 2π
    apply(ph) {
      root.updateMatrixWorld(true);
      toRoot.copy(root.matrixWorld).invert();
      const f = ((ph / (Math.PI * 2)) % 1 + 1) % 1 * n;
      for (const j of joints) {
        sample(j.name, f);
        j.wq = (j.wq || new THREE.Quaternion()).copy(q);
        j.wp = (j.wp || new THREE.Vector3()).copy(p);
        if (j.up) j.wp.copy(j.off).applyQuaternion(j.up.wq).add(j.up.wp);
        toLocal(j.node, new THREE.Matrix4().compose(j.wp, q.clone().multiply(j.q), j.s));
      }
      for (const fl of follow) {
        q.copy(fl.of.wq); p.copy(fl.of.wp);
        // turn about the joint's pivot as it was bound, then carry it to where the pivot is now
        const mRoot = new THREE.Matrix4().makeTranslation(p.x, p.y, p.z)
          .multiply(T.makeRotationFromQuaternion(q))
          .multiply(new THREE.Matrix4().makeTranslation(-fl.of.p.x, -fl.of.p.y, -fl.of.p.z))
          .multiply(fl.bind);
        toLocal(fl.node, mRoot);
      }
    },
  };
}
