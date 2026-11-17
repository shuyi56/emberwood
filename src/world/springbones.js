// Spring bones: secondary motion for capes, skirts and hair, solved in the game every frame.
//
// The model follows the VRM spring-bone model (pixiv three-vrm): each joint's tail is a point integrated with
// Verlet. It keeps its momentum, is pulled back toward the direction it has at rest relative to its parent
// (`stiffness`), and falls (`gravity`). The tail is then pushed out of the body's colliders and put back at the
// bone's length from its head, and the bone is turned to point at it. For cloth, three things are added that
// VRM leaves out:
//
//  - Two kinds of drag. `damping` takes away the tail's swing relative to the body (it settles a sway), and
//    `airDrag` takes away its speed through the air (it makes the cloth trail when she walks). VRM has one
//    drag for both, so a cape that settles well also streams out behind her like a flag.
//  - Fixed sub-steps. The solver always advances in steps of `STEP` seconds whatever the frame rate, with the
//    body's pose blended between the last frame and this one across the steps, so the motion is the same at
//    30, 60 or 144 fps. A frame longer than `MAX_STEPS` steps is clamped (a dropped frame doesn't throw the
//    cape), and a looping walk settles into a cycle that repeats.
//  - Links across chains. Neighbouring chains (the columns of a cape) keep the distance between their tails at
//    each row near what it is at rest, and each chain's swing is pulled toward its neighbours' (`sheetBend`),
//    so the columns move as one sheet: the cloth doesn't part, fold through itself or go jagged at the hem.
//
// Units are the model's own (the game's models are 80 units tall, so a metre is about 46 units), and they
// follow the model's scale in the world: the same settings work on the bench and on the scaled-down hero.
//
// It runs on any THREE.Object3D joints (bones of a skinned mesh, or plain groups), after the animation has
// posed their parents: call `update(dt)` after the mixer, with world matrices up to date.
import * as THREE from 'three';

export const STEP = 1 / 120;          // seconds per solver step
export const MAX_STEPS = 12;
const SETTLE_STEPS = 120;             // steps of settling on reset          // at most this many steps per update (0.1 s): a longer frame is dropped

export const DEFAULTS = {
  stiffness: 250,       // pull toward the rest direction (model units / s^2)
  damping: 0.15,        // share of the swing (velocity relative to the body) lost per 1/60 s
  airDrag: 0.02,        // share of the speed through the air lost per 1/60 s
  gravity: 450,         // fall (model units / s^2; 450 is real gravity at 46 units a metre)
  gravityDir: [0, -1, 0],
  hitRadius: 1.0,       // the tail's radius against colliders, in model units
  contactSoftness: 0,   // the band (model units) over which a collider's push eases in
  linkStiffness: 0.5,   // 0..1: how hard neighbouring chains keep their spacing each step
  linkSlack: 0.15,      // spacing may grow or shrink this share of its rest length freely
  // 0..1: how much each chain's swing is pulled toward its neighbours' at the same row, every step. It makes
  // the columns bend as one sheet, so the hem stays a smooth curve instead of going jagged in a hard turn
  sheetBend: 0.35,
  // 0..1: how much of the character's own movement in the world carries the cloth directly (see `root`)
  follow: 0,
  // how far a bone may turn away from its rest direction (radians); keeps a cape from flipping up over the
  // shoulders in a hard turn
  maxAngle: 1.2,
};

const _v0 = new THREE.Vector3(), _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();
const _q0 = new THREE.Quaternion(), _qf = new THREE.Quaternion();
const _m0 = new THREE.Matrix4();
const _one = new THREE.Vector3(1, 1, 1), _zero = new THREE.Vector3();
const BONE_SAMPLES = [0.5, 1];      // where along a bone it is tested against the colliders
const _q1 = new THREE.Vector3(), _q2 = new THREE.Vector3();   // (Collider.push uses _v1.._v3)

// A collider is a capsule between two points, or a sphere when they are the same. Each point is a node's origin
// plus an offset given in the axes of `frame` (the model's root), so colliders can be placed between joints of
// any rig without knowing how its bones are rolled: `{ node: chest, offset: [0, 0, -2] }` is 2 units behind
// the chest joint, whichever way the chest bone points.
export class Collider {
  constructor(frame, a, b, radius) {
    this.frame = frame;
    this.a = { node: a.node, offset: new THREE.Vector3(...(a.offset || [0, 0, 0])) };
    b = b || a;
    this.b = { node: b.node, offset: new THREE.Vector3(...(b.offset || [0, 0, 0])) };
    this.radius = radius;
    this.wa = new THREE.Vector3(); this.wb = new THREE.Vector3();          // where it is for this step
    this.pa = new THREE.Vector3(); this.pb = new THREE.Vector3();          // last frame
    this.ca = new THREE.Vector3(); this.cb = new THREE.Vector3();          // this frame
    this.scale = 1;
  }
  // read where it is now (world)
  refresh() {
    this.frame.getWorldQuaternion(_qf);
    this.scale = _v0.setFromMatrixScale(this.frame.matrixWorld).x;
    for (const [end, out] of [[this.a, this.wa], [this.b, this.wb]]) {
      out.setFromMatrixPosition(end.node.matrixWorld)
        .add(_v1.copy(end.offset).applyQuaternion(_qf).multiplyScalar(this.scale));
    }
  }
  // push point p (world) out of the surface grown by `extra`. With `soft`, the push eases in over a band `soft`
  // wide that ends at that surface (it starts half of `soft` outside it, and holds the point on the surface once
  // it would go deeper), so cloth meets a moving leg smoothly instead of being snapped flat against it every
  // step. Returns true if it moved.
  push(p, extra, soft = 0) {
    const ab = _v1.subVectors(this.wb, this.wa);
    const len2 = ab.lengthSq();
    let t = len2 > 1e-9 ? _v2.subVectors(p, this.wa).dot(ab) / len2 : 0;
    t = Math.min(1, Math.max(0, t));
    const c = _v2.copy(this.wa).addScaledVector(ab, t);
    const d = _v3.subVectors(p, c);
    const band = soft * this.scale, r = (this.radius + extra) * this.scale - band / 2;
    const dist = d.length();
    if (dist >= r + band) return false;
    const x = r + band - dist;
    const to = x < band ? dist + x * x / (2 * band) : r + band / 2;
    if (dist < 1e-6) d.set(0, 0, -1); else d.divideScalar(dist);
    p.copy(c).addScaledVector(d, to);
    return true;
  }
}

class Joint {
  constructor(bone, child) {
    this.bone = bone;
    this.restQuat = bone.quaternion.clone();
    // the bone's axis toward its tail, in its own space (the chain fills in the last one's)
    this.axis = child ? child.position.clone() : null;
    this.length = 0;
    this.tail = new THREE.Vector3();
    this.prevTail = new THREE.Vector3();
    this.head = new THREE.Vector3();
    this.prevHead = new THREE.Vector3();
    this.next = new THREE.Vector3();
    this.pre = new THREE.Vector3();
    this.restDir = new THREE.Vector3();
    this.worldLength = 0;
  }
}

export class SpringChain {
  // bones: from the root (whose parent the animation poses) to the tip. A tip's tail is taken from a child
  // object if it has one (a `<name>_tip` empty), or else carries on the bone before it.
  constructor(bones, settings = {}) {
    this.settings = { ...DEFAULTS, ...settings };
    this.joints = bones.map((b, i) => new Joint(b, bones[i + 1] || b.children.find((c) => !c.isMesh)));
    const last = this.joints[this.joints.length - 1];
    if (!last.axis) {
      const prev = this.joints[this.joints.length - 2];
      last.axis = prev ? prev.axis.clone() : new THREE.Vector3(0, -1, 0);
    }
    for (const j of this.joints) j.length = j.axis.length();
  }
}

export class SpringBones {
  // root: the character's root object. With `follow` set, part of its movement in the world (walking and,
  // above all, turning) carries the cloth along directly instead of only through the springs.
  constructor({ chains = [], colliders = [], links = [], root = null } = {}) {
    this.chains = chains;
    this.root = root && {
      node: root, prev: new THREE.Matrix4(), cur: new THREE.Matrix4(), step: new THREE.Matrix4(), last: new THREE.Matrix4(),
      pp: new THREE.Vector3(), pq: new THREE.Quaternion(), ps: new THREE.Vector3(),
      cp: new THREE.Vector3(), cq: new THREE.Quaternion(), cs: new THREE.Vector3(),
    };
    this.colliders = colliders;
    // links: [chainA, chainB] pairs whose joints, row by row, keep their spacing
    this.links = links.map(([a, b]) => ({ a, b, rest: [] }));
    // the objects the chains hang from: their pose is blended across a frame's steps
    this.anchors = [...new Set(chains.map((c) => c.joints[0].bone.parent))].map((node) => ({
      node, prev: new THREE.Matrix4(), cur: new THREE.Matrix4(),
      pp: new THREE.Vector3(), pq: new THREE.Quaternion(), ps: new THREE.Vector3(),
      cp: new THREE.Vector3(), cq: new THREE.Quaternion(), cs: new THREE.Vector3(),
    }));
    this.acc = 0;
    this.ready = false;
  }

  // Put every tail at rest under the current pose, with no velocity (on load, and after a teleport).
  reset() {
    for (const ch of this.chains) {
      for (const j of ch.joints) j.bone.quaternion.copy(j.restQuat);
      ch.joints[0].bone.updateMatrixWorld(true);
      for (const j of ch.joints) {
        this._restTail(j, j.tail);
        j.prevTail.copy(j.tail);
        j.head.setFromMatrixPosition(j.bone.matrixWorld);
        j.prevHead.copy(j.head);
      }
    }
    for (const L of this.links) {
      L.rest = L.a.joints.map((ja, i) => {
        const jb = L.b.joints[i];
        return jb ? ja.tail.distanceTo(jb.tail) / (_v0.setFromMatrixScale(ja.bone.matrixWorld).x || 1) : 0;
      });
    }
    for (const a of this.anchors) a.prev.copy(a.node.matrixWorld);
    if (this.root) this.root.prev.copy(this.root.node.matrixWorld);
    for (const c of this.colliders) { c.refresh(); c.pa.copy(c.wa); c.pb.copy(c.wb); }
    // let the cloth settle against the body as it stands, without gaining speed, so it doesn't jump on the
    // first frames
    for (let k = 0; k < SETTLE_STEPS; k++) {
      this._step(STEP);
      for (const ch of this.chains) for (const j of ch.joints) j.prevTail.copy(j.tail);
    }
    this.acc = 0;
    this.ready = true;
  }

  // the tail where the joint points at rest relative to its parent's current pose (world)
  _restTail(j, out) {
    const head = _v0.setFromMatrixPosition(j.bone.matrixWorld);
    _m0.compose(_zero, j.restQuat, _one).premultiply(j.bone.parent.matrixWorld);
    out.copy(j.axis).applyMatrix4(_m0.setPosition(head));
    return out;
  }

  update(dt) {
    if (!this.ready) this.reset();
    this.acc += Math.min(dt, STEP * MAX_STEPS);
    const n = Math.min(MAX_STEPS, Math.floor(this.acc / STEP + 1e-9));
    this.acc -= n * STEP;
    // no step due this frame (a fast display): the cloth holds, and the next step blends from the last pose
    // it was solved at
    if (n === 0) {
      for (const ch of this.chains) ch.joints[0].bone.updateMatrixWorld(true);
      return;
    }
    for (const a of this.anchors) {
      a.cur.copy(a.node.matrixWorld);
      a.prev.decompose(a.pp, a.pq, a.ps);
      a.cur.decompose(a.cp, a.cq, a.cs);
    }
    for (const c of this.colliders) { c.refresh(); c.ca.copy(c.wa); c.cb.copy(c.wb); }
    const R = this.root;
    if (R) {
      R.cur.copy(R.node.matrixWorld);
      R.prev.decompose(R.pp, R.pq, R.ps);
      R.cur.decompose(R.cp, R.cq, R.cs);
      R.last.copy(R.prev);
    }
    for (let i = 1; i <= n; i++) {
      // the body's pose part way from last frame to this one
      const f = i / n;
      for (const a of this.anchors) {
        a.node.matrixWorld.compose(
          _v0.lerpVectors(a.pp, a.cp, f), _q0.copy(a.pq).slerp(a.cq, f), _v1.lerpVectors(a.ps, a.cs, f));
      }
      for (const c of this.colliders) { c.wa.lerpVectors(c.pa, c.ca, f); c.wb.lerpVectors(c.pb, c.cb, f); }
      if (R) {
        // the root's move over this step, applied in part to every tail (and its last position, so it adds
        // no speed): the cloth is carried round a turn instead of being slung out by it
        R.step.compose(_v0.lerpVectors(R.pp, R.cp, f), _q0.copy(R.pq).slerp(R.cq, f), _v1.lerpVectors(R.ps, R.cs, f));
        const move = _m0.copy(R.last).invert().premultiply(R.step);
        R.last.copy(R.step);
        for (const ch of this.chains) {
          const k = ch.settings.follow;
          if (!k) continue;
          for (const j of ch.joints) {
            j.tail.lerp(_v2.copy(j.tail).applyMatrix4(move), k);
            j.prevTail.lerp(_v2.copy(j.prevTail).applyMatrix4(move), k);
          }
        }
      }
      this._step(STEP);
    }
    // back to this frame's pose exactly (the chains as solved hang from it)
    for (const a of this.anchors) { a.node.matrixWorld.copy(a.cur); a.prev.copy(a.cur); }
    if (R) R.prev.copy(R.cur);
    for (const c of this.colliders) { c.pa.copy(c.ca); c.pb.copy(c.cb); }
    for (const ch of this.chains) ch.joints[0].bone.updateMatrixWorld(true);
  }

  _step(h) {
    const rows = Math.max(...this.chains.map((c) => c.joints.length));
    // the chains' first bones hang from the anchors, which have moved: bring them along
    for (const ch of this.chains) ch.joints[0].bone.updateMatrixWorld(true);
    for (let r = 0; r < rows; r++) {
      // 1. integrate every chain's tail at this row
      for (const ch of this.chains) {
        const j = ch.joints[r];
        if (!j) continue;
        const s = ch.settings;
        j.prevHead.copy(j.head);
        j.head.setFromMatrixPosition(j.bone.matrixWorld);
        const sc = _v0.setFromMatrixScale(j.bone.matrixWorld).x;
        j.worldLength = j.length * sc;
        this._restTail(j, j.restDir).sub(j.head).normalize();
        // velocity through the air, and the swing: velocity relative to the head, which the body carries
        const v = _v1.subVectors(j.tail, j.prevTail);
        const swing = _v2.subVectors(v, _v3.subVectors(j.head, j.prevHead));
        const k = h * 60;
        v.multiplyScalar(Math.pow(1 - s.airDrag, k));
        v.addScaledVector(swing, -(1 - Math.pow(1 - s.damping, k)));
        j.next.copy(j.tail).add(v);
        j.next.addScaledVector(j.restDir, s.stiffness * sc * h * h);
        j.next.addScaledVector(_v1.fromArray(s.gravityDir), s.gravity * sc * h * h);
        j.prevTail.copy(j.tail);
      }
      // 2. keep neighbouring chains at their spacing
      for (const Lk of this.links) {
        const ja = Lk.a.joints[r], jb = Lk.b.joints[r];
        if (!ja || !jb) continue;
        const rest = Lk.rest[r] * _v0.setFromMatrixScale(ja.bone.matrixWorld).x;
        const d = _v1.subVectors(jb.next, ja.next);
        const len = d.length();
        if (len < 1e-6) continue;
        const slack = Lk.a.settings.linkSlack;
        const target = Math.min(rest * (1 + slack), Math.max(rest * (1 - slack), len));
        if (target === len) continue;
        const k = Lk.a.settings.linkStiffness * 0.5 * (len - target) / len;
        ja.next.addScaledVector(d, k);
        jb.next.addScaledVector(d, -k);
      }
      // 3. bend as a sheet: blend each chain's swing (its tail away from rest) toward its neighbours'
      this._smooth(r);
      // (where each tail was before the swing limit and the collisions moved it)
      for (const ch of this.chains) { const j = ch.joints[r]; if (j) j.pre.copy(j.next); }
      // 4. limit the swing, and collide the whole bone, not only its tail
      for (const ch of this.chains) {
        const j = ch.joints[r];
        if (!j) continue;
        const s = ch.settings;
        const p = j.next, head = j.head, L = j.worldLength;
        const dir = _v1.subVectors(p, head).normalize();
        if (dir.dot(j.restDir) < Math.cos(s.maxAngle)) {
          const axis = _v2.crossVectors(j.restDir, dir);
          if (axis.lengthSq() < 1e-9) axis.set(1, 0, 0); else axis.normalize();
          dir.copy(j.restDir).applyAxisAngle(axis, s.maxAngle);
        }
        p.copy(head).addScaledVector(dir, L);
        for (let it = 0; it < 3; it++) {
          let hit = false;
          // points along the bone: a hit part way down turns the bone about its head (by the lever)
          for (const t of BONE_SAMPLES) {
            const q = _q1.copy(head).lerp(p, t);
            const before = _q2.copy(q);
            for (const c of this.colliders) hit = c.push(q, s.hitRadius, s.contactSoftness) || hit;
            p.addScaledVector(q.sub(before), 1 / t);
          }
          p.sub(head).setLength(L).add(head);
          if (!hit) break;
        }
      }
      // 5. collide the cloth between neighbouring chains: the middle of the hem segment between two tails,
      // and the middle of the panel the two bones span, so an arm or a leg can't pass between the columns
      for (let it = 0; it < 2; it++) {
        let hit = false;
        for (const Lk of this.links) {
          const ja = Lk.a.joints[r], jb = Lk.b.joints[r];
          if (!ja || !jb) continue;
          const extra = Lk.a.settings.hitRadius;
          // (weights: how much of a push at that point the two tails take)
          for (const [wHead, wTail] of [[0, 1], [0.5, 0.5]]) {
            const q = _q1.copy(ja.head).add(jb.head).multiplyScalar(wHead / 2)
              .addScaledVector(ja.next, wTail / 2).addScaledVector(jb.next, wTail / 2);
            const before = _q2.copy(q);
            let moved = false;
            for (const c of this.colliders) moved = c.push(q, extra, Lk.a.settings.contactSoftness) || moved;
            if (!moved) continue;
            hit = true;
            const d = q.sub(before).multiplyScalar(1 / wTail);
            ja.next.add(d); jb.next.add(d);
          }
        }
        for (const ch of this.chains) {
          const j = ch.joints[r];
          if (j) j.next.sub(j.head).setLength(j.worldLength).add(j.head);
        }
        if (!hit) break;
      }
      // 6. turn each bone to point at its tail
      for (const ch of this.chains) {
        const j = ch.joints[r];
        if (!j) continue;
        // a push by the body or the swing limit is not kept as speed: the cloth comes to rest against the body
        // instead of bouncing off it every step, which reads as jitter
        j.prevTail.add(_v1.subVectors(j.next, j.pre));
        j.tail.copy(j.next);
        this._aim(j);
      }
    }
  }

  _smooth(r) {
    if (!this.links.length) return;
    // the chains in link order (a strip: a cape's columns side by side)
    const order = this._order || (this._order = [this.links[0].a, ...this.links.map((L) => L.b)]);
    const js = order.map((ch) => ch.joints[r]);
    if (js.some((j) => !j)) return;
    const off = js.map((j) => j.next.clone().sub(j.head).addScaledVector(j.restDir, -j.worldLength));
    for (let i = 0; i < js.length; i++) {
      const k = order[i].settings.sheetBend;
      if (!k) continue;
      const nb = [off[i - 1], off[i + 1]].filter(Boolean);
      const avg = nb.reduce((a, o) => a.add(o), new THREE.Vector3()).divideScalar(nb.length);
      const d = avg.sub(off[i]).multiplyScalar(k * (nb.length === 1 ? 0.5 : 1));
      js[i].next.add(d);
      // the same shift to where it was a step ago: the sheet is evened out without being set swinging
      js[i].prevTail.add(d);
    }
  }

  // turn the bone (from its rest rotation) so its axis points at its tail
  _aim(j) {
    const bone = j.bone;
    bone.quaternion.copy(j.restQuat);
    bone.updateMatrixWorld(false);
    const local = _v1.copy(j.tail).applyMatrix4(_m0.copy(bone.matrixWorld).invert()).normalize();
    _q0.setFromUnitVectors(_v2.copy(j.axis).normalize(), local);
    bone.quaternion.multiply(_q0);
    bone.updateMatrixWorld(true);
  }
}

// A cape's chains found by name in a loaded model: bones named `<prefix>_<column>_<row>` (row 0 at the collar).
// The first `pinned` rows are not simulated: they ride the parent (row 0 is the band over the shoulders), and
// the springs start below them. Returns the chains in column order and the links between neighbouring columns.
export function findChains(root, prefix = 'cape', settings = {}, pinned = 1) {
  const re = new RegExp('^' + prefix + '_(\\d+)_(\\d+)$');
  const cols = new Map();
  root.traverse((o) => {
    const m = re.exec(o.name);
    if (!m) return;
    const c = Number(m[1]), r = Number(m[2]);
    if (!cols.has(c)) cols.set(c, []);
    cols.get(c)[r] = o;
  });
  const keys = [...cols.keys()].sort((a, b) => a - b);
  const chains = keys.map((k) => new SpringChain(cols.get(k).filter(Boolean).slice(pinned), settings));
  const links = [];
  for (let i = 0; i + 1 < chains.length; i++) links.push([chains[i], chains[i + 1]]);
  return { chains, links };
}

// Change settings on every chain (the bench's sliders, or a tuned preset)
export function applySettings(springs, settings) {
  for (const ch of springs.chains) Object.assign(ch.settings, settings);
}
