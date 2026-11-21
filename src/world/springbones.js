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
  // the air's drag as it really is, growing with the square of the speed: a = drag * v^2 (1 / model units). From the
  // cloth's weight per area and how much of it meets the air (tools/cape-lab/physics.mjs); 0 leaves only airDrag
  drag: 0,
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
  // 0..1: how much of the movement of the object a chain hangs from (its anchor: the chest for a cape, the head
  // for hair) carries the chain directly. Use it instead of `follow` (it includes the character's own movement)
  inherit: 0,
  // how far a bone may turn away from its rest direction (radians); keeps a cape from flipping up over the
  // shoulders in a hard turn
  maxAngle: 1.2,
  // per row, from the first swinging row: how far (share of the span) the cloth between two columns may bow out
  // off the straight line between them, following the columns either side (see linkPoint). For the rows where a
  // cape wraps closely round the shoulders and arms; none by default
  linkCurve: [],
};

const _v0 = new THREE.Vector3(), _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();
const _q0 = new THREE.Quaternion(), _qf = new THREE.Quaternion(), _qn = new THREE.Quaternion();
const _m0 = new THREE.Matrix4();
const _one = new THREE.Vector3(1, 1, 1), _zero = new THREE.Vector3();
// how deep a point between columns may rest inside a capsule (model units), and how far (model units, the two ends
// together, in the character's own space; taken along the capsule, see Collider.leaveAt) the capsule may move from where it was at rest before that leave is
// gone: the cloth lies as modelled against a limb that hangs where it was modelled, and a limb that swings away
// meets the cloth at its full size (a fixed leave lets a swinging hand through between two columns)
const REST_ALLOWANCE = 3, ALLOWANCE_FADE = 3;
const BONE_SAMPLES = [0.5, 1];      // where along a bone it is tested against the colliders
const _q1 = new THREE.Vector3(), _q2 = new THREE.Vector3();   // (Collider.push uses _v1.._v3)

// A collider is a capsule between two points, or a sphere when they are the same. Each point is a node's origin
// plus an offset given in the axes of `frame` (the model's root), so colliders can be placed between joints of
// any rig without knowing how its bones are rolled: `{ node: chest, offset: [0, 0, -2] }` is 2 units behind
// the chest joint, whichever way the chest bone points. An end can also reach on past its joint along the
// capsule (`beyond`: a share of the capsule's length), so a capsule from the wrist to the hand joint can cover
// the hand out to the fingertips and follow the forearm as it swings.
export class Collider {
  constructor(frame, a, b, radius) {
    this.frame = frame;
    // (an end with `turns` has its offset turn with its joint: given in the model's axes as the body stands when
    // the collider is made, then carried round by the joint's own turn from there, so a hand capsule follows the
    // wrist as it bends)
    const end = (e) => {
      const out = { node: e.node, offset: new THREE.Vector3(...(e.offset || [0, 0, 0])), turns: null };
      if (e.turns) {
        frame.updateMatrixWorld(true);
        out.turns = e.node.getWorldQuaternion(new THREE.Quaternion()).invert()
          .multiply(frame.getWorldQuaternion(new THREE.Quaternion()));
      }
      return out;
    };
    this.a = end(a);
    this.b = end(b || a);
    this.beyond = (b && b.beyond) || 0;
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
      const q = end.turns ? end.node.getWorldQuaternion(_qn).multiply(end.turns) : _qf;
      out.setFromMatrixPosition(end.node.matrixWorld)
        .add(_v1.copy(end.offset).applyQuaternion(q).multiplyScalar(this.scale));
    }
    if (this.beyond) this.wb.add(_v1.subVectors(this.wb, this.wa).multiplyScalar(this.beyond));
  }
  // how much of its rest leave holds where point p (world) is nearest it
  leaveAt(p) {
    const ab = _v1.subVectors(this.wb, this.wa), len2 = ab.lengthSq();
    const t = len2 > 1e-9 ? Math.min(1, Math.max(0, _v2.subVectors(p, this.wa).dot(ab) / len2)) : 0;
    return (this.leaveA ?? 1) * (1 - t) + (this.leaveB ?? 1) * t;
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
    const band = soft * this.scale, r = Math.max(0, (this.radius + extra) * this.scale - band / 2);
    const dist = d.length();
    if (dist >= r + band) return false;
    const x = r + band - dist;
    const to = x < band ? dist + x * x / (2 * band) : r + band / 2;
    if (dist < 1e-6) d.set(0, 0, -1); else d.divideScalar(dist);
    p.copy(c).addScaledVector(d, to);
    return true;
  }
}

// the points tested on the cloth between two neighbouring columns, as [head share, tail share, across]: along
// the hem segment between the two tails and across the middle of the panel the two bones span, each at a
// quarter, the middle and three quarters of the way from one column to the other, so an arm or a hand swinging
// back meets the cloth wherever it comes between two columns
export const LINK_SAMPLES = [];
for (const [wHead, wTail] of [[0, 1], [0.5, 0.5]]) for (const u of [0.25, 0.5, 0.75]) LINK_SAMPLES.push([wHead, wTail, u]);
// such a point, from the joints' heads and their tails (`next` while solving, else the tails). Where a cape wraps
// closely round her shoulders and arms, the cloth between two columns isn't the straight line between them but
// bows out round the body: on the rows the chains' `linkCurve` names, the point is taken on a curve through the
// columns either side as well (Catmull-Rom; a column with no neighbour carries the line straight on), bowing no
// further off the line than that share of the span. Taken on the straight line, the cape lay most of a unit
// inside its own cloth over the shoulders, and an arm swinging back shoved it out that much further than the
// sleeve reached, into a hump.
const CR = (u) => [(-u + 2 * u * u - u * u * u) / 2, (2 - 5 * u * u + 3 * u * u * u) / 2,
  (u + 4 * u * u - 3 * u * u * u) / 2, (-u * u + u * u * u) / 2];
const _lp = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
const curveOf = (L, r) => (L.a.settings.linkCurve && L.a.settings.linkCurve[r]) || 0;
export function linkPoint(L, r, [wHead, wTail, u], out, solving = false) {
  const at = (j, o) => o.copy(j.head).multiplyScalar(wHead).addScaledVector(solving ? j.next : j.tail, wTail);
  const [p0, p1, p2, p3] = _lp;
  at(L.a.joints[r], p1); at(L.b.joints[r], p2);
  const line = out.copy(p1).multiplyScalar(1 - u).addScaledVector(p2, u);
  const most = curveOf(L, r) * p1.distanceTo(p2);
  if (!most) return line;
  const j0 = L.before && L.before.joints[r], j3 = L.after && L.after.joints[r];
  if (j0) at(j0, p0); else p0.copy(p1).multiplyScalar(2).sub(p2);
  if (j3) at(j3, p3); else p3.copy(p2).multiplyScalar(2).sub(p1);
  const c = CR(u);
  const off = p0.multiplyScalar(c[0]).addScaledVector(p1, c[1]).addScaledVector(p2, c[2]).addScaledVector(p3, c[3]).sub(line);
  if (off.length() > most) off.setLength(most);
  return out.add(off);
}
// the shares of the two columns' tails in such a point (what a push on it is split by)
export function linkShares(L, r, [, wTail, u]) {
  if (!curveOf(L, r)) return [wTail * (1 - u), wTail * u];
  const c = CR(u);
  // (a missing neighbour is carried on from the two columns, so its share falls on them)
  let a = c[1], b = c[2];
  if (!(L.before && L.before.joints[r])) { a += 2 * c[0]; b -= c[0]; }
  if (!(L.after && L.after.joints[r])) { b += 2 * c[3]; a -= c[3]; }
  return [wTail * a, wTail * b];
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
  // fit: shrink any collider the cape's rest shape already sits inside (see fitColliders): true or 'band', or 'surface'
  constructor({ chains = [], colliders = [], links = [], root = null, fit = false } = {}) {
    this.chains = chains;
    this.fit = fit;
    this.root = root && {
      node: root, prev: new THREE.Matrix4(), cur: new THREE.Matrix4(), step: new THREE.Matrix4(), last: new THREE.Matrix4(),
      pp: new THREE.Vector3(), pq: new THREE.Quaternion(), ps: new THREE.Vector3(),
      cp: new THREE.Vector3(), cq: new THREE.Quaternion(), cs: new THREE.Vector3(),
    };
    this.colliders = colliders;
    // links: [chainA, chainB] pairs whose joints, row by row, keep their spacing
    this.links = links.map(([a, b]) => ({ a, b, rest: [] }));
    for (const L of this.links) {
      L.before = (this.links.find((K) => K.b === L.a) || {}).a || null;
      L.after = (this.links.find((K) => K.a === L.b) || {}).b || null;
    }
    // the objects the chains hang from: their pose is blended across a frame's steps
    this.anchors = [...new Set(chains.map((c) => c.joints[0].bone.parent))].map((node) => ({
      node, prev: new THREE.Matrix4(), cur: new THREE.Matrix4(), last: new THREE.Matrix4(),
      chains: chains.filter((c) => c.joints[0].bone.parent === node && c.settings.inherit),
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
    if (this.fit && !this.fitCuts) this.fitCuts = this.fitColliders(this.fit === true ? 'band' : this.fit);
    // let the cloth settle against the body as it stands, without gaining speed, so it doesn't jump on the
    // first frames
    for (let k = 0; k < SETTLE_STEPS; k++) {
      this._step(STEP);
      for (const ch of this.chains) for (const j of ch.joints) j.prevTail.copy(j.tail);
    }
    this.acc = 0;
    this.ready = true;
  }

  // Fit the colliders to the cloth as it hangs at rest. The cloth as modelled is the statement of where it sits on
  // this body; a capsule sized for another body (or a generous one) that reaches into it would push the whole
  // cape out into a stiff bell the moment the springs start. Run at rest pose:
  //  - each capsule is shrunk until the points along every bone are clear of it;
  //  - the points between columns (the middle of the hem segment, and of the panel) may still start inside
  //    one, where the cloth wraps closely round a limb and the straight line between two columns cuts through
  //    it: each such point may rest up to REST_ALLOWANCE deep, for that capsule only, while the capsule stays
  //    near where it was at rest (ALLOWANCE_FADE), instead of the capsule shrinking for every point.
  // Returns how much each radius came down (model units).
  fitColliders(mode = 'band', minShare = 0.35) {
    const reach = (c, s) => c.radius + s.hitRadius + (mode === 'band' ? s.contactSoftness / 2 : 0);
    const depth = (c, p) => {
      const ab = _v1.subVectors(c.wb, c.wa), len2 = ab.lengthSq();
      const t = len2 > 1e-9 ? Math.min(1, Math.max(0, _v2.subVectors(p, c.wa).dot(ab) / len2)) : 0;
      return p.distanceTo(_v2.copy(c.wa).addScaledVector(ab, t)) / c.scale;
    };
    const pts = [];
    for (const ch of this.chains) for (const j of ch.joints) {
      for (const t of BONE_SAMPLES) pts.push([j.head.clone().lerp(j.tail, t), ch.settings]);
    }
    // ('band': clear of the whole soft band, so the cloth rests exactly as modelled; 'surface': only out of the
    // surface the cloth is pushed to, so resting in the band is allowed, and the band holds it out a little)
    const cuts = this.colliders.map((c) => {
      if (c.fit === false) return 0;
      let over = 0;
      for (const [p, s] of pts) over = Math.max(over, reach(c, s) - depth(c, p));
      const cut = Math.min(over, c.radius * (1 - minShare));
      c.radius -= cut;
      return cut;
    });
    const toRoot = new THREE.Matrix4();
    if (this.root) toRoot.copy(this.root.node.matrixWorld).invert();
    for (const c of this.colliders) { c.restA = c.wa.clone().applyMatrix4(toRoot); c.restB = c.wb.clone().applyMatrix4(toRoot); }
    for (const L of this.links) {
      const s = L.a.settings;
      L.allow = L.a.joints.map((ja, i) => {
        const jb = L.b.joints[i];
        if (!jb) return null;
        return LINK_SAMPLES.map((w) => linkPoint(L, i, w, new THREE.Vector3()))
          .map((q) => this.colliders.map((c) => (c.fit === false ? 0 : Math.min(REST_ALLOWANCE, Math.max(0, reach(c, s) - depth(c, q))))));
      });
    }
    return cuts;
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
      a.last.copy(a.prev);
    }
    for (const c of this.colliders) { c.refresh(); c.ca.copy(c.wa); c.cb.copy(c.wb); }
    // how much of each capsule's rest leave still holds, at each end: none once that end has moved half of
    // ALLOWANCE_FADE from where it was. (Taken along the capsule between its ends, so an arm swinging back from
    // the shoulder keeps its leave at the shoulder, where it hasn't moved, and the cape there isn't shoved out)
    if (this.root) _m0.copy(this.root.node.matrixWorld).invert();
    for (const c of this.colliders) {
      if (!c.restA) { c.leaveA = c.leaveB = 1; continue; }
      c.leaveA = Math.max(0, 1 - 2 * _v0.copy(c.ca).applyMatrix4(_m0).distanceTo(c.restA) / c.scale / ALLOWANCE_FADE);
      c.leaveB = Math.max(0, 1 - 2 * _v0.copy(c.cb).applyMatrix4(_m0).distanceTo(c.restB) / c.scale / ALLOWANCE_FADE);
    }
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
        // the anchor's move over this step, carried in part by the chains hanging from it that ask for it
        // (`inherit`): hair rides most of the head's bob instead of being flung by every footfall
        if (!a.chains.length) continue;
        const move = _m0.copy(a.last).invert().premultiply(a.node.matrixWorld);
        a.last.copy(a.node.matrixWorld);
        for (const ch of a.chains) {
          const k = ch.settings.inherit;
          for (const j of ch.joints) {
            j.tail.lerp(_v2.copy(j.tail).applyMatrix4(move), k);
            j.prevTail.lerp(_v2.copy(j.prevTail).applyMatrix4(move), k);
          }
        }
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
        // (a = drag v^2 over the step, taken implicitly so a fast tail is slowed, never turned back)
        if (s.drag) v.multiplyScalar(1 / (1 + s.drag / sc * v.length()));
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
      // 5. collide the cloth between neighbouring chains (LINK_SAMPLES), so an arm or a leg can't pass between
      // the columns
      for (let it = 0; it < 2; it++) {
        let hit = false;
        for (const Lk of this.links) {
          const ja = Lk.a.joints[r], jb = Lk.b.joints[r];
          if (!ja || !jb) continue;
          const extra = Lk.a.settings.hitRadius;
          const allow = Lk.allow && Lk.allow[r];
          LINK_SAMPLES.forEach((w, k) => {
            const q = linkPoint(Lk, r, w, _q1, true);
            const before = _q2.copy(q);
            let moved = false;
            this.colliders.forEach((c, ci) => {
              moved = c.push(q, extra - (allow ? allow[k][ci] * c.leaveAt(q) : 0), Lk.a.settings.contactSoftness) || moved;
            });
            if (!moved) return;
            hit = true;
            // the push shared between the two tails so that the point moves by it: each takes it in proportion
            // to its share of the point (the heads are fixed)
            const [ka, kb] = linkShares(Lk, r, w), kk = ka * ka + kb * kb;
            const d = q.sub(before);
            ja.next.addScaledVector(d, ka / kk); jb.next.addScaledVector(d, kb / kk);
          });
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
    // each chain with its neighbours in the sheet (a cape's columns side by side; hair has several sheets, each
    // its own strip)
    const order = this._order || (this._order = [...new Set(this.links.flatMap((L) => [L.a, L.b]))].map((ch) => ({
      ch, nb: this.links.filter((L) => L.a === ch || L.b === ch).map((L) => (L.a === ch ? L.b : L.a)),
    })));
    const off = new Map();
    for (const { ch } of order) {
      const j = ch.joints[r];
      if (j) off.set(ch, j.next.clone().sub(j.head).addScaledVector(j.restDir, -j.worldLength));
    }
    for (const { ch, nb: around } of order) {
      const k = ch.settings.sheetBend;
      const j = ch.joints[r];
      if (!k || !j) continue;
      const nb = around.map((c) => off.get(c)).filter(Boolean);
      if (!nb.length) continue;
      const avg = nb.reduce((a, o) => a.add(o), new THREE.Vector3()).divideScalar(nb.length);
      const d = avg.sub(off.get(ch)).multiplyScalar(k * (nb.length === 1 ? 0.5 : 1));
      j.next.add(d);
      // the same shift to where it was a step ago: the sheet is evened out without being set swinging
      j.prevTail.add(d);
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
