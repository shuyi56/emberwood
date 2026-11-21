// The Warden's head on a spring over her neck: the walk carries her head bolt upright (the baked walk's neck has
// no turn at all), which reads as a mannequin's. This layers the motion a head has on a walking body:
//  - a nod: the head lags the body's rise and fall at each step and its speeding up and slowing down, on a
//    damped spring, so it dips a little as each foot lands and rocks forward as she stops;
//  - a sway: it tips a little against her side-to-side shift;
//  - a turn: it follows part of her chest's twist, and looks a little ahead into a turn.
// All of it turns the head about the top of her neck (the Head node). The hair hangs from the head, so it moves with it.
//
//     import { HeadMotion } from './head-motion.js';
//     const headMotion = HeadMotion.attach(model);        // null without a Head and Neck
//     ...each frame, after posing and before the cape and hair: headMotion.update(dt);
import * as THREE from 'three';
import { PropertyBinding } from 'three';

export const HEAD_SETTINGS = {
  freq: 2.2,          // the spring's own frequency (Hz): low enough to lag a step, high enough not to wobble
  damping: 0.8,       // share of critical damping
  nod: 0.0002,        // radians of nod per unit/s^2 the neck speeds up or slows down (forward and down)
  bob: 0.0001,        // radians of nod per unit/s^2 the neck rises or falls
  sway: 0.00045,      // radians of tip per unit/s^2 of side-to-side acceleration
  twist: 0.12,        // share of the chest's twist the head follows
  lookAhead: 0.02,    // seconds of turning the head looks ahead into a turn
  most: 0.05,         // no more than this (radians) from the pose on any axis
};

const find = (root, name) => {
  const want = [name, PropertyBinding.sanitizeNodeName(name)];
  want.push(want[1] + '_1');
  let o = null;
  root.traverse((n) => { if (!o && n.isBone && want.includes(n.name)) o = n; });
  if (!o) root.traverse((n) => { if (!o && want.includes(n.name)) o = n; });
  return o;
};

const _m = new THREE.Matrix4(), _m2 = new THREE.Matrix4(), _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const _v = new THREE.Vector3(), _e = new THREE.Euler();

export class HeadMotion {
  static attach(root, settings = HEAD_SETTINGS) {
    const head = root.getObjectByName('Head'), neck = find(root, 'Neck'), chest = find(root, 'Chest');
    return head && neck && chest && head !== neck ? new HeadMotion(root, head, neck, chest, settings) : null;
  }
  constructor(root, head, neck, chest, settings) {
    Object.assign(this, { root, head, neck, chest, s: settings });
    root.updateMatrixWorld(true);
    // her chest's turn in her own space as she was made, to measure its twist from
    this.chestRest = new THREE.Quaternion();
    this.inRoot(chest).decompose(new THREE.Vector3(), this.chestRest, new THREE.Vector3());
    this.reset();
  }
  inRoot(o) { return _m.copy(this.root.matrixWorld).invert().multiply(o.matrixWorld); }
  reset() {
    this.angle = new THREE.Vector3();     // nod (x), turn (y), tip (z), in her own axes
    this.speed = new THREE.Vector3();
    this.vel = null; this.pos = null; this.acc = new THREE.Vector3(); this.yaw = null; this.yawRate = 0;
    this.base = null; this.set = null;
  }
  update(dt) {
    if (!(dt > 0)) return;
    const { head, s } = this;
    // (the head as the pose left it: a pose that doesn't move the head leaves last frame's turn on it, so take that
    // back off first)
    if (this.set && head.quaternion.equals(this.set.q) && head.position.equals(this.set.p)) {
      head.quaternion.copy(this.base.q); head.position.copy(this.base.p);
    }
    this.base = { q: head.quaternion.clone(), p: head.position.clone() };
    this.root.updateMatrixWorld(true);
    // the neck's movement in the world, read in her own axes (x her left, y up, z her front)
    const pos = _v.setFromMatrixPosition(this.neck.matrixWorld).clone();
    const toHer = _q.setFromRotationMatrix(this.root.matrixWorld).invert();
    if (this.pos) {
      const vel = pos.clone().sub(this.pos).divideScalar(dt);
      if (this.vel) {
        const acc = vel.clone().sub(this.vel).divideScalar(dt).applyQuaternion(toHer);
        // (smoothed over a few frames: a frame's step in a baked walk is not a clean acceleration)
        this.acc.lerp(acc, 1 - Math.exp(-dt * 30));
      }
      this.vel = vel;
    }
    this.pos = pos;
    // her heading's turn rate, to look into a turn
    _e.setFromQuaternion(_q2.setFromRotationMatrix(this.root.matrixWorld), 'YXZ');
    if (this.yaw !== null) {
      let d = _e.y - this.yaw;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      this.yawRate += (d / dt - this.yawRate) * (1 - Math.exp(-dt * 20));
    }
    this.yaw = _e.y;
    // the chest's twist (about her up axis) away from how she was made
    this.inRoot(this.chest).decompose(new THREE.Vector3(), _q2, new THREE.Vector3());
    _e.setFromQuaternion(_q2.multiply(_q.copy(this.chestRest).invert()), 'YXZ');
    const target = new THREE.Vector3(
      // forward (+z) acceleration tips the head back, slowing tips it forward; falling (-y) nods it forward
      -this.acc.z * s.nod - this.acc.y * s.bob,
      _e.y * s.twist + this.yawRate * s.lookAhead,
      // pushed toward her left (+x), the head tips to her right, about her front axis
      -this.acc.x * s.sway,
    );
    const w = 2 * Math.PI * s.freq, steps = Math.max(1, Math.ceil(dt / (1 / 240))), h = dt / steps;
    for (let i = 0; i < steps; i++) {
      for (const k of ['x', 'y', 'z']) {
        const a = w * w * (target[k] - this.angle[k]) - 2 * s.damping * w * this.speed[k];
        this.speed[k] += a * h;
        this.angle[k] += this.speed[k] * h;
      }
    }
    this.angle.clampScalar(-s.most, s.most);
    // turn the head about the top of her neck (the head's own origin) by that, in her own axes
    const turn = new THREE.Quaternion().setFromEuler(new THREE.Euler(this.angle.x, this.angle.y, this.angle.z, 'YXZ'));
    const her = _q2.setFromRotationMatrix(this.root.matrixWorld);
    const worldTurn = her.clone().multiply(turn).multiply(her.clone().invert());
    const pivot = _v.setFromMatrixPosition(head.matrixWorld);
    _m.makeTranslation(pivot.x, pivot.y, pivot.z).multiply(_m2.makeRotationFromQuaternion(worldTurn))
      .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z)).multiply(head.matrixWorld);
    _m.premultiply(_m2.copy(head.parent.matrixWorld).invert()).decompose(head.position, head.quaternion, new THREE.Vector3());
    head.updateMatrixWorld(true);
    this.set = { q: head.quaternion.clone(), p: head.position.clone() };
  }
}
