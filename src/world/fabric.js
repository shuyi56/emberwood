// Fabric on bones: the cape and the hair as cloth (fabric-core.js) instead of spring bones (springbones.js).
//
// It is the same rig and the same object to the game: `cape_<column>_<row>` bones found by findChains, the same
// colliders, fitted to the cloth at rest the same way, the same fixed steps with the body's pose blended across
// a frame, the same `follow` and `inherit`, and `update(dt)` and `reset()`. Only the step differs. A spring bone
// swings about its parent and is pulled back to the direction it was modelled in, row by row from the top, and
// its neighbours are evened out afterwards. Here every bone's tail is a point of one sheet, and all of them are
// solved together: threads down the columns that keep their length, threads across with a little slack, the
// diagonals, and ties two points along that make it roll over an edge in a curve. Nothing pulls it back to its
// modelled shape unless `stiffness` asks for some: it hangs as gravity and the body leave it.
//
//     import { attachFabric } from './fabric.js';
//     const cape = attachFabric(model);            // in place of attachCapeSprings(model)
//     ...each frame, after posing: model.updateMatrixWorld(true); cape.update(dt);
import * as THREE from 'three';
import { SpringBones, findChains, DEFAULTS, LINK_SAMPLES, linkPoint, linkShares } from './springbones.js';
import { Fabric, FABRIC_DEFAULTS } from './fabric-core.js';

// The chains' settings: the spring bones' (gravity, airDrag, damping, hitRadius, contactSoftness, linkSlack,
// follow, inherit mean what they mean there), and the cloth's own
export const FABRIC_SETTINGS = {
  ...DEFAULTS,
  stiffness: 0,              // pull toward the modelled shape (model units / s^2): none, cloth has no shape
  damping: 0.1,
  airDrag: 0.03,
  linkSlack: 0.05,           // the threads across may change length by this share freely
  weft: 1e-5,                // compliance of the threads across beyond that slack
  shear: FABRIC_DEFAULTS.shear,
  bend: FABRIC_DEFAULTS.bend,
  friction: 0.2,             // how well the body holds the cloth that lies on it
  iterations: 2,
};

const _p = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3(), _sc = new THREE.Vector3();

export class FabricBones extends SpringBones {
  reset() {
    this.fabric = null;       // (built again from the rest pose on the first step)
    super.reset();
  }
  // one sheet per strip of linked chains (a cape is one strip; hair has several, and single locks), all in one
  // solver. A chain's points: the head of its first bone, pinned to what it hangs from, then each bone's tail
  _build() {
    const s = this.chains[0].settings;
    let n = 0;
    for (const ch of this.chains) { ch.first = n; n += ch.joints.length + 1; }
    const pos = new Float64Array(3 * n);
    for (const ch of this.chains) {
      ch.joints[0].head.toArray(pos, 3 * ch.first);
      ch.joints.forEach((j, r) => j.tail.toArray(pos, 3 * (ch.first + r + 1)));
    }
    const f = new Fabric(pos, { damping: 0, iterations: s.iterations, stretch: 0, shear: s.shear, bend: s.bend, friction: s.friction });
    this.owner = new Array(n);
    for (const ch of this.chains) {
      f.pin(ch.first);
      ch.joints.forEach((j, r) => { this.owner[ch.first + r + 1] = ch; });
    }
    // the strips: each chain with no chain linked before it starts one
    const after = new Map(this.links.map((L) => [L.a, L.b])), linked = new Set(this.links.map((L) => L.b));
    for (const ch of this.chains) {
      if (linked.has(ch)) continue;
      const strip = [];
      for (let c = ch; c && !strip.includes(c); c = after.get(c)) strip.push(c);
      const rows = Math.max(...strip.map((c) => c.joints.length)) + 1;
      f.sheet(strip.length, rows, (c, r) => (r <= strip[c].joints.length ? strip[c].first + r : -1),
        { hold: true, weft: s.weft, weftSlack: s.linkSlack });
    }
    this.fabric = f;
  }
  _step(h) {
    for (const ch of this.chains) ch.joints[0].bone.updateMatrixWorld(true);
    if (!this.fabric) {
      for (const ch of this.chains) { const j = ch.joints[0]; j.head.setFromMatrixPosition(j.bone.matrixWorld); j.prevHead.copy(j.head); }
      this._build();
    }
    const f = this.fabric, x = f.pos, pv = f.prev, k = h * 60;
    const s0 = this.chains[0].settings;
    const sc = _sc.setFromMatrixScale(this.chains[0].joints[0].bone.matrixWorld).x;
    f.settings.gravity = s0.gravityDir.map((g) => g * s0.gravity * sc);
    f.settings.friction = s0.friction; f.settings.iterations = s0.iterations;
    // 1. the points as the frame left them (`follow` and `inherit` have carried the tails), the pins where
    // the body has them now, and the speed each point keeps
    for (const ch of this.chains) {
      const s = ch.settings, top = ch.joints[0], a0 = 3 * ch.first;
      top.prevHead.copy(top.head);
      top.head.setFromMatrixPosition(top.bone.matrixWorld);
      top.head.toArray(x, a0); top.head.toArray(pv, a0);
      const carry = _b.subVectors(top.head, top.prevHead);
      const air = Math.pow(1 - s.airDrag, k), swingLoss = 1 - Math.pow(1 - s.damping, k);
      ch.joints.forEach((j, r) => {
        // (through the air, and the swing: its speed relative to what the chain hangs from)
        const v = _p.subVectors(j.tail, j.prevTail).multiplyScalar(air);
        if (s.drag) v.multiplyScalar(1 / (1 + s.drag / sc * v.length()));      // (the air's drag, as in springbones.js)
        v.addScaledVector(_d.subVectors(v, carry), -swingLoss);
        if (s.stiffness) {
          j.worldLength = j.length * sc;
          v.addScaledVector(this._restTail(j, j.restDir).sub(_d.setFromMatrixPosition(j.bone.matrixWorld)).normalize(), s.stiffness * sc * h * h);
        }
        const a = 3 * (ch.first + r + 1);
        j.tail.toArray(x, a);
        pv[a] = x[a] - v.x; pv[a + 1] = x[a + 1] - v.y; pv[a + 2] = x[a + 2] - v.z;
      });
    }
    // 2. the cloth, against the body's capsules
    const body = (p, i) => {
      const s = this.owner[i].settings;
      _p.fromArray(p);
      let hit = false;
      for (const c of this.colliders) hit = c.push(_p, s.hitRadius, s.contactSoftness) || hit;
      if (hit) _p.toArray(p);
      return hit;
    };
    f.step(h, body, { neutral: true });
    // 3. the cloth between two columns (an arm or a hand swinging back between them): the same points the
    // spring bones test, the push shared between the two tails, and not kept as speed
    for (const ch of this.chains) ch.joints.forEach((j, r) => j.next.fromArray(x, 3 * (ch.first + r + 1)));
    let hit = false;
    for (const Lk of this.links) {
      const s = Lk.a.settings;
      for (let r = 0; r < Lk.a.joints.length; r++) {
        const ja = Lk.a.joints[r], jb = Lk.b.joints[r];
        if (!jb) continue;
        if (r > 0) { ja.head.copy(Lk.a.joints[r - 1].next); jb.head.copy(Lk.b.joints[r - 1].next); }
        const allow = Lk.allow && Lk.allow[r];
        LINK_SAMPLES.forEach((w, q) => {
          const pt = linkPoint(Lk, r, w, _p, true), before = _b.copy(pt);
          let moved = false;
          this.colliders.forEach((c, ci) => {
            moved = c.push(pt, s.hitRadius - (allow ? allow[q][ci] * c.leaveAt(pt) : 0), s.contactSoftness) || moved;
          });
          if (!moved) return;
          hit = true;
          const [ka, kb] = linkShares(Lk, r, w), kk = ka * ka + kb * kb, d = pt.sub(before);
          ja.next.addScaledVector(d, ka / kk); jb.next.addScaledVector(d, kb / kk);
        });
      }
    }
    if (hit) {
      for (const ch of this.chains) ch.joints.forEach((j, r) => {
        const a = 3 * (ch.first + r + 1);
        f.shift(ch.first + r + 1, j.next.x - x[a], j.next.y - x[a + 1], j.next.z - x[a + 2]);
      });
      f.hold(null, false, true);
    }
    // 4. the bones, from the top down, each turned to point at its tail
    for (const ch of this.chains) {
      ch.joints.forEach((j, r) => {
        const a = 3 * (ch.first + r + 1);
        j.tail.fromArray(x, a);
        j.prevTail.fromArray(pv, a);
        j.head.setFromMatrixPosition(j.bone.matrixWorld);
        this._aim(j);
      });
    }
  }
}

// The cape (or any `<prefix>_<column>_<row>` bones) as fabric. colliders: made ones (cape-rig.js makeColliders)
export function attachFabric(model, { prefix = 'cape', settings = FABRIC_SETTINGS, colliders = [], fit = 'surface', pinned = 1 } = {}) {
  model.updateMatrixWorld(true);
  const { chains, links } = findChains(model, prefix, settings, pinned);
  if (!chains.length) return null;
  const cloth = new FabricBones({ chains, links, root: model, colliders, fit });
  cloth.reset();
  return cloth;
}

// The cape by her gait: spring bones while she walks or stands (walkOn 'springs', the game's), or fabric (walkOn
// 'fabric'), and spring bones with their running settings while she runs. Both solve the same bones and
// keep their state on them (each joint's tail and where it was a step ago), so the cape passes from one to the
// other without a jump. The game says which she is doing (`setGait('run')` while it plays her run, 'walk' otherwise):
// her arms pump and her heels kick up from the run's first step, and fabric let a hand through then. Left unsaid
// (setGait(null)), it goes by her speed: her root's, eased over GAIT_EASE seconds; she runs above runAbove (model
// units a second) and walks again below walkBelow, the gap between them keeping it from flicking back and forth.
// walk, run: the chains' settings in each (the spring bones' own or FABRIC_SETTINGS, and the running springs, cape-rig.js)
const GAIT_EASE = 0.15;
export class CapeGait extends FabricBones {
  constructor(o, { walk, run, runAbove = 100, walkBelow = 85, walkOn = 'springs' } = {}) {
    super(o);
    for (const ch of this.chains) { ch.walkSettings = walk ? { ...ch.settings, ...walk } : ch.settings; ch.runSettings = { ...ch.settings, ...(run || {}) }; ch.settings = ch.walkSettings; }
    Object.assign(this, { runAbove, walkBelow, walkOn, running: false, speed: 0, _at: null, gait: null });
  }
  setGait(g) { this.gait = g; }
  reset() { this.speed = 0; this._at = null; this._set(false); super.reset(); }
  _set(running) {
    this.running = running;
    for (const ch of this.chains) ch.settings = running ? ch.runSettings : ch.walkSettings;
  }
  update(dt) {
    if (this.root && dt > 0) {
      const at = _p.setFromMatrixPosition(this.root.node.matrixWorld), sc = _sc.setFromMatrixScale(this.root.node.matrixWorld).x || 1;
      if (this._at) {
        const v = Math.hypot(at.x - this._at.x, at.z - this._at.z) / dt / sc;
        this.speed += (v - this.speed) * Math.min(1, dt / GAIT_EASE);
      }
      this._at = (this._at || new THREE.Vector3()).copy(at);
    }
    if (this.gait) { if ((this.gait === 'run') !== this.running) this._set(this.gait === 'run'); }
    else if (!this.running && this.speed > this.runAbove) this._set(true);
    else if (this.running && this.speed < this.walkBelow) this._set(false);
    super.update(dt);
  }
  _step(h) {
    if (this.running || this.walkOn === 'springs') return SpringBones.prototype._step.call(this, h);
    return super._step(h);
  }
}
