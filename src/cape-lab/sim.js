// The cape bench's simulation, without any drawing: the stand-in, what she does over time, and the springs.
// Shared by the bench page (main.js) and the numbers check (tools/cape-lab/check.mjs).
import * as THREE from 'three';
import { makeDummy, CYCLE } from './dummy.js';
import { SpringBones, findChains } from '../world/springbones.js';
import { FabricBones } from '../world/fabric.js';
import { WARDEN_COLLIDERS, makeColliders } from '../world/cape-rig.js';
import { attachHairSprings } from '../world/hair-rig.js';
import { HeadMotion } from '../world/head-motion.js';

// the bench advances in fixed frames of 1/60 s whatever the display does, so every run is the same
export const FRAME = 1 / 60;
// the walk: one cycle in 0.7 s (42 frames), close to the game's pace (2.2 world units a second is 66.7 model
// units, a cycle in 0.69 s); a whole number of frames a cycle lets the bench compare one cycle with the next
export const CYCLE_FRAMES = 42;
export const PACE = CYCLE / (CYCLE_FRAMES * FRAME);
const ACCEL = 170;          // units/s^2 when setting off or stopping (to full pace in about 0.4 s)
const TURN_EASE = 12;       // the game's turn: the heading closes on the goal at this rate (1/s)

// what she does over time: target pace (a share of PACE) and heading
export const SCENARIOS = {
  walk: { label: 'Walk (loop check)', period: Infinity, at: () => ({ pace: 1, heading: 0 }) },
  startstop: {
    label: 'Start and stop', period: 6,
    at: (t) => ({ pace: t >= 1 && t < 3.5 ? 1 : 0, heading: 0 }),
  },
  turn: {
    label: 'Sharp turns', period: 7,
    at: (t) => ({ pace: t < 6.2 ? 1 : 0, heading: t < 2 ? 0 : t < 3.5 ? Math.PI / 2 : t < 5 ? Math.PI : 0 }),
  },
  run: { label: 'Run', period: 6, at: (t) => ({ pace: t >= 0.5 && t < 4 ? 1.8 : 0, heading: 0 }) },
  spin: {
    label: 'Turn on the spot', period: 4,
    at: (t) => ({ pace: 0, heading: t >= 1 && t < 3 ? Math.PI : 0 }),
  },
};

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export class CapeSim {
  constructor(settings, opts) {
    this.settings = settings;     // the springs' settings, plus colliderScale
    this.opts = opts;             // columns, rows, scenario, springs (on or off), and the body (see build)
    this.build();
  }
  // The body is the stand-in unless opts.body gives another: an object with root, bones (by joint name), cape,
  // pose(phase, walkWeight, t), setCape({ columns, rows }) and restPose() (src/cape-lab/model-body.js).
  build() {
    if (this.opts.body) {
      this.dummy = this.opts.body;
      this.dummy.root.position.set(0, 0, 0);
      this.dummy.root.rotation.set(0, 0, 0);
      this.dummy.restPose();
      this.dummy.setCape({ columns: this.opts.columns, rows: this.opts.rows });
    } else {
      this.dummy = makeDummy({ columns: this.opts.columns, rows: this.opts.rows });
      // a baked walk in place of the walk made in code (the check's --walk; the page sets it on the body)
      if (this.opts.walkData) this.dummy.setWalk(this.opts.walkData);
    }
    this.restart();
  }
  colliderDefs() {
    const k = this.settings.colliderScale ?? 1;
    return WARDEN_COLLIDERS.map(([a, b, r, o]) => [a, b, r * k, o]);
  }
  makeSprings() {
    const root = this.dummy.root;
    root.updateMatrixWorld(true);
    const { chains, links } = findChains(root, 'cape', this.settings, 1);
    // (opts.solver 'fabric': the cape as cloth, fabric.js, on the same bones and colliders)
    const Solver = this.opts.solver === 'fabric' ? FabricBones : SpringBones;
    this.springs = new Solver({ chains, links, root, colliders: makeColliders(root, this.colliderDefs()),
      fit: 'surface' });
    this.springs.reset();
    // her hair, when the body has it (the Warden's model): its own springs, with its own settings
    this.hair = this.dummy.hair ? attachHairSprings(root) : null;
    // her head on its spring over the neck (the Warden's model; the stand-in has no head)
    this.head = this.opts.headMotion === false ? null : HeadMotion.attach(root);
  }
  restart() {
    this.t = 0; this.frame = 0; this.x = 0; this.z = 0; this.heading = 0; this.speed = 0; this.phase = 0;
    this.loop = []; this.drift = null;
    this.place();
    this.dummy.pose(0, 0, 0);
    for (const b of this.dummy.cape.bones) b.quaternion.copy(b.userData.rest || b.quaternion);
    for (const b of this.dummy.hair ? this.dummy.hair.bones : []) b.quaternion.copy(b.userData.rest);
    this.makeSprings();
  }
  place() {
    this.dummy.root.position.set(this.x, 0, this.z);
    this.dummy.root.rotation.set(0, this.heading, 0);
  }
  scenarioTime() {
    const sc = SCENARIOS[this.opts.scenario];
    return sc.period === Infinity ? this.t : this.t % sc.period;
  }
  // advance by dt (the bench always uses FRAME; the check also tries other frame rates)
  step(dt = FRAME) {
    const target = SCENARIOS[this.opts.scenario].at(this.scenarioTime());
    const dv = target.pace * PACE - this.speed;
    this.speed += Math.sign(dv) * Math.min(Math.abs(dv), ACCEL * dt);
    // turn as the game turns a hero (src/world/main.js): an ease that closes the angle at 12 a second, so a
    // reversal starts at over 30 radians a second
    this.heading += wrapAngle(target.heading - this.heading) * Math.min(1, dt * TURN_EASE);
    // heading 0 walks toward +z (her front)
    this.x += Math.sin(this.heading) * this.speed * dt;
    this.z += Math.cos(this.heading) * this.speed * dt;
    this.phase = (this.phase + this.speed * dt / CYCLE * Math.PI * 2) % (Math.PI * 2);
    this.t += dt;
    this.frame++;
    this.place();
    const w = this.speed / PACE;
    if (w < 0.02) this.phase = 0;      // standing: back to the stance
    this.dummy.pose(this.phase, w, this.t);
    if (this.head) this.head.update(dt);
    this.dummy.root.updateMatrixWorld(true);
    if (this.opts.springs) { this.springs.update(dt); if (this.hair) this.hair.update(dt); }
    else for (const b of [...this.dummy.cape.bones, ...(this.dummy.hair ? this.dummy.hair.bones : [])]) b.quaternion.copy(b.userData.rest);
    // the loop check: on the straight walk, compare the cape one cycle apart (in her own space)
    if (this.opts.scenario === 'walk' && dt === FRAME && this.frame % CYCLE_FRAMES === 0) {
      const tails = this.tails();
      if (this.loop.length) {
        const prev = this.loop[this.loop.length - 1];
        this.drift = Math.max(...tails.map((p, i) => p.distanceTo(prev[i])));
      }
      this.loop.push(tails);
      if (this.loop.length > 4) this.loop.shift();
    }
  }
  // every joint's tail in her own space
  tails() {
    const inv = new THREE.Matrix4().copy(this.dummy.root.matrixWorld).invert();
    return this.springs.chains.flatMap((ch) => ch.joints.map((j) => j.tail.clone().applyMatrix4(inv)));
  }
}
