// The cape bench's simulation, without any drawing: the stand-in, what she does over time, and the springs.
// Shared by the bench page (main.js) and the numbers check (tools/cape-lab/check.mjs).
import * as THREE from 'three';
import { makeDummy, pose, CYCLE } from './dummy.js';
import { SpringBones, findChains } from '../world/springbones.js';
import { WARDEN_COLLIDERS, makeColliders } from '../world/cape-rig.js';

// the bench advances in fixed frames of 1/60 s whatever the display does, so every run is the same
export const FRAME = 1 / 60;
// the walk: one cycle in 0.7 s (42 frames), close to the game's pace (2.2 world units a second is 66.7 model
// units, a cycle in 0.69 s); a whole number of frames a cycle lets the bench compare one cycle with the next
export const CYCLE_FRAMES = 42;
export const PACE = CYCLE / (CYCLE_FRAMES * FRAME);
const ACCEL = 170;          // units/s^2 when setting off or stopping (to full pace in about 0.4 s)
const TURN_RATE = 7;        // radians a second at most

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
    this.opts = opts;             // columns, rows, scenario, springs (on or off)
    this.build();
  }
  build() {
    this.dummy = makeDummy({ columns: this.opts.columns, rows: this.opts.rows });
    this.restart();
  }
  colliderDefs() {
    const k = this.settings.colliderScale ?? 1;
    return WARDEN_COLLIDERS.map(([a, b, r]) => [a, b, r * k]);
  }
  makeSprings() {
    const root = this.dummy.root;
    root.updateMatrixWorld(true);
    const { chains, links } = findChains(root, 'cape', this.settings, 1);
    this.springs = new SpringBones({ chains, links, colliders: makeColliders(root, this.colliderDefs()) });
    this.springs.reset();
  }
  restart() {
    this.t = 0; this.frame = 0; this.x = 0; this.z = 0; this.heading = 0; this.speed = 0; this.phase = 0;
    this.loop = []; this.drift = null;
    this.place();
    pose(this.dummy.bones, 0, 0, 0);
    for (const b of this.dummy.cape.bones) b.quaternion.identity();
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
    const dh = wrapAngle(target.heading - this.heading);
    // ease into and out of a turn: the rate follows the angle left, up to TURN_RATE
    this.heading += Math.sign(dh) * Math.min(Math.abs(dh), Math.min(TURN_RATE, 2 + Math.abs(dh) * 10) * dt);
    // heading 0 walks toward +z (her front)
    this.x += Math.sin(this.heading) * this.speed * dt;
    this.z += Math.cos(this.heading) * this.speed * dt;
    this.phase = (this.phase + this.speed * dt / CYCLE * Math.PI * 2) % (Math.PI * 2);
    this.t += dt;
    this.frame++;
    this.place();
    const w = this.speed / PACE;
    if (w < 0.02) this.phase = 0;      // standing: back to the stance
    pose(this.dummy.bones, this.phase, w, this.t);
    this.dummy.root.updateMatrixWorld(true);
    if (this.opts.springs) this.springs.update(dt);
    else for (const b of this.dummy.cape.bones) b.quaternion.identity();
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
