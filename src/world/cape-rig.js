// The Warden's cape on spring bones: the tuned settings and the body's colliders, by joint name.
//
// Row 0 of each column (`cape_<column>_0`) is the band over the shoulders: it rides `Chest` and only the rows
// under it swing.
//
// Used by the cape bench (cape-lab.html), on a stand-in body with the Warden's joints, and meant for the game:
// once blender/warden/warden.py gives the cape `cape_<column>_<row>` bones under `Chest`, the world attaches
// this to the loaded model and calls `update(dt)` after posing the walk:
//
//     import { attachCapeSprings } from './cape-rig.js';
//     const cape = attachCapeSprings(model);       // after GLTFLoader, model = res.scene
//     ...each frame, after mixer.setTime(...):
//     model.updateMatrixWorld(true); cape.update(dt);
//
// The joint names and the collider sizes are the Warden's (blender/warden/warden.py, in model units: she is
// 80 tall, her pelvis at 31, chest at 41, shoulders at 49, knees at 17). Her robe skirt reaches 6 units behind
// the legs and 8.5 to the sides, and the cape hangs from the shoulders (52.7) to just under the knees (19).
import { SpringBones, Collider, findChains, DEFAULTS } from './springbones.js';

export const CAPE_SETTINGS = {
  ...DEFAULTS,
  stiffness: 260,
  damping: 0.3,
  airDrag: 0.04,
  gravity: 450,
  hitRadius: 0.6,
  linkStiffness: 0.6,
  linkSlack: 0.12,
  sheetBend: 0.3,
  contactSoftness: 2,
  follow: 0.5,
  maxAngle: 1.25,
};

// [from joint, offset], [to joint, offset, beyond], radius: offsets are in the model's axes (+z is her front);
// `beyond` carries the far end on past its joint by that share of the capsule's length
export const WARDEN_COLLIDERS = [
  // back and shoulders: the robe's back is 4 behind the spine, the shoulder blades a little more
  [['Pelvis', [0, 2, -0.5]], ['Chest', [0, 6, -0.5]], 4.5],
  [['Chest', [-3.2, 7, -0.8]], ['Chest', [3.2, 7, -0.8]], 4.2],
  // hips under the skirt, wide at the sides
  [['Thigh.L', [2.2, -2, -1.0]], ['Thigh.R', [-2.2, -2, -1.0]], 4.8],
  // thighs and shins, with the skirt over the thighs
  [['Thigh.L', [0, -2, 0]], ['Shin.L', [0, 1, 0]], 4.4],
  [['Thigh.R', [0, -2, 0]], ['Shin.R', [0, 1, 0]], 4.4],
  [['Shin.L', [0, 0, 0]], ['Foot.L', [0, 1, 0]], 3.0],
  [['Shin.R', [0, 0, 0]], ['Foot.R', [0, 1, 0]], 3.0],
  // arms, which swing back into the cape
  [['UpperArm.L', [0.5, 0, 0]], ['Forearm.L', [0.3, 0, 0]], 2.6],
  [['UpperArm.R', [-0.5, 0, 0]], ['Forearm.R', [-0.3, 0, 0]], 2.6],
  // forearms and hands: on past the hand joint to the fingertips (her hand reaches 4 below it, 25.5)
  [['Forearm.L', [0, 0, 0]], ['Hand.L', [0, 0, 0], 0.42], 2.2],
  [['Forearm.R', [0, 0, 0]], ['Hand.R', [0, 0, 0], 0.42], 2.2],
];

export function makeColliders(model, defs = WARDEN_COLLIDERS) {
  const find = (name) => {
    const o = model.getObjectByName(name);
    if (!o) throw new Error('cape colliders: no joint named ' + name);
    return o;
  };
  return defs.map(([a, b, r]) => new Collider(model,
    { node: find(a[0]), offset: a[1] }, { node: find(b[0]), offset: b[1], beyond: b[2] || 0 }, r));
}

// model: the character's root (the object whose axes are hers: +y up, +z forward)
export function attachCapeSprings(model, settings = CAPE_SETTINGS, colliders = WARDEN_COLLIDERS) {
  model.updateMatrixWorld(true);
  const { chains, links } = findChains(model, 'cape', settings, 1);
  if (!chains.length) return null;
  const springs = new SpringBones({ chains, links, root: model, colliders: makeColliders(model, colliders) });
  springs.reset();
  return springs;
}
