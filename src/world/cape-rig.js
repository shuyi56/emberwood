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
import { PropertyBinding } from 'three';
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

// on her own model (src/cape-lab/model-body.js, and the game): her cape is modelled hugging her sleeves and
// skirt, and a wider soft band would hold it out from them, so it stood off her back in a stiff bell
export const WARDEN_MODEL_SETTINGS = { ...CAPE_SETTINGS, contactSoftness: 0.5 };

// [from joint, offset], [to joint, offset, beyond], radius: offsets are in the model's axes (+z is her front);
// `beyond` carries the far end on past its joint by that share of the capsule's length; options { turns: true }
// turn both offsets with their joints (see Collider)
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
  // hands, from the wrist to the fingertips along the hand itself (turning with the hand joint, so the capsule
  // follows the wrist as it bends back in the swing)
  [['Hand.L', [0, 0, 0]], ['Hand.L', [-0.5, -3.6, 0.3]], 1.7, { turns: true }],
  [['Hand.R', [0, 0, 0]], ['Hand.R', [0.5, -3.6, 0.3]], 1.7, { turns: true }],
];

export function makeColliders(model, defs = WARDEN_COLLIDERS) {
  // (a joint, not another node of the same name: a glTF export can have both. GLTFLoader also drops the dots
  // from names, so `Thigh.L` arrives as `ThighL`)
  const find = (name) => {
    // (and adds `_1` when a joint shares its name with another node: her `Hand.R` joint is `HandR_1`, beside the
    // `HandR` mesh)
    const want = [name, PropertyBinding.sanitizeNodeName(name)];
    want.push(want[1] + '_1');
    let o = null;
    model.traverse((n) => { if (!o && n.isBone && want.includes(n.name)) o = n; });
    if (!o) model.traverse((n) => { if (!o && want.includes(n.name)) o = n; });
    if (!o) throw new Error('cape colliders: no joint named ' + name);
    return o;
  };
  return defs.map(([a, b, r, o = {}]) => new Collider(model,
    { node: find(a[0]), offset: a[1], turns: o.turns }, { node: find(b[0]), offset: b[1], beyond: b[2] || 0, turns: o.turns }, r));
}

// model: the character's root (the object whose axes are hers: +y up, +z forward)
export function attachCapeSprings(model, settings = WARDEN_MODEL_SETTINGS, colliders = WARDEN_COLLIDERS) {
  model.updateMatrixWorld(true);
  const { chains, links } = findChains(model, 'cape', settings, 1);
  if (!chains.length) return null;
  const springs = new SpringBones({ chains, links, root: model, colliders: makeColliders(model, colliders), fit: 'surface' });
  springs.reset();
  return springs;
}
