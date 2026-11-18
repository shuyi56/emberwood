// The Warden's hair on spring bones: the settings and the colliders it rests against, by joint name.
//
// Three groups of chains hang from her `Head`: `hair_<column>_<row>` across her back (linked into one sheet, like
// the cape), and a lock in front of each shoulder, `hairL_<column>_<row>` and `hairR_<column>_<row>`. Row 0 of
// each column sits on the scalp and rides the head; the rows under it swing.
//
//     import { attachHairSprings } from './hair-rig.js';
//     const hair = attachHairSprings(model);
//     ...each frame, after posing: model.updateMatrixWorld(true); hair.update(dt);
//
// Sizes are in her model units (she is 80 tall, her head node at 54.2, her hair ends at about 44.6).
import { SpringBones, findChains, DEFAULTS } from './springbones.js';
import { makeColliders } from './cape-rig.js';

export const HAIR_GROUPS = ['hair', 'hairL', 'hairR'];
// the front locks lie on her chest, which leans and bobs under them in a run: they swing less than the back
export const HAIR_GROUP_SETTINGS = { hairL: { maxAngle: 0.35 }, hairR: { maxAngle: 0.35 } };

export const HAIR_SETTINGS = {
  ...DEFAULTS,
  stiffness: 2000,
  damping: 0.5,
  airDrag: 0.05,
  gravity: 350,
  hitRadius: 0.3,
  contactSoftness: 1,
  linkStiffness: 0.4,
  linkSlack: 0.2,
  sheetBend: 0.3,
  follow: 0.4,
  maxAngle: 0.9,
};

// what the hair rests on: her head, neck, the cape over her shoulders and back, and her chest for the front
// locks. Generous on purpose: on reset they are fitted down to where her hair hangs as modelled.
export const HAIR_COLLIDERS = [
  [['Head', [0, 3, -0.3]], ['Head', [0, 3, -0.3]], 4.3],
  [['Neck_1', [0, -1, -0.3]], ['Neck_1', [0, 2, -0.3]], 2.4],
  [['Chest', [-5, 8, -1]], ['Chest', [5, 8, -1]], 5.5],
  [['Chest', [0, 2, -1.5]], ['Chest', [0, 8, -1.5]], 6.5],
  [['Chest', [-4, 6, 1]], ['Chest', [4, 6, 1]], 5],
];

export function attachHairSprings(model, settings = HAIR_SETTINGS, colliders = HAIR_COLLIDERS) {
  model.updateMatrixWorld(true);
  const chains = [], links = [];
  for (const g of HAIR_GROUPS) {
    const found = findChains(model, g, { ...settings, ...HAIR_GROUP_SETTINGS[g] }, 1);
    chains.push(...found.chains);
    links.push(...found.links);
  }
  if (!chains.length) return null;
  const springs = new SpringBones({ chains, links, root: model, colliders: makeColliders(model, colliders), fit: true });
  springs.reset();
  return springs;
}
