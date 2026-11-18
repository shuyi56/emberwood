// The Warden's hair on spring bones: the settings and the colliders it rests against, by joint name.
//
// Chains hang from her `Head`: `hair_<column>_<row>` across her back (linked into one sheet, like
// the cape), and a lock in front of each shoulder, `hairL_<column>_<row>` and `hairR_<column>_<row>`; or, for a
// ponytail, one chain `hairT_0_<row>` from the tie. Row 0 of each column sits on the scalp (or the tie) and
// rides the head; the rows under it swing.
//
//     import { attachHairSprings } from './hair-rig.js';
//     const hair = attachHairSprings(model);
//     ...each frame, after posing (and after the cape): model.updateMatrixWorld(true); hair.update(dt);
//
// Sizes are in her model units (she is 80 tall, her head node at 54.2, her hair ends at about 44.6).
import { SpringBones, findChains, DEFAULTS } from './springbones.js';
import { makeColliders } from './cape-rig.js';

export const HAIR_GROUPS = ['hair', 'hairL', 'hairR', 'hairT'];
// the front locks lie on her chest, which rocks under them with every step: stiffer, and riding the head more,
// they sway a little on it instead of being levered off it into spikes
const FRONT_LOCK = { stiffness: 800, inherit: 0.95, maxAngle: 0.1 };
// a ponytail (`hairT_0_<row>`, one chain from the tie): heavier and freer than loose hair, it swings as one rope,
// carried less by the head so it lags each step and swings back across her nape
const PONYTAIL = { stiffness: 40, damping: 0.08, airDrag: 0.03, gravity: 450, inherit: 0.2, maxAngle: 1.3, hitRadius: 0.8,
  contactSoftness: 0.5, sheetBend: 0 };
export const HAIR_GROUP_SETTINGS = { hairL: FRONT_LOCK, hairR: FRONT_LOCK, hairT: PONYTAIL };

export const HAIR_SETTINGS = {
  ...DEFAULTS,
  stiffness: 400,
  damping: 0.3,
  airDrag: 0.05,
  gravity: 350,
  hitRadius: 0.3,
  contactSoftness: 1,
  linkStiffness: 0.4,
  linkSlack: 0.2,
  sheetBend: 0.3,
  follow: 0,
  inherit: 0.85,
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
  // the band of the cape over her shoulder blades, which rides the chest (4.5 behind it from 44 to 51; these reach up to 47). Her hair
  // is modelled a little inside it, so these are kept out of the fitting; they stop short of where the hair
  // hinges (52.7), which would lever the hair off the cape
  ...[-3, 0, 3].map((x) => [['Chest', [x, 1, 0]], ['Chest', [x, 6, 0]], 3.9, { fit: false }]),
];

// The cape's back, which her hair hangs over: a capsule down each of the three middle columns' first swinging
// bones (`cape_<column>_1` to `cape_<column>_2`), turning with them, so the hair rides the cape as it swings
// instead of sinking into it. Set 5 in toward her body and 3.7 round, so they hold the hair just off the cloth
// and close the gaps between the columns. Kept out of the fitting: her hair is modelled a little inside the
// cape. Only when the model has the cape's bones; update the cape before the hair each frame.
export function capeBackColliders(model) {
  const cols = [];
  model.traverse((o) => { const m = /^cape_(\d+)_1$/.exec(o.name); if (m) cols.push(+m[1]); });
  if (!cols.length) return [];
  const mid = Math.round((Math.min(...cols) + Math.max(...cols)) / 2);
  return [mid - 1, mid, mid + 1].filter((c) => cols.includes(c) && model.getObjectByName(`cape_${c}_2`))
    .map((c) => [[`cape_${c}_1`, [0, 0, 5]], [`cape_${c}_2`, [0, 0, 5]], 3.7, { turns: true, fit: false }]);
}

export function attachHairSprings(model, settings = HAIR_SETTINGS, colliders = [...HAIR_COLLIDERS, ...capeBackColliders(model)]) {
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
