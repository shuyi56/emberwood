// The party picked on the sprite bench: which characters walk in the world, in what order (the first
// leads), and the hairstyle each one wears. Kept in the browser's storage, which both pages share.
// Storage can be missing or blocked (private windows), so every read falls back to the default party.
import { FIGHTER, MAGE, RANGER, CLERIC, ROGUE } from './sprites.js';
import { HAIRSTYLES } from './presets/female-head.js';

const ROSTER = [
  { key: 'knight', name: 'Knight', note: 'Mail and leather', spec: FIGHTER },
  { key: 'mage', name: 'Mage', note: 'Robe and pointed hat', spec: MAGE },
  // the ranger is modelled in Blender (blender/ranger/): the world shows the model itself (public/models/ranger.glb,
  // from export_glb.py); her sheet, for the bench, is the 3D render, copied in rather than painted, so she has no
  // hairstyle choice (the painted spec is only the fallback if the render is missing)
  { key: 'ranger', name: 'Ranger', note: 'Jerkin and mantle, modelled in 3D', spec: { ...RANGER, hairStyle: null }, rendered: 'blender/ranger/ranger-blender.png', model: 'models/ranger.glb' },
  // the warden: her body and clothes from blender/warden/export_glb.py, her head, face and hair from the Face Creator
  // (tools/creator-head); she walks the world as the model. She has no sheet of her own yet: the bench falls back
  // to the ranger's painted card
  { key: 'warden', name: 'Warden', note: 'Tabard, breastplate and cape, modelled in 3D', spec: { ...RANGER, hairStyle: null }, model: 'models/warden.glb' },
  { key: 'cleric', name: 'Cleric', note: 'White robe and black scapular', spec: CLERIC },
  // the rogue is modelled in Blender too (blender/rogue/): the world shows public/models/rogue.glb, from her export_glb.py
  { key: 'rogue', name: 'Rogue', note: 'Corset, blouse and dagger, modelled in 3D', spec: ROGUE, model: 'models/rogue.glb' }
];
const DEFAULT_MEMBERS = ['knight', 'mage', 'ranger'];
const STORE = 'emberwood.party';

function loadParty() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORE)); } catch { /* fall back */ }
  const known = (k) => ROSTER.some((c) => c.key === k);
  const members = saved && Array.isArray(saved.members) ? saved.members.filter(known) : [];
  const hair = {};
  if (saved && saved.hair) Object.entries(saved.hair).forEach(([k, s]) => { if (known(k) && HAIRSTYLES.some((h) => h.key === s)) hair[k] = s; });
  return { members: members.length ? [...new Set(members)] : DEFAULT_MEMBERS.slice(), hair };
}
function saveParty(party) {
  try { localStorage.setItem(STORE, JSON.stringify(party)); } catch { /* not kept */ }
}
// a member's spec, wearing the hairstyle picked for them (characters without hair are unchanged)
function memberSpec(key, party) {
  const c = ROSTER.find((r) => r.key === key);
  return c.spec.hairStyle && party.hair[key] ? { ...c.spec, hairStyle: party.hair[key] } : c.spec;
}

export { ROSTER, loadParty, saveParty, memberSpec };
