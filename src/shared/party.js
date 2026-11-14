// The party picked on the sprite bench: which characters walk in the world, in what order (the first
// leads), and the hairstyle each one wears. Kept in the browser's storage, which both pages share.
// Storage can be missing or blocked (private windows), so every read falls back to the default party.
import { FIGHTER, MAGE, RANGER, CLERIC, ROGUE } from './sprites.js';
import { HAIRSTYLES } from './presets/female-head.js';

const ROSTER = [
  { key: 'knight', name: 'Knight', note: 'Mail and leather', spec: FIGHTER },
  { key: 'mage', name: 'Mage', note: 'Robe and pointed hat', spec: MAGE },
  { key: 'ranger', name: 'Ranger', note: 'Jerkin and mantle', spec: RANGER },
  // the ranger modelled in Blender (blender/ranger/): her sheet is the 3D render, copied in rather than painted,
  // so she has no hairstyle choice (the spec is only the fallback painted if the render is missing)
  { key: 'ranger3d', name: 'Ranger (3D)', note: 'Modelled and rendered in Blender', spec: { ...RANGER, hairStyle: null }, rendered: 'blender/ranger/ranger-blender.png' },
  { key: 'cleric', name: 'Cleric', note: 'White robe and black scapular', spec: CLERIC },
  { key: 'rogue', name: 'Rogue', note: 'Corset, blouse and dagger', spec: ROGUE }
];
const DEFAULT_MEMBERS = ['ranger3d', 'knight', 'mage'];
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
