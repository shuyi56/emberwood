// The party picked on the sprite bench: which characters walk in the world, in what order (the first
// leads), and the hairstyle and skin tone each one has. Kept in the browser's storage, which both pages share.
// Storage can be missing or blocked (private windows), so every read falls back to the default party.
import { FIGHTER, MAGE, RANGER, CLERIC } from './sprites.js';
import { HAIRSTYLES } from './presets/female-head.js';
import { SKINS } from './core.js';

const ROSTER = [
  { key: 'knight', name: 'Knight', note: 'Mail and leather', spec: FIGHTER },
  { key: 'mage', name: 'Mage', note: 'Robe and pointed hat', spec: MAGE },
  { key: 'ranger', name: 'Ranger', note: 'Jerkin and mantle', spec: RANGER },
  { key: 'cleric', name: 'Cleric', note: 'White robe and black scapular', spec: CLERIC }
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
  const skin = {};
  if (saved && saved.skin) Object.entries(saved.skin).forEach(([k, t]) => { if (known(k) && SKINS.some((x) => x.key === t)) skin[k] = t; });
  return { members: members.length ? [...new Set(members)] : DEFAULT_MEMBERS.slice(), hair, skin };
}
function saveParty(party) {
  try { localStorage.setItem(STORE, JSON.stringify(party)); } catch { /* not kept */ }
}
// a member's spec, with the hairstyle (for characters with hair) and skin tone picked for them
function memberSpec(key, party) {
  const c = ROSTER.find((r) => r.key === key), spec = { ...c.spec };
  if (c.spec.hairStyle && party.hair[key]) spec.hairStyle = party.hair[key];
  if (party.skin && party.skin[key]) spec.skinTone = party.skin[key];
  return spec;
}

export { ROSTER, loadParty, saveParty, memberSpec };
