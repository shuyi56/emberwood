// The pre-painted sprite sheets: one PNG per character, and one per hairstyle for characters with hair.
// scripts/build-sprites.mjs paints them into public/sprites/ (with a manifest.json), and the bench and the
// world load those files instead of painting sheets at runtime.
import { ROSTER } from './party.js';
import { HAIRSTYLES } from './presets/female-head.js';
import { SKINS } from './core.js';

const SHEET_DIR = 'sprites';
// (the default skin tone, the first of SKINS, keeps the plain name; any other adds its key)
const sheetFile = (key, hair, skin) => (hair ? `${key}-${hair}` : key) + (skin && skin !== SKINS[0].key ? `-${skin}` : '') + '.png';

// every sheet to paint: { key, hair (or null), skin, file, spec }, one for each hairstyle and skin tone
function allSheets() {
  return ROSTER.flatMap((c) => (c.spec.hairStyle ? HAIRSTYLES.map((h) => h.key) : [null]).flatMap((hair) => SKINS.map(({ key: skin }) => ({
    key: c.key, hair, skin, file: sheetFile(c.key, hair, skin), spec: { ...c.spec, ...(hair ? { hairStyle: hair } : {}), skinTone: skin }
  }))));
}

export { SHEET_DIR, sheetFile, allSheets };
