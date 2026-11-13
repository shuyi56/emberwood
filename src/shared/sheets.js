// The pre-painted sprite sheets: one PNG per character, and one per hairstyle for characters with hair.
// scripts/build-sprites.mjs paints them into public/sprites/ (with a manifest.json), and the bench and the
// world load those files instead of painting sheets at runtime.
import { ROSTER } from './party.js';
import { HAIRSTYLES } from './presets/female-head.js';

const SHEET_DIR = 'sprites';
const sheetFile = (key, hair) => (hair ? `${key}-${hair}.png` : `${key}.png`);

// every sheet to paint: { key, hair (or null), file, spec }
function allSheets() {
  return ROSTER.flatMap((c) => (c.spec.hairStyle
    ? HAIRSTYLES.map((h) => ({ key: c.key, hair: h.key, file: sheetFile(c.key, h.key), spec: { ...c.spec, hairStyle: h.key } }))
    : [{ key: c.key, hair: null, file: sheetFile(c.key, null), spec: c.spec }]));
}

export { SHEET_DIR, sheetFile, allSheets };
