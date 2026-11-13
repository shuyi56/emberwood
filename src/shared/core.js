// Shared basics for the sprite painter and its presets: sizes, the palette ramp, the figure's
// reference points, skin tones and a few surface patterns. Kept apart from sprites.js so the presets
// in ./presets/ can use them without importing sprites.js (which imports the presets).
const lerp = (a, b, t) => a + (b - a) * t;

// Frame size in design pixels, sheet layout (columns per facing row, facings) and walk length.
const FW = 64, FH = 80, COLS = 10, ROWS = 8, WALK = 8;

// Palettes: five tones per material. Shadows drift toward cool violet, lights toward warm straw,
// and both lose saturation, so each ramp covers a range of hues instead of one colour scaled.
function rgb2hsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l];
  const d = mx - mn, s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  let h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h / 6, s, l];
}
function hsl2rgb(h, s, l) {
  const f = (n) => { const k = (n + h * 12) % 12, a = s * Math.min(l, 1 - l); return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))); };
  return [f(0), f(8), f(4)];
}
function hueToward(h, target, t) { let d = target - h; d -= Math.round(d); return (h + d * t + 1) % 1; }
function ramp(hex, spread) {
  const [h, s0, l] = rgb2hsl((hex >> 16) & 255, (hex >> 8) & 255, hex & 255);
  const s = Math.min(s0, 0.46);
  spread = (spread || 0.1) * 0.86;
  return [-2, -1, 0, 1, 2].map((t) => {
    const hh = t < 0 ? hueToward(h, 0.62, -t * 0.05) : hueToward(h, 0.13, t * 0.045);
    const ss = Math.max(0, Math.min(1, s * (1 - Math.abs(t) * 0.08)));
    const ll = Math.max(0.06, Math.min(0.9, l + t * spread * (t < 0 ? 1.0 : 0.8)));
    return hsl2rgb(hh, ss, ll);
  });
}
// The painter works at RES times the design grid.
const RES = 2, PW = FW * RES, PH = FH * RES;

// Skin tones, and the figure's reference points: CX the centre line, GY the ground, HEAD the head's centre height.
// Skin is set by hand rather than with ramp(): from skin's orange hue, ramp()'s shadow shift toward blue
// runs through red, which turns modelled cheeks and jaws brick red. Here the shadows keep their warmth
// but lean toward orange-brown (hue about 15-19 degrees) rather than red, so the face can carry strong
// form without looking either flushed or grey.
const SKIN = [[160, 100, 80], [191, 134, 108], [207, 159, 132], [218, 180, 156], [229, 203, 184]];
// The skin tones a character can be given (spec.skinTone), each a hand-set ramp from shadow to highlight; the
// first is the default. Fair: paler and pinker, with less colour in the light. Its shadows still keep warmth,
// leaning rose-brown (hue about 12-16 degrees) so they model the form instead of greying it, and it steps less
// between the lit tones, as pale skin does, while the shadow end stays deep enough to carry the face's planes.
const SKINS = [
  { key: 'medium', name: 'Medium', ramp: SKIN },
  { key: 'fair', name: 'Fair', ramp: [[178, 118, 102], [212, 157, 137], [231, 186, 164], [240, 206, 187], [247, 226, 211]] }
];
// a skin tone's ramp by key (unknown or missing: the default)
const skinRamp = (key) => (SKINS.find((t) => t.key === key) || SKINS[0]).ramp;
const CX = 32, GY = 77, HEAD = 52;

// surface patterns: a fine random grain, and hair strands (soft with grain, or clean for crisp detail on top)
const grain = (x, y) => { const h = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return ((h - Math.floor(h)) - 0.5) * 0.14; };
const softStrands = (x, y) => { const k = (x * 5 + Math.floor(y / 4) * 2) % 11; return k === 0 ? -0.13 : k === 6 ? 0.06 : grain(x, y) * 0.4; };
// the same streaks without the grain noise, for surfaces that carry crisp drawn detail on top
const cleanStrands = (x, y) => { const k = (x * 5 + Math.floor(y / 4) * 2) % 11; return k === 0 ? -0.1 : k === 6 ? 0.05 : 0; };

export { lerp, FW, FH, COLS, ROWS, WALK, ramp, RES, PW, PH, SKIN, SKINS, skinRamp, CX, GY, HEAD, grain, softStrands, cleanStrands };
