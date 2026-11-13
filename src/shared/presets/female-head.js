// Female head preset: the face and hair first made for the Ranger, with a choice of hairstyles.
//
// FEMALE_HEAD is spread into a character spec ({ ...FEMALE_HEAD, ... }). `female: true` switches
// drawFigure to the female face: the rounder face and jaw, the eye placement, lashes and mouth.
// `brow` and `iris` are the eyebrow and eye colours, and `hairStyle` picks one of HAIRSTYLES.
//
// drawFemaleHead(add, P, A, hair, style) paints the hair. Call it from a character's gear(add, P, A),
// passing the hair ramp (five tones, made with ramp()) and a style key; with no key it uses the
// spec's hairStyle (A.S.hairStyle), then 'bob'. It adds its own depth-sorted parts: the far side of
// the hair behind the body, and the crown, fringe and near side over the face.
//
// Every style shares the same build: the hair is a shell of rings around the head, from the crown
// down to the ends, projected for each facing so it holds together from all eight sides; the face is
// cut out along the fringe, whose shape (one of BANGS) each style picks. A style sets how far the rings fall and how full they are, and can add
// a piece of its own (a ponytail, bun or braid) built the same way.
import { HEAD, CX, PW, PH, RES, SKIN, lerp, softStrands, cleanStrands } from '../core.js';

const FEMALE_HEAD = { female: true, brow: [96, 70, 56], iris: [88, 108, 70], hairStyle: 'bob' };

// Ring lists for each length. Each ring is a loop around the head: r is its half-width across the
// head, d its depth front to back, h its height (all may vary with the angle p around the head), and
// k < 1 pulls the ring round toward the back of the head, as hair falling behind the shoulders does.
// endH(p) is the height of the ends at angle p; bulge(p) makes the outline swell from clump to clump.
// the swell of a wave at t down the hair (0 at the top of the waves, 1 at the ends) and angle p round the
// head: it grows in from nothing below the ears and is strongest toward the ends
const WAVE = (t, p) => 0.9 * Math.sin(t * Math.PI * 3.6 + p * 1.6) * Math.min(1, t * 2.5);
// and the sideways swing of the waves, the same all round the head at each height, so the strands ripple
// together in S-curves
const SWING = (t) => 0.13 * Math.sin(t * Math.PI * 3.6) * Math.min(1, t * 2.5);
const LENGTHS = {
  // falls from the crown to just below the jaw, swelling softly over the cheeks, curling under at the ends
  bob: (endH, bulge) => [
    { r: () => 6.35, d: () => 6.7, h: () => HEAD + 3.6 },
    { r: () => 7.1, d: () => 7.45, h: () => HEAD + 1.4 },
    { r: (p) => 7.45 + bulge(p) * 0.6 + 0.6 * Math.max(0, Math.cos(p) + 0.3), d: (p) => 7.9 + bulge(p) * 0.6, h: (p) => HEAD - 1.8 + Math.sin(p * 3) * 0.3 },
    { r: (p) => 7.55 + bulge(p) * 0.55 + 0.55 * Math.max(0, Math.cos(p) + 0.3), d: (p) => 7 + bulge(p) * 0.5, h: () => HEAD - 4.8, k: 0.93 },
    { r: (p) => 7.35 + bulge(p) * 0.5 + 0.45 * Math.max(0, Math.cos(p) + 0.3), d: (p) => 6 + bulge(p) * 0.4, h: () => HEAD - 7.8, k: 0.86 },
    { r: (p) => 6.95 + bulge(p) * 0.5, d: (p) => 5.7 + bulge(p) * 0.4, h: (p) => endH(p) + 1.4, k: 0.74 },
    { r: (p) => 6.3 + bulge(p) * 0.45, d: () => 5.2, h: (p) => endH(p) + 0.5, k: 0.69 },
    { r: (p) => 5.6 + bulge(p) * 0.4, d: () => 4.6, h: (p) => endH(p), k: 0.65 }
  ],
  // past the shoulders: the rings widen over the shoulders and hang down the back
  long: (endH, bulge) => [
    { r: () => 6.35, d: () => 6.7, h: () => HEAD + 3.6 },
    { r: () => 7.1, d: () => 7.45, h: () => HEAD + 1.4 },
    { r: (p) => 7.45 + bulge(p) * 0.6 + 0.5 * Math.max(0, Math.cos(p) + 0.3), d: (p) => 7.9 + bulge(p) * 0.6, h: (p) => HEAD - 1.8 + Math.sin(p * 3) * 0.3 },
    { r: (p) => 7.6 + bulge(p) * 0.55, d: (p) => 7.3 + bulge(p) * 0.5, h: () => HEAD - 5, k: 0.95 },
    { r: (p) => 8.1 + bulge(p) * 0.5, d: (p) => 6.9 + bulge(p) * 0.4, h: () => HEAD - 9.5, k: 0.9 },
    { r: (p) => 8.7 + bulge(p) * 0.5, d: (p) => 6.8 + bulge(p) * 0.4, h: () => HEAD - 13, k: 0.84 },
    { r: (p) => 8.6 + bulge(p) * 0.6, d: (p) => 6.5 + bulge(p) * 0.4, h: (p) => endH(p) + 2.6, k: 0.8 },
    { r: (p) => 8 + bulge(p) * 0.5, d: () => 6, h: (p) => endH(p) + 0.8, k: 0.78 },
    { r: (p) => 7.3 + bulge(p) * 0.4, d: () => 5.4, h: (p) => endH(p), k: 0.76 }
  ],
  // past the shoulders in waves: like long, but with many rings so the outline can swell in and out
  // down the length. The waves (wave(t, p), t from the crown down to the ends) shift round the head, so
  // they twist like real waves rather than stacking in flat rows.
  wavy: (endH, bulge) => {
    const rings = [
      { r: () => 6.35, d: () => 6.7, h: () => HEAD + 3.6 },
      { r: () => 7.1, d: () => 7.45, h: () => HEAD + 1.4 },
      { r: (p) => 7.45 + bulge(p) * 0.6 + 0.5 * Math.max(0, Math.cos(p) + 0.3), d: (p) => 7.9 + bulge(p) * 0.6, h: (p) => HEAD - 1.8 + Math.sin(p * 3) * 0.3 }
    ];
    const N = 12;
    for (let i = 1; i <= N; i++) {
      const t = i / N, w = (p) => WAVE(t, p);
      rings.push({
        r: (p) => lerp(7.7, 8.9, Math.min(1, t * 1.5)) - t * t * 1.3 + bulge(p) * 0.5 + w(p),
        d: (p) => lerp(7.4, 6.2, t) + bulge(p) * 0.4 + w(p) * 0.6,
        h: (p) => lerp(HEAD - 3.6, endH(p), t),
        k: lerp(0.95, 0.76, t),
        tw: SWING(t), t
      });
    }
    return rings;
  },
  // close to the head, ending at the nape and above the ears: the base for short and tied-back styles
  short: (endH, bulge) => [
    { r: () => 6.35, d: () => 6.7, h: () => HEAD + 3.6 },
    { r: () => 7.0, d: () => 7.35, h: () => HEAD + 1.4 },
    { r: (p) => 7.15 + bulge(p) * 0.5, d: (p) => 7.6 + bulge(p) * 0.5, h: (p) => lerp(HEAD + 1.4, endH(p) + 0.9, 0.5) },
    // at the nape the hair tapers in to hug the neck, narrower than the jaw, so it does not show beside the
    // jaw from the front
    { r: (p) => 5.2 + bulge(p) * 0.4, d: (p) => 6.9 + bulge(p) * 0.4, h: (p) => endH(p) + 0.9, k: 0.92 },
    { r: (p) => 4.1 + bulge(p) * 0.3, d: () => 6, h: (p) => endH(p), k: 0.88 }
  ]
};

// Bang shapes. edge(t, s) is the height of the fringe's edge across the forehead, before the locks are
// cut into it: t runs from the middle of the face (0) out to the temples (1), s from the parting side (0)
// across to the far side (1). The locks: freq is how many there are per radian, tip how deep the notches
// between them are cut, jag how much their lengths vary. dir(phi) is the way the locks sweep as they
// fall (1: across from a side parting, the sign of phi: out from a centre parting, near 0: straight
// down); partA is where the parting sits across the crown (0 for a centre parting).
// every bang shape hangs this much lower than its edge, and falls this much further at the temples
// (a shape can set its own drop and drape)
const DROP = 0.8, DRAPE = 1.1, EYE_CLEAR = 0.7;
const BANGS = {
  // swept across from a side parting, longest at the far temple
  swept: { edge: (t, s) => HEAD + 3.3 - 1.5 * s - 0.6 * t * t, freq: 7.5, tip: 0.9, jag: 0.7, dir: () => 1, partA: -2.4 },
  // parted in the middle, lifting off the forehead and falling away to both temples
  curtain: { edge: (t) => HEAD + 4.3 - 2.3 * Math.pow(t, 0.8) + 0.8 * Math.max(0, 1 - t / 0.12), freq: 6, tip: 0.5, jag: 0.35, dir: (phi) => Math.sign(phi) || 1, partA: 0 },
  // short, choppy locks of uneven length, mussed slightly to one side
  choppy: { edge: (t, s) => HEAD + 3.2 - 0.6 * s - 0.4 * t * t, freq: 6.5, tip: 1.9, jag: 1.3, dir: () => 0.5, partA: -2.4 },
  // cut straight across the brow, rounding off at the temples
  blunt: { edge: (t) => HEAD + 2.7 - 0.7 * Math.pow(t, 4), freq: 11, tip: 0.3, jag: 0.15, dir: () => 0.15, partA: -2.4 },
  // a soft fringe parted at the middle that drapes: it lifts a little at the parting and falls away in a
  // curve to each temple, longer at the sides, the locks sweeping outward as they fall
  soft: { edge: (t) => HEAD + 3.7 - 1.7 * Math.pow(t, 1.3) + 0.6 * Math.max(0, 1 - t / 0.12), freq: 6, tip: 0.55, jag: 0.3, dir: (phi) => Math.sign(phi) * 0.9, partA: 0 },
  // one long lock swept low across from a deep side parting
  deep: { edge: (t, s) => HEAD + 4.7 - 2.8 * s - 0.3 * t * t, freq: 5.5, tip: 0.6, jag: 0.45, dir: () => 1.3, partA: -3.4 },
  // a few fine wisps with the forehead showing between them
  wispy: { edge: (t, s) => HEAD + 3.6 - 0.4 * s - 0.6 * t * t, freq: 4.5, tip: 2.6, jag: 0.6, dir: () => 0.6, partA: -2.4 }
};

// The styles. length picks a ring list; ends sets where the ends fall (base: their height at the back,
// front: how much higher they sit beside the face, side: how much higher round the sides of the head; tuck: how much
// higher from the face back to the ear, dropping sharply behind it, as tied-back hair is;
// shape: how the ends fall, see endH in drawFemaleHead); volume scales the clump-to-clump swell; cap trims the skull below that height (for hair that stops
// above the jaw); bangs picks one of BANGS; extra adds a piece of the style's own.
const HAIRSTYLES = [
  { key: 'bob', name: 'Bob', note: 'Jaw length, side-swept bangs', length: 'bob', ends: { base: 37.4, front: 4.2, shape: 'blunt' }, volume: 0.6, bangs: 'swept' },
  { key: 'long', name: 'Long', note: 'Past the shoulders, curtain bangs', length: 'long', ends: { base: 29.5, front: 5.5, shape: 'sections', sections: 5 }, volume: 0.7, bangs: 'curtain' },
  { key: 'wavy', name: 'Wavy', note: 'Long and wavy, curtain bangs', length: 'wavy', ends: { base: 30, front: 5, shape: 'sections', sections: 6 }, volume: 0.8, bangs: 'curtain', wavy: true },
  { key: 'princess', name: 'Princess', note: 'Waves dressed with braids', length: 'wavy', ends: { base: 27.5, front: 5, shape: 'sections', sections: 7 }, volume: 0.9, bangs: 'soft', wavy: true, extra: princess },
  { key: 'pixie', name: 'Pixie', note: 'Short crop, choppy bangs', length: 'short', ends: { base: HEAD - 6.2, front: 1.5, side: 1.5 }, volume: 1.9, cap: HEAD - 5.5, bangs: 'choppy' },
  { key: 'ponytail', name: 'Ponytail', note: 'Loose and low, a soft fringe', length: 'short', ends: { base: HEAD - 5.2, front: 1.5, tuck: 4.5 }, volume: 0.5, cap: HEAD - 0.2, bangs: 'soft', extra: ponytail },
  { key: 'bun', name: 'Bun', note: 'A knot, curtain bangs', length: 'short', ends: { base: HEAD - 5.2, front: 1.5, tuck: 4.5 }, volume: 0.35, cap: HEAD - 0.2, bangs: 'curtain', extra: bun },
  { key: 'braid', name: 'Braid', note: 'A plait, wispy bangs', length: 'short', ends: { base: HEAD - 5.6, front: 1.5, tuck: 4.5 }, volume: 0.35, cap: HEAD - 0.2, bangs: 'wispy', extra: braid }
];
const STYLE = Object.fromEntries(HAIRSTYLES.map((s) => [s.key, s]));

// Pieces drawn apart from the head shell (tails, buns): when the figure faces the camera they sit
// behind the head and body; seen from the side or back they are drawn over them.
const pieceDepth = (V, front) => (V.fz > 0.2 ? front : 20.4);
// a chain of tapered segments through 3D points [a, b, h] with a width at each, filled as one shape
function chain(P, V, pts, widths) {
  for (let i = 0; i < pts.length - 1; i++) {
    const p = V.pt(...pts[i]), q = V.pt(...pts[i + 1]);
    P.t(p[0], p[1], q[0], q[1], widths[i], widths[i + 1]);
  }
}
// a test for P.fill's on: only the fine pixels already painted in one of the hair's tones
function isHairOf(P, HAIR) {
  const keys = new Set(HAIR.map((c) => (Math.floor(c[0]) << 16) | (Math.floor(c[1]) << 8) | Math.floor(c[2])));
  return (k) => { const b = P.buf, q = k * 4; return b[q + 3] && keys.has((b[q] << 16) | (b[q + 1] << 8) | b[q + 2]); };
}
// a tie wrapped round the hair: a short band in the hair's darkest tone with a lit edge
function tie(P, V, at, w, HAIR) {
  const c = V.pt(...at);
  P.e(c[0], c[1], w / 2, 0.75).fill(HAIR, { flat: true, tone: 0 });
  P.dot(c[0] - w * 0.25, c[1] - 0.5, HAIR[2]);
}

function ponytail(add, P, A, HAIR) {
  const V = A.V, sway = A.walking ? -Math.sin(A.ph) * 1.1 : 0, lift = A.walking ? Math.abs(Math.cos(A.ph)) * 0.4 : 0;
  // A loose ponytail: tied low at the nape, the tail soft, waving a little from side to side as it falls
  // and loosening into a few tufts at the end. Seen from the side it hangs straight down from the tie
  // (gravity, not the curve of her back), tapering evenly, so it never bends into a crescent.
  const tieAt = [0, -6.9, HEAD - 4.4];
  const wave = (i) => Math.sin(i * 1.9) * 0.45;
  const pts = [tieAt];
  for (let i = 1; i <= 5; i++) pts.push([sway * i / 5 + wave(i) * 0.6, -7.3 - i * 0.04, HEAD - 4.4 - i * 2.9 + lift * (1 - i / 5)]);
  // the tail is flatter than it is round: SPREAD times wider across her back than front to back, so it
  // looks fuller from behind than from the side
  const SPREAD = 1.5, across = V.w(SPREAD, 1);
  const widths = [3.3, 3.8, 3.8, 3.6, 3.2, 2.6].map((w) => w * across);
  // the outline of the tail, as fine-pixel points, for its cast shadow
  const outline = () => {
    const seen = new Set();
    for (let i = 0; i < pts.length - 1; i++) {
      const p = V.pt(...pts[i]), q = V.pt(...pts[i + 1]);
      for (let t = 0; t <= 1.001; t += 0.1) {
        const x = lerp(p[0], q[0], t), y = lerp(p[1], q[1], t), w = lerp(widths[i], widths[i + 1], t) / 2;
        for (let d = -w; d <= w; d += 0.25) seen.add(Math.floor((y + 0.7) * 2) * 1000 + Math.floor((x + d + 0.6) * 2));
      }
    }
    return seen;
  };
  add(pieceDepth(V, -1), () => {
    // the tail lies on the head's hair and on her back, so it casts a soft shadow just below and beside
    // itself; this is what keeps it reading as a separate lock rather than one shape with the head
    if (V.fz <= 0.2) outline().forEach((k) => P.darken((k % 1000) / 2 + 0.25, Math.floor(k / 1000) / 2 + 0.25, 0.84));
    chain(P, V, pts, widths);
    // the end comes apart into thin threads of different lengths, splaying a little
    const end = pts[pts.length - 1], e = V.pt(...end);
    [[-1.1, 1.8], [-0.4, 2.9], [0.3, 2.3], [0.9, 3.3], [1.3, 1.5]].forEach(([dx, len]) => {
      const q = V.pt(end[0] + dx * 1.2 * across + sway * 0.3, end[1], end[2] - len);
      P.t(e[0] + dx * 0.9 * across, e[1] - 0.6, q[0], q[1], 0.9, 0.4);
    });
    // with a dark contour round its edge, so it stands off the hair behind it
    P.fill(HAIR, { pattern: cleanStrands, round: 1, dither: 0.3, dim: V.fz > 0.2, contour: true });
    // Threads: the tail is painted as many fine strands running its length, not one smooth shape. Each
    // strand sits at its own place across the tail (u, -1 at one edge to 1 at the other), alternates dark
    // groove and lit strand, drifts a little across as it falls, and some stop short so the tail thins
    // out into separate threads toward the end. The light comes from the upper left, so strands on that
    // side are lit brighter.
    const screen = pts.map((p) => V.pt(...p));
    const along = (t) => {
      const f = Math.min(pts.length - 1.001, t * (pts.length - 1)), i = Math.floor(f), k = f - i;
      const p = screen[i], q = screen[i + 1], dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1;
      return { x: lerp(p[0], q[0], k), y: lerp(p[1], q[1], k), nx: -dy / len, ny: dx / len, w: lerp(widths[i], widths[i + 1], k) / 2 };
    };
    // [place across, where it starts, where it ends, tone]
    const THREADS = [[-0.58, 0.1, 0.92, 3], [-0.18, 0.06, 1, 1], [0.22, 0.2, 0.85, 3], [0.56, 0.08, 0.97, 0]];
    THREADS.forEach(([u, t0, t1, tone], n) => {
      for (let t = t0; t <= t1; t += 0.012) {
        const c = along(t), drift = Math.sin(t * 5 + n * 1.7) * 0.12, uu = u + drift;
        P.dotOn(c.x + c.nx * uu * c.w, c.y + c.ny * uu * c.w, HAIR[tone === 3 && u > 0 ? 2 : tone]);
      }
    });
    // a couple of bright glints on the lit side, where the tail curves out of the tie
    [[-0.5, 0.2], [-0.35, 0.3], [-0.6, 0.45]].forEach(([u, t]) => { const c = along(t); P.dotOn(c.x + c.nx * u * c.w, c.y + c.ny * u * c.w, HAIR[4]); });
    // loose wisps: single strands that come away from the edge of the tail for a stretch, so its outline
    // is broken up rather than one clean curve
    [[-1.18, 0.32, 0.55, 2], [1.16, 0.48, 0.72, 1], [-1.12, 0.66, 0.86, 1], [1.2, 0.18, 0.34, 1]].forEach(([u, t0, t1, tone]) => {
      for (let t = t0; t <= t1; t += 0.012) {
        const c = along(t), bow = Math.sin((t - t0) / (t1 - t0) * Math.PI) * 0.18 * Math.sign(u);
        P.dot(c.x + c.nx * (u + bow) * c.w, c.y + c.ny * (u + bow) * c.w, HAIR[tone]);
      }
    });
  });
  // the tie goes over the tail, covering where it starts; where the tail is drawn behind the body (seen
  // from the front) the tie goes with it
  add(V.fz > 0.2 ? -1 : 20.45, () => {
    const t = V.pt(...tieAt), hair = isHairOf(P, HAIR);
    const onHair = (x, y, c) => { const X = Math.floor(x * 2), Y = Math.floor(y * 2); if (X >= 0 && Y >= 0 && X < PW && Y < PH && hair(Y * PW + X)) P.dot(x, y, c); };
    // the hair of the head is drawn back into the tie: grooves from round the back of the head
    // converging on it, every other one with a lit strand beside it
    if (V.fz <= 0.2) for (let j = -3; j <= 3; j++) {
      const a = j * 1.9, b = -Math.sqrt(Math.max(0, 52 - a * a)), h = HEAD + 3.2 - Math.abs(j) * 1.1;
      if (V.d(a, b) < 0.5) continue;
      const s0 = V.pt(a, b, h);
      for (let k = 0; k <= 12; k++) {
        const u = 0.2 + k / 12 * 0.7, x = lerp(s0[0], t[0] + j * 0.35, u), y = lerp(s0[1], t[1] - 0.6, u * u);
        onHair(x, y, HAIR[1]);
        if (j % 2 === 0 && u < 0.75) onHair(x - 0.5, y, HAIR[3]);
      }
    }
    // the tail just below the tie sits in the shadow of the tie
    P.e(t[0], t[1] + 1.1, 1.5 * across, 0.6).fill(HAIR, { flat: true, tone: 1, on: hair });
    tie(P, V, tieAt, 3.1 * across, HAIR);
  });
}

function bun(add, P, A, HAIR) {
  // An elaborate bun: three fat loops of hair wound into a knot on the crown, a fine plait wrapped round
  // its foot, the hair below swept up into it in a few lit strands.
  // The knot lies on a plane facing back and up, so it reads as a round knot from behind, a tall dome
  // from the side and a crest over the crown from the front.
  const V = A.V, C = [0, -4.4, HEAD + 6.7];
  const U = [1, 0, 0], Vv = [0, 0.5, 0.87], N = [0, -0.87, 0.5];
  const on3 = (r, th, dome) => [0, 1, 2].map((k) => C[k] + U[k] * r * Math.cos(th) * 1.08 + Vv[k] * r * Math.sin(th) + N[k] * dome);
  add(pieceDepth(V, 19.9), () => {
    const dim = V.fz > 0.2;
    // the hair gathered up off the crown into the knot: a soft swell joining the two, so the knot grows
    // out of the head instead of sitting on it like a ball
    const j = V.pt(0, -2.8, HEAD + 5.6), j2 = V.pt(0, -3.8, HEAD + 4.2);
    P.e(j[0], j[1], V.w(4.6, 4.4), 2.6).e(j2[0], j2[1], V.w(5.2, 4.6), 2.6).fill(HAIR, { pattern: cleanStrands, round: 0.9, dither: 0.3, dim });
    // the hair swept up from the nape and sides into the knot, seen from behind and the side
    if (V.fz < 0.4) {
      [-2.6, -1.3, 0, 1.3, 2.6].forEach((a, i) => {
        const p = V.pt(a * 1.25, -6.4 + Math.abs(a) * 0.35, HEAD - 2.2), q = V.pt(a * 0.45, -5.6, HEAD + 3.6);
        P.l(p[0], p[1], q[0], q[1], 0.5);
        if (i % 2 === 0) { const r = V.pt(a * 1.1, -6.3, HEAD - 0.6), m = V.pt(a * 0.6, -5.9, HEAD + 2.4); P.dotOn(r[0], r[1], HAIR[3]).dotOn(m[0], m[1], HAIR[3]); }
      });
      P.fill(HAIR, { flat: true, tone: 1, on: isHairOf(P, HAIR) });
    }
    // the base of the knot, a little wider than the coil, shading the hair it sits on
    const c = V.pt(...C);
    for (let a = 0.2; a < Math.PI - 0.2; a += 0.2) P.darken(c[0] + Math.cos(a) * V.w(5.1, 4.4), c[1] + Math.sin(a) * 4.4 + 0.6, 0.9);
    P.e(c[0], c[1], V.w(4.9, 4.2), 4.3).fill(HAIR, { pattern: cleanStrands, round: 0.8, dither: 0.3, dim, lift: -0.1 });
    // the knot: three fat loops of hair wound together, each standing off the others with its own dark
    // edge, lit along its upper curve and shadowed where it tucks under its neighbour
    const loops = [[Math.PI * 0.55, 1.9, 2.9, 2.2], [Math.PI * 1.2, 2.0, 2.6, 2.1], [Math.PI * 1.85, 2.0, 2.6, 2.1]];
    (V.fz < 0.2 ? [1, 2, 0] : [0, 1, 2]).forEach((n) => {
      const [th, r, rx0, ry] = loops[n], q = V.pt(...on3(r, th, 0.7)), rx = V.w(rx0, rx0 * 0.8);
      P.e(q[0], q[1], rx, ry).fill(HAIR, { pattern: cleanStrands, round: 1, dither: 0.3, dim, contour: true, lift: dim ? 0.04 : 0.2 });
      // the strands running round the loop: lit over its top, darker where it curls under
      // (one clean lit band over the top and a dark crease along the bottom, so each loop reads at this size)
      for (let a = Math.PI * 1.15; a <= Math.PI * 1.75; a += 0.12) P.dotOn(q[0] + Math.cos(a) * rx * 0.55, q[1] + Math.sin(a) * ry * 0.55, HAIR[dim ? 2 : 3]);
      if (!dim) for (let a = Math.PI * 1.3; a <= Math.PI * 1.6; a += 0.1) P.dotOn(q[0] + Math.cos(a) * rx * 0.62, q[1] + Math.sin(a) * ry * 0.62, HAIR[4]);
      for (let a = Math.PI * 0.25; a <= Math.PI * 0.75; a += 0.12) P.dotOn(q[0] + Math.cos(a) * rx * 0.75, q[1] + Math.sin(a) * ry * 0.75, HAIR[0]);
    });
    // a fine plait wrapped round the foot of the knot, binding it to the head
    const ring = [], rw = [];
    for (let k = 0; k <= 12; k++) { ring.push(on3(4.1, Math.PI * (0.95 + k / 12 * 1.1), 0.1)); rw.push(1.6); }
    plait(P, V, ring, rw, HAIR, dim);
    // a few loose hairs escaping the knot
    if (!dim) { const a = V.pt(...on3(3.1, 0.6, 0)), b = V.pt(...on3(3.7, 0.9, -0.3)); P.dotOn(a[0], a[1], HAIR[3]); P.dot(b[0], b[1], HAIR[2]); }
  });
}

// A plait along any path of 3D points (widths at each point): the strands cross over in lobes, each
// crossing a lit swell with a dark crease slanting across it, alternating sides. Unlike the hanging braid
// below, it works whichever way the plait runs on screen, so it can wrap round the head.
function plait(P, V, pts, widths, HAIR, dim) {
  if (pts.length < 2) return;
  chain(P, V, pts, widths);
  // (a touch lighter than the hair it lies on, so a braid over the waves still stands out)
  P.fill(HAIR, { pattern: cleanStrands, round: 0.9, dither: 0.3, dim, contour: true, lift: dim ? 0 : 0.12 });
  const sc = pts.map((q) => V.pt(...q));
  for (let i = 0; i < sc.length - 1; i++) {
    const p = sc[i], q = sc[i + 1], dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len, nx = -uy, ny = ux, side = i % 2 ? 1 : -1, w = (widths[i] + widths[i + 1]) / 4;
    const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
    // the crease between two lobes
    P.l(mx - nx * side * w * 0.8 - ux * 0.4, my - ny * side * w * 0.8 - uy * 0.4, mx + nx * side * w * 0.8 + ux * 0.4, my + ny * side * w * 0.8 + uy * 0.4, 0.5);
  }
  P.fill(HAIR, { flat: true, tone: dim ? 0 : 1, on: isHairOf(P, HAIR) });
  // the lit swell of each lobe, on the side toward the light
  for (let i = 0; i < sc.length - 1; i++) {
    const p = sc[i], q = sc[i + 1], dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len, nx = -uy, ny = ux, side = i % 2 ? 1 : -1, w = (widths[i] + widths[i + 1]) / 4;
    const mx = (p[0] + q[0]) / 2 - ux * 0.55, my = (p[1] + q[1]) / 2 - uy * 0.55;
    P.dotOn(mx - nx * side * w * 0.35, my - ny * side * w * 0.35, HAIR[dim ? 2 : 3]);
    if (!dim && i % 2) P.dotOn(mx - nx * side * w * 0.1 - ux * 0.5, my - ny * side * w * 0.1 - uy * 0.5, HAIR[4]);
  }
}

// Princess: long waves (the wavy style underneath) dressed with braids. A crown braid runs over the top
// of the head from ear to ear behind the fringe; two finer braids start behind the ears and meet at the
// back of the head, where they are tied, and one braid falls from the tie down over the waves.
function princess(add, P, A, HAIR) {
  const V = A.V, sway = A.walking ? -Math.sin(A.ph) * 0.7 : 0;
  // the crown braid: a band from one ear over the crown to the other (theta), set well back from the
  // fringe so the bangs fall free in front of it, lying on the skull inside the head's outline
  const crown = [];
  for (let k = 0; k <= 16; k++) {
    const th = -1.22 + k / 16 * 2.44, da = Math.sin(th) * 6.6, db = 0.6 + Math.cos(th) * 0.9, dh = 0.4 + Math.cos(th) * 7.2;
    // (sat a little deeper at the top of the head, where the outline is tightest)
    // set onto the skull, a little into the hair so its outer edge is flush (an ellipsoid of half-widths 6.75 across, 7.1 deep, 8 high)
    const k2 = Math.sqrt((da / 6.75) ** 2 + (db / 7.1) ** 2 + (dh / 8) ** 2), sc = (0.93 - Math.cos(th) * 0.05) / k2;
    crown.push({ at: [da * sc, db * sc, HEAD + 0.6 + dh * sc], n: [da / 6.75, db / 7.1, dh / 8], th });
  }
  // a point of it shows where the skull under it faces us (the upper part shows a little from every side)
  const shows = (c) => V.d(c.n[0], c.n[1]) + c.n[2] * 0.45 > 0.05;
  add(20.42, () => {
    let run = [];
    const flush = () => { if (run.length > 1) plait(P, V, run.map((c) => c.at), run.map((c) => 2.1 - Math.abs(c.th) * 0.4), HAIR, false); run = []; };
    crown.forEach((c) => { if (shows(c)) run.push(c); else flush(); });
    flush();
  });
  // the half-up braids and the braid that falls from where they meet, seen from the side and back
  if (V.fz > 0.25) return;
  const tieAt = [0, -7.9, HEAD - 2];
  add(20.43, () => {
    [-1, 1].forEach((sd) => {
      const pts = [[sd * 6.6, -2.4, HEAD + 0.8], [sd * 5.9, -4.6, HEAD + 0.2], [sd * 4.4, -6.5, HEAD - 0.6], [sd * 2.2, -7.6, HEAD - 1.4], [sd * 0.4, -7.9, HEAD - 1.9]];
      // only the side of the head facing us
      if (V.d(sd, -0.4) < -0.45) return;
      plait(P, V, pts, [1.5, 1.7, 1.8, 1.8, 1.6], HAIR, false);
    });
    // the braid falling from the tie over the waves
    const fall = [];
    for (let i = 0; i <= 9; i++) { const t = i / 9; fall.push([sway * t + Math.sin(t * 5) * 0.25, -7.9 + t * 0.9, HEAD - 2 - t * 14]); }
    plait(P, V, fall, fall.map((_, i) => lerp(2.4, 1.6, i / 9)), HAIR, false);
    // a tuft at its end, and the tie where the braids meet
    const e = V.pt(...fall[9]), e2 = V.pt(fall[9][0] + sway * 0.3, fall[9][1], fall[9][2] - 2.4);
    P.t(e[0], e[1], e2[0], e2[1], 2, 0.7).fill(HAIR, { pattern: cleanStrands, round: 0.8, contour: true });
    tie(P, V, fall[9], 2, HAIR);
    tie(P, V, tieAt, 3, HAIR);
  });
}

function braid(add, P, A, HAIR) {
  const V = A.V, sway = A.walking ? -Math.sin(A.ph) * 0.5 : 0, N = 7, dim = V.fz > 0.2;
  // the plait lies on her back from the nape down, a little fuller at the top
  const pts = [];
  for (let i = 0; i <= N; i++) { const t = i / N; pts.push([sway * t, -4.9 - 0.6 * Math.min(1, t * 3) - Math.sin(t * Math.PI) * 0.25, HEAD - 5.4 - t * 13]); }
  // (it starts tucked in against the nape, inside the hair's outline, and comes away from the neck as it falls)
  // (seen from the side it is a little narrower, as a plait is flatter than it is wide)
  const widths = pts.map((_, i) => lerp(5.2, 3.4, i / N) * (1 - 0.3 * V.fx * V.fx));
  // a point a fraction u of the way down the plait on screen, with its direction (ux, uy) and width
  const sc = pts.map((q) => V.pt(...q));
  const along = (u) => {
    const f = Math.min(N - 1e-6, Math.max(0, u * N)), i = Math.floor(f), t = f - i, p = sc[i], q = sc[i + 1];
    const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy) || 1;
    return { x: lerp(p[0], q[0], t), y: lerp(p[1], q[1], t), ux: dx / len, uy: dy / len, w: lerp(widths[i], widths[i + 1], t) };
  };
  add(pieceDepth(V, -1), () => {
    // The plait is a rope of three strands crossing over the middle in turn, so from behind it is a column
    // of plump lobes alternating left and right. Its body is a slim core with a rounded lobe swelling out on
    // alternate sides, all filled as one shape, so the outline scallops in and out down the plait and the
    // shading rounds across it like a rope rather than a flat band.
    const M = 8, LH = 13 / M;
    chain(P, V, pts, widths.map((w) => w * 0.62));
    for (let k = 0; k < M; k++) {
      const c = along((k + 0.5) / M), side = k % 2 ? 1 : -1, nx = -c.uy, ny = c.ux;
      // each lobe a fat lock slanting from the plait's outer edge down and in across the middle
      const x0 = c.x + nx * side * c.w * 0.26 - c.ux * LH * 0.45, y0 = c.y + ny * side * c.w * 0.26 - c.uy * LH * 0.45;
      const x1 = c.x - nx * side * c.w * 0.1 + c.ux * LH * 0.4, y1 = c.y - ny * side * c.w * 0.1 + c.uy * LH * 0.4;
      P.t(x0, y0, x1, y1, c.w * 0.5, c.w * 0.4);
    }
    P.fill(HAIR, { pattern: cleanStrands, round: 1.25, dither: 0.3, dim, contour: true, lift: dim ? 0 : 0.1 });
    // where the plait leaves the head it is still part of the hair: its first stretch is painted again
    // without the outline, so it runs out of the hair above with no seam
    if (V.fz > -0.6) { const a = sc[0], b = along(0.1); P.t(a[0], a[1], b.x, b.y, widths[0] * 0.55, b.w * 0.5).fill(HAIR, { pattern: cleanStrands, round: 1, dither: 0.3, dim, flat: true }); }
    const isHair = isHairOf(P, HAIR);
    for (let k = 0; k < M; k++) {
      // (from the side the first lobe is hidden in the hair it grows out of)
      if (k === 0 && V.fz > -0.6) continue;
      const c = along((k + 0.5) / M), side = k % 2 ? 1 : -1, nx = -c.uy, ny = c.ux, h = LH;
      // across the lobe from its outer edge (o = 1) to just past the middle (o = -0.15), dropping as it goes
      const at = (o, d) => [c.x + nx * side * c.w * 0.5 * o + c.ux * (d - o * h * 0.35), c.y + ny * side * c.w * 0.5 * o + c.uy * (d - o * h * 0.35)];
      // the lobe's rounded back catches the light along its upper edge, the body of it sits in the hair's
      // tone, and below it a dark crease where it tucks under the next lobe
      const L0 = at(0.85, -h * 0.2), L1 = at(-0.1, -h * 0.2), D0 = at(0.95, h * 0.42), D1 = at(-0.2, h * 0.42);
      P.l(L0[0], L0[1], L1[0], L1[1], 0.5).fill(HAIR, { flat: true, tone: dim ? 2 : 3, on: isHair });
      P.l(D0[0], D0[1], D1[0], D1[1], 0.5).fill(HAIR, { flat: true, tone: dim ? 0 : 1, on: isHair });
      // the brightest glint where the lit lobes (on the side toward the light) turn
      if (!dim && side < 0) { const g = at(0.45, -h * 0.2); P.dotOn(g[0], g[1], HAIR[4]); }
      // the far end of the crease, in the middle of the plait, is the deepest
      const m = at(-0.05, h * 0.42); P.dotOn(m[0], m[1], HAIR[dim ? 0 : 1]);
    }
    // a few loose hairs escaping the plait along its edges
    [[0.22, -1], [0.47, 1], [0.7, -1]].forEach(([u, sd]) => {
      const c = along(u), nx = -c.uy, ny = c.ux, ex = c.x + nx * sd * c.w * 0.5, ey = c.y + ny * sd * c.w * 0.5;
      P.dotOn(ex, ey, HAIR[3]).dot(ex + nx * sd * 0.5 + c.ux * 0.5, ey + ny * sd * 0.5 + c.uy * 0.5, HAIR[2]);
    });
    const end = pts[N];
    tie(P, V, end, 3.1, HAIR);
    // the tail below the tie: a short brush that flares out and splits into three soft points
    const e = V.pt(end[0], end[1], end[2] - 0.4), e2 = V.pt(end[0] + sway * 0.3, end[1], end[2] - 2);
    P.t(e[0], e[1], e2[0], e2[1], 2.6, 3.3);
    [-1.1, 0, 1.1].forEach((o, i) => P.t(e2[0] + o, e2[1], e2[0] + o * 1.3 + sway * 0.2, e2[1] + (i === 1 ? 1.6 : 1.2), 1.2, 0.5));
    P.fill(HAIR, { pattern: cleanStrands, round: 0.8, dim, contour: true });
    // strands spreading down the brush, and the light across it just below the tie
    [-0.6, 0.6].forEach((o) => P.l(e[0] + o * 0.4, e[1] + 0.6, e2[0] + o * 1.1, e2[1] + 0.6, 0.45));
    P.fill(HAIR, { flat: true, tone: dim ? 0 : 1, on: isHairOf(P, HAIR) });
    P.dotOn(e[0] - 0.4, e[1] + 0.7, HAIR[dim ? 2 : 3]).dotOn(e[0] + 0.1, e[1] + 0.7, HAIR[dim ? 2 : 3]);
  });
  // from behind, the hair on the back of the head is drawn together toward the top of the plait: a few
  // strands sweep down and in to the nape, so the plait grows out of the hair instead of sitting on it
  if (V.fz < 0.2) add(20.41, () => {
    const top = sc[0];
    [[-4.6, 1.5], [-2.6, 3], [-0.9, 3.8], [1.1, 3.6], [2.9, 2.6], [4.7, 1.2]].forEach(([a, h], i) => {
      const s0 = V.pt(a, -6.9, HEAD + h);
      for (let t = 0.08; t <= 0.95; t += 0.07) {
        // a gentle curve: straight down at first, then drawn in toward the plait
        const ct = t * t, x = lerp(s0[0], top[0] + Math.sign(a) * 0.6, ct), y = lerp(s0[1], top[1] - 0.6, t);
        P.dotOn(x, y, HAIR[t > 0.75 ? 1 : i % 2 ? 2 : (a < 0 ? 3 : 2)]);
      }
    });
  });
}

function drawFemaleHead(add, P, A, HAIR, styleKey) {
  const V = A.V, st = STYLE[styleKey || (A.S && A.S.hairStyle)] || STYLE.bob;
  // The head is covered by one shell with the face cut out along a hairline that turns with the head
  // (each half of the face cut separately, so it holds together in profile).
  // the face opening is narrow from the front (hair frames the cheeks) and opens up as she turns to profile
  const FACE = 0.98 + 0.32 * V.fx * V.fx;
  // the fringe: the style's bang shape, with locks of different lengths cut into its edge
  const bg = BANGS[st.bangs] || BANGS.swept, F = bg.freq, nLocks = Math.round(8 * F / 7.5);
  const hairline = (phi) => {
    const t = Math.min(1, Math.abs(phi) / FACE), s = (phi + FACE) / (2 * FACE);
    const tipN = phi * F + 0.3, tip = Math.abs(tipN - Math.round(tipN)) * 2;   // 0 at a lock's tip, 1 between
    const len = 0.5 + 0.5 * Math.sin(Math.round(tipN) * 2.3);                   // locks of different lengths
    // the bangs hang: the whole edge sits lower (drop), the locks fall longer toward the temples (drape),
    // and each lock is rounded at its tip with the notch cut sharply up between locks
    const drop = bg.drop == null ? DROP : bg.drop, drape = bg.drape == null ? DRAPE : bg.drape;
    const h = bg.edge(t, s) - drop - drape * Math.pow(t, 1.6) + (Math.pow(tip, 1.6) * bg.tip - len * bg.jag) * (1 - t * 0.7);
    // over the eyes the tips may brush the brows but never reach the eyes: below EYE_CLEAR they give way softly
    const clear = HEAD + EYE_CLEAR;
    return t < 0.8 && h < clear ? clear + (h - clear) * 0.25 : h;
  };
  // ends: clumps of different lengths, each coming to a soft point, sitting higher beside the face
  // How the ends fall (ends.shape):
  //   blunt     one smooth cut edge that curls under, rising a little toward the face
  //   sections  the hair parts into a few broad sections (ends.sections round the head), each with a
  //             rounded end, longest at the middle of the back so the ends form a soft V
  //   (default) many small clumps of different lengths, each coming to a soft point
  const clumpN = 13, shape = st.ends.shape, nSec = st.ends.sections || 5, BEHIND = 2.4;
  // which section an angle is in, and where across it (0 at its middle, 1 at its edges)
  const section = (phi) => { const u = (phi - Math.PI) / (Math.PI * 2) * nSec; return { i: Math.round(u), f: Math.abs(u - Math.round(u)) * 2 }; };
  const endH = (phi) => {
    // tuck: tied-back hair is pulled up off the face and over the ear, then drops sharply just behind the
    // ear to cover the back of the head down to the nape
    const tuck = (st.ends.tuck || 0) * Math.min(1, Math.max(0, (Math.cos(phi) + 0.3) / 0.35));
    const lift = st.ends.front * Math.pow(Math.max(0, Math.cos(phi) + 0.25), 1.4) + (st.ends.side || 0) * Math.max(0, Math.cos(phi) + 0.6) / 1.6 + tuck;
    if (shape === 'blunt') return st.ends.base - Math.sin(phi * 3 + 0.5) * 0.15 - (1 - Math.cos(phi)) * 0.9 + lift;
    if (shape === 'sections') {
      const { i, f } = section(phi);
      return st.ends.base - Math.sin(i * 2.1) * 0.3 - (1 - f * f) * 1.5 - (1 - Math.cos(phi)) * 1.4 + lift;
    }
    const u = phi / (Math.PI * 2) * clumpN, f = Math.abs(u - Math.round(u)) * 2;
    const len = Math.sin(Math.round(u) * 3.1) * 0.9 + Math.sin(Math.round(u) * 1.3) * 0.5;
    return st.ends.base - len * 0.6 - (1 - f) * 0.45 - (1 - Math.cos(phi)) * 0.9 + lift;
  };
  // the outline swells and dips from clump to clump instead of being a smooth shell
  const bulge = (phi) => (Math.sin(phi * clumpN) * 0.28 + Math.sin(phi * 5 + 1) * 0.22) * st.volume;
  const rings = LENGTHS[st.length](endH, bulge), lastBands = rings.length - 3;
  // the hair drawn in two passes: the side facing us over the head, the far side behind everything,
  // so it never covers the face
  // a ring with k < 1 pulls its angles toward the back of the head
  // (and a wavy ring swings its angles a little to the side, tw)
  const g = (R, p) => (R.k ? Math.PI + (p - Math.PI) * R.k : p) + (R.tw || 0);
  const bob = (() => {
    const N = 78, quads = [];
    for (let k = 0; k < N; k++) {
      const p0 = FACE + k / N * (Math.PI * 2 - 2 * FACE), p1 = FACE + (k + 1) / N * (Math.PI * 2 - 2 * FACE);
      const clump = Math.round((p0 + p1) / 2 / (Math.PI * 2) * clumpN);
      for (let r = 0; r < rings.length - 1; r++) {
        const A0 = rings[r], A1 = rings[r + 1];
        const at = (R, p) => V.seam(R.r(p), R.d(p), g(R, p), R.h(p));
        const q = [at(A0, p0), at(A0, p1), at(A1, p1), at(A1, p0)];
        const mid = V.seam(7.6, 7.8, (p0 + p1) / 2, HEAD);
        let facing = mid[2] > -0.4;
        // Below the jaw the hair curves back round the neck and shoulders. Whether it lies in front of the
        // shoulders or behind them is decided on the figure itself (how far forward it sits, b), the same
        // for every facing, so the hair falls the same way from the front as from three-quarters. Seen
        // from the front half, hair behind the shoulders is drawn behind the body.
        if (shape && facing && V.fz > 0.2 && A1.h(p0) < HEAD - 7) {
          const fwd = (R, p) => R.d(p) * Math.cos(g(R, p));
          facing = (fwd(A0, p0) + fwd(A0, p1) + fwd(A1, p1) + fwd(A1, p0)) / 4 > BEHIND;
        }
        quads.push({ q, facing, band: r, clump });
      }
    }
    return quads;
  })();
  // is the fine pixel at index k painted in one of the hair's tones?
  const isHair = isHairOf(P, HAIR);
  const drawBob = (front) => {
    const mine = bob.filter((b) => b.facing === front);
    if (!mine.length) return;
    if (!front) { mine.forEach((b) => P.p(b.q.map((p) => [p[0], p[1]]))); P.fill(HAIR, { pattern: cleanStrands, round: 1, dim: true, dither: 0.3 }); return; }
    // (the shape itself was added to the head's hair mask and filled with it; here come the details on top)
    // the curled-under ends sit in shadow
    mine.filter((b) => b.band >= lastBands).forEach((b) => P.p(b.q.map((p) => [p[0], p[1]])));
    P.fill(HAIR, { pattern: softStrands, round: 0.6, lift: -0.2 });
    // wavy hair: strands that follow the shell ring by ring, so they ripple with the waves, and a lit
    // fleck on each crest where the wave swells out toward the light
    if (st.wavy) {
      const crest = [];
      for (let k = 0; k <= 30; k++) {
        const phi = FACE + 0.06 + k / 30 * (Math.PI * 2 - 2 * FACE - 0.12) + Math.sin(k * 2.7) * 0.03;
        const pts = rings.slice(1 + (k % 3)).map((R, i, all) => {
          const last = i === all.length - 1, inset = last ? 0.9 : 0.25;
          return V.seam(R.r(phi) - inset, R.d(phi) - inset, g(R, phi), R.h(phi) + (last ? 1.6 + (k % 2) : 0));
        });
        if (V.seam(7.6, 7.8, phi, HEAD - 3.4)[2] < 3.2) continue;
        for (let i = 0; i < pts.length - 1; i++) P.l(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 0.5);
        // where a wave swings out, the strand beside this one catches the light; where it swings back in,
        // it falls into shadow
        rings.forEach((R) => {
          if (R.t == null || R.t < 0.2 || R.t > 0.95) return;
          const sw = Math.sin(R.t * Math.PI * 3.6), q = V.seam(R.r(phi) - 0.15, R.d(phi) - 0.15, g(R, phi) + 0.035, R.h(phi));
          if (sw > 0.55 && k % 2 === 0) crest.push([q, 3]);
          else if (sw < -0.6 && k % 2 === 1) crest.push([q, 0]);
        });
      }
      P.fill(HAIR, { flat: true, tone: 1, on: isHair });
      crest.forEach(([q, tone]) => { const X = Math.floor(q[0] * RES), Y = Math.floor(q[1] * RES); if (X >= 0 && Y >= 0 && X < PW && Y < PH && isHair(Y * PW + X)) P.dot(q[0], q[1], HAIR[tone]); });
    }
    // strands: curved lines down each lock, longer and shorter, not evenly spaced
    // (long straight hair gets the locks below instead, which these would stripe over)
    if (!st.wavy && !shape) for (let k = 0; k <= 26; k++) {
      const phi = FACE + 0.05 + k / 26 * (Math.PI * 2 - 2 * FACE - 0.1) + Math.sin(k * 2.7) * 0.04;
      const top = HEAD + 0.6 - (k % 3) * 0.9, bend = Math.sin(k * 1.7) * 0.08;
      const stop = shape ? 2.6 + (k % 2) * 1.4 : 1.2 + (k % 2) * 0.8;
      const a = V.seam(7, 7.4, phi, top), m = V.seam(7.35 + bulge(phi) * 0.6, 7.55 + bulge(phi) * 0.6, phi + bend, HEAD - 3.4), b2 = V.seam(7.4, 7.45, phi + bend * 1.6, endH(phi) + stop);
      // skip strands where the hair turns away at a grazing angle: there they would pack into stripes
      if (m[2] < 3.2) continue;
      P.l(a[0], a[1], m[0], m[1], 0.5).l(m[0], m[1], b2[0], b2[1], 0.5);
    }
    // (the strands land on hair pixels only, never hanging over the clothes as loose single strands)
    P.fill(HAIR, { flat: true, tone: 1, on: isHair });
    // sections: a dark groove up each split between sections, and a lit curve across each rounded end
    // (details for the ends land on hair pixels only, never on the clothes the hair lies over)
    const onHair = (x, y, c) => {
      const X = Math.floor(x * RES), Y = Math.floor(y * RES);
      if (X >= 0 && Y >= 0 && X < PW && Y < PH && isHair(Y * PW + X)) P.dot(x, y, c);
    };
    if (shape === 'sections') {
      for (let n = -nSec; n <= nSec; n++) {
        const phi = Math.PI + (n + 0.5) / nSec * Math.PI * 2;
        for (let j = 0; j <= 9; j++) { const q = V.seam(7.3, 6.6, phi, endH(phi) + 0.2 + j * 0.5); if (q[2] > 1.5) onHair(q[0], q[1], HAIR[j < 6 ? 0 : 1]); }
        const mid = Math.PI + n / nSec * Math.PI * 2;
        for (let t = -0.6; t <= 0.6; t += 0.08) { const ph = mid + t / nSec * Math.PI, q = V.seam(7.3, 6.6, ph, endH(ph) + 1.6); if (q[2] > 1.5) onHair(q[0], q[1], HAIR[3]); }
      }
    }
    // blunt: the curve of the curl-under catches the light in one clean line just above the edge
    if (shape === 'blunt') for (let k = 0; k <= 60; k++) {
      const phi = FACE + 0.1 + k / 60 * (Math.PI * 2 - 2 * FACE - 0.2), q = V.seam(6.9, 5.6, phi, endH(phi) + 1.9);
      if (q[2] > 1.5) onHair(q[0], q[1], HAIR[k % 7 === 3 ? 2 : 3]);
    }
    // a few lit strands catching the light on the upper part of each lock
    for (let k = 0; k <= 12; k++) {
      const phi = FACE + 0.2 + k / 12 * (Math.PI * 2 - 2 * FACE - 0.4), m = V.seam(7.3 + bulge(phi) * 0.6, 7.5 + bulge(phi) * 0.6, phi + 0.06, HEAD - 2.2);
      const a = V.seam(7.05, 7.45, phi + 0.04, HEAD + 0.2);
      if (m[2] > 2 && (k % 2 === 0)) P.l(a[0] + 0.3, a[1], m[0] + 0.3, m[1], 0.4);
    }
    P.fill(HAIR, { flat: true, tone: 3 });
    // Locks. Each of the clumps the outline swells into (bulge) is drawn as a rounded lock, so the hair has
    // form from every side, not only where the fringe and crown face us: a dark groove where two locks part,
    // and a lit ridge down the upper part of each lock, as bright as that lock is turned toward the light.
    // (wavy hair has its own strands that follow the waves)
    if (!st.wavy) {
      const [Lx, , Lz] = [-0.5, 0.55, 0.67];
      const lightOn = (phi) => { const s = Math.sin(phi), c = Math.cos(phi); return ((s * V.rx + c * V.fx) * Lx + (s * V.rz + c * V.fz) * Lz) / 0.84; };
      // a point a fraction t down a lock line at phi, from below the crown to just above its end
      const path = (phi, t, inset) => {
        const b = bulge(phi) * 0.6, top = HEAD + 0.8, end = endH(phi) + (shape ? 1.4 : 0.8);
        const h = t < 0.4 ? lerp(top, HEAD - 3.4, t / 0.4) : lerp(HEAD - 3.4, end, (t - 0.4) / 0.6);
        const r = t < 0.4 ? lerp(7.05, 7.35 + b, t / 0.4) : lerp(7.35 + b, 7.4, (t - 0.4) / 0.6);
        return V.seam(r - inset, r + 0.2 - inset, phi, h);
      };
      const run = (phi, t0, t1, inset, tone) => {
        // (a step of a quarter of a fine pixel or less along the lock, so the line has no gaps on long hair)
        const len = Math.abs(path(phi, 0, inset)[1] - path(phi, 1, inset)[1]) + 1, dt = 1 / (len * RES * 4);
        for (let t = t0; t <= t1; t += dt) { const q = path(phi, t, inset); if (q[2] > 2.6) onHair(q[0], q[1], HAIR[typeof tone === 'function' ? tone(t) : tone]); }
      };
      for (let n = 0; n < clumpN; n++) {
        // the swell of the lock (bulge peaks) and the parting before it (bulge dips)
        const ridge = (Math.PI / 2 + Math.PI * 2 * n) / clumpN, groove = ridge - Math.PI / clumpN;
        if (Math.abs(Math.atan2(Math.sin(groove), Math.cos(groove))) < FACE + 0.05) continue;
        // the groove: shallow under the crown, deepening down the length where the locks fall apart
        run(groove, 0.08 + (n % 3) * 0.04, 0.96, 0.05, (t) => (t < 0.45 ? 1 : 0));
        if (Math.abs(Math.atan2(Math.sin(ridge), Math.cos(ridge))) < FACE + 0.05) continue;
        const lit = lightOn(ridge), up = lit > 0.45 ? 4 : lit > 0.1 ? 3 : 2;
        // the ridge: brightest just under the crown, fading down the lock, a little to the lit side of centre
        // (two fine pixels wide, so the lock reads as a rounded form rather than another line)
        [0.03, 0.1].forEach((dp, j) => {
          run(ridge + dp, 0.05 + j * 0.04, 0.3 + (n % 2) * 0.12, 0.15, j ? Math.max(2, up - 1) : up);
          run(ridge + dp, 0.3 + (n % 2) * 0.12, 0.62 + (n % 3) * 0.08 - j * 0.1, 0.15, Math.max(2, up - 1 - j));
        });
      }
    }
  };
  add(-1, () => drawBob(false));
  if (st.extra) st.extra(add, P, A, HAIR);
  // Crown, fringe and the near side of the hair are one shape, filled in one pass so the shading runs
  // continuously from the bangs down the sides with no seam; the face is cut out of the whole.
  add(20.3, () => {
    P.e(V.x(0, -0.4), V.y(HEAD + 0.6), V.w(6.75, 7.1), 8);
    // short and tied-back styles trim the skull below their cap, so the rings alone set where the hair ends
    if (st.cap != null) P.cut([[CX - 12, V.y(st.cap)], [CX + 12, V.y(st.cap)], [CX + 12, V.y(HEAD - 9)], [CX - 12, V.y(HEAD - 9)]]);
    bob.filter((b) => b.facing).forEach((b) => P.p(b.q.map((p) => [p[0], p[1]])));
    if (V.fz > -0.25) {
      [1, -1].forEach((sd) => {
        const pts = [];
        for (let k = 0; k <= 10; k++) { const phi = sd * k / 10 * FACE; pts.push(V.seam(7.1, 7.5, phi, hairline(phi)).slice(0, 2)); }
        // down the cheek the opening follows a rounded curve (staying wide, then turning in under the jaw)
        // rather than a straight diagonal, so the hair frames a round lower face, not a V
        for (let k = 0; k <= 8; k++) { const t = k / 8; pts.push(V.seam(lerp(7.1, 5.4, t * t), lerp(7.5, 7, t), sd * FACE * (1 - 0.1 * t), HEAD - 3.4 - 8.6 * t).slice(0, 2)); }
        pts.push(V.seam(5.6, 7.2, 0, HEAD - 12).slice(0, 2));
        P.cut(pts);
      });
    }
    P.fill(HAIR, { pattern: cleanStrands, round: 1.1, dither: 0.3 });
    drawBob(true);
    // Crown detail, crisp like the fringe instead of a soft speckle: strands running from the top of the
    // head down over the crown into the fringe, and one clean band of sheen across the curve of the head,
    // broken where the strands cross it and brightest toward the light (upper left, positive phi).
    const crownR = (h) => Math.sqrt(Math.max(0, 1 - Math.pow((h - HEAD - 0.6) / 8, 2)));
    const onCrown = (phi, h, fn) => { const k = crownR(h), q = V.seam(6.75 * k + 0.05, 7.1 * k + 0.05, phi, h); if (q[2] > 1.2) fn(q[0], q[1]); };
    const strandPhis = [-0.95, -0.5, -0.05, 0.4, 0.85];
    const sheenH = (phi) => HEAD + 5.3 + Math.cos(phi + 0.4) * 0.7;
    if (V.fz > -0.6) strandPhis.forEach((phi, i) => {
      const end = sheenH(phi) + 0.4, top = HEAD + 7.6 - (i % 2) * 0.5;
      for (let h = top; h >= end; h -= 0.5) { const t = (top - h) / Math.max(0.5, top - end); onCrown(lerp(phi * 0.75 - 0.08, phi, t), h, (x, y) => P.dotOn(x, y, HAIR[t < 0.35 ? 2 : phi > 0 ? 2 : 1])); }
    });
    // the band of sheen runs right round the crown (seen from the side or behind it still catches the
    // light), brightest where the crown turns toward the light and dull on the far side of it
    const crownLight = (phi) => { const s = Math.sin(phi), c = Math.cos(phi); return ((s * V.rx + c * V.fx) * -0.5 + (s * V.rz + c * V.fz) * 0.67) / 0.84; };
    for (let k = 0; k <= 120; k++) {
      const phi = -Math.PI + k / 120 * Math.PI * 2, lit = crownLight(phi);
      if (strandPhis.some((sp) => Math.abs(sp - phi) < 0.05) || lit < -0.35) continue;
      onCrown(phi, sheenH(phi), (x, y) => P.dotOn(x, y, HAIR[lit > -0.05 ? 3 : 2]));
      if (lit > 0.25) onCrown(phi, sheenH(phi) + 0.5, (x, y) => P.dotOn(x, y, HAIR[lit > 0.6 ? 4 : 3]));
    }
    // (the fringe's shadow on the forehead is cast below, by stepping skin down its own ramp: a plain
    // darken here turned the warm skin grey, a pale band under the fringe like the rim of a cap)
    // side parting
    for (let k = 0; k <= 7; k++) { const q = V.seam(2.6, 6.6 - k * 1.5, 0, HEAD + 7.4 - Math.abs(k - 3) * 0.25); if (q[2] > -3) P.dotOn(V.x(bg.partA, 6.6 - k * 1.5), q[1], HAIR[1]); }
    // fringe: strand shadows sweeping across the forehead, and darker tips along the edge
    if (V.fz > -0.25) {
      [-0.9, -0.45, 0, 0.45].forEach((p0) => {
        // with a centre parting the locks sweep outward, so no strand runs across the middle
        const phi = bg.partA === 0 && p0 === 0 ? 0.9 : p0, dir = bg.dir(phi);
        const a = V.seam(6.6, 7, phi - 0.25 * dir, hairline(phi - 0.25 * dir) + 2.4), b = V.seam(7.05, 7.45, phi + 0.3 * dir, hairline(phi + 0.3 * dir) + 0.6);
        if (b[2] > 3 && a[2] > 3) P.l(a[0], a[1], b[0], b[1], 0.5);
      });
      P.fill(HAIR, { flat: true, tone: 1, on: true });
      // the edge of the fringe: darkest in the notches between locks and lighter at the tips, where the
      // hair thins out, so the edge reads as soft ends rather than one drawn line
      const between = (phi) => { const u = phi * F + 0.3; return Math.abs(u - Math.round(u)) * 2; };
      for (let k = 0; k <= 48; k++) { const phi = -FACE + k / 48 * 2 * FACE, h = V.seam(7.1, 7.5, phi, hairline(phi) + 0.25); if (h[2] > 1) P.dotOn(h[0], h[1], HAIR[between(phi) > 0.55 ? 1 : 2]); }
      // Fringe detail, so it reads as locks of hair rather than a flat band. The light comes from the
      // upper left (positive phi on screen), so locks on that side get the brighter highlights.
      const onF = (phi, h, fn) => { const q = V.seam(7.1, 7.5, phi, h); if (q[2] > 1) fn(q[0], q[1]); };
      for (let n = -nLocks; n <= nLocks; n++) {
        const tipPhi = (n - 0.3) / F, notch = (n + 0.5 - 0.3) / F, lit = tipPhi > -0.25, dir = bg.dir(notch);
        if (Math.abs(tipPhi) > FACE - 0.1) continue;
        // a groove parting two locks, curving toward the parting as it rises and fading out as the locks
        // merge higher up: deep only right at the notch, longer at every other one
        const gl = n % 2 === 0 ? 5 : 4;
        for (let j = 0; j <= gl; j++) onF(notch - j * 0.02 * dir, hairline(notch) + 0.3 + j * 0.5, (x, y) => P.dotOn(x, y, HAIR[j < 1 ? 0 : j < 3 ? 1 : 2]));
        // a fine strand down each lock, following its sweep, one shade off the hair around it, so the lock
        // has a grain without a hard line; it thins out (every third pixel left out) as it rises
        { const sp = tipPhi + 0.3 / F, top = 3 + (n & 1) * 0.5;
          for (let j = 1; j * 0.5 <= top; j++) {
            if (j > 3 && j % 3 === 0) continue;
            onF(sp - j * 0.012 * dir, hairline(sp) + 0.4 + j * 0.5, (x, y) => P.dotOn(x, y, HAIR[lit ? 3 : 2]));
          } }
        // locks on the lit side catch a soft sheen along the top of their curve, rising from the hair tone
        // to the brightest and back, so the light lies on the hair rather than on top of it
        if (lit) for (let j = 0; j <= 3; j++) onF(tipPhi + 0.02 - j * 0.015, hairline(tipPhi) + 1.4 + j * 0.5, (x, y) => P.dotOn(x, y, HAIR[j === 1 || j === 2 ? (n % 2 ? 4 : 3) : 3]));
        // a few fine ends slip past the cut edge onto the forehead, softening it
        if ((n * 7 + 3) % 3 === 0) onF(tipPhi + 0.03 * dir, hairline(tipPhi) - 0.2, (x, y) => P.dotOn(x, y, HAIR[2]));
      }
      // a centre parting: a dark line up the middle where the two sides fall apart
      if (bg.partA === 0) for (let j = 0; j <= 6; j++) onF(0, hairline(0) + 0.2 + j * 0.5, (x, y) => P.dotOn(x, y, HAIR[j < 4 ? 0 : 1]));
      // the fringe tucks under at its tips: a darker band just above the edge
      for (let k = 0; k <= 40; k++) { const phi = -FACE + 0.05 + k / 40 * (2 * FACE - 0.1); onF(phi, hairline(phi) + 0.7, (x, y) => P.darken(x, y, 0.94)); }
    }
    // The hair casts a faint shadow onto the cheek where it overlaps it: lit skin right against the hair
    // steps one shade down the skin ramp. Only skin pixels change, so the eyes, brows and hair are untouched.
    if (V.fz > -0.25) {
      const buf = P.buf, key = (c) => (Math.floor(c[0]) << 16) | (Math.floor(c[1]) << 8) | Math.floor(c[2]);
      const skinIdx = new Map(SKIN.map((c, i) => [key(c), i])), hairSet = new Set(HAIR.map(key));
      const px = (X, Y) => { const q = (Y * PW + X) * 4; return buf[q + 3] ? (buf[q] << 16) | (buf[q + 1] << 8) | buf[q + 2] : -1; };
      const Y0 = Math.max(0, Math.floor(V.y(HEAD + 3) * RES)), Y1 = Math.min(PH - 1, Math.ceil(V.y(HEAD - 9) * RES));
      const X0 = Math.max(2, Math.floor((CX - 9) * RES)), X1 = Math.min(PW - 3, Math.ceil((CX + 9) * RES));
      for (let Y = Y0; Y <= Y1; Y++) {
        const row = [];
        for (let X = X0 - 2; X <= X1 + 2; X++) row.push(px(X, Y));
        for (let X = X0; X <= X1; X++) {
          const i = skinIdx.get(row[X - X0 + 2]);
          if (i == null) continue;
          // only the pixel right against the hair, and only where the skin is lit (shadowed skin is left as is)
          if (i < 2 || !(hairSet.has(row[X - X0 + 1]) || hairSet.has(row[X - X0 + 3]))) continue;
          const c = SKIN[i - 1], q = (Y * PW + X) * 4;
          buf[q] = c[0]; buf[q + 1] = c[1]; buf[q + 2] = c[2];
        }
      }
    }
  });
}

export { FEMALE_HEAD, HAIRSTYLES, BANGS, drawFemaleHead };
