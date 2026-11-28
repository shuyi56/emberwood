// Hair fidelity: show a hairstyle as the creator sculpts it, or as the physics groom (tools/hair) baked for the game:
// cards at three levels of detail, or the strands themselves. Added to the page by apply-hair-fidelity.py, with each
// groomed style's asset (tools/hair/creator_asset.py) in a <script type="application/json" id="groom-asset-STYLE">.
import * as THREE from 'three';

const HH = 8.6;                                   // the creator's units per head height (rq_hair_export.cjs)
const MODES = [['creator', 'Creator (sculpted locks)'], ['lod2', 'Groom: cards, LOD 2'], ['lod1', 'Groom: cards, LOD 1'],
  ['lod0', 'Groom: cards, LOD 0'], ['strands', 'Groom: strands']];
const LOD_INDEX = { lod0: 0, lod1: 1, lod2: 2 };
const CAP_FADE_MM = 10;                           // STYLE: the scalp cap fades into the skin over this width at the hairline
const SCALP_SHADE = 0.4;                          // STYLE: the scalp cap against the hair colour: the shade map's root (0.5) in
                                                  // the shadow of the hair over it (render.py: 0.35)

function bytes(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; }
// head heights in the hair's frame (x across, y back, z up) to the page's units (x, y up, z toward the viewer)
function toPage(src, scale) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) { out[i] = src[i] * scale * HH; out[i + 1] = src[i + 2] * scale * HH; out[i + 2] = -src[i + 1] * scale * HH; }
  return out;
}

// How the cards look: the creator's own hair surface, so the groom reads as the same hair. The strand lines, the dark
// partings between locks, the single light-catching strands and the rounded relief of each lock all come from the
// creator's hair texture and bump map (hairField and paintHair in the creator; hairCapMat.map and .bumpMap here),
// mapped onto each card where the card lies round the head, so neighbouring cards carry on the creator's locks.
// The cut-out (each lock of the card ending at its own length, the soft root) is this file's own map, on uv1.
// The shade map: one value along every hair, multiplied into its colour, shared by the cards, the strands and the
// scalp. It is the game hair "root" map (UE4 Photorealistic Character: "a mask which is used to change the color of
// the hair from the root to the tip"; inZOI's DIRO map: "Hair root color and gradient starting point"), with the
// per-strand ID (each strand a little lighter or darker) and the baked AO (creator_asset.py) beside it.
// Dark at the root: the root lies under all the hair over it, so it gets the least light (self-shadowing: Yuksel &
// Keyser, Deep Opacity Maps; Scheuermann 2004 stores it as per-vertex AO). Lighter toward the ends: scalp hair grows
// about 1 cm a month (Wikipedia, Human hair growth), so the end of a 45 cm length is nearly four years old, and its
// colour has been changed by that much sunlight, mainly UVA (Nogueira & Joekes 2004, via Nogueira et al. 2006), light
// hair more than dark (Signori 2004, same review). No measured root-to-tip lightness was found: the amounts are STYLE.
const ROOT_DARK = 0.85;                           // STYLE: the root against the mid-length. It was 0.5, on top of the baked
                                                  // self-shadowing that already darkens covered roots: exposed roots along the
                                                  // fringe boundary then drew a dark band across the head
const ROOT_CM = 4;                                // STYLE: how far from the root the darkening reaches
const TIP_LIGHT = 1.35;                           // STYLE: the very end against the mid-length
const TIP_FRAC = 0.45;                            // STYLE: the share of each hair's length that lightens toward its end
const sstep = x => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
// s: distance from the root (page units); t: the share of the hair's length (0 root, 1 end)
const shadeAt = (s, t) => (ROOT_DARK + (1 - ROOT_DARK) * sstep(s / (ROOT_CM * 10 * HH / HEAD_MM))) *
  (1 + (TIP_LIGHT - 1) * sstep((t - (1 - TIP_FRAC)) / TIP_FRAC));
const CLUMPS = 6;                                 // STYLE: locks across one card in the cut-out
const ROOT_FADE = 0.05;                           // STYLE: the cut-out fades in over this much of the card at the root
const TEX_V_HH = 1.6;                             // the creator's hair texture runs this many head heights down the hair
const HAIR_SHADE = 0.8;                           // the creator's own (HAIR_SHADE): its hair is drawn at 0.8 of its paint
const CENTRE = [0, 0.56 * HH, -0.06 * HH];        // the skull's centre in the page's units (scalp_field.CENTRE)
const HAIR_MM = 0.07;                             // SOURCED: a scalp hair is 50-100 um across (Robbins, Chemical and Physical
                                                  // Behavior of Human Hair, ch. 9); the middle of that range
const HEAD_MM = 219;                              // one head height in mm (scalp_field.HH_MM)
// The fuzzy look of real hair: a strand's edge is soft, not a cut line, and among the full strands lie fine single
// hairs. Each ribbon is drawn FUZZ_W wider than the hairs it stands for and its cover falls off across it (full in
// the middle, none at the edge: the hairs of a strand lie scattered across it, not side by side), dithered per pixel
// so it still sorts by depth; FINE_SHARE of the strands are drawn FINE_W wide, as the loose single hairs. STYLE.
const FUZZ_W = 2.0, FUZZ_A = 0.9, FINE_SHARE = 0.3, FINE_W = 0.35;
// Layers: hair is dressed in sections, the top combed over the sides, the sides over the nape (hair_volume.py), and the
// under layers are darker than the canopy over them: less light reaches them, and the outer hair is the weathered,
// sun-lightened hair (Robbins ch. 5, photochemical lightening). A strand takes its section's tone (the innermost at
// LAYER_DARK of the outermost), and any point buried under other hair (its baked ambient) goes down to INNER_DARK,
// so an under layer seen past the edge of an upper one reads as a separate, deeper layer. STYLE amounts.
const LAYER_DARK = 0.75, INNER_DARK = 0.65;   // (0.5 and 0.45 stacked with the mid and coverage tones to flat black on the lower side)
// Glints: a hair's section is an ellipse and the hair twists along its length, so the light it focuses back comes and
// goes along it: "glints", bright near-white streaks that sit at different places on neighbouring hairs
// (Marschner et al. 2003, "Light Scattering from Human Hair Fibers", sect. 5.2.3 and their fig. on eccentricity). Here:
// a tight white lobe (GLINT_EXP) on top of the two highlights, let through only where the strand's own slow wave
// along its length (GLINT_MM long cycles, phase and rate its own) is near its crest (GLINT_DUTY of the length).
// SOURCED: the effect and its cause. STYLE: the amounts. STRAND_TONE: each strand this much lighter or darker (STYLE).
const GLINT = 0.5, GLINT_EXP = 110, GLINT_MM = 24, GLINT_DUTY = 0.35, STRAND_TONE = 0.32;
// The coverage layer. Hair for games and film is built from the inside out: first an opaque base that blocks the
// overall shape so nothing shows through ("make sure you cannot see through the hair onto the skull", Verhoolen /
// Lithvall, 80.lv; "the bottom layer of hair is 100 opaque", Hindsbol Hansen, CGMA; Unreal's Hair Card Generator
// names it Coverage: "the core layer, meant for creating coverage"), then the layers that carry the detail, with
// gaps left in them "to give the hair depth". Without it the soft strands lie over skin and over each other's gaps
// and the eye sees speckle, not depth. Here: every strand drawn a second time under itself, COVER_IN mm in toward
// the head, COVER_W wider, opaque, with no highlights, and at COVER_DARK of its tone: the light that reaches the
// inner hair is what many hairs let through (Zinke et al. 2008, dual scattering: T = d a^n for n hairs in front,
// d 0.7 in Sadeghi and Tamstorf's implementation), dark and without a highlight of its own.
// SOURCED: the layer and its job. STYLE: the three amounts.
// (COVER_DARK 0.78 and COVER_SPEC 0.35 until the fringe's face was checked in debug colours: between its locks most of
// what shows is this layer, and at that tone it stood as black patches on a face the light reaches in full.)
// A loose hair is not a thin slice of the mass (strands.iso, tools/hair/strand_iso.py: 0 in a mass, 1 alone). The mass
// is dark because light is absorbed hair after hair on its way in and out (Zinke et al. 2008); one fibre passes most of
// the light that enters it, tinted by its pigment (the TT lobe, Marschner et al. 2003), and shows the lit skin behind it
// between and through its few hairs. So a loose strand has no coverage layer under it, is lifted toward the fibre's own
// lit colour (LOOSE_LIFT, STYLE: by eye against the skin), and where the ribbon was widened to a pixel it covers only
// its true share of that pixel, never under LOOSE_MIN.
const LOOSE_LIFT = 1.6, LOOSE_MIN = 0.3, LOOSE_SPEC = 0.5;
const COVER_IN = 2.8, COVER_W = 1.8, COVER_DARK = 0.92, COVER_SPEC = 0.8, COVER_END = 0.72;
// (0.6 with no highlight read as a flat black band wherever the layer shows past the top hair: under the fringe's
// edge, along the lower side. So it is hair there too: COVER_DARK of the tone, COVER_SPEC of the highlights, its
// ribbons' edges broken like the strands', and it stops short of each hair's end (fading out from COVER_END of the
// length), so a hem is the top hair's own tips, not a dark slab behind them.)
const THONG_MM = 3.2;                             // STYLE: the tie's leather thong, across
// The mid layer (hair_layers.py: the strands gathered into thick clumps): drawn MID_W wider than a top strand and
// opaque, MID_IN mm under the top layer, at MID_DARK of its tone and MID_SPEC of its highlights, so the loose top
// hair lies over solid clumps with gaps between them down to the coverage layer. STYLE amounts.
const MID_W = 1.5, MID_IN = 1.0, MID_DARK = 0.8, MID_SPEC = 0.5;
const STRAND_TWIST = 0.7;                         // STYLE: rad each strand's normal is turned round it, either way
const STRAND_LIGHT = 0.85;                        // STYLE: the strands' paint against the hair colour: less light, so the
                                                  // lit side does not wash the strands into one flat sheet (cards: 0.8)
// Depth and sheen of the strands, after Scheuermann 2004 ("Practical Real-Time Hair Rendering and Shading", SIGGRAPH
// sketch; the ATI Ruby hair): (diffuse + two Kajiya-Kay highlights) x base x baked ambient occlusion. The primary
// highlight takes the light's colour and sits toward the tip; the secondary takes the hair's own colour, sits toward
// the root and is broken up strand to strand ("sparkle"). The tangent is shifted along the normal, T' = T + s N, to
// place each. SOURCED: the model and its structure. STYLE: every amount below (the paper prints none); the lobe
// widths follow Marschner's 5-10 degrees for the primary, wider for the secondary (Gerard, arXiv 2302.06468).
const SPEC1 = 0.09, SPEC1_EXP = 90, SPEC1_SHIFT = 0.1;     // STYLE: primary: strength, tightness, shift toward the tip
const SPEC2 = 0.22, SPEC2_EXP = 22, SPEC2_SHIFT = -0.14;    // STYLE: secondary: strength (x hair colour), toward the root
const SPEC_JITTER = 0.12;                         // STYLE: each strand's own extra shift, so the band breaks into glints
const LOCK_TONE = 0.2, CLUMP_TONE = 0.2;          // STYLE: a whole lock, and a clump in it, this much lighter or darker
                                                  // (the ID map of card hair: big clumps, then subclumps; 80.lv / CGMA)
const LIGHT_FLOOR = 0.08;                         // STYLE: a small rest of direct light in deep hair. It was 0.3 and flat; the light
                                                  // that many scatterings carry inward is now the forward-scatter term below
// Light under other hair (dual scattering: Zinke et al. 2008; the artist's form, Sadeghi & Tamstorf 2010). Light that
// passes n hairs comes out forward-scattered with T_f = d_f * a_f^n: it is spread wide, so it lights a hair from all
// round its axis, not by a surface normal, and it has taken the hair's hue on the way (Karis 2016).
const SC_DF = 0.7;                                // forward-scatter density d_f (Sadeghi & Tamstorf 2010: 0.7)
const SC_AF = 0.75;                               // DRAFT: share of the light one hair passes on forward, a_f. No source
                                                  // gives it for dark hair; the baked transmittance is exp(-0.6 n), so
const SC_EXP = Math.log(1 / SC_AF) / 0.6;         // a_f^n = T^(ln(1/a_f) / 0.6)
const SCATTER = 8.0;                              // STYLE: strength of the scattered diffuse. At 1 (the albedo's own share) dark
                                                  // brown hair gains 1 level of 255 under the filmic toe; Sadeghi & Tamstorf
                                                  // make the forward- and back-scatter intensities artist controls for this
const SC_SPEC = 1.5;                              // STYLE: the highlight the scattered light makes, x hair colour
const SC_BF2 = 0.02;                              // DRAFT: its spread per hair passed, sigma_f^2 = beta_f^2 n (Zinke et al. 2008);
                                                  // beta_f about 8 degrees, no cited value for this lobe form
// The glow of the lit top layer. A directly lit hair also returns the light the hairs under it scatter back, a wide
// soft lobe round the highlight (local back scattering, d_b * f_back: Zinke et al. 2008; Sadeghi & Tamstorf 2010), and
// passes light straight through itself toward a viewer facing the lamp (the TT lobe, Marschner et al. 2003): the pale
// halo on the outline.
const GLOW_BACK = 0.0, GLOW_BACK_EXP = 5;        // STYLE: back-scatter glow: strength and width
const GLOW_TT = 0.0, GLOW_TT_EXP = 4;            // STYLE: transmitted glow: strength and tightness round the lamp's direction
const GLOW_WHITE = 0.8;                           // STYLE: the glow's colour, from the hair's hue (0) to the lamp's white (1)
const KK_GAIN = 3.0;                              // STYLE: strength of the fibre diffuse past the Lambert term
// Direct light on a hair is not a surface's: a fibre is lit from all round its axis, by sin(T, L) (Kajiya & Kay 1989),
// and what darkens it is other hair, which the baked transmittance already holds. Lambert on the head's outward normal
// put a second, solid terminator on top and left the hair that faces away from every lamp black.
const KK_WRAP = 0.5;                              // STYLE: how far the light wraps round the head (Scheuermann 2004 wraps
                                                  // N.L to 0.25..1; Karis 2016 wraps the scattered part)
const SC_SAT = 0.5;                               // STYLE: how far the scattered light takes the hair's hue (Karis: absorption along the path)
const STRAND_ROUGH = 0.6;                         // STYLE: a soft sheen along the lit strands (0.8 read matte and flat)
const END_ON = 0.35;                               // STYLE: a ribbon is at full width once sin(tangent, view) passes this
const SPEC_SOFT = 1.5;                            // STYLE: soft ceiling on the summed highlights (x / (1 + k luma)): a lone wide
                                                  // ribbon lying flat to the lamp lit up edge to edge as a pale band
const SPEC_EDGE = 0.6;                            // STYLE: the highlights fall off by this much toward a ribbon's edges
const PX_WORLD = { value: 0.002 };                // one pixel's width at unit depth: set each frame from the camera
function cutTexture() {
  const W = 256, H = 512, c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const cw = W / CLUMPS;
  for (let j = 0; j < CLUMPS; j++) {
    const cx = (j + 0.5 + (rnd() - 0.5) * 0.3) * cw, len = H * (0.72 + 0.28 * rnd()), tipX = cx + (rnd() - 0.5) * cw * 0.4;
    const edge = Math.min(cx, W - cx) / (W * 0.5);              // the card's outer locks are thinner, so its edge is soft
    for (let k = 0; k < 90; k++) {
      const off = (rnd() - 0.5) * cw * 0.95, x0 = cx + off, end = len * (0.8 + 0.2 * rnd());
      const x1 = tipX + off * 0.25 * (end / len);                 // the lock narrows to its tip
      const centre = 1 - Math.abs(off) / (cw * 0.5);
      const a = Math.min(1, 0.45 + 0.9 * centre) * Math.min(1, 0.35 + 1.3 * edge) * (0.7 + 0.3 * rnd()), f = end / H;
      const grd = g.createLinearGradient(0, 0, 0, end);
      grd.addColorStop(0, 'rgba(255,255,255,0)'); grd.addColorStop(Math.min(0.5, ROOT_FADE / f), `rgba(255,255,255,${a})`);
      grd.addColorStop(0.92, `rgba(255,255,255,${a * 0.9})`); grd.addColorStop(1, 'rgba(255,255,255,0)');
      g.strokeStyle = grd; g.lineWidth = 1 + rnd() * 1.6; g.beginPath();
      g.moveTo(x0, 0); g.quadraticCurveTo(x0 + (rnd() - 0.5) * 4, end * 0.55, x1, end); g.stroke();
    }
  }
  const t = new THREE.CanvasTexture(c); t.flipY = false;          // canvas row y is v = y / H: the root at the top
  t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping; t.anisotropy = 8; t.channel = 1;
  return t;
}

// Per card vertex: the creator's texture coordinates (round the head from the card's root, down the hair by length),
// the shading normal (out from the head, as the creator's head of hair is lit: a flat card's own normal turned its
// underside black wherever it lifted off the head) and the colour (the baked light through the hair, darker roots).
function cardAttributes(P, UV, AO) {
  const n = P.length / 3, uv0 = new Float32Array(n * 2), N = new Float32Array(n * 3), C = new Float32Array(n * 3);
  let start = 0;
  for (let i = 0; i <= n; i += 2) {
    if (i < n && !(i > start && UV[2 * i + 1] === 0)) continue;
    // a card is the rows from `start` to i (two vertices a row, v 0 at its root)
    const mx = (P[3 * start] + P[3 * start + 3]) / 2, mz = (P[3 * start + 2] + P[3 * start + 5]) / 2 - CENTRE[2];
    const u0 = Math.atan2(mx, mz) / (2 * Math.PI) + 0.5, r = Math.max(1, Math.hypot(mx, mz));
    const w0 = Math.hypot(P[3 * start] - P[3 * start + 3], P[3 * start + 1] - P[3 * start + 4], P[3 * start + 2] - P[3 * start + 5]);
    const du = w0 / (2 * Math.PI * r);
    let s = 0, px = 0, py = 0, pz = 0;
    for (let j = start; j < i; j += 2) {
      const cx = (P[3 * j] + P[3 * j + 3]) / 2, cy = (P[3 * j + 1] + P[3 * j + 4]) / 2, cz = (P[3 * j + 2] + P[3 * j + 5]) / 2;
      if (j > start) s += Math.hypot(cx - px, cy - py, cz - pz);
      px = cx; py = cy; pz = cz;
      for (const q of [j, j + 1]) {
        uv0[2 * q] = u0 + (UV[2 * q] - 0.5) * du; uv0[2 * q + 1] = Math.min(1, s / (TEX_V_HH * HH));
        const ox = P[3 * q] - CENTRE[0], oy = P[3 * q + 1] - Math.min(P[3 * q + 1], CENTRE[1]), oz = P[3 * q + 2] - CENTRE[2];
        const l = Math.hypot(ox, oy, oz) || 1; N[3 * q] = ox / l; N[3 * q + 1] = oy / l; N[3 * q + 2] = oz / l;
        C[3 * q] = C[3 * q + 1] = C[3 * q + 2] = (AO ? AO[q] : 1) * shadeAt(s, UV[2 * q + 1]);
      }
    }
    start = i;
  }
  return { uv0, N, C };
}

function buildStyle(asset, tone, creatorMat) {
  const group = new THREE.Group(), parts = {};
  const col = new THREE.Color(tone);
  const cardMat = new THREE.MeshStandardMaterial({ color: 0xffffff, map: creatorMat.map, bumpMap: creatorMat.bumpMap,
    bumpScale: creatorMat.bumpScale, alphaMap: cutTexture(), alphaTest: 0.35, side: THREE.DoubleSide, vertexColors: true,
    roughness: 1, metalness: 0, envMapIntensity: 0.25 });
  // the creator's shade, and no flipping of the normal on a card's far side: both sides are the hair's outer surface
  cardMat.onBeforeCompile = sh => {
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <color_fragment>', '#include <color_fragment>\n\tdiffuseColor.rgb *= ' + HAIR_SHADE.toFixed(3) + ';')
      .replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\n\tnormal = normalize( vNormal );');
  };
  // the scalp under the hair in the root colour, as render.py draws it (SCALP_SHADE), so no skin shows between hairs
  const capMat = new THREE.MeshStandardMaterial({ color: col.clone().multiplyScalar(SCALP_SHADE), roughness: 1, metalness: 0, vertexColors: true, transparent: true });   // (fully rough, as UE's scalp: no stray highlight)
  asset.lods.forEach((L, i) => {
    const g = new THREE.BufferGeometry(), P = toPage(new Float32Array(bytes(L.pos)), 1), UV = new Float32Array(bytes(L.uv));
    const A = cardAttributes(P, UV, L.ao ? new Float32Array(bytes(L.ao)) : null);
    g.setAttribute('position', new THREE.BufferAttribute(P, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(A.uv0, 2));          // the creator's hair texture and relief
    g.setAttribute('uv1', new THREE.BufferAttribute(UV, 2));            // the card's own: the cut-out
    g.setAttribute('normal', new THREE.BufferAttribute(A.N, 3));
    g.setAttribute('color', new THREE.BufferAttribute(A.C, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(L.idx)), 1));
    const m = new THREE.Mesh(g, cardMat); m.castShadow = true; m.name = 'groom-cards-' + i;
    parts['lod' + i] = m; group.add(m);
  });
  const cg = new THREE.BufferGeometry();
  cg.setAttribute('position', new THREE.BufferAttribute(toPage(new Float32Array(bytes(asset.cap.pos)), 1), 3));
  cg.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(asset.cap.idx)), 1)); cg.computeVertexNormals();
  // The cap fades into the skin at the hairline. Solid to its edge, it stood as a black band wherever the roots thin out:
  // behind the ear, under the nape hem, at the temple. (UE4 Photorealistic Character paints the scalp under the hair
  // and fades that paint out at the hairline; the width is STYLE.)
  { const cp = cg.getAttribute('position'), ci = cg.getIndex().array, nV = cp.count, eCount = new Map();
    const ek = (a, b) => a < b ? a * nV + b : b * nV + a;
    for (let t = 0; t < ci.length; t += 3) for (let k = 0; k < 3; k++) { const e = ek(ci[t + k], ci[t + (k + 1) % 3]); eCount.set(e, (eCount.get(e) || 0) + 1); }
    const edge = new Set(); eCount.forEach((n, e) => { if (n === 1) { edge.add(Math.floor(e / nV)); edge.add(e % nV); } });
    const eb = [...edge], fade = CAP_FADE_MM * HH / HEAD_MM, rgba = new Float32Array(nV * 4);
    for (let v = 0; v < nV; v++) {
      let d2 = Infinity; const x = cp.getX(v), y = cp.getY(v), z = cp.getZ(v);
      for (const b of eb) { const dx = cp.getX(b) - x, dy = cp.getY(b) - y, dz = cp.getZ(b) - z; d2 = Math.min(d2, dx * dx + dy * dy + dz * dz); }
      const a = Math.min(1, Math.sqrt(d2) / fade);
      rgba[4 * v] = rgba[4 * v + 1] = rgba[4 * v + 2] = 1; rgba[4 * v + 3] = a * a * (3 - 2 * a);
    }
    cg.setAttribute('color', new THREE.BufferAttribute(rgba, 4)); }
  parts.cap = new THREE.Mesh(cg, capMat); group.add(parts.cap);
  // The strands: each drawn child stands for `hairs_per_strand` real hairs (about 9.6), so it is a ribbon that wide:
  // that many hairs of HAIR_MM side by side, turned to face the camera, and never under a pixel. (As one-pixel lines
  // they covered a ninth of what the hair covers, and seen close the skin and the ear showed through.) Lit like the
  // cards, with normals out from the head; a little lighter or darker strand to strand, dark at the root.
  const S = asset.strands, cnt = new Uint16Array(bytes(S.counts)), raw = new Int16Array(bytes(S.pos));
  const pts = new Float32Array(raw.length); for (let i = 0; i < raw.length; i++) pts[i] = raw[i] * S.q;
  const P = toPage(pts, 1), np = P.length / 3;
  const sAO = S.ao ? new Uint8Array(bytes(S.ao)) : null, sID = S.ids ? new Uint8Array(bytes(S.ids)) : null;
  // the light through the hair (tools/hair/hair_light.py), when the asset has it: per point, the transmittance toward
  // the key, the rim and the fill, and the ambient, from the hair's own flow and density. It then replaces the AO.
  // which way is out at each point (the head's on the head, the tail's own axis along the tail), when the asset has it
  const sOut = S.out ? toPage(Float32Array.from(new Int8Array(bytes(S.out))), 1 / HH) : null;
  const sTier = S.tier ? new Uint8Array(bytes(S.tier)) : null, tierA = new Float32Array(np * 2);
  const sLay = S.layer ? new Uint8Array(bytes(S.layer)) : null;
  const sL = S.light ? new Uint8Array(bytes(S.light)) : null, lit = new Float32Array(np * 8).fill(1);
  const wHair = (S.hairs_per_strand || 1) * HAIR_MM * HH / HEAD_MM;
  const pos = new Float32Array(np * 6), tan = new Float32Array(np * 6), nrm = new Float32Array(np * 6);
  const side = new Float32Array(np * 2), wid = new Float32Array(np * 2), vc = new Float32Array(np * 6), idx = [];
  // (the asset's own look, see below at `compile`. light_floor: the least of each lamp's light that reaches a hair
  // under other hair; inner_dark and mid_dark: this asset's INNER_DARK and MID_DARK. On a long loose fall the baked
  // light is all or nothing a lock down (median transmittance 0.01), and with the two layer tones on top the hair
  // between the outer locks went near black: every inner lock read as a hard shadow (John, 2026-11-28). Light in
  // hair past the first layers is light scattered many times and from the open sides (Zinke et al. 2008), which the
  // one-direction bake does not hold; Sadeghi & Tamstorf 2010 leave its strength to the artist. STYLE amounts.)
  const lk = asset.look || {}, lFloor = lk.light_floor || 0, innerDark = lk.inner_dark === undefined ? INNER_DARK : lk.inner_dark, midDark = lk.mid_dark === undefined ? MID_DARK : lk.mid_dark;
  const sIso = S.iso ? new Uint8Array(bytes(S.iso)) : null, isoA = new Float32Array(np * 2);   // how alone the hair is
  const sArc = new Float32Array(np * 2);          // per vertex: mm along the strand
  const occ = new Float32Array(np * 4);            // per vertex: the baked AO, and the strand's own random value
  let k = 0, si = 0, seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (const c of cnt) {
    const mid = sTier ? sTier[si] : 0;
    const lay = (mid ? midDark : 1) * (sLay ? LAYER_DARK + (1 - LAYER_DARK) * sLay[si] / 255 : 1);
    const jit = rnd(), tl0 = sID ? sID[2 * si] / 250 - 0.5 : 0, tc0 = sID ? sID[2 * si + 1] / 250 - 0.5 : 0; si++;
    const b = lay * (1 - STRAND_TONE + 2 * STRAND_TONE * rnd()) * (1 + 2 * LOCK_TONE * tl0) * (1 + 2 * CLUMP_TONE * tc0), twist = (2 * rnd() - 1) * STRAND_TWIST;
    const tc = Math.cos(twist), ts = Math.sin(twist), fineR = rnd(), fine = mid ? FUZZ_W * MID_W : fineR < FINE_SHARE ? FINE_W : FUZZ_W;
    for (let j = 0, sj = 0; j < c; j++) {
      const q = k + j, a0 = k + Math.max(0, j - 1), a1 = k + Math.min(c - 1, j + 1), t = j / (c - 1);
      if (j) sj += Math.hypot(P[3 * q] - P[3 * q - 3], P[3 * q + 1] - P[3 * q - 2], P[3 * q + 2] - P[3 * q - 1]);
      let tx = P[3 * a1] - P[3 * a0], ty = P[3 * a1 + 1] - P[3 * a0 + 1], tz = P[3 * a1 + 2] - P[3 * a0 + 2];
      const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
      const ox = sOut ? sOut[3 * q] : P[3 * q] - CENTRE[0], oy = sOut ? sOut[3 * q + 1] : P[3 * q + 1] - Math.min(P[3 * q + 1], CENTRE[1]), oz = sOut ? sOut[3 * q + 2] : P[3 * q + 2] - CENTRE[2];
      // the normal: out from the head but square to the strand (a hair is lit across, never along it, as in
      // Kajiya-Kay's hair model), then turned round the strand by its own twist: so each wave, bend and lock catches
      // the light by its own direction, where the head's shape alone lit the whole fall as one smooth shell
      const od = ox * tx + oy * ty + oz * tz;
      let nx = ox - od * tx, ny = oy - od * ty, nz = oz - od * tz;
      const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
      const bx = ty * nz - tz * ny, by = tz * nx - tx * nz, bz = tx * ny - ty * nx;
      const ol = 1; const ox2 = nx * tc + bx * ts, oy2 = ny * tc + by * ts, oz2 = nz * tc + bz * ts;
      const ao = sL ? 1 : sAO ? sAO[q] / 255 : 1, shade = b * shadeAt(sj, t) * ao * (sL ? innerDark + (1 - innerDark) * sstep(sL[4 * q + 3] / 128) : 1);
      for (let e = 0; e < 2; e++) {
        const v = 2 * q + e;
        pos.set(P.subarray(3 * q, 3 * q + 3), 3 * v); tan[3 * v] = tx; tan[3 * v + 1] = ty; tan[3 * v + 2] = tz;
        nrm[3 * v] = ox2; nrm[3 * v + 1] = oy2; nrm[3 * v + 2] = oz2;
        side[v] = e ? 1 : -1; wid[v] = wHair * fine * (1 - 0.6 * Math.max(0, (t - 0.85) / 0.15));      // (thinning over the last 15%)
        vc[3 * v] = vc[3 * v + 1] = vc[3 * v + 2] = shade; occ[2 * v] = ao; occ[2 * v + 1] = jit; tierA[v] = mid + 0.45 * t;      // (the tier, and under it the share of the length: 0..0.45)
        sArc[v] = sj * HEAD_MM / HH; isoA[v] = sIso ? sIso[q] / 255 : 0;
        if (sL) for (let c4 = 0; c4 < 4; c4++) lit[4 * v + c4] = lFloor + (1 - lFloor) * sL[4 * q + c4] / 255;
      }
      if (j < c - 1) idx.push(2 * q, 2 * q + 1, 2 * q + 3, 2 * q, 2 * q + 3, 2 * q + 2);
    }
    k += c;
  }
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); sg.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  sg.setAttribute('strandT', new THREE.BufferAttribute(tan, 3)); sg.setAttribute('strandSide', new THREE.BufferAttribute(side, 1));
  sg.setAttribute('strandW', new THREE.BufferAttribute(wid, 1)); sg.setAttribute('color', new THREE.BufferAttribute(vc, 3));
  sg.setAttribute('strandOcc', new THREE.BufferAttribute(occ, 2)); sg.setAttribute('strandArc', new THREE.BufferAttribute(sArc, 1)); sg.setAttribute('strandTier', new THREE.BufferAttribute(tierA, 1));
  sg.setAttribute('strandLight', new THREE.BufferAttribute(lit, 4)); sg.setAttribute('strandIso', new THREE.BufferAttribute(isoA, 1));
  sg.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
  sg.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, CENTRE[1], 0), 2 * HH);
  const strandMat = new THREE.MeshStandardMaterial({ color: col, vertexColors: true, side: THREE.DoubleSide, roughness: STRAND_ROUGH,
    metalness: 0, envMapIntensity: 0.15 });
  // The asset's own look (asset.look, optional): how far the primary highlight and the glints take the hair's hue.
  // Glints are part of the TRT light, which has crossed the fibre twice and carries its pigment (Marschner et al. 2003,
  // sect. 5.2.3), so a near-white glint is wrong for dark hair; and on a long wavy fall, where the highlight is broken
  // over every wave, a half-white primary read as white ashy hair (John, 2026-11-28). The defaults keep the looks
  // that were accepted before (the ponytail).
  const look = asset.look || {}, spec1Hue = look.spec1_hue === undefined ? 0.5 : look.spec1_hue;
  // (and how light the hair is: dark hair absorbs most of the light that enters it, so its coloured highlight, its
  // glints and its scattered light are weak (the TT and TRT lobes fall with absorption, Marschner et al. 2003; the
  // multiple scattering with the albedo, Zinke et al. 2008). The hue-normalised highlights above keep their full
  // strength whatever the hair's tone, so a dark style states its own amounts. Measured against the creator's own
  // hair from the same cameras. Defaults are the constants.)
  const LK = Object.assign({ spec1: SPEC1, spec2: SPEC2, glint: GLINT, scatter: SCATTER, kk_gain: KK_GAIN, strand_light: STRAND_LIGHT }, look);
  const hueGLSL = 'diffuseColor.rgb / max( max( diffuseColor.r, diffuseColor.g ), 0.001 )';
  const glintCol = look.glint_hue === undefined ? 'vec3( 1.0, 0.97, 0.92 )' : 'mix( vec3( 1.0 ), ' + hueGLSL + ', ' + look.glint_hue.toFixed(3) + ' )';
  const compile = cover => sh => {
    sh.uniforms.pxWorld = PX_WORLD; sh.uniforms.coverIn = { value: cover ? COVER_IN * HH / HEAD_MM : 0 }; sh.uniforms.coverW = { value: cover ? COVER_W : 1 };
    sh.vertexShader = 'attribute vec3 strandT;\nattribute float strandSide;\nattribute float strandW;\nuniform float pxWorld;\nuniform float coverIn;\nuniform float coverW;\n' +
      'attribute vec2 strandOcc;\nvarying vec3 vStrandT;\nvarying vec2 vStrandOcc;\nattribute vec4 strandLight;\nvarying vec4 vLight;\nvarying float vSide;\nattribute float strandArc;\nvarying float vArc;\nattribute float strandTier;\nvarying float vTier;\nvarying float vTip;\nattribute float strandIso;\nvarying float vIso;\nvarying float vThin;\n' +
      sh.vertexShader.replace('#include <project_vertex>', [
        'vec4 mvPosition = modelViewMatrix * vec4( transformed - ( coverIn + floor( strandTier + 0.01 ) * ' + (MID_IN * HH / HEAD_MM).toFixed(5) + ' ) * normal, 1.0 );',
        'vec3 tv = normalize( ( modelViewMatrix * vec4( strandT, 0.0 ) ).xyz );',
        'vStrandT = tv; vStrandOcc = strandOcc; vLight = strandLight; vSide = strandSide; vArc = strandArc; vTier = floor( strandTier + 0.01 ); vTip = fract( strandTier + 0.01 ) / 0.45;',
        // (a ribbon that runs toward the camera has no side to face it with: its side vector flips from vertex to vertex
        // and the ribbon knots into black facets. It narrows to a hair's width there instead.)
        'vec3 sdc = cross( tv, normalize( -mvPosition.xyz ) ); float sdl = length( sdc );',
        'vec3 sd = sdc / max( sdl, 1e-4 );',
        'float wTrue = strandW * coverW * smoothstep( 0.0, ' + END_ON.toFixed(2) + ', sdl ), wPx = -mvPosition.z * pxWorld;',
        'vIso = strandIso; vThin = clamp( wTrue / wPx, 0.0, 1.0 );',
        'mvPosition.xyz += sd * ( 0.5 * max( wTrue, wPx ) * strandSide );',
        'gl_Position = projectionMatrix * mvPosition;'].join('\n'));
    const f = x => x.toFixed(3);
    sh.fragmentShader = 'varying vec3 vStrandT;\nvarying vec2 vStrandOcc;\nvarying vec4 vLight;\nvarying float vSide;\nvarying float vArc;\nvarying float vTier;\nvarying float vTip;\nvarying float vIso;\nvarying float vThin;\n' + sh.fragmentShader
      // each light by the hair's own transmittance toward it (the lights in the scene's order: key, rim, fill; the key
      // casts the only shadow, so it stays first), the ambient by the hair's averaged transmittance
      .replace('#include <lights_fragment_begin>', THREE.ShaderChunk.lights_fragment_begin.replace('getDirectionalLightInfo( directionalLight, directLight );',
        'getDirectionalLightInfo( directionalLight, directLight ); directLight.color *= max( vLight[ i ], ' + LIGHT_FLOOR.toFixed(3) + ' );'))
      .replace('#include <lights_fragment_end>', '#include <lights_fragment_end>\n\t{ float sAmb = max( vLight.w, ' + SC_DF.toFixed(3) + ' * pow( max( vLight.w, 0.004 ), ' + SC_EXP.toFixed(3) + ' ) ); reflectedLight.indirectDiffuse *= sAmb; reflectedLight.indirectSpecular *= sAmb; }' +
        (cover ? ' reflectedLight.directSpecular *= 0.0; reflectedLight.indirectSpecular *= 0.0;' : ''))     // (no sheen of its own on the coverage layer: at grazing angles it showed as grey patches on the tail)
      .replace('#include <color_fragment>', '#include <color_fragment>\n\tdiffuseColor.rgb *= ' + LK.strand_light.toFixed(3) + ';\n' +
        '\t{ float fz = ' + FUZZ_A.toFixed(3) + ' * ( 1.0 - vSide * vSide );\n' +
        '\t  float hn = fract( 52.9829189 * fract( dot( gl_FragCoord.xy, vec2( 0.06711056, 0.00583715 ) ) ) + vStrandOcc.y );\n' +
        (cover ? '\t  diffuseColor.rgb *= ' + COVER_DARK.toFixed(3) + ';\n\t  if ( min( 1.6 * ( 1.0 - vSide * vSide ), 1.0 - smoothstep( ' + COVER_END.toFixed(3) + ', 0.98, vTip ) ) * ( 1.0 - smoothstep( 0.15, 0.6, vIso ) ) < hn ) discard; }' :
          '\t  diffuseColor.rgb *= 1.0 + ' + LOOSE_LIFT.toFixed(3) + ' * vIso;\n\t  if ( vTier < 0.5 && fz < hn ) discard;\n' +
          '\t  if ( mix( 1.0, max( vThin, ' + LOOSE_MIN.toFixed(3) + ' ), vIso ) < fract( hn + 0.618 ) ) discard; }'))
      // (look.view_side: hair seen from its inner side, the back hair behind the neck seen from the front, has an
      // "out" that faces away from the viewer and every lamp and came out black. A fibre has no inner side (Kajiya &
      // Kay 1989): its normal is mirrored toward the viewer, and the baked light still darkens it.)
      .replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\n\tnormal = normalize( vNormal );' +
        (look.view_side ? ' { vec3 vv = normalize( vViewPosition ); normal = normalize( normal - 2.0 * min( 0.0, dot( normal, vv ) ) * vv ); }' : ''))
      // the two highlights, per directional light, on top of the standard shading and under the same baked AO
      .replace('#include <opaque_fragment>', [
        '#if NUM_DIR_LIGHTS > 0',
        '{ vec3 sT = normalize( vStrandT ), sV = normalize( vViewPosition ), sAcc = vec3( 0.0 ), sSc = vec3( 0.0 ), sKK = vec3( 0.0 ), sScS = vec3( 0.0 ), sGl = vec3( 0.0 );',
        '  vec3 t1 = normalize( sT + ' + f(SPEC1_SHIFT) + ' * normal );',
        '  vec3 t2 = normalize( sT + ( ' + f(SPEC2_SHIFT) + ' + ' + f(SPEC_JITTER) + ' * ( vStrandOcc.y - 0.5 ) ) * normal );',
        // (the primary half-way to the hair's hue: pure light colour on single strands read as grey hairs. STYLE: Scheuermann's is untinted)
        '  vec3 sWarm = mix( vec3( 1.0 ), ' + hueGLSL + ', ' + spec1Hue.toFixed(3) + ' );',
        '  float gPh = 6.2832 * ( vArc / ( ' + f(GLINT_MM) + ' * ( 0.6 + 0.8 * vStrandOcc.y ) ) + 17.0 * vStrandOcc.y );',
        '  float gOn = smoothstep( ' + f(1 - 2 * GLINT_DUTY) + ', 1.0, 0.6 * sin( gPh ) + 0.4 * sin( 2.7 * gPh + 40.0 * vStrandOcc.y ) );',
        '  for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {',
        '    vec3 sL = directionalLights[ i ].direction, sH = normalize( sL + sV );',
        '    float d1 = dot( t1, sH ), d2 = dot( t2, sH );',
        '    float s1 = smoothstep( -1.0, 0.0, d1 ) * pow( sqrt( max( 0.0, 1.0 - d1 * d1 ) ), ' + f(SPEC1_EXP) + ' );',
        '    float s2 = smoothstep( -1.0, 0.0, d2 ) * pow( sqrt( max( 0.0, 1.0 - d2 * d2 ) ), ' + f(SPEC2_EXP) + ' );',
        '    float dg = dot( sT, sH ), sg = pow( max( 0.0, 1.0 - dg * dg ), ' + f(GLINT_EXP / 2) + ' );',
        '    float wrap = max( 0.0, 0.75 * dot( normal, sL ) + 0.25 );',
        '    sAcc += directionalLights[ i ].color * max( vLight[ i ], ' + f(0.5 * LIGHT_FLOOR) + ' ) * wrap * ( ' + f(LK.spec1) + ' * s1 * sWarm + ' + f(LK.spec2) + ' * s2 * ( 0.4 + 1.2 * vStrandOcc.y ) * diffuseColor.rgb + ' + f(LK.glint) + ' * gOn * sg * vLight[ i ] * ' + glintCol + ' );',
        // forward-scattered light on the part the direct light does not reach: lit along the strand (Kajiya-Kay
        // sin(T, L)), wrapped wide round the head instead of cut off at the surface normal (Karis's wrapped Lambert)
        '    float sTf = ' + f(SC_DF) + ' * pow( max( vLight[ i ], 0.004 ), ' + f(SC_EXP) + ' ) * ( 1.0 - vLight[ i ] );',
        '    sGl += directionalLights[ i ].color * vLight[ i ] * ( ' + f(GLOW_BACK) + ' * wrap * smoothstep( -1.0, 0.0, d2 ) * pow( sqrt( max( 0.0, 1.0 - d2 * d2 ) ), ' + f(GLOW_BACK_EXP) + ' ) + ' + f(GLOW_TT) + ' * pow( max( 0.0, -dot( sV, sL ) ), ' + f(GLOW_TT_EXP) + ' ) * sqrt( max( 0.0, 1.0 - dot( sT, sL ) * dot( sT, sL ) ) ) );',
        '    float sLT = dot( sT, sL ), sSin = sqrt( max( 0.0, 1.0 - sLT * sLT ) ), sNL = dot( normal, sL );',
        '    sSc += directionalLights[ i ].color * sTf * ( 0.5 + 0.5 * sNL ) * sSin;',
        // and its highlight: the secondary lobe, spread wider by every hair the light has passed
        '    float sES = 1.0 / ( ' + f(1 / SPEC2_EXP) + ' + ' + f(SC_BF2) + ' * ( -log( max( vLight[ i ], 0.004 ) ) / 0.6 ) );',
        '    sScS += directionalLights[ i ].color * sTf * ( 0.5 + 0.5 * sNL ) * smoothstep( -1.0, 0.0, d2 ) * pow( sqrt( max( 0.0, 1.0 - d2 * d2 ) ), sES );',
        // the fibre's own diffuse where it exceeds the surface's Lambert term, under the same transmittance
        '    sKK += directionalLights[ i ].color * max( vLight[ i ], ' + f(LIGHT_FLOOR) + ' ) * max( 0.0, sSin * clamp( ( sNL + ' + f(KK_WRAP) + ' ) / ' + f(1 + KK_WRAP) + ', 0.0, 1.0 ) - clamp( sNL, 0.0, 1.0 ) );',
        '  }',
        '' + (cover ? '' : '  outgoingLight += ( 1.0 - vTier ) * sGl * mix( diffuseColor.rgb / max( max( diffuseColor.r, max( diffuseColor.g, diffuseColor.b ) ), 0.001 ), vec3( 1.0, 0.97, 0.92 ), ' + f(GLOW_WHITE) + ' ) * vStrandOcc.x;'),
        '  outgoingLight += ( ' + f(LK.kk_gain) + ' * RECIPROCAL_PI * sKK + ' + f(SC_SPEC) + ' * ( 0.4 + 1.2 * vStrandOcc.y ) * sScS ) * diffuseColor.rgb * vStrandOcc.x;',
        '  outgoingLight += ' + f(LK.scatter) + ' * RECIPROCAL_PI * sSc * diffuseColor.rgb * pow( diffuseColor.rgb / max( max( diffuseColor.r, max( diffuseColor.g, diffuseColor.b ) ), 0.001 ), vec3( ' + f(SC_SAT) + ' ) ) * mix( 1.0, vStrandOcc.x, 0.5 );',
        '  sAcc *= ( 1.0 - ' + f(SPEC_EDGE) + ' * vSide * vSide ) / ( 1.0 + ' + f(SPEC_SOFT) + ' * dot( sAcc, vec3( 0.2126, 0.7152, 0.0722 ) ) );',
        '  outgoingLight += ' + (cover ? COVER_SPEC.toFixed(3) + ' * ' : '') + 'mix( 1.0, ' + MID_SPEC.toFixed(3) + ', vTier ) * mix( 1.0, ' + LOOSE_SPEC.toFixed(3) + ', vIso ) * sAcc * vStrandOcc.x * vStrandOcc.x; }',
        '#endif',
        '#include <opaque_fragment>'].join('\n'));
  };
  // the tie, on the field's own tie point: a leather thong wound three times round the gathered hair (as the
  // creator's ponytail has it, which stands where that style's sculpted tail leaves the head, not this groom's)
  if (asset.tie) {
    const c = toPage(new Float32Array(asset.tie.c), 1), t = toPage(new Float32Array(asset.tie.t), 1 / HH);
    const ax = new THREE.Vector3(t[0], t[1], t[2]).normalize(), R = asset.tie.r * HH;
    // fitted to the hair it holds (tie.fit 'hair'): the thong's inner face lies on the hair, its thickness fixed (THONG_MM)
    const tube = asset.tie.fit ? 0.5 * THONG_MM * HH / HEAD_MM : R * 0.11, major = asset.tie.fit ? R + 0.7 * tube : R * 1.04, pitch = asset.tie.fit ? 2.1 * tube : R * 0.2;
    const leather = new THREE.MeshStandardMaterial({ color: 0x5c3d2b, roughness: 0.7, envMapIntensity: 0.4 });
    parts.tie = new THREE.Group(); parts.tie.name = 'groom-tie';
    for (let w = -1; w <= 1; w++) {
      const m = new THREE.Mesh(new THREE.TorusGeometry(major, tube, 10, 40), leather);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), ax);
      m.position.set(c[0] + ax.x * w * pitch, c[1] + ax.y * w * pitch, c[2] + ax.z * w * pitch); m.castShadow = true;
      parts.tie.add(m);
    }
    group.add(parts.tie);
  }
  strandMat.onBeforeCompile = compile(false); strandMat.customProgramCacheKey = () => 'strand' + JSON.stringify(look);
  const coverMat = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 1, metalness: 0, envMapIntensity: 0 });
  coverMat.color = strandMat.color; coverMat.onBeforeCompile = compile(true); coverMat.customProgramCacheKey = () => 'strand-cover' + JSON.stringify(look);
  const cover = new THREE.Mesh(sg, coverMat); cover.frustumCulled = false; cover.name = 'groom-coverage';
  parts.strands = new THREE.Mesh(sg, strandMat); parts.strands.add(cover); parts.strands.frustumCulled = false; parts.strands.castShadow = false;
  // look.cast_shadow: the hair's shadow on the skin. Without it the skin beside and under the hair is as bright as the
  // open face, and the hair reads as lying in the face's own plane, cutting it (John, 2026-11-28): what sets hair off
  // the face is its shadow on it (Yuksel & Keyser 2008 on hair shadows giving the hair its depth; the creator's own
  // hair casts one). The ribbons have no width until the vertex shader gives it, so the shadow pass widens them the
  // same way, toward the lamp. The hair does not take this shadow itself: its own is the baked light.
  if (look.cast_shadow) {
    const dm = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, side: THREE.DoubleSide });
    dm.onBeforeCompile = sh => {
      sh.vertexShader = 'attribute vec3 strandT;\nattribute float strandSide;\nattribute float strandW;\n' +
        sh.vertexShader.replace('#include <project_vertex>', [
          'vec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0 );',
          'vec3 tv = normalize( ( modelViewMatrix * vec4( strandT, 0.0 ) ).xyz );',
          'vec3 sdc = cross( tv, vec3( 0.0, 0.0, 1.0 ) ); vec3 sd = sdc / max( length( sdc ), 1e-4 );',
          'mvPosition.xyz += sd * ( 0.5 * strandW * ' + (look.shadow_width || 1).toFixed(2) + ' * strandSide );',
          'gl_Position = projectionMatrix * mvPosition;'].join('\n'));
    };
    dm.customProgramCacheKey = () => 'strand-depth' + JSON.stringify(look);
    parts.strands.customDepthMaterial = dm; parts.strands.castShadow = true;
  }
  group.add(parts.strands);
  const segs = idx.length / 6;
  return { group, parts, cardMat, capMat, strandMat, segs, tone };
}

// The body the groom was draped on (creator_asset.py --body: the character's sections, as the groom's colliders read
// them, groom.py character_body): each section a superellipse (|x/w|^p + |y/d|^p = 1, its front and back depths apart)
// round x 0, y CY, the sections joined by the same smoothstep the colliders use. Shown so the hair that lies on her
// shoulders and cape reads as lying on them, not as splaying into the air.
const BODY_ROWS = 8, BODY_SEG = 64;
const DRAPE_VIEW = { az: 35, el: 10, dist: 62, ty: -0.25 };   // STYLE: 3/4 from a little above, shoulders to mid-back
const BODY_COLOUR = 0x4d4a47;                     // STYLE: a neutral cloth grey, clear of every hair tone in hue and value
function buildBody(body) {
  const S = body.sections, CY = body.cy, e = 2 / body.p;
  const ring = smooth => {
    const rows = [];
    for (let k = 0; k < S.length - 1; k++) for (let j = 0; j < BODY_ROWS; j++) {
      let t = j / BODY_ROWS; const z = S[k][0] + (S[k + 1][0] - S[k][0]) * t; if (smooth) t = t * t * (3 - 2 * t);
      rows.push([z, ...[1, 2, 3].map(i => S[k][i] + (S[k + 1][i] - S[k][i]) * t)]);
    }
    rows.push(S[S.length - 1]);
    const P = new Float32Array(rows.length * BODY_SEG * 3);
    const sp = v => Math.sign(v) * Math.pow(Math.abs(v), e);
    rows.forEach(([z, w, df, db], r) => {
      for (let i = 0; i < BODY_SEG; i++) {
        const a = i / BODY_SEG * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a), o = (r * BODY_SEG + i) * 3;
        P[o] = w * sp(s); P[o + 1] = CY + (c < 0 ? df : db) * sp(c); P[o + 2] = z;       // (c < 0: the front, -y)
      }
    });
    return [rows.length, P];
  };
  const [nr, P] = ring(true), idx = [];
  for (let r = 0; r < nr - 1; r++) for (let i = 0; i < BODY_SEG; i++) {
    const a = r * BODY_SEG + i, b = r * BODY_SEG + (i + 1) % BODY_SEG;
    idx.push(a, a + BODY_SEG, b, b, a + BODY_SEG, b + BODY_SEG);
  }
  // the shape is the colliders' (smoothstep between sections), the shading a straight blend's: the smoothstep's flat
  // spot at every section lit as a row of terraces
  const lin = new THREE.BufferGeometry();
  lin.setAttribute('position', new THREE.BufferAttribute(toPage(ring(false)[1], 1), 3));
  lin.setIndex(idx); lin.computeVertexNormals();
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(toPage(P, 1), 3));
  g.setIndex(idx); g.setAttribute('normal', lin.getAttribute('normal')); lin.dispose();
  const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: BODY_COLOUR, roughness: 0.95, metalness: 0, side: THREE.DoubleSide }));
  m.name = 'groom-body-' + body.name; m.receiveShadow = true;
  return m;
}

function start(fc) {
  const assets = {};
  document.querySelectorAll('script[id^="groom-asset-"]').forEach(s => { assets[s.id.slice(12)] = s; });
  const scene = fc.hairMesh.parent, built = {};
  let mode = 'creator';
  const ui = document.createElement('label'); ui.className = 'check'; ui.appendChild(document.createTextNode('Fidelity '));
  const sel = document.createElement('select'); sel.id = 'sel-hair-fidelity'; sel.setAttribute('aria-label', 'Hair fidelity');
  MODES.forEach(([v, t]) => { const o = document.createElement('option'); o.value = v; o.textContent = t; sel.appendChild(o); });
  sel.addEventListener('change', () => { mode = sel.value; });
  ui.appendChild(sel);
  // the shoulders: the body the groom was draped on, on by default with a groom
  const bodyUi = document.createElement('label'); bodyUi.className = 'check';
  const bodyBox = document.createElement('input'); bodyBox.type = 'checkbox'; bodyBox.checked = true; bodyBox.id = 'chk-hair-body';
  bodyUi.appendChild(bodyBox); bodyUi.appendChild(document.createTextNode(' Shoulders (the drape body)'));
  const note = document.createElement('div'); note.style.cssText = 'font-size:11px;opacity:.7;margin:2px 0 6px';
  const place = () => { const h = document.getElementById('sel-hair'); if (h && !ui.isConnected) { h.parentElement.after(ui); ui.after(bodyUi); bodyUi.after(note); } };
  // the creator's hair: the head of hair, its locks, and the sheer brown veils it lays over the scalp between strands
  // (hairVeil: vertex-coloured, custom-blended); a veil left showing under the groom read as a flat red-brown patch
  const isVeil = mat => mat && mat.isMeshBasicMaterial && mat.vertexColors && mat.blending === THREE.CustomBlending;
  const isCreatorHair = m => m.isMesh && (m === fc.hairMesh || m.material === fc.hairCapMat || isVeil(m.material) ||
    (m.material && fc.hairCapMat.map && m.material.map === fc.hairCapMat.map));
  // the creator's own ponytail tie (LEATHER, main.js): it marks that style's sculpted tail, so it goes with it
  const isCreatorTie = m => m.isMesh && m.material && m.material.color && m.material.color.getHex() === 0x5c3d2b && m.material.vertexColors;
  const asset_n = el => (el._n = el._n || JSON.parse(el.textContent).strands.n).toLocaleString();
  function frame() {
    place();
    const el = fc.renderer.domElement;
    PX_WORLD.value = 2 * Math.tan(fc.camera.fov * Math.PI / 360) / Math.max(1, el.clientHeight);
    const style = fc.P.hairStyle, have = !!assets[style];
    const lods = have ? (assets[style]._lods = assets[style]._lods ?? JSON.parse(assets[style].dataset.summary || '{}')) : {};
    [...sel.options].forEach(o => { o.disabled = o.value !== 'creator' && (!have || (o.value in LOD_INDEX && !lods[o.value])); });
    if (!have && mode !== 'creator') { mode = 'creator'; sel.value = 'creator'; }
    note.textContent = have ? '' : 'No groom for this style yet (groomed: ' + Object.keys(assets).join(', ') + ')';
    const groom = mode !== 'creator' && fc.P.hair;
    let changed = false;
    const show = (m, v) => { if (m && m.visible !== v) { m.visible = v; changed = true; } };      // (a groom may have no cards yet: strands only)
    scene.children.forEach(m => { if (isCreatorHair(m) || isCreatorTie(m)) show(m, fc.P.hair && !groom); });
    Object.values(built).forEach(b => { if (!(groom && b === built[style])) show(b.group, false); });
    if (groom) {
      let b = built[style];
      if (!b) { const A = JSON.parse(assets[style].textContent);
        b = built[style] = buildStyle(A, fc.P.hairTone, fc.hairCapMat); scene.add(b.group);
        if (A.body) { b.parts.body = buildBody(A.body); b.group.add(b.parts.body); }
        note.textContent = ''; }
      if (b.tone !== fc.P.hairTone) { const c = new THREE.Color(fc.P.hairTone); b.capMat.color.copy(c).multiplyScalar(SCALP_SHADE); b.strandMat.color.copy(c); b.tone = fc.P.hairTone; changed = true; }
      show(b.group, true);
      show(b.parts.lod0, mode === 'lod0'); show(b.parts.lod1, mode === 'lod1'); show(b.parts.lod2, mode === 'lod2');
      show(b.parts.cap, true); show(b.parts.strands, mode === 'strands');
      if (b.parts.body) show(b.parts.body, bodyBox.checked);
      if (mode in LOD_INDEX) { const a = JSON.parse(assets[style].dataset.summary || '{}'); note.textContent = a[mode] || ''; }
      else note.textContent = b.segs.toLocaleString() + ' segments, ' + asset_n(assets[style]) + ' strands';
    }
    if (changed) fc.controls.dispatchEvent({ type: 'change' });      // (the page draws on demand: ask for a frame)
    requestAnimationFrame(frame);
  }
  // a "Drape" camera: back far enough, and low enough, to see the hair on her shoulders and down her back
  fc.VIEWS.drape = DRAPE_VIEW;
  const seg = document.querySelector('.seg[aria-label="Camera"]');
  if (seg && !seg.querySelector('[data-view="drape"]')) {
    const d = document.createElement('button'); d.type = 'button'; d.dataset.view = 'drape'; d.textContent = 'Drape';
    d.addEventListener('click', () => fc.setView('drape')); seg.appendChild(d);
  }
  window.__hairFidelity = { set: v => { mode = v; sel.value = v; }, get: () => mode, built, body: v => { bodyBox.checked = v; } };
  frame();
}

(function wait() { if (window.__fc && window.__fc.hairMesh && window.__fc.hairMesh.parent) start(window.__fc); else setTimeout(wait, 200); })();
