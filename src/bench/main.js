import './styles.css';
import { PW, PH, RES, WALK } from '../shared/sprites.js';
import { loadSheet } from '../shared/sheet-loader.js';
import { HAIRSTYLES } from '../shared/presets/female-head.js';
import { SKINS } from '../shared/core.js';
import { ROSTER, loadParty, saveParty, memberSpec } from '../shared/party.js';

async function main() {

  // ---------- Bench ----------
  // the cast, and the party the world uses: who is in it and each one's hairstyle (saved on every change)
  const party = loadParty();
  const CAST = ROSTER.map((c) => ({ ...c, spec: memberSpec(c.key, party) }));
  const FACINGS = ['Front', 'Front right', 'Right', 'Back right', 'Back', 'Back left', 'Left', 'Front left'];
  const FRAMES = ['Idle 1', 'Idle 2'].concat(Array.from({ length: WALK }, (_, i) => 'Walk ' + (i + 1)));
  const GROUNDS = [
    { name: 'Grass', top: '#5c7a3e', bot: '#4a6634', dot: '#6c8a48' },
    { name: 'Trail', top: '#6a4d33', bot: '#563d29', dot: '#7a5c3e' },
    { name: 'Stone', top: '#8a8a80', bot: '#74746b', dot: '#9a9a90' },
    { name: 'Night', top: '#1f2a24', bot: '#141c18', dot: '#2b3a31' },
    { name: 'Checker', checker: true }
  ];

  // Every sheet is pre-painted (scripts/build-sprites.mjs) and loaded here up front: one per character and skin
  // tone, and per hairstyle for characters with hair, so switching style or skin is just a swap of images.
  // (c.sheets is keyed by hairstyle and skin tone: see sheetKey)
  const sheetKey = (hair, skin) => `${hair}|${skin}`;
  await Promise.all(CAST.map(async (c) => {
    c.hair = c.spec.hairStyle;
    c.skin = c.spec.skinTone || SKINS[0].key;
    c.sheets = {};
    const styles = c.hair ? HAIRSTYLES.map((h) => h.key) : [null];
    await Promise.all(styles.flatMap((hair) => SKINS.map(async ({ key: skin }) => {
      c.sheets[sheetKey(hair, skin)] = await loadSheet(c.key, hair, { ...c.spec, ...(hair ? { hairStyle: hair } : {}), skinTone: skin });
    })));
    c.sheet = c.sheets[sheetKey(c.hair || null, c.skin)];
  }));

  const state = { who: 0, facing: 0, anim: 'idle', frame: 0, playing: true, zoom: 3, ground: 0, turntable: false };
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduceMotion) state.playing = false;

  const $ = (id) => document.getElementById(id);
  const stage = $('stage'), stageCanvas = $('stageCanvas'), sctx = stageCanvas.getContext('2d');

  function blit(ctx, who, row, col, dx, dy, scale) {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(CAST[who].sheet, col * PW, row * PH, PW, PH, dx, dy, PW * scale, PH * scale);
  }
  function paintGround(ctx, w, h, g, unit) {
    if (g.checker) {
      const s = Math.max(8, unit * 4);
      for (let y = 0; y < h; y += s) for (let x = 0; x < w; x += s) { ctx.fillStyle = ((x + y) / s) % 2 ? '#c9c6bc' : '#e2dfd5'; ctx.fillRect(x, y, s, s); }
      return;
    }
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, g.top); grad.addColorStop(1, g.bot);
    ctx.fillStyle = grad; ctx.fillRect(0, 0, w, h);
    // a few pixel flecks so the ground has texture at the sprite's scale
    ctx.fillStyle = g.dot;
    let seed = 7;
    for (let i = 0; i < 220; i++) {
      seed = (seed * 9301 + 49297) % 233280;
      const x = Math.floor(seed / 233280 * w / unit) * unit;
      seed = (seed * 9301 + 49297) % 233280;
      const y = Math.floor(seed / 233280 * h / unit) * unit;
      ctx.fillRect(x, y, unit, unit);
    }
  }

  function frameCol() { return state.anim === 'idle' ? state.frame % 2 : 2 + (state.frame % WALK); }

  function drawStage() {
    const r = stage.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
    if (stageCanvas.width !== w || stageCanvas.height !== h) { stageCanvas.width = w; stageCanvas.height = h; }
    // zoom is the size of one sprite pixel on screen
    const scale = Math.max(1, Math.round(state.zoom * dpr)), unit = scale * RES;
    paintGround(sctx, w, h, GROUNDS[state.ground], unit);
    const sw = PW * scale, shh = PH * scale;
    const dx = Math.round((w - sw) / 2), dy = Math.round(h * 0.88 - shh);
    // soft contact shadow, as in the game
    sctx.fillStyle = 'rgba(0, 0, 0, 0.28)';
    sctx.beginPath();
    sctx.ellipse(w / 2, dy + 76 * unit, 13 * unit, 3.6 * unit, 0, 0, Math.PI * 2);
    sctx.fill();
    blit(sctx, state.who, state.facing, frameCol(), dx, dy, scale);
    const who = CAST[state.who], hairName = who.hair ? ', ' + HAIRSTYLES.find((h) => h.key === who.hair).name.toLowerCase() + ' hair' : '';
    $('caption').textContent = `${who.name}${hairName}, ${FACINGS[state.facing].toLowerCase()}, ${FRAMES[frameCol()].toLowerCase()}`;
  }

  // cast list with small live previews
  const castEl = $('cast'), castCanvases = [];
  CAST.forEach((c, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('aria-pressed', String(i === state.who));
    const cv = document.createElement('canvas'); cv.width = PW; cv.height = PH;
    const t = document.createElement('span');
    t.innerHTML = `${c.name}<small>${c.note}</small>`;
    b.append(cv, t);
    b.addEventListener('click', () => { state.who = i; render(); });
    castEl.append(b);
    castCanvases.push(cv);
  });

  // hairstyle picker: one button per style, each with a head-and-shoulders thumbnail of that style
  const hairWrap = $('hairWrap'), hairsEl = $('hairs'), hairButtons = [];
  const THUMB = { x: 36, y: 20, w: 56, h: 56 };   // crop of the front idle frame, in fine pixels
  HAIRSTYLES.forEach((st) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.title = st.note;
    const cv = document.createElement('canvas'); cv.width = THUMB.w; cv.height = THUMB.h;
    b.append(cv, document.createTextNode(st.name));
    b.addEventListener('click', () => pickHair(st.key));
    hairsEl.append(b);
    hairButtons.push({ b, cv, key: st.key });
  });
  // skin tone picker: one button per tone, each with a head-and-shoulders thumbnail in the current hairstyle
  const skinsEl = $('skins'), skinButtons = [];
  SKINS.forEach((t) => {
    const b = document.createElement('button');
    b.type = 'button';
    const cv = document.createElement('canvas'); cv.width = THUMB.w; cv.height = THUMB.h;
    b.append(cv, document.createTextNode(t.name));
    b.addEventListener('click', () => pickSkin(t.key));
    skinsEl.append(b);
    skinButtons.push({ b, cv, key: t.key });
  });
  // the thumbnails are crops of the loaded sheets (front, idle), redrawn when the character, hairstyle or skin
  // tone changes: the hairstyles in the current skin tone, the skin tones in the current hairstyle
  let thumbsFor = null;
  const thumb = (cv, img) => {
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, THUMB.w, THUMB.h);
    ctx.drawImage(img, THUMB.x, THUMB.y, THUMB.w, THUMB.h, 0, 0, THUMB.w, THUMB.h);
  };
  function paintThumbs() {
    const c = CAST[state.who], key = c.key + sheetKey(c.hair, c.skin);
    if (thumbsFor === key) return;
    thumbsFor = key;
    if (c.hair) hairButtons.forEach(({ cv, key: hair }) => thumb(cv, c.sheets[sheetKey(hair, c.skin)]));
    skinButtons.forEach(({ cv, key: skin }) => thumb(cv, c.sheets[sheetKey(c.hair || null, skin)]));
  }
  function pickHair(key) {
    const c = CAST[state.who];
    if (!c.hair || c.hair === key) return;
    c.hair = key; c.sheet = c.sheets[sheetKey(key, c.skin)];
    party.hair[c.key] = key; saveParty(party);
    render(); paintCast();
  }
  function pickSkin(key) {
    const c = CAST[state.who];
    if (c.skin === key) return;
    c.skin = key; c.sheet = c.sheets[sheetKey(c.hair || null, key)];
    party.skin[c.key] = key; saveParty(party);
    render(); paintCast();
  }

  // party picker: which characters walk in the world. At least one must stay in; the first in the list leads.
  const partyEl = $('party'), partyButtons = CAST.map((c) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = c.name;
    b.addEventListener('click', () => {
      const i = party.members.indexOf(c.key);
      if (i >= 0) { if (party.members.length > 1) party.members.splice(i, 1); }
      else party.members = CAST.map((m) => m.key).filter((k) => k === c.key || party.members.includes(k));
      saveParty(party);
      render();
    });
    partyEl.append(b);
    return { b, key: c.key };
  });

  // facing dial: buttons placed around a circle, with the camera at the bottom
  const dial = $('dial'), dialButtons = [];
  FACINGS.forEach((name, k) => {
    // facing k turns the character k*45° from facing the camera; place it where they'd be looking
    const ang = Math.PI / 2 - k * Math.PI / 4;
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = name.replace('Front ', 'F ').replace('Back ', 'B ').replace('right', 'R').replace('left', 'L');
    b.title = name;
    b.setAttribute('aria-label', name);
    b.style.left = (50 + Math.cos(ang) * 40) + '%';
    b.style.top = (50 + Math.sin(ang) * 40) + '%';
    b.addEventListener('click', () => { state.facing = k; state.turntable = false; render(); });
    dial.append(b);
    dialButtons.push(b);
  });

  const groundsEl = $('grounds'), groundButtons = [];
  GROUNDS.forEach((g, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.title = g.name;
    b.setAttribute('aria-label', g.name + ' ground');
    b.style.background = g.checker ? 'repeating-conic-gradient(#c9c6bc 0% 25%, #e2dfd5 0% 50%) 0 0 / 12px 12px' : g.top;
    b.addEventListener('click', () => { state.ground = i; render(); });
    groundsEl.append(b);
    groundButtons.push(b);
  });

  // full sheet grid
  const grid = $('grid'), cells = [];
  function buildGrid() {
    grid.innerHTML = '';
    cells.length = 0;
    grid.append(Object.assign(document.createElement('div'), { textContent: '' }));
    FRAMES.forEach((f) => grid.append(Object.assign(document.createElement('div'), { textContent: f })));
    FACINGS.forEach((name, row) => {
      grid.append(Object.assign(document.createElement('div'), { className: 'rowname', textContent: name }));
      FRAMES.forEach((f, col) => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'cell';
        b.setAttribute('aria-label', `${name}, ${f}`);
        const cv = document.createElement('canvas'); cv.width = PW; cv.height = PH;
        b.append(cv);
        b.addEventListener('click', () => {
          state.facing = row; state.turntable = false;
          if (col < 2) { state.anim = 'idle'; state.frame = col; } else { state.anim = 'walk'; state.frame = col - 2; }
          state.playing = false;
          render();
        });
        grid.append(b);
        cells.push({ b, cv, row, col });
      });
    });
  }
  buildGrid();
  function paintGrid() {
    cells.forEach(({ b, cv, row, col }) => {
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, PW, PH);
      blit(ctx, state.who, row, col, 0, 0, 1);
      b.setAttribute('aria-current', String(row === state.facing && col === frameCol()));
    });
    $('sheetTitle').textContent = `Full sheet: ${CAST[state.who].name.toLowerCase()}, eight facings by ${FRAMES.length} frames`;
  }

  function render() {
    castEl.querySelectorAll('button').forEach((b, i) => b.setAttribute('aria-pressed', String(i === state.who)));
    const who = CAST[state.who];
    hairWrap.hidden = !who.hair;
    partyButtons.forEach(({ b, key }) => {
      const on = party.members.includes(key);
      b.setAttribute('aria-pressed', String(on));
      b.disabled = on && party.members.length === 1;
    });
    $('partyNote').textContent = `${CAST.find((c) => c.key === party.members[0]).name} leads in the world.`;
    paintThumbs();
    if (who.hair) hairButtons.forEach(({ b, key }) => b.setAttribute('aria-pressed', String(key === who.hair)));
    skinButtons.forEach(({ b, key }) => b.setAttribute('aria-pressed', String(key === who.skin)));
    dialButtons.forEach((b, k) => b.setAttribute('aria-pressed', String(k === state.facing)));
    groundButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === state.ground)));
    $('animIdle').setAttribute('aria-pressed', String(state.anim === 'idle'));
    $('animWalk').setAttribute('aria-pressed', String(state.anim === 'walk'));
    $('play').setAttribute('aria-pressed', String(state.playing));
    $('play').textContent = state.playing ? 'Pause' : 'Play';
    $('turntable').setAttribute('aria-pressed', String(state.turntable));
    $('zoomOut').textContent = state.zoom + '×';
    drawStage();
    paintGrid();
  }
  function paintCast() {
    castCanvases.forEach((cv, i) => {
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, PW, PH);
      blit(ctx, i, state.facing, frameCol(), 0, 0, 1);
    });
  }

  $('animIdle').addEventListener('click', () => { state.anim = 'idle'; state.frame = 0; render(); });
  $('animWalk').addEventListener('click', () => { state.anim = 'walk'; state.frame = 0; render(); });
  $('play').addEventListener('click', () => { state.playing = !state.playing; render(); });
  $('step').addEventListener('click', () => { state.playing = false; state.frame++; render(); paintCast(); });
  $('zoom').addEventListener('input', (e) => { state.zoom = +e.target.value; render(); });
  $('turntable').addEventListener('click', () => { state.turntable = !state.turntable; render(); });

  // drag on the stage to turn the character
  let dragX = null, dragStart = 0;
  stage.addEventListener('pointerdown', (e) => { dragX = e.clientX; dragStart = state.facing; stage.setPointerCapture(e.pointerId); });
  stage.addEventListener('pointermove', (e) => {
    if (dragX == null) return;
    const steps = Math.round((e.clientX - dragX) / 40);
    const f = ((dragStart - steps) % 8 + 8) % 8;
    if (f !== state.facing) { state.facing = f; state.turntable = false; render(); }
  });
  const endDrag = () => { dragX = null; };
  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);
  stage.tabIndex = 0;
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLElement && (e.target.tagName === 'INPUT')) return;
    if (e.key === 'ArrowLeft') { state.facing = (state.facing + 1) % 8; render(); e.preventDefault(); }
    if (e.key === 'ArrowRight') { state.facing = (state.facing + 7) % 8; render(); e.preventDefault(); }
    if (e.key === ' ' && !(e.target instanceof HTMLButtonElement)) { state.playing = !state.playing; render(); e.preventDefault(); }
  });
  new ResizeObserver(drawStage).observe(stage);

  // animation clock: idle breathes slowly, the walk steps faster, the turntable turns every second
  let last = performance.now(), acc = 0, turnAcc = 0;
  function tick(now) {
    const dt = now - last; last = now;
    if (state.playing) {
      acc += dt;
      const step = state.anim === 'idle' ? 650 : 95;
      if (acc >= step) { acc = 0; state.frame++; drawStage(); paintCast(); paintGrid(); }
    }
    if (state.turntable && !reduceMotion) {
      turnAcc += dt;
      if (turnAcc >= 900) { turnAcc = 0; state.facing = (state.facing + 1) % 8; render(); }
    }
    requestAnimationFrame(tick);
  }
  render();
  paintCast();
  requestAnimationFrame(tick);
}
main();
