# World capture

Frames of the party leader walking in the real world page (`index.html`), for judging motion as it appears in the
game. Nothing is rendered in Blender: this plays whatever `public/models/ranger.glb` (or sprite sheet) the branch has.

    npm run dev
    node tools/world-capture/capture.mjs --out /tmp/walk --views side,front3q,back3q --frames 30 --fps 60

| Flag | Default | What it does |
| --- | --- | --- |
| `--views` | `side,front3q,back3q` | Walk direction against the camera: `front`, `front3q`, `side`, `back3q`, `back` |
| `--frames`, `--fps` | `24`, `60` | Frames per view, and the game time between them (1/fps s) |
| `--dist` | `9` | Camera distance, as the zoom (7 is closest, 34 the starting view) |
| `--member` | `ranger` | Who leads the party for the capture |
| `--size`, `--scale` | `300x340`, `2` | The clip around the leader in CSS pixels, and the device pixel ratio |
| `--near`, `--clear` | `-6,6`, `5` | Where to look for open ground, and how far every obstacle must stay from the walk |
| `--url` | `http://127.0.0.1:5173/` | The world page |

How it works: the page is opened with `?capture`, which exposes `window.__ember` (the "Capture hook" in
`src/world/main.js`). The page's `requestAnimationFrame` and `performance.now` are replaced, so the script steps game
time itself and the frames are evenly spaced however slow the headless software renderer is (about 2 to 10 s a
frame at scale 2). Only captured frames are drawn. The leader is placed on open ground, the camera is locked onto
her at the game's own tilt for `--dist`, and she walks a straight line.

Output: `<out>/<view>/fNNN.png` and `<out>/frames.json` (the walk phase of each frame).

Don't edit `src/world/main.js` while a capture runs against `npm run dev`: the reload ends the capture.
