# Emberwood

A 3D forest valley with a party camera, built with three.js and custom GLSL shaders. Every texture is
generated in code, and the three party members are pixel-art sprites painted procedurally.

Two pages share one sprite painter:

| Page | Entry | What it is |
| --- | --- | --- |
| World | `index.html` | Click the ground to send the party there; drag, scroll, `Q`/`E`, `WASD` and `Space` control the camera. |
| Sprite bench | `bench.html` | Inspect each sprite, facing, animation and ground at zoom. |

## Develop

```sh
npm install
npm run dev      # http://localhost:5173 (bench at /bench.html)
npm run build    # outputs to dist/
npm run lint
```

## Layout

```
index.html, bench.html   page shells
src/shared/sprites.js    sprite painter, palettes, character specs (used by both pages)
src/world/main.js        terrain, scenery, navigation, party, camera, input, loop
src/world/shaders.js     GLSL chunks shared by all materials
src/world/styles.css
src/bench/main.js        bench UI
src/bench/styles.css
```

Notes: three.js is pinned to r128 (the version the project was written against).
