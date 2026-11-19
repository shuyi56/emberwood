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

## Hair from the Face Creator

A Face Creator hairstyle travels to Blender as one file, `<style>.hair.glb`. It holds the creator's geometry
(strands stored as their rings), its cut-out shader's numbers, its textures and its viewer. The spec is in
[`tools/hair-asset/FORMAT.md`](tools/hair-asset/FORMAT.md).

```sh
# export (headless Chromium): the Warden's hair, with the creator's environment and five reference views
NODE_PATH=/opt/node22/lib/node_modules node tools/hair-asset/export.cjs face-creator.html wavyFull \
    blender/warden/warden.hair.glb --face Warden --probe
# check Blender against the creator: same cameras, lights and tone mapping, with difference maps
python3 blender/shared/hair_bench.py ASSET.hair.glb --head head.glb --ref REFS --out OUT --crops
```

`blender/warden/warden.py` builds her hair from `warden.hair.glb`. To change the hair, change it in the creator
and re-export.
