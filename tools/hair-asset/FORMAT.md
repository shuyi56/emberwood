# Emberwood hair asset (`emberwood.hair/1`)

One file, `<style>.hair.glb`, holds a Face Creator hairstyle completely: every piece of hair as the creator draws
it, the creator's materials with the numbers of its cut-out shader, its three textures, the lock guides, and the
creator's own viewer (lights, environment, tone mapping, reference cameras). Any renderer that follows this page
draws the same hair as the creator, with nothing re-tuned by hand. Today there are two readers:

- the creator itself, which the file is exported from (`tools/hair-asset/export.cjs`), and
- Blender (`blender/shared/hair_asset.py`), checked against the creator by `blender/shared/hair_bench.py`.

The file is a valid glTF 2.0 binary. A generic glTF viewer shows the cap and the veils (the plain meshes) with a
plain alpha mask. The strands are stored in the compact form described below, so only an Emberwood reader shows
them.

## Export

```
NODE_PATH=/opt/node22/lib/node_modules node tools/hair-asset/export.cjs face-creator.html wavyFull \
    blender/warden/warden.hair.glb --face Warden --probe [--views front,three,profile,back3,back] [--head head.glb]
```

The exporter opens the creator in headless Chromium, selects the face preset, rebuilds the style fresh (never from
the creator's cache), and reads back everything the creator put in the scene for it. Re-export whenever the
creator's hairstyle changes. The Blender side has nothing to edit.

The export stops instead of writing a wrong file when:

- the creator's cut-out shader has changed shape (`cut-shader-1.glsl` is its skeleton with every number taken
  out), or has a different count of numbers;
- (per tube) the rings below do not rebuild the creator's own mesh within 1e-4 (positions, colours, texture
  coordinates, `along`) and 1e-3 for normals. Such a piece is written as a plain mesh, so the file stays exact.
  The rebuild is checked in float32, as stored.

## Frame and units

- Model units: the creator's head is 8.6 from chin to crown. The origin is at the chin, +Y is up and +Z is the
  front (the creator's world space).
- Blender (the base figure's head frame, x across, y back, z up): `(x, y, z) = (X, -Z, Y) * HEAD_H / 8.6`.
- `TEXCOORD_0` follows glTF (v runs down the image, `1 - v` of three.js). The ring data below uses the same
  convention.

## The extras block

`json.extras.emberwood_hair`:

| key | what |
| --- | --- |
| `schema` | `emberwood.hair/1` |
| `style`, `name`, `face`, `hairTone` | the creator's style key, its display name, the face preset and its hair tone |
| `source` | creator file name, the first 16 hex digits of its sha256, three.js revision, export time |
| `units`, `frame` | as above |
| `textures` | glTF texture indices of `hair` (sRGB strand colours), `bump` (linear height), `tips` (linear: the tip map) |
| `pieces` | one entry per mesh the creator draws, in the creator's order (see Pieces) |
| `rings` | the tubes' ring accessors (see Strands) |
| `guides`, `guideLines` | the lock guides the creator built the clumps along: `{seed, first, count}` into a VEC3 accessor |
| `depthShader` | the creator's shadow (depth) material shader text |
| `viewer` | the creator's renderer: lights, tone mapping, exposure, environment (see Viewer) |
| `cameras` | the reference cameras of `--views`: position, target, `fovY` 20, aspect 1 |

## Materials

Each glTF material has `extras.emberwood`, the creator's material:

- `kind`:
  - `cut` is the hair (MeshStandardMaterial plus the tip map and an alpha test);
  - `veil` is the sheer veils under the clumps (MeshBasicMaterial, vertex colour with alpha, SrcAlpha /
    OneMinusSrcAlpha, no depth write, drawn before the hair);
  - `shell` and `standard` cover anything else.
- `color`, `opacity`, `roughness`, `metalness`, `envMapIntensity`, `bumpScale`, `side`, `alphaTest`,
  `transparent`, `depthWrite`, `vertexColors`, `blend`, `maps`: as the three.js material has them.
- `shader`: the text its `onBeforeCompile` injects, and its uniforms, for reference.
- `model` and `params` (cut only): `emberwood-cut-1` and its 30 numbers, plus `hairline`.

### The cut model (`emberwood-cut-1`)

Per fragment the shader takes:

- `u`, the tip-map u (the mesh's texture u);
- `a`, the vertex colour's alpha: how far into its cut-out the hair is, 0 at a strand's end, 1 where it is whole;
- `tipPx`, the tip-map texels a screen pixel covers: `max(fwidth(u) * tipW, fwidth(a) * tipH)`.

It works out:

```
far   = smoothstep(farPx0, farPx1, tipPx)
gain  = gainNum / mix(gainNear, gainFar, far)                  # 1 up close, 2.5 far off
tIn   = (inOne - smoothstep(in0, in1, a)) * (farOne - inFar * far)
s1 = tips(u, a);  s2 = tips(u + du2, max(a - dv2, dv2min));  s3 = tips(u + du3, max(a - dv3, dv3min))
colour *= shade
A1 = s1.g * gain;  A2 = s2.b * gain;  A3 = s3.b * gain         # the top layer, then two under it
if hairline > hairlineOn: A1 = tipsSoft(u, a).b * hairlineGain; A2 = hairlineA2; A3 = hairlineA3
if A1 >= cut1:       alpha = A1, colour *= mix(toneOne, s1.r * toneGain, tIn)
elif A2 >= cut2:     alpha = A2, colour *= tone2
else:                alpha = A3, colour *= tone3
discard if alpha < alphaTest
```

`tips` repeats round (u) and clamps up and down (v). `tipsSoft` is the same map read at mip bias 4, which the
Blender reader stands in for with a 16x box-filtered copy. Shadows are cast through the depth shader:
`max(s1.g, s2.b, s3.b) * 1.7 >= alphaTest`.

A renderer without screen derivatives sets `tipPx` per camera: `tip_texels_per_unit x` the size of a pixel on the
hair. See `hair_asset.set_view` and `set_pixel`. three.js's bump (r151 and later) is likewise a height of
`bumpScale` per screen pixel, so a fixed-distance bump has to use `bumpScale x` the pixel size.

## Pieces

`pieces[i]`:

| key | what |
| --- | --- |
| `role` | `cap` (the opaque head of hair, a grid), `strand`, `lock-strand` (a strand of a clump), `veil`, `piece` |
| `material` | glTF material index |
| `renderOrder`, `castShadow`, `receiveShadow` | as the creator's mesh. Every piece has receiveShadow false; only the cap casts |
| `len`, `tipLen`, `rootLen`, `lock`, `layer`, `tipFrac`, `thin` | the creator's own data on the piece. `layer` is 0 for the top layer and 1 for the one under it |
| `grid` | the cap: `{rows, cols}` |
| `firstVertex`, `vertexCount`, `firstIndex`, `indexCount` | a plain mesh piece: its range in its material's mesh |
| `tube` | a strand stored as rings (see Strands) |
| `rawReason` | why a tube was stored as a plain mesh instead |

Plain-mesh pieces are packed into one glTF mesh per material. Attributes:

- `POSITION`, `NORMAL` (the creator's own normals);
- `TEXCOORD_0`;
- `COLOR_0` (VEC4: rgb the vertex tone, a the cut-out coordinate);
- `_ALONG` (distance along the strand from its root, -1 off the strands);
- `_PIECE` (the piece index, as float, since glTF wants 4-byte vertex attributes).

## Strands (tubes)

The creator builds each strand with `tube()`. A tube has R rings of S+1 points, where the last point repeats the
first so the texture closes round. Two end points follow, closing each end on its centre. A tube is stored as its
rings: `rings` holds five accessors with one row per ring, and the tube's rows are `firstRing .. firstRing + R - 1`.

| accessor | type | per ring |
| --- | --- | --- |
| `centre` | VEC3 | the ring's centre c |
| `axisA`, `axisB` | VEC3 | its two semi-axes (the section's size and turn, end rounding included) |
| `color` | VEC4 | the ring's tone rgb (before the round shading) and its alpha (the cut-out coordinate) |
| `tex` | VEC3 | `uSpan` (u once round), `v` (glTF convention), `along` |

`tube` holds `ringSize` (S+1), `rings` (R), `firstRing`, `u0` (u where the ring starts), `endColor` (the two end
points' rgba) and `endAlong`.

Rebuild, with `w = 2π s / S` for `s = 0..S`:

```
position = c + cos(w) A + sin(w) B
colour   = rgb x (0.9 + 0.1 sin w), alpha as stored
uv       = (u0 + uSpan x s / S, v);  along as stored
ends     = (c of ring 0 | ring R-1, endColor[e], (u0, v of that ring), endAlong[e])
```

Vertex order: ring by ring, then the root end point, then the tip end point. Triangles, in this order:

- for each ring `i < R-1` and each `s < S`, with `a = i(S+1) + s` and `b = a + 1`: `(a, b, a+S+1)` and
  `(b, b+S+1, a+S+1)`;
- the root end: `(s, s+1, root)`;
- the tip end: `(a, tip, a+1)`, with `a = (R-1)(S+1) + s`.

Normals as three.js's `computeVertexNormals`: each triangle `(a, b, c)` adds `(c - b) x (a - b)` to its three
vertices, then each vertex's sum is normalised. The seam column stays unwelded, as in the creator.

The ring centres are the strand's centre line. A figure moves its hair (the clearance off the clothes) by moving
ring centres, which keeps every section intact (`hair_asset.points`).

This form is 16 floats per ring instead of 15 vertices of 14 floats. The Warden's 771 strands go from 64 MB of
baked vertices to 13 MB in total, textures and environment included.

## Viewer

`viewer`, so a renderer can be checked against the creator:

- `renderer`, `toneMapping` (`aces-filmic-three`: three.js's ACES fit), `exposure`, `output: srgb`.
- `lights`: type, linear colour, intensity (three.js r155+ units: a white matte surface returns intensity / π, as
  a Cycles sun of that strength), `toLight` and `castShadow` for directional lights, and `groundColor` for the
  hemisphere light.
- With `--probe`, three maps, all equirectangular in Blender's layout and linear:
  - `environmentMap`: the RoomEnvironment's radiance;
  - `environmentResponse`: what three.js's shading takes from the environment for a matte white surface, by
    normal;
  - `environmentSpecular`: the same for a matte black surface.

  A surface of colour c gets `c x (white - black) + black`, at its normal and unoccluded, times its
  `envMapIntensity`.

## Choices made

- **glTF binary as the container.** It is one file, has a public spec, and opens in any viewer for the plain
  meshes. Blender reads it with nothing but numpy.
- **Geometry as the creator draws it, not a recipe.** The creator's strands come out of its own random seeds and
  solvers. Storing the result makes every reader match it, with no second implementation of the solver to drift.
  The rings are exact: they are the creator's own sections, not a fit, and the export proves it on every piece.
- **The cut-out shader as a named model with numbers.** Readers implement the model once. A change to the
  creator's shader stops the export until FORMAT.md, both readers and the skeleton are updated (then bump to
  `emberwood-cut-2`).
- **The viewer travels with the hair.** A renderer that matches the creator's own lighting shows the hair
  identically. A figure lights it its own way: `tone` and the views' pixel size are the only figure-side settings.
- **No Blender-only fudges in the file.** The old Warden import scaled the colour by 0.45 and fixed the tip gain
  at 1.4. Those are figure-view settings now (`HAIR_ONSCREEN` in `warden.py`, and the pixel size per view), not
  part of the hair.
