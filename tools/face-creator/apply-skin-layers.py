"""Put the layered skin (skin-layers.js beside this file) into a Face Creator main.js, in place, at named anchors, so it
can be laid onto whatever main.js is current. Then fill its parameters from skin_layers.py:

    python3 tools/face-creator/apply-skin-layers.py main.js tools/face-creator/skin-layers.js
    node tools/face-creator/skin-params.mjs main.js

It stops, changing nothing, if an anchor is missing. If the port is already in, it replaces the module alone (the
parameters then need skin-params.mjs again)."""
import sys
path, mod = sys.argv[1], sys.argv[2]
s = open(path).read(); M = open(mod).read()
if '@@SKIN_LAYERS@@' in s:
    # already in: lay the module itself in again (from its first line to the face texture), leaving the rest
    i = s.find(M[:M.index('\n', 120)])
    j = s.find('\nvar faceTex = new THREE.CanvasTexture(faceCanvas);', i)
    if i < 0 or j < 0:
        sys.exit('already applied, and the module is not where it was put')
    s = s[:i] + M + s[j:]
    # (and the hooks an older port put in, brought up to this module)
    s = s.replace('head.onBeforeRender = function (r, s, cam) { skinU.skinBounceDir.value.copy(SKIN_BOUNCE_W).transformDirection(cam.matrixWorldInverse); };', 'head.onBeforeRender = function (r, s, cam) { skinU.skinBounceDir.value.copy(SKIN_BOUNCE_W).transformDirection(cam.matrixWorldInverse); skinBakeTick(head.geometry); };')
    open(path, 'w').write(s)
    sys.exit(print('module replaced') or 0)
def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit('anchor found %d times (want %d): %r' % (n, count, old[:90]))
    s = s.replace(old, new)
# settings and sliders
rep("Object.assign(DEFAULTS, { sex: 0,",
    "Object.assign(DEFAULTS, { skinLayers: 1, veins: 1, sunspots: 1, moles: 1 });      // (the layered skin: skin_layers.py, below)\nObject.assign(DEFAULTS, { sex: 0,")
rep("['oil', 'Oily sheen', 0, 2, 0.02]]]",
    "['oil', 'Oily sheen', 0, 2, 0.02]]],\n    ['Layers', 'Built as Blender builds it (skin_layers.py): a red and brown flesh under the skin, veins where the skin is thin, sun spots and moles, and light that spreads under the surface. 0 is the painted skin alone.', [['skinLayers', 'Layered skin', 0, 1, 0.02], ['veins', 'Veins', 0, 2, 0.02], ['sunspots', 'Sun spots', 0, 2, 0.02], ['moles', 'Moles', 0, 2, 0.02]]]")
# the module, once the renderer exists
rep("var faceTex = new THREE.CanvasTexture(faceCanvas);", M + "\nvar faceTex = new THREE.CanvasTexture(faceCanvas);")
rep("var eyeCanvases = [document.createElement('canvas'), document.createElement('canvas')];",
    "(function () { var obc = skinMat.onBeforeCompile; skinMat.onBeforeCompile = function (sh) { obc(sh); skinLayersShader(sh); }; })();      // (the layered skin over this creator's own skin shader)\nvar eyeCanvases = [document.createElement('canvas'), document.createElement('canvas')];")
rep("var head = new THREE.Mesh(new THREE.BufferGeometry(), skinMat); head.castShadow = true; head.receiveShadow = true; scene.add(head);",
    "var head = new THREE.Mesh(new THREE.BufferGeometry(), skinMat); head.castShadow = true; head.receiveShadow = true; scene.add(head);\n"
    "head.onBeforeRender = function (r, s, cam) { skinU.skinBounceDir.value.copy(SKIN_BOUNCE_W).transformDirection(cam.matrixWorldInverse); skinBakeTick(head.geometry); };      // (the bounce's direction in view space)")
# the head's lookups
rep("  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));\n  geo.setAttribute('sssMask', new THREE.BufferAttribute(sss, 1));",
    """  // the layered skin's lookups (bakeSkinLayers): each point's place in the figure's rest space (model axes, the chin at
  // HEAD_Z, as Blender's 'rest'), its place round the head (uv1 over the full turn), and where the layers show
  var sRest = new Float32Array(n * 3), sWrap = new Float32Array(n * 2), sZones = new Float32Array(n * 2);
  for (k = 0; k < n; k++) {
    sRest[k * 3] = pos[k * 3]; sRest[k * 3 + 1] = -pos[k * 3 + 2]; sRest[k * 3 + 2] = pos[k * 3 + 1] + SKIN_LAYERS.FIGURE.HEAD_Z;
    sWrap[k * 2] = (uv1[k * 2] - PORE_U0) / (PORE_U1 - PORE_U0); sWrap[k * 2 + 1] = uv1[k * 2 + 1];
    var szn = skinZonesAt(pos[k * 3] / HH, pos[k * 3 + 1] / HH, -nrm[k * 3 + 2], nrm[k * 3 + 1]); sZones[k * 2] = szn[0]; sZones[k * 2 + 1] = szn[1];
  }
  geo.setAttribute('skinRest', new THREE.BufferAttribute(sRest, 3)); geo.setAttribute('skinWrap', new THREE.BufferAttribute(sWrap, 2)); geo.setAttribute('skinZones', new THREE.BufferAttribute(sZones, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('sssMask', new THREE.BufferAttribute(sss, 1));""")
# paintFace: the mask beside the face
rep("  var F = FIELDS, d = faceImg.data, base = hex(P.skinTone), b0 = [207, 159, 132], rt = ratio(b0, base);",
    "  var F = FIELDS, d = faceImg.data, base = hex(P.skinTone), b0 = [207, 159, 132], rt = ratio(b0, base);\n  var MK = skinMaskImg.data;      // (the layered skin's mask: red, the freckles; green, what is not skin)")
rep("      var r0 = base[0], g0 = base[1], b0_ = base[2], t;", "      var r0 = base[0], g0 = base[1], b0_ = base[2], t, frT = 0;")
rep("        t = clamp(F.fleck[i] * some * Math.sqrt(fmask) * Math.min(1, 0.45 + 0.6 * P.freckles), 0, 1);",
    "        t = clamp(F.fleck[i] * some * Math.sqrt(fmask) * Math.min(1, 0.45 + 0.6 * P.freckles), 0, 1); frT = t;")
rep("      var o = i * 4; d[o] = r0; d[o + 1] = g0; d[o + 2] = b0_; d[o + 3] = 255;",
    "      var o = i * 4; d[o] = r0; d[o + 1] = g0; d[o + 2] = b0_; d[o + 3] = 255; MK[o] = frT * 255; MK[o + 1] = 0; MK[o + 2] = 0; MK[o + 3] = 255;")
rep("        var kw = WASH.wl * (WASH.up ? 0.75 : 0.9) * (1 - 0.7 * sm(0.75, 0.97, WASH.cr)), wc",
    "        MK[o + 1] = Math.max(MK[o + 1], 255 * WASH.wl);\n        var kw = WASH.wl * (WASH.up ? 0.75 : 0.9) * (1 - 0.7 * sm(0.75, 0.97, WASH.cr)), wc")
rep("      if (a > 0) { d[o] += (LIP.c[0] - d[o]) * a; d[o + 1] += (LIP.c[1] - d[o + 1]) * a; d[o + 2] += (LIP.c[2] - d[o + 2]) * a; }",
    "      if (a > 0) { d[o] += (LIP.c[0] - d[o]) * a; d[o + 1] += (LIP.c[1] - d[o + 1]) * a; d[o + 2] += (LIP.c[2] - d[o + 2]) * a; MK[o + 1] = Math.max(MK[o + 1], 255 * a); }")
rep("  fctx.drawImage(lashLayer, 0, 0);\n", "  fctx.drawImage(lashLayer, 0, 0);\n  skinMaskCtx.putImageData(skinMaskImg, 0, 0); skinMaskOver(browLayer); skinMaskOver(lashLayer);\n")
# update(): bake per head shape, cache the mask with the face, the material
rep("    head.geometry = M.geo; state.eyes = M.eyes; state.meshKey = mk; state.geoVersion = M.gv; LID_CACHE = M.lid;",
    "    head.geometry = M.geo; state.eyes = M.eyes; state.meshKey = mk; state.geoVersion = M.gv; LID_CACHE = M.lid;\n    bakeSkinLayers(M.geo, M.gv);")
rep("    if (Fc) { fctx.putImageData(Fc.face, 0, 0); roughCanvas.getContext('2d').putImageData(Fc.rough, 0, 0); }",
    "    if (Fc) { fctx.putImageData(Fc.face, 0, 0); roughCanvas.getContext('2d').putImageData(Fc.rough, 0, 0); skinMaskCtx.putImageData(Fc.mask, 0, 0); }")
rep("rough: roughCanvas.getContext('2d').getImageData(0, 0, roughCanvas.width, roughCanvas.height) });",
    "rough: roughCanvas.getContext('2d').getImageData(0, 0, roughCanvas.width, roughCanvas.height), mask: skinMaskCtx.getImageData(0, 0, TW, TH) });")
rep("    faceTex.needsUpdate = true; roughTex.needsUpdate = true; state.faceKey = fk;",
    "    faceTex.needsUpdate = true; roughTex.needsUpdate = true; skinMaskTex.needsUpdate = true; state.faceKey = fk;")
rep("  skinMat.sheen = 0.35 * P.sss * (1 - 0.5 * fairM); skinMat.sheenColor.setRGB(1.0, 0.42, 0.32);",
    """  skinMat.sheen = 0.35 * P.sss * (1 - 0.5 * fairM); skinMat.sheenColor.setRGB(1.0, 0.42, 0.32);
  // the layered skin (skin_layers.py's surface): its reflection, oil film and down in place of the sheen that stood in
  // for the scatter, which the wrapped light now carries (red reaching furthest, by Blender's radius)
  var SL = SKIN_LAYERS, lay = clamp(P.skinLayers, 0, 1);
  skinU.skinLayers.value = lay; skinU.skinVeins.value = P.veins; skinU.skinSpots.value = P.sunspots; skinU.skinMoles.value = P.moles; skinU.skinPores.value = Math.min(1, P.texture);
  skinU.skinWrapRGB.value.set(1, 0.36, 0.18).multiplyScalar(SL.SSS_SCALE / SKIN_WRAP_RADIUS * P.sss * lay);
  var bk = SKIN_BOUNCE_W.share * key.intensity * lay; skinU.skinBounceCol.value.set(SKIN_BOUNCE_W.col[0] * bk, SKIN_BOUNCE_W.col[1] * bk, SKIN_BOUNCE_W.col[2] * bk);
  skinMat.specularIntensity = lerp(0.55, 2 * SL.SPECULAR, lay);
  skinMat.clearcoat = SL.COAT[0] * lay; skinMat.clearcoatRoughness = SL.COAT[1];
  skinMat.sheen = lerp(skinMat.sheen, SL.SHEEN[0], lay); skinMat.sheenRoughness = lerp(0.55, SL.SHEEN[1], lay);
  skinMat.sheenColor.setRGB(lerp(1.0, 1, lay), lerp(0.42, 1, lay), lerp(0.32, 1, lay));""")
rep("window.__fc = { HAIR_STYLES", "window.__skin = { THREE: THREE, U: skinU, bounce: SKIN_BOUNCE_W, B: SKIN_BAKE, head: head, eyes: eyeMeshes, ears: earMeshes, camera: camera, scene: scene, face: faceCanvas, mask: skinMaskCanvas, lights: { key: key, rim: rim, fill: fill, hemi: hemi } };      // (for the Blender bench: tools/face-creator/dump-head.cjs)\nwindow.__fc = { HAIR_STYLES")
open(path, 'w').write(s)
print('applied')
