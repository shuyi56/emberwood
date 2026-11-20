"""Put the layered lips into the Face Creator's main.js, as targeted edits, and rebuild face-creator.html from it.

    python3 tools/face-creator/apply-lip-layers.py /mnt/project-files/face-creator/main.js [face-creator.html]

Each edit is anchored on a line of the live file and checked: if an anchor is missing (someone changed that line), the
script stops and writes nothing. Run on a file that already has the lips, it changes nothing. The html is rebuilt by
replacing its copy of main.js (the end of its module script) with the new main.js.

The lips (John, 2026-11-20: "multi layered, a smoother pink under layer and a more rough layer on top"):
  under layer  the vascular tissue seen through a thin epithelium: smooth, pinker and more saturated, scattering
               redder (green and blue absorbed sooner by the blood), with only a trace of the furrows in its relief
               and a low, smooth specular
  top layer    the thin cornified layer: a second specular lobe (three.js clearcoat) with its own normal carrying the
               vertical lip lines at full depth and the dry flakes, rough and dry toward the vermilion border, smooth
               and wet toward the parting, whitening a little where it is dry
"""
import sys

MARK = '/* @@LIP_LAYERS@@ */'

EDITS = [
    # 1. the sliders' defaults
    ("lipTone: '#b26860', lipLine: 1, lipstick: '#9c2f3a', lipstickAmt: 0, lipGloss: 0,",
     "lipTone: '#b26860', lipLine: 1, lipstick: '#9c2f3a', lipstickAmt: 0, lipGloss: 0, lipLines: 1, lipMoist: 1, lipLayers: 1,"),
    # 2. the measured lips and the two layers' numbers, before the lips' colour
    ("/* ---- the lips' colour (lip_painting), per pixel; returns colour and alpha into LIP ---- */",
     MARK + """
/* ---- the lips in two layers (John 2026-11-20: "a smoother pink under layer and a more rough layer on top") ----
   What a lip is: the red of the lip (the vermilion) is a thin, lightly keratinised epithelium, a few cell layers of
   stratum corneum against the dozen or more of the face's skin, with no hair, no sweat glands and almost no sebum,
   over tall dermal papillae whose capillary loops come close to the surface and almost no melanin. So it is red
   because the blood shows through it, it holds little water (its loss of water through the surface is several times
   the cheek's, Kobayashi and Tagami 2004), and it is wettest where the tongue and the inner mucosa keep it wet, at
   the parting, drying toward the border. Its surface is cut by the fine grooves of the lip print, mostly vertical in
   women (Tsuchihashi type I), and where it dries the cornified layer lifts in fine flakes.
   How production shaders draw it: a lip mask over the one skin shader (NVIDIA's digital human: the skin's micro
   normal taken off the lips, more specular, a coat added for the wet with its roughness lowered; Unreal's digital
   humans: the most scatter on the lips, its own boundary bleed at the mouth), and a dual lobe. So here, under the
   lips' own mask:
   - the under layer: the painted lip colour, a little pinker and more saturated (the blood under a thin epithelium),
     scattered redder than the skin (LIP_LAYERS.WRAP scales the skin's wrapped light per channel: red as the skin,
     green absorbed soonest, blue a little less, so the shadowed edge stays rose, not orange), its relief keeping only BASE_FURROW of the grooves, and a smooth, low specular
     (BASE_ROUGH, dry to wet);
   - the top layer: three.js's clearcoat, as a second lobe with its own normal. It carries the grooves at full depth
     and the dry flakes, its roughness from COAT_ROUGH (dry at the border, wet at the parting) and its weight from
     COAT; where it is dry it whitens the colour a little (DRY_VEIL), as dry, flaking lips look paler at their edge.
   Measured from the head: HEAD_MM is the head's height (HH model units of 27 mm, skin_layers' scale), so the mouth
   (0.164 heads) is about 38 mm and each lip's red about 9 mm high. The grooves are PITCH_MM apart at the parting
   (fanning wider toward the border), about 0.2 mm wide, and the flakes FLAKE_MM across. */
var LIP_LAYERS = { HEAD_MM: HH * 27, PITCH_MM: 1.1, FLAKE_MM: 0.45, BASE_FURROW: 0.35, BASE_ROUGH: [0.36, 0.24],
  COAT: [0.6, 0.9], COAT_ROUGH: [0.46, 0.22], WRAP: [1.0, 0.55, 0.65], ROSE: 0.06, SATURATE: 0.12, DRY_VEIL: 0.05, FLAKE: 0.12, COAT_BUMP: 1.6, BASE_SPEC: 0.35 };
var LIP_PITCH = LIP_LAYERS.PITCH_MM / LIP_LAYERS.HEAD_MM, LIP_FLAKE = LIP_LAYERS.FLAKE_MM / LIP_LAYERS.HEAD_MM;      // (in head heights)
var LIPMAP = null;      // (the lips' own map beside the face (its planar uv): red, the lip; green, how wet; blue, the top layer's relief: grooves and flakes)
function lipMapCanvas() {
  if (!LIPMAP) { var c = document.createElement('canvas'); c.width = TW; c.height = TH; LIPMAP = { c: c, x: c.getContext('2d') }; LIPMAP.img = LIPMAP.x.createImageData(TW, TH); }
  return LIPMAP;
}
/* ---- the lips' colour (lip_painting), per pixel; returns colour and alpha into LIP ---- */"""),
    # 3. the grooves' spacing from the measured head
    ("var u = up ? 1 : 0, s = x / (1 + 0.45 * t), SP = 0.0052;",
     "var u = up ? 1 : 0, s = x / (1 + 0.45 * t), SP = LIP_PITCH;      // (LIP_LAYERS: 1.1 mm at the parting)"),
    # 4. the moisture slider, the flakes and the two layers' colour
    ("  var moist = Math.max(upper > lower ? 0.6 * sm(0.45, 0.1, tu) : sm(0.85, 0.35, tl), 0) * sm(0.075, 0.03, ax);\n",
     "  var moist = Math.max(upper > lower ? 0.6 * sm(0.45, 0.1, tu) : sm(0.85, 0.35, tl), 0) * sm(0.075, 0.03, ax);\n"
     "  moist = clamp(moist * (P.lipMoist == null ? 1 : P.lipMoist), 0, 1);\n"
     "  // (the two layers: the under layer pinker and more saturated, the blood under a thin epithelium; the top layer's dry\n"
     "  // flakes, toward the border where it is driest, and its pale veil where it is dry)\n"
     "  var LL = LIP_LAYERS, lay = P.lipLayers == null ? 1 : P.lipLayers, tt = upper > lower ? tu : tl, dry = 1 - moist;\n"
     "  var fn = noise2(X / LIP_FLAKE, w / (0.7 * LIP_FLAKE), 91) * 0.6 + noise2(X / (0.5 * LIP_FLAKE), w / (0.4 * LIP_FLAKE), 93) * 0.4;\n"
     "  var flake = lip > 0.01 ? sm(0.15, 0.55, fn) * sm(0.45, 0.95, tt) * dry * lay : 0;\n"
     "  if (lay > 0 && lip > 0.01) {\n"
     "    var gy = (col[0] + col[1] + col[2]) / 3, sat = 1 + LL.SATURATE * lay * (1 - P.lipstickAmt);\n"
     "    col = [gy + (col[0] - gy) * sat, gy + (col[1] - gy) * sat, gy + (col[2] - gy) * sat];\n"
     "    var rose = LL.ROSE * lay * (1 - P.lipstickAmt); col = [col[0], col[1] * (1 - 0.5 * rose), col[2] * (1 + rose)];      // (toward the rose of oxygenated blood: a little less green, a little more blue)\n"
     "    var veil = lay * (LL.DRY_VEIL * dry * sm(0.3, 1.0, tt) + 0.12 * flake) * (1 - 0.7 * P.lipstickAmt);\n"
     "    col = mix3(col, [Math.min(255, gy * 1.28 + 30), Math.min(255, gy * 1.18 + 22), Math.min(255, gy * 1.16 + 20)], clamp(veil, 0, 1));\n"
     "  }\n"),
    ("  LIP.c = col; LIP.a = clamp(a, 0, 1); LIP.lip = lip; LIP.groove = gF; LIP.moist = moist; LIP.edge = edge;",
     "  LIP.c = col; LIP.a = clamp(a, 0, 1); LIP.lip = lip; LIP.groove = gF; LIP.moist = moist; LIP.edge = edge; LIP.flake = flake;"),
    # 5. the under layer's relief keeps only a trace of the grooves (the top layer carries them)
    ("    var hh = lerp(L.base[q], 0.12 - 1.15 * LIP.groove, LIP.lip) + 0.28 * LIP.edge;",
     "    var hh = lerp(L.base[q], 0.12 - 1.15 * LIP.groove * lerp(1, LIP_LAYERS.BASE_FURROW, P.lipLayers == null ? 1 : P.lipLayers), LIP.lip) + 0.28 * LIP.edge;"),
    # 6. painting: the under layer's smooth specular, and the lips' own map
    ("  var lx0 = Math.floor(texX(-0.12 * P.mouthWidth * HH)), lx1 = Math.ceil(texX(0.12 * P.mouthWidth * HH)), ly0 = Math.floor(texY(0.32 * HH)), ly1 = Math.ceil(texY(0.16 * HH));\n",
     "  var lx0 = Math.floor(texX(-0.12 * P.mouthWidth * HH)), lx1 = Math.ceil(texX(0.12 * P.mouthWidth * HH)), ly0 = Math.floor(texY(0.32 * HH)), ly1 = Math.ceil(texY(0.16 * HH));\n"
     "  var LM = lipMapCanvas(), lmd = LM.img.data, LLy = P.lipLayers == null ? 1 : P.lipLayers; lmd.fill(0);\n"),
    ("      if ((xx & 3) === 0 && (yy & 3) === 0) { var ri = (yy >> 2) * RW + (xx >> 2); RGH[ri] = lerp(RGH[ri], 0.55 - 0.3 * LIP.moist - 0.4 * P.lipGloss - 0.1 * P.lipstickAmt, LIP.lip); }\n",
     "      var lr0 = 0.55 - 0.3 * LIP.moist - 0.4 * P.lipGloss - 0.1 * P.lipstickAmt, lr1 = lerp(LIP_LAYERS.BASE_ROUGH[0], LIP_LAYERS.BASE_ROUGH[1], LIP.moist) - 0.12 * P.lipGloss;\n"
     "      if ((xx & 3) === 0 && (yy & 3) === 0) { var ri = (yy >> 2) * RW + (xx >> 2); RGH[ri] = lerp(RGH[ri], lerp(lr0, lr1, LLy), LIP.lip); }\n"
     "      lmd[o] = 255 * LIP.lip; lmd[o + 1] = 255 * LIP.moist; lmd[o + 2] = 255 * clamp(LIP.groove * 0.8 * P.lipLines * (1 - 0.4 * P.lipstickAmt) + LIP_LAYERS.FLAKE * LIP.flake, 0, 1); lmd[o + 3] = 255;\n"),
    ("  fctx.putImageData(faceImg, 0, 0);\n  var bk = keyFor(",
     "  fctx.putImageData(faceImg, 0, 0); LM.x.putImageData(LM.img, 0, 0);\n  var bk = keyFor("),
    # 7. the face's cache keeps the lips' map with the face
    ("    if (Fc) { fctx.putImageData(Fc.face, 0, 0); roughCanvas.getContext('2d').putImageData(Fc.rough, 0, 0); skinMaskCtx.putImageData(Fc.mask, 0, 0); }",
     "    if (Fc && !Fc.lip) Fc = null;      // (a face kept from before the layered lips: paint it again)\n"
     "    if (Fc) { fctx.putImageData(Fc.face, 0, 0); roughCanvas.getContext('2d').putImageData(Fc.rough, 0, 0); skinMaskCtx.putImageData(Fc.mask, 0, 0); lipMapCanvas().x.putImageData(Fc.lip, 0, 0); }"),
    ("mask: skinMaskCtx.getImageData(0, 0, TW, TH) };\n      builtPut('face', fk, Fn);",
     "mask: skinMaskCtx.getImageData(0, 0, TW, TH), lip: lipMapCanvas().x.getImageData(0, 0, TW, TH) };\n      builtPut('face', fk, Fn);"),
    ("    faceTex.needsUpdate = true; roughTex.needsUpdate = true; skinMaskTex.needsUpdate = true; state.faceKey = fk;",
     "    faceTex.needsUpdate = true; roughTex.needsUpdate = true; skinMaskTex.needsUpdate = true; state.faceKey = fk;\n"
     "    if (!skinU.lipMap.value) { skinU.lipMap.value = new THREE.CanvasTexture(lipMapCanvas().c); skinU.lipMap.value.anisotropy = 4; }\n"
     "    skinU.lipMap.value.needsUpdate = true;"),
    # 8. the uniforms
    ("skinBounceDir: { value: new THREE.Vector3(0, -1, 0) }, skinBounceCol: { value: new THREE.Vector3() } };",
     "skinBounceDir: { value: new THREE.Vector3(0, -1, 0) }, skinBounceCol: { value: new THREE.Vector3() },\n"
     "  lipMap: { value: null }, lipWrap: { value: new THREE.Vector3(1, 1, 1) }, lipCoat: { value: new THREE.Vector4() }, lipCoatBump: { value: 0 }, lipBaseSpec: { value: 1 } };"),
    # 9. the shader: the lips' mask and the under layer's redder scatter
    ("'uniform vec3 skinWrapRGB, skinBounceDir, skinBounceCol;\\nvarying vec2 vSkinWrap;\\nvarying vec2 vSkinZones;\\nfloat skHeld = 0.0, skPit = 0.0, skRough = 0.0, skLay = 0.0, skSh = 1.0;\\n' +",
     "'uniform vec3 skinWrapRGB, skinBounceDir, skinBounceCol;\\nvarying vec2 vSkinWrap;\\nvarying vec2 vSkinZones;\\nfloat skHeld = 0.0, skPit = 0.0, skRough = 0.0, skLay = 0.0, skSh = 1.0;\\n' +\n"
     "    'uniform sampler2D lipMap;\\nuniform vec3 lipWrap;\\nuniform vec4 lipCoat;\\nuniform float lipCoatBump, lipBaseSpec;\\nfloat skLip = 0.0, skMoist = 0.0;\\n' +"),
    ("    '    vec4 mk = texture2D(skinMask, vMapUv);',",
     "    '    vec4 mk = texture2D(skinMask, vMapUv);',\n"
     "    '    vec4 lm = texture2D(lipMap, vMapUv); skLip = lm.r; skMoist = lm.g;',"),
    ("'float skNL = dot( geometryNormal, directLight.direction ); vec3 skW = skinWrapRGB * ( 1.0 - skHeld );\\n' +",
     "'float skNL = dot( geometryNormal, directLight.direction ); vec3 skW = skinWrapRGB * mix( vec3( 1.0 ), lipWrap, skLip ) * ( 1.0 - skHeld );\\n' +"),
    # 10. the shader: the top layer, a second lobe with its own normal, weight and roughness
    ("  // the bounce from below (the skin alone)\n",
     "  // the lips' top layer: the clearcoat lobe, its normal the lip's own with the grooves and flakes at full depth\n"
     "  fs = fs.replace('#include <clearcoat_normal_fragment_maps>', '#include <clearcoat_normal_fragment_maps>\\n#if defined( USE_CLEARCOAT ) && defined( USE_BUMPMAP )\\n' +\n"
     "    '  if ( skLip > 0.0 ) { vec2 ldx = dFdx( vMapUv ), ldy = dFdy( vMapUv ); float lh = texture2D( lipMap, vMapUv ).b;\\n' +\n"
     "    '    vec2 ldH = clamp( -lipCoatBump * skLip * vec2( texture2D( lipMap, vMapUv + ldx ).b - lh, texture2D( lipMap, vMapUv + ldy ).b - lh ), -0.25, 0.25 );\\n' +\n"
     "    '    clearcoatNormal = perturbNormalArb( - vViewPosition, normalize( mix( clearcoatNormal, normal, skLip ) ), ldH, faceDirection ); }\\n#endif');\n"
     "  fs = fs.replace('#include <lights_physical_fragment>', '#include <lights_physical_fragment>\\n#ifdef USE_CLEARCOAT\\n' +\n"
     "    '  if ( skLip > 0.0 ) { float lf = texture2D( lipMap, vMapUv ).b;\\n' +\n"
     "    '    material.clearcoat = mix( material.clearcoat, mix( lipCoat.x, lipCoat.y, skMoist ), skLip );\\n' +\n"
     "    '    material.clearcoatRoughness = mix( material.clearcoatRoughness, clamp( mix( lipCoat.z, lipCoat.w, skMoist ) + 0.3 * lf, 0.06, 1.0 ), skLip );\\n' +\n"
     "    '    material.specularColor *= mix( 1.0, lipBaseSpec, skLip ); }\\n#endif');\n"
     "  // the bounce from below (the skin alone)\n"),
    # 11. the uniforms' values from the sliders, and a clearcoat always compiled in for the lips
    ("  skinMat.clearcoat = SL.COAT[0] * lay; skinMat.clearcoatRoughness = SL.COAT[1];",
     "  skinMat.clearcoat = Math.max(SL.COAT[0] * lay, 1e-3); skinMat.clearcoatRoughness = SL.COAT[1];      // (never 0: the lips' top layer is the clearcoat)\n"
     "  var LL = LIP_LAYERS, ll = clamp(P.lipLayers == null ? 1 : P.lipLayers, 0, 1), gl_ = P.lipGloss, ls_ = P.lipstickAmt;\n"
     "  skinU.lipWrap.value.set(lerp(1, LL.WRAP[0], ll), lerp(1, LL.WRAP[1], ll), lerp(1, LL.WRAP[2], ll));\n"
     "  skinU.lipCoat.value.set(LL.COAT[0] * ll, LL.COAT[1] * ll, lerp(LL.COAT_ROUGH[0], 0.3, 0.6 * gl_ + 0.3 * ls_), lerp(LL.COAT_ROUGH[1], 0.1, gl_));\n"
     "  skinU.lipCoatBump.value = LL.COAT_BUMP * ll; skinU.lipBaseSpec.value = lerp(1, LL.BASE_SPEC, ll);"),
    # 12. the sliders
    ("['lipLine', 'Lip line (pale rim)', 0, 2, 0.02]], 'liptone'],",
     "['lipLine', 'Lip line (pale rim)', 0, 2, 0.02]], 'liptone'],\n"
     "    ['Layers', 'Two layers: a smooth pink under layer (the blood under a thin skin) and a rougher top layer with the vertical lip lines, wet at the parting and dry at the border.', [['lipLayers', 'Layered lips', 0, 1, 0.01], ['lipLines', 'Lip lines', 0, 2, 0.02], ['lipMoist', 'Moisture', 0, 2, 0.02]]],"),
]


def apply(src):
    if MARK in src:
        return src, False
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise SystemExit('anchor found %d times, nothing written:\n%s' % (n, old[:200]))
        src = src.replace(old, new)
    return src, True


def rebuild_html(html_path, main_src, old_main):
    h = open(html_path, encoding='utf-8').read()
    i = h.index('<script type="module">') + len('<script type="module">')
    j = h.index('</script>', i)
    body = h[i:j]
    k = body.rfind(old_main.strip()[:4000])
    if k < 0 or not body.rstrip().endswith(old_main.strip()):
        raise SystemExit('the html does not end its module with the current main.js; rebuild it by hand')
    open(html_path, 'w', encoding='utf-8').write(h[:i] + body[:k] + main_src.strip() + '\n' + h[j:])


if __name__ == '__main__':
    path = sys.argv[1]
    old = open(path, encoding='utf-8').read()
    new, changed = apply(old)
    if changed:
        open(path, 'w', encoding='utf-8').write(new)
        print('lips written into', path)
    else:
        print('already has the layered lips:', path)
    if len(sys.argv) > 2:
        rebuild_html(sys.argv[2], new, old if changed else new)
        print('rebuilt', sys.argv[2])
