"""Pores moved into the layers they belong to (thread: Study pores and fix skin layers, 2026-11-21).
python3 patch_pores.py face-creator.html   (patches in place; asserts every anchor once)"""
import sys
f = sys.argv[1]; s = open(f).read()

def rep(a, b, n=1):
    global s
    c = s.count(a)
    assert c == n, (c, a[:90])
    s = s.replace(a, b)

# ---- 1. sizes and density from measurement, and a bake fine enough to hold them
rep("var PORE_U0 = 0.5 - 0.36 * Math.PI, PORE_U1 = 0.5 + 0.36 * Math.PI;",
"""var PORE_U0 = 0.5 - 0.36 * Math.PI, PORE_U1 = 0.5 + 0.36 * Math.PI;
/* the pores' scale, measured (pores thread, 2026-11-21). Flament et al. 2015 (Clin Cosmet Investig Dermatol 8:85,
   2,585 women, cheek, pores counted from 250 um across): Caucasian women 18-29 have 64 visible pores per cm2 of
   cheek and a mean pore area of 0.16 mm2 (about 0.45 mm across). The lattice's cell is set so the medial cheek's
   density weight (0.6, poreRegion) gives 64/cm2: cell = sqrt(0.6 / 64) cm = 0.968 mm, a head height being HH = 8.6
   units of 27 mm (skin_layers.py: 1 unit is about 2.7 cm). The radius factor puts the dark part of a
   medial-cheek pore (where the pit is past a tenth of its depth, 0.85 of its radius) at 0.16 mm2. It was 1.86 mm cells
   and pits about 0.95 mm across: four times the measured area at a quarter of the density, read as dark pinpricks.
   The other regions keep their weights relative to the cheek (nose and forehead most, round the eyes least:
   Kim et al. 2018, Skin Res Technol 24:367). */
var PORE_CELL_MM = Math.sqrt(0.6 / 64) * 10, PORE_RAD = 0.29, UNIT_MM = 27;""")
rep("var W = 3072, H = 1536, cv = document.createElement('canvas');",
    "var W = 4096, H = 2048, cv = document.createElement('canvas');      // (0.14 mm a texel: a cheek pore is about 3 texels across)\n  ")
rep("var C0 = 1 / 125, TILT", "var C0 = PORE_CELL_MM / (HH * UNIT_MM), TILT")
rep("rad = 0.27 * Sz * (0.7 + 0.6 * r2 * r2);", "rad = PORE_RAD * Sz * (0.7 + 0.6 * r2 * r2);")

# ---- 2. the pore is not paint: no pore tint in the epidermis's colour when the skin is layered
rep("float poreCav = clamp((0.6 - texture2D(bumpMap, vBumpMapUv).r) * 3.0, 0.0, 1.0);\\n  diffuseColor.rgb *= mix(vec3(1.0), poreTint, poreCav);",
    "float poreCav = clamp((0.6 - texture2D(bumpMap, vBumpMapUv).r) * 3.0, 0.0, 1.0) * (1.0 - skinLayers * skinBaked);\\n  diffuseColor.rgb *= mix(vec3(1.0), poreTint, poreCav);")

# ---- 3. the layered shader: globals, the pit as a cavity, nothing of it in the scatter's hold or the roughness
rep("float skHeld = 0.0, skPit = 0.0, skRough = 0.0, skLay = 0.0, skSh = 1.0;\\n' +",
    "float skHeld = 0.0, skPit = 0.0, skRough = 0.0, skLay = 0.0, skSh = 1.0;\\n' +\n    'uniform float skinDebug;\\nvec3 skN0 = vec3(0.0, 0.0, 1.0); float skCav = 0.0;\\n' + skinPoreGlsl() +")
rep("'    skHeld = skLay * min(1.0, max(pig * SL_PIGMENT_HOLD, skPit * SL_PORES * skinPores * SL_PORE_HOLD));',",
    "'    skHeld = skLay * min(1.0, pig * SL_PIGMENT_HOLD);',      // (the pores are no longer held out of the scatter: see skinPoreGlsl)\n"
    "    '    skCav = skLay * skPit * SL_PORES * skinPores;',")
rep("'    skRough = mix(SL_ROUGH_0, SL_ROUGH_1, t2.a) + skPit * 0.18;',",
    "'    skRough = mix(SL_ROUGH_0, SL_ROUGH_1, t2.a);',      // (the pit's specular is occluded instead of roughened)")
rep("'    if (skLay > 0.0) {',      // (none of it when the layers are off or not baked yet)",
    "'    if (skinDebug > 1.5) diffuseColor.rgb = vec3(0.42);',      // (clay: one grey albedo, the layers' lighting kept)\n"
    "    '    if (skLay > 0.0) {',      // (none of it when the layers are off or not baked yet)")
rep("'    diffuseColor.rgb = mix(diffuseColor.rgb, comb, skLay);',",
    "'    diffuseColor.rgb = mix(diffuseColor.rgb, comb, skLay * (skinDebug > 1.5 ? 0.0 : 1.0));',")
# (the unbumped normal, for the diffuse normals)
rep("  var fs = sh.fragmentShader.replace('#include <color_fragment>', body);",
    "  var fs = sh.fragmentShader.replace('#include <color_fragment>', body);\n"
    "  fs = fs.replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\\n  skN0 = nonPerturbedNormal;');")
# (direct light: per-channel diffuse normals, the cavity's occlusion of the local diffuse and of the specular)
rep("'float skNL = dot( geometryNormal, directLight.direction ); vec3 skW = skinWrapRGB * mix( vec3( 1.0 ), lipWrap, skLip ) * ( 1.0 - skHeld );\\n' +",
    "'float skNL = dot( geometryNormal, directLight.direction ); vec3 skW = skinWrapRGB * mix( vec3( 1.0 ), lipWrap, skLip ) * ( 1.0 - skHeld );\\n' +\n"
    "    '\\tvec3 skNLc = mix( vec3( skNL ), vec3( dot( normalize( mix( skN0, geometryNormal, SK_DW.r ) ), directLight.direction ), dot( normalize( mix( skN0, geometryNormal, SK_DW.g ) ), directLight.direction ), dot( normalize( mix( skN0, geometryNormal, SK_DW.b ) ), directLight.direction ) ), skLay );\\n' +")
rep("'\\treflectedLight.directDiffuse += clamp( ( vec3( skNL ) + skW ) / ( 1.0 + skW ), 0.0, 1.0 ) / ( 1.0 + skW ) * skVis * directLight.color * BRDF_Lambert( material.diffuseColor );'));",
    "'\\treflectedLight.directDiffuse += clamp( ( skNLc + skW ) / ( 1.0 + skW ), 0.0, 1.0 ) / ( 1.0 + skW ) * skVis * ( vec3( 1.0 ) - skCav * SK_AO * SK_DW ) * directLight.color * BRDF_Lambert( material.diffuseColor );')\n"
    "    .replace('reflectedLight.directSpecular += irradiance * BRDF_GGX(', 'reflectedLight.directSpecular += irradiance * ( 1.0 - skCav * SK_AO ) * BRDF_GGX('));")
# (indirect light and the colour-coded check)
rep("  sh.fragmentShader = pre + fs;\n}",
    "  fs = fs.replace('#include <lights_fragment_end>', '#include <lights_fragment_end>\\n  reflectedLight.indirectDiffuse *= vec3( 1.0 ) - skCav * SK_AO * SK_DW;\\n  reflectedLight.indirectSpecular *= 1.0 - skCav * SK_AO;');\n"
    "  fs = fs.replace('#include <opaque_fragment>', '#include <opaque_fragment>\\n  if ( skinDebug > 0.5 && skinDebug < 1.5 ) {\\n' +\n"
    "    '    float dbH = texture2D( bumpMap, vBumpMapUv ).r, dbCrease = clamp( ( dbH - 0.627 ) / 0.236, 0.0, 1.0 ), dbL = 0.45 + 0.55 * max( dot( skN0, normalize( vec3( 0.3, 0.5, 1.0 ) ) ), 0.0 );\\n' +\n"
    "    '    vec3 dbC = mix( vec3( 0.78 ), vec3( 0.1, 0.75, 0.85 ), dbCrease * 0.8 );\\n' +\n"
    "    '    dbC = mix( dbC, vec3( 0.85, 0.1, 0.65 ), clamp( skPit * 1.5, 0.0, 1.0 ) );\\n' +\n"
    "    '    dbC = mix( dbC, vec3( 0.95, 0.8, 0.1 ), skLip );\\n' +\n"
    "    '    dbC = mix( dbC, vec3( 0.15, 0.2, 0.45 ), 1.0 - step( 0.001, skLay ) );\\n' +\n"
    "    '    gl_FragColor = vec4( pow( dbC * dbL, vec3( 1.0 / 2.2 ) ), 1.0 );\\n  }');\n"
    "  sh.fragmentShader = pre + fs;\n}")
rep("skinLayers: { value: 1 }, skinVeins: { value: 1 },", "skinLayers: { value: 1 }, skinDebug: { value: 0 }, skinVeins: { value: 1 },")

# ---- 4. the constants, from the sources
rep("function skinGlslConsts() {",
"""/* Which layer a pore belongs to (pores thread, 2026-11-21).
   - A pore is a depression in the stratum corneum, the skin's topmost layer, lined with keratin (Maia Campos et al.
     2019, Front Physiol 10:254: "a depression in the stratum corneum layer, presenting a keratinized coating"; Flament
     et al. 2015: "slight depressions of the skin surface"). It carries no pigment of its own, so nothing of it is
     painted into the epidermis's colour (it was: poreTint, about 15% darker, in layer 3).
   - Its relief belongs to the surface. The specular reflection sees it whole; the diffuse light, which has travelled
     under the skin, sees it softened, red most (Ma et al. 2007, EGSR, hybrid normals: one normal for the specular and
     one for each diffuse colour; Graham et al. 2013, Comput Graph Forum 32: the pore-scale microgeometry is used for
     the specular alone, the subsurface normals stay at the scan's coarser scale). How much of a pit each diffuse colour
     keeps is the share of skin's diffusion profile that stays within a pore's radius (0.25 mm): d'Eon and Luebke 2007
     (GPU Gems 3, ch. 14, fig. 14-13) give skin's profile as six Gaussians; each keeps 1 - exp(-r^2 / 2v) of its light
     within r. That is SK_DW, about 0.31 red, 0.64 green, 0.81 blue.
   - The pit's shade is occlusion, not paint: the floor of a pit about 0.45 mm across and 0.13 mm deep (Maia Campos
     2019: a pore's lumen still seen at 126 um under the surface) sees the sky through a cone of half-angle
     atan(0.225 / 0.13) = 60 degrees, sin^2 of which is 0.75 of the cosine-weighted hemisphere: SK_AO = 0.25 is what it
     loses. The specular loses all of it; the diffuse only its local part (SK_DW), the rest arriving from under the
     surrounding skin, so a pit reads warmer than the skin round it, red filling it (Jimenez et al. 2015, colour bleeding
     into shadowed regions), without a tint chosen by eye.
   - The scatter's hold (Blender's PORE_HOLD) and the rougher pit (+0.18) are gone: they kept pits from blurring away
     under Blender's scatter, which the per-colour normals now handle as the skin does. */
var SK_GAUSS = [[0.0064, 0.233, 0.455, 0.649], [0.0484, 0.100, 0.336, 0.344], [0.187, 0.118, 0.198, 0], [0.567, 0.113, 0.007, 0.007], [1.99, 0.358, 0.004, 0], [7.41, 0.078, 0, 0]];
var SK_PORE_R_MM = 0.25, SK_PIT = { r: 0.225, d: 0.13 };
var SK_DW = [1, 2, 3].map(function (c) { return SK_GAUSS.reduce(function (a, g) { return a + g[c] * (1 - Math.exp(-SK_PORE_R_MM * SK_PORE_R_MM / (2 * g[0]))); }, 0); });
var SK_AO = 1 - Math.pow(Math.sin(Math.atan(SK_PIT.r / SK_PIT.d)), 2);
function skinPoreGlsl() { return 'const vec3 SK_DW = vec3(' + SK_DW.map(function (v) { return v.toFixed(4); }).join(', ') + ');\\nconst float SK_AO = ' + SK_AO.toFixed(4) + ';\\n'; }
function skinGlslConsts() {""")
# ---- 5. the header's note on the pores
rep("""     drive). Blender's pits are a third of a millimetre and vanish under its scatter at a face's distance; given
     PORE_SHADE, the creator's larger pits read as dark pinpricks, so they keep their own warm shade (poreTint) and take
     from Blender only the roughness in the pit and the hold out of the scatter.""",
"""     drive), sized from measurement (PORE_CELL_MM), and laid in the layers they belong to: their relief in the surface,
     whole for the specular and softened per colour for the diffuse, their shade an occlusion of the light rather than
     paint in the epidermis (skinPoreGlsl). poreTint is used only with the layers off.""")
open(f, 'w').write(s); print('ok')
