"""Creamy, milky eye whites in the Face Creator, as targeted edits, and the rebuild of face-creator.html.

    python3 tools/face-creator/apply-sclera-milk.py /mnt/project-files/face-creator/main.js /mnt/project-files/face-creator/face-creator.html

Every edit is anchored on a line of the live files and checked first: if an anchor is missing or not unique, nothing is
written. Run on files that already have it, it changes nothing.

John, 2026-11-20 (with a macro photo of a brown eye): "get that creamy milky white for the eyes, right now the white is too even".
What the photo shows, and what is built:
  - the white is not one tone: ivory to cream, creamier toward the corners (where the eye is most exposed to light and the
    conjunctiva is thicker), cooler and whiter over the middle, and the whole of it soft. Sclera also yellows and reddens with
    age (Russell et al. 2014, "Sclera color changes with age and is a cue for perceiving age, health, and beauty"), so a young
    white is lighter, a little cooler and less red, and a cream white sits between. -> makeSclera: a cream base, and a broad,
    warped patch field of cream and cool white (MILK of them, 0 to 2), the cream weighted to the corners;
  - its shade is warm, not grey: light that enters the sclera scatters and comes out warm, so the lid's shadow and the turn
    of the ball go ivory and pink-tan, never grey or brown-grey. -> paintEye: the grey shading blends become warm;
  - it glows: a milky translucent look is light that was scattered inside coming back out, which three.js has no scatter
    for. A warm self-glow in the white (and only the white: it follows the paint's luminance) stands in for it, as the skin
    does with sssMask. -> eyeMats: onBeforeCompile adds it, scaled by MILK;
  - the tear film stays (the clearcoat) and the wet highlight is not changed.
"""
import sys

MARK = '/* @@SCLERA_MILK@@ */'

HTML_EDITS = [
    ("  var tint = P.tint, c0 = [lerp(236, 232, tint), lerp(231, 220, tint), lerp(220, 186, tint)];",
     "  " + MARK + "\n  var tint = P.tint, milk = P.milk == null ? 1 : P.milk, c0 = [lerp(246, 240, tint), lerp(240, 226, tint), lerp(228, 186, tint)];      // (a cream white, not a grey one)"),
    ("    blend(c, [156, 124, 110], 0.75 * Math.pow(1 - lam, 1.1));\n    blend(c, [168, 140, 134], 0.5 * sm(0.75, 1.0, ru));",
     "    blend(c, [218, 176, 154], 1.0 * Math.pow(1 - lam, 1.0)); blend(c, [180, 132, 112], 0.4 * Math.pow(1 - lam, 2.2));      // (the turn of the ball: ivory and pink-tan, the light scattered under the surface; never grey)\n"
     "    blend(c, [206, 172, 158], 0.5 * sm(0.75, 1.0, ru));"),
    ("    var m = f1(tu, tv) * 0.6 + f2(tu, tv) * 0.4;\n    blend(c, [234, 212, 178], 0.35 * clamp(m, 0, 1)); blend(c, [214, 204, 192], 0.3 * clamp(-m, 0, 1));",
     "    var m = f1(tu, tv) * 0.6 + f2(tu, tv) * 0.4;\n"
     "    blend(c, [234, 212, 178], 0.2 * clamp(m, 0, 1)); blend(c, [222, 214, 204], 0.18 * clamp(-m, 0, 1));\n"
     "    // the broad patches: cream (the exposed zone toward the corners, where the conjunctiva is thicker and yellower) and a\n"
     "    // cool, bluish white over the middle, on a warped field a few cells wide\n"
     "    var wu = tu + 0.12 * f1(tu, tv), wv = tv + 0.12 * f2(tu, tv), bm = noise2(wu * 2.8, wv * 1.5, 15 + sd) * 0.7 + noise2(wu * 6.5, wv * 3.2, 16 + sd) * 0.3;\n"
     "    blend(c, [248, 226, 178], 0.55 * milk * clamp(bm * 1.8 + 0.2, 0, 1) * (0.25 + 0.75 * sm(0.3, 0.9, ru)));\n"
     "    blend(c, [226, 235, 248], 0.55 * milk * clamp(-bm * 1.8, 0, 1) * (1 - sm(0.3, 0.85, ru)));"),
]

MAIN_EDITS = [
    ("blendTo(sc, [176, 160, 154], 0.45 * sm(0.15, 0.85, up)); blendTo(sc, [128, 104, 100], 0.55 * sm(0.75, 1.0, up));",
     "blendTo(sc, [226, 184, 160], 0.75 * sm(0.0, 0.75, up)); blendTo(sc, [168, 114, 94], 0.5 * sm(0.7, 1.0, up));      // (warm shade under the upper lid, not grey: " + MARK + ")"),
    ("blendTo(sc, [182, 162, 156], 0.35 * sm(0.45, 0.85, lo));",
     "blendTo(sc, [222, 196, 180], 0.35 * sm(0.45, 0.85, lo));"),
    ("blendTo(sc, [60, 36, 32], 0.55 * sm(0.78, 1.0, up));",
     "blendTo(sc, [96, 58, 46], 0.45 * sm(0.78, 1.0, up));"),
    ("vessels: 1400, redness: 0.8, flush: 0.18, tint: 0,",
     "vessels: 1400, redness: 0.8, flush: 0.18, tint: 0, milk: 1,"),
    ("var sk = keyFor(['vessels', 'tint', 'seed'], state.geoVersion);",
     "var sk = keyFor(['vessels', 'tint', 'seed', 'milk'], state.geoVersion);"),
    ("cache.sclera = makeSclera({ vessels: P.vessels, tint: P.tint, seed: P.seed });",
     "cache.sclera = makeSclera({ vessels: P.vessels, tint: P.tint, seed: P.seed, milk: P.milk });"),
    ("['tint', 'Yellowing', 0, 1, 0.01]], 'seed']",
     "['tint', 'Yellowing', 0, 1, 0.01], ['milk', 'Cream and milk', 0, 2, 0.02]], 'seed']"),
    ("ior: 1.38, specularIntensity: 0.6, envMapIntensity: 1.0 }); });",
     "ior: 1.38, specularIntensity: 0.6, envMapIntensity: 1.0 }); });\n"
     "/* the white's milky glow: light scattered inside the sclera comes back out warm, which three.js has no scatter for. A warm\n"
     "   self-glow of the paint's own colour, only where the paint is light (the white, not the iris or the lids' shade) */\n"
     "var eyeGlow = { value: 1 };\n"
     "eyeMats.forEach(function (m) {\n"
     "  m.onBeforeCompile = function (sh) {\n"
     "    sh.uniforms.eyeGlow = eyeGlow;\n"
     "    sh.fragmentShader = 'uniform float eyeGlow;\\n' + sh.fragmentShader.replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\\n"
     "  float egl = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));\\n"
     "  totalEmissiveRadiance += diffuseColor.rgb * vec3(0.2, 0.15, 0.1) * smoothstep(0.22, 0.6, egl) * eyeGlow;');\n"
     "  };\n"
     "});"),
    ("  skinU.lipCoatBump.value = LL.COAT_BUMP * ll; skinU.lipBaseSpec.value = lerp(1, LL.BASE_SPEC, ll);",
     "  skinU.lipCoatBump.value = LL.COAT_BUMP * ll; skinU.lipBaseSpec.value = lerp(1, LL.BASE_SPEC, ll);\n"
     "  eyeGlow.value = clamp(P.milk == null ? 1 : P.milk, 0, 2);"),
]


def edit(src, edits, what):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise SystemExit('%s: anchor found %d times, nothing written:\n%s' % (what, n, old[:200]))
        src = src.replace(old, new)
    return src


def main(main_path, html_path):
    old_main = open(main_path, encoding='utf-8').read()
    html = open(html_path, encoding='utf-8').read()
    if MARK in old_main and MARK in html:
        print('already has the milky whites')
        return
    new_main = edit(old_main, MAIN_EDITS, 'main.js')
    i = html.index('<script type="module">') + len('<script type="module">')
    j = html.index('</script>', i)
    body = html[i:j]
    if not body.rstrip().endswith(old_main.strip()):
        raise SystemExit('the html does not end its module with the current main.js; rebuild it by hand')
    k = len(body.rstrip()) - len(old_main.strip())
    prefix = edit(body[:k], HTML_EDITS, 'face-creator.html prefix')
    open(main_path, 'w', encoding='utf-8').write(new_main)
    open(html_path, 'w', encoding='utf-8').write(html[:i] + prefix + new_main.strip() + '\n' + html[j:])
    print('milky whites written into', main_path, 'and', html_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
