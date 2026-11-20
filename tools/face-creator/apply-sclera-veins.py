"""The white's vessels grown from the corners of the eye, as a targeted edit of face-creator.html's makeSclera.

    python3 tools/face-creator/apply-sclera-veins.py /mnt/project-files/face-creator/main.js /mnt/project-files/face-creator/face-creator.html

John, 2026-11-20: "change the veins with proper branching. The trunk begins at the ends of the [eye] near the flesh".
Before: space colonisation (Runions et al.) rooted at five points along each lid margin and 36 scattered seeds, so
vessels began anywhere along the lids and in the middle of the white. Now the network is one tree per corner, rooted at the
corner itself, beside the pink flesh (the caruncle at the inner corner, the canthus at the outer):
  - three trunks leave each corner together (one thick stem that forks at once into an upper, a middle and a lower trunk, as the
    conjunctival arteries and veins do), run toward the iris in a long, slightly meandering curve, and end near the limbus;
  - the attractors then grow the branches off the trunks (as before), each fork a bifurcation, and the widths come from
    Murray's law (radius cubed adds at a fork), so the stem is the thickest and the twigs the finest;
  - the widths are no longer capped at the old 1.5 px, which flattened every large vessel to one width.
The visible white ends toward 0.82 of the way out, so the stems lie in the red shade at the corner and come out of it into the
white as the trunks. Idempotent; anchored; writes nothing if an anchor is missing.
"""
import sys

MARK = '/* @@SCLERA_VEINS@@ */'

V1_TRUNKS = """    function hAt(u, f) { return f >= 0 ? f * lidTop(u, OPEN) : f * lidBot(u, OPEN); }      // (f: -1 the lower margin .. 1 the upper)
    [0.62, 0.05, -0.6].forEach(function (f0, ti) {
      var ph = 2 * r(), end = 0.4 + 0.07 * r(), len = 0.985 - end, steps = Math.round(len / 0.0075), prev = root;
      for (var st = 1; st <= steps; st++) {
        var t = st / steps, u = side * (0.985 - len * t), f = f0 * Math.sin(Math.PI * 0.5 * Math.min(1, t * 1.25)) * (1 - 0.1 * t);
        var h = hAt(u, f) + 0.012 * Math.sin(t * 15 + ph + ti) * (0.3 + t) + 0.003 * normal(r);
        prev = addNode([u, h], prev);
      }
    });
"""
V2_TRUNKS = """    function hAt(u, f) { return f >= 0 ? f * lidTop(u, OPEN) : f * lidBot(u, OPEN); }      // (f: -1 the lower margin .. 1 the upper)
    // (a fan: five trunks spread over the whole height of the white, opening early, each ending at its own place on the
    // limbus's arc, and curving a little as they go)
    [0.88, 0.5, 0.08, -0.36, -0.8].forEach(function (f0, ti) {
      var ph = 2 * r(), end = 0.34 + 0.12 * r() + 0.06 * (1 - Math.abs(f0)), len = 0.985 - end, steps = Math.round(len / 0.0075), prev = root;
      for (var st = 1; st <= steps; st++) {
        var t = st / steps, u = side * (0.985 - len * t), f = f0 * (1 - Math.pow(1 - Math.min(1, t * 1.12), 1.7)) * (1 - 0.06 * t);
        var h = hAt(u, f) + 0.011 * Math.sin(t * 13 + ph + ti * 1.7) * (0.3 + t) + 0.003 * normal(r);
        prev = addNode([u, h], prev);
      }
    });
"""

OLD_ROOTS = """  var nodes = [], parent = [];
  function addNode(p, par) { nodes.push(p); parent.push(par); }
  [-1, 1].forEach(function (side) {
    addNode([0.985 * side, 0], -1);
    [0.35, 0.5, 0.62, 0.75, 0.88].forEach(function (uq) {
      addNode([uq * side + 0.04 * normal(r), lidTop(uq, OPEN) * 0.97], -1);
      addNode([uq * side + 0.04 * normal(r), -lidBot(uq, OPEN) * 0.97], -1);
    });
  });
  for (var s = 0; s < 36 && att.length; s++) { var a = att[Math.floor(r() * att.length)]; addNode([a[0], a[1]], -1); }
"""

NEW_ROOTS = """  var nodes = [], parent = [];
  function addNode(p, par) { nodes.push(p); parent.push(par); return nodes.length - 1; }
  """ + MARK + """
  // one tree per corner, rooted at the corner (beside the caruncle or the outer canthus): a stem that forks at once into three
  // trunks, an upper, a middle and a lower, each a long curve toward the limbus ending at |u| about 0.4
  [-1, 1].forEach(function (side) {
    var root = addNode([0.985 * side, 0], -1);
    function hAt(u, f) { return f >= 0 ? f * lidTop(u, OPEN) : f * lidBot(u, OPEN); }      // (f: -1 the lower margin .. 1 the upper)
    // (a fan: five trunks spread over the whole height of the white, opening early, each ending at its own place on the
    // limbus's arc, and curving a little as they go)
    [0.88, 0.5, 0.08, -0.36, -0.8].forEach(function (f0, ti) {
      var ph = 2 * r(), end = 0.34 + 0.12 * r() + 0.06 * (1 - Math.abs(f0)), len = 0.985 - end, steps = Math.round(len / 0.0075), prev = root;
      for (var st = 1; st <= steps; st++) {
        var t = st / steps, u = side * (0.985 - len * t), f = f0 * (1 - Math.pow(1 - Math.min(1, t * 1.12), 1.7)) * (1 - 0.06 * t);
        var h = hAt(u, f) + 0.011 * Math.sin(t * 13 + ph + ti * 1.7) * (0.3 + t) + 0.003 * normal(r);
        prev = addNode([u, h], prev);
      }
    });
  });
"""

EDITS = [
    (OLD_ROOTS, NEW_ROOTS),
    ("  for (i = 0; i < nN; i++) rad[i] = Math.min(rad[i] * 1.1, 1.5);",
     "  for (i = 0; i < nN; i++) rad[i] = Math.min(rad[i], 2.6);      // (Murray's law as it grows; no longer flattened at 1.5)"),
    # the trunks are the thickest but still fine: the width of the stroke is capped, the strength eased, so the white reads as
    # veined, not bloodshot
    ("sig = Math.max(0.55, rr), amp = 0.3 + 0.7 * Math.min(1, rr / 1.4);",
     "sig = Math.max(0.55, Math.min(rr, 1.2)), amp = 0.75 * (0.3 + 0.7 * Math.min(1, rr / 1.4));"),
]


def main(main_path, html_path):
    old_main = open(main_path, encoding='utf-8').read()
    html = open(html_path, encoding='utf-8').read()
    if MARK in html:
        if V2_TRUNKS in html:
            print('already has the fanned corner-rooted vessels')
            return
        if V1_TRUNKS not in html:
            raise SystemExit('the vessels were changed by hand; nothing written')
        i = html.index('<script type="module">') + len('<script type="module">')
        j = html.index('</script>', i)
        open(html_path, 'w', encoding='utf-8').write(html.replace(V1_TRUNKS, V2_TRUNKS))
        print('vessel trunks fanned out in', html_path)
        return
    i = html.index('<script type="module">') + len('<script type="module">')
    j = html.index('</script>', i)
    body = html[i:j]
    if not body.rstrip().endswith(old_main.strip()):
        raise SystemExit('the html does not end its module with the current main.js; rebuild it by hand')
    k = len(body.rstrip()) - len(old_main.strip())
    prefix = body[:k]
    for old, new in EDITS:
        n = prefix.count(old)
        if n != 1:
            raise SystemExit('anchor found %d times, nothing written:\n%s' % (n, old[:200]))
        prefix = prefix.replace(old, new)
    open(html_path, 'w', encoding='utf-8').write(html[:i] + prefix + old_main.strip() + '\n' + html[j:])
    print('corner-rooted vessels written into', html_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
