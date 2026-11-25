"""Add the hair fidelity mode to a Face Creator page: a "Fidelity" choice under the hairstyle that shows the style as
the creator sculpts it, or as the physics groom from tools/hair baked for the game (cards at LOD 0 / 1 / 2, or the
strands). The groom of each style comes from tools/hair/creator_asset.py and is written into the page, so it still
opens from a file.

    python3 tools/face-creator/apply-hair-fidelity.py IN.html OUT.html STYLE=ASSET.json [STYLE=ASSET.json ...]

e.g. wavyFull=asset-wavyFull.json ranger=asset-ranger.json. Running it again on its own output replaces the mode.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN, END = '<!-- hair-fidelity: begin -->', '<!-- hair-fidelity: end -->'


def main():
    src, out = sys.argv[1], sys.argv[2]
    html = open(src, encoding='utf-8').read()
    html = re.sub(re.escape(BEGIN) + '.*?' + re.escape(END), '', html, flags=re.S)
    blocks = [BEGIN]
    for arg in sys.argv[3:]:
        style, path = arg.split('=', 1)
        asset = json.load(open(path))
        summary = {'lod%d' % i: '%d cards, %s triangles' % (L['cards'], format(L['triangles'], ',')) for i, L in enumerate(asset['lods'])}
        body = json.dumps(asset, separators=(',', ':')).replace('</', '<\\/')
        blocks.append('<script type="application/json" id="groom-asset-%s" data-summary=\'%s\'>%s</script>'
                      % (style, json.dumps(summary), body))
    js = open(os.path.join(HERE, 'hair-fidelity.js'), encoding='utf-8').read()
    blocks.append('<script type="module">\n' + js + '\n</script>')
    blocks.append(END)
    i = html.rfind('</body>')
    html = (html[:i] + '\n'.join(blocks) + '\n' + html[i:]) if i >= 0 else html + '\n'.join(blocks)
    open(out, 'w', encoding='utf-8').write(html)
    print(out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
