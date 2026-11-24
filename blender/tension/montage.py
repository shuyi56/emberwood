"""Before / after montage of clay.py renders on fixed cameras:

    python3 blender/tension/montage.py OUT.png "label 1" DIR_1 "label 2" DIR_2 [...] view1 view2 ...

Each row is one view, one column per build, in the order given, labelled. A label is followed by a directory; the
first argument that is not followed by a directory starts the views."""
import sys, os
from PIL import Image, ImageDraw

a = sys.argv[1:]
out = a[0]; cols = []; i = 1
while i + 1 < len(a) and os.path.isdir(a[i + 1]):
    cols.append((a[i], a[i + 1])); i += 2
views = a[i:]
ims = [[Image.open('%s/%s.png' % (d, v)).convert('RGB') for _, d in cols] for v in views]
w, h = ims[0][0].size
pad = 28
M = Image.new('RGB', (len(cols) * w, len(views) * (h + pad)), (255, 255, 255))
dr = ImageDraw.Draw(M)
for r, row in enumerate(ims):
    y = r * (h + pad)
    for c, im in enumerate(row):
        dr.text((c * w + 6, y + 6), '%s: %s' % (cols[c][0], views[r]), fill=(0, 0, 0)); M.paste(im, (c * w, y + pad))
M.save(out)
print('montage: wrote', out)
