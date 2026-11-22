"""Before / after montage of clay.py renders on fixed cameras:

    python3 blender/tension/montage.py OUT.png "before label" BEFORE_DIR "after label" AFTER_DIR view1 view2 ...

Each row is one view, before on the left and after on the right, labelled."""
import sys
from PIL import Image, ImageDraw

out, la, da, lb, db = sys.argv[1:6]
views = sys.argv[6:]
ims = [(Image.open('%s/%s.png' % (da, v)).convert('RGB'), Image.open('%s/%s.png' % (db, v)).convert('RGB')) for v in views]
w, h = ims[0][0].size
pad = 28
M = Image.new('RGB', (2 * w, len(views) * (h + pad)), (255, 255, 255))
d = ImageDraw.Draw(M)
for i, (a, b) in enumerate(ims):
    y = i * (h + pad)
    d.text((6, y + 6), '%s: %s' % (la, views[i]), fill=(0, 0, 0)); d.text((w + 6, y + 6), '%s: %s' % (lb, views[i]), fill=(0, 0, 0))
    M.paste(a, (0, y + pad)); M.paste(b, (w, y + pad))
M.save(out)
print('montage: wrote', out)
