"""The face's outline against the hair, from two close-ups of the face taken by the same camera (creator_shots.cjs,
view face=0/4/22/0.42): the creator's own hair (mode creator) and the groom (mode strands).

    python3 tools/gauges/hair/face_edge.py CREATOR.png GROOM.png [GROOM2.png ...]

Per image row it finds where the skin ends on each side of the face, and reports the groom's face width against the
creator's (px, and mm on a rough 140 mm face), each edge's offset, and how light the 5 px of hair right at the skin
are against the hair 15 to 30 px out. A layer drawn past the top hair onto the cheek (the coverage layer, the mid
layer) narrows the face by its width and stands as a dark line at the edge: both read as a change in the face's shape.
"""
import sys
import numpy as np
from PIL import Image


def load(n):
    return np.asarray(Image.open(n).convert('RGB')).astype(float)


def skin(a):
    r,g,b=a[...,0],a[...,1],a[...,2]
    return (r>135)&(r>g+12)&(g>b-5)&(g>75)
def edges(a, rows):
    m=skin(a); L=[];R=[]
    for y in rows:
        xs=np.where(m[y])[0]
        # the face's run: the longest run of skin through the image's middle
        if len(xs)<20: L.append(np.nan);R.append(np.nan);continue
        c=a.shape[1]//2; row=m[y]; l=c; 
        while l>0 and row[l-8:l].any(): l-=1
        r_=c
        while r_<a.shape[1]-1 and row[r_:r_+8].any(): r_+=1
        L.append(l);R.append(r_)
    return np.array(L,float),np.array(R,float)
def band(a,L,R,rows,d0,d1):
    lum=a.mean(2); v=[]
    for y,l,r in zip(rows,L,R):
        if np.isnan(l): continue
        v+=list(lum[y,int(max(0,l-d1)):int(max(1,l-d0))]); v+=list(lum[y,int(r+d0):int(r+d1)])
    return float(np.median(v))
rows=range(230,520,2)
cr = load(sys.argv[1]); Lc,Rc=edges(cr,rows)
PX_MM=None
for n in sys.argv[2:]:
    a=load(n); L,R=edges(a,rows)
    ok=~np.isnan(L)&~np.isnan(Lc)
    wc=(Rc-Lc)[ok]; w=(R-L)[ok]
    mm=140.0/np.nanmax(Rc-Lc)   # the creator's face is about 140 mm at its widest: px to mm (rough scale)
    print('%s: face width vs creator, median %+.1f px (%+.1f mm), worst row %+.1f px; left edge %+.1f px, right edge %+.1f px; hair at the skin (0-5 px) lum %.0f, hair 15-30 px out %.0f, ratio %.2f'
          % (n, np.median(w-wc), np.median(w-wc)*mm, (w-wc)[np.argmax(np.abs(w-wc))], np.median((L-Lc)[ok]), np.median((R-Rc)[ok]), band(a,L,R,rows,0,5), band(a,L,R,rows,15,30), band(a,L,R,rows,0,5)/band(a,L,R,rows,15,30)))
a=cr; print('creator: hair at the skin lum %.0f, 15-30 px out %.0f, ratio %.2f' % (band(a,Lc,Rc,rows,0,5), band(a,Lc,Rc,rows,15,30), band(a,Lc,Rc,rows,0,5)/band(a,Lc,Rc,rows,15,30)))
