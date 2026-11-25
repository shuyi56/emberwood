"""How much of the side's solid shape the drawn hair fills.

    python3 tools/hair/side_fill.py HEAD.json.gz GROOM_DIR [--strands 8000]

The shape (groom.py SIDE_PIECE_*, John 2026-11-25: "draw sides as one solid shape first then make sure the hair fills
it"): the side piece in front of the ear, and the mass behind it. Each is cut into cells of 0.03 rad round the head by
0.03 HH of height (about 1 cm by 6.5 mm); a cell is filled when the hair drawn through it (children as render.py
grows them, each standing for its share of the hairs) covers it: FILL_HAIRS or more. Prints, per region and side,
the share of cells filled and the heights of the bare ones."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalp_field as SF
import render as RD
import groom as GR

CELL = (0.03, 0.03)
FILL_HAIRS = 40          # DRAFT: hairs through a cell for it to read as covered (a 1 cm cell of 0.08 mm hairs side by
                         # side takes 125; hair lies several deep, and 40 already hides the skin in the page's strand view)
REGIONS = {              # name: (|theta| lo, hi, z hi, z lo)
    'side piece': (GR.SIDE_PIECE_TH[0], GR.SIDE_PIECE_TH[1], 0.66, 0.0),
    'behind the ear': (1.6, 2.2, 0.66, -0.3),
}


def main():
    a = sys.argv[1:]
    n = int(a[a.index('--strands') + 1]) if '--strands' in a else 8000
    surf = SF.Surface(SF.load_head(a[0]))
    F = SF.build(surf)
    Gm = dict(np.load(os.path.join(a[1], 'groom.npz')))
    paths, share = RD.grow_children(F, Gm, n)
    hairs = share * Gm['n_hairs'].sum()
    for name, (t0, t1, z1, z0) in REGIONS.items():
        for sd, lab in ((-1, 'right'), (1, 'left')):
            nt, nz = int(round((t1 - t0) / CELL[0])), int(round((z1 - z0) / CELL[1]))
            H = np.zeros((nt, nz))
            for p, h in zip(paths, hairs):
                p = p[~np.isnan(p[:, 0])]
                th = SF.theta(p)
                m = (np.sign(th) == sd)
                i = np.floor((np.abs(th) - t0) / CELL[0]).astype(int)
                j = np.floor((z1 - p[:, 2]) / CELL[1]).astype(int)
                m &= (i >= 0) & (i < nt) & (j >= 0) & (j < nz)
                for c in set(zip(i[m], j[m])):
                    H[c] += h
            full = H >= FILL_HAIRS
            bare = [round(z1 - (j + 0.5) * CELL[1], 2) for j in range(nz) if not full[:, j].all()]
            print('%-15s %-5s filled %3.0f%% of %d cells; rows with a bare cell at z: %s' % (
                name, lab, 100 * full.mean(), full.size, bare if bare else 'none'))


if __name__ == '__main__':
    main()
