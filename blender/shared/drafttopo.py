"""Topology checks for the drafting stage (John, 2026-11-21: "Add topological diagnostics to the drafting stage").

A draft (the flat pattern pieces with their seams, the parts a garment or a hairstyle is held by, and its closures) is
read before anything is sewn or solved, and each check says pass or fail, why, and draws a cheap diagram. Topology
decides where cuts, openings, singular points and holds have to go, and whether every free part has a path to
something that holds it; a solve cannot fix a draft that is wrong here, so these run first.

No Blender in this file: numpy (and PIL for the sheet). The adapters that read a real draft (the boot's pieces as cut,
the bodice's closing edge and the lapels' break, a bun's strands and holds) are in tools/render-queue/rq_drafttopo.py.

The draft, as a dict (JSON-able; arrays may be numpy):

    name, unit_cm               what it is, and one model unit in centimetres
    pieces   [{name, V (n,3) or (n,2), F [[i, j, k, ...], ...],
               edges {label: vertex mask or index list}   the cut edge's runs: a seam letter, or an opening's name
               openings [labels]     the labels that are meant to stay open after sewing (neck, hem, top, lacing...)
               visible  True         the piece shows on the outside (singular points on it are flagged)
               allowance 0           seam allowance in units: singular points this near the cut edge are not 'visible'
               material ''           'leather', 'wool', 'linen'... (the flattening limit)
               flat False            the piece is already flat (V in the plane): its curvature is not the target's}]
    seams    [[label, pieceA, pieceB], ...]   label sewn on A to label on B (None: every label found on exactly two
             pieces, as the boot's letters are). [label, A, B, label_on_B] when the two runs carry different names.
    expected {pieces_chi: 1, chi: ..., openings: [...], genus: 0}   what the assembled garment should be
    parts    {nodes [names], anchors [names], holds [[a, b, kind], ...], loaded [names], min_holds 2}
    closures [{name, edge (n,3) top to bottom, buttons [arc length from the top], break_s, end_s, neck_s,
               pressure {name: s}, strain (m,2) [[s, strain], ...], button_d, extension, hole_len, hole 'horizontal'}]

The checks and where their rules come from (each check's 'source' in the report; 'experimental' where the limit is a
judgement and not taken from a source):

  euler       chi = V - E + F; genus g = (2c - chi - b) / 2 for an orientable surface with c components and b boundary
              loops. A pattern piece cut from cloth is a disc: chi 1, one boundary loop, genus 0 (Edelsbrunner and
              Harer, Computational Topology, 2010, ch. II and IV; Botsch et al., Polygon Mesh Processing, 2010, 1.3).
              The assembled garment's chi is counted on the pieces glued along their seam runs as a cell complex:
              chi = sum chi_i - (corners glued away) + (runs glued away). A bodice with neck, two armholes and a hem
              open, sewn shut everywhere else, is a sphere with 4 holes: chi = 2 - 4 = -2.
  boundary    the boundary loops of each piece and of the sewn garment, each named by the labels along it; after
              sewing every loop must be one of the openings the draft names. A loop that is not is an open seam or a
              hole. (Same sources; a loop is a connected component of the boundary, a closed 1-manifold.)
  seams       every seam run paired with exactly one partner run (an edge with two faces is the 2-manifold condition;
              three pieces on one seam is non-manifold: Botsch et al. 1.3). The two runs' lengths are compared too
              (a seam whose sides differ by more than the ease it is meant to take puckers).
  mesh        per piece: edges with 3+ faces, bow-tie vertices, flipped winding (neighbouring faces wound against
              each other: an orientation must exist for a piece of cloth), degenerate faces, loose vertices.
  singular    vertices whose valence differs from the regular grid's (a quad grid: 4 inside, 3 on the edge; a
              triangle grid: 6 and 4). Their indices must sum to the Euler characteristic, sum(4 - d) + sum(3 - d)
              = 4 chi on a quad mesh (sum(6 - d) + sum(4 - d) = 6 chi on triangles), the discrete Poincare-Hopf
              theorem (Bommes et al., Quad-Mesh Generation and Processing: A Survey, CGF 2013, 2.1). Interior ones
              on a visible piece, more than the seam allowance from its edge, are flagged: the cloth solver's
              stretch and bend springs are irregular there, and a fold kinks or a crease starts at them.
              EXPERIMENTAL: the 'visible' rule is ours.
  flatten     can the piece be cut flat? Topologically it needs 2g + b - 1 cuts to become a disc (Erickson and
              Har-Peled, Optimally cutting a surface into a disk, 2004; Lazarus et al. 2001). Geometrically its
              Gaussian curvature must be near zero: K at a vertex is its angle defect, 2 pi minus the angles round it
              (pi minus them on the edge, the geodesic turn), and they sum to 2 pi chi (discrete Gauss-Bonnet: Crane,
              Discrete Differential Geometry: An Applied Introduction, 2018, 5.4; Meyer et al. 2003). A geodesic disc
              of radius r on a surface of curvature K has a circumference 2 pi r (1 - K r^2 / 6) (Bertrand-Diguet-
              Puiseux, 1848), so laying it flat strains its rim by K r^2 / 6 = (total curvature) / (6 pi), whatever
              its size. The piece is then laid flat by least-squares conformal maps (Levy et al. 2002) and every
              edge's flat length is compared with its length on the body (the p5 and p95, pattern.py's measure);
              that give against what the cloth can be shrunk or stretched by in pressing or lasting is the check.
              The limit is EXPERIMENTAL (FLATTEN_LIMIT; no source we found gives a strain: SATRA's lastometer reads
              leather by a ball's distension in mm).
              Where the curvature gathers is where a dart or a seam has to go: the spot is drawn.
  anchors     the parts graph: every part that is not an anchor must have a path to one (a floating part falls
              off). The first Betti number, b1 = E - V + C (the cycle rank of the graph: Diestel, Graph Theory, 1.9),
              counts the independent loops of holds: 0 means every hold is the only one on its path, so each is a
              single point of failure; a part named 'loaded' needs min_holds edge-disjoint paths to an anchor
              (Menger's theorem: the edge connectivity, found by augmenting paths), and a hold whose loss frees a
              part (a bridge: Tarjan 1974) is listed. A part with far more holds than it needs is reported, not
              failed.
  closure     buttons and buttonholes along a closing edge:
                top button at the lapel's break point, the break 0 to 1 cm above it (Mueller und Sohn's women's
                  jacket draft, as collar.py already uses); with no lapel, about 5/8 in (1.6 cm) below the neck and
                  the hole never nearer than 1/4 in (0.6 cm) to the neck edge (American Sewing Guild, 2019 and 2025)
                a button on each pressure point: the bust line (Threads 104, Louise Cutting: "the most important
                  button position is the one on the bust point line"; ASG; NMSU Guide C-231: a button at the bra line
                  prevents gaposis), and the waist of a fitted bodice. Within a quarter of the spacing: EXPERIMENTAL
                the top and bottom buttons placed first, the rest evenly between them (ASG): every gap within 10 %
                  of the mean (EXPERIMENTAL tolerance)
                spacing at most 3.5 in (8.9 cm) (NMSU C-231: "standard buttonhole placement is 2 to 3 1/2 inches
                  apart")
                the extension past the centre front at least 5/8 in for buttons to 7/8 in, and as wide in proportion
                  for larger (ASG); the hole the button's size plus 1/8 in, a horizontal hole starting 1/8 in past
                  the centre front toward the edge (ASG; NMSU C-231)
                no gap at a high-strain spot: where the strain along the edge (from the strain map, or the draft's
                  own ease deficit) peaks more than a quarter-spacing from a button and above the flow check's
                  strain limit, the edge gapes between the buttons there (EXPERIMENTAL threshold)

Run on a draft: run(draft) -> report; sheet(report, draft, path) -> the diagram sheet (PNG)."""
import json
import math

import numpy as np

VERSION = 1
IN = 2.54                          # cm in an inch
# EXPERIMENTAL: how far an edge may give (p5 or p95 of flat length over length on the body) when a piece is pressed
# or lasted to its shape. No source we found gives a strain (SATRA's lastometer reads leather's distension in mm).
# Leather is set at the approved boot's own shafts and vamp (3.6-4.0 %, flattened by pattern.py's unwrap), wool a
# little under it (shrunk and stretched with the iron), linen and silk at half (they take little shaping).
FLATTEN_LIMIT = {'leather': 0.05, 'wool': 0.04, 'linen': 0.02, 'silk': 0.02, '': 0.04}
STRAIN_LIMIT = 0.05                # the flow check's strain limit for a closing edge (rq_flow: worst strain 5 %); EXPERIMENTAL here
SOURCES = {
    'euler': 'Edelsbrunner & Harer, Computational Topology (2010) ch. II, IV; Botsch et al., Polygon Mesh Processing (2010) 1.3',
    'boundary': 'Edelsbrunner & Harer (2010); Botsch et al. (2010) 1.3',
    'seams': 'Botsch et al. (2010) 1.3: a 2-manifold edge has one or two faces',
    'mesh': 'Botsch et al. (2010) 1.3; orientability',
    'singular': 'Bommes et al., Quad-Mesh Generation and Processing: A Survey, CGF 2013, 2.1 (index sum = 4 chi)',
    'flatten': 'Erickson & Har-Peled (2004) cut graph 2g+b-1; Crane, DDG (2018) 5.4 discrete Gauss-Bonnet; Bertrand-Diguet-Puiseux (1848)',
    'anchors': 'Diestel, Graph Theory, 1.9 (cycle rank); Menger; Tarjan (1974) bridges',
    'closure': 'ASG Button and Buttonhole Positioning (2019) and No More Gaping (2025); Threads 104 (L. Cutting); NMSU Guide C-231 (2019); Mueller und Sohn via collar.py',
}


# ------------------------------------------------------------------------------------------------ the mesh's topology

def _faces(F):
    return [list(map(int, f)) for f in F]


def half_edges(F):
    """Directed edges (a, b, face) of every polygon."""
    out = []
    for fi, f in enumerate(F):
        n = len(f)
        for k in range(n):
            out.append((f[k], f[(k + 1) % n], fi))
    return out


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[a] = b
        return a != b


def mesh_topology(V, F):
    """The counts and the faults of one polygon mesh: chi, components, boundary loops (vertex lists), genus per
    component, non-manifold edges, bow ties, flipped edges, degenerate faces, loose vertices, valences."""
    F = _faces(F)
    V = np.asarray(V, float)
    nV = len(V)
    he = half_edges(F)
    und = {}
    directed = {}
    for a, b, fi in he:
        und.setdefault((min(a, b), max(a, b)), []).append(fi)
        directed[(a, b)] = directed.get((a, b), 0) + 1
    E = len(und)
    used = np.zeros(nV, bool)
    for f in F:
        used[f] = True
    nonmanifold = [e for e, fs in und.items() if len(fs) > 2]
    flipped = [e for e, fs in und.items() if len(fs) == 2 and (directed.get(e, 0) == 2 or directed.get(e[::-1], 0) == 2)]
    bnd = [(a, b) for a, b, fi in he if len(und[(min(a, b), max(a, b))]) == 1]
    nxt = {}
    bow = set()
    for a, b in bnd:
        if a in nxt:
            bow.add(a)
        nxt.setdefault(a, []).append(b)
    loops, seen = [], set()
    for a, b in bnd:
        if (a, b) in seen:
            continue
        loop, cur = [a], (a, b)
        while cur not in seen:
            seen.add(cur)
            u, v = cur
            if v == loop[0]:
                break
            loop.append(v)
            cand = [w for w in nxt.get(v, []) if (v, w) not in seen]
            if not cand:
                break
            cur = (v, cand[0])
        loops.append(loop)
    uf = UF(len(F))
    for fs in und.values():
        for f in fs[1:]:
            uf.union(fs[0], f)
    comp_of_face = np.array([uf.find(i) for i in range(len(F))]) if F else np.zeros(0, int)
    comps = sorted(set(comp_of_face.tolist()))
    cidx = {c: i for i, c in enumerate(comps)}
    vcomp = np.full(nV, -1)
    for fi, f in enumerate(F):
        vcomp[f] = cidx[comp_of_face[fi]]
    deg = np.zeros(nV, int)
    for a, b in und:
        deg[a] += 1
        deg[b] += 1
    onb = np.zeros(nV, bool)
    for a, b in bnd:
        onb[a] = onb[b] = True
    degenerate = []
    for fi, f in enumerate(F):
        if len(set(f)) < len(f) or _poly_area(V[f]) < 1e-12:
            degenerate.append(fi)
    chi = int(used.sum()) - E + len(F)
    percomp = []
    for ci in range(len(comps)):
        vs = int(((vcomp == ci) & used).sum())
        fs = [f for fi, f in enumerate(F) if cidx[comp_of_face[fi]] == ci]
        es = sum(1 for e in und if vcomp[e[0]] == ci)
        b = sum(1 for lp in loops if vcomp[lp[0]] == ci)
        c_chi = vs - es + len(fs)
        percomp.append(dict(chi=c_chi, boundary=b, genus=(2 - c_chi - b) / 2.0, faces=len(fs), verts=vs))
    return dict(V=int(used.sum()), E=E, F=len(F), chi=chi, components=len(comps), percomp=percomp, loops=loops,
                nonmanifold=nonmanifold, bowtie=sorted(bow), flipped=flipped, degenerate=degenerate,
                loose=int((~used).sum()), deg=deg, onb=onb, und=und, vcomp=vcomp)


def _poly_area(P):
    P = np.asarray(P, float)
    if P.shape[1] == 2:
        P = np.c_[P, np.zeros(len(P))]
    c = np.zeros(3)
    for k in range(len(P)):
        c += np.cross(P[k], P[(k + 1) % len(P)])
    return 0.5 * float(np.linalg.norm(c))


def corner_angles(V, F):
    """The angle of every triangle corner, the polygons fanned into triangles (so each sums to pi exactly and the
    Gauss-Bonnet identity holds on the mesh as built). Returns (vertex index, angle, triangle area / 3) per corner."""
    V = np.asarray(V, float)
    if V.shape[1] == 2:
        V = np.c_[V, np.zeros(len(V))]
    T = []
    for f in _faces(F):
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1]))
    T = np.array(T, int).reshape(-1, 3)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]

    def ang(P, Q, R):
        u, v = Q - P, R - P
        return np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), np.einsum('ij,ij->i', u, v))
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    idx = np.r_[T[:, 0], T[:, 1], T[:, 2]]
    th = np.r_[ang(A, B, C), ang(B, C, A), ang(C, A, B)]
    return idx, th, np.r_[area, area, area] / 3.0


def gaussian(V, F, onb):
    """Angle defect per vertex (interior: 2 pi - sum; boundary: pi - sum, the geodesic turn), the vertex areas, and the
    Gauss-Bonnet total."""
    n = len(V)
    idx, th, ar = corner_angles(V, F)
    S = np.bincount(idx, th, n)
    Ar = np.bincount(idx, ar, n)
    used = np.bincount(idx, minlength=n) > 0
    d = np.where(onb, math.pi, 2 * math.pi) - S
    d[~used] = 0.0
    return d, Ar


# ------------------------------------------------------------------------------------------------ one piece

def _mask(m, n):
    m = np.asarray(m)
    if m.dtype == bool:
        return m
    out = np.zeros(n, bool)
    out[m.astype(int)] = True
    return out


def loop_runs(loop, labels):
    """A boundary loop cut into runs by label: [(label or None, [vertex, ...])]. A vertex carrying two labels (a
    corner where two seams meet) ends one run and starts the next."""
    n = len(loop)
    lab = []
    for v in loop:
        here = [k for k, m in labels.items() if m[v]]
        lab.append(here)
    if not n:
        return []
    # the run label per edge (v_i, v_i+1): a label both ends carry; else None
    el = []
    for i in range(n):
        a, b = lab[i], lab[(i + 1) % n]
        common = [k for k in a if k in b]
        el.append(common[0] if len(common) == 1 else (sorted(common)[0] if common else None))
    # rotate so a run starts at index 0
    start = 0
    for i in range(n):
        if el[i] != el[i - 1]:
            start = i
            break
    else:
        return [(el[0], list(loop) + [loop[0]])]
    runs, cur, verts = [], el[start], [loop[start]]
    for k in range(n):
        i = (start + k) % n
        j = (i + 1) % n
        if el[i] != cur:
            runs.append((cur, verts))
            cur, verts = el[i], [loop[i]]
        verts.append(loop[j])
    runs.append((cur, verts))
    return _absorb(runs, n)


FRAG = 6          # a run of this few points or fewer, at a corner, is the corner's labelling, not an edge of its own


def _absorb(runs, n):
    """Corner fragments taken back into the runs beside them: where two labels' masks overlap or leave a gap at a
    corner, the loop shows a run of a few points with another label or none between two runs. One between two of the
    same label joins them; an unlabelled one beside a labelled run joins it."""
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (lab, vs) in enumerate(runs):
            if len(vs) > FRAG or len(runs) < 2:
                continue
            pl, nl = runs[i - 1][0], runs[(i + 1) % len(runs)][0]
            if len(runs) >= 3 and pl == nl and pl is not None:
                j = (i + 1) % len(runs)
                merged = runs[i - 1][1] + vs[1:] + runs[j][1][1:]
                keep = [r for k, r in enumerate(runs) if k not in (i - 1 if i else len(runs) - 1, i, j)]
                at = min(i - 1, j) if i else 0
                runs = keep[:max(0, at)] + [(pl, merged)] + keep[max(0, at):]
                changed = True
                break
            if lab is None and (pl is not None or nl is not None):
                k = i - 1 if pl is not None else (i + 1) % len(runs)
                if k == i - 1 or (i == 0 and k == len(runs) - 1):
                    runs[k] = (runs[k][0], runs[k][1] + vs[1:])
                else:
                    runs[k] = (runs[k][0], vs[:-1] + runs[k][1])
                del runs[i]
                changed = True
                break
    return runs


def _arc_len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def main_component(p, share=0.1):
    """The piece without its stray islands (components with under `share` of its faces): (piece, faces dropped)."""
    F = _faces(p['F'])
    T = mesh_topology(p['V'], F)
    if T['components'] < 2:
        return p, 0
    sizes = [c['faces'] for c in T['percomp']]
    big = int(np.argmax(sizes))
    if any(sz >= share * sum(sizes) for i, sz in enumerate(sizes) if i != big):
        return p, 0
    keep = [f for f in F if T['vcomp'][f[0]] == big]
    return dict(p, F=keep), len(F) - len(keep)


def piece_report(p, flatten=True):
    V = np.asarray(p['V'], float)
    F = _faces(p['F'])
    n = len(V)
    T = mesh_topology(V, F)
    labels = {k: _mask(m, n) for k, m in (p.get('edges') or {}).items()}
    loops = []
    for lp in T['loops']:
        runs = loop_runs(lp, labels)
        loops.append(dict(verts=lp, length=_arc_len(V[lp + lp[:1]]), runs=[
            dict(label=lab, verts=vs, length=_arc_len(V[vs])) for lab, vs in runs]))
    # singular points
    quad = sum(len(f) == 4 for f in F) >= sum(len(f) == 3 for f in F)
    reg_i, reg_b = (4, 3) if quad else (6, 4)
    deg, onb = T['deg'], T['onb']
    used = np.zeros(n, bool)
    for f in F:
        used[f] = True
    interior = used & ~onb
    idx_sum = int(np.sum(reg_i - deg[interior]) + np.sum(reg_b - deg[used & onb]))
    pure = all(len(f) == (4 if quad else 3) for f in F)
    # how far each vertex is from the cut edge, in rings (breadth first over the edges), and in units
    ring = _rings(T['und'], onb, n)
    el = np.median([np.linalg.norm(V[a] - V[b]) for a, b in list(T['und'])[:2000]]) if T['und'] else 1.0
    allow = float(p.get('allowance', 0.0))
    solid = bool(p.get('solid'))
    sing = np.nonzero(interior & (deg != reg_i))[0] if not solid else np.zeros(0, int)
    # a singular point next to one of the opposite index (a 3 beside a 5 on a quad grid: a row of the grid ending,
    # a dislocation) adds nothing to the index sum and is a fault of the grid, not of the surface's topology; the rest
    # are the true singular points (Bommes et al. 2013: irregular vertices; a 3-5 pair is a dislocation)
    nbr = {}
    for a, b in T['und']:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    sset = set(sing.tolist())
    paired = set()
    for v in sing:
        if v in paired:
            continue
        iv = reg_i - deg[v]
        mate = next((w for w in nbr.get(v, []) if w in sset and w not in paired and (reg_i - deg[w]) == -iv), None)
        if mate is not None:
            paired.update((int(v), int(mate)))
    lone = [int(v) for v in sing if v not in paired]
    visible = bool(p.get('visible', True))
    vis = [v for v in lone if visible and ring[v] * el > max(allow, 2 * el)]
    # curvature, flattening
    d, Ar = gaussian(V, F, onb)
    Kint = float(d[interior].sum())
    kg = float(d[used & onb].sum())
    gb = Kint + kg - 2 * math.pi * T['chi']
    absK = float(np.abs(d[interior]).sum())
    area = float(Ar.sum())
    flat = solid or (p.get('flat') is not None and p.get('flat') is not False) or (V.shape[1] == 2) or float(np.ptp(V[used], axis=0).min() if used.any() else 0) < 1e-9
    patch = _worst_patch(V, d, interior, area) if not flat else (0.0, None)
    flat_pc, UV = (1.0, 1.0, 1.0), None
    if flatten and not flat and len(F) > 2:
        try:
            Fl = F
            if T['components'] > 1:           # (in pieces: the largest laid flat, the rest are the euler check's fault)
                big = int(np.argmax([c['faces'] for c in T['percomp']]))
                Fl = [f for f in F if T['vcomp'][f[0]] == big]
            UV, flat_pc = lscm(V, Fl, onb)
        except Exception:               # (a piece too broken to lay flat: the mesh check says why)
            flat_pc = (float('nan'),) * 3
    give = max(flat_pc[2] - 1.0, 1.0 - flat_pc[0]) if np.isfinite(flat_pc[0]) else float('nan')
    c0 = T['percomp'][0] if T['percomp'] else dict(genus=0, boundary=0)
    cuts = int(round(2 * c0['genus'] + max(c0['boundary'], 1) - 1)) if T['components'] == 1 else None    # (a closed surface: 2g loops; with b boundaries 2g + b - 1 arcs)
    return dict(name=p['name'], V=T['V'], E=T['E'], F=T['F'], chi=T['chi'], components=T['components'],
                percomp=T['percomp'], boundary=len(T['loops']), loops=loops, nonmanifold=len(T['nonmanifold']),
                bowtie=len(T['bowtie']), flipped=len(T['flipped']), degenerate=len(T['degenerate']), loose=T['loose'],
                mesh='quad' if quad else 'tri', regular=(reg_i, reg_b), pure=pure, index_sum=idx_sum,
                index_expected=(4 if quad else 6) * T['chi'], singular=lone, singular_visible=vis, dislocations=len(paired) // 2, solid=solid,
                singular_hist={int(k): int(c) for k, c in zip(*np.unique(deg[lone], return_counts=True))} if len(lone) else {},
                K_interior=Kint, kg_boundary=kg, gauss_bonnet_residual=gb, absK=absK, area=area,
                flatten_strain_total=absK / (6 * math.pi), flatten_strain_patch=patch[0], dart_spot=patch[1],
                flat_lengths=flat_pc, flatten_give=give, UV=UV,
                cuts_to_disc=cuts, flat=flat, material=p.get('material', ''), labels=sorted(labels),
                openings=list(p.get('openings', [])), edge_len=float(el), defect=d)


def lscm(V, F, onb=None, rounds=25):
    """The piece laid flat by least-squares conformal maps (Levy et al., SIGGRAPH 2002), then relaxed as-rigid-as-
    possible (Liu et al., A Local/Global Approach to Mesh Parameterization, SGP 2008: the least stretch, as Blender's
    minimum-stretch unwrap that pattern.py uses), scaled to its own area: the
    flat points (n, 2), and every edge's flat length over its length on the body as the 5th, 50th and 95th
    percentiles (pattern.py's own measure of how much the cloth has to give)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    V = np.asarray(V, float)
    T = []
    for f in _faces(F):
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1]))
    T = np.array(T, int)
    n = len(V)
    P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1 = P1 - P0
    nrm = np.cross(e1, P2 - P0)
    dbl = np.linalg.norm(nrm, axis=1)
    ok = dbl > 1e-12
    T, P0, P1, P2, e1, nrm, dbl = T[ok], P0[ok], P1[ok], P2[ok], e1[ok], nrm[ok], dbl[ok]
    ex = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    ey = np.cross(nrm / dbl[:, None], ex)
    loc = [np.c_[((P - P0) * ex).sum(1), ((P - P0) * ey).sum(1)] for P in (P0, P1, P2)]
    s = 1.0 / np.sqrt(dbl)
    # W_j = (x_k - x_i) + i (y_k - y_i) for the corner j opposite (i, k), over sqrt(2 A)
    rows, cols, vals = [], [], []
    m = len(T)
    for j, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        wr = (loc[b][:, 0] - loc[a][:, 0]) * s
        wi = (loc[b][:, 1] - loc[a][:, 1]) * s
        vj = T[:, j]
        t = np.arange(m)
        # real part: wr*u - wi*v ; imaginary: wi*u + wr*v
        rows += [t, t, m + t, m + t]
        cols += [vj, n + vj, vj, n + vj]
        vals += [wr, -wi, wi, wr]
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * m, 2 * n)).tocsc()
    # two pins: the two boundary points farthest apart (or any two)
    inT = np.zeros(n, bool)
    inT[T.ravel()] = True
    cand = np.nonzero(onb & inT)[0] if onb is not None and np.any(onb & inT) else np.unique(T)
    i0 = cand[0]
    i1 = cand[int(np.argmax(np.linalg.norm(V[cand] - V[i0], axis=1)))]
    i0 = cand[int(np.argmax(np.linalg.norm(V[cand] - V[i1], axis=1)))]
    L = float(np.linalg.norm(V[i1] - V[i0]))
    pin = {i0: (0.0, 0.0), i1: (L, 0.0)}
    free = np.ones(2 * n, bool)
    x = np.zeros(2 * n)
    for i, (u, v) in pin.items():
        free[i] = free[n + i] = False
        x[i], x[n + i] = u, v
    b = -A[:, ~free] @ x[~free]
    sol = lsqr(A[:, free], b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    x[free] = sol
    UV = np.c_[x[:n], x[n:]]
    UV = _arap(T, loc, UV, n, rounds)
    a3 = 0.5 * dbl.sum()
    q0, q1, q2 = UV[T[:, 0]], UV[T[:, 1]], UV[T[:, 2]]
    a2 = 0.5 * np.abs((q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) - (q2[:, 0] - q0[:, 0]) * (q1[:, 1] - q0[:, 1])).sum()
    UV *= math.sqrt(a3 / max(a2, 1e-12))
    E = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1), axis=0)
    l3 = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    l2 = np.linalg.norm(UV[E[:, 0]] - UV[E[:, 1]], axis=1)
    r = l2 / np.maximum(l3, 1e-12)
    return UV, tuple(float(v) for v in np.percentile(r, (5, 50, 95)))


def _arap(T, loc, UV, n, rounds):
    """Local/global ARAP parameterization (Liu et al. 2008) from the start UV: each triangle's best rotation, then the
    cotangent-weighted solve for the points."""
    if rounds <= 0:
        return UV
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import factorized
    X = np.stack(loc, 1)                          # (m, 3, 2) the triangle in its own plane
    m = len(T)
    cot = np.zeros((m, 3))
    for j in range(3):
        a, b = X[:, (j + 1) % 3] - X[:, j], X[:, (j + 2) % 3] - X[:, j]
        cr = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        cot[:, j] = (a * b).sum(1) / np.maximum(np.abs(cr), 1e-12)     # (the cot of the angle at corner j: weights edge opposite)
    cot = np.clip(cot, -1e3, 1e3)
    I, J, W = [], [], []
    for j in range(3):
        i, k = T[:, (j + 1) % 3], T[:, (j + 2) % 3]
        w = 0.5 * cot[:, j]
        I += [i, k, i, k]
        J += [k, i, i, k]
        W += [-w, -w, w, w]
    Lm = coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsc()
    used = np.unique(T)
    idle = np.setdiff1d(np.arange(n), used)          # (points no triangle uses: kept where they are)
    pin0 = used[:1]
    Lm = Lm + coo_matrix((np.r_[np.ones(len(idle)), np.full(1, 1e-8 * float(abs(Lm).max()))],
                          (np.r_[idle, pin0], np.r_[idle, pin0])), shape=(n, n))   # (and one point pinned against drift)
    solve = factorized(Lm.tocsc())
    for _ in range(rounds):
        U = UV[T]                                    # (m, 3, 2)
        S = np.zeros((m, 2, 2))
        for j in range(3):
            i2, k2 = (j + 1) % 3, (j + 2) % 3
            du = U[:, i2] - U[:, k2]
            dx = X[:, i2] - X[:, k2]
            S += cot[:, j, None, None] * np.einsum('ti,tj->tij', du, dx)
        u_, sg, vt = np.linalg.svd(S)
        Rm = np.einsum('tij,tjk->tik', u_, vt)
        flip = np.linalg.det(Rm) < 0
        u_[flip, :, 1] *= -1
        Rm = np.einsum('tij,tjk->tik', u_, vt)
        b = np.zeros((n, 2))
        for j in range(3):
            i2, k2 = (j + 1) % 3, (j + 2) % 3
            dx = X[:, i2] - X[:, k2]
            r = 0.5 * cot[:, j, None] * np.einsum('tij,tj->ti', Rm, dx)
            np.add.at(b, T[:, i2], r)
            np.add.at(b, T[:, k2], -r)
        new = np.c_[solve(b[:, 0]), solve(b[:, 1])]
        UV = np.where(np.isin(np.arange(n), used)[:, None], new, UV)
    return UV


def _rings(und, onb, n):
    nb = [[] for _ in range(n)]
    for a, b in und:
        nb[a].append(b)
        nb[b].append(a)
    r = np.full(n, 10 ** 6)
    frontier = list(np.nonzero(onb)[0])
    r[frontier] = 0
    k = 0
    while frontier:
        k += 1
        new = []
        for v in frontier:
            for w in nb[v]:
                if r[w] > k:
                    r[w] = k
                    new.append(w)
        frontier = new
    return r


def _worst_patch(V, d, interior, area):
    """The worst patch's flattening strain: total |K| over a disc a quarter of the piece across, / (6 pi)
    (Bertrand-Diguet-Puiseux). Returns (strain, centre point)."""
    P = V[interior]
    if len(P) < 4:
        return 0.0, None
    R = 0.25 * math.sqrt(area / math.pi) * 2
    from scipy.spatial import cKDTree
    t = cKDTree(P)
    di = d[interior]
    step = max(1, len(P) // 400)
    best, at = 0.0, None
    for i in range(0, len(P), step):
        nb = t.query_ball_point(P[i], R)
        s = abs(float(di[nb].sum()))          # (the net curvature of the patch: a saddle and a dome beside each other cancel, as they do in cloth)
        if s > best:
            best, at = s, P[i].tolist()
    return best / (6 * math.pi), at


# ------------------------------------------------------------------------------------------------ the sewn garment

def assemble(pieces, reps, seams, reach=None):
    """The pieces glued along their seam runs, as a cell complex: each boundary loop is cut into corners (0-cells)
    and runs (1-cells); a seam identifies a run on one piece with a run on another, end to end. Returns chi, the
    boundary loops left (as lists of run labels), the components, the pairing faults."""
    runs = []          # (piece, label, start corner id, end corner id, length, endpoints xyz)
    corners = []
    for pi, (p, r) in enumerate(zip(pieces, reps)):
        V = np.asarray(p['V'], float)
        if V.shape[1] == 2:
            V = np.c_[V, np.zeros(len(V))]
        for lp in r['loops']:
            rr = lp['runs']
            if len(rr) == 1 and rr[0]['verts'][0] == rr[0]['verts'][-1]:
                c = len(corners)
                corners.append(V[rr[0]['verts'][0]])
                runs.append(dict(piece=pi, label=rr[0]['label'], a=c, b=c, length=rr[0]['length'],
                                 A=V[rr[0]['verts'][0]], B=V[rr[0]['verts'][-1]], closed=True))
                continue
            first = len(corners)
            for k, run in enumerate(rr):
                corners.append(V[run['verts'][0]])
            for k, run in enumerate(rr):
                a = first + k
                b = first + (k + 1) % len(rr)
                runs.append(dict(piece=pi, label=run['label'], a=a, b=b, length=run['length'],
                                 A=V[run['verts'][0]], B=V[run['verts'][-1]], closed=False))
    names = [p['name'] for p in pieces]
    # which seams: given, or every label on exactly two pieces
    if seams is None:
        by = {}
        for k, ru in enumerate(runs):
            if ru['label'] is not None:
                by.setdefault(ru['label'], set()).add(ru['piece'])
        seams = [[lab, names[min(ps)], names[max(ps)]] for lab, ps in by.items() if len(ps) == 2]
    faults = []
    glued = {}
    pairs = []
    laps = []
    opening_labels = set()
    own_open = {}
    for pi, p in enumerate(pieces):
        opening_labels |= set(p.get('openings', []))
        own_open[pi] = {str(x) for x in p.get('openings', [])}
    for s in seams:
        lab, A, B = s[0], s[1], s[2]
        labB = s[3] if len(s) > 3 and s[3] is not None else lab
        ia, ib = names.index(A), names.index(B)
        if len(s) > 4 and s[4] == 'lap':
            # a lapped seam: A's edge run stitched onto B's face (the tongue under the panels): one arc glued into the
            # inside of B, chi - 1, A's run no longer an edge, B keeps its edges
            ra = [k for k, ru in enumerate(runs) if ru['piece'] == ia and ru['label'] == lab]
            if not ra:
                faults.append(dict(seam=lab, a=A, b=B, runs_a=0, runs_b=0, why='the lapped seam has no run on %s' % A))
                continue
            for k in ra:
                glued[k] = glued.get(k, 0) + 1
                laps.append((k, ib, lab))
            continue
        ra = [k for k, ru in enumerate(runs) if ru['piece'] == ia and ru['label'] == lab]
        rb = [k for k, ru in enumerate(runs) if ru['piece'] == ib and ru['label'] == labB]
        if A == B and lab == labB:      # (a piece sewn to itself: a tube's back seam)
            if len(ra) == 2:
                ra, rb = [ra[0]], [ra[1]]
            else:
                ra, rb = ra, []
        if len(ra) != 1 or len(rb) != 1:
            faults.append(dict(seam=lab, a=A, b=B, runs_a=len(ra), runs_b=len(rb),
                               why=('the seam has %d run(s) on %s and %d on %s: each side must be one unbroken run'
                                    % (len(ra), A, len(rb), B))))
            if not ra or not rb:
                continue
        # pair the runs: the longest on each side (a broken seam is reported above)
        ka = max(ra, key=lambda k: runs[k]['length'])
        kb = max(rb, key=lambda k: runs[k]['length'])
        for k in (ka, kb):
            glued[k] = glued.get(k, 0) + 1
        pairs.append((ka, kb, lab))
    for k, c in glued.items():
        if c > 1:
            faults.append(dict(seam=runs[k]['label'], a=names[runs[k]['piece']], why='a run sewn to %d partners (non-manifold)' % c))
    # corners: union by gluing, each run's ends matched end to end by position (right sides together: A to the nearer end)
    # a lapped run lies on its target's face along an arc; arcs on one target that meet (a welt round the sole made of
    # several pieces' edges) share their ends there. Those ends are new points in the target (a subdivision of it,
    # chi unchanged): clustered by position within reach, each lapped run's ends are glued to them, and the run to its
    # own arc (+1 per run): one arc alone takes chi down by 1, a closed chain of them by 0 (chi of a circle).
    if reach is None:
        allV = np.concatenate([np.asarray(p['V'], float)[:, :2] if np.asarray(p['V']).shape[1] == 2 else np.asarray(p['V'], float)[:, :3][:, :2] for p in pieces]) if pieces else np.zeros((1, 2))
        reach = 0.02 * float(np.ptp(allV, axis=0).max() or 1.0)
    lap_nodes = {}
    base = len(corners)
    extra = []
    for ka, ib, lab in laps:
        for end, pos in (('a', runs[ka]['A']), ('b', runs[ka]['B'])):
            lst = lap_nodes.setdefault(ib, [])
            hit = next((j for j, q in lst if np.linalg.norm(q - pos) < reach), None)
            if hit is None:
                hit = base + len(extra)
                extra.append(pos)
                lst.append((hit, pos))
            runs[ka]['lap_' + end] = hit
    corners = list(corners) + extra
    uf = UF(len(corners))
    for ka, ib, lab in laps:
        uf.union(runs[ka]['a'], runs[ka]['lap_a'])
        uf.union(runs[ka]['b'], runs[ka]['lap_b'])
    for ka, kb, lab in pairs:
        ra, rb = runs[ka], runs[kb]
        if ra['closed'] or rb['closed']:
            uf.union(ra['a'], rb['a'])
            continue
        direct = np.linalg.norm(ra['A'] - rb['A']) + np.linalg.norm(ra['B'] - rb['B'])
        cross = np.linalg.norm(ra['A'] - rb['B']) + np.linalg.norm(ra['B'] - rb['A'])
        if direct <= cross:
            uf.union(ra['a'], rb['a'])
            uf.union(ra['b'], rb['b'])
        else:
            uf.union(ra['a'], rb['b'])
            uf.union(ra['b'], rb['a'])
    ccls = {uf.find(i) for i in range(len(corners))}
    chi0 = sum(r['chi'] for r in reps)
    chi = chi0 - (len(corners) - len(ccls)) + len(pairs) + len(laps)
    # components: pieces joined by seams
    pu = UF(len(pieces))
    for ka, kb, lab in pairs:
        pu.union(runs[ka]['piece'], runs[kb]['piece'])
    for ka, ib, lab in laps:
        pu.union(runs[ka]['piece'], ib)
    ncomp = len({pu.find(i) for i in range(len(pieces))})
    # the boundary left: unglued runs, joined end to end at the corner classes
    free = [k for k in range(len(runs)) if k not in glued]
    adj = {}
    for k in free:
        a, b = uf.find(runs[k]['a']), uf.find(runs[k]['b'])
        adj.setdefault(a, []).append(k)
        adj.setdefault(b, []).append(k)
    pinch = [c for c, ks in adj.items() if len(ks) > 2]
    loops, used = [], set()
    for k in free:
        if k in used:
            continue
        stack, comp = [k], []
        while stack:
            q = stack.pop()
            if q in used:
                continue
            used.add(q)
            comp.append(q)
            for c in (uf.find(runs[q]['a']), uf.find(runs[q]['b'])):
                stack += [w for w in adj.get(c, []) if w not in used]
        labs = [(names[runs[q]['piece']], runs[q]['label']) for q in comp]
        L = sum(runs[q]['length'] for q in comp)
        loops.append(dict(runs=labs, length=L))
    b = len(loops)
    genus = (2 * ncomp - chi - b) / 2.0
    mism = []
    for ka, kb, lab in pairs:
        la, lb = runs[ka]['length'], runs[kb]['length']
        mism.append(dict(seam=lab, a=names[runs[ka]['piece']], b=names[runs[kb]['piece']], len_a=la, len_b=lb,
                         ratio=(max(la, lb) / max(min(la, lb), 1e-9))))
    unpaired = sorted({(names[runs[k]['piece']], runs[k]['label']) for k in free
                       if runs[k]['label'] is not None and str(runs[k]['label']) not in own_open[runs[k]['piece']]},
                      key=lambda x: (x[0], str(x[1])))
    return dict(chi=chi, components=ncomp, boundary=b, genus=genus, loops=loops, faults=faults, pinch=len(pinch),
                seams=[list(s) for s in seams], pairs=len(pairs), laps=len(laps), lengths=mism, unpaired=unpaired,
                corners=len(corners), corner_classes=len(ccls), runs=len(runs))


# ------------------------------------------------------------------------------------------------ the parts graph

def anchor_graph(G):
    """Every free part's path to an anchor, the floating parts, b1, bridges, and the edge connectivity of loaded parts."""
    nodes = list(G['nodes'])
    ix = {n: i for i, n in enumerate(nodes)}
    anchors = set(G.get('anchors', []))
    captured = set(G.get('captured', []))
    # a hold may be sliding (a cord through an eyelet, a strand through a loop): it holds only a part that cannot slide
    # out of it, one closed on itself (a knotted lace, a looped strand: 'captured')
    allh = [h for h in G['holds'] if h[0] in ix and h[1] in ix]
    fixed = [h for h in allh if not (len(h) > 3 and h[3] == 'sliding') or h[0] in captured or h[1] in captured]
    holds = [(ix[h[0]], ix[h[1]], h[2]) for h in fixed]
    n = len(nodes)

    def ungrounded(H):
        uf = UF(n)
        for a, b, k in H:
            uf.union(a, b)
        comps = {}
        for i in range(n):
            comps.setdefault(uf.find(i), []).append(i)
        return [nodes[i] for c, ms in comps.items() if not any(nodes[m] in anchors for m in ms) for i in ms]
    floating = ungrounded(holds)
    loose_all = set(ungrounded([(ix[h[0]], ix[h[1]], h[2]) for h in allh]))
    friction = [x for x in floating if x not in loose_all]      # (held only by sliding holds: it pulls out)
    # b1 with all anchors identified as one ground node (they are all held by the same body)
    g = n + 1
    gi = lambda i: n if nodes[i] in anchors else i      # noqa: E731
    E = [(gi(a), gi(b), k) for a, b, k in holds if gi(a) != gi(b)]
    uf2 = UF(g)
    for a, b, k in E:
        uf2.union(a, b)
    present = {gi(i) for i in range(n)}
    C = len({uf2.find(i) for i in present})
    b1 = len(E) - len(present) + C
    bridges = _bridges(g, E)
    need = int(G.get('min_holds', 2))
    loaded = []
    for name in G.get('loaded', []):
        if name not in ix or name in anchors:
            continue
        k = _edge_conn(g, E, gi(ix[name]), n)
        loaded.append(dict(part=name, paths=k, need=need))
    # what hangs on each bridge: the parts cut from the ground if it fails
    hang = []
    for (a, b, kind) in bridges:
        side = _side(g, E, (a, b), n)
        hang.append(dict(hold=kind, between=[nodes[a] if a < n else 'anchor', nodes[b] if b < n else 'anchor'],
                         frees=[nodes[i] for i in side if i < n]))
    eu = set(G.get('expect_used', []))      # (holds that must carry something: an eyelet set in a panel and nothing through it)
    unused = [x for x in nodes if x in eu and sum(1 for h in allh if x in h[:2]) < 2]
    return dict(nodes=n, holds=len(holds), anchors=sorted(anchors), floating=floating, friction=friction, unused=unused, b1=b1, components=C,
                bridges=hang, loaded=loaded)


def _bridges(n, E):
    adj = [[] for _ in range(n)]
    for k, (a, b, kind) in enumerate(E):
        adj[a].append((b, k))
        adj[b].append((a, k))
    disc, low, out, t = [-1] * n, [0] * n, [], [0]
    import sys
    sys.setrecursionlimit(max(10000, 4 * n))

    def dfs(u, pe):
        disc[u] = low[u] = t[0]
        t[0] += 1
        for v, k in adj[u]:
            if k == pe:
                continue
            if disc[v] < 0:
                dfs(v, k)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    out.append(E[k])
            else:
                low[u] = min(low[u], disc[v])
    for u in range(n):
        if disc[u] < 0 and adj[u]:
            dfs(u, -1)
    return out


def _side(n, E, cut, ground):
    a, b = cut
    adj = [[] for _ in range(n)]
    removed = False
    for (x, y, k) in E:
        if not removed and {x, y} == {a, b}:
            removed = True
            continue
        adj[x].append(y)
        adj[y].append(x)
    seen, st = {ground}, [ground]
    while st:
        u = st.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                st.append(v)
    return [i for i in {a, b} if i not in seen] and _reach(adj, [i for i in (a, b) if i not in seen][0])


def _reach(adj, s):
    seen, st = {s}, [s]
    while st:
        u = st.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                st.append(v)
    return sorted(seen)


def _edge_conn(n, E, s, t):
    """Edge-disjoint paths from s to t (unit capacities both ways: Edmonds-Karp)."""
    from collections import deque
    cap = {}
    adj = [set() for _ in range(n)]
    for a, b, k in E:
        cap[(a, b)] = cap.get((a, b), 0) + 1
        cap[(b, a)] = cap.get((b, a), 0) + 1
        adj[a].add(b)
        adj[b].add(a)
    flow = 0
    while True:
        par = {s: None}
        q = deque([s])
        while q and t not in par:
            u = q.popleft()
            for v in adj[u]:
                if v not in par and cap.get((u, v), 0) > 0:
                    par[v] = u
                    q.append(v)
        if t not in par:
            return flow
        v = t
        while par[v] is not None:
            u = par[v]
            cap[(u, v)] -= 1
            cap[(v, u)] = cap.get((v, u), 0) + 1
            v = u
        flow += 1


# ------------------------------------------------------------------------------------------------ closures

def closure_report(c, unit_cm):
    """The buttons along one closing edge against the rules (see the top of the file). Lengths in cm."""
    P = np.asarray(c['edge'], float)
    s_edge = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] * unit_cm
    L = float(s_edge[-1])
    u = unit_cm
    b = np.sort(np.asarray(c.get('buttons', []), float)) * u
    d = float(c.get('button_d', 0.0)) * u
    checks = []

    def add(cid, ok, value, limit, why, src, exp=False):
        checks.append(dict(id=cid, ok=bool(ok), value=value, limit=limit, why=why, source=src, experimental=exp))
    if len(b) == 0:
        add('buttons', False, 0, '>= 1', 'no buttons on the closing edge', 'ASG')
        return dict(name=c['name'], length=L, buttons=[], checks=checks, s_edge=s_edge.tolist())
    gaps = np.diff(b)
    mean = float(gaps.mean()) if len(gaps) else 0.0
    if c.get('break_s') is not None:
        br = float(c['break_s']) * u
        off = float(b[0] - br)              # (the top button this far below the break; the break 0 to 1 cm above it)
        add('top at break', 0.0 <= off <= 1.0, round(off, 2), '0 to 1 cm below the break',
            'top button %.1f cm %s the lapel\'s break point' % (abs(off), 'below' if off >= 0 else 'above')
            + ('' if 0 <= off <= 1 else (': a button above the break holds the lapel shut across its roll'
                                          if off < 0 else ': the opening gapes between the break and the first button')),
            'Mueller und Sohn women\'s jacket draft (via collar.py): break 0-1 cm above the top button')
        above = int((b < br - 0.01).sum())
        add('none above break', above == 0, above, 0, '%d button(s) above the break, on the cloth the lapel turns back' % above,
            'tailoring: the lapel rolls from the break; a button above it holds the roll shut')
    elif c.get('neck_s') is not None:
        nk = float(c['neck_s']) * u
        off = float(b[0] - nk)
        add('top below neck', off >= 0.6 + 0.5 * d and abs(off - 5 / 8 * IN) <= 0.5 * max(mean, 1.0), round(off, 2),
            'about 1.6 cm, hole >= 0.6 cm below the neck edge', 'top button %.1f cm below the neck edge' % off,
            'ASG (2019, 2025): 5/8 in down, never nearer than 1/4 in')
    br_cm = float(c['break_s']) * u if c.get('break_s') is not None else None
    for name, s in (c.get('pressure') or {}).items():
        s = float(s) * u
        if br_cm is not None and s < br_cm:
            add('on %s' % name, True, None, None, 'the %s line is %.1f cm above the break, inside the lapels\' opening: '
                'nothing to button there' % (name, br_cm - s), 'the rule applies to the closed part of the front', exp=True)
            continue
        k = int(np.argmin(np.abs(b - s)))
        off = float(b[k] - s)
        tol = 0.25 * (mean if mean else 3.0)
        add('on %s' % name, abs(off) <= tol, round(off, 2), '|off| <= %.1f cm (a quarter-spacing)' % tol,
            'nearest button %.1f cm %s the %s line' % (abs(off), 'below' if off > 0 else 'above', name),
            'Threads 104 (L. Cutting): bust point line; ASG: a button on each pressure point; NMSU C-231: bra line',
            exp=True)
    if len(gaps) >= 2:
        dev = float(np.max(np.abs(gaps - mean)) / mean)
        add('even spacing', dev <= 0.10, round(100 * dev, 1), '<= 10 %',
            'gaps %s cm: the worst %.0f %% off the mean %.2f cm' % (', '.join('%.2f' % g for g in gaps), 100 * dev, mean),
            'ASG: top and bottom first, the rest evenly between (tolerance EXPERIMENTAL)', exp=True)
    if len(gaps):
        add('max spacing', gaps.max() <= 3.5 * IN, round(float(gaps.max()), 2), '<= 8.9 cm (3 1/2 in)',
            'widest gap %.2f cm' % gaps.max(), 'NMSU C-231: standard placement 2 to 3 1/2 in apart')
    if c.get('end_s') is not None:
        es = float(c['end_s']) * u
        off = float(es - b[-1])
        add('bottom near end', 0 <= off <= max(mean, 1.0), round(off, 2), '0 to one spacing (%.1f cm) above the closing\'s end' % mean,
            'the last button %.1f cm above where the closing edge ends' % off,
            'ASG: the bottom button placed first, at the closing\'s end (one-spacing tolerance EXPERIMENTAL)', exp=True)
    if c.get('extension') is not None and d:
        ext = float(c['extension']) * u
        need = 5 / 8 * IN if d <= 7 / 8 * IN else 5 / 8 * IN * d / (7 / 8 * IN)
        add('extension', ext >= need - 1e-6, round(ext, 2), '>= %.2f cm for a %.2f cm button' % (need, d),
            'the closing laps %.2f cm past the centre front' % ext, 'ASG: 5/8 in for buttons to 7/8 in')
    if c.get('hole_len') is not None and d:
        hl = float(c['hole_len']) * u
        need = d + IN / 8
        add('hole length', hl >= need - 0.05, round(hl, 2), '>= %.2f cm (button + 1/8 in)' % need,
            'buttonhole %.2f cm for a %.2f cm button' % (hl, d), 'ASG: the hole the button size plus 1/8 in')
    if c.get('strain') is not None and len(b):
        S = np.asarray(c['strain'], float)
        ss, sv = S[:, 0] * u, S[:, 1]
        bad = []
        for k in range(len(b) + 1):
            lo = b[k - 1] if k else 0.0
            hi = b[k] if k < len(b) else (float(c.get('end_s')) * u if c.get('end_s') is not None else L)
            m = (ss > lo) & (ss < hi)
            if not m.any():
                continue
            j = np.argmax(sv[m])
            s_pk, v_pk = float(ss[m][j]), float(sv[m][j])
            dist = min(abs(s_pk - lo) if k else 1e9, abs(hi - s_pk))
            q = 0.25 * (mean if mean else 3.0)
            if v_pk > STRAIN_LIMIT and dist > q:
                bad.append(dict(s=round(s_pk, 2), strain=round(v_pk, 3), from_button=round(dist, 2)))
        add('strain gaps', not bad, len(bad), 'no strain peak > %.0f %% more than a quarter-spacing from a button' % (100 * STRAIN_LIMIT),
            ('gaps at %s' % '; '.join('%.1f cm (strain %.0f %%, %.1f cm from a button)' % (x['s'], 100 * x['strain'], x['from_button']) for x in bad))
            if bad else 'every strain peak sits on or near a button',
            'NMSU C-231 (gaposis between buttons at the bust); threshold EXPERIMENTAL', exp=True)
    return dict(name=c['name'], length=L, buttons=b.tolist(), gaps=gaps.tolist(), checks=checks, s_edge=s_edge.tolist())


# ------------------------------------------------------------------------------------------------ the whole check

def run(draft):
    """All the checks on a draft. Returns the report: per piece, the assembly, the parts, the closures, and a flat
    list of checks, each {id, scope, ok, value, limit, why, source, experimental}."""
    pieces = draft.get('pieces', [])
    reps = [piece_report(p) for p in pieces]
    exp = draft.get('expected', {})
    checks = []

    def add(cid, scope, ok, value, limit, why, src, experimental=False, level='fail'):
        checks.append(dict(id=cid, scope=scope, ok=bool(ok), level='pass' if ok else level, value=value, limit=limit,
                           why=why, source=src, experimental=experimental))
    for p, r in zip(pieces, reps):
        nm = r['name']
        want = exp.get('pieces_chi', {}).get(nm, 1) if isinstance(exp.get('pieces_chi'), dict) else exp.get('pieces_chi', 1)
        c0 = r['percomp'][0] if r['percomp'] else {}
        add('euler', nm, r['chi'] == want and r['components'] == 1, r['chi'], want,
            'chi %d (V %d - E %d + F %d), %d component(s), %d boundary loop(s), genus %g%s' % (
                r['chi'], r['V'], r['E'], r['F'], r['components'], r['boundary'], c0.get('genus', 0),
                '' if r['chi'] == want else ': a cut piece should be one disc (chi 1); %s' % (
                    'it has a hole cut in it' if r['chi'] < want and r['boundary'] > 1 else
                    'it is in %d pieces' % r['components'] if r['components'] > 1 else 'it closes on itself')),
            SOURCES['euler'])
        holes = [lp for lp in r['loops'][1:]] if len(r['loops']) > 1 else []
        if r['loops']:
            # (per component: its longest loop is its outer edge, the rest are holes in it)
            vc = mesh_topology(p['V'], p['F'])['vcomp'] if r['components'] > 1 else None
            comp = [int(vc[lp['verts'][0]]) if vc is not None else 0 for lp in r['loops']]
            outer = {c: max((i for i in range(len(r['loops'])) if comp[i] == c), key=lambda i: r['loops'][i]['length'])
                     for c in set(comp)}
            holes = [lp for i, lp in enumerate(r['loops']) if outer[comp[i]] != i]
        hole_names = [sorted({str(x['label']) for x in lp['runs']}) for lp in holes]
        add('boundary', nm, not holes or all(set(h) <= set(r['openings']) for h in hole_names), len(r['loops']),
            '1 outer edge' + (' + declared openings' if r['openings'] else ''),
            ('the edge is one loop' if not holes else 'inner loops (holes) labelled %s' % hole_names), SOURCES['boundary'])
        bad = r['nonmanifold'] + r['bowtie'] + r['flipped'] + r['degenerate'] + r['loose']
        add('mesh', nm, bad == 0, bad, 0, 'non-manifold edges %d, bow ties %d, flipped %d, degenerate %d, loose %d' % (
            r['nonmanifold'], r['bowtie'], r['flipped'], r['degenerate'], r['loose']), SOURCES['mesh'])
        if r['pure']:
            add('index sum', nm, r['index_sum'] == r['index_expected'], r['index_sum'], r['index_expected'],
                'sum of valence indices %d against %d x chi = %d (Poincare-Hopf on the mesh)' % (
                    r['index_sum'], 4 if r['mesh'] == 'quad' else 6, r['index_expected']), SOURCES['singular'])
        nv = len(r['singular_visible'])
        if not r['solid']:
            add('singular', nm, nv == 0, nv, 0, '%d singular point(s) inside (valence: count %s), %d in the visible field away from the edge; '
                '%d dislocation(s), a valence %d beside a %d, where a row of the grid ends' % (
                    len(r['singular']), r['singular_hist'], nv, r['dislocations'], r['regular'][0] - 1, r['regular'][0] + 1),
                SOURCES['singular'], experimental=True, level='warn')
            if r['dislocations']:
                add('grid', nm, r['dislocations'] == 0, r['dislocations'], 0,
                    '%d dislocations: the cloth grid is not regular there (the springs differ), so folds may kink along them'
                    % r['dislocations'], SOURCES['singular'] + '; a 3-5 pair has index 0', experimental=True, level='info')
        if not r['flat'] or r['solid']:
            gbr = r['gauss_bonnet_residual']
            add('gauss-bonnet', nm, abs(gbr) < 1e-6 * max(1, r['absK']) + 1e-6, round(gbr, 9), 0,
                'interior curvature %.3f + boundary turn %.3f = 2 pi chi %.3f' % (r['K_interior'], r['kg_boundary'], 2 * math.pi * r['chi']),
                SOURCES['flatten'])
        if not r['flat']:
            lim = FLATTEN_LIMIT.get(r['material'], FLATTEN_LIMIT[''])
            g = r['flatten_give']
            okf = (g <= lim) if np.isfinite(g) else False
            add('flatten', nm, okf and (r['cuts_to_disc'] in (0, None)),
                round(100 * g, 2) if np.isfinite(g) else None, '<= %.0f %% (%s)' % (100 * lim, r['material'] or 'cloth'),
                'laid flat (LSCM) its edges give %.1f %% to %+.1f %% (p5, p95); %s; curvature to lose %.3f rad, the worst '
                'patch %.1f %% by Bertrand-Diguet-Puiseux' % (
                    100 * (r['flat_lengths'][0] - 1), 100 * (r['flat_lengths'][2] - 1),
                    'a disc, no cut needed' if r['cuts_to_disc'] == 0 else ('in pieces (its main part laid flat)' if r['cuts_to_disc'] is None else '%s cut(s) needed to open it to a disc' % r['cuts_to_disc']),
                    r['absK'], 100 * r['flatten_strain_patch'])
                + ('' if okf else ': more than the cloth gives, so a dart or seam through the marked spot, or block it wet'),
                SOURCES['flatten'] + '; Levy et al. LSCM (2002)', experimental=True)
    asm = None
    if len(pieces) > 1 or 'seams' in draft:
        # (stray islands are the euler check's fault; the garment is sewn from each piece's main part)
        mains = [main_component(p) for p in pieces]
        apieces = [m[0] for m in mains]
        areps = [r if m[1] == 0 else piece_report(m[0], flatten=False) for r, m in zip(reps, mains)]
        asm = assemble(apieces, areps, draft.get('seams'), draft.get('reach'))
        asm['islands_dropped'] = {p['name']: m[1] for p, m in zip(pieces, mains) if m[1]}
        want_chi = exp.get('chi')
        add('assembled euler', 'garment', want_chi is None or asm['chi'] == want_chi, asm['chi'], want_chi,
            'sewn: chi %d, %d component(s), %d boundary loop(s), genus %g%s' % (asm['chi'], asm['components'], asm['boundary'], asm['genus'],
                                                                             ' (stray islands left out: %s)' % asm['islands_dropped'] if asm['islands_dropped'] else ''),
            SOURCES['euler'])
        add('assembled genus', 'garment', asm['genus'] == exp.get('genus', 0), asm['genus'], exp.get('genus', 0),
            'genus %g: %s' % (asm['genus'], 'no handle' if asm['genus'] == 0 else 'a seam closes a loop round a handle: a tunnel'), SOURCES['euler'])
        names = [sorted({'%s %s' % (pn, l) for pn, l in lp['runs']}) for lp in asm['loops']]
        want_open = exp.get('openings')
        own = {p['name']: {str(x) for x in p.get('openings', [])} for p in pieces}
        # (a loop is an opening when every run along it is one its own piece names as an opening)
        stray = [n for n, lp in zip(names, asm['loops']) if not all(str(l) in own.get(pn, ()) for pn, l in lp['runs'])]
        add('openings', 'garment', not stray and (want_open is None or len(names) == len(want_open)), len(names),
            len(want_open) if want_open is not None else None,
            'loops left open after sewing: %s%s' % ('; '.join('%s (%.1f)' % (', '.join(n), lp['length']) for n, lp in zip(names, asm['loops'])),
                                                    ('; NOT openings (an open seam or a hole): %s' % stray) if stray else ''),
            SOURCES['boundary'])
        add('seam pairing', 'garment', not asm['faults'] and not asm['unpaired'] and asm['pinch'] == 0,
            len(asm['faults']) + len(asm['unpaired']) + asm['pinch'], 0,
            '; '.join([f['why'] + ' (%s)' % f['seam'] for f in asm['faults']]
                      + ['%s has no partner for its %s edge' % u for u in asm['unpaired']]
                      + (['%d pinch point(s): an opening that touches itself' % asm['pinch']] if asm['pinch'] else []))
            or 'every seam run has exactly one partner', SOURCES['seams'])
        worst = max(asm['lengths'], key=lambda m: m['ratio']) if asm['lengths'] else None
        if worst:
            lim = float(draft.get('seam_ease', 0.03))
            add('seam lengths', 'garment', worst['ratio'] - 1 <= lim, round(100 * (worst['ratio'] - 1), 1), '<= %.0f %%' % (100 * lim),
                'the most unequal seam %s (%s %.2f, %s %.2f)' % (worst['seam'], worst['a'], worst['len_a'], worst['b'], worst['len_b']),
                'pattern drafting: a seam is walked, its two sides equal but for the ease meant there (limit EXPERIMENTAL)',
                experimental=True, level='warn')
    parts = None
    if draft.get('parts'):
        parts = anchor_graph(draft['parts'])
        fl = [x for x in parts['floating'] if x not in parts['friction']]
        add('connected', 'parts', not fl, len(fl), 0,
            'every part has a path to an anchor' if not fl else 'floating parts (no hold at all): %s' % ', '.join(fl[:12])
            + (' and %d more' % (len(fl) - 12) if len(fl) > 12 else ''), SOURCES['anchors'])
        fr = parts['friction']
        add('captured', 'parts', not fr, len(fr), 0,
            'every part held by a sliding hold is closed on itself (knotted or looped)' if not fr else
            'held only by sliding holds, nothing closes them, so they pull out: %s' % ', '.join(fr[:12])
            + (' and %d more' % (len(fr) - 12) if len(fr) > 12 else ''),
            SOURCES['anchors'] + '; a cord threaded through holes is held only while its ends are knotted into a closed loop')
        if parts['unused']:
            add('unused holds', 'parts', False, len(parts['unused']), 0, 'holds nothing passes through: %s' % ', '.join(parts['unused'][:8]),
                'drafting: every eyelet or pin in the draft carries something', level='warn')
        short = [x for x in parts['loaded'] if x['paths'] < x['need']]
        add('redundancy', 'parts', not short, len(short), 0,
            'b1 = %d independent hold loops; %s' % (parts['b1'], 'every loaded part has %d+ separate paths to an anchor' % (
                draft['parts'].get('min_holds', 2)) if not short else 'held by fewer paths than needed: %s' % ', '.join(
                '%s (%d of %d)' % (x['part'], x['paths'], x['need']) for x in short[:10])), SOURCES['anchors'])
        add('bridges', 'parts', True, len(parts['bridges']), None,
            '%d single holds whose loss frees a part%s' % (len(parts['bridges']), (': ' + '; '.join(
                '%s frees %s' % (h['hold'], ', '.join(h['frees'][:4]) + ('...' if len(h['frees']) > 4 else '')) for h in parts['bridges'][:6])) if parts['bridges'] else ''),
            SOURCES['anchors'], level='info')
    clos = []
    for c in draft.get('closures', []):
        cr = closure_report(c, float(draft.get('unit_cm', 1.0)))
        clos.append(cr)
        for k in cr['checks']:
            add('closure: ' + k['id'], c['name'], k['ok'], k['value'], k['limit'], k['why'], k['source'], k['experimental'])
    fails = [c for c in checks if c['level'] == 'fail']
    warns = [c for c in checks if c['level'] == 'warn']
    verdict = 'fail' if fails else ('watch' if warns else 'pass')
    slim = [{k: v for k, v in r.items() if k not in ('loops', 'defect', 'singular')} | {
        'loops': [dict(length=lp['length'], runs=[(x['label'], round(x['length'], 3)) for x in lp['runs']]) for lp in r['loops']]}
        for r in reps]
    return dict(version=VERSION, name=draft.get('name', 'draft'), verdict=verdict, fails=len(fails), warns=len(warns),
                checks=checks, pieces=slim, assembly=asm, parts=parts, closures=clos, _reps=reps)


def summary(rep):
    lines = ['draft topology %s: %s (%d fail, %d watch)' % (rep['name'], rep['verdict'], rep['fails'], rep['warns'])]
    for c in rep['checks']:
        mark = {'pass': 'ok  ', 'fail': 'FAIL', 'warn': 'warn', 'info': 'info'}[c['level']]
        lines.append('  %s %-22s %-14s %s%s' % (mark, c['id'], c['scope'][:14], c['why'], ' [experimental]' if c['experimental'] and c['level'] != 'pass' else ''))
    return '\n'.join(lines)


def to_json(rep):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items() if not str(k).startswith('_')}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    return json.dumps(conv(rep), indent=1)


# ------------------------------------------------------------------------------------------------ the sheet

PAL = dict(bg=(246, 244, 240), ink=(40, 40, 44), soft=(150, 150, 150), ok=(40, 140, 70), bad=(200, 40, 40),
           warn=(220, 140, 20), open=(40, 140, 70), seam=(60, 100, 200), hole=(200, 40, 40), anchor=(30, 30, 30))
SEAM_COLS = [(196, 60, 50), (40, 110, 190), (30, 140, 80), (150, 70, 170), (210, 130, 20), (20, 150, 160),
             (120, 120, 30), (200, 40, 120), (90, 90, 90), (100, 60, 20), (0, 90, 140), (160, 100, 140)]


def _flat2d(V, F):
    """A cheap flat layout of a piece for the diagram: its own plane (the two main axes), or the plane it lies in."""
    V = np.asarray(V, float)
    if V.shape[1] == 2:
        return V
    c = V - V.mean(0)
    u, s, vt = np.linalg.svd(c[::max(1, len(c) // 4000)], full_matrices=False)
    return c @ vt[:2].T


def sheet(rep, draft, path, title=None, tile=300):
    """The diagram sheet: one tile per piece (the edge loops by label, holes red, singular points, the curvature and
    the dart spot), the seam graph, the parts graph, the closures, and the list of checks."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
        bold = ImageFont.truetype('DejaVuSans-Bold.ttf', 14)
    except OSError:
        font = bold = ImageFont.load_default()
    pieces = draft.get('pieces', [])
    reps = rep['_reps']
    labs = sorted({str(l) for r in reps for l in r['labels']})
    lcol = {l: SEAM_COLS[i % len(SEAM_COLS)] for i, l in enumerate(labs)}
    tiles = []
    for p, r in zip(pieces, reps):
        im = Image.new('RGB', (tile, tile + 40), PAL['bg'])
        dr = ImageDraw.Draw(im)
        X = r['UV'] if r.get('UV') is not None else _flat2d(p['V'], p['F'])
        lo, hi = X.min(0), X.max(0)
        sc = (tile - 30) / max(float((hi - lo).max()), 1e-9)
        q = lambda P: [(15 + (x - lo[0]) * sc, 15 + (hi[1] - y) * sc) for x, y in P]    # noqa: E731
        d = r['defect']
        if not r['flat'] and np.abs(d).max() > 0:
            # (the curvature as dots: warm for domes, cool for saddles, size by |K|)
            m = np.percentile(np.abs(d), 99.5) or 1
            idx = np.nonzero(np.abs(d) > 0.15 * m)[0][::max(1, len(d) // 3000)]
            for i, (x, y) in zip(idx, q(X[idx])):
                t = min(1.0, abs(d[i]) / m)
                col = (int(230 - 30 * t), int(200 - 120 * t), int(160 - 120 * t)) if d[i] > 0 else (int(160 - 100 * t), int(190 - 60 * t), int(230 - 10 * t))
                dr.point((x, y), fill=col)
        for lp in r['loops']:
            outer = lp is max(r['loops'], key=lambda z: z['length'])
            for run in lp['runs']:
                lab = run['label']
                if lab is None:
                    col = PAL['soft'] if outer else PAL['hole']
                else:
                    col = lcol[str(lab)]
                pts = q(X[run['verts']])
                if len(pts) > 1:
                    dr.line(pts, fill=col, width=3 if (lab is not None or not outer) else 1)
        for v in r['singular']:
            x, y = q(X[[v]])[0]
            vis = v in r['singular_visible']
            dr.ellipse((x - 3, y - 3, x + 3, y + 3), outline=PAL['bad'] if vis else PAL['soft'], width=1)
        if r.get('dart_spot') is not None and not r['flat']:
            V3 = np.asarray(p['V'], float)
            k = int(np.argmin(np.linalg.norm(V3 - np.asarray(r['dart_spot']), axis=1)))
            x, y = q(X[[k]])[0]
            dr.line((x - 8, y - 8, x + 8, y + 8), fill=PAL['bad'], width=2)
            dr.line((x - 8, y + 8, x + 8, y - 8), fill=PAL['bad'], width=2)
        dr.text((6, tile - 8), r['name'], fill=PAL['ink'], font=bold)
        bad_here = [c['id'] for c in rep['checks'] if c['scope'] == r['name'] and c['level'] == 'fail']
        dr.text((6, tile + 10), 'chi %d  b %d  g %g  singular %d  flat give %s%s' % (
            r['chi'], r['boundary'], r['percomp'][0]['genus'] if r['percomp'] else 0, len(r['singular_visible']),
            '-' if r['flat'] or not np.isfinite(r['flatten_give']) else '%.1f%%' % (100 * r['flatten_give']),
            ('  FAIL ' + ','.join(bad_here)) if bad_here else ''), fill=PAL['bad'] if bad_here else PAL['ok'], font=font)
        tiles.append(im)
    # the seam graph
    if rep.get('assembly'):
        tiles.append(_seam_graph(rep, pieces, lcol, tile, font, bold))
    if rep.get('parts'):
        tiles.append(_parts_graph(rep, draft, tile, font, bold))
    for c, cr in zip(draft.get('closures', []), rep['closures']):
        tiles.append(_closure_tile(c, cr, tile, font, bold))
    cols = 4
    # (a wide tile takes two places)
    slots, k = [], 0
    for t in tiles:
        span = max(1, t.width // tile)
        if k % cols + span > cols:
            k += cols - k % cols
        slots.append(k)
        k += span
    rows = max(1, (k + cols - 1) // cols)
    W = cols * tile
    lines = [c for c in rep['checks']]
    th = 40 + 16 * (len(lines) + len(labs) // 6 + 2)
    out = Image.new('RGB', (W, rows * (tile + 40) + th), PAL['bg'])
    for t, k in zip(tiles, slots):
        out.paste(t, ((k % cols) * tile, (k // cols) * (tile + 40)))
    dr = ImageDraw.Draw(out)
    y = rows * (tile + 40) + 6
    dr.text((8, y), title or ('Draft topology: %s  (%s: %d fail, %d watch)' % (rep['name'], rep['verdict'], rep['fails'], rep['warns'])),
            fill=PAL['ink'], font=bold)
    y += 22
    x = 8
    for l in labs:
        dr.line((x, y + 7, x + 18, y + 7), fill=lcol[l], width=3)
        dr.text((x + 22, y), l, fill=PAL['ink'], font=font)
        x += 30 + 7 * len(l)
        if x > W - 120:
            x, y = 8, y + 16
    dr.text((x, y), '  red ring: visible singular point; red X: where the curvature gathers (dart spot)', fill=PAL['soft'], font=font)
    y += 20
    for c in lines:
        col = {'pass': PAL['ok'], 'fail': PAL['bad'], 'warn': PAL['warn'], 'info': PAL['soft']}[c['level']]
        txt = '%-5s %s | %s: %s%s' % (c['level'].upper(), c['scope'], c['id'], c['why'], '  [experimental]' if c['experimental'] else '')
        dr.text((8, y), txt[:int(W / 6.6)], fill=col, font=font)
        y += 16
    out.save(path)
    return path


def _seam_graph(rep, pieces, lcol, tile, font, bold):
    from PIL import Image, ImageDraw
    im = Image.new('RGB', (tile, tile + 40), PAL['bg'])
    dr = ImageDraw.Draw(im)
    names = [p['name'] for p in pieces]
    n = len(names)
    cx, cy, R = tile / 2, tile / 2 + 5, tile * 0.34
    pos = {nm: (cx + R * math.cos(2 * math.pi * i / n - math.pi / 2), cy + R * math.sin(2 * math.pi * i / n - math.pi / 2)) for i, nm in enumerate(names)}
    A = rep['assembly']
    for k, s in enumerate(A['seams']):
        a, b = pos[s[1]], pos[s[2]]
        off = 6 * ((k % 3) - 1)
        mid = ((a[0] + b[0]) / 2 + off, (a[1] + b[1]) / 2 + off)
        col = lcol.get(str(s[0]), PAL['seam'])
        bad = any(f['seam'] == s[0] for f in A['faults'])
        dr.line((a, mid, b), fill=PAL['bad'] if bad else col, width=3)
        dr.text(mid, str(s[0])[:10], fill=col, font=font)
    for (pn, lab) in A['unpaired']:
        x, y = pos[pn]
        dr.text((x + 8, y + 10), 'unpaired %s' % lab, fill=PAL['bad'], font=font)
    for nm, (x, y) in pos.items():
        dr.ellipse((x - 7, y - 7, x + 7, y + 7), fill=PAL['ink'])
        dr.text((x + 9, y - 7), nm, fill=PAL['ink'], font=font)
    dr.text((6, 4), 'Seams (pieces, edges sewn)', fill=PAL['ink'], font=bold)
    dr.text((6, tile + 4), 'sewn: chi %d  genus %g  open loops %d' % (A['chi'], A['genus'], A['boundary']), fill=PAL['ink'], font=font)
    dr.text((6, tile + 20), '; '.join('/'.join(sorted({str(l) for _, l in lp['runs']})) for lp in A['loops'])[:48], fill=PAL['soft'], font=font)
    return im


def _parts_geom(rep, draft, tile, font, bold):
    """The parts drawn where they are (strands as lines; holds as points): from behind and from the side, each part
    coloured by its state: anchored by two or more paths green, by one orange, floating or pulled out red."""
    from PIL import Image, ImageDraw
    G, P = draft['parts'], rep['parts']
    geo = G['geom']
    im = Image.new('RGB', (2 * tile, tile + 40), PAL['bg'])
    dr = ImageDraw.Draw(im)
    allp = np.concatenate([np.asarray(v, float).reshape(-1, 3) for v in geo.values()] + [np.asarray(G.get('backdrop', np.zeros((0, 3))), float)])
    lo, hi = allp.min(0), allp.max(0)
    paths = {x['part']: x['paths'] for x in P['loaded']}
    fl = set(P['floating'])
    for k, (ax, name) in enumerate((((0, 2), 'from behind'), ((1, 2), 'from the side'))):
        sc = (tile - 30) / max(float(max(hi[ax[0]] - lo[ax[0]], hi[ax[1]] - lo[ax[1]])), 1e-9)
        ox = k * tile + 15

        def q(Pp):
            Pp = np.asarray(Pp, float).reshape(-1, 3)
            xs = (Pp[:, ax[0]] - lo[ax[0]]) * sc if k == 1 else (hi[ax[0]] - Pp[:, ax[0]]) * sc
            return list(zip(ox + xs, 15 + (hi[ax[1]] - Pp[:, ax[1]]) * sc))
        bd = G.get('backdrop')
        if bd is not None:
            for x, y in q(np.asarray(bd)[::max(1, len(bd) // 4000)]):
                dr.point((x, y), fill=(205, 203, 198))
        order = sorted(geo, key=lambda n: (n in fl, paths.get(n, 9) < 2))
        for nm in order:
            pts = q(geo[nm])
            col = PAL['bad'] if nm in fl else (PAL['warn'] if paths.get(nm, 9) < 2 else PAL['ok'])
            if nm in G.get('anchors', []):
                continue
            if len(pts) > 1:
                dr.line(pts, fill=col, width=2 if nm in fl else 1)
            else:
                x, y = pts[0]
                dr.ellipse((x - 4, y - 4, x + 4, y + 4), outline=PAL['seam'], width=2)
        dr.text((ox, 2), name, fill=PAL['ink'], font=font)
    dr.text((6, tile + 4), 'b1 %d  floating %d (red)  one path %d (orange)  holds as blue rings' % (
        P['b1'], len(P['floating']), sum(1 for x in P['loaded'] if x['paths'] == 1)), fill=PAL['ink'], font=font)
    return im


def _parts_graph(rep, draft, tile, font, bold):
    from PIL import Image, ImageDraw
    if draft['parts'].get('geom'):
        return _parts_geom(rep, draft, tile, font, bold)
    im = Image.new('RGB', (tile, tile + 40), PAL['bg'])
    dr = ImageDraw.Draw(im)
    G = draft['parts']
    P = rep['parts']
    nodes = list(G['nodes'])
    pos = G.get('pos')
    anchors = set(G.get('anchors', []))
    if not pos:
        # layered: anchors at the top, then by distance in holds
        from collections import deque
        adj = {n: [] for n in nodes}
        for h in G['holds']:
            a, b = h[0], h[1]
            if a in adj and b in adj:
                adj[a].append(b)
                adj[b].append(a)
        lev = {a: 0 for a in anchors if a in adj}
        dq = deque(lev)
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in lev:
                    lev[v] = lev[u] + 1
                    dq.append(v)
        mx = max(lev.values()) + 1 if lev else 1
        rowsn = {}
        for nm in nodes:
            rowsn.setdefault(lev.get(nm, mx), []).append(nm)
        pos = {}
        nl = max(rowsn) + 1
        for L_, ms in rowsn.items():
            for i, nm in enumerate(ms):
                pos[nm] = (20 + (tile - 40) * (i + 0.5) / len(ms), 34 + (tile - 60) * L_ / max(1, nl - 1))
    kinds = sorted({h[2] for h in G['holds']})
    kcol = {k: SEAM_COLS[i % len(SEAM_COLS)] for i, k in enumerate(kinds)}
    for h in G['holds']:
        a, b, k = h[0], h[1], h[2]
        if a in pos and b in pos:
            dr.line((pos[a], pos[b]), fill=kcol[k], width=1)
    float_ = set(P['floating'])
    short = {x['part'] for x in P['loaded'] if x['paths'] < x['need']}
    big = len(nodes) <= 60
    for nm in nodes:
        x, y = pos[nm]
        r = 5 if big else 2
        col = PAL['anchor'] if nm in anchors else (PAL['bad'] if nm in float_ else (PAL['warn'] if nm in short else PAL['ok']))
        dr.ellipse((x - r, y - r, x + r, y + r), fill=col)
        if big or nm in anchors:
            dr.text((x + 6, y - 6), nm[:14], fill=PAL['ink'], font=font)
    y0 = tile - 14 - 14 * len(kinds)
    for i, k in enumerate(kinds):
        dr.line((8, y0 + 14 * i + 6, 26, y0 + 14 * i + 6), fill=kcol[k], width=3)
        dr.text((30, y0 + 14 * i), k, fill=PAL['ink'], font=font)
    dr.text((6, 4), 'Parts and holds (anchor black, floating red, short of holds orange)', fill=PAL['ink'], font=font)
    dr.text((6, tile + 4), 'b1 %d  floating %d  bridges %d' % (P['b1'], len(P['floating']), len(P['bridges'])), fill=PAL['ink'], font=font)
    dr.text((6, tile + 20), 'loaded short: %d of %d' % (len(short), len(P['loaded'])), fill=PAL['bad'] if short else PAL['ok'], font=font)
    return im


def _closure_tile(c, cr, tile, font, bold):
    from PIL import Image, ImageDraw
    im = Image.new('RGB', (tile, tile + 40), PAL['bg'])
    dr = ImageDraw.Draw(im)
    L = max(cr['length'], 1e-6)
    x0 = 70
    ys = lambda s: 30 + (tile - 50) * s / L     # noqa: E731
    dr.line((x0, ys(0), x0, ys(L)), fill=PAL['ink'], width=3)
    if c.get('strain') is not None:
        S = np.asarray(c['strain'], float)
        mx = max(S[:, 1].max(), STRAIN_LIMIT * 1.5)
        u = c.get('_unit', 1.0)
        pts = [(x0 + 30 + (tile - x0 - 50) * v / mx, ys(sv * u)) for sv, v in S]
        dr.line(pts, fill=PAL['warn'], width=2)
        xl = x0 + 30 + (tile - x0 - 50) * STRAIN_LIMIT / mx
        dr.line((xl, ys(0), xl, ys(L)), fill=PAL['soft'], width=1)
        dr.text((xl + 2, ys(L) - 12), 'strain %.0f%%' % (100 * STRAIN_LIMIT), fill=PAL['soft'], font=font)
    for nm, key in (('break', 'break_s'), ('neck', 'neck_s'), ('end', 'end_s')):
        if c.get(key) is not None:
            y = ys(float(c[key]) * c.get('_unit', 1.0))
            dr.line((x0 - 30, y, x0 + 30, y), fill=PAL['seam'], width=2)
            dr.text((4, y - 7), nm, fill=PAL['seam'], font=font)
    for nm, s in (c.get('pressure') or {}).items():
        y = ys(float(s) * c.get('_unit', 1.0))
        dr.line((x0 - 22, y, x0 + 22, y), fill=PAL['warn'], width=1)
        dr.text((4, y - 7), nm, fill=PAL['warn'], font=font)
    for s in cr['buttons']:
        y = ys(s)
        dr.ellipse((x0 - 5, y - 5, x0 + 5, y + 5), fill=PAL['ink'])
    bad = [k for k in cr['checks'] if not k['ok']]
    dr.text((6, 4), 'Closure: %s' % c['name'], fill=PAL['ink'], font=bold)
    dr.text((6, tile + 4), '%d buttons over %.1f cm, gaps %s' % (len(cr['buttons']), L, ', '.join('%.1f' % g for g in cr.get('gaps', [])[:6])), fill=PAL['ink'], font=font)
    dr.text((6, tile + 20), ('fails: ' + ', '.join(k['id'] for k in bad))[:48] if bad else 'all closure rules pass', fill=PAL['bad'] if bad else PAL['ok'], font=font)
    return im


# ------------------------------------------------------------------------------------------------ self test

def _grid(nx, ny, w=1.0, h=1.0, f=None, tri=False):
    xs, ys = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, h, ny), indexing='ij')
    V = np.c_[xs.ravel(), ys.ravel(), np.zeros(nx * ny)]
    if f is not None:
        V = f(V)
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            F += [[a, b, c], [a, c, d]] if tri else [[a, b, c, d]]
    return V, F


def selftest():
    """Known shapes: a disc, an annulus, a tube sewn from a strip, a closed box, a torus, a spherical cap (its
    flattening strain against Bertrand-Diguet-Puiseux), a parts graph with a floating part and a bridge, and a closure."""
    out = []
    V, F = _grid(11, 11)
    r = piece_report(dict(name='square', V=V, F=F))
    out.append(('disc chi 1, 1 loop, genus 0', r['chi'] == 1 and r['boundary'] == 1 and r['percomp'][0]['genus'] == 0))
    out.append(('quad index sum = 4 chi', r['index_sum'] == 4 * r['chi']))
    Ft = [f for f in F if not (4 <= (f[0] // 11) < 6 and 4 <= (f[0] % 11) < 6)]
    r = piece_report(dict(name='annulus', V=V, F=Ft))
    out.append(('annulus chi 0, 2 loops', r['chi'] == 0 and r['boundary'] == 2 and r['cuts_to_disc'] == 1))
    # a strip sewn into a tube: left and right edges sewn ('S'), top and bottom open
    V, F = _grid(13, 6, w=2 * math.pi, h=1.0)
    lab = dict(S=(np.abs(V[:, 0]) < 1e-9) | (np.abs(V[:, 0] - 2 * math.pi) < 1e-9),
               top=np.abs(V[:, 1] - 1) < 1e-9, bottom=np.abs(V[:, 1]) < 1e-9)
    Vc = np.c_[np.cos(V[:, 0]), np.sin(V[:, 0]), V[:, 1]]
    d = dict(name='tube', unit_cm=1.0, pieces=[dict(name='strip', V=Vc, F=F, edges=lab, openings=['top', 'bottom'], flat=True)],
             seams=[['S', 'strip', 'strip']], expected=dict(chi=0, openings=['top', 'bottom']))
    R = run(d)
    a = R['assembly']
    out.append(('strip sewn to a tube: chi 0, 2 openings, genus 0', a['chi'] == 0 and a['boundary'] == 2 and a['genus'] == 0))
    # unsewn: the seam is open, the loop is not an opening
    d2 = dict(d, seams=[])
    R2 = run(d2)
    out.append(('unsewn strip: open seam flagged', any(c['id'] == 'openings' and not c['ok'] for c in R2['checks'])))
    # a spherical cap of angle 40 degrees: total K = 2 pi (1 - cos a), strain = that / 6 pi
    al = math.radians(40)

    def cap(Vg):
        t = Vg[:, 0] * al
        p = Vg[:, 1] * 2 * math.pi
        return np.c_[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
    Vg, Fg = _grid(61, 121, tri=True)        # (the discrete K misses the last half-ring by the boundary: about 1 / rings)
    Vs = cap(Vg)
    # weld the seam and the pole
    key = np.round(Vs, 9)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Fw = [[int(inv[i]) for i in f] for f in Fg]
    Fw = [f for f in Fw if len(set(f)) == 3]
    Vw = np.zeros((inv.max() + 1, 3))
    Vw[inv] = Vs
    r = piece_report(dict(name='cap', V=Vw, F=Fw))
    Kt = 2 * math.pi * (1 - math.cos(al))
    out.append(('cap: total K = 2 pi (1 - cos a) within 2 %', abs(r['K_interior'] - Kt) / Kt < 0.02))
    out.append(('cap: Gauss-Bonnet residual ~ 0', abs(r['gauss_bonnet_residual']) < 1e-9))
    out.append(('cap: flatten strain = K_total / 6 pi', abs(r['flatten_strain_total'] - Kt / (6 * math.pi)) < 0.002))
    # a torus: chi 0, genus 1
    Vg, Fg = _grid(21, 13, tri=False)
    Vt = np.c_[(2 + np.cos(Vg[:, 1] * 2 * math.pi)) * np.cos(Vg[:, 0] * 2 * math.pi),
               (2 + np.cos(Vg[:, 1] * 2 * math.pi)) * np.sin(Vg[:, 0] * 2 * math.pi), np.sin(Vg[:, 1] * 2 * math.pi)]
    key = np.round(Vt, 9)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Fw = [[int(inv[i]) for i in f] for f in Fg]
    Vw = np.zeros((inv.max() + 1, 3))
    Vw[inv] = Vt
    r = piece_report(dict(name='torus', V=Vw, F=Fw))
    out.append(('torus: chi 0, genus 1, 2 cuts to a disc', r['chi'] == 0 and r['percomp'][0]['genus'] == 1 and r['cuts_to_disc'] == 2))
    # parts: a lace with two tails, one tail tied, the other free; the bow knot a bridge
    G = dict(nodes=['boot', 'eyelet 1', 'eyelet 2', 'lace', 'knot', 'tail L', 'tail R', 'aglet'], anchors=['boot'],
             holds=[['boot', 'eyelet 1', 'set'], ['boot', 'eyelet 2', 'set'], ['eyelet 1', 'lace', 'thread'],
                    ['eyelet 2', 'lace', 'thread'], ['lace', 'knot', 'tie'], ['knot', 'tail L', 'tie'],
                    ['knot', 'tail R', 'tie']], loaded=['lace', 'tail L'], min_holds=2)
    P = anchor_graph(G)
    out.append(('parts: aglet floating', P['floating'] == ['aglet']))
    out.append(('parts: b1 1 (the lace round two eyelets)', P['b1'] == 1))
    out.append(('parts: lace held twice, tail once', [x['paths'] for x in P['loaded']] == [2, 1]))
    out.append(('parts: the knot is a bridge', any('knot' in h['between'] for h in P['bridges'])))
    # closure: 6 buttons 2 cm apart from the break, bust at 4 cm
    E = np.c_[np.zeros(30), np.zeros(30), -np.linspace(0, 14, 30)]
    C = closure_report(dict(name='front', edge=E, buttons=[0.5, 2.5, 4.5, 6.5, 8.5, 10.5], break_s=0.0, end_s=11.0,
                            pressure=dict(bust=4.5), button_d=1.2, extension=1.6, hole_len=1.6), 1.0)
    out.append(('closure: all pass', all(k['ok'] for k in C['checks'])))
    C = closure_report(dict(name='front', edge=E, buttons=[0.5, 2.5, 4.5, 7.5, 8.5], break_s=1.0, pressure=dict(bust=6.0)), 1.0)
    out.append(('closure: above break, off bust, uneven all fail', sum(not k['ok'] for k in C['checks']) >= 3))
    st = np.c_[np.linspace(0, 11, 111), 0.02 + 0.08 * np.exp(-((np.linspace(0, 11, 111) - 3.5) / 0.4) ** 2)]
    C = closure_report(dict(name='front', edge=E, buttons=[0.5, 2.5, 4.5, 6.5, 8.5, 10.5], end_s=11.0, strain=st), 1.0)
    out.append(('closure: a strain peak between two buttons is a gap', any(k['id'] == 'strain gaps' and not k['ok'] for k in C['checks'])))
    st[:, 1] = 0.02 + 0.08 * np.exp(-((st[:, 0] - 4.5) / 0.4) ** 2)
    C = closure_report(dict(name='front', edge=E, buttons=[0.5, 2.5, 4.5, 6.5, 8.5, 10.5], end_s=11.0, strain=st), 1.0)
    out.append(('closure: a strain peak on a button is held', any(k['id'] == 'strain gaps' and k['ok'] for k in C['checks'])))
    return out


if __name__ == '__main__':
    res = selftest()
    for name, ok in res:
        print('%s  %s' % ('ok  ' if ok else 'FAIL', name))
    print('%d of %d pass' % (sum(ok for _, ok in res), len(res)))
