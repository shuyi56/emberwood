"""The lace study's contour sheet: one page per solved strip, from the JSON study.mjs writes.

    python3 tools/lace-study/contours.py RUN.json [RUN.json ...] --out DIR

Panels: the sections across the strip (from above, or from the front for a jabot) at 0, 25, 50, 75 and 100% of
its depth; its side and front profiles; the threads' stretch on the flat pattern; the colour-coded pattern with
its grey-value check (two parts must not fall to the same grey); the convergence of the solve; the numbers and the
flags. The project's rule: no render before this sheet has been looked at."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

PART_COLORS = ['#8fb3d9', '#e0b35a', '#5e3070', '#2f6b4f', '#d2523a']
SEC_COLORS = ['#2c4f9e', '#3f8fb0', '#3fa77a', '#c79a2c', '#d9532f']


def grey(hexc):
    h = hexc.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def sheet(d, out):
    P = np.array(d['pos']).reshape(-1, 3)
    T = np.array(d['tris']).reshape(-1, 3)
    F = np.array(d['flat']).reshape(-1, 2)
    parts = np.array(d['parts'])
    strain = np.array(d['strain']) * 100
    mount = d['o']['mount']
    fig = plt.figure(figsize=(16, 10), dpi=110)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 0.9])
    fig.suptitle(f"{d['name']}: {d['o']['kind']} pleat, {mount}, {d['o']['lace']} lace  (cache {d['key']}, {d['from']})", fontsize=13, x=0.01, ha='left')

    # sections
    ax = fig.add_subplot(gs[0, 0])
    for k, s in enumerate(d['sections']):
        q = np.array(s['pts'])
        if mount == 'jabot':
            ax.plot(q[:, 0], -q[:, 1], color=SEC_COLORS[k], lw=1.1, label=f"{int(s['share'] * 100)}%")
        else:
            ax.plot(q[:, 0], q[:, 2], color=SEC_COLORS[k], lw=1.1, label=f"{int(s['share'] * 100)}%")
    ax.set_aspect('equal')
    ax.set_title('Sections ' + ('from the front (x across, down the line)' if mount == 'jabot' else 'from above'), fontsize=10)
    ax.legend(fontsize=7, loc='upper right', title='depth', title_fontsize=7)
    ax.grid(alpha=0.25)

    # side and front profiles
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(P[:, 2], P[:, 1], s=0.6, c='#555')
    ax.set_aspect('equal'); ax.set_title('Side profile (z, y)', fontsize=10); ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(P[:, 0], P[:, 1], s=0.6, c='#555')
    ax.set_aspect('equal'); ax.set_title('Front profile (x, y)', fontsize=10); ax.grid(alpha=0.25)

    # the numbers
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off')
    n = d['numbers']
    rows = [
        ('points', f"{n['points']}"), ('cut / finished', f"{n['cut']:.1f} / {n['finished']:.1f} cm = {n['ratio']:.2f}x"),
        ('settled at frame', f"{d['settled']}" if d['settled'] else f"not in {d['frames']}"),
        ('thread stretch max / p99 / mean', f"{n['strainMaxPct']:.2f} / {n['strainP99Pct']:.2f} / {n['strainMeanPct']:.3f} %"),
        ('folds near heading / at scallop', f"{n['foldTopDeg']:.0f} / {n['foldBottomDeg']:.0f} deg"),
        ('flare at the scallop', f"{n['flareAtScallop']:.2f} x the heading"),
        ('ripple / waves at the scallop', f"{n['rippleAtScallopCm']:.2f} cm / {n['wavesAtScallop']}"),
        ('bending length, weight', f"{d['mat']['bendingLength'] * 100:.2f} cm, {d['mat']['weight'] * 1000:.0f} g/m2"),
    ]
    for k, s in enumerate(n['sections']):
        rows.append((f"section {int(s['share'] * 100)}%", f"ripple {s['ripple']:.2f}  waves {s['lobes']}  kink {s['kinkDeg']:.0f} deg"))
    y = 1.0
    for a, b in rows:
        ax.text(0, y, a, fontsize=8.5, va='top'); ax.text(0.55, y, b, fontsize=8.5, va='top', family='monospace'); y -= 0.065
    flags = d['flags'] or ['none']
    ax.text(0, y - 0.02, 'Flags: ' + '; '.join(flags), fontsize=8.5, va='top', color='#b5462e' if d['flags'] else '#2f6b4f', wrap=True)

    # strain on the flat pattern
    tri = mtri.Triangulation(F[:, 0], -F[:, 1], T)
    ax = fig.add_subplot(gs[1, :2])
    tc = ax.tripcolor(tri, np.clip(strain, 0, 5), shading='gouraud', cmap='inferno', vmin=0, vmax=5)
    fig.colorbar(tc, ax=ax, fraction=0.02, label='thread stretch %')
    ax.set_aspect('equal'); ax.set_title('Strain on the flat pattern (as cut, heading at top)', fontsize=10)

    # colour-coded pattern and its grey check
    ax = fig.add_subplot(gs[1, 2:])
    cols = np.array([matplotlib.colors.to_rgb(c) for c in PART_COLORS])
    ax.tripcolor(tri, facecolors=parts[T].max(axis=1).astype(float), cmap=matplotlib.colors.ListedColormap(PART_COLORS), vmin=-0.5, vmax=4.5, shading='flat')
    ax.set_aspect('equal'); ax.set_title('Colour code on the flat pattern', fontsize=10)
    for k, (nm, c) in enumerate(zip(d['partNames'], PART_COLORS)):
        ax.text(1.01, 0.95 - 0.12 * k, f"{nm}  (grey {grey(c):.2f})", transform=ax.transAxes, fontsize=8, color=c, va='top')
    gs_ = sorted(grey(c) for c in PART_COLORS)
    gap = min(b - a for a, b in zip(gs_, gs_[1:]))
    ax.text(1.01, 0.3, f"smallest grey gap {gap:.2f}" + ('  OK' if gap >= 0.04 else '  FLAG'), transform=ax.transAxes, fontsize=8, va='top')
    void = cols

    # convergence
    ax = fig.add_subplot(gs[2, :2])
    cv = d['curve']
    if cv:
        fr = [c['frame'] for c in cv]
        ax.semilogy(fr, [max(1e-9, c['speed']) for c in cv], color='#3d78c9', lw=1, label='fastest point (cm a step)')
        ax.axhline(2e-4, color='#3d78c9', ls=':', lw=0.8)
        ax2 = ax.twinx()
        sf = [(c['frame'], c['strain'] * 100) for c in cv if c.get('strain') is not None]
        if sf:
            ax2.plot(*zip(*sf), color='#d9532f', lw=1, label='worst stretch %')
        ax2.set_ylabel('worst stretch %', color='#d9532f', fontsize=8)
        ax.set_xlabel('frame (1/60 s)', fontsize=8); ax.set_title('Convergence', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
    # per-section numbers along the depth
    ax = fig.add_subplot(gs[2, 2:])
    sh = [s['share'] * 100 for s in n['sections']]
    ax.plot(sh, [s['ripple'] for s in n['sections']], 'o-', color='#3fa77a', label='ripple (cm)')
    ax.plot(sh, [s['mean'] for s in n['sections']] if mount in ('cuff', 'collar') else [s['spread'] for s in n['sections']], 's-', color='#2c4f9e', label='radius (cm)' if mount in ('cuff', 'collar') else 'spread (cm)')
    ax.set_xlabel('depth down the frill (%)', fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_title('Flare and ripple down the frill', fontsize=10)
    del void
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out, f"{d['name']}-contours.png")
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == '__main__':
    args = sys.argv[1:]
    out = args[args.index('--out') + 1] if '--out' in args else '.'
    os.makedirs(out, exist_ok=True)
    files = [a for a in args if a.endswith('.json') and 'summary' not in a and 'sweep' not in a]
    for f in files:
        with open(f) as fh:
            print(sheet(json.load(fh), out))
