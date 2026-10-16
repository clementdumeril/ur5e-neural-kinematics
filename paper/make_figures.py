"""
Figures for paper/main.tex, generated from result files in the repository.

    python paper/make_figures.py

Reads   results/seed_study/*.json, results/benchmark/benchmark.json,
        results/heatmap/workspace_error.npz (computed on first run, ~3 min)
Writes  paper/figures/*.pdf (vector) and copies the Webots screenshot.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG = os.path.join(ROOT, 'paper', 'figures')
SEEDS = os.path.join(ROOT, 'results', 'seed_study')

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
BLUE, GREY, ORANGE = '#2563eb', '#9ca3af', '#d97706'
W1, W2 = 3.3, 6.6     # one-column and full text width, inches


def load(pattern):
    return [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob(os.path.join(SEEDS, pattern)))]


def fig_ablation():
    runs = load('A_*.json')
    if not runs:
        print('  ablation: no Part A results yet'); return
    by = {}
    for r in runs:
        by.setdefault(r['seed'], {})[r['w_phys']] = r
    seeds = sorted(s for s in by if 0.0 in by[s] and 1.0 in by[s])
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.1))
    for ax, key, unit in ((axes[0], 'test_pos_mm', 'position error (mm)'),
                          (axes[1], 'test_rot_deg', 'orientation error (deg)')):
        for s in seeds:
            a, b = by[s][0.0][key], by[s][1.0][key]
            ax.plot([0, 1], [a, b], color=GREY, lw=0.8, zorder=1)
            ax.scatter([0, 1], [a, b], color=[GREY, BLUE], s=14, zorder=2)
        for x, w, c in ((0, 0.0, GREY), (1, 1.0, BLUE)):
            v = np.array([by[s][w][key] for s in seeds])
            ax.errorbar(x + 0.12, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0,
                        fmt='_', color=c, ms=10, capsize=2, lw=1)
        ax.set_xticks([0, 1], ['$w_p=0$\ndata only', '$w_p=1$\ndata + FK'])
        ax.set_xlim(-0.4, 1.5)
        ax.set_ylabel(unit)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ablation.pdf'))
    fig.savefig(os.path.join(ROOT, 'assets', 'fig_ablation.svg'))   # README copy
    plt.close(fig)
    print(f'  ablation.pdf ({len(seeds)} paired seeds)')


def fig_multihead():
    sets = [('B_*.json', 'elbow-up branches only (4)', GREY, 'o'),
            ('D_*.json', 'all 8 branches', BLUE, 's')]
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.85, 2.2))
    any_data = False
    for i, (pat, label, c, m) in enumerate(sets):
        runs = load(pat)
        if not runs:
            continue
        any_data = True
        Ks = sorted({r['K'] for r in runs})
        off = (i - 0.5) * 0.12
        for ax, key, f in ((axes[0], 'test_pos_mm', 1.0), (axes[1], 'recall', 100.0)):
            for K in Ks:
                v = np.array([r[key] for r in runs if r['K'] == K]) * f
                x = np.log2(K) + off
                ax.scatter([x] * len(v), v, color=c, s=10, marker=m, alpha=0.6, lw=0)
                ax.plot([x - 0.08, x + 0.08], [np.median(v)] * 2, color=c, lw=1.4)
            means = [np.median([r[key] for r in runs if r['K'] == K]) * f for K in Ks]
            ax.plot(np.log2(Ks) + off, means, color=c, lw=0.8,
                    label=label if ax is axes[0] else None)
    if not any_data:
        plt.close(fig); print('  multihead: no Part B/D results yet'); return
    axes[0].set_yscale('log'); axes[1].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
    axes[0].set_ylabel('test position error (mm)')
    axes[1].set_ylabel('branch recall (%)')
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xticks([0, 1, 2, 3], ['1', '2', '4', '8'])
        ax.set_xlabel('number of heads $K$')
    axes[0].legend(loc='upper right', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'multihead.pdf'))
    fig.savefig(os.path.join(ROOT, 'assets', 'fig_multihead.svg'))   # README copy
    plt.close(fig)
    print('  multihead.pdf')


def fig_benchmark():
    res = json.load(open(os.path.join(ROOT, 'results', 'benchmark', 'benchmark.json')))['resultats']
    names = {
        'forme fermee analytique': 'Closed form',
        'reseau, sortie unique': 'Network, 1 output',
        'reseau, 2 tetes + selection': 'Network, 2 heads',
        'IKPY, pose complete': 'IKPy',
        'DLS, depart unique': 'DLS, 1 start',
        'DLS, 8 relances': 'DLS, 8 restarts',
    }
    rows = sorted([(names[r['solveur']], r['temps_median_ms'], r['temps_p95_ms'],
                    r['solveur'].startswith('reseau')) for r in res if r['solveur'] in names],
                  key=lambda t: t[1], reverse=True)
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[BLUE if r[3] else GREY for r in rows], height=0.6)
    ax.errorbar([r[1] for r in rows], y, xerr=[[0] * len(rows), [r[2] - r[1] for r in rows]],
                fmt='none', ecolor='#374151', lw=0.7, capsize=2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale('log')
    ax.set_xlabel('time per target (ms)')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.savefig(os.path.join(FIG, 'benchmark.pdf'))
    plt.close(fig)
    print('  benchmark.pdf')


def fig_workspace():
    cache = os.path.join(ROOT, 'results', 'heatmap', 'workspace_error.npz')
    sys.path.insert(0, os.path.join(ROOT, 'experiments'))
    import heatmap_erreur as h
    if not os.path.exists(cache):
        print('  computing the workspace error map (~3 min)...')
        XX, YY, err, reach = h.calculer()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, XX=XX, YY=YY, err=err, reach=reach, Z=h.Z)
    d = np.load(cache)
    XX, YY, err, reach = d['XX'], d['YY'], d['err'], d['reach']
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.9))
    im = ax.pcolormesh(XX, YY, np.clip(err, 0.02, 2000), norm=LogNorm(vmin=0.05, vmax=1000),
                       cmap='magma_r', shading='auto', rasterized=True)
    ax.contourf(XX, YY, (~reach).astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
    ax.contour(XX, YY, reach.astype(float), levels=[0.5], colors='white', linewidths=1.0,
               linestyles='--')
    z = h.ZONE
    ax.add_patch(Rectangle((z['x0'], z['y0']), z['x1'] - z['x0'], z['y1'] - z['y0'],
                           fill=False, edgecolor=BLUE, lw=1.2))
    offsets = {'drop tray': (4, -10), 'reading pose': (4, 5), 'cube': (4, 4)}
    for x, yv, name in h.REPERES:
        ax.plot(x, yv, 'o', ms=3.5, mfc='white', mec='black', mew=0.8)
        dx, dy = offsets.get(name, (4, 4))
        ax.annotate(name, (x, yv), textcoords='offset points', xytext=(dx, dy), fontsize=7,
                    ha='right' if dx < 0 else 'left')
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (m)'); ax.set_ylabel('$y$ (m)')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('position error (mm)')
    fig.savefig(os.path.join(FIG, 'workspace_error.pdf'), dpi=300)
    plt.close(fig)
    inside = reach & (XX >= z['x0']) & (XX <= z['x1']) & (YY >= z['y0']) & (YY <= z['y1'])
    print(f"  workspace_error.pdf (inside box median {np.median(err[inside]):.3f} mm, "
          f"outside reach max {err[~reach].max():.0f} mm)")


def fig_overview_frame():
    """Grasp panel of webots_sequence.png, cropped for Figure 1."""
    from PIL import Image
    seq = np.asarray(Image.open(os.path.join(FIG, 'webots_sequence.png')).convert('RGB'))
    white = (seq > 245).all(axis=2).all(axis=0)        # white gutter columns between panels
    edges = np.flatnonzero(np.diff(white.astype(int)))
    starts = [0] + [e + 1 for e in edges if white[e]]
    ends = [e + 1 for e in edges if not white[e]] + [seq.shape[1]]
    x0, x1 = starts[1], ends[1]                         # second panel: gripper on the cube
    frame = seq[40:, x0 + 60:x1]                        # drop the "Neural IK" label
    Image.fromarray(frame).save(os.path.join(FIG, 'webots_frame.png'))
    print(f"  webots_frame.png {frame.shape[1]}x{frame.shape[0]}")


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    # figures/webots_sequence.png is written by experiments/make_demo_gif.py
    fig_overview_frame()
    fig_ablation()
    fig_multihead()
    fig_benchmark()
    fig_workspace()
