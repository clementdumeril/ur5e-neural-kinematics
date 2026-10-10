"""
Writes paper/numbers.tex: every campaign number quoted in main.tex, as a LaTeX
macro computed from a result file of the repository.

    python paper/make_numbers.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEEDS = os.path.join(ROOT, 'results', 'seed_study')
OUT = os.path.join(ROOT, 'paper', 'numbers.tex')

macros = {}


def put(name, value):
    macros[name] = value


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} \\pm {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def load(pattern):
    return [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob(os.path.join(SEEDS, pattern)))]


# --- Part A: paired ablation -------------------------------------------------
A = load('A_*.json')
by = {}
for r in A:
    by.setdefault(r['seed'], {})[r['w_phys']] = r
seeds = sorted(s for s in by if 0.0 in by[s] and 1.0 in by[s])
if seeds:
    p0 = [by[s][0.0]['test_pos_mm'] for s in seeds]
    p1 = [by[s][1.0]['test_pos_mm'] for s in seeds]
    r0 = [by[s][0.0]['test_rot_deg'] for s in seeds]
    r1 = [by[s][1.0]['test_rot_deg'] for s in seeds]
    red = [100 * (a - b) / a for a, b in zip(p0, p1)]
    put('nSeedsA', str(len(seeds)))
    put('ablPosZero', ms(p0)); put('ablPosOne', ms(p1))
    put('ablRotZero', ms(r0, 4)); put('ablRotOne', ms(r1, 4))
    put('ablRed', ms(red, 1))
    put('ablRedMin', f"{min(red):.1f}"); put('ablRedMax', f"{max(red):.1f}")
    put('ablWins', f"{sum(b < a for a, b in zip(p0, p1))}")
    # paired t-test on the per-seed position difference, and 95 % CI
    from scipy import stats
    diff = np.array(p0) - np.array(p1)
    n = len(diff)
    half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    put('ablDiff', f"{diff.mean():.3f}")
    put('ablDiffLo', f"{diff.mean() - half:.3f}"); put('ablDiffHi', f"{diff.mean() + half:.3f}")
    halfr = stats.t.ppf(0.975, n - 1) * np.std(red, ddof=1) / np.sqrt(n)
    put('ablRedLo', f"{np.mean(red) - halfr:.0f}"); put('ablRedHi', f"{np.mean(red) + halfr:.0f}")
    put('ablP', f"{stats.ttest_rel(p0, p1).pvalue:.2f}")
    put('ablWilcoxonP', f"{stats.wilcoxon(p0, p1).pvalue:.2f}")
    # seeds needed to detect the observed mean difference (paired, alpha 0.05, power 0.8)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    put('ablSeedsNeeded', f"{int(np.ceil((z * diff.std(ddof=1) / diff.mean()) ** 2)) + 2}")
    put('splitTrain', f"{A[0]['split'][0]:,}"); put('splitVal', f"{A[0]['split'][1]:,}")
    put('splitTest', f"{A[0]['split'][2]:,}"); put('nTargets', f"{A[0]['n_targets']:,}")
    put('fingerprint', A[0]['fingerprint'])

# --- Parts B and D: multi-head sweep ------------------------------------------
for part, tag in (('B', 'Four'), ('D', 'Eight')):
    runs = load(f'{part}_*.json')
    if not runs:
        continue
    put(f'nSeeds{tag}', str(len({r['seed'] for r in runs})))
    for K in sorted({r['K'] for r in runs}):
        rr = [r for r in runs if r['K'] == K]
        w = {1: 'One', 2: 'Two', 4: 'FourK', 8: 'EightK'}[K]
        put(f'mh{tag}{w}Pos', ms([r['test_pos_mm'] for r in rr], 3 if K > 1 else 2))
        put(f'mh{tag}{w}Rot', ms([r['test_rot_deg'] for r in rr], 2))
        put(f'mh{tag}{w}Rec', ms([100 * r['recall'] for r in rr], 1))
        put(f'mh{tag}{w}RecMin', f"{100 * min(r['recall'] for r in rr):.1f}")
        put(f'mh{tag}{w}RecMax', f"{100 * max(r['recall'] for r in rr):.1f}")
        put(f'mh{tag}{w}Params', f"{rr[0]['n_params']:,}")
    # recall ceiling: K heads can cover at most min(K, n_i) of the n_i branches of target i
    cache = (os.path.join(ROOT, 'checkpoints', 'dataset_branches.npz') if part == 'B'
             else os.path.join(SEEDS, 'dataset_8branches.npz'))
    if os.path.exists(cache):
        import sys
        sys.path.insert(0, os.path.join(ROOT, 'experiments'))
        from seed_study import split
        Mte = np.load(cache)['M'][split(len(np.load(cache)['M']))[2]]
        nb = Mte.sum(1)
        for K in (1, 2, 4, 8):
            w = {1: 'One', 2: 'Two', 4: 'FourK', 8: 'EightK'}[K]
            put(f'mh{tag}{w}Ceil', f"{100 * np.minimum(K, nb).sum() / nb.sum():.1f}")
    if part == 'D':
        put('branchesPerTarget', f"{runs[0]['branches_per_target']:.2f}")
        av = runs[0]['branch_availability']
        put('availUp', f"{100 * av['left-up-up']:.0f}")
        put('availDown', f"{100 * av['left-down-up']:.1f}")

# --- Part C: shipped model on the test split ----------------------------------
c = os.path.join(SEEDS, 'C_shipped.json')
if os.path.exists(c):
    C = json.load(open(c))
    put('shipTestPos', f"{C['test_pos_mm']:.3f}"); put('shipTestRot', f"{C['test_rot_deg']:.4f}")
    put('shipTestMed', f"{C['test_pos_median_mm']:.3f}"); put('shipTestPninetyfive', f"{C['test_pos_p95_mm']:.3f}")
    put('shipTestMax', f"{C['test_pos_max_mm']:.3f}")

# --- Benchmark ----------------------------------------------------------------
B = {r['solveur']: r for r in json.load(open(os.path.join(ROOT, 'results', 'benchmark', 'benchmark.json')))['resultats']}
cf, nn1 = B['forme fermee analytique'], B['reseau, sortie unique']
nn2, ik = B['reseau, 2 tetes + selection'], B['IKPY, pose complete']
d1, d8 = B['DLS, depart unique'], B['DLS, 8 relances']
for name, r in (('Cf', cf), ('NnOne', nn1), ('NnTwo', nn2), ('Ikpy', ik), ('DlsOne', d1), ('DlsEight', d8)):
    put(f'bench{name}Med', f"{r['temps_median_ms']:.3f}" if r['temps_median_ms'] < 10 else f"{r['temps_median_ms']:.1f}")
    put(f'bench{name}Pninetyfive', f"{r['temps_p95_ms']:.3f}" if r['temps_p95_ms'] < 10 else f"{r['temps_p95_ms']:.1f}")
    put(f'bench{name}Fail', f"{100 * r['taux_echec']:.1f}")
put('benchNnOneErr', f"{nn1['erreur_mediane_mm']:.3f}")
put('benchNnTwoErr', f"{nn2['erreur_mediane_mm']:.3f}")
put('benchSlowerCf', f"{nn1['temps_median_ms'] / cf['temps_median_ms']:.1f}")
put('benchFasterIkpy', f"{ik['temps_median_ms'] / nn1['temps_median_ms']:.0f}")
put('benchTwoHeadsSlower', f"{nn2['temps_median_ms'] / nn1['temps_median_ms']:.0f}")

# --- Verification -------------------------------------------------------------
V = json.load(open(os.path.join(ROOT, 'results', 'verification', 'verification.json')))
Vi = json.load(open(os.path.join(ROOT, 'results', 'verification', 'verification_inherited_ik.json')))
put('fkDiff', f"{V['fk_torch_vs_numpy']['max_position_diff_m']:.1e}".replace('e-', '\\times 10^{-') + '}')
put('jacMax', f"{V['jacobian_vs_finite_differences']['max_abs_diff']:.1e}".replace('e-', '\\times 10^{-') + '}')
put('jacRestCond', f"{V['jacobian_vs_finite_differences']['rest_condition']:.1f}")
put('jacRestManip', f"{V['jacobian_vs_finite_differences']['rest_manipulability']:.4f}")
bi = Vi['closed_form_roundtrip']['branches']
put('bugWorst', f"{max(v['worst_mm'] for v in bi.values()):.0f}")
put('rtTargets', f"{Vi['closed_form_roundtrip']['n_targets']:,}")
put('nnMin', f"{V['split']['test_to_train_nn_mm']['min']:.2f}")
put('nnMedian', f"{V['split']['test_to_train_nn_mm']['median']:.1f}")

# --- Workspace map ------------------------------------------------------------
hm = os.path.join(ROOT, 'results', 'heatmap', 'workspace_error.npz')
if os.path.exists(hm):
    d = np.load(hm)
    XX, YY, err, reach = d['XX'], d['YY'], d['err'], d['reach']
    box = (XX >= 0.0) & (XX <= 0.4) & (YY >= -0.9) & (YY <= -0.5)
    put('mapZ', f"{float(d['Z']):.2f}")
    put('mapIn', f"{np.median(err[box & reach]):.3f}")
    put('mapOut', f"{np.median(err[~box & reach]):.1f}")
    put('mapRatio', f"{np.median(err[~box & reach]) / np.median(err[box & reach]):.0f}")
    put('mapUnreachMax', f"{err[~reach].max() / 1000:.2f}")

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('% Generated by paper/make_numbers.py -- do not edit by hand.\n')
    for k, v in sorted(macros.items()):
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(f'{len(macros)} macros -> {OUT}')
