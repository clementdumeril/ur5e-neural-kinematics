"""
Verification checks reported in the paper (paper/main.tex, Section 5).

    python experiments/verify_paper.py      (~1 min)

Writes results/verification/verification.json:
  1. the PyTorch FK against the inherited NumPy FK, on random configurations;
  2. the geometric Jacobian against central finite differences of the FK;
  3. closed-form IK round trips on targets in the training box, per branch,
     to quantify "finite but wrong" solutions that a NaN filter would keep;
  4. the train / validation / test split of results/seed_study: disjoint
     indices and nearest-neighbour distance from test to training targets.
"""
import os
import sys
import json
import math

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ('src/kinematics', 'src/models', 'src/control', 'experiments'):
    _p = os.path.join(_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)
if 'controller' not in sys.modules:
    sys.modules['controller'] = type('controller', (), {'Supervisor': type('Supervisor', (), {})})

from ur5 import forward_kinematics as fk_numpy, build_matrix, inverse_kinematics
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch
import jacobian
from multihypothesis_ik import jeu_de_donnees, BRANCHES
from seed_study import split

ROT_DOWN = [math.pi, 0.0, -math.pi / 2]


def check_fk(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, (n, 6))
    T_t = UR5ForwardKinematicsPyTorch().forward(torch.tensor(q, dtype=torch.float64)).numpy()
    dp, dr = [], []
    for i in range(n):
        T_n = fk_numpy(q[i])[0]
        dp.append(np.linalg.norm(T_n[:3, 3] - T_t[i, :3, 3]))
        dr.append(np.abs(T_n[:3, :3] - T_t[i, :3, :3]).max())
    return {'n': n, 'max_position_diff_m': float(max(dp)),
            'max_rotation_entry_diff': float(max(dr))}


def check_jacobian():
    worst, mean = jacobian.verifier_jacobien(n=200, seed=0)
    q_rest = np.array([0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, -np.pi / 2, 0.0])
    return {'n': 200, 'max_abs_diff': worst, 'mean_abs_diff': mean,
            'rest_manipulability': float(jacobian.manipulability(q_rest)),
            'rest_condition': float(jacobian.condition_number(q_rest)),
            'extended_manipulability': float(jacobian.manipulability(np.zeros(6)))}


def check_ik_roundtrip(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    t = np.column_stack((rng.uniform(0.0, 0.4, n), rng.uniform(-0.9, -0.5, n),
                         rng.uniform(0.05, 0.45, n)))
    out = {}
    for sh, wr, el in BRANCHES:
        finite, wrong, worst = 0, 0, 0.0
        for p in t:
            T = build_matrix(p, ROT_DOWN, euler='XYZ')
            try:
                q = inverse_kinematics(T, shoulder=sh, wrist=wr, elbow=el)
            except Exception:
                continue
            if q is None or np.any(np.isnan(q)):
                continue
            finite += 1
            e = np.linalg.norm(fk_numpy(np.asarray(q, dtype=float))[0][:3, 3] - p) * 1000
            if e > 1.0:
                wrong += 1
                worst = max(worst, e)
        out[f'{sh}-{wr}-{el}'] = {'finite': finite, 'off_by_more_than_1mm': wrong,
                                  'worst_mm': round(worst, 1)}
    return {'n_targets': n, 'branches': out}


def check_split():
    X, _, _, fingerprint = jeu_de_donnees()
    tr, va, te = split(len(X))
    sets = [set(tr.tolist()), set(va.tolist()), set(te.tolist())]
    disjoint = not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    Xt, Xs = torch.tensor(X[tr]), torch.tensor(X[te])
    nn = torch.cdist(Xs, Xt).min(1).values.numpy() * 1000
    return {'fingerprint': fingerprint, 'sizes': [len(tr), len(va), len(te)],
            'disjoint': disjoint, 'covers_all': sum(map(len, sets)) == len(X),
            'test_to_train_nn_mm': {'min': float(nn.min()), 'median': float(np.median(nn))}}


def main():
    res = {'fk_torch_vs_numpy': check_fk(),
           'jacobian_vs_finite_differences': check_jacobian(),
           'closed_form_roundtrip': check_ik_roundtrip(),
           'split': check_split(),
           'torch': torch.__version__, 'numpy': np.__version__}
    out = os.path.join(_ROOT, 'results', 'verification')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'verification.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
