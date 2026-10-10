"""
Seed study: how much of each headline number is training noise?

The original experiments each ran once (seed 42) and selected the best epoch on
the same validation set they reported. This script reruns them with several
training seeds and a three-way split:

    train 80 %  /  validation 10 % (epoch selection)  /  test 10 % (reported)

so every number it writes is measured on targets that played no part in
training or model selection.

Part A  single-output network, one IK branch, w_phys in {0, 1}, 5 seeds
        -> mean and spread of the physics-term effect, paired by seed
Part B  multi-head network on branch-mixed data, K in {1, 2, 4, 8}, 3 seeds
        -> accuracy and branch recall per K, and how stable recall is
Part C  the shipped checkpoint, evaluated on the same test targets
Part D  Part B again on a dataset that contains all eight closed-form branches.
        The inherited solver returned wrong angles for every elbow-down branch
        (theta2 was not paired with theta3); after the fix in src/control/ur5.py
        those branches exist, and the cached dataset of Part B lacks them.

All runs share the cached dataset of experiments/multihypothesis_ik.py
(fingerprint reported in every file). Results go to results/seed_study/, one
JSON per run, so an interrupted campaign can be resumed: finished runs are
skipped.

    python experiments/seed_study.py            # everything, ~3.5 h on 4 CPU threads
    SEED_PART=A python experiments/seed_study.py
    SEED_EPOCHS=2 SEED_OUT=results/seed_smoke python experiments/seed_study.py   # quick check
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ('src/kinematics', 'src/models', 'src/control', 'src/training', 'experiments'):
    _p = os.path.join(_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)
if 'controller' not in sys.modules:
    sys.modules['controller'] = type('controller', (), {'Supervisor': type('Supervisor', (), {})})

from pinn import PINN6DOF
from multihead import MultiHeadIK, perte_multi_hypotheses
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch
from multihypothesis_ik import (jeu_de_donnees, erreur_pose, ecart_angulaire,
                                BRANCHES, K_MAX)

EPOCHS = int(os.environ.get('SEED_EPOCHS', 100))
PART = os.environ.get('SEED_PART', 'ABC').upper()
OUT = os.path.join(_ROOT, os.environ.get('SEED_OUT', os.path.join('results', 'seed_study')))
R_OUTIL = 0.05
SEEDS_A = [0, 1, 2, 3, 4]
SEEDS_B = [0, 1, 2]
KS = [1, 2, 4, 8]
SPLIT_SEED = 1234
BRANCH_A = BRANCHES.index(('left', 'up', 'up'))


def lr_schedule(opt, ep):
    for seuil, lr in ((30, 5e-4), (60, 1e-4), (80, 2e-5)):
        if ep == seuil:
            for g in opt.param_groups:
                g['lr'] = lr


def split(n):
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    a, b = int(0.8 * n), int(0.9 * n)
    return perm[:a], perm[a:b], perm[b:]


def pose_errors(fk, q, x, R_ref):
    """Mean position error (mm) and mean geodesic orientation error (deg)."""
    T = fk.forward(q)
    pos = torch.norm(T[:, :3, 3] - x, dim=1).mean().item() * 1000
    Rres = torch.bmm(R_ref.transpose(1, 2), T[:, :3, :3])
    cos = ((Rres[:, 0, 0] + Rres[:, 1, 1] + Rres[:, 2, 2]) - 1) / 2
    rot = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()
    return pos, rot


# ---------------------------------------------------------------------------
# Part A: single output, one branch
# ---------------------------------------------------------------------------
def run_single(X, y, idx, fk, w_phys, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = idx
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])
    Xte, yte = torch.tensor(X[te]), torch.tensor(y[te])
    with torch.no_grad():
        Rva = fk.forward(yva)[:, :3, :3]
        Rte = fk.forward(yte)[:, :3, :3]

    model = PINN6DOF(hidden_dim=512)
    model.mean_in.copy_(Xtr.mean(0)); model.std_in.copy_(Xtr.std(0).clamp(min=1e-3))
    model.mean_out.copy_(ytr.mean(0)); model.std_out.copy_(ytr.std(0).clamp(min=1e-3))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)

    best = {'val_pos_mm': float('inf')}
    best_state = None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            pq = model(bx)
            loss = 0.1 * mse(pq, by)
            if w_phys > 0:
                Tp, Tr = fk.forward(pq), fk.forward(by)
                loss = loss + w_phys * (mse(Tp[:, :3, 3], bx)
                                        + R_OUTIL ** 2 * mse(Tp[:, :3, :3], Tr[:, :3, :3]))
            loss.backward()
            opt.step()
        lr_schedule(opt, ep)
        model.eval()
        with torch.no_grad():
            vp, vr = pose_errors(fk, model(Xva), Xva, Rva)
        if vp < best['val_pos_mm']:
            best = {'val_pos_mm': vp, 'val_rot_deg': vr, 'epoch': ep + 1}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp, trot = pose_errors(fk, model(Xte), Xte, Rte)
    return dict(best, test_pos_mm=tp, test_rot_deg=trot,
                seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
# Part B: multi-head, branch-mixed, no labels
# ---------------------------------------------------------------------------
def best_of_k(fk, q, x, R_ref):
    B, K, _ = q.shape
    T = fk.forward(q.reshape(B * K, 6))
    pos = T[:, :3, 3].reshape(B, K, 3)
    rot = T[:, :3, :3].reshape(B, K, 3, 3)
    pick = erreur_pose(pos, rot, x, R_ref).argmin(1)
    i = torch.arange(B)
    pos_mm = torch.norm(pos[i, pick] - x, dim=-1).mean().item() * 1000
    Rres = torch.bmm(R_ref.transpose(1, 2), rot[i, pick])
    cos = ((Rres[:, 0, 0] + Rres[:, 1, 1] + Rres[:, 2, 2]) - 1) / 2
    rot_deg = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()
    return pos_mm, rot_deg


def recall(q, Q, M, seuil=0.25):
    """Fraction of existing branches reproduced by at least one head (periodic distance)."""
    found = np.zeros_like(M)
    for i in range(len(q)):
        valid = np.where(M[i])[0]
        for k in range(q.shape[1]):
            d = ecart_angulaire(Q[i][valid], q[i, k])
            j = int(d.argmin())
            if d[j] < seuil:
                found[i, valid[j]] = True
    per_branch = {"-".join(BRANCHES[b]): round(float(found[M[:, b], b].mean()), 4)
                  for b in range(K_MAX) if M[:, b].any()}
    return float((found & M).sum() / M.sum()), per_branch


def run_multi(X, Q, M, idx, fk, K, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = idx
    first = M.argmax(1)
    qref = Q[np.arange(len(Q)), first]
    Xtr, Xva, Xte = (torch.tensor(X[s]) for s in (tr, va, te))
    with torch.no_grad():
        Rtr, Rva, Rte = (fk.forward(torch.tensor(qref[s]))[:, :3, :3] for s in (tr, va, te))

    model = MultiHeadIK(n_heads=K)
    model.mean_in.copy_(Xtr.mean(0)); model.std_in.copy_(Xtr.std(0).clamp(min=1e-3))
    ang = torch.tensor(Q[tr][M[tr]])
    model.mean_out.copy_(ang.mean(0)); model.std_out.copy_(ang.std(0).clamp(min=1e-3))
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(Xtr, Rtr), batch_size=256, shuffle=True)

    best = {'val_pos_mm': float('inf')}
    best_state = None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for bx, br in loader:
            opt.zero_grad()
            loss, _, _ = perte_multi_hypotheses(model(bx), bx, br, fk, R_OUTIL)
            loss.backward()
            opt.step()
        lr_schedule(opt, ep)
        model.eval()
        with torch.no_grad():
            vp, vr = best_of_k(fk, model(Xva), Xva, Rva)
        if vp < best['val_pos_mm']:
            best = {'val_pos_mm': vp, 'val_rot_deg': vr, 'epoch': ep + 1}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp, trot = best_of_k(fk, model(Xte), Xte, Rte)
        qte = model(Xte).numpy()
    rec, per_branch = recall(qte, Q[te], M[te])
    n_params = sum(p.numel() for p in model.parameters())
    return dict(best, test_pos_mm=tp, test_rot_deg=trot, recall=rec,
                recall_per_branch=per_branch, n_params=n_params,
                seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
# Part D dataset: all eight closed-form branches, after the theta2 fix
# ---------------------------------------------------------------------------
def dataset_8_branches():
    """
    Same targets as the cached dataset (same generator, same seed), solved with
    the corrected closed-form IK, so the elbow-down branches are now present.
    Every solution is still checked by forward kinematics before it is kept.
    """
    import hashlib
    from multihypothesis_ik import construire, N_TARGETS
    path = os.path.join(_ROOT, 'results', 'seed_study', 'dataset_8branches.npz')
    if not os.path.exists(path):
        print("  generating the 8-branch dataset (~11 min)...", flush=True)
        t0 = time.time()
        X, Q, M = construire(N_TARGETS)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, X=X, Q=Q, M=M)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    d = np.load(path)
    X, Q, M = d['X'], d['Q'], d['M']
    fp = hashlib.sha256(X.tobytes() + M.tobytes()).hexdigest()[:16]
    return X, Q, M, fp


# ---------------------------------------------------------------------------
def save(name, payload):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def done(name):
    return os.path.exists(os.path.join(OUT, name))


def main():
    torch.set_num_threads(int(os.environ.get('SEED_THREADS', 4)))
    fk = UR5ForwardKinematicsPyTorch()
    X, Q, M, fingerprint = jeu_de_donnees()
    idx = split(len(X))
    common = {'fingerprint': fingerprint, 'n_targets': len(X), 'epochs': EPOCHS,
              'split': [len(i) for i in idx], 'split_seed': SPLIT_SEED,
              'torch': torch.__version__}
    print(f"{len(X)} targets | fingerprint {fingerprint} | split {common['split']} "
          f"| {EPOCHS} epochs | out {OUT}", flush=True)

    if 'A' in PART:
        y = Q[:, BRANCH_A]
        assert M[:, BRANCH_A].all(), "left-up-up must be valid on every target"
        for seed in SEEDS_A:
            for w in (0.0, 1.0):
                name = f"A_wphys{w:g}_seed{seed}.json"
                if done(name):
                    continue
                print(f"[A] w_phys={w:g} seed={seed}", flush=True)
                r = run_single(X, y, idx, fk, w, seed)
                save(name, dict(common, part='A', w_phys=w, seed=seed, **r))
                print(f"    test {r['test_pos_mm']:.4f} mm / {r['test_rot_deg']:.4f} deg "
                      f"(ep {r['epoch']}, {r['seconds']:.0f} s)", flush=True)

    if 'B' in PART:
        for seed in SEEDS_B:
            for K in KS:
                name = f"B_K{K}_seed{seed}.json"
                if done(name):
                    continue
                print(f"[B] K={K} seed={seed}", flush=True)
                r = run_multi(X, Q, M, idx, fk, K, seed)
                save(name, dict(common, part='B', K=K, seed=seed, **r))
                print(f"    test {r['test_pos_mm']:.4f} mm / {r['test_rot_deg']:.4f} deg, "
                      f"recall {r['recall']:.3f} (ep {r['epoch']}, {r['seconds']:.0f} s)",
                      flush=True)

    if 'D' in PART:
        X8, Q8, M8, fp8 = dataset_8_branches()
        idx8 = split(len(X8))
        common8 = dict(common, fingerprint=fp8, n_targets=len(X8),
                       split=[len(i) for i in idx8],
                       branches_per_target=float(M8.sum(1).mean()),
                       branch_availability={"-".join(b): round(float(M8[:, i].mean()), 4)
                                            for i, b in enumerate(BRANCHES)})
        for seed in SEEDS_B:
            for K in KS:
                name = f"D_K{K}_seed{seed}.json"
                if done(name):
                    continue
                print(f"[D] K={K} seed={seed} (8-branch data)", flush=True)
                r = run_multi(X8, Q8, M8, idx8, fk, K, seed)
                save(name, dict(common8, part='D', K=K, seed=seed, **r))
                print(f"    test {r['test_pos_mm']:.4f} mm / {r['test_rot_deg']:.4f} deg, "
                      f"recall {r['recall']:.3f} (ep {r['epoch']}, {r['seconds']:.0f} s)",
                      flush=True)

    if 'C' in PART and not done('C_shipped.json'):
        model = PINN6DOF.from_file(os.path.join(_ROOT, 'checkpoints',
                                                'pinn_model_true_physics.pth'))
        model.eval()
        te = idx[2]
        Xte = torch.tensor(X[te])
        with torch.no_grad():
            Rte = fk.forward(torch.tensor(Q[te, BRANCH_A]))[:, :3, :3]
            q = model(Xte)
            tp, trot = pose_errors(fk, q, Xte, Rte)
            d = torch.norm(fk.forward(q)[:, :3, 3] - Xte, dim=1).numpy() * 1000
        save('C_shipped.json', dict(common, part='C', test_pos_mm=tp, test_rot_deg=trot,
                                    test_pos_median_mm=float(np.median(d)),
                                    test_pos_p95_mm=float(np.percentile(d, 95)),
                                    test_pos_max_mm=float(d.max())))
        print(f"[C] shipped model on test: {tp:.4f} mm / {trot:.4f} deg", flush=True)


if __name__ == "__main__":
    main()
