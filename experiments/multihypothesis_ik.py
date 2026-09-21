"""
IK multi-hypotheses : modeliser les 8 branches au lieu de les contourner.

Ce que l'ablation avait etabli
------------------------------
Entraine sur des donnees melangeant les 8 branches de solution, un reseau a une
seule sortie echoue :

    supervise (w_phys=0)   911,06 mm
    physique  (w_phys=1)    69,57 mm
    physique  (w_phys=20)    3,68 mm, mais 71,6 deg d'orientation

Le depot s'en sort en imposant une seule famille de solutions dans le
generateur. Le probleme est evite, pas resolu.

Ce que cette experience teste
-----------------------------
Un reseau a K tetes, entraine avec un min relache sur la perte PHYSIQUE seule :

    L = min_k  L_physique( q(k) )

Aucune etiquette de branche n'est fournie. Si le reseau atteint une precision
comparable a celle du modele entraine sur une branche unique (0,187 mm), alors
il n'est plus necessaire de choisir la branche a la main : le reseau decouvre
lui-meme la structure multivaluee du probleme.

On mesure aussi ce que les tetes ont appris : combien de branches distinctes
elles couvrent, et si elles se sont effondrees les unes sur les autres.
"""
import os
import sys
import json
import math
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

if 'controller' not in sys.modules:
    sys.modules['controller'] = type(
        'controller', (), {'Supervisor': type('Supervisor', (), {})})

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ('src/kinematics', 'src/models', 'src/control', 'src/training'):
    _p = os.path.join(_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ur5 import build_matrix, inverse_kinematics            # noqa: E402
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch      # noqa: E402
from multihead import MultiHeadIK, perte_multi_hypotheses   # noqa: E402

ROT_DOWN = [math.pi, 0.0, -math.pi / 2]
BRANCHES = [(s, w, e)
            for s in ('left', 'right')
            for w in ('up', 'down')
            for e in ('up', 'down')]
K_MAX = len(BRANCHES)

N_TARGETS = int(os.environ.get('MH_TARGETS', 25000))
EPOCHS = int(os.environ.get('MH_EPOCHS', 100))
N_HEADS = int(os.environ.get('MH_HEADS', 8))
R_OUTIL = 0.05
CACHE = os.path.join(_ROOT, 'checkpoints', 'dataset_branches.npz')


# =====================================================================
# 1. Jeu de donnees : toutes les branches valides de chaque cible
# =====================================================================
def construire(n_targets, seed=42):
    """
    Pour chaque cible, resout les 8 branches et ne garde que celles qui
    atteignent REELLEMENT la pose demandee.

    Le controle par cinematique directe n'est pas une precaution de style :
    inverse_kinematics rend pour certaines combinaisons des angles finis mais
    faux, jusqu'a 784 mm de la cible. Filtrer sur NaN ne suffit pas.
    """
    fk = UR5ForwardKinematicsPyTorch()
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 0.4, n_targets)
    ys = rng.uniform(-0.9, -0.5, n_targets)
    zs = rng.uniform(0.05, 0.45, n_targets)
    cibles = np.column_stack((xs, ys, zs))

    X, Q, M = [], [], []
    for t in cibles:
        T = build_matrix(t, ROT_DOWN, euler='XYZ')
        sols = np.zeros((K_MAX, 6), dtype=np.float32)
        masque = np.zeros(K_MAX, dtype=bool)
        for i, (sh, wr, el) in enumerate(BRANCHES):
            try:
                q = inverse_kinematics(T, shoulder=sh, wrist=wr, elbow=el)
            except Exception:
                continue
            if q is None or np.any(np.isnan(q)):
                continue
            q = np.asarray(q, dtype=np.float32)
            with torch.no_grad():
                p = fk.forward_pos(torch.tensor(q).unsqueeze(0))[0].numpy()
            if np.linalg.norm(p - t) > 1e-4:
                continue
            sols[i], masque[i] = q, True

        if masque.sum() >= 1:
            X.append(t)
            Q.append(sols)
            M.append(masque)

    return (np.asarray(X, dtype=np.float32),
            np.asarray(Q, dtype=np.float32),
            np.asarray(M))


def jeu_de_donnees():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        print(f"  cache relu : {CACHE}")
        return d['X'], d['Q'], d['M']
    print("  generation (environ 11 minutes)...", flush=True)
    t0 = time.time()
    X, Q, M = construire(N_TARGETS)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, Q=Q, M=M)
    print(f"  genere en {time.time() - t0:.0f} s, mis en cache")
    return X, Q, M


# =====================================================================
# 2. Entrainement
# =====================================================================
def entrainer(X, Q, M, fk, epochs=EPOCHS, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Orientation de reference : celle de la premiere branche valide. Toutes
    # les branches atteignent la meme pose, le choix est donc sans effet.
    premiere = M.argmax(axis=1)
    q_ref = Q[np.arange(len(Q)), premiere]

    n = len(X)
    coupe = int(0.9 * n)
    Xtr = torch.tensor(X[:coupe]);  Xva = torch.tensor(X[coupe:])
    Rtr = torch.tensor(q_ref[:coupe]); Rva = torch.tensor(q_ref[coupe:])

    with torch.no_grad():
        R_ref_tr = fk.forward(Rtr)[:, :3, :3]
        R_ref_va = fk.forward(Rva)[:, :3, :3]

    loader = DataLoader(TensorDataset(Xtr, R_ref_tr), batch_size=256, shuffle=True)

    modele = MultiHeadIK(n_heads=N_HEADS)
    modele.mean_in.copy_(Xtr.mean(0))
    modele.std_in.copy_(Xtr.std(0).clamp(min=1e-3))
    # Les sorties sont des angles : on normalise sur l'ensemble des branches
    # valides, pas sur une seule, pour ne privilegier aucune famille.
    angles = torch.tensor(Q[M])
    modele.mean_out.copy_(angles.mean(0))
    modele.std_out.copy_(angles.std(0).clamp(min=1e-3))

    opt = optim.Adam(modele.parameters(), lr=1e-3)
    meilleur = float('inf')
    chemin = os.path.join(_ROOT, 'checkpoints', 'multihead_ik.pth')

    print(f"\n  {sum(p.numel() for p in modele.parameters()):,} parametres, "
          f"{N_HEADS} tetes, {epochs} epoques\n")

    for ep in range(epochs):
        modele.train()
        for bx, br in loader:
            opt.zero_grad()
            perte, _, _ = perte_multi_hypotheses(modele(bx), bx, br, fk, R_OUTIL)
            perte.backward()
            opt.step()

        if ep == 30:
            for g in opt.param_groups: g['lr'] = 5e-4
        if ep == 60:
            for g in opt.param_groups: g['lr'] = 1e-4
        if ep == 80:
            for g in opt.param_groups: g['lr'] = 2e-5

        modele.eval()
        with torch.no_grad():
            q = modele(Xva)
            B, Kh, _ = q.shape
            T = fk.forward(q.reshape(B * Kh, 6))
            pos = T[:, :3, 3].reshape(B, Kh, 3)
            rot = T[:, :3, :3].reshape(B, Kh, 3, 3)
            d = torch.norm(pos - Xva.unsqueeze(1), dim=-1)          # (B,K)
            meilleure = d.argmin(1)
            pos_mm = d.min(1).values.mean().item() * 1000

            idx = torch.arange(B)
            Rp = rot[idx, meilleure]
            Rres = torch.bmm(R_ref_va.transpose(1, 2), Rp)
            cos = ((Rres[:, 0, 0] + Rres[:, 1, 1] + Rres[:, 2, 2]) - 1) / 2
            rot_deg = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()
            tetes_utiles = len(torch.unique(meilleure))

        if pos_mm < meilleur:
            meilleur = pos_mm
            modele.save_model(chemin)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    ep {ep+1:03d} | best-of-K {pos_mm:8.3f} mm | "
                  f"{rot_deg:6.3f} deg | tetes gagnantes utilisees : "
                  f"{tetes_utiles}/{N_HEADS}", flush=True)

    return modele, meilleur, (Xva, Rva, R_ref_va), chemin


# =====================================================================
# 3. Qu'ont appris les tetes ?
# =====================================================================
def analyser(modele, X, Q, M, fk, n=2000):
    """
    Chaque tete s'est-elle specialisee sur une branche, et lesquelles sont
    couvertes ? Une tete est attribuee a la branche dont elle est la plus
    proche en distance articulaire.
    """
    Xs = torch.tensor(X[-n:]); Qs = Q[-n:]; Ms = M[-n:]
    with torch.no_grad():
        q = modele(Xs).numpy()                     # (n, K, 6)

    attribution = np.full((len(Xs), N_HEADS), -1)
    for i in range(len(Xs)):
        valides = np.where(Ms[i])[0]
        if len(valides) == 0:
            continue
        for k in range(N_HEADS):
            d = np.linalg.norm(Qs[i][valides] - q[i, k], axis=1)
            attribution[i, k] = valides[d.argmin()]

    print("\n  Specialisation des tetes")
    print("  " + "-" * 62)
    print(f"  {'tete':>5}  {'branche dominante':<26} {'constance':>10}")
    couvertes = set()
    for k in range(N_HEADS):
        col = attribution[:, k]
        col = col[col >= 0]
        if len(col) == 0:
            print(f"  {k:>5}  {'(aucune)':<26} {'-':>10}")
            continue
        vals, cnt = np.unique(col, return_counts=True)
        dom = vals[cnt.argmax()]
        couvertes.add(int(dom))
        nom = "-".join(BRANCHES[dom])
        print(f"  {k:>5}  {nom:<26} {cnt.max() / len(col):>9.0%}")

    moy_valides = Ms.sum(axis=1).mean()
    print(f"\n  Branches distinctes couvertes : {len(couvertes)} "
          f"(moyenne de {moy_valides:.2f} branches valides par cible)")
    return len(couvertes)


# =====================================================================
def main():
    print("=" * 74)
    print("IK MULTI-HYPOTHESES")
    print("=" * 74)

    fk = UR5ForwardKinematicsPyTorch()

    print("\n[1/3] Jeu de donnees")
    X, Q, M = jeu_de_donnees()
    print(f"  {len(X)} cibles, {M.sum() / len(M):.2f} branches valides en moyenne")

    print("\n[2/3] Entrainement (perte physique seule, aucune etiquette de branche)")
    modele, meilleur, _, chemin = entrainer(X, Q, M, fk)

    print("\n[3/3] Analyse")
    modele = MultiHeadIK.from_file(chemin)
    modele.eval()
    couvertes = analyser(modele, X, Q, M, fk)

    # --- comparaison avec les resultats mesures de l'ablation ---------
    print("\n" + "=" * 74)
    print("COMPARAISON -- meme probleme, meme jeu de cibles")
    print("=" * 74)
    abl = os.path.join(_ROOT, 'checkpoints', 'ablation_physics_loss.json')
    lignes = []
    if os.path.exists(abl):
        for r in json.load(open(abl))['resultats']:
            lignes.append((r['nom'], r['best_pos_mm'], r['best_rot_deg']))
    lignes.append((f"multi-hypotheses, {N_HEADS} tetes (best-of-K)", meilleur, float('nan')))

    print(f"  {'Configuration':<40} {'Position':>12} {'Orientation':>13}")
    print("  " + "-" * 68)
    for nom, p, r in lignes:
        r_txt = "     --" if r != r else f"{r:8.3f} deg"
        print(f"  {nom:<40} {p:>9.3f} mm {r_txt:>13}")
    print("=" * 74)

    sortie = os.path.join(_ROOT, 'checkpoints', 'multihypothesis_ik.json')
    json.dump({'n_heads': N_HEADS, 'epoques': EPOCHS, 'n_cibles': len(X),
               'best_of_k_mm': meilleur, 'branches_couvertes': couvertes},
              open(sortie, 'w'), indent=2)
    print(f"\nResultats : {sortie}")


if __name__ == "__main__":
    main()
