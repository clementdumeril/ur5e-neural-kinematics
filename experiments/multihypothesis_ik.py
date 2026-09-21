"""
IK multi-hypotheses : modeliser les branches au lieu de les eviter.

Ce que l'ablation avait etabli
------------------------------
Entraine sur des donnees melangeant les branches de solution, un reseau a une
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

Aucune etiquette de branche n'est fournie. Le reseau doit decouvrir seul la
structure multivaluee du probleme.

Choix de mesure, et pourquoi ils comptent
-----------------------------------------
1. La selection best-of-K se fait sur la POSE COMPLETE, avec la meme ponderation
   que la perte d'entrainement. Selectionner sur la position seule puis
   rapporter l'orientation de la tete ainsi choisie ferait diverger l'objectif
   d'entrainement, la regle d'inference et la metrique publiee.

2. Les distances entre configurations articulaires sont PERIODIQUES. Deux angles
   a +179 et -179 degres different physiquement de 2 degres, pas de 358. Une
   norme naive fausserait l'attribution des tetes aux branches.

3. On mesure le RAPPEL par branche, pas seulement le nombre de branches
   couvertes. Couvrir 2 branches sur 2 disponibles et 2 sur 4 sont deux
   resultats tres differents, et seul le second serait un effondrement partiel.

Usage
-----
    python experiments/multihypothesis_ik.py            # K = 8
    MH_HEADS=4 python experiments/multihypothesis_ik.py # K = 4
"""
import os
import sys
import json
import math
import time
import hashlib

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
# Outils
# =====================================================================
def ecart_angulaire(a, b):
    """
    Distance entre deux jeux d'angles, en tenant compte de la PERIODICITE.

    atan2(sin(d), cos(d)) ramene chaque ecart dans [-pi, pi] : +179 et -179
    degres se retrouvent a 2 degres l'un de l'autre, et non a 358.
    """
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return np.linalg.norm(d, axis=-1)


def erreur_pose(pos, rot, cible, R_ref, r_outil=R_OUTIL):
    """Meme combinaison que la perte d'entrainement : position + orientation."""
    e_pos = ((pos - cible.unsqueeze(1)) ** 2).sum(-1)
    e_rot = ((rot - R_ref.unsqueeze(1)) ** 2).sum((-1, -2))
    return e_pos + (r_outil ** 2) * e_rot


# =====================================================================
# 1. Jeu de donnees
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
    """
    Un seul jeu de donnees, mis en cache, partage par toutes les experiences.

    Son empreinte est reportee dans chaque fichier de resultats : c'est ce qui
    permet d'affirmer que deux experiences ont bien tourne sur les memes cibles,
    au lieu de l'esperer.
    """
    if not os.path.exists(CACHE):
        print("  generation (environ 11 minutes)...", flush=True)
        t0 = time.time()
        X, Q, M = construire(N_TARGETS)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, X=X, Q=Q, M=M)
        print(f"  genere en {time.time() - t0:.0f} s, mis en cache")
    d = np.load(CACHE)
    X, Q, M = d['X'], d['Q'], d['M']
    empreinte = hashlib.sha256(X.tobytes()).hexdigest()[:16]
    return X, Q, M, empreinte


# =====================================================================
# 2. Entrainement
# =====================================================================
def entrainer(X, Q, M, fk, n_heads, epochs=EPOCHS, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    premiere = M.argmax(axis=1)
    q_ref = Q[np.arange(len(Q)), premiere]

    coupe = int(0.9 * len(X))
    Xtr = torch.tensor(X[:coupe]);  Xva = torch.tensor(X[coupe:])
    Rtr = torch.tensor(q_ref[:coupe]); Rva = torch.tensor(q_ref[coupe:])

    with torch.no_grad():
        R_ref_tr = fk.forward(Rtr)[:, :3, :3]
        R_ref_va = fk.forward(Rva)[:, :3, :3]

    loader = DataLoader(TensorDataset(Xtr, R_ref_tr), batch_size=256, shuffle=True)

    modele = MultiHeadIK(n_heads=n_heads)
    modele.mean_in.copy_(Xtr.mean(0))
    modele.std_in.copy_(Xtr.std(0).clamp(min=1e-3))
    angles = torch.tensor(Q[M])
    modele.mean_out.copy_(angles.mean(0))
    modele.std_out.copy_(angles.std(0).clamp(min=1e-3))

    n_params = sum(p.numel() for p in modele.parameters())
    opt = optim.Adam(modele.parameters(), lr=1e-3)
    meilleur, meilleure_ep, meilleur_rot = float('inf'), -1, float('nan')
    historique = []
    chemin = os.path.join(_ROOT, 'checkpoints', f'multihead_K{n_heads}.pth')

    print(f"\n  K = {n_heads} | {n_params:,} parametres | {epochs} epoques\n")
    t0 = time.time()

    for ep in range(epochs):
        modele.train()
        for bx, br in loader:
            opt.zero_grad()
            perte, _, _ = perte_multi_hypotheses(modele(bx), bx, br, fk, R_OUTIL)
            perte.backward()
            opt.step()

        for seuil, lr in ((30, 5e-4), (60, 1e-4), (80, 2e-5)):
            if ep == seuil:
                for g in opt.param_groups:
                    g['lr'] = lr

        modele.eval()
        with torch.no_grad():
            q = modele(Xva)
            B, Kh, _ = q.shape
            T = fk.forward(q.reshape(B * Kh, 6))
            pos = T[:, :3, 3].reshape(B, Kh, 3)
            rot = T[:, :3, :3].reshape(B, Kh, 3, 3)

            # Selection sur la POSE COMPLETE, comme a l'entrainement.
            choisie = erreur_pose(pos, rot, Xva, R_ref_va).argmin(1)
            idx = torch.arange(B)
            pos_mm = torch.norm(pos[idx, choisie] - Xva, dim=-1).mean().item() * 1000
            Rres = torch.bmm(R_ref_va.transpose(1, 2), rot[idx, choisie])
            cos = ((Rres[:, 0, 0] + Rres[:, 1, 1] + Rres[:, 2, 2]) - 1) / 2
            rot_deg = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()
            tetes_utiles = len(torch.unique(choisie))

        historique.append({'epoque': ep + 1, 'position_mm': round(pos_mm, 4),
                           'orientation_deg': round(rot_deg, 4),
                           'tetes_gagnantes': tetes_utiles})

        if pos_mm < meilleur:
            meilleur, meilleur_rot, meilleure_ep = pos_mm, rot_deg, ep + 1
            modele.save_model(chemin)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    ep {ep+1:03d} | best-of-K {pos_mm:8.3f} mm | "
                  f"{rot_deg:6.3f} deg | tetes utilisees {tetes_utiles}/{n_heads}",
                  flush=True)

    return {'modele': chemin, 'n_params': n_params,
            'position_mm': meilleur, 'orientation_deg': meilleur_rot,
            'meilleure_epoque': meilleure_ep, 'historique': historique,
            'secondes': round(time.time() - t0, 1)}


# =====================================================================
# 3. Qu'ont appris les tetes ?
# =====================================================================
def analyser(modele, X, Q, M, n_heads, n=2000):
    """
    Attribution des tetes aux branches, en distance ANGULAIRE PERIODIQUE, puis
    rappel par branche.

    Le rappel est la mesure qui tranche : couvrir 2 branches sur 2 disponibles
    n'a rien a voir avec 2 sur 4.
    """
    Xs = torch.tensor(X[-n:]); Qs = Q[-n:]; Ms = M[-n:]
    with torch.no_grad():
        q = modele(Xs).numpy()

    attribution = np.full((len(Xs), n_heads), -1)
    retrouvee = np.zeros((len(Xs), K_MAX), dtype=bool)
    SEUIL = 0.25          # rad ; une tete "retrouve" une branche sous ce seuil

    for i in range(len(Xs)):
        valides = np.where(Ms[i])[0]
        if len(valides) == 0:
            continue
        for k in range(n_heads):
            d = ecart_angulaire(Qs[i][valides], q[i, k])
            j = int(d.argmin())
            attribution[i, k] = valides[j]
            if d[j] < SEUIL:
                retrouvee[i, valides[j]] = True

    print("\n  Specialisation des tetes (distance angulaire periodique)")
    print("  " + "-" * 60)
    print(f"  {'tete':>5}  {'branche dominante':<24} {'constance':>10}")
    specialisation = {}
    for k in range(n_heads):
        col = attribution[:, k]; col = col[col >= 0]
        if len(col) == 0:
            continue
        vals, cnt = np.unique(col, return_counts=True)
        dom = int(vals[cnt.argmax()])
        const = float(cnt.max() / len(col))
        specialisation[k] = {'branche': "-".join(BRANCHES[dom]),
                             'constance': round(const, 3)}
        print(f"  {k:>5}  {'-'.join(BRANCHES[dom]):<24} {const:>9.0%}")

    print("\n  Rappel par branche")
    print("  " + "-" * 60)
    print(f"  {'branche':<24} {'disponible':>11} {'retrouvee':>11}")
    histo = {}
    for i, b in enumerate(BRANCHES):
        dispo = float(Ms[:, i].mean())
        if dispo == 0.0:
            continue
        rap = float(retrouvee[Ms[:, i], i].mean()) if Ms[:, i].any() else 0.0
        histo["-".join(b)] = {'disponible': round(dispo, 4),
                              'rappel': round(rap, 4)}
        print(f"  {'-'.join(b):<24} {dispo:>10.1%} {rap:>10.1%}")

    total_dispo = int(Ms.sum())
    total_retrouve = int((retrouvee & Ms).sum())
    rappel_global = total_retrouve / total_dispo if total_dispo else 0.0
    print(f"\n  Rappel global : {rappel_global:.1%} "
          f"({total_retrouve} solutions retrouvees sur {total_dispo} disponibles)")

    return {'specialisation': specialisation, 'branches': histo,
            'rappel_global': round(rappel_global, 4),
            'branches_existantes': len(histo)}


def latence(modele, fk, n=1000):
    """Cout reel d'un appel : K sorties + FK + selection sur la pose complete."""
    torch.set_num_threads(1)
    x = torch.tensor([[0.198, -0.728, 0.030]])
    R_ref = fk.forward(modele(x)[:, 0, :])[:, :3, :3]
    with torch.no_grad():
        for _ in range(200):
            q = modele(x); T = fk.forward(q[0])
            erreur_pose(T[:, :3, 3].unsqueeze(0), T[:, :3, :3].unsqueeze(0),
                        x, R_ref).argmin()
        t0 = time.perf_counter()
        for _ in range(n):
            q = modele(x); T = fk.forward(q[0])
            erreur_pose(T[:, :3, 3].unsqueeze(0), T[:, :3, :3].unsqueeze(0),
                        x, R_ref).argmin()
        return (time.perf_counter() - t0) / n * 1000


# =====================================================================
def main():
    print("=" * 74)
    print(f"IK MULTI-HYPOTHESES -- K = {N_HEADS}")
    print("=" * 74)

    fk = UR5ForwardKinematicsPyTorch()

    print("\n[1/3] Jeu de donnees")
    X, Q, M, empreinte = jeu_de_donnees()
    print(f"  {len(X)} cibles | empreinte {empreinte} | "
          f"{M.sum() / len(M):.2f} branches valides en moyenne")

    print("\n[2/3] Entrainement (perte physique seule, aucune etiquette de branche)")
    res = entrainer(X, Q, M, fk, N_HEADS)

    print("\n[3/3] Analyse")
    modele = MultiHeadIK.from_file(res['modele']); modele.eval()
    res.update(analyser(modele, X, Q, M, N_HEADS))
    res['latence_ms'] = round(latence(modele, fk), 3)
    res.update({'n_heads': N_HEADS, 'epoques': EPOCHS, 'n_cibles': len(X),
                'graine': 42, 'empreinte_donnees': empreinte,
                'r_outil': R_OUTIL})

    print(f"\n  Latence (K sorties + FK + selection) : {res['latence_ms']:.3f} ms")
    print(f"  Meilleur : {res['position_mm']:.3f} mm / "
          f"{res['orientation_deg']:.3f} deg (epoque {res['meilleure_epoque']})")

    sortie = os.path.join(_ROOT, 'checkpoints', f'multihypothesis_K{N_HEADS}.json')
    json.dump(res, open(sortie, 'w'), indent=2)
    print(f"\nResultats : {sortie}")
    print("=" * 74)


if __name__ == "__main__":
    main()
