"""
Mesure de la CONTINUITE des solveurs le long d'une trajectoire cartesienne.

Le README affirme que le solveur neuronal est "continuous in the target", la ou
les methodes analytiques "jump between their 8 solution branches". Cette
affirmation n'etait pas mesuree -- exactement comme l'etait celle sur la moyenne
des 8 solutions avant l'ablation.

Protocole
---------
On parcourt un cercle dans le plan horizontal, entierement contenu dans la zone
d'entrainement, et on echantillonne 400 poses consecutives. A chaque pas on
resout l'IK par trois voies, puis on mesure le SAUT ARTICULAIRE entre deux poses
voisines :

    saut = || q(t+1) - q(t) ||

Sur une trajectoire lisse, un solveur continu produit de petits sauts reguliers.
Un saut de branche produit une discontinuite franche, de l'ordre du radian.

Les trois voies
---------------
1. analytique, branche figee   : ce que le depot utilise reellement
   (wrist='up', shoulder='left', elbow='up')
2. analytique, premiere branche valide : l'usage generique, celui auquel
   l'affirmation du README fait implicitement reference
3. reseau                      : le PINN entraine

La voie 2 est la comparaison honnete. Se comparer uniquement a la voie 1 serait
se donner le beau role : une branche figee ne peut pas, par construction,
changer de branche.
"""
import os
import sys
import math

import numpy as np
import torch

if 'controller' not in sys.modules:
    sys.modules['controller'] = type(
        'controller', (), {'Supervisor': type('Supervisor', (), {})})

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['reference_ur5_repo', 'robotics_utils', 'training']:
    p = os.path.join(BASE, _d)
    if p not in sys.path:
        sys.path.insert(0, p)

from ur5 import build_matrix, inverse_kinematics          # noqa: E402
from train_pinn_6dof import PINN6DOF                      # noqa: E402
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch    # noqa: E402

ROT_DOWN = [math.pi, 0.0, -math.pi / 2]
BRANCHES = [(s, w, e)
            for s in ('left', 'right')
            for w in ('up', 'down')
            for e in ('up', 'down')]

N = 400
CENTRE = np.array([0.20, -0.70, 0.25])
RAYON = 0.12


def trajectoire():
    """Cercle horizontal, entierement dans la zone d'entrainement."""
    t = np.linspace(0.0, 2.0 * math.pi, N)
    return np.column_stack((
        CENTRE[0] + RAYON * np.cos(t),
        CENTRE[1] + RAYON * np.sin(t),
        np.full_like(t, CENTRE[2]),
    ))


def resoudre(T, branche, fk):
    """Une branche donnee, validee par cinematique directe."""
    try:
        q = inverse_kinematics(T, shoulder=branche[0],
                               wrist=branche[1], elbow=branche[2])
    except Exception:
        return None
    if q is None or np.any(np.isnan(q)):
        return None
    q = np.asarray(q, dtype=np.float32)
    with torch.no_grad():
        p = fk.forward_pos(torch.tensor(q).unsqueeze(0))[0].numpy()
    return q if np.linalg.norm(p - T[:3, 3]) < 1e-4 else None


def stats(sauts, nom):
    if not sauts:
        print(f"  {nom:<34} aucune paire exploitable")
        return
    s = np.array(sauts)
    print(f"  {nom:<34} median {np.median(s):7.4f}   "
          f"p95 {np.percentile(s, 95):7.4f}   max {s.max():8.4f} rad")


def main():
    fk = UR5ForwardKinematicsPyTorch()
    modele = PINN6DOF.from_file(
        os.path.join(BASE, 'models', 'pinn_model_true_physics.pth'))
    modele.eval()

    cibles = trajectoire()
    Ts = [build_matrix(c, ROT_DOWN, euler='XYZ') for c in cibles]

    # --- voie 1 : branche figee -----------------------------------------
    q_fige = [resoudre(T, ('left', 'up', 'up'), fk) for T in Ts]

    # --- voie 2 : premiere branche valide, ordre fixe --------------------
    q_premiere, branche_choisie = [], []
    for T in Ts:
        trouve = None
        for b in BRANCHES:
            q = resoudre(T, b, fk)
            if q is not None:
                trouve, choisie = q, b
                break
        q_premiere.append(trouve)
        branche_choisie.append(choisie if trouve is not None else None)

    # --- voie 3 : le reseau ---------------------------------------------
    with torch.no_grad():
        q_reseau = modele(torch.tensor(cibles, dtype=torch.float32)).numpy()
        p_reseau = fk.forward_pos(torch.tensor(q_reseau)).numpy()
    err_reseau = np.linalg.norm(p_reseau - cibles, axis=1) * 1000.0

    # --- sauts articulaires ---------------------------------------------
    def sauts(serie):
        out = []
        for a, b in zip(serie[:-1], serie[1:]):
            if a is None or b is None:
                continue
            out.append(float(np.linalg.norm(np.asarray(b) - np.asarray(a))))
        return out

    print("=" * 78)
    print(f"CONTINUITE le long d'un cercle de rayon {RAYON} m, {N} echantillons")
    print("=" * 78)
    print(f"\nEchecs de resolution : "
          f"branche figee {sum(q is None for q in q_fige)}/{N}, "
          f"premiere valide {sum(q is None for q in q_premiere)}/{N}\n")

    print("Saut articulaire entre deux poses consecutives :")
    stats(sauts(q_fige), "analytique, branche figee")
    stats(sauts(q_premiere), "analytique, premiere valide")
    stats(sauts(list(q_reseau)), "reseau (PINN)")

    changements = sum(1 for a, b in zip(branche_choisie[:-1], branche_choisie[1:])
                      if a is not None and b is not None and a != b)
    print(f"\nChangements de branche (voie 2) : {changements} sur {N - 1} pas")

    print(f"\nSuivi de la cible par le reseau : "
          f"median {np.median(err_reseau):.3f} mm, max {err_reseau.max():.3f} mm")

    frontiere(fk, modele)
    print("=" * 78)


def frontiere(fk, modele):
    """
    Que font les deux solveurs au BORD de l'espace atteignable ?

    C'est la vraie difference entre eux, et elle ne joue pas en faveur du
    reseau : l'analytique echoue franchement (exception ou NaN), le reseau rend
    toujours six angles d'apparence plausible, sans aucun signal d'echec.
    """
    print("\n" + "-" * 78)
    print("COMPORTEMENT AU BORD DE L'ESPACE ATTEIGNABLE (y = -0,70 ; z = 0,25)")
    print("-" * 78)
    print(f"{'x':>8}   {'analytique':<22} {'erreur de la pose rendue':>26}")

    for x in (0.40, 0.50, 0.55, 0.60, 0.70):
        t = np.array([x, -0.70, 0.25], dtype=np.float32)
        T = build_matrix(t, ROT_DOWN, euler='XYZ')
        try:
            q = inverse_kinematics(T, shoulder='left', wrist='up', elbow='up')
            verdict = ("NaN (echec franc)"
                       if q is None or np.any(np.isnan(q)) else "solution")
        except Exception:
            verdict = "exception (echec franc)"

        with torch.no_grad():
            p = fk.forward_pos(modele(torch.tensor(t).unsqueeze(0)))[0].numpy()
        print(f"{x:+8.2f}   {verdict:<22} {np.linalg.norm(p - t) * 1000:>21.1f} mm")

    print("\n-> Le reseau ne signale jamais l'echec. C'est un DEFAUT, pas un")
    print("   avantage : un solveur qui se tait quand il sort de son domaine")
    print("   est plus dangereux qu'un solveur qui leve une exception.")


if __name__ == "__main__":
    main()
