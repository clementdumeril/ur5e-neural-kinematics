"""
Banc d'essai des solveurs de cinematique inverse.

Ce fichier existe parce que le README citait des chiffres -- 45,0 ms pour IKPY,
0,223 ms pour la forme fermee -- sans script pour les reproduire. Une mesure
qu'on ne peut pas rejouer n'est pas un resultat.

Les cinq solveurs compares
--------------------------
1. forme fermee analytique  : exacte, mais derivee a la main pour CE robot
2. moindres carres amortis  : Jacobien + Levenberg-Marquardt, generique
3. IKPY                     : solveur numerique de reference, generique
4. reseau a sortie unique   : une branche imposee a l'entrainement
5. reseau multi-tetes K=2   : branches decouvertes, sans etiquette

Protocole
---------
Meme jeu de cibles pour tous. 200 appels d'echauffement jetes -- sans quoi on
compare un PyTorch froid a un NumPy chaud, erreur commise dans une version
precedente de ce projet et qui avait fausse le tableau dans les deux sens.
torch.set_num_threads(1) pour que la mesure ne depende pas du nombre de coeurs.

Les methodes locales (DLS, IKPY) partent toutes de la meme configuration de
repos : leur resultat en depend, c'est la nature d'une methode iterative, et le
dire fait partie de la mesure.
"""
import os
import sys
import json
import math
import time

import numpy as np

if 'controller' not in sys.modules:
    sys.modules['controller'] = type(
        'controller', (), {'Supervisor': type('Supervisor', (), {})})

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ('src/kinematics', 'src/models', 'src/control', 'src/training'):
    _p = os.path.join(_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                # noqa: E402
from ur5 import build_matrix, inverse_kinematics            # noqa: E402
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch      # noqa: E402
from jacobian import forward_kinematics as fk_np            # noqa: E402
from ik_dls import solve_ik_dls, solve_ik_dls_restarts      # noqa: E402
from pinn import PINN6DOF                                   # noqa: E402
from multihead import MultiHeadIK                           # noqa: E402

torch.set_num_threads(1)

ROT_DOWN = [math.pi, 0.0, -math.pi / 2]
Q_REPOS = np.array([0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, -np.pi / 2, 0.0])
N_CIBLES = int(os.environ.get('BM_CIBLES', 300))
N_ECHAUF = 200


def cibles(n, seed=7):
    """Cibles tirees dans la zone d'entrainement, toutes atteignables."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        t = np.array([rng.uniform(0.0, 0.4),
                      rng.uniform(-0.9, -0.5),
                      rng.uniform(0.05, 0.45)])
        T = build_matrix(t, ROT_DOWN, euler='XYZ')
        try:
            q = inverse_kinematics(T, shoulder='left', wrist='up', elbow='up')
        except Exception:
            continue
        if q is None or np.any(np.isnan(q)):
            continue
        if np.linalg.norm(fk_np(np.asarray(q, float))[:3, 3] - t) > 1e-4:
            continue
        out.append((t, T))
    return out


def mesurer(nom, resoudre, jeu, differentiable):
    """
    Chronometre un solveur et mesure ou sa solution atterrit REELLEMENT.

    L'erreur est toujours evaluee par cinematique directe sur les angles rendus,
    jamais rapportee par le solveur lui-meme : un solveur qui se trompe sur sa
    propre erreur existe, on en a rencontre un dans ce projet.
    """
    for t, T in jeu[:min(N_ECHAUF, len(jeu))]:
        try:
            resoudre(t, T)
        except Exception:
            pass

    temps, erreurs, echecs = [], [], 0
    for t, T in jeu:
        t0 = time.perf_counter()
        try:
            q = resoudre(t, T)
        except Exception:
            q = None
        temps.append((time.perf_counter() - t0) * 1000.0)
        if q is None or np.any(np.isnan(q)):
            echecs += 1
            continue
        erreurs.append(np.linalg.norm(fk_np(np.asarray(q, float))[:3, 3] - t) * 1000.0)

    temps = np.array(temps)
    erreurs = np.array(erreurs) if erreurs else np.array([np.nan])
    return {
        'solveur': nom,
        'temps_median_ms': float(np.median(temps)),
        'temps_moyen_ms': float(np.mean(temps)),
        'temps_p95_ms': float(np.percentile(temps, 95)),
        'erreur_mediane_mm': float(np.median(erreurs)),
        'erreur_p95_mm': float(np.percentile(erreurs, 95)),
        'taux_echec': echecs / len(jeu),
        'differentiable': differentiable,
    }


def main():
    print("=" * 92)
    print(f"BANC D'ESSAI DES SOLVEURS -- {N_CIBLES} cibles, "
          f"{N_ECHAUF} appels d'echauffement jetes, 1 thread")
    print("=" * 92)

    jeu = cibles(N_CIBLES)
    fk_t = UR5ForwardKinematicsPyTorch()
    resultats = []

    # --- 1. forme fermee ------------------------------------------------
    resultats.append(mesurer(
        "forme fermee analytique",
        lambda t, T: inverse_kinematics(T, shoulder='left', wrist='up', elbow='up'),
        jeu, False))

    # --- 2. moindres carres amortis -------------------------------------
    def dls_simple(t, T):
        q, ok, _ = solve_ik_dls(T, Q_REPOS)
        return q if ok else None
    resultats.append(mesurer("DLS, depart unique", dls_simple, jeu, False))

    def dls_relances(t, T):
        q, ok, _ = solve_ik_dls_restarts(T, Q_REPOS, n_restarts=8)
        return q if ok else None
    resultats.append(mesurer("DLS, 8 relances", dls_relances, jeu, False))

    # --- 3. IKPY, DANS SON PROPRE REPERE --------------------------------
    #
    # L'URDF et la table DH ne decrivent PAS la meme chaine : pour les memes
    # angles articulaires, les deux cinematiques directes placent l'effecteur a
    # 1,3 a 1,8 METRE l'une de l'autre. Evaluer IKPY sur des cibles exprimees
    # en coordonnees DH mesurerait ce desaccord de convention, pas le solveur.
    #
    # On lui donne donc des cibles tirees de SA propre cinematique directe, et
    # on mesure son erreur avec SA propre FK. Le temps devient comparable -- les
    # deux resolvent une IK de UR5e de difficulte equivalente -- mais les
    # erreurs, elles, ne sont pas dans le meme repere et ne se comparent pas.
    try:
        from ikpy_ur5e_solver import IKPYUR5eSolver
        ik = IKPYUR5eSolver(os.path.join(_ROOT, 'src', 'kinematics', 'ur5e.urdf'))

        rng = np.random.default_rng(11)
        jeu_ikpy = []
        for _ in range(len(jeu)):
            q = rng.uniform(-1.0, 1.0, 6)
            p = ik.chain.forward_kinematics(np.concatenate(([0.0], q)))[:3, 3]
            jeu_ikpy.append((p, None))

        temps, erreurs = [], []
        for p, _ in jeu_ikpy[:N_ECHAUF]:
            ik.solve_ik(p)
        for p, _ in jeu_ikpy:
            t0 = time.perf_counter()
            q = ik.solve_ik(p)[0]
            temps.append((time.perf_counter() - t0) * 1000.0)
            atteint = ik.chain.forward_kinematics(np.concatenate(([0.0], q)))[:3, 3]
            erreurs.append(np.linalg.norm(atteint - p) * 1000.0)
        temps, erreurs = np.array(temps), np.array(erreurs)
        resultats.append({
            'solveur': "IKPY (repere URDF) *",
            'temps_median_ms': float(np.median(temps)),
            'temps_moyen_ms': float(np.mean(temps)),
            'temps_p95_ms': float(np.percentile(temps, 95)),
            'erreur_mediane_mm': float(np.median(erreurs)),
            'erreur_p95_mm': float(np.percentile(erreurs, 95)),
            'taux_echec': 0.0, 'differentiable': False,
            'note': 'cibles et erreurs dans le repere URDF, non comparables aux autres',
        })
    except Exception as e:
        print(f"  [IKPY indisponible : {e}]")

    # --- 4. reseau a sortie unique --------------------------------------
    m1 = PINN6DOF.from_file(os.path.join(
        _ROOT, 'checkpoints', 'pinn_model_true_physics.pth'))
    m1.eval()

    def reseau(t, T):
        with torch.no_grad():
            return m1(torch.tensor(t, dtype=torch.float32).unsqueeze(0))[0].numpy()
    resultats.append(mesurer("reseau, sortie unique", reseau, jeu, True))

    # --- 5. reseau multi-tetes ------------------------------------------
    chemin = os.path.join(_ROOT, 'checkpoints', 'multihead_K2.pth')
    if os.path.exists(chemin):
        m2 = MultiHeadIK.from_file(chemin); m2.eval()

        def multi(t, T):
            x = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = m2(x)[0]
                p = fk_t.forward_pos(q)
                return q[int(torch.norm(p - x, dim=1).argmin())].numpy()
        resultats.append(mesurer("reseau, 2 tetes + selection", multi, jeu, True))

    # --- tableau ---------------------------------------------------------
    print(f"\n{'Solveur':<30} {'median':>9} {'moyen':>9} {'p95':>9} "
          f"{'erreur':>11} {'echecs':>8} {'deriv.':>7}")
    print("-" * 92)
    for r in resultats:
        print(f"{r['solveur']:<30} {r['temps_median_ms']:>7.3f}ms "
              f"{r['temps_moyen_ms']:>7.3f}ms {r['temps_p95_ms']:>7.3f}ms "
              f"{r['erreur_mediane_mm']:>9.4f}mm {r['taux_echec']:>7.1%} "
              f"{'oui' if r['differentiable'] else 'non':>7}")
    print("=" * 92)
    print("* IKPY est mesure dans le repere de l'URDF, qui ne coincide pas avec")
    print("  la table DH du projet : pour les memes angles, les deux cinematiques")
    print("  directes different de 1,3 a 1,8 m. Seuls les TEMPS sont comparables ;")
    print("  les erreurs ne sont pas exprimees dans le meme repere.")

    sortie = os.path.join(_ROOT, 'checkpoints', 'benchmark_solveurs.json')
    json.dump({'n_cibles': N_CIBLES, 'echauffement': N_ECHAUF,
               'threads': 1, 'resultats': resultats}, open(sortie, 'w'), indent=2)
    print(f"\nResultats : {sortie}")


if __name__ == "__main__":
    main()
