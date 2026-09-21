"""
Cinematique inverse par moindres carres amortis (Damped Least Squares).

Le principe
-----------
On ne sait pas inverser la cinematique directe en forme fermee pour un robot
quelconque, mais on sait la LINEARISER : au voisinage de q,

    dx = J(q) dq

Il suffirait donc d'inverser J. Sauf qu'au voisinage d'une singularite J perd
son rang : la pseudo-inverse explose et le solveur demande des vitesses
articulaires infinies pour un deplacement fini.

L'amortissement de Levenberg-Marquardt, introduit en robotique par Nakamura et
Wampler puis Chiaverini, remplace l'inversion brutale par

    dq = J^T (J J^T + lambda^2 I)^-1 e

Le terme lambda^2 I garantit que la matrice reste inversible. Le prix est un
biais : loin des singularites la solution amortie s'ecarte legerement de la
pseudo-inverse. C'est le compromis classique entre exactitude et robustesse, et
il est explicite ici plutot que subi.

Amortissement adaptatif
-----------------------
Un lambda constant amortit meme la ou ce n'est pas necessaire. On le module
donc selon la plus petite valeur singuliere de J : nul quand le bras est bien
conditionne, croissant a l'approche de la singularite.

Ce que ce fichier apporte au projet
-----------------------------------
Il complete la chaine geometrie -> Jacobien -> IK numerique, et fournit au banc
d'essai une troisieme reference, entre la forme fermee (exacte mais propre a ce
robot) et le reseau (rapide mais silencieux hors domaine).
"""
import numpy as np

from jacobian import geometric_jacobian, forward_kinematics


def log_so3(R):
    """
    Vecteur de rotation (axe x angle) d'une matrice de rotation.

    C'est le logarithme sur SO(3). Les deux cas limites sont traites
    explicitement : l'angle nul, ou le vecteur est nul, et l'angle pi, ou la
    formule generale divise par sin(theta) = 0.
    """
    cos = (np.trace(R) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)

    if theta < 1e-9:
        return np.zeros(3)

    if theta > np.pi - 1e-6:
        # Pres de pi : on lit l'axe sur la diagonale de (R + I)
        A = (R + np.eye(3)) / 2.0
        axe = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axe))
        if axe[i] > 1e-9:
            axe = A[:, i] / axe[i]
        return theta * axe / (np.linalg.norm(axe) + 1e-12)

    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    return theta * w / (2.0 * np.sin(theta))


def erreur_pose(T_courant, T_cible):
    """
    Erreur de pose en 6 composantes : translation puis rotation.

    La partie rotation est le logarithme de la rotation residuelle, donc un
    vecteur dont la norme est l'angle a rattraper -- une grandeur directement
    comparable a une distance.
    """
    e = np.zeros(6)
    e[:3] = T_cible[:3, 3] - T_courant[:3, 3]
    e[3:] = log_so3(T_cible[:3, :3] @ T_courant[:3, :3].T)
    return e


def solve_ik_dls(T_cible, q0, lam=0.05, tol_pos=1e-6, tol_rot=1e-6,
                 max_iter=200, pas_max=0.3, adaptatif=True):
    """
    Resout l'IK par moindres carres amortis.

    Parametres
    ----------
    T_cible   : pose demandee, matrice 4x4
    q0        : configuration de depart -- le resultat en depend, c'est la
                nature d'une methode locale
    lam       : amortissement de base (rad)
    pas_max   : borne sur la norme d'un increment, pour eviter qu'un grand
                residu initial ne projette le bras n'importe ou
    adaptatif : module l'amortissement selon le conditionnement de J

    Renvoie
    -------
    (q, succes, infos) ou infos contient le nombre d'iterations et les erreurs
    finales en position et en orientation.
    """
    q = np.array(q0, dtype=float)

    for it in range(max_iter):
        T = forward_kinematics(q)
        e = erreur_pose(T, T_cible)
        err_pos = np.linalg.norm(e[:3])
        err_rot = np.linalg.norm(e[3:])

        if err_pos < tol_pos and err_rot < tol_rot:
            return q, True, {'iterations': it, 'erreur_pos_m': err_pos,
                             'erreur_rot_rad': err_rot}

        J = geometric_jacobian(q)

        if adaptatif:
            # Amortissement nul loin des singularites, croissant a l'approche.
            sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
            seuil = 0.04
            l = 0.0 if sigma_min >= seuil else lam * (1.0 - (sigma_min / seuil) ** 2)
        else:
            l = lam

        JJt = J @ J.T + (l ** 2) * np.eye(6)
        try:
            dq = J.T @ np.linalg.solve(JJt, e)
        except np.linalg.LinAlgError:
            break

        n = np.linalg.norm(dq)
        if n > pas_max:
            dq *= pas_max / n
        q = q + dq

    T = forward_kinematics(q)
    e = erreur_pose(T, T_cible)
    return q, False, {'iterations': max_iter,
                      'erreur_pos_m': float(np.linalg.norm(e[:3])),
                      'erreur_rot_rad': float(np.linalg.norm(e[3:]))}


def solve_ik_dls_restarts(T_cible, q0, n_restarts=8, amplitude=0.8, seed=0, **kw):
    """
    DLS avec relances aleatoires.

    Une methode locale depend de son point de depart : depuis une pose de repos
    unique, ce solveur converge en une vingtaine d'iterations ou reste bloque a
    plusieurs centimetres dans un minimum local -- il n'y a pas d'entre-deux.
    Mesure sur la zone de travail : environ deux tiers d'echecs.

    La reponse standard n'est pas de mieux regler l'amortissement mais de
    relancer depuis des configurations perturbees. On garde la premiere qui
    converge.

    Le nombre de relances consommees est renvoye : il mesure directement a quel
    point le probleme est sensible a l'initialisation.
    """
    rng = np.random.default_rng(seed)
    q0 = np.asarray(q0, dtype=float)
    meilleur, meilleure_info = None, None

    for essai in range(n_restarts):
        depart = q0 if essai == 0 else q0 + rng.uniform(-amplitude, amplitude, 6)
        q, ok, info = solve_ik_dls(T_cible, depart, **kw)
        info['relances'] = essai + 1
        if ok:
            return q, True, info
        if meilleure_info is None or info['erreur_pos_m'] < meilleure_info['erreur_pos_m']:
            meilleur, meilleure_info = q, info

    return meilleur, False, meilleure_info


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    q_repos = np.array([0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, -np.pi / 2, 0.0])

    # Cibles atteignables par construction : on tire une configuration, on en
    # prend la pose. Aucune cible impossible ne vient donc polluer le taux de
    # reussite.
    reussites, iters, erreurs, temps = 0, [], [], []
    N = 300
    for _ in range(N):
        q_vrai = q_repos + rng.uniform(-0.6, 0.6, 6)
        T_cible = forward_kinematics(q_vrai)
        depart = q_repos + rng.uniform(-0.3, 0.3, 6)

        t0 = time.perf_counter()
        q, ok, info = solve_ik_dls(T_cible, depart)
        temps.append((time.perf_counter() - t0) * 1000)

        if ok:
            reussites += 1
            iters.append(info['iterations'])
            erreurs.append(info['erreur_pos_m'] * 1000)

    print(f"IK par moindres carres amortis, {N} cibles atteignables")
    print(f"  taux de convergence : {reussites / N:.1%}")
    print(f"  iterations (mediane): {np.median(iters):.0f}")
    print(f"  erreur de position  : {np.median(erreurs):.2e} mm")
    print(f"  temps (mediane)     : {np.median(temps):.2f} ms")
