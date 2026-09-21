"""
Jacobien geometrique du UR5e, ecrit depuis les principes.

Le Jacobien relie les vitesses articulaires a la vitesse de l'effecteur :

    [ v ]           [ v ]   vitesse lineaire  (m/s)
    [   ] = J(q) q̇  [ w ]   vitesse angulaire (rad/s)
    [ w ]

Pour une articulation ROTOIDE i, la contribution se lit directement sur la
geometrie, sans aucune derivation symbolique :

    J_v,i = z_{i-1} x (p_e - p_{i-1})      partie lineaire
    J_w,i = z_{i-1}                        partie angulaire

ou z_{i-1} est l'axe de rotation de l'articulation i exprime dans le repere de
base, p_{i-1} l'origine de son repere, et p_e la position de l'effecteur.

L'intuition : faire tourner l'articulation i fait tourner tout ce qui est en
aval autour de l'axe z_{i-1}. La main, situee au bras de levier (p_e - p_{i-1}),
part donc en vitesse selon le produit vectoriel.

Convention
----------
Denavit-Hartenberg standard, exactement la table de ur5.py et de
ur5_pytorch_fk.py. Si ces trois fichiers divergeaient, le Jacobien decrirait un
robot qui n'est pas celui de la simulation -- d'ou la verification par
differences finies dans verifier_jacobien().
"""
import numpy as np

# [a, alpha, d, offset_theta] -- identique a ur5.py et ur5_pytorch_fk.py
D1, A2, A3 = 0.1625, 0.425, 0.3922
D4, D5, D6 = 0.1333, 0.0997, 0.0996 + 0.1237

DH = np.array([
    [0.0,  np.pi / 2, D1,  0.0],
    [A2,   0.0,       0.0, np.pi / 2],
    [A3,   0.0,       0.0, 0.0],
    [0.0, -np.pi / 2, D4, -np.pi / 2],
    [0.0,  np.pi / 2, D5,  0.0],
    [0.0,  0.0,       D6,  0.0],
])


def transform_dh(theta, a, alpha, d, offset):
    """Matrice homogene d'un maillon, DH standard."""
    q = theta + offset
    ct, st = np.cos(q), np.sin(q)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ])


def frames(q):
    """
    Reperes cumules T_0^i pour i = 0 .. 6.

    Renvoie une liste de 7 matrices 4x4, la premiere etant l'identite (le
    repere de base). C'est tout ce dont le Jacobien a besoin.
    """
    T = np.eye(4)
    out = [T.copy()]
    for i in range(6):
        a, alpha, d, offset = DH[i]
        T = T @ transform_dh(q[i], a, alpha, d, offset)
        out.append(T.copy())
    return out


def forward_kinematics(q):
    """Pose de l'effecteur, matrice 4x4."""
    return frames(q)[-1]


def geometric_jacobian(q):
    """
    Jacobien geometrique 6x6, exprime dans le repere de BASE.

    Lignes 0-2 : vitesse lineaire. Lignes 3-5 : vitesse angulaire.
    """
    T = frames(q)
    p_e = T[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = T[i][:3, 2]            # axe de l'articulation i+1
        p = T[i][:3, 3]            # origine de son repere
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def body_jacobian(q):
    """
    Jacobien exprime dans le repere de l'OUTIL.

    J_b = diag(R^T, R^T) . J_s, ou R est l'orientation de l'effecteur. C'est
    celui qu'on utilise quand l'erreur de pose est mesuree dans le repere outil,
    par exemple avec le logarithme de SE(3).
    """
    T = forward_kinematics(q)
    Rt = T[:3, :3].T
    J = geometric_jacobian(q)
    return np.vstack((Rt @ J[:3, :], Rt @ J[3:, :]))


# =====================================================================
# Mesures de proximite aux singularites
# =====================================================================
def manipulability(q):
    """
    Mesure de Yoshikawa : w(q) = sqrt(det(J J^T)).

    Elle vaut le volume de l'ellipsoide des vitesses atteignables. Elle tend
    vers zero a l'approche d'une singularite : le bras perd la capacite de se
    deplacer dans au moins une direction.
    """
    J = geometric_jacobian(q)
    d = np.linalg.det(J @ J.T)
    return float(np.sqrt(max(d, 0.0)))


def condition_number(q):
    """
    Conditionnement de J, rapport des valeurs singuliere extremes.

    Vaut 1 pour un robot parfaitement isotrope, et diverge a la singularite.
    Complementaire de la manipulabilite : celle-ci peut rester grande alors que
    le bras est deja tres mal conditionne dans une direction.
    """
    s = np.linalg.svd(geometric_jacobian(q), compute_uv=False)
    return float(np.inf if s[-1] < 1e-15 else s[0] / s[-1])


# =====================================================================
# Verification : le Jacobien analytique contre les differences finies
# =====================================================================
def jacobien_numerique(q, h=1e-6):
    """
    Jacobien obtenu en derivant numeriquement la cinematique directe.

    Partie lineaire : derivee centree de la position.
    Partie angulaire : la rotation residuelle R(q+h) R(q)^T, petite, s'ecrit
    I + [w]x au premier ordre ; on en extrait w par sa partie antisymetrique.
    """
    J = np.zeros((6, 6))
    for i in range(6):
        qp, qm = np.array(q, float), np.array(q, float)
        qp[i] += h
        qm[i] -= h
        Tp, Tm = forward_kinematics(qp), forward_kinematics(qm)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        dR = (Tp[:3, :3] - Tm[:3, :3]) / (2 * h) @ forward_kinematics(q)[:3, :3].T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2],
                             dR[0, 2] - dR[2, 0],
                             dR[1, 0] - dR[0, 1]]) / 2.0
    return J


def verifier_jacobien(n=200, seed=0):
    """
    Le Jacobien analytique doit coincider avec la derivee numerique de la FK.

    C'est la seule verification qui vaille : elle ne compare pas le code a
    lui-meme, elle compare une formule geometrique a une derivee mesuree.
    """
    rng = np.random.default_rng(seed)
    pires = []
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 6)
        ecart = np.abs(geometric_jacobian(q) - jacobien_numerique(q)).max()
        pires.append(ecart)
    return float(np.max(pires)), float(np.mean(pires))


if __name__ == "__main__":
    pire, moyen = verifier_jacobien()
    print(f"Jacobien analytique contre differences finies, 200 configurations :")
    print(f"  ecart maximal : {pire:.3e}")
    print(f"  ecart moyen   : {moyen:.3e}")
    print("  -> " + ("coherent" if pire < 1e-5 else "INCOHERENT"))

    q = np.array([0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, -np.pi / 2, 0.0])
    print(f"\nPose de repos :")
    print(f"  manipulabilite      : {manipulability(q):.5f}")
    print(f"  conditionnement     : {condition_number(q):.1f}")

    q_sing = np.zeros(6)          # bras tendu : singularite d'elongation
    print(f"\nBras tendu (q = 0) :")
    print(f"  manipulabilite      : {manipulability(q_sing):.5f}")
    print(f"  conditionnement     : {condition_number(q_sing):.1f}")
