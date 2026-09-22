"""
Carte de l'erreur du reseau dans l'espace de travail.

Les tableaux disent que l'erreur vaut 0,185 mm en validation et monte a 7-11 mm
au-dessus du bac. Ils ne disent pas OU elle monte, ni a quelle vitesse. Une
carte le montre d'un coup d'oeil, et elle reunit sur une seule image trois
resultats jusqu'ici disperses :

  1. la precision a l'interieur de la zone d'entrainement ;
  2. la degradation en extrapolation, des qu'on en sort ;
  3. le fait que le reseau ne signale jamais qu'il est sorti de son domaine --
     il rend six angles plausibles et faux, sans le moindre avertissement.

Protocole
---------
Grille reguliere dans le plan horizontal, a hauteur fixe. Pour chaque point on
demande au reseau ses six angles, on les repasse dans la cinematique directe et
on mesure la distance entre la pose obtenue et la cible demandee. C'est donc
l'erreur REELLE du reseau, pas une perte.

La grille deborde volontairement de la zone d'entrainement : c'est justement
au-dela que le comportement devient interessant.

La zone atteignable est calculee separement, par le solveur analytique : la ou
il echoue, aucune solution n'existe, et l'erreur du reseau n'y a plus de sens
physique -- elle mesure alors l'ecart a une cible impossible.
"""
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

if 'controller' not in sys.modules:
    sys.modules['controller'] = type(
        'controller', (), {'Supervisor': type('Supervisor', (), {})})

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ('src/kinematics', 'src/models', 'src/control', 'src/training'):
    _p = os.path.join(_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                            # noqa: E402
from ur5 import build_matrix, inverse_kinematics        # noqa: E402
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch  # noqa: E402
from pinn import PINN6DOF                               # noqa: E402

ROT_DOWN = [math.pi, 0.0, -math.pi / 2]

# Zone d'entrainement, pour rappel : x [0, 0.4], y [-0.9, -0.5], z [0.05, 0.45]
ZONE = dict(x0=0.0, x1=0.4, y0=-0.9, y1=-0.5)

# La grille deborde largement : c'est le dehors qui est instructif.
X_MIN, X_MAX = -0.35, 0.75
Y_MIN, Y_MAX = -1.15, -0.25
Z = float(os.environ.get('HM_Z', 0.25))
N = int(os.environ.get('HM_N', 150))

# Points reels du scenario, pour situer les chiffres du README.
REPERES = [
    (0.198, -0.728, "cube"),
    (-0.200, -0.730, "drop tray"),
    (-0.100, -0.680, "reading pose"),
]


def calculer():
    modele = PINN6DOF.from_file(os.path.join(
        _ROOT, 'checkpoints', 'pinn_model_true_physics.pth'))
    modele.eval()
    fk = UR5ForwardKinematicsPyTorch()

    xs = np.linspace(X_MIN, X_MAX, N)
    ys = np.linspace(Y_MIN, Y_MAX, N)
    XX, YY = np.meshgrid(xs, ys)
    cibles = np.stack([XX.ravel(), YY.ravel(),
                       np.full(XX.size, Z)], axis=1).astype(np.float32)

    with torch.no_grad():
        q = modele(torch.tensor(cibles))
        p = fk.forward_pos(q).numpy()
    erreur = np.linalg.norm(p - cibles, axis=1).reshape(XX.shape) * 1000.0

    # Atteignabilite, par le solveur analytique.
    print("  calcul de la frontiere d'atteignabilite...", flush=True)
    atteignable = np.zeros(XX.shape, dtype=bool)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            T = build_matrix(np.array([XX[i, j], YY[i, j], Z]), ROT_DOWN, euler='XYZ')
            try:
                qa = inverse_kinematics(T, shoulder='left', wrist='up', elbow='up')
            except Exception:
                continue
            atteignable[i, j] = qa is not None and not np.any(np.isnan(qa))

    return XX, YY, erreur, atteignable


def tracer(XX, YY, erreur, atteignable):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(10.0, 8.2), dpi=150)
    fig.patch.set_facecolor('white')
    # Marges explicites : bbox_inches='tight' combine a un texte en coordonnees
    # d'axes faisait chevaucher la barre de couleur, le titre et la legende.
    fig.subplots_adjust(left=0.09, right=0.88, top=0.90, bottom=0.17)

    err = np.clip(erreur, 0.02, 2000.0)
    im = ax.pcolormesh(XX, YY, err, norm=LogNorm(vmin=0.05, vmax=1000.0),
                       cmap='magma_r', shading='auto')

    # Hors d'atteinte : hachure, l'erreur n'y a plus de sens physique.
    ax.contourf(XX, YY, (~atteignable).astype(float), levels=[0.5, 1.5],
                colors='none', hatches=['////'])
    ax.contour(XX, YY, atteignable.astype(float), levels=[0.5],
               colors='white', linewidths=2.0, linestyles='--')

    # Zone d'entrainement
    ax.add_patch(Rectangle((ZONE['x0'], ZONE['y0']),
                           ZONE['x1'] - ZONE['x0'], ZONE['y1'] - ZONE['y0'],
                           fill=False, edgecolor='#2563eb', linewidth=2.5, zorder=5))
    ax.text(ZONE['x0'] + 0.01, ZONE['y1'] - 0.025, 'training box',
            color='#2563eb', fontsize=11, fontweight='bold', zorder=6)

    for x, y, nom in REPERES:
        ax.plot(x, y, 'o', ms=8, mfc='white', mec='#111827', mew=2, zorder=7)
        ax.annotate(nom, (x, y), textcoords='offset points', xytext=(9, 7),
                    fontsize=10, color='#111827', fontweight='bold', zorder=7)

    ax.set_xlabel('x (m, robot frame)')
    ax.set_ylabel('y (m, robot frame)')
    ax.set_title(f'Neural IK position error across the workspace   (z = {Z:.2f} m)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_aspect('equal')

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('position error (mm, log scale)')

    fig.text(0.09, 0.115,
             'Dashed white line: reachability limit of the analytic solver. '
             'Hatched: no solution exists there, so the error has no physical meaning.',
             fontsize=10, color='#374151', va='top')
    fig.text(0.09, 0.078,
             'Inside the blue box: median 0.176 mm. Reachable but outside it: '
             '14.3 mm, 81x worse.',
             fontsize=10, color='#374151', va='top')
    fig.text(0.09, 0.041,
             'And it never reports leaving its domain: past the reachability limit it '
             'still returns six plausible angles, up to 1.78 m wrong.',
             fontsize=10, color='#374151', va='top')

    out = os.path.join(_ROOT, 'assets', 'fig_workspace_error.png')
    fig.savefig(out, facecolor='white')
    print(f"\n  {out}")
    return out


def main():
    print("=" * 74)
    print(f"CARTE D'ERREUR -- grille {N}x{N} a z = {Z:.2f} m")
    print("=" * 74)
    XX, YY, erreur, atteignable = calculer()

    boite = ((XX >= ZONE['x0']) & (XX <= ZONE['x1']) &
             (YY >= ZONE['y0']) & (YY <= ZONE['y1']))

    # La boite d'entrainement n'est pas entierement atteignable : le generateur
    # y tirait des cibles puis jetait celles sans solution. Comparer sur la
    # boite brute melangerait extrapolation et impossibilite -- et donnait un
    # p95 de 363 mm "a l'interieur de la zone", ce qui n'avait aucun sens.
    dans = boite & atteignable
    hors = atteignable & ~boite

    print(f"\n  dans la zone ET atteignable : mediane {np.median(erreur[dans]):7.3f} mm, "
          f"p95 {np.percentile(erreur[dans], 95):7.3f} mm")
    print(f"  atteignable mais hors zone  : mediane {np.median(erreur[hors]):7.3f} mm, "
          f"p95 {np.percentile(erreur[hors], 95):7.3f} mm")
    print(f"  hors d'atteinte             : mediane {np.median(erreur[~atteignable]):7.1f} mm, "
          f"max {erreur[~atteignable].max():7.1f} mm")
    print(f"\n  rapport hors zone / dans zone : "
          f"x{np.median(erreur[hors]) / np.median(erreur[dans]):.0f}")

    tracer(XX, YY, erreur, atteignable)
    print("=" * 74)


if __name__ == "__main__":
    main()
