"""
Test de fumee : tout s'importe-t-il encore, et les chemins tiennent-ils ?

Ce fichier existe a cause d'une panne reelle. Lors du passage a l'arborescence
src/, chaque point d'entree calculait la racine du projet par un nombre de
'..' qui dependait de sa profondeur, et un controleur importait un module
AVANT d'avoir ajoute son repertoire au chemin -- ce qui ne fonctionnait que
grace a un fichier duplique. Rien de tout cela n'apparait a la lecture ; il
faut l'executer pour le voir.

    python tests/test_smoke.py

Aucune dependance a pytest : ce test doit pouvoir tourner partout.
"""
import os
import sys
import traceback

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Webots fournit le module `controller` ; hors simulation on le simule.
if 'controller' not in sys.modules:
    sys.modules['controller'] = type(
        'controller', (), {'Supervisor': type('Supervisor', (), {})})

SRC = ('src/kinematics', 'src/models', 'src/control', 'src/training')

# Chaque point d'entree, avec sa profondeur par rapport a la racine.
POINTS_ENTREE = [
    ('src/control/ur5.py', 2),
    ('src/training/train_true_pinn.py', 2),
    ('src/training/train_supervised_ik.py', 2),
    ('experiments/ablation_physics_loss.py', 1),
    ('experiments/mesure_continuite.py', 1),
    ('webots/controllers/ur5_controller/ur5_controller.py', 3),
    ('webots/controllers/ur5_controller_pandahand/ur5_controller_pandahand.py', 3),
    ('webots/controllers/comparison_controller/comparison_controller.py', 3),
    ('webots/controllers/multi_controller/multi_controller.py', 3),
    ('webots/controllers/data_collector/data_collector.py', 3),
]

MODULES = [
    ('ur5_pytorch_fk', 'UR5ForwardKinematicsPyTorch'),
    ('ur5e_6dof_ik', 'UR5e6DOF'),
    ('ur5e_se3_ik', 'UR5eSE3'),
    ('ur5e_trajectory', None),
    ('pinn', 'PINN6DOF'),
    ('ur5', 'UR5'),
]

FICHIERS = [
    'checkpoints/pinn_model_true_physics.pth',
    'data/calibration.json',
    'requirements.txt',
    'CREDITS.md',
]

echecs = []


def verifie(nom, condition, detail=''):
    if condition:
        print(f"  [OK]   {nom}")
    else:
        print(f"  [KO]   {nom}  {detail}")
        echecs.append(nom)


print("=" * 70)
print("TEST DE FUMEE")
print("=" * 70)

print("\n1. Les repertoires source existent")
for d in SRC:
    verifie(d, os.path.isdir(os.path.join(RACINE, *d.split('/'))))

print("\n2. Chaque point d'entree remonte bien a la racine")
for chemin, profondeur in POINTS_ENTREE:
    absolu = os.path.join(RACINE, *chemin.split('/'))
    if not os.path.isfile(absolu):
        verifie(chemin, False, '(fichier absent)')
        continue
    calcule = os.path.abspath(
        os.path.join(os.path.dirname(absolu), *(['..'] * profondeur)))
    verifie(chemin, calcule == RACINE, f'-> {calcule}')

print("\n3. Les modules s'importent")
for d in SRC:
    p = os.path.join(RACINE, *d.split('/'))
    if p not in sys.path:
        sys.path.insert(0, p)

for module, symbole in MODULES:
    try:
        m = __import__(module)
        if symbole is not None and not hasattr(m, symbole):
            verifie(f'{module}.{symbole}', False, '(symbole absent)')
        else:
            verifie(module, True)
    except Exception:
        verifie(module, False, f'\n{traceback.format_exc(limit=1)}')

print("\n4. Les fichiers de donnees sont la ou on les cherche")
for f in FICHIERS:
    verifie(f, os.path.isfile(os.path.join(RACINE, *f.split('/'))))

print("\n5. Le reseau se charge et repond")
try:
    import torch
    from pinn import PINN6DOF
    from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch
    modele = PINN6DOF.from_file(
        os.path.join(RACINE, 'checkpoints', 'pinn_model_true_physics.pth'))
    modele.eval()
    with torch.no_grad():
        cible = torch.tensor([[0.198, -0.728, 0.030]])
        pose = UR5ForwardKinematicsPyTorch().forward_pos(modele(cible))
    err = torch.norm(pose - cible).item() * 1000
    verifie(f'erreur sur une cible connue : {err:.3f} mm', err < 1.0)
except Exception:
    verifie('chargement du reseau', False, f'\n{traceback.format_exc(limit=1)}')

print("\n" + "=" * 70)
if echecs:
    print(f"ECHEC : {len(echecs)} verification(s) en erreur")
    for e in echecs:
        print(f"  - {e}")
    sys.exit(1)
print("Tout est vert.")
print("=" * 70)
