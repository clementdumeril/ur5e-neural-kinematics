import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
for _d in ('src/kinematics', 'src/models', 'src/control', 'src/training'):
    _p = os.path.join(_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)
parent_dir = _ROOT

from controller import Supervisor

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    print("\n=======================================================")
    print("🥊 COMBAT : TRUE PINN vs MATHÉMATIQUES CLASSIQUES")
    print("=======================================================\n")
    
    # Textes HUD Statiques (Titres)
    robot.setLabel(0, "UR5e A (Rouge) : IA TRUE PINN", 0.02, 0.02, 0.1, 0xff0000, 0, "Arial")
    robot.setLabel(1, "UR5e B (Vert) : IK ANALYTIQUE", 0.02, 0.06, 0.1, 0x00ff00, 0, "Arial")
    
    # Boucle infinie pour maintenir le controleur en vie
    while robot.step(timestep) != -1:
        pass

if __name__ == "__main__":
    main()
