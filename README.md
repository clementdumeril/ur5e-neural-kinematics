# UR5e Neural Kinematics

A small neural network that solves inverse kinematics for a UR5e arm. I train it through a differentiable PyTorch copy of the robot's forward kinematics, and it runs inside a camera-guided pick-and-place loop in Webots. Personal project, August to October 2026.

The full write-up, with the multi-seed results and the verification, is [`paper/main.pdf`](paper/main.pdf) (12 pages).

[![smoke](https://github.com/clementdumeril/ur5e-neural-kinematics/actions/workflows/smoke.yml/badge.svg)](https://github.com/clementdumeril/ur5e-neural-kinematics/actions/workflows/smoke.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![UR5e in Webots: the camera finds the cube, the network solves each move, the arm grasps the cube and drops it in the tray](assets/demo.gif)

*One run of [`my_first_simulation_pandahand.wbt`](webots/worlds/my_first_simulation_pandahand.wbt) at 2× speed.*

## Results

| | Value | Source |
| :-- | --: | :-- |
| Position error, 2,021 held-out test targets | 0.19 mm | [`C_shipped.json`](results/seed_study/C_shipped.json) |
| Solve time, median, one thread | 0.53 ms | [`benchmark.json`](results/benchmark/benchmark.json) |
| Against IKPy on the same pose problem (IKPy fails on 58 % of targets) | 72× faster | same |
| Network vs closed form at the grasp point, Webots closed camera loop (one run; mean 4.8 mm over 4 points) | 0.33 mm | [`closed_loop_console.txt`](results/webots/closed_loop_console.txt) |

A multi-head version trained on branch-mixed data without labels recovers 51% of the 8 IK branches with 8 heads, but has about 1.5 times the error (0.31 against 0.19 to 0.21 mm), so the shipped model keeps one branch.

The network solves position IK with the tool pointing down (R³ → R⁶), not full 6-DoF pose IK. The closed-form solver is still 2.3× faster.

## What I built

I started from [`allan-almeida1/ur5-pick-and-place-webots`](https://github.com/allan-almeida1/ur5-pick-and-place-webots) (MIT), which gave me the Webots scene, a NumPy DH forward kinematics, the closed-form IK and the device wrappers. On top of it I wrote:

- [`ur5_pytorch_fk.py`](src/kinematics/ur5_pytorch_fk.py): the DH chain as a batched PyTorch module, so an end-effector error can send a gradient back to the joint angles.
- [`pinn.py`](src/models/pinn.py) and [`train_true_pinn.py`](src/training/train_true_pinn.py): the MLP (530k parameters) and its training. The loss is computed on the pose the predicted angles reach, plus a data term that keeps one IK branch. On single-branch data the FK terms gave no significant gain over five seeds.
- [`multihead.py`](src/models/multihead.py): K heads with a winner-take-all loss, trained without branch labels.
- [`jacobian.py`](src/kinematics/jacobian.py) and [`ik_dls.py`](src/kinematics/ik_dls.py): a geometric Jacobian and a damped least-squares solver, used as baselines.
- A fix to the inherited closed form, which returned elbow-down solutions about 0.78 m off. All eight branches now round-trip and the smoke test checks them.
- Perception: the inherited pixel-to-table constants were 148 mm off, so I fit a homography from a calibration grid (1.1 mm median residual). A color threshold beat the VGG16 regressor I tried (8 mm against 49 mm).
- Control: a closed-loop joint hold at the end of each move, which removed up to 114 mm of open-loop error.

[`CREDITS.md`](CREDITS.md) lists what I inherited and what I wrote. The full tables are in [`docs/experiments.md`](docs/experiments.md).

## Limitations

- The error grows outside the training box: 14.3 mm median there on a z = 0.25 m slice, 7 to 11 mm above the tray.
- Past the reach limit the network still returns six angles, up to 1.78 m off, with no warning. It needs a reachability check in front of it.
- One tool orientation, one IK branch, simulation only.

## Reproduce

```bash
git clone https://github.com/clementdumeril/ur5e-neural-kinematics.git
cd ur5e-neural-kinematics
pip install -r requirements.txt

python tests/test_smoke.py                    # seconds
python src/training/train_true_pinn.py        # retrain, ~8 min
python experiments/ablation_physics_loss.py   # loss ablation, ~1 h
python experiments/multihypothesis_ik.py      # multi-head IK, ~55 min
python experiments/benchmark_solveurs.py      # solver timing, ~4 min
python experiments/figures.py                 # figures from the result files
```

For the simulation, open `webots/worlds/my_first_simulation_pandahand.wbt` in Webots R2023a or later (tested on R2025a) and press Play. The first launch needs internet to fetch the PROTOs. To force a Python interpreter, set `COMMAND` under `[python]` in `webots/controllers/*/runtime.ini`.

## Credits

The Webots scene and the base of `src/control/ur5.py` come from [`allan-almeida1/ur5-pick-and-place-webots`](https://github.com/allan-almeida1/ur5-pick-and-place-webots) by Allan Souza Almeida, MIT license.

Clément Dumeril · [clement.dumeril.net](https://clement.dumeril.net) · MIT License, see [`LICENSE`](LICENSE)
