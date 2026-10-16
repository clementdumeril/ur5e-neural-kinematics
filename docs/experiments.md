# Experiments

The tables behind the README. Each number comes from a script in `experiments/` and its JSON output in `checkpoints/` or `results/`. The multi-seed runs and the full protocol are in [`paper/main.pdf`](../paper/main.pdf).

- [1. Loss ablation](#1-loss-ablation)
- [2. Multi-head IK](#2-multi-head-ik)
- [3. Continuity and the reach boundary](#3-continuity-and-the-reach-boundary)
- [4. Jacobian and damped least squares](#4-jacobian-and-damped-least-squares)
- [5. Solver benchmark](#5-solver-benchmark)
- [6. Measuring on the simulated robot](#6-measuring-on-the-simulated-robot)

## 1. Loss ablation

Script: [`experiments/ablation_physics_loss.py`](../experiments/ablation_physics_loss.py), seed study in `results/seed_study/`.

On single-branch data, over 5 paired seeds on a held-out test set, the FK terms move the error from 0.241 ± 0.021 to 0.214 ± 0.039 mm. They help in 3 of 5 seeds and the difference is not significant (p = 0.26).

On branch-mixed data (single run), the supervised network collapses to 911 mm, the mean of several valid configurations. With the FK terms weighted 20× it reaches 3.68 mm with 71.6° of orientation error, still unusable for a grasp.

## 2. Multi-head IK

Model: [`src/models/multihead.py`](../src/models/multihead.py). Script: [`experiments/multihypothesis_ik.py`](../experiments/multihypothesis_ik.py).

A shared trunk with K heads, trained on branch-mixed data with a relaxed winner-take-all loss and no branch labels. At inference the FK scores the K proposals and the closest one wins.

Over seeds on a test set (paper, Section 6.3):

- going from 1 to 8 heads, the share of branches recovered goes from 13% to 78% on the 4 elbow-up branches, and from 14% to 51% on all 8;
- every run with 2 to 8 heads stays between 0.26 and 0.62 mm;
- with one head and the FK loss alone, the network settles on one branch in 2 of 3 seeds (0.33 to 0.44 mm) and stalls between branches in the third.

Two heads have about twice the error of single-branch training and are 8× slower (4.0 against 0.53 ms). For one tool orientation, fixing the branch in the data generator is the better choice, and that is what I ship. Multiple heads become useful when the branch can't be fixed in advance.

## 3. Continuity and the reach boundary

Script: [`experiments/mesure_continuite.py`](../experiments/mesure_continuite.py).

On a 400-sample circle through the workspace, the network and the closed form have the same joint-space jumps (median 0.0084 rad) and no branch change.

Walking along x past the reach limit:

| x (m) | Closed form | Network, error of the returned pose |
| ---: | :-- | ---: |
| +0.40 | solution | 0.2 mm |
| +0.50 | solution | 2.0 mm |
| +0.55 | exception | 68 mm |
| +0.60 | exception | 389 mm |
| +0.70 | exception | 1,460 mm |

The network gives no sign that anything is wrong, so it needs a reachability check in front of it.

## 4. Jacobian and damped least squares

Code: [`src/kinematics/jacobian.py`](../src/kinematics/jacobian.py), [`src/kinematics/ik_dls.py`](../src/kinematics/ik_dls.py).

The geometric Jacobian matches a central-difference derivative of the FK to 3.2 × 10⁻¹⁰ over 200 random configurations. The damped least-squares solver uses adaptive damping. From one fixed start it fails on 59.5% of targets; eight random restarts bring that to 0% at about twice the cost.

## 5. Solver benchmark

Script: [`experiments/benchmark_solveurs.py`](../experiments/benchmark_solveurs.py). Results: [`results/benchmark/benchmark.json`](../results/benchmark/benchmark.json).

Same 200 pose targets for every solver, 200 warm-up calls on other targets, one thread, tensor conversion included. A solution more than 1 mm or 0.1° off counts as a failure.

| Solver | Median | p95 | Error | Failures |
| :-- | ---: | ---: | ---: | ---: |
| Closed form | 0.232 ms | 0.607 | exact | 0% |
| Network, single output | 0.531 ms | 1.296 | 0.165 mm | 0% |
| Network, 2 heads + selection | 4.029 ms | 7.159 | 0.261 mm | 0% |
| IKPy, full pose | 38.1 ms | 61.9 | exact | 58.0% |
| Damped least squares, one start | 136.0 ms | 260.4 | exact | 59.5% |
| Damped least squares, 8 restarts | 270.0 ms | 457.4 | exact | 0% |

IKPy runs on a chain built from this project's DH table and starts from the same rest configuration as damped least squares. The network is 2.3× slower than the closed form and about 72× faster than IKPy.

## 6. Measuring on the simulated robot

`mesurer_erreur_pinn()` in [`ur5_controller_pandahand.py`](../webots/controllers/ur5_controller_pandahand/ur5_controller_pandahand.py) sends the arm to the same target once with the closed form and once with the network, and compares the poses reached in the physics engine. Repeating a target with the same solver gives the arm's repeatability.

Shipped model, Webots R2025a, October 2026 ([`closed_loop_console.txt`](../results/webots/closed_loop_console.txt)), one target per point:

| Point | Error | Repeatability |
| :-- | ---: | ---: |
| Above the cube | 0.45 mm | 0.00 mm |
| Grasp point | 0.33 mm | 0.04 mm |
| Above the tray (x = −0.20, outside the training box) | 10.92 mm | 0.00 mm |
| Drop height over the tray | 7.34 mm | 0.00 mm |

The campaign runs after the scenario. Run first, it nudged the cube and the scenario then used stale coordinates.
