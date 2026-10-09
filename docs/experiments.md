# Experiments in detail

This page holds the full tables behind the README. Every number comes from a
script in `experiments/` and, where one exists, from its JSON output in
`checkpoints/`.

- [1. Physics-term ablation](#1-physics-term-ablation)
- [2. Multi-head IK for the multi-valued problem](#2-multi-head-ik-for-the-multi-valued-problem)
- [3. Continuity and the reach boundary](#3-continuity-and-the-reach-boundary)
- [4. Geometric Jacobian and damped least squares](#4-geometric-jacobian-and-damped-least-squares)
- [5. Solver benchmark](#5-solver-benchmark)
- [6. Measuring on the simulated robot](#6-measuring-on-the-simulated-robot)
- [7. Claims I withdrew](#7-claims-i-withdrew)

---

## 1. Physics-term ablation

Script: [`experiments/ablation_physics_loss.py`](../experiments/ablation_physics_loss.py).
Results: [`checkpoints/ablation_physics_loss.json`](../checkpoints/ablation_physics_loss.json).

A 2 × 3 design. Everything is held constant (seed 42, the same 20,201 targets,
architecture, 100 epochs, learning-rate schedule, best-epoch selection) except
two variables: whether the training data keeps one IK branch or mixes all of
them, and the weight $w_p$ of the physics term.

| Training data | $w_p = 0$ (supervised) | $w_p = 1$ (shipped) | $w_p = 20$ |
| :-- | ---: | ---: | ---: |
| One IK branch | 0.238 mm / 0.012° | 0.187 mm / 0.011° | 0.180 mm / 0.044° |
| Branches mixed | 911 mm / 99.0° | 69.6 mm / 98.6° | 3.68 mm / 71.6° |

On single-branch data, which is what the shipped model uses, the physics term
lowers the position error by 21% (0.238 to 0.187 mm). Raising $w_p$ to 20
gains another 0.007 mm of position but makes orientation 4× worse, so I kept
$w_p = 1$.

On mixed-branch data, the supervised network collapses to 911 mm: it learns the
mean of several valid joint configurations, which is not itself a solution. The
physics term alone does not fix that. At $w_p = 1$ the model never improves
after its first epoch; at $w_p = 20$ it reaches 3.68 mm with 71.6° of
orientation error, still unusable for a grasp.

---

## 2. Multi-head IK for the multi-valued problem

Model: [`src/models/multihead.py`](../src/models/multihead.py).
Script: [`experiments/multihypothesis_ik.py`](../experiments/multihypothesis_ik.py).
Results: `checkpoints/multihypothesis_K{1,2,4,8}.json`.

A network is a function: one output per input. When several joint
configurations reach the same target, a single output can only average them.
The multi-head model gives the network K outputs from a shared trunk:

```
                      +-- head 1 --> q(1)
(x,y,z) --> trunk ----+-- head 2 --> q(2)
                      +-- ...
                      +-- head K --> q(K)
```

$$L = \min_k L_{phys}\left( q^{(k)} \right)$$

The minimum is relaxed: the winning head gets weight $1 - \varepsilon$ and the
others share $\varepsilon$. With a strict minimum, the first head to win takes
all the gradient and the others never train. No branch labels are used anywhere
in training. At inference, the forward kinematics runs on all K outputs and the
closest pose wins, using the same position-plus-orientation weighting as the
loss; the reported latency includes that selection.

### Result at K = 2

| Training | Position | Orientation | Solve time (benchmark) | Branch chosen by hand |
| :-- | ---: | ---: | ---: | :-: |
| Single head, one branch | 0.187 mm | 0.011° | 0.543 ms | yes |
| Single head, mixed branches ($w_p = 20$) | 3.684 mm | 71.6° | 0.543 ms | no |
| 2 heads, mixed branches | 0.314 mm | 0.202° | 4.911 ms | no |

Solve times are the benchmark medians of section 5, which include the tensor conversion and, for two heads, the FK and the selection.

### Which branches exist

With the tool fixed pointing down, only 4 of the 8 closed-form branches ever give
a valid solution on this workspace. `left-up-up` and `right-up-up` are valid on
every target; `left-down-up` and `right-down-up` on about 35% of them; the
other four never. Every target has 2 or 4 valid branches.

Per-branch recall is computed with a periodic angular distance: +179° and −179°
are 2° apart, not 358°.

### Sweep over K

Same cached dataset (fingerprint `b9e74b3f`), same seed, 100 epochs. Latency here is timed inside the training script on a tensor input, without the conversion the benchmark includes: compare it across K only.

| K | Parameters | Position | Orientation | Branch recall | Latency |
| --: | --: | --: | --: | --: | --: |
| 1 | 660 k | 8.881 mm | 17.87° | 11.8% | 4.34 ms |
| 2 | 793 k | 0.314 mm | 0.202° | 36.7% | 4.61 ms |
| 4 | 1.06 M | 0.357 mm | 0.322° | 82.2% | 5.23 ms |
| 8 | 1.59 M | 0.439 mm | 0.278° | 86.8% | 5.67 ms |

K = 2 gives the most accurate single answer. Recall keeps rising with K because
best-of-K needs only one correct head, while recovering the whole solution set
needs more. At K = 2 one head reproduces `right-up-up` 99% of the time and the
other does not land on a second branch. At K = 8, three of the four branches are
recovered above 88%.

K = 1 is the control: same code, same loss, one head, 8.881 mm. It has no data
term, unlike the mixed-branch runs of the ablation, so the two are not directly
comparable.

Latency: each extra head costs about 0.19 ms. Most of the 4 to 6 ms is running
the forward kinematics and the selection at all, which the single-output
network does not need.

### Caveats

- Against the shipped single-branch model, K = 2 is 1.7× worse in position, 18×
  worse in orientation and about 9× slower (4.9 ms against 0.54 ms). For one tool orientation and
  top-down grasps, fixing the branch in the data generator is the better
  engineering choice, and that is what the repository ships.
- Branch coverage is not stable across runs. At K = 4, `right-down-up` is
  recovered 0% of the time and `left-down-up` 81%; at K = 8 it is 88% and
  27%.
- Validation error oscillates during training because the winner assignment
  keeps changing. The saved checkpoint is the best epoch, not the last.

This approach becomes useful when the branch cannot be fixed in advance: several
tool orientations, obstacle avoidance that needs an elbow-down route, or full
SE(3) pose IK.

---

## 3. Continuity and the reach boundary

Script: [`experiments/mesure_continuite.py`](../experiments/mesure_continuite.py).

A 400-sample circle through the workspace, recording the joint-space jump
between consecutive poses:

| Solver | Median jump | p95 | Max |
| :-- | ---: | ---: | ---: |
| Closed form, fixed branch | 0.0084 rad | 0.0108 | 0.0109 |
| Closed form, first valid branch | 0.0084 rad | 0.0108 | 0.0109 |
| Network | 0.0084 rad | 0.0108 | 0.0109 |

Identical, with no branch change in 399 steps: with the tool pointing down, the
preferred branch is valid wherever the target is reachable.

Walking along $x$ past the reach limit:

| $x$ (m) | Closed form | Network, error of the returned pose |
| ---: | :-- | ---: |
| +0.40 | solution | 0.2 mm |
| +0.50 | solution | 2.0 mm |
| +0.55 | exception | 68 mm |
| +0.60 | exception | 389 mm |
| +0.70 | exception | 1,460 mm |

The closed-form solver raises an exception. The network returns six angles with
no sign that anything is wrong. Any deployment needs an explicit reachability
check in front of it.

---

## 4. Geometric Jacobian and damped least squares

Code: [`src/kinematics/jacobian.py`](../src/kinematics/jacobian.py),
[`src/kinematics/ik_dls.py`](../src/kinematics/ik_dls.py).

The 6×6 geometric Jacobian is built column by column:

$$J_{v,i} = z_{i-1} \times (p_e - p_{i-1}), \qquad J_{\omega,i} = z_{i-1}$$

It is checked against a central-difference derivative of the forward kinematics
over 200 random configurations: maximum deviation 3.2 × 10⁻¹⁰, mean
1.5 × 10⁻¹⁰.

| Configuration | Manipulability $\sqrt{\det(JJ^T)}$ | Condition number |
| :-- | ---: | ---: |
| Rest pose | 0.0287 | 16.5 |
| Arm fully extended ($q = 0$) | 0.0000 | ∞ |

The damped least-squares solver uses

$$\Delta q = J^T\left(JJ^T + \lambda^2 I\right)^{-1} e$$

with $e$ the 6-vector pose error, its rotation part taken as the logarithm on
SO(3). The damping is adaptive: zero when the arm is well conditioned, growing
as the smallest singular value of $J$ approaches a threshold.

From a single fixed start it either converges in about 23 iterations to
sub-micron accuracy or stalls in a local minimum tens of millimetres away. It
fails on 59.5% of workspace targets. Eight random restarts bring that to 0%
at roughly twice the cost.

---

## 5. Solver benchmark

Script: [`experiments/benchmark_solveurs.py`](../experiments/benchmark_solveurs.py).
Results: [`checkpoints/benchmark_solveurs.json`](../checkpoints/benchmark_solveurs.json).

200 targets, 200 warm-up calls discarded, `torch.set_num_threads(1)`. Times are
measured at the call site and include the NumPy-to-tensor conversion a caller
has to pay.

| Solver | Median | Mean | p95 | Error | Failures | Differentiable |
| :-- | ---: | ---: | ---: | ---: | ---: | :-: |
| Closed form | 0.232 ms | 0.272 | 0.488 | exact | 0% | no |
| Network, single output | 0.543 ms | 0.607 | 1.196 | 0.165 mm | 0% | yes |
| Network, 2 heads + selection | 4.911 ms | 5.085 | 7.679 | 0.253 mm | 0% | yes |
| Damped least squares, one start | 135.8 ms | 96.2 | 165.0 | exact | 59.5% | no |
| Damped least squares, 8 restarts | 287.4 ms | 190.3 | 341.5 | exact | 0% | no |
| IKPY (iterative) * | 29.6 ms | 31.1 | 45.6 | n/a | 0% | no |

\* IKPY loads the URDF chain, which does not describe the same robot as this
project's DH table: for identical joint angles the two forward kinematics place
the end-effector 1.3 to 1.8 m apart. IKPY is therefore given targets drawn from
its own forward kinematics, and only its timing is comparable. Reconciling the
two models is open work.

The network is 2.3× slower than the closed form and about 55× faster than IKPY.

---

## 6. Measuring on the simulated robot

`mesurer_erreur_pinn()` in
[`ur5_controller_pandahand.py`](../webots/controllers/ur5_controller_pandahand/ur5_controller_pandahand.py)
drives the arm to the same target twice, once with the closed-form solver and
once with the network, and compares the poses actually reached in the physics
engine. The comparison cancels a constant 108 mm offset between the DH tool frame
and the `frame6` node that Webots reports. What remains includes the DH model,
the motor servo loop and the physics. The campaign also repeats the same target
with the same solver to measure the arm's own repeatability.

| Point | Error |
| :-- | ---: |
| Grasp point | 0.30 mm |
| Above the drop tray ($x = -0.20$, outside the training box) | 7 to 11 mm |

These figures were measured before the orientation term was added to the loss.
The current model is better on validation and has not been re-measured in
Webots yet.

The campaign runs after the scenario. When it ran first, it nudged the cube and
the scenario then used stale coordinates, which looked exactly like the network
failing.

---

## 7. Claims I withdrew

Earlier versions of the README made three claims that the experiments above
refuted:

1. "A supervised network averages the 8 IK solutions and our PINN resolves the
   ambiguity." Both dataset generators keep a single branch before training,
   so there was nothing to average. The measured effect of the physics term is
   21% (section 1).
2. "The network is continuous where analytic solvers jump between branches."
   The jumps are identical and no branch change occurs (section 3).
3. "The network is faster than the closed-form solver." That first table timed
   a cold PyTorch against a warm NumPy. A later version timed only the forward
   pass, without the tensor conversion, and quoted a 182× speed-up over IKPY
   measured across incompatible frames. Section 5 is the corrected protocol.

What came out of fixing them:

- every timing discards 200 warm-up calls and pins one thread;
- accuracy is measured on the poses the simulated robot reaches, not by feeding
  the network's output back through the same forward kinematics;
- every closed-form solution is checked by forward kinematics before it enters
  a dataset (the closed form returns finite but wrong angles for some branch
  combinations, up to 784 mm off, so filtering on NaN is not enough);
- experiments share one cached dataset and report its fingerprint.
