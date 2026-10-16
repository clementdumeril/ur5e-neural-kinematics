# Credits and provenance

This project is built on an existing open-source Webots UR5 environment. This file says what I inherited and what I wrote, so the boundary is clear.

## Upstream

[`allan-almeida1/ur5-pick-and-place-webots`](https://github.com/allan-almeida1/ur5-pick-and-place-webots) by Allan Souza Almeida (2023), MIT license.

What I took from it:

- the Webots UR5e scene and the general pick-and-place setup;
- the base of `ur5.py`: the Denavit–Hartenberg forward kinematics, the closed-form analytic inverse kinematics with its 8 solution branches, the geometric Jacobian (`get_jacobian`), the quintic-polynomial trajectory generator, and the Webots device wrappers;
- the original idea of locating the target with a VGG16-based CNN.

### A correction to the inherited code

The upstream closed-form IK chooses `theta2` from the shoulder configuration only, while `theta3` depends on the elbow. As a result every `elbow="down"` solution is finite but wrong, about 0.78 m from the target. The same lines are in the upstream repository (`ur5.py`, the "Theta 2" block of `inverse_kinematics`). This repository pairs `theta2` with the elbow choice, and `tests/test_smoke.py` round-trips all eight branches. I have not proposed the change upstream.

## What I wrote

Everything below was written for this project, and it is what the README's results refer to.

Neural inverse kinematics:

- `ur5_pytorch_fk.py`, the Denavit–Hartenberg forward kinematics rewritten as a batched, differentiable PyTorch module. The physics-informed training depends on it, since the upstream FK is NumPy and carries no gradient.
- the `PINN6DOF` network, its normalization buffers and checkpoint format;
- the hybrid loss: data term, position term, and the orientation term with its `r_tool²` weighting;
- the dataset generator, including the single-branch restriction and the forward-kinematics check of every candidate solution;
- `ablation_physics_loss.py`, the 2 × 3 experiment that measures what the physics term is actually worth.

Perception:

- the color-threshold detector and the pixel-to-world homography, with the calibration campaign behind them (148 mm → ~1 mm);
- the camera pose calibration by ray-casting (17.2° tilt → 0.07°);
- the measured comparison against the CNN, which came out negative: 8 mm for the color detector against 49 mm for the CNN on the same 200 images.

Control and validation:

- the closed-loop joint-position hold added to `move_to_config`, which removed a measured 114 mm of accumulated open-loop error;
- `mesurer_erreur_pinn()`, which measures the network's error on the robot rather than on its own model, by comparing the poses actually reached against the closed-form solver;
- the timing protocol (200 discarded warm-up calls, single thread) and the benchmark against IKPY.

Documentation: `README.md`.

## License

This project and the upstream work are both under the MIT license. [`LICENSE`](LICENSE) carries both copyright notices; the upstream one covers the inherited base of `src/control/ur5.py` and the Webots scene.
