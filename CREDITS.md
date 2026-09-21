# Credits and provenance

This project builds on an existing open-source Webots UR5 environment. This file
states precisely what was inherited and what was developed here, so the boundary
is never ambiguous.

## Upstream

**[`allan-almeida1/ur5-pick-and-place-webots`](https://github.com/allan-almeida1/ur5-pick-and-place-webots)**
by **Allan Souza Almeida** (2023), MIT licence.

Inherited from it:

- the Webots UR5e scene and the general pick-and-place setup;
- the base of `ur5.py` — the Denavit–Hartenberg forward kinematics, the
  closed-form analytic inverse kinematics with its 8 solution branches, the
  geometric Jacobian (`get_jacobian`), the quintic-polynomial trajectory
  generator, and the Webots device wrappers;
- the original idea of locating the target with a VGG16-based CNN.

## Developed in this project

Everything below was written here, and is what the README's results refer to.

**Neural inverse kinematics**

- `ur5_pytorch_fk.py` — the Denavit–Hartenberg forward kinematics rewritten as a
  batched, differentiable PyTorch module. This is what makes physics-informed
  training possible at all; the upstream FK is NumPy and carries no gradient.
- the `PINN6DOF` network, its normalisation buffers and checkpoint format;
- the hybrid loss — data term, position term, and the orientation term with its
  `r_tool²` weighting;
- the dataset generator, including the single-branch restriction and the
  forward-kinematics validation of every candidate solution;
- `ablation_physics_loss.py` — the 2 × 3 experiment that measures what the
  physics term is actually worth.

**Perception**

- the colour-threshold detector and the pixel-to-world homography, with the
  calibration campaign behind them (148 mm → ~1 mm);
- the camera pose calibration by ray-casting (17.2° tilt → 0.07°);
- the measured comparison against the CNN — a documented negative result: 8 mm
  for the colour detector against 49 mm for the CNN on the same 200 images.

**Control and validation**

- the closed-loop joint-position hold added to `move_to_config`, which removed a
  measured 114 mm of accumulated open-loop error;
- `mesurer_erreur_pinn()` — measuring the network's error on the robot rather
  than on its own model, by comparing poses actually reached against the
  closed-form solver;
- the timing protocol (200 discarded warm-up calls, single thread) and the
  benchmark against IKPY.

**Documentation**

- `README.md`, `DOCUMENTATION.md` and the 22-chapter `GUIDE_DES_CODES.md`.

## Licence

Both this project and the upstream work are distributed under the MIT licence.
See [`LICENSE`](LICENSE).
