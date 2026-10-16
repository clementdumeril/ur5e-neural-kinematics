# Credits

## Upstream

[`allan-almeida1/ur5-pick-and-place-webots`](https://github.com/allan-almeida1/ur5-pick-and-place-webots) by Allan Souza Almeida (2023), MIT license. I took from it:

- the Webots UR5e scene and the pick-and-place setup;
- the base of `ur5.py`: DH forward kinematics, closed-form IK with its 8 branches, the Jacobian (`get_jacobian`), quintic trajectories and the device wrappers;
- the idea of locating the target with a VGG16 CNN.

The upstream closed-form IK picks `theta2` from the shoulder only, so every `elbow="down"` solution is about 0.78 m off (the "Theta 2" block of `inverse_kinematics`). I pair `theta2` with the elbow choice here, and `tests/test_smoke.py` round-trips all eight branches. I have not proposed the fix upstream.

## What I wrote

- `ur5_pytorch_fk.py`, the differentiable PyTorch FK the training depends on;
- the `PINN6DOF` network, the loss, the dataset generator and the ablation script;
- the color detector, the homography calibration and the camera pose correction;
- the closed-loop hold in `move_to_config`;
- `mesurer_erreur_pinn()`, which measures the network's error on the simulated robot;
- the solver benchmark against IKPy.

## License

Both this project and the upstream work are MIT. [`LICENSE`](LICENSE) carries both copyright notices.
