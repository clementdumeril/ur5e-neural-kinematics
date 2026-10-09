"""
Trains the shipped neural IK model (checkpoints/pinn_model_true_physics.pth).

    python src/training/train_true_pinn.py        (~8 min on a laptop CPU)

1. Dataset: 25 000 targets drawn uniformly in the box above the Webots table,
   tool pointing down, solved by the closed-form IK on a single branch
   (shoulder left, elbow up, wrist up). Unreachable targets are dropped.
2. Model: PINN6DOF (src/models/pinn.py), 3 x 512 SiLU.
3. Loss: the predicted angles go through the differentiable forward kinematics
   (src/kinematics/ur5_pytorch_fk.py) and the error is taken on the pose they
   reach:
       0.1 * |q - q_ref|^2  +  |p - p*|^2  +  r_tool^2 * |R - R_ref|_F^2
   The data term keeps the network on the chosen branch; the position and
   orientation terms do the accuracy work.
4. Adam 1e-3, step decay to 2e-5, 100 epochs; the best validation epoch is kept.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math

# --- 1. Webots stub ---
# ur5.py imports the Webots `controller` module; stub it to run outside Webots.
if 'controller' not in sys.modules:
    sys.modules['controller'] = type('controller', (), {'Supervisor': type('Supervisor', (), {})})
    
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _d in ('src/kinematics', 'src/models', 'src/control', 'src/training'):
    _p = os.path.join(_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)
base_dir = _ROOT

from ur5 import build_matrix, inverse_kinematics
from pinn import PINN6DOF
from ur5_pytorch_fk import UR5ForwardKinematicsPyTorch

# --- Weights always go to checkpoints/, never the working directory:
# --- that is where ur5.py loads them from.
MODELS_DIR = os.path.join(_ROOT, 'checkpoints')
os.makedirs(MODELS_DIR, exist_ok=True)


def model_path(filename):
    return os.path.join(MODELS_DIR, filename)


# --- 2. Dataset ---
def generate_webots_dataset(num_samples=25000):
    print(f"Generating {num_samples} targets over the Webots table...")

    # Box above the Webots table, where the cube can be
    xs = np.random.uniform(0.0, 0.4, num_samples)
    ys = np.random.uniform(-0.9, -0.5, num_samples)
    zs = np.random.uniform(0.05, 0.45, num_samples)

    targets = np.column_stack((xs, ys, zs))
    valid_targets = []
    valid_q = []

    for t in targets:
        try:
            # Tool pointing down: [pi, 0, -pi/2]
            T = build_matrix(t, [math.pi, 0, -math.pi/2], euler='XYZ')
            # One solution family (shoulder left, elbow up) keeps the arm clear of the table
            q_sol = inverse_kinematics(T, wrist='up', shoulder='left', elbow='up')

            if not np.any(np.isnan(q_sol)):
                valid_targets.append(t)
                valid_q.append(q_sol)
        except Exception:
            pass

    valid_targets = np.array(valid_targets, dtype=np.float32)
    valid_q = np.array(valid_q, dtype=np.float32)
    print(f"-> {len(valid_targets)} reachable samples.")
    return valid_targets, valid_q


def train():
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Dataset
    X, y = generate_webots_dataset(num_samples=25000)

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 2. Model and normalisation statistics
    model = PINN6DOF(hidden_dim=512)
    model.mean_in = X_train_t.mean(dim=0)
    model.std_in = X_train_t.std(dim=0)
    model.mean_out = y_train_t.mean(dim=0)
    model.std_out = y_train_t.std(dim=0)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # 3. Differentiable forward kinematics
    fk_engine = UR5ForwardKinematicsPyTorch()

    # Characteristic tool length (m). It turns a dimensionless rotation error
    # into the displacement it causes at the gripper tip, so the position and
    # orientation terms share a unit.
    R_OUTIL = 0.05

    epochs = 100
    best_err = float('inf')

    print(f"\n--- Training, {epochs} epochs ---")
    print("Loss = 0.1 * data term (solution branch) + 1.0 * physics term (pose reached)\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()

            # Predicted joint angles
            pred_q = model(bx)

            # Data term: keeps the network on the chosen branch
            loss_data = mse(pred_q, by)

            # Physics term: run the predicted angles through the FK and see
            # where the end-effector lands
            T_pred = fk_engine.forward(pred_q)
            T_ref = fk_engine.forward(by)

            # Position: is the tip on the target?
            loss_pos = mse(T_pred[:, :3, 3], bx)

            # Orientation: is the gripper pointing the right way?
            #
            # Without this term the loss is blind to orientation: turning q6 by
            # 180 deg moves the point by zero microns but flips the gripper, and
            # both poses get the same score.
            #
            # Compared against FK(reference) rather than a rebuilt matrix: both
            # sides go through the same kinematics, so no Euler-angle convention
            # can distort the comparison. Frobenius norm rather than geodesic
            # distance, whose arccos has a gradient that diverges near zero,
            # exactly where the network has to converge.
            loss_rot = mse(T_pred[:, :3, :3], T_ref[:, :3, :3])

            # Bring the rotation error to an equivalent tip displacement
            loss_phys = loss_pos + (R_OUTIL ** 2) * loss_rot

            loss = 0.1 * loss_data + 1.0 * loss_phys

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            T_val = fk_engine.forward(val_pred)
            T_ref_val = fk_engine.forward(y_val_t)

            # Mean position error of the pose reached, in millimetres
            dist_err_mm = torch.mean(
                torch.norm(T_val[:, :3, 3] - X_val_t, dim=1)).numpy() * 1000.0

            # Orientation error in degrees. arccos is fine here: this is a
            # metric, not a loss, so its gradient does not matter.
            R_res = torch.bmm(T_ref_val[:, :3, :3].transpose(1, 2),
                              T_val[:, :3, :3])
            cos = ((R_res[:, 0, 0] + R_res[:, 1, 1] + R_res[:, 2, 2]) - 1.0) / 2.0
            rot_err_deg = torch.rad2deg(
                torch.acos(cos.clamp(-1.0, 1.0))).mean().item()

            angle_mse = mse(val_pred, y_val_t).item()

        # Step decay of the learning rate
        if epoch == 30:
            for g in optimizer.param_groups: g['lr'] = 5e-4
        if epoch == 60:
            for g in optimizer.param_groups: g['lr'] = 1e-4
        if epoch == 80:
            for g in optimizer.param_groups: g['lr'] = 2e-5

        lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:03d}/{epochs} | Position: {dist_err_mm:6.3f} mm | "
              f"Orientation: {rot_err_deg:6.3f} deg | "
              f"Angles: {angle_mse:.5f} | LR: {lr:.1e}")

        if dist_err_mm < best_err:
            best_err = dist_err_mm
            model.save_model(model_path("pinn_model_true_physics.pth"))

    print(f"\n[OK] Best model saved to checkpoints/pinn_model_true_physics.pth ({best_err:.3f} mm)")

if __name__ == "__main__":
    train()
