"""
Neural IK network for the UR5e.

A three-hidden-layer MLP that maps a Cartesian target to the six joint angles:

    (x, y, z) -> Linear 512 -> SiLU -> Linear 512 -> SiLU
                            -> Linear 512 -> SiLU -> Linear -> (q1 .. q6)

SiLU rather than ReLU
---------------------
SiLU(x) = x * sigmoid(x) is smooth. During training the gradient goes through
the sines and cosines of the differentiable forward kinematics, and a kinked
activation like ReLU would make that gradient discontinuous.

Normalisation lives inside the model
------------------------------------
Inputs are metres (~0.3) and outputs radians (~3), which is badly conditioned
without normalisation. The means and standard deviations are registered as
buffers, so they are saved in the .pth file with the weights. A caller writes
model(xyz) and gets radians back without knowing about the normalisation.
"""
import torch
import torch.nn as nn


class PINN6DOF(nn.Module):
    def __init__(self, input_dim=3, output_dim=6, hidden_dim=256):
        super(PINN6DOF, self).__init__()
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.register_buffer("mean_in", torch.zeros(input_dim))
        self.register_buffer("std_in", torch.ones(input_dim))
        self.register_buffer("mean_out", torch.zeros(output_dim))
        self.register_buffer("std_out", torch.ones(output_dim))

    def forward(self, x):
        x_norm = (x - self.mean_in) / self.std_in
        out_norm = self.net(x_norm)
        return out_norm * self.std_out + self.mean_out

    def save_model(self, filepath):
        checkpoint = {
            'state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'mean_in': self.mean_in,
            'std_in': self.std_in,
            'mean_out': self.mean_out,
            'std_out': self.std_out
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def from_file(cls, filepath):
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
        hidden_dim = checkpoint.get('hidden_dim', 256)
        model = cls(hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['state_dict'])
        return model
