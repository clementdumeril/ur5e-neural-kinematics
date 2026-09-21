"""
Architecture du reseau d'IK neuronale.

Un perceptron a trois couches cachees, qui prend une position cartesienne et
rend les six angles articulaires du UR5e.

    (x, y, z) -> Linear 512 -> SiLU -> Linear 512 -> SiLU
                            -> Linear 512 -> SiLU -> Linear -> (q1 .. q6)

Deux choix meritent d'etre expliques.

SiLU plutot que ReLU
--------------------
SiLU(x) = x * sigmoid(x) est lisse et infiniment derivable. Pendant
l'entrainement le gradient traverse les sin/cos de la cinematique directe
differentiable ; une non-linearite cassee comme ReLU y injecterait des a-coups.

La normalisation vit DANS le modele
-----------------------------------
Les entrees sont des metres (~0,3), les sorties des radians (~+-3) : sans
normalisation le probleme est mal conditionne. En stockant les moyennes et
ecarts-types comme buffers, ils partent dans le fichier .pth et reviennent avec
les poids. L'appelant fait model(xyz) et recoit des radians, sans rien savoir
de la normalisation.
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
