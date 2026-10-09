"""
Multi-hypothesis IK network.

Why this file exists
--------------------
The ablation (experiments/ablation_physics_loss.py) showed that a single-output
network trained on data mixing the IK solution branches collapses to 911 mm when
supervised. A network is a function, one output per input, and the average of
two valid joint configurations is a solution to neither.

The shipped model avoids the problem by keeping one branch in the data
generator. This network handles it directly by producing K outputs:

                          +-- head 1 --> q(1)
    (x,y,z) --> trunk ----+-- head 2 --> q(2)
                          +-- ...
                          +-- head K --> q(K)

and the loss keeps only the best one:

    L = min_k  L_phys( q(k) )

Each head is free to specialise on a different branch and is not penalised for
missing the others. No branch labels are used.

Head collapse
-------------
With a strict min, the first head to win takes all the gradient and the others
never train. The min is therefore relaxed: the winner gets weight (1 - eps) and
the K-1 others share eps, so they keep learning the general structure without
being pulled onto the winner's solution.
"""
import torch
import torch.nn as nn


class MultiHeadIK(nn.Module):
    """
    Tronc partage, K tetes, normalisation embarquee comme PINN6DOF.

    forward(x) -> (B, K, 6)
    """

    def __init__(self, input_dim=3, output_dim=6, hidden_dim=512,
                 n_heads=8, head_dim=256):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.output_dim = output_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, head_dim), nn.SiLU(),
                nn.Linear(head_dim, output_dim),
            )
            for _ in range(n_heads)
        ])

        self.register_buffer("mean_in", torch.zeros(input_dim))
        self.register_buffer("std_in", torch.ones(input_dim))
        self.register_buffer("mean_out", torch.zeros(output_dim))
        self.register_buffer("std_out", torch.ones(output_dim))

    def forward(self, x):
        h = self.trunk((x - self.mean_in) / self.std_in)
        sorties = [tete(h) * self.std_out + self.mean_out for tete in self.heads]
        return torch.stack(sorties, dim=1)          # (B, K, 6)

    def save_model(self, filepath):
        torch.save({
            'state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'head_dim': self.head_dim,
            'n_heads': self.n_heads,
            'output_dim': self.output_dim,
        }, filepath)

    @classmethod
    def from_file(cls, filepath):
        c = torch.load(filepath, map_location='cpu', weights_only=True)
        m = cls(hidden_dim=c.get('hidden_dim', 512),
                head_dim=c.get('head_dim', 256),
                n_heads=c.get('n_heads', 8),
                output_dim=c.get('output_dim', 6))
        m.load_state_dict(c['state_dict'])
        return m


def perte_multi_hypotheses(q_pred, cible, R_ref, fk, r_outil=0.05, eps=0.05):
    """
    Min relache sur les K tetes.

    q_pred : (B, K, 6)  les K propositions
    cible  : (B, 3)     position demandee
    R_ref  : (B, 3, 3)  orientation demandee, obtenue par FK(reference)

    Renvoie (perte, erreurs_par_tete, indice_de_la_gagnante).

    Note : la perte est purement PHYSIQUE. Aucun terme ne compare les angles a
    une reference, donc rien ne dit au reseau quelle tete doit apprendre quelle
    branche. Il decouvre la structure seul -- c'est precisement ce qu'on veut
    montrer.
    """
    B, K, _ = q_pred.shape

    T = fk.forward(q_pred.reshape(B * K, 6))
    pos = T[:, :3, 3].reshape(B, K, 3)
    rot = T[:, :3, :3].reshape(B, K, 3, 3)

    err_pos = ((pos - cible.unsqueeze(1)) ** 2).sum(-1)                  # (B,K)
    err_rot = ((rot - R_ref.unsqueeze(1)) ** 2).sum((-1, -2))            # (B,K)
    err = err_pos + (r_outil ** 2) * err_rot                             # (B,K)

    gagnante = err.argmin(dim=1)                                         # (B,)

    poids = torch.full_like(err, eps / max(K - 1, 1))
    poids[torch.arange(B, device=err.device), gagnante] = 1.0 - eps

    return (poids * err).sum(1).mean(), err, gagnante
