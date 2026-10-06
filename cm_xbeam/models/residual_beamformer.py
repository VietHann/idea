"""Neural corrections explicitly constrained to the steering nullspace."""
import torch
from torch import nn


def project_orthogonal(delta, d, eps=1e-8):
    denominator = d.abs().square().sum(-1).clamp_min(eps)
    coefficient = (d.conj() * delta).sum(-1) / denominator
    return delta - d * coefficient[..., None]


class ResidualBeamformer(nn.Module):
    def __init__(self, input_dim, microphones, zero_init=True):
        super().__init__()
        self.microphones = microphones
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, 2 * microphones))
        if zero_init:
            nn.init.normal_(self.net[-1].weight, std=1e-4)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        real, imag = self.net(features).float().split(self.microphones, dim=-1)
        return torch.complex(real, imag)
