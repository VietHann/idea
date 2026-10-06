"""Nonredundant Hermitian features: M^2 real coordinates plus energy/IPD."""
import torch
from torch import nn


class SpatialFeatures(nn.Module):
    def __init__(self, microphones, reference_mic=0, include_ipd=True, eps=1e-8):
        super().__init__()
        self.microphones, self.reference_mic, self.include_ipd, self.eps = microphones, reference_mic, include_ipd, eps
        indices = torch.triu_indices(microphones, microphones, offset=1)
        self.register_buffer('rows', indices[0], persistent=False)
        self.register_buffer('cols', indices[1], persistent=False)
        self.other = [m for m in range(microphones) if m != reference_mic]
        self.output_dim = microphones ** 2 + 2 + (2 * (microphones - 1) if include_ipd else 0)

    def forward(self, y):
        power = y.abs().square()
        energy = power.sum(-1, keepdim=True)
        g = y.unsqueeze(-1) * y.conj().unsqueeze(-2) / (energy.unsqueeze(-1) + self.eps)
        parts = [g.diagonal(dim1=-2, dim2=-1).real, g[..., self.rows, self.cols].real,
                 g[..., self.rows, self.cols].imag,
                 torch.log(power[..., self.reference_mic:self.reference_mic + 1] + self.eps),
                 torch.log1p(energy / self.microphones)]
        if self.include_ipd:
            cross = y[..., self.other] * y[..., self.reference_mic:self.reference_mic + 1].conj()
            unit = cross / cross.abs().clamp_min(self.eps)
            parts.extend([unit.real, unit.imag])
        return torch.cat(parts, dim=-1).float()
