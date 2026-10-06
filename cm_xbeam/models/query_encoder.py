"""Unified learnable source slots and circular, time-varying DoA queries."""
import math
import torch
from torch import nn


class QueryEncoder(nn.Module):
    def __init__(self, num_sources, dim, harmonics=4):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_sources, dim) * 0.02)
        self.harmonics = harmonics
        self.doa = nn.Sequential(nn.Linear(2 * harmonics, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, batch_size, doa=None):
        # Public cue convention is degrees; NaN means cue dropout => slot fallback.
        slots = self.slots.unsqueeze(0).expand(batch_size, -1, -1)
        if doa is None:
            return self.proj(slots)
        if doa.shape != slots.shape[:2]:
            raise ValueError(f'DoA frame must have shape {slots.shape[:2]}, got {doa.shape}')
        valid = torch.isfinite(doa)
        angle = torch.nan_to_num(doa).float() * (math.pi / 180)
        harmonic = torch.arange(1, self.harmonics + 1, device=angle.device)
        angles = angle.unsqueeze(-1) * harmonic
        circular = torch.cat([angles.sin(), angles.cos()], dim=-1)
        cues = self.doa(circular)
        return self.proj(torch.where(valid.unsqueeze(-1), cues, slots))
