"""Causal depthwise-separable encoder; normalization never mixes frames."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, dim, time_kernel=3, frequency_stride=1):
        super().__init__()
        self.time_kernel = time_kernel
        self.dw = nn.Conv2d(dim, dim, (time_kernel, 3), stride=(1, frequency_stride), groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.stride = frequency_stride

    def sequence(self, x, history=None):
        # Apply the same causal convolution to all frames in a bounded chunk.
        current = x.permute(0, 3, 1, 2)
        if history is None:
            history = current.new_zeros(current.size(0), current.size(1), self.time_kernel - 1, current.size(-1))
        sequence = torch.cat([history, current], dim=2)
        out = self.pw(F.silu(self.dw(F.pad(sequence, (1, 1, 0, 0))))).permute(0, 2, 3, 1)
        out = x[:, :, ::self.stride] + F.silu(self.norm(out))
        out = out + self.ff(self.norm(out))
        updated = sequence[:, :, -(self.time_kernel - 1):] if self.time_kernel > 1 else sequence[:, :, :0]
        return out, updated

    def step(self, x, history=None):
        # x [B,F,D], history [B,D,time_kernel-1,F]
        current = x.transpose(1, 2).unsqueeze(2)
        if history is None:
            history = current.new_zeros(current.size(0), current.size(1), self.time_kernel - 1, current.size(-1))
        sequence = torch.cat([history, current], dim=2)
        out = self.pw(F.silu(self.dw(F.pad(sequence, (1, 1, 0, 0)))))
        out = out.squeeze(2).transpose(1, 2)
        skip = x[:, ::self.stride]
        out = skip + F.silu(self.norm(out))
        out = out + self.ff(self.norm(out))
        return out, sequence[:, :, -(self.time_kernel - 1):] if self.time_kernel > 1 else sequence[:, :, :0]


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, dim=96, blocks=3, time_kernel=3):
        super().__init__()
        self.input = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList([CausalDepthwiseBlock(dim, time_kernel, 2 if i == 0 else 1) for i in range(blocks)])
        if blocks < 1:
            raise ValueError('At least one encoder block is required')

    def sequence(self, features, state=None):
        x = self.input(features)
        state = [None] * len(self.blocks) if state is None else state
        updated = []
        for block, history in zip(self.blocks, state):
            x, history = block.sequence(x, history)
            updated.append(history)
        return x, updated

    def step(self, features, state=None):
        x = self.input(features)
        state = [None] * len(self.blocks) if state is None else state
        updated = []
        for block, history in zip(self.blocks, state):
            x, history = block.step(x, history)
            updated.append(history)
        return x, updated
