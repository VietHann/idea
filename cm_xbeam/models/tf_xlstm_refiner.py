"""Small TF-mLSTM spectral refiner: bidirectional frequency, causal time."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .covariance_xlstm import TemporalMemory, official_block


class TFBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.freq_forward = official_block(dim, heads, 2.0)
        self.freq_backward = official_block(dim, heads, 2.0)
        self.mix = nn.Linear(2 * dim, dim)
        self.time = TemporalMemory(dim, 1, heads, 2.0, ff_dim)
        self.norm = nn.LayerNorm(dim)

    def sequence(self, x, state=None):
        b, k, t, f, d = x.shape
        freq = x.reshape(b * k * t, f, d)
        forward = self.freq_forward(freq)
        backward = self.freq_backward(freq.flip(1)).flip(1)
        x = x + self.mix(torch.cat([forward, backward], dim=-1)).reshape(b, k, t, f, d)
        temporal = self.norm(x).permute(0, 1, 3, 2, 4).reshape(b * k * f, t, d)
        temporal, state = self.time.sequence(temporal, state)
        return temporal.reshape(b, k, f, t, d).permute(0, 1, 3, 2, 4), state

    def step(self, x, state=None):
        b, k, f, d = x.shape
        freq = x.reshape(b * k, f, d)
        forward = self.freq_forward(freq)
        backward = self.freq_backward(freq.flip(1)).flip(1)
        x = x + self.mix(torch.cat([forward, backward], dim=-1)).reshape(b, k, f, d)
        time, state = self.time.step(self.norm(x).reshape(b * k * f, d), state)
        return time.reshape(b, k, f, d), state


class TFXLSTMRefiner(nn.Module):
    def __init__(self, blocks=2, dim=96, heads=4, ff_dim=384, bound=0.5, checkpoint_blocks=False):
        super().__init__()
        self.bound = bound
        self.checkpoint_blocks = checkpoint_blocks
        self.input = nn.Linear(3, dim)
        self.blocks = nn.ModuleList([TFBlock(dim, heads, ff_dim) for _ in range(blocks)])
        self.output = nn.Linear(dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def sequence(self, z, state=None):
        features = torch.stack([z.real, z.imag, torch.log(z.abs() + 1e-8)], dim=-1)
        x = self.input(features)
        state = [None] * len(self.blocks) if state is None else state
        updated = []
        for block, previous in zip(self.blocks, state):
            # A single frequency block can retain large F x F attention tensors.
            # Nested block checkpointing bounds the full stage-D recompute peak
            # without changing the number of layers, frequencies or sources.
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                x, previous = checkpoint(block.sequence, x, previous, use_reentrant=False)
            else:
                x, previous = block.sequence(x, previous)
            updated.append(previous)
        ratio = self.bound * self.output(x).float().tanh()
        correction = torch.complex(ratio[..., 0], ratio[..., 1])
        return z * (1 + correction), updated

    def step(self, z, state=None):
        features = torch.stack([z.real, z.imag, torch.log(z.abs() + 1e-8)], dim=-1)
        x = self.input(features)
        state = [None] * len(self.blocks) if state is None else state
        updated = []
        for block, previous in zip(self.blocks, state):
            x, previous = block.step(x, previous)
            updated.append(previous)
        ratio = self.bound * self.output(x).float().tanh()
        correction = torch.complex(ratio[..., 0], ratio[..., 1])
        return z * (1 + correction), updated
