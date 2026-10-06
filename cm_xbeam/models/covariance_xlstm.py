"""Official NX-AI mLSTM wrapper and explicit vector-memory ablation controls."""
import torch
from torch import nn
from torch.nn import functional as F


def official_block(dim, heads, expansion, context=512, blocks=1):
    try:
        from xlstm.blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
        from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
    except ImportError as exc:
        raise ImportError('Install the official xlstm==2.0.5 (project-local .vendor is supported). No substitute recurrent backbone is used.') from exc
    cfg = mLSTMLayerConfig(embedding_dim=dim, num_heads=heads, proj_factor=expansion,
                          context_length=context, conv1d_kernel_size=4, dropout=0.0)
    return mLSTMBlock(mLSTMBlockConfig(mlstm=cfg, _num_blocks=blocks, _block_idx=0))


class TemporalMemory(nn.Module):
    def __init__(self, dim, blocks=2, heads=4, expansion=2.0, ff_dim=384, kind='mlstm', lstm_hidden=160, dropout=0.0):
        super().__init__()
        self.kind = kind
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(blocks)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim)) for _ in range(blocks)])
        self.dropout = nn.Dropout(dropout)
        if kind == 'mlstm':
            self.blocks = nn.ModuleList([official_block(dim, heads, expansion, blocks=blocks) for _ in range(blocks)])
        elif kind == 'lstm':
            self.blocks = nn.ModuleList([nn.LSTMCell(dim, lstm_hidden) for _ in range(blocks)])
            self.output = nn.ModuleList([nn.Linear(lstm_hidden, dim) for _ in range(blocks)])
        elif kind == 'mlp':
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)) for _ in range(blocks)])
        elif kind == 'none':
            self.blocks = nn.ModuleList()
        else:
            raise ValueError(kind)

    def sequence(self, x, state=None):
        from .mlstm_sequence import mlstm_block_sequence
        state = [None] * len(self.blocks) if state is None else state
        updated = []
        for i, block in enumerate(self.blocks):
            if self.kind == 'mlstm':
                x, current = mlstm_block_sequence(block, x, state[i])
            elif self.kind == 'lstm':
                hidden_state = state[i]
                outputs = []
                normalized = self.norms[i](x)
                for t in range(x.size(1)):
                    hidden_state = block(normalized[:, t], hidden_state)
                    outputs.append(hidden_state[0])
                current = hidden_state
                x = x + self.output[i](torch.stack(outputs, dim=1))
            else:
                current = None
                x = x + block(self.norms[i](x))
            x = x + self.dropout(self.ff[i](self.norms[i](x)))
            updated.append(current)
        return x, updated

    def step(self, x, state=None):
        state = [None] * len(self.blocks) if state is None else state
        updated = []
        for i, block in enumerate(self.blocks):
            if self.kind == 'mlstm':
                previous = dict(state[i] or {})
                # Official inference step rolls its convolution cache in-place.
                # Clone the cache so BPTT/checkpoint recomputation cannot mutate
                # a tensor retained by an earlier frame's autograd graph.
                if previous.get('conv_state') is not None:
                    previous['conv_state'] = tuple(v.clone() for v in previous['conv_state'])
                y, s = block.step(x.unsqueeze(1), **previous)
                x = y.squeeze(1)
            elif self.kind == 'lstm':
                s = block(self.norms[i](x), state[i])
                x = x + self.output[i](s[0])
            else:
                s = None
                x = x + block(self.norms[i](x))
            x = x + self.dropout(self.ff[i](self.norms[i](x)))
            updated.append(s)
        return x, updated


class CovarianceController(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stream_type = nn.Parameter(torch.randn(2, cfg.dim) * 0.02)
        self.condition = nn.Linear(1, cfg.dim)
        self.memory = TemporalMemory(cfg.dim, cfg.memory_blocks, cfg.memory_heads, cfg.memory_expansion,
                                     cfg.memory_ff_dim, cfg.memory_type, cfg.lstm_hidden, cfg.dropout)
        self.gates = nn.ModuleList([nn.Linear(cfg.dim, 2) for _ in range(2)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
            # Start from the fixed-EMA operating point, not random short memory.
            ratio = (cfg.fixed_lambda - cfg.lambda_min) / (cfg.lambda_max - cfg.lambda_min)
            ratio = max(1e-4, min(1 - 1e-4, ratio))
            with torch.no_grad():
                gate.bias[0] = torch.logit(torch.tensor(ratio))

    def sequence(self, h, query, masks_low, state=None):
        b, k, t, f = masks_low.shape
        tokens = h[:, None] + query[:, :, :, None]
        probabilities = torch.stack([masks_low, 1 - masks_low], dim=2)
        tokens = tokens[:, :, None] + self.condition(probabilities.unsqueeze(-1)) + self.stream_type[None, None, :, None, None]
        ordered = tokens.permute(0, 1, 2, 4, 3, 5).reshape(b * k * 2 * f, t, -1)
        out, state = self.memory.sequence(ordered, state)
        out = out.reshape(b, k, 2, f, t, -1).permute(0, 1, 2, 4, 3, 5)
        gates = torch.stack([self.gates[j](out[:, :, j]) for j in range(2)], dim=2)
        forgetting = self.cfg.lambda_min + (self.cfg.lambda_max - self.cfg.lambda_min) * gates[..., 0].sigmoid()
        update = F.softplus(gates[..., 1])
        if not self.cfg.learned_forgetting:
            forgetting = torch.full_like(forgetting, self.cfg.fixed_lambda)
        if not self.cfg.learned_update:
            update = torch.ones_like(update)
        return out, forgetting, update, state

    def step(self, h, query, masks_low, state=None):
        b, k, f = masks_low.shape
        tokens = h[:, None] + query[:, :, None]
        probabilities = torch.stack([masks_low, 1 - masks_low], dim=2)
        tokens = tokens[:, :, None] + self.condition(probabilities.unsqueeze(-1)) + self.stream_type[None, None, :, None]
        tokens = tokens.reshape(b * k * 2 * f, -1)
        out, state = self.memory.step(tokens, state)
        out = out.reshape(b, k, 2, f, -1)
        gates = torch.stack([self.gates[j](out[:, :, j]) for j in range(2)], dim=2)
        forgetting = self.cfg.lambda_min + (self.cfg.lambda_max - self.cfg.lambda_min) * gates[..., 0].sigmoid()
        update = F.softplus(gates[..., 1])
        if not self.cfg.learned_forgetting:
            forgetting = torch.full_like(forgetting, self.cfg.fixed_lambda)
        if not self.cfg.learned_update:
            update = torch.ones_like(update)
        raw = None
        if self.cfg.scm_mode == 'raw_memory':
            c = state[-1]['mlstm_state'][0][:, 0, :self.cfg.num_channels, :self.cfg.num_channels]
            raw = c.reshape(b, k, 2, f, self.cfg.num_channels, self.cfg.num_channels).to(torch.complex64)
        return out, forgetting, update, state, raw
