"""Batched layer execution using NX-AI's official chunkwise mLSTM backend.

No parameters or recurrence equations are replaced. State scaling translates
between the official cell's key-scaled state and the chunk backend's query-scaled
state. Accumulation is FP32, while linear/convolution projections may use BF16.
"""
import math
import torch
from torch.nn import functional as F


def mlstm_block_sequence(block, x, previous=None):
    from xlstm.blocks.mlstm.backends import chunkwise_simple
    layer = block.xlstm
    previous = previous or {}
    normalized = block.xlstm_norm(x)
    projected = layer.proj_up(normalized)
    hidden, z = projected.chunk(2, dim=-1)
    width = hidden.size(-1)
    conv = layer.conv1d
    kernel = conv.config.kernel_size
    if kernel:
        # The recurrent cache contains KS positions, including the last input;
        # a whole-chunk convolution needs exactly KS-1 historical positions.
        old = previous.get('conv_state')
        history = hidden.new_zeros(hidden.size(0), kernel - 1, width) if old is None or kernel == 1 else old[0][:, -(kernel - 1):].to(hidden.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            joined = torch.cat([history.float(), hidden.float()], dim=1)
            filtered = F.conv1d(joined.transpose(1, 2), conv.conv.weight.float(),
                                None if conv.conv.bias is None else conv.conv.bias.float(),
                                groups=conv.groups).transpose(1, 2)
            cache_input = torch.cat([hidden.new_zeros(hidden.size(0), kernel, width).float(), hidden.float()], dim=1) if old is None else torch.cat([old[0].float(), hidden.float()], dim=1)
            conv_state = (cache_input[:, -kernel:],)
    else:
        filtered, conv_state = hidden, previous.get('conv_state')
    activated = layer.conv_act_fn(filtered)
    q, k, v = layer.q_proj(activated), layer.k_proj(activated), layer.v_proj(hidden)
    cell = layer.mlstm_cell
    heads = cell.config.num_heads
    b, t, dimension = q.shape
    head_dim = dimension // heads
    gate_input = torch.cat([q, k, v], dim=-1)
    input_gate = cell.igate(gate_input).transpose(1, 2)
    forget_gate = cell.fgate(gate_input).transpose(1, 2)
    queries = q.reshape(b, t, heads, head_dim).transpose(1, 2).contiguous()
    keys = k.reshape(b, t, heads, head_dim).transpose(1, 2).contiguous()
    values = v.reshape(b, t, heads, head_dim).transpose(1, 2).contiguous()
    native = previous.get('mlstm_state')
    scale = math.sqrt(head_dim)
    initial_c = None if native is None else native[0].float() * scale
    initial_n = None if native is None else native[1].squeeze(-1).float() * scale
    # The backend accepts [B,NH] here despite its docstring's extra axis.
    initial_m = None if native is None else native[2].squeeze(-1).squeeze(-1).float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        retrieved, (c, n, m) = chunkwise_simple(
            queries.float(), keys.float(), values.float(), input_gate.float(), forget_gate.float(),
            initial_C=initial_c, initial_n=initial_n, initial_m=initial_m,
            chunk_size=t, return_last_state=True)
    retrieved = retrieved.reshape(b, heads, t, head_dim)
    retrieved = cell.outnorm(retrieved).transpose(1, 2).reshape(b, t, dimension)
    h = (retrieved + layer.learnable_skip * activated) * layer.ogate_act_fn(z)
    out = x + layer.dropout(layer.proj_down(h))
    if block.ffn is not None:
        out = out + block.ffn(block.ffn_norm(out))
    state = {'mlstm_state': (c / scale, n.unsqueeze(-1) / scale, m), 'conv_state': conv_state}
    return out, state
