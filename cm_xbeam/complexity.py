"""Find a vector LSTM width matching the matrix controller's actual parameters."""
from dataclasses import replace
from .models.covariance_xlstm import CovarianceController


def count_parameters(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def match_lstm_controller(config, tolerance=0.05):
    reference = count_parameters(CovarianceController(replace(config, memory_type='mlstm', scm_mode='psd')))
    # LSTMCell: 4h(d+h+2); output projection: hd+d. Other layers are fixed.
    template = replace(config, memory_type='lstm', scm_mode='psd', lstm_hidden=1)
    base = count_parameters(CovarianceController(template))
    d, blocks = config.dim, config.memory_blocks
    cell1 = blocks * (4 * (d + 1 + 2) + d + d)
    constant = base - cell1
    candidates = []
    for hidden in range(1, 2049):
        value = constant + blocks * (4 * hidden * (d + hidden + 2) + hidden * d + d)
        candidates.append((abs(value - reference), hidden, value))
    _, hidden, estimate = min(candidates)
    matched = replace(config, memory_type='lstm', scm_mode='psd', lstm_hidden=hidden)
    actual = count_parameters(CovarianceController(matched))
    relative = abs(actual - reference) / reference
    if relative > tolerance:
        raise RuntimeError(f'Unable to match LSTM: relative parameter difference {relative:.3%}')
    return matched, {'mlstm_parameters': reference, 'lstm_parameters': actual,
                     'lstm_hidden': hidden, 'relative_difference': relative}
