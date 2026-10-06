"""Research variants, preserving explicit distinctions in the supplied design."""
from copy import deepcopy
from .complexity import match_lstm_controller
from .config import ModelConfig

VARIANTS = {
    'A0': dict(memory_type='none', learned_forgetting=False, learned_update=False, residual='none', uncertainty=False, refiner_blocks=0),
    'A1': dict(memory_type='mlp', residual='none', uncertainty=False, refiner_blocks=0),
    'A2': dict(memory_type='lstm', residual='none', uncertainty=False, refiner_blocks=0),
    'A3': dict(memory_type='mlstm', residual='none', uncertainty=False, refiner_blocks=0),
    'A4': dict(memory_type='mlstm', beamformer='direct', residual='none', uncertainty=False, refiner_blocks=0),
    'A5': dict(memory_type='mlstm', residual='unconstrained', uncertainty=False, refiner_blocks=0),
    'A6': dict(memory_type='mlstm', residual='nullspace', uncertainty=False, refiner_blocks=0),
    'A7': dict(memory_type='mlstm', residual='nullspace', uncertainty=True, refiner_blocks=0),
    'A8': dict(memory_type='mlstm', residual='nullspace', uncertainty=True),
    'A9': dict(memory_type='mlstm', scm_mode='raw_memory', residual='none', uncertainty=False, refiner_blocks=0),
    'A10': dict(memory_type='lstm', residual='nullspace', uncertainty=True),
    'B00': dict(memory_type='lstm', learned_forgetting=False, learned_update=True, residual='none', uncertainty=False, refiner_blocks=0),
    'B01': dict(memory_type='lstm', learned_forgetting=True, learned_update=True, residual='none', uncertainty=False, refiner_blocks=0),
    'B10': dict(memory_type='mlstm', learned_forgetting=False, learned_update=True, residual='none', uncertainty=False, refiner_blocks=0),
    'B11': dict(memory_type='mlstm', learned_forgetting=True, learned_update=True, residual='none', uncertainty=False, refiner_blocks=0),
}


def apply_variant(config, variant=None):
    cfg = deepcopy(config)
    receipt = None
    if variant:
        if variant not in VARIANTS:
            raise ValueError(f'Unknown ablation {variant}; choose {list(VARIANTS)}')
        cfg.setdefault('model', {}).update(VARIANTS[variant])
        if variant in {'A10', 'B00', 'B01'}:
            matched, receipt = match_lstm_controller(ModelConfig(**cfg['model']))
            cfg['model'] = matched.to_dict()
        cfg['variant'] = variant
    return cfg, receipt
