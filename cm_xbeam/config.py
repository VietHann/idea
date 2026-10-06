"""Typed model configuration and strict YAML composition."""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import yaml


@dataclass
class ModelConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    num_channels: int = 6
    num_sources: int = 2
    reference_mic: int = 0
    dim: int = 96
    encoder_blocks: int = 3
    encoder_kernel_time: int = 3
    memory_blocks: int = 2
    memory_heads: int = 4
    memory_expansion: float = 2.0
    memory_ff_dim: int = 384
    memory_type: str = 'mlstm'
    lstm_hidden: int = 160
    refiner_blocks: int = 2
    refiner_dim: int = 96
    refiner_heads: int = 4
    refiner_ff_dim: int = 384
    dropout: float = 0.0
    doa_harmonics: int = 4
    include_ipd: bool = True
    lambda_min: float = 0.50
    lambda_max: float = 0.999
    fixed_lambda: float = 0.95
    learned_forgetting: bool = True
    learned_update: bool = True
    scm_mode: str = 'psd'
    eps: float = 1e-8
    load_scale: float = 1e-3
    load_floor: float = 1e-6
    beamformer: str = 'mvdr'
    residual: str = 'nullspace'
    uncertainty: bool = True
    alpha_max: float = 0.25
    fixed_alpha: float = 0.10
    refiner_bound: float = 0.5
    oracle_lambda: float = 0.95
    execution_backend: str = 'chunkwise'
    checkpoint_frames: bool = False
    checkpoint_chunk_size: int = 32

    def __post_init__(self):
        if self.execution_backend not in {'chunkwise', 'framewise'}:
            raise ValueError('execution_backend must be chunkwise or framewise')
        if self.checkpoint_chunk_size < 1:
            raise ValueError('checkpoint_chunk_size must be positive')
        if self.n_fft % 2 or not 0 < self.hop_length <= self.n_fft // 2:
            raise ValueError('Require even n_fft and 0 < hop <= n_fft/2 for Hann overlap-add.')
        if self.num_channels < 2 or not 1 <= self.num_sources <= 4:
            raise ValueError('Require at least 2 microphones and 1–4 output sources.')
        if not 0 <= self.reference_mic < self.num_channels:
            raise ValueError('reference_mic out of range')
        if not 0 <= self.lambda_min < self.lambda_max < 1 or not 0 <= self.fixed_lambda < 1:
            raise ValueError('Invalid forgetting-factor range')
        if self.memory_type not in {'mlstm', 'lstm', 'mlp', 'none'}:
            raise ValueError('Unknown memory_type')
        if self.beamformer not in {'mvdr', 'direct'} or self.residual not in {'none', 'nullspace', 'unconstrained'}:
            raise ValueError('Unknown beamformer/residual')
        if self.scm_mode not in {'psd', 'raw_memory'}:
            raise ValueError('Unknown scm_mode')
        if self.scm_mode == 'raw_memory' and self.memory_type != 'mlstm':
            raise ValueError('Raw-memory ablation requires mLSTM')
        if self.dim % self.memory_heads or self.refiner_dim % self.refiner_heads:
            raise ValueError('Embedding dimensions must be divisible by attention/memory heads')
        if self.scm_mode == 'raw_memory' and int(self.dim * self.memory_expansion) // self.memory_heads < self.num_channels:
            raise ValueError('Native matrix memory too small for raw-SCM ablation')

    def to_dict(self):
        return asdict(self)


def merge_dict(base, override):
    result = copy.deepcopy(base)
    for k, v in override.items():
        result[k] = merge_dict(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else copy.deepcopy(v)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f'Cyclic config inheritance: {path}')
    seen.add(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a mapping')
    parents = cfg.pop('extends', [])
    if isinstance(parents, str):
        parents = [parents]
    merged = {}
    for p in parents:
        merged = merge_dict(merged, load_config(path.parent / p, seen))
    cfg = merge_dict(merged, cfg)
    ModelConfig(**cfg.get('model', {}))
    return cfg
