"""Training/evaluation shared data and checkpoint interfaces."""
from pathlib import Path
import math
import random
import numpy as np
import torch
from .config import ModelConfig
from .models import CMxBeam
from .utils import PROJECT_ROOT


def project_input(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def dataset_from_config(config, split, crop_seconds=None, training=False, seed=None):
    from .data import MultichannelDataset
    from .data.selection import direction_options
    c = ModelConfig(**config.get('model', {}))
    d = config['data']
    return MultichannelDataset(project_input(d[f'{split}_manifest']), sample_rate=c.sample_rate,
        num_channels=c.num_channels, num_sources=c.num_sources, reference_mic=c.reference_mic,
        crop_seconds=d.get('crop_seconds') if training and crop_seconds is None else crop_seconds,
        training=training, task=d.get('task', 'separation'), augmentation=d.get('augmentation') if training else None,
        seed=config.get('train', {}).get('seed', 0) if seed is None else seed,
        **direction_options(d, split))


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def frame_doa(batch, frames, sample_rate, hop_length):
    cue = batch.get('doa')
    if cue is None:
        return None
    if cue.ndim == 2:
        return cue
    if cue.ndim != 3:
        raise ValueError('DoA must be [B,K] or [B,K,T] in degrees')
    if cue.size(-1) == frames:
        return cue
    if cue.size(-1) == batch['mixture'].size(-1):
        indices = (torch.arange(frames, device=cue.device) + 1) * hop_length - 1
        indices = indices.clamp_max(cue.size(-1) - 1)
        return cue.index_select(-1, indices)
    # Interpolate circular coordinates rather than raw angles across +/-180.
    finite = torch.isfinite(cue)
    radians = torch.nan_to_num(cue).float() * math.pi / 180
    sin = torch.nn.functional.interpolate(radians.sin(), size=frames, mode='linear', align_corners=False)
    cos = torch.nn.functional.interpolate(radians.cos(), size=frames, mode='linear', align_corners=False)
    valid = torch.nn.functional.interpolate(finite.float(), size=frames, mode='nearest') > 0.5
    angles = torch.atan2(sin, cos) * 180 / math.pi
    return torch.where(valid, angles, torch.full_like(angles, float('nan')))


def load_checkpoint(path, device='cpu', inference_stage='D'):
    record = torch.load(project_input(path), map_location=device, weights_only=False)
    config = ModelConfig(**record['config']['model'])
    config.execution_backend = record.get('execution_backend', config.execution_backend)
    if config.execution_backend not in {'chunkwise', 'framewise'}:
        raise ValueError('Checkpoint specifies an unknown execution backend')
    model = CMxBeam(config)
    model.load_state_dict(record['model'], strict=True)
    model.to(device).eval()
    model.set_stage(record.get('stage', inference_stage) if inference_stage is None else inference_stage, freeze=False)
    return model, record


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
