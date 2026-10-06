"""Reproducibility and project-contained output paths."""
from pathlib import Path
import random
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def safe_output_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f'Output must stay inside {PROJECT_ROOT}: {resolved}')
    return resolved


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def detach_tree(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach()
    if isinstance(tree, dict):
        return {k: detach_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [detach_tree(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(detach_tree(v) for v in tree)
    return tree


def resolve_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu or a project-local PyTorch build compatible with the installed driver.')
    return torch.device(name)
