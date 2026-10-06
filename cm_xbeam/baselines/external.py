"""Version-checked adapters for upstream research baselines, without reimplementation."""
import importlib
import subprocess
from pathlib import Path
import torch
from torch import nn


class ExternalBaseline(nn.Module):
    """A factory supplies the actual upstream model, checkpoint and input convention.

    Factory contract: factory(config: dict) -> nn.Module; forward receives waveform
    [B,M,N] and returns [B,K,N], or a dict with 'waveform' and optional 'pre_waveform'.
    The wrapper checks the upstream Git SHA so reported baselines remain auditable.
    """
    def __init__(self, factory, config, source_directory, expected_commit):
        super().__init__()
        directory = Path(source_directory).resolve()
        if not expected_commit or len(expected_commit) != 40:
            raise ValueError('An exact upstream 40-character Git commit is required')
        actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected_commit:
            raise ValueError(f'Upstream revision mismatch: expected {expected_commit}, found {actual}')
        if subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Upstream baseline checkout has uncommitted changes')
        module, symbol = factory.split(':', 1)
        self.model = getattr(importlib.import_module(module), symbol)(config)
        if not isinstance(self.model, nn.Module):
            raise TypeError('Baseline factory must return a torch.nn.Module')
        self.revision = actual

    def forward(self, waveform, **kwargs):
        output = self.model(waveform, **kwargs)
        result = {'waveform': output, 'pre_waveform': output} if isinstance(output, torch.Tensor) else output
        if result['waveform'].ndim != 3 or result['waveform'].size(-1) != waveform.size(-1):
            raise ValueError('External baseline must return [B,K,N], preserving input length')
        if not result['waveform'].isfinite().all():
            raise FloatingPointError('External baseline returned nonfinite waveform')
        return result
