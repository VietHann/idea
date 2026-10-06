"""Covariance-aligned xLSTM beamforming."""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True

# Keep lazy extension/JIT caches local as well as explicit application outputs.
_root = Path(__file__).resolve().parent.parent
for _name, _subdir in [('TORCH_EXTENSIONS_DIR', 'torch_extensions'),
                      ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
                      ('CUDA_CACHE_PATH', 'cuda')]:
    os.environ[_name] = str(_root / '.cache' / _subdir)

# An optional project-local installation preserves the user's global environment.
_vendor = Path(__file__).resolve().parent.parent / '.vendor'
if _vendor.is_dir():
    sys.path.insert(0, str(_vendor))

__version__ = '0.1.0'
