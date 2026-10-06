"""Inspect runtime compatibility and a bounded, read-only dataset layout."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import torch
import soundfile as sf
from cm_xbeam.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--root')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--examples', type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config) if args.config else {}
    root = Path(args.root or cfg.get('data', {}).get('root', '.')).resolve()
    result = {'python': platform.python_version(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'dataset_root': str(root), 'dataset_exists': root.is_dir(),
              'optional_dependencies': {name: importlib.util.find_spec(name) is not None for name in ['pesq', 'pystoi', 'pyroomacoustics', 'gpuRIR']}}
    try:
        result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        result['gpu'] = []
    if root.is_dir():
        directories, recordings, containers = [], [], []
        pending = [(root, 0)]
        while pending:
            directory, depth = pending.pop(0)
            children = sorted(directory.iterdir())
            directories.append({'path': str(directory.relative_to(root)) or '.', 'children': [x.name for x in children[:20]],
                                'child_count': len(children)})
            for child in children:
                if child.is_file() and child.suffix == '.hdf5' and len(containers) < args.examples:
                    import h5py
                    with h5py.File(child, 'r') as container:
                        def describe(node, level=0):
                            if isinstance(node, h5py.Dataset):
                                return {'shape': list(node.shape), 'dtype': str(node.dtype)}
                            keys = list(node.keys())
                            return {'count': len(keys), 'examples': {key: describe(node[key], level + 1) for key in keys[:2]}} if level < 3 else {'count': len(keys)}
                        containers.append({'path': str(child.relative_to(root)), 'attributes': {k: str(v) for k,v in container.attrs.items()}, 'structure': describe(container)})
                if child.is_file() and child.suffix.lower() in {'.wav', '.flac'} and len(recordings) < args.examples:
                    info = sf.info(child)
                    recordings.append({'path': str(child.relative_to(root)), 'sample_rate': info.samplerate,
                                       'channels': info.channels, 'frames': info.frames})
            if depth < args.depth:
                pending.extend((child, depth + 1) for child in children if child.is_dir())
            if len(directories) >= 30 and len(recordings) >= args.examples:
                break
        result['layout'], result['recordings'], result['containers'] = directories, recordings, containers
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
