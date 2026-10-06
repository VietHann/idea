"""Evaluate a version-pinned upstream baseline via its actual waveform factory."""
import argparse
import json
import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from cm_xbeam.baselines import ExternalBaseline
from cm_xbeam.config import load_config, ModelConfig
from cm_xbeam.data import collate_multichannel
from cm_xbeam.runner import dataset_from_config, move_batch
from cm_xbeam.metrics.separation import evaluate_sources
from cm_xbeam.utils import safe_output_path, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Dataset recipe, not a replacement implementation of the baseline')
    p.add_argument('--baseline-config', required=True, help='YAML with factory, source_directory, expected_commit, parameters, seed, name')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int)
    p.add_argument('--no-perceptual', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    with open(args.baseline_config) as handle:
        baseline = yaml.safe_load(handle)
    sys.path.insert(0, str(Path(baseline['source_directory']).resolve()))
    device = resolve_device(args.device)
    model = ExternalBaseline(baseline['factory'], baseline.get('parameters', {}), baseline['source_directory'], baseline['expected_commit']).to(device).eval()
    c = ModelConfig(**cfg['model'])
    data = dataset_from_config(cfg, 'test')
    if args.limit:
        data = Subset(data, range(min(args.limit, len(data))))
    loader = DataLoader(data, batch_size=1, collate_fn=collate_multichannel)
    output = safe_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with torch.inference_mode(), output.open('w') as handle:
        for batch in loader:
            batch = move_batch(batch, device)
            kwargs = {'doa': batch['doa']} if cfg['data'].get('task') == 'extraction' else {}
            estimate = model(batch['mixture'], **kwargs)['waveform']
            if estimate.shape[:2] != batch['targets'].shape[:2]:
                raise ValueError('Upstream baseline source count does not match the target corpus')
            n = int(batch['lengths'][0])
            result = evaluate_sources(estimate[0, :, :n], batch['targets'][0, :, :n], batch['mixture'][0, c.reference_mic, :n],
                c.sample_rate, permutation=list(range(c.num_sources)) if kwargs else None, perceptual=not args.no_perceptual)
            metadata = {**batch['metadata'][0], 'baseline_name': baseline['name'], 'upstream_commit': model.revision,
                        'sample_rate': c.sample_rate, 'reference_mic': c.reference_mic}
            for slot, scores in enumerate(result['sources']):
                for stage in ['pre_refiner', 'post_refiner']:
                    row = {'id': batch['id'][0], 'seed': baseline['seed'], 'dataset': cfg.get('dataset', 'unspecified'),
                           'stage': stage, 'source': result['permutation'][slot], 'output_slot': slot, **scores, 'metadata': metadata}
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
            count += 1
    print(json.dumps({'utterances': count, 'records': str(output), 'upstream_commit': model.revision}))


if __name__ == '__main__':
    main()
