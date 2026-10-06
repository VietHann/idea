"""Aggregate seed means without selecting the best seed or treating sources as utterances."""
import argparse
import json
import numpy as np
from cm_xbeam.metrics.statistics import read_records
from cm_xbeam.utils import safe_output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--metric', default='si_sdri')
    p.add_argument('--stage', choices=['pre_refiner', 'post_refiner'], default='post_refiner')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    records = read_records(args.inputs, args.stage, args.metric)
    keys = set(next(iter(records.values())))
    if any(set(values) != keys for values in records.values()):
        raise ValueError('All seeds must be evaluated on exactly the same utterance/source keys')
    means = {}
    for seed, values in records.items():
        utterances = {}
        for key, value in values.items():
            if value is not None:
                utterances.setdefault(key[:2], []).append(value)
        means[seed] = float(np.mean([np.mean(x) for x in utterances.values()])) if utterances else None
    if any(value is None for value in means.values()):
        raise ValueError('At least one seed has no valid metrics')
    numbers = list(means.values())
    result = {'metric': args.metric, 'stage': args.stage, 'seed_means': means, 'mean': float(np.mean(numbers)),
              'seed_std': float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
              'seeds': len(numbers), 'source_records_per_seed': len(keys)}
    path = safe_output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'results': str(path), 'seeds': len(numbers)}))


if __name__ == '__main__':
    main()
