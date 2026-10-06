"""Group measured utterance results by actual motion metadata; no synthetic scores."""
import argparse
from collections import defaultdict
import json
import numpy as np
from cm_xbeam.utils import safe_output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--metric', default='si_sdri')
    p.add_argument('--stage', choices=['pre_refiner', 'post_refiner'], default='post_refiner')
    p.add_argument('--group-by', choices=['motion_scenario', 'angular_velocity', 'jump_degrees', 'angular_separation'], default='motion_scenario')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for path in args.inputs:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if row['stage'] != args.stage:
                    continue
                key = (row['seed'], row.get('dataset'), row['id'], row['source'])
                if key in seen:
                    raise ValueError(f'Duplicate evaluation record {key}')
                seen.add(key)
                group = row.get('metadata', {}).get(args.group_by)
                if group is None:
                    raise ValueError(f'Missing {args.group_by} metadata for {row["id"]}')
                if row.get(args.metric) is not None:
                    groups[str(group)][key[:3]].append(float(row[args.metric]))
    results = []
    for group, utterances in sorted(groups.items()):
        seeds = defaultdict(list)
        for key, scores in utterances.items():
            seeds[key[0]].append(float(np.mean(scores)))
        seed_means = {seed: float(np.mean(values)) for seed, values in seeds.items()}
        results.append({'condition': group, 'seed_means': seed_means, 'mean': float(np.mean(list(seed_means.values()))),
                        'seed_std': float(np.std(list(seed_means.values()), ddof=1)) if len(seed_means) > 1 else None,
                        'utterances': len(utterances), 'metric': args.metric, 'stage': args.stage})
    output = safe_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'conditions': len(results), 'results': str(output)}))


if __name__ == '__main__':
    main()
