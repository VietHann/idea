"""Reproducible DoA bias/error/jitter/latency/dropout evaluation sweeps."""
import argparse
import json
import subprocess
import sys
from cm_xbeam.utils import safe_output_path, PROJECT_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--condition', choices=['error', 'bias', 'jitter', 'delay', 'dropout'], default='error')
    p.add_argument('--values', nargs='+', type=float, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--no-perceptual', action='store_true')
    args = p.parse_args()
    defaults = {'error': [0, 5, 10, 20, 30], 'bias': [0, 5, 10, 20, 30], 'jitter': [0, 5, 10, 20, 30],
                'delay': [0, .04, .08, .16, .32], 'dropout': [0, .1, .25, .5]}
    output = safe_output_path(args.output)
    for value in args.values or defaults[args.condition]:
        command = [sys.executable, '-m', 'scripts.evaluate', '--checkpoint', args.checkpoint,
                   '--output', str(output / f'{args.condition}_{value:g}.jsonl'), '--use-doa',
                   '--device', args.device, '--seed', str(args.seed), f'--doa-{args.condition}', str(value)]
        if args.config:
            command += ['--config', args.config]
        if args.limit:
            command += ['--limit', str(args.limit)]
        if args.no_perceptual:
            command += ['--no-perceptual']
        print(json.dumps(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == '__main__':
    main()
