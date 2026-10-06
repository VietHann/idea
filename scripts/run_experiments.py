"""Launch explicit seed/ablation recipes; no background training is implicit."""
import argparse
import json
import subprocess
import sys
from cm_xbeam.experiments import VARIANTS
from cm_xbeam.utils import safe_output_path, PROJECT_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variants', nargs='+', default=['A0', 'A3', 'A8', 'A10'], choices=VARIANTS)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--device', default='auto')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-steps', type=int)
    args = p.parse_args()
    output = safe_output_path(args.output)
    for variant in args.variants:
        for seed in args.seeds:
            command = [sys.executable, '-m', 'scripts.train', '--config', args.config,
                       '--output', str(output / variant / f'seed_{seed}'), '--seed', str(seed),
                       '--ablation', variant, '--device', args.device]
            if args.max_steps:
                command += ['--max-steps', str(args.max_steps)]
            print(json.dumps(command), flush=True)
            if args.execute:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == '__main__':
    main()
