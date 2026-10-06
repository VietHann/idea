"""Paired confidence intervals and Holm-corrected tests for predeclared comparisons."""
import argparse
import json
from cm_xbeam.metrics.statistics import read_records, paired_matrix, hierarchical_bootstrap, paired_bootstrap, sign_flip_test, holm_bonferroni
from cm_xbeam.utils import safe_output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair', nargs=3, action='append', required=True, metavar=('NAME', 'SYSTEM_A', 'SYSTEM_B'),
                   help='Files may be comma-separated to include all seeds; repeat for H1/H2/H3')
    p.add_argument('--metric', default='si_sdri')
    p.add_argument('--stage', choices=['pre_refiner', 'post_refiner'], default='post_refiner')
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    results = []
    for name, a, b in args.pair:
        matrix, seeds, utterances = paired_matrix(read_records(a.split(','), args.stage, args.metric), read_records(b.split(','), args.stage, args.metric))
        averaged = matrix.mean(0)
        results.append({'comparison': name, 'metric': args.metric, 'stage': args.stage, 'seed_ids': seeds,
                        'hierarchical_bootstrap': hierarchical_bootstrap(matrix, args.resamples, args.seed),
                        'utterance_bootstrap': paired_bootstrap(averaged, args.resamples, args.seed),
                        'sign_flip': sign_flip_test(averaged, args.resamples, args.seed)})
    corrected = holm_bonferroni([x['sign_flip']['p_value'] for x in results])
    for item, value in zip(results, corrected):
        item['holm_corrected_p'] = value
    path = safe_output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'comparisons': len(results), 'results': str(path)}))


if __name__ == '__main__':
    main()
