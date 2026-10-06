"""Generate actual reverberant motion mixtures inside the project from frozen source splits."""
import argparse
import json
from pathlib import Path
import soundfile as sf
from cm_xbeam.config import load_config
from cm_xbeam.data.rir_generation import simulate_example, SCENARIOS
from cm_xbeam.runner import project_input
from cm_xbeam.utils import safe_output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--input-manifest')
    p.add_argument('--split', choices=['train', 'valid', 'test'], required=True)
    p.add_argument('--output', required=True, help='All mixtures/manifests are written only below this project-contained directory')
    p.add_argument('--num-examples', type=int, default=1000)
    p.add_argument('--duration', type=float, default=2.0)
    p.add_argument('--scenarios', choices=SCENARIOS, nargs='+', default=SCENARIOS)
    p.add_argument('--velocities', type=float, nargs='+', default=[5, 15, 30])
    p.add_argument('--jumps', type=float, nargs='+', default=[30, 60, 90])
    p.add_argument('--separations', type=float, nargs='+', default=[10, 20, 40])
    p.add_argument('--rir-interval', type=float, default=.1)
    p.add_argument('--rt60', type=float, nargs=2, default=[.1, .8])
    p.add_argument('--snr', type=float, nargs=2, default=[-5, 20])
    p.add_argument('--microphone-jitter', type=float, default=0, help='Meters')
    p.add_argument('--max-order', type=int, help='Explicit image-source truncation; omitted uses inverse-Sabine recommendation')
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    if args.num_examples < 1 or args.duration <= 0 or args.rir_interval <= 0:
        raise ValueError('Examples, duration and RIR interval must be positive')
    if args.rt60[0] <= 0 or args.rt60[1] < args.rt60[0]:
        raise ValueError('Invalid reverberation interval')
    cfg = load_config(args.config)
    if cfg['model'].get('num_sources', 2) != 2:
        raise ValueError('This motion protocol uses two physical speakers')
    source = project_input(args.input_manifest or cfg['data'][f'{args.split}_manifest'])
    output = safe_output_path(args.output)
    manifest = safe_output_path(output / f'{args.split}.jsonl')
    if manifest.exists():
        raise FileExistsError(f'Frozen generation manifest already exists: {manifest}; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    audio = safe_output_path(output / args.split)
    audio.mkdir(parents=True, exist_ok=True)
    c = cfg['model']
    rate, microphones = c.get('sample_rate', 16000), c.get('num_channels', 3)
    config_record = vars(args).copy()
    config_record.update({'sample_rate': rate, 'microphones': microphones, 'source_manifest': str(source)})
    safe_output_path(output / f'{args.split}_generation.json').write_text(json.dumps(config_record, indent=2) + '\n')
    count = 0
    with source.open() as source_file, manifest.open('w') as destination:
        for line in source_file:
            if count >= args.num_examples:
                break
            row = json.loads(line)
            if row['split'] != args.split:
                raise ValueError('Source corpus split differs from requested motion split')
            scenario = args.scenarios[count % len(args.scenarios)]
            condition_index = count // len(args.scenarios)
            mixture, images, doa, metadata = simulate_example(row, rate, microphones, args.duration, scenario,
                args.velocities[condition_index % len(args.velocities)], args.jumps[condition_index % len(args.jumps)],
                args.separations[condition_index % len(args.separations)], args.seed, args.rir_interval,
                tuple(args.rt60), tuple(args.snr), args.microphone_jitter, args.max_order)
            identifier = f'{args.split}_{scenario}_{count:06d}'
            mixture_path = safe_output_path(audio / f'{identifier}_mixture.wav')
            sources = [safe_output_path(audio / f'{identifier}_source{k + 1}.wav') for k in range(2)]
            sf.write(mixture_path, mixture.T, rate, subtype='FLOAT')
            for k, path in enumerate(sources):
                sf.write(path, images[k].T, rate, subtype='FLOAT')
            record = {'id': identifier, 'split': args.split, 'sample_rate': rate, 'num_samples': mixture.shape[-1],
                      'target_kind': 'reverberant', 'target_index': 0, 'mixture': str(mixture_path),
                      'sources': [str(path) for path in sources], 'doa': doa.tolist(), 'doa_sample_rate': 100,
                      'metadata': metadata}
            destination.write(json.dumps(record, allow_nan=False) + '\n')
            destination.flush()
            count += 1
            print(json.dumps({'generated': count, 'id': identifier, 'scenario': scenario}), flush=True)
    if count != args.num_examples:
        raise ValueError(f'Only {count} source records were available; partial manifest retained for inspection')


if __name__ == '__main__':
    main()
