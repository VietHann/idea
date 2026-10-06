"""Evaluate paired source metrics before/after the refiner using one SI-SDR assignment."""
import argparse
import json
import math
import re
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from cm_xbeam.config import load_config
from cm_xbeam.runner import dataset_from_config, move_batch, frame_doa, load_checkpoint
from cm_xbeam.losses.sisdr import pairwise_si_sdr, best_permutation
from cm_xbeam.utils import safe_output_path, resolve_device, seed_everything


def corrupt_cue(cue, args, sample_id, hop_seconds):
    if cue is None:
        if any([args.doa_error, args.doa_bias, args.doa_jitter, args.doa_delay, args.doa_dropout]):
            raise ValueError('DoA corruption requires stored target direction metadata')
        return None
    import hashlib
    stable = int.from_bytes(hashlib.sha256(str(sample_id).encode()).digest()[:8], 'little')
    generator = torch.Generator(device=cue.device).manual_seed((args.seed + stable) % (2**63 - 1))
    if cue.ndim == 2:
        cue = cue.unsqueeze(-1)
    out = cue.clone().float()
    if args.doa_error:
        signs = (torch.randint(0, 2, out.shape[:2] + (1,), generator=generator, device=out.device) * 2 - 1)
        out = out + signs * args.doa_error
    if args.doa_bias:
        out = out + args.doa_bias
    if args.doa_jitter:
        out = out + torch.randn(out.shape, generator=generator, device=out.device) * args.doa_jitter
    if args.doa_delay:
        delay = min(out.size(-1), round(args.doa_delay / hop_seconds))
        out = torch.cat([out[..., :1].expand(*out.shape[:-1], delay), out[..., :out.size(-1) - delay]], dim=-1)
    if args.doa_dropout:
        length = max(1, round(args.doa_dropout * out.size(-1)))
        start = (out.size(-1) - length) // 2
        out[..., start:start + length] = float('nan')
    return torch.remainder(out + 180, 360) - 180


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', help='Optional cross-domain dataset config; model remains checkpoint-defined')
    p.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    p.add_argument('--output', required=True, help='Per-utterance/source JSONL, not independently optimized metrics')
    p.add_argument('--device', default='auto')
    p.add_argument('--stage', choices=list('ABCD'), default=None)
    p.add_argument('--seed', type=int, default=0, help='Seed for deterministic cue corruption')
    p.add_argument('--limit', type=int)
    p.add_argument('--crop-seconds', type=float)
    p.add_argument('--no-perceptual', action='store_true')
    p.add_argument('--doa-error', type=float, default=0)
    p.add_argument('--doa-bias', type=float, default=0)
    p.add_argument('--doa-jitter', type=float, default=0)
    p.add_argument('--doa-delay', type=float, default=0, help='Seconds of tracking delay')
    p.add_argument('--doa-dropout', type=float, default=0, help='Fraction of central frames with missing cue')
    p.add_argument('--use-doa', action='store_true', help='Use directions even for a separation dataset')
    p.add_argument('--diagnostics-dir')
    p.add_argument('--audio-dir')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if not 0 <= args.doa_dropout <= 1 or args.doa_delay < 0:
        raise ValueError('Require 0<=dropout<=1 and nonnegative cue delay')
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, record = load_checkpoint(args.checkpoint, device, args.stage)
    cfg = load_config(args.config) if args.config else record['config']
    cfg['model'] = model.config.to_dict()
    data = dataset_from_config(cfg, args.split, crop_seconds=args.crop_seconds)
    if args.limit:
        data = Subset(data, range(min(args.limit, len(data))))
    from cm_xbeam.data import collate_multichannel
    from cm_xbeam.metrics.separation import evaluate_sources
    loader = DataLoader(data, batch_size=1, collate_fn=collate_multichannel)
    output_path = safe_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = safe_output_path(args.diagnostics_dir) if args.diagnostics_dir else None
    audio_dir = safe_output_path(args.audio_dir) if args.audio_dir else None
    for path in [diagnostics_dir, audio_dir]:
        if path:
            path.mkdir(parents=True, exist_ok=True)
    count = 0
    started = time.perf_counter()
    with output_path.open('w') as handle, torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            c = model.config
            frames = (batch['mixture'].size(-1) + c.hop_length - 1) // c.hop_length
            extraction = cfg['data'].get('task') == 'extraction'
            cue = frame_doa(batch, frames, c.sample_rate, c.hop_length) if extraction or args.use_doa else None
            if cue is not None and cue.ndim == 2:
                cue = cue.unsqueeze(-1).expand(-1, -1, frames)
            identifier = batch['id'][0]
            cue = corrupt_cue(cue, args, identifier, c.hop_length / c.sample_rate)
            result = model(batch['mixture'], doa=cue, lengths=batch['lengths'], extraction=extraction,
                           source_images=batch['source_images'] if diagnostics_dir else None,
                           return_details=True, diagnostics=diagnostics_dir is not None)
            assignment = torch.arange(c.num_sources, device=device)[None] if extraction else best_permutation(
                pairwise_si_sdr(result['waveform'], batch['targets'], batch['lengths']))
            n = int(batch['lengths'][0])
            metadata = batch.get('metadata', [{}])[0]
            metadata = dict(metadata or {})
            metadata.update({'sample_rate': c.sample_rate, 'reference_mic': c.reference_mic,
                             'checkpoint_stage': record.get('stage'), 'inference_stage': model.stage,
                             'evaluation_crop_seconds': args.crop_seconds,
                             'doa_error_deg': args.doa_error, 'doa_bias_deg': args.doa_bias,
                             'doa_jitter_deg': args.doa_jitter, 'doa_delay_seconds': args.doa_delay,
                             'doa_dropout_fraction': args.doa_dropout})
            inverse = torch.argsort(assignment[0])
            # Metric API receives estimates already aligned to target order; all stages share final assignment.
            for label, key in [('pre_refiner', 'pre_waveform'), ('post_refiner', 'waveform')]:
                estimate = result[key][0, inverse, :n].cpu()
                scores = evaluate_sources(estimate, batch['targets'][0, :, :n].cpu(),
                                          batch['mixture'][0, c.reference_mic, :n].cpu(), c.sample_rate,
                                          permutation=np.arange(c.num_sources), perceptual=not args.no_perceptual)
                for source_index, metrics in enumerate(scores['sources']):
                    row = {'id': identifier, 'seed': record['config'].get('train', {}).get('seed', 0),
                           'dataset': cfg.get('dataset', 'unspecified'), 'stage': label, 'source': source_index,
                           'output_slot': int(inverse[source_index]), **metrics, 'metadata': metadata}
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
            safe_id = re.sub(r'[^A-Za-z0-9._-]', '_', str(identifier))
            if diagnostics_dir:
                trace = {k: v[0].cpu().numpy() for k, v in result['diagnostics'].items()}
                if 'covariance_error_pairs' in trace:
                    pair_errors = trace.pop('covariance_error_pairs')
                    trace['covariance_error'] = np.stack([pair_errors[slot, int(assignment[0, slot])] for slot in range(c.num_sources)])
                trace['time_seconds'] = (np.arange(frames) + 1) * c.hop_length / c.sample_rate
                if cue is not None:
                    trace['doa_degrees'] = cue[0].cpu().numpy()
                np.savez_compressed(diagnostics_dir / f'{safe_id}.npz', **trace)
            if audio_dir:
                import soundfile as sf
                for k, wave in enumerate(result['waveform'][0, inverse, :n].cpu().numpy()):
                    sf.write(audio_dir / f'{safe_id}_source{k + 1}.wav', wave, c.sample_rate, subtype='FLOAT')
            count += 1
            if count % 10 == 0:
                print(json.dumps({'utterances_completed': count}), flush=True)
    print(json.dumps({'utterances': count, 'elapsed_seconds': time.perf_counter() - started,
                      'records': str(output_path), 'assignment': 'final SI-SDR; reused for every metric and pre-refiner output'}))


if __name__ == '__main__':
    main()
