"""Separate a multichannel recording, optionally using persistent waveform streaming."""
import argparse
import json
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from cm_xbeam.runner import load_checkpoint
from cm_xbeam.streaming import StreamingSession
from cm_xbeam.utils import safe_output_path, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True, nargs='+', help='One multichannel WAV or one mono WAV per microphone, in array order')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--stage', choices=list('ABCD'), default=None)
    p.add_argument('--doa', type=float, nargs='+', help='Static angles in degrees, one per output query')
    p.add_argument('--chunk-samples', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, record = load_checkpoint(args.checkpoint, device, args.stage)
    arrays, rates = zip(*(sf.read(path, dtype='float32', always_2d=True) for path in args.input))
    if len(set(rates)) != 1 or len({len(a) for a in arrays}) != 1:
        raise ValueError('Input microphone files must have identical length/sample rate')
    audio = np.concatenate(arrays, axis=1)
    if audio.shape[1] != model.config.num_channels:
        raise ValueError(f'Expected {model.config.num_channels} channels, got {audio.shape[1]}')
    sr = rates[0]
    if sr != model.config.sample_rate:
        common = gcd(sr, model.config.sample_rate)
        audio = resample_poly(audio, model.config.sample_rate // common, sr // common, axis=0).astype(np.float32)
    waveform = torch.from_numpy(audio.T.copy())[None].to(device)
    doa = None if args.doa is None else torch.tensor([args.doa], device=device)
    if doa is not None and doa.size(1) != model.config.num_sources:
        raise ValueError('One DoA angle is required per output source')
    with torch.inference_mode():
        if args.chunk_samples:
            if args.chunk_samples < 1:
                raise ValueError('chunk-samples must be nonnegative')
            session = StreamingSession(model, doa=doa, extraction=doa is not None)
            parts = [session.push(waveform[..., i:i + args.chunk_samples]) for i in range(0, waveform.size(-1), args.chunk_samples)]
            parts.append(session.flush())
            separated = torch.cat(parts, dim=-1)
        else:
            separated = model(waveform, doa, extraction=doa is not None)
    output = safe_output_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for k, estimate in enumerate(separated[0].cpu().numpy()):
        sf.write(output / f'source_{k + 1}.wav', estimate, model.config.sample_rate, subtype='FLOAT')
    print(json.dumps({'output': str(output), 'sources': model.config.num_sources,
                      'sample_rate': model.config.sample_rate, 'checkpoint_stage': record.get('stage'),
                      'inference_stage': model.stage, 'streaming': bool(args.chunk_samples)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
