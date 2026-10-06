"""Benchmark real forward/backward batches with identical data and model weights."""
import argparse
from dataclasses import replace
import json
import statistics
import time
import torch
from cm_xbeam.config import load_config, ModelConfig
from cm_xbeam.models import CMxBeam
from cm_xbeam.runner import dataset_from_config, move_batch
from cm_xbeam.data import collate_multichannel
from cm_xbeam.losses import SeparationObjective
from cm_xbeam.utils import resolve_device, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--backends', nargs='+', choices=['framewise', 'chunkwise'], default=['framewise', 'chunkwise'])
    p.add_argument('--stage', choices=list('ABCD'), default='A')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--crop-seconds', type=float, default=None)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--first-item', type=int, default=250)
    args = p.parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        raise ValueError('Require nonnegative warmup and positive repetitions')
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    cfg = load_config(args.config)
    c = ModelConfig(**cfg['model'])
    batch_size = args.batch_size or cfg['train']['batch_size']
    duration = args.crop_seconds or cfg['data'].get('crop_seconds')
    device = resolve_device(args.device)
    if args.precision == 'bf16' and device.type != 'cuda':
        raise ValueError('Use --precision fp32 for CPU benchmark')
    dataset = dataset_from_config(cfg, 'train', crop_seconds=duration, training=True, seed=args.seed)
    if args.first_item + batch_size > len(dataset):
        raise ValueError('Requested fixed batch exceeds dataset length')
    start = time.perf_counter()
    batch = move_batch(collate_multichannel([dataset[args.first_item + k] for k in range(batch_size)]), device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    data_seconds = time.perf_counter() - start
    seed_model = CMxBeam(c)
    weights = {key: value.detach().clone() for key, value in seed_model.state_dict().items()}
    del seed_model
    objective = SeparationObjective(hop_length=c.hop_length, **cfg.get('loss', {})).to(device)
    reports = []
    for backend in args.backends:
        model = CMxBeam(replace(c, execution_backend=backend)).to(device)
        model.load_state_dict(weights)
        model.set_stage(args.stage, freeze=False)
        times, forwards, backwards, losses = [], [], [], []
        for i in range(args.warmup + args.repetitions):
            model.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            from scripts.train import forward_loss
            loss, terms = forward_loss(model, objective, batch, c, device, args.stage, args.precision == 'bf16')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            forward = time.perf_counter() - start
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            total = time.perf_counter() - start
            if not torch.isfinite(loss) or not all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite loss/gradient in backend {backend}')
            if i >= args.warmup:
                times.append(total); forwards.append(forward); backwards.append(total - forward); losses.append(float(loss))
            print(json.dumps({'backend': backend, 'iteration': i + 1, 'warmup': i < args.warmup,
                              'forward_s': forward, 'backward_s': total - forward, 'total_s': total}), flush=True)
            del loss, terms
        report = {'backend': backend, 'stage': args.stage, 'parameters': sum(p.numel() for p in model.parameters()),
                  'batch_size': batch_size, 'waveform_samples': batch['mixture'].size(-1),
                  'frames': (batch['mixture'].size(-1) + c.hop_length - 1) // c.hop_length,
                  'precision': args.precision, 'median_forward_s': statistics.median(forwards),
                  'median_backward_s': statistics.median(backwards), 'median_total_s': statistics.median(times),
                  'peak_gpu_mb': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                  'losses': losses, 'data_load_seconds': data_seconds,
                  'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  'torch': torch.__version__, 'includes': 'STFT, model, loss, backward; excludes optimizer/accumulation'}
        reports.append(report)
        print(json.dumps(report), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    dataset.close()
    if len(reports) > 1:
        print(json.dumps({'speedup': reports[0]['median_total_s'] / reports[1]['median_total_s'],
                          'reference_backend': reports[0]['backend'], 'compared_backend': reports[1]['backend']}))


if __name__ == '__main__':
    main()
