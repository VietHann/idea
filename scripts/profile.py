"""Measure actual parameters, hardware-specific RTF, latency and inference memory."""
import argparse
from contextlib import nullcontext
import json
import platform
import time
import numpy as np
import torch
from cm_xbeam.config import load_config, ModelConfig
from cm_xbeam.experiments import apply_variant, VARIANTS
from cm_xbeam.models import CMxBeam
from cm_xbeam.runner import load_checkpoint
from cm_xbeam.complexity import count_parameters
from cm_xbeam.utils import safe_output_path, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--config')
    source.add_argument('--checkpoint')
    p.add_argument('--ablation', choices=VARIANTS)
    p.add_argument('--device', default='auto')
    p.add_argument('--duration', type=float, default=10)
    p.add_argument('--repetitions', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--parameters-only', action='store_true')
    p.add_argument('--flops', action='store_true', help='Run torch.profiler; counts may exclude FFT/complex solvers')
    p.add_argument('--output')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    matching = None
    if args.config:
        cfg, matching = apply_variant(load_config(args.config), args.ablation)
        model = CMxBeam(ModelConfig(**cfg['model'])).to(device).eval()
    else:
        model, _ = load_checkpoint(args.checkpoint, device, 'D')
    c = model.config
    result = {'parameters': count_parameters(model), 'components': {name: count_parameters(module) for name, module in model.named_children()},
              'sample_rate': c.sample_rate, 'n_fft': c.n_fft, 'hop_length': c.hop_length,
              'algorithmic_latency_ms': model.stft.algorithmic_latency_samples / c.sample_rate * 1000,
              'minimum_wola_lookahead_ms': model.stft.left_pad / c.sample_rate * 1000,
              'frame_acquisition_ms': c.hop_length / c.sample_rate * 1000,
              'includes_stft_istft': True, 'device': str(device), 'precision': args.precision,
              'batch_size': 1, 'duration_seconds': args.duration, 'torch_version': torch.__version__,
              'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or platform.machine(),
              'threads': args.threads, 'parameter_matching': matching,
              'execution_backend': c.execution_backend if c.scm_mode == 'psd' else 'framewise',
              'implementation': 'official mLSTM chunk backend; bounded batched affine SCM scan' if c.execution_backend == 'chunkwise' and c.scm_mode == 'psd' else 'official mLSTM recurrent framewise reference'}
    if not args.parameters_only:
        if args.duration <= 0 or args.repetitions < 1 or args.warmup < 0:
            raise ValueError('Require positive duration/repetitions and nonnegative warmup')
        if args.precision == 'bf16' and device.type != 'cuda':
            raise ValueError('BF16 benchmark is currently supported on CUDA only')
        wav = torch.randn(1, c.num_channels, round(args.duration * c.sample_rate), device=device) * 0.05
        timings = []
        def synchronize():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            for _ in range(args.warmup):
                model(wav)
            synchronize()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(args.repetitions):
                synchronize()
                started = time.perf_counter()
                model(wav)
                synchronize()
                timings.append(time.perf_counter() - started)
            if args.flops:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profiler:
                    model(wav)
                measured = sum(x.flops for x in profiler.key_averages())
                result['counted_flops_per_second'] = measured / args.duration
                result['flops_scope'] = 'Partial profiler-supported operators; not total GFLOP/s or MAC/s'
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result.update({'peak_process_rss_mb': rss_mb, 'cpu_memory_scope': 'Peak process RSS, including model and imports',
                       'repetitions': args.repetitions, 'warmup': args.warmup,
                       'median_seconds': float(np.median(timings)), 'p95_seconds': float(np.percentile(timings, 95)),
                       'median_rtf': float(np.median(timings)) / args.duration,
                       'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else None})
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        path = safe_output_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
