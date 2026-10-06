"""Train curriculum A–D with gradient accumulation and resumable checkpoints."""
import argparse
from contextlib import nullcontext
import json
import math
import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel
from cm_xbeam.config import load_config, ModelConfig
from cm_xbeam.models import CMxBeam
from cm_xbeam.losses import SeparationObjective
from cm_xbeam.experiments import apply_variant, VARIANTS
from cm_xbeam.runner import dataset_from_config, move_batch, frame_doa, rng_state, restore_rng
from cm_xbeam.utils import safe_output_path, seed_everything, resolve_device
from cm_xbeam.progress import TrainingProgress
from cm_xbeam.data.selection import direction_options
from cm_xbeam.metrics.training import METRIC_SUMS, mean_si_sdr_metrics


def metric_label(value):
    return 'n/a' if value is None else f'{value:.3f}'


def format_training_terms(terms):
    raw = set(METRIC_SUMS) | {'metric_valid_sources', 'metric_excluded_sources'}
    return {**{key: value for key, value in terms.items() if key not in raw},
            **mean_si_sdr_metrics(terms),
            'metric_valid_sources': float(terms.get('metric_valid_sources', 0)),
            'metric_excluded_sources': float(terms.get('metric_excluded_sources', 0))}


def stage_at(step, curriculum):
    boundary = 0
    for stage, steps in curriculum.items():
        boundary += int(steps)
        if step < boundary:
            return stage
    return list(curriculum)[-1]


def schedule(step, total, warmup):
    warmup = min(warmup, max(1, total - 1))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def forward_loss(model, objective, batch, config, device, stage, bf16):
    frames = (batch['mixture'].size(-1) + config.hop_length - 1) // config.hop_length
    extraction = batch.get('task', 'separation') == 'extraction'
    doa = frame_doa(batch, frames, config.sample_rate, config.hop_length) if extraction else None
    if extraction and doa is None:
        raise ValueError('DoA-guided extraction training requires a target direction cue')
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        output = model(batch['mixture'], doa=doa, source_images=batch['source_images'], lengths=batch['lengths'],
                       extraction=batch.get('task', 'separation') == 'extraction', return_details=True)
    with torch.autocast(device_type=device.type, enabled=False):
        return objective(output, batch['targets'], batch['lengths'], stage,
                         extraction=batch.get('task', 'separation') == 'extraction')[:2]


@torch.no_grad()
def validate(model, objective, dataset, config, device, bf16, max_items, show_progress=True):
    from cm_xbeam.data import collate_multichannel
    model.eval()
    data = Subset(dataset, range(min(len(dataset), max_items))) if max_items else dataset
    loader = DataLoader(data, batch_size=1, collate_fn=collate_multichannel)
    total, count = 0.0, 0
    metric_totals = {key: 0.0 for key in [*METRIC_SUMS, 'metric_valid_sources', 'metric_excluded_sources']}
    with TrainingProgress(len(loader), f'Valid {model.stage}', enabled=show_progress,
                          position=1, leave=False, unit='utt') as progress:
        for batch in loader:
            progress.set_phase('forward')
            batch = move_batch(batch, device)
            loss, terms = forward_loss(model, objective, batch, config, device, model.stage, bf16)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite validation objective: {batch["id"]}')
            total += float(terms['si_sdr_loss'])
            for key in metric_totals:
                metric_totals[key] += float(terms[key])
            count += 1
            metrics = mean_si_sdr_metrics(metric_totals)
            progress.advance(si_sdr_db=metric_label(metrics['si_sdr_db']),
                             si_sdri_db=metric_label(metrics['si_sdri_db']))
            progress.set_phase('loading')
    model.train()
    if not count:
        raise ValueError('Validation dataset is empty')
    return {'si_sdr_loss': total / count, **mean_si_sdr_metrics(metric_totals),
            'metric_valid_sources': int(metric_totals['metric_valid_sources']),
            'metric_excluded_sources': int(metric_totals['metric_excluded_sources']),
            'utterances': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ablation', choices=VARIANTS)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--crop-seconds', type=float)
    parser.add_argument('--validation-items', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--execution-backend', choices=['chunkwise', 'framewise'],
                        help='Chunkwise batches the same causal equations; framewise is the slower reference')
    parser.add_argument('--no-progress', action='store_true', help='Disable console bars while keeping JSON logs')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('max-steps must be positive')
    if args.crop_seconds is not None and args.crop_seconds <= 0:
        raise ValueError('crop-seconds must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    torch.set_num_threads(args.threads)
    cfg, matching = apply_variant(load_config(args.config), args.ablation)
    if args.seed is not None:
        cfg['train']['seed'] = args.seed
    if args.batch_size:
        cfg['train']['batch_size'] = args.batch_size
    if args.accumulation_steps is not None:
        cfg['train']['accumulation_steps'] = args.accumulation_steps
    if args.crop_seconds:
        cfg['data']['crop_seconds'] = args.crop_seconds
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = resolve_device(args.device)
    if world > 1:
        if device.type == 'cuda':
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    train_cfg, model_cfg = cfg['train'], ModelConfig(**cfg['model'])
    if args.execution_backend:
        model_cfg.execution_backend = args.execution_backend
    seed = int(train_cfg.get('seed', 0))
    seed_everything(seed + rank, train_cfg.get('deterministic', False))
    output = safe_output_path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    model = CMxBeam(model_cfg).to(device)
    objective = SeparationObjective(hop_length=model_cfg.hop_length, **cfg.get('loss', {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    curriculum = train_cfg['curriculum']
    if not curriculum or any(k not in {'A', 'B', 'C', 'D'} or int(v) <= 0 for k, v in curriculum.items()):
        raise ValueError('Curriculum must map stages A–D to positive optimizer-step counts')
    total_steps = sum(int(x) for x in curriculum.values())
    stop_step = min(args.max_steps or total_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: schedule(s, total_steps, train_cfg['warmup_steps']))
    step, epoch, cursor, best = 0, 0, 0, float('inf')
    elapsed_before = 0.0
    restored = None
    if args.resume:
        restored = torch.load(args.resume, map_location='cpu', weights_only=False)
        if int(restored.get('world_size', 1)) != world:
            raise ValueError('Resume GPU/process count mismatch; use the same world_size to preserve sampler and accumulation state')
        restored_model = ModelConfig(**restored['config']['model']).to_dict()
        current_model = model_cfg.to_dict()
        # Execution backend changes operator batching, not architecture/weights.
        restored_model.pop('execution_backend'); current_model.pop('execution_backend')
        if restored_model != current_model:
            raise ValueError('Resume model config mismatch; use the same configuration and ablation')
        if args.execution_backend is None:
            model_cfg.execution_backend = restored.get('execution_backend', model_cfg.execution_backend)
        def dataset_identity(config):
            data = config['data']
            return (str(data['train_manifest']), data.get('task', 'separation'), direction_options(data, 'train'))
        if dataset_identity(restored['config']) != dataset_identity(cfg):
            raise ValueError('Resume train-data/direction selection mismatch. Start a separate run when changing the sampled direction groups.')
        if restored['config'].get('train') != cfg.get('train'):
            raise ValueError('Resume training config mismatch; preserve seed/optimizer/curriculum/batch settings')
        model.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler'])
        step, epoch, cursor = restored['step'], restored['epoch'], restored['batch_in_epoch']
        best, elapsed_before = restored['best_valid'], restored.get('elapsed_seconds', 0)
        saved_rng = restored.get('rng_by_rank')
        restore_rng(saved_rng[rank] if saved_rng is not None else restored['rng'])
    train_data = dataset_from_config(cfg, 'train', training=True, seed=seed)
    valid_data = dataset_from_config(cfg, 'valid', seed=seed)
    from cm_xbeam.data import collate_multichannel
    parallel = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                       find_unused_parameters=True) if world > 1 else model
    accumulation = int(train_cfg['accumulation_steps'])
    if accumulation < 1:
        raise ValueError('accumulation_steps must be positive')
    bf16 = bool(train_cfg.get('bf16', False) and device.type == 'cuda')
    started = time.perf_counter()
    model.train()
    if rank == 0:
        print(json.dumps({'device': str(device), 'world_size': world,
                          'visible_gpus': os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
                          'batch_per_gpu': train_cfg['batch_size'], 'accumulation_steps': accumulation,
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'matching': matching, 'effective_batch': train_cfg['batch_size'] * accumulation * world,
                          'execution_backend': model_cfg.execution_backend if model_cfg.scm_mode == 'psd' else 'framewise (raw-memory ablation)',
                          'train_items': len(train_data), 'validation_items': len(valid_data),
                          'train_direction_selection': train_data.direction_selection}, ensure_ascii=False), flush=True)

    def save(name):
        local_rng = rng_state()
        if world > 1:
            rank_rng = [None] * world
            torch.distributed.all_gather_object(rank_rng, local_rng)
        else:
            rank_rng = [local_rng]
        if rank:
            return
        record = {'config': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(), 'step': step, 'stage': model.stage,
                  'epoch': epoch, 'batch_in_epoch': cursor, 'rng': local_rng,
                  'rng_by_rank': rank_rng, 'world_size': world, 'best_valid': best,
                  'elapsed_seconds': elapsed_before + time.perf_counter() - started, 'parameter_matching': matching,
                  'execution_backend': model_cfg.execution_backend}
        target = safe_output_path(output / name)
        temporary = target.with_suffix('.tmp')
        torch.save(record, temporary)
        temporary.replace(target)

    optimizer.zero_grad(set_to_none=True)
    micro = 0
    running = {}
    while step < stop_step:
        if hasattr(train_data, 'set_epoch'):
            train_data.set_epoch(epoch)
        sampler = DistributedSampler(train_data, world, rank, shuffle=True, seed=seed) if world > 1 else None
        if sampler:
            sampler.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(train_data, batch_size=train_cfg['batch_size'], shuffle=sampler is None,
                            sampler=sampler, num_workers=train_cfg.get('num_workers', 0), generator=generator,
                            collate_fn=collate_multichannel, pin_memory=device.type == 'cuda', drop_last=False)
        with TrainingProgress(len(loader), f'Train | epoch {epoch + 1}', initial=cursor,
                              enabled=rank == 0 and not args.no_progress) as progress:
            progress.set_phase('loading', step=f'{step}/{stop_step}', acc=f'{micro}/{accumulation}')
            for index, batch in enumerate(loader):
                if index < cursor:
                    continue
                stage = stage_at(step, curriculum)
                model.set_stage(stage, freeze=False)
                progress.set_description(f'Train {stage} | epoch {epoch + 1}')
                progress.set_phase('transfer', step=f'{step}/{stop_step}', acc=f'{micro + 1}/{accumulation}')
                batch = move_batch(batch, device)
                sync = (micro + 1) % accumulation == 0
                context = parallel.no_sync() if world > 1 and not sync else nullcontext()
                with context:
                    progress.set_phase('forward')
                    loss, terms = forward_loss(parallel, objective, batch, model_cfg, device, stage, bf16)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite training loss at step {step}: {batch["id"]}')
                    progress.set_phase('backward')
                    (loss / accumulation).backward()
                micro += 1
                cursor = index + 1
                metric_batch = mean_si_sdr_metrics(terms)
                for key, value in terms.items():
                    divisor = 1 if key in METRIC_SUMS or key.startswith('metric_') else accumulation
                    running[key] = running.get(key, 0.0) + float(value.detach()) / divisor
                progress.advance(loss=f'{float(loss.detach()):.3f}',
                                 si_sdr_db=metric_label(metric_batch['si_sdr_db']),
                                 si_sdri_db=metric_label(metric_batch['si_sdri_db']),
                                 lr=f'{scheduler.get_last_lr()[0]:.2e}')
                if not sync:
                    progress.set_phase('loading')
                    continue
                progress.set_phase('optimizer')
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                micro = 0
                if world > 1:
                    names = list(running)
                    metrics = torch.tensor([running[name] for name in names], device=device, dtype=torch.float64)
                    torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
                    running = {name: value if name in METRIC_SUMS or name.startswith('metric_') else value / world
                               for name, value in zip(names, metrics.cpu().tolist())}
                log = {'step': step, 'stage': stage, 'epoch': epoch, 'lr': scheduler.get_last_lr()[0],
                       'gradient_norm': float(norm), **format_training_terms(running)}
                if rank == 0 and (step % train_cfg.get('log_every', 10) == 0 or step == stop_step):
                    TrainingProgress.write(json.dumps(log))
                    with (output / 'train.jsonl').open('a') as handle:
                        handle.write(json.dumps(log) + '\n')
                running = {}
                if step % train_cfg.get('validate_every', 1000) == 0 or step == stop_step:
                    progress.set_phase('validation', step=f'{step}/{stop_step}', acc=f'0/{accumulation}')
                    if world > 1:
                        torch.distributed.barrier()
                    validation_result = [None, False]
                    if rank == 0:
                        validation_metrics = validate(model, objective, valid_data, model_cfg, device, bf16,
                                                      args.validation_items, show_progress=not args.no_progress)
                        score = validation_metrics['si_sdr_loss']
                        validation_log = {'step': step, 'stage': stage, 'epoch': epoch,
                                          **validation_metrics, 'valid_negative_si_sdr': score}
                        TrainingProgress.write(json.dumps(validation_log, allow_nan=False))
                        with (output / 'valid.jsonl').open('a') as handle:
                            handle.write(json.dumps(validation_log, allow_nan=False) + '\n')
                        validation_result = [score, score < best]
                    if world > 1:
                        torch.distributed.broadcast_object_list(validation_result, src=0)
                    score, improved = validation_result
                    if improved:
                        best = score
                        progress.set_phase('save best.pt')
                        # All ranks contribute RNG state; only rank 0 writes.
                        save('best.pt')
                    if world > 1:
                        torch.distributed.barrier()
                if step % train_cfg.get('checkpoint_every', 1000) == 0 or step == stop_step:
                    progress.set_phase('save latest.pt')
                    save('latest.pt')
                progress.set_phase('loading', step=f'{step}/{stop_step}', acc=f'0/{accumulation}')
                if step >= stop_step:
                    break
            else:
                epoch += 1
                cursor = 0
                if not len(loader):
                    raise ValueError('Training dataset is empty')
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
