"""Paired statistical inference; sources are averaged within utterances first."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def paired_bootstrap(differences, resamples=10000, seed=0):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or resamples < 1:
        raise ValueError('Bootstrap requires nonempty finite differences and positive resamples')
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 256):
        stop = min(resamples, start + 256)
        means[start:stop] = values[rng.integers(len(values), size=(stop - start, len(values)))].mean(-1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {'mean_difference': float(values.mean()), 'ci95': [float(low), float(high)],
            'utterances': len(values), 'resamples': resamples}


def hierarchical_bootstrap(differences, resamples=10000, seed=0):
    # Matrix [seed, utterance], matched IDs/seed definitions across both systems.
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Hierarchical bootstrap requires finite matched [seed,utterance] values')
    rng = np.random.default_rng(seed)
    seeds, utterances = values.shape
    means = np.empty(resamples)
    for i in range(resamples):
        selected_seeds = rng.integers(seeds, size=seeds)
        selected_utterances = rng.integers(utterances, size=(seeds, utterances))
        means[i] = values[selected_seeds[:, None], selected_utterances].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    seed_means = values.mean(-1)
    return {'mean_difference': float(values.mean()), 'ci95': [float(lo), float(hi)],
            'seed_means': seed_means.tolist(), 'seed_std': float(seed_means.std(ddof=1)) if seeds > 1 else None,
            'seeds': seeds, 'utterances_per_seed': utterances, 'resamples': resamples}


def sign_flip_test(differences, resamples=10000, seed=0):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Sign-flip test requires finite utterance differences')
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    if len(values) <= 16:
        indices = np.arange(2 ** len(values), dtype=np.uint32)
        signs = (((indices[:, None] >> np.arange(len(values))) & 1) * 2.0 - 1)
        p = float((np.abs((signs * values).mean(-1)) >= observed - 1e-12).mean())
        return {'p_value': p, 'exact': True, 'resamples': len(indices)}
    exceed = 0
    for start in range(0, resamples, 256):
        count = min(256, resamples - start)
        signs = rng.integers(0, 2, size=(count, len(values))) * 2 - 1
        exceed += int((np.abs((signs * values).mean(-1)) >= observed - 1e-12).sum())
    return {'p_value': (exceed + 1) / (resamples + 1), 'exact': False, 'resamples': resamples}


def holm_bonferroni(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('p-values must be in [0,1]')
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def read_records(paths, stage='post_refiner', metric='si_sdri'):
    groups = defaultdict(dict)
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['stage'] != stage:
                    continue
                seed = int(row['seed'])
                key = (str(row.get('dataset', '')), str(row['id']), int(row['source']))
                if key in groups[seed]:
                    raise ValueError(f'Duplicate result at seed={seed}, {key}')
                value = row.get(metric)
                if value is not None and not np.isfinite(float(value)):
                    raise ValueError(f'Nonfinite result {key}')
                groups[seed][key] = None if value is None else float(value)
    if not groups:
        raise ValueError('No matching evaluation records')
    return dict(groups)


def paired_matrix(a, b):
    if set(a) != set(b):
        raise ValueError('Compared systems must have identical seed sets')
    seeds = sorted(a)
    source_keys = set(a[seeds[0]])
    for seed in seeds:
        if set(a[seed]) != set(b[seed]) or set(a[seed]) != source_keys:
            raise ValueError('Every seed/system must contain the same dataset/utterance/source keys')
    valid_keys = []
    for key in sorted(source_keys):
        valid = [a[seed][key] is not None and b[seed][key] is not None for seed in seeds]
        asymmetric = [((a[seed][key] is None) != (b[seed][key] is None)) for seed in seeds]
        if any(asymmetric):
            raise ValueError(f'Asymmetric missing metric for {key}; repair evaluation rather than dropping difficult examples')
        if all(valid):
            valid_keys.append(key)
    utterances = sorted({key[:2] for key in valid_keys})
    if not utterances:
        raise ValueError('No paired valid utterances')
    grouped = {utt: [key for key in valid_keys if key[:2] == utt] for utt in utterances}
    differences = np.array([[np.mean([a[seed][key] - b[seed][key] for key in grouped[utt]]) for utt in utterances] for seed in seeds])
    return differences, seeds, utterances
