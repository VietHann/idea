"""Length-aware separation metrics with a single shared source assignment."""
import itertools
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .perceptual import perceptual_scores


def si_sdr(estimate, target, eps=1e-10):
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    estimate, target = estimate - estimate.mean(), target - target.mean()
    energy = float(target @ target)
    if energy <= eps:
        return None
    projection = ((estimate @ target) / energy) * target
    residual = estimate - projection
    return float(10 * np.log10((projection @ projection + eps) / (residual @ residual + eps)))


def _array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def evaluate_sources(estimates, targets, mixture, sample_rate, permutation=None, perceptual=True, length=None):
    estimates, targets, mixture = _array(estimates), _array(targets), _array(mixture)
    if estimates.ndim != 2 or estimates.shape != targets.shape or mixture.shape != estimates.shape[1:]:
        raise ValueError('Expected estimates/targets [K,N], mixture [N], with identical lengths')
    if length is not None:
        estimates, targets, mixture = estimates[:, :length], targets[:, :length], mixture[:length]
    if not all(np.isfinite(x).all() for x in [estimates, targets, mixture]):
        raise ValueError('Metrics require finite inputs')
    if estimates.shape[-1] < 2:
        raise ValueError('At least two waveform samples are required')
    k = targets.shape[0]
    if permutation is None:
        pairwise = np.array([[si_sdr(e, t) for t in targets] for e in estimates], dtype=object)
        scores = np.array([[float(x) if x is not None else -1e6 for x in row] for row in pairwise])
        rows, columns = linear_sum_assignment(-scores)
        permutation = np.empty(k, dtype=int)
        permutation[rows] = columns
    else:
        permutation = np.asarray(permutation, dtype=int)
        if sorted(permutation.tolist()) != list(range(k)):
            raise ValueError('permutation must map each output slot to one target')
    sources = []
    for slot, target_index in enumerate(permutation):
        target, estimate = targets[target_index], estimates[slot]
        score, baseline = si_sdr(estimate, target), si_sdr(mixture, target)
        metrics = {'si_sdr': score, 'si_sdri': None if score is None or baseline is None else score - baseline,
                   'mixture_si_sdr': baseline, 'stoi': None, 'wb_pesq': None}
        if score is None:
            metrics['metric_errors'] = {'reference': 'silent_reference; SI-SDR/intelligibility undefined'}
        elif perceptual:
            metrics.update(perceptual_scores(estimate, target, sample_rate))
        sources.append(metrics)
    return {'sources': sources, 'permutation': permutation.tolist()}
