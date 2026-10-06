"""Detached, length-aware SI-SDR metrics with shared PIT assignment and counts."""
import torch

METRIC_SUMS = {'si_sdr_sum_db': 'si_sdr_db',
               'mixture_si_sdr_sum_db': 'mixture_si_sdr_db',
               'si_sdri_sum_db': 'si_sdri_db'}


@torch.no_grad()
def aligned_si_sdr_sums(estimates, targets, mixture_reference, permutation, lengths=None,
                        eps=1e-8, output_scores=None):
    from ..losses.sisdr import pairwise_si_sdr, sample_mask, reorder_sources
    if mixture_reference.shape != (targets.size(0), targets.size(-1)):
        raise ValueError('Reference mixture must be [B,N], matching target length')
    targets = targets.detach().float()
    estimates = estimates.detach().float()
    mixture_reference = mixture_reference.detach().float()
    scores = pairwise_si_sdr(estimates, targets, lengths, eps) if output_scores is None else output_scores.detach()
    baseline = pairwise_si_sdr(mixture_reference[:, None], targets, lengths, eps)[:, 0]
    b, k, _ = targets.shape
    batch = torch.arange(b, device=targets.device)[:, None]
    slots = torch.arange(k, device=targets.device)[None]
    output_db = scores[batch, slots, permutation]
    baseline_db = baseline[batch, permutation]
    mask = sample_mask(lengths, targets.size(-1), targets.device)
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    centered = (targets - (targets * mask).sum(-1, keepdim=True) / count) * mask
    active = (centered.square().sum(-1) / count.squeeze(-1)) >= eps
    active = reorder_sources(active, permutation)
    valid = active & torch.isfinite(output_db) & torch.isfinite(baseline_db)
    def total(values):
        return torch.where(valid, values, torch.zeros_like(values)).double().sum()
    return {'si_sdr_sum_db': total(output_db),
            'mixture_si_sdr_sum_db': total(baseline_db),
            'si_sdri_sum_db': total(output_db - baseline_db),
            'metric_valid_sources': valid.double().sum(),
            'metric_excluded_sources': (~valid).double().sum()}


def mean_si_sdr_metrics(totals):
    """Normalize sums only after accumulation/all-reduce, avoiding mean-of-means."""
    count = float(totals.get('metric_valid_sources', 0))
    return {name: float(totals[key]) / count if count > 0 else None
            for key, name in METRIC_SUMS.items()}
