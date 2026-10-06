"""Length-aware SI-SDR and one shared PIT assignment for all loss terms."""
import itertools
import torch


def sample_mask(lengths, samples, device):
    if lengths is None:
        return torch.ones(1, 1, samples, device=device)
    return (torch.arange(samples, device=device)[None, None] < lengths[:, None, None]).float()


def pairwise_si_sdr(estimates, targets, lengths=None, eps=1e-8):
    estimates, targets = estimates.float(), targets.float()
    mask = sample_mask(lengths, estimates.size(-1), estimates.device)
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    estimates = (estimates - (estimates * mask).sum(-1, keepdim=True) / count) * mask
    targets = (targets - (targets * mask).sum(-1, keepdim=True) / count) * mask
    dot = torch.einsum('bkn,bjn->bkj', estimates, targets)
    target_energy = targets.square().sum(-1)
    projection_energy = dot.square() / target_energy[:, None].clamp_min(eps)
    error_energy = estimates.square().sum(-1)[:, :, None] - projection_energy
    return 10 * torch.log10((projection_energy + eps) / (error_energy.clamp_min(0) + eps))


def best_permutation(scores):
    b, k, j = scores.shape
    if k != j or k > 4:
        raise ValueError('PIT requires equal source counts <=4')
    permutations = torch.tensor(list(itertools.permutations(range(k))), device=scores.device)
    rows = torch.arange(k, device=scores.device)
    values = scores[:, rows[None], permutations].mean(-1)
    return permutations[values.argmax(-1)]


def reorder_sources(targets, permutation):
    batch = torch.arange(targets.size(0), device=targets.device)[:, None]
    return targets[batch, permutation]
