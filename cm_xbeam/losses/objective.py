"""Curriculum-aware six-term objective with SI-SDR-selected PIT assignment."""
import torch
from torch import nn
from torch.nn import functional as F
from .sisdr import pairwise_si_sdr, best_permutation, reorder_sources, sample_mask
from .complex_loss import complex_loss
from ..metrics.training import aligned_si_sdr_sums


class SeparationObjective(nn.Module):
    def __init__(self, complex_weight=0.2, scm_weight=0.05, uncertainty_weight=0.02,
                 residual_weight=1e-3, mask_weight=0.05, hop_length=128, eps=1e-8):
        super().__init__()
        self.weights = {'complex': complex_weight, 'scm': scm_weight, 'uncertainty': uncertainty_weight,
                        'residual': residual_weight, 'mask': mask_weight}
        self.hop_length, self.eps = hop_length, eps

    def forward(self, output, targets, lengths=None, stage='D', extraction=False):
        waveform = output['waveform']
        scores = pairwise_si_sdr(waveform, targets, lengths, self.eps)
        b, k, n = targets.shape
        valid_samples = sample_mask(lengths, n, targets.device)
        target_power = (targets.square() * valid_samples).sum(-1) / valid_samples.sum(-1).clamp_min(1)
        silent = target_power < self.eps
        estimate_power = (waveform.square() * valid_samples).sum(-1) / valid_samples.sum(-1).clamp_min(1)
        silence_cost = 10 * torch.log10(1 + estimate_power[:, :, None] / self.eps)
        assignment_score = torch.where(silent[:, None], -silence_cost, scores)
        permutation = torch.arange(k, device=targets.device)[None].expand(b, -1) if extraction else best_permutation(assignment_score.detach())
        batch = torch.arange(b, device=targets.device)[:, None]
        slots = torch.arange(k, device=targets.device)[None]
        waveform_loss = -assignment_score[batch, slots, permutation].mean()
        target_spectrum = output['target_spectrum']
        if target_spectrum is None:
            raise ValueError('Training requires source images for oracle spatial supervision')
        target_spectrum = reorder_sources(target_spectrum, permutation)
        frames = target_spectrum.size(2)
        frame_mask = torch.ones(b, 1, frames, 1, device=targets.device) if lengths is None else (
            torch.arange(frames, device=targets.device)[None, None, :, None] * self.hop_length < lengths[:, None, None, None]).float()
        spectral = complex_loss(output['spectrum'], target_spectrum, frame_mask)
        aux = output['auxiliary']
        scm = aux['scm'][batch, slots, permutation].mean()
        uncertainty = aux['uncertainty'][batch, slots, permutation].mean()
        residual = aux['residual'].mean()
        power = target_spectrum.abs().square()
        # Noise is a true residual source-image signal, not mixture-power subtraction.
        noise = output['mixture_spectrum'] - target_spectrum.sum(dim=1)
        total_power = power.sum(dim=1, keepdim=True) + noise.abs().square()[:, None]
        oracle_masks = (power / total_power.clamp_min(self.eps)).clamp(0, 1)
        mask_error = F.binary_cross_entropy(output['masks'].float().clamp(1e-6, 1 - 1e-6), oracle_masks, reduction='none')
        mask = (mask_error * frame_mask).sum() / (frame_mask.sum() * k * power.size(-1)).clamp_min(1)
        terms = {'si_sdr_loss': waveform_loss, 'complex': spectral, 'scm': scm,
                 'uncertainty': uncertainty, 'residual': residual, 'mask': mask}
        terms.update(aligned_si_sdr_sums(waveform, targets, output['mixture_waveform'],
                                       permutation, lengths, self.eps, output_scores=scores))
        loss = waveform_loss + self.weights['complex'] * spectral + self.weights['scm'] * scm + self.weights['mask'] * mask
        if stage in {'C', 'D'}:
            loss = loss + self.weights['uncertainty'] * uncertainty + self.weights['residual'] * residual
        return loss, terms, permutation
