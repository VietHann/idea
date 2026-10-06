"""Complex real/imaginary L1 and magnitude supervision without padded frames."""
import torch


def complex_loss(estimate, target, frame_mask):
    error = (estimate.real - target.real).abs() + (estimate.imag - target.imag).abs()
    error = error + 0.5 * (estimate.abs() - target.abs()).abs()
    return (error * frame_mask).sum() / (frame_mask.sum() * error.size(1) * error.size(-1)).clamp_min(1)
