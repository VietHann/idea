"""Reference-vector relative transfer function, with silent-bin fallback."""
import torch


def reference_rtf(phi, reference_mic=0, eps=1e-8):
    column = phi[..., :, reference_mic]
    denominator = phi[..., reference_mic, reference_mic].real
    d = column / denominator.clamp_min(eps)[..., None]
    unit = torch.zeros_like(d)
    unit[..., reference_mic] = 1
    d = torch.where((denominator > eps)[..., None], d, unit)
    # Explicit reference normalization avoids an epsilon-induced constraint bias.
    d = d.clone()
    d[..., reference_mic] = 1
    return d
