"""Query-conditioned masks; source probabilities include an explicit noise slot."""
import torch
from torch import nn
from torch.nn import functional as F


def resize_frequency(x, frequency):
    # Any leading axes, with [F,D] last. Only real scalar/features are resized.
    if x.size(-2) == frequency:
        return x
    shape = x.shape
    out = F.interpolate(x.reshape(-1, shape[-2], shape[-1]).transpose(1, 2), size=frequency, mode='linear', align_corners=False)
    return out.transpose(1, 2).reshape(*shape[:-2], frequency, shape[-1])


class MaskHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.source = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.noise = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, h, query, frequency, extraction=False):
        conditioned = h[:, None] + query[:, :, None]
        logits = resize_frequency(self.source(conditioned), frequency).squeeze(-1)
        if extraction:
            masks = logits.sigmoid()
            entropy = -(masks * masks.clamp_min(1e-8).log() + (1 - masks) * (1 - masks).clamp_min(1e-8).log())
        else:
            noise = resize_frequency(self.noise(h), frequency).squeeze(-1)
            probabilities = torch.cat([logits, noise[:, None]], dim=1).softmax(dim=1)
            masks = probabilities[:, :-1]
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1, keepdim=True).expand_as(masks)
        return masks, entropy
