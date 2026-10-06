"""Physical diagnostics and calibrated covariance-error-controlled gating."""
import torch
from torch import nn
from torch.nn import functional as F
from .scm_accumulator import normalize_trace


def physical_features(phi_s, loaded_i, previous_s, instantaneous, entropy, eps=1e-8):
    # Reliability inputs are observations, not unstable eigenvector gradient paths.
    with torch.no_grad():
        ev_s = torch.linalg.eigvalsh(0.5 * (phi_s.detach() + phi_s.detach().mH))
        ev_i = torch.linalg.eigvalsh(0.5 * (loaded_i.detach() + loaded_i.detach().mH))
        condition = (ev_i[..., -1].abs().clamp_min(eps) / ev_i[..., 0].abs().clamp_min(eps)).log().clamp(0, 20)
        trace = ev_s.sum(-1).clamp_min(eps)
        rank = (1 - ev_s[..., -1] / trace).clamp(0, 1)
        delta = torch.linalg.matrix_norm(instantaneous.detach() - previous_s.detach(), ord='fro')
        previous_norm = torch.linalg.matrix_norm(previous_s.detach(), ord='fro').clamp_min(eps)
        innovation = torch.log1p(delta / previous_norm).clamp(0, 20)
        return torch.stack([condition, rank, entropy.detach(), innovation], dim=-1).float()


class UncertaintyHead(nn.Module):
    def __init__(self, input_dim, alpha_max=0.25):
        super().__init__()
        self.predict = nn.Sequential(nn.Linear(input_dim, 96), nn.SiLU(), nn.Linear(96, 1))
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.offset = nn.Parameter(torch.tensor(1.0))
        self.alpha_max = alpha_max

    def forward(self, features):
        log_error = self.predict(features).squeeze(-1)
        # Positive slope makes the correction monotonic in predicted uncertainty.
        alpha = self.alpha_max * torch.sigmoid(F.softplus(self.raw_slope) * log_error + self.offset)
        return log_error, alpha


def oracle_error(estimate, oracle, eps=1e-8):
    difference = normalize_trace(estimate, eps) - normalize_trace(oracle, eps)
    return torch.linalg.matrix_norm(difference, ord='fro')
