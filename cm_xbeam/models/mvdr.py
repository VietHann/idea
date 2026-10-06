"""Differentiable, diagonally loaded MVDR without explicit matrix inversion."""
import torch
from .scm_accumulator import hermitian


def diagonal_load(phi, scale=1e-3, floor=1e-6):
    m = phi.size(-1)
    energy = phi.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(0) / m
    loading = scale * energy + floor
    eye = torch.eye(m, device=phi.device, dtype=phi.dtype)
    return phi + loading[..., None, None] * eye


def distortionless_normalize(weight, d, eps=1e-8):
    response = (weight.conj() * d).sum(-1)
    safe = response.abs() > eps
    scale = torch.where(safe, response.conj(), torch.ones_like(response))
    normalized = weight / scale[..., None]
    fallback = d / d.abs().square().sum(-1).clamp_min(eps)[..., None]
    return torch.where(safe[..., None], normalized, fallback)


def mvdr_solve(loaded, d, eps=1e-8, raw_memory=False):
    if raw_memory:
        v = torch.linalg.solve(loaded, d.unsqueeze(-1)).squeeze(-1)
    else:
        chol, info = torch.linalg.cholesky_ex(hermitian(loaded))
        # Avoid feeding failed decompositions into autograd even for masked bins.
        ok = info == 0
        eye = torch.eye(loaded.size(-1), device=loaded.device, dtype=loaded.dtype)
        safe_chol = torch.where(ok[..., None, None], chol, eye)
        solved = torch.cholesky_solve(d.unsqueeze(-1), safe_chol).squeeze(-1)
        if bool(ok.all()):
            v = solved
        else:
            fallback_matrix = torch.where(ok[..., None, None], eye, loaded)
            fallback = torch.linalg.solve(fallback_matrix, d.unsqueeze(-1)).squeeze(-1)
            v = torch.where(ok[..., None], solved, fallback)
    return distortionless_normalize(v, d, eps)
