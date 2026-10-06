"""PSD-preserving normalized covariance recurrence, with only current states."""
import torch


def hermitian(phi):
    return 0.5 * (phi + phi.mH)


def outer(y):
    return y.unsqueeze(-1) * y.conj().unsqueeze(-2)


def normalize_trace(phi, eps=1e-8):
    trace = phi.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    return phi / trace.clamp_min(eps)[..., None, None]


def update_scm(state, observation, forgetting, weight, eps=1e-8):
    observation = observation.to(torch.complex64)
    if state is None:
        numerator = torch.zeros_like(observation)
        normalizer = torch.zeros_like(weight, dtype=torch.float32)
    else:
        numerator, normalizer = state
    numerator = forgetting[..., None, None].float() * numerator + weight[..., None, None].float() * observation
    normalizer = forgetting.float() * normalizer + weight.float()
    phi = hermitian(numerator / (normalizer + eps)[..., None, None])
    return phi, (numerator, normalizer)


def previous_covariance(state, shape_like, eps=1e-8):
    if state is None:
        return torch.zeros_like(shape_like)
    return hermitian(state[0] / (state[1] + eps)[..., None, None])


def update_scm_sequence(state, observation, forgetting, weight, eps=1e-8):
    """Bounded associative affine scan; observations [B,K,T,F,M,M].

    Composition (lambda, update) stays nonnegative, so the same PSD recurrence
    is evaluated with log2(T) batched operations. No cumulative-product division
    is used, including when a forgetting coefficient is zero.
    """
    observation = observation.to(torch.complex64)
    decay = forgetting.float()
    numerator = weight.float()[..., None, None] * observation
    normalizer = weight.float()
    frames = forgetting.size(2)
    offset = 1
    while offset < frames:
        suffix_decay = decay[:, :, offset:]
        numerator = torch.cat([numerator[:, :, :offset], numerator[:, :, offset:]
            + suffix_decay[..., None, None] * numerator[:, :, :-offset]], dim=2)
        normalizer = torch.cat([normalizer[:, :, :offset], normalizer[:, :, offset:]
            + suffix_decay * normalizer[:, :, :-offset]], dim=2)
        decay = torch.cat([decay[:, :, :offset], suffix_decay * decay[:, :, :-offset]], dim=2)
        offset *= 2
    if state is not None:
        numerator = numerator + decay[..., None, None] * state[0][:, :, None]
        normalizer = normalizer + decay * state[1][:, :, None]
    phi = hermitian(numerator / (normalizer + eps)[..., None, None])
    return phi, (numerator[:, :, -1], normalizer[:, :, -1])
