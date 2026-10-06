"""Bounded chunk execution of the unchanged CM-xBeam causal architecture."""
import torch
from torch.nn import functional as F
from .mask_head import resize_frequency
from .scm_accumulator import outer, update_scm_sequence, previous_covariance
from .steering import reference_rtf
from .mvdr import diagonal_load, mvdr_solve
from .residual_beamformer import project_orthogonal
from .uncertainty import physical_features, oracle_error


def run_chunk(model, y, previous, cues, images, valid, extraction, diagnostics):
    c = model.config
    previous = previous or {}
    b, t, f, m = y.shape
    with torch.autocast(device_type=y.device.type, enabled=False):
        y = y.to(torch.complex64)
        features = model.features(y)
    h, encoder_state = model.encoder.sequence(features, previous.get('encoder'))
    low_f = h.size(2)
    flat_cues = None if cues is None else cues.transpose(1, 2).reshape(b * t, c.num_sources)
    flat_query = model.query(b * t, flat_cues)
    query = flat_query.reshape(b, t, c.num_sources, c.dim).transpose(1, 2)
    masks, entropy = model.mask_head(h.reshape(b * t, low_f, c.dim), flat_query, f, extraction)
    masks = masks.reshape(b, t, c.num_sources, f).transpose(1, 2)
    entropy = entropy.reshape(b, t, c.num_sources, f).transpose(1, 2)
    low_masks = resize_frequency(masks.unsqueeze(-1), low_f).squeeze(-1)
    memories, forgetting, update, memory_state = model.controller.sequence(h, query, low_masks, previous.get('memory'))
    forgetting = resize_frequency(forgetting.unsqueeze(-1), f).squeeze(-1).float()
    update = resize_frequency(update.unsqueeze(-1), f).squeeze(-1).float()
    state = {'encoder': encoder_state, 'memory': memory_state}
    with torch.autocast(device_type=y.device.type, enabled=False):
        observation = outer(y)[:, None].expand(-1, c.num_sources, -1, -1, -1, -1)
        phi_s, state['scm_s'] = update_scm_sequence(previous.get('scm_s'), observation,
            forgetting[:, :, 0], update[:, :, 0] * masks.float(), c.eps)
        phi_i, state['scm_i'] = update_scm_sequence(previous.get('scm_i'), observation,
            forgetting[:, :, 1], update[:, :, 1] * (1 - masks.float()), c.eps)
        unit = torch.zeros((b, c.num_sources, t, f, m), device=y.device, dtype=torch.complex64)
        unit[..., c.reference_mic] = 1
        # A/B do not train the residual/uncertainty branch. Do not spend hundreds
        # of eigensolver/MLP launches on an inactive branch during warm-up.
        neural_active = model.context is not None and (
            model.stage in {'C', 'D'} or (model.direct_head is not None and model.stage != 'A') or diagnostics)
        needs_statistics = neural_active or diagnostics
        loaded_i = diagonal_load(phi_i, c.load_scale, c.load_floor) if model.stage != 'A' or needs_statistics else None
        d = reference_rtf(phi_s, c.reference_mic, c.eps) if model.stage != 'A' or needs_statistics else unit
        w_mvdr = unit if model.stage == 'A' or c.beamformer == 'direct' else mvdr_solve(loaded_i, d, c.eps)
        phy = None
        if needs_statistics:
            initial_phi = previous_covariance(previous.get('scm_s'), phi_s[:, :, 0], c.eps)
            past_phi = torch.cat([initial_phi[:, :, None], phi_s[:, :, :-1]], dim=2)
            phy = physical_features(phi_s, loaded_i, past_phi, masks[..., None, None] * observation, entropy, c.eps)
    joint = None
    if neural_active:
        context = model.context(torch.cat([memories[:, :, 0], memories[:, :, 1]], dim=-1))
        context = resize_frequency(context, f)
        joint = torch.cat([context, phy.to(context.dtype)], dim=-1)
    if model.uncertainty_head is not None and neural_active:
        uncertainty, predicted_alpha = model.uncertainty_head(joint)
    else:
        uncertainty = masks.new_zeros(masks.shape)
        predicted_alpha = masks.new_full(masks.shape, c.fixed_alpha)
    with torch.autocast(device_type=y.device.type, enabled=False):
        uncertainty = uncertainty.float()
        alpha = predicted_alpha.float() if c.uncertainty else torch.full_like(predicted_alpha.float(), c.fixed_alpha)
        if c.residual == 'none' or model.stage in {'A', 'B'}:
            alpha = torch.zeros_like(alpha)
        delta = model.residual_head(joint.float()) if model.residual_head is not None and neural_active else torch.zeros_like(d)
        residual = project_orthogonal(delta, d, c.eps) if c.residual == 'nullspace' else delta
        applied = alpha[..., None] * residual
        weights = w_mvdr + applied
        if model.direct_head is not None and model.stage != 'A':
            weights = model.direct_head(joint.float())
        z = (weights.conj() * y[:, None]).sum(-1)
        if model.stage == 'A':
            z = masks.float() * y[:, None, :, :, c.reference_mic]
        penalty = applied.abs().square().sum(-1) / w_mvdr.abs().square().sum(-1).clamp_min(c.eps)
        residual_sum = (penalty.mean(dim=(1, 3)) * valid).sum(-1)
        scm_sum = y.real.new_zeros(b, c.num_sources, c.num_sources)
        unc_sum = torch.zeros_like(scm_sum)
        error = None
        if images is not None:
            oracle_s_obs = outer(images.to(torch.complex64))
            oracle_i_obs = outer(y[:, None] - images.to(torch.complex64))
            lam = torch.full(images.shape[:-1], c.oracle_lambda, device=y.device)
            ones = torch.ones_like(lam)
            oracle_s, state['oracle_s'] = update_scm_sequence(previous.get('oracle_s'), oracle_s_obs, lam, ones, c.eps)
            oracle_i, state['oracle_i'] = update_scm_sequence(previous.get('oracle_i'), oracle_i_obs, lam, ones, c.eps)
            error = 0.5 * (oracle_error(phi_s[:, :, None], oracle_s[:, None], c.eps)
                         + oracle_error(phi_i[:, :, None], oracle_i[:, None], c.eps))
            scm_sum = (error.mean(-1) * valid[:, None, None]).sum(-1)
            if model.uncertainty_head is not None and neural_active:
                target_unc = torch.log(error.detach() + c.eps)
                unc_error = F.huber_loss(uncertainty[:, :, None].expand_as(target_unc), target_unc, reduction='none')
                unc_sum = (unc_error.mean(-1) * valid[:, None, None]).sum(-1)
        traces = {}
        if diagnostics:
            constraint = ((weights.conj() * d).sum(-1) - 1).abs()
            traces = {'lambda_s': forgetting[:, :, 0].mean(-1).detach(),
                      'lambda_i': forgetting[:, :, 1].mean(-1).detach(),
                      'alpha': alpha.mean(-1).detach(), 'uncertainty': uncertainty.mean(-1).detach(),
                      'distortionless_error': constraint.mean(-1).detach(),
                      'mask_entropy': entropy.mean(-1).detach(),
                      'log_condition': phy[..., 0].mean(-1).detach(),
                      'rank_impurity': phy[..., 1].mean(-1).detach(),
                      'log_innovation': phy[..., 3].mean(-1).detach()}
            if error is not None:
                traces['covariance_error_pairs'] = error.mean(-1).detach()
    if model.refiner is not None and model.stage == 'D':
        refined, state['refiner'] = model.refiner.sequence(z, previous.get('refiner'))
    else:
        refined = z
    return {'pre': z, 'post': refined, 'masks': masks, 'scm': scm_sum, 'unc': unc_sum,
            'residual': residual_sum, 'state': state, 'traces': traces}
