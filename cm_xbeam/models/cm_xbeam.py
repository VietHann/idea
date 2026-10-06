"""CM-xBeam end-to-end model with one causal spectral step as canonical path."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..config import ModelConfig
from .stft import CausalSTFT
from .spatial_features import SpatialFeatures
from .spatial_encoder import SpatialEncoder
from .query_encoder import QueryEncoder
from .mask_head import MaskHead, resize_frequency
from .covariance_xlstm import CovarianceController
from .scm_accumulator import outer, update_scm, previous_covariance
from .steering import reference_rtf
from .mvdr import diagonal_load, mvdr_solve
from .residual_beamformer import ResidualBeamformer, project_orthogonal
from .uncertainty import UncertaintyHead, physical_features, oracle_error
from .tf_xlstm_refiner import TFXLSTMRefiner


class CMxBeam(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**(config or {}))
        c = self.config
        self.stft = CausalSTFT(c.n_fft, c.hop_length)
        self.features = SpatialFeatures(c.num_channels, c.reference_mic, c.include_ipd, c.eps)
        self.encoder = SpatialEncoder(self.features.output_dim, c.dim, c.encoder_blocks, c.encoder_kernel_time)
        self.query = QueryEncoder(c.num_sources, c.dim, c.doa_harmonics)
        self.mask_head = MaskHead(c.dim)
        self.controller = CovarianceController(c)
        needs_context = c.residual != 'none' or c.uncertainty or c.beamformer == 'direct'
        self.context = nn.Linear(2 * c.dim, 64) if needs_context else None
        self.residual_head = ResidualBeamformer(68, c.num_channels) if c.residual != 'none' else None
        self.direct_head = ResidualBeamformer(68, c.num_channels, zero_init=False) if c.beamformer == 'direct' else None
        self.uncertainty_head = UncertaintyHead(68, c.alpha_max) if c.uncertainty else None
        self.refiner = TFXLSTMRefiner(c.refiner_blocks, c.refiner_dim, c.refiner_heads,
                                    c.refiner_ff_dim, c.refiner_bound, checkpoint_blocks=c.checkpoint_frames) if c.refiner_blocks else None
        self.stage = 'D'

    def set_stage(self, stage, freeze=True):
        if stage not in {'A', 'B', 'C', 'D'}:
            raise ValueError('Curriculum stage must be A, B, C or D')
        self.stage = stage
        if freeze:
            for name, parameter in self.named_parameters():
                enabled = True
                if name.startswith(('residual_head.', 'uncertainty_head.', 'context.', 'direct_head.')):
                    enabled = stage in {'C', 'D'} or self.config.beamformer == 'direct'
                if name.startswith('refiner.'):
                    enabled = stage == 'D'
                parameter.requires_grad_(enabled)

    def step_spectrum(self, y, state=None, doa=None, source_images=None, extraction=False):
        if y.ndim != 3 or y.size(-1) != self.config.num_channels:
            raise ValueError('Expected complex spectral frame [B,F,M]')
        if not y.is_complex():
            raise ValueError('Spectral input must be complex')
        c, state = self.config, (state or {})
        b, f, m = y.shape
        with torch.autocast(device_type=y.device.type, enabled=False):
            y = y.to(torch.complex64)
            feat = self.features(y)
        h, encoder_state = self.encoder.step(feat, state.get('encoder'))
        query = self.query(b, doa)
        masks, entropy = self.mask_head(h, query, f, extraction)
        low_masks = resize_frequency(masks.unsqueeze(-1), h.size(1)).squeeze(-1)
        memories, forgetting, update, memory_state, raw = self.controller.step(h, query, low_masks, state.get('memory'))
        forgetting = resize_frequency(forgetting.unsqueeze(-1), f).squeeze(-1).float()
        update = resize_frequency(update.unsqueeze(-1), f).squeeze(-1).float()
        context = None
        if self.context is not None:
            context = resize_frequency(self.context(torch.cat([memories[:, :, 0], memories[:, :, 1]], dim=-1)), f)
        new_state = {'encoder': encoder_state, 'memory': memory_state}
        with torch.autocast(device_type=y.device.type, enabled=False):
            observation = outer(y)[:, None].expand(-1, c.num_sources, -1, -1, -1)
            previous_s = previous_covariance(state.get('scm_s'), observation, c.eps)
            phi_s, new_state['scm_s'] = update_scm(state.get('scm_s'), observation,
                                                  forgetting[:, :, 0], update[:, :, 0] * masks.float(), c.eps)
            phi_i, new_state['scm_i'] = update_scm(state.get('scm_i'), observation,
                                                  forgetting[:, :, 1], update[:, :, 1] * (1 - masks.float()), c.eps)
            if raw is not None:
                # Deliberately non-PSD A9 control; use a frequency index, never SCM interpolation.
                indices = torch.linspace(0, raw.size(3) - 1, f, device=y.device).round().long()
                phi_s, phi_i = raw[:, :, 0, indices], raw[:, :, 1, indices]
            loaded_i = diagonal_load(phi_i, c.load_scale, c.load_floor)
            d = reference_rtf(phi_s, c.reference_mic, c.eps)
            unit = torch.zeros_like(d)
            unit[..., c.reference_mic] = 1
            w_mvdr = unit if self.stage == 'A' or c.beamformer == 'direct' else mvdr_solve(loaded_i, d, c.eps, c.scm_mode == 'raw_memory')
            phy = physical_features(phi_s, loaded_i, previous_s, masks[..., None, None] * observation, entropy, c.eps)
        joint = None if context is None else torch.cat([context, phy.to(context.dtype)], dim=-1)
        if self.uncertainty_head is not None:
            uncertainty, predicted_alpha = self.uncertainty_head(joint)
        else:
            uncertainty = masks.new_zeros(masks.shape)
            predicted_alpha = masks.new_full(masks.shape, c.fixed_alpha)
        with torch.autocast(device_type=y.device.type, enabled=False):
            uncertainty = uncertainty.float()
            alpha = predicted_alpha.float() if c.uncertainty else torch.full_like(predicted_alpha.float(), c.fixed_alpha)
            delta = torch.zeros_like(d) if self.residual_head is None else self.residual_head(joint.float())
            residual = project_orthogonal(delta, d, c.eps) if c.residual == 'nullspace' else delta
            if c.residual == 'none' or self.stage in {'A', 'B'}:
                alpha = torch.zeros_like(alpha)
            applied = alpha[..., None] * residual
            weights = w_mvdr + applied
            if self.direct_head is not None and self.stage != 'A':
                weights = self.direct_head(joint.float())
            z = (weights.conj() * y[:, None]).sum(-1)
            if self.stage == 'A':
                z = masks.float() * y[:, None, :, c.reference_mic]
            residual_penalty = applied.abs().square().sum(-1) / w_mvdr.abs().square().sum(-1).clamp_min(c.eps)
            constraint = ((weights.conj() * d).sum(-1) - 1).abs()
            auxiliary = None
            if source_images is not None:
                # Oracle memories are target-source indexed, not output-slot indexed.
                oracle_s_obs = outer(source_images.to(torch.complex64))
                interference = y[:, None] - source_images.to(torch.complex64)
                oracle_i_obs = outer(interference)
                shape = source_images.shape[:-1]
                lam = torch.full(shape, c.oracle_lambda, device=y.device)
                ones = torch.ones_like(lam)
                oracle_s, new_state['oracle_s'] = update_scm(state.get('oracle_s'), oracle_s_obs, lam, ones, c.eps)
                oracle_i, new_state['oracle_i'] = update_scm(state.get('oracle_i'), oracle_i_obs, lam, ones, c.eps)
                error = 0.5 * (oracle_error(phi_s[:, :, None], oracle_s[:, None], c.eps)
                               + oracle_error(phi_i[:, :, None], oracle_i[:, None], c.eps))
                target_uncertainty = torch.log(error.detach() + c.eps)
                unc_error = F.huber_loss(uncertainty[:, :, None].expand_as(target_uncertainty), target_uncertainty, reduction='none') if self.uncertainty_head is not None else torch.zeros_like(error)
                auxiliary = {'scm': error.mean(-1), 'uncertainty': unc_error.mean(-1)}
        if self.refiner is not None and self.stage == 'D':
            refined, new_state['refiner'] = self.refiner.step(z, state.get('refiner'))
        else:
            refined = z
        return {'pre_refiner': z, 'post_refiner': refined, 'masks': masks,
                'auxiliary': auxiliary, 'residual_penalty': residual_penalty.mean(dim=(1, 2)),
                'diagnostics': {'lambda_s': forgetting[:, :, 0].mean(-1).detach(),
                                'lambda_i': forgetting[:, :, 1].mean(-1).detach(),
                                'alpha': alpha.mean(-1).detach(), 'uncertainty': uncertainty.mean(-1).detach(),
                                'distortionless_error': constraint.mean(-1).detach(),
                                'mask_entropy': entropy.mean(-1).detach(),
                                'log_condition': phy[..., 0].mean(-1).detach(),
                                'rank_impurity': phy[..., 1].mean(-1).detach(),
                                'log_innovation': phy[..., 3].mean(-1).detach(),
                                **({'covariance_error_pairs': error.mean(-1).detach()} if source_images is not None else {})},
                'state': new_state}

    def forward_spectrum(self, y, doa=None, source_images=None, lengths=None, extraction=False,
                         state=None, diagnostics=False):
        if y.ndim != 4:
            raise ValueError('Expected [B,T,F,M]')
        b, frames, _, _ = y.shape
        if doa is not None and doa.ndim == 2:
            doa = doa.unsqueeze(-1).expand(-1, -1, frames)
        if doa is not None and doa.shape != (b, self.config.num_sources, frames):
            raise ValueError(f'Expected DoA [B,K,T] or [B,K], got {doa.shape}')
        def run_segment(segment, previous, cues, images, valid):
            pre, post, masks, traces = [], [], [], []
            scm_sum = segment.real.new_zeros(b, self.config.num_sources, self.config.num_sources)
            unc_sum = torch.zeros_like(scm_sum)
            residual_sum = segment.real.new_zeros(b)
            for local in range(segment.size(1)):
                result = self.step_spectrum(segment[:, local], previous,
                    None if cues is None else cues[:, :, local],
                    None if images is None else images[:, :, local], extraction)
                previous = result['state']
                pre.append(result['pre_refiner'])
                post.append(result['post_refiner'])
                masks.append(result['masks'])
                residual_sum = residual_sum + result['residual_penalty'] * valid[:, local]
                if result['auxiliary'] is not None:
                    scm_sum = scm_sum + result['auxiliary']['scm'] * valid[:, local, None, None]
                    unc_sum = unc_sum + result['auxiliary']['uncertainty'] * valid[:, local, None, None]
                if diagnostics:
                    traces.append(result['diagnostics'])
            return {'pre': torch.stack(pre, dim=2), 'post': torch.stack(post, dim=2),
                    'masks': torch.stack(masks, dim=2), 'scm': scm_sum, 'unc': unc_sum,
                    'residual': residual_sum, 'state': previous,
                    'traces': {key: torch.stack([d[key] for d in traces], dim=-1) for key in traces[0]} if traces else {}}

        checkpointed = self.training and self.config.checkpoint_frames and torch.is_grad_enabled()
        accelerated = self.config.execution_backend == 'chunkwise' and self.config.scm_mode == 'psd'
        chunk_size = self.config.checkpoint_chunk_size if checkpointed or accelerated else frames
        if accelerated:
            from .chunkwise import run_chunk
            def execute_segment(segment, previous, cues, images, valid):
                return run_chunk(self, segment, previous, cues, images, valid, extraction, diagnostics)
        else:
            execute_segment = run_segment
        validity = torch.ones(b, frames, device=y.device) if lengths is None else (
            torch.arange(frames, device=y.device)[None] * self.config.hop_length < lengths[:, None]).float()
        counts = validity.sum(-1).clamp_min(1)
        chunks = []
        for start in range(0, frames, chunk_size):
            stop = min(start + chunk_size, frames)
            arguments = (y[:, start:stop], state, None if doa is None else doa[:, :, start:stop],
                         None if source_images is None else source_images[:, :, start:stop], validity[:, start:stop])
            # Chunk-level checkpointing retains matrix-memory states only at
            # boundaries; per-frame checkpointing would still store O(T*D^2).
            chunk = checkpoint(execute_segment, *arguments, use_reentrant=False) if checkpointed else execute_segment(*arguments)
            state = chunk.pop('state')
            chunks.append(chunk)
        return {'pre_spectrum': torch.cat([x['pre'] for x in chunks], dim=2),
                'spectrum': torch.cat([x['post'] for x in chunks], dim=2),
                'masks': torch.cat([x['masks'] for x in chunks], dim=2), 'state': state,
                'auxiliary': {'scm': sum(x['scm'] for x in chunks) / counts[:, None, None],
                              'uncertainty': sum(x['unc'] for x in chunks) / counts[:, None, None],
                              'residual': sum(x['residual'] for x in chunks) / counts},
                'diagnostics': {key: torch.cat([x['traces'][key] for x in chunks], dim=-1) for key in chunks[0]['traces']}}

    def forward(self, wav, doa=None, source_images=None, lengths=None, extraction=False,
                return_details=False, diagnostics=False):
        if wav.ndim != 3 or wav.size(1) != self.config.num_channels:
            raise ValueError('Expected waveforms [B,M,N] with configured microphone count')
        with torch.autocast(device_type=wav.device.type, enabled=False):
            y = self.stft.analysis(wav.float())
            oracle = None
            if source_images is not None:
                b, k, m, n = source_images.shape
                if k != self.config.num_sources or m != self.config.num_channels or n != wav.size(-1):
                    raise ValueError('Source images must be [B,K,M,N] matching mixture')
                oracle = self.stft.analysis(source_images.reshape(b * k, m, n)).reshape(b, k, y.size(1), y.size(2), m)
        result = self.forward_spectrum(y, doa, oracle, lengths, extraction, diagnostics=diagnostics)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            result['waveform'] = self.stft.synthesis(result['spectrum'], wav.size(-1))
            result['pre_waveform'] = self.stft.synthesis(result['pre_spectrum'], wav.size(-1))
            result['target_spectrum'] = None if oracle is None else oracle[..., self.config.reference_mic]
            result['mixture_spectrum'] = y[..., self.config.reference_mic]
            result['mixture_waveform'] = wav[:, self.config.reference_mic].detach()
        return result if return_details else result['waveform']
