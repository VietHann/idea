"""Identical channel calibration perturbations on mixture and source images."""
import numpy as np
from scipy.ndimage import shift


def augment_channels(mixture, sources, rng, reference_mic=0, gain_db=0.0,
                     dropout_probability=0.0, delay_samples=0.0, random_reference=False,
                     **unknown):
    if unknown:
        raise ValueError(f'Unknown augmentation keys: {sorted(unknown)}')
    mixture, sources = mixture.copy(), sources.copy()
    microphones = mixture.shape[0]
    if random_reference:
        selected = int(rng.integers(microphones))
        order = np.arange(microphones)
        order[reference_mic], order[selected] = order[selected], order[reference_mic]
        mixture, sources = mixture[order], sources[:, order]
    gains = 10 ** (rng.uniform(-gain_db, gain_db, microphones) / 20)
    mixture *= gains[:, None]
    sources *= gains[None, :, None]
    if delay_samples:
        delays = rng.uniform(-delay_samples, delay_samples, microphones)
        for channel, delay in enumerate(delays):
            mixture[channel] = shift(mixture[channel], delay, order=3, mode='constant')
            for source in range(sources.shape[0]):
                sources[source, channel] = shift(sources[source, channel], delay, order=3, mode='constant')
    if not 0 <= dropout_probability < 1:
        raise ValueError('Require 0 <= dropout_probability < 1')
    dropped = rng.random(microphones) < dropout_probability
    dropped[reference_mic] = False
    mixture[dropped] = 0
    sources[:, dropped] = 0
    return mixture, sources


def corrupt_doa(doa, rng, bias_deg=0.0, jitter_deg=0.0, delay_frames=0, dropout_fraction=0.0):
    cue = np.asarray(doa, dtype=np.float32).copy()
    cue += bias_deg
    if jitter_deg:
        cue += rng.normal(0, jitter_deg, cue.shape)
    if cue.ndim > 1 and delay_frames:
        delay = min(int(delay_frames), cue.shape[-1])
        cue = np.concatenate([np.repeat(cue[..., :1], delay, axis=-1), cue[..., :cue.shape[-1] - delay]], axis=-1)
    if dropout_fraction:
        if not 0 <= dropout_fraction <= 1:
            raise ValueError('dropout_fraction must be in [0,1]')
        cue[rng.random(cue.shape) < dropout_fraction] = np.nan
    return ((cue + 180) % 360 - 180).astype(np.float32)
