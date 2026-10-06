"""Read-only multichannel WAV/HDF5 manifests for LRS2 and spatial LibriSpeech."""
import hashlib
import json
import math
import os
from pathlib import Path
import h5py
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset
from .augmentation import augment_channels, corrupt_doa
from .selection import select_direction_groups


class MultichannelDataset(Dataset):
    def __init__(self, manifest, sample_rate, num_channels, num_sources, reference_mic=0,
                 crop_seconds=None, training=False, task='separation', augmentation=None, seed=0,
                 direction_stride=1, direction_offset=0):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open() as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f'Empty manifest: {self.manifest}')
        ids = [row['id'] for row in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate mixture IDs in manifest')
        self.records, self.direction_selection = select_direction_groups(self.records, direction_stride, direction_offset)
        self.sample_rate, self.num_channels, self.num_sources = sample_rate, num_channels, num_sources
        self.reference_mic, self.crop_seconds, self.training = reference_mic, crop_seconds, training
        self.task, self.augmentation, self.seed, self.epoch = task, dict(augmentation or {}), seed, 0
        if task not in {'separation', 'extraction'}:
            raise ValueError('task must be separation or extraction')
        if task == 'extraction' and num_sources != 1:
            raise ValueError('Single-target extraction requires num_sources=1')
        self._handles, self._pid = {}, os.getpid()

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_handles'] = {}
        return state

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __del__(self):
        if hasattr(self, '_handles'):
            self.close()

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def _hdf5(self, path):
        if os.getpid() != self._pid:
            self._handles, self._pid = {}, os.getpid()
        key = str(path.resolve())
        if key not in self._handles:
            if len(self._handles) >= 4:
                first = next(iter(self._handles))
                self._handles.pop(first).close()
            self._handles[key] = h5py.File(path, 'r')
        return self._handles[key]

    def _read(self, reference, expected_rate=None):
        if isinstance(reference, list):
            parts = [self._read(item, expected_rate) for item in reference]
            if any(x.shape[0] != 1 for x in parts) or len({x.shape[-1] for x in parts}) != 1:
                raise ValueError('A microphone path list must contain equal-length mono WAVs')
            return np.concatenate(parts, axis=0)
        if isinstance(reference, str):
            audio, rate = sf.read(self._path(reference), dtype='float32', always_2d=True)
            if expected_rate is not None and rate != expected_rate:
                raise ValueError(f'WAV/manifest sample-rate mismatch for {reference}: {rate} != {expected_rate}')
            return audio.T.copy()
        if not isinstance(reference, dict):
            raise TypeError('Audio reference must be a WAV path, channel list or HDF5 descriptor')
        if 'dataset' not in reference:
            return self._read(reference['path'], expected_rate)
        dataset = self._hdf5(self._path(reference['path']))[reference['dataset']]
        index = int(reference['index'])
        if 'components' in reference:
            # LibriSpeech stores target and interferer, not their mixture.
            parts = [np.asarray(dataset[index, component], dtype=np.float32) for component in reference['components']]
            audio = np.sum(parts, axis=0, dtype=np.float32)
        elif 'component' in reference:
            audio = np.asarray(dataset[index, int(reference['component'])], dtype=np.float32)
        elif 'source_index' in reference:
            audio = np.asarray(dataset[index, int(reference['source_index'])], dtype=np.float32)
        else:
            audio = np.asarray(dataset[index], dtype=np.float32)
        if audio.ndim != 2:
            raise ValueError(f'Expected HDF5 channel-first [M,N], got {audio.shape}')
        return audio

    def __getitem__(self, index):
        row = self.records[index]
        native_rate = int(row['sample_rate'])
        mixture = self._read(row['mixture'], native_rate)
        sources = np.stack([self._read(ref, native_rate) for ref in row['sources']], axis=0)
        if mixture.shape[0] != self.num_channels or sources.shape[1:] != mixture.shape:
            raise ValueError(f'{row["id"]}: source/mixture channel or length mismatch: {mixture.shape}, {sources.shape}')
        valid = int(row.get('num_samples', mixture.shape[-1]))
        if not 0 < valid <= mixture.shape[-1]:
            raise ValueError(f'{row["id"]}: invalid num_samples={valid}')
        mixture, sources = mixture[..., :valid], sources[..., :valid]
        if native_rate != self.sample_rate:
            common = math.gcd(native_rate, self.sample_rate)
            mixture = resample_poly(mixture, self.sample_rate // common, native_rate // common, axis=-1).astype(np.float32)
            sources = resample_poly(sources, self.sample_rate // common, native_rate // common, axis=-1).astype(np.float32)
        if not np.isfinite(mixture).all() or not np.isfinite(sources).all():
            raise ValueError(f'{row["id"]}: nonfinite input audio')
        selected = int(row.get('target_index', 0))
        doa = row.get('doa')
        if doa is not None:
            doa = np.asarray(doa, dtype=np.float32)
            if doa.shape[0] != sources.shape[0]:
                raise ValueError(f'{row["id"]}: DoA/source count mismatch')
        if self.task == 'extraction':
            sources = sources[selected:selected + 1]
            doa = None if doa is None else doa[selected:selected + 1]
        if sources.shape[0] != self.num_sources:
            raise ValueError(f'{row["id"]}: expected {self.num_sources} source images, got {sources.shape[0]}')
        stable = int.from_bytes(hashlib.sha256(str(row['id']).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, stable % 2**32]))
        original_length = mixture.shape[-1]
        crop = int(round(self.crop_seconds * self.sample_rate)) if self.crop_seconds else original_length
        if crop <= 0:
            raise ValueError('crop_seconds must be positive')
        start = int(rng.integers(original_length - crop + 1)) if self.training and original_length > crop else 0
        length = min(crop, original_length - start)
        mixture, sources = mixture[..., start:start + length].copy(), sources[..., start:start + length].copy()
        if doa is not None and doa.ndim == 2:
            rate = float(row.get('doa_sample_rate', doa.shape[-1] / (original_length / self.sample_rate)))
            times = (np.arange(start, start + length, dtype=np.float64) / self.sample_rate) * rate
            grid = np.arange(doa.shape[-1], dtype=np.float64)
            sampled = []
            for trajectory in doa:
                finite = np.isfinite(trajectory)
                angle = np.deg2rad(np.nan_to_num(trajectory))
                sine = np.interp(times, grid, np.sin(angle))
                cosine = np.interp(times, grid, np.cos(angle))
                validity = finite[np.rint(times).astype(int).clip(0, len(trajectory) - 1)]
                sampled.append(np.where(validity, np.rad2deg(np.arctan2(sine, cosine)), np.nan))
            doa = np.asarray(sampled, dtype=np.float32)
        if self.training and self.augmentation:
            augmentation = dict(self.augmentation)
            cue_noise = augmentation.pop('doa', None)
            mixture, sources = augment_channels(mixture, sources, rng, self.reference_mic, **augmentation)
            if doa is not None and cue_noise:
                doa = corrupt_doa(doa, rng, **cue_noise)
        metadata = dict(row.get('metadata') or {})
        metadata.update({'crop_start_samples': start, 'native_sample_rate': native_rate,
                         'target_kind': row.get('target_kind', 'reverberant'), 'split': row.get('split')})
        result = {'mixture': torch.from_numpy(mixture.astype(np.float32)),
                  'source_images': torch.from_numpy(sources.astype(np.float32)),
                  'targets': torch.from_numpy(sources[:, self.reference_mic].astype(np.float32)),
                  'lengths': torch.tensor(length, dtype=torch.long), 'id': row['id'],
                  'metadata': metadata, 'task': self.task}
        if doa is not None:
            result['doa'] = torch.from_numpy(doa.copy())
        return result


def collate_multichannel(examples):
    if not examples:
        raise ValueError('Empty batch')
    tasks = {item['task'] for item in examples}
    if len(tasks) != 1:
        raise ValueError('A batch cannot mix extraction and separation')
    max_length = max(int(item['lengths']) for item in examples)
    result = {key: torch.stack([torch.nn.functional.pad(item[key], (0, max_length - item[key].size(-1))) for item in examples])
              for key in ['mixture', 'source_images', 'targets']}
    result.update({'lengths': torch.stack([item['lengths'] for item in examples]),
                   'id': [item['id'] for item in examples], 'metadata': [item['metadata'] for item in examples],
                   'task': next(iter(tasks))})
    cues = [item.get('doa') for item in examples]
    if all(cue is not None for cue in cues):
        if all(cue.ndim == 1 for cue in cues):
            result['doa'] = torch.stack(cues)
        else:
            # Align each trajectory to the padded waveform's sample grid. The
            # runner later samples circular cues at the actual STFT frame times.
            aligned = []
            for item, cue in zip(examples, cues):
                if cue.ndim == 1:
                    cue = cue.unsqueeze(-1)
                length = int(item['lengths'])
                angles = torch.deg2rad(torch.nan_to_num(cue))
                sine = torch.nn.functional.interpolate(angles.sin()[None], size=length, mode='linear', align_corners=False)[0]
                cosine = torch.nn.functional.interpolate(angles.cos()[None], size=length, mode='linear', align_corners=False)[0]
                valid = torch.nn.functional.interpolate(torch.isfinite(cue).float()[None], size=length, mode='nearest')[0] > 0.5
                angle = torch.where(valid, torch.rad2deg(torch.atan2(sine, cosine)), torch.full_like(sine, float('nan')))
                aligned.append(torch.nn.functional.pad(angle, (0, max_length - length), value=float('nan')))
            result['doa'] = torch.stack(aligned)
    elif examples[0]['task'] == 'extraction':
        raise ValueError('Every extraction example must have a DoA cue')
    return result
