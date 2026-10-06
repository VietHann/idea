"""Normalize the actual LRS2 and LibriSpeech HDF5 schemas into read-only manifests."""
import json
import math
from pathlib import Path
import h5py
from ..utils import safe_output_path


def _metadata(row, dataset):
    if dataset == 'lrs2':
        sources = row['sources']
        angles = [s['angle_deg_relative_to_array'] for s in sources]
        source_files = [s['path'] for s in sources]
        groups = [s.get('source_group_id') for s in sources]
        room = row['room']
        result = {'rt60': room.get('rt60_target_s'), 'room_dimensions': room.get('dimensions_m'),
                  'microphone_positions': row['microphone_array']['positions_m'],
                  'array_rotation_rad': row['microphone_array'].get('rotation_rad'), 'sample_seed': row['seed']}
    else:
        meta = row.get('metadata', row)
        center = meta['mic_pos']
        rotation = meta['mic_phi']
        positions = [meta['target_pos'], meta['interf0_pos']]
        angles = [((math.degrees(math.atan2(pos[1] - center[1], pos[0] - center[0]) - rotation) + 180) % 360) - 180 for pos in positions]
        source_files = [meta['target_file'], meta['interf0_file']]
        # Isolated LibriMix s1/s2 files retain the two-utterance mixture basename.
        groups = []
        for file in source_files:
            tokens = Path(file).stem.split('_')
            which = 1 if Path(file).parent.name == 's2' and len(tokens) > 1 else 0
            groups.append(tokens[which].split('-')[0])
        result = {'rt60': meta['rt'], 'room_dimensions': meta['room_dim'],
                  'array_center': center, 'array_rotation_rad': rotation}
    separation = abs((angles[0] - angles[1] + 180) % 360 - 180) if len(angles) > 1 else None
    result.update({'source_angles': angles, 'source_audio': source_files, 'speaker_ids': groups,
                   'motion_scenario': 'stationary', 'angular_velocity': 0.0, 'angular_separation': separation})
    return angles, result


def prepare_lrs2(root, output, speakers=2, limit=None, overwrite=False):
    root, output = Path(root).resolve(), safe_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, directory in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        folder = root / f'{speakers}speakers' / directory
        hdf = folder / 'audio.hdf5'
        metadata = folder / 'metadata.jsonl'
        target = safe_output_path(output / f'{name}.jsonl')
        if target.exists() and not overwrite:
            raise FileExistsError(f'{target} already exists; use --overwrite intentionally')
        with h5py.File(hdf, 'r') as container:
            rate = int(container.attrs['sample_rate_hz'])
            microphones = int(container.attrs['n_channels'])
            images = container['reverberant_sources']
            mixture = container['mixture']
            total = int(container.attrs.get('completed_samples', mixture.shape[0]))
            if images.shape[:3] != (mixture.shape[0], speakers, microphones) or mixture.shape[1] != microphones:
                raise ValueError('LRS2 source image schema mismatch')
            total = min(total, limit) if limit is not None else total
            seen = set()
            with metadata.open() as source, target.open('w') as destination:
                for line in source:
                    row = json.loads(line)
                    index = int(row['index'])
                    if index >= total:
                        continue
                    if index in seen:
                        raise ValueError(f'Duplicate LRS2 metadata index {index}')
                    seen.add(index)
                    if int(row['sample_rate_hz']) != rate or int(row['n_channels']) != microphones:
                        raise ValueError('Metadata/HDF5 sample-rate or channel mismatch')
                    angles, meta = _metadata(row, 'lrs2')
                    record = {'id': row['sample_id'], 'split': name, 'sample_rate': rate,
                              'num_samples': row['valid_samples'], 'target_kind': 'reverberant',
                              'mixture': {'path': str(hdf), 'dataset': 'mixture', 'index': index},
                              'sources': [{'path': str(hdf), 'dataset': 'reverberant_sources', 'index': index, 'source_index': k} for k in range(speakers)],
                              'target_index': 0, 'doa': angles, 'metadata': meta}
                    destination.write(json.dumps(record, allow_nan=False) + '\n')
                    if len(seen) == total:
                        break
            if len(seen) != total:
                raise ValueError(f'LRS2 split {name} has {len(seen)} metadata rows but expects {total}')
            result[name] = {'manifest': str(target), 'items': total, 'sample_rate': rate, 'channels': microphones, 'sources': speakers}
    return result


def prepare_librispeech(root, output, limit=None, overwrite=False):
    root, output = Path(root).resolve(), safe_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    hdf = root / 'prep_mix_ch3_sp1_var_target.hdf5'
    launch = json.loads((root / 'prep_mix_ch3_sp1_var_target.launch.json').read_text())
    if int(launch['upstream_recipe']['n_interfering_speakers']) != 1:
        raise ValueError('Individual separation targets cannot be recovered from a sum of multiple interferers')
    rate = int(launch['native_sample_rate'])
    result = {}
    with h5py.File(hdf, 'r') as container:
        for name, group in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            array = container[group]
            if array.ndim != 4 or array.shape[1:3] != (3, 3):
                raise ValueError('Expected official LibriSpeech [N,3_components,3_microphones,L]')
            total = int(container.attrs.get(f'completed_{group}', array.shape[0]))
            total = min(total, limit) if limit is not None else total
            target = safe_output_path(output / f'{name}.jsonl')
            if target.exists() and not overwrite:
                raise FileExistsError(f'{target} already exists; use --overwrite intentionally')
            metadata = root / f'prep_mix_ch3_sp1_var_target.{group}.jsonl'
            seen = set()
            with metadata.open() as source, target.open('w') as destination:
                for line in source:
                    row = json.loads(line)
                    index = int(row['index'])
                    if index >= total:
                        continue
                    if index in seen:
                        raise ValueError(f'Duplicate LibriSpeech metadata index {index}')
                    seen.add(index)
                    angles, meta = _metadata(row, 'librispeech')
                    samples = min(int(row['metadata']['n_samples']), array.shape[-1])
                    record = {'id': f'{group}_{index:06d}', 'split': name, 'sample_rate': rate,
                              'num_samples': samples, 'target_kind': 'reverberant',
                              'mixture': {'path': str(hdf), 'dataset': group, 'index': index, 'components': [0, 1]},
                              'sources': [{'path': str(hdf), 'dataset': group, 'index': index, 'component': k} for k in [0, 1]],
                              'target_index': 0, 'doa': angles, 'metadata': meta}
                    destination.write(json.dumps(record, allow_nan=False) + '\n')
                    if len(seen) == total:
                        break
            if len(seen) != total:
                raise ValueError(f'LibriSpeech split {name} has {len(seen)} metadata rows but expects {total}')
            result[name] = {'manifest': str(target), 'items': total, 'sample_rate': rate, 'channels': 3, 'sources': 2,
                            'mixture_definition': 'reverberant target component0 + interferer component1; component2 is NOT a speaker'}
    return result


def split_audit(manifests):
    speakers = {}
    ids = {}
    rooms = {}
    for split, info in manifests.items():
        speakers[split], ids[split], rooms[split] = set(), set(), set()
        with Path(info['manifest']).open() as handle:
            for line in handle:
                row = json.loads(line)
                ids[split].add(row['id'])
                metadata = row.get('metadata', {})
                speakers[split].update(s for s in metadata.get('speaker_ids', []) if s is not None)
                if metadata.get('room_dimensions') is not None:
                    geometry = [round(float(x), 8) for x in metadata['room_dimensions']]
                    geometry.append(round(float(metadata.get('rt60', 0)), 8))
                    rooms[split].add(tuple(geometry))
    overlaps = {}
    splits = list(speakers)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlaps[f'{a}:{b}'] = {'speaker_overlap_count': len(speakers[a] & speakers[b]),
                                     'mixture_id_overlap_count': len(ids[a] & ids[b]),
                                     'room_geometry_overlap_count': len(rooms[a] & rooms[b])}
    return overlaps
