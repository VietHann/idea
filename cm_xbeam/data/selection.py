"""Select static simulation-direction groups without altering any audio or labels."""
import math


def select_direction_groups(records, stride=1, offset=0):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError('direction_stride must be a positive integer')
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < stride:
        raise ValueError('direction_offset must be an integer in [0, direction_stride)')
    if stride == 1:
        return records, None
    directions = []
    for record in records:
        cue = record.get('doa')
        target = int(record.get('target_index', 0))
        if not isinstance(cue, list) or not 0 <= target < len(cue):
            raise ValueError(f'{record["id"]}: static target DoA is required for direction subsampling')
        angle = cue[target]
        if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not math.isfinite(angle):
            raise ValueError(f'{record["id"]}: direction selection requires a finite static angle, not a trajectory')
        normalized = round((float(angle) + 180.0) % 360.0 - 180.0, 6)
        if normalized >= 180.0:
            normalized = -180.0
        directions.append(normalized)
    grid = sorted(set(directions))
    indices = list(range(offset, len(grid), stride))
    selected = {grid[index] for index in indices}
    kept = [row for row, angle in zip(records, directions) if angle in selected]
    if not kept:
        raise ValueError('Direction selection produced an empty dataset')
    return kept, {'stride': stride, 'offset': offset, 'original_items': len(records),
                  'selected_items': len(kept), 'original_direction_groups': len(grid),
                  'selected_direction_groups': len(selected), 'selected_group_indices': indices,
                  'selected_angles_degrees': [grid[index] for index in indices],
                  'basis': 'sorted static target/source-0 direction groups; not microphone channels'}


def direction_options(data_config, split):
    splits = data_config.get('direction_subset_splits', ['train'])
    if not isinstance(splits, (list, tuple)) or any(s not in {'train', 'valid', 'test'} for s in splits):
        raise ValueError('direction_subset_splits must list train/valid/test')
    if split not in splits:
        return {'direction_stride': 1, 'direction_offset': 0}
    return {'direction_stride': data_config.get('direction_stride', 1),
            'direction_offset': data_config.get('direction_offset', 0)}
