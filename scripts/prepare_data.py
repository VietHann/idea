"""Prepare local manifests without copying or modifying either source HDF5 dataset."""
import argparse
import json
from pathlib import Path
from cm_xbeam.config import load_config
from cm_xbeam.data.manifests import prepare_lrs2, prepare_librispeech, split_audit
from cm_xbeam.utils import safe_output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--dataset', choices=['lrs2', 'librispeech'])
    p.add_argument('--root')
    p.add_argument('--output')
    p.add_argument('--speakers', type=int, default=None)
    p.add_argument('--limit', type=int, help='Optional bounded subset, explicitly unsuitable for final research results')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--strict-speaker-splits', action='store_true')
    p.add_argument('--strict-room-splits', action='store_true')
    p.add_argument('--audit-only', action='store_true', help='Read existing local manifests without regenerating them')
    args = p.parse_args()
    cfg = load_config(args.config) if args.config else {}
    dataset = args.dataset or cfg.get('dataset', '').split('_')[0]
    if dataset not in {'lrs2', 'librispeech'}:
        raise ValueError('Specify --dataset lrs2/librispeech or an appropriate --config')
    root = args.root or cfg.get('data', {}).get('root')
    if not root:
        raise ValueError('Dataset root is required')
    output = args.output or str(Path(cfg.get('data', {}).get('train_manifest', f'manifests/{dataset}/train.jsonl')).parent)
    output = safe_output_path(output)
    if args.limit is not None and args.limit <= 0:
        raise ValueError('limit must be positive')
    if args.audit_only:
        result = {split: {'manifest': str(output / f'{split}.jsonl')} for split in ['train', 'valid', 'test']}
    elif dataset == 'lrs2':
        speakers = args.speakers or cfg.get('model', {}).get('num_sources', 2)
        # Extraction queries still load the original complete two-speaker mixture.
        speakers = max(2, speakers)
        result = prepare_lrs2(root, output, speakers, args.limit, args.overwrite)
    else:
        result = prepare_librispeech(root, output, args.limit, args.overwrite)
    audit = split_audit(result)
    if args.strict_speaker_splits and any(x['speaker_overlap_count'] or x['mixture_id_overlap_count'] for x in audit.values()):
        raise ValueError(f'Existing dataset split leakage detected: {audit}; source dataset was left unchanged')
    if args.strict_room_splits and any(x['room_geometry_overlap_count'] for x in audit.values()):
        raise ValueError(f'Existing room geometry overlap detected: {audit}; keep this recipe separate from held-out-room claims')
    print(json.dumps({'dataset': dataset, 'splits': result, 'existing_split_audit': audit,
                      'bounded_subset': args.limit is not None, 'source_files_modified': False}, indent=2))


if __name__ == '__main__':
    main()
