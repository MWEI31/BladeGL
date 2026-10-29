"""Build a dataset manifest from an existing sample assignment table."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.protocol import load_config, read_manifest, resolve, sha256


def data_file(root, value):
    path = Path(value.replace('\\', '/'))
    if not value or path.is_absolute():
        raise ValueError('Dataset file paths must be relative and non-empty.')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Dataset file path is outside --data-root.')
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/bladegl.json')
    parser.add_argument('--dataset', choices=('aebad', 'ksdd2'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--assignments', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = args.data_root.resolve()
    if not root.is_dir():
        parser.error('--data-root must be an existing directory')
    output = args.output.resolve() if args.output else resolve(cfg[args.dataset]['manifest'])
    if output.exists():
        parser.error('Output already exists; choose a new --output path')
    with args.assignments.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'role', 'image', 'mask', 'label'} <= set(reader.fieldnames or []):
            parser.error('Assignment columns must include role,image,mask,label')
        assignments = list(reader)
    if not assignments:
        parser.error('Assignment table is empty')
    allowed = set().union(*(cfg[args.dataset][r + '_roles']
                            for r in ('fit', 'dev', 'cal', 'test')))
    rows, seen = [], {}
    for item in assignments:
        role = item['role'].strip()
        label = int(item['label'])
        if role not in allowed or label not in (0, 1):
            raise ValueError('Invalid assignment role or label')
        image = item['image'].strip().replace('\\', '/')
        mask = item['mask'].strip().replace('\\', '/')
        image_hash = sha256(data_file(root, image))
        if label and not mask:
            raise ValueError('Anomalous images require a mask path')
        if mask:
            data_file(root, mask)
        assignment = (role, label, mask)
        if image_hash in seen:
            if seen[image_hash] != assignment:
                raise ValueError('Duplicate image content has conflicting assignments')
            continue
        seen[image_hash] = assignment
        sample_id = item.get('sample_id', '').strip() or f'{args.dataset}_{image_hash[:24]}'
        common = {'sample_id': sample_id, 'label': label,
                  'condition': item.get('condition', ''),
                  'defect_type': item.get('defect_type', ''),
                  'image_sha256': image_hash}
        if args.dataset == 'aebad':
            row = {**common, 'protocol_role': role,
                   'relative_image_path': image, 'relative_mask_path': mask}
        else:
            row = {**common, 'role': role, 'image': image, 'mask': mask}
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=output.parent, suffix='.csv', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        check_cfg = {**cfg, args.dataset: {**cfg[args.dataset], 'manifest': str(temporary)}}
        normalized = read_manifest(args.dataset, check_cfg)
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
        print(json.dumps({'manifest': str(output), 'rows': len(normalized)}, ensure_ascii=False))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()
