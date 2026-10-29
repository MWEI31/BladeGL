"""Audit fixed data roles and optional input files before training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.protocol import load_config, read_manifest, resolve, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/bladegl.json')
    parser.add_argument('--dataset', choices=('aebad', 'ksdd2'), required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--verify-files', action='store_true')
    parser.add_argument('--verify-hashes', action='store_true', help='Verify recorded image hashes when available.')
    parser.add_argument('--verify-backbone', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = read_manifest(args.dataset, cfg)
    report = {'dataset': args.dataset, 'normalized_rows': len(rows), 'roles': {},
              'manifest_sha256': sha256(resolve(cfg[args.dataset]['manifest']))}
    for role in ('fit', 'dev', 'cal', 'test'):
        selected = [r for r in rows if r['role'] in cfg[args.dataset][role + '_roles']]
        report['roles'][role] = dict(Counter('anomalous' if r['label'] else 'normal' for r in selected))
    if args.verify_files or args.verify_hashes:
        if args.data_root is None:
            parser.error('--data-root is required with file or hash verification')
        failures = []
        for row in rows:
            image = args.data_root / row['image']
            if not image.is_file():
                failures.append(row['sample_id'] + ':missing_image')
            elif args.verify_hashes and row.get('image_sha256') and sha256(image) != row['image_sha256']:
                failures.append(row['sample_id'] + ':image_hash_mismatch')
            if row['label'] and not (args.data_root / row['mask']).is_file():
                failures.append(row['sample_id'] + ':missing_mask')
        report['file_checks_passed'] = not failures
        report['failures'] = failures
    if args.verify_backbone:
        ref = json.loads((ROOT / 'docs/backbone_reference.json').read_text())
        path = resolve(cfg['dino_weights'])
        report['backbone_matches'] = path.is_file() and sha256(path) == ref['sha256']
    print(json.dumps(report, indent=2))
    if report.get('failures') or report.get('backbone_matches') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
