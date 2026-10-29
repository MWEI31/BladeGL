"""Download and verify only the official frozen DINOv2 checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'weights/dinov2_vits14_pretrain.pth')
    args = parser.parse_args()
    reference = json.loads((ROOT / 'docs/backbone_reference.json').read_text())
    if args.output.exists():
        if digest(args.output) != reference['sha256']:
            raise RuntimeError('Existing weight file does not match the experiment reference; move it aside first.')
        print('Existing backbone SHA-256 matches the experiment reference.')
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.download')
    urllib.request.urlretrieve(reference['url'], temporary)
    if digest(temporary) != reference['sha256']:
        raise RuntimeError('Downloaded checkpoint hash mismatch; temporary file retained for inspection.')
    os.replace(temporary, args.output)
    print('Backbone downloaded and SHA-256 verified.')


if __name__ == '__main__':
    main()
