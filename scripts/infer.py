"""Single-image BladeGL inference with optional calibration."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from src.protocol import load_config, resolve
from src.bladegl.backbone import Encoder
from src.bladegl.variants import make_head
from src.bladegl.runner import predict
from src.checkpoints import model_identity, validate_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/bladegl.json')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--arm', default='E0')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--readout', choices=('head', 'map', 'combined'), default='combined')
    parser.add_argument('--readout-weight', type=float, default=.5)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/inference'))
    args = parser.parse_args()
    if not 0 <= args.readout_weight <= 1:
        parser.error('--readout-weight must be in [0,1]')
    if not args.input.is_file():
        parser.error('Input image does not exist')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint does not exist')
    if not torch.cuda.is_available():
        raise RuntimeError('Single-image inference requires CUDA.')
    cfg = load_config(args.config)
    # Load only checkpoints that you created or otherwise trust.
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    identity = model_identity(args.checkpoint, ck, cfg, args.arm)
    mode = 'map' if args.arm == 'R5' else args.readout
    cal = None
    if args.calibration:
        cal = json.loads(args.calibration.read_text(encoding='utf-8'))
        validate_calibration(cal, identity, mode, args.readout_weight)
    device = torch.device('cuda')
    encoder = Encoder(resolve('.'), cfg).to(device).eval()
    head = make_head(args.arm).to(device).eval()
    head.load_state_dict(ck['head'], strict=True)
    row = {'sample_id': args.input.stem, 'image': args.input.name, 'mask': '', 'label': 0}
    scores, maps, shapes, extras = predict(encoder, head, args.input.parent, [row],
        args.arm, mode=mode, weight=args.readout_weight)
    output_dir = args.output_dir / f'{args.input.stem}_{uuid4().hex[:12]}'
    output_dir.mkdir(parents=True, exist_ok=False)
    original_map = cv2.resize(maps[0].astype(np.float32), tuple(reversed(shapes[0])), interpolation=cv2.INTER_NEAREST)
    np.save(output_dir / 'probability_map.npy', original_map, allow_pickle=False)
    cv2.imencode('.png', np.round(original_map * 255).astype(np.uint8))[1].tofile(output_dir / 'probability_map.png')
    report = {'image': args.input.name, 'shape': list(shapes[0]), 'score': scores[0],
              'head_score': extras[0]['head_score'], 'map_top830': extras[0]['map_top830'],
              'readout': mode, 'head_weight': args.readout_weight,
              'model': identity, 'output_dir': str(output_dir)}
    if cal is not None:
        report['defect_present'] = bool(scores[0] >= cal['image_threshold'])
        report['image_threshold'] = cal['image_threshold']
        report['pixel_threshold'] = cal['pixel_threshold']
        mask = (original_map >= cal['pixel_threshold']).astype(np.uint8) * 255
        cv2.imencode('.png', mask)[1].tofile(output_dir / 'defect_mask.png')
    (output_dir / 'prediction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
