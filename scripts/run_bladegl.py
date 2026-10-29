#!/usr/bin/env python3
"""Run BladeGL source and structural/task ablations on one GPU."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protocol import load_config  # noqa: E402
from src.bladegl.runner import train, evaluate  # noqa: E402
from src.bladegl.variants import make_head  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('train', 'evaluate', 'all'))
    p.add_argument('--config', default='configs/bladegl.json')
    p.add_argument('--dataset', choices=('aebad', 'ksdd2'), required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--budget', type=float, default=1.)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--readout', choices=('combined', 'head', 'map'), default='combined')
    p.add_argument('--readout-weight', type=float, default=.5)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed not in cfg['seeds'] or not 0 < args.budget <= 1:
        p.error('Seed or budget is outside the locked protocol')
    if not 0 <= args.readout_weight <= 1:
        p.error('Readout weight must be in [0,1]')
    make_head(args.arm)
    if not args.data_root.is_dir():
        p.error(f'Missing data directory: {args.data_root}')
    import torch
    device = torch.device('cuda')
    if args.mode in ('train', 'all'):
        train(args.dataset, args.arm, args.seed, args.budget, cfg,
              args.data_root.resolve(), device)
    if args.mode in ('evaluate', 'all'):
        result = evaluate(args.dataset, args.arm, args.seed, args.budget, cfg,
                          args.data_root.resolve(), device,
                          args.readout, args.readout_weight)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
