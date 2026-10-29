"""Model, metric, and dataset protocol checks."""
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import geometry
from src.metrics import ap_from_hist, grid_counts, evaluate_maps, score_buckets
from src.protocol import load_config, read_manifest, resolve
from src.bladegl.variants import make_head, loss_value
from src.bladegl.runner import readout, load_image_and_target, map_to_grid

torch.set_num_threads(min(2, torch.get_num_threads()))


@pytest.mark.parametrize('dataset,counts', [
    ('aebad', (602, 158, 158, 1399)), ('ksdd2', (1865, 233, 233, 1004))])
def test_fixed_manifest_counts_and_no_fit_test_overlap(dataset, counts):
    cfg = load_config('configs/bladegl.json')
    if not resolve(cfg[dataset]['manifest']).is_file():
        pytest.skip('Dataset manifest is not provided.')
    rows = read_manifest(dataset, cfg)
    groups = [{r['sample_id'] for r in rows if r['role'] in cfg[dataset][role + '_roles']}
              for role in ('fit', 'dev', 'cal', 'test')]
    assert tuple(map(len, groups)) == counts
    assert not groups[0] & groups[3]
    assert not groups[1] & groups[3]
    if dataset == 'ksdd2':
        assert not groups[1] & groups[2]
    assert all('\\' not in row['image'] for row in rows)


def test_exact_original_pixel_counts(tmp_path):
    mask = np.zeros((5, 7), np.uint8)
    mask[1:4, 2:5] = 255
    cv2.imencode('.png', mask)[1].tofile(tmp_path / 'mask.png')
    pos, area = grid_counts(tmp_path, {'sample_id': 'one', 'label': 1, 'mask': 'mask.png'}, (2, 3), (5, 7))
    assert pos.sum() == 9 and area.sum() == 35
    assert np.all(pos <= area)


def test_signed_fp16_order_and_perfect_hist_ap():
    buckets = score_buckets(np.array([-3, -1, -.0, .0, .5, 2], np.float16))
    assert np.all(np.diff(buckets.astype(np.int32)) >= 0)
    pos, area = np.zeros(65536), np.zeros(65536)
    pos[15360] = area[15360] = 2
    area[0] = 2
    assert ap_from_hist(pos, area) == 1


def test_aupro_original_eight_connected_regions(tmp_path):
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = mask[1, 1] = 255
    cv2.imencode('.png', mask)[1].tofile(tmp_path / 'mask.png')
    rows = [{'sample_id': 'a', 'label': 1, 'mask': 'mask.png'},
            {'sample_id': 'n', 'label': 0, 'mask': ''}]
    result = evaluate_maps(tmp_path, rows,
        [(mask > 0).astype(np.float16), np.zeros((4, 4), np.float16)], [(4, 4), (4, 4)], .5, tmp_path / 'curves.npz')
    assert result['aupro_region_count'] == 1
    assert abs(result['aupro_030'] - 1) < 1e-12


@pytest.mark.parametrize('arm,expected', [('F0', 376642), ('E0', 388383), ('E01_CBAM', 381554)])
def test_paper_parameter_counts_and_backward(arm, expected):
    torch.manual_seed(91025)
    model = make_head(arm)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected
    features = {'global_features': torch.randn(1, 1152, 37, 37),
                'tiles': torch.randn(1, 4, 1152, 37, 37), 'cls': torch.randn(1, 384)}
    z, score = model(features)
    assert z.shape == (1, 288, 288) and score.shape == (1,)
    target = torch.zeros_like(z)
    target[:, 50:55, 50:55] = 1
    loss = loss_value(arm, z, score, target, torch.ones_like(z), torch.ones(1))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.segment.weight.grad is not None
    assert torch.isfinite(model.segment.weight.grad).all()


def test_logit_readout_matches_geometric_odds():
    grid = np.full((288, 288), .2, np.float16)
    raw = torch.tensor(np.log(.8 / .2))
    joint, head, top = readout(grid, raw)
    exact = 1 / (1 + np.exp(-.5 * (np.log(head/(1-head)) + np.log(top/(1-top)))))
    assert abs(joint - exact) < 1e-6
    assert abs(readout(grid, raw, 'head')[0] - .8) < 1e-6
    assert abs(readout(grid, raw, 'map')[0] - top) < 1e-6


def test_non_square_letterbox_and_restoration(tmp_path):
    image = np.full((645, 229, 3), 127, np.uint8)
    cv2.imencode('.png', image)[1].tofile(tmp_path / 'sample.png')
    row = {'image': 'sample.png', 'sample_id': 'sample', 'label': 0, 'mask': ''}
    rgb, target, valid, meta = load_image_and_target(tmp_path, row)
    assert rgb.shape == (1008, 1008, 3)
    assert target.shape == valid.shape == (288, 288)
    assert 0 < valid.sum() < 288 * 288
    grid = map_to_grid(np.full((288, 288), .5, np.float16), meta)
    assert grid.shape == (288, round(288 * 229 / 645))
    assert np.all(grid == .5)


def test_optional_frozen_encoder_cuda():
    weight_path = os.environ.get('BLADEGL_TEST_BACKBONE')
    if not weight_path or not torch.cuda.is_available():
        pytest.skip('Set BLADEGL_TEST_BACKBONE to official weights on a CUDA host.')
    from src.bladegl.backbone import Encoder
    cfg = load_config('configs/bladegl.json')
    cfg['dino_weights'] = weight_path
    encoder = Encoder(ROOT, cfg).cuda().eval()
    feat = encoder.extract(np.zeros((1008, 1008, 3), np.uint8), include_direct=False)
    assert feat['global_features'].shape == (1, 1152, 37, 37)
    assert feat['tiles'].shape == (1, 4, 1152, 37, 37)
    assert feat['cls'].shape == (1, 384)
    assert not any(p.requires_grad for p in encoder.parameters())
    head = make_head('E0').cuda().eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        z, score = head(feat)
    assert z.shape == (1, 288, 288) and torch.isfinite(z).all()
    assert torch.isfinite(score).all()
