"""BladeGL training and evaluation with frozen DINOv2 features.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from ..data import geometry, imread, unletterbox
from ..runtime import save_json, seed_all
from ..metrics import evaluate_maps, image_metrics, image_threshold_recall_floor, select_pixel_threshold
from ..protocol import read_manifest, resolve, sha256
from ..cost import profile_rgb_path
from ..artifacts import readout_artifact
from .backbone import Encoder
from .variants import loss_value, make_head, required_views


def role_rows(dataset, cfg, role):
    return [r for r in read_manifest(dataset, cfg)
            if r['role'] in cfg[dataset][f'{role}_roles']]


def budget_rows(rows, fraction, seed):
    """Nested anomaly sets; all normal training images remain available."""
    normal = [r for r in rows if not r['label']]
    anomalous = [r for r in rows if r['label']]
    if fraction == 1:
        return rows
    # Stable, stratified order within available defect/condition strata.
    groups = {}
    for row in anomalous:
        key = (row.get('defect_type', ''), row.get('condition', ''))
        groups.setdefault(key, []).append(row)
    ordered_groups = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x['sample_id'])
        random.Random(f'{seed}:{key}').shuffle(group)
        ordered_groups.append(group)
    sequence = []
    # The fixed round-robin order gives nested 25/50/100% sets with exact
    # global counts, while spreading early selections across known strata.
    for index in range(max(map(len, ordered_groups))):
        for group in ordered_groups:
            if index < len(group):
                sequence.append(group[index])
    target = math.ceil(len(anomalous) * fraction)
    return normal + sequence[:target]


def load_image_and_target(root: Path, row: dict, read_mask=True):
    image = imread(root / row['image'], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(root / row['image'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    ih, iw, it, il = geometry(h, w, 1008, 1008)
    square = np.full((1008, 1008, 3), [124, 116, 104], np.uint8)
    square[it:it+ih, il:il+iw] = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA)
    mask = np.zeros((h, w), np.float32)
    if row['label'] and read_mask:
        raw = imread(root / row['mask'], cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (h, w) or raw.max() == 0:
            raise ValueError(f'Invalid mask: {row["sample_id"]}')
        mask = (raw > 0).astype(np.float32)
    gh, gw, gt, gl = geometry(h, w, 288, 288)
    target = np.zeros((288, 288), np.float32)
    target[gt:gt+gh, gl:gl+gw] = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_AREA)
    valid = np.zeros_like(target)
    valid[gt:gt+gh, gl:gl+gw] = 1
    return square, torch.from_numpy(target), torch.from_numpy(valid), (h, w, gh, gw, gt, gl)


def encode_rows(encoder, root, rows, arm):
    direct, tiles, global_view = required_views(arm)
    features, targets, valids, labels = [], [], [], []
    for row in rows:
        rgb, target, valid, _ = load_image_and_target(root, row)
        with torch.inference_mode():
            encoded = encoder.extract(rgb, include_direct=direct,
                                      include_tiles=tiles, include_global=global_view)
        # Frozen features must be ordinary tensors for head autograd to save.
        features.append({key: value.clone() for key, value in encoded.items()})
        targets.append(target)
        valids.append(valid)
        labels.append(float(row['label']))
    batch = {key: torch.cat([feature[key] for feature in features], dim=0)
             for key in features[0]}
    device = encoder.mean.device
    return batch, torch.stack(targets).to(device), torch.stack(valids).to(device), \
        torch.tensor(labels, dtype=torch.float32, device=device)


def map_to_grid(square_map, meta):
    h, w, gh, gw, gt, gl = meta
    if h == w:
        return square_map.astype(np.float16)
    projected = unletterbox(square_map, (h, w, gh, gw, gt, gl))
    out_h = max(1, round(288 * h / max(h, w)))
    out_w = max(1, round(288 * w / max(h, w)))
    return cv2.resize(projected, (out_w, out_h),
                      interpolation=cv2.INTER_LINEAR).astype(np.float16)


def readout(map_grid, head_logit, mode='combined', weight=.5):
    flat = np.asarray(map_grid, np.float32).ravel()
    top = float(np.partition(flat, len(flat)-min(830, len(flat)))[-min(830, len(flat)):].mean())
    head = float(torch.sigmoid(torch.as_tensor(head_logit, dtype=torch.float32)))
    if mode == 'head':
        return head, head, top
    if mode == 'map':
        return top, head, top
    a = np.clip(np.float32(head), 1e-5, 1-1e-5)
    b = np.clip(np.float32(top), 1e-5, 1-1e-5)
    logit = weight * np.log(a / (1-a)) + (1-weight) * np.log(b / (1-b))
    return float(1 / (1 + np.exp(-logit))), head, top


@torch.inference_mode()
def predict(encoder, head, root, rows, arm, save_dir=None, mode='combined', weight=.5):
    head.eval()
    scores, maps, shapes, extras = [], [], [], []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        rgb, _, _, meta = load_image_and_target(root, row, read_mask=False)
        direct, tiles, global_view = required_views(arm)
        feat = encoder.extract(rgb, include_direct=direct, include_tiles=tiles,
                               include_global=global_view)
        with torch.autocast('cuda', dtype=torch.float16,
                            enabled=encoder.mean.device.type == 'cuda'):
            z, raw = head(feat)
        square = z[0].float().sigmoid().cpu().numpy().astype(np.float16)
        grid = map_to_grid(square, meta)
        effective_mode = 'map' if arm == 'R5' else mode
        score, raw_head, top = readout(grid, raw[0].float().cpu(), effective_mode, weight)
        scores.append(score); maps.append(grid); shapes.append(meta[:2])
        extras.append({'sample_id': row['sample_id'], 'label': row['label'],
                       'score': score, 'head_score': raw_head, 'map_top830': top})
        if save_dir is not None:
            np.save(save_dir / f'{row["sample_id"]}.npy', grid, allow_pickle=False)
    return scores, maps, shapes, extras


def grid_ap(root, rows, maps):
    y, p = [], []
    for row, score in zip(rows, maps):
        if row['label']:
            mask = imread(root / row['mask'], cv2.IMREAD_GRAYSCALE)
            target = cv2.resize((mask > 0).astype(np.float32),
                                (score.shape[1], score.shape[0]),
                                interpolation=cv2.INTER_AREA) > .02
        else:
            target = np.zeros(score.shape, bool)
        y.append(target.ravel()); p.append(score.astype(np.float32).ravel())
    return float(average_precision_score(np.concatenate(y), np.concatenate(p)))


def run_dir(cfg, dataset, arm, seed, budget):
    suffix = f'{arm}_B{round(budget*100)}' if budget < 1 else arm
    return resolve(cfg['output_dir']) / dataset / suffix / f'seed_{seed}'


def lock_run(cfg, dataset, arm, seed, budget, fit, dev):
    return {'dataset': dataset, 'arm': arm, 'seed': seed, 'budget': budget,
            'config_sha256': cfg['_config_sha256'],
            'manifest_sha256': sha256(resolve(cfg[dataset]['manifest'])),
            'backbone_sha256': sha256(resolve(cfg['dino_weights'])),
            'fit_ids': [r['sample_id'] for r in fit],
            'dev_ids': [r['sample_id'] for r in dev]}


def train(dataset, arm, seed, budget, cfg, data_root, device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Formal BladeGL training requires CUDA')
    seed_all(seed)
    fit = budget_rows(role_rows(dataset, cfg, 'fit'), budget, seed)
    dev = role_rows(dataset, cfg, 'dev')
    normal = [r for r in fit if not r['label']]
    abnormal = [r for r in fit if r['label']]
    if not normal or not abnormal:
        raise ValueError('Both normal and anomalous fit rows are required')
    out = run_dir(cfg, dataset, arm, seed, budget)
    out.mkdir(parents=True, exist_ok=True)
    lock = lock_run(cfg, dataset, arm, seed, budget, fit, dev)
    lock_path = out / 'run_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf-8')) != lock:
        raise RuntimeError(f'Run lock changed: {lock_path}')
    save_json(lock_path, lock)
    if (out / 'training_complete.json').exists():
        return
    encoder = Encoder(resolve('.'), cfg).to(device).eval()
    head = make_head(arm).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['max_steps'], eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['mixed_precision'])
    step, best, records = 0, -math.inf, []
    progress = out / 'progress.pt'
    if progress.exists():
        ck = torch.load(progress, map_location='cpu', weights_only=False)
        if ck['lock'] != lock:
            raise RuntimeError('Progress lock changed')
        head.load_state_dict(ck['head']); opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        step, best, records = ck['step'], ck['best'], ck['records']
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    start = time.time()
    while step < cfg['max_steps']:
        batch = random.choices(normal, k=cfg['batch_size']//2) + \
                random.choices(abnormal, k=cfg['batch_size']-cfg['batch_size']//2)
        random.shuffle(batch)
        feat, target, valid, labels = encode_rows(encoder, data_root, batch, arm)
        retry = 0
        while True:
            head.train(); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['mixed_precision']):
                z, s = head(feat)
                loss = loss_value(arm, z, s, target, valid, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step+1}')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            gradients = [p.grad for p in head.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError(f'No gradient at step {step+1}')
            if not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
                old_scale = scaler.get_scale()
                scaler.step(opt); scaler.update()
                retry += 1
                if scaler.get_scale() >= old_scale or retry >= 12:
                    raise RuntimeError(f'Persistent AMP overflow at step {step+1}')
                print(json.dumps({'event': 'AMP_OVERFLOW_RETRY', 'step': step+1,
                                  'retry': retry, 'old_scale': old_scale,
                                  'new_scale': scaler.get_scale()}), flush=True)
                continue
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Nonfinite gradient norm at step {step+1}')
            scale = scaler.get_scale(); scaler.step(opt); scaler.update()
            if scaler.get_scale() < scale:
                raise RuntimeError(f'AMP skipped finite-gradient update {step+1}')
            break
        sched.step(); step += 1
        if step % cfg['eval_every'] == 0 or step == cfg['max_steps']:
            _, maps, _, _ = predict(encoder, head, data_root, dev, arm)
            ap = grid_ap(data_root, dev, maps)
            row = {'step': step, 'dev_grid_pixel_ap': ap,
                   'last_loss': float(loss.detach()), 'elapsed_s': time.time()-start,
                   'peak_vram_mib': torch.cuda.max_memory_allocated()/2**20}
            records.append(row); save_json(out/'training_log.json', records)
            if ap > best:
                best = ap
                torch.save({'head': head.state_dict(), 'lock': lock, 'best_step': step,
                            'best_dev_ap': ap}, out/'best.pt')
            state = {'head': head.state_dict(), 'optimizer': opt.state_dict(),
                     'scheduler': sched.state_dict(), 'scaler': scaler.state_dict(),
                     'step': step, 'best': best, 'records': records, 'lock': lock,
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all()}
            temp = progress.with_suffix('.tmp'); torch.save(state, temp); os.replace(temp, progress)
            print(json.dumps(row), flush=True)
    torch.save({'head': head.state_dict(), 'lock': lock, 'step': step}, out/'last.pt')
    save_json(out/'training_complete.json', {'accepted_updates': step,
                                              'best_dev_ap': best, 'seconds': time.time()-start})


def evaluate(dataset, arm, seed, budget, cfg, data_root, device,
             readout_mode='combined', readout_weight=.5):
    out = run_dir(cfg, dataset, arm, seed, budget)
    if not (out/'training_complete.json').is_file():
        raise RuntimeError(f'Complete training first: {out}')
    lock = json.loads((out/'run_lock.json').read_text(encoding='utf-8'))
    if lock['manifest_sha256'] != sha256(resolve(cfg[dataset]['manifest'])):
        raise RuntimeError('Manifest changed since training')
    ck = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
    if ck['lock'] != lock:
        raise RuntimeError('Checkpoint lock mismatch')
    encoder = Encoder(resolve('.'), cfg).to(device).eval()
    head = make_head(arm).to(device).eval()
    head.load_state_dict(ck['head'], strict=True)
    cal = role_rows(dataset, cfg, 'cal')
    test = role_rows(dataset, cfg, 'test')
    cal_scores, cal_maps, cal_shapes, _ = predict(
        encoder, head, data_root, cal, arm, mode=readout_mode, weight=readout_weight)
    image_t, image_rule = image_threshold_recall_floor(cal, cal_scores, cfg['image_min_recall'])
    pixel_t, pixel_rule = select_pixel_threshold(
        data_root, cal, cal_maps, cal_shapes, cfg['pixel_threshold_candidates'])
    save_json(readout_artifact(out, 'calibration', readout_mode, readout_weight), {
        'image_threshold': image_t, 'pixel_threshold': pixel_t,
        'image_rule': image_rule, 'pixel_rule': pixel_rule,
        'readout': readout_mode, 'head_weight': readout_weight,
        'cal_ids': [r['sample_id'] for r in cal]})
    test_scores, test_maps, shapes, extras = predict(
        encoder, head, data_root, test, arm, out/'test_maps',
        mode=readout_mode, weight=readout_weight)
    with readout_artifact(out, 'image_scores', readout_mode, readout_weight).open(
            'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(extras[0]))
        writer.writeheader(); writer.writerows(extras)
    result = {'dataset': dataset, 'arm': arm, 'seed': seed, 'budget': budget,
              'checkpoint_step': ck['best_step'], 'checkpoint_dev_ap': ck['best_dev_ap'],
              'image': image_metrics(test, test_scores, image_t),
              'pixel': evaluate_maps(data_root, test, test_maps, shapes, pixel_t,
                                     readout_artifact(out, 'curves', readout_mode,
                                                      readout_weight)),
              'readout': readout_mode, 'head_weight': readout_weight}
    cost = profile_rgb_path(test, lambda row: predict(
        encoder, head, data_root, [row], arm, mode=readout_mode,
        weight=readout_weight), [encoder, head], device)
    save_json(readout_artifact(out, 'cost_rgb', readout_mode, readout_weight), cost)
    result['cost'] = cost
    save_json(readout_artifact(out, 'metrics', readout_mode, readout_weight), result)
    return result
