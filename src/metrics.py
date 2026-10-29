"""Original-pixel weighted evaluation for constant nearest-neighbour score cells."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import imread


def grid_counts(data_root, row: dict, grid_shape: tuple[int, int], image_shape: tuple[int, int]):
    h, w = image_shape
    gh, gw = grid_shape
    ys = np.ceil(np.arange(gh + 1) * h / gh).astype(np.int32)
    xs = np.ceil(np.arange(gw + 1) * w / gw).astype(np.int32)
    area = np.outer(np.diff(ys), np.diff(xs)).astype(np.int64)
    if not row["label"]:
        return np.zeros((gh, gw), np.int64), area
    mask = imread(data_root / row["mask"], cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (h, w):
        raise ValueError(f"Missing or mismatched original mask: {row['sample_id']}")
    binary = (mask > 0).astype(np.uint8)
    integral = cv2.integral(binary, sdepth=cv2.CV_32S)
    pos = (integral[ys[1:, None], xs[None, 1:]]
           - integral[ys[:-1, None], xs[None, 1:]]
           - integral[ys[1:, None], xs[None, :-1]]
           + integral[ys[:-1, None], xs[None, :-1]])
    if pos.sum() != binary.sum() or area.sum() != h * w:
        raise AssertionError("Grid-to-original count mismatch")
    return pos.astype(np.int64), area


def ap_from_hist(pos_hist, area_hist):
    # Buckets are the monotone signed-FP16 transform from score_buckets().
    p = np.cumsum(pos_hist[::-1], dtype=np.float64)
    a = np.cumsum(area_hist[::-1], dtype=np.float64)
    if not p[-1]:
        return None
    return float(np.sum(pos_hist[::-1] / p[-1] * p / np.maximum(a, 1)))


def score_buckets(values):
    """Unsigned buckets that preserve the order of finite signed FP16 scores."""
    half = np.ascontiguousarray(values, dtype=np.float16)
    if not np.isfinite(half).all():
        raise ValueError('Prediction map has nonfinite score')
    bits = half.view(np.uint16)
    return np.where(bits & 0x8000, np.bitwise_not(bits), bits ^ 0x8000).astype(np.uint16)


def image_metrics(rows, scores, threshold: float):
    labels = np.asarray([int(x["label"]) for x in rows], np.int32)
    scores = np.asarray(scores, np.float64)
    pred = scores >= threshold
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    return {
        "n": len(labels), "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "ap": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": (tp + tn) / max(len(labels), 1),
    }


def normal_quantile(rows, scores, q: float):
    normal = np.sort(np.asarray([float(s) for r, s in zip(rows, scores) if not r["label"]]))
    if not len(normal):
        raise ValueError("No normal calibration scores")
    index = min(len(normal) - 1, int(np.ceil(q * len(normal))) - 1)
    return float(np.nextafter(normal[index], np.inf)), {"normal_n": len(normal), "order_index": index, "q": q}


def image_threshold_recall_floor(rows, scores, minimum_recall=.90):
    """Maximize image F1 subject to recall >= 0.90."""
    labels = np.asarray([int(r['label']) for r in rows], bool)
    values = np.asarray(scores, np.float64)
    if not labels.any() or labels.all():
        raise ValueError('Calibration requires both image classes')
    options = []
    for threshold in np.unique(values):
        pred = values >= threshold
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        if tp / max(tp + fn, 1) >= minimum_recall:
            options.append((2 * tp / max(2 * tp + fp + fn, 1), float(threshold)))
    if not options:
        raise ValueError('No threshold meets the recall floor')
    best = max(options)
    return best[1], {'objective': 'max_F1_subject_to_recall',
                     'minimum_recall': minimum_recall, 'calibration_F1': best[0]}


def evaluate_maps(data_root, rows, probabilities, original_shapes,
                  pixel_threshold: float, curve_path: Path | None = None):
    hp = np.zeros(65536, np.float64)
    ha = np.zeros(65536, np.float64)
    positive_dice, positive_iou = [], []
    tp = fp = fn = 0
    normal_pred_area = 0
    normal_area = 0
    normal_pred_fractions = []
    zero_hit = 0
    background_hist = np.zeros(65536, np.float64)
    region_pro_sum = np.zeros(65536, np.float64)
    region_count = 0
    for row, score, shape in zip(rows, probabilities, original_shapes):
        pos, area = grid_counts(data_root, row, score.shape, shape)
        bucket = score_buckets(score).ravel()
        hp += np.bincount(bucket, weights=pos.ravel(), minlength=65536)
        ha += np.bincount(bucket, weights=area.ravel(), minlength=65536)
        background_hist += np.bincount(bucket, weights=(area-pos).ravel(), minlength=65536)
        if row['label']:
            original = imread(data_root / row['mask'], cv2.IMREAD_GRAYSCALE)
            region_count += _accumulate_region_pro(original, score, region_pro_sum)
        pred = score >= pixel_threshold
        tpi = int(pos[pred].sum())
        fpi = int((area[pred] - pos[pred]).sum())
        fni = int(pos[~pred].sum())
        tp += tpi; fp += fpi; fn += fni
        if row["label"]:
            positive_dice.append(2 * tpi / max(2 * tpi + fpi + fni, 1))
            positive_iou.append(tpi / max(tpi + fpi + fni, 1))
            zero_hit += int(tpi == 0)
        else:
            normal_pred_area += int(area[pred].sum())
            normal_area += int(area.sum())
            normal_pred_fractions.append(float(area[pred].sum()/area.sum()))
    if curve_path is not None:
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(curve_path, positive_hist=hp, area_hist=ha,
                            background_hist=background_hist,
                            region_pro_sum=region_pro_sum,
                            region_count=np.int64(region_count),
                            pixel_threshold=np.float64(pixel_threshold),
                            score_rule='monotone signed FP16 buckets; descending threshold')
    return {
        "original_pixel_ap": ap_from_hist(hp, ha),
        "aupro_030": aupro_from_hist(background_hist, region_pro_sum, region_count),
        "aupro_region_count": region_count,
        "positive_macro_dice": float(np.mean(positive_dice)) if positive_dice else None,
        "positive_macro_iou": float(np.mean(positive_iou)) if positive_iou else None,
        "pixel_precision": tp / max(tp + fp, 1), "pixel_recall": tp / max(tp + fn, 1),
        "pixel_tp": tp, "pixel_fp": fp, "pixel_fn": fn,
        "normal_false_area_fraction": normal_pred_area / max(normal_area, 1),
        "mean_normal_false_area_fraction": float(np.mean(normal_pred_fractions))
        if normal_pred_fractions else None,
        "zero_hit_defect_images": zero_hit,
        "score_grid_rule": "positive FP16 probability, nearest-neighbour projected to original coordinates",
    }


def _accumulate_region_pro(original_mask, score_grid, out):
    """Accumulate per-region TPR curves using original 8-connected masks."""
    if original_mask is None:
        raise ValueError('Missing original mask for AUPRO')
    binary = (original_mask > 0).astype(np.uint8)
    n, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        raise ValueError('Positive image has no original connected region')
    h, w = binary.shape
    expanded = cv2.resize(np.asarray(score_grid, np.float16), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    bitpattern = score_buckets(expanded)
    for label in range(1, n):
        x, y, width, height, area = stats[label]
        roi = components[y:y+height, x:x+width] == label
        bins = np.bincount(bitpattern[y:y+height, x:x+width][roi], minlength=65536)
        out += np.cumsum(bins[::-1], dtype=np.float64) / float(area)
    return n - 1


def aupro_from_hist(background_hist, region_pro_sum, region_count, limit=.30):
    """Mean original-region PRO, integrated over global background FPR [0,.30]."""
    if region_count == 0 or background_hist.sum() <= 0:
        return None
    fpr = np.r_[0., np.cumsum(background_hist[::-1], dtype=np.float64) /
                background_hist.sum()]
    pro = np.r_[0., region_pro_sum / region_count]
    if fpr[-1] < limit:
        raise AssertionError('Background FPR curve does not reach integration limit')
    cross = int(np.searchsorted(fpr, limit, side='left'))
    area = float(np.trapezoid(pro[:cross+1], fpr[:cross+1])) if fpr[cross] == limit else \
        float(np.trapezoid(pro[:cross], fpr[:cross]))
    if fpr[cross] != limit:
        before = cross - 1
        ylimit = pro[before] + (pro[cross]-pro[before]) * (
            limit-fpr[before]) / (fpr[cross]-fpr[before])
        area += .5 * (pro[before]+ylimit) * (limit-fpr[before])
    return area / limit


def select_pixel_threshold(data_root, rows, probabilities, original_shapes, candidates: int):
    positive = [(r, p, shape) for r, p, shape in zip(rows, probabilities, original_shapes) if r["label"]]
    if not positive:
        raise ValueError("No positive calibration masks")
    cached = [(p, *grid_counts(data_root, r, p.shape, shape)) for r, p, shape in positive]
    options = []
    minimum = min(float(np.min(p)) for _, p, _ in positive)
    maximum = max(float(np.max(p)) for _, p, _ in positive)
    # Use the .01..99 grid for probabilities and a calibration-derived
    # range for raw anomaly distances.
    thresholds = np.linspace(.01, .99, candidates) if 0 <= minimum and maximum <= 1 \
        else np.linspace(minimum, maximum, candidates)
    for threshold in thresholds:
        dice = []
        for score, pos, area in cached:
            pred = score >= threshold
            tpi = float(pos[pred].sum())
            dice.append(2 * tpi / max(float(area[pred].sum() + pos.sum()), 1))
        options.append((float(np.mean(dice)), float(threshold)))
    return max(options)[1], {"objective": "positive_image_macro_dice", "candidates": candidates,
                              "range_source": "calibration_anomaly_maps", "range": [minimum, maximum],
                              "best_macro_dice": max(x[0] for x in options)}
