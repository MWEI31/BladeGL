"""RGB/mask reader with reversible letterbox geometry."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]


def imread(path: Path, flags: int):
    raw = np.fromfile(path, dtype=np.uint8) if path.is_file() else np.empty(0, np.uint8)
    return cv2.imdecode(raw, flags) if len(raw) else None


def geometry(height: int, width: int, target_h: int, target_w: int) -> tuple[int, int, int, int]:
    scale = min(target_h / height, target_w / width)
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))
    return new_h, new_w, (target_h - new_h) // 2, (target_w - new_w) // 2


def read_pair(root: Path, row: dict, shape: tuple[int, int], include_mask: bool = True):
    image_path = root / row["image"]
    image = imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    if include_mask and row["label"]:
        mask_path = root / row["mask"]
        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.shape != (h, w):
            raise ValueError(f"Mask size mismatch: {row['sample_id']}")
        mask = mask > 0
        if not mask.any():
            raise ValueError(f"Positive label with empty mask: {row['sample_id']}")
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
    th, tw = shape
    nh, nw, top, left = geometry(h, w, th, tw)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    image_canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    image_canvas[:] = np.array([124, 116, 104], np.uint8)
    image_canvas[top:top + nh, left:left + nw] = resized
    mask_canvas = np.zeros((th, tw), dtype=np.float32)
    # Area coverage preserves tiny defects better than nearest-neighbor labels.
    mask_canvas[top:top + nh, left:left + nw] = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    valid = np.zeros((th, tw), dtype=np.float32)
    valid[top:top + nh, left:left + nw] = 1
    rgb = image_canvas.transpose(2, 0, 1).astype(np.float32) / 255
    rgb = (rgb - MEAN) / STD
    return (torch.from_numpy(rgb.copy()), torch.from_numpy(mask_canvas),
            torch.from_numpy(valid), (h, w, nh, nw, top, left))


def unletterbox(score: np.ndarray, meta: tuple[int, ...], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    h, w, nh, nw, top, left = [int(x) for x in meta]
    crop = score[top:top + nh, left:left + nw]
    return cv2.resize(crop.astype(np.float32), (w, h), interpolation=interpolation)


class ImageDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], shape: tuple[int, int]):
        self.root, self.rows, self.shape = root, rows, shape

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image, mask, valid, meta = read_pair(self.root, row, self.shape)
        return image, mask, valid, torch.tensor(float(row["label"])), torch.tensor(meta), row["sample_id"]
