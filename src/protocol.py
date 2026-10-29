"""Strict, explicit manifests; test rows never enter model construction."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: str | Path) -> dict:
    p = resolve(path)
    config = json.loads(p.read_text(encoding="utf-8"))
    config["_config_path"] = str(p)
    config["_config_sha256"] = sha256(p)
    return config


def read_manifest(dataset: str, cfg: dict) -> list[dict]:
    path = resolve(cfg[dataset]["manifest"])
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if dataset == "aebad":
        # Match the locked BladeGL protocol: one duplicate normal train file is
        # excluded by first-occurrence SHA256, yielding 416 rather than 417.
        seen = set()
        unique = []
        for row in rows:
            key = row["image_sha256"]
            if key not in seen:
                unique.append(row)
                seen.add(key)
        rows = unique
    normalized = []
    for row in rows:
        if dataset == "aebad":
            role = row["protocol_role"]
            item = {
                "sample_id": row["sample_id"], "role": role,
                "image": row["relative_image_path"].replace("\\", "/"),
                "mask": row["relative_mask_path"].replace("\\", "/"),
                "label": int(row["label"]),
                "condition": row.get("condition", ""),
                "defect_type": row.get("defect_type", ""),
                "image_sha256": row["image_sha256"],
            }
        else:
            item = {
                "sample_id": row["sample_id"], "role": row["role"],
                "image": row["image"].replace("\\", "/"),
                "mask": row["mask"].replace("\\", "/"),
                "label": int(row["label"]),
                "condition": row.get("condition", ""),
                "defect_type": row.get("defect_type", ""),
                "image_sha256": row.get("image_sha256", ""),
            }
        normalized.append(item)
    ids = [x["sample_id"] for x in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id in manifest")
    roles = {r: [x for x in normalized if x["role"] in cfg[dataset][f"{r}_roles"]]
             for r in ("fit", "dev", "cal", "test")}
    if any(not v for v in roles.values()):
        raise ValueError({k: len(v) for k, v in roles.items()})
    for a in ("fit", "dev", "cal"):
        if {x["sample_id"] for x in roles[a]} & {x["sample_id"] for x in roles["test"]}:
            raise ValueError(f"{a}/test overlap")
    if dataset == "ksdd2":
        for a, b in (("fit", "dev"), ("fit", "cal"), ("dev", "cal")):
            if {x["sample_id"] for x in roles[a]} & {x["sample_id"] for x in roles[b]}:
                raise ValueError(f"KSDD2 {a}/{b} overlap")
    if dataset == "aebad" and [len(roles[x]) for x in ("fit", "dev", "test")] != [602, 158, 1399]:
        raise ValueError("AeBAD canonical counts changed")
    if dataset == "ksdd2" and len(roles["test"]) != 1004:
        raise ValueError("KSDD2 official test count changed")
    return normalized


def split_ksdd2(canonical: Path, data_root: Path, output: Path, seed: int = 20260928) -> dict:
    """Stratified train-internal 80/10/10 split; official test is untouched."""
    with canonical.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    train = [x for x in records if x["split"] == "train"]
    test = [x for x in records if x["split"] == "test"]
    if len(train) != 2331 or len(test) != 1004:
        raise ValueError("Canonical KSDD2 counts do not match official split")
    rng = random.Random(seed)
    rows = []
    from .data import imread
    import cv2
    groups = {0: [], 1: []}
    for x in train:
        mask = imread(data_root / x["mask"].replace("\\", "/"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(x["mask"])
        groups[int(mask.max() > 0)].append(x)
    for label, group in groups.items():
        rng.shuffle(group)
        n_fit = round(len(group) * .8)
        n_dev = round(len(group) * .1)
        for i, x in enumerate(group):
            role = "fit" if i < n_fit else "dev" if i < n_fit + n_dev else "cal"
            rows.append({"sample_id": f"train_{Path(x['image']).stem}", "role": role,
                         "image": x["image"], "mask": x["mask"], "label": label,
                         "condition": "", "defect_type": ""})
    for x in test:
        mask = imread(data_root / x["mask"].replace("\\", "/"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(x["mask"])
        rows.append({"sample_id": f"test_{Path(x['image']).stem}", "role": "test",
                     "image": x["image"], "mask": x["mask"],
                     "label": int(mask.max() > 0), "condition": "", "defect_type": ""})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda x: (x["role"], x["sample_id"])))
    counts = {r: {str(label): sum(x["role"] == r and x["label"] == label for x in rows)
                  for label in (0, 1)} for r in ("fit", "dev", "cal", "test")}
    return {"seed": seed, "counts": counts, "manifest_sha256": sha256(output),
            "note": "No verified object IDs available; image-level stratification only."}
