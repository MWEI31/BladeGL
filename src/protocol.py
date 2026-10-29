"""Dataset manifests and split validation."""
from __future__ import annotations

import csv
import hashlib
import json
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
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset manifest: {path}. "
            "Use scripts/prepare_manifest.py with your sample assignments.")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if dataset == "aebad":
        # Deduplicate images by SHA-256, retaining the first occurrence.
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
    allowed_roles = set().union(*(cfg[dataset][f"{r}_roles"]
                                  for r in ("fit", "dev", "cal", "test")))
    if any(x["label"] not in (0, 1) or x["role"] not in allowed_roles for x in normalized):
        raise ValueError("Invalid label or dataset role in manifest")
    roles = {r: [x for x in normalized if x["role"] in cfg[dataset][f"{r}_roles"]]
             for r in ("fit", "dev", "cal", "test")}
    if any(not v for v in roles.values()):
        raise ValueError({k: len(v) for k, v in roles.items()})
    for a in ("fit", "dev", "cal"):
        if {x["sample_id"] for x in roles[a]} & {x["sample_id"] for x in roles["test"]}:
            raise ValueError(f"{a}/test overlap")
    pairs = [("fit", "dev"), ("fit", "cal"), ("fit", "test"),
             ("dev", "test"), ("cal", "test")]
    if dataset == "ksdd2":
        pairs.append(("dev", "cal"))
    for a, b in pairs:
        for key in ("image", "image_sha256"):
            left = {x[key] for x in roles[a] if x.get(key)}
            right = {x[key] for x in roles[b] if x.get(key)}
            if left & right:
                raise ValueError(f"{a}/{b} overlap by {key}")
    if dataset == "ksdd2":
        for a, b in (("fit", "dev"), ("fit", "cal"), ("dev", "cal")):
            if {x["sample_id"] for x in roles[a]} & {x["sample_id"] for x in roles[b]}:
                raise ValueError(f"KSDD2 {a}/{b} overlap")
    if dataset == "aebad" and [len(roles[x]) for x in ("fit", "dev", "test")] != [602, 158, 1399]:
        raise ValueError("AeBAD canonical counts changed")
    if dataset == "ksdd2" and len(roles["test"]) != 1004:
        raise ValueError("KSDD2 official test count changed")
    return normalized
