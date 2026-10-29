# Dataset manifests / 数据划分清单

CSV files used by `configs/bladegl.json`:

- `aebad_split_41.csv`
- `ksdd2_split.csv`

## Required columns

AeBAD-S:

```text
sample_id,protocol_role,relative_image_path,relative_mask_path,label,image_sha256
```

Optional: `condition`, `defect_type`.
The loader uses the image hash to deduplicate image contents.

KSDD2:

```text
sample_id,role,image,mask,label
```

Optional: `condition`, `defect_type`, `image_sha256`.

Image and mask paths are relative to the dataset root. A label of 0 denotes a
normal image and 1 denotes an anomalous image. Roles and expected counts are
checked by `src/protocol.py` using `configs/bladegl.json`.

CSV 中的图片与掩膜路径相对于数据集根目录。标签 0 表示正常图像，1 表示异常图像。
