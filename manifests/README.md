# Dataset manifests / 数据划分清单

CSV files used by `configs/bladegl.json`:

- `aebad_split_41.csv`
- `ksdd2_split.csv`

## Prepare from sample assignments

Input columns: `role,image,mask,label`. Optional columns: `sample_id,condition,defect_type`.
Use the role names in `configs/bladegl.json` and paths relative to the dataset root.

```bash
python scripts/prepare_manifest.py --dataset aebad --data-root data/AeBAD_S --assignments path/to/assignments.csv
```

The script preserves the supplied groups, computes image hashes, and validates the
manifest. It does not reconstruct missing experimental assignments.

分组表中的 `role` 使用配置文件里的角色名，图片与掩膜路径相对于数据集根目录。脚本保留已有分组，不自动重划实验样本。

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
