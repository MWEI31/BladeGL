# Dataset manifests / 数据划分清单

The original sample-level manifests are not distributed. Provide these files
before running preflight, training, or evaluation:

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

原始样本清单不随代码分发。请按以上字段准备 CSV，图片与掩膜路径相对于数据集根目录；0 表示正常图像，1 表示异常图像。数据角色和样本数量由配置及协议代码检查。缺少原始样本分配时，无法还原原实验的数据划分。
