# BladeGL

[中文](README.md)

Code for aero-engine blade defect detection and localization. BladeGL uses
DINOv2 features and CECF to combine full-image context with local detail,
producing image-level predictions and pixel-level defect maps. Training uses
image labels and defect masks.

## Installation

Use Python 3.10+. Install a CUDA-compatible [PyTorch and torchvision build](https://pytorch.org/get-started/locally/), then run:

```bash
pip install -r requirements.txt
python scripts/download_backbone.py
```

## Data

Datasets: [AeBAD-S](https://github.com/zhangzilongc/MMR) ·
[KSDD2](https://www.vicos.si/resources/kolektorsdd2/)

Pass the dataset path with `--data-root` and set the split manifests in
[configs/bladegl.json](configs/bladegl.json). See [manifests](manifests/README.md)
for the CSV format. Data and weights are not included.

Use [prepare_manifest.py](scripts/prepare_manifest.py) to build and validate a
manifest from an existing sample assignment table.

## Training and evaluation

AeBAD-S example; replace `data/AeBAD_S` with your dataset path:

```bash
python scripts/run_bladegl.py all --dataset aebad --arm E0 --seed 91025 --data-root data/AeBAD_S
```

## Inference

```bash
python scripts/infer.py --input path/to/image.png --checkpoint path/to/model.pt --output-dir outputs/inference
```

Each run saves a localization probability map and an image-level score in a
separate subdirectory. Add `--calibration path/to/calibration.json` for thresholded
predictions, using calibration generated from the same checkpoint.

[License](LICENSE_STATUS.md) · [Third-party components](THIRD_PARTY_NOTICES.md)
