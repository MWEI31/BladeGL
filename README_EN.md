# BladeGL

[中文](README.md)

BladeGL detects defects in aero-engine blade images and predicts their pixel-level
locations. It uses a frozen DINOv2 ViT-S/14 encoder to extract features from a full
image and four overlapping crops. Convolution-Enhanced Cosine Fusion (CECF) brings
full-image context into local features. The image-level prediction combines the
classification output with regional responses from the defect map.

Training uses image labels and defect masks. This repository includes the model,
training and evaluation scripts, ablation settings, and single-image inference.

## Repository structure

```text
configs/       Training settings and dataset roles
src/bladegl/   Backbone, CECF, prediction heads, and training loop
src/           Data loading, metrics, and protocol checks
scripts/       Training, inference, input checks, and backbone download
third_party/   DINOv2 source and upstream license
manifests/     Instructions for supplying dataset split files
weights/       Instructions for obtaining the pretrained backbone
tests/         Model, metric, and protocol checks
```

## Installation

Use Python 3.10 or newer. Install PyTorch and torchvision with a CUDA build suited
to your GPU from the [PyTorch website](https://pytorch.org/get-started/locally/),
then run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/download_backbone.py
```

The download script verifies the pretrained DINOv2 checkpoint before placing it
in `weights/`. Training uses CUDA. NumPy 2 or newer is required by the AUPRO code.

## Data preparation

Obtain [AeBAD-S](https://github.com/zhangzilongc/MMR) and
[KSDD2](https://www.vicos.si/resources/kolektorsdd2/) from their authors. Images,
annotations, sample-level split files, pretrained weights, trained checkpoints,
and experiment outputs are not included in this repository.

Provide the original protocol manifests at the paths in `configs/bladegl.json`:

- `manifests/aebad_split_41.csv`
- `manifests/ksdd2_split.csv`

See [manifest instructions](manifests/README.md) for the required columns. Image
and mask paths in each CSV are relative to the dataset root passed on the command
line. These files are needed before preflight, training, or evaluation can run.

AeBAD-S uses 602 training images, 158 shared development/calibration images, and
1,399 evaluation images. Its anomalous training images come from the original
test partition, under a supervised protocol. KSDD2 uses 1,865 training, 233
development, 233 calibration, and 1,004 test images. The datasets are trained
separately. See [protocol details](docs/PROTOCOL.md).

## Training and evaluation

Replace the dataset root below with its location on your machine:

```bash
python scripts/preflight.py --dataset aebad --data-root data/AeBAD_S --verify-files --verify-hashes --verify-backbone
python scripts/run_bladegl.py all --dataset aebad --arm E0 --seed 91025 --data-root data/AeBAD_S
python scripts/run_bladegl.py all --dataset ksdd2 --arm E0 --seed 91025 --data-root data/KolektorSDD2
```

`all` runs training, model selection on development data, calibration, and final
evaluation. The configuration specifies 3,000 updates, batch size 4, and AdamW.
The experimental seeds are `91025`, `92025`, and `93025`.

Results are written to `outputs/bladegl/<dataset>/<arm>/seed_<seed>/`, including
checkpoints, calibration settings, image scores, prediction arrays, and metrics.

## Ablations

Select a variant with `--arm`. The definitions are in
[`src/bladegl/variants.py`](src/bladegl/variants.py).

| Setting | Arm |
|---|---|
| Full BladeGL | `E0` |
| Direct full-image/local fusion | `F0` |
| Full-image or local-only localization | `S1`, `S2`, `S3` |
| Attention and fusion variants | `F1`–`F8` |
| Regional-input and loss variants | `R4`, `R5`, `R6` |
| Window sizes | `W4`, `W6`, `W8` |
| CBAM comparison head | `E01_CBAM` |

For a fixed checkpoint, `evaluate` accepts `--readout head`, `--readout map`, or
`--readout combined --readout-weight 0.5`. Each readout is calibrated separately.

## Single-image inference

Use a trained checkpoint and an input image:

```bash
python scripts/infer.py --input path/to/image.png --checkpoint outputs/bladegl/aebad/E0/seed_91025/best.pt --output-dir outputs/inference
```

The script saves a probability array, a preview image, and image scores. Add
`--calibration path/to/calibration.json` to obtain thresholded predictions.
Calibration must match the model and readout used for inference.

## License

The project code has no selected distribution license at present. See
[license status](LICENSE_STATUS.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
DINOv2 retains its upstream Apache-2.0 license.
