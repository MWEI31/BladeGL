# DINOv2 weights

Download the DINOv2 ViT-S/14 pretrained checkpoint from the repository root:

```bash
python scripts/download_backbone.py
```

The checkpoint is saved as `weights/dinov2_vits14_pretrain.pth` and verified
against `docs/backbone_reference.json`. Weight files are excluded from Git.
