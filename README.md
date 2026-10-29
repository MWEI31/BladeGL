# BladeGL

[English](README_EN.md)

航空发动机叶片缺陷检测与定位代码。模型基于 DINOv2，通过 CECF 交互整图与局部特征，输出图像级检测结果和像素级定位图。训练使用图像标签和缺陷掩膜。

## 安装

使用 Python 3.10+，先安装适配 CUDA 的 [PyTorch 和 torchvision](https://pytorch.org/get-started/locally/)，再执行：

```bash
pip install -r requirements.txt
python scripts/download_backbone.py
```

## 数据

数据集：[AeBAD-S](https://github.com/zhangzilongc/MMR) · [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)

数据路径通过 `--data-root` 指定，划分清单在 [配置文件](configs/bladegl.json)中设置，格式见 [manifests](manifests/README.md)。仓库不包含数据和权重。

## 训练与评估

以 AeBAD-S 为例，将 `data/AeBAD_S` 替换为实际数据路径：

```bash
python scripts/run_bladegl.py all --dataset aebad --arm E0 --seed 91025 --data-root data/AeBAD_S
```

## 推理

```bash
python scripts/infer.py --input path/to/image.png --checkpoint path/to/model.pt --output-dir outputs/inference
```

输出包括定位概率图和图像级得分。

[项目许可](LICENSE_STATUS.md) · [第三方组件](THIRD_PARTY_NOTICES.md)
