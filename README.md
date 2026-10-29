# BladeGL

[English](README_EN.md)

BladeGL 用于航空发动机叶片的图像级缺陷检测和像素级缺陷定位。模型使用冻结的 DINOv2 ViT-S/14，分别提取整幅图像和四个重叠局部裁剪的特征，通过卷积增强余弦融合模块（CECF）将整图上下文引入局部特征。最终的图像级判断结合分类分支输出与缺陷定位图中的区域响应。

模型训练需要图像标签和缺陷掩膜。本仓库提供模型实现、训练与评估脚本、消融设置和单图推理入口。

## 目录结构

```text
configs/       训练参数和数据角色配置
src/bladegl/   特征提取、CECF、预测分支及训练流程
src/           数据读取、指标计算和协议检查
scripts/       训练、推理、输入检查及预训练权重下载
third_party/   DINOv2 源码及其原始许可
manifests/     数据划分清单的准备说明
weights/       预训练权重的获取说明
tests/         模型、指标和协议检查
```

## 环境安装

使用 Python 3.10 或更新版本。先从 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装与显卡环境匹配的 PyTorch 和 torchvision，再在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/download_backbone.py
```

下载脚本会校验 DINOv2 预训练权重，并将其保存到 `weights/`。训练使用 CUDA；AUPRO 的计算需要 NumPy 2 或更新版本。

## 数据准备

从原作者提供的入口获取 [AeBAD-S](https://github.com/zhangzilongc/MMR) 和 [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)。仓库不附带数据图片、标注、样本划分清单、预训练权重、训练后的模型或实验输出。

训练前需按 `configs/bladegl.json` 中的路径准备原始协议清单：

- `manifests/aebad_split_41.csv`
- `manifests/ksdd2_split.csv`

CSV 的必要字段见[清单说明](manifests/README.md)。其中的图片和掩膜路径相对于命令行传入的数据集根目录。预检查、训练和评估均使用这些清单。

AeBAD-S 使用 602 张训练图像、158 张开发与校准共用图像和 1,399 张评估图像；异常训练样本来自数据集原测试分区，属于有监督设置。KSDD2 使用 1,865 张训练图像、233 张开发图像、233 张校准图像和 1,004 张测试图像。两个数据集分别训练。具体实现见[协议说明](docs/PROTOCOL.md)。

## 训练与评估

将下面命令中的数据路径替换为本机路径：

```bash
python scripts/preflight.py --dataset aebad --data-root data/AeBAD_S --verify-files --verify-hashes --verify-backbone
python scripts/run_bladegl.py all --dataset aebad --arm E0 --seed 91025 --data-root data/AeBAD_S
python scripts/run_bladegl.py all --dataset ksdd2 --arm E0 --seed 91025 --data-root data/KolektorSDD2
```

`all` 依次完成训练、开发集模型选择、校准和最终评估。默认配置使用 AdamW、批量大小 4 和 3,000 次更新。实验种子为 `91025`、`92025`、`93025`，需要分别执行。

输出保存在 `outputs/bladegl/<dataset>/<arm>/seed_<seed>/`，包括模型检查点、校准参数、图像得分、预测数组和指标。

## 消融设置

通过 `--arm` 选择配置，定义见 [`src/bladegl/variants.py`](src/bladegl/variants.py)。

| 设置 | 参数 |
|---|---|
| 完整 BladeGL | `E0` |
| 整图与局部特征直接融合 | `F0` |
| 整图或仅局部特征定位 | `S1`、`S2`、`S3` |
| 注意力与融合方式 | `F1`—`F8` |
| 区域输入与损失设置 | `R4`、`R5`、`R6` |
| 窗口大小 | `W4`、`W6`、`W8` |
| CBAM 对照分支 | `E01_CBAM` |

固定检查点后，可在 `evaluate` 命令中使用 `--readout head`、`--readout map` 或 `--readout combined --readout-weight 0.5` 比较读出方式。不同读出方式分别校准。

## 单图推理

准备已训练的模型和待检测图像：

```bash
python scripts/infer.py --input path/to/image.png --checkpoint outputs/bladegl/aebad/E0/seed_91025/best.pt --output-dir outputs/inference
```

脚本输出概率数组、预览图和图像得分。如需阈值化结果，增加 `--calibration path/to/calibration.json`，并确保校准文件与当前模型及读出方式一致。

## 许可

项目代码目前尚未选定分发许可证，详见[许可说明](LICENSE_STATUS.md)和[第三方说明](THIRD_PARTY_NOTICES.md)。DINOv2 保留原始 Apache-2.0 许可。
