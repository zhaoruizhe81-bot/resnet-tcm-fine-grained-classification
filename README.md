# 基于 ResNet 的中药细粒度分类

这是一个从零搭建的 PyTorch 项目，用于在 `50类中草药数据集` 上进行中药图像细粒度分类。项目默认使用 `ResNet50` 作为骨干网络，内置训练、验证、测试和单图预测脚本，并且默认对接你当前 Windows 机器上的数据目录：

`D:\project\caoyao\50类中草药数据集\split_dataset`

目前已按环境核对过：

- 数据集共有 50 个类别。
- `split_dataset` 已包含 `train / val / test` 三个切分。
- 远端 Windows 机器有 Git、Python 和 NVIDIA RTX 4060 Ti。

## 项目结构

```text
.
├── configs/
│   └── default.yaml
├── src/
│   └── caoyao_resnet/
│       ├── config.py
│       ├── data.py
│       ├── engine.py
│       ├── models.py
│       └── utils.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── train.py
```

## 主要功能

- 支持 `resnet18 / resnet34 / resnet50 / resnet101`
- 自动读取 `ImageFolder` 目录结构
- 自动保存 `best.pt` 和 `last.pt`
- 自动输出训练历史、数据集摘要、测试集指标
- 训练、评估、预测全流程日志
- 训练双层进度条（epoch 总进度 + train/val/test batch 进度）
- 支持单图 Top-K 预测
- 支持快速验证模式，只跑前几个 batch 检查环境

## Windows 推荐运行方式

建议先创建独立环境，再安装依赖。

### 1. 创建环境

```powershell
conda create -n caoyao-resnet python=3.10 -y
conda activate caoyao-resnet
```

### 2. 安装依赖

对你现在这台带 `RTX 4060 Ti` 的 Windows 机器，建议直接安装 PyTorch 官方 CUDA 12.6 轮子，再安装项目依赖：

```powershell
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

如果只是想走 CPU，可以改成：

```powershell
pip install torch torchvision
pip install -r requirements.txt
```

之所以这里不直接推荐通用命令，是因为在这台机器上实测通用安装容易落到 `+cpu` 版本，导致无法调用显卡。

### 3. 快速验证环境

第一次建议先跑一个超短的 sanity check：

```powershell
python train.py --epochs 1 --batch-size 8 --model-name resnet18 --limit-train-batches 2 --limit-val-batches 1 --limit-test-batches 1
```

这条命令只会跑极少量 batch，用来确认数据读取、模型前向、保存 checkpoint 都正常。

### 4. 正式训练

```powershell
python train.py
```

如果要换模型：

```powershell
python train.py --model-name resnet18
python train.py --model-name resnet101
```

## 评估与预测

### 测试集评估

```powershell
python evaluate.py --checkpoint outputs/resnet18_tcm/best.pt --split test
```

### 单图预测

```powershell
python predict.py --checkpoint outputs/resnet18_tcm/best.pt --image "D:\project\caoyao\50类中草药数据集\split_dataset\test\乌梅\1.jpg" --top-k 5
```

## 配置说明

默认配置文件位于 `configs/default.yaml`，可调整的核心参数包括：

- 数据根目录
- 图像尺寸
- batch size
- 训练轮数
- 学习率
- 模型名称
- dropout
- 是否启用混合精度

## 输出结果

训练完成后会在对应实验目录下生成，例如 `outputs/resnet18_tcm/` 或 `outputs/resnet50_tcm/`：

- `best.pt`：验证集最优模型
- `last.pt`：最后一个 epoch 模型
- `history.json`：训练历史
- `dataset_summary.json`：类别与样本统计
- `test_metrics.json`：测试集评估结果
- `train.log`：训练完整日志
- `evaluate_<split>.log`：评估日志
- `predict_<image_stem>.log`：预测日志

## 备注

- Windows 下 `num_workers` 默认会自动回落到 `0`，优先保证兼容性。
- 如果你把数据目录改到别的位置，可通过 `--data-root` 参数覆盖。
- 如果你要断点续训，可用 `--resume outputs/resnet50_tcm/last.pt`，或者改成对应模型的目录名。
- 如果未显式传 `--run-name`，输出目录会默认跟随当前模型名，例如 `resnet18_tcm`。
- Windows 上更新代码后可直接执行：

```powershell
git pull
python train.py --help
```
