# 基于 ResNet 的中药细粒度分类

这是一个面向课程展示和项目交付的中药图像分类项目。仓库现在同时包含两套入口：

- CLI：训练、评估、单图预测
- Web：基于 Streamlit 的多页面展示端，支持图片识别、批量识别、视频抽帧识别、数据集概览、训练结果看板和识别历史

默认数据目录仍然使用你当前 Windows 机器上的：

`D:\project\caoyao\50类中草药数据集\split_dataset`

## 项目能力

- 支持 `resnet18 / resnet34 / resnet50 / resnet101`
- 自动读取 `ImageFolder` 数据集结构
- 自动保存 `best.pt`、`last.pt`、训练历史与测试指标
- 训练、评估、预测全流程日志
- 训练双层进度条：`epoch 总进度 + train/val/test batch 进度`
- Streamlit 多页面展示系统
- 单图识别、批量图片识别、视频抽帧识别
- 训练结果看板、数据集概览、系统信息页
- SQLite 持久化识别历史

## 项目结构

```text
.
├── app.py
├── configs/
│   └── default.yaml
├── pages/
│   ├── 01_image_recognition.py
│   ├── 02_batch_recognition.py
│   ├── 03_video_recognition.py
│   ├── 04_dataset_overview.py
│   ├── 05_training_dashboard.py
│   ├── 06_recognition_history.py
│   └── 07_system_info.py
├── src/
│   └── caoyao_resnet/
│       ├── config.py
│       ├── data.py
│       ├── engine.py
│       ├── history_store.py
│       ├── inference_service.py
│       ├── logging_utils.py
│       ├── models.py
│       ├── project_service.py
│       ├── streamlit_views.py
│       └── utils.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── streamlit_bootstrap.py
└── train.py
```

## 依赖安装

### Windows 环境

建议继续使用你已经验证通过的环境：

```powershell
conda create -n caoyao-resnet python=3.10 -y
conda activate caoyao-resnet
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 新增 Web 端依赖

`requirements.txt` 已包含：

- `streamlit`
- `pandas`
- `plotly`
- `opencv-python-headless`

如果你之前环境里只装过训练依赖，更新代码后请重新执行一次：

```powershell
pip install -r requirements.txt
```

## CLI 运行方式

### 1. 快速验证环境

```powershell
python train.py --epochs 1 --batch-size 8 --model-name resnet18 --limit-train-batches 2 --limit-val-batches 1 --limit-test-batches 1
```

### 2. 正式训练

默认是 `resnet50`：

```powershell
python train.py
```

如果要换模型：

```powershell
python train.py --model-name resnet18
python train.py --model-name resnet101
```

### 3. 测试集评估

```powershell
python evaluate.py --checkpoint outputs/resnet18_tcm/best.pt --split test
```

### 4. 单图预测

```powershell
python predict.py --checkpoint outputs/resnet18_tcm/best.pt --image "D:\project\caoyao\50类中草药数据集\split_dataset\test\乌梅\乌梅_101.jpg" --top-k 5
```

## Streamlit Web 运行方式

### 1. 启动应用

```powershell
streamlit run app.py
```

### 2. 页面说明

- 首页：项目介绍、能力概览、快速入口
- 单图识别：上传一张图片，输出 Top-K 结果
- 批量图片识别：上传多张图片，生成结果表和 CSV
- 视频识别：上传视频，按秒抽帧分类并输出汇总
- 数据集概览：展示训练/验证/测试样本分布和样例图片
- 训练结果看板：展示训练曲线、验证准确率和测试指标
- 识别历史：查看单图/批量/视频识别历史
- 系统信息：查看当前模型、设备、数据库和实验信息

### 3. 模型选择规则

- Web 应用默认扫描 `outputs/*/best.pt`
- 页面左侧可以直接选择已发现模型
- 也可以手动输入 checkpoint 路径
- 不会自动下载模型，也不会把 `.pt` 提交到 GitHub

## 输出目录

### 训练输出

训练完成后会在对应实验目录下生成，例如：

- `outputs/resnet18_tcm/`
- `outputs/resnet50_tcm/`

其中包含：

- `best.pt`
- `last.pt`
- `history.json`
- `dataset_summary.json`
- `resolved_config.yaml`
- `test_metrics.json`
- `train.log`
- `evaluate_<split>.log`
- `predict_<image_stem>.log`

### Web 应用输出

Web 端附加产物保存在：

- `outputs/webapp/app_history.db`：SQLite 历史记录数据库
- `outputs/webapp/exports/`：批量识别、视频识别等导出的 CSV

## 默认命名规则

- 如果未显式传 `--run-name`，训练输出目录会默认跟随模型名
- 例如：
  - `--model-name resnet18` -> `outputs/resnet18_tcm`
  - `--model-name resnet50` -> `outputs/resnet50_tcm`

同一个 `run_name` 会持续写入最新的模型和 JSON 结果；日志文件会追加。

## 同步流程

本地开发完成后：

```powershell
git add .
git commit -m "your message"
git push
```

Windows 端同步：

```powershell
cd /d D:\project\resnet-tcm-fine-grained-classification
git pull
pip install -r requirements.txt
streamlit run app.py
```

## 备注

- Windows 下 `num_workers` 默认会自动回落到 `0`
- 如果数据目录改了，可通过 CLI 的 `--data-root` 覆盖
- 如果要断点续训，可用 `--resume outputs/resnet50_tcm/last.pt`
- Web 端默认依赖本地已有 checkpoint；建议先完成至少一次训练再启动识别页面
