# SE-DCNet: 基于双通道融合与注意力机制的无人机射频识别

本项目实现了一个面向无人机射频信号识别的深度学习框架 **SE-DCNet**，核心思想是融合：
- 1D 时域分支（IQ 序列）
- 2D 时频分支（STFT 图）
- SE 注意力融合模块（自适应重标定融合特征）

同时，仓库提供了多组消融/对照模型（1D-CNN、2D-EfficientNet、ResNet、TCN）以及噪声鲁棒性测试脚本，便于复现实验与对比分析。

## 1. 项目结构

```text
SE-DCNet/
├── dataset.py               # 数据读取与双分支特征构建（支持训练期随机加噪）
├── model.py                 # SE-DCNet 与各对照模型定义
├── prepare_labels.py        # 按类别扫描数据并生成 train/val/test 标签文件
├── train.py                 # 训练入口（支持多模型切换）
├── test.py                  # 分类评估与混淆矩阵生成
├── test_robustness.py       # 不同 SNR 下的鲁棒性评估
├── results/                 # 已生成的图表结果
└── weight/                  # 预训练权重
```

## 2. 环境依赖

推荐 Python 3.9+。

安装依赖：

```bash
pip install torch torchvision numpy h5py scipy matplotlib seaborn scikit-learn tqdm
```

如果你使用 CUDA，请安装与你 CUDA 版本匹配的 `torch/torchvision`。

## 3. 数据组织与标签生成

### 3.1 数据目录约定

`prepare_labels.py` 默认从 `./Dataset` 扫描类别子目录，目录名需以 `T` 开头，例如：

```text
Dataset/
├── T10000/
│   ├── xxx.mat
│   └── yyy.mat
├── T10001/
│   └── ...
└── ...
```

### 3.2 单条标签格式

生成的 `train.txt / val.txt / test.txt` 每行格式为：

```text
绝对路径,类别索引,offset
```

其中：
- `绝对路径` 指向 `.mat` 数据文件
- `类别索引` 为从 0 开始的整数标签
- `offset` 为采样起始偏移

### 3.3 生成标签文件

先按需修改 `prepare_labels.py` 中的数据根目录，再运行：

```bash
python prepare_labels.py
```

脚本会自动按 6:2:2 划分并生成：
- `train.txt`
- `val.txt`
- `test.txt`

## 4. 模型说明

### 4.1 Proposed: SE-DCNet（`DualChannelDroneNet`）

- 分支 A（1D）：多层 Conv1D 提取长序列 IQ 时域特征
- 分支 B（2D）：EfficientNet-B0 提取 STFT 时频特征（输入改为 2 通道）
- 融合：特征拼接后进入 `SEBlock` 做通道重标定
- 分类器：全连接 + BN + Dropout

### 4.2 对照模型

- `DroneNet_1D_Only`：仅 1D 分支
- `DroneNet_2D_Only`：仅 2D 分支（EfficientNet）
- `DroneNet_ResNet_Only`：仅 2D 分支（ResNet18）
- `DroneNet_TCN_Only`：仅时域 TCN

## 5. 训练

`train.py` 支持命令行切换模型：

```bash
python train.py --model SE_Dual
python train.py --model 1D_Only
python train.py --model 2D_Only
python train.py --model ResNet_Only
python train.py --model TCN_Only
```

训练过程中会：
- 自动从 `train.txt/val.txt/test.txt` 推断类别数
- 保存最优权重 `best_drone_model_<ModelName>.pth`
- 输出训练曲线与历史：
  - `<ModelName>_loss.png`
  - `<ModelName>_acc.png`
  - `<ModelName>_history.json`

## 6. 测试与评估

### 6.1 标准测试

```bash
python test.py
```

将输出：
- 分类报告（Precision/Recall/F1）
- 混淆矩阵图（默认保存为 `confusion_matrix_Dual.png`）

### 6.2 噪声鲁棒性测试

```bash
python test_robustness.py
```

默认在 SNR `[-10, -5, 0, 5, 10] dB` 上评估多模型，输出：
- `robustness_comparison.png`
- `robustness_results.json`
- 控制台 LaTeX 表格数据（可直接用于论文）

## 7. 预训练权重使用

当前仓库权重位于 `weight/`：

- `best_drone_model_CNN_Only.pth`
- `best_drone_model_EfficientNet_Only.pth`
- `best_drone_model_ResNet_Only.pth`
- `best_drone_model_TCN_Only.pth`
- `best_drone_model_SEdual.pth`

注意：部分脚本默认从项目根目录加载权重文件名。若你直接使用 `weight/` 目录，请任选一种方式：

1. 将对应权重复制到项目根目录；
2. 修改 `test.py` / `test_robustness.py` 中的 `model_path` 或 `paths` 为 `weight/xxx.pth`。

## 8. 已有结果（results/）

仓库已包含若干可视化结果：

- `ablation_bar.png`
- `confusion_matrix_SE-DCNet.png`
- `robustness_comparison.png`
- `SE-DCNet_acc.png`
- `SE-DCNet_loss.png`

## 9. 常见问题

### 9.1 报错找不到权重文件

请检查：
- 权重是否在脚本指定路径
- 文件名是否完全一致（例如 `SEdual` 与 `SE_Dual` 的差异）

### 9.2 类别数不一致

`train.py` 会自动推断类别数，但 `test.py` / `test_robustness.py` 中当前写死为 9 类。若你数据集类别数变化，需要同步修改：
- `class_names`
- `num_classes`

### 9.3 数据读取失败

请确认 `.mat` 文件中包含 IQ 数据键：
- 优先读取 `RF0_I` / `RF0_Q`
- 否则读取文件前两个键作为 I/Q 通道

## 10. 引用与致谢

如果本项目对你的研究有帮助，欢迎在论文或项目中引用本仓库并注明 SE-DCNet。
