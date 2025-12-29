# KER-KT 实验运行指南

## 一、环境准备

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据

本项目使用 **EduData** 下载和处理真实数据集。数据准备分为两个步骤：

#### 步骤1：下载数据

使用EduData下载数据集（支持assist09、assist17、junyi）：

```bash
# 下载ASSIST09数据集
python data/download_with_edudata.py --dataset assist09

# 下载ASSIST17数据集
python data/download_with_edudata.py --dataset assist17

# 下载Junyi数据集
python data/download_with_edudata.py --dataset junyi

# 如果数据已下载，可以跳过下载步骤
python data/download_with_edudata.py --dataset assist09 --skip-download
```

**说明**：
- 数据会下载到当前目录下的相应文件夹
- 下载的数据会自动转换为项目需要的格式（CSV）
- 转换后的数据保存在 `./data/{dataset_name}.csv`

#### 步骤2：预处理数据

将下载的CSV数据预处理为实验所需的pkl格式：

```bash
# 处理单个数据集
python data/process_downloaded_data.py --datasets assist09

# 处理多个数据集
python data/process_downloaded_data.py --datasets assist09 assist17 junyi

# 指定输出路径
python data/process_downloaded_data.py --datasets assist09 --output ./data/processed_datasets.pkl
```

**说明**：
- 预处理包括：数据划分（70%训练，10%验证，20%测试）、构建concept_graph、构建Q-matrix
- 处理后的数据保存在 `./data/processed_datasets.pkl`
- 数据划分采用**严格时序划分**（论文4.2.2节要求）

#### 数据准备完成检查

```bash
# 检查pkl文件是否存在
python -c "import os; print('数据文件存在:', os.path.exists('./data/processed_datasets.pkl'))"
```

## 二、运行基线对比实验（论文4.3节）

### 基本用法

运行所有基线模型（DKT, DKVMN, SAKT, AKT, GKT）并与KER-KT对比：

```bash
# 运行所有数据集和所有基线模型（5次运行）
python experiments/run_baseline_experiments.py

# 只运行ASSIST09数据集
python experiments/run_baseline_experiments.py --datasets assist09

# 只运行特定模型
python experiments/run_baseline_experiments.py --models DKT SAKT

# 自定义运行次数和训练轮数
python experiments/run_baseline_experiments.py --n_runs 3 --n_epochs 30

# 指定设备
python experiments/run_baseline_experiments.py --device cuda
```

### 参数说明

- `--datasets`: 数据集列表，默认 `['assist09', 'assist17', 'junyi']`
- `--models`: 基线模型列表，默认 `['DKT', 'DKVMN', 'SAKT', 'AKT', 'GKT']`
- `--n_runs`: 每个模型运行次数，默认 `5`（论文4.3.2节要求）
- `--n_epochs`: 训练轮数，默认 `50`
- `--device`: 设备选择，`auto`/`cpu`/`cuda`，默认 `auto`
- `--save_results`: 结果保存文件名，默认 `baseline_results.json`

### 输出结果

实验结果会：
1. 在控制台打印结果表格（论文表4.5格式）
2. 保存到 `results/baseline_results.json`

结果格式：
```json
[
  {
    "model_name": "DKT",
    "dataset": "assist09",
    "auc_mean": 0.8234,
    "auc_std": 0.0012,
    "acc_mean": 0.7654,
    "acc_std": 0.0008,
    "all_aucs": [0.8231, 0.8235, 0.8238, 0.8230, 0.8236],
    "all_accs": [0.7652, 0.7656, 0.7658, 0.7650, 0.7654],
    "n_runs": 5
  },
  ...
]
```

## 三、运行KER-KT实验

### 基本用法

```bash
# 运行所有数据集
python experiments/run_experiment.py

# 运行特定数据集
python experiments/run_experiment.py --dataset assist09

# 自定义运行次数
python experiments/run_experiment.py --dataset assist09 --n_runs 5
```

### 参数说明

- `--dataset`: 数据集名称，`assist09`/`assist17`/`junyi`/`all`，默认 `all`
- `--n_runs`: 运行次数，默认 `5`（论文4.3.2节要求）

## 四、完整实验流程

### 步骤0：数据准备（首次运行必须）

```bash
# 1. 下载数据
python data/download_with_edudata.py --dataset assist09
python data/download_with_edudata.py --dataset assist17
python data/download_with_edudata.py --dataset junyi

# 2. 预处理数据
python data/process_downloaded_data.py --datasets assist09 assist17 junyi
```

### 步骤1：运行基线模型实验

```bash
# 运行所有基线模型（这可能需要较长时间）
python experiments/run_baseline_experiments.py \
    --datasets assist09 assist17 junyi \
    --models DKT DKVMN SAKT AKT GKT \
    --n_runs 5 \
    --n_epochs 50
```

### 步骤2：运行KER-KT实验

```bash
# 运行KER-KT模型
python experiments/run_experiment.py \
    --dataset all \
    --n_runs 5
```

### 步骤3：合并结果并生成表格

结果会自动保存，可以手动合并KER-KT和基线模型的结果生成完整的对比表格。

## 五、实验配置说明

### 超参数设置（论文表4.4）

每个数据集的超参数已按照论文表4.4自动配置：

- **ASSIST09/ASSIST17**: 
  - embed_dim=128, hidden_dim=256, n_layers=2
  - batch_size=32, dropout=0.2
  
- **Junyi**: 
  - embed_dim=256, hidden_dim=512, n_layers=3
  - batch_size=64, dropout=0.3

### 数据划分（论文4.2.2节）

- 严格时序划分：每个学生前70%→训练，中间10%→验证，最后20%→测试
- 确保不利用"未来"信息

### 评估指标（论文4.3.2节）

- AUC（Area Under ROC Curve）
- ACC（Accuracy）
- 每个模型运行5次，报告均值±标准差

## 六、注意事项

1. **运行时间**：完整实验可能需要数小时甚至数天，建议：
   - 先用小数据集测试（如只运行assist09）
   - 减少运行次数进行快速测试（`--n_runs 1`）
   - 减少训练轮数（`--n_epochs 20`）

2. **GPU使用**：如果有GPU，会自动使用。可以通过 `--device cuda` 强制使用GPU。

3. **内存要求**：
   - ASSIST09/ASSIST17: 约需要4-8GB内存
   - Junyi: 约需要16-32GB内存（数据量大）

4. **结果保存**：所有结果会自动保存到 `results/` 目录。

## 七、快速测试

如果想快速测试代码是否正常工作：

```bash
# 1. 快速数据准备（只下载assist09）
python data/download_with_edudata.py --dataset assist09
python data/process_downloaded_data.py --datasets assist09

# 2. 快速测试：只运行DKT，1次运行，20轮训练
python experiments/run_baseline_experiments.py \
    --datasets assist09 \
    --models DKT \
    --n_runs 1 \
    --n_epochs 20
```

## 八、常见问题

### Q: 数据下载失败怎么办？
A: 
- 检查网络连接
- 确认EduData已正确安装：`pip install EduData`
- 可以手动从EduData官网下载数据

### Q: 数据预处理失败怎么办？
A:
- 确认CSV文件已正确下载到 `./data/{dataset_name}.csv`
- 检查CSV文件格式是否正确（包含student_id, question_id, concept_id, correct, timestamp列）
- 查看错误信息，可能是数据格式问题

### Q: 内存不足怎么办？
A: 可以减少batch_size或使用更小的数据集。

### Q: 训练太慢怎么办？
A: 可以减少n_epochs或n_runs，或者使用GPU加速。

### Q: 如何只运行特定模型？
A: 使用 `--models` 参数指定，如 `--models DKT SAKT`。

### Q: 结果保存在哪里？
A: 保存在 `results/` 目录下，文件名由 `--save_results` 参数指定。

### Q: 数据划分方式是什么？
A: 采用**严格时序划分**（论文4.2.2节）：
- 对每个学生，按时间顺序划分
- 前70% → 训练集
- 中间10% → 验证集
- 最后20% → 测试集
- 确保不利用"未来"信息

