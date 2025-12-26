# 使用EduData快速开始指南

## 前置条件

✅ EduData已成功安装（你已完成）

## 完整操作流程（3步完成）

### 步骤1: 下载并转换ASSIST09数据

```bash
python download_with_edudata.py --dataset assist09
```

这个脚本会：
- 使用 `edudata download assistment-2009-2010-skill ./data` 下载数据
- 自动找到下载的CSV文件
- 转换为项目标准格式（student_id, question_id, concept_id, correct, timestamp）
- 保存到 `data/assist09.csv`

**预期输出示例**:
```
============================================================
使用EduData下载 ASSIST09 数据集
============================================================
执行命令: edudata download assistment-2009-2010-skill ./data
...
✓ 下载成功

转换 ASSIST09 数据格式
...
✓ 转换完成
学生数: 4,163
题目数: 17,751
知识点数: 123
答题记录数: 325,637
平均正确率: 0.6900
```

### 步骤2: 生成实验所需的PKL文件

```bash
python process_downloaded_data.py --datasets assist09
```

这个脚本会：
- 读取 `data/assist09.csv`
- 按学生划分train/val/test (70%/10%/20%)
- 构建概念图（concept_graph）
- 构建Q矩阵（question->concept映射）
- 保存到 `data/processed_datasets.pkl`

**预期输出示例**:
```
============================================================
处理数据集: ASSIST09
============================================================
数据集统计:
  学生数: 4,163
  题目数: 17,751
  知识点数: 123
  答题记录数: 325,637

划分数据集...
  训练集: 227,945 条记录, 2,914 个学生
  验证集: 32,564 条记录, 416 个学生  测试集: 65,128 条记录, 833 个学生

✓ 已保存到: ./data/processed_datasets.pkl
```

### 步骤3: 运行实验验证

**快速测试（1个模型，10个epoch）**:
```bash
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1 --n_epochs 10
```

**预期结果**:
- AUC应该 > 0.65（如果接近0.50说明有问题）
- 训练大约需要5-10分钟（CPU）或1-2分钟（GPU）

**完整实验（论文标准）**:
```bash
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT DKVMN SAKT AKT GKT --n_runs 5 --n_epochs 50
```

---

## 如果遇到问题

### 问题1: download_with_edudata.py下载失败

**可能原因**: 网络问题或EduData服务器故障

**解决方案1 - 手动运行edudata命令**:
```bash
# 在PowerShell或命令行中直接运行
edudata download assistment-2009-2010-skill ./data

# 然后运行转换脚本（使用--skip-download跳过下载）
python download_with_edudata.py --dataset assist09 --skip-download
```

**解决方案2 - 检查下载的文件**:
```bash
# 查看下载了什么文件
ls data/assistment-2009-2010-skill/

# 如果看到CSV文件，手动指定路径修改脚本或直接运行步骤2
```

### 问题2: 找不到某个列（例如'skill_id'）

**原因**: EduData下载的文件格式可能不同

**解决方案**:
1. 检查下载的CSV文件的列名:
   ```python
   import pandas as pd
   df = pd.read_csv('data/assistment-2009-2010-skill/XXX.csv')
   print(df.columns.tolist())
   ```

2. 修改`download_with_edudata.py`中的`column_mappings`字典，添加正确的列名映射

### 问题3: process_downloaded_data.py报错

**检查步骤**:
1. 确认`data/assist09.csv`存在
   ```bash
   ls -lh data/assist09.csv
   ```

2. 检查CSV格式是否正确
   ```python
   import pandas as pd
   df = pd.read_csv('data/assist09.csv')
   print(df.head())
   print(df.columns.tolist())
   ```

3. 确保有以下5列: student_id, question_id, concept_id, correct, timestamp

### 问题4: 实验AUC接近0.50（随机猜测）

**可能原因**:
- 数据处理有误
- 模型没有正确训练

**诊断方法**:
```bash
# 检查处理后的数据
python experiments/inspect_pkl_data.py

# 运行诊断脚本
python diagnose_data.py
```

---

## 完整命令速查

```bash
# 1. 下载ASSIST09
python download_with_edudata.py --dataset assist09

# 2. 处理数据
python process_downloaded_data.py --datasets assist09

# 3. 快速测试
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1 --n_epochs 10

# 4. 完整实验（5个模型，每个运行5次）
python experiments/run_baseline_experiments.py \
  --datasets assist09 \
  --models DKT DKVMN SAKT AKT GKT \
  --n_runs 5 \
  --n_epochs 50 \
  --device cuda
```

---

## 下载其他数据集

**ASSIST17**:
```bash
python download_with_edudata.py --dataset assist17
python process_downloaded_data.py --datasets assist17
```

**Junyi**:
```bash
python download_with_edudata.py --dataset junyi
python process_downloaded_data.py --datasets junyi
```

**处理多个数据集**:
```bash
# 下载
python download_with_edudata.py --dataset assist09
python download_with_edudata.py --dataset assist17
python download_with_edudata.py --dataset junyi

# 一次性处理所有数据集
python process_downloaded_data.py --datasets assist09 assist17 junyi
```

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `download_with_edudata.py` | 使用EduData CLI下载并转换数据 |
| `process_downloaded_data.py` | 将CSV处理成实验所需的PKL格式 |
| `data/assist09.csv` | 转换后的CSV数据（中间文件） |
| `data/processed_datasets.pkl` | 实验脚本读取的PKL文件（最终文件） |
| `experiments/run_baseline_experiments.py` | 运行基线模型实验 |

---

## 预期实验结果（ASSIST09）

| 模型 | 预期AUC | 预期ACC |
|------|---------|---------|
| DKT | 0.72-0.75 | 0.70-0.73 |
| DKVMN | 0.73-0.76 | 0.71-0.74 |
| SAKT | 0.74-0.77 | 0.72-0.75 |
| AKT | 0.75-0.78 | 0.73-0.76 |
| GKT | 0.74-0.77 | 0.72-0.75 |

**注意**:
- 具体数值会因数据集版本、超参数设置、随机种子而略有差异
- 如果AUC < 0.60，说明可能有问题
- 如果AUC ≈ 0.50，说明模型没有学到任何东西（数据或模型bug）

---

## 你现在应该做什么？

**立即开始（推荐）**:

```bash
# 在PowerShell中运行以下命令
python download_with_edudata.py --dataset assist09
```

等待下载和转换完成后，运行:

```bash
python process_downloaded_data.py --datasets assist09
```

最后，运行快速验证:

```bash
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1 --n_epochs 10
```

如果一切顺利，你将在10分钟内看到第一个实验结果！
