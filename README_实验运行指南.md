# KRD-KT 实验运行快速指南

本指南提供快速运行实验的命令。详细配置参见 `docs/` 目录。

---

## 🚀 快速开始

### 1. 环境配置
```bash
# 创建环境并安装依赖
conda create -n krd-kt python=3.9
conda activate krd-kt
pip install -r requirements.txt
```

详见：`docs/环境配置指南.md`

### 2. 数据准备
```bash
# ASSIST09和Junyi数据已处理完成，可直接运行实验
# 验证数据文件
ls data/*.pkl
```

详见：`docs/数据准备指南.md`

---

## 🧪 运行实验

### 主实验（论文表4.5）

#### ASSIST09（推荐首选）
```bash
# 单次快速测试
python experiments/run_experiment.py --dataset assist09 --n_runs 1

# 完整5次实验
python experiments/run_experiment.py --dataset assist09 --n_runs 5
```

#### Junyi（大规模）
```bash
python experiments/run_experiment.py --dataset junyi --n_runs 5
```

#### 运行所有数据集
```bash
python experiments/run_experiment.py --dataset all --n_runs 5
```

### 消融实验（论文表4.7）

#### 运行全部6个消融实验
```bash
python experiments/run_ablation.py --dataset assist09 --ablation all --n_runs 5
```

#### 运行单个消融实验
```bash
# w/o 3WD - 移除三支决策
python experiments/run_ablation.py --dataset assist09 --ablation "w/o_3WD" --n_runs 5

# w/o Multiorder - 只用1阶邻居
python experiments/run_ablation.py --dataset assist09 --ablation "w/o_Multiorder" --n_runs 5

# w/o Diff-Msg - 移除差异化消息传递
python experiments/run_ablation.py --dataset assist09 --ablation "w/o_Diff-Msg" --n_runs 5

# w/o Decay - 移除距离衰减
python experiments/run_ablation.py --dataset assist09 --ablation "w/o_Decay" --n_runs 5

# w/o NegSupp - 移除负域抑制
python experiments/run_ablation.py --dataset assist09 --ablation "w/o_NegSupp" --n_runs 5

# w/o RL - 纯监督学习
python experiments/run_ablation.py --dataset assist09 --ablation "w/o_RL" --n_runs 5
```

### 基线对比（论文表4.5）

```bash
# 运行所有基线模型
python experiments/run_baseline_experiments.py --dataset assist09 --n_runs 5

# 运行特定基线
python experiments/run_baseline_experiments.py --dataset assist09 --models DKT DKVMN --n_runs 5
```

---

## 📊 实验结果

### 结果保存位置
- 主实验：`checkpoints/{dataset}/krd_kt_best_run{N}.pt`
- 日志文件：控制台输出包含详细训练过程
- 最终指标：训练结束时自动显示

### 预期性能（论文表4.5）

| 数据集 | AUC | ACC |
|--------|-----|-----|
| ASSIST09 | 0.819±0.003 | 0.761±0.002 |
| Junyi | 0.852±0.004 | 0.782±0.003 |

---

## ⚙️ 常用参数

```bash
--dataset    # assist09/junyi/ednet/all
--n_runs     # 运行次数（默认5）
--device     # cuda/cpu（自动检测）
--ablation   # 消融实验类型
--models     # 基线模型列表
```

详见：`docs/实验配置说明.md`

---

## 🐛 常见问题

**Q: 数据文件不存在？**
```bash
# 检查数据文件
python -c "import os; print(os.path.exists('data/assist09_processed.pkl'))"
```

**Q: CUDA不可用？**
```bash
# 验证PyTorch和CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: 内存不足？**
- 减小batch_size（在`experiments/run_experiment.py`中修改）
- 减小max_seq_len
- 使用CPU（速度较慢）

更多问题见：`docs/环境配置指南.md`

---

## 📁 项目结构

```
experiments/
├── run_experiment.py           # 主实验
├── run_ablation.py            # 消融实验
└── run_baseline_experiments.py # 基线对比

docs/
├── 实验配置说明.md             # 详细超参数配置
├── 数据准备指南.md             # 数据集处理说明
└── 环境配置指南.md             # 环境安装指南
```

---

**最后更新**: 2026-02-06

