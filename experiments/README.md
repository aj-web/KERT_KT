# Experiments 目录说明

本目录包含 KRD-KT 项目的所有实验脚本，按功能分类组织。

---

## 📁 目录结构

```
experiments/
├── core/                    # 核心实验脚本
├── visualization/           # 可视化脚本
├── analysis/               # 分析脚本
├── utils/                  # 工具和诊断脚本
└── __init__.py            # Python 包初始化文件
```

---

## 📂 各目录详细说明

### 1. `core/` - 核心实验脚本

**用途**: 主要的实验运行脚本，用于训练模型和对比实验

| 文件 | 功能 | 论文章节 |
|------|------|----------|
| `run_experiment.py` | 主模型训练（KRD-KT及其变体） | 4.4 |
| `run_baseline_experiments.py` | 基线模型对比实验 | 4.4 |
| `run_ablation.py` | 消融实验 | 4.7 |

**使用示例**:
```bash
# 主模型训练
python experiments/core/run_experiment.py --dataset assist09 --mode sl --n_runs 5

# 基线对比
python experiments/core/run_baseline_experiments.py --dataset assist09 --n_runs 5

# 消融实验
python experiments/core/run_ablation.py --dataset assist09 --mode wo_3wd --n_runs 5
```

---

### 2. `visualization/` - 可视化脚本

**用途**: 生成论文所需的各种可视化图表

| 文件 | 功能 | 论文章节 |
|------|------|----------|
| `visualize_knowledge_state.py` | 知识状态演化可视化 | 4.8 |
| `visualize_triple_decision.py` | 三支决策邻域划分可视化 | 4.9 |
| `generate_all_visualizations.py` | 一键生成所有可视化 | 4.8-4.9 |

**使用示例**:
```bash
# 知识状态演化
python experiments/visualization/visualize_knowledge_state.py \
    --dataset assist09 \
    --checkpoint checkpoints/assist09/krd_kt_sl_run1_assist09.pt

# 三支决策划分
python experiments/visualization/visualize_triple_decision.py \
    --dataset assist09 \
    --checkpoint checkpoints/assist09/krd_kt_sl_run1_assist09.pt \
    --k 1

# 一键生成所有可视化
python experiments/visualization/generate_all_visualizations.py \
    --dataset assist09 \
    --checkpoint checkpoints/assist09/krd_kt_sl_run1_assist09.pt
```

---

### 3. `analysis/` - 分析脚本

**用途**: 实验结果分析和超参数敏感性分析

| 文件 | 功能 | 论文章节 |
|------|------|----------|
| `hyperparameter_sensitivity.py` | 超参数敏感性分析 | 4.6 |
| `analyze_results.py` | 实验结果分析和统计 | - |
| `baseline_comparison.py` | 基线模型对比分析 | 4.4 |

**使用示例**:
```bash
# 超参数敏感性分析
python experiments/analysis/hyperparameter_sensitivity.py \
    --dataset assist09 \
    --param lambda_decay \
    --n_runs 3

# 分析实验结果
python experiments/analysis/analyze_results.py \
    --results_dir results/

# 基线对比分析
python experiments/analysis/baseline_comparison.py \
    --dataset assist09
```

---

### 4. `utils/` - 工具和诊断脚本

**用途**: 开发和调试过程中使用的辅助工具

| 文件 | 功能 | 用途 |
|------|------|------|
| `diagnose_krd_kt.py` | 诊断 KRD-KT 模型 | 调试 |
| `diagnose_training.py` | 诊断训练过程 | 调试 |
| `diagnose_issue.py` | 诊断一般问题 | 调试 |
| `inspect_pkl_data.py` | 检查数据集文件 | 数据验证 |
| `quick_test.py` | 快速测试 | 开发测试 |

**使用示例**:
```bash
# 检查数据集
python experiments/utils/inspect_pkl_data.py \
    --file data/assist09_processed.pkl

# 诊断模型
python experiments/utils/diagnose_krd_kt.py \
    --dataset assist09

# 快速测试
python experiments/utils/quick_test.py
```

---

## 🚀 快速开始

### 1. 完整实验流程

```bash
# 阶段1: 数据准备（在 data/ 目录）
python data/process_assist09.py
python data/process_junyi.py

# 阶段2: 主模型训练
python experiments/core/run_experiment.py --dataset assist09 --mode sl --n_runs 5

# 阶段3: 基线对比
python experiments/core/run_baseline_experiments.py --dataset assist09 --n_runs 5

# 阶段4: 消融实验
python experiments/core/run_ablation.py --dataset assist09 --mode wo_3wd --n_runs 5

# 阶段5: 超参数敏感性分析
python experiments/analysis/hyperparameter_sensitivity.py \
    --dataset assist09 --param lambda_decay --n_runs 3

# 阶段6: 可视化
python experiments/visualization/generate_all_visualizations.py \
    --dataset assist09 \
    --checkpoint checkpoints/assist09/krd_kt_sl_run1_assist09.pt
```

### 2. 调试和验证

```bash
# 检查数据集
python experiments/utils/inspect_pkl_data.py --file data/assist09_processed.pkl

# 快速测试模型
python experiments/utils/quick_test.py

# 诊断训练问题
python experiments/utils/diagnose_training.py --dataset assist09
```

---

## 📊 输出文件位置

```
project_root/
├── results/
│   ├── krd_kt_sl_{dataset}.json           # 主模型结果
│   ├── baselines/                         # 基线模型结果
│   │   ├── dkt_{dataset}.json
│   │   └── baseline_comparison_{dataset}.json
│   └── sensitivity/                       # 敏感性分析结果
│       └── sensitivity_{param}_{dataset}.json
│
├── figures/
│   ├── sensitivity/                       # 敏感性分析图表
│   ├── knowledge_state/                   # 知识状态演化图
│   └── triple_decision/                   # 三支决策划分图
│
└── checkpoints/
    ├── assist09/                          # ASSIST09 模型检查点
    └── junyi/                             # Junyi 模型检查点
```

---

## 📖 相关文档

- **实验任务清单**: `docs/实验任务清单.md`
- **阶段5和6使用指南**: `docs/阶段5和阶段6使用指南.md`
- **消融实验使用指南**: `docs/消融实验使用指南.md`
- **环境配置指南**: `docs/环境配置指南.md`

---

## 🔧 开发说明

### 添加新的实验脚本

1. 确定脚本类型（core/visualization/analysis/utils）
2. 放入对应的子目录
3. 遵循现有的命令行参数风格
4. 更新本 README 文档

### 命令行参数规范

所有脚本应支持以下标准参数（如适用）：
- `--dataset`: 数据集名称 (assist09/junyi/ednet)
- `--n_runs`: 运行次数
- `--device`: 设备 (auto/cpu/cuda)
- `--checkpoint`: 模型检查点路径

---

## ⚠️ 注意事项

1. **路径问题**: 由于脚本移动到了子目录，运行时需要使用相对于项目根目录的路径
2. **导入问题**: 所有脚本的导入路径已经正确配置，无需修改
3. **向后兼容**: 旧的命令行调用方式仍然有效，只需更新路径

---

**最后更新**: 2026-02-13  
**版本**: 1.0.0

