# Experiments 目录结构详解

> **最后更新**: 2026-02-13  
> **版本**: 1.0.0

本文档详细说明 `experiments/` 目录的组织结构和设计理念。

---

## 📁 完整目录树

```
experiments/
│
├── __init__.py                              # Python 包初始化
├── README.md                                # 目录说明文档
├── STRUCTURE.md                             # 本文件：结构详解
│
├── core/                                    # 核心实验脚本
│   ├── __init__.py
│   ├── run_experiment.py                    # 主模型训练（KRD-KT及变体）
│   ├── run_baseline_experiments.py          # 基线模型对比实验
│   └── run_ablation.py                      # 消融实验（已集成到run_experiment.py）
│
├── visualization/                           # 可视化脚本
│   ├── __init__.py
│   ├── visualize_knowledge_state.py         # 知识状态演化可视化
│   ├── visualize_triple_decision.py         # 三支决策邻域划分可视化
│   └── generate_all_visualizations.py       # 一键生成所有可视化
│
├── analysis/                                # 分析脚本
│   ├── __init__.py
│   ├── hyperparameter_sensitivity.py        # 超参数敏感性分析
│   ├── analyze_results.py                   # 实验结果分析和统计
│   └── baseline_comparison.py               # 基线模型对比分析
│
└── utils/                                   # 工具和诊断脚本
    ├── __init__.py
    ├── diagnose_krd_kt.py                   # 诊断 KRD-KT 模型
    ├── diagnose_training.py                 # 诊断训练过程
    ├── diagnose_issue.py                    # 诊断一般问题
    ├── inspect_pkl_data.py                  # 检查数据集文件
    └── quick_test.py                        # 快速测试
```

---

## 🎯 设计理念

### 1. 按功能分类

| 目录 | 功能定位 | 使用频率 | 重要性 |
|------|----------|----------|--------|
| `core/` | 核心实验运行 | 高 | ⭐⭐⭐⭐⭐ |
| `visualization/` | 论文图表生成 | 中 | ⭐⭐⭐⭐ |
| `analysis/` | 结果分析 | 中 | ⭐⭐⭐⭐ |
| `utils/` | 开发调试 | 低 | ⭐⭐⭐ |

### 2. 清晰的职责划分

- **core/**: 只包含核心实验脚本，直接对应论文的实验章节
- **visualization/**: 专注于可视化，不包含实验逻辑
- **analysis/**: 专注于分析，不包含训练逻辑
- **utils/**: 辅助工具，不直接用于论文实验

### 3. 独立性原则

每个脚本都可以独立运行，不依赖于其他实验脚本。

---

## 📂 各目录详细说明

### 1. `core/` - 核心实验脚本 ⭐⭐⭐⭐⭐

**定位**: 论文实验的核心，包含所有训练和对比实验。

#### 文件说明

##### `run_experiment.py` (主模型训练)

**功能**:
- 训练 KRD-KT 模型及其所有变体
- 支持 7 种模式（full, sl, wo_3wd, wo_multi, wo_diff, wo_decay, wo_neg）
- 自动保存检查点和结果

**对应论文**:
- 4.4 节：预测性能对比实验
- 4.7 节：消融实验

**关键参数**:
```bash
--dataset      # 数据集 (assist09/junyi/ednet)
--mode         # 模型变体 (full/sl/wo_3wd/...)
--n_runs       # 运行次数（默认5）
```

**输出**:
```
checkpoints/{dataset}/
  └── krd_kt_{mode}_run{N}_{dataset}.pt

results/
  ├── krd_kt_{mode}_run{N}_{dataset}.json
  └── krd_kt_{mode}_{dataset}.json  # 汇总
```

**使用场景**:
- 阶段2：主模型训练
- 阶段4：消融实验

---

##### `run_baseline_experiments.py` (基线对比)

**功能**:
- 训练 5 个基线模型（DKT, DKVMN, SAKT, AKT, GKT）
- 自动生成对比结果
- 保存详细的实验数据

**对应论文**:
- 4.4 节：预测性能对比实验

**关键参数**:
```bash
--dataset      # 数据集
--models       # 要运行的基线模型列表
--n_runs       # 运行次数（默认5）
```

**输出**:
```
results/baselines/
  ├── dkt_{dataset}.json
  ├── dkvmn_{dataset}.json
  ├── sakt_{dataset}.json
  ├── akt_{dataset}.json
  ├── gkt_{dataset}.json
  └── baseline_comparison_{dataset}.json
```

**使用场景**:
- 阶段3：基线模型对比实验

---

##### `run_ablation.py` (消融实验)

**说明**: 此功能已集成到 `run_experiment.py` 的 `--mode` 参数中。

**迁移方式**:
```bash
# 旧方式（如果有单独的脚本）
python experiments/run_ablation.py --variant wo_3wd

# 新方式（推荐）
python experiments/core/run_experiment.py --mode wo_3wd
```

---

### 2. `visualization/` - 可视化脚本 ⭐⭐⭐⭐

**定位**: 生成论文所需的所有图表和可视化。

#### 文件说明

##### `visualize_knowledge_state.py` (知识状态演化)

**功能**:
- 可视化学生知识状态随时间的演化
- 生成热力图（时间 × 知识点）
- 自动选择有代表性的学生

**对应论文**:
- 4.8 节：知识状态演化可视化
- 论文图4-6

**关键参数**:
```bash
--dataset      # 数据集
--checkpoint   # 模型检查点路径
--student_id   # 学生ID（可选，默认随机）
--n_concepts   # 可视化的知识点数量（默认8）
```

**输出**:
```
figures/knowledge_state/
  ├── knowledge_evolution_student{id}_{dataset}.png
  └── knowledge_evolution_student{id}_{dataset}.pdf
```

---

##### `visualize_triple_decision.py` (三支决策划分)

**功能**:
- 可视化三支决策的邻域划分
- 生成网络图和柱状图
- 展示正域、边界域、负域

**对应论文**:
- 4.9 节：三支决策邻域划分可视化
- 论文图4-7

**关键参数**:
```bash
--dataset      # 数据集
--checkpoint   # 模型检查点路径
--concept_id   # 知识点ID（可选，默认随机）
--k            # 邻域阶数（1或2）
```

**输出**:
```
figures/triple_decision/
  ├── triple_decision_graph_c{id}_k{k}_{dataset}.png
  ├── triple_decision_graph_c{id}_k{k}_{dataset}.pdf
  ├── triple_decision_bar_c{id}_k{k}_{dataset}.png
  └── triple_decision_bar_c{id}_k{k}_{dataset}.pdf
```

---

##### `generate_all_visualizations.py` (一键生成)

**功能**:
- 批量运行所有可视化脚本
- 自动生成论文所需的所有图表
- 支持选择性跳过

**关键参数**:
```bash
--dataset              # 数据集
--checkpoint           # 模型检查点路径
--n_students           # 知识状态可视化的学生数量
--n_concepts           # 三支决策可视化的知识点数量
--skip_sensitivity     # 跳过敏感性分析
--skip_knowledge_state # 跳过知识状态可视化
--skip_triple_decision # 跳过三支决策可视化
```

**使用场景**:
- 批量生成论文图表
- 快速验证可视化效果

---

### 3. `analysis/` - 分析脚本 ⭐⭐⭐⭐

**定位**: 实验结果的深入分析和统计。

#### 文件说明

##### `hyperparameter_sensitivity.py` (超参数敏感性)

**功能**:
- 分析 4 个关键超参数的影响
- 自动运行多组实验
- 生成参数-性能曲线图

**对应论文**:
- 4.6 节：超参数敏感性分析
- 论文图4-4、图4-5

**支持的参数**:
- `lambda_decay`: 距离衰减系数
- `max_k`: 邻域阶数
- `alpha`: 三支决策正域阈值
- `beta`: 三支决策负域阈值

**关键参数**:
```bash
--dataset      # 数据集
--param        # 要分析的参数
--values       # 参数值列表（可选）
--n_runs       # 每组参数运行次数（默认3）
```

**输出**:
```
results/sensitivity/
  └── sensitivity_{param}_{dataset}.json

figures/sensitivity/
  ├── sensitivity_{param}_{dataset}.png
  └── sensitivity_{param}_{dataset}.pdf
```

---

##### `analyze_results.py` (结果分析)

**功能**:
- 统计分析实验结果
- 生成对比表格
- 计算显著性检验

**关键参数**:
```bash
--results_dir  # 结果目录
--output       # 输出文件
```

---

##### `baseline_comparison.py` (基线对比分析)

**功能**:
- 分析基线模型对比结果
- 生成对比图表
- 计算改进百分比

**关键参数**:
```bash
--dataset      # 数据集
--baseline_dir # 基线结果目录
```

---

### 4. `utils/` - 工具和诊断脚本 ⭐⭐⭐

**定位**: 开发和调试过程中使用的辅助工具。

#### 文件说明

##### `diagnose_krd_kt.py` (模型诊断)

**功能**:
- 诊断 KRD-KT 模型的各个组件
- 检查模型参数和前向传播
- 验证图模块初始化

---

##### `diagnose_training.py` (训练诊断)

**功能**:
- 诊断训练过程中的问题
- 检查梯度流
- 分析损失曲线

---

##### `diagnose_issue.py` (一般问题诊断)

**功能**:
- 诊断一般性问题
- 环境检查
- 依赖验证

---

##### `inspect_pkl_data.py` (数据检查)

**功能**:
- 检查 `.pkl` 数据集文件
- 显示数据统计信息
- 验证数据格式

**关键参数**:
```bash
--file         # .pkl 文件路径
--verbose      # 详细输出
```

---

##### `quick_test.py` (快速测试)

**功能**:
- 快速测试模型和数据加载
- 验证环境配置
- 单元测试

---

## 🚀 使用指南

### 按实验阶段使用

#### 阶段1：数据准备
```bash
# 在 data/ 目录下运行
python data/process_assist09.py
python data/process_junyi.py
```

#### 阶段2：主模型训练
```bash
python experiments/core/run_experiment.py \
    --dataset assist09 \
    --mode sl \
    --n_runs 5
```

#### 阶段3：基线对比
```bash
python experiments/core/run_baseline_experiments.py \
    --dataset assist09 \
    --n_runs 5
```

#### 阶段4：消融实验
```bash
# 运行所有消融变体
for mode in wo_3wd wo_multi wo_diff wo_decay wo_neg; do
    python experiments/core/run_experiment.py \
        --dataset assist09 \
        --mode $mode \
        --n_runs 5
done
```

#### 阶段5：超参数敏感性分析
```bash
# 分析所有超参数
for param in lambda_decay max_k alpha beta; do
    python experiments/analysis/hyperparameter_sensitivity.py \
        --dataset assist09 \
        --param $param \
        --n_runs 3
done
```

#### 阶段6：可视化
```bash
python experiments/visualization/generate_all_visualizations.py \
    --dataset assist09 \
    --checkpoint checkpoints/assist09/krd_kt_sl_run1_assist09.pt \
    --n_students 5 \
    --n_concepts 5
```

---

### 按功能使用

#### 训练模型
```bash
# 使用 core/ 目录下的脚本
python experiments/core/run_experiment.py [options]
python experiments/core/run_baseline_experiments.py [options]
```

#### 生成图表
```bash
# 使用 visualization/ 目录下的脚本
python experiments/visualization/visualize_knowledge_state.py [options]
python experiments/visualization/visualize_triple_decision.py [options]
```

#### 分析结果
```bash
# 使用 analysis/ 目录下的脚本
python experiments/analysis/hyperparameter_sensitivity.py [options]
python experiments/analysis/analyze_results.py [options]
```

#### 调试问题
```bash
# 使用 utils/ 目录下的脚本
python experiments/utils/diagnose_krd_kt.py
python experiments/utils/inspect_pkl_data.py --file data/assist09_processed.pkl
```

---

## 📊 输出文件组织

```
project_root/
│
├── results/                                 # 实验结果
│   ├── krd_kt_sl_assist09.json             # 主模型结果（汇总）
│   ├── krd_kt_sl_run1_assist09.json        # 主模型结果（单次）
│   ├── krd_kt_wo_3wd_assist09.json         # 消融实验结果
│   │
│   ├── baselines/                          # 基线模型结果
│   │   ├── dkt_assist09.json
│   │   ├── dkvmn_assist09.json
│   │   └── baseline_comparison_assist09.json
│   │
│   └── sensitivity/                        # 敏感性分析结果
│       ├── sensitivity_lambda_decay_assist09.json
│       └── sensitivity_max_k_assist09.json
│
├── figures/                                # 可视化图表
│   ├── sensitivity/                        # 敏感性分析图
│   │   ├── sensitivity_lambda_decay_assist09.png
│   │   └── sensitivity_lambda_decay_assist09.pdf
│   │
│   ├── knowledge_state/                    # 知识状态演化图
│   │   ├── knowledge_evolution_student0_assist09.png
│   │   └── knowledge_evolution_student0_assist09.pdf
│   │
│   └── triple_decision/                    # 三支决策划分图
│       ├── triple_decision_graph_c10_k1_assist09.png
│       └── triple_decision_bar_c10_k1_assist09.pdf
│
└── checkpoints/                            # 模型检查点
    ├── assist09/
    │   ├── krd_kt_sl_run1_assist09.pt
    │   └── krd_kt_wo_3wd_run1_assist09.pt
    │
    └── junyi/
        └── krd_kt_sl_run1_junyi.pt
```

---

## 🔧 开发规范

### 添加新脚本的步骤

1. **确定脚本类型**
   - 核心实验 → `core/`
   - 可视化 → `visualization/`
   - 分析 → `analysis/`
   - 工具 → `utils/`

2. **创建脚本文件**
   ```python
   """
   脚本功能说明
   论文章节: X.X
   """
   
   import argparse
   # ... imports ...
   
   def main():
       parser = argparse.ArgumentParser(description='...')
       # ... 添加参数 ...
       args = parser.parse_args()
       # ... 实现功能 ...
   
   if __name__ == "__main__":
       main()
   ```

3. **遵循命名规范**
   - 使用小写字母和下划线
   - 名称应清晰表达功能
   - 例如: `run_experiment.py`, `visualize_knowledge_state.py`

4. **标准参数**
   所有脚本应支持（如适用）：
   - `--dataset`: 数据集名称
   - `--n_runs`: 运行次数
   - `--device`: 设备选择
   - `--output_dir`: 输出目录

5. **更新文档**
   - 更新 `README.md`
   - 更新 `STRUCTURE.md`（本文件）
   - 更新 `docs/实验任务清单.md`

---

## 📖 相关文档

- **实验任务清单**: `docs/实验任务清单.md`
- **阶段5和6使用指南**: `docs/阶段5和阶段6使用指南.md`
- **消融实验使用指南**: `docs/消融实验使用指南.md`
- **环境配置指南**: `docs/环境配置指南.md`
- **项目总体规划**: `项目总体规划.md`

---

## ⚠️ 注意事项

### 1. 路径问题

由于脚本在子目录中，运行时需要注意：

```bash
# ✅ 正确：从项目根目录运行
python experiments/core/run_experiment.py --dataset assist09

# ❌ 错误：从 experiments/ 目录运行
cd experiments
python core/run_experiment.py --dataset assist09  # 可能导致路径错误
```

### 2. 导入问题

所有脚本的导入已正确配置：

```python
# 脚本内部会自动添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
```

### 3. 向后兼容

如果之前的文档或脚本使用旧路径：

```bash
# 旧路径（如果之前在根目录）
python run_experiment.py --dataset assist09

# 新路径
python experiments/core/run_experiment.py --dataset assist09
```

---

## 🎯 最佳实践

### 1. 实验管理

```bash
# 使用 nohup 后台运行长时间实验
nohup python experiments/core/run_experiment.py \
    --dataset assist09 --mode sl --n_runs 5 \
    > logs/assist09_sl.log 2>&1 &

# 使用 tmux 或 screen 管理会话
tmux new -s experiments
python experiments/core/run_experiment.py --dataset assist09 --mode sl --n_runs 5
# Ctrl+B, D 分离会话
```

### 2. 结果备份

```bash
# 定期备份结果
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/ figures/ checkpoints/
```

### 3. 批量运行

```bash
# 使用脚本批量运行实验
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash
for dataset in assist09 junyi; do
    for mode in sl wo_3wd wo_multi wo_diff wo_decay wo_neg; do
        python experiments/core/run_experiment.py \
            --dataset $dataset \
            --mode $mode \
            --n_runs 5
    done
done
EOF

chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

---

## 📈 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| 1.0.0 | 2026-02-13 | 初始版本，建立目录结构 |

---

**维护者**: KRD-KT Team  
**最后更新**: 2026-02-13

