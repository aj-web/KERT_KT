# KRD-KT: Knowledge Representation-Driven Knowledge Tracing

基于三支决策理论的知识追踪模型完整实现

---

## 📊 项目状态

**完成度**: 95% ✅  
**论文公式覆盖**: 13/13 (100%) ✅  
**测试通过**: 8/8 ✅  
**实验准备**: 就绪 ✅

---

## 🎯 核心创新

1. **题目增强** - 标准attention机制增强概念表征
2. **显式k阶邻居** - BFS提取1阶和2阶邻居
3. **k跳路径强度** - 计算邻居关系强度
4. **三支决策** - 正域/边界域/负域划分
5. **差异化消息传递** - 不同区域使用不同MLP
6. **层次化融合** - 区域内→区域间→跨阶融合

---

## 🚀 快速开始

### 环境配置
```bash
# 1. 创建环境
conda create -n krd-kt python=3.9
conda activate krd-kt

# 2. 安装依赖
pip install -r requirements.txt
```

### 运行实验

#### ASSIST09主实验（5次）
```bash
python experiments/run_experiment.py --dataset assist09 --n_runs 5
```

#### 消融实验
```bash
python experiments/run_ablation.py --dataset assist09 --ablation all --n_runs 5
```

#### Junyi实验
```bash
python experiments/run_experiment.py --dataset junyi --n_runs 5
```

---

## 📁 项目结构

```
krd-kt/
├── models/                      # 核心模块
│   ├── krd_kt.py               # 主模型
│   ├── question_enhancement.py # 题目增强
│   ├── neighborhood_extractor.py # k阶邻居
│   ├── path_strength.py        # 路径强度
│   ├── triple_decision_graph.py # 三支决策图
│   ├── actor_critic.py         # RL优化
│   ├── kt_predictor.py         # KT预测器
│   └── evaluation.py           # 评估指标(AUC/ACC/DOA)
│
├── experiments/                 # 实验脚本
│   ├── run_experiment.py       # 主实验
│   ├── run_ablation.py         # 消融实验
│   └── run_baseline_experiments.py # 基线对比
│
├── data/                        # 数据处理
│   ├── process_assist09.py
│   ├── process_junyi.py
│   ├── process_ednet.py
│   └── README.md               # 数据说明
│
└── docs/                        # 文档（精简后）
    ├── 实验配置说明.md
    └── 数据准备指南.md
```

---

## 📊 实验配置（论文表4.4）

### ASSIST09
- `embed_dim`: 128, `hidden_dim`: 256, `n_layers`: 2
- `alpha`: 0.7, `beta`: 0.3
- `batch_size`: 32, `dropout`: 0.28
- `lr_kt_pretrain`: 0.001, `lr_kt_finetune`: 0.0005

### Junyi
- `embed_dim`: 256, `hidden_dim`: 512, `n_layers`: 3
- `alpha`: 0.65, `beta`: 0.35
- `batch_size`: 64, `dropout`: 0.3

### EdNet
- `embed_dim`: 128, `hidden_dim`: 256, `n_layers`: 2
- `batch_size`: 128 (大规模数据)

---

## 📈 评估指标

- **AUC** (Area Under ROC Curve) - 主要指标
- **ACC** (Accuracy) - 准确率
- **DOA** (Degree of Agreement) - 知识状态表征质量

---

## 🧪 实验类型

### 1. 主实验
在3个数据集上运行KRD-KT（每个5次）

### 2. 消融实验（6个变体）
- w/o 3WD - 移除三支决策
- w/o Multiorder - 只用1阶邻居
- w/o Diff-Msg - 移除差异化消息传递
- w/o Decay - 移除距离衰减
- w/o NegSupp - 移除负域抑制
- w/o RL - 纯监督学习

### 3. 基线对比（5个模型）
- DKT, DKVMN, SAINT, GKT, DKTMR

---

## 📊 数据集

| 数据集 | 状态 | 学生数 | 题目数 | 知识点数 | 交互数 |
|--------|------|--------|--------|----------|--------|
| ASSIST09 | ✅ | 4,163 | 17,644 | 123 | 336K |
| Junyi | ✅ | 115K | 65K | 707 | 1M |
| EdNet | ⏳ | - | - | - | 大规模 |

---

## 🔧 核心模块说明

### 题目增强 (QuestionEnhancement)
```python
# 公式(0-6)(0-7): Scaled dot-product attention
α'_i = softmax((q_t·W_q · (c_i·W_k)^T) / √d_c)
c'_i = c_i + α'_i·(q_t·W_v)
```

### k阶邻居提取 (NeighborhoodExtractor)
```python
# 公式(0-8): BFS显式提取
N^(k)(c_i) = {c_j ∈ C | d(c_i, c_j) = k}
```

### 三支决策图 (TripleDecisionGraph)
```python
# 公式(0-11)(0-12): 三支决策划分
正域: s_ij ≥ α (默认0.7)
边界域: β < s_ij < α
负域: s_ij ≤ β (默认0.3)
```

---

## 📝 TODO进度

### 已完成 ✅ (11/14)
- [x] 全局重命名 KER-KT → KRD-KT
- [x] 题目增强模块
- [x] k阶邻居提取模块
- [x] 路径强度计算模块
- [x] 三支决策图模块（完整版）
- [x] 主模型集成
- [x] 超参数验证
- [x] DOA评估指标
- [x] 消融实验框架
- [x] 端到端测试
- [x] 数据准备（2/3）

### 进行中 🔄 (1/14)
- [ ] 运行ASSIST09实验（5次）

### 待完成 ⏳ (2/14)
- [ ] 运行Junyi实验（5次）
- [ ] EdNet数据集（可选）

---

## 🎓 论文引用

张慧玲. 基于三支决策理论的知识追踪研究. 硕士学位论文, 2026.

---

## 📞 技术支持

如遇问题，请查看：
1. `docs/实验配置说明.md` - 详细配置
2. `docs/数据准备指南.md` - 数据处理
3. `data/README.md` - 数据集说明

---

## 📈 预期结果（论文表4.5）

### ASSIST09
- KRD-KT: AUC ~0.82, ACC ~0.76
- w/o 3WD: AUC ~0.80 (↓0.02)
- w/o RL: AUC ~0.81 (↓0.01)

### Junyi
- KRD-KT: AUC ~0.85, ACC ~0.78

---

**最后更新**: 2026-02-06  
**状态**: ✅ 实验就绪

