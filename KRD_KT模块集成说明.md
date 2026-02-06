# KRD-KT 模块集成说明

## 已实现的核心模块

### 1. 题目增强模块 (`models/question_enhancement.py`)
- ✅ **QuestionEnhancement**: 标准scaled dot-product attention
- ✅ **QuestionEnhancementSimplified**: 简化版（小论文）
- **功能**: 实现公式(0-6)(0-7)，使用W_q, W_k, W_v进行题目增强

### 2. k阶邻居提取模块 (`models/neighborhood_extractor.py`)
- ✅ **NeighborhoodExtractor**: BFS显式提取k阶邻居
- ✅ **AdaptiveNeighborhoodExtractor**: 自适应top-k邻居选择
- **功能**: 实现公式(0-8)，严格的k跳距离定义

### 3. 路径强度计算模块 (`models/path_strength.py`)
- ✅ **PathStrengthCalculator**: 计算k跳路径强度
- ✅ **PrecomputedPathStrengthCalculator**: 预计算版本（加速训练）
- **功能**: 实现公式(0-9)，支持1阶和2阶路径强度

### 4. 三支决策图模块 (`models/triple_decision_graph.py`)
- ✅ **TripleDecisionGraphComplete**: 完整论文实现
  - 差异化消息传递（MLP_pos, MLP_bnd, MLP_neg）
  - 区域间融合（W_r）
  - 跨阶融合（W_h）
  - 距离衰减（λ）
  - Readout函数
- **功能**: 实现公式(0-13~0-21)的完整流程

### 5. 现有模块（需要更新）
- 🔄 **KRD_KT主模型** (`models/krd_kt.py`): 需要集成新模块
- ✅ **Actor-Critic** (`models/actor_critic.py`): 已实现
- ✅ **KT Predictor** (`models/kt_predictor.py`): 已实现

## 集成计划

### Phase 1: 更新KRD_KT主模型
需要在`models/krd_kt.py`中：
1. 导入新模块
2. 在`__init__`中初始化：
   - QuestionEnhancement
   - NeighborhoodExtractor  
   - PathStrengthCalculator
   - 更新TripleDecisionGraph为新版本
3. 在`forward`中集成题目增强
4. 在训练前预计算邻居和路径强度
5. 更新余弦相似度矩阵的计算

### Phase 2: 修改训练流程
1. **Epoch开始时**：
   - 计算余弦相似度矩阵
   - 提取k阶邻居
   - 预计算路径强度矩阵
   
2. **Forward时**：
   - 题目增强 → 增强后的概念嵌入
   - 三支决策图传播 → 图级嵌入
   - KT预测器 → 预测结果

3. **训练循环**：
   - KT预训练阶段（前20% epochs）
   - RL微调阶段（后80% epochs）

## 关键技术决策记录

### 1. 题目增强
- **决策**: 使用标准attention（W_q, W_k, W_v）
- **理由**: 遵循大论文公式(0-6)(0-7)

### 2. 邻居提取
- **决策**: 2阶邻居排除1阶邻居
- **理由**: 严格的k跳距离定义

### 3. 路径强度
- **决策**: 每个epoch开始时预计算
- **理由**: 平衡准确性和效率

### 4. 消息传递
- **决策**: 边特征e_ij扩展为向量
- **理由**: 便于拼接和MLP处理
- **MLP结构**: [3d → 2d → d]

### 5. 融合机制
- **决策**: 
  - W_r: [d_c, 3*d_c] (区域间)
  - W_h: [d_c, 3*d_c] (跨阶)
- **理由**: 遵循论文公式(0-20)(0-21)

### 6. Readout
- **决策**: 使用mean pooling
- **理由**: 参考小论文，效果稳定

### 7. 余弦相似度更新
- **决策**: 每个epoch开始时计算一次
- **理由**: 平衡动态性和计算成本

### 8. 训练流程
- **阶段1** (前10 epochs): 纯监督KT训练
- **阶段2** (后续epochs): KT + RL联合训练

## 模块依赖关系

```
KRD_KT (主模型)
├── QuestionEnhancement (题目增强)
├── TripleDecisionGraph (图传播)
│   ├── NeighborhoodExtractor (邻居提取)
│   ├── PathStrengthCalculator (路径强度)
│   ├── MLP_pos/bnd/neg (差异化传递)
│   ├── W_r (区域间融合)
│   └── W_h (跨阶融合)
├── KTPredictor (KT预测)
└── ActorCritic (RL优化)
```

## 下一步工作

### 立即任务
1. ✅ 实现所有核心模块
2. 🔄 **当前**: 集成模块到KRD_KT主模型
3. ⏳ 测试端到端训练
4. ⏳ 在ASSIST09上运行实验

### 后续任务
- 验证超参数与论文一致
- 实现DOA评估指标
- 运行完整实验（5次重复）
- 实现消融实验
- 实现基线模型对比

## 测试状态

### 单元测试
- ✅ QuestionEnhancement: 通过
- ✅ NeighborhoodExtractor: 通过
- ✅ PathStrengthCalculator: 通过
- ✅ TripleDecisionGraph: 通过

### 集成测试
- ⏳ 端到端forward: 待测试
- ⏳ 训练循环: 待测试
- ⏳ RL微调: 待测试

## 性能优化记录

1. **概念嵌入更新频率**: 每50个batch更新一次（原10个）
2. **路径强度计算**: 矢量化实现
3. **邻居提取**: 预计算并缓存
4. **余弦相似度**: epoch级缓存

## 参考文献
- 大论文: 张慧玲-论文0201.txt 第3章
- 小论文: KT-RCR简化版本

