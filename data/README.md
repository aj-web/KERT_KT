# 数据集处理说明

本目录包含KRD-KT模型所需的三个数据集的独立处理脚本。

## 📁 文件结构

```
data/
├── README.md                  # 本文件
├── process_assist09.py        # ASSIST09数据集处理
├── process_ednet.py           # EdNet数据集处理
├── process_junyi.py           # Junyi数据集处理
├── download_ednet.py          # EdNet数据下载（独立）
└── processed_datasets.pkl     # 最终合并的数据文件
```

## 🎯 数据集概览

| 数据集 | 学生数 | 题目数 | 知识点数 | 交互数 | 处理脚本 |
|--------|--------|--------|----------|--------|----------|
| ASSIST09 | ~4K | ~17K | ~123 | ~325K | `process_assist09.py` |
| EdNet | ~80K | ~13K | ~188 | ~10M | `process_ednet.py` |
| Junyi | ~25K | ~40K | ~721 | ~25M | `process_junyi.py` |

## 🚀 使用方法

### 1. ASSIST09数据集

```bash
# 自动下载并处理
python data/process_assist09_v2.py

# 输出文件
data/assist09_processed.pkl
```

**特点**：
- 使用EduData库自动下载
- 使用Q-matrix方法构建概念图
- 数据量适中，适合快速测试

### 2. EdNet数据集

```bash
# 步骤1：下载数据（约3-5GB）
python data/download_ednet.py

# 步骤2：处理数据
python data/process_ednet.py

# 输出文件
data/ednet_processed.pkl
```

**特点**：
- 大规模数据集（10M+交互）
- 需要手动下载（文件较大）
- 使用Q-matrix方法构建概念图

### 3. Junyi数据集

```bash
# 自动下载并处理
python data/process_junyi_v2.py

# 输出文件
data/junyi_processed.pkl
```

**特点**：
- 使用EduData库自动下载
- 优先使用先修关系构建概念图
- 如果先修关系不存在，使用Q-matrix方法

## 📦 合并数据集

处理完所有数据集后，需要合并到统一的文件中：

```python
import pickle

# 加载各数据集
with open('data/assist09_processed.pkl', 'rb') as f:
    assist09_data = pickle.load(f)

with open('data/ednet_processed.pkl', 'rb') as f:
    ednet_data = pickle.load(f)

with open('data/junyi_processed.pkl', 'rb') as f:
    junyi_data = pickle.load(f)

# 合并
all_datasets = {}
all_datasets.update(assist09_data)
all_datasets.update(ednet_data)
all_datasets.update(junyi_data)

# 保存
with open('data/processed_datasets.pkl', 'wb') as f:
    pickle.dump(all_datasets, f)

print("✅ 数据集已合并到 processed_datasets.pkl")
```

或使用提供的脚本：

```bash
python data/merge_datasets.py
```

## 📊 数据格式

每个数据集的格式统一为：

```python
{
    'dataset_name': {
        'train': [
            {'student_id': 0, 'question_id': 1, 'concept_id': 2, 'correct': 1},
            ...
        ],
        'val': [...],
        'test': [...],
        'q_matrix': np.array,        # [n_questions, n_concepts]
        'concept_graph': np.array,   # [n_concepts, n_concepts]
        'n_students': int,
        'n_questions': int,
        'n_concepts': int
    }
}
```

## ⚙️ 数据预处理步骤

每个脚本都执行以下标准步骤：

1. **下载数据**（如果需要）
2. **数据清洗**
   - 移除缺失值
   - 过滤异常作答时间（1秒-1小时）
   - 按时间排序
3. **ID重编码**
   - student_id: 0, 1, 2, ...
   - question_id: 0, 1, 2, ...
   - concept_id: 0, 1, 2, ...
4. **构建Q矩阵**
   - 题目-知识点关联矩阵
5. **构建概念图**
   - ASSIST09/EdNet: Q-matrix方法（题目共现）
   - Junyi: 先修关系（如果可用）
6. **数据划分**
   - 训练集：70%
   - 验证集：15%
   - 测试集：15%
   - 按学生维度划分
7. **保存数据**

## 🔧 依赖库

```bash
pip install pandas numpy EduData
```

## ⚠️ 注意事项

1. **EdNet数据集**：
   - 文件较大（3-5GB），下载需要时间
   - 处理需要较大内存（建议16GB+）
   - 可使用`--max_students`参数处理部分数据进行测试

2. **Junyi数据集**：
   - 如果先修关系文件不存在，会自动使用Q-matrix方法
   - 数据量大，处理时间较长

3. **数据路径**：
   - 原始数据默认保存在`data/raw/`
   - 处理后的数据保存在`data/`
   - 确保有足够的磁盘空间（约30GB）

## 📝 论文对齐

根据论文要求：
- ✅ 使用Q-matrix方法构建概念图（公式3.3.1）
- ✅ 过滤异常作答时间（1秒-1小时）
- ✅ 按学生维度划分数据（70/15/15）
- ✅ ID连续编码（从0开始）

## 🐛 故障排除

### 问题1：EduData下载失败
**解决方案**：
- 检查网络连接
- 使用代理或VPN
- 手动下载数据并放置到`data/raw/`目录

### 问题2：内存不足
**解决方案**：
- 对于EdNet，使用`--max_students`参数
- 增加系统swap空间
- 使用更强大的机器

### 问题3：处理速度慢
**解决方案**：
- 使用SSD而非HDD
- 先用小数据集测试
- 考虑并行处理（需修改脚本）

---

**最后更新**：2026-02-06  
**维护者**：KRD-KT项目组

