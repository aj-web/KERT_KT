# ASSIST09真实数据集成指南

## 背景

由于EduData CLI工具存在依赖问题（`longling.ML`模块缺失），我们提供了三种方案来获取和使用ASSIST09真实数据集。

## 方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方案A: 自动下载脚本** | 一键运行，自动化 | 下载链接可能失效 | ⭐⭐⭐⭐ |
| **方案B: 手动下载** | 最可靠 | 需要手动操作 | ⭐⭐⭐⭐⭐ |
| **方案C: 使用合成数据** | 无需下载，立即可用 | 不是真实数据 | ⭐⭐⭐ |

---

## 方案A: 使用自动下载脚本（推荐优先尝试）

### 步骤1: 运行下载脚本

```bash
python download_assist09_direct.py
```

这个脚本会：
1. 尝试从多个公开源下载ASSIST09数据
2. 自动转换为项目需要的格式
3. 保存到 `data/assist09.csv`

### 步骤2: 重新生成处理后的数据

```bash
python regenerate_data.py
```

这会生成 `data/processed_datasets.pkl` 文件，供实验使用。

### 步骤3: 运行实验验证

```bash
# 快速测试（1次运行）
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1 --n_epochs 10

# 完整实验（5次运行，论文标准）
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT DKVMN SAKT --n_runs 5 --n_epochs 50
```

---

## 方案B: 手动下载（最可靠）

### 步骤1: 手动下载ASSIST09数据

访问以下任一网站下载数据：

**选项1: 官方数据集网站**
- URL: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
- 下载文件: `skill_builder_data.csv`

**选项2: Kaggle**
- URL: https://www.kaggle.com/datasets/
- 搜索: "ASSISTments 2009-2010"

**选项3: GitHub备份**
- 一些研究者会在GitHub上分享数据集备份

### 步骤2: 放置数据文件

将下载的CSV文件重命名为 `assist09_raw.csv`，并放到项目的 `data/` 目录下：

```bash
# 假设你下载的文件在 ~/Downloads/skill_builder_data.csv
mv ~/Downloads/skill_builder_data.csv ./data/assist09_raw.csv
```

### 步骤3: 运行转换脚本

```bash
python download_assist09_direct.py
```

虽然脚本名叫"download"，但它会检测到 `assist09_raw.csv` 已存在，直接进行格式转换。

### 步骤4: 生成处理后的数据

```bash
python regenerate_data.py
```

### 步骤5: 运行实验

```bash
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1
```

---

## 方案C: 使用修复后的合成数据（备选）

如果暂时无法获取真实数据，可以使用我们修复后的合成数据（基于IRT模型）。

### 使用方法

```bash
# 直接重新生成合成数据
python regenerate_data.py

# 运行实验
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1
```

**注意**: 合成数据已经修复了随机噪声问题，现在使用IRT模型生成，能够训练出有意义的结果（AUC > 0.50）。

---

## 数据格式说明

### 我们项目需要的格式（assist09.csv）

| 列名 | 说明 | 示例 |
|------|------|------|
| student_id | 学生ID（从0开始） | 0, 1, 2, ... |
| question_id | 题目ID（从0开始） | 0, 1, 2, ... |
| concept_id | 知识点ID（从0开始） | 0, 1, 2, ... |
| correct | 答题正确性（0或1） | 0, 1 |
| timestamp | 答题顺序/时间戳 | 0, 1, 2, ... |

### ASSIST09原始格式（assist09_raw.csv）

原始数据可能包含以下列：
- `user_id` → 转换为 `student_id`
- `problem_id` → 转换为 `question_id`
- `skill_id` → 转换为 `concept_id`
- `correct` → 保持不变
- `order_id` → 转换为 `timestamp`

---

## 常见问题

### Q1: 下载脚本报错 "Connection timeout"

**解决方案**:
- 使用方案B手动下载
- 检查网络连接
- 如果在国内，可能需要使用代理

### Q2: 转换脚本报错 "缺少必要的列"

**解决方案**:
1. 检查下载的文件是否正确
2. 打开CSV文件查看列名
3. 修改 `download_assist09_direct.py` 中的 `required_columns` 字典，匹配实际列名

### Q3: 生成的数据集太小或太大

ASSIST09数据集规模（预期）:
- 学生数: ~4,000
- 题目数: ~16,000
- 知识点数: ~120
- 答题记录: ~300,000+

如果数据规模差异很大，可能是：
- 下载的数据集版本不对
- 数据清洗过程删除了太多记录

### Q4: 实验运行很慢

**解决方案**:
- 使用GPU: `--device cuda`
- 减少运行次数: `--n_runs 1`
- 减少训练轮数: `--n_epochs 20`

---

## 预期实验结果

使用真实ASSIST09数据，预期的基线模型性能（论文参考值）：

| 模型 | AUC | ACC |
|------|-----|-----|
| DKT | 0.72-0.75 | 0.70-0.73 |
| DKVMN | 0.73-0.76 | 0.71-0.74 |
| SAKT | 0.74-0.77 | 0.72-0.75 |
| AKT | 0.75-0.78 | 0.73-0.76 |
| GKT | 0.74-0.77 | 0.72-0.75 |

**注意**:
- 如果AUC ≈ 0.50，说明数据有问题
- 如果AUC > 0.65，说明数据和模型正常工作
- 具体数值会因数据集版本和超参数设置而变化

---

## 下一步操作（推荐流程）

### 立即开始（最快路径）:

```bash
# 1. 尝试自动下载
python download_assist09_direct.py

# 2. 如果第1步失败，手动下载数据到 data/assist09_raw.csv，然后再次运行第1步

# 3. 重新生成处理后的数据
python regenerate_data.py

# 4. 快速验证（1个模型，1次运行）
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1 --n_epochs 10

# 5. 如果第4步结果正常（AUC > 0.65），运行完整实验
python experiments/run_baseline_experiments.py --datasets assist09 --models DKT DKVMN SAKT AKT GKT --n_runs 5 --n_epochs 50
```

---

## 联系与支持

如果遇到问题：
1. 检查 `data/` 目录下的文件
2. 查看脚本输出的错误信息
3. 确认数据格式是否正确

