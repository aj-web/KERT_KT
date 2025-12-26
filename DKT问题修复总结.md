# DKT模型问题修复总结

## 一、问题诊断结果

### 发现的关键问题

#### 🔴 **问题1：Padding冲突（已修复）**

**问题**：
- `question_id`从0开始（范围：0-17750）
- Padding也使用0值
- 导致模型无法区分真实的`question_id=0`和padding位置

**影响**：
- 模型学习错误的模式
- AUC始终接近0.5（随机猜测）

**修复**：
- ✅ 添加`attention_mask`标记padding位置
- ✅ 使用`pack_padded_sequence`处理变长序列
- ✅ DKT模型现在可以正确处理padding

#### 🔴 **问题2：输入编码不正确（已修复）**

**问题**：
- 原始实现只是简单拼接question和answer embedding

**修复**：
- ✅ 改为使用交互编码：`[q*a, q*(1-a)]`（按原始DKT论文）

## 二、已实施的修复

### 修复1：DataCollator添加attention_mask

**文件**：`models/kt_predictor.py`

**修改**：
- 添加`attention_mask`标记真实数据位置
- `True`表示真实数据，`False`表示padding

### 修复2：DKT使用pack_padded_sequence

**文件**：`baselines/dkt.py`

**修改**：
- `forward`方法接收`attention_mask`参数
- 使用`pack_padded_sequence`处理变长序列
- 使用最后一个有效位置的hidden state

### 修复3：实验脚本支持mask

**文件**：`experiments/run_baseline_experiments.py`

**修改**：
- 训练和验证时传递`attention_mask`给DKT模型

## 三、预期效果

修复后，DKT应该能够：

1. ✅ 正确区分padding和真实数据
2. ✅ 使用正确的输入编码方式
3. ✅ 训练后AUC达到0.70+（而不是0.50）

## 四、验证修复效果

### 快速测试命令

```bash
# 只测试DKT，1次运行，50轮训练（快速验证）
python experiments/run_baseline_experiments.py \
    --datasets assist09 \
    --models DKT \
    --n_runs 1 \
    --n_epochs 50
```

### 预期结果

修复后应该观察到：
- ✅ 训练过程中AUC逐步提升（而不是始终0.50）
- ✅ 50轮训练后AUC达到0.70+
- ✅ 损失和AUC同步提升

## 五、如果修复后仍然有问题

如果修复padding和输入编码后，AUC仍然很低，需要检查：

1. **数据预处理**：
   - 验证数据是否正确加载
   - 验证标签是否正确

2. **模型实现**：
   - 验证LSTM输出是否正确
   - 验证输出层是否正确

3. **训练过程**：
   - 检查学习率是否合适
   - 检查是否有梯度消失/爆炸

## 六、其他基线模型

### DKVMN
- 当前正在运行，训练很慢
- 可能也需要类似的padding修复

### SAKT, AKT, GKT
- 这些模型可能也需要处理padding
- 建议统一修复所有模型的padding问题

## 七、下一步行动

### 立即行动

1. **验证DKT修复效果**
   ```bash
   python experiments/run_baseline_experiments.py \
       --datasets assist09 \
       --models DKT \
       --n_runs 1 \
       --n_epochs 50
   ```

2. **如果修复有效**：
   - 重新运行完整的基线实验
   - 修复其他模型的padding问题（如果需要）

3. **如果修复无效**：
   - 深入检查数据预处理
   - 检查模型架构
   - 添加更多调试信息

