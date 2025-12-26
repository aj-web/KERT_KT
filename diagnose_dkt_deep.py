"""
深度诊断DKT模型问题
检查模型输出、数据分布、梯度等
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from baselines.dkt import DKT
from models.kert_kt import KTSequenceDataset
from models.kt_predictor import DataCollator

# 加载数据
print("加载数据...")
with open('data/processed_datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

assist09 = datasets['assist09']
print(f"\n数据集信息:")
print(f"  Questions: {assist09['n_questions']}")
print(f"  Concepts: {assist09['n_concepts']}")
print(f"  Train samples: {len(assist09['train'])}")

# 创建数据加载器
print("\n创建数据加载器...")
train_dataset = KTSequenceDataset(assist09['train'], max_seq_len=200)
collator = DataCollator(max_seq_len=200)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collator.collate_fn)

# 初始化模型
print("\n初始化DKT模型...")
model = DKT(
    n_questions=assist09['n_questions'],
    n_concepts=assist09['n_concepts'],
    embed_dim=128,
    hidden_dim=256,
    dropout=0.2
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"使用设备: {device}")

# 获取一个batch进行诊断
print("\n获取测试batch...")
batch = next(iter(train_loader))

question_seq = batch['question_seq'].to(device)
answer_seq = batch['answer_seq'].to(device)
target_concept = batch['target_concept'].to(device)
labels = batch['labels'].to(device)
attention_mask = batch['attention_mask'].to(device)

print(f"\nBatch信息:")
print(f"  Batch size: {question_seq.size(0)}")
print(f"  Sequence length: {question_seq.size(1)}")
print(f"  标签分布: {labels.float().mean().item():.4f}")

# 检查数据
print(f"\n数据检查:")
print(f"  Question ID范围: {question_seq.min().item()} - {question_seq.max().item()}")
print(f"  Answer范围: {answer_seq.min().item()} - {answer_seq.max().item()}")
print(f"  Target concept范围: {target_concept.min().item()} - {target_concept.max().item()}")
print(f"  Attention mask sum: {attention_mask.sum(dim=1).float().mean().item():.2f}")

# 前向传播（未训练）
print(f"\n=== 未训练模型的输出 ===")
model.eval()
with torch.no_grad():
    predictions = model.predict_single_concept(question_seq, answer_seq, target_concept, attention_mask)
    print(f"  预测值范围: {predictions.min().item():.4f} - {predictions.max().item():.4f}")
    print(f"  预测值均值: {predictions.mean().item():.4f}")
    print(f"  预测值标准差: {predictions.std().item():.4f}")
    print(f"  前5个预测: {predictions[:5].cpu().numpy()}")

# 训练几个step看看
print(f"\n=== 训练5个batch ===")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for i, batch in enumerate(train_loader):
    if i >= 5:
        break

    question_seq = batch['question_seq'].to(device)
    answer_seq = batch['answer_seq'].to(device)
    target_concept = batch['target_concept'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    predictions = model.predict_single_concept(question_seq, answer_seq, target_concept, attention_mask)
    loss = loss_fn(predictions, labels)

    optimizer.zero_grad()
    loss.backward()

    # 检查梯度
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    optimizer.step()

    print(f"  Batch {i+1}: Loss={loss.item():.4f}, Grad Norm={total_norm:.4f}, Pred mean={predictions.mean().item():.4f}")

# 训练后的输出
print(f"\n=== 训练5个batch后的输出 ===")
model.eval()
batch = next(iter(train_loader))
question_seq = batch['question_seq'].to(device)
answer_seq = batch['answer_seq'].to(device)
target_concept = batch['target_concept'].to(device)
attention_mask = batch['attention_mask'].to(device)

with torch.no_grad():
    predictions = model.predict_single_concept(question_seq, answer_seq, target_concept, attention_mask)
    print(f"  预测值范围: {predictions.min().item():.4f} - {predictions.max().item():.4f}")
    print(f"  预测值均值: {predictions.mean().item():.4f}")
    print(f"  预测值标准差: {predictions.std().item():.4f}")
    print(f"  前5个预测: {predictions[:5].cpu().numpy()}")

# 检查模型权重
print(f"\n=== 模型权重检查 ===")
for name, param in model.named_parameters():
    print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

print("\n诊断完成！")
