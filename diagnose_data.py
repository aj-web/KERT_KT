"""
诊断数据和模型问题
"""
import pickle
import torch
import numpy as np

# 加载数据
print("正在加载数据...")
with open('data/processed_datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

assist09 = datasets['assist09']

print('\n=== ASSIST09 数据统计 ===')
print(f'Questions: {assist09["n_questions"]}')
print(f'Concepts: {assist09["n_concepts"]}')
print(f'Students: {assist09["n_students"]}')
print(f'Train samples: {len(assist09["train"])}')
print(f'Val samples: {len(assist09["val"])}')
print(f'Test samples: {len(assist09["test"])}')

# 检查训练数据
train_df = assist09['train']
print(f'\n=== 训练数据检查 ===')
print(f'列名: {train_df.columns.tolist()}')
print(f'数据形状: {train_df.shape}')
print(f'\n前5行数据:')
print(train_df.head())

print(f'\n=== ID范围检查 ===')
print(f'Question ID: {train_df["question_id"].min()} 到 {train_df["question_id"].max()}')
print(f'Concept ID: {train_df["concept_id"].min()} 到 {train_df["concept_id"].max()}')
print(f'Student ID: {train_df["student_id"].min()} 到 {train_df["student_id"].max()}')

print(f'\n=== 标签分布 ===')
label_counts = train_df["correct"].value_counts()
print(f'标签0: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(train_df)*100:.2f}%)')
print(f'标签1: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(train_df)*100:.2f}%)')
print(f'标签均值: {train_df["correct"].mean():.4f}')

print(f'\n=== 潜在问题检查 ===')
print(f'⚠️ Question ID从0开始: {train_df["question_id"].min() == 0}')
print(f'⚠️ 有 {(train_df["question_id"] == 0).sum()} 个question_id=0的样本')
print(f'⚠️ Concept ID从0开始: {train_df["concept_id"].min() == 0}')

# 检查序列长度
print(f'\n=== 序列长度统计 ===')
seq_lengths = train_df.groupby('student_id').size()
print(f'平均序列长度: {seq_lengths.mean():.2f}')
print(f'最短序列: {seq_lengths.min()}')
print(f'最长序列: {seq_lengths.max()}')
print(f'中位数序列长度: {seq_lengths.median():.2f}')

# 检查概念图
print(f'\n=== 概念图检查 ===')
concept_graph = assist09['concept_graph']
print(f'概念图形状: {concept_graph.shape}')
print(f'概念图中非零元素: {np.count_nonzero(concept_graph)}')
print(f'概念图稀疏度: {np.count_nonzero(concept_graph) / (concept_graph.shape[0] * concept_graph.shape[1]):.4f}')

print("\n诊断完成！")
