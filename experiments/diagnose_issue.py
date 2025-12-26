"""
快速诊断脚本：检查数据加载和模型问题
用于诊断为什么DKT模型AUC接近0.5
"""

import torch
import numpy as np
import pickle
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.run_experiment import load_processed_data, create_data_loaders
from baselines.dkt import DKT, DKTLoss
from torch.utils.data import DataLoader


def diagnose_data_loading():
    """诊断数据加载问题"""
    print("="*60)
    print("1. 数据加载诊断")
    print("="*60)
    
    # 加载数据
    dataset_info = load_processed_data('assist09')
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_info, batch_size=32, max_seq_len=200
    )
    
    # 检查一个batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Question seq shape: {batch['question_seq'].shape}")
    print(f"Concept seq shape: {batch['concept_seq'].shape}")
    print(f"Answer seq shape: {batch['answer_seq'].shape}")
    print(f"Target concept shape: {batch['target_concept'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # 检查标签分布
    labels = batch['labels'].float()
    print(f"\n标签统计:")
    print(f"  均值: {labels.mean():.4f}")
    print(f"  标准差: {labels.std():.4f}")
    print(f"  最小值: {labels.min():.4f}")
    print(f"  最大值: {labels.max():.4f}")
    print(f"  正样本比例: {labels.mean():.4f} (应该接近0.69对于ASSIST09)")
    
    # 检查数据范围
    print(f"\n数据范围:")
    print(f"  Question IDs: [{batch['question_seq'].min()}, {batch['question_seq'].max()}]")
    print(f"  Concept IDs: [{batch['concept_seq'].min()}, {batch['concept_seq'].max()}]")
    print(f"  Answer values: {torch.unique(batch['answer_seq'])}")
    print(f"  Target concepts: [{batch['target_concept'].min()}, {batch['target_concept'].max()}]")
    
    # 检查整个数据集的标签分布
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    
    all_labels = np.array(all_labels)
    print(f"\n训练集整体标签分布:")
    print(f"  均值: {all_labels.mean():.4f}")
    print(f"  正样本比例: {all_labels.mean():.4f}")
    
    return dataset_info, train_loader, val_loader


def diagnose_model_forward(dataset_info, train_loader):
    """诊断模型前向传播"""
    print("\n" + "="*60)
    print("2. 模型前向传播诊断")
    print("="*60)
    
    # 创建模型
    model = DKT(
        n_questions=dataset_info['n_questions'],
        n_concepts=dataset_info['n_concepts'],
        embed_dim=128,
        hidden_dim=256,
        dropout=0.2
    )
    
    model.eval()
    
    # 测试一个batch
    batch = next(iter(train_loader))
    question_seq = batch['question_seq']
    answer_seq = batch['answer_seq']
    target_concept = batch['target_concept']
    labels = batch['labels']
    
    print(f"\n输入形状:")
    print(f"  Question seq: {question_seq.shape}")
    print(f"  Answer seq: {answer_seq.shape}")
    print(f"  Target concept: {target_concept.shape}")
    print(f"  Labels: {labels.shape}")
    
    # 前向传播
    with torch.no_grad():
        predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
    
    print(f"\n模型输出:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Predictions mean: {predictions.mean():.4f}")
    print(f"  Predictions std: {predictions.std():.4f}")
    print(f"  前5个预测值: {predictions[:5]}")
    print(f"  前5个真实标签: {labels[:5]}")
    
    # 检查预测是否合理
    if predictions.mean() < 0.3 or predictions.mean() > 0.7:
        print(f"\n[WARNING] 预测均值异常 ({predictions.mean():.4f})")
    if predictions.std() < 0.01:
        print(f"\n[WARNING] 预测方差过小 ({predictions.std():.4f})，模型可能没有学习")
    
    # 计算初始AUC（随机预测应该接近0.5）
    from sklearn.metrics import roc_auc_score
    try:
        initial_auc = roc_auc_score(labels.numpy(), predictions.numpy())
        print(f"\n初始AUC (未训练): {initial_auc:.4f}")
        if initial_auc < 0.45 or initial_auc > 0.55:
            print(f"[WARNING] 初始AUC异常，模型初始化可能有问题")
    except Exception as e:
        print(f"\n无法计算AUC: {e}")
    
    return model, predictions


def diagnose_training_step(model, train_loader):
    """诊断训练步骤"""
    print("\n" + "="*60)
    print("3. 训练步骤诊断")
    print("="*60)
    
    from baselines.dkt import DKTLoss
    import torch.optim as optim
    
    loss_fn = DKTLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    # 一个训练步骤
    batch = next(iter(train_loader))
    question_seq = batch['question_seq']
    answer_seq = batch['answer_seq']
    target_concept = batch['target_concept']
    labels = batch['labels']
    
    # 前向传播
    predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
    
    # 计算损失
    loss = loss_fn(predictions, labels)
    
    print(f"\n训练步骤:")
    print(f"  损失值: {loss.item():.4f}")
    print(f"  损失是否合理: {'是' if 0.1 < loss.item() < 1.0 else '否 (可能有问题)'}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            if param_count <= 3:  # 只打印前3个参数
                print(f"  {name} gradient norm: {param_norm.item():.6f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"\n  总梯度范数: {total_norm:.6f}")
    
    if total_norm < 1e-6:
        print(f"  [WARNING] 梯度过小，可能存在梯度消失")
    elif total_norm > 100:
        print(f"  [WARNING] 梯度过大，可能存在梯度爆炸")
    else:
        print(f"  [OK] 梯度正常")
    
    # 更新参数
    optimizer.step()
    
    # 再次前向传播，检查是否有变化
    with torch.no_grad():
        predictions_after = model.predict_single_concept(question_seq, answer_seq, target_concept)
    
    change = (predictions_after - predictions).abs().mean()
    print(f"\n  参数更新后预测变化: {change:.6f}")
    if change < 1e-6:
        print(f"  [WARNING] 参数更新后预测几乎没有变化")
    else:
        print(f"  [OK] 参数更新正常")


def diagnose_data_consistency(dataset_info):
    """诊断数据一致性"""
    print("\n" + "="*60)
    print("4. 数据一致性诊断")
    print("="*60)
    
    # 检查数据集统计
    train_data = dataset_info['train']
    val_data = dataset_info['val']
    test_data = dataset_info['test']
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 检查标签分布（如果是DataFrame）
    if hasattr(train_data, 'columns'):
        if 'correct' in train_data.columns:
            print(f"\n训练集标签分布:")
            print(f"  正样本比例: {train_data['correct'].mean():.4f}")
        if 'answer' in train_data.columns:
            print(f"  正样本比例: {train_data['answer'].mean():.4f}")


def main():
    """主诊断流程"""
    print("\n" + "="*80)
    print("KER-KT 实验问题诊断工具")
    print("="*80)
    
    try:
        # 1. 数据加载诊断
        dataset_info, train_loader, val_loader = diagnose_data_loading()
        
        # 2. 模型前向传播诊断
        model, predictions = diagnose_model_forward(dataset_info, train_loader)
        
        # 3. 训练步骤诊断
        diagnose_training_step(model, train_loader)
        
        # 4. 数据一致性诊断
        diagnose_data_consistency(dataset_info)
        
        print("\n" + "="*80)
        print("诊断完成！")
        print("="*80)
        print("\n建议:")
        print("1. 如果标签均值不在0.6-0.8范围内，检查数据预处理")
        print("2. 如果预测均值异常，检查模型初始化")
        print("3. 如果梯度异常，检查学习率和模型结构")
        print("4. 如果AUC始终接近0.5，检查损失函数和训练循环")
        
    except Exception as e:
        print(f"\n[ERROR] 诊断过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

