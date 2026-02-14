"""
训练过程详细诊断：检查为什么训练后AUC仍然接近0.5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.run_experiment import load_processed_data, create_data_loaders
from baselines.dkt import DKT, DKTLoss


def diagnose_training_process():
    """诊断训练过程，观察AUC变化"""
    print("="*80)
    print("训练过程详细诊断")
    print("="*80)
    
    # 加载数据
    dataset_info = load_processed_data('assist09')
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_info, batch_size=32, max_seq_len=200
    )
    
    # 创建模型
    model = DKT(
        n_questions=dataset_info['n_questions'],
        n_concepts=dataset_info['n_concepts'],
        embed_dim=128,
        hidden_dim=256,
        dropout=0.0  # 先不用dropout，避免干扰
    )
    
    loss_fn = DKTLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"\n使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练几个epoch，观察AUC变化
    n_epochs = 10
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        train_predictions = []
        train_labels = []
        
        # 训练一个epoch（只取前几个batch用于快速诊断）
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # 只训练前10个batch
                break
                
            question_seq = batch['question_seq'].to(device)
            answer_seq = batch['answer_seq'].to(device)
            target_concept = batch['target_concept'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
            
            # 计算损失
            loss = loss_fn(predictions, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度
            if batch_idx == 0 and epoch == 0:
                total_grad_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                        if 'output_layer' in name:
                            print(f"  {name} gradient: {grad_norm.item():.6f}")
                total_grad_norm = total_grad_norm ** 0.5
                print(f"  总梯度范数: {total_grad_norm:.6f}")
            
            optimizer.step()
            
            epoch_loss += loss.item()
            train_predictions.extend(predictions.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # 验证
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 10:  # 只验证前10个batch
                    break
                    
                question_seq = batch['question_seq'].to(device)
                answer_seq = batch['answer_seq'].to(device)
                target_concept = batch['target_concept'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        train_auc = roc_auc_score(train_labels, train_predictions)
        val_auc = roc_auc_score(val_labels, val_predictions)
        train_acc = accuracy_score(train_labels, np.round(train_predictions))
        val_acc = accuracy_score(val_labels, np.round(val_predictions))
        
        print(f"\nEpoch {epoch+1}/{n_epochs}:")
        print(f"  Train Loss: {epoch_loss/10:.4f}")
        print(f"  Train AUC: {train_auc:.4f}, Train ACC: {train_acc:.4f}")
        print(f"  Val AUC: {val_auc:.4f}, Val ACC: {val_acc:.4f}")
        print(f"  预测范围: [{np.min(train_predictions):.4f}, {np.max(train_predictions):.4f}]")
        print(f"  预测均值: {np.mean(train_predictions):.4f}")
        
        # 如果AUC开始提升，说明训练有效
        if epoch > 0 and val_auc > 0.55:
            print(f"  [OK] AUC开始提升，训练有效")
            break
        elif epoch == n_epochs - 1 and val_auc < 0.55:
            print(f"  [WARNING] 训练后AUC仍然很低，可能存在以下问题：")
            print(f"    1. 输入编码方式可能不正确")
            print(f"    2. 模型容量可能不足")
            print(f"    3. 学习率可能不合适")
            print(f"    4. 数据预处理可能有问题")


def check_input_encoding():
    """检查输入编码方式"""
    print("\n" + "="*80)
    print("输入编码检查")
    print("="*80)
    
    dataset_info = load_processed_data('assist09')
    train_loader, _, _ = create_data_loaders(
        dataset_info, batch_size=4, max_seq_len=10
    )
    
    batch = next(iter(train_loader))
    
    print(f"\n输入数据示例:")
    print(f"  Question seq: {batch['question_seq'][0]}")
    print(f"  Answer seq: {batch['answer_seq'][0]}")
    print(f"  Target concept: {batch['target_concept'][0]}")
    print(f"  Labels: {batch['labels'][0]}")
    
    # 检查问题和答案的对应关系
    print(f"\n问题和答案对应关系:")
    for i in range(min(5, len(batch['question_seq'][0]))):
        q = batch['question_seq'][0][i].item()
        a = batch['answer_seq'][0][i].item()
        print(f"  位置{i}: Question={q}, Answer={a}")
    
    # 检查target_concept和question的关系
    print(f"\n目标概念和问题的关系:")
    print(f"  Target question: {batch['target_question'][0]}")
    print(f"  Target concept: {batch['target_concept'][0]}")
    print(f"  问题ID % 概念数 = {batch['target_question'][0] % dataset_info['n_concepts']}")


if __name__ == "__main__":
    check_input_encoding()
    diagnose_training_process()

