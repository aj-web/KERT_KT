"""
诊断KER-KT模型预测范围问题
检查模型初始化、梯度更新、训练过程
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.kert_kt import KERKT
from experiments.run_experiment import get_dataset_config, load_processed_data, create_data_loaders


def diagnose_model():
    """诊断模型问题"""
    print("="*60)
    print("KER-KT模型诊断")
    print("="*60)
    
    # 加载数据
    dataset_name = 'assist09'
    config = get_dataset_config(dataset_name)
    dataset_info = load_processed_data(dataset_name)
    concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    concept_graph = concept_graph.to(device)
    
    # 创建模型
    model = KERKT(
        n_questions=dataset_info['n_questions'],
        n_concepts=dataset_info['n_concepts'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        alpha=config['alpha'],
        beta=config['beta'],
        lambda_decay=config['lambda_decay'],
        gamma=config['gamma'],
        lr_kt=config['lr_kt_pretrain'],
        lr_rl=config['lr_rl'],
        lambda_rl=config['lambda_rl']
    )
    model = model.to(device)
    
    # 创建数据加载器
    train_loader, _, _ = create_data_loaders(
        dataset_info, batch_size=32, max_seq_len=config['max_seq_len']
    )
    
    # 获取一个batch
    batch = next(iter(train_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    print("\n1. 检查模型初始化")
    print("-" * 60)
    
    # 检查predictor层的初始化
    predictor = model.kt_predictor.predictor
    print("Predictor层结构:")
    for i, layer in enumerate(predictor):
        print(f"  Layer {i}: {layer}")
        if isinstance(layer, nn.Linear):
            weight_mean = layer.weight.data.mean().item()
            weight_std = layer.weight.data.std().item()
            print(f"    Weight: mean={weight_mean:.6f}, std={weight_std:.6f}")
    
    # 检查初始预测分布
    print("\n2. 检查初始预测分布")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        predictions, _ = model.forward(batch, concept_graph)
        pred_np = predictions.cpu().numpy()
        print(f"预测值统计:")
        print(f"  最小值: {pred_np.min():.4f}")
        print(f"  最大值: {pred_np.max():.4f}")
        print(f"  均值: {pred_np.mean():.4f}")
        print(f"  标准差: {pred_np.std():.4f}")
        print(f"  中位数: {np.median(pred_np):.4f}")
        
        # 检查预测值分布
        print(f"\n预测值分布:")
        bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        hist, _ = np.histogram(pred_np, bins=bins)
        for i in range(len(hist)):
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]} ({hist[i]/len(pred_np)*100:.1f}%)")
    
    # 检查梯度
    print("\n3. 检查梯度更新")
    print("-" * 60)
    model.train()
    
    # 记录初始参数
    initial_params = {}
    for name, param in model.kt_predictor.predictor.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
    
    # 执行一步训练
    losses = model.train_step(batch, concept_graph)
    print(f"训练损失:")
    for k, v in losses.items():
        print(f"  {k}: {v:.6f}")
    
    # 检查参数是否更新
    print(f"\n参数更新情况:")
    updated = False
    for name, param in model.kt_predictor.predictor.named_parameters():
        if param.requires_grad:
            initial = initial_params[name]
            diff = (param.data - initial).abs().max().item()
            if diff > 1e-6:
                updated = True
                print(f"  {name}: 已更新 (最大变化: {diff:.6f})")
            else:
                print(f"  {name}: 未更新 (最大变化: {diff:.6f})")
    
    if not updated:
        print("\n[WARNING] 参数未更新！可能存在梯度问题")
    
    # 检查梯度
    print(f"\n梯度统计:")
    has_grad = False
    for name, param in model.kt_predictor.predictor.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item()
            grad_max = param.grad.data.abs().max().item()
            print(f"  {name}:")
            print(f"    mean={grad_mean:.6f}, std={grad_std:.6f}, max_abs={grad_max:.6f}")
        else:
            print(f"  {name}: 无梯度")
    
    if not has_grad:
        print("\n[WARNING] 没有梯度！模型无法学习")
    
    # 再次检查预测分布
    print("\n4. 训练一步后的预测分布")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        predictions, _ = model.forward(batch, concept_graph)
        pred_np = predictions.cpu().numpy()
        print(f"预测值统计:")
        print(f"  最小值: {pred_np.min():.4f}")
        print(f"  最大值: {pred_np.max():.4f}")
        print(f"  均值: {pred_np.mean():.4f}")
        print(f"  标准差: {pred_np.std():.4f}")
    
    # 检查标签分布
    print("\n5. 检查标签分布")
    print("-" * 60)
    labels = batch['labels'].cpu().numpy()
    print(f"标签统计:")
    print(f"  0的数量: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"  1的数量: {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    print(f"  均值: {labels.mean():.4f}")
    
    # 检查预测和标签的相关性
    print("\n6. 预测-标签相关性")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        predictions, _ = model.forward(batch, concept_graph)
        pred_np = predictions.cpu().numpy()
        labels_np = labels
        
        # 计算AUC（如果可能）
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_np, pred_np)
            print(f"  AUC: {auc:.4f}")
        except:
            print(f"  无法计算AUC（可能所有标签相同）")
        
        # 计算准确率
        pred_binary = (pred_np > 0.5).astype(float)
        acc = (pred_binary == labels_np).mean()
        print(f"  准确率: {acc:.4f}")
        
        # 检查预测值范围是否合理
        if pred_np.min() > 0.4 and pred_np.max() < 0.6:
            print(f"\n[WARNING] 预测值范围过窄 ({pred_np.min():.4f}-{pred_np.max():.4f})")
            print(f"  可能原因:")
            print(f"  1. 模型初始化导致输出接近0.5")
            print(f"  2. 梯度太小，模型未更新")
            print(f"  3. 学习率太小")
            print(f"  4. 需要更多训练")
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == '__main__':
    diagnose_model()

