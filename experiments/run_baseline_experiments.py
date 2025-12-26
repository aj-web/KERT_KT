"""
基线模型对比实验脚本
运行所有基线模型（DKT, DKVMN, SAKT, AKT, GKT）并与KER-KT对比
论文4.3节：性能对比实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import sys
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from datetime import datetime
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.kert_kt import KERKT, KTSequenceDataset, train_kert_kt
from models.kt_predictor import DataCollator
from baselines.dkt import DKT, DKTLoss
from baselines.dkvmn import DKVMN, DKVMNLoss
from baselines.sakt import SAKT, SAKTLoss
from baselines.akt import AKT, AKTLoss
from baselines.gkt import GKT, GKTLoss
from experiments.run_experiment import get_dataset_config, load_processed_data, create_data_loaders


def train_baseline_model(model_name, dataset_name, dataset_info, concept_graph, 
                        config, n_epochs=50, device='cpu', n_runs=1):
    """
    训练基线模型（论文4.3.1节）
    
    Args:
        model_name: 模型名称 ('DKT', 'DKVMN', 'SAKT', 'AKT', 'GKT')
        dataset_name: 数据集名称
        dataset_info: 数据集信息
        concept_graph: 知识点图
        config: 数据集配置（论文表4.4）
        n_epochs: 训练轮数
        device: 设备
        n_runs: 运行次数（论文4.3.2节要求5次）
    
    Returns:
        results: 结果字典，包含所有运行的指标
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {dataset_name.upper()}")
    print(f"{'='*60}")
    
    all_aucs = []
    all_accs = []
    
    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_info, 
            batch_size=config['batch_size'], 
            max_seq_len=200
        )
        
        # 初始化模型
        if model_name == 'DKT':
            model = DKT(
                n_questions=dataset_info['n_questions'],
                n_concepts=dataset_info['n_concepts'],
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout']
            )
            loss_fn = DKTLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
        elif model_name == 'DKVMN':
            model = DKVMN(
                n_questions=dataset_info['n_questions'],
                n_concepts=dataset_info['n_concepts'],
                embed_dim=config['embed_dim'],
                dropout=config['dropout']
            )
            loss_fn = DKVMNLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
        elif model_name == 'SAKT':
            model = SAKT(
                n_questions=dataset_info['n_questions'],
                n_concepts=dataset_info['n_concepts'],
                embed_dim=config['embed_dim'],
                num_heads=8,
                num_layers=2,
                dropout=config['dropout']
            )
            loss_fn = SAKTLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
        elif model_name == 'AKT':
            # AKT需要concept_graph
            concept_graph_np = concept_graph.numpy() if isinstance(concept_graph, torch.Tensor) else concept_graph
            model = AKT(
                n_questions=dataset_info['n_questions'],
                n_concepts=dataset_info['n_concepts'],
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim'],
                num_heads=8,
                num_layers=2,
                dropout=config['dropout'],
                concept_graph=concept_graph_np
            )
            loss_fn = AKTLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
        elif model_name == 'GKT':
            # GKT需要concept_graph
            concept_graph_np = concept_graph.numpy() if isinstance(concept_graph, torch.Tensor) else concept_graph
            model = GKT(
                n_questions=dataset_info['n_questions'],
                n_concepts=dataset_info['n_concepts'],
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim'],
                num_gcn_layers=config['n_layers'],
                dropout=config['dropout'],
                concept_graph=concept_graph_np
            )
            loss_fn = GKTLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(device)
        
        # 训练循环
        best_auc = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                # 移动到设备
                question_seq = batch['question_seq'].to(device)
                concept_seq = batch['concept_seq'].to(device)
                answer_seq = batch['answer_seq'].to(device)
                target_concept = batch['target_concept'].to(device)
                labels = batch['labels'].to(device)
                
                # 前向传播
                attention_mask = batch.get('attention_mask', None)
                if model_name in ['DKT', 'DKVMN', 'SAKT']:
                    if model_name == 'DKT' and attention_mask is not None:
                        predictions = model.predict_single_concept(question_seq, answer_seq, target_concept, attention_mask)
                    else:
                        predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
                elif model_name in ['AKT', 'GKT']:
                    predictions = model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # 计算损失
                loss = loss_fn(predictions, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 验证
            model.eval()
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    question_seq = batch['question_seq'].to(device)
                    concept_seq = batch['concept_seq'].to(device)
                    answer_seq = batch['answer_seq'].to(device)
                    target_concept = batch['target_concept'].to(device)
                    labels = batch['labels'].to(device)
                    
                    attention_mask = batch.get('attention_mask', None)
                    if model_name in ['DKT', 'DKVMN', 'SAKT']:
                        if model_name == 'DKT' and attention_mask is not None:
                            predictions = model.predict_single_concept(question_seq, answer_seq, target_concept, attention_mask)
                        else:
                            predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
                    elif model_name in ['AKT', 'GKT']:
                        predictions = model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_auc = roc_auc_score(val_labels, val_predictions)
            val_acc = accuracy_score(val_labels, np.round(val_predictions))
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, "
                      f"Val AUC={val_auc:.4f}, Val ACC={val_acc:.4f}")
            
            # 早停
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 测试集评估
        model.eval()
        test_predictions = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                question_seq = batch['question_seq'].to(device)
                concept_seq = batch['concept_seq'].to(device)
                answer_seq = batch['answer_seq'].to(device)
                target_concept = batch['target_concept'].to(device)
                labels = batch['labels'].to(device)
                
                if model_name in ['DKT', 'DKVMN', 'SAKT']:
                    predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
                elif model_name in ['AKT', 'GKT']:
                    predictions = model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        test_auc = roc_auc_score(test_labels, test_predictions)
        test_acc = accuracy_score(test_labels, np.round(test_predictions))
        
        print(f"Run {run_idx + 1} Test Results: AUC={test_auc:.4f}, ACC={test_acc:.4f}")
        
        all_aucs.append(test_auc)
        all_accs.append(test_acc)
    
    # 计算均值±标准差（论文4.3.2节）
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    
    print(f"\n{model_name} Final Results ({n_runs} runs):")
    print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  ACC: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return {
        'model_name': model_name,
        'dataset': dataset_name,
        'auc_mean': mean_auc,
        'auc_std': std_auc,
        'acc_mean': mean_acc,
        'acc_std': std_acc,
        'all_aucs': all_aucs,
        'all_accs': all_accs,
        'n_runs': n_runs
    }


def run_all_baseline_experiments(datasets=['assist09', 'assist17', 'junyi'], 
                                 models=['DKT', 'DKVMN', 'SAKT', 'AKT', 'GKT'],
                                 n_runs=5, n_epochs=50, device='cpu'):
    """
    运行所有基线模型实验（论文4.3节）
    
    Args:
        datasets: 数据集列表
        models: 模型列表
        n_runs: 每个模型运行次数（论文要求5次）
        n_epochs: 训练轮数
        device: 设备
    
    Returns:
        all_results: 所有实验结果
    """
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # 加载数据集
        dataset_info = load_processed_data(dataset_name)
        concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32)
        
        # 获取数据集配置（论文表4.4）
        config = get_dataset_config(dataset_name)
        
        # 训练所有基线模型
        for model_name in models:
            try:
                result = train_baseline_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    dataset_info=dataset_info,
                    concept_graph=concept_graph,
                    config=config,
                    n_epochs=n_epochs,
                    device=device,
                    n_runs=n_runs
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error training {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return all_results


def print_results_table(all_results):
    """
    打印结果表格（论文表4.5格式）
    """
    print("\n" + "="*80)
    print("实验结果汇总（论文表4.5格式）")
    print("="*80)
    
    # 按数据集分组
    datasets = ['assist09', 'assist17', 'junyi']
    models = ['DKT', 'DKVMN', 'SAKT', 'AKT', 'GKT', 'KER-KT']
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print(f"{'Model':<15} {'AUC':<20} {'ACC':<20}")
        print("-" * 55)
        
        for model in models:
            # 查找该模型在该数据集上的结果
            result = None
            for r in all_results:
                if r['model_name'] == model and r['dataset'] == dataset:
                    result = r
                    break
            
            if result:
                auc_str = f"{result['auc_mean']:.4f} ± {result['auc_std']:.4f}"
                acc_str = f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f}"
                print(f"{model:<15} {auc_str:<20} {acc_str:<20}")
            else:
                print(f"{model:<15} {'N/A':<20} {'N/A':<20}")


def main():
    parser = argparse.ArgumentParser(description='运行基线模型对比实验（论文4.3节）')
    parser.add_argument('--datasets', nargs='+', default=['assist09', 'assist17', 'junyi'],
                       help='数据集列表')
    parser.add_argument('--models', nargs='+', 
                       default=['DKT', 'DKVMN', 'SAKT', 'AKT', 'GKT'],
                       help='基线模型列表')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='每个模型运行次数（论文4.3.2节要求5次）')
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cpu/cuda)')
    parser.add_argument('--save_results', type=str, default='baseline_results.json',
                       help='结果保存路径')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 运行实验
    all_results = run_all_baseline_experiments(
        datasets=args.datasets,
        models=args.models,
        n_runs=args.n_runs,
        n_epochs=args.n_epochs,
        device=device
    )
    
    # 打印结果表格
    print_results_table(all_results)
    
    # 保存结果
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, args.save_results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")
    print("\n实验完成！")


if __name__ == "__main__":
    main()

