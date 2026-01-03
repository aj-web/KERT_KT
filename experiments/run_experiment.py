"""
Complete experiment pipeline for KER-KT model
Runs training and evaluation on all datasets
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle
import os
import sys
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.kert_kt import KERKT, KTSequenceDataset, train_kert_kt
from models.kt_predictor import DataCollator


def load_processed_data(dataset_name):
    """Load processed dataset"""
    data_path = os.path.join(project_root, 'data', 'processed_datasets.pkl')
    with open(data_path, 'rb') as f:
        datasets = pickle.load(f)

    return datasets[dataset_name]


def create_data_loaders(dataset_info, batch_size=32, max_seq_len=200):
    """Create data loaders for training and evaluation"""
    # Create datasets
    train_dataset = KTSequenceDataset(dataset_info['train'], max_seq_len)
    val_dataset = KTSequenceDataset(dataset_info['val'], max_seq_len)
    test_dataset = KTSequenceDataset(dataset_info['test'], max_seq_len)

    # Create data collator
    collator = DataCollator(max_seq_len)

    # Create data loaders
    # 性能优化：使用多进程加载数据（num_workers > 0）
    # Windows系统可能需要num_workers=0，Linux/Mac可以使用num_workers=4
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collator.collate_fn, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator.collate_fn, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator.collate_fn, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader


def run_single_experiment(dataset_name, config=None, n_runs=5):
    """
    Run experiment on single dataset (论文4.3.2节：每个模型运行5次)

    Args:
        dataset_name: name of dataset ('assist09', 'assist17', 'junyi')
        config: experiment configuration (if None, use dataset-specific config from 论文表4.4)
        n_runs: number of runs (论文要求5次)

    Returns:
        results: experiment results with mean ± std
    """
    # Use dataset-specific config if not provided (论文表4.4)
    if config is None:
        config = get_dataset_config(dataset_name)
    
    print(f"\n{'='*50}")
    print(f"Running experiment on {dataset_name.upper()}")
    print(f"Configuration (论文表4.4):")
    print(f"  embed_dim: {config['embed_dim']}, hidden_dim: {config['hidden_dim']}")
    print(f"  n_layers: {config['n_layers']}, alpha: {config['alpha']}, beta: {config['beta']}")
    print(f"  batch_size: {config['batch_size']}, dropout: {config['dropout']}")
    print(f"{'='*50}")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    # Load dataset
    dataset_info = load_processed_data(dataset_name)
    concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32).to(device)

    print(f"Dataset statistics:")
    print(f"  Questions: {dataset_info['n_questions']}")
    print(f"  Concepts: {dataset_info['n_concepts']}")
    print(f"  Students: {dataset_info['n_students']}")
    print(f"  Train samples: {len(dataset_info['train'])}")
    print(f"  Val samples: {len(dataset_info['val'])}")
    print(f"  Test samples: {len(dataset_info['test'])}")

    # Run multiple times (论文4.3.2节：每个模型运行5次)
    all_aucs = []
    all_accs = []
    
    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_info, config['batch_size'], config['max_seq_len']
        )

        # Initialize model
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
            lr_kt=config['lr_kt_pretrain'],  # 预训练阶段学习率
            lr_rl=config['lr_rl'],
            lambda_rl=config['lambda_rl'],
            l2_lambda=config.get('l2_lambda', 1e-5),  # L2正则化系数
            dropout=config.get('dropout', 0.2)  # Dropout率
        )
        
        # Move model to device
        model = model.to(device)
        print(f"Model moved to {device}")

        # Create checkpoint directory
        checkpoint_dir = os.path.join(project_root, 'checkpoints', dataset_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Train model
        print("\nStarting training...")
        best_model_path = os.path.join(checkpoint_dir, f'kert_kt_best_run{run_idx+1}.pt')
        
        # Update model learning rate for fine-tuning stage
        # Note: This should be handled in train_kert_kt function
        train_kert_kt(
            model, train_loader, val_loader, concept_graph,
            n_epochs=config['n_epochs'], patience=config['patience'],
            checkpoint_path=best_model_path,
            lr_kt_pretrain=config['lr_kt_pretrain'],
            lr_kt_finetune=config['lr_kt_finetune'],
            warmup_steps=config.get('warmup_steps', 0),
            lr_decay_patience=config.get('lr_decay_patience', None),
            lr_decay_factor=config.get('lr_decay_factor', 0.5)
        )

        # Load best model for testing
        model.load_model(best_model_path)

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_metrics = model.evaluate(test_loader, concept_graph)

        print(f"Run {run_idx + 1} Test Results:")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  ACC: {test_metrics['acc']:.4f}")
        
        all_aucs.append(test_metrics['auc'])
        all_accs.append(test_metrics['acc'])

    # Compute mean ± std (论文4.3.2节：报告均值±标准差)
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    
    print(f"\n{'='*50}")
    print(f"Final Results ({n_runs} runs):")
    print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  ACC: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*50}")

    # Save results
    results = {
        'dataset': dataset_name,
        'config': config,
        'test_metrics': {
            'auc_mean': mean_auc,
            'auc_std': std_auc,
            'acc_mean': mean_acc,
            'acc_std': std_acc,
            'all_aucs': all_aucs,
            'all_accs': all_accs
        },
        'dataset_stats': {
            'n_questions': dataset_info['n_questions'],
            'n_concepts': dataset_info['n_concepts'],
            'n_students': dataset_info['n_students'],
            'train_samples': len(dataset_info['train']),
            'val_samples': len(dataset_info['val']),
            'test_samples': len(dataset_info['test'])
        },
        'n_runs': n_runs,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return results


def run_all_experiments(n_runs=5):
    """
    Run experiments on all datasets (论文4.3.2节：每个模型运行5次)
    
    Args:
        n_runs: number of runs per dataset (论文要求5次)
    """
    datasets = ['assist09', 'assist17', 'junyi']
    all_results = []

    for dataset_name in datasets:
        try:
            # Use dataset-specific config (论文表4.4)
            results = run_single_experiment(dataset_name, config=None, n_runs=n_runs)
            all_results.append(results)
        except Exception as e:
            print(f"Error running experiment on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'experiment_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll experiments completed. Results saved to {results_file}")

    # Print summary (论文表4.5格式)
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY (论文表4.5格式)")
    print("="*60)

    for result in all_results:
        print(f"\n{result['dataset'].upper()}:")
        metrics = result['test_metrics']
        print(f"  Test AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
        print(f"  Test ACC: {metrics['acc_mean']:.4f} ± {metrics['acc_std']:.4f}")
        print(f"  Questions: {result['dataset_stats']['n_questions']}")
        print(f"  Concepts: {result['dataset_stats']['n_concepts']}")
        print(f"  Students: {result['dataset_stats']['n_students']}")
        print(f"  Runs: {result['n_runs']}")

    return all_results


def get_dataset_config(dataset_name):
    """
    Get configuration for specific dataset (论文表4.4)
    
    Args:
        dataset_name: 'assist09', 'assist17', or 'junyi'
    
    Returns:
        config: dataset-specific configuration
    """
    configs = {
        'assist09': {
            # Model hyperparameters (论文表4.4)
            'embed_dim': 128,      # d_k, d_q
            'hidden_dim': 256,     # d_h
            'n_layers': 2,        # L
            
            # Triple decision parameters
            'alpha': 0.7,
            'beta': 0.3,
            'lambda_decay': 0.1,
            
            # RL parameters
            'gamma': 0.99,
            'lambda1': 0.3,        # 奖励函数平衡性权重
            'lambda2': 0.2,        # 奖励函数稳定性权重
            'lr_rl': 1e-4,         # α_a, α_c
            'lambda_rl': 0.1,      # λ_RL
            
            # Training parameters
            'lr_kt_pretrain': 0.001,   # 优化：降低预训练学习率，减缓过拟合
            'lr_kt_finetune': 0.0005,  # 优化：相应降低微调学习率
            'batch_size': 32,          # 平衡：适度增加batch size（32→48），加速约1.5倍
            'dropout': 0.28,            # 优化：进一步增大dropout，防止过拟合
            'max_seq_len': 150,        # 平衡：适度减少序列长度（200→150），加速约1.3倍
            'n_epochs': 30,            # 优化：减少总轮数（模型通常在前10个epoch收敛）
            'patience': 5,             # 关键：更激进的Early Stopping，在峰值后尽早停止
            'l2_lambda': 1e-5,         # 降低：过强的L2可能限制模型表达能力
            'warmup_steps': 1800,      # 降低：约0.5个epoch完成Warmup（1800/3602≈0.5）
            'lr_decay_patience': 5,    # 增加：更保守的学习率衰减
            'lr_decay_factor': 0.5     # 新增：学习率衰减因子
        },
        'assist17': {
            # Model hyperparameters (论文表4.4)
            'embed_dim': 128,      # d_k, d_q
            'hidden_dim': 256,    # d_h
            'n_layers': 2,         # L
            
            # Triple decision parameters
            'alpha': 0.7,
            'beta': 0.3,
            'lambda_decay': 0.1,
            
            # RL parameters
            'gamma': 0.99,
            'lambda1': 0.3,
            'lambda2': 0.2,
            'lr_rl': 1e-4,
            'lambda_rl': 0.1,
            
            # Training parameters
            'lr_kt_pretrain': 0.001,
            'lr_kt_finetune': 0.0005,
            'batch_size': 64,      # 保持64（平衡泛化与速度）
            'dropout': 0.28,       # 关键：0.3→0.28，微调正则化强度
            'max_seq_len': 150,    # 关键：回归150（最优平衡点）
            'n_epochs': 30,        # 保持30
            'patience': 7,         # 5→7，给模型更多机会找到最优点
            'l2_lambda': 1e-5,     # 恢复1e-5，增强正则化
        },
        'junyi': {
            # Model hyperparameters (论文表4.4)
            'embed_dim': 256,      # d_k, d_q (Junyi知识点最多，用更大维度)
            'hidden_dim': 512,     # d_h (Junyi规模最大，用更大隐藏层)
            'n_layers': 3,         # L (Junyi知识点图最复杂，用3层)
            
            # Triple decision parameters
            'alpha': 0.65,         # Junyi略微调整
            'beta': 0.35,
            'lambda_decay': 0.1,
            
            # RL parameters
            'gamma': 0.99,
            'lambda1': 0.3,
            'lambda2': 0.2,
            'lr_rl': 1e-4,
            'lambda_rl': 0.1,
            
            # Training parameters
            'lr_kt_pretrain': 0.001,
            'lr_kt_finetune': 0.0005,
            'batch_size': 64,      # Junyi数据量最大，用更大batch
            'dropout': 0.3,        # Junyi防止过拟合，用更大dropout
            'max_seq_len': 200,
            'n_epochs': 100,
            'patience': 10
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(configs.keys())}")
    
    return configs[dataset_name]


def get_default_config():
    """Get default experiment configuration (deprecated, use get_dataset_config instead)"""
    # Return ASSIST09 config as default for backward compatibility
    return get_dataset_config('assist09')


def main():
    parser = argparse.ArgumentParser(description='Run KER-KT experiments (论文第4章)')
    parser.add_argument('--dataset', type=str, choices=['assist09', 'assist17', 'junyi', 'all'],
                       default='all', help='Dataset to run experiment on')
    parser.add_argument('--n_runs', type=int, default=5, 
                       help='Number of runs (论文4.3.2节要求5次)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run experiments
    if args.dataset == 'all':
        results = run_all_experiments(n_runs=args.n_runs)
    else:
        # Use dataset-specific config (论文表4.4)
        config = get_dataset_config(args.dataset)
        print(f"\nDataset: {args.dataset.upper()}")
        print("Configuration (论文表4.4):")
        for k, v in config.items():
            print(f"  {k}: {v}")
        results = [run_single_experiment(args.dataset, config=config, n_runs=args.n_runs)]

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
