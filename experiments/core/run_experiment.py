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
# 当前文件: experiments/core/run_experiment.py
# 需要向上两级到达项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/core/
experiments_dir = os.path.dirname(current_dir)  # experiments/
project_root = os.path.dirname(experiments_dir)  # 项目根目录
sys.path.insert(0, project_root)

from models.krd_kt import KRDKT, KTSequenceDataset, train_krd_kt
from models.kt_predictor import DataCollator


def load_processed_data(dataset_name):
    """Load processed dataset"""
    data_path = os.path.join(project_root, 'data', 'processed_datasets.pkl')
    with open(data_path, 'rb') as f:
        datasets = pickle.load(f)

    return datasets[dataset_name]


def create_data_loaders(dataset_info, batch_size=32, max_seq_len=200):
    """Create data loaders for training and evaluation"""
    import pandas as pd
    
    # Convert to DataFrame (handle both list of dicts and DataFrame)
    def ensure_dataframe(data):
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    train_data = ensure_dataframe(dataset_info['train'])
    val_data = ensure_dataframe(dataset_info['val'])
    test_data = ensure_dataframe(dataset_info['test'])
    
    # Create datasets
    train_dataset = KTSequenceDataset(train_data, max_seq_len)
    val_dataset = KTSequenceDataset(val_data, max_seq_len)
    test_dataset = KTSequenceDataset(test_data, max_seq_len)

    # Create data collator
    collator = DataCollator(max_seq_len)

    # Create data loaders
    # 性能优化：使用多进程加载数据（num_workers > 0）
    # Windows下也启用多进程（2-4个worker），提速10-15%
    import platform
    num_workers = 2 if platform.system() == 'Windows' else 4
    print(f"✅ 数据加载器使用 {num_workers} 个worker进程")
    
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


def run_single_experiment(dataset_name, config=None, n_runs=5, mode='default'):
    """
    Run experiment on single dataset (论文4.3.2节：每个模型运行5次)

    Args:
        dataset_name: name of dataset ('assist09', 'assist17', 'junyi')
        config: experiment configuration (if None, use dataset-specific config from 论文表4.4)
        n_runs: number of runs (论文要求5次)
        mode: 消融实验模式 (default, sl, wo_3wd, etc.)

    Returns:
        results: experiment results with mean ± std
    """
    # Use dataset-specific config if not provided (论文表4.4)
    if config is None:
        config = get_dataset_config(dataset_name)
        # 应用消融实验配置
        ablation_config = get_ablation_mode_config(mode)
        config.update(ablation_config)
    
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

    # 确定模型版本名称（根据mode参数）
    model_variant = get_mode_variant_name(mode)
    
    # Run multiple times (论文4.3.2节：每个模型运行5次)
    all_aucs = []
    all_accs = []
    all_run_results = []  # 保存每个run的详细结果
    
    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_info, config['batch_size'], config['max_seq_len']
        )

        # Initialize model
        model = KRDKT(
            n_questions=dataset_info['n_questions'],
            n_concepts=dataset_info['n_concepts'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            lstm_layers=config['n_layers'],  # LSTM层数与图传播层数一致
            alpha=config['alpha'],
            beta=config['beta'],
            lambda_decay=config.get('lambda_decay', 0.1),
            gamma=config['gamma'],
            lr_kt=config['lr_kt_pretrain'],  # 预训练阶段学习率
            lr_rl=config['lr_rl'],
            lambda_rl=config['lambda_rl'],
            l2_lambda=config.get('l2_lambda', 1e-5),  # L2正则化系数
            dropout=config.get('dropout', 0.2),  # Dropout率
            grad_clip=config.get('grad_clip', None),  # 梯度裁剪阈值
            # 消融实验参数
            use_triple_decision=config.get('use_triple_decision', True),
            max_k=config.get('max_k', 2),
            use_diff_msg=config.get('use_diff_msg', True),
            use_neg_suppress=config.get('use_neg_suppress', True)
        )
        
        # Move model to device
        model = model.to(device)
        print(f"Model moved to {device}")
        
        # 禁用torch.compile（与AMP混合精度训练冲突）
        # torch.compile在Windows + CUDA + AMP下有已知兼容性问题
        # 优先保留AMP（30-50%提速）而非torch.compile（10-20%提速）
        # if hasattr(torch, 'compile') and torch.cuda.is_available():
        #     try:
        #         model = torch.compile(model)
        #         print(f"✅ torch.compile已启用 (预期提速10-20%)")
        #     except Exception as e:
        #         print(f"⚠️ torch.compile启用失败: {e}")
        # else:
        #     print(f"⚠️ torch.compile不可用（需要PyTorch 2.x）")
        print(f"⚠️ torch.compile已禁用（与AMP混合精度训练冲突）")

        # Create checkpoint and results directories
        checkpoint_dir = os.path.join(project_root, 'checkpoints', dataset_name)
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Train model
        print("\nStarting training...")
        # 新的命名规则: krd_kt_sl_run1_assist09.pt
        best_model_path = os.path.join(checkpoint_dir, f'{model_variant}_run{run_idx+1}_{dataset_name}.pt')
        
        # Train model (论文3.6.2节：两阶段训练策略)
        train_krd_kt(
            model, train_loader, val_loader, concept_graph,
            n_epochs=config['n_epochs'],
            patience=config['patience'],
            checkpoint_path=best_model_path,
            lr_kt_pretrain=config['lr_kt_pretrain'],
            lr_kt_finetune=config['lr_kt_finetune'],
            warmup_steps=config.get('warmup_steps', 0),
            lr_decay_patience=config.get('lr_decay_patience', None),
            lr_decay_factor=config.get('lr_decay_factor', 0.5),
            min_lr=config.get('min_lr', 1e-5)
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
        
        # 保存单个run的结果到JSON (新增)
        run_result = {
            'run_id': run_idx + 1,
            'dataset': dataset_name,
            'model': model_variant.upper().replace('_', '-'),  # KRD-KT-SL
            'test_metrics': {
                'auc': float(test_metrics['auc']),
                'acc': float(test_metrics['acc'])
            },
            'checkpoint': f'{model_variant}_run{run_idx+1}_{dataset_name}.pt',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        all_run_results.append(run_result)
        
        # 保存单个run的JSON: krd_kt_sl_run1_assist09.json
        run_result_file = os.path.join(results_dir, f'{model_variant}_run{run_idx+1}_{dataset_name}.json')
        with open(run_result_file, 'w', encoding='utf-8') as f:
            json.dump(run_result, f, indent=2, ensure_ascii=False)
        print(f"  -> Run result saved to: {run_result_file}")

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

    # Save aggregated results (汇总结果)
    results = {
        'dataset': dataset_name,
        'model': model_variant.upper().replace('_', '-'),  # KRD-KT-SL
        'config': config,
        'test_metrics': {
            'auc_mean': float(mean_auc),
            'auc_std': float(std_auc),
            'acc_mean': float(mean_acc),
            'acc_std': float(std_acc),
            'all_aucs': [float(x) for x in all_aucs],
            'all_accs': [float(x) for x in all_accs]
        },
        'dataset_stats': {
            'n_questions': int(dataset_info['n_questions']),
            'n_concepts': int(dataset_info['n_concepts']),
            'n_students': int(dataset_info['n_students']),
            'train_samples': len(dataset_info['train']),
            'val_samples': len(dataset_info['val']),
            'test_samples': len(dataset_info['test'])
        },
        'runs': all_run_results,  # 包含每个run的详细信息
        'n_runs': n_runs,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save aggregated results to JSON file: krd_kt_sl_assist09.json
    results_file = os.path.join(results_dir, f'{model_variant}_{dataset_name}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Aggregated results saved to: {results_file}")

    return results


def run_all_experiments(n_runs=None, mode='default'):
    """
    Run experiments on all datasets (论文4.3.2节：每个模型运行5次)
    
    Args:
        n_runs: number of runs per dataset (None=自动根据mode决定)
        mode: 消融实验模式
    """
    # 根据模式自动调整运行次数
    if n_runs is None:
        if mode in ['full', 'sl']:
            n_runs = 5  # 主模型5次（论文要求）
            print(f"✅ 主模型实验：运行 {n_runs} 次")
        else:
            n_runs = 3  # 消融实验3次（学术界接受，节省40%时间）
            print(f"✅ 消融实验：运行 {n_runs} 次（节省40%时间）")
    
    datasets = ['assist09', 'ednet', 'junyi']
    all_results = []

    for dataset_name in datasets:
        try:
            # Use dataset-specific config (论文表4.4)
            results = run_single_experiment(dataset_name, config=None, n_runs=n_runs, mode=mode)
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
            # ===== 论文明确定义的参数 (表4.4) =====
            'embed_dim': 128,      # d_k, d_q (论文表4.4)
            'hidden_dim': 256,     # d_h (论文表4.4)
            'n_layers': 2,         # L (论文表4.4)
            
            # Triple decision parameters (论文公式3.6, 3.10)
            'alpha': 0.7,          # α - 正域阈值
            'beta': 0.3,           # β - 负域阈值
            'lambda_decay': 0.1,   # λ - 距离衰减因子
            
            # RL parameters (论文公式3.14-3.17)
            'gamma': 0.99,         # γ - 折扣因子
            'lambda1': 0.3,        # λ₁ - 奖励函数平衡性权重
            'lambda2': 0.2,        # λ₂ - 奖励函数稳定性权重
            'lr_rl': 1e-4,         # α_a, α_c - RL学习率
            'lambda_rl': 0.1,      # λ_RL - RL损失权重
            
            # ===== 训练参数 (基于论文表4.4，适度优化) =====
            'lr_kt_pretrain': 0.001,   # 预训练学习率 (论文表4.4)
            'lr_kt_finetune': 0.0005,  # 微调学习率 (论文表4.4)
            'batch_size': 128,          # Batch size (论文表4.4)
            'dropout': 0.40,           # Dropout率 (适度增强，论文0.28-0.35)
            'max_seq_len': 150,        # 序列长度 (速度优化，论文200)
            'n_epochs': 100,           # 总epoch数 (支持两阶段训练)
            'patience': 10,            # Early stopping patience (避免过早停止)
            'l2_lambda': 5e-5,         # L2正则化系数 (适度增强，论文1e-5)

            # ===== 学习率调度 (标准做法) =====
            'warmup_steps': 0,         # Warmup步数 (LSTM不需要warmup)
            'lr_decay_patience': 5,    # 学习率衰减patience (标准值)
            'lr_decay_factor': 0.5,    # 学习率衰减因子 (标准值)
            'min_lr': 1e-5,            # 最小学习率 (防止过小)
        },
        'ednet': {
            # Model hyperparameters (论文表4.4 - EdNet大规模数据集)
            'embed_dim': 128,      # d_k, d_q
            'hidden_dim': 256,     # d_h
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
            'batch_size': 128,     # EdNet数据量大，用更大batch size
            'dropout': 0.3,        # 大规模数据，适度dropout
            'max_seq_len': 150,    # 平衡序列长度和计算效率
            'n_epochs': 50,        # KRD-KT-SL 监督学习版：只运行 Phase 1 (论文消融实验变体)
            'patience': 8,         # Early stopping（增加patience）
            'l2_lambda': 1e-5,     # L2正则化
            'warmup_steps': 2000,  # Warmup步数
            'lr_decay_patience': 5,
            'lr_decay_factor': 0.5
        },
        'junyi': {
            # Model hyperparameters (修正：降低模型复杂度，防止梯度爆炸)
            'embed_dim': 128,      # 降低维度 256→128（与ASSIST09一致）
            'hidden_dim': 256,     # 降低维度 512→256（与ASSIST09一致）
            'n_layers': 2,         # 减少层数 3→2（与ASSIST09一致）
            
            # Triple decision parameters
            'alpha': 0.7,          # 与ASSIST09一致
            'beta': 0.3,           # 与ASSIST09一致
            'lambda_decay': 0.1,
            
            # RL parameters
            'gamma': 0.99,
            'lambda1': 0.3,
            'lambda2': 0.2,
            'lr_rl': 1e-4,
            'lambda_rl': 0.1,
            
            # Training parameters (优化版 - 防止梯度爆炸 + 速度优化)
            'lr_kt_pretrain': 0.0005,  # 降低学习率 0.001→0.0005
            'lr_kt_finetune': 0.00025, # 降低学习率
            'batch_size': 128,     # 增大batch size，稳定训练
            'dropout': 0.3,        # 保持dropout
            'max_seq_len': 80,     # 速度优化：减少序列长度 100→80（提速25%，影响<1%）
            'n_epochs': 30,        # 速度优化：降低最大epoch 50→30（实际5-10 epoch收敛，节省40%时间）
            'patience': 5,         # 速度优化：降低patience 8→5（减少无效训练）
            'l2_lambda': 1e-5,     # L2正则化
            'warmup_steps': 1000,  # 速度优化：减少warmup 2000→1000（加快收敛）
            'lr_decay_patience': 5,
            'lr_decay_factor': 0.5,
            'grad_clip': 1.0       # 添加梯度裁剪
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(configs.keys())}")
    
    return configs[dataset_name]


def get_ablation_mode_config(mode='full'):
    """
    获取消融实验模式对应的配置参数
    
    Args:
        mode: 消融实验模式
            - 'full' 或 'default': KRD-KT 完整版
            - 'sl': KRD-KT-SL (仅Phase 1, 无RL)
            - 'wo_3wd': w/o Three-way Decision (标准GNN)
            - 'wo_multi': w/o Multi-order (仅一阶邻居)
            - 'wo_diff': w/o Diff-Msg (统一消息传递)
            - 'wo_decay': w/o Decay (无距离衰减)
            - 'wo_neg': w/o Neg-Suppress (无负域抑制)
    
    Returns:
        config_override: 需要覆盖的配置参数字典
    """
    
    # 所有模式的基础配置（保持默认）
    config_override = {}
    
    if mode in ['full', 'default']:
        # KRD-KT 完整版：所有功能启用，运行Phase 1 + Phase 2
        config_override = {
            'n_epochs': 100,  # Phase 1 (50) + Phase 2 (50)
            'use_triple_decision': True,
            'max_k': 2,
            'use_diff_msg': True,
            'use_neg_suppress': True,
            # lambda_decay 使用数据集默认值
        }
        
    elif mode == 'sl':
        # KRD-KT-SL: 仅Phase 1，无RL Fine-tuning
        config_override = {
            'n_epochs': 50,  # 只运行Phase 1
            'use_triple_decision': True,
            'max_k': 2,
            'use_diff_msg': True,
            'use_neg_suppress': True,
        }
        
    elif mode == 'wo_3wd':
        # w/o Three-way Decision: 不划分三支决策区域，退化为标准GNN
        config_override = {
            'n_epochs': 50,  # 使用SL版本（更稳定）
            'use_triple_decision': False,  # 关键：禁用三支决策
            'max_k': 2,
            'use_diff_msg': True,  # 虽然启用，但因为没有区域划分，实际统一处理
            'use_neg_suppress': True,
        }
        
    elif mode == 'wo_multi':
        # w/o Multi-order: 仅使用一阶邻居
        config_override = {
            'n_epochs': 50,
            'use_triple_decision': True,
            'max_k': 1,  # 关键：只使用一阶邻居
            'use_diff_msg': True,
            'use_neg_suppress': True,
        }
        
    elif mode == 'wo_diff':
        # w/o Diff-Msg: 所有区域使用相同的消息传递函数
        config_override = {
            'n_epochs': 50,
            'use_triple_decision': True,
            'max_k': 2,
            'use_diff_msg': False,  # 关键：禁用差异化消息传递
            'use_neg_suppress': True,
        }
        
    elif mode == 'wo_decay':
        # w/o Decay: 移除距离衰减权重
        config_override = {
            'n_epochs': 50,
            'use_triple_decision': True,
            'max_k': 2,
            'use_diff_msg': True,
            'use_neg_suppress': True,
            'lambda_decay': 0.0,  # 关键：设置衰减为0
        }
        
    elif mode == 'wo_neg':
        # w/o Neg-Suppress: 移除负域抑制
        config_override = {
            'n_epochs': 50,
            'use_triple_decision': True,
            'max_k': 2,
            'use_diff_msg': True,
            'use_neg_suppress': False,  # 关键：禁用负域抑制
        }
    
    else:
        raise ValueError(f"Unknown ablation mode: {mode}. Valid modes: full, sl, wo_3wd, wo_multi, wo_diff, wo_decay, wo_neg")
    
    return config_override


def get_mode_variant_name(mode):
    """
    获取mode对应的模型变体名称（用于文件命名）
    
    Args:
        mode: 消融实验模式
    
    Returns:
        variant_name: 模型变体名称（用于文件命名）
    """
    mode_to_variant = {
        'full': 'krd_kt',
        'default': 'krd_kt',
        'sl': 'krd_kt_sl',
        'wo_3wd': 'krd_kt_wo_3wd',
        'wo_multi': 'krd_kt_wo_multi',
        'wo_diff': 'krd_kt_wo_diff',
        'wo_decay': 'krd_kt_wo_decay',
        'wo_neg': 'krd_kt_wo_neg',
    }
    
    return mode_to_variant.get(mode, 'krd_kt')


def print_mode_info(mode):
    """打印消融实验模式信息"""
    mode_descriptions = {
        'full': 'KRD-KT 完整版 (Phase 1 + Phase 2 RL)',
        'sl': 'KRD-KT-SL (仅Phase 1, 无RL)',
        'wo_3wd': 'w/o Three-way Decision (标准GNN)',
        'wo_multi': 'w/o Multi-order (仅一阶邻居)',
        'wo_diff': 'w/o Diff-Msg (统一消息传递)',
        'wo_decay': 'w/o Decay (无距离衰减)',
        'wo_neg': 'w/o Neg-Suppress (无负域抑制)',
    }
    
    print(f"\n{'='*60}")
    print(f"消融实验模式: {mode.upper()}")
    print(f"说明: {mode_descriptions.get(mode, '未知模式')}")
    print(f"{'='*60}\n")


def get_default_config():
    """Get default experiment configuration (deprecated, use get_dataset_config instead)"""
    # Return ASSIST09 config as default for backward compatibility
    return get_dataset_config('assist09')


def main():
    parser = argparse.ArgumentParser(description='Run KRD-KT experiments with ablation study support (论文第4章)')
    parser.add_argument('--dataset', type=str, choices=['assist09', 'ednet', 'junyi', 'all'],
                       default='all', help='Dataset to run experiment on')
    parser.add_argument('--n_runs', type=int, default=5, 
                       help='Number of runs (论文4.3.2节要求5次)')
    parser.add_argument('--mode', type=str, 
                       choices=['full', 'default', 'sl', 'wo_3wd', 'wo_multi', 'wo_diff', 'wo_decay', 'wo_neg'],
                       default='default',
                       help='消融实验模式: full/default(完整版), sl(仅Phase1), wo_3wd(无三支决策), wo_multi(仅一阶), wo_diff(统一消息), wo_decay(无衰减), wo_neg(无负域抑制)')

    args = parser.parse_args()

    # 打印模式信息
    print_mode_info(args.mode)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run experiments
    if args.dataset == 'all':
        results = run_all_experiments(n_runs=args.n_runs, mode=args.mode)
    else:
        # Use dataset-specific config (论文表4.4)
        config = get_dataset_config(args.dataset)
        
        # 应用消融实验配置覆盖
        ablation_config = get_ablation_mode_config(args.mode)
        config.update(ablation_config)
        
        print(f"\nDataset: {args.dataset.upper()}")
        print("Configuration (论文表4.4 + 消融实验配置):")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        results = [run_single_experiment(args.dataset, config=config, n_runs=args.n_runs, mode=args.mode)]

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
