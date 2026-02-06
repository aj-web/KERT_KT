"""
消融实验 (Ablation Study)

实现论文第4.7节的6个消融实验变体：
1. KRD-KT w/o 3WD: 移除三支决策
2. KRD-KT w/o Multiorder: 只用1阶邻居
3. KRD-KT w/o Diff-Msg: 移除差异化消息传递
4. KRD-KT w/o Decay: 移除距离衰减
5. KRD-KT w/o NegSupp: 移除负域抑制
6. KRD-KT-SL: 纯监督学习（移除RL）

论文：张慧玲-论文0201.txt 第4.7节
"""

import torch
import numpy as np
import sys
import os
import argparse
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.krd_kt import KRDKT, train_krd_kt
from experiments.run_experiment import (
    load_processed_data, 
    create_data_loaders,
    get_dataset_config
)


class AblationKRDKT(KRDKT):
    """
    支持消融实验的KRD-KT变体
    
    通过配置参数控制不同组件的启用/禁用
    """
    
    def __init__(self, ablation_type='none', **kwargs):
        """
        Args:
            ablation_type: str, 消融类型
                - 'none': 完整模型
                - 'w/o_3WD': 移除三支决策
                - 'w/o_multiorder': 只用1阶邻居
                - 'w/o_diff_msg': 移除差异化消息传递
                - 'w/o_decay': 移除距离衰减
                - 'w/o_neg_supp': 移除负域抑制
                - 'w/o_rl': 纯监督学习
            **kwargs: 其他模型参数
        """
        self.ablation_type = ablation_type
        
        # 根据消融类型修改配置
        if ablation_type == 'w/o_3WD':
            # 移除三支决策：所有邻居使用相同处理
            kwargs['alpha'] = 1.0  # 所有邻居都是"正域"
            kwargs['beta'] = 0.0
            print(f"  [消融] 移除三支决策: 所有邻居统一处理")
        
        elif ablation_type == 'w/o_multiorder':
            # 只用1阶邻居
            kwargs['max_k'] = 1
            print(f"  [消融] 只使用1阶邻居")
        
        elif ablation_type == 'w/o_decay':
            # 移除距离衰减：ω(k) = 1
            kwargs['lambda_decay'] = 1.0  # 设为1表示无衰减
            print(f"  [消融] 移除距离衰减")
        
        elif ablation_type == 'w/o_neg_supp':
            # 移除负域抑制：γ_neg = 0
            # 需要在graph_module初始化后设置
            print(f"  [消融] 移除负域抑制")
        
        elif ablation_type == 'w/o_rl':
            # 纯监督学习
            kwargs['lambda_rl'] = 0.0
            print(f"  [消融] 移除RL（纯监督学习）")
        
        super().__init__(**kwargs)
        
        # 后处理：某些消融需要在初始化后修改
        if ablation_type == 'w/o_neg_supp':
            # 将负域抑制系数设为0
            if hasattr(self.graph_module, 'gamma_neg'):
                self.graph_module.gamma_neg.data.fill_(0.0)
                self.graph_module.gamma_neg.requires_grad = False
        
        elif ablation_type == 'w/o_diff_msg':
            # 差异化消息传递的消融比较复杂
            # 需要修改graph_module中的MLP，使其对所有区域使用相同处理
            print(f"  [消融] 移除差异化消息传递（使用统一MLP）")
            # 实现方式：让所有区域的MLP共享参数
            if hasattr(self.graph_module, 'mlp_pos') and \
               hasattr(self.graph_module, 'mlp_bnd') and \
               hasattr(self.graph_module, 'mlp_neg'):
                # 让bnd和neg使用pos的参数
                self.graph_module.mlp_bnd = self.graph_module.mlp_pos
                self.graph_module.mlp_neg = self.graph_module.mlp_pos


def run_ablation_experiment(dataset_name, ablation_type, config=None, n_runs=5):
    """
    运行单个消融实验
    
    Args:
        dataset_name: 数据集名称
        ablation_type: 消融类型
        config: 实验配置
        n_runs: 运行次数
    
    Returns:
        results: 实验结果
    """
    print(f"\n{'='*60}")
    print(f"消融实验: {ablation_type}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # 获取配置
    if config is None:
        config = get_dataset_config(dataset_name)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    dataset_info = load_processed_data(dataset_name)
    concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32).to(device)
    
    # 运行多次
    all_aucs = []
    all_accs = []
    
    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_info, config['batch_size'], config['max_seq_len']
        )
        
        # 创建消融模型
        model = AblationKRDKT(
            ablation_type=ablation_type,
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
            lambda_rl=config['lambda_rl'],
            l2_lambda=config.get('l2_lambda', 1e-5),
            dropout=config.get('dropout', 0.2)
        )
        
        model = model.to(device)
        
        # 创建checkpoint目录
        checkpoint_dir = os.path.join(
            project_root, 'checkpoints', 'ablation', dataset_name, ablation_type
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, f'best_run{run_idx+1}.pt')
        
        # 训练模型
        print("\n开始训练...")
        train_krd_kt(
            model, train_loader, val_loader, concept_graph,
            n_epochs=config['n_epochs'],
            patience=config['patience'],
            checkpoint_path=best_model_path,
            lr_kt_pretrain=config['lr_kt_pretrain'],
            lr_kt_finetune=config['lr_kt_finetune'],
            warmup_steps=config.get('warmup_steps', 0),
            lr_decay_patience=config.get('lr_decay_patience', None),
            lr_decay_factor=config.get('lr_decay_factor', 0.5)
        )
        
        # 加载最佳模型并测试
        model.load_model(best_model_path)
        test_metrics = model.evaluate(test_loader, concept_graph)
        
        print(f"Run {run_idx + 1} 测试结果:")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  ACC: {test_metrics['acc']:.4f}")
        
        all_aucs.append(test_metrics['auc'])
        all_accs.append(test_metrics['acc'])
    
    # 计算均值和标准差
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    
    print(f"\n{'='*60}")
    print(f"消融实验结果 ({n_runs} runs):")
    print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  ACC: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*60}")
    
    results = {
        'ablation_type': ablation_type,
        'dataset': dataset_name,
        'test_metrics': {
            'auc_mean': mean_auc,
            'auc_std': std_auc,
            'acc_mean': mean_acc,
            'acc_std': std_acc,
            'all_aucs': all_aucs,
            'all_accs': all_accs
        },
        'n_runs': n_runs,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return results


def run_all_ablations(dataset_name='assist09', n_runs=5):
    """
    运行所有消融实验
    
    Args:
        dataset_name: 数据集名称
        n_runs: 每个实验运行次数
    """
    ablation_types = [
        'none',           # 完整模型（baseline）
        'w/o_3WD',        # 移除三支决策
        'w/o_multiorder', # 只用1阶邻居
        'w/o_diff_msg',   # 移除差异化消息传递
        'w/o_decay',      # 移除距离衰减
        'w/o_neg_supp',   # 移除负域抑制
        'w/o_rl'          # 纯监督学习
    ]
    
    all_results = []
    
    for ablation_type in ablation_types:
        try:
            results = run_ablation_experiment(dataset_name, ablation_type, n_runs=n_runs)
            all_results.append(results)
        except Exception as e:
            print(f"消融实验 {ablation_type} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    results_dir = os.path.join(project_root, 'results', 'ablation')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(
        results_dir, f'ablation_{dataset_name}_{timestamp}.json'
    )
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n消融实验完成！结果保存到: {results_file}")
    
    # 打印汇总表（论文表4.7格式）
    print("\n" + "="*80)
    print("消融实验汇总表 (论文表4.7格式)")
    print("="*80)
    print(f"{'变体':<25} {'AUC':<20} {'ACC':<20}")
    print("-"*80)
    
    for result in all_results:
        variant = result['ablation_type']
        auc = f"{result['test_metrics']['auc_mean']:.4f} ± {result['test_metrics']['auc_std']:.4f}"
        acc = f"{result['test_metrics']['acc_mean']:.4f} ± {result['test_metrics']['acc_std']:.4f}"
        print(f"{variant:<25} {auc:<20} {acc:<20}")
    
    print("="*80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='运行KRD-KT消融实验 (论文4.7节)')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'ednet', 'junyi'],
                       help='数据集')
    parser.add_argument('--ablation', type=str, default='all',
                       choices=['all', 'none', 'w/o_3WD', 'w/o_multiorder',
                               'w/o_diff_msg', 'w/o_decay', 'w/o_neg_supp', 'w/o_rl'],
                       help='消融类型')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='运行次数')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行实验
    if args.ablation == 'all':
        results = run_all_ablations(args.dataset, n_runs=args.n_runs)
    else:
        results = [run_ablation_experiment(
            args.dataset, args.ablation, n_runs=args.n_runs
        )]
    
    print("\n消融实验完成！")


if __name__ == '__main__':
    main()

