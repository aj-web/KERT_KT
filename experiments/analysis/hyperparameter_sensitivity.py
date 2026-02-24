"""
超参数敏感性分析实验脚本
论文4.6节：超参数敏感性分析

分析以下超参数对模型性能的影响：
1. 距离衰减系数 λ_decay: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
2. 邻域阶数 max_k: [1, 2, 3, 4]
3. 三支决策阈值 α: [0.5, 0.6, 0.7, 0.8, 0.9]
4. 三支决策阈值 β: [0.1, 0.2, 0.3, 0.4, 0.5]
"""

import torch
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# Add project root to Python path
# 当前文件: experiments/analysis/hyperparameter_sensitivity.py
# 需要向上两级到达项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/analysis/
experiments_dir = os.path.dirname(current_dir)  # experiments/
project_root = os.path.dirname(experiments_dir)  # 项目根目录
sys.path.insert(0, project_root)

# 直接导入 run_experiment 模块
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_experiment", 
    os.path.join(experiments_dir, "core", "run_experiment.py")
)
run_experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment)

get_dataset_config = run_experiment.get_dataset_config
load_processed_data = run_experiment.load_processed_data
create_data_loaders = run_experiment.create_data_loaders
run_single_experiment = run_experiment.run_single_experiment


def run_sensitivity_experiment(dataset_name, param_name, param_value, 
                               n_runs=3, device='cuda'):
    """
    运行单个敏感性分析实验
    
    Args:
        dataset_name: 数据集名称
        param_name: 参数名称 ('lambda_decay', 'max_k', 'alpha', 'beta')
        param_value: 参数值
        n_runs: 运行次数
        device: 设备
    
    Returns:
        result: 实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"Testing {param_name} = {param_value}")
    print(f"{'='*60}")
    
    # 获取基础配置
    config = get_dataset_config(dataset_name)
    
    # 修改目标参数
    config[param_name] = param_value
    
    # 使用 sl 模式（稳定且快速）
    config['use_triple_decision'] = True
    config['max_k'] = config.get('max_k', 2)
    config['use_diff_msg'] = True
    config['use_neg_suppress'] = True
    config['n_epochs'] = 100  # KRD-KT-SL (两阶段训练: Phase 1=50, Phase 2=50)
    
    # 如果测试的是 max_k，需要特殊处理
    if param_name == 'max_k':
        config['max_k'] = int(param_value)
    
    # 运行实验
    try:
        # 临时修改 run_single_experiment 来支持自定义配置
        from models.krd_kt import KRDKT, train_krd_kt
        
        # 加载数据
        dataset_info = load_processed_data(dataset_name)
        concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32).to(device)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_info,
            batch_size=config['batch_size'],
            max_seq_len=config['max_seq_len']
        )
        
        all_aucs = []
        all_accs = []
        
        for run_idx in range(n_runs):
            print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
            
            # 初始化模型
            model = KRDKT(
                n_questions=dataset_info['n_questions'],
                n_concepts=dataset_info['n_concepts'],
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim'],
                n_layers=config['n_layers'],
                lstm_layers=config.get('lstm_layers', config.get('n_layers', 2)),
                alpha=config['alpha'],
                beta=config['beta'],
                lambda_decay=config['lambda_decay'],
                gamma=config['gamma'],
                lr_kt=config['lr_kt_pretrain'],
                lr_rl=config['lr_rl'],
                lambda_rl=config['lambda_rl'],
                l2_lambda=config['l2_lambda'],
                dropout=config['dropout'],
                grad_clip=config.get('grad_clip', None),
                use_triple_decision=config['use_triple_decision'],
                max_k=config['max_k'],
                use_diff_msg=config['use_diff_msg'],
                use_neg_suppress=config['use_neg_suppress']
            )
            
            model = model.to(device)
            
            # 训练模型
            # 创建临时checkpoint目录
            temp_checkpoint_dir = os.path.join(project_root, 'checkpoints', 'temp_sensitivity')
            os.makedirs(temp_checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(temp_checkpoint_dir, f'{param_name}_{param_value}_run{run_idx+1}.pt')
            
            best_model, train_history = train_krd_kt(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                concept_graph=concept_graph,
                n_epochs=config['n_epochs'],
                patience=config['patience'],
                checkpoint_path=checkpoint_path,
                lr_kt_pretrain=config.get('lr_kt_pretrain', 0.001),
                lr_kt_finetune=config.get('lr_kt_finetune', 0.0005),
                warmup_steps=config.get('warmup_steps', 0),
                lr_decay_patience=config.get('lr_decay_patience', 5),
                lr_decay_factor=config.get('lr_decay_factor', 0.5)
            )
            
            # 测试集评估
            test_metrics = best_model.evaluate(test_loader, concept_graph)
            
            print(f"Run {run_idx + 1} Test Results: "
                  f"AUC={test_metrics['auc']:.4f}, ACC={test_metrics['acc']:.4f}")
            
            all_aucs.append(test_metrics['auc'])
            all_accs.append(test_metrics['acc'])
            
            # 清理临时checkpoint文件
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"  -> Cleaned up temporary checkpoint: {checkpoint_path}")
        
        # 计算均值和标准差
        mean_auc = np.mean(all_aucs)
        std_auc = np.std(all_aucs)
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        
        result = {
            'param_name': param_name,
            'param_value': float(param_value),
            'dataset': dataset_name,
            'n_runs': n_runs,
            'auc_mean': float(mean_auc),
            'auc_std': float(std_auc),
            'acc_mean': float(mean_acc),
            'acc_std': float(std_acc),
            'all_aucs': [float(x) for x in all_aucs],
            'all_accs': [float(x) for x in all_accs]
        }
        
        print(f"\n{param_name}={param_value} Results ({n_runs} runs):")
        print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  ACC: {mean_acc:.4f} ± {std_acc:.4f}")
        
        return result
        
    except Exception as e:
        print(f"Error in sensitivity experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_sensitivity_results(results, param_name, dataset_name, output_dir):
    """
    绘制敏感性分析结果图
    
    Args:
        results: 实验结果列表
        param_name: 参数名称
        dataset_name: 数据集名称
        output_dir: 输出目录
    """
    # 提取数据
    param_values = [r['param_value'] for r in results]
    auc_means = [r['auc_mean'] for r in results]
    auc_stds = [r['auc_std'] for r in results]
    acc_means = [r['acc_mean'] for r in results]
    acc_stds = [r['acc_std'] for r in results]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 参数名称映射（用于图表标题）
    param_labels = {
        'lambda_decay': 'Distance Decay λ',
        'max_k': 'Neighborhood Order k',
        'alpha': 'Three-way Decision Threshold α',
        'beta': 'Three-way Decision Threshold β'
    }
    
    param_label = param_labels.get(param_name, param_name)
    
    # 绘制 AUC 曲线
    ax1.errorbar(param_values, auc_means, yerr=auc_stds, 
                 marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel(param_label, fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title(f'AUC vs {param_label} ({dataset_name.upper()})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 绘制 ACC 曲线
    ax2.errorbar(param_values, acc_means, yerr=acc_stds,
                 marker='s', linewidth=2, markersize=8, capsize=5, color='orange')
    ax2.set_xlabel(param_label, fontsize=12)
    ax2.set_ylabel('ACC', fontsize=12)
    ax2.set_title(f'ACC vs {param_label} ({dataset_name.upper()})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为 PNG
    png_path = os.path.join(output_dir, f'sensitivity_{param_name}_{dataset_name}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {png_path}")
    
    # 保存为 PDF
    pdf_path = os.path.join(output_dir, f'sensitivity_{param_name}_{dataset_name}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"  [OK] Saved: {pdf_path}")
    
    plt.close()


def run_sensitivity_analysis(dataset_name='assist09', param_name='lambda_decay',
                             param_values=None, n_runs=3, device='auto'):
    """
    运行完整的敏感性分析
    
    Args:
        dataset_name: 数据集名称
        param_name: 参数名称
        param_values: 参数值列表
        n_runs: 每组参数运行次数
        device: 设备
    """
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"超参数敏感性分析: {param_name}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"设备: {device}")
    print(f"{'='*80}")
    
    # 默认参数值
    if param_values is None:
        default_values = {
            'lambda_decay': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'max_k': [1, 2, 3, 4],
            'alpha': [0.5, 0.6, 0.7, 0.8, 0.9],
            'beta': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        param_values = default_values.get(param_name, [])
    
    print(f"测试参数值: {param_values}")
    print(f"每组运行次数: {n_runs}")
    
    # 运行实验
    all_results = []
    for param_value in param_values:
        result = run_sensitivity_experiment(
            dataset_name=dataset_name,
            param_name=param_name,
            param_value=param_value,
            n_runs=n_runs,
            device=device
        )
        if result:
            all_results.append(result)
    
    # 保存结果
    results_dir = os.path.join(project_root, 'results', 'sensitivity')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'sensitivity_{param_name}_{dataset_name}.json')
    results_data = {
        'param_name': param_name,
        'dataset': dataset_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_runs': n_runs,
        'param_values': param_values,
        'results': all_results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"结果已保存: {results_file}")
    print(f"{'='*60}")
    
    # 绘制图表
    print("\n绘制敏感性分析图表...")
    figures_dir = os.path.join(project_root, 'figures', 'sensitivity')
    plot_sensitivity_results(all_results, param_name, dataset_name, figures_dir)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("敏感性分析总结")
    print(f"{'='*80}")
    print(f"{'参数值':<15} {'AUC (mean±std)':<25} {'ACC (mean±std)':<25}")
    print("-" * 65)
    for result in all_results:
        param_val = result['param_value']
        auc_str = f"{result['auc_mean']:.4f} ± {result['auc_std']:.4f}"
        acc_str = f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f}"
        print(f"{param_val:<15} {auc_str:<25} {acc_str:<25}")
    
    # 找出最佳参数值
    best_result = max(all_results, key=lambda x: x['auc_mean'])
    print(f"\n最佳参数值: {param_name} = {best_result['param_value']}")
    print(f"  AUC: {best_result['auc_mean']:.4f} ± {best_result['auc_std']:.4f}")
    print(f"  ACC: {best_result['acc_mean']:.4f} ± {best_result['acc_std']:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='超参数敏感性分析（论文4.6节）')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'junyi', 'ednet'],
                       help='数据集名称')
    parser.add_argument('--param', type=str, required=True,
                       choices=['lambda_decay', 'max_k', 'alpha', 'beta'],
                       help='要分析的超参数')
    parser.add_argument('--values', nargs='+', type=float, default=None,
                       help='参数值列表（可选，默认使用预设范围）')
    parser.add_argument('--n_runs', type=int, default=3,
                       help='每组参数运行次数（默认3次）')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 运行敏感性分析
    run_sensitivity_analysis(
        dataset_name=args.dataset,
        param_name=args.param,
        param_values=args.values,
        n_runs=args.n_runs,
        device=args.device
    )
    
    print("\n敏感性分析完成！")


if __name__ == "__main__":
    main()

