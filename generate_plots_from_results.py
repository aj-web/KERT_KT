"""
直接从JSON结果文件生成图表
无需运行实验，使用现有结果数据生成论文图表
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = r"E:\IDAEWOREKSPACE\demo"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
SENSITIVITY_DIR = os.path.join(RESULTS_DIR, "sensitivity")
BASELINES_DIR = os.path.join(RESULTS_DIR, "baselines")


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def load_json(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_baseline_comparison(dataset='assist09'):
    """生成基线对比图表 (图4-1, 4-2, 4-3)"""
    print("\n生成基线对比图表...")
    
    # 加载数据
    if dataset == 'assist09':
        summary_file = os.path.join(RESULTS_DIR, 'summary_assist09.json')
    else:
        summary_file = os.path.join(RESULTS_DIR, 'summary_junyi.json')
    
    baseline_file = os.path.join(BASELINES_DIR, f'baseline_comparison_{dataset}.json')
    
    if not os.path.exists(baseline_file):
        print(f"  [跳过] 文件不存在: {baseline_file}")
        return False
    
    data = load_json(baseline_file)
    
    # 提取数据
    models = ['DKT', 'DKVMN', 'SAINT', 'GKT', 'DKTMR', 'KRD-KT-SL']
    aucs = []
    accs = []
    doas = []
    auc_stds = []
    acc_stds = []
    doa_stds = []
    
    for model in models:
        if model == 'KRD-KT-SL':
            m_data = data.get('krd_kt_sl', {})
        else:
            m_data = data.get('baselines', {}).get(model, {})
        
        aucs.append(m_data.get('auc_mean', 0))
        accs.append(m_data.get('acc_mean', 0))
        doas.append(m_data.get('doa_mean', 0))
        auc_stds.append(m_data.get('auc_std', 0.01))
        acc_stds.append(m_data.get('acc_std', 0.01))
        doa_stds.append(m_data.get('doa_std', 0.02))
    
    # 设置颜色
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c']
    
    x = np.arange(len(models))
    width = 0.6
    
    # 图4-1: AUC对比
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, aucs, width, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, aucs, yerr=auc_stds, fmt='none', color='black', capsize=5, capthick=1)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'图4-1 {dataset.upper()} 不同模型AUC对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    
    ensure_dir(os.path.join(FIGURES_DIR, 'baseline'))
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'baseline', f'fig4-1_{dataset}_auc.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    # 图4-2: ACC对比
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, accs, width, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, accs, yerr=acc_stds, fmt='none', color='black', capsize=5, capthick=1)
    ax.set_ylabel('ACC', fontsize=12)
    ax.set_title(f'图4-2 {dataset.upper()} 不同模型ACC对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'baseline', f'fig4-2_{dataset}_acc.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    # 图4-3: DOA对比
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, doas, width, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(x, doas, yerr=doa_stds, fmt='none', color='black', capsize=5, capthick=1)
    ax.set_ylabel('DOA', fontsize=12)
    ax.set_title(f'图4-3 {dataset.upper()} 不同模型DOA对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, doa in zip(bars, doas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{doa:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'baseline', f'fig4-3_{dataset}_doa.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    return True


def plot_sensitivity(param_name, dataset='assist09'):
    """生成敏感性分析图表 (图4-4, 4-5)"""
    print(f"\n生成敏感性分析图表: {param_name}")
    
    # 加载数据
    sensitivity_file = os.path.join(SENSITIVITY_DIR, f'sensitivity_{param_name}_{dataset}.json')
    
    if not os.path.exists(sensitivity_file):
        print(f"  [跳过] 文件不存在: {sensitivity_file}")
        return False
    
    data = load_json(sensitivity_file)
    results = data.get('results', [])
    
    if not results:
        print(f"  [跳过] 无结果数据")
        return False
    
    # 提取数据
    param_values = [r['param_value'] for r in results]
    aucs = [r['test_metrics']['auc_mean'] for r in results]
    auc_stds = [r['test_metrics']['auc_std'] for r in results]
    
    # DOA数据（如果有）
    doas = []
    doa_stds = []
    has_doa = 'doa_mean' in results[0].get('test_metrics', {})
    if has_doa:
        doas = [r['test_metrics'].get('doa_mean', 0) for r in results]
        doa_stds = [r['test_metrics'].get('doa_std', 0.02) for r in results]
    
    # 设置图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制AUC曲线
    ax.plot(param_values, aucs, 'o-', color='#3498db', linewidth=2, markersize=10, label='AUC')
    ax.fill_between(param_values, 
                    np.array(aucs) - np.array(auc_stds),
                    np.array(aucs) + np.array(auc_stds),
                    alpha=0.2, color='#3498db')
    
    # 如果有DOA数据，也绘制DOA曲线
    if has_doa and doas:
        ax.plot(param_values, doas, 's--', color='#e74c3c', linewidth=2, markersize=8, label='DOA')
        ax.fill_between(param_values, 
                        np.array(doas) - np.array(doa_stds),
                        np.array(doas) + np.array(doa_stds),
                        alpha=0.15, color='#e74c3c')
    
    # 标注最优点
    best_idx = np.argmax(aucs)
    best_value = param_values[best_idx]
    best_auc = aucs[best_idx]
    ax.axvline(x=best_value, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label=f'最优: {best_value}')
    ax.scatter([best_value], [best_auc], color='green', s=200, zorder=5, marker='*', edgecolors='black', linewidths=1)
    
    # 设置标签
    param_labels = {
        'lambda_decay': r'$\lambda$ (距离衰减系数)',
        'max_k': 'k (邻域阶数)',
        'alpha': r'$\alpha$ (三支决策上阈值)',
        'beta': r'$\beta$ (三支决策下阈值)'
    }
    
    ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=13)
    ax.set_ylabel('分数', fontsize=13)
    ax.set_title(f'图4-4 {dataset.upper()} {param_labels.get(param_name, param_name)}敏感性分析', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0.75, 0.90)
    
    # 保存图表
    ensure_dir(os.path.join(FIGURES_DIR, 'sensitivity'))
    fig_path = os.path.join(FIGURES_DIR, 'sensitivity', f'fig4-4_{param_name}_sensitivity.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    return True


def plot_ablation(dataset='assist09'):
    """生成消融实验图表"""
    print("\n生成消融实验图表...")
    
    # 加载数据
    if dataset == 'assist09':
        summary_file = os.path.join(RESULTS_DIR, 'summary_assist09.json')
    else:
        summary_file = os.path.join(RESULTS_DIR, 'summary_junyi.json')
    
    if not os.path.exists(summary_file):
        print(f"  [跳过] 文件不存在: {summary_file}")
        return False
    
    data = load_json(summary_file)
    models = data.get('models', {})
    
    # 提取数据
    model_names = []
    aucs = []
    accs = []
    doas = []
    
    for name, m_data in models.items():
        model_names.append(name.replace('KRD-KT-', '').replace('KRD-KT ', ''))
        aucs.append(m_data.get('auc_mean', m_data.get('auc', 0)))
        accs.append(m_data.get('acc_mean', m_data.get('acc', 0)))
        doas.append(m_data.get('doa_mean', m_data.get('doa', 0)))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, aucs, width, label='AUC', color='#3498db')
    bars2 = ax.bar(x, accs, width, label='ACC', color='#2ecc71')
    bars3 = ax.bar(x + width, doas, width, label='DOA', color='#e74c3c')
    
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title(f'消融实验结果对比 - {dataset.upper()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    ensure_dir(os.path.join(FIGURES_DIR, 'ablation'))
    fig_path = os.path.join(FIGURES_DIR, 'ablation', f'ablation_{dataset}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    return True


def main():
    """主函数"""
    print("="*60)
    print("直接从JSON结果生成论文图表")
    print("="*60)
    
    dataset = 'assist09'  # 可改为 'junyi'
    
    # 1. 生成基线对比图表 (图4-1, 4-2, 4-3)
    plot_baseline_comparison(dataset)
    
    # 2. 生成敏感性分析图表
    for param in ['lambda_decay', 'max_k', 'alpha', 'beta']:
        plot_sensitivity(param, dataset)
    
    # 3. 生成消融实验图表
    plot_ablation(dataset)
    
    print("\n" + "="*60)
    print("所有图表生成完成！")
    print(f"图表保存位置: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

