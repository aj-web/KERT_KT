"""
三支决策邻域划分可视化
论文4.9节：三支决策邻域划分可视化

功能：
1. 展示特定时间步下的三支决策结果
2. 可视化邻域知识点在正域、边界域、负域的划分
3. 展示不同域的知识点及其路径强度
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import networkx as nx
import os
import sys
import argparse
from datetime import datetime

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/visualization/
experiments_dir = os.path.dirname(current_dir)  # experiments/
project_root = os.path.dirname(experiments_dir)  # 项目根目录
sys.path.insert(0, project_root)

from models.krd_kt import KRDKT
from models.neighborhood_extractor import NeighborhoodExtractor
from models.path_strength import PathStrengthCalculator

# 直接导入 run_experiment 模块
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_experiment", 
    os.path.join(experiments_dir, "core", "run_experiment.py")
)
run_experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment)

load_processed_data = run_experiment.load_processed_data
get_dataset_config = run_experiment.get_dataset_config


def load_trained_model(checkpoint_path, dataset_info, config, device='cuda'):
    """
    加载训练好的模型
    """
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
        use_triple_decision=config.get('use_triple_decision', True),
        max_k=config.get('max_k', 2),
        use_diff_msg=config.get('use_diff_msg', True),
        use_neg_suppress=config.get('use_neg_suppress', True)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 使用模型自带的load_model方法
    model.graph_module.load_state_dict(checkpoint['graph_module'])
    model.kt_predictor.load_state_dict(checkpoint['kt_predictor'])
    model.actor_critic.load_state_dict(checkpoint['actor_critic'])
    model.concept_embeddings = checkpoint['concept_embeddings']
    model.current_thresholds = checkpoint['current_thresholds']
    
    model = model.to(device)
    model.eval()
    
    return model


def compute_triple_decision_partition(concept_id, neighborhoods, strength_matrices,
                                      similarity_matrix, alpha=0.7, beta=0.3, k=1):
    """
    计算三支决策的邻域划分
    
    Args:
        concept_id: 目标知识点ID
        neighborhoods: 邻域字典
        strength_matrices: 路径强度矩阵字典
        similarity_matrix: 相似度矩阵
        alpha: 正域阈值
        beta: 负域阈值
        k: 邻域阶数
    
    Returns:
        partition: {'positive': [], 'boundary': [], 'negative': []}
    """
    partition = {'positive': [], 'boundary': [], 'negative': []}
    
    if concept_id not in neighborhoods or k not in neighborhoods[concept_id]:
        return partition
    
    k_neighbors = neighborhoods[concept_id][k]
    if len(k_neighbors) == 0:
        return partition
    
    # 获取路径强度
    strength_matrix = strength_matrices[k]
    neighbor_strengths = strength_matrix[concept_id, k_neighbors]
    
    # 三支决策划分
    for i, neighbor_id in enumerate(k_neighbors):
        strength = neighbor_strengths[i]
        
        if strength >= alpha:
            partition['positive'].append((neighbor_id, float(strength)))
        elif strength <= beta:
            partition['negative'].append((neighbor_id, float(strength)))
        else:
            partition['boundary'].append((neighbor_id, float(strength)))
    
    return partition


def plot_triple_decision_graph(concept_id, partition, concept_graph, 
                               concept_names=None, output_path=None,
                               alpha=0.7, beta=0.3, k=1):
    """
    绘制三支决策邻域划分图
    
    Args:
        concept_id: 目标知识点ID
        partition: 三支决策划分结果
        concept_graph: 知识点图
        concept_names: 知识点名称字典
        output_path: 输出路径
        alpha: 正域阈值
        beta: 负域阈值
        k: 邻域阶数
    """
    # 创建图
    G = nx.Graph()
    
    # 添加中心节点
    G.add_node(concept_id, node_type='center')
    
    # 添加邻域节点
    all_neighbors = []
    node_colors = []
    node_sizes = []
    
    # 中心节点
    all_neighbors.append(concept_id)
    node_colors.append('gold')
    node_sizes.append(1000)
    
    # 正域节点（绿色）
    for neighbor_id, strength in partition['positive']:
        G.add_node(neighbor_id, node_type='positive')
        G.add_edge(concept_id, neighbor_id, weight=strength)
        all_neighbors.append(neighbor_id)
        node_colors.append('lightgreen')
        node_sizes.append(800)
    
    # 边界域节点（黄色）
    for neighbor_id, strength in partition['boundary']:
        G.add_node(neighbor_id, node_type='boundary')
        G.add_edge(concept_id, neighbor_id, weight=strength)
        all_neighbors.append(neighbor_id)
        node_colors.append('lightyellow')
        node_sizes.append(800)
    
    # 负域节点（红色）
    for neighbor_id, strength in partition['negative']:
        G.add_node(neighbor_id, node_type='negative')
        G.add_edge(concept_id, neighbor_id, weight=strength)
        all_neighbors.append(neighbor_id)
        node_colors.append('lightcoral')
        node_sizes.append(800)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 使用 spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, ax=ax)
    
    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                          alpha=0.6, ax=ax)
    
    # 绘制标签
    if concept_names:
        labels = {n: concept_names.get(n, f'C{n}') for n in all_neighbors}
    else:
        labels = {n: f'C{n}' for n in all_neighbors}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gold', label=f'Target Concept (C{concept_id})'),
        Patch(facecolor='lightgreen', label=f'Positive Domain (≥{alpha})'),
        Patch(facecolor='lightyellow', label=f'Boundary Domain ({beta}<s<{alpha})'),
        Patch(facecolor='lightcoral', label=f'Negative Domain (≤{beta})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # 设置标题
    ax.set_title(f'Three-way Decision Neighborhood Partition for Concept {concept_id} (k={k})',
                fontsize=14, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        # PNG
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {png_path}")
        
        # PDF
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  [OK] Saved: {output_path}")
    
    plt.close()


def plot_triple_decision_bar(partition, concept_id, output_path=None):
    """
    绘制三支决策的柱状图
    
    Args:
        partition: 三支决策划分结果
        concept_id: 目标知识点ID
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备数据
    categories = ['Positive Domain', 'Boundary Domain', 'Negative Domain']
    counts = [len(partition['positive']), len(partition['boundary']), len(partition['negative'])]
    colors = ['lightgreen', 'lightyellow', 'lightcoral']
    
    # 绘制柱状图
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Neighbors', fontsize=12)
    ax.set_title(f'Three-way Decision Partition Statistics for Concept {concept_id}',
                fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        # PNG
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {png_path}")
        
        # PDF
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  [OK] Saved: {output_path}")
    
    plt.close()


def visualize_triple_decision(dataset_name='assist09',
                              checkpoint_path=None,
                              concept_id=None,
                              k=1,
                              output_dir=None,
                              device='auto'):
    """
    可视化三支决策邻域划分
    
    Args:
        dataset_name: 数据集名称
        checkpoint_path: 模型检查点路径
        concept_id: 知识点ID（None表示随机选择）
        k: 邻域阶数
        output_dir: 输出目录
        device: 设备
    """
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"三支决策邻域划分可视化")
    print(f"数据集: {dataset_name.upper()}")
    print(f"邻域阶数: {k}")
    print(f"设备: {device}")
    print(f"{'='*80}")
    
    # 加载数据集
    dataset_info = load_processed_data(dataset_name)
    concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32)
    n_concepts = dataset_info['n_concepts']
    
    # 如果没有指定检查点，尝试找到最新的
    if checkpoint_path is None:
        checkpoint_dir = os.path.join(project_root, 'checkpoints', dataset_name)
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"使用检查点: {checkpoint_path}")
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"错误：找不到模型检查点！")
        print(f"请指定 --checkpoint 参数")
        return
    
    # 加载模型
    config = get_dataset_config(dataset_name)
    model = load_trained_model(checkpoint_path, dataset_info, config, device)
    
    # 初始化图模块
    model.initialize_graph_modules(concept_graph)
    
    # 选择知识点
    if concept_id is None:
        # 选择一个有足够邻居的知识点
        concept_graph_np = concept_graph.numpy()
        neighbor_counts = np.sum(concept_graph_np > 0, axis=1)
        valid_concepts = np.where(neighbor_counts >= 5)[0]
        
        if len(valid_concepts) > 0:
            concept_id = np.random.choice(valid_concepts)
        else:
            concept_id = np.random.randint(0, n_concepts)
    
    print(f"知识点ID: {concept_id}")
    
    # 获取邻域和路径强度
    neighborhoods = model.neighborhood_extractor.neighborhoods
    
    # 计算相似度矩阵
    model.eval()
    with torch.no_grad():
        concept_embeds = model.graph_module.concept_embed.weight
        similarity_matrix = torch.matmul(concept_embeds, concept_embeds.T)
        similarity_matrix = torch.sigmoid(similarity_matrix)
    
    # 获取路径强度矩阵（直接从模型中获取）
    strength_matrices = {
        1: model.path_strength_calculator.get_strength_matrix(1),
        2: model.path_strength_calculator.get_strength_matrix(2)
    }
    
    # 计算三支决策划分
    print(f"\n计算 {k} 阶邻域的三支决策划分...")
    partition = compute_triple_decision_partition(
        concept_id, 
        neighborhoods, 
        strength_matrices,
        similarity_matrix,
        alpha=config['alpha'],
        beta=config['beta'],
        k=k
    )
    
    print(f"正域节点数: {len(partition['positive'])}")
    print(f"边界域节点数: {len(partition['boundary'])}")
    print(f"负域节点数: {len(partition['negative'])}")
    
    # 输出详细信息
    if partition['positive']:
        print(f"\n正域节点 (强度 ≥ {config['alpha']}):")
        for nid, strength in partition['positive'][:5]:
            print(f"  C{nid}: {strength:.4f}")
    
    if partition['boundary']:
        print(f"\n边界域节点 ({config['beta']} < 强度 < {config['alpha']}):")
        for nid, strength in partition['boundary'][:5]:
            print(f"  C{nid}: {strength:.4f}")
    
    if partition['negative']:
        print(f"\n负域节点 (强度 ≤ {config['beta']}):")
        for nid, strength in partition['negative'][:5]:
            print(f"  C{nid}: {strength:.4f}")
    
    # 绘制可视化
    print("\n绘制三支决策可视化图...")
    if output_dir is None:
        output_dir = os.path.join(project_root, 'figures', 'triple_decision')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取知识点名称（如果有）
    concept_names = dataset_info.get('id_to_concept', None)
    
    # 1. 绘制网络图
    graph_output_path = os.path.join(output_dir, 
                                     f'triple_decision_graph_c{concept_id}_k{k}_{dataset_name}.pdf')
    plot_triple_decision_graph(
        concept_id, 
        partition, 
        concept_graph,
        concept_names=concept_names,
        output_path=graph_output_path,
        alpha=config['alpha'],
        beta=config['beta'],
        k=k
    )
    
    # 2. 绘制柱状图
    bar_output_path = os.path.join(output_dir,
                                   f'triple_decision_bar_c{concept_id}_k{k}_{dataset_name}.pdf')
    plot_triple_decision_bar(partition, concept_id, output_path=bar_output_path)
    
    print(f"\n{'='*80}")
    print("三支决策邻域划分可视化完成！")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='三支决策邻域划分可视化（论文4.9节）')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'junyi', 'ednet'],
                       help='数据集名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--concept_id', type=int, default=None,
                       help='知识点ID（None表示随机选择）')
    parser.add_argument('--k', type=int, default=1,
                       choices=[1, 2],
                       help='邻域阶数')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    visualize_triple_decision(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        concept_id=args.concept_id,
        k=args.k,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()

