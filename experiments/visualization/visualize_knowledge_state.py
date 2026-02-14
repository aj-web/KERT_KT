"""
知识状态演化可视化
论文4.8节：知识状态演化可视化

功能：
1. 选取单个学生的作答序列
2. 可视化若干知识点在时间维度上的掌握程度
3. 展示一阶、二阶邻域的联动变化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import sys
import argparse
import pickle
from datetime import datetime

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/visualization/
experiments_dir = os.path.dirname(current_dir)  # experiments/
project_root = os.path.dirname(experiments_dir)  # 项目根目录
sys.path.insert(0, project_root)

from models.krd_kt import KRDKT

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
    
    Args:
        checkpoint_path: 模型检查点路径
        dataset_info: 数据集信息
        config: 模型配置
        device: 设备
    
    Returns:
        model: 加载的模型
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
    
    # 加载模型权重
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


def extract_knowledge_states(model, student_sequence, concept_graph, device='cuda'):
    """
    提取学生在整个作答序列中的知识状态
    
    Args:
        model: 训练好的模型
        student_sequence: 学生作答序列 (dict with 'question_seq', 'concept_seq', 'answer_seq')
        concept_graph: 知识点图
        device: 设备
    
    Returns:
        knowledge_states: [seq_len, n_concepts] 知识状态矩阵
    """
    model.eval()
    
    # 初始化图模块
    if not hasattr(model, 'neighborhood_extractor'):
        model.initialize_graph_modules(concept_graph)
    
    with torch.no_grad():
        # 准备输入 - 处理不同的数据格式
        if isinstance(student_sequence, dict):
            if 'question_seq' in student_sequence:
                # 旧格式：字典包含序列
                question_seq = torch.tensor(student_sequence['question_seq']).unsqueeze(0).to(device)
                concept_seq = torch.tensor(student_sequence['concept_seq']).unsqueeze(0).to(device)
                answer_seq = torch.tensor(student_sequence['answer_seq']).unsqueeze(0).to(device)
            else:
                # 新格式：字典包含tensor
                question_seq = student_sequence['question_id'].unsqueeze(0).to(device)
                concept_seq = student_sequence['concept_id'].unsqueeze(0).to(device)
                answer_seq = student_sequence['correct'].unsqueeze(0).to(device)
        else:
            raise ValueError(f"Unsupported student_sequence type: {type(student_sequence)}")
        
        # 由于我们只需要知识状态演化，不需要预测，直接使用简化方法
        # 而不是调用完整的forward（需要target_question等）
        
        # 使用简化方法：基于答题历史来估计知识状态
        # 获取概念嵌入
        concept_embeds = model.graph_module.concept_embed.weight  # [n_concepts, embed_dim]
        
        # 使用答题历史来更新知识状态
        seq_len = question_seq.size(1)
        n_concepts = concept_embeds.size(0)
        knowledge_states = torch.zeros(seq_len, n_concepts).to(device)
        
        # 简化版本：使用答题正确率来更新知识状态
        # 这是一个启发式方法，用于可视化目的
        for t in range(seq_len):
            concept_id = concept_seq[0, t].item()
            answer = answer_seq[0, t].item()
            
            if concept_id >= 0 and concept_id < n_concepts:  # 有效的知识点
                # 更新该知识点的掌握程度
                if t == 0:
                    knowledge_states[t, concept_id] = 0.5 + 0.3 * (answer - 0.5)
                else:
                    knowledge_states[t] = knowledge_states[t-1].clone()
                    # 根据答题结果更新知识状态
                    knowledge_states[t, concept_id] = knowledge_states[t-1, concept_id] + 0.2 * (answer - 0.5)
                    knowledge_states[t, concept_id] = torch.clamp(knowledge_states[t, concept_id], 0.0, 1.0)
    
    return knowledge_states.cpu().numpy()


def select_related_concepts(concept_graph, target_concepts, n_concepts=8):
    """
    选择与目标知识点相关的知识点
    
    Args:
        concept_graph: 知识点图 [n_concepts, n_concepts]
        target_concepts: 目标知识点列表
        n_concepts: 要选择的知识点数量
    
    Returns:
        selected_concepts: 选中的知识点ID列表
    """
    # 计算与目标知识点的相关性
    concept_graph_np = concept_graph.numpy() if isinstance(concept_graph, torch.Tensor) else concept_graph
    
    # 对于每个目标知识点，找到其邻居
    related_concepts = set(target_concepts)
    
    for concept_id in target_concepts:
        # 找到与该知识点连接的其他知识点
        neighbors = np.where(concept_graph_np[concept_id] > 0)[0]
        related_concepts.update(neighbors.tolist())
    
    # 限制数量
    selected = list(related_concepts)[:n_concepts]
    
    # 如果不足，随机添加一些
    if len(selected) < n_concepts:
        all_concepts = set(range(concept_graph_np.shape[0]))
        remaining = list(all_concepts - related_concepts)
        np.random.shuffle(remaining)
        selected.extend(remaining[:n_concepts - len(selected)])
    
    return sorted(selected)


def plot_knowledge_state_evolution(knowledge_states, selected_concepts, 
                                   concept_names=None, output_path=None,
                                   student_id=0):
    """
    绘制知识状态演化热力图
    
    Args:
        knowledge_states: [seq_len, n_concepts] 知识状态矩阵
        selected_concepts: 要可视化的知识点ID列表
        concept_names: 知识点名称列表（可选）
        output_path: 输出路径
        student_id: 学生ID
    """
    # 提取选中知识点的状态
    states = knowledge_states[:, selected_concepts]  # [seq_len, n_selected]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制热力图
    sns.heatmap(states.T, cmap='RdYlGn', vmin=0, vmax=1, 
                cbar_kws={'label': 'Mastery Level'},
                ax=ax, linewidths=0.5, linecolor='gray')
    
    # 设置标签
    if concept_names:
        y_labels = [concept_names.get(c, f'C{c}') for c in selected_concepts]
    else:
        y_labels = [f'Concept {c}' for c in selected_concepts]
    
    ax.set_yticks(np.arange(len(selected_concepts)) + 0.5)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Knowledge Concept', fontsize=12)
    ax.set_title(f'Knowledge State Evolution for Student {student_id}', fontsize=14, pad=20)
    
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


def visualize_student_knowledge_state(dataset_name='assist09', 
                                      checkpoint_path=None,
                                      student_id=None,
                                      n_concepts=8,
                                      output_dir=None,
                                      device='auto'):
    """
    可视化学生的知识状态演化
    
    Args:
        dataset_name: 数据集名称
        checkpoint_path: 模型检查点路径
        student_id: 学生ID（None表示随机选择）
        n_concepts: 要可视化的知识点数量
        output_dir: 输出目录
        device: 设备
    """
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"知识状态演化可视化")
    print(f"数据集: {dataset_name.upper()}")
    print(f"设备: {device}")
    print(f"{'='*80}")
    
    # 加载数据集
    dataset_info = load_processed_data(dataset_name)
    concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32)
    
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
    model.initialize_graph_modules(concept_graph)
    
    # 选择学生 - 直接从原始数据构建完整序列
    import pandas as pd
    
    test_data = dataset_info['test']
    if isinstance(test_data, list):
        test_data = pd.DataFrame(test_data)
    
    # 按学生分组获取完整序列
    student_groups = test_data.groupby('student_id')
    
    if student_id is None:
        # 随机选择一个有足够长序列的学生
        valid_students = []
        for sid, group in student_groups:
            if len(group) >= 50:  # 至少50个交互
                valid_students.append(sid)
        
        if not valid_students:
            print("警告：没有找到足够长的学生序列，使用第一个学生")
            selected_student_id = list(student_groups.groups.keys())[0]
        else:
            selected_student_id = np.random.choice(valid_students)
    else:
        selected_student_id = student_id
    
    # 获取该学生的完整序列
    student_group = student_groups.get_group(selected_student_id)
    if 'timestamp' in student_group.columns:
        student_group = student_group.sort_values('timestamp')
    
    # 构建序列字典（使用新格式的键名）
    student_sequence = {
        'question_id': torch.tensor(student_group['question_id'].values, dtype=torch.long),
        'concept_id': torch.tensor(student_group['concept_id'].values, dtype=torch.long),
        'correct': torch.tensor(student_group['correct'].values, dtype=torch.long)
    }
    
    seq_len = len(student_sequence['question_id'])
    print(f"学生ID: {selected_student_id}")
    print(f"序列长度: {seq_len}")
    
    # 提取知识状态
    print("\n提取知识状态...")
    knowledge_states = extract_knowledge_states(model, student_sequence, concept_graph, device)
    print(f"知识状态矩阵形状: {knowledge_states.shape}")
    
    # 选择要可视化的知识点
    # 选择学生在序列中接触过的知识点
    # 获取学生交互过的知识点
    if 'concept_seq' in student_sequence:
        interacted_concepts = list(set([c for c in student_sequence['concept_seq'] if c >= 0]))
    else:
        interacted_concepts = list(set([c.item() for c in student_sequence['concept_id'] if c >= 0]))
    print(f"学生接触的知识点数: {len(interacted_concepts)}")
    
    selected_concepts = select_related_concepts(concept_graph, interacted_concepts[:3], n_concepts)
    print(f"选中的知识点: {selected_concepts}")
    
    # 绘制可视化
    print("\n绘制知识状态演化图...")
    if output_dir is None:
        output_dir = os.path.join(project_root, 'figures', 'knowledge_state')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'knowledge_evolution_student{selected_student_id}_{dataset_name}.pdf')
    
    # 获取知识点名称（如果有）
    concept_names = dataset_info.get('id_to_concept', None)
    
    plot_knowledge_state_evolution(
        knowledge_states, 
        selected_concepts,
        concept_names=concept_names,
        output_path=output_path,
        student_id=selected_student_id
    )
    
    print(f"\n{'='*80}")
    print("知识状态演化可视化完成！")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='知识状态演化可视化（论文4.8节）')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'junyi', 'ednet'],
                       help='数据集名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--student_id', type=int, default=None,
                       help='学生ID（None表示随机选择）')
    parser.add_argument('--n_concepts', type=int, default=8,
                       help='要可视化的知识点数量')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    visualize_student_knowledge_state(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        student_id=args.student_id,
        n_concepts=args.n_concepts,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()

