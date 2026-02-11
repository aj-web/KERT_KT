"""
ASSIST09数据集下载和预处理脚本（优化版）

使用方法：
    python data/process_assist09_v2.py

修改记录：
    2026-02-11: 图构建方法从Q-matrix共现改为学生作答转移关系
                解决原Q-matrix方法因ASSIST09每题仅标注1个skill导致图极度稀疏的问题
                （原方法: 平均1.06个邻居 → 新方法: 目标3-8个邻居）
    2026-02-11: 优化版 - 修复自环处理、添加时间窗口、改进归一化
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def download_assist09():
    """下载ASSIST09数据集"""
    print("="*50)
    print("ASSIST09数据集下载")
    print("="*50)
    
    # 使用EduData库下载
    try:
        from EduData import get_data
        print("使用EduData库下载ASSIST09...")
        get_data("assistment-2009-2010-skill", data_dir="data/raw/")
        print("[OK] 下载完成")
        return True
    except ImportError:
        print("[ERROR] EduData库未安装")
        print("请运行: pip install EduData")
        return False
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        return False


def build_concept_graph_qmatrix(df, n_concepts):
    """
    [旧方法] 使用Q-matrix共现方法构建概念图
    
    问题：ASSIST09每道题只标注1个skill，几乎没有共现，导致图极度稀疏
    保留此函数作为备用/对比
    
    Args:
        df: 数据DataFrame
        n_concepts: 知识点总数
    
    Returns:
        concept_graph: [n_concepts, n_concepts]
    """
    print("  构建概念图（Q-matrix方法）...")
    
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    
    # 获取每个题目的知识点集合
    question_concepts = df.groupby('question_id')['concept_id'].apply(set).to_dict()
    
    # 统计知识点共现次数
    for q_id, concept_set in question_concepts.items():
        concepts = list(concept_set)
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                c1, c2 = concepts[i], concepts[j]
                concept_graph[c1, c2] += 1
                concept_graph[c2, c1] += 1
    
    # 归一化
    row_sums = concept_graph.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    concept_graph = concept_graph / row_sums
    
    print(f"    概念图密度: {(concept_graph > 0).sum() / (n_concepts * n_concepts):.4f}")
    
    return concept_graph


def build_concept_graph_transition(df, n_concepts, 
                                   min_transition_count=3, 
                                   top_k_neighbors=10,
                                   time_window=None):
    """
    [新方法] 使用学生作答序列的转移关系构建概念图（优化版）
    
    原理：
    - 如果学生在练习中先做了skill A的题，紧接着做了skill B的题，
      说明A→B存在学习路径上的关联关系
    - 统计所有学生的转移频率，归一化后作为边权重
    - 这与GKT(Nakagawa等,2019)使用的图构建方法一致，
      是知识追踪领域构建知识点关系图的主流方法
    
    优势：
    - 不依赖题目标注多个skill（ASSIST09每题只标1个skill）
    - 从33万+条交互记录中挖掘丰富的转移关系
    - 产生的图密度远高于Q-matrix共现方法
    
    优化点：
    - 修复：自环处理一致性（统计时排除，归一化后添加）
    - 新增：时间窗口过滤（可选，避免跨天的虚假转移）
    - 改进：归一化前添加自环，确保每个节点至少有自己
    
    Args:
        df: 数据DataFrame，需包含 student_id, concept_id 列，且已按时间排序
        n_concepts: 知识点总数
        min_transition_count: 最小转移次数阈值，低于此值的边被过滤（去噪）
        top_k_neighbors: 每个知识点最多保留的邻居数（防止过度连接）
        time_window: 时间窗口（秒），只统计此时间内的转移（None表示不限制）
    
    Returns:
        concept_graph: [n_concepts, n_concepts] 归一化后的概念图邻接矩阵
    """
    print("  构建概念图（学生作答转移关系方法 - 优化版）...")
    print(f"    参数: min_transition_count={min_transition_count}, "
          f"top_k_neighbors={top_k_neighbors}, time_window={time_window}")
    
    # Step 1: 统计转移频率（排除自环）
    transition_count = np.zeros((n_concepts, n_concepts), dtype=np.float64)
    
    # 按学生分组，统计相邻做题的skill转移
    total_transitions = 0
    filtered_by_time = 0
    self_loops = 0
    
    for student_id, group in df.groupby('student_id'):
        group = group.sort_values('order_id') if 'order_id' in group.columns else group
        concept_seq = group['concept_id'].values
        
        # 如果有时间戳，也获取时间序列
        if time_window is not None and 'timestamp' in group.columns:
            time_seq = group['timestamp'].values
        else:
            time_seq = None
        
        for i in range(len(concept_seq) - 1):
            c_from = concept_seq[i]
            c_to = concept_seq[i + 1]
            
            # 检查时间窗口（如果启用）
            if time_seq is not None:
                time_diff = time_seq[i + 1] - time_seq[i]
                if time_diff > time_window:
                    filtered_by_time += 1
                    continue
            
            # 排除自环（同一个skill连续做多题）
            if c_from == c_to:
                self_loops += 1
                continue
            
            transition_count[c_from, c_to] += 1
            total_transitions += 1
    
    print(f"    总转移次数: {total_transitions}")
    print(f"    排除的自环: {self_loops}")
    if time_window is not None:
        print(f"    时间窗口过滤: {filtered_by_time}")
    print(f"    非零转移对数: {(transition_count > 0).sum()}")
    
    # Step 2: 对称化（无向图：A→B 和 B→A 合并）
    transition_symmetric = transition_count + transition_count.T
    
    # Step 3: 过滤低频噪声边
    transition_filtered = transition_symmetric.copy()
    transition_filtered[transition_filtered < min_transition_count] = 0
    
    edges_before = (transition_symmetric > 0).sum()
    edges_after = (transition_filtered > 0).sum()
    print(f"    过滤前边数: {edges_before}, 过滤后边数: {edges_after} "
          f"(保留 {edges_after/max(edges_before,1)*100:.1f}%)")
    
    # Step 4: 每个知识点只保留top-k个最强连接的邻居
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    
    for i in range(n_concepts):
        row = transition_filtered[i].copy()
        
        if row.sum() == 0:
            continue
        
        # 找top-k邻居（不包括自己）
        nonzero_indices = np.where(row > 0)[0]
        if len(nonzero_indices) > top_k_neighbors:
            # 只保留权重最大的top_k个
            top_indices = np.argsort(row)[-top_k_neighbors:]
            mask = np.zeros_like(row, dtype=bool)
            mask[top_indices] = True
            row = row * mask
        
        concept_graph[i] = row
    
    # 再次对称化（top-k可能破坏对称性）
    concept_graph = np.maximum(concept_graph, concept_graph.T)
    
    # Step 5: 添加自环（在归一化之前）
    # 自环权重设为每个节点所有出边权重的平均值（或固定值1.0）
    for i in range(n_concepts):
        if concept_graph[i].sum() > 0:
            # 自环权重 = 平均出边权重
            concept_graph[i, i] = concept_graph[i, concept_graph[i] > 0].mean()
        else:
            # 孤立节点：自环权重 = 1.0
            concept_graph[i, i] = 1.0
    
    # Step 6: 行归一化（包含自环）
    row_sums = concept_graph.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 防止除零（理论上不会发生，因为有自环）
    concept_graph = concept_graph / row_sums
    
    # 打印统计信息
    density = (concept_graph > 0).sum() / (n_concepts * n_concepts)
    neighbor_counts = (concept_graph > 0).sum(axis=1) - 1  # 减去自环
    avg_neighbors = neighbor_counts.mean()
    isolated = (neighbor_counts == 0).sum()
    
    print(f"    概念图统计:")
    print(f"      密度: {density:.4f} (包含自环)")
    print(f"      平均邻居数: {avg_neighbors:.2f} (不含自环)")
    print(f"      最大邻居数: {neighbor_counts.max()}")
    print(f"      最小邻居数: {neighbor_counts.min()}")
    print(f"      中位数邻居数: {np.median(neighbor_counts):.1f}")
    print(f"      孤立节点数: {isolated} ({isolated/n_concepts*100:.1f}%)")
    print(f"      邻居数分布: "
          f"0个={np.sum(neighbor_counts==0)}, "
          f"1-3个={np.sum((neighbor_counts>=1)&(neighbor_counts<=3))}, "
          f"4-6个={np.sum((neighbor_counts>=4)&(neighbor_counts<=6))}, "
          f"7-10个={np.sum((neighbor_counts>=7)&(neighbor_counts<=10))}, "
          f">10个={np.sum(neighbor_counts>10)}")
    
    # 验证归一化
    row_sums_check = concept_graph.sum(axis=1)
    if not np.allclose(row_sums_check, 1.0):
        print(f"      [WARNING] 归一化检查失败: min={row_sums_check.min():.4f}, "
              f"max={row_sums_check.max():.4f}")
    
    return concept_graph


def process_assist09():
    """处理ASSIST09数据集"""
    print("\n" + "="*50)
    print("ASSIST09数据集预处理")
    print("="*50)
    
    # 数据路径（使用绝对路径，EduData下载后的实际路径）
    data_path = os.path.join(project_root, "data", "raw", "2009_skill_builder_data_corrected", "skill_builder_data_corrected.csv")
    
    # 如果新路径不存在，尝试旧路径
    if not os.path.exists(data_path):
        data_path_old = os.path.join(project_root, "data", "raw", "assistment-2009-2010-skill", "skill_builder_data.csv")
        if os.path.exists(data_path_old):
            data_path = data_path_old
        else:
            print(f"[ERROR] 数据文件不存在")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"项目根目录: {project_root}")
            print("请先运行下载...")
            if not download_assist09():
                return False
            # 下载后再次检查
            if not os.path.exists(data_path):
                print(f"[ERROR] 下载后仍未找到数据文件: {data_path}")
                return False
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    df = pd.read_csv(data_path, encoding='latin1')
    print(f"  原始记录数: {len(df)}")
    
    # 2. 数据清洗
    print("\n2. 数据清洗...")
    
    # 移除缺失值
    df = df.dropna(subset=['user_id', 'problem_id', 'skill_id', 'correct'])
    print(f"  移除缺失值后: {len(df)}")
    
    # 过滤异常作答时间（论文要求：1秒-1小时）
    if 'ms_first_response' in df.columns:
        df = df[(df['ms_first_response'] >= 1000) & (df['ms_first_response'] <= 3600000)]
        print(f"  过滤异常时间后: {len(df)}")
    
    # 按时间排序（对转移关系方法至关重要）
    if 'order_id' in df.columns:
        df = df.sort_values(['user_id', 'order_id'])
        print(f"  已按学生和时间排序")
    
    # 3. ID重编码
    print("\n3. ID重编码...")
    df['student_id'], _ = pd.factorize(df['user_id'])
    df['question_id'], _ = pd.factorize(df['problem_id'])
    df['concept_id'], _ = pd.factorize(df['skill_id'])
    
    n_students = df['student_id'].nunique()
    n_questions = df['question_id'].nunique()
    n_concepts = df['concept_id'].nunique()
    
    print(f"  学生数: {n_students}")
    print(f"  题目数: {n_questions}")
    print(f"  知识点数: {n_concepts}")
    
    # 验证每题标注的skill数量（解释为什么Q-matrix方法失效）
    skills_per_question = df.groupby('question_id')['concept_id'].nunique()
    single_skill_ratio = (skills_per_question == 1).mean()
    print(f"\n  [诊断] 每道题标注的skill数量:")
    print(f"    1个skill的题目: {(skills_per_question == 1).sum()} "
          f"({single_skill_ratio*100:.1f}%)")
    print(f"    2+个skill的题目: {(skills_per_question >= 2).sum()} "
          f"({(skills_per_question >= 2).mean()*100:.1f}%)")
    if single_skill_ratio > 0.9:
        print(f"    → Q-matrix共现方法几乎无法建边，必须改用转移关系方法")
    
    # 4. 构建Q矩阵
    print("\n4. 构建Q矩阵...")
    q_matrix = np.zeros((n_questions, n_concepts), dtype=np.float32)
    for _, row in df.iterrows():
        q_matrix[int(row['question_id']), int(row['concept_id'])] = 1.0
    print(f"  Q矩阵形状: {q_matrix.shape}")
    
    # 5. 构建概念图（核心修改：使用转移关系方法）
    print("\n5. 构建概念图...")
    
    # 新方法：基于学生作答序列的转移关系（优化版）
    concept_graph = build_concept_graph_transition(
        df, n_concepts,
        min_transition_count=3,   # 至少3次转移才保留（去噪）
        top_k_neighbors=10,       # 每个知识点最多10个邻居（防止过连接）
        time_window=None          # 不限制时间窗口（可选：86400秒=1天）
    )
    
    # 同时用旧方法构建，作为对比（不保存，仅打印）
    print("\n  [对比] 旧Q-matrix方法:")
    concept_graph_old = build_concept_graph_qmatrix(df, n_concepts)
    old_density = (concept_graph_old > 0).sum() / (n_concepts * n_concepts)
    old_avg_neighbors = (concept_graph_old > 0).sum(axis=1).mean()
    print(f"    旧方法 - 密度: {old_density:.4f}, 平均邻居数: {old_avg_neighbors:.2f}")
    new_density = (concept_graph > 0).sum() / (n_concepts * n_concepts)
    new_avg_neighbors = ((concept_graph > 0).sum(axis=1) - 1).mean()  # 减去自环
    print(f"    新方法 - 密度: {new_density:.4f}, 平均邻居数: {new_avg_neighbors:.2f}")
    improvement_density = new_density / max(old_density, 1e-10)
    improvement_neighbors = new_avg_neighbors / max(old_avg_neighbors, 1e-10)
    print(f"    改善倍数: 密度 {improvement_density:.1f}x, 邻居数 {improvement_neighbors:.1f}x")
    
    # 6. 构建交互序列
    print("\n6. 构建交互序列...")
    interactions = []
    for _, row in df.iterrows():
        interactions.append({
            'student_id': int(row['student_id']),
            'question_id': int(row['question_id']),
            'concept_id': int(row['concept_id']),
            'correct': int(row['correct'])
        })
    print(f"  交互记录数: {len(interactions)}")
    
    # 7. 数据划分（按学生）
    print("\n7. 数据划分...")
    student_interactions = defaultdict(list)
    for interaction in interactions:
        student_interactions[interaction['student_id']].append(interaction)
    
    student_ids = list(student_interactions.keys())
    np.random.seed(42)
    np.random.shuffle(student_ids)
    
    n_train = int(len(student_ids) * 0.7)
    n_val = int(len(student_ids) * 0.15)
    
    train_students = student_ids[:n_train]
    val_students = student_ids[n_train:n_train+n_val]
    test_students = student_ids[n_train+n_val:]
    
    train_data = []
    val_data = []
    test_data = []
    
    for sid in train_students:
        train_data.extend(student_interactions[sid])
    for sid in val_students:
        val_data.extend(student_interactions[sid])
    for sid in test_students:
        test_data.extend(student_interactions[sid])
    
    print(f"  训练集: {len(train_data)} ({len(train_students)} 学生)")
    print(f"  验证集: {len(val_data)} ({len(val_students)} 学生)")
    print(f"  测试集: {len(test_data)} ({len(test_students)} 学生)")
    
    # 8. 保存数据
    print("\n8. 保存数据...")
    # 使用绝对路径，避免路径重复问题
    output_file = os.path.join(project_root, "data", "assist09_processed.pkl")
    
    processed_data = {
        'assist09': {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'q_matrix': q_matrix,
            'concept_graph': concept_graph,
            'n_students': n_students,
            'n_questions': n_questions,
            'n_concepts': n_concepts
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"  [OK] 数据已保存: {output_file}")
    
    print("\n" + "="*50)
    print("[OK] ASSIST09预处理完成")
    print("="*50)
    print("\n[提示] 下一步:")
    print("  1. 运行 python data/merge_datasets.py 合并数据集")
    print("  2. 运行 python experiments/run_experiment.py --dataset assist09 --n_runs 5")
    print("  3. 对比新旧图方法的Phase 2 RL效果")
    
    return True


def main():
    """主函数"""
    success = process_assist09()
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

