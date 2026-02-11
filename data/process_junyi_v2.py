"""
Junyi数据集预处理脚本（优化版 - 混合图构建）

使用方法：
    python data/process_junyi_v2.py

修改记录：
    2026-02-11: 图构建方法改为"先修关系 + 转移关系"混合方法
                解决原先修关系方法导致70.7%节点只有自环的问题
                （原方法: 平均0.74个邻居 → 新方法: 目标5-10个邻居）
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


def download_junyi():
    """下载Junyi数据集"""
    print("="*50)
    print("Junyi数据集下载")
    print("="*50)
    
    # 使用EduData库下载
    try:
        from EduData import get_data
        print("使用EduData库下载Junyi...")
        get_data("junyi", data_dir=os.path.join(project_root, "data", "raw"))
        print("[OK] 下载完成")
        return True
    except ImportError:
        print("[ERROR] EduData库未安装")
        print("请运行: pip install EduData")
        return False
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("\n[备选方案] 手动下载:")
        print("1. 访问: https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset")
        print("2. 下载 junyi_ProblemLog_original.csv 和 relationship_annotation_training.csv")
        print(f"3. 放到目录: {os.path.join(project_root, 'data', 'junyi')}")
        return False


def build_concept_graph_prerequisite(n_concepts, prerequisite_file, concept_to_id):
    """
    使用先修关系构建概念图（Junyi特有，领域知识）
    
    Args:
        n_concepts: 知识点总数
        prerequisite_file: 先修关系文件路径
        concept_to_id: 概念名到ID的映射字典
    
    Returns:
        concept_graph: [n_concepts, n_concepts] 有向图
    """
    print("  [方法1] 先修关系图（领域知识）...")
    
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    
    if prerequisite_file and os.path.exists(prerequisite_file):
        prereq_df = pd.read_csv(prerequisite_file)
        print(f"    读取到 {len(prereq_df)} 对先修关系")
        
        edge_count = 0
        for _, row in prereq_df.iterrows():
            exercise_a = row['Exercise_A']
            exercise_b = row['Exercise_B']
            prereq_score = row['Prerequisite_avg']
            
            if prereq_score > 5:
                if exercise_a in concept_to_id and exercise_b in concept_to_id:
                    id_a = concept_to_id[exercise_a]
                    id_b = concept_to_id[exercise_b]
                    # A是B的先修，A→B有边（有向）
                    concept_graph[id_a, id_b] = prereq_score / 10.0
                    edge_count += 1
        
        print(f"    成功添加 {edge_count} 条有向边")
    else:
        print("    [WARNING] 先修关系文件不存在")
    
    # 统计
    non_zero = (concept_graph > 0).sum()
    avg_out_degree = (concept_graph > 0).sum(axis=1).mean()
    print(f"    非零边数: {non_zero}, 平均出度: {avg_out_degree:.2f}")
    
    return concept_graph


def build_concept_graph_transition(df, n_concepts, 
                                   min_transition_count=5,
                                   top_k_neighbors=15):
    """
    使用学生作答序列的转移关系构建概念图（数据驱动）
    
    原理：统计学生学习路径中的知识点转移频率
    
    Args:
        df: 数据DataFrame
        n_concepts: 知识点总数
        min_transition_count: 最小转移次数阈值
        top_k_neighbors: 每个知识点最多保留的邻居数
    
    Returns:
        concept_graph: [n_concepts, n_concepts] 无向图（对称）
    """
    print("  [方法2] 转移关系图（数据驱动）...")
    print(f"    参数: min_transition={min_transition_count}, top_k={top_k_neighbors}")
    
    # Step 1: 统计转移频率
    transition_count = np.zeros((n_concepts, n_concepts), dtype=np.float64)
    
    total_transitions = 0
    self_loops = 0
    
    for student_id, group in df.groupby('user_id'):
        # 按时间排序
        group = group.sort_values('timestamp')
        concept_seq = group['exercise_id'].values  # Junyi中exercise_id就是concept_id
        
        for i in range(len(concept_seq) - 1):
            c_from = concept_seq[i]
            c_to = concept_seq[i + 1]
            
            if c_from == c_to:
                self_loops += 1
                continue
            
            transition_count[c_from, c_to] += 1
            total_transitions += 1
    
    print(f"    总转移次数: {total_transitions}")
    print(f"    排除的自环: {self_loops}")
    print(f"    非零转移对数: {(transition_count > 0).sum()}")
    
    # Step 2: 对称化（无向图）
    transition_symmetric = transition_count + transition_count.T
    
    # Step 3: 过滤低频边
    transition_filtered = transition_symmetric.copy()
    transition_filtered[transition_filtered < min_transition_count] = 0
    
    edges_before = (transition_symmetric > 0).sum()
    edges_after = (transition_filtered > 0).sum()
    print(f"    过滤前边数: {edges_before}, 过滤后边数: {edges_after} "
          f"(保留 {edges_after/max(edges_before,1)*100:.1f}%)")
    
    # Step 4: Top-K限制
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    
    for i in range(n_concepts):
        row = transition_filtered[i].copy()
        
        if row.sum() == 0:
            continue
        
        nonzero_indices = np.where(row > 0)[0]
        if len(nonzero_indices) > top_k_neighbors:
            top_indices = np.argsort(row)[-top_k_neighbors:]
            mask = np.zeros_like(row, dtype=bool)
            mask[top_indices] = True
            row = row * mask
        
        concept_graph[i] = row
    
    # 再次对称化
    concept_graph = np.maximum(concept_graph, concept_graph.T)
    
    # 统计
    non_zero = (concept_graph > 0).sum()
    avg_degree = (concept_graph > 0).sum(axis=1).mean()
    print(f"    非零边数: {non_zero}, 平均度数: {avg_degree:.2f}")
    
    return concept_graph


def build_concept_graph_hybrid(df, n_concepts, prerequisite_file, concept_to_id,
                               w_prereq=0.4, w_transition=0.6,
                               min_transition_count=5, top_k_neighbors=15):
    """
    混合方法：先修关系 + 转移关系
    
    策略：
    1. 先修关系提供确定性的课程结构（有向，稀疏，高质量）
    2. 转移关系补充数据驱动的关联（无向，稠密，覆盖广）
    3. 加权融合，取长补短
    
    Args:
        df: 数据DataFrame
        n_concepts: 知识点总数
        prerequisite_file: 先修关系文件路径
        concept_to_id: 概念名到ID的映射字典
        w_prereq: 先修关系权重
        w_transition: 转移关系权重
        min_transition_count: 转移关系最小次数阈值
        top_k_neighbors: 转移关系Top-K限制
    
    Returns:
        concept_graph: [n_concepts, n_concepts] 混合图
    """
    print("  构建概念图（混合方法：先修 + 转移）...")
    print(f"    权重: 先修={w_prereq}, 转移={w_transition}")
    
    # 方法1：先修关系（有向）
    graph_prereq = build_concept_graph_prerequisite(n_concepts, prerequisite_file, concept_to_id)
    
    # 方法2：转移关系（无向）
    graph_transition = build_concept_graph_transition(
        df, n_concepts,
        min_transition_count=min_transition_count,
        top_k_neighbors=top_k_neighbors
    )
    
    # 对称化先修关系图（A→B 和 B→A 都保留）
    graph_prereq_sym = np.maximum(graph_prereq, graph_prereq.T)
    
    # 归一化到相同尺度（0-1）
    if graph_prereq_sym.max() > 0:
        graph_prereq_norm = graph_prereq_sym / graph_prereq_sym.max()
    else:
        graph_prereq_norm = graph_prereq_sym
    
    if graph_transition.max() > 0:
        graph_transition_norm = graph_transition / graph_transition.max()
    else:
        graph_transition_norm = graph_transition
    
    # 加权融合
    graph_hybrid = w_prereq * graph_prereq_norm + w_transition * graph_transition_norm
    
    # 添加自环（在归一化之前）
    for i in range(n_concepts):
        if graph_hybrid[i].sum() > 0:
            graph_hybrid[i, i] = graph_hybrid[i, graph_hybrid[i] > 0].mean()
        else:
            graph_hybrid[i, i] = 1.0
    
    # 行归一化
    row_sums = graph_hybrid.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    graph_hybrid = graph_hybrid / row_sums
    
    # 统计
    print("\n  [混合图统计]:")
    density = (graph_hybrid > 0).sum() / (n_concepts * n_concepts)
    neighbor_counts = (graph_hybrid > 0).sum(axis=1) - 1  # 减去自环
    avg_neighbors = neighbor_counts.mean()
    isolated = (neighbor_counts == 0).sum()
    
    print(f"    密度: {density:.4f} (包含自环)")
    print(f"    平均邻居数: {avg_neighbors:.2f} (不含自环)")
    print(f"    最大邻居数: {neighbor_counts.max()}")
    print(f"    最小邻居数: {neighbor_counts.min()}")
    print(f"    中位数邻居数: {np.median(neighbor_counts):.1f}")
    print(f"    孤立节点数: {isolated} ({isolated/n_concepts*100:.1f}%)")
    print(f"    邻居数分布: "
          f"0个={np.sum(neighbor_counts==0)}, "
          f"1-3个={np.sum((neighbor_counts>=1)&(neighbor_counts<=3))}, "
          f"4-6个={np.sum((neighbor_counts>=4)&(neighbor_counts<=6))}, "
          f"7-10个={np.sum((neighbor_counts>=7)&(neighbor_counts<=10))}, "
          f">10个={np.sum(neighbor_counts>10)}")
    
    # 验证归一化
    row_sums_check = graph_hybrid.sum(axis=1)
    if not np.allclose(row_sums_check, 1.0):
        print(f"    [WARNING] 归一化检查失败: "
              f"min={row_sums_check.min():.4f}, max={row_sums_check.max():.4f}")
    
    # 对比原先修关系方法
    print("\n  [对比] 原先修关系方法:")
    graph_old = graph_prereq_sym.copy()
    for i in range(n_concepts):
        graph_old[i, i] = 1.0
    row_sums_old = graph_old.sum(axis=1, keepdims=True)
    graph_old = graph_old / row_sums_old
    
    old_neighbors = ((graph_old > 0).sum(axis=1) - 1).mean()
    new_neighbors = avg_neighbors
    print(f"    旧方法平均邻居数: {old_neighbors:.2f}")
    print(f"    新方法平均邻居数: {new_neighbors:.2f}")
    print(f"    改善倍数: {new_neighbors/max(old_neighbors, 0.1):.1f}x")
    
    return graph_hybrid


def process_junyi():
    """处理Junyi数据集"""
    print("\n" + "="*50)
    print("Junyi数据集预处理（混合图方法）")
    print("="*50)
    
    # 数据路径（自动检测，支持多种可能的位置）
    possible_data_dirs = [
        os.path.join(project_root, "data", "junyi"),  # 标准位置
        os.path.join(project_root, "data", "raw", "junyi"),  # EduData下载位置
    ]
    
    log_file = None
    prerequisite_file = None
    data_dir = None
    
    # 尝试找到数据文件
    for dir_path in possible_data_dirs:
        test_log = os.path.join(dir_path, "junyi_ProblemLog_original.csv")
        test_prereq = os.path.join(dir_path, "relationship_annotation_training.csv")
        if os.path.exists(test_log):
            log_file = test_log
            prerequisite_file = test_prereq
            data_dir = dir_path
            print(f"[OK] 找到数据文件: {data_dir}")
            break
    
    if log_file is None or not os.path.exists(log_file):
        print(f"[ERROR] 数据文件不存在")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"项目根目录: {project_root}")
        print(f"已检查的位置:")
        for dir_path in possible_data_dirs:
            print(f"  - {dir_path}")
        print("\n正在尝试下载...")
        if not download_junyi():
            return False
        
        # 下载后再次检查
        for dir_path in possible_data_dirs:
            test_log = os.path.join(dir_path, "junyi_ProblemLog_original.csv")
            if os.path.exists(test_log):
                log_file = test_log
                prerequisite_file = os.path.join(dir_path, "relationship_annotation_training.csv")
                data_dir = dir_path
                break
        
        if log_file is None or not os.path.exists(log_file):
            print(f"[ERROR] 下载后仍未找到数据文件")
            return False
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    print(f"  [WARNING] Junyi数据集约2500万条记录，采用采样策略...")
    print(f"  采样100万条数据（与论文实验规模一致）...")
    
    df = pd.read_csv(log_file, nrows=1050000)  # 多读一些，清洗后约100万
    print(f"  采样后记录数: {len(df):,}")
    
    # 2. 数据清洗
    print("\n2. 数据清洗...")
    
    # 自动检测列名（不同版本的Junyi数据集列名可能不同）
    print(f"  数据集列名: {df.columns.tolist()}")
    
    # 检测正确率列名
    correct_col = None
    for col in ['is_correct', 'correct', 'is_correct_first_attempt']:
        if col in df.columns:
            correct_col = col
            break
    
    if correct_col is None:
        print(f"[ERROR] 未找到正确率列，可用列: {df.columns.tolist()}")
        return False
    
    # 检测时间戳列名
    timestamp_col = None
    for col in ['timestamp_TW', 'timestamp', 'time_done', 'time_done_timestamp']:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print(f"[ERROR] 未找到时间戳列，可用列: {df.columns.tolist()}")
        return False
    
    print(f"  使用列: user_id, exercise, {correct_col}, {timestamp_col}")
    
    # 移除缺失值
    df = df.dropna(subset=['user_id', 'exercise', correct_col, timestamp_col])
    print(f"  移除缺失值后: {len(df)}")
    
    # 统一列名
    df = df.rename(columns={correct_col: 'is_correct', timestamp_col: 'timestamp_orig'})
    
    # 过滤异常时间
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp_orig']).astype(np.int64) // 10**9
    except:
        # 如果已经是时间戳格式
        df['timestamp'] = df['timestamp_orig'].astype(np.int64)
    
    df = df[(df['timestamp'] > 0)]
    print(f"  过滤异常时间后: {len(df)}")
    
    # 3. 创建ID
    print("\n3. 创建ID...")
    df['user_id_orig'] = df['user_id']
    df['exercise_orig'] = df['exercise']
    
    # 4. ID重编码
    print("\n4. ID重编码...")
    df['user_id'], _ = pd.factorize(df['user_id'])
    df['exercise_id'], exercise_names = pd.factorize(df['exercise'])
    
    # 创建concept_to_id映射（Junyi中exercise就是concept）
    concept_to_id = {name: idx for idx, name in enumerate(exercise_names)}
    
    n_students = df['user_id'].nunique()
    n_questions = df['exercise_id'].nunique()
    n_concepts = n_questions  # Junyi中exercise=concept
    
    print(f"  学生数: {n_students}")
    print(f"  题目数: {n_questions}")
    print(f"  知识点数: {n_concepts}")
    
    # 5. 构建Q矩阵（Junyi中是单位矩阵，因为exercise=concept）
    print("\n5. 构建Q矩阵...")
    q_matrix = np.eye(n_concepts, dtype=np.float32)
    print(f"  Q矩阵形状: {q_matrix.shape} (单位矩阵)")
    
    # 6. 构建概念图（混合方法）
    print("\n6. 构建概念图...")
    concept_graph = build_concept_graph_hybrid(
        df, n_concepts,
        prerequisite_file=prerequisite_file,
        concept_to_id=concept_to_id,
        w_prereq=0.4,              # 先修关系权重40%
        w_transition=0.6,          # 转移关系权重60%
        min_transition_count=5,    # 至少5次转移
        top_k_neighbors=8          # 每个知识点最多8个邻居（降低以加速训练）
    )
    
    # 7. 构建交互序列
    print("\n7. 构建交互序列...")
    interactions = []
    for _, row in df.iterrows():
        interactions.append({
            'student_id': int(row['user_id']),
            'question_id': int(row['exercise_id']),
            'concept_id': int(row['exercise_id']),
            'correct': int(row['is_correct'])
        })
    print(f"  交互记录数: {len(interactions)}")
    
    # 8. 数据划分
    print("\n8. 数据划分...")
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
    
    # 9. 保存数据
    print("\n9. 保存数据...")
    output_file = os.path.join(project_root, "data", "junyi_processed.pkl")
    
    processed_data = {
        'junyi': {
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
    print("[OK] Junyi预处理完成")
    print("="*50)
    print("\n[提示] 下一步:")
    print("  1. 运行 python data/merge_datasets.py 合并数据集")
    print("  2. 运行 python experiments/run_experiment.py --dataset junyi --n_runs 3")
    
    return True


def main():
    """主函数"""
    success = process_junyi()
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

