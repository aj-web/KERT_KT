"""
Junyi数据集下载和预处理脚本

使用方法：
    python data/process_junyi.py
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
        get_data("junyi", data_dir="data/raw/")
        print("[OK] 下载完成")
        return True
    except ImportError:
        print("[ERROR] EduData库未安装")
        print("请运行: pip install EduData")
        return False
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("\n备选方案：手动下载")
        print("1. 访问: https://pslcdatashop.web.cmu.edu/")
        print("2. 搜索 'Junyi Academy'")
        print("3. 下载数据文件")
        return False


def build_concept_graph_from_prerequisite(n_concepts, prerequisite_file, concept_to_id):
    """
    使用先修关系构建概念图（Junyi特有）
    
    Args:
        n_concepts: 知识点总数
        prerequisite_file: 先修关系文件路径
        concept_to_id: 概念名到ID的映射字典
    
    Returns:
        concept_graph: [n_concepts, n_concepts]
    """
    print("  构建概念图（先修关系方法）...")
    
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    
    if prerequisite_file and os.path.exists(prerequisite_file):
        # 读取先修关系
        prereq_df = pd.read_csv(prerequisite_file)
        print(f"    读取到 {len(prereq_df)} 对先修关系")
        
        # Junyi先修关系格式：Exercise_A → Exercise_B
        # Prerequisite_avg > 5 表示A是B的先修
        edge_count = 0
        for _, row in prereq_df.iterrows():
            exercise_a = row['Exercise_A']
            exercise_b = row['Exercise_B']
            prereq_score = row['Prerequisite_avg']
            
            # 只保留明确的先修关系（得分>5）
            if prereq_score > 5:
                # 将exercise名称映射到concept_id
                if exercise_a in concept_to_id and exercise_b in concept_to_id:
                    id_a = concept_to_id[exercise_a]
                    id_b = concept_to_id[exercise_b]
                    # A是B的先修，则A→B有边
                    concept_graph[id_a, id_b] = prereq_score / 10.0  # 归一化到[0,1]
                    edge_count += 1
        
        print(f"    成功添加 {edge_count} 条边")
    else:
        print("    [WARNING] 先修关系文件不存在")
    
    # 添加自环
    for i in range(n_concepts):
        concept_graph[i, i] = 1.0
    
    # 归一化（行和为1）
    row_sums = concept_graph.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    concept_graph = concept_graph / row_sums
    
    print(f"    概念图密度: {(concept_graph > 0).sum() / (n_concepts * n_concepts):.4f}")
    
    return concept_graph


def build_concept_graph_qmatrix(df, n_concepts):
    """
    使用Q-matrix方法构建概念图（备用方法）
    
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


def process_junyi():
    """处理Junyi数据集"""
    print("\n" + "="*50)
    print("Junyi数据集预处理")
    print("="*50)
    
    # 数据路径（多个可能的位置）
    possible_paths = [
        "data/raw/junyi/junyi_ProblemLog_original.csv",  # EduData标准路径
        "junyi/junyi_ProblemLog_original.csv",           # 项目根目录
        "data/raw/junyi/junyi_ProblemLog_for_PSLC.txt",  # 备用格式
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            print(f"[OK] 找到数据文件: {path}")
            break
    
    if data_path is None:
        print(f"[ERROR] 数据文件不存在，尝试的路径：")
        for path in possible_paths:
            print(f"  - {path}")
        
        print("\n尝试下载...")
        if not download_junyi():
            print("\n[WARNING] 自动下载失败，使用现有数据（如果有）")
            # 再次检查所有路径
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    print(f"[OK] 找到数据: {path}")
                    break
        else:
            # 下载成功后再检查
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    print(f"[OK] 找到数据: {path}")
                    break
        
        if data_path is None:
            print("\n[ERROR] 无法找到Junyi数据文件")
            print("\n手动解决方案：")
            print("1. 检查 data/raw/junyi/ 目录")
            print("2. 或者使用项目根目录的 junyi/ 文件夹")
            print("3. 确保有 junyi_ProblemLog_original.csv 文件")
            return False
    
    # 1. 读取数据（Junyi数据集很大，采用采样策略）
    print("\n1. 读取数据...")
    print("  [WARNING] Junyi数据集约2500万条记录，采用采样策略加速处理...")
    print("  采样100万条数据（与论文实验规模一致）...")
    
    # 采样策略：每N行取1行（skiprows参数）
    # 2500万条采样到100万条，需要skip约96%的行
    # 使用lambda函数随机跳过行，保持数据分布
    np.random.seed(42)
    # 采样率约4%（100万/2500万）
    sample_rate = 0.04
    df = pd.read_csv(data_path, 
                     skiprows=lambda x: x > 0 and np.random.random() > sample_rate)
    
    print(f"  采样后记录数: {len(df):,}")
    
    # 2. 数据清洗
    print("\n2. 数据清洗...")
    
    # Junyi数据集的列名映射
    # user_id: 学生ID（保留）
    # exercise: 练习ID（作为知识点concept_id）
    # problem_number: 题目编号（与exercise组合成question_id）
    # correct: 正确性（保留）
    # time_done: 完成时间（用于排序）
    # time_taken: 作答时间（用于过滤）
    
    # 移除缺失值
    df = df.dropna(subset=['user_id', 'exercise', 'correct'])
    print(f"  移除缺失值后: {len(df)}")
    
    # 过滤异常作答时间（time_taken是秒数）
    if 'time_taken' in df.columns:
        # 过滤0秒和超过1小时的记录
        df = df[(df['time_taken'] > 0) & (df['time_taken'] <= 3600)]
        print(f"  过滤异常时间后: {len(df)}")
    
    # 按用户和时间排序
    if 'time_done' in df.columns:
        df = df.sort_values(['user_id', 'time_done'])
    
    # 3. 创建ID（Junyi特有的处理）
    print("\n3. 创建ID...")
    # concept_id: 使用exercise（练习ID）
    df['concept_id'] = df['exercise']
    # question_id: 组合exercise和problem_number
    df['question_id'] = df['exercise'].astype(str) + '_' + df['problem_number'].astype(str)
    
    # 4. ID重编码
    print("\n4. ID重编码...")
    df['student_id'], _ = pd.factorize(df['user_id'])
    df['question_id'], _ = pd.factorize(df['question_id'])
    # 保存concept名称到ID的映射
    df['concept_id'], concept_names = pd.factorize(df['concept_id'])
    concept_to_id = {name: idx for idx, name in enumerate(concept_names)}
    
    n_students = df['student_id'].nunique()
    n_questions = df['question_id'].nunique()
    n_concepts = df['concept_id'].nunique()
    
    print(f"  学生数: {n_students}")
    print(f"  题目数: {n_questions}")
    print(f"  知识点数: {n_concepts}")
    
    # 5. 构建Q矩阵
    print("\n5. 构建Q矩阵...")
    q_matrix = np.zeros((n_questions, n_concepts), dtype=np.float32)
    for _, row in df.iterrows():
        q_matrix[int(row['question_id']), int(row['concept_id'])] = 1.0
    print(f"  Q矩阵形状: {q_matrix.shape}")
    
    # 6. 构建概念图
    print("\n6. 构建概念图...")
    # 尝试使用先修关系（多个可能路径）
    prerequisite_paths = [
        "junyi/relationship_annotation_training.csv",  # 项目根目录
        "data/raw/junyi/relationship_annotation_training.csv"  # data目录
    ]
    
    prerequisite_file = None
    for path in prerequisite_paths:
        if os.path.exists(path):
            prerequisite_file = path
            print(f"  [OK] 找到先修关系文件: {path}")
            break
    
    if prerequisite_file:
        concept_graph = build_concept_graph_from_prerequisite(n_concepts, prerequisite_file, concept_to_id)
    else:
        print("  [WARNING] 未找到先修关系文件，使用Q-matrix方法")
        concept_graph = build_concept_graph_qmatrix(df, n_concepts)
    
    # 7. 构建交互序列
    print("\n7. 构建交互序列...")
    interactions = []
    for _, row in df.iterrows():
        interactions.append({
            'student_id': int(row['student_id']),
            'question_id': int(row['question_id']),
            'concept_id': int(row['concept_id']),
            'correct': int(row['correct'])
        })
    print(f"  交互记录数: {len(interactions)}")
    
    # 8. 数据划分（按学生）
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
    # 使用绝对路径，避免路径重复问题
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
    
    return True


def main():
    """主函数"""
    success = process_junyi()
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

