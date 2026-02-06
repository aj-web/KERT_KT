"""
EdNet数据集下载和预处理脚本

使用方法：
    python data/process_ednet.py
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import glob
from collections import defaultdict


def load_questions_and_tags(data_dir):
    """
    加载题目和知识点信息
    
    Args:
        data_dir: EdNet数据目录
    
    Returns:
        questions_df: 题目信息DataFrame
        tags_dict: 知识点ID到名称的映射
        q_matrix: 题目-知识点关联矩阵
    """
    print("加载题目和知识点信息...")
    
    # 加载题目信息
    questions_path = os.path.join(data_dir, 'questions.csv')
    questions_df = pd.read_csv(questions_path)
    
    # 解析tags字段（JSON格式的列表）
    import json
    questions_df['tags'] = questions_df['tags'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    
    # 构建Q矩阵
    all_tags = set()
    for tags in questions_df['tags']:
        all_tags.update(tags)
    
    tag_to_id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    n_questions = len(questions_df)
    n_concepts = len(tag_to_id)
    
    q_matrix = np.zeros((n_questions, n_concepts), dtype=np.float32)
    question_id_to_idx = {qid: idx for idx, qid in enumerate(questions_df['question_id'])}
    
    for idx, row in questions_df.iterrows():
        for tag in row['tags']:
            concept_idx = tag_to_id[tag]
            q_matrix[idx, concept_idx] = 1.0
    
    print(f"  题目数: {n_questions}")
    print(f"  知识点数: {n_concepts}")
    
    return questions_df, tag_to_id, q_matrix, question_id_to_idx


def load_student_interactions(data_dir, question_id_to_idx, max_students=None):
    """
    加载学生交互数据
    
    Args:
        data_dir: EdNet数据目录
        question_id_to_idx: 题目ID到索引的映射
        max_students: 最大学生数（用于调试）
    
    Returns:
        interactions: 学生交互列表
    """
    print("加载学生交互数据...")
    
    train_dir = os.path.join(data_dir, 'train')
    student_files = sorted(glob.glob(os.path.join(train_dir, 'u*.csv')))
    
    if max_students:
        student_files = student_files[:max_students]
    
    interactions = []
    student_id = 0
    
    for file_path in tqdm(student_files, desc="处理学生文件"):
        try:
            df = pd.read_csv(file_path)
            
            # 过滤：只保留question类型的交互
            df = df[df['question_id'].notna()].copy()
            
            # 过滤：移除异常作答时间（<1秒 或 >1小时）
            if 'elapsed_time' in df.columns:
                df = df[(df['elapsed_time'] >= 1000) & (df['elapsed_time'] <= 3600000)]
            
            # 过滤：序列长度至少5
            if len(df) < 5:
                continue
            
            # 按时间排序
            df = df.sort_values('timestamp')
            
            # 构建交互序列
            for _, row in df.iterrows():
                question_id = row['question_id']
                if question_id not in question_id_to_idx:
                    continue  # 跳过未知题目
                
                interactions.append({
                    'student_id': student_id,
                    'question_id': question_id_to_idx[question_id],
                    'correct': int(row['user_answer']),
                    'timestamp': row['timestamp']
                })
            
            student_id += 1
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    print(f"  学生数: {student_id}")
    print(f"  交互记录数: {len(interactions)}")
    
    return interactions, student_id


def build_concept_graph_from_qmatrix(q_matrix):
    """
    根据Q矩阵构建概念图（基于题目共现）
    
    Args:
        q_matrix: [n_questions, n_concepts] 题目-知识点关联矩阵
    
    Returns:
        concept_graph: [n_concepts, n_concepts] 概念相似度矩阵
    """
    print("构建概念图...")
    
    n_concepts = q_matrix.shape[1]
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    
    # 计算知识点共现次数
    # co_occurrence[i,j] = 包含知识点i和j的题目数量
    co_occurrence = q_matrix.T @ q_matrix  # [n_concepts, n_concepts]
    
    # 计算每个知识点出现的题目数
    concept_counts = q_matrix.sum(axis=0)  # [n_concepts]
    
    # 计算相似度：Jaccard系数
    for i in range(n_concepts):
        for j in range(n_concepts):
            if i == j:
                concept_graph[i, j] = 1.0
            else:
                union_count = concept_counts[i] + concept_counts[j] - co_occurrence[i, j]
                if union_count > 0:
                    concept_graph[i, j] = co_occurrence[i, j] / union_count
    
    print(f"  概念图密度: {(concept_graph > 0).sum() / (n_concepts * n_concepts):.4f}")
    
    return concept_graph


def split_data(interactions, train_ratio=0.7, val_ratio=0.15):
    """
    按学生维度划分数据集
    
    Args:
        interactions: 交互记录列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_data, val_data, test_data
    """
    print("划分数据集...")
    
    # 按学生分组
    student_interactions = defaultdict(list)
    for interaction in interactions:
        student_interactions[interaction['student_id']].append(interaction)
    
    # 随机打乱学生ID
    student_ids = list(student_interactions.keys())
    np.random.shuffle(student_ids)
    
    # 划分
    n_students = len(student_ids)
    n_train = int(n_students * train_ratio)
    n_val = int(n_students * val_ratio)
    
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
    
    print(f"  训练集: {len(train_data)} 条记录 ({len(train_students)} 学生)")
    print(f"  验证集: {len(val_data)} 条记录 ({len(val_students)} 学生)")
    print(f"  测试集: {len(test_data)} 条记录 ({len(test_students)} 学生)")
    
    return train_data, val_data, test_data


def process_ednet(data_dir, output_file, max_students=None):
    """
    处理EdNet数据集的主函数
    
    Args:
        data_dir: EdNet数据目录
        output_file: 输出文件路径
        max_students: 最大学生数（用于调试）
    """
    print("="*50)
    print("EdNet数据集预处理")
    print("="*50)
    
    # 检查数据是否存在
    questions_path = os.path.join(data_dir, 'questions.csv')
    if not os.path.exists(questions_path):
        print(f"\n[ERROR] EdNet数据不存在: {questions_path}")
        print("\nEdNet数据集需要手动下载:")
        print("1. 访问: https://github.com/riiid/ednet")
        print("2. 下载 EdNet-KT1 数据集")
        print("3. 解压到: data/raw/ednet/KT1/")
        print("\n或运行自动下载脚本（需要kaggle认证）:")
        print("  python data/download_ednet.py")
        return False
    
    # 1. 加载题目和知识点信息
    questions_df, tag_to_id, q_matrix, question_id_to_idx = load_questions_and_tags(data_dir)
    
    # 2. 加载学生交互数据
    interactions, n_students = load_student_interactions(data_dir, question_id_to_idx, max_students)
    
    # 3. 构建概念图
    concept_graph = build_concept_graph_from_qmatrix(q_matrix)
    
    # 4. 划分数据集
    train_data, val_data, test_data = split_data(interactions)
    
    # 5. 保存处理后的数据
    print(f"\n保存数据到: {output_file}")
    processed_data = {
        'ednet': {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'q_matrix': q_matrix,
            'concept_graph': concept_graph,
            'n_students': n_students,
            'n_questions': len(questions_df),
            'n_concepts': len(tag_to_id),
            'tag_to_id': tag_to_id,
            'question_id_to_idx': question_id_to_idx
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("\n" + "="*50)
    print("预处理完成！")
    print("="*50)
    print(f"数据统计:")
    print(f"  学生数: {n_students}")
    print(f"  题目数: {len(questions_df)}")
    print(f"  知识点数: {len(tag_to_id)}")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")
    
    return True
    print(f"\n输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='预处理EdNet数据集')
    parser.add_argument('--data_dir', type=str, default='data/raw/ednet/KT1',
                        help='EdNet数据目录')
    parser.add_argument('--output_file', type=str, default='data/ednet_processed.pkl',
                        help='输出文件路径')
    parser.add_argument('--max_students', type=int, default=None,
                        help='最大学生数（用于调试）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    process_ednet(args.data_dir, args.output_file, args.max_students)


if __name__ == '__main__':
    main()

