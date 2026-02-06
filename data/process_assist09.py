"""
ASSIST09数据集下载和预处理脚本

使用方法：
    python data/process_assist09.py
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
    使用Q-matrix方法构建概念图（论文方法）
    
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


def process_assist09():
    """处理ASSIST09数据集"""
    print("\n" + "="*50)
    print("ASSIST09数据集预处理")
    print("="*50)
    
    # 数据路径（EduData下载后的实际路径）
    data_path = "data/raw/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv"
    
    # 如果新路径不存在，尝试旧路径
    if not os.path.exists(data_path):
        data_path_old = "data/raw/assistment-2009-2010-skill/skill_builder_data.csv"
        if os.path.exists(data_path_old):
            data_path = data_path_old
        else:
            print(f"[ERROR] 数据文件不存在")
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
    
    # 按时间排序
    if 'order_id' in df.columns:
        df = df.sort_values(['user_id', 'order_id'])
    
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
    
    # 4. 构建Q矩阵
    print("\n4. 构建Q矩阵...")
    q_matrix = np.zeros((n_questions, n_concepts), dtype=np.float32)
    for _, row in df.iterrows():
        q_matrix[int(row['question_id']), int(row['concept_id'])] = 1.0
    print(f"  Q矩阵形状: {q_matrix.shape}")
    
    # 5. 构建概念图
    print("\n5. 构建概念图...")
    concept_graph = build_concept_graph_qmatrix(df, n_concepts)
    
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
    
    return True


def main():
    """主函数"""
    success = process_assist09()
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()

