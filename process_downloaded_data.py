"""
处理下载的真实数据集，生成实验所需的pkl文件
支持ASSIST09, ASSIST17, Junyi数据集
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


def build_concept_graph(df, n_concepts):
    """
    构建概念图（知识点关系图）

    Args:
        df: 数据DataFrame，包含question_id和concept_id列
        n_concepts: 知识点总数

    Returns:
        concept_graph: np.array, shape=(n_concepts, n_concepts)
    """
    print("  构建概念图...")

    # 初始化邻接矩阵
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)

    # 统计概念共现关系
    # 如果两个概念在同一个学生的学习序列中相邻出现，则它们有关系
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id].sort_values('timestamp')
        concepts = student_data['concept_id'].values

        # 统计相邻概念
        for i in range(len(concepts) - 1):
            c1, c2 = concepts[i], concepts[i+1]
            if c1 != c2:  # 不同的概念
                concept_graph[c1, c2] += 1
                concept_graph[c2, c1] += 1  # 无向图

    # 归一化（按行）
    row_sums = concept_graph.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以0
    concept_graph = concept_graph / row_sums

    # 添加自环
    concept_graph += np.eye(n_concepts)

    print(f"    概念图非零元素: {np.count_nonzero(concept_graph)}/{n_concepts*n_concepts}")
    print(f"    概念图密度: {np.count_nonzero(concept_graph)/(n_concepts*n_concepts):.4f}")

    return concept_graph


def split_dataset(df, test_size=0.2, val_size=0.1, random_seed=42):
    """
    按学生划分数据集为train/val/test

    Args:
        df: 完整数据DataFrame
        test_size: 测试集比例
        val_size: 验证集比例（相对于train+val）
        random_seed: 随机种子

    Returns:
        train_df, val_df, test_df
    """
    print("  划分数据集...")

    # 获取所有学生ID
    student_ids = df['student_id'].unique()
    np.random.seed(random_seed)

    # 先划分test
    train_val_ids, test_ids = train_test_split(
        student_ids,
        test_size=test_size,
        random_state=random_seed
    )

    # 再从train_val中划分val
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size / (1 - test_size),  # 调整比例
        random_state=random_seed
    )

    # 根据学生ID划分数据
    train_df = df[df['student_id'].isin(train_ids)].copy()
    val_df = df[df['student_id'].isin(val_ids)].copy()
    test_df = df[df['student_id'].isin(test_ids)].copy()

    print(f"    训练集: {len(train_df)} 条记录, {len(train_ids)} 个学生")
    print(f"    验证集: {len(val_df)} 条记录, {len(val_ids)} 个学生")
    print(f"    测试集: {len(test_df)} 条记录, {len(test_ids)} 个学生")

    return train_df, val_df, test_df


def process_dataset(csv_path, dataset_name):
    """
    处理单个数据集

    Args:
        csv_path: CSV文件路径
        dataset_name: 数据集名称（用于显示）

    Returns:
        dataset_info: 包含所有必要信息的字典
    """
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name.upper()}")
    print('='*60)

    # 读取CSV
    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  原始数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}")

    # 验证必要的列
    required_cols = ['student_id', 'question_id', 'concept_id', 'correct', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")

    # 统计信息
    n_students = df['student_id'].nunique()
    n_questions = df['question_id'].nunique()
    n_concepts = df['concept_id'].nunique()

    print(f"\n数据集统计:")
    print(f"  学生数: {n_students:,}")
    print(f"  题目数: {n_questions:,}")
    print(f"  知识点数: {n_concepts:,}")
    print(f"  答题记录数: {len(df):,}")
    print(f"  平均正确率: {df['correct'].mean():.4f}")

    # 划分数据集
    train_df, val_df, test_df = split_dataset(df)

    # 构建概念图
    concept_graph = build_concept_graph(df, n_concepts)

    # 构建Q矩阵（question -> concept映射）
    print("  构建Q矩阵...")
    q_matrix = np.zeros((n_questions, n_concepts), dtype=np.float32)
    for _, row in df[['question_id', 'concept_id']].drop_duplicates().iterrows():
        q_id = int(row['question_id'])
        c_id = int(row['concept_id'])
        q_matrix[q_id, c_id] = 1.0

    print(f"    Q矩阵形状: {q_matrix.shape}")
    print(f"    Q矩阵非零元素: {np.count_nonzero(q_matrix)}")

    # 组装dataset_info
    dataset_info = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'n_students': n_students,
        'n_questions': n_questions,
        'n_concepts': n_concepts,
        'concept_graph': concept_graph,
        'q_matrix': q_matrix
    }

    print(f"\n✓ {dataset_name.upper()} 处理完成")

    return dataset_info


def main():
    """
    主函数：处理所有下载的数据集
    """
    import argparse

    parser = argparse.ArgumentParser(description='处理下载的真实数据集')
    parser.add_argument('--datasets', nargs='+', default=['assist09'],
                       choices=['assist09', 'assist17', 'junyi'],
                       help='要处理的数据集')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--output', type=str, default='./data/processed_datasets.pkl',
                       help='输出PKL文件路径')

    args = parser.parse_args()

    print("=" * 60)
    print("真实数据集处理工具")
    print("=" * 60)
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出文件: {args.output}")

    # 处理每个数据集
    all_datasets = {}

    for dataset_name in args.datasets:
        csv_path = os.path.join(args.data_dir, f'{dataset_name}.csv')

        if not os.path.exists(csv_path):
            print(f"\n⚠ 警告: 未找到 {csv_path}")
            print(f"  请先运行: python download_with_edudata.py --dataset {dataset_name}")
            continue

        try:
            dataset_info = process_dataset(csv_path, dataset_name)
            all_datasets[dataset_name] = dataset_info
        except Exception as e:
            print(f"\n✗ 处理 {dataset_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_datasets:
        print("\n✗ 没有成功处理任何数据集")
        return

    # 保存为pkl文件
    print(f"\n{'='*60}")
    print("保存处理后的数据")
    print('='*60)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'wb') as f:
        pickle.dump(all_datasets, f)

    print(f"✓ 已保存到: {args.output}")

    # 显示文件信息
    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"  文件大小: {file_size:.2f} MB")

    # 汇总统计
    print(f"\n{'='*60}")
    print("处理完成！数据集汇总")
    print('='*60)

    for dataset_name, info in all_datasets.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  学生: {info['n_students']:,}")
        print(f"  题目: {info['n_questions']:,}")
        print(f"  知识点: {info['n_concepts']:,}")
        print(f"  训练集: {len(info['train']):,} 条")
        print(f"  验证集: {len(info['val']):,} 条")
        print(f"  测试集: {len(info['test']):,} 条")
        print(f"  训练集正确率: {info['train']['correct'].mean():.4f}")

    print(f"\n{'='*60}")
    print("✅ 全部完成！")
    print('='*60)
    print(f"\n下一步操作:")
    print(f"运行基线实验:")
    print(f"  python experiments/run_baseline_experiments.py --datasets {' '.join(args.datasets)} --models DKT --n_runs 1 --n_epochs 10")
    print(f"\n运行完整实验（论文标准）:")
    print(f"  python experiments/run_baseline_experiments.py --datasets {' '.join(args.datasets)} --models DKT DKVMN SAKT AKT GKT --n_runs 5 --n_epochs 50")


if __name__ == "__main__":
    main()
