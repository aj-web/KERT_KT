"""
处理下载的真实数据集，生成实验所需的pkl文件
支持ASSIST09, ASSIST17, Junyi数据集
"""
import os
import pandas as pd
import numpy as np
import pickle


def build_concept_graph(df, n_concepts, method='qmatrix'):
    """
    构建概念图（知识点关系图）
    
    论文3.3.1节：使用Q-matrix构建知识点图
    边权重 = 知识点共现次数：w_{ij} = Σ_q I(k_i ∈ K_q ∧ k_j ∈ K_q)

    Args:
        df: 数据DataFrame，包含question_id和concept_id列
        n_concepts: 知识点总数
        method: 构建方法
            - 'qmatrix': 基于Q-matrix（题目-知识点关联），论文方法
            - 'sequence': 基于学习序列中的相邻关系（原方法）

    Returns:
        concept_graph: np.array, shape=(n_concepts, n_concepts)
    """
    print(f"  构建概念图（方法: {method}）...")

    # 初始化邻接矩阵
    concept_graph = np.zeros((n_concepts, n_concepts), dtype=np.float32)

    if method == 'qmatrix':
        # 论文方法：基于Q-matrix（题目-知识点关联）
        # 如果两个知识点在同一题目中被考查，则在它们之间建立边
        print("    使用Q-matrix方法（论文3.3.1节）")
        
        # 获取每个题目考查的知识点集合
        question_concepts = df.groupby('question_id')['concept_id'].apply(set).to_dict()
        
        # 统计知识点共现次数
        for q_id, concept_set in question_concepts.items():
            concepts = list(concept_set)
            # 对于题目中的每对知识点，增加共现次数
            for i in range(len(concepts)):
                for j in range(i+1, len(concepts)):
                    c1, c2 = concepts[i], concepts[j]
                    concept_graph[c1, c2] += 1
                    concept_graph[c2, c1] += 1  # 无向图
        
        print(f"    共现关系来源: {len(question_concepts)} 个题目")
    
    elif method == 'sequence':
        # 原方法：基于学习序列中的相邻关系
        print("    使用学习序列方法（原实现）")
        
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
        
        print(f"    共现关系来源: {df['student_id'].nunique()} 个学生的学习序列")
    
    else:
        raise ValueError(f"未知的构建方法: {method}")

    # 归一化（按行）
    row_sums = concept_graph.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以0
    concept_graph = concept_graph / row_sums

    # 添加自环
    concept_graph += np.eye(n_concepts)

    print(f"    概念图非零元素: {np.count_nonzero(concept_graph)}/{n_concepts*n_concepts}")
    print(f"    概念图密度: {np.count_nonzero(concept_graph)/(n_concepts*n_concepts):.4f}")

    return concept_graph


def split_dataset(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    严格时序划分数据集（论文4.2.2节）
    对每个学生，按时间顺序划分：前70%→训练，中间10%→验证，最后20%→测试

    Args:
        df: 完整数据DataFrame
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.2）

    Returns:
        train_df, val_df, test_df
    """
    print("  划分数据集（严格时序）...")

    train_list = []
    val_list = []
    test_list = []

    for student_id, group in df.groupby('student_id'):
        # 按时间排序
        group = group.sort_values('timestamp').reset_index(drop=True)
        n_interactions = len(group)

        # 计算划分点
        train_end = int(n_interactions * train_ratio)
        val_end = int(n_interactions * (train_ratio + val_ratio))

        # 严格时序划分
        train_list.append(group.iloc[:train_end])
        val_list.append(group.iloc[train_end:val_end])
        test_list.append(group.iloc[val_end:])

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print(f"    训练集: {len(train_df)} 条记录")
    print(f"    验证集: {len(val_df)} 条记录")
    print(f"    测试集: {len(test_df)} 条记录")

    return train_df, val_df, test_df


def process_dataset(csv_path, dataset_name, graph_method='qmatrix'):
    """
    处理单个数据集

    Args:
        csv_path: CSV文件路径
        dataset_name: 数据集名称（用于显示）
        graph_method: 概念图构建方法（'qmatrix'或'sequence'）

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

    # 数据清洗（论文4.1节）
    print("\n数据清洗:")
    original_len = len(df)
    
    # 1. 移除交互记录少于5次的学生
    print("  (1) 移除交互记录少于5次的学生...")
    student_counts = df['student_id'].value_counts()
    valid_students = student_counts[student_counts >= 5].index
    df = df[df['student_id'].isin(valid_students)]
    print(f"      移除 {original_len - len(df):,} 条记录")
    
    # 2. 删除答题时间异常的记录（论文4.1节：<1秒或>1小时）
    print("  (2) 删除答题时间异常的记录（<1秒或>1小时）...")
    # 检查是否有时间列（duration, time_taken, time_done等）
    time_cols = ['duration', 'time_taken', 'response_time', 'time_done']
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is not None:
        before_len = len(df)
        # 假设时间单位是秒，过滤<1秒或>3600秒（1小时）的记录
        if df[time_col].dtype in ['int64', 'float64']:
            df = df[(df[time_col] >= 1) & (df[time_col] <= 3600)]
            print(f"      使用列 '{time_col}'，移除 {before_len - len(df):,} 条异常记录")
        else:
            print(f"      警告：时间列 '{time_col}' 类型不是数值，跳过时间过滤")
    else:
        print(f"      警告：未找到时间列（尝试过：{time_cols}），跳过时间过滤")
    
    print(f"  清洗后剩余记录数: {len(df):,}")
    
    # 3. 重新编码ID（确保ID从0开始连续）
    print("  (3) 重新编码ID（确保连续性）...")
    from sklearn.preprocessing import LabelEncoder
    
    student_encoder = LabelEncoder()
    question_encoder = LabelEncoder()
    concept_encoder = LabelEncoder()
    
    df['student_id'] = student_encoder.fit_transform(df['student_id'])
    df['question_id'] = question_encoder.fit_transform(df['question_id'])
    df['concept_id'] = concept_encoder.fit_transform(df['concept_id'])
    
    # 重新统计（清洗后）
    n_students = df['student_id'].nunique()
    n_questions = df['question_id'].nunique()
    n_concepts = df['concept_id'].nunique()
    print(f"\n清洗后统计:")
    print(f"  学生数: {n_students:,}")
    print(f"  题目数: {n_questions:,}")
    print(f"  知识点数: {n_concepts:,}")
    print(f"  答题记录数: {len(df):,}")
    print(f"  平均正确率: {df['correct'].mean():.4f}")

    # 划分数据集
    train_df, val_df, test_df = split_dataset(df)

    # 构建概念图（论文3.3.1节：使用Q-matrix方法）
    concept_graph = build_concept_graph(df, n_concepts, method=graph_method)

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

    print(f"\n[OK] {dataset_name.upper()} 处理完成")

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
    parser.add_argument('--graph-method', type=str, default='qmatrix',
                       choices=['qmatrix', 'sequence'],
                       help='概念图构建方法：qmatrix（论文方法，基于题目-知识点关联）或sequence（基于学习序列）')

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
            print(f"\n[WARNING] 未找到 {csv_path}")
            print(f"  请先运行: python download_with_edudata.py --dataset {dataset_name}")
            continue

        try:
            dataset_info = process_dataset(csv_path, dataset_name, graph_method=args.graph_method)
            all_datasets[dataset_name] = dataset_info
        except Exception as e:
            print(f"\n[ERROR] 处理 {dataset_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_datasets:
        print("\n[ERROR] 没有成功处理任何数据集")
        return

    # 保存为pkl文件
    print(f"\n{'='*60}")
    print("保存处理后的数据")
    print('='*60)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'wb') as f:
        pickle.dump(all_datasets, f)

    print(f"[OK] 已保存到: {args.output}")

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
    print("[OK] 全部完成！")
    print('='*60)
    print(f"\n下一步操作:")
    print(f"运行基线实验:")
    print(f"  python experiments/run_baseline_experiments.py --datasets {' '.join(args.datasets)} --models DKT --n_runs 1 --n_epochs 10")
    print(f"\n运行完整实验（论文标准）:")
    print(f"  python experiments/run_baseline_experiments.py --datasets {' '.join(args.datasets)} --models DKT DKVMN SAKT AKT GKT --n_runs 5 --n_epochs 50")


if __name__ == "__main__":
    main()
