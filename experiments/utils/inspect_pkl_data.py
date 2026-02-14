"""
检查processed_datasets.pkl文件的数据结构
输出少量样本数据用于验证
"""

import pickle
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def inspect_pkl_data(pkl_path='data/processed_datasets.pkl', max_samples=10):
    """
    检查pkl文件的数据结构并输出样本
    
    Args:
        pkl_path: pkl文件路径
        max_samples: 每个数据集输出的最大样本数
    """
    print("="*80)
    print("检查 processed_datasets.pkl 文件")
    print("="*80)
    
    # 检查文件是否存在
    full_path = os.path.join(project_root, pkl_path)
    if not os.path.exists(full_path):
        print(f"[错误] 文件不存在: {full_path}")
        return
    
    print(f"\n文件路径: {full_path}")
    file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    # 加载数据
    print("\n正在加载数据...")
    try:
        with open(full_path, 'rb') as f:
            datasets = pickle.load(f)
        print("[OK] 数据加载成功")
    except Exception as e:
        print(f"[错误] 加载失败: {e}")
        return
    
    # 检查顶层结构
    print("\n" + "="*80)
    print("顶层结构")
    print("="*80)
    print(f"数据类型: {type(datasets)}")
    
    if isinstance(datasets, dict):
        print(f"数据集数量: {len(datasets)}")
        print(f"数据集名称: {list(datasets.keys())}")
    else:
        print(f"数据内容: {datasets}")
        return
    
    # 检查每个数据集
    for dataset_name, dataset_info in datasets.items():
        print("\n" + "="*80)
        print(f"数据集: {dataset_name.upper()}")
        print("="*80)
        
        if not isinstance(dataset_info, dict):
            print(f"数据类型: {type(dataset_info)}")
            print(f"数据内容: {dataset_info}")
            continue
        
        # 检查数据集包含的键
        print(f"\n数据集键: {list(dataset_info.keys())}")
        
        # 检查基本信息
        if 'n_questions' in dataset_info:
            print(f"\n基本信息:")
            print(f"  问题数 (n_questions): {dataset_info['n_questions']}")
        if 'n_concepts' in dataset_info:
            print(f"  知识点数 (n_concepts): {dataset_info['n_concepts']}")
        if 'n_students' in dataset_info:
            print(f"  学生数 (n_students): {dataset_info['n_students']}")
        
        # 检查Q-matrix
        if 'q_matrix' in dataset_info:
            q_matrix = dataset_info['q_matrix']
            print(f"\nQ-matrix:")
            print(f"  形状: {q_matrix.shape}")
            print(f"  非零元素数: {np.count_nonzero(q_matrix)}")
            print(f"  非零比例: {np.count_nonzero(q_matrix) / q_matrix.size * 100:.2f}%")
            # 显示前几行几列
            if q_matrix.shape[0] > 0 and q_matrix.shape[1] > 0:
                print(f"  前5行5列:")
                print(q_matrix[:5, :5])
        
        # 检查concept_graph
        if 'concept_graph' in dataset_info:
            concept_graph = dataset_info['concept_graph']
            print(f"\nConcept Graph:")
            print(f"  形状: {concept_graph.shape}")
            print(f"  类型: {type(concept_graph)}")
            if isinstance(concept_graph, np.ndarray):
                print(f"  数据类型: {concept_graph.dtype}")
                print(f"  非零元素数: {np.count_nonzero(concept_graph)}")
                print(f"  非零比例: {np.count_nonzero(concept_graph) / concept_graph.size * 100:.2f}%")
                print(f"  最大值: {concept_graph.max():.4f}")
                print(f"  最小值: {concept_graph.min():.4f}")
                print(f"  均值: {concept_graph.mean():.4f}")
                # 显示前几行几列
                if concept_graph.shape[0] > 0 and concept_graph.shape[1] > 0:
                    print(f"  前5行5列:")
                    print(concept_graph[:5, :5])
        
        # 检查训练/验证/测试数据
        for split_name in ['train', 'val', 'test']:
            if split_name in dataset_info:
                split_data = dataset_info[split_name]
                print(f"\n{split_name.upper()} 数据:")
                print(f"  类型: {type(split_data)}")
                
                if isinstance(split_data, pd.DataFrame):
                    print(f"  形状: {split_data.shape}")
                    print(f"  列名: {list(split_data.columns)}")
                    print(f"  数据量: {len(split_data)}")
                    
                    # 显示前几行
                    print(f"\n  前{min(max_samples, len(split_data))}行数据:")
                    print(split_data.head(max_samples).to_string())
                    
                    # 统计信息
                    print(f"\n  统计信息:")
                    if 'student_id' in split_data.columns:
                        print(f"    唯一学生数: {split_data['student_id'].nunique()}")
                    if 'question_id' in split_data.columns:
                        print(f"    唯一问题数: {split_data['question_id'].nunique()}")
                        print(f"    问题ID范围: [{split_data['question_id'].min()}, {split_data['question_id'].max()}]")
                    if 'concept_id' in split_data.columns:
                        print(f"    唯一知识点数: {split_data['concept_id'].nunique()}")
                        print(f"    知识点ID范围: [{split_data['concept_id'].min()}, {split_data['concept_id'].max()}]")
                    if 'correct' in split_data.columns:
                        correct_rate = split_data['correct'].mean()
                        print(f"    正确率: {correct_rate:.4f}")
                        print(f"    正确数: {split_data['correct'].sum()}")
                        print(f"    错误数: {(~split_data['correct'].astype(bool)).sum()}")
                    if 'timestamp' in split_data.columns:
                        print(f"    时间范围: [{split_data['timestamp'].min()}, {split_data['timestamp'].max()}]")
                
                elif isinstance(split_data, (list, tuple)):
                    print(f"  长度: {len(split_data)}")
                    if len(split_data) > 0:
                        print(f"  第一个元素类型: {type(split_data[0])}")
                        print(f"  前{min(max_samples, len(split_data))}个元素:")
                        for i, item in enumerate(split_data[:max_samples]):
                            print(f"    [{i}]: {item}")
                else:
                    print(f"  内容: {split_data}")
        
        # 检查其他可能的键
        other_keys = [k for k in dataset_info.keys() 
                     if k not in ['train', 'val', 'test', 'q_matrix', 'concept_graph', 
                                  'n_questions', 'n_concepts', 'n_students']]
        if other_keys:
            print(f"\n其他键: {other_keys}")
            for key in other_keys:
                value = dataset_info[key]
                print(f"  {key}: {type(value)} = {value}")
    
    print("\n" + "="*80)
    print("检查完成")
    print("="*80)


def show_student_sequence(dataset_name='assist09', student_id=None, max_interactions=20):
    """
    显示某个学生的交互序列
    
    Args:
        dataset_name: 数据集名称
        student_id: 学生ID（如果为None，随机选择一个）
        max_interactions: 显示的最大交互数
    """
    print("\n" + "="*80)
    print(f"学生交互序列示例 - {dataset_name.upper()}")
    print("="*80)
    
    pkl_path = os.path.join(project_root, 'data/processed_datasets.pkl')
    with open(pkl_path, 'rb') as f:
        datasets = pickle.load(f)
    
    if dataset_name not in datasets:
        print(f"[错误] 数据集 {dataset_name} 不存在")
        return
    
    dataset_info = datasets[dataset_name]
    train_data = dataset_info['train']
    
    if not isinstance(train_data, pd.DataFrame):
        print("[错误] 训练数据不是DataFrame格式")
        return
    
    # 选择学生
    if student_id is None:
        available_students = train_data['student_id'].unique()
        student_id = np.random.choice(available_students)
        print(f"\n随机选择学生ID: {student_id}")
    else:
        print(f"\n学生ID: {student_id}")
    
    # 获取该学生的所有交互
    student_interactions = train_data[train_data['student_id'] == student_id].copy()
    student_interactions = student_interactions.sort_values('timestamp')
    
    print(f"总交互数: {len(student_interactions)}")
    print(f"\n前{min(max_interactions, len(student_interactions))}次交互:")
    print("-" * 80)
    
    display_cols = ['question_id', 'concept_id', 'correct', 'timestamp']
    display_cols = [col for col in display_cols if col in student_interactions.columns]
    
    print(student_interactions[display_cols].head(max_interactions).to_string(index=False))
    
    # 统计信息
    print(f"\n该学生统计:")
    print(f"  总交互数: {len(student_interactions)}")
    print(f"  正确数: {student_interactions['correct'].sum()}")
    print(f"  错误数: {(~student_interactions['correct'].astype(bool)).sum()}")
    print(f"  正确率: {student_interactions['correct'].mean():.4f}")
    print(f"  涉及问题数: {student_interactions['question_id'].nunique()}")
    print(f"  涉及知识点数: {student_interactions['concept_id'].nunique()}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检查processed_datasets.pkl文件')
    parser.add_argument('--pkl_path', type=str, default='data/processed_datasets.pkl',
                       help='pkl文件路径')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='每个数据集输出的最大样本数')
    parser.add_argument('--show_student', action='store_true',
                       help='显示学生交互序列示例')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'assist17', 'junyi'],
                       help='数据集名称（用于显示学生序列）')
    parser.add_argument('--student_id', type=int, default=None,
                       help='学生ID（如果为None，随机选择）')
    
    args = parser.parse_args()
    
    # 检查pkl文件
    inspect_pkl_data(args.pkl_path, args.max_samples)
    
    # 如果指定，显示学生序列
    if args.show_student:
        show_student_sequence(args.dataset, args.student_id)

