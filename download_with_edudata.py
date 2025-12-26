"""
使用EduData CLI下载和处理真实数据集
支持ASSIST09, ASSIST17, Junyi三个数据集
"""
import os
import subprocess
import pandas as pd
import numpy as np
import glob


def download_dataset_with_edudata(dataset_name):
    """
    使用EduData CLI下载数据集

    Args:
        dataset_name: 数据集名称 ('assist09', 'assist17', 'junyi')

    Returns:
        下载的数据文件路径
    """
    # 映射数据集名称到EduData的名称
    edudata_names = {
        'assist09': 'assistment-2009-2010-skill',
        'assist17': 'assistment-2017',
        'junyi': 'junyi'
    }

    if dataset_name not in edudata_names:
        raise ValueError(f"不支持的数据集: {dataset_name}. 支持: {list(edudata_names.keys())}")

    edudata_name = edudata_names[dataset_name]

    print("=" * 60)
    print(f"使用EduData下载 {dataset_name.upper()} 数据集")
    print("=" * 60)

    # 创建数据目录
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    # 下载数据
    print(f"\n执行命令: edudata download {edudata_name} {data_dir}")
    print("-" * 60)

    try:
        result = subprocess.run(
            ['edudata', 'download', edudata_name, data_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("警告信息:")
            print(result.stderr)
        print("✓ 下载成功")
    except subprocess.CalledProcessError as e:
        print(f"✗ 下载失败: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return None
    except FileNotFoundError:
        print("✗ 找不到edudata命令")
        print("请确认EduData已正确安装: pip install EduData")
        return None

    # 查找下载的文件
    dataset_dir = os.path.join(data_dir, edudata_name)
    if not os.path.exists(dataset_dir):
        print(f"⚠ 未找到数据目录: {dataset_dir}")
        return None

    # 查找CSV文件
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    txt_files = glob.glob(os.path.join(dataset_dir, '*.txt'))
    data_files = csv_files + txt_files

    print(f"\n找到 {len(data_files)} 个数据文件:")
    for f in data_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  - {os.path.basename(f)} ({size_mb:.2f} MB)")

    if not data_files:
        print(f"⚠ 在 {dataset_dir} 中未找到数据文件")
        return None

    # 返回最大的文件（通常是主数据文件）
    main_file = max(data_files, key=os.path.getsize)
    print(f"\n使用主数据文件: {os.path.basename(main_file)}")

    return main_file


def convert_to_our_format(input_path, output_path, dataset_name):
    """
    将EduData格式转换为项目格式

    Args:
        input_path: EduData下载的数据文件路径
        output_path: 输出CSV路径
        dataset_name: 数据集名称 ('assist09', 'assist17', 'junyi')

    Returns:
        转换后的DataFrame
    """
    print(f"\n{'='*60}")
    print(f"转换 {dataset_name.upper()} 数据格式")
    print('='*60)

    # 读取数据
    print(f"读取文件: {input_path}")

    # 尝试不同的编码
    for encoding in ['utf-8', 'latin-1', 'gbk']:
        try:
            # 检查文件扩展名
            if input_path.endswith('.csv'):
                df = pd.read_csv(input_path, encoding=encoding)
            else:
                # txt文件，可能是制表符分隔
                df = pd.read_csv(input_path, sep='\t', encoding=encoding)
            print(f"✓ 使用编码 {encoding} 成功读取")
            break
        except Exception as e:
            print(f"尝试编码 {encoding} 失败: {e}")
            if encoding == 'gbk':
                raise

    print(f"\n原始数据信息:")
    print(f"  形状: {df.shape}")
    print(f"  列数: {len(df.columns)}")

    # 显示列名（截断显示）
    if len(df.columns) > 15:
        print(f"  列名（前15个）: {df.columns.tolist()[:15]}...")
    else:
        print(f"  列名: {df.columns.tolist()}")

    # 不同数据集的列名映射
    column_mappings = {
        'assist09': {
            'user_id': 'student_id',
            'problem_id': 'question_id',
            'skill_id': 'concept_id',
            'correct': 'correct',
            'order_id': 'timestamp',
            # 备选列名
            'student_id': 'student_id',
            'question_id': 'question_id',
            'concept_id': 'concept_id',
            'item_id': 'question_id',
            'skill': 'concept_id',
        },
        'assist17': {
            'user_id': 'student_id',
            'problem_id': 'question_id',
            'skill_id': 'concept_id',
            'correct': 'correct',
            'order_id': 'timestamp',
            'student_id': 'student_id',
            'question_id': 'question_id',
            'concept_id': 'concept_id',
        },
        'junyi': {
            'user_id': 'student_id',
            'problem_id': 'question_id',
            'skill_id': 'concept_id',
            'correct': 'correct',
            'order_id': 'timestamp',
            'student_id': 'student_id',
            'exercise': 'question_id',
            'topic': 'concept_id',
        }
    }

    mapping = column_mappings.get(dataset_name, column_mappings['assist09'])

    # 查找匹配的列
    print(f"\n列名映射:")
    found_columns = {}
    target_names = ['student_id', 'question_id', 'concept_id', 'correct', 'timestamp']

    for target in target_names:
        found = False
        # 查找可能的源列名
        for src_col, tgt_col in mapping.items():
            if tgt_col == target and src_col in df.columns:
                found_columns[src_col] = target
                print(f"  ✓ {src_col} -> {target}")
                found = True
                break

        if not found:
            print(f"  ⚠ 未找到 {target} 对应的列")

    # 如果缺少关键列，显示所有列让用户检查
    if len(found_columns) < 4:
        print(f"\n可用的列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        raise ValueError(f"缺少必要的列。需要至少4个核心列（student_id, question_id, concept_id, correct）")

    # 提取列
    df_converted = df[[col for col in found_columns.keys()]].copy()
    df_converted.columns = [found_columns[col] for col in df_converted.columns]

    # 数据清洗
    print(f"\n数据清洗:")
    original_len = len(df_converted)

    # 1. 删除缺失值
    df_converted = df_converted.dropna()
    print(f"  删除缺失值: {original_len} -> {len(df_converted)}")

    # 2. 确保correct是0/1
    if 'correct' in df_converted.columns:
        # 可能是True/False或其他格式
        unique_values = df_converted['correct'].unique()
        print(f"  correct列的唯一值: {unique_values[:10]}")

        # 转换为0/1
        if df_converted['correct'].dtype == bool or set(unique_values).issubset({True, False, 0, 1}):
            df_converted['correct'] = df_converted['correct'].astype(int)
        else:
            # 可能需要特殊处理
            print(f"  ⚠ correct列格式异常，尝试转换...")
            df_converted['correct'] = pd.to_numeric(df_converted['correct'], errors='coerce')
            df_converted = df_converted.dropna(subset=['correct'])
            df_converted['correct'] = df_converted['correct'].astype(int)

        # 只保留0和1
        df_converted = df_converted[df_converted['correct'].isin([0, 1])]
        print(f"  过滤correct列后: {len(df_converted)} 条记录")

    # 3. 删除无效的concept_id
    if 'concept_id' in df_converted.columns:
        # 删除负数和NaN
        df_converted = df_converted[df_converted['concept_id'] >= 0]

    # 4. 重新编码ID（从0开始）
    print(f"\nID重新编码:")
    for col in ['student_id', 'question_id', 'concept_id']:
        if col in df_converted.columns:
            unique_vals = sorted(df_converted[col].unique())
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_vals)}
            df_converted[col] = df_converted[col].map(id_mapping)
            print(f"  {col}: {len(unique_vals)} 个唯一值 (0 到 {len(unique_vals)-1})")

    # 5. 创建timestamp列（如果没有）
    if 'timestamp' not in df_converted.columns:
        print(f"  创建timestamp列（基于学生分组的序号）")
        df_converted = df_converted.sort_values(['student_id'])
        df_converted['timestamp'] = df_converted.groupby('student_id').cumcount()
    else:
        # 如果有timestamp，按学生和时间排序
        df_converted = df_converted.sort_values(['student_id', 'timestamp'])
        # 重新编号timestamp
        df_converted['timestamp'] = df_converted.groupby('student_id').cumcount()

    # 重置索引
    df_converted = df_converted.reset_index(drop=True)

    # 保存
    df_converted.to_csv(output_path, index=False)

    # 统计信息
    print(f"\n{'='*60}")
    print("✓ 转换完成")
    print('='*60)
    print(f"输出文件: {output_path}")
    print(f"数据形状: {df_converted.shape}")
    print(f"\n数据集统计:")
    print(f"  学生数: {df_converted['student_id'].nunique():,}")
    print(f"  题目数: {df_converted['question_id'].nunique():,}")
    print(f"  知识点数: {df_converted['concept_id'].nunique():,}")
    print(f"  答题记录数: {len(df_converted):,}")
    print(f"  平均正确率: {df_converted['correct'].mean():.4f}")

    # 序列长度统计
    seq_lengths = df_converted.groupby('student_id').size()
    print(f"\n序列长度统计:")
    print(f"  平均: {seq_lengths.mean():.2f}")
    print(f"  中位数: {seq_lengths.median():.2f}")
    print(f"  最小: {seq_lengths.min()}")
    print(f"  最大: {seq_lengths.max()}")

    # 显示示例数据
    print(f"\n示例数据（前5行）:")
    print(df_converted.head())

    return df_converted


def main():
    """
    主函数：下载并转换数据集
    """
    import argparse

    parser = argparse.ArgumentParser(description='使用EduData下载和转换数据集')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'assist17', 'junyi'],
                       help='要下载的数据集')
    parser.add_argument('--skip-download', action='store_true',
                       help='跳过下载步骤（如果已下载）')

    args = parser.parse_args()

    print("=" * 60)
    print(f"EduData数据集下载和转换工具")
    print(f"数据集: {args.dataset.upper()}")
    print("=" * 60)

    # 步骤1: 下载数据
    if not args.skip_download:
        input_path = download_dataset_with_edudata(args.dataset)
        if input_path is None:
            print("\n✗ 下载失败，请检查错误信息")
            return
    else:
        # 手动指定已下载的文件
        print("\n跳过下载步骤")
        edudata_names = {
            'assist09': 'assistment-2009-2010-skill',
            'assist17': 'assistment-2017',
            'junyi': 'junyi'
        }
        dataset_dir = os.path.join('./data', edudata_names[args.dataset])
        csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
        if csv_files:
            input_path = max(csv_files, key=os.path.getsize)
            print(f"使用已下载的文件: {input_path}")
        else:
            print(f"✗ 未找到已下载的数据文件")
            return

    # 步骤2: 转换格式
    output_path = f'./data/{args.dataset}.csv'
    try:
        df = convert_to_our_format(input_path, output_path, args.dataset)

        print(f"\n{'='*60}")
        print("✅ 全部完成！")
        print('='*60)
        print(f"\n下一步操作:")
        print(f"1. 运行数据预处理脚本:")
        print(f"   python regenerate_data.py")
        print(f"\n2. 运行快速验证实验:")
        print(f"   python experiments/run_baseline_experiments.py --datasets {args.dataset} --models DKT --n_runs 1 --n_epochs 10")
        print(f"\n3. 运行完整实验（论文标准）:")
        print(f"   python experiments/run_baseline_experiments.py --datasets {args.dataset} --models DKT DKVMN SAKT AKT GKT --n_runs 5 --n_epochs 50")

    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
