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

    # 创建数据目录（使用当前目录）
    data_dir = '.'
    os.makedirs(data_dir, exist_ok=True)

    # 使用EduData Python API下载
    try:
        from EduData import get_data
        print(f"\n正在下载 {edudata_name}...")
        print("-" * 60)
        
        data_path = get_data(edudata_name, data_dir=data_dir, override=False)
        print("[OK] 下载成功")
        print(f"数据路径: {data_path}")
    except ImportError:
        print("[ERROR] 找不到EduData模块")
        print("请确认EduData已正确安装: pip install EduData")
        return None
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 查找下载的文件
    # data_path可能是目录或文件
    if os.path.isdir(data_path):
        dataset_dir = data_path
        
        # 对于Junyi数据集，明确指定使用CSV文件
        if dataset_name == 'junyi':
            problem_log_csv = os.path.join(dataset_dir, 'junyi_ProblemLog_original.csv')
            if os.path.exists(problem_log_csv):
                print(f"\n找到Junyi主数据文件: {os.path.relpath(problem_log_csv)}")
                return problem_log_csv
            else:
                print(f"[WARNING] 未找到 junyi_ProblemLog_original.csv")
        
        # 查找CSV文件（优先）和TXT文件
        csv_files = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)
        txt_files = glob.glob(os.path.join(dataset_dir, '**', '*.txt'), recursive=True)
        
        # 排除小文件（如README、annotation等）
        csv_files = [f for f in csv_files if os.path.getsize(f) > 1 * 1024 * 1024]  # 大于1MB
        txt_files = [f for f in txt_files if os.path.getsize(f) > 1 * 1024 * 1024]  # 大于1MB
        
        # 优先使用CSV文件
        if csv_files:
            print(f"\n找到 {len(csv_files)} 个CSV数据文件:")
            for f in csv_files:
                size_mb = os.path.getsize(f) / 1024 / 1024
                print(f"  - {os.path.relpath(f)} ({size_mb:.2f} MB)")
            
            # 返回最大的CSV文件
            main_file = max(csv_files, key=os.path.getsize)
            print(f"\n使用主数据文件: {os.path.relpath(main_file)}")
            return main_file
        elif txt_files:
            # 如果没有CSV，才使用TXT
            print(f"\n找到 {len(txt_files)} 个TXT数据文件:")
            for f in txt_files:
                size_mb = os.path.getsize(f) / 1024 / 1024
                print(f"  - {os.path.relpath(f)} ({size_mb:.2f} MB)")
            
            main_file = max(txt_files, key=os.path.getsize)
            print(f"\n使用主数据文件: {os.path.relpath(main_file)}")
            return main_file
        else:
            print(f"[WARNING] 在 {dataset_dir} 中未找到数据文件")
            return None
    else:
        # 直接是文件
        return data_path


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
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']:
        try:
            # 检查文件扩展名
            if input_path.endswith('.csv'):
                df = pd.read_csv(input_path, encoding=encoding, low_memory=False)
            else:
                # txt文件，可能是制表符分隔
                df = pd.read_csv(input_path, sep='\t', encoding=encoding, low_memory=False)
            print(f"[OK] 使用编码 {encoding} 成功读取")
            break
        except Exception as e:
            if encoding == 'gbk':
                # 最后一次尝试，使用errors='ignore'
                try:
                    if input_path.endswith('.csv'):
                        df = pd.read_csv(input_path, encoding='utf-8', errors='ignore', low_memory=False)
                    else:
                        df = pd.read_csv(input_path, sep='\t', encoding='utf-8', errors='ignore', low_memory=False)
                    print(f"[OK] 使用UTF-8 (errors=ignore) 成功读取")
                    break
                except:
                    raise Exception(f"无法读取文件，尝试了多种编码都失败")

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
            'studentId': 'student_id',  # ASSIST17实际列名
            'problem_id': 'question_id',
            'problemId': 'question_id',  # ASSIST17实际列名
            'skill_id': 'concept_id',
            'skill': 'concept_id',  # ASSIST17实际列名
            'correct': 'correct',
            'order_id': 'timestamp',
            'startTime': 'timestamp',  # ASSIST17实际列名
            'student_id': 'student_id',
            'question_id': 'question_id',
            'concept_id': 'concept_id',
        },
        'junyi': {
            'user_id': 'student_id',
            'correct': 'correct',
            'time_done': 'timestamp',  # Junyi实际列名
            'exercise': 'question_id',  # Junyi实际列名：exercise作为题目
            # 注意：exercise将同时作为concept_id（在特殊处理中实现）
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
                print(f"  [OK] {src_col} -> {target}")
                found = True
                break

        if not found:
            print(f"  [WARNING] 未找到 {target} 对应的列")

    # 如果缺少关键列，显示所有列让用户检查
    if len(found_columns) < 4:
        print(f"\n可用的列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        raise ValueError(f"缺少必要的列。需要至少4个核心列（student_id, question_id, concept_id, correct）")

    # 特殊处理：Junyi数据集 - exercise同时作为question_id和concept_id
    if dataset_name == 'junyi':
        if 'exercise' in df.columns and 'exercise' in found_columns:
            print(f"\n[INFO] Junyi数据集：使用exercise作为concept_id")
            print(f"  策略：每个exercise既是题目也是知识点（细粒度：721个概念）")
            # 将exercise列复制一份作为concept_id
            # 注意：found_columns中已有 'exercise' -> 'question_id'
            # 我们需要在后续处理中复制这一列
            found_columns['exercise_concept'] = 'concept_id'  # 标记需要复制
    
    # 提取列
    # 特殊处理：Junyi的exercise_concept标记表示需要复制exercise列
    if 'exercise_concept' in found_columns:
        # Junyi特殊处理：exercise同时作为question_id和concept_id
        cols_to_extract = [col for col in found_columns.keys() if col != 'exercise_concept']
        df_converted = df[cols_to_extract].copy()
        df_converted.columns = [found_columns[col] for col in cols_to_extract]
        
        # 复制exercise列作为concept_id（如果还没有concept_id列）
        if 'concept_id' not in df_converted.columns and 'question_id' in df_converted.columns:
            df_converted['concept_id'] = df_converted['question_id'].copy()
            print(f"  已将question_id复制为concept_id")
    else:
        # 常规处理
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
            print(f"  [WARNING] correct列格式异常，尝试转换...")
            df_converted['correct'] = pd.to_numeric(df_converted['correct'], errors='coerce')
            df_converted = df_converted.dropna(subset=['correct'])
            df_converted['correct'] = df_converted['correct'].astype(int)

        # 只保留0和1
        df_converted = df_converted[df_converted['correct'].isin([0, 1])]
        print(f"  过滤correct列后: {len(df_converted)} 条记录")

    # 3. 删除无效的concept_id
    if 'concept_id' in df_converted.columns:
        # 尝试转换为数值类型（如果concept_id是字符串，如技能名称）
        original_type = df_converted['concept_id'].dtype
        if df_converted['concept_id'].dtype == 'object':
            # 如果是字符串类型，先不进行数值过滤，后面会重新编码
            print(f"  concept_id是字符串类型，将在ID重新编码时处理")
        else:
            # 如果是数值类型，删除负数和NaN
            df_converted = df_converted[df_converted['concept_id'] >= 0]
            print(f"  过滤无效concept_id后: {len(df_converted)} 条记录")

    # 4. 重新编码ID（从0开始）
    print(f"\nID重新编码:")
    for col in ['student_id', 'question_id', 'concept_id']:
        if col in df_converted.columns:
            # 删除NaN值
            before_len = len(df_converted)
            df_converted = df_converted.dropna(subset=[col])
            if len(df_converted) < before_len:
                print(f"  删除{col}的NaN值: {before_len} -> {len(df_converted)}")
            
            # 获取唯一值并排序（排除NaN）
            unique_vals = sorted([v for v in df_converted[col].unique() if pd.notna(v)])
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
    print("[OK] 转换完成")
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
            print("\n[ERROR] 下载失败，请检查错误信息")
            return
    else:
        # 手动指定已下载的文件
        print("\n跳过下载步骤")
        edudata_names = {
            'assist09': 'assistment-2009-2010-skill',
            'assist17': 'assistment-2017',
            'junyi': 'junyi'
        }
        dataset_dir = os.path.join('', edudata_names[args.dataset])
        # 对于Junyi，优先使用ProblemLog_original.csv
        if args.dataset == 'junyi':
            problem_log = os.path.join(dataset_dir, 'junyi_ProblemLog_original.csv')
            if os.path.exists(problem_log):
                input_path = problem_log
                print(f"使用Junyi主数据文件: {input_path}")
            else:
                # 回退到查找所有CSV文件
                csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
                if csv_files:
                    # 排除小文件（如Exercise_table等），选择最大的
                    data_files = [f for f in csv_files if os.path.getsize(f) > 100 * 1024 * 1024]  # 大于100MB
                    if data_files:
                        input_path = max(data_files, key=os.path.getsize)
                        print(f"使用数据文件: {input_path}")
                    else:
                        input_path = max(csv_files, key=os.path.getsize)
                        print(f"使用文件: {input_path}")
                else:
                    print(f"[ERROR] 未找到已下载的数据文件")
                    print(f"请检查目录: {os.path.abspath(dataset_dir)}")
                    return
        else:
            csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
            csv_files_recursive = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)
            all_csv_files = list(set(csv_files + csv_files_recursive))
            
            if all_csv_files:
                input_path = max(all_csv_files, key=os.path.getsize)
                print(f"使用已下载的文件: {input_path}")
            else:
                print(f"[ERROR] 未找到已下载的数据文件")
                print(f"请检查目录: {os.path.abspath(dataset_dir)}")
                return

    # 步骤2: 转换格式
    output_path = f'./data/{args.dataset}.csv'
    try:
        df = convert_to_our_format(input_path, output_path, args.dataset)

        print(f"\n{'='*60}")
        print("[OK] 全部完成！")
        print('='*60)
        print(f"\n下一步操作:")
        print(f"1. 运行数据预处理脚本:")
        print(f"   python data/process_downloaded_data.py")
        print(f"\n2. 运行快速验证实验:")
        print(f"   python experiments/run_baseline_experiments.py --datasets {args.dataset} --models DKT --n_runs 1 --n_epochs 10")
        print(f"\n3. 运行完整实验（论文标准）:")
        print(f"   python experiments/run_baseline_experiments.py --datasets {args.dataset} --models DKT DKVMN SAKT AKT GKT --n_runs 5 --n_epochs 50")

    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
