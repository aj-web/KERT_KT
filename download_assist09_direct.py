"""
直接下载ASSIST09数据集（绕过EduData CLI）
从Kaggle或官方源下载真实的ASSIST09数据
"""
import os
import requests
import zipfile
import pandas as pd
from io import BytesIO

def download_assist09_from_url():
    """
    从公开URL下载ASSIST09数据集
    """
    print("=" * 60)
    print("下载ASSIST09数据集")
    print("=" * 60)

    # 数据集URL（这是ASSISTments 2009-2010 Skill Builder数据集）
    # 注意：如果这个链接失效，可以从以下来源获取：
    # 1. https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
    # 2. Kaggle: https://www.kaggle.com/datasets/
    # 3. 手动下载后放到data目录

    urls = [
        # 尝试多个可能的源
        "https://drive.google.com/uc?export=download&id=1eHmszoaXYDG_f8KVhZBxMkOVJONxJSN0",  # Google Drive备份
        "http://www.cs.cmu.edu/~nlao/data/skill_builder_data.csv",  # CMU镜像
    ]

    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'assist09_raw.csv')

    # 如果文件已存在，询问是否重新下载
    if os.path.exists(output_path):
        print(f"✓ 数据文件已存在: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        return output_path

    # 尝试下载
    for idx, url in enumerate(urls):
        print(f"\n尝试从源 {idx+1} 下载...")
        print(f"URL: {url[:80]}...")

        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                print(f"文件大小: {total_size / 1024 / 1024:.2f} MB")

                # 保存文件
                with open(output_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r下载进度: {percent:.1f}%", end='')

                print(f"\n✓ 下载成功: {output_path}")
                return output_path

        except Exception as e:
            print(f"✗ 下载失败: {e}")
            continue

    # 所有源都失败
    print("\n" + "=" * 60)
    print("⚠ 自动下载失败，请手动下载数据集")
    print("=" * 60)
    print("\n手动下载步骤：")
    print("1. 访问: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data")
    print("2. 下载 'skill_builder_data.csv' 文件")
    print(f"3. 将文件重命名为 'assist09_raw.csv' 并放到: {data_dir}/")
    print("4. 再次运行本脚本进行数据转换")

    return None


def convert_assist09_to_our_format(input_path, output_path):
    """
    将ASSIST09原始数据转换为我们项目需要的格式

    Args:
        input_path: ASSIST09原始CSV路径
        output_path: 输出CSV路径
    """
    print(f"\n{'='*60}")
    print("转换ASSIST09数据格式")
    print('='*60)

    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path, encoding='latin-1')

    print(f"\n原始数据信息:")
    print(f"  形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()[:10]}...")  # 只显示前10列

    # ASSIST09数据集关键列（根据实际情况调整）:
    # - user_id: 学生ID
    # - problem_id: 题目ID
    # - skill_id: 知识点ID
    # - correct: 0/1 表示答题正确性
    # - order_id: 答题顺序

    # 检查必要的列
    required_columns = {
        'user_id': ['user_id', 'student_id'],
        'problem_id': ['problem_id', 'question_id', 'item_id'],
        'skill_id': ['skill_id', 'concept_id', 'kc_id'],
        'correct': ['correct', 'is_correct'],
        'order_id': ['order_id', 'timestamp', 'row_id']
    }

    column_mapping = {}
    for target_col, possible_names in required_columns.items():
        found = False
        for name in possible_names:
            if name in df.columns:
                column_mapping[name] = target_col
                found = True
                print(f"  找到列: {name} -> {target_col}")
                break
        if not found:
            print(f"  ⚠ 警告: 未找到 {target_col} 对应的列")

    # 如果找不到关键列，显示所有列名供用户检查
    if len(column_mapping) < 4:
        print("\n可用列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        raise ValueError("缺少必要的列，请检查数据格式")

    # 提取和重命名列
    df_converted = df[[col for col in column_mapping.keys()]].copy()
    df_converted.columns = [column_mapping[col] for col in df_converted.columns]

    # 重命名为我们的标准列名
    final_mapping = {
        'user_id': 'student_id',
        'problem_id': 'question_id',
        'skill_id': 'concept_id',
        'correct': 'correct',
        'order_id': 'timestamp'
    }
    df_converted.rename(columns=final_mapping, inplace=True)

    # 数据清洗
    print(f"\n数据清洗...")

    # 1. 删除缺失值
    original_len = len(df_converted)
    df_converted = df_converted.dropna()
    print(f"  删除缺失值: {original_len} -> {len(df_converted)}")

    # 2. 确保correct是0/1
    if df_converted['correct'].dtype != int:
        df_converted['correct'] = df_converted['correct'].astype(int)
    df_converted = df_converted[df_converted['correct'].isin([0, 1])]

    # 3. 删除skill_id为空或负数的行
    if 'concept_id' in df_converted.columns:
        df_converted = df_converted[df_converted['concept_id'] >= 0]

    # 4. 重新编码ID（从0开始）
    print(f"\n重新编码ID...")
    for col in ['student_id', 'question_id', 'concept_id']:
        if col in df_converted.columns:
            unique_vals = sorted(df_converted[col].unique())
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_vals)}
            df_converted[col] = df_converted[col].map(id_mapping)
            print(f"  {col}: {len(unique_vals)} 个唯一值")

    # 5. 按学生和时间排序
    if 'timestamp' in df_converted.columns:
        df_converted = df_converted.sort_values(['student_id', 'timestamp'])
    else:
        # 如果没有时间戳，创建一个
        df_converted = df_converted.sort_values('student_id')
        df_converted['timestamp'] = df_converted.groupby('student_id').cumcount()

    # 重置索引
    df_converted = df_converted.reset_index(drop=True)

    # 保存
    df_converted.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("✓ 数据转换完成")
    print('='*60)
    print(f"  输出文件: {output_path}")
    print(f"  数据形状: {df_converted.shape}")
    print(f"\n数据集统计:")
    print(f"  学生数: {df_converted['student_id'].nunique()}")
    print(f"  题目数: {df_converted['question_id'].nunique()}")
    print(f"  知识点数: {df_converted['concept_id'].nunique()}")
    print(f"  总答题记录: {len(df_converted)}")
    print(f"  正确率: {df_converted['correct'].mean():.4f}")

    # 显示示例数据
    print(f"\n前5行数据:")
    print(df_converted.head())

    return df_converted


def main():
    """
    主函数：下载并转换ASSIST09数据
    """
    print("=" * 60)
    print("ASSIST09数据集下载和转换工具")
    print("=" * 60)

    # 步骤1: 下载数据
    raw_data_path = download_assist09_from_url()

    if raw_data_path is None or not os.path.exists(raw_data_path):
        print("\n请先手动下载数据集，然后再次运行本脚本")
        return

    # 步骤2: 转换数据格式
    output_path = './data/assist09.csv'
    try:
        df = convert_assist09_to_our_format(raw_data_path, output_path)

        print(f"\n{'='*60}")
        print("✅ 全部完成！")
        print('='*60)
        print(f"\n下一步操作:")
        print(f"1. 运行数据重新生成脚本:")
        print(f"   python regenerate_data.py")
        print(f"2. 运行基线实验:")
        print(f"   python experiments/run_baseline_experiments.py --datasets assist09 --models DKT --n_runs 1")

    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查数据文件格式是否正确")


if __name__ == "__main__":
    main()
