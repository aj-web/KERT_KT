"""
使用EduData下载和处理真实数据集
"""
import os
import subprocess
import pandas as pd
import numpy as np
import pickle

def download_edudata_datasets():
    """
    使用EduData CLI下载ASSIST09和ASSIST17数据集
    """
    print("=" * 60)
    print("安装EduData...")
    print("=" * 60)

    # 安装EduData
    try:
        subprocess.run(['pip', 'install', 'EduData'], check=True)
        print("✓ EduData安装成功")
    except:
        print("⚠ EduData安装失败，请手动安装: pip install EduData")
        return False

    print("\n" + "=" * 60)
    print("下载ASSIST09数据集...")
    print("=" * 60)

    # 下载ASSIST09
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    try:
        subprocess.run([
            'edudata', 'download', 'assistment-2009-2010-skill', data_dir
        ], check=True)
        print("✓ ASSIST09下载成功")
    except:
        print("⚠ 下载失败，请检查网络或手动下载")
        return False

    return True


def convert_edudata_to_our_format(edudata_path, output_path):
    """
    将EduData格式转换为我们项目需要的格式

    Args:
        edudata_path: EduData下载的数据路径
        output_path: 输出CSV路径
    """
    print(f"\n正在转换数据格式: {edudata_path}")

    # 读取EduData格式的数据
    # EduData可能是CSV格式，包含: user_id, problem_id, skill_id, correct等列
    df = pd.read_csv(edudata_path)

    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 转换为我们的格式
    # 我们需要的列: student_id, question_id, concept_id, correct, timestamp

    # 映射列名
    column_mapping = {
        'user_id': 'student_id',
        'problem_id': 'question_id',
        'skill_id': 'concept_id',
        'correct': 'correct',
        'order_id': 'timestamp'  # 使用order_id作为时间顺序
    }

    # 检查哪些列存在
    available_columns = {}
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            available_columns[old_col] = new_col
        else:
            print(f"⚠ 警告：列 '{old_col}' 不存在")

    # 重命名列
    df_converted = df[list(available_columns.keys())].copy()
    df_converted.columns = list(available_columns.values())

    # 如果没有timestamp列，创建一个
    if 'timestamp' not in df_converted.columns:
        print("创建timestamp列...")
        df_converted['timestamp'] = range(len(df_converted))

    # 重新编码ID（确保从0开始）
    print("\n重新编码ID...")
    for col in ['student_id', 'question_id', 'concept_id']:
        if col in df_converted.columns:
            unique_vals = df_converted[col].unique()
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            df_converted[col] = df_converted[col].map(mapping)
            print(f"  {col}: {len(unique_vals)} 个唯一值")

    # 保存转换后的数据
    df_converted.to_csv(output_path, index=False)

    print(f"\n✓ 数据转换完成")
    print(f"  保存到: {output_path}")
    print(f"  数据形状: {df_converted.shape}")
    print(f"  学生数: {df_converted['student_id'].nunique()}")
    print(f"  题目数: {df_converted['question_id'].nunique()}")
    print(f"  概念数: {df_converted['concept_id'].nunique()}")
    print(f"  正确率: {df_converted['correct'].mean():.3f}")

    return df_converted


if __name__ == "__main__":
    print("=" * 60)
    print("EduData数据集下载和转换工具")
    print("=" * 60)

    # 步骤1：下载数据
    # success = download_edudata_datasets()
    # if not success:
    #     print("\n请手动下载数据集")
    #     exit(1)

    # 步骤2：查找下载的数据文件
    # EduData通常会下载到 data/assistment-2009-2010-skill 目录
    # 需要根据实际下载的文件名调整

    print("\n请按以下步骤操作：")
    print("1. 安装EduData: pip install EduData")
    print("2. 下载数据: edudata download assistment-2009-2010-skill ./data")
    print("3. 查看下载的文件，找到主数据文件（通常是.csv或.txt）")
    print("4. 修改本脚本，指定正确的文件路径")
    print("5. 运行转换: python edudata_converter.py")
