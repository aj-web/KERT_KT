"""
合并所有数据集到统一的processed_datasets.pkl文件

使用方法：
    python data/merge_datasets.py
"""

import os
import pickle


def merge_datasets():
    """合并所有处理后的数据集"""
    print("="*50)
    print("合并数据集")
    print("="*50)
    
    all_datasets = {}
    
    # 数据集列表
    datasets = {
        'assist09': 'data/assist09_processed.pkl',
        'ednet': 'data/ednet_processed.pkl',
        'junyi': 'data/junyi_processed.pkl'
    }
    
    # 加载并合并
    for name, filepath in datasets.items():
        if os.path.exists(filepath):
            print(f"\n加载 {name}...")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            all_datasets.update(data)
            print(f"  [OK] {name} 已加载")
            
            # 打印统计信息
            if name in data:
                dataset_info = data[name]
                print(f"    学生数: {dataset_info['n_students']}")
                print(f"    题目数: {dataset_info['n_questions']}")
                print(f"    知识点数: {dataset_info['n_concepts']}")
                print(f"    训练集: {len(dataset_info['train'])}")
        else:
            print(f"\n[WARNING] {name} 数据文件不存在: {filepath}")
            print(f"    请先运行: python data/process_{name}.py")
    
    # 保存合并后的数据
    if all_datasets:
        output_file = 'data/processed_datasets.pkl'
        print(f"\n保存合并数据到: {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(all_datasets, f)
        
        print("\n" + "="*50)
        print("[OK] 数据集合并完成")
        print("="*50)
        print(f"包含的数据集: {list(all_datasets.keys())}")
        print(f"输出文件: {output_file}")
    else:
        print("\n[ERROR] 没有找到任何数据集")
        print("请先处理至少一个数据集")


if __name__ == '__main__':
    merge_datasets()

