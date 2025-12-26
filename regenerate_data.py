"""
重新生成合成数据（使用修复后的逻辑）
"""
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.data_loader import KTDataLoader

# 创建数据加载器
loader = KTDataLoader()

print("=" * 60)
print("重新生成ASSIST09数据（使用修复后的IRT模型）")
print("=" * 60)

# 删除旧数据
data_dir = os.path.join(project_root, 'data')
old_csv = os.path.join(data_dir, 'assist09.csv')
old_pkl = os.path.join(data_dir, 'processed_datasets.pkl')

if os.path.exists(old_csv):
    os.remove(old_csv)
    print(f"✓ 已删除旧数据: {old_csv}")

if os.path.exists(old_pkl):
    os.remove(old_pkl)
    print(f"✓ 已删除旧缓存: {old_pkl}")

print("\n开始生成新数据...")
print("-" * 60)

# 重新生成并预处理数据
dataset_info = loader.preprocess_data('assist09', force_download=True)

print("\n" + "=" * 60)
print("数据生成完成！统计信息：")
print("=" * 60)
print(f"Questions: {dataset_info['n_questions']}")
print(f"Concepts: {dataset_info['n_concepts']}")
print(f"Students: {dataset_info['n_students']}")
print(f"Train samples: {len(dataset_info['train'])}")
print(f"Val samples: {len(dataset_info['val'])}")
print(f"Test samples: {len(dataset_info['test'])}")
print(f"\n训练数据标签分布: {dataset_info['train']['correct'].mean():.3f}")
print("\n✅ 数据重新生成成功！现在可以运行实验了。")
