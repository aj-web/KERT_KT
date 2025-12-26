"""
快速测试KER-KT模型
用于验证模型是否能正常运行，不进行完整训练
"""

import torch
import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.kert_kt import KERKT, train_kert_kt
from experiments.run_experiment import get_dataset_config, load_processed_data, create_data_loaders


def quick_test_kert_kt(dataset_name='assist09', n_epochs=5, quick_mode=True):
    """
    快速测试KER-KT模型
    
    Args:
        dataset_name: 数据集名称
        n_epochs: 训练轮数（快速测试用少量轮数）
        quick_mode: 是否使用快速模式（减少数据量）
    """
    print(f"\n{'='*60}")
    print(f"快速测试 KER-KT 模型 - {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    # 获取数据集配置
    config = get_dataset_config(dataset_name)
    
    # 加载数据
    print("加载数据...")
    dataset_info = load_processed_data(dataset_name)
    concept_graph = torch.tensor(dataset_info['concept_graph'], dtype=torch.float32)
    
    print(f"数据集统计:")
    print(f"  问题数: {dataset_info['n_questions']}")
    print(f"  知识点数: {dataset_info['n_concepts']}")
    print(f"  学生数: {dataset_info['n_students']}")
    print(f"  训练样本: {len(dataset_info['train'])}")
    print(f"  验证样本: {len(dataset_info['val'])}")
    print(f"  测试样本: {len(dataset_info['test'])}")
    
    # 快速模式：使用更小的batch size和更少的数据
    if quick_mode:
        config['batch_size'] = min(config['batch_size'], 32)
        print(f"\n快速模式：batch_size={config['batch_size']}, epochs={n_epochs}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_info, config['batch_size'], config['max_seq_len']
    )
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 初始化模型
    print("\n初始化KER-KT模型...")
    print(f"配置参数:")
    print(f"  embed_dim: {config['embed_dim']}")
    print(f"  hidden_dim: {config['hidden_dim']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  alpha: {config['alpha']}, beta: {config['beta']}")
    
    model = KERKT(
        n_questions=dataset_info['n_questions'],
        n_concepts=dataset_info['n_concepts'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        alpha=config['alpha'],
        beta=config['beta'],
        lambda_decay=config['lambda_decay'],
        gamma=config['gamma'],
        lr_kt=config['lr_kt_pretrain'],
        lr_rl=config['lr_rl'],
        lambda_rl=config['lambda_rl']
    )
    model = model.to(device)
    concept_graph = concept_graph.to(device)
    
    # 测试前向传播
    print("\n测试前向传播...")
    try:
        # 获取一个batch
        batch = next(iter(train_loader))
        question_seq = batch['question_seq'].to(device)
        concept_seq = batch['concept_seq'].to(device)
        answer_seq = batch['answer_seq'].to(device)
        target_question = batch['target_question'].to(device)
        target_concept = batch['target_concept'].to(device)
        
        # 前向传播 - 注意KERKT的forward需要batch字典
        with torch.no_grad():
            batch_dict = {
                'question_seq': question_seq,
                'concept_seq': concept_seq,
                'answer_seq': answer_seq,
                'target_question': target_question,
                'target_concept': target_concept
            }
            predictions, hidden_states = model(batch_dict, concept_graph)
        
        print(f"  [OK] 前向传播成功")
        print(f"  预测形状: {predictions.shape}")
        print(f"  预测范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        
    except Exception as e:
        print(f"  [FAIL] 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试训练（少量轮数）
    print(f"\n开始快速训练测试（{n_epochs}轮）...")
    try:
        checkpoint_dir = os.path.join(project_root, 'checkpoints', dataset_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'kert_kt_quick_test.pt')
        
        train_kert_kt(
            model, train_loader, val_loader, concept_graph,
            n_epochs=n_epochs,
            patience=10,  # 快速测试时使用较大的patience
            checkpoint_path=checkpoint_path,
            lr_kt_pretrain=config['lr_kt_pretrain'],
            lr_kt_finetune=config['lr_kt_finetune']
        )
        
        print(f"  [OK] 训练完成")
        
    except Exception as e:
        print(f"  [FAIL] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试评估
    print("\n测试评估...")
    try:
        # 加载最佳模型
        if os.path.exists(checkpoint_path):
            model.load_model(checkpoint_path)
            print("  已加载最佳模型")
        
        # 在验证集上评估
        val_metrics = model.evaluate(val_loader, concept_graph)
        print(f"  验证集结果:")
        print(f"    AUC: {val_metrics['auc']:.4f}")
        print(f"    ACC: {val_metrics['acc']:.4f}")
        
        # 在测试集上评估
        test_metrics = model.evaluate(test_loader, concept_graph)
        print(f"  测试集结果:")
        print(f"    AUC: {test_metrics['auc']:.4f}")
        print(f"    ACC: {test_metrics['acc']:.4f}")
        
        print(f"\n  [OK] 评估完成")
        
        # 检查结果是否合理
        if test_metrics['auc'] > 0.5:
            print(f"  [OK] AUC > 0.5，模型正在学习")
        else:
            print(f"  [WARNING] AUC <= 0.5，模型可能存在问题")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='快速测试KER-KT模型')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'assist17', 'junyi'],
                       help='数据集名称')
    parser.add_argument('--n_epochs', type=int, default=5,
                       help='训练轮数（快速测试）')
    parser.add_argument('--full_test', action='store_true',
                       help='完整测试（不使用快速模式）')
    
    args = parser.parse_args()
    
    success = quick_test_kert_kt(
        dataset_name=args.dataset,
        n_epochs=args.n_epochs,
        quick_mode=not args.full_test
    )
    
    if success:
        print(f"\n{'='*60}")
        print("[OK] KER-KT模型测试通过！")
        print(f"{'='*60}\n")
        print("下一步建议：")
        print("1. 运行完整实验：python experiments/run_experiment.py")
        print("2. 运行基线对比：python experiments/run_baseline_experiments.py")
    else:
        print(f"\n{'='*60}")
        print("[FAIL] KER-KT模型测试失败，请检查错误信息")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

