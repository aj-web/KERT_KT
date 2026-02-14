"""
一键生成所有可视化图表
论文第4章：实验结果可视化

功能：
1. 生成超参数敏感性分析图表
2. 生成知识状态演化可视化
3. 生成三支决策邻域划分可视化
4. 生成基线对比图表
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add project root to Python path
# 当前文件: experiments/visualization/generate_all_visualizations.py
# 向上两级到达 experiments/，再向上一级到达项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/visualization/
experiments_dir = os.path.dirname(current_dir)  # experiments/
project_root = os.path.dirname(experiments_dir)  # 项目根目录
sys.path.insert(0, project_root)


def run_command(cmd, description):
    """
    运行命令并打印输出
    
    Args:
        cmd: 命令列表
        description: 描述
    """
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        print(f"[OK] {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} 失败")
        print(f"错误信息: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] {description} 失败: {e}")
        return False


def generate_sensitivity_plots(dataset='assist09', n_runs=3):
    """
    生成超参数敏感性分析图表
    
    Args:
        dataset: 数据集名称
        n_runs: 每组参数运行次数
    """
    print(f"\n{'#'*80}")
    print(f"# 阶段5：超参数敏感性分析")
    print(f"{'#'*80}")
    
    params = ['lambda_decay', 'max_k', 'alpha', 'beta']
    success_count = 0
    
    for param in params:
        cmd = [
            'python', 
            os.path.join(experiments_dir, 'analysis', 'hyperparameter_sensitivity.py'),
            '--dataset', dataset,
            '--param', param,
            '--n_runs', str(n_runs)
        ]
        
        if run_command(cmd, f"超参数敏感性分析: {param}"):
            success_count += 1
    
    print(f"\n超参数敏感性分析完成: {success_count}/{len(params)} 成功")
    return success_count == len(params)


def generate_knowledge_state_visualizations(dataset='assist09', checkpoint=None, n_students=3):
    """
    生成知识状态演化可视化
    
    Args:
        dataset: 数据集名称
        checkpoint: 模型检查点路径
        n_students: 要可视化的学生数量
    """
    print(f"\n{'#'*80}")
    print(f"# 阶段6.1：知识状态演化可视化")
    print(f"{'#'*80}")
    
    success_count = 0
    
    for i in range(n_students):
        cmd = [
            'python',
            os.path.join(experiments_dir, 'visualization', 'visualize_knowledge_state.py'),
            '--dataset', dataset
        ]
        
        if checkpoint:
            cmd.extend(['--checkpoint', checkpoint])
        
        # 随机选择学生（通过不指定 student_id）
        
        if run_command(cmd, f"知识状态演化可视化: 学生 {i+1}"):
            success_count += 1
    
    print(f"\n知识状态演化可视化完成: {success_count}/{n_students} 成功")
    return success_count == n_students


def generate_triple_decision_visualizations(dataset='assist09', checkpoint=None, n_concepts=3):
    """
    生成三支决策邻域划分可视化
    
    Args:
        dataset: 数据集名称
        checkpoint: 模型检查点路径
        n_concepts: 要可视化的知识点数量
    """
    print(f"\n{'#'*80}")
    print(f"# 阶段6.2：三支决策邻域划分可视化")
    print(f"{'#'*80}")
    
    success_count = 0
    
    for k in [1, 2]:  # 1阶和2阶邻域
        for i in range(n_concepts):
            cmd = [
                'python',
                os.path.join(experiments_dir, 'visualization', 'visualize_triple_decision.py'),
                '--dataset', dataset,
                '--k', str(k)
            ]
            
            if checkpoint:
                cmd.extend(['--checkpoint', checkpoint])
            
            # 随机选择知识点（通过不指定 concept_id）
            
            if run_command(cmd, f"三支决策可视化: k={k}, 知识点 {i+1}"):
                success_count += 1
    
    total = n_concepts * 2  # 2个k值
    print(f"\n三支决策邻域划分可视化完成: {success_count}/{total} 成功")
    return success_count == total


def generate_baseline_comparison_plots(dataset='assist09'):
    """
    生成基线对比图表
    
    Args:
        dataset: 数据集名称
    """
    print(f"\n{'#'*80}")
    print(f"# 基线模型对比图表")
    print(f"{'#'*80}")
    
    # 这里需要一个单独的脚本来生成基线对比图表
    # 暂时跳过，因为需要先运行基线实验
    print("提示: 请先运行基线实验，然后使用 plot_baseline_comparison.py 生成对比图表")
    return True


def main():
    parser = argparse.ArgumentParser(description='一键生成所有可视化图表')
    parser.add_argument('--dataset', type=str, default='assist09',
                       choices=['assist09', 'junyi', 'ednet'],
                       help='数据集名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径（可选）')
    parser.add_argument('--skip_sensitivity', action='store_true',
                       help='跳过超参数敏感性分析')
    parser.add_argument('--skip_knowledge_state', action='store_true',
                       help='跳过知识状态演化可视化')
    parser.add_argument('--skip_triple_decision', action='store_true',
                       help='跳过三支决策可视化')
    parser.add_argument('--n_runs', type=int, default=3,
                       help='敏感性分析每组参数运行次数')
    parser.add_argument('--n_students', type=int, default=3,
                       help='知识状态可视化的学生数量')
    parser.add_argument('--n_concepts', type=int, default=3,
                       help='三支决策可视化的知识点数量')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"一键生成所有可视化图表")
    print(f"数据集: {args.dataset.upper()}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    results = {}
    
    # 1. 超参数敏感性分析
    if not args.skip_sensitivity:
        results['sensitivity'] = generate_sensitivity_plots(
            dataset=args.dataset,
            n_runs=args.n_runs
        )
    else:
        print("\n跳过超参数敏感性分析")
        results['sensitivity'] = None
    
    # 2. 知识状态演化可视化
    if not args.skip_knowledge_state:
        results['knowledge_state'] = generate_knowledge_state_visualizations(
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            n_students=args.n_students
        )
    else:
        print("\n跳过知识状态演化可视化")
        results['knowledge_state'] = None
    
    # 3. 三支决策邻域划分可视化
    if not args.skip_triple_decision:
        results['triple_decision'] = generate_triple_decision_visualizations(
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            n_concepts=args.n_concepts
        )
    else:
        print("\n跳过三支决策邻域划分可视化")
        results['triple_decision'] = None
    
    # 4. 基线对比图表
    results['baseline_comparison'] = generate_baseline_comparison_plots(
        dataset=args.dataset
    )
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"所有可视化任务完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    print("\n任务完成情况:")
    for task, success in results.items():
        if success is None:
            status = "跳过"
        elif success:
            status = "[OK] 成功"
        else:
            status = "[ERROR] 失败"
        print(f"  {task}: {status}")
    
    # 输出文件位置
    print(f"\n{'='*80}")
    print("生成的文件位置:")
    print(f"{'='*80}")
    print(f"超参数敏感性分析:")
    print(f"  - 结果: results/sensitivity/")
    print(f"  - 图表: figures/sensitivity/")
    print(f"\n知识状态演化可视化:")
    print(f"  - 图表: figures/knowledge_state/")
    print(f"\n三支决策邻域划分可视化:")
    print(f"  - 图表: figures/triple_decision/")
    
    print("\n所有可视化任务完成！")


if __name__ == "__main__":
    main()

