"""
生成知识状态演化和三支决策可视化图表
使用mock数据，无需运行实验
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = r"E:\IDAEWOREKSPACE\demo"
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def plot_knowledge_state_evolution():
    """生成知识状态演化可视化 (图4-6)"""
    print("\n生成知识状态演化可视化...")
    
    # Mock数据：3个知识点在10个时间步的掌握概率
    n_concepts = 5
    n_timesteps = 12
    
    # 模拟学生知识掌握情况：逐渐提升
    np.random.seed(42)
    knowledge_states = np.zeros((n_concepts, n_timesteps))
    for i in range(n_concepts):
        base = 0.2 + i * 0.1  # 不同知识点初始掌握程度不同
        for t in range(n_timesteps):
            # 添加随机波动和上升趋势
            knowledge_states[i, t] = min(0.95, base + t * 0.06 + np.random.uniform(-0.05, 0.05))
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制热力图
    concept_names = [f'知识点{i+1}' for i in range(n_concepts)]
    timesteps = [f'T{t+1}' for t in range(n_timesteps)]
    
    im = ax.imshow(knowledge_states, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 设置轴标签
    ax.set_xticks(np.arange(n_timesteps))
    ax.set_yticks(np.arange(n_concepts))
    ax.set_xticklabels(timesteps)
    ax.set_yticklabels(concept_names)
    
    # 添加数值标签
    for i in range(n_concepts):
        for j in range(n_timesteps):
            text = ax.text(j, i, f'{knowledge_states[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('图4-6 学生知识状态演化热力图', fontsize=14)
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('知识点', fontsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('掌握概率', fontsize=11)
    
    # 标记答题事件
    answer_events = [(2, 3, 1), (4, 1, 0), (6, 2, 1), (8, 4, 1), (10, 0, 1)]
    for t, c, correct in answer_events:
        if t < n_timesteps and c < n_concepts:
            marker = 'o' if correct == 1 else 'x'
            color = 'blue' if correct == 1 else 'red'
            ax.scatter(t, c, marker=marker, s=100, color=color, edgecolors='black', linewidths=1.5, zorder=5)
    
    plt.tight_layout()
    
    ensure_dir(os.path.join(FIGURES_DIR, 'knowledge_state'))
    fig_path = os.path.join(FIGURES_DIR, 'knowledge_state', 'knowledge_evolution_student_mock_assist09.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    # 第二张图：知识状态随时间变化曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_concepts))
    for i in range(n_concepts):
        ax.plot(range(n_timesteps), knowledge_states[i], 'o-', 
                label=concept_names[i], color=colors[i], linewidth=2, markersize=6)
    
    ax.set_title('图4-6 知识点掌握概率随时间变化', fontsize=14)
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('掌握概率', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', ncol=3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'knowledge_state', 'knowledge_curve_student_mock_assist09.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    return True


def plot_triple_decision_neighborhood():
    """生成三支决策邻域划分可视化 (图4-7)"""
    print("\n生成三支决策邻域划分可视化...")
    
    # Mock数据：5个概念的一阶和二阶邻域
    # 每个概念有：正域(P)、边界域(B)、负域(N)的概念数量
    concepts = ['概念1', '概念2', '概念3', '概念4', '概念5']
    
    # 一阶邻域数据
    k1_pos = [8, 6, 7, 5, 9]
    k1_bound = [3, 4, 3, 5, 2]
    k1_neg = [4, 5, 5, 5, 4]
    
    # 二阶邻域数据
    k2_pos = [15, 12, 14, 10, 16]
    k2_bound = [8, 9, 7, 10, 6]
    k2_neg = [7, 9, 9, 10, 8]
    
    x = np.arange(len(concepts))
    width = 0.25
    
    # 图4-7a: 一阶邻域
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, k1_pos, width, label='正域(P)', color='#2ecc71')
    bars2 = ax.bar(x, k1_bound, width, label='边界域(B)', color='#f39c12')
    bars3 = ax.bar(x + width, k1_neg, width, label='负域(N)', color='#e74c3c')
    
    ax.set_ylabel('概念数量', fontsize=12)
    ax.set_title('图4-7(a) 一阶邻域三支决策划分', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(concepts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    ensure_dir(os.path.join(FIGURES_DIR, 'triple_decision'))
    fig_path = os.path.join(FIGURES_DIR, 'triple_decision', 'triple_decision_k1_mock_assist09.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    # 图4-7b: 二阶邻域
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, k2_pos, width, label='正域(P)', color='#2ecc71')
    bars2 = ax.bar(x, k2_bound, width, label='边界域(B)', color='#f39c12')
    bars3 = ax.bar(x + width, k2_neg, width, label='负域(N)', color='#e74c3c')
    
    ax.set_ylabel('概念数量', fontsize=12)
    ax.set_title('图4-7(b) 二阶邻域三支决策划分', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(concepts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'triple_decision', 'triple_decision_k2_mock_assist09.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    # 图4-7c: 饼图展示整体分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 一阶邻域整体分布
    total_k1 = sum(k1_pos + k1_bound + k1_neg)
    sizes_k1 = [sum(k1_pos)/total_k1*100, sum(k1_bound)/total_k1*100, sum(k1_neg)/total_k1*100]
    labels = ['正域(P)', '边界域(B)', '负域(N)']
    colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax1.pie(sizes_k1, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax1.set_title('一阶邻域整体分布', fontsize=13)
    
    # 二阶邻域整体分布
    total_k2 = sum(k2_pos + k2_bound + k2_neg)
    sizes_k2 = [sum(k2_pos)/total_k2*100, sum(k2_bound)/total_k2*100, sum(k2_neg)/total_k2*100]
    
    ax2.pie(sizes_k2, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('二阶邻域整体分布', fontsize=13)
    
    plt.suptitle('图4-7(c) 三支决策邻域分布比例', fontsize=14, y=1.02)
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'triple_decision', 'triple_decision_pie_mock_assist09.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  [OK] 保存: {fig_path}")
    plt.close()
    
    return True


def main():
    """主函数"""
    print("="*60)
    print("生成知识状态演化和三支决策可视化")
    print("="*60)
    
    # 1. 生成知识状态演化图表
    plot_knowledge_state_evolution()
    
    # 2. 生成三支决策邻域划分图表
    plot_triple_decision_neighborhood()
    
    print("\n" + "="*60)
    print("可视化图表生成完成！")
    print(f"图表保存位置: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

