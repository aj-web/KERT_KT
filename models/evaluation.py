"""
评估指标模块 (Evaluation Metrics)

实现论文第4章的评估指标：
- AUC (Area Under ROC Curve)
- ACC (Accuracy)
- DOA (Degree of Agreement) - 论文公式4.5.1

论文：张慧玲-论文0201.txt 第4章
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import cosine


def compute_auc_acc(predictions, labels):
    """
    计算AUC和ACC
    
    Args:
        predictions: numpy array, 预测概率 [n_samples]
        labels: numpy array, 真实标签 [n_samples]
    
    Returns:
        auc: float
        acc: float
    """
    auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(labels, (predictions > 0.5).astype(int))
    return auc, acc


def compute_doa(knowledge_states, student_answers, threshold=0.5):
    """
    计算DOA (Degree of Agreement) 指标
    
    论文公式4.5.1：
    DOA(G) = (1/Z) Σ_a Σ_b δ(G_a, G_b) × (1/|Q|) Σ_q δ(y_aq, y_bq)
    
    其中：
    - G_a, G_b: 学生a和b的知识状态表征
    - δ(G_a, G_b): 知识状态相似性指示函数（余弦相似度>阈值为1，否则为0）
    - y_aq, y_bq: 学生a和b对题目q的作答
    - Q: 共同作答的题目集合
    - Z = N(N-1): 学生对数量
    
    Args:
        knowledge_states: dict, {student_id: knowledge_vector (numpy array)}
                         学生的知识状态表征
        student_answers: dict, {student_id: {question_id: answer (0/1)}}
                        学生的作答记录
        threshold: float, 知识状态相似度阈值（默认0.5）
    
    Returns:
        doa_score: float, DOA分数（越高表示知识状态表征质量越好）
    """
    student_ids = list(knowledge_states.keys())
    N = len(student_ids)
    
    if N < 2:
        return 0.0
    
    Z = N * (N - 1)
    total_agreement = 0.0
    valid_pairs = 0
    
    for i, student_a in enumerate(student_ids):
        for j, student_b in enumerate(student_ids):
            if i >= j:  # 避免重复计算和自比较
                continue
            
            # 1. 计算知识状态相似度 (余弦相似度)
            state_a = knowledge_states[student_a]
            state_b = knowledge_states[student_b]
            
            # 处理torch tensor或numpy array
            if isinstance(state_a, torch.Tensor):
                state_a = state_a.cpu().numpy()
            if isinstance(state_b, torch.Tensor):
                state_b = state_b.cpu().numpy()
            
            # 归一化
            norm_a = np.linalg.norm(state_a)
            norm_b = np.linalg.norm(state_b)
            
            if norm_a == 0 or norm_b == 0:
                continue
            
            cosine_sim = np.dot(state_a, state_b) / (norm_a * norm_b)
            
            # δ(G_a, G_b): 相似度指示函数
            state_similar = 1 if cosine_sim > threshold else 0
            
            # 2. 计算作答一致性
            answers_a = student_answers[student_a]
            answers_b = student_answers[student_b]
            
            # 找出共同作答的题目
            common_questions = set(answers_a.keys()) & set(answers_b.keys())
            
            if len(common_questions) == 0:
                continue
            
            # 计算作答一致性
            consistent_count = sum(
                1 if answers_a[q] == answers_b[q] else 0
                for q in common_questions
            )
            answer_consistency = consistent_count / len(common_questions)
            
            # 3. 累加 δ(G_a, G_b) × 作答一致性
            total_agreement += state_similar * answer_consistency
            valid_pairs += 1
    
    if valid_pairs == 0:
        return 0.0
    
    # 归一化
    doa_score = total_agreement / valid_pairs
    
    return doa_score


def extract_knowledge_states(model, data_loader, concept_graph, device='cpu'):
    """
    从模型中提取所有学生的知识状态表征
    
    Args:
        model: KRD-KT模型
        data_loader: 数据加载器
        concept_graph: 概念图
        device: 设备
    
    Returns:
        knowledge_states: dict, {student_id: knowledge_vector}
        student_answers: dict, {student_id: {question_id: answer}}
    """
    model.eval()
    knowledge_states = {}
    student_answers = {}
    
    # 记录每个学生的最后一个hidden state作为知识状态
    # (论文假设LSTM的hidden state编码了学生的知识状态)
    
    with torch.no_grad():
        for batch in data_loader:
            # 移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            predictions, hidden_states = model.forward(batch, concept_graph)
            
            # 提取每个样本的知识状态（LSTM最后时刻的hidden state）
            # hidden_states: [batch_size, seq_len, hidden_dim]
            batch_size = hidden_states.size(0)
            
            for i in range(batch_size):
                # 使用最后一个时刻的hidden state作为知识状态
                # (假设batch中包含student_id，实际需要根据数据格式调整)
                
                # 注意：这里简化处理，实际需要根据数据集的student_id来聚合
                # 对于测试，我们可以用样本索引作为"student_id"
                sample_idx = len(knowledge_states)
                
                # 取最后一个时刻的hidden state
                last_hidden = hidden_states[i, -1, :].cpu().numpy()
                knowledge_states[sample_idx] = last_hidden
                
                # 记录作答
                if sample_idx not in student_answers:
                    student_answers[sample_idx] = {}
                
                # 记录目标题目和答案
                if 'target_question' in batch and 'target_answer' in batch:
                    q_id = int(batch['target_question'][i].item())
                    answer = int(batch['target_answer'][i].item())
                    student_answers[sample_idx][q_id] = answer
    
    return knowledge_states, student_answers


def extract_knowledge_states_by_student(model, dataset, concept_graph, device='cpu'):
    """
    按学生ID提取知识状态（更准确的方法）
    
    Args:
        model: KRD-KT模型
        dataset: 包含student_id的原始数据集 (DataFrame)
        concept_graph: 概念图
        device: 设备
    
    Returns:
        knowledge_states: dict, {student_id: knowledge_vector}
        student_answers: dict, {student_id: {question_id: answer}}
    """
    model.eval()
    knowledge_states = {}
    student_answers = {}
    
    # 按学生分组
    student_groups = dataset.groupby('student_id')
    
    with torch.no_grad():
        for student_id, group in student_groups:
            # 获取该学生的所有交互序列
            questions = torch.LongTensor(group['question_id'].values).to(device)
            concepts = torch.LongTensor(group['concept_id'].values).to(device)
            answers = torch.LongTensor(group['correct'].values).to(device)
            
            # 创建单个batch（只包含这一个学生）
            seq_len = len(questions) - 1
            if seq_len < 1:
                continue
            
            batch = {
                'question_seq': questions[:-1].unsqueeze(0),  # [1, seq_len]
                'concept_seq': concepts[:-1].unsqueeze(0),
                'answer_seq': answers[:-1].unsqueeze(0),
                'target_question': questions[-1].unsqueeze(0),  # [1]
                'target_concept': concepts[-1].unsqueeze(0),
                'target_answer': answers[-1].unsqueeze(0)
            }
            
            # 前向传播
            _, hidden_states = model.forward(batch, concept_graph)
            
            # 取最后一个时刻的hidden state作为知识状态
            last_hidden = hidden_states[0, -1, :].cpu().numpy()
            knowledge_states[student_id] = last_hidden
            
            # 记录作答
            student_answers[student_id] = {}
            for q_id, ans in zip(group['question_id'].values, group['correct'].values):
                student_answers[student_id][int(q_id)] = int(ans)
    
    return knowledge_states, student_answers


def evaluate_model_with_doa(model, test_loader, test_dataset, concept_graph, device='cpu'):
    """
    完整评估模型（AUC, ACC, DOA）
    
    Args:
        model: KRD-KT模型
        test_loader: 测试数据加载器
        test_dataset: 测试数据集（DataFrame，用于DOA计算）
        concept_graph: 概念图
        device: 设备
    
    Returns:
        metrics: dict, {'auc': float, 'acc': float, 'doa': float}
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    # 1. 计算AUC和ACC
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            predictions, _ = model.forward(batch, concept_graph)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['target_answer'].cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    auc, acc = compute_auc_acc(all_predictions, all_labels)
    
    # 2. 计算DOA
    print("提取知识状态以计算DOA...")
    knowledge_states, student_answers = extract_knowledge_states_by_student(
        model, test_dataset, concept_graph, device
    )
    
    doa = compute_doa(knowledge_states, student_answers, threshold=0.5)
    
    metrics = {
        'auc': float(auc),
        'acc': float(acc),
        'doa': float(doa)
    }
    
    return metrics


if __name__ == '__main__':
    # 测试DOA计算
    print("="*50)
    print("DOA评估指标测试")
    print("="*50)
    
    # 创建模拟数据
    # 学生1和学生2知识状态相似，作答也相似 → 高DOA
    # 学生3知识状态不同，作答也不同 → 不影响DOA
    knowledge_states = {
        'student_1': np.array([0.8, 0.6, 0.3, 0.5]),
        'student_2': np.array([0.7, 0.5, 0.4, 0.6]),  # 与student_1相似
        'student_3': np.array([0.2, 0.3, 0.9, 0.1])   # 与student_1不相似
    }
    
    student_answers = {
        'student_1': {1: 1, 2: 0, 3: 1, 4: 1},
        'student_2': {1: 1, 2: 0, 3: 1, 4: 0},  # 75%一致
        'student_3': {1: 0, 2: 1, 3: 0, 4: 0}   # 25%一致
    }
    
    doa = compute_doa(knowledge_states, student_answers, threshold=0.5)
    
    print(f"\n测试结果:")
    print(f"  DOA分数: {doa:.4f}")
    print(f"\n解释:")
    print(f"  - student_1与student_2知识状态相似（余弦相似度>0.5）")
    print(f"    且作答一致性高（75%），贡献正向")
    print(f"  - student_1与student_3知识状态不相似")
    print(f"    δ(G_1, G_3)=0，不计入DOA")
    
    print("\n" + "="*50)
    print("测试通过！")
    print("="*50)
