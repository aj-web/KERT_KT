"""
题目增强模块 (Question Enhancement Module)

实现论文公式(0-6)(0-7)：使用标准scaled dot-product attention增强知识点表征

论文：张慧玲-论文0201.txt 第3章
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QuestionEnhancement(nn.Module):
    """
    题目增强模块
    
    论文公式(0-6)(0-7)：
    α'_i = softmax_i((q_t·W_q · (c_i·W_k)^T) / √d_c)
    c'_i = c_i + α'_i·(q_t·W_v)
    
    使用标准的scaled dot-product attention机制对知识点表征进行增强
    """
    
    def __init__(self, embed_dim, dropout=0.1):
        """
        Args:
            embed_dim: 嵌入维度 d_c
            dropout: Dropout率
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        
        # Query, Key, Value投影矩阵 (论文公式0-6)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier均匀初始化"""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
    
    def forward(self, question_embed, concept_embeds, q_matrix=None):
        """
        前向传播
        
        Args:
            question_embed: [batch_size, embed_dim] 或 [embed_dim] 题目嵌入
            concept_embeds: [n_concepts, embed_dim] 知识点嵌入
            q_matrix: [batch_size, n_concepts] 或 [n_concepts] Q矩阵（可选）
                     如果提供，只对相关知识点计算attention
        
        Returns:
            enhanced_concepts: [n_concepts, embed_dim] 或 [batch_size, n_concepts, embed_dim]
                             增强后的知识点表征
        """
        # 处理维度
        if question_embed.dim() == 1:
            question_embed = question_embed.unsqueeze(0)  # [1, embed_dim]
            single_question = True
        else:
            single_question = False
        
        batch_size = question_embed.size(0)
        n_concepts = concept_embeds.size(0)
        
        # Query, Key, Value投影 (论文公式0-6)
        query = self.W_q(question_embed)  # [batch_size, embed_dim]
        keys = self.W_k(concept_embeds)   # [n_concepts, embed_dim]
        values = self.W_v(question_embed) # [batch_size, embed_dim]
        
        # Scaled dot-product attention (论文公式0-6)
        # scores: [batch_size, n_concepts]
        scores = torch.matmul(query, keys.T) / self.scale
        
        # 如果提供Q矩阵，只对相关知识点计算attention
        if q_matrix is not None:
            if q_matrix.dim() == 1:
                q_matrix = q_matrix.unsqueeze(0)  # [1, n_concepts]
            # 将不相关知识点的分数设为很小的值
            mask = (q_matrix == 0)
            scores = scores.masked_fill(mask, -1e9)
        
        # Softmax归一化
        attention = F.softmax(scores, dim=-1)  # [batch_size, n_concepts]
        attention = self.dropout(attention)
        
        # 加性增强 (论文公式0-7)
        # c'_i = c_i + α'_i * (q_t · W_v)
        if batch_size == 1:
            # 单个题目的情况
            enhanced = concept_embeds + attention.T * values  # [n_concepts, embed_dim]
            if single_question:
                return enhanced
            else:
                return enhanced.unsqueeze(0)  # [1, n_concepts, embed_dim]
        else:
            # 批量题目的情况
            # enhanced: [batch_size, n_concepts, embed_dim]
            enhanced = concept_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            enhanced = enhanced + attention.unsqueeze(-1) * values.unsqueeze(1)
            return enhanced
    
    def compute_attention_weights(self, question_embed, concept_embeds, q_matrix=None):
        """
        仅计算attention权重（用于可视化）
        
        Args:
            question_embed: [batch_size, embed_dim] 或 [embed_dim]
            concept_embeds: [n_concepts, embed_dim]
            q_matrix: [batch_size, n_concepts] 或 [n_concepts] (可选)
        
        Returns:
            attention: [batch_size, n_concepts] 或 [n_concepts] attention权重
        """
        if question_embed.dim() == 1:
            question_embed = question_embed.unsqueeze(0)
            single_question = True
        else:
            single_question = False
        
        # 计算attention
        query = self.W_q(question_embed)
        keys = self.W_k(concept_embeds)
        scores = torch.matmul(query, keys.T) / self.scale
        
        if q_matrix is not None:
            if q_matrix.dim() == 1:
                q_matrix = q_matrix.unsqueeze(0)
            mask = (q_matrix == 0)
            scores = scores.masked_fill(mask, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        
        if single_question:
            return attention.squeeze(0)  # [n_concepts]
        else:
            return attention  # [batch_size, n_concepts]


class QuestionEnhancementSimplified(nn.Module):
    """
    题目增强模块的简化版本（小论文版本）
    
    使用单个投影矩阵W_a，计算更简单但效果相近
    
    α'_i = softmax_i(q_t·c_i·W_a / √d_c)
    c'_i = c_i + α'_i·c_i
    """
    
    def __init__(self, embed_dim, dropout=0.1):
        """
        Args:
            embed_dim: 嵌入维度
            dropout: Dropout率
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        
        # 单个投影矩阵
        self.W_a = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W_a.weight)
    
    def forward(self, question_embed, concept_embeds, q_matrix=None):
        """
        简化版前向传播
        
        Args:
            question_embed: [batch_size, embed_dim] 或 [embed_dim]
            concept_embeds: [n_concepts, embed_dim]
            q_matrix: [batch_size, n_concepts] 或 [n_concepts] (可选)
        
        Returns:
            enhanced_concepts: [n_concepts, embed_dim] 或 [batch_size, n_concepts, embed_dim]
        """
        if question_embed.dim() == 1:
            question_embed = question_embed.unsqueeze(0)
            single_question = True
        else:
            single_question = False
        
        batch_size = question_embed.size(0)
        
        # 简化的attention计算
        transformed_concepts = self.W_a(concept_embeds)  # [n_concepts, embed_dim]
        scores = torch.matmul(question_embed, transformed_concepts.T) / self.scale
        
        if q_matrix is not None:
            if q_matrix.dim() == 1:
                q_matrix = q_matrix.unsqueeze(0)
            mask = (q_matrix == 0)
            scores = scores.masked_fill(mask, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 加性增强
        if batch_size == 1:
            enhanced = concept_embeds + attention.T * concept_embeds
            if single_question:
                return enhanced
            else:
                return enhanced.unsqueeze(0)
        else:
            enhanced = concept_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            enhanced = enhanced + attention.unsqueeze(-1) * concept_embeds.unsqueeze(0)
            return enhanced


if __name__ == '__main__':
    # 简单测试
    print("="*50)
    print("题目增强模块测试")
    print("="*50)
    
    # 参数
    batch_size = 2
    n_concepts = 10
    embed_dim = 128
    
    # 创建模块
    enhancer = QuestionEnhancement(embed_dim)
    
    # 测试数据
    question_embed = torch.randn(batch_size, embed_dim)
    concept_embeds = torch.randn(n_concepts, embed_dim)
    q_matrix = torch.randint(0, 2, (batch_size, n_concepts)).float()
    
    print(f"\n输入:")
    print(f"  question_embed: {question_embed.shape}")
    print(f"  concept_embeds: {concept_embeds.shape}")
    print(f"  q_matrix: {q_matrix.shape}")
    
    # 前向传播
    enhanced = enhancer(question_embed, concept_embeds, q_matrix)
    print(f"\n输出:")
    print(f"  enhanced: {enhanced.shape}")
    
    # 测试attention权重
    attention = enhancer.compute_attention_weights(question_embed, concept_embeds, q_matrix)
    print(f"  attention: {attention.shape}")
    print(f"  attention sum: {attention.sum(dim=-1)}")  # 应该接近1
    
    # 测试单个题目
    print("\n单个题目测试:")
    single_q = torch.randn(embed_dim)
    single_qm = torch.randint(0, 2, (n_concepts,)).float()
    enhanced_single = enhancer(single_q, concept_embeds, single_qm)
    print(f"  输入: {single_q.shape}, 输出: {enhanced_single.shape}")
    
    # 测试简化版本
    print("\n简化版本测试:")
    enhancer_simple = QuestionEnhancementSimplified(embed_dim)
    enhanced_simple = enhancer_simple(question_embed, concept_embeds, q_matrix)
    print(f"  输出: {enhanced_simple.shape}")
    
    print("\n" + "="*50)
    print("测试通过！")
    print("="*50)

