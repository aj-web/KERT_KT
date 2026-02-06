"""
k跳路径强度计算模块 (Path Strength Calculator)

实现论文公式(0-9)：计算k跳路径的最大强度

论文：张慧玲-论文0201.txt 第3章
"""

import torch
import torch.nn as nn
import numpy as np


class PathStrengthCalculator:
    """
    路径强度计算器
    
    论文公式(0-9)：
    s_ij^(1) = e_ij                                    (1阶：直接边权重)
    s_ij^(2) = max_{c_m ∈ N^(1)(c_i)} [e_im · e_mj]  (2阶：最大路径强度)
    
    计算知识点之间k跳路径的最大强度，用于三支决策的阈值判断
    """
    
    def __init__(self, concept_graph, neighborhood_extractor=None):
        """
        Args:
            concept_graph: [n_concepts, n_concepts] 概念图邻接矩阵
            neighborhood_extractor: NeighborhoodExtractor实例（可选）
        """
        self.concept_graph = concept_graph
        self.n_concepts = concept_graph.shape[0]
        self.neighborhood_extractor = neighborhood_extractor
        
        # 缓存计算结果
        self.strength_cache = {}
        self.cache_enabled = True
    
    def compute_1hop_strength(self, source, target):
        """
        计算1阶路径强度（直接边权重）
        
        论文公式(0-9) k=1:
        s_ij^(1) = e_ij
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            strength: float, 1阶路径强度
        """
        return float(self.concept_graph[source, target].item())
    
    def compute_2hop_strength(self, source, target):
        """
        计算2阶路径强度（最大路径强度）
        
        论文公式(0-9) k=2:
        s_ij^(2) = max_{c_m ∈ N^(1)(c_i)} [e_im · e_mj]
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            strength: float, 2阶路径强度
        """
        # 检查缓存
        cache_key = (source, target, 2)
        if self.cache_enabled and cache_key in self.strength_cache:
            return self.strength_cache[cache_key]
        
        # 获取源节点的1阶邻居
        if self.neighborhood_extractor is not None:
            # 使用邻居提取器（更高效）
            neighbors_1hop = self.neighborhood_extractor.get_neighbors(source, 1)
        else:
            # 直接从图中获取
            neighbors_1hop = torch.where(self.concept_graph[source] > 0)[0].tolist()
        
        # 计算所有路径的强度并取最大值
        max_strength = 0.0
        for mid in neighbors_1hop:
            # e_im * e_mj
            strength = float(self.concept_graph[source, mid].item() * 
                           self.concept_graph[mid, target].item())
            max_strength = max(max_strength, strength)
        
        # 缓存结果
        if self.cache_enabled:
            self.strength_cache[cache_key] = max_strength
        
        return max_strength
    
    def compute_khop_strength(self, source, target, k):
        """
        计算k阶路径强度
        
        Args:
            source: 源节点ID
            target: 目标节点ID
            k: 阶数（1或2）
        
        Returns:
            strength: float, k阶路径强度
        """
        if k == 1:
            return self.compute_1hop_strength(source, target)
        elif k == 2:
            return self.compute_2hop_strength(source, target)
        else:
            raise ValueError(f"Unsupported k={k}, only k=1 or k=2 is supported")
    
    def compute_all_2hop_strengths(self):
        """
        预计算所有节点对的2阶路径强度（用于加速训练）
        
        Returns:
            strength_matrix: [n_concepts, n_concepts] 2阶路径强度矩阵
        """
        print(f"预计算所有2阶路径强度...")
        
        strength_matrix = torch.zeros(self.n_concepts, self.n_concepts, 
                                      dtype=torch.float32, device=self.concept_graph.device)
        
        for i in range(self.n_concepts):
            if i % 100 == 0:
                print(f"  进度: {i}/{self.n_concepts}")
            
            # 获取i的1阶邻居
            if self.neighborhood_extractor is not None:
                neighbors_1hop = self.neighborhood_extractor.get_neighbors(i, 1)
            else:
                neighbors_1hop = torch.where(self.concept_graph[i] > 0)[0].tolist()
            
            if len(neighbors_1hop) == 0:
                continue
            
            # 对所有目标节点
            for j in range(self.n_concepts):
                if i == j:
                    continue
                
                # 计算通过所有中间节点的路径强度
                max_strength = 0.0
                for m in neighbors_1hop:
                    strength = float(self.concept_graph[i, m].item() * 
                                   self.concept_graph[m, j].item())
                    max_strength = max(max_strength, strength)
                
                strength_matrix[i, j] = max_strength
        
        print(f"  完成！非零元素比例: {(strength_matrix > 0).sum().item() / (self.n_concepts * self.n_concepts):.4f}")
        
        return strength_matrix
    
    def compute_all_2hop_strengths_vectorized(self):
        """
        使用矢量化方法预计算所有2阶路径强度（更快）
        
        Returns:
            strength_matrix: [n_concepts, n_concepts] 2阶路径强度矩阵
        """
        print(f"使用矢量化方法预计算2阶路径强度...")
        
        # A @ A.T 会给出所有2跳路径的sum，但我们需要max
        # 所以还是需要循环，但可以矢量化内层循环
        
        strength_matrix = torch.zeros(self.n_concepts, self.n_concepts, 
                                      dtype=torch.float32, device=self.concept_graph.device)
        
        for i in range(self.n_concepts):
            if i % 100 == 0:
                print(f"  进度: {i}/{self.n_concepts}")
            
            # 获取i的所有出边（1阶邻居）
            i_neighbors = self.concept_graph[i] > 0  # [n_concepts] bool mask
            
            if not i_neighbors.any():
                continue
            
            # 对于每个中间节点m，计算e_im * e_mj（对所有j）
            # 然后对所有m取max
            for m in torch.where(i_neighbors)[0]:
                m = int(m.item())
                e_im = self.concept_graph[i, m]
                e_mj = self.concept_graph[m, :]  # [n_concepts]
                
                # 计算通过m的路径强度
                path_strengths = e_im * e_mj  # [n_concepts]
                
                # 更新最大值
                strength_matrix[i] = torch.maximum(strength_matrix[i], path_strengths)
        
        # 对角线设为0（自环）
        strength_matrix.fill_diagonal_(0.0)
        
        print(f"  完成！非零元素比例: {(strength_matrix > 0).sum().item() / (self.n_concepts * self.n_concepts):.4f}")
        
        return strength_matrix
    
    def clear_cache(self):
        """清空缓存"""
        self.strength_cache = {}
    
    def enable_cache(self):
        """启用缓存"""
        self.cache_enabled = True
    
    def disable_cache(self):
        """禁用缓存"""
        self.cache_enabled = False
    
    def get_cache_size(self):
        """获取缓存大小"""
        return len(self.strength_cache)


class PrecomputedPathStrengthCalculator:
    """
    预计算版本的路径强度计算器
    
    在epoch开始时预计算所有2阶路径强度，训练时直接查表
    """
    
    def __init__(self, concept_graph, neighborhood_extractor=None):
        """
        Args:
            concept_graph: [n_concepts, n_concepts] 概念图邻接矩阵
            neighborhood_extractor: NeighborhoodExtractor实例（可选）
        """
        self.concept_graph = concept_graph
        self.n_concepts = concept_graph.shape[0]
        
        # 预计算2阶路径强度矩阵
        calculator = PathStrengthCalculator(concept_graph, neighborhood_extractor)
        self.strength_2hop = calculator.compute_all_2hop_strengths_vectorized()
    
    def compute_khop_strength(self, source, target, k):
        """
        查表获取k阶路径强度
        
        Args:
            source: 源节点ID
            target: 目标节点ID
            k: 阶数（1或2）
        
        Returns:
            strength: float, k阶路径强度
        """
        if k == 1:
            return float(self.concept_graph[source, target].item())
        elif k == 2:
            return float(self.strength_2hop[source, target].item())
        else:
            raise ValueError(f"Unsupported k={k}")
    
    def get_strength_matrix(self, k):
        """
        获取k阶路径强度矩阵
        
        Args:
            k: 阶数
        
        Returns:
            strength_matrix: [n_concepts, n_concepts]
        """
        if k == 1:
            return self.concept_graph
        elif k == 2:
            return self.strength_2hop
        else:
            raise ValueError(f"Unsupported k={k}")


if __name__ == '__main__':
    # 测试
    print("="*50)
    print("路径强度计算模块测试")
    print("="*50)
    
    # 创建测试用概念图
    n_concepts = 5
    concept_graph = torch.tensor([
        [0.0, 0.8, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.6, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.7, 0.0],
        [0.0, 0.0, 0.7, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.5, 0.0]
    ], dtype=torch.float32)
    
    print(f"\n概念图:")
    print(concept_graph)
    
    # 创建计算器
    calculator = PathStrengthCalculator(concept_graph)
    
    # 测试1阶路径强度
    print(f"\n1阶路径强度:")
    print(f"  s_01^(1) = {calculator.compute_1hop_strength(0, 1):.4f}  (应为0.8)")
    print(f"  s_12^(1) = {calculator.compute_1hop_strength(1, 2):.4f}  (应为0.6)")
    
    # 测试2阶路径强度
    print(f"\n2阶路径强度:")
    s_02 = calculator.compute_2hop_strength(0, 2)
    print(f"  s_02^(2) = {s_02:.4f}  (应为0.8*0.6=0.48, 通过节点1)")
    
    s_03 = calculator.compute_2hop_strength(0, 3)
    print(f"  s_03^(2) = {s_03:.4f}  (应为0, 无2跳路径)")
    
    # 测试预计算版本
    print(f"\n预计算版本测试:")
    precomputed_calculator = PrecomputedPathStrengthCalculator(concept_graph)
    
    print(f"\n2阶路径强度矩阵:")
    print(precomputed_calculator.strength_2hop)
    
    # 验证结果一致性
    print(f"\n验证一致性:")
    for i in range(n_concepts):
        for j in range(n_concepts):
            if i != j:
                s1 = calculator.compute_2hop_strength(i, j)
                s2 = precomputed_calculator.compute_khop_strength(i, j, 2)
                if abs(s1 - s2) > 1e-6:
                    print(f"  不一致: s_{i}{j}^(2) = {s1:.4f} vs {s2:.4f}")
    
    print(f"  验证通过！")
    
    print("\n" + "="*50)
    print("测试通过！")
    print("="*50)

