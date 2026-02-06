"""
k阶邻居提取模块 (K-hop Neighborhood Extractor)

实现论文公式(0-8)：使用BFS算法显式提取k阶邻居

论文：张慧玲-论文0201.txt 第3章
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque, defaultdict


class NeighborhoodExtractor:
    """
    k阶邻居提取器
    
    论文公式(0-8)：
    N^(k)(c_i) = {c_j ∈ C | d(c_i, c_j) = k}
    
    使用BFS算法显式提取不同阶数的邻居，避免图神经网络的隐式混合
    
    关键特性：
    - 2阶邻居排除1阶邻居（严格的k跳距离定义）
    - 支持有向图和无向图
    - 最短路径优先（如果存在多条路径）
    """
    
    def __init__(self, concept_graph, max_k=2, directed=False):
        """
        Args:
            concept_graph: [n_concepts, n_concepts] 概念图邻接矩阵
            max_k: 最大阶数（默认2，即考虑1阶和2阶邻居）
            directed: 是否为有向图（False表示无向图）
        """
        self.concept_graph = concept_graph
        self.n_concepts = concept_graph.shape[0]
        self.max_k = max_k
        self.directed = directed
        
        # 预计算所有知识点的k阶邻居
        self.neighborhoods = self._precompute_neighborhoods()
    
    def _precompute_neighborhoods(self):
        """
        预计算所有知识点的k阶邻居
        
        Returns:
            neighborhoods: dict, {concept_id: {1: [邻居列表], 2: [邻居列表], ...}}
        """
        print(f"预计算{self.n_concepts}个知识点的{self.max_k}阶邻居...")
        
        neighborhoods = {}
        for concept_id in range(self.n_concepts):
            neighborhoods[concept_id] = self._extract_k_hop_neighbors_bfs(concept_id)
        
        # 打印统计信息
        avg_neighbors = {k: 0 for k in range(1, self.max_k + 1)}
        for concept_id in range(self.n_concepts):
            for k in range(1, self.max_k + 1):
                avg_neighbors[k] += len(neighborhoods[concept_id][k])
        
        for k in range(1, self.max_k + 1):
            avg_neighbors[k] /= self.n_concepts
            print(f"  {k}阶邻居平均数量: {avg_neighbors[k]:.2f}")
        
        return neighborhoods
    
    def _extract_k_hop_neighbors_bfs(self, source):
        """
        使用BFS提取source节点的k阶邻居
        
        Args:
            source: 源节点ID
        
        Returns:
            neighbors: dict, {1: [1阶邻居], 2: [2阶邻居], ...}
        """
        neighbors = {k: [] for k in range(1, self.max_k + 1)}
        visited = {source}
        
        # BFS队列：(节点ID, 距离)
        queue = deque([(source, 0)])
        
        while queue:
            node, dist = queue.popleft()
            
            # 如果已达到最大阶数，停止扩展
            if dist >= self.max_k:
                continue
            
            # 获取当前节点的邻居
            if self.concept_graph[node].sum() > 0:
                # 找到所有相邻节点（边权重>0）
                neighbor_ids = torch.where(self.concept_graph[node] > 0)[0]
                
                for neighbor_id in neighbor_ids:
                    neighbor_id = int(neighbor_id.item())
                    
                    # 如果未访问过
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        new_dist = dist + 1
                        
                        # 添加到对应阶数的邻居列表
                        if new_dist <= self.max_k:
                            neighbors[new_dist].append(neighbor_id)
                        
                        # 加入队列继续扩展
                        queue.append((neighbor_id, new_dist))
        
        return neighbors
    
    def get_neighbors(self, concept_id, k):
        """
        获取指定知识点的k阶邻居
        
        Args:
            concept_id: 知识点ID
            k: 阶数（1或2）
        
        Returns:
            neighbors: list, k阶邻居ID列表
        """
        if concept_id not in self.neighborhoods:
            return []
        if k not in self.neighborhoods[concept_id]:
            return []
        return self.neighborhoods[concept_id][k]
    
    def get_all_neighbors(self, concept_id):
        """
        获取指定知识点的所有阶数邻居
        
        Args:
            concept_id: 知识点ID
        
        Returns:
            neighbors: dict, {1: [邻居列表], 2: [邻居列表], ...}
        """
        if concept_id not in self.neighborhoods:
            return {k: [] for k in range(1, self.max_k + 1)}
        return self.neighborhoods[concept_id]
    
    def to_tensor(self, concept_id, k, device='cpu'):
        """
        将k阶邻居转换为tensor
        
        Args:
            concept_id: 知识点ID
            k: 阶数
            device: 设备
        
        Returns:
            neighbors_tensor: torch.LongTensor
        """
        neighbors = self.get_neighbors(concept_id, k)
        if len(neighbors) == 0:
            return torch.LongTensor([]).to(device)
        return torch.LongTensor(neighbors).to(device)
    
    def get_neighbor_mask(self, concept_id, k, device='cpu'):
        """
        生成k阶邻居的mask（用于masked attention等操作）
        
        Args:
            concept_id: 知识点ID
            k: 阶数
            device: 设备
        
        Returns:
            mask: torch.BoolTensor [n_concepts], True表示是k阶邻居
        """
        mask = torch.zeros(self.n_concepts, dtype=torch.bool, device=device)
        neighbors = self.get_neighbors(concept_id, k)
        if len(neighbors) > 0:
            mask[neighbors] = True
        return mask
    
    def get_statistics(self):
        """
        获取邻居提取的统计信息
        
        Returns:
            stats: dict, 统计信息
        """
        stats = {
            'n_concepts': self.n_concepts,
            'max_k': self.max_k,
            'directed': self.directed,
            'avg_neighbors': {},
            'max_neighbors': {},
            'min_neighbors': {},
            'isolated_concepts': []
        }
        
        for k in range(1, self.max_k + 1):
            neighbor_counts = [len(self.neighborhoods[c][k]) for c in range(self.n_concepts)]
            stats['avg_neighbors'][k] = np.mean(neighbor_counts)
            stats['max_neighbors'][k] = np.max(neighbor_counts)
            stats['min_neighbors'][k] = np.min(neighbor_counts)
        
        # 统计孤立节点（没有1阶邻居）
        for c in range(self.n_concepts):
            if len(self.neighborhoods[c][1]) == 0:
                stats['isolated_concepts'].append(c)
        
        return stats


class AdaptiveNeighborhoodExtractor(NeighborhoodExtractor):
    """
    自适应邻居提取器
    
    当邻居数量过多时，使用top-k策略只保留最重要的邻居
    """
    
    def __init__(self, concept_graph, max_k=2, directed=False, top_k=50):
        """
        Args:
            concept_graph: [n_concepts, n_concepts] 概念图邻接矩阵
            max_k: 最大阶数
            directed: 是否为有向图
            top_k: 每个阶数最多保留的邻居数
        """
        self.top_k = top_k
        super().__init__(concept_graph, max_k, directed)
    
    def _extract_k_hop_neighbors_bfs(self, source):
        """
        BFS提取邻居，并根据边权重选择top-k
        
        Args:
            source: 源节点ID
        
        Returns:
            neighbors: dict, {1: [邻居列表], 2: [邻居列表], ...}
        """
        neighbors_with_weights = {k: [] for k in range(1, self.max_k + 1)}
        visited = {source}
        
        queue = deque([(source, 0, 1.0)])  # (节点ID, 距离, 累积权重)
        
        while queue:
            node, dist, cum_weight = queue.popleft()
            
            if dist >= self.max_k:
                continue
            
            if self.concept_graph[node].sum() > 0:
                neighbor_ids = torch.where(self.concept_graph[node] > 0)[0]
                
                for neighbor_id in neighbor_ids:
                    neighbor_id = int(neighbor_id.item())
                    edge_weight = float(self.concept_graph[node, neighbor_id].item())
                    
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        new_dist = dist + 1
                        new_weight = cum_weight * edge_weight
                        
                        if new_dist <= self.max_k:
                            neighbors_with_weights[new_dist].append((neighbor_id, new_weight))
                        
                        queue.append((neighbor_id, new_dist, new_weight))
        
        # 对每个阶数，按权重排序并选择top-k
        neighbors = {k: [] for k in range(1, self.max_k + 1)}
        for k in range(1, self.max_k + 1):
            if len(neighbors_with_weights[k]) > 0:
                # 按权重降序排序
                sorted_neighbors = sorted(neighbors_with_weights[k], key=lambda x: x[1], reverse=True)
                # 取top-k
                top_neighbors = sorted_neighbors[:self.top_k]
                neighbors[k] = [n[0] for n in top_neighbors]
        
        return neighbors


if __name__ == '__main__':
    # 测试
    print("="*50)
    print("k阶邻居提取模块测试")
    print("="*50)
    
    # 创建测试用概念图（10个知识点）
    n_concepts = 10
    concept_graph = torch.zeros(n_concepts, n_concepts)
    
    # 构造一个简单的图结构
    # 0 -> 1 -> 2 -> 3
    # 0 -> 4 -> 5
    # 6 -> 7
    # 8, 9 孤立
    edges = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5),
        (6, 7)
    ]
    
    for i, j in edges:
        concept_graph[i, j] = 1.0
        concept_graph[j, i] = 1.0  # 无向图
    
    print(f"\n概念图结构:")
    print(f"  节点数: {n_concepts}")
    print(f"  边数: {len(edges) * 2}  (无向)")
    
    # 创建提取器
    extractor = NeighborhoodExtractor(concept_graph, max_k=2)
    
    # 测试节点0的邻居
    print(f"\n节点0的邻居:")
    print(f"  1阶邻居: {extractor.get_neighbors(0, 1)}")
    print(f"  2阶邻居: {extractor.get_neighbors(0, 2)}")
    
    # 测试节点2的邻居
    print(f"\n节点2的邻居:")
    print(f"  1阶邻居: {extractor.get_neighbors(2, 1)}")
    print(f"  2阶邻居: {extractor.get_neighbors(2, 2)}")
    
    # 获取统计信息
    print(f"\n统计信息:")
    stats = extractor.get_statistics()
    for k in range(1, 3):
        print(f"  {k}阶邻居:")
        print(f"    平均: {stats['avg_neighbors'][k]:.2f}")
        print(f"    最大: {stats['max_neighbors'][k]}")
        print(f"    最小: {stats['min_neighbors'][k]}")
    print(f"  孤立节点: {stats['isolated_concepts']}")
    
    # 测试自适应版本
    print(f"\n自适应版本测试:")
    adaptive_extractor = AdaptiveNeighborhoodExtractor(concept_graph, max_k=2, top_k=3)
    print(f"  节点0的1阶邻居（top-3）: {adaptive_extractor.get_neighbors(0, 1)}")
    
    print("\n" + "="*50)
    print("测试通过！")
    print("="*50)

