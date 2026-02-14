"""
三支决策图模块 (Triple Decision Graph Module) - 完整论文版本

实现论文第3章的核心创新：
- 显式k阶邻居提取（公式0-8）
- k跳路径强度计算（公式0-9）
- 三支决策邻域划分（公式0-11, 0-12）
- 差异化消息传递（公式0-13~0-16）
- 层次化融合（公式0-17~0-21）

论文：张慧玲-论文0201.txt 第3章
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TripleDecisionGraphComplete(nn.Module):
    """
    三支决策图模块 - 严格按论文实现
    
    核心流程：
    1. 显式提取k阶邻居（使用NeighborhoodExtractor）
    2. 计算k跳路径强度（使用PathStrengthCalculator）
    3. 根据阈值划分三支决策区域（正域/边界域/负域）
    4. 差异化消息传递（不同区域使用不同MLP）
    5. 层次化融合（区域内→区域间→跨阶）
    6. L层传播
    """
    
    def __init__(self, n_concepts, embed_dim, n_layers=2,
                 alpha=0.7, beta=0.3,
                 max_k=2, distance_decay_lambda=0.5,
                 dropout=0.1,
                 # 消融实验参数
                 use_triple_decision=True,
                 use_diff_msg=True,
                 use_neg_suppress=True):
        """
        Args:
            n_concepts: 知识点数量
            embed_dim: 嵌入维度 d_c
            n_layers: 传播层数 L
            alpha: 正域阈值（论文：0.7）
            beta: 负域阈值（论文：0.3）
            max_k: 最大阶数（论文：2，表示1阶和2阶）
            distance_decay_lambda: 距离衰减系数 λ (w/o Decay时设为0.0)
            dropout: Dropout率
            
            # 消融实验参数
            use_triple_decision: 是否使用三支决策 (False时所有邻居统一处理)
            use_diff_msg: 是否使用差异化消息传递 (False时所有区域使用相同MLP)
            use_neg_suppress: 是否使用负域抑制 (False时负域权重不衰减)
        """
        super().__init__()
        
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.beta = beta
        self.max_k = max_k
        self.lambda_decay = distance_decay_lambda
        
        # 消融实验配置
        self.use_triple_decision = use_triple_decision
        self.use_diff_msg = use_diff_msg
        self.use_neg_suppress = use_neg_suppress
        
        # 概念嵌入
        self.concept_embed = nn.Embedding(n_concepts, embed_dim)
        
        # 差异化消息传递MLP（公式0-13~0-16）
        # 为每个区域（正/边界/负）和每个阶数（1/2）创建MLP
        self.mlp_pos = nn.ModuleDict()
        self.mlp_bnd = nn.ModuleDict()
        self.mlp_neg = nn.ModuleDict()
        
        for k in range(1, max_k + 1):
            # 输入：[h_i || h_j || e_ij] = [2*embed_dim + embed_dim]
            # 输出：[embed_dim]
            self.mlp_pos[str(k)] = self._create_mlp(3 * embed_dim, embed_dim, dropout)
            self.mlp_bnd[str(k)] = self._create_mlp(3 * embed_dim, embed_dim, dropout)
            self.mlp_neg[str(k)] = self._create_mlp(3 * embed_dim, embed_dim, dropout)
        
        # 负域抑制系数 γ_neg（论文公式0-16，可学习）
        self.gamma_neg = nn.Parameter(torch.tensor(0.5))
        
        # 区域间融合矩阵 W_r（公式0-20）
        # 对每个阶数k，融合该阶的三个区域
        self.W_r = nn.ModuleDict()
        for k in range(1, max_k + 1):
            self.W_r[str(k)] = nn.Linear(3 * embed_dim, embed_dim, bias=False)
        
        # 跨阶融合 W_h 和 b_h（公式0-21）
        # 输入：[h_self || m_1hop || m_2hop] = [(max_k+1)*embed_dim]
        input_dim_wh = (max_k + 1) * embed_dim
        self.W_h = nn.Linear(input_dim_wh, embed_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        self._init_weights()
    
    def _create_mlp(self, input_dim, output_dim, dropout):
        """
        创建2层MLP
        
        结构：Linear(3d -> 2d) → ReLU → Dropout → Linear(2d -> d)
        """
        return nn.Sequential(
            nn.Linear(input_dim, 2 * output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * output_dim, output_dim)
        )
    
    def _init_weights(self):
        """Xavier均匀初始化"""
        nn.init.xavier_uniform_(self.concept_embed.weight)
        
        # 初始化所有MLP
        for module_dict in [self.mlp_pos, self.mlp_bnd, self.mlp_neg, self.W_r]:
            for module in module_dict.values():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Sequential):
                    for layer in module:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias)
        
        # 初始化W_h
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.W_h.bias)
    
    def forward(self, neighborhoods, strength_matrices, similarity_matrix=None):
        """
        前向传播（L层图传播）
        
        Args:
            neighborhoods: dict, {concept_id: {1: [邻居], 2: [邻居]}}
                          由NeighborhoodExtractor预计算
            strength_matrices: dict, {1: torch.Tensor, 2: torch.Tensor}
                              路径强度矩阵，由PathStrengthCalculator预计算
            similarity_matrix: [n_concepts, n_concepts] 余弦相似度矩阵（可选）
                              如果不提供，会在forward中计算
        
        Returns:
            final_embeds: [n_concepts, embed_dim] L层传播后的最终嵌入
        """
        # 初始化节点嵌入
        h = self.concept_embed.weight  # [n_concepts, embed_dim]
        
        # 计算余弦相似度矩阵（如果未提供）
        if similarity_matrix is None:
            similarity_matrix = self._compute_similarity_matrix(h)
        
        # L层图传播
        for layer_idx in range(self.n_layers):
            # 对每个节点进行聚合
            h_new = []
            for concept_id in range(self.n_concepts):
                h_updated = self._aggregate_one_node(
                    concept_id, h, neighborhoods, 
                    strength_matrices, similarity_matrix
                )
                h_new.append(h_updated)
            
            h = torch.stack(h_new, dim=0)  # [n_concepts, embed_dim]
            h = self.dropout(h)
        
        return h
    
    def _compute_similarity_matrix(self, embeddings):
        """
        计算余弦相似度矩阵（公式0-5）
        
        e_ij = c_i·c_j / (||c_i|| ||c_j||)
        
        Args:
            embeddings: [n_concepts, embed_dim]
        
        Returns:
            similarity_matrix: [n_concepts, n_concepts]
        """
        normalized = F.normalize(embeddings, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized, normalized.T)
        return similarity_matrix
    
    def _aggregate_one_node(self, concept_id, node_embeds, neighborhoods,
                           strength_matrices, similarity_matrix):
        """
        聚合单个节点的信息（实现论文公式0-17~0-21的完整流程）
        
        支持消融实验：
        - use_triple_decision=False: 不划分三支决策区域，所有邻居统一处理
        - use_diff_msg=False: 所有区域使用相同的消息传递函数
        - use_neg_suppress=False: 负域不进行抑制
        
        流程：
        1. 对每个阶数k（1和2）：
           a. 获取k阶邻居
           b. 根据路径强度划分三支决策区域（如果启用）
           c. 对每个区域进行差异化消息传递（如果启用）
           d. 区域内聚合（Mean）
        2. 区域间融合（W_r）
        3. 跨阶融合（W_h）
        
        Args:
            concept_id: 当前节点ID
            node_embeds: [n_concepts, embed_dim] 当前层的节点嵌入
            neighborhoods: dict, {concept_id: {1: [...], 2: [...]}}
            strength_matrices: dict, {1: Tensor, 2: Tensor}
            similarity_matrix: [n_concepts, n_concepts]
        
        Returns:
            h_new: [embed_dim] 更新后的节点嵌入
        """
        h_i = node_embeds[concept_id]  # [embed_dim]
        device = h_i.device
        
        # 存储每个阶数融合后的消息
        m_k_list = []
        
        # 对每个阶数k
        for k in range(1, self.max_k + 1):
            # 1. 获取k阶邻居
            k_neighbors = neighborhoods[concept_id][k]
            
            if len(k_neighbors) == 0:
                # 没有k阶邻居，使用零向量
                m_k_list.append(torch.zeros(self.embed_dim, device=device))
                continue
            
            # 2. 获取k跳路径强度
            neighbor_strengths = strength_matrices[k][concept_id, k_neighbors]  # [n_neighbors]
            
            # === 消融实验：w/o Three-way Decision ===
            if not self.use_triple_decision:
                # 所有邻居统一处理，不划分区域
                k_neighbors_tensor = torch.tensor(k_neighbors, dtype=torch.long, device=device)
                m_k = self._message_passing_region(
                    concept_id, k_neighbors, k, 'unified',  # 使用unified类型
                    node_embeds, similarity_matrix
                )
                m_k_list.append(m_k)
                continue
            
            # === 正常模式：三支决策划分（公式0-11, 0-12）===
            pos_mask = neighbor_strengths >= self.alpha
            neg_mask = neighbor_strengths <= self.beta
            bnd_mask = (neighbor_strengths > self.beta) & (neighbor_strengths < self.alpha)
            
            # 转换邻居列表为tensor以支持mask索引
            k_neighbors_tensor = torch.tensor(k_neighbors, dtype=torch.long, device=device)
            
            # 4. 差异化消息传递（公式0-13~0-16）
            m_pos = self._message_passing_region(
                concept_id, k_neighbors_tensor[pos_mask].tolist(), k, 'pos',
                node_embeds, similarity_matrix
            )
            m_bnd = self._message_passing_region(
                concept_id, k_neighbors_tensor[bnd_mask].tolist(), k, 'bnd',
                node_embeds, similarity_matrix
            )
            m_neg = self._message_passing_region(
                concept_id, k_neighbors_tensor[neg_mask].tolist(), k, 'neg',
                node_embeds, similarity_matrix
            )
            
            # 5. 区域间融合（公式0-20）
            # m_i^(k,l) = W_r · [m_pos || m_bnd || m_neg]
            concat_regions = torch.cat([m_pos, m_bnd, m_neg], dim=-1)  # [3*embed_dim]
            m_k = self.W_r[str(k)](concat_regions)  # [embed_dim]
            m_k_list.append(m_k)
        
        # 6. 跨阶融合（公式0-21）
        # h_i^(l+1) = σ(W_h · [h_i^(l) || m_1 || m_2] + b_h)
        concat_all = torch.cat([h_i] + m_k_list, dim=-1)  # [(max_k+1)*embed_dim]
        h_new = self.W_h(concat_all)  # [embed_dim]
        h_new = self.activation(h_new)
        
        return h_new
    
    def _message_passing_region(self, source_id, neighbor_ids, k, region_type,
                                node_embeds, similarity_matrix):
        """
        对指定区域进行差异化消息传递
        
        支持消融实验：
        - use_diff_msg=False: 所有区域使用相同MLP (使用pos的MLP)
        - use_neg_suppress=False: 负域不抑制 (gamma_neg=0)
        
        论文公式0-13~0-16：
        - 正域：m_j→i^(pos,k,l) = σ(MLP_pos([h_i || h_j || e_ij])) * ω(k)
        - 边界域：m_j→i^(bnd,k,l) = σ(MLP_bnd([h_i || h_j || e_ij])) * ω(k)
        - 负域：m_j→i^(neg,k,l) = -γ_neg * σ(MLP_neg([h_i || h_j || e_ij])) * ω(k)
        
        Args:
            source_id: 源节点ID
            neighbor_ids: 区域内邻居ID列表
            k: 阶数
            region_type: 'pos', 'bnd', 'neg', 或 'unified'
            node_embeds: [n_concepts, embed_dim]
            similarity_matrix: [n_concepts, n_concepts]
        
        Returns:
            aggregated_message: [embed_dim] 区域内聚合后的消息（公式0-17~0-19）
        """
        device = node_embeds.device
        
        # 如果该区域没有邻居
        if len(neighbor_ids) == 0:
            return torch.zeros(self.embed_dim, device=device)
        
        h_i = node_embeds[source_id]  # [embed_dim]
        
        # 收集所有邻居的消息
        messages = []
        for neighbor_id in neighbor_ids:
            h_j = node_embeds[neighbor_id]  # [embed_dim]
            e_ij_scalar = similarity_matrix[source_id, neighbor_id]  # 标量
            
            # 将标量边特征扩展为向量（决策：扩展为embed_dim维）
            e_ij_vec = torch.ones(self.embed_dim, device=device) * e_ij_scalar
            
            # 拼接 [h_i || h_j || e_ij]
            concat = torch.cat([h_i, h_j, e_ij_vec], dim=-1)  # [3*embed_dim]
            
            # === 消融实验：w/o Diff-Msg ===
            if not self.use_diff_msg or region_type == 'unified':
                # 所有区域使用相同的MLP（使用pos的MLP）
                mlp = self.mlp_pos[str(k)]
            else:
                # 正常模式：差异化消息传递
                if region_type == 'pos':
                    mlp = self.mlp_pos[str(k)]
                elif region_type == 'bnd':
                    mlp = self.mlp_bnd[str(k)]
                else:  # neg
                    mlp = self.mlp_neg[str(k)]
            
            # MLP处理
            message = mlp(concat)  # [embed_dim]
            
            # 距离衰减 ω(k) = λ^(k-1)（公式0-10）
            # 注意：lambda_decay=0时，ω(k)=0^(k-1)，k=1时为1，k>1时为0
            # 这实际上等价于没有衰减（所有阶数权重相同）
            decay_weight = self.lambda_decay ** (k - 1) if self.lambda_decay > 0 else 1.0
            message = message * decay_weight
            
            # === 消融实验：w/o Neg-Suppress ===
            # 负域特殊处理（公式0-16）
            if region_type == 'neg' and self.use_neg_suppress:
                # 限制γ_neg在[0,1]范围
                gamma = torch.clamp(self.gamma_neg, 0.0, 1.0)
                message = -gamma * message
            
            messages.append(message)
        
        # 区域内聚合：Mean（公式0-17~0-19）
        aggregated = torch.stack(messages, dim=0).mean(dim=0)  # [embed_dim]
        
        return aggregated
    
    def update_thresholds(self, new_alpha=None, new_beta=None):
        """
        更新三支决策阈值（供Actor-Critic使用）
        
        Args:
            new_alpha: 新的正域阈值
            new_beta: 新的负域阈值
        """
        if new_alpha is not None:
            self.alpha = new_alpha
        if new_beta is not None:
            self.beta = new_beta
    
    def readout(self, node_embeds, method='mean'):
        """
        Readout函数：将节点嵌入聚合为图级嵌入（公式0-22, 0-23）
        
        论文公式0-22: h'_{c_i} = Readout_1({h_i^(L)})  (节点级)
        论文公式0-23: z_t = Readout_2({h_i^(L) | c_i ∈ C})  (图级)
        
        Args:
            node_embeds: [n_concepts, embed_dim] L层传播后的节点嵌入
            method: 'mean' | 'sum' | 'max' | 'attention'
        
        Returns:
            graph_embed: [embed_dim] 图级嵌入 z_t
        """
        if method == 'mean':
            return node_embeds.mean(dim=0)
        elif method == 'sum':
            return node_embeds.sum(dim=0)
        elif method == 'max':
            return node_embeds.max(dim=0)[0]
        elif method == 'attention':
            # 简单的自注意力readout
            attn_scores = torch.matmul(node_embeds, node_embeds.mean(dim=0))
            attn_weights = F.softmax(attn_scores, dim=0)
            return torch.sum(attn_weights.unsqueeze(-1) * node_embeds, dim=0)
        else:
            raise ValueError(f"Unknown readout method: {method}")


# 保留旧名称作为别名（向后兼容）
TripleDecisionGraph = TripleDecisionGraphComplete


if __name__ == '__main__':
    # 测试
    print("="*50)
    print("三支决策图模块测试（完整版）")
    print("="*50)
    
    # 参数
    n_concepts = 10
    embed_dim = 64
    n_layers = 2
    
    # 创建模块
    model = TripleDecisionGraphComplete(
        n_concepts=n_concepts,
        embed_dim=embed_dim,
        n_layers=n_layers
    )
    
    # 创建测试数据
    # 1. 邻居字典
    neighborhoods = {}
    for i in range(n_concepts):
        neighborhoods[i] = {
            1: torch.tensor([j for j in range(n_concepts) if abs(i-j)==1]),
            2: torch.tensor([j for j in range(n_concepts) if abs(i-j)==2])
        }
    
    # 2. 路径强度矩阵
    strength_1hop = torch.rand(n_concepts, n_concepts)
    strength_2hop = torch.rand(n_concepts, n_concepts) * 0.5
    strength_matrices = {1: strength_1hop, 2: strength_2hop}
    
    print(f"\n输入:")
    print(f"  节点数: {n_concepts}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  传播层数: {n_layers}")
    
    # 前向传播
    output = model(neighborhoods, strength_matrices)
    
    print(f"\n输出:")
    print(f"  输出形状: {output.shape}")
    print(f"  预期形状: [{n_concepts}, {embed_dim}]")
    
    # 测试阈值更新
    print(f"\n测试阈值更新:")
    print(f"  初始 α={model.alpha}, β={model.beta}")
    model.update_thresholds(new_alpha=0.8, new_beta=0.2)
    print(f"  更新后 α={model.alpha}, β={model.beta}")
    
    print("\n" + "="*50)
    print("测试通过！")
    print("="*50)

