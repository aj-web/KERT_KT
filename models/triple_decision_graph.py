"""
Triple Decision Graph Representation Module
Implements the three-way decision theory for knowledge graph representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class TripleDecisionGraph(nn.Module):
    """
    Knowledge Point Graph Representation with Triple Decision Theory

    This module implements the core innovation of KER-KT:
    - Builds knowledge point relationship graph
    - Applies three-way decision theory for neighbor classification
    - Performs differentiated aggregation for different regions
    - Multi-layer graph propagation with inter-layer fusion
    """

    def __init__(self, n_concepts, embed_dim, n_layers=2, alpha=0.7, beta=0.3, lambda_decay=0.1):
        """
        Initialize Triple Decision Graph module

        Args:
            n_concepts: number of knowledge concepts
            embed_dim: embedding dimension for concepts
            n_layers: number of graph propagation layers
            alpha: acceptance threshold for positive region
            beta: rejection threshold for negative region
            lambda_decay: decay factor for negative region
        """
        super(TripleDecisionGraph, self).__init__()

        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.beta = beta
        self.lambda_decay = lambda_decay

        # Concept embedding layer
        self.concept_embed = nn.Embedding(n_concepts, embed_dim)

        # Graph convolution layers for each propagation layer
        self.gc_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])

        # Attention mechanism for boundary region
        self.attention_query = nn.Linear(embed_dim, embed_dim)
        self.attention_key = nn.Linear(embed_dim, embed_dim)
        self.attention_value = nn.Linear(embed_dim, embed_dim)

        # Gate mechanism for multi-region fusion
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 4),  # 4 gates: self, positive, boundary, negative
            nn.Softmax(dim=-1)
        )

        # Layer-wise fusion weights (learnable)
        self.layer_weights = Parameter(torch.ones(n_layers) / n_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.concept_embed.weight)

        for layer in self.gc_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Attention weights
        for module in [self.attention_query, self.attention_key, self.attention_value]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Gate network weights
        for layer in self.gate_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, concept_graph):
        """
        Forward pass of Triple Decision Graph

        Args:
            concept_graph: adjacency matrix of concept graph [n_concepts, n_concepts]

        Returns:
            enhanced_concept_embed: enhanced concept embeddings [n_concepts, embed_dim]
        """
        # Initialize concept embeddings
        concept_embed = self.concept_embed.weight  # [n_concepts, embed_dim]

        # Multi-layer graph propagation
        layer_outputs = []
        current_embed = concept_embed

        for layer_idx in range(self.n_layers):
            # Apply triple decision aggregation
            aggregated_embed = self._triple_decision_aggregate(current_embed, concept_graph)

            # Graph convolution
            conv_embed = self.gc_layers[layer_idx](aggregated_embed)
            conv_embed = F.relu(conv_embed)

            layer_outputs.append(conv_embed)
            current_embed = conv_embed

        # Inter-layer fusion
        enhanced_embed = self._inter_layer_fusion(layer_outputs)

        return enhanced_embed

    def _triple_decision_aggregate(self, node_embed, adj_matrix):
        """
        Triple decision aggregation for each node

        Args:
            node_embed: node embeddings [n_concepts, embed_dim]
            adj_matrix: adjacency matrix [n_concepts, n_concepts]

        Returns:
            aggregated_embed: aggregated embeddings [n_concepts, embed_dim]
        """
        batch_size = node_embed.size(0)
        aggregated_embeddings = []

        for i in range(batch_size):
            # Get neighbors of node i
            neighbors = torch.nonzero(adj_matrix[i]).squeeze(-1)
            if len(neighbors) == 0:
                # Isolated node, use self embedding
                aggregated_embeddings.append(node_embed[i])
                continue

            # Compute similarities with neighbors
            node_i_embed = node_embed[i].unsqueeze(0)  # [1, embed_dim]
            neighbors_embed = node_embed[neighbors]    # [n_neighbors, embed_dim]

            # Cosine similarity
            similarities = F.cosine_similarity(node_i_embed, neighbors_embed, dim=-1)  # [n_neighbors]

            # Triple decision classification
            pos_mask = similarities >= self.alpha  # Positive region
            neg_mask = similarities <= self.beta   # Negative region
            bound_mask = (similarities > self.beta) & (similarities < self.alpha)  # Boundary region

            # Differentiated aggregation
            pos_embed = self._positive_aggregate(neighbors_embed, pos_mask)
            bound_embed = self._boundary_aggregate(neighbors_embed, bound_mask, node_i_embed)
            neg_embed = self._negative_aggregate(neighbors_embed, neg_mask)

            # Multi-region fusion with gating
            fused_embed = self._region_fusion(node_i_embed.squeeze(0), pos_embed, bound_embed, neg_embed)
            aggregated_embeddings.append(fused_embed)

        return torch.stack(aggregated_embeddings)

    def _positive_aggregate(self, neighbors_embed, pos_mask):
        """
        Positive region aggregation: simple mean pooling

        Args:
            neighbors_embed: neighbor embeddings [n_neighbors, embed_dim]
            pos_mask: boolean mask for positive region [n_neighbors]

        Returns:
            pos_embed: positive region embedding [embed_dim]
        """
        if not pos_mask.any():
            return torch.zeros(self.embed_dim, device=neighbors_embed.device)

        pos_neighbors = neighbors_embed[pos_mask]  # [n_pos, embed_dim]
        pos_embed = torch.mean(pos_neighbors, dim=0)  # [embed_dim]

        return pos_embed

    def _boundary_aggregate(self, neighbors_embed, bound_mask, node_embed):
        """
        Boundary region aggregation: attention mechanism

        Args:
            neighbors_embed: neighbor embeddings [n_neighbors, embed_dim]
            bound_mask: boolean mask for boundary region [n_neighbors]
            node_embed: current node embedding [1, embed_dim]

        Returns:
            bound_embed: boundary region embedding [embed_dim]
        """
        if not bound_mask.any():
            return torch.zeros(self.embed_dim, device=neighbors_embed.device)

        bound_neighbors = neighbors_embed[bound_mask]  # [n_bound, embed_dim]

        # Attention computation
        query = self.attention_query(node_embed)      # [1, embed_dim]
        key = self.attention_key(bound_neighbors)     # [n_bound, embed_dim]
        value = self.attention_value(bound_neighbors) # [n_bound, embed_dim]

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # [1, n_bound]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, n_bound]

        # Weighted aggregation
        bound_embed = torch.matmul(attention_weights, value).squeeze(0)  # [embed_dim]

        return bound_embed

    def _negative_aggregate(self, neighbors_embed, neg_mask):
        """
        Negative region aggregation: weighted decay

        Args:
            neighbors_embed: neighbor embeddings [n_neighbors, embed_dim]
            neg_mask: boolean mask for negative region [n_neighbors]

        Returns:
            neg_embed: negative region embedding [embed_dim]
        """
        if not neg_mask.any():
            return torch.zeros(self.embed_dim, device=neighbors_embed.device)

        neg_neighbors = neighbors_embed[neg_mask]  # [n_neg, embed_dim]
        neg_embed = torch.mean(neg_neighbors, dim=0)  # [embed_dim]

        # Apply decay factor
        neg_embed = neg_embed * self.lambda_decay

        return neg_embed

    def _region_fusion(self, self_embed, pos_embed, bound_embed, neg_embed):
        """
        Multi-region fusion with gating mechanism

        Args:
            self_embed: self embedding [embed_dim]
            pos_embed: positive region embedding [embed_dim]
            bound_embed: boundary region embedding [embed_dim]
            neg_embed: negative region embedding [embed_dim]

        Returns:
            fused_embed: fused embedding [embed_dim]
        """
        # Concatenate all region embeddings
        concat_embed = torch.cat([self_embed, pos_embed, bound_embed, neg_embed], dim=-1)  # [4*embed_dim]

        # Compute gating weights
        gate_weights = self.gate_net(concat_embed)  # [4]

        # Weighted fusion
        regions = torch.stack([self_embed, pos_embed, bound_embed, neg_embed], dim=0)  # [4, embed_dim]
        fused_embed = torch.sum(gate_weights.unsqueeze(-1) * regions, dim=0)  # [embed_dim]

        return fused_embed

    def _inter_layer_fusion(self, layer_outputs):
        """
        Inter-layer fusion of multi-layer propagation results

        Args:
            layer_outputs: list of layer outputs [n_layers, n_concepts, embed_dim]

        Returns:
            fused_embed: fused embedding [n_concepts, embed_dim]
        """
        # Normalize layer weights
        weights = F.softmax(self.layer_weights, dim=0)  # [n_layers]

        # Weighted sum of layer outputs
        stacked_outputs = torch.stack(layer_outputs, dim=0)  # [n_layers, n_concepts, embed_dim]
        fused_embed = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * stacked_outputs, dim=0)  # [n_concepts, embed_dim]

        return fused_embed

    def update_thresholds(self, new_alpha=None, new_beta=None):
        """
        Update triple decision thresholds (for reinforcement learning)

        Args:
            new_alpha: new acceptance threshold
            new_beta: new rejection threshold
        """
        if new_alpha is not None:
            self.alpha = new_alpha
        if new_beta is not None:
            self.beta = new_beta


class TripleDecisionLoss(nn.Module):
    """
    Loss function for triple decision graph training
    """

    def __init__(self, margin=0.1):
        super(TripleDecisionLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, adj_matrix):
        """
        Compute triple decision loss

        Args:
            embeddings: concept embeddings [n_concepts, embed_dim]
            adj_matrix: adjacency matrix [n_concepts, n_concepts]

        Returns:
            loss: triple decision loss
        """
        # Cosine similarity matrix
        norm_embed = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(norm_embed, norm_embed.t())  # [n_concepts, n_concepts]

        # Mask self-similarities
        mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1)

        # Positive pairs (connected concepts)
        pos_mask = (adj_matrix > 0).float()
        pos_sim = sim_matrix * pos_mask
        pos_loss = torch.sum((1 - pos_sim) * pos_mask) / (torch.sum(pos_mask) + 1e-8)

        # Negative pairs (disconnected concepts)
        neg_mask = (adj_matrix == 0).float() * (1 - torch.eye(embeddings.size(0), device=embeddings.device))
        neg_sim = sim_matrix * neg_mask
        neg_loss = torch.sum(torch.relu(neg_sim - self.margin) * neg_mask) / (torch.sum(neg_mask) + 1e-8)

        total_loss = pos_loss + neg_loss
        return total_loss


if __name__ == "__main__":
    # Test the module
    n_concepts = 124  # ASSIST09
    embed_dim = 128

    # Create synthetic concept graph
    concept_graph = torch.rand(n_concepts, n_concepts)
    concept_graph = (concept_graph + concept_graph.t()) / 2  # Symmetric
    concept_graph = (concept_graph > 0.3).float()  # Sparsify

    # Initialize model
    model = TripleDecisionGraph(n_concepts, embed_dim)
    loss_fn = TripleDecisionLoss()

    # Forward pass
    enhanced_embed = model(concept_graph)
    loss = loss_fn(enhanced_embed, concept_graph)

    print(f"Enhanced embeddings shape: {enhanced_embed.shape}")
    print(f"Triple decision loss: {loss.item():.4f}")
    print("Triple Decision Graph module test passed!")
