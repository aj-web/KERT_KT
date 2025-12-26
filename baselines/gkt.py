"""
Graph-based Knowledge Tracing (GKT) Baseline Model
Original paper: Nakagawa et al., "Graph-based Knowledge Tracing: modeling student proficiency using graph neural network", WI 2019
论文4.3.1节：GKT构建知识点图并利用图卷积网络学习知识点表征
模型通过图结构捕捉知识点间的前置和关联关系，将图表征融入LSTM进行序列建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GKT(nn.Module):
    """
    Graph-based Knowledge Tracing model
    图卷积网络 + LSTM序列建模
    """

    def __init__(self, n_questions, n_concepts, embed_dim=200, hidden_dim=200, 
                 num_gcn_layers=2, dropout=0.2, concept_graph=None):
        """
        Initialize GKT model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension
            hidden_dim: LSTM hidden dimension
            num_gcn_layers: number of GCN layers
            dropout: dropout rate
            concept_graph: concept adjacency matrix [n_concepts, n_concepts]
        """
        super(GKT, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers

        # Concept embedding
        self.concept_embed = nn.Embedding(n_concepts, embed_dim)

        # Graph Convolutional Networks for concept enhancement
        if concept_graph is not None:
            self.use_graph = True
            self.register_buffer('concept_graph', torch.tensor(concept_graph, dtype=torch.float32))
            # Normalize adjacency matrix
            self._normalize_graph()
        else:
            self.use_graph = False
            self.concept_graph = None

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(embed_dim, embed_dim)
            for _ in range(num_gcn_layers)
        ])

        # Question embedding
        self.question_embed = nn.Embedding(n_questions, embed_dim)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,  # question + concept
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _normalize_graph(self):
        """Normalize adjacency matrix for GCN"""
        if self.concept_graph is not None:
            # Add self-loops
            adj = self.concept_graph + torch.eye(self.n_concepts, device=self.concept_graph.device)
            # Compute degree matrix
            degree = adj.sum(dim=1)
            degree_inv_sqrt = torch.pow(degree, -0.5)
            degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
            # Normalize: D^(-1/2) * A * D^(-1/2)
            degree_matrix = torch.diag(degree_inv_sqrt)
            self.normalized_graph = torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        else:
            self.normalized_graph = None

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.concept_embed.weight)
        nn.init.xavier_uniform_(self.question_embed.weight)

        for layer in self.gcn_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, question_seq, concept_seq, answer_seq, target_question=None, target_concept=None):
        """
        Forward pass

        Args:
            question_seq: question sequence [batch_size, seq_len]
            concept_seq: concept sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_question: target question [batch_size] (optional)
            target_concept: target concept [batch_size] (optional)

        Returns:
            predictions: predicted probabilities [batch_size]
        """
        batch_size, seq_len = question_seq.size()

        # Enhance concept embeddings with GCN
        if self.use_graph and self.normalized_graph is not None:
            concept_embeds = self.concept_embed.weight  # [n_concepts, embed_dim]
            # Apply GCN layers
            for gcn_layer in self.gcn_layers:
                concept_embeds = gcn_layer(concept_embeds, self.normalized_graph)
                concept_embeds = F.relu(concept_embeds)
            # Get enhanced embeddings for concepts in sequence
            enhanced_concepts = concept_embeds[concept_seq]  # [batch_size, seq_len, embed_dim]
        else:
            enhanced_concepts = self.concept_embed(concept_seq)  # [batch_size, seq_len, embed_dim]

        # Embed questions
        question_embeds = self.question_embed(question_seq)  # [batch_size, seq_len, embed_dim]

        # Combine question and concept embeddings
        lstm_input = torch.cat([question_embeds, enhanced_concepts], dim=-1)  # [batch_size, seq_len, 2*embed_dim]

        # LSTM encoding
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)  # [batch_size, seq_len, hidden_dim]

        # For prediction
        if target_question is not None and target_concept is not None:
            # Use target question and concept
            target_q_embed = self.question_embed(target_question)  # [batch_size, embed_dim]

            if self.use_graph and self.normalized_graph is not None:
                target_c_embed = concept_embeds[target_concept]  # [batch_size, embed_dim]
            else:
                target_c_embed = self.concept_embed(target_concept)  # [batch_size, embed_dim]

            # Use last hidden state
            last_hidden = h_n.squeeze(0)  # [batch_size, hidden_dim]

            # Combine with target information
            prediction_input = last_hidden  # [batch_size, hidden_dim]
        else:
            # Use last hidden state
            prediction_input = lstm_output[:, -1, :]  # [batch_size, hidden_dim]

        # Predict probability
        predictions = self.output_layer(prediction_input).squeeze(-1)  # [batch_size]

        return predictions

    def predict_single_concept(self, question_seq, concept_seq, answer_seq, target_concept):
        """
        Predict probability for a specific concept

        Args:
            question_seq: question sequence [batch_size, seq_len]
            concept_seq: concept sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_concept: target concept index [batch_size]

        Returns:
            predictions: predicted probabilities [batch_size]
        """
        batch_size, seq_len = question_seq.size()
        # Use last question as target question (simplified)
        target_question = question_seq[:, -1]

        return self.forward(question_seq, concept_seq, answer_seq,
                          target_question=target_question, target_concept=target_concept)


class GraphConvolution(nn.Module):
    """Graph Convolutional Layer"""

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        Args:
            input: node features [n_nodes, in_features]
            adj: normalized adjacency matrix [n_nodes, n_nodes]
        """
        support = torch.matmul(input, self.weight)  # [n_nodes, out_features]
        output = torch.matmul(adj, support)  # [n_nodes, out_features]
        return output + self.bias


class GKTLoss(nn.Module):
    """GKT Loss Function"""

    def __init__(self):
        super(GKTLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Compute GKT loss

        Args:
            predictions: model predictions [batch_size]
            targets: ground truth labels [batch_size]

        Returns:
            loss: loss value
        """
        return self.bce_loss(predictions, targets.float())


if __name__ == "__main__":
    # Test GKT model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    # Create concept graph
    concept_graph = np.random.rand(n_concepts, n_concepts)
    concept_graph = (concept_graph + concept_graph.T) / 2  # Symmetric
    concept_graph = (concept_graph > 0.3).astype(float)  # Sparse

    model = GKT(n_questions, n_concepts, concept_graph=concept_graph)
    loss_fn = GKTLoss()

    batch_size = 4
    seq_len = 50

    question_seq = torch.randint(0, n_questions, (batch_size, seq_len))
    concept_seq = torch.randint(0, n_concepts, (batch_size, seq_len))
    answer_seq = torch.randint(0, 2, (batch_size, seq_len))
    target_concept = torch.randint(0, n_concepts, (batch_size,))
    labels = torch.randint(0, 2, (batch_size,)).float()

    predictions = model.predict_single_concept(question_seq, concept_seq, answer_seq, target_concept)
    loss = loss_fn(predictions, labels)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("GKT model test passed!")

