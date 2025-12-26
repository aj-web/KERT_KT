"""
Context-Aware Attentive Knowledge Tracing (AKT) Baseline Model
Original paper: Ghosh et al., "Context-Aware Attentive Knowledge Tracing", KDD 2020
论文4.3.1节：AKT结合图神经网络和注意力机制，利用知识点间的结构关系增强表征学习
模型通过上下文感知的注意力机制动态聚合相关历史交互，实现个性化的知识追踪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AKT(nn.Module):
    """
    Context-Aware Attentive Knowledge Tracing model
    知识点图集成 + 上下文感知注意力
    """

    def __init__(self, n_questions, n_concepts, embed_dim=200, hidden_dim=200, 
                 num_heads=8, num_layers=2, dropout=0.2, concept_graph=None):
        """
        Initialize AKT model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension
            hidden_dim: hidden dimension
            num_heads: number of attention heads
            num_layers: number of layers
            dropout: dropout rate
            concept_graph: concept adjacency matrix [n_concepts, n_concepts] (optional)
        """
        super(AKT, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Question embedding
        self.question_embed = nn.Embedding(n_questions, embed_dim)

        # Concept embedding (enhanced by graph if provided)
        self.concept_embed = nn.Embedding(n_concepts, embed_dim)

        # Answer embedding
        self.answer_embed = nn.Embedding(2, embed_dim)

        # Graph enhancement (if concept_graph provided)
        if concept_graph is not None:
            self.use_graph = True
            self.register_buffer('concept_graph', torch.tensor(concept_graph, dtype=torch.float32))
            # Graph convolution for concept enhancement
            self.graph_conv = nn.Linear(embed_dim, embed_dim)
        else:
            self.use_graph = False
            self.concept_graph = None

        # Context-aware attention layers
        self.attention_layers = nn.ModuleList([
            ContextAwareAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization and feed-forward
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.question_embed.weight)
        nn.init.xavier_uniform_(self.concept_embed.weight)
        nn.init.xavier_uniform_(self.answer_embed.weight)

        if self.use_graph:
            nn.init.xavier_uniform_(self.graph_conv.weight)
            nn.init.zeros_(self.graph_conv.bias)

        for layer in self.feed_forward:
            if isinstance(layer, nn.Linear):
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

        # Embed questions, concepts, and answers
        q_embeds = self.question_embed(question_seq)  # [batch_size, seq_len, embed_dim]
        c_embeds = self.concept_embed(concept_seq)     # [batch_size, seq_len, embed_dim]
        a_embeds = self.answer_embed(answer_seq)       # [batch_size, seq_len, embed_dim]

        # Enhance concept embeddings with graph if available
        if self.use_graph:
            # Graph convolution: aggregate neighbor information
            # Simplified: use concept_graph to enhance embeddings
            graph_enhanced = torch.matmul(self.concept_graph, self.concept_embed.weight)  # [n_concepts, embed_dim]
            # Select enhanced embeddings for concepts in sequence
            c_embeds = c_embeds + graph_enhanced[concept_seq]  # [batch_size, seq_len, embed_dim]

        # Combine embeddings: question + concept + answer
        combined_embeds = q_embeds + c_embeds + a_embeds  # [batch_size, seq_len, embed_dim]

        # Apply context-aware attention layers
        x = combined_embeds
        for i, (attention_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Self-attention
            attn_output = attention_layer(x, x, x)  # [batch_size, seq_len, embed_dim]
            x = layer_norm(x + attn_output)

            # Feed-forward
            ff_output = self.feed_forward(x)
            x = layer_norm(x + ff_output)

        # For prediction
        if target_question is not None and target_concept is not None:
            # Use target question and concept
            target_q_embed = self.question_embed(target_question)  # [batch_size, embed_dim]
            target_c_embed = self.concept_embed(target_concept)    # [batch_size, embed_dim]

            if self.use_graph:
                target_c_embed = target_c_embed + graph_enhanced[target_concept]

            # Context-aware attention between target and history
            target_embed = (target_q_embed + target_c_embed).unsqueeze(1)  # [batch_size, 1, embed_dim]

            # Attention over history
            attention_scores = torch.matmul(target_embed, x.transpose(-2, -1))  # [batch_size, 1, seq_len]
            attention_scores = attention_scores / math.sqrt(self.embed_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, seq_len]

            # Weighted context
            context = torch.matmul(attention_weights, x).squeeze(1)  # [batch_size, embed_dim]

            # Combine target and context
            prediction_input = torch.cat([target_q_embed + target_c_embed, context], dim=-1)  # [batch_size, 2*embed_dim]
        else:
            # Use last position
            last_embed = x[:, -1, :]  # [batch_size, embed_dim]
            context = x[:, -2, :] if seq_len > 1 else last_embed  # Use second-to-last as context
            prediction_input = torch.cat([last_embed, context], dim=-1)  # [batch_size, 2*embed_dim]

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


class ContextAwareAttention(nn.Module):
    """Context-aware attention mechanism"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(ContextAwareAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            mask: attention mask (optional)
        """
        batch_size, seq_len, embed_dim = query.size()

        # Linear projections
        Q = self.query(query)  # [batch_size, seq_len, embed_dim]
        K = self.key(key)
        V = self.value(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]

        return output


class AKTLoss(nn.Module):
    """AKT Loss Function"""

    def __init__(self):
        super(AKTLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Compute AKT loss

        Args:
            predictions: model predictions [batch_size]
            targets: ground truth labels [batch_size]

        Returns:
            loss: loss value
        """
        return self.bce_loss(predictions, targets.float())


if __name__ == "__main__":
    # Test AKT model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    # Create dummy concept graph
    concept_graph = np.random.rand(n_concepts, n_concepts)
    concept_graph = (concept_graph + concept_graph.T) / 2  # Symmetric
    concept_graph = (concept_graph > 0.3).astype(float)  # Sparse

    model = AKT(n_questions, n_concepts, concept_graph=concept_graph)
    loss_fn = AKTLoss()

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
    print("AKT model test passed!")

