"""
Self-Attentive Knowledge Tracing (SAKT) Baseline Model
Original paper: Pandey & Karypis, "A Self-Attentive model for knowledge tracing", EDM 2019
论文4.3.1节：SAKT首次将自注意力机制应用于知识追踪，摒弃了循环结构，通过多头自注意力捕捉学生历史交互间的依赖关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SAKT(nn.Module):
    """
    Self-Attentive Knowledge Tracing model
    Transformer架构，位置编码，多头自注意力
    """

    def __init__(self, n_questions, n_concepts, embed_dim=200, num_heads=8, num_layers=2, dropout=0.2, max_seq_len=200):
        """
        Initialize SAKT model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension (must be divisible by num_heads)
            num_heads: number of attention heads
            num_layers: number of transformer layers
            dropout: dropout rate
            max_seq_len: maximum sequence length
        """
        super(SAKT, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Question embedding
        self.question_embed = nn.Embedding(n_questions, embed_dim)

        # Answer embedding (binary: correct/incorrect)
        self.answer_embed = nn.Embedding(2, embed_dim)

        # Positional encoding
        self.pos_embed = PositionalEncoding(embed_dim, dropout, max_seq_len)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
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
        nn.init.xavier_uniform_(self.answer_embed.weight)

        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, question_seq, answer_seq, target_question=None, target_concept=None):
        """
        Forward pass

        Args:
            question_seq: question sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_question: target question for prediction [batch_size] (optional)
            target_concept: target concept for prediction [batch_size] (optional)

        Returns:
            predictions: predicted probabilities [batch_size]
        """
        batch_size, seq_len = question_seq.size()

        # Embed questions and answers
        q_embeds = self.question_embed(question_seq)  # [batch_size, seq_len, embed_dim]
        a_embeds = self.answer_embed(answer_seq)      # [batch_size, seq_len, embed_dim]

        # Combine question and answer embeddings
        # SAKT uses: [question_embed, answer_embed] concatenation
        combined_embeds = q_embeds + a_embeds  # [batch_size, seq_len, embed_dim]

        # Add positional encoding
        combined_embeds = self.pos_embed(combined_embeds)

        # Apply transformer (self-attention)
        # Create attention mask to prevent looking ahead
        mask = self._generate_square_subsequent_mask(seq_len).to(question_seq.device)
        encoded = self.transformer(combined_embeds, mask=mask)  # [batch_size, seq_len, embed_dim]

        # For prediction, use the last position's encoding
        if target_question is not None:
            # Use target question embedding as query
            target_q_embed = self.question_embed(target_question)  # [batch_size, embed_dim]

            # Attention between target question and encoded sequence
            # Simplified: use dot-product attention
            attention_scores = torch.matmul(target_q_embed.unsqueeze(1), encoded.transpose(-2, -1))  # [batch_size, 1, seq_len]
            attention_scores = attention_scores / math.sqrt(self.embed_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, seq_len]

            # Weighted sum of encoded sequence
            context = torch.matmul(attention_weights, encoded).squeeze(1)  # [batch_size, embed_dim]

            # Combine target question and context
            prediction_input = target_q_embed + context  # [batch_size, embed_dim]
        else:
            # Use last position
            prediction_input = encoded[:, -1, :]  # [batch_size, embed_dim]

        # Predict probability
        predictions = self.output_layer(prediction_input).squeeze(-1)  # [batch_size]

        return predictions

    def _generate_square_subsequent_mask(self, sz):
        """Generate mask to prevent attention to future positions"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def predict_single_concept(self, question_seq, answer_seq, target_concept):
        """
        Predict probability for a specific concept

        Args:
            question_seq: question sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_concept: target concept index [batch_size]

        Returns:
            predictions: predicted probabilities [batch_size]
        """
        # For SAKT, we predict based on question sequence
        # Simplified: use last question as target
        batch_size, seq_len = question_seq.size()
        target_question = question_seq[:, -1]  # Use last question as proxy

        return self.forward(question_seq, answer_seq, target_question=target_question)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class SAKTLoss(nn.Module):
    """SAKT Loss Function"""

    def __init__(self):
        super(SAKTLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Compute SAKT loss

        Args:
            predictions: model predictions [batch_size]
            targets: ground truth labels [batch_size]

        Returns:
            loss: loss value
        """
        return self.bce_loss(predictions, targets.float())


if __name__ == "__main__":
    # Test SAKT model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    model = SAKT(n_questions, n_concepts, embed_dim=200, num_heads=8)
    loss_fn = SAKTLoss()

    batch_size = 4
    seq_len = 50

    question_seq = torch.randint(0, n_questions, (batch_size, seq_len))
    answer_seq = torch.randint(0, 2, (batch_size, seq_len))
    target_concept = torch.randint(0, n_concepts, (batch_size,))
    labels = torch.randint(0, 2, (batch_size,)).float()

    predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
    loss = loss_fn(predictions, labels)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("SAKT model test passed!")

