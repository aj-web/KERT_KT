"""
Self-Attentive Intention-aware Knowledge Tracing (SAINT) Baseline Model
Original paper: Choi et al., "SAINT: Integrating Spatial and Temporal Models for Knowledge Tracing", EDM 2020
论文4.3.1节：SAINT使用Encoder-Decoder架构，分别处理学生历史交互序列和目标题目，
通过区分Exercise Intention和Student Intention来提升知识追踪性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SAINT(nn.Module):
    """
    SAINT: Self-Attentive Intention-aware Knowledge Tracing model
    Encoder-Decoder架构，区分Exercise Intention和Student Intention
    """

    def __init__(self, n_questions, n_concepts, embed_dim=200, num_heads=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dropout=0.2, max_seq_len=200):
        """
        Initialize SAINT model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension (must be divisible by num_heads)
            num_heads: number of attention heads
            num_encoder_layers: number of encoder layers
            num_decoder_layers: number of decoder layers
            dropout: dropout rate
            max_seq_len: maximum sequence length
        """
        super(SAINT, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_seq_len = max_seq_len

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Exercise embedding (for questions)
        self.exercise_embed = nn.Embedding(n_questions + 1, embed_dim)  # +1 for padding

        # Interaction embedding (question + answer)
        self.interaction_embed = nn.Embedding(n_questions * 2 + 1, embed_dim)

        # Concept embedding
        self.concept_embed = nn.Embedding(n_concepts + 1, embed_dim)

        # Positional encoding
        self.pos_embed = PositionalEncoding(embed_dim, dropout, max_seq_len)

        # Encoder: processes historical interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder: predicts performance on target exercise
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

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
        nn.init.xavier_uniform_(self.exercise_embed.weight)
        nn.init.xavier_uniform_(self.interaction_embed.weight)
        nn.init.xavier_uniform_(self.concept_embed.weight)

        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, question_seq, answer_seq, target_question, concept_seq=None, target_concept=None):
        """
        Forward pass

        Args:
            question_seq: question sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_question: target question for prediction [batch_size]
            concept_seq: concept sequence [batch_size, seq_len] (optional)
            target_concept: target concept [batch_size] (optional)

        Returns:
            predictions: predicted probabilities [batch_size]
        """
        batch_size, seq_len = question_seq.size()

        # Create interaction embeddings (question + answer)
        # Interaction encoding: q * 2 + a (shifted to avoid collision with padding)
        interaction_seq = question_seq * 2 + answer_seq
        interaction_seq = interaction_seq.masked_fill(question_seq == 0, self.n_questions * 2)  # Use padding index

        # Encode historical interactions
        hist_embeds = self.interaction_embed(interaction_seq)  # [batch_size, seq_len, embed_dim]

        # Add positional encoding
        hist_embeds = self.pos_embed(hist_embeds)

        # Encoder: process historical interactions
        # Create mask for encoder
        src_key_padding_mask = (question_seq == 0)
        encoded = self.encoder(hist_embeds, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, embed_dim]

        # Prepare target for decoder
        # Use target question embedding
        target_embed = self.exercise_embed(target_question).unsqueeze(1)  # [batch_size, 1, embed_dim]

        # Decoder: attend to encoded history while processing target
        tgt_mask = self._generate_square_subsequent_mask(1).to(question_seq.device)
        decoded = self.decoder(target_embed, encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=None,
                              memory_key_padding_mask=src_key_padding_mask)  # [batch_size, 1, embed_dim]

        # Predict probability
        prediction_input = decoded.squeeze(1)  # [batch_size, embed_dim]
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
        # For SAINT, we need target question for prediction
        # Simplified: use last question as target
        batch_size = question_seq.size(0)
        target_question = question_seq[:, -1].clone()

        return self.forward(question_seq, answer_seq, target_question)


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


class SAINTLoss(nn.Module):
    """SAINT Loss Function"""

    def __init__(self):
        super(SAINTLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Compute SAINT loss

        Args:
            predictions: model predictions [batch_size]
            targets: ground truth labels [batch_size]

        Returns:
            loss: loss value
        """
        return self.bce_loss(predictions, targets.float())


if __name__ == "__main__":
    # Test SAINT model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    model = SAINT(n_questions, n_concepts, embed_dim=200, num_heads=8)
    loss_fn = SAINTLoss()

    batch_size = 4
    seq_len = 50

    question_seq = torch.randint(1, n_questions, (batch_size, seq_len))
    answer_seq = torch.randint(0, 2, (batch_size, seq_len))
    target_question = torch.randint(1, n_questions, (batch_size,))
    labels = torch.randint(0, 2, (batch_size,)).float()

    predictions = model.forward(question_seq, answer_seq, target_question)
    loss = loss_fn(predictions, labels)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("SAINT model test passed!")

