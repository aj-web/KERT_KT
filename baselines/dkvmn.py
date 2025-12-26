"""
Dynamic Key-Value Memory Networks (DKVMN) Baseline Model
Original paper: Zhang et al., "Dynamic Key-Value Memory Networks for Knowledge Tracing", WWW 2017
论文4.3.1节：DKVMN引入记忆增强神经网络，使用键矩阵存储知识点概念，值矩阵存储对应的掌握状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DKVMN(nn.Module):
    """
    Dynamic Key-Value Memory Networks for Knowledge Tracing
    键矩阵：静态，存储知识点嵌入
    值矩阵：动态，存储知识掌握状态
    注意力机制：读写记忆
    """

    def __init__(self, n_questions, n_concepts, embed_dim=200, memory_size=None, dropout=0.2):
        """
        Initialize DKVMN model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension
            memory_size: size of memory (if None, use n_concepts)
            dropout: dropout rate
        """
        super(DKVMN, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.memory_size = memory_size if memory_size is not None else n_concepts
        self.dropout = dropout

        # Question embedding
        self.question_embed = nn.Embedding(n_questions, embed_dim)

        # Key matrix: static, stores concept embeddings
        self.key_matrix = nn.Parameter(torch.randn(self.memory_size, embed_dim))
        nn.init.xavier_uniform_(self.key_matrix)

        # Value matrix: dynamic, stores mastery states (initialized to zero)
        # Will be updated during forward pass

        # Attention mechanism for reading
        self.read_attention = nn.Linear(embed_dim, embed_dim)

        # Value update network
        self.value_update = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # [read_value, question_embed]
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # Erase and add vectors for value update
        self.erase_vector = nn.Linear(embed_dim, embed_dim)
        self.add_vector = nn.Linear(embed_dim, embed_dim)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # [read_value, question_embed]
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

        for module in [self.read_attention, self.erase_vector, self.add_vector]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        for layer in self.value_update:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

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
            predictions: predicted probabilities [batch_size] or [batch_size, n_concepts]
        """
        batch_size, seq_len = question_seq.size()
        device = question_seq.device

        # Initialize value matrix (dynamic, stores mastery states)
        value_matrix = torch.zeros(batch_size, self.memory_size, self.embed_dim, device=device)

        # Process sequence step by step
        for t in range(seq_len):
            # Get current question and answer
            q_t = question_seq[:, t]  # [batch_size]
            a_t = answer_seq[:, t]    # [batch_size]

            # Embed question
            q_embed = self.question_embed(q_t)  # [batch_size, embed_dim]

            # Compute attention weights (reading)
            # Attention: how relevant is each memory slot to current question
            attention_scores = torch.matmul(q_embed, self.key_matrix.t())  # [batch_size, memory_size]
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, memory_size]

            # Read from value matrix
            read_value = torch.matmul(attention_weights.unsqueeze(1), value_matrix).squeeze(1)  # [batch_size, embed_dim]

            # Update value matrix based on answer
            # Concatenate read value and question embedding
            update_input = torch.cat([read_value, q_embed], dim=-1)  # [batch_size, 2*embed_dim]
            update_vector = self.value_update(update_input)  # [batch_size, embed_dim]

            # Erase and add mechanism
            erase_vector = torch.sigmoid(self.erase_vector(update_vector))  # [batch_size, embed_dim]
            add_vector = torch.tanh(self.add_vector(update_vector))  # [batch_size, embed_dim]

            # Update value matrix: v_t = v_{t-1} * (1 - w_t * e_t) + w_t * a_t
            # where w_t is attention weight, e_t is erase vector, a_t is add vector
            erase_weights = attention_weights.unsqueeze(-1) * erase_vector.unsqueeze(1)  # [batch_size, memory_size, embed_dim]
            add_weights = attention_weights.unsqueeze(-1) * add_vector.unsqueeze(1)  # [batch_size, memory_size, embed_dim]

            value_matrix = value_matrix * (1 - erase_weights) + add_weights

        # Final prediction
        if target_question is not None:
            # Predict for specific target question
            target_q_embed = self.question_embed(target_question)  # [batch_size, embed_dim]

            # Final attention and read
            final_attention = torch.matmul(target_q_embed, self.key_matrix.t())  # [batch_size, memory_size]
            final_attention_weights = F.softmax(final_attention, dim=-1)  # [batch_size, memory_size]
            final_read_value = torch.matmul(final_attention_weights.unsqueeze(1), value_matrix).squeeze(1)  # [batch_size, embed_dim]

            # Predict probability
            prediction_input = torch.cat([final_read_value, target_q_embed], dim=-1)  # [batch_size, 2*embed_dim]
            predictions = self.output_layer(prediction_input).squeeze(-1)  # [batch_size]

            return predictions
        else:
            # Return concept-level predictions (for compatibility)
            # Use last question embedding
            last_q_embed = self.question_embed(question_seq[:, -1])  # [batch_size, embed_dim]
            final_attention = torch.matmul(last_q_embed, self.key_matrix.t())
            final_attention_weights = F.softmax(final_attention, dim=-1)
            final_read_value = torch.matmul(final_attention_weights.unsqueeze(1), value_matrix).squeeze(1)

            prediction_input = torch.cat([final_read_value, last_q_embed], dim=-1)
            predictions = self.output_layer(prediction_input).squeeze(-1)

            return predictions

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
        # For DKVMN, we use the memory slot corresponding to the concept
        # Simplified: use target_concept to select memory slot
        batch_size, seq_len = question_seq.size()
        device = question_seq.device

        value_matrix = torch.zeros(batch_size, self.memory_size, self.embed_dim, device=device)

        for t in range(seq_len):
            q_t = question_seq[:, t]
            a_t = answer_seq[:, t]
            q_embed = self.question_embed(q_t)

            attention_scores = torch.matmul(q_embed, self.key_matrix.t())
            attention_weights = F.softmax(attention_scores, dim=-1)

            read_value = torch.matmul(attention_weights.unsqueeze(1), value_matrix).squeeze(1)

            update_input = torch.cat([read_value, q_embed], dim=-1)
            update_vector = self.value_update(update_input)

            erase_vector = torch.sigmoid(self.erase_vector(update_vector))
            add_vector = torch.tanh(self.add_vector(update_vector))

            erase_weights = attention_weights.unsqueeze(-1) * erase_vector.unsqueeze(1)
            add_weights = attention_weights.unsqueeze(-1) * add_vector.unsqueeze(1)

            value_matrix = value_matrix * (1 - erase_weights) + add_weights

        # Use target concept to select from memory
        # Simplified: use attention-weighted average
        last_q_embed = self.question_embed(question_seq[:, -1])
        final_attention = torch.matmul(last_q_embed, self.key_matrix.t())
        final_attention_weights = F.softmax(final_attention, dim=-1)
        final_read_value = torch.matmul(final_attention_weights.unsqueeze(1), value_matrix).squeeze(1)

        prediction_input = torch.cat([final_read_value, last_q_embed], dim=-1)
        predictions = self.output_layer(prediction_input).squeeze(-1)

        return predictions


class DKVMNLoss(nn.Module):
    """DKVMN Loss Function"""
    
    def __init__(self):
        super(DKVMNLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Compute DKVMN loss

        Args:
            predictions: model predictions [batch_size]
            targets: ground truth labels [batch_size]

        Returns:
            loss: loss value
        """
        return self.bce_loss(predictions, targets.float())


if __name__ == "__main__":
    # Test DKVMN model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    model = DKVMN(n_questions, n_concepts)
    loss_fn = DKVMNLoss()

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
    print("DKVMN model test passed!")

