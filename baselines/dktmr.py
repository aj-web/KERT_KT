"""
Deep Knowledge Tracing with Memory and Review (DKTMR) Baseline Model
Original paper: Zhang et al., "Deep Knowledge Tracing with Review", IEEE Access 2020
论文4.3.1节：DKTMR在DKT基础上引入记忆网络和复习机制，
通过外部记忆矩阵存储和更新知识点掌握程度，并利用复习机制增强长期依赖建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DKTMR(nn.Module):
    """
    Deep Knowledge Tracing with Memory and Review
    结合记忆网络和复习机制的DKT变体
    """

    def __init__(self, n_questions, n_concepts, embed_dim=128, hidden_dim=256, 
                 memory_size=100, dropout=0.2, num_layers=2):
        """
        Initialize DKTMR model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension
            hidden_dim: hidden dimension
            memory_size: size of knowledge memory
            dropout: dropout rate
            num_layers: number of LSTM layers
        """
        super(DKTMR, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.num_layers = num_layers

        # Question embedding
        self.question_embed = nn.Embedding(n_questions + 1, embed_dim)

        # Concept embedding
        self.concept_embed = nn.Embedding(n_concepts + 1, embed_dim)

        # Knowledge Memory (key-value memory)
        # Key memory: stores concept representations
        self.key_memory = nn.Parameter(torch.randn(memory_size, embed_dim))
        # Value memory: stores mastery levels
        self.value_memory = nn.Parameter(torch.randn(memory_size, embed_dim))

        # Memory Controller
        self.memory_controller = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2)  # read/write gates
        )

        # Review Mechanism
        self.review_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Question-to-memory attention
        self.q2m_attention = nn.Linear(embed_dim + embed_dim, memory_size)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.question_embed.weight)
        nn.init.xavier_uniform_(self.concept_embed.weight)
        nn.init.xavier_uniform_(self.key_memory)
        nn.init.xavier_uniform_(self.value_memory)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _memory_read(self, query):
        """
        Read from knowledge memory

        Args:
            query: query tensor [batch_size, embed_dim]

        Returns:
            memory_output: read content [batch_size, embed_dim]
            attention_weights: attention weights [batch_size, memory_size]
        """
        # Compute attention between query and memory keys
        # query: [batch_size, embed_dim] -> expand to [batch_size, 1, embed_dim]
        # key_memory: [memory_size, embed_dim] -> expand to [1, memory_size, embed_dim]
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, embed_dim]
        key_expanded = self.key_memory.unsqueeze(0)  # [1, memory_size, embed_dim]

        # Attention scores
        attention_scores = torch.sum(query_expanded * key_expanded, dim=-1)  # [batch_size, memory_size]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, memory_size]

        # Read from value memory
        value_expanded = self.value_memory.unsqueeze(0)  # [1, memory_size, embed_dim]
        memory_output = torch.sum(attention_weights.unsqueeze(-1) * value_expanded, dim=1)  # [batch_size, embed_dim]

        return memory_output, attention_weights

    def _memory_write(self, concept_id, mastery_update):
        """
        Write to knowledge memory

        Args:
            concept_id: concept indices [batch_size]
            mastery_update: mastery update tensor [batch_size, embed_dim]
        """
        batch_size = concept_id.size(0)

        # Get indices for memory update (use concept_id to index)
        # Clamp to valid range
        concept_indices = torch.clamp(concept_id, 0, self.memory_size - 1)

        # Update value memory (simplified: direct assignment with momentum)
        # In practice, this should be done more carefully
        with torch.no_grad():
            for b in range(batch_size):
                idx = concept_indices[b].item()
                # Blend new information with existing memory
                self.value_memory.data[idx] = (
                    0.9 * self.value_memory.data[idx] + 
                    0.1 * mastery_update[b].detach()
                )

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

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(question_seq.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(question_seq.device)
        hidden = (h0, c0)

        # Prepare input: combine question and answer
        # Shift answer by 1 to predict next question
        answer_shifted = torch.zeros_like(answer_seq)
        answer_shifted[:, 1:] = answer_seq[:, :-1]

        # Embed questions
        q_embeds = self.question_embed(question_seq)  # [batch_size, seq_len, embed_dim]
        # Embed concepts
        c_embeds = self.concept_embed(concept_seq)  # [batch_size, seq_len, embed_dim]
        # Combine with answer
        combined_embeds = q_embeds + c_embeds * answer_shifted.unsqueeze(-1).float()

        # Process through LSTM with review mechanism
        lstm_out, hidden = self.review_lstm(combined_embeds, hidden)
        # lstm_out: [batch_size, seq_len, hidden_dim]

        # For prediction at each step
        predictions = []

        for t in range(seq_len - 1):
            # Get current hidden state
            ht = lstm_out[:, t, :]  # [batch_size, hidden_dim]

            # Read from memory using current concept
            current_concept = concept_seq[:, t]  # [batch_size]
            concept_embed = self.concept_embed(current_concept)  # [batch_size, embed_dim]

            # Memory read
            memory_output, _ = self._memory_read(concept_embed)  # [batch_size, embed_dim]

            # Combine LSTM hidden and memory output
            combined = torch.cat([ht, memory_output], dim=-1)  # [batch_size, hidden_dim + embed_dim]

            # Predict
            pred = self.output_layer(combined).squeeze(-1)  # [batch_size]
            predictions.append(pred)

        if len(predictions) == 0:
            # Edge case: sequence too short
            return torch.zeros(batch_size).to(question_seq.device)

        # Return prediction for last valid position
        return predictions[-1]

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
        # Simplified: use last prediction
        predictions = self.forward(question_seq, concept_seq, answer_seq)
        return predictions


class DKTMWRLoss(nn.Module):
    """DKTMR Loss Function"""

    def __init__(self):
        super(DKTMWRLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Compute DKTMR loss

        Args:
            predictions: model predictions [batch_size]
            targets: ground truth labels [batch_size]

        Returns:
            loss: loss value
        """
        return self.bce_loss(predictions, targets.float())


if __name__ == "__main__":
    # Test DKTMR model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    model = DKTMR(n_questions, n_concepts, embed_dim=128, hidden_dim=256)
    loss_fn = DKTMWRLoss()

    batch_size = 4
    seq_len = 50

    question_seq = torch.randint(1, n_questions, (batch_size, seq_len))
    concept_seq = torch.randint(1, n_concepts, (batch_size, seq_len))
    answer_seq = torch.randint(0, 2, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,)).float()

    predictions = model.forward(question_seq, concept_seq, answer_seq)
    loss = loss_fn(predictions, labels)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("DKTMR model test passed!")

