"""
Deep Knowledge Tracing (DKT) Baseline Model
Original paper: Piech et al., "Deep Knowledge Tracing", NIPS 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DKT(nn.Module):
    """
    Deep Knowledge Tracing model
    Uses LSTM to model student learning sequences
    """

    def __init__(self, n_questions, n_concepts, embed_dim=100, hidden_dim=100, dropout=0.2):
        """
        Initialize DKT model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension
            hidden_dim: LSTM hidden dimension
            dropout: dropout rate
        """
        super(DKT, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Question embedding
        self.question_embed = nn.Embedding(n_questions, embed_dim)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,  # question + answer
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        # Output layer (predicts probability for all concepts)
        self.output_layer = nn.Linear(hidden_dim, n_concepts)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.question_embed.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, question_seq, answer_seq, target_question=None, attention_mask=None):
        """
        Forward pass

        Args:
            question_seq: question sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_question: target question for prediction [batch_size] (optional)
            attention_mask: mask for padding positions [batch_size, seq_len] (optional)

        Returns:
            predictions: predicted probabilities [batch_size, n_concepts] or [batch_size]
        """
        batch_size, seq_len = question_seq.size()

        # Embed questions
        question_embeds = self.question_embed(question_seq)  # [batch_size, seq_len, embed_dim]

        # DKT输入编码：根据原始论文，使用 [question * answer, question * (1-answer)]
        # 这样可以更好地编码问题和答案的交互
        answer_expanded = answer_seq.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        answer_expanded = answer_expanded.expand(-1, -1, self.embed_dim)  # [batch_size, seq_len, embed_dim]
        
        # 正确回答：question * answer
        input_correct = question_embeds * answer_expanded  # [batch_size, seq_len, embed_dim]
        # 错误回答：question * (1-answer)
        input_wrong = question_embeds * (1 - answer_expanded)  # [batch_size, seq_len, embed_dim]
        
        # 拼接：[correct, wrong]
        lstm_inputs = torch.cat([input_correct, input_wrong], dim=-1)  # [batch_size, seq_len, 2*embed_dim]

        # 处理padding：使用pack_padded_sequence处理变长序列
        if attention_mask is not None:
            # 计算每个序列的实际长度（非padding长度）
            lengths = attention_mask.sum(dim=1).cpu().long()  # [batch_size]
            
            # 过滤掉长度为0的序列（虽然不应该有，但安全起见）
            valid_mask = lengths > 0
            if valid_mask.all():
                # 所有序列都有效
                lstm_inputs_packed = nn.utils.rnn.pack_padded_sequence(
                    lstm_inputs, lengths, batch_first=True, enforce_sorted=False
                )
                lstm_outputs_packed, (h_n, c_n) = self.lstm(lstm_inputs_packed)
                # 解包
                lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_outputs_packed, batch_first=True
                )
                # 使用最后一个有效位置的hidden state
                final_hidden = h_n.squeeze(0)  # [batch_size, hidden_dim]
            else:
                # 有无效序列，需要特殊处理
                # 对于长度为0的序列，使用零向量
                final_hidden = torch.zeros(batch_size, self.hidden_dim, device=question_seq.device)
                if valid_mask.any():
                    valid_inputs = lstm_inputs[valid_mask]
                    valid_lengths = lengths[valid_mask]
                    lstm_inputs_packed = nn.utils.rnn.pack_padded_sequence(
                        valid_inputs, valid_lengths, batch_first=True, enforce_sorted=False
                    )
                    _, (h_n_valid, _) = self.lstm(lstm_inputs_packed)
                    final_hidden[valid_mask] = h_n_valid.squeeze(0)
        else:
            # 如果没有mask，使用原来的方式
            lstm_outputs, (h_n, c_n) = self.lstm(lstm_inputs)
            # Get last hidden state
            final_hidden = lstm_outputs[:, -1, :]  # [batch_size, hidden_dim]

        # Predict probabilities for all concepts
        concept_logits = self.output_layer(final_hidden)  # [batch_size, n_concepts]
        concept_probs = torch.sigmoid(concept_logits)  # [batch_size, n_concepts]

        if target_question is not None:
            # For single concept prediction, get concept associated with target question
            # This requires Q-matrix mapping - simplified assumption here
            target_concept = target_question % self.n_concepts  # Simplified mapping
            predictions = concept_probs[torch.arange(batch_size), target_concept]
            return predictions
        else:
            return concept_probs

    def predict_single_concept(self, question_seq, answer_seq, target_concept, attention_mask=None):
        """
        Predict probability for a specific concept

        Args:
            question_seq: question sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_concept: target concept index [batch_size]
            attention_mask: mask for padding positions [batch_size, seq_len] (optional)

        Returns:
            predictions: predicted probabilities [batch_size]
        """
        concept_probs = self.forward(question_seq, answer_seq, attention_mask=attention_mask)  # [batch_size, n_concepts]
        predictions = concept_probs[torch.arange(concept_probs.size(0)), target_concept]
        return predictions


class DKTLoss(nn.Module):
    """
    DKT Loss Function
    Binary cross-entropy for concept prediction
    """

    def __init__(self):
        super(DKTLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Compute DKT loss

        Args:
            predictions: model predictions [batch_size] or [batch_size, n_concepts]
            targets: ground truth labels [batch_size]

        Returns:
            loss: loss value
        """
        return self.bce_loss(predictions, targets.float())


def create_dkt_input_sequences(df, max_seq_len=200):
    """
    Create input sequences for DKT from DataFrame

    Args:
        df: DataFrame with student sequences
        max_seq_len: maximum sequence length

    Returns:
        sequences: list of (question_seq, answer_seq, target_question, target_answer, target_concept)
    """
    sequences = []

    # Group by student
    for student_id, group in df.groupby('student_id'):
        group = group.sort_values('timestamp')

        questions = group['question_id'].values
        answers = group['correct'].values
        concepts = group['concept_id'].values

        # Create sequences
        seq_len = len(questions)
        for i in range(1, seq_len):
            q_seq = questions[:i]
            a_seq = answers[:i]

            target_q = questions[i]
            target_a = answers[i]
            target_c = concepts[i]

            # Pad/truncate sequences
            if len(q_seq) > max_seq_len:
                q_seq = q_seq[-max_seq_len:]
                a_seq = a_seq[-max_seq_len:]

            sequences.append((q_seq, a_seq, target_q, target_a, target_c))

    return sequences


if __name__ == "__main__":
    # Test DKT model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    # Initialize model
    model = DKT(n_questions, n_concepts)
    loss_fn = DKTLoss()

    # Test data
    batch_size = 4
    seq_len = 50

    question_seq = torch.randint(0, n_questions, (batch_size, seq_len))
    answer_seq = torch.randint(0, 2, (batch_size, seq_len))
    target_concept = torch.randint(0, n_concepts, (batch_size,))
    labels = torch.randint(0, 2, (batch_size,)).float()

    # Forward pass
    predictions = model.predict_single_concept(question_seq, answer_seq, target_concept)
    loss = loss_fn(predictions, labels)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Loss: {loss.item():.4f}")

    print("DKT model test passed!")
