"""
Knowledge Tracing Predictor Module
Implements LSTM-based sequence modeling and attention-based prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KTPredictor(nn.Module):
    """
    Knowledge Tracing Predictor with LSTM and Attention
    Models student learning sequences and predicts performance
    """

    def __init__(self, n_questions, n_concepts, embed_dim, hidden_dim,
                 concept_graph_embed, dropout=0.2):
        """
        Initialize KT Predictor

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension
            hidden_dim: LSTM hidden dimension
            concept_graph_embed: enhanced concept embeddings from graph module
            dropout: dropout rate
        """
        super(KTPredictor, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.question_embed = nn.Embedding(n_questions, embed_dim)
        self.concept_embed = nn.Embedding(n_concepts, embed_dim)

        # Use enhanced concept embeddings from graph module
        if concept_graph_embed is not None:
            self.concept_embed.weight.data = concept_graph_embed.clone()

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim * 3,  # question + concept + answer
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        # Attention mechanism
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim, hidden_dim)

        # Prediction network
        predictor_input_dim = hidden_dim * 2 + embed_dim * 2  # current_state + context + question + concept
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
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
        # Embedding初始化：使用较小的标准差，避免初始值过大
        nn.init.normal_(self.question_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.concept_embed.weight, mean=0.0, std=0.02)

        # Attention weights：使用Xavier初始化
        for module in [self.attention_query, self.attention_key, self.attention_value]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Predictor weights：使用He初始化（适合ReLU）和较小的bias
        for i, layer in enumerate(self.predictor):
            if isinstance(layer, nn.Linear):
                # 使用He初始化（适合ReLU激活）
                nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
                # 最后一层（输出层）使用较小的bias，避免初始输出接近0.5
                if i == len([l for l in self.predictor if isinstance(l, nn.Linear)]) - 1:
                    # 最后一层：bias初始化为0，让初始预测更均匀
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.zeros_(layer.bias)

    def forward(self, question_seq, concept_seq, answer_seq, target_question, target_concept):
        """
        Forward pass for prediction

        Args:
            question_seq: question sequence [batch_size, seq_len]
            concept_seq: concept sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]
            target_question: target question [batch_size]
            target_concept: target concept [batch_size]

        Returns:
            predictions: predicted probabilities [batch_size]
            hidden_states: LSTM hidden states [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = question_seq.size()

        # Embed sequences
        question_embeds = self.question_embed(question_seq)  # [batch_size, seq_len, embed_dim]
        concept_embeds = self.concept_embed(concept_seq)     # [batch_size, seq_len, embed_dim]
        answer_embeds = answer_seq.unsqueeze(-1).float()     # [batch_size, seq_len, 1]

        # Concatenate inputs for LSTM
        lstm_inputs = torch.cat([
            question_embeds,  # [batch_size, seq_len, embed_dim]
            concept_embeds,   # [batch_size, seq_len, embed_dim]
            answer_embeds.expand(-1, -1, self.embed_dim)  # [batch_size, seq_len, embed_dim]
        ], dim=-1)  # [batch_size, seq_len, 3*embed_dim]

        # LSTM encoding
        lstm_outputs, (h_n, c_n) = self.lstm(lstm_inputs)  # [batch_size, seq_len, hidden_dim]

        # Current state (last hidden state)
        current_state = h_n.squeeze(0)  # [batch_size, hidden_dim]

        # Attention-based context aggregation
        context_vector = self._attention_mechanism(lstm_outputs)  # [batch_size, hidden_dim]

        # Target embeddings
        target_question_embed = self.question_embed(target_question)  # [batch_size, embed_dim]
        target_concept_embed = self.concept_embed(target_concept)     # [batch_size, embed_dim]

        # Prediction
        prediction_input = torch.cat([
            current_state,       # [batch_size, hidden_dim]
            context_vector,      # [batch_size, hidden_dim]
            target_question_embed,  # [batch_size, embed_dim]
            target_concept_embed    # [batch_size, embed_dim]
        ], dim=-1)  # [batch_size, hidden_dim * 2 + embed_dim * 2]

        predictions = self.predictor(prediction_input).squeeze(-1)  # [batch_size]

        return predictions, lstm_outputs

    def _attention_mechanism(self, lstm_outputs):
        """
        Attention mechanism for context aggregation

        Args:
            lstm_outputs: LSTM outputs [batch_size, seq_len, hidden_dim]

        Returns:
            context_vector: aggregated context [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = lstm_outputs.size()

        # Attention computation
        query = self.attention_query(lstm_outputs[:, -1, :]).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        key = self.attention_key(lstm_outputs)    # [batch_size, seq_len, hidden_dim]
        value = self.attention_value(lstm_outputs)  # [batch_size, seq_len, hidden_dim]

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_dim ** 0.5)  # [batch_size, 1, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, 1, seq_len]

        # Context aggregation
        context_vector = torch.matmul(attention_weights, value).squeeze(1)  # [batch_size, hidden_dim]

        return context_vector

    def get_hidden_states(self, question_seq, concept_seq, answer_seq):
        """
        Get LSTM hidden states for a sequence (used by Actor-Critic)

        Args:
            question_seq: question sequence [batch_size, seq_len]
            concept_seq: concept sequence [batch_size, seq_len]
            answer_seq: answer sequence [batch_size, seq_len]

        Returns:
            hidden_states: LSTM hidden states [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = question_seq.size()

        # Embed sequences
        question_embeds = self.question_embed(question_seq)
        concept_embeds = self.concept_embed(concept_seq)
        answer_embeds = answer_seq.unsqueeze(-1).float()

        # Concatenate inputs
        lstm_inputs = torch.cat([
            question_embeds,
            concept_embeds,
            answer_embeds.expand(-1, -1, self.embed_dim)
        ], dim=-1)

        # LSTM encoding
        lstm_outputs, _ = self.lstm(lstm_inputs)

        return lstm_outputs


class KTLoss(nn.Module):
    """
    Knowledge Tracing Loss Function
    Binary cross-entropy with regularization
    """

    def __init__(self, l2_lambda=1e-5):
        """
        Initialize KT Loss

        Args:
            l2_lambda: L2 regularization coefficient
        """
        super(KTLoss, self).__init__()
        self.l2_lambda = l2_lambda
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets, model):
        """
        Compute knowledge tracing loss

        Args:
            predictions: model predictions [batch_size]
            targets: ground truth labels [batch_size]
            model: model instance for L2 regularization

        Returns:
            total_loss: total loss value
        """
        # Binary cross-entropy loss
        bce_loss = self.bce_loss(predictions, targets.float())

        # L2 regularization
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)

        total_loss = bce_loss + self.l2_lambda * l2_reg

        return total_loss, bce_loss, l2_reg


class DataCollator:
    """
    Data collator for batch processing
    """

    def __init__(self, max_seq_len=200):
        self.max_seq_len = max_seq_len

    def collate_fn(self, batch):
        """
        Collate batch of sequences

        Args:
            batch: list of (question_seq, concept_seq, answer_seq, target_question, target_concept, label)

        Returns:
            collated batch
        """
        question_seqs, concept_seqs, answer_seqs = [], [], []
        target_questions, target_concepts, labels = [], [], []

        for item in batch:
            q_seq, c_seq, a_seq, target_q, target_c, label = item

            # Pad or truncate sequences
            if len(q_seq) > self.max_seq_len:
                q_seq = q_seq[-self.max_seq_len:]
                c_seq = c_seq[-self.max_seq_len:]
                a_seq = a_seq[-self.max_seq_len:]

            question_seqs.append(torch.tensor(q_seq, dtype=torch.long))
            concept_seqs.append(torch.tensor(c_seq, dtype=torch.long))
            answer_seqs.append(torch.tensor(a_seq, dtype=torch.long))

            target_questions.append(target_q)
            target_concepts.append(target_c)
            labels.append(label)

        # Pad sequences to same length
        max_len = max(len(seq) for seq in question_seqs)
        max_len = min(max_len, self.max_seq_len)  # 限制最大长度

        padded_question_seqs = []
        padded_concept_seqs = []
        padded_answer_seqs = []
        attention_masks = []  # 用于标记padding位置

        for q_seq, c_seq, a_seq in zip(question_seqs, concept_seqs, answer_seqs):
            seq_len = len(q_seq)
            pad_len = max_len - seq_len

            if pad_len > 0:
                # 左侧padding（保留最近的交互）
                # 使用mask标记padding位置，而不是使用0（因为question_id可能从0开始）
                q_seq = F.pad(q_seq, (pad_len, 0), value=0)
                c_seq = F.pad(c_seq, (pad_len, 0), value=0)
                a_seq = F.pad(a_seq, (pad_len, 0), value=0)
                
                # 创建attention mask：1表示真实位置，0表示padding
                mask = torch.zeros(max_len, dtype=torch.bool)
                mask[pad_len:] = True  # 真实数据位置为True
                attention_masks.append(mask)
            else:
                # 不需要padding
                mask = torch.ones(seq_len, dtype=torch.bool)
                attention_masks.append(mask)

            padded_question_seqs.append(q_seq)
            padded_concept_seqs.append(c_seq)
            padded_answer_seqs.append(a_seq)

        # Stack tensors
        question_seqs = torch.stack(padded_question_seqs)
        concept_seqs = torch.stack(padded_concept_seqs)
        answer_seqs = torch.stack(padded_answer_seqs)
        attention_masks = torch.stack(attention_masks)  # [batch_size, seq_len]
        target_questions = torch.tensor(target_questions, dtype=torch.long)
        target_concepts = torch.tensor(target_concepts, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            'question_seq': question_seqs,
            'concept_seq': concept_seqs,
            'answer_seq': answer_seqs,
            'target_question': target_questions,
            'target_concept': target_concepts,
            'labels': labels,
            'attention_mask': attention_masks  # 添加mask用于忽略padding
        }


if __name__ == "__main__":
    # Test KT Predictor module
    n_questions = 17751  # ASSIST09
    n_concepts = 124
    embed_dim = 128
    hidden_dim = 256

    # Create enhanced concept embeddings (simulated)
    concept_graph_embed = torch.randn(n_concepts, embed_dim)

    # Initialize model
    model = KTPredictor(n_questions, n_concepts, embed_dim, hidden_dim, concept_graph_embed)
    loss_fn = KTLoss()

    # Test data
    batch_size = 4
    seq_len = 50

    question_seq = torch.randint(0, n_questions, (batch_size, seq_len))
    concept_seq = torch.randint(0, n_concepts, (batch_size, seq_len))
    answer_seq = torch.randint(0, 2, (batch_size, seq_len))
    target_question = torch.randint(0, n_questions, (batch_size,))
    target_concept = torch.randint(0, n_concepts, (batch_size,))
    labels = torch.randint(0, 2, (batch_size,)).float()

    # Forward pass
    predictions, hidden_states = model(question_seq, concept_seq, answer_seq, target_question, target_concept)
    loss, bce_loss, l2_reg = loss_fn(predictions, labels, model)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Total loss: {loss.item():.4f} (BCE: {bce_loss.item():.4f}, L2: {l2_reg.item():.6f})")

    # Test hidden states extraction
    hidden_only = model.get_hidden_states(question_seq, concept_seq, answer_seq)
    print(f"Hidden states only shape: {hidden_only.shape}")

    print("KT Predictor module test passed!")
