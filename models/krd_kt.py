"""
KRD-KT: Knowledge point Representation-Driven Knowledge Tracing
Complete implementation integrating all modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import os
from tqdm import tqdm

from .triple_decision_graph import TripleDecisionGraph
from .actor_critic import ActorCritic, ExperienceBuffer
from .kt_predictor import KTPredictor, KTLoss, DataCollator
from .question_enhancement import QuestionEnhancement
from .neighborhood_extractor import NeighborhoodExtractor
from .path_strength import PrecomputedPathStrengthCalculator


class KTSequenceDataset(Dataset):
    """
    Dataset for Knowledge Tracing sequences
    """

    def __init__(self, data_df, max_seq_len=200):
        """
        Initialize dataset

        Args:
            data_df: DataFrame with columns [student_id, question_id, concept_id, correct, timestamp]
            max_seq_len: maximum sequence length
        """
        self.data_df = data_df
        self.max_seq_len = max_seq_len

        # Group by student
        self.student_groups = data_df.groupby('student_id')

        # Create sequence data
        self.sequences = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        """Prepare sequences for each student"""
        for student_id, group in self.student_groups:
            # Sort by timestamp if available, otherwise assume data is already sorted
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
            # If no timestamp, data is assumed to be in chronological order already

            questions = group['question_id'].values
            concepts = group['concept_id'].values
            answers = group['correct'].values

            # Create sliding windows for training
            seq_len = len(questions)
            if seq_len < 2:
                continue

            for i in range(1, seq_len):
                # Input sequence: first i interactions
                q_seq = questions[:i]
                c_seq = concepts[:i]
                a_seq = answers[:i]

                # Target: i-th interaction
                target_q = questions[i]
                target_c = concepts[i]
                target_a = answers[i]

                self.sequences.append((q_seq, c_seq, a_seq, target_q, target_c, target_a))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class KRDKT(nn.Module):
    """
    Complete KER-KT model integrating all components
    """

    def __init__(self, n_questions, n_concepts, embed_dim=128, hidden_dim=256,
                 n_layers=2, lstm_layers=2, alpha=0.7, beta=0.3, lambda_decay=0.1,
                 gamma=0.99, lr_kt=1e-3, lr_rl=1e-4, lambda_rl=0.1, 
                 l2_lambda=1e-5, dropout=0.2):
        """
        Initialize KRD-KT model

        Args:
            n_questions: number of questions
            n_concepts: number of concepts
            embed_dim: embedding dimension
            hidden_dim: LSTM hidden dimension
            n_layers: graph propagation layers
            lstm_layers: number of LSTM layers
            alpha, beta: initial triple decision thresholds
            lambda_decay: negative region decay factor
            gamma: RL discount factor
            lr_kt: KT learning rate
            lr_rl: RL learning rate
            lambda_rl: RL loss weight
            l2_lambda: L2 regularization coefficient
            dropout: dropout rate
        """
        super(KRDKT, self).__init__()

        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm_layers = lstm_layers
        self.alpha = alpha
        self.beta = beta

        # 题目增强模块（新增）
        self.question_enhancer = QuestionEnhancement(embed_dim, dropout=dropout)

        # 三支决策图模块（更新为新版本）
        self.graph_module = TripleDecisionGraph(
            n_concepts=n_concepts,
            embed_dim=embed_dim,
            n_layers=n_layers,
            alpha=alpha,
            beta=beta,
            max_k=2,
            distance_decay_lambda=lambda_decay,
            dropout=dropout
        )

        # 邻居提取器和路径强度计算器（训练前初始化）
        self.neighborhood_extractor = None
        self.path_strength_calculator = None
        
        # 相似度矩阵缓存
        self.similarity_matrix = None

        # 增强后的概念嵌入（动态更新）
        self.concept_embeddings = None

        # KT预测器模块
        self.kt_predictor = KTPredictor(
            n_questions, n_concepts, embed_dim, hidden_dim, self.concept_embeddings, 
            lstm_layers=lstm_layers, dropout=dropout
        )

        # Actor-Critic模块（阈值优化）
        state_dim = hidden_dim + 2 + 3  # lstm_hidden + thresholds + region_stats
        action_dim = 5  # 5 actions per threshold
        self.actor_critic = ActorCritic(
            state_dim, action_dim, gamma=gamma, lr_actor=lr_rl, lr_critic=lr_rl
        )

        # 经验回放缓冲区
        self.experience_buffer = ExperienceBuffer(capacity=1000)

        # 损失函数
        self.kt_loss_fn = KTLoss(l2_lambda=l2_lambda)

        # Optimizers
        kt_params = list(self.kt_predictor.parameters()) + list(self.graph_module.parameters())
        self.kt_optimizer = optim.Adam(kt_params, lr=lr_kt)

        # Training hyperparameters
        self.lambda_rl = lambda_rl
        self.current_thresholds = (alpha, beta)

        # Training state
        self.graph_trained = False
        self.rl_enabled = False
        
        # For RL reward calculation: cache validation AUC
        self.last_val_auc = 0.0
        self.val_loader = None
        self.eval_frequency = 10  # Evaluate every N batches
        self.batch_count = 0
        self.concept_embeddings_update_frequency = 50  # 优化：每N个batch更新一次concept_embeddings（10 → 50，减少计算）
        self._concept_graph_hash = None  # 用于检测concept_graph是否变化
    
    def initialize_graph_modules(self, concept_graph):
        """
        初始化图相关模块（在训练开始前调用）
        
        Args:
            concept_graph: [n_concepts, n_concepts] 概念图邻接矩阵
        """
        print("初始化图模块...")
        
        # 1. 初始化邻居提取器
        print("  - 提取k阶邻居...")
        self.neighborhood_extractor = NeighborhoodExtractor(
            concept_graph, max_k=2, directed=False
        )
        
        # 2. 初始化路径强度计算器
        print("  - 预计算路径强度...")
        self.path_strength_calculator = PrecomputedPathStrengthCalculator(
            concept_graph, self.neighborhood_extractor
        )
        
        print("图模块初始化完成！")
    
    def update_epoch_cache(self):
        """
        在每个epoch开始时更新缓存（相似度矩阵等）
        """
        # 计算当前概念嵌入的余弦相似度矩阵
        concept_embeds = self.graph_module.concept_embed.weight
        normalized = F.normalize(concept_embeds, p=2, dim=-1)
        self.similarity_matrix = torch.matmul(normalized, normalized.T)

    def forward(self, batch, concept_graph, use_question_enhancement=True):
        """
        前向传播（集成所有新模块）
        
        流程：
        1. （可选）题目增强
        2. 三支决策图传播（使用预计算的邻居和路径强度）
        3. KT预测

        Args:
            batch: batch data dictionary
            concept_graph: concept adjacency matrix [n_concepts, n_concepts]
            use_question_enhancement: 是否使用题目增强

        Returns:
            predictions: predicted probabilities
            hidden_states: LSTM hidden states
        """
        # 1. 题目增强（可选）
        if use_question_enhancement and 'target_question' in batch:
            # 获取目标题目的嵌入
            target_q_embed = self.kt_predictor.question_embed(batch['target_question'])
            
            # 增强概念嵌入
            concept_embeds = self.graph_module.concept_embed.weight
            enhanced_concepts = self.question_enhancer(target_q_embed, concept_embeds)
            
            # 如果是batch，取平均（或使用第一个）
            if enhanced_concepts.dim() == 3:  # [batch_size, n_concepts, embed_dim]
                enhanced_concepts = enhanced_concepts.mean(dim=0)  # [n_concepts, embed_dim]
            
            # 更新图模块的概念嵌入（临时）
            self.graph_module.concept_embed.weight.data = enhanced_concepts.detach()
        
        # 2. 三支决策图传播
        if self.neighborhood_extractor is not None and self.path_strength_calculator is not None:
            # 使用新版本的图传播（带k阶邻居和路径强度）
            neighborhoods = self.neighborhood_extractor.neighborhoods
            strength_matrices = {
                1: self.path_strength_calculator.get_strength_matrix(1),
                2: self.path_strength_calculator.get_strength_matrix(2)
            }
            self.concept_embeddings = self.graph_module(
                neighborhoods, strength_matrices, self.similarity_matrix
            )
        else:
            # 回退到简单版本（向后兼容）
            self.concept_embeddings = self.graph_module.concept_embed.weight
        
        # 3. 更新KT预测器的概念嵌入
        self.kt_predictor.concept_embed.weight.data = self.concept_embeddings.detach()

        # 4. KT预测
        predictions, hidden_states = self.kt_predictor(
            batch['question_seq'],
            batch['concept_seq'],
            batch['answer_seq'],
            batch['target_question'],
            batch['target_concept']
        )

        return predictions, hidden_states

    def train_step(self, batch, concept_graph):
        """
        训练步骤（简化版，逻辑已移至forward）

        Args:
            batch: training batch
            concept_graph: concept adjacency matrix

        Returns:
            losses: dictionary of loss values
        """
        self.batch_count += 1

        # Forward传播
        predictions, hidden_states = self.forward(batch, concept_graph)

        # KT损失
        kt_loss, bce_loss, l2_reg = self.kt_loss_fn(
            predictions, batch['labels'], self.kt_predictor
        )

        # KT优化
        self.kt_optimizer.zero_grad()
        kt_loss.backward()
        self.kt_optimizer.step()

        losses = {
            'kt_loss': kt_loss.item(),
            'bce_loss': bce_loss.item(),
            'l2_reg': l2_reg.item()
        }

        # RL training (if enabled)
        if self.rl_enabled:
            rl_losses = self._rl_train_step(batch, hidden_states, concept_graph)
            losses.update(rl_losses)

        return losses

    def set_val_loader(self, val_loader):
        """Set validation loader for reward calculation"""
        self.val_loader = val_loader

    def _rl_train_step(self, batch, hidden_states, concept_graph):
        """
        Reinforcement learning training step (论文3.4.1节)

        Args:
            batch: training batch
            hidden_states: LSTM hidden states
            concept_graph: concept adjacency matrix

        Returns:
            rl_losses: RL loss values
        """
        batch_size = batch['question_seq'].size(0)
        self.batch_count += 1

        # Get current thresholds
        current_alpha, current_beta = self.actor_critic.get_current_thresholds()

        # Compute region statistics (基于实际图结构和阈值)
        region_stats = self._compute_region_stats(concept_graph, current_alpha, current_beta)

        # Construct states for each sample
        states = []
        for i in range(batch_size):
            lstm_hidden = hidden_states[i, -1, :]  # Last hidden state
            state = self.actor_critic.get_state_representation(
                lstm_hidden.unsqueeze(0),
                [current_alpha, current_beta],
                region_stats
            )
            states.append(state.squeeze(0))

        states = torch.stack(states)

        # Select actions
        actions, log_probs, alpha_adjustments, beta_adjustments = self.actor_critic.select_action(states)

        # Apply threshold updates
        new_alphas, new_betas = [], []
        for i in range(batch_size):
            new_alpha, new_beta = self.actor_critic.update_thresholds(
                alpha_adjustments[i], beta_adjustments[i]
            )
            new_alphas.append(new_alpha)
            new_betas.append(new_beta)

        # Update graph module thresholds
        avg_alpha = np.mean(new_alphas)
        avg_beta = np.mean(new_betas)
        self.graph_module.update_thresholds(avg_alpha, avg_beta)
        self.current_thresholds = (avg_alpha, avg_beta)

        # Compute rewards (论文3.4.1节：使用真实验证集AUC)
        # Evaluate on validation set periodically to get real AUC
        if self.val_loader is not None and self.batch_count % self.eval_frequency == 0:
            val_metrics = self.evaluate(self.val_loader, concept_graph)
            current_val_auc = val_metrics['auc']
            auc_improvement = current_val_auc - self.last_val_auc
            self.last_val_auc = current_val_auc
        else:
            # Use cached AUC improvement if not evaluating this batch
            auc_improvement = 0.0  # Will be updated on next evaluation

        # Compute rewards for each sample in batch
        rewards = []
        for i in range(batch_size):
            # 论文3.4.1节：三部分奖励
            # 1. 准确性奖励：验证集AUC改进
            # 2. 平衡性奖励：避免三域分布极端
            region_balance = np.std(region_stats) / (np.mean(region_stats) + 1e-8)
            # 3. 稳定性奖励：抑制阈值剧烈波动
            threshold_stability = abs(alpha_adjustments[i].item()) + abs(beta_adjustments[i].item())

            reward = self.actor_critic.compute_reward(
                auc_improvement, region_balance, threshold_stability
            )
            rewards.append(reward)

        # Store experiences
        next_states = states  # Simplified - in practice would use next state
        dones = [False] * batch_size

        for i in range(batch_size):
            self.experience_buffer.push(
                states[i].detach().cpu().numpy(),
                actions[i].detach().cpu().numpy(),
                rewards[i],
                next_states[i].detach().cpu().numpy(),
                dones[i],
                log_probs[i].detach().cpu().numpy()
            )

        # Update Actor-Critic if buffer is full
        rl_losses = {'actor_loss': 0.0, 'critic_loss': 0.0}
        if len(self.experience_buffer) >= 32:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_log_probs = \
                self.experience_buffer.sample(32)

            actor_loss, critic_loss = self.actor_critic.update(
                batch_states, batch_actions, batch_rewards,
                batch_next_states, batch_dones, batch_log_probs
            )

            rl_losses = {
                'actor_loss': actor_loss,
                'critic_loss': critic_loss
            }

        return rl_losses

    def _compute_region_stats(self, concept_graph, alpha, beta):
        """
        Compute region statistics for reward calculation
        Based on actual graph structure and thresholds (论文3.4.1节)

        Args:
            concept_graph: concept adjacency matrix [n_concepts, n_concepts]
            alpha, beta: current thresholds

        Returns:
            region_stats: [|POS|_avg, |BND|_avg, |NEG|_avg] (所有知识点的三域平均节点数)
        """
        n_concepts = concept_graph.size(0)
        device = concept_graph.device
        
        # Get current concept embeddings
        if self.concept_embeddings is None:
            # Use initial embeddings if not computed yet
            concept_embeds = self.graph_module.concept_embed.weight
        else:
            concept_embeds = self.concept_embeddings
        
        # Compute similarities for all concept pairs
        # Normalize embeddings for cosine similarity
        norm_embeds = F.normalize(concept_embeds, p=2, dim=-1)
        similarity_matrix = torch.matmul(norm_embeds, norm_embeds.t())  # [n_concepts, n_concepts]
        
        # Initialize counters for each concept
        pos_counts = []
        bound_counts = []
        neg_counts = []
        
        # For each concept, compute its three regions based on neighbors
        for i in range(n_concepts):
            # Get neighbors (only consider connected concepts)
            neighbors = torch.nonzero(concept_graph[i] > 0).squeeze(-1)
            
            if len(neighbors) == 0:
                # Isolated node, no neighbors
                pos_counts.append(0)
                bound_counts.append(0)
                neg_counts.append(0)
                continue
            
            # Get similarities with neighbors
            neighbor_similarities = similarity_matrix[i, neighbors]
            
            # Triple decision classification
            pos_mask = neighbor_similarities >= alpha
            neg_mask = neighbor_similarities <= beta
            bound_mask = (neighbor_similarities > beta) & (neighbor_similarities < alpha)
            
            # Count nodes in each region
            pos_counts.append(pos_mask.sum().item())
            bound_counts.append(bound_mask.sum().item())
            neg_counts.append(neg_mask.sum().item())
        
        # Compute average counts across all concepts
        pos_avg = np.mean(pos_counts) if pos_counts else 0.0
        bound_avg = np.mean(bound_counts) if bound_counts else 0.0
        neg_avg = np.mean(neg_counts) if neg_counts else 0.0
        
        return [pos_avg, bound_avg, neg_avg]

    def evaluate(self, data_loader, concept_graph):
        """
        Evaluate model on test data

        Args:
            data_loader: test data loader
            concept_graph: concept adjacency matrix

        Returns:
            metrics: evaluation metrics
        """
        self.eval()
        all_predictions = []
        all_labels = []
        device = concept_graph.device

        with torch.no_grad():
            for batch in data_loader:
                # 将batch数据移动到正确的设备
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                predictions, _ = self.forward(batch, concept_graph)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        # Compute metrics
        auc = roc_auc_score(all_labels, all_predictions)
        acc = accuracy_score(all_labels, np.round(all_predictions))

        return {'auc': auc, 'acc': acc}

    def enable_rl_training(self):
        """Enable reinforcement learning training"""
        self.rl_enabled = True
        print("RL training enabled")

    def save_model(self, path):
        """Save model state"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'graph_module': self.graph_module.state_dict(),
            'kt_predictor': self.kt_predictor.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'concept_embeddings': self.concept_embeddings,
            'current_thresholds': self.current_thresholds
        }, path)

    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path)
        self.graph_module.load_state_dict(checkpoint['graph_module'])
        self.kt_predictor.load_state_dict(checkpoint['kt_predictor'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.concept_embeddings = checkpoint['concept_embeddings']
        self.current_thresholds = checkpoint['current_thresholds']


def train_krd_kt(model, train_loader, val_loader, concept_graph, n_epochs=100, patience=10, 
                  checkpoint_path='checkpoint_path', lr_kt_pretrain=0.001, lr_kt_finetune=0.0005,
                  warmup_steps=0, lr_decay_patience=None, lr_decay_factor=0.5):
    """
    Complete training pipeline for KER-KT (论文3.6.2节：两阶段训练策略)

    Args:
        model: KER-KT model
        train_loader: training data loader
        val_loader: validation data loader
        concept_graph: concept adjacency matrix
        n_epochs: number of training epochs
        patience: early stopping patience
        checkpoint_path: path to save best model checkpoint
        lr_kt_pretrain: 预训练阶段学习率 (论文表4.4)
        lr_kt_finetune: 微调阶段学习率 (论文表4.4)
        warmup_steps: 学习率Warmup步数（0表示不使用Warmup）
        lr_decay_patience: 学习率衰减patience（None表示不使用衰减）
        lr_decay_factor: 学习率衰减因子
    """
    # 获取设备
    device = concept_graph.device
    
    # 初始化图模块（邻居提取、路径强度计算等）
    if model.neighborhood_extractor is None:
        model.initialize_graph_modules(concept_graph)
    
    # 初始化相似度矩阵缓存
    model.update_epoch_cache()
    
    best_auc = 0.0
    patience_counter = 0
    
    # 初始化学习率调度器（如果启用）
    scheduler = None
    if lr_decay_patience is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            model.kt_optimizer, mode='max', factor=lr_decay_factor,
            patience=lr_decay_patience
        )
        print(f"  Learning rate scheduler: ReduceLROnPlateau (patience={lr_decay_patience}, factor={lr_decay_factor})")
    
    # 全局步数计数器（用于Warmup）
    global_step = 0

    # Phase 1: Knowledge Tracing Pre-training (Epochs 1-50, 论文3.6.2节)
    print("Phase 1: KT Pre-training (Epochs 1-50)")
    print(f"  Learning rate: {lr_kt_pretrain}")
    if warmup_steps > 0:
        print(f"  Warmup steps: {warmup_steps}")
    
    # Set learning rate for pre-training
    for param_group in model.kt_optimizer.param_groups:
        param_group['lr'] = lr_kt_pretrain if warmup_steps == 0 else 1e-6  # Warmup起始学习率
    
    for epoch in range(50):
        model.train()
        
        # 在每个epoch开始时更新相似度矩阵缓存
        model.update_epoch_cache()
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
            # 将batch数据移动到正确的设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 学习率Warmup
            if warmup_steps > 0 and global_step < warmup_steps:
                # 线性Warmup：从1e-6线性增加到lr_kt_pretrain
                warmup_lr = 1e-6 + (lr_kt_pretrain - 1e-6) * (global_step / warmup_steps)
                for param_group in model.kt_optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            losses = model.train_step(batch, concept_graph)
            epoch_losses.append(losses)
            global_step += 1

        # Average losses
        avg_losses = {k: np.mean([loss[k] for loss in epoch_losses]) for k in epoch_losses[0].keys()}

        # Validation
        val_metrics = model.evaluate(val_loader, concept_graph)
        
        # 学习率调度器更新（基于验证AUC）
        if scheduler is not None:
            scheduler.step(val_metrics['auc'])
            current_lr = model.kt_optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: KT Loss: {avg_losses['kt_loss']:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, Val ACC: {val_metrics['acc']:.4f}, "
                  f"LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}: KT Loss: {avg_losses['kt_loss']:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, Val ACC: {val_metrics['acc']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            model.save_model(checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Phase 1 early stopping triggered (best AUC: {best_auc:.4f})")
            break

    # Phase 2: RL Fine-tuning (Epochs 51-100, 论文3.6.2节)
    print("\n" + "="*50)
    print("Phase 2: RL Fine-tuning (Epochs 51-100)")
    print("="*50)
    print(f"  KT Learning rate: {lr_kt_finetune} (降低)")
    print(f"  RL Learning rate: {model.actor_critic.actor_optimizer.param_groups[0]['lr']}")
    print(f"  Loading best Phase 1 model (AUC: {best_auc:.4f})")
    
    model.load_model(checkpoint_path)  # Load best KT model
    model.enable_rl_training()
    model.set_val_loader(val_loader)  # Set validation loader for reward calculation
    
    # Update learning rate for fine-tuning (论文3.6.2节)
    for param_group in model.kt_optimizer.param_groups:
        param_group['lr'] = lr_kt_finetune
    
    # 重置patience计数器（Phase 2独立早停）
    patience_counter = 0
    phase2_best_auc = best_auc
    
    # 为微调阶段创建新的学习率调度器（如果需要）
    if lr_decay_patience is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            model.kt_optimizer, mode='max', factor=lr_decay_factor,
            patience=lr_decay_patience
        )

    for epoch in range(50, n_epochs):
        model.train()
        # 重置batch计数，确保每个epoch开始时更新concept_embeddings
        model.batch_count = 0
        
        # 在每个epoch开始时更新相似度矩阵缓存
        model.update_epoch_cache()
        
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            # 将batch数据移动到正确的设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            losses = model.train_step(batch, concept_graph)
            epoch_losses.append(losses)

        # Average losses
        avg_losses = {k: np.mean([loss[k] for loss in epoch_losses]) for k in epoch_losses[0].keys()}

        # Validation
        val_metrics = model.evaluate(val_loader, concept_graph)
        
        # 学习率调度器更新（基于验证AUC）
        if scheduler is not None:
            scheduler.step(val_metrics['auc'])
            current_lr = model.kt_optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: KT Loss: {avg_losses.get('kt_loss', 0):.4f}, "
                  f"Actor Loss: {avg_losses.get('actor_loss', 0):.4f}, "
                  f"Critic Loss: {avg_losses.get('critic_loss', 0):.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, Val ACC: {val_metrics['acc']:.4f}, "
                  f"LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}: KT Loss: {avg_losses.get('kt_loss', 0):.4f}, "
                  f"Actor Loss: {avg_losses.get('actor_loss', 0):.4f}, "
                  f"Critic Loss: {avg_losses.get('critic_loss', 0):.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, Val ACC: {val_metrics['acc']:.4f}")

        # Save best model (Phase 2)
        if val_metrics['auc'] > phase2_best_auc:
            phase2_best_auc = val_metrics['auc']
            best_auc = phase2_best_auc  # 更新全局最佳
            patience_counter = 0
            model.save_model(checkpoint_path)
            print(f"  -> New best model saved (AUC: {phase2_best_auc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Phase 2 early stopping triggered (best AUC: {phase2_best_auc:.4f})")
            break

    print("\n" + "="*50)
    print(f"Training completed. Best validation AUC: {best_auc:.4f}")
    print("="*50)


if __name__ == "__main__":
    # Test KRD-KT model
    n_questions = 17751  # ASSIST09
    n_concepts = 124

    # Initialize model
    model = KRDKT(n_questions, n_concepts)

    # Create dummy data for testing
    batch_size = 4
    seq_len = 50

    batch = {
        'question_seq': torch.randint(0, n_questions, (batch_size, seq_len)),
        'concept_seq': torch.randint(0, n_concepts, (batch_size, seq_len)),
        'answer_seq': torch.randint(0, 2, (batch_size, seq_len)),
        'target_question': torch.randint(0, n_questions, (batch_size,)),
        'target_concept': torch.randint(0, n_concepts, (batch_size,)),
        'labels': torch.randint(0, 2, (batch_size,)).float()
    }

    # Test forward pass
    predictions, hidden_states = model(batch)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")

    # Test training step
    concept_graph = torch.rand(n_concepts, n_concepts)
    losses = model.train_step(batch, concept_graph)

    print("Training losses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")

    print("KER-KT model test passed!")
