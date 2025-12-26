"""
Actor-Critic Module for Adaptive Threshold Optimization
Implements reinforcement learning for triple decision threshold tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """
    Actor Network: Learns threshold adjustment policy
    Maps state to action probabilities (threshold adjustments)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize Actor network

        Args:
            state_dim: dimension of state representation
            action_dim: number of possible actions per threshold
            hidden_dim: hidden layer dimension
        """
        super(ActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)  # 2 thresholds: alpha and beta
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        Forward pass: compute action probabilities

        Args:
            state: current state [batch_size, state_dim]

        Returns:
            action_probs: action probabilities [batch_size, 2, action_dim]
        """
        logits = self.policy_net(state)  # [batch_size, 2 * action_dim]
        logits = logits.view(-1, 2, self.action_dim)  # [batch_size, 2, action_dim]

        # Apply softmax to get probabilities
        action_probs = F.softmax(logits, dim=-1)  # [batch_size, 2, action_dim]

        return action_probs

    def sample_action(self, state, deterministic=False):
        """
        Sample actions from policy

        Args:
            state: current state [batch_size, state_dim]
            deterministic: if True, select argmax action

        Returns:
            actions: sampled actions [batch_size, 2]
            log_probs: log probabilities of actions [batch_size, 2]
        """
        action_probs = self.forward(state)  # [batch_size, 2, action_dim]

        actions = []
        log_probs = []

        for i in range(action_probs.size(1)):  # For each threshold (alpha, beta)
            threshold_probs = action_probs[:, i, :]  # [batch_size, action_dim]

            if deterministic:
                action = torch.argmax(threshold_probs, dim=-1)  # [batch_size]
            else:
                dist = Categorical(threshold_probs)
                action = dist.sample()  # [batch_size]
                log_prob = dist.log_prob(action)  # [batch_size]
                log_probs.append(log_prob)

            actions.append(action)

        actions = torch.stack(actions, dim=-1)  # [batch_size, 2]

        if not deterministic:
            log_probs = torch.stack(log_probs, dim=-1)  # [batch_size, 2]

        return actions, log_probs


class CriticNetwork(nn.Module):
    """
    Critic Network: Estimates state value function
    Evaluates the quality of current state
    """

    def __init__(self, state_dim, hidden_dim=256):
        """
        Initialize Critic network

        Args:
            state_dim: dimension of state representation
            hidden_dim: hidden layer dimension
        """
        super(CriticNetwork, self).__init__()

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        Forward pass: estimate state value

        Args:
            state: current state [batch_size, state_dim]

        Returns:
            value: estimated state value [batch_size, 1]
        """
        value = self.value_net(state)
        return value


class ActorCritic(nn.Module):
    """
    Actor-Critic module for threshold optimization
    Combines Actor and Critic networks with training logic
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99,
                 lr_actor=1e-4, lr_critic=1e-4, alpha_min=0.5, beta_max=0.5):
        """
        Initialize Actor-Critic module

        Args:
            state_dim: state dimension
            action_dim: actions per threshold
            hidden_dim: hidden layer dimension
            gamma: discount factor
            lr_actor: actor learning rate
            lr_critic: critic learning rate
            alpha_min: minimum alpha threshold
            beta_max: maximum beta threshold
        """
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.beta_max = beta_max

        # Actor and Critic networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Threshold step sizes (discrete actions)
        self.threshold_steps = torch.linspace(0.05, 0.15, action_dim)  # Action space

        # Current thresholds
        self.current_alpha = 0.7
        self.current_beta = 0.3

    def get_state_representation(self, lstm_hidden, thresholds, region_stats):
        """
        Construct state representation for Actor-Critic

        Args:
            lstm_hidden: LSTM hidden state [batch_size, hidden_dim]
            thresholds: current thresholds [alpha, beta]
            region_stats: region statistics [pos_count, bound_count, neg_count]

        Returns:
            state: state representation [batch_size, state_dim]
        """
        # Concatenate all state components
        state_components = [
            lstm_hidden,  # Knowledge state
            torch.tensor(thresholds, device=lstm_hidden.device).unsqueeze(0).expand(lstm_hidden.size(0), -1),  # Current thresholds
            torch.tensor(region_stats, device=lstm_hidden.device).unsqueeze(0).expand(lstm_hidden.size(0), -1)  # Region statistics
        ]

        state = torch.cat(state_components, dim=-1)
        return state

    def select_action(self, state, deterministic=False):
        """
        Select threshold adjustment actions

        Args:
            state: current state [batch_size, state_dim]
            deterministic: deterministic action selection

        Returns:
            actions: selected actions [batch_size, 2]
            log_probs: action log probabilities [batch_size, 2]
        """
        actions, log_probs = self.actor.sample_action(state, deterministic)

        # Convert discrete actions to threshold adjustments
        alpha_adjustments = self.threshold_steps[actions[:, 0]]  # Alpha adjustments
        beta_adjustments = self.threshold_steps[actions[:, 1]]   # Beta adjustments

        # Apply adjustments (with some randomness for exploration)
        if not deterministic:
            noise_alpha = torch.randn_like(alpha_adjustments) * 0.01
            noise_beta = torch.randn_like(beta_adjustments) * 0.01
            alpha_adjustments += noise_alpha
            beta_adjustments += noise_beta

        return actions, log_probs, alpha_adjustments, beta_adjustments

    def update_thresholds(self, alpha_adjustment, beta_adjustment):
        """
        Update current thresholds based on actions

        Args:
            alpha_adjustment: alpha threshold adjustment
            beta_adjustment: beta threshold adjustment

        Returns:
            new_alpha: updated alpha threshold
            new_beta: updated beta threshold
        """
        # Update thresholds
        new_alpha = self.current_alpha + alpha_adjustment.item()
        new_beta = self.current_beta + beta_adjustment.item()

        # Clip to valid ranges
        new_alpha = np.clip(new_alpha, self.alpha_min, 1.0)
        new_beta = np.clip(new_beta, 0.0, self.beta_max)

        # Ensure alpha > beta with minimum gap
        min_gap = 0.1
        if new_alpha - new_beta < min_gap:
            # Adjust to maintain minimum gap
            center = (new_alpha + new_beta) / 2
            new_alpha = min(center + min_gap / 2, 1.0)
            new_beta = max(center - min_gap / 2, 0.0)

        self.current_alpha = new_alpha
        self.current_beta = new_beta

        return new_alpha, new_beta

    def compute_reward(self, auc_improvement, region_balance, threshold_stability):
        """
        Compute reward signal for reinforcement learning

        Args:
            auc_improvement: AUC improvement compared to baseline
            region_balance: balance score of three regions (lower is better)
            threshold_stability: stability score (lower threshold changes is better)

        Returns:
            reward: computed reward
        """
        # Reward components
        accuracy_reward = auc_improvement * 10.0  # Main reward: AUC improvement

        # Balance reward: penalize extreme region distributions
        balance_penalty = region_balance * 2.0

        # Stability reward: penalize large threshold changes
        stability_penalty = threshold_stability * 1.0

        # Total reward
        reward = accuracy_reward - balance_penalty - stability_penalty

        return reward

    def update(self, states, actions, rewards, next_states, dones, log_probs):
        """
        Update Actor and Critic networks

        Args:
            states: batch of states [batch_size, state_dim]
            actions: batch of actions [batch_size, 2]
            rewards: batch of rewards [batch_size]
            next_states: batch of next states [batch_size, state_dim]
            dones: batch of done flags [batch_size]
            log_probs: batch of action log probabilities [batch_size, 2]
        """
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)

        # Critic update
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)  # [batch_size]
            target_values = rewards + self.gamma * next_values * (1 - dones)  # [batch_size]

        current_values = self.critic(states).squeeze(-1)  # [batch_size]
        critic_loss = F.mse_loss(current_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (PPO-style)
        advantages = target_values - current_values.detach()  # [batch_size]

        # Compute new log probabilities
        new_action_probs = self.actor(states)  # [batch_size, 2, action_dim]
        new_log_probs = []

        for i in range(actions.size(1)):  # For each threshold
            action_indices = actions[:, i]  # [batch_size]
            threshold_probs = new_action_probs[:, i, :]  # [batch_size, action_dim]
            dist = Categorical(threshold_probs)
            new_log_prob = dist.log_prob(action_indices)  # [batch_size]
            new_log_probs.append(new_log_prob)

        new_log_probs = torch.stack(new_log_probs, dim=-1)  # [batch_size, 2]

        # PPO objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages.unsqueeze(-1)

        actor_loss = -torch.mean(torch.min(surr1, surr2))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def get_current_thresholds(self):
        """Get current threshold values"""
        return self.current_alpha, self.current_beta


class ExperienceBuffer:
    """
    Experience replay buffer for Actor-Critic training
    """

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, log_prob):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done, log_prob)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones, log_probs = zip(*batch)

        return states, actions, rewards, next_states, dones, log_probs

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # Test Actor-Critic module
    state_dim = 128 + 2 + 3  # lstm_hidden + thresholds + region_stats
    action_dim = 5  # 5 possible actions per threshold

    # Initialize module
    ac = ActorCritic(state_dim, action_dim)

    # Test state construction
    batch_size = 4
    lstm_hidden = torch.randn(batch_size, 128)
    thresholds = [0.7, 0.3]
    region_stats = [40, 30, 20]  # pos, bound, neg counts

    state = ac.get_state_representation(lstm_hidden, thresholds, region_stats)
    print(f"State shape: {state.shape}")

    # Test action selection
    actions, log_probs, alpha_adj, beta_adj = ac.select_action(state)
    print(f"Actions shape: {actions.shape}")
    print(f"Alpha adjustments: {alpha_adj}")
    print(f"Beta adjustments: {beta_adj}")

    # Test threshold update
    new_alpha, new_beta = ac.update_thresholds(alpha_adj[0], beta_adj[0])
    print(f"Updated thresholds: alpha={new_alpha:.3f}, beta={new_beta:.3f}")

    # Test reward computation
    reward = ac.compute_reward(auc_improvement=0.02, region_balance=0.1, threshold_stability=0.05)
    print(f"Computed reward: {reward:.3f}")

    print("Actor-Critic module test passed!")
