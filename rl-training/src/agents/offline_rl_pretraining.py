"""
Offline Reinforcement Learning Pretraining Module

This module implements offline RL algorithms for pretraining agents on historical data:
- Conservative Q-Learning (CQL)
- Batch-Constrained Deep Q-Learning (BCQ)
- Behavior Cloning (BC)
- Dataset preprocessing and augmentation
- Offline evaluation metrics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import random
from sklearn.preprocessing import StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)

class OfflineAlgorithm(Enum):
    CQL = "cql"  # Conservative Q-Learning
    BCQ = "bcq"  # Batch-Constrained Deep Q-Learning
    BC = "bc"    # Behavior Cloning
    BEAR = "bear"  # Bootstrapping Error Accumulation Reduction

@dataclass
class OfflineConfig:
    """Configuration for offline RL training"""
    algorithm: OfflineAlgorithm
    dataset_path: str
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    learning_rate: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 1000
    validation_split: float = 0.2
    cql_alpha: float = 1.0  # CQL regularization weight
    bcq_threshold: float = 0.3  # BCQ threshold for action selection
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class OfflineDataset:
    """Offline dataset for RL training"""
    
    def __init__(self, data: Dict[str, np.ndarray], config: OfflineConfig):
        self.data = data
        self.config = config
        self.scaler = StandardScaler()
        
        # Normalize states
        self.states = self.scaler.fit_transform(data['states'])
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_states = self.scaler.transform(data['next_states'])
        self.dones = data['dones']
        
        # Create train/validation split
        self._create_splits()
    
    def _create_splits(self):
        """Create train/validation splits"""
        n_samples = len(self.states)
        val_size = int(n_samples * self.config.validation_split)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        self.train_data = {
            'states': self.states[train_indices],
            'actions': self.actions[train_indices],
            'rewards': self.rewards[train_indices],
            'next_states': self.next_states[train_indices],
            'dones': self.dones[train_indices]
        }
        
        self.val_data = {
            'states': self.states[val_indices],
            'actions': self.actions[val_indices],
            'rewards': self.rewards[val_indices],
            'next_states': self.next_states[val_indices],
            'dones': self.dones[val_indices]
        }
    
    def get_batch(self, batch_size: int, split: str = 'train') -> Dict[str, torch.Tensor]:
        """Get batch of data"""
        data = self.train_data if split == 'train' else self.val_data
        n_samples = len(data['states'])
        
        if batch_size > n_samples:
            batch_size = n_samples
        
        indices = np.random.choice(n_samples, batch_size, replace=False)
        
        batch = {}
        for key in data:
            batch[key] = torch.FloatTensor(data[key][indices])
        
        return batch

class ConservativeQLearning(nn.Module):
    """Conservative Q-Learning implementation"""
    
    def __init__(self, config: OfflineConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Q-networks
        self.q1 = self._build_q_network(config.state_dim, config.action_dim, config.hidden_dims)
        self.q2 = self._build_q_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Target networks
        self.target_q1 = self._build_q_network(config.state_dim, config.action_dim, config.hidden_dims)
        self.target_q2 = self._build_q_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Copy weights to target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.learning_rate)
        
        # CQL parameters
        self.alpha = config.cql_alpha
        self.tau = 0.005
        self.gamma = 0.99
        
    def _build_q_network(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build Q-network"""
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        q1_input = torch.cat([states, actions], dim=-1)
        q2_input = torch.cat([states, actions], dim=-1)
        
        q1 = self.q1(q1_input)
        q2 = self.q2(q2_input)
        
        return q1, q2
    
    def get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get Q-values for given state-action pairs"""
        q1, q2 = self.forward(states, actions)
        return torch.min(q1, q2)
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update CQL agent"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Current Q-values
        current_q1, current_q2 = self.forward(states, actions)
        
        # Target Q-values
        with torch.no_grad():
            # Use current policy for target (simplified)
            target_q1, target_q2 = self.target_q1(torch.cat([next_states, actions], dim=-1)), \
                                 self.target_q2(torch.cat([next_states, actions], dim=-1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(-1) + self.gamma * target_q * (~dones).unsqueeze(-1)
        
        # Standard Q-learning loss
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        # CQL regularization
        # Sample random actions for CQL penalty
        random_actions = torch.rand_like(actions) * 2 - 1  # Assume actions in [-1, 1]
        random_q1, random_q2 = self.forward(states, random_actions)
        
        # CQL loss: maximize Q-values for dataset actions, minimize for random actions
        cql_loss1 = torch.mean(random_q1) - torch.mean(current_q1)
        cql_loss2 = torch.mean(random_q2) - torch.mean(current_q2)
        
        # Total loss
        total_loss1 = q1_loss + self.alpha * cql_loss1
        total_loss2 = q2_loss + self.alpha * cql_loss2
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        total_loss1.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        total_loss2.backward()
        self.q2_optimizer.step()
        
        # Update target networks
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql_loss1': cql_loss1.item(),
            'cql_loss2': cql_loss2.item(),
            'total_loss1': total_loss1.item(),
            'total_loss2': total_loss2.item()
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class BatchConstrainedQNetwork(nn.Module):
    """Batch-Constrained Deep Q-Network implementation"""
    
    def __init__(self, config: OfflineConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Q-network
        self.q_network = self._build_q_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # VAE for action generation
        self.vae = self._build_vae(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Target network
        self.target_q_network = self._build_q_network(config.state_dim, config.action_dim, config.hidden_dims)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=config.learning_rate)
        
        # BCQ parameters
        self.threshold = config.bcq_threshold
        self.tau = 0.005
        self.gamma = 0.99
        
    def _build_q_network(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build Q-network"""
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def _build_vae(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build VAE for action generation"""
        # Encoder
        encoder_layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Latent dimension
        latent_dim = action_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim * 2))  # mean and log_var
        
        # Decoder
        decoder_layers = []
        prev_dim = state_dim + latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, action_dim))
        decoder_layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        return nn.ModuleDict({
            'encoder': nn.Sequential(*encoder_layers),
            'decoder': nn.Sequential(*decoder_layers)
        })
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass for Q-network"""
        q_input = torch.cat([states, actions], dim=-1)
        return self.q_network(q_input)
    
    def generate_action(self, states: torch.Tensor) -> torch.Tensor:
        """Generate action using VAE"""
        with torch.no_grad():
            # Sample from latent space
            latent = torch.randn(states.size(0), self.config.action_dim).to(self.device)
            
            # Decode to action
            vae_input = torch.cat([states, latent], dim=-1)
            actions = self.vae['decoder'](vae_input)
            
            return actions
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update BCQ agent"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update VAE
        vae_loss = self._update_vae(states, actions)
        
        # Update Q-network
        q_loss = self._update_q_network(states, actions, rewards, next_states, dones)
        
        # Update target network
        self._soft_update(self.target_q_network, self.q_network)
        
        return {
            'vae_loss': vae_loss,
            'q_loss': q_loss
        }
    
    def _update_vae(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update VAE"""
        # Encode
        vae_input = torch.cat([states, actions], dim=-1)
        encoder_output = self.vae['encoder'](vae_input)
        mean, log_var = torch.chunk(encoder_output, 2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        
        # Decode
        decoder_input = torch.cat([states, latent], dim=-1)
        reconstructed_actions = self.vae['decoder'](decoder_input)
        
        # VAE loss
        reconstruction_loss = nn.MSELoss()(reconstructed_actions, actions)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        vae_loss = reconstruction_loss + 0.5 * kl_loss
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        return vae_loss.item()
    
    def _update_q_network(self, states: torch.Tensor, actions: torch.Tensor, 
                         rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> float:
        """Update Q-network"""
        # Current Q-values
        current_q = self.forward(states, actions)
        
        # Target Q-values
        with torch.no_grad():
            # Generate actions using VAE
            next_actions = self.generate_action(next_states)
            
            # Apply threshold
            next_actions = next_actions * self.threshold + actions * (1 - self.threshold)
            
            target_q = self.target_q_network(torch.cat([next_states, next_actions], dim=-1))
            target_q = rewards.unsqueeze(-1) + self.gamma * target_q * (~dones).unsqueeze(-1)
        
        # Q-learning loss
        q_loss = nn.MSELoss()(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return q_loss.item()
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class BehaviorCloning(nn.Module):
    """Behavior Cloning implementation"""
    
    def __init__(self, config: OfflineConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Policy network
        self.policy_network = self._build_policy_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        
    def _build_policy_network(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build policy network"""
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        return nn.Sequential(*layers)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.policy_network(states)
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update behavior cloning agent"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        
        # Predict actions
        predicted_actions = self.forward(states)
        
        # MSE loss
        loss = nn.MSELoss()(predicted_actions, actions)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'bc_loss': loss.item()}

class OfflineRLTrainer:
    """Offline RL trainer"""
    
    def __init__(self, config: OfflineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Initialize algorithm
        if config.algorithm == OfflineAlgorithm.CQL:
            self.agent = ConservativeQLearning(config)
        elif config.algorithm == OfflineAlgorithm.BCQ:
            self.agent = BatchConstrainedQNetwork(config)
        elif config.algorithm == OfflineAlgorithm.BC:
            self.agent = BehaviorCloning(config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        
        self.agent.to(self.device)
        
    def _load_dataset(self) -> OfflineDataset:
        """Load offline dataset"""
        with open(self.config.dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        return OfflineDataset(data, self.config)
    
    def train(self) -> Dict[str, List[float]]:
        """Train offline RL agent"""
        training_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_batch = self.dataset.get_batch(self.config.batch_size, 'train')
            train_metrics = self.agent.update(train_batch)
            
            # Validation
            val_batch = self.dataset.get_batch(self.config.batch_size, 'val')
            val_metrics = self._evaluate(val_batch)
            
            # Record metrics
            if train_metrics:
                training_history['train_loss'].append(list(train_metrics.values())[0])
            if val_metrics:
                training_history['val_loss'].append(list(val_metrics.values())[0])
            
            # Log progress
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {training_history['train_loss'][-1]:.4f}, "
                          f"Val Loss = {training_history['val_loss'][-1]:.4f}")
        
        return training_history
    
    def _evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate agent on batch"""
        with torch.no_grad():
            if self.config.algorithm == OfflineAlgorithm.BC:
                # Behavior cloning evaluation
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                predicted_actions = self.agent.forward(states)
                loss = nn.MSELoss()(predicted_actions, actions)
                return {'bc_loss': loss.item()}
            else:
                # Q-learning evaluation
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                q_values = self.agent.get_q_values(states, actions)
                return {'q_value': q_values.mean().item()}
    
    def save_agent(self, path: str):
        """Save trained agent"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'config': self.config,
            'scaler': self.dataset.scaler
        }, path)
    
    def load_agent(self, path: str):
        """Load trained agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.dataset.scaler = checkpoint['scaler']

def create_offline_dataset(market_data: pd.DataFrame, 
                          signal_data: pd.DataFrame,
                          actions: np.ndarray,
                          rewards: np.ndarray) -> Dict[str, np.ndarray]:
    """Create offline dataset from market data"""
    # Prepare states (market data + signals)
    states = np.concatenate([
        market_data[['Open', 'High', 'Low', 'Close', 'Volume']].values,
        signal_data.select_dtypes(include=[np.number]).values
    ], axis=1)
    
    # Prepare next states (shifted by 1)
    next_states = np.roll(states, -1, axis=0)
    
    # Prepare dones (end of episode indicators)
    dones = np.zeros(len(states), dtype=bool)
    dones[-1] = True  # Last timestep is done
    
    return {
        'states': states[:-1],  # Exclude last state
        'actions': actions[:-1],  # Exclude last action
        'rewards': rewards[:-1],  # Exclude last reward
        'next_states': next_states[:-1],  # Exclude last next_state
        'dones': dones[:-1]  # Exclude last done
    }
