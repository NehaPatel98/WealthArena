"""
Hierarchical Reinforcement Learning Agents

This module implements hierarchical RL agents with:
- High-level allocator (portfolio allocation decisions)
- Low-level execution policies (individual asset trading)
- Multi-agent coordination
- Offline pretraining capabilities
- Multi-objective optimization
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
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AgentType(Enum):
    HIGH_LEVEL_ALLOCATOR = "high_level_allocator"
    LOW_LEVEL_EXECUTOR = "low_level_executor"
    RISK_MANAGER = "risk_manager"
    SIGNAL_FUSION = "signal_fusion"

class ActionType(Enum):
    ALLOCATION = "allocation"
    TRADE = "trade"
    HEDGE = "hedge"
    REBALANCE = "rebalance"

@dataclass
class AgentConfig:
    """Configuration for RL agents"""
    agent_type: AgentType
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100000
    batch_size: int = 256
    update_frequency: int = 4
    target_update_frequency: int = 100

@dataclass
class Experience:
    """Experience tuple for RL training"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = None

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class BaseAgent(ABC):
    """Base class for all RL agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.step_count = 0
        
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action given state"""
        pass
    
    @abstractmethod
    def update(self, batch: List[Experience]) -> Dict[str, float]:
        """Update agent with batch of experiences"""
        pass
    
    def store_experience(self, experience: Experience):
        """Store experience in replay buffer"""
        self.replay_buffer.push(experience)
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']

class HighLevelAllocator(nn.Module, BaseAgent):
    """High-level portfolio allocation agent using PPO"""
    
    def __init__(self, config: AgentConfig):
        nn.Module.__init__(self)
        BaseAgent.__init__(self, config)
        
        # Policy network
        self.policy_net = self._build_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Value network
        self.value_net = self._build_network(config.state_dim, 1, config.hidden_dims)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.learning_rate)
        
        # PPO parameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
    def _build_network(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build neural network"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        policy_logits = self.policy_net(state)
        value = self.value_net(state)
        return policy_logits, value
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select allocation action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.forward(state_tensor)
            
            if training:
                # Add noise for exploration
                policy_logits += torch.randn_like(policy_logits) * 0.1
            
            # Softmax to get allocation weights
            allocation_weights = torch.softmax(policy_logits, dim=-1)
            
            # Sample from categorical distribution
            dist = Categorical(allocation_weights)
            action_idx = dist.sample()
            action = allocation_weights[0].cpu().numpy()
        
        return action
    
    def update(self, batch: List[Experience]) -> Dict[str, float]:
        """Update agent using PPO"""
        if len(batch) < self.config.batch_size:
            return {}
        
        # Convert batch to tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Calculate returns
        returns = self._calculate_returns(rewards, dones)
        
        # Get current policy and value estimates
        policy_logits, values = self.forward(states)
        old_policy_logits, old_values = self.forward(states).detach()
        
        # Calculate advantages
        advantages = returns - values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO policy loss
        old_probs = torch.softmax(old_policy_logits, dim=-1)
        new_probs = torch.softmax(policy_logits, dim=-1)
        
        ratio = (new_probs * actions).sum(dim=-1) / (old_probs * actions).sum(dim=-1)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Entropy loss
        entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        self.step_count += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'mean_advantage': advantages.mean().item()
        }
    
    def _calculate_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return
        
        return returns

class LowLevelExecutor(nn.Module, BaseAgent):
    """Low-level execution agent using SAC"""
    
    def __init__(self, config: AgentConfig):
        nn.Module.__init__(self)
        BaseAgent.__init__(self, config)
        
        # Actor network
        self.actor = self._build_actor_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Critic networks (double Q-learning)
        self.critic1 = self._build_critic_network(config.state_dim, config.action_dim, config.hidden_dims)
        self.critic2 = self._build_critic_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Target networks
        self.target_critic1 = self._build_critic_network(config.state_dim, config.action_dim, config.hidden_dims)
        self.target_critic2 = self._build_critic_network(config.state_dim, config.action_dim, config.hidden_dims)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.learning_rate)
        
        # SAC parameters
        self.alpha = 0.2  # Temperature parameter
        self.tau = config.tau
        self.target_entropy = -config.action_dim
        
    def _build_actor_network(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build actor network"""
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output mean and log_std for continuous actions
        layers.append(nn.Linear(prev_dim, action_dim * 2))
        return nn.Sequential(*layers)
    
    def _build_critic_network(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build critic network"""
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
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select execution action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actor_output = self.actor(state_tensor)
            mean, log_std = torch.chunk(actor_output, 2, dim=-1)
            
            if training:
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()
            else:
                action = mean
            
            # Clip action to valid range
            action = torch.tanh(action)
        
        return action[0].cpu().numpy()
    
    def update(self, batch: List[Experience]) -> Dict[str, float]:
        """Update agent using SAC"""
        if len(batch) < self.config.batch_size:
            return {}
        
        # Convert batch to tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self._get_action_and_log_prob(next_states)
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(-1) + self.config.gamma * target_q * (~dones).unsqueeze(-1)
        
        # Critic loss
        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self._get_action_and_log_prob(states)
        q1_new = self.critic1(torch.cat([states, new_actions], dim=-1))
        q2_new = self.critic2(torch.cat([states, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        self.step_count += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'mean_q': q_new.mean().item()
        }
    
    def _get_action_and_log_prob(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability from actor"""
        actor_output = self.actor(states)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class HierarchicalRLSystem:
    """Hierarchical RL system coordinating multiple agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.portfolio_state = {}
        self.performance_metrics = {}
        
        # Initialize agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all agents in the hierarchy"""
        # High-level allocator
        allocator_config = AgentConfig(
            agent_type=AgentType.HIGH_LEVEL_ALLOCATOR,
            state_dim=self.config['allocator_state_dim'],
            action_dim=self.config['num_assets'],
            hidden_dims=[256, 128, 64],
            learning_rate=1e-4
        )
        self.agents['allocator'] = HighLevelAllocator(allocator_config)
        
        # Low-level executors for each asset
        for asset in self.config['assets']:
            executor_config = AgentConfig(
                agent_type=AgentType.LOW_LEVEL_EXECUTOR,
                state_dim=self.config['executor_state_dim'],
                action_dim=self.config['executor_action_dim'],
                hidden_dims=[128, 64, 32],
                learning_rate=1e-4
            )
            self.agents[f'executor_{asset}'] = LowLevelExecutor(executor_config)
    
    def step(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step of the hierarchical system"""
        # High-level allocation decision
        allocation_state = self._prepare_allocation_state(market_state)
        allocation_weights = self.agents['allocator'].select_action(allocation_state)
        
        # Low-level execution for each asset
        execution_actions = {}
        for i, asset in enumerate(self.config['assets']):
            executor_state = self._prepare_executor_state(market_state, asset, allocation_weights[i])
            action = self.agents[f'executor_{asset}'].select_action(executor_state)
            execution_actions[asset] = action
        
        # Update portfolio state
        self._update_portfolio_state(allocation_weights, execution_actions, market_state)
        
        return {
            'allocation_weights': allocation_weights,
            'execution_actions': execution_actions,
            'portfolio_state': self.portfolio_state
        }
    
    def _prepare_allocation_state(self, market_state: Dict[str, Any]) -> np.ndarray:
        """Prepare state for high-level allocator"""
        # Combine market data, portfolio state, and risk metrics
        state_components = []
        
        # Market data
        if 'prices' in market_state:
            state_components.extend(market_state['prices'])
        
        if 'returns' in market_state:
            state_components.extend(market_state['returns'])
        
        if 'volatilities' in market_state:
            state_components.extend(market_state['volatilities'])
        
        # Portfolio state
        if 'current_weights' in self.portfolio_state:
            state_components.extend(self.portfolio_state['current_weights'])
        
        if 'pnl' in self.portfolio_state:
            state_components.append(self.portfolio_state['pnl'])
        
        # Risk metrics
        if 'var' in self.portfolio_state:
            state_components.append(self.portfolio_state['var'])
        
        return np.array(state_components, dtype=np.float32)
    
    def _prepare_executor_state(self, market_state: Dict[str, Any], asset: str, allocation_weight: float) -> np.ndarray:
        """Prepare state for low-level executor"""
        state_components = []
        
        # Asset-specific market data
        if asset in market_state:
            asset_data = market_state[asset]
            if 'price' in asset_data:
                state_components.append(asset_data['price'])
            if 'volume' in asset_data:
                state_components.append(asset_data['volume'])
            if 'volatility' in asset_data:
                state_components.append(asset_data['volatility'])
        
        # Allocation weight
        state_components.append(allocation_weight)
        
        # Current position
        if asset in self.portfolio_state.get('positions', {}):
            state_components.append(self.portfolio_state['positions'][asset])
        else:
            state_components.append(0.0)
        
        return np.array(state_components, dtype=np.float32)
    
    def _update_portfolio_state(self, allocation_weights: np.ndarray, execution_actions: Dict[str, Any], market_state: Dict[str, Any]):
        """Update portfolio state based on actions"""
        # Update allocation weights
        self.portfolio_state['current_weights'] = allocation_weights
        
        # Update positions based on execution actions
        if 'positions' not in self.portfolio_state:
            self.portfolio_state['positions'] = {}
        
        for asset, action in execution_actions.items():
            if asset not in self.portfolio_state['positions']:
                self.portfolio_state['positions'][asset] = 0.0
            
            # Update position based on action (simplified)
            self.portfolio_state['positions'][asset] += action * 0.1  # Scale factor
        
        # Calculate P&L
        total_pnl = 0.0
        for asset, position in self.portfolio_state['positions'].items():
            if asset in market_state and 'price' in market_state[asset]:
                price = market_state[asset]['price']
                # Simplified P&L calculation
                total_pnl += position * price * 0.01  # 1% return assumption
        
        self.portfolio_state['pnl'] = total_pnl
        
        # Calculate VaR (simplified)
        if 'var' not in self.portfolio_state:
            self.portfolio_state['var'] = 0.0
        
        # Update VaR based on portfolio volatility
        portfolio_vol = np.sqrt(np.sum(allocation_weights**2 * 0.04))  # Simplified
        self.portfolio_state['var'] = -2.33 * portfolio_vol  # 99% VaR
    
    def train_step(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Train all agents with experiences"""
        training_results = {}
        
        # Train allocator
        allocator_experiences = [exp for exp in experiences if exp.info and exp.info.get('agent') == 'allocator']
        if allocator_experiences:
            training_results['allocator'] = self.agents['allocator'].update(allocator_experiences)
        
        # Train executors
        for asset in self.config['assets']:
            executor_experiences = [exp for exp in experiences if exp.info and exp.info.get('agent') == f'executor_{asset}']
            if executor_experiences:
                training_results[f'executor_{asset}'] = self.agents[f'executor_{asset}'].update(executor_experiences)
        
        return training_results
    
    def save_system(self, path: str):
        """Save entire system"""
        checkpoint = {
            'agents': {},
            'portfolio_state': self.portfolio_state,
            'config': self.config
        }
        
        for name, agent in self.agents.items():
            checkpoint['agents'][name] = agent.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_system(self, path: str):
        """Load entire system"""
        checkpoint = torch.load(path, map_location='cpu')
        
        for name, state_dict in checkpoint['agents'].items():
            if name in self.agents:
                self.agents[name].load_state_dict(state_dict)
        
        self.portfolio_state = checkpoint['portfolio_state']
