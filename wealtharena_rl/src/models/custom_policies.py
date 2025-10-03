"""
Custom Policies for WealthArena Trading System

This module contains custom policy implementations for the WealthArena trading system,
including specialized trading policies and multi-agent coordination policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID, TensorType

logger = logging.getLogger(__name__)


class TradingPolicy(Policy):
    """
    Custom trading policy for WealthArena
    
    A specialized policy that incorporates trading-specific logic and constraints
    for the WealthArena trading system.
    """
    
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        
        # Trading-specific configuration
        self.trading_config = config.get("trading_config", {})
        self.risk_tolerance = self.trading_config.get("risk_tolerance", 0.2)
        self.max_position_size = self.trading_config.get("max_position_size", 0.3)
        self.trading_frequency = self.trading_config.get("trading_frequency", 0.2)
        
        # Policy state
        self.last_action = None
        self.action_history = []
        self.portfolio_state = None
        
        logger.info(f"TradingPolicy initialized with risk_tolerance={self.risk_tolerance}")
    
    def compute_actions(self, 
                       obs_batch: List[TensorType],
                       state_batches: List[TensorType],
                       prev_action_batch: List[TensorType] = None,
                       prev_reward_batch: List[TensorType] = None,
                       info_batch: List[Dict[str, Any]] = None,
                       episodes: List[Any] = None,
                       explore: bool = None,
                       timestep: Optional[int] = None,
                       **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Compute actions with trading-specific logic"""
        
        # Get base actions from the model
        actions, state_out, info = super().compute_actions(
            obs_batch, state_batches, prev_action_batch, prev_reward_batch,
            info_batch, episodes, explore, timestep, **kwargs
        )
        
        # Apply trading constraints
        if self.trading_config.get("apply_constraints", True):
            actions = self._apply_trading_constraints(actions, obs_batch)
        
        # Apply risk management
        if self.trading_config.get("apply_risk_management", True):
            actions = self._apply_risk_management(actions, obs_batch)
        
        # Apply trading frequency filter
        if self.trading_config.get("apply_frequency_filter", True):
            actions = self._apply_frequency_filter(actions, timestep)
        
        # Update policy state
        self._update_policy_state(actions, obs_batch)
        
        return actions, state_out, info
    
    def _apply_trading_constraints(self, actions: TensorType, obs_batch: List[TensorType]) -> TensorType:
        """Apply trading-specific constraints to actions"""
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        
        # Clamp actions to valid range
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Apply position size constraints
        if self.max_position_size < 1.0:
            actions = actions * self.max_position_size
        
        # Apply minimum trade threshold
        min_trade_threshold = self.trading_config.get("min_trade_threshold", 0.01)
        actions = torch.where(
            torch.abs(actions) < min_trade_threshold,
            torch.zeros_like(actions),
            actions
        )
        
        return actions
    
    def _apply_risk_management(self, actions: TensorType, obs_batch: List[TensorType]) -> TensorType:
        """Apply risk management to actions"""
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        
        # Extract portfolio information from observations
        for i, obs in enumerate(obs_batch):
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = obs
            
            # Assume portfolio value is in the last few elements of observation
            # This is a simplified assumption - in practice, you'd need to know the exact structure
            if len(obs_np) > 10:
                portfolio_value = obs_np[-3]  # Assuming portfolio value is at index -3
                
                # Apply risk-based scaling
                if portfolio_value < 0.8:  # If portfolio value is below 80% of initial
                    actions[i] = actions[i] * 0.5  # Reduce action magnitude
                elif portfolio_value > 1.2:  # If portfolio value is above 120% of initial
                    actions[i] = actions[i] * 1.2  # Increase action magnitude
        
        return actions
    
    def _apply_frequency_filter(self, actions: TensorType, timestep: Optional[int]) -> TensorType:
        """Apply trading frequency filter"""
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        
        # Randomly filter actions based on trading frequency
        if timestep is not None and timestep % 10 == 0:  # Check every 10 steps
            filter_mask = torch.rand(actions.shape[0]) > self.trading_frequency
            actions = torch.where(
                filter_mask.unsqueeze(-1),
                torch.zeros_like(actions),
                actions
            )
        
        return actions
    
    def _update_policy_state(self, actions: TensorType, obs_batch: List[TensorType]):
        """Update policy state with current actions and observations"""
        
        # Store action history
        if isinstance(actions, torch.Tensor):
            self.action_history.append(actions.cpu().numpy())
        else:
            self.action_history.append(actions)
        
        # Keep only recent history
        max_history = self.trading_config.get("max_action_history", 100)
        if len(self.action_history) > max_history:
            self.action_history = self.action_history[-max_history:]
        
        # Update last action
        self.last_action = actions
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get trading statistics from policy state"""
        
        if not self.action_history:
            return {}
        
        actions_array = np.array(self.action_history)
        
        # Calculate statistics
        stats = {
            "total_actions": len(self.action_history),
            "mean_action_magnitude": np.mean(np.abs(actions_array)),
            "action_volatility": np.std(actions_array),
            "positive_action_ratio": np.mean(actions_array > 0),
            "negative_action_ratio": np.mean(actions_array < 0),
            "zero_action_ratio": np.mean(actions_array == 0)
        }
        
        return stats


class MultiAgentTradingPolicy(TradingPolicy):
    """
    Multi-agent trading policy for WealthArena
    
    A specialized policy that handles coordination between multiple trading agents
    and implements multi-agent specific logic.
    """
    
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        
        # Multi-agent configuration
        self.multiagent_config = config.get("multiagent_config", {})
        self.agent_id = self.multiagent_config.get("agent_id", "unknown")
        self.coordination_enabled = self.multiagent_config.get("coordination_enabled", True)
        self.communication_range = self.multiagent_config.get("communication_range", 0.1)
        
        # Coordination state
        self.other_agents_actions = {}
        self.coordination_reward = 0.0
        
        logger.info(f"MultiAgentTradingPolicy initialized for agent {self.agent_id}")
    
    def compute_actions(self, 
                       obs_batch: List[TensorType],
                       state_batches: List[TensorType],
                       prev_action_batch: List[TensorType] = None,
                       prev_reward_batch: List[TensorType] = None,
                       info_batch: List[Dict[str, Any]] = None,
                       episodes: List[Any] = None,
                       explore: bool = None,
                       timestep: Optional[int] = None,
                       **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Compute actions with multi-agent coordination"""
        
        # Get base actions
        actions, state_out, info = super().compute_actions(
            obs_batch, state_batches, prev_action_batch, prev_reward_batch,
            info_batch, episodes, explore, timestep, **kwargs
        )
        
        # Apply multi-agent coordination
        if self.coordination_enabled:
            actions = self._apply_coordination(actions, obs_batch, timestep)
        
        # Update coordination state
        self._update_coordination_state(actions, obs_batch)
        
        return actions, state_out, info
    
    def _apply_coordination(self, actions: TensorType, obs_batch: List[TensorType], timestep: Optional[int]) -> TensorType:
        """Apply multi-agent coordination to actions"""
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        
        # Get other agents' actions from observation or communication
        other_actions = self._get_other_agents_actions(obs_batch)
        
        if other_actions is not None:
            # Calculate coordination adjustment
            coordination_adjustment = self._calculate_coordination_adjustment(actions, other_actions)
            
            # Apply coordination adjustment
            actions = actions + coordination_adjustment * self.communication_range
        
        return actions
    
    def _get_other_agents_actions(self, obs_batch: List[TensorType]) -> Optional[TensorType]:
        """Get other agents' actions from observation or communication"""
        
        # This is a simplified implementation
        # In practice, you'd need to implement proper communication between agents
        
        if not obs_batch:
            return None
        
        # Assume other agents' actions are encoded in the observation
        # This is a placeholder - actual implementation would depend on the observation structure
        obs = obs_batch[0]
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = obs
        
        # Extract other agents' actions (simplified)
        # In practice, you'd need to know the exact structure of the observation
        if len(obs_np) > 20:  # Assuming there's enough space for other agents' actions
            other_actions = obs_np[-10:]  # Last 10 elements as other agents' actions
            return torch.tensor(other_actions)
        
        return None
    
    def _calculate_coordination_adjustment(self, 
                                         own_actions: TensorType, 
                                         other_actions: TensorType) -> TensorType:
        """Calculate coordination adjustment based on other agents' actions"""
        
        if other_actions is None:
            return torch.zeros_like(own_actions)
        
        # Calculate difference between own actions and other agents' actions
        action_diff = other_actions - own_actions
        
        # Apply coordination logic (simplified)
        # In practice, you'd implement more sophisticated coordination strategies
        coordination_adjustment = action_diff * 0.1  # Small adjustment factor
        
        return coordination_adjustment
    
    def _update_coordination_state(self, actions: TensorType, obs_batch: List[TensorType]):
        """Update coordination state with current actions"""
        
        # Store own actions for other agents to use
        if isinstance(actions, torch.Tensor):
            self.other_agents_actions[self.agent_id] = actions.cpu().numpy()
        else:
            self.other_agents_actions[self.agent_id] = actions
        
        # Calculate coordination reward
        self.coordination_reward = self._calculate_coordination_reward(actions)
    
    def _calculate_coordination_reward(self, actions: TensorType) -> float:
        """Calculate coordination reward for the current actions"""
        
        if not self.other_agents_actions:
            return 0.0
        
        # Calculate coordination reward based on action similarity
        # This is a simplified implementation
        other_actions = list(self.other_agents_actions.values())
        if not other_actions:
            return 0.0
        
        # Calculate average action similarity
        similarities = []
        for other_action in other_actions:
            if isinstance(actions, torch.Tensor):
                own_action_np = actions.cpu().numpy()
            else:
                own_action_np = actions
            
            # Calculate cosine similarity
            similarity = np.dot(own_action_np, other_action) / (
                np.linalg.norm(own_action_np) * np.linalg.norm(other_action) + 1e-8
            )
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        
        stats = {
            "coordination_reward": self.coordination_reward,
            "num_other_agents": len(self.other_agents_actions),
            "coordination_enabled": self.coordination_enabled
        }
        
        # Add trading statistics from parent class
        trading_stats = self.get_trading_statistics()
        stats.update(trading_stats)
        
        return stats


class RiskAwareTradingPolicy(TradingPolicy):
    """
    Risk-aware trading policy for WealthArena
    
    A specialized policy that incorporates advanced risk management techniques
    and dynamic risk adjustment based on market conditions.
    """
    
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        
        # Risk management configuration
        self.risk_config = config.get("risk_config", {})
        self.var_threshold = self.risk_config.get("var_threshold", 0.05)  # 5% VaR
        self.max_drawdown_threshold = self.risk_config.get("max_drawdown_threshold", 0.15)  # 15% max drawdown
        self.volatility_threshold = self.risk_config.get("volatility_threshold", 0.3)  # 30% volatility
        
        # Risk state
        self.portfolio_volatility = 0.0
        self.current_drawdown = 0.0
        self.risk_level = 1.0  # 1.0 = normal, 0.5 = reduced, 0.0 = no trading
        
        logger.info(f"RiskAwareTradingPolicy initialized with VaR threshold={self.var_threshold}")
    
    def compute_actions(self, 
                       obs_batch: List[TensorType],
                       state_batches: List[TensorType],
                       prev_action_batch: List[TensorType] = None,
                       prev_reward_batch: List[TensorType] = None,
                       info_batch: List[Dict[str, Any]] = None,
                       episodes: List[Any] = None,
                       explore: bool = None,
                       timestep: Optional[int] = None,
                       **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Compute actions with risk-aware logic"""
        
        # Update risk state
        self._update_risk_state(obs_batch, prev_reward_batch)
        
        # Get base actions
        actions, state_out, info = super().compute_actions(
            obs_batch, state_batches, prev_action_batch, prev_reward_batch,
            info_batch, episodes, explore, timestep, **kwargs
        )
        
        # Apply risk-aware adjustments
        actions = self._apply_risk_aware_adjustments(actions, obs_batch)
        
        return actions, state_out, info
    
    def _update_risk_state(self, obs_batch: List[TensorType], prev_reward_batch: List[TensorType]):
        """Update risk state based on current observations and rewards"""
        
        # Calculate portfolio volatility from recent rewards
        if prev_reward_batch and len(prev_reward_batch) > 10:
            rewards = np.array(prev_reward_batch[-20:])  # Last 20 rewards
            self.portfolio_volatility = np.std(rewards) * np.sqrt(252)  # Annualized
        
        # Calculate current drawdown from observations
        if obs_batch:
            obs = obs_batch[0]
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = obs
            
            # Assume portfolio value is in the observation
            if len(obs_np) > 10:
                portfolio_value = obs_np[-3]  # Assuming portfolio value is at index -3
                self.current_drawdown = max(0, 1.0 - portfolio_value)  # Simplified drawdown calculation
        
        # Update risk level based on current conditions
        self._update_risk_level()
    
    def _update_risk_level(self):
        """Update risk level based on current risk metrics"""
        
        # Start with normal risk level
        risk_level = 1.0
        
        # Reduce risk if volatility is too high
        if self.portfolio_volatility > self.volatility_threshold:
            risk_level *= 0.5
        
        # Reduce risk if drawdown is too high
        if self.current_drawdown > self.max_drawdown_threshold:
            risk_level *= 0.3
        
        # Further reduce risk if both conditions are met
        if (self.portfolio_volatility > self.volatility_threshold and 
            self.current_drawdown > self.max_drawdown_threshold):
            risk_level *= 0.1
        
        # Update risk level
        self.risk_level = max(0.0, min(1.0, risk_level))
    
    def _apply_risk_aware_adjustments(self, actions: TensorType, obs_batch: List[TensorType]) -> TensorType:
        """Apply risk-aware adjustments to actions"""
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        
        # Scale actions based on risk level
        actions = actions * self.risk_level
        
        # Apply additional risk constraints
        if self.risk_level < 0.5:
            # If risk level is low, further reduce action magnitude
            actions = actions * 0.5
        
        # Apply volatility-based scaling
        if self.portfolio_volatility > self.volatility_threshold:
            # Reduce actions when volatility is high
            volatility_scale = 1.0 - (self.portfolio_volatility - self.volatility_threshold)
            actions = actions * max(0.1, volatility_scale)
        
        return actions
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get risk statistics"""
        
        stats = {
            "portfolio_volatility": self.portfolio_volatility,
            "current_drawdown": self.current_drawdown,
            "risk_level": self.risk_level,
            "var_threshold": self.var_threshold,
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "volatility_threshold": self.volatility_threshold
        }
        
        # Add trading statistics from parent class
        trading_stats = self.get_trading_statistics()
        stats.update(trading_stats)
        
        return stats


# Policy registration functions
def register_trading_policies():
    """Register custom policies with RLlib"""
    
    try:
        from ray.rllib.policy.policy import Policy
        
        # Register trading policy
        Policy.register("trading_policy", TradingPolicy)
        
        # Register multi-agent trading policy
        Policy.register("multi_agent_trading_policy", MultiAgentTradingPolicy)
        
        # Register risk-aware trading policy
        Policy.register("risk_aware_trading_policy", RiskAwareTradingPolicy)
        
        logger.info("Custom trading policies registered with RLlib")
        
    except Exception as e:
        logger.error(f"Error registering custom policies: {e}")


if __name__ == "__main__":
    # Test the policies
    import torch
    
    # Create dummy spaces
    obs_space = type('obs_space', (), {'shape': (100,)})
    action_space = type('action_space', (), {'shape': (10,)})
    
    # Test trading policy
    trading_policy = TradingPolicy(
        obs_space, action_space,
        {"trading_config": {"risk_tolerance": 0.2, "max_position_size": 0.3}}
    )
    
    # Test multi-agent trading policy
    multiagent_policy = MultiAgentTradingPolicy(
        obs_space, action_space,
        {
            "trading_config": {"risk_tolerance": 0.2},
            "multiagent_config": {"agent_id": "trader_0", "coordination_enabled": True}
        }
    )
    
    # Test risk-aware trading policy
    risk_aware_policy = RiskAwareTradingPolicy(
        obs_space, action_space,
        {
            "trading_config": {"risk_tolerance": 0.2},
            "risk_config": {"var_threshold": 0.05, "max_drawdown_threshold": 0.15}
        }
    )
    
    print("All policies tested successfully!")
    print(f"Trading policy stats: {trading_policy.get_trading_statistics()}")
    print(f"Multi-agent policy stats: {multiagent_policy.get_coordination_statistics()}")
    print(f"Risk-aware policy stats: {risk_aware_policy.get_risk_statistics()}")
