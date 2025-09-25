"""
WealthArena Multi-Agent Trading Environment

A multi-agent trading environment for RLlib that supports multiple trading agents
operating in a shared market environment with coordination mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym

from .trading_env import WealthArenaTradingEnv

logger = logging.getLogger(__name__)


class WealthArenaMultiAgentEnv(MultiAgentEnv):
    """
    WealthArena Multi-Agent Trading Environment
    
    A multi-agent trading environment that supports multiple trading agents with
    different strategies operating in a shared market environment. Each agent
    can have different risk tolerances, trading frequencies, and coordination
    mechanisms.
    
    Args:
        config: Configuration dictionary containing environment parameters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Multi-agent parameters
        self.num_agents = self.config.get("num_agents", 3)
        self.agent_ids = [f"trader_{i}" for i in range(self.num_agents)]
        
        # Agent configurations
        self.agent_configs = self.config.get("agent_configs", {})
        
        # Environment parameters
        self.num_assets = self.config.get("num_assets", 10)
        self.episode_length = self.config.get("episode_length", 1000)
        self.initial_cash = self.config.get("initial_cash", 100000)
        
        # Coordination parameters
        self.coordination_enabled = self.config.get("coordination_enabled", True)
        self.coordination_weight = self.config.get("coordination_weight", 0.1)
        
        # Setup spaces
        self._setup_spaces()
        
        # Initialize market simulator
        self.market_simulator = self._create_market_simulator()
        
        # Initialize agent states
        self.agent_states = {}
        self.agent_portfolios = {}
        self.agent_trade_histories = {}
        
        # Global state
        self.current_step = 0
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        # Reset environment
        self.reset()
        
        logger.info(f"Initialized WealthArenaMultiAgentEnv with {self.num_agents} agents, "
                   f"{self.num_assets} assets, episode_length={self.episode_length}")
    
    def _setup_spaces(self):
        """Setup observation and action spaces for multi-agent environment"""
        
        # Observation space: market data + portfolio state + risk metrics + time
        obs_dim = (
            6 * self.num_assets +  # OHLCV + technical indicators
            self.num_assets + 3 +  # portfolio state
            4 +                    # risk metrics
            1                      # time step
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: continuous trading actions for each asset
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_assets,), dtype=np.float32
        )
    
    def _create_market_simulator(self):
        """Create market simulator for shared market data"""
        
        from .market_simulator import MarketSimulator
        
        market_config = self.config.get("market_config", {})
        market_config.update({
            "num_assets": self.num_assets,
            "episode_length": self.episode_length
        })
        
        return MarketSimulator(market_config)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset environment for new episode"""
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset global state
        self.current_step = 0
        
        # Reset market simulator
        self.market_simulator.reset(seed=seed)
        
        # Initialize agent states
        self.agent_states = {}
        self.agent_portfolios = {}
        self.agent_trade_histories = {}
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        # Initialize each agent
        observations = {}
        infos = {}
        
        for agent_id in self.agent_ids:
            # Get agent configuration
            agent_config = self.agent_configs.get(agent_id, {})
            
            # Initialize agent state
            self.agent_states[agent_id] = {
                "cash": self.initial_cash,
                "positions": np.zeros(self.num_assets),
                "portfolio_history": [self.initial_cash],
                "prev_portfolio_value": self.initial_cash,
                "risk_tolerance": agent_config.get("risk_tolerance", 0.2),
                "trading_frequency": agent_config.get("trading_frequency", 0.2),
                "max_position_size": agent_config.get("max_position_size", 0.3)
            }
            
            # Initialize portfolio
            self.agent_portfolios[agent_id] = {
                "total_value": self.initial_cash,
                "asset_values": np.zeros(self.num_assets),
                "weights": np.zeros(self.num_assets)
            }
            
            # Initialize trade history
            self.agent_trade_histories[agent_id] = []
            
            # Get initial observation
            observations[agent_id] = self._get_observation(agent_id)
            infos[agent_id] = self._get_info(agent_id)
        
        logger.debug(f"Multi-agent environment reset: {self.num_agents} agents initialized")
        
        return observations, infos
    
    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """Execute one step of the multi-agent environment"""
        
        # Update market state
        self.market_simulator.step()
        
        # Process actions for each agent
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        # Calculate coordination reward if enabled
        coordination_reward = 0.0
        if self.coordination_enabled:
            coordination_reward = self._calculate_coordination_reward(action_dict)
        
        for agent_id in self.agent_ids:
            if agent_id in action_dict:
                # Execute trading action
                reward = self._execute_trades(agent_id, action_dict[agent_id])
                
                # Add coordination reward
                if self.coordination_enabled:
                    reward += coordination_reward * self.coordination_weight
                
                rewards[agent_id] = reward
                self.episode_rewards[agent_id].append(reward)
            else:
                rewards[agent_id] = 0.0
            
            # Get new observation
            observations[agent_id] = self._get_observation(agent_id)
            
            # Check termination conditions
            portfolio_value = self._get_portfolio_value(agent_id)
            terminateds[agent_id] = (
                portfolio_value <= 0 or 
                self.current_step >= self.episode_length
            )
            truncateds[agent_id] = False
            
            infos[agent_id] = self._get_info(agent_id)
        
        # Global termination
        terminateds["__all__"] = any(terminateds.values()) or self.current_step >= self.episode_length
        truncateds["__all__"] = False
        
        # Update step counter
        self.current_step += 1
        
        logger.debug(f"Step {self.current_step}: rewards={rewards}, terminated={terminateds['__all__']}")
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _execute_trades(self, agent_id: str, actions: np.ndarray) -> float:
        """Execute trading actions for a specific agent"""
        
        agent_state = self.agent_states[agent_id]
        prev_portfolio_value = self._get_portfolio_value(agent_id)
        
        # Apply trading frequency filter
        trading_frequency = agent_state["trading_frequency"]
        if np.random.random() > trading_frequency:
            # Skip trading this step
            return 0.0
        
        # Apply risk constraints
        actions = self._apply_risk_constraints(agent_id, actions)
        
        # Execute trades
        total_transaction_cost = 0.0
        
        for asset_idx, action in enumerate(actions):
            if abs(action) > 0.01:  # Minimum trade threshold
                if action > 0:  # Buy
                    cost = self._buy_asset(agent_id, asset_idx, action)
                    total_transaction_cost += cost
                else:  # Sell
                    cost = self._sell_asset(agent_id, asset_idx, abs(action))
                    total_transaction_cost += cost
        
        # Calculate reward
        current_portfolio_value = self._get_portfolio_value(agent_id)
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
        
        reward = self._calculate_reward(agent_id, portfolio_return, total_transaction_cost)
        
        # Update portfolio history
        agent_state["portfolio_history"].append(current_portfolio_value)
        agent_state["prev_portfolio_value"] = current_portfolio_value
        
        return reward
    
    def _apply_risk_constraints(self, agent_id: str, actions: np.ndarray) -> np.ndarray:
        """Apply risk constraints to trading actions"""
        
        agent_state = self.agent_states[agent_id]
        max_position_size = agent_state["max_position_size"]
        
        # Limit position sizes
        current_weights = self._get_position_weights(agent_id)
        
        for asset_idx, action in enumerate(actions):
            if action > 0:  # Buy action
                new_weight = current_weights[asset_idx] + action * 0.1  # Scale action
                if new_weight > max_position_size:
                    actions[asset_idx] = max(0, (max_position_size - current_weights[asset_idx]) * 10)
            elif action < 0:  # Sell action
                # Can always sell
                pass
        
        return actions
    
    def _buy_asset(self, agent_id: str, asset_idx: int, amount: float) -> float:
        """Buy asset for specific agent"""
        
        agent_state = self.agent_states[agent_id]
        current_price = self.market_simulator.get_current_price(asset_idx)
        
        if current_price <= 0 or agent_state["cash"] <= 0:
            return 0.0
        
        # Calculate trade amount
        trade_value = agent_state["cash"] * amount * 0.1  # Scale action
        shares = trade_value / current_price
        transaction_cost = trade_value * 0.001  # 0.1% transaction cost
        
        if agent_state["cash"] >= trade_value + transaction_cost:
            agent_state["cash"] -= (trade_value + transaction_cost)
            agent_state["positions"][asset_idx] += shares
            
            # Record trade
            self.agent_trade_histories[agent_id].append({
                "step": self.current_step,
                "asset": asset_idx,
                "action": "buy",
                "shares": shares,
                "price": current_price,
                "cost": transaction_cost
            })
            
            return transaction_cost
        
        return 0.0
    
    def _sell_asset(self, agent_id: str, asset_idx: int, amount: float) -> float:
        """Sell asset for specific agent"""
        
        agent_state = self.agent_states[agent_id]
        current_price = self.market_simulator.get_current_price(asset_idx)
        
        if current_price <= 0 or agent_state["positions"][asset_idx] <= 0:
            return 0.0
        
        # Calculate shares to sell
        shares_to_sell = agent_state["positions"][asset_idx] * amount * 0.1  # Scale action
        trade_value = shares_to_sell * current_price
        transaction_cost = trade_value * 0.001  # 0.1% transaction cost
        
        if shares_to_sell > 0:
            agent_state["cash"] += trade_value - transaction_cost
            agent_state["positions"][asset_idx] -= shares_to_sell
            
            # Record trade
            self.agent_trade_histories[agent_id].append({
                "step": self.current_step,
                "asset": asset_idx,
                "action": "sell",
                "shares": shares_to_sell,
                "price": current_price,
                "cost": transaction_cost
            })
            
            return transaction_cost
        
        return 0.0
    
    def _get_portfolio_value(self, agent_id: str) -> float:
        """Calculate portfolio value for specific agent"""
        
        agent_state = self.agent_states[agent_id]
        current_prices = self.market_simulator.get_current_prices()
        
        asset_value = np.sum(agent_state["positions"] * current_prices)
        total_value = agent_state["cash"] + asset_value
        
        return total_value
    
    def _get_position_weights(self, agent_id: str) -> np.ndarray:
        """Get position weights for specific agent"""
        
        portfolio_value = self._get_portfolio_value(agent_id)
        if portfolio_value <= 0:
            return np.zeros(self.num_assets)
        
        agent_state = self.agent_states[agent_id]
        current_prices = self.market_simulator.get_current_prices()
        
        position_values = agent_state["positions"] * current_prices
        weights = position_values / portfolio_value
        
        return weights
    
    def _calculate_reward(self, agent_id: str, portfolio_return: float, transaction_cost: float) -> float:
        """Calculate reward for specific agent"""
        
        # Base reward: portfolio return
        reward = portfolio_return
        
        # Transaction cost penalty
        cost_penalty = transaction_cost / (self.initial_cash + 1e-8)
        reward -= cost_penalty * 0.1
        
        # Risk penalty
        risk_penalty = self._calculate_risk_penalty(agent_id)
        reward -= risk_penalty * 0.05
        
        return reward
    
    def _calculate_risk_penalty(self, agent_id: str) -> float:
        """Calculate risk penalty for specific agent"""
        
        agent_state = self.agent_states[agent_id]
        
        if len(agent_state["portfolio_history"]) < 5:
            return 0.0
        
        # Calculate portfolio volatility
        returns = np.diff(agent_state["portfolio_history"]) / (np.array(agent_state["portfolio_history"][:-1]) + 1e-8)
        volatility = np.std(returns)
        
        # Calculate concentration risk
        position_weights = self._get_position_weights(agent_id)
        concentration = np.max(position_weights)
        
        # Combined risk penalty
        risk_penalty = volatility * 0.1 + concentration * 0.2
        
        return risk_penalty
    
    def _calculate_coordination_reward(self, action_dict: Dict[str, np.ndarray]) -> float:
        """Calculate coordination reward for multi-agent cooperation"""
        
        if len(action_dict) < 2:
            return 0.0
        
        # Calculate market impact reduction
        total_action = np.sum(list(action_dict.values()), axis=0)
        market_impact = np.linalg.norm(total_action) / len(action_dict)
        impact_bonus = max(0, 0.1 - market_impact) * 0.5
        
        # Calculate diversification bonus
        action_vectors = np.array(list(action_dict.values()))
        if action_vectors.shape[0] > 1:
            # Calculate correlation between agent actions
            correlation_matrix = np.corrcoef(action_vectors)
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            diversification = 1.0 - avg_correlation
        else:
            diversification = 0.0
        
        div_bonus = diversification * 0.3
        
        return impact_bonus + div_bonus
    
    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for specific agent"""
        
        # Market data
        market_obs = self.market_simulator.get_market_observation()
        
        # Agent-specific portfolio state
        portfolio_obs = self._get_portfolio_observation(agent_id)
        
        # Risk metrics
        risk_obs = self._get_risk_observation(agent_id)
        
        # Time features
        time_obs = self._get_time_observation()
        
        return np.concatenate([market_obs, portfolio_obs, risk_obs, time_obs]).astype(np.float32)
    
    def _get_portfolio_observation(self, agent_id: str) -> np.ndarray:
        """Get portfolio observation for specific agent"""
        
        agent_state = self.agent_states[agent_id]
        portfolio_value = self._get_portfolio_value(agent_id)
        
        if portfolio_value > 0:
            # Position weights
            position_weights = self._get_position_weights(agent_id)
            # Cash ratio
            cash_ratio = agent_state["cash"] / portfolio_value
            # Portfolio value ratio
            value_ratio = portfolio_value / self.initial_cash
        else:
            position_weights = np.zeros(self.num_assets)
            cash_ratio = 1.0
            value_ratio = 0.0
        
        return np.concatenate([position_weights, [cash_ratio, value_ratio]])
    
    def _get_risk_observation(self, agent_id: str) -> np.ndarray:
        """Get risk observation for specific agent"""
        
        agent_state = self.agent_states[agent_id]
        
        if len(agent_state["portfolio_history"]) < 5:
            return np.zeros(4)
        
        # Portfolio volatility
        returns = np.diff(agent_state["portfolio_history"]) / (np.array(agent_state["portfolio_history"][:-1]) + 1e-8)
        volatility = np.std(returns)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(agent_state["portfolio_history"])
        drawdown = (peak - np.array(agent_state["portfolio_history"])) / (peak + 1e-8)
        max_drawdown = np.max(drawdown)
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = np.mean(returns) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Position concentration
        position_weights = self._get_position_weights(agent_id)
        concentration = np.max(position_weights)
        
        return np.array([volatility, max_drawdown, sharpe_ratio, concentration])
    
    def _get_time_observation(self) -> np.ndarray:
        """Get time-based observation"""
        
        time_ratio = self.current_step / self.episode_length
        return np.array([time_ratio])
    
    def _get_info(self, agent_id: str) -> Dict[str, Any]:
        """Get info dictionary for specific agent"""
        
        agent_state = self.agent_states[agent_id]
        portfolio_value = self._get_portfolio_value(agent_id)
        
        return {
            "portfolio_value": portfolio_value,
            "cash": agent_state["cash"],
            "positions": agent_state["positions"].copy(),
            "step": self.current_step,
            "num_trades": len(self.agent_trade_histories[agent_id]),
            "episode_reward": sum(self.episode_rewards[agent_id])
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment state"""
        
        if mode == 'human':
            print(f"\n=== Multi-Agent Trading Environment (Step {self.current_step}) ===")
            for agent_id in self.agent_ids:
                portfolio_value = self._get_portfolio_value(agent_id)
                agent_state = self.agent_states[agent_id]
                print(f"{agent_id}: Portfolio={portfolio_value:.2f}, Cash={agent_state['cash']:.2f}, "
                      f"Positions={agent_state['positions']}")
        elif mode == 'rgb_array':
            # Return RGB array for visualization
            return None  # Placeholder
    
    def close(self):
        """Clean up environment resources"""
        
        # Clean up any resources
        pass


# Environment registration
def register_multi_agent_env():
    """Register multi-agent environment with Gymnasium"""
    
    gym.register(
        id='WealthArenaMultiAgent-v0',
        entry_point='wealtharena_rllib.src.environments.multi_agent_env:WealthArenaMultiAgentEnv',
        max_episode_steps=1000,
    )


if __name__ == "__main__":
    # Test the multi-agent environment
    env = WealthArenaMultiAgentEnv({
        "num_agents": 3,
        "num_assets": 5,
        "episode_length": 100,
        "coordination_enabled": True
    })
    
    obs, info = env.reset()
    print(f"Number of agents: {len(obs)}")
    print(f"Observation shape: {obs[env.agent_ids[0]].shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps
    for i in range(5):
        actions = {agent_id: env.action_space.sample() for agent_id in env.agent_ids}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        print(f"Step {i}: rewards={rewards}, terminated={terminateds['__all__']}")
        
        if terminateds["__all__"]:
            break
    
    env.close()
