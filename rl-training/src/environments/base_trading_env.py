"""
Base Trading Environment for WealthArena Trading System

This module provides the base TradingEnv class compliant with OpenAI Gym API
for all financial instrument types in the WealthArena system.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment"""
    # Environment parameters
    num_assets: int = 20
    initial_cash: float = 1_000_000.0
    episode_length: int = 252
    lookback_window_size: int = 30
    
    # Transaction costs
    transaction_cost_rate: float = 0.0005
    slippage_rate: float = 0.0002
    
    # Risk management
    max_position_size: float = 0.15
    max_portfolio_risk: float = 0.12
    stop_loss_threshold: float = 0.08
    take_profit_threshold: float = 0.20
    max_drawdown_limit: float = 0.15
    
    # Reward weights
    reward_weights: Dict[str, float] = None
    
    # Data parameters
    use_real_data: bool = True
    data_path: str = "data/processed"
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                "profit": 2.0,
                "risk": 0.5,
                "cost": 0.1,
                "stability": 0.05,
                "sharpe": 1.0,
                "momentum": 0.3,
                "diversification": 0.2
            }


class BaseTradingEnv(gym.Env, ABC):
    """
    Base Trading Environment for WealthArena
    
    This abstract base class defines the interface for all trading environments
    in the WealthArena system, ensuring compliance with OpenAI Gym API.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config: TradingEnvConfig = None, render_mode: str = None):
        super().__init__()
        
        self.config = config or TradingEnvConfig()
        self.render_mode = render_mode
        
        # Environment state
        self.current_step = 0
        self.episode_length = self.config.episode_length
        self.lookback_window_size = self.config.lookback_window_size
        
        # Portfolio state
        self.initial_cash = self.config.initial_cash
        self.cash = self.initial_cash
        self.positions = {}  # {symbol: quantity}
        self.portfolio_value_history = [self.initial_cash]
        
        # Market data
        self.market_data = None
        self.market_data_buffer = []
        self.symbols = []
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
        # Define spaces
        self._setup_spaces()
        
        # Load market data
        self._load_market_data()
        
        # Initialize environment
        self.reset()
        
        logger.info(f"BaseTradingEnv initialized: {self.config.num_assets} assets, ${self.initial_cash:,.0f} initial cash")
    
    @abstractmethod
    def _setup_spaces(self):
        """Setup observation and action spaces - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _load_market_data(self):
        """Load market data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_market_features(self) -> np.ndarray:
        """Get market data features - must be implemented by subclasses"""
        pass
    
    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = {}
        self.portfolio_value_history = [self.initial_cash]
        self.market_data_buffer = []
        self.trade_history = []
        self.performance_metrics = {}
        
        # Populate market data buffer
        for i in range(self.lookback_window_size):
            if i < len(self.market_data):
                step_data = self._get_market_data_step(i)
                self.market_data_buffer.append(step_data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        # Store previous state
        prev_portfolio_value = self._get_portfolio_value()
        prev_positions = self.positions.copy()
        
        # Execute trades
        trades_executed = self._execute_trades(action)
        
        # Update market data
        self.current_step += 1
        if self.current_step < len(self.market_data):
            current_data = self._get_market_data_step(self.current_step)
            self.market_data_buffer.append(current_data)
            if len(self.market_data_buffer) > self.lookback_window_size:
                self.market_data_buffer.pop(0)
        else:
            # End of data
            done = True
            truncated = False
            observation = self._get_observation()
            reward = self._calculate_reward(prev_portfolio_value, prev_positions, trades_executed)
            info = self._get_info()
            return observation, reward, done, truncated, info
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, prev_positions, trades_executed)
        
        # Update portfolio performance
        self._update_portfolio_performance()
        
        # Check termination conditions
        current_value = self._get_portfolio_value()
        done = (self.current_step >= self.episode_length or 
                current_value <= self.initial_cash * 0.1)  # 90% loss limit
        
        truncated = False
        
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, done, truncated, info
    
    def _execute_trades(self, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Execute trades based on actions"""
        trades_executed = []
        current_prices = self._get_current_prices()
        
        # Normalize actions to ensure they sum to reasonable values
        action_magnitude = np.sum(np.abs(actions))
        if action_magnitude > 1.0:
            actions = actions / action_magnitude
        
        for asset_idx, action in enumerate(actions):
            if abs(action) < 0.01:  # Skip small actions
                continue
            
            symbol = self.symbols[asset_idx]
            price = current_prices[asset_idx]
            
            if price <= 0:
                continue
            
            # Calculate position change
            current_position = self.positions.get(symbol, 0)
            target_position_value = action * self._get_portfolio_value()
            target_position = target_position_value / price if price > 0 else 0
            position_change = target_position - current_position
            
            # Execute trade if significant change
            if abs(position_change) > 0.01:
                success = self._execute_trade(symbol, position_change, price)
                
                if success:
                    trades_executed.append({
                        "symbol": symbol,
                        "action": action,
                        "position_change": position_change,
                        "price": price,
                        "step": self.current_step
                    })
        
        return trades_executed
    
    def _execute_trade(self, symbol: str, position_change: float, price: float) -> bool:
        """Execute a single trade"""
        trade_value = abs(position_change) * price
        transaction_cost = trade_value * (self.config.transaction_cost_rate + self.config.slippage_rate)
        
        # Check if we have enough cash for the trade
        if position_change > 0:  # Buying
            total_cost = trade_value + transaction_cost
            if total_cost > self.cash:
                return False
            self.cash -= total_cost
        else:  # Selling
            self.cash += trade_value - transaction_cost
        
        # Update position
        self.positions[symbol] = self.positions.get(symbol, 0) + position_change
        
        # Remove position if very small
        if abs(self.positions[symbol]) < 0.01:
            del self.positions[symbol]
        
        return True
    
    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all assets"""
        if self.current_step >= len(self.market_data):
            return self._get_market_data_step(-1)["Close"].values
        
        current_data = self._get_market_data_step(self.current_step)
        return current_data["Close"].values
    
    def _get_market_data_step(self, step: int) -> pd.DataFrame:
        """Get market data for a specific step"""
        if step < 0:
            step = len(self.market_data) - 1
        
        if step >= len(self.market_data):
            step = len(self.market_data) - 1
        
        # Get the data for this step
        step_data = self.market_data.iloc[step]
        
        # Convert to DataFrame format
        if isinstance(step_data, pd.Series):
            data_dict = {}
            for symbol in self.symbols:
                symbol_data = {}
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    try:
                        symbol_data[col] = step_data[(symbol, col)]
                    except KeyError:
                        # Fallback for missing data
                        symbol_data[col] = 100.0 if col != 'Volume' else 1000.0
                data_dict[symbol] = symbol_data
            
            return pd.DataFrame(data_dict).T
        else:
            return step_data
    
    def _calculate_reward(self, prev_value: float, prev_positions: Dict, trades: List) -> float:
        """Calculate reward based on multiple objectives"""
        current_value = self._get_portfolio_value()
        
        # 1. Profit component
        profit_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
        profit_reward = self.config.reward_weights["profit"] * profit_return * 100
        
        # 2. Risk component
        risk_reward = self._calculate_risk_reward(current_value)
        
        # 3. Cost component
        cost_reward = self._calculate_cost_reward(trades)
        
        # 4. Stability component
        stability_reward = self._calculate_stability_reward(prev_positions, trades)
        
        # 5. Sharpe ratio component
        sharpe_reward = self._calculate_sharpe_reward()
        
        # 6. Momentum component
        momentum_reward = self._calculate_momentum_reward()
        
        # 7. Diversification component
        diversification_reward = self._calculate_diversification_reward()
        
        # Combine all components
        total_reward = (profit_reward + risk_reward + cost_reward + 
                       stability_reward + sharpe_reward + momentum_reward + 
                       diversification_reward)
        
        # Store for analysis
        self.performance_metrics = {
            "profit_reward": profit_reward,
            "risk_reward": risk_reward,
            "cost_reward": cost_reward,
            "stability_reward": stability_reward,
            "sharpe_reward": sharpe_reward,
            "momentum_reward": momentum_reward,
            "diversification_reward": diversification_reward,
            "total_reward": total_reward
        }
        
        return total_reward
    
    def _calculate_risk_reward(self, current_value: float) -> float:
        """Calculate risk-based reward"""
        if len(self.portfolio_value_history) < 10:
            return 0.0
        
        # Portfolio volatility
        returns = np.diff(self.portfolio_value_history[-20:]) / self.portfolio_value_history[-20:-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        # Drawdown
        peak = np.maximum.accumulate(self.portfolio_value_history)
        drawdown = (np.array(self.portfolio_value_history) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Risk penalties
        volatility_penalty = max(0, volatility - self.config.max_portfolio_risk) * 100
        drawdown_penalty = max(0, abs(max_drawdown) - self.config.max_drawdown_limit) * 100
        
        return -self.config.reward_weights["risk"] * (volatility_penalty + drawdown_penalty)
    
    def _calculate_cost_reward(self, trades: List) -> float:
        """Calculate transaction cost penalty"""
        total_cost = 0.0
        for trade in trades:
            trade_value = abs(trade["position_change"]) * trade["price"]
            total_cost += trade_value * (self.config.transaction_cost_rate + self.config.slippage_rate)
        
        return -self.config.reward_weights["cost"] * total_cost / self.initial_cash * 100
    
    def _calculate_stability_reward(self, prev_positions: Dict, trades: List) -> float:
        """Calculate stability reward"""
        # Trade frequency penalty
        trade_frequency_penalty = len(trades) * 0.1
        
        # Position change penalty
        position_change = 0.0
        for symbol in self.symbols:
            prev_pos = prev_positions.get(symbol, 0)
            curr_pos = self.positions.get(symbol, 0)
            position_change += abs(curr_pos - prev_pos)
        
        position_change_penalty = position_change * 0.01
        
        return -self.config.reward_weights["stability"] * (trade_frequency_penalty + position_change_penalty)
    
    def _calculate_sharpe_reward(self) -> float:
        """Calculate Sharpe ratio reward"""
        if len(self.portfolio_value_history) < 20:
            return 0.0
        
        returns = np.diff(self.portfolio_value_history[-20:]) / self.portfolio_value_history[-20:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return self.config.reward_weights["sharpe"] * sharpe_ratio * 10
    
    def _calculate_momentum_reward(self) -> float:
        """Calculate momentum reward"""
        if len(self.portfolio_value_history) < 10:
            return 0.0
        
        # Portfolio momentum
        recent_returns = np.diff(self.portfolio_value_history[-10:]) / self.portfolio_value_history[-10:-1]
        momentum = np.mean(recent_returns)
        
        return self.config.reward_weights["momentum"] * momentum * 100
    
    def _calculate_diversification_reward(self) -> float:
        """Calculate diversification reward"""
        current_prices = self._get_current_prices()
        portfolio_value = self._get_portfolio_value()
        
        if portfolio_value <= 0:
            return 0.0
        
        # Calculate position weights
        weights = []
        for symbol in self.symbols:
            position_value = self.positions.get(symbol, 0) * current_prices[self.symbols.index(symbol)]
            weight = position_value / portfolio_value
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Diversification metric (inverse of concentration)
        concentration = np.sum(weights ** 2)
        diversification = 1 - concentration
        
        return self.config.reward_weights["diversification"] * diversification * 10
    
    def _get_observation(self) -> np.ndarray:
        """Generate comprehensive observation"""
        # Market data features
        market_features = self._get_market_features()
        
        # Portfolio features
        portfolio_features = self._get_portfolio_features()
        
        # Risk features
        risk_features = self._get_risk_features()
        
        # Market state features
        market_state_features = self._get_market_state_features()
        
        # Time features
        time_features = self._get_time_features()
        
        # Combine all features
        observation = np.concatenate([
            market_features,
            portfolio_features,
            risk_features,
            market_state_features,
            time_features
        ]).astype(np.float32)
        
        return observation
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Get portfolio features"""
        current_prices = self._get_current_prices()
        portfolio_value = self._get_portfolio_value()
        
        # Position weights
        position_weights = []
        for symbol in self.symbols:
            position_value = self.positions.get(symbol, 0) * current_prices[self.symbols.index(symbol)]
            weight = position_value / portfolio_value if portfolio_value > 0 else 0
            position_weights.append(weight)
        
        # Cash ratio
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 1.0
        
        # Total value ratio
        value_ratio = portfolio_value / self.initial_cash
        
        # Leverage (simplified)
        leverage = 1.0  # No leverage for now
        
        return np.array(position_weights + [cash_ratio, value_ratio, leverage])
    
    def _get_risk_features(self) -> np.ndarray:
        """Get risk features"""
        if len(self.portfolio_value_history) < 10:
            return np.zeros(10)
        
        returns = np.diff(self.portfolio_value_history[-20:]) / self.portfolio_value_history[-20:-1]
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(self.portfolio_value_history)
        drawdown = (np.array(self.portfolio_value_history) - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # VaR (simplified)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # CVaR
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0
        
        # Additional risk metrics
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        
        # Correlation with market (simplified)
        market_correlation = 0.5  # Placeholder
        
        # Beta (simplified)
        beta = 1.0  # Placeholder
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = np.mean(returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        return np.array([
            volatility, sharpe, max_drawdown, var_95, cvar_95,
            skewness, kurtosis, market_correlation, beta, sortino
        ])
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness"""
        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis"""
        if len(returns) < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 4) - 3
    
    def _get_market_state_features(self) -> np.ndarray:
        """Get market state features"""
        if len(self.portfolio_value_history) < 10:
            return np.zeros(5)
        
        # Market regime (simplified)
        recent_returns = np.diff(self.portfolio_value_history[-10:]) / self.portfolio_value_history[-10:-1]
        market_return = np.mean(recent_returns)
        market_volatility = np.std(recent_returns)
        
        # Regime classification
        if market_return > 0.01:
            regime = 1.0  # Bull
        elif market_return < -0.01:
            regime = -1.0  # Bear
        else:
            regime = 0.0  # Neutral
        
        # Volatility regime
        if market_volatility > 0.02:
            vol_regime = 1.0  # High volatility
        else:
            vol_regime = 0.0  # Low volatility
        
        # Trend strength
        trend_strength = abs(market_return) / market_volatility if market_volatility > 0 else 0
        
        return np.array([regime, vol_regime, trend_strength, market_return, market_volatility])
    
    def _get_time_features(self) -> np.ndarray:
        """Get time features"""
        progress = self.current_step / self.episode_length
        time_to_end = 1.0 - progress
        
        return np.array([progress, time_to_end])
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        current_prices = self._get_current_prices()
        portfolio_value = self.cash
        
        for symbol in self.symbols:
            if symbol in self.positions:
                price = current_prices[self.symbols.index(symbol)]
                portfolio_value += self.positions[symbol] * price
        
        return portfolio_value
    
    def _update_portfolio_performance(self):
        """Update portfolio performance tracking"""
        current_value = self._get_portfolio_value()
        self.portfolio_value_history.append(current_value)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        current_prices = self._get_current_prices()
        portfolio_value = self._get_portfolio_value()
        
        return {
            "current_step": self.current_step,
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "total_profit": portfolio_value - self.initial_cash,
            "profit_pct": (portfolio_value - self.initial_cash) / self.initial_cash * 100,
            "performance_metrics": self.performance_metrics.copy(),
            "num_trades": len(self.trade_history)
        }
    
    def render(self) -> None:
        """Render environment"""
        if self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render frame (placeholder)"""
        # This would implement actual rendering
        pass
    
    def close(self) -> None:
        """Close environment"""
        pass
