"""
WealthArena Trading Environment - Production Ready

Advanced trading environment optimized for profit generation and risk management.
Implements sophisticated reward functions and market dynamics.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import custom modules
from src.data.market_data import MarketDataProcessor
from src.data.data_adapter import DataAdapter
from src.models.portfolio_manager import Portfolio, RiskMetrics

logger = logging.getLogger(__name__)


class WealthArenaTradingEnv(gym.Env):
    """
    Advanced Trading Environment for WealthArena
    
    Features:
    - Real market data integration
    - Advanced reward functions
    - Risk management
    - Portfolio optimization
    - Market dynamics simulation
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config: Dict[str, Any] = None, render_mode: str = None):
        super().__init__()
        
        self.config = config or {}
        self.render_mode = render_mode
        
        # Environment parameters
        self.num_assets = self.config.get("num_assets", 20)
        self.initial_cash = self.config.get("initial_cash", 1_000_000.0)
        self.episode_length = self.config.get("episode_length", 252)
        self.lookback_window_size = self.config.get("lookback_window_size", 30)
        self.transaction_cost_rate = self.config.get("transaction_cost_rate", 0.0005)
        self.slippage_rate = self.config.get("slippage_rate", 0.0002)
        
        # Advanced reward weights
        self.reward_weights = self.config.get("reward_weights", {
            "profit": 2.0,
            "risk": 0.5,
            "cost": 0.1,
            "stability": 0.05,
            "sharpe": 1.0,
            "momentum": 0.3,
            "diversification": 0.2
        })
        
        # Risk management
        self.risk_config = self.config.get("risk_management", {
            "max_position_size": 0.15,
            "max_portfolio_risk": 0.12,
            "stop_loss_threshold": 0.08,
            "take_profit_threshold": 0.20,
            "max_drawdown_limit": 0.15,
            "var_confidence": 0.95,
            "correlation_limit": 0.7
        })
        
        # Data components
        self.data_adapter = DataAdapter(self.config.get("data_adapter_config", {}))
        self.market_data_processor = MarketDataProcessor(self.config.get("market_data_processor_config", {}))
        
        # Portfolio management
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission_rate=self.transaction_cost_rate
        )
        
        # Internal state
        self.current_step = 0
        self.market_data_buffer = []
        self.price_history = []
        self.portfolio_value_history = []
        self.trade_history = []
        self.performance_metrics = {}
        
        # Market data
        self.market_data = None
        self.symbols = self.config.get("symbols", [f"ASSET_{i}" for i in range(self.num_assets)])
        
        # Define spaces
        self._setup_spaces()
        
        # Load market data
        self._load_market_data()
        
        # Reset environment
        self.reset()
        
        logger.info(f"WealthArenaTradingEnv initialized: {self.num_assets} assets, ${self.initial_cash:,.0f} initial cash")
    
    def _setup_spaces(self):
        """Define observation and action spaces"""
        # Action Space: Portfolio weight allocation
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        # Observation Space: Comprehensive market and portfolio state
        # Market features: OHLCV + technical indicators
        market_features = (5 + 20) * self.num_assets  # OHLCV + 20 technical indicators
        
        # Portfolio features
        portfolio_features = self.num_assets + 3  # positions, cash, total_value, leverage
        
        # Risk features
        risk_features = 10  # volatility, drawdown, sharpe, var, etc.
        
        # Market state features
        market_state_features = 5  # market regime, volatility regime, etc.
        
        # Time features
        time_features = 2  # current_step, time_to_end
        
        obs_dim = market_features + portfolio_features + risk_features + market_state_features + time_features
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _load_market_data(self):
        """Load market data from files or API"""
        if self.config.get("use_real_data", True):
            self._load_real_market_data()
        else:
            self._generate_synthetic_data()
    
    def _load_real_market_data(self):
        """Load real market data"""
        data_path = Path("data/processed")
        
        if not data_path.exists():
            logger.warning("No processed data found, generating synthetic data")
            self._generate_synthetic_data()
            return
        
        all_data = []
        for symbol in self.symbols:
            file_path = data_path / f"{symbol}_processed.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                all_data.append(df)
            else:
                logger.warning(f"No data found for {symbol}")
        
        if all_data:
            # Align all data by date
            self.market_data = pd.concat(all_data, keys=self.symbols, axis=1)
            logger.info(f"Loaded real market data: {len(self.market_data)} records")
        else:
            logger.warning("No real data available, generating synthetic data")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate sophisticated synthetic market data"""
        logger.info("Generating synthetic market data...")
        
        np.random.seed(42)
        n_days = self.episode_length + self.lookback_window_size
        
        # Generate correlated returns
        correlation_matrix = self._generate_correlation_matrix()
        returns = np.random.multivariate_normal(
            mean=np.full(self.num_assets, 0.0008),  # 0.08% daily return
            cov=correlation_matrix * 0.0004,  # 2% daily volatility
            size=n_days
        )
        
        # Add market regimes
        returns = self._add_market_regimes(returns)
        
        # Generate OHLCV data
        self.market_data = self._generate_ohlcv_data(returns)
        
        logger.info(f"Generated synthetic data: {len(self.market_data)} records")
    
    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate realistic correlation matrix"""
        # Create base correlation structure
        base_corr = 0.3
        correlation_matrix = np.full((self.num_assets, self.num_assets), base_corr)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Add sector-like correlations
        sector_size = self.num_assets // 4
        for i in range(0, self.num_assets, sector_size):
            end_idx = min(i + sector_size, self.num_assets)
            sector_corr = 0.6
            correlation_matrix[i:end_idx, i:end_idx] = sector_corr
            np.fill_diagonal(correlation_matrix[i:end_idx, i:end_idx], 1.0)
        
        return correlation_matrix
    
    def _add_market_regimes(self, returns: np.ndarray) -> np.ndarray:
        """Add market regime changes"""
        n_days = len(returns)
        
        # Define regime periods
        regime_periods = [
            (0, n_days // 3, "bull"),      # Bull market
            (n_days // 3, 2 * n_days // 3, "bear"),  # Bear market
            (2 * n_days // 3, n_days, "volatile")  # Volatile market
        ]
        
        for start, end, regime in regime_periods:
            if regime == "bull":
                returns[start:end] *= 1.5  # Higher returns
            elif regime == "bear":
                returns[start:end] *= -0.8  # Negative returns
            elif regime == "volatile":
                returns[start:end] *= 2.0  # Higher volatility
        
        return returns
    
    def _generate_ohlcv_data(self, returns: np.ndarray) -> pd.DataFrame:
        """Generate OHLCV data from returns"""
        n_days, n_assets = returns.shape
        
        # Initialize price arrays
        prices = np.zeros((n_days, n_assets))
        prices[0] = 100.0  # Starting price
        
        # Generate price series
        for i in range(1, n_days):
            prices[i] = prices[i-1] * (1 + returns[i])
        
        # Generate OHLCV data
        ohlcv_data = {}
        
        for i, symbol in enumerate(self.symbols):
            asset_prices = prices[:, i]
            
            # Generate OHLCV
            open_prices = asset_prices * (1 + np.random.normal(0, 0.001, n_days))
            close_prices = asset_prices
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
            volumes = np.random.lognormal(8, 0.5, n_days)  # Realistic volume distribution
            
            ohlcv_data[symbol] = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes
            })
        
        # Create proper multi-level DataFrame structure
        multi_level_data = {}
        for symbol, df in ohlcv_data.items():
            for col in df.columns:
                multi_level_data[(symbol, col)] = df[col]
        
        return pd.DataFrame(multi_level_data)
    
    def reset(self, *, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio.reset()
        self.market_data_buffer = []
        self.price_history = []
        self.portfolio_value_history = [self.initial_cash]
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
        prev_portfolio_value = self.portfolio.get_portfolio_value(self._get_current_prices())
        prev_positions = self.portfolio.positions.copy()
        
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
        current_prices = self._get_current_prices()
        self.portfolio.update_performance({symbol: price for symbol, price in zip(self.symbols, current_prices)})
        
        # Check termination conditions
        current_value = self.portfolio.get_portfolio_value(current_prices)
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
            
            # Execute trade
            success = self.portfolio.execute_trade(symbol, action, price)
            
            if success:
                trades_executed.append({
                    "symbol": symbol,
                    "action": action,
                    "price": price,
                    "step": self.current_step
                })
        
        return trades_executed
    
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
        
        # Get the data for this step and reshape it properly
        step_data = self.market_data.iloc[step]
        
        # If it's a Series, convert to DataFrame with proper structure
        if isinstance(step_data, pd.Series):
            # Reshape the multi-level columns into a proper DataFrame
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
        """Calculate sophisticated reward function"""
        current_prices = self._get_current_prices()
        current_value = self.portfolio.get_portfolio_value({symbol: price for symbol, price in zip(self.symbols, current_prices)})
        
        # 1. Profit component (scaled)
        profit_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
        profit_reward = self.reward_weights["profit"] * profit_return * 100  # Scale to percentage
        
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
        volatility_penalty = max(0, volatility - self.risk_config["max_portfolio_risk"]) * 100
        drawdown_penalty = max(0, abs(max_drawdown) - self.risk_config["max_drawdown_limit"]) * 100
        
        return -self.reward_weights["risk"] * (volatility_penalty + drawdown_penalty)
    
    def _calculate_cost_reward(self, trades: List) -> float:
        """Calculate transaction cost penalty"""
        total_cost = 0.0
        for trade in trades:
            trade_value = abs(trade["action"]) * self.portfolio.get_portfolio_value(self._get_current_prices())
            total_cost += trade_value * (self.transaction_cost_rate + self.slippage_rate)
        
        return -self.reward_weights["cost"] * total_cost / self.initial_cash * 100
    
    def _calculate_stability_reward(self, prev_positions: Dict, trades: List) -> float:
        """Calculate stability reward"""
        # Trade frequency penalty
        trade_frequency_penalty = len(trades) * 0.1
        
        # Position change penalty
        current_positions = self.portfolio.positions
        position_change = 0.0
        for symbol in self.symbols:
            prev_pos = prev_positions.get(symbol, 0)
            curr_pos = current_positions.get(symbol, 0)
            position_change += abs(curr_pos - prev_pos)
        
        position_change_penalty = position_change * 0.01
        
        return -self.reward_weights["stability"] * (trade_frequency_penalty + position_change_penalty)
    
    def _calculate_sharpe_reward(self) -> float:
        """Calculate Sharpe ratio reward"""
        if len(self.portfolio_value_history) < 20:
            return 0.0
        
        returns = np.diff(self.portfolio_value_history[-20:]) / self.portfolio_value_history[-20:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return self.reward_weights["sharpe"] * sharpe_ratio * 10  # Scale for visibility
    
    def _calculate_momentum_reward(self) -> float:
        """Calculate momentum reward"""
        if len(self.portfolio_value_history) < 10:
            return 0.0
        
        # Portfolio momentum
        recent_returns = np.diff(self.portfolio_value_history[-10:]) / self.portfolio_value_history[-10:-1]
        momentum = np.mean(recent_returns)
        
        # Market momentum (simplified)
        if len(self.price_history) >= 10:
            market_returns = np.diff(self.price_history[-10:]) / self.price_history[-10:-1]
            market_momentum = np.mean(market_returns)
            
            # Reward for positive momentum alignment
            momentum_alignment = momentum * market_momentum
        else:
            momentum_alignment = momentum
        
        return self.reward_weights["momentum"] * momentum_alignment * 100
    
    def _calculate_diversification_reward(self) -> float:
        """Calculate diversification reward"""
        current_prices = self._get_current_prices()
        portfolio_value = self.portfolio.get_portfolio_value({symbol: price for symbol, price in zip(self.symbols, current_prices)})
        
        if portfolio_value <= 0:
            return 0.0
        
        # Calculate position weights
        weights = []
        for symbol in self.symbols:
            position_value = self.portfolio.positions.get(symbol, 0) * current_prices[self.symbols.index(symbol)]
            weight = position_value / portfolio_value
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Diversification metric (inverse of concentration)
        concentration = np.sum(weights ** 2)
        diversification = 1 - concentration
        
        return self.reward_weights["diversification"] * diversification * 10
    
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
    
    def _get_market_features(self) -> np.ndarray:
        """Get market data features"""
        if not self.market_data_buffer:
            return np.zeros((5 + 20) * self.num_assets)
        
        # Get current market data
        current_data = self.market_data_buffer[-1]
        
        features = []
        for symbol in self.symbols:
            try:
                # Check if symbol exists in the data structure
                if hasattr(current_data, 'columns') and symbol in current_data.columns:
                    # OHLCV - handle both DataFrame and Series cases
                    if isinstance(current_data[symbol], pd.Series):
                        ohlcv = current_data[symbol][['Open', 'High', 'Low', 'Close', 'Volume']].values
                    else:
                        ohlcv = current_data[symbol][['Open', 'High', 'Low', 'Close', 'Volume']].values
                    features.extend(ohlcv)
                    
                    # Technical indicators (simplified)
                    close_price = current_data[symbol]['Close']
                    features.extend([
                        close_price,  # Price
                        close_price,  # SMA (simplified)
                        close_price,  # EMA (simplified)
                        50.0,  # RSI (neutral)
                        0.0,  # MACD
                        0.0,  # MACD Signal
                        0.0,  # MACD Histogram
                        close_price * 1.02,  # BB Upper
                        close_price,  # BB Middle
                        close_price * 0.98,  # BB Lower
                        close_price * 0.02,  # ATR
                        0.0,  # OBV
                        0.0,  # Stochastic K
                        0.0,  # Stochastic D
                        0.0,  # Williams %R
                        0.0,  # CCI
                        0.0,  # ADX
                        0.0,  # Plus DI
                        0.0,  # Minus DI
                        0.0,  # Aroon Up
                        0.0   # Aroon Down
                    ])
                else:
                    # Use default values for synthetic data
                    base_price = 100.0
                    features.extend([
                        base_price,  # Open
                        base_price * 1.01,  # High
                        base_price * 0.99,  # Low
                        base_price,  # Close
                        1000.0,  # Volume
                        base_price,  # Price
                        base_price,  # SMA (simplified)
                        base_price,  # EMA (simplified)
                        50.0,  # RSI (neutral)
                        0.0,  # MACD
                        0.0,  # MACD Signal
                        0.0,  # MACD Histogram
                        base_price * 1.02,  # BB Upper
                        base_price,  # BB Middle
                        base_price * 0.98,  # BB Lower
                        base_price * 0.02,  # ATR
                        0.0,  # OBV
                        0.0,  # Stochastic K
                        0.0,  # Stochastic D
                        0.0,  # Williams %R
                        0.0,  # CCI
                        0.0,  # ADX
                        0.0,  # Plus DI
                        0.0,  # Minus DI
                        0.0,  # Aroon Up
                        0.0   # Aroon Down
                    ])
            except Exception as e:
                # Fill with zeros if symbol not found
                features.extend([0.0] * (5 + 20))
        
        return np.array(features)
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Get portfolio features"""
        current_prices = self._get_current_prices()
        portfolio_value = self.portfolio.get_portfolio_value({symbol: price for symbol, price in zip(self.symbols, current_prices)})
        
        # Position weights
        position_weights = []
        for symbol in self.symbols:
            position_value = self.portfolio.positions.get(symbol, 0) * current_prices[self.symbols.index(symbol)]
            weight = position_value / portfolio_value if portfolio_value > 0 else 0
            position_weights.append(weight)
        
        # Cash ratio
        cash_ratio = self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0
        
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
        if len(self.price_history) < 10:
            return np.zeros(5)
        
        # Market regime (simplified)
        recent_returns = np.diff(self.price_history[-10:]) / self.price_history[-10:-1]
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
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        current_prices = self._get_current_prices()
        portfolio_value = self.portfolio.get_portfolio_value({symbol: price for symbol, price in zip(self.symbols, current_prices)})
        
        return {
            "current_step": self.current_step,
            "portfolio_value": portfolio_value,
            "cash": self.portfolio.cash,
            "positions": self.portfolio.positions.copy(),
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


if __name__ == "__main__":
    # Test the environment
    config = {
        "num_assets": 5,
        "initial_cash": 100000,
        "episode_length": 100,
        "use_real_data": False
    }
    
    env = WealthArenaTradingEnv(config)
    
    # Test environment
    obs, info = env.reset()
    print(f"Environment created: obs shape {obs.shape}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, portfolio_value={info['portfolio_value']:.2f}")
        
        if terminated or truncated:
            break
    
    print("Environment test completed!")