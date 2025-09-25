"""
Market Simulator for WealthArena Trading Environment

A market simulator that generates realistic market data including OHLCV prices,
technical indicators, and market microstructure for the trading environment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketSimulator:
    """
    Market Simulator for WealthArena Trading Environment
    
    Generates realistic market data including OHLCV prices, technical indicators,
    and market microstructure. Supports correlated asset movements and various
    market conditions.
    
    Args:
        config: Configuration dictionary containing market parameters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Market parameters
        self.num_assets = self.config.get("num_assets", 10)
        self.episode_length = self.config.get("episode_length", 1000)
        self.volatility = self.config.get("volatility", 0.02)
        self.correlation = self.config.get("correlation", 0.1)
        self.trend = self.config.get("trend", 0.001)
        self.mean_reversion = self.config.get("mean_reversion", 0.01)
        
        # Market microstructure
        self.bid_ask_spread = self.config.get("bid_ask_spread", 0.001)
        self.volume_volatility = self.config.get("volume_volatility", 0.3)
        
        # Technical indicator parameters
        self.technical_windows = {
            "sma_short": 20,
            "sma_long": 50,
            "rsi_period": 14,
            "bb_period": 20,
            "bb_std": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9
        }
        
        # Initialize market state
        self.reset()
        
        logger.info(f"Initialized MarketSimulator with {self.num_assets} assets, "
                   f"volatility={self.volatility}, correlation={self.correlation}")
    
    def reset(self, seed: Optional[int] = None):
        """Reset market simulator"""
        
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # Initialize prices
        self.prices = np.ones(self.num_assets) * 100.0  # Starting prices
        self.price_history = [self.prices.copy()]
        
        # Initialize technical indicators
        self.technical_indicators = {
            "sma_short": np.ones(self.num_assets) * 100.0,
            "sma_long": np.ones(self.num_assets) * 100.0,
            "rsi": np.ones(self.num_assets) * 50.0,
            "bb_upper": np.ones(self.num_assets) * 102.0,
            "bb_lower": np.ones(self.num_assets) * 98.0,
            "macd": np.zeros(self.num_assets),
            "macd_signal": np.zeros(self.num_assets),
            "macd_histogram": np.zeros(self.num_assets)
        }
        
        # Initialize volume
        self.volume = np.ones(self.num_assets) * 1000.0
        self.volume_history = [self.volume.copy()]
        
        # Generate correlation matrix
        self.correlation_matrix = self._generate_correlation_matrix()
        
        # Initialize market microstructure
        self.bid_prices = self.prices * (1 - self.bid_ask_spread / 2)
        self.ask_prices = self.prices * (1 + self.bid_ask_spread / 2)
        
        logger.debug("Market simulator reset")
    
    def step(self):
        """Update market prices and indicators"""
        
        # Generate correlated returns
        returns = self._generate_correlated_returns()
        
        # Update prices
        self.prices *= (1 + returns)
        self.price_history.append(self.prices.copy())
        
        # Update volume
        self._update_volume()
        
        # Update technical indicators
        self._update_technical_indicators()
        
        # Update market microstructure
        self._update_market_microstructure()
        
        # Update step counter
        self.current_step += 1
        
        logger.debug(f"Market step {self.current_step}: prices={self.prices[:3]}")
    
    def _generate_correlated_returns(self) -> np.ndarray:
        """Generate correlated returns for all assets"""
        
        # Base returns with trend and mean reversion
        base_returns = np.random.normal(
            self.trend, 
            self.volatility, 
            self.num_assets
        )
        
        # Apply mean reversion
        if self.current_step > 0:
            price_ratios = self.prices / 100.0  # Normalize to initial price
            mean_reversion_force = -self.mean_reversion * (price_ratios - 1.0)
            base_returns += mean_reversion_force
        
        # Generate correlated returns
        correlated_returns = np.random.multivariate_normal(
            base_returns,
            self.correlation_matrix * self.volatility**2
        )
        
        return correlated_returns
    
    def _update_volume(self):
        """Update trading volume for all assets"""
        
        # Volume follows a log-normal distribution with some correlation to price changes
        if self.current_step > 0:
            price_change = self.prices / self.price_history[-2] - 1.0
            volume_change = np.random.normal(0, self.volume_volatility, self.num_assets)
            volume_change += 0.1 * np.abs(price_change)  # Higher volume on price changes
            
            self.volume *= np.exp(volume_change)
        else:
            self.volume *= np.exp(np.random.normal(0, self.volume_volatility, self.num_assets))
        
        # Ensure minimum volume
        self.volume = np.maximum(self.volume, 100.0)
        
        self.volume_history.append(self.volume.copy())
    
    def _update_technical_indicators(self):
        """Update technical indicators"""
        
        if self.current_step < 2:
            return
        
        # Simple Moving Averages
        if self.current_step >= self.technical_windows["sma_short"]:
            recent_prices = np.array(self.price_history[-self.technical_windows["sma_short"]:])
            self.technical_indicators["sma_short"] = np.mean(recent_prices, axis=0)
        
        if self.current_step >= self.technical_windows["sma_long"]:
            recent_prices = np.array(self.price_history[-self.technical_windows["sma_long"]:])
            self.technical_indicators["sma_long"] = np.mean(recent_prices, axis=0)
        
        # RSI (Relative Strength Index)
        if self.current_step >= self.technical_windows["rsi_period"]:
            self.technical_indicators["rsi"] = self._calculate_rsi()
        
        # Bollinger Bands
        if self.current_step >= self.technical_windows["bb_period"]:
            bb_upper, bb_lower = self._calculate_bollinger_bands()
            self.technical_indicators["bb_upper"] = bb_upper
            self.technical_indicators["bb_lower"] = bb_lower
        
        # MACD
        if self.current_step >= self.technical_windows["macd_slow"]:
            macd, signal, histogram = self._calculate_macd()
            self.technical_indicators["macd"] = macd
            self.technical_indicators["macd_signal"] = signal
            self.technical_indicators["macd_histogram"] = histogram
    
    def _calculate_rsi(self) -> np.ndarray:
        """Calculate RSI for all assets"""
        
        if self.current_step < self.technical_windows["rsi_period"]:
            return np.ones(self.num_assets) * 50.0
        
        rsi_values = np.zeros(self.num_assets)
        
        for asset_idx in range(self.num_assets):
            prices = [self.price_history[i][asset_idx] for i in range(max(0, self.current_step - self.technical_windows["rsi_period"]), self.current_step + 1)]
            
            if len(prices) < 2:
                rsi_values[asset_idx] = 50.0
                continue
            
            # Calculate price changes
            changes = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            # Calculate RSI
            if avg_loss == 0:
                rsi_values[asset_idx] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[asset_idx] = 100 - (100 / (1 + rs))
        
        return rsi_values
    
    def _calculate_bollinger_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands for all assets"""
        
        if self.current_step < self.technical_windows["bb_period"]:
            return self.prices * 1.02, self.prices * 0.98
        
        bb_upper = np.zeros(self.num_assets)
        bb_lower = np.zeros(self.num_assets)
        
        for asset_idx in range(self.num_assets):
            prices = [self.price_history[i][asset_idx] for i in range(max(0, self.current_step - self.technical_windows["bb_period"]), self.current_step + 1)]
            
            if len(prices) < 2:
                bb_upper[asset_idx] = self.prices[asset_idx] * 1.02
                bb_lower[asset_idx] = self.prices[asset_idx] * 0.98
                continue
            
            # Calculate SMA and standard deviation
            sma = np.mean(prices)
            std = np.std(prices)
            
            # Calculate Bollinger Bands
            bb_upper[asset_idx] = sma + (self.technical_windows["bb_std"] * std)
            bb_lower[asset_idx] = sma - (self.technical_windows["bb_std"] * std)
        
        return bb_upper, bb_lower
    
    def _calculate_macd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD for all assets"""
        
        if self.current_step < self.technical_windows["macd_slow"]:
            return np.zeros(self.num_assets), np.zeros(self.num_assets), np.zeros(self.num_assets)
        
        macd_values = np.zeros(self.num_assets)
        signal_values = np.zeros(self.num_assets)
        histogram_values = np.zeros(self.num_assets)
        
        for asset_idx in range(self.num_assets):
            prices = [self.price_history[i][asset_idx] for i in range(max(0, self.current_step - self.technical_windows["macd_slow"]), self.current_step + 1)]
            
            if len(prices) < self.technical_windows["macd_slow"]:
                continue
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, self.technical_windows["macd_fast"])
            ema_slow = self._calculate_ema(prices, self.technical_windows["macd_slow"])
            
            # Calculate MACD
            macd_values[asset_idx] = ema_fast - ema_slow
            
            # Calculate signal line (EMA of MACD)
            if self.current_step >= self.technical_windows["macd_slow"] + self.technical_windows["macd_signal"]:
                macd_history = [self.technical_indicators["macd"][asset_idx] for _ in range(self.technical_windows["macd_signal"])]
                signal_values[asset_idx] = self._calculate_ema(macd_history, self.technical_windows["macd_signal"])
            
            # Calculate histogram
            histogram_values[asset_idx] = macd_values[asset_idx] - signal_values[asset_idx]
        
        return macd_values, signal_values, histogram_values
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        
        if len(prices) < period:
            return np.mean(prices)
        
        # Use the last 'period' prices
        recent_prices = prices[-period:]
        
        # Calculate EMA
        multiplier = 2.0 / (period + 1)
        ema = recent_prices[0]
        
        for price in recent_prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _update_market_microstructure(self):
        """Update market microstructure (bid-ask spreads)"""
        
        # Update bid-ask spreads based on volatility
        spread_factor = 1.0 + np.random.normal(0, 0.1, self.num_assets)
        spread_factor = np.clip(spread_factor, 0.5, 2.0)
        
        self.bid_prices = self.prices * (1 - self.bid_ask_spread * spread_factor / 2)
        self.ask_prices = self.prices * (1 + self.bid_ask_spread * spread_factor / 2)
    
    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate correlation matrix for assets"""
        
        # Create random correlation matrix
        A = np.random.randn(self.num_assets, self.num_assets)
        correlation_matrix = np.dot(A, A.T)
        
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(d, d)
        
        # Adjust correlation strength
        correlation_matrix = (1 - self.correlation) * np.eye(self.num_assets) + self.correlation * correlation_matrix
        
        return correlation_matrix
    
    def get_current_price(self, asset_idx: int) -> float:
        """Get current price for specific asset"""
        
        if asset_idx >= self.num_assets:
            return 0.0
        
        return self.prices[asset_idx]
    
    def get_current_prices(self) -> np.ndarray:
        """Get current prices for all assets"""
        
        return self.prices.copy()
    
    def get_market_observation(self) -> np.ndarray:
        """Get comprehensive market observation"""
        
        # OHLCV data (simplified - using current price for all)
        ohlcv_data = np.zeros(self.num_assets * 5)
        for i in range(self.num_assets):
            ohlcv_data[i*5:(i+1)*5] = [
                self.prices[i],  # Open
                self.prices[i] * 1.01,  # High (simplified)
                self.prices[i] * 0.99,  # Low (simplified)
                self.prices[i],  # Close
                self.volume[i]   # Volume
            ]
        
        # Technical indicators
        technical_data = np.concatenate([
            self.technical_indicators["sma_short"],
            self.technical_indicators["sma_long"],
            self.technical_indicators["rsi"] / 100.0,  # Normalize RSI
            self.technical_indicators["bb_upper"] / self.prices,  # Normalize BB
            self.technical_indicators["bb_lower"] / self.prices,
            self.technical_indicators["macd"] / self.prices,  # Normalize MACD
            self.technical_indicators["macd_signal"] / self.prices,
            self.technical_indicators["macd_histogram"] / self.prices
        ])
        
        # Market microstructure
        microstructure_data = np.concatenate([
            self.bid_prices / self.prices,  # Normalized bid prices
            self.ask_prices / self.prices   # Normalized ask prices
        ])
        
        return np.concatenate([ohlcv_data, technical_data, microstructure_data]).astype(np.float32)
    
    def get_price_history(self, asset_idx: int, window: int = None) -> np.ndarray:
        """Get price history for specific asset"""
        
        if asset_idx >= self.num_assets:
            return np.array([])
        
        if window is None:
            window = len(self.price_history)
        
        start_idx = max(0, len(self.price_history) - window)
        return np.array([self.price_history[i][asset_idx] for i in range(start_idx, len(self.price_history))])
    
    def get_volume_history(self, asset_idx: int, window: int = None) -> np.ndarray:
        """Get volume history for specific asset"""
        
        if asset_idx >= self.num_assets:
            return np.array([])
        
        if window is None:
            window = len(self.volume_history)
        
        start_idx = max(0, len(self.volume_history) - window)
        return np.array([self.volume_history[i][asset_idx] for i in range(start_idx, len(self.volume_history))])
    
    def get_technical_indicator(self, indicator_name: str, asset_idx: int = None) -> np.ndarray:
        """Get technical indicator values"""
        
        if indicator_name not in self.technical_indicators:
            raise ValueError(f"Unknown technical indicator: {indicator_name}")
        
        if asset_idx is not None:
            if asset_idx >= self.num_assets:
                return np.array([])
            return np.array([self.technical_indicators[indicator_name][asset_idx]])
        
        return self.technical_indicators[indicator_name].copy()
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get comprehensive market state"""
        
        return {
            "current_step": self.current_step,
            "prices": self.prices.copy(),
            "volume": self.volume.copy(),
            "bid_prices": self.bid_prices.copy(),
            "ask_prices": self.ask_prices.copy(),
            "technical_indicators": {k: v.copy() for k, v in self.technical_indicators.items()},
            "correlation_matrix": self.correlation_matrix.copy()
        }


if __name__ == "__main__":
    # Test the market simulator
    simulator = MarketSimulator({
        "num_assets": 5,
        "volatility": 0.02,
        "correlation": 0.1,
        "episode_length": 100
    })
    
    print("Testing Market Simulator")
    print(f"Initial prices: {simulator.get_current_prices()}")
    
    # Run a few steps
    for i in range(10):
        simulator.step()
        print(f"Step {i+1}: prices={simulator.get_current_prices()[:3]}")
    
    # Test technical indicators
    print(f"RSI: {simulator.get_technical_indicator('rsi')[:3]}")
    print(f"SMA Short: {simulator.get_technical_indicator('sma_short')[:3]}")
    
    # Test market observation
    obs = simulator.get_market_observation()
    print(f"Market observation shape: {obs.shape}")
