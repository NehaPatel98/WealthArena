"""
Trend Reversal Reward Component for WealthArena Trading Environment

This module implements a sophisticated reward component that incentivizes
the RL agent to correctly identify and trade trend reversals and continuations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from ..data.trend_reversal_detector import TrendReversalDetector, ReversalSignal, TrendDirection, ReversalConfig
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.trend_reversal_detector import TrendReversalDetector, ReversalSignal, TrendDirection, ReversalConfig

logger = logging.getLogger(__name__)


@dataclass
class TrendRewardConfig:
    """Configuration for trend reversal reward component"""
    
    # Base reward weights
    reversal_reward_weight: float = 2.0
    continuation_reward_weight: float = 1.5
    false_signal_penalty: float = -1.0
    missed_signal_penalty: float = -0.5
    
    # Signal strength thresholds
    min_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    
    # Timing rewards/penalties
    early_entry_bonus: float = 0.3
    late_entry_penalty: float = -0.2
    optimal_timing_window: int = 3  # Days
    
    # Position sizing rewards
    correct_size_bonus: float = 0.2
    oversized_penalty: float = -0.3
    
    # Risk management rewards
    stop_loss_reward: float = 0.1
    take_profit_reward: float = 0.15
    
    # Trend following rewards
    trend_following_reward: float = 0.1
    counter_trend_penalty: float = -0.2
    
    # Learning incentives
    exploration_bonus: float = 0.05
    consistency_bonus: float = 0.1


class TrendReversalReward:
    """
    Advanced reward component for trend reversal and continuation trading
    
    This component evaluates the agent's ability to:
    1. Identify trend reversals and continuations
    2. Time entries and exits correctly
    3. Manage position sizes appropriately
    4. Follow risk management rules
    """
    
    def __init__(self, config: TrendRewardConfig = None):
        self.config = config or TrendRewardConfig()
        
        # Initialize trend reversal detector
        reversal_config = ReversalConfig()
        self.trend_detector = TrendReversalDetector(reversal_config)
        
        # Track agent performance
        self.performance_history = []
        self.signal_history = []
        self.trade_history = []
        
        # Performance metrics
        self.correct_reversals = 0
        self.correct_continuations = 0
        self.false_signals = 0
        self.missed_signals = 0
        
        logger.info("Trend reversal reward component initialized")
    
    def calculate_reward(self, 
                        market_data: pd.DataFrame,
                        action: np.ndarray,
                        prev_action: np.ndarray,
                        portfolio_value: float,
                        prev_portfolio_value: float,
                        current_step: int,
                        symbol_index: int = 0) -> Dict[str, float]:
        """
        Calculate trend reversal reward for the current step
        
        Args:
            market_data: Current market data with OHLCV
            action: Current action taken by the agent
            prev_action: Previous action taken
            portfolio_value: Current portfolio value
            prev_portfolio_value: Previous portfolio value
            current_step: Current step in the episode
            symbol_index: Index of the symbol being traded
            
        Returns:
            Dictionary with reward components
        """
        
        if len(market_data) < 50:  # Need sufficient history
            return {"trend_reversal_reward": 0.0}
        
        # Get trend reversal signals
        signals_df = self.trend_detector.detect_reversal_signals(market_data)
        
        if signals_df.empty or current_step >= len(signals_df):
            return {"trend_reversal_reward": 0.0}
        
        current_signal = signals_df.iloc[current_step]
        
        # Calculate individual reward components
        reward_components = {}
        
        # 1. Signal detection reward
        signal_reward = self._calculate_signal_detection_reward(
            current_signal, action, prev_action, market_data, current_step
        )
        reward_components["signal_detection"] = signal_reward
        
        # 2. Timing reward
        timing_reward = self._calculate_timing_reward(
            current_signal, action, market_data, current_step
        )
        reward_components["timing"] = timing_reward
        
        # 3. Position sizing reward
        sizing_reward = self._calculate_sizing_reward(
            current_signal, action, portfolio_value, market_data, current_step
        )
        reward_components["position_sizing"] = sizing_reward
        
        # 4. Risk management reward
        risk_reward = self._calculate_risk_management_reward(
            current_signal, action, portfolio_value, prev_portfolio_value
        )
        reward_components["risk_management"] = risk_reward
        
        # 5. Trend following reward
        trend_following_reward = self._calculate_trend_following_reward(
            current_signal, action, market_data, current_step
        )
        reward_components["trend_following"] = trend_following_reward
        
        # 6. Learning incentive reward
        learning_reward = self._calculate_learning_incentive_reward(
            current_signal, action, current_step
        )
        reward_components["learning_incentive"] = learning_reward
        
        # Combine all components
        total_reward = sum(reward_components.values())
        
        # Update performance tracking
        self._update_performance_tracking(current_signal, action, total_reward)
        
        reward_components["trend_reversal_reward"] = total_reward
        
        return reward_components
    
    def _calculate_signal_detection_reward(self, 
                                         signal_data: pd.Series,
                                         action: np.ndarray,
                                         prev_action: np.ndarray,
                                         market_data: pd.DataFrame,
                                         current_step: int) -> float:
        """Calculate reward for correctly detecting signals"""
        
        signal_type = signal_data['reversal_signal']
        confidence = signal_data['confidence_score']
        
        # Only reward high-confidence signals
        if confidence < self.config.min_confidence_threshold:
            return 0.0
        
        # Determine if agent took appropriate action
        action_magnitude = np.abs(action[0])  # Assuming first element is main action
        
        if signal_type == ReversalSignal.BULLISH_REVERSAL:
            # Agent should buy (positive action)
            if action[0] > 0.1:  # Significant buy action
                reward = self.config.reversal_reward_weight * confidence
                if confidence > self.config.high_confidence_threshold:
                    reward *= 1.5  # Bonus for high confidence
                return reward
            else:
                return self.config.missed_signal_penalty * confidence
        
        elif signal_type == ReversalSignal.BEARISH_REVERSAL:
            # Agent should sell (negative action)
            if action[0] < -0.1:  # Significant sell action
                reward = self.config.reversal_reward_weight * confidence
                if confidence > self.config.high_confidence_threshold:
                    reward *= 1.5  # Bonus for high confidence
                return reward
            else:
                return self.config.missed_signal_penalty * confidence
        
        elif signal_type == ReversalSignal.UPTREND_CONTINUATION:
            # Agent should continue buying or hold
            if action[0] >= 0:  # Buy or hold
                reward = self.config.continuation_reward_weight * confidence
                return reward
            else:
                return self.config.false_signal_penalty * confidence
        
        elif signal_type == ReversalSignal.DOWNTREND_CONTINUATION:
            # Agent should continue selling or hold
            if action[0] <= 0:  # Sell or hold
                reward = self.config.continuation_reward_weight * confidence
                return reward
            else:
                return self.config.false_signal_penalty * confidence
        
        return 0.0
    
    def _calculate_timing_reward(self,
                               signal_data: pd.Series,
                               action: np.ndarray,
                               market_data: pd.DataFrame,
                               current_step: int) -> float:
        """Calculate reward for optimal timing"""
        
        signal_type = signal_data['reversal_signal']
        confidence = signal_data['confidence_score']
        
        if confidence < self.config.min_confidence_threshold:
            return 0.0
        
        # Check if this is an optimal entry point
        if current_step < self.config.optimal_timing_window:
            return 0.0
        
        # Look at recent price action to determine timing quality
        recent_prices = market_data['close'].iloc[current_step-5:current_step+1]
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        if signal_type in [ReversalSignal.BULLISH_REVERSAL, ReversalSignal.UPTREND_CONTINUATION]:
            # Good timing if entering near the low
            if action[0] > 0.1 and price_change < 0.02:  # Entering near recent low
                return self.config.early_entry_bonus * confidence
            elif action[0] > 0.1 and price_change > 0.05:  # Entering after significant rise
                return self.config.late_entry_penalty * confidence
        
        elif signal_type in [ReversalSignal.BEARISH_REVERSAL, ReversalSignal.DOWNTREND_CONTINUATION]:
            # Good timing if entering near the high
            if action[0] < -0.1 and price_change > -0.02:  # Entering near recent high
                return self.config.early_entry_bonus * confidence
            elif action[0] < -0.1 and price_change < -0.05:  # Entering after significant decline
                return self.config.late_entry_penalty * confidence
        
        return 0.0
    
    def _calculate_sizing_reward(self,
                               signal_data: pd.Series,
                               action: np.ndarray,
                               portfolio_value: float,
                               market_data: pd.DataFrame,
                               current_step: int) -> float:
        """Calculate reward for appropriate position sizing"""
        
        signal_type = signal_data['reversal_signal']
        confidence = signal_data['confidence_score']
        
        if confidence < self.config.min_confidence_threshold:
            return 0.0
        
        action_magnitude = np.abs(action[0])
        
        # Determine optimal position size based on signal strength
        if confidence > self.config.high_confidence_threshold:
            optimal_size = 0.8  # Large position for high confidence
        elif confidence > self.config.min_confidence_threshold:
            optimal_size = 0.5  # Medium position for medium confidence
        else:
            optimal_size = 0.2  # Small position for low confidence
        
        # Calculate sizing accuracy
        size_accuracy = 1.0 - abs(action_magnitude - optimal_size)
        
        if size_accuracy > 0.8:
            return self.config.correct_size_bonus * confidence
        elif action_magnitude > optimal_size * 1.5:
            return self.config.oversized_penalty * confidence
        
        return 0.0
    
    def _calculate_risk_management_reward(self,
                                        signal_data: pd.Series,
                                        action: np.ndarray,
                                        portfolio_value: float,
                                        prev_portfolio_value: float) -> float:
        """Calculate reward for risk management"""
        
        # Portfolio return
        portfolio_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Reward for cutting losses early
        if portfolio_return < -0.05 and action[0] < -0.1:  # Stop loss
            return self.config.stop_loss_reward
        
        # Reward for taking profits
        if portfolio_return > 0.1 and action[0] < -0.1:  # Take profit
            return self.config.take_profit_reward
        
        return 0.0
    
    def _calculate_trend_following_reward(self,
                                        signal_data: pd.Series,
                                        action: np.ndarray,
                                        market_data: pd.DataFrame,
                                        current_step: int) -> float:
        """Calculate reward for trend following behavior"""
        
        trend_direction = signal_data['trend_direction']
        signal_type = signal_data['reversal_signal']
        
        # Reward trend following
        if (trend_direction == TrendDirection.UPTREND and action[0] > 0) or \
           (trend_direction == TrendDirection.DOWNTREND and action[0] < 0):
            return self.config.trend_following_reward
        
        # Penalty for counter-trend trading without reversal signal
        if signal_type == ReversalSignal.NO_SIGNAL:
            if (trend_direction == TrendDirection.UPTREND and action[0] < 0) or \
               (trend_direction == TrendDirection.DOWNTREND and action[0] > 0):
                return self.config.counter_trend_penalty
        
        return 0.0
    
    def _calculate_learning_incentive_reward(self,
                                           signal_data: pd.Series,
                                           action: np.ndarray,
                                           current_step: int) -> float:
        """Calculate reward for learning incentives"""
        
        # Exploration bonus for trying new actions
        if current_step < 100:  # Early in episode
            return self.config.exploration_bonus
        
        # Consistency bonus for consistent performance
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            if np.std(recent_performance) < 0.1:  # Low volatility in performance
                return self.config.consistency_bonus
        
        return 0.0
    
    def _update_performance_tracking(self,
                                   signal_data: pd.Series,
                                   action: np.ndarray,
                                   total_reward: float):
        """Update performance tracking metrics"""
        
        self.performance_history.append(total_reward)
        self.signal_history.append(signal_data['reversal_signal'])
        
        # Update counters
        signal_type = signal_data['reversal_signal']
        confidence = signal_data['confidence_score']
        
        if confidence > self.config.min_confidence_threshold:
            if signal_type in [ReversalSignal.BULLISH_REVERSAL, ReversalSignal.BEARISH_REVERSAL]:
                if np.abs(action[0]) > 0.1:
                    self.correct_reversals += 1
                else:
                    self.missed_signals += 1
            
            elif signal_type in [ReversalSignal.UPTREND_CONTINUATION, ReversalSignal.DOWNTREND_CONTINUATION]:
                if np.abs(action[0]) > 0.05:
                    self.correct_continuations += 1
                else:
                    self.missed_signals += 1
        
        # Track false signals (low confidence but large action)
        if confidence < self.config.min_confidence_threshold and np.abs(action[0]) > 0.2:
            self.false_signals += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the agent"""
        
        total_signals = len(self.signal_history)
        if total_signals == 0:
            return {"error": "No signals processed"}
        
        return {
            "total_signals": total_signals,
            "correct_reversals": self.correct_reversals,
            "correct_continuations": self.correct_continuations,
            "false_signals": self.false_signals,
            "missed_signals": self.missed_signals,
            "reversal_accuracy": self.correct_reversals / max(1, self.correct_reversals + self.missed_signals),
            "continuation_accuracy": self.correct_continuations / max(1, self.correct_continuations + self.missed_signals),
            "false_signal_rate": self.false_signals / total_signals,
            "average_reward": np.mean(self.performance_history) if self.performance_history else 0.0,
            "reward_volatility": np.std(self.performance_history) if len(self.performance_history) > 1 else 0.0
        }
    
    def reset_performance_tracking(self):
        """Reset performance tracking for new episode"""
        self.performance_history = []
        self.signal_history = []
        self.correct_reversals = 0
        self.correct_continuations = 0
        self.false_signals = 0
        self.missed_signals = 0
        logger.info("Performance tracking reset")


def create_trend_reversal_reward(config: TrendRewardConfig = None) -> TrendReversalReward:
    """Create a trend reversal reward component instance"""
    return TrendReversalReward(config)


if __name__ == "__main__":
    # Test the trend reversal reward component
    import pandas as pd
    import numpy as np
    
    print("Testing Trend Reversal Reward Component...")
    print("=" * 50)
    
    # Create sample market data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    market_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Create reward component
    reward_component = TrendReversalReward()
    
    # Test with different actions
    test_actions = [
        np.array([0.8]),   # Large buy
        np.array([0.3]),   # Medium buy
        np.array([0.0]),   # Hold
        np.array([-0.3]),  # Medium sell
        np.array([-0.8])   # Large sell
    ]
    
    print("Testing reward calculation for different actions...")
    
    for i, action in enumerate(test_actions):
        reward = reward_component.calculate_reward(
            market_data=market_data,
            action=action,
            prev_action=np.array([0.0]),
            portfolio_value=100000,
            prev_portfolio_value=100000,
            current_step=50,
            symbol_index=0
        )
        
        print(f"Action {i+1} ({action[0]:+.1f}): Total reward = {reward['trend_reversal_reward']:.3f}")
        print(f"  Components: {', '.join([f'{k}: {v:.3f}' for k, v in reward.items() if k != 'trend_reversal_reward'])}")
    
    # Get performance metrics
    metrics = reward_component.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nTest completed successfully!")
