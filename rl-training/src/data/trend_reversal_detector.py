"""
Trend Reversal Detection for WealthArena Trading System

This module implements advanced trend reversal detection algorithms combining
multiple technical indicators and statistical methods for robust signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import talib

logger = logging.getLogger(__name__)


class ReversalSignal(Enum):
    """Trend reversal signal types"""
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    UPTREND_CONTINUATION = "uptrend_continuation"
    DOWNTREND_CONTINUATION = "downtrend_continuation"
    NO_SIGNAL = "no_signal"


class TrendDirection(Enum):
    """Trend direction"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


@dataclass
class ReversalConfig:
    """Configuration for trend reversal detection"""
    
    # Moving Average parameters
    sma_short_period: int = 10
    sma_long_period: int = 30
    ema_short_period: int = 12
    ema_long_period: int = 26
    
    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_divergence_threshold: float = 5.0
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Volume parameters
    volume_spike_threshold: float = 1.5
    volume_ma_period: int = 20
    
    # Trend strength parameters
    trend_strength_threshold: float = 0.6
    min_trend_duration: int = 5
    
    # Reversal confirmation parameters
    confirmation_periods: int = 3
    min_reversal_magnitude: float = 0.02  # 2% minimum price change
    
    # Trend continuation parameters
    continuation_momentum_threshold: float = 0.6
    pullback_max_percentage: float = 0.05  # 5% max pullback for continuation
    continuation_volume_threshold: float = 1.2
    
    # Signal weights for composite scoring
    signal_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.signal_weights is None:
            self.signal_weights = {
                "ma_crossover": 0.20,
                "rsi_divergence": 0.15,
                "macd_signal": 0.15,
                "bollinger_squeeze": 0.10,
                "volume_confirmation": 0.15,
                "candlestick_pattern": 0.10,
                "trend_strength": 0.05,
                "momentum_continuation": 0.10
            }


class TrendReversalDetector:
    """
    Advanced trend reversal detection system
    
    Combines multiple technical indicators and statistical methods to identify
    high-probability trend reversal opportunities with confidence scoring.
    """
    
    def __init__(self, config: ReversalConfig = None):
        self.config = config or ReversalConfig()
        
        # Signal history for tracking
        self.signal_history = []
        self.trend_history = []
        self.confidence_history = []
        
        logger.info("Trend reversal detector initialized")
    
    def detect_reversal_signals(self, data: pd.DataFrame, lookback_periods: int = 50) -> pd.DataFrame:
        """
        Detect trend reversal signals for the given market data
        
        Args:
            data: DataFrame with OHLCV data
            lookback_periods: Number of periods to analyze for trend context
            
        Returns:
            DataFrame with reversal signals and confidence scores
        """
        if len(data) < max(self.config.sma_long_period, self.config.bb_period, lookback_periods):
            logger.warning("Insufficient data for trend reversal detection")
            return pd.DataFrame()
        
        # Calculate all required indicators
        indicators_df = self._calculate_indicators(data)
        
        # Detect trend direction
        trend_signals = self._detect_trend_direction(indicators_df)
        
        # Detect reversal signals
        reversal_signals = self._detect_reversal_patterns(indicators_df, trend_signals)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(indicators_df, reversal_signals)
        
        # Combine all signals
        result_df = pd.DataFrame({
            'trend_direction': trend_signals,
            'reversal_signal': reversal_signals,
            'confidence_score': confidence_scores,
            'timestamp': data.index
        })
        
        # Add individual signal components for analysis
        signal_components = self._extract_signal_components(indicators_df, reversal_signals)
        result_df = pd.concat([result_df, signal_components], axis=1)
        
        return result_df
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required technical indicators"""
        
        indicators = {}
        
        # Price data
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        open_prices = data['open'].values
        volume = data['volume'].values
        
        # Moving Averages
        indicators['sma_short'] = talib.SMA(close_prices, timeperiod=self.config.sma_short_period)
        indicators['sma_long'] = talib.SMA(close_prices, timeperiod=self.config.sma_long_period)
        indicators['ema_short'] = talib.EMA(close_prices, timeperiod=self.config.ema_short_period)
        indicators['ema_long'] = talib.EMA(close_prices, timeperiod=self.config.ema_long_period)
        
        # RSI and momentum
        indicators['rsi'] = talib.RSI(close_prices, timeperiod=self.config.rsi_period)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices, 
                                                 fastperiod=self.config.macd_fast,
                                                 slowperiod=self.config.macd_slow,
                                                 signalperiod=self.config.macd_signal)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices,
                                                    timeperiod=self.config.bb_period,
                                                    nbdevup=self.config.bb_std,
                                                    nbdevdn=self.config.bb_std)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # Volume indicators
        indicators['volume_sma'] = talib.SMA(volume.astype(float), timeperiod=self.config.volume_ma_period)
        indicators['volume_ratio'] = volume / indicators['volume_sma']
        
        # Price position relative to Bollinger Bands
        indicators['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
        
        # Price momentum
        indicators['price_change'] = np.diff(close_prices, prepend=close_prices[0])
        indicators['price_change_pct'] = indicators['price_change'] / close_prices
        
        # Trend strength indicators
        indicators['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Candlestick patterns
        indicators['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        indicators['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        indicators['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
        indicators['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        
        return pd.DataFrame(indicators, index=data.index)
    
    def _detect_trend_direction(self, indicators: pd.DataFrame) -> pd.Series:
        """Detect overall trend direction"""
        
        trend_signals = pd.Series(TrendDirection.SIDEWAYS, index=indicators.index)
        
        # Use multiple moving average crossovers for trend detection
        sma_short = indicators['sma_short']
        sma_long = indicators['sma_long']
        ema_short = indicators['ema_short']
        ema_long = indicators['ema_long']
        
        # Uptrend conditions
        uptrend_conditions = (
            (sma_short > sma_long) &
            (ema_short > ema_long) &
            (indicators['close'] > sma_short) &
            (indicators['close'] > ema_short)
        )
        
        # Downtrend conditions
        downtrend_conditions = (
            (sma_short < sma_long) &
            (ema_short < ema_long) &
            (indicators['close'] < sma_short) &
            (indicators['close'] < ema_short)
        )
        
        # Assign trend directions
        trend_signals[uptrend_conditions] = TrendDirection.UPTREND
        trend_signals[downtrend_conditions] = TrendDirection.DOWNTREND
        
        return trend_signals
    
    def _detect_reversal_patterns(self, indicators: pd.DataFrame, trend_signals: pd.Series) -> pd.Series:
        """Detect potential trend reversal and continuation patterns"""
        
        reversal_signals = pd.Series(ReversalSignal.NO_SIGNAL, index=indicators.index)
        
        # Bullish reversal signals
        bullish_signals = self._detect_bullish_reversals(indicators, trend_signals)
        reversal_signals[bullish_signals] = ReversalSignal.BULLISH_REVERSAL
        
        # Bearish reversal signals
        bearish_signals = self._detect_bearish_reversals(indicators, trend_signals)
        reversal_signals[bearish_signals] = ReversalSignal.BEARISH_REVERSAL
        
        # Uptrend continuation signals
        uptrend_continuation = self._detect_uptrend_continuation(indicators, trend_signals)
        reversal_signals[uptrend_continuation] = ReversalSignal.UPTREND_CONTINUATION
        
        # Downtrend continuation signals
        downtrend_continuation = self._detect_downtrend_continuation(indicators, trend_signals)
        reversal_signals[downtrend_continuation] = ReversalSignal.DOWNTREND_CONTINUATION
        
        return reversal_signals
    
    def _detect_bullish_reversals(self, indicators: pd.DataFrame, trend_signals: pd.Series) -> pd.Series:
        """Detect bullish reversal patterns"""
        
        bullish_conditions = pd.Series(False, index=indicators.index)
        
        # RSI divergence (price making lower lows, RSI making higher lows)
        rsi_divergence = self._detect_rsi_bullish_divergence(indicators)
        
        # Oversold RSI with reversal
        rsi_oversold_reversal = (
            (indicators['rsi'] < self.config.rsi_oversold) &
            (indicators['rsi'].shift(1) < indicators['rsi']) &
            (indicators['price_change_pct'] > 0)
        )
        
        # MACD bullish crossover
        macd_bullish = (
            (indicators['macd'] > indicators['macd_signal']) &
            (indicators['macd'].shift(1) <= indicators['macd_signal'].shift(1))
        )
        
        # Bollinger Bands bounce from lower band
        bb_bounce = (
            (indicators['bb_position'] < 0.1) &  # Near lower band
            (indicators['price_change_pct'] > 0)  # Price moving up
        )
        
        # Moving average support
        ma_support = (
            (indicators['close'] > indicators['sma_short']) &
            (indicators['close'].shift(1) <= indicators['sma_short'].shift(1))
        )
        
        # Volume confirmation
        volume_spike = indicators['volume_ratio'] > self.config.volume_spike_threshold
        
        # Candlestick patterns
        bullish_patterns = (
            (indicators['hammer'] > 0) |
            (indicators['engulfing'] > 0) |
            (indicators['doji'] > 0)
        )
        
        # Combine conditions (need multiple confirmations)
        bullish_conditions = (
            rsi_divergence |
            (rsi_oversold_reversal & volume_spike) |
            (macd_bullish & bb_bounce) |
            (ma_support & bullish_patterns & volume_spike)
        )
        
        return bullish_conditions
    
    def _detect_bearish_reversals(self, indicators: pd.DataFrame, trend_signals: pd.Series) -> pd.Series:
        """Detect bearish reversal patterns"""
        
        bearish_conditions = pd.Series(False, index=indicators.index)
        
        # RSI divergence (price making higher highs, RSI making lower highs)
        rsi_divergence = self._detect_rsi_bearish_divergence(indicators)
        
        # Overbought RSI with reversal
        rsi_overbought_reversal = (
            (indicators['rsi'] > self.config.rsi_overbought) &
            (indicators['rsi'].shift(1) > indicators['rsi']) &
            (indicators['price_change_pct'] < 0)
        )
        
        # MACD bearish crossover
        macd_bearish = (
            (indicators['macd'] < indicators['macd_signal']) &
            (indicators['macd'].shift(1) >= indicators['macd_signal'].shift(1))
        )
        
        # Bollinger Bands rejection from upper band
        bb_rejection = (
            (indicators['bb_position'] > 0.9) &  # Near upper band
            (indicators['price_change_pct'] < 0)  # Price moving down
        )
        
        # Moving average resistance
        ma_resistance = (
            (indicators['close'] < indicators['sma_short']) &
            (indicators['close'].shift(1) >= indicators['sma_short'].shift(1))
        )
        
        # Volume confirmation
        volume_spike = indicators['volume_ratio'] > self.config.volume_spike_threshold
        
        # Candlestick patterns
        bearish_patterns = (
            (indicators['shooting_star'] > 0) |
            (indicators['engulfing'] < 0) |
            (indicators['doji'] > 0)
        )
        
        # Combine conditions (need multiple confirmations)
        bearish_conditions = (
            rsi_divergence |
            (rsi_overbought_reversal & volume_spike) |
            (macd_bearish & bb_rejection) |
            (ma_resistance & bearish_patterns & volume_spike)
        )
        
        return bearish_conditions
    
    def _detect_uptrend_continuation(self, indicators: pd.DataFrame, trend_signals: pd.Series) -> pd.Series:
        """Detect uptrend continuation patterns after pullbacks"""
        
        continuation_signals = pd.Series(False, index=indicators.index)
        
        # Check for uptrend context
        uptrend_context = trend_signals == TrendDirection.UPTREND
        
        # Pullback detection (temporary decline in uptrend)
        pullback_conditions = self._detect_pullback_in_uptrend(indicators, trend_signals)
        
        # Momentum recovery after pullback
        momentum_recovery = self._detect_momentum_recovery(indicators, pullback_conditions, direction='up')
        
        # Volume confirmation during recovery
        volume_confirmation = indicators['volume_ratio'] > self.config.continuation_volume_threshold
        
        # Moving average support during pullback
        ma_support = (
            (indicators['close'] > indicators['sma_short']) |
            (indicators['close'] > indicators['ema_short'])
        )
        
        # MACD momentum alignment
        macd_momentum = indicators['macd'] > indicators['macd_signal']
        
        # RSI not oversold (healthy pullback)
        rsi_healthy = indicators['rsi'] > 40
        
        # Combine conditions
        continuation_signals = (
            uptrend_context &
            pullback_conditions &
            momentum_recovery &
            volume_confirmation &
            ma_support &
            macd_momentum &
            rsi_healthy
        )
        
        return continuation_signals
    
    def _detect_downtrend_continuation(self, indicators: pd.DataFrame, trend_signals: pd.Series) -> pd.Series:
        """Detect downtrend continuation patterns after bounces"""
        
        continuation_signals = pd.Series(False, index=indicators.index)
        
        # Check for downtrend context
        downtrend_context = trend_signals == TrendDirection.DOWNTREND
        
        # Bounce detection (temporary rise in downtrend)
        bounce_conditions = self._detect_bounce_in_downtrend(indicators, trend_signals)
        
        # Momentum resumption after bounce
        momentum_resumption = self._detect_momentum_recovery(indicators, bounce_conditions, direction='down')
        
        # Volume confirmation during resumption
        volume_confirmation = indicators['volume_ratio'] > self.config.continuation_volume_threshold
        
        # Moving average resistance during bounce
        ma_resistance = (
            (indicators['close'] < indicators['sma_short']) |
            (indicators['close'] < indicators['ema_short'])
        )
        
        # MACD momentum alignment
        macd_momentum = indicators['macd'] < indicators['macd_signal']
        
        # RSI not overbought (healthy bounce)
        rsi_healthy = indicators['rsi'] < 60
        
        # Combine conditions
        continuation_signals = (
            downtrend_context &
            bounce_conditions &
            momentum_resumption &
            volume_confirmation &
            ma_resistance &
            macd_momentum &
            rsi_healthy
        )
        
        return continuation_signals
    
    def _detect_pullback_in_uptrend(self, indicators: pd.DataFrame, trend_signals: pd.Series) -> pd.Series:
        """Detect pullbacks within an uptrend"""
        
        pullback_signals = pd.Series(False, index=indicators.index)
        
        # Look for temporary decline in uptrend
        for i in range(5, len(indicators)):
            if trend_signals.iloc[i] != TrendDirection.UPTREND:
                continue
            
            # Check recent price action
            recent_prices = indicators['close'].iloc[i-5:i+1]
            recent_highs = indicators['high'].iloc[i-5:i+1]
            
            # Find recent high and current low
            recent_high = recent_highs.max()
            current_low = recent_prices.iloc[-1]
            
            # Check if this is a pullback (decline from recent high)
            pullback_percentage = (recent_high - current_low) / recent_high
            
            # Valid pullback: decline but not too severe
            if (pullback_percentage > 0.01 and  # At least 1% decline
                pullback_percentage < self.config.pullback_max_percentage and  # Not more than max pullback
                indicators['close'].iloc[i-3] > indicators['close'].iloc[i-1]):  # Recent decline
                
                pullback_signals.iloc[i] = True
        
        return pullback_signals
    
    def _detect_bounce_in_downtrend(self, indicators: pd.DataFrame, trend_signals: pd.Series) -> pd.Series:
        """Detect bounces within a downtrend"""
        
        bounce_signals = pd.Series(False, index=indicators.index)
        
        # Look for temporary rise in downtrend
        for i in range(5, len(indicators)):
            if trend_signals.iloc[i] != TrendDirection.DOWNTREND:
                continue
            
            # Check recent price action
            recent_prices = indicators['close'].iloc[i-5:i+1]
            recent_lows = indicators['low'].iloc[i-5:i+1]
            
            # Find recent low and current high
            recent_low = recent_lows.min()
            current_high = recent_prices.iloc[-1]
            
            # Check if this is a bounce (rise from recent low)
            bounce_percentage = (current_high - recent_low) / recent_low
            
            # Valid bounce: rise but not too severe
            if (bounce_percentage > 0.01 and  # At least 1% rise
                bounce_percentage < self.config.pullback_max_percentage and  # Not more than max bounce
                indicators['close'].iloc[i-3] < indicators['close'].iloc[i-1]):  # Recent rise
                
                bounce_signals.iloc[i] = True
        
        return bounce_signals
    
    def _detect_momentum_recovery(self, indicators: pd.DataFrame, pullback_conditions: pd.Series, direction: str) -> pd.Series:
        """Detect momentum recovery after pullback/bounce"""
        
        recovery_signals = pd.Series(False, index=indicators.index)
        
        for i in range(3, len(indicators)):
            if not pullback_conditions.iloc[i]:
                continue
            
            # Check momentum in the direction of trend
            if direction == 'up':
                # For uptrend: look for upward momentum
                price_momentum = indicators['close'].iloc[i] > indicators['close'].iloc[i-1]
                macd_momentum = indicators['macd'].iloc[i] > indicators['macd'].iloc[i-1]
                rsi_momentum = indicators['rsi'].iloc[i] > indicators['rsi'].iloc[i-2]
                
                recovery_signals.iloc[i] = price_momentum and macd_momentum and rsi_momentum
                
            else:  # direction == 'down'
                # For downtrend: look for downward momentum
                price_momentum = indicators['close'].iloc[i] < indicators['close'].iloc[i-1]
                macd_momentum = indicators['macd'].iloc[i] < indicators['macd'].iloc[i-1]
                rsi_momentum = indicators['rsi'].iloc[i] < indicators['rsi'].iloc[i-2]
                
                recovery_signals.iloc[i] = price_momentum and macd_momentum and rsi_momentum
        
        return recovery_signals
    
    def _detect_rsi_bullish_divergence(self, indicators: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Detect RSI bullish divergence"""
        
        divergence_signals = pd.Series(False, index=indicators.index)
        
        for i in range(lookback, len(indicators)):
            # Look for price making lower lows while RSI makes higher lows
            recent_data = indicators.iloc[i-lookback:i+1]
            
            if len(recent_data) < 10:
                continue
            
            # Find recent lows
            price_lows = recent_data['close'].rolling(5).min()
            rsi_lows = recent_data['rsi'].rolling(5).min()
            
            # Check for divergence pattern
            if len(price_lows.dropna()) >= 2 and len(rsi_lows.dropna()) >= 2:
                recent_price_low = price_lows.iloc[-1]
                previous_price_low = price_lows.iloc[-2]
                recent_rsi_low = rsi_lows.iloc[-1]
                previous_rsi_low = rsi_lows.iloc[-2]
                
                # Bullish divergence: lower price low, higher RSI low
                if (recent_price_low < previous_price_low and 
                    recent_rsi_low > previous_rsi_low and
                    abs(recent_price_low - previous_price_low) / previous_price_low > 0.01):  # 1% minimum difference
                    divergence_signals.iloc[i] = True
        
        return divergence_signals
    
    def _detect_rsi_bearish_divergence(self, indicators: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Detect RSI bearish divergence"""
        
        divergence_signals = pd.Series(False, index=indicators.index)
        
        for i in range(lookback, len(indicators)):
            # Look for price making higher highs while RSI makes lower highs
            recent_data = indicators.iloc[i-lookback:i+1]
            
            if len(recent_data) < 10:
                continue
            
            # Find recent highs
            price_highs = recent_data['close'].rolling(5).max()
            rsi_highs = recent_data['rsi'].rolling(5).max()
            
            # Check for divergence pattern
            if len(price_highs.dropna()) >= 2 and len(rsi_highs.dropna()) >= 2:
                recent_price_high = price_highs.iloc[-1]
                previous_price_high = price_highs.iloc[-2]
                recent_rsi_high = rsi_highs.iloc[-1]
                previous_rsi_high = rsi_highs.iloc[-2]
                
                # Bearish divergence: higher price high, lower RSI high
                if (recent_price_high > previous_price_high and 
                    recent_rsi_high < previous_rsi_high and
                    abs(recent_price_high - previous_price_high) / previous_price_high > 0.01):  # 1% minimum difference
                    divergence_signals.iloc[i] = True
        
        return divergence_signals
    
    def _calculate_confidence_scores(self, indicators: pd.DataFrame, reversal_signals: pd.Series) -> pd.Series:
        """Calculate confidence scores for reversal signals"""
        
        confidence_scores = pd.Series(0.0, index=indicators.index)
        
        for i, signal in enumerate(reversal_signals):
            if signal == ReversalSignal.NO_SIGNAL:
                continue
            
            score = 0.0
            weights = self.config.signal_weights
            
            # RSI score
            if signal == ReversalSignal.BULLISH_REVERSAL:
                if indicators['rsi'].iloc[i] < self.config.rsi_oversold:
                    score += weights['rsi_divergence'] * (1.0 - indicators['rsi'].iloc[i] / 100.0)
            else:  # BEARISH_REVERSAL
                if indicators['rsi'].iloc[i] > self.config.rsi_overbought:
                    score += weights['rsi_divergence'] * (indicators['rsi'].iloc[i] / 100.0)
            
            # MACD score
            if signal == ReversalSignal.BULLISH_REVERSAL:
                if indicators['macd'].iloc[i] > indicators['macd_signal'].iloc[i]:
                    score += weights['macd_signal'] * 0.8
            else:  # BEARISH_REVERSAL
                if indicators['macd'].iloc[i] < indicators['macd_signal'].iloc[i]:
                    score += weights['macd_signal'] * 0.8
            
            # Bollinger Bands score
            bb_pos = indicators['bb_position'].iloc[i]
            if signal == ReversalSignal.BULLISH_REVERSAL and bb_pos < 0.2:
                score += weights['bollinger_squeeze'] * (1.0 - bb_pos)
            elif signal == ReversalSignal.BEARISH_REVERSAL and bb_pos > 0.8:
                score += weights['bollinger_squeeze'] * bb_pos
            
            # Volume score
            volume_ratio = indicators['volume_ratio'].iloc[i]
            if volume_ratio > self.config.volume_spike_threshold:
                score += weights['volume_confirmation'] * min(1.0, volume_ratio / 2.0)
            
            # Trend strength score
            adx = indicators['adx'].iloc[i]
            if adx > 25:  # Strong trend
                score += weights['trend_strength'] * min(1.0, adx / 50.0)
            
            # Momentum continuation score
            if signal in [ReversalSignal.UPTREND_CONTINUATION, ReversalSignal.DOWNTREND_CONTINUATION]:
                momentum_score = self._calculate_momentum_score(indicators, i, signal)
                score += weights['momentum_continuation'] * momentum_score
            
            confidence_scores.iloc[i] = min(1.0, score)
        
        return confidence_scores
    
    def _calculate_momentum_score(self, indicators: pd.DataFrame, index: int, signal: ReversalSignal) -> float:
        """Calculate momentum score for trend continuation signals"""
        
        if index < 5:
            return 0.0
        
        # Get recent momentum indicators
        recent_data = indicators.iloc[index-5:index+1]
        
        if signal == ReversalSignal.UPTREND_CONTINUATION:
            # Positive momentum indicators for uptrend continuation
            price_momentum = recent_data['close'].iloc[-1] > recent_data['close'].iloc[-3]
            macd_momentum = recent_data['macd'].iloc[-1] > recent_data['macd'].iloc[-2]
            rsi_trend = recent_data['rsi'].iloc[-1] > recent_data['rsi'].iloc[-2]
            ma_alignment = (recent_data['close'].iloc[-1] > recent_data['sma_short'].iloc[-1] and
                           recent_data['sma_short'].iloc[-1] > recent_data['sma_long'].iloc[-1])
            
            momentum_score = sum([price_momentum, macd_momentum, rsi_trend, ma_alignment]) / 4.0
            
        else:  # DOWNTREND_CONTINUATION
            # Negative momentum indicators for downtrend continuation
            price_momentum = recent_data['close'].iloc[-1] < recent_data['close'].iloc[-3]
            macd_momentum = recent_data['macd'].iloc[-1] < recent_data['macd'].iloc[-2]
            rsi_trend = recent_data['rsi'].iloc[-1] < recent_data['rsi'].iloc[-2]
            ma_alignment = (recent_data['close'].iloc[-1] < recent_data['sma_short'].iloc[-1] and
                           recent_data['sma_short'].iloc[-1] < recent_data['sma_long'].iloc[-1])
            
            momentum_score = sum([price_momentum, macd_momentum, rsi_trend, ma_alignment]) / 4.0
        
        return momentum_score
    
    def _extract_signal_components(self, indicators: pd.DataFrame, reversal_signals: pd.Series) -> pd.DataFrame:
        """Extract individual signal components for analysis"""
        
        components = pd.DataFrame(index=indicators.index)
        
        # RSI levels
        components['rsi_oversold'] = (indicators['rsi'] < self.config.rsi_oversold).astype(int)
        components['rsi_overbought'] = (indicators['rsi'] > self.config.rsi_overbought).astype(int)
        
        # MACD signals
        components['macd_bullish'] = (indicators['macd'] > indicators['macd_signal']).astype(int)
        components['macd_bearish'] = (indicators['macd'] < indicators['macd_signal']).astype(int)
        
        # Bollinger Bands position
        components['bb_oversold'] = (indicators['bb_position'] < 0.2).astype(int)
        components['bb_overbought'] = (indicators['bb_position'] > 0.8).astype(int)
        
        # Volume spikes
        components['volume_spike'] = (indicators['volume_ratio'] > self.config.volume_spike_threshold).astype(int)
        
        # Moving average crossovers
        components['ma_bullish_cross'] = (
            (indicators['sma_short'] > indicators['sma_long']) &
            (indicators['sma_short'].shift(1) <= indicators['sma_long'].shift(1))
        ).astype(int)
        
        components['ma_bearish_cross'] = (
            (indicators['sma_short'] < indicators['sma_long']) &
            (indicators['sma_short'].shift(1) >= indicators['sma_long'].shift(1))
        ).astype(int)
        
        # Candlestick patterns
        components['bullish_patterns'] = (
            (indicators['hammer'] > 0) |
            (indicators['engulfing'] > 0)
        ).astype(int)
        
        components['bearish_patterns'] = (
            (indicators['shooting_star'] > 0) |
            (indicators['engulfing'] < 0)
        ).astype(int)
        
        # Trend continuation components
        components['uptrend_continuation'] = (
            (indicators['close'] > indicators['sma_short']) &
            (indicators['sma_short'] > indicators['sma_long']) &
            (indicators['macd'] > indicators['macd_signal']) &
            (indicators['rsi'] > 40) &
            (indicators['rsi'] < 70)
        ).astype(int)
        
        components['downtrend_continuation'] = (
            (indicators['close'] < indicators['sma_short']) &
            (indicators['sma_short'] < indicators['sma_long']) &
            (indicators['macd'] < indicators['macd_signal']) &
            (indicators['rsi'] > 30) &
            (indicators['rsi'] < 60)
        ).astype(int)
        
        # Pullback and bounce indicators
        components['pullback_detected'] = self._detect_pullback_in_uptrend(indicators, pd.Series()).astype(int)
        components['bounce_detected'] = self._detect_bounce_in_downtrend(indicators, pd.Series()).astype(int)
        
        return components
    
    def get_reversal_signal_strength(self, data: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """
        Get detailed reversal signal information for a specific point in time
        
        Args:
            data: Market data DataFrame
            current_index: Index position to analyze
            
        Returns:
            Dictionary with signal details and confidence
        """
        
        if current_index < max(self.config.sma_long_period, self.config.bb_period):
            return {"error": "Insufficient historical data"}
        
        # Get signals for the current point
        signals_df = self.detect_reversal_signals(data.iloc[:current_index+1])
        
        if signals_df.empty or current_index >= len(signals_df):
            return {"error": "No signals available"}
        
        current_signal = signals_df.iloc[-1]
        
        return {
            "trend_direction": current_signal['trend_direction'].value,
            "reversal_signal": current_signal['reversal_signal'].value,
            "confidence_score": current_signal['confidence_score'],
            "timestamp": current_signal['timestamp'],
            "signal_components": {
                "rsi_oversold": current_signal.get('rsi_oversold', 0),
                "rsi_overbought": current_signal.get('rsi_overbought', 0),
                "macd_bullish": current_signal.get('macd_bullish', 0),
                "macd_bearish": current_signal.get('macd_bearish', 0),
                "bb_oversold": current_signal.get('bb_oversold', 0),
                "bb_overbought": current_signal.get('bb_overbought', 0),
                "volume_spike": current_signal.get('volume_spike', 0),
                "ma_bullish_cross": current_signal.get('ma_bullish_cross', 0),
                "ma_bearish_cross": current_signal.get('ma_bearish_cross', 0),
                "bullish_patterns": current_signal.get('bullish_patterns', 0),
                "bearish_patterns": current_signal.get('bearish_patterns', 0)
            }
        }


def create_trend_reversal_detector(config: ReversalConfig = None) -> TrendReversalDetector:
    """Create a trend reversal detector instance"""
    return TrendReversalDetector(config)


if __name__ == "__main__":
    # Test the trend reversal detector
    import pandas as pd
    import numpy as np
    
    # Create sample data with a clear trend reversal
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    # Create a downtrend followed by an uptrend (reversal)
    trend_change_point = 50
    prices = []
    base_price = 100
    
    for i in range(100):
        if i < trend_change_point:
            # Downtrend
            base_price += np.random.normal(-0.5, 1.0)
        else:
            # Uptrend (reversal)
            base_price += np.random.normal(0.5, 1.0)
        prices.append(base_price)
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.uniform(0, 2) for p in prices],
        'low': [p - np.random.uniform(0, 2) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Test the detector
    detector = TrendReversalDetector()
    signals = detector.detect_reversal_signals(data)
    
    print("Trend Reversal & Continuation Detection Test Results:")
    print("=" * 60)
    print(f"Total signals detected: {len(signals[signals['reversal_signal'] != 'no_signal'])}")
    print(f"Bullish reversals: {len(signals[signals['reversal_signal'] == 'bullish_reversal'])}")
    print(f"Bearish reversals: {len(signals[signals['reversal_signal'] == 'bearish_reversal'])}")
    print(f"Uptrend continuations: {len(signals[signals['reversal_signal'] == 'uptrend_continuation'])}")
    print(f"Downtrend continuations: {len(signals[signals['reversal_signal'] == 'downtrend_continuation'])}")
    
    # Show signals with confidence > 0.5
    high_confidence_signals = signals[signals['confidence_score'] > 0.5]
    if not high_confidence_signals.empty:
        print(f"\nHigh confidence signals (confidence > 0.5):")
        for idx, row in high_confidence_signals.iterrows():
            print(f"  {idx.date()}: {row['reversal_signal']} (confidence: {row['confidence_score']:.3f})")
    
    # Show trend direction distribution
    trend_distribution = signals['trend_direction'].value_counts()
    print(f"\nTrend direction distribution:")
    for trend, count in trend_distribution.items():
        print(f"  {trend}: {count}")
    
    print(f"\nTest completed successfully!")
