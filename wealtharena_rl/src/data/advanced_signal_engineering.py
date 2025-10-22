"""
Advanced Signal Engineering Module

This module provides comprehensive signal engineering capabilities including:
- Advanced technical indicators
- Volatility estimators (GARCH, EWMA, etc.)
- Fundamental signals
- Cross-asset correlation signals
- Regime detection algorithms
- Feature versioning system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import talib

logger = logging.getLogger(__name__)

class SignalType(Enum):
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    FUNDAMENTAL = "fundamental"
    CROSS_ASSET = "cross_asset"
    REGIME = "regime"
    NEWS = "news"

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    signal_type: SignalType
    parameters: Dict[str, Any]
    lookback_period: int = 252
    update_frequency: str = "daily"

class AdvancedSignalEngine:
    """Advanced signal engineering engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signal_cache = {}
        self.feature_versions = {}
        
    def generate_all_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate all types of signals for a given dataset"""
        df = data.copy()
        
        # Technical signals
        df = self._add_technical_signals(df)
        
        # Volatility signals
        df = self._add_volatility_signals(df)
        
        # Fundamental signals (if available)
        df = self._add_fundamental_signals(df)
        
        # Cross-asset correlation signals
        df = self._add_cross_asset_signals(df)
        
        # Regime detection signals
        df = self._add_regime_signals(df)
        
        # Clean and validate signals
        df = self._clean_signals(df)
        
        logger.info(f"Generated {len([c for c in df.columns if c not in data.columns])} signals for {symbol}")
        return df
    
    def _add_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators"""
        if len(df) < 50:
            return df
        
        # Advanced Moving Averages
        df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_100'] = talib.SMA(df['Close'], timeperiod=100)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        
        # Exponential Moving Averages
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
        df['EMA_200'] = talib.EMA(df['Close'], timeperiod=200)
        
        # Weighted Moving Averages
        df['WMA_20'] = talib.WMA(df['Close'], timeperiod=20)
        df['WMA_50'] = talib.WMA(df['Close'], timeperiod=50)
        
        # Hull Moving Average
        df['HMA_20'] = self._hull_moving_average(df['Close'], 20)
        df['HMA_50'] = self._hull_moving_average(df['Close'], 50)
        
        # Momentum Indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['High'], df['Low'], df['Close'])
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
        
        # MACD variations
        macd, macd_sig, macd_hist = talib.MACD(df['Close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd, macd_sig, macd_hist
        df['MACD_12_26_9'] = macd
        df['MACD_12_26_9_signal'] = macd_sig
        df['MACD_12_26_9_hist'] = macd_hist
        
        # Bollinger Bands variations
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = bb_upper, bb_middle, bb_lower
        df['BB_width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Keltner Channels
        df['KC_upper'], df['KC_middle'], df['KC_lower'] = self._keltner_channels(df, 20, 2)
        
        # Donchian Channels
        df['DC_upper'] = df['High'].rolling(20).max()
        df['DC_lower'] = df['Low'].rolling(20).min()
        df['DC_middle'] = (df['DC_upper'] + df['DC_lower']) / 2
        
        # Ichimoku Cloud
        ichimoku = self._ichimoku_cloud(df)
        df = pd.concat([df, ichimoku], axis=1)
        
        # Volume indicators
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        df['ADOSC'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        
        # Price patterns
        df['DOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['HAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        df['HARAMI'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
        df['MORNINGSTAR'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['EVENINGSTAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Trend indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['AROON_UP'], df['AROON_DOWN'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
        df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
        df['DX'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Support/Resistance
        df['SAR'] = talib.SAR(df['High'], df['Low'])
        df['SAREXT'] = talib.SAREXT(df['High'], df['Low'])
        
        return df
    
    def _add_volatility_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility indicators"""
        if len(df) < 50:
            return df
        
        # ATR variations
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])
        
        # Volatility calculations
        returns = df['Close'].pct_change()
        df['Volatility_5'] = returns.rolling(5).std() * np.sqrt(252)
        df['Volatility_10'] = returns.rolling(10).std() * np.sqrt(252)
        df['Volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        df['Volatility_50'] = returns.rolling(50).std() * np.sqrt(252)
        
        # EWMA Volatility
        df['EWMA_Vol_20'] = returns.ewm(span=20).std() * np.sqrt(252)
        df['EWMA_Vol_50'] = returns.ewm(span=50).std() * np.sqrt(252)
        
        # GARCH-like volatility (simplified)
        df['GARCH_Vol'] = self._garch_volatility(returns)
        
        # Parkinson Volatility
        df['Parkinson_Vol'] = self._parkinson_volatility(df)
        
        # Garman-Klass Volatility
        df['GK_Vol'] = self._garman_klass_volatility(df)
        
        # Rogers-Satchell Volatility
        df['RS_Vol'] = self._rogers_satchell_volatility(df)
        
        # Yang-Zhang Volatility
        df['YZ_Vol'] = self._yang_zhang_volatility(df)
        
        # Volatility of Volatility
        df['Vol_of_Vol'] = df['Volatility_20'].rolling(20).std()
        
        # Volatility percentiles
        df['Vol_Percentile_20'] = df['Volatility_20'].rolling(252).rank(pct=True)
        df['Vol_Percentile_50'] = df['Volatility_50'].rolling(252).rank(pct=True)
        
        return df
    
    def _add_fundamental_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental analysis signals"""
        # Price-to-Earnings ratio (if available)
        if 'PE' in df.columns:
            df['PE_Ratio'] = df['PE']
            df['PE_Percentile'] = df['PE_Ratio'].rolling(252).rank(pct=True)
        
        # Price-to-Book ratio (if available)
        if 'PB' in df.columns:
            df['PB_Ratio'] = df['PB']
            df['PB_Percentile'] = df['PB_Ratio'].rolling(252).rank(pct=True)
        
        # Dividend yield (if available)
        if 'Dividend_Yield' in df.columns:
            df['Div_Yield'] = df['Dividend_Yield']
            df['Div_Yield_Percentile'] = df['Div_Yield'].rolling(252).rank(pct=True)
        
        # Market cap signals (if available)
        if 'Market_Cap' in df.columns:
            df['Market_Cap_Log'] = np.log(df['Market_Cap'])
            df['Market_Cap_Growth'] = df['Market_Cap'].pct_change(252)
        
        # Revenue growth (if available)
        if 'Revenue' in df.columns:
            df['Revenue_Growth'] = df['Revenue'].pct_change(252)
            df['Revenue_Growth_MA'] = df['Revenue_Growth'].rolling(4).mean()
        
        return df
    
    def _add_cross_asset_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset correlation and momentum signals"""
        # This would typically require data from multiple assets
        # For now, we'll add placeholder signals that would be calculated
        # when multiple assets are available
        
        # Cross-asset momentum (placeholder)
        df['Cross_Asset_Momentum'] = 0.0
        df['Cross_Asset_Correlation'] = 0.0
        df['Cross_Asset_Volatility'] = 0.0
        
        return df
    
    def _add_regime_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection signals"""
        if len(df) < 100:
            return df
        
        # Volatility regime
        vol_20 = df['Volatility_20'].fillna(method='ffill')
        vol_regime = self._detect_volatility_regime(vol_20)
        df['Vol_Regime'] = vol_regime
        
        # Trend regime
        trend_regime = self._detect_trend_regime(df['Close'])
        df['Trend_Regime'] = trend_regime
        
        # Market regime (combination of volatility and trend)
        df['Market_Regime'] = self._detect_market_regime(df)
        
        # Regime persistence
        df['Regime_Persistence'] = self._calculate_regime_persistence(df['Market_Regime'])
        
        # Regime transition probability
        df['Regime_Transition_Prob'] = self._calculate_regime_transition_prob(df['Market_Regime'])
        
        return df
    
    def _hull_moving_average(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average"""
        wma_half = prices.rolling(period // 2).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
        wma_full = prices.rolling(period).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
        hma_raw = 2 * wma_half - wma_full
        hma = hma_raw.rolling(int(np.sqrt(period))).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
        return hma
    
    def _keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        middle = df['Close'].rolling(period).mean()
        atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        return upper, middle, lower
    
    def _ichimoku_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators"""
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        
        ichimoku = pd.DataFrame(index=df.index)
        ichimoku['Tenkan_Sen'] = (high_9 + low_9) / 2
        ichimoku['Kijun_Sen'] = (high_26 + low_26) / 2
        ichimoku['Senkou_Span_A'] = ((ichimoku['Tenkan_Sen'] + ichimoku['Kijun_Sen']) / 2).shift(26)
        ichimoku['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        ichimoku['Chikou_Span'] = df['Close'].shift(-26)
        
        return ichimoku
    
    def _garch_volatility(self, returns: pd.Series, omega: float = 0.1, alpha: float = 0.1, beta: float = 0.8) -> pd.Series:
        """Simplified GARCH volatility model"""
        vol = pd.Series(index=returns.index, dtype=float)
        vol.iloc[0] = returns.var()
        
        for i in range(1, len(returns)):
            vol.iloc[i] = omega + alpha * returns.iloc[i-1]**2 + beta * vol.iloc[i-1]
        
        return np.sqrt(vol * 252)
    
    def _parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        return np.sqrt(0.25 * np.log(df['High'] / df['Low'])**2).rolling(20).mean() * np.sqrt(252)
    
    def _garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        return np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2).rolling(20).mean() * np.sqrt(252)
    
    def _rogers_satchell_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Rogers-Satchell volatility estimator"""
        log_ho = np.log(df['High'] / df['Open'])
        log_hc = np.log(df['High'] / df['Close'])
        log_lo = np.log(df['Low'] / df['Open'])
        log_lc = np.log(df['Low'] / df['Close'])
        
        rs = log_ho * log_hc + log_lo * log_lc
        return np.sqrt(rs.rolling(20).mean() * 252)
    
    def _yang_zhang_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Yang-Zhang volatility estimator"""
        log_co = np.log(df['Close'] / df['Open'])
        log_oc = np.log(df['Open'] / df['Close'].shift(1))
        
        var_co = log_co.rolling(20).var()
        var_oc = log_oc.rolling(20).var()
        
        # Rogers-Satchell component
        rs_vol = self._rogers_satchell_volatility(df)
        
        # Yang-Zhang formula
        yz_vol = var_oc + 0.5 * var_co + 0.5 * rs_vol**2
        
        return np.sqrt(yz_vol * 252)
    
    def _detect_volatility_regime(self, volatility: pd.Series, threshold_low: float = 0.2, threshold_high: float = 0.8) -> pd.Series:
        """Detect volatility regime based on percentile thresholds"""
        vol_percentile = volatility.rolling(252).rank(pct=True)
        regime = pd.Series(index=volatility.index, dtype=int)
        regime[vol_percentile < threshold_low] = 0  # Low volatility
        regime[(vol_percentile >= threshold_low) & (vol_percentile <= threshold_high)] = 1  # Normal volatility
        regime[vol_percentile > threshold_high] = 2  # High volatility
        return regime.fillna(1)
    
    def _detect_trend_regime(self, prices: pd.Series) -> pd.Series:
        """Detect trend regime using multiple timeframes"""
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()
        sma_200 = prices.rolling(200).mean()
        
        regime = pd.Series(index=prices.index, dtype=int)
        regime[(sma_20 > sma_50) & (sma_50 > sma_200)] = 1  # Uptrend
        regime[(sma_20 < sma_50) & (sma_50 < sma_200)] = -1  # Downtrend
        regime[((sma_20 > sma_50) & (sma_50 < sma_200)) | ((sma_20 < sma_50) & (sma_50 > sma_200))] = 0  # Sideways
        return regime.fillna(0)
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect overall market regime combining volatility and trend"""
        vol_regime = df['Vol_Regime']
        trend_regime = df['Trend_Regime']
        
        # Combine regimes: 0=Low Vol+Uptrend, 1=Normal Vol+Uptrend, 2=High Vol+Uptrend,
        # 3=Low Vol+Sideways, 4=Normal Vol+Sideways, 5=High Vol+Sideways,
        # 6=Low Vol+Downtrend, 7=Normal Vol+Downtrend, 8=High Vol+Downtrend
        market_regime = vol_regime * 3 + (trend_regime + 1)
        return market_regime
    
    def _calculate_regime_persistence(self, regime: pd.Series) -> pd.Series:
        """Calculate how long the current regime has persisted"""
        persistence = pd.Series(index=regime.index, dtype=int)
        current_regime = regime.iloc[0]
        count = 0
        
        for i, r in enumerate(regime):
            if r == current_regime:
                count += 1
            else:
                current_regime = r
                count = 1
            persistence.iloc[i] = count
        
        return persistence
    
    def _calculate_regime_transition_prob(self, regime: pd.Series) -> pd.Series:
        """Calculate probability of regime transition"""
        # This is a simplified version - in practice, you'd use a Markov model
        transition_prob = pd.Series(index=regime.index, dtype=float)
        
        for i in range(1, len(regime)):
            if regime.iloc[i] != regime.iloc[i-1]:
                # Recent transition occurred
                transition_prob.iloc[i] = 1.0
            else:
                # No recent transition
                transition_prob.iloc[i] = 0.0
        
        # Smooth the transition probabilities
        transition_prob = transition_prob.rolling(5).mean()
        return transition_prob.fillna(0)
    
    def _clean_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate generated signals"""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values for most signals
        signal_columns = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
        df[signal_columns] = df[signal_columns].fillna(method='ffill')
        
        # Fill remaining NaN values with 0
        df = df.fillna(0)
        
        # Clip extreme values
        for col in signal_columns:
            if df[col].dtype in ['float64', 'float32']:
                q1, q99 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(q1, q99)
        
        return df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of generated signals"""
        signal_columns = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
        
        summary = {
            "total_signals": len(signal_columns),
            "signal_types": {},
            "missing_data_pct": {},
            "signal_categories": {
                "technical": len([c for c in signal_columns if any(x in c.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr'])]),
                "volatility": len([c for c in signal_columns if 'vol' in c.lower()]),
                "regime": len([c for c in signal_columns if 'regime' in c.lower()]),
                "momentum": len([c for c in signal_columns if any(x in c.lower() for x in ['momentum', 'roc', 'mom'])]),
                "volume": len([c for c in signal_columns if any(x in c.lower() for x in ['volume', 'obv', 'ad', 'mfi'])])
            }
        }
        
        for col in signal_columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            summary["missing_data_pct"][col] = missing_pct
        
        return summary
