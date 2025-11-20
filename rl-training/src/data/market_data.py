"""
Market Data Processing for WealthArena Trading System

This module provides market data processing capabilities including technical analysis,
data transformation, and feature engineering for the trading environment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path

# Try to import talib, set flag if available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicator:
    """Technical indicator specification"""
    name: str
    function: str
    parameters: Dict[str, Any]
    description: str = ""


class TechnicalCalculator:
    """
    Technical analysis calculator for market data
    
    Provides comprehensive technical analysis capabilities including
    trend indicators, momentum indicators, volatility indicators, and volume indicators.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.indicators = self._initialize_indicators()
        
        logger.info("Technical calculator initialized")
    
    def _initialize_indicators(self) -> Dict[str, TechnicalIndicator]:
        """Initialize available technical indicators"""
        
        indicators = {
            # Trend Indicators
            "sma": TechnicalIndicator(
                name="Simple Moving Average",
                function="SMA",
                parameters={"timeperiod": 20},
                description="Simple moving average over specified period"
            ),
            "ema": TechnicalIndicator(
                name="Exponential Moving Average", 
                function="EMA",
                parameters={"timeperiod": 20},
                description="Exponential moving average over specified period"
            ),
            "wma": TechnicalIndicator(
                name="Weighted Moving Average",
                function="WMA", 
                parameters={"timeperiod": 20},
                description="Weighted moving average over specified period"
            ),
            "bb": TechnicalIndicator(
                name="Bollinger Bands",
                function="BBANDS",
                parameters={"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
                description="Bollinger Bands with upper, middle, and lower bands"
            ),
            "macd": TechnicalIndicator(
                name="MACD",
                function="MACD",
                parameters={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                description="Moving Average Convergence Divergence"
            ),
            
            # Momentum Indicators
            "rsi": TechnicalIndicator(
                name="Relative Strength Index",
                function="RSI",
                parameters={"timeperiod": 14},
                description="Relative Strength Index momentum oscillator"
            ),
            "stoch": TechnicalIndicator(
                name="Stochastic Oscillator",
                function="STOCH",
                parameters={"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
                description="Stochastic oscillator"
            ),
            "williams_r": TechnicalIndicator(
                name="Williams %R",
                function="WILLR",
                parameters={"timeperiod": 14},
                description="Williams %R momentum indicator"
            ),
            "cci": TechnicalIndicator(
                name="Commodity Channel Index",
                function="CCI",
                parameters={"timeperiod": 14},
                description="Commodity Channel Index"
            ),
            
            # Volatility Indicators
            "atr": TechnicalIndicator(
                name="Average True Range",
                function="ATR",
                parameters={"timeperiod": 14},
                description="Average True Range volatility indicator"
            ),
            "natr": TechnicalIndicator(
                name="Normalized Average True Range",
                function="NATR",
                parameters={"timeperiod": 14},
                description="Normalized Average True Range"
            ),
            "trange": TechnicalIndicator(
                name="True Range",
                function="TRANGE",
                parameters={},
                description="True Range"
            ),
            
            # Volume Indicators
            "obv": TechnicalIndicator(
                name="On Balance Volume",
                function="OBV",
                parameters={},
                description="On Balance Volume"
            ),
            "ad": TechnicalIndicator(
                name="Accumulation/Distribution",
                function="AD",
                parameters={},
                description="Accumulation/Distribution Line"
            ),
            "adx": TechnicalIndicator(
                name="Average Directional Index",
                function="ADX",
                parameters={"timeperiod": 14},
                description="Average Directional Index trend strength"
            ),
            
            # Price Pattern Recognition
            "doji": TechnicalIndicator(
                name="Doji",
                function="CDLDOJI",
                parameters={},
                description="Doji candlestick pattern"
            ),
            "hammer": TechnicalIndicator(
                name="Hammer",
                function="CDLHAMMER",
                parameters={},
                description="Hammer candlestick pattern"
            ),
            "engulfing": TechnicalIndicator(
                name="Engulfing",
                function="CDLENGULFING",
                parameters={},
                description="Engulfing candlestick pattern"
            )
        }
        
        return indicators
    
    def calculate_indicators(self, data: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """Calculate technical indicators for market data"""
        
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return data
        
        # Convert to numpy arrays for talib
        open_prices = data["open"].values.astype(np.float64)
        high_prices = data["high"].values.astype(np.float64)
        low_prices = data["low"].values.astype(np.float64)
        close_prices = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)
        
        # Calculate indicators
        result_data = data.copy()
        
        if indicators is None:
            indicators = list(self.indicators.keys())
        
        # Check if TA-Lib is available, use fallback if not
        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available, using pandas-based fallbacks for technical indicators")
            result_data = self._calculate_indicators_fallback(data, indicators)
            return result_data
        
        for indicator_name in indicators:
            if indicator_name not in self.indicators:
                logger.warning(f"Unknown indicator: {indicator_name}")
                continue
            
            try:
                indicator = self.indicators[indicator_name]
                indicator_values = self._calculate_single_indicator(
                    indicator, open_prices, high_prices, low_prices, close_prices, volume
                )
                
                # Add to result
                if isinstance(indicator_values, tuple):
                    # Multiple values returned (e.g., Bollinger Bands)
                    for i, values in enumerate(indicator_values):
                        if values is not None:
                            col_name = f"{indicator_name}_{i}" if i > 0 else indicator_name
                            result_data[col_name] = values
                else:
                    # Single value returned
                    if indicator_values is not None:
                        result_data[indicator_name] = indicator_values
                
            except Exception as e:
                logger.error(f"Error calculating {indicator_name}: {e}")
                continue
        
        return result_data
    
    def _calculate_indicators_fallback(self, data: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Calculate indicators using pandas-based fallbacks when TA-Lib is unavailable"""
        result_data = data.copy()
        
        # Simplified pandas-based implementations
        for indicator_name in indicators:
            try:
                if indicator_name == "sma":
                    timeperiod = self.indicators.get("sma", {}).get("parameters", {}).get("timeperiod", 20)
                    result_data["sma"] = data["close"].rolling(timeperiod).mean()
                elif indicator_name == "ema":
                    timeperiod = self.indicators.get("ema", {}).get("parameters", {}).get("timeperiod", 20)
                    result_data["ema"] = data["close"].ewm(span=timeperiod).mean()
                elif indicator_name == "rsi":
                    delta = data["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    result_data["rsi"] = 100 - (100 / (1 + rs))
                elif indicator_name == "macd":
                    ema12 = data["close"].ewm(span=12).mean()
                    ema26 = data["close"].ewm(span=26).mean()
                    result_data["macd"] = ema12 - ema26
                    result_data["macd_signal"] = result_data["macd"].ewm(span=9).mean()
                    result_data["macd_hist"] = result_data["macd"] - result_data["macd_signal"]
                elif indicator_name == "bb":
                    bb_period = 20
                    bb_middle = data["close"].rolling(bb_period).mean()
                    bb_std = data["close"].rolling(bb_period).std()
                    result_data["bb_upper"] = bb_middle + (bb_std * 2)
                    result_data["bb_middle"] = bb_middle
                    result_data["bb_lower"] = bb_middle - (bb_std * 2)
                elif indicator_name == "atr":
                    high_low = data["high"] - data["low"]
                    high_close = np.abs(data["high"] - data["close"].shift())
                    low_close = np.abs(data["low"] - data["close"].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = np.max(ranges, axis=1)
                    result_data["atr"] = true_range.rolling(14).mean()
                # Add more fallbacks as needed
            except Exception as e:
                logger.warning(f"Fallback calculation failed for {indicator_name}: {e}")
                continue
        
        return result_data
    
    def _calculate_single_indicator(self, 
                                  indicator: TechnicalIndicator,
                                  open_prices: np.ndarray,
                                  high_prices: np.ndarray,
                                  low_prices: np.ndarray,
                                  close_prices: np.ndarray,
                                  volume: np.ndarray) -> Any:
        """Calculate a single technical indicator"""
        
        # Guard against TA-Lib unavailability
        if not TALIB_AVAILABLE:
            logger.warning("TA-Lib not available, cannot calculate indicator using TA-Lib")
            return None
        
        function_name = indicator.function
        parameters = indicator.parameters
        
        try:
            if function_name == "SMA":
                return talib.SMA(close_prices, timeperiod=parameters["timeperiod"])
            
            elif function_name == "EMA":
                return talib.EMA(close_prices, timeperiod=parameters["timeperiod"])
            
            elif function_name == "WMA":
                return talib.WMA(close_prices, timeperiod=parameters["timeperiod"])
            
            elif function_name == "BBANDS":
                return talib.BBANDS(
                    close_prices,
                    timeperiod=parameters["timeperiod"],
                    nbdevup=parameters["nbdevup"],
                    nbdevdn=parameters["nbdevdn"]
                )
            
            elif function_name == "MACD":
                return talib.MACD(
                    close_prices,
                    fastperiod=parameters["fastperiod"],
                    slowperiod=parameters["slowperiod"],
                    signalperiod=parameters["signalperiod"]
                )
            
            elif function_name == "RSI":
                return talib.RSI(close_prices, timeperiod=parameters["timeperiod"])
            
            elif function_name == "STOCH":
                return talib.STOCH(
                    high_prices, low_prices, close_prices,
                    fastk_period=parameters["fastk_period"],
                    slowk_period=parameters["slowk_period"],
                    slowd_period=parameters["slowd_period"]
                )
            
            elif function_name == "WILLR":
                return talib.WILLR(
                    high_prices, low_prices, close_prices,
                    timeperiod=parameters["timeperiod"]
                )
            
            elif function_name == "CCI":
                return talib.CCI(
                    high_prices, low_prices, close_prices,
                    timeperiod=parameters["timeperiod"]
                )
            
            elif function_name == "ATR":
                return talib.ATR(
                    high_prices, low_prices, close_prices,
                    timeperiod=parameters["timeperiod"]
                )
            
            elif function_name == "NATR":
                return talib.NATR(
                    high_prices, low_prices, close_prices,
                    timeperiod=parameters["timeperiod"]
                )
            
            elif function_name == "TRANGE":
                return talib.TRANGE(high_prices, low_prices, close_prices)
            
            elif function_name == "OBV":
                return talib.OBV(close_prices, volume)
            
            elif function_name == "AD":
                return talib.AD(high_prices, low_prices, close_prices, volume)
            
            elif function_name == "ADX":
                return talib.ADX(
                    high_prices, low_prices, close_prices,
                    timeperiod=parameters["timeperiod"]
                )
            
            elif function_name == "CDLDOJI":
                return talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            
            elif function_name == "CDLHAMMER":
                return talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            
            elif function_name == "CDLENGULFING":
                return talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            
            else:
                logger.warning(f"Unknown function: {function_name}")
                return None
        
        except Exception as e:
            logger.error(f"Error calculating {function_name}: {e}")
            return None
    
    def get_indicator_info(self, indicator_name: str) -> Optional[TechnicalIndicator]:
        """Get information about a specific indicator"""
        
        return self.indicators.get(indicator_name)
    
    def list_indicators(self) -> List[str]:
        """List all available indicators"""
        
        return list(self.indicators.keys())


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming DataFrames to lowercase column names for consistency.
    
    Args:
        df: DataFrame with potentially mixed case column names
        
    Returns:
        DataFrame with lowercase column names
    """
    if df.empty:
        return df
    
    # Create mapping from current to lowercase
    rename_map = {col: col.lower() for col in df.columns}
    df_normalized = df.rename(columns=rename_map)
    
    return df_normalized


class MarketDataProcessor:
    """
    Market data processor for WealthArena trading system
    
    Handles data transformation, feature engineering, and preprocessing
    for the trading environment.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.technical_calculator = TechnicalCalculator(config.get("technical", {}))
        
        # Processing parameters
        self.normalize_features = self.config.get("normalize_features", True)
        self.feature_scaling = self.config.get("feature_scaling", "standard")  # standard, minmax, robust
        self.handle_missing = self.config.get("handle_missing", "forward_fill")
        
        logger.info("Market data processor initialized")
    
    def process_market_data(self, 
                          data: pd.DataFrame, 
                          symbols: List[str] = None,
                          indicators: List[str] = None) -> pd.DataFrame:
        """Process market data with technical indicators and feature engineering"""
        
        if data.empty:
            return data
        
        # Normalize column names to lowercase at the start
        data = normalize_column_names(data)
        logger.info(f"Processing market data: {len(data)} records")
        
        # Calculate technical indicators
        if indicators:
            data = self.technical_calculator.calculate_indicators(data, indicators)
        
        # Feature engineering
        data = self._engineer_features(data)
        
        # Handle missing data
        data = self._handle_missing_data(data)
        
        # Normalize features
        if self.normalize_features:
            data = self._normalize_features(data)
        
        # Add derived features
        data = self._add_derived_features(data)
        
        logger.info(f"Processed market data: {data.shape}")
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from market data"""
        
        if data.empty:
            return data
        
        # Price-based features
        if "close" in data.columns:
            # Returns
            data["returns"] = data["close"].pct_change()
            data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
            
            # Price ratios
            if "open" in data.columns:
                data["open_close_ratio"] = data["open"] / data["close"]
            if "high" in data.columns:
                data["high_close_ratio"] = data["high"] / data["close"]
            if "low" in data.columns:
                data["low_close_ratio"] = data["low"] / data["close"]
            
            # Volatility
            data["volatility"] = data["returns"].rolling(window=20).std()
            data["log_volatility"] = data["log_returns"].rolling(window=20).std()
        
        # Volume-based features
        if "volume" in data.columns:
            # Volume ratios
            data["volume_ma"] = data["volume"].rolling(window=20).mean()
            data["volume_ratio"] = data["volume"] / data["volume_ma"]
            
            # Volume-weighted average price (VWAP)
            if "high" in data.columns and "low" in data.columns and "close" in data.columns:
                typical_price = (data["high"] + data["low"] + data["close"]) / 3
                data["vwap"] = (typical_price * data["volume"]).rolling(window=20).sum() / data["volume"].rolling(window=20).sum()
        
        # Time-based features
        if data.index.dtype == 'datetime64[ns]':
            data["hour"] = data.index.hour
            data["day_of_week"] = data.index.dayofweek
            data["month"] = data.index.month
            data["quarter"] = data.index.quarter
        
        return data
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in the dataset"""
        
        if data.empty:
            return data
        
        if self.handle_missing == "forward_fill":
            data = data.ffill()
        elif self.handle_missing == "backward_fill":
            # WARNING: Backward fill can cause data leakage in production
            # Only use for specific cases where future data is acceptable
            logger.warning("Using backward fill - ensure this is intentional to avoid data leakage")
            data = data.bfill()
        elif self.handle_missing == "interpolate":
            data = data.interpolate()
        elif self.handle_missing == "drop":
            data = data.dropna()
        else:
            logger.warning(f"Unknown missing data handling method: {self.handle_missing}")
        
        return data
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for machine learning"""
        
        if data.empty:
            return data
        
        # Select numeric columns for normalization
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if self.feature_scaling == "standard":
            # Z-score normalization
            data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()
        
        elif self.feature_scaling == "minmax":
            # Min-max normalization
            data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].min()) / (data[numeric_columns].max() - data[numeric_columns].min())
        
        elif self.feature_scaling == "robust":
            # Robust normalization (using median and IQR)
            median = data[numeric_columns].median()
            iqr = data[numeric_columns].quantile(0.75) - data[numeric_columns].quantile(0.25)
            data[numeric_columns] = (data[numeric_columns] - median) / iqr
        
        return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for trading"""
        
        if data.empty:
            return data
        
        # Market regime features
        if "returns" in data.columns:
            # Bull/bear market indicators
            data["bull_market"] = (data["returns"] > 0).astype(int)
            data["bear_market"] = (data["returns"] < 0).astype(int)
            
            # Volatility regime
            data["high_volatility"] = (data["volatility"] > data["volatility"].quantile(0.8)).astype(int)
            data["low_volatility"] = (data["volatility"] < data["volatility"].quantile(0.2)).astype(int)
        
        # Trend features
        if "sma" in data.columns and "close" in data.columns:
            data["above_sma"] = (data["close"] > data["sma"]).astype(int)
            data["below_sma"] = (data["close"] < data["sma"]).astype(int)
        
        # Momentum features
        if "rsi" in data.columns:
            data["rsi_overbought"] = (data["rsi"] > 70).astype(int)
            data["rsi_oversold"] = (data["rsi"] < 30).astype(int)
        
        # Volume features
        if "volume_ratio" in data.columns:
            data["high_volume"] = (data["volume_ratio"] > 1.5).astype(int)
            data["low_volume"] = (data["volume_ratio"] < 0.5).astype(int)
        
        return data
    
    def create_training_features(self, data: pd.DataFrame, target_column: str = "returns") -> Tuple[pd.DataFrame, pd.Series]:
        """Create features and target for training"""
        
        if data.empty:
            return data, pd.Series()
        
        # Select feature columns (exclude target and non-numeric columns)
        feature_columns = data.select_dtypes(include=[np.number]).columns
        feature_columns = feature_columns[feature_columns != target_column]
        
        # Create features and target
        X = data[feature_columns]
        y = data[target_column] if target_column in data.columns else pd.Series()
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def get_feature_importance(self, data: pd.DataFrame, target_column: str = "returns") -> pd.Series:
        """Calculate feature importance using correlation"""
        
        if data.empty:
            return pd.Series()
        
        # Select numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if target_column not in numeric_columns:
            return pd.Series()
        
        # Calculate correlations
        correlations = data[numeric_columns].corr()[target_column].abs().sort_values(ascending=False)
        
        # Remove target column from results
        correlations = correlations[correlations.index != target_column]
        
        return correlations


# Utility functions
def create_rolling_features(data: pd.DataFrame, 
                          columns: List[str], 
                          windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """Create rolling window features"""
    
    if data.empty:
        return data
    
    result_data = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
        
        for window in windows:
            # Rolling mean
            result_data[f"{col}_ma_{window}"] = data[col].rolling(window=window).mean()
            
            # Rolling std
            result_data[f"{col}_std_{window}"] = data[col].rolling(window=window).std()
            
            # Rolling min/max
            result_data[f"{col}_min_{window}"] = data[col].rolling(window=window).min()
            result_data[f"{col}_max_{window}"] = data[col].rolling(window=window).max()
    
    return result_data


def create_lag_features(data: pd.DataFrame, 
                       columns: List[str], 
                       lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Create lagged features"""
    
    if data.empty:
        return data
    
    result_data = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
        
        for lag in lags:
            result_data[f"{col}_lag_{lag}"] = data[col].shift(lag)
    
    return result_data


# Utility functions for data download with retry and validation
# NOTE: This function is currently not integrated into collect_training_data.py
# TODO: Either refactor collect_training_data.py to use this function, or remove it if duplicate logic exists
def download_with_retry_and_validation(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    timeout: int = 10,
    asset_type: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Robust wrapper for yfinance downloads with retry logic and validation.
    
    Args:
        symbol: Symbol to download (e.g., 'BHP.AX')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        max_retries: Maximum number of retry attempts (default 3)
        timeout: Timeout in seconds per request (default 10)
        asset_type: Optional asset type for validation
        
    Returns:
        DataFrame with market data or None if download fails
    """
    import yfinance as yf
    import time
    
    retry_delays = [2.0, 5.0, 10.0]  # Exponential backoff: 2s, 5s, 10s
    
    for attempt in range(max_retries):
        try:
            # Download data - Ticker.history() doesn't support timeout parameter
            # If timeout is needed, use yf.download() instead which supports timeout
            # For now, use Ticker.history() without timeout and handle timeouts at request level
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True
            )
            
            # Validate response
            if df is None or df.empty:
                logger.warning(f"{symbol}: Empty dataframe returned (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delays[attempt])
                    continue
                return None
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"{symbol}: Missing required columns: {missing_columns}")
                return None
            
            # Validate data quality
            if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
                logger.warning(f"{symbol}: Invalid price data (non-positive values)")
                # Filter out invalid rows
                mask = (df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
                df = df.loc[mask].copy()
                if df.empty:
                    logger.warning(f"{symbol}: All rows filtered out after validation")
                    return None
            
            # Log success
            logger.info(f"{symbol}: Successfully downloaded {len(df)} rows")
            return df
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"{symbol}: Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
            
            # Check for specific error types
            if "timeout" in error_msg or "read timed out" in error_msg:
                logger.warning(f"{symbol}: Timeout error, will retry with longer delay")
            elif "no timezone" in error_msg or "delisted" in error_msg:
                logger.warning(f"{symbol}: Symbol may be delisted, skipping immediately")
                return None
            elif "expecting value" in error_msg or "json" in error_msg:
                # JSON parsing error - likely rate limiting
                logger.warning(f"{symbol}: JSON parse error (likely rate limiting), will retry with longer delay")
                if attempt < max_retries - 1:
                    time.sleep(retry_delays[attempt] * 2)  # Double delay for JSON errors
                    continue
            elif "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"{symbol}: Rate limit detected, waiting longer before retry")
                if attempt < max_retries - 1:
                    time.sleep(retry_delays[attempt] * 2)  # Double delay for rate limits
                    continue
            
            # Retry with exponential backoff
            if attempt < max_retries - 1:
                time.sleep(retry_delays[attempt])
            else:
                logger.error(f"{symbol}: All {max_retries} attempts failed")
                return None
    
    return None


def _load_symbol_health_cache(cache_file: Path = None) -> Dict[str, Dict[str, Any]]:
    """
    Load symbol health check cache from JSON file.
    
    Args:
        cache_file: Path to cache file (default: data/cache/symbol_health.json)
        
    Returns:
        Dictionary mapping symbols to cache entries with timestamp and result
    """
    import json
    from datetime import datetime, timedelta
    
    if cache_file is None:
        cache_file = Path("data/cache/symbol_health.json")
    
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        logger.warning(f"Failed to load health cache: {e}")
        return {}


def _save_symbol_health_cache(cache_data: Dict[str, Dict[str, Any]], cache_file: Path = None):
    """
    Save symbol health check cache to JSON file.
    
    Args:
        cache_data: Dictionary mapping symbols to cache entries
        cache_file: Path to cache file (default: data/cache/symbol_health.json)
    """
    import json
    
    if cache_file is None:
        cache_file = Path("data/cache/symbol_health.json")
    
    # Create cache directory if it doesn't exist
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save health cache: {e}")


def _prune_old_cache_entries(cache_data: Dict[str, Dict[str, Any]], max_age_hours: int = 24) -> Dict[str, Dict[str, Any]]:
    """
    Prune old cache entries older than max_age_hours.
    
    Args:
        cache_data: Dictionary mapping symbols to cache entries
        max_age_hours: Maximum age in hours (default 24)
        
    Returns:
        Pruned cache dictionary
    """
    from datetime import datetime, timedelta
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    pruned_cache = {}
    
    for symbol, entry in cache_data.items():
        try:
            timestamp_str = entry.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff_time:
                    pruned_cache[symbol] = entry
        except Exception:
            # If timestamp parsing fails, keep entry for now
            pruned_cache[symbol] = entry
    
    return pruned_cache


def quick_symbol_health_check(symbols: List[str], sample_size: int = 10) -> List[str]:
    """
    Quick health check on a sample of symbols to identify likely valid ones.
    
    Args:
        symbols: List of symbols to check
        sample_size: Number of symbols to test (default 10, or 10% of total)
        
    Returns:
        List of symbols that are likely to succeed
    """
    import yfinance as yf
    import time
    
    if not symbols:
        return []
    
    # Load health check cache
    cache_file = Path("data/cache/symbol_health.json")
    cache_data = _load_symbol_health_cache(cache_file)
    cache_data = _prune_old_cache_entries(cache_data, max_age_hours=24)
    
    # Test sample size (10% or minimum 10 symbols)
    test_size = max(min(sample_size, len(symbols)), int(len(symbols) * 0.1))
    test_symbols = symbols[:test_size]
    
    valid_symbols = []
    invalid_symbols = []
    
    logger.info(f"Quick health check: Testing {len(test_symbols)} symbols out of {len(symbols)}...")
    
    for symbol in test_symbols:
        # Check cache first
        from datetime import datetime, timedelta
        if symbol in cache_data:
            entry = cache_data[symbol]
            timestamp_str = entry.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                if age_hours < 24:
                    # Use cached result
                    if entry.get("valid", False):
                        valid_symbols.append(symbol)
                        logger.debug(f"✓ {symbol}: Valid (from cache, age: {age_hours:.1f}h)")
                    else:
                        invalid_symbols.append(symbol)
                        logger.debug(f"✗ {symbol}: Invalid (from cache, age: {age_hours:.1f}h)")
                    continue
            except Exception:
                # Cache entry invalid, re-validate
                pass
        try:
            # Use lightweight yf.download() instead of Ticker.info for faster validation
            # Try to download minimal data (1 month) to check if symbol is valid
            df = yf.download(
                symbol, 
                period='1mo', 
                interval='1d', 
                progress=False, 
                group_by='column', 
                threads=False, 
                timeout=10
            )
            
            # Consider symbol valid if download returns non-empty OHLCV data
            if df is not None and not df.empty:
                # Check if we have OHLCV columns
                required_cols = ['Open', 'High', 'Low', 'Close']
                if isinstance(df.columns, pd.MultiIndex):
                    # Multi-level columns, check first level
                    if any(col in [c[1] if len(c) > 1 else c[0] for c in df.columns] for col in required_cols):
                        valid_symbols.append(symbol)
                        logger.debug(f"✓ {symbol}: Valid (data available)")
                    else:
                        invalid_symbols.append(symbol)
                        logger.debug(f"✗ {symbol}: Missing required columns")
                        # Cache invalid result
                        cache_data[symbol] = {
                            "timestamp": datetime.now().isoformat(),
                            "valid": False
                        }
                elif any(col in df.columns for col in required_cols):
                    valid_symbols.append(symbol)
                    logger.debug(f"✓ {symbol}: Valid (data available)")
                    # Cache valid result
                    cache_data[symbol] = {
                        "timestamp": datetime.now().isoformat(),
                        "valid": True
                    }
                else:
                    invalid_symbols.append(symbol)
                    logger.debug(f"✗ {symbol}: Missing required columns")
                    # Cache invalid result
                    cache_data[symbol] = {
                        "timestamp": datetime.now().isoformat(),
                        "valid": False
                    }
            else:
                invalid_symbols.append(symbol)
                logger.debug(f"✗ {symbol}: No data returned")
                # Cache invalid result
                cache_data[symbol] = {
                    "timestamp": datetime.now().isoformat(),
                    "valid": False
                }
        
        except Exception as e:
            invalid_symbols.append(symbol)
            error_msg = str(e).lower()
            # Handle specific errors with backoff
            if "json" in error_msg or "429" in error_msg or "rate limit" in error_msg:
                logger.debug(f"✗ {symbol}: API rate limit/JSON error - waiting longer")
                time.sleep(1.0)  # Longer delay for rate limits
            else:
                logger.debug(f"✗ {symbol}: Validation failed - {str(e)[:50]}")
        
        # Small delay between requests (0.2-0.5s as suggested)
        time.sleep(0.3)
    
    # Save updated cache
    _save_symbol_health_cache(cache_data, cache_file)
    
    # Calculate validity rate
    if len(test_symbols) > 0:
        validity_rate = len(valid_symbols) / len(test_symbols)
        logger.info(f"Health check results: {len(valid_symbols)}/{len(test_symbols)} valid ({validity_rate*100:.1f}%)")
        
        # If sample validity rate is very low, filter all symbols more aggressively
        if validity_rate < 0.3:
            logger.warning(f"Low validity rate ({validity_rate*100:.1f}%), applying aggressive filtering")
            # Return only symbols that match known good patterns
            from src.data.asx.asx_symbols import is_symbol_likely_valid
            filtered = [s for s in symbols if is_symbol_likely_valid(s)]
            logger.info(f"Filtered from {len(symbols)} to {len(filtered)} symbols")
            return filtered
    
    # If sample looks good, apply pattern-based validation to remaining symbols
    if len(symbols) > test_size:
        remaining_symbols = symbols[test_size:]
        from src.data.asx.asx_symbols import is_symbol_likely_valid
        
        for symbol in remaining_symbols:
            if is_symbol_likely_valid(symbol):
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)
    
    logger.info(f"Health check complete: {len(valid_symbols)} valid, {len(invalid_symbols)} invalid")
    
    return valid_symbols


if __name__ == "__main__":
    # Test the market data processor
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "open": 100 + np.random.randn(100).cumsum(),
        "high": 100 + np.random.randn(100).cumsum() + 2,
        "low": 100 + np.random.randn(100).cumsum() - 2,
        "close": 100 + np.random.randn(100).cumsum(),
        "volume": np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Test technical calculator
    calculator = TechnicalCalculator()
    data_with_indicators = calculator.calculate_indicators(data, ["sma", "rsi", "macd"])
    print(f"Data with indicators shape: {data_with_indicators.shape}")
    print(f"New columns: {[col for col in data_with_indicators.columns if col not in data.columns]}")
    
    # Test market data processor
    processor = MarketDataProcessor()
    processed_data = processor.process_market_data(data_with_indicators)
    print(f"Processed data shape: {processed_data.shape}")
    
    # Test feature importance
    importance = processor.get_feature_importance(processed_data, "returns")
    print(f"Top 5 most important features: {importance.head()}")
