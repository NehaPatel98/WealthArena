"""
Live Market Data Service
Fetches real-time market data for RL model inference and game system

IMPORTANT: Financial Asset Code Formats (yfinance conventions)
=============================================================

This service uses yfinance library, which follows Yahoo Finance symbol conventions.
DO NOT reinvent the wheel - use these established formats:

1. STOCKS (US & International):
   - US Stocks: Standard ticker (e.g., 'AAPL', 'MSFT', 'GOOGL', 'TSLA')
   - ASX Stocks: Add '.AX' suffix (e.g., 'BHP.AX', 'CBA.AX', 'ANZ.AX')
   - LSE Stocks: Add '.L' suffix (e.g., 'BP.L', 'VOD.L')
   - TSE Stocks: Add '.T' suffix (e.g., 'RY.T')
   - Other exchanges: Check yfinance documentation for suffix patterns

2. FOREX (Currency Pairs):
   - Format: 'BASEQUOTE=X' (e.g., 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X')
   - Major pairs: EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X
   - Cross pairs: EURGBP=X, EURJPY=X, GBPJPY=X
   - Note: The '=X' suffix is REQUIRED for forex pairs in yfinance
   - Volume may be 0 or missing for forex (this is normal)

3. CRYPTOCURRENCIES:
   - Format: 'COIN-USD' or 'COIN-USDT' (e.g., 'BTC-USD', 'ETH-USD', 'SOL-USD')
   - Major cryptos: BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD, XRP-USD
   - Stablecoins: USDT-USD, USDC-USD
   - Note: Use '-USD' for USD pairs, '-USDT' for USDT pairs
   - yfinance supports most major cryptocurrencies

4. COMMODITIES:
   - Format: 'SYMBOL=F' (e.g., 'GC=F' for Gold, 'CL=F' for Crude Oil)
   - Gold: 'GC=F' (COMEX Gold Futures)
   - Silver: 'SI=F' (COMEX Silver Futures)
   - Crude Oil: 'CL=F' (WTI Crude Oil Futures)
   - Natural Gas: 'NG=F'
   - Corn: 'ZC=F', Wheat: 'ZW=F', Soybeans: 'ZS=F'
   - Note: The '=F' suffix indicates futures contracts

5. ETFs (Exchange-Traded Funds):
   - US ETFs: Standard ticker (e.g., 'SPY', 'QQQ', 'IWM', 'VTI')
   - International ETFs: May need exchange suffix
   - Bond ETFs: 'TLT', 'HYG', 'LQD'
   - Commodity ETFs: 'GLD', 'SLV', 'USO'
   - Note: ETFs work like stocks in yfinance

6. INDICES:
   - S&P 500: '^GSPC'
   - Dow Jones: '^DJI'
   - NASDAQ: '^IXIC'
   - Russell 2000: '^RUT'
   - Note: Use '^' prefix for indices

KNOWN LIMITATIONS & BEST PRACTICES:
===================================

1. Market Hours:
   - Stock data updates during market hours (9:30 AM - 4:00 PM ET for US markets)
   - After hours data may be limited
   - Forex and crypto trade 24/7, but yfinance may have delays

2. Data Availability:
   - Some symbols may not be available in yfinance
   - International stocks may have limited historical data
   - Always check data availability before using

3. Volume Data:
   - Forex pairs typically have 0 volume (this is normal)
   - Some commodities may have limited volume data
   - Crypto volume is usually available

4. Rate Limiting:
   - yfinance may throttle requests if too many are made quickly
   - This service uses 60-second caching to minimize API calls
   - If you see rate limit errors, increase cache duration

5. Symbol Validation:
   - Always validate symbol format before fetching
   - Use check_data_availability() to verify data exists
   - Handle None returns gracefully

EXAMPLES:
=========

# Stocks
get_live_data('AAPL')  # Apple Inc.
get_live_data('BHP.AX')  # BHP Group (ASX)

# Forex
get_live_data('EURUSD=X')  # Euro/US Dollar
get_live_data('GBPJPY=X')  # British Pound/Japanese Yen

# Crypto
get_live_data('BTC-USD')  # Bitcoin
get_live_data('ETH-USD')  # Ethereum

# Commodities
get_live_data('GC=F')  # Gold Futures
get_live_data('CL=F')  # Crude Oil Futures

# ETFs
get_live_data('SPY')  # S&P 500 ETF
get_live_data('QQQ')  # NASDAQ ETF

# Indices
get_live_data('^GSPC')  # S&P 500 Index

For more information, see: https://github.com/ranaroussi/yfinance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LiveDataService:
    """
    Service to fetch live market data for RL models and games
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # Cache for 60 seconds
        logger.info("Live Data Service initialized")
    
    def get_live_data(self, symbol: str, days: int = 60, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get live market data for a symbol
        
        Supports multiple asset types following yfinance conventions:
        - Stocks: 'AAPL', 'MSFT', 'BHP.AX' (with exchange suffix)
        - Forex: 'EURUSD=X', 'GBPUSD=X' (requires =X suffix)
        - Crypto: 'BTC-USD', 'ETH-USD' (requires -USD or -USDT suffix)
        - Commodities: 'GC=F', 'CL=F' (requires =F suffix for futures)
        - ETFs: 'SPY', 'QQQ' (standard ticker)
        - Indices: '^GSPC', '^DJI' (requires ^ prefix)
        
        Args:
            symbol: Financial symbol in yfinance format
                    Examples:
                    - Stocks: 'AAPL', 'MSFT', 'BHP.AX'
                    - Forex: 'EURUSD=X', 'GBPUSD=X'
                    - Crypto: 'BTC-USD', 'ETH-USD'
                    - Commodities: 'GC=F', 'CL=F'
                    - ETFs: 'SPY', 'QQQ'
                    - Indices: '^GSPC', '^DJI'
            days: Number of days of historical data to fetch (default: 60)
            use_cache: Whether to use cached data if available (default: True)
            
        Returns:
            DataFrame with OHLCV data, or None if data unavailable
            
        Note:
            - Forex pairs may have 0 volume (this is normal)
            - Some symbols may not be available in yfinance
            - Data updates depend on market hours for stocks
            - Crypto and forex trade 24/7 but may have delays
        """
        cache_key = f"{symbol}_{days}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                logger.debug(f"Using cached data for {symbol}")
                return cached_data.copy()
        
        try:
            logger.info(f"Fetching live data for {symbol} (last {days} days)")
            ticker = yf.Ticker(symbol.upper())
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Ensure we have required columns
            # Note: Volume may be 0 or missing for forex pairs (this is normal)
            required_cols = ['Open', 'High', 'Low', 'Close']
            optional_cols = ['Volume']
            
            for col in required_cols:
                if col not in hist.columns:
                    logger.warning(f"Missing required column {col} for {symbol}")
                    return None
            
            # Volume is optional (forex pairs often have 0 volume)
            if 'Volume' not in hist.columns:
                logger.debug(f"Volume column missing for {symbol}, adding with zeros (normal for forex)")
                hist['Volume'] = 0
            
            # Add Date column if not present
            if 'Date' not in hist.columns:
                hist['Date'] = hist.index
            
            # Reset index to make Date a column
            hist = hist.reset_index()
            
            # Cache the data
            self.cache[cache_key] = (hist.copy(), datetime.now())
            
            logger.info(f"Successfully fetched {len(hist)} data points for {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest price for a symbol
        
        Supports all asset types (stocks, forex, crypto, commodities, ETFs, indices)
        Uses yfinance symbol conventions (see module docstring for formats)
        
        Args:
            symbol: Financial symbol in yfinance format
                    Examples:
                    - 'AAPL' (stock), 'EURUSD=X' (forex), 'BTC-USD' (crypto)
                    - 'GC=F' (commodity), 'SPY' (ETF), '^GSPC' (index)
            
        Returns:
            Current price or None if unavailable
            
        Note:
            - For forex pairs, price is the exchange rate
            - For indices, price is the index value
            - May return None if market is closed or symbol invalid
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            if info and 'regularMarketPrice' in info:
                return float(info['regularMarketPrice'])
            
            # Fallback: get from history
            hist = ticker.history(period='1d', interval='1m')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def check_data_availability(self, symbol: str) -> Dict[str, Any]:
        """
        Check if live data is available for a symbol
        
        Validates symbol format and checks if yfinance can fetch data.
        Useful for validating symbols before attempting to fetch full data.
        
        Args:
            symbol: Financial symbol in yfinance format
                    Supports: stocks, forex, crypto, commodities, ETFs, indices
                    See module docstring for format requirements
            
        Returns:
            Dict with:
            - 'available': bool - Whether data is available
            - 'up_to_date': bool - Whether data is current (within 1 day)
            - 'latest_date': str - ISO date of latest data point
            - 'current_date': str - Current date for comparison
            - 'data_points': int - Number of data points available
            - 'has_info': bool - Whether ticker info is available
            - 'error': str - Error message if check failed
            
        Example:
            >>> service.check_data_availability('AAPL')
            {
                'symbol': 'AAPL',
                'available': True,
                'up_to_date': True,
                'latest_date': '2024-11-18',
                'current_date': '2024-11-18',
                'data_points': 5,
                'has_info': True
            }
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            # Try to get recent data
            hist = ticker.history(period='5d', interval='1d')
            
            is_available = not hist.empty and len(hist) > 0
            latest_date = hist.index[-1].date() if not hist.empty else None
            current_date = datetime.now().date()
            
            # Check if data is up to date (within 1 day)
            is_up_to_date = False
            if latest_date:
                days_diff = (current_date - latest_date).days
                is_up_to_date = days_diff <= 1
            
            return {
                'symbol': symbol.upper(),
                'available': is_available,
                'up_to_date': is_up_to_date,
                'latest_date': latest_date.isoformat() if latest_date else None,
                'current_date': current_date.isoformat(),
                'data_points': len(hist) if not hist.empty else 0,
                'has_info': info is not None and len(info) > 0
            }
            
        except Exception as e:
            logger.error(f"Error checking data availability for {symbol}: {e}")
            return {
                'symbol': symbol.upper(),
                'available': False,
                'up_to_date': False,
                'error': str(e)
            }
    
    def get_multiple_symbols(self, symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Get live data for multiple symbols
        
        Supports mixed asset types (stocks, forex, crypto, etc.)
        Each symbol must follow yfinance format conventions.
        
        Args:
            symbols: List of financial symbols in yfinance format
                    Can mix asset types:
                    - ['AAPL', 'MSFT'] (stocks)
                    - ['EURUSD=X', 'GBPUSD=X'] (forex)
                    - ['BTC-USD', 'ETH-USD'] (crypto)
                    - ['GC=F', 'CL=F'] (commodities)
                    - Mixed: ['AAPL', 'EURUSD=X', 'BTC-USD']
            days: Number of days of historical data (default: 60)
            
        Returns:
            Dict mapping symbol to DataFrame
            Only includes symbols that were successfully fetched
            Failed symbols are silently skipped (check logs for errors)
            
        Example:
            >>> service.get_multiple_symbols(['AAPL', 'EURUSD=X', 'BTC-USD'])
            {
                'AAPL': DataFrame(...),
                'EURUSD=X': DataFrame(...),
                'BTC-USD': DataFrame(...)
            }
        """
        results = {}
        for symbol in symbols:
            data = self.get_live_data(symbol, days)
            if data is not None:
                results[symbol] = data
        return results
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    @staticmethod
    def detect_asset_type(symbol: str) -> str:
        """
        Detect asset type from symbol format (yfinance conventions)
        
        This method helps identify what type of asset a symbol represents
        based on yfinance formatting patterns. Useful for validation and logging.
        
        Args:
            symbol: Financial symbol to analyze
            
        Returns:
            Asset type string: 'stock', 'forex', 'crypto', 'commodity', 'etf', 'index', or 'unknown'
            
        Examples:
            >>> LiveDataService.detect_asset_type('AAPL')
            'stock'
            >>> LiveDataService.detect_asset_type('EURUSD=X')
            'forex'
            >>> LiveDataService.detect_asset_type('BTC-USD')
            'crypto'
            >>> LiveDataService.detect_asset_type('GC=F')
            'commodity'
            >>> LiveDataService.detect_asset_type('SPY')
            'etf'  # or 'stock' - ETFs are hard to distinguish
            >>> LiveDataService.detect_asset_type('^GSPC')
            'index'
        """
        symbol_upper = symbol.upper().strip()
        
        # Indices: start with ^
        if symbol_upper.startswith('^'):
            return 'index'
        
        # Forex: ends with =X
        if symbol_upper.endswith('=X'):
            return 'forex'
        
        # Commodities: ends with =F
        if symbol_upper.endswith('=F'):
            return 'commodity'
        
        # Crypto: contains -USD or -USDT
        if '-USD' in symbol_upper or '-USDT' in symbol_upper:
            return 'crypto'
        
        # Stocks/ETFs: standard ticker format
        # Note: ETFs are hard to distinguish from stocks without a symbol list
        # Common ETFs that might be detected: SPY, QQQ, IWM, VTI, etc.
        # For now, we'll classify as 'stock' (ETFs work the same way in yfinance)
        if '.' in symbol_upper:
            # Exchange suffix (e.g., .AX, .L, .T) - likely a stock
            return 'stock'
        
        # Default to stock for standard tickers
        # Could be stock or ETF, but both work the same in yfinance
        return 'stock'
    
    @staticmethod
    def validate_symbol_format(symbol: str) -> Tuple[bool, str]:
        """
        Validate symbol format and provide helpful error messages
        
        Checks if symbol follows yfinance conventions and suggests corrections.
        
        Args:
            symbol: Financial symbol to validate
            
        Returns:
            Tuple of (is_valid: bool, message: str)
            If invalid, message contains suggestions for correct format
            
        Examples:
            >>> LiveDataService.validate_symbol_format('AAPL')
            (True, 'Valid stock symbol')
            >>> LiveDataService.validate_symbol_format('EURUSD')
            (False, 'Forex pairs require =X suffix. Try: EURUSD=X')
            >>> LiveDataService.validate_symbol_format('BTC')
            (False, 'Crypto symbols require -USD or -USDT suffix. Try: BTC-USD')
        """
        symbol_upper = symbol.upper().strip()
        
        if not symbol_upper:
            return False, "Symbol cannot be empty"
        
        # Check for common mistakes
        asset_type = LiveDataService.detect_asset_type(symbol_upper)
        
        # Forex without =X suffix
        if asset_type == 'stock' and len(symbol_upper) == 6 and symbol_upper.isalpha():
            # Could be a forex pair missing =X
            if symbol_upper[:3].isalpha() and symbol_upper[3:].isalpha():
                return False, f"Forex pairs require =X suffix. Try: {symbol_upper}=X"
        
        # Crypto without -USD suffix
        if asset_type == 'stock' and len(symbol_upper) <= 5 and symbol_upper.isalpha():
            # Common crypto symbols
            common_cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'DOT', 'MATIC', 'AVAX']
            if symbol_upper in common_cryptos:
                return False, f"Crypto symbols require -USD or -USDT suffix. Try: {symbol_upper}-USD"
        
        # Commodity without =F suffix
        if asset_type == 'stock' and len(symbol_upper) <= 3:
            # Common commodity symbols
            common_commodities = {'GC': 'Gold', 'SI': 'Silver', 'CL': 'Crude Oil', 'NG': 'Natural Gas'}
            if symbol_upper in common_commodities:
                return False, f"Commodity futures require =F suffix. Try: {symbol_upper}=F"
        
        return True, f"Valid {asset_type} symbol"


# Singleton instance
_live_data_service = None

def get_live_data_service() -> LiveDataService:
    """Get singleton instance of LiveDataService"""
    global _live_data_service
    if _live_data_service is None:
        _live_data_service = LiveDataService()
    return _live_data_service

