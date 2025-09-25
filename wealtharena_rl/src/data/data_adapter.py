"""
Data Adapter for WealthArena Trading System

This module provides data integration capabilities for the WealthArena trading system,
including SYS1 API integration, data caching, and real-time data processing.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import hashlib
import redis
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketDataRequest:
    """Market data request specification"""
    symbols: List[str]
    start_date: str
    end_date: str
    interval: str = "1d"  # 1m, 5m, 15m, 1h, 1d
    fields: List[str] = None  # OHLCV, volume, etc.
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = ["open", "high", "low", "close", "volume"]


class SYS1APIClient:
    """
    SYS1 API Client for market data integration
    
    Handles authentication, rate limiting, and data retrieval from the SYS1 API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "https://api.sys1.com")
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.rate_limit = config.get("rate_limit", 100)  # requests per minute
        self.timeout = config.get("timeout", 30)
        
        # Rate limiting
        self.request_times = []
        self.session = None
        
        if not self.api_key:
            logger.warning("No API key provided for SYS1 API")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
        
        # If we're at the rate limit, wait
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.request_times = []
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to SYS1 API"""
        
        await self._rate_limit_check()
        
        if not self.session:
            raise RuntimeError("API client not initialized. Use async context manager.")
        
        # Add authentication
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add timestamp for request signing
        timestamp = str(int(datetime.now().timestamp()))
        headers["X-Timestamp"] = timestamp
        
        # Add request signature
        if self.secret_key:
            signature = self._generate_signature(endpoint, params, timestamp)
            headers["X-Signature"] = signature
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                self.request_times.append(datetime.now())
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    await asyncio.sleep(60)
                    return await self._make_request(endpoint, params)
                else:
                    logger.error(f"API request failed: {response.status} - {await response.text()}")
                    return {}
        
        except asyncio.TimeoutError:
            logger.error("API request timeout")
            return {}
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {}
    
    def _generate_signature(self, endpoint: str, params: Dict[str, Any], timestamp: str) -> str:
        """Generate request signature for authentication"""
        
        # Create signature string
        param_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())]) if params else ""
        signature_string = f"{endpoint}?{param_string}&timestamp={timestamp}"
        
        # Generate HMAC signature
        signature = hashlib.sha256(
            f"{signature_string}{self.secret_key}".encode()
        ).hexdigest()
        
        return signature
    
    async def get_ohlcv_data(self, request: MarketDataRequest) -> pd.DataFrame:
        """Get OHLCV data from SYS1 API"""
        
        endpoint = "/api/market-data/ohlcv"
        params = {
            "symbols": ",".join(request.symbols),
            "start_date": request.start_date,
            "end_date": request.end_date,
            "interval": request.interval,
            "fields": ",".join(request.fields)
        }
        
        data = await self._make_request(endpoint, params)
        
        if not data or "data" not in data:
            logger.error("No data received from SYS1 API")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["data"])
        
        if not df.empty:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            # Ensure numeric columns
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get real-time market data"""
        
        endpoint = "/api/market-data/realtime"
        params = {"symbols": ",".join(symbols)}
        
        data = await self._make_request(endpoint, params)
        
        if not data or "data" not in data:
            return {}
        
        return data["data"]
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status and trading hours"""
        
        endpoint = "/api/market-data/status"
        data = await self._make_request(endpoint)
        
        return data.get("data", {})
    
    async def get_symbols(self, exchange: str = None) -> List[Dict[str, Any]]:
        """Get available trading symbols"""
        
        endpoint = "/api/market-data/symbols"
        params = {"exchange": exchange} if exchange else {}
        
        data = await self._make_request(endpoint, params)
        
        return data.get("data", [])


class DataCache:
    """
    Data cache for market data using Redis
    
    Provides efficient caching of market data to reduce API calls and improve performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour default
        
        # Initialize Redis connection
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        
        try:
            self.redis_client = redis.Redis(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6379),
                db=self.config.get("db", 0),
                password=self.config.get("password"),
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, request: MarketDataRequest) -> str:
        """Generate cache key for market data request"""
        
        key_data = {
            "symbols": sorted(request.symbols),
            "start_date": request.start_date,
            "end_date": request.end_date,
            "interval": request.interval,
            "fields": sorted(request.fields)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"market_data:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def get(self, request: MarketDataRequest) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(request)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data_dict = json.loads(cached_data)
                df = pd.DataFrame(data_dict["data"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                return df
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    def store(self, request: MarketDataRequest, data: pd.DataFrame):
        """Store data in cache"""
        
        if not self.redis_client or data.empty:
            return
        
        try:
            cache_key = self._generate_cache_key(request)
            
            # Convert DataFrame to JSON-serializable format
            data_copy = data.reset_index()
            data_dict = {
                "data": data_copy.to_dict("records"),
                "cached_at": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(data_dict, default=str)
            )
            
            logger.debug(f"Data cached with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")


class DataAdapter:
    """
    Main data adapter for WealthArena trading system
    
    Coordinates data retrieval from various sources, caching, and processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.api_client = SYS1APIClient(config.get("api", {}))
        self.cache = DataCache(config.get("cache", {}))
        self.data_processor = None  # Will be initialized when needed
        
        # Data validation
        self.validate_config = config.get("validate", True)
        self.data_quality_checks = config.get("quality_checks", True)
        
        logger.info("Data adapter initialized")
    
    async def get_ohlcv_data(self, 
                           symbols: List[str], 
                           start_date: str, 
                           end_date: str,
                           interval: str = "1d",
                           use_cache: bool = True) -> pd.DataFrame:
        """Get OHLCV data with caching and validation"""
        
        # Create request
        request = MarketDataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(request)
            if cached_data is not None:
                logger.debug(f"Cache hit for {symbols}")
                return cached_data
        
        # Fetch from API
        logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
        
        async with self.api_client as client:
            data = await client.get_ohlcv_data(request)
        
        if data.empty:
            logger.warning(f"No data received for {symbols}")
            return data
        
        # Validate data
        if self.validate_config:
            data = self._validate_data(data, symbols)
        
        # Quality checks
        if self.data_quality_checks:
            data = self._quality_check_data(data, symbols)
        
        # Cache the data
        if use_cache:
            self.cache.store(request, data)
        
        logger.info(f"Retrieved {len(data)} records for {symbols}")
        return data
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get real-time market data"""
        
        async with self.api_client as client:
            return await client.get_real_time_data(symbols)
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status"""
        
        async with self.api_client as client:
            return await client.get_market_status()
    
    def _validate_data(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Validate market data"""
        
        if data.empty:
            return data
        
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Check for negative prices
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if (data[col] <= 0).any():
                logger.warning(f"Negative or zero prices found in {col}")
                data = data[data[col] > 0]
        
        # Check for invalid OHLC relationships
        invalid_ohlc = (
            (data["high"] < data["low"]) |
            (data["high"] < data["open"]) |
            (data["high"] < data["close"]) |
            (data["low"] > data["open"]) |
            (data["low"] > data["close"])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Invalid OHLC relationships found in {invalid_ohlc.sum()} records")
            data = data[~invalid_ohlc]
        
        # Check for missing values
        if data.isnull().any().any():
            logger.warning("Missing values found in data")
            data = data.dropna()
        
        return data
    
    def _quality_check_data(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Perform data quality checks"""
        
        if data.empty:
            return data
        
        # Check for extreme price movements
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            returns = data[col].pct_change()
            extreme_moves = (returns.abs() > 0.5).any()  # 50% move threshold
            
            if extreme_moves:
                logger.warning(f"Extreme price movements detected in {col}")
        
        # Check for volume anomalies
        if "volume" in data.columns:
            volume_mean = data["volume"].mean()
            volume_std = data["volume"].std()
            volume_threshold = volume_mean + 3 * volume_std
            
            extreme_volume = (data["volume"] > volume_threshold).any()
            if extreme_volume:
                logger.warning("Extreme volume detected")
        
        # Check for data gaps
        if len(data) > 1:
            time_diff = data.index.to_series().diff()
            expected_interval = pd.Timedelta(days=1)  # Assuming daily data
            
            large_gaps = time_diff > expected_interval * 2
            if large_gaps.any():
                logger.warning(f"Large time gaps detected: {large_gaps.sum()} gaps")
        
        return data
    
    def get_available_symbols(self) -> List[Dict[str, Any]]:
        """Get available trading symbols"""
        
        # This would typically be cached or stored locally
        # For now, return a placeholder
        return [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"}
        ]
    
    def close(self):
        """Close data adapter and cleanup resources"""
        
        if self.cache and self.cache.redis_client:
            self.cache.redis_client.close()
        
        logger.info("Data adapter closed")


# Utility functions for data processing
def resample_data(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample data to different time intervals"""
    
    if data.empty:
        return data
    
    # Resample based on interval
    if interval == "1m":
        return data.resample("1T").last()
    elif interval == "5m":
        return data.resample("5T").last()
    elif interval == "15m":
        return data.resample("15T").last()
    elif interval == "1h":
        return data.resample("1H").last()
    elif interval == "1d":
        return data.resample("1D").last()
    else:
        logger.warning(f"Unknown interval: {interval}")
        return data


def fill_missing_data(data: pd.DataFrame, method: str = "forward") -> pd.DataFrame:
    """Fill missing data using specified method"""
    
    if data.empty:
        return data
    
    if method == "forward":
        return data.fillna(method="ffill")
    elif method == "backward":
        return data.fillna(method="bfill")
    elif method == "interpolate":
        return data.interpolate()
    else:
        logger.warning(f"Unknown fill method: {method}")
        return data


if __name__ == "__main__":
    # Test the data adapter
    import asyncio
    
    async def test_data_adapter():
        config = {
            "api": {
                "base_url": "https://api.sys1.com",
                "api_key": "test_key",
                "secret_key": "test_secret"
            },
            "cache": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            }
        }
        
        adapter = DataAdapter(config)
        
        # Test data retrieval
        symbols = ["AAPL", "GOOGL"]
        start_date = "2023-01-01"
        end_date = "2023-01-31"
        
        data = await adapter.get_ohlcv_data(symbols, start_date, end_date)
        print(f"Retrieved data shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        adapter.close()
    
    # Run test
    asyncio.run(test_data_adapter())
